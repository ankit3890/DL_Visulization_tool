import asyncio
import json
import sys
import threading
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

app = FastAPI()

# Mount static files
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
async def root():
    return FileResponse(static_dir / "index.html")

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []
        self._last_message: str | None = None   # serialized cache
        self._pending_data: dict | None = None  # raw data for lazy serialization

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        # Replay last known state to late-joining clients
        msg = self._last_message
        if msg is None and self._pending_data is not None:
            msg = json.dumps(self._pending_data)
            self._last_message = msg
            self._pending_data = None
        if msg is not None:
            try:
                await websocket.send_text(msg)
            except Exception:
                pass

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        self._last_message = message
        dead = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception:
                dead.append(connection)
        # Clean up dead connections
        for c in dead:
            self.disconnect(c)

manager = ConnectionManager()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

def run_server(port=8000):
    """Run the uvicorn server in a background thread."""
    # Use SelectorEventLoop policy for uvicorn on Windows to avoid
    # ProactorEventLoop assertion errors with concurrent writes
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="error")

class LiveServer:
    def __init__(self, port=8000):
        self.port = port
        self._sending = False  # guard against concurrent broadcasts

        # Broadcast loop — use SelectorEventLoop on Windows
        if sys.platform == "win32":
            self.loop = asyncio.SelectorEventLoop()
        else:
            self.loop = asyncio.new_event_loop()

        self.loop_thread = threading.Thread(target=self._start_loop, daemon=True)
        self.server_thread = threading.Thread(target=run_server, args=(port,), daemon=True)

        self.loop_thread.start()
        self.server_thread.start()

    def _start_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def broadcast_data(self, data: dict):
        """Send data to connected clients. Lazy serialization + concurrent-write guard."""
        if self._sending:
            # Previous send still in flight — skip to avoid ProactorEventLoop assertion
            return

        if manager.active_connections:
            message = json.dumps(data)
            manager._last_message = message
            manager._pending_data = None
            self._sending = True

            async def send():
                try:
                    await manager.broadcast(message)
                finally:
                    self._sending = False

            asyncio.run_coroutine_threadsafe(send(), self.loop)
        else:
            manager._pending_data = data
