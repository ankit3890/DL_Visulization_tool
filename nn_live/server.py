import asyncio
import json
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
            # Lazy serialize now that someone is actually connecting
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
        self._last_message = message  # cache before sending
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception:
                pass

manager = ConnectionManager()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

def run_server(port=8000):
    """Run the uvicorn server. This is meant to be run in a thread."""
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="error")

class LiveServer:
    def __init__(self, port=8000):
        self.port = port
        self.thread = threading.Thread(target=run_server, args=(port,), daemon=True)
        self.loop = asyncio.new_event_loop()
        self.loop_thread = threading.Thread(target=self._start_loop, daemon=True)

        # Start threads
        self.thread.start()
        self.loop_thread.start()

    def _start_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def broadcast_data(self, data: dict):
        """Send data to connected clients. Serialization is lazy — only happens when a client is connected."""
        if manager.active_connections:
            # Clients are connected — serialize and send now
            message = json.dumps(data)
            manager._last_message = message
            manager._pending_data = None

            async def send():
                await manager.broadcast(message)

            asyncio.run_coroutine_threadsafe(send(), self.loop)
        else:
            # No clients yet — store raw dict, serialize lazily on connect
            manager._pending_data = data
