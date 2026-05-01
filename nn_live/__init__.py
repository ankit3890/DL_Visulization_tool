import time
import webbrowser
from .server import LiveServer
from .tracker import ModelTracker


def _is_colab():
    """Detect if we're running inside Google Colab."""
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


class Visualizer:
    def __init__(self, model, port=8000, open_browser=True):
        """
        Initializes the live visualizer.

        Args:
            model:        A PyTorch nn.Module instance.
            port:         Port for the local web server.
            open_browser: Automatically open the browser / Colab iframe to the dashboard.
        """
        self.model = model
        self.port = port
        self.tracker = ModelTracker(model)
        self.server = LiveServer(port=port)

        # Give the server a moment to start
        time.sleep(1)

        self._colab = _is_colab()

        if self._colab:
            self.url = f"https://localhost:{port}"
            print(f"nn_live: Live Visualizer started (Google Colab mode) — port {port}")
        else:
            self.url = f"http://127.0.0.1:{port}"
            print(f"nn_live: Live Visualizer started at {self.url}")

        # Print any performance warnings collected during architecture parsing
        for w in self.tracker.get_perf_warnings():
            print(f"  [PERF WARNING] {w}")

        if open_browser:
            self._open_dashboard()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _open_dashboard(self):
        """Open the dashboard — uses Colab port-forwarding when in Colab."""
        if self._colab:
            try:
                from google.colab.output import serve_kernel_port_as_iframe
                serve_kernel_port_as_iframe(self.port, height="700px")
            except Exception as e:
                print(
                    f"nn_live: Could not open Colab iframe automatically ({e}).\n"
                    f"  Run this cell manually to view the dashboard:\n\n"
                    f"    from google.colab.output import serve_kernel_port_as_iframe\n"
                    f"    serve_kernel_port_as_iframe({self.port}, height='700px')\n"
                )
        else:
            webbrowser.open(self.url)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self, epoch=None, loss=None, accuracy=None):
        """
        Call this method in your training loop (e.g. after a forward/backward pass)
        to push the latest weights, biases, activations, and stats to the visualizer.
        """
        data = {
            "type": "update",
            "architecture": self.tracker.get_architecture_data(),
            "weights":      self.tracker.get_weights_data(),
            "biases":       self.tracker.get_biases_data(),
            "activations":  self.tracker.get_activations_data(),
            "warnings":     self.tracker.get_perf_warnings(),
            "stats": {
                "epoch":    epoch,
                "loss":     loss.item() if hasattr(loss, 'item') else loss,
                "accuracy": accuracy.item() if hasattr(accuracy, 'item') else accuracy,
            },
        }
        self.server.broadcast_data(data)

    def cleanup(self):
        """Removes PyTorch hooks."""
        self.tracker.cleanup()

