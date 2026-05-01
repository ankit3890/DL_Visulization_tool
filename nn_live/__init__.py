import time
import webbrowser
from .server import LiveServer
from .tracker import ModelTracker


def _is_jupyter():
    """Detect if we're running inside any Jupyter/notebook environment (Colab, VS Code, modelcode, Kaggle, etc.)."""
    try:
        from IPython import get_ipython
        shell = get_ipython()
        return shell is not None
    except Exception:
        return False


class Visualizer:
    def __init__(self, model, port=8000, open_browser=True):
        """
        Initializes the live visualizer.

        Args:
            model:        A PyTorch nn.Module instance.
            port:         Port for the local web server.
            open_browser: Automatically open the browser / notebook iframe to the dashboard.
        """
        self.model = model
        self.port = port
        self.tracker = ModelTracker(model)
        self.server = LiveServer(port=port)

        # Give the server a moment to start
        time.sleep(1)

        self._jupyter = _is_jupyter()

        if self._jupyter:
            print(f"nn_live: Live Visualizer started (Notebook mode) — port {port}")
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
        """Open the dashboard inline (Jupyter) or in a browser tab (local)."""
        if self._jupyter:
            # Try Colab-native iframe first, fall back to IPython IFrame
            try:
                from google.colab.output import serve_kernel_port_as_iframe
                serve_kernel_port_as_iframe(self.port, height="700px")
                return
            except Exception:
                pass

            # Generic Jupyter fallback — works in VS Code, modelcode, Kaggle, etc.
            try:
                from IPython.display import display, IFrame
                display(IFrame(src=f"http://localhost:{self.port}", width="100%", height="700px"))
                return
            except Exception:
                pass

            # Last resort — just print the URL
            print(
                f"nn_live: Could not open iframe automatically.\n"
                f"  Open this URL in your browser: http://localhost:{self.port}"
            )
        else:
            webbrowser.open(f"http://127.0.0.1:{self.port}")

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


