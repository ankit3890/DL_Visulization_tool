import time
import webbrowser
from .server import LiveServer
from .tracker import ModelTracker

class Visualizer:
    def __init__(self, model, port=8000, open_browser=True):
        """
        Initializes the live visualizer.
        
        Args:
            model: A PyTorch nn.Module instance.
            port: Port for the local web server.
            open_browser: Automatically open the browser to the dashboard.
        """
        self.model = model
        self.tracker = ModelTracker(model)
        self.server = LiveServer(port=port)
        
        # Give the server a moment to start
        time.sleep(1)
        
        self.url = f"http://127.0.0.1:{port}"
        print(f"nn_live: Live Visualizer started at {self.url}")

        # Print any performance warnings collected during architecture parsing
        for w in self.tracker.get_perf_warnings():
            print(f"  [PERF WARNING] {w}")
        
        if open_browser:
            webbrowser.open(self.url)

    def step(self, epoch=None, loss=None, accuracy=None):
        """
        Call this method in your training loop (e.g. after a forward/backward pass)
        to push the latest weights, biases, activations, and stats to the visualizer.
        """
        # Pack data to send
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
                "accuracy": accuracy.item() if hasattr(accuracy, 'item') else accuracy
            }
        }
        
        self.server.broadcast_data(data)

    def cleanup(self):
        """Removes PyTorch hooks."""
        self.tracker.cleanup()
