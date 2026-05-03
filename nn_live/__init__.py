import time
import webbrowser
from .server import LiveServer
from .tracker import ModelTracker


def _is_remote_jupyter():
    """Detect if we're in a REMOTE notebook environment where localhost is not accessible in the browser.
    Returns False for local Jupyter (VS Code, JupyterLab) — those can open a browser tab normally.
    """
    import os
    # Check Colab env vars first — set early in boot, most reliable signal
    if os.environ.get('COLAB_BACKEND_VERSION') or os.environ.get('COLAB_RELEASE_TAG') or os.environ.get('COLAB_GPU'):
        return True
    # Google Colab import check
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        pass
    # Kaggle
    if os.path.exists('/kaggle/working'):
        return True
    return False


class Visualizer:
    def __init__(self, model, port=8000, open_browser=True, optimizer=None):
        """
        Initializes the live visualizer.

        Args:
            model:        A PyTorch nn.Module instance.
            port:         Port for the local web server.
            open_browser: Automatically open the browser / notebook iframe to the dashboard.
            optimizer:    (Optional) PyTorch optimizer. When passed, LR, momentum, weight_decay
                          etc. are automatically shown in the dashboard and update with schedulers.
        """
        self.model     = model
        self.port      = port
        self.optimizer = optimizer
        self.tracker   = ModelTracker(model)
        self.server    = LiveServer(port=port)

        # Give the server a moment to start
        time.sleep(1)

        self._remote = _is_remote_jupyter()

        if self._remote:
            print(f"nn_live: Live Visualizer started (Remote Notebook mode) — port {port}")
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

    def _extract_optimizer_info(self):
        """Extract LR, optimizer type, momentum, betas, weight_decay from optimizer param_groups."""
        if self.optimizer is None:
            return None
        try:
            pg   = self.optimizer.param_groups[0]
            info = {'type': type(self.optimizer).__name__}
            if 'lr'           in pg: info['lr']           = round(pg['lr'], 8)
            if 'momentum'     in pg and pg['momentum']:  info['momentum']     = pg['momentum']
            if 'weight_decay' in pg and pg['weight_decay']: info['weight_decay'] = pg['weight_decay']
            if 'betas'        in pg: info['betas']        = [round(b, 4) for b in pg['betas']]
            if 'eps'          in pg: info['eps']          = pg['eps']
            return info
        except Exception:
            return None

    def _open_dashboard(self):
        """Open the dashboard — inline iframe for remote envs (Colab/Kaggle), browser tab for local."""
        if self._remote:
            # Try Colab-native iframe first
            try:
                from google.colab.output import serve_kernel_port_as_iframe
                serve_kernel_port_as_iframe(self.port, height="700px")
                return
            except Exception:
                pass

            # Generic remote fallback (Kaggle, etc.)
            try:
                from IPython.display import display, IFrame
                display(IFrame(src=f"http://localhost:{self.port}", width="100%", height="700px"))
                return
            except Exception:
                pass

            print(
                f"nn_live: Could not open iframe automatically.\n"
                f"  Open this URL in your browser: http://localhost:{self.port}"
            )
        else:
            # Local environment (VS Code Jupyter, plain Python, JupyterLab, etc.) — open browser tab
            webbrowser.open(f"http://127.0.0.1:{self.port}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self, epoch=None, loss=None, accuracy=None, verbose=False):
        """
        Call this method in your training loop (e.g. after a forward/backward pass)
        to push the latest weights, biases, activations, and stats to the visualizer.

        Args:
            epoch:    Current epoch number (int).
            loss:     Current loss value (torch.Tensor or float).
            accuracy: Current accuracy 0-1 (torch.Tensor or float), optional.
            verbose:  If True, prints epoch/loss/accuracy to the notebook output.

        NOTE: This method will NEVER raise an exception — visualization errors are
        printed as warnings so your training loop always continues.
        """
        import time

        # Throttle check first (cheapest path out)
        now = time.monotonic()
        if hasattr(self, '_last_step_time') and (now - self._last_step_time) < 0.2:
            return
        self._last_step_time = now

        try:
            loss_val = loss.item() if hasattr(loss, 'item') else loss
            acc_val  = accuracy.item() if hasattr(accuracy, 'item') else accuracy

            data = {
                "type": "update",
                "architecture": self.tracker.get_architecture_data(),
                "weights":      self.tracker.get_weights_data(),
                "biases":       self.tracker.get_biases_data(),
                "activations":  self.tracker.get_activations_data(),
                "gradients":    self.tracker.get_gradients_data(),
                "warnings":     self.tracker.get_perf_warnings(),
                "optimizer_info": self._extract_optimizer_info(),
                "stats": {
                    "epoch":    epoch,
                    "loss":     loss_val,
                    "accuracy": acc_val,
                },
            }
            self.server.broadcast_data(data)

            if verbose:
                parts = []
                if epoch    is not None: parts.append(f"Epoch: {epoch}")
                if loss_val is not None: parts.append(f"Loss: {loss_val:.4f}")
                if acc_val  is not None: parts.append(f"Acc: {acc_val*100:.2f}%")
                if parts:
                    print(" | ".join(parts), flush=True)

        except Exception as e:
            # Visualization must never crash training — just warn
            print(f"[nn_live] step() skipped (error: {e})", flush=True)

    def cleanup(self):
        """Removes PyTorch hooks."""
        self.tracker.cleanup()





