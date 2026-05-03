import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Warning Thresholds (no hard caps — user decides in browser) ---
WARN_NEURONS = 32   # Show a performance warning in Jupyter output
WARN_LAYERS  = 6    # Show a complexity warning in Jupyter output

# --- Activation function registry ---
ACTIVATION_MODULES = {
    nn.ReLU:        'ReLU',
    nn.LeakyReLU:   'LeakyReLU',
    nn.Sigmoid:     'Sigmoid',
    nn.Tanh:        'Tanh',
    nn.GELU:        'GELU',
    nn.ELU:         'ELU',
    nn.SELU:        'SELU',
    nn.SiLU:        'SiLU',
    nn.Mish:        'Mish',
    nn.PReLU:       'PReLU',
    nn.Softmax:     'Softmax',
    nn.Softplus:    'Softplus',
    nn.Hardswish:   'Hardswish',
    nn.Hardsigmoid: 'Hardsigmoid',
}

BATCHNORM_MODULES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm)


class ModelTracker:
    def __init__(self, model: nn.Module):
        self.model = model
        self.layers = []
        self.activations = {}
        self.hooks = []
        self.perf_warnings = []

        self._parse_architecture()
        self._detect_activations()
        self._register_hooks()

    # ------------------------------------------------------------------
    # Architecture parsing
    # ------------------------------------------------------------------

    def _parse_architecture(self):
        """
        Extracts Linear layers and detects associated Dropout / BatchNorm modules
        by scanning the leaf module sequence after each Linear layer.
        """
        # Ordered flat list of all leaf modules (no children)
        leaf_modules = [
            (name, m)
            for name, m in self.model.named_modules()
            if not list(m.children())
        ]
        name_to_leaf_idx = {name: i for i, (name, _) in enumerate(leaf_modules)}

        all_linear = [
            (name, module)
            for name, module in self.model.named_modules()
            if isinstance(module, nn.Linear)
        ]

        # --- Layer count warning ---
        if len(all_linear) >= WARN_LAYERS:
            msg = (
                f"Model has {len(all_linear)} Linear layers. "
                "Consider using fewer layers for smoother browser performance."
            )
            warnings.warn(f"nn_live: {msg}", UserWarning, stacklevel=3)
            self.perf_warnings.append(msg)

        layer_idx = 0
        for name, module in all_linear:
            in_f  = module.in_features
            out_f = module.out_features

            # --- Scan forward to find Dropout / BatchNorm for this layer ---
            dropout_p      = None
            dropout_module = None
            has_batchnorm  = False
            leaf_idx = name_to_leaf_idx.get(name)
            if leaf_idx is not None:
                for j in range(leaf_idx + 1, len(leaf_modules)):
                    _, next_mod = leaf_modules[j]
                    if isinstance(next_mod, nn.Linear):
                        break  # stop at next linear layer
                    if isinstance(next_mod, BATCHNORM_MODULES):
                        has_batchnorm = True
                    if isinstance(next_mod, nn.Dropout) and dropout_module is None:
                        dropout_p      = next_mod.p
                        dropout_module = next_mod

            # --- Neuron count warnings ---
            if layer_idx == 0:
                if in_f > WARN_NEURONS:
                    msg = (
                        f"Input layer has {in_f} neurons. "
                        "The browser will ask you whether to render all neurons or cap the display for performance."
                    )
                    warnings.warn(f"nn_live: {msg}", UserWarning, stacklevel=3)
                    self.perf_warnings.append(msg)

                self.layers.append({
                    "id":   "input",
                    "size": in_f,
                    "type": "input",
                    "dropout_p":      None,
                    "dropout_module": None,
                    "has_batchnorm":  False,
                })

            if out_f > WARN_NEURONS:
                msg = (
                    f"Layer '{name}' has {out_f} neurons. "
                    "The browser will ask you whether to render all neurons or cap the display for performance."
                )
                warnings.warn(f"nn_live: {msg}", UserWarning, stacklevel=3)
                self.perf_warnings.append(msg)

            self.layers.append({
                "id":            f"layer_{layer_idx}",
                "name":          name,
                "size":          out_f,
                "type":          "linear",
                "module":        module,
                "activation":    None,          # filled by _detect_activations()
                "dropout_p":     dropout_p,     # e.g. 0.3, or None
                "dropout_module": dropout_module,
                "has_batchnorm": has_batchnorm,
            })
            layer_idx += 1

    # ------------------------------------------------------------------
    # Activation function detection
    # ------------------------------------------------------------------

    def _detect_activations(self):
        """Detects activation functions following each Linear layer (module + FX)."""
        leaf_modules = [
            (name, m)
            for name, m in self.model.named_modules()
            if not list(m.children())
        ]
        name_to_leaf_idx = {name: i for i, (name, _) in enumerate(leaf_modules)}

        # Strategy 1: module-based
        for layer in self.layers:
            if layer["type"] != "linear":
                continue
            leaf_idx = name_to_leaf_idx.get(layer["name"])
            if leaf_idx is None:
                continue
            next_idx = leaf_idx + 1
            if next_idx < len(leaf_modules):
                _, next_module = leaf_modules[next_idx]
                for act_type, act_label in ACTIVATION_MODULES.items():
                    if isinstance(next_module, act_type):
                        layer["activation"] = act_label
                        break

        # Strategy 2: FX graph tracing (functional activations)
        FUNCTIONAL_ACTIVATIONS = {
            torch.relu:     'ReLU',
            torch.sigmoid:  'Sigmoid',
            torch.tanh:     'Tanh',
            F.relu:         'ReLU',
            F.sigmoid:      'Sigmoid',
            F.tanh:         'Tanh',
            F.gelu:         'GELU',
            F.elu:          'ELU',
            F.selu:         'SELU',
            F.silu:         'SiLU',
            F.leaky_relu:   'LeakyReLU',
            F.softmax:      'Softmax',
            F.mish:         'Mish',
        }
        METHOD_ACTIVATIONS = {'sigmoid': 'Sigmoid', 'relu': 'ReLU', 'tanh': 'Tanh'}

        try:
            traced = torch.fx.symbolic_trace(self.model)
            linear_names = {
                l["name"]: i
                for i, l in enumerate(self.layers)
                if l["type"] == "linear"
            }
            for node in traced.graph.nodes:
                act_label = None
                if node.op == 'call_function' and node.target in FUNCTIONAL_ACTIVATIONS:
                    act_label = FUNCTIONAL_ACTIVATIONS[node.target]
                elif node.op == 'call_method' and node.target in METHOD_ACTIVATIONS:
                    act_label = METHOD_ACTIVATIONS[node.target]
                else:
                    continue
                for arg in node.args:
                    if hasattr(arg, 'target') and arg.target in linear_names:
                        idx = linear_names[arg.target]
                        if self.layers[idx]["activation"] is None:
                            self.layers[idx]["activation"] = act_label
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Hook registration
    # ------------------------------------------------------------------

    def _register_hooks(self):
        """Registers forward hooks to capture activations and dropout masks."""

        # Activation hooks on Linear layers
        for layer_info in self.layers:
            if layer_info["type"] == "linear":
                module   = layer_info["module"]
                layer_id = layer_info["id"]

                def hook_fn(module, input, output, lid=layer_id):
                    if lid == "layer_0" and isinstance(input, tuple):
                        self.activations["input"] = input[0][0].detach().cpu().numpy().tolist()
                    self.activations[lid] = output[0].detach().cpu().numpy().tolist()

                self.hooks.append(module.register_forward_hook(hook_fn))

        # Dropout mask hooks — detect which neurons are dropped each forward pass
        for layer_info in self.layers:
            dp_module = layer_info.get("dropout_module")
            if dp_module is None:
                continue
            layer_id = layer_info["id"]

            def dropout_hook(module, input, output, lid=layer_id):
                if not module.training:
                    # Eval mode: dropout is disabled, clear any stale mask
                    self.activations.pop(f"{lid}_dropped", None)
                    return
                # Compare pre- and post-dropout values:
                # A neuron is "dropped" if it had a non-zero value before but is 0 after.
                inp = input[0][0].detach().cpu()
                out = output[0].detach().cpu()
                was_active = inp.abs() > 1e-6
                now_zero   = out.abs() < 1e-6
                dropped    = (was_active & now_zero).numpy().tolist()
                self.activations[f"{lid}_dropped"] = dropped

            self.hooks.append(dp_module.register_forward_hook(dropout_hook))

    # ------------------------------------------------------------------
    # Data accessors
    # ------------------------------------------------------------------

    def get_architecture_data(self):
        """Returns the full network structure including Dropout and BatchNorm info."""
        return [
            {
                "id":            l["id"],
                "size":          l["size"],
                "type":          l["type"],
                "activation":    l.get("activation"),
                "dropout_p":     l.get("dropout_p"),       # float or null
                "has_batchnorm": l.get("has_batchnorm", False),
            }
            for l in self.layers
        ]

    def get_weights_data(self):
        """Returns the full weight values for all connections."""
        weights_data = {}
        for i in range(1, len(self.layers)):
            source_id = self.layers[i - 1]["id"]
            target_id = self.layers[i]["id"]
            module    = self.layers[i]["module"]
            w = module.weight.detach().cpu().numpy().T
            weights_data[f"{source_id}->{target_id}"] = w.tolist()
        return weights_data

    def get_biases_data(self):
        """Returns the full bias values for all layers."""
        biases_data = {}
        for i in range(1, len(self.layers)):
            target_id = self.layers[i]["id"]
            module    = self.layers[i]["module"]
            if module.bias is not None:
                biases_data[target_id] = module.bias.detach().cpu().numpy().tolist()
        return biases_data

    def get_activations_data(self):
        """Returns latest activations + dropout masks (keys ending in '_dropped')."""
        return self.activations

    def get_perf_warnings(self):
        """Returns performance warnings generated at initialization."""
        return self.perf_warnings

    def cleanup(self):
        for hook in self.hooks:
            hook.remove()
