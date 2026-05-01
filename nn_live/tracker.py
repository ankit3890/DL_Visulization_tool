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


class ModelTracker:
    def __init__(self, model: nn.Module):
        self.model = model
        self.layers = []
        self.activations = {}
        self.hooks = []
        self.perf_warnings = []  # collected for the frontend dialog

        self._parse_architecture()
        self._detect_activations()
        self._register_hooks()

    def _parse_architecture(self):
        """
        Extracts the architecture, focusing on Linear layers.
        Sends soft warnings when layers are large — no hard caps applied here.
        The browser will ask the user whether to render all neurons or cap the display.
        """
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

            # --- Neuron count warnings (input layer) ---
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
                    "type": "input"
                })

            # --- Neuron count warnings (output of this layer) ---
            if out_f > WARN_NEURONS:
                msg = (
                    f"Layer '{name}' has {out_f} neurons. "
                    "The browser will ask you whether to render all neurons or cap the display for performance."
                )
                warnings.warn(f"nn_live: {msg}", UserWarning, stacklevel=3)
                self.perf_warnings.append(msg)

            self.layers.append({
                "id":        f"layer_{layer_idx}",
                "name":      name,
                "size":      out_f,
                "type":      "linear",
                "module":    module,
                "activation": None  # filled in by _detect_activations()
            })
            layer_idx += 1

    def _detect_activations(self):
        """
        Detects activation functions that follow each Linear layer.
        Uses two strategies:
          1. Module-based: inspects leaf modules in definition order (catches nn.ReLU(), etc.)
          2. FX tracing:   traces the forward() graph to find functional calls
                           (catches torch.sigmoid(), torch.relu(), F.relu(), etc.)
        """
        # --- Strategy 1: Module-based detection ---
        leaf_modules = [
            (name, m)
            for name, m in self.model.named_modules()
            if not list(m.children())
        ]

        name_to_leaf_idx = {name: i for i, (name, m) in enumerate(leaf_modules)}

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

        # --- Strategy 2: FX graph tracing (for functional activations) ---
        # Only runs for layers that Strategy 1 didn't find an activation for.
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

        # Method calls on tensors: e.g. z.sigmoid(), x.relu()
        METHOD_ACTIVATIONS = {
            'sigmoid': 'Sigmoid',
            'relu':    'ReLU',
            'tanh':    'Tanh',
        }

        try:
            traced = torch.fx.symbolic_trace(self.model)

            # Map module attribute names to our tracked layer indices
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

                # Walk backwards through args to find which Linear produced the input
                for arg in node.args:
                    if hasattr(arg, 'target') and arg.target in linear_names:
                        idx = linear_names[arg.target]
                        if self.layers[idx]["activation"] is None:
                            self.layers[idx]["activation"] = act_label
        except Exception:
            # FX tracing can fail for models with dynamic control flow,
            # data-dependent shapes, etc. Module-based detection still works.
            pass

    def _register_hooks(self):
        """Registers forward hooks to capture full activations (no truncation)."""
        for layer_info in self.layers:
            if layer_info["type"] == "linear":
                module   = layer_info["module"]
                layer_id = layer_info["id"]

                def hook_fn(module, input, output, lid=layer_id):
                    if lid == "layer_0" and isinstance(input, tuple):
                        self.activations["input"] = input[0][0].detach().cpu().numpy().tolist()
                    self.activations[lid] = output[0].detach().cpu().numpy().tolist()

                hook = module.register_forward_hook(hook_fn)
                self.hooks.append(hook)

    def get_architecture_data(self):
        """Returns the full structure of the network, including detected activation functions."""
        return [
            {
                "id":         l["id"],
                "size":       l["size"],
                "type":       l["type"],
                "activation": l.get("activation")  # e.g. 'ReLU', 'Sigmoid', or None
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
            # PyTorch weight: (out_features, in_features) -> transpose to (in, out)
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
        """Returns the latest captured activations."""
        return self.activations

    def get_perf_warnings(self):
        """Returns performance warnings generated at initialization."""
        return self.perf_warnings

    def cleanup(self):
        for hook in self.hooks:
            hook.remove()
