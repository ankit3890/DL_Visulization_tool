# nn_live — Real-Time Neural Network Visualizer

<img width="1918" height="887" alt="Screenshot 2026-05-01 205452" src="https://github.com/user-attachments/assets/6becc8c6-fd5c-4997-847f-6730701be9c2" />

<!-- <img width="698" height="658" alt="image" src="https://github.com/user-attachments/assets/a19d37e3-5e53-48ec-b34c-dcc43cb9e227" /> -->

> Test colab link - https://colab.research.google.com/drive/1QdJB_BhDSFmEgEnTbHOLgQyl151ax2DS#scrollTo=ot-QGgSYJ-3Q

> Watch your PyTorch model think. `nn_live` hooks directly into your training loop and streams weights, biases, and activations to a beautiful, interactive browser dashboard — updated live, every epoch.

---

## Installation

### Prerequisites

- **Python** ≥ 3.7
- **PyTorch** ≥ 1.9.0 — install from [pytorch.org](https://pytorch.org/get-started/locally/) for your platform/CUDA version

### Install from PyPI (Recommended)

```bash
pip install nn_live
```

This automatically installs all required dependencies:

| Dependency | Version |
|---|---|
| `torch` | ≥ 1.9.0 |
| `fastapi` | ≥ 0.68.0 |
| `uvicorn` | ≥ 0.15.0 |
| `websockets` | ≥ 10.0 |

### Install from Source (Latest / Development)

```bash
git clone https://github.com/ankit3890/nn_live.git
cd nn_live
pip install -e .
```

### Verify Installation

```python
import nn_live
print(nn_live.__version__)   # e.g. 0.1.0
```

---

## Features

| Feature | Description |
|---|---|
| **Live Nodes** | Neurons glow cyan (positive) or red (negative) with intensity scaled to activation strength |
| **Weight Lines** | Connections colored and sized by weight magnitude. Weak weights auto-fade to reduce clutter |
| **Flow Particles** | Animated signal particles travel left to right along edges; speed scales with weight strength |
| **Bias Pills** | Small badges below each node show the neuron's bias value at a glance |
| **Stats HUD** | Live Epoch, Loss, and Accuracy panel in the top-right corner |
| **Hover Tooltips** | Hover over any neuron to see its exact activation, bias, and layer info |
| **Focus Mode** | Click a neuron to highlight only its connections. Everything else dims to near-black |
| **Controls** | Toggle normalization, particle flow, and adjust animation speed in real-time |
| **Activation Badges** | Auto-detects activation functions (ReLU, Sigmoid, Tanh, etc.) and displays them between layer columns |
| **Safety Limits** | Automatically caps large networks to protect browser performance, with clear warnings |

---

## Quick Start

> **Requirement:** Your model must use `nn.Linear` layers. The visualizer is optimized for Dense/Fully Connected networks.

```python
import torch
import torch.nn as nn
from nn_live import Visualizer

# 1. Define your model
#    Use unique names for each activation (sigmoid1, sigmoid2, etc.)
#    so the visualizer can detect and display all of them.
class Model1(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.linear1  = nn.Linear(num_features, 5)
        self.relu     = nn.ReLU()
        self.linear2  = nn.Linear(5, 3)
        self.sigmoid1 = nn.Sigmoid()
        self.linear3  = nn.Linear(3, 5)
        self.sigmoid2 = nn.Sigmoid()
        self.linear4  = nn.Linear(5, 1)
        self.sigmoid3 = nn.Sigmoid()

    def forward(self, features):
        out = self.linear1(features)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.sigmoid1(out)
        out = self.linear3(out)
        out = self.sigmoid2(out)
        out = self.linear4(out)
        out = self.sigmoid3(out)
        return out

num_features = 30
model = Model1(num_features)

# 2. Attach the visualizer — opens browser automatically
viz = Visualizer(model, port=8000)

# 3. Train and push live updates each epoch
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
loss_fn = nn.BCELoss()

for epoch in range(100):
    inputs  = torch.rand(10, num_features)
    targets = (torch.rand(10, 1) > 0.5).float()

    # Forward pass — must call model(), NOT model.forward()
    outputs = model(inputs)
    loss    = loss_fn(outputs, targets)

    # Accuracy
    predictions = (outputs > 0.5).float()
    accuracy    = (predictions == targets).sum().item() / len(targets)

    # Backward pass
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # Push to the dashboard
    viz.step(epoch=epoch + 1, loss=loss, accuracy=accuracy)
```

---

## Reading the Dashboard

Once the dashboard is open in your browser, here is a complete guide to every visual element.

---

### Nodes (Neurons)

Each **circle** on the canvas represents a single neuron.

| Visual Property | What it means |
|---|---|
| **Number inside the node** | The **Activation** (`Act`) — the value this neuron outputs to the next layer |
| **Cyan color** | Activation is **positive** |
| **Red color** | Activation is **negative** |
| **Glow brightness** | How large the activation is — brighter glow means the neuron is firing strongly |
| **Scale on hover** | Node smoothly enlarges 1.25x when hovered to reveal details clearly |

#### What is "Activation" (Act)?

The **Activation** is the output value of a neuron — what it passes forward to the next layer:

```
Activation = f( sum(weight * input) + bias )
```

Where `f` is your activation function (ReLU, Sigmoid, Tanh, etc.).

**Examples from the dashboard:**

| Node appearance | What it means |
|---|---|
| Bright cyan, `0.85` | Strongly activated — large positive signal flowing forward |
| Dark, near `0.00` | Barely contributing — neuron is essentially silent |
| Bright red, `-0.72` | Strongly negative — actively suppressing the next layer |

---

### Bias Pill (`b: -0.04`)

The small rounded badge below each node shows the neuron's **Bias** value.

Bias shifts the activation threshold independently of the inputs. A neuron with a large negative bias is much harder to activate, even with strong inputs. As training progresses, watch these values shift as the model learns.

---

### Connection Lines (Weights)

The lines between nodes represent the **weights** of the network.

| Visual Property | What it means |
|---|---|
| **Cyan line** | Positive weight — amplifies the signal |
| **Red line** | Negative weight — suppresses the signal |
| **Line thickness** | Proportional to weight magnitude — thick = strong, thin = weak |
| **Near-invisible line** | Weight magnitude < 0.1 — dimmed heavily to remove visual noise |
| **Moving particles** | Signal flow direction (left to right). Particle speed = weight strength |

---

### Stats Panel (Top Right)

| Field | What it means |
|---|---|
| **Epoch** | Current training epoch, passed via `viz.step(epoch=...)` |
| **Loss** | Training loss — watch this decrease as your model learns |
| **Accuracy** | Classification accuracy (0.0 to 1.0), passed via `viz.step(accuracy=...)` |

> All three fields are **optional**. Pass only what you have — unused fields simply display `-`.

---

### Hover Tooltip

Hovering over any neuron shows a detailed tooltip:

- **Layer name** and **Neuron index**
- `Act:` — Activation value with a visual magnitude bar
- `Bias:` — Bias value with a visual magnitude bar
- A hint to click for Focus Mode

---

### Focus Mode (Click a Node)

Click any neuron to **lock focus** on it.

- The entire network dims to near-black
- Only the **incoming and outgoing connections** for that neuron remain highlighted
- Click the same neuron again, or click empty canvas space, to unlock

This is useful for understanding the exact role of a single neuron and tracing which neurons influence it.

---

### Activation Function Badges

`nn_live` automatically detects the activation function applied after each layer and renders a **pill-shaped badge** (e.g. `ReLU`, `Sigmoid`) between the corresponding layer columns on the canvas.

**Two detection strategies are used:**

1. **Module-based** — Detects activations defined as `nn.Module` attributes in `__init__`:
   ```python
   self.relu = nn.ReLU()       # detected
   self.sigmoid = nn.Sigmoid() # detected
   ```

2. **Graph tracing** — Uses `torch.fx` to trace the `forward()` method and detect functional calls:
   ```python
   x = torch.relu(x)           # detected
   x = torch.sigmoid(x)        # detected
   x = F.relu(x)               # detected
   x = x.sigmoid()             # detected
   ```

**Important: use unique attribute names.** If you reuse the same attribute name for multiple activations, PyTorch only registers the last one:

```python
# Wrong — PyTorch overwrites self.sigmoid each time, only the last survives
self.sigmoid = nn.Sigmoid()
self.sigmoid = nn.Sigmoid()

# Correct — each activation has a unique name
self.sigmoid1 = nn.Sigmoid()
self.sigmoid2 = nn.Sigmoid()
```

If no activation is detected for a layer gap (e.g. the model uses an unsupported activation or complex control flow), no badge is drawn — no visual clutter.

---

## Architecture Examples

`nn_live` tracks all `nn.Linear` layers. For advanced architectures (CNNs, RNNs), it skips spatial and recurrent layers and visualizes only the final Dense classification layers — preventing browser overload from massive feature maps.

---

### 1. The Perceptron (Single Neuron)

```python
class Perceptron(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(3, 1)  # 3 inputs -> 1 output

    def forward(self, x):
        return torch.sigmoid(self.layer(x))

viz = Visualizer(Perceptron())
```

---

### 2. Standard ANN (Multi-Layer Perceptron)

```python
class ANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 8)
        self.fc2 = nn.Linear(8, 6)
        self.fc3 = nn.Linear(6, 4)
        self.out = nn.Linear(4, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return torch.sigmoid(self.out(x))

viz = Visualizer(ANN())
```

---

### 3. CNN (Convolutional Neural Network)

`nn_live` ignores `nn.Conv2d` layers and visualizes only the final `nn.Linear` classification head.

```python
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1   = nn.Conv2d(1, 4, kernel_size=3)  # ignored by visualizer
        self.flatten = nn.Flatten()
        self.fc1     = nn.Linear(16, 5)                # visualized
        self.fc2     = nn.Linear(5, 2)                 # visualized

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

model = SimpleCNN()
viz   = Visualizer(model)

dummy_image = torch.rand(1, 1, 6, 6)
output = model(dummy_image)
viz.step()
```

---

### 4. RNN (Recurrent Neural Network)

`nn_live` ignores `nn.RNN` / `nn.LSTM` layers and visualizes only the final `nn.Linear` readout layers.

```python
class SimpleRNN(nn.Module):
    def __init__(self, input_size=10, hidden_size=8):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)  # ignored
        self.fc1 = nn.Linear(hidden_size, 4)                          # visualized
        self.fc2 = nn.Linear(4, 1)                                    # visualized

    def forward(self, x):
        out, _ = self.rnn(x)
        x = torch.relu(self.fc1(out[:, -1, :]))
        return torch.sigmoid(self.fc2(x))

model = SimpleRNN()
viz   = Visualizer(model)

dummy_sequence = torch.rand(1, 5, 10)
output = model(dummy_sequence)
viz.step()
```

---

## Safety Limits

`nn_live` automatically detects when your model contains large layers and gives you full control over how they are rendered in the browser.

### Warning Thresholds

| Threshold | Behavior |
|---|---|
| Any layer > 32 neurons | A performance warning is printed in Jupyter output |
| Any layer > 32 neurons | A modal dialog appears in the browser the first time data arrives |

### How it works

**In Jupyter**, a warning is printed at startup:
```
nn_live: Live Visualizer started at http://127.0.0.1:8000
  [PERF WARNING] Input layer has 100 neurons. The browser will ask you whether
  to render all neurons or cap the display for performance.
```

**In the browser**, a dialog appears the first time the large network is detected:

> **Large Network Detected**
> One or more layers in your model are large. Rendering all neurons may slow down or crash your browser.
>
> **What would you like to do?**
> `[Show All Neurons]`   `[Cap at 64 (Recommended)]`

- Choosing **Show All Neurons** renders every neuron in the layer. Node sizes dynamically shrink so all neurons fit within the visible canvas height.
- Choosing **Cap at 64** limits the display to 64 neurons per layer. The layer header will show `64 / 100 Neurons (capped)` so it is always clear that the view is partial.

The dialog appears **only once per browser session** and does not affect your Python training loop in any way. Full weight, bias, and activation data is always sent from the backend — the cap is purely a rendering decision.

### Why large layers are a concern

Rendering connections in the browser is GPU/CPU intensive. A `Linear(500, 500)` layer creates **250,000 curved lines and animated particles** simultaneously, which can crash the browser tab.

---

## Troubleshooting

> Always restart the Jupyter Kernel when you re-run `viz = Visualizer(...)`.

| Problem | Cause | Fix |
|---|---|---|
| `[Errno 10048]` Address in use | `Visualizer()` was called twice on the same port | Restart kernel, or use `Visualizer(model, port=8001)` |
| Values frozen at `0.00` | Used `model.forward(inputs)` instead of `model(inputs)` | Always call `outputs = model(inputs)` — this triggers PyTorch hooks |
| Browser lag / FPS drops | Model has very large layers | Choose "Cap at 64" when the browser dialog appears |
| Accuracy not showing | Accuracy not passed to `viz.step()` | Add `accuracy=acc_value` to your `viz.step()` call |
| Dialog appeared but neurons still capped | Old `tracker.py` cached in Jupyter memory | Restart the Jupyter Kernel so the new code is loaded |

---

## Author

Built by **Ankit Kumar Singh**  
GitHub: [ankit3890](https://github.com/ankit3890)
