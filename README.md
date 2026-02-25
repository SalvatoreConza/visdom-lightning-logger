🧿 # Visdom Logger for PyTorch & PyTorch Lightning

Real-time training visualization for PyTorch Lightning using Visdom, with per-layer gradient norm monitoring via hooks.

## Overview

The project is split into four stages:

1. **Manual training & logging** — Train a simple CNN on MNIST and push loss, accuracy, and learning-rate curves to Visdom by hand (boileplate code).
2. **Lightning Logger interface exploration** — Inspect the `lightning.pytorch.loggers.Logger` base class to understand the methods a custom logger must implement.
3. **Custom `VisdomLogger`** — Implement a full Lightning-compatible logger that automatically plots every metric logged via `self.log()`, then re-train the same CNN through a `LightningModule` / `Trainer` pipeline.
4. **Hook-based gradient monitoring** — Build a `GradientNormLogger` that attaches backward hooks to every trainable parameter and streams per-layer and total gradient L2-norms to Visdom in real time.

Between stages 3 and 4 there is also a **benchmarking section** that measures the overhead introduced by PyTorch hooks (empty hooks, store-only hooks, sampled-norm hooks, and `.item()` vs no-`.item()` variants).

## Dashboard Progression

### Stage 1 — Manual Training Loop (Cell 6)

After running the manual training loop for 10 epochs, Visdom shows loss, accuracy, and learning rate plotted against epochs:

![Manual training metrics](images/1.png)

### Stage 2 — Adding the Lightning Logger (Cell 10)

The custom `VisdomLogger` automatically picks up `train_loss`, `val_acc`, and `epoch` from Lightning's `self.log()` calls, adding three new step-level plots alongside the previous ones:

![Lightning logger metrics](images/2.png)

### Stage 3 — Gradient Norm Monitoring (Cell 19)

The `GradientNormLogger` hooks into every trainable parameter and streams per-layer L2 gradient norms plus a total norm, completing the full dashboard with 15 plots:

![Full dashboard with gradient norms](images/3.png)

## Model

A lightweight CNN used throughout every experiment:

```
Conv2d(1→32, 3×3) → ReLU
Conv2d(32→64, 3×3) → ReLU → MaxPool2d(2)
Flatten → Linear(9216→128) → ReLU
Linear(128→10)
```

Trained on MNIST with Adam (lr = 1e-3) and CrossEntropyLoss.

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- [Visdom](https://github.com/fossasia/visdom)
- PyTorch Lightning
- *(Optional)* pyngrok — only needed if you want a public URL for the Visdom dashboard

Install everything at once:

```bash
pip install torch torchvision visdom lightning pyngrok
```

## Quick Start

```bash
# 1. Launch the Visdom server
python -m visdom.server -port 8097

# 2. Open the dashboard
#    http://localhost:8097

# 3. Run the notebook
jupyter notebook visdom.ipynb
```

## Benchmarking Results

The notebook measures hook overhead on a 20-layer `Linear(512→512)` model (500 forward + backward passes). Key findings:

| Variant | Typical Overhead |
|---------|-----------------|
| Empty hooks (just `pass`) | ~1–3 % |
| Store tensor ref only | ~3–5 % |
| `.norm()` every step + `.item()` | ~150–160 % |
| `.norm()` every step, no `.item()` | Significantly lower |
| `.norm()` every 100 steps | Near baseline |

**Takeaway:** avoid `.item()` inside hooks (it forces a GPU sync) and sample at a reasonable interval instead of every step.
