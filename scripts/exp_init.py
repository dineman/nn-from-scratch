"""Experiment C1: initialization.  Deep MLP (10 hidden layers x 256) on MNIST.

    python -m scripts.exp_init          # ~15 min on CPU

Hypotheses:
  H1 (scale): with ReLU, Xavier variance 2/(fan_in+fan_out) = 1/256 is half of
     what is needed. Activations should shrink by a factor of sqrt(2) per layer.
     He, with 2/fan_in, should keep them constant. "small" should vanish
     exponentially.
  H2 (symmetry): zeros and constant init make every unit in a layer identical.
     They should stay identical forever (the gradients are identical too), so
     the network learns as if each layer had width 1.
  H3 (training): under a fixed budget and optimizer, the schemes whose signal
     survives depth (He for ReLU; Xavier and He for tanh) train fastest.

Setup: identical architecture, data, optimizer (SGD lr 0.01, momentum 0.9),
batch 128, 3 epochs, seeds 0-2. Only the init scheme and the activation vary.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from torchexp.models import INIT_SCHEMES, MLP, apply_init
from torchexp.train import fit, layer_stats, load, set_seed

OUT = Path("results/exp_init")
DEPTH, WIDTH, EPOCHS, SEEDS, LR = 10, 256, 3, (0, 1, 2), 0.01
FLOOR = 1e-12  # exact zeros can't go on a log axis; they are drawn at this floor


def unit_diversity(model: MLP) -> float:
    """How different the units of the 5th hidden layer are from each other.

    The weight matrix W has shape (256 out, 256 in). Take the std across the
    256 output units of each input column, then average over the columns.
    Returns 0 exactly when every unit has the same incoming weights, i.e.
    symmetry was never broken.
    """
    W = [m for m in model.modules() if isinstance(m, torch.nn.Linear)][4].weight.detach()
    return W.std(dim=0).mean().item()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = load("mnist")
    X, y = data.X_tr[:512], data.y_tr[:512]
    results: dict = {}
    for act in ("relu", "tanh"):
        for scheme in INIT_SCHEMES:
            key = f"{act}/{scheme}"
            set_seed(0)
            stats = layer_stats(apply_init(MLP(784, [WIDTH] * DEPTH, act=act), scheme), X, y)
            runs = []
            for seed in SEEDS:
                set_seed(seed)  # the init draw depends on the seed too
                model = apply_init(MLP(784, [WIDTH] * DEPTH, act=act), scheme)
                h = fit(model, data, lambda p: torch.optim.SGD(p, lr=LR, momentum=0.9), EPOCHS, seed)
                runs.append({"val_acc": h.val_acc, "train_loss": h.train_loss, "diversity": unit_diversity(model)})
            results[key] = {"stats": stats, "runs": runs}
            va = [r["val_acc"][-1] for r in runs]
            print(f"{key:15s} val acc {100 * np.mean(va):5.1f} ± {100 * np.std(va, ddof=1):.1f}   "
                  f"diversity {np.mean([r['diversity'] for r in runs]):.2e}", flush=True)
    (OUT / "results.json").write_text(json.dumps(results, indent=1))
    plot(results)
    table(results)


def plot(results: dict) -> None:
    colors = dict(zip(INIT_SCHEMES, ["#7f7f7f", "#9467bd", "#d62728", "#1f77b4", "#2ca02c"]))
    layers = np.arange(1, DEPTH + 2)

    fig, ax = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for r, act in enumerate(("relu", "tanh")):
        for s in INIT_SCHEMES:
            st = results[f"{act}/{s}"]["stats"]
            ax[r, 0].semilogy(layers, np.maximum(st["act_std"], FLOOR), "-o", ms=3, c=colors[s], label=s)
            ax[r, 1].semilogy(layers, np.maximum(st["grad_norm"], FLOOR), "-o", ms=3, c=colors[s], label=s)
        ax[r, 0].set(ylabel=f"{act}: std of activations", title=f"{act.upper()}: forward signal at init")
        ax[r, 1].set(ylabel=f"{act}: ||dL/dW||", title=f"{act.upper()}: weight-gradient norm at init")
    for a in ax[1]:
        a.set_xlabel(f"layer (1-{DEPTH} hidden, {DEPTH + 1} = output logits)")
    ax[0, 0].legend(fontsize=8)
    fig.text(0.5, 0.005, f"Values that are exactly 0 are drawn at {FLOOR:g}.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(OUT / "stats_at_init.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    ep = np.arange(1, EPOCHS + 1)
    for c, act in enumerate(("relu", "tanh")):
        for s in INIT_SCHEMES:
            A = 100 * np.array([r["val_acc"] for r in results[f"{act}/{s}"]["runs"]])
            ax[c].errorbar(ep, A.mean(0), yerr=A.std(0, ddof=1), fmt="-o", ms=3, capsize=3, c=colors[s], label=s)
        ax[c].set(xlabel="epoch", title=f"{act.upper()}: validation accuracy (mean ± std, 3 seeds)", xticks=ep)
    ax[0].set_ylabel("val accuracy (%)")
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "training.png", dpi=150)
    plt.close(fig)


def table(results: dict) -> None:
    lines = ["| activation | init | act std, layer 1 -> 10 | ‖dL/dW‖, layer 1 -> 10 | val acc after 3 epochs (%) | unit diversity after training |",
             "|---|---|---|---|---|---|"]
    for key, r in results.items():
        act, s = key.split("/")
        st = r["stats"]
        va = 100 * np.array([x["val_acc"][-1] for x in r["runs"]])
        div = np.mean([x["diversity"] for x in r["runs"]])
        lines.append(f"| {act} | {s} | {st['act_std'][0]:.2g} -> {st['act_std'][DEPTH - 1]:.2g} | "
                     f"{st['grad_norm'][0]:.2g} -> {st['grad_norm'][DEPTH - 1]:.2g} | {va.mean():.1f} ± {va.std(ddof=1):.1f} | {div:.2g} |")
    (OUT / "table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
