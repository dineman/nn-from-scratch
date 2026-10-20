"""Experiment C3: SGD vs SGD+momentum vs Adam, with a learning-rate sweep.  CifarCNN, 10k CIFAR-10 subset.

    python -m scripts.exp_optim          # ~90 min on CPU (runs in parallel)

Hypotheses:
  H1: each optimizer has a different best learning rate. Adam's is about 100x
      smaller than SGD's, because Adam divides by sqrt(v), so its step size is
      ~lr regardless of the gradient's scale. Comparing optimizers at ONE
      shared lr would therefore be unfair, which is why we sweep.
  H2: momentum with mu = 0.9 behaves like plain SGD with a ~1/(1-mu) = 10x
      larger lr, so its best lr should be ~10x smaller than plain SGD's.
  H3: at their own best lr, momentum and Adam beat plain SGD within a fixed budget.

Setup: same model (no dropout), data, batch 128, budget 15 epochs. Stage 1:
lr in {1e-4 ... 3e-1} (8 values, half-decade steps) x 3 optimizers, seed 0.
Stage 2: each optimizer at the lr with the best FINAL validation accuracy,
seeds 0-2. Selection uses validation data only.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from torchexp.models import CifarCNN
from torchexp.train import fit, load, set_seed

OUT = Path("results/exp_optim")
N_TRAIN, EPOCHS = 10000, 15
LRS = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1)
OPTS = {
    "SGD": lambda p, lr: torch.optim.SGD(p, lr=lr),
    "SGD+momentum": lambda p, lr: torch.optim.SGD(p, lr=lr, momentum=0.9),
    "Adam": lambda p, lr: torch.optim.Adam(p, lr=lr),
}


def _worker_init() -> None:
    torch.set_num_threads(3)


def run(job: tuple[str, float, int]) -> dict:
    opt, lr, seed = job
    data = load("cifar10", n_train=N_TRAIN)
    set_seed(seed)
    h = fit(CifarCNN(), data, lambda p: OPTS[opt](p, lr), EPOCHS, seed)
    return {"opt": opt, "lr": lr, "seed": seed, **asdict(h)}


def final_val(r: dict) -> float:
    """Final validation accuracy; a diverged run counts as chance (10%)."""
    return 0.10 if r["diverged"] else r["val_acc"][-1]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(3, initializer=_worker_init) as ex:
        sweep = list(ex.map(run, [(o, lr, 0) for o in OPTS for lr in LRS]))
        best = {o: max((r for r in sweep if r["opt"] == o), key=final_val)["lr"] for o in OPTS}
        print("best lr per optimizer (by val acc):", best, flush=True)
        extra = list(ex.map(run, [(o, best[o], s) for o in OPTS for s in (1, 2)]))
    seeds = {o: [r for r in sweep + extra if r["opt"] == o and r["lr"] == best[o]] for o in OPTS}
    (OUT / "results.json").write_text(json.dumps({"sweep": sweep, "best_lr": best, "seed_runs": extra}, indent=1))

    colors = dict(zip(OPTS, ("C1", "C2", "C0")))
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for o in OPTS:
        rs = sorted((r for r in sweep if r["opt"] == o), key=lambda r: r["lr"])
        ax[0].semilogx([r["lr"] for r in rs], [100 * final_val(r) for r in rs], "-o", c=colors[o], label=o)
        ax[0].scatter([r["lr"] for r in rs if r["diverged"]], [10] * sum(r["diverged"] for r in rs), marker="x", c="k", zorder=3)
        A = 100 * np.array([r["val_acc"] for r in seeds[o]])
        L = np.array([r["train_loss"] for r in seeds[o]])
        ep = np.arange(1, A.shape[1] + 1)
        lab = f"{o} (lr {best[o]:g})"
        ax[1].plot(ep, L.mean(0), c=colors[o], label=lab)
        ax[1].fill_between(ep, L.min(0), L.max(0), color=colors[o], alpha=0.2)
        ax[2].plot(ep, A.mean(0), c=colors[o], label=lab)
        ax[2].fill_between(ep, A.mean(0) - A.std(0), A.mean(0) + A.std(0), color=colors[o], alpha=0.2)
    ax[0].axhline(10, color="k", ls=":", lw=0.8)
    ax[0].set(xlabel="learning rate", ylabel=f"val accuracy after {EPOCHS} epochs (%)", title="LR sweep (seed 0); x = diverged")
    ax[1].set(xlabel="epoch", ylabel="train loss (eval mode)", yscale="log", title="best lr each: training loss (3 seeds)")
    ax[2].set(xlabel="epoch", ylabel="val accuracy (%)", title="best lr each: validation accuracy (3 seeds)")
    for a in ax:
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "optim.png", dpi=150)

    lines = ["| optimizer | " + " | ".join(f"lr {lr:g}" for lr in LRS) + " |", "|---|" + "---|" * len(LRS)]
    for o in OPTS:
        cells = []
        for lr in LRS:
            r = next(r for r in sweep if r["opt"] == o and r["lr"] == lr)
            v = "div." if r["diverged"] else f"{100 * r['val_acc'][-1]:.1f}"
            cells.append(f"**{v}**" if lr == best[o] else v)
        lines.append(f"| {o} | " + " | ".join(cells) + " |")
    lines += ["", "| optimizer | best lr | val acc, 3 seeds | final train loss |", "|---|---|---|---|"]
    for o in OPTS:
        va = 100 * np.array([r["val_acc"][-1] for r in seeds[o]])
        tl = np.array([r["train_loss"][-1] for r in seeds[o]])
        lines.append(f"| {o} | {best[o]:g} | {va.mean():.1f} ± {va.std(ddof=1):.1f} | {tl.mean():.3f} |")
    (OUT / "table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
