"""Experiment C1b: does Xavier-on-ReLU's shrinking signal start to hurt in deeper nets?

    python -m scripts.exp_init_depth     # ~15 min on CPU (runs in parallel)

Motivation: in C1, at depth 10, ReLU + Xavier trained almost exactly as well
as ReLU + He (96.2 vs 96.3%), even though the activations at init shrank about
26x through the stack. The variance argument predicts a shrinkage of
sqrt(2)^depth, which is exponential in depth, so the gap should open up as
the network gets deeper.

Hypothesis: as depth grows (10 -> 20 -> 30), Xavier on ReLU trains
increasingly worse than He within the same budget. He stays roughly flat.

Setup: ReLU MLP of width 256, depth in {10, 20, 30}, init in {xavier, he},
seeds 0-2. Everything else is identical to C1: SGD lr 0.01, momentum 0.9,
batch 128, 3 epochs.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from torchexp.models import MLP, apply_init
from torchexp.train import fit, layer_stats, load, set_seed

OUT = Path("results/exp_init")
DEPTHS, SCHEMES, SEEDS, EPOCHS, LR = (10, 20, 30), ("xavier", "he"), (0, 1, 2), 3, 0.01


def _worker_init() -> None:
    torch.set_num_threads(3)


def run(job: tuple[int, str, int]) -> dict:
    depth, scheme, seed = job
    data = load("mnist")
    set_seed(seed)
    model = apply_init(MLP(784, [256] * depth), scheme)
    stats = layer_stats(model, data.X_tr[:512], data.y_tr[:512])
    h = fit(model, data, lambda p: torch.optim.SGD(p, lr=LR, momentum=0.9), EPOCHS, seed)
    return {"depth": depth, "scheme": scheme, "seed": seed, "val_acc": h.val_acc, "diverged": h.diverged,
            "logit_std": stats["act_std"][-1], "grad_norm_layer1": stats["grad_norm"][0]}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(d, s, seed) for d in DEPTHS for s in SCHEMES for seed in SEEDS]
    with ProcessPoolExecutor(3, initializer=_worker_init) as ex:
        runs = list(ex.map(run, jobs))
    (OUT / "depth_results.json").write_text(json.dumps(runs, indent=1))

    lines = ["| depth | init | logit std at init | ‖dL/dW‖ layer 1 at init | val acc after 3 epochs (%) |", "|---|---|---|---|---|"]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for s, c in zip(SCHEMES, ("#1f77b4", "#2ca02c")):
        means, stds = [], []
        for d in DEPTHS:
            rs = [r for r in runs if r["depth"] == d and r["scheme"] == s]
            va = 100 * np.array([r["val_acc"][-1] for r in rs])
            means.append(va.mean())
            stds.append(va.std(ddof=1))
            lines.append(f"| {d} | {s} | {np.mean([r['logit_std'] for r in rs]):.2g} | "
                         f"{np.mean([r['grad_norm_layer1'] for r in rs]):.2g} | {va.mean():.1f} ± {va.std(ddof=1):.1f} |")
        ax.errorbar(DEPTHS, means, yerr=stds, fmt="-o", capsize=4, c=c, label=f"ReLU + {s}")
    ax.set(xlabel="number of hidden layers", ylabel="val accuracy after 3 epochs (%)", xticks=DEPTHS,
           title="Init vs depth (mean ± std, 3 seeds)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "depth.png", dpi=150)
    (OUT / "depth_table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
