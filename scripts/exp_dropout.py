"""Experiment C2: dropout on vs off.  CifarCNN on a 10k-image CIFAR-10 training subset.

    python -m scripts.exp_dropout        # ~45 min on CPU (runs in parallel)

Hypothesis: without dropout, the network memorises the 10k training images
(train accuracy -> ~100%) while validation accuracy plateaus and validation
LOSS rises: a large overfitting gap. Dropout (p = 0.5 on the FC layers, 0.25
after the conv blocks) should shrink the gap and give equal or better
validation accuracy. It should also slow down training-set fitting, since it
is a regulariser.

Setup: same architecture, data subset, optimizer (Adam lr 1e-3), batch 128,
40 epochs, seeds 0-2. Only the dropout probability varies. No data
augmentation (augmentation is also a regulariser and would blur the effect).
Train metrics are measured in eval mode on 10k training images, so they are
directly comparable to the validation metrics.
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

OUT = Path("results/exp_dropout")
N_TRAIN, EPOCHS, SEEDS, PS = 10000, 40, (0, 1, 2), (0.0, 0.5)


def _worker_init() -> None:
    # 3 parallel workers x 3 threads ~ fills a 10-core laptop without oversubscribing it.
    torch.set_num_threads(3)


def run(job: tuple[float, int]) -> dict:
    p, seed = job
    data = load("cifar10", n_train=N_TRAIN)
    set_seed(seed)
    h = fit(CifarCNN(dropout=p), data, lambda ps: torch.optim.Adam(ps, lr=1e-3), EPOCHS, seed)
    return {"p": p, "seed": seed, **asdict(h)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(p, s) for p in PS for s in SEEDS]
    with ProcessPoolExecutor(3, initializer=_worker_init) as ex:
        runs = list(ex.map(run, jobs))
    (OUT / "results.json").write_text(json.dumps(runs, indent=1))

    ep = np.arange(1, EPOCHS + 1)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    lines = ["| dropout p | final train acc | final val acc | best val acc (epoch) | gap train−val at end | min val loss (epoch) | final val loss |",
             "|---|---|---|---|---|---|---|"]
    for p, c in zip(PS, ("C3", "C0")):
        rs = [r for r in runs if r["p"] == p]
        tr = 100 * np.array([r["train_acc"] for r in rs])
        va = 100 * np.array([r["val_acc"] for r in rs])
        vl = np.array([r["val_loss"] for r in rs])
        lab = f"dropout {p}" if p else "no dropout"
        for A, ls, what in [(tr, "--", "train"), (va, "-", "val")]:
            ax[0].plot(ep, A.mean(0), ls, c=c, label=f"{lab}: {what}")
            ax[0].fill_between(ep, A.mean(0) - A.std(0), A.mean(0) + A.std(0), color=c, alpha=0.15)
        ax[1].plot(ep, vl.mean(0), c=c, label=f"{lab}: val")
        ax[1].plot(ep, np.array([r["train_loss"] for r in rs]).mean(0), "--", c=c, label=f"{lab}: train")
        lines.append(
            f"| {p} | {tr[:, -1].mean():.1f} ± {tr[:, -1].std(ddof=1):.1f} | {va[:, -1].mean():.1f} ± {va[:, -1].std(ddof=1):.1f} | "
            f"{va.max(1).mean():.1f} ({np.mean(va.argmax(1) + 1):.0f}) | {(tr[:, -1] - va[:, -1]).mean():.1f} | "
            f"{vl.min(1).mean():.2f} ({np.mean(vl.argmin(1) + 1):.0f}) | {vl[:, -1].mean():.2f} |"
        )
    ax[0].set(xlabel="epoch", ylabel="accuracy (%)", title=f"CIFAR-10, {N_TRAIN // 1000}k training images: accuracy")
    ax[1].set(xlabel="epoch", ylabel="cross-entropy", yscale="log", title="loss (mean of 3 seeds)")
    for a in ax:
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "dropout.png", dpi=150)
    (OUT / "table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
