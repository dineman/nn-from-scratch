"""Parts A and B: train the NumPy MLP or LeNet on MNIST with mini-batch SGD.

    python -m scripts.train_numpy --model mlp     # Part A, ~1 min on a laptop CPU
    python -m scripts.train_numpy --model lenet   # Part B, slow by design (pure NumPy convs)

Protocol: 55k train / 5k validation (fixed split), test set evaluated ONCE at
the end. Normalisation statistics come from the training split only.
Outputs go to results/numpy_<model>/: metrics.json, curves.png.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from nnlib.data import load_mnist, standardize, train_val_split
from nnlib.layers import Conv2D, Flatten, Linear, MaxPool2D, ReLU, Sequential
from nnlib.losses import softmax_cross_entropy
from nnlib.optim import SGD


@dataclass
class Config:
    model: str = "mlp"
    seed: int = 0
    epochs: int = 15
    batch_size: int = 64
    lr: float = 0.1
    momentum: float = 0.0  # Part A asks for plain mini-batch SGD
    n_val: int = 5000


def build_model(name: str, rng: np.random.Generator) -> Sequential:
    """Construct the network.

    mlp:   784 -> 256 -> 128 -> 10, ReLU, He init. Input is the flattened image (N, 784).
    lenet: LeNet-5 layout (LeCun et al. 1998), modernised with ReLU and max-pool:
           (N,1,28,28) conv5x5 pad2 -> (N,6,28,28) pool -> (N,6,14,14)
           conv5x5 -> (N,16,10,10) pool -> (N,16,5,5) flatten -> 400 -> 120 -> 84 -> 10.
           The 1998 original used tanh-like units and learned average-subsampling.
    """
    if name == "mlp":
        return Sequential(Linear(784, 256, rng), ReLU(), Linear(256, 128, rng), ReLU(), Linear(128, 10, rng))
    if name == "lenet":
        return Sequential(
            Conv2D(1, 6, 5, rng, pad=2), ReLU(), MaxPool2D(2),
            Conv2D(6, 16, 5, rng), ReLU(), MaxPool2D(2), Flatten(),
            Linear(400, 120, rng), ReLU(), Linear(120, 84, rng), ReLU(), Linear(84, 10, rng),
        )
    raise ValueError(name)


def evaluate(model: Sequential, X: np.ndarray, y: np.ndarray, batch: int = 1000) -> tuple[float, float]:
    """Mean loss and accuracy over a dataset, in eval mode (no dropout), in chunks to bound memory."""
    loss_sum, correct = 0.0, 0
    for i in range(0, len(X), batch):
        z = model.forward(X[i : i + batch], train=False)
        loss_sum += softmax_cross_entropy(z, y[i : i + batch])[0] * len(z)
        correct += int((z.argmax(1) == y[i : i + batch]).sum())
    return loss_sum / len(X), correct / len(X)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["mlp", "lenet"], default="mlp")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = Config(model=args.model, seed=args.seed)
    if args.epochs:
        cfg.epochs = args.epochs
    out_dir = Path(f"results/numpy_{cfg.model}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- data ----
    X_all, y_all, X_te, y_te = load_mnist()
    X_tr, y_tr, X_va, y_va = train_val_split(X_all, y_all, cfg.n_val, seed=0)
    mean, std = X_tr.mean() / 255.0, X_tr.std() / 255.0  # training split only
    shape = (-1, 784) if cfg.model == "mlp" else (-1, 1, 28, 28)
    X_tr, X_va, X_te = (standardize(X, mean, std).reshape(shape) for X in (X_tr, X_va, X_te))

    # ---- model ----
    rng = np.random.default_rng(cfg.seed)
    model = build_model(cfg.model, rng)
    opt = SGD(model, lr=cfg.lr, momentum=cfg.momentum)
    print(f"{cfg.model}: {model.num_params():,} parameters; initial val loss {evaluate(model, X_va, y_va)[0]:.3f} (log 10 = 2.303)")

    hist = {k: [] for k in ("train_loss", "train_acc", "val_loss", "val_acc", "epoch_seconds")}
    for epoch in range(cfg.epochs):
        t0 = time.perf_counter()
        perm = rng.permutation(len(X_tr))  # reshuffle every epoch so batches differ
        loss_sum, correct = 0.0, 0
        for i in range(0, len(perm), cfg.batch_size):
            idx = perm[i : i + cfg.batch_size]
            z = model.forward(X_tr[idx], train=True)
            loss, dz = softmax_cross_entropy(z, y_tr[idx])
            # No zero_grad needed here, unlike Project 1: every backward() OVERWRITES
            # layer.grads (=, not +=), because a chain of layers has no reused nodes.
            model.backward(dz)
            opt.step()
            loss_sum += loss * len(idx)
            correct += int((z.argmax(1) == y_tr[idx]).sum())
        # The training metrics are running averages over the epoch, while the weights
        # were still changing, so they lag slightly behind the end-of-epoch val metrics.
        hist["train_loss"].append(loss_sum / len(X_tr))
        hist["train_acc"].append(correct / len(X_tr))
        vl, va = evaluate(model, X_va, y_va)
        hist["val_loss"].append(vl)
        hist["val_acc"].append(va)
        hist["epoch_seconds"].append(time.perf_counter() - t0)
        print(f"epoch {epoch + 1:2d}  train loss {hist['train_loss'][-1]:.4f} acc {hist['train_acc'][-1]:.4f}  "
              f"val loss {vl:.4f} acc {va:.4f}  ({hist['epoch_seconds'][-1]:.1f}s)", flush=True)

    test_loss, test_acc = evaluate(model, X_te, y_te)  # touched exactly once
    print(f"TEST  loss {test_loss:.4f}  acc {test_acc:.4f}")
    metrics = {"test_loss": test_loss, "test_acc": test_acc, "n_params": model.num_params(),
               "final_val_acc": hist["val_acc"][-1], "best_val_acc": max(hist["val_acc"]),
               "best_val_epoch": int(np.argmax(hist["val_acc"])) + 1}
    (out_dir / "metrics.json").write_text(json.dumps({"config": asdict(cfg), "metrics": metrics, "history": hist}, indent=1))

    ep = np.arange(1, cfg.epochs + 1)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].plot(ep, hist["train_loss"], "-o", ms=3, label="train (running avg)")
    ax[0].plot(ep, hist["val_loss"], "-o", ms=3, label="validation")
    ax[0].set(xlabel="epoch", ylabel="cross-entropy", yscale="log", title=f"NumPy {cfg.model}: loss")
    ax[1].plot(ep, 100 * np.array(hist["train_acc"]), "-o", ms=3, label="train (running avg)")
    ax[1].plot(ep, 100 * np.array(hist["val_acc"]), "-o", ms=3, label="validation")
    ax[1].axhline(97, color="k", ls=":", lw=0.8, label="97% target")
    ax[1].set(xlabel="epoch", ylabel="accuracy (%)", ylim=(90, 100.2), title=f"accuracy (test: {100 * test_acc:.2f}%)")
    for a in ax:
        a.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "curves.png", dpi=150)


if __name__ == "__main__":
    main()
