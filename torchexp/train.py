"""A minimal, fully seeded PyTorch training loop shared by all Part C experiments.

Data is held in memory as tensors and batched by hand, with a seeded
permutation. No DataLoader is used, so the batch order is a pure function of
the seed.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from nnlib.data import load_cifar10, load_mnist, train_val_split


def set_seed(seed: int) -> None:
    """Seed every RNG that can affect a run (Python, NumPy, torch)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class Data:
    """Train/val/test tensors.  X float32 (N, ...) standardised with TRAIN statistics; y int64 (N,)."""

    X_tr: torch.Tensor
    y_tr: torch.Tensor
    X_va: torch.Tensor
    y_va: torch.Tensor
    X_te: torch.Tensor
    y_te: torch.Tensor


def load(name: str, n_val: int = 5000, n_train: int | None = None) -> Data:
    """Load MNIST or CIFAR-10 as standardised tensors.

    Args:
        name: "mnist" -> X (N, 1, 28, 28); "cifar10" -> X (N, 3, 32, 32).
        n_val: Validation examples held out from the official training set (fixed split, seed 0).
        n_train: If given, keep only the first ``n_train`` of the remaining training examples.
            A smaller training set makes overfitting visible within a CPU-sized budget.

    Normalisation: per-channel mean/std computed on the (possibly subsampled) training split only.
    """
    X, y, X_te, y_te = load_mnist() if name == "mnist" else load_cifar10()
    if X.ndim == 3:  # MNIST: add the channel axis
        X, X_te = X[:, None], X_te[:, None]
    X_tr, y_tr, X_va, y_va = train_val_split(X, y, n_val, seed=0)
    if n_train is not None:
        X_tr, y_tr = X_tr[:n_train], y_tr[:n_train]
    Xf = X_tr.astype(np.float32) / 255.0
    mean = Xf.mean(axis=(0, 2, 3), keepdims=True)
    std = Xf.std(axis=(0, 2, 3), keepdims=True)

    def prep(A: np.ndarray) -> torch.Tensor:
        return torch.from_numpy((A.astype(np.float32) / 255.0 - mean) / std)

    return Data(prep(X_tr), torch.from_numpy(y_tr), prep(X_va), torch.from_numpy(y_va), prep(X_te), torch.from_numpy(y_te))


@dataclass
class History:
    """Per-epoch curves. train_* are measured in EVAL mode after the epoch (no dropout), so they are
    directly comparable to val_*. This matters for the dropout experiment."""

    train_loss: list[float] = field(default_factory=list)
    train_acc: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_acc: list[float] = field(default_factory=list)
    seconds: float = 0.0
    diverged: bool = False


@torch.no_grad()
def evaluate(model: nn.Module, X: torch.Tensor, y: torch.Tensor, device: str, batch: int = 2000) -> tuple[float, float]:
    """Mean cross-entropy and accuracy in eval mode."""
    model.eval()
    loss, correct = 0.0, 0
    for i in range(0, len(X), batch):
        xb, yb = X[i : i + batch].to(device), y[i : i + batch].to(device)
        z = model(xb)
        loss += F.cross_entropy(z, yb, reduction="sum").item()
        correct += (z.argmax(1) == yb).sum().item()
    return loss / len(X), correct / len(X)


def fit(
    model: nn.Module,
    data: Data,
    make_opt: Callable[[list[nn.Parameter]], torch.optim.Optimizer],
    epochs: int,
    seed: int,
    batch_size: int = 128,
    device: str = "cpu",
    train_eval_subset: int = 10000,
) -> History:
    """Train with a fixed budget (``epochs`` passes over the training set) and record curves.

    Args:
        model: Already initialised (the caller controls the init scheme).
        data: From ``load``.
        make_opt: Builds the optimizer from the parameters, e.g. ``lambda p: torch.optim.SGD(p, lr=0.1)``.
        epochs: Budget. All runs within an experiment get the same budget.
        seed: Controls batch order (and dropout masks, through the global torch RNG).
        batch_size: Mini-batch size.
        device: "cpu" or "mps".
        train_eval_subset: Number of training examples used to measure train loss/acc each epoch.

    Returns:
        History. If the loss becomes NaN/inf, training stops early and ``diverged`` is set.
    """
    set_seed(seed)
    model.to(device)
    opt = make_opt(list(model.parameters()))
    g = torch.Generator().manual_seed(seed)
    n = len(data.X_tr)
    sub = slice(0, min(train_eval_subset, n))
    hist = History()
    t0 = time.perf_counter()
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            xb, yb = data.X_tr[idx].to(device), data.y_tr[idx].to(device)
            loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        tl, ta = evaluate(model, data.X_tr[sub], data.y_tr[sub], device)
        vl, va = evaluate(model, data.X_va, data.y_va, device)
        for k, v in zip(("train_loss", "train_acc", "val_loss", "val_acc"), (tl, ta, vl, va)):
            getattr(hist, k).append(v)
        if not np.isfinite(tl):
            hist.diverged = True
            break
    hist.seconds = time.perf_counter() - t0
    return hist


def layer_stats(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> dict[str, list[float]]:
    """One forward/backward pass on a batch, returning per-layer statistics.

    Returns a dict with one entry per Linear/Conv layer, in depth order:
        act_std:   std of that layer's post-activation output (the input to the next layer).
        grad_std:  std of dL/dW for that layer's weight.
        grad_norm: Frobenius norm of dL/dW.
    For the last (output) layer, act_std is the std of the logits.

    A healthy network keeps act_std and grad_std roughly constant across depth.
    Exponential decay in depth means vanishing activations/gradients; exponential
    growth means exploding.
    """
    model.train()
    weighted = [m for m in model.modules() if isinstance(m, (nn.Linear, nn.Conv2d))]
    acts: dict[int, torch.Tensor] = {}
    # Record the output of the activation that FOLLOWS each weighted layer: walk the
    # flat module list and remember the last module after each weighted one.
    flat = [m for m in model.modules() if not isinstance(m, (nn.Sequential, type(model)))]
    owner, hooks = None, []
    after: dict[nn.Module, int] = {}
    for m in flat:
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            owner = weighted.index(m)
            after[m] = owner
        elif owner is not None and isinstance(m, (nn.ReLU, nn.Tanh)):
            after[m] = owner
    for m, i in after.items():
        hooks.append(m.register_forward_hook(lambda mod, inp, out, i=i: acts.__setitem__(i, out.detach())))
    model.zero_grad()
    F.cross_entropy(model(X), y).backward()
    for h in hooks:
        h.remove()
    return {
        "act_std": [acts[i].std().item() for i in range(len(weighted))],
        "grad_std": [m.weight.grad.std().item() for m in weighted],
        "grad_norm": [m.weight.grad.norm().item() for m in weighted],
    }
