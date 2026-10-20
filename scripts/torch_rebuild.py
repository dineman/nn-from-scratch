"""Part C (rebuild): the Part A/B models in PyTorch, plus a CIFAR-10 reference run.

    python -m scripts.torch_rebuild      # ~10 min on CPU

1. MNIST MLP 784-256-128-10, He init, plain SGD lr 0.1, batch 64, 15 epochs: same recipe as the NumPy run.
2. MNIST LeNet, same recipe.
3. CIFAR-10 CifarCNN on the full 45k training split, Adam 1e-3, dropout 0.5, 15 epochs: a reference
   point for the CIFAR experiments, which use a 10k subset to make overfitting visible.

Test accuracy is reported for these final models only.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from torchexp.models import MLP, CifarCNN, LeNet, apply_init
from torchexp.train import evaluate, fit, load, set_seed

OUT = Path("results/torch_rebuild")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    out = {}
    mnist = load("mnist")
    for name, build in [("mlp", lambda: MLP(784, [256, 128])), ("lenet", LeNet)]:
        set_seed(0)
        model = apply_init(build(), "he")
        h = fit(model, mnist, lambda p: torch.optim.SGD(p, lr=0.1), epochs=15, seed=0, batch_size=64)
        tl, ta = evaluate(model, mnist.X_te, mnist.y_te, "cpu")
        out[f"mnist_{name}"] = {"test_acc": ta, "test_loss": tl, "val_acc": h.val_acc, "train_acc": h.train_acc,
                                "sec_per_epoch": h.seconds / 15}
        print(f"mnist {name}: test acc {ta:.4f}  ({h.seconds / 15:.1f} s/epoch)", flush=True)

    cifar = load("cifar10")
    set_seed(0)
    model = CifarCNN(dropout=0.5)  # PyTorch default init (Kaiming-uniform variant) for the CNN
    h = fit(model, cifar, lambda p: torch.optim.Adam(p, lr=1e-3), epochs=15, seed=0)
    tl, ta = evaluate(model, cifar.X_te, cifar.y_te, "cpu")
    out["cifar_cnn_full"] = {"test_acc": ta, "test_loss": tl, "val_acc": h.val_acc, "train_acc": h.train_acc,
                             "sec_per_epoch": h.seconds / 15}
    print(f"cifar10 cnn (45k train): test acc {ta:.4f}  ({h.seconds / 15:.1f} s/epoch)")
    (OUT / "results.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
