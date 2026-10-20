"""PyTorch models for Part C, plus the initialization schemes under study."""

from __future__ import annotations

import math

import torch
from torch import nn

ACTS = {"relu": nn.ReLU, "tanh": nn.Tanh}


class MLP(nn.Module):
    """Plain MLP: in -> hidden[0] -> ... -> hidden[-1] -> n_out, with ``act`` after every hidden layer.

    Args:
        n_in: Input features (784 for MNIST).
        hidden: Hidden widths, e.g. [256, 128] (Part A's net) or [256] * 10 (deep, for the init study).
        n_out: Number of classes.
        act: "relu" or "tanh".
        dropout: Drop probability applied after every hidden activation (0 = off).

    Shapes: x (N, n_in) or (N, 1, 28, 28) -> logits (N, n_out).
    """

    def __init__(self, n_in: int, hidden: list[int], n_out: int = 10, act: str = "relu", dropout: float = 0.0) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Flatten()]
        sizes = [n_in, *hidden]
        for a, b in zip(sizes[:-1], sizes[1:]):
            layers += [nn.Linear(a, b), ACTS[act]()]
            if dropout:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(sizes[-1], n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LeNet(nn.Module):
    """Same architecture as the NumPy LeNet in scripts/train_numpy.py.  (N, 1, 28, 28) -> (N, 10)."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 6, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(6, 16, 5), nn.ReLU(), nn.MaxPool2d(2), nn.Flatten(),
            nn.Linear(400, 120), nn.ReLU(), nn.Linear(120, 84), nn.ReLU(), nn.Linear(84, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CifarCNN(nn.Module):
    """A small VGG-style CNN for CIFAR-10.

    (N,3,32,32) -> [conv3x3(32) ReLU conv3x3(32) ReLU pool]  -> (N,32,16,16)
                -> [conv3x3(64) ReLU conv3x3(64) ReLU pool]  -> (N,64,8,8)
                -> flatten 4096 -> FC 256 ReLU -> FC 10

    Args:
        dropout: Drop probability. If > 0, it is applied after each conv block at
            p/2 (lighter, because conv features are spatially correlated, so
            dropping single pixels removes little information) and before each
            FC layer at p. The FC layers hold ~95% of the parameters, which is
            where memorisation happens.
    """

    def __init__(self, dropout: float = 0.0) -> None:
        super().__init__()

        def block(c_in: int, c_out: int) -> list[nn.Module]:
            return [nn.Conv2d(c_in, c_out, 3, padding=1), nn.ReLU(), nn.Conv2d(c_out, c_out, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)]

        drop_conv = [nn.Dropout(dropout / 2)] if dropout else []
        drop_fc = [nn.Dropout(dropout)] if dropout else []
        self.net = nn.Sequential(
            *block(3, 32), *drop_conv, *block(32, 64), *drop_conv, nn.Flatten(),
            *drop_fc, nn.Linear(64 * 8 * 8, 256), nn.ReLU(), *drop_fc, nn.Linear(256, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


INIT_SCHEMES = ("zeros", "constant", "small", "xavier", "he")


@torch.no_grad()
def apply_init(model: nn.Module, scheme: str) -> nn.Module:
    """Re-initialise every Linear/Conv2d weight with ``scheme``; biases are set to 0.

    Schemes (fan_in = inputs per unit, fan_out = outputs per unit):
        zeros:    W = 0.
        constant: W = 1/fan_in for every entry. Not zero, but every unit in a layer
                  is identical, which isolates the SYMMETRY problem from the scale problem.
        small:    W ~ N(0, 0.01^2). Breaks symmetry, but the scale is too small.
        xavier:   W ~ N(0, 2/(fan_in + fan_out))   (Glorot & Bengio 2010; derived for tanh/linear).
        he:       W ~ N(0, 2/fan_in)               (He et al. 2015; derived for ReLU).

    Returns the model, to allow chaining.
    """
    for m in model.modules():
        if not isinstance(m, (nn.Linear, nn.Conv2d)):
            continue
        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight)
        if scheme == "zeros":
            m.weight.zero_()
        elif scheme == "constant":
            m.weight.fill_(1.0 / fan_in)
        elif scheme == "small":
            m.weight.normal_(0.0, 0.01)
        elif scheme == "xavier":
            m.weight.normal_(0.0, math.sqrt(2.0 / (fan_in + fan_out)))
        elif scheme == "he":
            m.weight.normal_(0.0, math.sqrt(2.0 / fan_in))
        else:
            raise ValueError(f"unknown init {scheme!r}")
        m.bias.zero_()
    return model
