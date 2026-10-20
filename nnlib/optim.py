"""SGD (optionally with momentum) for ``nnlib.layers.Sequential`` models."""

from __future__ import annotations

import numpy as np

from nnlib.layers import Sequential


class SGD:
    """Stochastic gradient descent, using the same update convention as ``torch.optim.SGD``.

    Args:
        model: The model whose parameters are updated.
        lr: Learning rate (eta).
        momentum: mu in [0, 1). 0 gives plain SGD.
        weight_decay: L2 coefficient lambda; adds lambda * p to the gradient.

    Update, per parameter p with gradient g:
        g <- g + lambda * p
        v <- mu * v + g          (v starts at 0)
        p <- p - eta * v
    With mu = 0 this is p <- p - eta * g.
    """

    def __init__(self, model: Sequential, lr: float, momentum: float = 0.0, weight_decay: float = 0.0) -> None:
        self.model, self.lr, self.momentum, self.weight_decay = model, lr, momentum, weight_decay
        self.velocity = {id(layer.params[name]): np.zeros_like(layer.params[name]) for layer, name in model.parameters()}

    def step(self) -> None:
        for layer, name in self.model.parameters():
            p, g = layer.params[name], layer.grads[name]
            if self.weight_decay:
                g = g + self.weight_decay * p
            if self.momentum:
                v = self.velocity[id(p)]
                v *= self.momentum
                v += g
                g = v
            p -= self.lr * g  # in place, so id(p) and the velocity lookup stay valid
