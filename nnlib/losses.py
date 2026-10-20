"""Numerically stable softmax cross-entropy.

Notation: z (N, K) logits, y (N,) integer class labels, p = softmax(z).

    L = (1/N) * sum_n [ -log p[n, y_n] ]
      = (1/N) * sum_n [ logsumexp(z[n]) - z[n, y_n] ]

Gradient (derived step by step in NOTES.md, section 2):
    dL/dz = (p - onehot(y)) / N

Stability: exp(z) overflows float32 for z > ~88 (float64: ~709). The
log-sum-exp trick subtracts the row maximum m first:
    logsumexp(z) = m + log(sum_k exp(z_k - m)).
This is exact algebraically, and every exponent is <= 0, so nothing can
overflow. At least one term equals exp(0) = 1, so the sum is >= 1 and the log
never sees 0 either.
"""

from __future__ import annotations

import numpy as np


def logsumexp(z: np.ndarray) -> np.ndarray:
    """Row-wise log(sum(exp(z))) without overflow.  z (N, K) -> (N,)."""
    m = z.max(axis=1, keepdims=True)
    return (m + np.log(np.exp(z - m).sum(axis=1, keepdims=True)))[:, 0]


def softmax(z: np.ndarray) -> np.ndarray:
    """Row-wise softmax, stable for any finite z.  (N, K) -> (N, K), rows sum to 1."""
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def softmax_cross_entropy(z: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    """Mean cross-entropy loss and its gradient with respect to the logits.

    Args:
        z: (N, K) float logits (unnormalised scores).
        y: (N,) int class indices in [0, K).

    Returns:
        loss: Python float, the mean over the batch.
        dz: (N, K), dL/dz = (softmax(z) - onehot(y)) / N.

    Fusing softmax and cross-entropy matters. Computing log(softmax(z)) in two
    steps can produce log(0) = -inf when a probability underflows to 0. The
    fused form never takes the log of a probability at all.
    """
    N = z.shape[0]
    rows = np.arange(N)
    loss = float(np.mean(logsumexp(z) - z[rows, y]))
    dz = softmax(z)
    dz[rows, y] -= 1.0
    return loss, dz / N
