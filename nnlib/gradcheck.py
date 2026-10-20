"""Finite-difference gradient checking for array-valued functions."""

from __future__ import annotations

from typing import Callable

import numpy as np


def numerical_grad(f: Callable[[], float], x: np.ndarray, h: float = 1e-6) -> np.ndarray:
    """Central-difference estimate of df/dx, perturbing x IN PLACE one element at a time.

    Args:
        f: Zero-argument function returning a scalar. It must read ``x`` (by
            reference) every time it is called.
        x: float64 array to differentiate with respect to. It is restored afterwards.
        h: Step size. ~1e-6 is near-optimal for central differences in float64
            (truncation error O(h^2) vs roundoff error O(eps/h); see Project 1).

    Returns:
        Array shaped like x.

    Cost: 2 * x.size evaluations of f, so only use it on tiny inputs.
    """
    assert x.dtype == np.float64, "gradient checks need float64; float32 roundoff swamps the O(h^2) error"
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    for _ in it:
        i = it.multi_index
        old = x[i]
        x[i] = old + h
        fp = f()
        x[i] = old - h
        fm = f()
        x[i] = old
        grad[i] = (fp - fm) / (2 * h)
    return grad


def rel_error(a: np.ndarray, b: np.ndarray, floor: float = 1e-8) -> float:
    """max |a - b| / max(|a|, |b|, floor), elementwise and then the max over elements."""
    return float(np.max(np.abs(a - b) / np.maximum(np.maximum(np.abs(a), np.abs(b)), floor)))
