"""im2col / col2im: turn convolution into matrix multiplication and back.

im2col copies every K x K x C receptive field ("patch") of the padded input
into one row of a matrix. col2im is its adjoint (the transpose of the linear
map im2col): it takes one gradient row per patch and ADDS each entry back to
the pixel it was copied from. A pixel that sits in several overlapping
patches receives the sum of all their contributions.

Test of the adjoint property (tests/test_im2col.py): for any x and G,
    <im2col(x), G> == <x, col2im(G)>.
That single identity is what makes col2im the correct backward of im2col.
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def conv_output_size(size: int, k: int, stride: int, pad: int) -> int:
    """Spatial output size of a convolution: floor((size + 2*pad - k) / stride) + 1."""
    out = (size + 2 * pad - k) // stride + 1
    if out <= 0:
        raise ValueError(f"kernel {k} too large for input {size} with pad {pad}")
    return out


def im2col(x: np.ndarray, k: int, stride: int = 1, pad: int = 0) -> np.ndarray:
    """Extract all K x K patches.

    Args:
        x: (N, C, H, W) input.
        k: Kernel size.
        stride, pad: Convolution stride and zero padding.

    Returns:
        cols: (N*Ho*Wo, C*K*K). Row index = (n, i, j) of the output pixel in
        row-major order. Column index = (c, di, dj) within the patch, which
        matches ``W.reshape(F, C*K*K)``.
    """
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad))) if pad else x
    # sliding_window_view returns a VIEW (no copy) of shape (N, C, H', W', K, K);
    # stepping by `stride` keeps only the windows the convolution actually visits.
    win = sliding_window_view(xp, (k, k), axis=(2, 3))[:, :, ::stride, ::stride]
    N, C, Ho, Wo = win.shape[:4]
    # The reshape after the transpose is where the actual copy happens.
    return win.transpose(0, 2, 3, 1, 4, 5).reshape(N * Ho * Wo, C * k * k)


def col2im(cols: np.ndarray, x_shape: tuple[int, int, int, int], k: int, stride: int = 1, pad: int = 0) -> np.ndarray:
    """Adjoint of im2col: scatter-add patch rows back onto the image.

    Args:
        cols: (N*Ho*Wo, C*K*K), e.g. dL/d(cols).
        x_shape: (N, C, H, W) of the original, unpadded input.
        k, stride, pad: Same as for im2col.

    Returns:
        (N, C, H, W) array; each pixel holds the sum over all patches containing it.

    Loops over the K*K kernel offsets only (25 iterations for 5x5) and is
    vectorized over everything else. Each iteration adds one kernel position's
    gradient into a strided slice of the padded image.
    """
    N, C, H, W = x_shape
    Ho = conv_output_size(H, k, stride, pad)
    Wo = conv_output_size(W, k, stride, pad)
    g = cols.reshape(N, Ho, Wo, C, k, k).transpose(0, 3, 4, 5, 1, 2)  # (N, C, K, K, Ho, Wo)
    xp = np.zeros((N, C, H + 2 * pad, W + 2 * pad), dtype=cols.dtype)
    for di in range(k):
        for dj in range(k):
            # Kernel offset (di, dj) of output pixel (i, j) reads input pixel (i*stride + di, j*stride + dj)
            xp[:, :, di : di + stride * Ho : stride, dj : dj + stride * Wo : stride] += g[:, :, di, dj]
    return xp[:, :, pad : pad + H, pad : pad + W] if pad else xp
