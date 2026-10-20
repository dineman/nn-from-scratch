"""Vectorized NumPy layers with hand-written backward passes.

Every layer follows the same contract:

    out = layer.forward(x, train)   # caches whatever backward will need
    dx  = layer.backward(dout)      # dout = dL/d(out), same shape as out
                                    # fills layer.grads[name] = dL/d(param), same shape as the param
                                    # returns dx = dL/dx, same shape as x

This is exactly Project 1's ``_backward`` closure, scaled up from one
scalar to a whole batch of arrays. The chain rule is the same; the only new
work is writing each local derivative as a matrix expression (a
vector-Jacobian product) instead of a scalar product.

Shape conventions:
    N = batch size, D = features, C = channels, H/W = height/width,
    F = number of conv filters, K = kernel size.
"""

from __future__ import annotations

import numpy as np

from nnlib.im2col import col2im, conv_output_size, im2col


class Layer:
    """Base class. Parameter-free layers simply keep the empty dicts."""

    def __init__(self) -> None:
        self.params: dict[str, np.ndarray] = {}
        self.grads: dict[str, np.ndarray] = {}

    def forward(self, x: np.ndarray, train: bool = True) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def backward(self, dout: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


def init_weight(shape: tuple[int, ...], fan_in: int, fan_out: int, scheme: str, rng: np.random.Generator) -> np.ndarray:
    """Draw an initial weight array (float32).

    Args:
        shape: Output shape, e.g. (D_in, D_out) or (F, C, K, K).
        fan_in: Number of inputs feeding each output unit (D_in, or C*K*K for conv).
        fan_out: Number of outputs each input feeds (D_out, or F*K*K for conv).
        scheme: ``"he"``  -> N(0, 2/fan_in): keeps Var(activations) constant through ReLU layers.
                ``"xavier"`` -> U(+-sqrt(6/(fan_in+fan_out))): Var = 2/(fan_in+fan_out), designed
                for tanh/linear units (keeps both forward and backward variance roughly constant).
        rng: NumPy random generator (seeded by the caller).

    Returns:
        float32 array of ``shape``.
    """
    if scheme == "he":
        w = rng.normal(0.0, np.sqrt(2.0 / fan_in), size=shape)
    elif scheme == "xavier":
        a = np.sqrt(6.0 / (fan_in + fan_out))
        w = rng.uniform(-a, a, size=shape)
    else:
        raise ValueError(f"unknown init scheme {scheme!r}")
    return w.astype(np.float32)


class Linear(Layer):
    """Fully connected layer: out = x @ W + b.

    Args:
        d_in, d_out: Input / output feature sizes.
        rng: Random generator for the weights.
        init: ``"he"`` (default, for ReLU nets) or ``"xavier"``.

    Shapes:
        x (N, d_in) -> out (N, d_out);  W (d_in, d_out);  b (d_out,).
    """

    def __init__(self, d_in: int, d_out: int, rng: np.random.Generator, init: str = "he") -> None:
        super().__init__()
        self.params["W"] = init_weight((d_in, d_out), d_in, d_out, init, rng)
        self.params["b"] = np.zeros(d_out, dtype=np.float32)

    def forward(self, x: np.ndarray, train: bool = True) -> np.ndarray:
        self.x = x
        return x @ self.params["W"] + self.params["b"]

    def backward(self, dout: np.ndarray) -> np.ndarray:
        # out[n, j] = sum_i x[n, i] W[i, j] + b[j]
        #   dW = x^T dout:  (d_in, N) @ (N, d_out) -> sums each sample's contribution over the batch
        #   db = sum over the batch of dout: b is broadcast to all N rows, so its grad sums them
        #   dx = dout W^T:  (N, d_out) @ (d_out, d_in)
        self.grads["W"] = self.x.T @ dout
        self.grads["b"] = dout.sum(axis=0)
        return dout @ self.params["W"].T


class ReLU(Layer):
    """out = max(0, x), elementwise.  Any shape."""

    def forward(self, x: np.ndarray, train: bool = True) -> np.ndarray:
        self.mask = x > 0
        return x * self.mask

    def backward(self, dout: np.ndarray) -> np.ndarray:
        # Local derivative is 1 where the input was positive, 0 elsewhere (0 at exactly 0, like PyTorch).
        return dout * self.mask


class Dropout(Layer):
    """Inverted dropout: zero each unit with probability p during training.

    Args:
        p: Drop probability, in [0, 1).
        rng: Random generator for the masks.

    Why "inverted": the kept units are scaled by 1/(1-p) at TRAIN time, so the
    expected activation is unchanged: E[out] = (1-p) * x/(1-p) = x. Then at
    test time the layer is simply the identity, with no rescaling to remember.
    """

    def __init__(self, p: float, rng: np.random.Generator) -> None:
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError("p must be in [0, 1)")
        self.p, self.rng = p, rng

    def forward(self, x: np.ndarray, train: bool = True) -> np.ndarray:
        if not train or self.p == 0.0:
            self.mask = None
            return x
        self.mask = (self.rng.random(x.shape) >= self.p).astype(x.dtype) / (1.0 - self.p)
        return x * self.mask

    def backward(self, dout: np.ndarray) -> np.ndarray:
        # Dropped units contributed nothing, so they receive no gradient; kept units pass it through, scaled.
        return dout if self.mask is None else dout * self.mask


class Flatten(Layer):
    """(N, C, H, W) -> (N, C*H*W). Pure reshape, so backward is the inverse reshape."""

    def forward(self, x: np.ndarray, train: bool = True) -> np.ndarray:
        self.shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        return dout.reshape(self.shape)


class Conv2D(Layer):
    """2-D convolution (strictly: cross-correlation, as in every DL library) via im2col.

    Args:
        c_in: Input channels C.
        c_out: Number of filters F.
        k: Square kernel size K.
        stride, pad: Usual meanings; zero padding.
        rng: Random generator.
        init: ``"he"`` or ``"xavier"``.

    Shapes:
        x (N, C, H, W) -> out (N, F, Ho, Wo), with Ho = (H + 2*pad - K) // stride + 1.
        W (F, C, K, K), b (F,).

    The idea of im2col: every output pixel is a dot product between one filter
    and one K x K x C input patch. Copy every patch into a row of a matrix
    ``cols`` (N*Ho*Wo, C*K*K) and reshape the filters to (F, C*K*K). Then the
    whole convolution is ONE matrix multiply, cols @ W_col.T, and its backward
    pass is exactly Linear's backward, followed by col2im to scatter the patch
    gradients back to the pixels they came from.
    """

    def __init__(self, c_in: int, c_out: int, k: int, rng: np.random.Generator, stride: int = 1, pad: int = 0, init: str = "he") -> None:
        super().__init__()
        self.k, self.stride, self.pad = k, stride, pad
        self.params["W"] = init_weight((c_out, c_in, k, k), c_in * k * k, c_out * k * k, init, rng)
        self.params["b"] = np.zeros(c_out, dtype=np.float32)

    def forward(self, x: np.ndarray, train: bool = True) -> np.ndarray:
        N, _, H, W = x.shape
        F = self.params["W"].shape[0]
        Ho = conv_output_size(H, self.k, self.stride, self.pad)
        Wo = conv_output_size(W, self.k, self.stride, self.pad)
        self.x_shape = x.shape
        self.cols = im2col(x, self.k, self.stride, self.pad)  # (N*Ho*Wo, C*K*K)
        w_col = self.params["W"].reshape(F, -1)  # (F, C*K*K)
        out = self.cols @ w_col.T + self.params["b"]  # (N*Ho*Wo, F)
        return out.reshape(N, Ho, Wo, F).transpose(0, 3, 1, 2)  # (N, F, Ho, Wo)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        F = self.params["W"].shape[0]
        # Undo the final reshape/transpose so that dout lines up row-for-row with self.cols
        dout_rows = dout.transpose(0, 2, 3, 1).reshape(-1, F)  # (N*Ho*Wo, F)
        # From here on it is Linear's backward, with cols in the role of x:
        self.grads["W"] = (dout_rows.T @ self.cols).reshape(self.params["W"].shape)
        self.grads["b"] = dout_rows.sum(axis=0)
        dcols = dout_rows @ self.params["W"].reshape(F, -1)  # (N*Ho*Wo, C*K*K)
        # Overlapping patches share pixels; col2im ADDS every patch's contribution back
        # (the "sum over uses" rule from Project 1, now applied to pixels).
        return col2im(dcols, self.x_shape, self.k, self.stride, self.pad)


class MaxPool2D(Layer):
    """Non-overlapping K x K max-pooling (stride = K).

    Args:
        k: Window size, which is also the stride. H and W must be divisible by k.

    Shapes: x (N, C, H, W) -> out (N, C, H/k, W/k).

    Backward: max is "pass the winner through". The gradient of each output
    goes entirely to the input position that was the max of its window; every
    other position in the window gets 0. Ties go to the first maximum in
    row-major order, which is the same as PyTorch.

    Restricting to stride = k lets us do it with a pure reshape (no im2col):
    windows don't overlap, so each input pixel belongs to exactly one window.
    """

    def __init__(self, k: int = 2) -> None:
        super().__init__()
        self.k = k

    def forward(self, x: np.ndarray, train: bool = True) -> np.ndarray:
        N, C, H, W = x.shape
        k = self.k
        if H % k or W % k:
            raise ValueError(f"MaxPool2D(k={k}) needs H, W divisible by k; got {H}x{W}")
        # (N, C, H/k, k, W/k, k) -> (N, C, H/k, W/k, k*k): each window flattened on the last axis
        win = x.reshape(N, C, H // k, k, W // k, k).transpose(0, 1, 2, 4, 3, 5).reshape(N, C, H // k, W // k, k * k)
        self.argmax = win.argmax(axis=-1)  # (N, C, H/k, W/k): which of the k*k positions won
        self.x_shape = x.shape
        return np.take_along_axis(win, self.argmax[..., None], axis=-1)[..., 0]

    def backward(self, dout: np.ndarray) -> np.ndarray:
        N, C, H, W = self.x_shape
        k = self.k
        dwin = np.zeros((N, C, H // k, W // k, k * k), dtype=dout.dtype)
        np.put_along_axis(dwin, self.argmax[..., None], dout[..., None], axis=-1)
        # Inverse of the forward reshape/transpose
        return dwin.reshape(N, C, H // k, W // k, k, k).transpose(0, 1, 2, 4, 3, 5).reshape(N, C, H, W)


class Sequential:
    """A stack of layers applied in order; backward runs them in reverse.

    Reverse order is the topological order from Project 1, specialised to a
    chain: each layer's output is used by exactly one later layer, so there is
    nothing to accumulate between layers.
    """

    def __init__(self, *layers: Layer) -> None:
        self.layers = list(layers)

    def forward(self, x: np.ndarray, train: bool = True) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x, train)
        return x

    def backward(self, dout: np.ndarray) -> np.ndarray:
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def parameters(self) -> list[tuple[Layer, str]]:
        """Every (layer, param_name) pair, so an optimizer can read layer.params[name] and layer.grads[name]."""
        return [(layer, name) for layer in self.layers for name in layer.params]

    def num_params(self) -> int:
        return sum(layer.params[name].size for layer, name in self.parameters())

    def astype(self, dtype) -> "Sequential":
        """Cast all parameters (e.g. to float64 for gradient checking). Returns self."""
        for layer, name in self.parameters():
            layer.params[name] = layer.params[name].astype(dtype)
        return self
