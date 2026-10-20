"""Layer tests: output shapes, finite-difference gradient checks, and the im2col adjoint.

Gradient-check recipe for a layer (out = layer(x)):
    pick a fixed random R shaped like out, and define the scalar L = sum(out * R).
    Then dL/d(out) = R, so ``layer.backward(R)`` must return dL/dx and fill
    dL/dW, and both can be compared with finite differences of L.
A random R checks every output's gradient at once, with random weights, so
there is no lucky cancellation.
"""

from __future__ import annotations

import numpy as np
import pytest

from nnlib.gradcheck import numerical_grad, rel_error
from nnlib.im2col import col2im, im2col
from nnlib.layers import Conv2D, Dropout, Flatten, Linear, MaxPool2D, ReLU, Sequential
from nnlib.losses import logsumexp, softmax, softmax_cross_entropy

TOL = 1e-6


def rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def _check_layer(layer, x: np.ndarray, train: bool = True) -> None:
    """Compare analytic dL/dx and dL/dparams with finite differences, for L = sum(layer(x) * R)."""
    for name in layer.params:
        layer.params[name] = layer.params[name].astype(np.float64)
    out = layer.forward(x, train)
    R = rng(99).normal(size=out.shape)

    def L() -> float:
        return float(np.sum(layer.forward(x, train) * R))

    layer.forward(x, train)
    dx = layer.backward(R)
    analytic = {name: g.copy() for name, g in layer.grads.items()}

    assert dx.shape == x.shape
    assert rel_error(dx, numerical_grad(L, x)) < TOL, "dx mismatch"
    for name, p in layer.params.items():
        assert analytic[name].shape == p.shape
        assert rel_error(analytic[name], numerical_grad(L, p)) < TOL, f"d{name} mismatch"


# ----------------------------------------------------------------------------- shapes
@pytest.mark.parametrize(
    "layer, in_shape, out_shape",
    [
        (Linear(5, 3, rng()), (4, 5), (4, 3)),
        (ReLU(), (2, 3, 4, 4), (2, 3, 4, 4)),
        (Dropout(0.5, rng()), (4, 7), (4, 7)),
        (Flatten(), (2, 3, 4, 5), (2, 60)),
        (Conv2D(3, 8, 3, rng()), (2, 3, 10, 10), (2, 8, 8, 8)),
        (Conv2D(3, 8, 3, rng(), pad=1), (2, 3, 10, 10), (2, 8, 10, 10)),
        (Conv2D(3, 8, 3, rng(), stride=2, pad=1), (2, 3, 10, 10), (2, 8, 5, 5)),
        (Conv2D(1, 6, 5, rng(), pad=2), (2, 1, 28, 28), (2, 6, 28, 28)),
        (MaxPool2D(2), (2, 3, 8, 6), (2, 3, 4, 3)),
    ],
)
def test_forward_and_backward_shapes(layer, in_shape, out_shape):
    x = rng().normal(size=in_shape).astype(np.float32)
    out = layer.forward(x, train=True)
    assert out.shape == out_shape
    dx = layer.backward(np.ones_like(out))
    assert dx.shape == in_shape
    for name, p in layer.params.items():
        assert layer.grads[name].shape == p.shape


def test_maxpool_rejects_indivisible_input():
    with pytest.raises(ValueError):
        MaxPool2D(2).forward(np.zeros((1, 1, 5, 4)))


# ----------------------------------------------------------------------------- gradient checks
def test_linear_gradcheck():
    _check_layer(Linear(6, 4, rng()), rng(1).normal(size=(5, 6)))


def test_relu_gradcheck():
    # Keep inputs away from the kink at 0, where finite differences are meaningless.
    x = rng(1).normal(size=(4, 5))
    x[np.abs(x) < 0.1] = 0.5
    _check_layer(ReLU(), x)


def test_dropout_gradcheck_with_fixed_mask():
    layer = Dropout(0.4, rng())
    x = rng(1).normal(size=(6, 5))
    layer.forward(x, train=True)
    mask = layer.mask
    # Freeze the mask so that the function is deterministic for the finite differences
    layer.forward = lambda x, train=True: x * mask
    _check_layer(layer, x)


def test_dropout_is_identity_at_eval_and_preserves_mean_in_training():
    layer = Dropout(0.3, rng())
    x = np.ones((2000, 50))
    assert np.array_equal(layer.forward(x, train=False), x)
    # E[out] = x thanks to the 1/(1-p) scaling
    assert abs(layer.forward(x, train=True).mean() - 1.0) < 0.01


@pytest.mark.parametrize("stride, pad", [(1, 0), (1, 1), (2, 1), (2, 0)])
def test_conv_gradcheck(stride, pad):
    _check_layer(Conv2D(2, 3, 3, rng(), stride=stride, pad=pad), rng(1).normal(size=(2, 2, 5, 5)))


def test_maxpool_gradcheck():
    # Random normals: ties have probability 0, and the max is locally differentiable.
    _check_layer(MaxPool2D(2), rng(1).normal(size=(2, 3, 4, 4)))


def test_flatten_gradcheck():
    _check_layer(Flatten(), rng(1).normal(size=(2, 3, 2, 2)))


# ----------------------------------------------------------------------------- im2col
@pytest.mark.parametrize("k, stride, pad", [(3, 1, 0), (3, 1, 1), (3, 2, 1), (2, 2, 0), (5, 1, 2)])
def test_col2im_is_adjoint_of_im2col(k, stride, pad):
    # <im2col(x), G> == <x, col2im(G)> for all x, G. This is the definition of the
    # transpose of a linear map, and it is exactly why col2im is im2col's backward.
    x = rng(0).normal(size=(2, 3, 7, 7))
    cols = im2col(x, k, stride, pad)
    G = rng(1).normal(size=cols.shape)
    assert np.isclose(np.sum(cols * G), np.sum(x * col2im(G, x.shape, k, stride, pad)))


def test_im2col_rows_are_patches():
    x = np.arange(16.0).reshape(1, 1, 4, 4)
    cols = im2col(x, k=2, stride=2)
    np.testing.assert_array_equal(cols, [[0, 1, 4, 5], [2, 3, 6, 7], [8, 9, 12, 13], [10, 11, 14, 15]])


# ----------------------------------------------------------------------------- loss
def test_softmax_ce_gradient_is_p_minus_y_over_n():
    z = rng(0).normal(size=(5, 4))
    y = np.array([0, 3, 1, 1, 2])
    _, dz = softmax_cross_entropy(z, y)
    expected = softmax(z)
    expected[np.arange(5), y] -= 1
    np.testing.assert_allclose(dz, expected / 5)
    numeric = numerical_grad(lambda: softmax_cross_entropy(z, y)[0], z)
    assert rel_error(dz, numeric) < TOL


def test_softmax_ce_is_stable_for_huge_logits():
    # The naive log(exp(z)/sum(exp(z))) gives inf/inf = nan here.
    z = np.array([[1000.0, 0.0, -1000.0], [-1e4, 1e4, 0.0]])
    loss, dz = softmax_cross_entropy(z, np.array([0, 1]))
    assert np.isfinite(loss) and np.all(np.isfinite(dz))
    assert loss == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_allclose(logsumexp(np.array([[1000.0, 1000.0]])), [1000 + np.log(2)])


def test_softmax_ce_uniform_logits_give_log_k():
    # A sanity check worth memorising: a network that outputs equal logits has
    # loss log(K). At initialization, the MNIST loss should be close to log(10) = 2.30.
    loss, _ = softmax_cross_entropy(np.zeros((3, 10)), np.array([1, 2, 3]))
    assert loss == pytest.approx(np.log(10))


# ----------------------------------------------------------------------------- whole network
def test_lenet_style_network_gradcheck_end_to_end():
    # Every layer type chained together, with the gradient of the actual training loss.
    r = rng(0)
    net = Sequential(
        Conv2D(1, 2, 3, r, pad=1), ReLU(), MaxPool2D(2),
        Conv2D(2, 3, 3, r), ReLU(), Flatten(),
        Linear(3 * 2 * 2, 5, r), ReLU(), Linear(5, 3, r),
    ).astype(np.float64)
    x = rng(1).normal(size=(2, 1, 8, 8))
    y = np.array([0, 2])

    def L() -> float:
        return softmax_cross_entropy(net.forward(x), y)[0]

    _, dz = softmax_cross_entropy(net.forward(x), y)
    net.backward(dz)
    for layer, name in net.parameters():
        analytic = layer.grads[name].copy()
        assert rel_error(analytic, numerical_grad(L, layer.params[name])) < 1e-5, f"{type(layer).__name__}.{name}"
