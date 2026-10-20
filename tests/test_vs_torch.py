"""NumPy layers vs PyTorch on identical weights: forward outputs AND gradients.

Finite differences check that the backward pass is consistent with our own
forward. Comparing with PyTorch additionally checks that our forward computes
the SAME function as the standard implementation: same convolution
orientation (cross-correlation), same padding, same pooling tie-breaking, same
loss reduction.
"""

from __future__ import annotations

import numpy as np
import pytest

from nnlib.layers import Conv2D, Flatten, Linear, MaxPool2D, ReLU, Sequential
from nnlib.losses import softmax_cross_entropy

torch = pytest.importorskip("torch")
F = torch.nn.functional


def _t(a: np.ndarray, grad: bool = False):
    return torch.tensor(a, dtype=torch.float64, requires_grad=grad)


def test_linear_matches_torch():
    r = np.random.default_rng(0)
    layer = Linear(7, 4, r)
    layer.params = {k: v.astype(np.float64) for k, v in layer.params.items()}
    x, dout = r.normal(size=(5, 7)), r.normal(size=(5, 4))
    out = layer.forward(x)
    dx = layer.backward(dout)

    tx, tW, tb = _t(x, True), _t(layer.params["W"], True), _t(layer.params["b"], True)
    tout = tx @ tW + tb
    tout.backward(_t(dout))
    np.testing.assert_allclose(out, tout.detach().numpy(), rtol=1e-12)
    np.testing.assert_allclose(dx, tx.grad.numpy(), rtol=1e-10)
    np.testing.assert_allclose(layer.grads["W"], tW.grad.numpy(), rtol=1e-10)
    np.testing.assert_allclose(layer.grads["b"], tb.grad.numpy(), rtol=1e-10)


@pytest.mark.parametrize("stride, pad", [(1, 0), (1, 2), (2, 1)])
def test_conv_matches_torch(stride, pad):
    r = np.random.default_rng(0)
    layer = Conv2D(3, 4, 5, r, stride=stride, pad=pad)
    layer.params = {k: v.astype(np.float64) for k, v in layer.params.items()}
    x = r.normal(size=(2, 3, 9, 9))
    out = layer.forward(x)
    dout = r.normal(size=out.shape)
    dx = layer.backward(dout)

    tx, tW, tb = _t(x, True), _t(layer.params["W"], True), _t(layer.params["b"], True)
    tout = F.conv2d(tx, tW, tb, stride=stride, padding=pad)
    tout.backward(_t(dout))
    np.testing.assert_allclose(out, tout.detach().numpy(), rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(dx, tx.grad.numpy(), rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(layer.grads["W"], tW.grad.numpy(), rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(layer.grads["b"], tb.grad.numpy(), rtol=1e-10, atol=1e-12)


def test_maxpool_matches_torch_including_ties():
    # Integer-valued input creates many ties. Both implementations must route
    # the gradient to the same element of each window.
    r = np.random.default_rng(0)
    x = r.integers(0, 3, size=(2, 3, 6, 6)).astype(np.float64)
    layer = MaxPool2D(2)
    out = layer.forward(x)
    dout = r.normal(size=out.shape)
    dx = layer.backward(dout)

    tx = _t(x, True)
    tout = F.max_pool2d(tx, 2)
    tout.backward(_t(dout))
    np.testing.assert_allclose(out, tout.detach().numpy())
    np.testing.assert_allclose(dx, tx.grad.numpy())


def test_softmax_ce_matches_torch():
    r = np.random.default_rng(0)
    z, y = r.normal(size=(8, 10)) * 5, r.integers(0, 10, size=8)
    loss, dz = softmax_cross_entropy(z, y)
    tz = _t(z, True)
    tloss = F.cross_entropy(tz, torch.tensor(y))
    tloss.backward()
    assert loss == pytest.approx(tloss.item(), rel=1e-12)
    np.testing.assert_allclose(dz, tz.grad.numpy(), rtol=1e-10)


def test_full_lenet_matches_torch():
    # The same LeNet as Part B, in NumPy and in torch.nn, with the same weights:
    # compare the loss and every parameter gradient.
    r = np.random.default_rng(0)
    net = Sequential(
        Conv2D(1, 6, 5, r, pad=2), ReLU(), MaxPool2D(2),
        Conv2D(6, 16, 5, r), ReLU(), MaxPool2D(2), Flatten(),
        Linear(400, 120, r), ReLU(), Linear(120, 84, r), ReLU(), Linear(84, 10, r),
    ).astype(np.float64)
    x, y = r.normal(size=(4, 1, 28, 28)), r.integers(0, 10, size=4)
    loss, dz = softmax_cross_entropy(net.forward(x), y)
    net.backward(dz)

    nn = torch.nn
    tnet = nn.Sequential(
        nn.Conv2d(1, 6, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(6, 16, 5), nn.ReLU(), nn.MaxPool2d(2), nn.Flatten(),
        nn.Linear(400, 120), nn.ReLU(), nn.Linear(120, 84), nn.ReLU(), nn.Linear(84, 10),
    ).double()
    ours = [(layer, n) for layer, n in net.parameters()]
    theirs = [p for p in tnet.parameters()]
    with torch.no_grad():
        for (layer, n), tp in zip(ours, theirs):
            w = layer.params[n]
            # torch.nn.Linear stores W as (out, in), and computes x @ W.T; ours is (in, out)
            tp.copy_(torch.from_numpy(w.T.copy() if isinstance(layer, Linear) and n == "W" else w))
    tloss = F.cross_entropy(tnet(_t(x)), torch.tensor(y))
    tloss.backward()

    assert loss == pytest.approx(tloss.item(), rel=1e-10)
    for (layer, n), tp in zip(ours, theirs):
        g = tp.grad.numpy()
        g = g.T if isinstance(layer, Linear) and n == "W" else g
        np.testing.assert_allclose(layer.grads[n], g, rtol=1e-8, atol=1e-12, err_msg=f"{type(layer).__name__}.{n}")
