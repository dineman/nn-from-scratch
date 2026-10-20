"""End-to-end: the NumPy MLP and a torch MLP with the same weights and batches train identically."""

from __future__ import annotations

import numpy as np
import pytest

from nnlib.layers import Linear, ReLU, Sequential
from nnlib.losses import softmax_cross_entropy
from nnlib.optim import SGD

torch = pytest.importorskip("torch")


@pytest.mark.parametrize("momentum", [0.0, 0.9])
def test_numpy_and_torch_sgd_trajectories_match(momentum):
    r = np.random.default_rng(0)
    net = Sequential(Linear(20, 16, r), ReLU(), Linear(16, 8, r), ReLU(), Linear(8, 5, r)).astype(np.float64)
    tnet = torch.nn.Sequential(
        torch.nn.Linear(20, 16), torch.nn.ReLU(), torch.nn.Linear(16, 8), torch.nn.ReLU(), torch.nn.Linear(8, 5)
    ).double()
    with torch.no_grad():
        for (layer, n), tp in zip(net.parameters(), tnet.parameters()):
            tp.copy_(torch.from_numpy(layer.params[n].T.copy() if n == "W" else layer.params[n]))

    opt = SGD(net, lr=0.1, momentum=momentum)
    topt = torch.optim.SGD(tnet.parameters(), lr=0.1, momentum=momentum)
    X, y = r.normal(size=(200, 20)), r.integers(0, 5, size=200)
    ours, theirs = [], []
    for step in range(30):
        idx = np.arange(step * 16, step * 16 + 16) % 200
        loss, dz = softmax_cross_entropy(net.forward(X[idx]), y[idx])
        net.backward(dz)
        opt.step()
        ours.append(loss)

        tloss = torch.nn.functional.cross_entropy(tnet(torch.from_numpy(X[idx])), torch.from_numpy(y[idx]))
        topt.zero_grad()
        tloss.backward()
        topt.step()
        theirs.append(tloss.item())
    np.testing.assert_allclose(ours, theirs, rtol=1e-9)
