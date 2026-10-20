"""Sanity checks on the raw-data parsers (skipped if the data has not been downloaded)."""

from __future__ import annotations

import numpy as np
import pytest

from nnlib.data import RAW, load_cifar10, load_mnist, train_val_split


@pytest.mark.skipif(not (RAW / "train-images-idx3-ubyte.gz").exists(), reason="MNIST not downloaded")
def test_mnist_shapes_and_labels():
    X, y, Xt, yt = load_mnist()
    assert X.shape == (60000, 28, 28) and Xt.shape == (10000, 28, 28) and X.dtype == np.uint8
    assert set(np.unique(y)) == set(range(10))
    # Known class counts of the MNIST test set
    assert np.bincount(yt).tolist() == [980, 1135, 1032, 1010, 982, 892, 958, 1028, 974, 1009]


@pytest.mark.skipif(not (RAW / "cifar-10-batches-bin").exists(), reason="CIFAR-10 not downloaded")
def test_cifar_shapes_balance_and_channel_order():
    X, y, Xt, yt = load_cifar10()
    assert X.shape == (50000, 3, 32, 32) and Xt.shape == (10000, 3, 32, 32)
    assert np.bincount(y).tolist() == [5000] * 10 and np.bincount(yt).tolist() == [1000] * 10
    # If the record layout were misparsed (e.g. off by the label byte), neighbouring pixels would be
    # uncorrelated. In real images, horizontally adjacent pixels are strongly correlated.
    a = X[:500, :, :, :-1].astype(float).ravel()
    b = X[:500, :, :, 1:].astype(float).ravel()
    assert np.corrcoef(a, b)[0, 1] > 0.8


def test_train_val_split_is_disjoint_and_deterministic():
    X = np.arange(100)
    y = np.arange(100)
    a = train_val_split(X, y, 20, seed=0)
    b = train_val_split(X, y, 20, seed=0)
    assert all(np.array_equal(u, v) for u, v in zip(a, b))
    assert len(set(a[0]) & set(a[2])) == 0 and len(a[0]) == 80
