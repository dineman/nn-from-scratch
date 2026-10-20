"""Load MNIST (IDX format) and CIFAR-10 (binary format) from data/raw with plain NumPy.

Run ``python -m scripts.download_data`` first. Both loaders return uint8
images and int64 labels; normalisation is done separately so its choice is
explicit at the call site.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"

CIFAR10_CLASSES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")


def _read_idx(path: Path, header_ints: int) -> np.ndarray:
    """Read a gzipped IDX file: a big-endian int32 header (magic, dims...), then raw uint8 data."""
    with gzip.open(path, "rb") as f:
        data = f.read()
    dims = np.frombuffer(data, dtype=">i4", count=header_ints)[1:]
    return np.frombuffer(data, dtype=np.uint8, offset=4 * header_ints).reshape(dims)


def load_mnist(root: Path = RAW) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns X_train (60000, 28, 28) uint8, y_train (60000,) int64, X_test (10000, 28, 28), y_test (10000,)."""
    X_tr = _read_idx(root / "train-images-idx3-ubyte.gz", 4)
    y_tr = _read_idx(root / "train-labels-idx1-ubyte.gz", 2).astype(np.int64)
    X_te = _read_idx(root / "t10k-images-idx3-ubyte.gz", 4)
    y_te = _read_idx(root / "t10k-labels-idx1-ubyte.gz", 2).astype(np.int64)
    return X_tr, y_tr, X_te, y_te


def load_cifar10(root: Path = RAW) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns X_train (50000, 3, 32, 32) uint8, y_train (50000,) int64, X_test (10000, 3, 32, 32), y_test (10000,).

    Binary format: each record is 1 label byte followed by 3072 pixel bytes
    (1024 red, then 1024 green, then 1024 blue, each 32x32 row-major), which
    is already channels-first.
    """
    d = root / "cifar-10-batches-bin"

    def read(files: list[str]) -> tuple[np.ndarray, np.ndarray]:
        raw = np.concatenate([np.fromfile(d / f, dtype=np.uint8).reshape(-1, 3073) for f in files])
        return raw[:, 1:].reshape(-1, 3, 32, 32), raw[:, 0].astype(np.int64)

    X_tr, y_tr = read([f"data_batch_{i}.bin" for i in range(1, 6)])
    X_te, y_te = read(["test_batch.bin"])
    return X_tr, y_tr, X_te, y_te


def train_val_split(X: np.ndarray, y: np.ndarray, n_val: int, seed: int = 0):
    """Shuffle with a fixed seed and hold out ``n_val`` examples for validation.

    The test set is never used for any choice (hyperparameters, early
    stopping). The validation set plays that role, and the test set is
    reported once at the end.

    Returns: X_train, y_train, X_val, y_val.
    """
    perm = np.random.default_rng(seed).permutation(len(X))
    val, tr = perm[:n_val], perm[n_val:]
    return X[tr], y[tr], X[val], y[val]


def standardize(X: np.ndarray, mean: np.ndarray | float, std: np.ndarray | float) -> np.ndarray:
    """(X/255 - mean) / std as float32. mean/std must come from the TRAINING split only."""
    return ((X.astype(np.float32) / 255.0 - mean) / std).astype(np.float32)
