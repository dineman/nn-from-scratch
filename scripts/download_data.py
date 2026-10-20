"""Download MNIST and CIFAR-10 (binary version) into data/raw and verify checksums.

    python -m scripts.download_data

Uses only the standard library. The CIFAR-10 *binary* archive is used rather
than the "python version", which is a pickle: unpickling a downloaded file
can run arbitrary code, whereas the binary format is just raw bytes that
we parse with NumPy (see nnlib/data.py).
"""

from __future__ import annotations

import hashlib
import tarfile
import urllib.request
from pathlib import Path

RAW = Path("data/raw")

FILES = {
    # MNIST mirror used by torchvision (the original yann.lecun.com host is unreliable)
    "train-images-idx3-ubyte.gz": ("https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
    "train-labels-idx1-ubyte.gz": ("https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
    "t10k-images-idx3-ubyte.gz": ("https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3"),
    "t10k-labels-idx1-ubyte.gz": ("https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c"),
    "cifar-10-binary.tar.gz": ("https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz", "c32a1d4ab5d03f1284b67883e8d87530"),
}


def md5(path: Path) -> str:
    """Hex MD5 of a file, read in 1 MB chunks."""
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for name, (url, expected) in FILES.items():
        path = RAW / name
        if path.exists() and md5(path) == expected:
            print(f"ok      {name}")
            continue
        print(f"fetch   {name} ...", flush=True)
        urllib.request.urlretrieve(url, path)
        got = md5(path)
        if got != expected:
            path.unlink()
            raise RuntimeError(f"checksum mismatch for {name}: got {got}, expected {expected}")
        print(f"ok      {name}")

    cifar_dir = RAW / "cifar-10-batches-bin"
    if not cifar_dir.exists():
        with tarfile.open(RAW / "cifar-10-binary.tar.gz") as tar:
            # filter="data" rejects absolute paths, "..", links and device files in the archive
            tar.extractall(RAW, filter="data")
    print("done:", sorted(p.name for p in cifar_dir.iterdir()))


if __name__ == "__main__":
    main()
