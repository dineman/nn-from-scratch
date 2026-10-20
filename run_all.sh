#!/usr/bin/env bash
# Reproduce every number and figure in README.md.
# Usage: conda activate mlpcnn && ./run_all.sh        (~3 h on a 10-core laptop CPU; no GPU needed)
set -euo pipefail
cd "$(dirname "$0")"

python -m scripts.download_data             # MNIST + CIFAR-10 (binary), checksummed
python -m pytest -q                         # 42 tests, ~2 s

python -m scripts.train_numpy --model mlp   # Part A  (~20 s)
python -m scripts.train_numpy --model lenet # Part B  (~3 min)
python -m scripts.torch_rebuild             # Part C rebuild + CIFAR reference (~16 min)
python -m scripts.exp_init                  # C1 initialization (~15 min)
python -m scripts.exp_init_depth            # C1b init vs depth (~15 min)
python -m scripts.exp_dropout               # C2 dropout (~45 min)
python -m scripts.exp_optim                 # C3 optimizers + LR sweep (~90 min)
