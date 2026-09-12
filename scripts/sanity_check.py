"""
Sanity check script.

Verifies that the environment, dependencies, and configuration
system are all working correctly together.
"""

import sys
import platform

import torch
import numpy
import yaml

from src.config import load_config, resolve_device
from src.utils.seed import set_seed


def main() -> None:
    print("=" * 50)
    print("ENVIRONMENT SANITY CHECK")
    print("=" * 50)

    print(f"Python version   : {platform.python_version()}")
    print(f"Python executable: {sys.executable}")
    print(f"PyTorch version  : {torch.__version__}")
    print(f"NumPy version    : {numpy.__version__}")
    print(f"PyYAML version   : {yaml.__version__}")

    print("=" * 50)
    print("CONFIGURATION CHECK")
    print("=" * 50)

    config = load_config("configs/base.yaml")
    print("Loaded config:")
    print(config)

    set_seed(config["seed"])
    print(f"Seed set to: {config['seed']}")

    device = resolve_device(config["device"])
    print(f"Resolved device: {device}")

    print("=" * 50)
    print("Sanity check passed.")


if __name__ == "__main__":
    main()