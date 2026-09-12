"""
Sanity check script.

Purpose: verify that the Python environment, virtual environment,
and core dependencies (torch, numpy, yaml) are correctly installed
and behaving as expected before any real project code is written.
"""

import sys
import platform

import torch
import numpy
import yaml


def main() -> None:
    print("=" * 50)
    print("ENVIRONMENT SANITY CHECK")
    print("=" * 50)

    print(f"Python version   : {platform.python_version()}")
    print(f"Python executable: {sys.executable}")

    print(f"PyTorch version  : {torch.__version__}")
    print(f"NumPy version    : {numpy.__version__}")
    print(f"PyYAML version   : {yaml.__version__}")

    cuda_available = torch.cuda.is_available()
    print(f"CUDA available   : {cuda_available}")

    if cuda_available:
        device_name = torch.cuda.get_device_name(0)
        print(f"GPU device       : {device_name}")
        selected_device = "cuda"
    else:
        print("GPU device       : None (falling back to CPU)")
        selected_device = "cpu"

    print(f"Selected device  : {selected_device}")
    print("=" * 50)
    print("Sanity check passed.")


if __name__ == "__main__":
    main()