"""
Reproducibility utilities.

Setting a fixed seed does not guarantee bit-for-bit identical results
across all hardware/driver versions, but it removes randomness as a
variable when comparing experiments on the same machine.
"""

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    Seed all relevant random number generators.

    Args:
        seed: The seed value to use across random, numpy, and torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Forces cuDNN to use deterministic algorithms where possible.
    # Tradeoff: this can make some operations slower, but it is worth it
    # for reproducible experiment comparisons.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False