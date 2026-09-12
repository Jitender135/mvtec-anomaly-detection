"""
Configuration loading utilities.

Keeps a clean separation between "what the experiment settings are"
(YAML files) and "how the code behaves" (Python logic). Nothing in
this project should hardcode a path, hyperparameter, or device string
directly in model/script code — it should come from a loaded config.
"""

from pathlib import Path
from typing import Any

import torch
import yaml


def load_config(path: str) -> dict[str, Any]:
    """
    Load a YAML configuration file into a plain dictionary.

    Args:
        path: Path to the YAML config file.

    Returns:
        The parsed configuration as a nested dictionary.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path.resolve()}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def resolve_device(requested_device: str) -> str:
    """
    Resolve a requested device string into an actual usable device.

    Args:
        requested_device: One of "auto", "cuda", or "cpu".

    Returns:
        Either "cuda" or "cpu", depending on availability.
    """
    if requested_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"

    if requested_device == "cuda" and not torch.cuda.is_available():
        print("Warning: 'cuda' requested but not available. Falling back to CPU.")
        return "cpu"

    return requested_device