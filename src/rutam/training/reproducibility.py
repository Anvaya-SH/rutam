"""Reproducibility utilities for local deterministic training runs."""

from __future__ import annotations

import os
import random

import torch


def set_seed(seed: int = 42) -> None:
    """Set seeds across Python standard library, NumPy (if available), and PyTorch."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
