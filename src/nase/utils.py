from __future__ import annotations

import random

import numpy as np


def set_global_seed(seed: int) -> np.random.Generator:
    """Set deterministic seeds and return a NumPy generator."""
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)
