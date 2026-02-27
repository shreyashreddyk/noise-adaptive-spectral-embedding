from __future__ import annotations

import numpy as np
from scipy.linalg import subspace_angles


def principal_angles(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.asarray(subspace_angles(u, v), dtype=float)


def subspace_similarity(u: np.ndarray, v: np.ndarray) -> float:
    angles = principal_angles(u, v)
    return float(np.mean(np.cos(angles)))
