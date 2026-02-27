from __future__ import annotations

import numpy as np


def select_k_eigengap(eigenvalues: np.ndarray, min_k: int, max_k: int) -> int:
    if max_k >= len(eigenvalues):
        max_k = len(eigenvalues) - 1
    gaps = eigenvalues[min_k:max_k] - eigenvalues[min_k + 1 : max_k + 1]
    if gaps.size == 0:
        return min_k
    return int(np.argmax(gaps) + min_k)
