from __future__ import annotations

import numpy as np


def select_k_eigengap(eigenvalues: np.ndarray, min_k: int, max_k: int) -> int:
    """Choose k by largest eigengap in a bounded non-trivial range.

    `k` is interpreted as the index of the last retained non-trivial mode where the
    relevant gap is `lambda_k - lambda_{k+1}`. Guardrails ensure we never return
    `k=0` and avoid accidental use of the trivial leading eigenvalue.
    """
    evals = np.asarray(eigenvalues, dtype=float).reshape(-1)
    if evals.size < 2:
        return 1

    # k=0 is never allowed; also avoid indexing beyond available lambda_{k+1}.
    k_min = max(1, int(min_k))
    k_max = min(max(int(max_k), k_min), evals.size - 2)
    if k_min > k_max:
        return max(1, min(k_min, evals.size - 2))

    gaps = evals[k_min : k_max + 1] - evals[k_min + 1 : k_max + 2]
    if gaps.size == 0:
        return k_min
    return int(np.argmax(gaps) + k_min)
