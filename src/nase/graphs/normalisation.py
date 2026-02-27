from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix, diags


def row_normalise_dense(affinity: np.ndarray) -> np.ndarray:
    row_sums = affinity.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    return np.asarray(affinity / row_sums, dtype=float)


def row_normalise_sparse(affinity: csr_matrix) -> csr_matrix:
    row_sums = np.asarray(affinity.sum(axis=1)).ravel()
    row_sums[row_sums == 0.0] = 1.0
    inv = 1.0 / row_sums
    return diags(inv) @ affinity


def alpha_normalise_dense(affinity: np.ndarray, alpha: float) -> np.ndarray:
    degree = affinity.sum(axis=1)
    degree[degree == 0.0] = 1.0
    scale = degree ** (-alpha)
    return np.asarray((scale[:, None] * affinity) * scale[None, :], dtype=float)


def alpha_normalise_sparse(affinity: csr_matrix, alpha: float) -> csr_matrix:
    degree = np.asarray(affinity.sum(axis=1)).ravel()
    degree[degree == 0.0] = 1.0
    scale = degree ** (-alpha)
    d = diags(scale)
    return d @ affinity @ d


def alpha_normalise(affinity: np.ndarray | csr_matrix, alpha: float) -> np.ndarray | csr_matrix:
    if isinstance(affinity, csr_matrix):
        return alpha_normalise_sparse(affinity=affinity, alpha=alpha)
    return alpha_normalise_dense(affinity=affinity, alpha=alpha)


def row_normalise(affinity: np.ndarray | csr_matrix) -> np.ndarray | csr_matrix:
    if isinstance(affinity, csr_matrix):
        return row_normalise_sparse(affinity=affinity)
    return row_normalise_dense(affinity=affinity)
