from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix, diags, eye


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


def markov_matrix(affinity: np.ndarray | csr_matrix, alpha: float = 0.5) -> np.ndarray | csr_matrix:
    """Return diffusion-maps Markov operator.

    This applies anisotropic alpha-normalisation followed by row-normalisation so rows
    sum to one. For diffusion maps, `alpha=0.5` is a common practical default.
    """
    return row_normalise(alpha_normalise(affinity=affinity, alpha=alpha))


def laplacian_eigenmaps_matrices(
    affinity: np.ndarray | csr_matrix, normalized: bool = True
) -> tuple[np.ndarray | csr_matrix, np.ndarray | csr_matrix]:
    """Construct Laplacian eigenmaps matrices.

    Returns:
        (L, D), where D is degree diagonal and L is either:
        - normalized Laplacian: I - D^{-1/2} W D^{-1/2} (if `normalized=True`)
        - unnormalized Laplacian: D - W (if `normalized=False`)
    """
    if isinstance(affinity, csr_matrix):
        degree = np.asarray(affinity.sum(axis=1)).ravel()
        d = diags(degree)
        if normalized:
            inv_sqrt = np.zeros_like(degree, dtype=float)
            positive = degree > 0.0
            inv_sqrt[positive] = 1.0 / np.sqrt(degree[positive])
            d_inv_sqrt = diags(inv_sqrt)
            l = eye(affinity.shape[0], format="csr") - d_inv_sqrt @ affinity @ d_inv_sqrt
            return l.tocsr(), d.tocsr()
        return (d - affinity).tocsr(), d.tocsr()

    w = np.asarray(affinity, dtype=float)
    degree = w.sum(axis=1)
    d = np.diag(degree)
    if normalized:
        inv_sqrt = np.zeros_like(degree, dtype=float)
        positive = degree > 0.0
        inv_sqrt[positive] = 1.0 / np.sqrt(degree[positive])
        d_inv_sqrt = np.diag(inv_sqrt)
        l = np.eye(w.shape[0], dtype=float) - d_inv_sqrt @ w @ d_inv_sqrt
        return l, d
    return d - w, d
