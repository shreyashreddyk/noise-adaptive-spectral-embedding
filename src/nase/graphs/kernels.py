from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix


def _dense_local_scales_from_sq_distances(distances_sq: np.ndarray, local_k: int) -> np.ndarray:
    n = distances_sq.shape[0]
    kth = min(max(local_k, 1), max(n - 1, 1))
    sorted_sq = np.sort(distances_sq, axis=1)
    scales_sq = sorted_sq[:, kth]
    scales_sq[scales_sq <= 0.0] = 1.0
    return scales_sq


def _sparse_local_scales_from_sq_distances(distances_sq: csr_matrix) -> np.ndarray:
    n = distances_sq.shape[0]
    scales_sq = np.ones(n, dtype=float)
    for i in range(n):
        start = distances_sq.indptr[i]
        end = distances_sq.indptr[i + 1]
        row = distances_sq.data[start:end]
        if row.size > 0:
            scales_sq[i] = max(float(np.max(row)), 1e-12)
    return scales_sq


def gaussian_kernel_dense(
    distances_sq: np.ndarray,
    epsilon: float,
    *,
    local_scaling: bool,
    local_k: int,
    zero_diagonal: bool,
    symmetric: bool,
) -> np.ndarray:
    if local_scaling:
        scales_sq = _dense_local_scales_from_sq_distances(distances_sq=distances_sq, local_k=local_k)
        denom = epsilon * np.sqrt(np.outer(scales_sq, scales_sq))
        denom[denom <= 0.0] = 1.0
        affinity = np.exp(-distances_sq / denom)
    else:
        affinity = np.exp(-distances_sq / epsilon)

    if symmetric:
        affinity = 0.5 * (affinity + affinity.T)
    if zero_diagonal:
        np.fill_diagonal(affinity, 0.0)
    return np.asarray(affinity, dtype=float)


def gaussian_kernel_sparse(
    distances_sq: csr_matrix,
    epsilon: float,
    *,
    local_scaling: bool,
    zero_diagonal: bool,
    symmetric: bool,
) -> csr_matrix:
    if local_scaling:
        scales_sq = _sparse_local_scales_from_sq_distances(distances_sq=distances_sq)
        row_scales = scales_sq[np.repeat(np.arange(distances_sq.shape[0]), np.diff(distances_sq.indptr))]
        col_scales = scales_sq[distances_sq.indices]
        denom = epsilon * np.sqrt(row_scales * col_scales)
        denom[denom <= 0.0] = 1.0
        data = np.exp(-distances_sq.data / denom)
    else:
        data = np.exp(-distances_sq.data / epsilon)

    affinity = csr_matrix((data, distances_sq.indices, distances_sq.indptr), shape=distances_sq.shape)
    if symmetric:
        affinity = (affinity + affinity.T) * 0.5
    if zero_diagonal:
        affinity.setdiag(0.0)
    affinity.eliminate_zeros()
    return affinity


def gaussian_kernel(
    distances: np.ndarray | csr_matrix,
    epsilon: float,
    *,
    local_scaling: bool = False,
    local_k: int = 7,
    zero_diagonal: bool = True,
    symmetric: bool = False,
) -> np.ndarray | csr_matrix:
    """Build Gaussian affinity from pairwise squared distances.

    Args:
        distances: Dense matrix or sparse kNN graph of pairwise squared distances.
        epsilon: Global bandwidth; larger values produce smoother (less local) affinities.
        local_scaling: If True, applies self-tuning scaling using per-point local scales.
        local_k: Neighbour rank used for dense local scales (ignored for sparse input).
        zero_diagonal: If True, removes self-loops by zeroing diagonal entries.
            This often improves spectral contrast; set False if self-transition mass is
            desired for downstream random-walk dynamics.
        symmetric: If True, symmetrises affinity (`0.5 * (K + K^T)`), useful for
            Laplacian eigenmaps and other symmetric eigendecompositions.
    """
    if epsilon <= 0.0:
        raise ValueError(f"`epsilon` must be > 0, got {epsilon}.")

    if isinstance(distances, csr_matrix):
        return gaussian_kernel_sparse(
            distances_sq=distances,
            epsilon=epsilon,
            local_scaling=local_scaling,
            zero_diagonal=zero_diagonal,
            symmetric=symmetric,
        )
    return gaussian_kernel_dense(
        distances_sq=np.asarray(distances, dtype=float),
        epsilon=epsilon,
        local_scaling=local_scaling,
        local_k=local_k,
        zero_diagonal=zero_diagonal,
        symmetric=symmetric,
    )
