from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix


def gaussian_kernel_dense(distances: np.ndarray, epsilon: float) -> np.ndarray:
    affinity = np.exp(-(distances**2) / epsilon)
    np.fill_diagonal(affinity, 0.0)
    return affinity


def gaussian_kernel_sparse(distances: csr_matrix, epsilon: float) -> csr_matrix:
    data = np.exp(-(distances.data**2) / epsilon)
    affinity = csr_matrix((data, distances.indices, distances.indptr), shape=distances.shape)
    affinity.setdiag(0.0)
    affinity.eliminate_zeros()
    return affinity


def gaussian_kernel(distances: np.ndarray | csr_matrix, epsilon: float) -> np.ndarray | csr_matrix:
    if isinstance(distances, csr_matrix):
        return gaussian_kernel_sparse(distances=distances, epsilon=epsilon)
    return gaussian_kernel_dense(distances=distances, epsilon=epsilon)
