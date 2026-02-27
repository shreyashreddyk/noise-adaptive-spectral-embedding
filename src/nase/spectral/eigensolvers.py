from __future__ import annotations

import numpy as np
from scipy.sparse import issparse, spmatrix
from scipy.sparse.linalg import eigsh


def _canonicalise_eigenvector_signs(eigenvectors: np.ndarray) -> np.ndarray:
    fixed = eigenvectors.copy()
    for j in range(fixed.shape[1]):
        col = fixed[:, j]
        pivot = np.argmax(np.abs(col))
        if col[pivot] < 0:
            fixed[:, j] *= -1.0
    return fixed


def _sort_eigenpairs_descending(
    eigenvalues: np.ndarray, eigenvectors: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(eigenvalues)[::-1]
    vals = eigenvalues[order]
    vecs = eigenvectors[:, order]
    return vals, _canonicalise_eigenvector_signs(vecs)


def dense_top_eigs_symmetric(matrix: np.ndarray, n_eigs: int) -> tuple[np.ndarray, np.ndarray]:
    if n_eigs <= 0:
        raise ValueError("n_eigs must be positive.")
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square.")
    if n_eigs > matrix.shape[0]:
        raise ValueError("n_eigs must be <= matrix dimension.")

    vals, vecs = np.linalg.eigh(matrix)
    vals = vals[-n_eigs:]
    vecs = vecs[:, -n_eigs:]
    return _sort_eigenpairs_descending(vals, vecs)


def sparse_top_eigs_symmetric(matrix: spmatrix, n_eigs: int) -> tuple[np.ndarray, np.ndarray]:
    if n_eigs <= 0:
        raise ValueError("n_eigs must be positive.")
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square.")
    if n_eigs >= matrix.shape[0]:
        raise ValueError("For sparse eigsh, n_eigs must be < matrix dimension.")

    vals, vecs = eigsh(matrix, k=n_eigs, which="LA")
    return _sort_eigenpairs_descending(vals, vecs)


def top_eigs_symmetric(matrix: np.ndarray | spmatrix, n_eigs: int) -> tuple[np.ndarray, np.ndarray]:
    if issparse(matrix):
        return sparse_top_eigs_symmetric(matrix=matrix, n_eigs=n_eigs)
    return dense_top_eigs_symmetric(matrix=np.asarray(matrix, dtype=float), n_eigs=n_eigs)
