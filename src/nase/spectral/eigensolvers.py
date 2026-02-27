from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh


def _fix_eigenvector_signs(eigenvectors: np.ndarray) -> np.ndarray:
    fixed = eigenvectors.copy()
    for j in range(fixed.shape[1]):
        col = fixed[:, j]
        pivot = np.argmax(np.abs(col))
        if col[pivot] < 0:
            fixed[:, j] *= -1.0
    return fixed


def top_eigs_symmetric(
    matrix: np.ndarray | csr_matrix, n_eigs: int
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(matrix, csr_matrix):
        vals, vecs = eigsh(matrix, k=n_eigs, which="LA")
    else:
        vals, vecs = np.linalg.eigh(matrix)
        vals = vals[-n_eigs:]
        vecs = vecs[:, -n_eigs:]

    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    vecs = _fix_eigenvector_signs(vecs)
    return vals, vecs
