from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix

from nase.graphs.normalisations import alpha_normalise, row_normalise
from nase.spectral.eigensolvers import top_eigs_symmetric


def diffusion_operator(affinity: np.ndarray | csr_matrix, alpha: float) -> np.ndarray | csr_matrix:
    k_alpha = alpha_normalise(affinity=affinity, alpha=alpha)
    return row_normalise(k_alpha)


def diffusion_map_embedding(
    operator: np.ndarray | csr_matrix, k: int, t: float = 1.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if k <= 0:
        raise ValueError("k must be positive.")
    eigenvalues, eigenvectors = top_eigs_symmetric(operator, n_eigs=k + 1)
    nontrivial_evals = eigenvalues[1:]
    nontrivial_evecs = eigenvectors[:, 1:]
    embedding = nontrivial_evecs * (nontrivial_evals**t)[None, :]
    return embedding, eigenvalues, eigenvectors
