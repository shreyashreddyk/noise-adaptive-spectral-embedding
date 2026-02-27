from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix

from nase.graphs.normalisations import alpha_normalise, row_normalise
from nase.spectral.eigensolvers import top_eigs_symmetric


def diffusion_operator(affinity: np.ndarray | csr_matrix, alpha: float) -> np.ndarray | csr_matrix:
    k_alpha = alpha_normalise(affinity=affinity, alpha=alpha)
    return row_normalise(k_alpha)


def diffusion_map_embedding(
    operator: np.ndarray | csr_matrix, n_eigs: int, time: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    eigenvalues, eigenvectors = top_eigs_symmetric(operator, n_eigs=n_eigs)
    # Skip the first trivial eigenvector/eigenvalue.
    lambdas = eigenvalues[1:]
    psi = eigenvectors[:, 1:]
    embedding = psi * (lambdas**time)[None, :]
    return embedding, eigenvalues, eigenvectors
