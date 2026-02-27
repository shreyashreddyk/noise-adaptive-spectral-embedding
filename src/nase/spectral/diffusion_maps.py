from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix

from nase.spectral.embedding import diffusion_map_embedding as _diffusion_map_embedding
from nase.spectral.embedding import diffusion_operator


def diffusion_map_embedding(
    operator: np.ndarray | csr_matrix, n_eigs: int, time: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if n_eigs <= 1:
        raise ValueError("n_eigs must be at least 2 to exclude the trivial component.")
    return _diffusion_map_embedding(operator=operator, k=n_eigs - 1, t=time)
