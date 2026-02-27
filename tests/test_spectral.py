from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix

from nase.spectral.eigensolvers import top_eigs_symmetric
from nase.spectral.embedding import diffusion_map_embedding, diffusion_operator


def _tiny_synthetic_affinity(seed: int = 13, n: int = 8) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 2))
    sq_dists = np.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=-1)
    affinity = np.exp(-sq_dists / 0.7)
    np.fill_diagonal(affinity, 0.0)
    return affinity


def test_eigenvalues_sorted_descending_for_dense_and_sparse() -> None:
    affinity = _tiny_synthetic_affinity(seed=5, n=7)
    dense_vals, _ = top_eigs_symmetric(affinity, n_eigs=4)
    sparse_vals, _ = top_eigs_symmetric(csr_matrix(affinity), n_eigs=4)

    assert np.all(np.diff(dense_vals) <= 0.0)
    assert np.all(np.diff(sparse_vals) <= 0.0)
    assert np.allclose(dense_vals, sparse_vals, atol=1e-10)


def test_diffusion_embedding_shape_is_n_by_k() -> None:
    affinity = _tiny_synthetic_affinity(seed=9, n=10)
    op = diffusion_operator(affinity, alpha=0.5)
    embedding, eigenvalues, eigenvectors = diffusion_map_embedding(op, k=3, t=2)

    assert embedding.shape == (10, 3)
    assert eigenvalues.shape == (4,)
    assert eigenvectors.shape == (10, 4)


def test_diffusion_embedding_is_deterministic_for_fixed_seed() -> None:
    op_first = diffusion_operator(_tiny_synthetic_affinity(seed=21, n=9), alpha=0.5)
    op_second = diffusion_operator(_tiny_synthetic_affinity(seed=21, n=9), alpha=0.5)

    emb_first, evals_first, evecs_first = diffusion_map_embedding(op_first, k=2, t=1.0)
    emb_second, evals_second, evecs_second = diffusion_map_embedding(op_second, k=2, t=1.0)

    assert np.allclose(evals_first, evals_second, atol=1e-12)
    assert np.allclose(evecs_first, evecs_second, atol=1e-12)
    assert np.allclose(emb_first, emb_second, atol=1e-12)
