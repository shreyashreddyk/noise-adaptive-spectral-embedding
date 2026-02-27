from __future__ import annotations

import numpy as np

from nase.graphs.distances import choose_distance_backend
from nase.graphs.kernels import gaussian_kernel
from nase.graphs.normalisations import markov_matrix


def test_dense_graph_pipeline() -> None:
    x = np.array([[0.0], [1.0], [2.0]])
    distances = choose_distance_backend(x, use_knn=False, knn_k=2, sparse_threshold_n=10)
    affinity = gaussian_kernel(distances, epsilon=1.0, symmetric=True, zero_diagonal=True)
    p = markov_matrix(affinity)
    assert p.shape == (3, 3)


def test_sparse_graph_pipeline() -> None:
    x = np.random.default_rng(0).normal(size=(20, 2))
    distances = choose_distance_backend(x, use_knn=True, knn_k=4, sparse_threshold_n=100)
    affinity = gaussian_kernel(distances, epsilon=1.0, zero_diagonal=True)
    p = markov_matrix(affinity)
    assert p.shape == (20, 20)


def test_gaussian_kernel_symmetric_mode_is_symmetric() -> None:
    x = np.array([[0.0, 0.0], [1.0, 0.0], [0.2, 0.7], [1.4, 0.9]])
    distances = choose_distance_backend(x, use_knn=False, knn_k=2, sparse_threshold_n=100)
    affinity = gaussian_kernel(distances, epsilon=0.3, symmetric=True)
    assert np.allclose(affinity, affinity.T, atol=1e-12)


def test_gaussian_kernel_zero_diagonal() -> None:
    x = np.array([[0.0], [1.0], [3.0]])
    distances = choose_distance_backend(x, use_knn=False, knn_k=2, sparse_threshold_n=100)
    affinity = gaussian_kernel(distances, epsilon=1.0, zero_diagonal=True)
    assert np.allclose(np.diag(affinity), 0.0, atol=1e-12)


def test_markov_matrix_rows_sum_to_one() -> None:
    x = np.array([[0.0], [0.8], [1.9], [3.0]])
    distances = choose_distance_backend(x, use_knn=False, knn_k=2, sparse_threshold_n=100)
    affinity = gaussian_kernel(distances, epsilon=0.7, symmetric=True)
    p = markov_matrix(affinity, alpha=0.5)
    row_sums = p.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-10)
