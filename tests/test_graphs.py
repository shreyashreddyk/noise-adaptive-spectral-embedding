from __future__ import annotations

import numpy as np

from nase.graphs.distances import choose_distance_backend
from nase.graphs.kernels import gaussian_kernel
from nase.graphs.normalisation import row_normalise


def test_dense_graph_pipeline() -> None:
    x = np.array([[0.0], [1.0], [2.0]])
    distances = choose_distance_backend(x, use_knn=False, knn_k=2, sparse_threshold_n=10)
    affinity = gaussian_kernel(distances, epsilon=1.0)
    p = row_normalise(affinity)
    assert p.shape == (3, 3)


def test_sparse_graph_pipeline() -> None:
    x = np.random.default_rng(0).normal(size=(20, 2))
    distances = choose_distance_backend(x, use_knn=True, knn_k=4, sparse_threshold_n=100)
    affinity = gaussian_kernel(distances, epsilon=1.0)
    p = row_normalise(affinity)
    assert p.shape == (20, 20)
