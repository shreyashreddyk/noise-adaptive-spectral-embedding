from __future__ import annotations

import warnings

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors


def pairwise_distance_matrix(points: np.ndarray) -> np.ndarray:
    return np.asarray(pairwise_distances(points, metric="euclidean"), dtype=float)


def knn_distance_graph(points: np.ndarray, k: int) -> csr_matrix:
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nn.fit(points)
    distances, indices = nn.kneighbors(points, return_distance=True)
    n = points.shape[0]
    rows = np.repeat(np.arange(n), k)
    cols = indices[:, 1:].reshape(-1)
    vals = distances[:, 1:].reshape(-1)
    return csr_matrix((vals, (rows, cols)), shape=(n, n))


def choose_distance_backend(
    points: np.ndarray, use_knn: bool, knn_k: int, sparse_threshold_n: int
) -> np.ndarray | csr_matrix:
    n_samples = points.shape[0]
    if use_knn or n_samples >= sparse_threshold_n:
        warnings.warn(
            "Using sparse kNN graph for scalability. This path is not used for noise-amplitude "
            "estimation.",
            UserWarning,
            stacklevel=2,
        )
        return knn_distance_graph(points=points, k=knn_k)
    return pairwise_distance_matrix(points)
