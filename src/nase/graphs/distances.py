from __future__ import annotations

import hashlib
import warnings

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import NearestNeighbors

_DENSE_DISTANCE_CACHE: dict[tuple[int, int, str], np.ndarray] = {}
_DENSE_CACHE_MAX_SIZE = 8


def _dense_cache_key(points: np.ndarray) -> tuple[int, int, str]:
    contiguous = np.ascontiguousarray(points)
    digest = hashlib.blake2b(contiguous.view(np.uint8), digest_size=16).hexdigest()
    return contiguous.shape[0], contiguous.shape[1], digest


def pairwise_squared_distance_matrix(points: np.ndarray, use_cache: bool = True) -> np.ndarray:
    """Return pairwise squared distances for dense graph construction.

    Dense all-pairs distances are accurate and convenient for small `n`, but require
    O(n^2) memory and O(n^2 d) compute, so this path is intended for smaller datasets.
    When `use_cache=True`, this function caches recent results for repeated calls with
    identical point clouds (common during epsilon sweeps).
    """
    if points.ndim != 2:
        raise ValueError(f"`points` must be 2D, got shape={points.shape}.")

    key = _dense_cache_key(points)
    if use_cache and key in _DENSE_DISTANCE_CACHE:
        return _DENSE_DISTANCE_CACHE[key].copy()

    distances_sq = np.asarray(euclidean_distances(points, squared=True), dtype=float)
    np.maximum(distances_sq, 0.0, out=distances_sq)

    if use_cache:
        if len(_DENSE_DISTANCE_CACHE) >= _DENSE_CACHE_MAX_SIZE:
            oldest_key = next(iter(_DENSE_DISTANCE_CACHE))
            _DENSE_DISTANCE_CACHE.pop(oldest_key, None)
        _DENSE_DISTANCE_CACHE[key] = distances_sq.copy()
    return distances_sq


def pairwise_distance_matrix(points: np.ndarray) -> np.ndarray:
    """Backwards-compatible alias for squared dense distances."""
    return pairwise_squared_distance_matrix(points=points, use_cache=True)


def knn_squared_distance_graph(points: np.ndarray, k: int) -> csr_matrix:
    """Return sparse directed kNN graph with squared edge distances.

    Sparse kNN mode scales better than dense mode for larger `n` because it stores only
    O(nk) edges. The trade-off is that non-neighbour distances are discarded.
    """
    if points.ndim != 2:
        raise ValueError(f"`points` must be 2D, got shape={points.shape}.")
    if k < 1:
        raise ValueError(f"`k` must be >= 1, got {k}.")

    n = points.shape[0]
    if n <= 1:
        return csr_matrix((n, n), dtype=float)
    effective_k = min(k, max(n - 1, 1))
    nn = NearestNeighbors(n_neighbors=effective_k + 1, metric="euclidean")
    nn.fit(points)
    distances, indices = nn.kneighbors(points, return_distance=True)
    distances_sq = distances[:, 1:] ** 2
    rows = np.repeat(np.arange(n), effective_k)
    cols = indices[:, 1:].reshape(-1)
    vals = distances_sq.reshape(-1)
    return csr_matrix((vals, (rows, cols)), shape=(n, n))


def knn_distance_graph(points: np.ndarray, k: int) -> csr_matrix:
    """Backwards-compatible alias for squared kNN distances."""
    return knn_squared_distance_graph(points=points, k=k)


def choose_distance_backend(
    points: np.ndarray, use_knn: bool, knn_k: int, sparse_threshold_n: int
) -> np.ndarray | csr_matrix:
    """Select dense cached squared distances or sparse kNN squared distances.

    - Dense mode (`n < sparse_threshold_n` and `use_knn=False`) computes all pairwise
      squared distances and caches them. It is exact but O(n^2) memory.
    - Sparse mode (forced by `use_knn=True` or large `n`) builds a kNN graph using
      scikit-learn `NearestNeighbors`. It is memory efficient but approximate globally.
    """
    n_samples = points.shape[0]
    if use_knn or n_samples >= sparse_threshold_n:
        warnings.warn(
            "Using sparse kNN graph for scalability. This path is not used for noise-amplitude "
            "estimation.",
            UserWarning,
            stacklevel=2,
        )
        return knn_squared_distance_graph(points=points, k=knn_k)
    return pairwise_squared_distance_matrix(points=points, use_cache=True)
