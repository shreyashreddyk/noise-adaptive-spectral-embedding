from __future__ import annotations

import numpy as np
from sklearn.manifold import trustworthiness
from sklearn.neighbors import NearestNeighbors


def trustworthiness_score(
    high_dim: np.ndarray, low_dim: np.ndarray, n_neighbors: int = 10
) -> float:
    return float(trustworthiness(high_dim, low_dim, n_neighbors=n_neighbors))


def continuity_score(high_dim: np.ndarray, low_dim: np.ndarray, n_neighbors: int = 10) -> float:
    nn_high = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(high_dim)
    nn_low = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(low_dim)
    idx_high = nn_high.kneighbors(return_distance=False)[:, 1:]
    idx_low = nn_low.kneighbors(return_distance=False)[:, 1:]

    overlap = [
        len(set(a).intersection(set(b))) / n_neighbors
        for a, b in zip(idx_high, idx_low, strict=True)
    ]
    return float(np.mean(overlap))
