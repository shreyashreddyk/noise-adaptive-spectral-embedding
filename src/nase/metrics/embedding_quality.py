from __future__ import annotations

import numpy as np
from scipy.sparse.csgraph import shortest_path
from scipy.stats import spearmanr
from sklearn.manifold import trustworthiness
from sklearn.metrics import pairwise_distances
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


def geodesic_consistency_score(
    reference_coords: np.ndarray, embedding: np.ndarray, n_neighbors: int = 10
) -> float:
    """
    Correlate embedding distances with approximate manifold geodesics.

    Geodesics are approximated by shortest-path distances on a kNN graph built
    from clean reference coordinates. This choice uses clean geometry as the
    manifold proxy and avoids noise-induced shortcuts in the reference graph.
    """
    ref = np.asarray(reference_coords, dtype=float)
    emb = np.asarray(embedding, dtype=float)
    if ref.ndim != 2 or emb.ndim != 2:
        raise ValueError("`reference_coords` and `embedding` must be 2D arrays.")
    if ref.shape[0] != emb.shape[0]:
        raise ValueError("`reference_coords` and `embedding` must have same number of samples.")
    n_samples = ref.shape[0]
    if n_samples < 3:
        raise ValueError("Need at least 3 samples to compute geodesic consistency.")

    n_neighbors_eff = int(np.clip(n_neighbors, 1, n_samples - 1))
    knn_graph = NearestNeighbors(n_neighbors=n_neighbors_eff).fit(ref).kneighbors_graph(
        mode="distance"
    )
    geodesic_dists = shortest_path(knn_graph, directed=False, unweighted=False)
    emb_dists = pairwise_distances(emb)

    tri_rows, tri_cols = np.triu_indices(n_samples, k=1)
    geodesic_flat = geodesic_dists[tri_rows, tri_cols]
    emb_flat = emb_dists[tri_rows, tri_cols]
    finite_mask = np.isfinite(geodesic_flat) & np.isfinite(emb_flat)
    if not np.any(finite_mask):
        return 0.0

    corr = spearmanr(geodesic_flat[finite_mask], emb_flat[finite_mask]).correlation
    if corr is None or not np.isfinite(corr):
        return 0.0
    return float(corr)
