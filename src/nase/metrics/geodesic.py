from __future__ import annotations

import numpy as np
from sklearn.metrics import pairwise_distances, r2_score


def geodesic_consistency_score(reference_coords: np.ndarray, embedding: np.ndarray) -> float:
    d_ref = pairwise_distances(reference_coords)
    d_emb = pairwise_distances(embedding)
    return float(r2_score(d_ref.ravel(), d_emb.ravel()))
