from __future__ import annotations

import numpy as np

from nase.metrics.embedding_quality import geodesic_consistency_score as _geodesic_consistency_score_impl


def geodesic_consistency_score(
    reference_coords: np.ndarray, embedding: np.ndarray, n_neighbors: int = 10
) -> float:
    return _geodesic_consistency_score_impl(
        reference_coords=reference_coords, embedding=embedding, n_neighbors=n_neighbors
    )
