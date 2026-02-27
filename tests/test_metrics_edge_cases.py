from __future__ import annotations

import numpy as np

from nase.metrics.embedding_quality import continuity_score, trustworthiness_score
from nase.metrics.geodesic import geodesic_consistency_score


def test_embedding_quality_scores_stay_in_range() -> None:
    x = np.random.default_rng(1).normal(size=(30, 3))
    y = x[:, :2]
    trust = trustworthiness_score(x, y, n_neighbors=5)
    cont = continuity_score(x, y, n_neighbors=5)
    assert 0.0 <= trust <= 1.0
    assert 0.0 <= cont <= 1.0


def test_geodesic_consistency_is_finite() -> None:
    coords = np.random.default_rng(2).normal(size=(20, 2))
    score = geodesic_consistency_score(reference_coords=coords, embedding=coords)
    assert np.isfinite(score)
