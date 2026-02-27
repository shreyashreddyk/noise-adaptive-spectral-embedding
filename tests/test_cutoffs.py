from __future__ import annotations

import numpy as np

from nase.cutoffs.bandwidth_stability import select_k_bandwidth_stability
from nase.cutoffs.eigengap import select_k_eigengap


def test_eigengap_selects_expected_index() -> None:
    evals = np.array([1.0, 0.9, 0.88, 0.4, 0.39])
    k = select_k_eigengap(evals, min_k=1, max_k=4)
    assert k == 2


def test_bandwidth_stability_returns_valid_k() -> None:
    rng = np.random.default_rng(0)
    v1 = rng.normal(size=(30, 8))
    v2 = v1.copy()
    v2[:, 6:] = rng.normal(size=(30, 2))
    result = select_k_bandwidth_stability([v1, v2], min_k=1, max_k=6, threshold=0.8)
    assert 1 <= result.k_star <= 6
