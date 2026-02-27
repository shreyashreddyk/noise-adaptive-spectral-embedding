from __future__ import annotations

import numpy as np

from nase.robust.dss import maybe_doubly_stochastic_scale, sinkhorn_knopp_scale


def test_sinkhorn_knopp_scales_positive_matrix_to_near_doubly_stochastic() -> None:
    rng = np.random.default_rng(123)
    matrix = rng.uniform(0.05, 2.0, size=(40, 40))

    result = sinkhorn_knopp_scale(matrix, max_iter=2000, tol=1e-8)
    assert result.converged
    assert result.max_deviation <= 1e-8
    assert np.allclose(result.scaled_matrix.sum(axis=1), 1.0, atol=2e-7)
    assert np.allclose(result.scaled_matrix.sum(axis=0), 1.0, atol=2e-7)


def test_maybe_doubly_stochastic_scale_is_noop_by_default() -> None:
    rng = np.random.default_rng(321)
    matrix = rng.uniform(0.1, 1.0, size=(12, 12))
    scaled = maybe_doubly_stochastic_scale(matrix, enabled=False)
    assert np.allclose(scaled, matrix, atol=0.0)
