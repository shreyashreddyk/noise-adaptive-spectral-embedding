from __future__ import annotations

import numpy as np

from nase.estimators.intrinsic_dimension import levina_bickel_mle_intrinsic_dimension


def test_levina_bickel_recovers_approximately_1d_curve() -> None:
    rng = np.random.default_rng(7)
    n = 900
    t = rng.uniform(-1.0, 1.0, size=n)
    points = np.column_stack(
        [
            t,
            t**2,
            np.sin(2.0 * np.pi * t),
        ]
    )
    points += 0.005 * rng.normal(size=points.shape)

    estimate = levina_bickel_mle_intrinsic_dimension(points, k=10)
    assert 0.7 <= estimate <= 1.7


def test_levina_bickel_recovers_approximately_2d_plane_in_ambient_5d() -> None:
    rng = np.random.default_rng(9)
    n = 1000
    latent = rng.uniform(-1.0, 1.0, size=(n, 2))
    basis = rng.normal(size=(2, 5))
    basis /= np.linalg.norm(basis, axis=1, keepdims=True)
    points = latent @ basis
    points += 0.01 * rng.normal(size=points.shape)

    estimate, local = levina_bickel_mle_intrinsic_dimension(points, k=14, return_pointwise=True)
    assert local.shape == (n,)
    assert np.isfinite(local).sum() > int(0.9 * n)
    assert 1.5 <= estimate <= 2.7
