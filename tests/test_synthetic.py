from __future__ import annotations

import numpy as np
import pytest

from nase.data.synthetic import generate_synthetic


@pytest.mark.parametrize(
    ("manifold", "expected_d", "latent_keys"),
    [
        ("circle", 2, {"theta"}),
        ("sphere", 3, {"theta", "phi"}),
        ("swiss_roll", 3, {"t", "h"}),
        ("s_curve", 3, {"t", "h"}),
        ("torus", 3, {"theta", "phi"}),
    ],
)
def test_generate_synthetic_shapes_and_metadata(
    manifold: str, expected_d: int, latent_keys: set[str]
) -> None:
    data = generate_synthetic(manifold=manifold, n=128, ambient_dim=7, r=0.15, seed=11)

    assert data.X_clean.shape == (128, 7)
    assert data.X_noisy.shape == (128, 7)
    assert data.metadata["n"] == 128
    assert data.metadata["d"] == expected_d
    assert data.metadata["D"] == 7
    assert data.metadata["r"] == 0.15
    assert data.metadata["seed"] == 11
    assert data.metadata["manifold"] == manifold

    assert set(data.latent_params) == latent_keys
    for key in latent_keys:
        assert data.latent_params[key].shape == (128, 1)


def test_generate_synthetic_is_deterministic_for_same_seed() -> None:
    first = generate_synthetic(manifold="sphere", n=100, ambient_dim=5, r=0.2, seed=19)
    second = generate_synthetic(manifold="sphere", n=100, ambient_dim=5, r=0.2, seed=19)

    assert np.array_equal(first.X_clean, second.X_clean)
    assert np.array_equal(first.X_noisy, second.X_noisy)
    assert first.metadata == second.metadata
    assert set(first.latent_params) == set(second.latent_params)
    for key in first.latent_params:
        assert np.array_equal(first.latent_params[key], second.latent_params[key])


def test_generate_synthetic_changes_when_seed_changes() -> None:
    first = generate_synthetic(manifold="swiss_roll", n=120, ambient_dim=6, r=0.1, seed=3)
    second = generate_synthetic(manifold="swiss_roll", n=120, ambient_dim=6, r=0.1, seed=4)
    assert not np.array_equal(first.X_clean, second.X_clean)
    assert not np.array_equal(first.X_noisy, second.X_noisy)


def test_noise_zero_leaves_clean_data_unchanged() -> None:
    data = generate_synthetic(manifold="circle", n=80, ambient_dim=4, r=0.0, seed=17)
    assert np.array_equal(data.X_clean, data.X_noisy)


def test_positive_noise_has_reasonable_scale() -> None:
    r = 0.2
    data = generate_synthetic(manifold="sphere", n=400, ambient_dim=8, r=r, seed=31)
    perturbation = data.X_noisy - data.X_clean
    empirical_std = float(np.std(perturbation))

    assert np.linalg.norm(perturbation) > 0.0
    assert 0.5 * r < empirical_std < 1.5 * r


def test_reasonable_radius_bounds_for_rotation_invariant_manifolds() -> None:
    circle = generate_synthetic(manifold="circle", n=200, ambient_dim=2, r=0.0, seed=5)
    sphere = generate_synthetic(manifold="sphere", n=200, ambient_dim=3, r=0.0, seed=6)
    torus = generate_synthetic(
        manifold="torus",
        n=300,
        ambient_dim=3,
        r=0.0,
        seed=7,
        major_radius=2.0,
        minor_radius=0.5,
    )

    circle_norms = np.linalg.norm(circle.X_clean, axis=1)
    sphere_norms = np.linalg.norm(sphere.X_clean, axis=1)
    torus_norms = np.linalg.norm(torus.X_clean, axis=1)

    assert np.allclose(circle_norms, 1.0, atol=1e-10)
    assert np.allclose(sphere_norms, 1.0, atol=1e-10)
    assert np.min(torus_norms) >= 1.5 - 1e-10
    assert np.max(torus_norms) <= 2.5 + 1e-10


def test_invalid_inputs_raise_value_error() -> None:
    with pytest.raises(ValueError, match="Unsupported manifold"):
        generate_synthetic(manifold="unknown", n=10, ambient_dim=3, r=0.1, seed=1)

    with pytest.raises(ValueError, match="D="):
        generate_synthetic(manifold="circle", n=10, ambient_dim=1, r=0.1, seed=1)

    with pytest.raises(ValueError, match="r >= 0"):
        generate_synthetic(manifold="sphere", n=10, ambient_dim=3, r=-0.1, seed=1)
