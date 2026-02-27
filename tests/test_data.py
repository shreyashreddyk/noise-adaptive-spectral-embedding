from __future__ import annotations

from nase.data.manifolds import generate_manifold
from nase.data.noise import add_gaussian_noise
from nase.utils import set_global_seed


def test_generate_manifold_shapes() -> None:
    rng = set_global_seed(1)
    data = generate_manifold("swiss_roll", n_samples=100, ambient_dim=3, rng=rng)
    assert data.points.shape == (100, 3)
    assert data.intrinsic_coords.shape[0] == 100


def test_add_noise_returns_known_r() -> None:
    rng = set_global_seed(2)
    data = generate_manifold("circle", n_samples=50, ambient_dim=4, rng=rng)
    noisy, r = add_gaussian_noise(data.points, noise_std=0.15, rng=rng)
    assert noisy.shape == data.points.shape
    assert r == 0.15
