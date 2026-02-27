from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import make_s_curve, make_swiss_roll


@dataclass(slots=True)
class ManifoldData:
    points: np.ndarray
    intrinsic_coords: np.ndarray
    name: str


def make_circle(n_samples: int, ambient_dim: int, rng: np.random.Generator) -> ManifoldData:
    theta = rng.uniform(0.0, 2.0 * np.pi, size=n_samples)
    x = np.cos(theta)
    y = np.sin(theta)
    base = np.column_stack([x, y])
    if ambient_dim > 2:
        pad = np.zeros((n_samples, ambient_dim - 2))
        points = np.hstack([base, pad])
    else:
        points = base[:, :ambient_dim]
    return ManifoldData(points=points, intrinsic_coords=theta[:, None], name="circle")


def make_swiss_roll_data(
    n_samples: int, ambient_dim: int, rng: np.random.Generator
) -> ManifoldData:
    data, t = make_swiss_roll(n_samples=n_samples, random_state=int(rng.integers(0, 10_000)))
    if ambient_dim > 3:
        pad = np.zeros((n_samples, ambient_dim - 3))
        data = np.hstack([data, pad])
    elif ambient_dim < 3:
        data = data[:, :ambient_dim]
    return ManifoldData(points=data, intrinsic_coords=t[:, None], name="swiss_roll")


def make_s_curve_data(n_samples: int, ambient_dim: int, rng: np.random.Generator) -> ManifoldData:
    data, t = make_s_curve(n_samples=n_samples, random_state=int(rng.integers(0, 10_000)))
    if ambient_dim > 3:
        pad = np.zeros((n_samples, ambient_dim - 3))
        data = np.hstack([data, pad])
    elif ambient_dim < 3:
        data = data[:, :ambient_dim]
    return ManifoldData(points=data, intrinsic_coords=t[:, None], name="s_curve")


def generate_manifold(
    manifold: str, n_samples: int, ambient_dim: int, rng: np.random.Generator
) -> ManifoldData:
    generators = {
        "circle": make_circle,
        "swiss_roll": make_swiss_roll_data,
        "s_curve": make_s_curve_data,
    }
    if manifold not in generators:
        valid = ", ".join(sorted(generators))
        raise ValueError(f"Unsupported manifold '{manifold}'. Valid options: {valid}.")
    return generators[manifold](n_samples=n_samples, ambient_dim=ambient_dim, rng=rng)
