from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class SyntheticManifoldData:
    """Container for clean/noisy manifold samples and generation metadata."""

    X_clean: np.ndarray
    X_noisy: np.ndarray
    latent_params: dict[str, np.ndarray]
    metadata: dict[str, int | float | str]


def _sample_circle(
    n: int, rng: np.random.Generator
) -> tuple[np.ndarray, dict[str, np.ndarray], int]:
    theta = rng.uniform(0.0, 2.0 * np.pi, size=n)
    x = np.column_stack([np.cos(theta), np.sin(theta)])
    latent = {"theta": theta[:, None]}
    return x, latent, 2


def _sample_sphere(
    n: int, rng: np.random.Generator
) -> tuple[np.ndarray, dict[str, np.ndarray], int]:
    theta = rng.uniform(0.0, 2.0 * np.pi, size=n)
    u = rng.uniform(-1.0, 1.0, size=n)
    phi = np.arccos(u)
    radial = np.sqrt(1.0 - u**2)
    x = np.column_stack([radial * np.cos(theta), radial * np.sin(theta), u])
    latent = {"theta": theta[:, None], "phi": phi[:, None]}
    return x, latent, 3


def _sample_swiss_roll(
    n: int,
    rng: np.random.Generator,
    *,
    t_min: float = 1.5 * np.pi,
    t_max: float = 4.5 * np.pi,
    height: float = 2.0,
) -> tuple[np.ndarray, dict[str, np.ndarray], int]:
    if t_max <= t_min:
        raise ValueError("Expected t_max > t_min for swiss roll generation.")
    if height <= 0.0:
        raise ValueError("Expected swiss roll height > 0.")
    t = rng.uniform(t_min, t_max, size=n)
    h = rng.uniform(0.0, height, size=n)
    x = np.column_stack([t * np.cos(t), h, t * np.sin(t)])
    latent = {"t": t[:, None], "h": h[:, None]}
    return x, latent, 3


def _sample_s_curve(
    n: int,
    rng: np.random.Generator,
    *,
    height: float = 2.0,
) -> tuple[np.ndarray, dict[str, np.ndarray], int]:
    if height <= 0.0:
        raise ValueError("Expected s-curve height > 0.")
    t = rng.uniform(-np.pi, np.pi, size=n)
    h = rng.uniform(0.0, height, size=n)
    x = np.column_stack([np.sin(t), h, np.sign(t) * (np.cos(t) - 1.0)])
    latent = {"t": t[:, None], "h": h[:, None]}
    return x, latent, 3


def _sample_torus(
    n: int,
    rng: np.random.Generator,
    *,
    major_radius: float = 2.0,
    minor_radius: float = 0.5,
) -> tuple[np.ndarray, dict[str, np.ndarray], int]:
    if major_radius <= 0.0 or minor_radius <= 0.0:
        raise ValueError("Expected torus radii to be positive.")
    if major_radius <= minor_radius:
        raise ValueError(
            "Expected major_radius > minor_radius for a non-self-intersecting torus."
        )
    theta = rng.uniform(0.0, 2.0 * np.pi, size=n)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=n)
    ring = major_radius + minor_radius * np.cos(phi)
    x = np.column_stack(
        [ring * np.cos(theta), ring * np.sin(theta), minor_radius * np.sin(phi)]
    )
    latent = {"theta": theta[:, None], "phi": phi[:, None]}
    return x, latent, 3


def _random_orthogonal(ambient_dim: int, rng: np.random.Generator) -> np.ndarray:
    """Sample a deterministic orthogonal matrix from an RNG state."""
    gaussian = rng.normal(size=(ambient_dim, ambient_dim))
    q_mat, r_mat = np.linalg.qr(gaussian)
    signs = np.sign(np.diag(r_mat))
    signs[signs == 0.0] = 1.0
    return q_mat * signs


def _embed_in_ambient(x: np.ndarray, ambient_dim: int, rng: np.random.Generator) -> np.ndarray:
    n, d = x.shape
    if ambient_dim < d:
        raise ValueError(
            "Ambient dimension D="
            f"{ambient_dim} must satisfy D >= intrinsic embedding dimension d={d}."
        )
    if ambient_dim == d:
        padded = x
    else:
        padded = np.hstack([x, np.zeros((n, ambient_dim - d))])
    return padded @ _random_orthogonal(ambient_dim, rng)


def generate_synthetic(
    manifold: str,
    n: int,
    ambient_dim: int,
    r: float,
    seed: int,
    **kwargs: float,
) -> SyntheticManifoldData:
    """Generate clean/noisy synthetic manifold data with deterministic randomness."""
    if n <= 0:
        raise ValueError("Expected n > 0.")
    if r < 0.0:
        raise ValueError("Expected noise amplitude r >= 0.")

    rng = np.random.default_rng(seed)
    name = manifold.lower()

    if name in {"circle", "s1"}:
        x_native, latent, d = _sample_circle(n=n, rng=rng)
    elif name in {"sphere", "s2"}:
        x_native, latent, d = _sample_sphere(n=n, rng=rng)
    elif name == "swiss_roll":
        x_native, latent, d = _sample_swiss_roll(n=n, rng=rng, **kwargs)
    elif name in {"s_curve", "scurve"}:
        x_native, latent, d = _sample_s_curve(n=n, rng=rng, **kwargs)
    elif name == "torus":
        x_native, latent, d = _sample_torus(n=n, rng=rng, **kwargs)
    else:
        valid = ", ".join(["circle", "s_curve", "sphere", "swiss_roll", "torus"])
        raise ValueError(f"Unsupported manifold '{manifold}'. Valid options: {valid}.")

    x_clean = _embed_in_ambient(x_native, ambient_dim=ambient_dim, rng=rng)
    x_noisy = x_clean + rng.normal(loc=0.0, scale=r, size=(n, ambient_dim))

    metadata: dict[str, int | float | str] = {
        "n": n,
        "d": d,
        "D": ambient_dim,
        "r": float(r),
        "seed": int(seed),
        "manifold": name,
    }
    return SyntheticManifoldData(
        X_clean=x_clean,
        X_noisy=x_noisy,
        latent_params=latent,
        metadata=metadata,
    )
