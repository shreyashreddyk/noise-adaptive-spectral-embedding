from __future__ import annotations

import numpy as np


def add_gaussian_noise(
    points: np.ndarray, noise_std: float, rng: np.random.Generator
) -> tuple[np.ndarray, float]:
    """Inject isotropic additive Gaussian noise with known amplitude r=noise_std."""
    noise = rng.normal(loc=0.0, scale=noise_std, size=points.shape)
    return points + noise, noise_std
