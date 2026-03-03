from __future__ import annotations

import math


def select_k_from_noise(
    r: float,
    c_constant: float = 1.0,
    min_k: int = 1,
    max_k: int = 20,
) -> int:
    """Compute spectral truncation cutoff from noise amplitude.

    Implements the formula k* = floor(C / r^2) from the noisy Laplacian
    theory, where C is a geometry-dependent constant and r is the noise
    standard deviation.

    When r is small, many eigenvectors carry signal and k* is large.
    When r is large, noise corrupts most modes and k* shrinks. The
    relationship is inverse-quadratic: doubling r reduces k* by a factor
    of four.

    Args:
        r: Noise standard deviation (must be > 0).
        c_constant: Geometry constant C. Depends on manifold curvature and
            reach. Must be calibrated empirically or set to a default.
        min_k: Minimum allowed cutoff.
        max_k: Maximum allowed cutoff.

    Returns:
        Integer cutoff k* clipped to [min_k, max_k].
    """
    if r <= 0:
        return max_k
    k_star = math.floor(c_constant / (r * r))
    return max(min_k, min(k_star, max_k))
