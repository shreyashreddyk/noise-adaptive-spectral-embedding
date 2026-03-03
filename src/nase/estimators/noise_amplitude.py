from __future__ import annotations

import numpy as np


def estimate_noise_amplitude_simple(
    points: np.ndarray,
    *,
    k: int = 2,
    min_value: float = 1e-12,
) -> float:
    """Estimate noise standard deviation from kNN distances.

    For data X = M + noise with noise ~ N(0, r^2 I_D), the expected squared
    distance between nearby points is approximately:

        E[||x_i - x_j||^2] = ||m_i - m_j||^2 + 2 D r^2

    At small scales (nearest neighbors), the manifold contribution is small
    relative to the noise term. The simple estimator uses:

        r_hat^2 = median(d_k^2) / (2D)

    This overestimates r because it includes the manifold contribution, but
    provides a reasonable upper bound when noise dominates at the NN scale.

    Args:
        points: Data array (n_samples, n_features).
        k: Neighbourhood rank for distance estimation. Smaller k gives
           estimates more dominated by noise (less manifold bias) but higher
           variance. k=1 or 2 is typical.
        min_value: Floor on distances for numerical stability.

    Returns:
        Estimated noise standard deviation r_hat.
    """
    x = np.asarray(points, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"`points` must be 2D, got shape {x.shape}.")
    n, ambient_dim = x.shape
    if n < k + 1:
        raise ValueError(f"Need at least {k + 1} samples, got {n}.")
    if k < 1:
        raise ValueError(f"`k` must be >= 1, got {k}.")

    sq_norm = np.sum(x * x, axis=1, keepdims=True)
    sq_dist = sq_norm + sq_norm.T - 2.0 * (x @ x.T)
    np.maximum(sq_dist, 0.0, out=sq_dist)
    np.fill_diagonal(sq_dist, np.inf)

    kth_sq = np.partition(sq_dist, kth=k - 1, axis=1)[:, k - 1]
    kth_sq = np.clip(kth_sq, min_value, None)

    r_hat_sq = float(np.median(kth_sq)) / (2.0 * ambient_dim)
    return float(np.sqrt(max(r_hat_sq, 0.0)))


def estimate_noise_amplitude_twoscale(
    points: np.ndarray,
    *,
    k1: int = 1,
    k2: int = 5,
    min_value: float = 1e-12,
) -> float:
    """Estimate noise std using distances at two neighborhood scales.

    The noise contribution to E[d_k^2] is approximately constant (2Dr^2)
    regardless of k, while the manifold contribution grows with k (farther
    neighbors sample more of the manifold). By comparing mean squared
    distances at two scales, we can partially subtract out the manifold
    contribution:

        noise_sq ≈ (k2 * mean(d_k1^2) - k1 * mean(d_k2^2)) / ((k2 - k1) * 2D)

    This is heuristic and assumes the manifold contribution scales roughly
    linearly with k in the local regime. It tends to reduce the upward bias
    of the simple estimator.

    Args:
        points: Data array (n_samples, n_features).
        k1: Smaller neighbourhood rank.
        k2: Larger neighbourhood rank. Must be > k1.
        min_value: Floor on distances for numerical stability.

    Returns:
        Estimated noise standard deviation r_hat.
    """
    x = np.asarray(points, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"`points` must be 2D, got shape {x.shape}.")
    n, ambient_dim = x.shape
    if n < k2 + 1:
        raise ValueError(f"Need at least {k2 + 1} samples, got {n}.")
    if k1 < 1 or k2 <= k1:
        raise ValueError(f"Need 1 <= k1 < k2, got k1={k1}, k2={k2}.")

    sq_norm = np.sum(x * x, axis=1, keepdims=True)
    sq_dist = sq_norm + sq_norm.T - 2.0 * (x @ x.T)
    np.maximum(sq_dist, 0.0, out=sq_dist)
    np.fill_diagonal(sq_dist, np.inf)

    sorted_sq = np.sort(sq_dist, axis=1)
    d_k1_sq = np.clip(sorted_sq[:, k1 - 1], min_value, None)
    d_k2_sq = np.clip(sorted_sq[:, k2 - 1], min_value, None)

    mean_dk1 = float(np.mean(d_k1_sq))
    mean_dk2 = float(np.mean(d_k2_sq))

    numerator = k2 * mean_dk1 - k1 * mean_dk2
    denominator = (k2 - k1) * 2.0 * ambient_dim
    r_hat_sq = numerator / denominator

    if r_hat_sq < 0:
        return estimate_noise_amplitude_simple(points, k=k1, min_value=min_value)

    return float(np.sqrt(r_hat_sq))
