from __future__ import annotations

import numpy as np


def levina_bickel_mle_intrinsic_dimension(
    points: np.ndarray,
    *,
    k: int = 10,
    min_value: float = 1e-12,
    return_pointwise: bool = False,
) -> float | tuple[float, np.ndarray]:
    """Estimate intrinsic dimension with the Levina-Bickel kNN MLE.

    For each sample ``x_i``, let ``T_j(x_i)`` be the distance to its j-th nearest
    neighbour (excluding itself), with ``j = 1, ..., k`` and ``k >= 2``.
    The local estimate is:

    ``m_i = [ (1/(k-1)) * sum_{j=1}^{k-1} log(T_k(x_i) / T_j(x_i)) ]^{-1}``

    The global estimate is the mean of valid ``m_i`` values.

    Args:
        points: Data array of shape ``(n_samples, n_features)``.
        k: Neighbourhood size for the estimator. Must satisfy ``2 <= k < n_samples``.
        min_value: Positive floor applied to neighbour distances for numerical stability.
        return_pointwise: If True, return ``(global_estimate, local_estimates)``.

    Returns:
        Global intrinsic-dimension estimate, or global + per-point estimates.
    """
    x = np.asarray(points, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"`points` must be 2D, got shape {x.shape}.")
    n_samples = x.shape[0]
    if n_samples < 3:
        raise ValueError("Need at least 3 samples for intrinsic-dimension estimation.")
    if not (2 <= k < n_samples):
        raise ValueError(f"`k` must satisfy 2 <= k < n_samples, got k={k}, n={n_samples}.")
    if min_value <= 0.0:
        raise ValueError(f"`min_value` must be > 0, got {min_value}.")

    sq_norm = np.sum(x * x, axis=1, keepdims=True)
    sq_dist = sq_norm + sq_norm.T - 2.0 * (x @ x.T)
    np.maximum(sq_dist, 0.0, out=sq_dist)
    distances = np.sqrt(sq_dist)
    np.fill_diagonal(distances, np.inf)

    nearest = np.partition(distances, kth=k - 1, axis=1)[:, :k]
    nearest.sort(axis=1)
    nearest = np.clip(nearest, min_value, None)

    t_k = nearest[:, -1]
    ratios = t_k[:, None] / nearest[:, :-1]
    logs = np.log(np.clip(ratios, min_value, None))
    denom = np.mean(logs, axis=1)

    valid = np.isfinite(denom) & (denom > 0.0)
    local = np.full(n_samples, np.nan, dtype=float)
    local[valid] = 1.0 / denom[valid]
    if not np.any(valid):
        raise ValueError("Estimator failed: no valid local neighbourhoods were found.")

    estimate = float(np.nanmean(local))
    if return_pointwise:
        return estimate, local
    return estimate
