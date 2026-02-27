from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class SinkhornResult:
    """Outputs from Sinkhorn-Knopp doubly-stochastic scaling."""

    scaled_matrix: np.ndarray
    row_scaling: np.ndarray
    col_scaling: np.ndarray
    converged: bool
    n_iter: int
    max_deviation: float


def sinkhorn_knopp_scale(
    matrix: np.ndarray,
    *,
    max_iter: int = 500,
    tol: float = 1e-6,
    min_value: float = 1e-12,
) -> SinkhornResult:
    """Scale a strictly positive matrix toward doubly stochastic form.

    This is intentionally lightweight scaffolding for future robust-kernel phases.
    It computes diagonal factors ``D_r`` and ``D_c`` such that
    ``S = D_r @ A @ D_c`` has row/column sums close to one.

    Args:
        matrix: Dense non-negative matrix. Values are clipped below by
            ``min_value`` to keep scaling numerically stable.
        max_iter: Maximum normalization iterations.
        tol: Convergence tolerance on max absolute row/column-sum deviation from 1.
        min_value: Positive floor to avoid divide-by-zero in scaling updates.

    Returns:
        ``SinkhornResult`` with the scaled matrix and convergence metadata.
    """
    a = np.asarray(matrix, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"`matrix` must be square 2D, got shape {a.shape}.")
    if max_iter <= 0:
        raise ValueError(f"`max_iter` must be > 0, got {max_iter}.")
    if tol <= 0.0:
        raise ValueError(f"`tol` must be > 0, got {tol}.")
    if min_value <= 0.0:
        raise ValueError(f"`min_value` must be > 0, got {min_value}.")

    positive = np.clip(a, min_value, None)
    n = positive.shape[0]
    r = np.ones(n, dtype=float)
    c = np.ones(n, dtype=float)

    converged = False
    max_deviation = np.inf
    n_iter = 0

    for step in range(1, max_iter + 1):
        rc = positive @ c
        rc = np.clip(rc, min_value, None)
        r = 1.0 / rc

        ctr = positive.T @ r
        ctr = np.clip(ctr, min_value, None)
        c = 1.0 / ctr

        scaled = (r[:, None] * positive) * c[None, :]
        row_err = np.max(np.abs(scaled.sum(axis=1) - 1.0))
        col_err = np.max(np.abs(scaled.sum(axis=0) - 1.0))
        max_deviation = float(max(row_err, col_err))
        n_iter = step
        if max_deviation <= tol:
            converged = True
            break

    scaled_matrix = (r[:, None] * positive) * c[None, :]
    return SinkhornResult(
        scaled_matrix=np.asarray(scaled_matrix, dtype=float),
        row_scaling=np.asarray(r, dtype=float),
        col_scaling=np.asarray(c, dtype=float),
        converged=converged,
        n_iter=n_iter,
        max_deviation=max_deviation,
    )


def maybe_doubly_stochastic_scale(
    matrix: np.ndarray,
    *,
    enabled: bool = False,
    max_iter: int = 500,
    tol: float = 1e-6,
    min_value: float = 1e-12,
) -> np.ndarray:
    """Optionally apply Sinkhorn scaling, preserving current defaults.

    Integration point:
    - Build affinity with ``gaussian_kernel``.
    - If ``enabled``, run this function before diffusion row-normalization.
    - Keep ``enabled=False`` to preserve the current bandwidth-stability baseline.
    """
    if not enabled:
        return np.asarray(matrix, dtype=float)
    result = sinkhorn_knopp_scale(
        matrix=matrix,
        max_iter=max_iter,
        tol=tol,
        min_value=min_value,
    )
    return result.scaled_matrix
