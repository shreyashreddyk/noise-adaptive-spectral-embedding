from __future__ import annotations

import numpy as np


def _validate_subspace_inputs(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    u_arr = np.asarray(u, dtype=float)
    v_arr = np.asarray(v, dtype=float)
    if u_arr.ndim != 2 or v_arr.ndim != 2:
        raise ValueError("Subspace bases must be 2D arrays.")
    if u_arr.shape[0] != v_arr.shape[0]:
        raise ValueError("Subspace bases must share ambient dimension.")
    if u_arr.shape[1] == 0 or v_arr.shape[1] == 0:
        raise ValueError("Subspace bases must have at least one basis vector.")
    return u_arr, v_arr


def _orthonormal_basis(x: np.ndarray) -> np.ndarray:
    q, _ = np.linalg.qr(x)
    return q


def principal_angles(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Compute principal angles (radians) between two subspaces via SVD."""
    u_arr, v_arr = _validate_subspace_inputs(u=u, v=v)
    q_u = _orthonormal_basis(u_arr)
    q_v = _orthonormal_basis(v_arr)
    cosine_matrix = q_u.T @ q_v
    singular_values = np.linalg.svd(cosine_matrix, compute_uv=False)
    singular_values = np.clip(singular_values, -1.0, 1.0)
    return np.arccos(singular_values)


def subspace_distance(u: np.ndarray, v: np.ndarray) -> float:
    """Chordal distance between subspaces from principal angles."""
    angles = principal_angles(u, v)
    return float(np.linalg.norm(np.sin(angles)))


def subspace_similarity(u: np.ndarray, v: np.ndarray) -> float:
    angles = principal_angles(u, v)
    return float(np.mean(np.cos(angles)))


def oracle_cutoff(
    clean_eigenvectors: np.ndarray,
    noisy_eigenvectors: np.ndarray,
    min_k: int = 1,
    max_k: int | None = None,
) -> tuple[int, dict[int, float]]:
    """
    Compute oracle cutoff k by comparing clean/noisy leading eigenspaces.

    Returns the minimizer of subspace distance over k in [min_k, max_k], using
    the smallest k in case of ties.
    """
    clean, noisy = _validate_subspace_inputs(u=clean_eigenvectors, v=noisy_eigenvectors)
    upper = min(clean.shape[1], noisy.shape[1])
    if max_k is not None:
        upper = min(upper, int(max_k))
    if upper < 1:
        raise ValueError("Need at least one column in each eigenspace basis.")
    if min_k < 1:
        raise ValueError("`min_k` must be >= 1.")
    if min_k > upper:
        raise ValueError("`min_k` must not exceed available dimensions.")

    distances: dict[int, float] = {}
    for k in range(min_k, upper + 1):
        distances[k] = subspace_distance(clean[:, :k], noisy[:, :k])

    k_oracle = min(distances, key=lambda k: (distances[k], k))
    return int(k_oracle), distances
