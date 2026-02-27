from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix

from nase.graphs.distances import pairwise_squared_distance_matrix
from nase.graphs.kernels import gaussian_kernel
from nase.spectral.embedding import diffusion_map_embedding, diffusion_operator

DistanceBuilder = Callable[[np.ndarray], np.ndarray | csr_matrix]


@dataclass(slots=True)
class StabilityResult:
    k_star: int
    per_k_stability: dict[int, float]
    pairwise_bandwidth_scores: np.ndarray
    epsilon_pair_matrix: np.ndarray
    considered_max_k: int
    threshold: float
    adjacent_vector_stability: np.ndarray | None = None
    subspace_adjacent_stability: np.ndarray | None = None
    bandwidths: np.ndarray | None = None


def vector_alignment_score(v1: np.ndarray, v2: np.ndarray) -> float:
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 <= 0.0 or n2 <= 0.0:
        return 0.0
    score = abs(float(np.dot(v1 / n1, v2 / n2)))
    return float(np.clip(score, 0.0, 1.0))


def compute_eigenvectors_across_bandwidths(
    *,
    bandwidths: list[float] | np.ndarray,
    k_max: int,
    distance_matrix: np.ndarray | csr_matrix | None = None,
    data: np.ndarray | None = None,
    distance_builder: DistanceBuilder | None = None,
    alpha: float = 0.5,
) -> list[np.ndarray]:
    if k_max < 1:
        raise ValueError(f"`k_max` must be >= 1, got {k_max}.")

    bandwidths_arr = np.asarray(bandwidths, dtype=float).reshape(-1)
    if bandwidths_arr.size == 0:
        raise ValueError("`bandwidths` must not be empty.")
    if np.any(bandwidths_arr <= 0.0):
        raise ValueError("All bandwidths must be > 0.")

    if distance_matrix is None:
        if data is None:
            raise ValueError("Provide either `distance_matrix` or `data`.")
        builder = distance_builder or pairwise_squared_distance_matrix
        distance_matrix = builder(np.asarray(data, dtype=float))

    evecs_by_bandwidth: list[np.ndarray] = []
    for epsilon in bandwidths_arr:
        affinity = gaussian_kernel(distances=distance_matrix, epsilon=float(epsilon))
        operator = diffusion_operator(affinity=affinity, alpha=alpha)
        if not isinstance(operator, np.ndarray):
            operator = operator.toarray()
        _, _, evecs = diffusion_map_embedding(
            operator=np.asarray(operator, dtype=float), k=k_max, t=1.0
        )
        evecs_by_bandwidth.append(evecs)
    return evecs_by_bandwidth


def _safe_max_k(eigenvectors_by_epsilon: list[np.ndarray], requested_max_k: int) -> int:
    if not eigenvectors_by_epsilon:
        return max(1, requested_max_k)
    available = int(min(max(int(ev.shape[1]) - 1, 1) for ev in eigenvectors_by_epsilon))
    return min(requested_max_k, available)


def compute_adjacent_vector_stability(
    eigenvectors_by_epsilon: list[np.ndarray], max_k: int
) -> np.ndarray:
    n_eps = len(eigenvectors_by_epsilon)
    if n_eps < 2:
        return np.empty((0, max_k), dtype=float)
    scores = np.zeros((n_eps - 1, max_k), dtype=float)
    for i in range(n_eps - 1):
        left = eigenvectors_by_epsilon[i]
        right = eigenvectors_by_epsilon[i + 1]
        for k in range(1, max_k + 1):
            scores[i, k - 1] = vector_alignment_score(left[:, k], right[:, k])
    return scores


def compute_per_k_stability(adjacent_vector_scores: np.ndarray, max_k: int) -> dict[int, float]:
    if adjacent_vector_scores.size == 0:
        return {k: 1.0 for k in range(1, max_k + 1)}
    scores: dict[int, float] = {}
    for k in range(1, max_k + 1):
        scores[k] = float(np.mean(adjacent_vector_scores[:, k - 1]))
    return scores


def compute_epsilon_pair_matrix(
    eigenvectors_by_epsilon: list[np.ndarray], max_k: int
) -> np.ndarray:
    n_eps = len(eigenvectors_by_epsilon)
    matrix = np.eye(n_eps, dtype=float)
    if n_eps < 2:
        return matrix
    for i in range(n_eps):
        for j in range(i + 1, n_eps):
            vals = [
                vector_alignment_score(
                    eigenvectors_by_epsilon[i][:, k], eigenvectors_by_epsilon[j][:, k]
                )
                for k in range(1, max_k + 1)
            ]
            score = float(np.mean(vals)) if vals else 1.0
            matrix[i, j] = score
            matrix[j, i] = score
    return matrix


def adjacent_subspace_stability_stub(
    eigenvectors_by_epsilon: list[np.ndarray], max_k: int
) -> np.ndarray | None:
    """Placeholder for principal-angle stability of span(V[:, :k]) across bandwidths."""
    _ = (eigenvectors_by_epsilon, max_k)
    return None


def select_k_bandwidth_stability(
    eigenvectors_by_epsilon: list[np.ndarray] | None = None,
    min_k: int = 1,
    max_k: int = 20,
    threshold: float = 0.9,
    *,
    distance_matrix: np.ndarray | csr_matrix | None = None,
    data: np.ndarray | None = None,
    distance_builder: DistanceBuilder | None = None,
    bandwidths: list[float] | np.ndarray | None = None,
    alpha: float = 0.5,
    enable_subspace_stability: bool = False,
) -> StabilityResult:
    if eigenvectors_by_epsilon is None:
        if bandwidths is None:
            raise ValueError(
                "When `eigenvectors_by_epsilon` is not provided, `bandwidths` must be provided."
            )
        eigenvectors_by_epsilon = compute_eigenvectors_across_bandwidths(
            bandwidths=bandwidths,
            k_max=max_k,
            distance_matrix=distance_matrix,
            data=data,
            distance_builder=distance_builder,
            alpha=alpha,
        )

    safe_max_k = _safe_max_k(eigenvectors_by_epsilon, requested_max_k=max_k)
    adjacent_vector_scores = compute_adjacent_vector_stability(
        eigenvectors_by_epsilon=eigenvectors_by_epsilon, max_k=safe_max_k
    )
    score_map = compute_per_k_stability(
        adjacent_vector_scores=adjacent_vector_scores, max_k=safe_max_k
    )
    bounded_min_k = max(1, min(min_k, safe_max_k))
    passing = [
        k for k in range(bounded_min_k, safe_max_k + 1) if score_map.get(k, 0.0) >= threshold
    ]
    k_star = max(passing) if passing else bounded_min_k
    pairwise = np.array([score_map[k] for k in range(bounded_min_k, safe_max_k + 1)], dtype=float)
    epsilon_pair_matrix = compute_epsilon_pair_matrix(
        eigenvectors_by_epsilon=eigenvectors_by_epsilon, max_k=safe_max_k
    )
    subspace_scores = (
        adjacent_subspace_stability_stub(
            eigenvectors_by_epsilon=eigenvectors_by_epsilon, max_k=safe_max_k
        )
        if enable_subspace_stability
        else None
    )
    bandwidths_arr = None if bandwidths is None else np.asarray(bandwidths, dtype=float).reshape(-1)
    return StabilityResult(
        k_star=k_star,
        per_k_stability=score_map,
        pairwise_bandwidth_scores=pairwise,
        epsilon_pair_matrix=epsilon_pair_matrix,
        considered_max_k=safe_max_k,
        threshold=threshold,
        adjacent_vector_stability=adjacent_vector_scores,
        subspace_adjacent_stability=subspace_scores,
        bandwidths=bandwidths_arr,
    )
