from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class StabilityResult:
    k_star: int
    per_k_stability: dict[int, float]
    pairwise_bandwidth_scores: np.ndarray
    epsilon_pair_matrix: np.ndarray
    considered_max_k: int
    threshold: float


def vector_alignment_score(v1: np.ndarray, v2: np.ndarray) -> float:
    return float(abs(np.dot(v1, v2)))


def _safe_max_k(eigenvectors_by_epsilon: list[np.ndarray], requested_max_k: int) -> int:
    if not eigenvectors_by_epsilon:
        return requested_max_k
    available = int(min(max(int(ev.shape[1]) - 1, 1) for ev in eigenvectors_by_epsilon))
    return min(requested_max_k, available)


def compute_per_k_stability(
    eigenvectors_by_epsilon: list[np.ndarray], max_k: int
) -> dict[int, float]:
    n_eps = len(eigenvectors_by_epsilon)
    if n_eps < 2:
        return {k: 1.0 for k in range(1, max_k + 1)}
    scores: dict[int, float] = {}
    for k in range(1, max_k + 1):
        vals: list[float] = []
        for i in range(n_eps):
            for j in range(i + 1, n_eps):
                vals.append(
                    vector_alignment_score(
                        eigenvectors_by_epsilon[i][:, k], eigenvectors_by_epsilon[j][:, k]
                    )
                )
        scores[k] = float(np.mean(vals)) if vals else 1.0
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


def select_k_bandwidth_stability(
    eigenvectors_by_epsilon: list[np.ndarray],
    min_k: int,
    max_k: int,
    threshold: float,
) -> StabilityResult:
    safe_max_k = _safe_max_k(eigenvectors_by_epsilon, requested_max_k=max_k)
    score_map = compute_per_k_stability(
        eigenvectors_by_epsilon=eigenvectors_by_epsilon, max_k=safe_max_k
    )
    bounded_min_k = min(min_k, safe_max_k)
    passing = [
        k for k in range(bounded_min_k, safe_max_k + 1) if score_map.get(k, 0.0) >= threshold
    ]
    k_star = max(passing) if passing else bounded_min_k
    pairwise = np.array([score_map[k] for k in range(bounded_min_k, safe_max_k + 1)], dtype=float)
    epsilon_pair_matrix = compute_epsilon_pair_matrix(
        eigenvectors_by_epsilon=eigenvectors_by_epsilon, max_k=safe_max_k
    )
    return StabilityResult(
        k_star=k_star,
        per_k_stability=score_map,
        pairwise_bandwidth_scores=pairwise,
        epsilon_pair_matrix=epsilon_pair_matrix,
        considered_max_k=safe_max_k,
        threshold=threshold,
    )
