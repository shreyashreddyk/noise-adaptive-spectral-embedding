from __future__ import annotations

import numpy as np

from nase.cutoffs.bandwidth_stability import select_k_bandwidth_stability
from nase.cutoffs.eigengap import select_k_eigengap
from nase.graphs.distances import pairwise_squared_distance_matrix


def test_eigengap_selects_expected_nontrivial_index() -> None:
    evals = np.array([1.0, 0.99, 0.98, 0.70, 0.69, 0.68])
    k = select_k_eigengap(evals, min_k=1, max_k=4)
    assert k == 2


def test_eigengap_guardrail_never_returns_zero() -> None:
    evals = np.array([1.0, 0.95, 0.90, 0.89])
    k = select_k_eigengap(evals, min_k=0, max_k=10)
    assert k >= 1


def test_eigengap_respects_bounded_search_window() -> None:
    evals = np.array([1.0, 0.98, 0.96, 0.95, 0.60, 0.59, 0.58])
    k = select_k_eigengap(evals, min_k=1, max_k=2)
    assert k == 1


def test_bandwidth_stability_scores_are_bounded_and_select_valid_k() -> None:
    theta = np.linspace(0.0, 2.0 * np.pi, 96, endpoint=False)
    points = np.column_stack([np.cos(theta), np.sin(theta)])
    distances = pairwise_squared_distance_matrix(points, use_cache=False)
    bandwidths = [0.08, 0.12, 0.2, 0.35, 0.5]

    result = select_k_bandwidth_stability(
        distance_matrix=distances,
        bandwidths=bandwidths,
        min_k=1,
        max_k=8,
        threshold=0.8,
    )

    assert 1 <= result.k_star <= 8
    assert result.considered_max_k == 8
    assert result.adjacent_vector_stability is not None
    assert result.adjacent_vector_stability.shape == (len(bandwidths) - 1, 8)
    assert np.all(result.adjacent_vector_stability >= 0.0)
    assert np.all(result.adjacent_vector_stability <= 1.0)
    assert np.all(result.epsilon_pair_matrix >= 0.0)
    assert np.all(result.epsilon_pair_matrix <= 1.0)
    assert np.all(result.pairwise_bandwidth_scores >= 0.0)
    assert np.all(result.pairwise_bandwidth_scores <= 1.0)


def test_bandwidth_stability_output_shapes_match_requested_grid_and_k() -> None:
    theta = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
    points = np.column_stack([np.cos(theta), np.sin(theta)])
    distances = pairwise_squared_distance_matrix(points, use_cache=False)
    bandwidths = [0.08, 0.12, 0.2, 0.35]
    max_k = 6

    result = select_k_bandwidth_stability(
        distance_matrix=distances,
        bandwidths=bandwidths,
        min_k=1,
        max_k=max_k,
        threshold=0.75,
    )

    assert result.bandwidths is not None
    assert result.bandwidths.shape == (len(bandwidths),)
    assert result.epsilon_pair_matrix.shape == (len(bandwidths), len(bandwidths))
    assert result.adjacent_vector_stability is not None
    assert result.adjacent_vector_stability.shape == (len(bandwidths) - 1, max_k)
    assert result.pairwise_bandwidth_scores.shape == (max_k,)


def test_noiseless_circle_low_modes_are_more_stable_than_high_modes() -> None:
    theta = np.linspace(0.0, 2.0 * np.pi, 128, endpoint=False)
    points = np.column_stack([np.cos(theta), np.sin(theta)])
    distances = pairwise_squared_distance_matrix(points, use_cache=False)
    bandwidths = [0.07, 0.1, 0.15, 0.22, 0.32, 0.45]

    result = select_k_bandwidth_stability(
        distance_matrix=distances,
        bandwidths=bandwidths,
        min_k=1,
        max_k=10,
        threshold=0.5,
    )

    low = np.mean([result.per_k_stability[k] for k in (1, 2, 3)])
    high = np.mean([result.per_k_stability[k] for k in (8, 9, 10)])
    assert low > high
