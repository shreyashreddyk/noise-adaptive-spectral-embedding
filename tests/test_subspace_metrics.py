from __future__ import annotations

import numpy as np

from nase.metrics.subspace import oracle_cutoff, principal_angles, subspace_distance


def test_identical_subspaces_have_zero_distance() -> None:
    basis = np.eye(3)[:, :2]
    rotation = np.array([[0.0, -1.0], [1.0, 0.0]])
    rotated_basis = basis @ rotation

    angles = principal_angles(basis, rotated_basis)
    assert np.allclose(angles, np.zeros_like(angles), atol=1e-10)
    assert np.isclose(subspace_distance(basis, rotated_basis), 0.0, atol=1e-10)


def test_orthogonal_lines_have_right_angle_and_unit_distance() -> None:
    e1 = np.array([[1.0], [0.0], [0.0]])
    e2 = np.array([[0.0], [1.0], [0.0]])

    angles = principal_angles(e1, e2)
    assert np.allclose(angles, np.array([np.pi / 2.0]), atol=1e-10)
    assert np.isclose(subspace_distance(e1, e2), 1.0, atol=1e-10)


def test_oracle_cutoff_selects_known_best_k() -> None:
    clean = np.eye(4)[:, :3]
    noisy = np.column_stack(
        (
            np.array([1.0, 1.0, 0.0, 0.0]) / np.sqrt(2.0),
            np.array([0.0, 1.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0, 1.0]),
        )
    )

    k_oracle, per_k = oracle_cutoff(clean, noisy, min_k=1, max_k=3)
    assert k_oracle == 2
    assert per_k[2] < per_k[1]
    assert per_k[2] < per_k[3]
