from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nase.cutoffs.bandwidth_stability import StabilityResult
from nase.experiments.io import write_json


@dataclass(slots=True)
class RunMetadata:
    seed: int
    method: str
    manifold: str
    n_samples: int
    epsilon: float
    epsilon_grid: list[float]
    known_noise_r: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "method": self.method,
            "manifold": self.manifold,
            "n_samples": self.n_samples,
            "epsilon": self.epsilon,
            "epsilon_grid": self.epsilon_grid,
            "known_noise_r": self.known_noise_r,
        }


def stability_diagnostics_payload(
    stability: StabilityResult, epsilon_grid: list[float], selected_k: int
) -> dict[str, Any]:
    return {
        "selected_k": selected_k,
        "considered_max_k": stability.considered_max_k,
        "threshold": stability.threshold,
        "mode_scores": {str(k): v for k, v in stability.per_k_stability.items()},
        "pairwise_bandwidth_scores": stability.pairwise_bandwidth_scores.tolist(),
        "epsilon_grid": epsilon_grid,
        "epsilon_pair_matrix": stability.epsilon_pair_matrix.tolist(),
        "selection_rationale": (
            "Choose largest k with average cross-bandwidth eigenvector stability above threshold."
        ),
    }


def write_diagnostics(
    output_path: Path, metadata: RunMetadata, stability_payload: dict[str, Any]
) -> None:
    payload: dict[str, Any] = {
        "metadata": metadata.as_dict(),
        "stability": stability_payload,
    }
    write_json(path=output_path, payload=payload)
