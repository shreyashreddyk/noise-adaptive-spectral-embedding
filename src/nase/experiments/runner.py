from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from nase.config import ExperimentConfig
from nase.cutoffs.bandwidth_stability import StabilityResult, select_k_bandwidth_stability
from nase.cutoffs.eigengap import select_k_eigengap
from nase.data.manifolds import generate_manifold
from nase.data.noise import add_gaussian_noise
from nase.experiments.configs import config_to_dict
from nase.experiments.diagnostics import (
    RunMetadata,
    stability_diagnostics_payload,
    write_diagnostics,
)
from nase.experiments.io import make_run_dir, write_json, write_yaml
from nase.graphs.distances import choose_distance_backend
from nase.graphs.kernels import gaussian_kernel
from nase.metrics.embedding_quality import continuity_score, trustworthiness_score
from nase.metrics.geodesic import geodesic_consistency_score
from nase.plots.ablations import plot_cutoff_ablation
from nase.plots.embeddings import plot_embedding_2d
from nase.plots.spectrum import plot_spectrum
from nase.plots.stability import plot_stability_scores
from nase.plots.stability_heatmap import plot_stability_heatmap
from nase.spectral.diffusion_maps import diffusion_map_embedding, diffusion_operator
from nase.utils import set_global_seed


@dataclass(slots=True)
class RunResult:
    run_dir: Path
    metrics: dict[str, Any]
    selected_k: int


def _compute_operator(points: np.ndarray, epsilon: float, config: ExperimentConfig) -> np.ndarray:
    distances = choose_distance_backend(
        points=points,
        use_knn=config.graph.use_knn,
        knn_k=config.graph.knn_k,
        sparse_threshold_n=config.graph.sparse_threshold_n,
    )
    affinity = gaussian_kernel(distances=distances, epsilon=epsilon)
    operator = diffusion_operator(affinity=affinity, alpha=config.spectral.alpha)
    if not isinstance(operator, np.ndarray):
        operator = operator.toarray()
    return np.asarray(operator, dtype=float)


def _evaluate_metrics(
    high_dim: np.ndarray, embedding: np.ndarray, intrinsic_coords: np.ndarray
) -> dict[str, float]:
    emb2 = embedding[:, :2] if embedding.shape[1] >= 2 else embedding
    return {
        "trustworthiness": trustworthiness_score(high_dim=high_dim, low_dim=emb2),
        "continuity": continuity_score(high_dim=high_dim, low_dim=emb2),
        "geodesic_consistency": geodesic_consistency_score(
            reference_coords=intrinsic_coords, embedding=emb2
        ),
    }


def run_experiment(config: ExperimentConfig) -> RunResult:
    rng = set_global_seed(config.data.seed)
    manifold = generate_manifold(
        manifold=config.data.manifold,
        n_samples=config.data.n_samples,
        ambient_dim=config.data.ambient_dim,
        rng=rng,
    )
    noisy_points, known_r = add_gaussian_noise(
        points=manifold.points,
        noise_std=config.data.noise_std,
        rng=rng,
    )

    per_eps_evecs: list[np.ndarray] = []
    base_evals: np.ndarray | None = None
    base_embedding: np.ndarray | None = None

    for epsilon in config.graph.epsilon_grid:
        operator = _compute_operator(points=noisy_points, epsilon=epsilon, config=config)
        embedding, evals, evecs = diffusion_map_embedding(
            operator=operator, n_eigs=config.spectral.n_eigs, time=config.spectral.diffusion_time
        )
        per_eps_evecs.append(evecs)
        if np.isclose(epsilon, config.graph.epsilon):
            base_evals = evals
            base_embedding = embedding

    if base_evals is None or base_embedding is None:
        operator = _compute_operator(
            points=noisy_points, epsilon=config.graph.epsilon, config=config
        )
        base_embedding, base_evals, _ = diffusion_map_embedding(
            operator=operator, n_eigs=config.spectral.n_eigs, time=config.spectral.diffusion_time
        )

    k_eigengap = select_k_eigengap(
        eigenvalues=base_evals,
        min_k=config.cutoff.eigengap_min_k,
        max_k=config.cutoff.eigengap_max_k,
    )
    stability_result: StabilityResult = select_k_bandwidth_stability(
        eigenvectors_by_epsilon=per_eps_evecs,
        min_k=config.cutoff.stability_min_k,
        max_k=config.cutoff.stability_max_k,
        threshold=config.cutoff.stability_threshold,
    )

    if config.cutoff.method == "eigengap":
        selected_k = k_eigengap
    else:
        selected_k = stability_result.k_star

    selected_embedding = base_embedding[:, :selected_k]
    quality = _evaluate_metrics(
        high_dim=noisy_points,
        embedding=selected_embedding,
        intrinsic_coords=manifold.intrinsic_coords,
    )

    metrics: dict[str, Any] = {
        "metadata": {
            "seed": config.data.seed,
            "method": config.cutoff.method,
            "epsilon": config.graph.epsilon,
            "epsilon_grid": config.graph.epsilon_grid,
            "manifold": config.data.manifold,
            "n_samples": config.data.n_samples,
        },
        "known_noise_r": known_r,
        "selected_k": selected_k,
        "k_eigengap": int(k_eigengap),
        "k_bandwidth_stability": int(stability_result.k_star),
        "bandwidth_stability_scores": {
            str(k): v for k, v in stability_result.per_k_stability.items()
        },
        **quality,
    }

    run_dir = make_run_dir(config.output_root, config.name)
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    write_yaml(run_dir / "config.yaml", config_to_dict(config))
    write_json(run_dir / "metrics.json", metrics)
    metadata = RunMetadata(
        seed=config.data.seed,
        method=config.cutoff.method,
        manifold=config.data.manifold,
        n_samples=config.data.n_samples,
        epsilon=config.graph.epsilon,
        epsilon_grid=config.graph.epsilon_grid,
        known_noise_r=known_r,
    )
    stability_payload = stability_diagnostics_payload(
        stability=stability_result, epsilon_grid=config.graph.epsilon_grid, selected_k=selected_k
    )
    write_diagnostics(
        run_dir / "diagnostics.json", metadata=metadata, stability_payload=stability_payload
    )

    for ext in config.plot.formats:
        plot_spectrum(base_evals, plots_dir / f"spectrum.{ext}", dpi=config.plot.dpi)
        plot_stability_scores(
            stability_result.per_k_stability, plots_dir / f"stability.{ext}", dpi=config.plot.dpi
        )
        plot_stability_heatmap(
            matrix=stability_result.epsilon_pair_matrix,
            epsilon_grid=config.graph.epsilon_grid,
            out_path=plots_dir / f"stability_heatmap.{ext}",
            dpi=config.plot.dpi,
        )
        emb2 = selected_embedding[:, :2] if selected_embedding.shape[1] >= 2 else selected_embedding
        plot_embedding_2d(
            emb2,
            colour=manifold.intrinsic_coords.ravel(),
            out_path=plots_dir / f"embedding.{ext}",
            dpi=config.plot.dpi,
        )
        plot_cutoff_ablation(
            {
                "eigengap_k": float(k_eigengap),
                "stability_k": float(stability_result.k_star),
                "selected_k": float(selected_k),
            },
            metric_name="Chosen cutoff",
            out_path=plots_dir / f"ablation_cutoff.{ext}",
            dpi=config.plot.dpi,
        )

    return RunResult(run_dir=run_dir, metrics=metrics, selected_k=selected_k)
