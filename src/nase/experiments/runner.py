from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from nase.config import ExperimentConfig
from nase.cutoffs.bandwidth_stability import StabilityResult, select_k_bandwidth_stability
from nase.cutoffs.eigengap import select_k_eigengap
from nase.data.manifolds import generate_manifold
from nase.data.noise import add_gaussian_noise
from nase.experiments.configs import config_from_dict, config_to_dict
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
from nase.plots.embeddings import plot_embedding_2d, plot_embedding_3d
from nase.plots.spectrum import plot_eigengap, plot_spectrum
from nase.plots.stability import plot_stability_heatmap, plot_stability_scores
from nase.spectral.embedding import diffusion_map_embedding, diffusion_operator
from nase.utils import set_global_seed


@dataclass(slots=True)
class RunResult:
    run_dir: Path
    metrics: dict[str, Any]
    selected_k: int


@dataclass(slots=True)
class ComputationBundle:
    manifold_intrinsic_coords: np.ndarray
    noisy_points: np.ndarray
    base_evals: np.ndarray
    selected_embedding: np.ndarray
    k_eigengap: int
    stability_result: StabilityResult
    selected_k: int
    known_noise_r: float


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


def _compute_bundle(config: ExperimentConfig) -> ComputationBundle:
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
            operator=operator, k=config.spectral.n_eigs - 1, t=config.spectral.diffusion_time
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
            operator=operator, k=config.spectral.n_eigs - 1, t=config.spectral.diffusion_time
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
    return ComputationBundle(
        manifold_intrinsic_coords=manifold.intrinsic_coords,
        noisy_points=noisy_points,
        base_evals=base_evals,
        selected_embedding=selected_embedding,
        k_eigengap=int(k_eigengap),
        stability_result=stability_result,
        selected_k=int(selected_k),
        known_noise_r=float(known_r),
    )


def _write_plots(
    *,
    config: ExperimentConfig,
    plots_dir: Path,
    bundle: ComputationBundle,
    selected_k_override: int | None = None,
    dpi_override: int | None = None,
    formats_override: list[str] | None = None,
) -> None:
    selected_k = int(selected_k_override) if selected_k_override is not None else bundle.selected_k
    dpi = int(dpi_override) if dpi_override is not None else config.plot.dpi
    formats = formats_override if formats_override is not None else list(config.plot.formats)
    colour = bundle.manifold_intrinsic_coords.ravel()
    for ext in formats:
        plot_spectrum(
            bundle.base_evals,
            plots_dir / f"spectrum.{ext}",
            dpi=dpi,
            selected_k=selected_k,
        )
        plot_eigengap(
            bundle.base_evals,
            plots_dir / f"eigengap.{ext}",
            dpi=dpi,
            selected_k=selected_k,
            min_k=config.cutoff.eigengap_min_k,
            max_k=config.cutoff.eigengap_max_k,
        )
        plot_stability_scores(
            bundle.stability_result.per_k_stability,
            plots_dir / f"stability.{ext}",
            dpi=dpi,
            selected_k=selected_k,
            threshold=config.cutoff.stability_threshold,
        )
        plot_stability_heatmap(
            matrix=bundle.stability_result.epsilon_pair_matrix,
            epsilon_grid=config.graph.epsilon_grid,
            out_path=plots_dir / f"stability_heatmap.{ext}",
            dpi=dpi,
        )
        emb2 = (
            bundle.selected_embedding[:, :2]
            if bundle.selected_embedding.shape[1] >= 2
            else bundle.selected_embedding
        )
        plot_embedding_2d(
            emb2,
            colour=colour,
            out_path=plots_dir / f"embedding.{ext}",
            dpi=dpi,
            colorbar_label="Latent parameter",
        )
        plot_embedding_3d(
            bundle.selected_embedding,
            colour=colour,
            out_path=plots_dir / f"embedding_3d.{ext}",
            dpi=dpi,
            colorbar_label="Latent parameter",
        )
        plot_cutoff_ablation(
            {
                "eigengap_k": float(bundle.k_eigengap),
                "stability_k": float(bundle.stability_result.k_star),
                "selected_k": float(selected_k),
            },
            metric_name="Chosen cutoff",
            out_path=plots_dir / f"ablation_cutoff.{ext}",
            dpi=dpi,
        )


def run_experiment(config: ExperimentConfig) -> RunResult:
    bundle = _compute_bundle(config)
    quality = _evaluate_metrics(
        high_dim=bundle.noisy_points,
        embedding=bundle.selected_embedding,
        intrinsic_coords=bundle.manifold_intrinsic_coords,
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
        "known_noise_r": bundle.known_noise_r,
        "selected_k": bundle.selected_k,
        "k_eigengap": int(bundle.k_eigengap),
        "k_bandwidth_stability": int(bundle.stability_result.k_star),
        "bandwidth_stability_scores": {
            str(k): v for k, v in bundle.stability_result.per_k_stability.items()
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
        known_noise_r=bundle.known_noise_r,
    )
    stability_payload = stability_diagnostics_payload(
        stability=bundle.stability_result,
        epsilon_grid=config.graph.epsilon_grid,
        selected_k=bundle.selected_k,
    )
    write_diagnostics(
        run_dir / "diagnostics.json", metadata=metadata, stability_payload=stability_payload
    )

    _write_plots(config=config, plots_dir=plots_dir, bundle=bundle)

    return RunResult(run_dir=run_dir, metrics=metrics, selected_k=bundle.selected_k)


def regenerate_plots_from_run_dir(
    *,
    run_dir: Path,
    output_dir: Path | None = None,
    dpi: int | None = None,
    formats: list[str] | None = None,
) -> Path:
    config_path = run_dir / "config.yaml"
    metrics_path = run_dir / "metrics.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config file: {config_path}")
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle) or {}
    config = config_from_dict(raw_config)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    selected_k_saved = int(metrics.get("selected_k", 0)) or None

    bundle = _compute_bundle(config)
    target_dir = output_dir if output_dir is not None else run_dir / "plots"
    target_dir.mkdir(parents=True, exist_ok=True)
    _write_plots(
        config=config,
        plots_dir=target_dir,
        bundle=bundle,
        selected_k_override=selected_k_saved,
        dpi_override=dpi,
        formats_override=formats,
    )
    return target_dir
