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
from nase.data.synthetic import SyntheticManifoldData, generate_synthetic
from nase.estimators.intrinsic_dimension import levina_bickel_mle_intrinsic_dimension
from nase.experiments.configs import config_from_dict, config_to_dict
from nase.experiments.io import make_run_dir, write_json, write_yaml
from nase.graphs.distances import choose_distance_backend
from nase.graphs.kernels import gaussian_kernel
from nase.metrics.embedding_quality import continuity_score, trustworthiness_score
from nase.metrics.geodesic import geodesic_consistency_score
from nase.metrics.subspace import oracle_cutoff
from nase.plots.ablations import plot_cutoff_ablation
from nase.plots.embeddings import plot_embedding_2d, plot_embedding_3d
from nase.plots.spectrum import plot_eigengap, plot_spectrum
from nase.plots.stability import plot_stability_heatmap, plot_stability_scores
from nase.robust.dss import maybe_doubly_stochastic_scale
from nase.spectral.embedding import diffusion_map_embedding, diffusion_operator


@dataclass(slots=True)
class RunResult:
    run_dir: Path
    metrics: dict[str, Any]
    selected_k: int


@dataclass(slots=True)
class ComputationBundle:
    clean_points: np.ndarray
    noisy_points: np.ndarray
    intrinsic_coords: np.ndarray
    epsilon_grid: np.ndarray
    noisy_embeddings_by_epsilon: dict[str, np.ndarray]
    base_embedding_noisy: np.ndarray
    base_evals_noisy: np.ndarray
    base_evecs_noisy: np.ndarray
    base_embedding_clean: np.ndarray
    base_evals_clean: np.ndarray
    base_evecs_clean: np.ndarray
    k_eigengap: int
    stability_result: StabilityResult
    k_oracle: int
    oracle_distances: dict[int, float]
    selected_k: int
    known_noise_r: float
    estimated_intrinsic_dim_noisy: float | None
    estimated_intrinsic_dim_clean: float | None


def _compute_operator(points: np.ndarray, epsilon: float, config: ExperimentConfig) -> np.ndarray:
    distances = choose_distance_backend(
        points=points,
        use_knn=config.graph.use_knn,
        knn_k=config.graph.knn_k,
        sparse_threshold_n=config.graph.sparse_threshold_n,
    )
    affinity = gaussian_kernel(distances=distances, epsilon=epsilon)
    if config.graph.enable_dss:
        if not isinstance(affinity, np.ndarray):
            affinity = affinity.toarray()
        affinity = maybe_doubly_stochastic_scale(
            affinity,
            enabled=True,
            max_iter=config.graph.dss_max_iter,
            tol=config.graph.dss_tol,
            min_value=config.graph.dss_min_value,
        )
    operator = diffusion_operator(affinity=affinity, alpha=config.spectral.alpha)
    if not isinstance(operator, np.ndarray):
        operator = operator.toarray()
    return np.asarray(operator, dtype=float)


def _primary_intrinsic_coords(latent_params: dict[str, np.ndarray]) -> np.ndarray:
    if not latent_params:
        raise ValueError("Expected at least one latent parameter for synthetic data.")
    if "theta" in latent_params:
        return np.asarray(latent_params["theta"], dtype=float)
    first_key = sorted(latent_params.keys())[0]
    return np.asarray(latent_params[first_key], dtype=float)


def _compute_embedding(
    *, points: np.ndarray, epsilon: float, config: ExperimentConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    operator = _compute_operator(points=points, epsilon=epsilon, config=config)
    return diffusion_map_embedding(
        operator=operator, k=config.spectral.n_eigs - 1, t=config.spectral.diffusion_time
    )


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
    data: SyntheticManifoldData = generate_synthetic(
        manifold=config.data.manifold,
        n=config.data.n_samples,
        ambient_dim=config.data.ambient_dim,
        r=config.data.noise_std,
        seed=config.data.seed,
    )
    intrinsic = _primary_intrinsic_coords(data.latent_params)
    clean_points = np.asarray(data.X_clean, dtype=float)
    noisy_points = np.asarray(data.X_noisy, dtype=float)
    epsilon_grid = np.asarray(config.graph.epsilon_grid, dtype=float).reshape(-1)
    per_eps_evecs: list[np.ndarray] = []
    noisy_embeddings_by_epsilon: dict[str, np.ndarray] = {}
    base_evals_noisy: np.ndarray | None = None
    base_embedding_noisy: np.ndarray | None = None
    base_evecs_noisy: np.ndarray | None = None

    for epsilon in epsilon_grid:
        embedding, evals, evecs = _compute_embedding(
            points=noisy_points, epsilon=float(epsilon), config=config
        )
        noisy_embeddings_by_epsilon[f"{float(epsilon):.12g}"] = embedding
        per_eps_evecs.append(evecs)
        if np.isclose(float(epsilon), config.graph.epsilon):
            base_evals_noisy = evals
            base_embedding_noisy = embedding
            base_evecs_noisy = evecs

    if base_evals_noisy is None or base_embedding_noisy is None or base_evecs_noisy is None:
        base_embedding_noisy, base_evals_noisy, base_evecs_noisy = _compute_embedding(
            points=noisy_points, epsilon=config.graph.epsilon, config=config
        )

    k_eigengap = select_k_eigengap(
        eigenvalues=base_evals_noisy,
        min_k=config.cutoff.eigengap_min_k,
        max_k=config.cutoff.eigengap_max_k,
    )
    stability_result: StabilityResult = select_k_bandwidth_stability(
        eigenvectors_by_epsilon=per_eps_evecs,
        min_k=config.cutoff.stability_min_k,
        max_k=config.cutoff.stability_max_k,
        threshold=config.cutoff.stability_threshold,
    )
    base_embedding_clean, base_evals_clean, base_evecs_clean = _compute_embedding(
        points=clean_points, epsilon=config.graph.epsilon, config=config
    )
    k_oracle, oracle_distances = oracle_cutoff(
        clean_eigenvectors=base_evecs_clean[:, 1:],
        noisy_eigenvectors=base_evecs_noisy[:, 1:],
        min_k=max(1, config.cutoff.stability_min_k),
        max_k=min(config.cutoff.stability_max_k, config.spectral.n_eigs - 1),
    )

    if config.cutoff.method == "eigengap":
        selected_k = k_eigengap
    else:
        selected_k = stability_result.k_star

    dim_noisy: float | None = None
    dim_clean: float | None = None
    if config.estimators.enable_intrinsic_dim:
        id_k = min(config.estimators.intrinsic_dim_k, config.data.n_samples - 1)
        dim_noisy = float(levina_bickel_mle_intrinsic_dimension(noisy_points, k=id_k))
        if config.estimators.intrinsic_dim_estimate_clean:
            dim_clean = float(levina_bickel_mle_intrinsic_dimension(clean_points, k=id_k))

    return ComputationBundle(
        clean_points=clean_points,
        noisy_points=noisy_points,
        intrinsic_coords=intrinsic,
        epsilon_grid=epsilon_grid,
        noisy_embeddings_by_epsilon=noisy_embeddings_by_epsilon,
        base_embedding_noisy=base_embedding_noisy,
        base_evals_noisy=base_evals_noisy,
        base_evecs_noisy=base_evecs_noisy,
        base_embedding_clean=base_embedding_clean,
        base_evals_clean=base_evals_clean,
        base_evecs_clean=base_evecs_clean,
        k_eigengap=int(k_eigengap),
        stability_result=stability_result,
        k_oracle=int(k_oracle),
        oracle_distances={int(k): float(v) for k, v in oracle_distances.items()},
        selected_k=int(selected_k),
        known_noise_r=float(data.metadata["r"]),
        estimated_intrinsic_dim_noisy=dim_noisy,
        estimated_intrinsic_dim_clean=dim_clean,
    )


def _write_plots(
    *,
    config: ExperimentConfig,
    figures_dir: Path,
    bundle: ComputationBundle,
    selected_k_override: int | None = None,
    dpi_override: int | None = None,
    formats_override: list[str] | None = None,
) -> None:
    selected_k = int(selected_k_override) if selected_k_override is not None else bundle.selected_k
    dpi = int(dpi_override) if dpi_override is not None else config.plot.dpi
    formats = formats_override if formats_override is not None else list(config.plot.formats)
    colour = bundle.intrinsic_coords.ravel()
    selected_embedding = bundle.base_embedding_noisy[:, :selected_k]
    for ext in formats:
        plot_spectrum(
            bundle.base_evals_noisy,
            figures_dir / f"spectrum.{ext}",
            dpi=dpi,
            selected_k=selected_k,
        )
        plot_eigengap(
            bundle.base_evals_noisy,
            figures_dir / f"eigengap.{ext}",
            dpi=dpi,
            selected_k=selected_k,
            min_k=config.cutoff.eigengap_min_k,
            max_k=config.cutoff.eigengap_max_k,
        )
        plot_stability_scores(
            bundle.stability_result.per_k_stability,
            figures_dir / f"stability.{ext}",
            dpi=dpi,
            selected_k=selected_k,
            threshold=config.cutoff.stability_threshold,
        )
        plot_stability_heatmap(
            matrix=bundle.stability_result.epsilon_pair_matrix,
            epsilon_grid=bundle.epsilon_grid.tolist(),
            out_path=figures_dir / f"stability_heatmap.{ext}",
            dpi=dpi,
        )
        emb2 = selected_embedding[:, :2] if selected_embedding.shape[1] >= 2 else selected_embedding
        plot_embedding_2d(
            emb2,
            colour=colour,
            out_path=figures_dir / f"embedding.{ext}",
            dpi=dpi,
            colorbar_label="Latent parameter",
        )
        plot_embedding_3d(
            selected_embedding,
            colour=colour,
            out_path=figures_dir / f"embedding_3d.{ext}",
            dpi=dpi,
            colorbar_label="Latent parameter",
        )
        ablation_data: dict[str, float] = {
            "eigengap_k": float(bundle.k_eigengap),
            "stability_k": float(bundle.stability_result.k_star),
            "oracle_k": float(bundle.k_oracle),
            "selected_k": float(selected_k),
        }
        if bundle.estimated_intrinsic_dim_noisy is not None:
            ablation_data["intrinsic_dim_k"] = float(np.ceil(bundle.estimated_intrinsic_dim_noisy))
        plot_cutoff_ablation(
            ablation_data,
            metric_name="Chosen cutoff",
            out_path=figures_dir / f"ablation_cutoff.{ext}",
            dpi=dpi,
        )


def run_experiment(config: ExperimentConfig) -> RunResult:
    bundle = _compute_bundle(config)
    selected_embedding = bundle.base_embedding_noisy[:, : bundle.selected_k]
    quality = _evaluate_metrics(
        high_dim=bundle.noisy_points,
        embedding=selected_embedding,
        intrinsic_coords=bundle.intrinsic_coords,
    )
    quality_by_cutoff = {
        "eigengap": _evaluate_metrics(
            high_dim=bundle.noisy_points,
            embedding=bundle.base_embedding_noisy[:, : bundle.k_eigengap],
            intrinsic_coords=bundle.intrinsic_coords,
        ),
        "bandwidth_stability": _evaluate_metrics(
            high_dim=bundle.noisy_points,
            embedding=bundle.base_embedding_noisy[:, : bundle.stability_result.k_star],
            intrinsic_coords=bundle.intrinsic_coords,
        ),
        "oracle": _evaluate_metrics(
            high_dim=bundle.noisy_points,
            embedding=bundle.base_embedding_noisy[:, : bundle.k_oracle],
            intrinsic_coords=bundle.intrinsic_coords,
        ),
    }

    intrinsic_dim_block: dict[str, Any] = {}
    if bundle.estimated_intrinsic_dim_noisy is not None:
        intrinsic_dim_block["estimated_intrinsic_dim_noisy"] = bundle.estimated_intrinsic_dim_noisy
        intrinsic_dim_block["k_intrinsic_dim"] = int(np.ceil(bundle.estimated_intrinsic_dim_noisy))
    if bundle.estimated_intrinsic_dim_clean is not None:
        intrinsic_dim_block["estimated_intrinsic_dim_clean"] = bundle.estimated_intrinsic_dim_clean

    metrics: dict[str, Any] = {
        "metadata": {
            "seed": config.data.seed,
            "method": config.cutoff.method,
            "epsilon": config.graph.epsilon,
            "epsilon_grid": [float(v) for v in bundle.epsilon_grid],
            "manifold": config.data.manifold,
            "n_samples": config.data.n_samples,
        },
        "known_noise_r": bundle.known_noise_r,
        "selected_k": bundle.selected_k,
        "k_eigengap": int(bundle.k_eigengap),
        "k_bandwidth_stability": int(bundle.stability_result.k_star),
        "k_oracle": int(bundle.k_oracle),
        **intrinsic_dim_block,
        "bandwidth_stability_scores": {
            str(k): v for k, v in bundle.stability_result.per_k_stability.items()
        },
        "oracle_subspace_distance_by_k": {str(k): v for k, v in bundle.oracle_distances.items()},
        "metrics_by_cutoff": quality_by_cutoff,
        **quality,
    }
    cutoffs: dict[str, Any] = {
        "selected_method": config.cutoff.method,
        "selected_k": int(bundle.selected_k),
        "eigengap_k": int(bundle.k_eigengap),
        "bandwidth_stability_k": int(bundle.stability_result.k_star),
        "oracle_k": int(bundle.k_oracle),
        **intrinsic_dim_block,
        "stability_threshold": float(config.cutoff.stability_threshold),
        "oracle_subspace_distance_by_k": {str(k): v for k, v in bundle.oracle_distances.items()},
    }

    run_dir = make_run_dir(config.output_root, config.name)
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    write_yaml(run_dir / "config.yaml", config_to_dict(config))
    write_json(run_dir / "metrics.json", metrics)
    write_json(run_dir / "cutoffs.json", cutoffs)

    stacked_embeddings = np.stack(
        [bundle.noisy_embeddings_by_epsilon[f"{float(e):.12g}"] for e in bundle.epsilon_grid],
        axis=0,
    )
    stability_scores = np.array(
        [
            float(bundle.stability_result.per_k_stability[k])
            for k in sorted(bundle.stability_result.per_k_stability)
        ],
        dtype=float,
    )
    np.savez_compressed(
        run_dir / "arrays.npz",
        x_clean=bundle.clean_points,
        x_noisy=bundle.noisy_points,
        intrinsic_coords=bundle.intrinsic_coords,
        epsilon_grid=bundle.epsilon_grid,
        noisy_embeddings_grid=stacked_embeddings,
        base_embedding_noisy=bundle.base_embedding_noisy,
        base_embedding_clean=bundle.base_embedding_clean,
        base_evals_noisy=bundle.base_evals_noisy,
        base_evals_clean=bundle.base_evals_clean,
        base_evecs_noisy=bundle.base_evecs_noisy,
        base_evecs_clean=bundle.base_evecs_clean,
        stability_scores=stability_scores,
        stability_pair_matrix=bundle.stability_result.epsilon_pair_matrix,
    )

    _write_plots(config=config, figures_dir=figures_dir, bundle=bundle)

    return RunResult(run_dir=run_dir, metrics=metrics, selected_k=bundle.selected_k)


def regenerate_plots_from_run_dir(
    *,
    run_dir: Path,
    output_dir: Path | None = None,
    dpi: int | None = None,
    formats: list[str] | None = None,
) -> Path:
    config_path = run_dir / "config.yaml"
    cutoffs_path = run_dir / "cutoffs.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config file: {config_path}")
    if not cutoffs_path.exists():
        raise FileNotFoundError(f"Missing cutoffs file: {cutoffs_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle) or {}
    config = config_from_dict(raw_config)
    cutoffs = json.loads(cutoffs_path.read_text(encoding="utf-8"))
    selected_k_saved = int(cutoffs.get("selected_k", 0)) or None

    bundle = _compute_bundle(config)
    target_dir = output_dir if output_dir is not None else run_dir / "figures"
    target_dir.mkdir(parents=True, exist_ok=True)
    _write_plots(
        config=config,
        figures_dir=target_dir,
        bundle=bundle,
        selected_k_override=selected_k_saved,
        dpi_override=dpi,
        formats_override=formats,
    )
    return target_dir
