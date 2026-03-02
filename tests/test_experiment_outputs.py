from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from nase.config import ExperimentConfig
from nase.experiments.runner import run_experiment


def test_runner_writes_expected_output_manifest(tmp_path: Path) -> None:
    cfg = ExperimentConfig()
    cfg.output_root = tmp_path
    cfg.data.n_samples = 100
    cfg.spectral.n_eigs = 8
    cfg.cutoff.eigengap_max_k = 5
    cfg.cutoff.stability_max_k = 5
    result = run_experiment(cfg)

    assert (result.run_dir / "config.yaml").exists()
    assert (result.run_dir / "metrics.json").exists()
    assert (result.run_dir / "cutoffs.json").exists()
    assert (result.run_dir / "arrays.npz").exists()

    figure_names = {p.name for p in (result.run_dir / "figures").iterdir()}
    assert "spectrum.png" in figure_names
    assert "eigengap.png" in figure_names
    assert "stability.png" in figure_names
    assert "stability_heatmap.png" in figure_names
    assert "embedding.png" in figure_names
    assert "embedding_3d.png" in figure_names
    assert "ablation_cutoff.png" in figure_names

    metrics = json.loads((result.run_dir / "metrics.json").read_text(encoding="utf-8"))
    cutoffs = json.loads((result.run_dir / "cutoffs.json").read_text(encoding="utf-8"))
    arrays = np.load(result.run_dir / "arrays.npz")
    assert "k_oracle" in metrics
    assert "oracle_k" in cutoffs
    assert "x_clean" in arrays


def test_runner_with_intrinsic_dim_estimation(tmp_path: Path) -> None:
    cfg = ExperimentConfig()
    cfg.output_root = tmp_path
    cfg.data.manifold = "circle"
    cfg.data.n_samples = 120
    cfg.data.ambient_dim = 3
    cfg.data.noise_std = 0.03
    cfg.spectral.n_eigs = 8
    cfg.cutoff.eigengap_max_k = 5
    cfg.cutoff.stability_max_k = 5
    cfg.estimators.enable_intrinsic_dim = True
    cfg.estimators.intrinsic_dim_k = 10
    cfg.estimators.intrinsic_dim_estimate_clean = True

    result = run_experiment(cfg)
    metrics = json.loads((result.run_dir / "metrics.json").read_text(encoding="utf-8"))
    cutoffs = json.loads((result.run_dir / "cutoffs.json").read_text(encoding="utf-8"))

    assert "estimated_intrinsic_dim_noisy" in metrics
    assert "estimated_intrinsic_dim_clean" in metrics
    assert "k_intrinsic_dim" in metrics
    assert "k_intrinsic_dim" in cutoffs

    dim_noisy = metrics["estimated_intrinsic_dim_noisy"]
    dim_clean = metrics["estimated_intrinsic_dim_clean"]
    assert 0.5 < dim_noisy < 4.0, f"Expected ~1 for circle, got {dim_noisy}"
    assert 0.5 < dim_clean < 3.0, f"Expected ~1 for clean circle, got {dim_clean}"


def test_runner_without_intrinsic_dim_has_no_dim_keys(tmp_path: Path) -> None:
    cfg = ExperimentConfig()
    cfg.output_root = tmp_path
    cfg.data.n_samples = 80
    cfg.spectral.n_eigs = 8
    cfg.cutoff.eigengap_max_k = 5
    cfg.cutoff.stability_max_k = 5
    cfg.estimators.enable_intrinsic_dim = False

    result = run_experiment(cfg)
    metrics = json.loads((result.run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert "estimated_intrinsic_dim_noisy" not in metrics
    assert "k_intrinsic_dim" not in metrics
