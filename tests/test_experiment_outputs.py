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
