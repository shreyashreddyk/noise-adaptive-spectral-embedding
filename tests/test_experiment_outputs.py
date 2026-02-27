from __future__ import annotations

import json
from pathlib import Path

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
    assert (result.run_dir / "diagnostics.json").exists()

    plot_names = {p.name for p in (result.run_dir / "plots").iterdir()}
    assert "spectrum.png" in plot_names
    assert "eigengap.png" in plot_names
    assert "stability.png" in plot_names
    assert "stability_heatmap.png" in plot_names
    assert "embedding.png" in plot_names
    assert "embedding_3d.png" in plot_names
    assert "ablation_cutoff.png" in plot_names

    payload = json.loads((result.run_dir / "diagnostics.json").read_text(encoding="utf-8"))
    assert "metadata" in payload
    assert "stability" in payload
