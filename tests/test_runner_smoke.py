from __future__ import annotations

from pathlib import Path

from nase.config import ExperimentConfig
from nase.experiments.runner import run_experiment


def test_runner_smoke(tmp_path: Path) -> None:
    cfg = ExperimentConfig()
    cfg.output_root = tmp_path
    cfg.data.n_samples = 120
    cfg.spectral.n_eigs = 10
    cfg.cutoff.stability_max_k = 6
    cfg.cutoff.eigengap_max_k = 6
    result = run_experiment(cfg)
    assert result.run_dir.exists()
    assert (result.run_dir / "metrics.json").exists()
    assert (result.run_dir / "cutoffs.json").exists()
    assert (result.run_dir / "arrays.npz").exists()
