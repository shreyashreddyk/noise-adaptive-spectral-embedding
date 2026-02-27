from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from nase.cli import app
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


def test_runner_cli_end_to_end_with_tiny_config(tmp_path: Path) -> None:
    config_path = tmp_path / "tiny.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: tiny_e2e",
                f"output_root: {tmp_path.as_posix()}",
                "data:",
                "  manifold: circle",
                "  n_samples: 60",
                "  ambient_dim: 3",
                "  noise_std: 0.03",
                "  seed: 7",
                "graph:",
                "  epsilon: 0.8",
                "  epsilon_grid: [0.5, 0.8, 1.2]",
                "  use_knn: false",
                "spectral:",
                "  n_eigs: 8",
                "cutoff:",
                "  method: bandwidth_stability",
                "  eigengap_min_k: 1",
                "  eigengap_max_k: 5",
                "  stability_min_k: 1",
                "  stability_max_k: 5",
                "  stability_threshold: 0.8",
                "plot:",
                "  formats: [png]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    runner = CliRunner()
    completed = runner.invoke(app, ["run", "--config", str(config_path)])
    assert completed.exit_code == 0, completed.stdout
    assert "Run directory:" in completed.stdout

    run_dirs = [p for p in tmp_path.iterdir() if p.is_dir() and p.name.endswith("tiny_e2e")]
    assert run_dirs, "Expected a run directory to be created for tiny_e2e."
    run_dir = run_dirs[0]
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "cutoffs.json").exists()
    assert (run_dir / "arrays.npz").exists()
