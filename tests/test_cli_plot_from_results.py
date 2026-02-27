from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from nase.cli import app
from nase.config import ExperimentConfig
from nase.experiments.runner import run_experiment


def test_plot_from_results_regenerates_plots(tmp_path: Path) -> None:
    cfg = ExperimentConfig()
    cfg.output_root = tmp_path
    cfg.plot.formats = ["png"]
    cfg.data.n_samples = 80
    cfg.spectral.n_eigs = 8
    cfg.cutoff.eigengap_max_k = 5
    cfg.cutoff.stability_max_k = 5
    result = run_experiment(cfg)

    regen_dir = result.run_dir / "figures_regen"
    runner = CliRunner()
    completed = runner.invoke(
        app,
        [
            "plot",
            "--run-dir",
            str(result.run_dir),
            "--output-dir",
            str(regen_dir),
            "--formats",
            "png",
        ],
    )
    assert completed.exit_code == 0, completed.stdout
    expected = {
        "spectrum.png",
        "eigengap.png",
        "stability.png",
        "stability_heatmap.png",
        "embedding.png",
        "embedding_3d.png",
        "ablation_cutoff.png",
    }
    observed = {p.name for p in regen_dir.iterdir() if p.is_file()}
    assert expected.issubset(observed)
