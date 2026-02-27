from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from nase.experiments.configs import load_config
from nase.experiments.runner import regenerate_plots_from_run_dir, run_experiment
from nase.experiments.sweeps import run_sweep

app = typer.Typer(help="NASE experiment CLI.")


@app.command("run")
def run(config: Annotated[Path, typer.Option(..., exists=True, readable=True)]) -> None:
    """Run a config-driven experiment and write artefacts."""
    cfg = load_config(config)
    result = run_experiment(cfg)
    typer.echo(f"Run directory: {result.run_dir}")
    typer.echo(f"Selected k*: {result.selected_k}")
    typer.echo(json.dumps(result.metrics, indent=2))


@app.command("plot")
def plot(run_dir: Annotated[Path, typer.Option(..., exists=True, file_okay=False)]) -> None:
    """Show where plot artefacts are located for a run."""
    plots_dir = run_dir / "plots"
    if not plots_dir.exists():
        raise typer.BadParameter(f"No plots directory found under {run_dir}")
    files = sorted(p.name for p in plots_dir.iterdir() if p.is_file())
    typer.echo("\n".join(files))


@app.command("plot-from-results")
def plot_from_results(
    run_dir: Annotated[Path, typer.Option(..., exists=True, file_okay=False)],
    output_dir: Annotated[Path | None, typer.Option(file_okay=False)] = None,
    dpi: Annotated[int | None, typer.Option(min=1)] = None,
    formats: Annotated[
        str | None, typer.Option(help="Comma-separated list from: png,svg")
    ] = None,
) -> None:
    """Regenerate plot artefacts from a saved run directory."""
    selected_formats: list[str] | None = None
    if formats is not None:
        selected_formats = [f.strip().lower() for f in formats.split(",") if f.strip()]
        unsupported = [fmt for fmt in selected_formats if fmt not in {"png", "svg"}]
        if unsupported:
            raise typer.BadParameter(f"Unsupported formats: {', '.join(sorted(set(unsupported)))}")
    target_dir = regenerate_plots_from_run_dir(
        run_dir=run_dir, output_dir=output_dir, dpi=dpi, formats=selected_formats
    )
    typer.echo(f"Plots written to: {target_dir}")


@app.command("sweep")
def sweep(config: Annotated[Path, typer.Option(..., exists=True, readable=True)]) -> None:
    """Run a multi-seed synthetic sweep and write aggregate artefacts."""
    result = run_sweep(config_path=config)
    typer.echo(f"Sweep directory: {result.sweep_dir}")
    typer.echo(f"Runs executed: {len(result.records)}")


if __name__ == "__main__":
    app()
