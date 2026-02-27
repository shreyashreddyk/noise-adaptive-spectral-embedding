from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from nase.experiments.configs import load_config
from nase.experiments.runner import run_experiment
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


@app.command("sweep")
def sweep(config: Annotated[Path, typer.Option(..., exists=True, readable=True)]) -> None:
    """Run a multi-seed synthetic sweep and write aggregate artefacts."""
    result = run_sweep(config_path=config)
    typer.echo(f"Sweep directory: {result.sweep_dir}")
    typer.echo(f"Runs executed: {len(result.records)}")


if __name__ == "__main__":
    app()
