from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from nase.config import (
    CutoffConfig,
    DataConfig,
    ExperimentConfig,
    GraphConfig,
    PlotConfig,
    SpectralConfig,
)


def _build_config(raw: dict[str, Any]) -> ExperimentConfig:
    return ExperimentConfig(
        name=raw.get("name", "nase_experiment"),
        output_root=Path(raw.get("output_root", "results")),
        data=DataConfig(**raw.get("data", {})),
        graph=GraphConfig(**raw.get("graph", {})),
        spectral=SpectralConfig(**raw.get("spectral", {})),
        cutoff=CutoffConfig(**raw.get("cutoff", {})),
        plot=PlotConfig(**raw.get("plot", {})),
    )


def _validate_config(config: ExperimentConfig) -> None:
    if config.data.n_samples <= 0:
        raise ValueError("data.n_samples must be > 0")
    if config.data.ambient_dim <= 0:
        raise ValueError("data.ambient_dim must be > 0")
    if config.data.noise_std < 0:
        raise ValueError("data.noise_std must be >= 0")

    if config.graph.epsilon <= 0:
        raise ValueError("graph.epsilon must be > 0")
    if not config.graph.epsilon_grid:
        raise ValueError("graph.epsilon_grid must not be empty")
    if any(e <= 0 for e in config.graph.epsilon_grid):
        raise ValueError("graph.epsilon_grid values must be > 0")

    if config.spectral.n_eigs < 2:
        raise ValueError("spectral.n_eigs must be >= 2")
    if config.spectral.diffusion_time < 1:
        raise ValueError("spectral.diffusion_time must be >= 1")

    if config.cutoff.eigengap_min_k <= 0 or config.cutoff.stability_min_k <= 0:
        raise ValueError("cutoff minimum k values must be >= 1")
    if config.cutoff.eigengap_min_k > config.cutoff.eigengap_max_k:
        raise ValueError("cutoff eigengap_min_k must be <= eigengap_max_k")
    if config.cutoff.stability_min_k > config.cutoff.stability_max_k:
        raise ValueError("cutoff stability_min_k must be <= stability_max_k")
    if not (0.0 <= config.cutoff.stability_threshold <= 1.0):
        raise ValueError("cutoff.stability_threshold must be in [0, 1]")
    if config.cutoff.method not in {"eigengap", "bandwidth_stability"}:
        raise ValueError("cutoff.method must be 'eigengap' or 'bandwidth_stability'")

    if config.plot.dpi <= 0:
        raise ValueError("plot.dpi must be > 0")
    valid_formats = {"png", "svg"}
    unknown_formats = [f for f in config.plot.formats if f not in valid_formats]
    if unknown_formats:
        unknown = ", ".join(unknown_formats)
        raise ValueError(f"Unsupported plot formats: {unknown}")


def load_config(path: Path) -> ExperimentConfig:
    suffix = path.suffix.lower()
    with path.open("r", encoding="utf-8") as handle:
        if suffix == ".json":
            raw = json.load(handle) or {}
        else:
            raw = yaml.safe_load(handle) or {}

    config = _build_config(raw)
    _validate_config(config)
    return config


def config_to_dict(config: ExperimentConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["output_root"] = str(config.output_root)
    return payload


def config_from_dict(raw: dict[str, Any]) -> ExperimentConfig:
    config = _build_config(raw)
    _validate_config(config)
    return config
