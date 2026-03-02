from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from nase.experiments.configs import config_from_dict, config_to_dict, load_config
from nase.experiments.io import make_run_dir, write_json
from nase.experiments.runner import run_experiment
from nase.plots.ablations import plot_case_metric_comparison


@dataclass(slots=True)
class SweepResult:
    sweep_dir: Path
    records: list[dict[str, Any]]


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"cases": []}
    frame = pd.DataFrame.from_records(records)
    grouped = frame.groupby("case", as_index=False).agg(
        selected_k_mean=("selected_k", "mean"),
        selected_k_std=("selected_k", "std"),
        trustworthiness_mean=("trustworthiness", "mean"),
        continuity_mean=("continuity", "mean"),
        geodesic_consistency_mean=("geodesic_consistency", "mean"),
    )
    grouped = grouped.fillna(0.0)
    return {"cases": grouped.to_dict(orient="records")}


def run_sweep(config_path: Path) -> SweepResult:
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    name = str(raw.get("name", "synthetic_sweep"))
    output_root = Path(raw.get("output_root", "results"))
    base_config_path = (config_path.parent / raw["base_config"]).resolve()
    seeds: list[int] = list(raw.get("seeds", [42]))
    cases: list[dict[str, Any]] = list(raw.get("cases", []))

    base_config = load_config(base_config_path)
    base_dict = config_to_dict(base_config)
    sweep_dir = make_run_dir(output_root=output_root, run_name=name)

    records: list[dict[str, Any]] = []
    for case in cases:
        case_name = str(case.get("name", "case"))
        overrides = dict(case.get("overrides", {}))
        for seed in seeds:
            candidate = _deep_merge(base_dict, overrides)
            candidate["name"] = f"{name}_{case_name}_seed{seed}"
            candidate["output_root"] = str(sweep_dir / "runs")
            candidate.setdefault("data", {})
            candidate["data"]["seed"] = int(seed)

            config = config_from_dict(candidate)
            result = run_experiment(config)
            metrics = result.metrics
            record: dict[str, Any] = {
                "case": case_name,
                "seed": seed,
                "run_dir": str(result.run_dir),
                "selected_k": metrics["selected_k"],
                "k_eigengap": metrics["k_eigengap"],
                "k_bandwidth_stability": metrics["k_bandwidth_stability"],
                "trustworthiness": metrics["trustworthiness"],
                "continuity": metrics["continuity"],
                "geodesic_consistency": metrics["geodesic_consistency"],
            }
            if "estimated_intrinsic_dim_noisy" in metrics:
                record["estimated_intrinsic_dim_noisy"] = metrics["estimated_intrinsic_dim_noisy"]
                record["k_intrinsic_dim"] = metrics["k_intrinsic_dim"]
            records.append(record)

    aggregate = _aggregate(records)
    frame = pd.DataFrame.from_records(records)
    frame.to_csv(sweep_dir / "records.csv", index=False)
    write_json(sweep_dir / "records.json", {"records": records})
    write_json(sweep_dir / "aggregate.json", aggregate)
    if aggregate["cases"]:
        labels = [row["case"] for row in aggregate["cases"]]
        values = [float(row["selected_k_mean"]) for row in aggregate["cases"]]
        plot_case_metric_comparison(
            labels=labels,
            values=values,
            metric_name="Mean selected k*",
            out_path=sweep_dir / "selected_k_comparison.png",
            dpi=160,
        )
    write_json(
        sweep_dir / "manifest.json",
        {"name": name, "base_config": str(base_config_path), "seeds": seeds, "cases": cases},
    )
    return SweepResult(sweep_dir=sweep_dir, records=records)
