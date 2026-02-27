from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def plot_cutoff_ablation(
    method_to_score: dict[str, float], metric_name: str, out_path: Path, dpi: int
) -> None:
    labels = list(method_to_score)
    values = [method_to_score[k] for k in labels]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, values)
    ax.set_ylabel(metric_name)
    ax.set_title(f"Ablation: {metric_name}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_case_metric_comparison(
    labels: list[str], values: list[float], metric_name: str, out_path: Path, dpi: int
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, values)
    ax.set_ylabel(metric_name)
    ax.set_title(f"Sweep comparison: {metric_name}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
