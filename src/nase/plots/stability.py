from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np


def _sorted_xy(scores: Mapping[int, float]) -> tuple[np.ndarray, np.ndarray]:
    ks = np.array(sorted(int(k) for k in scores), dtype=int)
    vals = np.array([float(scores[int(k)]) for k in ks], dtype=float)
    return ks, vals


def plot_stability_scores(
    scores: Mapping[int, float],
    out_path: Path,
    dpi: int,
    selected_k: int | None = None,
    threshold: float | None = None,
) -> None:
    ks, vals = _sorted_xy(scores)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ks, vals, marker="o")
    ax.set_title("Bandwidth stability by mode")
    ax.set_xlabel("k")
    ax.set_ylabel("Stability")
    ax.set_ylim(0.0, 1.05)
    if threshold is not None:
        ax.axhline(float(threshold), color="tab:orange", linestyle="--", linewidth=1.0)
    if selected_k is not None and int(selected_k) in set(ks.tolist()):
        sk = int(selected_k)
        sval = float(vals[np.where(ks == sk)[0][0]])
        ax.axvline(sk, color="tab:red", linestyle="--", linewidth=1.0)
        ax.scatter([sk], [sval], color="tab:red", zorder=3)
        ax.annotate(
            f"k*={sk}",
            xy=(sk, sval),
            xytext=(6, 8),
            textcoords="offset points",
            color="tab:red",
        )
    ax.grid(alpha=0.25, linestyle=":")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_stability_curves(
    curves: Mapping[str, Mapping[int, float]],
    out_path: Path,
    dpi: int,
    selected_k: int | None = None,
    threshold: float | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, score_map in curves.items():
        ks, vals = _sorted_xy(score_map)
        ax.plot(ks, vals, marker="o", linewidth=1.5, label=label)
    ax.set_title("Bandwidth stability curves")
    ax.set_xlabel("k")
    ax.set_ylabel("Stability")
    ax.set_ylim(0.0, 1.05)
    if threshold is not None:
        ax.axhline(float(threshold), color="tab:orange", linestyle="--", linewidth=1.0)
    if selected_k is not None:
        ax.axvline(int(selected_k), color="tab:red", linestyle="--", linewidth=1.0)
        ax.annotate(
            f"k*={int(selected_k)}",
            xy=(int(selected_k), 1.0),
            xytext=(6, -10),
            textcoords="offset points",
            color="tab:red",
        )
    ax.grid(alpha=0.25, linestyle=":")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_stability_heatmap(
    matrix: np.ndarray, epsilon_grid: list[float], out_path: Path, dpi: int
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(np.asarray(matrix, dtype=float), vmin=0.0, vmax=1.0, cmap="viridis")
    labels = [f"{eps:.2g}" for eps in epsilon_grid]
    ticks = np.arange(len(labels))
    ax.set_xticks(ticks, labels=labels)
    ax.set_yticks(ticks, labels=labels)
    ax.set_xlabel("Bandwidth epsilon")
    ax.set_ylabel("Bandwidth epsilon")
    ax.set_title("Cross-bandwidth stability heatmap")
    fig.colorbar(im, ax=ax, label="Mean mode alignment")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
