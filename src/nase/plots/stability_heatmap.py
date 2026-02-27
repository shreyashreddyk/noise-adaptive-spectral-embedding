from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_stability_heatmap(
    matrix: np.ndarray, epsilon_grid: list[float], out_path: Path, dpi: int
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="viridis")
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
