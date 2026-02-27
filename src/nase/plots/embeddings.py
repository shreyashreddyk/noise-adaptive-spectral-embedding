from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_embedding_2d(embedding: np.ndarray, colour: np.ndarray, out_path: Path, dpi: int) -> None:
    if embedding.shape[1] == 1:
        y = np.zeros(embedding.shape[0], dtype=float)
    else:
        y = embedding[:, 1]

    fig, ax = plt.subplots(figsize=(6, 5))
    scatter = ax.scatter(embedding[:, 0], y, c=colour, s=8, cmap="viridis")
    ax.set_title("Diffusion-map embedding")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    fig.colorbar(scatter, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
