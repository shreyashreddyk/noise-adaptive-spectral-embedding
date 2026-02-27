from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_stability_scores(scores: dict[int, float], out_path: Path, dpi: int) -> None:
    ks = np.array(sorted(scores))
    vals = np.array([scores[k] for k in ks])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ks, vals, marker="o")
    ax.set_title("Bandwidth stability by mode")
    ax.set_xlabel("k")
    ax.set_ylabel("Stability")
    ax.set_ylim(0.0, 1.05)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
