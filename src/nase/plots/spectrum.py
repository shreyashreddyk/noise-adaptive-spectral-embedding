from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_spectrum(eigenvalues: np.ndarray, out_path: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.arange(len(eigenvalues)), eigenvalues, marker="o")
    ax.set_title("Eigenvalue spectrum")
    ax.set_xlabel("Index")
    ax.set_ylabel("Eigenvalue")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
