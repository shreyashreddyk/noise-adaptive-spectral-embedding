from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

def _valid_selected_k(selected_k: int | None, n_eigs: int) -> int | None:
    if selected_k is None:
        return None
    if 1 <= int(selected_k) <= max(1, n_eigs - 1):
        return int(selected_k)
    return None


def plot_spectrum(
    eigenvalues: np.ndarray,
    out_path: Path,
    dpi: int,
    selected_k: int | None = None,
) -> None:
    evals = np.asarray(eigenvalues, dtype=float).reshape(-1)
    if evals.size == 0:
        raise ValueError("`eigenvalues` must contain at least one value.")
    chosen_k = _valid_selected_k(selected_k, evals.size)

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(evals.size)
    ax.plot(x, evals, marker="o")
    ax.set_title("Scree plot (eigenvalue spectrum)")
    ax.set_xlabel("Eigenvalue index")
    ax.set_ylabel("Eigenvalue")
    if chosen_k is not None and chosen_k < evals.size:
        ax.axvline(chosen_k, color="tab:red", linestyle="--", linewidth=1.0)
        ax.scatter([chosen_k], [evals[chosen_k]], color="tab:red", zorder=3)
        ax.annotate(
            f"k*={chosen_k}",
            xy=(chosen_k, evals[chosen_k]),
            xytext=(6, 8),
            textcoords="offset points",
            color="tab:red",
        )
    ax.grid(alpha=0.25, linestyle=":")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_eigengap(
    eigenvalues: np.ndarray,
    out_path: Path,
    dpi: int,
    selected_k: int | None = None,
    min_k: int = 1,
    max_k: int | None = None,
) -> None:
    evals = np.asarray(eigenvalues, dtype=float).reshape(-1)
    if evals.size < 2:
        raise ValueError("`eigenvalues` must contain at least two values for an eigengap plot.")

    upper = evals.size - 2 if max_k is None else min(int(max_k), evals.size - 2)
    lower = max(1, int(min_k))
    if lower > upper:
        lower = upper
    ks = np.arange(lower, upper + 1)
    gaps = evals[ks] - evals[ks + 1]
    chosen_k = _valid_selected_k(selected_k, evals.size)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ks, gaps, marker="o")
    ax.set_title("Eigengap plot")
    ax.set_xlabel("k")
    ax.set_ylabel(r"$\lambda_k - \lambda_{k+1}$")
    if chosen_k is not None and lower <= chosen_k <= upper:
        chosen_gap = float(evals[chosen_k] - evals[chosen_k + 1])
        ax.axvline(chosen_k, color="tab:red", linestyle="--", linewidth=1.0)
        ax.scatter([chosen_k], [chosen_gap], color="tab:red", zorder=3)
        ax.annotate(
            f"k*={chosen_k}",
            xy=(chosen_k, chosen_gap),
            xytext=(6, 8),
            textcoords="offset points",
            color="tab:red",
        )
    ax.grid(alpha=0.25, linestyle=":")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
