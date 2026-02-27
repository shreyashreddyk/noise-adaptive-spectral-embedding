from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def _prepare_embedding(embedding: np.ndarray, dims: int) -> np.ndarray:
    emb = np.asarray(embedding, dtype=float)
    if emb.ndim != 2:
        raise ValueError("`embedding` must be a 2D array.")
    if emb.shape[1] < dims:
        pad = np.zeros((emb.shape[0], dims - emb.shape[1]), dtype=float)
        emb = np.hstack([emb, pad])
    return emb[:, :dims]


def _is_numeric(colour: np.ndarray) -> bool:
    return np.issubdtype(colour.dtype, np.number)


def _scatter_colour_2d(
    ax: plt.Axes,
    fig: plt.Figure,
    x: np.ndarray,
    y: np.ndarray,
    colour: np.ndarray,
    colorbar_label: str,
) -> None:
    if _is_numeric(colour):
        raw = np.asarray(colour).reshape(-1)
        numeric = raw.astype(float)
        unique = np.unique(numeric)
        if np.issubdtype(raw.dtype, np.integer) and unique.size <= 12:
            classes = unique
            cmap = plt.get_cmap("tab10")
            for idx, cls in enumerate(classes):
                mask = numeric == cls
                ax.scatter(x[mask], y[mask], s=10, color=cmap(idx % 10), label=str(int(cls)))
            ax.legend(loc="best", title="Class")
            return
        scatter = ax.scatter(x, y, c=numeric, s=10, cmap="viridis")
        fig.colorbar(scatter, ax=ax, label=colorbar_label)
        return

    labels, inverse = np.unique(colour.astype(str), return_inverse=True)
    cmap = plt.get_cmap("tab10")
    for idx, label in enumerate(labels):
        mask = inverse == idx
        ax.scatter(x[mask], y[mask], s=10, color=cmap(idx % 10), label=str(label))
    ax.legend(loc="best", title="Class")


def _scatter_colour_3d(
    ax: Axes3D,
    fig: plt.Figure,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    colour: np.ndarray,
    colorbar_label: str,
) -> None:
    if _is_numeric(colour):
        raw = np.asarray(colour).reshape(-1)
        numeric = raw.astype(float)
        unique = np.unique(numeric)
        if np.issubdtype(raw.dtype, np.integer) and unique.size <= 12:
            classes = unique
            cmap = plt.get_cmap("tab10")
            for idx, cls in enumerate(classes):
                mask = numeric == cls
                ax.scatter(x[mask], y[mask], z[mask], s=10, color=cmap(idx % 10), label=str(int(cls)))
            ax.legend(loc="best", title="Class")
            return
        scatter = ax.scatter(x, y, z, c=numeric, s=10, cmap="viridis")
        fig.colorbar(scatter, ax=ax, label=colorbar_label, pad=0.1)
        return

    labels, inverse = np.unique(colour.astype(str), return_inverse=True)
    cmap = plt.get_cmap("tab10")
    for idx, label in enumerate(labels):
        mask = inverse == idx
        ax.scatter(x[mask], y[mask], z[mask], s=10, color=cmap(idx % 10), label=str(label))
    ax.legend(loc="best", title="Class")


def plot_embedding_2d(
    embedding: np.ndarray,
    colour: np.ndarray,
    out_path: Path,
    dpi: int,
    colorbar_label: str = "Latent parameter",
) -> None:
    emb = _prepare_embedding(embedding, dims=2)
    col = np.asarray(colour).reshape(-1)
    fig, ax = plt.subplots(figsize=(6, 5))
    _scatter_colour_2d(ax=ax, fig=fig, x=emb[:, 0], y=emb[:, 1], colour=col, colorbar_label=colorbar_label)
    ax.set_title("Diffusion-map embedding (2D)")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_embedding_3d(
    embedding: np.ndarray,
    colour: np.ndarray,
    out_path: Path,
    dpi: int,
    colorbar_label: str = "Latent parameter",
) -> None:
    emb = _prepare_embedding(embedding, dims=3)
    col = np.asarray(colour).reshape(-1)
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")
    _scatter_colour_3d(
        ax=ax,
        fig=fig,
        x=emb[:, 0],
        y=emb[:, 1],
        z=emb[:, 2],
        colour=col,
        colorbar_label=colorbar_label,
    )
    ax.set_title("Diffusion-map embedding (3D)")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_zlabel("Component 3")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
