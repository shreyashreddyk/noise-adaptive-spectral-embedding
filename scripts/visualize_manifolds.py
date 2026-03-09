"""Generate publication-quality manifold visualizations at varying noise levels.

Produces:
  - Individual PNGs per manifold per noise level (for LaTeX includegraphics)
  - Panel figures per manifold showing clean → increasing noise side-by-side

All data is generated fresh using the same parameters and seeds as the experiments.
PCA coordinates are fit on the clean data so that noise effects are visually
comparable across panels without orientation drift.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from nase.data.synthetic import generate_synthetic

OUTPUT_DIR = Path("results/analysis/manifold_visualizations")

MANIFOLD_CONFIGS = {
    "circle": {
        "n": 450,
        "ambient_dim": 3,
        "noise_levels": [0.0, 0.02, 0.08, 0.16],
        "seed": 11,
        "pca_dim": 2,
    },
    "sphere": {
        "n": 450,
        "ambient_dim": 5,
        "noise_levels": [0.0, 0.02, 0.08, 0.16],
        "seed": 11,
        "pca_dim": 3,
    },
    "swiss_roll": {
        "n": 450,
        "ambient_dim": 3,
        "noise_levels": [0.0, 0.03, 0.08, 0.16],
        "seed": 11,
        "pca_dim": 3,
    },
    "s_curve": {
        "n": 450,
        "ambient_dim": 3,
        "noise_levels": [0.0, 0.04, 0.08, 0.16],
        "seed": 11,
        "pca_dim": 3,
    },
    "torus": {
        "n": 450,
        "ambient_dim": 5,
        "noise_levels": [0.0, 0.02, 0.08, 0.16],
        "seed": 11,
        "pca_dim": 3,
    },
}

DISPLAY_NAMES = {
    "circle": "Circle",
    "sphere": "Sphere",
    "swiss_roll": "Swiss Roll",
    "s_curve": "S-Curve",
    "torus": "Torus",
}


def _get_latent_color(latent_params: dict[str, np.ndarray]) -> np.ndarray:
    if "theta" in latent_params:
        return latent_params["theta"].ravel()
    return list(latent_params.values())[0].ravel()


def _plot_2d(ax: plt.Axes, points: np.ndarray, color: np.ndarray, title: str) -> None:
    ax.scatter(
        points[:, 0], points[:, 1],
        c=color, s=3, alpha=0.7, cmap="Spectral", rasterized=True,
    )
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_aspect("equal")
    ax.tick_params(labelsize=7)


def _plot_3d(ax: plt.Axes, points: np.ndarray, color: np.ndarray, title: str) -> None:
    ax.scatter(
        points[:, 0], points[:, 1], points[:, 2],
        c=color, s=3, alpha=0.6, cmap="Spectral", rasterized=True,
    )
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.tick_params(labelsize=6)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False


def visualize_manifold(manifold: str, cfg: dict) -> None:
    n = cfg["n"]
    ambient_dim = cfg["ambient_dim"]
    noise_levels = cfg["noise_levels"]
    seed = cfg["seed"]
    pca_dim = cfg["pca_dim"]
    display_name = DISPLAY_NAMES[manifold]

    # Generate clean data (r=0 gives X_noisy = X_clean) for PCA fitting.
    # Use a tiny r for clean to keep the same RNG path, but we only use X_clean.
    clean_data = generate_synthetic(manifold, n, ambient_dim, r=0.0, seed=seed)
    latent_color = _get_latent_color(clean_data.latent_params)

    pca = PCA(n_components=pca_dim)
    pca.fit(clean_data.X_clean)

    datasets = []
    for r in noise_levels:
        data = generate_synthetic(manifold, n, ambient_dim, r=r, seed=seed)
        if r == 0.0:
            projected = pca.transform(data.X_clean)
        else:
            projected = pca.transform(data.X_noisy)
        datasets.append((r, projected))

    # --- Individual plots ---
    for r, projected in datasets:
        if pca_dim == 2:
            fig, ax = plt.subplots(figsize=(5, 5))
            label = "Clean" if r == 0.0 else f"r = {r}"
            _plot_2d(ax, projected, latent_color, f"{display_name} — {label}")
        else:
            fig = plt.figure(figsize=(5, 5))
            ax = fig.add_subplot(111, projection="3d")
            label = "Clean" if r == 0.0 else f"r = {r}"
            _plot_3d(ax, projected, latent_color, f"{display_name} — {label}")

        fig.tight_layout()
        suffix = "clean" if r == 0.0 else f"r{r:.2f}".replace(".", "p")
        fig.savefig(OUTPUT_DIR / f"{manifold}_{suffix}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    # --- Panel figure ---
    n_panels = len(noise_levels)
    if pca_dim == 2:
        fig, axes = plt.subplots(1, n_panels, figsize=(4.2 * n_panels, 4.2))
    else:
        fig = plt.figure(figsize=(4.5 * n_panels, 4.5))
        axes = [fig.add_subplot(1, n_panels, i + 1, projection="3d") for i in range(n_panels)]

    for ax, (r, projected) in zip(axes, datasets, strict=True):
        label = "Clean" if r == 0.0 else f"r = {r}"
        if pca_dim == 2:
            _plot_2d(ax, projected, latent_color, label)
        else:
            _plot_3d(ax, projected, latent_color, label)

    fig.suptitle(
        f"{display_name} (n={n}, D={ambient_dim})",
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / f"{manifold}_noise_panel.png",
        dpi=200, bbox_inches="tight",
    )
    plt.close(fig)
    print(f"  {display_name}: {n_panels} individual + 1 panel saved")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=== Generating Manifold Visualizations ===\n")

    for manifold, cfg in MANIFOLD_CONFIGS.items():
        visualize_manifold(manifold, cfg)

    print(f"\nAll plots saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
