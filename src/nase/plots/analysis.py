from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np


def plot_oracle_scaling(
    manifold_data: dict[str, list[tuple[float, float]]],
    out_path: Path,
    dpi: int = 160,
) -> dict[str, float]:
    """Plot k_oracle vs 1/r^2 for each manifold. Returns estimated C per manifold.

    manifold_data: {manifold_name: [(r, k_oracle), ...]}
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    c_values: dict[str, float] = {}

    for manifold, pairs in manifold_data.items():
        rs = np.array([p[0] for p in pairs])
        ks = np.array([p[1] for p in pairs])
        inv_r2 = 1.0 / (rs**2)
        ax.scatter(inv_r2, ks, label=manifold, s=50, zorder=3)

        c_per_point = ks * rs**2
        c_values[manifold] = float(np.mean(c_per_point))

        if len(inv_r2) >= 2:
            x_fit = np.linspace(inv_r2.min() * 0.8, inv_r2.max() * 1.1, 50)
            y_fit = c_values[manifold] * x_fit
            ax.plot(x_fit, y_fit, "--", alpha=0.5, linewidth=1.5)

    ax.set_xlabel(r"$1 / r^2$")
    ax.set_ylabel(r"$k_{\mathrm{oracle}}$")
    ax.set_title(r"Oracle cutoff vs $1/r^2$ (testing $k^* \approx C/r^2$)")
    ax.legend()
    ax.grid(alpha=0.25, linestyle=":")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return c_values


def plot_method_comparison_across_noise(
    data: dict[str, dict[str, list[float]]],
    noise_levels: list[float],
    out_path: Path,
    dpi: int = 160,
    title: str = "Cutoff comparison across noise levels",
) -> None:
    """Bar chart comparing k* from different methods across noise levels.

    data: {method_name: {noise_label: [k values across seeds]}}
    """
    methods = list(data.keys())
    n_noise = len(noise_levels)
    n_methods = len(methods)
    x = np.arange(n_noise)
    width = 0.8 / n_methods

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, method in enumerate(methods):
        means = []
        stds = []
        for nl in noise_levels:
            label = f"{nl}"
            vals = data[method].get(label, [0])
            means.append(np.mean(vals))
            stds.append(np.std(vals))
        offset = (i - n_methods / 2 + 0.5) * width
        ax.bar(x + offset, means, width, yerr=stds, label=method, capsize=3)

    ax.set_xlabel("Noise level r")
    ax.set_ylabel("Selected k*")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([str(nl) for nl in noise_levels])
    ax.legend()
    ax.grid(alpha=0.25, linestyle=":", axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_metrics_across_noise(
    noise_levels: list[float],
    metrics: dict[str, list[float]],
    out_path: Path,
    dpi: int = 160,
    title: str = "Embedding quality vs noise",
) -> None:
    """Line plot of quality metrics across noise levels."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for metric_name, values in metrics.items():
        ax.plot(noise_levels, values, marker="o", linewidth=1.5, label=metric_name)
    ax.set_xlabel("Noise level r")
    ax.set_ylabel("Metric value")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.25, linestyle=":")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_oracle_subspace_profile(
    distance_profiles: dict[str, dict[int, float]],
    out_path: Path,
    dpi: int = 160,
    title: str = "Oracle subspace distance by k",
) -> None:
    """Plot subspace distance d(k) vs k for multiple runs/conditions."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, profile in distance_profiles.items():
        ks = sorted(profile.keys())
        vals = [profile[k] for k in ks]
        ax.plot(ks, vals, marker="o", linewidth=1.5, label=label)
    ax.set_xlabel("k (number of retained dimensions)")
    ax.set_ylabel("Subspace distance (chordal)")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.25, linestyle=":")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_r_estimation_accuracy(
    r_true: list[float],
    r_estimated: list[float],
    manifold_labels: list[str],
    out_path: Path,
    dpi: int = 160,
) -> None:
    """Scatter plot of r_estimated vs r_true with identity line."""
    fig, ax = plt.subplots(figsize=(6, 6))
    unique_manifolds = sorted(set(manifold_labels))
    for manifold in unique_manifolds:
        mask = [m == manifold for m in manifold_labels]
        rt = [r_true[i] for i in range(len(r_true)) if mask[i]]
        re = [r_estimated[i] for i in range(len(r_estimated)) if mask[i]]
        ax.scatter(rt, re, label=manifold, s=50, zorder=3)

    all_vals = r_true + r_estimated
    lo, hi = min(all_vals) * 0.8, max(all_vals) * 1.2
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.4, linewidth=1, label="identity")
    ax.set_xlabel(r"$r_{\mathrm{true}}$")
    ax.set_ylabel(r"$\hat{r}$")
    ax.set_title("Noise amplitude estimation accuracy")
    ax.legend()
    ax.set_aspect("equal")
    ax.grid(alpha=0.25, linestyle=":")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_threshold_sensitivity(
    thresholds: list[float],
    k_means: list[float],
    k_stds: list[float],
    out_path: Path,
    dpi: int = 160,
) -> None:
    """Line plot of k* vs stability threshold tau."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(thresholds, k_means, yerr=k_stds, marker="o", capsize=4, linewidth=1.5)
    ax.set_xlabel(r"Stability threshold $\tau$")
    ax.set_ylabel("Selected k*")
    ax.set_title(r"Sensitivity of $k^*$ to threshold $\tau$ (circle, $r=0.08$)")
    ax.grid(alpha=0.25, linestyle=":")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
