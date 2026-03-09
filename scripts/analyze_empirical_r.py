"""Compute empirical noise amplitude from synthetic data and compare with kNN estimates.

Since we generate synthetic manifolds with known additive Gaussian noise, we have
access to both X_clean and X_noisy. The noise vector is exactly (X_noisy - X_clean),
so we can compute the *realized* noise standard deviation directly:

    r_empirical = std(X_noisy - X_clean)

This gives a tighter ground truth than the configured noise_std parameter, which only
describes the distribution the noise was drawn from. For finite samples (n=450, D=3-5),
r_empirical will be close to r_configured but not identical.

We compare three quantities:
  - r_configured : the noise_std parameter in the YAML config
  - r_empirical  : std of the actual noise vector in each specific run
  - r_knn        : kNN-based estimates from noisy data alone

The gap between r_configured and r_empirical quantifies finite-sample fluctuation.
The gap between r_empirical and r_knn quantifies the estimation bias introduced by
conflating manifold geometry with noise.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

from nase.estimators.noise_amplitude import (
    estimate_noise_amplitude_simple,
    estimate_noise_amplitude_twoscale,
)

RESULTS_ROOT = Path("results")
ANALYSIS_DIR = Path("results/analysis")
OUTPUT_DIR = Path("results/analysis/empirical_r")


def _find_sweep(pattern: str) -> Path | None:
    for d in sorted(RESULTS_ROOT.iterdir()):
        if d.is_dir() and pattern in d.name:
            return d
    return None


def _load_run_data(run_dir: Path) -> dict:
    """Load arrays and metrics from a single experiment run."""
    arrays_path = run_dir / "arrays.npz"
    metrics_path = run_dir / "metrics.json"
    if not arrays_path.exists() or not metrics_path.exists():
        return {}
    arrays = np.load(arrays_path)
    metrics = json.loads(metrics_path.read_text())
    x_clean = arrays["x_clean"]
    x_noisy = arrays["x_noisy"]
    noise = x_noisy - x_clean
    r_empirical = float(np.std(noise))
    r_empirical_rms = float(np.sqrt(np.mean(noise**2)))
    return {
        "x_clean": x_clean,
        "x_noisy": x_noisy,
        "noise": noise,
        "r_empirical": r_empirical,
        "r_empirical_rms": r_empirical_rms,
        "metrics": metrics,
    }


def analyze_r_estimation_runs() -> list[dict]:
    """Walk through r-estimation experiment runs, compute empirical r, compare."""
    rows = []

    for sweep_pattern in [
        "r_estimation_circle_sphere",
        "r_estimation_swiss_roll",
    ]:
        sweep_dir = _find_sweep(sweep_pattern)
        if not sweep_dir:
            print(f"  Sweep not found: {sweep_pattern}")
            continue

        runs_dir = sweep_dir / "runs"
        if not runs_dir.exists():
            continue

        for run_dir in sorted(runs_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            data = _load_run_data(run_dir)
            if not data:
                continue

            metrics = data["metrics"]
            r_configured = metrics.get("known_noise_r", 0.0)

            r_knn_simple = metrics.get("r_estimated_simple")
            r_knn_twoscale = metrics.get("r_estimated_twoscale")

            if r_knn_simple is None:
                r_knn_simple = float(
                    estimate_noise_amplitude_simple(data["x_noisy"], k=2)
                )
            if r_knn_twoscale is None:
                r_knn_twoscale = float(
                    estimate_noise_amplitude_twoscale(data["x_noisy"], k1=1, k2=4)
                )

            manifold = metrics.get("metadata", {}).get("manifold", "unknown")
            seed = metrics.get("metadata", {}).get("seed", 0)
            n_samples, ambient_dim = data["x_clean"].shape

            rows.append({
                "manifold": manifold,
                "seed": seed,
                "n": n_samples,
                "D": ambient_dim,
                "r_configured": r_configured,
                "r_empirical": data["r_empirical"],
                "r_empirical_rms": data["r_empirical_rms"],
                "r_knn_simple": r_knn_simple,
                "r_knn_twoscale": r_knn_twoscale,
            })

    return rows


def also_analyze_noise_sweep_runs() -> list[dict]:
    """Compute empirical r for the main noise-sweep experiments too.

    These experiments used bandwidth-stability (not r_based), so they don't
    have kNN estimates stored in metrics. We compute them fresh here.
    """
    rows = []
    for sweep_pattern in [
        "noise_sweep_circle_sphere",
        "synthetic_noise_sweep",
        "torus_noise_sweep",
    ]:
        sweep_dir = _find_sweep(sweep_pattern)
        if not sweep_dir:
            continue
        runs_dir = sweep_dir / "runs"
        if not runs_dir.exists():
            continue
        for run_dir in sorted(runs_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            data = _load_run_data(run_dir)
            if not data:
                continue
            metrics = data["metrics"]
            r_configured = metrics.get("known_noise_r", 0.0)
            manifold = metrics.get("metadata", {}).get("manifold", "unknown")
            seed = metrics.get("metadata", {}).get("seed", 0)
            n_samples, ambient_dim = data["x_clean"].shape

            r_knn_simple = float(
                estimate_noise_amplitude_simple(data["x_noisy"], k=2)
            )
            r_knn_twoscale = float(
                estimate_noise_amplitude_twoscale(data["x_noisy"], k1=1, k2=4)
            )
            rows.append({
                "manifold": manifold,
                "seed": seed,
                "n": n_samples,
                "D": ambient_dim,
                "r_configured": r_configured,
                "r_empirical": data["r_empirical"],
                "r_empirical_rms": data["r_empirical_rms"],
                "r_knn_simple": r_knn_simple,
                "r_knn_twoscale": r_knn_twoscale,
            })
    return rows


def print_comparison_table(rows: list[dict], title: str) -> None:
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}")
    header = (
        f"  {'Manifold':>12} {'Seed':>5} {'r_config':>10} {'r_empir':>10} "
        f"{'r_knn_s':>10} {'r_knn_t':>10} "
        f"{'emp/cfg':>8} {'knn/emp':>8}"
    )
    print(header)
    print(f"  {'-' * 75}")

    for row in rows:
        ratio_emp_cfg = row["r_empirical"] / row["r_configured"] if row["r_configured"] > 0 else 0
        ratio_knn_emp = (
            row["r_knn_simple"] / row["r_empirical"]
            if row["r_empirical"] > 0
            else 0
        )
        print(
            f"  {row['manifold']:>12} {row['seed']:>5} "
            f"{row['r_configured']:>10.4f} {row['r_empirical']:>10.4f} "
            f"{row['r_knn_simple']:>10.4f} {row['r_knn_twoscale']:>10.4f} "
            f"{ratio_emp_cfg:>8.3f} {ratio_knn_emp:>8.3f}"
        )


def print_aggregated_table(rows: list[dict]) -> None:
    """Aggregate by manifold × r_configured, showing mean ± std across seeds."""
    print(f"\n{'=' * 90}")
    print("  Aggregated: mean across seeds (± std where applicable)")
    print(f"{'=' * 90}")
    header = (
        f"  {'Manifold':>12} {'D':>3} {'r_config':>10} {'r_empir':>12} "
        f"{'r_knn_s':>12} {'r_knn_t':>12} "
        f"{'knn_s / emp':>12}"
    )
    print(header)
    print(f"  {'-' * 85}")

    manifold_order = ["circle", "sphere", "swiss_roll", "torus"]
    manifolds_present = sorted(
        set(r["manifold"] for r in rows),
        key=lambda m: manifold_order.index(m) if m in manifold_order else 99,
    )

    agg_rows = []
    for manifold in manifolds_present:
        mf_rows = [r for r in rows if r["manifold"] == manifold]
        r_levels = sorted(set(r["r_configured"] for r in mf_rows))
        for r_cfg in r_levels:
            subset = [r for r in mf_rows if abs(r["r_configured"] - r_cfg) < 1e-6]
            emp = np.array([r["r_empirical"] for r in subset])
            knn_s = np.array([r["r_knn_simple"] for r in subset])
            knn_t = np.array([r["r_knn_twoscale"] for r in subset])
            ratio = knn_s / emp if np.all(emp > 0) else np.zeros_like(knn_s)

            amb_dim = subset[0]["D"]
            emp_str = f"{np.mean(emp):.5f}±{np.std(emp):.5f}"
            knn_s_str = f"{np.mean(knn_s):.5f}±{np.std(knn_s):.5f}"
            knn_t_str = f"{np.mean(knn_t):.5f}±{np.std(knn_t):.5f}"
            ratio_str = f"{np.mean(ratio):.3f}×"
            print(
                f"  {manifold:>12} {amb_dim:>3} {r_cfg:>10.4f} {emp_str:>12} "
                f"{knn_s_str:>12} {knn_t_str:>12} {ratio_str:>12}"
            )
            agg_rows.append({
                "manifold": manifold,
                "D": amb_dim,
                "r_configured": r_cfg,
                "r_empirical_mean": float(np.mean(emp)),
                "r_knn_simple_mean": float(np.mean(knn_s)),
                "r_knn_twoscale_mean": float(np.mean(knn_t)),
                "ratio_knn_simple_to_empirical": float(np.mean(ratio)),
            })

    return agg_rows


def plot_r_comparison(rows: list[dict], out_path: Path) -> None:
    """Scatter plot: r_empirical and r_knn vs r_configured, with identity line."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    manifold_markers = {
        "circle": "o", "sphere": "s", "swiss_roll": "^", "torus": "D",
    }
    manifold_colors = {
        "circle": "#2196F3", "sphere": "#FF5722",
        "swiss_roll": "#4CAF50", "torus": "#9C27B0",
    }
    manifold_names = {
        "circle": "Circle", "sphere": "Sphere",
        "swiss_roll": "Swiss Roll", "torus": "Torus",
    }

    # Left panel: r_empirical vs r_configured (should be very tight)
    ax = axes[0]
    for manifold in ["circle", "sphere", "swiss_roll", "torus"]:
        mf_rows = [r for r in rows if r["manifold"] == manifold]
        if not mf_rows:
            continue
        x = [r["r_configured"] for r in mf_rows]
        y = [r["r_empirical"] for r in mf_rows]
        ax.scatter(
            x, y, label=manifold_names[manifold],
            marker=manifold_markers[manifold],
            color=manifold_colors[manifold],
            s=40, alpha=0.7, zorder=3,
        )

    all_r = [r["r_configured"] for r in rows] + [r["r_empirical"] for r in rows]
    lo, hi = 0, max(all_r) * 1.15
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.4, linewidth=1, label="Identity")
    ax.set_xlabel(r"$r_{\mathrm{configured}}$ (noise_std parameter)", fontsize=10)
    ax.set_ylabel(
        r"$r_{\mathrm{empirical}}$ (from data: std of $X_{noisy} - X_{clean}$)",
        fontsize=10,
    )
    ax.set_title("Data-derived r vs configured r", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_aspect("equal")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.grid(alpha=0.25, linestyle=":")

    # Right panel: r_knn vs r_empirical (shows estimation bias)
    ax = axes[1]
    for manifold in ["circle", "sphere", "swiss_roll", "torus"]:
        mf_rows = [r for r in rows if r["manifold"] == manifold]
        if not mf_rows:
            continue
        x = [r["r_empirical"] for r in mf_rows]
        y = [r["r_knn_simple"] for r in mf_rows]
        ax.scatter(
            x, y, label=f"{manifold_names[manifold]} (simple)",
            marker=manifold_markers[manifold],
            color=manifold_colors[manifold],
            s=40, alpha=0.7, zorder=3,
        )
        y2 = [r["r_knn_twoscale"] for r in mf_rows]
        ax.scatter(
            x, y2,
            marker=manifold_markers[manifold],
            facecolors="none", edgecolors=manifold_colors[manifold],
            s=40, alpha=0.7, zorder=3, linewidths=1.2,
        )

    all_vals = (
        [r["r_empirical"] for r in rows]
        + [r["r_knn_simple"] for r in rows]
    )
    lo, hi = 0, max(all_vals) * 1.15
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.4, linewidth=1, label="Identity")
    ax.set_xlabel(r"$r_{\mathrm{empirical}}$ (ground truth from data)", fontsize=10)
    ax.set_ylabel(r"$\hat{r}_{\mathrm{kNN}}$ (estimated from distances)", fontsize=10)
    ax.set_title("kNN estimate vs empirical r", fontsize=11, fontweight="bold")
    ax.legend(fontsize=7, ncol=1)
    ax.grid(alpha=0.25, linestyle=":")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_ratio_by_noise(rows: list[dict], out_path: Path) -> None:
    """Plot estimation ratio (r_knn / r_empirical) as a function of noise level."""
    fig, ax = plt.subplots(figsize=(8, 5))

    manifold_styles = {
        "circle": ("o", "#2196F3", "Circle"),
        "sphere": ("s", "#FF5722", "Sphere"),
        "swiss_roll": ("^", "#4CAF50", "Swiss Roll"),
        "torus": ("D", "#9C27B0", "Torus"),
    }

    manifold_order = ["circle", "sphere", "swiss_roll", "torus"]
    manifolds_present = sorted(
        set(r["manifold"] for r in rows),
        key=lambda m: manifold_order.index(m) if m in manifold_order else 99,
    )

    for manifold in manifolds_present:
        marker, color, name = manifold_styles.get(manifold, ("x", "gray", manifold))
        mf_rows = [r for r in rows if r["manifold"] == manifold]
        r_levels = sorted(set(r["r_configured"] for r in mf_rows))

        noise_vals = []
        ratio_means = []
        ratio_stds = []
        for r_cfg in r_levels:
            subset = [r for r in mf_rows if abs(r["r_configured"] - r_cfg) < 1e-6]
            ratios = [
                r["r_knn_simple"] / r["r_empirical"]
                for r in subset
                if r["r_empirical"] > 0
            ]
            noise_vals.append(r_cfg)
            ratio_means.append(np.mean(ratios))
            ratio_stds.append(np.std(ratios))

        ax.errorbar(
            noise_vals, ratio_means, yerr=ratio_stds,
            marker=marker, color=color, label=name,
            capsize=3, linewidth=1.5, markersize=7,
        )

    ax.axhline(y=1.0, color="black", linestyle="--", alpha=0.4, label="Perfect (ratio = 1)")
    ax.set_xlabel("Configured noise level r", fontsize=10)
    ax.set_ylabel(r"$\hat{r}_{\mathrm{kNN}} \;/\; r_{\mathrm{empirical}}$", fontsize=10)
    ax.set_title("kNN estimation bias by noise level", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25, linestyle=":")
    ax.set_ylim(0, None)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Empirical Noise Amplitude Analysis ===")
    print("Computing r_empirical = std(X_noisy - X_clean) for all experiment runs\n")

    r_est_rows = analyze_r_estimation_runs()
    sweep_rows = also_analyze_noise_sweep_runs()
    all_rows = r_est_rows + sweep_rows

    if not all_rows:
        print("No data found.")
        return

    print_comparison_table(r_est_rows, "r-estimation experiments (detailed)")
    agg = print_aggregated_table(all_rows)

    plot_r_comparison(all_rows, OUTPUT_DIR / "r_empirical_vs_configured_and_knn.png")
    plot_ratio_by_noise(all_rows, OUTPUT_DIR / "knn_estimation_bias_by_noise.png")

    # Save aggregated results to JSON
    json_out = OUTPUT_DIR / "empirical_r_comparison.json"
    json_out.write_text(json.dumps(agg, indent=2))

    print(f"\nPlots and data saved to {OUTPUT_DIR}/")
    print("  r_empirical_vs_configured_and_knn.png — scatter comparison")
    print("  knn_estimation_bias_by_noise.png — ratio plot")
    print("  empirical_r_comparison.json — aggregated table")


if __name__ == "__main__":
    main()
