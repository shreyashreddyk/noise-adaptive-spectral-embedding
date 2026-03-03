"""Comprehensive analysis of all experiment results.

Generates summary tables, oracle scaling analysis, and additional plots
for the final report. Run from the project root:

    python3 scripts/analyze_results.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from nase.plots.analysis import (
    plot_method_comparison_across_noise,
    plot_metrics_across_noise,
    plot_oracle_scaling,
    plot_oracle_subspace_profile,
    plot_threshold_sensitivity,
)

RESULTS_ROOT = Path("results")
ANALYSIS_DIR = Path("results/analysis")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_run_dirs(sweep_dir: Path) -> list[Path]:
    runs_dir = sweep_dir / "runs"
    if not runs_dir.exists():
        return []
    return sorted(d for d in runs_dir.iterdir() if d.is_dir() and (d / "metrics.json").exists())


def _extract_oracle_data(sweep_dir: Path) -> list[dict]:
    """Extract k_oracle and noise_r from each run in a sweep."""
    rows = []
    for run_dir in _find_run_dirs(sweep_dir):
        m = _load_json(run_dir / "metrics.json")
        rows.append({
            "manifold": m.get("metadata", {}).get("manifold", "unknown"),
            "noise_r": m.get("known_noise_r", 0.0),
            "seed": m.get("metadata", {}).get("seed", 0),
            "k_oracle": m.get("k_oracle", 0),
            "k_eigengap": m.get("k_eigengap", 0),
            "k_bandwidth_stability": m.get("k_bandwidth_stability", 0),
            "trustworthiness": m.get("trustworthiness", 0.0),
            "continuity": m.get("continuity", 0.0),
            "geodesic_consistency": m.get("geodesic_consistency", 0.0),
            "oracle_subspace_distance_by_k": m.get("oracle_subspace_distance_by_k", {}),
            "bandwidth_stability_scores": m.get("bandwidth_stability_scores", {}),
            "metrics_by_cutoff": m.get("metrics_by_cutoff", {}),
        })
    return rows


def analyze_oracle_scaling() -> None:
    """Step 3: Analyze k_oracle vs 1/r^2 across manifolds."""
    print("\n=== Oracle Scaling Analysis ===")

    sweep_dirs = {
        "circle_sphere": RESULTS_ROOT / "20260302_174322_noise_sweep_circle_sphere",
        "swiss_roll": RESULTS_ROOT / "20260227_183932_synthetic_noise_sweep",
        "torus": None,
    }
    for d in sorted(RESULTS_ROOT.iterdir()):
        if d.is_dir() and "torus_noise_sweep" in d.name:
            sweep_dirs["torus"] = d

    manifold_data: dict[str, list[tuple[float, float]]] = {}

    for _sweep_name, sweep_dir in sweep_dirs.items():
        if sweep_dir is None or not sweep_dir.exists():
            continue
        rows = _extract_oracle_data(sweep_dir)
        for row in rows:
            manifold = row["manifold"]
            r = row["noise_r"]
            k_oracle = row["k_oracle"]
            if r > 0:
                manifold_data.setdefault(manifold, []).append((r, k_oracle))

    if not manifold_data:
        print("  No oracle data found.")
        return

    c_values = plot_oracle_scaling(
        manifold_data,
        ANALYSIS_DIR / "oracle_scaling.png",
        dpi=160,
    )

    print("\n  Estimated C = k_oracle * r^2 per manifold:")
    for manifold, c in sorted(c_values.items()):
        pairs = manifold_data[manifold]
        per_point_c = [k * r**2 for r, k in pairs]
        print(f"    {manifold}: C = {c:.4f} (range {min(per_point_c):.4f}–{max(per_point_c):.4f})")


def analyze_method_comparison() -> None:
    """Step 4/5: Full method comparison across noise levels with all metrics."""
    print("\n=== Method Comparison (Circle + Sphere) ===")

    sweep_dir = RESULTS_ROOT / "20260302_174322_noise_sweep_circle_sphere"
    if not sweep_dir.exists():
        print("  Sweep not found.")
        return

    rows = _extract_oracle_data(sweep_dir)

    for target_manifold in ["circle", "sphere"]:
        mrows = [r for r in rows if r["manifold"] == target_manifold]
        if not mrows:
            continue

        noise_levels_set = sorted(set(r["noise_r"] for r in mrows))
        methods = ["k_eigengap", "k_bandwidth_stability", "k_oracle"]
        method_labels = ["Eigengap", "Bandwidth stability", "Oracle"]

        data: dict[str, dict[str, list[float]]] = {}
        for method, label in zip(methods, method_labels, strict=False):
            data[label] = {}
            for nl in noise_levels_set:
                vals = [r[method] for r in mrows if abs(r["noise_r"] - nl) < 1e-6]
                data[label][str(nl)] = vals

        plot_method_comparison_across_noise(
            data,
            noise_levels_set,
            ANALYSIS_DIR / f"method_comparison_{target_manifold}.png",
            dpi=160,
            title=f"Cutoff comparison: {target_manifold}",
        )

        trust_means = []
        cont_means = []
        geo_means = []
        for nl in noise_levels_set:
            nl_rows = [r for r in mrows if abs(r["noise_r"] - nl) < 1e-6]
            trust_means.append(np.mean([r["trustworthiness"] for r in nl_rows]))
            cont_means.append(np.mean([r["continuity"] for r in nl_rows]))
            geo_means.append(np.mean([r["geodesic_consistency"] for r in nl_rows]))

        plot_metrics_across_noise(
            noise_levels_set,
            {
                "Trustworthiness": trust_means,
                "Continuity": cont_means,
                "Geodesic consistency": geo_means,
            },
            ANALYSIS_DIR / f"quality_metrics_{target_manifold}.png",
            dpi=160,
            title=f"Embedding quality vs noise: {target_manifold}",
        )

        print(f"\n  {target_manifold.upper()} — Full metrics table:")
        print(
            f"  {'r':>6} {'k_stab':>7} {'k_egap':>7} {'k_orac':>7} "
            f"{'trust':>8} {'cont':>8} {'geo':>8}"
        )
        for nl in noise_levels_set:
            nl_rows = [r for r in mrows if abs(r["noise_r"] - nl) < 1e-6]
            ks = np.mean([r["k_bandwidth_stability"] for r in nl_rows])
            ke = np.mean([r["k_eigengap"] for r in nl_rows])
            ko = np.mean([r["k_oracle"] for r in nl_rows])
            t = np.mean([r["trustworthiness"] for r in nl_rows])
            c = np.mean([r["continuity"] for r in nl_rows])
            g = np.mean([r["geodesic_consistency"] for r in nl_rows])
            print(f"  {nl:>6.2f} {ks:>7.1f} {ke:>7.1f} {ko:>7.1f} {t:>8.4f} {c:>8.4f} {g:>8.4f}")


def analyze_oracle_subspace_profiles() -> None:
    """Step 4: Plot oracle subspace distance profiles for representative runs."""
    print("\n=== Oracle Subspace Distance Profiles ===")

    sweep_dir = RESULTS_ROOT / "20260302_174322_noise_sweep_circle_sphere"
    if not sweep_dir.exists():
        return

    rows = _extract_oracle_data(sweep_dir)

    for manifold in ["circle", "sphere"]:
        profiles: dict[str, dict[int, float]] = {}
        for row in rows:
            if row["manifold"] != manifold:
                continue
            if row["seed"] != 11:
                continue
            r = row["noise_r"]
            raw_profile = row["oracle_subspace_distance_by_k"]
            profiles[f"r={r}"] = {int(k): float(v) for k, v in raw_profile.items()}

        if profiles:
            plot_oracle_subspace_profile(
                profiles,
                ANALYSIS_DIR / f"oracle_subspace_profile_{manifold}.png",
                dpi=160,
                title=f"Oracle subspace distance: {manifold}",
            )
            print(f"  Generated oracle subspace profile for {manifold}")


def analyze_threshold_sensitivity() -> None:
    """Step 1 results: plot k* vs threshold tau."""
    print("\n=== Threshold Sensitivity Analysis ===")

    sweep_dir = None
    for d in sorted(RESULTS_ROOT.iterdir()):
        if d.is_dir() and "threshold_sensitivity" in d.name:
            sweep_dir = d
    if sweep_dir is None:
        print("  Threshold sensitivity sweep not found.")
        return

    agg = _load_json(sweep_dir / "aggregate.json")
    thresholds = []
    k_means = []
    k_stds = []
    for case in sorted(agg["cases"], key=lambda c: c["case"]):
        tau_str = case["case"].replace("tau_0", "0.")
        tau = float(tau_str)
        thresholds.append(tau)
        k_means.append(case["selected_k_mean"])
        k_stds.append(case["selected_k_std"])
        print(f"  tau={tau:.2f}: k*={case['selected_k_mean']:.2f} +/- {case['selected_k_std']:.2f}")

    plot_threshold_sensitivity(
        thresholds, k_means, k_stds,
        ANALYSIS_DIR / "threshold_sensitivity.png",
        dpi=160,
    )


def analyze_torus() -> None:
    """Step 2 results: analyze torus noise sweep."""
    print("\n=== Torus Noise Sweep Analysis ===")

    sweep_dir = None
    for d in sorted(RESULTS_ROOT.iterdir()):
        if d.is_dir() and "torus_noise_sweep" in d.name:
            sweep_dir = d
    if sweep_dir is None:
        print("  Torus sweep not found.")
        return

    rows = _extract_oracle_data(sweep_dir)
    noise_levels = sorted(set(r["noise_r"] for r in rows))

    print(
        f"\n  {'r':>6} {'k_stab':>7} {'k_egap':>7} {'k_orac':>7} "
        f"{'trust':>8} {'cont':>8} {'geo':>8}"
    )
    for nl in noise_levels:
        nl_rows = [r for r in rows if abs(r["noise_r"] - nl) < 1e-6]
        ks = np.mean([r["k_bandwidth_stability"] for r in nl_rows])
        ke = np.mean([r["k_eigengap"] for r in nl_rows])
        ko = np.mean([r["k_oracle"] for r in nl_rows])
        t = np.mean([r["trustworthiness"] for r in nl_rows])
        c = np.mean([r["continuity"] for r in nl_rows])
        g = np.mean([r["geodesic_consistency"] for r in nl_rows])
        print(f"  {nl:>6.2f} {ks:>7.1f} {ke:>7.1f} {ko:>7.1f} {t:>8.4f} {c:>8.4f} {g:>8.4f}")

    methods_data: dict[str, dict[str, list[float]]] = {}
    for method, label in [
        ("k_bandwidth_stability", "Bandwidth stability"),
        ("k_eigengap", "Eigengap"),
        ("k_oracle", "Oracle"),
    ]:
        methods_data[label] = {}
        for nl in noise_levels:
            vals = [r[method] for r in rows if abs(r["noise_r"] - nl) < 1e-6]
            methods_data[label][str(nl)] = vals

    plot_method_comparison_across_noise(
        methods_data,
        noise_levels,
        ANALYSIS_DIR / "method_comparison_torus.png",
        dpi=160,
        title="Cutoff comparison: torus",
    )


def analyze_s_curve() -> None:
    """Analyze ambiguous gap suite (s-curve head-to-head)."""
    print("\n=== S-Curve Analysis ===")

    sweep_dir = RESULTS_ROOT / "20260302_173959_ambiguous_gap_suite"
    if not sweep_dir.exists():
        return

    rows = _extract_oracle_data(sweep_dir)
    for row in rows:
        print(f"  seed={row['seed']}: k_stab={row['k_bandwidth_stability']}, "
              f"k_egap={row['k_eigengap']}, k_oracle={row['k_oracle']}, "
              f"trust={row['trustworthiness']:.4f}, cont={row['continuity']:.4f}")


def analyze_metrics_by_cutoff() -> None:
    """Step 4: Extract and tabulate metrics_by_cutoff from circle/sphere sweep."""
    print("\n=== Metrics by Cutoff Method (Circle + Sphere) ===")

    sweep_dir = RESULTS_ROOT / "20260302_174322_noise_sweep_circle_sphere"
    if not sweep_dir.exists():
        return

    rows = _extract_oracle_data(sweep_dir)

    for manifold in ["circle", "sphere"]:
        mrows = [r for r in rows if r["manifold"] == manifold]
        noise_levels = sorted(set(r["noise_r"] for r in mrows))

        print(f"\n  {manifold.upper()} — Quality at each method's chosen k:")
        print(f"  {'r':>6} {'method':>14} {'trust':>8} {'cont':>8} {'geo':>8}")
        for nl in noise_levels:
            nl_rows = [r for r in mrows if abs(r["noise_r"] - nl) < 1e-6]
            for method in ["eigengap", "bandwidth_stability", "oracle"]:
                trusts, conts, geos = [], [], []
                for row in nl_rows:
                    mbc = row.get("metrics_by_cutoff", {}).get(method, {})
                    if mbc:
                        trusts.append(mbc.get("trustworthiness", 0))
                        conts.append(mbc.get("continuity", 0))
                        geos.append(mbc.get("geodesic_consistency", 0))
                if trusts:
                    print(f"  {nl:>6.2f} {method:>14} {np.mean(trusts):>8.4f} "
                          f"{np.mean(conts):>8.4f} {np.mean(geos):>8.4f}")


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    analyze_oracle_scaling()
    analyze_method_comparison()
    analyze_oracle_subspace_profiles()
    analyze_threshold_sensitivity()
    analyze_torus()
    analyze_s_curve()
    analyze_metrics_by_cutoff()
    print(f"\n=== All analysis complete. Plots saved to {ANALYSIS_DIR} ===")


if __name__ == "__main__":
    main()
