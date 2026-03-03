"""Analyze r-estimation experiment results and generate plots."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from nase.plots.analysis import plot_method_comparison_across_noise, plot_r_estimation_accuracy

RESULTS_ROOT = Path("results")
ANALYSIS_DIR = Path("results/analysis")


def _load_records(sweep_dir: Path) -> list[dict]:
    return json.loads((sweep_dir / "records.json").read_text())["records"]


def _find_sweep(pattern: str) -> Path | None:
    for d in sorted(RESULTS_ROOT.iterdir()):
        if d.is_dir() and pattern in d.name:
            return d
    return None


def analyze_r_accuracy() -> None:
    """Compare r_estimated vs r_true."""
    print("\n=== Noise Amplitude Estimation Accuracy ===")

    sweep_dir = _find_sweep("r_estimation_circle_sphere")
    if not sweep_dir:
        print("  Sweep not found.")
        return

    records = _load_records(sweep_dir)

    r_true_all, r_est_simple_all, r_est_twoscale_all, manifolds_all = [], [], [], []

    for r in records:
        noise_r = r["known_noise_r"]
        r_est_s = r["r_estimated_simple"]
        r_est_t = r["r_estimated_twoscale"]
        case = r["case"]
        manifold = "circle" if "circle" in case else "sphere"

        r_true_all.append(noise_r)
        r_est_simple_all.append(r_est_s)
        r_est_twoscale_all.append(r_est_t)
        manifolds_all.append(manifold)

    plot_r_estimation_accuracy(
        r_true_all, r_est_simple_all, manifolds_all,
        ANALYSIS_DIR / "r_estimation_simple.png", dpi=160,
    )
    plot_r_estimation_accuracy(
        r_true_all, r_est_twoscale_all, manifolds_all,
        ANALYSIS_DIR / "r_estimation_twoscale.png", dpi=160,
    )

    print(
        f"\n  {'Manifold':>10} {'r_true':>8} {'r_simple':>10} "
        f"{'r_twoscale':>12} {'ratio_s':>9} {'ratio_t':>9}"
    )
    for manifold in ["circle", "sphere"]:
        for r_true in sorted(set(r_true_all)):
            mask = [m == manifold and abs(rt - r_true) < 1e-6
                    for m, rt in zip(manifolds_all, r_true_all, strict=False)]
            if not any(mask):
                continue
            rs = [r_est_simple_all[i] for i in range(len(mask)) if mask[i]]
            rt = [r_est_twoscale_all[i] for i in range(len(mask)) if mask[i]]
            ratio_s = np.mean(rs) / r_true if r_true > 0 else float('inf')
            ratio_t = np.mean(rt) / r_true if r_true > 0 else float('inf')
            mean_rs, mean_rt = np.mean(rs), np.mean(rt)
            print(
                f"  {manifold:>10} {r_true:>8.3f} {mean_rs:>10.4f} "
                f"{mean_rt:>12.4f} {ratio_s:>9.2f}x {ratio_t:>9.2f}x"
            )

    # Swiss roll
    sr_dir = _find_sweep("r_estimation_swiss_roll")
    if sr_dir:
        sr_records = _load_records(sr_dir)
        print("\n  Swiss roll r-estimation:")
        print(f"  {'r_true':>8} {'r_simple':>10} {'r_twoscale':>12} {'ratio_s':>9}")
        for r_true in sorted(set(r["known_noise_r"] for r in sr_records)):
            recs = [r for r in sr_records if abs(r["known_noise_r"] - r_true) < 1e-6]
            rs = np.mean([r["r_estimated_simple"] for r in recs])
            rt = np.mean([r["r_estimated_twoscale"] for r in recs])
            print(f"  {r_true:>8.3f} {rs:>10.4f} {rt:>12.4f} {rs/r_true:>9.2f}x")


def analyze_r_based_cutoff() -> None:
    """Compare k* across methods including r-based."""
    print("\n=== R-Based Cutoff Comparison ===")

    est_dir = _find_sweep("r_estimation_circle_sphere")
    known_dir = _find_sweep("r_validation_known_r")

    if not est_dir:
        return

    est_records = _load_records(est_dir)
    known_records = _load_records(known_dir) if known_dir else []

    for manifold in ["circle", "sphere"]:
        print(f"\n  {manifold.upper()} — Cutoff comparison (estimated r):")
        print(f"  {'r_true':>8} {'k_r_est':>8} {'k_stab':>7} {'k_egap':>7} {'trust':>8}")
        mf_recs = [r for r in est_records if manifold in r["case"]]
        for r_true in sorted(set(r["known_noise_r"] for r in mf_recs)):
            recs = [r for r in mf_recs if abs(r["known_noise_r"] - r_true) < 1e-6]
            kr = np.mean([r["k_r_based"] for r in recs])
            ks = np.mean([r["k_bandwidth_stability"] for r in recs])
            ke = np.mean([r["k_eigengap"] for r in recs])
            t = np.mean([r["trustworthiness"] for r in recs])
            print(f"  {r_true:>8.3f} {kr:>8.1f} {ks:>7.1f} {ke:>7.1f} {t:>8.4f}")

        if known_records:
            print(f"\n  {manifold.upper()} — Cutoff comparison (known r):")
            print(f"  {'r_true':>8} {'k_r_known':>10} {'k_stab':>7} {'k_egap':>7} {'trust':>8}")
            kf_recs = [r for r in known_records if manifold in r["case"]]
            for r_true in sorted(set(r["known_noise_r"] for r in kf_recs)):
                recs = [r for r in kf_recs if abs(r["known_noise_r"] - r_true) < 1e-6]
                kr = np.mean([r["k_r_based"] for r in recs])
                ks = np.mean([r["k_bandwidth_stability"] for r in recs])
                ke = np.mean([r["k_eigengap"] for r in recs])
                t = np.mean([r["trustworthiness"] for r in recs])
                print(f"  {r_true:>8.3f} {kr:>10.1f} {ks:>7.1f} {ke:>7.1f} {t:>8.4f}")

    # Generate method comparison plots for circle
    circle_recs = [r for r in est_records if "circle" in r["case"]]
    noise_levels = sorted(set(r["known_noise_r"] for r in circle_recs))
    methods_data: dict[str, dict[str, list[float]]] = {}
    for method, key in [("r-based (estimated)", "k_r_based"),
                         ("Bandwidth stability", "k_bandwidth_stability"),
                         ("Eigengap", "k_eigengap")]:
        methods_data[method] = {}
        for nl in noise_levels:
            vals = [r[key] for r in circle_recs if abs(r["known_noise_r"] - nl) < 1e-6]
            methods_data[method][str(nl)] = vals

    plot_method_comparison_across_noise(
        methods_data, noise_levels,
        ANALYSIS_DIR / "r_based_comparison_circle.png",
        dpi=160, title="Cutoff comparison: circle (with r-based)",
    )

    sphere_recs = [r for r in est_records if "sphere" in r["case"]]
    noise_levels_s = sorted(set(r["known_noise_r"] for r in sphere_recs))
    methods_data_s: dict[str, dict[str, list[float]]] = {}
    for method, key in [("r-based (estimated)", "k_r_based"),
                         ("Bandwidth stability", "k_bandwidth_stability"),
                         ("Eigengap", "k_eigengap")]:
        methods_data_s[method] = {}
        for nl in noise_levels_s:
            vals = [r[key] for r in sphere_recs if abs(r["known_noise_r"] - nl) < 1e-6]
            methods_data_s[method][str(nl)] = vals

    plot_method_comparison_across_noise(
        methods_data_s, noise_levels_s,
        ANALYSIS_DIR / "r_based_comparison_sphere.png",
        dpi=160, title="Cutoff comparison: sphere (with r-based)",
    )


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    analyze_r_accuracy()
    analyze_r_based_cutoff()
    print(f"\n=== R-estimation analysis complete. Plots saved to {ANALYSIS_DIR} ===")


if __name__ == "__main__":
    main()
