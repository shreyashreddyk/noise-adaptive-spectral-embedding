# NASE: Noise-Adaptive Spectral Embedding

**Practical truncation rules for diffusion maps under noise**

A from-scratch Python toolkit that tackles the spectral truncation problem in manifold learning: *how many eigenvectors should you keep when your data is noisy?* Built for DSC 205 (Winter 2026, UC San Diego).

---

## The Problem

Diffusion maps embed high-dimensional point clouds into low-dimensional representations by computing eigenvectors of a graph operator. The critical decision is **how many eigenvectors to retain**. Keep too few and you lose structure; keep too many and you embed noise.

The standard approach — the **eigengap heuristic** — picks the dimension where consecutive eigenvalues show the largest drop. This works when the spectrum has a sharp cliff, but on many real manifolds the decay is gradual and the "largest gap" is unstable across random seeds.

The **noisy-Laplacian theory** predicts that eigenvectors split into signal and noise regimes at an index proportional to `C/r²`, where `r` is the noise standard deviation and `C` is a geometry-dependent constant. But neither `C` nor `r` is known in practice.

## Our Approach

We implemented and compared three strategies for data-driven spectral truncation:

| Method | Idea | Requires |
|--------|------|----------|
| **Bandwidth stability** | Compute eigenvectors at multiple kernel bandwidths; signal modes stay consistent, noise modes don't | An epsilon grid |
| **r-based cutoff** (k* = C/r²) | Estimate noise amplitude from kNN distances, plug into the theoretical formula | Calibrated C |
| **Eigengap baseline** | Largest drop in consecutive eigenvalues | Nothing (standard) |

All three are compared against an **oracle** that uses clean manifold data to find the subspace-distance-minimizing cutoff.

---

## Key Results

### Bandwidth stability adapts to noise

On manifolds with uniform curvature, the bandwidth-stability cutoff monotonically decreases as noise increases — exactly what the theory predicts.

**Circle** (n=450, D=3):

| Noise r | k* (stability) | k* (eigengap) | Trustworthiness |
|---------|----------------|---------------|-----------------|
| 0.02 | 8 ± 0.0 | 2 | 0.9997 |
| 0.08 | 6 ± 0.0 | 2 | 0.993 |
| 0.16 | 4 ± 0.0 | 2 | 0.981 |

Zero seed variance. The eigengap is blind to noise level — it always picks k=2.

**S-curve** (ambiguous eigengap, r=0.12):

| Method | k* | Trustworthiness | Continuity |
|--------|-----|-----------------|------------|
| Eigengap | 1 | 0.879 | 0.179 |
| Stability | 13 | 0.914 | 0.258 |

When the spectrum has no sharp gap, the eigengap gives a trivially low cutoff. Stability retains meaningful modes and outperforms by +3.5 pp trustworthiness.

### Where it fails: the swiss roll

Per-vector alignment breaks on manifolds with non-uniform curvature. On the swiss roll, even small bandwidth changes cause eigenvectors to rotate within their eigenspace, collapsing the stability cutoff to k=1. A subspace-based comparison (principal angles) would fix this — the infrastructure exists but integration is future work.

### Noise estimation is hard (as predicted)

The kNN noise amplitude estimator conflates manifold geometry with noise:

| Manifold | r=0.02 | r=0.08 | r=0.16 |
|----------|--------|--------|--------|
| Circle (D=3) | 0.69× true | 0.41× true | 0.32× true |
| Sphere (D=5) | 2.08× true | 0.82× true | 0.59× true |
| Swiss roll | 6.88× true | 2.63× true | 1.45× true |

On the circle, nearest-neighbor distances are dominated by arc length, underestimating noise. On the swiss roll, manifold distances completely dominate — the estimator returns ~0.21 regardless of true r. This validates the course feedback that "nearest neighbor distances are heavily biased by the curse of dimensionality and curvature, not just noise."

Despite these biases, the r-based cutoff with estimated r shows qualitatively correct behavior on the circle (k decreases from 12 to 5 as noise increases 0.02 to 0.16).

### Selected figures

**Cutoff comparison across noise levels (circle)**

![Method comparison — circle](results/analysis/method_comparison_circle.png)

**r-estimation accuracy**

![Noise estimation](results/analysis/r_estimation_simple.png)

**Oracle subspace distance profile**

![Subspace profile](results/analysis/oracle_subspace_profile_circle.png)

**Stability scores showing signal-noise boundary (circle, r=0.02)**

![Stability curve](results/20260302_174322_noise_sweep_circle_sphere/runs/20260302_174322_noise_sweep_circle_sphere_circle_r_0p02_seed11/figures/stability.png)

---

## Architecture

```
src/nase/
├── data/                    # Synthetic manifold generators (circle, sphere,
│   ├── synthetic.py         #   swiss roll, s-curve, torus) with controlled
│   └── noise.py             #   Gaussian noise and random ambient embedding
├── graphs/
│   ├── distances.py         # Dense pairwise + sparse kNN distance backends
│   └── kernels.py           # Gaussian kernel with optional local scaling
├── spectral/
│   └── embedding.py         # Diffusion operator construction and embedding
├── cutoffs/
│   ├── eigengap.py          # Eigengap heuristic
│   ├── bandwidth_stability.py   # Bandwidth-stability cutoff (primary method)
│   └── r_based_stub.py      # k* = C/r² cutoff with noise estimation
├── estimators/
│   ├── intrinsic_dimension.py   # Levina-Bickel kNN MLE
│   └── noise_amplitude.py       # kNN noise amplitude (simple + two-scale)
├── metrics/
│   ├── embedding_quality.py # Trustworthiness, continuity
│   ├── subspace.py          # Principal angles, subspace distance, oracle cutoff
│   └── geodesic.py          # Geodesic consistency (Spearman with kNN paths)
├── plots/
│   ├── spectrum.py          # Scree plots, eigengap charts
│   ├── stability.py         # Stability score curves, heatmaps
│   ├── embeddings.py        # 2D/3D scatter visualizations
│   ├── ablations.py         # Cutoff comparison bar charts
│   └── analysis.py          # Cross-experiment analysis plots
├── experiments/
│   ├── runner.py            # Single-experiment pipeline (data → embedding → metrics → plots)
│   ├── sweeps.py            # Multi-seed sweep orchestration with aggregation
│   └── configs.py           # YAML loading, validation, deep-merge overrides
├── robust/
│   └── dss.py               # Sinkhorn-Knopp doubly stochastic scaling
├── config.py                # Typed dataclass configuration system
└── cli.py                   # CLI: nase run / nase sweep / nase plot
```

The pipeline is fully config-driven: every experiment is defined by a YAML file, deterministically seeded, and produces structured JSON metrics alongside figures. Sweeps iterate over cases and seeds, then aggregate results.

---

## Experiments

120 experiment runs across 12 sweep/run configurations, testing 5 synthetic manifolds.

| Phase | Config | What it tests | Manifolds | Runs |
|-------|--------|---------------|-----------|------|
| 1 | `noise_sweep_circle_sphere.yaml` | Noise adaptation | Circle, Sphere | 18 |
| 1 | `synthetic_noise_sweep.yaml` | Swiss roll noise sweep | Swiss roll | 9 |
| 1 | `synthetic_bandwidth_sweep.yaml` | Epsilon grid sensitivity | Swiss roll | 4 |
| 1 | `ambiguous_gap_suite.yaml` | Eigengap vs stability | S-curve | 6 |
| 1 | `eigengap_ambiguous_suite.yaml` | High-D eigengap test | Sphere (D=6) | 6 |
| 1 | `threshold_sensitivity_circle.yaml` | Stability threshold τ | Circle | 12 |
| 1 | `torus_noise_sweep.yaml` | 5th manifold generality | Torus | 9 |
| 2a | `intrinsic_dim_circle_sphere.yaml` | Levina-Bickel estimation | Circle, Sphere | 12 |
| 2a | `intrinsic_dim_swiss_roll.yaml` | Levina-Bickel estimation | Swiss roll | 4 |
| 2c | `r_estimation_circle_sphere.yaml` | Noise estimation + r-based cutoff | Circle, Sphere | 30 |
| 2c | `r_validation_known_r.yaml` | r-based cutoff (known r) | Circle, Sphere | 18 |
| 2c | `r_estimation_swiss_roll.yaml` | Noise estimation on swiss roll | Swiss roll | 6 |

All experiment outputs (metrics, figures, arrays) are committed under `results/` for full reproducibility.

---

## Getting Started

### Installation

```bash
git clone git@github.com:shreyashreddyk/noise-adaptive-spectral-embedding.git
cd noise-adaptive-spectral-embedding
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

Requires Python >= 3.10. Tested on macOS (darwin 24.6.0).

### Quick start

```bash
# Fast smoke test (~5s)
nase run --config configs/smoke_small.yaml

# Full noise sweep (~8 min)
nase sweep --config configs/noise_sweep_circle_sphere.yaml

# r-estimation experiments (~4 min)
nase sweep --config configs/r_estimation_circle_sphere.yaml

# Regenerate plots from an existing run
nase plot --run-dir results/<run_directory>
```

### Analysis scripts

```bash
# Phase 1 analysis: oracle scaling, method comparisons, quality metrics
python3 scripts/analyze_results.py

# Phase 2c analysis: r-estimation accuracy, r-based cutoff comparison
python3 scripts/analyze_r_estimation.py
```

### Development

```bash
make lint       # ruff check + format check
make test       # pytest (49 tests across 13 files)
make format     # auto-fix lint + formatting
make verify     # lint + test together
```

Optional pre-commit hooks:

```bash
pre-commit install   # runs ruff lint + format checks before each commit
```

---

## Output Structure

Each single run produces:

```
results/<timestamp>_<name>/
├── config.yaml              # exact config snapshot
├── metrics.json             # cutoff values, quality scores, stability scores,
│                            #   noise estimation results, metrics_by_cutoff
├── cutoffs.json             # cutoff comparison across methods
├── arrays.npz               # embeddings, eigenvalues, eigenvectors
└── figures/
    ├── spectrum.{png,svg}
    ├── eigengap.{png,svg}
    ├── stability.{png,svg}
    ├── stability_heatmap.{png,svg}
    ├── embedding.{png,svg}
    ├── embedding_3d.{png,svg}
    └── ablation_cutoff.{png,svg}
```

Each sweep adds:

```
results/<timestamp>_<name>/
├── aggregate.json           # per-case means/std of k* and all quality metrics
├── records.csv / .json      # flat table of all runs
├── manifest.json            # sweep metadata
├── selected_k_comparison.png
└── runs/                    # individual run directories
```

Cross-experiment analysis plots are generated in `results/analysis/` by the scripts in `scripts/`.

---

## Evaluation Metrics

| Metric | What it measures | Range |
|--------|-----------------|-------|
| **Trustworthiness** | Whether embedding neighbors are true neighbors | [0, 1] — higher is better |
| **Continuity** | Whether original neighbors survive in the embedding | [0, 1] — higher is better |
| **Geodesic consistency** | Spearman correlation of embedding distances with kNN shortest-path distances | [-1, 1] — higher is better |
| **Subspace distance** | Chordal distance between clean and noisy eigenspaces (oracle only) | [0, 1] — lower is better |

---

## Limitations

- **Per-vector alignment** fails on manifolds with non-uniform curvature (swiss roll) where eigenvectors reorder across bandwidth changes. A subspace-based stability comparison (principal angles) would fix this.
- **Stability threshold** (τ = 0.9) is a fixed hyperparameter. Sensitivity analysis shows minimal impact on the circle, but it is decisive on the swiss roll.
- **kNN noise estimator** conflates noise with manifold geometry, especially in higher ambient dimensions and on manifolds with large local spread.
- **Calibrating C** requires oracle data (clean manifold access), limiting the r-based cutoff to settings where C is pre-computed or transferred from similar manifolds.
- **Sample sizes** are moderate (n ≤ 500). Larger samples would sharpen eigenvalue separation.
- **Synthetic data only** — real-world noise is structured and heteroscedastic, not isotropic Gaussian.

## Future Work

- **Subspace-based stability**: replace per-vector alignment with principal-angle comparison of eigenspaces. The machinery exists in `src/nase/metrics/subspace.py`.
- **Adaptive epsilon grids**: use percentiles of the pairwise distance distribution instead of fixed values.
- **Stability-gap heuristic**: detect the first significant drop in stability scores rather than using a fixed threshold.
- **Real-data experiments**: single-cell RNA-seq, image patches, or MNIST digit manifolds.
- **Noise-corrected intrinsic dimension**: use the Levina-Bickel estimate as a soft upper bound on k* with a bias correction.

---

## References

1. Coifman, R. R., & Lafon, S. (2006). *Diffusion maps*. Applied and Computational Harmonic Analysis, 21(1), 5–30.
2. von Luxburg, U. (2007). *A tutorial on spectral clustering*. Statistics and Computing, 17, 395–416.
3. Zelnik-Manor, L., & Perona, P. (2004). *Self-tuning spectral clustering*. NeurIPS 17.
4. Levina, E., & Bickel, P. J. (2004). *Maximum likelihood estimation of intrinsic dimension*. NeurIPS 17.
5. El Karoui, N. & Wu, H.-T. *Connection graph Laplacian methods can be made robust to noise*.
6. DSC 205 course materials and project proposal (`references/DSC205_ProjectProposal.pdf`).

Full report with all tables, figures, and analysis: [`reports/final_report.md`](reports/final_report.md)

---

## Skills and Highlights

<table>
<tr>
<td width="50%">

**Spectral Methods & Manifold Learning**
- Diffusion maps pipeline from scratch (kernel construction, alpha-normalization, spectral decomposition)
- Bandwidth-stability truncation criterion — a novel data-driven alternative to the eigengap heuristic
- Subspace distance via principal angles for oracle evaluation
- Levina-Bickel intrinsic dimension estimation

</td>
<td width="50%">

**Statistical Estimation & Analysis**
- kNN-based noise amplitude estimation with bias analysis across manifold geometries
- Empirical validation of theoretical scaling law (k* ~ C/r²) with calibrated constants
- Multi-metric evaluation: trustworthiness, continuity, geodesic consistency
- Honest quantification of when methods work and when they fail

</td>
</tr>
<tr>
<td>

**Software Engineering**
- Modular Python package with typed dataclass configs, YAML-driven experiments, and deep-merge override system
- CLI with three commands (`run`, `sweep`, `plot`) backed by Typer
- 49-test pytest suite covering data generation through end-to-end runner smoke tests
- Pre-commit hooks (ruff lint + format), Makefile automation, deterministic seeding

</td>
<td>

**Reproducibility & Scientific Computing**
- 120 seeded experiment runs across 12 configurations, all outputs version-controlled
- Automated sweep orchestration with multi-seed aggregation (mean/std)
- Every quantitative claim in the report cites its source JSON file
- Config-driven pipeline: change a YAML file, get a new experiment with full artifact trail

</td>
</tr>
</table>

**Tech stack**: Python, NumPy, SciPy, scikit-learn, Matplotlib, pandas, PyYAML, Typer, pytest, ruff

**Concepts demonstrated**: spectral graph theory, manifold learning, noise robustness, statistical estimation, eigenvalue problems, kNN methods, subspace geometry, experiment design, scientific reproducibility
