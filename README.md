# NASE: Noise-Adaptive Spectral Embedding

NASE is a reproducible toolkit and experiment suite for spectral manifold learning under additive noise.  
Its central goal is to choose a practical spectral truncation cutoff `k*` for diffusion maps and graph-based operators.

The primary method is **bandwidth-stability truncation**: eigenvectors that remain stable across kernel bandwidths are treated as signal, while unstable high-frequency modes are treated as noise.

## Project Overview

- Core comparison: `bandwidth_stability` vs `eigengap` baseline.
- Initial focus: synthetic manifolds with known additive noise amplitude `r`.
- Explicitly avoids relying on fragile nearest-neighbour heuristics for `r` in the core pipeline.
- Optional later-phase hooks are included for intrinsic-dimension and DSS-inspired ideas, but are not used as default decision drivers.

## Installation

```bash
python -m pip install -e .[dev]
pre-commit install
```

## Quick Start

Run a stability experiment:

```bash
python -m nase run --config configs/swiss_roll_stability.yaml
```

Run a circle/sphere noise sweep (varying known noise level `r`):

```bash
python -m nase sweep --config configs/noise_sweep_circle_sphere.yaml
```

Run an eigengap-ambiguous suite comparing eigengap vs stability:

```bash
python -m nase sweep --config configs/eigengap_ambiguous_suite.yaml
```

Regenerate figures for an existing run:

```bash
python -m nase plot --run-dir results/<timestamp>_<run_name>
```

## Output Artefacts

Each run writes a timestamped folder under `results/` with:

- `config.yaml`
- `metrics.json`
- `cutoffs.json`
- `arrays.npz` (compressed NumPy arrays)
- `figures/*.png`
- `figures/*.svg`

Sweep runs additionally write:

- `records.csv`
- `records.json`
- `aggregate.json`
- `manifest.json`
- `selected_k_comparison.png`

## Experiments

- Synthetic manifolds currently include circle, Swiss roll, and S-curve.
- Additive Gaussian noise is injected with known standard deviation `r`.
- Bandwidth stability is computed across an epsilon grid.
- Eigengap is reported as baseline for direct comparison.
- Sweep suites are provided for noise-level, bandwidth-grid, and ambiguous-gap comparisons.

## Experiment Matrix

| Config | Purpose | Expected insight |
|---|---|---|
| `configs/swiss_roll_stability.yaml` | Single-run bandwidth stability | Baseline `k*` from stable modes |
| `configs/noise_sweep_circle_sphere.yaml` | Multi-seed noise sweep over `r` on circle/sphere | `k*` sensitivity to known additive noise |
| `configs/eigengap_ambiguous_suite.yaml` | Eigengap-ambiguous method comparison suite | Why stability helps when eigengaps are ambiguous |

## Reproducibility and Quality

- Deterministic seeding is built into the runner.
- Ruff enforces formatting and lint checks.
- Pytest includes unit tests and a smoke test for the full runner.
- GitHub Actions runs lint, type checks, and tests on push and pull requests.

## No Plagiarism / Attribution Policy

This repository must use original wording in code comments and documentation.  
Do not paste text from papers, websites, or other repositories.  
If an external idea influences implementation, cite the source and restate it in your own words.

## Citations

Please cite foundational work when using this repository:

- Coifman, R. R., & Lafon, S. (2006). Diffusion maps.
- Berry, T., et al. (noise-robust manifold learning references).
- Additional references and attribution notes are tracked in `docs/references/CITATIONS.md`.
