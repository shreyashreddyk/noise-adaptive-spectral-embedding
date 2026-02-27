# NASE: Noise-Adaptive Spectral Embedding

NASE is a reproducible toolkit for studying spectral manifold learning under additive noise.
The main engineering goal is to select a practical truncation cutoff `k*` for diffusion-map style
embeddings in a way that is robust when high-frequency modes are unstable.

## Project Overview and Goals

- Implement a clean, testable pipeline from synthetic data to spectral embeddings and diagnostics.
- Compare two cutoff strategies: `bandwidth_stability` (primary) and `eigengap` (baseline).
- Quantify behavior across controlled noise settings where the true noise level is known.
- Produce repeatable artefacts (`metrics.json`, `cutoffs.json`, `arrays.npz`, and figures) for analysis.

## Clone and Run From Scratch

```bash
git clone <your-fork-or-repo-url>
cd noise-adaptive-spectral-embedding
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .[dev]
```

Optional tooling:

```bash
pre-commit install
```

## Common Commands

Using `Makefile`:

```bash
make lint
make test
make run-small
```

Equivalent direct commands:

```bash
ruff check .
ruff format --check .
pytest -q
python -m nase run --config configs/smoke_small.yaml
```

## Reproducing Core Figures

1) Run the core single-run experiment:

```bash
python -m nase run --config configs/swiss_roll_stability.yaml
```

2) Regenerate figures from a chosen run directory (if needed):

```bash
python -m nase plot --run-dir results/<timestamp>_swiss_roll_stability
```

3) Reproduce method-comparison sweeps:

```bash
python -m nase sweep --config configs/noise_sweep_circle_sphere.yaml
python -m nase sweep --config configs/eigengap_ambiguous_suite.yaml
```

## Bandwidth-Stability Truncation vs Eigengap

- `eigengap`: picks `k` where `lambda_k - lambda_{k+1}` is largest within configured bounds.
- `bandwidth_stability`: computes eigenvectors across an epsilon grid and scores per-mode agreement;
  modes that remain stable across bandwidth changes are treated as signal.
- Why include both: eigengap is simple and classical, but can be ambiguous in noisy settings with
  small or irregular gaps. Stability provides an orthogonal robustness signal tied to perturbation behavior.
- The chosen method is configured via `cutoff.method` in each experiment config.

## Output Artefacts

Each run directory under `results/` contains:

- `config.yaml`
- `metrics.json`
- `cutoffs.json`
- `arrays.npz`
- `figures/*.png` and/or `figures/*.svg`

Sweep runs also include aggregate records and manifests.

## Attribution and Citations

This project draws on established ideas from diffusion operators and spectral manifold learning.
Use and cite foundational work when building on these results:

- Coifman, R. R., & Lafon, S. (2006). *Diffusion maps*. Applied and Computational Harmonic Analysis.
- von Luxburg, U. (2007). *A tutorial on spectral clustering*. Statistics and Computing.
- Berry, T., et al. (noise-robust manifold learning and diffusion-operator literature).

Project-specific attribution notes are maintained in `docs/references/CITATIONS.md`.

## No Plagiarism Policy

- All code comments, docs, and reports in this repository must be written in original wording.
- Do not paste text from papers, websites, or other repositories.
- When an external idea is used, cite the source and explain the adaptation in your own words.
