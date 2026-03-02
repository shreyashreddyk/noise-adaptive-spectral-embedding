# Repo Audit — NASE

Quick-reference audit of repository structure, tooling, and how to reproduce experiments.

---

## Repository Layout

```
noise-adaptive-spectral-embedding/
├── src/nase/                 # Main Python package
│   ├── cutoffs/              # Bandwidth-stability and eigengap cutoff selection
│   ├── data/                 # Synthetic manifold generators + noise
│   ├── estimators/           # Intrinsic-dimension estimation (Levina-Bickel)
│   ├── experiments/          # Runner, sweeps, config parsing, I/O
│   ├── graphs/               # Distance computation, Gaussian kernel, normalisations
│   ├── metrics/              # Embedding quality, geodesic consistency, subspace distance
│   ├── plots/                # Spectrum, stability, embedding, ablation visualisations
│   ├── robust/               # Doubly-stochastic scaling (optional)
│   ├── spectral/             # Diffusion operator, eigensolvers, embedding
│   ├── cli.py                # Typer CLI entry point
│   ├── config.py             # Dataclass-based experiment config
│   └── utils.py              # Shared utilities
├── configs/                  # YAML experiment configs (single runs + sweeps)
├── results/                  # Timestamped experiment output directories
├── tests/                    # Pytest test suite
├── docs/                     # Methodology notes, experiment docs, citations
├── references/               # Source PDFs (papers, proposal)
├── reports/                  # This directory — analysis and final report
├── pyproject.toml            # Package metadata, dependencies, tool config
├── Makefile                  # Common dev commands
├── README.md                 # Project overview and usage
└── .github/workflows/ci.yml  # CI pipeline
```

## Key Tooling

| Tool | Purpose | Config location |
|------|---------|----------------|
| Python ≥ 3.10 | Runtime | `pyproject.toml` requires-python |
| setuptools | Build | `pyproject.toml` build-system |
| ruff | Linting + formatting | `pyproject.toml` [tool.ruff] |
| pytest | Testing | `pyproject.toml` [tool.pytest] |
| pre-commit | Git hooks | `.pre-commit-config.yaml` |
| make | Task runner | `Makefile` |

## Dependencies

Core: numpy, scipy, scikit-learn, matplotlib, pandas, pyyaml, typer, click<8.2

Dev: pytest, ruff, pre-commit

## How to Run

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -e .[dev]

# Quick checks
make lint     # ruff check + format check
make test     # pytest -q
make run-small  # smoke test (circle, n=120)

# Single experiment
python -m nase run --config configs/swiss_roll_stability.yaml

# Sweep experiment
python -m nase sweep --config configs/synthetic_noise_sweep.yaml

# Regenerate plots from existing run
python -m nase plot --run-dir results/<timestamp>_<name>
```

## Output Artifact Structure

Each single run produces:

```
results/<timestamp>_<name>/
├── config.yaml      # Frozen config used for this run
├── metrics.json     # All computed metrics
├── cutoffs.json     # Cutoff selection details (smoke_small only)
├── diagnostics.json # Stability diagnostics (later runs only)
├── arrays.npz       # Raw arrays: embeddings, eigenvalues, etc. (smoke_small only)
└── figures/ or plots/  # PNG and/or SVG visualisations
```

Each sweep produces:

```
results/<timestamp>_<name>/
├── manifest.json          # Sweep metadata
├── records.json           # Per-run records
├── records.csv            # Same as above, CSV format
├── aggregate.json         # Case-level summary statistics
├── selected_k_comparison.png  # Quick bar chart
└── runs/                  # Individual sub-run directories (same structure as single)
```

## Configs Available

| Config | Type | Status |
|--------|------|--------|
| `smoke_small.yaml` | single | Run ✓ |
| `swiss_roll_stability.yaml` | single | Run ✓ (multiple iterations) |
| `synthetic_noise_sweep.yaml` | sweep | Run ✓ |
| `ambiguous_gap_baseline.yaml` | single | Not run |
| `ambiguous_gap_suite.yaml` | sweep | Not run |
| `eigengap_ambiguous_suite.yaml` | sweep | Not run |
| `noise_sweep_circle_sphere.yaml` | sweep | Not run |
| `synthetic_bandwidth_sweep.yaml` | sweep | Not run |
| `synthetic_plan_base.yaml` | base (used by sweeps) | N/A |

## Reference PDFs

| File | Content |
|------|---------|
| `references/15271_The_Noisy_Laplacian_a_Th.pdf` | Noisy Laplacian threshold paper |
| `references/Connection graph Laplacian methods...pdf` | El Karoui & Wu — robustness to noise |
| `references/DSC205_ProjectProposal.pdf` | Our course project proposal |
| `references/Project Proposal 1_...pdf` | Alternative proposal on doubly stochastic scaling |
