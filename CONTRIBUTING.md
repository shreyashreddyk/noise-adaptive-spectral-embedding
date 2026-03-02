# Contributing to NASE

## Dev Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .[dev]
pre-commit install
```

Requires Python >= 3.10.

The `pre-commit install` step registers git hooks defined in `.pre-commit-config.yaml`. These currently run two hooks from `ruff-pre-commit` (v0.9.9):

- **`ruff-check`** — runs the same lint rules configured in `pyproject.toml` (E, F, I, B, UP, N, W with E203 ignored, line length 100).
- **`ruff-format`** — enforces the same formatting rules (double quotes, space indent).

If either hook fails, the commit is rejected. You can bypass hooks temporarily with `git commit --no-verify`, but this is discouraged — the CI-equivalent checks are `make lint`.

## Running Checks

```bash
make verify     # runs both lint and tests
```

Or separately:

```bash
ruff check .             # lint
ruff format --check .    # format check
pytest -q                # tests
```

Auto-fix formatting:

```bash
make format     # ruff check --fix + ruff format
```

## Adding a New Experiment Config

1. Create a YAML file in `configs/`. Use `configs/smoke_small.yaml` as a minimal template or `configs/swiss_roll_stability.yaml` for a full single-run config.

2. For a sweep config, define `base_config`, `seeds`, and `cases` with `overrides`:

```yaml
name: my_sweep
output_root: results
base_config: swiss_roll_stability.yaml
seeds: [42, 43]
cases:
  - name: low_noise
    overrides:
      data:
        noise_std: 0.03
  - name: high_noise
    overrides:
      data:
        noise_std: 0.20
```

3. Run it:

```bash
nase run --config configs/my_config.yaml       # single run
nase sweep --config configs/my_sweep.yaml      # sweep
```

4. Check output in `results/<timestamp>_<name>/`.

## Interpreting Run Outputs

Each run directory contains:

- **`config.yaml`** — the exact config that was used (with all defaults filled in).
- **`metrics.json`** — key results: `selected_k`, `k_eigengap`, `k_bandwidth_stability`, `k_oracle`, quality metrics (`trustworthiness`, `continuity`, `geodesic_consistency`), and per-k stability scores.
- **`cutoffs.json`** — cutoff comparison summary.
- **`arrays.npz`** — numpy arrays for downstream analysis (clean/noisy points, embeddings, eigenvalues, eigenvectors, stability scores).
- **`figures/`** — spectrum, eigengap, stability curve, heatmap, embedding scatter, and ablation comparison plots.

For sweeps, also check `aggregate.json` (case-level means/std) and `records.csv` (flat table of all runs).

## What Git Tracks (and Doesn't)

The `.gitignore` excludes:

- `.venv/` — local virtual environment
- `__pycache__/`, `*.pyc`, `*.pyo`, `*.pyd` — Python bytecode
- `.pytest_cache/`, `.ruff_cache/`, `.mypy_cache/` — tool caches
- `build/`, `dist/`, `*.egg-info/` — build artifacts

Everything else is tracked, including `results/`. We commit experiment outputs (metrics, figures, arrays) so the repo is a self-contained record. If a sweep produces very large `arrays.npz` files, consider adding specific run directories to `.gitignore` and noting their absence in the README.

## Style Guide

- Ruff config in `pyproject.toml`: line length 100, rules E/F/I/B/UP/N/W.
- Pre-commit hooks enforce lint and format checks automatically before each commit (see Dev Setup above).
- Use type annotations and dataclasses for structured data.
- Thread `seed` values through all random operations for determinism.
- Keep functions focused and unit-testable.
- Write docs in original language — do not copy from papers or external sources.
- Cite referenced ideas in `docs/references/CITATIONS.md`.

## Adding a New Manifold Generator

1. Add a `_sample_<name>` function in `src/nase/data/synthetic.py` following the existing pattern (returns `(x, latent_params, native_dim)`).
2. Register it in the `generate_synthetic` dispatch block.
3. Add a test in `tests/test_synthetic.py`.

## Adding a New Cutoff Method

1. Create a module in `src/nase/cutoffs/` with a `select_k_<method>` function.
2. Wire it into `_compute_bundle` in `src/nase/experiments/runner.py` alongside the existing eigengap and bandwidth-stability paths.
3. Add the method name to the `config.cutoff.method` dispatch logic.
4. Add tests in `tests/test_cutoffs.py`.
