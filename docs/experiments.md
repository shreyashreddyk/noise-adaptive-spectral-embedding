# Experiment Suites

This document describes the synthetic experiment suites bundled with NASE.

## Design Principles

- Start with synthetic manifolds where additive noise amplitude `r` is known by construction.
- Do not rely on nearest-neighbour heuristics to estimate `r` in the core pipeline.
- Treat geometry-dependent constants as non-identifiable in practice; use empirical diagnostics instead.

## Single-Run Configs

- `configs/swiss_roll_stability.yaml`: baseline bandwidth-stability run.
- `configs/ambiguous_gap_baseline.yaml`: baseline eigengap run in a noisier setting.

## Sweep Configs

- `configs/synthetic_noise_sweep.yaml`: compares low/medium/high additive noise across fixed seeds.
- `configs/synthetic_bandwidth_sweep.yaml`: compares narrow and wide epsilon grids.
- `configs/ambiguous_gap_suite.yaml`: compares eigengap and bandwidth-stability cutoffs in an ambiguous-gap scenario.

## Running Suites

```bash
python -m nase.cli sweep --config configs/synthetic_noise_sweep.yaml
python -m nase.cli sweep --config configs/synthetic_bandwidth_sweep.yaml
python -m nase.cli sweep --config configs/ambiguous_gap_suite.yaml
```

## Interpreting Outputs

Each sweep writes:

- `records.csv` and `records.json`: per-case/per-seed records,
- `aggregate.json`: case-level means and deviations,
- `selected_k_comparison.png`: quick visual comparison across cases.

For method comparisons, inspect both `k_eigengap` and `k_bandwidth_stability` alongside embedding-quality metrics.
