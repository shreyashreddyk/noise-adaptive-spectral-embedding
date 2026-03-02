# Completeness Audit

> Generated: 2026-03-02. Tracks what exists, what is missing, what we added, and why.

---

## 1. Repo Health

| Check | Status | Notes |
|-------|--------|-------|
| Tests (pytest -q) | **PASS** | 47/47 tests pass |
| Lint (ruff check) | **PASS** | All checks passed |
| Format (ruff format --check) | **PASS** | 55 files already formatted |
| Package install | **OK** | `import nase` works from system python3 |
| CLI entrypoint | **OK** | `python3 -m nase run`, `python3 -m nase sweep` |

---

## 2. Phase Checklist

### Phase 1: Synthetic Manifolds with Known Noise

| Item | Status | Evidence |
|------|--------|----------|
| Circle/S1 | **DONE** | `results/20260227_203217_smoke_small/metrics.json`, `results/20260302_174322_noise_sweep_circle_sphere/` (9 runs) |
| Swiss roll | **DONE** | `results/20260227_202434_swiss_roll_stability/`, noise sweep, bandwidth sweep |
| Sphere/S2 | **DONE** (new) | `results/20260302_174322_noise_sweep_circle_sphere/` — 9 sphere runs across 3 noise levels |
| S-curve | **DONE** (new) | `results/20260302_173959_ambiguous_gap_suite/` — 6 runs, eigengap vs stability |
| Noise sweep (swiss roll) | **DONE** | `results/20260227_183932_synthetic_noise_sweep/` — 9 runs |
| Noise sweep (circle + sphere) | **DONE** (new) | `results/20260302_174322_noise_sweep_circle_sphere/` — 18 runs |
| Bandwidth grid sensitivity | **DONE** (new) | `results/20260302_173937_synthetic_bandwidth_sweep/` — narrow vs wide on swiss roll |
| Compare cutoffs (stability vs eigengap) | **DONE** | Computed in every run; dedicated head-to-head in ambiguous gap and eigengap-ambiguous suites |
| Oracle cutoff on synthetic | **DONE** | Computed in every run (`k_oracle` in metrics.json) |

### Ambiguous Eigengap Case

| Item | Status | Evidence |
|------|--------|----------|
| Sphere in ℝ⁶ (ambiguous gap) | **DONE** (new) | `results/20260302_174201_eigengap_ambiguous_suite/` — 6 runs |
| S-curve (eigengap vs stability) | **DONE** (new) | `results/20260302_173959_ambiguous_gap_suite/` — 6 runs |

### Phase 2: Intrinsic Dimension Estimation

| Item | Status | Evidence |
|------|--------|----------|
| Levina-Bickel estimator | **DONE** (code) | `src/nase/estimators/intrinsic_dimension.py` |
| Integrated into pipeline | **NOT DONE** | Not called from runner; standalone only |
| Evaluated on synthetic data | **NOT DONE** | No test of accuracy vs true d |

Per course feedback, dimension estimation is fragile in high dimensions and is a later-phase add-on. We treat intrinsic dimension as known for all experiments.

### Phase 3: Real Data

| Item | Status | Evidence |
|------|--------|----------|
| Any real dataset | **NOT DONE** | No real-data configs or results |

Optional per project scope. Synthetic results are the priority.

---

## 3. Existing Results Before This Audit

| # | Run ID | Manifold | Grid ε | k_stability | k_eigengap | Trust. |
|---|--------|----------|--------|-------------|-----------|--------|
| 1 | `20260227_203217_smoke_small` | circle | [0.7,1.0,1.4] | **6** | 2 | **0.999** |
| 2 | `20260227_202434_swiss_roll_stability` | swiss_roll | [0.5,1.0,2.0,3.0] | 1 | 1 | 0.645 |
| 3 | noise_sweep (9 runs) | swiss_roll | [0.5,1.0,2.0,3.0] | 1 (all) | 1–3 | 0.61–0.72 |

---

## 4. Experiments Added in This Audit

### Experiment A: Bandwidth Grid Sensitivity (Swiss Roll)

- **Command**: `python3 -m nase sweep --config configs/synthetic_bandwidth_sweep.yaml`
- **Output**: `results/20260302_173937_synthetic_bandwidth_sweep/`
- **Runs**: 4 (2 cases × 2 seeds)
- **Result**: Both narrow [0.5,0.8,1.1] and wide [0.6,1.2,2.4,3.0] grids still give k_stability=1. The swiss roll's eigenvectors are inherently unstable across bandwidth changes. However, trustworthiness is higher for narrow (0.984) vs wide (0.876) grids, confirming grid width matters for eigengap selection (k_eigengap=9-11 narrow vs 9-10 wide).

### Experiment B: Circle + Sphere Noise Sweep

- **Command**: `python3 -m nase sweep --config configs/noise_sweep_circle_sphere.yaml`
- **Output**: `results/20260302_174322_noise_sweep_circle_sphere/`
- **Runs**: 18 (6 cases × 3 seeds)
- **Result**: Strong success. Circle: k_stability decreases cleanly with noise (8→6→4), zero variance across seeds, trust > 0.98 at all levels. Sphere: k_stability also decreases (12→10-12→8), trust ~0.85. Eigengap is constant (k=2 for circle, k=3 for sphere) regardless of noise.

### Experiment C: Ambiguous Gap Suite (S-Curve)

- **Command**: `python3 -m nase sweep --config configs/ambiguous_gap_suite.yaml`
- **Output**: `results/20260302_173959_ambiguous_gap_suite/`
- **Runs**: 6 (2 cases × 3 seeds)
- **Result**: Eigengap picks k=1 (all seeds), stability picks k=13 (all seeds). Stability gives higher trust (0.914 vs 0.879). Both are seed-consistent here, but eigengap is trivially low.

### Experiment D: Eigengap-Ambiguous Sphere (ℝ⁶)

- **Command**: `python3 -m nase sweep --config configs/eigengap_ambiguous_suite.yaml`
- **Output**: `results/20260302_174201_eigengap_ambiguous_suite/`
- **Runs**: 6 (2 cases × 3 seeds)
- **Result**: Eigengap gives k=3 (stable across seeds). Stability gives k=8-9 with slight variance. Trustworthiness is identical (0.853) — the extra modes from stability neither help nor hurt.

---

## 5. Execution Log

### Step 0: Baseline verification
- `python3 -m pytest -q` → 47/47 PASS
- `python3 -m ruff check .` → All checks passed
- `python3 -m ruff format --check .` → 55 files already formatted

### Step 1: Experiment A — Bandwidth Grid Sensitivity
- Command: `python3 -m nase sweep --config configs/synthetic_bandwidth_sweep.yaml`
- Runtime: ~75s
- Output: `results/20260302_173937_synthetic_bandwidth_sweep/`
- Status: **DONE**

### Step 2: Experiment C — Ambiguous Gap Suite
- Command: `python3 -m nase sweep --config configs/ambiguous_gap_suite.yaml`
- Runtime: ~25s
- Output: `results/20260302_173959_ambiguous_gap_suite/`
- Status: **DONE**

### Step 3: Experiment D — Eigengap-Ambiguous Sphere
- Command: `python3 -m nase sweep --config configs/eigengap_ambiguous_suite.yaml`
- Runtime: ~117s
- Output: `results/20260302_174201_eigengap_ambiguous_suite/`
- Status: **DONE**

### Step 4: Experiment B — Circle + Sphere Noise Sweep
- Command: `python3 -m nase sweep --config configs/noise_sweep_circle_sphere.yaml`
- Runtime: ~120s
- Output: `results/20260302_174322_noise_sweep_circle_sphere/`
- Status: **DONE**

---

## 6. Output Paths Created

```
results/20260302_173937_synthetic_bandwidth_sweep/
  aggregate.json, manifest.json, records.csv, records.json, selected_k_comparison.png
  runs/ (4 sub-runs, each with config.yaml, metrics.json, cutoffs.json, arrays.npz, figures/)

results/20260302_173959_ambiguous_gap_suite/
  aggregate.json, manifest.json, records.csv, records.json, selected_k_comparison.png
  runs/ (6 sub-runs)

results/20260302_174201_eigengap_ambiguous_suite/
  aggregate.json, manifest.json, records.csv, records.json, selected_k_comparison.png
  runs/ (6 sub-runs)

results/20260302_174322_noise_sweep_circle_sphere/
  aggregate.json, manifest.json, records.csv, records.json, selected_k_comparison.png
  runs/ (18 sub-runs)
```

---

## 7. Failures and Resolutions

No failures encountered. All sweeps completed successfully.

---

## 8. What Remains Not Done (and Why)

| Item | Why not done | Priority |
|------|-------------|----------|
| Intrinsic dimension evaluation | Course feedback: fragile in high-D, treat as later-phase | Low |
| Real-data experiment | Optional per project scope; synthetic results sufficient for core claims | Low |
| Narrower swiss-roll bandwidth grid (e.g. [0.7,0.8,0.9]) | Could test if very tight grid helps, but bandwidth sweep already shows swiss roll is structurally hard | Medium |
| Subspace-based stability (principal angles) | Placeholder exists in code but not implemented — would address the individual-vector alignment limitation | Medium |
