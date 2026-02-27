# Methodological Note: Choosing Spectral Cutoffs in NASE

## Motivation

Diffusion-style embeddings rely on truncating a spectral expansion at some `k`.  
In noisy settings, high-index modes can capture noise rather than geometry, so selecting `k` is a practical model-selection problem.

## Core Logic

NASE uses a **bandwidth-stability criterion**:

1. Build graph operators across a grid of kernel bandwidths (`epsilon` values).
2. Compute leading eigenvectors for each bandwidth.
3. Measure per-mode alignment across bandwidths.
4. Select the largest `k` where stability remains above a chosen threshold.

This uses empirical diagnostics rather than pretending unknown geometry constants can be solved exactly from finite noisy samples.

## Baseline Comparator

NASE also computes an **eigengap-based** cutoff:

- locate the largest gap among consecutive ordered eigenvalues in a configured range,
- return the associated index as baseline `k`.

Ambiguous-gap experiments are included because eigengaps are often non-unique in noisy problems.

## Noise-Amplitude Position

For early synthetic experiments, noise amplitude `r` is treated as known because it is controlled by the data generator.
The core pipeline does **not** depend on nearest-neighbour heuristics for `r`.

Optional later-phase ideas (intrinsic-dimension and DSS-based proxies) should remain clearly marked as exploratory.

## Diagnostics to Inspect

- spectrum plots,
- stability-vs-mode curves,
- embedding quality metrics (trustworthiness, continuity),
- geodesic consistency relative to known manifold coordinates.

## References

- Coifman, R. R., & Lafon, S. (2006). Diffusion maps.
- Follow-up noise-robust manifold learning literature should be cited as integrated.
