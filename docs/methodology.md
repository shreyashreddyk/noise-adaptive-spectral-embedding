# Methodology

## Objective

The project studies how additive noise alters graph spectra and downstream manifold
embeddings. The practical objective is to choose a truncation level that retains stable
structure while excluding noise-sensitive modes.

## Experimental Setup

We begin with synthetic manifolds where the clean geometry is known. Gaussian noise is
added at controlled amplitudes so each run has a clear noise setting. This lets us compare
methods across low-, medium-, and high-noise regimes with reproducible random seeds.

## Graph and Spectral Pipeline

For each dataset, we construct pairwise relationships and build a kernelized graph operator.
After normalization, we compute leading eigenpairs and produce a spectral embedding from a
selected number of components.

## Cutoff Selection

The default strategy is bandwidth-stability truncation. We evaluate whether eigenvectors stay
consistent across a bandwidth grid; persistent modes are treated as signal. For comparison, we
also report an eigengap-based cutoff to highlight where gap heuristics become ambiguous under
noise.

## Evaluation and Outputs

Each run saves:

- configuration used for execution,
- selected cutoff and summary metrics,
- diagnostic metadata,
- plots for spectra, embeddings, and stability behavior.

Sweep experiments aggregate records across seeds and noise levels to estimate variance and
method robustness.
