"""Compatibility shim; use `nase.graphs.normalisations` going forward."""

from nase.graphs.normalisations import (  # noqa: F401
    alpha_normalise,
    alpha_normalise_dense,
    alpha_normalise_sparse,
    laplacian_eigenmaps_matrices,
    markov_matrix,
    row_normalise,
    row_normalise_dense,
    row_normalise_sparse,
)
