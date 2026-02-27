"""Robust-method scaffolding modules."""

from nase.robust.dss import SinkhornResult, maybe_doubly_stochastic_scale, sinkhorn_knopp_scale

__all__ = ["SinkhornResult", "sinkhorn_knopp_scale", "maybe_doubly_stochastic_scale"]
