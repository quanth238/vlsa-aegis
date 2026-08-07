"""Opt-in whole-arm ellipsoid oracle for the AEGIS reproduction harness."""

from __future__ import annotations

from .barrier import PairConstraint, build_pair_constraint, support_gap
from .geometry import Ellipsoid, primitive_bounding_radii
from .qp import MultiConstraintQp, QpResult
from .shadow import MultilinkEllipsoidShadow, load_shadow_config

__all__ = [
    "Ellipsoid",
    "MultilinkEllipsoidShadow",
    "MultiConstraintQp",
    "PairConstraint",
    "QpResult",
    "build_pair_constraint",
    "load_shadow_config",
    "primitive_bounding_radii",
    "support_gap",
]
