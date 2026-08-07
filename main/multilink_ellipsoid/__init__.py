"""Opt-in whole-arm ellipsoid oracle for the AEGIS reproduction harness."""

from __future__ import annotations

from .active import (
    DistalThreeEllipsoidMultiCbf,
    load_active_config,
    resolved_rate_action_map,
    summarize_active_records,
)
from .barrier import PairConstraint, build_pair_constraint, support_gap
from .geometry import Ellipsoid, primitive_bounding_radii
from .qp import MultiConstraintQp, QpResult
from .shadow import MultilinkEllipsoidShadow, load_shadow_config

__all__ = [
    "Ellipsoid",
    "DistalThreeEllipsoidMultiCbf",
    "MultilinkEllipsoidShadow",
    "MultiConstraintQp",
    "PairConstraint",
    "QpResult",
    "build_pair_constraint",
    "load_active_config",
    "load_shadow_config",
    "primitive_bounding_radii",
    "resolved_rate_action_map",
    "summarize_active_records",
    "support_gap",
]
