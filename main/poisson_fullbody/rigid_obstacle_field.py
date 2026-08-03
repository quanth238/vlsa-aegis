"""Poisson-field queries for one moving rigid obstacle.

The certified field is constructed once in a reference obstacle frame.  At
runtime, robot sample points are mapped back into that reference frame instead
of pretending that a free SafeLIBERO obstacle remains fixed.  The returned
time derivative is the Eulerian ``partial h / partial t`` required by the
time-varying CBF condition.

This module does not rebuild, dilate, or otherwise change the Poisson field.
It only applies a rigid transform and the measured obstacle twist.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Tuple


def _numpy() -> Any:
    try:
        import numpy as np
    except ImportError as error:  # pragma: no cover - allocation dependency
        raise RuntimeError("NumPy is required for rigid Poisson queries") from error
    return np


def _vector(value: Any, length: int, label: str) -> Any:
    np = _numpy()
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (length,) or not np.all(np.isfinite(array)):
        raise ValueError("%s must be a finite %d-vector" % (label, length))
    return np.array(array, dtype=np.float64, copy=True)


def _rotation(value: Any, label: str) -> Any:
    np = _numpy()
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("%s must be a finite 3x3 matrix" % label)
    if not (
        np.allclose(matrix.T @ matrix, np.eye(3), rtol=0.0, atol=1.0e-10)
        and math.isclose(float(np.linalg.det(matrix)), 1.0, rel_tol=0.0, abs_tol=1.0e-10)
    ):
        raise ValueError("%s must be a proper rotation matrix" % label)
    return np.array(matrix, dtype=np.float64, copy=True)


@dataclass(frozen=True)
class RigidPose:
    """World pose of the rigid obstacle reference body."""

    position_world_m: Any
    rotation_world_from_body: Any

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "position_world_m",
            _vector(self.position_world_m, 3, "position_world_m"),
        )
        object.__setattr__(
            self,
            "rotation_world_from_body",
            _rotation(
                self.rotation_world_from_body,
                "rotation_world_from_body",
            ),
        )


@dataclass(frozen=True)
class RigidTwist:
    """World-frame linear and angular velocity of the obstacle body origin."""

    linear_world_m_per_s: Any
    angular_world_rad_per_s: Any

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "linear_world_m_per_s",
            _vector(
                self.linear_world_m_per_s,
                3,
                "linear_world_m_per_s",
            ),
        )
        object.__setattr__(
            self,
            "angular_world_rad_per_s",
            _vector(
                self.angular_world_rad_per_s,
                3,
                "angular_world_rad_per_s",
            ),
        )


@dataclass(frozen=True)
class RigidFieldQuery:
    """One time-varying field query in current world coordinates."""

    valid: bool
    value_m2: Any
    gradient_world_m: Any
    partial_time_m2_per_s: Any
    reference_point_world_m: Any
    obstacle_point_velocity_world_m_per_s: Any
    reason: Any
    cell_index: Any
    local_coordinates: Any
    outer_boundary_clearance_m: Any


def map_world_point_to_reference(
    point_world_m: Any,
    *,
    reference_pose: RigidPose,
    current_pose: RigidPose,
) -> Any:
    """Map a current-world point into the field's reference-world frame."""

    point = _vector(point_world_m, 3, "point_world_m")
    current_to_reference = (
        reference_pose.rotation_world_from_body
        @ current_pose.rotation_world_from_body.T
    )
    return (
        reference_pose.position_world_m
        + current_to_reference @ (point - current_pose.position_world_m)
    )


def query_rigid_obstacle_field(
    field: Any,
    point_world_m: Any,
    *,
    reference_pose: RigidPose,
    current_pose: RigidPose,
    current_twist: RigidTwist,
) -> RigidFieldQuery:
    """Query ``h(x,t)`` and its explicit time derivative.

    With ``x_ref = p_ref + R_ref R(t)^T (x - p(t))`` and
    ``h(x,t) = h_ref(x_ref)``, the world gradient and explicit derivative are

    ``grad_x h = R(t) R_ref^T grad_ref h`` and
    ``partial_t h = -grad_x h dot (v + omega cross (x - p))``.
    """

    np = _numpy()
    point = _vector(point_world_m, 3, "point_world_m")
    reference_point = map_world_point_to_reference(
        point,
        reference_pose=reference_pose,
        current_pose=current_pose,
    )
    raw = field.query(reference_point)
    raw_valid = bool(getattr(raw, "valid", False))
    raw_value = getattr(raw, "value", None)
    raw_gradient = getattr(raw, "gradient", None)
    raw_reason = getattr(raw, "reason", None)
    raw_cell_index = getattr(raw, "cell_index", None)
    raw_local_coordinates = getattr(raw, "local_coordinates", None)
    raw_outer_boundary_clearance = getattr(
        raw,
        "outer_boundary_clearance_m",
        None,
    )
    if not raw_valid or raw_value is None or raw_gradient is None:
        return RigidFieldQuery(
            valid=False,
            value_m2=None,
            gradient_world_m=None,
            partial_time_m2_per_s=None,
            reference_point_world_m=reference_point,
            obstacle_point_velocity_world_m_per_s=None,
            reason=raw_reason,
            cell_index=raw_cell_index,
            local_coordinates=raw_local_coordinates,
            outer_boundary_clearance_m=raw_outer_boundary_clearance,
        )
    value = float(raw_value)
    gradient_reference = _vector(raw_gradient, 3, "field gradient")
    if not math.isfinite(value):
        raise ValueError("field value must be finite")
    reference_to_current = (
        current_pose.rotation_world_from_body
        @ reference_pose.rotation_world_from_body.T
    )
    gradient_world = reference_to_current @ gradient_reference
    radius = point - current_pose.position_world_m
    obstacle_point_velocity = (
        current_twist.linear_world_m_per_s
        + np.cross(current_twist.angular_world_rad_per_s, radius)
    )
    partial_time = -float(gradient_world @ obstacle_point_velocity)
    return RigidFieldQuery(
        valid=True,
        value_m2=value,
        gradient_world_m=gradient_world,
        partial_time_m2_per_s=partial_time,
        reference_point_world_m=reference_point,
        obstacle_point_velocity_world_m_per_s=obstacle_point_velocity,
        reason=raw_reason,
        cell_index=raw_cell_index,
        local_coordinates=raw_local_coordinates,
        outer_boundary_clearance_m=raw_outer_boundary_clearance,
    )


def query_rigid_obstacle_field_batch(
    field: Any,
    points_world_m: Iterable[Any],
    *,
    reference_pose: RigidPose,
    current_pose: RigidPose,
    current_twist: RigidTwist,
) -> Tuple[RigidFieldQuery, ...]:
    """Query a sequence while preserving its sample order."""

    return tuple(
        query_rigid_obstacle_field(
            field,
            point,
            reference_pose=reference_pose,
            current_pose=current_pose,
            current_twist=current_twist,
        )
        for point in points_world_m
    )


def dynamic_cbf_rows(
    h_m2: Any,
    gradients_world_m: Any,
    point_jacobians_m_per_rad: Any,
    partial_time_m2_per_s: Any,
    *,
    alpha_per_s: float,
    margin_m2_per_s: float = 0.0,
) -> Tuple[Any, Any]:
    """Return ``A, b`` for the moving-field constraint ``A qdot >= b``.

    This is exactly

    ``grad(h) J qdot + partial_t(h) + alpha h - margin >= 0``.
    """

    np = _numpy()
    h = np.asarray(h_m2, dtype=np.float64)
    gradients = np.asarray(gradients_world_m, dtype=np.float64)
    jacobians = np.asarray(point_jacobians_m_per_rad, dtype=np.float64)
    partial_time = np.asarray(partial_time_m2_per_s, dtype=np.float64)
    if h.ndim != 1:
        raise ValueError("h_m2 must be one-dimensional")
    count = int(h.size)
    if gradients.shape != (count, 3):
        raise ValueError("gradients_world_m must have shape (N, 3)")
    if jacobians.shape != (count, 3, 7):
        raise ValueError("point_jacobians_m_per_rad must have shape (N, 3, 7)")
    if partial_time.shape != (count,):
        raise ValueError("partial_time_m2_per_s must have shape (N,)")
    if not all(
        np.all(np.isfinite(value))
        for value in (h, gradients, jacobians, partial_time)
    ):
        raise ValueError("dynamic CBF inputs must be finite")
    alpha = float(alpha_per_s)
    margin = float(margin_m2_per_s)
    if not math.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("alpha_per_s must be finite and positive")
    if not math.isfinite(margin) or margin < 0.0:
        raise ValueError("margin_m2_per_s must be finite and nonnegative")
    rows = np.einsum("ni,nij->nj", gradients, jacobians)
    lower = -alpha * h - partial_time + margin
    return rows, lower


__all__ = [
    "RigidFieldQuery",
    "RigidPose",
    "RigidTwist",
    "dynamic_cbf_rows",
    "map_world_point_to_reference",
    "query_rigid_obstacle_field",
    "query_rigid_obstacle_field_batch",
]
