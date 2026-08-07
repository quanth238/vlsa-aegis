"""Differentiable support-gap barriers for link and obstacle ellipsoids."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

from .geometry import Ellipsoid


def _numpy() -> Any:
    try:
        import numpy as np
    except ImportError as error:  # pragma: no cover - allocation dependency
        raise RuntimeError("ellipsoid barrier evaluation requires NumPy") from error
    return np


@dataclass(frozen=True)
class PairConstraint:
    body_name: str
    geom_name: str
    h_opt_m: float
    unbuffered_gap_m: float
    optimizer_clearance_m: float
    row: Any
    lower: float
    center_distance_m: float
    robot_support_radius_m: float
    obstacle_support_radius_m: float

    def to_record(self) -> dict[str, Any]:
        np = _numpy()
        row = np.asarray(self.row, dtype=np.float64)
        return {
            "body_name": self.body_name,
            "geom_name": self.geom_name,
            "h_opt_m": float(self.h_opt_m),
            "unbuffered_gap_m": float(self.unbuffered_gap_m),
            "optimizer_clearance_m": float(self.optimizer_clearance_m),
            "cbf_row_m_per_rad": row.tolist(),
            "cbf_lower_m_per_s": float(self.lower),
            "center_distance_m": float(self.center_distance_m),
            "robot_support_radius_m": float(self.robot_support_radius_m),
            "obstacle_support_radius_m": float(self.obstacle_support_radius_m),
        }


def _gap_terms(robot: Ellipsoid, obstacle: Ellipsoid) -> tuple[Any, ...]:
    np = _numpy()
    displacement = obstacle.center - robot.center
    distance = float(np.linalg.norm(displacement))
    if not math.isfinite(distance) or distance <= 1.0e-12:
        raise ValueError("ellipsoid centers are coincident or invalid")
    direction = displacement / distance
    robot_shape = robot.shape_matrix()
    obstacle_shape = obstacle.shape_matrix()
    robot_vector = robot_shape @ direction
    obstacle_vector = obstacle_shape @ direction
    robot_radius = math.sqrt(float(direction @ robot_vector))
    obstacle_radius = math.sqrt(float(direction @ obstacle_vector))
    if robot_radius <= 0.0 or obstacle_radius <= 0.0:
        raise ValueError("ellipsoid support radius is degenerate")
    gap = distance - robot_radius - obstacle_radius
    return (
        displacement,
        distance,
        direction,
        robot_shape,
        obstacle_shape,
        robot_vector,
        obstacle_vector,
        robot_radius,
        obstacle_radius,
        gap,
    )


def support_gap(
    robot: Ellipsoid,
    obstacle: Ellipsoid,
    *,
    optimizer_clearance_m: float = 0.0,
) -> float:
    clearance = float(optimizer_clearance_m)
    if not math.isfinite(clearance) or clearance < 0.0:
        raise ValueError("optimizer_clearance_m must be finite and nonnegative")
    return float(_gap_terms(robot, obstacle)[-1] - clearance)


def twist_coefficients(robot: Ellipsoid, obstacle: Ellipsoid) -> tuple[Any, Any]:
    """Return coefficients on the robot geom's world linear/angular twist."""

    np = _numpy()
    (
        _,
        distance,
        direction,
        _,
        _,
        robot_vector,
        obstacle_vector,
        robot_radius,
        obstacle_radius,
        _,
    ) = _gap_terms(robot, obstacle)
    projector = np.eye(3) - np.outer(direction, direction)
    linear = (
        -direction
        + projector @ robot_vector / (robot_radius * distance)
        + projector @ obstacle_vector / (obstacle_radius * distance)
    )
    angular = np.cross(direction, robot_vector) / robot_radius
    if not np.all(np.isfinite(linear)) or not np.all(np.isfinite(angular)):
        raise ValueError("derived ellipsoid twist coefficient is nonfinite")
    return linear, angular


def build_pair_constraint(
    robot: Ellipsoid,
    obstacle: Ellipsoid,
    point_jacobian: Any,
    angular_jacobian: Any,
    *,
    alpha: float,
    optimizer_clearance_m: float,
) -> PairConstraint:
    np = _numpy()
    jac_position = np.asarray(point_jacobian, dtype=np.float64)
    jac_rotation = np.asarray(angular_jacobian, dtype=np.float64)
    if jac_position.ndim != 2 or jac_position.shape[0] != 3:
        raise ValueError("point_jacobian must have shape (3, n)")
    if jac_rotation.shape != jac_position.shape:
        raise ValueError("angular_jacobian must match point_jacobian")
    if not np.all(np.isfinite(jac_position)) or not np.all(np.isfinite(jac_rotation)):
        raise ValueError("ellipsoid Jacobians must be finite")
    gain = float(alpha)
    clearance = float(optimizer_clearance_m)
    if not math.isfinite(gain) or gain <= 0.0:
        raise ValueError("alpha must be finite and positive")
    if not math.isfinite(clearance) or clearance < 0.0:
        raise ValueError("optimizer_clearance_m must be finite and nonnegative")
    linear, angular = twist_coefficients(robot, obstacle)
    row = linear @ jac_position + angular @ jac_rotation
    if not np.all(np.isfinite(row)):
        raise ValueError("joint-space ellipsoid constraint row is nonfinite")
    terms = _gap_terms(robot, obstacle)
    unbuffered = float(terms[-1])
    h_opt = unbuffered - clearance
    return PairConstraint(
        body_name=robot.body_name,
        geom_name=robot.geom_name,
        h_opt_m=h_opt,
        unbuffered_gap_m=unbuffered,
        optimizer_clearance_m=clearance,
        row=np.asarray(row, dtype=np.float64),
        lower=-gain * h_opt,
        center_distance_m=float(terms[1]),
        robot_support_radius_m=float(terms[7]),
        obstacle_support_radius_m=float(terms[8]),
    )
