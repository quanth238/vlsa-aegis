"""Controlled-pilot kinematics and swept sphere clearance."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .math3d import Matrix, Vector, add, dot, norm, scale, subtract, vector


@dataclass(frozen=True)
class SphereObstacle:
    center: Vector
    radius_m: float

    def __post_init__(self) -> None:
        if len(self.center) != 3:
            raise ValueError("Sphere center must be three-dimensional")
        if self.radius_m <= 0:
            raise ValueError("Sphere radius must be positive")


def rollout_positions(p0: Sequence[float], actions: Sequence[Sequence[float]], kappa: float) -> Matrix:
    if len(p0) != 3:
        raise ValueError("p0 must be three-dimensional")
    if kappa <= 0:
        raise ValueError("kappa must be positive")
    current = vector(p0)
    positions = [current]
    for action in actions:
        if len(action) != 3:
            raise ValueError("Only three-dimensional translation actions are supported")
        current = add(current, scale(action, kappa))
        positions.append(current)
    return tuple(positions)


def actions_from_positions(positions: Sequence[Sequence[float]], kappa: float) -> Matrix:
    if len(positions) < 2:
        raise ValueError("At least two positions are required")
    if kappa <= 0:
        raise ValueError("kappa must be positive")
    return tuple(scale(subtract(end, start), 1.0 / kappa) for start, end in zip(positions, positions[1:]))


def point_segment_distance(point: Sequence[float], start: Sequence[float], end: Sequence[float]) -> float:
    segment = subtract(end, start)
    denominator = dot(segment, segment)
    if denominator <= 1e-18:
        return norm(subtract(point, start))
    fraction = max(0.0, min(1.0, dot(subtract(point, start), segment) / denominator))
    closest = add(start, scale(segment, fraction))
    return norm(subtract(point, closest))


def swept_sphere_clearance(
    positions: Sequence[Sequence[float]],
    obstacle: SphereObstacle,
    eef_radius_m: float,
) -> float:
    if len(positions) < 2:
        raise ValueError("At least one swept segment is required")
    if eef_radius_m < 0:
        raise ValueError("EEF radius cannot be negative")
    inflated_radius = obstacle.radius_m + eef_radius_m
    return min(
        point_segment_distance(obstacle.center, start, end) - inflated_radius
        for start, end in zip(positions, positions[1:])
    )
