"""Counterfactual repair helpers.

`sine_bump_repair` is a deterministic dependency-free test oracle. It verifies
the harness and intervention mathematics but is not the SafeLIBERO research
teacher. The real experiment uses the constrained optimizer in the VLSA
integration and must verify every returned repair in MuJoCo.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from .geometry import SphereObstacle, actions_from_positions, rollout_positions, swept_sphere_clearance
from .math3d import Matrix, Vector, frobenius, matrix_subtract, norm, normalize, scale, subtract, vector


@dataclass(frozen=True)
class Repair:
    actions: Matrix
    correction: Matrix
    clearance_m: float
    endpoint_error_m: float
    objective: float
    direction: Vector


def sine_bump_repair(
    nominal_actions: Sequence[Sequence[float]],
    p0: Sequence[float],
    obstacle: SphereObstacle,
    eef_radius_m: float,
    safety_margin_m: float,
    kappa: float = 1.0,
    max_amplitude_m: float = 1.0,
    amplitude_step_m: float = 0.002,
    action_bound: float | None = None,
) -> Repair | None:
    """Find the smallest feasible sinusoidal lateral detour in two directions."""
    if safety_margin_m < 0:
        raise ValueError("safety_margin_m cannot be negative")
    if max_amplitude_m <= 0 or amplitude_step_m <= 0:
        raise ValueError("Amplitude search parameters must be positive")
    nominal_positions = rollout_positions(p0, nominal_actions, kappa)
    endpoint_direction = subtract(nominal_positions[-1], nominal_positions[0])
    if norm(endpoint_direction) <= 1e-12:
        return None
    directions = _lateral_directions(endpoint_direction)
    horizon = len(nominal_actions)
    best: Repair | None = None
    steps = int(max_amplitude_m / amplitude_step_m)
    for direction in directions:
        for index in range(steps + 1):
            amplitude = index * amplitude_step_m
            candidate_positions = tuple(
                vector(position)
                if h in (0, horizon)
                else tuple(
                    coordinate + amplitude * math.sin(math.pi * h / horizon) * lateral
                    for coordinate, lateral in zip(position, direction, strict=True)
                )
                for h, position in enumerate(nominal_positions)
            )
            candidate_actions = actions_from_positions(candidate_positions, kappa)
            if action_bound is not None and any(abs(value) > action_bound for row in candidate_actions for value in row):
                continue
            clearance = swept_sphere_clearance(candidate_positions, obstacle, eef_radius_m)
            if clearance + 1e-12 < safety_margin_m:
                continue
            correction = matrix_subtract(candidate_actions, nominal_actions)
            endpoint_error = norm(subtract(candidate_positions[-1], nominal_positions[-1]))
            candidate = Repair(
                actions=candidate_actions,
                correction=correction,
                clearance_m=clearance,
                endpoint_error_m=endpoint_error,
                objective=0.5 * frobenius(correction) ** 2,
                direction=direction,
            )
            if best is None or candidate.objective < best.objective:
                best = candidate
            break
    return best


def _lateral_directions(path: Sequence[float]) -> tuple[Vector, ...]:
    unit = normalize(path)
    axes = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    axis = min(axes, key=lambda candidate: abs(sum(a * b for a, b in zip(unit, candidate, strict=True))))
    cross = (
        unit[1] * axis[2] - unit[2] * axis[1],
        unit[2] * axis[0] - unit[0] * axis[2],
        unit[0] * axis[1] - unit[1] * axis[0],
    )
    lateral = normalize(cross)
    return lateral, scale(lateral, -1.0)
