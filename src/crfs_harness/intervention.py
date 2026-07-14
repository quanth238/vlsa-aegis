"""Endpoint-preserving action corrections and oracle flow interventions."""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence

from .math3d import Matrix, Vector, add, frobenius, matrix_add, matrix_scale, scale, subtract

VelocityField = Callable[[Matrix, float], Matrix]


def endpoint_project(raw: Sequence[Sequence[float]]) -> Matrix:
    if not raw:
        raise ValueError("Correction cannot be empty")
    width = len(raw[0])
    if width == 0 or any(len(row) != width for row in raw):
        raise ValueError("Correction must be a rectangular matrix")
    means = tuple(sum(row[column] for row in raw) / len(raw) for column in range(width))
    return tuple(subtract(row, means) for row in raw)


def endpoint_sum(correction: Sequence[Sequence[float]]) -> Vector:
    if not correction:
        raise ValueError("Correction cannot be empty")
    width = len(correction[0])
    return tuple(sum(row[column] for row in correction) for column in range(width))


def equal_norm_random(
    reference: Sequence[Sequence[float]],
    seed: int,
) -> Matrix:
    rng = random.Random(seed)
    raw = tuple(tuple(rng.gauss(0.0, 1.0) for _ in row) for row in reference)
    projected = endpoint_project(raw)
    random_norm = frobenius(projected)
    target_norm = frobenius(reference)
    if target_norm <= 1e-15:
        return tuple(tuple(0.0 for _ in row) for row in reference)
    if random_norm <= 1e-15:
        raise RuntimeError("Random projection unexpectedly has zero norm")
    return matrix_scale(projected, target_norm / random_norm)


def residual_velocity(correction: Sequence[Sequence[float]], intervention_time: float) -> Matrix:
    if not 0.0 < intervention_time <= 1.0:
        raise ValueError("intervention_time must be in (0, 1]")
    return matrix_scale(correction, -1.0 / intervention_time)


def bridge_latent_edit(correction: Sequence[Sequence[float]], intervention_time: float) -> Matrix:
    if not 0.0 <= intervention_time <= 1.0:
        raise ValueError("intervention_time must be in [0, 1]")
    return matrix_scale(correction, 1.0 - intervention_time)


def integrate_remaining_flow(
    x_t: Matrix,
    base_velocity: VelocityField,
    intervention_time: float,
    num_steps: int,
    correction: Sequence[Sequence[float]] | None = None,
    one_shot_edit: bool = False,
) -> Matrix:
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    state = tuple(tuple(item for item in row) for row in x_t)
    if correction is not None and one_shot_edit:
        state = matrix_add(state, bridge_latent_edit(correction, intervention_time))
    residual = None if correction is None or one_shot_edit else residual_velocity(correction, intervention_time)
    dt = -intervention_time / num_steps
    time = intervention_time
    for _ in range(num_steps):
        velocity = base_velocity(state, time)
        if residual is not None:
            velocity = matrix_add(velocity, residual)
        state = matrix_add(state, matrix_scale(velocity, dt))
        time += dt
    return state
