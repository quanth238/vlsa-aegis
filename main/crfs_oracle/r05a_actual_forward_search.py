"""Frozen derivative-free actual-forward teacher search for AF-00A.

This module is pure NumPy.  It never imports a model, simulator, or optimizer
framework.  The caller supplies the one ordinary-policy evaluation callback;
all proposal, transport, constraint, objective, update, and final-selection
semantics are fixed here so the CPU publisher can recompute them independently.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import numpy as np


DT_FLOAT32 = np.float32(-0.1)
SEARCH_SEED = 20260716
GENERATIONS = 8
POPULATION_SIZE = 65
ANTITHETIC_PAIRS = 32
ELITE_COUNT = 13
COMPACT_SHAPE = (5, 15)
NORMALIZED_SHAPE = (10, 32)
ACTION_SHAPE = (10, 7)
ERROR_SHAPE = (5, 7)
CONSTRAINT_SLACK_ULPS = 8


class ActualForwardSearchError(RuntimeError):
    """The fixed search or one policy evaluation violated its contract."""


@dataclass(frozen=True)
class ActualForwardEvaluation:
    """Authoritative scientific output of one ordinary schedule request."""

    executed_increment: np.ndarray
    normalized_final: np.ndarray
    returned_physical_action: np.ndarray
    physical_error: np.ndarray


@dataclass(frozen=True)
class ActualForwardCandidate:
    generation: int
    population_index: int
    cem_query_index: int
    global_policy_request_index: int
    raw_proposal: np.ndarray
    projected_increment: np.ndarray
    requested_velocity: np.ndarray
    evaluation: ActualForwardEvaluation
    objective: float
    energy: float
    metrics: tuple[float, float, float, float]
    gates: tuple[bool, bool, bool, bool]

    @property
    def passed(self) -> bool:
        return all(self.gates)


@dataclass(frozen=True)
class ActualForwardSelection:
    source: str
    pool_index: int
    cem_query_index: int | None
    global_policy_request_index: int
    requested_velocity: np.ndarray
    evaluation: ActualForwardEvaluation
    objective: float
    energy: float
    metrics: tuple[float, float, float, float]
    gates: tuple[bool, bool, bool, bool]

    @property
    def passed(self) -> bool:
        return all(self.gates)


@dataclass(frozen=True)
class ActualForwardCEMResult:
    numpy_version: str
    budget_float32: np.float32
    radius_float32: np.float32
    initial_sigma_float64: float
    raw_normals: np.ndarray
    mean_states: np.ndarray
    sigma_states: np.ndarray
    variance_states: np.ndarray
    elite_cem_query_indices: np.ndarray
    candidates: tuple[ActualForwardCandidate, ...]
    arm_a: ActualForwardSelection
    selected: ActualForwardSelection


def _finite_exact_array(
    value: np.ndarray, *, name: str, shape: tuple[int, ...], dtype: np.dtype
) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape:
        raise ActualForwardSearchError(f"{name} shape {array.shape} != {shape}")
    if array.dtype != dtype:
        raise ActualForwardSearchError(f"{name} dtype {array.dtype} != {dtype}")
    if not bool(np.isfinite(array).all()):
        raise ActualForwardSearchError(f"{name} contains nonfinite values")
    return np.ascontiguousarray(array)


def project_product_ball(value: np.ndarray, radius_float64: float) -> np.ndarray:
    """Project five float64 15-vectors onto the registered product ball."""

    array = _finite_exact_array(
        value, name="proposal", shape=COMPACT_SHAPE, dtype=np.dtype(np.float64)
    ).copy()
    if not math.isfinite(radius_float64) or radius_float64 <= 0:
        raise ActualForwardSearchError("radius must be finite and positive")
    for row in range(COMPACT_SHAPE[0]):
        norm = float(
            np.sqrt(
                np.sum(np.square(array[row]), dtype=np.float64),
                dtype=np.float64,
            )
        )
        if norm == 0.0:
            array[row] = np.zeros(COMPACT_SHAPE[1], dtype=np.float64)
        elif norm > radius_float64:
            array[row] *= radius_float64 / norm
    return np.ascontiguousarray(array)


def transport_increment(
    projected_float64: np.ndarray, *, dt_float32: np.float32 = DT_FLOAT32
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the frozen c64 -> c32 -> u32 -> dt32*u32 transport order."""

    projected = _finite_exact_array(
        projected_float64,
        name="projected increment",
        shape=COMPACT_SHAPE,
        dtype=np.dtype(np.float64),
    )
    dt = np.asarray(dt_float32)
    if dt.shape != () or dt.dtype != np.dtype(np.float32) or dt.tobytes() != DT_FLOAT32.tobytes():
        raise ActualForwardSearchError("dt must be exact float32 -0.1")
    c32 = np.ascontiguousarray(projected.astype(np.float32))
    u32 = np.ascontiguousarray((c32 / dt).astype(np.float32))
    executed = np.ascontiguousarray((dt * u32).astype(np.float32))
    return c32, u32, executed


def validate_executed_constraints(
    executed_increment: np.ndarray, budget_float32: np.float32
) -> tuple[np.ndarray, np.float32]:
    """Apply the existing float32 eight-ULP cap and path rule."""

    executed = _finite_exact_array(
        executed_increment,
        name="executed increment",
        shape=COMPACT_SHAPE,
        dtype=np.dtype(np.float32),
    )
    budget = np.asarray(budget_float32)
    if budget.shape != () or budget.dtype != np.dtype(np.float32) or not bool(np.isfinite(budget)):
        raise ActualForwardSearchError("budget must be one finite float32 scalar")
    if float(budget) <= 0:
        raise ActualForwardSearchError("budget must be positive")
    per_step = np.linalg.norm(executed.reshape(5, -1), axis=1).astype(np.float32)
    path = np.asarray(np.sum(per_step, dtype=np.float32), dtype=np.float32)
    allowed_budget = budget.copy()
    allowed_cap = np.asarray(budget / np.float32(5.0), dtype=np.float32)
    for _ in range(CONSTRAINT_SLACK_ULPS):
        allowed_budget = np.nextafter(allowed_budget, np.float32(np.inf), dtype=np.float32)
        allowed_cap = np.nextafter(allowed_cap, np.float32(np.inf), dtype=np.float32)
    if bool(np.any(per_step > allowed_cap)):
        raise ActualForwardSearchError("executed increment exceeds the eight-ULP per-step cap")
    if bool(path > allowed_budget):
        raise ActualForwardSearchError("executed increment exceeds the eight-ULP path budget")
    return np.ascontiguousarray(per_step), path


def objective_and_gates(
    physical_error: np.ndarray,
) -> tuple[float, tuple[float, float, float, float], tuple[bool, bool, bool, bool]]:
    """Compute the frozen float64 objective, energy-independent gates."""

    error = np.asarray(physical_error)
    if error.shape != ERROR_SHAPE or not np.issubdtype(error.dtype, np.number):
        raise ActualForwardSearchError("physical error must be numeric shape (5,7)")
    error64 = np.ascontiguousarray(error, dtype=np.float64)
    if not bool(np.isfinite(error64).all()):
        raise ActualForwardSearchError("physical error contains nonfinite values")
    scaled = np.concatenate(
        ((error64[:, :3] / 0.005).reshape(-1), (error64[:, 3:7] / 0.015).reshape(-1))
    )
    objective = float(np.mean(np.square(scaled), dtype=np.float64))
    xyz = error64[:, :3]
    full = error64
    metrics = (
        float(np.max(np.abs(xyz))),
        float(np.sqrt(np.sum(np.square(xyz), dtype=np.float64) / 15.0)),
        float(np.max(np.abs(full))),
        float(np.sqrt(np.sum(np.square(full), dtype=np.float64) / 35.0)),
    )
    gates = (
        metrics[0] <= 0.010,
        metrics[1] <= 0.005,
        metrics[2] <= 0.050,
        metrics[3] <= 0.015,
    )
    return objective, metrics, gates


def _validated_evaluation(
    value: ActualForwardEvaluation,
    *,
    expected_executed: np.ndarray,
) -> ActualForwardEvaluation:
    if not isinstance(value, ActualForwardEvaluation):
        raise ActualForwardSearchError("callback must return ActualForwardEvaluation")
    executed = _finite_exact_array(
        value.executed_increment,
        name="callback executed increment",
        shape=COMPACT_SHAPE,
        dtype=np.dtype(np.float32),
    )
    if executed.tobytes() != np.ascontiguousarray(expected_executed).tobytes():
        raise ActualForwardSearchError("callback executed increment differs from exact dt32*u32")
    final = _finite_exact_array(
        value.normalized_final,
        name="callback normalized final",
        shape=NORMALIZED_SHAPE,
        dtype=np.dtype(np.float32),
    )
    actions = np.asarray(value.returned_physical_action)
    if actions.shape != ACTION_SHAPE or not np.issubdtype(actions.dtype, np.number) or not bool(np.isfinite(actions).all()):
        raise ActualForwardSearchError("callback actions must be finite numeric shape (10,7)")
    error = np.asarray(value.physical_error)
    if error.shape != ERROR_SHAPE or not np.issubdtype(error.dtype, np.number) or not bool(np.isfinite(error).all()):
        raise ActualForwardSearchError("callback physical error must be finite numeric shape (5,7)")
    return ActualForwardEvaluation(
        executed_increment=executed.copy(),
        normalized_final=final.copy(),
        returned_physical_action=np.ascontiguousarray(actions.copy()),
        physical_error=np.ascontiguousarray(error.copy()),
    )


def _selection(
    *,
    source: str,
    pool_index: int,
    cem_query_index: int | None,
    global_policy_request_index: int,
    velocity: np.ndarray,
    evaluation: ActualForwardEvaluation,
) -> ActualForwardSelection:
    objective, metrics, gates = objective_and_gates(evaluation.physical_error)
    energy = float(
        np.sum(np.square(evaluation.executed_increment.astype(np.float64)), dtype=np.float64)
    )
    return ActualForwardSelection(
        source=source,
        pool_index=pool_index,
        cem_query_index=cem_query_index,
        global_policy_request_index=global_policy_request_index,
        requested_velocity=np.ascontiguousarray(velocity.copy()),
        evaluation=evaluation,
        objective=objective,
        energy=energy,
        metrics=metrics,
        gates=gates,
    )


def run_actual_forward_cem(
    evaluate: Callable[[np.ndarray], ActualForwardEvaluation],
    *,
    arm_a_velocity: np.ndarray,
    arm_a_evaluation: ActualForwardEvaluation,
    budget_float32: np.float32,
    dt_float32: np.float32 = DT_FLOAT32,
) -> ActualForwardCEMResult:
    """Run the exact 8x65 AF-00A CEM and final ADR-0028 selection."""

    velocity_a = _finite_exact_array(
        arm_a_velocity,
        name="Arm-A velocity",
        shape=COMPACT_SHAPE,
        dtype=np.dtype(np.float32),
    )
    expected_a = np.ascontiguousarray((np.asarray(dt_float32) * velocity_a).astype(np.float32))
    eval_a = _validated_evaluation(arm_a_evaluation, expected_executed=expected_a)
    validate_executed_constraints(eval_a.executed_increment, budget_float32)
    arm_a = _selection(
        source="A_equal_split",
        pool_index=0,
        cem_query_index=None,
        global_policy_request_index=4,
        velocity=velocity_a,
        evaluation=eval_a,
    )

    budget = np.asarray(budget_float32, dtype=np.float32).reshape(())
    radius32 = np.asarray(budget / np.float32(5.0), dtype=np.float32)
    radius64 = float(np.float64(radius32))
    sigma0 = radius64 / math.sqrt(15.0)
    mean = eval_a.executed_increment.astype(np.float64)
    sigma = np.full(COMPACT_SHAPE, sigma0, dtype=np.float64)
    rng = np.random.Generator(np.random.PCG64(SEARCH_SEED))
    raw_normals = np.empty((GENERATIONS, ANTITHETIC_PAIRS, *COMPACT_SHAPE), dtype=np.float64)
    means = np.empty((GENERATIONS + 1, *COMPACT_SHAPE), dtype=np.float64)
    sigmas = np.empty_like(means)
    variances = np.empty((GENERATIONS, *COMPACT_SHAPE), dtype=np.float64)
    elites = np.empty((GENERATIONS, ELITE_COUNT), dtype=np.int64)
    means[0] = mean
    sigmas[0] = sigma
    candidates: list[ActualForwardCandidate] = []

    for generation in range(GENERATIONS):
        z = rng.standard_normal((ANTITHETIC_PAIRS, *COMPACT_SHAPE), dtype=np.float64)
        raw_normals[generation] = z
        proposals = [mean.copy()]
        for pair in range(ANTITHETIC_PAIRS):
            proposals.append(mean + sigma * z[pair])
            proposals.append(mean - sigma * z[pair])
        generation_candidates: list[ActualForwardCandidate] = []
        for population_index, raw in enumerate(proposals):
            query = generation * POPULATION_SIZE + population_index
            projected64 = project_product_ball(np.ascontiguousarray(raw), radius64)
            projected32, velocity32, expected_executed = transport_increment(
                projected64, dt_float32=dt_float32
            )
            evaluation = _validated_evaluation(
                evaluate(velocity32.copy()), expected_executed=expected_executed
            )
            validate_executed_constraints(evaluation.executed_increment, budget)
            objective, metrics, gates = objective_and_gates(evaluation.physical_error)
            energy = float(
                np.sum(
                    np.square(evaluation.executed_increment.astype(np.float64)),
                    dtype=np.float64,
                )
            )
            candidate = ActualForwardCandidate(
                generation=generation,
                population_index=population_index,
                cem_query_index=query,
                global_policy_request_index=6 + query,
                raw_proposal=np.ascontiguousarray(raw.copy()),
                projected_increment=projected32,
                requested_velocity=velocity32,
                evaluation=evaluation,
                objective=objective,
                energy=energy,
                metrics=metrics,
                gates=gates,
            )
            candidates.append(candidate)
            generation_candidates.append(candidate)
        ranked = sorted(generation_candidates, key=lambda item: (item.objective, item.global_policy_request_index))
        chosen = ranked[:ELITE_COUNT]
        elites[generation] = np.asarray([item.cem_query_index for item in chosen], dtype=np.int64)
        elite_executed = np.stack(
            [item.evaluation.executed_increment.astype(np.float64) for item in chosen], axis=0
        )
        mean = project_product_ball(
            np.ascontiguousarray(np.mean(elite_executed, axis=0, dtype=np.float64)), radius64
        )
        variance = np.mean(np.square(elite_executed - mean), axis=0, dtype=np.float64)
        sigma = np.clip(np.sqrt(variance), sigma0 / 16.0, sigma0)
        variances[generation] = variance
        means[generation + 1] = mean
        sigmas[generation + 1] = sigma

    pool: list[ActualForwardSelection] = [arm_a]
    pool.extend(
        _selection(
            source="B_actual_forward_cem",
            pool_index=1 + item.cem_query_index,
            cem_query_index=item.cem_query_index,
            global_policy_request_index=item.global_policy_request_index,
            velocity=item.requested_velocity,
            evaluation=item.evaluation,
        )
        for item in candidates
    )
    feasible = [item for item in pool if item.passed]
    if feasible:
        selected = min(
            feasible,
            key=lambda item: (item.energy, item.objective, item.global_policy_request_index),
        )
    else:
        selected = min(
            pool,
            key=lambda item: (item.objective, item.energy, item.global_policy_request_index),
        )
    return ActualForwardCEMResult(
        numpy_version=np.__version__,
        budget_float32=np.float32(budget),
        radius_float32=np.float32(radius32),
        initial_sigma_float64=float(sigma0),
        raw_normals=raw_normals,
        mean_states=means,
        sigma_states=sigmas,
        variance_states=variances,
        elite_cem_query_indices=elites,
        candidates=tuple(candidates),
        arm_a=arm_a,
        selected=selected,
    )
