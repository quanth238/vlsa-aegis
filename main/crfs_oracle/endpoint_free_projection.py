"""Deterministic endpoint-free search over the frozen H04 kinematic model.

This module is deliberately additive to the stopped CRFS experiment.  It
searches all five translational commands independently, keeps every other
nominal action channel unchanged, and uses ``D_opt`` only to nominate actions
for later ``D_sim`` verification.  Exhausting the bounded search returns
``not_found_within_budget``; it is never represented as a geometric
certificate.

The implementation uses the Python standard library so its search contract can
be tested by the dependency-free repository bootstrap.  Production callers may
pass NumPy arrays because inputs are copied into ordinary Python tuples at the
module boundary.
"""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from numbers import Real
from typing import Callable, Iterable, Sequence


ProgressFn = Callable[[tuple[tuple[float, float, float], ...]], float]


@dataclass(frozen=True)
class EndpointFreeCandidate:
    """One proxy-evaluated action prefix proposed for simulator verification."""

    source: str
    actions: list[list[float]]
    d_opt_m: float
    progress_opt: float
    clearance_pass: bool
    progress_pass: bool
    objective: float
    within_h04_calibration_domain: bool

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class EndpointFreeSearchResult:
    """Bounded-search output; candidates still require independent ``D_sim``."""

    outcome: str
    candidates: list[EndpointFreeCandidate]
    best_attempt: EndpointFreeCandidate
    maximum_clearance_attempt: EndpointFreeCandidate
    maximum_progress_attempt: EndpointFreeCandidate
    controls: list[EndpointFreeCandidate]
    evaluations: int
    evaluation_budget: int
    generated_samples: int
    seed: int
    optimizer_clearance_margin_m: float
    minimum_progress: float
    message: str

    @property
    def candidate_found(self) -> bool:
        return bool(self.candidates)

    def to_dict(self) -> dict:
        value = asdict(self)
        value["candidate_found"] = self.candidate_found
        return value


@dataclass(frozen=True)
class _Evaluated:
    source: str
    translation: tuple[tuple[float, float, float], ...]
    d_opt_m: float
    progress_opt: float
    objective: float


def _plain(value):
    return value.tolist() if hasattr(value, "tolist") else value


def _matrix(value, *, rows: int | None = None, min_columns: int = 1, name: str) -> list[list[float]]:
    raw = _plain(value)
    try:
        result = [[float(item) for item in _plain(row)] for row in raw]
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a rectangular numeric matrix") from error
    if rows is not None and len(result) != rows:
        raise ValueError(f"{name} must have {rows} rows")
    if not result or any(len(row) < min_columns for row in result):
        raise ValueError(f"{name} must have at least {min_columns} columns")
    width = len(result[0])
    if any(len(row) != width for row in result):
        raise ValueError(f"{name} must be rectangular")
    if any(not math.isfinite(item) for row in result for item in row):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _vector3(value, *, name: str) -> tuple[float, float, float]:
    raw = _plain(value)
    try:
        result = tuple(float(item) for item in raw)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a numeric length-three vector") from error
    if len(result) != 3 or any(not math.isfinite(item) for item in result):
        raise ValueError(f"{name} must be a finite length-three vector")
    return result  # type: ignore[return-value]


def _translation_bound_matrix(
    value, *, rows: int, action_width: int, name: str
) -> list[list[float]]:
    raw = _plain(value)
    if isinstance(raw, Real):
        return [[float(raw)] * 3 for _ in range(rows)]
    try:
        items = list(raw)
    except TypeError as error:
        raise ValueError(f"{name} is not broadcastable to ({rows}, 3)") from error
    if len(items) in (3, action_width) and all(isinstance(_plain(item), Real) for item in items):
        row = [float(_plain(item)) for item in items[:3]]
        return [row.copy() for _ in range(rows)]
    matrix = _matrix(items, rows=rows, min_columns=3, name=name)
    if any(len(row) not in (3, action_width) for row in matrix):
        raise ValueError(f"{name} is not broadcastable to ({rows}, 3)")
    return [row[:3] for row in matrix]


def _rotation_matrix(value, *, name: str) -> tuple[tuple[float, float, float], ...]:
    raw = _plain(value)
    if len(raw) == 9 and all(isinstance(_plain(item), Real) for item in raw):
        flat = [float(_plain(item)) for item in raw]
        return tuple(tuple(flat[3 * row : 3 * row + 3]) for row in range(3))
    matrix = _matrix(raw, rows=3, min_columns=3, name=name)
    if any(len(row) != 3 for row in matrix):
        raise ValueError(f"{name} must have shape (3, 3) or length 9")
    return tuple(tuple(row) for row in matrix)  # type: ignore[return-value]


def _matvec(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> tuple[float, float, float]:
    return tuple(sum(float(matrix[row][column]) * float(vector[column]) for column in range(3)) for row in range(3))  # type: ignore[return-value]


def _waypoints(
    translation: Sequence[Sequence[float]],
    start: Sequence[float],
    response: Sequence[Sequence[float]],
) -> tuple[tuple[float, float, float], ...]:
    current = tuple(float(item) for item in start)
    points = [current]
    for action in translation:
        displacement = _matvec(response, action)
        current = tuple(current[axis] + displacement[axis] for axis in range(3))
        points.append(current)  # type: ignore[arg-type]
    return tuple(points)


def _point_box_signed_distance(
    point: Sequence[float],
    center: Sequence[float],
    rotation: Sequence[Sequence[float]],
    half_size: Sequence[float],
) -> float:
    delta = tuple(float(point[axis]) - float(center[axis]) for axis in range(3))
    local = tuple(
        sum(float(rotation[row][column]) * delta[row] for row in range(3))
        for column in range(3)
    )
    extent_delta = tuple(abs(local[axis]) - float(half_size[axis]) for axis in range(3))
    outside = math.sqrt(sum(max(value, 0.0) ** 2 for value in extent_delta))
    inside = min(max(extent_delta), 0.0)
    return outside + inside


def _prepared_boxes(obstacle_boxes: Iterable[dict]) -> tuple[dict, ...]:
    boxes = []
    for index, box in enumerate(obstacle_boxes):
        center = _vector3(box["center_m"], name=f"obstacle_boxes[{index}].center_m")
        half_size = _vector3(box["half_size_m"], name=f"obstacle_boxes[{index}].half_size_m")
        if any(value <= 0.0 for value in half_size):
            raise ValueError("Obstacle half-sizes must be positive")
        rotation = _rotation_matrix(
            box["rotation_world"], name=f"obstacle_boxes[{index}].rotation_world"
        )
        boxes.append({"center": center, "half_size": half_size, "rotation": rotation})
    if not boxes:
        raise ValueError("At least one obstacle box is required")
    return tuple(boxes)


def _d_opt(
    points: Sequence[Sequence[float]],
    boxes: Sequence[dict],
    eef_radius_m: float,
    samples_per_segment: int,
) -> float:
    minimum = float("inf")
    denominator = samples_per_segment - 1
    for first, second in zip(points[:-1], points[1:]):
        for sample in range(samples_per_segment):
            alpha = sample / denominator
            point = tuple((1.0 - alpha) * first[axis] + alpha * second[axis] for axis in range(3))
            for box in boxes:
                distance = _point_box_signed_distance(
                    point, box["center"], box["rotation"], box["half_size"]
                ) - eef_radius_m
                minimum = min(minimum, distance)
    return minimum


def _flat(translation: Sequence[Sequence[float]]) -> tuple[float, ...]:
    return tuple(float(value) for row in translation for value in row[:3])


def _unflat(value: Sequence[float]) -> tuple[tuple[float, float, float], ...]:
    if len(value) != 15:
        raise ValueError("Endpoint-free five-action translation must contain 15 values")
    return tuple(tuple(float(value[3 * row + column]) for column in range(3)) for row in range(5))  # type: ignore[return-value]


def _clip(value: float, lower: float, upper: float) -> float:
    return min(max(float(value), float(lower)), float(upper))


def _template_translations(
    nominal: Sequence[Sequence[float]],
    lower: Sequence[float],
    upper: Sequence[float],
    amplitude: float,
) -> list[tuple[str, tuple[tuple[float, float, float], ...]]]:
    nominal_flat = _flat(nominal)
    templates: list[tuple[str, tuple[tuple[float, float, float], ...]]] = [
        ("nominal", _unflat(nominal_flat)),
        ("stationary", _unflat(tuple(_clip(0.0, lower[i], upper[i]) for i in range(15)))),
    ]
    for axis, label in enumerate(("x", "y", "z")):
        for sign, sign_label in ((-1.0, "negative"), (1.0, "positive")):
            shifted = list(nominal_flat)
            front_loaded = list(nominal_flat)
            for row in range(5):
                index = 3 * row + axis
                shifted[index] = _clip(shifted[index] + sign * amplitude, lower[index], upper[index])
                if row < 2:
                    front_loaded[index] = _clip(
                        front_loaded[index] + sign * amplitude, lower[index], upper[index]
                    )
            templates.append((f"axis_shift_{label}_{sign_label}", _unflat(shifted)))
            templates.append((f"front_detour_{label}_{sign_label}", _unflat(front_loaded)))
    return templates


def _compose_actions(
    nominal: Sequence[Sequence[float]], translation: Sequence[Sequence[float]]
) -> list[list[float]]:
    actions = [list(row) for row in nominal]
    for row in range(5):
        actions[row][:3] = [float(value) for value in translation[row]]
    return actions


def solve_endpoint_free_projection(
    nominal_prefix,
    *,
    start_eef_center_m,
    response_matrix,
    obstacle_boxes: Iterable[dict],
    eef_radius_m: float,
    progress_fn: ProgressFn,
    minimum_progress: float,
    optimizer_clearance_margin_m: float = 0.010,
    action_low=-1.0,
    action_high=1.0,
    population_size: int = 256,
    generations: int = 10,
    restarts: int = 4,
    elite_fraction: float = 0.10,
    initial_std_fraction: float = 0.25,
    std_floor_fraction: float = 0.02,
    template_amplitude: float = 0.15,
    max_candidates: int = 12,
    samples_per_segment: int = 26,
    h04_calibration_limit: float = 0.15,
    seed: int = 20260714,
) -> EndpointFreeSearchResult:
    """Search all five translation commands without fixing their endpoint.

    ``progress_fn`` receives the six H04-predicted EEF-center waypoints,
    including the branch point.  Clearance and progress remain independent hard
    filters.  Returned candidates are proxy nominations, not simulator-verified
    repairs.
    """

    nominal = _matrix(nominal_prefix, rows=5, min_columns=3, name="nominal_prefix")
    width = len(nominal[0])
    start = _vector3(start_eef_center_m, name="start_eef_center_m")
    response_rows = _matrix(response_matrix, rows=3, min_columns=3, name="response_matrix")
    if any(len(row) != 3 for row in response_rows):
        raise ValueError("response_matrix must have shape (3, 3)")
    response = tuple(tuple(row) for row in response_rows)
    boxes = _prepared_boxes(obstacle_boxes)

    if eef_radius_m <= 0.0 or not math.isfinite(eef_radius_m):
        raise ValueError("eef_radius_m must be positive and finite")
    if not math.isfinite(optimizer_clearance_margin_m) or not math.isfinite(minimum_progress):
        raise ValueError("Clearance and progress thresholds must be finite")
    if population_size < 4 or generations < 1 or restarts < 1:
        raise ValueError("CEM requires population_size >= 4, generations >= 1, and restarts >= 1")
    if not 0.0 < elite_fraction <= 1.0:
        raise ValueError("elite_fraction must be in (0, 1]")
    if initial_std_fraction <= 0.0 or std_floor_fraction < 0.0:
        raise ValueError("CEM spread parameters are invalid")
    if template_amplitude <= 0.0 or max_candidates < 1 or samples_per_segment < 2:
        raise ValueError("Template, shortlist, and segment-sampling parameters must be positive")
    if h04_calibration_limit <= 0.0:
        raise ValueError("h04_calibration_limit must be positive")

    low_matrix = _translation_bound_matrix(
        action_low, rows=5, action_width=width, name="action_low"
    )
    high_matrix = _translation_bound_matrix(
        action_high, rows=5, action_width=width, name="action_high"
    )
    lower = tuple(low_matrix[row][column] for row in range(5) for column in range(3))
    upper = tuple(high_matrix[row][column] for row in range(5) for column in range(3))
    if any(not lo < hi for lo, hi in zip(lower, upper)):
        raise ValueError("Every translational action lower bound must be below its upper bound")

    nominal_translation = tuple(tuple(row[:3]) for row in nominal)
    nominal_flat = _flat(nominal_translation)
    if any(value < lo or value > hi for value, lo, hi in zip(nominal_flat, lower, upper)):
        raise ValueError("Nominal translation lies outside the declared action bounds")
    cache: dict[tuple[float, ...], _Evaluated] = {}

    def evaluate(flat_value: Sequence[float], source: str) -> _Evaluated:
        key = tuple(float(item) for item in flat_value)
        if key in cache:
            return cache[key]
        translation = _unflat(key)
        points = _waypoints(translation, start, response)
        d_opt_m = float(_d_opt(points, boxes, float(eef_radius_m), samples_per_segment))
        progress_opt = float(progress_fn(points))
        if not math.isfinite(d_opt_m) or not math.isfinite(progress_opt):
            raise ValueError("D_opt and progress_fn must return finite values")
        objective = 0.5 * sum((key[index] - nominal_flat[index]) ** 2 for index in range(15))
        result = _Evaluated(source, translation, d_opt_m, progress_opt, objective)
        cache[key] = result
        return result

    clearance_scale = max(abs(float(optimizer_clearance_margin_m)), 0.005)
    progress_scale = max(abs(float(minimum_progress)), 0.01)

    def passes(item: _Evaluated) -> bool:
        return bool(
            item.d_opt_m >= optimizer_clearance_margin_m
            and item.progress_opt >= minimum_progress
        )

    def rank(item: _Evaluated) -> tuple:
        clearance_violation = max(0.0, optimizer_clearance_margin_m - item.d_opt_m)
        progress_violation = max(0.0, minimum_progress - item.progress_opt)
        if clearance_violation == 0.0 and progress_violation == 0.0:
            return (0, item.objective, -item.d_opt_m, -item.progress_opt, item.source)
        normalized = (
            clearance_violation / clearance_scale,
            progress_violation / progress_scale,
        )
        return (
            1,
            sum(value > 0.0 for value in normalized),
            max(normalized),
            sum(normalized),
            item.objective,
            item.source,
        )

    templates = _template_translations(nominal_translation, lower, upper, template_amplitude)
    control_evaluations = [evaluate(_flat(translation), source) for source, translation in templates]
    ordered_controls = sorted(control_evaluations, key=rank)

    generated_samples = 0
    elite_count = max(2, min(population_size, math.ceil(population_size * elite_fraction)))
    midpoint = tuple((lo + hi) / 2.0 for lo, hi in zip(lower, upper))
    for restart in range(restarts):
        rng = random.Random(int(seed) + 1_000_003 * restart)
        if restart == restarts - 1:
            mean = list(midpoint)
            standard_deviation = [(hi - lo) * 0.50 for lo, hi in zip(lower, upper)]
        else:
            start_item = ordered_controls[restart % len(ordered_controls)]
            mean = list(_flat(start_item.translation))
            standard_deviation = [
                (hi - lo) * initial_std_fraction for lo, hi in zip(lower, upper)
            ]
        floor = [(hi - lo) * std_floor_fraction for lo, hi in zip(lower, upper)]

        for generation in range(generations):
            population: list[_Evaluated] = []
            for member in range(population_size):
                if member == 0:
                    sample = tuple(_clip(mean[index], lower[index], upper[index]) for index in range(15))
                else:
                    sample = tuple(
                        _clip(
                            rng.gauss(mean[index], standard_deviation[index]),
                            lower[index],
                            upper[index],
                        )
                        for index in range(15)
                    )
                population.append(evaluate(sample, f"cem_r{restart}_g{generation}"))
                generated_samples += 1
            elites = sorted(population, key=rank)[:elite_count]
            elite_flat = [_flat(item.translation) for item in elites]
            mean = [sum(row[index] for row in elite_flat) / elite_count for index in range(15)]
            standard_deviation = [
                max(
                    floor[index],
                    math.sqrt(
                        sum((row[index] - mean[index]) ** 2 for row in elite_flat) / elite_count
                    ),
                )
                for index in range(15)
            ]

    def public(item: _Evaluated) -> EndpointFreeCandidate:
        return EndpointFreeCandidate(
            source=item.source,
            actions=_compose_actions(nominal, item.translation),
            d_opt_m=item.d_opt_m,
            progress_opt=item.progress_opt,
            clearance_pass=item.d_opt_m >= optimizer_clearance_margin_m,
            progress_pass=item.progress_opt >= minimum_progress,
            objective=item.objective,
            within_h04_calibration_domain=all(
                abs(value) <= h04_calibration_limit + 1e-12 for value in _flat(item.translation)
            ),
        )

    all_evaluated = list(cache.values())
    best_attempt = min(all_evaluated, key=rank)
    maximum_clearance_attempt = max(
        all_evaluated,
        key=lambda item: (item.d_opt_m, item.progress_opt, -item.objective, item.source),
    )
    maximum_progress_attempt = max(
        all_evaluated,
        key=lambda item: (item.progress_opt, item.d_opt_m, -item.objective, item.source),
    )
    feasible = sorted((item for item in all_evaluated if passes(item)), key=rank)
    shortlisted = [public(item) for item in feasible[:max_candidates]]
    outcome = "candidate_found" if shortlisted else "not_found_within_budget"
    if shortlisted:
        message = (
            "D_opt nominated endpoint-free candidate actions; independent D_sim, contact, "
            "and task-progress verification is still required."
        )
    else:
        message = (
            "No candidate met both proxy constraints within the fixed search budget; "
            "this bounded result is not a geometric certificate."
        )
    return EndpointFreeSearchResult(
        outcome=outcome,
        candidates=shortlisted,
        best_attempt=public(best_attempt),
        maximum_clearance_attempt=public(maximum_clearance_attempt),
        maximum_progress_attempt=public(maximum_progress_attempt),
        controls=[public(item) for item in control_evaluations],
        evaluations=len(cache),
        evaluation_budget=len(templates) + restarts * generations * population_size,
        generated_samples=generated_samples,
        seed=int(seed),
        optimizer_clearance_margin_m=float(optimizer_clearance_margin_m),
        minimum_progress=float(minimum_progress),
        message=message,
    )


__all__ = [
    "EndpointFreeCandidate",
    "EndpointFreeSearchResult",
    "ProgressFn",
    "solve_endpoint_free_projection",
]
