"""Pure direction construction utilities for the endpoint-free R02 pilot.

The functions in this module deliberately stop before policy inference or
simulator execution.  They turn a completed R01 artifact into paired direction
inputs while preserving the registered five-action translation scope.  The
implementation uses only the Python standard library so malformed evidence and
direction semantics can be tested by the dependency-free local gate.

``D_opt`` remains a nomination/diagnostic model.  A direction returned here is
not evidence of simulator safety and must still be evaluated by paired R02
``D_sim`` rollouts.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
import random
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple


EXECUTED_ACTIONS = 5
TRANSLATION_DIMS = 3
DEFAULT_MODEL_HORIZON = 10
DEFAULT_MODEL_ACTION_DIM = 32
DEFAULT_SAMPLES_PER_SEGMENT = 26
SCENE_MOTION_LIMIT_M = 1.0e-3


class R02DirectionError(ValueError):
    """Raised when evidence cannot define a registered R02 direction."""


@dataclass(frozen=True)
class R01ChangedWitness:
    """A changed, simulator-verified p_min witness recomputed from raw attempts."""

    case_id: str
    search: str
    raw_attempt_index: int
    candidate_index: int
    source: str
    nominal_prefix: Tuple[Tuple[float, ...], ...]
    witness_prefix: Tuple[Tuple[float, ...], ...]
    delta_star_physical: Tuple[Tuple[float, ...], ...]
    physical_l2_norm: float
    random_seed: int
    action_low: float
    action_high: float
    model_action_horizon: int
    model_action_dimension: int


@dataclass(frozen=True)
class AnalyticDOptDirection:
    """Model-space clearance-ascent correction and its exact nomination trace."""

    direction_model: Tuple[Tuple[float, ...], ...]
    d_opt_m: float
    selected_segment_index: int
    selected_sample_index: int
    selected_alpha: float
    selected_box_index: int
    selected_box_name: Optional[str]
    selected_point_world_m: Tuple[float, float, float]
    box_signed_distance_m: float
    local_sdf_gradient: Tuple[float, float, float]
    world_sdf_gradient: Tuple[float, float, float]
    raw_physical_gradient: Tuple[Tuple[float, float, float], ...]
    raw_model_gradient: Tuple[Tuple[float, float, float], ...]
    raw_model_gradient_l2: float
    model_l2_budget: float

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-ready diagnostics for an R02 artifact."""

        return {
            "direction_model": [list(row) for row in self.direction_model],
            "d_opt_m": self.d_opt_m,
            "selected_pair": {
                "segment_index": self.selected_segment_index,
                "sample_index": self.selected_sample_index,
                "alpha": self.selected_alpha,
                "box_index": self.selected_box_index,
                "box_name": self.selected_box_name,
                "point_world_m": list(self.selected_point_world_m),
                "box_signed_distance_m": self.box_signed_distance_m,
            },
            "local_sdf_gradient": list(self.local_sdf_gradient),
            "world_sdf_gradient": list(self.world_sdf_gradient),
            "raw_physical_gradient": [list(row) for row in self.raw_physical_gradient],
            "raw_model_gradient": [list(row) for row in self.raw_model_gradient],
            "raw_model_gradient_l2": self.raw_model_gradient_l2,
            "model_l2_budget": self.model_l2_budget,
        }


def _plain(value: Any) -> Any:
    return value.tolist() if hasattr(value, "tolist") else value


def _finite(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise R02DirectionError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise R02DirectionError(f"{name} must be a finite number") from error
    if not math.isfinite(result):
        raise R02DirectionError(f"{name} must be a finite number")
    return result


def _positive(value: Any, *, name: str) -> float:
    result = _finite(value, name=name)
    if result <= 0.0:
        raise R02DirectionError(f"{name} must be positive")
    return result


def _integer(value: Any, *, name: str, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise R02DirectionError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _matrix(
    value: Any,
    *,
    name: str,
    rows: Optional[int] = None,
    min_columns: int = 1,
) -> Tuple[Tuple[float, ...], ...]:
    raw = _plain(value)
    if isinstance(raw, (str, bytes)):
        raise R02DirectionError(f"{name} must be a rectangular numeric matrix")
    try:
        converted = tuple(
            tuple(_finite(item, name=f"{name}[{row_index}]") for item in _plain(row))
            for row_index, row in enumerate(raw)
        )
    except TypeError as error:
        raise R02DirectionError(f"{name} must be a rectangular numeric matrix") from error
    if rows is not None and len(converted) != rows:
        raise R02DirectionError(f"{name} must have {rows} rows")
    if not converted or any(len(row) < min_columns for row in converted):
        raise R02DirectionError(f"{name} must have at least {min_columns} columns")
    width = len(converted[0])
    if any(len(row) != width for row in converted):
        raise R02DirectionError(f"{name} must be rectangular")
    return converted


def _vector3(value: Any, *, name: str) -> Tuple[float, float, float]:
    matrix = _matrix([_plain(value)], name=name, rows=1, min_columns=3)
    if len(matrix[0]) != 3:
        raise R02DirectionError(f"{name} must have length 3")
    return matrix[0]  # type: ignore[return-value]


def _rotation(value: Any, *, name: str) -> Tuple[Tuple[float, float, float], ...]:
    raw = _plain(value)
    try:
        items = list(raw)
    except TypeError as error:
        raise R02DirectionError(f"{name} must have shape (3, 3) or length 9") from error
    if len(items) == 9 and all(isinstance(_plain(item), Real) for item in items):
        flat = tuple(_finite(item, name=name) for item in items)
        return tuple(tuple(flat[3 * row : 3 * row + 3]) for row in range(3))  # type: ignore[return-value]
    matrix = _matrix(items, name=name, rows=3, min_columns=3)
    if any(len(row) != 3 for row in matrix):
        raise R02DirectionError(f"{name} must have shape (3, 3) or length 9")
    return matrix  # type: ignore[return-value]


def _bounds(low: Any, high: Any) -> Tuple[float, float]:
    lower = _finite(low, name="translation action lower bound")
    upper = _finite(high, name="translation action upper bound")
    if not lower < upper:
        raise R02DirectionError("translation action bounds must satisfy low < high")
    return lower, upper


def _l2(matrix: Sequence[Sequence[float]]) -> float:
    return math.sqrt(sum(float(value) ** 2 for row in matrix for value in row))


def _zero_direction(rows: int, columns: int) -> list[list[float]]:
    return [[0.0 for _ in range(columns)] for _ in range(rows)]


def _freeze(matrix: Sequence[Sequence[float]]) -> Tuple[Tuple[float, ...], ...]:
    return tuple(tuple(float(value) for value in row) for row in matrix)


def _require_controlled_only(direction: Sequence[Sequence[float]], *, name: str) -> None:
    for row_index, row in enumerate(direction):
        for column_index, value in enumerate(row):
            controlled = row_index < EXECUTED_ACTIONS and column_index < TRANSLATION_DIMS
            if not controlled and float(value) != 0.0:
                raise R02DirectionError(
                    f"{name} must be zero outside the first five translation commands"
                )


def _attempt_claims_changed_p_min(attempt: Mapping[str, Any]) -> bool:
    required_true = (
        "simulator_safety_pass",
        "scene_stationary",
        "action_changed_from_nominal",
        "nontranslation_preserved",
        "direct_replay_exact",
        "verified_at_p_min",
        "changed_witness_at_p_min",
    )
    return all(attempt.get(field) is True for field in required_true)


def select_r01_changed_p_min_witness(artifact: Mapping[str, Any]) -> R01ChangedWitness:
    """Recompute the first valid changed p_min witness from raw R01 attempts.

    The summary ``selected`` pointer is intentionally ignored.  Search order is
    fixed to ``p_min`` then ``p_zero`` and raw attempt order, matching the R01
    pooling rule.  Redundant top-level claims are checked for consistency, but
    never substitute for a raw verified attempt.
    """

    if not isinstance(artifact, Mapping):
        raise R02DirectionError("R01 artifact must be an object")
    if artifact.get("schema_version") != "1.0" or artifact.get("gate") != "R01":
        raise R02DirectionError("artifact must be an R01 schema_version 1.0 result")
    if artifact.get("status") != "verified_safe_progress":
        raise R02DirectionError("R01 artifact does not have verified_safe_progress status")

    case_id = artifact.get("case_id")
    if not isinstance(case_id, str) or not case_id:
        raise R02DirectionError("R01 artifact case_id must be a non-empty string")

    provenance = artifact.get("provenance")
    nominal_record = artifact.get("nominal")
    calibration = artifact.get("calibration")
    verification = artifact.get("verification")
    outcome = artifact.get("outcome")
    if not isinstance(provenance, Mapping):
        raise R02DirectionError("R01 artifact provenance must be an object")
    if not isinstance(nominal_record, Mapping):
        raise R02DirectionError("R01 artifact nominal must be an object")
    if not isinstance(calibration, Mapping):
        raise R02DirectionError("R01 artifact calibration must be an object")
    if not isinstance(verification, Mapping):
        raise R02DirectionError("R01 artifact verification must be an object")
    if not isinstance(outcome, Mapping):
        raise R02DirectionError("R01 artifact outcome must be an object")
    if nominal_record.get("collision_reproduced") is not True:
        raise R02DirectionError("R01 nominal collision must reproduce before R02")
    if outcome.get("p_min_verified") is not True:
        raise R02DirectionError("R01 outcome does not verify a changed p_min witness")
    _positive(calibration.get("p_min_m"), name="calibration.p_min_m")

    executed = _integer(
        provenance.get("executed_action_horizon"),
        name="provenance.executed_action_horizon",
        minimum=1,
    )
    model_horizon = _integer(
        provenance.get("model_action_horizon"),
        name="provenance.model_action_horizon",
        minimum=EXECUTED_ACTIONS,
    )
    model_dimension = _integer(
        provenance.get("model_action_dimension"),
        name="provenance.model_action_dimension",
        minimum=TRANSLATION_DIMS,
    )
    if executed != EXECUTED_ACTIONS:
        raise R02DirectionError("R02 requires exactly five executed actions")
    seed = _integer(
        provenance.get("random_control_seed"),
        name="provenance.random_control_seed",
    )
    registered_bounds = provenance.get("translation_action_bounds")
    if not isinstance(registered_bounds, Sequence) or isinstance(
        registered_bounds, (str, bytes)
    ) or len(registered_bounds) != 2:
        raise R02DirectionError("provenance.translation_action_bounds must be [low, high]")
    action_low, action_high = _bounds(registered_bounds[0], registered_bounds[1])

    nominal = _matrix(
        nominal_record.get("actions"),
        name="nominal.actions",
        rows=EXECUTED_ACTIONS,
        min_columns=TRANSLATION_DIMS,
    )
    nominal_width = len(nominal[0])
    for row in nominal:
        for value in row[:TRANSLATION_DIMS]:
            if value < action_low or value > action_high:
                raise R02DirectionError("nominal translation lies outside registered bounds")

    for search_name in ("p_min", "p_zero"):
        search_verification = verification.get(search_name)
        if not isinstance(search_verification, Mapping):
            raise R02DirectionError(f"verification.{search_name} must be an object")
        attempts = search_verification.get("attempts")
        if not isinstance(attempts, list):
            raise R02DirectionError(f"verification.{search_name}.attempts must be a list")
        for raw_attempt_index, attempt in enumerate(attempts):
            if not isinstance(attempt, Mapping):
                raise R02DirectionError(
                    f"verification.{search_name}.attempts[{raw_attempt_index}] must be an object"
                )
            if not _attempt_claims_changed_p_min(attempt):
                continue
            witness_actions = _matrix(
                attempt.get("actions"),
                name=f"verification.{search_name}.attempts[{raw_attempt_index}].actions",
                rows=EXECUTED_ACTIONS,
                min_columns=TRANSLATION_DIMS,
            )
            if len(witness_actions[0]) != nominal_width:
                raise R02DirectionError("witness and nominal action widths do not match")
            for row_index in range(EXECUTED_ACTIONS):
                if witness_actions[row_index][TRANSLATION_DIMS:] != nominal[row_index][
                    TRANSLATION_DIMS:
                ]:
                    raise R02DirectionError(
                        "raw p_min witness changes a nontranslation action channel"
                    )
                for value in witness_actions[row_index][:TRANSLATION_DIMS]:
                    if value < action_low or value > action_high:
                        raise R02DirectionError(
                            "raw p_min witness translation lies outside registered bounds"
                        )

            delta = _zero_direction(model_horizon, model_dimension)
            for row_index in range(EXECUTED_ACTIONS):
                for column_index in range(TRANSLATION_DIMS):
                    delta[row_index][column_index] = (
                        witness_actions[row_index][column_index]
                        - nominal[row_index][column_index]
                    )
            frozen_delta = _freeze(delta)
            physical_l2_norm = _l2(frozen_delta)
            if physical_l2_norm <= 0.0:
                raise R02DirectionError(
                    "raw attempt claims a changed witness but Delta_star has zero norm"
                )
            candidate_index = _integer(
                attempt.get("candidate_index"),
                name="raw witness candidate_index",
            )
            source = attempt.get("source")
            if not isinstance(source, str) or not source:
                raise R02DirectionError("raw witness source must be a non-empty string")
            return R01ChangedWitness(
                case_id=case_id,
                search=search_name,
                raw_attempt_index=raw_attempt_index,
                candidate_index=candidate_index,
                source=source,
                nominal_prefix=nominal,
                witness_prefix=witness_actions,
                delta_star_physical=frozen_delta,
                physical_l2_norm=physical_l2_norm,
                random_seed=seed,
                action_low=action_low,
                action_high=action_high,
                model_action_horizon=model_horizon,
                model_action_dimension=model_dimension,
            )

    raise R02DirectionError(
        "R01 outcome claims p_min verification but raw attempts contain no valid changed witness"
    )


def _controlled_action_scale(action_scale: Any) -> Tuple[Tuple[float, float, float], ...]:
    """Return a positive five-by-three physical-per-model scale matrix."""

    raw = _plain(action_scale)
    if isinstance(raw, (str, bytes)):
        raise R02DirectionError("action_scale must contain exactly 3 or 15 values")
    try:
        items = list(raw)
    except TypeError as error:
        raise R02DirectionError(
            "action_scale must contain exactly 3 or 15 values"
        ) from error
    if len(items) not in (TRANSLATION_DIMS, EXECUTED_ACTIONS * TRANSLATION_DIMS):
        raise R02DirectionError("action_scale must contain exactly 3 or 15 values")
    values = tuple(_positive(item, name="action_scale") for item in items)
    if len(values) == TRANSLATION_DIMS:
        return tuple(values for _ in range(EXECUTED_ACTIONS))  # type: ignore[return-value]
    return tuple(
        tuple(values[3 * row + axis] for axis in range(TRANSLATION_DIMS))
        for row in range(EXECUTED_ACTIONS)
    )  # type: ignore[return-value]


def delta_star_model_from_witness(
    witness: R01ChangedWitness,
    *,
    action_scale: Any,
) -> Tuple[Tuple[float, ...], ...]:
    """Recompute ``Delta_star`` in model coordinates with scale only.

    R01 direct-replay actions are physical, unnormalized controller actions.
    For a displacement, ``delta_model = delta_physical / action_scale``; no
    normalization mean is involved.
    """

    if not isinstance(witness, R01ChangedWitness):
        raise R02DirectionError("witness must be an R01ChangedWitness")
    scales = _controlled_action_scale(action_scale)
    result = _zero_direction(
        witness.model_action_horizon, witness.model_action_dimension
    )
    for row in range(EXECUTED_ACTIONS):
        for axis in range(TRANSLATION_DIMS):
            result[row][axis] = witness.delta_star_physical[row][axis] / scales[row][axis]
    frozen = _freeze(result)
    if _l2(frozen) <= 0.0:
        raise R02DirectionError("Delta_star_model has zero norm")
    return frozen


def deterministic_equal_l2_random_direction(
    delta_star_model: Any,
    *,
    base_physical_prefix: Any,
    action_scale: Any,
    seed: int,
    action_low: Any,
    action_high: Any,
    max_draws: int = 4096,
) -> Tuple[Tuple[float, ...], ...]:
    """Return a seeded endpoint-free random direction with equal L2 budget.

    A physical target is sampled uniformly inside the registered 15-dimensional
    action box.  Its offset from the physical predicted-clean base is converted
    to model coordinates with scale only.  When that model-space distance is at
    least ``||Delta_star_model||``, it is scaled down to the exact model-space
    budget.  No normalization mean, clipping, or intermediate bounds rejection
    is applied.
    """

    direction_template = _matrix(
        delta_star_model,
        name="delta_star_model",
        rows=None,
        min_columns=TRANSLATION_DIMS,
    )
    if len(direction_template) < EXECUTED_ACTIONS:
        raise R02DirectionError("delta_star_model must contain at least five action rows")
    _require_controlled_only(direction_template, name="delta_star_model")
    budget = _l2(direction_template)
    if budget <= 0.0:
        raise R02DirectionError(
            "equal-L2 random direction requires nonzero Delta_star_model"
        )
    base_physical = _matrix(
        base_physical_prefix,
        name="base_physical_prefix",
        rows=EXECUTED_ACTIONS,
        min_columns=TRANSLATION_DIMS,
    )
    scales = _controlled_action_scale(action_scale)
    lower, upper = _bounds(action_low, action_high)
    registered_seed = _integer(seed, name="registered random seed")
    draws = _integer(max_draws, name="max_draws", minimum=1)
    rng = random.Random(registered_seed)

    for _ in range(draws):
        target = [
            rng.uniform(lower, upper)
            for _ in range(EXECUTED_ACTIONS * TRANSLATION_DIMS)
        ]
        offset_model = []
        for index, value in enumerate(target):
            row_index, column_index = divmod(index, TRANSLATION_DIMS)
            offset_model.append(
                (value - base_physical[row_index][column_index])
                / scales[row_index][column_index]
            )
        target_distance = math.sqrt(sum(value * value for value in offset_model))
        if target_distance < budget or target_distance <= 0.0:
            continue
        flat = [budget * value / target_distance for value in offset_model]
        result = _zero_direction(len(direction_template), len(direction_template[0]))
        for index, value in enumerate(flat):
            row_index, column_index = divmod(index, TRANSLATION_DIMS)
            result[row_index][column_index] = value
        frozen = _freeze(result)
        if not math.isclose(_l2(frozen), budget, rel_tol=1.0e-12, abs_tol=1.0e-12):
            raise R02DirectionError("internal error: random direction did not preserve L2 norm")
        return frozen

    raise R02DirectionError(
        "no sampled feasible target supports the registered equal-L2 random budget"
    )


def random_direction_from_witness(
    witness: R01ChangedWitness,
    *,
    base_physical_prefix: Any,
    action_scale: Any,
    max_draws: int = 4096,
) -> Tuple[Tuple[float, ...], ...]:
    """Construct the midpoint random arm with R01 budget and registered seed."""

    if not isinstance(witness, R01ChangedWitness):
        raise R02DirectionError("witness must be an R01ChangedWitness")
    delta_star_model = delta_star_model_from_witness(
        witness, action_scale=action_scale
    )
    return deterministic_equal_l2_random_direction(
        delta_star_model,
        base_physical_prefix=base_physical_prefix,
        action_scale=action_scale,
        seed=witness.random_seed,
        action_low=witness.action_low,
        action_high=witness.action_high,
        max_draws=max_draws,
    )


def _matvec(
    matrix: Sequence[Sequence[float]], vector: Sequence[float]
) -> Tuple[float, float, float]:
    return tuple(
        sum(float(matrix[row][column]) * float(vector[column]) for column in range(3))
        for row in range(3)
    )  # type: ignore[return-value]


def _waypoints(
    nominal_prefix: Sequence[Sequence[float]],
    start_eef_center_m: Sequence[float],
    response_matrix: Sequence[Sequence[float]],
) -> Tuple[Tuple[float, float, float], ...]:
    current = tuple(float(value) for value in start_eef_center_m)
    points = [current]
    for row in nominal_prefix[:EXECUTED_ACTIONS]:
        displacement = _matvec(response_matrix, row[:TRANSLATION_DIMS])
        current = tuple(current[axis] + displacement[axis] for axis in range(3))
        points.append(current)  # type: ignore[arg-type]
    return tuple(points)


def _point_obb_signed_distance_and_gradient(
    point: Sequence[float],
    center: Sequence[float],
    rotation_world: Sequence[Sequence[float]],
    half_size: Sequence[float],
) -> Tuple[
    float,
    Tuple[float, float, float],
    Tuple[float, float, float],
]:
    world_delta = tuple(float(point[axis]) - float(center[axis]) for axis in range(3))
    local = tuple(
        sum(float(rotation_world[row][column]) * world_delta[row] for row in range(3))
        for column in range(3)
    )
    extent_delta = tuple(abs(local[axis]) - float(half_size[axis]) for axis in range(3))
    positive_extent = tuple(max(value, 0.0) for value in extent_delta)
    outside = math.sqrt(sum(value * value for value in positive_extent))
    if outside > 0.0:
        local_gradient = tuple(
            (1.0 if local[axis] >= 0.0 else -1.0)
            * positive_extent[axis]
            / outside
            for axis in range(3)
        )
        signed_distance = outside
    else:
        # Inside/on-boundary SDF is max(abs(local) - half_size).  Resolve an
        # argmax tie by the lowest axis, and resolve sign(0) toward +axis.
        maximum = max(extent_delta)
        active_axis = min(
            axis for axis, value in enumerate(extent_delta) if value == maximum
        )
        local_gradient_list = [0.0, 0.0, 0.0]
        local_gradient_list[active_axis] = (
            1.0 if local[active_axis] >= 0.0 else -1.0
        )
        local_gradient = tuple(local_gradient_list)
        signed_distance = maximum
    world_gradient = tuple(
        sum(
            float(rotation_world[row][column]) * local_gradient[column]
            for column in range(3)
        )
        for row in range(3)
    )
    return (
        float(signed_distance),
        local_gradient,  # type: ignore[return-value]
        world_gradient,  # type: ignore[return-value]
    )


def _prepared_boxes(obstacle_boxes: Iterable[Mapping[str, Any]]) -> Tuple[dict, ...]:
    boxes = []
    try:
        raw_boxes = list(obstacle_boxes)
    except TypeError as error:
        raise R02DirectionError("obstacle_boxes must be an iterable of objects") from error
    for index, box in enumerate(raw_boxes):
        if not isinstance(box, Mapping):
            raise R02DirectionError(f"obstacle_boxes[{index}] must be an object")
        try:
            center = _vector3(box["center_m"], name=f"obstacle_boxes[{index}].center_m")
            half_size = _vector3(
                box["half_size_m"], name=f"obstacle_boxes[{index}].half_size_m"
            )
            rotation = _rotation(
                box["rotation_world"], name=f"obstacle_boxes[{index}].rotation_world"
            )
        except KeyError as error:
            raise R02DirectionError(
                f"obstacle_boxes[{index}] is missing {error.args[0]!r}"
            ) from error
        if any(value <= 0.0 for value in half_size):
            raise R02DirectionError("obstacle half-sizes must be positive")
        name = box.get("name")
        if name is not None and (not isinstance(name, str) or not name):
            raise R02DirectionError(
                f"obstacle_boxes[{index}].name must be a non-empty string when present"
            )
        tie_identity = (0, name, index) if name is not None else (1, "", index)
        boxes.append(
            {
                "index": index,
                "name": name,
                "tie_identity": tie_identity,
                "center": center,
                "half_size": half_size,
                "rotation": rotation,
            }
        )
    if not boxes:
        raise R02DirectionError("at least one obstacle box is required")
    return tuple(boxes)


@dataclass(frozen=True)
class _DOptSelection:
    clearance_m: float
    segment_index: int
    sample_index: int
    alpha: float
    box_index: int
    box_name: Optional[str]
    point_world_m: Tuple[float, float, float]
    box_signed_distance_m: float
    local_sdf_gradient: Tuple[float, float, float]
    world_sdf_gradient: Tuple[float, float, float]


def _select_d_opt_pair(
    actions: Sequence[Sequence[float]],
    *,
    start: Sequence[float],
    response: Sequence[Sequence[float]],
    boxes: Sequence[Mapping[str, Any]],
    radius: float,
    samples: int,
) -> _DOptSelection:
    points = _waypoints(actions, start, response)
    selected = None
    selected_key = None
    denominator = samples - 1
    for segment_index, (first, second) in enumerate(zip(points[:-1], points[1:])):
        for sample_index in range(samples):
            alpha = sample_index / denominator
            point = tuple(
                (1.0 - alpha) * first[axis] + alpha * second[axis]
                for axis in range(3)
            )
            for box in boxes:
                signed_distance, local_gradient, world_gradient = (
                    _point_obb_signed_distance_and_gradient(
                        point, box["center"], box["rotation"], box["half_size"]
                    )
                )
                clearance = signed_distance - radius
                key = (
                    clearance,
                    segment_index,
                    sample_index,
                    box["tie_identity"],
                )
                if selected_key is None or key < selected_key:
                    selected_key = key
                    selected = _DOptSelection(
                        clearance_m=float(clearance),
                        segment_index=segment_index,
                        sample_index=sample_index,
                        alpha=float(alpha),
                        box_index=int(box["index"]),
                        box_name=box["name"],
                        point_world_m=point,  # type: ignore[arg-type]
                        box_signed_distance_m=float(signed_distance),
                        local_sdf_gradient=local_gradient,
                        world_sdf_gradient=world_gradient,
                    )
    if selected is None or not math.isfinite(selected.clearance_m):
        raise R02DirectionError("D_opt evaluation did not produce a finite clearance")
    return selected


def sphere_obb_d_opt(
    action_prefix: Any,
    *,
    start_eef_center_m: Any,
    response_matrix_m_per_action: Any,
    obstacle_boxes: Iterable[Mapping[str, Any]],
    eef_radius_m: Any,
    samples_per_segment: int = DEFAULT_SAMPLES_PER_SEGMENT,
) -> float:
    """Evaluate the registered swept EEF-sphere/oriented-box proxy clearance."""

    actions = _matrix(
        action_prefix,
        name="action_prefix",
        rows=EXECUTED_ACTIONS,
        min_columns=TRANSLATION_DIMS,
    )
    start = _vector3(start_eef_center_m, name="start_eef_center_m")
    response = _matrix(
        response_matrix_m_per_action,
        name="response_matrix_m_per_action",
        rows=3,
        min_columns=3,
    )
    if any(len(row) != 3 for row in response):
        raise R02DirectionError("response_matrix_m_per_action must have shape (3, 3)")
    boxes = _prepared_boxes(obstacle_boxes)
    radius = _positive(eef_radius_m, name="eef_radius_m")
    samples = _integer(samples_per_segment, name="samples_per_segment", minimum=2)
    return _select_d_opt_pair(
        actions,
        start=start,
        response=response,
        boxes=boxes,
        radius=radius,
        samples=samples,
    ).clearance_m


def analytic_d_opt_ascent_direction(
    base_physical_prefix: Any,
    *,
    start_eef_center_m: Any,
    response_matrix_m_per_action: Any,
    obstacle_boxes: Iterable[Mapping[str, Any]],
    eef_radius_m: Any,
    action_scale: Any,
    model_l2_budget: Any,
    samples_per_segment: int = DEFAULT_SAMPLES_PER_SEGMENT,
    model_action_horizon: int = DEFAULT_MODEL_HORIZON,
    model_action_dimension: int = DEFAULT_MODEL_ACTION_DIM,
) -> AnalyticDOptDirection:
    """Return the frozen exact sphere/OBB ``D_opt`` ascent correction.

    The base is the midpoint predicted-clean action in physical controller
    coordinates.  The globally minimum sampled sphere/OBB pair is selected by
    the deterministic key ``(distance, segment, sample, box name/index)``.
    Its exact local box-SDF subgradient is propagated through the cumulative
    response model and then through ``physical_delta = scale * model_delta``.
    The returned correction is normalized to ``model_l2_budget`` in model
    space.  Intermediate corrections are neither clipped nor bounds-rejected;
    the R02 runner must validate the final sampled 5x7 actions separately.
    """

    base_physical = _matrix(
        base_physical_prefix,
        name="base_physical_prefix",
        rows=EXECUTED_ACTIONS,
        min_columns=TRANSLATION_DIMS,
    )
    start = _vector3(start_eef_center_m, name="start_eef_center_m")
    response = _matrix(
        response_matrix_m_per_action,
        name="response_matrix_m_per_action",
        rows=3,
        min_columns=3,
    )
    if any(len(row) != 3 for row in response):
        raise R02DirectionError("response_matrix_m_per_action must have shape (3, 3)")
    boxes = _prepared_boxes(obstacle_boxes)
    radius = _positive(eef_radius_m, name="eef_radius_m")
    scales = _controlled_action_scale(action_scale)
    budget = _positive(model_l2_budget, name="model_l2_budget")
    samples = _integer(samples_per_segment, name="samples_per_segment", minimum=2)
    if samples != DEFAULT_SAMPLES_PER_SEGMENT:
        raise R02DirectionError(
            "frozen analytic D_opt direction requires 26 samples per segment"
        )
    horizon = _integer(
        model_action_horizon, name="model_action_horizon", minimum=EXECUTED_ACTIONS
    )
    dimension = _integer(
        model_action_dimension,
        name="model_action_dimension",
        minimum=TRANSLATION_DIMS,
    )

    selected = _select_d_opt_pair(
        base_physical,
        start=start,
        response=response,
        boxes=boxes,
        radius=radius,
        samples=samples,
    )
    response_transpose_gradient = tuple(
        sum(
            response[world_axis][action_axis]
            * selected.world_sdf_gradient[world_axis]
            for world_axis in range(3)
        )
        for action_axis in range(3)
    )
    raw_physical = []
    raw_model = []
    for action_index in range(EXECUTED_ACTIONS):
        if action_index < selected.segment_index:
            coefficient = 1.0
        elif action_index == selected.segment_index:
            coefficient = selected.alpha
        else:
            coefficient = 0.0
        physical_row = tuple(
            coefficient * response_transpose_gradient[axis]
            for axis in range(TRANSLATION_DIMS)
        )
        model_row = tuple(
            physical_row[axis] * scales[action_index][axis]
            for axis in range(TRANSLATION_DIMS)
        )
        raw_physical.append(physical_row)
        raw_model.append(model_row)

    gradient_norm = _l2(raw_model)
    if gradient_norm <= 0.0 or not math.isfinite(gradient_norm):
        raise R02DirectionError("analytic D_opt model-space gradient has zero norm")
    result = _zero_direction(horizon, dimension)
    for row in range(EXECUTED_ACTIONS):
        for axis in range(TRANSLATION_DIMS):
            result[row][axis] = budget * raw_model[row][axis] / gradient_norm
    frozen = _freeze(result)
    if not math.isclose(_l2(frozen), budget, rel_tol=1.0e-12, abs_tol=1.0e-12):
        raise R02DirectionError("internal error: analytic direction did not preserve L2 norm")
    return AnalyticDOptDirection(
        direction_model=frozen,
        d_opt_m=selected.clearance_m,
        selected_segment_index=selected.segment_index,
        selected_sample_index=selected.sample_index,
        selected_alpha=selected.alpha,
        selected_box_index=selected.box_index,
        selected_box_name=selected.box_name,
        selected_point_world_m=selected.point_world_m,
        box_signed_distance_m=selected.box_signed_distance_m,
        local_sdf_gradient=selected.local_sdf_gradient,
        world_sdf_gradient=selected.world_sdf_gradient,
        raw_physical_gradient=tuple(raw_physical),  # type: ignore[arg-type]
        raw_model_gradient=tuple(raw_model),  # type: ignore[arg-type]
        raw_model_gradient_l2=gradient_norm,
        model_l2_budget=budget,
    )


def scale_only_model_conversion(physical_delta: Any, scale: Any) -> Tuple[Tuple[float, ...], ...]:
    """Convert a physical displacement delta with division by scale only.

    ``scale`` may be a positive scalar, a per-column vector, or a matrix with
    the same shape as ``physical_delta``.  No centering/normalization mean is
    accepted or subtracted because a displacement is not an absolute action.
    """

    physical = _matrix(physical_delta, name="physical_delta", min_columns=1)
    rows = len(physical)
    columns = len(physical[0])
    raw_scale = _plain(scale)
    if isinstance(raw_scale, Real) and not isinstance(raw_scale, bool):
        scalar = _positive(raw_scale, name="scale")
        scales = tuple(tuple(scalar for _ in range(columns)) for _ in range(rows))
    else:
        try:
            scale_items = list(raw_scale)
        except TypeError as error:
            raise R02DirectionError(
                "scale must be a positive scalar, per-column vector, or matching matrix"
            ) from error
        if len(scale_items) == columns and all(
            isinstance(_plain(item), Real) and not isinstance(_plain(item), bool)
            for item in scale_items
        ):
            vector = tuple(_positive(item, name="scale") for item in scale_items)
            scales = tuple(vector for _ in range(rows))
        else:
            scales = _matrix(scale_items, name="scale", rows=rows, min_columns=columns)
            if any(len(row) != columns for row in scales):
                raise R02DirectionError("scale matrix must match physical_delta shape")
            if any(value <= 0.0 for row in scales for value in row):
                raise R02DirectionError("scale entries must be positive")
    return tuple(
        tuple(physical[row][column] / scales[row][column] for column in range(columns))
        for row in range(rows)
    )


# Explicit descriptive alias for callers that prefer unit-oriented naming.
physical_displacement_to_model_delta = scale_only_model_conversion


__all__ = [
    "AnalyticDOptDirection",
    "R01ChangedWitness",
    "R02DirectionError",
    "analytic_d_opt_ascent_direction",
    "delta_star_model_from_witness",
    "deterministic_equal_l2_random_direction",
    "physical_displacement_to_model_delta",
    "random_direction_from_witness",
    "scale_only_model_conversion",
    "select_r01_changed_p_min_witness",
    "sphere_obb_d_opt",
]
