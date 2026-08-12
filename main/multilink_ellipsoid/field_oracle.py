"""Deterministic direction sets for the archived E05 field-oracle gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .repulsive_force import normalized_direction, softmin_weights
from .shadow import _numpy


FIELD_ORACLE_SCHEMA = "vlsa_distal_field_mixture_oracle_e05.v1"


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def load_field_oracle_config(path: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    value = json.loads(raw)
    required = {
        "case_ids",
        "claim_scope",
        "comparison",
        "finite_difference",
        "nominal",
        "protected_geometry",
        "protocol_id",
        "schema_version",
        "search",
        "success_definition",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("field-oracle config keys differ")
    if value["schema_version"] != FIELD_ORACLE_SCHEMA:
        raise ValueError("field-oracle schema differs")
    if value["case_ids"] != ["vlsa-t1-goal-ii-t0-e05"]:
        raise ValueError("field-oracle case differs")
    if value["nominal"]["action_steps"] != [185, 186]:
        raise ValueError("field-oracle action steps differ")
    search = value["search"]
    if search["iterations"] != 4 or search["maximum_total_correction_l2"] != 1.0:
        raise ValueError("field-oracle search horizon differs")
    if search["step_sizes"] != [0.05, 0.1, 0.15, 0.25]:
        raise ValueError("field-oracle step sizes differ")
    if search["normal_tangent_ratios"] != [0.0, 0.25, 0.5, 1.0, 2.0]:
        raise ValueError("field-oracle tangent ratios differ")
    output = json.loads(_canonical(value).decode("utf-8"))
    output["config_file_sha256"] = hashlib.sha256(raw).hexdigest()
    output["config_payload_sha256"] = hashlib.sha256(_canonical(value)).hexdigest()
    return output


def _unique_directions(values: list[Any]) -> Any:
    np = _numpy()
    output = []
    seen = set()
    for value in values:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
        norm = float(np.linalg.norm(array))
        if array.shape != (6,) or not np.all(np.isfinite(array)) or norm <= 1.0e-12:
            continue
        unit = array / norm
        key = tuple(np.round(unit, 12).tolist())
        if key not in seen:
            seen.add(key)
            output.append(unit)
    if not output:
        raise ValueError("field-oracle direction set is empty")
    return np.asarray(output, dtype=np.float64)


def unrestricted_directions(count: int, seed: int, preferred: list[Any]) -> Any:
    """Return deterministic arbitrary directions in the full two-action XYZ space."""

    np = _numpy()
    if count < 12 or count % 2:
        raise ValueError("unrestricted direction count must be even and at least 12")
    values = list(preferred)
    for index in range(6):
        axis = np.zeros(6, dtype=np.float64)
        axis[index] = 1.0
        values.extend([axis, -axis])
    rng = np.random.RandomState(int(seed))
    while len(values) < count:
        direction = rng.normal(size=6)
        direction /= np.linalg.norm(direction)
        values.extend([direction, -direction])
    return _unique_directions(values[:count])


def normal_mixture_directions(
    rows: Any,
    margins: Any,
    *,
    count: int,
    seed: int,
    temperature_m: float = 0.002,
) -> Any:
    """Pull nonnegative mixtures of seven physical clearance normals to actions."""

    np = _numpy()
    matrix = np.asarray(rows, dtype=np.float64)
    values = np.asarray(margins, dtype=np.float64)
    if matrix.shape != (7, 6) or values.shape != (7,):
        raise ValueError("normal-mixture basis shape differs")
    directions = [matrix.T.dot(softmin_weights(values, temperature_m))]
    directions.extend(matrix[index] for index in range(7))
    rng = np.random.RandomState(int(seed))
    while len(directions) < int(count):
        weights = rng.exponential(scale=1.0, size=7)
        weights /= np.sum(weights)
        directions.append(matrix.T.dot(weights))
    return _unique_directions(directions[: int(count)])


def task_tangent_direction(
    safety_rows: Any,
    margins: Any,
    task_gradient: Any,
    *,
    active_band_m: float,
) -> tuple[Any, list[int]]:
    """Project task progress into the nullspace of currently active safety rows."""

    np = _numpy()
    rows = np.asarray(safety_rows, dtype=np.float64)
    values = np.asarray(margins, dtype=np.float64)
    task = np.asarray(task_gradient, dtype=np.float64)
    if rows.shape != (7, 6) or values.shape != (7,) or task.shape != (6,):
        raise ValueError("task-tangent inputs differ")
    active = np.flatnonzero(values <= float(np.min(values)) + float(active_band_m))
    active_rows = rows[active]
    projection = task.copy()
    if active_rows.size:
        projection -= active_rows.T.dot(
            np.linalg.pinv(active_rows.dot(active_rows.T), rcond=1.0e-10)
        ).dot(active_rows).dot(task)
    return normalized_direction(projection), active.astype(int).tolist()


def normal_plus_tangent_directions(
    normal_directions: Any,
    tangent_direction: Any,
    ratios: list[float],
) -> Any:
    np = _numpy()
    normals = np.asarray(normal_directions, dtype=np.float64)
    tangent = np.asarray(tangent_direction, dtype=np.float64)
    if normals.ndim != 2 or normals.shape[1] != 6 or tangent.shape != (6,):
        raise ValueError("normal-plus-tangent shapes differ")
    values = [normal + float(ratio) * tangent for normal in normals for ratio in ratios]
    return _unique_directions(values)
