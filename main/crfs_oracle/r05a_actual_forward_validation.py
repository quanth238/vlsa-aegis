"""Independent tensor validation for the AF-00A actual-forward canary.

This module deliberately contains no policy, optimizer-runner, publication, or
HPC code.  It consumes the sealed fixed-shape tensor archive and request
ledger, reconstructs the preregistered CEM arithmetic, and returns the only
scientific classification permitted by ADR-0050.  Stored GPU booleans are not
inputs to the decision.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = "1.0"
EXPECTED_BUDGET32 = np.float32(3.6398398876190186)
EXPECTED_DT32 = np.float32(-0.1)
GENERATIONS = 8
POPULATION_SIZE = 65
PAIR_COUNT = 32
ELITE_COUNT = 13
SEARCH_QUERIES = GENERATIONS * POPULATION_SIZE
POLICY_REQUESTS = 534
COMPACT_SHAPE = (5, 15)
METRIC_NAMES = ("xyz_max_abs", "xyz_rms", "full_max_abs", "full_rms")
FIDELITY_LIMITS64 = np.asarray((0.010, 0.005, 0.050, 0.015), dtype=np.float64)


class ActualForwardValidationError(ValueError):
    """The AF-00A raw archive is incomplete or scientifically inconsistent."""


def _spec(shape: tuple[int, ...], dtype: str) -> tuple[tuple[int, ...], np.dtype[Any]]:
    return shape, np.dtype(dtype)


# Flat names avoid ZIP path semantics.  Numeric width and byte order are part
# of the archive contract; no object, Unicode, or structured arrays are used.
ARRAY_SPECS = {
    "source_noise_f32": _spec((10, 32), "<f4"),
    "source_frozen_final_f32": _spec((10, 32), "<f4"),
    "source_delta_model_f32": _spec((10, 32), "<f4"),
    "source_target_normalized_f32": _spec((10, 32), "<f4"),
    "source_model_to_physical_scale_f32": _spec((10, 32), "<f4"),
    "source_target_physical_f64": _spec((10, 7), "<f8"),
    "source_budget_f32": _spec((), "<f4"),
    "source_dt_f32": _spec((), "<f4"),
    "source_radius_f32": _spec((), "<f4"),
    "reference_compiled_actions_f64": _spec((2, 10, 7), "<f8"),
    "reference_source_actions_f64": _spec((2, 10, 7), "<f8"),
    "reference_eager_actions_f64": _spec((2, 10, 7), "<f8"),
    "reference_eager_final_f32": _spec((2, 10, 32), "<f4"),
    "zero_requested_c_f32": _spec((2, 5, 15), "<f4"),
    "zero_requested_velocity_f32": _spec((2, 5, 15), "<f4"),
    "zero_applied_velocity_f32": _spec((2, 5, 15), "<f4"),
    "zero_executed_c_f32": _spec((2, 5, 15), "<f4"),
    "zero_final_normalized_f32": _spec((2, 10, 32), "<f4"),
    "zero_final_normalized_physical_f64": _spec((2, 10, 7), "<f8"),
    "zero_returned_actions_f64": _spec((2, 10, 7), "<f8"),
    "arm_a_requested_c_f32": _spec((2, 5, 15), "<f4"),
    "arm_a_requested_velocity_f32": _spec((2, 5, 15), "<f4"),
    "arm_a_applied_velocity_f32": _spec((2, 5, 15), "<f4"),
    "arm_a_executed_c_f32": _spec((2, 5, 15), "<f4"),
    "arm_a_final_normalized_f32": _spec((2, 10, 32), "<f4"),
    "arm_a_final_normalized_physical_f64": _spec((2, 10, 7), "<f8"),
    "arm_a_returned_actions_f64": _spec((2, 10, 7), "<f8"),
    "cem_raw_normals_f64": _spec((8, 32, 5, 15), "<f8"),
    "cem_mean_state_f64": _spec((9, 5, 15), "<f8"),
    "cem_sigma_state_f64": _spec((9, 5, 15), "<f8"),
    "cem_variance_after_f64": _spec((8, 5, 15), "<f8"),
    "cem_raw_proposal_f64": _spec((520, 5, 15), "<f8"),
    "cem_projected_c_f32": _spec((520, 5, 15), "<f4"),
    "cem_requested_velocity_f32": _spec((520, 5, 15), "<f4"),
    "cem_applied_velocity_f32": _spec((520, 5, 15), "<f4"),
    "cem_executed_c_f32": _spec((520, 5, 15), "<f4"),
    "cem_final_normalized_f32": _spec((520, 10, 32), "<f4"),
    "cem_final_normalized_physical_f64": _spec((520, 10, 7), "<f8"),
    "cem_returned_actions_f64": _spec((520, 10, 7), "<f8"),
    "cem_objective_f64": _spec((520,), "<f8"),
    "cem_fidelity_metrics_f64": _spec((520, 4), "<f8"),
    "cem_gate_pass_bool": _spec((520, 4), "|b1"),
    "cem_energy_f64": _spec((520,), "<f8"),
    "cem_generation_i16": _spec((520,), "<i2"),
    "cem_population_index_i16": _spec((520,), "<i2"),
    "cem_pair_index_i16": _spec((520,), "<i2"),
    "cem_sign_i8": _spec((520,), "|i1"),
    "cem_elite_query_index_i64": _spec((8, 13), "<i8"),
    "cem_query_elapsed_ns_u64": _spec((520,), "<u8"),
    "arm_b_requested_c_f32": _spec((2, 5, 15), "<f4"),
    "arm_b_requested_velocity_f32": _spec((2, 5, 15), "<f4"),
    "arm_b_applied_velocity_f32": _spec((2, 5, 15), "<f4"),
    "arm_b_executed_c_f32": _spec((2, 5, 15), "<f4"),
    "arm_b_final_normalized_f32": _spec((2, 10, 32), "<f4"),
    "arm_b_final_normalized_physical_f64": _spec((2, 10, 7), "<f8"),
    "arm_b_returned_actions_f64": _spec((2, 10, 7), "<f8"),
    "arm_c_requested_c_f32": _spec((2, 5, 15), "<f4"),
    "arm_c_requested_velocity_f32": _spec((2, 5, 15), "<f4"),
    "arm_c_applied_velocity_f32": _spec((2, 5, 15), "<f4"),
    "arm_c_executed_c_f32": _spec((2, 5, 15), "<f4"),
    "arm_c_final_normalized_f32": _spec((2, 10, 32), "<f4"),
    "arm_c_final_normalized_physical_f64": _spec((2, 10, 7), "<f8"),
    "arm_c_returned_actions_f64": _spec((2, 10, 7), "<f8"),
    "selected_pool_index_i64": _spec((), "<i8"),
}

SCHEDULE_GROUP_SIZES = {
    "zero": 2,
    "arm_a": 2,
    "cem": 520,
    "arm_b": 2,
    "arm_c": 2,
}
for _prefix, _count in SCHEDULE_GROUP_SIZES.items():
    ARRAY_SPECS.update(
        {
            f"{_prefix}_trace_step_index_i64": _spec((_count, 10), "<i8"),
            f"{_prefix}_trace_time_f32": _spec((_count, 10), "<f4"),
            f"{_prefix}_trace_active_bool": _spec((_count, 10), "|b1"),
            f"{_prefix}_trace_x_t_f32": _spec((_count, 10, 10, 32), "<f4"),
            f"{_prefix}_trace_v_base_f32": _spec((_count, 10, 10, 32), "<f4"),
            f"{_prefix}_trace_total_velocity_f32": _spec(
                (_count, 10, 10, 32), "<f4"
            ),
            f"{_prefix}_trace_x_next_f32": _spec((_count, 10, 10, 32), "<f4"),
            f"{_prefix}_trace_initial_noise_f32": _spec((_count, 10, 32), "<f4"),
        }
    )


def _bytes_equal(left: Any, right: Any) -> bool:
    first = np.ascontiguousarray(np.asarray(left))
    second = np.ascontiguousarray(np.asarray(right))
    return bool(
        first.shape == second.shape
        and first.dtype == second.dtype
        and first.tobytes() == second.tobytes()
    )


def _bytes_digest(value: Any) -> dict[str, Any]:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes())
    return {"dtype": str(array.dtype), "shape": list(array.shape), "sha256": digest.hexdigest()}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ActualForwardValidationError(message)


def _load_arrays(raw: Mapping[str, Any]) -> dict[str, np.ndarray]:
    keys = set(raw.keys())
    expected = set(ARRAY_SPECS)
    _require(keys == expected, f"AF-00A tensor names changed: missing={sorted(expected - keys)}, extra={sorted(keys - expected)}")
    arrays: dict[str, np.ndarray] = {}
    for name, (shape, dtype) in ARRAY_SPECS.items():
        value = np.asarray(raw[name])
        _require(value.shape == shape, f"{name} must have shape {shape}, got {value.shape}")
        _require(value.dtype == dtype, f"{name} must preserve {dtype}, got {value.dtype}")
        _require(not value.dtype.hasobject and value.dtype.fields is None, f"{name} must be a plain no-pickle array")
        if np.issubdtype(value.dtype, np.floating):
            _require(bool(np.isfinite(value).all()), f"{name} contains a nonfinite value")
        arrays[name] = np.ascontiguousarray(value)
    return arrays


def _project_blocks_float64(value: np.ndarray, radius64: np.float64) -> np.ndarray:
    source = np.asarray(value, dtype=np.float64)
    _require(source.shape == COMPACT_SHAPE, "CEM projection input must be 5x15")
    projected = np.empty_like(source)
    for block_index in range(5):
        block = source[block_index]
        norm = np.sqrt(np.sum(np.square(block), dtype=np.float64), dtype=np.float64)
        if norm == np.float64(0.0):
            projected[block_index] = np.zeros((15,), dtype=np.float64)
        elif norm <= radius64:
            projected[block_index] = block
        else:
            projected[block_index] = block * np.float64(radius64 / norm)
    return projected


def _allowed_after_ulps(value: np.float32, count: int = 8) -> np.float32:
    allowed = np.float32(value)
    for _ in range(count):
        allowed = np.nextafter(allowed, np.float32(np.inf), dtype=np.float32)
    return allowed


def _validate_constraints(executed: np.ndarray, radius32: np.float32, budget32: np.float32, *, name: str) -> None:
    value = np.asarray(executed, dtype=np.float32)
    norms = np.linalg.norm(value, axis=1).astype(np.float32)
    path = np.asarray(np.sum(norms, dtype=np.float32), dtype=np.float32)
    _require(bool(np.all(norms <= _allowed_after_ulps(radius32))), f"{name} exceeds the eight-ULP per-step cap")
    _require(bool(path <= _allowed_after_ulps(budget32)), f"{name} exceeds the eight-ULP path budget")


def _objective_metrics(actions: np.ndarray, target: np.ndarray) -> tuple[np.float64, np.ndarray, np.ndarray]:
    physical = np.asarray(actions, dtype=np.float64)
    wanted = np.asarray(target, dtype=np.float64)
    _require(physical.shape == (10, 7) and wanted.shape == (10, 7), "physical fidelity inputs must be 10x7")
    error = np.asarray(physical[:5, :7] - wanted[:5, :7], dtype=np.float64)
    weighted = np.concatenate(
        (
            np.asarray(error[:, :3] / np.float64(0.005), dtype=np.float64).reshape(-1),
            np.asarray(error[:, 3:7] / np.float64(0.015), dtype=np.float64).reshape(-1),
        )
    )
    objective = np.asarray(np.mean(np.square(weighted), dtype=np.float64), dtype=np.float64)
    xyz = error[:, :3]
    full = error
    metrics = np.asarray(
        (
            np.max(np.abs(xyz)),
            np.sqrt(np.mean(np.square(xyz), dtype=np.float64), dtype=np.float64),
            np.max(np.abs(full)),
            np.sqrt(np.mean(np.square(full), dtype=np.float64), dtype=np.float64),
        ),
        dtype=np.float64,
    )
    gates = np.asarray(metrics <= FIDELITY_LIMITS64, dtype=np.bool_)
    return objective, metrics, gates


def _energy(executed: np.ndarray) -> np.float64:
    value = np.asarray(executed, dtype=np.float64)
    return np.asarray(np.sum(np.square(value), dtype=np.float64), dtype=np.float64)


def _expected_ledger_row(ordinal: int) -> tuple[str, int, Any, Any, Any]:
    if ordinal == 0:
        return "compiled_frozen_pre", 0, None, None, None
    if ordinal == 1:
        return "eager_source_trace_pre", 0, None, None, None
    if ordinal == 2:
        return "eager_normalized_final_pre", 0, None, None, None
    if ordinal == 3:
        return "zero_schedule_pre", 0, None, None, None
    if ordinal in (4, 5):
        return "arm_a_equal_split", ordinal - 4, None, None, None
    if 6 <= ordinal < 526:
        query = ordinal - 6
        return "cem_search", query, query, query // 65, query % 65
    if ordinal in (526, 527):
        return "arm_b_replay", ordinal - 526, None, None, None
    if ordinal in (528, 529):
        return "arm_c_reverse_replay", ordinal - 528, None, None, None
    if ordinal == 530:
        return "zero_schedule_post", 0, None, None, None
    if ordinal == 531:
        return "eager_normalized_final_post", 0, None, None, None
    if ordinal == 532:
        return "eager_source_trace_post", 0, None, None, None
    if ordinal == 533:
        return "compiled_frozen_post", 0, None, None, None
    raise ActualForwardValidationError("request ordinal escaped the 534-call contract")


def validate_request_ledger(ledger: Sequence[Mapping[str, Any]]) -> None:
    _require(not isinstance(ledger, (str, bytes)), "request ledger must be a sequence of rows")
    _require(len(ledger) == POLICY_REQUESTS, f"request ledger must contain {POLICY_REQUESTS} rows")
    required = {"ordinal", "phase", "phase_index", "cem_query_index", "generation", "population_index"}
    for ordinal, row in enumerate(ledger):
        _require(isinstance(row, Mapping), f"request ledger row {ordinal} is not an object")
        _require(required <= set(row), f"request ledger row {ordinal} lacks ordinal metadata")
        phase, phase_index, query, generation, population = _expected_ledger_row(ordinal)
        expected = {
            "ordinal": ordinal,
            "phase": phase,
            "phase_index": phase_index,
            "cem_query_index": query,
            "generation": generation,
            "population_index": population,
        }
        for key, wanted in expected.items():
            observed = row.get(key)
            _require(observed == wanted and type(observed) is type(wanted), f"request ledger row {ordinal}.{key} changed")


def _validate_transport_group(
    arrays: Mapping[str, np.ndarray],
    prefix: str,
    expected_c: np.ndarray,
    *,
    radius32: np.float32,
    budget32: np.float32,
) -> None:
    requested_c_name = (
        "cem_projected_c_f32" if prefix == "cem" else f"{prefix}_requested_c_f32"
    )
    requested_c = arrays[requested_c_name]
    requested_u = arrays[f"{prefix}_requested_velocity_f32"]
    applied_u = arrays[f"{prefix}_applied_velocity_f32"]
    executed_c = arrays[f"{prefix}_executed_c_f32"]
    _require(_bytes_equal(requested_c, expected_c), f"{prefix} requested c changed")
    expected_u = np.asarray(requested_c / EXPECTED_DT32, dtype=np.float32)
    _require(_bytes_equal(requested_u, expected_u), f"{prefix} c-to-u transport changed")
    _require(_bytes_equal(applied_u, requested_u), f"{prefix} applied velocity differs from request")
    expected_executed = np.asarray(EXPECTED_DT32 * applied_u, dtype=np.float32)
    _require(_bytes_equal(executed_c, expected_executed), f"{prefix} executed increment differs from dt*u")
    for index in range(executed_c.shape[0]):
        _validate_constraints(executed_c[index], radius32, budget32, name=f"{prefix}[{index}]")


def _expected_times_float32() -> np.ndarray:
    values = np.empty((10,), dtype=np.float32)
    current = np.asarray(1.0, dtype=np.float32)
    for index in range(10):
        values[index] = current
        current = np.asarray(current + EXPECTED_DT32, dtype=np.float32)
    return values


def _expanded_control(compact: np.ndarray) -> np.ndarray:
    value = np.asarray(compact, dtype=np.float32)
    _require(value.ndim == 3 and value.shape[1:] == (5, 15), "compact applied control shape changed")
    expanded = np.zeros((value.shape[0], 10, 10, 32), dtype=np.float32)
    expanded[:, 5:10, :5, :3] = value.reshape(value.shape[0], 5, 5, 3)
    return expanded


def _ledger_action(arrays: Mapping[str, np.ndarray], ordinal: int) -> np.ndarray:
    if ordinal == 0:
        return arrays["reference_compiled_actions_f64"][0]
    if ordinal == 1:
        return arrays["reference_source_actions_f64"][0]
    if ordinal == 2:
        return arrays["reference_eager_actions_f64"][0]
    if ordinal == 3:
        return arrays["zero_returned_actions_f64"][0]
    if ordinal in (4, 5):
        return arrays["arm_a_returned_actions_f64"][ordinal - 4]
    if 6 <= ordinal < 526:
        return arrays["cem_returned_actions_f64"][ordinal - 6]
    if ordinal in (526, 527):
        return arrays["arm_b_returned_actions_f64"][ordinal - 526]
    if ordinal in (528, 529):
        return arrays["arm_c_returned_actions_f64"][ordinal - 528]
    if ordinal == 530:
        return arrays["zero_returned_actions_f64"][1]
    if ordinal == 531:
        return arrays["reference_eager_actions_f64"][1]
    if ordinal == 532:
        return arrays["reference_source_actions_f64"][1]
    if ordinal == 533:
        return arrays["reference_compiled_actions_f64"][1]
    raise ActualForwardValidationError("ledger action ordinal escaped the contract")


def _ledger_velocity(arrays: Mapping[str, np.ndarray], ordinal: int) -> np.ndarray | None:
    if ordinal == 3:
        return arrays["zero_requested_velocity_f32"][0]
    if ordinal in (4, 5):
        return arrays["arm_a_requested_velocity_f32"][ordinal - 4]
    if 6 <= ordinal < 526:
        return arrays["cem_requested_velocity_f32"][ordinal - 6]
    if ordinal in (526, 527):
        return arrays["arm_b_requested_velocity_f32"][ordinal - 526]
    if ordinal in (528, 529):
        return arrays["arm_c_requested_velocity_f32"][ordinal - 528]
    if ordinal == 530:
        return arrays["zero_requested_velocity_f32"][1]
    return None


def _full_schedule(compact_velocity: np.ndarray) -> np.ndarray:
    value = np.asarray(compact_velocity)
    _require(
        value.shape == COMPACT_SHAPE and value.dtype == np.dtype(np.float32),
        "ledger compact velocity changed shape or dtype",
    )
    schedule = np.zeros((10, 10, 32), dtype=np.float32)
    schedule[5:10, :5, :3] = value.reshape(5, 5, 3)
    return schedule


def _validate_ledger_artifact_bindings(
    arrays: Mapping[str, np.ndarray], ledger: Sequence[Mapping[str, Any]]
) -> None:
    for ordinal, row in enumerate(ledger):
        _require(
            row.get("actions") == _bytes_digest(_ledger_action(arrays, ordinal)),
            f"request ledger row {ordinal} action digest differs from the sealed reply",
        )
        velocity = _ledger_velocity(arrays, ordinal)
        expected_slot = ordinal - 3 if velocity is not None else None
        _require(
            row.get("trace_slot") == expected_slot
            and type(row.get("trace_slot")) is type(expected_slot),
            f"request ledger row {ordinal} recurrence slot changed",
        )
        if velocity is None:
            _require(
                "requested_schedule" not in row,
                f"request ledger row {ordinal} unexpectedly binds a schedule",
            )
        else:
            _require(
                row.get("requested_schedule") == _bytes_digest(_full_schedule(velocity)),
                f"request ledger row {ordinal} schedule digest differs from the sealed request",
            )


def _validate_recurrence_group(
    arrays: Mapping[str, np.ndarray], prefix: str, source_noise: np.ndarray
) -> None:
    count = SCHEDULE_GROUP_SIZES[prefix]
    step = arrays[f"{prefix}_trace_step_index_i64"]
    times = arrays[f"{prefix}_trace_time_f32"]
    active = arrays[f"{prefix}_trace_active_bool"]
    x_t = arrays[f"{prefix}_trace_x_t_f32"]
    v_base = arrays[f"{prefix}_trace_v_base_f32"]
    total = arrays[f"{prefix}_trace_total_velocity_f32"]
    x_next = arrays[f"{prefix}_trace_x_next_f32"]
    initial = arrays[f"{prefix}_trace_initial_noise_f32"]
    applied = arrays[f"{prefix}_applied_velocity_f32"]
    final = arrays[f"{prefix}_final_normalized_f32"]

    expected_step = np.broadcast_to(np.arange(10, dtype=np.int64), (count, 10)).copy()
    expected_time = np.broadcast_to(_expected_times_float32(), (count, 10)).copy()
    expected_active = np.broadcast_to(
        np.asarray([False] * 5 + [True] * 5, dtype=np.bool_), (count, 10)
    ).copy()
    expected_initial = np.broadcast_to(
        np.asarray(source_noise, dtype=np.float32), (count, 10, 32)
    ).copy()
    _require(_bytes_equal(step, expected_step), f"{prefix} trace step indices changed")
    _require(_bytes_equal(times, expected_time), f"{prefix} trace times changed")
    _require(_bytes_equal(active, expected_active), f"{prefix} trace active mask changed")
    _require(_bytes_equal(initial, expected_initial), f"{prefix} trace initial noise changed")

    control = _expanded_control(applied)
    expected_total = np.where(control == np.float32(0.0), v_base, v_base + control)
    expected_next = np.asarray(x_t + EXPECTED_DT32 * expected_total, dtype=np.float32)
    _require(_bytes_equal(total, expected_total), f"{prefix} trace total velocity violates zero-preserving addition")
    _require(_bytes_equal(x_next, expected_next), f"{prefix} trace x_next violates the float32 Euler recurrence")
    _require(_bytes_equal(x_t[:, 0], initial), f"{prefix} trace first state differs from paired noise")
    _require(_bytes_equal(x_t[:, 1:], x_next[:, :-1]), f"{prefix} trace states are disconnected")
    _require(_bytes_equal(final, x_next[:, -1]), f"{prefix} normalized final differs from the last recurrence state")


RECURRENCE_SUFFIXES = (
    "trace_step_index_i64",
    "trace_time_f32",
    "trace_active_bool",
    "trace_x_t_f32",
    "trace_v_base_f32",
    "trace_total_velocity_f32",
    "trace_x_next_f32",
    "trace_initial_noise_f32",
)


def _recurrence_row_bytes(
    arrays: Mapping[str, np.ndarray], prefix: str, index: int
) -> tuple[bytes, ...]:
    return tuple(
        np.ascontiguousarray(arrays[f"{prefix}_{suffix}"][index]).tobytes()
        for suffix in RECURRENCE_SUFFIXES
    )


def _require_recurrence_rows_equal(
    arrays: Mapping[str, np.ndarray],
    left_prefix: str,
    left_index: int,
    right_prefix: str,
    right_index: int,
    *,
    message: str,
) -> None:
    _require(
        _recurrence_row_bytes(arrays, left_prefix, left_index)
        == _recurrence_row_bytes(arrays, right_prefix, right_index),
        message,
    )


def validate_actual_forward_tensors(
    raw: Mapping[str, Any], ledger: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Validate one complete AF-00A raw tensor mapping and recompute outcome."""

    arrays = _load_arrays(raw)
    validate_request_ledger(ledger)
    _validate_ledger_artifact_bindings(arrays, ledger)

    budget32 = arrays["source_budget_f32"].reshape(())[()]
    dt32 = arrays["source_dt_f32"].reshape(())[()]
    radius32 = arrays["source_radius_f32"].reshape(())[()]
    _require(np.asarray(budget32, dtype=np.float32).tobytes() == EXPECTED_BUDGET32.tobytes(), "source float32 budget changed")
    _require(np.asarray(dt32, dtype=np.float32).tobytes() == EXPECTED_DT32.tobytes(), "sampler dt changed")
    expected_radius32 = np.asarray(EXPECTED_BUDGET32 / np.float32(5.0), dtype=np.float32)
    _require(np.asarray(radius32, dtype=np.float32).tobytes() == expected_radius32.tobytes(), "float32 product-ball radius changed")
    radius64 = np.float64(radius32)

    delta = arrays["source_delta_model_f32"]
    mask = np.zeros((10, 32), dtype=np.bool_)
    mask[:5, :3] = True
    _require(not bool(np.count_nonzero(delta[~mask])), "source delta is nonzero outside first-five XYZ")
    frozen = arrays["source_frozen_final_f32"]
    expected_target = np.where(
        delta == np.float32(0.0),
        frozen,
        np.asarray(frozen + delta, dtype=np.float32),
    ).astype(np.float32, copy=False)
    _require(_bytes_equal(arrays["source_target_normalized_f32"], expected_target), "zero-preserving normalized target reconstruction changed")

    compiled = arrays["reference_compiled_actions_f64"]
    source_actions = arrays["reference_source_actions_f64"]
    eager = arrays["reference_eager_actions_f64"]
    eager_final = arrays["reference_eager_final_f32"]
    for name, pair in (("compiled", compiled), ("source", source_actions), ("eager", eager), ("eager final", eager_final)):
        _require(_bytes_equal(pair[0], pair[1]), f"pre/post {name} reference changed")
    _require(_bytes_equal(source_actions, eager), "eager source and normalized-final actions differ")
    _require(_bytes_equal(eager_final[0], frozen), "fresh frozen normalized final changed")
    expected_target_physical = eager[0].copy()
    expected_target_physical[:5, :3] += np.asarray(delta[:5, :3], dtype=np.float64) * np.asarray(
        arrays["source_model_to_physical_scale_f32"][:5, :3], dtype=np.float64
    )
    target_physical = arrays["source_target_physical_f64"]
    _require(_bytes_equal(target_physical, expected_target_physical), "physical target reconstruction changed")

    zero_c = np.zeros((2, 5, 15), dtype=np.float32)
    for name in (
        "zero_requested_c_f32",
        "zero_requested_velocity_f32",
        "zero_applied_velocity_f32",
        "zero_executed_c_f32",
    ):
        _require(_bytes_equal(arrays[name], zero_c), f"{name} is not the exact positive-zero reference")
    _require(_bytes_equal(arrays["zero_final_normalized_f32"][0], frozen), "zero schedule final differs from frozen")
    _require(_bytes_equal(arrays["zero_final_normalized_f32"][0], arrays["zero_final_normalized_f32"][1]), "zero schedule final is not repeatable")
    _require(_bytes_equal(arrays["zero_returned_actions_f64"][0], eager[0]), "zero schedule actions differ from frozen")
    _require(_bytes_equal(arrays["zero_returned_actions_f64"][0], arrays["zero_returned_actions_f64"][1]), "zero schedule actions are not repeatable")
    _validate_recurrence_group(arrays, "zero", arrays["source_noise_f32"])
    _require_recurrence_rows_equal(
        arrays, "zero", 0, "zero", 1, message="zero schedule recurrence is not repeatable"
    )

    for prefix in SCHEDULE_GROUP_SIZES:
        _require(
            _bytes_equal(
                arrays[f"{prefix}_final_normalized_physical_f64"],
                arrays[f"{prefix}_returned_actions_f64"],
            ),
            f"{prefix} physical terminal trace differs from returned actions",
        )

    compact_delta = np.asarray(delta[:5, :3], dtype=np.float32).reshape(15)
    arm_a_row = np.asarray(compact_delta / np.float32(5.0), dtype=np.float32)
    arm_a_c = np.broadcast_to(arm_a_row, (2, 5, 15)).copy()
    _validate_transport_group(arrays, "arm_a", arm_a_c, radius32=radius32, budget32=budget32)
    _require(_bytes_equal(arrays["arm_a_final_normalized_f32"][0], arrays["arm_a_final_normalized_f32"][1]), "Arm A normalized final is not repeatable")
    _require(_bytes_equal(arrays["arm_a_returned_actions_f64"][0], arrays["arm_a_returned_actions_f64"][1]), "Arm A actions are not repeatable")
    _validate_recurrence_group(arrays, "arm_a", arrays["source_noise_f32"])
    _require_recurrence_rows_equal(
        arrays, "arm_a", 0, "arm_a", 1, message="Arm A recurrence is not repeatable"
    )

    expected_generation = np.repeat(np.arange(8, dtype=np.int16), 65)
    expected_population = np.tile(np.arange(65, dtype=np.int16), 8)
    expected_pair = np.full((520,), -1, dtype=np.int16)
    expected_sign = np.zeros((520,), dtype=np.int8)
    for generation in range(8):
        base = generation * 65
        for pair in range(32):
            expected_pair[base + 1 + 2 * pair : base + 3 + 2 * pair] = pair
            expected_sign[base + 1 + 2 * pair] = 1
            expected_sign[base + 2 + 2 * pair] = -1
    _require(_bytes_equal(arrays["cem_generation_i16"], expected_generation), "CEM generation indices changed")
    _require(_bytes_equal(arrays["cem_population_index_i16"], expected_population), "CEM population indices changed")
    _require(_bytes_equal(arrays["cem_pair_index_i16"], expected_pair), "CEM antithetic pair indices changed")
    _require(_bytes_equal(arrays["cem_sign_i8"], expected_sign), "CEM antithetic signs changed")
    _require(bool(np.all(arrays["cem_query_elapsed_ns_u64"] > 0)), "CEM query timing must be positive")

    mean_state = arrays["cem_mean_state_f64"]
    sigma_state = arrays["cem_sigma_state_f64"]
    raw_variance = arrays["cem_variance_after_f64"]
    raw_proposal = arrays["cem_raw_proposal_f64"]
    projected_c = arrays["cem_projected_c_f32"]
    normals = arrays["cem_raw_normals_f64"]
    expected_normals = np.random.Generator(np.random.PCG64(20260716)).standard_normal(
        (GENERATIONS, PAIR_COUNT, *COMPACT_SHAPE), dtype=np.float64
    )
    _require(
        _bytes_equal(normals, expected_normals),
        "CEM raw normals differ from the registered PCG64 seed stream",
    )
    initial_mean = np.asarray(arrays["arm_a_executed_c_f32"][0], dtype=np.float64)
    sigma0 = np.asarray(radius64 / np.sqrt(np.float64(15.0)), dtype=np.float64)
    _require(_bytes_equal(mean_state[0], initial_mean), "CEM initial mean differs from Arm A executed increments")
    _require(_bytes_equal(sigma_state[0], np.full(COMPACT_SHAPE, sigma0, dtype=np.float64)), "CEM initial sigma changed")

    recomputed_objective = np.empty((520,), dtype=np.float64)
    recomputed_metrics = np.empty((520, 4), dtype=np.float64)
    recomputed_gates = np.empty((520, 4), dtype=np.bool_)
    recomputed_energy = np.empty((520,), dtype=np.float64)
    for generation in range(8):
        start = generation * 65
        proposals = [mean_state[generation]]
        for pair in range(32):
            offset = sigma_state[generation] * normals[generation, pair]
            proposals.extend((mean_state[generation] + offset, mean_state[generation] - offset))
        expected_raw = np.stack(proposals, axis=0).astype(np.float64, copy=False)
        _require(_bytes_equal(raw_proposal[start : start + 65], expected_raw), f"CEM generation {generation} raw proposals changed")
        expected_projected = np.stack(
            [_project_blocks_float64(item, radius64).astype(np.float32) for item in expected_raw],
            axis=0,
        )
        _require(_bytes_equal(projected_c[start : start + 65], expected_projected), f"CEM generation {generation} projected proposals changed")

        for local in range(65):
            query = start + local
            objective, metrics, gates = _objective_metrics(
                arrays["cem_returned_actions_f64"][query], target_physical
            )
            recomputed_objective[query] = objective
            recomputed_metrics[query] = metrics
            recomputed_gates[query] = gates
            recomputed_energy[query] = _energy(arrays["cem_executed_c_f32"][query])

        ranked = sorted(range(start, start + 65), key=lambda query: (recomputed_objective[query], query))
        elites = np.asarray(ranked[:13], dtype=np.int64)
        _require(_bytes_equal(arrays["cem_elite_query_index_i64"][generation], elites), f"CEM generation {generation} elites changed")
        elite_values = np.asarray(arrays["cem_executed_c_f32"][elites], dtype=np.float64)
        unprojected_mean = np.mean(elite_values, axis=0, dtype=np.float64)
        next_mean = _project_blocks_float64(unprojected_mean, radius64)
        variance = np.mean(np.square(elite_values - next_mean), axis=0, dtype=np.float64)
        next_sigma = np.clip(
            np.sqrt(variance, dtype=np.float64),
            np.asarray(sigma0 / np.float64(16.0), dtype=np.float64),
            sigma0,
        )
        _require(_bytes_equal(raw_variance[generation], variance), f"CEM generation {generation} variance changed")
        _require(_bytes_equal(mean_state[generation + 1], next_mean), f"CEM generation {generation} mean update changed")
        _require(_bytes_equal(sigma_state[generation + 1], next_sigma), f"CEM generation {generation} sigma update changed")

    _validate_transport_group(arrays, "cem", projected_c, radius32=radius32, budget32=budget32)
    _validate_recurrence_group(arrays, "cem", arrays["source_noise_f32"])
    _require(_bytes_equal(arrays["cem_objective_f64"], recomputed_objective), "stored CEM objectives differ from physical actions")
    _require(_bytes_equal(arrays["cem_fidelity_metrics_f64"], recomputed_metrics), "stored CEM fidelity metrics differ from physical actions")
    _require(_bytes_equal(arrays["cem_gate_pass_bool"], recomputed_gates), "stored CEM gate bits differ from physical actions")
    _require(_bytes_equal(arrays["cem_energy_f64"], recomputed_energy), "stored CEM energies differ from executed increments")

    # Identical executed schedules must have identical scientific output.
    schedule_outputs: dict[bytes, tuple[bytes, ...]] = {}
    for query in range(520):
        key = arrays["cem_executed_c_f32"][query].tobytes()
        output = (
            arrays["cem_trace_x_t_f32"][query].tobytes(),
            arrays["cem_trace_v_base_f32"][query].tobytes(),
            arrays["cem_trace_total_velocity_f32"][query].tobytes(),
            arrays["cem_trace_x_next_f32"][query].tobytes(),
            arrays["cem_final_normalized_f32"][query].tobytes(),
            arrays["cem_returned_actions_f64"][query].tobytes(),
        )
        if key in schedule_outputs:
            _require(schedule_outputs[key] == output, f"identical CEM schedule has changed output at query {query}")
        else:
            schedule_outputs[key] = output

    arm_a_objective, arm_a_metrics, arm_a_gates = _objective_metrics(
        arrays["arm_a_returned_actions_f64"][0], target_physical
    )
    arm_a_energy = _energy(arrays["arm_a_executed_c_f32"][0])
    pool_objective = np.concatenate((np.asarray([arm_a_objective]), recomputed_objective))
    pool_energy = np.concatenate((np.asarray([arm_a_energy]), recomputed_energy))
    pool_pass = np.concatenate((np.asarray([bool(np.all(arm_a_gates))]), np.all(recomputed_gates, axis=1)))
    candidates = [int(index) for index in np.flatnonzero(pool_pass)]
    if candidates:
        selected = min(candidates, key=lambda index: (pool_energy[index], pool_objective[index], index))
    else:
        selected = min(range(521), key=lambda index: (pool_objective[index], pool_energy[index], index))
    stored_selected = int(arrays["selected_pool_index_i64"].reshape(())[()])
    _require(stored_selected == selected, "stored final pool selection changed")

    if selected == 0:
        selected_c = arrays["arm_a_requested_c_f32"][0]
        selected_u = arrays["arm_a_requested_velocity_f32"][0]
        selected_applied = arrays["arm_a_applied_velocity_f32"][0]
        selected_executed = arrays["arm_a_executed_c_f32"][0]
        selected_final = arrays["arm_a_final_normalized_f32"][0]
        selected_actions = arrays["arm_a_returned_actions_f64"][0]
    else:
        query = selected - 1
        selected_c = arrays["cem_projected_c_f32"][query]
        selected_u = arrays["cem_requested_velocity_f32"][query]
        selected_applied = arrays["cem_applied_velocity_f32"][query]
        selected_executed = arrays["cem_executed_c_f32"][query]
        selected_final = arrays["cem_final_normalized_f32"][query]
        selected_actions = arrays["cem_returned_actions_f64"][query]

    for replicate in range(2):
        for suffix, expected in (
            ("requested_c_f32", selected_c),
            ("requested_velocity_f32", selected_u),
            ("applied_velocity_f32", selected_applied),
            ("executed_c_f32", selected_executed),
            ("final_normalized_f32", selected_final),
            ("returned_actions_f64", selected_actions),
        ):
            _require(_bytes_equal(arrays[f"arm_b_{suffix}"][replicate], expected), f"Arm B replay {replicate} {suffix} differs from selected evaluation")
    _validate_transport_group(
        arrays,
        "arm_b",
        np.broadcast_to(selected_c, (2, 5, 15)).copy(),
        radius32=radius32,
        budget32=budget32,
    )
    _validate_recurrence_group(arrays, "arm_b", arrays["source_noise_f32"])
    selected_prefix = "arm_a" if selected == 0 else "cem"
    selected_index = 0 if selected == 0 else selected - 1
    for replicate in range(2):
        _require_recurrence_rows_equal(
            arrays,
            "arm_b",
            replicate,
            selected_prefix,
            selected_index,
            message=f"Arm B replay {replicate} recurrence differs from selected evaluation",
        )

    reverse_c = selected_c[::-1].copy()
    reverse_u = selected_u[::-1].copy()
    expected_reverse_c = np.broadcast_to(reverse_c, (2, 5, 15)).copy()
    _require(_bytes_equal(arrays["arm_c_requested_c_f32"], expected_reverse_c), "Arm C did not reverse B requested c rows")
    _require(_bytes_equal(arrays["arm_c_requested_velocity_f32"], np.broadcast_to(reverse_u, (2, 5, 15)).copy()), "Arm C did not reverse B requested velocity rows byte-for-byte")
    _validate_transport_group(arrays, "arm_c", expected_reverse_c, radius32=radius32, budget32=budget32)
    _validate_recurrence_group(arrays, "arm_c", arrays["source_noise_f32"])
    _require_recurrence_rows_equal(
        arrays, "arm_c", 0, "arm_c", 1, message="Arm C recurrence is not repeatable"
    )
    _require(_bytes_equal(arrays["arm_c_final_normalized_f32"][0], arrays["arm_c_final_normalized_f32"][1]), "Arm C normalized final is not repeatable")
    _require(_bytes_equal(arrays["arm_c_returned_actions_f64"][0], arrays["arm_c_returned_actions_f64"][1]), "Arm C actions are not repeatable")

    b_objective, b_metrics, b_gates = _objective_metrics(
        arrays["arm_b_returned_actions_f64"][0], target_physical
    )
    c_objective, c_metrics, c_gates = _objective_metrics(
        arrays["arm_c_returned_actions_f64"][0], target_physical
    )
    b_changed = bool(
        not _bytes_equal(arrays["arm_b_requested_velocity_f32"][0], arrays["arm_a_requested_velocity_f32"][0])
        and not _bytes_equal(arrays["arm_b_applied_velocity_f32"][0], arrays["arm_a_applied_velocity_f32"][0])
    )
    arm_a_passed = bool(np.all(arm_a_gates))
    b_passed = bool(np.all(b_gates))
    if arm_a_passed:
        outcome = "baseline_sufficient_no_incremental_support"
    elif b_changed and b_passed:
        outcome = "mechanism_pass"
    else:
        outcome = "frozen_cem_negative"

    return {
        "schema_version": SCHEMA_VERSION,
        "request_count": POLICY_REQUESTS,
        "selected_pool_index": selected,
        "selected_cem_query_index": None if selected == 0 else selected - 1,
        "arm_a": {
            "objective": float(arm_a_objective),
            "energy": float(arm_a_energy),
            "metrics": {name: float(value) for name, value in zip(METRIC_NAMES, arm_a_metrics)},
            "passed": arm_a_passed,
        },
        "arm_b": {
            "objective": float(b_objective),
            "energy": float(_energy(arrays["arm_b_executed_c_f32"][0])),
            "metrics": {name: float(value) for name, value in zip(METRIC_NAMES, b_metrics)},
            "passed": b_passed,
            "changed_from_arm_a": b_changed,
        },
        "arm_c": {
            "objective": float(c_objective),
            "energy": float(_energy(arrays["arm_c_executed_c_f32"][0])),
            "metrics": {name: float(value) for name, value in zip(METRIC_NAMES, c_metrics)},
            "passed": bool(np.all(c_gates)),
        },
        "outcome": outcome,
    }


def validate_actual_forward_npz(
    path: str | Path, ledger: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Load a no-pickle NPZ archive and apply the same mapping validator."""

    try:
        with np.load(Path(path), allow_pickle=False) as archive:
            _require(set(archive.files) == set(ARRAY_SPECS), "AF-00A NPZ member names changed")
            arrays = {name: np.asarray(archive[name]).copy() for name in archive.files}
    except ActualForwardValidationError:
        raise
    except (OSError, ValueError, TypeError) as error:
        raise ActualForwardValidationError(f"AF-00A NPZ cannot be loaded without pickle: {error}") from error
    return validate_actual_forward_tensors(arrays, ledger)


__all__ = [
    "ARRAY_SPECS",
    "ActualForwardValidationError",
    "FIDELITY_LIMITS64",
    "METRIC_NAMES",
    "POLICY_REQUESTS",
    "SEARCH_QUERIES",
    "validate_actual_forward_npz",
    "validate_actual_forward_tensors",
    "validate_request_ledger",
]
