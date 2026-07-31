#!/usr/bin/env python3
"""Deeply validate one final Poisson episode and its immutable arm trace.

The compact episode schema is necessary but not sufficient: a self-consistent
``result.json`` could otherwise disagree with the 2 ms audit ledger it cites.
This validator loads the referenced ``active_arm_audit_trace``, verifies both
layers of hashes, and independently reconstructs the reported endpoints from
the serialized raw ledgers.
"""

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main.poisson_fullbody.contracts import (  # noqa: E402
    ArtifactContractError,
    canonical_json_bytes,
    load_hashed_json,
    sha256_bytes,
    validate_artifact_reference,
)
from main.poisson_fullbody.result_schema import validate_episode_result  # noqa: E402


TRACE_SCHEMA_VERSION = "vlsa_poisson_active_arm_trace.v1"
TRACE_ARTIFACT_TYPE = "active_arm_audit_trace"
PHYSICS_DT_SECONDS = 0.002
INNER_UPDATES_PER_HIGH_LEVEL = 5
PHYSICS_SUBSTEPS_PER_INNER = 5
PAPER_CAR_THRESHOLD_M = 0.001


def _fail(label: str, message: str) -> None:
    raise ArtifactContractError("%s %s" % (label, message))


def _require(condition: bool, label: str, message: str = "is inconsistent") -> None:
    if not condition:
        _fail(label, message)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        _fail(label, "must be an object")
    return value


def _list(value: Any, label: str) -> List[Any]:
    if not isinstance(value, list):
        _fail(label, "must be a list")
    return value


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _fail(label, "must be a nonnegative integer")
    return int(value)


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        _fail(label, "must be finite")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ArtifactContractError("%s must be finite" % label) from error
    if not math.isfinite(number):
        _fail(label, "must be finite")
    return number


def _vector(value: Any, length: int, label: str) -> List[float]:
    items = _list(value, label)
    if len(items) != length:
        _fail(label, "must contain %d finite values" % length)
    return [_finite(item, "%s[%d]" % (label, index)) for index, item in enumerate(items)]


def _norm(value: Sequence[float]) -> float:
    return math.sqrt(sum(float(item) * float(item) for item in value))


def _canonical_equal(left: Any, right: Any) -> bool:
    try:
        return canonical_json_bytes(left) == canonical_json_bytes(right)
    except (TypeError, ValueError):
        return False


def _raw_ledger_sha256(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _close(left: Any, right: Any, label: str, *, absolute: float = 1e-14) -> None:
    observed = _finite(left, label)
    expected = _finite(right, label + ".recomputed")
    if not math.isclose(observed, expected, rel_tol=1e-12, abs_tol=absolute):
        _fail(label, "differs from the trace recomputation")


def _optional_minimum(values: Iterable[Any], label: str) -> Optional[float]:
    output: List[float] = []
    for index, value in enumerate(values):
        if value is not None:
            output.append(_finite(value, "%s[%d]" % (label, index)))
    return min(output) if output else None


def _check_measurement(record: Any, expected: Optional[float], label: str) -> None:
    measurement = _mapping(record, label)
    available = measurement.get("available")
    _require(isinstance(available, bool), label + ".available", "must be Boolean")
    _require(available == (expected is not None), label + ".available")
    value = measurement.get("value")
    if expected is None:
        _require(value is None, label + ".value", "must be null when unavailable")
    else:
        _close(value, expected, label + ".value")


def _coordinate_for_inner(global_index: int) -> Tuple[int, int]:
    return (
        global_index // INNER_UPDATES_PER_HIGH_LEVEL,
        global_index % INNER_UPDATES_PER_HIGH_LEVEL,
    )


def _coordinate_for_physics(global_index: int) -> Tuple[int, int, int]:
    inner_global = global_index // PHYSICS_SUBSTEPS_PER_INNER
    return (
        inner_global // INNER_UPDATES_PER_HIGH_LEVEL,
        inner_global % INNER_UPDATES_PER_HIGH_LEVEL,
        global_index % PHYSICS_SUBSTEPS_PER_INNER,
    )


def _row_coordinate(row: Mapping[str, Any], *, physics: bool, label: str) -> Tuple[int, ...]:
    high = _integer(row.get("high_level_index"), label + ".high_level_index")
    inner = _integer(row.get("inner_control_index"), label + ".inner_control_index")
    if physics:
        substep = _integer(
            row.get("physics_substep_index"), label + ".physics_substep_index"
        )
        return high, inner, substep
    return high, inner


def _validate_identity(result: Mapping[str, Any], trace: Mapping[str, Any]) -> Mapping[str, Any]:
    _require(trace.get("schema_version") == TRACE_SCHEMA_VERSION, "trace.schema_version")
    _require(trace.get("scientific_result") is False, "trace.scientific_result")
    for field in ("run_id", "case_id", "arm"):
        _require(trace.get(field) == result.get(field), "trace.%s" % field)
    outcome = _mapping(trace.get("outcome"), "trace.outcome")
    _require(outcome.get("arm") == result.get("arm"), "trace.outcome.arm")
    _require(
        outcome.get("completion_class") == result.get("completion_class"),
        "trace.outcome.completion_class",
    )
    return outcome


def _validate_execution(
    result: Mapping[str, Any], outcome: Mapping[str, Any]
) -> Tuple[int, int, int, List[Any], List[Any]]:
    execution = _mapping(result.get("execution"), "result.execution")
    prefix = _mapping(outcome.get("prefix"), "trace.outcome.prefix")
    counters: Dict[str, int] = {}
    for field in ("high_level_steps", "inner_control_steps", "physics_substeps"):
        counters[field] = _integer(prefix.get(field), "trace.outcome.prefix.%s" % field)
        _require(
            execution.get(field) == counters[field],
            "result.execution.%s" % field,
        )
    high_steps = counters["high_level_steps"]
    inner_steps = counters["inner_control_steps"]
    physics_steps = counters["physics_substeps"]
    exposure_seconds = physics_steps * PHYSICS_DT_SECONDS
    _close(prefix.get("physics_exposure_seconds"), exposure_seconds, "trace.outcome.prefix.physics_exposure_seconds")
    _close(execution.get("physics_exposure_seconds"), exposure_seconds, "result.execution.physics_exposure_seconds")

    entered = _list(outcome.get("entered_source_actions"), "trace.outcome.entered_source_actions")
    completed = _list(outcome.get("completed_source_actions"), "trace.outcome.completed_source_actions")
    _require(len(entered) == high_steps, "trace.outcome.entered_source_actions", "count differs from high-level entered counter")
    _require(
        len(completed) == _integer(execution.get("completed_high_level_steps"), "result.execution.completed_high_level_steps"),
        "trace.outcome.completed_source_actions",
        "count differs from completed high-level counter",
    )
    for index, action in enumerate(entered):
        _vector(action, 7, "trace.outcome.entered_source_actions[%d]" % index)
    for index, action in enumerate(completed):
        _vector(action, 7, "trace.outcome.completed_source_actions[%d]" % index)
    _require(
        _canonical_equal(completed, entered[: len(completed)]),
        "trace.outcome.completed_source_actions",
        "must be the exact entered-action prefix",
    )
    _require(
        _raw_ledger_sha256(entered) == execution.get("entered_source_action_prefix_sha256"),
        "result.execution.entered_source_action_prefix_sha256",
    )
    _require(
        _raw_ledger_sha256(completed)
        == execution.get("completed_high_level_source_action_prefix_sha256"),
        "result.execution.completed_high_level_source_action_prefix_sha256",
    )

    nominal_ledger = _list(outcome.get("nominal_ledger"), "trace.outcome.nominal_ledger")
    executed_ledger = _list(outcome.get("executed_ledger"), "trace.outcome.executed_ledger")
    inner_trace = _list(outcome.get("inner_trace"), "trace.outcome.inner_trace")
    _require(
        len(nominal_ledger) == len(executed_ledger) == len(inner_trace),
        "trace command ledgers",
        "must have equal lengths",
    )
    issued_count = len(inner_trace)
    _require(issued_count <= inner_steps <= issued_count + 1, "trace inner-control counter")
    _require(
        issued_count == ((physics_steps + PHYSICS_SUBSTEPS_PER_INNER - 1) // PHYSICS_SUBSTEPS_PER_INNER),
        "trace issued-command count",
        "differs from the physics-exposed command prefix",
    )
    for index, values in enumerate(zip(nominal_ledger, executed_ledger, inner_trace)):
        expected = _coordinate_for_inner(index)
        for name, raw_row in zip(("nominal_ledger", "executed_ledger", "inner_trace"), values):
            row = _mapping(raw_row, "trace.outcome.%s[%d]" % (name, index))
            _require(
                _row_coordinate(row, physics=False, label="trace.outcome.%s[%d]" % (name, index)) == expected,
                "trace.outcome.%s[%d] coordinate" % (name, index),
            )
        nominal = _vector(
            _mapping(nominal_ledger[index], "nominal ledger row").get("qdot_rad_s"),
            7,
            "trace.outcome.nominal_ledger[%d].qdot_rad_s" % index,
        )
        executed = _vector(
            _mapping(executed_ledger[index], "executed ledger row").get("qdot_rad_s"),
            7,
            "trace.outcome.executed_ledger[%d].qdot_rad_s" % index,
        )
        _require(
            _canonical_equal(
                _mapping(inner_trace[index], "inner trace row").get("source_action"),
                entered[expected[0]],
            ),
            "trace.outcome.inner_trace[%d].source_action" % index,
        )
        del nominal, executed

    fail_rows = _list(
        outcome.get("fail_closed_attempt_trace"),
        "trace.outcome.fail_closed_attempt_trace",
    )
    entered_inner_coordinates = set(_coordinate_for_inner(index) for index in range(inner_steps))
    for index, raw_row in enumerate(fail_rows):
        row = _mapping(raw_row, "trace.outcome.fail_closed_attempt_trace[%d]" % index)
        coordinate = _row_coordinate(
            row,
            physics=False,
            label="trace.outcome.fail_closed_attempt_trace[%d]" % index,
        )
        _require(coordinate in entered_inner_coordinates, "trace fail-closed coordinate")
    if inner_steps == issued_count + 1:
        terminal_coordinate = _coordinate_for_inner(issued_count)
        _require(
            any(
                _row_coordinate(
                    _mapping(row, "terminal fail row"),
                    physics=False,
                    label="terminal fail row",
                )
                == terminal_coordinate
                and _mapping(row, "terminal fail row").get("physics_executed_after_attempt") is False
                for row in fail_rows
            ),
            "trace terminal provider attempt",
            "is missing for an entered inner update with no issued command",
        )

    if inner_steps:
        _require(
            high_steps == _coordinate_for_inner(inner_steps - 1)[0] + 1,
            "trace high-level counter",
        )
    else:
        _require(high_steps == 0, "trace high-level counter")
    _require(physics_steps <= inner_steps * PHYSICS_SUBSTEPS_PER_INNER, "trace physics counter")

    clock = _mapping(outcome.get("physics_clock"), "trace.outcome.physics_clock")
    settled_time = _finite(clock.get("settled_time_s"), "trace.outcome.physics_clock.settled_time_s")
    terminal_time = _finite(clock.get("terminal_time_s"), "trace.outcome.physics_clock.terminal_time_s")
    observed = _finite(clock.get("observed_exposure_s"), "trace.outcome.physics_clock.observed_exposure_s")
    _close(terminal_time - settled_time, observed, "trace.outcome.physics_clock.delta", absolute=1e-10)
    _close(observed, exposure_seconds, "trace.outcome.physics_clock.observed_exposure_s", absolute=1e-10)
    _close(clock.get("expected_from_completed_substeps_s"), exposure_seconds, "trace.outcome.physics_clock.expected_from_completed_substeps_s", absolute=1e-10)
    _require(
        clock.get("exact_count_consistent_within_abs_1e_10_s") is True,
        "trace.outcome.physics_clock.exact_count_consistent_within_abs_1e_10_s",
    )

    _require(
        _raw_ledger_sha256(nominal_ledger)
        == execution.get("nominal_joint_velocity_ledger_sha256"),
        "result.execution.nominal_joint_velocity_ledger_sha256",
    )
    _require(
        _raw_ledger_sha256(executed_ledger)
        == execution.get("executed_controller_action_ledger_sha256"),
        "result.execution.executed_controller_action_ledger_sha256",
    )
    _require(
        execution.get("terminal_simulator_state_sha256")
        == outcome.get("terminal_simulator_state_sha256"),
        "result.execution.terminal_simulator_state_sha256",
    )
    _require(
        execution.get("terminal_observation_sha256")
        == outcome.get("terminal_observation_sha256"),
        "result.execution.terminal_observation_sha256",
    )

    registered = _integer(
        _mapping(result.get("case_identity"), "result.case_identity").get(
            "registered_source_exposure_high_level_steps"
        ),
        "result.case_identity.registered_source_exposure_high_level_steps",
    )
    exposure_complete = bool(outcome.get("exposure_complete"))
    _require(
        isinstance(outcome.get("exposure_complete"), bool),
        "trace.outcome.exposure_complete",
        "must be Boolean",
    )
    recomputed_complete = (
        high_steps == registered
        and inner_steps == registered * INNER_UPDATES_PER_HIGH_LEVEL
        and physics_steps
        == registered * INNER_UPDATES_PER_HIGH_LEVEL * PHYSICS_SUBSTEPS_PER_INNER
        and len(completed) == registered
        and outcome.get("completion_class") == "executed"
    )
    _require(exposure_complete == recomputed_complete, "trace.outcome.exposure_complete")
    _require(execution.get("exposure_complete") == exposure_complete, "result.execution.exposure_complete")
    _require(execution.get("fail_closed_no_further_physics") == (not exposure_complete), "result.execution.fail_closed_no_further_physics")
    _require(execution.get("terminal_reason") == outcome.get("terminal_reason"), "result.execution.terminal_reason")
    return high_steps, inner_steps, physics_steps, entered, completed


def _validate_motion_and_tracking(
    result: Mapping[str, Any], outcome: Mapping[str, Any], physics_steps: int
) -> None:
    nominal_ledger = _list(outcome.get("nominal_ledger"), "trace.outcome.nominal_ledger")
    executed_ledger = _list(outcome.get("executed_ledger"), "trace.outcome.executed_ledger")
    physics_trace = _list(outcome.get("physics_trace"), "trace.outcome.physics_trace")
    _require(len(physics_trace) == physics_steps, "trace.outcome.physics_trace", "count differs from physics counter")
    eef_audit = _mapping(outcome.get("eef_path_audit"), "trace.outcome.eef_path_audit")
    previous_eef = _vector(
        eef_audit.get("settled_forwarded_eef_position_m"),
        3,
        "trace.outcome.eef_path_audit.settled_forwarded_eef_position_m",
    )
    _require(
        _integer(eef_audit.get("position_sample_count"), "trace.outcome.eef_path_audit.position_sample_count")
        == physics_steps + 1,
        "trace.outcome.eef_path_audit.position_sample_count",
    )
    tracking = _mapping(outcome.get("tracking"), "trace.outcome.tracking")
    linf_threshold = _finite(tracking.get("maximum_linf_threshold_rad_s"), "trace.outcome.tracking.maximum_linf_threshold_rad_s")
    rmse_threshold = _finite(tracking.get("maximum_rmse_threshold_rad_s"), "trace.outcome.tracking.maximum_rmse_threshold_rad_s")
    _require(linf_threshold >= 0.0 and rmse_threshold >= 0.0, "trace tracking thresholds")

    nominal_motion = 0.0
    safe_motion = 0.0
    measured_motion = 0.0
    correction_motion = 0.0
    eef_path = 0.0
    squared_tracking_error = 0.0
    maximum_linf = 0.0
    first_crossing: Optional[Dict[str, Any]] = None
    exposed_commands = set()
    zero_commands = set()
    for index, raw_row in enumerate(physics_trace):
        label = "trace.outcome.physics_trace[%d]" % index
        row = _mapping(raw_row, label)
        coordinate = _coordinate_for_physics(index)
        _require(_row_coordinate(row, physics=True, label=label) == coordinate, label + " coordinate")
        command_index = index // PHYSICS_SUBSTEPS_PER_INNER
        _require(command_index < len(nominal_ledger), label + " command index")
        nominal = _vector(row.get("nominal_joint_velocity_command_rad_s"), 7, label + ".nominal_joint_velocity_command_rad_s")
        issued = _vector(row.get("issued_joint_velocity_command_rad_s"), 7, label + ".issued_joint_velocity_command_rad_s")
        measured = _vector(row.get("measured_arm_joint_velocity_rad_s"), 7, label + ".measured_arm_joint_velocity_rad_s")
        eef = _vector(row.get("forwarded_eef_position_m"), 3, label + ".forwarded_eef_position_m")
        _require(
            _canonical_equal(nominal, _mapping(nominal_ledger[command_index], "nominal row").get("qdot_rad_s")),
            label + ".nominal_joint_velocity_command_rad_s",
            "differs from the command ledger",
        )
        _require(
            _canonical_equal(issued, _mapping(executed_ledger[command_index], "executed row").get("qdot_rad_s")),
            label + ".issued_joint_velocity_command_rad_s",
            "differs from the command ledger",
        )
        difference = [safe - nominal_value for safe, nominal_value in zip(issued, nominal)]
        tracking_error = [observed - command for observed, command in zip(measured, issued)]
        nominal_motion += _norm(nominal) * PHYSICS_DT_SECONDS
        safe_motion += _norm(issued) * PHYSICS_DT_SECONDS
        measured_motion += _norm(measured) * PHYSICS_DT_SECONDS
        correction_motion += _norm(difference) * PHYSICS_DT_SECONDS
        eef_path += _norm([value - reference for value, reference in zip(eef, previous_eef)])
        previous_eef = eef
        key = coordinate[:2]
        exposed_commands.add(key)
        if _norm(issued) <= 1e-12:
            zero_commands.add(key)
        row_linf = max(abs(value) for value in tracking_error)
        squared_tracking_error += sum(value * value for value in tracking_error)
        maximum_linf = max(maximum_linf, row_linf)
        cumulative_rmse = math.sqrt(squared_tracking_error / (7 * (index + 1)))
        if first_crossing is None and (
            row_linf > linf_threshold or cumulative_rmse > rmse_threshold
        ):
            first_crossing = {
                "high_level_index": coordinate[0],
                "inner_control_index": coordinate[1],
                "physics_substep_index": coordinate[2],
                "error_linf_rad_s": row_linf,
                "cumulative_rmse_rad_s": cumulative_rmse,
                "linf_threshold_rad_s": linf_threshold,
                "rmse_threshold_rad_s": rmse_threshold,
                "command_rad_s": issued,
                "measured_rad_s": measured,
            }

    usefulness = _mapping(outcome.get("usefulness"), "trace.outcome.usefulness")
    for field, recomputed in (
        ("nominal_joint_motion_integral_rad", nominal_motion),
        ("safe_joint_motion_integral_rad", safe_motion),
        ("measured_joint_motion_integral_rad", measured_motion),
        ("correction_integral_rad", correction_motion),
        ("eef_path_length_m", eef_path),
    ):
        _close(usefulness.get(field), recomputed, "trace.outcome.usefulness.%s" % field)
    command_count = len(exposed_commands)
    zero_count = len(zero_commands)
    _require(command_count == len(executed_ledger), "trace exposed-command count")
    _close(
        usefulness.get("zero_motion_fraction"),
        float(zero_count) / command_count if command_count else 1.0,
        "trace.outcome.usefulness.zero_motion_fraction",
    )
    _require(
        usefulness.get("all_issued_arm_joint_commands_zero")
        is bool(command_count and zero_count == command_count),
        "trace.outcome.usefulness.all_issued_arm_joint_commands_zero",
    )
    _require(
        usefulness.get("safety_by_no_execution") is (physics_steps == 0),
        "trace.outcome.usefulness.safety_by_no_execution",
    )
    _close(
        usefulness.get("motion_retention_ratio"),
        safe_motion / nominal_motion if nominal_motion > 0.0 else 0.0,
        "trace.outcome.usefulness.motion_retention_ratio",
    )

    rmse = math.sqrt(squared_tracking_error / (7 * physics_steps)) if physics_steps else 0.0
    _close(tracking.get("maximum_linf_error_rad_s"), maximum_linf, "trace.outcome.tracking.maximum_linf_error_rad_s")
    _close(tracking.get("cumulative_rmse_rad_s"), rmse, "trace.outcome.tracking.cumulative_rmse_rad_s")
    _require(
        tracking.get("first_threshold_crossing") == first_crossing,
        "trace.outcome.tracking.first_threshold_crossing",
    )
    _require(
        _integer(tracking.get("observed_physics_substep_count"), "trace.outcome.tracking.observed_physics_substep_count") == physics_steps,
        "trace.outcome.tracking.observed_physics_substep_count",
    )
    _require(
        _integer(tracking.get("command_count"), "trace.outcome.tracking.command_count") == len(executed_ledger),
        "trace.outcome.tracking.command_count",
    )
    verified = sum(
        _mapping(row, "physics row").get("physics_substep_index")
        == PHYSICS_SUBSTEPS_PER_INNER - 1
        for row in physics_trace
    )
    _require(
        _integer(tracking.get("verified_terminal_command_count"), "trace.outcome.tracking.verified_terminal_command_count") == verified,
        "trace.outcome.tracking.verified_terminal_command_count",
    )
    _require(
        tracking.get("final_fifth_command_verified")
        is bool(physics_steps > 0 and physics_steps % PHYSICS_SUBSTEPS_PER_INNER == 0),
        "trace.outcome.tracking.final_fifth_command_verified",
    )
    validity = _mapping(outcome.get("validity"), "trace.outcome.validity")
    _close(validity.get("maximum_velocity_tracking_error_rad_s"), maximum_linf, "trace.outcome.validity.maximum_velocity_tracking_error_rad_s")
    _close(validity.get("velocity_tracking_rmse_rad_s"), rmse, "trace.outcome.validity.velocity_tracking_rmse_rad_s")
    _require(validity.get("first_velocity_tracking_threshold_crossing") == first_crossing, "trace.outcome.validity.first_velocity_tracking_threshold_crossing")
    _require(
        _integer(validity.get("velocity_tracking_observed_physics_substep_count"), "trace.outcome.validity.velocity_tracking_observed_physics_substep_count") == physics_steps,
        "trace.outcome.validity.velocity_tracking_observed_physics_substep_count",
    )
    _require(
        _canonical_equal(_mapping(result.get("endpoints"), "result.endpoints").get("usefulness"), usefulness),
        "result.endpoints.usefulness",
        "differs from the trace endpoint",
    )


def _validate_car(
    result: Mapping[str, Any], outcome: Mapping[str, Any], completed: Sequence[Any]
) -> None:
    ledger = _mapping(outcome.get("paper_car_position_ledger"), "trace.outcome.paper_car_position_ledger")
    settled = _vector(
        ledger.get("settled_active_obstacle_root_position_m"),
        3,
        "trace.outcome.paper_car_position_ledger.settled_active_obstacle_root_position_m",
    )
    for field in (
        "settled_active_obstacle_position_sha256",
        "source_settled_active_obstacle_position_sha256",
        "historical_settled_active_obstacle_position_sha256",
    ):
        value = ledger.get(field)
        _require(isinstance(value, str) and len(value) == 64, "trace CAR ledger.%s" % field)
    _require(
        ledger.get("settled_active_obstacle_position_sha256")
        == ledger.get("source_settled_active_obstacle_position_sha256")
        == ledger.get("historical_settled_active_obstacle_position_sha256"),
        "trace settled obstacle pairing hashes",
    )
    rows = _list(ledger.get("completed_post_step_positions"), "trace CAR completed positions")
    _require(len(rows) == len(completed), "trace CAR completed positions", "count differs from completed source actions")
    maximum = 0.0
    for index, raw_row in enumerate(rows):
        row = _mapping(raw_row, "trace CAR completed positions[%d]" % index)
        _require(row.get("high_level_index") == index, "trace CAR position index")
        position = _vector(row.get("position_m"), 3, "trace CAR position[%d]" % index)
        displacement = sum(abs(value - reference) for value, reference in zip(position, settled))
        _close(row.get("l1_displacement_from_settled_m"), displacement, "trace CAR displacement[%d]" % index, absolute=1e-15)
        maximum = max(maximum, displacement)

    car = _mapping(outcome.get("paper_car"), "trace.outcome.paper_car")
    complete = bool(outcome.get("exposure_complete"))
    _require(car.get("available") is complete, "trace.outcome.paper_car.available")
    _close(car.get("maximum_active_obstacle_l1_displacement_m"), maximum, "trace.outcome.paper_car.maximum_active_obstacle_l1_displacement_m")
    _close(car.get("threshold_m"), PAPER_CAR_THRESHOLD_M, "trace.outcome.paper_car.threshold_m")
    if complete:
        collision = maximum > PAPER_CAR_THRESHOLD_M
        _require(car.get("collision") is collision, "trace.outcome.paper_car.collision")
        _require(car.get("avoidance") is (not collision), "trace.outcome.paper_car.avoidance")
        _require(car.get("reason") is None, "trace.outcome.paper_car.reason")
    else:
        _require(car.get("collision") is None and car.get("avoidance") is None, "trace.outcome.paper_car right-censoring")
        _require(car.get("reason") == "fixed_exposure_incomplete", "trace.outcome.paper_car.reason")
    _require(
        car.get("position_ledger_sha256") == _raw_ledger_sha256(ledger),
        "trace.outcome.paper_car.position_ledger_sha256",
    )
    result_car = _mapping(
        _mapping(_mapping(result.get("endpoints"), "result.endpoints").get("safety"), "result.endpoints.safety").get("paper_car"),
        "result.endpoints.safety.paper_car",
    )
    _require(_canonical_equal(result_car, car), "result.endpoints.safety.paper_car", "differs from the trace endpoint")


def _validate_goal_snapshot(
    raw_snapshot: Any,
    previous_values: Optional[List[bool]],
    atom_count: int,
    label: str,
) -> Tuple[Mapping[str, Any], List[bool]]:
    snapshot = _mapping(raw_snapshot, label)
    raw_values = _list(snapshot.get("values"), label + ".values")
    _require(len(raw_values) == atom_count, label + ".values", "length differs from goal definition")
    _require(all(isinstance(value, bool) for value in raw_values), label + ".values", "must be Boolean")
    values = [bool(value) for value in raw_values]
    prior = [False] * atom_count if previous_values is None else previous_values
    satisfied = sum(values)
    _require(snapshot.get("satisfied_count") == satisfied, label + ".satisfied_count")
    _close(snapshot.get("fraction"), float(satisfied) / atom_count, label + ".fraction")
    _require(snapshot.get("all_satisfied") is all(values), label + ".all_satisfied")
    newly = [index for index, (old, new) in enumerate(zip(prior, values)) if not old and new]
    regressed = [index for index, (old, new) in enumerate(zip(prior, values)) if old and not new]
    _require(snapshot.get("newly_satisfied_indices") == newly, label + ".newly_satisfied_indices")
    _require(snapshot.get("regressed_indices") == regressed, label + ".regressed_indices")
    _require(snapshot.get("inert") is True, label + ".inert")
    _require(
        snapshot.get("simulator_state_sha256_before")
        == snapshot.get("simulator_state_sha256_after"),
        label + " simulator-state inertness",
    )
    return snapshot, values


def _validate_task(
    result: Mapping[str, Any],
    outcome: Mapping[str, Any],
    high_steps: int,
    completed: Sequence[Any],
) -> None:
    definition = _mapping(outcome.get("goal_definition"), "trace.outcome.goal_definition")
    atoms = _list(definition.get("goal_atoms"), "trace.outcome.goal_definition.goal_atoms")
    _require(bool(atoms), "trace.outcome.goal_definition.goal_atoms", "must be nonempty")
    for index, raw_atom in enumerate(atoms):
        atom = _mapping(raw_atom, "trace goal atom[%d]" % index)
        _require(atom.get("index") == index, "trace goal atom index")
        _require(atom.get("predicate") in ("in", "on"), "trace goal atom predicate")
        arguments = _list(atom.get("arguments"), "trace goal atom arguments")
        _require(len(arguments) == 2 and all(isinstance(value, str) and value for value in arguments), "trace goal atom arguments")
    definition_without_hash = dict(definition)
    expected_definition_hash = definition_without_hash.pop("goal_definition_sha256", None)
    _require(
        expected_definition_hash == _raw_ledger_sha256(definition_without_hash),
        "trace.outcome.goal_definition.goal_definition_sha256",
    )

    ledger = _list(outcome.get("goal_progress_ledger"), "trace.outcome.goal_progress_ledger")
    _require(bool(ledger), "trace.outcome.goal_progress_ledger", "must be nonempty")
    expected_length = 1 + len(completed) + (0 if outcome.get("exposure_complete") else 1)
    _require(len(ledger) == expected_length, "trace.outcome.goal_progress_ledger", "length differs from the completed/partial prefix")
    snapshots: List[Mapping[str, Any]] = []
    previous: Optional[List[bool]] = None
    for index, raw_snapshot in enumerate(ledger):
        snapshot, previous = _validate_goal_snapshot(
            raw_snapshot,
            previous,
            len(atoms),
            "trace.outcome.goal_progress_ledger[%d]" % index,
        )
        snapshots.append(snapshot)
    _require(snapshots[0].get("snapshot_kind") == "settled_pre_action", "trace initial goal snapshot kind")
    _require(snapshots[0].get("step") == -1, "trace initial goal snapshot step")
    for index in range(len(completed)):
        snapshot = snapshots[index + 1]
        _require(snapshot.get("snapshot_kind") == "completed_high_level_post_step", "trace completed goal snapshot kind")
        _require(snapshot.get("step") == index, "trace completed goal snapshot step")
    if not outcome.get("exposure_complete"):
        terminal = snapshots[-1]
        _require(terminal.get("snapshot_kind") == "partial_prefix_terminal_state_diagnostic", "trace partial goal snapshot kind")
        _require(terminal.get("step") == high_steps - 1, "trace partial goal snapshot step")

    task = _mapping(outcome.get("task"), "trace.outcome.task")
    initial = bool(snapshots[0].get("all_satisfied"))
    latched = any(bool(snapshot.get("all_satisfied")) for snapshot in snapshots)
    terminal_success = bool(snapshots[-1].get("all_satisfied"))
    first_success = None
    if not initial:
        for snapshot in snapshots[1:]:
            if snapshot.get("all_satisfied"):
                first_success = int(snapshot.get("step"))
                break
    maximum_fraction = max(float(snapshot.get("fraction")) for snapshot in snapshots)
    regressions = sum(len(_list(snapshot.get("regressed_indices"), "trace goal regressions")) for snapshot in snapshots[1:])
    _require(task.get("initial_task_success") is initial, "trace.outcome.task.initial_task_success")
    _require(task.get("prefix_task_success_latched") is latched, "trace.outcome.task.prefix_task_success_latched")
    _require(task.get("prefix_terminal_task_success") is terminal_success, "trace.outcome.task.prefix_terminal_task_success")
    _require(task.get("prefix_first_task_success_high_level_index") == first_success, "trace.outcome.task.prefix_first_task_success_high_level_index")
    _close(task.get("terminal_goal_fraction"), snapshots[-1].get("fraction"), "trace.outcome.task.terminal_goal_fraction")
    _close(task.get("maximum_goal_fraction"), maximum_fraction, "trace.outcome.task.maximum_goal_fraction")
    _require(task.get("goal_regression_count") == regressions, "trace.outcome.task.goal_regression_count")
    if outcome.get("exposure_complete"):
        _require(task.get("fixed_exposure_available") is True, "trace.outcome.task.fixed_exposure_available")
        _require(task.get("ever_task_success_within_registered_source_exposure") is latched, "trace.outcome.task.ever_task_success_within_registered_source_exposure")
        _require(task.get("terminal_task_success_within_registered_source_exposure") is terminal_success, "trace.outcome.task.terminal_task_success_within_registered_source_exposure")
        _require(task.get("first_task_success_high_level_index") == first_success, "trace.outcome.task.first_task_success_high_level_index")
    else:
        _require(task.get("fixed_exposure_available") is False, "trace.outcome.task.fixed_exposure_available")
        for field in (
            "ever_task_success_within_registered_source_exposure",
            "terminal_task_success_within_registered_source_exposure",
            "first_task_success_high_level_index",
        ):
            _require(task.get(field) is None, "trace.outcome.task.%s" % field)
    _require(
        _canonical_equal(_mapping(result.get("endpoints"), "result.endpoints").get("task"), task),
        "result.endpoints.task",
        "differs from the trace endpoint",
    )


def _contact_records(value: Any, label: str, *, physical_only: bool = False) -> List[Mapping[str, Any]]:
    output: List[Mapping[str, Any]] = []
    for index, raw_record in enumerate(_list(value, label)):
        record = _mapping(raw_record, "%s[%d]" % (label, index))
        flag = record.get("is_physical_nonpositive_distance_contact")
        _require(isinstance(flag, bool), "%s[%d].is_physical_nonpositive_distance_contact" % (label, index), "must be Boolean")
        if not physical_only or flag:
            output.append(record)
    return output


def _validate_contacts(
    result: Mapping[str, Any], outcome: Mapping[str, Any], physics_steps: int, trace: Mapping[str, Any]
) -> None:
    monitor = _mapping(outcome.get("monitor"), "trace.outcome.monitor")
    record = _mapping(monitor.get("record"), "trace.outcome.monitor.record")
    settled_record = _mapping(record.get("settled_state"), "trace.outcome.monitor.record.settled_state")
    settled = _contact_records(
        settled_record.get("physical_contact_point_records"),
        "trace settled physical contacts",
        physical_only=True,
    )
    for index, contact in enumerate(settled):
        _require(
            _finite(contact.get("contact_distance_m"), "trace settled contact distance")
            <= 0.0,
            "trace settled physical contact[%d]" % index,
        )
    if physics_steps:
        _require(_integer(record.get("observed_physics_substeps"), "trace contact observed substeps") == physics_steps, "trace contact observed substeps")
        live = _contact_records(
            record.get("live_solver_phase_contact_point_records"),
            "trace live-solver contact records",
            physical_only=True,
        )
        post = _contact_records(
            record.get("post_state_physical_contact_point_records"),
            "trace post-state physical contacts",
            physical_only=True,
        )
        for phase, contacts in (("live", live), ("post", post)):
            for index, contact in enumerate(contacts):
                _require(
                    _finite(contact.get("contact_distance_m"), "trace %s contact distance" % phase)
                    <= 0.0,
                    "trace %s physical contact[%d]" % (phase, index),
                )
        _require(record.get("total_physical_contact_point_record_count") == len(settled) + len(live) + len(post), "trace total physical contact count")
        _require(record.get("rollout_phase_physical_contact_point_record_count") == len(live) + len(post), "trace rollout physical contact count")
        _require(record.get("live_solver_nonpositive_contact_point_record_count") == len(live), "trace live-solver physical contact count")
        _require(record.get("post_state_physical_contact_point_record_count") == len(post), "trace post-state physical contact count")
        _require(record.get("any_robot_obstacle_contact") is bool(settled or live or post), "trace contact-any flag")
        _require(record.get("live_solver_any_robot_obstacle_contact") is bool(live), "trace live-solver contact-any flag")
        _require(record.get("post_state_any_robot_obstacle_contact") is bool(post), "trace post-state contact-any flag")
        _require(record.get("first_live_solver_physical_contact_point_record") == (live[0] if live else None), "trace first live-solver physical contact")
        _require(record.get("first_post_state_physical_contact_point_record") == (post[0] if post else None), "trace first post-state physical contact")
        sample_clearance = _mapping(
            record.get("sample_clearance"),
            "trace.outcome.monitor.record.sample_clearance",
        )
        coverage_lower_bound = _finite(
            sample_clearance.get("full_surface_clearance_lower_bound_m"),
            "trace contact coverage lower bound",
        )
        contact_distances = [
            _finite(contact.get("contact_distance_m"), "trace physical contact distance")
            for contact in settled + live + post
        ]
        recomputed_d_sim = (
            min([coverage_lower_bound, 0.0] + contact_distances)
            if contact_distances
            else coverage_lower_bound
        )
        _close(record.get("D_sim_min_m"), recomputed_d_sim, "trace contact D_sim recomputation")
        _close(monitor.get("D_sim_min_m"), recomputed_d_sim, "trace.outcome.monitor.D_sim_min_m")
        drift = _mapping(
            record.get("obstacle_pose_drift"),
            "trace.outcome.monitor.record.obstacle_pose_drift",
        )
        for monitor_field, record_field in (
            ("translation_drift_m", "maximum_translation_m"),
            ("rotation_drift_rad", "maximum_rotation_rad"),
            ("surface_drift_m", "maximum_surface_point_displacement_m"),
        ):
            _close(
                monitor.get(monitor_field),
                drift.get(record_field),
                "trace.outcome.monitor.%s" % monitor_field,
            )
    else:
        live = []
        post = []
        _require(record.get("observed_physics_substeps") == 0, "trace contact observed substeps")
        settled_clearance = _mapping(
            settled_record.get("sample_clearance"),
            "trace.outcome.monitor.record.settled_state.sample_clearance",
        )
        coverage_lower_bound = _finite(
            settled_clearance.get("full_surface_clearance_lower_bound_m"),
            "trace settled contact coverage lower bound",
        )
        settled_distances = [
            _finite(contact.get("contact_distance_m"), "trace settled contact distance")
            for contact in settled
        ]
        recomputed_d_sim = (
            min([coverage_lower_bound, 0.0] + settled_distances)
            if settled_distances
            else coverage_lower_bound
        )
        _close(settled_record.get("D_sim_m"), recomputed_d_sim, "trace settled D_sim recomputation")
        _close(monitor.get("D_sim_min_m"), recomputed_d_sim, "trace.outcome.monitor.D_sim_min_m")

    _require(settled_record.get("physical_contact_point_record_count") == len(settled), "trace settled physical contact count")
    _require(settled_record.get("any_robot_obstacle_contact") is bool(settled), "trace settled contact-any flag")
    _require(settled_record.get("first_physical_contact_point_record") == (settled[0] if settled else None), "trace first settled physical contact")
    total = len(settled) + len(live) + len(post)
    rollout = len(live) + len(post)
    _require(monitor.get("total") == total, "trace.outcome.monitor.total")
    _require(monitor.get("settled") == len(settled), "trace.outcome.monitor.settled")
    _require(monitor.get("rollout") == rollout, "trace.outcome.monitor.rollout")
    _require(monitor.get("live") == len(live), "trace.outcome.monitor.live")
    _require(monitor.get("post") == len(post), "trace.outcome.monitor.post")
    def rollout_order(contact: Mapping[str, Any]) -> Tuple[int, int, int]:
        observation = contact.get("observation_index")
        observation_index = -1 if observation is None else int(observation)
        phase_order = (
            0
            if contact.get("source_phase")
            == "live_solver_phase_preintegration_geometry"
            else 1
        )
        return (
            observation_index,
            phase_order,
            int(contact.get("mujoco_contact_index")),
        )

    rollout_records = sorted(live + post, key=rollout_order)
    all_records = settled + rollout_records
    _require(monitor.get("any_contact") is bool(all_records), "trace.outcome.monitor.any_contact")
    _require(monitor.get("first_settled") == (settled[0] if settled else None), "trace.outcome.monitor.first_settled")
    _require(monitor.get("first_live") == (live[0] if live else None), "trace.outcome.monitor.first_live")
    _require(monitor.get("first_post") == (post[0] if post else None), "trace.outcome.monitor.first_post")
    expected_first = all_records[0] if all_records else None
    _require(monitor.get("first") == expected_first, "trace.outcome.monitor.first")

    resolved = _mapping(trace.get("resolved_geometry"), "trace.resolved_geometry")
    link_ids = set(_integer(value, "trace.resolved_geometry.link56_geom_ids") for value in _list(resolved.get("link56_geom_ids"), "trace.resolved_geometry.link56_geom_ids"))
    link_contact = any(record_row.get("robot_geom_id") in link_ids for record_row in all_records)
    _require(monitor.get("link56_contact") is link_contact, "trace.outcome.monitor.link56_contact")
    settled_link = any(contact.get("robot_geom_id") in link_ids for contact in settled)
    live_link = any(contact.get("robot_geom_id") in link_ids for contact in live)
    post_link = any(contact.get("robot_geom_id") in link_ids for contact in post)
    _require(
        settled_record.get("link56_obstacle_contact") is settled_link,
        "trace settled link56 contact flag",
    )
    if physics_steps:
        _require(record.get("link56_obstacle_contact") is link_contact, "trace full link56 contact flag")
        _require(record.get("rollout_link56_obstacle_contact") is (live_link or post_link), "trace rollout link56 contact flag")
        _require(record.get("live_solver_link56_obstacle_contact") is live_link, "trace live link56 contact flag")
        _require(record.get("post_state_link56_obstacle_contact") is post_link, "trace post link56 contact flag")
        _require(record.get("first_physical_contact_point_record") == expected_first, "trace first physical contact")

    safety = _mapping(_mapping(result.get("endpoints"), "result.endpoints").get("safety"), "result.endpoints.safety")
    projections = {
        "any_robot_obstacle_contact": monitor.get("any_contact"),
        "link56_obstacle_contact": monitor.get("link56_contact"),
        "total_physical_contact_point_record_count": monitor.get("total"),
        "settled_physical_contact_point_record_count": monitor.get("settled"),
        "rollout_phase_physical_contact_point_record_count": monitor.get("rollout"),
        "live_solver_nonpositive_contact_point_record_count": monitor.get("live"),
        "post_state_physical_contact_point_record_count": monitor.get("post"),
        "first_physical_contact_point_record": monitor.get("first"),
        "first_settled_physical_contact_point_record": monitor.get("first_settled"),
        "first_live_solver_physical_contact_point_record": monitor.get("first_live"),
        "first_post_state_physical_contact_point_record": monitor.get("first_post"),
    }
    for field, value in projections.items():
        _require(_canonical_equal(safety.get(field), value), "result.endpoints.safety.%s" % field, "differs from contact trace")
    _check_measurement(safety.get("D_sim_min_m"), _finite(monitor.get("D_sim_min_m"), "trace D_sim"), "result.endpoints.safety.D_sim_min_m")


def _validate_optimizer_and_realized(
    result: Mapping[str, Any], outcome: Mapping[str, Any], physics_steps: int
) -> None:
    arm = outcome.get("arm")
    initial = _mapping(outcome.get("initial_protected_sample_audit"), "trace.outcome.initial_protected_sample_audit")
    sample_count = _integer(initial.get("protected_sample_count"), "trace initial protected sample count")
    query_count = _integer(initial.get("field_query_count"), "trace initial field query count")
    valid_count = _integer(initial.get("valid_field_query_count"), "trace initial valid field query count")
    _require(sample_count == query_count and 0 < sample_count and valid_count <= query_count, "trace initial sample/query counts")
    initial_h = initial.get("minimum_h_m2")
    if initial_h is not None:
        initial_h = _finite(initial_h, "trace initial h minimum")
    recomputed_safe_start = bool(valid_count == query_count and initial_h is not None and initial_h > 0.0)
    _require(initial.get("strict_safe_start") is recomputed_safe_start, "trace initial strict safe start")
    validity = _mapping(outcome.get("validity"), "trace.outcome.validity")
    _require(validity.get("poisson_safe_start") is recomputed_safe_start, "trace.outcome.validity.poisson_safe_start")

    inner_rows = [_mapping(row, "trace inner row") for row in _list(outcome.get("inner_trace"), "trace.outcome.inner_trace")]
    fail_rows = [_mapping(row, "trace fail row") for row in _list(outcome.get("fail_closed_attempt_trace"), "trace.outcome.fail_closed_attempt_trace")]
    realized_rows = [_mapping(row, "trace realized row") for row in _list(outcome.get("realized_cbf_trace"), "trace.outcome.realized_cbf_trace")]
    solved_rows = [row for row in inner_rows if isinstance(row.get("qp"), dict)]
    failed_optimizer_rows = [row for row in fail_rows if row.get("optimizer_outcome") in ("infeasible", "solver_failure", "postcheck_failure")]
    recomputed_counts = {
        "attempt_count": len(solved_rows) + len(failed_optimizer_rows),
        "solved_count": len(solved_rows),
        "infeasible_count": sum(row.get("optimizer_outcome") == "infeasible" for row in failed_optimizer_rows),
        "solver_failure_count": sum(row.get("optimizer_outcome") == "solver_failure" for row in failed_optimizer_rows),
        "postcheck_failure_count": sum(row.get("optimizer_outcome") == "postcheck_failure" for row in failed_optimizer_rows),
    }
    if arm == "joint_velocity_adapter_only":
        _require(not solved_rows and not failed_optimizer_rows, "adapter-only optimizer trace")
    counts = _mapping(outcome.get("optimizer_counts"), "trace.outcome.optimizer_counts")
    optimizer = _mapping(result.get("optimizer"), "result.optimizer")
    for field, expected in recomputed_counts.items():
        _require(counts.get(field) == expected, "trace.outcome.optimizer_counts.%s" % field)
        _require(optimizer.get(field) == expected, "result.optimizer.%s" % field)
    if arm == "joint_velocity_adapter_only":
        terminal_status = "not_applicable"
    elif recomputed_counts["postcheck_failure_count"]:
        terminal_status = "postcheck_failure"
    elif recomputed_counts["solver_failure_count"]:
        terminal_status = "solver_failure"
    elif recomputed_counts["infeasible_count"]:
        terminal_status = "infeasible"
    elif recomputed_counts["solved_count"]:
        terminal_status = "solved"
    else:
        terminal_status = "not_reached"
    _require(outcome.get("optimizer_terminal_status") == terminal_status, "trace.outcome.optimizer_terminal_status")
    _require(optimizer.get("terminal_status") == terminal_status, "result.optimizer.terminal_status")

    dopt = _optional_minimum(
        [initial.get("D_opt_min_m")]
        + [row.get("D_opt_min_m") for row in inner_rows]
        + [row.get("D_opt_min_m") for row in fail_rows]
        + [row.get("D_opt_min_m") for row in realized_rows],
        "trace D_opt minima",
    )
    h_minimum = _optional_minimum(
        [initial_h]
        + [row.get("minimum_h_m2") for row in inner_rows]
        + [row.get("minimum_h_m2") for row in fail_rows]
        + [row.get("minimum_h_m2") for row in realized_rows],
        "trace h minima",
    )
    nominal_minimum = _optional_minimum(
        [row.get("minimum_nominal_cbf_residual_m2_per_s") for row in solved_rows],
        "trace nominal CBF minima",
    )
    safe_minimum = _optional_minimum(
        [row.get("minimum_safe_cbf_residual_m2_per_s") for row in solved_rows],
        "trace safe CBF minima",
    )
    normalized_minimum = _optional_minimum(
        [_mapping(row.get("qp"), "trace QP diagnostics").get("minimum_normalized_cbf_residual") for row in solved_rows],
        "trace normalized CBF minima",
    )
    if arm == "joint_velocity_adapter_only":
        h_minimum = None
        nominal_minimum = None
        safe_minimum = None
        normalized_minimum = None
    for row in solved_rows:
        qp = _mapping(row.get("qp"), "trace QP diagnostics")
        _close(row.get("minimum_nominal_cbf_residual_m2_per_s"), qp.get("nominal_minimum_raw_cbf_residual_m2_per_s"), "trace QP nominal residual")
        _close(row.get("minimum_safe_cbf_residual_m2_per_s"), qp.get("minimum_raw_cbf_residual_m2_per_s"), "trace QP safe residual")

    valid_realized = [row for row in realized_rows if row.get("valid") is True]
    realized_minimum = _optional_minimum(
        [row.get("minimum_realized_cbf_residual_m2_per_s") for row in valid_realized],
        "trace realized CBF minima",
    )
    minima = _mapping(outcome.get("minimums"), "trace.outcome.minimums")
    expected_minima = {
        "h_m2": h_minimum,
        "D_opt_m": dopt,
        "nominal_raw_cbf_residual_m2_per_s": nominal_minimum,
        "safe_raw_cbf_residual_m2_per_s": safe_minimum,
        "safe_normalized_cbf_residual": normalized_minimum,
        "realized_raw_cbf_residual_m2_per_s": realized_minimum,
    }
    for field, expected in expected_minima.items():
        actual = minima.get(field)
        if expected is None:
            _require(actual is None, "trace.outcome.minimums.%s" % field)
        else:
            _close(actual, expected, "trace.outcome.minimums.%s" % field)
    if normalized_minimum is None:
        _require(optimizer.get("minimum_normalized_cbf_residual") is None, "result.optimizer.minimum_normalized_cbf_residual")
    else:
        _close(optimizer.get("minimum_normalized_cbf_residual"), normalized_minimum, "result.optimizer.minimum_normalized_cbf_residual")
    if safe_minimum is None:
        _require(optimizer.get("minimum_raw_cbf_residual_m2_per_s") is None, "result.optimizer.minimum_raw_cbf_residual_m2_per_s")
    else:
        _close(optimizer.get("minimum_raw_cbf_residual_m2_per_s"), safe_minimum, "result.optimizer.minimum_raw_cbf_residual_m2_per_s")

    if arm == "joint_velocity_psf_link56":
        _require(len(realized_rows) == physics_steps, "trace realized-CBF row count")
        for index, row in enumerate(realized_rows):
            _require(_row_coordinate(row, physics=True, label="trace realized row") == _coordinate_for_physics(index), "trace realized-CBF coordinates")
    else:
        _require(not realized_rows, "adapter-only realized-CBF trace")
    evaluations = sum(_integer(row.get("query_count"), "trace realized query count") for row in valid_realized)
    negative_rows = [row for row in valid_realized if _finite(row.get("minimum_realized_cbf_residual_m2_per_s"), "trace realized residual") < 0.0]
    first_negative = negative_rows[0].get("first_negative_crossing") if negative_rows else None
    audit = _mapping(outcome.get("realized_cbf_audit"), "trace.outcome.realized_cbf_audit")
    _require(audit.get("observed_physics_substep_count") == len(realized_rows), "trace realized audit observed count")
    _require(audit.get("residual_evaluation_count") == evaluations, "trace realized audit evaluation count")
    _require(audit.get("negative_residual_callback_count") == len(negative_rows), "trace realized audit negative count")
    _require(audit.get("first_negative_residual") == first_negative, "trace realized audit first negative")
    _require(validity.get("realized_cbf_observed_physics_substep_count") == len(realized_rows), "trace validity realized observed count")
    _require(validity.get("realized_cbf_residual_evaluation_count") == evaluations, "trace validity realized evaluation count")
    _require(validity.get("negative_realized_cbf_residual_count") == len(negative_rows), "trace validity realized negative count")
    _require(validity.get("first_negative_realized_cbf_residual") == first_negative, "trace validity realized first negative")
    _require(validity.get("qp_infeasible_count") == recomputed_counts["infeasible_count"], "trace validity QP infeasible count")
    _require(validity.get("qp_solver_failure_count") == recomputed_counts["solver_failure_count"], "trace validity QP solver count")
    _require(validity.get("qp_postcheck_failure_count") == recomputed_counts["postcheck_failure_count"], "trace validity QP postcheck count")
    invalid_queries = sum(_integer(row.get("invalid_query_count"), "trace invalid query count") for row in fail_rows + realized_rows if row.get("invalid_query_count") is not None)
    nonpositive_queries = sum(_integer(row.get("nonpositive_query_count"), "trace nonpositive query count") for row in fail_rows + realized_rows if row.get("nonpositive_query_count") is not None)
    _require(validity.get("invalid_field_query_count") == invalid_queries, "trace validity invalid-field count")
    _require(validity.get("nonpositive_runtime_field_query_count") == nonpositive_queries, "trace validity nonpositive-field count")

    safety = _mapping(_mapping(result.get("endpoints"), "result.endpoints").get("safety"), "result.endpoints.safety")
    for endpoint, expected in (
        ("h_min", h_minimum),
        ("minimum_nominal_cbf_residual", nominal_minimum),
        ("minimum_safe_cbf_residual", safe_minimum),
        ("minimum_realized_cbf_residual", realized_minimum),
        ("D_opt_min_m", dopt),
    ):
        _check_measurement(safety.get(endpoint), expected, "result.endpoints.safety.%s" % endpoint)
    _require(
        _canonical_equal(_mapping(result.get("endpoints"), "result.endpoints").get("validity"), validity),
        "result.endpoints.validity",
        "differs from the trace endpoint",
    )


def validate_active_arm_trace(result: Mapping[str, Any], trace: Mapping[str, Any]) -> None:
    """Independently reconstruct one compact result from its raw arm trace."""

    outcome = _validate_identity(result, trace)
    high_steps, _inner_steps, physics_steps, _entered, completed = _validate_execution(result, outcome)
    _validate_motion_and_tracking(result, outcome, physics_steps)
    _validate_car(result, outcome, completed)
    _validate_task(result, outcome, high_steps, completed)
    _validate_contacts(result, outcome, physics_steps, trace)
    _validate_optimizer_and_realized(result, outcome, physics_steps)


def validate_run_artifacts(
    result_path: Path, artifact_root: Optional[Path] = None
) -> Mapping[str, Any]:
    """Load and deeply validate a final result and its active-arm trace."""

    result = load_hashed_json(result_path)
    validate_episode_result(result)
    if artifact_root is not None:
        root = artifact_root.absolute()
    elif (
        result_path.parent.name == result.get("arm")
        and result_path.parent.parent.name == "arms"
    ):
        # The active runner publishes ``<run>/arms/<arm>/result.json`` while
        # artifact references are deliberately relative to the immutable run
        # root, not to the compact result's directory.
        root = result_path.parent.parent.parent.absolute()
    else:
        root = result_path.parent.absolute()
    references = _list(result.get("artifact_references"), "result.artifact_references")
    trace_candidates: List[Path] = []
    for index, raw_reference in enumerate(references):
        reference = _mapping(raw_reference, "result.artifact_references[%d]" % index)
        path = validate_artifact_reference(root, reference)
        if reference.get("artifact_type") == TRACE_ARTIFACT_TYPE:
            _require(reference.get("media_type") == "application/json", "active arm trace media type")
            trace_candidates.append(path)
    _require(
        len(trace_candidates) == 1,
        "result.artifact_references",
        "must contain exactly one active_arm_audit_trace",
    )
    trace = load_hashed_json(trace_candidates[0])
    validate_active_arm_trace(result, trace)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    parser.add_argument("--artifact-root", type=Path)
    arguments = parser.parse_args()
    try:
        result = validate_run_artifacts(arguments.result, arguments.artifact_root)
    except (ArtifactContractError, KeyError, TypeError, ValueError, OverflowError) as error:
        print("INVALID: %s" % error, file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": "valid",
                "case_id": result["case_id"],
                "arm": result["arm"],
                "completion_class": result["completion_class"],
                "result_payload_sha256": result["result_payload_sha256"],
                "active_arm_trace_deep_validation": "passed",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
