#!/usr/bin/env python3
"""Deeply validate Poisson arm traces or one complete paired active run.

The compact episode schema is necessary but not sufficient: a self-consistent
``result.json`` could otherwise disagree with the 2 ms audit ledger it cites.
This validator loads the referenced ``active_arm_audit_trace``, verifies both
layers of hashes, and reconstructs action, contact, motion, task, and optimizer
endpoints from serialized ledgers.  The coverage component of ``D_sim`` gets
strict typed/arithmetic/provenance consistency checks, but its exact
sample-to-OBB minimum remains a simulator-produced summary because v2 does not
serialize raw sample coordinates and obstacle OBB poses.
"""

import argparse
import json
import math
from pathlib import Path
import struct
import sys
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main.poisson_fullbody.contracts import (  # noqa: E402
    ArtifactContractError,
    canonical_json_bytes,
    load_hashed_json,
    sha256_file,
    sha256_bytes,
    validate_artifact_reference,
    verify_payload_hash,
)
from main.poisson_fullbody.result_schema import (  # noqa: E402
    validate_active_canary_pair,
    validate_episode_result,
)


TRACE_SCHEMA_VERSION = "vlsa_poisson_active_arm_trace.v2"
TRACE_ARTIFACT_TYPE = "active_arm_audit_trace"
RUN_RECEIPT_SCHEMA_VERSION = "vlsa_poisson_active_canary_run_receipt.v2"
ACTIVE_ARMS = (
    "joint_velocity_adapter_only",
    "joint_velocity_psf_link56",
)
RUN_RECEIPT_FIELDS = {
    "schema_version",
    "status",
    "scientific_result",
    "run_id",
    "case_id",
    "staged_scope",
    "identity",
    "arm_results",
    "pair_result_relative_path",
    "pair_result_sha256",
    "elapsed_seconds",
    "result_payload_sha256",
}
RUN_RECEIPT_IDENTITY_FIELDS = {
    "run_id",
    "case_id",
    "run_contract_sha256",
    "code_commit",
    "manifest_sha256",
    "manifest_record_sha256",
    "selection_config_sha256",
    "runtime_protocol_raw_sha256",
    "runtime_protocol_semantic_sha256",
    "runtime_parameter_block_sha256",
    "checkpoint_tree_sha256",
    "checkpoint_receipt_file_sha256",
    "historical_source_run_contract_sha256",
    "historical_result_file_sha256",
    "historical_result_payload_sha256",
    "source_action_ledger_sha256",
    "numeric_prerequisite_payload_sha256",
    "parity_prerequisite_payload_sha256",
    "identification_prerequisite_payload_sha256",
}
RUN_RECEIPT_ARM_RECORD_FIELDS = {"relative_path", "sha256"}
IDENTITY_FIELDS_NOT_PROJECTED_INSIDE_RUN = {
    "checkpoint_receipt_file_sha256",
    "historical_source_run_contract_sha256",
    "numeric_prerequisite_payload_sha256",
    "parity_prerequisite_payload_sha256",
    "identification_prerequisite_payload_sha256",
}
EXTERNAL_ARTIFACTS_REQUIRED_FOR_INDEPENDENT_ACCEPTANCE = (
    "run_contract_payload",
    "manifest_and_selected_record",
    "selection_configuration",
    "runtime_protocol",
    "checkpoint_tree_and_receipt",
    "historical_source_run_contract",
    "historical_result",
    "numeric_prerequisite",
    "exact_parity_prerequisite",
    "shadow_identification_prerequisite",
)
SOURCE_REPLAY_FIELDS = {
    "case_id",
    "source_arm",
    "action_count",
    "historical_result_payload_sha256",
    "historical_result_file_sha256",
    "executed_sequence_sha256",
    "source_policy_query_count",
    "source_policy_query_schedule_sha256",
    "settled_simulator_state_sha256",
    "initial_observation_sha256",
    "settled_active_obstacle_position_sha256",
    "policy_noise_schedule_sha256",
    "historical_task_success",
    "historical_car_collision",
    "historical_collision_first_step",
    "terminal_simulator_state_sha256",
    "replay_semantics",
}
FIELD_BUNDLE_HASH_FIELDS = {
    "protocol_sha256",
    "parameter_block_sha256",
    "obstacle_geometry_sha256",
    "occupancy_sha256",
    "domain_sha256",
    "system_sha256",
    "field_sha256",
    "protected_samples_sha256",
    "bundle_sha256",
}
RESOLVED_GEOMETRY_FIELDS = {
    "robot_root_body_ids",
    "robot_body_ids",
    "obstacle_root_body_ids",
    "obstacle_body_ids",
    "link56_body_ids",
    "robot_geom_ids",
    "obstacle_geom_ids",
    "link56_geom_ids",
    "collision_enabled_pairs",
    "robot_body_names",
    "obstacle_body_names",
    "robot_geom_names",
    "obstacle_geom_names",
    "link56_geom_names",
}
RESTORE_FIELDS = {
    "schema_version",
    "source_controller",
    "target_controller",
    "model_topology_sha256",
    "physical_model_sha256",
    "compiled_mjb_sha256",
    "official_integration_state_available",
    "official_integration_state_sha256",
    "target_official_integration_state_sha256",
    "settled_state_sha256",
    "target_state_sha256",
    "exact_flattened_state",
    "maximum_arm_qpos_error_rad",
    "maximum_arm_qvel_error_rad_s",
    "max_arm_qpos_error_tolerance_rad",
    "max_arm_qvel_error_tolerance_rad_s",
    "copied_timestep",
    "copied_cur_time_s",
    "copied_done",
    "controller",
    "controller_software_state",
    "pid_memory_reset",
}
PID_MEMORY_RESET_FIELDS = {
    "goal_velocity_zero",
    "current_velocity_zero",
    "last_error_zero",
    "summed_error_zero",
    "derivative_buffer_size",
    "saturated",
}
FIELD_BUNDLE_DIAGNOSTIC_FIELDS = {
    "obstacle_geom_count",
    "protected_surface_component_count",
    "protected_sample_count",
    "raw_occupied_cell_count",
    "buffered_occupied_cell_count",
    "connected_free_cell_count",
    "active_vertex_count",
    "boundary_vertex_count",
    "interior_vertex_count",
    "poisson_method",
    "poisson_iterations",
    "poisson",
    "minimum_initial_h_m2",
    "minimum_outer_boundary_clearance_m",
    "required_outer_boundary_clearance_m",
}
POISSON_DIAGNOSTIC_FIELDS = {
    "finite",
    "unknown_count",
    "residual_linf",
    "residual_l2",
    "relative_residual_linf",
    "relative_residual_l2",
    "backward_error_linf",
    "boundary_linf",
    "interior_min",
    "interior_max",
    "expected_sign",
    "sign_violation_count",
    "sign_violation_linf",
    "passed",
}
SETTLED_MOTION_FIELDS = {
    "reference",
    "max_linear_speed_threshold_m_per_s",
    "max_angular_speed_threshold_rad_per_s",
    "maximum_observed_linear_speed_m_per_s",
    "maximum_observed_angular_speed_rad_per_s",
    "admissible",
    "reasons",
    "bodies",
}
SETTLED_BODY_VELOCITY_FIELDS = {
    "body_id",
    "body_name",
    "angular_velocity_world_rad_per_s",
    "linear_velocity_world_m_per_s",
    "angular_speed_rad_per_s",
    "linear_speed_m_per_s",
    "angular_speed_exceeds_threshold",
    "linear_speed_exceeds_threshold",
}
OBSTACLE_DRIFT_FIELDS = {
    "reference",
    "maximum_translation_m",
    "maximum_rotation_rad",
    "maximum_surface_point_displacement_m",
    "surface_drift_threshold_m",
    "surface_drift_threshold_crossed",
    "first_surface_drift_threshold_crossing_observation_index",
    "geoms",
}
OBSTACLE_GEOM_DRIFT_FIELDS = {
    "geom_id",
    "geom_name",
    "body_id",
    "body_name",
    "maximum_translation_m",
    "maximum_rotation_rad",
    "maximum_surface_point_displacement_m",
    "final_translation_m",
    "final_rotation_rad",
    "final_surface_point_displacement_m",
    "maximum_translation_observation_index",
    "maximum_rotation_observation_index",
    "maximum_surface_displacement_observation_index",
}
EEF_REFERENCE_SITE_FIELDS = {
    "site_id",
    "site_name",
    "jacobian_row_order",
    "shape",
    "linear_finite_difference_delta_rad",
    "maximum_linear_jacobian_error_m_per_rad",
    "absolute_tolerance_m_per_rad",
    "passed",
    "live_state_mutated",
}
ADAPTER_STEP_FIELDS = {
    "normalized_action",
    "qdot_physical_rad_s",
    "qdot_unclipped_rad_s",
    "desired_twist",
    "achieved_twist",
    "diagnostics",
}
EXECUTION_RECORD_FIELDS = {
    "nominal_qdot_physical_rad_s",
    "executed_qdot_physical_rad_s",
    "filter_correction_rad_s",
    "filter_correction_l2_rad_s",
    "normalized_executed_action",
}
SOLVED_QP_DIAGNOSTIC_FIELDS = {
    "solver",
    "status",
    "status_value",
    "iterations",
    "solve_time_seconds",
    "input_constraint_count",
    "solved_constraint_count",
    "trivial_zero_constraint_count",
    "minimum_normalized_cbf_residual",
    "minimum_raw_cbf_residual_m2_per_s",
    "minimum_nonzero_row_scale_m2_per_rad",
    "maximum_nonzero_row_scale_m2_per_rad",
    "maximum_velocity_bound_violation_rad_s",
    "nominal_minimum_normalized_cbf_residual",
    "nominal_minimum_raw_cbf_residual_m2_per_s",
    "correction_l2_rad_s",
    "nominal_feasible",
}
PHYSICS_DT_SECONDS = 0.002
INNER_UPDATES_PER_HIGH_LEVEL = 5
PHYSICS_SUBSTEPS_PER_INNER = 5
PAPER_CAR_THRESHOLD_M = 0.001
SETTLED_LINEAR_SPEED_THRESHOLD_M_PER_S = 1e-4
SETTLED_ANGULAR_SPEED_THRESHOLD_RAD_PER_S = 1e-4
OBSTACLE_TRANSLATION_DRIFT_THRESHOLD_M = 1e-6
OBSTACLE_ROTATION_DRIFT_THRESHOLD_RAD = 1e-5
OBSTACLE_SURFACE_DRIFT_THRESHOLD_M = 1e-6
POISSON_MAX_ITERATIONS = 50000
POISSON_BACKWARD_ERROR_TOLERANCE = 1e-8
POISSON_BOUNDARY_TOLERANCE = 1e-12
REQUIRED_OUTER_BOUNDARY_CLEARANCE_M = 0.051
JOINT_VELOCITY_LIMIT_RAD_S = 0.5
REGISTERED_SOURCE_ACTION_COUNT = 237
JOINT_VELOCITY_QP_MAX_ITERATIONS = 10000
JOINT_VELOCITY_QP_POSTCHECK_TOLERANCE = 5e-7
JOINT_VELOCITY_BOUND_POSTCHECK_TOLERANCE_RAD_S = 5e-8
VELOCITY_TRACKING_LINF_THRESHOLD_RAD_S = 0.05
VELOCITY_TRACKING_RMSE_THRESHOLD_RAD_S = 0.02
RESTORE_ARM_STATE_TOLERANCE = 1e-10
REGISTERED_EEF_SITE_NAME = "gripper0_grip_site"
REGISTERED_CANARY_CASE_ID = "vlsa-t1-goal-ii-t0-e05"
REGISTERED_CANARY_GOAL_ATOMS = [
    {
        "index": 0,
        "predicate": "on",
        "arguments": ["akita_black_bowl_1", "plate_1"],
    }
]
D_SIM_COVERAGE_VALIDATION = (
    "typed_summary_consistency_only_raw_sample_and_obb_geometry_not_serialized"
)
FULL_ROBOT_COVERAGE_SEMANTICS = (
    "strict_open_ball_surface_cover_from_triangle_lattices_for_compiled_"
    "convex_hulls_and_exact_boxes_or_analytic_parameter_grids_for_exact_"
    "cylinders; MuJoCo collision-semantic equivalence requires allocation audit"
)


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


def _signed_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(label, "must be an integer")
    return int(value)


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        _fail(label, "must be Boolean")
    return bool(value)


def _exact_keys(value: Mapping[str, Any], expected: set, label: str) -> None:
    _require(set(value) == expected, label, "has missing or unexpected fields")


def _sha256_string(value: Any, label: str) -> str:
    _require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        label,
        "must be a lowercase SHA-256 digest",
    )
    return value


def _safe_path_component(value: Any, label: str) -> str:
    _require(
        isinstance(value, str)
        and bool(value)
        and value not in (".", "..")
        and Path(value).name == value,
        label,
        "must be one nonempty path component",
    )
    return value


def _real_child_path(root: Path, relative: str, label: str) -> Path:
    relative_path = Path(relative)
    _require(
        not relative_path.is_absolute()
        and bool(relative_path.parts)
        and all(part not in ("", ".", "..") for part in relative_path.parts),
        label,
        "is not a canonical relative path",
    )
    candidate = root
    for component in relative_path.parts:
        candidate = candidate / component
        _require(not candidate.is_symlink(), label, "traverses a symbolic link")
    _require(candidate.is_file(), label, "does not resolve a regular file")
    return candidate


def _finite(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        _fail(label, "must be a finite JSON number")
    return float(value)


def _finite_number(value: Any, label: str) -> float:
    """Require a JSON number, not a string that merely parses as one."""

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        _fail(label, "must be a finite JSON number")
    return float(value)


def _finite_number_vector(value: Any, length: int, label: str) -> List[float]:
    items = _list(value, label)
    _require(len(items) == length, label, "has the wrong length")
    return [
        _finite_number(item, "%s[%d]" % (label, index))
        for index, item in enumerate(items)
    ]


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


def _producer_configuration_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return sha256_bytes(encoded)


def _producer_float64_vector_sha256(value: Sequence[float]) -> str:
    """Reproduce ``evaluate_safelibero_aegis.array_sha256`` for float64 vectors."""

    header = canonical_json_bytes({"dtype": "<f8", "shape": [len(value)]})
    data = struct.pack("<%dd" % len(value), *[float(item) for item in value])
    return sha256_bytes(b"vlsa-table1-array-v1\0" + header + b"\0" + data)


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


def _validate_trace_configuration_bindings(
    result: Mapping[str, Any],
    trace: Mapping[str, Any],
    *,
    expected_source_action_count: int,
) -> None:
    _require(
        trace.get("staged_scope") == "first_two_arm_canary_only",
        "trace.staged_scope",
    )
    runtime = _mapping(result.get("runtime"), "result.runtime")
    measurement = _mapping(runtime.get("measurement"), "result.runtime.measurement")
    resolved = _mapping(trace.get("resolved_geometry"), "trace.resolved_geometry")
    expected_measurement_configuration = {
        "resolved_geometry": resolved,
        "D_sim_semantics": measurement.get("D_sim_semantics"),
    }
    _require(
        measurement.get("configuration_sha256")
        == _producer_configuration_sha256(expected_measurement_configuration),
        "result.runtime.measurement.configuration_sha256",
        "does not bind the serialized resolved geometry",
    )

    source_replay = _mapping(trace.get("source_replay"), "trace.source_replay")
    _exact_keys(source_replay, SOURCE_REPLAY_FIELDS, "trace.source_replay")
    sampler = _mapping(runtime.get("sampler"), "result.runtime.sampler")
    expected_sampler_configuration = {
        "source": source_replay,
        "active_policy_queries": 0,
        "active_policy_rng_exercised": False,
    }
    _require(
        sampler.get("configuration_sha256")
        == _producer_configuration_sha256(expected_sampler_configuration),
        "result.runtime.sampler.configuration_sha256",
        "does not bind the serialized historical replay provenance",
    )
    pairing = _mapping(result.get("pairing"), "result.pairing")
    _require(
        source_replay.get("case_id") == result.get("case_id")
        and source_replay.get("source_arm") == "pi05_plus_aegis_translational"
        and source_replay.get("replay_semantics")
        == "exact_actions[*].executed_not_nominal_raw",
        "trace.source_replay identity",
    )
    for field in (
        "historical_task_success",
        "historical_car_collision",
    ):
        _boolean(source_replay.get(field), "trace.source_replay.%s" % field)
    _require(
        source_replay.get("historical_task_success") is True,
        "trace.source_replay.historical_task_success",
        "must preserve the registered successful historical canary exposure",
    )
    first_collision = source_replay.get("historical_collision_first_step")
    if first_collision is not None:
        _integer(
            first_collision,
            "trace.source_replay.historical_collision_first_step",
        )
    action_count = _integer(
        source_replay.get("action_count"), "trace.source_replay.action_count"
    )
    _require(
        action_count == expected_source_action_count,
        "trace.source_replay.action_count",
        "differs from the registered source exposure",
    )
    query_count = _integer(
        source_replay.get("source_policy_query_count"),
        "trace.source_replay.source_policy_query_count",
    )
    replay_digest_fields = SOURCE_REPLAY_FIELDS - {
        "case_id",
        "source_arm",
        "action_count",
        "source_policy_query_count",
        "historical_task_success",
        "historical_car_collision",
        "historical_collision_first_step",
        "replay_semantics",
    }
    for field in sorted(replay_digest_fields):
        _sha256_string(
            source_replay.get(field), "trace.source_replay.%s" % field
        )
    replay_projections = {
        "initial_observation_sha256": pairing.get(
            "source_settled_observation_sha256"
        ),
        "policy_noise_schedule_sha256": pairing.get(
            "policy_noise_schedule_sha256"
        ),
        "source_policy_query_schedule_sha256": pairing.get(
            "policy_query_schedule_sha256"
        ),
        "executed_sequence_sha256": pairing.get(
            "nominal_high_level_action_ledger_sha256"
        ),
    }
    for field, expected in replay_projections.items():
        _require(
            source_replay.get(field) == expected,
            "trace.source_replay.%s" % field,
            "differs from the compact result pairing identity",
        )
    outcome = _mapping(trace.get("outcome"), "trace.outcome")
    restore = _mapping(outcome.get("restore"), "trace.outcome.restore")
    _exact_keys(restore, RESTORE_FIELDS, "trace.outcome.restore")
    _require(
        restore.get("schema_version")
        == "vlsa_poisson_controller_state_restore.v1"
        and restore.get("source_controller") == "OSC_POSE"
        and restore.get("target_controller") == "JOINT_VELOCITY",
        "trace.outcome.restore controller identity",
    )
    _sha256_string(
        restore.get("model_topology_sha256"),
        "trace.outcome.restore.model_topology_sha256",
    )
    _require(
        restore.get("official_integration_state_available") is True,
        "trace.outcome.restore.official_integration_state_available",
    )
    _require(
        source_replay.get("settled_simulator_state_sha256")
        == restore.get("settled_state_sha256"),
        "trace.source_replay.settled_simulator_state_sha256",
        "differs from the restored historical flattened-state identity",
    )
    _require(
        pairing.get("restored_settled_state_sha256")
        == restore.get("official_integration_state_sha256"),
        "result.pairing.restored_settled_state_sha256",
        "differs from the restored official MuJoCo integration-state identity",
    )
    car_position_ledger = _mapping(
        outcome.get("paper_car_position_ledger"),
        "trace.outcome.paper_car_position_ledger",
    )
    _require(
        source_replay.get("settled_active_obstacle_position_sha256")
        == car_position_ledger.get(
            "historical_settled_active_obstacle_position_sha256"
        ),
        "trace.source_replay.settled_active_obstacle_position_sha256",
        "differs from the historical settled obstacle state used by Paper CAR",
    )
    _sha256_string(
        restore.get("settled_state_sha256"),
        "trace.outcome.restore.settled_state_sha256",
    )
    _sha256_string(
        restore.get("official_integration_state_sha256"),
        "trace.outcome.restore.official_integration_state_sha256",
    )
    target_official_state_sha256 = _sha256_string(
        restore.get("target_official_integration_state_sha256"),
        "trace.outcome.restore.target_official_integration_state_sha256",
    )
    _require(
        target_official_state_sha256
        == restore.get("official_integration_state_sha256"),
        "trace.outcome.restore.target_official_integration_state_sha256",
        "differs from the source official MuJoCo integration state",
    )
    target_flattened_state_sha256 = _sha256_string(
        restore.get("target_state_sha256"),
        "trace.outcome.restore.target_state_sha256",
    )
    _require(
        target_flattened_state_sha256 == restore.get("settled_state_sha256")
        and restore.get("exact_flattened_state") is True,
        "trace.outcome.restore flattened state",
        "was not restored exactly",
    )
    qpos_error = _finite_number(
        restore.get("maximum_arm_qpos_error_rad"),
        "trace.outcome.restore.maximum_arm_qpos_error_rad",
    )
    qvel_error = _finite_number(
        restore.get("maximum_arm_qvel_error_rad_s"),
        "trace.outcome.restore.maximum_arm_qvel_error_rad_s",
    )
    qpos_tolerance = _finite_number(
        restore.get("max_arm_qpos_error_tolerance_rad"),
        "trace.outcome.restore.max_arm_qpos_error_tolerance_rad",
    )
    qvel_tolerance = _finite_number(
        restore.get("max_arm_qvel_error_tolerance_rad_s"),
        "trace.outcome.restore.max_arm_qvel_error_tolerance_rad_s",
    )
    _close(
        qpos_tolerance,
        RESTORE_ARM_STATE_TOLERANCE,
        "trace.outcome.restore.max_arm_qpos_error_tolerance_rad",
    )
    _close(
        qvel_tolerance,
        RESTORE_ARM_STATE_TOLERANCE,
        "trace.outcome.restore.max_arm_qvel_error_tolerance_rad_s",
    )
    _require(
        0.0 <= qpos_error <= qpos_tolerance and qpos_tolerance > 0.0,
        "trace.outcome.restore arm qpos error",
        "crosses its registered tolerance",
    )
    _require(
        0.0 <= qvel_error <= qvel_tolerance and qvel_tolerance > 0.0,
        "trace.outcome.restore arm qvel error",
        "crosses its registered tolerance",
    )
    copied_timestep = _integer(
        restore.get("copied_timestep"), "trace.outcome.restore.copied_timestep"
    )
    copied_time = _finite_number(
        restore.get("copied_cur_time_s"),
        "trace.outcome.restore.copied_cur_time_s",
    )
    _require(
        copied_timestep >= 0 and copied_time >= 0.0,
        "trace.outcome.restore copied simulator clock",
    )
    _require(
        _boolean(restore.get("copied_done"), "trace.outcome.restore.copied_done")
        is False,
        "trace.outcome.restore.copied_done",
        "must be false for the executable restored state",
    )
    pid_reset = _mapping(
        restore.get("pid_memory_reset"), "trace.outcome.restore.pid_memory_reset"
    )
    _exact_keys(
        pid_reset,
        PID_MEMORY_RESET_FIELDS,
        "trace.outcome.restore.pid_memory_reset",
    )
    for field in (
        "goal_velocity_zero",
        "current_velocity_zero",
        "last_error_zero",
        "summed_error_zero",
    ):
        _require(
            pid_reset.get(field) is True,
            "trace.outcome.restore.pid_memory_reset.%s" % field,
        )
    _require(
        _integer(
            pid_reset.get("derivative_buffer_size"),
            "trace.outcome.restore.pid_memory_reset.derivative_buffer_size",
        )
        == 0
        and pid_reset.get("saturated") is False,
        "trace.outcome.restore.pid_memory_reset",
        "does not describe a clean controller reset",
    )
    selected_initial_state_sha256 = _sha256_string(
        outcome.get("selected_initial_state_sha256"),
        "trace.outcome.selected_initial_state_sha256",
    )
    active_initial_observation_sha256 = _sha256_string(
        outcome.get("active_initial_observation_sha256"),
        "trace.outcome.active_initial_observation_sha256",
    )
    case_identity = _mapping(result.get("case_identity"), "result.case_identity")
    _require(
        case_identity.get("registered_source_exposure_high_level_steps")
        == expected_source_action_count,
        "result.case_identity.registered_source_exposure_high_level_steps",
        "differs from the registered source exposure",
    )
    active_obstacle_name = case_identity.get("active_obstacle_name")
    _require(
        isinstance(active_obstacle_name, str) and bool(active_obstacle_name),
        "result.case_identity.active_obstacle_name",
    )
    obstacle_body_ids = _list(
        resolved.get("obstacle_body_ids"),
        "trace.resolved_geometry.obstacle_body_ids",
    )
    obstacle_body_names = _list(
        resolved.get("obstacle_body_names"),
        "trace.resolved_geometry.obstacle_body_names",
    )
    obstacle_roots = _list(
        resolved.get("obstacle_root_body_ids"),
        "trace.resolved_geometry.obstacle_root_body_ids",
    )
    _require(
        len(obstacle_body_ids) == len(obstacle_body_names)
        and len(obstacle_roots) == 1
        and obstacle_roots[0] in obstacle_body_ids,
        "trace resolved selected-obstacle body identity",
    )
    root_name = obstacle_body_names[obstacle_body_ids.index(obstacle_roots[0])]
    _require(
        root_name == active_obstacle_name + "_main"
        and all(
            isinstance(name, str) and name.startswith(active_obstacle_name + "_")
            for name in obstacle_body_names
        )
        and all(
            isinstance(name, str) and name.startswith(active_obstacle_name + "_")
            for name in _list(
                resolved.get("obstacle_geom_names"),
                "trace.resolved_geometry.obstacle_geom_names",
            )
        ),
        "trace resolved selected-obstacle names",
        "do not match the manifest active obstacle",
    )
    _require(
        case_identity.get("initial_state_record_sha256")
        == selected_initial_state_sha256,
        "result.case_identity.initial_state_record_sha256",
        "differs from the selected simulator initial-state record",
    )
    _require(
        pairing.get("active_joint_velocity_initial_observation_sha256")
        == active_initial_observation_sha256,
        "result.pairing.active_joint_velocity_initial_observation_sha256",
        "differs from the restored joint-velocity observation",
    )

    for result_field, restore_field in (
        ("compiled_physical_model_sha256", "physical_model_sha256"),
        ("compiled_mjb_sha256", "compiled_mjb_sha256"),
    ):
        restored_digest = _sha256_string(
            restore.get(restore_field),
            "trace.outcome.restore.%s" % restore_field,
        )
        _require(
            pairing.get(result_field) == restored_digest,
            "result.pairing.%s" % result_field,
            "differs from the restored compiled-model authority",
        )

    controller_state = _mapping(
        restore.get("controller_software_state"),
        "trace.outcome.restore.controller_software_state",
    )
    _exact_keys(
        controller_state,
        {"schema_version", "fields", "sha256"},
        "trace.outcome.restore.controller_software_state",
    )
    _require(
        controller_state.get("schema_version")
        == "vlsa_poisson_joint_velocity_controller_software_state.v1",
        "trace.outcome.restore.controller_software_state.schema_version",
    )
    _mapping(
        controller_state.get("fields"),
        "trace.outcome.restore.controller_software_state.fields",
    )
    controller_digest = _sha256_string(
        controller_state.get("sha256"),
        "trace.outcome.restore.controller_software_state.sha256",
    )
    controller_payload = {
        key: value for key, value in controller_state.items() if key != "sha256"
    }
    _require(
        controller_digest == _producer_configuration_sha256(controller_payload),
        "trace.outcome.restore.controller_software_state.sha256",
        "does not bind the serialized controller state",
    )
    _require(
        pairing.get("controller_initial_state_sha256") == controller_digest,
        "result.pairing.controller_initial_state_sha256",
        "differs from the restored controller software state",
    )
    _require(
        pairing.get("settling_action_ledger_sha256")
        == _producer_configuration_sha256([[0.0] * 7] * 20),
        "result.pairing.settling_action_ledger_sha256",
        "differs from the registered twenty zero-action settling ledger",
    )

    controller_contract = _mapping(
        restore.get("controller"), "trace.outcome.restore.controller"
    )
    controller = _mapping(runtime.get("controller"), "result.runtime.controller")
    _close(
        controller.get("physical_velocity_limit_rad_s"),
        JOINT_VELOCITY_LIMIT_RAD_S,
        "result.runtime.controller.physical_velocity_limit_rad_s",
    )
    _require(
        controller.get("configuration_sha256")
        == _producer_configuration_sha256(controller_contract),
        "result.runtime.controller.configuration_sha256",
        "does not bind the restored controller contract",
    )
    _require(
        action_count == pairing.get("source_exposure_high_level_steps"),
        "trace.source_replay.action_count",
        "differs from the compact result source exposure",
    )
    _require(
        query_count == sampler.get("source_policy_query_count"),
        "trace.source_replay.source_policy_query_count",
        "differs from the compact result sampler",
    )
    field_hashes = _mapping(
        trace.get("field_bundle_hashes"), "trace.field_bundle_hashes"
    )
    field_diagnostics = _mapping(
        trace.get("field_bundle_diagnostics"), "trace.field_bundle_diagnostics"
    )
    _exact_keys(
        field_hashes, FIELD_BUNDLE_HASH_FIELDS, "trace.field_bundle_hashes"
    )
    for field in sorted(FIELD_BUNDLE_HASH_FIELDS):
        _sha256_string(
            field_hashes.get(field), "trace.field_bundle_hashes.%s" % field
        )
    provenance = _mapping(result.get("provenance"), "result.provenance")
    _require(
        field_hashes.get("field_sha256") == pairing.get("field_sha256"),
        "trace.field_bundle_hashes.field_sha256",
        "differs from the compact result pairing identity",
    )
    _require(
        field_hashes.get("protocol_sha256")
        == provenance.get("runtime_protocol_semantic_sha256"),
        "trace.field_bundle_hashes.protocol_sha256",
        "differs from the compact result protocol identity",
    )
    _require(
        field_hashes.get("parameter_block_sha256")
        == runtime.get("protocol_parameter_block_sha256"),
        "trace.field_bundle_hashes.parameter_block_sha256",
        "differs from the compact result runtime parameters",
    )
    bundle_identity = {
        field: field_hashes[field]
        for field in FIELD_BUNDLE_HASH_FIELDS
        if field != "bundle_sha256"
    }
    bundle_identity["diagnostics"] = field_diagnostics
    _require(
        field_hashes.get("bundle_sha256")
        == sha256_bytes(canonical_json_bytes(bundle_identity)),
        "trace.field_bundle_hashes.bundle_sha256",
        "does not bind the component hashes and field diagnostics",
    )
    _exact_keys(
        field_diagnostics,
        FIELD_BUNDLE_DIAGNOSTIC_FIELDS,
        "trace.field_bundle_diagnostics",
    )
    resolved_obstacle_geoms = _list(
        resolved.get("obstacle_geom_ids"),
        "trace.resolved_geometry.obstacle_geom_ids",
    )
    resolved_link_geoms = _list(
        resolved.get("link56_geom_ids"),
        "trace.resolved_geometry.link56_geom_ids",
    )
    initial_audit = _mapping(
        outcome.get("initial_protected_sample_audit"),
        "trace.outcome.initial_protected_sample_audit",
    )
    protected_sample_count = _integer(
        initial_audit.get("protected_sample_count"),
        "trace.outcome.initial_protected_sample_audit.protected_sample_count",
    )
    diagnostic_count_bindings = {
        "obstacle_geom_count": len(resolved_obstacle_geoms),
        "protected_surface_component_count": len(resolved_link_geoms),
        "protected_sample_count": protected_sample_count,
    }
    for field, expected in diagnostic_count_bindings.items():
        _require(
            _integer(
                field_diagnostics.get(field),
                "trace.field_bundle_diagnostics.%s" % field,
            )
            == expected,
            "trace.field_bundle_diagnostics.%s" % field,
        )
    count_fields = (
        "raw_occupied_cell_count",
        "buffered_occupied_cell_count",
        "connected_free_cell_count",
        "active_vertex_count",
        "boundary_vertex_count",
        "interior_vertex_count",
    )
    counts = {
        field: _integer(
            field_diagnostics.get(field),
            "trace.field_bundle_diagnostics.%s" % field,
        )
        for field in count_fields
    }
    _require(
        counts["raw_occupied_cell_count"] > 0
        and counts["buffered_occupied_cell_count"]
        >= counts["raw_occupied_cell_count"]
        and counts["connected_free_cell_count"] > 0
        and counts["active_vertex_count"]
        == counts["boundary_vertex_count"] + counts["interior_vertex_count"]
        and counts["boundary_vertex_count"] > 0
        and counts["interior_vertex_count"] > 0,
        "trace.field_bundle_diagnostics grid counts",
    )
    _require(
        field_diagnostics.get("poisson_method") == "red_black_sor",
        "trace.field_bundle_diagnostics.poisson_method",
    )
    poisson_iterations = _integer(
        field_diagnostics.get("poisson_iterations"),
        "trace.field_bundle_diagnostics.poisson_iterations",
    )
    _require(
        1 <= poisson_iterations <= POISSON_MAX_ITERATIONS,
        "trace.field_bundle_diagnostics.poisson_iterations",
    )
    minimum_initial_h = _finite_number(
        field_diagnostics.get("minimum_initial_h_m2"),
        "trace.field_bundle_diagnostics.minimum_initial_h_m2",
    )
    _require(
        minimum_initial_h > 0.0,
        "trace.field_bundle_diagnostics.minimum_initial_h_m2",
    )
    _close(
        minimum_initial_h,
        initial_audit.get("minimum_h_m2"),
        "trace.field_bundle_diagnostics.minimum_initial_h_m2",
    )
    minimum_outer = _finite_number(
        field_diagnostics.get("minimum_outer_boundary_clearance_m"),
        "trace.field_bundle_diagnostics.minimum_outer_boundary_clearance_m",
    )
    required_outer = _finite_number(
        field_diagnostics.get("required_outer_boundary_clearance_m"),
        "trace.field_bundle_diagnostics.required_outer_boundary_clearance_m",
    )
    _require(
        required_outer > 0.0 and minimum_outer + 1e-12 >= required_outer,
        "trace.field_bundle_diagnostics outer-boundary clearance",
    )
    _close(
        required_outer,
        REQUIRED_OUTER_BOUNDARY_CLEARANCE_M,
        "trace.field_bundle_diagnostics.required_outer_boundary_clearance_m",
    )
    poisson = _mapping(
        field_diagnostics.get("poisson"),
        "trace.field_bundle_diagnostics.poisson",
    )
    _exact_keys(
        poisson,
        POISSON_DIAGNOSTIC_FIELDS,
        "trace.field_bundle_diagnostics.poisson",
    )
    _require(
        poisson.get("finite") is True
        and poisson.get("passed") is True
        and poisson.get("expected_sign") == "nonnegative",
        "trace.field_bundle_diagnostics.poisson acceptance",
    )
    _require(
        _integer(
            poisson.get("unknown_count"),
            "trace.field_bundle_diagnostics.poisson.unknown_count",
        )
        == counts["interior_vertex_count"],
        "trace.field_bundle_diagnostics.poisson.unknown_count",
    )
    for field in (
        "residual_linf",
        "residual_l2",
        "relative_residual_linf",
        "relative_residual_l2",
        "backward_error_linf",
        "boundary_linf",
        "sign_violation_linf",
    ):
        _require(
            _finite_number(
                poisson.get(field),
                "trace.field_bundle_diagnostics.poisson.%s" % field,
            )
            >= 0.0,
            "trace.field_bundle_diagnostics.poisson.%s" % field,
        )
    backward_error = _finite_number(
        poisson.get("backward_error_linf"),
        "trace.field_bundle_diagnostics.poisson.backward_error_linf",
    )
    boundary_error = _finite_number(
        poisson.get("boundary_linf"),
        "trace.field_bundle_diagnostics.poisson.boundary_linf",
    )
    _require(
        backward_error <= POISSON_BACKWARD_ERROR_TOLERANCE,
        "trace.field_bundle_diagnostics.poisson.backward_error_linf",
        "crosses the registered normalized backward-error tolerance",
    )
    _require(
        boundary_error <= POISSON_BOUNDARY_TOLERANCE,
        "trace.field_bundle_diagnostics.poisson.boundary_linf",
        "crosses the registered Dirichlet-boundary tolerance",
    )
    interior_min = _finite_number(
        poisson.get("interior_min"),
        "trace.field_bundle_diagnostics.poisson.interior_min",
    )
    interior_max = _finite_number(
        poisson.get("interior_max"),
        "trace.field_bundle_diagnostics.poisson.interior_max",
    )
    _require(
        0.0 < interior_min <= interior_max,
        "trace.field_bundle_diagnostics.poisson interior range",
    )
    _require(
        _integer(
            poisson.get("sign_violation_count"),
            "trace.field_bundle_diagnostics.poisson.sign_violation_count",
        )
        == 0,
        "trace.field_bundle_diagnostics.poisson.sign_violation_count",
    )
    _close(
        poisson.get("sign_violation_linf"),
        0.0,
        "trace.field_bundle_diagnostics.poisson.sign_violation_linf",
    )

    pairing_key = {
        "pair_group_id": pairing.get("pair_group_id"),
        "official_settled_state": restore.get(
            "official_integration_state_sha256"
        ),
        "historical_action_ledger": source_replay.get(
            "executed_sequence_sha256"
        ),
        "field_sha256": field_hashes.get("field_sha256"),
        "source_observation_sha256": source_replay.get(
            "initial_observation_sha256"
        ),
    }
    _require(
        pairing.get("pairing_key_sha256")
        == _producer_configuration_sha256(pairing_key),
        "result.pairing.pairing_key_sha256",
        "does not bind the registered paired initial state and static field",
    )

    arm = result.get("arm")
    intervention = _mapping(
        runtime.get("intervention"), "result.runtime.intervention"
    )
    intervention_identity = {
        "arm": arm,
        "field_sha256": field_hashes.get("field_sha256"),
        "protected": (
            []
            if arm == "joint_velocity_adapter_only"
            else ["robot0_link5", "robot0_link6"]
        ),
    }
    _require(
        intervention.get("configuration_sha256")
        == _producer_configuration_sha256(intervention_identity),
        "result.runtime.intervention.configuration_sha256",
        "does not bind the registered arm and protected geometry",
    )
    optimizer = _mapping(result.get("optimizer"), "result.optimizer")
    optimizer_identity = {
        "runtime_protocol_parameter_block_sha256": runtime.get(
            "protocol_parameter_block_sha256"
        ),
        "arm": arm,
        "solver": "osqp",
    }
    _require(
        optimizer.get("configuration_sha256")
        == _producer_configuration_sha256(optimizer_identity),
        "result.optimizer.configuration_sha256",
        "does not bind the registered arm, solver, and protocol parameters",
    )


def _validate_full_robot_surface_sampling(trace: Mapping[str, Any]) -> None:
    """Validate the v2 trace's authoritative mixed-geometry D_sim sampler."""

    resolved = _mapping(trace.get("resolved_geometry"), "trace.resolved_geometry")
    raw_expected_geoms = _list(
        resolved.get("robot_geom_ids"), "trace.resolved_geometry.robot_geom_ids"
    )
    expected_geoms = [
        _integer(value, "trace.resolved_geometry.robot_geom_ids[%d]" % index)
        for index, value in enumerate(raw_expected_geoms)
    ]
    raw_expected_names = _list(
        resolved.get("robot_geom_names"),
        "trace.resolved_geometry.robot_geom_names",
    )
    expected_names = []
    for index, value in enumerate(raw_expected_names):
        _require(
            isinstance(value, str) and bool(value),
            "trace.resolved_geometry.robot_geom_names[%d]" % index,
            "must be a nonempty string",
        )
        expected_names.append(value)
    robot_bodies = {
        _integer(value, "trace.resolved_geometry.robot_body_ids[%d]" % index)
        for index, value in enumerate(
            _list(
                resolved.get("robot_body_ids"),
                "trace.resolved_geometry.robot_body_ids",
            )
        )
    }
    _require(bool(expected_geoms), "trace.resolved_geometry.robot_geom_ids", "must be nonempty")
    _require(
        expected_geoms == sorted(set(expected_geoms)),
        "trace.resolved_geometry.robot_geom_ids",
        "must be strictly ordered and unique",
    )
    _require(
        len(expected_names) == len(expected_geoms) and bool(robot_bodies),
        "trace.resolved_geometry robot identities",
    )

    sampling = _mapping(
        trace.get("full_robot_surface_sampling"),
        "trace.full_robot_surface_sampling",
    )
    epsilon = _finite(sampling.get("epsilon_m"), "trace.full_robot_surface_sampling.epsilon_m")
    maximum = _finite(
        sampling.get("maximum_surface_cover_radius_m"),
        "trace.full_robot_surface_sampling.maximum_surface_cover_radius_m",
    )
    _require(epsilon > 0.0, "trace.full_robot_surface_sampling.epsilon_m", "must be positive")
    _require(
        0.0 <= maximum < epsilon,
        "trace.full_robot_surface_sampling.maximum_surface_cover_radius_m",
        "must be nonnegative and strictly below epsilon",
    )
    semantics = sampling.get("coverage_semantics")
    _require(
        semantics == FULL_ROBOT_COVERAGE_SEMANTICS,
        "trace.full_robot_surface_sampling.coverage_semantics",
        "differs from the v2 registered semantics",
    )
    sample_hash = sampling.get("sample_ledger_sha256")
    _require(
        isinstance(sample_hash, str)
        and len(sample_hash) == 64
        and all(character in "0123456789abcdef" for character in sample_hash),
        "trace.full_robot_surface_sampling.sample_ledger_sha256",
        "must be a lowercase SHA-256 digest",
    )
    records = _list(
        sampling.get("geom_records"),
        "trace.full_robot_surface_sampling.geom_records",
    )
    _require(
        len(records) == len(expected_geoms),
        "trace.full_robot_surface_sampling.geom_records",
        "count differs from authoritative robot geoms",
    )
    observed_geoms: List[int] = []
    observed_names: List[str] = []
    sample_total = 0
    record_maximum = 0.0
    type_counts = {"mesh": 0, "box": 0, "cylinder": 0}
    expected_certificate = {
        "compiled_mesh_convex_hull": "analytic_triangle_lattice_covering_bound",
        "exact_box_faces": "analytic_triangle_lattice_covering_bound",
        "exact_cylinder_surface": "analytic_cylinder_parameter_grid_covering_bound",
    }
    expected_type = {
        "compiled_mesh_convex_hull": (7, "mesh"),
        "exact_box_faces": (6, "box"),
        "exact_cylinder_surface": (5, "cylinder"),
    }
    for index, raw_record in enumerate(records):
        label = "trace.full_robot_surface_sampling.geom_records[%d]" % index
        record = _mapping(raw_record, label)
        observed_geoms.append(_integer(record.get("geom_id"), label + ".geom_id"))
        body_id = _integer(record.get("body_id"), label + ".body_id")
        _require(body_id in robot_bodies, label + ".body_id", "is outside the robot tree")
        for field in ("geom_name", "body_name"):
            value = record.get(field)
            _require(
                isinstance(value, str) and bool(value),
                label + "." + field,
                "must be a nonempty string",
            )
        observed_names.append(record["geom_name"])
        geometry_kind = record.get("geometry_kind")
        _require(
            geometry_kind in expected_certificate,
            label + ".geometry_kind",
            "is unsupported",
        )
        _require(
            record.get("certificate_kind") == expected_certificate[geometry_kind],
            label + ".certificate_kind",
        )
        _require(
            record.get("geom_type_name") == expected_type[geometry_kind][1],
            label + ".geom_type_name",
        )
        _require(
            _integer(record.get("geom_type_id"), label + ".geom_type_id")
            == expected_type[geometry_kind][0],
            label + ".geom_type_id",
        )
        _vector(record.get("geom_size"), 3, label + ".geom_size")
        contype = _integer(record.get("contype"), label + ".contype")
        conaffinity = _integer(record.get("conaffinity"), label + ".conaffinity")
        _require(
            isinstance(record.get("mask_collision_enabled"), bool),
            label + ".mask_collision_enabled",
            "must be Boolean",
        )
        _require(
            record.get("mask_collision_enabled") is bool(contype or conaffinity),
            label + ".mask_collision_enabled",
            "differs from contype/conaffinity",
        )
        _require(
            record.get("selection_authority") == "authoritative_resolved_geom_ids",
            label + ".selection_authority",
        )
        parameters = _mapping(
            record.get("certificate_parameters"),
            label + ".certificate_parameters",
        )
        _close(
            parameters.get("requested_epsilon_m"),
            epsilon,
            label + ".certificate_parameters.requested_epsilon_m",
        )
        if geometry_kind == "exact_cylinder_surface":
            for field in (
                "angular_sample_count",
                "axial_interval_count",
                "cap_radial_interval_count",
            ):
                _require(
                    _integer(parameters.get(field), label + ".certificate_parameters." + field)
                    > 0,
                    label + ".certificate_parameters." + field,
                    "must be positive",
                )
        _require(
            _integer(record.get("surface_element_count"), label + ".surface_element_count") > 0,
            label + ".surface_element_count",
            "must be positive",
        )
        sample_count = _integer(record.get("sample_count"), label + ".sample_count")
        _require(sample_count > 0, label + ".sample_count", "must be positive")
        if geometry_kind == "exact_cylinder_surface":
            angular = int(parameters["angular_sample_count"])
            axial = int(parameters["axial_interval_count"])
            radial = int(parameters["cap_radial_interval_count"])
            _require(
                record["surface_element_count"]
                == angular * (axial + 2 * radial),
                label + ".surface_element_count",
                "differs from the cylinder grid",
            )
            _require(
                sample_count
                == angular * (axial + 1) + 2 * (1 + angular * (radial - 1)),
                label + ".sample_count",
                "differs from the unique cylinder surface grid",
            )
        sample_total += sample_count
        cover = _finite(
            record.get("certified_surface_cover_radius_m"),
            label + ".certified_surface_cover_radius_m",
        )
        _require(
            0.0 <= cover < epsilon,
            label + ".certified_surface_cover_radius_m",
            "must be nonnegative and strictly below epsilon",
        )
        if geometry_kind == "exact_cylinder_surface":
            geom_size = _vector(record.get("geom_size"), 3, label + ".geom_size")
            radius, half_length = geom_size[:2]
            _require(
                radius > 0.0 and half_length > 0.0,
                label + ".geom_size",
                "must contain a positive radius and half-length",
            )
            angular_bound = 2.0 * radius * math.sin(
                math.pi / (2.0 * float(angular))
            )
            recomputed_cover = max(
                math.hypot(angular_bound, half_length / float(axial)),
                math.hypot(angular_bound, radius / (2.0 * float(radial))),
            )
            _close(
                cover,
                recomputed_cover,
                label + ".certified_surface_cover_radius_m.reconstructed",
                absolute=1e-15,
            )
        type_counts[expected_type[geometry_kind][1]] += 1
        record_maximum = max(record_maximum, cover)
    _require(
        observed_geoms == expected_geoms and observed_names == expected_names,
        "trace.full_robot_surface_sampling.geom_records",
        "geom IDs differ from authoritative resolution",
    )
    _require(
        type_counts == {"mesh": 11, "box": 4, "cylinder": 1},
        "trace.full_robot_surface_sampling geometry type counts",
    )
    cylinder_records = [
        record for record in records if record.get("geom_type_name") == "cylinder"
    ]
    cylinder = cylinder_records[0]
    _require(
        cylinder.get("geom_id") == 84
        and cylinder.get("geom_name") == "mount0_pedestal_col"
        and cylinder.get("geom_type_id") == 5
        and cylinder.get("geom_size") == [0.18, 0.31, 0.0],
        "trace.full_robot_surface_sampling first-canary pedestal cylinder",
    )
    _require(
        _integer(sampling.get("sample_count"), "trace.full_robot_surface_sampling.sample_count")
        == sample_total,
        "trace.full_robot_surface_sampling.sample_count",
        "differs from component sum",
    )
    _close(
        maximum,
        record_maximum,
        "trace.full_robot_surface_sampling.maximum_surface_cover_radius_m",
    )
    roundtrip = _mapping(
        sampling.get("roundtrip"), "trace.full_robot_surface_sampling.roundtrip"
    )
    _require(
        roundtrip.get("passed") is True,
        "trace.full_robot_surface_sampling.roundtrip.passed",
    )
    _require(
        _integer(roundtrip.get("sample_count"), "trace.full_robot_surface_sampling.roundtrip.sample_count")
        == sample_total,
        "trace.full_robot_surface_sampling.roundtrip.sample_count",
    )
    roundtrip_error = _finite(
        roundtrip.get("maximum_roundtrip_error_m"),
        "trace.full_robot_surface_sampling.roundtrip.maximum_roundtrip_error_m",
    )
    roundtrip_tolerance = _finite(
        roundtrip.get("tolerance_m"),
        "trace.full_robot_surface_sampling.roundtrip.tolerance_m",
    )
    _require(
        0.0 <= roundtrip_error <= roundtrip_tolerance and roundtrip_tolerance > 0.0,
        "trace.full_robot_surface_sampling.roundtrip",
        "error exceeds its positive tolerance",
    )


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
        entered_action = _vector(
            action, 7, "trace.outcome.entered_source_actions[%d]" % index
        )
        _require(
            entered_action[3:6] == [0.0, 0.0, 0.0],
            "trace.outcome.entered_source_actions[%d]" % index,
            "contains rotation intent discarded by the registered translation-only adapter",
        )
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
        source_action = _vector(
            _mapping(inner_trace[index], "inner trace row").get("source_action"),
            7,
            "trace.outcome.inner_trace[%d].source_action" % index,
        )
        adapter = _mapping(
            _mapping(inner_trace[index], "inner trace row").get("adapter"),
            "trace.outcome.inner_trace[%d].adapter" % index,
        )
        _exact_keys(
            adapter,
            ADAPTER_STEP_FIELDS,
            "trace.outcome.inner_trace[%d].adapter" % index,
        )
        _mapping(
            adapter.get("diagnostics"),
            "trace.outcome.inner_trace[%d].adapter.diagnostics" % index,
        )
        adapter_qdot = _vector(
            adapter.get("qdot_physical_rad_s"),
            7,
            "trace.outcome.inner_trace[%d].adapter.qdot_physical_rad_s" % index,
        )
        adapter_normalized = _vector(
            adapter.get("normalized_action"),
            8,
            "trace.outcome.inner_trace[%d].adapter.normalized_action" % index,
        )
        _vector(
            adapter.get("qdot_unclipped_rad_s"),
            7,
            "trace.outcome.inner_trace[%d].adapter.qdot_unclipped_rad_s" % index,
        )
        _vector(
            adapter.get("desired_twist"),
            6,
            "trace.outcome.inner_trace[%d].adapter.desired_twist" % index,
        )
        _vector(
            adapter.get("achieved_twist"),
            6,
            "trace.outcome.inner_trace[%d].adapter.achieved_twist" % index,
        )
        _require(
            _canonical_equal(adapter_qdot, nominal),
            "trace.outcome.inner_trace[%d].adapter.qdot_physical_rad_s" % index,
            "differs from the nominal command ledger",
        )
        for joint, value in enumerate(adapter_normalized[:7]):
            _close(
                value,
                nominal[joint] / JOINT_VELOCITY_LIMIT_RAD_S,
                "trace.outcome.inner_trace[%d].adapter.normalized_action[%d]"
                % (index, joint),
            )
        _close(
            adapter_normalized[7],
            source_action[6],
            "trace.outcome.inner_trace[%d].adapter.normalized_action[7]" % index,
        )

        execution_record = _mapping(
            _mapping(inner_trace[index], "inner trace row").get("execution"),
            "trace.outcome.inner_trace[%d].execution" % index,
        )
        _exact_keys(
            execution_record,
            EXECUTION_RECORD_FIELDS,
            "trace.outcome.inner_trace[%d].execution" % index,
        )
        execution_nominal = _vector(
            execution_record.get("nominal_qdot_physical_rad_s"),
            7,
            "trace.outcome.inner_trace[%d].execution.nominal_qdot_physical_rad_s"
            % index,
        )
        execution_safe = _vector(
            execution_record.get("executed_qdot_physical_rad_s"),
            7,
            "trace.outcome.inner_trace[%d].execution.executed_qdot_physical_rad_s"
            % index,
        )
        correction = _vector(
            execution_record.get("filter_correction_rad_s"),
            7,
            "trace.outcome.inner_trace[%d].execution.filter_correction_rad_s"
            % index,
        )
        normalized = _vector(
            execution_record.get("normalized_executed_action"),
            8,
            "trace.outcome.inner_trace[%d].execution.normalized_executed_action"
            % index,
        )
        expected_correction = [
            safe - nominal_value
            for safe, nominal_value in zip(executed, nominal)
        ]
        _require(
            _canonical_equal(execution_nominal, nominal)
            and _canonical_equal(execution_safe, executed)
            and _canonical_equal(correction, expected_correction),
            "trace.outcome.inner_trace[%d].execution command binding" % index,
        )
        _close(
            execution_record.get("filter_correction_l2_rad_s"),
            _norm(expected_correction),
            "trace.outcome.inner_trace[%d].execution.filter_correction_l2_rad_s"
            % index,
        )
        for joint, value in enumerate(normalized[:7]):
            _close(
                value,
                executed[joint] / JOINT_VELOCITY_LIMIT_RAD_S,
                "trace.outcome.inner_trace[%d].execution.normalized_executed_action[%d]"
                % (index, joint),
            )
        _close(
            normalized[7],
            source_action[6],
            "trace.outcome.inner_trace[%d].execution.normalized_executed_action[7]"
            % index,
        )

    fail_rows = _list(
        outcome.get("fail_closed_attempt_trace"),
        "trace.outcome.fail_closed_attempt_trace",
    )
    if result.get("completion_class") == "executed":
        _require(
            not fail_rows,
            "trace.outcome.fail_closed_attempt_trace",
            "must be empty for an executed arm",
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
    _close(
        _mapping(outcome.get("restore"), "trace.outcome.restore").get(
            "copied_cur_time_s"
        ),
        settled_time,
        "trace.outcome.restore.copied_cur_time_s",
        absolute=1e-10,
    )
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
    _close(
        linf_threshold,
        VELOCITY_TRACKING_LINF_THRESHOLD_RAD_S,
        "trace.outcome.tracking.maximum_linf_threshold_rad_s",
    )
    _close(
        rmse_threshold,
        VELOCITY_TRACKING_RMSE_THRESHOLD_RAD_S,
        "trace.outcome.tracking.maximum_rmse_threshold_rad_s",
    )
    validity = _mapping(outcome.get("validity"), "trace.outcome.validity")
    _close(
        validity.get("velocity_tracking_linf_threshold_rad_s"),
        linf_threshold,
        "trace.outcome.validity.velocity_tracking_linf_threshold_rad_s",
    )
    _close(
        validity.get("velocity_tracking_rmse_threshold_rad_s"),
        rmse_threshold,
        "trace.outcome.validity.velocity_tracking_rmse_threshold_rad_s",
    )

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
            all(
                abs(value) <= JOINT_VELOCITY_LIMIT_RAD_S
                for value in nominal + issued
            ),
            label + " commands",
            "cross the registered joint-velocity limit",
        )
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


def _validate_eef_reference_site(outcome: Mapping[str, Any]) -> None:
    label = "trace.outcome.eef_reference_site"
    audit = _mapping(outcome.get("eef_reference_site"), label)
    _exact_keys(audit, EEF_REFERENCE_SITE_FIELDS, label)
    _require(
        _integer(audit.get("site_id"), label + ".site_id") >= 0,
        label + ".site_id",
    )
    _require(
        audit.get("site_name") == REGISTERED_EEF_SITE_NAME,
        label + ".site_name",
        "differs from the registered Panda grip site",
    )
    _require(
        audit.get("jacobian_row_order") == "linear_xyz_then_angular_xyz"
        and _canonical_equal(audit.get("shape"), [6, 7]),
        label + " Jacobian layout",
    )
    delta = _finite_number(
        audit.get("linear_finite_difference_delta_rad"),
        label + ".linear_finite_difference_delta_rad",
    )
    maximum = _finite_number(
        audit.get("maximum_linear_jacobian_error_m_per_rad"),
        label + ".maximum_linear_jacobian_error_m_per_rad",
    )
    tolerance = _finite_number(
        audit.get("absolute_tolerance_m_per_rad"),
        label + ".absolute_tolerance_m_per_rad",
    )
    _close(delta, 1e-6, label + ".linear_finite_difference_delta_rad")
    _close(tolerance, 2e-6, label + ".absolute_tolerance_m_per_rad")
    _require(
        0.0 <= maximum <= tolerance
        and audit.get("passed") is True
        and audit.get("live_state_mutated") is False,
        label,
        "does not certify the registered read-only Jacobian check",
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
    settled_digest = _producer_float64_vector_sha256(settled)
    _require(
        settled_digest
        == ledger.get("settled_active_obstacle_position_sha256")
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
    _require(
        result.get("case_id") == REGISTERED_CANARY_CASE_ID
        and definition.get("schema_version") == "vlsa_native_goal_progress.v1"
        and definition.get("source") == "native_bddl_goal_predicates"
        and definition.get("logic") == "conjunction"
        and _canonical_equal(atoms, REGISTERED_CANARY_GOAL_ATOMS),
        "trace.outcome.goal_definition",
        "differs from the registered first-canary native BDDL goal",
    )
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


def _id_name_map(
    resolved: Mapping[str, Any], *, id_field: str, name_field: str, label: str
) -> Dict[int, str]:
    raw_ids = _list(resolved.get(id_field), label + ".ids")
    raw_names = _list(resolved.get(name_field), label + ".names")
    _require(
        bool(raw_ids) and len(raw_ids) == len(raw_names),
        label,
        "ID/name coverage differs",
    )
    output: Dict[int, str] = {}
    for index, (raw_id, raw_name) in enumerate(zip(raw_ids, raw_names)):
        identifier = _integer(raw_id, "%s.ids[%d]" % (label, index))
        _require(identifier not in output, label, "contains duplicate IDs")
        _require(
            isinstance(raw_name, str) and bool(raw_name),
            "%s.names[%d]" % (label, index),
            "must be a nonempty string",
        )
        output[identifier] = raw_name
    _require(
        list(output) == sorted(output),
        label,
        "IDs must be strictly ordered",
    )
    return output


def _contact_authority(
    trace: Mapping[str, Any],
) -> Tuple[
    Dict[int, str],
    Dict[int, str],
    Dict[int, str],
    Dict[int, str],
    Dict[int, Tuple[int, str]],
    set,
    set,
]:
    resolved = _mapping(trace.get("resolved_geometry"), "trace.resolved_geometry")
    _exact_keys(resolved, RESOLVED_GEOMETRY_FIELDS, "trace.resolved_geometry")
    robot_geoms = _id_name_map(
        resolved,
        id_field="robot_geom_ids",
        name_field="robot_geom_names",
        label="trace resolved robot geoms",
    )
    obstacle_geoms = _id_name_map(
        resolved,
        id_field="obstacle_geom_ids",
        name_field="obstacle_geom_names",
        label="trace resolved obstacle geoms",
    )
    robot_bodies = _id_name_map(
        resolved,
        id_field="robot_body_ids",
        name_field="robot_body_names",
        label="trace resolved robot bodies",
    )
    obstacle_bodies = _id_name_map(
        resolved,
        id_field="obstacle_body_ids",
        name_field="obstacle_body_names",
        label="trace resolved obstacle bodies",
    )
    _require(
        not (set(robot_bodies) & set(obstacle_bodies)),
        "trace resolved body authority",
        "robot and obstacle bodies overlap",
    )
    for prefix, bodies in (
        ("robot", robot_bodies),
        ("obstacle", obstacle_bodies),
    ):
        root_sequence = [
            _integer(
                value,
                "trace.resolved_geometry.%s_root_body_ids[%d]" % (prefix, index),
            )
            for index, value in enumerate(
                _list(
                    resolved.get("%s_root_body_ids" % prefix),
                    "trace.resolved_geometry.%s_root_body_ids" % prefix,
                )
            )
        ]
        _require(
            bool(root_sequence)
            and root_sequence == sorted(set(root_sequence))
            and set(root_sequence) <= set(bodies),
            "trace resolved %s root-body authority" % prefix,
        )
    _require(
        not (set(robot_geoms) & set(obstacle_geoms)),
        "trace resolved contact authority",
        "robot and obstacle geoms overlap",
    )
    raw_link_ids = _list(
        resolved.get("link56_geom_ids"),
        "trace.resolved_geometry.link56_geom_ids",
    )
    raw_link_names = _list(
        resolved.get("link56_geom_names"),
        "trace.resolved_geometry.link56_geom_names",
    )
    _require(
        bool(raw_link_ids) and len(raw_link_ids) == len(raw_link_names),
        "trace resolved link56 geom authority",
        "ID/name coverage differs",
    )
    link_id_sequence = [
        _integer(value, "trace.resolved_geometry.link56_geom_ids[%d]" % index)
        for index, value in enumerate(raw_link_ids)
    ]
    link_ids = set(link_id_sequence)
    _require(
        len(link_ids) == len(link_id_sequence)
        and link_id_sequence == sorted(link_id_sequence)
        and link_ids <= set(robot_geoms)
        and all(
            isinstance(name, str)
            and bool(name)
            and robot_geoms[geom_id] == name
            for geom_id, name in zip(link_id_sequence, raw_link_names)
        ),
        "trace resolved link56 geom authority",
    )
    link_body_id_sequence = [
        _integer(value, "trace.resolved_geometry.link56_body_ids[%d]" % index)
        for index, value in enumerate(
            _list(
                resolved.get("link56_body_ids"),
                "trace.resolved_geometry.link56_body_ids",
            )
        )
    ]
    link_body_ids = set(link_body_id_sequence)
    _require(
        len(link_body_id_sequence) == 2
        and len(link_body_ids) == 2
        and link_body_id_sequence == sorted(link_body_id_sequence)
        and link_body_ids <= set(robot_bodies)
        and {robot_bodies[body_id] for body_id in link_body_ids}
        == {"robot0_link5", "robot0_link6"},
        "trace resolved link56 body authority",
    )
    raw_pairs = _list(
        resolved.get("collision_enabled_pairs"),
        "trace.resolved_geometry.collision_enabled_pairs",
    )
    collision_pairs = set()
    for index, raw_pair in enumerate(raw_pairs):
        pair_values = _list(
            raw_pair,
            "trace.resolved_geometry.collision_enabled_pairs[%d]" % index,
        )
        _require(
            len(pair_values) == 2,
            "trace.resolved_geometry.collision_enabled_pairs[%d]" % index,
            "must contain two geom IDs",
        )
        pair = (
            _integer(pair_values[0], "trace resolved pair robot geom"),
            _integer(pair_values[1], "trace resolved pair obstacle geom"),
        )
        _require(
            pair not in collision_pairs
            and pair[0] in robot_geoms
            and pair[1] in obstacle_geoms,
            "trace resolved collision pair authority",
        )
        collision_pairs.add(pair)
    _require(
        bool(collision_pairs)
        and [tuple(pair) for pair in raw_pairs] == sorted(collision_pairs),
        "trace resolved collision pair authority",
        "must be nonempty, unique, and strictly ordered",
    )
    paired_robot_geoms = {pair[0] for pair in collision_pairs}
    paired_obstacle_geoms = {pair[1] for pair in collision_pairs}
    _require(
        set(robot_geoms) == paired_robot_geoms
        and set(obstacle_geoms) == paired_obstacle_geoms,
        "trace resolved full-robot collision-pair authority",
        "does not expose every declared robot and selected-obstacle geom to contact monitoring",
    )
    _require(
        link_ids <= paired_robot_geoms,
        "trace resolved link56 collision-pair authority",
        "does not expose every protected link-5/6 geom to contact monitoring",
    )

    sampling = _mapping(
        trace.get("full_robot_surface_sampling"),
        "trace.full_robot_surface_sampling",
    )
    robot_geom_bodies: Dict[int, Tuple[int, str]] = {}
    for index, raw_record in enumerate(
        _list(
            sampling.get("geom_records"),
            "trace.full_robot_surface_sampling.geom_records",
        )
    ):
        label = "trace.full_robot_surface_sampling.geom_records[%d]" % index
        sample_record = _mapping(raw_record, label)
        geom_id = _integer(sample_record.get("geom_id"), label + ".geom_id")
        body_id = _integer(sample_record.get("body_id"), label + ".body_id")
        body_name = sample_record.get("body_name")
        _require(
            geom_id in robot_geoms
            and body_id in robot_bodies
            and body_name == robot_bodies[body_id],
            label,
            "geom/body provenance differs from resolved authority",
        )
        robot_geom_bodies[geom_id] = (body_id, body_name)
    _require(
        set(robot_geom_bodies) == set(robot_geoms),
        "trace robot geom/body contact authority",
    )
    _require(
        link_ids
        == {
            geom_id
            for geom_id, (body_id, _body_name_value) in robot_geom_bodies.items()
            if body_id in link_body_ids
        },
        "trace resolved link56 geom/body authority",
        "does not equal every collision geom owned by robot0_link5/link6",
    )
    return (
        robot_geoms,
        obstacle_geoms,
        robot_bodies,
        obstacle_bodies,
        robot_geom_bodies,
        collision_pairs,
        link_ids,
    )


def _validate_contact_record(
    raw_record: Any,
    *,
    label: str,
    expected_phase: str,
    physics_steps: int,
    near_tolerance_m: Optional[float],
    robot_geoms: Mapping[int, str],
    obstacle_geoms: Mapping[int, str],
    robot_bodies: Mapping[int, str],
    obstacle_bodies: Mapping[int, str],
    robot_geom_bodies: Mapping[int, Tuple[int, str]],
    collision_pairs: set,
) -> Mapping[str, Any]:
    record = _mapping(raw_record, label)
    _require(record.get("source_phase") == expected_phase, label + ".source_phase")
    observation = record.get("observation_index")
    if expected_phase == "settled_post_integration_recomputed":
        for field in (
            "observation_index",
            "high_level_index",
            "inner_control_index",
            "physics_substep_index",
        ):
            _require(record.get(field) is None, label + "." + field)
    else:
        observed_index = _integer(observation, label + ".observation_index")
        _require(
            observed_index < physics_steps,
            label + ".observation_index",
            "is outside completed physics exposure",
        )
        coordinate = _coordinate_for_physics(observed_index)
        _require(
            _row_coordinate(record, physics=True, label=label) == coordinate,
            label,
            "cadence coordinate differs from observation index",
        )
    _integer(record.get("mujoco_contact_index"), label + ".mujoco_contact_index")

    distance = _finite_number(
        record.get("contact_distance_m"), label + ".contact_distance_m"
    )
    physical = _boolean(
        record.get("is_physical_nonpositive_distance_contact"),
        label + ".is_physical_nonpositive_distance_contact",
    )
    _require(
        physical is (distance <= 0.0),
        label + ".is_physical_nonpositive_distance_contact",
        "differs from MuJoCo nonpositive distance",
    )
    near = _boolean(
        record.get("within_registered_near_contact_tolerance"),
        label + ".within_registered_near_contact_tolerance",
    )
    if near_tolerance_m is not None:
        _require(
            near is (distance <= near_tolerance_m),
            label + ".within_registered_near_contact_tolerance",
            "differs from the registered distance threshold",
        )
    efc_address = _signed_integer(record.get("efc_address"), label + ".efc_address")
    active = _boolean(
        record.get("solver_constraint_active"),
        label + ".solver_constraint_active",
    )
    _require(
        active is (efc_address >= 0),
        label + ".solver_constraint_active",
        "differs from efc_address",
    )

    robot_geom = _integer(record.get("robot_geom_id"), label + ".robot_geom_id")
    obstacle_geom = _integer(
        record.get("obstacle_geom_id"), label + ".obstacle_geom_id"
    )
    _require(
        robot_geom in robot_geoms
        and obstacle_geom in obstacle_geoms
        and (robot_geom, obstacle_geom) in collision_pairs
        and record.get("robot_geom_name") == robot_geoms[robot_geom]
        and record.get("obstacle_geom_name") == obstacle_geoms[obstacle_geom],
        label,
        "is outside resolved robot/selected-obstacle contact authority",
    )
    robot_body = _integer(record.get("robot_body_id"), label + ".robot_body_id")
    obstacle_body = _integer(
        record.get("obstacle_body_id"), label + ".obstacle_body_id"
    )
    _require(
        robot_body in robot_bodies
        and obstacle_body in obstacle_bodies
        and record.get("robot_body_name") == robot_bodies[robot_body]
        and record.get("obstacle_body_name") == obstacle_bodies[obstacle_body]
        and robot_geom_bodies[robot_geom] == (robot_body, record.get("robot_body_name")),
        label,
        "body provenance differs from resolved authority",
    )
    geom1 = _integer(record.get("mujoco_geom1_id"), label + ".mujoco_geom1_id")
    geom2 = _integer(record.get("mujoco_geom2_id"), label + ".mujoco_geom2_id")
    if (geom1, geom2) == (robot_geom, obstacle_geom):
        expected_name1 = robot_geoms[robot_geom]
        expected_name2 = obstacle_geoms[obstacle_geom]
    elif (geom1, geom2) == (obstacle_geom, robot_geom):
        expected_name1 = obstacle_geoms[obstacle_geom]
        expected_name2 = robot_geoms[robot_geom]
    else:
        _fail(label, "MuJoCo geom orientation differs from the resolved pair")
    _require(
        record.get("mujoco_geom1_name") == expected_name1
        and record.get("mujoco_geom2_name") == expected_name2,
        label,
        "MuJoCo geom-name provenance differs",
    )
    _finite_number_vector(
        record.get("position_world_m"), 3, label + ".position_world_m"
    )
    _finite_number_vector(
        record.get("frame_normal_mujoco_geom1_to_geom2_world"),
        3,
        label + ".frame_normal_mujoco_geom1_to_geom2_world",
    )
    for field in (
        "contact_includemargin_m",
        "robot_geom_margin_m",
        "robot_geom_gap_m",
        "obstacle_geom_margin_m",
        "obstacle_geom_gap_m",
    ):
        _finite_number(record.get(field), label + "." + field)
    explicit_pair_id = record.get("explicit_pair_id")
    if explicit_pair_id is None:
        _require(
            record.get("explicit_pair_margin_m") is None
            and record.get("explicit_pair_gap_m") is None,
            label + ".explicit_pair",
            "margin and gap must be null without an explicit pair",
        )
    else:
        _integer(explicit_pair_id, label + ".explicit_pair_id")
        _finite_number(
            record.get("explicit_pair_margin_m"),
            label + ".explicit_pair_margin_m",
        )
        _finite_number(
            record.get("explicit_pair_gap_m"), label + ".explicit_pair_gap_m"
        )
    force_available = _boolean(
        record.get("force_available"), label + ".force_available"
    )
    _require(
        isinstance(record.get("force_semantics"), str)
        and bool(record.get("force_semantics")),
        label + ".force_semantics",
        "must be a nonempty string",
    )
    force_fields = (
        "force_contact_frame_n",
        "force_world_n",
        "normal_force_n",
        "impulse_estimate_contact_frame_ns",
        "impulse_estimate_world_ns",
    )
    if force_available:
        _require(
            expected_phase == "live_solver_phase_preintegration_geometry"
            and active
            and record.get("force_unavailable_reason") is None,
            label + ".force_available",
            "is only valid for an active live-solver constraint",
        )
        wrench = _finite_number_vector(
            record.get("force_contact_frame_n"), 6, label + ".force_contact_frame_n"
        )
        _finite_number_vector(
            record.get("force_world_n"), 3, label + ".force_world_n"
        )
        _close(
            record.get("normal_force_n"),
            wrench[0],
            label + ".normal_force_n",
        )
        _finite_number_vector(
            record.get("impulse_estimate_contact_frame_ns"),
            6,
            label + ".impulse_estimate_contact_frame_ns",
        )
        _finite_number_vector(
            record.get("impulse_estimate_world_ns"),
            3,
            label + ".impulse_estimate_world_ns",
        )
    else:
        _require(
            all(record.get(field) is None for field in force_fields)
            and isinstance(record.get("force_unavailable_reason"), str)
            and bool(record.get("force_unavailable_reason")),
            label + ".force_unavailable",
            "must carry no force values and one reason",
        )
    return record


def _contact_ledger(
    value: Any,
    *,
    label: str,
    expected_phase: str,
    physics_steps: int,
    near_tolerance_m: Optional[float],
    authority: Tuple[
        Mapping[int, str],
        Mapping[int, str],
        Mapping[int, str],
        Mapping[int, str],
        Mapping[int, Tuple[int, str]],
        set,
        set,
    ],
) -> List[Mapping[str, Any]]:
    (
        robot_geoms,
        obstacle_geoms,
        robot_bodies,
        obstacle_bodies,
        robot_geom_bodies,
        collision_pairs,
        _link_ids,
    ) = authority
    records = [
        _validate_contact_record(
            raw,
            label="%s[%d]" % (label, index),
            expected_phase=expected_phase,
            physics_steps=physics_steps,
            near_tolerance_m=near_tolerance_m,
            robot_geoms=robot_geoms,
            obstacle_geoms=obstacle_geoms,
            robot_bodies=robot_bodies,
            obstacle_bodies=obstacle_bodies,
            robot_geom_bodies=robot_geom_bodies,
            collision_pairs=collision_pairs,
        )
        for index, raw in enumerate(_list(value, label))
    ]
    return records


def _rollout_contact_order(contact: Mapping[str, Any]) -> Tuple[int, int, int]:
    observation = _integer(
        contact.get("observation_index"), "trace contact observation_index"
    )
    phase = {
        "live_solver_phase_preintegration_geometry": 0,
        "post_integration_recomputed": 1,
    }.get(contact.get("source_phase"), 2)
    return (
        observation,
        phase,
        _integer(contact.get("mujoco_contact_index"), "trace contact index"),
    )


def _require_contact_order_and_uniqueness(
    records: Sequence[Mapping[str, Any]], *, settled: bool, label: str
) -> None:
    if settled:
        keys = [
            _integer(record.get("mujoco_contact_index"), label + ".contact_index")
            for record in records
        ]
        _require(keys == sorted(keys), label, "is not in MuJoCo contact-index order")
    else:
        keys = [_rollout_contact_order(record) for record in records]
        _require(keys == sorted(keys), label, "is not in producer cadence order")
    _require(len(keys) == len(set(keys)), label, "contains a duplicate contact row")


SAMPLE_CLEARANCE_FIELDS = {
    "available",
    "authority",
    "minimum_exact_sample_to_obb_distance_m",
    "certified_coverage_radius_m",
    "full_surface_clearance_lower_bound_m",
    "minimum_observation_index",
    "sample_id",
    "robot_geom_id",
    "obstacle_geom_id",
}


def _validate_sample_clearance(
    raw: Any,
    *,
    label: str,
    trace: Mapping[str, Any],
    authority: Tuple[
        Mapping[int, str],
        Mapping[int, str],
        Mapping[int, str],
        Mapping[int, str],
        Mapping[int, Tuple[int, str]],
        set,
        set,
    ],
    physics_steps: int,
    settled: bool,
) -> float:
    clearance = _mapping(raw, label)
    _exact_keys(clearance, SAMPLE_CLEARANCE_FIELDS, label)
    _require(clearance.get("available") is True, label + ".available")
    _require(
        clearance.get("authority")
        == "exact_sample_to_obb_plus_certified_coverage_lower_bound",
        label + ".authority",
    )
    exact_distance = _finite_number(
        clearance.get("minimum_exact_sample_to_obb_distance_m"),
        label + ".minimum_exact_sample_to_obb_distance_m",
    )
    coverage_radius = _finite_number(
        clearance.get("certified_coverage_radius_m"),
        label + ".certified_coverage_radius_m",
    )
    _require(coverage_radius >= 0.0, label + ".certified_coverage_radius_m")
    lower_bound = _finite_number(
        clearance.get("full_surface_clearance_lower_bound_m"),
        label + ".full_surface_clearance_lower_bound_m",
    )
    _close(
        lower_bound,
        exact_distance - coverage_radius,
        label + ".full_surface_clearance_lower_bound_m",
    )
    sampling = _mapping(
        trace.get("full_robot_surface_sampling"),
        "trace.full_robot_surface_sampling",
    )
    _close(
        coverage_radius,
        sampling.get("maximum_surface_cover_radius_m"),
        label + ".certified_coverage_radius_m",
    )

    sample_id = _integer(clearance.get("sample_id"), label + ".sample_id")
    robot_geom = _integer(
        clearance.get("robot_geom_id"), label + ".robot_geom_id"
    )
    obstacle_geom = _integer(
        clearance.get("obstacle_geom_id"), label + ".obstacle_geom_id"
    )
    robot_geoms, obstacle_geoms = authority[0], authority[1]
    _require(
        robot_geom in robot_geoms and obstacle_geom in obstacle_geoms,
        label,
        "sample minimum is outside resolved robot/obstacle authority",
    )
    cursor = 0
    owning_geom: Optional[int] = None
    for index, raw_geom_record in enumerate(
        _list(sampling.get("geom_records"), "trace sampling geom records")
    ):
        geom_record = _mapping(raw_geom_record, "trace sampling geom record")
        geom_sample_count = _integer(
            geom_record.get("sample_count"),
            "trace sampling geom record[%d].sample_count" % index,
        )
        if cursor <= sample_id < cursor + geom_sample_count:
            owning_geom = _integer(
                geom_record.get("geom_id"),
                "trace sampling geom record[%d].geom_id" % index,
            )
            break
        cursor += geom_sample_count
    _require(
        owning_geom == robot_geom,
        label + ".sample_id",
        "is not owned by the reported robot geom",
    )

    observation = clearance.get("minimum_observation_index")
    if settled:
        _require(
            observation is None,
            label + ".minimum_observation_index",
            "must be null for the settled snapshot",
        )
    elif observation is not None:
        _require(
            _integer(observation, label + ".minimum_observation_index")
            < physics_steps,
            label + ".minimum_observation_index",
            "is outside completed physics exposure",
        )
    return lower_bound


def _validate_settled_motion(
    raw: Any,
    *,
    obstacle_bodies: Mapping[int, str],
) -> bool:
    label = "trace.outcome.monitor.record.settled_state.obstacle_motion_admissibility"
    motion = _mapping(raw, label)
    _exact_keys(motion, SETTLED_MOTION_FIELDS, label)
    _require(
        motion.get("reference")
        == "clone_forwarded_settled_state_mj_objectVelocity_world_orientation_rot_then_lin",
        label + ".reference",
    )
    linear_threshold = _finite_number(
        motion.get("max_linear_speed_threshold_m_per_s"),
        label + ".max_linear_speed_threshold_m_per_s",
    )
    angular_threshold = _finite_number(
        motion.get("max_angular_speed_threshold_rad_per_s"),
        label + ".max_angular_speed_threshold_rad_per_s",
    )
    _require(
        linear_threshold >= 0.0 and angular_threshold >= 0.0,
        label + " thresholds",
    )
    _close(
        linear_threshold,
        SETTLED_LINEAR_SPEED_THRESHOLD_M_PER_S,
        label + ".max_linear_speed_threshold_m_per_s",
    )
    _close(
        angular_threshold,
        SETTLED_ANGULAR_SPEED_THRESHOLD_RAD_PER_S,
        label + ".max_angular_speed_threshold_rad_per_s",
    )
    bodies = _list(motion.get("bodies"), label + ".bodies")
    observed_body_ids: List[int] = []
    linear_speeds: List[float] = []
    angular_speeds: List[float] = []
    exceeded_count = 0
    for index, raw_body in enumerate(bodies):
        body_label = "%s.bodies[%d]" % (label, index)
        body = _mapping(raw_body, body_label)
        _exact_keys(body, SETTLED_BODY_VELOCITY_FIELDS, body_label)
        body_id = _integer(body.get("body_id"), body_label + ".body_id")
        _require(
            body_id in obstacle_bodies
            and body.get("body_name") == obstacle_bodies[body_id],
            body_label,
            "is outside the resolved obstacle-body authority",
        )
        observed_body_ids.append(body_id)
        angular_vector = _finite_number_vector(
            body.get("angular_velocity_world_rad_per_s"),
            3,
            body_label + ".angular_velocity_world_rad_per_s",
        )
        linear_vector = _finite_number_vector(
            body.get("linear_velocity_world_m_per_s"),
            3,
            body_label + ".linear_velocity_world_m_per_s",
        )
        angular_speed = math.sqrt(sum(value * value for value in angular_vector))
        linear_speed = math.sqrt(sum(value * value for value in linear_vector))
        _close(
            body.get("angular_speed_rad_per_s"),
            angular_speed,
            body_label + ".angular_speed_rad_per_s",
        )
        _close(
            body.get("linear_speed_m_per_s"),
            linear_speed,
            body_label + ".linear_speed_m_per_s",
        )
        angular_exceeded = angular_speed > angular_threshold
        linear_exceeded = linear_speed > linear_threshold
        _require(
            body.get("angular_speed_exceeds_threshold") is angular_exceeded
            and body.get("linear_speed_exceeds_threshold") is linear_exceeded,
            body_label + " threshold flags",
        )
        exceeded_count += int(angular_exceeded) + int(linear_exceeded)
        angular_speeds.append(angular_speed)
        linear_speeds.append(linear_speed)
    _require(
        observed_body_ids == sorted(obstacle_bodies),
        label + ".bodies",
        "does not cover every resolved selected-obstacle body exactly once",
    )
    _close(
        motion.get("maximum_observed_linear_speed_m_per_s"),
        max(linear_speeds),
        label + ".maximum_observed_linear_speed_m_per_s",
    )
    _close(
        motion.get("maximum_observed_angular_speed_rad_per_s"),
        max(angular_speeds),
        label + ".maximum_observed_angular_speed_rad_per_s",
    )
    reasons = _list(motion.get("reasons"), label + ".reasons")
    _require(
        all(isinstance(reason, str) and bool(reason) for reason in reasons)
        and len(reasons) == exceeded_count,
        label + ".reasons",
        "does not match the threshold crossings",
    )
    admissible = exceeded_count == 0
    _require(
        motion.get("admissible") is admissible,
        label + ".admissible",
    )
    return admissible


def _validate_obstacle_drift(
    raw: Any,
    *,
    resolved: Mapping[str, Any],
    physics_steps: int,
) -> bool:
    label = "trace.outcome.monitor.record.obstacle_pose_drift"
    drift = _mapping(raw, label)
    _exact_keys(drift, OBSTACLE_DRIFT_FIELDS, label)
    _require(
        drift.get("reference")
        == "every_selected_obstacle_collision_geom_pose_after_settling",
        label + ".reference",
    )
    geom_ids = [
        _integer(value, "trace.resolved_geometry.obstacle_geom_ids")
        for value in _list(
            resolved.get("obstacle_geom_ids"),
            "trace.resolved_geometry.obstacle_geom_ids",
        )
    ]
    geom_names = _list(
        resolved.get("obstacle_geom_names"),
        "trace.resolved_geometry.obstacle_geom_names",
    )
    expected_geoms = dict(zip(geom_ids, geom_names))
    body_ids = [
        _integer(value, "trace.resolved_geometry.obstacle_body_ids")
        for value in _list(
            resolved.get("obstacle_body_ids"),
            "trace.resolved_geometry.obstacle_body_ids",
        )
    ]
    body_names = _list(
        resolved.get("obstacle_body_names"),
        "trace.resolved_geometry.obstacle_body_names",
    )
    expected_bodies = dict(zip(body_ids, body_names))
    observed_geom_ids: List[int] = []
    rows: List[Mapping[str, Any]] = []
    for index, raw_geom in enumerate(_list(drift.get("geoms"), label + ".geoms")):
        geom_label = "%s.geoms[%d]" % (label, index)
        geom = _mapping(raw_geom, geom_label)
        _exact_keys(geom, OBSTACLE_GEOM_DRIFT_FIELDS, geom_label)
        geom_id = _integer(geom.get("geom_id"), geom_label + ".geom_id")
        body_id = _integer(geom.get("body_id"), geom_label + ".body_id")
        _require(
            geom_id in expected_geoms
            and geom.get("geom_name") == expected_geoms[geom_id]
            and body_id in expected_bodies
            and geom.get("body_name") == expected_bodies[body_id],
            geom_label,
            "is outside the resolved obstacle authority",
        )
        observed_geom_ids.append(geom_id)
        rows.append(geom)
        for maximum_field, final_field in (
            ("maximum_translation_m", "final_translation_m"),
            ("maximum_rotation_rad", "final_rotation_rad"),
            (
                "maximum_surface_point_displacement_m",
                "final_surface_point_displacement_m",
            ),
        ):
            maximum = _finite_number(
                geom.get(maximum_field), geom_label + "." + maximum_field
            )
            final = _finite_number(
                geom.get(final_field), geom_label + "." + final_field
            )
            _require(
                0.0 <= final <= maximum,
                geom_label + "." + maximum_field,
            )
        for field in (
            "maximum_translation_observation_index",
            "maximum_rotation_observation_index",
            "maximum_surface_displacement_observation_index",
        ):
            _require(
                _integer(geom.get(field), geom_label + "." + field)
                < physics_steps,
                geom_label + "." + field,
            )
    _require(
        observed_geom_ids == sorted(expected_geoms),
        label + ".geoms",
        "does not cover every selected-obstacle geom exactly once",
    )
    maximum_bindings = {
        "maximum_translation_m": max(
            _finite_number(row["maximum_translation_m"], label)
            for row in rows
        ),
        "maximum_rotation_rad": max(
            _finite_number(row["maximum_rotation_rad"], label) for row in rows
        ),
        "maximum_surface_point_displacement_m": max(
            _finite_number(
                row["maximum_surface_point_displacement_m"], label
            )
            for row in rows
        ),
    }
    for field, expected in maximum_bindings.items():
        _close(drift.get(field), expected, label + "." + field)
    threshold = _finite_number(
        drift.get("surface_drift_threshold_m"),
        label + ".surface_drift_threshold_m",
    )
    _require(threshold >= 0.0, label + ".surface_drift_threshold_m")
    _close(
        threshold,
        OBSTACLE_SURFACE_DRIFT_THRESHOLD_M,
        label + ".surface_drift_threshold_m",
    )
    crossed = maximum_bindings["maximum_surface_point_displacement_m"] > threshold
    _require(
        drift.get("surface_drift_threshold_crossed") is crossed,
        label + ".surface_drift_threshold_crossed",
    )
    crossing = drift.get(
        "first_surface_drift_threshold_crossing_observation_index"
    )
    if crossed:
        _require(
            _integer(crossing, label + ".first_surface_drift_threshold_crossing_observation_index")
            < physics_steps,
            label + ".first_surface_drift_threshold_crossing_observation_index",
        )
    else:
        _require(
            crossing is None,
            label + ".first_surface_drift_threshold_crossing_observation_index",
        )
    return crossed


def _validate_contacts(
    result: Mapping[str, Any], outcome: Mapping[str, Any], physics_steps: int, trace: Mapping[str, Any]
) -> None:
    monitor = _mapping(outcome.get("monitor"), "trace.outcome.monitor")
    record = _mapping(monitor.get("record"), "trace.outcome.monitor.record")
    settled_record = _mapping(record.get("settled_state"), "trace.outcome.monitor.record.settled_state")
    authority = _contact_authority(trace)
    link_ids = authority[-1]
    settled_motion_admissible = _validate_settled_motion(
        settled_record.get("obstacle_motion_admissibility"),
        obstacle_bodies=authority[3],
    )
    _mapping(
        settled_record.get("raw_mj_geom_distance"),
        "trace.outcome.monitor.record.settled_state.raw_mj_geom_distance",
    )
    near_tolerance = None
    if physics_steps:
        near_tolerance = _finite_number(
            record.get("near_contact_tolerance_m"),
            "trace contact near_contact_tolerance_m",
        )
        _require(
            near_tolerance >= 0.0,
            "trace contact near_contact_tolerance_m",
            "must be nonnegative",
        )
    settled_candidates = _contact_ledger(
        settled_record.get("candidate_contact_point_records"),
        label="trace settled candidate contacts",
        expected_phase="settled_post_integration_recomputed",
        physics_steps=physics_steps,
        near_tolerance_m=near_tolerance,
        authority=authority,
    )
    settled_serialized = _contact_ledger(
        settled_record.get("physical_contact_point_records"),
        label="trace settled physical contacts",
        expected_phase="settled_post_integration_recomputed",
        physics_steps=physics_steps,
        near_tolerance_m=near_tolerance,
        authority=authority,
    )
    _require_contact_order_and_uniqueness(
        settled_candidates,
        settled=True,
        label="trace settled candidate contacts",
    )
    _require_contact_order_and_uniqueness(
        settled_serialized,
        settled=True,
        label="trace settled physical contacts",
    )
    _require(
        all(contact["is_physical_nonpositive_distance_contact"] for contact in settled_serialized),
        "trace settled physical contacts",
        "contains a nonphysical record",
    )
    settled = [
        contact
        for contact in settled_candidates
        if contact["is_physical_nonpositive_distance_contact"]
    ]
    _require(
        _canonical_equal(settled_serialized, settled),
        "trace settled physical contacts",
        "is not the distance-derived candidate-ledger subset",
    )
    settled_coverage_lower_bound = _validate_sample_clearance(
        settled_record.get("sample_clearance"),
        label="trace.outcome.monitor.record.settled_state.sample_clearance",
        trace=trace,
        authority=authority,
        physics_steps=physics_steps,
        settled=True,
    )
    settled_distances = [
        _finite(contact.get("contact_distance_m"), "trace settled contact distance")
        for contact in settled
    ]
    settled_d_sim = (
        min([settled_coverage_lower_bound, 0.0] + settled_distances)
        if settled_distances
        else settled_coverage_lower_bound
    )
    _close(
        settled_record.get("D_sim_m"),
        settled_d_sim,
        "trace settled D_sim recomputation",
    )
    _require(
        settled_record.get("contact_authority_clamped_D_sim")
        is bool(settled and settled_d_sim < settled_coverage_lower_bound),
        "trace settled contact-authority D_sim clamp flag",
    )
    if physics_steps:
        _require(
            _integer(
                record.get("observed_physics_substeps"),
                "trace contact observed substeps",
            )
            == physics_steps,
            "trace contact observed substeps",
        )
        _require(
            _canonical_equal(record.get("first_index"), [0, 0, 0]),
            "trace contact first_index",
            "differs from the first scheduled physics callback",
        )
        _require(
            _canonical_equal(
                record.get("last_index"),
                list(_coordinate_for_physics(physics_steps - 1)),
            ),
            "trace contact last_index",
            "differs from the terminal scheduled physics callback",
        )
        live_candidates = _contact_ledger(
            record.get("live_solver_phase_contact_point_records"),
            label="trace live-solver candidate contacts",
            expected_phase="live_solver_phase_preintegration_geometry",
            physics_steps=physics_steps,
            near_tolerance_m=near_tolerance,
            authority=authority,
        )
        post_candidates = _contact_ledger(
            record.get("post_state_candidate_contact_point_records"),
            label="trace post-state candidate contacts",
            expected_phase="post_integration_recomputed",
            physics_steps=physics_steps,
            near_tolerance_m=near_tolerance,
            authority=authority,
        )
        post_serialized = _contact_ledger(
            record.get("post_state_physical_contact_point_records"),
            label="trace post-state physical contacts",
            expected_phase="post_integration_recomputed",
            physics_steps=physics_steps,
            near_tolerance_m=near_tolerance,
            authority=authority,
        )
        _require_contact_order_and_uniqueness(
            live_candidates,
            settled=False,
            label="trace live-solver candidate contacts",
        )
        _require_contact_order_and_uniqueness(
            post_candidates,
            settled=False,
            label="trace post-state candidate contacts",
        )
        _require_contact_order_and_uniqueness(
            post_serialized,
            settled=False,
            label="trace post-state physical contacts",
        )
        live = [
            contact
            for contact in live_candidates
            if contact["is_physical_nonpositive_distance_contact"]
        ]
        post = [
            contact
            for contact in post_candidates
            if contact["is_physical_nonpositive_distance_contact"]
        ]
        _require(
            all(contact["is_physical_nonpositive_distance_contact"] for contact in post_serialized),
            "trace post-state physical contacts",
            "contains a nonphysical record",
        )
        _require(
            _canonical_equal(post_serialized, post),
            "trace post-state physical contacts",
            "is not the distance-derived candidate-ledger subset",
        )
        rollout_physical = sorted(live + post, key=_rollout_contact_order)
        all_physical = settled + rollout_physical
        rollout_candidates = sorted(
            live_candidates + post_candidates, key=_rollout_contact_order
        )
        all_candidates = settled_candidates + rollout_candidates
        count_bindings = {
            "total_candidate_contact_point_record_count": len(all_candidates),
            "total_physical_contact_point_record_count": len(all_physical),
            "rollout_phase_physical_contact_point_record_count": len(
                rollout_physical
            ),
            "live_solver_candidate_contact_point_record_count": len(
                live_candidates
            ),
            "live_solver_active_contact_point_record_count": sum(
                contact["solver_constraint_active"] for contact in live_candidates
            ),
            "live_solver_within_near_contact_tolerance_point_record_count": sum(
                contact["within_registered_near_contact_tolerance"]
                for contact in live_candidates
            ),
            "live_solver_nonpositive_contact_point_record_count": len(live),
            "post_state_candidate_contact_point_record_count": len(post_candidates),
            "post_state_solver_active_contact_point_record_count": sum(
                contact["solver_constraint_active"] for contact in post_candidates
            ),
            "post_state_within_near_contact_tolerance_point_record_count": sum(
                contact["within_registered_near_contact_tolerance"]
                for contact in post_candidates
            ),
            "post_state_physical_contact_point_record_count": len(post),
        }
        for field, expected in count_bindings.items():
            _require(
                _integer(record.get(field), "trace contact %s" % field) == expected,
                "trace contact %s" % field,
            )
        _require(record.get("any_robot_obstacle_contact") is bool(all_physical), "trace contact-any flag")
        _require(record.get("rollout_any_robot_obstacle_contact") is bool(rollout_physical), "trace rollout contact-any flag")
        _require(record.get("live_solver_any_robot_obstacle_contact") is bool(live), "trace live-solver contact-any flag")
        _require(record.get("post_state_any_robot_obstacle_contact") is bool(post), "trace post-state contact-any flag")
        _require(record.get("first_candidate_contact_point_record") == (all_candidates[0] if all_candidates else None), "trace first candidate contact")
        _require(record.get("first_live_solver_physical_contact_point_record") == (live[0] if live else None), "trace first live-solver physical contact")
        _require(record.get("first_post_state_physical_contact_point_record") == (post[0] if post else None), "trace first post-state physical contact")
        _require(record.get("physical_contact_distance_semantics") == "mujoco_contact_dist_le_0", "trace physical contact semantics")
        _require(
            record.get("D_sim_semantics")
            == "union_of_settled_live_solver_and_forwarded_post_state_nonpositive_contacts_plus_exact_obb_coverage_lower_bound",
            "trace D_sim semantics",
        )
        coverage_lower_bound = _validate_sample_clearance(
            record.get("sample_clearance"),
            label="trace.outcome.monitor.record.sample_clearance",
            trace=trace,
            authority=authority,
            physics_steps=physics_steps,
            settled=False,
        )
        contact_distances = [
            _finite(contact.get("contact_distance_m"), "trace physical contact distance")
            for contact in all_physical
        ]
        recomputed_d_sim = (
            min([coverage_lower_bound, 0.0] + contact_distances)
            if contact_distances
            else coverage_lower_bound
        )
        _close(record.get("D_sim_min_m"), recomputed_d_sim, "trace contact D_sim recomputation")
        _close(monitor.get("D_sim_min_m"), recomputed_d_sim, "trace.outcome.monitor.D_sim_min_m")
        _require(
            record.get("contact_authority_clamped_D_sim")
            is bool(all_physical and recomputed_d_sim < coverage_lower_bound),
            "trace contact-authority D_sim clamp flag",
        )
        drift_crossed = _validate_obstacle_drift(
            record.get("obstacle_pose_drift"),
            resolved=_mapping(trace.get("resolved_geometry"), "trace.resolved_geometry"),
            physics_steps=physics_steps,
        )
        drift = _mapping(record.get("obstacle_pose_drift"), "trace obstacle drift")
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
        _mapping(
            record.get("raw_mj_geom_distance"),
            "trace.outcome.monitor.record.raw_mj_geom_distance",
        )
    else:
        live_candidates = []
        post_candidates = []
        live = []
        post = []
        all_candidates = settled_candidates
        all_physical = settled
        _exact_keys(
            record,
            {"settled_state", "observed_physics_substeps"},
            "trace zero-exposure contact record",
        )
        _require(
            _integer(
                record.get("observed_physics_substeps"),
                "trace contact observed substeps",
            )
            == 0,
            "trace contact observed substeps",
        )
        _close(
            monitor.get("D_sim_min_m"),
            settled_d_sim,
            "trace.outcome.monitor.D_sim_min_m",
        )
        for field in (
            "translation_drift_m",
            "rotation_drift_rad",
            "surface_drift_m",
        ):
            _close(
                monitor.get(field),
                0.0,
                "trace.outcome.monitor.%s" % field,
            )

    settled_count_bindings = {
        "candidate_contact_point_record_count": len(settled_candidates),
        "solver_active_contact_point_record_count": sum(
            contact["solver_constraint_active"] for contact in settled_candidates
        ),
        "within_near_contact_tolerance_point_record_count": sum(
            contact["within_registered_near_contact_tolerance"]
            for contact in settled_candidates
        ),
        "physical_contact_point_record_count": len(settled),
    }
    for field, expected in settled_count_bindings.items():
        _require(
            _integer(settled_record.get(field), "trace settled %s" % field)
            == expected,
            "trace settled %s" % field,
        )
    _require(settled_record.get("any_robot_obstacle_contact") is bool(settled), "trace settled contact-any flag")
    _require(settled_record.get("first_candidate_contact_point_record") == (settled_candidates[0] if settled_candidates else None), "trace first settled candidate contact")
    _require(settled_record.get("first_physical_contact_point_record") == (settled[0] if settled else None), "trace first settled physical contact")
    total = len(settled) + len(live) + len(post)
    rollout = len(live) + len(post)
    monitor_count_bindings = {
        "total": total,
        "settled": len(settled),
        "rollout": rollout,
        "live": len(live),
        "post": len(post),
    }
    for field, expected in monitor_count_bindings.items():
        _require(
            _integer(monitor.get(field), "trace.outcome.monitor.%s" % field)
            == expected,
            "trace.outcome.monitor.%s" % field,
        )
    rollout_records = sorted(live + post, key=_rollout_contact_order)
    all_records = settled + rollout_records
    _require(monitor.get("any_contact") is bool(all_records), "trace.outcome.monitor.any_contact")
    _require(monitor.get("first_settled") == (settled[0] if settled else None), "trace.outcome.monitor.first_settled")
    _require(monitor.get("first_live") == (live[0] if live else None), "trace.outcome.monitor.first_live")
    _require(monitor.get("first_post") == (post[0] if post else None), "trace.outcome.monitor.first_post")
    expected_first = all_records[0] if all_records else None
    _require(monitor.get("first") == expected_first, "trace.outcome.monitor.first")

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
    validity = _mapping(outcome.get("validity"), "trace.outcome.validity")
    _require(
        validity.get("coverage_audit_passed") is True,
        "trace.outcome.validity.coverage_audit_passed",
    )
    _close(
        validity.get("obstacle_translation_drift_m"),
        monitor.get("translation_drift_m"),
        "trace.outcome.validity.obstacle_translation_drift_m",
    )
    _close(
        validity.get("obstacle_rotation_drift_rad"),
        monitor.get("rotation_drift_rad"),
        "trace.outcome.validity.obstacle_rotation_drift_rad",
    )
    recomputed_static_admissible = bool(
        settled_motion_admissible
        and _finite(monitor.get("translation_drift_m"), "trace translation drift")
        <= OBSTACLE_TRANSLATION_DRIFT_THRESHOLD_M
        and _finite(monitor.get("rotation_drift_rad"), "trace rotation drift")
        <= OBSTACLE_ROTATION_DRIFT_THRESHOLD_RAD
        and _finite(monitor.get("surface_drift_m"), "trace surface drift")
        <= OBSTACLE_SURFACE_DRIFT_THRESHOLD_M
    )
    _require(
        validity.get("static_admissible") is recomputed_static_admissible,
        "trace.outcome.validity.static_admissible",
        "differs from the settled-motion and registered drift limits",
    )
    if result.get("arm") == "joint_velocity_psf_link56" and result.get(
        "completion_class"
    ) == "executed":
        _require(
            settled_motion_admissible
            and validity.get("static_admissible") is True
            and not drift_crossed,
            "trace executed PSF static-field admissibility",
        )


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
        _require(
            not solved_rows
            and not failed_optimizer_rows
            and all("qp" in row and row.get("qp") is None for row in inner_rows),
            "adapter-only optimizer trace",
        )
    elif result.get("completion_class") == "executed":
        _require(
            len(solved_rows) == len(inner_rows)
            and not failed_optimizer_rows,
            "trace executed PSF optimizer schedule",
            "must contain one solved QP for every issued 100 Hz command",
        )
        _require(
            _finite_number(
                initial.get("D_opt_min_m"),
                "trace.outcome.initial_protected_sample_audit.D_opt_min_m",
            )
            > 0.0,
            "trace.outcome.initial_protected_sample_audit.D_opt_min_m",
            "must be positive for an executed PSF arm",
        )
        for index, row in enumerate(inner_rows):
            _require(
                _finite_number(
                    row.get("D_opt_min_m"),
                    "trace.outcome.inner_trace[%d].D_opt_min_m" % index,
                )
                > 0.0,
                "trace.outcome.inner_trace[%d].D_opt_min_m" % index,
                "must be positive for an issued PSF command",
            )
            _require(
                _finite_number(
                    row.get("minimum_h_m2"),
                    "trace.outcome.inner_trace[%d].minimum_h_m2" % index,
                )
                > 0.0,
                "trace.outcome.inner_trace[%d].minimum_h_m2" % index,
                "must be positive for an issued PSF command",
            )
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
        _exact_keys(qp, SOLVED_QP_DIAGNOSTIC_FIELDS, "trace QP diagnostics")
        status = qp.get("status")
        _require(
            qp.get("solver") == "osqp"
            and status in ("solved", "solved inaccurate")
            and qp.get("status_value")
            == {"solved": 1, "solved inaccurate": 2}[status],
            "trace QP solver status",
        )
        iterations = _integer(qp.get("iterations"), "trace QP iterations")
        _require(
            0 <= iterations <= JOINT_VELOCITY_QP_MAX_ITERATIONS,
            "trace QP iterations",
            "cross the registered OSQP iteration limit",
        )
        _require(
            _finite_number(qp.get("solve_time_seconds"), "trace QP solve time")
            >= 0.0,
            "trace QP solve time",
        )
        input_count = _integer(
            qp.get("input_constraint_count"), "trace QP input constraint count"
        )
        solved_count = _integer(
            qp.get("solved_constraint_count"), "trace QP solved constraint count"
        )
        trivial_count = _integer(
            qp.get("trivial_zero_constraint_count"),
            "trace QP trivial constraint count",
        )
        _require(
            input_count == sample_count
            and solved_count > 0
            and trivial_count >= 0
            and solved_count + trivial_count == input_count,
            "trace QP constraint population",
            "differs from the exhaustive protected-sample population",
        )
        minimum_scale = _finite_number(
            qp.get("minimum_nonzero_row_scale_m2_per_rad"),
            "trace QP minimum nonzero row scale",
        )
        maximum_scale = _finite_number(
            qp.get("maximum_nonzero_row_scale_m2_per_rad"),
            "trace QP maximum nonzero row scale",
        )
        _require(
            0.0 < minimum_scale <= maximum_scale,
            "trace QP nonzero row scales",
        )
        normalized_residual = _finite_number(
            qp.get("minimum_normalized_cbf_residual"),
            "trace QP minimum normalized CBF residual",
        )
        _finite_number(
            qp.get("nominal_minimum_normalized_cbf_residual"),
            "trace QP nominal minimum normalized CBF residual",
        )
        _require(
            normalized_residual >= -JOINT_VELOCITY_QP_POSTCHECK_TOLERANCE,
            "trace QP minimum normalized CBF residual",
            "crosses the registered hard postcheck tolerance",
        )
        _require(
            0.0
            <= _finite_number(
                qp.get("maximum_velocity_bound_violation_rad_s"),
                "trace QP maximum velocity-bound violation",
            )
            <= JOINT_VELOCITY_BOUND_POSTCHECK_TOLERANCE_RAD_S,
            "trace QP maximum velocity-bound violation",
            "crosses the registered hard postcheck tolerance",
        )
        _boolean(qp.get("nominal_feasible"), "trace QP nominal_feasible")
        coordinate = _row_coordinate(row, physics=False, label="trace solved QP row")
        command_index = coordinate[0] * INNER_UPDATES_PER_HIGH_LEVEL + coordinate[1]
        nominal_command = _vector(
            _mapping(
                _list(outcome.get("nominal_ledger"), "trace nominal ledger")[command_index],
                "trace nominal ledger row",
            ).get("qdot_rad_s"),
            7,
            "trace nominal ledger qdot",
        )
        executed_command = _vector(
            _mapping(
                _list(outcome.get("executed_ledger"), "trace executed ledger")[command_index],
                "trace executed ledger row",
            ).get("qdot_rad_s"),
            7,
            "trace executed ledger qdot",
        )
        _close(
            qp.get("correction_l2_rad_s"),
            _norm(
                [
                    safe - nominal
                    for safe, nominal in zip(executed_command, nominal_command)
                ]
            ),
            "trace QP correction_l2_rad_s",
        )
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
        if result.get("completion_class") == "executed":
            executed_realized_fields = {
                "high_level_index",
                "inner_control_index",
                "physics_substep_index",
                "D_opt_min_m",
                "query_count",
                "valid",
                "minimum_h_m2",
                "minimum_realized_cbf_residual_m2_per_s",
            }
            for index, row in enumerate(realized_rows):
                label = "trace realized row[%d]" % index
                _exact_keys(row, executed_realized_fields, label)
                _require(row.get("valid") is True, label + ".valid")
                _require(
                    _integer(row.get("query_count"), label + ".query_count")
                    == sample_count,
                    label + ".query_count",
                    "differs from the exhaustive protected-sample population",
                )
                _finite_number(row.get("D_opt_min_m"), label + ".D_opt_min_m")
                _require(
                    _finite_number(row.get("D_opt_min_m"), label + ".D_opt_min_m")
                    > 0.0,
                    label + ".D_opt_min_m",
                    "must be positive for an executed PSF arm",
                )
                _require(
                    _finite_number(
                        row.get("minimum_h_m2"), label + ".minimum_h_m2"
                    )
                    > 0.0,
                    label + ".minimum_h_m2",
                )
                _require(
                    _finite_number(
                        row.get("minimum_realized_cbf_residual_m2_per_s"),
                        label + ".minimum_realized_cbf_residual_m2_per_s",
                    )
                    >= 0.0,
                    label + ".minimum_realized_cbf_residual_m2_per_s",
                    "would have terminated the executed arm",
                )
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
    if arm == "joint_velocity_psf_link56" and result.get("completion_class") == "executed":
        _require(
            evaluations == physics_steps * sample_count
            and invalid_queries == 0
            and nonpositive_queries == 0
            and not negative_rows
            and audit.get("first_negative_residual") is None,
            "trace executed PSF realized-CBF exposure",
            "is not one valid exhaustive sample evaluation per physics callback",
        )

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


def validate_active_arm_trace(
    result: Mapping[str, Any],
    trace: Mapping[str, Any],
    *,
    expected_source_action_count: int = REGISTERED_SOURCE_ACTION_COUNT,
) -> None:
    """Independently reconstruct one compact result from its raw arm trace."""

    outcome = _validate_identity(result, trace)
    _validate_trace_configuration_bindings(
        result,
        trace,
        expected_source_action_count=expected_source_action_count,
    )
    _validate_full_robot_surface_sampling(trace)
    high_steps, _inner_steps, physics_steps, _entered, completed = _validate_execution(result, outcome)
    _validate_eef_reference_site(outcome)
    _validate_motion_and_tracking(result, outcome, physics_steps)
    _validate_car(result, outcome, completed)
    _validate_task(result, outcome, high_steps, completed)
    _validate_contacts(result, outcome, physics_steps, trace)
    _validate_optimizer_and_realized(result, outcome, physics_steps)


def _artifact_root_for_result(
    result_path: Path,
    result: Mapping[str, Any],
    artifact_root: Optional[Path],
) -> Path:
    if artifact_root is not None:
        return artifact_root.absolute()
    if (
        result_path.parent.name == result.get("arm")
        and result_path.parent.parent.name == "arms"
    ):
        # Preserve compatibility with the earlier ``<run>/arms/<arm>`` test
        # and development layout.
        return result_path.parent.parent.parent.absolute()
    if (
        result_path.parent.name == result.get("arm")
        and result_path.parent.parent.name == result.get("case_id")
    ):
        # The current runner publishes ``<run>/<case>/<arm>/result.json`` and
        # keeps every trace reference relative to ``<run>``.
        return result_path.parent.parent.parent.absolute()
    return result_path.parent.absolute()


def _load_active_arm_trace(
    result: Mapping[str, Any], root: Path
) -> Mapping[str, Any]:
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
    return load_hashed_json(trace_candidates[0])


def _validate_run_artifacts_with_trace(
    result_path: Path,
    artifact_root: Optional[Path] = None,
    *,
    expected_source_action_count: int = REGISTERED_SOURCE_ACTION_COUNT,
) -> Tuple[Mapping[str, Any], Mapping[str, Any]]:
    result = load_hashed_json(result_path)
    validate_episode_result(result)
    root = _artifact_root_for_result(result_path, result, artifact_root)
    trace = _load_active_arm_trace(result, root)
    validate_active_arm_trace(
        result,
        trace,
        expected_source_action_count=expected_source_action_count,
    )
    return result, trace


def validate_run_artifacts(
    result_path: Path,
    artifact_root: Optional[Path] = None,
    *,
    expected_source_action_count: int = REGISTERED_SOURCE_ACTION_COUNT,
) -> Mapping[str, Any]:
    """Load and deeply validate a final result and its active-arm trace."""

    result, _trace = _validate_run_artifacts_with_trace(
        result_path,
        artifact_root,
        expected_source_action_count=expected_source_action_count,
    )
    return result


def validate_complete_run_artifacts(
    run_root: Path,
    *,
    candidate_receipt: Optional[Mapping[str, Any]] = None,
    expected_source_action_count: int = REGISTERED_SOURCE_ACTION_COUNT,
) -> Mapping[str, Any]:
    """Deeply validate both arms, the recomputed pair, and the run receipt.

    ``candidate_receipt`` supports the producer's prepublication gate; it must
    already carry its canonical payload hash.  With no candidate, the receipt
    is loaded from the immutable run root.

    This establishes internal consistency of the published run tree.  The v2
    layout does not embed its upstream manifests, protocols, checkpoint,
    historical source, run contract, or H100 prerequisite payloads, so those
    external artifacts still require separate byte-level validation before
    scientific interpretation.
    """

    root = run_root.absolute()
    _require(root.is_dir(), "run_root", "must be an existing directory")
    _require(not root.is_symlink(), "run_root", "must not be a symbolic link")
    if candidate_receipt is None:
        receipt_path = _real_child_path(root, "run_receipt.json", "run_receipt.json")
        receipt = load_hashed_json(receipt_path)
    else:
        receipt = dict(_mapping(candidate_receipt, "candidate run receipt"))
        verify_payload_hash(receipt)
    _exact_keys(receipt, RUN_RECEIPT_FIELDS, "run_receipt")
    _require(
        receipt.get("schema_version") == RUN_RECEIPT_SCHEMA_VERSION,
        "run_receipt.schema_version",
    )
    _require(receipt.get("status") == "complete", "run_receipt.status")
    _require(
        receipt.get("scientific_result") is False,
        "run_receipt.scientific_result",
    )
    _require(
        receipt.get("staged_scope")
        == "two_arms_one_canary_not_four_arm_109_case_study",
        "run_receipt.staged_scope",
    )
    elapsed_seconds = _finite_number(
        receipt.get("elapsed_seconds"), "run_receipt.elapsed_seconds"
    )
    _require(
        elapsed_seconds >= 0.0,
        "run_receipt.elapsed_seconds",
        "must be nonnegative",
    )
    identity = _mapping(receipt.get("identity"), "run_receipt.identity")
    _exact_keys(identity, RUN_RECEIPT_IDENTITY_FIELDS, "run_receipt.identity")
    run_id = _safe_path_component(identity.get("run_id"), "run_receipt.identity.run_id")
    case_id = _safe_path_component(
        identity.get("case_id"), "run_receipt.identity.case_id"
    )
    _require(
        receipt.get("run_id") == run_id,
        "run_receipt.run_id",
        "differs from identity",
    )
    _require(
        receipt.get("case_id") == case_id,
        "run_receipt.case_id",
        "differs from identity",
    )
    _require(
        root.name == run_id,
        "run_root",
        "basename differs from the immutable run ID",
    )
    code_commit = identity.get("code_commit")
    _require(
        isinstance(code_commit, str)
        and len(code_commit) == 40
        and all(character in "0123456789abcdef" for character in code_commit),
        "run_receipt.identity.code_commit",
        "must be a lowercase 40-character Git commit",
    )
    digest_fields = RUN_RECEIPT_IDENTITY_FIELDS - {"run_id", "case_id", "code_commit"}
    for field in sorted(digest_fields):
        _sha256_string(identity.get(field), "run_receipt.identity.%s" % field)
    run_contract_sha256 = identity["run_contract_sha256"]

    inventory = _mapping(receipt.get("arm_results"), "run_receipt.arm_results")
    _require(
        set(inventory) == set(ACTIVE_ARMS),
        "run_receipt.arm_results",
        "must inventory exactly the two registered arms",
    )
    results: Dict[str, Mapping[str, Any]] = {}
    traces: Dict[str, Mapping[str, Any]] = {}
    for arm in ACTIVE_ARMS:
        record = _mapping(inventory.get(arm), "run_receipt.arm_results.%s" % arm)
        _exact_keys(
            record,
            RUN_RECEIPT_ARM_RECORD_FIELDS,
            "run_receipt.arm_results.%s" % arm,
        )
        expected_relative = "%s/%s/result.json" % (case_id, arm)
        _require(
            record.get("relative_path") == expected_relative,
            "run_receipt.arm_results.%s.relative_path" % arm,
        )
        expected_file_sha256 = _sha256_string(
            record.get("sha256"), "run_receipt.arm_results.%s.sha256" % arm
        )
        result_path = _real_child_path(
            root, expected_relative, "run_receipt.arm_results.%s" % arm
        )
        _require(
            expected_file_sha256 == sha256_file(result_path),
            "run_receipt.arm_results.%s.sha256" % arm,
        )
        result, trace = _validate_run_artifacts_with_trace(
            result_path,
            root,
            expected_source_action_count=expected_source_action_count,
        )
        _require(result.get("run_id") == run_id, "result.%s.run_id" % arm)
        _require(result.get("case_id") == case_id, "result.%s.case_id" % arm)
        _require(result.get("arm") == arm, "result.%s.arm" % arm)
        trace_references = [
            _mapping(reference, "result.%s.artifact reference" % arm)
            for reference in _list(
                result.get("artifact_references"),
                "result.%s.artifact_references" % arm,
            )
            if isinstance(reference, dict)
            and reference.get("artifact_type") == TRACE_ARTIFACT_TYPE
        ]
        _require(
            len(trace_references) == 1
            and trace_references[0].get("relative_path")
            == "%s/%s/trace.json" % (case_id, arm),
            "result.%s active-arm trace path" % arm,
            "differs from the canonical producer layout",
        )
        provenance = _mapping(result.get("provenance"), "result.%s.provenance" % arm)
        _require(
            provenance.get("run_contract_sha256") == run_contract_sha256,
            "result.%s.provenance.run_contract_sha256" % arm,
        )
        _require(
            provenance.get("code_commit") == code_commit
            and provenance.get("code_dirty") is False,
            "result.%s.provenance.code" % arm,
        )
        runtime = _mapping(result.get("runtime"), "result.%s.runtime" % arm)
        model = _mapping(runtime.get("model"), "result.%s.runtime.model" % arm)
        pairing = _mapping(result.get("pairing"), "result.%s.pairing" % arm)
        cross_bindings = {
            "manifest_sha256": provenance.get("manifest_sha256"),
            "manifest_record_sha256": provenance.get("manifest_record_sha256"),
            "selection_config_sha256": provenance.get("protocol_config_sha256"),
            "runtime_protocol_raw_sha256": provenance.get(
                "runtime_protocol_raw_sha256"
            ),
            "runtime_protocol_semantic_sha256": provenance.get(
                "runtime_protocol_semantic_sha256"
            ),
            "runtime_parameter_block_sha256": runtime.get(
                "protocol_parameter_block_sha256"
            ),
            "checkpoint_tree_sha256": model.get("checkpoint_sha256"),
            "source_action_ledger_sha256": pairing.get(
                "nominal_high_level_action_ledger_sha256"
            ),
        }
        for field, observed in cross_bindings.items():
            _require(
                observed == identity[field],
                "result.%s.%s" % (arm, field),
                "differs from the run receipt identity",
            )
        source_replay = _mapping(
            trace.get("source_replay"), "trace.%s.source_replay" % arm
        )
        _require(
            source_replay.get("case_id") == case_id,
            "trace.%s.source_replay.case_id" % arm,
        )
        trace_source_bindings = {
            "historical_result_file_sha256": source_replay.get(
                "historical_result_file_sha256"
            ),
            "historical_result_payload_sha256": source_replay.get(
                "historical_result_payload_sha256"
            ),
            "source_action_ledger_sha256": source_replay.get(
                "executed_sequence_sha256"
            ),
        }
        for field, observed in trace_source_bindings.items():
            _require(
                observed == identity[field],
                "trace.%s.source_replay.%s" % (arm, field),
                "differs from the run receipt identity",
            )
        _require(
            source_replay.get("action_count")
            == pairing.get("source_exposure_high_level_steps"),
            "trace.%s.source_replay.action_count" % arm,
            "differs from the arm source exposure",
        )
        results[arm] = result
        traces[arm] = trace

    adapter_trace = traces["joint_velocity_adapter_only"]
    psf_trace = traces["joint_velocity_psf_link56"]
    for field in (
        "source_replay",
        "field_bundle_hashes",
        "field_bundle_diagnostics",
        "resolved_geometry",
        "full_robot_surface_sampling",
    ):
        _require(
            _canonical_equal(adapter_trace.get(field), psf_trace.get(field)),
            "paired trace.%s" % field,
            "differs between the adapter-only and PSF arms",
        )
    adapter_outcome = _mapping(adapter_trace.get("outcome"), "adapter trace outcome")
    psf_outcome = _mapping(psf_trace.get("outcome"), "PSF trace outcome")
    for field in (
        "selected_initial_state_sha256",
        "active_initial_observation_sha256",
        "restore",
        "initial_protected_sample_audit",
        "eef_reference_site",
    ):
        _require(
            _canonical_equal(adapter_outcome.get(field), psf_outcome.get(field)),
            "paired trace.outcome.%s" % field,
            "differs between the matched arms",
        )
    adapter_settled = _mapping(
        _mapping(
            _mapping(adapter_outcome.get("monitor"), "adapter outcome monitor").get(
                "record"
            ),
            "adapter monitor record",
        ).get("settled_state"),
        "adapter settled monitor record",
    )
    psf_settled = _mapping(
        _mapping(
            _mapping(psf_outcome.get("monitor"), "PSF outcome monitor").get(
                "record"
            ),
            "PSF monitor record",
        ).get("settled_state"),
        "PSF settled monitor record",
    )
    _require(
        _canonical_equal(adapter_settled, psf_settled),
        "paired trace.outcome.monitor.record.settled_state",
        "differs between the matched arms",
    )

    pair_relative = "%s/pair_result.json" % case_id
    _require(
        receipt.get("pair_result_relative_path") == pair_relative,
        "run_receipt.pair_result_relative_path",
    )
    pair_file_sha256 = _sha256_string(
        receipt.get("pair_result_sha256"), "run_receipt.pair_result_sha256"
    )
    pair_path = _real_child_path(root, pair_relative, "run_receipt.pair_result")
    _require(
        pair_file_sha256 == sha256_file(pair_path),
        "run_receipt.pair_result_sha256",
    )
    pair = load_hashed_json(pair_path)
    expected_pair = dict(
        validate_active_canary_pair(
            results["joint_velocity_adapter_only"],
            results["joint_velocity_psf_link56"],
        )
    )
    expected_pair.update(
        {
            "status": "complete",
            "scientific_result": False,
            "four_arm_109_case_study_complete": False,
            "run_contract_sha256": run_contract_sha256,
        }
    )
    pair_without_hash = {
        key: value for key, value in pair.items() if key != "result_payload_sha256"
    }
    _require(
        _canonical_equal(pair_without_hash, expected_pair),
        "pair_result",
        "differs from the two deeply validated arm results",
    )
    return {
        "receipt": receipt,
        "results": results,
        "traces": traces,
        "pair": pair,
        "identity_fields_not_projected_inside_run": sorted(
            IDENTITY_FIELDS_NOT_PROJECTED_INSIDE_RUN
        ),
        "external_artifacts_required_for_independent_acceptance": list(
            EXTERNAL_ARTIFACTS_REQUIRED_FOR_INDEPENDENT_ACCEPTANCE
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", nargs="?", type=Path)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument(
        "--expected-source-action-count",
        type=int,
        default=REGISTERED_SOURCE_ACTION_COUNT,
    )
    arguments = parser.parse_args()
    if arguments.expected_source_action_count < 1:
        parser.error("--expected-source-action-count must be positive")
    if (arguments.result is None) == (arguments.run_root is None):
        parser.error("provide exactly one result path or --run-root")
    if arguments.run_root is not None and arguments.artifact_root is not None:
        parser.error("--artifact-root is only valid with one result path")
    try:
        if arguments.run_root is not None:
            complete = validate_complete_run_artifacts(
                arguments.run_root,
                expected_source_action_count=arguments.expected_source_action_count,
            )
            receipt = complete["receipt"]
            pair = complete["pair"]
            output = {
                "status": "valid",
                "run_id": receipt["run_id"],
                "case_id": receipt["case_id"],
                "arms": list(ACTIVE_ARMS),
                "arm_trace_deep_validation": "passed",
                "pair_recomputation": "passed",
                "run_receipt_inventory": "passed",
                "D_sim_coverage_component_validation": D_SIM_COVERAGE_VALIDATION,
                "identity_fields_not_projected_inside_run": complete[
                    "identity_fields_not_projected_inside_run"
                ],
                "external_artifacts_required_for_independent_acceptance": complete[
                    "external_artifacts_required_for_independent_acceptance"
                ],
                "pair_result_payload_sha256": pair["result_payload_sha256"],
                "run_receipt_payload_sha256": receipt["result_payload_sha256"],
            }
        else:
            result = validate_run_artifacts(
                arguments.result,
                arguments.artifact_root,
                expected_source_action_count=arguments.expected_source_action_count,
            )
            output = {
                "status": "valid",
                "case_id": result["case_id"],
                "arm": result["arm"],
                "completion_class": result["completion_class"],
                "result_payload_sha256": result["result_payload_sha256"],
                "active_arm_trace_deep_validation": "passed",
                "D_sim_coverage_component_validation": D_SIM_COVERAGE_VALIDATION,
            }
    except (ArtifactContractError, KeyError, TypeError, ValueError, OverflowError) as error:
        print("INVALID: %s" % error, file=sys.stderr)
        return 1
    print(json.dumps(output, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
