"""Pure validation for the Stage-13 one-filter-interval counterfactual.

This module intentionally imports neither NumPy nor MuJoCo.  The allocation
producer records raw ledgers; this consumer reconstructs their deterministic
timing, arithmetic, hashes, and outcome classification after JSON loading.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


class OneStepCounterfactualError(RuntimeError):
    """Raised when a Stage-13 result is incomplete or self-inconsistent."""


PROTOCOL_SCHEMA = "vlsa_poisson_one_step_counterfactual_protocol.v1"
RESULT_SCHEMA = "vlsa_poisson_one_step_counterfactual_result.v1"
DIFFERENTIAL_AUDIT_SCHEMA = (
    "vlsa_poisson_protected_sample_differential_audit.v1"
)
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
INVALID_QUERY_REASONS = frozenset(
    (
        "invalid_point_shape",
        "nonfinite_point",
        "outside_grid",
        "invalid_cell",
        "nondifferentiable_internal_face",
        "nonregular_zero_cell",
        "nonfinite_cell_values",
    )
)
TOP_LEVEL_KEYS = frozenset(
    (
        "schema_version",
        "status",
        "scientific_result",
        "run_id",
        "case_id",
        "execution",
        "protocol_binding",
        "authority",
        "provenance",
        "source_boundary",
        "nominal_velocity_estimate",
        "boundary_B_filter",
        "arms",
        "diagnostics",
        "counts",
        "classification",
        "passed",
        "protocol_binding_sha256",
        "authority_sha256",
        "provenance_sha256",
        "nominal_velocity_estimate_sha256",
        "boundary_B_filter_sha256",
        "arm_ledger_sha256",
        "classification_ledger_sha256",
        "result_payload_sha256",
    )
)

EXECUTION_KEYS = (
    "outcome_kind",
    "source_prefix_complete",
    "qp_executed",
    "paired_joint_velocity_physics_executed",
    "no_hidden_clipping",
)
JOINT_VELOCITY_CONTROLLER_CONTRACT_KEYS = (
    "controller_class_module",
    "controller_class_qualname",
    "controller_implementation_file_sha256",
    "controller_configuration_file_sha256",
    "panda_robot_xml_file_sha256",
    "controller_name",
    "environment_action_dim",
    "arm_control_dim",
    "control_frequency_hz",
    "control_timestep_s",
    "physics_timestep_s",
    "physics_substeps_per_control",
    "normalized_input_lower",
    "normalized_input_upper",
    "physical_output_lower_rad_s",
    "physical_output_upper_rad_s",
    "velocity_gain_kp",
    "velocity_gain_ki",
    "velocity_gain_kd",
    "controller_policy_frequency_hz",
    "velocity_limits_rad_s",
    "controller_joint_index",
    "arm_qpos_indexes",
    "arm_qvel_indexes",
    "arm_joint_ids",
    "arm_joint_names",
    "arm_actuator_ids",
    "arm_actuator_names",
    "arm_actuator_lower",
    "arm_actuator_upper",
    "compiled_arm_actuator_ctrlrange",
    "interpolator",
)


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as error:
        raise OneStepCounterfactualError(
            "artifact contains a non-canonical or non-finite value"
        ) from error


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise OneStepCounterfactualError("%s must be an object" % label)
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, (list, tuple)):
        raise OneStepCounterfactualError("%s must be an array" % label)
    return value


def _exact_keys(value: Any, keys: Sequence[str], label: str) -> Mapping[str, Any]:
    output = _mapping(value, label)
    if set(output) != set(keys):
        missing = sorted(set(keys) - set(output))
        extra = sorted(set(output) - set(keys))
        raise OneStepCounterfactualError(
            "%s has invalid keys (missing=%s extra=%s)" % (label, missing, extra)
        )
    return output


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise OneStepCounterfactualError("%s must be boolean" % label)
    return value


def _integer(value: Any, label: str, minimum: Optional[int] = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise OneStepCounterfactualError("%s must be an integer" % label)
    output = int(value)
    if minimum is not None and output < minimum:
        raise OneStepCounterfactualError(
            "%s must be at least %d" % (label, minimum)
        )
    return output


def _number(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise OneStepCounterfactualError("%s must be finite" % label)
    return float(value)


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise OneStepCounterfactualError(
            "%s must be one lowercase SHA-256 digest" % label
        )
    return value


def _vector(value: Any, length: int, label: str) -> List[float]:
    sequence = _sequence(value, label)
    if len(sequence) != length:
        raise OneStepCounterfactualError(
            "%s must have length %d" % (label, length)
        )
    return [_number(item, "%s[%d]" % (label, index)) for index, item in enumerate(sequence)]


def _nullable_vector(value: Any, length: int, label: str) -> Optional[List[float]]:
    if value is None:
        return None
    return _vector(value, length, label)


def _matrix(value: Any, rows: int, columns: int, label: str) -> List[List[float]]:
    sequence = _sequence(value, label)
    if len(sequence) != rows:
        raise OneStepCounterfactualError(
            "%s must have %d rows" % (label, rows)
        )
    return [
        _vector(row, columns, "%s[%d]" % (label, index))
        for index, row in enumerate(sequence)
    ]


def _close(left: Any, right: Any, label: str, tolerance: float = 1e-12) -> None:
    lhs = _number(left, label + " observed")
    rhs = _number(right, label + " expected")
    if not math.isclose(lhs, rhs, rel_tol=0.0, abs_tol=tolerance):
        raise OneStepCounterfactualError("%s does not reconstruct" % label)


def _same(left: Any, right: Any, label: str) -> None:
    if _canonical(left) != _canonical(right):
        raise OneStepCounterfactualError("%s differs" % label)


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise OneStepCounterfactualError("dot-product dimensions differ")
    return float(sum(float(a) * float(b) for a, b in zip(left, right)))


def _norm(vector: Sequence[float]) -> float:
    return math.sqrt(sum(float(value) * float(value) for value in vector))


def _float64_array_sha256(values: Sequence[float]) -> str:
    """Reconstruct the repository's domain-separated 1-D float64 hash."""

    numbers = [_number(value, "float64 array element") for value in values]
    header = json.dumps(
        {"dtype": "<f8", "shape": [len(numbers)]},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(b"vlsa-table1-array-v1\0")
    digest.update(header)
    digest.update(b"\0")
    for number in numbers:
        digest.update(struct.pack("<d", number))
    return digest.hexdigest()


def _validate_section_hash(container: Mapping[str, Any], field: str, value: Any) -> str:
    expected = _sha(value)
    observed = _sha256(container.get(field), field)
    if observed != expected:
        raise OneStepCounterfactualError("%s does not bind its section" % field)
    return observed


def _validate_float_array_record(
    record: Mapping[str, Any], value_field: str, hash_field: str, label: str
) -> None:
    values = _sequence(record.get(value_field), label + "." + value_field)
    observed = _sha256(record.get(hash_field), label + "." + hash_field)
    if observed != _float64_array_sha256(values):
        raise OneStepCounterfactualError(
            "%s.%s does not reconstruct" % (label, hash_field)
        )


def _validate_physical_model_contract(value: Any, label: str) -> Mapping[str, Any]:
    contract = _exact_keys(
        value,
        (
            "schema_version",
            "sha256",
            "field_count",
            "option_field_count",
            "compiled_mjb_sha256",
            "compiled_mjb_bytes",
            "nq",
            "nv",
            "na",
            "mjstate_integration_size",
            "robosuite_flattened_state_size",
            "robosuite_flattened_state_layout",
        ),
        label,
    )
    if contract["schema_version"] != "vlsa_poisson_physical_model.v3":
        raise OneStepCounterfactualError("%s schema differs" % label)
    _sha256(contract["sha256"], label + ".sha256")
    _sha256(contract["compiled_mjb_sha256"], label + ".compiled_mjb_sha256")
    for field in ("field_count", "option_field_count", "compiled_mjb_bytes"):
        _integer(contract[field], label + "." + field, 1)
    nq = _integer(contract["nq"], label + ".nq", 1)
    nv = _integer(contract["nv"], label + ".nv", 1)
    na = _integer(contract["na"], label + ".na", 0)
    integration_size = _integer(
        contract["mjstate_integration_size"],
        label + ".mjstate_integration_size",
        1,
    )
    flattened_size = _integer(
        contract["robosuite_flattened_state_size"],
        label + ".robosuite_flattened_state_size",
        1,
    )
    common_size = 1 + nq + nv + na
    if (
        integration_size < common_size
        or flattened_size != common_size
        or contract["robosuite_flattened_state_layout"]
        != "time_qpos_qvel_act_no_udd_tail"
    ):
        raise OneStepCounterfactualError("%s state layout differs" % label)
    return contract


def _validate_joint_velocity_controller_contract(
    value: Any, protocol: Mapping[str, Any], label: str
) -> Mapping[str, Any]:
    """Bind the live JV execution interface to the H100-frozen authority."""

    contract = _exact_keys(
        value, JOINT_VELOCITY_CONTROLLER_CONTRACT_KEYS, label
    )
    authority = _exact_keys(
        protocol.get("joint_velocity_controller_authority"),
        (
            "authority_source",
            "independent_installed_source_rehash_required",
            "expected_restore_controller_contract",
        ),
        "protocol joint-velocity controller authority",
    )
    if (
        authority["authority_source"]
        != "VinUni_H100_safety_vla_robosuite_installation_and_frozen_Panda_XML"
        or authority[
            "independent_installed_source_rehash_required"
        ]
        is not True
    ):
        raise OneStepCounterfactualError(
            "protocol joint-velocity controller authority differs"
        )
    expected = _exact_keys(
        authority["expected_restore_controller_contract"],
        JOINT_VELOCITY_CONTROLLER_CONTRACT_KEYS,
        "protocol expected restore.controller",
    )
    _same(contract, expected, label + " and protocol authority")
    for field in (
        "controller_implementation_file_sha256",
        "controller_configuration_file_sha256",
        "panda_robot_xml_file_sha256",
    ):
        _sha256(contract[field], label + "." + field)
    if (
        contract["controller_class_module"] != "robosuite.controllers.joint_vel"
        or contract["controller_class_qualname"] != "JointVelocityController"
        or contract["controller_name"] != "JOINT_VELOCITY"
        or contract["environment_action_dim"] != 8
        or contract["arm_control_dim"] != 7
        or contract["control_frequency_hz"] != 100
        or contract["controller_policy_frequency_hz"] != 100
        or contract["control_timestep_s"] != 0.01
        or contract["physics_timestep_s"] != 0.002
        or contract["physics_substeps_per_control"] != 5
        or contract["normalized_input_lower"] != [-1.0] * 7
        or contract["normalized_input_upper"] != [1.0] * 7
        or contract["physical_output_lower_rad_s"] != [-0.5] * 7
        or contract["physical_output_upper_rad_s"] != [0.5] * 7
        or contract["interpolator"] is not None
    ):
        raise OneStepCounterfactualError(
            "%s timing/action normalization differs" % label
        )
    kp = _vector(contract["velocity_gain_kp"], 7, label + ".kp")
    ki = _vector(contract["velocity_gain_ki"], 7, label + ".ki")
    kd = _vector(contract["velocity_gain_kd"], 7, label + ".kd")
    lower = _vector(contract["arm_actuator_lower"], 7, label + ".actuator lower")
    upper = _vector(contract["arm_actuator_upper"], 7, label + ".actuator upper")
    for index in range(7):
        _close(kp[index], 3.0 * (upper[index] - lower[index]), label + ".kp scaling")
        _close(ki[index], kp[index] * 0.005, label + ".ki scaling")
        _close(kd[index], kp[index] * 0.001, label + ".kd scaling")
    if _matrix(
        contract["compiled_arm_actuator_ctrlrange"],
        7,
        2,
        label + ".compiled actuator ranges",
    ) != [[lower[index], upper[index]] for index in range(7)]:
        raise OneStepCounterfactualError(
            "%s compiled actuator ranges differ" % label
        )
    for field in (
        "arm_qpos_indexes",
        "arm_qvel_indexes",
        "arm_joint_ids",
        "arm_actuator_ids",
    ):
        if contract[field] != list(range(7)):
            raise OneStepCounterfactualError("%s %s differs" % (label, field))
    joint_index = _exact_keys(
        contract["controller_joint_index"],
        ("joints", "qpos", "qvel"),
        label + ".controller_joint_index",
    )
    if any(joint_index[field] != list(range(7)) for field in joint_index):
        raise OneStepCounterfactualError("%s joint index differs" % label)
    if (
        contract["arm_joint_names"]
        != ["robot0_joint%d" % index for index in range(1, 8)]
        or contract["arm_actuator_names"]
        != ["robot0_torq_j%d" % index for index in range(1, 8)]
    ):
        raise OneStepCounterfactualError("%s joint/actuator identity differs" % label)
    return contract


def contact_physical_boundary_index(observation_index: int, source_phase: str) -> int:
    """Map a recorded contact to its physical state boundary.

    A post-integration observation ``N`` is state boundary ``N + 1``.  A live
    pre-integration solver observation ``N`` is state boundary ``N``.
    """

    if (
        isinstance(observation_index, bool)
        or not isinstance(observation_index, int)
        or observation_index < 0
    ):
        raise OneStepCounterfactualError(
            "contact observation index must be a nonnegative integer"
        )
    if source_phase == "post_integration_recomputed":
        return int(observation_index) + 1
    if source_phase == "live_solver_phase_preintegration_geometry":
        return int(observation_index)
    raise OneStepCounterfactualError("contact source phase is invalid")


def first_filter_boundary_at_or_after_warning(
    warning_boundary: int, period_substeps: int = 5
) -> int:
    """Return the first scheduled filter boundary at or after ``W``.

    The Stage-13 intervention is authorized by the primary registered warning,
    not by looking backward from the eventual historical contact.  For a filter
    scheduled every ``period_substeps`` this is ``ceil(W / period) * period``.
    """

    if (
        isinstance(warning_boundary, bool)
        or not isinstance(warning_boundary, int)
        or warning_boundary < 0
        or isinstance(period_substeps, bool)
        or not isinstance(period_substeps, int)
        or period_substeps <= 0
    ):
        raise OneStepCounterfactualError("filter/warning boundary inputs are invalid")
    period = int(period_substeps)
    return ((int(warning_boundary) + period - 1) // period) * period


def _validate_protocol(
    protocol: Mapping[str, Any], protocol_raw_sha256: str
) -> Dict[str, Any]:
    if protocol.get("schema_version") != PROTOCOL_SCHEMA:
        raise OneStepCounterfactualError("Stage-13 protocol schema differs")
    if protocol.get("result_contract", {}).get("schema_version") != RESULT_SCHEMA:
        raise OneStepCounterfactualError("protocol result binding differs")
    expected_claim_scope = (
        "one_100hz_interval_local_geometry_field_jacobian_qp_and_tracking_"
        "feasibility_not_task_success_or_full_episode_safety"
    )
    if protocol.get("result_contract", {}).get("claim_scope") != expected_claim_scope:
        raise OneStepCounterfactualError("protocol claim scope differs")
    _sha256(protocol_raw_sha256, "protocol_raw_sha256")
    boundary = _mapping(protocol.get("counterfactual_boundary"), "protocol boundary")
    period = _integer(
        boundary.get("physics_substeps_per_filter_update"),
        "protocol filter period",
        1,
    )
    horizon = _integer(
        boundary.get("counterfactual_horizon_substeps"),
        "protocol horizon",
        1,
    )
    timestep = _number(boundary.get("physics_timestep_s"), "protocol timestep")
    horizon_s = _number(
        boundary.get("counterfactual_horizon_s"), "protocol horizon seconds"
    )
    if period != 5 or horizon != 5 or timestep != 0.002 or horizon_s != 0.01:
        raise OneStepCounterfactualError("Stage-13 cadence is not the frozen 5x2ms gate")
    _close(horizon_s, horizon * timestep, "protocol horizon arithmetic")
    acceptance = _mapping(protocol.get("acceptance"), "protocol acceptance")
    if acceptance.get("prediction_error_acceptance_threshold") is not None:
        raise OneStepCounterfactualError("post-hoc prediction tolerance is forbidden")
    if (
        acceptance.get("prediction_error_policy")
        != "diagnostic_only_no_post_hoc_tolerance"
    ):
        raise OneStepCounterfactualError("prediction error policy differs")
    expected_trend_quality = [
        "complete_exposure",
        "exact_paired_start",
        "qp_valid",
        "both_tracking_valid",
        "static_field_admissible",
    ]
    if acceptance.get("trend_only_quality_prerequisites") != expected_trend_quality:
        raise OneStepCounterfactualError("trend-only quality prerequisites differ")
    expected_trend_evidence = [
        "first_order_predicted_h_after_horizon_m2",
        "actual_h_after_each_physics_substep_m2",
        "nominal_D_sim_at_B_m",
        "nominal_minimum_D_sim_over_horizon_m",
        "psf_h_after_horizon_m2",
        "psf_minus_nominal_h_after_horizon_m2",
        "psf_warning_sample_h_strictly_greater_than_nominal_at_horizon",
    ]
    expected_diagnostic_only = [
        "prediction_error_m2_without_acceptance_threshold"
    ]
    required_evidence = _mapping(
        protocol.get("required_evidence"), "protocol required evidence"
    )
    observed_trend_evidence = required_evidence.get(
        "trend_reference_acceptance_evidence"
    )
    observed_diagnostic_only = required_evidence.get("diagnostic_only")
    if (
        observed_trend_evidence != expected_trend_evidence
        or observed_diagnostic_only != expected_diagnostic_only
        or set(observed_trend_evidence).intersection(observed_diagnostic_only)
    ):
        raise OneStepCounterfactualError(
            "trend acceptance and diagnostic-only evidence scopes differ"
        )
    boundary_preflight = _mapping(
        protocol.get("boundary_B_preflight"), "protocol boundary-B preflight"
    )
    if (
        boundary_preflight.get("failure_policy")
        != "apparatus_failure_no_QP_no_arm_physics_no_scientific_interpretation"
    ):
        raise OneStepCounterfactualError(
            "boundary-B preflight apparatus-failure policy differs"
        )
    gripper_policy = _mapping(
        _mapping(protocol.get("arms"), "protocol arms").get("gripper_policy"),
        "protocol gripper policy",
    )
    if gripper_policy.get("required_evidence") != [
        "source_action_index",
        "source_action_7d",
        "source_action_sha256",
        "exact_gripper_value",
        "exact_gripper_value_sha256",
    ]:
        raise OneStepCounterfactualError("gripper evidence contract differs")
    reference = _exact_keys(
        _mapping(protocol.get("qp_execution"), "protocol QP execution").get(
            "independent_postproducer_reference_check"
        ),
        (
            "execution",
            "solver",
            "eps_abs",
            "eps_rel",
            "max_iterations",
            "qdot_linf_tolerance_rad_s",
            "required_before_scientific_interpretation",
        ),
        "protocol independent QP reference check",
    )
    expected_reference = {
        "execution": "separate_artifact_consumer_after_producer_exit",
        "solver": "cvxpy_osqp",
        "eps_abs": 1e-9,
        "eps_rel": 1e-9,
        "max_iterations": 20000,
        "qdot_linf_tolerance_rad_s": 2e-5,
        "required_before_scientific_interpretation": True,
    }
    _same(reference, expected_reference, "protocol independent QP reference check")
    expected_motion_acceptance = {
        "minimum_filter_correction_norm_rad_s": 1e-4,
        "minimum_safe_command_norm_rad_s": 0.05,
        "minimum_safe_to_nominal_command_norm_ratio": 0.25,
        "minimum_safe_measured_joint_motion_rad": 1e-4,
        "minimum_safe_measured_motion_to_command_integral_ratio": 0.25,
        "local_motion_interpretation": (
            "nonzero_local_joint_motion_not_nominal_direction_progress_or_task_success"
        ),
    }
    for field, expected in expected_motion_acceptance.items():
        if acceptance.get(field) != expected:
            raise OneStepCounterfactualError(
                "protocol local-motion acceptance differs at %s" % field
            )
    estimator = _exact_keys(
        protocol.get("nominal_velocity_estimator"),
        (
            "method",
            "qpos_endpoints",
            "interval_s",
            "full_nv_producer_diagnostic_retained",
            "arm_velocity_dof_slice",
            "expected_arm_qpos_indices",
            "independent_claim_bearing_reconstruction",
            "nonarm_values_validation_scope",
            "finite_values_required",
            "arm_velocity_lower_rad_s",
            "arm_velocity_upper_rad_s",
            "out_of_bounds_policy",
            "direct_qpos_subtraction_prohibited",
        ),
        "protocol nominal velocity estimator",
    )
    if (
        estimator["method"] != "mujoco_mj_differentiatePos_full_nv"
        or estimator["full_nv_producer_diagnostic_retained"] is not True
        or estimator["independent_claim_bearing_reconstruction"]
        != "seven_scalar_hinge_arm_slice_only"
        or estimator["nonarm_values_validation_scope"]
        != "producer_diagnostic_not_independently_reconstructed_no_claim"
        or estimator["expected_arm_qpos_indices"] != list(range(7))
    ):
        raise OneStepCounterfactualError(
            "protocol nominal velocity validation scope overclaims"
        )
    controller_authority = _mapping(
        protocol.get("joint_velocity_controller_authority"),
        "protocol joint-velocity controller authority",
    )
    _validate_joint_velocity_controller_contract(
        controller_authority.get("expected_restore_controller_contract"),
        protocol,
        "protocol expected restore.controller",
    )
    return {
        "protocol": protocol,
        "period": period,
        "horizon": horizon,
        "timestep": timestep,
        "horizon_s": horizon_s,
        "acceptance": acceptance,
        "reference_check": reference,
        "semantic_sha256": _sha(protocol),
    }


def _validate_execution(value: Any) -> Mapping[str, Any]:
    execution = _exact_keys(value, EXECUTION_KEYS, "execution")
    kind = execution["outcome_kind"]
    if kind not in (
        "paired_counterfactual_complete",
        "preflight_inadmissible_nominal_velocity",
    ):
        raise OneStepCounterfactualError("execution outcome kind is invalid")
    expected = {
        "source_prefix_complete": True,
        "qp_executed": kind == "paired_counterfactual_complete",
        "paired_joint_velocity_physics_executed": (
            kind == "paired_counterfactual_complete"
        ),
        "no_hidden_clipping": True,
    }
    for field, expected_value in expected.items():
        if execution[field] is not expected_value:
            raise OneStepCounterfactualError(
                "execution.%s differs for %s" % (field, kind)
            )
    return execution


def _validate_authority(
    result: Mapping[str, Any], expected_authority: Mapping[str, Any]
) -> Mapping[str, Any]:
    authority_keys = (
        "expected_code_commit",
        "expected_run_id",
        "expected_slurm_job_id",
        "expected_host",
        "expected_device",
        "protocol",
        "numeric_prerequisite",
        "parity_prerequisite",
        "identification_prerequisite",
        "dynamic_authority",
    )
    expected = _exact_keys(expected_authority, authority_keys, "expected_authority")
    observed = _exact_keys(result.get("authority"), authority_keys, "result.authority")
    _same(observed, expected, "result authority and external authority")
    commit = observed["expected_code_commit"]
    if not isinstance(commit, str) or COMMIT_PATTERN.fullmatch(commit) is None:
        raise OneStepCounterfactualError("expected source commit is invalid")
    for field in (
        "expected_run_id",
        "expected_slurm_job_id",
        "expected_host",
        "expected_device",
    ):
        if not isinstance(observed[field], str) or not observed[field]:
            raise OneStepCounterfactualError("authority.%s is empty" % field)
    protocol_identity = _exact_keys(
        observed["protocol"],
        ("path", "file_sha256", "semantic_sha256", "schema_version", "protocol_id"),
        "authority.protocol",
    )
    _sha256(protocol_identity["file_sha256"], "authority protocol file hash")
    _sha256(protocol_identity["semantic_sha256"], "authority protocol semantic hash")
    for kind, schema in (
        ("numeric_prerequisite", "vlsa_poisson_numeric_validation.v1"),
        ("parity_prerequisite", "vlsa_poisson_shadow_parity.v2"),
        ("identification_prerequisite", "vlsa_poisson_shadow_identification.v3"),
    ):
        identity = _exact_keys(
            observed[kind],
            ("path", "file_sha256", "payload_sha256", "schema_version", "status"),
            "authority.%s" % kind,
        )
        _sha256(identity["file_sha256"], kind + " file hash")
        _sha256(identity["payload_sha256"], kind + " payload hash")
        if identity["schema_version"] != schema or identity["status"] != "passed":
            raise OneStepCounterfactualError("%s identity is not passing" % kind)
    dynamic = _exact_keys(
        observed["dynamic_authority"],
        ("case", "selection", "historical", "runtime", "parity", "identification"),
        "authority.dynamic_authority",
    )
    case = _exact_keys(
        dynamic["case"],
        ("case_id", "manifest_file_sha256", "manifest_row_sha256"),
        "dynamic_authority.case",
    )
    for field in ("manifest_file_sha256", "manifest_row_sha256"):
        _sha256(case[field], "dynamic case." + field)
    selection = _exact_keys(
        dynamic["selection"],
        ("relative_path", "schema_version", "file_sha256", "protocol_id"),
        "dynamic_authority.selection",
    )
    _sha256(selection["file_sha256"], "dynamic selection file hash")
    if any(
        not isinstance(selection[field], str) or not selection[field]
        for field in ("relative_path", "schema_version", "protocol_id")
    ):
        raise OneStepCounterfactualError("dynamic selection protocol ID is empty")
    historical = _exact_keys(
        dynamic["historical"],
        (
            "result_file_sha256",
            "result_payload_sha256",
            "action_count",
            "executed_action_sequence_sha256",
            "policy_noise_schedule_sha256",
        ),
        "dynamic_authority.historical",
    )
    for field in (
        "result_file_sha256",
        "result_payload_sha256",
        "executed_action_sequence_sha256",
        "policy_noise_schedule_sha256",
    ):
        _sha256(historical[field], "dynamic historical." + field)
    _integer(historical["action_count"], "dynamic historical action count", 1)
    runtime = _exact_keys(
        dynamic["runtime"],
        (
            "raw_file_sha256",
            "semantic_sha256",
            "parameter_block_sha256",
            "schema_version",
            "protocol_id",
            "registered_parameters",
        ),
        "dynamic_authority.runtime",
    )
    for field in ("raw_file_sha256", "semantic_sha256", "parameter_block_sha256"):
        _sha256(runtime[field], "dynamic runtime." + field)
    registered = _exact_keys(
        runtime["registered_parameters"],
        ("admissibility", "coverage", "cbf", "qp", "cadence"),
        "dynamic runtime registered parameters",
    )
    _exact_keys(
        registered["admissibility"],
        (
            "max_state_restore_qpos_error_rad",
            "max_state_restore_qvel_error_rad_s",
            "joint_position_margin_rad",
            "max_joint_velocity_tracking_linf_rad_s",
            "max_joint_velocity_tracking_rmse_rad_s",
            "max_selected_body_linear_speed_m_s",
            "max_selected_body_angular_speed_rad_s",
            "max_selected_geom_translation_drift_m",
            "max_selected_geom_rotation_drift_rad",
            "max_selected_geom_surface_drift_m",
            "maximum_invalid_field_queries",
            "require_safe_initial_samples",
            "static_field_refresh_policy",
            "violation_policy",
        ),
        "registered admissibility",
    )
    _exact_keys(
        registered["coverage"],
        (
            "epsilon_m",
            "ball_semantics",
            "certificate_method",
            "certificate_relation",
            "require_every_collision_surface_component",
        ),
        "registered coverage",
    )
    _exact_keys(
        registered["cbf"],
        (
            "alpha_gain_per_s",
            "constraint_form",
            "time_derivative_policy",
            "issf_mode",
            "issf_epsilon0",
            "unsafe_initial_policy",
        ),
        "registered CBF",
    )
    _exact_keys(
        registered["qp"],
        (
            "solver",
            "objective",
            "weight_diagonal",
            "hard_cbf_constraints",
            "slack_enabled",
            "velocity_lower_rad_s",
            "velocity_upper_rad_s",
            "joint_limit_alpha_per_s",
            "joint_position_margin_rad",
            "eps_abs",
            "eps_rel",
            "max_iterations",
            "postcheck_cbf_tolerance",
            "postcheck_bound_tolerance_rad_s",
            "row_scaling",
            "failure_policy",
        ),
        "registered QP",
    )
    cadence = _exact_keys(
        registered["cadence"],
        ("physics_timestep_s", "high_level_frequency_hz", "shadow", "active"),
        "registered cadence",
    )
    _exact_keys(
        cadence["shadow"],
        (
            "mode",
            "filter_updates_per_high_level_action",
            "physics_substeps_per_filter_update",
        ),
        "registered shadow cadence",
    )
    _exact_keys(
        cadence["active"],
        (
            "mode",
            "filter_frequency_hz",
            "filter_updates_per_high_level_action",
            "physics_substeps_per_filter_update",
        ),
        "registered active cadence",
    )
    parity = _exact_keys(
        dynamic["parity"],
        (
            "executed_action_count",
            "action_boundary_state_sha256_ledger",
            "state_sequence_sha256",
            "official_integration_state_count",
            "official_integration_state_sha256_ledger",
            "official_integration_state_sequence_sha256",
        ),
        "dynamic_authority.parity",
    )
    action_states = _sequence(
        parity["action_boundary_state_sha256_ledger"], "parity action state ledger"
    )
    callback_states = _sequence(
        parity["official_integration_state_sha256_ledger"], "parity callback state ledger"
    )
    if (
        _integer(parity["executed_action_count"], "parity action count", 1)
        != len(action_states)
        or _integer(parity["official_integration_state_count"], "parity callback count", 1)
        != len(callback_states)
    ):
        raise OneStepCounterfactualError("dynamic parity population count differs")
    for label, ledger, hash_field in (
        ("action", action_states, "state_sequence_sha256"),
        ("callback", callback_states, "official_integration_state_sequence_sha256"),
    ):
        for index, digest in enumerate(ledger):
            _sha256(digest, "dynamic parity %s state %d" % (label, index))
        if _sha(list(ledger)) != parity[hash_field]:
            raise OneStepCounterfactualError("dynamic parity %s ledger hash differs" % label)
    identification = _exact_keys(
        dynamic["identification"],
        (
            "first_link56_contact",
            "primary_registered_warning",
            "contact_model_authority_sha256",
            "physical_model",
            "resolved_geometry",
            "field_bundle_hashes",
            "ordered_protected_sample_ledger_sha256",
            "protected_sample_count",
            "arm_dof_indices",
            "settled_mjstate_integration_sha256",
            "differential_binding_sha256",
            "differential_classification_ledger_sha256",
            "differential_validation",
            "full_robot_measurement_sampling",
            "callback_state_read_only_count",
            "callback_state_read_only_after_sha256_ledger",
            "callback_state_sequence_sha256",
        ),
        "dynamic_authority.identification",
    )
    for field in (
        "contact_model_authority_sha256",
        "ordered_protected_sample_ledger_sha256",
        "settled_mjstate_integration_sha256",
        "differential_binding_sha256",
        "differential_classification_ledger_sha256",
    ):
        _sha256(identification[field], "dynamic identification." + field)
    _validate_physical_model_contract(
        identification["physical_model"], "dynamic identification physical model"
    )
    _integer(identification["protected_sample_count"], "dynamic protected sample count", 1)
    if len(_sequence(identification["arm_dof_indices"], "dynamic arm DOFs")) != 7:
        raise OneStepCounterfactualError("dynamic identification arm DOF count differs")
    identification_states = _sequence(
        identification["callback_state_read_only_after_sha256_ledger"],
        "identification callback state ledger",
    )
    if _integer(
        identification["callback_state_read_only_count"],
        "identification callback state count",
        1,
    ) != len(identification_states):
        raise OneStepCounterfactualError("identification callback count differs")
    for index, digest in enumerate(identification_states):
        _sha256(digest, "identification callback state %d" % index)
    if _sha(list(identification_states)) != identification[
        "callback_state_sequence_sha256"
    ]:
        raise OneStepCounterfactualError("identification callback ledger hash differs")
    if (
        list(identification_states) != list(callback_states)
        or identification["callback_state_sequence_sha256"]
        != parity["official_integration_state_sequence_sha256"]
    ):
        raise OneStepCounterfactualError("identification and parity callback ledgers differ")
    return observed


def _validate_protocol_binding(
    result: Mapping[str, Any],
    protocol: Mapping[str, Any],
    protocol_raw_sha256: str,
    protocol_values: Mapping[str, Any],
    authority: Mapping[str, Any],
) -> None:
    binding = _exact_keys(
        result.get("protocol_binding"),
        (
            "raw_file_sha256",
            "semantic_sha256",
            "schema_version",
            "protocol_id",
            "result_schema_version",
        ),
        "protocol_binding",
    )
    expected = {
        "raw_file_sha256": protocol_raw_sha256,
        "semantic_sha256": protocol_values["semantic_sha256"],
        "schema_version": PROTOCOL_SCHEMA,
        "protocol_id": protocol.get("protocol_id"),
        "result_schema_version": RESULT_SCHEMA,
    }
    _same(binding, expected, "protocol binding")
    protocol_identity = authority["protocol"]
    if (
        protocol_identity["file_sha256"] != protocol_raw_sha256
        or protocol_identity["semantic_sha256"] != protocol_values["semantic_sha256"]
        or protocol_identity["schema_version"] != PROTOCOL_SCHEMA
        or protocol_identity["protocol_id"] != protocol.get("protocol_id")
    ):
        raise OneStepCounterfactualError("external protocol authority differs")


def _validate_provenance(
    value: Any,
    *,
    authority: Mapping[str, Any],
    protocol: Mapping[str, Any],
    case_id: str,
) -> Mapping[str, Any]:
    provenance = _exact_keys(
        value,
        (
            "source",
            "allocation",
            "manifest",
            "selection",
            "runtime_protocol",
            "historical_result",
            "physical_model",
            "joint_velocity_controller",
            "contact_model",
            "field_bundle",
            "execution",
            "python",
        ),
        "provenance",
    )
    source = _exact_keys(
        provenance["source"], ("commit", "branch", "status_short"), "provenance.source"
    )
    if (
        source["commit"] != authority["expected_code_commit"]
        or not isinstance(source["branch"], str)
        or not source["branch"]
        or not isinstance(source["status_short"], list)
        or source["status_short"]
    ):
        raise OneStepCounterfactualError("provenance source is not the expected clean commit")
    allocation = _exact_keys(
        provenance["allocation"],
        ("execution_environment", "slurm_job_id", "host_name", "device"),
        "provenance.allocation",
    )
    if (
        allocation["execution_environment"] != "slurm_allocation"
        or str(allocation["slurm_job_id"]) != authority["expected_slurm_job_id"]
        or allocation["host_name"] != authority["expected_host"]
    ):
        raise OneStepCounterfactualError("allocation provenance differs from authority")
    device = _exact_keys(
        allocation["device"],
        (
            "device_type",
            "visible_device_ids",
            "model",
            "uuid",
            "driver_version",
            "cuda_runtime_version",
        ),
        "provenance allocation device",
    )
    visible_ids = _sequence(device["visible_device_ids"], "visible device IDs")
    if (
        device.get("device_type") != "cuda"
        or "H100" not in str(device.get("model", ""))
        or len(visible_ids) != 1
        or not isinstance(visible_ids[0], str)
        or not visible_ids[0]
        or any(
            not isinstance(device[field], str) or not device[field]
            for field in ("uuid", "driver_version", "cuda_runtime_version")
        )
        or authority["expected_device"] not in _canonical(device).decode("utf-8")
    ):
        raise OneStepCounterfactualError("allocation is not the expected single H100")
    dynamic = authority["dynamic_authority"]
    case_authority = dynamic["case"]
    manifest = _exact_keys(
        provenance["manifest"],
        ("path", "file_sha256", "row_sha256", "case_id"),
        "provenance.manifest",
    )
    if (
        manifest["case_id"] != case_id
        or manifest["file_sha256"] != case_authority["manifest_file_sha256"]
        or manifest["row_sha256"] != case_authority["manifest_row_sha256"]
    ):
        raise OneStepCounterfactualError("manifest provenance differs")
    selection = _exact_keys(
        provenance["selection"],
        ("path", "schema_version", "file_sha256", "protocol_id"),
        "provenance.selection",
    )
    _sha256(selection["file_sha256"], "selection file hash")
    if not isinstance(selection["path"], str) or not selection["path"]:
        raise OneStepCounterfactualError("selection path is empty")
    if (
        selection["file_sha256"] != dynamic["selection"]["file_sha256"]
        or selection["schema_version"] != dynamic["selection"]["schema_version"]
        or selection["protocol_id"] != dynamic["selection"]["protocol_id"]
        or not selection["path"].endswith(dynamic["selection"]["relative_path"])
    ):
        raise OneStepCounterfactualError("selection provenance differs")
    frozen_selection = protocol["prerequisites"]
    if (
        dynamic["selection"]["relative_path"]
        != frozen_selection["selection_protocol_relative_path"]
        or dynamic["selection"]["schema_version"]
        != frozen_selection["selection_schema_version"]
        or dynamic["selection"]["file_sha256"]
        != frozen_selection["selection_raw_file_sha256"]
        or dynamic["selection"]["protocol_id"]
        != frozen_selection["selection_protocol_id"]
    ):
        raise OneStepCounterfactualError("selection authority differs from protocol")
    runtime = _exact_keys(
        provenance["runtime_protocol"],
        (
            "path",
            "raw_file_sha256",
            "semantic_sha256",
            "parameter_block_sha256",
            "schema_version",
            "protocol_id",
        ),
        "provenance.runtime_protocol",
    )
    expected_runtime = dynamic["runtime"]
    for field in (
        "raw_file_sha256",
        "semantic_sha256",
        "parameter_block_sha256",
        "schema_version",
        "protocol_id",
    ):
        if runtime[field] != expected_runtime[field]:
            raise OneStepCounterfactualError("runtime provenance differs at %s" % field)
    prerequisites = protocol["prerequisites"]
    if (
        runtime["raw_file_sha256"] != prerequisites["runtime_raw_file_sha256"]
        or runtime["semantic_sha256"]
        != prerequisites["runtime_semantic_protocol_sha256"]
        or runtime["parameter_block_sha256"]
        != prerequisites["runtime_parameter_block_sha256"]
        or runtime["schema_version"] != prerequisites["runtime_schema_version"]
        or runtime["protocol_id"] != prerequisites["runtime_protocol_id"]
    ):
        raise OneStepCounterfactualError("runtime provenance differs from protocol")
    historical = _exact_keys(
        provenance["historical_result"],
        (
            "path",
            "file_sha256",
            "payload_sha256",
            "action_count",
            "executed_action_sequence_sha256",
            "policy_noise_schedule_sha256",
        ),
        "provenance.historical_result",
    )
    expected_historical = dynamic["historical"]
    historical_projection = {
        "result_file_sha256": historical["file_sha256"],
        "result_payload_sha256": historical["payload_sha256"],
        "action_count": historical["action_count"],
        "executed_action_sequence_sha256": historical[
            "executed_action_sequence_sha256"
        ],
        "policy_noise_schedule_sha256": historical["policy_noise_schedule_sha256"],
    }
    _same(historical_projection, expected_historical, "historical provenance")
    physical = _validate_physical_model_contract(
        provenance["physical_model"], "provenance.physical_model"
    )
    _same(
        physical,
        dynamic["identification"]["physical_model"],
        "provenance and identification physical model",
    )
    _validate_joint_velocity_controller_contract(
        provenance["joint_velocity_controller"],
        protocol,
        "provenance.joint_velocity_controller",
    )
    contact = _exact_keys(
        provenance["contact_model"],
        ("authority_sha256", "active_obstacle_name", "resolved_geometry"),
        "provenance.contact_model",
    )
    identification = dynamic["identification"]
    if (
        contact["authority_sha256"] != identification["contact_model_authority_sha256"]
        or _canonical(contact["resolved_geometry"])
        != _canonical(identification["resolved_geometry"])
        or not isinstance(contact["active_obstacle_name"], str)
        or not contact["active_obstacle_name"]
        or contact["active_obstacle_name"]
        != protocol["case"]["selected_obstacle_name"]
    ):
        raise OneStepCounterfactualError("contact-model provenance differs")
    field = _exact_keys(
        provenance["field_bundle"],
        (
            "bundle_sha256",
            "protected_samples_sha256",
            "protocol_sha256",
            "parameter_block_sha256",
        ),
        "provenance.field_bundle",
    )
    for key, digest in field.items():
        _sha256(digest, "field provenance." + key)
    expected_field = identification["field_bundle_hashes"]
    if (
        field["bundle_sha256"] != expected_field.get("bundle_sha256")
        or field["protected_samples_sha256"]
        != identification["ordered_protected_sample_ledger_sha256"]
        or field["protocol_sha256"] != expected_runtime["semantic_sha256"]
        or field["parameter_block_sha256"]
        != expected_runtime["parameter_block_sha256"]
    ):
        raise OneStepCounterfactualError("field-bundle provenance differs")
    execution = _exact_keys(
        provenance["execution"],
        ("active_policy_query_count", "policy_server_started", "source_prefix_replayed_once"),
        "provenance.execution",
    )
    if (
        _integer(execution["active_policy_query_count"], "active policy query count", 0)
        != 0
        or execution["policy_server_started"] is not False
        or execution["source_prefix_replayed_once"] is not True
    ):
        raise OneStepCounterfactualError("Stage-13 execution queried a policy process")
    python = _exact_keys(provenance["python"], ("executable", "version"), "provenance.python")
    if any(not isinstance(python[field], str) or not python[field] for field in python):
        raise OneStepCounterfactualError("Python provenance is incomplete")
    return provenance


def _validate_captured_state(
    value: Any,
    *,
    expected_boundary: int,
    physical_model: Mapping[str, Any],
    label: str,
) -> Mapping[str, Any]:
    record = _exact_keys(
        value,
        (
            "physical_boundary",
            "mujoco_state_specification",
            "integration_state",
            "integration_state_sha256",
            "flattened_simulator_state",
            "flattened_simulator_state_sha256",
            "qpos",
            "qpos_sha256",
            "qvel",
            "qvel_sha256",
            "wrapper_bookkeeping",
        ),
        label,
    )
    if _integer(record["physical_boundary"], label + ".physical_boundary", 0) != expected_boundary:
        raise OneStepCounterfactualError("%s boundary differs" % label)
    if record["mujoco_state_specification"] != "mjSTATE_INTEGRATION":
        raise OneStepCounterfactualError("%s state specification differs" % label)
    for value_field, hash_field in (
        ("integration_state", "integration_state_sha256"),
        ("flattened_simulator_state", "flattened_simulator_state_sha256"),
        ("qpos", "qpos_sha256"),
        ("qvel", "qvel_sha256"),
    ):
        _validate_float_array_record(record, value_field, hash_field, label)
        if not _sequence(record[value_field], label + "." + value_field):
            raise OneStepCounterfactualError("%s.%s is empty" % (label, value_field))
    integration = [
        _number(item, label + ".integration_state")
        for item in record["integration_state"]
    ]
    qpos = [_number(item, label + ".qpos") for item in record["qpos"]]
    qvel = [_number(item, label + ".qvel") for item in record["qvel"]]
    flattened = [
        _number(item, label + ".flattened_simulator_state")
        for item in record["flattened_simulator_state"]
    ]
    nq = _integer(physical_model["nq"], label + ".compiled nq", 1)
    nv = _integer(physical_model["nv"], label + ".compiled nv", 1)
    na = _integer(physical_model["na"], label + ".compiled na", 0)
    common_size = 1 + nq + nv + na
    if (
        len(qpos) != nq
        or len(qvel) != nv
        or len(integration) != physical_model["mjstate_integration_size"]
        or len(flattened) != physical_model["robosuite_flattened_state_size"]
        or common_size != physical_model["robosuite_flattened_state_size"]
        or flattened != integration[:common_size]
        or integration[1 : 1 + nq] != qpos
        or integration[1 + nq : 1 + nq + nv] != qvel
    ):
        raise OneStepCounterfactualError(
            "%s compiled MuJoCo/Robosuite state layout differs" % label
        )
    bookkeeping = _exact_keys(
        record["wrapper_bookkeeping"], ("timestep", "cur_time_s", "done"), label + ".wrapper"
    )
    _integer(bookkeeping["timestep"], label + ".wrapper.timestep", 0)
    _number(bookkeeping["cur_time_s"], label + ".wrapper.cur_time_s")
    _boolean(bookkeeping["done"], label + ".wrapper.done")
    _close(integration[0], bookkeeping["cur_time_s"], label + ".state time")
    return record


def _validate_exact_prefix(
    value: Any,
    *,
    boundary: int,
    horizon: int,
    physical_model: Mapping[str, Any],
) -> Mapping[str, Any]:
    prefix = _exact_keys(
        value,
        (
            "execution",
            "observed_callback_count",
            "expected_callback_count",
            "observed_callback_sha256_ledger",
            "observed_callback_prefix_sha256",
            "parity_callback_prefix_sha256",
            "identification_callback_prefix_sha256",
            "historical_executed_action_sequence_sha256",
            "state_at_B",
            "state_at_B_plus_5",
        ),
        "source_boundary.exact_prefix",
    )
    if prefix["execution"] != "one_exact_historical_OSC_prefix_no_policy_query":
        raise OneStepCounterfactualError("source prefix execution mode differs")
    count = boundary + horizon
    if (
        _integer(prefix["observed_callback_count"], "prefix observed count", 0) != count
        or _integer(prefix["expected_callback_count"], "prefix expected count", 0) != count
    ):
        raise OneStepCounterfactualError("source prefix callback count differs")
    ledger = _sequence(prefix["observed_callback_sha256_ledger"], "prefix state ledger")
    if len(ledger) != count:
        raise OneStepCounterfactualError("source prefix state ledger is incomplete")
    for index, digest in enumerate(ledger):
        _sha256(digest, "prefix state hash %d" % index)
    ledger_hash = _sha(list(ledger))
    for field in (
        "observed_callback_prefix_sha256",
        "parity_callback_prefix_sha256",
        "identification_callback_prefix_sha256",
    ):
        if _sha256(prefix[field], "prefix." + field) != ledger_hash:
            raise OneStepCounterfactualError("%s differs from the exact prefix" % field)
    _sha256(
        prefix["historical_executed_action_sequence_sha256"],
        "historical action sequence hash",
    )
    state_B = _validate_captured_state(
        prefix["state_at_B"],
        expected_boundary=boundary,
        physical_model=physical_model,
        label="prefix.state_at_B",
    )
    state_B5 = _validate_captured_state(
        prefix["state_at_B_plus_5"],
        expected_boundary=boundary + horizon,
        physical_model=physical_model,
        label="prefix.state_at_B_plus_5",
    )
    if boundary > 0 and ledger[boundary - 1] != state_B["integration_state_sha256"]:
        raise OneStepCounterfactualError("prefix boundary-B state hash differs")
    if ledger[boundary + horizon - 1] != state_B5["integration_state_sha256"]:
        raise OneStepCounterfactualError("prefix endpoint state hash differs")
    return prefix


def _validate_samples(value: Any) -> Dict[str, Any]:
    binding = _exact_keys(
        value,
        (
            "contact_model_authority_sha256",
            "physical_model",
            "resolved_geometry",
            "field_bundle_hashes",
            "field_bundle_sha256",
            "ordered_protected_sample_ledger",
            "ordered_protected_sample_ledger_sha256",
            "protected_sample_count",
            "protected_surface_components",
            "protected_sampling_epsilon_m",
            "protected_sampling_maximum_surface_cover_radius_m",
            "protected_sampling_coverage_semantics",
            "full_robot_measurement_sampling",
            "arm_dof_indices",
            "settled_mjstate_integration_sha256",
            "settled_link56_differential_audit_binding_sha256",
            "settled_link56_differential_audit_classification_ledger_sha256",
            "settled_link56_differential_audit_validation",
            "fresh_stable_construction_binding_equals_identification",
        ),
        "shadow_construction_binding",
    )
    for field in (
        "contact_model_authority_sha256",
        "field_bundle_sha256",
        "ordered_protected_sample_ledger_sha256",
        "settled_mjstate_integration_sha256",
        "settled_link56_differential_audit_binding_sha256",
        "settled_link56_differential_audit_classification_ledger_sha256",
    ):
        _sha256(binding[field], "shadow." + field)
    _validate_physical_model_contract(
        binding["physical_model"], "shadow physical model"
    )
    if binding["fresh_stable_construction_binding_equals_identification"] is not True:
        raise OneStepCounterfactualError("fresh construction is not shadow-identical")
    samples = _sequence(
        binding["ordered_protected_sample_ledger"], "ordered protected samples"
    )
    sample_count = _integer(binding["protected_sample_count"], "protected sample count", 1)
    if len(samples) != sample_count:
        raise OneStepCounterfactualError("protected sample count differs")
    normalized_samples = []
    for index, raw in enumerate(samples):
        sample = _exact_keys(
            raw,
            (
                "sample_id",
                "body_id",
                "body_name",
                "geom_id",
                "geom_name",
                "point_body_local_m",
                "source",
            ),
            "protected sample %d" % index,
        )
        if _integer(sample["sample_id"], "sample id", 0) != index:
            raise OneStepCounterfactualError("protected sample IDs are not contiguous")
        for field in ("body_id", "geom_id"):
            _integer(sample[field], "sample." + field, 0)
        for field in ("body_name", "geom_name", "source"):
            if not isinstance(sample[field], str) or not sample[field]:
                raise OneStepCounterfactualError("sample.%s is empty" % field)
        _vector(sample["point_body_local_m"], 3, "sample local point")
        normalized_samples.append(dict(sample))
    components = _sequence(
        binding["protected_surface_components"], "protected surface components"
    )
    if not components:
        raise OneStepCounterfactualError("protected component population is empty")
    component_by_geom: Dict[int, Mapping[str, Any]] = {}
    radii = []
    for index, raw in enumerate(components):
        component = _exact_keys(
            raw,
            (
                "geom_id",
                "geom_name",
                "body_id",
                "body_name",
                "geometry_kind",
                "certificate_kind",
                "surface_element_count",
                "sample_count",
                "certified_surface_cover_radius_m",
            ),
            "protected component %d" % index,
        )
        geom = _integer(component["geom_id"], "component geom id", 0)
        if geom in component_by_geom:
            raise OneStepCounterfactualError("protected component geom is duplicated")
        component_by_geom[geom] = component
        for field in ("geom_name", "body_name", "geometry_kind", "certificate_kind"):
            if not isinstance(component[field], str) or not component[field]:
                raise OneStepCounterfactualError("component.%s is empty" % field)
        _integer(component["body_id"], "component body id", 0)
        _integer(component["surface_element_count"], "surface element count", 1)
        declared_count = _integer(component["sample_count"], "component sample count", 1)
        observed_count = sum(int(sample["geom_id"]) == geom for sample in samples)
        if declared_count != observed_count:
            raise OneStepCounterfactualError("component sample count differs")
        radii.append(_number(component["certified_surface_cover_radius_m"], "coverage radius"))
    epsilon = _number(binding["protected_sampling_epsilon_m"], "sampling epsilon")
    maximum = _number(
        binding["protected_sampling_maximum_surface_cover_radius_m"],
        "maximum sampling radius",
    )
    if epsilon <= 0.0 or maximum < 0.0 or maximum >= epsilon:
        raise OneStepCounterfactualError("protected sampling radius is inadmissible")
    if any(radius < 0.0 or radius >= epsilon for radius in radii):
        raise OneStepCounterfactualError("component radius is not strictly inside epsilon")
    _close(maximum, max(radii), "global protected coverage radius", 0.0)
    for sample in samples:
        component = component_by_geom.get(int(sample["geom_id"]))
        if component is None:
            raise OneStepCounterfactualError("sample has no component certificate")
        for field in ("geom_name", "body_id", "body_name"):
            if sample[field] != component[field]:
                raise OneStepCounterfactualError("sample/component identity differs")
    coverage = binding["protected_sampling_coverage_semantics"]
    if not isinstance(coverage, str) or not coverage:
        raise OneStepCounterfactualError("sampling coverage semantics are empty")
    sample_payload = {
        "epsilon_m": epsilon,
        "maximum_surface_cover_radius_m": maximum,
        "coverage_semantics": coverage,
        "components": list(components),
        "samples": list(samples),
    }
    reconstructed_sample_hash = _sha(sample_payload)
    if reconstructed_sample_hash != binding["ordered_protected_sample_ledger_sha256"]:
        raise OneStepCounterfactualError("protected sample ledger hash does not reconstruct")
    field_hashes = _mapping(binding["field_bundle_hashes"], "field bundle hashes")
    for key, digest in field_hashes.items():
        if not isinstance(key, str):
            raise OneStepCounterfactualError("field bundle hash key is invalid")
        if key != "diagnostics":
            _sha256(digest, "field_bundle_hashes.%s" % key)
    if field_hashes.get("protected_samples_sha256") != reconstructed_sample_hash:
        raise OneStepCounterfactualError("field bundle sample hash differs")
    if field_hashes.get("bundle_sha256") != binding["field_bundle_sha256"]:
        raise OneStepCounterfactualError("field bundle identity differs")
    dofs = _sequence(binding["arm_dof_indices"], "arm DOF indices")
    if len(dofs) != 7:
        raise OneStepCounterfactualError("arm DOF index count differs")
    parsed_dofs = [_integer(value, "arm DOF index", 0) for value in dofs]
    if len(set(parsed_dofs)) != 7:
        raise OneStepCounterfactualError("arm DOF indices are duplicated")
    validation = _exact_keys(
        binding["settled_link56_differential_audit_validation"],
        (
            "schema_version",
            "audit_payload_sha256",
            "ordered_sample_identity_sha256",
            "integration_state_sha256",
            "specification_sha256",
            "binding_sha256",
            "classification_ledger_sha256",
            "counts",
            "passed",
        ),
        "settled differential audit validation",
    )
    if validation["schema_version"] != DIFFERENTIAL_AUDIT_SCHEMA or validation["passed"] is not True:
        raise OneStepCounterfactualError("settled differential audit did not pass")
    for field in (
        "audit_payload_sha256",
        "ordered_sample_identity_sha256",
        "integration_state_sha256",
        "specification_sha256",
        "binding_sha256",
        "classification_ledger_sha256",
    ):
        _sha256(validation[field], "differential validation." + field)
    if (
        validation["ordered_sample_identity_sha256"] != _sha(list(samples))
        or validation["integration_state_sha256"]
        != binding["settled_mjstate_integration_sha256"]
        or validation["binding_sha256"]
        != binding["settled_link56_differential_audit_binding_sha256"]
        or validation["classification_ledger_sha256"]
        != binding[
            "settled_link56_differential_audit_classification_ledger_sha256"
        ]
    ):
        raise OneStepCounterfactualError("settled differential audit binding differs")
    counts = _mapping(validation["counts"], "differential validation counts")
    if counts.get("sample_count") != sample_count or counts.get("passed_sample_count") != sample_count:
        raise OneStepCounterfactualError("differential audit sample population differs")
    return {
        "binding": binding,
        "samples": list(samples),
        "components": list(components),
        "sample_count": sample_count,
        "arm_dof_indices": parsed_dofs,
        "coverage_radius": maximum,
        "full_robot_measurement_sampling": binding[
            "full_robot_measurement_sampling"
        ],
        "resolved_geometry": _mapping(binding["resolved_geometry"], "resolved geometry"),
    }


def _validate_target_contact(
    value: Any, identification: Mapping[str, Any]
) -> Mapping[str, Any]:
    contact = _mapping(value, "target contact")
    _same(contact, identification["first_link56_contact"], "target contact authority")
    required = (
        "observation_index",
        "source_phase",
        "mujoco_contact_index",
        "robot_geom_id",
        "robot_geom_name",
        "robot_body_id",
        "robot_body_name",
        "obstacle_geom_id",
        "obstacle_geom_name",
        "obstacle_body_id",
        "obstacle_body_name",
        "contact_distance_m",
        "is_physical_nonpositive_distance_contact",
    )
    for field in required:
        if field not in contact:
            raise OneStepCounterfactualError("target contact lacks %s" % field)
    _integer(contact["observation_index"], "target contact observation", 0)
    for field in (
        "mujoco_contact_index",
        "robot_geom_id",
        "robot_body_id",
        "obstacle_geom_id",
        "obstacle_body_id",
    ):
        _integer(contact[field], "target contact." + field, 0)
    for field in (
        "robot_geom_name",
        "robot_body_name",
        "obstacle_geom_name",
        "obstacle_body_name",
    ):
        if not isinstance(contact[field], str) or not contact[field]:
            raise OneStepCounterfactualError("target contact.%s is empty" % field)
    distance = _number(contact["contact_distance_m"], "target contact distance")
    if contact["is_physical_nonpositive_distance_contact"] is not (distance <= 0.0):
        raise OneStepCounterfactualError("target contact physical flag differs")
    if distance > 0.0:
        raise OneStepCounterfactualError("target contact is not a MuJoCo physical contact")
    contact_physical_boundary_index(
        contact["observation_index"], str(contact["source_phase"])
    )
    return contact


def _validate_source_restore(
    value: Any, state_B: Mapping[str, Any]
) -> Mapping[str, Any]:
    restore = _exact_keys(
        value,
        (
            "method",
            "integration_state_sha256",
            "flattened_simulator_state_sha256",
            "exact_official_integration_state",
            "exact_flattened_simulator_state",
            "wrapper_bookkeeping",
        ),
        "source restore",
    )
    if (
        restore["method"]
        != "mj_setState_forward_mj_setState_plus_wrapper_bookkeeping"
        or restore["integration_state_sha256"] != state_B["integration_state_sha256"]
        or restore["flattened_simulator_state_sha256"]
        != state_B["flattened_simulator_state_sha256"]
        or restore["exact_official_integration_state"] is not True
        or restore["exact_flattened_simulator_state"] is not True
        or _canonical(restore["wrapper_bookkeeping"])
        != _canonical(state_B["wrapper_bookkeeping"])
    ):
        raise OneStepCounterfactualError("source restore differs from exact boundary B")
    return restore


def _validate_gripper(
    value: Any, *, boundary: int, historical_action_count: int
) -> Mapping[str, Any]:
    gripper = _exact_keys(
        value,
        (
            "source_action_index",
            "source_action_7d",
            "source_action_sha256",
            "exact_gripper_value",
            "exact_gripper_value_sha256",
        ),
        "gripper evidence",
    )
    index = _integer(gripper["source_action_index"], "source action index", 0)
    if index != boundary // 25 or index >= historical_action_count:
        raise OneStepCounterfactualError("gripper source action index differs")
    action = _vector(gripper["source_action_7d"], 7, "source action")
    if _float64_array_sha256(action) != _sha256(
        gripper["source_action_sha256"], "source action hash"
    ):
        raise OneStepCounterfactualError("source action hash does not reconstruct")
    gripper_value = _number(gripper["exact_gripper_value"], "source gripper")
    if gripper_value != action[6]:
        raise OneStepCounterfactualError("gripper differs from source action dimension 6")
    if _sha(gripper_value) != _sha256(
        gripper["exact_gripper_value_sha256"], "gripper scalar hash"
    ):
        raise OneStepCounterfactualError("gripper scalar hash does not reconstruct")
    return gripper


def _validate_warning(
    value: Any,
    *,
    target_contact: Mapping[str, Any],
    contact_boundary: int,
    period: int,
    timestep: float,
    identification: Mapping[str, Any],
) -> Mapping[str, Any]:
    warning_auth = _exact_keys(
        value,
        (
            "primary_registered_warning",
            "warning_physical_boundary",
            "contact_physical_boundary",
            "lead_physics_substeps",
            "lead_time_s",
            "next_scheduled_filter_boundary",
            "next_filter_boundary_strictly_before_contact",
            "invalid_query_warning_used_for_authorization",
            "passed",
        ),
        "warning authorization",
    )
    warning = _mapping(
        warning_auth["primary_registered_warning"], "primary registered warning"
    )
    _same(warning, identification["primary_registered_warning"], "warning authority")
    for field in ("observation_index", "geom_id", "signal_kind", "evidence"):
        if field not in warning:
            raise OneStepCounterfactualError("primary warning lacks %s" % field)
    warning_observation = _integer(
        warning["observation_index"], "warning observation", 0
    )
    warning_boundary = warning_observation + 1
    warning_geom = _integer(warning["geom_id"], "warning geom", 0)
    if warning["signal_kind"] != "cbf_lhs_negative" or warning_geom != target_contact["robot_geom_id"]:
        raise OneStepCounterfactualError("warning is not negative CBF lhs on contact geom")
    evidence = _mapping(warning["evidence"], "warning evidence")
    lhs = evidence.get("observed_cbf_lhs_m2_per_s")
    if lhs is None or _number(lhs, "warning observed CBF lhs") >= 0.0:
        raise OneStepCounterfactualError("warning evidence is not a negative CBF lhs")
    lead = contact_boundary - warning_boundary
    next_filter = first_filter_boundary_at_or_after_warning(
        warning_boundary, period
    )
    expected = {
        "warning_physical_boundary": warning_boundary,
        "contact_physical_boundary": contact_boundary,
        "lead_physics_substeps": lead,
        "lead_time_s": lead * timestep,
        "next_scheduled_filter_boundary": next_filter,
        "next_filter_boundary_strictly_before_contact": next_filter < contact_boundary,
        "invalid_query_warning_used_for_authorization": False,
        "passed": bool(lead > 0 and next_filter < contact_boundary),
    }
    for field, expected_value in expected.items():
        if isinstance(expected_value, float):
            _close(warning_auth[field], expected_value, "warning." + field)
        elif warning_auth[field] != expected_value:
            raise OneStepCounterfactualError("warning.%s does not reconstruct" % field)
    if expected["passed"] is not True:
        raise OneStepCounterfactualError("warning is not actionable before contact")
    return warning_auth


def _validate_boundary_preflight(
    value: Any,
    *,
    boundary_start: Mapping[str, Any],
    registered_admissibility: Mapping[str, Any],
) -> Mapping[str, Any]:
    preflight = _exact_keys(
        value,
        (
            "strict_all_sample_h_positive",
            "strict_D_sim_positive",
            "zero_robot_selected_obstacle_contact",
            "invalid_field_query_count",
            "selected_obstacle_static_observed",
            "selected_obstacle_static_thresholds",
            "selected_obstacle_static_admissible",
            "QP_or_arm_physics_executed_before_this_gate",
            "passed",
            "measurement",
        ),
        "boundary-B admissibility",
    )
    _same(preflight["measurement"], boundary_start, "boundary preflight measurement")
    h = _sequence(boundary_start["ordered_sample_h_m2"], "boundary-B h")
    strict_h = bool(h and all(item is not None and _number(item, "boundary h") > 0.0 for item in h))
    strict_d = _number(boundary_start["minimum_D_sim_m"], "boundary D_sim") > 0.0
    zero_contact = boundary_start["any_robot_to_selected_obstacle_contact_present"] is False
    invalid_count = _integer(
        boundary_start["invalid_field_query_count"], "boundary invalid query count", 0
    )
    observed = {
        "translation_m": boundary_start["selected_obstacle_translation_drift_m"],
        "rotation_rad": boundary_start["selected_obstacle_rotation_drift_rad"],
        "surface_m": boundary_start["selected_obstacle_surface_drift_m"],
        "linear_speed_m_s": boundary_start[
            "selected_obstacle_max_body_linear_speed_m_s"
        ],
        "angular_speed_rad_s": boundary_start[
            "selected_obstacle_max_body_angular_speed_rad_s"
        ],
    }
    _same(preflight["selected_obstacle_static_observed"], observed, "boundary static observation")
    thresholds = _exact_keys(
        preflight["selected_obstacle_static_thresholds"],
        ("translation_m", "rotation_rad", "surface_m", "linear_speed_m_s", "angular_speed_rad_s"),
        "boundary static thresholds",
    )
    expected_thresholds = {
        "translation_m": registered_admissibility[
            "max_selected_geom_translation_drift_m"
        ],
        "rotation_rad": registered_admissibility[
            "max_selected_geom_rotation_drift_rad"
        ],
        "surface_m": registered_admissibility[
            "max_selected_geom_surface_drift_m"
        ],
        "linear_speed_m_s": registered_admissibility[
            "max_selected_body_linear_speed_m_s"
        ],
        "angular_speed_rad_s": registered_admissibility[
            "max_selected_body_angular_speed_rad_s"
        ],
    }
    _same(thresholds, expected_thresholds, "boundary registered static thresholds")
    static = all(
        _number(observed[key], "boundary static " + key)
        <= _number(thresholds[key], "boundary threshold " + key)
        for key in thresholds
    )
    expected_flags = {
        "strict_all_sample_h_positive": strict_h,
        "strict_D_sim_positive": strict_d,
        "zero_robot_selected_obstacle_contact": zero_contact,
        "invalid_field_query_count": invalid_count,
        "selected_obstacle_static_admissible": static,
        "QP_or_arm_physics_executed_before_this_gate": False,
        "passed": bool(
            strict_h
            and strict_d
            and zero_contact
            and invalid_count
            <= int(registered_admissibility["maximum_invalid_field_queries"])
            and static
        ),
    }
    for field, expected in expected_flags.items():
        if preflight[field] != expected:
            raise OneStepCounterfactualError("boundary preflight.%s differs" % field)
    if preflight["passed"] is not True:
        raise OneStepCounterfactualError("boundary B is not a strict safe start")
    return preflight


def _validate_source_boundary(
    value: Any,
    *,
    protocol_values: Mapping[str, Any],
    authority: Mapping[str, Any],
    physical_model: Mapping[str, Any],
    outcome_kind: str,
) -> Dict[str, Any]:
    source = _exact_keys(
        value,
        (
            "target_contact",
            "contact_physical_boundary_C",
            "filter_boundary_B",
            "exact_prefix",
            "source_restore",
            "gripper_evidence",
            "warning_authorization",
            "boundary_B_admissibility",
            "shadow_construction_binding",
        ),
        "source_boundary",
    )
    dynamic = authority["dynamic_authority"]
    identification = dynamic["identification"]
    target = _validate_target_contact(source["target_contact"], identification)
    contact_boundary = contact_physical_boundary_index(
        target["observation_index"], str(target["source_phase"])
    )
    if _integer(source["contact_physical_boundary_C"], "contact boundary C", 1) != contact_boundary:
        raise OneStepCounterfactualError("contact boundary C differs")
    warning = _validate_warning(
        source["warning_authorization"],
        target_contact=target,
        contact_boundary=contact_boundary,
        period=protocol_values["period"],
        timestep=protocol_values["timestep"],
        identification=identification,
    )
    boundary = _integer(
        warning["next_scheduled_filter_boundary"],
        "first scheduled filter boundary after warning",
        0,
    )
    if _integer(source["filter_boundary_B"], "filter boundary B", 0) != boundary:
        raise OneStepCounterfactualError("filter boundary B differs")
    if not boundary < contact_boundary:
        raise OneStepCounterfactualError(
            "warning-authorized filter boundary is not strictly before contact"
        )
    prefix = _validate_exact_prefix(
        source["exact_prefix"],
        boundary=boundary,
        horizon=protocol_values["horizon"],
        physical_model=physical_model,
    )
    external_callback_ledger = dynamic["parity"][
        "official_integration_state_sha256_ledger"
    ]
    if prefix["observed_callback_sha256_ledger"] != external_callback_ledger[: boundary + protocol_values["horizon"]]:
        raise OneStepCounterfactualError("source prefix differs from external parity ledger")
    identification_callback_ledger = dynamic["identification"][
        "callback_state_read_only_after_sha256_ledger"
    ]
    if prefix["observed_callback_sha256_ledger"] != identification_callback_ledger[: boundary + protocol_values["horizon"]]:
        raise OneStepCounterfactualError("source prefix differs from external identification ledger")
    if (
        prefix["historical_executed_action_sequence_sha256"]
        != dynamic["historical"]["executed_action_sequence_sha256"]
    ):
        raise OneStepCounterfactualError("source prefix action authority differs")
    _validate_source_restore(source["source_restore"], prefix["state_at_B"])
    gripper = _validate_gripper(
        source["gripper_evidence"],
        boundary=boundary,
        historical_action_count=dynamic["historical"]["action_count"],
    )
    samples = _validate_samples(source["shadow_construction_binding"])
    if outcome_kind == "preflight_inadmissible_nominal_velocity":
        if source["boundary_B_admissibility"] is not None:
            raise OneStepCounterfactualError(
                "inadmissible nominal preflight must not execute boundary-B field preflight"
            )
    elif not isinstance(source["boundary_B_admissibility"], Mapping):
        raise OneStepCounterfactualError(
            "paired execution lacks boundary-B admissibility evidence"
        )
    binding = samples["binding"]
    if (
        binding["contact_model_authority_sha256"]
        != identification["contact_model_authority_sha256"]
        or _canonical(binding["physical_model"])
        != _canonical(identification["physical_model"])
        or _canonical(binding["resolved_geometry"])
        != _canonical(identification["resolved_geometry"])
        or _canonical(binding["field_bundle_hashes"])
        != _canonical(identification["field_bundle_hashes"])
        or binding["ordered_protected_sample_ledger_sha256"]
        != identification["ordered_protected_sample_ledger_sha256"]
        or binding["protected_sample_count"] != identification["protected_sample_count"]
        or binding["arm_dof_indices"] != identification["arm_dof_indices"]
        or binding["settled_mjstate_integration_sha256"]
        != identification["settled_mjstate_integration_sha256"]
        or binding["settled_link56_differential_audit_binding_sha256"]
        != identification["differential_binding_sha256"]
        or binding[
            "settled_link56_differential_audit_classification_ledger_sha256"
        ]
        != identification["differential_classification_ledger_sha256"]
        or _canonical(binding["settled_link56_differential_audit_validation"])
        != _canonical(identification["differential_validation"])
    ):
        raise OneStepCounterfactualError("shadow construction differs from external identification")
    full_sampling_projection = {
        key: item
        for key, item in binding["full_robot_measurement_sampling"].items()
        if key != "sample_ledger"
    }
    if _canonical(full_sampling_projection) != _canonical(
        identification["full_robot_measurement_sampling"]
    ):
        raise OneStepCounterfactualError(
            "fresh full-robot sampling certificate differs from identification"
        )
    if boundary == 0:
        if prefix["state_at_B"]["integration_state_sha256"] != binding["settled_mjstate_integration_sha256"]:
            raise OneStepCounterfactualError("settled boundary state differs from construction")
    return {
        "source": source,
        "target": target,
        "contact_boundary": contact_boundary,
        "boundary": boundary,
        "prefix": prefix,
        "gripper": gripper,
        "warning": warning,
        "samples": samples,
    }


def _validate_nominal_velocity(
    value: Any,
    *,
    source: Mapping[str, Any],
    protocol: Mapping[str, Any],
    registered_parameters: Mapping[str, Any],
) -> Dict[str, Any]:
    nominal = _exact_keys(
        value,
        (
            "method",
            "interval_s",
            "qpos_B_sha256",
            "qpos_B_plus_5_sha256",
            "full_nv_count",
            "arm_dof_indices",
            "arm_qpos_indices",
            "arm_joint_types",
            "arm_joint_qpos_widths",
            "qdot_nom_full_nv",
            "qdot_nom_arm_slice",
            "instantaneous_source_qvel_full_nv_at_B",
            "instantaneous_source_arm_qvel_at_B",
            "estimated_minus_instantaneous_arm_qvel",
            "registered_lower_rad_s",
            "registered_upper_rad_s",
            "finite",
            "arm_velocity_within_registered_bounds",
            "bound_violation_indices",
            "lower_bound_excess_rad_s",
            "upper_bound_excess_rad_s",
            "maximum_bound_excess_rad_s",
            "out_of_bounds_policy",
            "hidden_clipping_applied",
            "nominal_derivation_sha256",
        ),
        "nominal_velocity_estimate",
    )
    prefix = source["prefix"]
    state_B = prefix["state_at_B"]
    state_B5 = prefix["state_at_B_plus_5"]
    if (
        nominal["method"] != "mujoco_mj_differentiatePos_full_nv"
        or nominal["interval_s"]
        != float(registered_parameters["cadence"]["physics_timestep_s"])
        * int(
            registered_parameters["cadence"]["active"][
                "physics_substeps_per_filter_update"
            ]
        )
        or nominal["qpos_B_sha256"] != state_B["qpos_sha256"]
        or nominal["qpos_B_plus_5_sha256"] != state_B5["qpos_sha256"]
        or nominal["out_of_bounds_policy"] != "inadmissible_no_hidden_clipping"
        or nominal["hidden_clipping_applied"] is not False
        or nominal["finite"] is not True
    ):
        raise OneStepCounterfactualError("nominal velocity estimator binding differs")
    dofs = source["samples"]["arm_dof_indices"]
    if nominal["arm_dof_indices"] != dofs:
        raise OneStepCounterfactualError("nominal velocity arm DOF slice differs")
    full_count = _integer(nominal["full_nv_count"], "nominal full nv count", 7)
    qdot_full = _vector(nominal["qdot_nom_full_nv"], full_count, "qdot_nom_full_nv")
    qdot_arm = _vector(nominal["qdot_nom_arm_slice"], 7, "qdot_nom_arm_slice")
    if [qdot_full[index] for index in dofs] != qdot_arm:
        raise OneStepCounterfactualError("nominal arm velocity is not the full-nv slice")
    qpos_indices_raw = _sequence(nominal["arm_qpos_indices"], "arm qpos indices")
    if len(qpos_indices_raw) != 7:
        raise OneStepCounterfactualError("arm qpos index count differs")
    qpos_indices = [_integer(item, "arm qpos index", 0) for item in qpos_indices_raw]
    if len(set(qpos_indices)) != 7:
        raise OneStepCounterfactualError("arm qpos indices are duplicated")
    estimator = protocol["nominal_velocity_estimator"]
    if qpos_indices != estimator.get("expected_arm_qpos_indices"):
        raise OneStepCounterfactualError(
            "arm qpos indices differ from external protocol authority"
        )
    if nominal["arm_joint_types"] != ["hinge"] * 7 or nominal[
        "arm_joint_qpos_widths"
    ] != [1] * 7:
        raise OneStepCounterfactualError("arm joints are not seven scalar hinges")
    qpos_B = [_number(item, "state-B qpos") for item in state_B["qpos"]]
    qpos_B5 = [_number(item, "state-B+5 qpos") for item in state_B5["qpos"]]
    if len(qpos_B) != len(qpos_B5) or any(index >= len(qpos_B) for index in qpos_indices):
        raise OneStepCounterfactualError("arm qpos indices exceed exact endpoint state")
    reconstructed_arm = [
        (qpos_B5[index] - qpos_B[index]) / 0.01 for index in qpos_indices
    ]
    for observed, expected in zip(qdot_arm, reconstructed_arm):
        _close(observed, expected, "scalar-hinge nominal velocity reconstruction")
    source_qvel = _vector(
        nominal["instantaneous_source_qvel_full_nv_at_B"],
        full_count,
        "instantaneous source qvel",
    )
    if source_qvel != _vector(state_B["qvel"], full_count, "state-B qvel"):
        raise OneStepCounterfactualError("instantaneous qvel differs from state B")
    instantaneous_arm = _vector(
        nominal["instantaneous_source_arm_qvel_at_B"], 7, "instantaneous arm qvel"
    )
    if [source_qvel[index] for index in dofs] != instantaneous_arm:
        raise OneStepCounterfactualError("instantaneous arm qvel slice differs")
    delta = _vector(
        nominal["estimated_minus_instantaneous_arm_qvel"],
        7,
        "estimated minus instantaneous qvel",
    )
    for index in range(7):
        _close(delta[index], qdot_arm[index] - instantaneous_arm[index], "nominal velocity delta")
    lower = _vector(nominal["registered_lower_rad_s"], 7, "nominal lower")
    upper = _vector(nominal["registered_upper_rad_s"], 7, "nominal upper")
    if (
        lower != estimator["arm_velocity_lower_rad_s"]
        or upper != estimator["arm_velocity_upper_rad_s"]
        or lower != registered_parameters["qp"]["velocity_lower_rad_s"]
        or upper != registered_parameters["qp"]["velocity_upper_rad_s"]
    ):
        raise OneStepCounterfactualError(
            "nominal velocity bounds differ from protocol/runtime authority"
        )
    lower_excess = [
        max(lower[index] - value, 0.0)
        for index, value in enumerate(qdot_arm)
    ]
    upper_excess = [
        max(value - upper[index], 0.0)
        for index, value in enumerate(qdot_arm)
    ]
    violation_indices = [
        index
        for index in range(7)
        if lower_excess[index] > 0.0 or upper_excess[index] > 0.0
    ]
    recorded_indices = _sequence(
        nominal["bound_violation_indices"], "nominal bound violation indices"
    )
    if [
        _integer(item, "nominal bound violation index", 0)
        for item in recorded_indices
    ] != violation_indices:
        raise OneStepCounterfactualError(
            "nominal bound-violation indices do not reconstruct"
        )
    for field, expected_values in (
        ("lower_bound_excess_rad_s", lower_excess),
        ("upper_bound_excess_rad_s", upper_excess),
    ):
        observed_values = _vector(nominal[field], 7, "nominal " + field)
        for observed, expected in zip(observed_values, expected_values):
            _close(observed, expected, "nominal " + field)
    maximum_excess = max(lower_excess + upper_excess)
    _close(
        nominal["maximum_bound_excess_rad_s"],
        maximum_excess,
        "nominal maximum bound excess",
    )
    within_bounds = not violation_indices
    if nominal["arm_velocity_within_registered_bounds"] is not within_bounds:
        raise OneStepCounterfactualError(
            "nominal velocity bound classification differs"
        )
    unhashed = dict(nominal)
    observed_derivation = _sha256(
        unhashed.pop("nominal_derivation_sha256"), "nominal derivation hash"
    )
    expected_derivation = _sha(unhashed)
    if observed_derivation != expected_derivation:
        raise OneStepCounterfactualError("nominal derivation hash does not reconstruct")
    return {
        "record": nominal,
        "qdot_full": qdot_full,
        "qdot_arm": qdot_arm,
        "arm_qpos_indices": qpos_indices,
        "within_bounds": within_bounds,
        "violation_indices": violation_indices,
        "lower_excess": lower_excess,
        "upper_excess": upper_excess,
        "maximum_excess": maximum_excess,
    }


def _validate_qp(
    value: Any,
    *,
    samples: Mapping[str, Any],
    nominal_qdot: Sequence[float],
    boundary_qpos: Sequence[float],
    arm_qpos_indices: Sequence[int],
    target_geom: int,
    protocol: Mapping[str, Any],
    registered_parameters: Mapping[str, Any],
    warning_authorization: Mapping[str, Any],
) -> Dict[str, Any]:
    qp = _exact_keys(
        value,
        (
            "arm_dof_indices",
            "ordered_sample_identity",
            "ordered_sample_count",
            "ordered_sample_ledger_sha256",
            "ordered_sample_points_world",
            "ordered_sample_point_jacobians",
            "ordered_sample_h_m2",
            "ordered_sample_grad_h_m",
            "ordered_sample_D_opt_m",
            "velocity_lower_rad_s",
            "velocity_upper_rad_s",
            "one_CBF_row_per_exact_bound_sample",
            "joint_velocity_bound_rows",
            "joint_position_constraint_rows",
            "nominal_CBF_residuals_m2_per_s",
            "safe_CBF_residuals_m2_per_s",
            "qdot_safe",
            "correction_norm",
            "QP_status",
            "QP_iterations",
            "QP_postcheck",
            "QP_reason",
            "trend_sample_index",
            "trend_sample_id",
            "trend_sample_robot_geom_id",
            "trend_sample_nominal_residual_m2_per_s",
        ),
        "boundary_B_filter",
    )
    count = samples["sample_count"]
    if (
        qp["arm_dof_indices"] != samples["arm_dof_indices"]
        or qp["ordered_sample_identity"] != samples["samples"]
        or _integer(qp["ordered_sample_count"], "QP sample count", 1) != count
        or qp["ordered_sample_ledger_sha256"]
        != samples["binding"]["ordered_protected_sample_ledger_sha256"]
    ):
        raise OneStepCounterfactualError("QP sample population differs")
    points_raw = _sequence(qp["ordered_sample_points_world"], "QP points")
    jacobians_raw = _sequence(qp["ordered_sample_point_jacobians"], "QP Jacobians")
    gradients_raw = _sequence(qp["ordered_sample_grad_h_m"], "QP gradients")
    if len(points_raw) != count or len(jacobians_raw) != count or len(gradients_raw) != count:
        raise OneStepCounterfactualError("QP point/Jacobian population differs")
    points = [_vector(row, 3, "QP point") for row in points_raw]
    jacobians = [_matrix(row, 3, 7, "QP point Jacobian") for row in jacobians_raw]
    gradients = [_vector(row, 3, "QP gradient") for row in gradients_raw]
    h = _vector(qp["ordered_sample_h_m2"], count, "QP h")
    d_opt = _vector(qp["ordered_sample_D_opt_m"], count, "QP D_opt")
    if any(value <= 0.0 for value in h) or any(value < 0.0 for value in d_opt):
        raise OneStepCounterfactualError("boundary-B QP is not a valid strict safe start")
    rows = []
    for gradient, jacobian in zip(gradients, jacobians):
        rows.append(
            [sum(gradient[axis] * jacobian[axis][joint] for axis in range(3)) for joint in range(7)]
        )
    cbf = _exact_keys(
        qp["one_CBF_row_per_exact_bound_sample"],
        ("row_count", "sample_count", "rows_m_per_rad", "lower_bounds_m2_per_s", "alpha_gain_per_s", "no_slack"),
        "QP CBF rows",
    )
    if (
        _integer(cbf["row_count"], "QP CBF row count", 1) != count
        or _integer(cbf["sample_count"], "QP CBF sample count", 1) != count
        or cbf["no_slack"] is not True
    ):
        raise OneStepCounterfactualError("QP is not one hard row per sample")
    recorded_rows = _sequence(cbf["rows_m_per_rad"], "recorded CBF rows")
    if len(recorded_rows) != count:
        raise OneStepCounterfactualError("recorded CBF row count differs")
    for index in range(count):
        observed = _vector(recorded_rows[index], 7, "recorded CBF row")
        for column in range(7):
            _close(observed[column], rows[index][column], "CBF row arithmetic")
    alpha = _number(cbf["alpha_gain_per_s"], "CBF alpha")
    if alpha != _number(
        registered_parameters["cbf"]["alpha_gain_per_s"],
        "registered CBF alpha",
    ):
        raise OneStepCounterfactualError("CBF alpha differs from runtime authority")
    lower_rows = _vector(cbf["lower_bounds_m2_per_s"], count, "CBF lower rows")
    nominal_residuals = _vector(
        qp["nominal_CBF_residuals_m2_per_s"], count, "nominal CBF residuals"
    )
    safe_residuals = _vector(
        qp["safe_CBF_residuals_m2_per_s"], count, "safe CBF residuals"
    )
    qsafe = _vector(qp["qdot_safe"], 7, "qdot_safe")
    for index in range(count):
        _close(lower_rows[index], -alpha * h[index], "CBF lower bound arithmetic")
        _close(
            nominal_residuals[index],
            _dot(rows[index], nominal_qdot) + alpha * h[index],
            "nominal CBF residual arithmetic",
        )
        _close(
            safe_residuals[index],
            _dot(rows[index], qsafe) + alpha * h[index],
            "safe CBF residual arithmetic",
        )
    velocity_rows = _exact_keys(
        qp["joint_velocity_bound_rows"],
        ("physical_lower_rad_s", "physical_upper_rad_s", "final_lower_rad_s", "final_upper_rad_s"),
        "QP velocity rows",
    )
    position_rows = _exact_keys(
        qp["joint_position_constraint_rows"],
        (
            "q_arm_rad",
            "q_min_rad",
            "q_max_rad",
            "position_margin_rad",
            "allowed_lower_rad",
            "allowed_upper_rad",
            "joint_limit_alpha_per_s",
            "continuous_lower_rad_s",
            "continuous_upper_rad_s",
            "one_step_dt_s",
            "one_step_lower_rad_s",
            "one_step_upper_rad_s",
        ),
        "QP joint position rows",
    )
    physical_lower = _vector(velocity_rows["physical_lower_rad_s"], 7, "physical lower")
    physical_upper = _vector(velocity_rows["physical_upper_rad_s"], 7, "physical upper")
    estimator = protocol["nominal_velocity_estimator"]
    registered_qp = registered_parameters["qp"]
    if (
        physical_lower != estimator["arm_velocity_lower_rad_s"]
        or physical_upper != estimator["arm_velocity_upper_rad_s"]
        or physical_lower != registered_qp["velocity_lower_rad_s"]
        or physical_upper != registered_qp["velocity_upper_rad_s"]
    ):
        raise OneStepCounterfactualError(
            "QP physical bounds differ from protocol/runtime authority"
        )
    final_lower = _vector(velocity_rows["final_lower_rad_s"], 7, "final lower")
    final_upper = _vector(velocity_rows["final_upper_rad_s"], 7, "final upper")
    if qp["velocity_lower_rad_s"] != final_lower or qp["velocity_upper_rad_s"] != final_upper:
        raise OneStepCounterfactualError("QP final bound copies differ")
    q_arm = _vector(position_rows["q_arm_rad"], 7, "QP q_arm")
    q_min = _vector(position_rows["q_min_rad"], 7, "QP q_min")
    q_max = _vector(position_rows["q_max_rad"], 7, "QP q_max")
    qp_protocol = _mapping(protocol.get("qp_execution"), "protocol QP execution")
    if qp_protocol.get("joint_limit_match_absolute_tolerance_rad") != 1e-12:
        raise OneStepCounterfactualError(
            "protocol joint-limit match tolerance differs"
        )
    expected_q_min = _vector(
        qp_protocol.get("expected_arm_q_min_rad"),
        7,
        "protocol expected arm q_min",
    )
    expected_q_max = _vector(
        qp_protocol.get("expected_arm_q_max_rad"),
        7,
        "protocol expected arm q_max",
    )
    if q_min != expected_q_min or q_max != expected_q_max:
        raise OneStepCounterfactualError(
            "QP joint limits differ from external protocol authority"
        )
    expected_q_arm = [
        _number(boundary_qpos[index], "boundary-B arm qpos")
        for index in arm_qpos_indices
    ]
    if q_arm != expected_q_arm:
        raise OneStepCounterfactualError("QP q_arm differs from exact boundary-B qpos")
    margin = _number(position_rows["position_margin_rad"], "QP position margin")
    alpha_joint = _number(position_rows["joint_limit_alpha_per_s"], "QP joint alpha")
    dt = _number(position_rows["one_step_dt_s"], "QP one-step dt")
    registered_dt = _number(
        registered_parameters["cadence"]["physics_timestep_s"],
        "registered physics timestep",
    ) * _integer(
        registered_parameters["cadence"]["active"][
            "physics_substeps_per_filter_update"
        ],
        "registered active filter substeps",
        1,
    )
    if (
        margin != registered_qp["joint_position_margin_rad"]
        or margin != registered_parameters["admissibility"][
            "joint_position_margin_rad"
        ]
        or alpha_joint != registered_qp["joint_limit_alpha_per_s"]
        or dt != registered_dt
        or dt != 0.01
    ):
        raise OneStepCounterfactualError("QP joint-limit parameters differ")
    derived = {
        "allowed_lower_rad": [value + margin for value in q_min],
        "allowed_upper_rad": [value - margin for value in q_max],
        "continuous_lower_rad_s": [-alpha_joint * (q_arm[i] - q_min[i]) for i in range(7)],
        "continuous_upper_rad_s": [alpha_joint * (q_max[i] - q_arm[i]) for i in range(7)],
    }
    derived["one_step_lower_rad_s"] = [
        (derived["allowed_lower_rad"][i] - q_arm[i]) / dt for i in range(7)
    ]
    derived["one_step_upper_rad_s"] = [
        (derived["allowed_upper_rad"][i] - q_arm[i]) / dt for i in range(7)
    ]
    for field, expected_values in derived.items():
        observed_values = _vector(position_rows[field], 7, "QP " + field)
        for observed, expected in zip(observed_values, expected_values):
            _close(observed, expected, "QP " + field)
    expected_lower = [
        max(physical_lower[i], derived["continuous_lower_rad_s"][i], derived["one_step_lower_rad_s"][i])
        for i in range(7)
    ]
    expected_upper = [
        min(physical_upper[i], derived["continuous_upper_rad_s"][i], derived["one_step_upper_rad_s"][i])
        for i in range(7)
    ]
    for index in range(7):
        _close(final_lower[index], expected_lower[index], "QP final lower")
        _close(final_upper[index], expected_upper[index], "QP final upper")
        if final_lower[index] > final_upper[index]:
            raise OneStepCounterfactualError("QP final bounds are contradictory")
        if qsafe[index] < final_lower[index] or qsafe[index] > final_upper[index]:
            raise OneStepCounterfactualError("safe command is outside hard bounds")
    correction = _norm([qsafe[index] - nominal_qdot[index] for index in range(7)])
    _close(qp["correction_norm"], correction, "QP correction norm")
    if qp["QP_reason"] != "solved" or qp["QP_status"] not in ("solved", "solved inaccurate"):
        raise OneStepCounterfactualError("hard QP is not solved")
    _integer(qp["QP_iterations"], "QP iterations", 1)
    _number(
        protocol["acceptance"]["qp_safe_residual_minimum"], "QP residual threshold"
    )
    postcheck = _exact_keys(
        qp["QP_postcheck"],
        (
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
        ),
        "QP postcheck",
    )
    if (
        postcheck.get("status") != qp["QP_status"]
        or postcheck.get("iterations") != qp["QP_iterations"]
        or postcheck.get("input_constraint_count") != count
        or postcheck.get("solver") != registered_qp["solver"]
    ):
        raise OneStepCounterfactualError("QP postcheck identity differs")
    _integer(postcheck["status_value"], "QP status value")
    if _number(postcheck["solve_time_seconds"], "QP solve time") < 0.0:
        raise OneStepCounterfactualError("QP solve time is negative")
    if (
        _integer(postcheck["solved_constraint_count"], "QP solved rows", 0)
        != len(nonzero_indexes := [
            index
            for index, row in enumerate(rows)
            if max(abs(value) for value in row) > 0.0
        ])
        or _integer(postcheck["trivial_zero_constraint_count"], "QP zero rows", 0)
        != count - len(nonzero_indexes)
    ):
        raise OneStepCounterfactualError("QP postcheck row counts differ")
    row_scales = [max(abs(value) for value in row) for row in rows]
    nonzero_indexes = [index for index, scale in enumerate(row_scales) if scale > 0.0]
    minimum_raw = (
        min(safe_residuals[index] for index in nonzero_indexes)
        if nonzero_indexes
        else None
    )
    minimum_normalized = (
        min(safe_residuals[index] / row_scales[index] for index in nonzero_indexes)
        if nonzero_indexes
        else None
    )
    if minimum_raw is not None:
        _close(
            postcheck["minimum_raw_cbf_residual_m2_per_s"],
            minimum_raw,
            "QP minimum raw CBF residual",
        )
        _close(
            postcheck["minimum_nonzero_row_scale_m2_per_rad"],
            min(row_scales[index] for index in nonzero_indexes),
            "QP minimum row scale",
        )
        _close(
            postcheck["maximum_nonzero_row_scale_m2_per_rad"],
            max(row_scales[index] for index in nonzero_indexes),
            "QP maximum row scale",
        )
    elif any(
        postcheck[field] is not None
        for field in (
            "minimum_raw_cbf_residual_m2_per_s",
            "minimum_normalized_cbf_residual",
            "minimum_nonzero_row_scale_m2_per_rad",
            "maximum_nonzero_row_scale_m2_per_rad",
        )
    ):
        raise OneStepCounterfactualError("zero-row QP postcheck must use null minima")
    if minimum_normalized is not None:
        _close(
            postcheck["minimum_normalized_cbf_residual"],
            minimum_normalized,
            "QP minimum normalized CBF residual",
        )
    _close(postcheck["correction_l2_rad_s"], correction, "QP postcheck correction")
    nominal_normalized = (
        min(
            nominal_residuals[index] / row_scales[index]
            for index in nonzero_indexes
        )
        if nonzero_indexes
        else None
    )
    nominal_raw = (
        min(nominal_residuals[index] for index in nonzero_indexes)
        if nonzero_indexes
        else None
    )
    for field, expected in (
        ("nominal_minimum_normalized_cbf_residual", nominal_normalized),
        ("nominal_minimum_raw_cbf_residual_m2_per_s", nominal_raw),
    ):
        if expected is None:
            if postcheck[field] is not None:
                raise OneStepCounterfactualError("QP nominal null residual differs")
        else:
            _close(postcheck[field], expected, "QP " + field)
    nominal_feasible = bool(
        (
            nominal_normalized is None
            or nominal_normalized
            >= -float(registered_qp["postcheck_cbf_tolerance"])
        )
        and all(
            nominal_qdot[index]
            >= final_lower[index]
            - float(registered_qp["postcheck_bound_tolerance_rad_s"])
            and nominal_qdot[index]
            <= final_upper[index]
            + float(registered_qp["postcheck_bound_tolerance_rad_s"])
            for index in range(7)
        )
    )
    if postcheck["nominal_feasible"] is not nominal_feasible:
        raise OneStepCounterfactualError("QP nominal-feasible flag differs")
    bound_violation = _number(
        postcheck.get("maximum_velocity_bound_violation_rad_s", 0.0),
        "QP bound violation",
    )
    if (
        bound_violation < 0.0
        or bound_violation
        > float(registered_qp["postcheck_bound_tolerance_rad_s"])
    ):
        raise OneStepCounterfactualError("QP bound postcheck failed")
    if (
        minimum_normalized is not None
        and minimum_normalized
        < -float(registered_qp["postcheck_cbf_tolerance"])
    ):
        raise OneStepCounterfactualError("QP internal normalized postcheck failed")
    trend_index = _integer(qp["trend_sample_index"], "trend sample index", 0)
    if trend_index >= count:
        raise OneStepCounterfactualError("trend sample index is outside population")
    trend_sample = samples["samples"][trend_index]
    warning = _mapping(
        warning_authorization["primary_registered_warning"],
        "accepted primary warning",
    )
    warning_evidence = _mapping(warning.get("evidence"), "primary warning evidence")
    warning_sample_id = _integer(
        warning_evidence.get("sample_id"), "primary warning sample id", 0
    )
    warning_geom_id = _integer(warning.get("geom_id"), "primary warning geom id", 0)
    warning_body_id = _integer(warning.get("body_id"), "primary warning body id", 0)
    if (
        qp["trend_sample_id"] != trend_sample["sample_id"]
        or qp["trend_sample_robot_geom_id"] != trend_sample["geom_id"]
        or trend_sample["geom_id"] != target_geom
        or trend_sample["sample_id"] != warning_sample_id
        or trend_sample["geom_id"] != warning_geom_id
        or trend_sample["body_id"] != warning_body_id
        or warning_evidence.get("geom_id") != warning_geom_id
        or warning_evidence.get("body_id") != warning_body_id
    ):
        raise OneStepCounterfactualError(
            "dangerous trend sample differs from the primary warning identity"
        )
    _close(
        qp["trend_sample_nominal_residual_m2_per_s"],
        nominal_residuals[trend_index],
        "trend sample residual",
    )
    candidates = list(range(count))
    if trend_index != min(candidates, key=lambda index: nominal_residuals[index]):
        raise OneStepCounterfactualError(
            "dangerous warning sample is not the boundary-B residual minimum"
        )
    return {
        "record": qp,
        "h": h,
        "gradients": gradients,
        "jacobians": jacobians,
        "rows": rows,
        "nominal_residuals": nominal_residuals,
        "safe_residuals": safe_residuals,
        "qdot_safe": qsafe,
        "correction_norm": correction,
        "trend_index": trend_index,
    }


_CONTACT_RECORD_KEYS = (
    "source_phase",
    "physical_boundary",
    "mujoco_contact_index",
    "mujoco_geom1_id",
    "mujoco_geom1_name",
    "mujoco_body1_id",
    "mujoco_body1_name",
    "mujoco_geom2_id",
    "mujoco_geom2_name",
    "mujoco_body2_id",
    "mujoco_body2_name",
    "contact_distance_m",
    "is_physical_nonpositive_distance_contact",
    "solver_constraint_active",
    "efc_address",
    "contact_includemargin_m",
    "position_world_m",
)

_MEASUREMENT_KEYS = (
    "physical_boundary",
    "integration_state_sha256",
    "integration_state",
    "qpos",
    "qvel",
    "commanded_arm_qdot",
    "normalized_joint_velocity_action_8d",
    "measured_arm_qdot",
    "ordered_sample_h_m2",
    "ordered_sample_grad_h_m",
    "ordered_field_query_validity_and_reason",
    "ordered_sample_D_opt_m",
    "minimum_D_opt_m",
    "ordered_full_robot_sample_distance_m",
    "registered_full_robot_sample_count",
    "registered_full_robot_sample_ledger_sha256",
    "minimum_D_sim_m",
    "interval_union_minimum_D_sim_m",
    "post_state_minimum_D_sim_m",
    "minimum_exact_full_robot_sample_to_current_obstacle_m",
    "certified_full_robot_coverage_radius_m",
    "all_mujoco_contact_pairs_with_distances",
    "exact_ordered_nonpositive_physical_contact_subset",
    "invalid_field_query_count",
    "selected_obstacle_translation_drift_m",
    "selected_obstacle_rotation_drift_rad",
    "selected_obstacle_surface_drift_m",
    "selected_obstacle_body_velocity_records",
    "selected_obstacle_max_body_linear_speed_m_s",
    "selected_obstacle_max_body_angular_speed_rad_s",
    "velocity_reference",
    "target_contact_present",
    "any_robot_to_selected_obstacle_contact_present",
    "any_shifted_robot_to_selected_obstacle_contact_present",
    "target_physical_contact_records",
    "robot_to_selected_obstacle_physical_contact_records",
    "shifted_robot_to_selected_obstacle_physical_contact_records",
)


def _validate_resolved_geometry(
    value: Mapping[str, Any], target: Mapping[str, Any], protocol: Mapping[str, Any]
) -> Dict[str, Any]:
    resolved = _exact_keys(
        value,
        (
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
        ),
        "resolved geometry",
    )
    integer_lists: Dict[str, List[int]] = {}
    for field in (
        "robot_root_body_ids",
        "robot_body_ids",
        "obstacle_root_body_ids",
        "obstacle_body_ids",
        "link56_body_ids",
        "robot_geom_ids",
        "obstacle_geom_ids",
        "link56_geom_ids",
    ):
        raw = _sequence(resolved[field], "resolved." + field)
        parsed = [_integer(item, "resolved." + field, 0) for item in raw]
        if not parsed or len(parsed) != len(set(parsed)):
            raise OneStepCounterfactualError("resolved.%s is empty or duplicated" % field)
        integer_lists[field] = parsed
    name_pairs = (
        ("robot_body_ids", "robot_body_names"),
        ("obstacle_body_ids", "obstacle_body_names"),
        ("robot_geom_ids", "robot_geom_names"),
        ("obstacle_geom_ids", "obstacle_geom_names"),
        ("link56_geom_ids", "link56_geom_names"),
    )
    for id_field, name_field in name_pairs:
        names = _sequence(resolved[name_field], "resolved." + name_field)
        if len(names) != len(integer_lists[id_field]) or any(
            not isinstance(name, str) or not name for name in names
        ):
            raise OneStepCounterfactualError("resolved %s/name population differs" % id_field)
    pairs_raw = _sequence(resolved["collision_enabled_pairs"], "collision pairs")
    pairs = []
    for raw in pairs_raw:
        pair = _sequence(raw, "collision pair")
        if len(pair) != 2:
            raise OneStepCounterfactualError("collision pair has wrong length")
        parsed = (_integer(pair[0], "collision robot geom", 0), _integer(pair[1], "collision obstacle geom", 0))
        if parsed[0] not in integer_lists["robot_geom_ids"] or parsed[1] not in integer_lists["obstacle_geom_ids"]:
            raise OneStepCounterfactualError("collision pair is outside resolved populations")
        pairs.append(parsed)
    if not pairs or len(pairs) != len(set(pairs)):
        raise OneStepCounterfactualError("collision pair population is empty or duplicated")
    target_pair = (int(target["robot_geom_id"]), int(target["obstacle_geom_id"]))
    if (
        target["robot_body_name"]
        not in protocol["case"]["required_nominal_contact_body_names"]
        or int(target["robot_body_id"]) not in integer_lists["link56_body_ids"]
        or int(target["robot_geom_id"]) not in integer_lists["link56_geom_ids"]
        or int(target["obstacle_body_id"]) not in integer_lists["obstacle_body_ids"]
        or int(target["obstacle_geom_id"]) not in integer_lists["obstacle_geom_ids"]
        or target_pair not in pairs
    ):
        raise OneStepCounterfactualError("target contact is outside link5/6 obstacle authority")
    return {"record": resolved, "ids": integer_lists, "pairs": pairs, "target_pair": target_pair}


def _validate_full_robot_sampling(
    value: Any, *, resolved: Mapping[str, Any]
) -> Dict[str, Any]:
    evidence = _exact_keys(
        value,
        (
            "sample_count",
            "sample_ledger",
            "sample_ledger_sha256",
            "geom_records",
            "epsilon_m",
            "maximum_surface_cover_radius_m",
            "coverage_semantics",
            "rigid_roundtrip",
        ),
        "full-robot measurement sampling",
    )
    ledger_raw = _sequence(evidence["sample_ledger"], "full-robot sample ledger")
    count = _integer(evidence["sample_count"], "full-robot sample count", 1)
    if len(ledger_raw) != count:
        raise OneStepCounterfactualError("full-robot sample ledger count differs")
    ledger = []
    for index, raw in enumerate(ledger_raw):
        sample = _exact_keys(
            raw,
            (
                "sample_id",
                "body_id",
                "body_name",
                "geom_id",
                "geom_name",
                "point_body_local_m",
                "source",
            ),
            "full-robot sample %d" % index,
        )
        if _integer(sample["sample_id"], "full-robot sample id", 0) != index:
            raise OneStepCounterfactualError("full-robot sample IDs are not contiguous")
        if int(sample["body_id"]) not in resolved["ids"]["robot_body_ids"] or int(
            sample["geom_id"]
        ) not in resolved["ids"]["robot_geom_ids"]:
            raise OneStepCounterfactualError("full-robot sample is outside robot authority")
        _vector(sample["point_body_local_m"], 3, "full-robot local point")
        ledger.append(dict(sample))
    if _sha(ledger) != _sha256(
        evidence["sample_ledger_sha256"], "full-robot sample ledger hash"
    ):
        raise OneStepCounterfactualError("full-robot sample ledger hash differs")
    try:
        from main.poisson_fullbody.surface_sampling import (
            validate_robot_sample_evidence,
        )

        validate_robot_sample_evidence(
            evidence,
            resolved_geom_ids=resolved["ids"]["robot_geom_ids"],
            resolved_geom_names=resolved["record"]["robot_geom_names"],
            resolved_body_ids=resolved["ids"]["robot_body_ids"],
            roundtrip_field="rigid_roundtrip",
        )
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise OneStepCounterfactualError(
            "full-robot sampling certificate is invalid"
        ) from error
    maximum = _number(
        evidence["maximum_surface_cover_radius_m"],
        "full-robot maximum coverage radius",
    )
    return {"record": evidence, "ledger": ledger, "count": count, "maximum": maximum}


def _validate_contact_record(
    value: Any, *, enclosing_boundary: int, label: str
) -> Mapping[str, Any]:
    record = _exact_keys(value, _CONTACT_RECORD_KEYS, label)
    phase = record["source_phase"]
    physical = _integer(record["physical_boundary"], label + ".physical_boundary", 0)
    if phase == "post_integration_recomputed":
        expected_boundary = enclosing_boundary
    elif phase == "live_solver_phase_preintegration_geometry":
        expected_boundary = enclosing_boundary - 1
    else:
        raise OneStepCounterfactualError("%s source phase is invalid" % label)
    if physical != expected_boundary:
        raise OneStepCounterfactualError("%s physical phase boundary differs" % label)
    for field in (
        "mujoco_contact_index",
        "mujoco_geom1_id",
        "mujoco_body1_id",
        "mujoco_geom2_id",
        "mujoco_body2_id",
    ):
        _integer(record[field], label + "." + field, 0)
    for field in (
        "mujoco_geom1_name",
        "mujoco_body1_name",
        "mujoco_geom2_name",
        "mujoco_body2_name",
    ):
        if not isinstance(record[field], str) or not record[field]:
            raise OneStepCounterfactualError("%s.%s is empty" % (label, field))
    distance = _number(record["contact_distance_m"], label + ".distance")
    physical_flag = _boolean(
        record["is_physical_nonpositive_distance_contact"], label + ".physical"
    )
    if physical_flag is not (distance <= 0.0):
        raise OneStepCounterfactualError("%s physical flag differs" % label)
    _boolean(record["solver_constraint_active"], label + ".solver active")
    _integer(record["efc_address"], label + ".efc_address")
    _number(record["contact_includemargin_m"], label + ".includemargin")
    _vector(record["position_world_m"], 3, label + ".position")
    return record


def _selected_contact_projection(
    records: Sequence[Mapping[str, Any]], resolved: Mapping[str, Any]
) -> Dict[str, List[Mapping[str, Any]]]:
    pair_set = set(resolved["pairs"])
    target_pair = resolved["target_pair"]
    selected = []
    target = []
    shifted = []
    for record in records:
        if record["is_physical_nonpositive_distance_contact"] is not True:
            continue
        geom1 = int(record["mujoco_geom1_id"])
        geom2 = int(record["mujoco_geom2_id"])
        pair = None
        if (geom1, geom2) in pair_set:
            pair = (geom1, geom2)
        elif (geom2, geom1) in pair_set:
            pair = (geom2, geom1)
        if pair is None:
            continue
        selected.append(record)
        if pair == target_pair:
            target.append(record)
        else:
            shifted.append(record)
    return {"selected": selected, "target": target, "shifted": shifted}


def _validate_measurement_row(
    value: Any,
    *,
    expected_boundary: int,
    command: Sequence[float],
    normalized_action: Sequence[float],
    gripper: float,
    samples: Mapping[str, Any],
    resolved: Mapping[str, Any],
    full_nv_count: int,
    physical_model: Mapping[str, Any],
    qp: Mapping[str, Any],
    elapsed_s: Optional[float],
    label: str,
) -> Dict[str, Any]:
    keys = list(_MEASUREMENT_KEYS)
    if elapsed_s is not None:
        keys.extend(("first_order_predicted_h_m2", "prediction_error_m2"))
    row = _exact_keys(value, keys, label)
    if _integer(row["physical_boundary"], label + ".physical_boundary", 0) != expected_boundary:
        raise OneStepCounterfactualError("%s boundary differs" % label)
    integration_state = _sequence(row["integration_state"], label + ".integration_state")
    if not integration_state or _float64_array_sha256(integration_state) != _sha256(
        row["integration_state_sha256"], label + ".integration_state_sha256"
    ):
        raise OneStepCounterfactualError("%s integration state hash differs" % label)
    qpos = [_number(item, label + ".qpos") for item in _sequence(row["qpos"], label + ".qpos")]
    qvel = _vector(row["qvel"], full_nv_count, label + ".qvel")
    nq = _integer(physical_model["nq"], label + ".compiled nq", 1)
    nv = _integer(physical_model["nv"], label + ".compiled nv", 1)
    na = _integer(physical_model["na"], label + ".compiled na", 0)
    common_size = 1 + nq + nv + na
    if (
        full_nv_count != nv
        or len(qpos) != nq
        or len(integration_state) != physical_model["mjstate_integration_size"]
        or len(integration_state) < common_size
        or [float(item) for item in integration_state[1 : 1 + nq]]
        != qpos
        or [
            float(item)
            for item in integration_state[
                1 + nq : 1 + nq + nv
            ]
        ]
        != qvel
    ):
        raise OneStepCounterfactualError(
            "%s mjSTATE_INTEGRATION qpos/qvel projection differs" % label
        )
    observed_command = _vector(row["commanded_arm_qdot"], 7, label + ".command")
    observed_action = _vector(row["normalized_joint_velocity_action_8d"], 8, label + ".action")
    if observed_command != list(command) or observed_action != list(normalized_action):
        raise OneStepCounterfactualError("%s did not hold the registered command" % label)
    for index in range(7):
        _close(observed_action[index], command[index] / 0.5, label + ".normalized action")
    if observed_action[7] != gripper:
        raise OneStepCounterfactualError("%s gripper differs" % label)
    measured = _vector(row["measured_arm_qdot"], 7, label + ".measured qdot")
    if measured != [qvel[index] for index in samples["arm_dof_indices"]]:
        raise OneStepCounterfactualError("%s measured qdot is not the qvel slice" % label)
    count = samples["sample_count"]
    h_raw = _sequence(row["ordered_sample_h_m2"], label + ".h")
    gradients_raw = _sequence(row["ordered_sample_grad_h_m"], label + ".gradients")
    validity = _sequence(
        row["ordered_field_query_validity_and_reason"], label + ".validity"
    )
    if len(h_raw) != count or len(gradients_raw) != count or len(validity) != count:
        raise OneStepCounterfactualError("%s field-query population differs" % label)
    h: List[Optional[float]] = []
    gradients: List[Optional[List[float]]] = []
    invalid = 0
    for index in range(count):
        validity_row = _exact_keys(
            validity[index], ("sample_index", "sample_id", "valid", "reason"), "%s validity %d" % (label, index)
        )
        if (
            _integer(validity_row["sample_index"], "query sample index", 0) != index
            or validity_row["sample_id"] != samples["samples"][index]["sample_id"]
        ):
            raise OneStepCounterfactualError("%s query/sample identity differs" % label)
        valid = _boolean(validity_row["valid"], "query valid flag")
        if valid:
            if validity_row["reason"] is not None or h_raw[index] is None or gradients_raw[index] is None:
                raise OneStepCounterfactualError("valid field query lacks h/gradient")
            h.append(_number(h_raw[index], label + ".h[%d]" % index))
            gradients.append(_vector(gradients_raw[index], 3, label + ".gradient"))
        else:
            invalid += 1
            if (
                validity_row["reason"] not in INVALID_QUERY_REASONS
                or h_raw[index] is not None
                or gradients_raw[index] is not None
            ):
                raise OneStepCounterfactualError("invalid field query evidence differs")
            h.append(None)
            gradients.append(None)
    if _integer(row["invalid_field_query_count"], label + ".invalid count", 0) != invalid:
        raise OneStepCounterfactualError("%s invalid query count differs" % label)
    d_opt = _vector(row["ordered_sample_D_opt_m"], count, label + ".D_opt")
    if any(value < 0.0 for value in d_opt):
        raise OneStepCounterfactualError("%s D_opt is negative" % label)
    _close(row["minimum_D_opt_m"], min(d_opt), label + ".minimum D_opt")
    raw_contacts = _sequence(
        row["all_mujoco_contact_pairs_with_distances"], label + ".contacts"
    )
    contacts = [
        _validate_contact_record(
            record, enclosing_boundary=expected_boundary, label="%s contact %d" % (label, index)
        )
        for index, record in enumerate(raw_contacts)
    ]
    expected_nonpositive = [
        record for record in contacts if record["is_physical_nonpositive_distance_contact"]
    ]
    _same(
        row["exact_ordered_nonpositive_physical_contact_subset"],
        expected_nonpositive,
        label + " nonpositive contact subset",
    )
    selected = _selected_contact_projection(contacts, resolved)
    _same(row["target_physical_contact_records"], selected["target"], label + " target contacts")
    _same(
        row["robot_to_selected_obstacle_physical_contact_records"],
        selected["selected"],
        label + " selected-obstacle contacts",
    )
    _same(row["shifted_robot_to_selected_obstacle_physical_contact_records"], selected["shifted"], label + " shifted contacts")
    flags = {
        "target_contact_present": bool(selected["target"]),
        "any_robot_to_selected_obstacle_contact_present": bool(selected["selected"]),
        "any_shifted_robot_to_selected_obstacle_contact_present": bool(selected["shifted"]),
    }
    for field, expected in flags.items():
        if row[field] is not expected:
            raise OneStepCounterfactualError("%s.%s differs" % (label, field))
    full_sample_distances = _vector(
        row["ordered_full_robot_sample_distance_m"],
        samples["full_sampling"]["count"],
        label + ".ordered full-robot sample distances",
    )
    if any(distance < 0.0 for distance in full_sample_distances):
        raise OneStepCounterfactualError(
            "%s full-robot sample distance is negative" % label
        )
    if (
        _integer(
            row["registered_full_robot_sample_count"],
            label + ".registered full-robot sample count",
            1,
        )
        != samples["full_sampling"]["count"]
        or _sha256(
            row["registered_full_robot_sample_ledger_sha256"],
            label + ".registered full-robot sample ledger hash",
        )
        != samples["full_sampling"]["record"]["sample_ledger_sha256"]
    ):
        raise OneStepCounterfactualError(
            "%s full-robot sample population differs from authority" % label
        )
    exact_sample_distance = _number(
        row["minimum_exact_full_robot_sample_to_current_obstacle_m"],
        label + ".exact sample distance",
    )
    _close(
        exact_sample_distance,
        min(full_sample_distances),
        label + ".minimum exact full-robot sample distance",
    )
    coverage = _number(
        row["certified_full_robot_coverage_radius_m"], label + ".coverage radius"
    )
    if coverage != samples["full_sampling"]["maximum"]:
        raise OneStepCounterfactualError(
            "%s coverage radius differs from full-robot authority" % label
        )
    lower_bound = exact_sample_distance - coverage
    selected_distances = [float(record["contact_distance_m"]) for record in selected["selected"]]
    interval_dsim = min([lower_bound, 0.0] + selected_distances) if selected_distances else lower_bound
    post_selected = [
        record for record in selected["selected"] if record["source_phase"] == "post_integration_recomputed"
    ]
    post_distances = [float(record["contact_distance_m"]) for record in post_selected]
    post_dsim = min([lower_bound, 0.0] + post_distances) if post_distances else lower_bound
    _close(row["minimum_D_sim_m"], interval_dsim, label + ".minimum D_sim")
    _close(row["interval_union_minimum_D_sim_m"], interval_dsim, label + ".interval Dsim")
    _close(row["post_state_minimum_D_sim_m"], post_dsim, label + ".post Dsim")
    drift_fields = (
        "selected_obstacle_translation_drift_m",
        "selected_obstacle_rotation_drift_rad",
        "selected_obstacle_surface_drift_m",
    )
    for field in drift_fields:
        if _number(row[field], label + "." + field) < 0.0:
            raise OneStepCounterfactualError("%s.%s is negative" % (label, field))
    velocity_records = _sequence(
        row["selected_obstacle_body_velocity_records"], label + ".body velocities"
    )
    if not velocity_records:
        raise OneStepCounterfactualError("%s has no obstacle velocity records" % label)
    linear_speeds = []
    angular_speeds = []
    body_ids = []
    for index, raw in enumerate(velocity_records):
        velocity = _exact_keys(
            raw,
            (
                "body_id",
                "body_name",
                "angular_velocity_world_rad_per_s",
                "linear_velocity_world_m_per_s",
                "angular_speed_rad_per_s",
                "linear_speed_m_per_s",
            ),
            "%s velocity %d" % (label, index),
        )
        body_id = _integer(velocity["body_id"], "obstacle velocity body id", 0)
        if body_id not in resolved["ids"]["obstacle_body_ids"] or body_id in body_ids:
            raise OneStepCounterfactualError("obstacle velocity body identity differs")
        body_ids.append(body_id)
        angular = _vector(velocity["angular_velocity_world_rad_per_s"], 3, "obstacle angular velocity")
        linear = _vector(velocity["linear_velocity_world_m_per_s"], 3, "obstacle linear velocity")
        angular_speed = _number(velocity["angular_speed_rad_per_s"], "obstacle angular speed")
        linear_speed = _number(velocity["linear_speed_m_per_s"], "obstacle linear speed")
        _close(angular_speed, _norm(angular), "obstacle angular speed")
        _close(linear_speed, _norm(linear), "obstacle linear speed")
        angular_speeds.append(angular_speed)
        linear_speeds.append(linear_speed)
    if body_ids != sorted(resolved["ids"]["obstacle_body_ids"]):
        raise OneStepCounterfactualError(
            "%s obstacle velocity ledger omits or reorders a body" % label
        )
    _close(
        row["selected_obstacle_max_body_linear_speed_m_s"],
        max(linear_speeds),
        label + ".max linear speed",
    )
    _close(
        row["selected_obstacle_max_body_angular_speed_rad_s"],
        max(angular_speeds),
        label + ".max angular speed",
    )
    if row["velocity_reference"] != "mj_objectVelocity_world_orientation_rot_then_lin":
        raise OneStepCounterfactualError("obstacle velocity semantics differ")
    if elapsed_s is not None:
        predicted = _vector(
            row["first_order_predicted_h_m2"], count, label + ".predicted h"
        )
        errors_raw = _sequence(row["prediction_error_m2"], label + ".prediction errors")
        if len(errors_raw) != count:
            raise OneStepCounterfactualError("prediction-error population differs")
        for index in range(count):
            derivative = _dot(qp["rows"][index], command)
            expected_predicted = qp["h"][index] + elapsed_s * derivative
            _close(predicted[index], expected_predicted, label + ".predicted h arithmetic")
            if h[index] is None:
                if errors_raw[index] is not None:
                    raise OneStepCounterfactualError("invalid query has prediction error")
            else:
                _close(
                    errors_raw[index],
                    h[index] - predicted[index],
                    label + ".prediction error arithmetic",
                )
    return {
        "record": row,
        "qpos": qpos,
        "qvel": qvel,
        "measured": measured,
        "h": h,
        "gradients": gradients,
        "invalid": invalid,
        "contacts": contacts,
        "selected": selected,
        "minimum_D_sim": interval_dsim,
    }


def _validate_arm_restore(
    value: Any,
    *,
    state_B: Mapping[str, Any],
    label: str,
    registered_admissibility: Mapping[str, Any],
    physical_model: Mapping[str, Any],
    controller_authority: Mapping[str, Any],
) -> Mapping[str, Any]:
    restore = _exact_keys(
        value,
        (
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
        ),
        label + ".restore",
    )
    if (
        restore["schema_version"] != "vlsa_poisson_controller_state_restore.v1"
        or restore["source_controller"] != "OSC_POSE"
        or restore["target_controller"] != "JOINT_VELOCITY"
        or restore["official_integration_state_available"] is not True
        or restore["official_integration_state_sha256"]
        != state_B["integration_state_sha256"]
        or restore["target_official_integration_state_sha256"]
        != state_B["integration_state_sha256"]
        or restore["settled_state_sha256"]
        != state_B["flattened_simulator_state_sha256"]
        or restore["target_state_sha256"]
        != state_B["flattened_simulator_state_sha256"]
        or restore["exact_flattened_state"] is not True
    ):
        raise OneStepCounterfactualError("%s restore state binding differs" % label)
    for field in ("model_topology_sha256", "physical_model_sha256", "compiled_mjb_sha256"):
        _sha256(restore[field], label + ".restore." + field)
    if (
        restore["physical_model_sha256"] != physical_model["sha256"]
        or restore["compiled_mjb_sha256"] != physical_model["compiled_mjb_sha256"]
    ):
        raise OneStepCounterfactualError(
            "%s restore physical model differs from provenance" % label
        )
    _same(
        restore["controller"],
        controller_authority,
        "%s restore.controller authority" % label,
    )
    qpos_error = _number(restore["maximum_arm_qpos_error_rad"], label + ".restore qpos error")
    qvel_error = _number(restore["maximum_arm_qvel_error_rad_s"], label + ".restore qvel error")
    qpos_tolerance = _number(
        restore["max_arm_qpos_error_tolerance_rad"], label + ".restore qpos tolerance"
    )
    qvel_tolerance = _number(
        restore["max_arm_qvel_error_tolerance_rad_s"], label + ".restore qvel tolerance"
    )
    if (
        qpos_tolerance
        != registered_admissibility["max_state_restore_qpos_error_rad"]
        or qvel_tolerance
        != registered_admissibility["max_state_restore_qvel_error_rad_s"]
    ):
        raise OneStepCounterfactualError(
            "%s restore tolerances differ from runtime authority" % label
        )
    if qpos_error < 0.0 or qvel_error < 0.0 or qpos_error > qpos_tolerance or qvel_error > qvel_tolerance:
        raise OneStepCounterfactualError("%s restore tolerance crossed" % label)
    if (
        restore["copied_timestep"] != state_B["wrapper_bookkeeping"]["timestep"]
        or restore["copied_cur_time_s"] != state_B["wrapper_bookkeeping"]["cur_time_s"]
        or restore["copied_done"] != state_B["wrapper_bookkeeping"]["done"]
    ):
        raise OneStepCounterfactualError("%s wrapper bookkeeping differs" % label)
    pid = _exact_keys(
        restore["pid_memory_reset"],
        (
            "goal_velocity_zero",
            "current_velocity_zero",
            "last_error_zero",
            "summed_error_zero",
            "derivative_buffer_size",
            "saturated",
        ),
        label + ".PID reset",
    )
    if (
        any(pid[field] is not True for field in ("goal_velocity_zero", "current_velocity_zero", "last_error_zero", "summed_error_zero"))
        or pid["derivative_buffer_size"] != 0
        or pid["saturated"] is not False
    ):
        raise OneStepCounterfactualError("%s PID memory was not reset" % label)
    software = _exact_keys(
        restore["controller_software_state"],
        ("schema_version", "fields", "sha256"),
        label + ".controller software state",
    )
    if (
        software["schema_version"]
        != "vlsa_poisson_joint_velocity_controller_software_state.v1"
    ):
        raise OneStepCounterfactualError(
            "%s controller software schema differs" % label
        )
    fields = _exact_keys(
        software["fields"],
        (
            "initial_joint",
            "goal_vel",
            "current_vel",
            "last_err",
            "summed_err",
            "last_joint_vel",
            "torques",
            "new_update",
            "saturated",
            "joint_pos",
            "joint_vel",
            "torque_compensation",
            "derivative_ring_buffer",
        ),
        label + ".controller software fields",
    )
    for field_name in fields:
        if field_name == "derivative_ring_buffer":
            derivative = _exact_keys(
                fields[field_name],
                ("available", "fields"),
                label + ".derivative ring buffer",
            )
            if derivative["available"] is not True or not isinstance(
                derivative["fields"], Mapping
            ):
                raise OneStepCounterfactualError(
                    "%s derivative ring buffer differs" % label
                )
        else:
            state_field = _exact_keys(
                fields[field_name],
                ("available", "value"),
                "%s.controller software %s" % (label, field_name),
            )
            if state_field["available"] is not True:
                raise OneStepCounterfactualError(
                    "%s controller software field %s is unavailable"
                    % (label, field_name)
                )
    software_hash = _sha256(software.get("sha256"), label + ".controller software hash")
    software_unhashed = {key: item for key, item in software.items() if key != "sha256"}
    if _sha(software_unhashed) != software_hash:
        raise OneStepCounterfactualError("%s controller software hash differs" % label)
    return restore


def _validate_arm(
    value: Any,
    *,
    arm_name: str,
    command: Sequence[float],
    source: Mapping[str, Any],
    samples: Mapping[str, Any],
    resolved: Mapping[str, Any],
    nominal: Mapping[str, Any],
    qp: Mapping[str, Any],
    protocol_values: Mapping[str, Any],
    registered_parameters: Mapping[str, Any],
    physical_model: Mapping[str, Any],
    controller_authority: Mapping[str, Any],
) -> Dict[str, Any]:
    arm = _exact_keys(
        value,
        (
            "arm_name",
            "controller",
            "filter_solve_count",
            "held_command_physics_substeps",
            "commanded_arm_qdot",
            "normalized_joint_velocity_action_8d",
            "source_gripper_command",
            "hidden_clipping_applied",
            "restore",
            "start_boundary",
            "physics_substep_ledger",
            "tracking",
            "motion",
            "maximum_selected_obstacle_drift",
            "static_field_admissible_for_complete_interval",
            "minimum_h_m2",
            "invalid_field_query_count",
            "minimum_D_sim_m",
            "any_target_contact",
            "any_robot_to_selected_obstacle_contact",
            "any_shifted_robot_to_selected_obstacle_contact",
        ),
        "arm %s" % arm_name,
    )
    if arm["arm_name"] != arm_name or arm["controller"] != "JOINT_VELOCITY_100Hz":
        raise OneStepCounterfactualError("arm identity/controller differs")
    expected_solves = 0 if arm_name == "nominal" else 1
    if _integer(arm["filter_solve_count"], "arm filter solves", 0) != expected_solves:
        raise OneStepCounterfactualError("arm filter solve count differs")
    horizon = protocol_values["horizon"]
    if _integer(arm["held_command_physics_substeps"], "held command count", 0) != horizon:
        raise OneStepCounterfactualError("arm does not expose all five held substeps")
    gripper = source["gripper"]["exact_gripper_value"]
    observed_command = _vector(arm["commanded_arm_qdot"], 7, "arm command")
    action = _vector(arm["normalized_joint_velocity_action_8d"], 8, "arm action")
    if observed_command != list(command) or arm["source_gripper_command"] != gripper:
        raise OneStepCounterfactualError("arm command/gripper differs")
    for index in range(7):
        _close(action[index], command[index] / 0.5, "arm normalized action")
    if action[7] != gripper or arm["hidden_clipping_applied"] is not False:
        raise OneStepCounterfactualError("arm action encoding or clipping differs")
    state_B = source["prefix"]["state_at_B"]
    restore = _validate_arm_restore(
        arm["restore"],
        state_B=state_B,
        label=arm_name,
        registered_admissibility=registered_parameters["admissibility"],
        physical_model=physical_model,
        controller_authority=controller_authority,
    )
    start = _validate_measurement_row(
        arm["start_boundary"],
        expected_boundary=source["boundary"],
        command=command,
        normalized_action=action,
        gripper=gripper,
        samples=samples,
        resolved=resolved,
        full_nv_count=nominal["record"]["full_nv_count"],
        physical_model=physical_model,
        qp=qp,
        elapsed_s=None,
        label=arm_name + ".start",
    )
    if (
        start["record"]["integration_state_sha256"] != state_B["integration_state_sha256"]
        or start["qpos"] != state_B["qpos"]
        or start["qvel"] != state_B["qvel"]
    ):
        raise OneStepCounterfactualError("arm start is not exact boundary B")
    ledger_raw = _sequence(arm["physics_substep_ledger"], arm_name + ".ledger")
    if len(ledger_raw) != horizon:
        raise OneStepCounterfactualError("arm physics ledger is incomplete")
    ledger = []
    for index, raw in enumerate(ledger_raw):
        ledger.append(
            _validate_measurement_row(
                raw,
                expected_boundary=source["boundary"] + index + 1,
                command=command,
                normalized_action=action,
                gripper=gripper,
                samples=samples,
                resolved=resolved,
                full_nv_count=nominal["record"]["full_nv_count"],
                physical_model=physical_model,
                qp=qp,
                elapsed_s=(index + 1) * protocol_values["timestep"],
                label="%s.ledger[%d]" % (arm_name, index),
            )
        )
    tracking = _exact_keys(
        arm["tracking"],
        (
            "error_ledger_rad_s",
            "linf_rad_s",
            "rmse_rad_s",
            "registered_linf_max_rad_s",
            "registered_rmse_max_rad_s",
            "within_registered_thresholds",
        ),
        arm_name + ".tracking",
    )
    error_rows = _sequence(tracking["error_ledger_rad_s"], arm_name + ".tracking errors")
    if len(error_rows) != horizon:
        raise OneStepCounterfactualError("tracking error ledger is incomplete")
    errors = []
    for index, raw in enumerate(error_rows):
        row = _vector(raw, 7, "tracking error row")
        for joint in range(7):
            _close(row[joint], ledger[index]["measured"][joint] - command[joint], "tracking error")
        errors.extend(row)
    linf = max(abs(value) for value in errors)
    rmse = math.sqrt(sum(value * value for value in errors) / len(errors))
    _close(tracking["linf_rad_s"], linf, "tracking Linf")
    _close(tracking["rmse_rad_s"], rmse, "tracking RMSE")
    acceptance = protocol_values["acceptance"]
    linf_threshold = _number(
        acceptance["joint_velocity_tracking_linf_max_rad_s"], "tracking Linf threshold"
    )
    rmse_threshold = _number(
        acceptance["joint_velocity_tracking_rmse_max_rad_s"], "tracking RMSE threshold"
    )
    if (
        tracking["registered_linf_max_rad_s"] != linf_threshold
        or tracking["registered_rmse_max_rad_s"] != rmse_threshold
        or linf_threshold
        != registered_parameters["admissibility"][
            "max_joint_velocity_tracking_linf_rad_s"
        ]
        or rmse_threshold
        != registered_parameters["admissibility"][
            "max_joint_velocity_tracking_rmse_rad_s"
        ]
        or tracking["within_registered_thresholds"]
        is not bool(linf <= linf_threshold and rmse <= rmse_threshold)
    ):
        raise OneStepCounterfactualError("tracking threshold classification differs")
    motion = _exact_keys(
        arm["motion"],
        (
            "measured_arm_tangent_displacement",
            "measured_arm_motion_l2_rad",
            "integrated_measured_joint_speed_l2_rad",
            "command_norm_rad_s",
            "command_integral_over_horizon_rad",
            "measured_motion_to_command_integral_ratio",
            "nonzero_command",
            "nonzero_measured_joint_motion",
        ),
        arm_name + ".motion",
    )
    tangent = _vector(motion["measured_arm_tangent_displacement"], 7, "measured tangent motion")
    arm_qpos_indices = nominal["arm_qpos_indices"]
    expected_tangent = [
        ledger[-1]["qpos"][index] - start["qpos"][index]
        for index in arm_qpos_indices
    ]
    for observed, expected in zip(tangent, expected_tangent):
        _close(observed, expected, "measured scalar-hinge endpoint displacement")
    motion_norm = _norm(tangent)
    integrated = sum(_norm(row["measured"]) for row in ledger) * protocol_values["timestep"]
    command_norm = _norm(command)
    command_integral = command_norm * protocol_values["horizon_s"]
    motion_ratio = None if command_integral == 0.0 else motion_norm / command_integral
    _close(motion["measured_arm_motion_l2_rad"], motion_norm, "measured motion norm")
    _close(motion["integrated_measured_joint_speed_l2_rad"], integrated, "integrated measured speed")
    _close(motion["command_norm_rad_s"], command_norm, "command norm")
    _close(
        motion["command_integral_over_horizon_rad"],
        command_integral,
        "command integral",
    )
    if motion_ratio is None:
        if motion["measured_motion_to_command_integral_ratio"] is not None:
            raise OneStepCounterfactualError(
                "zero-command motion ratio must be null"
            )
    else:
        _close(
            motion["measured_motion_to_command_integral_ratio"],
            motion_ratio,
            "measured-motion/command-integral ratio",
        )
    if motion["nonzero_command"] is not (_norm(command) > 0.0) or motion[
        "nonzero_measured_joint_motion"
    ] is not (motion_norm > 0.0):
        raise OneStepCounterfactualError("motion positivity flags differ")
    all_rows = [start] + ledger
    drift_fields = (
        "selected_obstacle_translation_drift_m",
        "selected_obstacle_rotation_drift_rad",
        "selected_obstacle_surface_drift_m",
        "selected_obstacle_max_body_linear_speed_m_s",
        "selected_obstacle_max_body_angular_speed_rad_s",
    )
    maximum = {field: max(float(row["record"][field]) for row in all_rows) for field in drift_fields}
    _same(arm["maximum_selected_obstacle_drift"], maximum, arm_name + " maximum drift")
    thresholds = source["source"]["boundary_B_admissibility"][
        "selected_obstacle_static_thresholds"
    ]
    threshold_by_field = {
        "selected_obstacle_translation_drift_m": thresholds["translation_m"],
        "selected_obstacle_rotation_drift_rad": thresholds["rotation_rad"],
        "selected_obstacle_surface_drift_m": thresholds["surface_m"],
        "selected_obstacle_max_body_linear_speed_m_s": thresholds["linear_speed_m_s"],
        "selected_obstacle_max_body_angular_speed_rad_s": thresholds["angular_speed_rad_s"],
    }
    static = all(maximum[field] <= threshold_by_field[field] for field in drift_fields)
    if arm["static_field_admissible_for_complete_interval"] is not static:
        raise OneStepCounterfactualError("arm static-field classification differs")
    valid_h = [value for row in ledger for value in row["h"] if value is not None]
    expected_min_h = min(valid_h) if valid_h else None
    if expected_min_h is None:
        if arm["minimum_h_m2"] is not None:
            raise OneStepCounterfactualError("arm minimum h differs")
    else:
        _close(arm["minimum_h_m2"], expected_min_h, "arm minimum h")
    invalid_count = sum(row["invalid"] for row in ledger)
    if arm["invalid_field_query_count"] != invalid_count:
        raise OneStepCounterfactualError("arm invalid-query aggregate differs")
    minimum_dsim = min(row["minimum_D_sim"] for row in ledger)
    _close(arm["minimum_D_sim_m"], minimum_dsim, "arm minimum Dsim")
    expected_target = any(row["selected"]["target"] for row in ledger)
    expected_selected = any(row["selected"]["selected"] for row in ledger)
    expected_shifted = any(row["selected"]["shifted"] for row in ledger)
    for field, expected in (
        ("any_target_contact", expected_target),
        ("any_robot_to_selected_obstacle_contact", expected_selected),
        ("any_shifted_robot_to_selected_obstacle_contact", expected_shifted),
    ):
        if arm[field] is not expected:
            raise OneStepCounterfactualError("arm %s aggregate differs" % field)
    return {
        "record": arm,
        "restore": restore,
        "start": start,
        "ledger": ledger,
        "tracking_valid": tracking["within_registered_thresholds"],
        "static": static,
        "motion_norm": motion_norm,
        "command_norm": command_norm,
        "command_integral": command_integral,
        "motion_ratio": motion_ratio,
        "invalid_count": invalid_count,
        "minimum_h": expected_min_h,
        "minimum_D_sim": minimum_dsim,
        "any_target": expected_target,
        "any_selected": expected_selected,
        "any_shifted": expected_shifted,
    }


def _reconstruct_static_windows(
    *,
    protocol: Mapping[str, Any],
    source: Mapping[str, Any],
    nominal_arm: Mapping[str, Any],
    psf_arm: Mapping[str, Any],
    C_target_nom: Optional[int],
    C_any_nom: Optional[int],
) -> Dict[str, Any]:
    scope_protocol = _exact_keys(
        protocol["acceptance"].get("static_field_admissibility_scopes"),
        (
            "nominal_with_selected_obstacle_contact",
            "nominal_without_selected_obstacle_contact",
            "poisson_filtered_velocity",
        ),
        "protocol static-field admissibility scopes",
    )
    if any(not isinstance(item, str) or not item for item in scope_protocol.values()):
        raise OneStepCounterfactualError("static-field scope label is empty")
    raw_thresholds = source["source"]["boundary_B_admissibility"][
        "selected_obstacle_static_thresholds"
    ]
    thresholds = {
        key: _number(raw_thresholds[key], "static threshold " + key)
        for key in (
            "translation_m",
            "rotation_rad",
            "surface_m",
            "linear_speed_m_s",
            "angular_speed_rad_s",
        )
    }
    fields = {
        "translation_m": "selected_obstacle_translation_drift_m",
        "rotation_rad": "selected_obstacle_rotation_drift_rad",
        "surface_m": "selected_obstacle_surface_drift_m",
        "linear_speed_m_s": "selected_obstacle_max_body_linear_speed_m_s",
        "angular_speed_rad_s": "selected_obstacle_max_body_angular_speed_rad_s",
    }

    def window(
        arm: Mapping[str, Any], *, scope: str, cutoff: Optional[int]
    ) -> Dict[str, Any]:
        rows = [arm["start"]] + list(arm["ledger"])
        included = [
            row
            for row in rows
            if cutoff is None or row["record"]["physical_boundary"] < cutoff
        ]
        maxima: Dict[str, Optional[float]] = {}
        for key, field in fields.items():
            values = [float(row["record"][field]) for row in included]
            maxima[key] = None if not values else max(values)
        return {
            "scope": scope,
            "cutoff_physical_boundary_exclusive": cutoff,
            "included_physical_boundaries": [
                int(row["record"]["physical_boundary"]) for row in included
            ],
            "observed_maxima": maxima,
            "thresholds": dict(thresholds),
            "admissible": bool(
                included
                and all(
                    maxima[key] is not None
                    and float(maxima[key]) <= thresholds[key]
                    for key in thresholds
                )
            ),
        }

    nominal_scope_key = (
        "nominal_with_selected_obstacle_contact"
        if C_any_nom is not None
        else "nominal_without_selected_obstacle_contact"
    )
    return {
        "scope_protocol": dict(scope_protocol),
        "nominal": window(
            nominal_arm,
            scope=str(scope_protocol[nominal_scope_key]),
            cutoff=C_any_nom,
        ),
        "psf": window(
            psf_arm,
            scope=str(scope_protocol["poisson_filtered_velocity"]),
            cutoff=None,
        ),
    }


def _reconstruct_tracking_windows(
    *,
    protocol: Mapping[str, Any],
    nominal_arm: Mapping[str, Any],
    psf_arm: Mapping[str, Any],
    C_any_nom: Optional[int],
) -> Dict[str, Any]:
    scope_protocol = _exact_keys(
        protocol["acceptance"].get("tracking_admissibility_scopes"),
        (
            "nominal_with_selected_obstacle_contact",
            "nominal_without_selected_obstacle_contact",
            "poisson_filtered_velocity",
        ),
        "protocol tracking admissibility scopes",
    )
    if any(not isinstance(item, str) or not item for item in scope_protocol.values()):
        raise OneStepCounterfactualError("tracking scope label is empty")
    thresholds = {
        "linf_rad_s": float(
            protocol["acceptance"]["joint_velocity_tracking_linf_max_rad_s"]
        ),
        "rmse_rad_s": float(
            protocol["acceptance"]["joint_velocity_tracking_rmse_max_rad_s"]
        ),
    }

    def window(
        arm: Mapping[str, Any], *, scope: str, cutoff: Optional[int]
    ) -> Dict[str, Any]:
        included = [
            row
            for row in arm["ledger"]
            if cutoff is None or row["record"]["physical_boundary"] < cutoff
        ]
        errors = [
            measured - commanded
            for row in included
            for measured, commanded in zip(
                row["measured"], arm["record"]["commanded_arm_qdot"]
            )
        ]
        linf = None if not errors else max(abs(value) for value in errors)
        rmse = (
            None
            if not errors
            else math.sqrt(sum(value * value for value in errors) / len(errors))
        )
        return {
            "scope": scope,
            "cutoff_physical_boundary_exclusive": cutoff,
            "included_physical_boundaries": [
                int(row["record"]["physical_boundary"]) for row in included
            ],
            "linf_rad_s": linf,
            "rmse_rad_s": rmse,
            "thresholds": dict(thresholds),
            "admissible": bool(
                errors
                and linf is not None
                and rmse is not None
                and linf <= thresholds["linf_rad_s"]
                and rmse <= thresholds["rmse_rad_s"]
            ),
        }

    nominal_scope_key = (
        "nominal_with_selected_obstacle_contact"
        if C_any_nom is not None
        else "nominal_without_selected_obstacle_contact"
    )
    return {
        "scope_protocol": dict(scope_protocol),
        "nominal": window(
            nominal_arm,
            scope=str(scope_protocol[nominal_scope_key]),
            cutoff=C_any_nom,
        ),
        "psf": window(
            psf_arm,
            scope=str(scope_protocol["poisson_filtered_velocity"]),
            cutoff=None,
        ),
    }


def _validate_diagnostics(
    value: Any,
    *,
    source: Mapping[str, Any],
    qp: Mapping[str, Any],
    nominal_arm: Mapping[str, Any],
    psf_arm: Mapping[str, Any],
    protocol_values: Mapping[str, Any],
) -> Dict[str, Any]:
    diagnostics = _exact_keys(
        value,
        (
            "trend_sample_index",
            "trend_sample_id",
            "h_at_B_m2",
            "first_order_predicted_h_after_horizon_m2",
            "actual_h_after_each_physics_substep_m2",
            "psf_h_after_horizon_m2",
            "psf_minus_nominal_h_after_horizon_m2",
            "psf_warning_sample_h_strictly_greater_than_nominal_at_horizon",
            "prediction_error_m2_without_acceptance_threshold",
            "prediction_error_policy",
            "nominal_D_sim_at_B_m",
            "nominal_minimum_D_sim_over_horizon_m",
            "trend_only_conditions_all_true_without_target_contact",
            "historical_identification_contact_boundary_C",
            "historical_contact_boundary_within_horizon",
            "counterfactual_nominal_first_target_contact_boundary_C_nom",
            "counterfactual_nominal_first_any_robot_selected_obstacle_contact_boundary_C_any_nom",
            "counterfactual_nominal_target_contact_within_horizon",
            "nominal_clearance_at_C_nom_m",
            "psf_clearance_at_C_nom_m",
            "static_field_admissibility_windows",
            "tracking_admissibility_windows",
            "local_motion_retention",
        ),
        "diagnostics",
    )
    trend_index = qp["trend_index"]
    if (
        diagnostics["trend_sample_index"] != trend_index
        or diagnostics["trend_sample_id"]
        != qp["record"]["trend_sample_id"]
        or diagnostics["prediction_error_policy"]
        != "diagnostic_only_no_post_hoc_tolerance"
    ):
        raise OneStepCounterfactualError("trend diagnostic identity differs")
    h_B = qp["h"][trend_index]
    _close(diagnostics["h_at_B_m2"], h_B, "diagnostic h at B")
    nominal_rows = nominal_arm["ledger"]
    psf_rows = psf_arm["ledger"]
    predicted_end = nominal_rows[-1]["record"]["first_order_predicted_h_m2"][trend_index]
    _close(
        diagnostics["first_order_predicted_h_after_horizon_m2"],
        predicted_end,
        "diagnostic predicted h",
    )
    actual_h = [row["h"][trend_index] for row in nominal_rows]
    psf_h_after = psf_rows[-1]["h"][trend_index]
    prediction_error = [
        row["record"]["prediction_error_m2"][trend_index] for row in nominal_rows
    ]
    _same(
        diagnostics["actual_h_after_each_physics_substep_m2"],
        actual_h,
        "diagnostic actual h ledger",
    )
    if psf_h_after is None:
        if diagnostics["psf_h_after_horizon_m2"] is not None:
            raise OneStepCounterfactualError("invalid PSF warning query must use null h")
    else:
        _close(
            diagnostics["psf_h_after_horizon_m2"],
            psf_h_after,
            "diagnostic PSF warning-sample h",
        )
    nominal_h_after = actual_h[-1]
    h_delta = (
        None
        if nominal_h_after is None or psf_h_after is None
        else psf_h_after - nominal_h_after
    )
    if h_delta is None:
        if diagnostics["psf_minus_nominal_h_after_horizon_m2"] is not None:
            raise OneStepCounterfactualError(
                "invalid paired warning query must use null h delta"
            )
    else:
        _close(
            diagnostics["psf_minus_nominal_h_after_horizon_m2"],
            h_delta,
            "diagnostic paired warning-sample h delta",
        )
    h_improved = h_delta is not None and h_delta > 0.0
    if (
        diagnostics[
            "psf_warning_sample_h_strictly_greater_than_nominal_at_horizon"
        ]
        is not h_improved
    ):
        raise OneStepCounterfactualError(
            "warning-sample improvement classification differs"
        )
    _same(
        diagnostics["prediction_error_m2_without_acceptance_threshold"],
        prediction_error,
        "diagnostic prediction-error ledger",
    )
    nominal_start_dsim = nominal_arm["start"]["minimum_D_sim"]
    _close(diagnostics["nominal_D_sim_at_B_m"], nominal_start_dsim, "nominal Dsim at B")
    _close(
        diagnostics["nominal_minimum_D_sim_over_horizon_m"],
        nominal_arm["minimum_D_sim"],
        "nominal horizon Dsim",
    )
    historical_C = source["contact_boundary"]
    if diagnostics["historical_identification_contact_boundary_C"] != historical_C:
        raise OneStepCounterfactualError("historical contact boundary diagnostic differs")
    historical_within = bool(
        source["boundary"] < historical_C <= source["boundary"] + 5
    )
    if diagnostics["historical_contact_boundary_within_horizon"] is not historical_within:
        raise OneStepCounterfactualError("historical contact horizon flag differs")
    nominal_target_records = [
        record
        for row in nominal_rows
        for record in row["selected"]["target"]
    ]
    C_nom = (
        None
        if not nominal_target_records
        else min(int(record["physical_boundary"]) for record in nominal_target_records)
    )
    nominal_selected_records = [
        record
        for row in nominal_rows
        for record in row["selected"]["selected"]
    ]
    C_any_nom = (
        None
        if not nominal_selected_records
        else min(
            int(record["physical_boundary"])
            for record in nominal_selected_records
        )
    )
    if diagnostics["counterfactual_nominal_first_target_contact_boundary_C_nom"] != C_nom:
        raise OneStepCounterfactualError("phase-correct counterfactual C_nom differs")
    if (
        diagnostics[
            "counterfactual_nominal_first_any_robot_selected_obstacle_contact_boundary_C_any_nom"
        ]
        != C_any_nom
    ):
        raise OneStepCounterfactualError(
            "phase-correct counterfactual C_any_nom differs"
        )
    within = bool(C_nom is not None and source["boundary"] < C_nom <= source["boundary"] + 5)
    if diagnostics["counterfactual_nominal_target_contact_within_horizon"] is not within:
        raise OneStepCounterfactualError("counterfactual nominal contact horizon differs")
    nominal_clearance = None
    psf_clearance = None
    if C_nom is not None:
        nominal_state = next(
            (row for row in nominal_rows if row["record"]["physical_boundary"] == C_nom),
            None,
        )
        psf_state = next(
            (row for row in psf_rows if row["record"]["physical_boundary"] == C_nom),
            None,
        )
        if nominal_state is None or psf_state is None:
            raise OneStepCounterfactualError("paired rows do not expose C_nom")
        nominal_records_at_C = [
            record
            for row in nominal_rows
            for record in row["selected"]["selected"]
            if record["physical_boundary"] == C_nom
        ]
        psf_records_at_C = [
            record
            for row in psf_rows
            for record in row["selected"]["selected"]
            if record["physical_boundary"] == C_nom
        ]
        nominal_clearance = min(
            [nominal_state["record"]["post_state_minimum_D_sim_m"]]
            + [record["contact_distance_m"] for record in nominal_records_at_C]
            + ([0.0] if nominal_records_at_C else [])
        )
        psf_clearance = min(
            [psf_state["record"]["post_state_minimum_D_sim_m"]]
            + [record["contact_distance_m"] for record in psf_records_at_C]
            + ([0.0] if psf_records_at_C else [])
        )
    for field, expected in (
        ("nominal_clearance_at_C_nom_m", nominal_clearance),
        ("psf_clearance_at_C_nom_m", psf_clearance),
    ):
        if expected is None:
            if diagnostics[field] is not None:
                raise OneStepCounterfactualError("%s must be null without C_nom" % field)
        else:
            _close(diagnostics[field], expected, field)
    predicted_decreased = predicted_end < h_B
    actual_decreased = actual_h[-1] is not None and actual_h[-1] < h_B
    dsim_decreased = nominal_arm["minimum_D_sim"] < nominal_start_dsim
    trend_all = bool(
        predicted_decreased
        and actual_decreased
        and dsim_decreased
        and not nominal_arm["any_target"]
        and h_improved
    )
    if diagnostics["trend_only_conditions_all_true_without_target_contact"] is not trend_all:
        raise OneStepCounterfactualError("trend-only diagnostic flag differs")
    static_windows = _reconstruct_static_windows(
        protocol=protocol_values["protocol"],
        source=source,
        nominal_arm=nominal_arm,
        psf_arm=psf_arm,
        C_target_nom=C_nom,
        C_any_nom=C_any_nom,
    )
    _same(
        diagnostics["static_field_admissibility_windows"],
        static_windows,
        "static-field admissibility windows",
    )
    tracking_windows = _reconstruct_tracking_windows(
        protocol=protocol_values["protocol"],
        nominal_arm=nominal_arm,
        psf_arm=psf_arm,
        C_any_nom=C_any_nom,
    )
    _same(
        diagnostics["tracking_admissibility_windows"],
        tracking_windows,
        "tracking admissibility windows",
    )
    acceptance = protocol_values["acceptance"]
    local_motion = _exact_keys(
        diagnostics["local_motion_retention"],
        (
            "filter_correction_norm_rad_s",
            "nominal_command_norm_rad_s",
            "safe_command_norm_rad_s",
            "safe_to_nominal_command_norm_ratio",
            "safe_measured_joint_motion_rad",
            "safe_command_integral_over_horizon_rad",
            "safe_measured_motion_to_command_integral_ratio",
            "thresholds",
            "interpretation",
        ),
        "diagnostics.local_motion_retention",
    )
    nominal_norm = nominal_arm["command_norm"]
    safe_norm = psf_arm["command_norm"]
    safe_to_nominal = None if nominal_norm == 0.0 else safe_norm / nominal_norm
    expected_motion_values = {
        "filter_correction_norm_rad_s": qp["correction_norm"],
        "nominal_command_norm_rad_s": nominal_norm,
        "safe_command_norm_rad_s": safe_norm,
        "safe_to_nominal_command_norm_ratio": safe_to_nominal,
        "safe_measured_joint_motion_rad": psf_arm["motion_norm"],
        "safe_command_integral_over_horizon_rad": psf_arm["command_integral"],
        "safe_measured_motion_to_command_integral_ratio": psf_arm["motion_ratio"],
    }
    for field, expected in expected_motion_values.items():
        if expected is None:
            if local_motion[field] is not None:
                raise OneStepCounterfactualError(
                    "local-motion %s must be null" % field
                )
        else:
            _close(local_motion[field], expected, "local-motion " + field)
    threshold_fields = (
        "minimum_filter_correction_norm_rad_s",
        "minimum_safe_command_norm_rad_s",
        "minimum_safe_to_nominal_command_norm_ratio",
        "minimum_safe_measured_joint_motion_rad",
        "minimum_safe_measured_motion_to_command_integral_ratio",
    )
    thresholds = _exact_keys(
        local_motion["thresholds"], threshold_fields, "local-motion thresholds"
    )
    _same(
        thresholds,
        {field: acceptance[field] for field in threshold_fields},
        "local-motion registered thresholds",
    )
    if local_motion["interpretation"] != acceptance["local_motion_interpretation"]:
        raise OneStepCounterfactualError("local-motion interpretation differs")
    return {
        "record": diagnostics,
        "C_nom": C_nom,
        "C_any_nom": C_any_nom,
        "nominal_clearance": nominal_clearance,
        "psf_clearance": psf_clearance,
        "predicted_decreased": predicted_decreased,
        "actual_decreased": actual_decreased,
        "target_valid": actual_h[-1] is not None,
        "psf_target_valid": psf_h_after is not None,
        "psf_h_after": psf_h_after,
        "h_delta": h_delta,
        "h_improved": h_improved,
        "dsim_decreased": dsim_decreased,
        "trend_all": trend_all,
        "local_motion": {
            **expected_motion_values,
            "thresholds": dict(thresholds),
        },
        "static_windows": static_windows,
        "tracking_windows": tracking_windows,
    }


def _reconstruct_classification(
    *,
    protocol: Mapping[str, Any],
    source: Mapping[str, Any],
    qp: Mapping[str, Any],
    nominal_arm: Mapping[str, Any],
    psf_arm: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    exact_paired_start: bool,
    nominal_within_bounds: bool,
    no_hidden_clipping: bool,
) -> Dict[str, Any]:
    acceptance = protocol["acceptance"]
    local_conditions = {
        "nominal_velocity_within_registered_controller_envelope": bool(
            nominal_within_bounds
        ),
        "no_hidden_clipping": bool(no_hidden_clipping),
        "boundary_B_preflight_admissible": bool(
            source["source"]["boundary_B_admissibility"]["passed"] is True
        ),
        "complete_exposure": bool(
            len(nominal_arm["ledger"]) == 5 and len(psf_arm["ledger"]) == 5
        ),
        "exact_paired_start": bool(exact_paired_start),
        "qp_valid": bool(
            psf_arm["record"]["filter_solve_count"] == 1
            and qp["record"]["one_CBF_row_per_exact_bound_sample"]["no_slack"]
            is True
            and qp["record"]["QP_status"] in ("solved", "solved inaccurate")
            and min(qp["safe_residuals"])
            >= float(acceptance["qp_safe_residual_minimum"])
        ),
        "both_tracking_valid": bool(
            diagnostics["tracking_windows"]["nominal"]["admissible"]
            and diagnostics["tracking_windows"]["psf"]["admissible"]
        ),
        "static_field_admissible": bool(
            diagnostics["static_windows"]["nominal"]["admissible"]
            and diagnostics["static_windows"]["psf"]["admissible"]
        ),
        "filter_correction_above_registered_minimum": bool(
            qp["correction_norm"]
            >= float(acceptance["minimum_filter_correction_norm_rad_s"])
        ),
        "safe_command_above_registered_minimum": bool(
            psf_arm["command_norm"]
            >= float(acceptance["minimum_safe_command_norm_rad_s"])
        ),
        "safe_command_retains_registered_nominal_norm_fraction": bool(
            diagnostics["local_motion"]["safe_to_nominal_command_norm_ratio"]
            is not None
            and diagnostics["local_motion"][
                "safe_to_nominal_command_norm_ratio"
            ]
            >= float(
                acceptance["minimum_safe_to_nominal_command_norm_ratio"]
            )
        ),
        "safe_measured_motion_above_registered_minimum": bool(
            psf_arm["motion_norm"]
            >= float(acceptance["minimum_safe_measured_joint_motion_rad"])
        ),
        "safe_measured_motion_retains_registered_command_integral_fraction": bool(
            psf_arm["motion_ratio"] is not None
            and psf_arm["motion_ratio"]
            >= float(
                acceptance[
                    "minimum_safe_measured_motion_to_command_integral_ratio"
                ]
            )
        ),
        "safe_all_queries_valid": bool(
            psf_arm["invalid_count"]
            <= int(acceptance["maximum_safe_arm_invalid_field_queries"])
        ),
        "safe_minimum_h_nonnegative": bool(
            psf_arm["minimum_h"] is not None and psf_arm["minimum_h"] >= 0.0
        ),
        "safe_minimum_D_sim_strictly_positive": bool(
            all(row["minimum_D_sim"] > 0.0 for row in psf_arm["ledger"])
        ),
        "safe_all_robot_selected_obstacle_contact_absent": not psf_arm["any_selected"],
    }
    contact_conditions = {
        "nominal_target_contact_within_horizon": bool(
            nominal_arm["any_target"]
            and diagnostics["C_nom"] is not None
            and source["boundary"] < diagnostics["C_nom"] <= source["boundary"] + 5
        ),
        "nominal_first_selected_obstacle_contact_matches_target_boundary": bool(
            diagnostics["C_any_nom"] is not None
            and diagnostics["C_nom"] is not None
            and diagnostics["C_any_nom"] == diagnostics["C_nom"]
        ),
        "safe_clearance_greater_at_nominal_target_contact_boundary": bool(
            diagnostics["C_nom"] is not None
            and diagnostics["nominal_clearance"] is not None
            and diagnostics["psf_clearance"] is not None
            and diagnostics["psf_clearance"] > diagnostics["nominal_clearance"]
        ),
    }
    trend_conditions = {
        "nominal_target_contact_absent": not nominal_arm["any_target"],
        "warning_sample_valid_in_both_arms_at_horizon": bool(
            diagnostics["target_valid"] and diagnostics["psf_target_valid"]
        ),
        "warning_sample_first_order_predicted_h_decreased": diagnostics[
            "predicted_decreased"
        ],
        "nominal_warning_sample_actual_h_decreased": diagnostics[
            "actual_decreased"
        ],
        "nominal_D_sim_decreased": diagnostics["dsim_decreased"],
        "psf_warning_sample_h_strictly_greater_than_nominal_at_horizon": diagnostics[
            "h_improved"
        ],
    }
    if nominal_arm["any_target"]:
        reference_kind = "target_contact_within_horizon"
        reference_conditions = contact_conditions
    else:
        reference_kind = "unsafe_trend_without_target_contact"
        reference_conditions = trend_conditions
    reference_passed = all(reference_conditions.values())
    local_passed = all(local_conditions.values())
    stage_passed = bool(local_passed and reference_passed)
    contact_prevention_observed = bool(
        stage_passed and reference_kind == "target_contact_within_horizon"
    )
    if stage_passed and reference_kind == "target_contact_within_horizon":
        label = acceptance["strong_result_label"]
    elif stage_passed:
        label = acceptance["trend_only_label"]
    else:
        label = acceptance["other_outcome_label"]
    typed_reasons = sorted(
        {
            key for key, passed in local_conditions.items() if passed is not True
        }
        | {
            key
            for key, passed in reference_conditions.items()
            if passed is not True
        }
    )
    return {
        "label": label,
        "stage_13_passed": stage_passed,
        "contact_prevention_observed": contact_prevention_observed,
        "claim_scope": (
            "one_100hz_interval_local_geometry_field_jacobian_qp_and_tracking_feasibility_not_task_success_or_full_episode_safety"
        ),
        "local_feasibility_conditions": local_conditions,
        "nominal_reference": {
            "kind": reference_kind,
            "conditions": reference_conditions,
            "passed": reference_passed,
        },
        "typed_reasons": typed_reasons,
    }


def _validate_counts(
    value: Any,
    *,
    source: Mapping[str, Any],
    samples: Mapping[str, Any],
    nominal_arm: Mapping[str, Any],
    psf_arm: Mapping[str, Any],
) -> Mapping[str, Any]:
    counts = _exact_keys(
        value,
        (
            "bound_sample_count",
            "source_prefix_callback_count",
            "nominal_physics_substep_count",
            "psf_physics_substep_count",
            "qp_solve_count",
            "nominal_invalid_field_query_count",
            "psf_invalid_field_query_count",
            "nominal_target_contact_record_count",
            "psf_robot_selected_obstacle_contact_record_count",
        ),
        "counts",
    )
    expected = {
        "bound_sample_count": samples["sample_count"],
        "source_prefix_callback_count": source["prefix"]["observed_callback_count"],
        "nominal_physics_substep_count": len(nominal_arm["ledger"]),
        "psf_physics_substep_count": len(psf_arm["ledger"]),
        "qp_solve_count": psf_arm["record"]["filter_solve_count"],
        "nominal_invalid_field_query_count": nominal_arm["invalid_count"],
        "psf_invalid_field_query_count": psf_arm["invalid_count"],
        "nominal_target_contact_record_count": sum(
            len(row["selected"]["target"]) for row in nominal_arm["ledger"]
        ),
        "psf_robot_selected_obstacle_contact_record_count": sum(
            len(row["selected"]["selected"]) for row in psf_arm["ledger"]
        ),
    }
    _same(counts, expected, "aggregate counts")
    return counts


def _validate_oob_terminal(
    artifact: Mapping[str, Any],
    *,
    source: Mapping[str, Any],
    samples: Mapping[str, Any],
    nominal: Mapping[str, Any],
) -> Tuple[Mapping[str, Any], Mapping[str, Any]]:
    if nominal["within_bounds"] or not nominal["violation_indices"]:
        raise OneStepCounterfactualError(
            "inadmissible nominal terminal has no reconstructed bound violation"
        )
    if (
        artifact["boundary_B_filter"] is not None
        or artifact["arms"] != []
        or artifact["diagnostics"] is not None
    ):
        raise OneStepCounterfactualError(
            "inadmissible nominal terminal executed or serialized counterfactual work"
        )
    counts = _exact_keys(
        artifact["counts"],
        (
            "bound_sample_count",
            "source_prefix_callback_count",
            "nominal_physics_substep_count",
            "psf_physics_substep_count",
            "qp_solve_count",
            "nominal_invalid_field_query_count",
            "psf_invalid_field_query_count",
            "nominal_target_contact_record_count",
            "psf_robot_selected_obstacle_contact_record_count",
        ),
        "inadmissible terminal counts",
    )
    expected_counts = {
        "bound_sample_count": samples["sample_count"],
        "source_prefix_callback_count": source["prefix"][
            "observed_callback_count"
        ],
        "nominal_physics_substep_count": 0,
        "psf_physics_substep_count": 0,
        "qp_solve_count": 0,
        "nominal_invalid_field_query_count": 0,
        "psf_invalid_field_query_count": 0,
        "nominal_target_contact_record_count": 0,
        "psf_robot_selected_obstacle_contact_record_count": 0,
    }
    _same(counts, expected_counts, "inadmissible terminal counts")
    classification = _exact_keys(
        artifact["classification"],
        (
            "label",
            "stage_13_passed",
            "contact_prevention_observed",
            "claim_scope",
            "local_feasibility_conditions",
            "nominal_reference",
            "typed_reasons",
        ),
        "inadmissible terminal classification",
    )
    expected_classification = {
        "label": (
            "inadmissible_nominal_velocity_outside_registered_controller_envelope"
        ),
        "stage_13_passed": False,
        "contact_prevention_observed": False,
        "claim_scope": (
            "registered_controller_envelope_inadmissibility_only_not_poisson_qp_or_contact_prevention"
        ),
        "local_feasibility_conditions": {
            "nominal_velocity_within_registered_controller_envelope": False,
            "no_hidden_clipping": True,
        },
        "nominal_reference": {
            "kind": "not_evaluated_preflight_inadmissible",
            "conditions": {},
            "passed": False,
        },
        "typed_reasons": [
            "nominal_velocity_outside_registered_controller_envelope"
        ],
    }
    _same(
        classification,
        expected_classification,
        "inadmissible terminal classification",
    )
    if artifact["passed"] is not False:
        raise OneStepCounterfactualError(
            "inadmissible nominal terminal cannot pass Stage 13"
        )
    return counts, classification


def _validate_result_hashes(artifact: Mapping[str, Any]) -> str:
    _validate_section_hash(artifact, "protocol_binding_sha256", artifact["protocol_binding"])
    _validate_section_hash(artifact, "authority_sha256", artifact["authority"])
    _validate_section_hash(artifact, "provenance_sha256", artifact["provenance"])
    _validate_section_hash(
        artifact,
        "nominal_velocity_estimate_sha256",
        artifact["nominal_velocity_estimate"],
    )
    _validate_section_hash(
        artifact, "boundary_B_filter_sha256", artifact["boundary_B_filter"]
    )
    _validate_section_hash(artifact, "arm_ledger_sha256", artifact["arms"])
    _validate_section_hash(
        artifact, "classification_ledger_sha256", artifact["classification"]
    )
    payload = dict(artifact)
    observed_payload_sha256 = _sha256(
        payload.pop("result_payload_sha256"), "result payload hash"
    )
    if observed_payload_sha256 != _sha(payload):
        raise OneStepCounterfactualError("result payload hash does not reconstruct")
    return observed_payload_sha256


def validate_one_step_counterfactual_result(
    result: Mapping[str, Any],
    *,
    protocol: Mapping[str, Any],
    protocol_raw_sha256: str,
    expected_authority: Mapping[str, Any],
) -> Dict[str, Any]:
    """Validate one complete Stage-13 result or raise.

    A scientifically negative but complete paired result is valid input.  Its
    returned ``stage_13_passed`` is false.  Only incomplete, malformed, or
    internally/external-authority-inconsistent evidence raises.
    """

    artifact = _exact_keys(result, tuple(TOP_LEVEL_KEYS), "Stage-13 result")
    if not isinstance(protocol, Mapping):
        raise OneStepCounterfactualError("protocol must be an object")
    if artifact["schema_version"] != RESULT_SCHEMA:
        raise OneStepCounterfactualError("result schema differs from protocol")
    if artifact["status"] != "complete":
        raise OneStepCounterfactualError("partial Stage-13 result is rejected")
    if artifact["scientific_result"] is not True:
        raise OneStepCounterfactualError("result is not a scientific paired result")
    run_id = artifact["run_id"]
    case_id = artifact["case_id"]
    if any(not isinstance(value, str) or not value for value in (run_id, case_id)):
        raise OneStepCounterfactualError("result run/case identity is empty")

    protocol_values = _validate_protocol(protocol, protocol_raw_sha256)
    authority = _validate_authority(artifact, expected_authority)
    if (
        run_id != authority["expected_run_id"]
        or case_id != protocol.get("case", {}).get("case_id")
        or case_id != authority["dynamic_authority"]["case"]["case_id"]
    ):
        raise OneStepCounterfactualError("result run/case identity differs")
    _validate_protocol_binding(
        artifact, protocol, protocol_raw_sha256, protocol_values, authority
    )
    provenance = _validate_provenance(
        artifact["provenance"],
        authority=authority,
        protocol=protocol,
        case_id=case_id,
    )
    registered = authority["dynamic_authority"]["runtime"][
        "registered_parameters"
    ]
    execution = _validate_execution(artifact["execution"])
    cadence = registered["cadence"]
    if (
        cadence["physics_timestep_s"] != protocol_values["timestep"]
        or cadence["active"]["physics_substeps_per_filter_update"]
        != protocol_values["period"]
        or cadence["active"]["filter_updates_per_high_level_action"] != 5
        or cadence["active"]["filter_frequency_hz"] != 100.0
    ):
        raise OneStepCounterfactualError(
            "runtime cadence differs from the Stage-13 protocol"
        )
    if (
        registered["admissibility"][
            "max_joint_velocity_tracking_linf_rad_s"
        ]
        != protocol_values["acceptance"][
            "joint_velocity_tracking_linf_max_rad_s"
        ]
        or registered["admissibility"][
            "max_joint_velocity_tracking_rmse_rad_s"
        ]
        != protocol_values["acceptance"][
            "joint_velocity_tracking_rmse_max_rad_s"
        ]
        or registered["admissibility"]["maximum_invalid_field_queries"]
        != protocol_values["acceptance"][
            "maximum_safe_arm_invalid_field_queries"
        ]
    ):
        raise OneStepCounterfactualError(
            "runtime admissibility differs from Stage-13 acceptance"
        )

    source = _validate_source_boundary(
        artifact["source_boundary"],
        protocol_values=protocol_values,
        authority=authority,
        physical_model=provenance["physical_model"],
        outcome_kind=str(execution["outcome_kind"]),
    )
    samples = source["samples"]
    resolved = _validate_resolved_geometry(
        samples["resolved_geometry"], source["target"], protocol
    )
    for sample in samples["samples"]:
        if (
            int(sample["body_id"]) not in resolved["ids"]["link56_body_ids"]
            or int(sample["geom_id"]) not in resolved["ids"]["link56_geom_ids"]
        ):
            raise OneStepCounterfactualError(
                "protected sample is outside resolved link5/6 authority"
            )
    full_sampling = _validate_full_robot_sampling(
        samples["full_robot_measurement_sampling"], resolved=resolved
    )
    samples["full_sampling"] = full_sampling
    if (
        full_sampling["record"]["epsilon_m"]
        != registered["coverage"]["epsilon_m"]
        or samples["binding"]["protected_sampling_epsilon_m"]
        != registered["coverage"]["epsilon_m"]
    ):
        raise OneStepCounterfactualError(
            "surface sampling epsilon differs from runtime authority"
        )

    nominal = _validate_nominal_velocity(
        artifact["nominal_velocity_estimate"],
        source=source,
        protocol=protocol,
        registered_parameters=registered,
    )
    if execution["outcome_kind"] == "preflight_inadmissible_nominal_velocity":
        counts, classification = _validate_oob_terminal(
            artifact,
            source=source,
            samples=samples,
            nominal=nominal,
        )
        payload_sha256 = _validate_result_hashes(artifact)
        return {
            "schema_version": (
                "vlsa_poisson_one_step_counterfactual_core_validation.v1"
            ),
            "status": "validated",
            "scientific_result": True,
            "run_id": run_id,
            "case_id": case_id,
            "contact_physical_boundary_C": source["contact_boundary"],
            "filter_boundary_B": source["boundary"],
            "counterfactual_nominal_contact_boundary_C_nom": None,
            "classification_label": classification["label"],
            "stage_13_passed": False,
            "complete_honest_negative": True,
            "result_payload_sha256": payload_sha256,
            "authority_sha256": artifact["authority_sha256"],
            "arm_ledger_sha256": artifact["arm_ledger_sha256"],
            "classification_ledger_sha256": artifact[
                "classification_ledger_sha256"
            ],
            "counts": dict(counts),
            "passed": True,
        }
    if not nominal["within_bounds"]:
        raise OneStepCounterfactualError(
            "paired execution used nominal velocity outside registered bounds"
        )
    state_B = source["prefix"]["state_at_B"]
    qp = _validate_qp(
        artifact["boundary_B_filter"],
        samples=samples,
        nominal_qdot=nominal["qdot_arm"],
        boundary_qpos=state_B["qpos"],
        arm_qpos_indices=nominal["arm_qpos_indices"],
        target_geom=int(source["target"]["robot_geom_id"]),
        protocol=protocol,
        registered_parameters=registered,
        warning_authorization=source["warning"],
    )

    gripper = float(source["gripper"]["exact_gripper_value"])
    nominal_action = [float(value) / 0.5 for value in nominal["qdot_arm"]] + [
        gripper
    ]
    boundary_measurement_raw = source["source"]["boundary_B_admissibility"][
        "measurement"
    ]
    boundary_measurement = _validate_measurement_row(
        boundary_measurement_raw,
        expected_boundary=source["boundary"],
        command=nominal["qdot_arm"],
        normalized_action=nominal_action,
        gripper=gripper,
        samples=samples,
        resolved=resolved,
        full_nv_count=nominal["record"]["full_nv_count"],
        physical_model=provenance["physical_model"],
        qp=qp,
        elapsed_s=None,
        label="boundary-B measurement",
    )
    if (
        boundary_measurement["record"]["integration_state_sha256"]
        != state_B["integration_state_sha256"]
        or boundary_measurement["qpos"] != state_B["qpos"]
        or boundary_measurement["qvel"] != state_B["qvel"]
        or boundary_measurement["h"] != qp["h"]
        or boundary_measurement["gradients"] != qp["gradients"]
        or boundary_measurement["record"]["ordered_sample_D_opt_m"]
        != qp["record"]["ordered_sample_D_opt_m"]
    ):
        raise OneStepCounterfactualError(
            "boundary-B measurement differs from exact state/QP inputs"
        )
    _validate_boundary_preflight(
        source["source"]["boundary_B_admissibility"],
        boundary_start=boundary_measurement["record"],
        registered_admissibility=registered["admissibility"],
    )

    arms_raw = _sequence(artifact["arms"], "counterfactual arms")
    if len(arms_raw) != 2:
        raise OneStepCounterfactualError("paired counterfactual requires two arms")
    nominal_arm = _validate_arm(
        arms_raw[0],
        arm_name="nominal",
        command=nominal["qdot_arm"],
        source=source,
        samples=samples,
        resolved=resolved,
        nominal=nominal,
        qp=qp,
        protocol_values=protocol_values,
        registered_parameters=registered,
        physical_model=provenance["physical_model"],
        controller_authority=provenance["joint_velocity_controller"],
    )
    psf_arm = _validate_arm(
        arms_raw[1],
        arm_name="psf",
        command=qp["qdot_safe"],
        source=source,
        samples=samples,
        resolved=resolved,
        nominal=nominal,
        qp=qp,
        protocol_values=protocol_values,
        registered_parameters=registered,
        physical_model=provenance["physical_model"],
        controller_authority=provenance["joint_velocity_controller"],
    )
    for field in (
        "model_topology_sha256",
        "physical_model_sha256",
        "compiled_mjb_sha256",
        "controller",
        "controller_software_state",
        "pid_memory_reset",
    ):
        _same(
            nominal_arm["restore"][field],
            psf_arm["restore"][field],
            "paired restore " + field,
        )
    start_drop = {"commanded_arm_qdot", "normalized_joint_velocity_action_8d"}
    nominal_start_projection = {
        key: item
        for key, item in nominal_arm["start"]["record"].items()
        if key not in start_drop
    }
    psf_start_projection = {
        key: item
        for key, item in psf_arm["start"]["record"].items()
        if key not in start_drop
    }
    boundary_start_projection = {
        key: item
        for key, item in boundary_measurement["record"].items()
        if key not in start_drop
    }
    exact_paired_start = bool(
        _canonical(nominal_start_projection)
        == _canonical(psf_start_projection)
        == _canonical(boundary_start_projection)
        and _canonical(
            nominal_arm["restore"]["controller_software_state"]
        )
        == _canonical(psf_arm["restore"]["controller_software_state"])
        and nominal_arm["restore"]["controller_software_state"]["sha256"]
        == psf_arm["restore"]["controller_software_state"]["sha256"]
    )
    if not exact_paired_start:
        raise OneStepCounterfactualError(
            "paired arm controller software or physical start differs"
        )

    diagnostics = _validate_diagnostics(
        artifact["diagnostics"],
        source=source,
        qp=qp,
        nominal_arm=nominal_arm,
        psf_arm=psf_arm,
        protocol_values=protocol_values,
    )
    counts = _validate_counts(
        artifact["counts"],
        source=source,
        samples=samples,
        nominal_arm=nominal_arm,
        psf_arm=psf_arm,
    )
    classification = _reconstruct_classification(
        protocol=protocol,
        source=source,
        qp=qp,
        nominal_arm=nominal_arm,
        psf_arm=psf_arm,
        diagnostics=diagnostics,
        exact_paired_start=exact_paired_start,
        nominal_within_bounds=nominal["within_bounds"],
        no_hidden_clipping=bool(execution["no_hidden_clipping"]),
    )
    _same(artifact["classification"], classification, "classification ledger")
    if artifact["passed"] is not classification["stage_13_passed"]:
        raise OneStepCounterfactualError("top-level passed flag differs")

    observed_payload_sha256 = _validate_result_hashes(artifact)

    return {
        "schema_version": (
            "vlsa_poisson_one_step_counterfactual_core_validation.v1"
        ),
        "status": "validated",
        "scientific_result": True,
        "run_id": run_id,
        "case_id": case_id,
        "contact_physical_boundary_C": source["contact_boundary"],
        "filter_boundary_B": source["boundary"],
        "counterfactual_nominal_contact_boundary_C_nom": diagnostics["C_nom"],
        "classification_label": classification["label"],
        "stage_13_passed": classification["stage_13_passed"],
        "complete_honest_negative": not classification["stage_13_passed"],
        "result_payload_sha256": observed_payload_sha256,
        "authority_sha256": artifact["authority_sha256"],
        "arm_ledger_sha256": artifact["arm_ledger_sha256"],
        "classification_ledger_sha256": artifact[
            "classification_ledger_sha256"
        ],
        "counts": dict(counts),
        "passed": True,
    }
