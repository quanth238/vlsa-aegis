from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import struct
import tempfile
import unittest

from main.poisson_fullbody.contracts import (
    ArtifactContractError,
    canonical_json_bytes,
    publish_hashed_json,
    sha256_bytes,
    sha256_file,
)
from main.poisson_fullbody.feasibility_protocol import PARAMETER_SECTIONS
from main.poisson_fullbody.jacobians import (
    PROTECTED_SAMPLE_DIFFERENTIAL_SCHEMA,
    _coupled_comparison,
    _parse_differential_config,
    _point_metrics,
    _stable_audit_hashes,
    _tangent_reconstruction,
    validate_protected_sample_differential_audit,
)
from scripts.validate_poisson_run_artifacts import (
    validate_run_artifacts as _validate_run_artifacts,
)
from tests.test_poisson_result_schema import valid_payload
from tests.test_poisson_shadow_identification import (
    synthetic_integration_state,
    synthetic_integration_state_sha256,
)


ROOT = Path(__file__).resolve().parents[1]


def validate_run_artifacts(result_path, artifact_root=None):
    return _validate_run_artifacts(
        result_path,
        artifact_root,
        expected_source_action_count=2,
    )


def _float64_vector_sha256(values):
    header = canonical_json_bytes({"dtype": "<f8", "shape": [len(values)]})
    data = struct.pack("<%dd" % len(values), *values)
    return sha256_bytes(b"vlsa-table1-array-v1\0" + header + b"\0" + data)


def _producer_sha256(value):
    return sha256_bytes(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    )


def _registered_runtime_parameter_block():
    protocol = json.loads(
        (ROOT / "configs/vlsa_poisson_runtime_protocol.canary.v3.json").read_text(
            encoding="utf-8"
        )
    )
    return {section: protocol[section] for section in PARAMETER_SECTIONS}


def _valid_query(point, value_m2):
    return {
        "point_world_m": list(point),
        "valid": True,
        "value_m2": float(value_m2),
        "gradient_m": [1.0, 0.0, 0.0],
        "reason": None,
        "cell_index": [1, 1, 1],
        "local_coordinates": [0.5, 0.5, 0.5],
        "outer_boundary_clearance_m": 0.1,
    }


def _exact_tangent_record(requested, arm_dofs, base_arm_qpos, step, config):
    requested = [float(value) for value in requested]
    plus_qpos = [
        float(base + float(step) * velocity)
        for base, velocity in zip(base_arm_qpos, requested)
    ]
    minus_qpos = [
        float(base - float(step) * velocity)
        for base, velocity in zip(base_arm_qpos, requested)
    ]
    plus_reconstructed = [
        float((perturbed - base) / float(step))
        for perturbed, base in zip(plus_qpos, base_arm_qpos)
    ]
    minus_reconstructed = [
        float((perturbed - base) / float(step))
        for perturbed, base in zip(minus_qpos, base_arm_qpos)
    ]
    return _tangent_reconstruction(
        requested,
        plus_reconstructed,
        minus_reconstructed,
        base_arm_qpos,
        plus_qpos,
        minus_qpos,
        arm_dofs,
        step,
        config,
    )


def _synthetic_protected_sample_differential_audit(
    samples,
    arm_dofs,
    integration_state_sha256,
    differential_audit_config,
):
    """Build a compact, exact linear-field audit accepted by the pure verifier."""

    config = _parse_differential_config(differential_audit_config)
    source_state = synthetic_integration_state()
    if integration_state_sha256 != synthetic_integration_state_sha256():
        raise AssertionError("trace fixture has the wrong integration-state hash")
    base_arm_qpos = source_state[1:8]
    delta = config["point_jacobian_delta_rad"]
    analytic = [[1.0] * 7, [0.0] * 7, [0.0] * 7]
    sample_records = []
    for sample in samples:
        base_point = [
            float(value) for value in sample["point_body_local_m"]
        ]
        base_value = 0.02 + base_point[0]
        plus_points = [
            [base_point[0] + delta, base_point[1], base_point[2]]
            for _ in range(7)
        ]
        minus_points = [
            [base_point[0] - delta, base_point[1], base_point[2]]
            for _ in range(7)
        ]
        numerical = [[1.0] * 7, [0.0] * 7, [0.0] * 7]
        point_tangents = []
        for column in range(7):
            direction = [0.0] * 7
            direction[column] = 1.0
            point_tangents.append(
                _exact_tangent_record(
                    direction, arm_dofs, base_arm_qpos, delta, config
                )
            )

        directions = []
        eta = config["coupled_eta_ladder_s"][0]
        base_query = _valid_query(base_point, base_value)
        for direction_index, arm_direction in enumerate(
            config["joint_velocity_directions_rad_s"]
        ):
            point_velocity = float(sum(arm_direction))
            plus_point = [
                base_point[0] + eta * point_velocity,
                base_point[1],
                base_point[2],
            ]
            minus_point = [
                base_point[0] - eta * point_velocity,
                base_point[1],
                base_point[2],
            ]
            attempt = {
                "eta_s": eta,
                "plus_point_world_m": plus_point,
                "minus_point_world_m": minus_point,
                "plus_query": _valid_query(
                    plus_point, base_value + eta * point_velocity
                ),
                "minus_query": _valid_query(
                    minus_point, base_value - eta * point_velocity
                ),
                "tangent_reconstruction": _exact_tangent_record(
                    arm_direction, arm_dofs, base_arm_qpos, eta, config
                ),
                "eligible_same_cell_stencil": True,
                "noneligible_reasons": [],
                "selected": True,
            }
            comparison = _coupled_comparison(
                attempt,
                analytic,
                base_query["gradient_m"],
                arm_direction,
                config,
            )
            directions.append(
                {
                    "direction_index": direction_index,
                    "arm_joint_velocity_rad_s": list(arm_direction),
                    "attempts": [attempt],
                    "selected_attempt_index": 0,
                    "derived_comparison": comparison,
                    "passed": True,
                }
            )

        point_diagnostics = _point_metrics(analytic, numerical, config)
        sample_records.append(
            {
                "identity": copy.deepcopy(sample),
                "base_point_world_m": base_point,
                "base_field_query": base_query,
                "analytic_point_jacobian_m_per_rad_3x7": copy.deepcopy(analytic),
                "plus_points_world_m_by_arm_dof": plus_points,
                "minus_points_world_m_by_arm_dof": minus_points,
                "numerical_point_jacobian_m_per_rad_3x7": numerical,
                "point_jacobian_metrics": point_diagnostics,
                "point_perturbation_tangent_reconstruction_by_arm_dof": (
                    point_tangents
                ),
                "point_jacobian_passed": True,
                "coupled_directions": directions,
                "eligible_coupled_direction_count": len(directions),
                "coupled_directions_passed": True,
                "passed": True,
            }
        )

    roundoff_authority = {
        "criterion": config["arm_tangent_roundtrip_criterion"],
        "mujoco_version": config["expected_mujoco_version"],
        "state_specification": "mjSTATE_INTEGRATION",
        "state_layout": "mjSTATE_INTEGRATION_qpos_prefix_at_offset_1",
        "model_nq": 7,
        "model_nv": 7,
        "arm_dof_indices": list(arm_dofs),
        "arm_dof_jntid": list(arm_dofs),
        "arm_joint_ids": list(arm_dofs),
        "arm_joint_names": config["expected_arm_joint_names"],
        "arm_qpos_indices": list(arm_dofs),
        "arm_joint_types": ["hinge"] * 7,
        "arm_jnt_dofadr": list(arm_dofs),
        "arm_jnt_qposadr": list(arm_dofs),
        "qpos_state_offset": 1,
        "source_state_element_count": len(source_state),
        "base_arm_qpos_rad": base_arm_qpos,
        "base_positions_match_source_state": True,
    }
    stable_hashes = _stable_audit_hashes(
        integration_state_sha256,
        arm_dofs,
        samples,
        config,
        roundoff_authority,
        sample_records,
    )
    state = {
        "state_specification": "mjSTATE_INTEGRATION",
        "state_layout": "mjSTATE_INTEGRATION_qpos_prefix_at_offset_1",
        "element_count": len(source_state),
        "source_initial_state_f64_le_hex": [
            struct.pack("<d", value).hex() for value in source_state
        ],
        "source_initial_sha256": integration_state_sha256,
        "clone_base_sha256": integration_state_sha256,
        "clone_final_sha256": integration_state_sha256,
        "source_final_sha256": integration_state_sha256,
        "source_state_unchanged": True,
        "clone_state_restored": True,
    }
    coupled_count = len(samples) * len(
        config["joint_velocity_directions_rad_s"]
    )
    audit = {
        "schema_version": PROTECTED_SAMPLE_DIFFERENTIAL_SCHEMA,
        "integration_state": state,
        "arm_dof_indices": list(arm_dofs),
        "arm_scalar_hinge_roundoff_authority": roundoff_authority,
        "arm_scalar_hinge_roundoff_authority_sha256": stable_hashes[
            "arm_scalar_hinge_roundoff_authority_sha256"
        ],
        "ordered_samples": copy.deepcopy(samples),
        "ordered_sample_identity_sha256": sha256_bytes(
            canonical_json_bytes(samples)
        ),
        "specification_sha256": stable_hashes["specification_sha256"],
        "binding_sha256": stable_hashes["binding_sha256"],
        "classification_ledger_sha256": stable_hashes[
            "classification_ledger_sha256"
        ],
        "differential_audit_config": config,
        "sample_records": sample_records,
        "counts": {
            "sample_count": len(samples),
            "point_jacobian_passed_sample_count": len(samples),
            "required_coupled_direction_count": coupled_count,
            "selected_coupled_direction_count": coupled_count,
            "passed_coupled_direction_count": coupled_count,
            "coupled_attempt_count": coupled_count,
            "noneligible_coupled_attempt_count": 0,
            "passed_sample_count": len(samples),
        },
        "passed": True,
    }
    audit["audit_payload_sha256"] = sha256_bytes(canonical_json_bytes(audit))
    summary = validate_protected_sample_differential_audit(
        audit,
        expected_samples=samples,
        expected_arm_dof_indices=arm_dofs,
        expected_integration_state_sha256=integration_state_sha256,
        expected_differential_audit_config=config,
    )
    return audit, summary


def _snapshot(step, value, previous, kind):
    values = [bool(value)]
    prior = [False] if previous is None else [bool(previous)]
    return {
        "step": step,
        "values": values,
        "satisfied_count": int(value),
        "fraction": float(value),
        "all_satisfied": bool(value),
        "newly_satisfied_indices": [0] if value and not prior[0] else [],
        "regressed_indices": [0] if prior[0] and not value else [],
        "argument_poses": [],
        "simulator_state_sha256_before": "%064x" % (step + 2),
        "simulator_state_sha256_after": "%064x" % (step + 2),
        "inert": True,
        "snapshot_kind": kind,
    }


def _contact_record(
    source_phase,
    *,
    observation_index=0,
    distance_m=-0.001,
    physical_flag=None,
):
    settled = source_phase == "settled_post_integration_recomputed"
    if physical_flag is None:
        physical_flag = distance_m <= 0.0
    high = None if settled else observation_index // 25
    inner = None if settled else (observation_index // 5) % 5
    physics = None if settled else observation_index % 5
    return {
        "source_phase": source_phase,
        "observation_index": None if settled else observation_index,
        "high_level_index": high,
        "inner_control_index": inner,
        "physics_substep_index": physics,
        "mujoco_contact_index": 0,
        "is_physical_nonpositive_distance_contact": bool(physical_flag),
        "within_registered_near_contact_tolerance": bool(distance_m <= 0.001),
        "solver_constraint_active": True,
        "efc_address": 0,
        "mujoco_geom1_id": 55,
        "mujoco_geom1_name": "robot_geom_55",
        "mujoco_geom2_id": 100,
        "mujoco_geom2_name": "moka_pot_obstacle_1_g0",
        "robot_geom_id": 55,
        "robot_geom_name": "robot_geom_55",
        "robot_body_id": 55,
        "robot_body_name": "robot0_link5",
        "obstacle_geom_id": 100,
        "obstacle_geom_name": "moka_pot_obstacle_1_g0",
        "obstacle_body_id": 100,
        "obstacle_body_name": "moka_pot_obstacle_1_main",
        "contact_distance_m": distance_m,
        "contact_includemargin_m": 0.001,
        "robot_geom_margin_m": 0.0,
        "robot_geom_gap_m": 0.0,
        "obstacle_geom_margin_m": 0.0,
        "obstacle_geom_gap_m": 0.0,
        "explicit_pair_id": None,
        "explicit_pair_margin_m": None,
        "explicit_pair_gap_m": None,
        "position_world_m": [0.0, 0.0, 0.0],
        "frame_normal_mujoco_geom1_to_geom2_world": [1.0, 0.0, 0.0],
        "force_available": False,
        "force_semantics": (
            "live_solver_constraint_wrench; impulse fields are wrench_times_"
            "physics_timestep_estimates"
            if source_phase == "live_solver_phase_preintegration_geometry"
            else "omitted_recomputed_not_applied"
        ),
        "force_contact_frame_n": None,
        "force_world_n": None,
        "normal_force_n": None,
        "impulse_estimate_contact_frame_ns": None,
        "impulse_estimate_world_ns": None,
        "force_unavailable_reason": "test fixture force unavailable",
    }


def _add_candidate_without_physical_summary(
    trace, phase, *, physical_flag, distance_m=-0.001
):
    record = _contact_record(
        phase,
        physical_flag=physical_flag,
        distance_m=distance_m,
    )
    measurement = trace["outcome"]["monitor"]["record"]
    measurement["total_candidate_contact_point_record_count"] += 1
    if measurement["first_candidate_contact_point_record"] is None:
        measurement["first_candidate_contact_point_record"] = record
    if phase == "settled_post_integration_recomputed":
        settled = measurement["settled_state"]
        settled["candidate_contact_point_records"].append(record)
        settled["candidate_contact_point_record_count"] += 1
        settled["solver_active_contact_point_record_count"] += 1
        settled["within_near_contact_tolerance_point_record_count"] += 1
        settled["first_candidate_contact_point_record"] = record
    elif phase == "live_solver_phase_preintegration_geometry":
        measurement["live_solver_phase_contact_point_records"].append(record)
        measurement["live_solver_candidate_contact_point_record_count"] += 1
        measurement["live_solver_active_contact_point_record_count"] += 1
        measurement[
            "live_solver_within_near_contact_tolerance_point_record_count"
        ] += 1
    else:
        measurement["post_state_candidate_contact_point_records"].append(record)
        measurement["post_state_candidate_contact_point_record_count"] += 1
        measurement["post_state_solver_active_contact_point_record_count"] += 1
        measurement[
            "post_state_within_near_contact_tolerance_point_record_count"
        ] += 1
    return record


def _add_consistent_post_physical_contact(payload, trace):
    record = _add_candidate_without_physical_summary(
        trace,
        "post_integration_recomputed",
        physical_flag=True,
    )
    monitor = trace["outcome"]["monitor"]
    measurement = monitor["record"]
    measurement["post_state_physical_contact_point_records"] = [record]
    measurement["total_physical_contact_point_record_count"] = 1
    measurement["rollout_phase_physical_contact_point_record_count"] = 1
    measurement["post_state_physical_contact_point_record_count"] = 1
    measurement["any_robot_obstacle_contact"] = True
    measurement["link56_obstacle_contact"] = True
    measurement["rollout_any_robot_obstacle_contact"] = True
    measurement["rollout_link56_obstacle_contact"] = True
    measurement["post_state_any_robot_obstacle_contact"] = True
    measurement["post_state_link56_obstacle_contact"] = True
    measurement["first_physical_contact_point_record"] = record
    measurement["first_post_state_physical_contact_point_record"] = record
    measurement["D_sim_min_m"] = -0.001
    measurement["contact_authority_clamped_D_sim"] = True
    monitor.update(
        {
            "D_sim_min_m": -0.001,
            "any_contact": True,
            "link56_contact": True,
            "total": 1,
            "rollout": 1,
            "post": 1,
            "first": record,
            "first_post": record,
        }
    )
    safety = payload["endpoints"]["safety"]
    safety["D_sim_min_m"]["value"] = -0.001
    safety.update(
        {
            "any_robot_obstacle_contact": True,
            "link56_obstacle_contact": True,
            "total_physical_contact_point_record_count": 1,
            "rollout_phase_physical_contact_point_record_count": 1,
            "post_state_physical_contact_point_record_count": 1,
            "first_physical_contact_point_record": record,
            "first_post_state_physical_contact_point_record": record,
        }
    )
    return record


def _forge_source_settled_obstacle_hash(payload, trace):
    trace["source_replay"]["settled_active_obstacle_position_sha256"] = "0" * 64
    payload["runtime"]["sampler"]["configuration_sha256"] = _producer_sha256(
        {
            "source": trace["source_replay"],
            "active_policy_queries": 0,
            "active_policy_rng_exercised": False,
        }
    )


def _forge_static_drift_as_admissible(payload, trace):
    monitor = trace["outcome"]["monitor"]
    drift = monitor["record"]["obstacle_pose_drift"]
    geom = drift["geoms"][0]
    drift.update(
        {
            "maximum_translation_m": 1.0,
            "maximum_rotation_rad": 1.0,
            "maximum_surface_point_displacement_m": 1.0,
            "surface_drift_threshold_crossed": True,
            "first_surface_drift_threshold_crossing_observation_index": 0,
        }
    )
    geom.update(
        {
            "maximum_translation_m": 1.0,
            "maximum_rotation_rad": 1.0,
            "maximum_surface_point_displacement_m": 1.0,
            "final_translation_m": 1.0,
            "final_rotation_rad": 1.0,
            "final_surface_point_displacement_m": 1.0,
        }
    )
    monitor.update(
        {
            "translation_drift_m": 1.0,
            "rotation_drift_rad": 1.0,
            "surface_drift_m": 1.0,
        }
    )
    validity = trace["outcome"]["validity"]
    validity.update(
        {
            "obstacle_translation_drift_m": 1.0,
            "obstacle_rotation_drift_rad": 1.0,
            "static_admissible": True,
        }
    )
    payload["endpoints"]["validity"] = copy.deepcopy(validity)


def _forge_failed_poisson_diagnostics(payload, trace):
    trace["field_bundle_diagnostics"]["poisson"][
        "backward_error_linf"
    ] = 1.0
    identity = {
        key: value
        for key, value in trace["field_bundle_hashes"].items()
        if key != "bundle_sha256"
    }
    identity["diagnostics"] = trace["field_bundle_diagnostics"]
    trace["field_bundle_hashes"]["bundle_sha256"] = sha256_bytes(
        canonical_json_bytes(identity)
    )


def _forge_realized_query_population(payload, trace):
    rows = trace["outcome"]["realized_cbf_trace"]
    for row in rows:
        row["query_count"] = 999
    evaluations = len(rows) * 999
    trace["outcome"]["realized_cbf_audit"][
        "residual_evaluation_count"
    ] = evaluations
    trace["outcome"]["validity"][
        "realized_cbf_residual_evaluation_count"
    ] = evaluations
    payload["endpoints"]["validity"][
        "realized_cbf_residual_evaluation_count"
    ] = evaluations


def _fixture():
    payload = valid_payload()
    payload["pairing"][
        "restored_settled_state_sha256"
    ] = synthetic_integration_state_sha256()
    entered_actions = [
        [0.0] * 7,
        [0.1, 0.1, 0.1, 0.0, 0.0, 0.0, 0.1],
    ]
    nominal = [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    issued = [0.08, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    correction = [safe - source for safe, source in zip(issued, nominal)]
    nominal_ledger = []
    executed_ledger = []
    inner_trace = []
    physics_trace = []
    realized_trace = []
    for inner_global in range(10):
        high = inner_global // 5
        inner = inner_global % 5
        nominal_ledger.append(
            {
                "high_level_index": high,
                "inner_control_index": inner,
                "qdot_rad_s": list(nominal),
            }
        )
        executed_ledger.append(
            {
                "high_level_index": high,
                "inner_control_index": inner,
                "qdot_rad_s": list(issued),
            }
        )
        inner_trace.append(
            {
                "high_level_index": high,
                "inner_control_index": inner,
                "source_action": list(entered_actions[high]),
                "adapter": {
                    "normalized_action": [
                        0.2,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        entered_actions[high][6],
                    ],
                    "qdot_physical_rad_s": list(nominal),
                    "qdot_unclipped_rad_s": list(nominal),
                    "desired_twist": [0.0] * 6,
                    "achieved_twist": [0.0] * 6,
                    "diagnostics": {},
                },
                "execution": {
                    "nominal_qdot_physical_rad_s": list(nominal),
                    "executed_qdot_physical_rad_s": list(issued),
                    "filter_correction_rad_s": list(correction),
                    "filter_correction_l2_rad_s": math.sqrt(
                        sum(value * value for value in correction)
                    ),
                    "normalized_executed_action": [
                        0.16,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        entered_actions[high][6],
                    ],
                },
                "D_opt_min_m": 0.04,
                "minimum_h_m2": 0.005,
                "minimum_nominal_cbf_residual_m2_per_s": -0.2,
                "minimum_safe_cbf_residual_m2_per_s": -1e-8,
                "qp": {
                    "solver": "osqp",
                    "status": "solved",
                    "status_value": 1,
                    "iterations": 25,
                    "solve_time_seconds": 0.001,
                    "input_constraint_count": 2,
                    "solved_constraint_count": 2,
                    "trivial_zero_constraint_count": 0,
                    "minimum_normalized_cbf_residual": -1e-9,
                    "minimum_raw_cbf_residual_m2_per_s": -1e-8,
                    "minimum_nonzero_row_scale_m2_per_rad": 0.1,
                    "maximum_nonzero_row_scale_m2_per_rad": 0.2,
                    "maximum_velocity_bound_violation_rad_s": 0.0,
                    "nominal_minimum_normalized_cbf_residual": -0.02,
                    "nominal_minimum_raw_cbf_residual_m2_per_s": -0.2,
                    "correction_l2_rad_s": math.sqrt(
                        sum(value * value for value in correction)
                    ),
                    "nominal_feasible": False,
                },
            }
        )
    for global_substep in range(50):
        high = global_substep // 25
        inner = (global_substep // 5) % 5
        physics = global_substep % 5
        position = [(global_substep + 1) * 0.001, 0.0, 0.0]
        physics_trace.append(
            {
                "high_level_index": high,
                "inner_control_index": inner,
                "physics_substep_index": physics,
                "nominal_joint_velocity_command_rad_s": list(nominal),
                "issued_joint_velocity_command_rad_s": list(issued),
                "measured_arm_joint_velocity_rad_s": list(issued),
                "forwarded_eef_position_m": position,
            }
        )
        realized_trace.append(
            {
                "high_level_index": high,
                "inner_control_index": inner,
                "physics_substep_index": physics,
                "D_opt_min_m": 0.04,
                "query_count": 2,
                "valid": True,
                "minimum_h_m2": 0.005,
                "minimum_realized_cbf_residual_m2_per_s": 1e-6,
            }
        )

    nominal_motion = 50 * 0.1 * 0.002
    safe_motion = 50 * 0.08 * 0.002
    correction_motion = 50 * 0.02 * 0.002
    usefulness = {
        "nominal_joint_motion_integral_rad": nominal_motion,
        "safe_joint_motion_integral_rad": safe_motion,
        "measured_joint_motion_integral_rad": safe_motion,
        "eef_path_length_m": 0.05,
        "correction_integral_rad": correction_motion,
        "motion_retention_ratio": safe_motion / nominal_motion,
        "zero_motion_fraction": 0.0,
        "all_issued_arm_joint_commands_zero": False,
        "safety_by_no_execution": False,
    }
    task = copy.deepcopy(payload["endpoints"]["task"])
    validity = copy.deepcopy(payload["endpoints"]["validity"])
    validity.update(
        {
            "maximum_velocity_tracking_error_rad_s": 0.0,
            "velocity_tracking_rmse_rad_s": 0.0,
            "realized_cbf_observed_physics_substep_count": 50,
            "realized_cbf_residual_evaluation_count": 100,
        }
    )
    car_ledger = {
        "settled_active_obstacle_root_position_m": [0.0, 0.0, 0.0],
        "settled_active_obstacle_position_sha256": _float64_vector_sha256(
            [0.0, 0.0, 0.0]
        ),
        "source_settled_active_obstacle_position_sha256": _float64_vector_sha256(
            [0.0, 0.0, 0.0]
        ),
        "historical_settled_active_obstacle_position_sha256": _float64_vector_sha256(
            [0.0, 0.0, 0.0]
        ),
        "completed_post_step_positions": [
            {
                "high_level_index": 0,
                "position_m": [0.0, 0.0, 0.0],
                "l1_displacement_from_settled_m": 0.0,
            },
            {
                "high_level_index": 1,
                "position_m": [0.001, 0.0, 0.0],
                "l1_displacement_from_settled_m": 0.001,
            },
        ],
    }
    car = copy.deepcopy(payload["endpoints"]["safety"]["paper_car"])
    car["position_ledger_sha256"] = sha256_bytes(canonical_json_bytes(car_ledger))
    goal_definition = {
        "schema_version": "vlsa_native_goal_progress.v1",
        "source": "native_bddl_goal_predicates",
        "logic": "conjunction",
        "goal_atoms": [
            {
                "index": 0,
                "predicate": "on",
                "arguments": ["akita_black_bowl_1", "plate_1"],
            }
        ],
    }
    goal_definition["goal_definition_sha256"] = sha256_bytes(
        canonical_json_bytes(goal_definition)
    )
    goal_ledger = [
        _snapshot(-1, False, None, "settled_pre_action"),
        _snapshot(0, False, False, "completed_high_level_post_step"),
        _snapshot(1, True, False, "completed_high_level_post_step"),
    ]
    settled_record = {
        "candidate_contact_point_record_count": 0,
        "solver_active_contact_point_record_count": 0,
        "within_near_contact_tolerance_point_record_count": 0,
        "physical_contact_point_record_count": 0,
        "any_robot_obstacle_contact": False,
        "link56_obstacle_contact": False,
        "first_candidate_contact_point_record": None,
        "first_physical_contact_point_record": None,
        "candidate_contact_point_records": [],
        "physical_contact_point_records": [],
        "obstacle_motion_admissibility": {
            "reference": (
                "clone_forwarded_settled_state_mj_objectVelocity_world_"
                "orientation_rot_then_lin"
            ),
            "max_linear_speed_threshold_m_per_s": 1e-4,
            "max_angular_speed_threshold_rad_per_s": 1e-4,
            "maximum_observed_linear_speed_m_per_s": 0.0,
            "maximum_observed_angular_speed_rad_per_s": 0.0,
            "admissible": True,
            "reasons": [],
            "bodies": [
                {
                    "body_id": 100,
                    "body_name": "moka_pot_obstacle_1_main",
                    "angular_velocity_world_rad_per_s": [0.0, 0.0, 0.0],
                    "linear_velocity_world_m_per_s": [0.0, 0.0, 0.0],
                    "angular_speed_rad_per_s": 0.0,
                    "linear_speed_m_per_s": 0.0,
                    "angular_speed_exceeds_threshold": False,
                    "linear_speed_exceeds_threshold": False,
                }
            ],
        },
        "raw_mj_geom_distance": {
            "available": False,
            "authority": "advisory_only_never_contact_authority",
            "minimum_distance_m": None,
            "distance_query_limit_m": 1.0,
            "minimum_observation_index": None,
            "robot_geom_id": None,
            "robot_geom_name": None,
            "obstacle_geom_id": None,
            "obstacle_geom_name": None,
            "fromto_world_m": None,
        },
        "sample_clearance": {
            "available": True,
            "authority": "exact_sample_to_obb_plus_certified_coverage_lower_bound",
            "minimum_exact_sample_to_obb_distance_m": 0.05931058286977641,
            "certified_coverage_radius_m": 0.04931058286977641,
            "full_surface_clearance_lower_bound_m": 0.01,
            "minimum_observation_index": None,
            "sample_id": 0,
            "robot_geom_id": 55,
            "obstacle_geom_id": 100,
        },
        "D_sim_m": 0.01,
        "contact_authority_clamped_D_sim": False,
    }
    measurement_record = {
        "observed_physics_substeps": 50,
        "first_index": [0, 0, 0],
        "last_index": [1, 4, 4],
        "settled_state": settled_record,
        "total_candidate_contact_point_record_count": 0,
        "total_physical_contact_point_record_count": 0,
        "rollout_phase_physical_contact_point_record_count": 0,
        "rollout_any_robot_obstacle_contact": False,
        "live_solver_candidate_contact_point_record_count": 0,
        "live_solver_active_contact_point_record_count": 0,
        "live_solver_within_near_contact_tolerance_point_record_count": 0,
        "live_solver_nonpositive_contact_point_record_count": 0,
        "post_state_candidate_contact_point_record_count": 0,
        "post_state_solver_active_contact_point_record_count": 0,
        "post_state_within_near_contact_tolerance_point_record_count": 0,
        "post_state_physical_contact_point_record_count": 0,
        "any_robot_obstacle_contact": False,
        "link56_obstacle_contact": False,
        "rollout_link56_obstacle_contact": False,
        "live_solver_any_robot_obstacle_contact": False,
        "live_solver_link56_obstacle_contact": False,
        "post_state_any_robot_obstacle_contact": False,
        "post_state_link56_obstacle_contact": False,
        "first_candidate_contact_point_record": None,
        "first_physical_contact_point_record": None,
        "first_live_solver_physical_contact_point_record": None,
        "first_post_state_physical_contact_point_record": None,
        "live_solver_phase_contact_point_records": [],
        "post_state_candidate_contact_point_records": [],
        "post_state_physical_contact_point_records": [],
        "raw_mj_geom_distance": {
            "available": False,
            "authority": "advisory_only_never_contact_authority",
            "minimum_distance_m": None,
            "distance_query_limit_m": 1.0,
            "minimum_observation_index": None,
            "robot_geom_id": None,
            "robot_geom_name": None,
            "obstacle_geom_id": None,
            "obstacle_geom_name": None,
            "fromto_world_m": None,
        },
        "sample_clearance": {
            "available": True,
            "authority": "exact_sample_to_obb_plus_certified_coverage_lower_bound",
            "minimum_exact_sample_to_obb_distance_m": 0.05931058286977641,
            "certified_coverage_radius_m": 0.04931058286977641,
            "full_surface_clearance_lower_bound_m": 0.01,
            "minimum_observation_index": None,
            "sample_id": 0,
            "robot_geom_id": 55,
            "obstacle_geom_id": 100,
        },
        "obstacle_pose_drift": {
            "reference": "every_selected_obstacle_collision_geom_pose_after_settling",
            "maximum_translation_m": 1e-6,
            "maximum_rotation_rad": 2e-6,
            "maximum_surface_point_displacement_m": 5e-7,
            "surface_drift_threshold_m": 1e-6,
            "surface_drift_threshold_crossed": False,
            "first_surface_drift_threshold_crossing_observation_index": None,
            "geoms": [
                {
                    "geom_id": 100,
                    "geom_name": "moka_pot_obstacle_1_g0",
                    "body_id": 100,
                    "body_name": "moka_pot_obstacle_1_main",
                    "maximum_translation_m": 1e-6,
                    "maximum_rotation_rad": 2e-6,
                    "maximum_surface_point_displacement_m": 5e-7,
                    "final_translation_m": 1e-6,
                    "final_rotation_rad": 2e-6,
                    "final_surface_point_displacement_m": 5e-7,
                    "maximum_translation_observation_index": 49,
                    "maximum_rotation_observation_index": 49,
                    "maximum_surface_displacement_observation_index": 49,
                }
            ],
        },
        "D_sim_min_m": 0.01,
        "D_sim_semantics": (
            "union_of_settled_live_solver_and_forwarded_post_state_"
            "nonpositive_contacts_plus_exact_obb_coverage_lower_bound"
        ),
        "physical_contact_distance_semantics": "mujoco_contact_dist_le_0",
        "near_contact_tolerance_m": 0.001,
        "contact_authority_clamped_D_sim": False,
    }
    monitor = {
        "D_sim_min_m": 0.01,
        "any_contact": False,
        "link56_contact": False,
        "total": 0,
        "settled": 0,
        "rollout": 0,
        "live": 0,
        "post": 0,
        "first": None,
        "first_settled": None,
        "first_live": None,
        "first_post": None,
        "translation_drift_m": 1e-6,
        "rotation_drift_rad": 2e-6,
        "surface_drift_m": 5e-7,
        "record": measurement_record,
    }
    controller_state = {
        "schema_version": (
            "vlsa_poisson_joint_velocity_controller_software_state.v1"
        ),
        "fields": {"fixture": {"available": True, "value": 0.0}},
    }
    controller_state["sha256"] = _producer_sha256(controller_state)
    controller_contract = {"fixture": "joint_velocity_controller_contract"}
    outcome = {
        "arm": payload["arm"],
        "completion_class": "executed",
        "terminal_reason": payload["execution"]["terminal_reason"],
        "exposure_complete": True,
        "selected_initial_state_sha256": payload["case_identity"][
            "initial_state_record_sha256"
        ],
        "active_initial_observation_sha256": payload["pairing"][
            "active_joint_velocity_initial_observation_sha256"
        ],
        "restore": {
            "schema_version": "vlsa_poisson_controller_state_restore.v1",
            "source_controller": "OSC_POSE",
            "target_controller": "JOINT_VELOCITY",
            "model_topology_sha256": "c" * 64,
            "settled_state_sha256": "f" * 64,
            "official_integration_state_sha256": payload["pairing"][
                "restored_settled_state_sha256"
            ],
            "target_official_integration_state_sha256": payload["pairing"][
                "restored_settled_state_sha256"
            ],
            "official_integration_state_available": True,
            "target_state_sha256": "f" * 64,
            "exact_flattened_state": True,
            "maximum_arm_qpos_error_rad": 0.0,
            "maximum_arm_qvel_error_rad_s": 0.0,
            "max_arm_qpos_error_tolerance_rad": 1e-10,
            "max_arm_qvel_error_tolerance_rad_s": 1e-10,
            "copied_timestep": 20,
            "copied_cur_time_s": 10.0,
            "copied_done": False,
            "physical_model_sha256": payload["pairing"][
                "compiled_physical_model_sha256"
            ],
            "compiled_mjb_sha256": payload["pairing"]["compiled_mjb_sha256"],
            "controller": controller_contract,
            "controller_software_state": controller_state,
            "pid_memory_reset": {
                "goal_velocity_zero": True,
                "current_velocity_zero": True,
                "last_error_zero": True,
                "summed_error_zero": True,
                "derivative_buffer_size": 0,
                "saturated": False,
            },
        },
        "goal_definition": goal_definition,
        "goal_progress_ledger": goal_ledger,
        "initial_protected_sample_audit": {
            "protected_sample_count": 2,
            "field_query_count": 2,
            "valid_field_query_count": 2,
            "minimum_h_m2": 0.004,
            "D_opt_min_m": 0.03,
            "strict_safe_start": True,
        },
        "prefix": {
            "high_level_steps": 2,
            "inner_control_steps": 10,
            "physics_substeps": 50,
            "physics_exposure_seconds": 0.1,
        },
        "physics_clock": {
            "settled_time_s": 10.0,
            "terminal_time_s": 10.1,
            "observed_exposure_s": 0.1,
            "expected_from_completed_substeps_s": 0.1,
            "exact_count_consistent_within_abs_1e_10_s": True,
        },
        "entered_source_actions": entered_actions,
        "completed_source_actions": copy.deepcopy(entered_actions),
        "nominal_ledger": nominal_ledger,
        "executed_ledger": executed_ledger,
        "inner_trace": inner_trace,
        "physics_trace": physics_trace,
        "fail_closed_attempt_trace": [],
        "realized_cbf_trace": realized_trace,
        "monitor": monitor,
        "paper_car": car,
        "paper_car_position_ledger": car_ledger,
        "task": task,
        "usefulness": usefulness,
        "validity": validity,
        "optimizer_counts": {
            "attempt_count": 10,
            "solved_count": 10,
            "infeasible_count": 0,
            "solver_failure_count": 0,
            "postcheck_failure_count": 0,
        },
        "optimizer_terminal_status": "solved",
        "minimums": {
            "h_m2": 0.004,
            "D_opt_m": 0.03,
            "nominal_raw_cbf_residual_m2_per_s": -0.2,
            "safe_raw_cbf_residual_m2_per_s": -1e-8,
            "safe_normalized_cbf_residual": -1e-9,
            "realized_raw_cbf_residual_m2_per_s": 1e-6,
        },
        "realized_cbf_audit": {
            "semantics": "every_completed_2ms_post_state_grad_h_T_J_qvel_actual_plus_alpha_h",
            "observed_physics_substep_count": 50,
            "residual_evaluation_count": 100,
            "negative_residual_callback_count": 0,
            "first_negative_residual": None,
        },
        "eef_path_audit": {
            "semantics": "settled_forwarded_grip_site_plus_every_completed_2ms_post_state",
            "position_sample_count": 51,
            "settled_forwarded_eef_position_m": [0.0, 0.0, 0.0],
        },
        "eef_reference_site": {
            "site_id": 0,
            "site_name": "gripper0_grip_site",
            "jacobian_row_order": "linear_xyz_then_angular_xyz",
            "shape": [6, 7],
            "linear_finite_difference_delta_rad": 1e-6,
            "maximum_linear_jacobian_error_m_per_rad": 1e-7,
            "absolute_tolerance_m_per_rad": 2e-6,
            "passed": True,
            "live_state_mutated": False,
        },
        "tracking": {
            "command_count": 10,
            "verified_terminal_command_count": 10,
            "observed_physics_substep_count": 50,
            "maximum_linf_error_rad_s": 0.0,
            "cumulative_rmse_rad_s": 0.0,
            "maximum_linf_threshold_rad_s": 0.05,
            "maximum_rmse_threshold_rad_s": 0.02,
            "final_fifth_command_verified": True,
            "first_threshold_crossing": None,
        },
        "terminal_simulator_state_sha256": payload["execution"]["terminal_simulator_state_sha256"],
        "terminal_observation_sha256": payload["execution"]["terminal_observation_sha256"],
    }
    robot_geom_ids = [55, 56] + list(range(70, 83)) + [84]
    robot_geom_names = [
        "robot_geom_%d" % value for value in robot_geom_ids[:-1]
    ] + ["mount0_pedestal_col"]
    surface_records = []
    for index, (geom_id, geom_name) in enumerate(
        zip(robot_geom_ids, robot_geom_names)
    ):
        if index < 11:
            type_id, type_name = 7, "mesh"
            geometry_kind = "compiled_mesh_convex_hull"
            certificate_kind = "analytic_triangle_lattice_covering_bound"
            certificate_parameters = {
                "triangle_count": 12,
                "requested_epsilon_m": 0.05,
            }
            geom_size = [1.0, 1.0, 1.0]
            sample_count = 8
            cover = 0.04
        elif index < 15:
            type_id, type_name = 6, "box"
            geometry_kind = "exact_box_faces"
            certificate_kind = "analytic_triangle_lattice_covering_bound"
            certificate_parameters = {
                "triangle_count": 12,
                "requested_epsilon_m": 0.05,
            }
            geom_size = [0.01, 0.02, 0.03]
            sample_count = 8
            cover = 0.04
        else:
            type_id, type_name = 5, "cylinder"
            geometry_kind = "exact_cylinder_surface"
            certificate_kind = "analytic_cylinder_parameter_grid_covering_bound"
            certificate_parameters = {
                "angular_sample_count": 16,
                "axial_interval_count": 9,
                "cap_radial_interval_count": 3,
                "requested_epsilon_m": 0.05,
                "implementation_caps": {
                    "maximum_angular_samples": 100000,
                    "maximum_axial_intervals": 100000,
                    "maximum_radial_intervals": 100000,
                    "maximum_raw_sample_count": 1000000,
                },
            }
            geom_size = [0.18, 0.31, 0.0]
            sample_count = 226
            cover = 0.04931058286977641
        surface_records.append(
            {
                "geom_id": geom_id,
                "geom_name": geom_name,
                "body_id": geom_id,
                "body_name": (
                    "robot0_link5"
                    if geom_id == 55
                    else "robot0_link6"
                    if geom_id == 56
                    else "robot_body_%d" % geom_id
                ),
                "geom_type_id": type_id,
                "geom_type_name": type_name,
                "geom_size": geom_size,
                "contype": 1,
                "conaffinity": 1,
                "mask_collision_enabled": True,
                "selection_authority": "authoritative_resolved_geom_ids",
                "geometry_kind": geometry_kind,
                "certificate_kind": certificate_kind,
                "certificate_parameters": certificate_parameters,
                "surface_element_count": (
                    240 if type_name == "cylinder" else 12
                ),
                "sample_count": sample_count,
                "certified_surface_cover_radius_m": cover,
            }
        )
    full_sample_count = sum(record["sample_count"] for record in surface_records)
    trace = {
        "schema_version": "vlsa_poisson_active_arm_trace.v4",
        "scientific_result": False,
        "run_id": payload["run_id"],
        "case_id": payload["case_id"],
        "arm": payload["arm"],
        "staged_scope": "first_two_arm_canary_only",
        "resolved_geometry": {
            "robot_root_body_ids": [55],
            "robot_body_ids": robot_geom_ids,
            "robot_body_names": [
                (
                    "robot0_link5"
                    if value == 55
                    else "robot0_link6"
                    if value == 56
                    else "robot_body_%d" % value
                )
                for value in robot_geom_ids
            ],
            "robot_geom_ids": robot_geom_ids,
            "robot_geom_names": robot_geom_names,
            "obstacle_body_ids": [100],
            "obstacle_root_body_ids": [100],
            "obstacle_body_names": ["moka_pot_obstacle_1_main"],
            "obstacle_geom_ids": [100],
            "obstacle_geom_names": ["moka_pot_obstacle_1_g0"],
            "link56_geom_ids": [55, 56],
            "link56_geom_names": ["robot_geom_55", "robot_geom_56"],
            "link56_body_ids": [55, 56],
            "collision_enabled_pairs": [
                [geom_id, 100] for geom_id in robot_geom_ids
            ],
        },
        "full_robot_surface_sampling": {
            "sample_count": full_sample_count,
            "sample_ledger_sha256": "a" * 64,
            "epsilon_m": 0.05,
            "maximum_surface_cover_radius_m": 0.04931058286977641,
            "coverage_semantics": (
                "strict_open_ball_surface_cover_from_triangle_lattices_for_compiled_"
                "convex_hulls_and_exact_boxes_or_analytic_parameter_grids_for_exact_"
                "cylinders; MuJoCo collision-semantic equivalence requires allocation audit"
            ),
            "geom_records": surface_records,
            "roundtrip": {
                "sample_count": full_sample_count,
                "maximum_roundtrip_error_m": 1e-16,
                "tolerance_m": 1e-10,
                "passed": True,
            },
        },
        "outcome": outcome,
    }

    parameter_block = _registered_runtime_parameter_block()
    payload["runtime"]["protocol_parameter_block_sha256"] = _producer_sha256(
        parameter_block
    )
    protected_samples = [
        {
            "sample_id": 0,
            "body_id": 55,
            "body_name": "robot0_link5",
            "geom_id": 55,
            "geom_name": "robot_geom_55",
            "point_body_local_m": [0.0, 0.0, 0.0],
            "source": "collision_geom_surface",
        },
        {
            "sample_id": 1,
            "body_id": 56,
            "body_name": "robot0_link6",
            "geom_id": 56,
            "geom_name": "robot_geom_56",
            "point_body_local_m": [0.0, 0.01, 0.0],
            "source": "collision_geom_surface",
        },
    ]
    protected_sampling = {
        "epsilon_m": 0.05,
        "maximum_surface_cover_radius_m": 0.01,
        "coverage_semantics": "fixture_strict_open_ball_collision_surface_cover",
        "components": [
            {
                "geom_id": 55,
                "geom_name": "robot_geom_55",
                "body_id": 55,
                "body_name": "robot0_link5",
                "geometry_kind": "compiled_collision_geom",
                "certificate_kind": "deterministic_fixture_cover",
                "surface_element_count": 1,
                "sample_count": 1,
                "certified_surface_cover_radius_m": 0.01,
            },
            {
                "geom_id": 56,
                "geom_name": "robot_geom_56",
                "body_id": 56,
                "body_name": "robot0_link6",
                "geometry_kind": "compiled_collision_geom",
                "certificate_kind": "deterministic_fixture_cover",
                "surface_element_count": 1,
                "sample_count": 1,
                "certified_surface_cover_radius_m": 0.01,
            },
        ],
        "samples": protected_samples,
    }
    arm_dof_indices = list(range(7))
    differential_audit, differential_summary = (
        _synthetic_protected_sample_differential_audit(
            protected_samples,
            arm_dof_indices,
            outcome["restore"]["official_integration_state_sha256"],
            parameter_block["differential_audit"],
        )
    )
    trace.update(
        {
            "runtime_protocol_parameter_block": parameter_block,
            "protected_link_surface_sampling": protected_sampling,
            "arm_dof_indices": arm_dof_indices,
            "settled_link56_differential_audit": differential_audit,
            "settled_link56_differential_audit_validation": differential_summary,
        }
    )

    source_hash = sha256_bytes(canonical_json_bytes(entered_actions))
    payload["pairing"]["nominal_high_level_action_ledger_sha256"] = source_hash
    source_replay = {
        "case_id": payload["case_id"],
        "source_arm": "pi05_plus_aegis_translational",
        "action_count": 2,
        "historical_result_payload_sha256": "1" * 64,
        "historical_result_file_sha256": "2" * 64,
        "executed_sequence_sha256": source_hash,
        "source_policy_query_count": payload["runtime"]["sampler"][
            "source_policy_query_count"
        ],
        "source_policy_query_schedule_sha256": payload["pairing"][
            "policy_query_schedule_sha256"
        ],
        "settled_simulator_state_sha256": "f" * 64,
        "initial_observation_sha256": payload["pairing"][
            "source_settled_observation_sha256"
        ],
        "settled_active_obstacle_position_sha256": trace["outcome"][
            "paper_car_position_ledger"
        ]["historical_settled_active_obstacle_position_sha256"],
        "policy_noise_schedule_sha256": payload["pairing"][
            "policy_noise_schedule_sha256"
        ],
        "historical_task_success": True,
        "historical_car_collision": True,
        "historical_collision_first_step": 1,
        "terminal_simulator_state_sha256": "4" * 64,
        "replay_semantics": "exact_actions[*].executed_not_nominal_raw",
    }
    trace["source_replay"] = source_replay
    trace["field_bundle_hashes"] = {
        "protocol_sha256": payload["provenance"][
            "runtime_protocol_semantic_sha256"
        ],
        "parameter_block_sha256": payload["runtime"][
            "protocol_parameter_block_sha256"
        ],
        "obstacle_geometry_sha256": "5" * 64,
        "occupancy_sha256": "6" * 64,
        "domain_sha256": "7" * 64,
        "system_sha256": "8" * 64,
        "field_sha256": payload["pairing"]["field_sha256"],
        "protected_samples_sha256": sha256_bytes(
            canonical_json_bytes(protected_sampling)
        ),
        "bundle_sha256": "a" * 64,
    }
    trace["field_bundle_diagnostics"] = {
        "obstacle_geom_count": len(trace["resolved_geometry"]["obstacle_geom_ids"]),
        "protected_surface_component_count": len(
            trace["resolved_geometry"]["link56_geom_ids"]
        ),
        "protected_sample_count": trace["outcome"][
            "initial_protected_sample_audit"
        ]["protected_sample_count"],
        "raw_occupied_cell_count": 10,
        "buffered_occupied_cell_count": 20,
        "connected_free_cell_count": 30,
        "active_vertex_count": 40,
        "boundary_vertex_count": 10,
        "interior_vertex_count": 30,
        "poisson_method": "red_black_sor",
        "poisson_iterations": 10,
        "poisson": {
            "finite": True,
            "unknown_count": 30,
            "residual_linf": 1e-10,
            "residual_l2": 1e-10,
            "relative_residual_linf": 1e-10,
            "relative_residual_l2": 1e-10,
            "backward_error_linf": 1e-10,
            "boundary_linf": 0.0,
            "interior_min": 0.004,
            "interior_max": 0.01,
            "expected_sign": "nonnegative",
            "sign_violation_count": 0,
            "sign_violation_linf": 0.0,
            "passed": True,
        },
        "minimum_initial_h_m2": trace["outcome"][
            "initial_protected_sample_audit"
        ]["minimum_h_m2"],
        "minimum_outer_boundary_clearance_m": 0.1,
        "required_outer_boundary_clearance_m": 0.051,
    }
    bundle_identity = {
        key: value
        for key, value in trace["field_bundle_hashes"].items()
        if key != "bundle_sha256"
    }
    bundle_identity["diagnostics"] = trace["field_bundle_diagnostics"]
    trace["field_bundle_hashes"]["bundle_sha256"] = sha256_bytes(
        canonical_json_bytes(bundle_identity)
    )
    pairing_key = {
        "pair_group_id": payload["pairing"]["pair_group_id"],
        "official_settled_state": trace["outcome"]["restore"][
            "official_integration_state_sha256"
        ],
        "historical_action_ledger": source_replay["executed_sequence_sha256"],
        "field_sha256": trace["field_bundle_hashes"]["field_sha256"],
        "source_observation_sha256": source_replay[
            "initial_observation_sha256"
        ],
    }
    payload["pairing"]["pairing_key_sha256"] = _producer_sha256(pairing_key)
    payload["pairing"]["controller_initial_state_sha256"] = controller_state[
        "sha256"
    ]
    payload["pairing"]["settling_action_ledger_sha256"] = _producer_sha256(
        [[0.0] * 7] * 20
    )
    payload["runtime"]["controller"][
        "configuration_sha256"
    ] = _producer_sha256(controller_contract)
    payload["runtime"]["intervention"][
        "configuration_sha256"
    ] = _producer_sha256(
        {
            "arm": payload["arm"],
            "field_sha256": trace["field_bundle_hashes"]["field_sha256"],
            "protected": ["robot0_link5", "robot0_link6"],
        }
    )
    payload["optimizer"]["configuration_sha256"] = _producer_sha256(
        {
            "runtime_protocol_parameter_block_sha256": payload["runtime"][
                "protocol_parameter_block_sha256"
            ],
            "arm": payload["arm"],
            "solver": "osqp",
        }
    )
    payload["runtime"]["measurement"]["configuration_sha256"] = sha256_bytes(
        canonical_json_bytes(
            {
                "resolved_geometry": trace["resolved_geometry"],
                "D_sim_semantics": payload["runtime"]["measurement"][
                    "D_sim_semantics"
                ],
            }
        )
    )
    payload["runtime"]["sampler"]["configuration_sha256"] = sha256_bytes(
        canonical_json_bytes(
            {
                "source": source_replay,
                "active_policy_queries": 0,
                "active_policy_rng_exercised": False,
            }
        )
    )
    payload["execution"].update(
        {
            "planned_source_action_ledger_sha256": source_hash,
            "entered_source_action_prefix_sha256": source_hash,
            "completed_high_level_source_action_prefix_sha256": source_hash,
            "nominal_joint_velocity_ledger_sha256": sha256_bytes(
                canonical_json_bytes(nominal_ledger)
            ),
            "executed_controller_action_ledger_sha256": sha256_bytes(
                canonical_json_bytes(executed_ledger)
            ),
        }
    )
    payload["endpoints"]["task"] = copy.deepcopy(task)
    payload["endpoints"]["usefulness"] = copy.deepcopy(usefulness)
    payload["endpoints"]["validity"] = copy.deepcopy(validity)
    payload["endpoints"]["safety"]["paper_car"] = copy.deepcopy(car)
    return payload, trace


def _partial_tracking_failure_fixture():
    """Return the scientifically important fail-closed, right-censored shape."""

    payload, trace = _fixture()
    outcome = trace["outcome"]
    measured = [0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    error = measured[0] - 0.08
    rmse = math.sqrt(error * error / 7.0)
    crossing = {
        "high_level_index": 0,
        "inner_control_index": 0,
        "physics_substep_index": 0,
        "error_linf_rad_s": error,
        "cumulative_rmse_rad_s": rmse,
        "linf_threshold_rad_s": 0.05,
        "rmse_threshold_rad_s": 0.02,
        "command_rad_s": [0.08, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "measured_rad_s": measured,
    }
    outcome["completion_class"] = "controller_tracking_invalid"
    outcome["terminal_reason"] = "runtime_joint_velocity_tracking_threshold_crossed"
    outcome["exposure_complete"] = False
    outcome["prefix"].update(
        {
            "high_level_steps": 1,
            "inner_control_steps": 1,
            "physics_substeps": 1,
            "physics_exposure_seconds": 0.002,
        }
    )
    outcome["physics_clock"].update(
        {
            "terminal_time_s": 10.002,
            "observed_exposure_s": 0.002,
            "expected_from_completed_substeps_s": 0.002,
        }
    )
    outcome["entered_source_actions"] = outcome["entered_source_actions"][:1]
    outcome["completed_source_actions"] = []
    outcome["nominal_ledger"] = outcome["nominal_ledger"][:1]
    outcome["executed_ledger"] = outcome["executed_ledger"][:1]
    outcome["inner_trace"] = outcome["inner_trace"][:1]
    outcome["physics_trace"] = outcome["physics_trace"][:1]
    outcome["physics_trace"][0]["measured_arm_joint_velocity_rad_s"] = measured
    outcome["realized_cbf_trace"] = outcome["realized_cbf_trace"][:1]
    outcome["goal_progress_ledger"] = [
        _snapshot(-1, False, None, "settled_pre_action"),
        _snapshot(0, False, False, "partial_prefix_terminal_state_diagnostic"),
    ]
    task = outcome["task"]
    task.update(
        {
            "fixed_exposure_available": False,
            "ever_task_success_within_registered_source_exposure": None,
            "terminal_task_success_within_registered_source_exposure": None,
            "first_task_success_high_level_index": None,
            "prefix_task_success_latched": False,
            "prefix_first_task_success_high_level_index": None,
            "prefix_terminal_task_success": False,
            "terminal_goal_fraction": 0.0,
            "maximum_goal_fraction": 0.0,
            "goal_regression_count": 0,
            "unavailable_reason": "fixed_exposure_incomplete",
        }
    )
    usefulness = outcome["usefulness"]
    usefulness.update(
        {
            "nominal_joint_motion_integral_rad": 0.0002,
            "safe_joint_motion_integral_rad": 0.00016,
            "measured_joint_motion_integral_rad": 0.0004,
            "eef_path_length_m": 0.001,
            "correction_integral_rad": 0.00004,
            "motion_retention_ratio": 0.8,
            "zero_motion_fraction": 0.0,
        }
    )
    outcome["tracking"].update(
        {
            "command_count": 1,
            "verified_terminal_command_count": 0,
            "observed_physics_substep_count": 1,
            "maximum_linf_error_rad_s": error,
            "cumulative_rmse_rad_s": rmse,
            "final_fifth_command_verified": False,
            "first_threshold_crossing": crossing,
        }
    )
    validity = outcome["validity"]
    validity.update(
        {
            "realized_cbf_observed_physics_substep_count": 1,
            "realized_cbf_residual_evaluation_count": 2,
            "maximum_velocity_tracking_error_rad_s": error,
            "velocity_tracking_rmse_rad_s": rmse,
            "velocity_tracking_observed_physics_substep_count": 1,
            "first_velocity_tracking_threshold_crossing": crossing,
        }
    )
    outcome["optimizer_counts"].update({"attempt_count": 1, "solved_count": 1})
    outcome["realized_cbf_audit"].update(
        {"observed_physics_substep_count": 1, "residual_evaluation_count": 2}
    )
    outcome["eef_path_audit"]["position_sample_count"] = 2
    outcome["paper_car_position_ledger"]["completed_post_step_positions"] = []
    car_ledger = outcome["paper_car_position_ledger"]
    outcome["paper_car"].update(
        {
            "available": False,
            "maximum_active_obstacle_l1_displacement_m": 0.0,
            "collision": None,
            "avoidance": None,
            "reason": "fixed_exposure_incomplete",
            "position_ledger_sha256": sha256_bytes(canonical_json_bytes(car_ledger)),
        }
    )
    outcome["monitor"]["record"]["observed_physics_substeps"] = 1
    outcome["monitor"]["record"]["last_index"] = [0, 0, 0]
    for geom_drift in outcome["monitor"]["record"]["obstacle_pose_drift"][
        "geoms"
    ]:
        geom_drift["maximum_translation_observation_index"] = 0
        geom_drift["maximum_rotation_observation_index"] = 0
        geom_drift["maximum_surface_displacement_observation_index"] = 0

    payload["completion_class"] = "controller_tracking_invalid"
    payload["execution"].update(
        {
            "high_level_steps": 1,
            "inner_control_steps": 1,
            "physics_substeps": 1,
            "physics_exposure_seconds": 0.002,
            "exposure_complete": False,
            "fail_closed_no_further_physics": True,
            "terminal_reason": outcome["terminal_reason"],
            "completed_high_level_steps": 0,
            "terminal_entered_high_level_index": 0,
            "terminal_entered_inner_control_index": 0,
            "completed_physics_substeps_in_terminal_inner_control": 1,
            "entered_source_action_prefix_sha256": sha256_bytes(
                canonical_json_bytes(outcome["entered_source_actions"])
            ),
            "completed_high_level_source_action_prefix_sha256": sha256_bytes(
                canonical_json_bytes([])
            ),
            "nominal_joint_velocity_ledger_sha256": sha256_bytes(
                canonical_json_bytes(outcome["nominal_ledger"])
            ),
            "executed_controller_action_ledger_sha256": sha256_bytes(
                canonical_json_bytes(outcome["executed_ledger"])
            ),
        }
    )
    payload["optimizer"].update({"attempt_count": 1, "solved_count": 1})
    payload["endpoints"]["task"] = copy.deepcopy(task)
    payload["endpoints"]["usefulness"] = copy.deepcopy(usefulness)
    payload["endpoints"]["validity"] = copy.deepcopy(validity)
    safety = payload["endpoints"]["safety"]
    safety["paper_car"] = copy.deepcopy(outcome["paper_car"])
    safety["contact_observation_complete"] = False
    safety["contact_absence_right_censored"] = True
    return payload, trace


def _publish(root, payload, trace):
    arm_directory = root / "arms" / payload["arm"]
    arm_directory.mkdir(parents=True)
    trace_path = arm_directory / "trace.json"
    publish_hashed_json(trace_path, trace)
    payload = copy.deepcopy(payload)
    payload["artifact_references"] = [
        {
            "relative_path": str(trace_path.relative_to(root)),
            "bytes": trace_path.stat().st_size,
            "sha256": sha256_file(trace_path),
            "artifact_type": "active_arm_audit_trace",
            "media_type": "application/json",
        }
    ]
    result_path = arm_directory / "result.json"
    publish_hashed_json(result_path, payload)
    return result_path, trace_path


def _rehash_differential_audit(trace):
    audit = trace["settled_link56_differential_audit"]
    audit.pop("audit_payload_sha256", None)
    audit["audit_payload_sha256"] = sha256_bytes(canonical_json_bytes(audit))


def _rehash_field_bundle(trace):
    identity = {
        key: value
        for key, value in trace["field_bundle_hashes"].items()
        if key != "bundle_sha256"
    }
    identity["diagnostics"] = trace["field_bundle_diagnostics"]
    trace["field_bundle_hashes"]["bundle_sha256"] = sha256_bytes(
        canonical_json_bytes(identity)
    )


class PoissonTraceArtifactValidationTest(unittest.TestCase):
    def test_active_prerequisite_accepts_registered_sampler_evidence(self):
        from scripts.run_poisson_active_canary import (
            _require_full_robot_sampling_evidence,
        )

        _, trace = _fixture()
        _require_full_robot_sampling_evidence(
            trace["resolved_geometry"],
            trace["full_robot_surface_sampling"],
            roundtrip_field="roundtrip",
        )

    def test_complete_trace_reconstructs_all_scientific_endpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result_path, _ = _publish(root, *_fixture())
            result = validate_run_artifacts(result_path, root)
            self.assertEqual(result["case_id"], "vlsa-t1-goal-ii-t0-e05")
            inferred = validate_run_artifacts(result_path)
            self.assertEqual(inferred["result_payload_sha256"], result["result_payload_sha256"])

    def test_legacy_v2_trace_is_rejected(self):
        payload, trace = _fixture()
        trace["schema_version"] = "vlsa_poisson_active_arm_trace.v2"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result_path, _ = _publish(root, payload, trace)
            with self.assertRaisesRegex(ArtifactContractError, "trace.schema_version"):
                validate_run_artifacts(result_path, root)

    def test_v4_trace_requires_every_differential_authority(self):
        fields = (
            "runtime_protocol_parameter_block",
            "protected_link_surface_sampling",
            "arm_dof_indices",
            "settled_link56_differential_audit",
            "settled_link56_differential_audit_validation",
        )
        for field in fields:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                payload, trace = _fixture()
                trace.pop(field)
                root = Path(directory)
                result_path, _ = _publish(root, payload, trace)
                with self.assertRaises(ArtifactContractError):
                    validate_run_artifacts(result_path, root)

    def test_differential_audit_matrix_tamper_is_rejected_after_rehash(self):
        payload, trace = _fixture()
        trace["settled_link56_differential_audit"]["sample_records"][0][
            "analytic_point_jacobian_m_per_rad_3x7"
        ][0][0] += 0.25
        _rehash_differential_audit(trace)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result_path, _ = _publish(root, payload, trace)
            with self.assertRaisesRegex(
                ArtifactContractError,
                "settled_link56_differential_audit",
            ):
                validate_run_artifacts(result_path, root)

    def test_differential_audit_sample_reorder_is_rejected_after_rehash(self):
        payload, trace = _fixture()
        samples = trace["protected_link_surface_sampling"]["samples"]
        samples.reverse()
        for sample_id, sample in enumerate(samples):
            sample["sample_id"] = sample_id
        trace["field_bundle_hashes"]["protected_samples_sha256"] = sha256_bytes(
            canonical_json_bytes(trace["protected_link_surface_sampling"])
        )
        _rehash_field_bundle(trace)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result_path, _ = _publish(root, payload, trace)
            with self.assertRaisesRegex(
                ArtifactContractError,
                "settled_link56_differential_audit",
            ):
                validate_run_artifacts(result_path, root)

    def test_differential_audit_config_tamper_is_rejected_after_rehash(self):
        payload, trace = _fixture()
        trace["settled_link56_differential_audit"]["differential_audit_config"][
            "point_jacobian_delta_rad"
        ] *= 2.0
        _rehash_differential_audit(trace)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result_path, _ = _publish(root, payload, trace)
            with self.assertRaisesRegex(
                ArtifactContractError,
                "settled_link56_differential_audit",
            ):
                validate_run_artifacts(result_path, root)

    def test_differential_audit_state_tamper_is_rejected_after_rehash(self):
        payload, trace = _fixture()
        state = trace["settled_link56_differential_audit"]["integration_state"]
        for field in (
            "source_initial_sha256",
            "clone_base_sha256",
            "clone_final_sha256",
            "source_final_sha256",
        ):
            state[field] = "0" * 64
        _rehash_differential_audit(trace)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result_path, _ = _publish(root, payload, trace)
            with self.assertRaisesRegex(
                ArtifactContractError,
                "settled_link56_differential_audit",
            ):
                validate_run_artifacts(result_path, root)

    def test_differential_audit_serialized_summary_is_reconstructed(self):
        payload, trace = _fixture()
        trace["settled_link56_differential_audit_validation"]["counts"][
            "passed_sample_count"
        ] = 1
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result_path, _ = _publish(root, payload, trace)
            with self.assertRaisesRegex(
                ArtifactContractError,
                "differs from independent reconstruction",
            ):
                validate_run_artifacts(result_path, root)

    def test_right_censored_tracking_failure_reconstructs_exact_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result_path, _ = _publish(root, *_partial_tracking_failure_fixture())
            result = validate_run_artifacts(result_path, root)
            self.assertEqual(result["completion_class"], "controller_tracking_invalid")
            self.assertEqual(result["execution"]["physics_substeps"], 1)

    def test_full_robot_sampler_must_match_authoritative_geom_resolution(self):
        payload, trace = _fixture()
        trace["full_robot_surface_sampling"]["geom_records"][0]["geom_id"] = 54
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result_path, _ = _publish(root, payload, trace)
            with self.assertRaisesRegex(
                ArtifactContractError, "geom IDs differ from authoritative resolution"
            ):
                validate_run_artifacts(result_path, root)

    def test_cylinder_cover_is_recomputed_from_exact_grid_and_size(self):
        payload, trace = _fixture()
        cylinder = trace["full_robot_surface_sampling"]["geom_records"][-1]
        cylinder["certified_surface_cover_radius_m"] = 0.04
        trace["full_robot_surface_sampling"][
            "maximum_surface_cover_radius_m"
        ] = 0.04
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result_path, _ = _publish(root, payload, trace)
            with self.assertRaisesRegex(
                ArtifactContractError, "reconstructed"
            ):
                validate_run_artifacts(result_path, root)

    def test_collusive_endpoint_tampering_is_rejected_against_raw_ledgers(self):
        mutations = {
            "motion": lambda payload, trace: (
                trace["outcome"]["usefulness"].__setitem__(
                    "nominal_joint_motion_integral_rad", 0.25
                ),
                payload["endpoints"]["usefulness"].__setitem__(
                    "nominal_joint_motion_integral_rad", 0.25
                ),
            ),
            "car": lambda payload, trace: (
                trace["outcome"]["paper_car"].__setitem__(
                    "maximum_active_obstacle_l1_displacement_m", 0.0005
                ),
                payload["endpoints"]["safety"]["paper_car"].__setitem__(
                    "maximum_active_obstacle_l1_displacement_m", 0.0005
                ),
            ),
            "task": lambda payload, trace: (
                trace["outcome"]["task"].__setitem__("goal_regression_count", 1),
                payload["endpoints"]["task"].__setitem__("goal_regression_count", 1),
            ),
            "optimizer": lambda payload, trace: (
                trace["outcome"]["optimizer_counts"].__setitem__("attempt_count", 11),
                payload["optimizer"].__setitem__("attempt_count", 11),
            ),
            "contact": lambda payload, trace: (
                trace["outcome"]["monitor"].__setitem__("total", 1),
                payload["endpoints"]["safety"].__setitem__(
                    "total_physical_contact_point_record_count", 1
                ),
            ),
            "realized": lambda payload, trace: (
                trace["outcome"]["realized_cbf_audit"].__setitem__(
                    "residual_evaluation_count", 99
                ),
                trace["outcome"]["validity"].__setitem__(
                    "realized_cbf_residual_evaluation_count", 99
                ),
                payload["endpoints"]["validity"].__setitem__(
                    "realized_cbf_residual_evaluation_count", 99
                ),
            ),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                payload, trace = _fixture()
                mutate(payload, trace)
                result_path, _ = _publish(Path(directory), payload, trace)
                with self.assertRaises(ArtifactContractError):
                    validate_run_artifacts(result_path, Path(directory))

    def test_registered_canary_authorities_reject_rehashed_claim_mutations(self):
        def wrong_obstacle(payload, trace):
            resolved = trace["resolved_geometry"]
            resolved["obstacle_body_names"] = ["bottle_1_main"]
            resolved["obstacle_geom_names"] = ["bottle_1_g0"]
            payload["runtime"]["measurement"][
                "configuration_sha256"
            ] = _producer_sha256(
                {
                    "resolved_geometry": resolved,
                    "D_sim_semantics": payload["runtime"]["measurement"][
                        "D_sim_semantics"
                    ],
                }
            )

        def wrong_goal(payload, trace):
            definition = trace["outcome"]["goal_definition"]
            definition["goal_atoms"][0]["arguments"] = [
                "wrong_object",
                "wrong_target",
            ]
            identity = {
                key: value
                for key, value in definition.items()
                if key != "goal_definition_sha256"
            }
            definition["goal_definition_sha256"] = sha256_bytes(
                canonical_json_bytes(identity)
            )

        def omit_link56_contact_pairs(payload, trace):
            resolved = trace["resolved_geometry"]
            link_ids = set(resolved["link56_geom_ids"])
            resolved["collision_enabled_pairs"] = [
                pair
                for pair in resolved["collision_enabled_pairs"]
                if pair[0] not in link_ids
            ]
            payload["runtime"]["measurement"][
                "configuration_sha256"
            ] = _producer_sha256(
                {
                    "resolved_geometry": resolved,
                    "D_sim_semantics": payload["runtime"]["measurement"][
                        "D_sim_semantics"
                    ],
                }
            )

        def omit_shifted_robot_contact_pairs(payload, trace):
            resolved = trace["resolved_geometry"]
            victim = next(
                geom_id
                for geom_id in resolved["robot_geom_ids"]
                if geom_id not in set(resolved["link56_geom_ids"])
            )
            resolved["collision_enabled_pairs"] = [
                pair
                for pair in resolved["collision_enabled_pairs"]
                if pair[0] != victim
            ]
            payload["runtime"]["measurement"][
                "configuration_sha256"
            ] = _producer_sha256(
                {
                    "resolved_geometry": resolved,
                    "D_sim_semantics": payload["runtime"]["measurement"][
                        "D_sim_semantics"
                    ],
                }
            )

        mutations = {
            "settled_position_vector": lambda payload, trace: trace["outcome"][
                "paper_car_position_ledger"
            ].__setitem__(
                "settled_active_obstacle_root_position_m", [1.0, 0.0, 0.0]
            ),
            "restore_tolerance": lambda payload, trace: trace["outcome"][
                "restore"
            ].update(
                {
                    "maximum_arm_qpos_error_rad": 0.5,
                    "maximum_arm_qvel_error_rad_s": 0.5,
                    "max_arm_qpos_error_tolerance_rad": 1.0,
                    "max_arm_qvel_error_tolerance_rad_s": 1.0,
                }
            ),
            "restore_clock": lambda payload, trace: trace["outcome"][
                "restore"
            ].__setitem__("copied_cur_time_s", 0.0),
            "eef_site": lambda payload, trace: trace["outcome"][
                "eef_reference_site"
            ].__setitem__("site_name", "wrong_camera_site"),
            "tracking_thresholds": lambda payload, trace: trace["outcome"][
                "tracking"
            ].update(
                {
                    "maximum_linf_threshold_rad_s": 100.0,
                    "maximum_rmse_threshold_rad_s": 100.0,
                }
            ),
            "negative_D_opt": lambda payload, trace: trace["outcome"][
                "inner_trace"
            ][0].__setitem__("D_opt_min_m", -0.1),
            "nonpositive_provider_h": lambda payload, trace: trace["outcome"][
                "inner_trace"
            ][0].__setitem__("minimum_h_m2", -0.1),
            "contradictory_qp_status": lambda payload, trace: trace["outcome"][
                "inner_trace"
            ][0]["qp"].update(
                {
                    "solver": "bogus",
                    "status": "primal infeasible",
                    "status_value": 3,
                }
            ),
            "missing_scheduled_qp": lambda payload, trace: trace["outcome"][
                "inner_trace"
            ][0].__setitem__("qp", None),
            "wrong_executed_gripper": lambda payload, trace: trace["outcome"][
                "inner_trace"
            ][0]["execution"]["normalized_executed_action"].__setitem__(7, 0.7),
            "executed_with_failure_record": lambda payload, trace: trace[
                "outcome"
            ]["fail_closed_attempt_trace"].append(
                {
                    "high_level_index": 0,
                    "inner_control_index": 0,
                    "completion_class": "inexplicable_failure",
                    "physics_executed_after_attempt": False,
                }
            ),
            "wrong_obstacle": wrong_obstacle,
            "wrong_goal": wrong_goal,
            "missing_link56_contact_pairs": omit_link56_contact_pairs,
            "missing_shifted_robot_contact_pairs": (
                omit_shifted_robot_contact_pairs
            ),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                payload, trace = _fixture()
                mutate(payload, trace)
                result_path, _ = _publish(Path(directory), payload, trace)
                with self.assertRaises(ArtifactContractError):
                    validate_run_artifacts(result_path, Path(directory))

    def test_translation_only_source_rotation_cannot_be_hidden_by_rehashing(self):
        payload, trace = _fixture()
        outcome = trace["outcome"]
        outcome["entered_source_actions"][1][3] = 0.5
        outcome["completed_source_actions"][1][3] = 0.5
        for row in outcome["inner_trace"]:
            if row["high_level_index"] == 1:
                row["source_action"][3] = 0.5
        source_hash = sha256_bytes(
            canonical_json_bytes(outcome["entered_source_actions"])
        )
        payload["execution"]["entered_source_action_prefix_sha256"] = source_hash
        payload["execution"][
            "completed_high_level_source_action_prefix_sha256"
        ] = source_hash
        payload["execution"]["planned_source_action_ledger_sha256"] = source_hash
        payload["pairing"]["nominal_high_level_action_ledger_sha256"] = source_hash
        trace["source_replay"]["executed_sequence_sha256"] = source_hash
        payload["runtime"]["sampler"]["configuration_sha256"] = _producer_sha256(
            {
                "source": trace["source_replay"],
                "active_policy_queries": 0,
                "active_policy_rng_exercised": False,
            }
        )
        pairing_key = {
            "pair_group_id": payload["pairing"]["pair_group_id"],
            "official_settled_state": outcome["restore"][
                "official_integration_state_sha256"
            ],
            "historical_action_ledger": source_hash,
            "field_sha256": trace["field_bundle_hashes"]["field_sha256"],
            "source_observation_sha256": trace["source_replay"][
                "initial_observation_sha256"
            ],
        }
        payload["pairing"]["pairing_key_sha256"] = _producer_sha256(pairing_key)
        with tempfile.TemporaryDirectory() as directory:
            result_path, _ = _publish(Path(directory), payload, trace)
            with self.assertRaisesRegex(ArtifactContractError, "rotation intent"):
                validate_run_artifacts(result_path, Path(directory))

    def test_production_default_requires_the_exact_237_action_exposure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result_path, _ = _publish(root, *_fixture())
            with self.assertRaisesRegex(
                ArtifactContractError, "registered source exposure"
            ):
                _validate_run_artifacts(result_path, root)

    def test_nominal_and_filtered_commands_obey_registered_velocity_limit(self):
        payload, trace = _fixture()
        outcome = trace["outcome"]
        outcome["nominal_ledger"][0]["qdot_rad_s"][0] = 0.500001
        inner = outcome["inner_trace"][0]
        inner["adapter"]["qdot_physical_rad_s"][0] = 0.500001
        inner["adapter"]["normalized_action"][0] = 0.500001 / 0.5
        inner["execution"]["nominal_qdot_physical_rad_s"][0] = 0.500001
        correction = [
            safe - nominal
            for safe, nominal in zip(
                outcome["executed_ledger"][0]["qdot_rad_s"],
                outcome["nominal_ledger"][0]["qdot_rad_s"],
            )
        ]
        inner["execution"]["filter_correction_rad_s"] = correction
        inner["execution"]["filter_correction_l2_rad_s"] = math.sqrt(
            sum(value * value for value in correction)
        )
        inner["qp"]["correction_l2_rad_s"] = inner["execution"][
            "filter_correction_l2_rad_s"
        ]
        trace["outcome"]["physics_trace"][0][
            "nominal_joint_velocity_command_rad_s"
        ][0] = 0.500001
        payload["execution"][
            "nominal_joint_velocity_ledger_sha256"
        ] = sha256_bytes(
            canonical_json_bytes(trace["outcome"]["nominal_ledger"])
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result_path, _ = _publish(root, payload, trace)
            with self.assertRaisesRegex(
                ArtifactContractError, "registered joint-velocity limit"
            ):
                validate_run_artifacts(result_path, root)

    def test_initial_state_model_and_field_identities_are_trace_bound(self):
        mutations = {
            "initial_state_record": lambda payload, trace: payload[
                "case_identity"
            ].__setitem__("initial_state_record_sha256", "0" * 64),
            "active_initial_observation": lambda payload, trace: payload[
                "pairing"
            ].__setitem__(
                "active_joint_velocity_initial_observation_sha256", "0" * 64
            ),
            "compiled_physical_model": lambda payload, trace: payload[
                "pairing"
            ].__setitem__("compiled_physical_model_sha256", "0" * 64),
            "compiled_mjb": lambda payload, trace: payload["pairing"].__setitem__(
                "compiled_mjb_sha256", "0" * 64
            ),
            "controller_initial_state": lambda payload, trace: payload[
                "pairing"
            ].__setitem__("controller_initial_state_sha256", "0" * 64),
            "pairing_key": lambda payload, trace: payload["pairing"].__setitem__(
                "pairing_key_sha256", "0" * 64
            ),
            "settling_action_ledger": lambda payload, trace: payload[
                "pairing"
            ].__setitem__("settling_action_ledger_sha256", "0" * 64),
            "controller_configuration": lambda payload, trace: payload[
                "runtime"
            ]["controller"].__setitem__("configuration_sha256", "0" * 64),
            "intervention_configuration": lambda payload, trace: payload[
                "runtime"
            ]["intervention"].__setitem__("configuration_sha256", "0" * 64),
            "optimizer_configuration": lambda payload, trace: payload[
                "optimizer"
            ].__setitem__("configuration_sha256", "0" * 64),
            "controller_state_payload": lambda payload, trace: trace["outcome"][
                "restore"
            ]["controller_software_state"]["fields"].update({"forged": True}),
            "field_bundle_diagnostics": lambda payload, trace: trace[
                "field_bundle_diagnostics"
            ].update({"forged": True}),
            "settled_obstacle_pairing": _forge_source_settled_obstacle_hash,
            "restore_exactness": lambda payload, trace: trace["outcome"][
                "restore"
            ].__setitem__("exact_flattened_state", False),
            "settled_motion": lambda payload, trace: trace["outcome"][
                "monitor"
            ]["record"]["settled_state"][
                "obstacle_motion_admissibility"
            ].__setitem__("admissible", False),
            "eef_jacobian": lambda payload, trace: trace["outcome"][
                "eef_reference_site"
            ].__setitem__("passed", False),
            "coverage_audit": lambda payload, trace: (
                trace["outcome"]["validity"].__setitem__(
                    "coverage_audit_passed", False
                ),
                payload["endpoints"]["validity"].__setitem__(
                    "coverage_audit_passed", False
                ),
            ),
            "static_drift": _forge_static_drift_as_admissible,
            "poisson_diagnostics": _forge_failed_poisson_diagnostics,
            "realized_query_population": _forge_realized_query_population,
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                payload, trace = _fixture()
                mutate(payload, trace)
                result_path, _ = _publish(Path(directory), payload, trace)
                with self.assertRaises(ArtifactContractError):
                    validate_run_artifacts(result_path, Path(directory))

    def test_nonpositive_candidate_cannot_be_hidden_in_any_contact_phase(self):
        phases = (
            "settled_post_integration_recomputed",
            "live_solver_phase_preintegration_geometry",
            "post_integration_recomputed",
        )
        for phase in phases:
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as directory:
                payload, trace = _fixture()
                _add_candidate_without_physical_summary(
                    trace,
                    phase,
                    physical_flag=False,
                )
                result_path, _ = _publish(Path(directory), payload, trace)
                with self.assertRaisesRegex(
                    ArtifactContractError,
                    "differs from MuJoCo nonpositive distance",
                ):
                    validate_run_artifacts(result_path, Path(directory))

    def test_contact_distance_requires_an_actual_json_number(self):
        for phase in (
            "settled_post_integration_recomputed",
            "live_solver_phase_preintegration_geometry",
            "post_integration_recomputed",
        ):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as directory:
                payload, trace = _fixture()
                record = _add_candidate_without_physical_summary(
                    trace,
                    phase,
                    physical_flag=False,
                    distance_m=0.0005,
                )
                record["contact_distance_m"] = "0.0005"
                result_path, _ = _publish(Path(directory), payload, trace)
                with self.assertRaisesRegex(
                    ArtifactContractError, "finite JSON number"
                ):
                    validate_run_artifacts(result_path, Path(directory))

    def test_positive_candidate_cannot_claim_physical_contact(self):
        payload, trace = _fixture()
        _add_candidate_without_physical_summary(
            trace,
            "live_solver_phase_preintegration_geometry",
            physical_flag=True,
            distance_m=0.0005,
        )
        with tempfile.TemporaryDirectory() as directory:
            result_path, _ = _publish(Path(directory), payload, trace)
            with self.assertRaisesRegex(
                ArtifactContractError, "differs from MuJoCo nonpositive distance"
            ):
                validate_run_artifacts(result_path, Path(directory))

    def test_link56_authority_cannot_be_relabelled_or_omitted(self):
        mutations = {
            "relabel_pedestal": lambda resolved: resolved.update(
                {
                    "link56_body_ids": [84, 84],
                    "link56_geom_ids": [84],
                    "link56_geom_names": ["mount0_pedestal_col"],
                }
            ),
            "omit_link6": lambda resolved: resolved.update(
                {
                    "link56_geom_ids": [55],
                    "link56_geom_names": ["robot_geom_55"],
                }
            ),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                payload, trace = _fixture()
                mutate(trace["resolved_geometry"])
                payload["runtime"]["measurement"][
                    "configuration_sha256"
                ] = sha256_bytes(
                    canonical_json_bytes(
                        {
                            "resolved_geometry": trace["resolved_geometry"],
                            "D_sim_semantics": payload["runtime"]["measurement"][
                                "D_sim_semantics"
                            ],
                        }
                    )
                )
                result_path, _ = _publish(Path(directory), payload, trace)
                with self.assertRaisesRegex(
                    ArtifactContractError, "link56|protected_surface"
                ):
                    validate_run_artifacts(result_path, Path(directory))

    def test_sample_clearance_lower_bound_is_reconstructed(self):
        payload, trace = _fixture()
        settled = trace["outcome"]["monitor"]["record"]["settled_state"]
        measurement = trace["outcome"]["monitor"]["record"]
        for record in (settled, measurement):
            record["sample_clearance"][
                "full_surface_clearance_lower_bound_m"
            ] = 10.0
        settled["D_sim_m"] = 10.0
        measurement["D_sim_min_m"] = 10.0
        trace["outcome"]["monitor"]["D_sim_min_m"] = 10.0
        payload["endpoints"]["safety"]["D_sim_min_m"]["value"] = 10.0
        with tempfile.TemporaryDirectory() as directory:
            result_path, _ = _publish(Path(directory), payload, trace)
            with self.assertRaisesRegex(
                ArtifactContractError, "full_surface_clearance_lower_bound_m"
            ):
                validate_run_artifacts(result_path, Path(directory))

    def test_positive_candidate_ledgers_do_not_create_physical_contact(self):
        payload, trace = _fixture()
        for phase in (
            "settled_post_integration_recomputed",
            "live_solver_phase_preintegration_geometry",
            "post_integration_recomputed",
        ):
            _add_candidate_without_physical_summary(
                trace,
                phase,
                physical_flag=False,
                distance_m=0.0005,
            )
        with tempfile.TemporaryDirectory() as directory:
            result_path, _ = _publish(Path(directory), payload, trace)
            result = validate_run_artifacts(result_path, Path(directory))
            self.assertFalse(
                result["endpoints"]["safety"]["any_robot_obstacle_contact"]
            )

    def test_distance_derived_post_contact_population_is_accepted(self):
        payload, trace = _fixture()
        record = _add_consistent_post_physical_contact(payload, trace)
        with tempfile.TemporaryDirectory() as directory:
            result_path, _ = _publish(Path(directory), payload, trace)
            result = validate_run_artifacts(result_path, Path(directory))
            safety = result["endpoints"]["safety"]
            self.assertTrue(safety["link56_obstacle_contact"])
            self.assertEqual(safety["first_physical_contact_point_record"], record)
            self.assertEqual(safety["D_sim_min_m"]["value"], -0.001)

    def test_settled_and_post_physical_lists_equal_candidate_subsets(self):
        for phase, field in (
            (
                "settled_post_integration_recomputed",
                "trace settled physical contacts",
            ),
            (
                "post_integration_recomputed",
                "trace post-state physical contacts",
            ),
        ):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as directory:
                payload, trace = _fixture()
                _add_candidate_without_physical_summary(
                    trace,
                    phase,
                    physical_flag=True,
                )
                result_path, _ = _publish(Path(directory), payload, trace)
                with self.assertRaisesRegex(
                    ArtifactContractError,
                    field + ".*distance-derived candidate-ledger subset",
                ):
                    validate_run_artifacts(result_path, Path(directory))

    def test_positive_candidate_cannot_be_inserted_into_physical_subset(self):
        for phase, physical_field in (
            (
                "settled_post_integration_recomputed",
                ("settled_state", "physical_contact_point_records"),
            ),
            (
                "post_integration_recomputed",
                (None, "post_state_physical_contact_point_records"),
            ),
        ):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as directory:
                payload, trace = _fixture()
                record = _add_candidate_without_physical_summary(
                    trace,
                    phase,
                    physical_flag=False,
                    distance_m=0.0005,
                )
                measurement = trace["outcome"]["monitor"]["record"]
                parent, field = physical_field
                target = measurement[parent] if parent is not None else measurement
                target[field] = [record]
                result_path, _ = _publish(Path(directory), payload, trace)
                with self.assertRaisesRegex(
                    ArtifactContractError, "contains a nonphysical record"
                ):
                    validate_run_artifacts(result_path, Path(directory))

    def test_contact_counts_are_typed_and_reconstructed(self):
        payload, trace = _fixture()
        trace["outcome"]["monitor"]["record"][
            "total_candidate_contact_point_record_count"
        ] = False
        with tempfile.TemporaryDirectory() as directory:
            result_path, _ = _publish(Path(directory), payload, trace)
            with self.assertRaisesRegex(ArtifactContractError, "nonnegative integer"):
                validate_run_artifacts(result_path, Path(directory))

    def test_trace_file_tampering_fails_reference_or_self_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result_path, trace_path = _publish(root, *_fixture())
            trace_path.write_bytes(trace_path.read_bytes() + b" ")
            with self.assertRaises(ArtifactContractError):
                validate_run_artifacts(result_path, root)

        # Even if an attacker updates the outer byte/hash reference, the
        # trace's own payload hash remains an independent immutable layer.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload, trace = _fixture()
            arm_directory = root / "arms" / payload["arm"]
            arm_directory.mkdir(parents=True)
            trace_path = arm_directory / "trace.json"
            publish_hashed_json(trace_path, trace)
            serialized = json.loads(trace_path.read_text(encoding="utf-8"))
            serialized["outcome"]["physics_clock"]["terminal_time_s"] = 12.0
            trace_path.write_text(
                json.dumps(serialized, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            payload["artifact_references"] = [
                {
                    "relative_path": str(trace_path.relative_to(root)),
                    "bytes": trace_path.stat().st_size,
                    "sha256": sha256_file(trace_path),
                    "artifact_type": "active_arm_audit_trace",
                    "media_type": "application/json",
                }
            ]
            result_path = arm_directory / "result.json"
            publish_hashed_json(result_path, payload)
            with self.assertRaisesRegex(ArtifactContractError, "result_payload_sha256"):
                validate_run_artifacts(result_path, root)


if __name__ == "__main__":
    unittest.main()
