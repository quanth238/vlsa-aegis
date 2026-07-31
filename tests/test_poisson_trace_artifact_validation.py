from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import tempfile
import unittest

from main.poisson_fullbody.contracts import (
    ArtifactContractError,
    canonical_json_bytes,
    publish_hashed_json,
    sha256_bytes,
    sha256_file,
)
from scripts.validate_poisson_run_artifacts import validate_run_artifacts
from tests.test_poisson_result_schema import valid_payload


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


def _fixture():
    payload = valid_payload()
    entered_actions = [[0.0] * 7, [0.1] * 7]
    nominal = [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    issued = [0.08, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
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
                "adapter": {},
                "execution": {},
                "D_opt_min_m": 0.04,
                "minimum_h_m2": 0.005,
                "minimum_nominal_cbf_residual_m2_per_s": -0.2,
                "minimum_safe_cbf_residual_m2_per_s": -1e-8,
                "qp": {
                    "minimum_normalized_cbf_residual": -1e-9,
                    "minimum_raw_cbf_residual_m2_per_s": -1e-8,
                    "nominal_minimum_raw_cbf_residual_m2_per_s": -0.2,
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
        "settled_active_obstacle_position_sha256": "1" * 64,
        "source_settled_active_obstacle_position_sha256": "1" * 64,
        "historical_settled_active_obstacle_position_sha256": "1" * 64,
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
        "goal_atoms": [{"index": 0, "predicate": "on", "arguments": ["a", "b"]}],
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
        "physical_contact_point_record_count": 0,
        "any_robot_obstacle_contact": False,
        "link56_obstacle_contact": False,
        "first_physical_contact_point_record": None,
        "physical_contact_point_records": [],
        "sample_clearance": {"full_surface_clearance_lower_bound_m": 0.01},
        "D_sim_m": 0.01,
    }
    measurement_record = {
        "observed_physics_substeps": 50,
        "settled_state": settled_record,
        "total_physical_contact_point_record_count": 0,
        "rollout_phase_physical_contact_point_record_count": 0,
        "live_solver_nonpositive_contact_point_record_count": 0,
        "post_state_physical_contact_point_record_count": 0,
        "any_robot_obstacle_contact": False,
        "link56_obstacle_contact": False,
        "rollout_link56_obstacle_contact": False,
        "live_solver_any_robot_obstacle_contact": False,
        "live_solver_link56_obstacle_contact": False,
        "post_state_any_robot_obstacle_contact": False,
        "post_state_link56_obstacle_contact": False,
        "first_physical_contact_point_record": None,
        "first_live_solver_physical_contact_point_record": None,
        "first_post_state_physical_contact_point_record": None,
        "live_solver_phase_contact_point_records": [],
        "post_state_physical_contact_point_records": [],
        "sample_clearance": {"full_surface_clearance_lower_bound_m": 0.01},
        "obstacle_pose_drift": {
            "maximum_translation_m": 1e-6,
            "maximum_rotation_rad": 2e-6,
            "maximum_surface_point_displacement_m": 2e-6,
        },
        "D_sim_min_m": 0.01,
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
        "surface_drift_m": 2e-6,
        "record": measurement_record,
    }
    outcome = {
        "arm": payload["arm"],
        "completion_class": "executed",
        "terminal_reason": payload["execution"]["terminal_reason"],
        "exposure_complete": True,
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
    trace = {
        "schema_version": "vlsa_poisson_active_arm_trace.v1",
        "scientific_result": False,
        "run_id": payload["run_id"],
        "case_id": payload["case_id"],
        "arm": payload["arm"],
        "resolved_geometry": {"link56_geom_ids": [55, 56]},
        "outcome": outcome,
    }

    source_hash = sha256_bytes(canonical_json_bytes(entered_actions))
    payload["pairing"]["nominal_high_level_action_ledger_sha256"] = source_hash
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


class PoissonTraceArtifactValidationTest(unittest.TestCase):
    def test_complete_trace_reconstructs_all_scientific_endpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result_path, _ = _publish(root, *_fixture())
            result = validate_run_artifacts(result_path, root)
            self.assertEqual(result["case_id"], "vlsa-t1-goal-ii-t0-e05")
            inferred = validate_run_artifacts(result_path)
            self.assertEqual(inferred["result_payload_sha256"], result["result_payload_sha256"])

    def test_right_censored_tracking_failure_reconstructs_exact_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result_path, _ = _publish(root, *_partial_tracking_failure_fixture())
            result = validate_run_artifacts(result_path, root)
            self.assertEqual(result["completion_class"], "controller_tracking_invalid")
            self.assertEqual(result["execution"]["physics_substeps"], 1)

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
