from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from main.poisson_fullbody.contracts import (
    attach_payload_hash,
    canonical_json_bytes,
    load_hashed_json,
    publish_hashed_json,
    sha256_bytes,
    sha256_file,
)
from main.poisson_fullbody.result_schema import (
    validate_active_canary_pair,
    validate_episode_result,
)
from scripts.run_poisson_active_canary import (
    ActiveRunnerError,
    _canonical,
    _scientific_payload,
    _sha256,
    _validate_complete_run_receipt,
    _validate_trace_payload_against_scientific,
)
from tests.test_poisson_result_schema import valid_partial_payload


ROOT = Path(__file__).resolve().parents[1]


def serialized_trace_fixture():
    """One completed 2 ms callback followed by a tracking fail-closed stop."""

    scientific = valid_partial_payload("controller_tracking_invalid")
    source_action = [0.0] * 7
    nominal = [0.1] + [0.0] * 6
    issued = list(nominal)
    measured = [0.16] + [0.0] * 6
    settled_eef = [0.0, 0.0, 0.0]
    forwarded_eef = [0.001, 0.0, 0.0]
    tracking_rmse = math.sqrt((0.06**2) / 7.0)
    crossing = {
        "high_level_index": 0,
        "inner_control_index": 0,
        "physics_substep_index": 0,
        "error_linf_rad_s": 0.06,
        "cumulative_rmse_rad_s": tracking_rmse,
        "linf_threshold_rad_s": 0.05,
        "rmse_threshold_rad_s": 0.02,
        "command_rad_s": issued,
        "measured_rad_s": measured,
    }
    nominal_ledger = [
        {
            "high_level_index": 0,
            "inner_control_index": 0,
            "qdot_rad_s": nominal,
        }
    ]
    executed_ledger = [
        {
            "high_level_index": 0,
            "inner_control_index": 0,
            "qdot_rad_s": issued,
        }
    ]
    entered_actions = [source_action]
    completed_actions = []
    scientific["execution"].update(
        {
            "entered_source_action_prefix_sha256": _sha256(
                _canonical(entered_actions)
            ),
            "completed_high_level_source_action_prefix_sha256": _sha256(
                _canonical(completed_actions)
            ),
            "nominal_joint_velocity_ledger_sha256": _sha256(
                _canonical(nominal_ledger)
            ),
            "executed_controller_action_ledger_sha256": _sha256(
                _canonical(executed_ledger)
            ),
        }
    )
    scientific["endpoints"]["usefulness"].update(
        {
            "nominal_joint_motion_integral_rad": 0.1 * 0.002,
            "safe_joint_motion_integral_rad": 0.1 * 0.002,
            "measured_joint_motion_integral_rad": 0.16 * 0.002,
            "eef_path_length_m": 0.001,
            "correction_integral_rad": 0.0,
            "motion_retention_ratio": 1.0,
            "zero_motion_fraction": 0.0,
            "all_issued_arm_joint_commands_zero": False,
            "safety_by_no_execution": False,
        }
    )
    scientific["endpoints"]["validity"].update(
        {
            "maximum_velocity_tracking_error_rad_s": 0.06,
            "velocity_tracking_rmse_rad_s": tracking_rmse,
            "velocity_tracking_observed_physics_substep_count": 1,
            "first_velocity_tracking_threshold_crossing": crossing,
        }
    )
    car_ledger = {
        "settled_active_obstacle_root_position_m": [0.0, 0.0, 0.0],
        "settled_active_obstacle_position_sha256": "1" * 64,
        "source_settled_active_obstacle_position_sha256": "1" * 64,
        "historical_settled_active_obstacle_position_sha256": "1" * 64,
        "completed_post_step_positions": [],
    }
    scientific["endpoints"]["safety"]["paper_car"].update(
        {
            "maximum_active_obstacle_l1_displacement_m": 0.0,
            "position_ledger_sha256": _sha256(_canonical(car_ledger)),
        }
    )

    validity = scientific["endpoints"]["validity"]
    usefulness = scientific["endpoints"]["usefulness"]
    paper_car = scientific["endpoints"]["safety"]["paper_car"]
    outcome = {
        "arm": scientific["arm"],
        "completion_class": scientific["completion_class"],
        "exposure_complete": False,
        "prefix": {
            "high_level_steps": 1,
            "inner_control_steps": 1,
            "physics_substeps": 1,
            "physics_exposure_seconds": 0.002,
        },
        "physics_clock": {
            "settled_time_s": 2.0,
            "terminal_time_s": 2.002,
            "observed_exposure_s": 0.002,
            "expected_from_completed_substeps_s": 0.002,
            "exact_count_consistent_within_abs_1e_10_s": True,
        },
        "entered_source_actions": entered_actions,
        "completed_source_actions": completed_actions,
        "nominal_ledger": nominal_ledger,
        "executed_ledger": executed_ledger,
        "inner_trace": [
            {
                "high_level_index": 0,
                "inner_control_index": 0,
                "D_opt_min_m": 0.03,
                "minimum_h_m2": 0.004,
                "minimum_nominal_cbf_residual_m2_per_s": -0.2,
                "minimum_safe_cbf_residual_m2_per_s": -1e-8,
            }
        ],
        "physics_trace": [
            {
                "high_level_index": 0,
                "inner_control_index": 0,
                "physics_substep_index": 0,
                "nominal_joint_velocity_command_rad_s": nominal,
                "issued_joint_velocity_command_rad_s": issued,
                "measured_arm_joint_velocity_rad_s": measured,
                "forwarded_eef_position_m": forwarded_eef,
            }
        ],
        "fail_closed_attempt_trace": [],
        "realized_cbf_trace": [
            {
                "high_level_index": 0,
                "inner_control_index": 0,
                "physics_substep_index": 0,
                "valid": True,
                "query_count": 1,
                "minimum_realized_cbf_residual_m2_per_s": 1e-6,
                "minimum_h_m2": 0.004,
                "D_opt_min_m": 0.03,
                "first_negative_crossing": None,
            }
        ],
        "monitor": {
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
            "record": {
                "observed_physics_substeps": 1,
                "total_physical_contact_point_record_count": 0,
                "rollout_phase_physical_contact_point_record_count": 0,
                "live_solver_nonpositive_contact_point_record_count": 0,
                "post_state_physical_contact_point_record_count": 0,
            },
        },
        "paper_car": paper_car,
        "paper_car_position_ledger": car_ledger,
        "task": scientific["endpoints"]["task"],
        "usefulness": usefulness,
        "validity": validity,
        "optimizer_counts": {
            "attempt_count": 1,
            "solved_count": 1,
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
        "initial_protected_sample_audit": {
            "protected_sample_count": 1,
            "field_query_count": 1,
            "valid_field_query_count": 1,
            "minimum_h_m2": 0.004,
            "D_opt_min_m": 0.03,
            "strict_safe_start": True,
        },
        "realized_cbf_audit": {
            "semantics": "every_completed_2ms_post_state_grad_h_T_J_qvel_actual_plus_alpha_h",
            "observed_physics_substep_count": 1,
            "residual_evaluation_count": 1,
            "negative_residual_callback_count": 0,
            "first_negative_residual": None,
        },
        "eef_path_audit": {
            "semantics": "settled_forwarded_grip_site_plus_every_completed_2ms_post_state",
            "position_sample_count": 2,
            "settled_forwarded_eef_position_m": settled_eef,
        },
        "tracking": {
            "command_count": 1,
            "observed_physics_substep_count": 1,
            "maximum_linf_error_rad_s": 0.06,
            "cumulative_rmse_rad_s": tracking_rmse,
            "maximum_linf_threshold_rad_s": 0.05,
            "maximum_rmse_threshold_rad_s": 0.02,
            "first_threshold_crossing": crossing,
        },
        "terminal_simulator_state_sha256": scientific["execution"][
            "terminal_simulator_state_sha256"
        ],
        "terminal_observation_sha256": scientific["execution"][
            "terminal_observation_sha256"
        ],
    }
    trace = {
        "schema_version": "vlsa_poisson_active_arm_trace.v3",
        "scientific_result": False,
        "run_id": scientific["run_id"],
        "case_id": scientific["case_id"],
        "arm": scientific["arm"],
        "outcome": outcome,
    }
    # Exercise the same plain JSON representation that is read back from disk.
    return json.loads(json.dumps(trace)), json.loads(json.dumps(scientific))


def _adapter_complete_fixture():
    from tests.test_poisson_trace_artifact_validation import (
        _fixture,
        _producer_sha256,
    )

    payload, trace = _fixture()
    arm = "joint_velocity_adapter_only"
    payload["arm"] = arm
    trace["arm"] = arm
    outcome = trace["outcome"]
    outcome["arm"] = arm
    payload["runtime"]["intervention"].update(
        {
            "enabled": False,
            "mode": arm,
            "protected_robot_bodies": [],
        }
    )
    payload["optimizer"].update(
        {
            "enabled": False,
            "attempt_count": 0,
            "solved_count": 0,
            "terminal_status": "not_applicable",
            "minimum_normalized_cbf_residual": None,
            "minimum_raw_cbf_residual_m2_per_s": None,
        }
    )
    payload["runtime"]["intervention"][
        "configuration_sha256"
    ] = _producer_sha256(
        {
            "arm": arm,
            "field_sha256": trace["field_bundle_hashes"]["field_sha256"],
            "protected": [],
        }
    )
    payload["optimizer"]["configuration_sha256"] = _producer_sha256(
        {
            "runtime_protocol_parameter_block_sha256": payload["runtime"][
                "protocol_parameter_block_sha256"
            ],
            "arm": arm,
            "solver": "osqp",
        }
    )
    for executed, nominal in zip(
        outcome["executed_ledger"], outcome["nominal_ledger"]
    ):
        executed["qdot_rad_s"] = copy.deepcopy(nominal["qdot_rad_s"])
    for row in outcome["inner_trace"]:
        row["qp"] = None
        nominal_command = list(
            row["execution"]["nominal_qdot_physical_rad_s"]
        )
        row["execution"].update(
            {
                "executed_qdot_physical_rad_s": nominal_command,
                "filter_correction_rad_s": [0.0] * 7,
                "filter_correction_l2_rad_s": 0.0,
                "normalized_executed_action": [
                    value / 0.5 for value in nominal_command
                ]
                + [row["source_action"][6]],
            }
        )
    for index, row in enumerate(outcome["physics_trace"]):
        command = outcome["nominal_ledger"][index // 5]["qdot_rad_s"]
        row["issued_joint_velocity_command_rad_s"] = copy.deepcopy(command)
        row["measured_arm_joint_velocity_rad_s"] = copy.deepcopy(command)
    outcome["realized_cbf_trace"] = []
    outcome["optimizer_counts"].update({"attempt_count": 0, "solved_count": 0})
    outcome["optimizer_terminal_status"] = "not_applicable"
    outcome["minimums"].update(
        {
            "h_m2": None,
            "nominal_raw_cbf_residual_m2_per_s": None,
            "safe_raw_cbf_residual_m2_per_s": None,
            "safe_normalized_cbf_residual": None,
            "realized_raw_cbf_residual_m2_per_s": None,
        }
    )
    nominal_motion = outcome["usefulness"]["nominal_joint_motion_integral_rad"]
    outcome["usefulness"].update(
        {
            "safe_joint_motion_integral_rad": nominal_motion,
            "measured_joint_motion_integral_rad": nominal_motion,
            "correction_integral_rad": 0.0,
            "motion_retention_ratio": 1.0,
        }
    )
    outcome["realized_cbf_audit"].update(
        {
            "observed_physics_substep_count": 0,
            "residual_evaluation_count": 0,
            "negative_residual_callback_count": 0,
            "first_negative_residual": None,
        }
    )
    outcome["validity"].update(
        {
            "realized_cbf_observed_physics_substep_count": 0,
            "realized_cbf_residual_evaluation_count": 0,
        }
    )
    payload["execution"]["executed_controller_action_ledger_sha256"] = (
        sha256_bytes(canonical_json_bytes(outcome["executed_ledger"]))
    )
    payload["endpoints"]["usefulness"] = copy.deepcopy(outcome["usefulness"])
    payload["endpoints"]["validity"] = copy.deepcopy(outcome["validity"])
    unavailable_reasons = {
        "h_min": "adapter_intervention_disabled",
        "minimum_nominal_cbf_residual": "no_solved_poisson_qp",
        "minimum_safe_cbf_residual": "no_solved_poisson_qp",
        "minimum_realized_cbf_residual": (
            "adapter_intervention_disabled_or_no_valid_completed_post_state_query"
        ),
    }
    for field, reason in unavailable_reasons.items():
        payload["endpoints"]["safety"][field].update(
            {"available": False, "value": None, "reason": reason}
        )
    return payload, trace


def _replace_hashed_json(path, value):
    payload = dict(value)
    payload.pop("result_payload_sha256", None)
    path.unlink()
    publish_hashed_json(path, payload)


def _publish_complete_resume_fixture(parent):
    from tests.test_poisson_trace_artifact_validation import _fixture

    adapter_payload, adapter_trace = _adapter_complete_fixture()
    psf_payload, psf_trace = _fixture()
    run_root = parent / psf_payload["run_id"]
    run_root.mkdir()
    results = {}
    for payload, trace in (
        (adapter_payload, adapter_trace),
        (psf_payload, psf_trace),
    ):
        arm_directory = run_root / payload["case_id"] / payload["arm"]
        arm_directory.mkdir(parents=True)
        trace_path = arm_directory / "trace.json"
        publish_hashed_json(trace_path, trace)
        result_payload = copy.deepcopy(payload)
        result_payload["artifact_references"] = [
            {
                "relative_path": str(trace_path.relative_to(run_root)),
                "bytes": trace_path.stat().st_size,
                "sha256": sha256_file(trace_path),
                "artifact_type": "active_arm_audit_trace",
                "media_type": "application/json",
            }
        ]
        result_path = arm_directory / "result.json"
        publish_hashed_json(result_path, result_payload)
        results[payload["arm"]] = load_hashed_json(result_path)

    pair_payload = dict(
        validate_active_canary_pair(
            results["joint_velocity_adapter_only"],
            results["joint_velocity_psf_link56"],
        )
    )
    pair_payload.update(
        {
            "status": "complete",
            "scientific_result": False,
            "four_arm_109_case_study_complete": False,
            "run_contract_sha256": psf_payload["provenance"][
                "run_contract_sha256"
            ],
        }
    )
    pair_path = run_root / psf_payload["case_id"] / "pair_result.json"
    publish_hashed_json(pair_path, pair_payload)
    receipt_identity = {
        "run_id": psf_payload["run_id"],
        "case_id": psf_payload["case_id"],
        "run_contract_sha256": psf_payload["provenance"]["run_contract_sha256"],
        "code_commit": psf_payload["provenance"]["code_commit"],
        "manifest_sha256": psf_payload["provenance"]["manifest_sha256"],
        "manifest_record_sha256": psf_payload["provenance"][
            "manifest_record_sha256"
        ],
        "selection_config_sha256": psf_payload["provenance"][
            "protocol_config_sha256"
        ],
        "runtime_protocol_raw_sha256": psf_payload["provenance"][
            "runtime_protocol_raw_sha256"
        ],
        "runtime_protocol_semantic_sha256": psf_payload["provenance"][
            "runtime_protocol_semantic_sha256"
        ],
        "runtime_parameter_block_sha256": psf_payload["runtime"][
            "protocol_parameter_block_sha256"
        ],
        "checkpoint_tree_sha256": psf_payload["runtime"]["model"][
            "checkpoint_sha256"
        ],
        "checkpoint_receipt_file_sha256": "1" * 64,
        "historical_source_run_contract_sha256": "2" * 64,
        "historical_result_file_sha256": psf_trace["source_replay"][
            "historical_result_file_sha256"
        ],
        "historical_result_payload_sha256": psf_trace["source_replay"][
            "historical_result_payload_sha256"
        ],
        "source_action_ledger_sha256": psf_payload["pairing"][
            "nominal_high_level_action_ledger_sha256"
        ],
        "numeric_prerequisite_payload_sha256": "5" * 64,
        "parity_prerequisite_payload_sha256": "6" * 64,
        "identification_prerequisite_payload_sha256": "7" * 64,
    }
    receipt = {
        "schema_version": "vlsa_poisson_active_canary_run_receipt.v2",
        "status": "complete",
        "scientific_result": False,
        "run_id": psf_payload["run_id"],
        "case_id": psf_payload["case_id"],
        "staged_scope": "two_arms_one_canary_not_four_arm_109_case_study",
        "identity": receipt_identity,
        "arm_results": {
            arm: {
                "relative_path": "%s/%s/result.json"
                % (psf_payload["case_id"], arm),
                "sha256": sha256_file(
                    run_root / psf_payload["case_id"] / arm / "result.json"
                ),
            }
            for arm in results
        },
        "pair_result_relative_path": str(pair_path.relative_to(run_root)),
        "pair_result_sha256": sha256_file(pair_path),
        "elapsed_seconds": 1.0,
    }
    receipt_path = run_root / "run_receipt.json"
    publish_hashed_json(receipt_path, receipt)
    return run_root, receipt_path, receipt_identity, results


class ActiveRunnerTraceContractTest(unittest.TestCase):
    def test_live_scientific_payload_uses_schema_valid_seed_scopes(self):
        from tests.test_poisson_trace_artifact_validation import _fixture

        expected, trace = _fixture()
        identity = expected["case_identity"]
        case = {
            "case_id": expected["case_id"],
            "case_ordinal": identity["manifest_case_ordinal"],
            "pair_group_id": identity["pair_group_id"],
            "suite": identity["suite"],
            "safety_level": identity["safety_level"],
            "logical_task_index": identity["logical_task_index"],
            "resolved_task_index": identity["resolved_task_index"],
            "episode_index": identity["episode_index"],
            "task_name": identity["task_name"],
            "task_family_id": identity["task_family_id"],
            "task_level_group_id": identity["task_level_group_id"],
            "bddl_path": identity["bddl_relative_path"],
            "bddl_sha256": identity["bddl_sha256"],
            "initial_states_path": identity["initial_states_relative_path"],
            "initial_states_sha256": identity["initial_states_sha256"],
            "active_obstacle_name": identity["active_obstacle_name"],
            "split": identity["split"],
            "stratum": identity["stratum"],
            "max_steps": identity["suite_horizon_high_level_steps"],
            "environment_seed": 7,
            "policy_noise_seed": 2026122900,
        }

        source_replay = trace["source_replay"]

        class Replay:
            arm = source_replay["source_arm"]
            actions = tuple(tuple(row) for row in trace["outcome"]["entered_source_actions"])
            executed_sequence_sha256 = source_replay["executed_sequence_sha256"]
            initial_observation_sha256 = source_replay["initial_observation_sha256"]
            policy_noise_schedule_sha256 = source_replay[
                "policy_noise_schedule_sha256"
            ]
            source_policy_query_schedule_sha256 = source_replay[
                "source_policy_query_schedule_sha256"
            ]
            source_policy_query_count = source_replay["source_policy_query_count"]

            @staticmethod
            def provenance():
                return copy.deepcopy(source_replay)

        checkpoint = {
            "receipt_payload_sha256": "1" * 64,
            "receipt_file_sha256": "2" * 64,
            "execution_semantics": "historical_openpi_checkpoint_not_requeried",
            "checkpoint_sha256": expected["runtime"]["model"]["checkpoint_sha256"],
            "normalization_statistics_sha256": expected["runtime"]["model"][
                "normalization_statistics_sha256"
            ],
        }
        protocol_hashes = SimpleNamespace(
            protocol_sha256=expected["provenance"][
                "runtime_protocol_semantic_sha256"
            ],
            parameter_block_sha256=expected["runtime"][
                "protocol_parameter_block_sha256"
            ],
        )
        bundle = SimpleNamespace(
            hashes=SimpleNamespace(**trace["field_bundle_hashes"])
        )

        class Resolved:
            @staticmethod
            def to_dict():
                return copy.deepcopy(trace["resolved_geometry"])

        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            manifest_path = temporary / "manifest.jsonl"
            selection_path = temporary / "selection.json"
            runtime_path = temporary / "runtime.json"
            manifest_path.write_text("{}\n", encoding="utf-8")
            selection_path.write_text("{}\n", encoding="utf-8")
            runtime_path.write_text("{}\n", encoding="utf-8")
            scientific = _scientific_payload(
                root=ROOT,
                run_id=expected["run_id"],
                case=case,
                case_row_hash=expected["provenance"]["manifest_record_sha256"],
                manifest_path=manifest_path,
                selection_path=selection_path,
                runtime_protocol_path=runtime_path,
                protocol_hashes=protocol_hashes,
                checkpoint=checkpoint,
                replay=Replay(),
                allocation=expected["allocation"],
                source_git={"commit": expected["provenance"]["code_commit"]},
                source_pairing={},
                bundle=bundle,
                resolved=Resolved(),
                arm_outcome=trace["outcome"],
                artifact_reference={
                    "relative_path": "%s/%s/trace.json"
                    % (expected["case_id"], expected["arm"]),
                    "bytes": 1,
                    "sha256": "3" * 64,
                    "artifact_type": "active_arm_audit_trace",
                    "media_type": "application/json",
                },
                run_contract_sha256=expected["provenance"][
                    "run_contract_sha256"
                ],
            )
        scopes = [entry["scope"] for entry in scientific["seeds"]["entries"]]
        self.assertEqual(scopes, ["simulator", "numpy", "policy_noise"])
        validate_episode_result(attach_payload_hash(scientific))

    def test_one_substep_tracking_failure_trace_matches_scientific_projection(self):
        trace, scientific = serialized_trace_fixture()
        validate_episode_result(attach_payload_hash(scientific))
        _validate_trace_payload_against_scientific(trace, scientific)

    def test_tampered_measured_qvel_is_rejected(self):
        trace, scientific = serialized_trace_fixture()
        tampered = copy.deepcopy(trace)
        tampered["outcome"]["physics_trace"][0][
            "measured_arm_joint_velocity_rad_s"
        ][0] = 0.17
        with self.assertRaisesRegex(ActiveRunnerError, "trace/scientific mismatch"):
            _validate_trace_payload_against_scientific(tampered, scientific)

    def test_tampered_forwarded_eef_position_is_rejected(self):
        trace, scientific = serialized_trace_fixture()
        tampered = copy.deepcopy(trace)
        tampered["outcome"]["physics_trace"][0][
            "forwarded_eef_position_m"
        ][0] = 0.002
        with self.assertRaisesRegex(ActiveRunnerError, "trace/scientific mismatch"):
            _validate_trace_payload_against_scientific(tampered, scientific)

    def test_complete_resume_rejects_rehashed_nested_trace_tamper(self):
        with tempfile.TemporaryDirectory() as directory:
            (
                run_root,
                receipt_path,
                receipt_identity,
                original_results,
            ) = _publish_complete_resume_fixture(Path(directory))
            resumed = _validate_complete_run_receipt(
                receipt_path=receipt_path,
                output=run_root,
                expected_receipt_identity=receipt_identity,
                expected_arm_identities=original_results,
                expected_source_action_count=2,
            )
            self.assertEqual(set(resumed), set(original_results))

            case_id = receipt_identity["case_id"]
            psf_arm = "joint_velocity_psf_link56"
            psf_directory = run_root / case_id / psf_arm
            trace_path = psf_directory / "trace.json"
            trace = load_hashed_json(trace_path)
            trace["outcome"]["physics_trace"][0][
                "forwarded_eef_position_m"
            ][0] = 0.125
            _replace_hashed_json(trace_path, trace)

            # Rehash every outer artifact so shallow self-hash, reference,
            # pair, and receipt checks remain internally consistent.
            result_path = psf_directory / "result.json"
            psf_result = load_hashed_json(result_path)
            trace_reference = psf_result["artifact_references"][0]
            trace_reference["bytes"] = trace_path.stat().st_size
            trace_reference["sha256"] = sha256_file(trace_path)
            _replace_hashed_json(result_path, psf_result)
            psf_result = load_hashed_json(result_path)

            adapter_result = load_hashed_json(
                run_root
                / case_id
                / "joint_velocity_adapter_only"
                / "result.json"
            )
            pair_payload = dict(
                validate_active_canary_pair(adapter_result, psf_result)
            )
            pair_payload.update(
                {
                    "status": "complete",
                    "scientific_result": False,
                    "four_arm_109_case_study_complete": False,
                    "run_contract_sha256": receipt_identity[
                        "run_contract_sha256"
                    ],
                }
            )
            pair_path = run_root / case_id / "pair_result.json"
            _replace_hashed_json(pair_path, pair_payload)

            receipt = load_hashed_json(receipt_path)
            receipt["arm_results"][psf_arm]["sha256"] = sha256_file(
                result_path
            )
            receipt["pair_result_sha256"] = sha256_file(pair_path)
            _replace_hashed_json(receipt_path, receipt)

            with self.assertRaisesRegex(
                ActiveRunnerError,
                "failed deep validation",
            ):
                _validate_complete_run_receipt(
                    receipt_path=receipt_path,
                    output=run_root,
                    expected_receipt_identity=receipt_identity,
                    expected_arm_identities=original_results,
                    expected_source_action_count=2,
                )

    def test_complete_resume_rejects_individually_valid_paired_audit_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            (
                run_root,
                receipt_path,
                receipt_identity,
                original_results,
            ) = _publish_complete_resume_fixture(Path(directory))
            case_id = receipt_identity["case_id"]
            psf_arm = "joint_velocity_psf_link56"
            psf_directory = run_root / case_id / psf_arm
            trace_path = psf_directory / "trace.json"
            trace = load_hashed_json(trace_path)

            # This query remains structurally and numerically valid; clearance
            # is not used in the derivative arithmetic.  Rehashing both the
            # audit and its independently reconstructed receipt therefore
            # creates one individually valid arm with different settled-state
            # evidence.  The matched-arm gate must still reject it.
            audit = trace["settled_link56_differential_audit"]
            audit["sample_records"][0]["base_field_query"][
                "outer_boundary_clearance_m"
            ] = 0.2
            audit.pop("audit_payload_sha256")
            audit["audit_payload_sha256"] = sha256_bytes(
                canonical_json_bytes(audit)
            )
            trace["settled_link56_differential_audit_validation"][
                "audit_payload_sha256"
            ] = audit["audit_payload_sha256"]
            _replace_hashed_json(trace_path, trace)

            result_path = psf_directory / "result.json"
            psf_result = load_hashed_json(result_path)
            trace_reference = psf_result["artifact_references"][0]
            trace_reference["bytes"] = trace_path.stat().st_size
            trace_reference["sha256"] = sha256_file(trace_path)
            _replace_hashed_json(result_path, psf_result)
            psf_result = load_hashed_json(result_path)

            adapter_result = load_hashed_json(
                run_root
                / case_id
                / "joint_velocity_adapter_only"
                / "result.json"
            )
            pair_payload = dict(
                validate_active_canary_pair(adapter_result, psf_result)
            )
            pair_payload.update(
                {
                    "status": "complete",
                    "scientific_result": False,
                    "four_arm_109_case_study_complete": False,
                    "run_contract_sha256": receipt_identity[
                        "run_contract_sha256"
                    ],
                }
            )
            pair_path = run_root / case_id / "pair_result.json"
            _replace_hashed_json(pair_path, pair_payload)

            receipt = load_hashed_json(receipt_path)
            receipt["arm_results"][psf_arm]["sha256"] = sha256_file(
                result_path
            )
            receipt["pair_result_sha256"] = sha256_file(pair_path)
            _replace_hashed_json(receipt_path, receipt)

            with self.assertRaisesRegex(
                ActiveRunnerError,
                "paired trace.settled_link56_differential_audit",
            ):
                _validate_complete_run_receipt(
                    receipt_path=receipt_path,
                    output=run_root,
                    expected_receipt_identity=receipt_identity,
                    expected_arm_identities=original_results,
                    expected_source_action_count=2,
                )

    def test_publication_and_partial_resume_guards_are_structurally_ordered(self):
        source = (ROOT / "scripts/run_poisson_active_canary.py").read_text(
            encoding="utf-8"
        )
        validate = source.index(
            "validate_episode_result(attach_payload_hash(payload))"
        )
        publish = source.index("publish_episode_result(final_path, payload")
        self.assertLess(validate, publish)

        complete_receipt = source.index("if receipt_path.exists():")
        complete_return = source.index("            return 0", complete_receipt)
        partial_guard = source.index("                if final_path.exists():")
        arm_execution = source.index("                    outcome = _run_arm(", partial_guard)
        refusal = (
            "partial arm result exists without a complete immutable run "
            "\"\n                        \"receipt; choose a new run ID"
        )
        self.assertLess(complete_receipt, complete_return)
        self.assertLess(complete_return, partial_guard)
        self.assertLess(partial_guard, arm_execution)
        self.assertIn(refusal, source[partial_guard:arm_execution])


if __name__ == "__main__":
    unittest.main()
