from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from main.poisson_fullbody.contracts import (
    ArtifactContractError,
    attach_payload_hash,
    canonical_json_bytes,
    load_hashed_json,
    publish_hashed_json,
    sha256_bytes,
    sha256_file,
)
from main.poisson_fullbody.result_schema import validate_active_canary_pair
from scripts.validate_poisson_run_artifacts import (
    ACTIVE_ARMS,
    validate_complete_run_artifacts as _validate_complete_run_artifacts,
)
from tests.test_poisson_trace_artifact_validation import _fixture
from tests.test_poisson_trace_artifact_validation import _producer_sha256


ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "scripts" / "validate_poisson_run_artifacts.py"
CASE_ID = "vlsa-t1-goal-ii-t0-e05"
RUN_ID = "poisson-canary-a"
RUN_CONTRACT_SHA256 = "b" * 64


def validate_complete_run_artifacts(run_root, *, candidate_receipt=None):
    return _validate_complete_run_artifacts(
        run_root,
        candidate_receipt=candidate_receipt,
        expected_source_action_count=2,
    )


def _unavailable(units, reason):
    return {
        "available": False,
        "value": None,
        "units": units,
        "reason": reason,
    }


def _adapter_fixture():
    payload, trace = _fixture()
    arm = "joint_velocity_adapter_only"
    payload["arm"] = arm
    trace["arm"] = arm
    trace["outcome"]["arm"] = arm

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
            "infeasible_count": 0,
            "solver_failure_count": 0,
            "postcheck_failure_count": 0,
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

    outcome = trace["outcome"]
    nominal = [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    for row in outcome["executed_ledger"]:
        row["qdot_rad_s"] = list(nominal)
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
    for row in outcome["physics_trace"]:
        row["issued_joint_velocity_command_rad_s"] = list(nominal)
        row["measured_arm_joint_velocity_rad_s"] = list(nominal)
    outcome["realized_cbf_trace"] = []
    outcome["optimizer_counts"] = {
        "attempt_count": 0,
        "solved_count": 0,
        "infeasible_count": 0,
        "solver_failure_count": 0,
        "postcheck_failure_count": 0,
    }
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
    outcome["realized_cbf_audit"].update(
        {
            "observed_physics_substep_count": 0,
            "residual_evaluation_count": 0,
            "negative_residual_callback_count": 0,
            "first_negative_residual": None,
        }
    )
    validity = outcome["validity"]
    validity.update(
        {
            "realized_cbf_observed_physics_substep_count": 0,
            "realized_cbf_residual_evaluation_count": 0,
            "negative_realized_cbf_residual_count": 0,
            "first_negative_realized_cbf_residual": None,
        }
    )
    nominal_motion = 50 * 0.1 * 0.002
    outcome["usefulness"].update(
        {
            "nominal_joint_motion_integral_rad": nominal_motion,
            "safe_joint_motion_integral_rad": nominal_motion,
            "measured_joint_motion_integral_rad": nominal_motion,
            "correction_integral_rad": 0.0,
            "motion_retention_ratio": 1.0,
        }
    )

    safety = payload["endpoints"]["safety"]
    safety["h_min"] = _unavailable(
        "poisson_field_m2", "adapter_intervention_disabled"
    )
    for field in (
        "minimum_nominal_cbf_residual",
        "minimum_safe_cbf_residual",
        "minimum_realized_cbf_residual",
    ):
        safety[field] = _unavailable(
            "poisson_cbf_m2_per_s", "adapter_intervention_disabled"
        )
    payload["execution"]["executed_controller_action_ledger_sha256"] = sha256_bytes(
        canonical_json_bytes(outcome["executed_ledger"])
    )
    payload["endpoints"]["validity"] = copy.deepcopy(validity)
    payload["endpoints"]["usefulness"] = copy.deepcopy(outcome["usefulness"])
    return payload, trace


def _publish_arm(root, payload, trace):
    arm = payload["arm"]
    arm_root = root / payload["case_id"] / arm
    arm_root.mkdir(parents=True)
    trace_path = arm_root / "trace.json"
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
    result_path = arm_root / "result.json"
    publish_hashed_json(result_path, payload)
    return result_path, trace_path


def _pair_payload(adapter, psf):
    payload = dict(validate_active_canary_pair(adapter, psf))
    payload.update(
        {
            "status": "complete",
            "scientific_result": False,
            "four_arm_109_case_study_complete": False,
            "run_contract_sha256": RUN_CONTRACT_SHA256,
        }
    )
    return payload


def _receipt_identity(source_action_ledger_sha256):
    return {
        "run_id": RUN_ID,
        "case_id": CASE_ID,
        "run_contract_sha256": RUN_CONTRACT_SHA256,
        "code_commit": "a" * 40,
        "manifest_sha256": "c" * 64,
        "manifest_record_sha256": "d" * 64,
        "selection_config_sha256": "e" * 64,
        "runtime_protocol_raw_sha256": "6" * 64,
        "runtime_protocol_semantic_sha256": "7" * 64,
        "runtime_parameter_block_sha256": "c" * 64,
        "checkpoint_tree_sha256": "e" * 64,
        "checkpoint_receipt_file_sha256": "f" * 64,
        "historical_source_run_contract_sha256": "1" * 64,
        "historical_result_file_sha256": "2" * 64,
        "historical_result_payload_sha256": "1" * 64,
        "source_action_ledger_sha256": source_action_ledger_sha256,
        "numeric_prerequisite_payload_sha256": "5" * 64,
        "parity_prerequisite_payload_sha256": "6" * 64,
        "identification_prerequisite_payload_sha256": "7" * 64,
    }


def _build_complete_run(parent):
    root = parent / RUN_ID
    root.mkdir()
    adapter_payload, adapter_trace = _adapter_fixture()
    psf_payload, psf_trace = _fixture()
    identity = _receipt_identity(
        psf_payload["pairing"]["nominal_high_level_action_ledger_sha256"]
    )
    adapter_path, adapter_trace_path = _publish_arm(
        root, adapter_payload, adapter_trace
    )
    psf_path, psf_trace_path = _publish_arm(root, psf_payload, psf_trace)
    adapter = load_hashed_json(adapter_path)
    psf = load_hashed_json(psf_path)

    pair_path = root / CASE_ID / "pair_result.json"
    publish_hashed_json(pair_path, _pair_payload(adapter, psf))
    receipt = {
        "schema_version": "vlsa_poisson_active_canary_run_receipt.v2",
        "status": "complete",
        "scientific_result": False,
        "run_id": RUN_ID,
        "case_id": CASE_ID,
        "staged_scope": "two_arms_one_canary_not_four_arm_109_case_study",
        "identity": identity,
        "arm_results": {
            arm: {
                "relative_path": "%s/%s/result.json" % (CASE_ID, arm),
                "sha256": sha256_file(
                    root / CASE_ID / arm / "result.json"
                ),
            }
            for arm in ACTIVE_ARMS
        },
        "pair_result_relative_path": "%s/pair_result.json" % CASE_ID,
        "pair_result_sha256": sha256_file(pair_path),
        "elapsed_seconds": 1.0,
    }
    receipt_path = root / "run_receipt.json"
    publish_hashed_json(receipt_path, receipt)
    return {
        "root": root,
        "receipt": receipt_path,
        "pair": pair_path,
        "results": {
            "joint_velocity_adapter_only": adapter_path,
            "joint_velocity_psf_link56": psf_path,
        },
        "traces": {
            "joint_velocity_adapter_only": adapter_trace_path,
            "joint_velocity_psf_link56": psf_trace_path,
        },
    }


def _rewrite_hashed_json(path, mutate):
    payload = load_hashed_json(path)
    payload.pop("result_payload_sha256")
    mutate(payload)
    path.unlink()
    publish_hashed_json(path, payload)


def _refresh_receipt_file_hashes(run):
    def mutate(receipt):
        for arm, result_path in run["results"].items():
            receipt["arm_results"][arm]["sha256"] = sha256_file(result_path)
        receipt["pair_result_sha256"] = sha256_file(run["pair"])

    _rewrite_hashed_json(run["receipt"], mutate)


class PoissonCompleteRunValidationTest(unittest.TestCase):
    def test_pair_rejects_different_runtime_parameter_blocks(self):
        adapter, _adapter_trace = _adapter_fixture()
        psf, _psf_trace = _fixture()
        psf["runtime"]["protocol_parameter_block_sha256"] = "d" * 64
        with self.assertRaisesRegex(
            ArtifactContractError, "protocol_parameter_block_sha256"
        ):
            validate_active_canary_pair(
                attach_payload_hash(adapter),
                attach_payload_hash(psf),
            )

    def test_pair_does_not_call_unrealized_command_motion_nonzero(self):
        adapter, _adapter_trace = _adapter_fixture()
        psf, _psf_trace = _fixture()
        psf["endpoints"]["usefulness"].update(
            {
                "measured_joint_motion_integral_rad": 0.0,
                "eef_path_length_m": 0.0,
            }
        )
        pair = validate_active_canary_pair(
            attach_payload_hash(adapter),
            attach_payload_hash(psf),
        )
        for field in (
            "nonzero_motion_link56_contact_prevention_eligible",
            "all_robot_selected_obstacle_contact_prevention_eligible",
            "paper_car_displacement_prevention_eligible",
            "task_successful_full_safety_correction_eligible",
            "task_success_preservation_eligible",
            "task_rescue_eligible",
        ):
            self.assertFalse(pair[field], field)
        for field in (
            "link56_prevention_ineligibility_reasons",
            "all_robot_prevention_ineligibility_reasons",
            "paper_car_prevention_ineligibility_reasons",
            "task_success_ineligibility_reasons",
            "task_preservation_ineligibility_reasons",
            "task_rescue_ineligibility_reasons",
        ):
            self.assertIn(
                "psf_realized_no_nonzero_arm_joint_motion",
                pair[field],
            )
        self.assertEqual(
            pair["psf_continuous_motion_diagnostics"][
                "measured_joint_motion_integral_rad"
            ],
            0.0,
        )

    def test_valid_two_arm_run_is_deeply_validated(self):
        with tempfile.TemporaryDirectory() as directory:
            run = _build_complete_run(Path(directory))
            validated = validate_complete_run_artifacts(run["root"])
            self.assertEqual(set(validated["results"]), set(ACTIVE_ARMS))
            self.assertEqual(validated["receipt"]["status"], "complete")
            self.assertEqual(
                validated["pair"]["schema_version"],
                "vlsa_poisson_staged_two_arm_canary_pair.v2",
            )

    def test_hashed_candidate_receipt_validates_before_final_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            run = _build_complete_run(Path(directory))
            candidate = load_hashed_json(run["receipt"])
            run["receipt"].unlink()
            validated = validate_complete_run_artifacts(
                run["root"], candidate_receipt=candidate
            )
            self.assertEqual(validated["receipt"], candidate)
            self.assertFalse(run["receipt"].exists())

    def test_bad_inventory_path_and_hash_are_rejected(self):
        mutations = {
            "inventory": lambda receipt: receipt["arm_results"].update(
                {"unexpected_arm": copy.deepcopy(next(iter(receipt["arm_results"].values())))}
            ),
            "path": lambda receipt: receipt["arm_results"][
                "joint_velocity_adapter_only"
            ].update({"relative_path": "../escaped/result.json"}),
            "hash": lambda receipt: receipt["arm_results"][
                "joint_velocity_psf_link56"
            ].update({"sha256": "f" * 64}),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                run = _build_complete_run(Path(directory))
                _rewrite_hashed_json(run["receipt"], mutate)
                with self.assertRaises(ArtifactContractError):
                    validate_complete_run_artifacts(run["root"])

    def test_receipt_shape_identity_binding_and_path_containment_are_enforced(self):
        receipt_mutations = {
            "extra_receipt_field": lambda receipt: receipt.update(
                {"unregistered": True}
            ),
            "extra_identity_field": lambda receipt: receipt["identity"].update(
                {"unregistered": "f" * 64}
            ),
            "cross_bound_checkpoint": lambda receipt: receipt["identity"].update(
                {"checkpoint_tree_sha256": "8" * 64}
            ),
            "unsafe_case_component": lambda receipt: (
                receipt.update({"case_id": "../outside"}),
                receipt["identity"].update({"case_id": "../outside"}),
            ),
        }
        for label, mutate in receipt_mutations.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                run = _build_complete_run(Path(directory))
                _rewrite_hashed_json(run["receipt"], mutate)
                with self.assertRaises(ArtifactContractError):
                    validate_complete_run_artifacts(run["root"])

        with tempfile.TemporaryDirectory() as directory:
            run = _build_complete_run(Path(directory))
            case_root = run["root"] / CASE_ID
            escaped_case_root = Path(directory) / "escaped-case"
            case_root.rename(escaped_case_root)
            case_root.symlink_to(escaped_case_root, target_is_directory=True)
            with self.assertRaises(ArtifactContractError):
                validate_complete_run_artifacts(run["root"])

    def test_forged_pair_is_rejected_even_when_rehashed_and_reinventoried(self):
        with tempfile.TemporaryDirectory() as directory:
            run = _build_complete_run(Path(directory))
            _rewrite_hashed_json(
                run["pair"],
                lambda pair: pair.update(
                    {
                        "paired_fixed_exposure_complete": not pair[
                            "paired_fixed_exposure_complete"
                        ]
                    }
                ),
            )
            _refresh_receipt_file_hashes(run)
            with self.assertRaisesRegex(ArtifactContractError, "pair_result"):
                validate_complete_run_artifacts(run["root"])

    def test_fully_rehashed_nested_trace_tamper_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            run = _build_complete_run(Path(directory))
            arm = "joint_velocity_psf_link56"
            trace_path = run["traces"][arm]
            result_path = run["results"][arm]
            _rewrite_hashed_json(
                trace_path,
                lambda trace: trace["outcome"]["physics_trace"][0].update(
                    {
                        "measured_arm_joint_velocity_rad_s": [
                            0.123,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                        ]
                    }
                ),
            )

            def bind_new_trace(result):
                reference = result["artifact_references"][0]
                reference["bytes"] = trace_path.stat().st_size
                reference["sha256"] = sha256_file(trace_path)

            _rewrite_hashed_json(result_path, bind_new_trace)
            adapter = load_hashed_json(
                run["results"]["joint_velocity_adapter_only"]
            )
            psf = load_hashed_json(result_path)
            run["pair"].unlink()
            publish_hashed_json(run["pair"], _pair_payload(adapter, psf))
            _refresh_receipt_file_hashes(run)

            with self.assertRaises(ArtifactContractError):
                validate_complete_run_artifacts(run["root"])

    def test_additive_run_root_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            run = _build_complete_run(Path(directory))
            completed = subprocess.run(
                [
                    sys.executable,
                    str(VALIDATOR),
                    "--run-root",
                    str(run["root"]),
                    "--expected-source-action-count",
                    "2",
                ],
                cwd=str(ROOT),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            output = json.loads(completed.stdout)
            self.assertEqual(output["status"], "valid")
            self.assertEqual(output["arms"], list(ACTIVE_ARMS))
            self.assertEqual(output["pair_recomputation"], "passed")

    def test_legacy_positional_result_cli_remains_available(self):
        with tempfile.TemporaryDirectory() as directory:
            run = _build_complete_run(Path(directory))
            result_path = run["results"]["joint_velocity_psf_link56"]
            completed = subprocess.run(
                [
                    sys.executable,
                    str(VALIDATOR),
                    str(result_path),
                    "--artifact-root",
                    str(run["root"]),
                    "--expected-source-action-count",
                    "2",
                ],
                cwd=str(ROOT),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            output = json.loads(completed.stdout)
            self.assertEqual(output["status"], "valid")
            self.assertEqual(output["arm"], "joint_velocity_psf_link56")
            self.assertEqual(output["active_arm_trace_deep_validation"], "passed")

            inferred = subprocess.run(
                [
                    sys.executable,
                    str(VALIDATOR),
                    str(result_path),
                    "--expected-source-action-count",
                    "2",
                ],
                cwd=str(ROOT),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.assertEqual(inferred.returncode, 0, inferred.stderr)
            self.assertEqual(json.loads(inferred.stdout)["status"], "valid")


if __name__ == "__main__":
    unittest.main()
