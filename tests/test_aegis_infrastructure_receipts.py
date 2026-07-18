from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts.aegis_receipt_utils import (
    ReceiptError,
    canonical_json_bytes,
    sha256_bytes,
    sha256_path,
    write_json_exclusive,
)
from scripts import compute_pi05_tree_receipt as checkpoint_receipt
from scripts import validate_aegis_assets as asset_validation
from scripts import validate_aegis_run_artifacts as artifacts


def _valid_aegis_integration_result() -> dict:
    return {
        "arm": "aegis-arm",
        "status": "complete",
        "obstacle": {"selector_matches_active": True},
        "perception": {"status": "ready"},
        "actions": [
            {
                "step": 0,
                "control_path": "aegis_qp",
                "qp": {
                    "solver": "OSQP",
                    "solver_status": "optimal",
                    "objective": 0.25,
                    "barrier_h": 0.1,
                    "constraint_lhs": 0.2,
                    "u_solution": [0.0] * 6,
                    "z_after": [1.0, 0.0, 0.0],
                },
            }
        ],
    }


def _canary_bound_result(
    *,
    policy_mode: str,
    label_record: dict,
) -> dict:
    if policy_mode == "aegis":
        result = _valid_aegis_integration_result()
    else:
        result = {"status": "complete"}
    result.update(
        {
            "mode": policy_mode,
            "arm": artifacts.action_canary.EXPECTED_ARM_BY_POLICY_MODE[
                policy_mode
            ],
            "settled_observation": {
                "label_record": copy.deepcopy(label_record),
            },
        }
    )
    return result


class AegisInfrastructureReceiptTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _valid_evaluation_preflight(self) -> tuple[dict, dict, dict]:
        tree_sha256 = "1" * 64
        filesystem_sha256 = "2" * 64
        contract = {
            "git_commit": "release-commit",
            "config_sha256": "3" * 64,
            "manifest_sha256": "4" * 64,
            "manifest_receipt_sha256": "5" * 64,
            "pi05_tree_sha256": tree_sha256,
            "pi05_hash_receipt_sha256": "6" * 64,
            "label_manifest_sha256": "7" * 64,
            "label_publication_receipt_sha256": "8" * 64,
            "groundingdino_device": "cpu",
            "case_ordinal": str(artifacts.CANARY_CASE_ORDINAL),
        }
        slurm = {
            "job_id": "200",
            "array_job_id": "200",
            "array_task_id": "0",
            "host": "worker-1",
        }
        receipt = {
            "schema_version": artifacts.PREFLIGHT_SCHEMA,
            "status": "passed",
            "scientific_result": False,
            "profile": "evaluation",
            "source": {
                "git_commit": contract["git_commit"],
                "git_dirty": False,
            },
            "protocol": {
                "config": {"sha256": contract["config_sha256"]},
                "manifest": {"sha256": contract["manifest_sha256"]},
                "receipt": {
                    "sha256": contract["manifest_receipt_sha256"]
                },
            },
            "assets": {
                "policy_model_executed": False,
                "groundingdino_executed": False,
                "qp_executed": False,
                "pi05_checkpoint": {
                    "full_content_hash_verified": True,
                    "full_content_tree_sha256": tree_sha256,
                    "full_content_hash_verification": (
                        "one_time_allocation_receipt_plus_exact_stat_identity"
                    ),
                    "full_content_rehashed_in_this_allocation": False,
                    "filesystem_identity_sha256": filesystem_sha256,
                },
                "pi05_hash_receipt": {
                    "sha256": contract["pi05_hash_receipt_sha256"],
                    "full_content_hash_verified": True,
                    "full_content_tree_sha256": tree_sha256,
                    "filesystem_identity_sha256": filesystem_sha256,
                },
                "frozen_codex_labels": {
                    "sha256": contract["label_manifest_sha256"],
                    "rows": 1600,
                    "required_cases": 1,
                    "all_population_cases_bound": True,
                },
                "label_publication_receipt": {
                    "sha256": contract[
                        "label_publication_receipt_sha256"
                    ],
                    "labels_sha256": contract["label_manifest_sha256"],
                    "case_count": 1600,
                    "outcome_blind": True,
                },
                "groundingdino": {"device": "cpu"},
                "video_runtime": {
                    "aegis_python": {
                        "path": str(
                            asset_validation.DEFAULT_AEGIS_PYTHON
                        ),
                        "resolved_path": str(
                            asset_validation.DEFAULT_AEGIS_PYTHON_RESOLVED
                        ),
                        "python_version": (
                            asset_validation.AEGIS_PYTHON_VERSION
                        ),
                    },
                    "imageio": {
                        "version": asset_validation.IMAGEIO_VERSION
                    },
                    "imageio_ffmpeg": {
                        "version": (
                            asset_validation.IMAGEIO_FFMPEG_VERSION
                        )
                    },
                    "ffmpeg": {
                        "path": str(
                            asset_validation.DEFAULT_IMAGEIO_FFMPEG_EXE
                        ),
                        "resolved_path": str(
                            asset_validation.DEFAULT_IMAGEIO_FFMPEG_EXE
                        ),
                        "sha256": (
                            asset_validation.IMAGEIO_FFMPEG_SHA256
                        ),
                        "executable": True,
                    },
                },
            },
            "slurm": slurm,
        }
        return receipt, contract, slurm

    def test_preflight_revalidation_binds_exact_python_runtime(self) -> None:
        receipt, contract, slurm = self._valid_evaluation_preflight()

        def publish(name: str, value: dict) -> Path:
            value["receipt_payload_sha256"] = sha256_bytes(
                canonical_json_bytes(value)
            )
            path = self.root / name
            write_json_exclusive(path, value)
            return path

        valid_path = publish("valid-preflight.json", copy.deepcopy(receipt))
        artifacts.validate_preflight_receipt(
            valid_path,
            contract=contract,
            expected_slurm=slurm,
        )
        mutations = {
            "resolved-python": (
                "resolved_path",
                "/tmp/replaced-python",
                "resolved AEGIS interpreter",
            ),
            "python-version": (
                "python_version",
                "3.8.19",
                "AEGIS Python version",
            ),
        }
        for name, (field, changed, message) in mutations.items():
            with self.subTest(name=name):
                corrupted = copy.deepcopy(receipt)
                corrupted["assets"]["video_runtime"]["aegis_python"][
                    field
                ] = changed
                path = publish(f"{name}.json", corrupted)
                with self.assertRaisesRegex(ReceiptError, message):
                    artifacts.validate_preflight_receipt(
                        path,
                        contract=contract,
                        expected_slurm=slurm,
                    )

    def test_run_contract_rejects_non_cpu_groundingdino(self) -> None:
        run_root = self.root / "run"
        run_root.mkdir()
        inputs = {
            "config_sha256": self.root / "config.json",
            "manifest_sha256": self.root / "manifest.jsonl",
            "manifest_receipt_sha256": self.root / "manifest-receipt.json",
            "label_manifest_sha256": self.root / "labels.jsonl",
        }
        for path in inputs.values():
            path.write_bytes(path.name.encode("utf-8"))
        fields = {
            "schema_version": artifacts.RUN_CONTRACT_SCHEMA,
            "run_id": "canary",
            "run_stage": "paired-canary",
            "case_ordinal": str(artifacts.CANARY_CASE_ORDINAL),
            "git_commit": "release-commit",
            **{
                field: sha256_path(path)
                for field, path in inputs.items()
            },
            "label_publication_receipt_sha256": (
                asset_validation.LABEL_PUBLICATION_RECEIPT_SHA256
            ),
            "pi05_tree_sha256": "a" * 64,
            "pi05_hash_receipt_sha256": "b" * 64,
            "paired_canary_receipt_sha256": "none",
            "groundingdino_device": "cuda",
        }
        (run_root / "run-contract.tsv").write_text(
            "".join(f"{key}\t{value}\n" for key, value in fields.items()),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(
            ReceiptError,
            "frozen GroundingDINO device",
        ):
            artifacts.validate_run_contract(
                run_root=run_root,
                expected_stage="paired-canary",
                expected_commit="release-commit",
                config_path=inputs["config_sha256"],
                manifest_path=inputs["manifest_sha256"],
                manifest_receipt_path=inputs[
                    "manifest_receipt_sha256"
                ],
                label_manifest_path=inputs["label_manifest_sha256"],
            )

    def test_checkpoint_hash_requires_exact_compute_allocation(self) -> None:
        valid = {
            "SLURM_JOB_ID": "101",
            "SLURM_ARRAY_JOB_ID": "101",
            "SLURM_ARRAY_TASK_ID": "0",
            "SLURMD_NODENAME": "worker-1",
        }
        self.assertEqual(
            checkpoint_receipt.allocation_identity(valid)["host"],
            "worker-1",
        )
        for changed in (
            {"SLURM_ARRAY_TASK_ID": "1"},
            {"SLURMD_NODENAME": "worker-3"},
            {"SLURMD_NODENAME": "login"},
            {"SLURM_JOB_ID": ""},
        ):
            environment = {**valid, **changed}
            with self.subTest(changed=changed):
                with self.assertRaises(ReceiptError):
                    checkpoint_receipt.allocation_identity(environment)

    def test_paired_canary_receipt_rejects_payload_tampering(self) -> None:
        tree_sha256 = "a" * 64
        receipt = {
            "schema_version": artifacts.CANARY_SCHEMA,
            "status": "validated",
            "scientific_result": False,
            "paired_result_valid": True,
            "source_git_commit": "release-commit",
            "pi05_tree_sha256": tree_sha256,
            "groundingdino_device": "cpu",
            "run_id": "canary-a",
            "case_id": "case-0",
            "case_ordinal": 0,
            "slurm": {
                "job_id": "200",
                "array_job_id": "200",
                "array_task_id": "0",
                "host": "worker-1",
            },
        }
        receipt["receipt_payload_sha256"] = sha256_bytes(
            canonical_json_bytes(receipt)
        )
        path = self.root / "paired-canary.json"
        write_json_exclusive(path, receipt)
        with self.assertRaisesRegex(
            ReceiptError, "nested validation evidence"
        ):
            artifacts.validate_paired_canary_receipt(
                path,
                expected_file_sha256=sha256_path(path),
                expected_commit="release-commit",
                expected_tree_sha256=tree_sha256,
                expected_groundingdino_device="cpu",
                config_path=self.root / "config.json",
                manifest_path=self.root / "manifest.jsonl",
                manifest_receipt_path=self.root / "manifest-receipt.json",
                config={},
                manifests=[],
            )

        tampered = json.loads(path.read_text(encoding="utf-8"))
        tampered["case_id"] = "changed-after-validation"
        tampered_path = self.root / "tampered-canary.json"
        tampered_path.write_text(
            json.dumps(tampered, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        with self.assertRaises(ReceiptError):
            artifacts.validate_paired_canary_receipt(
                tampered_path,
                expected_file_sha256=sha256_path(tampered_path),
                expected_commit="release-commit",
                expected_tree_sha256=tree_sha256,
                expected_groundingdino_device="cpu",
                config_path=self.root / "config.json",
                manifest_path=self.root / "manifest.jsonl",
                manifest_receipt_path=self.root / "manifest-receipt.json",
                config={},
                manifests=[],
            )

    def test_aegis_canary_integration_gate_requires_real_qp_execution(
        self,
    ) -> None:
        valid = _valid_aegis_integration_result()
        gate = artifacts.validate_aegis_canary_integration(valid)
        self.assertEqual(gate["status"], "passed")
        self.assertTrue(gate["selector_match"])
        self.assertEqual(gate["executed_aegis_qp_actions"], 1)

        invalid_results: dict[str, dict] = {}
        selector_mismatch = copy.deepcopy(valid)
        selector_mismatch["obstacle"]["selector_matches_active"] = False
        invalid_results["selector_mismatch"] = selector_mismatch

        fail_open = copy.deepcopy(valid)
        fail_open["perception"] = {
            "status": "empty",
            "method_failure": "no_grounded_points",
        }
        fail_open["actions"][0]["control_path"] = (
            artifacts.aggregate.TRANSLATIONAL_FAIL_OPEN
        )
        fail_open["actions"][0]["qp"] = None
        invalid_results["fail_open"] = fail_open

        step_zero_failure = copy.deepcopy(valid)
        step_zero_failure["status"] = "method_failure"
        step_zero_failure["actions"] = []
        step_zero_failure["method_failure"] = {
            "status": "method_failure",
            "phase": "precontrol",
            "step": 0,
        }
        invalid_results["step_zero_failure"] = step_zero_failure

        no_qp = copy.deepcopy(valid)
        no_qp["actions"][0]["control_path"] = (
            artifacts.aggregate.TRANSLATIONAL_FAIL_OPEN
        )
        no_qp["actions"][0]["qp"] = None
        invalid_results["no_qp"] = no_qp

        nonfinite_qp = copy.deepcopy(valid)
        nonfinite_qp["actions"][0]["qp"]["objective"] = float("nan")
        invalid_results["nonfinite_qp"] = nonfinite_qp

        invalid_status = copy.deepcopy(valid)
        invalid_status["actions"][0]["qp"]["solver_status"] = "infeasible"
        invalid_results["invalid_qp_status"] = invalid_status

        for label, result in invalid_results.items():
            with self.subTest(label=label):
                with self.assertRaisesRegex(
                    ReceiptError, "integration is inconclusive"
                ):
                    artifacts.validate_aegis_canary_integration(result)

    def test_canary_requires_canonical_action_reference_fixture(self) -> None:
        canonical = (
            artifacts.ROOT
            / "fixtures/vlsa_table1_canary_action_reference.json"
        )
        self.assertEqual(
            artifacts._validate_canary_action_reference_path(canonical),
            canonical.resolve(),
        )
        copied = self.root / canonical.name
        copied.write_bytes(canonical.read_bytes())
        with self.assertRaisesRegex(
            ReceiptError,
            "canonical action-reference fixture",
        ):
            artifacts._validate_canary_action_reference_path(copied)

    def test_canary_result_binding_rejects_identity_and_label_mutations(
        self,
    ) -> None:
        label_record = {
            "case_id": artifacts.CANARY_CASE_ID,
            "obstacle_label": "blue moka pot",
        }
        valid = _canary_bound_result(
            policy_mode="pi05",
            label_record=label_record,
        )
        artifacts._validate_canary_result_binding(
            valid,
            diagnostics_mode="diagnostics-off",
            policy_mode="pi05",
            expected_arm=(
                artifacts.action_canary.EXPECTED_ARM_BY_POLICY_MODE["pi05"]
            ),
            expected_label_record=label_record,
        )
        mutations = {
            "mode": lambda result: result.__setitem__("mode", "aegis"),
            "arm": lambda result: result.__setitem__(
                "arm",
                artifacts.action_canary.EXPECTED_ARM_BY_POLICY_MODE["aegis"],
            ),
            "label": lambda result: result["settled_observation"][
                "label_record"
            ].__setitem__("obstacle_label", "red mug"),
        }
        for label, mutate in mutations.items():
            with self.subTest(change=label):
                changed = copy.deepcopy(valid)
                mutate(changed)
                with self.assertRaises(ReceiptError):
                    artifacts._validate_canary_result_binding(
                        changed,
                        diagnostics_mode="diagnostics-off",
                        policy_mode="pi05",
                        expected_arm=(
                            artifacts.action_canary
                            .EXPECTED_ARM_BY_POLICY_MODE["pi05"]
                        ),
                        expected_label_record=label_record,
                    )

    def test_live_canary_rejects_mutated_settled_label_record(self) -> None:
        run_root = self.root / "live-canary"
        task_root = run_root / "tasks/task-0"
        output_root = task_root / "results"
        case_id = artifacts.CANARY_CASE_ID
        for diagnostics_mode in artifacts.action_canary.DIAGNOSTIC_MODES:
            for policy_mode in artifacts.action_canary.POLICY_MODES:
                result_path = (
                    output_root
                    / diagnostics_mode
                    / policy_mode
                    / case_id
                    / "result.json"
                )
                result_path.parent.mkdir(parents=True, exist_ok=True)
                result_path.write_text("{}\n", encoding="utf-8")
        label_record = {
            "case_id": case_id,
            "obstacle_label": "blue moka pot",
        }
        label_path = self.root / "live-labels.jsonl"
        label_path.write_text(
            json.dumps(label_record, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        changed = _canary_bound_result(
            policy_mode="pi05",
            label_record=label_record,
        )
        changed["settled_observation"]["label_record"][
            "obstacle_label"
        ] = "red mug"
        args = argparse.Namespace(
            run_root=run_root,
            config=self.root / "config.json",
            manifest=self.root / "manifest.jsonl",
            manifest_receipt=self.root / "manifest-receipt.json",
            labels=label_path,
            expected_commit="release-commit",
            case_ordinal=None,
            pi05_hash_receipt=self.root / "pi05-hash.json",
            action_reference=(
                artifacts.ROOT
                / "fixtures/vlsa_table1_canary_action_reference.json"
            ),
        )
        config = {
            "arms": [
                artifacts.action_canary.EXPECTED_ARM_BY_POLICY_MODE["pi05"],
                artifacts.action_canary.EXPECTED_ARM_BY_POLICY_MODE["aegis"],
            ]
        }
        manifest = {"case_id": case_id}
        manifests = [
            {"case_id": f"unused-{index}"}
            for index in range(artifacts.CANARY_CASE_ORDINAL)
        ] + [manifest]
        contract = {
            "case_ordinal": str(artifacts.CANARY_CASE_ORDINAL),
            "pi05_hash_receipt_sha256": "1" * 64,
            "pi05_tree_sha256": "2" * 64,
        }
        slurm = {
            "job_id": "200",
            "array_job_id": "200",
            "array_task_id": "0",
            "host": "worker-1",
        }
        with (
            mock.patch.object(
                artifacts,
                "validate_run_contract",
                return_value=(contract, {}),
            ),
            mock.patch.object(
                artifacts.aggregate,
                "load_protocol",
                return_value=(config, manifests),
            ),
            mock.patch.object(
                artifacts,
                "validate_full_label_manifest",
                return_value={},
            ),
            mock.patch.object(
                artifacts,
                "allocation_identity",
                return_value=slurm,
            ),
            mock.patch.object(
                artifacts,
                "validate_preflight_receipt",
                return_value={},
            ),
            mock.patch.object(
                artifacts,
                "validate_pi05_hash_receipt",
                return_value={},
            ),
            mock.patch.object(
                artifacts,
                "_validate_result_artifact",
                return_value=(changed, {}),
            ),
        ):
            with self.assertRaisesRegex(
                ReceiptError,
                "canonical frozen label-record bytes",
            ):
                artifacts.validate_paired_canary(args)

    def test_full_label_manifest_requires_exact_canary_row(self) -> None:
        source_root = self.root / "source"
        labels_dir = source_root / "labels"
        labels_dir.mkdir(parents=True)
        canary_row = {
            "case_id": artifacts.CANARY_CASE_ID,
            "obstacle_label": "blue moka pot",
        }
        canary_bytes = (
            json.dumps(canary_row, sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode("utf-8")
        canary_path = labels_dir / "vlsa_table1_canary_labels.jsonl"
        canary_path.write_bytes(canary_bytes)
        rows = [
            {
                "case_id": f"case-{index:04d}",
                "obstacle_label": "red mug",
            }
            for index in range(1599)
        ]
        rows.insert(artifacts.CANARY_CASE_ORDINAL, canary_row)
        full_path = self.root / "labels.jsonl"
        full_path.write_bytes(
            b"".join(
                (
                    json.dumps(
                        row,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                ).encode("utf-8")
                for row in rows
            )
        )
        with (
            mock.patch.object(artifacts, "ROOT", source_root),
            mock.patch.object(
                artifacts,
                "FULL_LABEL_MANIFEST_SHA256",
                sha256_path(full_path),
            ),
            mock.patch.object(
                artifacts,
                "CANARY_LABEL_ROW_SHA256",
                sha256_path(canary_path),
            ),
        ):
            record = artifacts.validate_full_label_manifest(
                full_path,
                canary_case_id=artifacts.CANARY_CASE_ID,
            )
            self.assertEqual(record["rows"], 1600)
            self.assertTrue(record["canary_row_byte_identical"])
            canary_path.write_bytes(canary_bytes.replace(b"blue", b"navy"))
            with self.assertRaisesRegex(
                ReceiptError,
                "one-row canary label SHA-256",
            ):
                artifacts.validate_full_label_manifest(
                    full_path,
                    canary_case_id=artifacts.CANARY_CASE_ID,
                )

    def test_complete_paired_canary_receipt_revalidates_nested_evidence(
        self,
    ) -> None:
        run_root = self.root / "canary-run"
        task_root = run_root / "tasks/task-0"
        output_root = task_root / "results"
        case_id = artifacts.CANARY_CASE_ID
        result_paths = (
            output_root
            / "diagnostics-off"
            / "pi05"
            / case_id
            / "result.json",
            output_root
            / "diagnostics-off"
            / "aegis"
            / case_id
            / "result.json",
            output_root
            / "diagnostics-on"
            / "pi05"
            / case_id
            / "result.json",
            output_root
            / "diagnostics-on"
            / "aegis"
            / case_id
            / "result.json",
        )
        for result_path in result_paths:
            result_path.parent.mkdir(parents=True, exist_ok=True)
            result_path.write_text("{}\n", encoding="utf-8")
        label_record = {
            "case_id": case_id,
            "obstacle_label": "blue moka pot",
        }
        label_path = self.root / "labels.jsonl"
        label_path.write_text(
            json.dumps(label_record, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        preflight_path = task_root / "allocation-preflight.json"
        preflight_path.write_text(
            json.dumps(
                {
                    "assets": {
                        "frozen_codex_labels": {
                            "path": str(label_path),
                        }
                    }
                }
            )
            + "\n",
            encoding="utf-8",
        )
        contract_path = run_root / "run-contract.tsv"
        contract_path.write_text("placeholder\n", encoding="utf-8")
        hash_path = self.root / "pi05-hash.json"
        hash_path.write_text("{}\n", encoding="utf-8")
        contract = {
            "run_id": "canary-run",
            "case_ordinal": str(artifacts.CANARY_CASE_ORDINAL),
            "pi05_hash_receipt_sha256": "1" * 64,
            "pi05_tree_sha256": "2" * 64,
        }
        contract_record = {
            "path": str(contract_path),
            "sha256": "3" * 64,
        }
        hash_record = {
            "path": str(hash_path),
            "sha256": "1" * 64,
        }
        preflight_record = {
            "path": str(preflight_path),
            "sha256": "4" * 64,
        }
        full_label_record = {
            "path": str(label_path),
            "sha256": artifacts.FULL_LABEL_MANIFEST_SHA256,
            "rows": 1600,
            "unique_case_ids": 1600,
            "canary_case_id": case_id,
            "canary_row_sha256": artifacts.CANARY_LABEL_ROW_SHA256,
            "canary_row_byte_identical": True,
        }
        valid_aegis_result = _canary_bound_result(
            policy_mode="aegis",
            label_record=label_record,
        )
        valid_pi05_result = _canary_bound_result(
            policy_mode="pi05",
            label_record=label_record,
        )
        integration_gate = artifacts.validate_aegis_canary_integration(
            valid_aegis_result
        )
        result_records = [
            {
                "path": str(result_paths[0]),
                "sha256": "5" * 64,
                "status": "complete",
                "diagnostics_mode": "diagnostics-off",
                "policy_mode": "pi05",
            },
            {
                "path": str(result_paths[1]),
                "sha256": "6" * 64,
                "status": "complete",
                "diagnostics_mode": "diagnostics-off",
                "policy_mode": "aegis",
            },
            {
                "path": str(result_paths[2]),
                "sha256": "7" * 64,
                "status": "complete",
                "diagnostics_mode": "diagnostics-on",
                "policy_mode": "pi05",
            },
            {
                "path": str(result_paths[3]),
                "sha256": "8" * 64,
                "status": "complete",
                "diagnostics_mode": "diagnostics-on",
                "policy_mode": "aegis",
            },
        ]
        action_evidence = {
            "schema_version": (
                "vlsa_table1_action_invariant_canary_evidence.v1"
            ),
            "status": "validated",
        }
        receipt = {
            "schema_version": artifacts.CANARY_SCHEMA,
            "status": "validated",
            "scientific_result": False,
            "paired_result_valid": True,
            "action_invariance_valid": True,
            "failure_diagnostics_valid": True,
            "cross_arm_pairing": {
                "diagnostics-off": True,
                "diagnostics-on": True,
            },
            "aegis_integration_gates": {
                "diagnostics-off": integration_gate,
                "diagnostics-on": integration_gate,
            },
            "action_invariant_evidence": action_evidence,
            "source_git_commit": "release-commit",
            "pi05_tree_sha256": "2" * 64,
            "groundingdino_device": "cpu",
            "run_id": "canary-run",
            "case_id": case_id,
            "case_ordinal": artifacts.CANARY_CASE_ORDINAL,
            "slurm": {
                "job_id": "200",
                "array_job_id": "200",
                "array_task_id": "0",
                "host": "worker-1",
            },
            "run_contract": contract_record,
            "pi05_hash_receipt": hash_record,
            "allocation_preflight": preflight_record,
            "full_label_manifest": full_label_record,
            "results": result_records,
        }
        receipt["receipt_payload_sha256"] = sha256_bytes(
            canonical_json_bytes(receipt)
        )
        receipt_path = run_root / "paired-canary-validation.json"
        write_json_exclusive(receipt_path, receipt)
        validated_results = (
            (valid_pi05_result, result_records[0]),
            (valid_aegis_result, result_records[1]),
            (valid_pi05_result, result_records[2]),
            (valid_aegis_result, result_records[3]),
        )
        config = {
            "arms": [
                artifacts.action_canary.EXPECTED_ARM_BY_POLICY_MODE["pi05"],
                artifacts.action_canary.EXPECTED_ARM_BY_POLICY_MODE["aegis"],
            ]
        }
        manifests = [
            {"case_id": f"unused-{index}"}
            for index in range(artifacts.CANARY_CASE_ORDINAL)
        ] + [{"case_id": case_id}]

        def revalidate(
            result_side_effect,
        ) -> tuple[dict, int]:
            with (
                mock.patch.object(
                    artifacts,
                    "validate_run_contract",
                    return_value=(contract, contract_record),
                ),
                mock.patch.object(
                    artifacts,
                    "validate_pi05_hash_receipt",
                    return_value=hash_record,
                ),
                mock.patch.object(
                    artifacts,
                    "validate_preflight_receipt",
                    return_value=preflight_record,
                ),
                mock.patch.object(
                    artifacts,
                    "validate_full_label_manifest",
                    return_value=full_label_record,
                ),
                mock.patch.object(
                    artifacts,
                    "_validate_result_artifact",
                    side_effect=result_side_effect,
                ),
                mock.patch.object(
                    artifacts.aggregate, "validate_pairs"
                ) as pairs,
                mock.patch.object(
                    artifacts.action_canary,
                    "validate_four_run_canary",
                    return_value=action_evidence,
                ),
            ):
                record = artifacts.validate_paired_canary_receipt(
                    receipt_path,
                    expected_file_sha256=sha256_path(receipt_path),
                    expected_commit="release-commit",
                    expected_tree_sha256="2" * 64,
                    expected_groundingdino_device="cpu",
                    config_path=self.root / "config.json",
                    manifest_path=self.root / "manifest.jsonl",
                    manifest_receipt_path=(
                        self.root / "manifest-receipt.json"
                    ),
                    config=config,
                    manifests=manifests,
                )
                return record, pairs.call_count

        record, pair_call_count = revalidate(validated_results)
        self.assertEqual(record["run_contract_sha256"], "3" * 64)
        self.assertEqual(
            record["result_artifact_sha256"],
            ["5" * 64, "6" * 64, "7" * 64, "8" * 64],
        )
        self.assertEqual(pair_call_count, 2)

        mutated_pi05 = copy.deepcopy(valid_pi05_result)
        mutated_pi05["settled_observation"]["label_record"][
            "obstacle_label"
        ] = "red mug"
        with self.assertRaisesRegex(
            ReceiptError,
            "canonical frozen label-record bytes",
        ):
            revalidate(
                (
                    (mutated_pi05, result_records[0]),
                    (valid_aegis_result, result_records[1]),
                    (valid_pi05_result, result_records[2]),
                    (valid_aegis_result, result_records[3]),
                )
            )

    def test_checkpoint_receipt_rejects_forbidden_host(self) -> None:
        tree_sha256 = "b" * 64
        checkpoint = self.root / "pi05"
        checkpoint.mkdir()
        (checkpoint / "meta").write_bytes(b"meta")
        (checkpoint / "data").write_bytes(b"data")
        with (
            mock.patch.dict(
                asset_validation.PI05_METADATA_FILES,
                {"meta": (4, sha256_bytes(b"meta"))},
                clear=True,
            ),
            mock.patch.dict(
                asset_validation.PI05_DATA_FILES,
                {"data": 4},
                clear=True,
            ),
        ):
            filesystem_identity = (
                asset_validation.pi05_checkpoint_filesystem_identity(
                    checkpoint, ("meta", "data")
                )
            )
        receipt = {
            "schema_version": artifacts.PI05_HASH_SCHEMA,
            "status": "passed",
            "scientific_result": False,
            "source": {
                "git_commit": "release-commit",
                "git_dirty": False,
            },
            "checkpoint": {
                "path": str(checkpoint.resolve()),
                "full_content_hash_verified": True,
                "full_content_tree_sha256": tree_sha256,
                "filesystem_identity": filesystem_identity,
            },
            "slurm": {
                "job_id": "150",
                "array_job_id": "150",
                "array_task_id": "0",
                "host": "worker-3",
            },
            "execution": {
                "policy_model_executed": False,
                "simulator_executed": False,
                "groundingdino_executed": False,
                "qp_executed": False,
                "training_executed": False,
            },
        }
        receipt["receipt_payload_sha256"] = sha256_bytes(
            canonical_json_bytes(receipt)
        )
        path = self.root / "pi05-hash.json"
        write_json_exclusive(path, receipt)
        with (
            mock.patch.dict(
                asset_validation.PI05_METADATA_FILES,
                {"meta": (4, sha256_bytes(b"meta"))},
                clear=True,
            ),
            mock.patch.dict(
                asset_validation.PI05_DATA_FILES,
                {"data": 4},
                clear=True,
            ),
            self.assertRaises(ReceiptError),
        ):
            artifacts.validate_pi05_hash_receipt(
                path,
                expected_file_sha256=sha256_path(path),
                expected_tree_sha256=tree_sha256,
                expected_commit="release-commit",
            )

    def test_population_accounting_requires_all_32_completed_tasks(self) -> None:
        path = self.root / "sacct.txt"
        rows = [
            f"300_{task}|{400 + task}|COMPLETED"
            for task in range(32)
        ]
        rows.extend(
            (
                "300_0.batch|400.batch|COMPLETED",
                "300|300|COMPLETED",
            )
        )
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        record = artifacts.validate_slurm_accounting(
            path, population_array_job_id="300"
        )
        self.assertEqual(len(record["tasks"]), 32)
        self.assertEqual(record["ignored_non_task_rows"], 2)

        rows[7] = "300_7|407|OUT_OF_MEMORY"
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        with self.assertRaises(ReceiptError):
            artifacts.validate_slurm_accounting(
                path, population_array_job_id="300"
            )

        path.write_text("\n".join(rows[:31]) + "\n", encoding="utf-8")
        with self.assertRaises(ReceiptError):
            artifacts.validate_slurm_accounting(
                path, population_array_job_id="300"
            )

    def test_population_publisher_requires_exact_afterany_dependency(
        self,
    ) -> None:
        valid = {
            "SLURM_JOB_ID": "500",
            "SLURM_JOB_DEPENDENCY": "afterany:300",
            "SLURMD_NODENAME": "worker-2",
        }
        record = artifacts.publisher_allocation_identity(
            valid, expected_population_array_job_id="300"
        )
        self.assertEqual(record["dependency"], "afterany:300")
        for dependency in ("afterok:300", "afterany:301", ""):
            with self.subTest(dependency=dependency):
                with self.assertRaises(ReceiptError):
                    artifacts.publisher_allocation_identity(
                        {**valid, "SLURM_JOB_DEPENDENCY": dependency},
                        expected_population_array_job_id="300",
                    )

    def test_receipt_writer_never_replaces_existing_artifact(self) -> None:
        path = self.root / "receipt.json"
        write_json_exclusive(path, {"status": "first"})
        original = path.read_bytes()
        with self.assertRaises(ReceiptError):
            write_json_exclusive(path, {"status": "second"})
        self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
