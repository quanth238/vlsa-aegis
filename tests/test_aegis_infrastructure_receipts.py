from __future__ import annotations

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


class AegisInfrastructureReceiptTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

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

    def test_complete_paired_canary_receipt_revalidates_nested_evidence(
        self,
    ) -> None:
        run_root = self.root / "canary-run"
        task_root = run_root / "tasks/task-0"
        output_root = task_root / "results"
        case_id = "case-0"
        result_paths = (
            output_root / "pi05" / case_id / "result.json",
            output_root / "aegis" / case_id / "result.json",
        )
        for result_path in result_paths:
            result_path.parent.mkdir(parents=True, exist_ok=True)
            result_path.write_text("{}\n", encoding="utf-8")
        label_path = self.root / "labels.jsonl"
        label_path.write_text("{}\n", encoding="utf-8")
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
            "case_ordinal": "0",
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
        valid_aegis_result = _valid_aegis_integration_result()
        integration_gate = artifacts.validate_aegis_canary_integration(
            valid_aegis_result
        )
        result_records = [
            {
                "path": str(result_paths[0]),
                "sha256": "5" * 64,
                "status": "complete",
            },
            {
                "path": str(result_paths[1]),
                "sha256": "6" * 64,
                "status": "complete",
            },
        ]
        receipt = {
            "schema_version": artifacts.CANARY_SCHEMA,
            "status": "validated",
            "scientific_result": False,
            "paired_result_valid": True,
            "aegis_integration_gate": integration_gate,
            "source_git_commit": "release-commit",
            "pi05_tree_sha256": "2" * 64,
            "groundingdino_device": "cpu",
            "run_id": "canary-run",
            "case_id": case_id,
            "case_ordinal": 0,
            "slurm": {
                "job_id": "200",
                "array_job_id": "200",
                "array_task_id": "0",
                "host": "worker-1",
            },
            "run_contract": contract_record,
            "pi05_hash_receipt": hash_record,
            "allocation_preflight": preflight_record,
            "results": result_records,
        }
        receipt["receipt_payload_sha256"] = sha256_bytes(
            canonical_json_bytes(receipt)
        )
        receipt_path = run_root / "paired-canary-validation.json"
        write_json_exclusive(receipt_path, receipt)
        validated_results = (
            ({"arm": "pi05-arm"}, result_records[0]),
            (valid_aegis_result, result_records[1]),
        )
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
                "_validate_result_artifact",
                side_effect=validated_results,
            ),
            mock.patch.object(artifacts.aggregate, "validate_pairs") as pairs,
        ):
            record = artifacts.validate_paired_canary_receipt(
                receipt_path,
                expected_file_sha256=sha256_path(receipt_path),
                expected_commit="release-commit",
                expected_tree_sha256="2" * 64,
                expected_groundingdino_device="cpu",
                config_path=self.root / "config.json",
                manifest_path=self.root / "manifest.jsonl",
                manifest_receipt_path=self.root / "manifest-receipt.json",
                config={"arms": ["pi05-arm", "aegis-arm"]},
                manifests=[{"case_id": case_id}],
            )
        self.assertEqual(record["run_contract_sha256"], "3" * 64)
        self.assertEqual(
            record["result_artifact_sha256"], ["5" * 64, "6" * 64]
        )
        pairs.assert_called_once()

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
