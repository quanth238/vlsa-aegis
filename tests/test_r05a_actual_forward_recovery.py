from __future__ import annotations

import json
from pathlib import Path
import stat
import subprocess
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "experiments" / "r05a_actual_forward_recovery_20260716c.json"
PUBLISHER = ROOT / "main" / "publish_crfs_r05a_actual_forward_canary.py"
WRAPPER = ROOT / "scripts" / "hpc" / "validate_r05a_actual_forward_recovery.sh"
SUBMITTER = ROOT / "scripts" / "hpc" / "submit_r05a_actual_forward_recovery.sh"
SBATCH = ROOT / "slurm" / "r05a_actual_forward_recovery_cpu.sbatch"
RESULT_SCHEMA = ROOT / "schemas" / "r05a-actual-forward-recovery-envelope.schema.json"
RECEIPT_SCHEMA = ROOT / "schemas" / "r05a-actual-forward-recovery-publication-receipt.schema.json"

try:
    import numpy as np

    from publish_crfs_r05a_actual_forward_canary import (
        _commit_publication,
        _validate_source_budget,
    )

    NUMPY_AVAILABLE = True
except ModuleNotFoundError:
    NUMPY_AVAILABLE = False


class ActualForwardRecoveryContractTest(unittest.TestCase):
    def test_config_is_fail_closed_or_exactly_released_and_binds_run_c(self) -> None:
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        if config["ready_to_run"] is False:
            self.assertTrue(config["blocked_on"])
            self.assertIsNone(config["execution_release"])
            self.assertEqual(
                config["config_status"], "implemented_fail_closed_pending_review"
            )
        else:
            self.assertTrue(config["ready_to_run"])
            self.assertEqual(config["blocked_on"], [])
            self.assertEqual(config["config_status"], "released_exact_cpu_republication")
            release = config["execution_release"]
            self.assertRegex(release["accepted_implementation_commit"], r"^[0-9a-f]{40}$")
            self.assertTrue(release["single_submission"])
            self.assertTrue(release["release_only_parent_required"])
            self.assertEqual(release["resources"], config["resources"])
            self.assertFalse(release["automatic_resubmission_allowed"])
            self.assertFalse(release["automatic_next_experiment_allowed"])
            self.assertEqual(
                set(release["allowed_release_diff_paths"]),
                {
                    "configs/experiments/r05a_actual_forward_recovery_20260716c.json",
                    release["decision_artifact"],
                },
            )
        self.assertEqual(
            config["source_binding"],
            {
                "run_id": "r05a-actual-forward-cem-canary-20260716c",
                "source_git_commit": "bd14f97eeffafd20525454db4d7a52614e1146c4",
                "source_job_id": "28281",
                "source_task_id": "28281_0",
                "source_state": "COMPLETED",
                "source_exit_code": "0:0",
                "source_node": "worker-1",
                "original_publisher_job_id": "28282",
                "original_publisher_state": "FAILED",
                "original_publisher_exit_code": "1:0",
            },
        )
        self.assertEqual(
            config["immutable_artifact_sha256"]["cpu-afterany-validation.json"],
            "f8e9ef108630cad9a9691dba0827a8be68d3b15e41ee3de2e4e6cfaf03736d27",
        )
        self.assertEqual(config["resources"]["dependency"], "afterany:28282")
        self.assertEqual(config["resources"]["gpus"], 0)
        self.assertTrue(all(value is False for value in config["claims"].values()))

    def test_budget_contract_is_exact_and_tolerance_is_not_tuned(self) -> None:
        budget = json.loads(CONFIG.read_text(encoding="utf-8"))["budget_contract"]
        self.assertEqual(budget["reported_float64"], 3.6398398429065115)
        self.assertEqual(budget["recomputed_float64"], 3.639839842906512)
        self.assertEqual(budget["float32"], 3.6398398876190186)
        self.assertEqual(budget["float32_hex"], "0x4068f323")
        self.assertEqual(budget["existing_source_consistency_tolerance"], 1e-12)
        self.assertFalse(budget["tolerance_change_allowed"])

    def test_recovery_repository_key_set_is_unique_and_full_runtime_closure(self) -> None:
        paths = json.loads(CONFIG.read_text(encoding="utf-8"))[
            "recovery_repository_paths"
        ]
        self.assertEqual(len(paths), len(set(paths)))
        required = {
            "main/publish_crfs_r05a_actual_forward_canary.py",
            "main/crfs_oracle/r05a_actual_forward_validation.py",
            "main/crfs_oracle/r05a_actual_forward_search.py",
            "main/crfs_oracle/r05a_canary.py",
            "main/crfs_oracle/r02_runner.py",
            "src/crfs_harness/artifacts.py",
            "src/crfs_harness/manifest.py",
            "scripts/hpc/lib/slurm_array_task_record_identity.sh",
            "schemas/r05a-actual-forward-recovery-envelope.schema.json",
            "schemas/r05a-actual-forward-recovery-publication-receipt.schema.json",
            "scripts/hpc/validate_r05a_actual_forward_recovery.sh",
            "scripts/hpc/submit_r05a_actual_forward_recovery.sh",
            "slurm/r05a_actual_forward_recovery_cpu.sbatch",
        }
        self.assertTrue(required.issubset(paths))
        for relative in paths:
            path = ROOT / relative
            self.assertTrue(path.is_file(), relative)
            self.assertFalse(path.is_symlink(), relative)

    def test_shell_surface_is_executable_cpu_only_and_syntax_valid(self) -> None:
        for path in (WRAPPER, SUBMITTER, SBATCH):
            self.assertTrue(path.stat().st_mode & stat.S_IXUSR, path)
            subprocess.run(["bash", "-n", str(path)], check=True)
        sbatch = SBATCH.read_text(encoding="utf-8")
        self.assertIn("#SBATCH --cpus-per-task=2", sbatch)
        self.assertIn("#SBATCH --mem=8G", sbatch)
        self.assertIn("#SBATCH --time=00:15:00", sbatch)
        self.assertNotIn("#SBATCH --gres", sbatch)
        submitter = SUBMITTER.read_text(encoding="utf-8")
        self.assertIn("sbatch --parsable --hold", submitter)
        self.assertIn("dependency=afterany:28282", submitter)
        self.assertIn('mkdir "$transaction_lock"', submitter)
        self.assertNotIn("scancel", submitter)
        self.assertNotIn("--gres", submitter)

    def test_jobs_records_and_original_receipt_are_separately_bound(self) -> None:
        publisher = PUBLISHER.read_text(encoding="utf-8")
        wrapper = WRAPPER.read_text(encoding="utf-8")
        submitter = SUBMITTER.read_text(encoding="utf-8")
        for source in (publisher, wrapper, submitter):
            self.assertIn("28281", source)
            self.assertIn("28282", source)
        self.assertIn("slurm_array_task_record_identity.sh", wrapper)
        self.assertIn("slurm_array_task_record_identity.sh", submitter)
        self.assertIn("republication-source-terminal-job-record.txt", submitter)
        self.assertIn("republication-original-publisher-terminal-job-record.txt", submitter)
        self.assertIn("cpu-republication-validation.json", publisher)
        self.assertIn("cpu-afterany-validation.json", publisher)
        self.assertIn("original_receipt_preserved", publisher)

    def test_recovery_result_and_receipt_expose_dual_provenance(self) -> None:
        result = json.loads(RESULT_SCHEMA.read_text(encoding="utf-8"))
        receipt = json.loads(RECEIPT_SCHEMA.read_text(encoding="utf-8"))
        self.assertIn("recovery_identity", result["required"])
        identity = result["properties"]["recovery_identity"]
        required = set(identity["required"])
        self.assertTrue(
            {
                "source_git_commit",
                "recovery_git_commit",
                "original_publisher_job_id",
                "recovery_publisher_job_id",
                "recovery_source_contract_sha256",
                "recovery_submission_sha256",
                "recovery_release_fingerprint_sha256",
            }.issubset(required)
        )
        self.assertEqual(result["properties"]["status"], {"const": "frozen_cem_negative"})
        self.assertEqual(receipt["properties"]["outcome"], {"const": "frozen_cem_negative"})
        self.assertFalse(receipt["properties"]["original_publisher_published"]["const"])
        publisher = PUBLISHER.read_text(encoding="utf-8")
        self.assertIn("base_candidate.pop(\"recovery_identity\", None)", publisher)
        self.assertIn("Draft202012Validator(base_schema).validate(base_candidate)", publisher)

    def test_recovery_arguments_are_all_optional_for_normal_mode_but_all_or_none(self) -> None:
        source = PUBLISHER.read_text(encoding="utf-8")
        for option in (
            "--recovery-config",
            "--original-publisher-job-id",
            "--original-receipt",
            "--recovery-source-contract",
            "--recovery-submission",
            "--recovery-release-fingerprint",
        ):
            self.assertIn(f'parser.add_argument("{option}")', source)
        self.assertIn("recovery arguments must be supplied as one complete set", source)
        self.assertIn("if recovery_mode", source)
        self.assertIn("else original_receipt_path", source)

    def test_publication_rolls_back_unreceipted_new_result(self) -> None:
        source = PUBLISHER.read_text(encoding="utf-8")
        rename = source.index("os.replace(candidate, output)")
        receipt = source.index("_atomic_json(receipt, receipt_value)", rename)
        unlink = source.index("output.unlink(missing_ok=True)", receipt)
        self.assertLess(rename, receipt)
        self.assertLess(receipt, unlink)


@unittest.skipUnless(NUMPY_AVAILABLE, "AF-00A budget recovery checks require NumPy")
class ActualForwardRecoveryBudgetTest(unittest.TestCase):
    def setUp(self) -> None:
        self.raw = {
            "directions": {
                "l2_norms": {"delta_star_model": 3.6398398429065115}
            }
        }
        self.payload = {
            "source_budget_float64": 3.639839842906512,
            "source_budget_float32": 3.6398398876190186,
        }
        self.config = {
            "target_contract": {
                "source_reported_budget_float64": 3.6398398429065115,
                "source_budget_float32": 3.6398398876190186,
            }
        }

    def test_exact_run_c_reported_recomputed_and_float32_pair_passes(self) -> None:
        with mock.patch(
            "publish_crfs_r05a_actual_forward_canary._source_delta",
            return_value=(None, None, 3.639839842906512),
        ):
            _validate_source_budget(self.payload, self.raw, self.config)

    def test_one_ulp_payload_tamper_fails(self) -> None:
        changed = dict(self.payload)
        changed["source_budget_float64"] = float(
            np.nextafter(np.float64(3.639839842906512), np.float64(4.0))
        )
        with mock.patch(
            "publish_crfs_r05a_actual_forward_canary._source_delta",
            return_value=(None, None, 3.639839842906512),
        ):
            with self.assertRaisesRegex(ValueError, "recomputed source budget"):
                _validate_source_budget(changed, self.raw, self.config)

    def test_reported_budget_above_existing_tolerance_is_rejected(self) -> None:
        from crfs_oracle.r02_runner import _array_record

        delta = np.zeros((10, 32), dtype=np.float64)
        delta[0, 0] = 1.0
        raw = {
            "directions": {
                "arrays": {"delta_star_model": _array_record(delta)},
                "l2_norms": {"delta_star_model": 1.0 + 2e-12},
            }
        }
        payload = {
            "source_budget_float64": 1.0,
            "source_budget_float32": float(np.float32(1.0)),
        }
        config = {
            "target_contract": {
                "source_reported_budget_float64": 1.0 + 2e-12,
                "source_budget_float32": float(np.float32(1.0 + 2e-12)),
            }
        }
        with self.assertRaisesRegex(Exception, "budget conflicts"):
            _validate_source_budget(payload, raw, config)

    def test_receipt_failure_removes_only_the_newly_published_result(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate = root / ".results.candidate.json"
            output = root / "results.json"
            receipt = root / "cpu-republication-validation.json"
            original = root / "cpu-afterany-validation.json"
            candidate.write_text('{"status":"frozen_cem_negative"}\n', encoding="utf-8")
            original.write_text('{"published":false}\n', encoding="utf-8")
            import hashlib

            digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
            with mock.patch(
                "publish_crfs_r05a_actual_forward_canary._atomic_json",
                side_effect=OSError("simulated receipt failure"),
            ):
                with self.assertRaisesRegex(OSError, "simulated receipt failure"):
                    _commit_publication(
                        candidate,
                        output,
                        receipt,
                        {"published": True},
                        expected_output_sha256=digest,
                    )
            self.assertFalse(output.exists())
            self.assertFalse(receipt.exists())
            self.assertEqual(original.read_text(encoding="utf-8"), '{"published":false}\n')


if __name__ == "__main__":
    unittest.main()
