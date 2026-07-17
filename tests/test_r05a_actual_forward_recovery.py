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
SOURCE_CONFIG = ROOT / "configs" / "experiments" / "r05a_actual_forward_canary.json"
EVIDENCE = ROOT / "evidence" / "r05a" / "af00a-actual-forward-run-c-publication-recovery.json"
TERMINAL_ADR = ROOT / "docs" / "decisions" / "0058-accept-af00a-frozen-cem-negative.md"
FEATURES = ROOT / "feature_list.json"
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
            self.assertIn(
                config["config_status"],
                {
                    "implemented_fail_closed_pending_review",
                    "terminal_cpu_republication_completed_consumed",
                },
            )
            if (
                config["config_status"]
                == "terminal_cpu_republication_completed_consumed"
            ):
                self.assertEqual(
                    config["blocked_on"],
                    [
                        "single_authorized_cpu_republication_consumed_by_terminal_job_28291"
                    ],
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

    def test_terminal_publication_evidence_is_exact_and_both_launches_are_closed(self) -> None:
        recovery = json.loads(CONFIG.read_text(encoding="utf-8"))
        source = json.loads(SOURCE_CONFIG.read_text(encoding="utf-8"))
        evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))

        self.assertFalse(source["ready_to_run"])
        self.assertEqual(source["config_status"], "draft_preregistered_not_released")
        self.assertFalse(source["preregistration"]["h100_submission_authorized"])
        self.assertIsNone(source["execution_release"])
        self.assertFalse(recovery["ready_to_run"])
        self.assertEqual(
            recovery["config_status"],
            "terminal_cpu_republication_completed_consumed",
        )
        self.assertIsNone(recovery["execution_release"])

        job = evidence["jobs"]["recovery_cpu_publisher"]
        self.assertEqual(
            job,
            {
                "job_id": "28291",
                "state": "COMPLETED",
                "exit_code": "0:0",
                "elapsed": "00:00:08",
                "node": "worker-0",
                "requested_tres": "billing=2,cpu=2,mem=8G,node=1",
                "allocated_tres": "billing=2,cpu=2,mem=8G,node=1",
                "gpus": 0,
                "dependency": "afterany:28282",
                "log_sha256": "47b768bda62f801a16ca3fc5a224c9ae18396411238dd67a1d3e1d9f15204131",
            },
        )
        artifacts = evidence["immutable_artifacts"]
        self.assertEqual(
            artifacts["results_json_sha256"],
            "507f25bc381b7bfeaa30abe3a7df970681f3b175f39221034c6e768e72dae914",
        )
        self.assertEqual(
            artifacts["recovery_receipt_sha256"],
            "5462f875ffb41a84a06c4df715366776af3811f3229385d05be8f5b0dafcb632",
        )
        self.assertEqual(
            artifacts["original_failed_receipt_sha256"],
            "f8e9ef108630cad9a9691dba0827a8be68d3b15e41ee3de2e4e6cfaf03736d27",
        )
        self.assertEqual(
            artifacts["recovery_source_contract_sha256"],
            "249087a42023d73e65589158bc8be5407605b57773b81a2c1d6138c13a820eab",
        )
        self.assertEqual(
            artifacts["recovery_submission_sha256"],
            "68a6cd3a08c4d502fb97f0a0db18afcd88fe23935e6c2bf5c2d4d807204e4c7c",
        )
        self.assertEqual(
            artifacts["recovery_release_fingerprint_sha256"],
            "1fa71d24d37b1ee6d7d19e130f29c9d3de55f82967b8e648fd3981a932eaa921",
        )
        self.assertTrue(artifacts["results_json_exists"])
        publication = evidence["publication_recovery"]
        self.assertEqual(
            publication,
            {
                "release_commit": "eb3be3a86c1336a9090413cf7d6c83f28a0f365c",
                "recovery_job_id": "28291",
                "published": True,
                "passed": True,
                "outcome": "frozen_cem_negative",
                "original_receipt_preserved": True,
                "recovery_cpu_is_sole_results_json_publisher": True,
                "schema_validation_passed": True,
                "terminal_semantic_validation_passed": True,
            },
        )
        self.assertEqual(
            evidence["independent_raw_validation"],
            {
                "passed": True,
                "exact_policy_requests": 534,
                "cem_policy_requests": 520,
                "selected_pool_index": 465,
                "selected_cem_query_index": 464,
                "outcome": "frozen_cem_negative",
                "arm_a": {
                    "objective": 2218.1335502517986,
                    "energy": 2.649686839308922,
                    "xyz_max_abs": 0.7806643492412315,
                    "xyz_rms": 0.35969860072305654,
                    "full_max_abs": 0.7806643492412315,
                    "full_rms": 0.23554385122689606,
                    "passed": False,
                },
                "arm_b": {
                    "changed": True,
                    "objective": 2121.5404274315442,
                    "energy": 2.5488477410268158,
                    "xyz_max_abs": 0.7899218065691934,
                    "xyz_rms": 0.35177249996585547,
                    "full_max_abs": 0.7899218065691934,
                    "full_rms": 0.23039493162555724,
                    "passed": False,
                },
                "arm_c": {
                    "objective": 2968.7447114541746,
                    "energy": 2.5488477410268153,
                    "xyz_max_abs": 0.9606940421772645,
                    "xyz_rms": 0.416131221925226,
                    "full_max_abs": 0.9606940421772645,
                    "full_rms": 0.2725038054457127,
                    "passed": False,
                },
                "fidelity_limits": {
                    "xyz_max_abs": 0.01,
                    "xyz_rms": 0.005,
                    "full_max_abs": 0.05,
                    "full_rms": 0.015,
                },
            },
        )
        self.assertEqual(
            evidence["execution_boundary"],
            {
                "allocation_focused_tests_passed": 36,
                "allocation_focused_tests_skipped": 0,
                "policy_generated_action_steps_executed": 0,
                "teacher_generated_action_steps_executed": 0,
                "efficacy_rollouts_executed": 0,
                "simulator_efficacy_evaluated": False,
                "training": False,
            },
        )
        interpretation = evidence["interpretation_after_recovery"]
        self.assertEqual(
            interpretation["official_status"], "published_frozen_cem_negative"
        )
        self.assertEqual(
            interpretation["flow_transport_feasibility_status"],
            "undetermined_reachability_vs_optimizer",
        )
        self.assertFalse(interpretation["ift01_authorized"])
        self.assertFalse(interpretation["probe_or_mlp_training_authorized"])

        adr = TERMINAL_ADR.read_text(encoding="utf-8")
        normalized_adr = " ".join(adr.split())
        self.assertIn("rejects only this exact one-case CEM teacher", normalized_adr)
        self.assertIn("not an infeasibility certificate", normalized_adr)
        features = json.loads(FEATURES.read_text(encoding="utf-8"))
        by_id = {item["id"]: item for item in features["features"]}
        active = [
            item["id"] for item in features["features"] if item["status"] == "active"
        ]
        self.assertEqual(active, ["R06"])
        self.assertEqual(by_id["R04"]["status"], "blocked")
        self.assertEqual(by_id["R05A"]["status"], "blocked")
        self.assertIn("frozen_cem_negative", by_id["R05A"]["evidence"])
        self.assertIn(
            "No teacher-generated action entered the simulator",
            by_id["R05A"]["evidence"],
        )
        self.assertIn(
            "probe/MLP training remains forbidden",
            by_id["R06"]["evidence"],
        )

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
