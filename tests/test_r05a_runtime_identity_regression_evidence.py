import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence" / "r05a" / "runtime-identity-regression-20260716a.json"
ADR0045 = ROOT / "docs" / "decisions" / "0045-accept-runtime-identity-regression.md"
PROGRESS = ROOT / "PROGRESS.md"
EXPERIMENTS = ROOT / "EXPERIMENTS.md"
FEATURES = ROOT / "feature_list.json"
IDENTITY_CONFIG = ROOT / "configs" / "experiments" / "r05a_runtime_identity_regression.json"
CFS_CONFIG = ROOT / "configs" / "experiments" / "r05a_constrained_flow_canary.json"
CFS_APPARATUS = (
    ROOT / "configs" / "experiments" / "r05a_constrained_flow_canary_apparatus.json"
)


class R05ARuntimeIdentityRegressionEvidenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.value = json.loads(EVIDENCE.read_text(encoding="utf-8"))

    def test_exact_terminal_job_resources_and_artifact_hashes_are_preserved(self) -> None:
        self.assertEqual(
            self.value["release_commit"],
            "8415b659a46699757de1e99558713e56b95255b5",
        )
        job = self.value["job"]
        self.assertEqual(
            (job["job_id"], job["state"], job["exit_code"], job["allocated_node"]),
            ("28043", "COMPLETED", "0:0", "worker-1"),
        )
        self.assertEqual((job["requested_cpus"], job["allocated_logical_cpus"]), (1, 2))
        self.assertEqual(job["requested_host_memory_mib"], 256)
        self.assertEqual(job["allocated_host_memory_mib"], 256)
        self.assertEqual((job["requested_gpus"], job["allocated_gpus"]), (0, 0))
        artifacts = self.value["immutable_artifacts"]
        self.assertEqual(
            artifacts["source_contract"]["sha256"],
            "45656005bf2154c232ed601555e4172a6d18b96f5feac2415ec8dbbf9a4b0fbc",
        )
        self.assertEqual(artifacts["source_contract"]["bound_repository_files"], 8)
        self.assertEqual(artifacts["source_contract"]["binding_mismatches"], 0)
        self.assertEqual(
            artifacts["result"]["sha256"],
            "3bda039cd94bf283ebd2b2d9ff1839ce411037a729e6664efaabff20ce38b0de",
        )
        self.assertIs(artifacts["failure_artifacts_exist"], False)
        self.assertIs(artifacts["resume_or_reuse_allowed"], False)

    def test_two_frozen_link_chains_pass_without_executing_compute(self) -> None:
        identities = self.value["observed_runtime_identity"]
        self.assertEqual(
            identities["openpi_python"]["resolved_sha256"],
            "c71718900fe84a9124d39abdd9d68d029930e0dcff1764686d8d6aad97216bc9",
        )
        self.assertEqual(
            identities["libero_python"]["resolved_sha256"],
            "c70efda0ee43d9a0014ee570cad3abb4f46b0c11f6ea88f7c467a91faafd4f62",
        )
        self.assertTrue(all(identity["passed"] for identity in identities.values()))
        execution = self.value["execution"]
        self.assertIs(execution["shell_only"], True)
        for key in (
            "openpi_python_executed",
            "libero_python_executed",
            "checkpoint_loaded",
            "model_inference_executed",
            "simulator_executed",
            "rendering_executed",
            "metrics_executed",
            "training_executed",
            "h100_allocated",
        ):
            with self.subTest(key=key):
                self.assertIs(execution[key], False)

    def test_interpretation_stops_before_cfs_or_h100_claims(self) -> None:
        interpretation = self.value["interpretation"]
        self.assertEqual(interpretation["apparatus_status"], "passed")
        self.assertIs(
            interpretation["standalone_validator_accepts_frozen_canonical_symlink_chains"],
            True,
        )
        self.assertIs(interpretation["cfs_runtime_integration_evaluated"], False)
        self.assertEqual(interpretation["scientific_status"], "not_evaluated")
        for key in (
            "h100_submission_authorized_by_this_result",
            "automatic_resubmission_authorized",
            "ift01_authorized",
            "probe_or_mlp_training_authorized",
        ):
            with self.subTest(key=key):
                self.assertIs(interpretation[key], False)

    def test_vinuni_guide_and_preserved_preflight_are_bound(self) -> None:
        compliance = self.value["vinuni_guide_compliance"]
        self.assertEqual(
            compliance["guide_sha256"],
            "acee44c535e2fc25f8986e41efe233f21683a71c7fb5fa0ae726f0dae573b108",
        )
        self.assertIs(compliance["guide_read_in_full"], True)
        preflight = ROOT / compliance["preflight_path"]
        self.assertEqual(
            hashlib.sha256(preflight.read_bytes()).hexdigest(),
            compliance["preflight_sha256"],
        )
        self.assertIs(compliance["queue_empty_before_submission"], True)
        self.assertIs(compliance["login_compute_processes_observed"], False)
        self.assertIs(compliance["compute_submitted_through_slurm"], True)
        self.assertIs(compliance["bulk_data_transfer_performed"], False)
        self.assertIs(compliance["broad_storage_scan_performed"], False)
        self.assertIs(compliance["cancellation_performed"], False)

    def test_identity_stays_closed_and_cfs_is_closed_or_exactly_released(self) -> None:
        identity = json.loads(IDENTITY_CONFIG.read_text(encoding="utf-8"))
        self.assertIs(identity["ready_to_run"], False)
        self.assertGreater(len(identity["blocked_on"]), 0)
        self.assertIsNone(identity["execution_release"])

        cfs = json.loads(CFS_CONFIG.read_text(encoding="utf-8"))
        apparatus = json.loads(CFS_APPARATUS.read_text(encoding="utf-8"))
        self.assertEqual(cfs["ready_to_run"], apparatus["ready_to_run"])
        if cfs["ready_to_run"]:
            self.assertEqual(cfs["blocked_on"], [])
            self.assertEqual(apparatus["blocked_on"], [])
            self.assertIs(cfs["preregistration"]["h100_submission_authorized"], True)
            self.assertEqual(cfs["execution_release"], apparatus["execution_release"])
            release = cfs["execution_release"]
            self.assertEqual(
                release["decision_artifact"],
                "docs/decisions/0047-preserve-cfs00a-run-b-and-record-failed-jacobian-diagnostics.md",
            )
            self.assertIs(release["single_submission"], True)
            self.assertEqual(release["source_host"], "worker-1")
            self.assertIs(release["automatic_resubmission_allowed"], False)
            self.assertIs(release["automatic_next_experiment_allowed"], False)
        else:
            for config in (cfs, apparatus):
                self.assertGreater(len(config["blocked_on"]), 0)
                self.assertIsNone(config["execution_release"])
            self.assertIs(cfs["preregistration"]["h100_submission_authorized"], False)
        r05a = next(
            feature
            for feature in json.loads(FEATURES.read_text(encoding="utf-8"))["features"]
            if feature["id"] == "R05A"
        )
        self.assertEqual(r05a["status"], "active")
        combined = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (ADR0045, PROGRESS, EXPERIMENTS)
        )
        for phrase in (
            "standalone validator",
            "full CFS integration",
            "no H100 authority",
            "IFT-01",
            "probe/MLP",
        ):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, combined)


if __name__ == "__main__":
    unittest.main()
