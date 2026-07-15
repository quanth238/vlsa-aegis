import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence" / "r05a" / "ift00a-sampled-current-launch-b.json"
ADR0039 = (
    ROOT
    / "docs"
    / "decisions"
    / "0039-preserve-sampled-current-launch-b-and-repair-exact-task-accounting.md"
)
PROGRESS = ROOT / "PROGRESS.md"
EXPERIMENTS = ROOT / "EXPERIMENTS.md"
FEATURES = ROOT / "feature_list.json"
APPARATUS = (
    ROOT / "configs" / "experiments" / "r05a_sampled_current_canary_apparatus.json"
)
SCIENCE = ROOT / "configs" / "experiments" / "r05a_inverse_flow_canary.json"


class R05ASampledCurrentLaunchBEvidenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.value = json.loads(EVIDENCE.read_text(encoding="utf-8"))

    def test_exact_terminal_jobs_and_immutable_bindings_are_preserved(self) -> None:
        self.assertEqual(
            self.value["run_id"],
            "r05a-inverse-flow-sampled-current-canary-20260715b",
        )
        self.assertEqual(
            self.value["release_commit"],
            "06b365b5899c2cb31db12187350cce48a3a0ea20",
        )
        gpu = self.value["jobs"]["gpu_array_task"]
        cpu = self.value["jobs"]["cpu_afterany_validator"]
        self.assertEqual((gpu["job_id"], gpu["state"], gpu["exit_code"]), ("27962_0", "COMPLETED", "0:0"))
        self.assertEqual((cpu["job_id"], cpu["state"], cpu["exit_code"]), ("27963", "FAILED", "3:0"))
        self.assertEqual(gpu["node"], "worker-1")
        self.assertEqual(cpu["node"], "worker-1")
        receipts = self.value["immutable_receipts"]
        self.assertEqual(
            receipts["source_contract_sha256"],
            "b9a9e253c9a5e27815018c13c839805c97752545189085caac016db4ee81e1e6",
        )
        self.assertIs(receipts["repository_bindings_match_release_commit"], True)
        self.assertEqual(receipts["repository_binding_count"], 28)
        self.assertIs(receipts["identity_consumed"], True)
        self.assertIs(receipts["resume_or_reuse_allowed"], False)

    def test_raw_payload_is_diagnostic_finite_nonconvergence_only(self) -> None:
        diagnostic = self.value["raw_scientific_diagnostic"]
        self.assertEqual(diagnostic["payload_status"], "completed_nonconverged")
        self.assertIs(diagnostic["accepted_terminal_result"], False)
        self.assertEqual(diagnostic["teacher_calls_executed"], 2)
        for name in ("first", "duplicate"):
            search = diagnostic[name]
            self.assertEqual(search["status_code"], 3)
            self.assertIs(search["converged"], False)
            self.assertIs(search["nonfinite"], False)
            self.assertEqual(search["iterations"], 128)
        self.assertEqual(
            diagnostic["first"]["schedule_sha256"],
            diagnostic["duplicate"]["schedule_sha256"],
        )
        self.assertEqual(
            diagnostic["first"]["returned_actions_sha256"],
            diagnostic["duplicate"]["returned_actions_sha256"],
        )
        self.assertIs(diagnostic["failed_schedule_applied"], False)
        self.assertIs(diagnostic["returned_actions_exact_frozen"], True)
        self.assertIs(diagnostic["infeasibility_certificate"], False)
        self.assertEqual(diagnostic["policy_generated_action_steps_executed"], 0)
        self.assertEqual(diagnostic["teacher_generated_action_steps_executed"], 0)
        self.assertEqual(diagnostic["efficacy_rollouts_executed"], 0)

    def test_publication_failure_cannot_be_promoted(self) -> None:
        publication = self.value["publication"]
        self.assertIs(publication["python_publisher_invoked"], False)
        self.assertIs(publication["hidden_candidate_exists"], False)
        self.assertIs(publication["results_json_exists"], False)
        self.assertIs(publication["cpu_failure_receipt_passed"], False)
        self.assertIs(publication["cpu_failure_receipt_published"], False)
        self.assertIsNone(publication["cpu_failure_receipt_result_sha256"])
        self.assertIs(publication["cpu_failure_receipt_publisher_job_id_present"], False)
        self.assertIs(publication["cpu_failure_receipt_source_task_id_present"], False)
        self.assertIs(publication["cpu_failure_receipt_failure_stage_present"], False)
        failure = self.value["failure"]
        self.assertEqual(
            failure["classification"], "fail_closed_slurm_job_identity_mismatch"
        )
        self.assertIs(failure["accounting_delay_would_fix"], False)

    def test_trackers_leave_r05a_active_and_h100_apparatus_unreleased(self) -> None:
        apparatus = json.loads(APPARATUS.read_text(encoding="utf-8"))
        self.assertIs(apparatus["ready_to_run"], False)
        self.assertGreater(len(apparatus["blocked_on"]), 0)
        self.assertNotIn("execution_release", apparatus)
        r05a = next(
            feature
            for feature in json.loads(FEATURES.read_text(encoding="utf-8"))["features"]
            if feature["id"] == "R05A"
        )
        self.assertEqual(r05a["status"], "active")
        combined = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (ADR0039, PROGRESS, EXPERIMENTS)
        )
        for phrase in (
            "apparatus-inconclusive",
            "no accepted IFT-00A",
            "zero-GPU",
            "IFT-01",
            "training",
        ):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, combined)

    def test_frozen_scientific_config_remains_byte_exact(self) -> None:
        self.assertEqual(
            hashlib.sha256(SCIENCE.read_bytes()).hexdigest(),
            "c31401867f3cdce2b3f443ad021c39dfb812f573b570e1e7434e1f149f79abfb",
        )


if __name__ == "__main__":
    unittest.main()
