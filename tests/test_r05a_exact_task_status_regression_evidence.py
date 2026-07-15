import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = (
    ROOT / "evidence" / "r05a" / "exact-array-task-afterany-regression-a.json"
)
APPARATUS = (
    ROOT / "configs" / "experiments" / "r05a_sampled_current_canary_apparatus.json"
)
PROGRESS = ROOT / "PROGRESS.md"
EXPERIMENTS = ROOT / "EXPERIMENTS.md"
ADR0039 = (
    ROOT
    / "docs"
    / "decisions"
    / "0039-preserve-sampled-current-launch-b-and-repair-exact-task-accounting.md"
)
H100_PUBLICATION = ROOT / "main" / "crfs_oracle" / "r05a_sampled_current_canary.py"
H100_SUBMITTER = ROOT / "scripts" / "hpc" / "submit_r05a_sampled_current_canary.sh"


class R05AExactTaskStatusRegressionEvidenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.value = json.loads(EVIDENCE.read_text(encoding="utf-8"))

    def test_exact_terminal_jobs_and_result_hash_are_preserved(self) -> None:
        self.assertEqual(
            self.value["git_commit"],
            "5e595a366cb95d50ee86776f5de701627bf09669",
        )
        source = self.value["jobs"]["source_array_task"]
        validator = self.value["jobs"]["afterany_validator"]
        self.assertEqual(
            (source["job_id"], source["state"], source["exit_code"]),
            ("27975_0", "COMPLETED", "0:0"),
        )
        self.assertEqual(
            (validator["job_id"], validator["state"], validator["exit_code"]),
            ("27976", "COMPLETED", "0:0"),
        )
        self.assertEqual(validator["dependency"], "afterany:27975")
        self.assertEqual(
            self.value["artifacts"]["result_sha256"],
            "0de4b8b736bd750a82e7439cf737b9d16e248d67ee0d2f741d82f09d9bcd6b74",
        )
        self.assertEqual(
            self.value["artifacts"]["validation_receipt_sha256"],
            "35c7e2cd0388b148e69ff1e780bee08c34e6725bc9f715da12345843fbc44a4f",
        )

    def test_live_regression_validates_accounting_and_nothing_scientific(self) -> None:
        binding = self.value["source_binding"]
        self.assertEqual(binding["bound_file_count"], 9)
        self.assertIs(binding["all_bound_files_match_commit"], True)
        validation = self.value["validation"]
        self.assertEqual(validation["exact_source_task_id"], "27975_0")
        self.assertEqual(validation["observed_source_state"], "COMPLETED")
        self.assertEqual(validation["observed_source_exit_code"], "0:0")
        self.assertEqual(validation["source_query_status"], 0)
        self.assertIs(validation["result_passed"], True)
        self.assertIs(validation["receipt_published"], True)
        self.assertIs(validation["gpu_allocated"], False)
        self.assertIs(validation["python_executed"], False)
        self.assertIs(validation["model_inference_executed"], False)
        self.assertEqual(validation["simulator_steps_executed"], 0)
        self.assertEqual(validation["teacher_searches_executed"], 0)
        self.assertIs(validation["training_executed"], False)
        interpretation = self.value["interpretation"]
        self.assertEqual(interpretation["status"], "passed_apparatus_regression")
        self.assertIs(interpretation["exact_task_accounting_repair_validated"], True)
        self.assertIs(interpretation["scientific_claim_allowed"], False)
        self.assertIs(interpretation["teacher_transport_evaluated"], False)
        self.assertIs(interpretation["h100_retry_authorized_by_this_artifact_alone"], False)
        self.assertIs(interpretation["ift01_authorized"], False)

    def test_trackers_stop_with_h100_apparatus_unreleased(self) -> None:
        apparatus = json.loads(APPARATUS.read_text(encoding="utf-8"))
        self.assertIs(apparatus["ready_to_run"], False)
        self.assertNotIn("execution_release", apparatus)
        self.assertIn("new_exact_h100_execution_identity_not_selected", apparatus["blocked_on"])
        self.assertIn("separate_h100_release_review_not_completed", apparatus["blocked_on"])
        combined = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (PROGRESS, EXPERIMENTS, ADR0039)
        )
        for phrase in (
            "27975_0",
            "27976",
            "validates only the accounting",
            "does not authorize",
            "IFT-01",
        ):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, combined)

    def test_future_h100_contract_explicitly_binds_this_terminal_evidence(self) -> None:
        relative = "evidence/r05a/exact-array-task-afterany-regression-a.json"
        self.assertIn(relative, H100_PUBLICATION.read_text(encoding="utf-8"))
        self.assertEqual(
            H100_SUBMITTER.read_text(encoding="utf-8").count(relative),
            2,
        )


if __name__ == "__main__":
    unittest.main()
