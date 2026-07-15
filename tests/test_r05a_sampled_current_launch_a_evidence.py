import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence" / "r05a" / "ift00a-sampled-current-launch-a.json"
PROGRESS = ROOT / "PROGRESS.md"
EXPERIMENTS = ROOT / "EXPERIMENTS.md"
ADR0038 = (
    ROOT
    / "docs"
    / "decisions"
    / "0038-preserve-sampled-current-launch-a-and-repair-token-parsing.md"
)
SUBMITTER = ROOT / "scripts" / "hpc" / "submit_r05a_sampled_current_canary.sh"


class R05ASampledCurrentLaunchAEvidenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.value = json.loads(EVIDENCE.read_text(encoding="utf-8"))

    def test_terminal_record_proves_zero_runtime_and_consumed_identity(self) -> None:
        self.assertEqual(
            self.value["run_id"],
            "r05a-inverse-flow-sampled-current-canary-20260715a",
        )
        self.assertEqual(self.value["release_commit"], "223667c91b05be9ab403e4d92d0cd1a96b45246f")
        self.assertEqual(self.value["submission"]["exact_gpu_task_id"], "27928_0")
        terminal = self.value["terminal_slurm_record"]
        self.assertEqual(terminal["state"], "CANCELLED by 1073")
        self.assertEqual(terminal["elapsed"], "00:00:00")
        self.assertEqual(terminal["start_time"], "None")
        self.assertEqual(terminal["node_list"], "None assigned")
        self.assertIsNone(terminal["allocated_tres"])

        run_root = self.value["immutable_run_root"]
        self.assertEqual(run_root["files"], ["launch-reservation.json"])
        self.assertEqual(
            run_root["launch_reservation_sha256"],
            "bec4583c5c046c7ca9f1155debc1f389d058daede42d3c485a39398f19c36ed4",
        )
        self.assertIs(run_root["identity_consumed"], True)
        self.assertIs(run_root["resume_or_reuse_allowed"], False)

    def test_transaction_stopped_before_every_scientific_boundary(self) -> None:
        boundary = self.value["transaction_boundary"]
        self.assertIs(boundary["launch_reservation_written"], True)
        self.assertIs(boundary["held_gpu_sbatch_returned"], True)
        for key in (
            "held_gpu_submission_receipt_written",
            "source_contract_written",
            "cpu_afterany_submitted",
            "atomic_submission_receipt_written",
            "gpu_job_released",
            "allocation_started",
            "gpu_allocated",
            "checkpoint_loaded",
            "policy_server_started",
            "pi05_sampler_called",
            "teacher_search_called",
        ):
            with self.subTest(key=key):
                self.assertIs(boundary[key], False)
        self.assertEqual(boundary["simulator_policy_steps"], 0)
        self.assertEqual(boundary["simulator_teacher_steps"], 0)

        interpretation = self.value["interpretation"]
        self.assertEqual(interpretation["status"], "apparatus_inconclusive")
        self.assertIs(interpretation["teacher_transport_evaluated"], False)
        self.assertIs(interpretation["simulator_efficacy_evaluated"], False)
        self.assertEqual(interpretation["scientific_conclusion"], "none")

    def test_trackers_and_decision_preserve_the_no_claim_boundary(self) -> None:
        combined = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (PROGRESS, EXPERIMENTS, ADR0038)
        )
        for phrase in (
            "apparatus-inconclusive",
            "zero runtime",
            "No checkpoint",
            "do not launch IFT-01",
        ):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, combined)
        self.assertIn("Do not resume job `27928`", combined)

    def test_submitter_checks_adjacent_slurm_fields_independently(self) -> None:
        source = SUBMITTER.read_text(encoding="utf-8")
        self.assertIn(
            'for field in "JobState=PENDING" "Reason=JobHeldUser" "ReqNodeList=worker-1"; do',
            source,
        )
        self.assertIn(
            'for field in "JobState=PENDING" "Partition=main"; do',
            source,
        )
        self.assertNotIn(
            '*" JobState=PENDING "*" Reason=JobHeldUser "*" ReqNodeList=worker-1 "*',
            source,
        )


if __name__ == "__main__":
    unittest.main()
