from __future__ import annotations

import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_PATH = ROOT / "evidence" / "r05a" / "cfs00a-fd-diagnostic-20260716a.json"
ADR_PATH = (
    ROOT
    / "docs"
    / "decisions"
    / "0048-preserve-fd-diagnostic-and-normalize-terminal-transport.md"
)


class CFS00AFailedDerivativeDiagnosticAEvidenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.evidence = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))

    def test_exact_release_jobs_and_receipts_are_frozen(self) -> None:
        self.assertEqual(
            self.evidence["run_id"],
            "r05a-constrained-flow-fd-diagnostic-20260716a",
        )
        self.assertEqual(
            self.evidence["release_commit"],
            "3d44b2c5a4779725101d672ed29658a841c85541",
        )
        gpu = self.evidence["jobs"]["gpu_array_task"]
        cpu = self.evidence["jobs"]["cpu_afterany_validator"]
        self.assertEqual(
            (gpu["job_id"], gpu["state"], gpu["exit_code"], gpu["node"], gpu["elapsed"]),
            ("28212_0", "COMPLETED", "0:0", "worker-1", "00:03:06"),
        )
        self.assertEqual(
            (cpu["job_id"], cpu["state"], cpu["exit_code"], cpu["node"], cpu["elapsed"]),
            ("28213", "COMPLETED", "0:0", "worker-0", "00:00:10"),
        )
        artifacts = self.evidence["immutable_artifacts"]
        self.assertEqual(
            artifacts["source_contract_sha256"],
            "7c8eee72c63c17a3922162a6f78eff6467f6be4301e9e42ca6df07377bed3b99",
        )
        self.assertEqual(
            artifacts["submission_sha256"],
            "73eb2320a6e7498bb4c7ab968bd5f90293ed804f3a9ae6b5dbf8f4174cf283cd",
        )
        self.assertEqual(
            artifacts["constrained_flow_payload_sha256"],
            "d423da9f1434c7be2ed0c894c7252f764d9a9939e88cae5955c01e1a9a5cbb2d",
        )
        self.assertEqual(
            artifacts["published_result_sha256"],
            "5c843ce5ed4f0e836b43dd6e0b42fec3ae120fa61cb1c56350eac046001062e0",
        )
        self.assertEqual(
            artifacts["cpu_publication_receipt_sha256"],
            "be91ec15ea08431af63af85298faf3a4015bd527e5eb26d3ea28ae2bd21d32d0",
        )

    def test_transport_envelope_root_cause_is_exact(self) -> None:
        boundary = self.evidence["execution_boundary"]
        self.assertEqual(boundary["adapter_terminal_outer_keys"], ["__crfs_terminal__"])
        self.assertEqual(
            boundary["transport_outer_keys_received_by_client"],
            ["__crfs_terminal__", "server_timing"],
        )
        self.assertTrue(boundary["adapter_typed_finite_difference_terminal_reached"])
        self.assertTrue(boundary["websocket_server_appended_standard_timing"])
        self.assertFalse(boundary["client_terminal_validation_completed"])
        failure = self.evidence["failure"]
        self.assertEqual(failure["root_error_type"], "ConstrainedFlowCanaryError")
        self.assertEqual(
            failure["root_message"],
            "terminal constrained-flow response must contain only its reserved key",
        )
        self.assertTrue(failure["root_cause_identified"])
        self.assertFalse(failure["derivative_mismatch_root_cause_identified"])

    def test_no_numeric_or_method_claim_crosses_failure(self) -> None:
        boundary = self.evidence["execution_boundary"]
        for key in (
            "numeric_jacobian_persisted",
            "finite_difference_numeric_comparisons_persisted",
            "arm_a_scientific_reproduction_validated",
            "arm_b_fista_started",
            "arm_b_candidate_exists",
            "arm_b_nonlinear_replay_executed",
            "arm_c_executed",
            "four_fidelity_gates_evaluated",
            "simulator_efficacy_evaluated",
        ):
            self.assertFalse(boundary[key], key)
        self.assertEqual(boundary["policy_generated_action_steps_executed"], 0)
        self.assertEqual(boundary["teacher_generated_action_steps_executed"], 0)
        interpretation = self.evidence["interpretation"]
        self.assertEqual(interpretation["status"], "apparatus_inconclusive")
        self.assertFalse(interpretation["mechanism_pass"])
        self.assertFalse(interpretation["frozen_method_negative"])
        self.assertFalse(interpretation["research_hypothesis_confirmed"])
        self.assertFalse(interpretation["research_hypothesis_refuted"])
        self.assertFalse(interpretation["automatic_h100_resubmission_authorized"])
        self.assertFalse(interpretation["probe_or_mlp_training_authorized"])

    def test_allocation_health_cannot_be_misreported_as_the_failure(self) -> None:
        validation = self.evidence["operational_validation"]
        self.assertEqual(validation["allocation_tests_expected"], 99)
        self.assertEqual(validation["allocation_tests_observed"], 99)
        self.assertEqual(validation["allocation_test_skips"], 0)
        self.assertEqual(validation["host_sampled_current_high_water_bytes"], 16074977280)
        self.assertEqual(validation["gpu_sampled_device_high_water_mib"], 8513)
        self.assertEqual(validation["memory_event_deltas"], {"max": 0, "oom": 0, "oom_kill": 0})
        self.assertTrue(validation["cpu_publisher_validation_passed"])
        self.assertTrue(validation["result_published_atomically"])

    def test_trackers_and_adr_preserve_fail_closed_next_step(self) -> None:
        adr = ADR_PATH.read_text(encoding="utf-8")
        self.assertIn("client-side normalization", adr)
        normalized_adr = " ".join(adr.split())
        self.assertIn(
            "do not select its run ID or authorize execution in this ADR",
            normalized_adr,
        )
        scientific = json.loads(
            (ROOT / "configs/experiments/r05a_constrained_flow_canary.json").read_text(
                encoding="utf-8"
            )
        )
        apparatus = json.loads(
            (
                ROOT
                / "configs/experiments/r05a_constrained_flow_canary_apparatus.json"
            ).read_text(encoding="utf-8")
        )
        if scientific["ready_to_run"]:
            self.assertTrue(scientific["ready_to_run"])
            self.assertTrue(apparatus["ready_to_run"])
            self.assertEqual(scientific["blocked_on"], [])
            self.assertEqual(apparatus["blocked_on"], [])
            self.assertEqual(
                scientific["execution_release"], apparatus["execution_release"]
            )
            release = scientific["execution_release"]
            self.assertTrue(release["single_submission"])
            self.assertFalse(release["automatic_resubmission_allowed"])
            self.assertFalse(release["automatic_next_experiment_allowed"])
            self.assertIn(release["run_id"], adr)
            self.assertIn(release["accepted_implementation_commit"], adr)
        else:
            self.assertFalse(scientific["ready_to_run"])
            self.assertFalse(apparatus["ready_to_run"])
            self.assertTrue(scientific["blocked_on"])
            self.assertTrue(apparatus["blocked_on"])
            self.assertIsNone(scientific["execution_release"])
            self.assertIsNone(apparatus["execution_release"])
        decisions = (ROOT / "DECISIONS.md").read_text(encoding="utf-8")
        self.assertIn(ADR_PATH.relative_to(ROOT).as_posix(), decisions)
        for relative in ("PROGRESS.md", "EXPERIMENTS.md", "feature_list.json"):
            text = (ROOT / relative).read_text(encoding="utf-8")
            self.assertIn("28212_0", text, relative)
            self.assertIn("28213", text, relative)
            self.assertTrue(
                "apparatus_inconclusive" in text or "apparatus-inconclusive" in text,
                relative,
            )
            self.assertIn("server_timing", text, relative)
        feature = json.loads((ROOT / "feature_list.json").read_text(encoding="utf-8"))
        r05a = next(item for item in feature["features"] if item["id"] == "R05A")
        self.assertEqual(r05a["status"], "active")
        self.assertIn("28212_0", r05a["evidence"])
        self.assertIn("server_timing", r05a["evidence"])
        self.assertIn("actual-forward", r05a["evidence"])


if __name__ == "__main__":
    unittest.main()
