from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence/r05a/cfs00a-same-budget-launch-b.json"
SCIENTIFIC = ROOT / "configs/experiments/r05a_constrained_flow_canary.json"
APPARATUS = ROOT / "configs/experiments/r05a_constrained_flow_canary_apparatus.json"
DECISION = ROOT / "docs/decisions/0047-preserve-cfs00a-run-b-and-record-failed-jacobian-diagnostics.md"
SCIENTIFIC_PROJECTION = (
    "7dc2c8f63838ae4e22db8a927d87cae89daf0025c931b33e225c946abe8dc915"
)


def _object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssertionError(f"{path} is not an object")
    return value


def _content_hash(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class R05ACFS00ALaunchBEvidenceTest(unittest.TestCase):
    def test_exact_terminal_jobs_publication_and_hashes_are_preserved(self) -> None:
        evidence = _object(EVIDENCE)
        self.assertEqual(
            evidence["run_id"],
            "r05a-constrained-flow-same-budget-canary-20260716b",
        )
        self.assertEqual(
            evidence["release_commit"],
            "12a7da69d3d34050709d08b9ca903a99f9d30862",
        )
        gpu = evidence["jobs"]["gpu_array_task"]
        cpu = evidence["jobs"]["cpu_afterany_validator"]
        self.assertEqual(
            (gpu["job_id"], gpu["state"], gpu["exit_code"], gpu["node"]),
            ("28048_0", "COMPLETED", "0:0", "worker-1"),
        )
        self.assertEqual(
            (cpu["job_id"], cpu["state"], cpu["exit_code"], cpu["node"]),
            ("28049", "COMPLETED", "0:0", "worker-0"),
        )
        immutable = evidence["immutable_artifacts"]
        self.assertEqual(
            immutable["source_contract_sha256"],
            "5cbc304ffda2841e824da9ed0dd926f8f4e912f2aba9cbcd64d6f243fccf2716",
        )
        self.assertEqual(
            immutable["submission_sha256"],
            "ad7d9764897c1a4297fe10c22573d51f35939d36a00b9ad0b1039b76be18e98e",
        )
        self.assertEqual(
            immutable["published_result_sha256"],
            "655b48f421af3e639a472d6bef8f20c41e2b2ed60dbb33a0a643bb4aa3e2315f",
        )
        self.assertEqual(
            immutable["cpu_publication_receipt_sha256"],
            "b8bee6bc497ee70b5d621ef0c13281a6ebfbc0fbeeb9a2f0373fae413fd0b769",
        )
        self.assertTrue(immutable["identity_consumed"])
        self.assertFalse(immutable["resume_or_reuse_allowed"])

    def test_partial_arm_boundary_cannot_be_promoted_to_method_result(self) -> None:
        evidence = _object(EVIDENCE)
        boundary = evidence["execution_boundary"]
        self.assertTrue(boundary["arm_b_autograd_jacobian_computed"])
        self.assertTrue(boundary["arm_b_finite_difference_checks_executed"])
        self.assertFalse(boundary["arm_b_finite_difference_passed"])
        self.assertFalse(boundary["arm_b_numeric_diagnostics_persisted"])
        self.assertFalse(boundary["arm_b_fista_started"])
        self.assertFalse(boundary["arm_b_candidate_exists"])
        self.assertFalse(boundary["arm_b_nonlinear_replay_executed"])
        self.assertFalse(boundary["arm_c_executed"])
        self.assertFalse(boundary["four_fidelity_gates_evaluated"])
        self.assertFalse(boundary["arm_a_scientific_reproduction_validated"])
        self.assertEqual(boundary["policy_generated_action_steps_executed"], 0)
        self.assertEqual(boundary["teacher_generated_action_steps_executed"], 0)
        interpretation = evidence["interpretation"]
        self.assertEqual(interpretation["status"], "apparatus_inconclusive")
        self.assertFalse(interpretation["mechanism_pass"])
        self.assertFalse(interpretation["frozen_method_negative"])
        self.assertFalse(interpretation["research_hypothesis_refuted"])
        self.assertFalse(interpretation["ift01_authorized"])
        self.assertFalse(interpretation["probe_or_mlp_training_authorized"])

    def test_memory_and_test_evidence_exclude_resource_failure(self) -> None:
        operational = _object(EVIDENCE)["operational_validation"]
        self.assertEqual(
            operational["allocation_tests_expected"],
            operational["allocation_tests_observed"],
        )
        self.assertEqual(operational["allocation_test_skips"], 0)
        self.assertLess(
            operational["host_sampled_current_high_water_bytes"],
            operational["host_memory_max_before_bytes"],
        )
        self.assertEqual(operational["memory_event_deltas"]["max"], 0)
        self.assertEqual(operational["memory_event_deltas"]["oom"], 0)
        self.assertEqual(operational["memory_event_deltas"]["oom_kill"], 0)
        self.assertTrue(operational["cpu_publisher_validation_passed"])
        self.assertTrue(operational["result_published_atomically"])

    def test_configs_are_closed_or_exactly_released_and_projection_unchanged(self) -> None:
        scientific = _object(SCIENTIFIC)
        apparatus = _object(APPARATUS)
        self.assertEqual(scientific["ready_to_run"], apparatus["ready_to_run"])
        if scientific["ready_to_run"]:
            for value in (scientific, apparatus):
                self.assertEqual(value["blocked_on"], [])
                self.assertIsInstance(value["execution_release"], dict)
            self.assertTrue(
                scientific["preregistration"]["h100_submission_authorized"]
            )
            release = scientific["execution_release"]
            self.assertEqual(release, apparatus["execution_release"])
            self.assertEqual(
                release["decision_artifact"],
                "docs/decisions/0047-preserve-cfs00a-run-b-and-record-failed-jacobian-diagnostics.md",
            )
            self.assertTrue(release["single_submission"])
            self.assertEqual(release["source_host"], "worker-1")
            self.assertFalse(release["automatic_resubmission_allowed"])
            self.assertFalse(release["automatic_next_experiment_allowed"])
        else:
            for value in (scientific, apparatus):
                self.assertTrue(value["blocked_on"])
                self.assertIsNone(value["execution_release"])
            self.assertFalse(
                scientific["preregistration"]["h100_submission_authorized"]
            )
        projection = copy.deepcopy(scientific)
        for key in ("config_status", "ready_to_run", "blocked_on", "execution_release"):
            projection.pop(key, None)
        projection["preregistration"].pop("h100_submission_authorized", None)
        self.assertEqual(_content_hash(projection), SCIENTIFIC_PROJECTION)
        self.assertTrue(DECISION.is_file())

    def test_trackers_keep_only_r05a_active_and_training_blocked(self) -> None:
        features = _object(ROOT / "feature_list.json")
        active = [item for item in features["features"] if item["status"] == "active"]
        self.assertEqual([item["id"] for item in active], ["R05A"])
        evidence = active[0]["evidence"]
        self.assertIn("28048_0", evidence)
        self.assertIn("finite-difference", evidence)
        self.assertIn("apparatus-inconclusive", evidence)
        progress = (ROOT / "PROGRESS.md").read_text(encoding="utf-8")
        experiments = (ROOT / "EXPERIMENTS.md").read_text(encoding="utf-8")
        for text in (progress, experiments):
            self.assertIn("28048_0", text)
            self.assertIn("apparatus-inconclusive", text)
            self.assertIn("before FISTA", text)
            self.assertIn("probe", text.lower())


if __name__ == "__main__":
    unittest.main()
