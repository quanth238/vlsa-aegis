from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_PATH = ROOT / "evidence" / "r05a" / "cfs00a-fd-diagnostic-20260716b.json"
ADR_PATH = (
    ROOT
    / "docs"
    / "decisions"
    / "0049-accept-numeric-diagnostic-and-retire-autograd-linearization.md"
)


class CFS00AFailedDerivativeDiagnosticBEvidenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.evidence = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))

    def test_exact_terminal_identity_and_publication_are_frozen(self) -> None:
        self.assertEqual(
            self.evidence["run_id"],
            "r05a-constrained-flow-fd-diagnostic-20260716b",
        )
        self.assertEqual(
            self.evidence["release_commit"],
            "1e2d36de9cb009a07e75d3535ab6f7070c5ea34c",
        )
        gpu = self.evidence["jobs"]["gpu_array_task"]
        cpu = self.evidence["jobs"]["cpu_afterany_validator"]
        self.assertEqual(
            (gpu["job_id"], gpu["state"], gpu["exit_code"], gpu["node"], gpu["elapsed"]),
            ("28222_0", "COMPLETED", "0:0", "worker-1", "00:02:56"),
        )
        self.assertEqual(
            (cpu["job_id"], cpu["state"], cpu["exit_code"], cpu["node"], cpu["elapsed"]),
            ("28223", "COMPLETED", "0:0", "worker-0", "00:00:06"),
        )
        artifacts = self.evidence["immutable_artifacts"]
        self.assertEqual(
            artifacts["source_contract_sha256"],
            "956b4f8b109e4a7f2498ad0fe6871db1249078152bed03aecc43d98494658b78",
        )
        self.assertEqual(
            artifacts["submission_sha256"],
            "7f36201e847672c7c67887ff6176cbc899c934b1b594c1420b9d5e985b55e16c",
        )
        self.assertEqual(
            artifacts["constrained_flow_payload_sha256"],
            "37edb7bc6f0961563cbcbfbf938e156211efc9397f6e7431ec97bde3ee846f41",
        )
        self.assertEqual(
            artifacts["published_result_sha256"],
            "955d7b88f1b9c55dff77dfeb2ee19a3e055670f59e1ddab5acb41d67b13659aa",
        )
        self.assertEqual(
            artifacts["cpu_publication_receipt_sha256"],
            "29e926ad9edba1e94d448d2938ca7965b73117647e0b78900d77d04bb8011200",
        )

    def test_all_registered_numeric_checks_failed_without_tuning(self) -> None:
        numeric = self.evidence["numeric_payload"]
        self.assertEqual(numeric["reason_code"], "AUTOGRAD_JACOBIAN_FD_REJECTED")
        self.assertEqual(numeric["jacobian"]["shape"], [35, 75])
        self.assertEqual(numeric["jacobian"]["dtype"], "float32")
        self.assertEqual(numeric["budget"]["value"], 3.6398398876190186)
        self.assertEqual(
            numeric["epsilon_values"],
            [0.0028436249122023582, 0.0014218124561011791],
        )
        self.assertEqual(numeric["relative_l2_tolerance"], 0.1)
        self.assertEqual(numeric["absolute_l2_tolerance"], 0.001)
        self.assertEqual(
            numeric["checks_passed"],
            [[False, False], [False, False], [False, False]],
        )
        self.assertEqual(numeric["directions_passed"], [False, False, False])
        self.assertIs(numeric["global_passed"], False)

    def test_central_differences_are_unstable_at_both_frozen_scales(self) -> None:
        analysis = self.evidence["descriptive_numeric_analysis"]
        self.assertEqual(
            analysis["cross_epsilon_central_relative_l2"],
            [0.818859503, 0.937044896, 1.186637160],
        )
        self.assertEqual(
            analysis["cross_epsilon_central_cosine"],
            [0.657143657, 0.552737909, 0.162468229],
        )
        self.assertTrue(
            all(value > 0.8 for value in analysis["cross_epsilon_central_relative_l2"])
        )
        self.assertIn("not a faithful local model", ADR_PATH.read_text(encoding="utf-8"))

    def test_mixed_precision_is_plausible_but_not_claimed_as_isolated(self) -> None:
        audit = self.evidence["mixed_precision_source_audit"]
        self.assertEqual(audit["runtime_model_dtype"], "bfloat16")
        self.assertIs(audit["mechanism_isolated_by_this_run"], False)
        source = ROOT / audit["gemma_source_path"]
        self.assertEqual(
            hashlib.sha256(source.read_bytes()).hexdigest(),
            audit["gemma_source_sha256"],
        )
        text = source.read_text(encoding="utf-8")
        self.assertIn("hidden_states = hidden_states.to(torch.bfloat16)", text)
        adr = ADR_PATH.read_text(encoding="utf-8")
        normalized_adr = " ".join(adr.split())
        self.assertIn("credible mechanism", normalized_adr)
        self.assertIn("does not isolate quantization", normalized_adr)

    def test_boundary_remains_before_candidate_efficacy_and_training(self) -> None:
        boundary = self.evidence["execution_boundary"]
        self.assertTrue(boundary["client_terminal_normalization_completed"])
        self.assertTrue(boundary["numeric_jacobian_persisted"])
        self.assertTrue(boundary["cpu_independent_reconstruction_passed"])
        for key in (
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
        self.assertFalse(interpretation["autograd_linearized_submethod_supported"])
        self.assertFalse(interpretation["broader_actual_forward_flow_transport_confirmed"])
        self.assertFalse(interpretation["broader_actual_forward_flow_transport_refuted"])
        self.assertFalse(interpretation["ift01_authorized"])
        self.assertFalse(interpretation["probe_or_mlp_training_authorized"])

    def test_resources_are_excluded_and_checked_in_execution_is_closed(self) -> None:
        operational = self.evidence["operational_validation"]
        self.assertEqual(operational["allocation_tests_expected"], 99)
        self.assertEqual(operational["allocation_tests_observed"], 99)
        self.assertEqual(operational["allocation_test_skips"], 0)
        self.assertLess(
            operational["host_sampled_current_high_water_bytes"],
            operational["host_memory_max_before_bytes"],
        )
        self.assertEqual(operational["gpu_sampled_device_high_water_mib"], 8513)
        self.assertEqual(
            operational["memory_event_deltas"],
            {"max": 0, "oom": 0, "oom_kill": 0},
        )
        for path in (
            ROOT / "configs" / "experiments" / "r05a_constrained_flow_canary.json",
            ROOT
            / "configs"
            / "experiments"
            / "r05a_constrained_flow_canary_apparatus.json",
        ):
            value = json.loads(path.read_text(encoding="utf-8"))
            self.assertFalse(value["ready_to_run"])
            self.assertTrue(value["blocked_on"])
            self.assertIsNone(value["execution_release"])

    def test_trackers_preserve_the_narrow_conclusion_and_next_direct_test(self) -> None:
        decisions = (ROOT / "DECISIONS.md").read_text(encoding="utf-8")
        self.assertIn(ADR_PATH.relative_to(ROOT).as_posix(), decisions)
        for relative in ("README.md", "PROGRESS.md", "EXPERIMENTS.md"):
            text = (ROOT / relative).read_text(encoding="utf-8")
            self.assertIn("28222_0", text, relative)
            self.assertIn("28223", text, relative)
            self.assertIn("actual-forward", text, relative)
            self.assertIn("probe", text.lower(), relative)
        feature = json.loads((ROOT / "feature_list.json").read_text(encoding="utf-8"))
        by_id = {item["id"]: item for item in feature["features"]}
        active = [
            item["id"] for item in feature["features"] if item["status"] == "active"
        ]
        self.assertEqual(active, ["R06"])
        self.assertEqual(by_id["R04"]["status"], "blocked")
        self.assertEqual(by_id["R05A"]["status"], "blocked")
        self.assertIn("finite-difference", by_id["R05A"]["evidence"])
        self.assertIn(
            "probe/MLP training remains forbidden",
            by_id["R06"]["evidence"],
        )


if __name__ == "__main__":
    unittest.main()
