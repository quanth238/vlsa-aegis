from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence/r05a/af00a-sealed-population-diagnostic.json"
ADR = ROOT / "docs/decisions/0059-interpret-af00a-population-and-test-reference-lift.md"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class R05AAF00APopulationDiagnosticEvidenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.value = json.loads(EVIDENCE.read_text(encoding="utf-8"))

    def test_analysis_sources_and_sealed_inputs_are_exactly_bound(self) -> None:
        analysis = self.value["reproducible_analysis"]
        for path_key, hash_key in (
            ("module", "module_sha256"),
            ("cli", "cli_sha256"),
            ("focused_test", "focused_test_sha256"),
        ):
            path = ROOT / analysis[path_key]
            self.assertTrue(path.is_file())
            self.assertEqual(_sha256(path), analysis[hash_key])
        self.assertEqual(
            analysis["deterministic_full_report_sha256"],
            "b1c6265441aa4c9ea0b364e1bf792a11cfdaf57744a07be532174c7414aedc19",
        )
        self.assertEqual(len(self.value["sealed_input_sha256"]), 5)
        self.assertTrue(self.value["integrity_validation"]["all_registered_hashes_match"])
        self.assertTrue(
            self.value["integrity_validation"]["existing_af_semantic_validator_passed"]
        )

    def test_exact_decomposition_records_the_opposing_field_response(self) -> None:
        exact = self.value["exact_transport_decomposition"]
        arm_a = exact["arm_a"]
        self.assertAlmostEqual(arm_a["direct_target_alpha"], 1.0, places=7)
        self.assertLess(arm_a["field_feedback_target_alpha"], -0.40)
        self.assertLess(
            arm_a["terminal_target_alpha"],
            exact["necessary_xyz_rms_pass_alpha_lower_bound"],
        )
        population = exact["population"]
        self.assertEqual(population["strictly_negative_field_feedback_count"], 520)
        self.assertEqual(
            population["terminal_alpha_meets_necessary_xyz_rms_bound_count"], 0
        )
        self.assertLess(
            population["terminal_target_alpha_max"],
            exact["necessary_xyz_rms_pass_alpha_lower_bound"],
        )

    def test_search_and_surrogate_are_not_promoted_to_feasibility(self) -> None:
        search = self.value["exact_frozen_search_diagnostic"]
        self.assertEqual(search["first_generation_better_than_arm_a"], 7)
        self.assertEqual(search["queries_better_than_arm_a"], 3)
        self.assertEqual(search["search_queries"], 520)
        surrogate = self.value["exploratory_surrogate"]
        self.assertFalse(
            surrogate["same_budget_constrained_prediction"]["actual_sampler_evaluation"]
        )
        boundaries = self.value["claim_boundaries"]
        for name, allowed in boundaries.items():
            with self.subTest(name=name):
                self.assertFalse(allowed)

    def test_decision_selects_the_direct_reference_field_test(self) -> None:
        interpretation = self.value["interpretation"]
        self.assertFalse(interpretation["constant_delta_over_time_conversion_supported"])
        self.assertTrue(interpretation["state_and_time_dependent_reference_lift_test_required"])
        self.assertFalse(interpretation["global_reachability_resolved"])
        decision = " ".join(ADR.read_text(encoding="utf-8").split())
        for phrase in (
            "state- and time-dependent",
            "reference-trajectory lift",
            "not global infeasibility",
            "Continue to forbid simulator execution of generated actions, IFT-01",
            "evidence/r05a/af00a-sealed-population-diagnostic.json",
        ):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, decision)


if __name__ == "__main__":
    unittest.main()
