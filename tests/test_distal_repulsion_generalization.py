import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class RepulsionGeneralizationTests(unittest.TestCase):
    def test_config_and_selection_manifest_are_frozen(self) -> None:
        from main.multilink_ellipsoid.repulsion_generalization import load_cases, load_config

        config = load_config(
            ROOT / "configs/vlsa_distal_repulsion_generalization_pilot.v1.json"
        )
        cases = load_cases(
            ROOT / "manifests/vlsa_distal_repulsion_generalization_pilot.v1.jsonl",
            config,
        )
        self.assertEqual(
            [case["case_id"] for case in cases],
            [
                "vlsa-t1-goal-ii-t0-e00",
                "vlsa-t1-goal-ii-t0-e24",
                "vlsa-t1-spatial-i-t1-e12",
            ],
        )
        self.assertEqual([case["intervention_step"] for case in cases], [165, 154, 105])
        self.assertTrue(config["state_protocol"]["policy_query_disabled"])

        from main.multilink_ellipsoid.shadow import load_shadow_config

        geometry = load_shadow_config(
            ROOT
            / "configs/vlsa_distal_slabbed_ellipsoid_shadow_generalization_pilot.v4.json"
        )
        self.assertEqual(geometry["case_ids"], [case["case_id"] for case in cases])

    def test_aggregate_keeps_mechanism_and_strict_gates_separate(self) -> None:
        from main.multilink_ellipsoid.repulsion_generalization import aggregate_results

        config = json.loads(
            (ROOT / "configs/vlsa_distal_repulsion_generalization_pilot.v1.json").read_text()
        )

        def result(case_id: str, smooth_0: bool, smooth_1: bool, fixed_0: bool) -> dict:
            def arm(zero: bool, one: bool, gain: float) -> dict:
                return {"clearance_gain_m": gain, "gates": {"buffer_0mm": zero, "buffer_1mm": one}}

            return {
                "case_id": case_id,
                "arms": {
                    "raw_aegis": arm(False, False, 0.0),
                    "analytical": arm(fixed_0, False, 0.001),
                    "smooth": arm(smooth_0, smooth_1, 0.002),
                    "exact_search": arm(True, True, 0.003),
                },
            }

        values = [
            result("a", True, True, True),
            result("b", True, False, False),
            result("c", False, False, False),
        ]
        aggregate = aggregate_results(values, config)
        self.assertTrue(aggregate["mechanism_gate_pass"])
        self.assertFalse(aggregate["strict_generalization_gate_pass"])
        self.assertTrue(aggregate["smooth_beats_or_matches_fixed_safe_rate"])

    def test_evaluator_is_parameterized_and_has_no_policy_server(self) -> None:
        source = (
            ROOT / "scripts/evaluate_distal_repulsion_generalization_pilot.py"
        ).read_text()
        self.assertIn("--case-index", source)
        self.assertIn("InstrumentedContinuationProbe", source)
        self.assertIn("_run_exact_search", source)
        self.assertNotIn("WebsocketClientPolicy", source)
        self.assertNotIn("policy_port", source)

    def test_task_valid_replacement_is_frozen_and_preserves_algorithm(self) -> None:
        from main.multilink_ellipsoid.repulsion_generalization import (
            load_task_valid_cases,
            load_task_valid_config,
        )

        config = load_task_valid_config(
            ROOT / "configs/vlsa_distal_repulsion_task_valid_e38.v1.json"
        )
        cases = load_task_valid_cases(
            ROOT / "manifests/vlsa_distal_repulsion_task_valid_e38.v1.jsonl",
            config,
        )
        self.assertEqual([case["case_id"] for case in cases], ["vlsa-t1-goal-ii-t3-e38"])
        self.assertEqual(cases[0]["intervention_step"], 108)
        self.assertEqual(cases[0]["first_relevant_contact_step"], 113)
        self.assertEqual(config["eligibility_gate"]["aegis_native_task_success"], True)
        self.assertEqual(config["eligibility_gate"]["baseline_native_task_success"], True)
        self.assertTrue(config["state_protocol"]["policy_query_disabled"])

        from main.multilink_ellipsoid.shadow import load_shadow_config

        geometry = load_shadow_config(
            ROOT / "configs/vlsa_distal_slabbed_ellipsoid_shadow_task_valid_e38.v4.json"
        )
        self.assertEqual(geometry["case_ids"], ["vlsa-t1-goal-ii-t3-e38"])

    def test_task_valid_validator_checks_eligibility_and_internal_steps(self) -> None:
        source = (
            ROOT / "scripts/validate_distal_repulsion_task_valid_e38.py"
        ).read_text()
        self.assertIn('eligibility["baseline_native_task_success"]', source)
        self.assertIn('eligibility["task_object_eef_distance_m"] <= 0.02', source)
        self.assertIn("all(count == 25", source)

    def test_improper_mvee_basis_is_canonicalized_without_refitting(self) -> None:
        source = (
            ROOT / "scripts/evaluate_distal_repulsion_generalization_pilot.py"
        ).read_text()
        self.assertIn("obstacle_rotation[:, -1] *= -1.0", source)
        self.assertIn("flip_last_eigenvector_preserves_centered_ellipsoid", source)


if __name__ == "__main__":
    unittest.main()
