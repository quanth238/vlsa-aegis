import unittest

from main.multilink_ellipsoid.whole_body_action_selection_audit import (
    candidate_records, evaluate_ranked_exact_verification, evaluate_rule,
    validation_optimistic_margin,
)


def sample(state, candidate, order, correction, row, risk):
    return {
        "state_id": state, "candidate_name": candidate,
        "candidate_order": order, "applied_correction_l2_action": correction,
        "row_index": row, "risk": risk,
    }


class WholeBodyActionSelectionAuditTest(unittest.TestCase):
    def setUp(self):
        self.samples = []
        self.predictions = []
        for name, order, correction, actual, predicted in (
            ("nominal", 0, 0.0, 0.2, -0.1),
            ("corrected", 1, 1.0, -0.2, -0.3),
        ):
            for row in range(1, 7):
                self.samples.append(sample("s", name, order, correction, row, actual))
                self.predictions.append([predicted])
        self.records = candidate_records(self.samples, self.predictions)

    def test_zero_threshold_can_select_false_safe_nominal(self):
        result = evaluate_rule(
            self.records, rule="least_intervention_predicted_safe",
        )
        self.assertEqual(result["false_safe_selected_state_count"], 1)
        self.assertEqual(result["states"][0]["selected_candidate"], "nominal")

    def test_minimum_risk_selects_corrected_candidate(self):
        result = evaluate_rule(self.records, rule="minimum_predicted_risk")
        self.assertEqual(result["false_safe_selected_state_count"], 0)
        self.assertEqual(result["states"][0]["selected_candidate"], "corrected")

    def test_validation_margin_bounds_observed_optimism(self):
        self.assertAlmostEqual(validation_optimistic_margin(self.records), 0.3)
        result = evaluate_rule(
            self.records, rule="least_intervention_predicted_safe", margin=0.3,
        )
        self.assertEqual(result["false_safe_selected_state_count"], 0)

    def test_ranked_verifier_finds_safe_second_candidate(self):
        result = evaluate_ranked_exact_verification(self.records)
        self.assertEqual(result["maximum_candidates_checked"], 1)
        # Reverse the predictions so the unsafe nominal is ranked first.
        reversed_predictions = []
        for sample_row in self.samples:
            reversed_predictions.append(
                [-0.3] if sample_row["candidate_name"] == "nominal" else [-0.1]
            )
        records = candidate_records(self.samples, reversed_predictions)
        result = evaluate_ranked_exact_verification(records)
        self.assertEqual(result["maximum_candidates_checked"], 2)
        self.assertEqual(result["top_k_safe_support"]["1"], 0)
        self.assertEqual(result["top_k_safe_support"]["3"], 1)


if __name__ == "__main__":
    unittest.main()
