import unittest

from main.multilink_ellipsoid.whole_body_action_selection_audit import (
    candidate_records, evaluate_rule, validation_optimistic_margin,
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


if __name__ == "__main__":
    unittest.main()
