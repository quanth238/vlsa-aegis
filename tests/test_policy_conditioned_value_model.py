import unittest

from main.multilink_ellipsoid.policy_conditioned_value_model import (
    risk_metrics, state_balanced_weights,
)


class PolicyConditionedValueModelTest(unittest.TestCase):
    def test_state_constraint_pairs_are_balanced(self):
        samples = [
            {"state_id": "E00", "constraint": "L5"},
            {"state_id": "E00", "constraint": "L5"},
            {"state_id": "E01", "constraint": "L5"},
        ]
        weights = state_balanced_weights(samples)
        self.assertAlmostEqual(weights[0] + weights[1], weights[2])

    def test_metrics_fail_false_safe_and_preserve_support(self):
        samples = [
            {"state_id": "E00", "candidate_name": "safe", "constraint": "L5", "target": -0.1},
            {"state_id": "E00", "candidate_name": "unsafe", "constraint": "L5", "target": 0.2},
            {"state_id": "E01", "candidate_name": "safe", "constraint": "L5", "target": -0.05},
        ]
        metrics = risk_metrics(samples, [-0.2, -0.1, -0.01], 0.25)
        self.assertEqual(metrics["false_safe_count"], 1)
        self.assertEqual(metrics["recoverable_state_count"], 2)
        self.assertEqual(metrics["safe_support_state_count"], 2)


if __name__ == "__main__":
    unittest.main()
