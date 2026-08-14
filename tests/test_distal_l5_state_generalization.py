import unittest
from pathlib import Path

from main.multilink_ellipsoid.l5_state_generalization import (
    load_config, prediction_diagnostic, prediction_gate,
)


class L5StateGeneralizationTest(unittest.TestCase):
    def test_config_freezes_matched_action_family(self):
        config = load_config(
            Path("configs/vlsa_distal_l5_state_generalization.v1.json")
        )
        self.assertEqual(
            config["matched_split"]["test_A_definition"],
            "new_actions_at_known_states",
        )
        self.assertEqual(
            config["matched_split"]["test_B_definition"],
            "same_action_family_at_new_episode_states",
        )
        self.assertIn("reserved_test_label_access", config["forbidden"])

    def test_metrics_distinguish_false_safe_and_support(self):
        samples = [
            {"state_id": "a", "risk_l5": [-0.1, -0.1, -0.1]},
            {"state_id": "a", "risk_l5": [0.1, -0.1, -0.1]},
            {"state_id": "b", "risk_l5": [-0.2, -0.1, -0.1]},
        ]
        metrics = prediction_diagnostic([
            [-0.1, -0.1, -0.1],
            [-0.1, -0.1, -0.1],
            [0.1, -0.1, -0.1],
        ], samples)
        self.assertEqual(metrics["L5_false_safe_candidate_count"], 1)
        self.assertEqual(metrics["per_row_false_safe_count"], [1, 0, 0])
        self.assertEqual(metrics["recoverable_state_count"], 2)
        self.assertEqual(metrics["supported_recoverable_state_count"], 1)
        gates = prediction_gate(metrics, {
            "minimum_L5_safe_candidate_recall": 0.5,
        })
        self.assertFalse(gates["zero_observed_L5_false_safe_candidates"])
        self.assertFalse(gates["safe_support_every_recoverable_state"])


if __name__ == "__main__":
    unittest.main()
