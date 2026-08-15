import unittest
from pathlib import Path

from main.multilink_ellipsoid.l5_row01_endpoint_ablation import (
    classify, endpoint_feature_vector, load_config,
)


class L5Row01EndpointAblationTest(unittest.TestCase):
    def test_endpoint_uses_scale_only_and_all_five_commands(self):
        actions = [[0.0] * 7 for _ in range(5)]
        for index in range(5):
            actions[index][0] = 0.1
            actions[index][1] = -0.2
        feature = endpoint_feature_vector(
            eef_position_m=[1.0, 2.0, 3.0], candidate_actions=actions,
            translation_scale_m_per_action_unit=0.05,
        )
        self.assertEqual(feature[:3], [1.0, 2.0, 3.0])
        self.assertAlmostEqual(feature[3], 1.025)
        self.assertAlmostEqual(feature[4], 1.95)
        self.assertAlmostEqual(feature[5], 3.0)

    def test_registered_config_is_endpoint_only(self):
        config = load_config(Path(
            "configs/vlsa_distal_l5_row01_endpoint_ablation.v1.json"
        ))
        self.assertEqual(config["endpoint"]["input_dimension"], 6)
        self.assertTrue(
            config["endpoint"][
                "no_executed_future_EE_position_or_other_rollout_quantity"
            ]
        )
        self.assertIn("dataset_or_split_change", config["forbidden"])

    def test_decision_requires_prediction_and_support_improvement(self):
        complete = {
            "rmse_m": 0.011, "row01_false_safe_count": 13,
            "supported_recoverable_state_count": 3,
        }
        endpoint = {
            "rmse_m": 0.008, "row01_false_safe_count": 8,
            "supported_recoverable_state_count": 3,
        }
        interpretation, _ = classify(
            endpoint, complete, minimum_relative_rmse_improvement=0.1,
        )
        self.assertIn("materially_improves", interpretation)
        endpoint["row01_false_safe_count"] = 13
        interpretation, _ = classify(
            endpoint, complete, minimum_relative_rmse_improvement=0.1,
        )
        self.assertIn("insufficient", interpretation)


if __name__ == "__main__":
    unittest.main()
