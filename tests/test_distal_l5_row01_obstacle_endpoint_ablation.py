import unittest
from pathlib import Path

from main.multilink_ellipsoid.l5_row01_obstacle_endpoint_ablation import (
    classify, load_config, obstacle_endpoint_feature_vector,
)


class ObstacleEndpointAblationTest(unittest.TestCase):
    def test_relative_feature(self):
        actions = [[0.0] * 7 for _ in range(5)]
        for row in actions:
            row[0] = 0.1
        feature = obstacle_endpoint_feature_vector(
            eef_position_m=[1.0, 2.0, 3.0], candidate_actions=actions,
            obstacle_center_m=[0.5, 1.0, 2.0],
            obstacle_semiaxes_m=[0.1, 0.2, 0.3],
            translation_scale_m_per_action_unit=0.05,
        )
        self.assertEqual(feature[:3], [0.5, 1.0, 1.0])
        self.assertAlmostEqual(feature[3], 0.525)
        self.assertEqual(feature[6:], [0.1, 0.2, 0.3])

    def test_config_and_decision(self):
        config = load_config(Path(
            "configs/vlsa_distal_l5_row01_obstacle_endpoint_ablation.v1.json"
        ))
        self.assertEqual(config["features"]["input_dimension"], 9)
        endpoint = {"rmse_m": 0.016, "row01_false_safe_count": 6,
                    "supported_recoverable_state_count": 3}
        current = {"rmse_m": 0.012, "row01_false_safe_count": 6,
                   "supported_recoverable_state_count": 3}
        interpretation, _ = classify(
            current, endpoint, minimum_relative_rmse_improvement=0.1,
        )
        self.assertIn("is_useful", interpretation)


if __name__ == "__main__":
    unittest.main()
