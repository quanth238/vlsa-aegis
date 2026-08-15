import unittest
from pathlib import Path

from main.multilink_ellipsoid.l5_row01_clearance_endpoint_ablation import classify, clearance_endpoint_feature_vector, load_config


class ClearanceEndpointAblationTest(unittest.TestCase):
    def test_feature_and_config(self):
        feature = clearance_endpoint_feature_vector(obstacle_endpoint_feature=[0.1] * 9, current_L5_clearances_m=[0.01, 0.02, 0.03])
        self.assertEqual(len(feature), 12)
        self.assertEqual(feature[-3:], [0.01, 0.02, 0.03])
        config = load_config(Path("configs/vlsa_distal_l5_row01_clearance_endpoint_ablation.v1.json"))
        self.assertEqual(config["features"]["input_dimension"], 12)

    def test_decision(self):
        prior = {"rmse_m": 0.0073, "row01_false_safe_count": 6, "supported_recoverable_state_count": 4, "selected_exact_safe_recoverable_state_count": 1}
        current = {"rmse_m": 0.005, "row01_false_safe_count": 4, "supported_recoverable_state_count": 4, "selected_exact_safe_recoverable_state_count": 2}
        interpretation, _ = classify(current, prior, minimum_relative_rmse_improvement=0.1)
        self.assertIn("are_useful", interpretation)


if __name__ == "__main__":
    unittest.main()
