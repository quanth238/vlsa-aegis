import json
import unittest
from pathlib import Path

from main.multilink_ellipsoid.generic_l5_9d_capacity import (
    diagnostic_metrics, feature_vector, load_config, state_balanced_weights,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/vlsa_distal_generic_l5_9d_capacity.v1.json"


class GenericL59DCapacityTest(unittest.TestCase):
    def test_config_freezes_diagnostic_only_protocol(self):
        config = load_config(CONFIG)
        self.assertEqual(config["model"]["output_count"], 3)
        self.assertEqual(config["dataset"]["L5_row_indices"], [0, 1, 2])
        self.assertIn("candidate_correction", config["forbidden"])

    def test_feature_uses_start_and_commanded_endpoint_relative_to_obstacle(self):
        actions = [[0.0] * 7 for _ in range(5)]
        for action in actions:
            action[:3] = [1.0, -0.5, 0.25]
        feature = feature_vector(
            eef_position_m=[0.2, 0.3, 0.4], candidate_actions=actions,
            obstacle_center_m=[0.1, 0.1, 0.1],
            obstacle_semiaxes_m=[0.2, 0.3, 0.4],
            translation_scale_m_per_action_unit=0.05,
        )
        self.assertEqual(feature[:3], [0.1, 0.19999999999999998, 0.30000000000000004])
        self.assertAlmostEqual(feature[3], 0.35)
        self.assertAlmostEqual(feature[4], 0.075)
        self.assertAlmostEqual(feature[5], 0.3625)
        self.assertEqual(feature[6:], [0.2, 0.3, 0.4])

    def test_state_balancing_gives_equal_total_weight_per_state(self):
        weights = state_balanced_weights(["a", "a", "a", "b"])
        self.assertAlmostEqual(sum(weights[:3]), weights[3])
        self.assertAlmostEqual(sum(weights), 4.0)

    def test_metrics_keep_false_safes_and_support_separate(self):
        samples = [
            {
                "state_id": "s", "candidate_name": "nominal", "candidate_order": 0,
                "risk_rows": [0.2, -0.1, -0.2], "applied_correction_l2_action": 0.0,
            },
            {
                "state_id": "s", "candidate_name": "escape", "candidate_order": 1,
                "risk_rows": [-0.1, -0.2, -0.3], "applied_correction_l2_action": 1.0,
            },
        ]
        metrics = diagnostic_metrics(
            samples, [[-0.1, -0.1, -0.1], [-0.2, -0.2, -0.2]],
            near_boundary_abs_risk=0.05,
        )
        self.assertEqual(metrics["false_safe_count"], 1)
        self.assertEqual(metrics["supported_state_count"], 1)
        self.assertEqual(metrics["selected_exact_safe_state_count"], 0)
        self.assertEqual(metrics["improvement_direction_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
