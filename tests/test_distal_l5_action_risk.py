import unittest
from pathlib import Path

import numpy as np

from main.multilink_ellipsoid.l5_action_risk import (
    feature_vector, load_config, prediction_metrics,
)


class L5ActionRiskTest(unittest.TestCase):
    def test_config_and_feature_shape(self):
        config = load_config(Path("configs/vlsa_distal_l5_action_risk_mlp.v1.json"))
        self.assertEqual(config["learned_scope"]["rows"], [0, 1, 2])
        nominal = np.zeros((5, 7))
        candidate = nominal.copy()
        candidate[:, 0] = 0.25
        feature = feature_vector(
            initial_clearance=np.arange(7) * 0.01,
            local_frame={
                "normal": [1, 0, 0], "tangent_up": [0, 1, 0],
                "tangent_side": [0, 0, 1],
            }, nominal_actions=nominal, candidate_actions=candidate,
        )
        self.assertEqual(len(feature), 86)
        self.assertEqual(feature[-35], 0.25)

    def test_metrics_separate_l5_false_safe_and_physical_veto(self):
        samples = [
            {"state_id": "a", "risk_l5": [-0.1, -0.1, -0.1], "exact_safe": True,
             "applied_correction_l2_action": 0.2, "candidate_order": 1,
             "candidate_name": "safe"},
            {"state_id": "a", "risk_l5": [0.1, -0.1, -0.1], "exact_safe": False,
             "applied_correction_l2_action": 0.0, "candidate_order": 0,
             "candidate_name": "false-safe"},
            {"state_id": "b", "risk_l5": [-0.1, -0.1, -0.1], "exact_safe": False,
             "applied_correction_l2_action": 0.1, "candidate_order": 2,
             "candidate_name": "physical-veto"},
        ]
        metrics = prediction_metrics([
            [-0.1, -0.1, -0.1], [-0.1, -0.1, -0.1], [-0.1, -0.1, -0.1]
        ], samples)
        self.assertEqual(metrics["L5_false_safe_count"], 1)
        self.assertEqual(metrics["predicted_L5_safe_but_all_seven_physical_veto_count"], 2)
        self.assertEqual(metrics["supported_recoverable_state_count"], 1)
        self.assertTrue(metrics["all_selected_candidates_exact_all_seven_safe"])


if __name__ == "__main__":
    unittest.main()
