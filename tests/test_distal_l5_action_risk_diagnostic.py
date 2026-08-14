import unittest
from pathlib import Path

from main.multilink_ellipsoid.l5_action_risk_diagnostic import (
    load_config, split_train_actions,
)
from scripts.train_distal_l5_action_risk_diagnostic import _applied_correction_l2


class L5ActionRiskDiagnosticTests(unittest.TestCase):
    def test_config_is_capacity_only(self):
        config = load_config(Path(
            "configs/vlsa_distal_l5_action_risk_diagnostic.v1.json"
        ))
        self.assertEqual(config["learned_scope"]["rows"], [0, 1, 2])
        self.assertTrue(config["learned_scope"][
            "rows_1_2_local_interpolation_only"
        ])
        self.assertTrue(config["reporting"]["no_deployment_pass_gate"])
        self.assertIn("QP", config["forbidden"])
        self.assertIn("closed_loop_execution", config["forbidden"])

    def test_action_holdout_uses_order_not_target(self):
        samples = [
            {"state_id": "b", "candidate_order": 2, "candidate_name": "b2"},
            {"state_id": "a", "candidate_order": 3, "candidate_name": "a3"},
            {"state_id": "a", "candidate_order": 1, "candidate_name": "a1"},
            {"state_id": "b", "candidate_order": 0, "candidate_name": "b0"},
            {"state_id": "c", "candidate_order": 7, "candidate_name": "c7"},
        ]
        fit, heldout = split_train_actions(samples)
        self.assertEqual(
            [(item["state_id"], item["candidate_name"]) for item in heldout],
            [("a", "a3"), ("b", "b2")],
        )
        self.assertEqual(
            {(item["state_id"], item["candidate_name"]) for item in fit},
            {("a", "a1"), ("b", "b0"), ("c", "c7")},
        )

    def test_adaptive_midpoint_norm_uses_final_residual_binding(self):
        self.assertEqual(_applied_correction_l2({
            "applied_correction_l2_action": None,
            "residual_binding": {"applied_residual_l2_action": 1.25},
        }), 1.25)


if __name__ == "__main__":
    unittest.main()
