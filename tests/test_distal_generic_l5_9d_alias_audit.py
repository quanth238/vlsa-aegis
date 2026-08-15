import unittest
from pathlib import Path

from main.multilink_ellipsoid.generic_l5_9d_alias_audit import (
    direct_l5_context_vector, load_config, oracle_anchor_predictions,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/vlsa_distal_generic_l5_9d_alias_audit.v1.json"


class GenericL59DAliasAuditTest(unittest.TestCase):
    def test_config_forbids_training_and_control(self):
        config = load_config(CONFIG)
        self.assertIn("model_training_or_finetuning", config["forbidden"])
        self.assertIn("candidate_correction", config["forbidden"])

    def test_oracle_anchor_changes_only_row_offset(self):
        samples = [
            {"candidate_name": "nominal", "risk_rows": [0.2, 0.1, -0.1]},
            {"candidate_name": "escape", "risk_rows": [0.0, -0.1, -0.2]},
        ]
        anchored = oracle_anchor_predictions(
            samples, [[-0.3, -0.4, -0.5], [-0.4, -0.6, -0.7]],
        )
        self.assertAlmostEqual(anchored[0][0], 0.2)
        self.assertAlmostEqual(anchored[1][0], 0.1)
        self.assertAlmostEqual(anchored[1][1], -0.1)

    def test_direct_context_contains_three_relative_L5_poses(self):
        row = {
            "body_name": "robot0_link5", "center_m": [2.0, 3.0, 4.0],
            "rotation": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "semiaxes_m": [0.1, 0.2, 0.3],
        }
        context = {
            "obstacle": {"center_m": [1.0, 1.0, 1.0]},
            "geometry_rows": [dict(row), dict(row), dict(row)],
        }
        vector = direct_l5_context_vector(context)
        self.assertEqual(len(vector), 45)
        self.assertEqual(vector[:3], [1.0, 2.0, 3.0])


if __name__ == "__main__":
    unittest.main()
