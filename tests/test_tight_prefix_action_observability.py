from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

from main.multilink_ellipsoid.tight_prefix_action_observability import (
    DIRECTION_NAMES, load_config, symmetric_action_overrides,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/vlsa_tight_prefix_action_observability.v1.json"


class TightPrefixActionObservabilityTests(unittest.TestCase):
    def test_config_freezes_validation_only_representation_gate(self):
        config = load_config(CONFIG)
        self.assertEqual(len(config["cases"]), 4)
        self.assertTrue(all(row["split"] == "validation" for row in config["cases"]))
        self.assertEqual(config["perturbation"]["directions"], list(DIRECTION_NAMES))
        self.assertIn("model_training_or_checkpoint_change", config["forbidden"])
        self.assertIn("full_episode_or_closed_loop", config["forbidden"])

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy unavailable")
    def test_symmetric_actions_share_radius_and_preserve_other_controls(self):
        import numpy as np

        nominal = np.zeros((5, 7), dtype=np.float64)
        nominal[:, 3:] = np.asarray([0.1, 0.2, 0.3, -0.7])
        frame = {
            "normal": [1.0, 0.0, 0.0],
            "tangent_up": [0.0, 0.0, 1.0],
            "tangent_side": [0.0, -1.0, 0.0],
        }
        actions, audit = symmetric_action_overrides(
            nominal, frame, maximum_radius=0.25, minimum_radius=0.01,
            headroom_fraction=0.8, action_limit=1.0,
        )
        self.assertEqual([row["name"] for row in actions[1:]], list(DIRECTION_NAMES))
        self.assertAlmostEqual(audit["common_translation_l2_action"], 0.25)
        self.assertLessEqual(audit["maximum_pair_symmetry_error"], 1e-12)
        self.assertLessEqual(audit["maximum_equal_norm_error"], 1e-12)
        for row in actions:
            np.testing.assert_array_equal(np.asarray(row["actions"])[:, 3:], nominal[:, 3:])

    def test_validator_detects_tangent_blind_material_descent(self):
        from scripts.validate_tight_prefix_action_observability import _case_report

        config = load_config(CONFIG)
        risks = {
            "nominal": 0.10,
            "normal_pos": 0.12,
            "normal_neg": 0.09,
            "tangent_up_pos": 0.08,
            "tangent_up_neg": 0.11,
            "tangent_side_pos": 0.10,
            "tangent_side_neg": 0.10,
        }

        def candidate(name):
            row_slack = [-risks[name]] + [1.0] * 9
            return {
                "name": name,
                "exact_group_target": {
                    "trace": [{"row_normalized_radial_slack": row_slack}],
                    "group_order": ["palm", "L5", "L6"],
                    "group_minimum_normalized_radial_slack": {
                        "palm": row_slack[0], "L5": 1.0, "L6": 1.0,
                    },
                    "group_contact_sample_count": {
                        "palm": 0, "L5": 0, "L6": 0,
                    },
                },
            }

        result = {
            "case_id": "synthetic",
            "active_frame": {"active_row": 0, "active_initial_slack": 0.1},
            "maximum_active_row_tangent_7D_feature_change": 0.0,
            "symmetry_audit": {"common_translation_l2_action": 0.25},
            "case": {"candidates": [candidate(name) for name in (
                "nominal", "normal_pos", "normal_neg", "tangent_up_pos",
                "tangent_up_neg", "tangent_side_pos", "tangent_side_neg",
            )]},
        }
        report = _case_report(result, config)
        self.assertTrue(report["active_row_7D_tangent_blind"])
        self.assertTrue(report["structural_counterexample"])
        self.assertTrue(report["material_tangent_descent"])
        self.assertEqual(report["best_tangent_direction"], "tangent_up_pos")


if __name__ == "__main__":
    unittest.main()
