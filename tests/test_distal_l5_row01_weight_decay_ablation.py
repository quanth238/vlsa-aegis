import unittest
from pathlib import Path

from main.multilink_ellipsoid.l5_row01_weight_decay_ablation import (
    classify_effect, load_config,
)


class L5Row01WeightDecayAblationTest(unittest.TestCase):
    def test_registered_config_changes_only_weight_decay(self):
        config = load_config(Path(
            "configs/vlsa_distal_l5_row01_weight_decay_ablation.v1.json"
        ))
        self.assertEqual(config["arms"]["no_weight_decay"]["weight_decay"], 0.0)
        self.assertEqual(
            config["arms"]["established_weight_decay"]["weight_decay"],
            1.0e-4,
        )
        self.assertIn("loss_change", config["forbidden"])
        self.assertIn("reserved_test_access", config["forbidden"])

    def test_effect_requires_rmse_false_safe_and_support_improvements(self):
        no_decay = {
            "rmse_m": 0.014, "row01_false_safe_count": 14,
            "supported_recoverable_state_count": 2,
        }
        established = {
            "rmse_m": 0.011, "row01_false_safe_count": 12,
            "supported_recoverable_state_count": 2,
        }
        interpretation, comparison = classify_effect(
            no_decay, established,
            minimum_relative_validation_improvement=0.1,
        )
        self.assertIn("materially_helps", interpretation)
        self.assertGreater(
            comparison["relative_validation_RMSE_improvement"], 0.1
        )
        established["supported_recoverable_state_count"] = 1
        interpretation, _ = classify_effect(
            no_decay, established,
            minimum_relative_validation_improvement=0.1,
        )
        self.assertIn("does_not_explain", interpretation)


if __name__ == "__main__":
    unittest.main()
