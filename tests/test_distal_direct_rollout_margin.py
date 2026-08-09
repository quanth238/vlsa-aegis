import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class DistalDirectRolloutMarginTests(unittest.TestCase):
    def test_config_freezes_grouped_e05_direct_margin_test(self):
        from main.multilink_ellipsoid.direct_rollout_margin import (
            load_direct_margin_config,
        )

        config = load_direct_margin_config(
            ROOT / "configs/vlsa_distal_direct_rollout_margin_e05.v1.json"
        )
        self.assertEqual(config["state_step"], 185)
        self.assertEqual(config["split"]["primary_e05_use"], "test_only_never_training_or_calibration")
        self.assertEqual(config["model"]["output_activation"], "linear_signed_margin_mm")
        self.assertEqual(config["matched_random"]["direction_count"], 256)
        self.assertFalse(config["decision_gate"]["closed_loop_authorized"])

    def test_mutated_binary_or_positive_output_is_rejected(self):
        from main.multilink_ellipsoid.direct_rollout_margin import (
            load_direct_margin_config,
        )

        source = ROOT / "configs/vlsa_distal_direct_rollout_margin_e05.v1.json"
        config = json.loads(source.read_text())
        config["model"]["output_activation"] = "sigmoid_contact_probability"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "model differs"):
                load_direct_margin_config(path)

    def test_training_uses_direct_value_direction_and_magnitude_losses(self):
        source = (
            ROOT / "main/multilink_ellipsoid/direct_rollout_margin.py"
        ).read_text()
        self.assertIn("nn.Softplus()", source)
        self.assertIn("layers.append(nn.Linear(previous, 1))", source)
        self.assertNotIn("BCEWithLogitsLoss", source)
        self.assertIn("cosine_loss", source)
        self.assertIn("magnitude_loss", source)
        self.assertIn("arrays[\"minimum\"] * 1000.0", source)

    def test_evaluator_compares_exact_margin_at_equal_radius(self):
        source = (
            ROOT / "scripts/evaluate_distal_direct_rollout_margin_e05.py"
        ).read_text()
        self.assertIn('"rho_improvement_m"', source)
        self.assertIn("sample_feasible_unit_directions", source)
        self.assertIn("project_direct_margin_action", source)
        self.assertIn('calibrated=calibrated', source)
        self.assertNotIn("_closed_loop", source)

    def test_h100_job_orders_training_simulation_and_validation(self):
        source = (
            ROOT / "slurm/distal_direct_rollout_margin_e05.sbatch"
        ).read_text()
        self.assertIn("H100", source)
        train = source.index("train_distal_direct_rollout_margin_e05.py")
        evaluate = source.index("evaluate_distal_direct_rollout_margin_e05.py")
        validate = source.index("validate_distal_direct_rollout_margin_e05.py")
        self.assertLess(train, evaluate)
        self.assertLess(evaluate, validate)


if __name__ == "__main__":
    unittest.main()
