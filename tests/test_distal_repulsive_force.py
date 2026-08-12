from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class DistalRepulsiveForceTests(unittest.TestCase):
    def _numpy(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("NumPy is available in the allocation evaluation environment")
        return np

    def test_preregistered_config_is_strict(self) -> None:
        from main.multilink_ellipsoid.repulsive_force import (
            load_repulsive_force_config,
        )

        config = load_repulsive_force_config(
            ROOT / "configs/vlsa_distal_repulsive_force_direction_e05.v1.json"
        )
        self.assertEqual(config["state_split"]["train_steps"], [182, 183])
        self.assertEqual(config["state_split"]["validation_steps"], [184])
        self.assertEqual(config["state_split"]["test_steps"], [185])
        self.assertEqual(config["comparators"]["random"], "paired_uniform_unit_sphere_directions")
        self.assertIn("not_population_generalization", config["claim_scope"])

    def test_softmin_and_fixed_direction(self) -> None:
        np = self._numpy()

        from main.multilink_ellipsoid.repulsive_force import (
            fixed_repulsive_direction,
            softmin,
            softmin_weights,
        )

        values = np.asarray([0.001, 0.01, 0.02])
        observed = softmin(values, 0.002)
        self.assertLess(observed, float(np.min(values)))
        weights = softmin_weights(values, 0.002)
        self.assertAlmostEqual(float(np.sum(weights)), 1.0)
        self.assertGreater(float(weights[0]), float(weights[1]))
        rows = np.zeros((7, 3), dtype=np.float64)
        rows[:, 0] = 1.0
        direction = fixed_repulsive_direction(np.arange(7) * 0.01, rows, 0.002)
        np.testing.assert_allclose(direction, [1.0, 0.0, 0.0])

    def test_random_directions_are_antithetic_equal_norm_and_feasible(self) -> None:
        np = self._numpy()

        from main.multilink_ellipsoid.repulsive_force import (
            paired_feasible_directions,
        )

        nominal = np.asarray([0.1, 0.2, -0.6])
        directions = paired_feasible_directions(128, 17, nominal, 0.25, 1.0)
        self.assertEqual(directions.shape, (128, 3))
        np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1.0)
        np.testing.assert_allclose(directions[0::2], -directions[1::2])
        self.assertLessEqual(float(np.max(np.abs(nominal + 0.25 * directions))), 1.0)

    def test_matched_random_p_value_uses_add_one_rule(self) -> None:
        np = self._numpy()

        from main.multilink_ellipsoid.repulsive_force import matched_random_p_value

        self.assertEqual(matched_random_p_value(2.0, np.asarray([0.0, 1.0])), 1.0 / 3.0)
        self.assertEqual(matched_random_p_value(1.0, np.asarray([1.0, 2.0])), 1.0)

    def test_runner_freezes_model_before_test_and_conditionally_continues(self) -> None:
        source = (
            ROOT / "scripts/evaluate_distal_repulsive_force_direction_e05.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"frozen_before_test_step"] = True', source)
        self.assertIn("if direction_pass:", source)
        self.assertIn("matched_random_p_value", source)
        self.assertNotIn("MultiConstraintQp", source)

    def test_independent_validator_recomputes_direction_gate(self) -> None:
        source = (
            ROOT / "scripts/validate_distal_repulsive_force_direction_e05.py"
        ).read_text(encoding="utf-8")
        self.assertIn("matched_random_p_value", source)
        self.assertIn("random correction norms differ", source)
        self.assertIn("result payload hash differs", source)
        self.assertIn("failed direction gate executed continuation", source)


if __name__ == "__main__":
    unittest.main()
