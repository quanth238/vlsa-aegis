from pathlib import Path
import importlib.util
import unittest


ROOT = Path(__file__).resolve().parents[1]
HAS_NUMPY = importlib.util.find_spec("numpy") is not None


class DistalFieldOracleTests(unittest.TestCase):
    def test_config_freezes_dangerous_actions_and_three_arms(self) -> None:
        from main.multilink_ellipsoid.field_oracle import load_field_oracle_config

        config = load_field_oracle_config(
            ROOT / "configs/vlsa_distal_field_mixture_oracle_e05.v1.json"
        )
        self.assertEqual(config["nominal"]["action_steps"], [185, 186])
        self.assertEqual(config["search"]["iterations"], 4)
        self.assertEqual(len(config["comparison"]["arms"]), 3)

    @unittest.skipUnless(HAS_NUMPY, "NumPy is available in the H100 evaluation environment")
    def test_task_tangent_is_orthogonal_to_active_rows(self) -> None:
        import numpy as np

        from main.multilink_ellipsoid.field_oracle import task_tangent_direction

        rows = np.zeros((7, 6), dtype=np.float64)
        rows[0, 0] = 1.0
        rows[1, 1] = 1.0
        rows[2:, 2] = 1.0
        margins = np.asarray([-0.01, -0.009, 0.1, 0.1, 0.1, 0.1, 0.1])
        task = np.asarray([1.0, 2.0, 0.0, 0.0, 1.0, 0.0])
        tangent, active = task_tangent_direction(
            rows, margins, task, active_band_m=0.005
        )
        self.assertEqual(active, [0, 1])
        self.assertLessEqual(float(np.max(np.abs(rows[active].dot(tangent)))), 1e-10)
        self.assertAlmostEqual(float(np.linalg.norm(tangent)), 1.0, places=12)

    @unittest.skipUnless(HAS_NUMPY, "NumPy is available in the H100 evaluation environment")
    def test_mixture_directions_have_unit_norm(self) -> None:
        import numpy as np

        from main.multilink_ellipsoid.field_oracle import normal_mixture_directions

        rows = np.arange(42, dtype=np.float64).reshape(7, 6) / 41.0
        directions = normal_mixture_directions(
            rows, np.linspace(-0.01, 0.02, 7), count=16, seed=7
        )
        self.assertEqual(directions.shape, (16, 6))
        np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1.0)

    def test_evaluator_uses_exact_rollouts_and_no_learning(self) -> None:
        source = (
            ROOT / "scripts/evaluate_distal_field_mixture_oracle_e05.py"
        ).read_text(encoding="utf-8")
        self.assertIn("probe.transition", source)
        self.assertIn('for arm_name in config["comparison"]["arms"]', source)
        self.assertIn('"safe_and_task_progressing"', source)
        self.assertNotIn("train_monotone_potential", source)
        self.assertNotIn("MultiConstraintQp", source)

    def test_independent_validator_recomputes_all_scientific_gates(self) -> None:
        source = (
            ROOT / "scripts/validate_distal_field_mixture_oracle_e05.py"
        ).read_text(encoding="utf-8")
        self.assertIn("result payload hash differs", source)
        self.assertIn("candidate exact-safe flag differs", source)
        self.assertIn("matched correction norm differs", source)
        self.assertIn("overall gate differs", source)


if __name__ == "__main__":
    unittest.main()
