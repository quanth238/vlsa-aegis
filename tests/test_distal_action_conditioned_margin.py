import unittest
from pathlib import Path

try:
    import numpy as np
except ImportError:  # pragma: no cover - local optional dependency
    np = None

from main.multilink_ellipsoid.action_conditioned_margin import (
    ACTION_FEATURE_NAMES, action_feature_matrix, build_model, load_config,
)
from main.multilink_ellipsoid.two_step_margin import (
    PAIR_CANDIDATE_SLICE, PAIR_FEATURE_NAMES,
)


ROOT = Path(__file__).resolve().parents[1]


@unittest.skipIf(np is None, "numpy is unavailable")
class ActionConditionedMarginTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(
            ROOT / "configs/vlsa_distal_action_conditioned_margin_moka10.v1.json"
        )

    def test_grouped_test_isolation_and_no_closed_loop(self):
        self.assertEqual(
            self.config["split"]["expected_state_counts"],
            {"train": 60, "validation": 10, "test": 15},
        )
        self.assertTrue(
            self.config["split"][
                "held_out_test_never_used_for_training_calibration_early_stopping_or_model_selection"
            ]
        )
        self.assertFalse(self.config["decision"]["closed_loop_in_this_gate"])

    def test_action_feature_replaces_candidate_and_adds_normalized_action(self):
        base = np.arange(7 * len(PAIR_FEATURE_NAMES), dtype=np.float64).reshape(
            7, len(PAIR_FEATURE_NAMES)
        )
        original = base.copy()
        action = np.asarray([0.2, -0.1, 0.5])
        lower = np.asarray([-0.2, -0.6, 0.0])
        upper = np.asarray([0.6, 0.4, 1.0])
        output = action_feature_matrix(base, action, lower, upper)
        self.assertEqual(output.shape, (7, len(ACTION_FEATURE_NAMES)))
        np.testing.assert_allclose(output[:, PAIR_CANDIDATE_SLICE], action)
        np.testing.assert_allclose(output[:, -3:], [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(base, original)

    def test_model_predicts_one_residual_per_constraint_action_row(self):
        import torch

        model = build_model(self.config["model"]["hidden_widths"]).to(
            dtype=torch.float64
        )
        values = torch.zeros((11, len(ACTION_FEATURE_NAMES)), dtype=torch.float64)
        self.assertEqual(tuple(model(values).shape), (11,))

    def test_value_not_coefficient_parameterization(self):
        self.assertEqual(
            self.config["model"]["output"], "scalar_margin_residual_mm"
        )
        self.assertEqual(
            self.config["regionalization"]["affine_fit_source"],
            "guarded_action_conditioned_values_not_learned_coefficients",
        )
        self.assertEqual(
            self.config["learned_gate"]["test_off_grid_false_safe_action_count"],
            0,
        )


if __name__ == "__main__":
    unittest.main()
