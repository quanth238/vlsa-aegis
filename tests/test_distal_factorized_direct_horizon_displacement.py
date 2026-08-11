from pathlib import Path
import unittest

import numpy as np

from main.multilink_ellipsoid.factorized_direct_horizon_displacement import (
    _build_model, direct_horizon_decision, horizon_time_encoding,
    load_direct_horizon_config, temporal_error_metrics,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/vlsa_distal_factorized_direct_horizon_displacement_moka10.v1.json"


class DirectHorizonDisplacementTests(unittest.TestCase):
    def test_config_freezes_reserved_and_future_untouched_episodes(self):
        config = load_direct_horizon_config(CONFIG)
        self.assertEqual(config["architecture"]["trace_state_count"], 51)
        self.assertEqual(len(config["population"]["reserved_prediction_episode_ids"]), 5)
        self.assertEqual(
            len(config["population"]["future_untouched_intervention_episode_ids"]), 5
        )
        self.assertTrue(config["forbidden_before_prediction_pass"]["QP"])

    def test_time_encoding_has_state_zero_and_terminal(self):
        encoding = horizon_time_encoding()
        self.assertEqual(encoding.shape, (51, 9))
        self.assertTrue(np.all(np.isfinite(encoding)))
        self.assertEqual(float(encoding[0, 0]), 0.0)
        self.assertEqual(float(encoding[-1, 0]), 1.0)

    def test_decoder_predicts_each_horizon_directly_with_exact_q0(self):
        import torch

        config = load_direct_horizon_config(CONFIG)
        model = _build_model(12, config["architecture"])
        output = model(torch.zeros((3, 12), dtype=torch.float32))
        self.assertEqual(tuple(output.shape), (3, 51, 7))
        self.assertTrue(torch.equal(output[:, 0], torch.zeros((3, 7))))
        # A direct decoder can assign adjacent horizons independently: its
        # source contains no cumulative sum and it exposes 51 decoder calls.
        source = Path(
            ROOT / "main/multilink_ellipsoid/factorized_direct_horizon_displacement.py"
        ).read_text()
        forward = source[source.index("def forward", source.index("class Direct")):
                         source.index("return DirectHorizonDisplacementNet")]
        self.assertNotIn("cumsum", forward)
        self.assertIn("displacement[:, 1:]", forward)

    def test_temporal_metric_exposes_terminal_drift(self):
        exact = np.zeros((4, 51, 7), dtype=np.float64)
        predicted = np.zeros_like(exact)
        predicted[:, :, :] = np.arange(51)[None, :, None] * 0.001
        metrics = temporal_error_metrics(predicted, exact)
        self.assertEqual(metrics["maximum_initial_joint_error_rad"], 0.0)
        self.assertAlmostEqual(metrics["terminal_joint_RMSE_rad"], 0.05)
        self.assertAlmostEqual(metrics["linear_RMSE_slope_rad_per_substep"], 0.001)

    def test_decision_is_hard_conjunction(self):
        config = load_direct_horizon_config(CONFIG)
        safety = {
            "false_safe_action_count": 0,
            "exact_safe_action_recall": 0.95,
            "near_boundary_RMSE_m": 0.001,
        }
        exact_static = {
            "false_safe_action_count": 0,
            "exact_safe_action_recall": 1.0,
        }
        support = {"all_eligible_states_supported": True}
        temporal = {
            "overall_joint_RMSE_rad": 0.01,
            "terminal_joint_RMSE_rad": 0.015,
            "linear_RMSE_slope_rad_per_substep": 0.0001,
            "terminal_to_overall_RMSE_ratio": 1.5,
            "maximum_initial_joint_error_rad": 0.0,
        }
        cosine = {"mean_cosine": 0.9}
        decision = direct_horizon_decision(
            validation_metrics=safety, reserved_metrics=safety,
            reserved_exact_static_metrics=exact_static,
            validation_support=support, reserved_support=support,
            reserved_temporal=temporal, joint_sensitivity=cosine,
            safety_sensitivity=cosine,
            matched_cumulative={"linear_RMSE_slope_rad_per_substep": 0.01},
            config=config,
        )
        self.assertTrue(decision["prediction_gate_GO"])
        safety["false_safe_action_count"] = 1
        decision = direct_horizon_decision(
            validation_metrics=safety, reserved_metrics=safety,
            reserved_exact_static_metrics=exact_static,
            validation_support=support, reserved_support=support,
            reserved_temporal=temporal, joint_sensitivity=cosine,
            safety_sensitivity=cosine,
            matched_cumulative={"linear_RMSE_slope_rad_per_substep": 0.01},
            config=config,
        )
        self.assertFalse(decision["prediction_gate_GO"])


if __name__ == "__main__":
    unittest.main()
