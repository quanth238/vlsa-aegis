from pathlib import Path
import unittest

import numpy as np

from main.multilink_ellipsoid.factorized_cumulative_delta_q import (
    _build_model, cumulative_delta_decision, load_cumulative_delta_config,
    transition_time_encoding,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/vlsa_distal_factorized_cumulative_delta_q_moka10.v1.json"


class CumulativeDeltaQTests(unittest.TestCase):
    def test_config_freezes_prediction_before_reserved_collection(self):
        config = load_cumulative_delta_config(CONFIG)
        self.assertEqual(config["architecture"]["transition_count"], 50)
        self.assertTrue(config["forbidden_actions"]["QP"])
        self.assertEqual(len(config["population"]["reserved_final_episode_ids"]), 5)

    def test_time_encoding_is_finite_midpoint_grid(self):
        encoding = transition_time_encoding()
        self.assertEqual(encoding.shape, (50, 9))
        self.assertTrue(np.all(np.isfinite(encoding)))
        self.assertAlmostEqual(float(encoding[0, 0]), 0.01)
        self.assertAlmostEqual(float(encoding[-1, 0]), 0.99)

    def test_decoder_has_exact_initial_condition_and_cumulative_output(self):
        import torch

        config = load_cumulative_delta_config(CONFIG)
        model = _build_model(12, config["architecture"])
        output = model(torch.zeros((3, 12), dtype=torch.float32))
        self.assertEqual(tuple(output.shape), (3, 51, 7))
        self.assertTrue(torch.equal(output[:, 0], torch.zeros((3, 7))))
        increments = output[:, 1:] - output[:, :-1]
        reconstructed = torch.cat((
            torch.zeros((3, 1, 7)), torch.cumsum(increments, dim=1),
        ), dim=1)
        self.assertTrue(torch.allclose(output, reconstructed))

    def test_decision_requires_all_prediction_gates(self):
        config = load_cumulative_delta_config(CONFIG)
        metrics = {
            "validation_safety": {
                "false_safe_action_count": 0, "exact_safe_action_recall": 0.8,
            },
            "test_safety": {
                "false_safe_action_count": 0, "exact_safe_action_recall": 0.95,
                "near_boundary_RMSE_m": 0.001,
            },
            "joint_sensitivity": {"mean_cosine": 0.9},
            "margin_sensitivity": {"mean_cosine": 0.9},
        }
        support = {"all_eligible_states_supported": True}
        temporal = {
            "test": {
                "terminal_joint_RMSE_rad": 0.01,
                "joint_RMSE_by_substep_rad": [0.0] * 51,
            }
        }
        decision = cumulative_delta_decision(
            metrics=metrics, validation_support=support, test_support=support,
            temporal=temporal, config=config,
        )
        self.assertTrue(decision["mechanism_prediction_GO"])
        metrics["test_safety"]["false_safe_action_count"] = 1
        decision = cumulative_delta_decision(
            metrics=metrics, validation_support=support, test_support=support,
            temporal=temporal, config=config,
        )
        self.assertFalse(decision["mechanism_prediction_GO"])


if __name__ == "__main__":
    unittest.main()
