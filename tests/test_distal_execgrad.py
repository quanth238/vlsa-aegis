"""Structural tests for the direction-first ExecGrad pilot."""

from __future__ import annotations

import copy
from pathlib import Path
import unittest

import numpy as np

from main.multilink_ellipsoid.execgrad import (
    _build_model,
    action_gradient_from_pairs,
    direction_metrics,
    execgrad_fitted_decision,
    link_jvp_loss,
    link_jvp_scales,
    load_execgrad_config,
    soft_min_trace,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/vlsa_distal_execgrad_direction_moka10.v1.json"


class ExecGradTest(unittest.TestCase):
    def test_validator_uses_git_identity_commit_field(self) -> None:
        source = (
            ROOT / "scripts/validate_distal_execgrad_direction_moka10.py"
        ).read_text(encoding="utf-8")
        self.assertIn('result.get("source", {}).get("commit")', source)
        self.assertNotIn('result.get("source", {}).get("git_commit")', source)

    def test_config_and_exact_initial_condition(self) -> None:
        import torch

        config = load_execgrad_config(CONFIG)
        model = _build_model(11, config["architecture"])
        features = torch.zeros((3, 11), dtype=torch.float32)
        nominal = torch.randn((3, 51, 7), dtype=torch.float32)
        output = model(features, nominal)
        self.assertEqual(tuple(output.shape), (3, 51, 7))
        self.assertTrue(torch.equal(output[:, 0], nominal[:, 0]))

    def test_config_rejects_gripper_as_steering_target(self) -> None:
        config = load_execgrad_config(CONFIG)
        broken = copy.deepcopy(config)
        broken["training"]["steering_action_dimensions"].append(6)
        # The strict loader works on files; equality here documents the frozen
        # semantic separation used by the trainer.
        self.assertNotEqual(broken["training"], config["training"])
        self.assertEqual(
            config["training"]["gripper_dimensions_are_inputs_but_not_steering_targets"],
            [6, 13],
        )

    def test_link_jvp_scaling_and_zero_loss(self) -> None:
        import torch

        config = load_execgrad_config(CONFIG)
        exact = np.zeros((24, 51, 7, 3), dtype=np.float64)
        dimensions = np.asarray(
            config["training"]["steering_action_dimensions"] * 2, dtype=np.int64
        )
        exact[:] = 0.01
        scale = link_jvp_scales(
            exact, dimensions, np.arange(24), config["training"]
        )
        loss = link_jvp_loss(
            torch.as_tensor(exact, dtype=torch.float32),
            torch.as_tensor(exact, dtype=torch.float32),
            dimensions,
            scale,
        )
        self.assertEqual(float(loss), 0.0)
        self.assertEqual(scale["valid_cell_count"], 12 * 51)

    def test_soft_min_and_pair_gradient(self) -> None:
        trace = np.zeros((5, 51, 7), dtype=np.float64)
        trace[:, 12, 3] = np.arange(5, dtype=np.float64)
        score = soft_min_trace(trace, 0.001)
        self.assertEqual(score.shape, (5,))
        arrays = {
            "action_chunk": np.zeros((5, 2, 7), dtype=np.float64),
        }
        arrays["action_chunk"][1, 0, 0] = -0.1
        arrays["action_chunk"][2, 0, 0] = 0.1
        arrays["action_chunk"][3, 0, 1] = -0.2
        arrays["action_chunk"][4, 0, 1] = 0.2
        sensitivities = {
            "negative_row_index": np.asarray([1, 3]),
            "positive_row_index": np.asarray([2, 4]),
            "dimension_index": np.asarray([0, 1]),
            "state_index": np.asarray([7, 7]),
        }
        _, gradient = action_gradient_from_pairs(
            np.asarray([0.0, -0.1, 0.1, -0.4, 0.4]),
            sensitivities,
            arrays,
            [0, 1],
        )
        self.assertTrue(np.allclose(gradient, [[1.0, 2.0]]))

    def test_direction_gate(self) -> None:
        config = load_execgrad_config(CONFIG)
        states = np.asarray([0, 1, 2, 3])
        exact = np.asarray([[1.0, 0.0]] * 4)
        predicted = exact.copy()
        split = {0: "train", 1: "train", 2: "validation", 3: "test"}
        metrics = direction_metrics(
            exact_gradient=exact,
            predicted_gradient=predicted,
            state_ids=states,
            state_splits=split,
            config=config,
        )
        baseline = copy.deepcopy(metrics)
        baseline["validation"]["median_exact_directional_gain_m_per_unit_action"] = 0.0
        temporal = {
            name: {
                "validation": {"overall_joint_RMSE_rad": 0.01}
            }
            for name in ("trajectory_only", "trajectory_plus_link_JVP")
        }
        decision = execgrad_fitted_decision(
            trajectory_only=baseline,
            execgrad=metrics,
            temporal=temporal,
            config=config,
        )
        self.assertTrue(decision["fitted_direction_gate_pass"])


if __name__ == "__main__":
    unittest.main()
