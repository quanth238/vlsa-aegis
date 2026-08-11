from pathlib import Path
import unittest

from main.multilink_ellipsoid.factorized_recurrent_nominal_residual import (
    _build_recurrent_model, fitted_prediction_gate, load_recurrent_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/vlsa_distal_factorized_recurrent_nominal_residual_moka10.v1.json"


def sensitivity_item(cosine=0.9, ratio=1.0, relative=0.1):
    return {
        "mean_cosine": cosine, "median_norm_ratio": ratio,
        "mean_relative_norm_error": relative, "valid_count": 10,
        "cosine_p05": 0.8, "median_cosine": cosine,
    }


class RecurrentNominalResidualTests(unittest.TestCase):
    def test_config_freezes_conditional_untouched_gate(self):
        config = load_recurrent_config(CONFIG)
        self.assertTrue(
            config["forbidden_before_fitted_pass"]["untouched_episode_opening"]
        )
        self.assertTrue(config["forbidden_before_fitted_pass"]["closed_loop"])
        self.assertEqual(
            config["architecture"]["output"],
            "direct_delta_q_k_from_q0_without_increment_integration",
        )

    def test_model_has_exact_q0_and_second_action_causality(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")
        config = load_recurrent_config(CONFIG)
        model = _build_recurrent_model(12, config["architecture"])
        features = torch.zeros((3, 12), dtype=torch.float32)
        anchor = torch.zeros((3, 2, 7), dtype=torch.float32)
        first = torch.zeros((3, 2, 7), dtype=torch.float32)
        second = first.clone()
        second[:, 1] = 0.5
        output_first = model(features, anchor, first)
        output_second = model(features, anchor, second)
        self.assertEqual(tuple(output_first.shape), (3, 51, 7))
        self.assertTrue(torch.equal(
            output_first[:, 0], torch.zeros_like(output_first[:, 0])
        ))
        self.assertTrue(torch.equal(output_first[:, :26], output_second[:, :26]))
        output_second.square().mean().backward()
        self.assertTrue(any(
            parameter.grad is not None for parameter in model.parameters()
        ))

    def test_fitted_gate_requires_safety_and_joint_sensitivity(self):
        config = load_recurrent_config(CONFIG)
        by_horizon = [{
            "substep": k, "joint": sensitivity_item(),
            "safety": sensitivity_item(),
        } for k in range(51)]
        split = {
            "all_horizon_trace": {
                "joint": sensitivity_item(), "safety": sensitivity_item(),
                "by_horizon": by_horizon,
            }
        }
        temporal = {
            "overall_joint_RMSE_rad": 0.01,
            "terminal_joint_RMSE_rad": 0.015,
            "linear_RMSE_slope_rad_per_substep": 0.0001,
            "terminal_to_overall_RMSE_ratio": 1.5,
            "maximum_initial_joint_error_rad": 0.0,
        }
        metrics = {
            "sensitivity": {"train": split, "validation": split},
            "safety": {"validation": {
                "false_safe_action_count": 0,
                "exact_safe_action_recall": 0.95,
                "near_boundary_RMSE_m": 0.001,
            }},
            "temporal_joint": {"validation": temporal},
        }
        support = {"validation": {"all_eligible_states_supported": True}}
        passed = fitted_prediction_gate(
            metrics=metrics, support=support, config=config,
        )
        self.assertTrue(passed["fitted_prediction_gate_pass"])
        metrics["sensitivity"]["validation"]["all_horizon_trace"][
            "safety"
        ] = sensitivity_item(cosine=0.2)
        failed = fitted_prediction_gate(
            metrics=metrics, support=support, config=config,
        )
        self.assertFalse(failed["fitted_prediction_gate_pass"])

    def test_source_contains_no_joint_increment_integration(self):
        source = (
            ROOT / "main/multilink_ellipsoid/"
            "factorized_recurrent_nominal_residual.py"
        ).read_text()
        self.assertNotIn("cumsum", source)
        self.assertNotIn("previous_q", source)
        self.assertIn("candidate_hidden, zero_hidden", source)
        self.assertIn("nominal + residual", source)


if __name__ == "__main__":
    unittest.main()
