import unittest
from pathlib import Path

import numpy as np

from main.multilink_ellipsoid.tight_prefix_oracle_flow_gradient import (
    case_direction_metrics,
    central_gradient,
    choose_closest_unsafe_safe_pair,
    choose_local_crossing,
    comparison_residuals,
    finite_difference_residuals,
    interpolation_residuals,
    load_config,
    terminal_branch_envelope,
)


class TightPrefixOracleFlowGradientTests(unittest.TestCase):
    def test_registered_config(self):
        root = Path(__file__).resolve().parents[1]
        config = load_config(
            root / "configs/vlsa_tight_prefix_oracle_flow_gradient.v1.json"
        )
        self.assertEqual(len(config["cases"]), 3)
        self.assertEqual(config["finite_difference"]["dimension"], 15)

    def test_terminal_envelope_requires_real_branch_zero(self):
        residuals = np.zeros((2, 10, 3))
        envelope = terminal_branch_envelope(["zero", "probe"], residuals)
        self.assertEqual(envelope["branch_after_euler_step"], 8)
        self.assertEqual(np.asarray(envelope["candidate_output_residuals"]).shape,
                         (2, 10, 3))

    def test_pair_and_interpolation_localize_crossing(self):
        residuals = np.zeros((3, 10, 3))
        residuals[1, 0, 0] = 1.0
        residuals[2, 0, 0] = 1.5
        records = [
            {"name": "unsafe_far", "hard_primary_future_risk": 0.2,
             "physical_primary_safe": False},
            {"name": "unsafe_near", "hard_primary_future_risk": 0.1,
             "physical_primary_safe": False},
            {"name": "safe", "hard_primary_future_risk": -0.1,
             "physical_primary_safe": True},
        ]
        pair = choose_closest_unsafe_safe_pair(
            records, residuals, maximum_distance=2.0,
        )
        self.assertEqual(pair["unsafe_name"], "unsafe_near")
        names, values = interpolation_residuals(
            residuals[1], residuals[2], segment_count=8,
        )
        localized = []
        for index, name in enumerate(names):
            localized.append({
                "name": name,
                "hard_primary_future_risk": 0.1 if index < 5 else -0.1,
                "physical_primary_safe": index >= 5,
            })
        crossing = choose_local_crossing(
            localized, values, minimum_distance=0.01, maximum_distance=0.25,
        )
        self.assertEqual(crossing["unsafe_index"], 4)
        self.assertEqual(crossing["safe_index"], 5)

    def test_central_difference_recovers_linear_gradient(self):
        anchor = np.zeros((10, 3))
        names, residuals = finite_difference_residuals(anchor, epsilon=0.05)
        weights = np.arange(1.0, 16.0)
        records = []
        for name, residual in zip(names, residuals):
            value = float(np.asarray(residual)[:5].reshape(-1) @ weights)
            records.append({"name": name, "risk": value})
        gradient = central_gradient(records, key="risk", epsilon=0.05)
        np.testing.assert_allclose(gradient, weights, atol=1e-12)

    def test_comparison_is_equal_norm_and_metrics_keep_controls_separate(self):
        names, residuals, audit = comparison_residuals(
            np.zeros((10, 3)), np.ones(15), np.arange(1.0, 16.0),
            correction_norm=0.2, random_seed=20260819,
        )
        self.assertEqual(len(names), 8)
        self.assertLessEqual(audit["maximum_requested_norm_error"], 1e-12)
        risks = {
            "comparison_anchor": 0.1,
            "learned_down": 0.08,
            "oracle_down": -0.05,
            "oracle_up": 0.2,
            "random_0": 0.0,
            "random_1": 0.02,
            "random_2": -0.01,
            "random_3": -0.1,
        }
        records = [
            {"name": name, "hard_primary_future_risk": risks[name],
             "physical_primary_safe": risks[name] <= 0.0}
            for name in names
        ]
        metrics = case_direction_metrics(records)
        self.assertTrue(metrics["oracle_descends"])
        self.assertTrue(metrics["oracle_safe_conversion"])
        self.assertEqual(metrics["oracle_random_advantage_count"], 3)
        self.assertFalse(metrics["learned_safe_conversion"])


if __name__ == "__main__":
    unittest.main()
