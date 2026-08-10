import json
from pathlib import Path
import unittest

import numpy as np

from main.multilink_ellipsoid.factorized_one_sided_geometry import (
    fit_local_geometry_jacobians,
    load_one_sided_config,
    one_sided_decision,
    signed_clearance_error,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/vlsa_distal_factorized_one_sided_geometry_moka10.v1.json"


class OneSidedGeometryTests(unittest.TestCase):
    def test_config_is_frozen_and_blocks_downstream_control(self):
        config = load_one_sided_config(CONFIG)
        self.assertEqual(config["population"]["state_count"], 85)
        self.assertFalse(config["local_geometry_jacobian"]["test_split_used_for_fit"])
        self.assertTrue(config["forbidden_actions"]["QP"])
        self.assertTrue(config["forbidden_actions"]["closed_loop"])

    def test_local_geometry_jacobian_recovers_linear_clearance(self):
        config = json.loads(CONFIG.read_text())
        config["population"]["state_count"] = 3
        config["population"]["state_split_counts"] = {
            "train": 1, "validation": 1, "test": 1,
        }
        row_count = 3 * 93
        state_index = np.repeat(np.arange(3), 93)
        candidate_index = np.tile(np.arange(93), 3)
        split = np.repeat(np.asarray(["train", "validation", "test"], dtype=object), 93)
        source_code = np.tile(np.asarray([0] + [1] * 28 + [2] * 64), 3)
        q = np.zeros((row_count, 51, 7), dtype=np.float64)
        for row in range(row_count):
            candidate = candidate_index[row]
            state = state_index[row]
            vector = np.asarray([
                0.001 * candidate,
                0.0001 * candidate ** 2,
                0.002 * ((candidate % 3) - 1),
                0.0015 * ((candidate % 5) - 2),
                0.0007 * ((candidate % 7) - 3),
                0.0011 * ((candidate % 11) - 5),
                0.0009 * ((candidate % 13) - 6),
            ])
            q[row] = 0.1 * state + vector[None, :]
        coefficient = np.arange(49, dtype=np.float64).reshape(7, 7) * 1.0e-3
        clearance = np.einsum("rc,bkc->bkr", coefficient, q) + 0.01
        arrays = {
            "joint_position_rad": q,
            "ellipsoid_clearance_m": clearance,
            "state_index": state_index,
            "candidate_index": candidate_index,
            "source_code": source_code,
            "split": split,
        }
        fitted = fit_local_geometry_jacobians(arrays, config)
        self.assertTrue(fitted["fit_state_mask"][0])
        self.assertTrue(fitted["fit_state_mask"][1])
        self.assertFalse(fitted["fit_state_mask"][2])
        self.assertLess(fitted["audit"]["validation"]["linearization_RMSE_m"], 1.0e-4)

    def test_signed_error_is_normal_projected_joint_error(self):
        exact = np.zeros((2, 51, 7), dtype=np.float64)
        predicted = exact.copy()
        predicted[:, :, 0] = 0.002
        jacobian = np.zeros((2, 51, 7, 7), dtype=np.float64)
        jacobian[:, :, :, 0] = 0.5
        error = signed_clearance_error(
            predicted, exact, np.asarray([0, 1]), jacobian,
        )
        np.testing.assert_allclose(error, 0.001)

    def test_decision_requires_zero_false_safe_and_all_support(self):
        config = load_one_sided_config(CONFIG)
        common = {
            "validation_audit": {
                "false_safe_action_count": 2,
                "terminal_L5_false_safe_fraction": 1.0,
            },
            "jacobian_audit": {
                "validation": {"linearization_RMSE_m": 0.001},
            },
            "baseline_test": {"false_safe_action_count": 44},
            "joint_sensitivity_cosine": 0.9,
            "margin_sensitivity_cosine": 0.9,
            "config": config,
        }
        passing = one_sided_decision(
            experimental_test={
                "false_safe_action_count": 0,
                "state_safe_support_count": 15,
                "exact_safe_action_recall": 0.8,
                "near_boundary_RMSE_m": 0.002,
            },
            **common,
        )
        self.assertTrue(passing["one_sided_geometry_GO"])
        failing = one_sided_decision(
            experimental_test={
                "false_safe_action_count": 1,
                "state_safe_support_count": 15,
                "exact_safe_action_recall": 0.8,
                "near_boundary_RMSE_m": 0.002,
            },
            **common,
        )
        self.assertFalse(failing["one_sided_geometry_GO"])


if __name__ == "__main__":
    unittest.main()
