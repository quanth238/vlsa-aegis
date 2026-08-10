import json
from pathlib import Path
import unittest

import numpy as np

from main.multilink_ellipsoid.factorized_terminal_bias_audit import (
    audit_decision, ensemble_disagreement_audit, geometry_signal_decision,
    load_terminal_bias_config, residual_audit,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/vlsa_distal_factorized_terminal_bias_audit_moka10.v1.json"


class TerminalBiasAuditTests(unittest.TestCase):
    def test_config_freezes_audit_and_blocks_training(self):
        config = load_terminal_bias_config(CONFIG)
        self.assertEqual(config["population"]["expected_action_count"], 5440)
        self.assertEqual(
            config["final_evaluation_policy"]["E05_E10_E15_role"],
            "diagnostic_failure_cases_only_after_repeated_model_selection",
        )
        self.assertTrue(config["forbidden_actions"]["training"])
        self.assertTrue(config["forbidden_actions"]["QP"])

    def test_split_residual_localizes_test_only_L5_bias(self):
        config = json.loads(CONFIG.read_text())
        count = 120
        exact = np.full((count, 51, 7), 0.003, dtype=np.float64)
        predicted = exact.copy()
        splits = np.repeat(np.asarray(["train", "validation", "test"], dtype=object), 40)
        predicted[80:, 50, :3] += 0.001
        audit = residual_audit(
            predicted_h=predicted, exact_h=exact, splits=splits, config=config,
        )
        self.assertEqual(
            audit["terminal_L5_cause"], "unseen_state_generalization_bias"
        )
        self.assertFalse(audit["material_terminal_bias"]["validation"]["L5"])
        self.assertTrue(audit["material_terminal_bias"]["test"]["L5"])

    def test_geometry_gate_is_strictly_submillimetre(self):
        config = load_terminal_bias_config(CONFIG)
        metrics = {
            "near_boundary_linearization_RMSE_m": 0.0004,
            "near_boundary_prediction_error_linearization_RMSE_m": 0.0003,
            "signed_delta_cosine": 0.95,
            "signed_delta_sign_agreement": 0.96,
            "resampled_action_gradient_median_cosine": 0.97,
            "resampled_action_gradient_p05_cosine": 0.85,
        }
        passing = geometry_signal_decision(metrics, config)
        self.assertTrue(passing["geometry_signal_valid"])
        metrics["near_boundary_linearization_RMSE_m"] = 0.0006
        self.assertFalse(geometry_signal_decision(metrics, config)["geometry_signal_valid"])

    def test_ensemble_disagreement_requires_ratio_and_auc(self):
        config = load_terminal_bias_config(CONFIG)
        member = np.asarray([
            [-0.004, -0.002, 0.0, 0.002, 0.004],
            [-0.003, -0.001, 0.001, 0.003, 0.005],
            [0.0010, 0.0011, 0.0009, 0.0010, 0.0010],
            [0.0020, 0.0021, 0.0019, 0.0020, 0.0020],
        ])
        audit = ensemble_disagreement_audit(
            member_minimum_m=member,
            class_code=np.asarray([1, 1, 2, 2], dtype=np.int8), config=config,
        )
        self.assertTrue(audit["uncertainty_detectable"])

    def test_ablation_requires_training_split_bias_and_valid_signal(self):
        decision = audit_decision(
            residual={
                "terminal_L5_cause": "unseen_state_generalization_bias",
                "terminal_distal_cause": "unseen_state_generalization_terminal_bias",
            },
            geometry={"geometry_signal_valid": True},
            ensemble={"uncertainty_detectable": False},
        )
        self.assertFalse(decision["matched_one_sided_ablation_authorized"])
        self.assertFalse(decision["training_submitted"])


if __name__ == "__main__":
    unittest.main()
