from pathlib import Path
import unittest

from main.multilink_ellipsoid.factorized_explicit_jacobian_audit import (
    audit_decision, load_explicit_jacobian_audit_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/vlsa_distal_factorized_explicit_jacobian_audit_moka10.v1.json"


def _metrics(cosine, ratio=1.0, relative=0.1):
    return {
        "mean_cosine": cosine, "median_norm_ratio": ratio,
        "mean_relative_norm_error": relative,
    }


def _member(train, validation):
    return {
        "train": {"all": {"aggregate": train, "terminal": train}},
        "validation": {
            "all": {"aggregate": validation, "terminal": validation},
        },
    }


class ExplicitJacobianAuditTests(unittest.TestCase):
    def test_config_forbids_training_and_control(self):
        config = load_explicit_jacobian_audit_config(CONFIG)
        self.assertEqual(config["audit"]["ensemble_member_count"], 5)
        self.assertTrue(config["forbidden_actions"]["training"])
        self.assertTrue(config["forbidden_actions"]["simulation"])
        self.assertTrue(config["forbidden_actions"]["QP"])

    def test_decision_classifies_member_level_failure(self):
        config = load_explicit_jacobian_audit_config(CONFIG)
        failed = _member(_metrics(0.2), _metrics(0.1))
        passed = _member(_metrics(0.9), _metrics(0.9))
        sensitivity = {"members": [failed, failed, failed, failed, passed]}
        cancellation = {
            "validation": {
                "median_ensemble_norm_to_mean_member_norm_ratio": 0.9,
            }
        }
        training = [
            {"seed": index + 1, "best_epoch": index + 10}
            for index in range(5)
        ]
        decision = audit_decision(
            sensitivity=sensitivity, cancellation=cancellation,
            training_audits=training, config=config,
        )
        self.assertEqual(
            decision["classification"],
            "member_level_optimization_or_checkpoint_failure",
        )
        self.assertEqual(decision["member_train_and_validation_pass_count"], 1)
        self.assertFalse(decision["retraining_authorized"])

    def test_decision_separates_ensemble_cancellation(self):
        config = load_explicit_jacobian_audit_config(CONFIG)
        passed = _member(_metrics(0.9), _metrics(0.9))
        failed = _member(_metrics(0.2), _metrics(0.1))
        sensitivity = {"members": [passed, passed, passed, passed, failed]}
        cancellation = {
            "validation": {
                "median_ensemble_norm_to_mean_member_norm_ratio": 0.2,
            }
        }
        training = [
            {"seed": index + 1, "best_epoch": index + 10}
            for index in range(5)
        ]
        decision = audit_decision(
            sensitivity=sensitivity, cancellation=cancellation,
            training_audits=training, config=config,
        )
        self.assertEqual(decision["classification"], "ensemble_cancellation")
        self.assertFalse(decision["calibration_QP_closed_loop_authorized"])

    def test_audit_source_contains_no_optimizer_or_training_call(self):
        source = (
            ROOT / "scripts/audit_distal_explicit_execution_jacobian_members_moka10.py"
        ).read_text()
        self.assertNotIn("train_explicit_jacobian_ensemble", source)
        self.assertNotIn("torch.optim", source)
        self.assertIn("load_explicit_jacobian_weights", source)


if __name__ == "__main__":
    unittest.main()
