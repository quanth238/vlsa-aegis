import unittest
import importlib.util
from pathlib import Path

from main.multilink_ellipsoid.l5_row01_clearance_distribution_audit import (
    audit_distribution, classify, load_config,
)


def sample(state, clearance, risk, exact, order):
    return {
        "state_id": state,
        "case_id": state,
        "clearance_endpoint_feature_vector": [0.0] * 9 + list(clearance),
        "risk_row01": list(risk),
        "exact_safe": exact,
        "candidate_order": order,
        "candidate_name": f"c{order}",
        "applied_correction_l2_action": float(order),
    }


class ClearanceDistributionAuditTest(unittest.TestCase):
    def test_config(self):
        config = load_config(Path(
            "configs/vlsa_distal_l5_row01_clearance_distribution_audit.v1.json"
        ))
        self.assertTrue(config["audit"]["state_balanced"])

    @unittest.skipIf(importlib.util.find_spec("numpy") is None, "numpy optional locally")
    def test_state_balancing_and_shift(self):
        train = [
            sample("a", [0.01, 0.02, 0.03], [0.01, -0.01], False, 0),
            sample("a", [0.01, 0.02, 0.03], [-0.01, -0.01], True, 1),
            sample("b", [0.03, 0.04, 0.05], [-0.01, -0.01], True, 0),
        ]
        validation = [
            sample("v", [0.05, 0.06, 0.07], [-0.01, -0.01], True, 0),
        ]
        audit = audit_distribution(
            train_samples=train, validation_samples=validation,
            train_predictions_9d=[[0.01, -0.01], [-0.01, -0.01], [-0.01, -0.01]],
            validation_predictions_9d=[[-0.01, -0.01]],
            train_predictions_12d=[[0.01, -0.01], [-0.01, -0.01], [-0.01, -0.01]],
            validation_predictions_12d=[[0.01, 0.01]],
            standard_deviation_floor_m=1e-6, range_tolerance_m=1e-12,
            high_distance_threshold=2.0,
        )
        self.assertEqual(audit["train_distinct_state_count"], 2)
        self.assertEqual(audit["train_candidate_count"], 3)
        record = audit["validation_state_records"][0]
        self.assertEqual(record["outside_training_range_count"], 3)
        self.assertTrue(record["support_lost_by_12D"])
        self.assertIn("distribution_shift", classify(audit))

    @unittest.skipIf(importlib.util.find_spec("numpy") is None, "numpy optional locally")
    def test_rejects_candidate_varying_state_clearance(self):
        train = [
            sample("a", [0.01, 0.02, 0.03], [-0.01, -0.01], True, 0),
            sample("a", [0.02, 0.02, 0.03], [-0.01, -0.01], True, 1),
        ]
        with self.assertRaisesRegex(ValueError, "vary across candidates"):
            audit_distribution(
                train_samples=train,
                validation_samples=[sample(
                    "v", [0.01, 0.02, 0.03], [-0.01, -0.01], True, 0
                )],
                train_predictions_9d=[[-0.01, -0.01]] * 2,
                validation_predictions_9d=[[-0.01, -0.01]],
                train_predictions_12d=[[-0.01, -0.01]] * 2,
                validation_predictions_12d=[[-0.01, -0.01]],
                standard_deviation_floor_m=1e-6, range_tolerance_m=1e-12,
                high_distance_threshold=2.0,
            )


if __name__ == "__main__":
    unittest.main()
