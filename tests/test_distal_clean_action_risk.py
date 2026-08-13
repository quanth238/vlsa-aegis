import json
from pathlib import Path
import tempfile
import unittest

try:
    import torch
except ImportError:
    torch = None

from main.multilink_ellipsoid.clean_action_risk import (
    compact_feature_vector,
    decision_steps,
    exact_safe,
    load_cases,
    load_config,
    prediction_metrics,
    risk_from_row_minimum,
)


ROOT = Path(__file__).resolve().parents[1]


class CleanActionRiskTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(ROOT / "configs/vlsa_distal_clean_action_risk.v1.json")
        self.cases = load_cases(
            ROOT / "manifests/vlsa_distal_clean_action_risk.v1.jsonl", self.config
        )

    def test_grouped_split_and_untouched_test(self):
        by_split = {}
        for row in self.cases:
            by_split.setdefault(row["split"], []).append(row["case_id"])
        self.assertEqual(len(by_split["train"]), 10)
        self.assertEqual(len(by_split["validation"]), 3)
        self.assertEqual(len(by_split["test"]), 3)
        self.assertEqual(len(by_split["diagnostic"]), 2)
        self.assertEqual(by_split["test"], self.config["cohort"]["untouched_test_cases"])

    def test_decision_steps_precede_contact(self):
        for case in self.cases:
            steps = decision_steps(case, self.config)
            self.assertEqual(len(steps), 4)
            self.assertTrue(all(step >= 0 for step in steps))
            self.assertTrue(all(step < case["first_relevant_contact_step"] for step in steps))

    def test_risk_and_physical_acceptance_are_distinct(self):
        risk = risk_from_row_minimum([0.002] * 7, 0.001)
        record = {
            "risk": risk,
            "protected_contact_count": 0,
            "maximum_active_obstacle_l1_displacement_m": 0.0005,
        }
        self.assertTrue(exact_safe(record, self.config))
        record["protected_contact_count"] = 1
        self.assertFalse(exact_safe(record, self.config))

    def test_prediction_metrics_count_false_safe_and_support(self):
        samples = [
            {"state_id": "a", "risk": [-0.001] * 7, "exact_safe": True},
            {"state_id": "a", "risk": [0.001] + [-0.001] * 6, "exact_safe": False},
            {"state_id": "b", "risk": [-0.001] * 7, "exact_safe": True},
        ]
        prediction = [[-0.001] * 7, [-0.001] * 7, [0.001] * 7]
        metrics = prediction_metrics(prediction, samples)
        self.assertEqual(metrics["false_safe_count"], 1)
        self.assertEqual(metrics["recoverable_state_count"], 2)
        self.assertEqual(metrics["supported_recoverable_state_count"], 1)

    def test_compact_feature_has_fixed_dimension(self):
        context = {
            "arm_joint_position_rad": [0.0] * 7,
            "arm_joint_velocity_rad_s": [0.0] * 7,
            "eef_position_m": [0.0] * 3,
            "eef_quaternion_xyzw": [0.0] * 4,
            "controller_snapshot": {"goal_pos": [0.0] * 3, "goal_ori": [0.0] * 9},
            "obstacle": {"center_m": [0.0] * 3, "rotation": [[0.0] * 3] * 3, "semiaxes_m": [0.0] * 3},
            "geometry_rows": [
                {"center_m": [0.0] * 3, "rotation": [[0.0] * 3] * 3,
                 "semiaxes_m": [0.0] * 3, "current_clearance_m": 0.0}
                for _ in range(7)
            ],
        }
        self.assertEqual(len(compact_feature_vector(context, [0.0] * 7)), 167)

    @unittest.skipIf(torch is None, "torch is optional locally")
    def test_action_risk_model_has_registered_shape(self):
        from main.multilink_ellipsoid.action_risk_model import build_model

        model = build_model(torch, 167, [256, 256, 128])
        output = model(torch.zeros((3, 167), dtype=torch.float32))
        self.assertEqual(tuple(output.shape), (3, 7))


if __name__ == "__main__":
    unittest.main()
