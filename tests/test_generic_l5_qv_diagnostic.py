from __future__ import annotations

import json
from pathlib import Path
import unittest

from main.multilink_ellipsoid.generic_l5_qv_diagnostic import (
    action_endpoint_feature, diagnostic_metrics, direct_l5_state_feature,
    load_config, state_balanced_weights,
)


def _boundary() -> dict:
    rows = []
    for index in range(6):
        rows.append({
            "body_name": "robot0_link5" if index in (1, 2, 3) else "other",
            "center_m": [0.1 * index, 0.0, 1.0],
            "rotation": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "semiaxes_m": [0.01, 0.02, 0.03],
        })
    return {
        "exact_robot_rows": rows,
        "row_normalized_radial_slack": [1.0, 0.4, -0.2, 0.1, 0.8, 0.9],
        "compiled_obstacle_boxes": [{
            "center_m": [0.0, 0.0, 1.0],
            "rotation": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "half_extents_m": [0.05, 0.06, 0.07],
        }],
    }


class GenericL5QVDiagnosticTest(unittest.TestCase):
    def assertSequenceAlmostEqual(self, left, right):
        self.assertEqual(len(left), len(right))
        for observed, expected in zip(left, right):
            self.assertAlmostEqual(observed, expected)

    def test_direct_l5_feature_uses_active_row_and_is_finite(self) -> None:
        feature = direct_l5_state_feature(_boundary(), row_indices=[1, 2, 3])
        self.assertEqual(len(feature), 15)
        self.assertSequenceAlmostEqual(feature[:3], [0.2, 0.0, 0.0])
        self.assertSequenceAlmostEqual(feature[9:12], [0.4, -0.2, 0.1])
        self.assertEqual(feature[12:], [0.0, 1.0, 0.0])

    def test_action_endpoint_uses_scale_only(self) -> None:
        actions = [[1.0, -0.5, 0.25, 0.0, 0.0, 0.0, 0.0] for _ in range(5)]
        observed = action_endpoint_feature(
            actions, translation_scale_m_per_action_unit=0.05,
        )
        self.assertSequenceAlmostEqual(observed, [0.25, -0.125, 0.0625])

    def test_metrics_preserve_false_safes_and_state_support(self) -> None:
        samples = [
            {"state_id": "a", "target": -0.1},
            {"state_id": "a", "target": 0.2},
            {"state_id": "b", "target": -0.3},
        ]
        metrics = diagnostic_metrics(samples, [-0.2, -0.1, 0.1])
        self.assertEqual(metrics["false_safe_count"], 1)
        self.assertEqual(metrics["false_unsafe_count"], 1)
        self.assertEqual(metrics["safe_support_state_count"], 1)

    def test_state_balanced_weights_equalize_groups(self) -> None:
        weights = state_balanced_weights(["a", "a", "b"])
        self.assertAlmostEqual(sum(weights[:2]), weights[2])

    def test_config_is_preregistered_and_forbids_control(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = load_config(root / "configs/vlsa_distal_generic_l5_qv_diagnostic.v1.json")
        self.assertEqual(config["model"]["arms"], ["q_only", "q_plus_v"])
        self.assertIn("QP_or_gradient_correction", config["forbidden"])
        self.assertTrue(json.loads(
            (root / "configs/vlsa_distal_generic_l5_qv_diagnostic.v1.json").read_text()
        ))


if __name__ == "__main__":
    unittest.main()
