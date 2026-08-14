import unittest
from pathlib import Path

import numpy as np

from main.multilink_ellipsoid.l5_row01_component_audit import (
    component_metrics, false_safe_phase_attribution, load_config,
)


class ComponentAuditTest(unittest.TestCase):
    def test_registered_config(self):
        config = load_config(
            Path(__file__).resolve().parents[1]
            / "configs/vlsa_distal_l5_row01_component_audit.v1.json"
        )
        self.assertEqual(config["targets"]["learned_rows"], [0, 1])
        self.assertEqual(config["matched_model"]["input_dimension"], 134)

    def test_component_false_safe_and_phase_are_explicit(self):
        prediction = np.asarray([[-0.1, -0.1], [-0.1, -0.1]])
        target = np.asarray([[0.2, -0.1], [-0.1, 0.3]])
        metrics = component_metrics(prediction, target, 0.005)
        self.assertEqual(metrics["false_safe_count"], 2)
        samples = [
            {"combined_target": target[0].tolist(), "witness_phase": ["prefix", "backup"],
             "state_id": "a", "candidate_name": "x"},
            {"combined_target": target[1].tolist(), "witness_phase": ["prefix", "backup"],
             "state_id": "b", "candidate_name": "y"},
        ]
        audit = false_safe_phase_attribution(prediction, samples)
        self.assertEqual(audit["count_by_active_phase"], {"prefix": 1, "backup": 1})


if __name__ == "__main__":
    unittest.main()
