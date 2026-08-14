import unittest
from pathlib import Path

import numpy as np

from main.multilink_ellipsoid.l5_row01_oracle_anchor_delta import (
    build_model, load_config, response_metrics,
)


class OracleAnchorDeltaTest(unittest.TestCase):
    def test_registered_config(self):
        config = load_config(
            Path(__file__).resolve().parents[1]
            / "configs/vlsa_distal_l5_row01_oracle_anchor_delta.v1.json"
        )
        self.assertEqual(config["representation"]["input_dimension"], 134)
        self.assertEqual(config["dataset"]["expected_state_counts"], {
            "train": 3, "validation": 3,
        })

    def test_model_shape(self):
        import torch
        model = build_model(torch, 134, [32, 32])
        self.assertEqual(tuple(model(torch.zeros(4, 134)).shape), (4, 2))

    def test_response_metrics_distinguish_ranking_and_acceptance(self):
        samples = [
            {
                "state_id": "s", "case_id": "c", "candidate_name": "nominal",
                "candidate_order": 0, "applied_correction_l2_action": 0.0,
                "anchor_target": [0.02, -0.01], "combined_target": [0.02, -0.01],
                "delta_target": [0.0, 0.0], "exact_safe": False,
            },
            {
                "state_id": "s", "case_id": "c", "candidate_name": "safe",
                "candidate_order": 1, "applied_correction_l2_action": 0.5,
                "anchor_target": [0.02, -0.01], "combined_target": [-0.01, -0.02],
                "delta_target": [-0.03, -0.01], "exact_safe": True,
            },
            {
                "state_id": "s", "case_id": "c", "candidate_name": "unsafe",
                "candidate_order": 2, "applied_correction_l2_action": 1.0,
                "anchor_target": [0.02, -0.01], "combined_target": [0.01, -0.02],
                "delta_target": [-0.01, -0.01], "exact_safe": False,
            },
        ]
        prediction = np.asarray([[0.0, 0.0], [-0.03, -0.01], [-0.01, -0.01]])
        report = response_metrics(
            prediction, samples, sign_deadband_m=1.0e-4,
            random_seed=1, random_draws=32,
        )
        self.assertEqual(report["false_safe_candidate_count"], 0)
        self.assertEqual(report["safe_unsafe_pair_order_accuracy"], 1.0)
        self.assertEqual(report["selected_exact_row01_safe_state_count"], 1)


if __name__ == "__main__":
    unittest.main()
