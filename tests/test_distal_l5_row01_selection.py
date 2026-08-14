import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from main.multilink_ellipsoid.l5_row01_selection import (
    load_config, payload_sha256, selection_metrics,
)


class L5Row01SelectionTest(unittest.TestCase):
    def test_registered_config(self):
        config = load_config(Path(
            "configs/vlsa_distal_l5_row01_selection.v1.json"
        ))
        self.assertEqual(config["selection"]["learned_rows"], [0, 1])
        self.assertEqual(config["model"]["output_count"], 2)
        self.assertIn("QP", config["forbidden"])

    def test_selection_prefers_closest_predicted_safe_candidate(self):
        samples = [
            {
                "state_id": "s0", "case_id": "c0", "candidate_name": "unsafe",
                "candidate_order": 0, "risk_row01": [0.001, -0.01],
                "risk_all_rows": [0.001, -0.01, -0.02, -0.03, -0.04, -0.05, -0.06],
                "exact_safe": False, "aegis_compatible": True,
                "applied_correction_l2_action": 0.0,
            },
            {
                "state_id": "s0", "case_id": "c0", "candidate_name": "safe_near",
                "candidate_order": 1, "risk_row01": [-0.001, -0.01],
                "risk_all_rows": [-0.001, -0.01, -0.02, -0.03, -0.04, -0.05, -0.06],
                "exact_safe": True, "aegis_compatible": True,
                "applied_correction_l2_action": 0.2,
            },
            {
                "state_id": "s0", "case_id": "c0", "candidate_name": "safe_far",
                "candidate_order": 2, "risk_row01": [-0.002, -0.01],
                "risk_all_rows": [-0.002, -0.01, -0.02, -0.03, -0.04, -0.05, -0.06],
                "exact_safe": True, "aegis_compatible": True,
                "applied_correction_l2_action": 0.4,
            },
        ]
        prediction = np.asarray([[0.001, -0.01], [-0.001, -0.01], [-0.002, -0.01]])
        report = selection_metrics(
            prediction, samples, random_seed=7, random_draws=100,
        )
        self.assertEqual(report["row01_false_safe_count"], 0)
        self.assertEqual(
            report["state_records"][0]["selected_candidate_name"], "safe_near"
        )
        self.assertTrue(report["all_selected_exact_all_seven_safe"])

    def test_false_safe_is_never_hidden_by_other_row(self):
        sample = [{
            "state_id": "s0", "case_id": "c0", "candidate_name": "x",
            "candidate_order": 0, "risk_row01": [0.001, -0.01],
            "risk_all_rows": [0.001, -0.01, -0.02, -0.03, -0.04, -0.05, -0.06],
            "exact_safe": False, "aegis_compatible": True,
            "applied_correction_l2_action": 0.0,
        }]
        report = selection_metrics(
            [[-0.001, -0.01]], sample, random_seed=7, random_draws=10,
        )
        self.assertEqual(report["row01_false_safe_count"], 1)
        self.assertFalse(report["all_selected_exact_all_seven_safe"])

    def test_payload_hash_ignores_only_self_hash(self):
        value = {"a": 1, "result_payload_sha256": "ignored"}
        self.assertEqual(payload_sha256(value), payload_sha256({"a": 1}))


if __name__ == "__main__":
    unittest.main()
