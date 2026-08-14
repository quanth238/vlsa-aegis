import json
import tempfile
import unittest
from pathlib import Path

from main.multilink_ellipsoid.normal_risk_curve import (
    candidate_definitions, curve_summary, load_config,
)


ROOT = Path(__file__).resolve().parents[1]


class NormalRiskCurveTest(unittest.TestCase):
    def test_config_and_candidates(self):
        try:
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("NumPy is unavailable in the local contract environment")
        config = load_config(ROOT / "configs/vlsa_distal_normal_risk_curve.v1.json")
        nominal = [[0.0] * 7 for _ in range(5)]
        frame = {"normal": [1.0, 0.0, 0.0]}
        rows = candidate_definitions(nominal, frame, config)
        self.assertEqual(len(rows), 9)
        self.assertEqual(rows[0]["requested_alpha"], 0.0)
        self.assertEqual(rows[-1]["requested_alpha"], 2.0)
        self.assertAlmostEqual(rows[4]["applied_correction_l2_action"], 1.0)

    def test_curve_summary_censors_timeout(self):
        base = {
            "combined_risk": [0.0] * 7,
            "effective_post_AEGIS_correction_l2_action": 0.0,
            "name": "candidate",
        }
        rows = [
            {**base, "requested_alpha": 0.0, "combined_risk": [0.002] * 7,
             "terminal_status": "UNSAFE_CONTACT_OR_CAR", "exact_safe": False},
            {**base, "requested_alpha": 0.5,
             "terminal_status": "UNKNOWN_TIMEOUT", "exact_safe": False},
            {**base, "requested_alpha": 1.0, "combined_risk": [-0.001] * 7,
             "effective_post_AEGIS_correction_l2_action": 0.9,
             "terminal_status": "SAFE_TERMINAL", "exact_safe": True},
        ]
        summary = curve_summary(rows, tolerance_m=1e-9)
        self.assertEqual(summary["unknown_timeout_count"], 1)
        self.assertEqual(summary["minimum_exact_safe_requested_alpha"], 1.0)
        self.assertEqual(summary["global_worst_risk_monotonicity_violation_count"], 0)


if __name__ == "__main__":
    unittest.main()
