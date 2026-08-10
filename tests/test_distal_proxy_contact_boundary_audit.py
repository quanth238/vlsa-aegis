import unittest
from pathlib import Path

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

from main.multilink_ellipsoid.proxy_contact_boundary_audit import load_config


ROOT = Path(__file__).resolve().parents[1]


class ProxyContactBoundaryConfigTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(
            ROOT / "configs/vlsa_distal_proxy_contact_boundary_audit_moka10.v1.json"
        )

    def test_native_positive_distance_is_forbidden(self):
        self.assertFalse(self.config["comparison"]["native_positive_distance_used"])

    def test_control_and_learning_are_forbidden(self):
        gate = self.config["gate"]
        self.assertFalse(gate["training_in_this_gate"])
        self.assertFalse(gate["QP_in_this_gate"])
        self.assertFalse(gate["simulation_in_this_gate"])
        self.assertFalse(gate["closed_loop_E05_authorized"])


@unittest.skipIf(np is None, "numpy is unavailable")
class ProxyContactBoundaryNumericTests(unittest.TestCase):
    def test_confusion_counts_false_safe_and_false_unsafe(self):
        from main.multilink_ellipsoid.proxy_contact_boundary_audit import _confusion

        records = [
            {"D_opt_m": 0.001, "D_sim_contact": True},
            {"D_opt_m": -0.001, "D_sim_contact": False},
            {"D_opt_m": 0.002, "D_sim_contact": False},
            {"D_opt_m": -0.002, "D_sim_contact": True},
        ]
        summary = _confusion(records, 0.0)
        self.assertEqual(summary["false_safe_count"], 1)
        self.assertEqual(summary["false_unsafe_count"], 1)
        self.assertEqual(summary["true_safe_count"], 1)
        self.assertEqual(summary["true_unsafe_count"], 1)


if __name__ == "__main__":
    unittest.main()
