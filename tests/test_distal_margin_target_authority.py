import unittest
from pathlib import Path

try:
    import numpy as np
except ImportError:  # pragma: no cover - local optional dependency
    np = None

from main.multilink_ellipsoid.margin_target_authority import load_config


ROOT = Path(__file__).resolve().parents[1]


class MarginTargetAuthorityConfigTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(
            ROOT / "configs/vlsa_distal_margin_target_authority_moka10.v1.json"
        )

    def test_target_gate_is_raw_distal_contact_only(self):
        target = self.config["target_definition"]
        self.assertEqual(target["horizon_actions"], 2)
        self.assertEqual(
            target["temporal_resolution"], "every_internal_MuJoCo_substep"
        )
        self.assertIn("released_AEGIS_EE_proxy", target["excluded_from_gate"])
        self.assertEqual(
            self.config["gate"][
                "maximum_ellipsoid_safe_raw_distal_unsafe_count"
            ],
            0,
        )

    def test_gate_trains_and_simulates_nothing(self):
        gate = self.config["gate"]
        self.assertFalse(gate["training_in_this_gate"])
        self.assertFalse(gate["QP_in_this_gate"])
        self.assertFalse(gate["simulation_in_this_gate"])
        self.assertFalse(gate["closed_loop_E05_authorized"])


@unittest.skipIf(np is None, "numpy is unavailable")
class MarginTargetAuthorityNumericTests(unittest.TestCase):
    def test_nonnegative_ellipsoid_with_raw_contact_is_dangerous(self):
        from main.multilink_ellipsoid.margin_target_authority import _classify

        ellipsoid, raw, dangerous = _classify([0.001] * 7, False, 1)
        self.assertTrue(ellipsoid)
        self.assertFalse(raw)
        self.assertTrue(dangerous)

    def test_negative_ellipsoid_is_not_a_false_safe(self):
        from main.multilink_ellipsoid.margin_target_authority import _classify

        ellipsoid, raw, dangerous = _classify(
            [-0.001] + [0.001] * 6, False, 1
        )
        self.assertFalse(ellipsoid)
        self.assertFalse(raw)
        self.assertFalse(dangerous)


if __name__ == "__main__":
    unittest.main()
