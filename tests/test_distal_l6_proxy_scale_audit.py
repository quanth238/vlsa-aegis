import unittest
from pathlib import Path

from main.multilink_ellipsoid.l6_proxy_scale_audit import (
    audit_source,
    load_config,
    rescale_row_slacks,
)


ROOT = Path(__file__).resolve().parents[1]


class L6ProxyScaleAuditTest(unittest.TestCase):
    def test_config(self):
        config = load_config(ROOT / "configs/vlsa_distal_l6_proxy_scale_audit.v1.json")
        self.assertEqual(config["calibration"]["scaled_rows"], [3, 4])

    def test_uniform_scale_transform(self):
        values = rescale_row_slacks(
            [0.1, 0.2, 0.3, -0.015667, 0.5, 0.6, 0.7],
            scale=0.98,
            scaled_rows=[3, 4],
        )
        self.assertAlmostEqual(values[3], (1.0 - 0.015667) / 0.98 - 1.0)
        self.assertEqual(values[2], 0.3)

    def test_smallest_shrink_selects_point_nine_eight(self):
        def candidate(name, status, contact, row3):
            rows = [0.5] * 7
            rows[3] = row3
            return {
                "name": name,
                "source_terminal_status": status,
                "source_physical_veto": bool(contact),
                "source_raw_protected_contact_count": int(contact),
                "compiled_box_row_minimum_normalized_radial_slack": rows,
            }

        source = {
            "strict_gate_pass": False,
            "cases": [
                {
                    "case_id": "a",
                    "candidates": [
                        candidate("contact-a", "UNSAFE_CONTACT_OR_CAR", 1, -0.1),
                        candidate("safe-a", "SAFE_TERMINAL", 0, 0.02),
                    ],
                },
                {
                    "case_id": "b",
                    "candidates": [
                        candidate("contact-b", "UNSAFE_CONTACT_OR_CAR", 1, -0.08),
                        candidate("safe-b", "SAFE_TERMINAL", 0, -0.015667),
                        candidate("safe-b2", "SAFE_TERMINAL", 0, 0.03),
                        candidate("safe-b3", "SAFE_TERMINAL", 0, 0.04),
                        candidate("contact-b2", "UNSAFE_CONTACT_OR_CAR", 1, -0.09),
                        candidate("timeout", "UNKNOWN_TIMEOUT", 0, -0.05),
                    ],
                },
            ],
        }
        config = load_config(ROOT / "configs/vlsa_distal_l6_proxy_scale_audit.v1.json")
        result = audit_source(source, config)
        self.assertEqual(result["selected_scale"], 0.98)
        self.assertTrue(result["strict_gate_pass"])


if __name__ == "__main__":
    unittest.main()
