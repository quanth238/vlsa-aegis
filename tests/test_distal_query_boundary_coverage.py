from pathlib import Path
import unittest

from main.multilink_ellipsoid.query_boundary_coverage import (
    initially_safe, load_config, nominal_prefix_unsafe, row_coverage,
)


ROOT = Path(__file__).resolve().parents[1]


class QueryBoundaryCoverageTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(
            ROOT / "configs/vlsa_distal_query_boundary_coverage.v1.json"
        )

    def test_current_state_is_eligibility_only(self):
        self.assertTrue(initially_safe(
            [0.001] * 7, protected_contact_count=0,
            active_obstacle_l1_displacement_m=0.001, config=self.config,
        ))
        self.assertFalse(initially_safe(
            [0.0009] + [0.01] * 6, protected_contact_count=0,
            active_obstacle_l1_displacement_m=0.0, config=self.config,
        ))

    def test_nominal_prefix_union_of_proxy_and_physical_failure(self):
        self.assertTrue(nominal_prefix_unsafe(
            [0.0009] + [0.01] * 6, protected_contact_count=0,
            maximum_active_obstacle_l1_displacement_m=0.0, config=self.config,
        ))
        self.assertTrue(nominal_prefix_unsafe(
            [0.01] * 7, protected_contact_count=1,
            maximum_active_obstacle_l1_displacement_m=0.0, config=self.config,
        ))
        self.assertFalse(nominal_prefix_unsafe(
            [0.01] * 7, protected_contact_count=0,
            maximum_active_obstacle_l1_displacement_m=0.001, config=self.config,
        ))

    def test_row_coverage_does_not_fill_unsupported_rows(self):
        record = {
            "current": {"row_clearance_m": [0.002] + [0.02] * 6},
            "nominal_prefix": {
                "row_minimum_clearance_m": [-0.001] + [0.02] * 6,
                "active_witness": {"row": 0},
            },
        }
        rows = row_coverage([record], self.config)
        self.assertEqual(rows[0]["nominal_active_witness_count"], 1)
        self.assertEqual(rows[0]["nominal_violated_row_count"], 1)
        self.assertEqual(rows[1]["nominal_active_witness_count"], 0)
        self.assertEqual(rows[1]["nominal_violated_row_count"], 0)


if __name__ == "__main__":
    unittest.main()
