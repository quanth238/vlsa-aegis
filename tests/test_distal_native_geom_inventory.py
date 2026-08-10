import unittest
from pathlib import Path

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

from main.multilink_ellipsoid.native_geom_margin import load_config


ROOT = Path(__file__).resolve().parents[1]


class NativeGeomInventoryConfigTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(
            ROOT / "configs/vlsa_distal_native_geom_inventory_moka10.v1.json"
        )

    def test_physical_rows_are_not_forced_to_seven(self):
        target = self.config["target_definition"]
        self.assertEqual(
            target["grouping"],
            "one_row_per_actual_compiled_protected_collision_geom",
        )
        self.assertIn("aggregates_only", target["link_and_global_minima"])

    def test_proxy_and_native_measurements_remain_distinct(self):
        target = self.config["target_definition"]
        self.assertEqual(target["ellipsoid_margins"], "D_opt_comparison_only")
        self.assertEqual(target["raw_contacts"], "independent_D_sim_witness")

    def test_no_learning_or_control_is_authorized(self):
        gate = self.config["gate"]
        self.assertFalse(gate["training_in_this_gate"])
        self.assertFalse(gate["QP_in_this_gate"])
        self.assertFalse(gate["closed_loop_in_this_gate"])
        self.assertFalse(self.config["decision"]["closed_loop_E05_authorized"])


@unittest.skipIf(np is None, "numpy is unavailable")
class NativeGeomInventoryNumericTests(unittest.TestCase):
    def test_collision_eligibility_is_symmetric(self):
        from main.multilink_ellipsoid.native_geom_margin import _collision_eligible

        class Model:
            geom_contype = np.asarray([1, 2, 0], dtype=np.int64)
            geom_conaffinity = np.asarray([2, 1, 0], dtype=np.int64)

        self.assertTrue(_collision_eligible(Model(), 0, 1))
        self.assertTrue(_collision_eligible(Model(), 1, 0))
        self.assertFalse(_collision_eligible(Model(), 0, 2))


if __name__ == "__main__":
    unittest.main()
