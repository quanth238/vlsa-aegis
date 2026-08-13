import unittest

from main.multilink_ellipsoid.pncbf_backup import select_repulsive_candidate, update_latch


class PncbfBackupTest(unittest.TestCase):
    def test_hysteresis(self):
        self.assertTrue(update_latch(False, 0.0005, 0.001, 0.005))
        self.assertTrue(update_latch(True, 0.003, 0.001, 0.005))
        self.assertFalse(update_latch(True, 0.006, 0.001, 0.005))

    def test_selects_smallest_buffered_then_best_safe_improvement(self):
        def item(norm, margin, contacts=0, car=0.0):
            return {"correction_l2_action": norm, "record": {"minimum_clearance_m": margin, "protected_contacts": [{}] * contacts, "maximum_active_obstacle_l1_displacement_m": car}}
        values = [item(0.25, -0.002), item(0.5, 0.0015), item(1.0, 0.004)]
        self.assertEqual(select_repulsive_candidate(values, nominal_clearance_m=-0.01, activation_clearance_m=0.001, paper_car_threshold_m=0.001)["correction_l2_action"], 0.5)
        values = [item(0.25, -0.008), item(0.5, -0.004), item(1.0, -0.006)]
        self.assertEqual(select_repulsive_candidate(values, nominal_clearance_m=-0.01, activation_clearance_m=0.001, paper_car_threshold_m=0.001)["correction_l2_action"], 0.5)

    def test_never_selects_contact_or_car_candidate(self):
        def item(norm, margin, contacts=0, car=0.0):
            return {"correction_l2_action": norm, "record": {"minimum_clearance_m": margin, "protected_contacts": [{}] * contacts, "maximum_active_obstacle_l1_displacement_m": car}}
        values = [item(0.25, 0.01, contacts=1), item(0.5, 0.01, car=0.0011)]
        self.assertIsNone(select_repulsive_candidate(values, nominal_clearance_m=-0.01, activation_clearance_m=0.001, paper_car_threshold_m=0.001))


if __name__ == "__main__":
    unittest.main()
