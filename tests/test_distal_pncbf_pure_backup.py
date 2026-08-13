import math
import unittest
from pathlib import Path

from main.multilink_ellipsoid.pure_backup import (
    load_pure_backup_config,
    orthonormal_local_frame,
    registered_directions,
    select_backup_with_fallback,
    select_verified_backup,
    temporal_profile,
)


class PureBackupTests(unittest.TestCase):
    def test_registered_config(self):
        config = load_pure_backup_config(
            Path(__file__).parents[1]
            / "configs/vlsa_distal_pncbf_pure_backup_audit_e05.v1.json"
        )
        self.assertTrue(config["candidate_family"]["vla_action_or_ledger_is_forbidden_as_policy_input"])

    def test_incremental_profile_has_unit_sum(self):
        profile = temporal_profile(5)
        self.assertEqual(len(profile), 5)
        self.assertAlmostEqual(sum(profile), 1.0)
        self.assertTrue(all(value >= 0.0 for value in profile))

    def test_local_frame_is_orthonormal(self):
        frame = orthonormal_local_frame([1.0, 2.0, 3.0])
        rows = list(frame.values())
        for row in rows:
            self.assertAlmostEqual(math.sqrt(sum(value * value for value in row)), 1.0)
        for left in range(3):
            for right in range(left + 1, 3):
                self.assertAlmostEqual(
                    sum(rows[left][i] * rows[right][i] for i in range(3)), 0.0
                )

    def test_registered_family_has_fixed_identity(self):
        directions = registered_directions([1.0, 0.0, 0.0])
        self.assertEqual(len(directions), 12)
        self.assertEqual(directions[0][0], "world_pos_x")
        self.assertEqual(directions[-1][0], "local_neg_tangent_side")

    def test_selection_is_order_field_not_iteration_order(self):
        def item(name, order, clearance):
            return {
                "name": name,
                "order": order,
                "record": {
                    "minimum_clearance_m": 0.002,
                    "future_minimum_clearance_m": clearance,
                    "protected_contact_count": 0,
                    "maximum_active_obstacle_l1_displacement_m": 0.0,
                },
            }

        candidates = [item("late", 2, 0.005), item("early", 1, 0.005)]
        forward = select_verified_backup(
            candidates, safety_buffer_m=0.001, paper_car_threshold_m=0.001
        )
        reverse = select_verified_backup(
            list(reversed(candidates)),
            safety_buffer_m=0.001,
            paper_car_threshold_m=0.001,
        )
        self.assertEqual(forward["name"], "early")
        self.assertEqual(reverse["name"], "early")

    def test_unsafe_candidate_is_never_selected(self):
        candidate = {
            "name": "unsafe",
            "order": 0,
            "record": {
                "minimum_clearance_m": 0.0009,
                "future_minimum_clearance_m": 1.0,
                "protected_contact_count": 0,
                "maximum_active_obstacle_l1_displacement_m": 0.0,
            },
        }
        self.assertIsNone(
            select_verified_backup(
                [candidate], safety_buffer_m=0.001, paper_car_threshold_m=0.001
            )
        )

    def test_complete_policy_has_deterministic_unsafe_fallback(self):
        candidates = [
            {
                "name": "a", "order": 0,
                "record": {
                    "minimum_clearance_m": -0.01,
                    "future_minimum_clearance_m": -0.01,
                    "protected_contact_count": 2,
                    "maximum_active_obstacle_l1_displacement_m": 0.0,
                },
            },
            {
                "name": "b", "order": 1,
                "record": {
                    "minimum_clearance_m": -0.005,
                    "future_minimum_clearance_m": -0.005,
                    "protected_contact_count": 3,
                    "maximum_active_obstacle_l1_displacement_m": 0.0,
                },
            },
        ]
        selected, mode = select_backup_with_fallback(
            candidates, safety_buffer_m=0.001, paper_car_threshold_m=0.001
        )
        self.assertEqual(selected["name"], "b")
        self.assertEqual(mode, "maximum_clearance_fallback")


if __name__ == "__main__":
    unittest.main()
