import json
import math
from pathlib import Path
import unittest

from main.multilink_ellipsoid.analytical_repulsion_generalization import (
    aggregate_case_results,
    corrected_proposal,
    frame_integrity_metrics,
    front_loaded_profile,
    load_cases,
    load_config,
    warning_trigger,
)


ROOT = Path(__file__).resolve().parents[1]


class AnalyticalRepulsionGeneralizationTest(unittest.TestCase):
    def test_contracts_load(self):
        config = load_config(
            ROOT / "configs/vlsa_distal_analytical_repulsion_generalization.v1.json"
        )
        cases = load_cases(
            ROOT / "manifests/vlsa_distal_analytical_repulsion_generalization.v1.jsonl"
        )
        self.assertEqual(config["population"]["case_count"], 3)
        self.assertEqual(len(cases), 3)
        self.assertEqual([row["source_split"] for row in cases], ["test"] * 3)

    def test_front_loaded_profile_is_unit_norm(self):
        profile = front_loaded_profile()
        self.assertEqual(len(profile), 5)
        self.assertAlmostEqual(sum(value * value for value in profile), 1.0)
        self.assertEqual(profile, sorted(profile, reverse=True))

    def test_correction_preserves_rotation_and_gripper(self):
        nominal = [[0.0, 0.0, 0.0, 0.1, -0.2, 0.3, -1.0] for _ in range(5)]
        result = corrected_proposal(nominal, [1.0, 0.0, 0.0])
        self.assertAlmostEqual(result["requested_correction_l2_action"], 2.0)
        self.assertTrue(result["clipped"])
        for source, corrected in zip(nominal, result["actions"]):
            self.assertEqual(corrected[3:], source[3:])
            self.assertEqual(corrected[1:3], source[1:3])
            self.assertLessEqual(abs(corrected[0]), 1.0)

    def test_warning_is_strict_or_of_all_authorities(self):
        safe = {
            "minimum_clearance_m": 0.001,
            "protected_contacts": [],
            "maximum_active_obstacle_l1_displacement_m": 0.001,
        }
        self.assertFalse(warning_trigger(safe))
        for key, value in (
            ("minimum_clearance_m", 0.0009),
            ("protected_contacts", [{"event": 1}]),
            ("maximum_active_obstacle_l1_displacement_m", 0.0011),
        ):
            row = dict(safe)
            row[key] = value
            self.assertTrue(warning_trigger(row))

    def test_strict_aggregate_keeps_failures(self):
        config = load_config(
            ROOT / "configs/vlsa_distal_analytical_repulsion_generalization.v1.json"
        )
        base = {
            "raw_L5_L7_contact_pass": True,
            "paper_car_pass": True,
            "native_task_success": True,
            "timeout": False,
            "warning_count": 1,
            "intervention_count": 1,
            "clipped_intervention_count": 1,
        }
        rows = [{"case_id": "case-%d" % i, **base} for i in range(3)]
        self.assertTrue(aggregate_case_results(rows, config)["strict_gate_pass"])
        rows[1]["native_task_success"] = False
        summary = aggregate_case_results(rows, config)
        self.assertFalse(summary["strict_gate_pass"])
        self.assertEqual(summary["safe_task_success_count"], 2)

    def test_frame_integrity_rejects_dense_checkerboard(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("NumPy is validated in the registered H100 runtime")

        smooth = np.zeros((32, 32, 3), dtype=np.uint8)
        checker = np.indices((32, 32)).sum(axis=0) % 2
        checker = np.repeat((checker * 255).astype(np.uint8)[..., None], 3, axis=2)
        self.assertEqual(frame_integrity_metrics(smooth)["maximum_neighbor_difference"], 0.0)
        self.assertGreater(
            frame_integrity_metrics(checker)["maximum_neighbor_difference"], 0.99
        )


if __name__ == "__main__":
    unittest.main()
