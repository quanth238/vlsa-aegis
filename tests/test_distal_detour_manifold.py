import unittest

try:
    import numpy as np
except ImportError:  # Local macOS gate lacks the scientific runtime.
    np = None

from main.multilink_ellipsoid.detour_manifold import (
    candidate_gate,
    controllability_gate,
    finite_soft_library,
    task_relative_basis,
)


@unittest.skipIf(np is None, "NumPy runtime unavailable")
class DetourManifoldTest(unittest.TestCase):
    def test_basis_and_endpoint_control(self):
        actions = np.asarray(
            [[0.2, -0.1, 0.1], [0.3, -0.1, 0.0], [0.2, 0.0, 0.0], [0.1, 0.1, 0.0], [0.1, 0.1, -0.1]]
        )
        value = task_relative_basis(actions)
        self.assertEqual(value["soft_matrix"].shape, (15, 5))
        self.assertEqual(value["endpoint_matrix"].shape, (15, 4))
        self.assertTrue(
            np.allclose(value["endpoint_matrix"].reshape(5, 3, 4).sum(axis=0), 0.0)
        )
        self.assertAlmostEqual(float(np.dot(value["progress_direction"], value["transverse_up"])), 0.0)
        self.assertAlmostEqual(float(np.dot(value["progress_direction"], value["transverse_side"])), 0.0)

    def test_finite_library_is_fixed_and_unique(self):
        rows = finite_soft_library()
        self.assertEqual(len(rows), 54)
        self.assertEqual(rows[0]["family"], "nominal")
        self.assertEqual(len({tuple(row["coefficients"]) for row in rows}), 54)

    def test_candidate_gate_separates_proxy(self):
        record = {
            "exact_overlap_sample_count": 0,
            "minimum_exact_normalized_radial_slack": 0.01,
            "protected_contact_count": 0,
            "maximum_active_obstacle_l1_displacement_m": 0.0002,
            "terminal_eef_error_m": 0.01,
            "task_progress_ratio": 0.7,
            "minimum_proxy_margin_m": -0.05,
        }
        gate = {
            "paper_car_threshold_m": 0.001,
            "maximum_terminal_eef_error_m": 0.015,
            "minimum_task_progress_ratio": 0.5,
        }
        self.assertTrue(candidate_gate(record, gate)["pass"])

    def test_controllability_requires_early_influence(self):
        record = {
            "initial_exact_overlap": False,
            "initial_protected_contact_count": 0,
            "first_influence_sample_index": 2,
            "first_physical_violation_sample_index": 8,
            "maximum_previolation_link_center_change_m": 0.002,
            "nominal_physical_violation_count": 3,
            "nominal_protected_contact_count": 2,
        }
        gate = {"minimum_previolation_link_center_change_m": 0.001}
        self.assertTrue(controllability_gate(record, gate)["pass"])
        record["first_influence_sample_index"] = 9
        self.assertFalse(controllability_gate(record, gate)["pass"])


if __name__ == "__main__":
    unittest.main()
