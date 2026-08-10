import unittest
from pathlib import Path

import numpy as np

from main.multilink_ellipsoid.geometry import Ellipsoid
from main.multilink_ellipsoid.surface_error_diagnostic import (
    ellipsoid_support_point, matched_control_indexes, spearman_correlation,
    surface_loss_decision, load_surface_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "vlsa_distal_factorized_surface_error_moka10.v1.json"


class FactorizedSurfaceErrorTest(unittest.TestCase):
    def test_frozen_config_loads(self):
        config = load_surface_config(CONFIG)
        self.assertEqual(config["population"]["expected_false_safe_action_count"], 44)

    def test_support_point_handles_oriented_anisotropic_ellipsoid(self):
        ellipsoid = Ellipsoid(
            center=np.asarray([1.0, 2.0, 3.0]), rotation=np.eye(3),
            semiaxes_m=np.asarray([2.0, 1.0, 0.5]),
        )
        point = ellipsoid_support_point(ellipsoid, [1.0, 1.0, 0.0])
        direction = np.asarray([1.0, 1.0, 0.0]) / np.sqrt(2.0)
        self.assertAlmostEqual(
            float(direction @ (point - ellipsoid.center)),
            ellipsoid.support_radius(direction), places=12,
        )
        local = (point - ellipsoid.center) / ellipsoid.semiaxes_m
        self.assertAlmostEqual(float(local @ local), 1.0, places=12)

    def test_matching_is_same_state_and_boundary_nearest(self):
        states = np.asarray([0, 0, 0, 1, 1, 1])
        margins = np.asarray([-0.004, -0.003, -0.020, -0.002, -0.001, -0.030])
        selected = matched_control_indexes(
            false_safe_indexes=np.asarray([0, 3]),
            control_indexes=np.asarray([1, 2, 4, 5]),
            state_index=states, exact_margin_m=margins,
        )
        self.assertEqual(selected.tolist(), [1, 4])

    def test_decision_requires_all_surface_evidence(self):
        config = {
            "population": {"expected_false_safe_action_count": 44},
            "decision_gate": {
                "minimum_false_safe_surface_error_median_excess_m": 0.0005,
                "minimum_paired_surface_error_win_fraction": 0.65,
                "minimum_surface_retraction_margin_overestimate_spearman": 0.5,
            },
        }
        metrics = {
            "false_safe_action_count": 44,
            "recomputed_geometry_exact": True,
            "false_safe_surface_error_median_excess_m": 0.0006,
            "paired_surface_error_win_fraction": 0.7,
            "unsafe_surface_retraction_margin_overestimate_spearman": 0.6,
        }
        self.assertTrue(surface_loss_decision(metrics, config)[
            "surface_loss_pilot_supported"
        ])
        metrics["paired_surface_error_win_fraction"] = 0.6
        self.assertFalse(surface_loss_decision(metrics, config)[
            "surface_loss_pilot_supported"
        ])

    def test_spearman_is_tie_aware(self):
        self.assertAlmostEqual(
            spearman_correlation([1.0, 2.0, 2.0, 4.0], [10.0, 20.0, 20.0, 40.0]),
            1.0, places=12,
        )


if __name__ == "__main__":
    unittest.main()
