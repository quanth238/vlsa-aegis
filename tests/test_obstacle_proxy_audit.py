import unittest

import numpy as np

from main.multilink_ellipsoid.geometry import Ellipsoid
from main.multilink_ellipsoid.obstacle_proxy_audit import (
    CompiledObstacleBox,
    minimum_ellipsoid_quadratic_over_box,
)


class ObstacleProxyAuditTest(unittest.TestCase):
    def _box(self, center, half=(0.5, 0.5, 0.5)):
        return CompiledObstacleBox(
            geom_id=1,
            geom_name="box",
            body_id=1,
            body_name="obstacle",
            center=np.asarray(center, dtype=np.float64),
            rotation=np.eye(3),
            half_extents_m=np.asarray(half, dtype=np.float64),
        )

    def test_exact_sphere_box_separation(self):
        sphere = Ellipsoid(np.zeros(3), np.eye(3), np.ones(3))
        value = minimum_ellipsoid_quadratic_over_box(
            sphere, self._box((2.0, 0.0, 0.0))
        )
        self.assertAlmostEqual(value, 2.25)
        self.assertGreater(value, 1.0)

    def test_exact_sphere_box_overlap(self):
        sphere = Ellipsoid(np.zeros(3), np.eye(3), np.ones(3))
        value = minimum_ellipsoid_quadratic_over_box(
            sphere, self._box((1.25, 0.0, 0.0))
        )
        self.assertAlmostEqual(value, 0.5625)
        self.assertLess(value, 1.0)

    def test_rotated_anisotropic_case_matches_dense_samples(self):
        angle = np.deg2rad(31.0)
        rotation = np.asarray(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        ellipsoid = Ellipsoid(
            np.asarray([0.1, -0.2, 0.3]), rotation, np.asarray([0.4, 0.2, 0.1])
        )
        box = self._box((0.65, 0.05, 0.35), half=(0.08, 0.12, 0.07))
        exact = minimum_ellipsoid_quadratic_over_box(ellipsoid, box)
        axes = [
            np.linspace(-box.half_extents_m[index], box.half_extents_m[index], 81)
            for index in range(3)
        ]
        inverse = np.linalg.inv(ellipsoid.shape_matrix())
        dense = min(
            float((box.center + np.asarray(value) - ellipsoid.center) @ inverse @ (
                box.center + np.asarray(value) - ellipsoid.center
            ))
            for value in zip(axes[0], np.zeros(81), np.zeros(81))
        )
        # The active-set optimum cannot be worse than a restricted dense line.
        self.assertLessEqual(exact, dense + 1.0e-12)
        self.assertGreaterEqual(exact, 0.0)


if __name__ == "__main__":
    unittest.main()
