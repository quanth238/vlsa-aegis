import unittest

import numpy as np

from main.multilink_ellipsoid.geometry import Ellipsoid
from main.multilink_ellipsoid.obstacle_proxy_audit import (
    CompiledObstacleBox,
    evaluate_obstacle_representations,
    minimum_ellipsoid_quadratic_over_box,
    minimum_ellipsoid_quadratics_over_boxes,
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

    def test_representation_audit_reports_per_row_exact_values(self):
        links = [
            Ellipsoid(np.asarray([float(index) * 3.0, 0.0, 0.0]), np.eye(3), np.ones(3))
            for index in range(7)
        ]
        perceived = Ellipsoid(np.asarray([0.0, 5.0, 0.0]), np.eye(3), np.ones(3))
        record = evaluate_obstacle_representations(
            links, perceived, [self._box((1.25, 0.0, 0.0))]
        )
        self.assertEqual(
            len(record["compiled_box_union_row_minimum_normalized_radial_slack"]), 7
        )
        self.assertEqual(
            record["compiled_box_union_row_any_exact_solid_overlap"][0], True
        )
        self.assertEqual(
            record["compiled_box_union_row_any_exact_solid_overlap"][6], False
        )

    def test_batched_active_set_exactly_matches_scalar_solver(self):
        robots = [
            Ellipsoid(
                np.asarray([0.1, -0.2, 0.3]), np.eye(3),
                np.asarray([0.4, 0.2, 0.1]),
            ),
            Ellipsoid(
                np.asarray([-0.3, 0.15, 0.2]), np.eye(3),
                np.asarray([0.15, 0.25, 0.35]),
            ),
        ]
        boxes = [
            self._box((0.65, 0.05, 0.35), half=(0.08, 0.12, 0.07)),
            self._box((-0.2, 0.2, 0.2), half=(0.05, 0.09, 0.11)),
        ]
        batched = minimum_ellipsoid_quadratics_over_boxes(robots, boxes)
        scalar = np.asarray([
            [minimum_ellipsoid_quadratic_over_box(robot, box) for box in boxes]
            for robot in robots
        ])
        np.testing.assert_allclose(batched, scalar, rtol=0.0, atol=1.0e-12)


if __name__ == "__main__":
    unittest.main()
