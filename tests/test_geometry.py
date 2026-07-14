from __future__ import annotations

import unittest

from crfs_harness.geometry import SphereObstacle, rollout_positions, swept_sphere_clearance


class GeometryTest(unittest.TestCase):
    def test_swept_segment_detects_collision_between_waypoints(self) -> None:
        actions = ((2.0, 0.0, 0.0),)
        positions = rollout_positions((-1.0, 0.0, 0.0), actions, 1.0)
        clearance = swept_sphere_clearance(positions, SphereObstacle((0.0, 0.0, 0.0), 0.2), 0.1)
        self.assertAlmostEqual(clearance, -0.3)

    def test_rollout_respects_kappa(self) -> None:
        positions = rollout_positions((0.0, 0.0, 0.0), ((1.0, -2.0, 0.5),), 0.2)
        self.assertEqual(positions[-1], (0.2, -0.4, 0.1))


if __name__ == "__main__":
    unittest.main()
