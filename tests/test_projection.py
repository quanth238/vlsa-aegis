from __future__ import annotations

import unittest

from crfs_harness.geometry import SphereObstacle, rollout_positions, swept_sphere_clearance
from crfs_harness.intervention import endpoint_sum
from crfs_harness.projection import sine_bump_repair


class ProjectionTest(unittest.TestCase):
    def test_synthetic_repair_is_safe_and_endpoint_preserving(self) -> None:
        nominal = tuple((0.4, 0.0, 0.0) for _ in range(5))
        obstacle = SphereObstacle((0.0, 0.05, 0.0), 0.24)
        nominal_positions = rollout_positions((-1.0, 0.0, 0.0), nominal, 1.0)
        self.assertLess(swept_sphere_clearance(nominal_positions, obstacle, 0.05), 0.0)
        repair = sine_bump_repair(nominal, (-1.0, 0.0, 0.0), obstacle, 0.05, 0.02)
        self.assertIsNotNone(repair)
        assert repair is not None
        self.assertGreaterEqual(repair.clearance_m, 0.02 - 1e-10)
        self.assertLess(repair.endpoint_error_m, 1e-12)
        for value in endpoint_sum(repair.correction):
            self.assertAlmostEqual(value, 0.0)


if __name__ == "__main__":
    unittest.main()
