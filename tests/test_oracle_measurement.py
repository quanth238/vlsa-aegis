from __future__ import annotations

import unittest

try:
    import numpy as np
    from crfs_oracle.measurement import signed_distance_point_to_oriented_box
except ModuleNotFoundError:  # Keep the repository bootstrap dependency-free.
    np = None
    signed_distance_point_to_oriented_box = None


@unittest.skipUnless(np is not None, "NumPy is exercised in the SafeLIBERO environment")
class OrientedBoxDistanceTest(unittest.TestCase):
    def test_axis_aligned_outside_distance(self) -> None:
        distance = signed_distance_point_to_oriented_box(
            (2.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            np.eye(3),
            (1.0, 0.5, 0.25),
        )
        self.assertAlmostEqual(distance, 1.0)

    def test_axis_aligned_inside_distance_is_negative(self) -> None:
        distance = signed_distance_point_to_oriented_box(
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            np.eye(3),
            (1.0, 0.5, 0.25),
        )
        self.assertAlmostEqual(distance, -0.25)

    def test_rotated_box_uses_world_rotation(self) -> None:
        rotation = np.asarray(((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)))
        # The box's local x half-extent (1 m) is aligned with world y.
        distance = signed_distance_point_to_oriented_box(
            (0.0, 1.2, 0.0),
            (0.0, 0.0, 0.0),
            rotation,
            (1.0, 0.5, 0.25),
        )
        self.assertAlmostEqual(distance, 0.2)

    def test_sphere_clearance_is_point_distance_minus_radius(self) -> None:
        point_distance = signed_distance_point_to_oriented_box(
            (1.1, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            np.eye(3),
            (1.0, 1.0, 1.0),
        )
        self.assertAlmostEqual(point_distance - 0.2, -0.1)


if __name__ == "__main__":
    unittest.main()
