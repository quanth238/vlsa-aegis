import math
import unittest


try:
    import numpy as np
except ImportError:  # pragma: no cover - baseline-only local environment
    np = None


@unittest.skipIf(np is None, "NumPy is an opt-in Poisson dependency")
class RigidObstacleFieldTests(unittest.TestCase):
    def setUp(self):
        from main.poisson_fullbody.rigid_obstacle_field import (
            RigidPose,
            RigidTwist,
        )

        self.RigidPose = RigidPose
        self.RigidTwist = RigidTwist

    @staticmethod
    def linear_field(gradient, offset=2.0):
        gradient = np.asarray(gradient, dtype=np.float64)

        class Query:
            def __init__(self, point):
                self.valid = True
                self.value = float(offset + gradient @ point)
                self.gradient = gradient.copy()

        class Field:
            def query(self, point):
                return Query(np.asarray(point, dtype=np.float64))

        return Field()

    def test_translation_maps_query_and_adds_time_derivative(self):
        from main.poisson_fullbody.rigid_obstacle_field import (
            query_rigid_obstacle_field,
        )

        reference = self.RigidPose([0.0, 0.0, 0.0], np.eye(3))
        current = self.RigidPose([1.0, 0.0, 0.0], np.eye(3))
        twist = self.RigidTwist([0.25, 0.0, 0.0], [0.0, 0.0, 0.0])
        query = query_rigid_obstacle_field(
            self.linear_field([3.0, 0.0, 0.0]),
            [1.5, 0.0, 0.0],
            reference_pose=reference,
            current_pose=current,
            current_twist=twist,
        )
        self.assertTrue(query.valid)
        np.testing.assert_array_equal(query.reference_point_world_m, [0.5, 0.0, 0.0])
        np.testing.assert_array_equal(query.gradient_world_m, [3.0, 0.0, 0.0])
        self.assertAlmostEqual(query.value_m2, 3.5)
        self.assertAlmostEqual(query.partial_time_m2_per_s, -0.75)

    def test_rotation_transforms_gradient_and_uses_point_velocity(self):
        from main.poisson_fullbody.rigid_obstacle_field import (
            query_rigid_obstacle_field,
        )

        rotation = np.asarray(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        )
        reference = self.RigidPose([0.0, 0.0, 0.0], np.eye(3))
        current = self.RigidPose([0.0, 0.0, 0.0], rotation)
        twist = self.RigidTwist([0.0, 0.0, 0.0], [0.0, 0.0, 2.0])
        query = query_rigid_obstacle_field(
            self.linear_field([1.0, 0.0, 0.0]),
            [0.0, 1.0, 0.0],
            reference_pose=reference,
            current_pose=current,
            current_twist=twist,
        )
        np.testing.assert_allclose(query.reference_point_world_m, [1.0, 0.0, 0.0])
        np.testing.assert_allclose(query.gradient_world_m, [0.0, 1.0, 0.0])
        np.testing.assert_allclose(
            query.obstacle_point_velocity_world_m_per_s,
            [-2.0, 0.0, 0.0],
        )
        self.assertAlmostEqual(query.partial_time_m2_per_s, 0.0)

    def test_dynamic_cbf_lower_bound_has_correct_obstacle_motion_sign(self):
        from main.poisson_fullbody.rigid_obstacle_field import dynamic_cbf_rows

        h = np.asarray([0.2])
        gradient = np.asarray([[1.0, 0.0, 0.0]])
        jacobian = np.zeros((1, 3, 7), dtype=np.float64)
        jacobian[0, 0, 0] = 2.0
        rows, lower = dynamic_cbf_rows(
            h,
            gradient,
            jacobian,
            np.asarray([-0.3]),
            alpha_per_s=4.0,
            margin_m2_per_s=0.1,
        )
        np.testing.assert_array_equal(rows, [[2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        # row*qdot - 0.3 + 0.8 - 0.1 >= 0 -> row*qdot >= -0.4
        self.assertTrue(math.isclose(float(lower[0]), -0.4, abs_tol=1.0e-15))


if __name__ == "__main__":
    unittest.main()
