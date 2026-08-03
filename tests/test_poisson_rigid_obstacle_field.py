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

    def test_preserves_underlying_invalid_query_metadata(self):
        from main.poisson_fullbody.rigid_obstacle_field import (
            query_rigid_obstacle_field,
        )
        from main.poisson_fullbody.poisson_field import (
            FieldQuery,
            QueryInvalidReason,
        )

        class InvalidField:
            @staticmethod
            def query(point):
                del point
                return FieldQuery(
                    valid=False,
                    value=None,
                    gradient=None,
                    reason=QueryInvalidReason.INVALID_CELL,
                    cell_index=(4, 5, 6),
                    local_coordinates=(0.25, 0.5, 0.75),
                    outer_boundary_clearance_m=0.8,
                )

        pose = self.RigidPose([0.0, 0.0, 0.0], np.eye(3))
        twist = self.RigidTwist([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        query = query_rigid_obstacle_field(
            InvalidField(),
            [0.1, 0.2, 0.3],
            reference_pose=pose,
            current_pose=pose,
            current_twist=twist,
        )

        self.assertFalse(query.valid)
        self.assertIs(query.reason, QueryInvalidReason.INVALID_CELL)
        self.assertEqual(query.cell_index, (4, 5, 6))
        self.assertEqual(query.local_coordinates, (0.25, 0.5, 0.75))
        self.assertEqual(query.outer_boundary_clearance_m, 0.8)
        self.assertIsNone(query.value_m2)
        self.assertIsNone(query.gradient_world_m)
        self.assertIsNone(query.partial_time_m2_per_s)
        self.assertIsNone(query.obstacle_point_velocity_world_m_per_s)

    def test_preserves_valid_query_metadata_without_changing_numerics(self):
        from main.poisson_fullbody.rigid_obstacle_field import (
            query_rigid_obstacle_field,
        )

        class ValidQuery:
            valid = True
            value = 2.5
            gradient = np.asarray([1.0, 0.0, 0.0])
            reason = None
            cell_index = (7, 8, 9)
            local_coordinates = (0.1, 0.2, 0.3)
            outer_boundary_clearance_m = 0.6

        class ValidField:
            @staticmethod
            def query(point):
                del point
                return ValidQuery()

        pose = self.RigidPose([0.0, 0.0, 0.0], np.eye(3))
        twist = self.RigidTwist([0.25, 0.0, 0.0], [0.0, 0.0, 0.0])
        query = query_rigid_obstacle_field(
            ValidField(),
            [0.1, 0.2, 0.3],
            reference_pose=pose,
            current_pose=pose,
            current_twist=twist,
        )

        self.assertTrue(query.valid)
        self.assertEqual(query.value_m2, 2.5)
        np.testing.assert_array_equal(query.gradient_world_m, [1.0, 0.0, 0.0])
        self.assertEqual(query.partial_time_m2_per_s, -0.25)
        self.assertIsNone(query.reason)
        self.assertEqual(query.cell_index, (7, 8, 9))
        self.assertEqual(query.local_coordinates, (0.1, 0.2, 0.3))
        self.assertEqual(query.outer_boundary_clearance_m, 0.6)

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
