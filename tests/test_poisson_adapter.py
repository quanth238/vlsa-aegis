import importlib.util
import math
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "main" / "poisson_fullbody" / "joint_velocity_adapter.py"
NUMPY_PRESENT = importlib.util.find_spec("numpy") is not None


@unittest.skipUnless(NUMPY_PRESENT, "NumPy unavailable")
class JointVelocityAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy as np

        from main.poisson_fullbody.joint_velocity_adapter import (
            HIGH_LEVEL_DT_SECONDS,
            INNER_DT_SECONDS,
            INNER_UPDATES_PER_ACTION,
            JV_PHYSICAL_LIMIT_RAD_S,
            OSC_POSITION_DELTA_SCALE_M,
            TranslationalJointVelocityAdapter,
            create_translational_pose_target,
            damped_least_squares_velocity,
            normalized_joint_velocity_action,
            rotation_vector_error,
        )

        cls.np = np
        cls.Adapter = TranslationalJointVelocityAdapter
        cls.create_target = staticmethod(create_translational_pose_target)
        cls.dls = staticmethod(damped_least_squares_velocity)
        cls.normalize_action = staticmethod(normalized_joint_velocity_action)
        cls.rotation_error = staticmethod(rotation_vector_error)
        cls.position_scale = OSC_POSITION_DELTA_SCALE_M
        cls.high_dt = HIGH_LEVEL_DT_SECONDS
        cls.inner_dt = INNER_DT_SECONDS
        cls.inner_updates = INNER_UPDATES_PER_ACTION
        cls.velocity_limit = JV_PHYSICAL_LIMIT_RAD_S

    def test_frozen_timing_and_controller_scales(self):
        self.assertEqual(self.position_scale, 0.05)
        self.assertEqual(self.high_dt, 0.05)
        self.assertEqual(self.inner_dt, 0.01)
        self.assertEqual(self.inner_updates, 5)
        self.assertEqual(self.velocity_limit, 0.5)
        self.assertAlmostEqual(self.high_dt, self.inner_dt * self.inner_updates)

    def test_target_matches_released_osc_xyz_clip_and_holds_orientation(self):
        np = self.np
        current_position = np.array([0.4, -0.1, 0.8])
        angle = 0.3
        current_rotation = np.array(
            [
                [math.cos(angle), -math.sin(angle), 0.0],
                [math.sin(angle), math.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        action = np.array([2.0, -0.5, -3.0, 0.7, -0.4, 0.1, -0.37])
        target = self.create_target(action, current_position, current_rotation)
        np.testing.assert_allclose(target.clipped_translation, [1.0, -0.5, -1.0])
        np.testing.assert_allclose(target.position_delta_m, [0.05, -0.025, -0.05])
        np.testing.assert_allclose(
            target.target_position,
            current_position + np.array([0.05, -0.025, -0.05]),
        )
        np.testing.assert_array_equal(target.target_rotation, current_rotation)
        np.testing.assert_array_equal(
            target.ignored_rotation_components, action[3:6]
        )
        self.assertEqual(target.gripper_command, action[6])

    def test_damped_six_by_seven_least_squares_fixture(self):
        np = self.np
        jacobian = np.zeros((6, 7))
        jacobian[:, :6] = np.eye(6)
        twist = np.array([0.01, -0.02, 0.03, 0.04, -0.05, 0.06])
        damping = 0.1
        expected = np.zeros(7)
        expected[:6] = twist / (1.0 + damping * damping)
        observed = self.dls(jacobian, twist, damping=damping)
        np.testing.assert_allclose(observed, expected, rtol=1e-13, atol=1e-13)

    def test_physical_clip_then_half_rad_s_normalization_and_gripper_copy(self):
        np = self.np
        adapter = self.Adapter(damping=1.0e-6)
        action = np.array([1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.231])
        adapter.begin_high_level_action(action, np.zeros(3), np.eye(3))
        jacobian = np.zeros((6, 7))
        jacobian[:, :6] = np.eye(6)
        step = adapter.compute_inner_action(np.zeros(3), np.eye(3), jacobian)
        np.testing.assert_allclose(step.qdot_physical[:2], [0.5, -0.5])
        np.testing.assert_allclose(step.normalized_action[:2], [1.0, -1.0])
        self.assertEqual(step.normalized_action[7], action[6])
        self.assertEqual(step.diagnostics["saturated_joint_count"], 2)
        self.assertEqual(
            step.diagnostics["joint_velocity_normalization_scale_rad_s"],
            0.5,
        )
        execution = adapter.record_executed_joint_velocity(step.qdot_physical)
        np.testing.assert_allclose(execution["filter_correction_rad_s"], 0.0)
        self.assertEqual(execution["normalized_executed_action"][7], action[6])

    def test_target_is_held_and_error_is_recomputed_for_exactly_five_updates(self):
        np = self.np
        adapter = self.Adapter(damping=0.01)
        target = adapter.begin_high_level_action(
            [1.0, 0.0, 0.0, 0.9, -0.3, 0.4, -1.0],
            np.zeros(3),
            np.eye(3),
        )
        jacobian = np.zeros((6, 7))
        jacobian[:, :6] = np.eye(6)
        errors = []
        for update in range(5):
            position = np.array([0.01 * update, 0.0, 0.0])
            step = adapter.compute_inner_action(position, np.eye(3), jacobian)
            errors.append(step.diagnostics["position_error_m"][0])
            self.assertEqual(step.diagnostics["inner_update_index"], update)
            np.testing.assert_allclose(adapter.target.target_position, target.target_position)
            adapter.record_executed_joint_velocity(step.qdot_physical)
        np.testing.assert_allclose(errors, [0.05, 0.04, 0.03, 0.02, 0.01])
        self.assertEqual(adapter.updates_remaining, 0)
        with self.assertRaisesRegex(RuntimeError, "already consumed"):
            adapter.compute_inner_action([0.05, 0, 0], np.eye(3), jacobian)

    def test_new_target_cannot_replace_partially_consumed_target(self):
        np = self.np
        adapter = self.Adapter()
        adapter.begin_high_level_action(np.zeros(7), np.zeros(3), np.eye(3))
        with self.assertRaisesRegex(RuntimeError, "cannot replace"):
            adapter.begin_high_level_action(np.zeros(7), np.zeros(3), np.eye(3))

    def test_world_frame_orientation_error_and_tracking_diagnostics(self):
        np = self.np
        angle = 0.1
        rotated = np.array(
            [
                [math.cos(angle), -math.sin(angle), 0.0],
                [math.sin(angle), math.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        error = self.rotation_error(np.eye(3), rotated)
        np.testing.assert_allclose(error, [0.0, 0.0, -angle], atol=1e-14)

        adapter = self.Adapter(damping=0.01)
        adapter.begin_high_level_action(np.zeros(7), np.zeros(3), np.eye(3))
        jacobian = np.zeros((6, 7))
        jacobian[:, :6] = np.eye(6)
        first = adapter.compute_inner_action(
            np.zeros(3), rotated, jacobian, measured_joint_velocity=np.zeros(7)
        )
        self.assertFalse(first.diagnostics["tracking_reference_available"])
        adapter.record_executed_joint_velocity(first.qdot_physical)
        second = adapter.compute_inner_action(
            np.zeros(3),
            rotated,
            jacobian,
            measured_joint_velocity=first.qdot_physical,
        )
        self.assertTrue(second.diagnostics["tracking_reference_available"])
        self.assertEqual(
            second.diagnostics["previous_interval_tracking_error_l2_rad_s"],
            0.0,
        )

    def test_tracking_uses_actual_post_filter_command_not_nominal(self):
        np = self.np
        adapter = self.Adapter(damping=1.0e-6)
        adapter.begin_high_level_action(
            [1, 0, 0, 0, 0, 0, -1], np.zeros(3), np.eye(3)
        )
        jacobian = np.zeros((6, 7))
        jacobian[:, :6] = np.eye(6)
        nominal = adapter.compute_inner_action(np.zeros(3), np.eye(3), jacobian)
        self.assertGreater(float(nominal.qdot_physical[0]), 0.0)
        execution = adapter.record_executed_joint_velocity(np.zeros(7))
        self.assertGreater(execution["filter_correction_l2_rad_s"], 0.0)
        following = adapter.compute_inner_action(
            np.zeros(3),
            np.eye(3),
            jacobian,
            measured_joint_velocity=np.zeros(7),
        )
        self.assertEqual(
            following.diagnostics[
                "previous_interval_tracking_error_l2_rad_s"
            ],
            0.0,
        )

    def test_every_inner_update_requires_an_execution_record(self):
        np = self.np
        adapter = self.Adapter()
        adapter.begin_high_level_action(np.zeros(7), np.zeros(3), np.eye(3))
        jacobian = np.zeros((6, 7))
        first = adapter.compute_inner_action(np.zeros(3), np.eye(3), jacobian)
        with self.assertRaisesRegex(RuntimeError, "record_executed"):
            adapter.compute_inner_action(np.zeros(3), np.eye(3), jacobian)
        adapter.record_executed_joint_velocity(first.qdot_physical)
        adapter.compute_inner_action(np.zeros(3), np.eye(3), jacobian)

    def test_normalization_rejects_unbounded_physical_commands(self):
        np = self.np
        action = self.normalize_action(
            [0.5, -0.5, 0, 0, 0, 0, 0], gripper_command=0.73
        )
        np.testing.assert_allclose(action[:2], [1.0, -1.0])
        self.assertEqual(action[7], 0.73)
        with self.assertRaisesRegex(ValueError, "physical limit"):
            self.normalize_action(
                [0.51, 0, 0, 0, 0, 0, 0], gripper_command=0.73
            )

    def test_singular_jacobian_is_finite_and_reported(self):
        np = self.np
        adapter = self.Adapter()
        adapter.begin_high_level_action(
            [1, 0, 0, 0, 0, 0, 1], np.zeros(3), np.eye(3)
        )
        step = adapter.compute_inner_action(
            np.zeros(3), np.eye(3), np.zeros((6, 7))
        )
        np.testing.assert_array_equal(step.qdot_physical, np.zeros(7))
        self.assertEqual(step.diagnostics["jacobian_rank"], 0)
        self.assertIsNone(step.diagnostics["jacobian_condition_number"])
        self.assertGreater(step.diagnostics["twist_residual_norm"], 0.0)

    def test_invalid_inputs_fail_before_state_advances(self):
        np = self.np
        adapter = self.Adapter()
        with self.assertRaises(ValueError):
            adapter.begin_high_level_action(
                [0, 0, 0, 0, 0, float("nan"), 1], np.zeros(3), np.eye(3)
            )
        adapter.begin_high_level_action(np.zeros(7), np.zeros(3), np.eye(3))
        with self.assertRaises(ValueError):
            adapter.compute_inner_action(np.zeros(3), np.eye(3), np.zeros((3, 7)))
        self.assertEqual(adapter.inner_update_index, 0)

    def test_source_contains_no_draft_velocity_scale_formula(self):
        source = MODULE_PATH.read_text(encoding="utf-8")
        self.assertNotIn("0.2", source)
        self.assertIsNone(
            re.search(r"0[.]2\s*[*]\s*(?:np[.])?(?:asarray[(])?action", source)
        )
        self.assertIn("OSC_POSITION_DELTA_SCALE_M = 0.05", source)
        self.assertIn("JV_PHYSICAL_LIMIT_RAD_S = 0.5", source)


if __name__ == "__main__":
    unittest.main()
