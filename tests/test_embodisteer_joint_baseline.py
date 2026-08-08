from pathlib import Path
import sys
import unittest

try:
    import numpy as np
except ImportError:  # local structural gate intentionally has no scientific stack
    np = None


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from main.multilink_ellipsoid.embodisteer_joint_baseline import (  # noqa: E402
    apply_joint_denoising_residual,
    damped_pseudoinverse,
    initialize_joint_trajectory,
    incremental_actions_to_world_poses,
    joint_trajectory_to_pose_actions,
    load_joint_baseline_config,
    matrix_to_rotation_vector,
    rotation_vector_to_matrix,
)


CONFIG = ROOT / "configs/vlsa_embodisteer_joint_baselines_e05.v1.json"


def _linear_kinematics(configuration):
    q = np.asarray(configuration, dtype=np.float64)
    jacobian = np.zeros((6, 7), dtype=np.float64)
    jacobian[:, :6] = np.eye(6)
    position = q[:3]
    rotation = rotation_vector_to_matrix(q[3:6])
    return position, rotation, jacobian


@unittest.skipIf(np is None, "NumPy is available in the H100 evaluation environment")
class EmbodiSteerJointBaselineTest(unittest.TestCase):
    def setUp(self):
        self.lower = np.full(7, -2.0)
        self.upper = np.full(7, 2.0)

    def test_preregistered_config_uses_same_policy_without_guidance(self):
        config = load_joint_baseline_config(CONFIG)

        self.assertEqual(
            config["arms"],
            ["cartesian_ee_no_guidance", "joint_denoising_no_guidance"],
        )
        self.assertEqual(config["nominal_policy"]["checkpoint"], "pi05_libero")
        self.assertTrue(config["nominal_policy"]["same_checkpoint_both_arms"])
        self.assertFalse(config["collision_guidance"]["barrier_projection_enabled"])
        self.assertFalse(config["collision_guidance"]["ellipsoid_constraints_enabled"])
        self.assertFalse(config["collision_guidance"]["qp_enabled"])

    def test_damped_pseudoinverse_is_finite_and_has_paper_shape(self):
        jacobian = np.zeros((6, 7))
        jacobian[:, :6] = np.eye(6)

        result = damped_pseudoinverse(jacobian, 0.001)

        self.assertEqual(result.shape, (7, 6))
        np.testing.assert_allclose(
            result[:6], np.eye(6) / 1.001, rtol=0.0, atol=1.0e-12
        )

    def test_rotation_vector_round_trip(self):
        for vector in (
            np.array([0.0, 0.0, 0.0]),
            np.array([0.1, -0.2, 0.3]),
            np.array([-0.7, 0.2, 0.4]),
        ):
            recovered = matrix_to_rotation_vector(rotation_vector_to_matrix(vector))
            np.testing.assert_allclose(recovered, vector, atol=1.0e-10, rtol=0.0)

    def test_joint_initialization_uses_chunk_start_jacobian(self):
        noise = np.zeros((10, 6), dtype=np.float64)
        noise[:, 0] = 1.0

        trajectory, diagnostics = initialize_joint_trajectory(
            noise,
            np.zeros(7),
            _linear_kinematics,
            alpha=0.1,
            damping=0.001,
            update_clip_rad=0.5,
            translation_scale_m=0.05,
            rotation_scale_rad=0.5,
            lower=self.lower,
            upper=self.upper,
        )
        actions = joint_trajectory_to_pose_actions(
            trajectory,
            np.zeros(7),
            _linear_kinematics,
            translation_scale_m=0.05,
            rotation_scale_rad=0.5,
        )

        expected_joint_step = 0.1 * 0.05 / 1.001
        np.testing.assert_allclose(
            trajectory[:, 0],
            expected_joint_step * np.arange(1, 11),
            atol=1.0e-12,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            actions[:, 0], expected_joint_step / 0.05, atol=1.0e-12, rtol=0.0
        )
        self.assertEqual(diagnostics["total_clipped_joint_dimensions"], 0)

    def test_incremental_actions_compose_into_chunk_start_targets(self):
        actions = np.zeros((10, 6), dtype=np.float64)
        actions[:, 0] = 0.2
        positions, rotations = incremental_actions_to_world_poses(
            actions,
            np.zeros(3),
            np.eye(3),
            translation_scale_m=0.05,
            rotation_scale_rad=0.5,
        )
        np.testing.assert_allclose(
            positions[:, 0], 0.01 * np.arange(1, 11), atol=1.0e-12, rtol=0.0
        )
        np.testing.assert_allclose(rotations, np.repeat(np.eye(3)[None], 10, axis=0))

    def test_joint_residual_lifts_each_cartesian_flow_update(self):
        trajectory = np.zeros((10, 7), dtype=np.float64)
        current = np.zeros((10, 6), dtype=np.float64)
        next_actions = np.zeros((10, 6), dtype=np.float64)
        next_actions[:, 1] = 0.2

        updated, diagnostics = apply_joint_denoising_residual(
            trajectory,
            np.zeros(7),
            current,
            next_actions,
            _linear_kinematics,
            damping=0.001,
            update_clip_rad=0.5,
            translation_scale_m=0.05,
            rotation_scale_rad=0.5,
            lower=self.lower,
            upper=self.upper,
        )

        np.testing.assert_allclose(
            updated[:, 1],
            (0.05 * 0.2 / 1.001) * np.arange(1, 11),
            atol=1.0e-12,
            rtol=0.0,
        )
        self.assertEqual(diagnostics["total_clipped_joint_dimensions"], 0)


if __name__ == "__main__":
    unittest.main()
