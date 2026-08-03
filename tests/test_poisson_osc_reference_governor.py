import inspect
from pathlib import Path
from types import SimpleNamespace
import unittest


try:
    import numpy as np
except ImportError:  # pragma: no cover - baseline-only local environment
    np = None

try:
    import osqp  # noqa: F401
    import scipy  # noqa: F401
except ImportError:  # pragma: no cover - allocation dependency
    osqp = None


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "main" / "poisson_fullbody" / "osc_reference_governor.py"


class OscReferenceGovernorStructuralTests(unittest.TestCase):
    def test_native_identity_branch_is_explicit(self):
        source = MODULE.read_text(encoding="utf-8")
        self.assertIn('reason="nominal_safe_exact_passthrough"', source)
        self.assertIn('returned.tobytes(order="C") == source.tobytes(order="C")', source)
        self.assertIn('"solver_attempted": False', source)

    def test_no_zero_or_stop_fallback_is_implemented(self):
        source = MODULE.read_text(encoding="utf-8")
        self.assertNotIn("zeros_like(source", source)
        self.assertNotIn("zero_velocity_fallback", source)
        self.assertIn("action=None", source)


@unittest.skipIf(np is None, "NumPy is an opt-in Poisson dependency")
class OscReferenceGovernorLiveControllerTests(unittest.TestCase):
    def test_live_native_input_clip_and_output_scale_are_bound(self):
        from main.poisson_fullbody.osc_reference_governor import osc_output_scale

        controller = SimpleNamespace(
            input_min=-1.0,
            input_max=1.0,
            output_min=-np.asarray([0.05, 0.05, 0.05, 0.5, 0.5, 0.5]),
            output_max=np.asarray([0.05, 0.05, 0.05, 0.5, 0.5, 0.5]),
        )
        np.testing.assert_array_equal(
            osc_output_scale(controller),
            controller.output_max,
        )

    def test_non_native_input_clip_is_rejected(self):
        from main.poisson_fullbody.osc_reference_governor import osc_output_scale

        controller = SimpleNamespace(
            input_min=-2.0,
            input_max=2.0,
            output_min=-np.ones(6),
            output_max=np.ones(6),
        )
        with self.assertRaisesRegex(ValueError, "exactly"):
            osc_output_scale(controller)


@unittest.skipIf(
    np is None or osqp is None,
    "NumPy, SciPy, and OSQP are allocation dependencies",
)
class OscReferenceGovernorNumericTests(unittest.TestCase):
    def setUp(self):
        from main.poisson_fullbody.osc_reference_governor import (
            OscPoseReferenceGovernor,
        )

        self.governor = OscPoseReferenceGovernor()
        self.jacobian = np.zeros((6, 7), dtype=np.float64)
        self.jacobian[:, :6] = np.eye(6)
        self.scale = np.asarray([0.05, 0.05, 0.05, 0.5, 0.5, 0.5])

    def filter(self, source, rows, lower, measured_qdot=None):
        if measured_qdot is None:
            measured_qdot = np.zeros(7, dtype=np.float64)
        return self.governor.filter_action(
            source,
            eef_jacobian=self.jacobian,
            measured_arm_qvel_rad_per_s=measured_qdot,
            cbf_rows_qdot=np.asarray(rows, dtype=np.float64),
            cbf_lower_m2_per_s=np.asarray(lower, dtype=np.float64),
            qdot_lower_rad_per_s=np.full(7, -20.0),
            qdot_upper_rad_per_s=np.full(7, 20.0),
            osc_output_scale=self.scale,
            control_dt_seconds=0.05,
        )

    def test_safe_nominal_action_is_byte_exact_and_skips_solver(self):
        source = np.asarray([0.25, -0.1, 0.0, 0.0, 0.0, 0.0, -1.0])
        result = self.filter(source, [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], [-1.0])
        self.assertTrue(result.valid)
        self.assertEqual(result.reason, "nominal_safe_exact_passthrough")
        self.assertEqual(result.action.tobytes(), source.tobytes())
        self.assertFalse(result.diagnostics["solver_attempted"])

    def test_safe_native_clipped_source_is_still_byte_exact(self):
        source = np.asarray([1.2, -1.1, 0.0, 0.0, 0.0, 0.0, -1.003])
        result = self.filter(source, np.empty((0, 7)), np.empty(0))
        self.assertTrue(result.valid)
        self.assertEqual(result.action.tobytes(), source.tobytes())
        self.assertTrue(result.diagnostics["source_pose_was_clipped_by_native"])
        np.testing.assert_array_equal(
            result.diagnostics["native_clipped_source_pose_action"],
            [1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
        )

    def test_correction_changes_pose_but_preserves_gripper(self):
        source = np.asarray([-0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.75])
        result = self.filter(source, [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], [0.0])
        self.assertTrue(result.valid, result.diagnostics)
        self.assertEqual(result.reason, "solved")
        self.assertGreaterEqual(float(result.safe_qdot_rad_per_s[0]), -5.0e-7)
        self.assertEqual(result.action[6:7].tobytes(), source[6:7].tobytes())
        self.assertTrue(result.diagnostics["material_correction"])

    def test_uncontrollable_constraint_fails_without_stop_command(self):
        source = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
        result = self.filter(source, [[0.0] * 7], [1.0])
        self.assertFalse(result.valid)
        self.assertEqual(result.reason, "cbf_not_controllable_in_osc_pose_subspace")
        self.assertIsNone(result.action)

    def test_measured_uncommanded_joint_motion_cannot_false_passthrough(self):
        source = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
        measured = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.4])
        result = self.filter(
            source,
            [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]],
            [0.0],
            measured_qdot=measured,
        )
        self.assertFalse(result.valid)
        self.assertEqual(
            result.reason,
            "cbf_not_controllable_in_osc_pose_subspace",
        )
        self.assertLess(result.nominal_qdot_rad_per_s[6], 0.0)


if __name__ == "__main__":
    unittest.main()
