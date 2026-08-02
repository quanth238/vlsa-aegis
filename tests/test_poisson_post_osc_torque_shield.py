from __future__ import annotations

import importlib.util
import unittest


DEPENDENCIES_PRESENT = all(
    importlib.util.find_spec(name) is not None
    for name in ("numpy", "scipy", "osqp")
)


@unittest.skipUnless(DEPENDENCIES_PRESENT, "Poisson numerical dependencies unavailable")
class SampledDataPostOscTorqueShieldTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import numpy as np

        from main.poisson_fullbody.post_osc_torque_shield import (
            SampledDataPostOscTorqueShield,
        )

        cls.np = np
        cls.shield = SampledDataPostOscTorqueShield()

    def solve(self, **overrides):
        np = self.np
        inputs = {
            "nominal_torque": np.array(
                [0.25, -0.5, 0.75, -1.0, 1.25, -1.5, 1.75],
                dtype=np.float64,
            ),
            "current_qvel": np.zeros(7),
            "nominal_next_qvel": np.zeros(7),
            "h": np.array([0.1]),
            "joint_gradient_rows": np.array([[1.0, 0, 0, 0, 0, 0, 0]]),
            "dt_seconds": 0.002,
            "torque_to_next_qvel_sensitivity": np.eye(7),
            "torque_lower": np.full(7, -4.0),
            "torque_upper": np.full(7, 4.0),
            "alpha": 1.0,
            "margin": 0.0,
        }
        inputs.update(overrides)
        return self.shield.solve(**inputs)

    def test_safe_nominal_is_byte_identical_and_skips_solver(self) -> None:
        np = self.np
        from main.poisson_fullbody.post_osc_torque_shield import TorqueShieldStatus

        nominal = np.array(
            [0.25, -0.5, 0.75, -1.0, 1.25, -1.5, 1.75], dtype=np.float64
        )
        before = nominal.tobytes(order="C")
        result = self.solve(nominal_torque=nominal)
        self.assertTrue(result.valid, result)
        self.assertEqual(result.status, TorqueShieldStatus.NOMINAL_SAFE)
        self.assertFalse(result.diagnostics["solver_attempted"])
        self.assertEqual(result.torque_command.tobytes(order="C"), before)
        self.assertEqual(nominal.tobytes(order="C"), before)
        np.testing.assert_array_equal(result.delta_torque, np.zeros(7))

    def test_one_constraint_matches_analytic_minimum_torque_correction(self) -> None:
        np = self.np
        from main.poisson_fullbody.post_osc_torque_shield import TorqueShieldStatus

        result = self.solve(
            nominal_torque=np.zeros(7),
            nominal_next_qvel=np.array([-0.2, 0, 0, 0, 0, 0, 0]),
            h=np.array([0.0]),
        )
        self.assertTrue(result.valid, result)
        self.assertEqual(result.status, TorqueShieldStatus.SOLVED)
        expected = np.zeros(7)
        expected[0] = 0.2
        np.testing.assert_allclose(result.delta_torque, expected, atol=2e-6)
        self.assertTrue(result.diagnostics["postcheck"]["feasible"])
        self.assertGreaterEqual(
            result.diagnostics["postcheck"]["minimum_raw_cbf_residual"],
            -1e-7,
        )

    def test_sensitivity_already_contains_dt_and_is_not_scaled_twice(self) -> None:
        np = self.np
        result = self.solve(
            nominal_torque=np.zeros(7),
            nominal_next_qvel=np.array([-0.2, 0, 0, 0, 0, 0, 0]),
            h=np.array([0.0]),
            dt_seconds=0.01,
            torque_to_next_qvel_sensitivity=2.0 * np.eye(7),
        )
        self.assertTrue(result.valid, result)
        self.assertAlmostEqual(float(result.delta_torque[0]), 0.1, places=6)
        self.assertTrue(
            result.diagnostics[
                "sensitivity_already_includes_dt_and_contact_effects"
            ]
        )

    def test_actuator_bound_can_make_hard_qp_infeasible(self) -> None:
        np = self.np
        from main.poisson_fullbody.post_osc_torque_shield import TorqueShieldStatus

        result = self.solve(
            nominal_torque=np.zeros(7),
            nominal_next_qvel=np.array([-1.0, 0, 0, 0, 0, 0, 0]),
            h=np.array([0.0]),
            torque_lower=np.full(7, -0.25),
            torque_upper=np.full(7, 0.25),
        )
        self.assertFalse(result.valid)
        self.assertEqual(result.status, TorqueShieldStatus.QP_INFEASIBLE)
        self.assertIsNone(result.torque_command)
        self.assertIsNone(result.delta_torque)

    def test_zero_torque_gain_with_violated_constraint_is_typed_failure(self) -> None:
        np = self.np
        from main.poisson_fullbody.post_osc_torque_shield import TorqueShieldStatus

        result = self.solve(
            nominal_torque=np.zeros(7),
            nominal_next_qvel=np.zeros(7),
            h=np.array([0.0]),
            joint_gradient_rows=np.zeros((1, 7)),
            margin=0.1,
        )
        self.assertFalse(result.valid)
        self.assertEqual(result.status, TorqueShieldStatus.UNCONTROLLABLE_CONSTRAINT)
        self.assertEqual(result.diagnostics["constraint_indexes"], [0])
        self.assertFalse(result.diagnostics["solver_attempted"])

    def test_gain_below_registered_sensitivity_error_is_uncontrollable(self) -> None:
        np = self.np
        from main.poisson_fullbody.post_osc_torque_shield import TorqueShieldStatus

        sensitivity = np.eye(7)
        sensitivity[0, 0] = 1e-6
        result = self.solve(
            nominal_torque=np.zeros(7),
            nominal_next_qvel=np.array([-0.1, 0, 0, 0, 0, 0, 0]),
            h=np.array([0.0]),
            torque_to_next_qvel_sensitivity=sensitivity,
            sensitivity_max_absolute_error=2e-6,
        )
        self.assertFalse(result.valid)
        self.assertEqual(result.status, TorqueShieldStatus.UNCONTROLLABLE_CONSTRAINT)
        self.assertEqual(result.diagnostics["constraint_indexes"], [0])

    def test_rank_deficient_irrelevant_mode_does_not_reject_safe_nominal(self) -> None:
        np = self.np
        from main.poisson_fullbody.post_osc_torque_shield import TorqueShieldStatus

        sensitivity = np.eye(7)
        sensitivity[-1, -1] = 0.0
        result = self.solve(torque_to_next_qvel_sensitivity=sensitivity)
        self.assertTrue(result.valid, result)
        self.assertEqual(result.status, TorqueShieldStatus.NOMINAL_SAFE)
        self.assertTrue(
            result.diagnostics["global_condition_limit_exceeded"]
        )
        self.assertTrue(result.diagnostics["global_condition_is_advisory_only"])

    def test_rank_deficient_irrelevant_mode_allows_controllable_correction(self) -> None:
        np = self.np
        from main.poisson_fullbody.post_osc_torque_shield import TorqueShieldStatus

        sensitivity = np.eye(7)
        sensitivity[-1, -1] = 0.0
        result = self.solve(
            nominal_torque=np.zeros(7),
            nominal_next_qvel=np.array([-0.2, 0, 0, 0, 0, 0, 0]),
            h=np.array([0.0]),
            torque_to_next_qvel_sensitivity=sensitivity,
        )
        self.assertTrue(result.valid, result)
        self.assertEqual(result.status, TorqueShieldStatus.SOLVED)
        self.assertAlmostEqual(float(result.delta_torque[0]), 0.2, places=6)

    def test_nonfinite_input_is_typed_failure(self) -> None:
        np = self.np
        from main.poisson_fullbody.post_osc_torque_shield import TorqueShieldStatus

        h = np.array([np.nan])
        result = self.solve(h=h)
        self.assertFalse(result.valid)
        self.assertEqual(result.status, TorqueShieldStatus.INVALID_INPUT)
        self.assertIn("finite", result.diagnostics["error"])

    def test_independent_postcheck_rejects_wrong_sign_candidate(self) -> None:
        np = self.np
        from main.poisson_fullbody.post_osc_torque_shield import (
            evaluate_torque_shield_residuals,
        )

        residual = evaluate_torque_shield_residuals(
            nominal_torque=np.zeros(7),
            candidate_torque=np.array([-0.2, 0, 0, 0, 0, 0, 0]),
            nominal_next_qvel=np.array([-0.1, 0, 0, 0, 0, 0, 0]),
            h=np.array([0.0]),
            joint_gradient_rows=np.array([[1.0, 0, 0, 0, 0, 0, 0]]),
            torque_to_next_qvel_sensitivity=np.eye(7),
            torque_lower=np.full(7, -1.0),
            torque_upper=np.full(7, 1.0),
            alpha=1.0,
            margin=0.0,
            normalized_cbf_tolerance=1e-8,
            torque_bound_tolerance=1e-8,
        )
        self.assertFalse(residual["feasible"])
        self.assertAlmostEqual(residual["minimum_raw_cbf_residual"], -0.3)


if __name__ == "__main__":
    unittest.main()
