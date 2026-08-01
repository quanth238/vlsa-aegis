from __future__ import annotations

import importlib.util
import unittest


DEPENDENCIES_PRESENT = all(
    importlib.util.find_spec(name) is not None
    for name in ("numpy", "scipy", "osqp")
)


@unittest.skipUnless(DEPENDENCIES_PRESENT, "Poisson numerical dependencies unavailable")
class HardCbfQpTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import numpy as np

        from main.poisson_fullbody.cbf_qp import HardCbfQp

        cls.np = np
        cls.filter = HardCbfQp()

    def test_nominal_feasible_command_is_unchanged(self) -> None:
        np = self.np
        nominal = np.array([0.1, -0.2, 0.05, 0.0, 0.3, -0.1, 0.2])
        rows = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        result = self.filter.solve_rows(
            nominal,
            rows,
            [-0.5],
            np.full(7, -0.5),
            np.full(7, 0.5),
        )
        self.assertTrue(result.valid, result)
        np.testing.assert_allclose(result.qdot_safe, nominal, atol=2e-7)
        self.assertTrue(result.diagnostics["nominal_feasible"])

    def test_one_halfspace_matches_analytic_projection(self) -> None:
        np = self.np
        nominal = np.array([-0.4, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0])
        row = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        bound = 0.2
        expected = nominal + ((bound - row @ nominal) / (row @ row)) * row
        result = self.filter.solve_rows(
            nominal,
            row[None, :],
            [bound],
            np.full(7, -1.0),
            np.full(7, 1.0),
        )
        self.assertTrue(result.valid, result)
        np.testing.assert_allclose(result.qdot_safe, expected, atol=2e-6)
        self.assertGreaterEqual(
            result.diagnostics["minimum_normalized_cbf_residual"], -5e-7
        )

    def test_constraint_scaling_and_permutation_do_not_change_solution(self) -> None:
        np = self.np
        nominal = np.array([-0.4, -0.3, 0.1, 0.0, 0.0, 0.0, 0.0])
        rows = np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
        bounds = np.array([-0.1, 0.05])
        kwargs = (np.full(7, -0.5), np.full(7, 0.5))
        first = self.filter.solve_rows(nominal, rows, bounds, *kwargs)
        second = self.filter.solve_rows(
            nominal, rows[::-1] * np.array([[100.0], [0.01]]),
            bounds[::-1] * np.array([100.0, 0.01]), *kwargs
        )
        self.assertTrue(first.valid, first)
        self.assertTrue(second.valid, second)
        np.testing.assert_allclose(first.qdot_safe, second.qdot_safe, atol=2e-6)

    def test_raw_and_normalized_residual_units_are_not_conflated(self) -> None:
        np = self.np
        nominal = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        row = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        first = self.filter.solve_rows(
            nominal, row, [-0.2], np.full(7, -0.5), np.full(7, 0.5)
        )
        second = self.filter.solve_rows(
            nominal,
            row * 100.0,
            [-20.0],
            np.full(7, -0.5),
            np.full(7, 0.5),
        )
        self.assertTrue(first.valid, first)
        self.assertTrue(second.valid, second)
        self.assertAlmostEqual(
            first.diagnostics["minimum_normalized_cbf_residual"],
            second.diagnostics["minimum_normalized_cbf_residual"],
            places=9,
        )
        self.assertAlmostEqual(
            second.diagnostics["minimum_raw_cbf_residual_m2_per_s"],
            100.0 * first.diagnostics["minimum_raw_cbf_residual_m2_per_s"],
            places=8,
        )
        self.assertAlmostEqual(
            second.diagnostics["minimum_nonzero_row_scale_m2_per_rad"],
            100.0 * first.diagnostics["minimum_nonzero_row_scale_m2_per_rad"],
        )

    def test_randomized_reference_active_bounds_infeasible_and_scaled_cases(self) -> None:
        if importlib.util.find_spec("cvxpy") is None:
            self.skipTest("CVXPY reference dependency unavailable")
        np = self.np
        from main.poisson_fullbody.cbf_qp import solve_reference_cvxpy

        rng = np.random.default_rng(20260731)
        velocity_lower = np.full(7, -0.5)
        velocity_upper = np.full(7, 0.5)
        for _ in range(12):
            feasible_anchor = rng.uniform(-0.2, 0.2, size=7)
            rows = rng.normal(size=(5, 7))
            row_lower = rows @ feasible_anchor - rng.uniform(0.01, 0.15, size=5)
            nominal = rng.uniform(-0.75, 0.75, size=7)
            weights = rng.uniform(0.5, 2.0, size=7)
            production = self.filter.solve_rows(
                nominal,
                rows,
                row_lower,
                velocity_lower,
                velocity_upper,
                weight_diagonal=weights,
            )
            reference = solve_reference_cvxpy(
                nominal,
                rows,
                row_lower,
                velocity_lower,
                velocity_upper,
                weight_diagonal=weights,
            )
            self.assertTrue(production.valid, production)
            self.assertTrue(np.all(production.qdot_safe >= velocity_lower))
            self.assertTrue(np.all(production.qdot_safe <= velocity_upper))
            np.testing.assert_allclose(
                production.qdot_safe, reference, rtol=2e-5, atol=2e-5
            )

            scales = np.power(10.0, rng.uniform(-12.0, 12.0, size=5))
            scaled = self.filter.solve_rows(
                nominal,
                rows * scales[:, None],
                row_lower * scales,
                velocity_lower,
                velocity_upper,
                weight_diagonal=weights,
            )
            self.assertTrue(scaled.valid, scaled)
            np.testing.assert_allclose(
                scaled.qdot_safe,
                production.qdot_safe,
                rtol=2e-5,
                atol=2e-5,
            )

        # No CBF row is needed to make a velocity bound active.
        bound_nominal = np.zeros(7)
        bound_nominal[0] = 0.9
        bounded = self.filter.solve_rows(
            bound_nominal,
            np.empty((0, 7)),
            np.empty(0),
            velocity_lower,
            np.asarray([0.2] + [0.5] * 6),
        )
        self.assertTrue(bounded.valid, bounded)
        self.assertGreaterEqual(float(bounded.qdot_safe[0]), velocity_lower[0])
        self.assertLessEqual(float(bounded.qdot_safe[0]), 0.2)
        self.assertAlmostEqual(float(bounded.qdot_safe[0]), 0.2, places=6)

        infeasible_rows = np.asarray([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        infeasible = self.filter.solve_rows(
            np.zeros(7),
            infeasible_rows,
            [0.6],
            velocity_lower,
            velocity_upper,
        )
        self.assertFalse(infeasible.valid)
        with self.assertRaises(RuntimeError):
            solve_reference_cvxpy(
                np.zeros(7),
                infeasible_rows,
                [0.6],
                velocity_lower,
                velocity_upper,
            )

    def test_positive_bound_on_zero_row_fails_closed(self) -> None:
        np = self.np
        result = self.filter.solve_rows(
            np.zeros(7),
            np.zeros((1, 7)),
            [0.1],
            np.full(7, -0.5),
            np.full(7, 0.5),
        )
        self.assertFalse(result.valid)
        self.assertEqual(result.reason, "uncontrollable_cbf_constraint")
        self.assertIsNone(result.qdot_safe)

    def test_small_raw_positive_zero_row_uses_normalized_tolerance(self) -> None:
        np = self.np
        result = self.filter.solve_rows(
            np.zeros(7),
            np.zeros((1, 7)),
            [1.0e-8],
            np.full(7, -0.5),
            np.full(7, 0.5),
        )
        self.assertFalse(result.valid)
        self.assertEqual(result.reason, "uncontrollable_cbf_constraint")
        self.assertEqual(
            result.diagnostics["positive_lower_bounds"], [1.0e-8]
        )

    def test_tiny_scaled_infeasible_halfspace_is_not_discarded(self) -> None:
        np = self.np
        result = self.filter.solve_rows(
            np.zeros(7),
            np.asarray([[1.0e-18, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
            [1.0e-18],
            np.full(7, -0.5),
            np.full(7, 0.5),
        )
        self.assertFalse(result.valid)
        self.assertIn(result.reason, {"qp_not_solved", "qp_postcheck_failed"})

    def test_overflowing_normalized_constraint_fails_closed(self) -> None:
        np = self.np
        result = self.filter.solve_rows(
            np.zeros(7),
            np.asarray([[np.finfo(float).tiny, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
            [np.finfo(float).max],
            np.full(7, -0.5),
            np.full(7, 0.5),
        )
        self.assertFalse(result.valid)
        self.assertEqual(result.reason, "invalid_qp_scaling")

    def test_overflowing_linear_objective_fails_closed(self) -> None:
        np = self.np
        nominal = np.zeros(7)
        nominal[0] = 2.0
        weights = np.ones(7)
        weights[0] = np.finfo(float).max
        result = self.filter.solve_rows(
            nominal,
            np.empty((0, 7)),
            np.empty(0),
            np.full(7, -0.5),
            np.full(7, 0.5),
            weight_diagonal=weights,
        )
        self.assertFalse(result.valid)
        self.assertEqual(result.reason, "invalid_qp_objective")
        self.assertIsNone(result.qdot_safe)

    def test_static_field_rows_have_paper_sign(self) -> None:
        np = self.np
        from main.poisson_fullbody.cbf_qp import static_cbf_rows

        h = np.array([0.1])
        gradient = np.array([[1.0, 2.0, 3.0]])
        jacobian = np.zeros((1, 3, 7))
        jacobian[0, :, :3] = np.eye(3)
        rows, lower = static_cbf_rows(h, gradient, jacobian, alpha=10.0)
        np.testing.assert_allclose(rows[0, :3], [1.0, 2.0, 3.0])
        np.testing.assert_allclose(rows[0, 3:], 0.0)
        self.assertEqual(float(lower[0]), -1.0)

    def test_negative_field_value_is_not_treated_as_an_ordinary_query(self) -> None:
        np = self.np
        result = self.filter.solve_from_field(
            np.zeros(7),
            [-1e-3],
            np.array([[1.0, 0.0, 0.0]]),
            np.zeros((1, 3, 7)),
            np.full(7, -0.5),
            np.full(7, 0.5),
            alpha=10.0,
        )
        self.assertFalse(result.valid)
        self.assertEqual(result.reason, "unsafe_or_invalid_field_start")
        self.assertIsNone(result.qdot_safe)

    def test_full_body_field_solve_rejects_zero_samples(self) -> None:
        np = self.np
        result = self.filter.solve_from_field(
            np.zeros(7),
            [],
            np.zeros((0, 3)),
            np.zeros((0, 3, 7)),
            np.full(7, -0.5),
            np.full(7, 0.5),
            alpha=10.0,
        )
        self.assertFalse(result.valid)
        self.assertEqual(result.reason, "no_cbf_samples")

    def test_safe_start_tolerance_must_be_finite_and_nonnegative(self) -> None:
        np = self.np
        kwargs = dict(
            qdot_nominal=np.zeros(7),
            h=[-1.0e-3],
            gradients_world=np.array([[1.0, 0.0, 0.0]]),
            point_jacobians=np.zeros((1, 3, 7)),
            velocity_lower=np.full(7, -0.5),
            velocity_upper=np.full(7, 0.5),
            alpha=10.0,
        )
        for invalid in (float("nan"), float("inf"), -1.0, True):
            with self.assertRaisesRegex(ValueError, "finite and nonnegative"):
                self.filter.solve_from_field(
                    **kwargs, safe_start_tolerance=invalid
                )

    def test_qp_max_iterations_must_be_an_integer(self) -> None:
        from main.poisson_fullbody.cbf_qp import HardCbfQp

        for invalid in (1.5, True, 0):
            with self.assertRaisesRegex(ValueError, "positive integer"):
                HardCbfQp(max_iter=invalid)

    def test_joint_bounds_include_one_step_position_constraint(self) -> None:
        np = self.np
        from main.poisson_fullbody.cbf_qp import joint_velocity_bounds

        q = np.zeros(7)
        q[0] = 0.99
        q_min = np.full(7, -1.0)
        q_max = np.full(7, 1.0)
        lower, upper = joint_velocity_bounds(
            q,
            q_min,
            q_max,
            np.full(7, -2.0),
            np.full(7, 2.0),
            alpha_joint=100.0,
            control_dt_seconds=0.01,
            position_margin_rad=0.005,
        )
        # Continuous CBF permits +1 rad/s and the physical bound permits +2;
        # the +0.5 result is uniquely the 10 ms one-step position constraint.
        self.assertAlmostEqual(float(upper[0]), 0.5)
        self.assertTrue(np.all(lower >= -2.0))
        self.assertTrue(np.all(upper <= 2.0))

    def test_state_outside_joint_margin_is_rejected(self) -> None:
        np = self.np
        from main.poisson_fullbody.cbf_qp import joint_velocity_bounds

        q = np.zeros(7)
        q[0] = 0.999
        with self.assertRaisesRegex(ValueError, "outside"):
            joint_velocity_bounds(
                q,
                np.full(7, -1.0),
                np.full(7, 1.0),
                np.full(7, -0.5),
                np.full(7, 0.5),
                alpha_joint=10.0,
                control_dt_seconds=0.01,
                position_margin_rad=0.005,
            )


if __name__ == "__main__":
    unittest.main()
