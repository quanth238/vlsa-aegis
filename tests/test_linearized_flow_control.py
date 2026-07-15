import importlib
from pathlib import Path
import sys
import unittest
from unittest import mock
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
OPENPI_SRC = ROOT / "openpi" / "src"
MODULE_PATH = OPENPI_SRC / "openpi" / "models_pytorch" / "crfs_linearized_control.py"

try:
    import torch
except ImportError:  # Dependency-free harness retains structural coverage.
    torch = None


def _load_module():
    sys.path.insert(0, str(OPENPI_SRC))
    try:
        return importlib.import_module("openpi.models_pytorch.crfs_linearized_control")
    finally:
        sys.path.pop(0)


class LinearizedFlowControlStructuralTest(unittest.TestCase):
    def test_registered_convex_problem_is_structural_and_dependency_free(self):
        source = MODULE_PATH.read_text(encoding="utf-8")
        self.assertIn("fista_iterations: int = 4096", source)
        self.assertIn("torch.linalg.svdvals(matrix)", source)
        self.assertIn("step_size = 1.0 / torch.square(sigma_max)", source)
        self.assertIn("_project_product_balls", source)
        self.assertIn("device=\"cpu\", dtype=torch.float64", source)
        self.assertIn("infeasibility_certificate: bool = False", source)
        self.assertIn("terminal_overwrite_used: bool = False", source)
        self.assertNotIn("import scipy", source)
        self.assertNotIn("import cvxpy", source)

    def test_registered_jacobian_and_fidelity_surfaces_are_structural(self):
        source = MODULE_PATH.read_text(encoding="utf-8")
        self.assertIn("jacobian.shape != (35, 75)", source)
        self.assertIn("candidate[local_index] / dt", source)
        self.assertIn("(terminal - target) * scale", source)
        self.assertIn("legacy_config.xyz_rms_tolerance", source)
        self.assertIn("legacy_config.full_rms_tolerance", source)
        self.assertIn("_legacy._fidelity_metrics", source)
        self.assertIn("_legacy.validate_schedule_constraints", source)
        self.assertIn("_finite_difference_diagnostics", source)
        self.assertIn("finite_difference_relative_l2_tolerance: float = 0.10", source)


@unittest.skipUnless(torch is not None, "PyTorch is unavailable")
class LinearizedFlowControlRuntimeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _load_module()
        cls.legacy = importlib.import_module(
            "openpi.models_pytorch.crfs_inverse_control"
        )

    def setUp(self):
        self.initial = torch.zeros((1, 5, 7), dtype=torch.float64)
        self.control_mask = self.legacy.first_five_xyz_mask_like(self.initial)
        self.target_mask = self.legacy.first_five_channels_mask_like(self.initial)
        self.scale = torch.ones_like(self.initial)
        self.config = self.legacy.InverseControlConfig()

    @staticmethod
    def zero_field(x_t, _time, _step):
        return torch.zeros_like(x_t)

    def solve(self, target, field=None, *, budget=None, scale=None):
        if budget is None:
            budget = torch.linalg.vector_norm(
                (target - self.initial)[self.control_mask]
            )
        return self.module.solve_linearized_control(
            self.initial,
            target,
            field or self.zero_field,
            control_mask=self.control_mask,
            target_mask=self.target_mask,
            model_to_physical_scale=self.scale if scale is None else scale,
            control_budget=budget,
            legacy_config=self.config,
        )

    def test_zero_field_has_exact_jacobian_and_known_same_budget_solution(self):
        target = self.initial.clone()
        target[0, 0, :3] = torch.tensor([0.04, -0.02, 0.01], dtype=target.dtype)
        budget = torch.linalg.vector_norm(target[self.control_mask])
        result = self.solve(target, budget=budget)

        xyz_rows = self.control_mask[self.target_mask]
        expected_xyz = torch.cat([torch.eye(15, dtype=target.dtype)] * 5, dim=1)
        self.assertEqual(result.status, self.module.LinearizedControlStatus.SOLVED)
        self.assertTrue(torch.equal(result.jacobian[xyz_rows], expected_xyz))
        self.assertEqual(
            torch.count_nonzero(result.jacobian[~xyz_rows]).item(),
            0,
        )
        expected_active = (
            (target - self.initial)[self.control_mask].reshape(1, 15).expand(5, 15)
            / 5.0
        )
        self.assertTrue(
            torch.allclose(
                result.active_increments[:, self.control_mask],
                expected_active,
                atol=1.0e-12,
                rtol=0.0,
            )
        )
        self.assertTrue(torch.allclose(result.rollout.final, target, atol=1.0e-12, rtol=0.0))
        self.assertTrue(result.linear_metrics.feasible)
        self.assertTrue(result.nonlinear_metrics.feasible)
        self.assertTrue(result.finite_difference.passed)
        self.assertEqual(result.selected_candidate_float64.shape, (5, 15))
        self.assertTrue(
            torch.equal(
                result.model_candidate_post_projection,
                result.active_increments,
            )
        )
        expected_pre = torch.zeros_like(result.model_candidate_pre_projection)
        expected_pre[:, self.control_mask] = result.selected_candidate_float64.to(
            dtype=self.initial.dtype
        )
        self.assertTrue(
            torch.equal(result.model_candidate_pre_projection, expected_pre)
        )
        self.assertEqual(result.jacobian_singular_values.shape, (35,))
        self.assertEqual(result.weighted_matrix_singular_values.shape, (35,))
        self.assertGreaterEqual(result.jacobian_effective_rank, 0)
        self.assertLessEqual(result.jacobian_effective_rank, 35)
        self.assertFalse(result.infeasibility_certificate)
        self.assertFalse(result.fista.infeasibility_certificate)

    def test_affine_coupled_jacobian_matches_central_difference(self):
        target = self.initial.clone()
        target[0, 0, 0] = 0.03
        scale = self.scale.clone()
        scale[..., 0] = 2.5
        scale[..., 1] = 0.75

        def coupled_field(x_t, _time, _step):
            velocity = torch.zeros_like(x_t)
            velocity[..., 0] = -0.7 * x_t[..., 0] + 0.4 * x_t[..., 1]
            velocity[..., 1] = 0.2 * x_t[..., 0] - 0.3 * x_t[..., 1]
            return velocity

        result = self.solve(target, coupled_field, scale=scale)
        self.assertTrue(result.finite_difference.passed)
        epsilon = torch.tensor(1.0e-6, dtype=self.initial.dtype)
        compact_columns = (0, 1, 31, 74)
        for column in compact_columns:
            local_step, compact_coordinate = divmod(column, 15)
            plus = torch.zeros((10, *self.initial.shape), dtype=self.initial.dtype)
            minus = torch.zeros_like(plus)
            plus_active = plus[5 + local_step]
            minus_active = minus[5 + local_step]
            plus_active[self.control_mask] = torch.nn.functional.one_hot(
                torch.tensor(compact_coordinate), num_classes=15
            ).to(self.initial.dtype) * epsilon / self.config.dt
            minus_active[self.control_mask] = -plus_active[self.control_mask]
            plus_rollout = self.legacy.replay_velocity_schedule(
                self.initial, coupled_field, plus, config=self.config
            )
            minus_rollout = self.legacy.replay_velocity_schedule(
                self.initial, coupled_field, minus, config=self.config
            )
            finite_difference = (
                ((plus_rollout.final - minus_rollout.final) * scale)[self.target_mask]
                / (2.0 * epsilon)
            )
            self.assertTrue(
                torch.allclose(
                    result.jacobian[:, column],
                    finite_difference,
                    atol=2.0e-9,
                    rtol=2.0e-9,
                )
            )

    def test_candidate_is_deterministic_and_obeys_each_group_ball(self):
        target = self.initial.clone()
        target[0, 0, 0] = 0.08

        def coupled_field(x_t, _time, _step):
            velocity = torch.zeros_like(x_t)
            velocity[..., 0] = 0.8 * x_t[..., 1]
            velocity[..., 1] = -0.5 * x_t[..., 0]
            return velocity

        budget = torch.tensor(0.06, dtype=self.initial.dtype)
        first = self.solve(target, coupled_field, budget=budget)
        second = self.solve(target, coupled_field, budget=budget)
        self.assertTrue(torch.equal(first.active_increments, second.active_increments))
        self.assertTrue(torch.equal(first.jacobian, second.jacobian))
        self.assertEqual(first.fista.selected_iteration, second.fista.selected_iteration)
        compact = first.active_increments[:, self.control_mask]
        group_norms = torch.linalg.vector_norm(compact, dim=1)
        self.assertTrue(torch.all(group_norms <= budget / 5.0 + 1.0e-15))
        self.assertLessEqual(first.path_length.item(), budget.item() + 1.0e-15)
        self.assertEqual(torch.count_nonzero(first.schedule[:5]).item(), 0)

    def test_linear_prediction_uses_exact_float32_executed_increment(self):
        initial = torch.zeros((1, 10, 32), dtype=torch.float32)
        target = initial.clone()
        # The registered solution distributes this value into increments whose
        # float32 c -> c/dt -> dt*(c/dt) round trip is intentionally non-exact.
        target[0, 0, 0] = torch.tensor(4.73540485, dtype=torch.float32)
        control_mask = self.legacy.first_five_xyz_mask_like(initial)
        target_mask = self.legacy.first_five_channels_mask_like(initial)
        result = self.module.solve_linearized_control(
            initial,
            target,
            self.zero_field,
            control_mask=control_mask,
            target_mask=target_mask,
            model_to_physical_scale=torch.ones_like(initial),
            control_budget=torch.linalg.vector_norm(target[control_mask]),
            legacy_config=self.config,
        )

        executed = result.increments[self.config.intervention_step :]
        self.assertFalse(torch.equal(result.active_increments, executed))
        expected = (
            result.baseline_target_error.to(torch.float64)
            + result.jacobian.to(torch.float64)
            @ executed[:, control_mask].reshape(-1).to(torch.float64)
        ).to(torch.float32)
        pre_transport_prediction = (
            result.baseline_target_error.to(torch.float64)
            + result.jacobian.to(torch.float64)
            @ result.active_increments[:, control_mask]
            .reshape(-1)
            .to(torch.float64)
        ).to(torch.float32)
        self.assertTrue(torch.equal(result.linear_predicted_target_error, expected))
        self.assertFalse(torch.equal(expected, pre_transport_prediction))

    def test_nonlinear_replay_exposes_linearization_mismatch(self):
        target = self.initial.clone()
        target[0, 0, 0] = 0.1

        def nonlinear_field(x_t, _time, _step):
            velocity = torch.zeros_like(x_t)
            velocity[..., 0] = -20.0 * torch.square(x_t[..., 0])
            return velocity

        result = self.solve(target, nonlinear_field)
        self.assertLess(
            torch.max(torch.abs(result.linear_predicted_target_error)).item(),
            1.0e-10,
        )
        self.assertGreater(
            torch.max(torch.abs(result.linearization_error)).item(),
            1.0e-3,
        )
        self.assertFalse(
            torch.equal(
                result.linear_predicted_target_error,
                result.nonlinear_target_error,
            )
        )

    def test_physical_scaling_and_all_four_metrics_are_retained(self):
        target = self.initial.clone()
        target[0, 0, 0] = 0.02
        scale = self.scale.clone()
        scale[0, 0, 0] = 10.0
        budget = torch.tensor(0.01, dtype=self.initial.dtype)
        result = self.solve(target, budget=budget, scale=scale)
        expected_error = torch.tensor(-0.1, dtype=self.initial.dtype)
        self.assertTrue(torch.allclose(result.nonlinear_fidelity_error[0, 0, 0], expected_error))
        self.assertTrue(
            torch.allclose(
                result.nonlinear_metrics.xyz_max_abs,
                torch.tensor(0.1, dtype=self.initial.dtype),
            )
        )
        self.assertTrue(
            torch.allclose(
                result.nonlinear_metrics.xyz_rms,
                torch.sqrt(torch.tensor(0.01 / 15.0, dtype=self.initial.dtype)),
            )
        )
        self.assertTrue(
            torch.allclose(
                result.nonlinear_metrics.full_max_abs,
                torch.tensor(0.1, dtype=self.initial.dtype),
            )
        )
        self.assertTrue(
            torch.allclose(
                result.nonlinear_metrics.full_rms,
                torch.sqrt(torch.tensor(0.01 / 35.0, dtype=self.initial.dtype)),
            )
        )
        self.assertFalse(result.nonlinear_metrics.feasible)

    def test_zero_matrix_solver_is_explicitly_not_an_infeasibility_certificate(self):
        jacobian = torch.zeros((35, 75), dtype=torch.float64)
        baseline = torch.ones(35, dtype=torch.float64)
        weights = torch.ones(35, dtype=torch.float64)
        candidate, singular_values, diagnostics = self.module._fista_product_balls(
            jacobian,
            baseline,
            weights,
            torch.tensor(1.0, dtype=torch.float64),
            self.module.LinearizedControlConfig(),
        )
        self.assertEqual(torch.count_nonzero(candidate).item(), 0)
        self.assertEqual(torch.count_nonzero(singular_values).item(), 0)
        self.assertEqual(diagnostics.updates, 0)
        self.assertFalse(diagnostics.infeasibility_certificate)
        self.assertFalse(diagnostics.optimality_certificate)

    def test_nonfinite_convex_input_fails_instead_of_returning_a_stale_candidate(self):
        jacobian = torch.zeros((35, 75), dtype=torch.float64)
        jacobian[0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "nonfinite weighted matrix"):
            self.module._fista_product_balls(
                jacobian,
                torch.ones(35, dtype=torch.float64),
                torch.ones(35, dtype=torch.float64),
                torch.tensor(1.0, dtype=torch.float64),
                self.module.LinearizedControlConfig(),
            )

    def test_failed_registered_finite_difference_check_fails_before_candidate(self):
        target = self.initial.clone()
        target[0, 0, 0] = 0.01
        invalid = SimpleNamespace(passed=False)
        with mock.patch.object(
            self.module,
            "_finite_difference_diagnostics",
            return_value=invalid,
        ):
            with self.assertRaisesRegex(ValueError, "finite-difference validation"):
                self.solve(target)

    def test_invalid_masks_and_nonfinite_field_fail_closed(self):
        target = self.initial.clone()
        target[0, 0, 0] = 0.01
        wrong_mask = self.control_mask.clone()
        wrong_mask[0, 0, 2] = False
        with self.assertRaisesRegex(ValueError, "exactly first-five XYZ"):
            self.module.solve_linearized_control(
                self.initial,
                target,
                self.zero_field,
                control_mask=wrong_mask,
                target_mask=self.target_mask,
                model_to_physical_scale=self.scale,
                control_budget=0.01,
                legacy_config=self.config,
            )

        def nonfinite_field(x_t, _time, step):
            value = torch.zeros_like(x_t)
            if step == 7:
                value[0, 0, 0] = float("nan")
            return value

        with self.assertRaisesRegex(RuntimeError, "nonfinite"):
            self.solve(target, nonfinite_field)


if __name__ == "__main__":
    unittest.main()
