import importlib.util
import dataclasses
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT / "openpi" / "src" / "openpi" / "models_pytorch" / "crfs_inverse_control.py"
)

try:
    import torch
except ImportError:  # Dependency-free harness runs retain structural coverage.
    torch = None


def _load_module():
    spec = importlib.util.spec_from_file_location("crfs_inverse_control_tested", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class InverseFlowControlStructuralTest(unittest.TestCase):
    def test_registered_recurrence_and_no_terminal_overwrite_are_structural(self):
        source = MODULE_PATH.read_text(encoding="utf-8")
        self.assertIn("x_next = x_t + dt_tensor * total_velocity", source)
        self.assertIn("torch.autograd.grad(objective, candidate", source)
        self.assertIn("class _ExactControlAdd(torch.autograd.Function)", source)
        self.assertIn("with torch.enable_grad():", source)
        self.assertIn("time += dt_tensor", source)
        self.assertNotIn("x_t = target", source)
        self.assertNotIn("final = target", source)
        self.assertIn("terminal_overwrite_used: bool = False", source)
        self.assertIn("optimality_certificate: bool = False", source)
        self.assertIn("infeasibility_certificate: bool = False", source)


@unittest.skipUnless(torch is not None, "PyTorch is unavailable")
class InverseFlowControlRuntimeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _load_module()

    def setUp(self):
        # Match the pi0.5 LIBERO normalized action tensor surface.
        self.initial = torch.zeros((1, 10, 32), dtype=torch.float64)
        self.control_mask = self.module.first_five_xyz_mask_like(self.initial)
        self.target_mask = self.module.first_five_channels_mask_like(self.initial)

    @staticmethod
    def zero_field(x_t, _time, _step):
        return torch.zeros_like(x_t)

    def tight_config(self, **overrides):
        values = {
            "max_iterations": 64,
            "learning_rate": 0.02,
            "xyz_max_abs_tolerance": 1.0e-7,
            "xyz_rms_tolerance": 1.0e-7,
            "full_max_abs_tolerance": 1.0e-7,
            "full_rms_tolerance": 1.0e-7,
        }
        values.update(overrides)
        return self.module.InverseControlConfig(**values)

    def test_zero_field_reaches_target_with_exact_mask_and_budget(self):
        target = self.initial.clone()
        target[0, 0, :3] = torch.tensor([0.04, -0.02, 0.01], dtype=target.dtype)
        result = self.module.solve_inverse_control(
            self.initial,
            target,
            self.zero_field,
            control_mask=self.control_mask,
            target_mask=self.target_mask,
            model_to_physical_scale=torch.ones_like(self.initial),
            config=self.tight_config(),
        )
        self.assertEqual(result.status, self.module.InverseControlStatus.CONVERGED)
        self.assertTrue(result.converged)
        self.assertEqual(result.iterations, 64)
        self.assertTrue(torch.equal(result.schedule[:5], torch.zeros_like(result.schedule[:5])))
        outside = ~self.control_mask.unsqueeze(0).expand_as(result.schedule)
        self.assertEqual(torch.count_nonzero(result.schedule[outside]).item(), 0)
        self.assertLessEqual(result.path_length.item(), result.budget.item() + 1.0e-12)
        self.assertTrue(
            torch.all(result.per_step_norms <= result.budget / 5 + 1.0e-12)
        )
        self.assertTrue(torch.allclose(result.rollout.final, target, atol=1.0e-12, rtol=0.0))
        self.assertFalse(result.terminal_overwrite_used)

    def test_nonlinear_transport_beats_constant_schedule_and_is_deterministic(self):
        target = self.initial.clone()
        target[0, 0, 0] = 0.1

        # With dt=-0.1 this field amplifies earlier increments. Repeating the
        # final-action delta uniformly therefore overshoots, while a schedule
        # found through the field can remain inside the same path budget.
        def amplifying_field(x_t, _time, _step):
            return -2.0 * x_t

        config = self.tight_config(
            max_iterations=128,
            learning_rate=0.02,
            xyz_max_abs_tolerance=1.0e-5,
            xyz_rms_tolerance=1.0e-5,
            full_max_abs_tolerance=1.0e-5,
            full_rms_tolerance=1.0e-5,
        )
        delta = target - self.initial
        constant_schedule = torch.zeros(
            (10, *self.initial.shape), dtype=self.initial.dtype
        )
        constant_schedule[5:] = (delta / 5) / config.dt
        constant = self.module.replay_velocity_schedule(
            self.initial, amplifying_field, constant_schedule, config=config
        )
        self.assertGreater(
            torch.max(torch.abs(constant.final - target)).item(),
            config.xyz_max_abs_tolerance,
        )

        first = self.module.solve_inverse_control(
            self.initial,
            target,
            amplifying_field,
            control_mask=self.control_mask,
            target_mask=self.target_mask,
            model_to_physical_scale=torch.ones_like(self.initial),
            config=config,
        )
        second = self.module.solve_inverse_control(
            self.initial,
            target,
            amplifying_field,
            control_mask=self.control_mask,
            target_mask=self.target_mask,
            model_to_physical_scale=torch.ones_like(self.initial),
            config=config,
        )
        self.assertEqual(first.status, self.module.InverseControlStatus.CONVERGED)
        self.assertTrue(first.metrics.feasible)
        self.assertLess(first.path_length.item(), first.budget.item())
        self.assertTrue(torch.equal(first.schedule, second.schedule))
        self.assertTrue(torch.equal(first.rollout.final, second.rollout.final))
        self.assertEqual(first.iterations, second.iterations)

    def test_outer_no_grad_can_discover_an_initially_zero_coupled_control(self):
        target = self.initial.clone()
        target[0, 0, 0] = 0.1
        coupling = torch.nn.Parameter(torch.tensor(-50.0, dtype=self.initial.dtype))

        # Direct X controls are strongly attenuated. A transient Y control,
        # whose initial value is exactly zero, is needed through cross-axis
        # coupling to reach X and return Y to its target.
        def coupled_field(x_t, _time, _step):
            velocity = torch.zeros_like(x_t)
            velocity[..., 0] = 5.0 * x_t[..., 0] + coupling * x_t[..., 1]
            return velocity

        config = self.tight_config(
            max_iterations=128,
            learning_rate=0.02,
            xyz_max_abs_tolerance=1.0e-4,
            xyz_rms_tolerance=1.0e-4,
            full_max_abs_tolerance=1.0e-4,
            full_rms_tolerance=1.0e-4,
        )
        with torch.no_grad():
            result = self.module.solve_inverse_control(
                self.initial,
                target,
                coupled_field,
                control_mask=self.control_mask,
                target_mask=self.target_mask,
                model_to_physical_scale=torch.ones_like(self.initial),
                config=config,
            )
        self.assertEqual(result.status, self.module.InverseControlStatus.CONVERGED)
        self.assertGreater(torch.max(torch.abs(result.schedule[5:, ..., 1])).item(), 0.0)
        self.assertLessEqual(result.path_length.item(), result.budget.item() + 1.0e-12)
        self.assertIsNone(coupling.grad)

    def test_replay_matches_hand_euler_and_zero_schedule_preserves_signed_zero(self):
        initial = self.initial.clone()
        initial[0, 0, 0] = -0.0
        schedule = torch.zeros((10, *initial.shape), dtype=initial.dtype)

        def signed_zero_field(x_t, _time, step):
            velocity = torch.zeros_like(x_t)
            velocity[0, 0, 0] = -0.0 if step % 2 == 0 else 0.0
            return velocity

        rollout = self.module.replay_velocity_schedule(
            initial, signed_zero_field, schedule, config=self.tight_config()
        )
        x_t = initial
        dt = torch.tensor(-0.1, dtype=initial.dtype)
        for step in range(10):
            time = torch.tensor(1.0 - 0.1 * step, dtype=initial.dtype)
            velocity = signed_zero_field(x_t, time, step)
            x_t = x_t + dt * velocity
            self.assertTrue(torch.equal(rollout.states[step + 1], x_t))
            self.assertTrue(
                torch.equal(
                    torch.signbit(rollout.base_velocities[step]),
                    torch.signbit(rollout.total_velocities[step]),
                )
            )

    def test_times_match_iterative_float32_sampler_not_closed_form(self):
        initial = torch.zeros((1, 5, 7), dtype=torch.float32)
        schedule = torch.zeros((10, *initial.shape), dtype=initial.dtype)
        rollout = self.module.replay_velocity_schedule(
            initial, self.zero_field, schedule, config=self.module.InverseControlConfig()
        )
        dt = torch.tensor(-0.1, dtype=torch.float32)
        time = torch.tensor(1.0, dtype=torch.float32)
        expected = []
        for _ in range(10):
            expected.append(time.clone())
            time += dt
        expected = torch.stack(expected)
        closed_form = torch.tensor(
            [1.0 - 0.1 * step for step in range(10)], dtype=torch.float32
        )
        self.assertTrue(torch.equal(rollout.times, expected))
        self.assertFalse(torch.equal(rollout.times, closed_form))

    def test_reverse_changes_time_dependent_flow_but_preserves_budget(self):
        dt = -0.1
        schedule = torch.zeros((10, *self.initial.shape), dtype=self.initial.dtype)
        schedule[5, 0, 0, 0] = -0.12 / dt
        schedule[9, 0, 0, 1] = -0.04 / dt
        reversed_schedule = self.module.reverse_active_schedule(schedule)

        def time_dependent_field(x_t, _time, step):
            velocity = torch.zeros_like(x_t)
            velocity[..., 0] = (step + 1) * 0.25 * x_t[..., 0]
            velocity[..., 1] = (step + 1) * 0.10 * x_t[..., 0]
            return velocity

        forward = self.module.replay_velocity_schedule(
            self.initial, time_dependent_field, schedule, config=self.tight_config()
        )
        reverse = self.module.replay_velocity_schedule(
            self.initial, time_dependent_field, reversed_schedule, config=self.tight_config()
        )
        forward_increments = (dt * schedule[5:]).reshape(5, -1)
        reverse_increments = (dt * reversed_schedule[5:]).reshape(5, -1)
        self.assertTrue(torch.equal(torch.flip(forward_increments, dims=(0,)), reverse_increments))
        self.assertAlmostEqual(
            torch.linalg.vector_norm(forward_increments, dim=1).sum().item(),
            torch.linalg.vector_norm(reverse_increments, dim=1).sum().item(),
        )
        self.assertFalse(torch.allclose(forward.final, reverse.final, atol=1.0e-10, rtol=0.0))

    def test_stale_non_xyz_target_is_explicit_pairing_mismatch(self):
        target = self.initial.clone()
        target[0, 0, 3] = 0.2
        result = self.module.solve_inverse_control(
            self.initial,
            target,
            self.zero_field,
            control_mask=self.control_mask,
            target_mask=self.target_mask,
            model_to_physical_scale=torch.ones_like(self.initial),
            config=self.tight_config(),
        )
        self.assertEqual(
            result.status, self.module.InverseControlStatus.TARGET_PAIRING_MISMATCH
        )
        self.assertFalse(result.converged)
        self.assertFalse(result.infeasibility_certificate)
        self.assertTrue(result.target_pairing_checked)
        self.assertFalse(result.target_pairing_exact)

        padded_target = self.initial.clone()
        padded_target[0, 9, 31] = 0.2
        padded_result = self.module.solve_inverse_control(
            self.initial,
            padded_target,
            self.zero_field,
            control_mask=self.control_mask,
            target_mask=self.target_mask,
            model_to_physical_scale=torch.ones_like(self.initial),
            config=self.tight_config(),
        )
        self.assertEqual(
            padded_result.status,
            self.module.InverseControlStatus.TARGET_PAIRING_MISMATCH,
        )

    def test_exact_frozen_target_is_zero_budget_converged(self):
        result = self.module.solve_inverse_control(
            self.initial,
            self.initial.clone(),
            self.zero_field,
            control_mask=self.control_mask,
            target_mask=self.target_mask,
            model_to_physical_scale=torch.ones_like(self.initial),
            config=self.tight_config(),
        )
        self.assertEqual(
            result.status, self.module.InverseControlStatus.ZERO_BUDGET_CONVERGED
        )
        self.assertTrue(result.target_pairing_exact)
        self.assertEqual(result.budget.item(), 0.0)
        self.assertEqual(torch.count_nonzero(result.schedule).item(), 0)

    def test_attenuating_flow_reports_nonconvergence(self):
        target = self.initial.clone()
        target[0, 0, 0] = 0.1

        def attenuating_field(x_t, _time, _step):
            return 5.0 * x_t

        result = self.module.solve_inverse_control(
            self.initial,
            target,
            attenuating_field,
            control_mask=self.control_mask,
            target_mask=self.target_mask,
            model_to_physical_scale=torch.ones_like(self.initial),
            config=self.tight_config(max_iterations=20, learning_rate=0.01),
        )
        self.assertEqual(result.status, self.module.InverseControlStatus.MAX_ITERATIONS)
        self.assertFalse(result.converged)
        self.assertFalse(result.infeasibility_certificate)
        self.assertLessEqual(result.path_length.item(), result.budget.item() + 1.0e-10)

    def test_explicit_source_budget_is_not_replaced_by_rounded_target_delta(self):
        target = self.initial.clone()
        target[0, 0, 0] = 0.1
        source_budget = torch.tensor(0.08, dtype=self.initial.dtype)
        result = self.module.solve_inverse_control(
            self.initial,
            target,
            self.zero_field,
            control_mask=self.control_mask,
            target_mask=self.target_mask,
            model_to_physical_scale=torch.ones_like(self.initial),
            control_budget=source_budget,
            config=self.tight_config(max_iterations=20),
        )
        self.assertEqual(result.status, self.module.InverseControlStatus.MAX_ITERATIONS)
        self.assertTrue(torch.equal(result.budget, source_budget))
        self.assertEqual(result.realized_target_delta_norm.item(), 0.1)
        self.assertLessEqual(result.path_length.item(), source_budget.item() + 1.0e-12)

        zero_budget = self.module.solve_inverse_control(
            self.initial,
            target,
            self.zero_field,
            control_mask=self.control_mask,
            target_mask=self.target_mask,
            model_to_physical_scale=torch.ones_like(self.initial),
            control_budget=torch.tensor(0.0, dtype=self.initial.dtype),
            config=self.tight_config(max_iterations=2),
        )
        self.assertEqual(
            zero_budget.status, self.module.InverseControlStatus.MAX_ITERATIONS
        )
        self.assertFalse(zero_budget.converged)
        self.assertEqual(zero_budget.path_length.item(), 0.0)

        for invalid in (-1.0, float("inf")):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(
                ValueError, "control_budget"
            ):
                self.module.solve_inverse_control(
                    self.initial,
                    target,
                    self.zero_field,
                    control_mask=self.control_mask,
                    target_mask=self.target_mask,
                    model_to_physical_scale=torch.ones_like(self.initial),
                    control_budget=invalid,
                    config=self.tight_config(max_iterations=1),
                )

    def test_nonfinite_flow_returns_status_without_throwing(self):
        target = self.initial.clone()
        target[0, 0, 0] = 0.1

        def nonfinite_field(x_t, _time, step):
            value = torch.zeros_like(x_t)
            if step == 7:
                value[0, 0, 0] = float("nan")
            return value

        result = self.module.solve_inverse_control(
            self.initial,
            target,
            nonfinite_field,
            control_mask=self.control_mask,
            target_mask=self.target_mask,
            model_to_physical_scale=torch.ones_like(self.initial),
            config=self.tight_config(),
        )
        self.assertEqual(result.status, self.module.InverseControlStatus.NONFINITE)
        self.assertTrue(result.nonfinite_detected)

    def test_prefix_is_cached_across_optimizer_iterations(self):
        calls = [0] * 10
        target = self.initial.clone()
        target[0, 0, 0] = 0.1

        def counted_attenuating_field(x_t, _time, step):
            calls[step] += 1
            return 5.0 * x_t

        self.module.solve_inverse_control(
            self.initial,
            target,
            counted_attenuating_field,
            control_mask=self.control_mask,
            target_mask=self.target_mask,
            model_to_physical_scale=torch.ones_like(self.initial),
            config=self.tight_config(max_iterations=3, learning_rate=0.01),
        )
        # Once for cached prefix, once for the final canonical replay.  Active
        # steps are evaluated repeatedly by the optimizer as intended.
        self.assertEqual(calls[:5], [1, 1, 1, 1, 1])
        self.assertTrue(all(value > 2 for value in calls[5:]))

    def test_registered_masks_scale_and_detached_results_are_fail_closed(self):
        target = self.initial.clone()
        target[0, 0, 3] = 0.01
        scale = torch.ones_like(self.initial)
        scale[..., 3] = 10.0
        result = self.module.solve_inverse_control(
            self.initial,
            target,
            self.zero_field,
            control_mask=self.control_mask,
            target_mask=self.target_mask,
            model_to_physical_scale=scale,
            config=self.module.InverseControlConfig(),
        )
        self.assertEqual(
            result.status, self.module.InverseControlStatus.TARGET_PAIRING_MISMATCH
        )
        self.assertAlmostEqual(result.fidelity_error[0, 0, 3].item(), -0.1)
        self.assertTrue(torch.equal(result.fidelity_scale, scale))

        signed_zero_target = self.initial.clone()
        signed_zero_target[0, 0, 3] = -0.0
        signed_zero_result = self.module.solve_inverse_control(
            self.initial,
            signed_zero_target,
            self.zero_field,
            control_mask=self.control_mask,
            target_mask=self.target_mask,
            model_to_physical_scale=torch.ones_like(self.initial),
        )
        self.assertEqual(
            signed_zero_result.status,
            self.module.InverseControlStatus.TARGET_PAIRING_MISMATCH,
        )
        self.assertEqual(signed_zero_result.target_pairing_max_abs.item(), 0.0)

        wrong_control = self.control_mask.clone()
        wrong_control[0, 0, 2] = False
        with self.assertRaisesRegex(ValueError, "exactly first-five XYZ"):
            self.module.solve_inverse_control(
                self.initial,
                target,
                self.zero_field,
                control_mask=wrong_control,
                target_mask=self.target_mask,
                model_to_physical_scale=scale,
            )

        def assert_graph_free(value):
            if isinstance(value, torch.Tensor):
                self.assertFalse(value.requires_grad)
                self.assertIsNone(value.grad_fn)
            elif dataclasses.is_dataclass(value):
                for field in dataclasses.fields(value):
                    assert_graph_free(getattr(value, field.name))

        assert_graph_free(result)

    def test_xyz_and_full_fidelity_gates_fail_independently(self):
        config = self.module.InverseControlConfig()
        xyz_error = torch.zeros_like(self.initial)
        xyz_error[0, 0, 0] = 0.011
        xyz_metrics = self.module._fidelity_metrics(
            xyz_error, self.control_mask, self.target_mask, config
        )
        self.assertGreater(xyz_metrics.xyz_max_abs.item(), config.xyz_max_abs_tolerance)
        self.assertLess(xyz_metrics.full_max_abs.item(), config.full_max_abs_tolerance)
        self.assertFalse(xyz_metrics.feasible)

        full_error = torch.zeros_like(self.initial)
        full_error[0, 0, 3] = 0.051
        full_metrics = self.module._fidelity_metrics(
            full_error, self.control_mask, self.target_mask, config
        )
        self.assertEqual(full_metrics.xyz_max_abs.item(), 0.0)
        self.assertGreater(full_metrics.full_max_abs.item(), config.full_max_abs_tolerance)
        self.assertFalse(full_metrics.feasible)

    def test_constraint_validator_rejects_wrong_timing_mask_and_cap(self):
        budget = torch.tensor(0.1, dtype=self.initial.dtype)
        schedule = torch.zeros((10, *self.initial.shape), dtype=self.initial.dtype)
        schedule[4, 0, 0, 0] = 0.1
        with self.assertRaisesRegex(ValueError, "steps 0--4"):
            self.module.validate_schedule_constraints(
                schedule, self.control_mask, budget, config=self.tight_config()
            )

        schedule.zero_()
        schedule[5, 0, 0, 3] = 0.1
        with self.assertRaisesRegex(ValueError, "outside"):
            self.module.validate_schedule_constraints(
                schedule, self.control_mask, budget, config=self.tight_config()
            )

        schedule.zero_()
        schedule[5, 0, 0, 0] = -0.3
        with self.assertRaisesRegex(ValueError, "B/5"):
            self.module.validate_schedule_constraints(
                schedule, self.control_mask, budget, config=self.tight_config()
            )

        schedule.zero_()
        schedule[5, 0, 0, 0] = 1.0e-8
        with self.assertRaisesRegex(ValueError, "zero budget"):
            self.module.validate_schedule_constraints(
                schedule,
                self.control_mask,
                torch.tensor(0.0, dtype=self.initial.dtype),
                config=self.tight_config(),
            )

    def test_nonfinite_configuration_is_rejected(self):
        for field in (
            "learning_rate",
            "adam_epsilon",
            "xyz_max_abs_tolerance",
            "full_rms_tolerance",
        ):
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    self.module.InverseControlConfig(**{field: float("inf")}).validate()


if __name__ == "__main__":
    unittest.main()
