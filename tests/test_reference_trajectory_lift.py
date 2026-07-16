from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
from types import MethodType
from types import SimpleNamespace
import sys
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
HELPER_PATH = ROOT / "openpi/src/openpi/models_pytorch/crfs_reference_trajectory.py"
SAMPLER_PATH = ROOT / "openpi/src/openpi/models_pytorch/pi0_pytorch.py"
POLICY_PATH = ROOT / "openpi/src/openpi/policies/policy.py"


class ReferenceTrajectoryLiftStructuralTest(unittest.TestCase):
    def test_mode_is_isolated_opt_in_and_preserves_the_euler_statement(self) -> None:
        helper = HELPER_PATH.read_text(encoding="utf-8")
        sampler = SAMPLER_PATH.read_text(encoding="utf-8")
        policy = POLICY_PATH.read_text(encoding="utf-8")
        ast.parse(helper)
        ast.parse(sampler)
        ast.parse(policy)

        self.assertNotIn("crfs_inverse_control", helper)
        self.assertNotIn("crfs_constrained_flow", helper)
        self.assertIn('"reference_trajectory_lift"', sampler)
        self.assertIn('intervention_mode == "reference_trajectory_lift"', policy)
        self.assertIn('crfs_intervention_mode="none"', sampler)
        self.assertEqual(sampler.count("x_t = x_t + dt * v_t"), 1)
        self.assertIn(
            "self._sample_actions_crfs if use_crfs_sampler else self._sample_actions",
            policy,
        )

    def test_online_formula_projection_and_trace_are_explicit(self) -> None:
        helper = HELPER_PATH.read_text(encoding="utf-8")
        sampler = SAMPLER_PATH.read_text(encoding="utf-8")
        for fragment in (
            "desired_next = reference_next",
            "uncontrolled_next = x_t + dt * v_base",
            "desired_next - uncontrolled_next",
            "requested_increment / dt",
            "dt * requested_velocity",
            "for coordinate in range(15)",
            "raw_increment.to(torch.float64)",
            "requested_increment = torch.where(",
        ):
            self.assertIn(fragment, helper)
        for leaf in (
            "reference_active_states",
            "reference_delta",
            "reference_alpha",
            "reference_anchor_exact",
            "reference_desired_next_steps",
            "reference_uncontrolled_next_steps",
            "reference_raw_increment_steps",
            "reference_requested_increment_steps",
            "reference_requested_velocity_steps",
            "reference_executed_increment_steps",
            "reference_raw_norm_f64_steps",
            "reference_requested_norm_f64_steps",
            "reference_executed_norm_f64_steps",
            "reference_projection_scale_f64_steps",
            "reference_tracking_error_steps",
            "reference_product_ball_valid",
        ):
            self.assertIn(leaf, sampler)

    def test_policy_envelope_is_strict_and_model_space_only(self) -> None:
        source = POLICY_PATH.read_text(encoding="utf-8")
        self.assertIn("def _reference_trajectory_lift_sample_kwargs(", source)
        self.assertIn("required = {", source)
        self.assertIn("unknown = set(controls) - required - optional", source)
        self.assertIn("reference_space='model'", source)
        self.assertIn("delta_space='model'", source)
        self.assertIn("exact positive zero outside first-five XYZ", source)
        self.assertIn('"crfs_reference_projection_mode": str(projection_mode)', source)
        self.assertNotIn("except RawFloat32Unrepresentable", source)
        self.assertNotIn('"__crfs_terminal__"', source)


HELPER_IMPORT_ERROR: Exception | None = None
try:
    import numpy as np
    import torch

    _helper_spec = importlib.util.spec_from_file_location(
        "_reference_trajectory_lift_helper_test",
        HELPER_PATH,
    )
    assert _helper_spec is not None and _helper_spec.loader is not None
    reference = importlib.util.module_from_spec(_helper_spec)
    sys.modules[_helper_spec.name] = reference
    _helper_spec.loader.exec_module(reference)
except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover - dependency-free gate.
    HELPER_IMPORT_ERROR = exc


OPENPI_IMPORT_ERROR: Exception | None = HELPER_IMPORT_ERROR
if OPENPI_IMPORT_ERROR is None:
    try:
        sys.path.insert(0, str(ROOT / "openpi/src"))
        from openpi.models_pytorch import crfs_reference_trajectory as openpi_reference
        from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
        import openpi.policies.policy as policy_module
        from openpi.policies.policy import Policy
    except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover - optional OpenPI gate.
        OPENPI_IMPORT_ERROR = exc


@unittest.skipIf(
    HELPER_IMPORT_ERROR is not None,
    f"reference-trajectory helper checks require NumPy and PyTorch: {HELPER_IMPORT_ERROR}",
)
class ReferenceTrajectoryHelperRuntimeTest(unittest.TestCase):
    def test_alpha_uses_registered_float32_bytes_and_positive_zero(self) -> None:
        alpha = reference.reference_alpha_like(torch.zeros((), dtype=torch.float32))
        self.assertEqual(alpha.dtype, torch.float32)
        self.assertEqual(
            alpha.cpu().numpy().view(np.uint32).tolist(),
            [0x00000000, 0x3E4CCCCD, 0x3ECCCCCD, 0x3F19999A, 0x3F4CCCCD, 0x3F800000],
        )
        self.assertFalse(torch.signbit(alpha[0]).item())

    def test_fixed_order_norm_and_raw_formula_are_exact(self) -> None:
        raw = torch.zeros((1, 5, 7), dtype=torch.float32)
        raw[0, 0, 0] = 3.0
        raw[0, 0, 1] = 4.0
        norm = reference.fixed_order_first_five_xyz_norm_f64(raw)
        self.assertEqual(norm.dtype, torch.float64)
        self.assertEqual(norm.item(), 5.0)

        x_t = torch.zeros_like(raw)
        v_base = torch.full_like(raw, 0.25)
        reference_next = torch.zeros_like(raw)
        reference_next[0, 0, 0] = 0.02
        result = reference.reference_step(
            x_t,
            v_base,
            reference_next,
            torch.tensor(0.1, dtype=torch.float32),
            torch.tensor(-0.1, dtype=torch.float32),
            reference.RAW_PROJECTION,
            step_index=5,
        )
        expected_uncontrolled = x_t + torch.tensor(-0.1, dtype=torch.float32) * v_base
        self.assertTrue(torch.equal(result.uncontrolled_next, expected_uncontrolled))
        expected_raw = torch.zeros_like(raw)
        expected_raw[:, :5, :3] = (
            reference_next - expected_uncontrolled
        )[:, :5, :3]
        self.assertTrue(torch.equal(result.raw_increment, expected_raw))
        self.assertTrue(torch.equal(result.requested_increment, expected_raw))
        self.assertTrue(
            torch.equal(
                result.executed_increment,
                torch.tensor(-0.1, dtype=torch.float32) * result.requested_velocity,
            )
        )
        self.assertFalse(result.projected.item())
        self.assertEqual(result.projection_scale_f64.item(), 1.0)

    def test_product_ball_uses_float64_scale_and_float32_authoritative_control(self) -> None:
        shape = (1, 5, 7)
        reference_next = torch.zeros(shape, dtype=torch.float32)
        reference_next[0, :5, :3] = 1.0
        result = reference.reference_step(
            torch.zeros(shape, dtype=torch.float32),
            torch.zeros(shape, dtype=torch.float32),
            reference_next,
            torch.tensor(0.5, dtype=torch.float32),
            torch.tensor(-0.1, dtype=torch.float32),
            reference.PRODUCT_BALL_PROJECTION,
            step_index=5,
        )
        self.assertTrue(result.projected.item())
        self.assertEqual(result.raw_norm_f64.dtype, torch.float64)
        self.assertEqual(result.projection_scale_f64.dtype, torch.float64)
        self.assertEqual(result.requested_increment.dtype, torch.float32)
        self.assertEqual(result.executed_increment.dtype, torch.float32)
        self.assertLess(result.projection_scale_f64.item(), 1.0)
        self.assertAlmostEqual(result.executed_norm_f64.item(), 0.1, places=7)
        self.assertEqual(torch.count_nonzero(result.executed_increment[..., 3:]).item(), 0)
        outside_mask = ~reference.first_five_xyz_mask_like(result.requested_velocity)
        self.assertFalse(torch.signbit(result.requested_velocity[outside_mask]).any().item())

    def test_product_ball_zero_and_noop_preserve_raw_signed_zero_bytes(self) -> None:
        shape = (1, 5, 7)
        reference_next = torch.zeros(shape, dtype=torch.float32)
        reference_next[0, 0, 0] = -0.0
        reference_next[0, 0, 1] = 0.01
        result = reference.reference_step(
            torch.zeros(shape, dtype=torch.float32),
            torch.zeros(shape, dtype=torch.float32),
            reference_next,
            torch.tensor(1.0, dtype=torch.float32),
            torch.tensor(-0.1, dtype=torch.float32),
            reference.PRODUCT_BALL_PROJECTION,
            step_index=5,
        )
        self.assertFalse(result.projected.item())
        self.assertTrue(
            reference.finite_bitwise_equal(
                result.raw_increment,
                result.requested_increment,
            )
        )
        self.assertTrue(torch.signbit(result.requested_increment[0, 0, 0]).item())
        outside_mask = ~reference.first_five_xyz_mask_like(result.requested_velocity)
        self.assertFalse(torch.signbit(result.requested_velocity[outside_mask]).any().item())
        self.assertFalse(torch.signbit(result.executed_increment[outside_mask]).any().item())

        zero_result = reference.reference_step(
            torch.zeros(shape, dtype=torch.float32),
            torch.zeros(shape, dtype=torch.float32),
            torch.zeros(shape, dtype=torch.float32),
            torch.tensor(1.0, dtype=torch.float32),
            torch.tensor(-0.1, dtype=torch.float32),
            reference.PRODUCT_BALL_PROJECTION,
            step_index=5,
        )
        self.assertEqual(zero_result.raw_norm_f64.item(), 0.0)
        self.assertEqual(zero_result.projection_scale_f64.item(), 1.0)
        self.assertTrue(
            reference.finite_bitwise_equal(
                zero_result.raw_increment,
                zero_result.requested_increment,
            )
        )

    def test_raw_nonfinite_raises_before_application(self) -> None:
        shape = (1, 5, 7)
        maximum = torch.finfo(torch.float32).max
        reference_next = torch.zeros(shape, dtype=torch.float32)
        reference_next[0, 0, 0] = maximum
        state = torch.zeros(shape, dtype=torch.float32)
        state[0, 0, 0] = -maximum
        with self.assertRaises(reference.RawFloat32Unrepresentable) as caught:
            reference.reference_step(
                state,
                torch.zeros(shape, dtype=torch.float32),
                reference_next,
                torch.tensor(1.0, dtype=torch.float32),
                torch.tensor(-0.1, dtype=torch.float32),
                reference.RAW_PROJECTION,
                step_index=7,
            )
        self.assertEqual(caught.exception.step, 7)
        self.assertEqual(caught.exception.leaf, "raw_increment")
        self.assertEqual(str(caught.exception), "RAW_FLOAT32_UNREPRESENTABLE(7,raw_increment)")

    def test_validation_rejects_outside_mask_and_signed_zero(self) -> None:
        initial = torch.zeros((1, 5, 7), dtype=torch.float32)
        states = torch.zeros((1, 6, 5, 7), dtype=torch.float32)
        delta = torch.zeros_like(initial)
        delta[0, 0, 3] = -0.0
        with self.assertRaisesRegex(ValueError, "positive zero"):
            reference.validate_reference_inputs(
                initial,
                states,
                delta,
                torch.tensor(0.1, dtype=torch.float32),
                reference.RAW_PROJECTION,
            )


@unittest.skipIf(
    OPENPI_IMPORT_ERROR is not None,
    f"reference-trajectory sampler checks require OpenPI dependencies: {OPENPI_IMPORT_ERROR}",
)
class ReferenceTrajectorySamplerRuntimeTest(unittest.TestCase):
    ACTION_SHAPE = (5, 7)

    class _PrefixModel(torch.nn.Module if OPENPI_IMPORT_ERROR is None else object):
        def __init__(self):
            super().__init__()
            self.paligemma = SimpleNamespace(
                language_model=SimpleNamespace(
                    config=SimpleNamespace(_attn_implementation="eager")
                )
            )

        def forward(self, **_kwargs):
            return None, None

    def _sampler(self, *, field_gain: float = 0.0):
        sampler = object.__new__(PI0Pytorch)
        torch.nn.Module.__init__(sampler)
        sampler.config = SimpleNamespace(
            action_horizon=self.ACTION_SHAPE[0], action_dim=self.ACTION_SHAPE[1]
        )
        sampler.paligemma_with_expert = self._PrefixModel()

        def preprocess(_self, _observation, *, train):
            self.assertFalse(train)
            return (
                [],
                [],
                torch.zeros((1, 1), dtype=torch.int64),
                torch.ones((1, 1), dtype=torch.bool),
                torch.zeros((1, 7), dtype=torch.float32),
            )

        def embed_prefix(_self, _images, _img_masks, _tokens, _masks):
            return (
                torch.zeros((1, 1, 1), dtype=torch.float32),
                torch.ones((1, 1), dtype=torch.bool),
                torch.zeros((1, 1), dtype=torch.bool),
            )

        def denoise(_self, _state, _prefix_masks, _cache, x_t, _time):
            return field_gain * x_t

        object.__setattr__(sampler, "_preprocess_observation", MethodType(preprocess, sampler))
        object.__setattr__(sampler, "embed_prefix", MethodType(embed_prefix, sampler))
        object.__setattr__(sampler, "denoise_step", MethodType(denoise, sampler))
        return sampler

    @staticmethod
    def _observation():
        return SimpleNamespace(state=torch.zeros((1, 7), dtype=torch.float32))

    def _reference_kwargs(self, *, projection: str, delta_value: float, budget: float):
        shape = (1, *self.ACTION_SHAPE)
        delta = torch.zeros(shape, dtype=torch.float32)
        delta[0, 0, 0] = delta_value
        reference_states = torch.zeros(
            (1, 6, *self.ACTION_SHAPE), dtype=torch.float32
        )
        alpha = torch.zeros((6,), dtype=torch.float32)
        alpha[1:] = torch.arange(1, 6, dtype=torch.float32) / torch.tensor(
            5, dtype=torch.float32
        )
        reference_states[0, :, 0, 0] = alpha * delta[0, 0, 0]
        return dict(
            noise=torch.zeros(shape, dtype=torch.float32),
            num_steps=10,
            crfs_intervention_step=5,
            crfs_intervention_mode="reference_trajectory_lift",
            crfs_return_trace=True,
            crfs_return_normalized_final=True,
            crfs_reference_states=reference_states,
            crfs_reference_delta=delta,
            crfs_reference_projection_mode=projection,
            crfs_reference_budget=torch.tensor(budget, dtype=torch.float32),
        )

    def test_raw_online_lift_consumes_full_reference_without_double_adding_delta(self) -> None:
        sampler = self._sampler(field_gain=2.0)
        final, trace = PI0Pytorch.sample_actions(
            sampler,
            "cpu",
            self._observation(),
            **self._reference_kwargs(projection="raw", delta_value=0.1, budget=0.3),
        )
        self.assertAlmostEqual(final[0, 0, 0].item(), 0.1, places=6)
        self.assertEqual(trace["control_source"].item(), 2)
        self.assertTrue(trace["reference_anchor_exact"].item())
        self.assertEqual(trace["reference_projection_mode"].item(), 0)
        self.assertEqual(torch.count_nonzero(trace["reference_projected_steps"]).item(), 0)
        self.assertTrue(
            torch.equal(
                trace["reference_desired_next_steps"][:, 5:],
                trace["reference_active_states"][:, 1:],
            )
        )
        self.assertTrue(
            torch.equal(
                trace["reference_executed_increment_steps"],
                trace["control_increment_steps"],
            )
        )
        self.assertTrue(
            torch.equal(
                trace["reference_executed_increment_steps"],
                trace["dt"].reshape(1, 1, 1, 1)
                * trace["reference_requested_velocity_steps"],
            )
        )
        self.assertLess(
            torch.max(torch.abs(trace["reference_tracking_error_steps"][:, :5])).item(),
            1.0e-6,
        )

        replay_final, replay_trace = PI0Pytorch.sample_actions(
            sampler,
            "cpu",
            self._observation(),
            noise=torch.zeros((1, *self.ACTION_SHAPE), dtype=torch.float32),
            num_steps=10,
            crfs_intervention_step=5,
            crfs_intervention_mode="residual_schedule",
            crfs_return_trace=True,
            crfs_return_normalized_final=True,
            crfs_residual_schedule=trace["control_velocity_steps"],
            crfs_schedule_budget=torch.tensor(0.3, dtype=torch.float32),
        )
        self.assertTrue(torch.equal(final, replay_final))
        for leaf in (
            "x_t_steps",
            "v_base_steps",
            "control_velocity_steps",
            "total_velocity_steps",
            "control_increment_steps",
            "x_next_steps",
        ):
            self.assertTrue(torch.equal(trace[leaf], replay_trace[leaf]), leaf)

    def test_product_ball_caps_each_step_and_preserves_budget(self) -> None:
        final, trace = PI0Pytorch.sample_actions(
            self._sampler(),
            "cpu",
            self._observation(),
            **self._reference_kwargs(
                projection="product_ball", delta_value=0.2, budget=0.1
            ),
        )
        self.assertAlmostEqual(final[0, 0, 0].item(), 0.1, places=6)
        self.assertEqual(trace["reference_projection_mode"].item(), 1)
        self.assertTrue(trace["reference_product_ball_valid"].item())
        self.assertEqual(torch.count_nonzero(trace["reference_projected_steps"][:, 5:]).item(), 5)
        self.assertTrue(bool((trace["schedule_per_step_increment_l2"] <= 0.02000001).all()))
        self.assertLessEqual(trace["schedule_path_length"].item(), 0.1000001)

    def test_anchor_mismatch_fails_before_any_reference_control(self) -> None:
        kwargs = self._reference_kwargs(projection="raw", delta_value=0.1, budget=0.1)
        kwargs["crfs_reference_states"][0, 0, 0, 0] = 1.0
        with self.assertRaisesRegex(RuntimeError, "live step-5 prefix"):
            PI0Pytorch.sample_actions(
                self._sampler(), "cpu", self._observation(), **kwargs
            )


@unittest.skipIf(
    OPENPI_IMPORT_ERROR is not None,
    f"reference-trajectory policy checks require OpenPI dependencies: {OPENPI_IMPORT_ERROR}",
)
class ReferenceTrajectoryPolicyRuntimeTest(unittest.TestCase):
    ACTION_SHAPE = (10, 32)

    def _policy(self):
        policy = object.__new__(Policy)
        policy._model = SimpleNamespace(
            config=SimpleNamespace(action_horizon=10, action_dim=32)
        )
        policy._pytorch_device = "cpu"
        return policy

    def _inference_policy(self, sample_function):
        policy = self._policy()
        policy._is_pytorch_model = True
        policy._input_transform = lambda value: value
        policy._output_transform = lambda value: value
        policy._sample_kwargs = {}
        policy._sample_actions = sample_function
        policy._sample_actions_crfs = sample_function
        return policy

    def _controls(self):
        return {
            "noise": np.zeros(self.ACTION_SHAPE, dtype=np.float32),
            "intervention_mode": "reference_trajectory_lift",
            "intervention_step": 5,
            "return_trace": True,
            "return_normalized_final": True,
            "reference_states": np.zeros((6, *self.ACTION_SHAPE), dtype=np.float32),
            "reference_space": "model",
            "delta": np.zeros(self.ACTION_SHAPE, dtype=np.float32),
            "delta_space": "model",
            "projection_mode": "product_ball",
            "model_l2_path_budget": np.asarray(0.1, dtype=np.float32),
        }

    def test_policy_tensorizes_exact_reference_contract(self) -> None:
        controls = self._controls()
        kwargs, noise = self._policy()._reference_trajectory_lift_sample_kwargs(
            controls,
            controls["noise"],
            noise_argument_supplied=False,
        )
        self.assertEqual(tuple(kwargs["crfs_reference_states"].shape), (1, 6, 10, 32))
        self.assertEqual(tuple(kwargs["crfs_reference_delta"].shape), (1, 10, 32))
        self.assertEqual(kwargs["crfs_reference_states"].dtype, torch.float32)
        self.assertEqual(kwargs["crfs_reference_projection_mode"], "product_ball")
        self.assertEqual(kwargs["crfs_reference_budget"].dtype, torch.float32)
        self.assertEqual(noise.dtype, np.dtype(np.float32))

    def test_policy_rejects_unknown_space_dtype_projection_and_ambiguous_noise(self) -> None:
        for mutation, message in (
            (lambda value: value.__setitem__("extra", 1), "unsupported"),
            (lambda value: value.__setitem__("reference_space", "physical"), "reference_space='model'"),
            (lambda value: value.__setitem__("delta_space", "physical"), "delta_space='model'"),
            (lambda value: value.__setitem__("projection_mode", "clip"), "projection_mode"),
            (
                lambda value: value.__setitem__(
                    "reference_states", value["reference_states"].astype(np.float64)
                ),
                "float32",
            ),
        ):
            with self.subTest(message=message):
                controls = self._controls()
                mutation(controls)
                with self.assertRaisesRegex(ValueError, message):
                    self._policy()._reference_trajectory_lift_sample_kwargs(
                        controls,
                        controls["noise"],
                        noise_argument_supplied=False,
                    )

        controls = self._controls()
        with self.assertRaisesRegex(ValueError, "exactly one source"):
            self._policy()._reference_trajectory_lift_sample_kwargs(
                controls,
                controls["noise"],
                noise_argument_supplied=True,
            )

    def test_policy_propagates_raw_nonfinite_as_apparatus_failure(self) -> None:
        def terminal_sample(*_args, **_kwargs):
            raise openpi_reference.RawFloat32Unrepresentable(8, "requested_velocity")

        observation = {
            "state": np.zeros((7,), dtype=np.float32),
            "__crfs__": self._controls(),
        }
        with mock.patch.object(
            policy_module._model.Observation,
            "from_dict",
            return_value=SimpleNamespace(state=torch.zeros((1, 7), dtype=torch.float32)),
        ):
            with self.assertRaises(openpi_reference.RawFloat32Unrepresentable):
                self._inference_policy(terminal_sample).infer(observation)

    def test_policy_does_not_convert_generic_or_nonraw_failures(self) -> None:
        def generic_sample(*_args, **_kwargs):
            raise RuntimeError("generic sampler failure")

        raw_observation = {
            "state": np.zeros((7,), dtype=np.float32),
            "__crfs__": self._controls(),
        }
        observation_result = SimpleNamespace(
            state=torch.zeros((1, 7), dtype=torch.float32)
        )
        with mock.patch.object(
            policy_module._model.Observation,
            "from_dict",
            return_value=observation_result,
        ):
            with self.assertRaisesRegex(RuntimeError, "generic sampler failure"):
                self._inference_policy(generic_sample).infer(raw_observation)

        def typed_sample(*_args, **_kwargs):
            raise openpi_reference.RawFloat32Unrepresentable(5, "state")

        product_controls = self._controls()
        product_controls["projection_mode"] = "product_ball"
        product_observation = {
            "state": np.zeros((7,), dtype=np.float32),
            "__crfs__": product_controls,
        }
        with mock.patch.object(
            policy_module._model.Observation,
            "from_dict",
            return_value=observation_result,
        ):
            with self.assertRaises(openpi_reference.RawFloat32Unrepresentable):
                self._inference_policy(typed_sample).infer(product_observation)


if __name__ == "__main__":
    unittest.main()
