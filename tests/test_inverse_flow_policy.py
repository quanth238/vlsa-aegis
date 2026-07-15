from __future__ import annotations

import ast
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "openpi/src/openpi/policies/policy.py"


class InverseFlowPolicyStructuralTest(unittest.TestCase):
    def test_new_modes_are_opt_in_and_route_only_reserved_kwargs(self) -> None:
        source = POLICY_PATH.read_text(encoding="utf-8")
        ast.parse(source)
        self.assertIn('intervention_mode == "inverse_flow_teacher"', source)
        self.assertIn('intervention_mode == "residual_schedule"', source)
        self.assertIn('"crfs_inverse_target": batched_tensor(target)', source)
        self.assertIn('"crfs_inverse_budget": torch.tensor(', source)
        self.assertIn('"crfs_model_to_physical_scale": batched_tensor(scale)', source)
        self.assertIn('"crfs_inverse_config": config', source)
        self.assertIn('"crfs_residual_schedule": schedule_tensor', source)
        self.assertIn('"crfs_schedule_budget": budget_tensor', source)
        self.assertIn('in {"analytic_trajectory_field", "inverse_flow_teacher", "residual_schedule"}', source)
        self.assertIn("audited_actions.dtype == returned_actions.dtype", source)
        self.assertIn("audited_actions.tobytes() == returned_actions.tobytes()", source)
        self.assertIn("inverse-flow sampler did not return its required audit trace", source)
        self.assertIn("inverse-flow audit trace is missing final_normalized", source)
        self.assertIn("use_crfs_sampler = False", source)
        self.assertIn("self._sample_actions_crfs if use_crfs_sampler else self._sample_actions", source)

    def test_ift00a_values_and_policy_owned_scale_are_frozen(self) -> None:
        source = POLICY_PATH.read_text(encoding="utf-8")
        for fragment in (
            '"num_steps": 10',
            '"intervention_step": 5',
            '"dt": -0.1',
            '"max_iterations": 128',
            '"learning_rate": 0.02',
            '"adam_beta1": 0.9',
            '"adam_beta2": 0.999',
            '"adam_epsilon": 1.0e-8',
            '"xyz_max_abs_tolerance": 0.010',
            '"xyz_rms_tolerance": 0.005',
            '"full_max_abs_tolerance": 0.050',
            '"full_rms_tolerance": 0.015',
            '"constraint_slack_ulps": 8',
            '"stop_on_first_feasible": False',
        ):
            self.assertIn(fragment, source)
        self.assertIn("scale = self._model_to_physical_action_scale()", source)
        self.assertIn("full_scale = np.ones((action_horizon, action_dim)", source)
        self.assertNotIn('controls["model_to_physical_scale"]', source)


IMPORT_ERROR: Exception | None = None
try:
    import numpy as np
    import torch

    sys.path.insert(0, str(ROOT / "openpi/src"))
    from openpi.models_pytorch.crfs_inverse_control import InverseControlConfig
    import openpi.policies.policy as policy_module
    from openpi.policies.policy import Policy
except ModuleNotFoundError as exc:  # pragma: no cover - dependency-free local gate.
    IMPORT_ERROR = exc


@unittest.skipIf(
    IMPORT_ERROR is not None,
    f"inverse-flow Policy runtime checks require OpenPI dependencies: {IMPORT_ERROR}",
)
class InverseFlowPolicyRuntimeTest(unittest.TestCase):
    ACTION_SHAPE = (10, 32)
    PHYSICAL_DIM = 7

    def _policy(self, *, quantile: bool = False):
        policy = object.__new__(Policy)
        policy._model = SimpleNamespace(
            config=SimpleNamespace(
                action_horizon=self.ACTION_SHAPE[0],
                action_dim=self.ACTION_SHAPE[1],
            )
        )
        policy._pytorch_device = "cpu"
        policy._use_quantile_norm = quantile
        policy._action_norm_stats = SimpleNamespace(
            mean=np.arange(self.PHYSICAL_DIM, dtype=np.float32),
            std=np.asarray([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], dtype=np.float32),
            q01=np.asarray([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32),
            q99=np.asarray([2.0, 1.5, 1.0, 0.5, 2.0, 2.5, 3.0], dtype=np.float32),
        )
        return policy

    def _solver_config(self) -> dict:
        return asdict(
            InverseControlConfig(
                num_steps=10,
                intervention_step=5,
                dt=-0.1,
                max_iterations=128,
                learning_rate=0.02,
                adam_beta1=0.9,
                adam_beta2=0.999,
                adam_epsilon=1.0e-8,
                xyz_max_abs_tolerance=0.010,
                xyz_rms_tolerance=0.005,
                full_max_abs_tolerance=0.050,
                full_rms_tolerance=0.015,
                constraint_slack_ulps=8,
                stop_on_first_feasible=False,
            )
        )

    def _teacher_controls(self) -> dict:
        return {
            "noise": np.zeros(self.ACTION_SHAPE, dtype=np.float32),
            "intervention_mode": "inverse_flow_teacher",
            "intervention_step": 5,
            "return_trace": True,
            "return_normalized_final": True,
            "target": np.zeros(self.ACTION_SHAPE, dtype=np.float32),
            "target_space": "model",
            "solver_config": self._solver_config(),
            "model_l2_path_budget": np.asarray(0.25, dtype=np.float32),
        }

    def _schedule_controls(self) -> dict:
        return {
            "noise": np.zeros(self.ACTION_SHAPE, dtype=np.float32),
            "intervention_mode": "residual_schedule",
            "intervention_step": 5,
            "return_trace": True,
            "return_normalized_final": True,
            "schedule": np.zeros((10, *self.ACTION_SHAPE), dtype=np.float32),
            "schedule_space": "model",
            "model_l2_path_budget": np.asarray(0.0, dtype=np.float32),
        }

    def test_teacher_tensorizes_target_config_and_policy_owned_std_scale(self) -> None:
        policy = self._policy()
        controls = self._teacher_controls()
        kwargs, noise = policy._inverse_flow_teacher_sample_kwargs(
            controls,
            controls["noise"],
            noise_argument_supplied=False,
        )
        self.assertEqual(tuple(kwargs["crfs_inverse_target"].shape), (1, *self.ACTION_SHAPE))
        self.assertEqual(kwargs["crfs_inverse_target"].dtype, torch.float32)
        self.assertEqual(tuple(kwargs["crfs_inverse_budget"].shape), ())
        self.assertEqual(kwargs["crfs_inverse_budget"].dtype, torch.float32)
        self.assertEqual(float(kwargs["crfs_inverse_budget"]), 0.25)
        self.assertEqual(tuple(kwargs["crfs_model_to_physical_scale"].shape), (1, *self.ACTION_SHAPE))
        expected = np.ones(self.ACTION_SHAPE, dtype=np.float32)
        expected[:, : self.PHYSICAL_DIM] = policy._action_norm_stats.std + 1.0e-6
        np.testing.assert_array_equal(
            kwargs["crfs_model_to_physical_scale"][0].numpy(),
            expected,
        )
        self.assertEqual(kwargs["crfs_inverse_config"], InverseControlConfig(**self._solver_config()))
        self.assertEqual(noise.dtype, np.dtype(np.float32))
        self.assertEqual(noise.shape, self.ACTION_SHAPE)

    def test_quantile_scale_uses_half_range_and_padded_channels_are_one(self) -> None:
        policy = self._policy(quantile=True)
        scale = policy._model_to_physical_action_scale()
        expected_physical = (
            policy._action_norm_stats.q99 - policy._action_norm_stats.q01 + 1.0e-6
        ) / 2.0
        np.testing.assert_allclose(
            scale[:, : self.PHYSICAL_DIM],
            np.broadcast_to(expected_physical, (self.ACTION_SHAPE[0], self.PHYSICAL_DIM)),
            rtol=0.0,
            atol=2.0e-7,
        )
        np.testing.assert_array_equal(
            scale[:, self.PHYSICAL_DIM :],
            np.ones((self.ACTION_SHAPE[0], self.ACTION_SHAPE[1] - self.PHYSICAL_DIM), dtype=np.float32),
        )
        self.assertTrue(bool((scale > 0.0).all()))

    def test_teacher_rejects_unknown_mixed_missing_or_changed_config(self) -> None:
        for mutation, message in (
            (lambda value: value.__setitem__("model_to_physical_scale", np.ones(self.ACTION_SHAPE)), "unsupported"),
            (lambda value: value.__setitem__("schedule", np.zeros((10, *self.ACTION_SHAPE))), "unsupported"),
            (lambda value: value["solver_config"].pop("dt"), "missing required keys"),
            (lambda value: value["solver_config"].__setitem__("extra", 1), "unsupported keys"),
            (lambda value: value["solver_config"].__setitem__("learning_rate", 0.03), "must equal 0.02"),
            (lambda value: value.pop("model_l2_path_budget"), "missing required controls"),
        ):
            with self.subTest(message=message):
                controls = self._teacher_controls()
                mutation(controls)
                with self.assertRaisesRegex(ValueError, message):
                    self._policy()._inverse_flow_teacher_sample_kwargs(
                        controls,
                        controls["noise"],
                        noise_argument_supplied=False,
                    )

    def test_teacher_rejects_bad_target_contract_and_ambiguous_noise(self) -> None:
        for mutation, message in (
            (lambda value: value.__setitem__("target", value["target"].astype(np.float64)), "float32"),
            (lambda value: value.__setitem__("target", value["target"][None, ...]), "unbatched shape"),
            (lambda value: value["target"].__setitem__((0, 0), np.nan), "nonfinite"),
            (lambda value: value.__setitem__("target_space", "physical"), "target_space='model'"),
            (lambda value: value.__setitem__("intervention_step", 4), "intervention_step=5"),
            (lambda value: value.__setitem__("return_trace", 1), "return_trace=true"),
            (lambda value: value.__setitem__("return_normalized_final", False), "return_normalized_final=true"),
            (
                lambda value: value.__setitem__("model_l2_path_budget", 0.25),
                "preserve float32 dtype",
            ),
            (
                lambda value: value.__setitem__(
                    "model_l2_path_budget", np.asarray([0.25], dtype=np.float32)
                ),
                "must be scalar",
            ),
            (
                lambda value: value.__setitem__(
                    "model_l2_path_budget", np.asarray(0.0, dtype=np.float32)
                ),
                "finite and positive",
            ),
            (
                lambda value: value.__setitem__(
                    "model_l2_path_budget", np.asarray(np.nan, dtype=np.float32)
                ),
                "finite and positive",
            ),
        ):
            with self.subTest(message=message):
                controls = self._teacher_controls()
                mutation(controls)
                with self.assertRaisesRegex(ValueError, message):
                    self._policy()._inverse_flow_teacher_sample_kwargs(
                        controls,
                        controls["noise"],
                        noise_argument_supplied=False,
                    )

        controls = self._teacher_controls()
        with self.assertRaisesRegex(ValueError, "exactly one source"):
            self._policy()._inverse_flow_teacher_sample_kwargs(
                controls,
                controls["noise"],
                noise_argument_supplied=True,
            )
        del controls["noise"]
        with self.assertRaisesRegex(ValueError, "explicit paired noise"):
            self._policy()._inverse_flow_teacher_sample_kwargs(
                controls,
                None,
                noise_argument_supplied=False,
            )

    def test_schedule_tensorizes_replay_and_scalar_budget(self) -> None:
        controls = self._schedule_controls()
        kwargs, noise = self._policy()._residual_schedule_sample_kwargs(
            controls,
            controls["noise"],
            noise_argument_supplied=False,
        )
        self.assertEqual(tuple(kwargs["crfs_residual_schedule"].shape), (1, 10, *self.ACTION_SHAPE))
        self.assertEqual(kwargs["crfs_residual_schedule"].dtype, torch.float32)
        self.assertEqual(tuple(kwargs["crfs_schedule_budget"].shape), ())
        self.assertEqual(kwargs["crfs_schedule_budget"].dtype, torch.float32)
        self.assertEqual(float(kwargs["crfs_schedule_budget"]), 0.0)
        self.assertEqual(noise.dtype, np.dtype(np.float32))

    def test_teacher_uses_eager_route_and_physical_trace_is_exact_reply(self) -> None:
        policy = self._policy()
        policy._input_transform = lambda value: value
        policy._sample_kwargs = {}
        policy._is_pytorch_model = True
        policy._output_transform = lambda value: {
            **value,
            "actions": np.asarray(value["actions"], dtype=np.float32) + np.float32(2.0),
        }
        normalized = torch.arange(
            self.ACTION_SHAPE[0] * self.ACTION_SHAPE[1], dtype=torch.float32
        ).reshape(1, *self.ACTION_SHAPE)
        seen_kwargs: dict = {}

        def eager_sample(device, observation, **kwargs):
            self.assertEqual(device, "cpu")
            self.assertEqual(tuple(observation.state.shape), (1, self.ACTION_SHAPE[1]))
            seen_kwargs.update(kwargs)
            return normalized.clone(), {"final_normalized": normalized.clone()}

        policy._sample_actions_crfs = eager_sample
        policy._sample_actions = lambda *args, **kwargs: self.fail("inverse teacher used baseline route")
        observation = {
            "state": np.zeros((self.ACTION_SHAPE[1],), dtype=np.float32),
            "__crfs__": self._teacher_controls(),
        }
        with mock.patch.object(
            policy_module._model.Observation,
            "from_dict",
            side_effect=lambda value: SimpleNamespace(state=value["state"]),
        ):
            result = policy.infer(observation)

        self.assertEqual(seen_kwargs["crfs_intervention_mode"], "inverse_flow_teacher")
        self.assertIn("crfs_inverse_target", seen_kwargs)
        self.assertIn("crfs_inverse_budget", seen_kwargs)
        self.assertIn("crfs_model_to_physical_scale", seen_kwargs)
        np.testing.assert_array_equal(
            result["crfs_trace"]["final_normalized_physical"],
            result["actions"],
        )

    def test_teacher_rejects_signed_zero_mismatch_between_trace_and_reply(self) -> None:
        policy = self._policy()
        policy._input_transform = lambda value: value
        policy._sample_kwargs = {}
        policy._is_pytorch_model = True
        transform_calls = 0

        def output_transform(value):
            nonlocal transform_calls
            transform_calls += 1
            actions = np.asarray(value["actions"], dtype=np.float32).copy()
            actions[0, 0] = -0.0 if transform_calls == 1 else 0.0
            return {**value, "actions": actions}

        policy._output_transform = output_transform
        normalized = torch.zeros((1, *self.ACTION_SHAPE), dtype=torch.float32)
        policy._sample_actions_crfs = lambda *_args, **_kwargs: (
            normalized.clone(),
            {"final_normalized": normalized.clone()},
        )
        policy._sample_actions = lambda *args, **kwargs: self.fail(
            "inverse teacher used baseline route"
        )
        observation = {
            "state": np.zeros((self.ACTION_SHAPE[1],), dtype=np.float32),
            "__crfs__": self._teacher_controls(),
        }
        with mock.patch.object(
            policy_module._model.Observation,
            "from_dict",
            side_effect=lambda value: SimpleNamespace(state=value["state"]),
        ):
            with self.assertRaisesRegex(RuntimeError, "not byte-exact"):
                policy.infer(observation)

    def test_schedule_rejects_mixed_shape_dtype_space_nonfinite_and_budget(self) -> None:
        for mutation, message in (
            (lambda value: value.__setitem__("target", np.zeros(self.ACTION_SHAPE)), "unsupported"),
            (lambda value: value.__setitem__("schedule", value["schedule"].astype(np.float64)), "float32"),
            (lambda value: value.__setitem__("schedule", value["schedule"][0]), "unbatched shape"),
            (lambda value: value["schedule"].__setitem__((0, 0, 0), np.inf), "nonfinite"),
            (lambda value: value.__setitem__("schedule_space", "physical"), "schedule_space='model'"),
            (lambda value: value.__setitem__("model_l2_path_budget", 0.1), "preserve float32 dtype"),
            (
                lambda value: value.__setitem__(
                    "model_l2_path_budget", np.asarray(-1.0, dtype=np.float32)
                ),
                "finite and nonnegative",
            ),
            (
                lambda value: value.__setitem__(
                    "model_l2_path_budget", np.asarray(np.nan, dtype=np.float32)
                ),
                "finite and nonnegative",
            ),
            (
                lambda value: value.__setitem__(
                    "model_l2_path_budget", np.asarray([0.1], dtype=np.float32)
                ),
                "must be scalar",
            ),
        ):
            with self.subTest(message=message):
                controls = self._schedule_controls()
                mutation(controls)
                with self.assertRaisesRegex(ValueError, message):
                    self._policy()._residual_schedule_sample_kwargs(
                        controls,
                        controls["noise"],
                        noise_argument_supplied=False,
                    )


if __name__ == "__main__":
    unittest.main()
