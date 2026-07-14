from __future__ import annotations

import ast
import math
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SAMPLER_PATH = ROOT / "openpi/src/openpi/models_pytorch/pi0_pytorch.py"
FIELD_PATH = ROOT / "openpi/src/openpi/models_pytorch/crfs_analytic.py"
POLICY_PATH = ROOT / "openpi/src/openpi/policies/policy.py"


class AnalyticTrajectoryFieldStructuralTest(unittest.TestCase):
    def test_mode_is_opt_in_and_default_policy_route_is_unchanged(self) -> None:
        sampler = SAMPLER_PATH.read_text(encoding="utf-8")
        policy = POLICY_PATH.read_text(encoding="utf-8")
        ast.parse(sampler)
        ast.parse(policy)
        self.assertIn('"analytic_trajectory_field"', sampler)
        self.assertIn('intervention_mode == "analytic_trajectory_field"', policy)
        self.assertIn("use_crfs_sampler = False", policy)
        self.assertIn("self._sample_actions_crfs if use_crfs_sampler else self._sample_actions", policy)
        self.assertIn("x_t = x_t + dt * v_t", sampler)

    def test_identity_gradient_sign_and_frozen_energy_are_explicit(self) -> None:
        source = FIELD_PATH.read_text(encoding="utf-8")
        ast.parse(source)
        self.assertIn("approximate_clean.detach().requires_grad_(True)", source)
        self.assertIn("F.softplus(margin_argument).square()", source)
        self.assertIn("SAFETY_MARGIN_M = 0.005", source)
        self.assertIn("SOFTPLUS_TAU_M = 0.005", source)
        self.assertIn("SAMPLES_PER_SEGMENT = 26", source)
        sampler = SAMPLER_PATH.read_text(encoding="utf-8")
        self.assertIn("v_t = v_t + guidance_velocity", sampler)
        self.assertIn("reverse-time Euler step below has dt < 0", sampler)

    def test_scope_budget_margin_stop_and_no_action_clipping_are_structural(self) -> None:
        field = FIELD_PATH.read_text(encoding="utf-8")
        sampler = SAMPLER_PATH.read_text(encoding="utf-8")
        self.assertIn("energy_gradient[:, :EXECUTED_ACTIONS, :TRANSLATION_DIMS]", field)
        self.assertIn("hard_min_clearance_m >= SAFETY_MARGIN_M", field)
        self.assertIn("gain = model_l2_path_budget / active_horizon", field)
        self.assertIn("analytic_active_horizon = (num_steps - crfs_intervention_step) / num_steps", sampler)
        self.assertNotIn("torch.clamp(x_t", sampler)
        self.assertNotIn("x_t.clamp", sampler)

    def test_reserved_envelope_is_strict_and_affine_is_policy_owned(self) -> None:
        source = POLICY_PATH.read_text(encoding="utf-8")
        self.assertIn('unknown = set(controls) - required - optional', source)
        self.assertIn('"branch_eef_center_m"', source)
        self.assertIn('"response_matrix_m_per_action"', source)
        self.assertIn('"obstacle_rotations_world"', source)
        self.assertIn('"model_l2_path_budget"', source)
        self.assertIn("action_offset_xyz, action_scale_xyz = self._action_xyz_affine()", source)
        self.assertIn("offset = q01 + scale", source)
        self.assertIn("scale = xyz(self._action_norm_stats.std", source)


TORCH_IMPORT_ERROR: Exception | None = None
try:
    import torch

    sys.path.insert(0, str(ROOT / "openpi/src"))
    from openpi.models_pytorch.crfs_analytic import analytic_trajectory_field
    from openpi.models_pytorch.crfs_analytic import scale_field_to_velocity
except ModuleNotFoundError as exc:  # pragma: no cover - dependency-free local gate.
    TORCH_IMPORT_ERROR = exc


@unittest.skipIf(TORCH_IMPORT_ERROR is not None, f"analytic field runtime tests require PyTorch: {TORCH_IMPORT_ERROR}")
class AnalyticTrajectoryFieldRuntimeTest(unittest.TestCase):
    ACTION_SHAPE = (1, 10, 32)

    def _field(self, *, branch_x: float = 0.12, approximate_clean=None):
        if approximate_clean is None:
            approximate_clean = torch.zeros(self.ACTION_SHAPE, dtype=torch.float32)
        return analytic_trajectory_field(
            approximate_clean,
            action_offset_xyz=torch.zeros(3, dtype=torch.float32),
            action_scale_xyz=torch.ones(3, dtype=torch.float32),
            branch_eef_center_m=torch.tensor([branch_x, 0.0, 0.0], dtype=torch.float32),
            response_matrix_m_per_action=torch.eye(3, dtype=torch.float32),
            obstacle_centers_m=torch.zeros((1, 3), dtype=torch.float32),
            obstacle_rotations_world=torch.eye(3, dtype=torch.float32).reshape(1, 3, 3),
            obstacle_half_sizes_m=torch.tensor([[0.1, 0.1, 0.1]], dtype=torch.float32),
            eef_radius_m=torch.tensor(0.02, dtype=torch.float32),
        )

    def test_energy_gradient_sign_and_zero_outside_first_five_xyz(self) -> None:
        field = self._field()
        self.assertFalse(bool(field["margin_satisfied"].item()))
        self.assertTrue(bool(field["applied"].item()))
        gradient = field["normalized_energy_gradient"]
        self.assertTrue(bool((gradient[0, :5, 0] < 0.0).all().item()))
        self.assertTrue(torch.equal(gradient[0, :5, 1:3], torch.zeros_like(gradient[0, :5, 1:3])))
        self.assertTrue(torch.equal(gradient[0, 5:, :], torch.zeros_like(gradient[0, 5:, :])))
        self.assertTrue(torch.equal(gradient[0, :5, 3:], torch.zeros_like(gradient[0, :5, 3:])))
        # dt is negative, so the final action displacement is -gradient.
        self.assertTrue(bool((-gradient[0, :5, 0] > 0.0).all().item()))
        epsilon = 1.0e-4
        lower_energy = self._field(approximate_clean=-epsilon * gradient)["energy"]
        higher_energy = self._field(approximate_clean=epsilon * gradient)["energy"]
        self.assertLess(float(lower_energy), float(field["energy"]))
        self.assertGreater(float(higher_energy), float(field["energy"]))

    def test_integrated_model_l2_budget_is_divided_over_active_horizon(self) -> None:
        field = self._field()
        budget = torch.tensor(0.3, dtype=torch.float32)
        velocity = scale_field_to_velocity(
            field["normalized_energy_gradient"],
            field["applied"],
            model_l2_path_budget=budget,
            active_horizon=0.5,
        )
        velocity_l2 = torch.linalg.vector_norm(velocity[0, :5, :3])
        self.assertAlmostEqual(float(velocity_l2), 0.6, places=6)
        self.assertAlmostEqual(float(5 * 0.1 * velocity_l2), 0.3, places=6)

    def test_hard_margin_stop_applies_no_velocity(self) -> None:
        field = self._field(branch_x=0.4)
        self.assertTrue(bool(field["margin_satisfied"].item()))
        self.assertFalse(bool(field["applied"].item()))
        velocity = scale_field_to_velocity(
            field["normalized_energy_gradient"],
            field["applied"],
            model_l2_path_budget=torch.tensor(0.3, dtype=torch.float32),
            active_horizon=0.5,
        )
        self.assertTrue(torch.equal(velocity, torch.zeros_like(velocity)))

    def test_rotated_obb_and_nonsymmetric_response_match_registered_conventions(self) -> None:
        approximate_clean = torch.zeros(self.ACTION_SHAPE, dtype=torch.float32)
        approximate_clean[0, :5, :3] = torch.tensor(
            [
                [0.08, -0.03, 0.04],
                [-0.02, 0.06, -0.01],
                [0.05, 0.02, 0.03],
                [-0.04, -0.01, 0.02],
                [0.01, 0.04, -0.02],
            ],
            dtype=torch.float32,
        )
        theta = math.radians(37.0)
        rotation_world = torch.tensor(
            [
                [math.cos(theta), -math.sin(theta), 0.0],
                [math.sin(theta), math.cos(theta), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        field = analytic_trajectory_field(
            approximate_clean,
            action_offset_xyz=torch.tensor([0.01, -0.02, 0.005], dtype=torch.float32),
            action_scale_xyz=torch.tensor([0.7, 1.1, 0.9], dtype=torch.float32),
            branch_eef_center_m=torch.tensor([0.16, -0.07, 0.11], dtype=torch.float32),
            response_matrix_m_per_action=torch.tensor(
                [
                    [0.012, 0.003, -0.002],
                    [-0.004, 0.009, 0.001],
                    [0.002, -0.005, 0.011],
                ],
                dtype=torch.float32,
            ),
            obstacle_centers_m=torch.tensor([[0.163, -0.071, 0.108]], dtype=torch.float32),
            obstacle_rotations_world=rotation_world.reshape(1, 3, 3),
            obstacle_half_sizes_m=torch.tensor([[0.018, 0.011, 0.014]], dtype=torch.float32),
            eef_radius_m=torch.tensor(0.006, dtype=torch.float32),
        )
        expected_centers = torch.tensor(
            [
                [0.1605509967, -0.0706999972, 0.1108480021],
                [0.1606490016, -0.0702740029, 0.1105659977],
                [0.1611309946, -0.0704040006, 0.1109979972],
                [0.1607759893, -0.0705880001, 0.1113699973],
                [0.1610779911, -0.0704530030, 0.1111409962],
            ],
            dtype=torch.float32,
        )
        torch.testing.assert_close(
            field["predicted_eef_centers_m"][0], expected_centers, rtol=0.0, atol=2e-8
        )
        self.assertAlmostEqual(float(field["hard_min_clearance_m"]), -0.0154064512, places=7)
        self.assertAlmostEqual(float(field["energy"]), 2137.35546875, places=3)
        self.assertTrue(bool(field["gradient_finite"].item()))
        self.assertTrue(bool(field["applied"].item()))


MODEL_IMPORT_ERROR: Exception | None = TORCH_IMPORT_ERROR
if MODEL_IMPORT_ERROR is None:
    try:
        from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
    except ModuleNotFoundError as exc:  # pragma: no cover - allocation dependency surface.
        MODEL_IMPORT_ERROR = exc


@unittest.skipIf(
    MODEL_IMPORT_ERROR is not None,
    f"sampler runtime test requires OpenPI dependencies: {MODEL_IMPORT_ERROR}",
)
class AnalyticTrajectorySamplerRuntimeTest(unittest.TestCase):
    def _sampler(self):
        sampler = object.__new__(PI0Pytorch)
        object.__setattr__(sampler, "config", SimpleNamespace(action_horizon=10, action_dim=32))
        object.__setattr__(
            sampler,
            "_preprocess_observation",
            lambda observation, train: (None, None, None, None, torch.zeros((1, 32), dtype=torch.float32)),
        )
        object.__setattr__(
            sampler,
            "embed_prefix",
            lambda *args: (
                torch.zeros((1, 1, 1), dtype=torch.float32),
                torch.ones((1, 1), dtype=torch.bool),
                torch.zeros((1, 1), dtype=torch.int64),
            ),
        )
        object.__setattr__(sampler, "_prepare_attention_masks_4d", lambda value: value)
        language_model = SimpleNamespace(config=SimpleNamespace(_attn_implementation="eager"))
        object.__setattr__(
            sampler,
            "paligemma_with_expert",
            SimpleNamespace(
                paligemma=SimpleNamespace(language_model=language_model),
                forward=lambda **kwargs: (None, object()),
            ),
        )
        object.__setattr__(sampler, "denoise_step", lambda *args: torch.full_like(args[3], 0.25))
        return sampler

    def _run_analytic(self, intervention_step: int):
        sampler = self._sampler()
        object.__setattr__(sampler, "denoise_step", lambda *args: torch.zeros_like(args[3]))
        return PI0Pytorch.sample_actions(
            sampler,
            "cpu",
            SimpleNamespace(state=torch.zeros((1, 32), dtype=torch.float32)),
            noise=torch.zeros((1, 10, 32), dtype=torch.float32),
            num_steps=10,
            crfs_intervention_step=intervention_step,
            crfs_intervention_mode="analytic_trajectory_field",
            crfs_return_trace=True,
            crfs_return_normalized_final=True,
            crfs_action_offset_xyz=torch.zeros(3, dtype=torch.float32),
            crfs_action_scale_xyz=torch.ones(3, dtype=torch.float32),
            crfs_branch_eef_center_m=torch.tensor([0.12, 0.0, 0.0], dtype=torch.float32),
            crfs_response_matrix_m_per_action=torch.eye(3, dtype=torch.float32),
            crfs_obstacle_centers_m=torch.zeros((1, 3), dtype=torch.float32),
            crfs_obstacle_rotations_world=torch.eye(3, dtype=torch.float32).reshape(1, 3, 3),
            crfs_obstacle_half_sizes_m=torch.tensor([[0.1, 0.1, 0.1]], dtype=torch.float32),
            crfs_eef_radius_m=torch.tensor(0.02, dtype=torch.float32),
            crfs_model_l2_path_budget=torch.tensor(0.001, dtype=torch.float32),
        )

    def test_no_control_path_executes_the_ordinary_euler_updates_exactly(self) -> None:
        noise = torch.zeros((1, 10, 32), dtype=torch.float32)
        output = PI0Pytorch.sample_actions(
            self._sampler(),
            "cpu",
            SimpleNamespace(state=torch.zeros((1, 32), dtype=torch.float32)),
            noise=noise,
            num_steps=10,
        )
        expected = noise
        dt = torch.tensor(-0.1, dtype=torch.float32)
        for _ in range(10):
            expected = expected + dt * torch.full_like(expected, 0.25)
        self.assertTrue(torch.equal(output, expected))

    def test_sampler_applies_sign_and_exact_integrated_budget_only_after_start(self) -> None:
        output, trace = self._run_analytic(5)
        self.assertTrue(torch.equal(trace["active"][0], torch.tensor([False] * 5 + [True] * 5)))
        torch.testing.assert_close(trace["integrated_field_l2"], torch.tensor([0.001]))
        self.assertEqual(int(torch.count_nonzero(trace["guidance_velocity_steps"][:, :, 5:, :])), 0)
        self.assertEqual(int(torch.count_nonzero(trace["guidance_velocity_steps"][:, :, :5, 3:])), 0)
        self.assertTrue(bool((output[0, :5, 0] > 0.0).all().item()))

    def test_early_start_preserves_the_first_ordinary_step(self) -> None:
        _, trace = self._run_analytic(1)
        self.assertTrue(torch.equal(trace["active"][0], torch.tensor([False] + [True] * 9)))
        self.assertEqual(int(torch.count_nonzero(trace["guidance_velocity_steps"][:, 0])), 0)
        torch.testing.assert_close(trace["integrated_field_l2"], torch.tensor([0.001]))


POLICY_IMPORT_ERROR: Exception | None = TORCH_IMPORT_ERROR
if POLICY_IMPORT_ERROR is None:
    try:
        import numpy as np

        from openpi.policies.policy import Policy
    except ModuleNotFoundError as exc:  # pragma: no cover - allocation dependency surface.
        POLICY_IMPORT_ERROR = exc


@unittest.skipIf(
    POLICY_IMPORT_ERROR is not None,
    f"policy runtime test requires OpenPI dependencies: {POLICY_IMPORT_ERROR}",
)
class AnalyticTrajectoryPolicyRuntimeTest(unittest.TestCase):
    def _policy(self, *, quantile: bool = False):
        policy = object.__new__(Policy)
        policy._model = SimpleNamespace(config=SimpleNamespace(action_horizon=10, action_dim=32))
        policy._pytorch_device = "cpu"
        policy._use_quantile_norm = quantile
        policy._action_norm_stats = SimpleNamespace(
            mean=np.asarray([1.0, 2.0, 3.0, 0.0], dtype=np.float32),
            std=np.asarray([0.1, 0.2, 0.3, 1.0], dtype=np.float32),
            q01=np.asarray([-1.0, -2.0, -3.0, 0.0], dtype=np.float32),
            q99=np.asarray([3.0, 4.0, 5.0, 1.0], dtype=np.float32),
        )
        return policy

    def _controls(self) -> dict:
        return {
            "noise": np.zeros((10, 32), dtype=np.float32),
            "intervention_mode": "analytic_trajectory_field",
            "intervention_step": 5,
            "return_trace": True,
            "model_l2_path_budget": 0.3,
            "branch_eef_center_m": [0.12, 0.0, 0.0],
            "response_matrix_m_per_action": np.eye(3).tolist(),
            "obstacle_centers_m": [[0.0, 0.0, 0.0]],
            "obstacle_rotations_world": [np.eye(3).reshape(-1).tolist()],
            "obstacle_half_sizes_m": [[0.1, 0.1, 0.1]],
            "eef_radius_m": 0.02,
        }

    def test_policy_owns_complete_affine_and_tensorizes_valid_controls(self) -> None:
        policy = self._policy()
        controls = self._controls()
        kwargs, noise = policy._analytic_field_sample_kwargs(
            controls,
            controls["noise"],
            noise_argument_supplied=False,
        )
        self.assertEqual(noise.dtype, np.dtype(np.float32))
        self.assertTrue(torch.equal(kwargs["crfs_action_offset_xyz"], torch.tensor([1.0, 2.0, 3.0])))
        torch.testing.assert_close(
            kwargs["crfs_action_scale_xyz"],
            torch.tensor([0.100001, 0.200001, 0.300001]),
        )
        self.assertEqual(tuple(kwargs["crfs_obstacle_rotations_world"].shape), (1, 3, 3))

    def test_quantile_affine_matches_complete_output_transform(self) -> None:
        offset, scale = self._policy(quantile=True)._action_xyz_affine()
        expected_scale = (np.asarray([4.0, 6.0, 8.0]) + 1.0e-6) / 2.0
        expected_offset = np.asarray([-1.0, -2.0, -3.0]) + expected_scale
        np.testing.assert_allclose(scale, expected_scale, rtol=0.0, atol=2.0e-7)
        np.testing.assert_allclose(offset, expected_offset, rtol=0.0, atol=2.0e-7)

    def test_policy_rejects_unknown_missing_and_improper_rotation_controls(self) -> None:
        for mutation, message in (
            (lambda value: value.__setitem__("softplus_tau_m", 0.01), "unsupported controls"),
            (lambda value: value.pop("model_l2_path_budget"), "missing required controls"),
            (
                lambda value: value.__setitem__("obstacle_rotations_world", [[[2.0, 0.0, 0.0]] * 3]),
                "proper orthonormal",
            ),
        ):
            with self.subTest(message=message):
                controls = self._controls()
                mutation(controls)
                with self.assertRaisesRegex(ValueError, message):
                    self._policy()._analytic_field_sample_kwargs(
                        controls,
                        controls.get("noise"),
                        noise_argument_supplied=False,
                    )


if __name__ == "__main__":
    unittest.main()
