from __future__ import annotations

import ast
import sys
from pathlib import Path
from types import MethodType
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]
SAMPLER_PATH = ROOT / "openpi/src/openpi/models_pytorch/pi0_pytorch.py"

IMPORT_ERROR: Exception | None = None
try:
    import torch

    sys.path.insert(0, str(ROOT / "openpi/src"))
    from openpi.models_pytorch.crfs_inverse_control import InverseControlConfig
    from openpi.models_pytorch.crfs_inverse_control import InverseControlStatus
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
except ImportError as exc:  # pragma: no cover - dependency-free local harness.
    IMPORT_ERROR = exc


class InverseFlowSamplerStructuralTest(unittest.TestCase):
    def test_modes_are_opt_in_and_default_surface_is_preserved(self) -> None:
        source = SAMPLER_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)
        sample = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "sample_actions"
        )
        self.assertIn('crfs_intervention_mode="none"', source)
        self.assertIn('"inverse_flow_teacher"', source)
        self.assertIn('"residual_schedule"', source)
        keyword_names = {argument.arg for argument in sample.args.kwonlyargs}
        self.assertTrue(
            {
                "crfs_inverse_target",
                "crfs_model_to_physical_scale",
                "crfs_inverse_config",
                "crfs_inverse_budget",
                "crfs_residual_schedule",
                "crfs_schedule_budget",
            }.issubset(keyword_names)
        )
        trace_literals = [
            node
            for node in ast.walk(sample)
            if isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Dict)
            and any(isinstance(target, ast.Name) and target.id == "crfs_trace" for target in node.targets)
        ]
        self.assertEqual(len(trace_literals), 2)

    def test_teacher_uses_solver_checkpoint_and_restores_parameters(self) -> None:
        source = SAMPLER_PATH.read_text(encoding="utf-8")
        self.assertIn("_inverse_control.solve_inverse_control(", source)
        self.assertIn("use_reentrant=False", source)
        self.assertIn("preserve_rng_state=False", source)
        self.assertIn("parameter.requires_grad_(requires_grad=False)", source)
        self.assertIn("finally:", source)
        self.assertIn("parameter.requires_grad_(requires_grad=requires_grad)", source)
        self.assertIn("parameter_grads_none_before", source)
        self.assertIn("parameter_grads_none_after", source)
        self.assertIn("max_iterations=128", source)
        self.assertIn("learning_rate=0.02", source)
        self.assertIn("constraint_slack_ulps=8", source)
        self.assertIn("stop_on_first_feasible=False", source)

    def test_schedule_is_validated_before_model_work_and_never_overwrites_terminal(self) -> None:
        source = SAMPLER_PATH.read_text(encoding="utf-8")
        validation = source.index("_inverse_control.validate_schedule_constraints(")
        model_work = source.index("self._preprocess_observation(observation, train=False)")
        self.assertLess(validation, model_work)
        self.assertIn("validated_schedule = crfs_residual_schedule.permute(1, 0, 2, 3)", source)
        self.assertIn("x_before_step = x_t", source)
        self.assertIn("x_t = x_t + dt * v_t", source)
        self.assertIn("torch.where(flow_control == 0, v_t, v_t + flow_control)", source)
        self.assertNotIn("x_t = crfs_inverse_target", source)
        self.assertNotIn("x_next = crfs_inverse_target", source)

    def test_trace_exposes_all_recurrence_leaves_and_fail_closed_status(self) -> None:
        source = SAMPLER_PATH.read_text(encoding="utf-8")
        required = (
            "step_index_steps",
            "time_steps",
            "x_t_steps",
            "v_base_steps",
            "control_velocity_steps",
            "total_velocity_steps",
            "control_increment_steps",
            "x_next_steps",
            "solver_status",
            "solver_converged",
            "solver_baseline_final",
            "solver_schedule",
            "schedule_budget",
            "source_control_budget",
            "realized_target_delta_norm",
            "schedule_path_length",
            "schedule_per_step_increment_l2",
            "fidelity_xyz_max_abs",
            "fidelity_xyz_rms",
            "fidelity_full_max_abs",
            "fidelity_full_rms",
            "canonical_replay_final",
            "model_to_physical_scale",
            "parameter_requires_grad_restored",
            "parameter_grads_none_before",
            "parameter_grads_none_after",
            "cuda_process_peak_allocated_bytes",
            "cuda_process_peak_reserved_bytes",
        )
        for name in required:
            self.assertIn(name, source)
        self.assertIn("teacher_candidate_valid = bool(", source)
        self.assertIn("flow_schedule_applied = False", source)


@unittest.skipIf(
    IMPORT_ERROR is not None,
    f"inverse-flow sampler runtime checks need OpenPI dependencies: {IMPORT_ERROR}",
)
class InverseFlowSamplerRuntimeTest(unittest.TestCase):
    ACTION_SHAPE = (5, 7)

    class _PrefixModel(torch.nn.Module if IMPORT_ERROR is None else object):
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
        sampler.contract_parameter = torch.nn.Parameter(torch.tensor(1.0))
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

    @staticmethod
    def _registered_config():
        return InverseControlConfig(
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

    def _teacher_kwargs(self, target):
        shape = (1, *self.ACTION_SHAPE)
        return dict(
            noise=torch.zeros(shape, dtype=torch.float32),
            num_steps=10,
            crfs_intervention_step=5,
            crfs_intervention_mode="inverse_flow_teacher",
            crfs_return_trace=True,
            crfs_return_normalized_final=True,
            crfs_inverse_target=target,
            crfs_model_to_physical_scale=torch.ones(shape, dtype=torch.float32),
            crfs_inverse_config=self._registered_config(),
            crfs_inverse_budget=torch.tensor(0.1, dtype=torch.float32),
        )

    def test_invalid_explicit_schedule_is_rejected_before_model_work(self) -> None:
        sampler = self._sampler()
        shape = (1, 10, *self.ACTION_SHAPE)
        schedule = torch.zeros(shape, dtype=torch.float32)
        schedule[0, 4, 0, 0] = 0.1
        with self.assertRaisesRegex(ValueError, "steps 0--4"):
            PI0Pytorch.sample_actions(
                sampler,
                "cpu",
                self._observation(),
                noise=torch.zeros((1, *self.ACTION_SHAPE), dtype=torch.float32),
                num_steps=10,
                crfs_intervention_step=5,
                crfs_intervention_mode="residual_schedule",
                crfs_return_trace=True,
                crfs_return_normalized_final=True,
                crfs_residual_schedule=schedule,
                crfs_schedule_budget=torch.tensor(0.1, dtype=torch.float32),
            )

    def test_explicit_schedule_replays_ten_exact_euler_steps(self) -> None:
        sampler = self._sampler()
        schedule = torch.zeros((1, 10, *self.ACTION_SHAPE), dtype=torch.float32)
        schedule[:, 5:, 0, 0] = -0.2
        final, trace = PI0Pytorch.sample_actions(
            sampler,
            "cpu",
            self._observation(),
            noise=torch.zeros((1, *self.ACTION_SHAPE), dtype=torch.float32),
            num_steps=10,
            crfs_intervention_step=5,
            crfs_intervention_mode="residual_schedule",
            crfs_return_trace=True,
            crfs_return_normalized_final=True,
            crfs_residual_schedule=schedule,
            crfs_schedule_budget=torch.tensor(0.1, dtype=torch.float32),
        )
        self.assertAlmostEqual(final[0, 0, 0].item(), 0.1, places=6)
        self.assertEqual(tuple(trace["x_t_steps"].shape), (1, 10, 5, 7))
        self.assertEqual(tuple(trace["x_next_steps"].shape), (1, 10, 5, 7))
        self.assertTrue(torch.equal(trace["control_velocity_steps"], schedule))
        self.assertTrue(
            torch.equal(trace["x_next_steps"][:, :-1], trace["x_t_steps"][:, 1:])
        )
        self.assertTrue(all(isinstance(value, torch.Tensor) for value in trace.values()))

    def test_teacher_converges_restores_parameters_and_records_contract(self) -> None:
        sampler = self._sampler()
        target = torch.zeros((1, *self.ACTION_SHAPE), dtype=torch.float32)
        target[0, 0, 0] = 0.1
        kwargs = self._teacher_kwargs(target)
        # Prove that the sampler passes the source-bound budget instead of
        # recomputing it from the rounded target subtraction.
        kwargs["crfs_inverse_budget"] = torch.tensor(0.11, dtype=torch.float32)
        final, trace = PI0Pytorch.sample_actions(
            sampler, "cpu", self._observation(), **kwargs
        )
        self.assertEqual(trace["solver_status"].item(), int(InverseControlStatus.CONVERGED))
        self.assertTrue(trace["solver_converged"].item())
        self.assertTrue(trace["control_valid"].item())
        self.assertTrue(trace["schedule_applied"].item())
        self.assertTrue(trace["parameter_requires_grad_restored"].item())
        self.assertTrue(trace["parameter_grads_none_before"].item())
        self.assertTrue(trace["parameter_grads_none_after"].item())
        self.assertTrue(sampler.contract_parameter.requires_grad)
        self.assertIsNone(sampler.contract_parameter.grad)
        self.assertTrue(torch.allclose(final, target, atol=1.0e-5, rtol=0.0))
        self.assertTrue(torch.equal(trace["canonical_replay_final"], final))
        self.assertEqual(tuple(trace["solver_schedule"].shape), (1, 10, 5, 7))
        self.assertEqual(tuple(trace["model_to_physical_scale"].shape), (1, 5, 7))
        self.assertAlmostEqual(trace["source_control_budget"].item(), 0.11, places=6)
        self.assertAlmostEqual(trace["schedule_budget"].item(), 0.11, places=6)
        self.assertAlmostEqual(trace["realized_target_delta_norm"].item(), 0.1, places=6)
        self.assertTrue(all(isinstance(value, torch.Tensor) for value in trace.values()))

    def test_nonconverged_teacher_is_diagnostic_only_and_replays_frozen(self) -> None:
        sampler = self._sampler(field_gain=5.0)
        target = torch.zeros((1, *self.ACTION_SHAPE), dtype=torch.float32)
        target[0, 0, 0] = 0.1
        final, trace = PI0Pytorch.sample_actions(
            sampler, "cpu", self._observation(), **self._teacher_kwargs(target)
        )
        self.assertEqual(trace["solver_status"].item(), int(InverseControlStatus.MAX_ITERATIONS))
        self.assertFalse(trace["solver_converged"].item())
        self.assertFalse(trace["control_valid"].item())
        self.assertFalse(trace["schedule_applied"].item())
        self.assertEqual(torch.count_nonzero(trace["control_velocity_steps"]).item(), 0)
        self.assertEqual(torch.count_nonzero(final).item(), 0)
        self.assertIn("solver_schedule", trace)
        self.assertGreater(torch.count_nonzero(trace["solver_schedule"]).item(), 0)


if __name__ == "__main__":
    unittest.main()
