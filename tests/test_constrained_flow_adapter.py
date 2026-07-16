from __future__ import annotations

import ast
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from dataclasses import fields
import hashlib
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
ADAPTER_PATH = ROOT / "openpi/src/openpi/policies/crfs_constrained_flow_adapter.py"
SERVER_PATH = ROOT / "openpi/scripts/serve_cfs_policy.py"
HISTORICAL_SOLVER_PATH = ROOT / "openpi/src/openpi/models_pytorch/crfs_inverse_control.py"


class ConstrainedFlowAdapterStructuralTest(unittest.TestCase):
    def test_adapter_and_separate_server_parse(self) -> None:
        adapter_source = ADAPTER_PATH.read_text(encoding="utf-8")
        server_source = SERVER_PATH.read_text(encoding="utf-8")
        ast.parse(adapter_source)
        ast.parse(server_source)
        self.assertIn('ContextVar(', adapter_source)
        self.assertIn('_EXPERIMENT_ARM = "linearized_warm_start"', adapter_source)
        self.assertIn("install_constrained_flow_hooks()", server_source)
        self.assertIn("ConstrainedFlowPolicyAdapter(policy)", server_source)
        self.assertIn("scripts import serve_policy", server_source)

    def test_historical_solver_remains_byte_preserved(self) -> None:
        digest = hashlib.sha256(HISTORICAL_SOLVER_PATH.read_bytes()).hexdigest()
        self.assertEqual(
            digest,
            "965082822466774e0a86eeb6f2d178c9e5f3d6d02bca5090c4e1fcc77bc52aa8",
        )

    def test_adapter_has_explicit_fail_closed_and_no_certificate_contract(self) -> None:
        source = ADAPTER_PATH.read_text(encoding="utf-8")
        for fragment in (
            "already installed",
            "must invoke the inverse solver exactly once",
            "must inject exactly once",
            "missing its linearized diagnostic",
            "projection is not bitwise idempotent",
            "infeasibility_certificate=False",
            "nonlinear_feasibility_certificate=False",
        ):
            self.assertIn(fragment, source)

    def test_finite_difference_terminal_wire_contract_is_exact_and_opt_in(self) -> None:
        source = ADAPTER_PATH.read_text(encoding="utf-8")
        for fragment in (
            '_TERMINAL_KEY = "__crfs_terminal__"',
            '_FINITE_DIFFERENCE_REJECTION_KIND = "finite_difference_rejection"',
            '"AUTOGRAD_JACOBIAN_FD_REJECTED"',
            '"jacobian_completed": True',
            '"finite_difference_completed": True',
            '"fista_started": False',
            '"candidate_created": False',
            '"arm_b_nonlinear_replay_executed": False',
            '"arm_c_refinement_executed": False',
            "except _linearized_control.FiniteDifferenceValidationError as error:",
        ):
            self.assertIn(fragment, source)


IMPORT_ERROR: Exception | None = None
try:
    import numpy as np
    import torch

    sys.path.insert(0, str(ROOT / "openpi/src"))
    from openpi.policies import crfs_constrained_flow_adapter as adapter
except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover - dependency-free local gate.
    IMPORT_ERROR = exc


@unittest.skipIf(
    IMPORT_ERROR is not None,
    f"constrained-flow adapter runtime checks require OpenPI dependencies: {IMPORT_ERROR}",
)
class ConstrainedFlowAdapterRuntimeTest(unittest.TestCase):
    @dataclass(frozen=True)
    class _FiniteDifference:
        passed: bool

    @dataclass(frozen=True)
    class _LinearResult:
        active_increments: "torch.Tensor"
        marker: int
        jacobian_valid: bool = True
        finite_difference: object = None
        optimality_certificate: bool = False
        infeasibility_certificate: bool = False

        def __post_init__(self):
            if self.finite_difference is None:
                object.__setattr__(
                    self,
                    "finite_difference",
                    ConstrainedFlowAdapterRuntimeTest._FiniteDifference(passed=True),
                )

    @classmethod
    def setUpClass(cls) -> None:
        if not adapter._INSTALLED:
            adapter.install_constrained_flow_hooks()

    class _PassthroughPolicy:
        metadata = {"name": "fake"}

        def __init__(self, result):
            self.result = result
            self.seen = None
            self.seen_noise = None

        def infer(self, obs, *, noise=None):
            self.seen = obs
            self.seen_noise = noise
            return self.result

    class _SolverPolicy:
        metadata = {"name": "fake"}

        def __init__(self):
            self.seen = None

        def infer(self, obs, *, noise=None):
            del noise
            self.seen = obs
            label = int(obs["label"])
            initial = torch.zeros((1,), dtype=torch.float32)
            result = adapter._inverse_control.solve_inverse_control(
                initial,
                torch.full_like(initial, float(label)),
                lambda *_args: initial,
                control_mask=torch.ones_like(initial, dtype=torch.bool),
                target_mask=torch.ones_like(initial, dtype=torch.bool),
                model_to_physical_scale=torch.ones_like(initial),
                control_budget=torch.tensor(1.0, dtype=torch.float32),
                config=adapter._inverse_control.InverseControlConfig(),
            )
            return {
                "actions": result,
                "crfs_trace": {"ordinary_trace": np.asarray([label], dtype=np.int64)},
            }

    @staticmethod
    def _request(label: int = 1) -> dict:
        return {
            "label": label,
            "__crfs__": {
                "experiment_arm": "linearized_warm_start",
                "intervention_mode": "inverse_flow_teacher",
            },
        }

    def _patch_fake_solver(self, *, barrier: threading.Barrier | None = None):
        projected_inputs: list[torch.Tensor] = []

        def linearized(initial_state, target, _velocity_fn, **_kwargs):
            if barrier is not None:
                barrier.wait(timeout=5.0)
            label = int(target.item())
            candidate = torch.full((5, *initial_state.shape), float(label), dtype=initial_state.dtype)
            return self._LinearResult(active_increments=candidate, marker=label)

        def projector(increments, _control_mask, _budget, _config):
            projected_inputs.append(increments.detach().clone())
            return increments.detach().clone()

        def historical(initial_state, _target, _velocity_fn, **kwargs):
            historical_initializer = torch.full(
                (5, *initial_state.shape),
                -1.0,
                dtype=initial_state.dtype,
            )
            first = adapter._inverse_control._project_increments(
                historical_initializer,
                kwargs["control_mask"],
                kwargs["control_budget"],
                kwargs["config"],
            )
            update = adapter._inverse_control._project_increments(
                torch.full_like(historical_initializer, 9.0),
                kwargs["control_mask"],
                kwargs["control_budget"],
                kwargs["config"],
            )
            self.assertTrue(torch.equal(update, torch.full_like(update, 9.0)))
            return first

        return (
            projected_inputs,
            mock.patch.object(adapter._linearized_control, "solve_linearized_control", side_effect=linearized),
            mock.patch.object(adapter, "_ORIGINAL_PROJECT_INCREMENTS", side_effect=projector),
            mock.patch.object(adapter, "_ORIGINAL_SOLVE_INVERSE_CONTROL", side_effect=historical),
        )

    @staticmethod
    def _finite_difference_rejection():
        diagnostics = adapter._linearized_control.FiniteDifferenceDiagnostics(
            directions=torch.zeros((3, 75), dtype=torch.float32),
            epsilon_values=torch.tensor([1.0e-4, 5.0e-5], dtype=torch.float32),
            plus_target_physical=torch.zeros((3, 2, 35), dtype=torch.float32),
            minus_target_physical=torch.zeros((3, 2, 35), dtype=torch.float32),
            autograd_directional_derivatives=torch.zeros((3, 35), dtype=torch.float32),
            central_directional_derivatives=torch.zeros((3, 2, 35), dtype=torch.float32),
            absolute_l2_errors=torch.ones((3, 2), dtype=torch.float32),
            relative_l2_errors=torch.ones((3, 2), dtype=torch.float32),
            checks_passed=torch.zeros((3, 2), dtype=torch.bool),
            directions_passed=torch.zeros((3,), dtype=torch.bool),
            passed=False,
            relative_l2_tolerance=0.10,
            absolute_l2_tolerance=1.0e-3,
        )
        return adapter._linearized_control.FiniteDifferenceValidationError(
            jacobian=torch.zeros((35, 75), dtype=torch.float32),
            control_budget=torch.tensor(0.025, dtype=torch.float32),
            finite_difference=diagnostics,
        )

    def test_default_request_delegates_exact_objects_and_actions(self) -> None:
        output = {"actions": object()}
        ordinary = self._PassthroughPolicy(output)
        wrapped = adapter.ConstrainedFlowPolicyAdapter(ordinary)
        obs = {"state": object(), "__crfs__": {"intervention_mode": "none"}}
        noise = np.zeros((1,), dtype=np.float32)
        returned = wrapped.infer(obs, noise=noise)
        self.assertIs(returned, output)
        self.assertIs(ordinary.seen, obs)
        self.assertIs(ordinary.seen_noise, noise)
        self.assertIs(returned["actions"], output["actions"])
        self.assertIsNone(adapter._REQUEST_CONTEXT.get())

    def test_default_request_does_not_convert_typed_failure_to_terminal_data(self) -> None:
        rejection = self._finite_difference_rejection()

        class RaisingPolicy:
            metadata = {"name": "fake"}

            def infer(self, _obs):
                raise rejection

        wrapped = adapter.ConstrainedFlowPolicyAdapter(RaisingPolicy())
        with self.assertRaises(adapter._linearized_control.FiniteDifferenceValidationError) as caught:
            wrapped.infer({"__crfs__": {"intervention_mode": "none"}})
        self.assertIs(caught.exception, rejection)
        self.assertIsNone(adapter._REQUEST_CONTEXT.get())

    def test_hooks_delegate_no_context_calls_to_saved_historical_functions(self) -> None:
        initial = torch.zeros((1,), dtype=torch.float32)
        mask = torch.ones_like(initial, dtype=torch.bool)
        budget = torch.tensor(1.0, dtype=torch.float32)
        config = adapter._inverse_control.InverseControlConfig()
        sentinel = object()
        with mock.patch.object(
            adapter,
            "_ORIGINAL_SOLVE_INVERSE_CONTROL",
            return_value=sentinel,
        ) as historical:
            result = adapter._inverse_control.solve_inverse_control(
                initial,
                initial,
                lambda *_args: initial,
                control_mask=mask,
                target_mask=mask,
                model_to_physical_scale=torch.ones_like(initial),
                control_budget=budget,
                config=config,
            )
        self.assertIs(result, sentinel)
        self.assertEqual(historical.call_count, 1)

        increments = torch.zeros((5, 1), dtype=torch.float32)
        with mock.patch.object(
            adapter,
            "_ORIGINAL_PROJECT_INCREMENTS",
            return_value=increments,
        ) as historical_projector:
            projected = adapter._inverse_control._project_increments(
                increments,
                mask,
                budget,
                config,
            )
        self.assertIs(projected, increments)
        historical_projector.assert_called_once_with(increments, mask, budget, config)
        self.assertIsNone(adapter._REQUEST_CONTEXT.get())

    def test_unknown_arm_fails_before_policy_inference(self) -> None:
        ordinary = self._PassthroughPolicy({})
        wrapped = adapter.ConstrainedFlowPolicyAdapter(ordinary)
        obs = {"__crfs__": {"experiment_arm": "unregistered"}}
        with self.assertRaisesRegex(ValueError, "Unsupported __crfs__.experiment_arm"):
            wrapped.infer(obs)
        self.assertIsNone(ordinary.seen)

    def test_known_arm_injects_only_first_projection_and_adds_diagnostic(self) -> None:
        projected, patch_linear, patch_project, patch_solve = self._patch_fake_solver()
        policy = self._SolverPolicy()
        request = self._request(label=3)
        with patch_linear, patch_project, patch_solve:
            result = adapter.ConstrainedFlowPolicyAdapter(policy).infer(request)

        self.assertEqual(len(projected), 2)
        torch.testing.assert_close(projected[0], torch.full_like(projected[0], 3.0))
        torch.testing.assert_close(projected[1], torch.full_like(projected[1], 9.0))
        torch.testing.assert_close(result["actions"], torch.full_like(result["actions"], 3.0))
        self.assertNotIn("experiment_arm", policy.seen["__crfs__"])
        self.assertEqual(request["__crfs__"]["experiment_arm"], "linearized_warm_start")
        diagnostic = result["crfs_trace"]["linearized_warm_start"]
        self.assertEqual(diagnostic["marker"], 3)
        self.assertEqual(diagnostic["adapter_solve_calls"], 1)
        self.assertEqual(diagnostic["adapter_injection_count"], 1)
        self.assertEqual(diagnostic["adapter_projection_idempotence_checks"], 1)
        self.assertTrue(diagnostic["adapter_projection_idempotence_exact"])
        self.assertFalse(diagnostic["infeasibility_certificate"])
        self.assertFalse(diagnostic["nonlinear_feasibility_certificate"])
        self.assertIsNone(adapter._REQUEST_CONTEXT.get())

    def test_missing_projection_injection_fails_closed(self) -> None:
        candidate = self._LinearResult(
            active_increments=torch.zeros((5, 1), dtype=torch.float32),
            marker=1,
        )

        def no_projection(*_args, **_kwargs):
            return object()

        with (
            mock.patch.object(adapter._linearized_control, "solve_linearized_control", return_value=candidate),
            mock.patch.object(adapter, "_ORIGINAL_SOLVE_INVERSE_CONTROL", side_effect=no_projection),
        ):
            with self.assertRaisesRegex(RuntimeError, "inject exactly once"):
                adapter.ConstrainedFlowPolicyAdapter(self._SolverPolicy()).infer(self._request())
        self.assertIsNone(adapter._REQUEST_CONTEXT.get())

    def test_missing_linearized_diagnostic_fails_closed(self) -> None:
        invalid_result = SimpleNamespace(
            active_increments=torch.zeros((5, 1), dtype=torch.float32),
            jacobian_valid=True,
            finite_difference=self._FiniteDifference(passed=True),
        )
        with mock.patch.object(
            adapter._linearized_control,
            "solve_linearized_control",
            return_value=invalid_result,
        ):
            with self.assertRaisesRegex(RuntimeError, "required dataclass diagnostic"):
                adapter.ConstrainedFlowPolicyAdapter(self._SolverPolicy()).infer(self._request())
        self.assertIsNone(adapter._REQUEST_CONTEXT.get())

    def test_failed_registered_jacobian_validation_cannot_be_injected(self) -> None:
        invalid = self._LinearResult(
            active_increments=torch.zeros((5, 1), dtype=torch.float32),
            marker=1,
            jacobian_valid=False,
            finite_difference=self._FiniteDifference(passed=False),
        )
        with mock.patch.object(
            adapter._linearized_control,
            "solve_linearized_control",
            return_value=invalid,
        ):
            with self.assertRaisesRegex(RuntimeError, "Jacobian validation did not pass"):
                adapter.ConstrainedFlowPolicyAdapter(self._SolverPolicy()).infer(
                    self._request()
                )
        self.assertIsNone(adapter._REQUEST_CONTEXT.get())

    def test_typed_finite_difference_rejection_returns_strict_terminal_only(self) -> None:
        rejection = self._finite_difference_rejection()
        with (
            mock.patch.object(
                adapter._linearized_control,
                "solve_linearized_control",
                side_effect=rejection,
            ) as linearized,
            mock.patch.object(adapter, "_ORIGINAL_PROJECT_INCREMENTS") as projector,
            mock.patch.object(adapter, "_ORIGINAL_SOLVE_INVERSE_CONTROL") as refinement,
        ):
            result = adapter.ConstrainedFlowPolicyAdapter(self._SolverPolicy()).infer(
                self._request()
            )

        linearized.assert_called_once()
        projector.assert_not_called()
        refinement.assert_not_called()
        self.assertEqual(set(result), {"__crfs_terminal__"})
        terminal = result["__crfs_terminal__"]
        self.assertEqual(
            set(terminal),
            {
                "kind",
                "reason_code",
                "jacobian",
                "budget_float32",
                "finite_difference",
                "execution_boundary",
            },
        )
        self.assertEqual(terminal["kind"], "finite_difference_rejection")
        self.assertEqual(
            terminal["reason_code"],
            "AUTOGRAD_JACOBIAN_FD_REJECTED",
        )
        self.assertEqual(terminal["jacobian"].shape, (35, 75))
        self.assertEqual(terminal["jacobian"].dtype, np.float32)
        self.assertEqual(terminal["budget_float32"].shape, ())
        self.assertEqual(terminal["budget_float32"].dtype, np.float32)
        self.assertEqual(
            set(terminal["finite_difference"]),
            {field.name for field in fields(adapter._linearized_control.FiniteDifferenceDiagnostics)},
        )
        self.assertEqual(
            terminal["execution_boundary"],
            {
                "jacobian_completed": True,
                "finite_difference_completed": True,
                "fista_started": False,
                "candidate_created": False,
                "arm_b_nonlinear_replay_executed": False,
                "arm_c_refinement_executed": False,
            },
        )
        self.assertIsNone(adapter._REQUEST_CONTEXT.get())

    def test_unexpected_linearized_exception_propagates_without_terminal_data(self) -> None:
        unexpected = RuntimeError("unexpected linearized failure")
        with (
            mock.patch.object(
                adapter._linearized_control,
                "solve_linearized_control",
                side_effect=unexpected,
            ),
            mock.patch.object(adapter, "_ORIGINAL_PROJECT_INCREMENTS") as projector,
            mock.patch.object(adapter, "_ORIGINAL_SOLVE_INVERSE_CONTROL") as refinement,
        ):
            with self.assertRaises(RuntimeError) as caught:
                adapter.ConstrainedFlowPolicyAdapter(self._SolverPolicy()).infer(
                    self._request()
                )
        self.assertIs(caught.exception, unexpected)
        projector.assert_not_called()
        refinement.assert_not_called()
        self.assertIsNone(adapter._REQUEST_CONTEXT.get())

    def test_comparison_requires_explicit_historical_config_and_budget(self) -> None:
        class MissingContractPolicy:
            metadata = {}

            def __init__(self, *, config, budget):
                self.config = config
                self.budget = budget

            def infer(self, _obs):
                initial = torch.zeros((1,), dtype=torch.float32)
                mask = torch.ones_like(initial, dtype=torch.bool)
                adapter._inverse_control.solve_inverse_control(
                    initial,
                    initial,
                    lambda *_args: initial,
                    control_mask=mask,
                    target_mask=mask,
                    model_to_physical_scale=torch.ones_like(initial),
                    control_budget=self.budget,
                    config=self.config,
                )
                return {"actions": initial, "crfs_trace": {}}

        for config, budget, message in (
            (None, torch.tensor(1.0, dtype=torch.float32), "explicit InverseControlConfig"),
            (adapter._inverse_control.InverseControlConfig(), None, "explicit control budget"),
        ):
            with self.subTest(message=message):
                policy = MissingContractPolicy(config=config, budget=budget)
                with self.assertRaisesRegex(RuntimeError, message):
                    adapter.ConstrainedFlowPolicyAdapter(policy).infer(self._request())
                self.assertIsNone(adapter._REQUEST_CONTEXT.get())

    def test_context_isolated_between_concurrent_requests(self) -> None:
        barrier = threading.Barrier(2)
        _projected, patch_linear, patch_project, patch_solve = self._patch_fake_solver(barrier=barrier)

        def run(label: int):
            policy = self._SolverPolicy()
            return adapter.ConstrainedFlowPolicyAdapter(policy).infer(self._request(label=label))

        with patch_linear, patch_project, patch_solve:
            with ThreadPoolExecutor(max_workers=2) as pool:
                first = pool.submit(run, 4)
                second = pool.submit(run, 7)
                results = [first.result(timeout=10.0), second.result(timeout=10.0)]

        for expected, result in zip((4, 7), results, strict=True):
            self.assertTrue(bool(torch.all(result["actions"] == float(expected)).item()))
            self.assertEqual(result["crfs_trace"]["linearized_warm_start"]["marker"], expected)
        self.assertIsNone(adapter._REQUEST_CONTEXT.get())

    def test_reinstallation_fails_closed(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "already installed"):
            adapter.install_constrained_flow_hooks()


if __name__ == "__main__":
    unittest.main()
