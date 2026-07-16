from __future__ import annotations

import asyncio
import ast
import contextlib
import copy
from pathlib import Path
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
CANARY_PATH = ROOT / "main/crfs_oracle/r05a_constrained_flow_canary.py"
RUNNER_PATH = ROOT / "main/run_crfs_r05a_constrained_flow_canary.py"
LEGACY_PATH = ROOT / "main/crfs_oracle/r05a_canary.py"

IMPORT_ERROR: Exception | None = None
try:
    import numpy as np

    sys.path.insert(0, str(ROOT / "main"))
    from crfs_oracle import r05a_constrained_flow_canary as cfs
except (ImportError, ModuleNotFoundError) as exc:  # dependency-free local gate.
    IMPORT_ERROR = exc

try:
    import torch
except (ImportError, ModuleNotFoundError):
    torch = None


class ConstrainedFlowCanaryStructuralTest(unittest.TestCase):
    def test_new_files_parse_and_legacy_runner_is_unchanged_dependency(self) -> None:
        canary = CANARY_PATH.read_text(encoding="utf-8")
        runner = RUNNER_PATH.read_text(encoding="utf-8")
        ast.parse(canary)
        ast.parse(runner)
        self.assertIn("run_r05a_canary(", canary)
        self.assertIn("PairedConstrainedFlowClient", canary)
        self.assertIn('with_name("constrained-flow-payload.json")', canary)
        self.assertIn("--legacy-config", runner)
        self.assertIn("ready_to_run", runner)
        self.assertNotIn("environment.step(", canary)
        self.assertNotIn("env.step(", canary)

    def test_contract_names_all_three_arms_and_three_outcomes(self) -> None:
        source = CANARY_PATH.read_text(encoding="utf-8")
        for fragment in (
            "A_historical_run_b",
            "B_linearized_candidate",
            "C_linearized_then_historical_adam",
            "mechanism_pass",
            "frozen_method_negative",
            "apparatus_inconclusive",
            'EXPERIMENT_ARM = "linearized_warm_start"',
            '"intervention_mode": "residual_schedule"',
            '"finite_nonconvergence_is_infeasibility": False',
            '"infeasibility_certificate": False',
            'nested.get("selected_candidate_float64")',
            'nested.get("model_candidate_pre_projection")',
            'nested.get("model_candidate_post_projection")',
            'nested.get("jacobian_singular_values")',
            'nested.get("weighted_matrix_singular_values")',
            'finite_difference.get("passed")',
            '"finite_difference_all_directions_passed"',
            '"returned_actions_exact_audited_physical"',
        ):
            self.assertIn(fragment, source)

    def test_legacy_source_is_not_copied_into_new_canary(self) -> None:
        new_tree = ast.parse(CANARY_PATH.read_text(encoding="utf-8"))
        function_names = {
            node.name for node in ast.walk(new_tree) if isinstance(node, ast.FunctionDef)
        }
        self.assertNotIn("run_r05a_canary", function_names)
        self.assertNotIn("validate_flow_recurrence", function_names)
        self.assertNotIn("_teacher_summary", function_names)
        self.assertTrue(LEGACY_PATH.exists())


@unittest.skipIf(
    IMPORT_ERROR is not None,
    f"paired canary runtime checks require allocation dependencies: {IMPORT_ERROR}",
)
class PairedConstrainedFlowClientTest(unittest.TestCase):
    class _RecordingClient:
        def __init__(self) -> None:
            self.requests: list[dict] = []
            self.replies: list[dict] = []

        def infer(self, request):
            recorded = copy.deepcopy(dict(request))
            self.requests.append(recorded)
            reply = {
                "actions": np.full((10, 7), len(self.requests), dtype=np.float32),
                "crfs_trace": {"request_index": np.asarray(len(self.requests), dtype=np.int64)},
            }
            controls = recorded.get("__crfs__", {})
            if controls.get("intervention_mode") == "inverse_flow_teacher":
                reply["crfs_trace"].update(
                    solver_schedule=np.zeros((10, 10, 32), dtype=np.float32),
                    solver_internal_replay_final=np.zeros(
                        (10, 32), dtype=np.float32
                    ),
                )
            if controls.get("experiment_arm") == "linearized_warm_start":
                reply["crfs_trace"].update(
                    solver_schedule=np.zeros((10, 10, 32), dtype=np.float32),
                    linearized_warm_start={
                        "schedule": np.zeros((10, 1, 10, 32), dtype=np.float32)
                    },
                )
            self.replies.append(reply)
            return reply

    @staticmethod
    def _teacher_request(label: int = 0) -> dict:
        return {
            "observation": np.asarray([label], dtype=np.int64),
            "__crfs__": {
                "noise": np.zeros((10, 32), dtype=np.float32),
                "intervention_mode": "inverse_flow_teacher",
                "intervention_step": 5,
                "return_trace": True,
                "return_normalized_final": True,
                "target": np.ones((10, 32), dtype=np.float32),
                "target_space": "model",
                "model_l2_path_budget": np.float32(3.0),
                "solver_config": {"fixed": True},
            },
        }

    @staticmethod
    def _terminal_reply() -> dict:
        finite_difference = {
            "directions": np.ones((3, 75), dtype=np.float32),
            "epsilon_values": np.asarray([0.01, 0.005], dtype=np.float32),
            "plus_target_physical": np.zeros((3, 2, 35), dtype=np.float32),
            "minus_target_physical": np.zeros((3, 2, 35), dtype=np.float32),
            "autograd_directional_derivatives": np.zeros(
                (3, 35), dtype=np.float32
            ),
            "central_directional_derivatives": np.zeros(
                (3, 2, 35), dtype=np.float32
            ),
            "absolute_l2_errors": np.zeros((3, 2), dtype=np.float32),
            "relative_l2_errors": np.zeros((3, 2), dtype=np.float32),
            "checks_passed": np.zeros((3, 2), dtype=np.bool_),
            "directions_passed": np.zeros((3,), dtype=np.bool_),
            "passed": False,
            "relative_l2_tolerance": 0.10,
            "absolute_l2_tolerance": 1.0e-3,
        }
        return {
            cfs.TERMINAL_RESPONSE_KEY: {
                "kind": cfs.FINITE_DIFFERENCE_REJECTION_KIND,
                "reason_code": cfs.FINITE_DIFFERENCE_REJECTION_REASON,
                "jacobian": np.zeros((35, 75), dtype=np.float32),
                "budget_float32": np.asarray(3.0, dtype=np.float32),
                "finite_difference": finite_difference,
                "execution_boundary": {
                    "jacobian_completed": True,
                    "finite_difference_completed": True,
                    "fista_started": False,
                    "candidate_created": False,
                    "arm_b_nonlinear_replay_executed": False,
                    "arm_c_refinement_executed": False,
                },
            }
        }

    def test_teacher_returns_exact_ordinary_object_then_issues_paired_request(self) -> None:
        underlying = self._RecordingClient()
        wrapped = cfs.PairedConstrainedFlowClient(underlying)
        request = self._teacher_request()
        returned = wrapped.infer(request)
        self.assertIs(returned, underlying.replies[0])
        self.assertEqual(len(underlying.requests), 2)
        self.assertNotIn("experiment_arm", underlying.requests[0]["__crfs__"])
        self.assertEqual(
            underlying.requests[1]["__crfs__"]["experiment_arm"],
            "linearized_warm_start",
        )
        self.assertNotIn("experiment_arm", request["__crfs__"])
        self.assertEqual(len(wrapped.paired_calls), 1)
        self.assertIs(wrapped.paired_calls[0].ordinary_reply, returned)

    def test_non_teacher_request_is_single_exact_passthrough(self) -> None:
        underlying = self._RecordingClient()
        wrapped = cfs.PairedConstrainedFlowClient(underlying)
        request = {"__crfs__": {"intervention_mode": "none"}}
        returned = wrapped.infer(request)
        self.assertIs(returned, underlying.replies[0])
        self.assertEqual(underlying.requests, [request])
        self.assertEqual(wrapped.paired_calls, [])

    def test_terminal_fd_response_stops_before_duplicate_or_replay(self) -> None:
        sys.path.insert(0, str(ROOT / "openpi/src"))
        from openpi.serving import websocket_policy_server
        from openpi_client import msgpack_numpy

        self.assertEqual(
            Path(websocket_policy_server.__file__).resolve(),
            ROOT / "openpi/src/openpi/serving/websocket_policy_server.py",
        )
        self.assertEqual(
            Path(msgpack_numpy.__file__).resolve(),
            ROOT
            / "openpi/packages/openpi-client/src/openpi_client/msgpack_numpy.py",
        )
        policy_terminal = self._terminal_reply()

        class TransportPolicy:
            def infer(inner_self, request):
                if request["kind"] == "ordinary":
                    return {"actions": np.zeros((1,), dtype=np.float32)}
                return policy_terminal

        class FakeWebsocket:
            remote_address = ("127.0.0.1", 12345)

            def __init__(inner_self):
                packer = msgpack_numpy.Packer()
                inner_self.requests = [
                    packer.pack({"kind": "ordinary"}),
                    packer.pack({"kind": "terminal"}),
                ]
                inner_self.sent = []

            async def recv(inner_self):
                if inner_self.requests:
                    return inner_self.requests.pop(0)
                await asyncio.Future()

            async def send(inner_self, value):
                inner_self.sent.append(value)

        async def baseline_transport_roundtrip():
            server = websocket_policy_server.WebsocketPolicyServer(
                policy=TransportPolicy(), metadata={}
            )
            websocket = FakeWebsocket()
            task = asyncio.create_task(server._handler(websocket))
            async def wait_for_replies():
                while len(websocket.sent) < 3:
                    if task.done():
                        await task
                    await asyncio.sleep(0)

            try:
                await asyncio.wait_for(wait_for_replies(), timeout=5.0)
            finally:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            return (
                msgpack_numpy.unpackb(websocket.sent[1]),
                msgpack_numpy.unpackb(websocket.sent[2]),
            )

        ordinary, terminal = asyncio.run(baseline_transport_roundtrip())
        self.assertEqual(set(ordinary), {"actions", "server_timing"})
        self.assertEqual(set(ordinary["server_timing"]), {"infer_ms"})
        self.assertEqual(ordinary["actions"].dtype, np.dtype(np.float32))
        self.assertEqual(ordinary["actions"].shape, (1,))
        self.assertEqual(
            ordinary["actions"].tobytes(),
            np.zeros((1,), dtype=np.float32).tobytes(),
        )
        self.assertEqual(
            set(terminal),
            {cfs.TERMINAL_RESPONSE_KEY, "server_timing"},
        )
        self.assertEqual(
            set(terminal["server_timing"]),
            {"infer_ms", "prev_total_ms"},
        )
        first_reply = self._terminal_reply()
        first_reply["server_timing"] = {"infer_ms": 0.0}
        normalized_first_reply = cfs._terminal_policy_reply_from_transport(
            first_reply
        )
        self.assertEqual(set(normalized_first_reply), {cfs.TERMINAL_RESPONSE_KEY})
        self.assertIn("server_timing", first_reply)

        class TerminalClient(self._RecordingClient):
            def infer(inner_self, request):
                controls = request.get("__crfs__", {})
                if controls.get("experiment_arm") == cfs.EXPERIMENT_ARM:
                    inner_self.requests.append(copy.deepcopy(dict(request)))
                    inner_self.replies.append(terminal)
                    return terminal
                return super(TerminalClient, inner_self).infer(request)

        underlying = TerminalClient()
        wrapped = cfs.PairedConstrainedFlowClient(underlying)
        with self.assertRaises(cfs.ConstrainedFlowFiniteDifferenceRejection) as caught:
            wrapped.infer(self._teacher_request())
        self.assertEqual(len(underlying.requests), 2)
        self.assertEqual(wrapped.paired_calls, [])
        self.assertEqual(wrapped.canonical_replies, {})
        self.assertEqual(
            caught.exception.reason_code,
            cfs.FINITE_DIFFERENCE_REJECTION_REASON,
        )
        self.assertEqual(
            caught.exception.diagnostic["jacobian"]["dtype"], "float32"
        )
        self.assertEqual(
            caught.exception.diagnostic["budget_float32"]["shape"], []
        )
        self.assertFalse(caught.exception.execution_boundary["fista_started"])
        self.assertIn("server_timing", terminal)

        direct_terminal = self._terminal_reply()

        class DirectTerminalClient(self._RecordingClient):
            def infer(inner_self, request):
                controls = request.get("__crfs__", {})
                if controls.get("experiment_arm") == cfs.EXPERIMENT_ARM:
                    inner_self.requests.append(copy.deepcopy(dict(request)))
                    inner_self.replies.append(direct_terminal)
                    return direct_terminal
                return super(DirectTerminalClient, inner_self).infer(request)

        direct_underlying = DirectTerminalClient()
        direct_wrapped = cfs.PairedConstrainedFlowClient(direct_underlying)
        with self.assertRaises(cfs.ConstrainedFlowFiniteDifferenceRejection):
            direct_wrapped.infer(self._teacher_request())
        self.assertEqual(len(direct_underlying.requests), 2)
        self.assertEqual(direct_wrapped.paired_calls, [])
        self.assertEqual(direct_wrapped.canonical_replies, {})
        self.assertNotIn("server_timing", direct_terminal)

    def test_terminal_fd_response_is_strict_about_wire_evidence(self) -> None:
        mutations = []
        extra = self._terminal_reply()
        extra["unexpected"] = True
        mutations.append(extra)
        for timing in (
            None,
            {},
            {"prev_total_ms": 1.0},
            {"infer_ms": 1.0, "unexpected": 2.0},
            {"infer_ms": float("nan")},
            {"infer_ms": float("inf")},
            {"infer_ms": -1.0},
            {"infer_ms": True},
            {"infer_ms": 1},
            {"infer_ms": np.float64(1.0)},
        ):
            malformed_timing = self._terminal_reply()
            malformed_timing["server_timing"] = timing
            mutations.append(malformed_timing)
        wrong_shape = self._terminal_reply()
        wrong_shape[cfs.TERMINAL_RESPONSE_KEY]["jacobian"] = np.zeros(
            (34, 75), dtype=np.float32
        )
        mutations.append(wrong_shape)
        wrong_dtype = self._terminal_reply()
        wrong_dtype[cfs.TERMINAL_RESPONSE_KEY]["finite_difference"][
            "directions"
        ] = np.ones((3, 75), dtype=np.float64)
        mutations.append(wrong_dtype)
        nonfinite = self._terminal_reply()
        nonfinite[cfs.TERMINAL_RESPONSE_KEY]["finite_difference"][
            "plus_target_physical"
        ][0, 0, 0] = np.nan
        mutations.append(nonfinite)
        promoted = self._terminal_reply()
        promoted[cfs.TERMINAL_RESPONSE_KEY]["finite_difference"]["passed"] = True
        mutations.append(promoted)
        wrong_budget = self._terminal_reply()
        wrong_budget[cfs.TERMINAL_RESPONSE_KEY]["budget_float32"] = np.asarray(
            2.0, dtype=np.float32
        )
        mutations.append(wrong_budget)
        boundary = self._terminal_reply()
        boundary[cfs.TERMINAL_RESPONSE_KEY]["execution_boundary"][
            "fista_started"
        ] = True
        mutations.append(boundary)

        for terminal in mutations:
            with self.subTest(terminal=terminal):
                class TerminalClient(self._RecordingClient):
                    def infer(inner_self, request):
                        controls = request.get("__crfs__", {})
                        if controls.get("experiment_arm") == cfs.EXPERIMENT_ARM:
                            inner_self.requests.append(copy.deepcopy(dict(request)))
                            return terminal
                        return super(TerminalClient, inner_self).infer(request)

                underlying = TerminalClient()
                wrapped = cfs.PairedConstrainedFlowClient(underlying)
                with self.assertRaises(cfs.ConstrainedFlowCanaryError):
                    wrapped.infer(self._teacher_request())
                self.assertEqual(len(underlying.requests), 2)
                self.assertEqual(wrapped.paired_calls, [])
                self.assertEqual(wrapped.canonical_replies, {})

    def test_frozen_legacy_request_wrapper_preserves_typed_fd_diagnostic(self) -> None:
        from crfs_oracle import r05a_canary as legacy

        rejection = cfs._finite_difference_rejection_from_reply(
            self._terminal_reply(),
            expected_budget_float32=np.float32(3.0),
        )

        class RejectingClient:
            def infer(self, _request):
                raise rejection

        def wrapped_legacy_run(*_args, **_kwargs):
            # Exercise the actual frozen catch-all request boundary rather
            # than constructing its wrapper exception by hand.
            legacy._request(
                RejectingClient(),
                {"observation": np.zeros((1,), dtype=np.float32)},
                {"return_trace": True},
                require_trace=False,
            )
            raise AssertionError("legacy request unexpectedly returned")

        config = __import__("json").loads(
            (ROOT / "configs/experiments/r05a_constrained_flow_canary.json").read_text(
                encoding="utf-8"
            )
        )
        with mock.patch.object(
            cfs, "run_r05a_canary", side_effect=wrapped_legacy_run
        ):
            with self.assertRaises(
                cfs.ConstrainedFlowFiniteDifferenceRejection
            ) as caught:
                cfs.run_r05a_constrained_flow_canary(
                    {"case_id": cfs.CASE_ID},
                    config,
                    object(),
                    repo_root=ROOT,
                    input_manifest_sha256="0" * 64,
                    constrained_config_path=(
                        ROOT / "configs/experiments/r05a_constrained_flow_canary.json"
                    ),
                    legacy_config_path=(
                        ROOT / "configs/experiments/r05a_inverse_flow_canary.json"
                    ),
                    client=object(),
                )
        self.assertIs(caught.exception, rejection)
        self.assertIsNone(caught.exception.__cause__)
        self.assertIsNone(caught.exception.__context__)

    def test_more_than_two_legacy_teacher_calls_fail_closed(self) -> None:
        underlying = self._RecordingClient()
        wrapped = cfs.PairedConstrainedFlowClient(underlying)
        wrapped.infer(self._teacher_request(1))
        wrapped.infer(self._teacher_request(2))
        with self.assertRaisesRegex(cfs.ConstrainedFlowCanaryError, "more than two"):
            wrapped.infer(self._teacher_request(3))
        self.assertEqual(len(wrapped.paired_calls), 2)

    def test_replay_is_ordinary_residual_schedule_without_teacher_fields(self) -> None:
        underlying = self._RecordingClient()
        wrapped = cfs.PairedConstrainedFlowClient(underlying)
        wrapped.infer(self._teacher_request(1))
        wrapped.infer(self._teacher_request(1))
        schedule = np.zeros((10, 10, 32), dtype=np.float32)
        reply, elapsed = wrapped.replay(schedule, np.float32(3.0))
        self.assertIs(reply, underlying.replies[-1])
        self.assertGreaterEqual(elapsed, 0.0)
        controls = underlying.requests[-1]["__crfs__"]
        self.assertEqual(controls["intervention_mode"], "residual_schedule")
        self.assertEqual(controls["schedule_space"], "model")
        self.assertNotIn("experiment_arm", controls)
        self.assertNotIn("target", controls)
        self.assertNotIn("target_space", controls)
        self.assertNotIn("solver_config", controls)
        np.testing.assert_array_equal(controls["schedule"], schedule)

    def test_automatic_replays_precede_legacy_after_check_requests(self) -> None:
        underlying = self._RecordingClient()
        wrapped = cfs.PairedConstrainedFlowClient(underlying)
        wrapped.infer(self._teacher_request(1))
        wrapped.infer(self._teacher_request(1))
        modes = [request["__crfs__"]["intervention_mode"] for request in underlying.requests]
        self.assertEqual(
            modes,
            [
                "inverse_flow_teacher",
                "inverse_flow_teacher",
                "inverse_flow_teacher",
                "inverse_flow_teacher",
                "residual_schedule",
                "residual_schedule",
                "residual_schedule",
                "residual_schedule",
                "residual_schedule",
                "residual_schedule",
            ],
        )
        self.assertEqual(
            set(wrapped.canonical_replies),
            {
                f"{arm}:{replicate}"
                for arm in (
                    "A_historical_run_b",
                    "B_linearized_candidate",
                    "C_linearized_then_historical_adam",
                )
                for replicate in ("first", "duplicate")
            },
        )

@unittest.skipIf(
    IMPORT_ERROR is not None,
    f"paired canary runtime checks require allocation dependencies: {IMPORT_ERROR}",
)
class ConstrainedFlowValidationTest(unittest.TestCase):
    def test_four_physical_gates_are_independent(self) -> None:
        error = np.zeros((10, 32), dtype=np.float32)
        metrics = cfs._fidelity_metrics(error)
        self.assertTrue(metrics["passed"])
        self.assertEqual(set(metrics["checks"]), set(cfs.FIDELITY_LIMITS))

        # XYZ passes, but a non-XYZ channel violates the full max/RMS gates.
        error[:5, 3] = np.float32(0.10)
        metrics = cfs._fidelity_metrics(error)
        self.assertTrue(metrics["checks"]["xyz_max_abs"])
        self.assertTrue(metrics["checks"]["xyz_rms"])
        self.assertFalse(metrics["checks"]["full_max_abs"])
        self.assertFalse(metrics["checks"]["full_rms"])
        self.assertFalse(metrics["passed"])

    def test_status_rule_never_upgrades_invalid_apparatus(self) -> None:
        for arm_b in (False, True):
            for arm_c in (False, True):
                self.assertEqual(
                    cfs._classify_outcome(
                        apparatus_valid=False,
                        arm_b_nonlinear_pass=arm_b,
                        arm_c_nonlinear_pass=arm_c,
                    ),
                    "apparatus_inconclusive",
                )
        self.assertEqual(
            cfs._classify_outcome(
                apparatus_valid=True,
                arm_b_nonlinear_pass=False,
                arm_c_nonlinear_pass=False,
            ),
            "frozen_method_negative",
        )
        self.assertEqual(
            cfs._classify_outcome(
                apparatus_valid=True,
                arm_b_nonlinear_pass=True,
                arm_c_nonlinear_pass=False,
            ),
            "mechanism_pass",
        )

    def test_diagnostic_batch_axes_are_strict(self) -> None:
        schedule = np.zeros((10, 1, 10, 32), dtype=np.float32)
        self.assertEqual(cfs._diagnostic_schedule(schedule, name="test").shape, (10, 10, 32))
        with self.assertRaisesRegex(cfs.ConstrainedFlowCanaryError, "must have shape"):
            cfs._diagnostic_schedule(schedule[:, 0], name="test")
        with self.assertRaisesRegex(cfs.ConstrainedFlowCanaryError, "preserve float32"):
            cfs._diagnostic_schedule(schedule.astype(np.float64), name="test")
        batched = np.zeros((1, 10, 32), dtype=np.float32)
        self.assertEqual(
            cfs._batched_array(batched, name="test", shape=(10, 32)).shape,
            (10, 32),
        )
        with self.assertRaisesRegex(cfs.ConstrainedFlowCanaryError, "retained batch shape"):
            cfs._batched_array(batched[0], name="test", shape=(10, 32))

    def test_invalid_tree_and_nonfinite_values_fail_closed(self) -> None:
        with self.assertRaisesRegex(cfs.ConstrainedFlowCanaryError, "nonfinite"):
            cfs._tree_record(np.asarray([np.nan], dtype=np.float32))
        with self.assertRaisesRegex(cfs.ConstrainedFlowCanaryError, "finite numeric"):
            cfs._finite_array(
                np.asarray([[object()]], dtype=object), name="bad", shape=(1, 1)
            )

    @unittest.skipUnless(torch is not None, "PyTorch is unavailable")
    def test_independent_numpy_audit_reproduces_real_core_candidate(self) -> None:
        sys.path.insert(0, str(ROOT / "openpi/src"))
        from openpi.models_pytorch import crfs_inverse_control as legacy
        from openpi.models_pytorch import crfs_linearized_control as linear
        from openpi.policies import crfs_constrained_flow_adapter as adapter

        initial = torch.zeros((1, 10, 32), dtype=torch.float32)
        target = initial.clone()
        target[0, 0, 0] = 0.08
        target[0, 1, 1] = -0.04
        control_mask = legacy.first_five_xyz_mask_like(initial)
        target_mask = legacy.first_five_channels_mask_like(initial)

        def coupled_field(x_t, _time, _step):
            value = torch.zeros_like(x_t)
            value[..., 0] = 0.8 * x_t[..., 1] + 0.1 * x_t[..., 2]
            value[..., 1] = -0.5 * x_t[..., 0] + 0.2 * x_t[..., 2]
            value[..., 3] = 0.3 * x_t[..., 0]
            return value

        result = linear.solve_linearized_control(
            initial,
            target,
            coupled_field,
            control_mask=control_mask,
            target_mask=target_mask,
            model_to_physical_scale=torch.ones_like(initial),
            control_budget=torch.tensor(0.06, dtype=torch.float32),
            legacy_config=legacy.InverseControlConfig(),
        )
        nested = adapter._wire_value(result)
        nested.update(
            experiment_arm="linearized_warm_start",
            adapter_solve_calls=1,
            adapter_historical_projection_calls=129,
            adapter_injection_count=1,
            adapter_projection_idempotence_checks=1,
            adapter_projection_idempotence_exact=True,
            adapter_injected_initial_increments=np.asarray(
                result.active_increments.detach().cpu().numpy()
            ),
            adapter_only_historical_initialization_changed=True,
            optimality_certificate=False,
            infeasibility_certificate=False,
            nonlinear_feasibility_certificate=False,
        )
        summary, _schedule, _error = cfs._arm_b_summary(
            nested,
            budget=np.float32(0.06),
            source_target=target[0].detach().cpu().numpy(),
            source_baseline=result.baseline_final[0].detach().cpu().numpy(),
            source_scale=np.ones((10, 32), dtype=np.float32),
        )
        self.assertTrue(summary["passed"])
        self.assertTrue(all(summary["checks"].values()))
        self.assertLessEqual(
            summary["independent_linear_audit"]["candidate_relative_l2_difference"],
            1.0e-8,
        )

        tampered = copy.deepcopy(nested)
        tampered["finite_difference"]["autograd_directional_derivatives"] = np.full(
            (3, 35), 123.0, dtype=np.float32
        )
        tampered["finite_difference"]["central_directional_derivatives"] = np.full(
            (3, 2, 35), -456.0, dtype=np.float32
        )
        tampered_summary, _schedule, _error = cfs._arm_b_summary(
            tampered,
            budget=np.float32(0.06),
            source_target=target[0].detach().cpu().numpy(),
            source_baseline=result.baseline_final[0].detach().cpu().numpy(),
            source_scale=np.ones((10, 32), dtype=np.float32),
        )
        self.assertFalse(tampered_summary["passed"])
        self.assertFalse(
            tampered_summary["checks"][
                "finite_difference_autograd_products_recomputed"
            ]
        )
        self.assertFalse(
            tampered_summary["checks"][
                "finite_difference_central_products_recomputed"
            ]
        )

        source_kwargs = {
            "budget": np.float32(0.06),
            "source_target": target[0].detach().cpu().numpy(),
            "source_baseline": result.baseline_final[0].detach().cpu().numpy(),
            "source_scale": np.ones((10, 32), dtype=np.float32),
        }

        forged_linear = copy.deepcopy(nested)
        forged_linear["linear_fidelity_error"] = np.zeros_like(
            forged_linear["linear_fidelity_error"]
        )
        zero = np.asarray(0.0, dtype=np.float32)
        forged_linear["linear_metrics"] = {
            "xyz_max_abs": zero,
            "xyz_rms": zero,
            "full_max_abs": zero,
            "full_rms": zero,
            "feasible": np.asarray(True, dtype=np.bool_),
        }
        forged_summary, _schedule, _error = cfs._arm_b_summary(
            forged_linear, **source_kwargs
        )
        self.assertFalse(forged_summary["passed"])
        self.assertFalse(
            forged_summary["checks"][
                "linear_fidelity_error_source_reconstructed"
            ]
        )

        forged_baseline = copy.deepcopy(nested)
        forged_baseline["baseline_target_error"] = np.zeros(
            35, dtype=np.float32
        )
        forged_summary, _schedule, _error = cfs._arm_b_summary(
            forged_baseline, **source_kwargs
        )
        self.assertFalse(forged_summary["passed"])
        self.assertFalse(
            forged_summary["checks"]["baseline_target_error_source_exact"]
        )

        contradictory_status = copy.deepcopy(nested)
        contradictory_status["status"] = 1
        contradictory_status["fista"]["updates"] = 0
        contradictory_status["fista"]["selected_iteration"] = 0
        contradictory_summary, _schedule, _error = cfs._arm_b_summary(
            contradictory_status, **source_kwargs
        )
        self.assertFalse(contradictory_summary["passed"])
        self.assertFalse(
            contradictory_summary["checks"][
                "solver_status_matches_exact_jacobian_zero_state"
            ]
        )

        for path, bad_value, failed_check in (
            (("adapter_historical_projection_calls",), 128, "historical_projection_calls_129"),
            (("fista", "convex_dtype"), "torch.float32", "fista_convex_dtype_exact"),
            (("fista", "convex_device"), "cuda", "fista_convex_device_exact"),
            (
                ("fista", "optimality_certificate"),
                True,
                "fista_optimality_certificate_false",
            ),
            (
                ("fista", "infeasibility_certificate"),
                True,
                "fista_infeasibility_certificate_false",
            ),
        ):
            with self.subTest(path=path):
                malformed = copy.deepcopy(nested)
                owner = malformed
                for key in path[:-1]:
                    owner = owner[key]
                owner[path[-1]] = bad_value
                malformed_summary, _schedule, _error = cfs._arm_b_summary(
                    malformed, **source_kwargs
                )
                self.assertFalse(malformed_summary["passed"])
                self.assertFalse(malformed_summary["checks"][failed_check])

        # Scientific status/count/flag fields are exact wire metadata, not
        # numerics that may be truncated or truth-coerced.  These mutations
        # previously survived as 0, 4096, 37, and False respectively.
        for path, bad_value in (
            (("status",), 0.5),
            (("fista", "updates"), 4096.9),
            (("fista", "selected_iteration"), 37.8),
            (("fista", "effective_rank"), 35.9),
            (("jacobian_effective_rank",), 35.9),
            (("fista", "early_stopping_used"), 0),
            (("fista", "adaptive_restart_used"), 0),
            (("fista", "step_size"), False),
            (("finite_difference", "passed"), 1),
            (("status",), np.asarray([0], dtype=np.int64)),
            (
                ("fista", "step_size"),
                np.asarray([nested["fista"]["step_size"]], dtype=np.float64),
            ),
        ):
            with self.subTest(strict_scalar_path=path):
                malformed = copy.deepcopy(nested)
                owner = malformed
                for key in path[:-1]:
                    owner = owner[key]
                owner[path[-1]] = bad_value
                with self.assertRaises(cfs.ConstrainedFlowCanaryError):
                    cfs._arm_b_summary(malformed, **source_kwargs)

        for path, bad_value, failed_check in (
            (("adapter_solve_calls",), True, "solve_calls_one"),
            (("adapter_historical_projection_calls",), 129.0, "historical_projection_calls_129"),
            (("adapter_injection_count",), True, "injection_count_one"),
            (("adapter_projection_idempotence_checks",), 1.0, "projection_check_one"),
            (("adapter_projection_idempotence_exact",), 1, "projection_idempotent"),
            (
                ("adapter_only_historical_initialization_changed",),
                1,
                "only_historical_initialization_changed",
            ),
        ):
            with self.subTest(strict_adapter_scalar_path=path):
                malformed = copy.deepcopy(nested)
                owner = malformed
                for key in path[:-1]:
                    owner = owner[key]
                owner[path[-1]] = bad_value
                malformed_summary, _schedule, _error = cfs._arm_b_summary(
                    malformed, **source_kwargs
                )
                self.assertFalse(malformed_summary["passed"])
                self.assertFalse(malformed_summary["checks"][failed_check])

        exact_zero_dimensional = copy.deepcopy(nested)
        exact_zero_dimensional["status"] = np.asarray(0, dtype=np.int64)
        exact_zero_dimensional["fista"]["updates"] = np.asarray(
            4096, dtype=np.int64
        )
        exact_zero_dimensional["fista"]["early_stopping_used"] = np.asarray(
            False, dtype=np.bool_
        )
        exact_summary, _schedule, _error = cfs._arm_b_summary(
            exact_zero_dimensional, **source_kwargs
        )
        self.assertTrue(exact_summary["passed"])

    def test_zero_jacobian_independent_audit_uses_registered_explicit_branch(self) -> None:
        weights = np.asarray(
            [
                1.0 / (0.005 if channel < 3 else 0.015)
                for _row in range(5)
                for channel in range(7)
            ],
            dtype=np.float64,
        )
        nested = {
            "baseline_target_error": np.zeros(35, dtype=np.float32),
            "weights": weights,
            "linear_predicted_target_error": np.zeros(35, dtype=np.float32),
            "fista": {
                "selected_iteration": np.asarray(0, dtype=np.int64),
                "step_size": np.asarray(0.0, dtype=np.float64),
                "objective_half_squared_l2": np.asarray(0.0, dtype=np.float64),
                "historical_weighted_mse": np.asarray(0.0, dtype=np.float64),
                "executed_historical_weighted_mse": np.asarray(
                    0.0, dtype=np.float64
                ),
                "projected_gradient_mapping_norm": np.asarray(
                    0.0, dtype=np.float64
                ),
            },
        }
        checks, audit = cfs._independent_linear_checks(
            nested,
            jacobian=np.zeros((35, 75), dtype=np.float32),
            selected64=np.zeros((5, 15), dtype=np.float64),
            pre_projection=np.zeros((5, 10, 32), dtype=np.float32),
            post_projection=np.zeros((5, 10, 32), dtype=np.float32),
            raw_singular_values=np.zeros(35, dtype=np.float64),
            weighted_singular_values=np.zeros(35, dtype=np.float64),
            source_baseline_target_error=np.zeros(35, dtype=np.float32),
            executed_increments=np.zeros((10, 10, 32), dtype=np.float32),
            budget=np.float32(0.06),
        )
        self.assertTrue(all(checks.values()))
        self.assertEqual(audit["independent_selected_iteration"], 0)

    def test_registered_config_passes_and_causal_mutations_fail(self) -> None:
        path = ROOT / "configs/experiments/r05a_constrained_flow_canary.json"
        config = __import__("json").loads(path.read_text(encoding="utf-8"))
        self.assertEqual(cfs.validate_constrained_flow_config(config), [])

        mutated = copy.deepcopy(config)
        mutated["flow_contract"]["decoded_action_budget_B_action"] = "enabled"
        self.assertTrue(cfs.validate_constrained_flow_config(mutated))
        mutated = copy.deepcopy(config)
        mutated["execution_boundary"]["efficacy_rollouts_executed"] = 1
        self.assertTrue(cfs.validate_constrained_flow_config(mutated))
        mutated = copy.deepcopy(config)
        mutated["acceptance_contract"]["finite_nonconvergence_is_infeasibility"] = True
        self.assertTrue(cfs.validate_constrained_flow_config(mutated))
        for path, value in (
            (("linearized_solver", "fixed_update_count"), 1),
            (("jacobian_validation", "relative_l2_tolerance"), 0.99),
            (("flow_contract", "active_steps"), [0]),
            (("provisional_resource_contract", "source_host"), "worker-2"),
            (("target_contract", "source_budget_float32"), 99.0),
        ):
            mutated = copy.deepcopy(config)
            mutated[path[0]][path[1]] = value
            self.assertTrue(
                cfs.validate_constrained_flow_config(mutated),
                f"scientific mutation unexpectedly accepted: {path}",
            )

        release_only = copy.deepcopy(config)
        release_only.update(
            config_status="released_exact_single_canary",
            ready_to_run=True,
            blocked_on=[],
            execution_release={"fixture": True},
        )
        release_only["preregistration"]["h100_submission_authorized"] = True
        self.assertEqual(cfs.validate_constrained_flow_config(release_only), [])


if __name__ == "__main__":
    unittest.main()
