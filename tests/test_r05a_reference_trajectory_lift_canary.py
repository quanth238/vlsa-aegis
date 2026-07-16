from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

try:
    import numpy as np
except ModuleNotFoundError:  # dependency-free ./init.sh intentionally has no NumPy
    np = None


ROOT = Path(__file__).resolve().parents[1]


@unittest.skipIf(np is None, "TRL-00A canary tests require NumPy")
class ReferenceTrajectoryLiftCanaryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from main.crfs_oracle import r05a_reference_trajectory_lift_canary as canary
        from main.crfs_oracle import r05a_reference_trajectory_lift_validation as validation
        from main import run_crfs_r05a_reference_trajectory_lift_canary as cli
        from main.crfs_oracle.r05a_reference_trajectory_lift_validation import (
            FINITE_ARRAY_SPECS,
            validate_request_ledger,
        )
        from main.crfs_oracle.r02_runner import _array_record, _trace_record

        cls.canary = canary
        cls.cli = cli
        cls.array_record = staticmethod(_array_record)
        cls.trace_record = staticmethod(_trace_record)
        cls.finite_array_specs = FINITE_ARRAY_SPECS
        cls.validate_request_ledger = staticmethod(validate_request_ledger)
        cls.validation = validation

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        config = json.loads(
            (
                ROOT
                / "configs/experiments/r05a_reference_trajectory_lift_canary.json"
            ).read_text(encoding="utf-8")
        )
        config["ready_to_run"] = True
        config["config_status"] = "released_for_test"
        config["blocked_on"] = []
        config["preregistration"]["h100_submission_authorized"] = True
        config["execution_release"] = {"run_id": "trl00a-test-run", "test_only": True}
        self.config = config
        self.actual_path = self.root / "trl.json"
        self.actual_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        self.legacy_path = self.root / "legacy.json"
        self.legacy_path.write_text("{}\n", encoding="utf-8")
        self.case = copy.deepcopy(config["frozen_case"])
        oracle = SimpleNamespace(
            output_root=str(self.root / "output"),
            run_id="trl00a-test-run",
            host="127.0.0.1",
            port=8000,
            resize_size=224,
        )
        self.legacy = SimpleNamespace(
            oracle=oracle, source_r02_results_root=str(self.root / "source")
        )
        self.noise = np.random.default_rng(int(self.case["policy_seed"])).normal(
            size=(10, 32)
        ).astype(np.float32)
        self.budget = np.float32(config["target_contract"]["source_budget_float32"])
        self.delta = np.zeros((10, 32), dtype=np.float64)
        self.delta[0, 0] = np.float64(self.budget)
        self.delta32 = self.delta.astype(np.float32)
        self.source_trace = self._midpoint_trace(self.noise)
        source_actions = self._physical(self.noise)
        self.raw_r02 = {
            "provenance": {
                "case_record": copy.deepcopy(self.case),
                "noise": self.array_record(self.noise, dtype=np.float32),
            },
            "directions": {
                "l2_norms": {
                    "delta_star_model": config["target_contract"][
                        "source_reported_budget_float64"
                    ]
                },
                "arrays": {"delta_star_model": self.array_record(self.delta)},
            },
            "pairing": {
                "policy_observation": {"fingerprint": "exact"},
                "branch_snapshot": {"branch": "exact"},
                "eager_actions": self.array_record(source_actions),
                "eager_trace": self.trace_record(self.source_trace),
            },
        }
        self.af_result, self.af_arrays = self._af_source()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def _times() -> np.ndarray:
        values = []
        value = np.float32(1.0)
        for _ in range(10):
            values.append(value)
            value = np.float32(value + np.float32(-0.1))
        return np.asarray(values, dtype=np.float32)

    @staticmethod
    def _physical(final: np.ndarray) -> np.ndarray:
        actions = np.asarray(final[:, :7], dtype=np.float64).copy()
        actions[:, :3] *= np.asarray(
            (0.8422505, 0.827813, 0.937313), dtype=np.float64
        )
        return actions

    @classmethod
    def _ordinary_trace(cls, schedule: np.ndarray, noise: np.ndarray, budget: np.float32):
        schedule = np.asarray(schedule, dtype=np.float32)
        dt = np.float32(-0.1)
        base = np.zeros_like(schedule)
        total = np.where(schedule == 0, base, base + schedule).astype(np.float32)
        x_t = np.empty_like(schedule)
        x_next = np.empty_like(schedule)
        current = np.asarray(noise, dtype=np.float32).copy()
        for index in range(10):
            x_t[index] = current
            x_next[index] = np.asarray(current + dt * total[index], dtype=np.float32)
            current = x_next[index]
        increments = np.asarray(dt * schedule, dtype=np.float32)
        per_step = np.linalg.norm(
            increments[5:, :5, :3].reshape(5, 15), axis=1
        ).astype(np.float32)
        return {
            "step_index_steps": np.arange(10, dtype=np.int64),
            "time_steps": cls._times(),
            "active_steps": np.asarray([False] * 5 + [True] * 5, dtype=np.bool_),
            "x_t_steps": x_t,
            "v_base_steps": base,
            "control_velocity_steps": schedule,
            "total_velocity_steps": total,
            "control_increment_steps": increments,
            "x_next_steps": x_next,
            "initial_noise": np.asarray(noise, dtype=np.float32).copy(),
            "canonical_replay_final": x_next[-1].copy(),
            "final_normalized": x_next[-1].copy(),
            "final_normalized_physical": cls._physical(x_next[-1]),
            "control_source": np.int64(1),
            "control_valid": np.bool_(True),
            "schedule_applied": np.bool_(True),
            "schedule_budget": np.float32(budget),
            "schedule_path_length": np.asarray(
                np.sum(per_step, dtype=np.float32), dtype=np.float32
            ),
            "schedule_per_step_increment_l2": per_step,
            "schedule_energy": np.asarray(
                np.sum(np.square(increments), dtype=np.float32), dtype=np.float32
            ),
            "dt": dt,
            "intervention_step": np.int64(5),
            "num_steps": np.int64(10),
        }

    @classmethod
    def _reference_trace(
        cls,
        reference: np.ndarray,
        delta: np.ndarray,
        noise: np.ndarray,
        budget: np.float32,
        projection_mode: str,
    ):
        dt = np.float32(-0.1)
        cap = np.float32(budget / np.float32(5))
        x_t = np.empty((10, 10, 32), dtype=np.float32)
        x_next = np.empty_like(x_t)
        base = np.zeros_like(x_t)
        desired = np.zeros_like(x_t)
        uncontrolled = np.empty_like(x_t)
        raw = np.zeros_like(x_t)
        requested = np.zeros_like(x_t)
        velocity = np.zeros_like(x_t)
        executed = np.zeros_like(x_t)
        tracking = np.zeros_like(x_t)
        raw_norm = np.zeros((10,), dtype=np.float64)
        requested_norm = np.zeros((10,), dtype=np.float64)
        executed_norm = np.zeros((10,), dtype=np.float64)
        scales = np.ones((10,), dtype=np.float64)
        projected = np.zeros((10,), dtype=np.bool_)
        current = np.asarray(noise, dtype=np.float32).copy()
        mask = np.zeros((10, 32), dtype=np.bool_)
        mask[:5, :3] = True
        for step in range(10):
            x_t[step] = current
            uncontrolled[step] = np.asarray(current + dt * base[step], dtype=np.float32)
            if step >= 5:
                desired[step] = reference[step - 4]
                raw[step] = np.where(
                    mask,
                    np.asarray(desired[step] - uncontrolled[step], dtype=np.float32),
                    np.zeros((10, 32), dtype=np.float32),
                )
                compact64 = raw[step, :5, :3].reshape(15).astype(np.float64)
                total64 = np.float64(0.0)
                for coordinate in compact64:
                    total64 = np.float64(total64 + np.float64(coordinate * coordinate))
                raw_norm[step] = np.sqrt(total64, dtype=np.float64)
                requested[step] = raw[step]
                if projection_mode == "product_ball" and raw_norm[step] > np.float64(cap):
                    projected[step] = True
                    scales[step] = np.float64(cap) / raw_norm[step]
                    requested[step] = np.where(
                        mask,
                        np.asarray(
                            raw[step].astype(np.float64) * scales[step], dtype=np.float32
                        ),
                        np.zeros((10, 32), dtype=np.float32),
                    )
                velocity[step] = np.where(
                    mask,
                    np.asarray(requested[step] / dt, dtype=np.float32),
                    np.zeros((10, 32), dtype=np.float32),
                )
                executed[step] = np.where(
                    mask,
                    np.asarray(dt * velocity[step], dtype=np.float32),
                    np.zeros((10, 32), dtype=np.float32),
                )
                requested_norm[step] = np.linalg.norm(
                    requested[step, :5, :3].reshape(15).astype(np.float64)
                )
                executed_norm[step] = np.linalg.norm(
                    executed[step, :5, :3].reshape(15).astype(np.float64)
                )
            else:
                velocity[step] = np.zeros((10, 32), dtype=np.float32)
                executed[step] = np.zeros((10, 32), dtype=np.float32)
            total_velocity = np.where(
                velocity[step] == 0, base[step], base[step] + velocity[step]
            ).astype(np.float32)
            x_next[step] = np.asarray(current + dt * total_velocity, dtype=np.float32)
            if step >= 5:
                tracking[step] = np.where(
                    mask,
                    np.asarray(x_next[step] - desired[step], dtype=np.float32),
                    np.zeros((10, 32), dtype=np.float32),
                )
            current = x_next[step]
        total = np.where(velocity == 0, base, base + velocity).astype(np.float32)
        trace = {
            "step_index_steps": np.arange(10, dtype=np.int64),
            "time_steps": cls._times(),
            "active_steps": np.asarray([False] * 5 + [True] * 5, dtype=np.bool_),
            "x_t_steps": x_t,
            "v_base_steps": base,
            "control_velocity_steps": velocity,
            "total_velocity_steps": total,
            "control_increment_steps": np.asarray(dt * velocity, dtype=np.float32),
            "x_next_steps": x_next,
            "initial_noise": np.asarray(noise, dtype=np.float32).copy(),
            "canonical_replay_final": x_next[-1].copy(),
            "final_normalized": x_next[-1].copy(),
            "final_normalized_physical": cls._physical(x_next[-1]),
            "control_source": np.int64(2),
            "control_valid": np.bool_(True),
            "schedule_applied": np.bool_(True),
            "dt": dt,
            "intervention_step": np.int64(5),
            "num_steps": np.int64(10),
            "reference_active_states": np.asarray(reference, dtype=np.float32).copy(),
            "reference_delta": np.asarray(delta, dtype=np.float32).copy(),
            "reference_alpha": cls.canary._alpha_float32(),
            "reference_anchor_exact": np.bool_(True),
            "reference_projection_mode": np.int64(
                0 if projection_mode == "raw" else 1
            ),
            "reference_source_budget": np.float32(budget),
            "reference_per_step_cap": cap,
            "reference_desired_next_steps": desired,
            "reference_uncontrolled_next_steps": uncontrolled,
            "reference_raw_increment_steps": raw,
            "reference_requested_increment_steps": requested,
            "reference_requested_velocity_steps": velocity,
            "reference_executed_increment_steps": executed,
            "reference_raw_norm_f64_steps": raw_norm,
            "reference_requested_norm_f64_steps": requested_norm,
            "reference_executed_norm_f64_steps": executed_norm,
            "reference_projection_scale_f64_steps": scales,
            "reference_projected_steps": projected,
            "reference_tracking_error_steps": tracking,
            "reference_raw_path_length_f64": np.asarray(
                np.sum(raw_norm[5:], dtype=np.float64), dtype=np.float64
            ),
            "reference_requested_path_length_f64": np.asarray(
                np.sum(requested_norm[5:], dtype=np.float64), dtype=np.float64
            ),
            "reference_executed_path_length_f64": np.asarray(
                np.sum(executed_norm[5:], dtype=np.float64), dtype=np.float64
            ),
            "reference_product_ball_constraint_applied": np.bool_(
                projection_mode == "product_ball"
            ),
            "reference_product_ball_valid": np.bool_(
                projection_mode == "product_ball"
            ),
            "solver_status": np.int64(-2),
            "solver_iterations": np.int64(0),
            "solver_fields_available": np.bool_(False),
            "solver_nonfinite": np.bool_(False),
        }
        if projection_mode == "product_ball":
            per_step = np.linalg.norm(
                executed[5:, :5, :3].reshape(5, 15), axis=1
            ).astype(np.float32)
            trace.update(
                schedule_budget=np.float32(budget),
                schedule_path_length=np.asarray(
                    np.sum(per_step, dtype=np.float32), dtype=np.float32
                ),
                schedule_per_step_increment_l2=per_step,
                schedule_energy=np.asarray(
                    np.sum(np.square(executed), dtype=np.float32), dtype=np.float32
                ),
            )
        return trace

    @classmethod
    def _midpoint_trace(cls, noise: np.ndarray, *, include_final: bool = False):
        trace = {
            "step_index": np.asarray(5, dtype=np.int64),
            "time": np.asarray(0.5, dtype=np.float32),
            "x_t": np.asarray(noise, dtype=np.float32).copy(),
            "v_base": np.zeros((10, 32), dtype=np.float32),
            "predicted_clean": np.asarray(noise, dtype=np.float32).copy(),
            "predicted_clean_physical": cls._physical(noise).astype(np.float32),
        }
        if include_final:
            trace["final_normalized"] = np.asarray(noise, dtype=np.float32).copy()
            trace["final_normalized_physical"] = cls._physical(noise)
        return trace

    def _af_source(self):
        frozen = self.noise.copy()
        target = np.where(
            self.delta32 == np.float32(0),
            frozen,
            np.asarray(frozen + self.delta32, dtype=np.float32),
        )
        scale = np.ones((10, 32), dtype=np.float32)
        scale[:, :3] = np.asarray((0.8422505, 0.827813, 0.937313), dtype=np.float32)
        target_physical = self._physical(frozen)
        target_physical[:5, :3] += self.delta32[:5, :3].astype(np.float64) * scale[
            :5, :3
        ].astype(np.float64)
        arm_c = np.broadcast_to(
            np.asarray(self.delta32[:5, :3].reshape(15) / np.float32(5), dtype=np.float32),
            (5, 15),
        ).copy()
        arm_u = np.asarray(arm_c / np.float32(-0.1), dtype=np.float32)
        schedule = np.zeros((10, 10, 32), dtype=np.float32)
        schedule[5:, :5, :3] = arm_u.reshape(5, 5, 3)
        traces = [self._ordinary_trace(schedule, self.noise, self.budget) for _ in range(2)]
        actions = np.stack([self._physical(trace["final_normalized"]) for trace in traces])
        error = actions[0, :5, :7] - target_physical[:5, :7]
        objective, metrics, gates = self.canary.objective_and_gates(error)
        arrays = {
            "source_noise_f32": self.noise,
            "source_frozen_final_f32": frozen,
            "source_delta_model_f32": self.delta32,
            "source_target_normalized_f32": target,
            "source_model_to_physical_scale_f32": scale,
            "source_target_physical_f64": target_physical,
            "source_budget_f32": np.asarray(self.budget, dtype=np.float32),
            "source_dt_f32": np.asarray(-0.1, dtype=np.float32),
            "source_radius_f32": np.asarray(self.budget / np.float32(5), dtype=np.float32),
            "arm_a_requested_c_f32": np.broadcast_to(arm_c, (2, 5, 15)).copy(),
            "arm_a_requested_velocity_f32": np.broadcast_to(arm_u, (2, 5, 15)).copy(),
            "arm_a_applied_velocity_f32": np.stack(
                [trace["control_velocity_steps"][5:, :5, :3].reshape(5, 15) for trace in traces]
            ),
            "arm_a_executed_c_f32": np.stack(
                [trace["control_increment_steps"][5:, :5, :3].reshape(5, 15) for trace in traces]
            ),
            "arm_a_final_normalized_f32": np.stack(
                [trace["final_normalized"] for trace in traces]
            ),
            "arm_a_final_normalized_physical_f64": np.stack(
                [trace["final_normalized_physical"] for trace in traces]
            ),
            "arm_a_returned_actions_f64": actions,
            "arm_a_trace_step_index_i64": np.stack(
                [trace["step_index_steps"] for trace in traces]
            ),
            "arm_a_trace_time_f32": np.stack([trace["time_steps"] for trace in traces]),
            "arm_a_trace_active_bool": np.stack([trace["active_steps"] for trace in traces]),
            "arm_a_trace_x_t_f32": np.stack([trace["x_t_steps"] for trace in traces]),
            "arm_a_trace_v_base_f32": np.stack([trace["v_base_steps"] for trace in traces]),
            "arm_a_trace_total_velocity_f32": np.stack(
                [trace["total_velocity_steps"] for trace in traces]
            ),
            "arm_a_trace_x_next_f32": np.stack([trace["x_next_steps"] for trace in traces]),
            "arm_a_trace_initial_noise_f32": np.stack(
                [trace["initial_noise"] for trace in traces]
            ),
        }
        result = {
            "validation": {
                "arm_a": {
                    "objective": objective,
                    "metrics": dict(
                        zip(
                            ("xyz_max_abs", "xyz_rms", "full_max_abs", "full_rms"),
                            metrics,
                        )
                    ),
                    "passed": all(gates),
                }
            }
        }
        return result, arrays

    class Environment:
        obstacle_name = "obstacle"
        prompt = "reach"

        def __init__(self):
            self.env = SimpleNamespace(check_success=lambda: False)
            self.rollout_calls = 0

        def configure_case(self, _case):
            return None

        def reset_and_settle(self):
            return {"image": "synthetic"}

        def rollout(self, *_args, **_kwargs):
            self.rollout_calls += 1
            raise AssertionError("TRL-00A must not execute generated actions")

    class Branch:
        def to_dict(self):
            return {"branch": "exact"}

    class Client:
        def __init__(self, owner, *, raw_terminal: bool):
            self.owner = owner
            self.raw_terminal = raw_terminal
            self.controls = []

        def infer(self, request):
            controls = copy.deepcopy(request["__crfs__"])
            self.controls.append(controls)
            mode = controls["intervention_mode"]
            noise = np.asarray(controls["noise"], dtype=np.float32)
            if mode == "reference_trajectory_lift" and controls["projection_mode"] == "raw" and self.raw_terminal:
                return {
                    "__crfs_terminal__": {
                        "type": "RAW_FLOAT32_UNREPRESENTABLE",
                        "step": 7,
                        "leaf": "requested_velocity",
                    },
                    "server_timing": {"infer_ms": 1.0},
                }
            if mode == "residual_schedule":
                trace = self.owner._ordinary_trace(
                    np.asarray(controls["schedule"], dtype=np.float32),
                    noise,
                    np.float32(controls["model_l2_path_budget"]),
                )
                final = trace["final_normalized"]
            elif mode == "reference_trajectory_lift":
                trace = self.owner._reference_trace(
                    np.asarray(controls["reference_states"], dtype=np.float32),
                    np.asarray(controls["delta"], dtype=np.float32),
                    noise,
                    np.float32(controls["model_l2_path_budget"]),
                    str(controls["projection_mode"]),
                )
                final = trace["final_normalized"]
            elif mode == "none":
                trace = self.owner._midpoint_trace(
                    noise, include_final=bool(controls.get("return_normalized_final"))
                )
                final = noise
            else:
                raise AssertionError(f"unregistered mode {mode}")
            reply = {"actions": self.owner._physical(final)}
            if controls.get("return_trace"):
                reply["crfs_trace"] = trace
            return reply

    def _run(self, *, raw_terminal: bool):
        client = self.Client(self, raw_terminal=raw_terminal)
        environment = self.Environment()
        source_path = Path(self.config["frozen_source_bindings"]["source_r02"]["path"])
        patches = (
            mock.patch.object(self.canary.socket, "gethostname", return_value="worker-1"),
            mock.patch.object(
                self.canary,
                "_load_source_r02",
                return_value=(
                    source_path,
                    self.raw_r02,
                    self.config["frozen_source_bindings"]["source_r02"]["sha256"],
                ),
            ),
            mock.patch.object(
                self.canary,
                "_source_delta",
                return_value=(
                    self.delta,
                    self.array_record(self.delta),
                    self.config["target_contract"]["source_reported_budget_float64"],
                ),
            ),
            mock.patch.object(
                self.canary, "_load_af_source", return_value=(self.af_result, self.af_arrays)
            ),
            mock.patch.object(
                self.canary, "capture_reach_snapshot", return_value=self.Branch()
            ),
            mock.patch.object(self.canary, "_target_contact_at_branch", return_value=False),
            mock.patch.object(self.canary, "policy_observation", return_value={"policy": "input"}),
            mock.patch.object(
                self.canary,
                "_observation_fingerprint",
                return_value={"fingerprint": "exact"},
            ),
            mock.patch.object(self.canary, "_trace_from_record", return_value=self.source_trace),
            mock.patch.object(
                self.canary,
                "_trace_pairing_diagnostics",
                return_value={"exact_native_leaf_pairing": True},
            ),
        )
        with mock.patch.dict(
            os.environ,
            {"SLURM_JOB_ID": "1", "CUDA_VISIBLE_DEVICES": "GPU-test"},
            clear=False,
        ):
            with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7], patches[8], patches[9]:
                output = self.canary.run_reference_trajectory_lift_canary(
                    self.case,
                    self.config,
                    self.legacy,
                    repo_root=ROOT,
                    input_manifest_sha256=self.canary.MANIFEST_SHA256,
                    actual_config_path=self.actual_path,
                    legacy_config_path=self.legacy_path,
                    client=client,
                    environment=environment,
                )
        return output, client, environment

    def test_cli_parser_exposes_exact_allocation_entrypoint_contract(self):
        args = self.cli._parser().parse_args(
            [
                "--manifest",
                "manifest.jsonl",
                "--config",
                "trl.json",
                "--legacy-config",
                "legacy.json",
                "--r02-raw-root",
                "/source/r02",
                "--output-root",
                "/output",
                "--run-id",
                "trl00a-run",
                "--case-index",
                "0",
                "--port",
                "8000",
                "--checkpoint-id",
                "checkpoint",
                "--checkpoint-sha256",
                "a" * 64,
            ]
        )
        self.assertEqual(args.host, "127.0.0.1")
        self.assertEqual(args.case_index, 0)
        self.assertEqual(args.run_id, "trl00a-run")

    def test_reference_construction_preserves_signed_zero_and_registered_alpha(self):
        noise = np.zeros((10, 32), dtype=np.float32)
        noise[7, 11] = np.float32(-0.0)
        trace = self._ordinary_trace(
            np.zeros((10, 10, 32), dtype=np.float32), noise, np.float32(0)
        )
        delta = np.zeros((10, 32), dtype=np.float32)
        delta[0, 0] = np.float32(1.0)
        target = noise.copy()
        target[0, 0] = np.float32(1.0)
        reference, xbar, alpha = self.canary.construct_reference_trajectory(
            trace, delta, target
        )
        self.assertEqual(
            alpha.view(np.uint32).tolist(),
            [0x00000000, 0x3E4CCCCD, 0x3ECCCCCD, 0x3F19999A, 0x3F4CCCCD, 0x3F800000],
        )
        self.assertTrue(np.signbit(reference[:, 7, 11]).all())
        self.assertEqual(reference[0].tobytes(), xbar[0].tobytes())
        self.assertEqual(reference[-1].tobytes(), target.tobytes())

    def test_terminal_transport_is_strict(self):
        terminal = {
            "__crfs_terminal__": {
                "type": "RAW_FLOAT32_UNREPRESENTABLE",
                "step": 8,
                "leaf": "next_state",
            },
            "server_timing": {"infer_ms": 1.0, "prev_total_ms": 2.0},
        }
        self.assertEqual(
            self.canary._normalize_raw_terminal(terminal), terminal["__crfs_terminal__"]
        )
        malformed = copy.deepcopy(terminal)
        malformed["extra"] = 1
        with self.assertRaisesRegex(
            self.canary.ReferenceTrajectoryLiftCanaryError, "outer key"
        ):
            self.canary._normalize_raw_terminal(malformed)
        malformed = copy.deepcopy(terminal)
        malformed["__crfs_terminal__"]["leaf"] = "desired_next"
        with self.assertRaisesRegex(
            self.canary.ReferenceTrajectoryLiftCanaryError, "leaf"
        ):
            self.canary._normalize_raw_terminal(malformed)

    def test_raw_replay_envelope_uses_smallest_upward_float32(self):
        within = np.zeros((5, 15), dtype=np.float32)
        exact = self.canary.raw_replay_envelope(within, np.float32(1.0))
        self.assertTrue(exact["exact_source_budget_valid"])
        self.assertEqual(exact["replay_budget_float32"].tobytes(), np.float32(1).tobytes())

        outside = np.zeros((5, 15), dtype=np.float32)
        outside[:, 0] = np.float32(1.0000001)
        result = self.canary.raw_replay_envelope(outside, np.float32(1.0))
        self.assertFalse(result["exact_source_budget_valid"])
        seed = max(
            np.float64(result["path_float32"]),
            np.float64(5) * np.float64(result["max_step_norm_float32"]),
            np.float64(1),
        )
        self.assertGreaterEqual(np.float64(result["replay_budget_float32"]), seed)
        prior = np.nextafter(
            result["replay_budget_float32"], np.float32(-np.inf), dtype=np.float32
        )
        self.assertLess(np.float64(prior), seed)

    def test_exact_finite_transaction_writes_raw_artifacts_only(self):
        (payload_path, branch), client, environment = self._run(raw_terminal=False)
        self.assertEqual(branch, "finite")
        self.assertEqual(len(client.controls), 18)
        self.assertEqual(environment.rollout_calls, 0)
        self.assertFalse((payload_path.parent / "results.json").exists())
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        ledger = json.loads(
            (payload_path.parent / "request-ledger.json").read_text(encoding="utf-8")
        )
        self.assertEqual(payload["request_accounting"]["total"], 18)
        self.assertEqual(
            tuple(row["phase"] for row in ledger["rows"]),
            self.canary.FINITE_REQUEST_PHASES,
        )
        self.assertEqual(
            {control["intervention_mode"] for control in client.controls},
            {"none", "residual_schedule", "reference_trajectory_lift"},
        )
        with np.load(payload_path.parent / "trl00a-tensors.npz", allow_pickle=False) as archive:
            self.assertEqual(set(archive.files), set(self.finite_array_specs))
            self.validate_request_ledger(ledger["rows"], archive)
            self.assertEqual(archive["reference_states_f32"].shape, (6, 10, 32))
            self.assertEqual(
                archive["arm_b_generation_reference_raw_increment_steps_f32"].shape,
                (2, 10, 10, 32),
            )
            self.assertEqual(
                archive["raw_replay_trace_control_velocity_f32"].shape,
                (2, 10, 10, 32),
            )
            self.assertTrue(all(archive[name].dtype != np.dtype("O") for name in archive.files))
            frozen_bindings = {
                name: self.validation._array_sha256(archive[name])
                for name in self.validation.FROZEN_ARRAY_BINDINGS
            }
            historical = self.af_result["validation"]["arm_a"]
            historical_metrics = np.asarray(
                [
                    historical["metrics"][name]
                    for name in ("xyz_max_abs", "xyz_rms", "full_max_abs", "full_rms")
                ],
                dtype=np.float64,
            )
            with mock.patch.object(
                self.validation,
                "EXPECTED_ARM_A_OBJECTIVE64",
                np.float64(historical["objective"]),
            ), mock.patch.object(
                self.validation,
                "EXPECTED_ARM_A_METRICS64",
                historical_metrics,
            ), mock.patch.object(
                self.validation,
                "_validate_reference_and_arm_a",
                return_value=(
                    np.asarray(archive["source_target_physical_f64"], dtype=np.float64),
                    np.zeros((4,), dtype=np.bool_),
                ),
            ), mock.patch.object(
                self.validation,
                "FROZEN_ARRAY_BINDINGS",
                frozen_bindings,
            ):
                validated = self.validation.validate_reference_trajectory_lift_tensors(
                    archive,
                    ledger["rows"],
                    payload["source_bindings"],
                )
            self.assertEqual(validated["request_count"], 18)
            self.assertEqual(validated["branch"], "finite")

    def test_raw_terminal_is_apparatus_inconclusive_and_publishes_nothing(self):
        with self.assertRaisesRegex(
            self.canary.ReferenceTrajectoryLiftCanaryError,
            "RAW_FLOAT32_UNREPRESENTABLE.*apparatus-inconclusive",
        ):
            self._run(raw_terminal=True)
        output = self.root / "output" / "trl00a-test-run" / self.canary.CASE_ID
        self.assertFalse((output / "results.json").exists())
        self.assertFalse((output / "trl00a-raw-payload.json").exists())
        self.assertFalse((output / "trl00a-tensors.npz").exists())

    def test_draft_config_fails_before_policy_or_artifact(self):
        draft = copy.deepcopy(self.config)
        draft["ready_to_run"] = False
        client = self.Client(self, raw_terminal=False)
        with self.assertRaisesRegex(
            self.canary.ReferenceTrajectoryLiftCanaryError, "not execution-released"
        ):
            self.canary.run_reference_trajectory_lift_canary(
                self.case,
                draft,
                self.legacy,
                repo_root=ROOT,
                input_manifest_sha256=self.canary.MANIFEST_SHA256,
                actual_config_path=self.actual_path,
                legacy_config_path=self.legacy_path,
                client=client,
                environment=self.Environment(),
            )
        self.assertEqual(client.controls, [])

    def test_allocation_python_38_has_no_zip_strict_runtime_call(self):
        for relative in (
            "main/crfs_oracle/r05a_reference_trajectory_lift_canary.py",
            "main/crfs_oracle/r05a_reference_trajectory_lift_validation.py",
            "main/run_crfs_r05a_reference_trajectory_lift_canary.py",
            "main/publish_crfs_r05a_reference_trajectory_lift_canary.py",
        ):
            path = ROOT / relative
            if path.exists():
                self.assertNotIn("strict=True", path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
