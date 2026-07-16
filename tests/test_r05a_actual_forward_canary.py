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


@unittest.skipIf(np is None, "actual-forward canary tests require NumPy")
class ActualForwardCanaryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from main.crfs_oracle import r05a_actual_forward_canary as canary
        from main.crfs_oracle.r05a_actual_forward_validation import (
            ARRAY_SPECS,
            validate_actual_forward_npz,
        )
        from main.crfs_oracle.r02_runner import _array_record, _trace_record

        cls.canary = canary
        cls.array_specs = ARRAY_SPECS
        cls.validate_npz = staticmethod(validate_actual_forward_npz)
        cls.array_record = staticmethod(_array_record)
        cls.trace_record = staticmethod(_trace_record)

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.actual_path = self.root / "actual.json"
        self.legacy_path = self.root / "legacy.json"
        self.legacy_path.write_text("{}\n", encoding="utf-8")
        config = json.loads(
            (ROOT / "configs/experiments/r05a_actual_forward_canary.json").read_text(
                encoding="utf-8"
            )
        )
        config["ready_to_run"] = True
        config["config_status"] = "released_for_test"
        config["blocked_on"] = []
        config["execution_release"] = {"test_only": True}
        self.config = config
        self.actual_path.write_text(
            json.dumps(config, indent=2) + "\n", encoding="utf-8"
        )
        self.case = copy.deepcopy(config["frozen_case"])
        self.case.update(
            {
                "task_suite": "safelibero_spatial",
                "safety_level": "II",
                "task_index": 0,
                "episode_index": 46,
            }
        )
        oracle = SimpleNamespace(
            output_root=str(self.root / "output"),
            run_id="af00a-test-run",
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
        self.delta = np.zeros((10, 32), dtype=np.float64)
        self.delta[:5, :3] = np.float64(0.02)
        budget64 = float(np.linalg.norm(self.delta[:5, :3]))
        # Preserve the preregistered source scalar while choosing a direction
        # with that exact norm for the synthetic policy.
        target_budget = float(np.float32(self.config["target_contract"]["source_budget_float32"]))
        self.delta *= target_budget / budget64
        self.budget64 = float(self.config["target_contract"]["source_reported_budget_float64"])
        frozen_recurrence = self._trace(
            np.zeros((10, 10, 32), dtype=np.float32), self.noise
        )
        self.source_trace = self._midpoint_trace(self.noise)
        source_actions = self._physical(frozen_recurrence["final_normalized"])
        self.raw_r02 = {
            "provenance": {
                "case_record": copy.deepcopy(self.case),
                "noise": self.array_record(self.noise, dtype=np.float32),
            },
            "directions": {
                "l2_norms": {"delta_star_model": self.budget64},
                "arrays": {"delta_star_model": self.array_record(self.delta)},
            },
            "pairing": {
                "policy_observation": {"fingerprint": "exact"},
                "branch_snapshot": {"branch": "exact"},
                "eager_actions": self.array_record(source_actions),
                "eager_trace": self.trace_record(self.source_trace),
            },
        }

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def _physical(final):
        actions = np.asarray(final[:, :7], dtype=np.float64).copy()
        actions[:, :3] *= np.asarray(
            (0.8422505, 0.827813, 0.937313), dtype=np.float64
        )
        return actions

    @classmethod
    def _trace(cls, schedule, noise):
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
        times = []
        current_time = np.float32(1.0)
        for _ in range(10):
            times.append(current_time)
            current_time = np.float32(current_time + dt)
        final = x_next[-1].copy()
        return {
            "step_index_steps": np.arange(10, dtype=np.int64),
            "time_steps": np.asarray(times, dtype=np.float32),
            "active_steps": np.asarray([False] * 5 + [True] * 5, dtype=np.bool_),
            "x_t_steps": x_t,
            "v_base_steps": base,
            "control_velocity_steps": schedule,
            "total_velocity_steps": total,
            "control_increment_steps": np.asarray(dt * schedule, dtype=np.float32),
            "x_next_steps": x_next,
            "initial_noise": np.asarray(noise, dtype=np.float32).copy(),
            "final_normalized": final,
            "final_normalized_physical": cls._physical(final),
            "control_source": np.int64(1),
            "control_valid": np.bool_(True),
            "schedule_applied": np.bool_(True),
            "schedule_budget": np.float32(0.0),
            "dt": np.float32(-0.1),
            "intervention_step": np.int64(5),
            "num_steps": np.int64(10),
        }

    @classmethod
    def _midpoint_trace(cls, noise, *, include_final=False):
        value = {
            "step_index": np.asarray(5, dtype=np.int64),
            "time": np.asarray(0.5, dtype=np.float32),
            "x_t": np.asarray(noise, dtype=np.float32).copy(),
            "v_base": np.zeros((10, 32), dtype=np.float32),
            "predicted_clean": np.asarray(noise, dtype=np.float32).copy(),
            "predicted_clean_physical": cls._physical(noise).astype(np.float32),
        }
        if include_final:
            value["final_normalized"] = np.asarray(noise, dtype=np.float32).copy()
            value["final_normalized_physical"] = cls._physical(noise)
        return value

    class Environment:
        obstacle_name = "obstacle"
        prompt = "reach"

        def __init__(self):
            self.env = SimpleNamespace(check_success=lambda: False)
            self.configured = []
            self.closed = False
            self.rollout_calls = 0

        def configure_case(self, case):
            self.configured.append(case)

        def reset_and_settle(self):
            return {"image": "synthetic"}

        def rollout(self, *args, **kwargs):
            self.rollout_calls += 1
            raise AssertionError("AF-00A must not execute a generated action")

        def close(self):
            self.closed = True

    class Branch:
        def to_dict(self):
            return {"branch": "exact"}

    class Client:
        def __init__(self, owner, corrupt_duplicate=False):
            self.owner = owner
            self.controls = []
            self.residual_count = 0
            self.corrupt_duplicate = corrupt_duplicate

        def infer(self, request):
            controls = copy.deepcopy(request["__crfs__"])
            self.controls.append(controls)
            mode = controls["intervention_mode"]
            if mode not in {"none", "residual_schedule"}:
                raise AssertionError(f"retired policy path used: {mode}")
            if mode == "residual_schedule":
                self.residual_count += 1
                schedule = np.asarray(controls["schedule"], dtype=np.float32)
                trace = self.owner._trace(
                    schedule, np.asarray(controls["noise"], dtype=np.float32)
                )
            else:
                schedule = np.zeros((10, 10, 32), dtype=np.float32)
                trace = self.owner._midpoint_trace(
                    np.asarray(controls["noise"], dtype=np.float32),
                    include_final=bool(controls.get("return_normalized_final")),
                )
            if mode == "residual_schedule":
                trace["schedule_budget"] = np.float32(
                    controls.get("model_l2_path_budget", 0.0)
                )
                final = trace["final_normalized"]
            else:
                final = np.asarray(controls["noise"], dtype=np.float32)
            actions = self.owner._physical(final)
            # residual call 1 is zero, 2 is A first, and 3 is A duplicate.
            if self.corrupt_duplicate and self.residual_count == 3:
                actions = actions.copy()
                actions[0, 0] = np.nextafter(actions[0, 0], np.float32(np.inf))
            reply = {"actions": actions}
            if controls.get("return_trace"):
                reply["crfs_trace"] = trace
            return reply

    def _patches(self):
        return (
            mock.patch.object(self.canary.socket, "gethostname", return_value="worker-1"),
            mock.patch.object(self.canary, "_load_source_r02", return_value=(self.root / "r02.json", self.raw_r02, "0" * 64)),
            mock.patch.object(self.canary, "_source_delta", return_value=(self.delta, self.array_record(self.delta), self.budget64)),
            mock.patch.object(self.canary, "capture_reach_snapshot", return_value=self.Branch()),
            mock.patch.object(self.canary, "_target_contact_at_branch", return_value=False),
            mock.patch.object(self.canary, "policy_observation", return_value={"policy": "input"}),
            mock.patch.object(self.canary, "_observation_fingerprint", return_value={"fingerprint": "exact"}),
            mock.patch.object(self.canary, "_trace_from_record", return_value=self.source_trace),
            mock.patch.object(self.canary, "_trace_pairing_diagnostics", return_value={"exact_native_leaf_pairing": True}),
        )

    def _run(self, client):
        environment = self.Environment()
        patches = self._patches()
        with mock.patch.dict(
            os.environ,
            {"SLURM_JOB_ID": "1", "CUDA_VISIBLE_DEVICES": "GPU-test"},
            clear=False,
        ):
            with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7]:
                result = self.canary.run_actual_forward_canary(
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
        return result, environment

    def test_exact_534_call_ordinary_path_writes_reconstructable_raw_artifacts(self):
        client = self.Client(self)
        (payload_path, status), environment = self._run(client)
        self.assertEqual(status, "baseline_sufficient_no_incremental_support")
        self.assertEqual(len(client.controls), 534)
        self.assertEqual(
            {value["intervention_mode"] for value in client.controls},
            {"none", "residual_schedule"},
        )
        self.assertEqual(environment.rollout_calls, 0)
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["request_accounting"]["total"], 534)
        self.assertEqual(payload["request_accounting"]["cem"], 520)
        self.assertEqual(payload["request_accounting"]["trace_rows"], 528)
        self.assertEqual(
            payload["search_runtime"],
            {
                "numpy_version": np.__version__,
                "bit_generator": "PCG64",
                "seed": 20260716,
                "generations": 8,
                "population_size": 65,
                "search_requests": 520,
            },
        )
        self.assertEqual(payload["execution_boundary"]["policy_generated_action_steps_executed"], 0)
        ledger = json.loads((payload_path.parent / "query-ledger.json").read_text(encoding="utf-8"))
        self.assertEqual([row["ordinal"] for row in ledger["rows"]], list(range(534)))
        self.assertEqual(ledger["rows"][0]["phase"], "compiled_frozen_pre")
        self.assertEqual(ledger["rows"][-1]["phase"], "compiled_frozen_post")
        tensor_path = payload_path.parent / "af00a-tensors.npz"
        with np.load(tensor_path, allow_pickle=False) as archive:
            self.assertEqual(set(archive.files), set(self.array_specs))
            self.assertEqual(archive["cem_raw_normals_f64"].shape, (8, 32, 5, 15))
            self.assertEqual(archive["cem_executed_c_f32"].shape, (520, 5, 15))
            self.assertEqual(archive["cem_trace_x_t_f32"].shape, (520, 10, 10, 32))
            self.assertTrue(
                np.array_equal(
                    archive["cem_final_normalized_physical_f64"],
                    archive["cem_returned_actions_f64"],
                )
            )
        validated = self.validate_npz(tensor_path, payload["request_ledger"])
        self.assertEqual(validated["outcome"], status)

    def test_arm_a_repeatability_failure_stops_before_cem(self):
        client = self.Client(self, corrupt_duplicate=True)
        with self.assertRaisesRegex(
            self.canary.ActualForwardCanaryError,
            "Arm A repeatability|arm_a_equal_split failed ordinary replay validation",
        ):
            self._run(client)
        self.assertEqual(len(client.controls), 6)

    def test_fail_closed_config_stops_before_any_request(self):
        client = self.Client(self)
        closed = copy.deepcopy(self.config)
        closed["ready_to_run"] = False
        with self.assertRaisesRegex(self.canary.ActualForwardCanaryError, "not execution-released"):
            self.canary.run_actual_forward_canary(
                self.case,
                closed,
                self.legacy,
                repo_root=ROOT,
                input_manifest_sha256=self.canary.MANIFEST_SHA256,
                actual_config_path=self.actual_path,
                legacy_config_path=self.legacy_path,
                client=client,
                environment=self.Environment(),
            )
        self.assertEqual(client.controls, [])


if __name__ == "__main__":
    unittest.main()
