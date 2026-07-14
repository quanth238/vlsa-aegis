from __future__ import annotations

import copy
import json
from pathlib import Path
import unittest

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - base harness intentionally skips deps
    np = None


@unittest.skipIf(np is None, "NumPy is not installed in the dependency-free harness")
class R03ARunnerContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from crfs_harness.artifacts import file_sha256
        from crfs_oracle.r03a_runner import r03a_config_from_mapping
        from crfs_oracle.r03a_validation import r03a_config_hash
        from crfs_oracle.runner import oracle_config_from_mapping

        cls.root = Path(__file__).resolve().parents[1]
        cls.config_path = cls.root / "configs/experiments/r03a_analytic_kill_test.json"
        cls.value = json.loads(cls.config_path.read_text(encoding="utf-8"))
        runtime = copy.deepcopy(cls.value)
        runtime.setdefault("intervention_step", 5)
        oracle = oracle_config_from_mapping(
            runtime,
            host="127.0.0.1",
            port=8000,
            checkpoint_id="/immutable/checkpoint",
            checkpoint_sha256=(
                "988055ccfd7032903c073a641f3c5f0f0541df444a315116a16f0bf4716d26ed"
            ),
            output_root="/immutable/output",
            run_id="r03a-unit",
        )
        cls.config = r03a_config_from_mapping(
            runtime,
            oracle,
            repo_root=cls.root,
            config_file_sha256=file_sha256(cls.config_path),
            scientific_config_hash=r03a_config_hash(cls.value),
        )

    def test_frozen_config_loads_and_binds_seventeen_cases(self) -> None:
        self.assertEqual(len(self.config.eligible_case_ids), 17)
        self.assertEqual(self.config.timing_warmup_repeats, 2)
        self.assertEqual(self.config.timing_measured_repeats, 2)
        self.assertEqual(
            self.config.scientific_config_hash,
            __import__(
                "crfs_oracle.r03a_validation", fromlist=["r03a_config_hash"]
            ).r03a_config_hash(self.value),
        )

    def test_config_rejects_post_registration_energy_change(self) -> None:
        from crfs_harness.artifacts import file_sha256
        from crfs_oracle.r03a_runner import r03a_config_from_mapping
        from crfs_oracle.r03a_validation import r03a_config_hash
        from crfs_oracle.runner import oracle_config_from_mapping

        changed = copy.deepcopy(self.value)
        changed["r03a"]["energy_margin_m"] = 0.006
        changed.setdefault("intervention_step", 5)
        oracle = oracle_config_from_mapping(
            changed,
            host="127.0.0.1",
            port=8000,
            checkpoint_id="/immutable/checkpoint",
            checkpoint_sha256=(
                "988055ccfd7032903c073a641f3c5f0f0541df444a315116a16f0bf4716d26ed"
            ),
            output_root="/immutable/output",
            run_id="r03a-unit",
        )
        with self.assertRaisesRegex(ValueError, "energy margin"):
            r03a_config_from_mapping(
                changed,
                oracle,
                repo_root=self.root,
                config_file_sha256=file_sha256(self.config_path),
                scientific_config_hash=r03a_config_hash(changed),
            )

    def test_inverted_cdf_timing_is_registered(self) -> None:
        from crfs_oracle.r03a_runner import _quantiles

        self.assertEqual(
            _quantiles([2.0, 1.0]),
            {"values": [2.0, 1.0], "p50": 1.0, "p95": 2.0},
        )

    def test_trace_pairing_diagnostic_reports_keys_bytes_and_error(self) -> None:
        from crfs_oracle.r03a_runner import _trace_pairing_diagnostics

        source = {
            "x_t": np.asarray([[1.0, 2.0]], dtype=np.float32),
            "time": np.asarray([0.5], dtype=np.float32),
        }
        fresh = {
            "x_t": np.asarray([[1.0, 2.25]], dtype=np.float32),
            "extra": np.asarray([1], dtype=np.int64),
        }
        diagnostic = _trace_pairing_diagnostics(source, fresh)
        self.assertEqual(diagnostic["missing_from_fresh"], ["time"])
        self.assertEqual(diagnostic["extra_in_fresh"], ["extra"])
        leaf = diagnostic["leaves"]["x_t"]
        self.assertTrue(leaf["same_dtype"])
        self.assertTrue(leaf["same_shape"])
        self.assertFalse(leaf["native_bytes_equal"])
        self.assertEqual(leaf["maximum_absolute_error"], 0.25)
        self.assertNotEqual(leaf["source_sha256"], leaf["fresh_sha256"])

    def test_unrun_policy_and_nonfinite_failures_remain_in_denominator(self) -> None:
        from crfs_oracle.r03a_runner import (
            _analytic_failure_status,
            _unrun_analytic_failure_arm,
        )

        self.assertEqual(
            _analytic_failure_status(RuntimeError("server disconnected")),
            "policy_failure",
        )
        self.assertEqual(
            _analytic_failure_status(RuntimeError("produced a nonfinite gradient")),
            "nonfinite_failure",
        )
        arm = _unrun_analytic_failure_arm(
            "analytic_trajectory_mid",
            status="policy_failure",
            reason="RuntimeError: server disconnected",
            budget=0.3,
        )
        self.assertEqual(arm["status"], "policy_failure")
        self.assertIs(arm["gate"]["passed"], False)
        self.assertIsNone(arm["full_actions"])
        self.assertEqual(arm["repeats"], [])
        self.assertIs(arm["timing"]["warmed_batch_one"], False)

    def test_only_client_inference_exceptions_become_policy_failures(self) -> None:
        from crfs_oracle.r03a_runner import R03APolicyInferenceError, _request

        class FailedClient:
            def infer(self, _request):
                raise RuntimeError("server disconnected")

        class MalformedReplyClient:
            def infer(self, _request):
                return {}

        with self.assertRaisesRegex(R03APolicyInferenceError, "server disconnected"):
            _request(FailedClient(), {"observation/state": [0.0]}, {})
        with self.assertRaisesRegex(RuntimeError, "no actions"):
            _request(MalformedReplyClient(), {"observation/state": [0.0]}, {})

    def test_saturation_uses_exact_inclusive_bound_without_epsilon(self) -> None:
        from crfs_oracle.r03a_runner import _exact_translation_saturation

        actions = np.zeros((10, 7), dtype=np.float64)
        actions[4, 2] = 1.0
        self.assertTrue(_exact_translation_saturation(actions, self.config))
        actions[4, 2] = np.nextafter(1.0, 0.0)
        self.assertFalse(_exact_translation_saturation(actions, self.config))

    def test_budget_uses_only_first_five_xyz(self) -> None:
        from crfs_oracle.r02_runner import _array_record
        from crfs_oracle.r03a_runner import _source_budget

        delta = np.zeros((10, 32), dtype=np.float64)
        delta[:5, :3] = 0.25
        norm = float(np.linalg.norm(delta[:5, :3]))
        raw = {
            "directions": {
                "arrays": {"delta_star_model": _array_record(delta)},
                "l2_norms": {"delta_star_model": norm},
            }
        }
        record, observed = _source_budget(raw)
        self.assertEqual(observed, norm)
        self.assertEqual(record["first_five_xyz_model_l2"], norm)
        delta[5, 0] = 1.0
        raw["directions"]["arrays"]["delta_star_model"] = _array_record(delta)
        with self.assertRaisesRegex(ValueError, "translation mask"):
            _source_budget(raw)

    def _valid_analytic_trace(self) -> tuple[dict, dict, np.ndarray]:
        active = np.arange(10) >= 5
        x_steps = np.zeros((10, 10, 32), dtype=np.float32)
        trace = {
            "step_index": np.arange(10, dtype=np.int64),
            "time": np.asarray(
                [np.float32(1.0) + np.float32(i) * np.float32(-0.1) for i in range(10)],
                dtype=np.float32,
            ),
            "active": active.copy(),
            "field_evaluated": active.copy(),
            "x_t_steps": x_steps,
            "v_base_steps": np.zeros_like(x_steps),
            "predicted_clean_steps": np.zeros_like(x_steps),
            "physical_predicted_clean_xyz_steps": np.zeros((10, 5, 3), dtype=np.float32),
            "predicted_eef_centers_m_steps": np.zeros((10, 5, 3), dtype=np.float32),
            "hard_min_clearance_m": np.where(active, 0.006, 0.0).astype(np.float32),
            "energy": np.zeros(10, dtype=np.float32),
            "energy_gradient_steps": np.zeros_like(x_steps),
            "normalized_energy_gradient_steps": np.zeros_like(x_steps),
            "gradient_l2": np.zeros(10, dtype=np.float32),
            "margin_satisfied": active,
            "gradient_finite": active,
            "gradient_valid": np.zeros(10, dtype=np.bool_),
            "applied": np.zeros(10, dtype=np.bool_),
            "guidance_velocity_steps": np.zeros_like(x_steps),
            "guidance_velocity_l2": np.zeros(10, dtype=np.float32),
            "path_increment_l2": np.zeros(10, dtype=np.float32),
            "cumulative_integrated_field_l2": np.zeros(10, dtype=np.float32),
            "analytic_gradient_ms": np.where(active, 1.0, 0.0).astype(np.float32),
            "final_normalized": np.zeros((10, 32), dtype=np.float32),
            "final_normalized_physical": np.zeros((10, 7), dtype=np.float32),
            "intervention_step": np.int64(5),
            "dt": np.float32(-0.1),
            "active_horizon": np.float32(0.5),
            "model_l2_path_budget": np.float32(0.3),
            "velocity_gain": np.float32(0.6),
            "safety_margin_m": np.float32(0.005),
            "softplus_tau_m": np.float32(0.005),
            "samples_per_segment": np.int64(26),
            "integrated_field_l2": np.float32(0.0),
        }
        frozen = {
            "step_index": np.int64(5),
            "x_t": np.zeros((10, 32), dtype=np.float32),
            "v_base": np.zeros((10, 32), dtype=np.float32),
            "predicted_clean": np.zeros((10, 32), dtype=np.float32),
            "predicted_clean_physical": np.zeros((10, 7), dtype=np.float32),
        }
        actions = np.zeros((10, 7), dtype=np.float32)
        return trace, frozen, actions

    def test_analytic_trace_binds_euler_recurrence_and_physical_reply(self) -> None:
        from crfs_oracle.r03a_runner import _validate_analytic_trace

        trace, frozen, actions = self._valid_analytic_trace()
        diagnostics = _validate_analytic_trace(
            trace,
            reply_actions=actions,
            intervention_step=5,
            budget=0.3,
            config=self.config,
            frozen_reference_trace=frozen,
        )
        self.assertTrue(diagnostics["physical_reply_exact"])
        self.assertEqual(diagnostics["euler_recurrence_max_abs_error"], 0.0)
        self.assertEqual(diagnostics["euler_terminal_max_abs_error"], 0.0)

    def test_analytic_trace_rejects_disconnected_trajectory_or_reply(self) -> None:
        from crfs_oracle.r03a_runner import _validate_analytic_trace

        trace, frozen, actions = self._valid_analytic_trace()
        broken_recurrence = copy.deepcopy(trace)
        broken_recurrence["x_t_steps"][6, 0, 0] = 0.1
        broken_recurrence["predicted_clean_steps"][6, 0, 0] = 0.1
        broken_recurrence["physical_predicted_clean_xyz_steps"][6, 0, 0] = (
            np.float32(0.1) * np.float32(self.config.action_scale[0])
        )
        with self.assertRaisesRegex(RuntimeError, "Euler recurrence"):
            _validate_analytic_trace(
                broken_recurrence,
                reply_actions=actions,
                intervention_step=5,
                budget=0.3,
                config=self.config,
                frozen_reference_trace=frozen,
            )

        broken_reply = actions.copy()
        broken_reply[0, 0] = 0.1
        with self.assertRaisesRegex(RuntimeError, "physical action reply"):
            _validate_analytic_trace(
                trace,
                reply_actions=broken_reply,
                intervention_step=5,
                budget=0.3,
                config=self.config,
                frozen_reference_trace=frozen,
            )

        wrong_dtype_reply = actions.astype(np.float64)
        with self.assertRaisesRegex(RuntimeError, "dtype differs"):
            _validate_analytic_trace(
                trace,
                reply_actions=wrong_dtype_reply,
                intervention_step=5,
                budget=0.3,
                config=self.config,
                frozen_reference_trace=frozen,
            )

    def test_analytic_trace_rejects_forged_field_and_budget_telemetry(self) -> None:
        from crfs_oracle.r03a_runner import _validate_analytic_trace

        mutations = []
        trace, frozen, actions = self._valid_analytic_trace()
        changed = copy.deepcopy(trace)
        changed["field_evaluated"][7] = False
        mutations.append((changed, "field-evaluated mask"))

        changed = copy.deepcopy(trace)
        changed["predicted_clean_steps"][7, 0, 0] = np.float32(0.1)
        mutations.append((changed, "approximate-clean"))

        changed = copy.deepcopy(trace)
        changed["guidance_velocity_steps"][7, 0, 0] = np.float32(1.0)
        mutations.append((changed, "guidance velocity"))

        changed = copy.deepcopy(trace)
        changed["path_increment_l2"][7] = np.float32(0.1)
        changed["cumulative_integrated_field_l2"][7:] = np.float32(0.1)
        changed["integrated_field_l2"] = np.float32(0.1)
        mutations.append((changed, "path increment"))

        changed = copy.deepcopy(trace)
        changed["x_t_steps"] = changed["x_t_steps"].astype(np.float64)
        mutations.append((changed, "dtype float32"))

        for changed, message in mutations:
            with self.subTest(message=message), self.assertRaisesRegex(RuntimeError, message):
                _validate_analytic_trace(
                    changed,
                    reply_actions=actions,
                    intervention_step=5,
                    budget=0.3,
                    config=self.config,
                    frozen_reference_trace=frozen,
                )


if __name__ == "__main__":
    unittest.main()
