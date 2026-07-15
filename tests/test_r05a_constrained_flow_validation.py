from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
IMPORT_ERROR: Exception | None = None
try:
    import numpy as np

    sys.path.insert(0, str(ROOT / "main"))
    from crfs_harness.artifacts import file_sha256  # noqa: E402
    from crfs_oracle import r05a_constrained_flow_canary as cfs  # noqa: E402
    from crfs_oracle import r05a_constrained_flow_validation as validation  # noqa: E402
    from crfs_oracle.r05a_canary import SOLVER_CONFIG, _expected_times  # noqa: E402
    from crfs_oracle.r02_runner import _array_record, _trace_record  # noqa: E402
    from crfs_oracle.r03a_runner import _trace_from_record  # noqa: E402
except (ImportError, ModuleNotFoundError) as exc:  # dependency-free local gate.
    IMPORT_ERROR = exc


@unittest.skipIf(
    IMPORT_ERROR is not None,
    f"constrained-flow semantic checks require allocation dependencies: {IMPORT_ERROR}",
)
class ConstrainedFlowSemanticValidatorTest(unittest.TestCase):
    def _config_fixture(self, root: Path) -> tuple[dict, Path, Path]:
        config = json.loads(
            (ROOT / "configs/experiments/r05a_constrained_flow_canary.json").read_text(
                encoding="utf-8"
            )
        )
        config_path = root / "constrained.json"
        config_path.write_text(json.dumps(config, sort_keys=True) + "\n", encoding="utf-8")
        legacy_path = ROOT / "configs/experiments/r05a_inverse_flow_canary.json"
        return config, config_path, legacy_path

    def _failure_payload(
        self, root: Path, config: dict, config_path: Path, legacy_config_path: Path
    ) -> dict:
        legacy_payload = root / "case/canary-payload.json"
        return {
            "schema_version": "1.0",
            "payload_type": cfs.PAYLOAD_TYPE,
            "payload_variant": "terminal_apparatus_failure",
            "status": "apparatus_inconclusive",
            "case_id": cfs.CASE_ID,
            "run_id": "r05a-cfs-validator-fixture",
            "config": {
                "constrained_flow_path": str(config_path),
                "constrained_flow_sha256": file_sha256(config_path),
                "constrained_flow_scientific_hash": cfs.constrained_flow_scientific_config_hash(
                    config
                ),
                "legacy_path": str(legacy_config_path),
                "legacy_sha256": file_sha256(legacy_config_path),
            },
            "failure": {
                "stage": "paired_transport_execution",
                "error_type": "RuntimeError",
                "message": "registered fixture failure",
                "finite_failure_is_infeasibility": False,
                "numeric_failure_is_method_negative": False,
            },
            "legacy_payload": {
                "path": str(legacy_payload),
                "exists": False,
                "sha256": None,
            },
            "provenance": {
                "git_commit": "1" * 40,
                "source_node": "worker-1",
                "slurm_job_id": "123_0",
                "slurm_array_job_id": "123",
                "slurm_array_task_id": "0",
                "timestamp_utc": "2026-07-15T00:00:00+00:00",
            },
            "outcome": {
                "status": "apparatus_inconclusive",
                "scientific_claim_allowed": False,
                "infeasibility_claim_allowed": False,
                "collision_or_progress_claim_allowed": False,
                "probe_training_authorized": False,
                "automatic_next_gate_authorized": False,
            },
            "simulator_use": dict(validation.SIMULATOR_USE),
        }

    def test_terminal_failure_is_valid_but_cannot_be_promoted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, config_path, legacy_path = self._config_fixture(root)
            payload = self._failure_payload(root, config, config_path, legacy_path)
            self.assertEqual(
                validation.validate_constrained_flow_payload_or_raise(
                    payload,
                    config,
                    expected_run_id=payload["run_id"],
                    constrained_config_path=config_path,
                    legacy_config_path=legacy_path,
                ),
                "apparatus_inconclusive",
            )

            for mutation in (
                lambda value: value["outcome"].__setitem__(
                    "scientific_claim_allowed", True
                ),
                lambda value: value["provenance"].__setitem__(
                    "source_node", "worker-2"
                ),
                lambda value: value["failure"].__setitem__(
                    "numeric_failure_is_method_negative", True
                ),
                lambda value: value["legacy_payload"].__setitem__(
                    "sha256", "0" * 64
                ),
            ):
                changed = copy.deepcopy(payload)
                mutation(changed)
                self.assertTrue(
                    validation.validate_constrained_flow_payload(
                        changed,
                        config,
                        expected_run_id=payload["run_id"],
                        constrained_config_path=config_path,
                        legacy_config_path=legacy_path,
                    )
                )

    def test_complete_path_recomputes_status_and_rejects_claim_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, config_path, legacy_config_path = self._config_fixture(root)
            run_id = "r05a-cfs-complete-fixture"
            schedule = np.zeros((10, 10, 32), dtype=np.float32)
            final = np.zeros((10, 32), dtype=np.float32)
            actions = np.zeros((10, 7), dtype=np.float64)
            solver_trace = _trace_record(
                {"solver_internal_replay_final": final.copy()}
            )
            solver_record = {
                "schedule": _array_record(schedule),
                "actions": _array_record(actions),
                "trace": solver_trace,
            }
            legacy = {
                "schema_version": "1.0",
                "payload_type": validation.LEGACY_PAYLOAD_TYPE,
                "complete_result_requires_memory_finalization": True,
                "result_without_memory": {
                    "run_id": run_id,
                    "case_id": cfs.CASE_ID,
                    "status": "completed_nonconverged",
                    "solver": {
                        "first": solver_record,
                        "duplicate": copy.deepcopy(solver_record),
                    },
                    "determinism": {"passed": True},
                    "simulator_use": {
                        "policy_generated_action_steps_executed": 0,
                        "teacher_generated_action_steps_executed": 0,
                        "efficacy_rollouts_executed": 0,
                    },
                },
            }
            legacy_path = root / "canary-payload.json"
            legacy_path.write_text(
                json.dumps(legacy, sort_keys=True) + "\n", encoding="utf-8"
            )
            arm_a = {"passed": True, "checks": {"fixture": True}}
            source_checks = {"request": True, "trace": True}
            duplicate_checks = {"duplicate": True}
            paired_checks = {
                "exactly_two_comparison_calls": True,
                "ordinary_teacher_requests_exact": True,
                "comparison_adapter_duplicates_exact": True,
                "comparison_source_pairing_exact": True,
            }
            apparatus_checks = {
                "arm_a_historical_reproduced": True,
                "arm_b_valid": True,
                "arm_c_valid": True,
                "arm_a_canonical_replays": True,
                "arm_b_canonical_replays": True,
                "arm_c_canonical_replays": True,
                "duplicates_valid": True,
                "paired_requests_valid": True,
                "zero_policy_generated_simulator_steps": True,
                "zero_teacher_generated_simulator_steps": True,
                "zero_efficacy_rollouts": True,
            }
            replay_return = {
                "schedule": schedule,
                "selected_final": final,
                "actions": actions,
                "trace": {"fixture": True},
                "metrics": cfs._fidelity_metrics(final),
                "schedule_diagnostics": {"fixture": True},
            }
            replay_arm = {
                "first": {},
                "duplicate": {},
                "duplicate_checks": {
                    "schedule_exact": True,
                    "selected_final_exact": True,
                    "returned_actions_exact": True,
                    "replay_final_exact": True,
                    "recurrence_trace_exact": True,
                },
                "passed": True,
            }
            arm_tuple = (
                {
                    "passed": True,
                    "linear_metrics": {"passed": True},
                },
                schedule,
                final,
                {},
            )
            outcome = {
                "status": "mechanism_pass",
                "arm_b_linear_prediction_passed": True,
                "arm_b_nonlinear_replay_passed": True,
                "arm_c_nonlinear_replay_passed": True,
                "finite_nonconvergence_is_infeasibility": False,
                "optimality_certificate": False,
                "infeasibility_certificate": False,
                "simulator_efficacy_evaluated": False,
                "collision_or_progress_claim_allowed": False,
                "student_or_generalization_claim_allowed": False,
                "probe_training_authorized": False,
                "automatic_next_gate_authorized": False,
            }
            payload = {
                "schema_version": "1.0",
                "payload_type": cfs.PAYLOAD_TYPE,
                "payload_variant": "complete_comparison",
                "status": "mechanism_pass",
                "case_id": cfs.CASE_ID,
                "run_id": run_id,
                "legacy_payload_path": str(legacy_path),
                "legacy_payload_sha256": file_sha256(legacy_path),
                "config": {
                    "constrained_flow_path": str(config_path),
                    "constrained_flow_sha256": file_sha256(config_path),
                    "constrained_flow_scientific_hash": cfs.constrained_flow_scientific_config_hash(
                        config
                    ),
                    "legacy_path": str(legacy_config_path),
                    "legacy_sha256": file_sha256(legacy_config_path),
                },
                "source_pairing": {
                    "comparison_checks": source_checks,
                    "comparison_checks_passed": True,
                },
                "arms": {
                    "A_historical_run_b": arm_a,
                    "B_linearized_candidate": {"first": {}, "duplicate": {}},
                    "C_linearized_then_historical_adam": {
                        "first": {},
                        "duplicate": {},
                    },
                },
                "duplicates": {
                    "checks": duplicate_checks,
                    "passed": True,
                    "comparison_elapsed_seconds": [0.1, 0.1],
                },
                "canonical_replays": {
                    "A_historical_run_b": copy.deepcopy(replay_arm),
                    "B_linearized_candidate": copy.deepcopy(replay_arm),
                    "C_linearized_then_historical_adam": copy.deepcopy(replay_arm),
                },
                "apparatus": {
                    "checks": apparatus_checks,
                    "paired_request_checks": paired_checks,
                    "passed": True,
                },
                "outcome": outcome,
                "simulator_use": {
                    "setup_only": True,
                    **validation.SIMULATOR_USE,
                },
            }
            patchers = (
                mock.patch.object(
                    validation,
                    "_source_arrays",
                    return_value=(
                        final,
                        final,
                        np.ones_like(final),
                        final,
                        np.float32(3.6398398876190186),
                    ),
                ),
                mock.patch.object(
                    validation,
                    "_validate_request_records",
                    return_value={"request": True},
                ),
                mock.patch.object(
                    validation,
                    "_validate_comparison_trace_pairing",
                    return_value={"trace": True},
                ),
                mock.patch.object(
                    validation,
                    "_recompute_duplicate_checks",
                    return_value=duplicate_checks,
                ),
                mock.patch.object(
                    validation, "_validate_arm_b", return_value=arm_tuple
                ),
                mock.patch.object(
                    validation, "_validate_arm_c", return_value=arm_tuple
                ),
                mock.patch.object(
                    validation, "_validate_replay", return_value=replay_return
                ),
                mock.patch.object(
                    cfs, "_legacy_arm_a_summary", return_value=arm_a
                ),
            )
            with patchers[0], patchers[1], patchers[2], patchers[3], patchers[
                4
            ], patchers[5], patchers[6], patchers[7]:
                self.assertEqual(
                    validation.validate_constrained_flow_payload_or_raise(
                        payload,
                        config,
                        legacy_payload=legacy,
                        expected_run_id=run_id,
                        constrained_config_path=config_path,
                        legacy_config_path=legacy_config_path,
                    ),
                    "mechanism_pass",
                )
                changed = copy.deepcopy(payload)
                changed["outcome"]["collision_or_progress_claim_allowed"] = True
                with self.assertRaisesRegex(
                    validation.ConstrainedFlowValidationError, "claim boundary"
                ):
                    validation.validate_constrained_flow_payload_or_raise(
                        changed,
                        config,
                        legacy_payload=legacy,
                        expected_run_id=run_id,
                        constrained_config_path=config_path,
                        legacy_config_path=legacy_config_path,
                    )

    def _request_source(self) -> tuple[dict, np.ndarray, np.ndarray, np.float32]:
        target = np.zeros((10, 32), dtype=np.float32)
        noise = np.ones((10, 32), dtype=np.float32)
        budget = np.float32(3.6398398876190186)
        controls = {
            "noise": noise.copy(),
            "intervention_mode": "inverse_flow_teacher",
            "intervention_step": 5,
            "return_trace": True,
            "return_normalized_final": True,
            "target": target.copy(),
            "target_space": "model",
            "model_l2_path_budget": budget,
            "solver_config": dict(SOLVER_CONFIG),
        }
        observation = {"image": np.zeros((2, 2, 3), dtype=np.uint8)}
        records = {}
        for replicate in ("first", "duplicate"):
            comparison = {**controls, "experiment_arm": cfs.EXPERIMENT_ARM}
            records[replicate] = {
                "ordinary_observation": cfs._tree_record(observation),
                "comparison_observation": cfs._tree_record(observation),
                "ordinary_observation_sha256": __import__(
                    "crfs_harness.artifacts", fromlist=["content_hash"]
                ).content_hash(cfs._tree_record(observation)),
                "comparison_observation_sha256": __import__(
                    "crfs_harness.artifacts", fromlist=["content_hash"]
                ).content_hash(cfs._tree_record(observation)),
                "ordinary_controls": cfs._tree_record(controls),
                "comparison_controls": cfs._tree_record(comparison),
            }
        return {"comparison_request_records": records}, target, noise, budget

    def test_request_pairing_binds_full_duplicate_observation_and_controls(self) -> None:
        source, target, noise, budget = self._request_source()
        checks = validation._validate_request_records(
            source, target=target, noise=noise, budget=budget
        )
        self.assertTrue(all(checks.values()))

        changed = copy.deepcopy(source)
        changed_observation = {"image": np.ones((2, 2, 3), dtype=np.uint8)}
        record = cfs._tree_record(changed_observation)
        changed["comparison_request_records"]["duplicate"][
            "ordinary_observation"
        ] = record
        changed["comparison_request_records"]["duplicate"][
            "ordinary_observation_sha256"
        ] = __import__("crfs_harness.artifacts", fromlist=["content_hash"]).content_hash(
            record
        )
        changed["comparison_request_records"]["duplicate"][
            "comparison_observation"
        ] = record
        changed["comparison_request_records"]["duplicate"][
            "comparison_observation_sha256"
        ] = __import__("crfs_harness.artifacts", fromlist=["content_hash"]).content_hash(
            record
        )
        with self.assertRaisesRegex(
            validation.ConstrainedFlowValidationError,
            "duplicate ordinary observations",
        ):
            validation._validate_request_records(
                changed, target=target, noise=noise, budget=budget
            )

    def _canonical_replay_fixture(self) -> tuple[dict, dict]:
        noise = np.zeros((10, 32), dtype=np.float32)
        schedule = np.zeros((10, 10, 32), dtype=np.float32)
        zero_steps = np.zeros((10, 10, 32), dtype=np.float32)
        trace = {
            "step_index_steps": np.arange(10, dtype=np.int64),
            "time_steps": _expected_times(),
            "active_steps": np.asarray([False] * 5 + [True] * 5, dtype=np.bool_),
            "x_t_steps": zero_steps.copy(),
            "v_base_steps": zero_steps.copy(),
            "control_velocity_steps": schedule.copy(),
            "total_velocity_steps": zero_steps.copy(),
            "control_increment_steps": np.asarray(-0.1, dtype=np.float32)
            * schedule,
            "x_next_steps": zero_steps.copy(),
            "initial_noise": noise.copy(),
            "final_normalized": noise.copy(),
            "final_normalized_physical": np.zeros((10, 7), dtype=np.float64),
            "control_source": np.asarray(1, dtype=np.int64),
            "control_valid": np.asarray(True, dtype=np.bool_),
            "schedule_applied": np.asarray(True, dtype=np.bool_),
            "schedule_budget": np.asarray(
                3.6398398876190186, dtype=np.float32
            ),
            "dt": np.asarray(-0.1, dtype=np.float32),
            "intervention_step": np.asarray(5, dtype=np.int64),
            "num_steps": np.asarray(10, dtype=np.int64),
        }
        reply = {"actions": np.zeros((10, 7), dtype=np.float64), "crfs_trace": trace}
        record = cfs._canonical_replay_record(
            reply,
            schedule=schedule,
            budget=np.float32(3.6398398876190186),
            noise=noise,
            target=noise,
            scale=np.ones((10, 32), dtype=np.float32),
            selected_final=noise,
            elapsed=0.01,
        )
        return record, trace

    def test_canonical_replay_recomputes_raw_control_invariants(self) -> None:
        record, _trace = self._canonical_replay_fixture()
        kwargs = {
            "schedule": np.zeros((10, 10, 32), dtype=np.float32),
            "selected_final": np.zeros((10, 32), dtype=np.float32),
            "target": np.zeros((10, 32), dtype=np.float32),
            "noise": np.zeros((10, 32), dtype=np.float32),
            "scale": np.ones((10, 32), dtype=np.float32),
            "budget": np.float32(3.6398398876190186),
            "action_dtype": np.dtype(np.float64),
        }
        validation._validate_replay(record, **kwargs)
        for key, replacement in (
            ("control_source", np.asarray(0, dtype=np.int64)),
            ("schedule_applied", np.asarray(False, dtype=np.bool_)),
            ("schedule_budget", np.asarray(1.0, dtype=np.float32)),
            ("intervention_step", np.asarray(4, dtype=np.int64)),
        ):
            changed = copy.deepcopy(record)
            decoded = dict(
                _trace_from_record(changed["trace"], name="mutated replay trace")
            )
            decoded[key] = replacement
            changed["trace"] = _trace_record(decoded)
            with self.assertRaises(validation.ConstrainedFlowValidationError):
                validation._validate_replay(changed, **kwargs)


if __name__ == "__main__":
    unittest.main()
