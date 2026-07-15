from __future__ import annotations

import ast
import hashlib
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
CANARY_PATH = ROOT / "main" / "crfs_oracle" / "r05a_canary.py"
RUNNER_PATH = ROOT / "scripts" / "hpc" / "run_r05a_canary.sh"
SUBMIT_PATH = ROOT / "scripts" / "hpc" / "submit_r05a_canary.sh"
GPU_SLURM_PATH = ROOT / "slurm" / "r05a_canary_h100.sbatch"


class R05ACanaryStructuralTest(unittest.TestCase):
    def test_canary_is_source_pinned_apparatus_only_and_never_steps_policy_actions(self) -> None:
        source = CANARY_PATH.read_text(encoding="utf-8")
        ast.parse(source)
        self.assertIn('SOURCE_HOST = "worker-1"', source)
        self.assertIn('"policy_generated_action_steps_executed": 0', source)
        self.assertIn('"teacher_generated_action_steps_executed": 0', source)
        self.assertIn('"simulator_efficacy_evaluated": False', source)
        self.assertNotIn("environment.rollout(", source)
        self.assertNotIn("environment.env.step(", source)
        self.assertIn("#SBATCH --mem=64G", GPU_SLURM_PATH.read_text(encoding="utf-8"))

    def test_submission_registers_exact_held_gpu_then_afterany_before_release(self) -> None:
        source = SUBMIT_PATH.read_text(encoding="utf-8")
        held_submit = source.index("sbatch --parsable --hold")
        held_receipt = source.index('"$run_root/held-gpu-submission.json"')
        dependency = source.index("dependency=afterany:$gpu_job_id")
        cpu_block = source.index("cpu_submission=$(", dependency)
        cpu_submit = source.index("sbatch --parsable", cpu_block)
        release = source.index('scontrol release "$gpu_job_id"')
        self.assertLess(held_submit, held_receipt)
        self.assertLess(held_receipt, dependency)
        self.assertLess(dependency, cpu_submit)
        self.assertLess(cpu_submit, release)
        self.assertIn("--nodelist=worker-1", source)
        self.assertIn("test ! -e \"$run_root\"", source)

    def test_validator_recomputes_raw_source_calls_and_direct_witness(self) -> None:
        source = CANARY_PATH.read_text(encoding="utf-8")
        for fragment in (
            "actual_source_sha = file_sha256(source_path)",
            "validate_r02_result(raw_r02)",
            'raw_source_pairing.get("eager_actions")',
            'raw_source_pairing.get("eager_trace")',
            'raw_r02.get("arms", {}).get("direct_witness", {}).get("executed_actions")',
            "recomputed_direct_target",
            "and recomputed_direct_target",
            "_parse_allocation_test_log(log_path)",
            "_read_gpu_samples(gpu_samples_path)",
            "CONFIG_FILE_SHA256",
            "SCIENTIFIC_CONFIG_HASH",
            "summary_checks",
            "finite search cannot claim an infeasibility certificate",
            "finite search cannot claim an optimality certificate",
            'outcome.get("teacher_converged") is raw_first_converged',
        ):
            self.assertIn(fragment, source)


IMPORT_ERROR: Exception | None = None
try:
    import numpy as np

    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "main"))
    sys.path.insert(0, str(ROOT / "safelibero"))
    from crfs_oracle import r05a_canary as canary
except ModuleNotFoundError as error:  # pragma: no cover - dependency-free gate.
    IMPORT_ERROR = error


@unittest.skipIf(
    IMPORT_ERROR is not None,
    f"R05A runtime contract checks require allocation dependencies: {IMPORT_ERROR}",
)
class R05ACanaryRuntimeTest(unittest.TestCase):
    def _zero_trace(self):
        dt = np.asarray(-0.1, dtype=np.float32)
        x_t = np.zeros((10, 10, 32), dtype=np.float32)
        v_base = np.zeros_like(x_t)
        control = np.zeros_like(x_t)
        total = np.where(control == 0, v_base, v_base + control)
        increment = dt * control
        x_next = np.asarray(x_t + dt * total, dtype=np.float32)
        return {
            "step_index_steps": np.arange(10, dtype=np.int64),
            "time_steps": canary._expected_times(),
            "active_steps": np.asarray([False] * 5 + [True] * 5, dtype=np.bool_),
            "x_t_steps": x_t,
            "v_base_steps": v_base,
            "control_velocity_steps": control,
            "total_velocity_steps": total,
            "control_increment_steps": increment,
            "x_next_steps": x_next,
            "final_normalized": x_next[-1].copy(),
            "initial_noise": np.zeros((10, 32), dtype=np.float32),
        }

    def test_recurrence_recomputation_rejects_a_mutated_state(self) -> None:
        trace = self._zero_trace()
        self.assertEqual(canary.validate_flow_recurrence(trace), [])
        broken = {key: np.array(value, copy=True) for key, value in trace.items()}
        broken["x_next_steps"][5, 0, 0] = np.float32(1.0)
        errors = canary.validate_flow_recurrence(broken)
        self.assertTrue(any("x_next" in error or "disconnected" in error for error in errors))

    def test_schedule_validation_rejects_timing_mask_and_budget_mutations(self) -> None:
        budget = np.float32(0.1)
        schedule = np.zeros((10, 10, 32), dtype=np.float32)
        _record, errors = canary._schedule_diagnostics(
            schedule, source_budget_float32=budget
        )
        self.assertEqual(errors, [])

        inactive = schedule.copy()
        inactive[0, 0, 0] = np.float32(0.1)
        self.assertTrue(
            any(
                "steps 0..4" in error
                for error in canary._schedule_diagnostics(
                    inactive, source_budget_float32=budget
                )[1]
            )
        )
        outside = schedule.copy()
        outside[5, 5, 0] = np.float32(0.1)
        self.assertTrue(
            any(
                "outside first-five XYZ" in error
                for error in canary._schedule_diagnostics(
                    outside, source_budget_float32=budget
                )[1]
            )
        )
        oversized = schedule.copy()
        oversized[5:, 0, 0] = np.float32(-0.3)
        oversized_errors = canary._schedule_diagnostics(
            oversized, source_budget_float32=budget
        )[1]
        self.assertTrue(any("B/5" in error for error in oversized_errors))
        self.assertTrue(any("source-bound B" in error for error in oversized_errors))

    def test_source_delta_rejects_budget_and_out_of_mask_tampering(self) -> None:
        delta = np.zeros((10, 32), dtype=np.float64)
        delta[0, 0] = 0.1

        def source(value, budget=0.1):
            return {
                "directions": {
                    "arrays": {
                        "delta_star_model": canary._array_record(
                            value, dtype=np.float64
                        )
                    },
                    "l2_norms": {"delta_star_model": budget},
                }
            }

        observed, _record, budget = canary._source_delta(source(delta))
        self.assertTrue(canary._array_exact(observed, delta))
        self.assertEqual(budget, 0.1)
        with self.assertRaisesRegex(canary.R05ACanarySourceError, "budget conflicts"):
            canary._source_delta(source(delta, budget=0.2))
        outside = delta.copy()
        outside[9, 31] = 0.01
        with self.assertRaisesRegex(canary.R05ACanarySourceError, "first-five XYZ"):
            canary._source_delta(source(outside, budget=float(np.linalg.norm(outside))))

    def test_raw_r02_file_tampering_fails_before_semantic_use(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case_dir = root / canary.CASE_ID
            case_dir.mkdir()
            (case_dir / "r02-paired.json").write_text("{}\n", encoding="utf-8")
            config = SimpleNamespace(source_r02_results_root=str(root))
            with self.assertRaisesRegex(canary.R05ACanarySourceError, "raw R02 hash mismatch"):
                canary._load_source_r02(config)

    def test_source_action_and_trace_pairing_detect_native_byte_tampering(self) -> None:
        source_actions = np.zeros((10, 7), dtype=np.float32)
        changed_actions = source_actions.copy()
        changed_actions[0, 0] = -0.0
        self.assertFalse(canary._array_exact(source_actions, changed_actions))

        source_trace = {
            "x_t": np.zeros((10, 32), dtype=np.float32),
            "time": np.asarray(0.5, dtype=np.float32),
        }
        paired = canary._trace_pairing_diagnostics(source_trace, source_trace)
        self.assertTrue(paired["exact_native_leaf_pairing"])
        changed_trace = {key: np.array(value, copy=True) for key, value in source_trace.items()}
        changed_trace["x_t"][0, 0] = np.float32(1.0)
        rejected = canary._trace_pairing_diagnostics(source_trace, changed_trace)
        self.assertFalse(rejected["exact_native_leaf_pairing"])

    def test_persisted_allocation_test_log_is_rehashed_and_reparsed(self) -> None:
        lines = []
        for suite, count in canary.ALLOCATION_TEST_COUNTS.items():
            lines.extend(
                [
                    f"verified_test_suite={suite} expected={count} observed={count} skips=0 status=passed",
                    f"[{suite}] Ran {count} tests in 0.001s",
                    f"[{suite}] OK",
                ]
            )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "allocation-focused-tests.log"
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            digest, counts = canary._parse_allocation_test_log(path)
            self.assertEqual(digest, hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(counts, dict(canary.ALLOCATION_TEST_COUNTS))
            path.write_text(path.read_text(encoding="utf-8") + "[test_r05a_canary.py] OK (skipped=1)\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "skipped or failed"):
                canary._parse_allocation_test_log(path)

    def test_gpu_samples_are_rehashed_and_reject_mixed_device_identity(self) -> None:
        uuid = "GPU-11111111-2222-3333-4444-555555555555"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gpu-memory-samples.csv"
            path.write_text(
                "timestamp_ns,gpu_uuid,compute_mib,device_mib\n"
                f"1,{uuid},100,120\n"
                f"2,{uuid},150,180\n",
                encoding="utf-8",
            )
            count, observed_uuid, compute, device, digest = canary._read_gpu_samples(path)
            self.assertEqual((count, observed_uuid, compute, device), (2, uuid, 150.0, 180.0))
            self.assertEqual(digest, hashlib.sha256(path.read_bytes()).hexdigest())
            path.write_text(
                path.read_text(encoding="utf-8").replace(
                    f"2,{uuid}", "2,GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "mixed allocation device identities"):
                canary._read_gpu_samples(path)


if __name__ == "__main__":
    unittest.main()
