from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
from types import SimpleNamespace
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
CANARY_PATH = ROOT / "main" / "crfs_oracle" / "r05a_canary.py"
RUNNER_PATH = ROOT / "scripts" / "hpc" / "run_r05a_canary.sh"
SUBMIT_PATH = ROOT / "scripts" / "hpc" / "submit_r05a_canary.sh"
GPU_SLURM_PATH = ROOT / "slurm" / "r05a_canary_h100.sbatch"
ALLOCATION_TEST_REGISTRY_PATH = ROOT / "main" / "crfs_oracle" / "r05a_allocation_tests.json"
ALLOCATION_TEST_HELPER_PATH = ROOT / "scripts" / "hpc" / "lib" / "r05a_allocation_tests.sh"


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

        # One ordered data file owns every expected count. The shell executes
        # that file, Python finalization loads it, and discovery independently
        # proves that its reviewed counts still describe the checked-in tests.
        registry = json.loads(ALLOCATION_TEST_REGISTRY_PATH.read_text(encoding="utf-8"))
        self.assertEqual(set(registry), {"schema_version", "suites"})
        self.assertEqual(registry["schema_version"], "1.0")
        self.assertIsInstance(registry["suites"], list)
        self.assertGreater(len(registry["suites"]), 0)
        registered_patterns = [item["pattern"] for item in registry["suites"]]
        self.assertEqual(
            registered_patterns,
            [
                "test_inverse_flow_control.py",
                "test_inverse_flow_sampler.py",
                "test_inverse_flow_policy.py",
                "test_r05a_canary.py",
            ],
        )
        self.assertEqual(len(registered_patterns), len(set(registered_patterns)))
        runner = RUNNER_PATH.read_text(encoding="utf-8")
        allocation_test_helper = ALLOCATION_TEST_HELPER_PATH.read_text(encoding="utf-8")
        canary_source = CANARY_PATH.read_text(encoding="utf-8")
        self.assertIn("ALLOCATION_TEST_REGISTRY=$REMOTE_REPO/main/crfs_oracle/", runner)
        self.assertIn("r05a_allocation_tests.json", runner)
        self.assertIn("crfs_run_r05a_allocation_tests", runner)
        self.assertIn('jq -e "$registry_contract"', allocation_test_helper)
        self.assertIn('--host-cgroup-diagnostic "$HOST_CGROUP_DIAGNOSTIC"', runner)
        self.assertIn(
            "ALLOCATION_TEST_COUNTS = _load_allocation_test_counts(", canary_source
        )
        for item in registry["suites"]:
            filename = item["pattern"]
            expected = item["expected_tests"]
            self.assertIs(type(expected), int)
            self.assertGreater(expected, 0)
            discovered = unittest.TestLoader().discover(
                str(ROOT / "tests"), pattern=filename
            ).countTestCases()
            self.assertEqual(discovered, expected, filename)
            self.assertNotIn(f"{filename}:{expected}", runner)
            self.assertNotIn(f"{filename}:{expected}", allocation_test_helper)
        submit_source = SUBMIT_PATH.read_text(encoding="utf-8")
        self.assertIn("ALLOCATION_TEST_REGISTRY_LOCAL", submit_source)
        self.assertIn("scripts/hpc/lib/cgroup_memory.sh", submit_source)

    def test_submission_allows_zero_free_gpu_and_records_truthful_pending_preflight(self) -> None:
        source = SUBMIT_PATH.read_text(encoding="utf-8")
        self.assertIn('test "$allocated_gpus" -le "$configured_gpus"', source)
        self.assertNotIn('test "$free_gpus" -ge 1', source)
        self.assertIn('status: "preflight_verified_before_queueing"', source)
        self.assertIn(
            "immediate_gpu_capacity_available: (($free_gpus | tonumber) >= 1)",
            source,
        )
        self.assertIn("pending_submission_allowed: true", source)

    def test_zero_free_gpu_fake_slurm_transaction_is_queued_after_receipting(self) -> None:
        source = SUBMIT_PATH.read_text(encoding="utf-8")
        remote_start = source.index("<<'REMOTE'\n") + len("<<'REMOTE'\n")
        remote_end = source.rindex("\nREMOTE")
        remote_script = source[remote_start:remote_end]
        fixed_hash = "055fcf18781071c6c3575b42a1b44c32a76b474ac1911ad8c5443f78b9c42593"

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            remote_repo = root / "repo"
            experiment_root = root / "experiments"
            r02_root = root / "r02"
            checkpoint_dir = root / "checkpoint"
            fake_bin = root / "bin"
            call_log = root / "calls.log"
            for path in (
                remote_repo / "schemas" / "r05a-inverse-flow-canary.schema.json",
                remote_repo / "docs" / "decisions" / "0028-pivot-to-inverse-flow-transport.md",
                remote_repo / "evidence" / "r03" / "r03-summary.json",
                remote_repo / "slurm" / "r05a_canary_h100.sbatch",
                remote_repo / "slurm" / "r05a_canary_validate_cpu.sbatch",
                remote_repo / "main" / "crfs_oracle" / "r05a_allocation_tests.json",
                remote_repo / "scripts" / "hpc" / "lib" / "cgroup_memory.sh",
                remote_repo / "scripts" / "hpc" / "lib" / "r05a_allocation_tests.sh",
                remote_repo / "scripts" / "hpc" / "run_r05a_canary.sh",
                remote_repo / "scripts" / "hpc" / "validate_r05a_canary.sh",
                remote_repo / "manifests" / "r05a_inverse_flow_teacher_smoke.jsonl",
                remote_repo / "configs" / "experiments" / "r05a_inverse_flow_canary.json",
                r02_root / "crfs-1069f29a8d76463a" / "r02-paired.json",
                checkpoint_dir / "model.safetensors",
            ):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("fixture\n", encoding="utf-8")
            experiment_root.mkdir()
            fake_bin.mkdir()

            def executable(name: str, body: str) -> None:
                path = fake_bin / name
                path.write_text("#!/usr/bin/env bash\nset -eu\n" + body, encoding="utf-8")
                path.chmod(path.stat().st_mode | stat.S_IXUSR)

            executable("squeue", "exit 0\n")
            executable(
                "git",
                'case " $* " in\n'
                '  *" rev-parse HEAD "*) printf "%s\\n" "$FAKE_COMMIT" ;;\n'
                '  *" status --porcelain "*) exit 0 ;;\n'
                '  *) exit 2 ;;\n'
                "esac\n",
            )
            executable(
                "sha256sum",
                'printf "%s  %s\\n" "$FAKE_HASH" "$1"\n',
            )
            executable(
                "sbatch",
                'printf "sbatch %s\\n" "$*" >>"$FAKE_CALL_LOG"\n'
                'case "$*" in\n'
                '  *r05a_canary_h100.sbatch) printf "9101\\n" ;;\n'
                '  *r05a_canary_validate_cpu.sbatch) printf "9102\\n" ;;\n'
                '  *) exit 2 ;;\n'
                "esac\n",
            )
            executable(
                "scontrol",
                'case "$1 $2" in\n'
                '  "show node") printf "%s\\n" "NodeName=worker-1 Gres=gpu:nvidia_h100_80gb_hbm3:8(S:0-1) FreeMem=200000 State=MIXED CfgTRES=cpu=128,mem=2000000M,gres/gpu=8 AllocTRES=cpu=64,mem=512G,gres/gpu=8" ;;\n'
                '  "show job")\n'
                '    case "$3" in\n'
                '      9101) printf "%s\\n" "JobId=9101 JobState=PENDING Reason=JobHeldUser ReqNodeList=worker-1" ;;\n'
                '      9102) printf "%s\\n" "JobId=9102 JobState=PENDING Partition=main ReqTRES=cpu=2,mem=8G Dependency=afterany:9101(unfulfilled)" ;;\n'
                '      *) exit 2 ;;\n'
                '    esac ;;\n'
                '  "release 9101") printf "release 9101\\n" >>"$FAKE_CALL_LOG" ;;\n'
                '  *) exit 2 ;;\n'
                "esac\n",
            )

            run_id = "r05a-zero-free-gpu-test"
            commit = "a" * 40
            manifest = remote_repo / "manifests" / "r05a_inverse_flow_teacher_smoke.jsonl"
            config = remote_repo / "configs" / "experiments" / "r05a_inverse_flow_canary.json"
            command = [
                "bash",
                "-s",
                "--",
                str(remote_repo),
                run_id,
                str(manifest),
                str(config),
                commit,
                str(experiment_root),
                str(r02_root),
                str(checkpoint_dir),
                fixed_hash,
                fixed_hash,
                fixed_hash,
                fixed_hash,
                fixed_hash,
                fixed_hash,
            ]
            environment = os.environ.copy()
            environment.update(
                {
                    "PATH": f"{fake_bin}:{environment['PATH']}",
                    "FAKE_CALL_LOG": str(call_log),
                    "FAKE_COMMIT": commit,
                    "FAKE_HASH": fixed_hash,
                }
            )
            completed = subprocess.run(
                command,
                input=remote_script,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=environment,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)

            run_root = experiment_root / run_id
            reservation = json.loads(
                (run_root / "launch-reservation.json").read_text(encoding="utf-8")
            )
            self.assertEqual(reservation["status"], "preflight_verified_before_queueing")
            self.assertEqual(reservation["observed_free_gpus"], 0)
            self.assertIs(reservation["immediate_gpu_capacity_available"], False)
            self.assertIs(reservation["pending_submission_allowed"], True)
            submission = json.loads(
                (run_root / "submission.json").read_text(encoding="utf-8")
            )
            self.assertEqual(submission["gpu_slurm_array_job_id"], "9101")
            self.assertEqual(submission["cpu_afterany_job_id"], "9102")
            calls = call_log.read_text(encoding="utf-8").splitlines()
            self.assertEqual(calls[-1], "release 9101")

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
            "_validate_compiled_eager_path_seam_record",
            "and recomputed_compiled_eager_path_seam",
            "_array_exact(eager_before_actions, source_before_actions)",
            "_array_exact(eager_after_actions, source_after_actions)",
            "exact current/source pairing checks are incomplete or failed",
            "current policy noise differs from immutable R02 seed/noise",
            "recorded teacher determinism differs from independent trace recomputation",
            "recorded zero/frozen checks differ from independent trace recomputation",
            "recorded nonconvergence fail-closed checks differ from raw traces",
            "apparatus summary differs from independent trace recomputation",
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

    def test_exact_source_pairing_and_numeric_compiled_eager_seam(self) -> None:
        singleton_trace_record = canary._trace_record(
            {
                "solver_status": np.asarray([3], dtype=np.int64),
                "cuda_memory_available": np.asarray([True], dtype=np.bool_),
                "cuda_process_peak_allocated_bytes": np.asarray(
                    [123], dtype=np.int64
                ),
                "cuda_process_peak_reserved_bytes": np.asarray(
                    [456], dtype=np.int64
                ),
            }
        )
        singleton_errors = []
        singleton_trace = canary._artifact_trace(
            singleton_trace_record,
            name="serialized singleton fixture",
            errors=singleton_errors,
        )
        self.assertEqual(singleton_errors, [])
        self.assertIsNotNone(singleton_trace)
        self.assertEqual(
            canary._artifact_scalar(
                singleton_trace["solver_status"], name="solver_status"
            ),
            3,
        )
        self.assertTrue(
            canary._artifact_scalar(
                singleton_trace["cuda_memory_available"],
                name="cuda_memory_available",
            )
        )
        self.assertEqual(
            canary._artifact_cuda_memory_peaks(singleton_trace, singleton_trace),
            (123, 456),
        )
        with self.assertRaisesRegex(ValueError, "serialized singleton scalar"):
            canary._artifact_scalar(np.asarray([1, 2]), name="nonscalar")
        with self.assertRaisesRegex(ValueError, "must be scalar"):
            canary._scalar(np.asarray([3]), name="live trace scalar")

        nonconverged_invariants = {
            key: key not in {"control_valid", "schedule_applied", "fidelity_passed"}
            for key in canary.TEACHER_INVARIANT_KEYS
        }
        self.assertTrue(
            canary._teacher_invariant_truth(
                nonconverged_invariants, converged=False
            )
        )
        self.assertTrue(
            canary._teacher_invariant_truth(
                {key: True for key in canary.TEACHER_INVARIANT_KEYS},
                converged=True,
            )
        )
        self.assertTrue(
            canary._artifact_teacher_invariant_summary_passed(
                {"invariants": nonconverged_invariants},
                nonconverged_invariants,
                converged=False,
            )
        )
        for key in ("dt_exact", "intervention_step_exact", "num_steps_exact"):
            with self.subTest(teacher_invariant=key):
                tampered_invariants = dict(nonconverged_invariants)
                tampered_invariants[key] = False
                self.assertFalse(
                    canary._teacher_invariant_truth(
                        tampered_invariants, converged=False
                    )
                )

        # Coupled tampering is still rejected: changing a raw timing leaf and
        # changing the stored invariant to agree cannot satisfy the registered
        # nonconverged truth pattern.
        coupled_teacher_trace = {
            "dt": np.asarray([-0.2], dtype=np.float32),
            "intervention_step": np.asarray([5], dtype=np.int64),
            "num_steps": np.asarray([10], dtype=np.int64),
        }
        coupled_teacher_invariants = dict(nonconverged_invariants)
        coupled_teacher_invariants.update(
            canary._artifact_timing_checks(coupled_teacher_trace)
        )
        coupled_teacher_summary = {"invariants": coupled_teacher_invariants}
        self.assertEqual(
            coupled_teacher_summary["invariants"], coupled_teacher_invariants
        )
        self.assertFalse(
            canary._artifact_teacher_invariant_summary_passed(
                coupled_teacher_summary,
                coupled_teacher_invariants,
                converged=False,
            )
        )

        zero_trace = self._zero_trace()
        zero_trace.update(
            {
                "control_source": np.asarray([1], dtype=np.int64),
                "control_valid": np.asarray([True], dtype=np.bool_),
                "schedule_applied": np.asarray([True], dtype=np.bool_),
                "final_normalized_physical": np.zeros((10, 7), dtype=np.float32),
                "dt": np.asarray([-0.1], dtype=np.float32),
                "intervention_step": np.asarray([5], dtype=np.int64),
                "num_steps": np.asarray([10], dtype=np.int64),
                "schedule_budget": np.asarray([0.0], dtype=np.float32),
            }
        )
        zero_checks = canary._artifact_zero_replay_checks(
            zero_trace,
            np.zeros((10, 7), dtype=np.float32),
            np.zeros((10, 32), dtype=np.float32),
        )
        self.assertTrue(all(zero_checks.values()))
        for key, value, expected_check in (
            ("dt", -0.2, "dt_exact"),
            ("intervention_step", 4, "intervention_step_exact"),
            ("num_steps", 9, "num_steps_exact"),
        ):
            with self.subTest(zero_trace=key):
                tampered_trace = {
                    name: np.array(item, copy=True)
                    for name, item in zero_trace.items()
                }
                tampered_trace[key][0] = value
                tampered_checks = canary._artifact_zero_replay_checks(
                    tampered_trace,
                    np.zeros((10, 7), dtype=np.float32),
                    np.zeros((10, 32), dtype=np.float32),
                )
                self.assertFalse(tampered_checks[expected_check])
                self.assertFalse(all(tampered_checks.values()))

        coupled_replay_trace = {
            name: np.array(item, copy=True) for name, item in zero_trace.items()
        }
        coupled_replay_trace["dt"][0] = np.float32(-0.2)
        coupled_replay_checks = canary._artifact_zero_replay_checks(
            coupled_replay_trace,
            np.zeros((10, 7), dtype=np.float32),
            np.zeros((10, 32), dtype=np.float32),
        )
        coupled_replay_summary = {
            "checks": coupled_replay_checks,
            "passed": all(coupled_replay_checks.values()),
        }
        self.assertTrue(
            canary._artifact_replay_summary_exact(
                coupled_replay_summary, coupled_replay_checks
            )
        )
        self.assertFalse(
            canary._artifact_replay_summary_passed(
                coupled_replay_summary, coupled_replay_checks
            )
        )

        canonical_checks = canary._artifact_canonical_replay_checks(
            zero_trace,
            np.zeros((10, 7), dtype=np.float32),
            np.zeros((10, 32), dtype=np.float32),
            np.zeros((10, 10, 32), dtype=np.float32),
            np.float32(0.0),
            np.zeros((10, 7), dtype=np.float32),
            np.zeros((10, 32), dtype=np.float32),
            np.zeros((10, 32), dtype=np.float32),
        )
        self.assertTrue(all(canonical_checks.values()))
        for expected_check, teacher_actions, teacher_final, teacher_internal in (
            (
                "teacher_actions_exact",
                np.full((10, 7), np.float32(1.0)),
                np.zeros((10, 32), dtype=np.float32),
                np.zeros((10, 32), dtype=np.float32),
            ),
            (
                "teacher_final_exact",
                np.zeros((10, 7), dtype=np.float32),
                np.full((10, 32), np.float32(1.0)),
                np.zeros((10, 32), dtype=np.float32),
            ),
            (
                "teacher_internal_final_exact",
                np.zeros((10, 7), dtype=np.float32),
                np.zeros((10, 32), dtype=np.float32),
                np.full((10, 32), np.float32(1.0)),
            ),
        ):
            with self.subTest(canonical_teacher_binding=expected_check):
                tampered_canonical_checks = canary._artifact_canonical_replay_checks(
                    zero_trace,
                    np.zeros((10, 7), dtype=np.float32),
                    np.zeros((10, 32), dtype=np.float32),
                    np.zeros((10, 10, 32), dtype=np.float32),
                    np.float32(0.0),
                    teacher_actions,
                    teacher_final,
                    teacher_internal,
                )
                self.assertFalse(tampered_canonical_checks[expected_check])

        converged_top = {
            "applicable": True,
            "actions": {"fixture": "actions"},
            "final_normalized": {"fixture": "final"},
            "trace": {"fixture": "trace"},
            "schedule_diagnostics": {"fixture": "diagnostics"},
            "recurrence_errors": [],
            "schedule_errors": [],
            "passed": True,
            "checks": {"recurrence_exact": True},
            "elapsed_seconds": 0.0,
        }
        converged_call = canary._canonical_policy_call_record(converged_top)
        self.assertEqual(
            set(converged_call), set(converged_top) - {"applicable"}
        )
        self.assertTrue(
            canary._artifact_canonical_record_pair_exact(
                converged_top, converged_call
            )
        )
        converged_call_with_top_only_key = dict(converged_call)
        converged_call_with_top_only_key["applicable"] = True
        self.assertFalse(
            canary._artifact_canonical_record_pair_exact(
                converged_top, converged_call_with_top_only_key
            )
        )
        nonconverged = dict(canary.NONCONVERGED_CANONICAL_REPLAY)
        self.assertEqual(
            canary._canonical_policy_call_record(nonconverged), nonconverged
        )
        self.assertEqual(
            nonconverged,
            {
                "applicable": False,
                "passed": False,
                "reason": "finite_teacher_search_did_not_converge",
            },
        )
        self.assertTrue(
            canary._artifact_canonical_record_pair_exact(
                nonconverged, nonconverged
            )
        )
        for key, value in (
            ("applicable", True),
            ("passed", True),
            ("reason", "finite_search_failed"),
        ):
            for location in ("top", "policy_call"):
                with self.subTest(
                    nonconverged_sentinel=key, sentinel_location=location
                ):
                    top = dict(nonconverged)
                    call = dict(nonconverged)
                    (top if location == "top" else call)[key] = value
                    self.assertFalse(
                        canary._artifact_canonical_record_pair_exact(top, call)
                    )

        self.assertEqual(
            canary._artifact_result_status(
                recomputed_converged=True,
                recomputed_clean_nonconvergence=False,
            ),
            "completed_converged",
        )
        self.assertEqual(
            canary._artifact_result_status(
                recomputed_converged=False,
                recomputed_clean_nonconvergence=True,
            ),
            "completed_nonconverged",
        )
        self.assertEqual(
            canary._artifact_result_status(
                recomputed_converged=False,
                recomputed_clean_nonconvergence=False,
            ),
            "completed_apparatus_failure",
        )

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

        self.assertEqual(
            canary.ADR0011_COMPILED_EAGER_PATH_LIMITS,
            {
                "first_five_xyz_max_abs": 0.010,
                "first_five_xyz_rms": 0.005,
                "full_10x7_max_abs": 0.050,
                "full_10x7_rms": 0.015,
            },
        )
        eager = np.zeros((10, 7), dtype=np.float64)
        compiled = eager.copy()
        compiled[0, 0] = 0.009
        seam = canary._compiled_eager_path_seam_record(
            compiled, eager, compiled, eager
        )
        self.assertFalse(seam["before"]["array_equal_diagnostic"])
        self.assertTrue(seam["passed"])

        errors = []
        self.assertTrue(
            canary._validate_compiled_eager_path_seam_record(
                seam,
                compiled_before=compiled,
                eager_before=eager,
                compiled_after=compiled,
                eager_after=eager,
                errors=errors,
            )
        )
        self.assertEqual(errors, [])
        tampered = json.loads(json.dumps(seam))
        tampered["before"]["physical_first_five_xyz"]["max_abs"] = 0.0
        tamper_errors = []
        self.assertTrue(
            canary._validate_compiled_eager_path_seam_record(
                tampered,
                compiled_before=compiled,
                eager_before=eager,
                compiled_after=compiled,
                eager_after=eager,
                errors=tamper_errors,
            )
        )
        self.assertTrue(any("independent action recomputation" in item for item in tamper_errors))

        xyz_max_failure = eager.copy()
        xyz_max_failure[0, 0] = 0.011
        xyz_max = canary._compiled_eager_path_diagnostics(xyz_max_failure, eager)
        self.assertFalse(xyz_max["physical_first_five_xyz"]["passed"])
        self.assertTrue(xyz_max["physical_full_10x7"]["passed"])

        xyz_rms_failure = eager.copy()
        xyz_rms_failure[:5, :3] = 0.006
        xyz_rms = canary._compiled_eager_path_diagnostics(xyz_rms_failure, eager)
        self.assertLess(xyz_rms["physical_first_five_xyz"]["max_abs"], 0.010)
        self.assertFalse(xyz_rms["physical_first_five_xyz"]["passed"])

        full_max_failure = eager.copy()
        full_max_failure[9, 6] = 0.051
        full_max = canary._compiled_eager_path_diagnostics(full_max_failure, eager)
        self.assertTrue(full_max["physical_first_five_xyz"]["passed"])
        self.assertFalse(full_max["physical_full_10x7"]["passed"])

        full_rms_failure = eager.copy()
        full_rms_failure[:, 3:] = 0.020
        full_rms = canary._compiled_eager_path_diagnostics(full_rms_failure, eager)
        self.assertLess(full_rms["physical_full_10x7"]["max_abs"], 0.050)
        self.assertFalse(full_rms["physical_full_10x7"]["passed"])

    def test_persisted_allocation_test_log_is_rehashed_and_reparsed(self) -> None:
        registry = json.loads(ALLOCATION_TEST_REGISTRY_PATH.read_text(encoding="utf-8"))
        registered_counts = {
            item["pattern"]: item["expected_tests"] for item in registry["suites"]
        }
        self.assertEqual(dict(canary.ALLOCATION_TEST_COUNTS), registered_counts)
        self.assertEqual(
            canary.ALLOCATION_TEST_REGISTRY_SHA256,
            hashlib.sha256(ALLOCATION_TEST_REGISTRY_PATH.read_bytes()).hexdigest(),
        )
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

            registry_path = Path(directory) / "registry.json"
            invalid_registries = (
                {**registry, "unknown": True},
                {
                    "schema_version": "1.0",
                    "suites": [registry["suites"][0], registry["suites"][0]],
                },
                {
                    "schema_version": "1.0",
                    "suites": [{"pattern": "../test_escape.py", "expected_tests": 1}],
                },
                {
                    "schema_version": "1.0",
                    "suites": [{"pattern": "test_bad.py", "expected_tests": True}],
                },
            )
            for invalid in invalid_registries:
                registry_path.write_text(json.dumps(invalid), encoding="utf-8")
                with self.assertRaises(RuntimeError):
                    canary._load_allocation_test_counts(registry_path)

            diagnostic_path = Path(directory) / "host-cgroup-memory.tsv"
            diagnostic = {
                "schema_version": "1.0",
                "artifact_role": "r05a_live_slurm_cgroup_memory_peak_diagnostic",
                "status": "measured",
                "reason": "live_positive_peak",
                "cgroup_version": "2",
                "membership_path": "/slurm/uid_1073/job_27726/step_batch",
                "mount_root": "/",
                "mount_point": "/sys/fs/cgroup",
                "membership_relative_to_mount_root": "/slurm/uid_1073/job_27726/step_batch",
                "peak_file": "/sys/fs/cgroup/slurm/uid_1073/job_27726/step_batch/memory.peak",
                "peak_bytes": "123456789",
                "proc_cgroup_file": "/proc/self/cgroup",
                "mountinfo_file": "/proc/self/mountinfo",
            }
            diagnostic_path.write_text(
                "".join(
                    f"{key}\t{diagnostic[key]}\n"
                    for key in canary.CGROUP_MEMORY_DIAGNOSTIC_KEYS
                ),
                encoding="utf-8",
            )
            diagnostic_sha, parsed_diagnostic = canary._parse_cgroup_memory_diagnostic(
                diagnostic_path
            )
            self.assertEqual(
                diagnostic_sha, hashlib.sha256(diagnostic_path.read_bytes()).hexdigest()
            )
            self.assertEqual(parsed_diagnostic["peak_bytes"], 123456789)
            diagnostic_path.write_text(
                diagnostic_path.read_text(encoding="utf-8").replace(
                    "/memory.peak", "/wrong.peak"
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "peak path is inconsistent"):
                canary._parse_cgroup_memory_diagnostic(diagnostic_path)

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
