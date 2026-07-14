from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "main/run_sampler_parity.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("run_sampler_parity", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load sampler parity runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SamplerParityContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runner = _load_runner()

    def test_registered_adr_limits_are_consumed_exactly(self) -> None:
        self.assertEqual(
            self.runner.REGISTERED_LIMITS,
            {
                "step_x_max": 0.10,
                "step_x_rms": 0.025,
                "step_v_max": 0.20,
                "step_v_rms": 0.050,
                "final_model_max": 0.10,
                "final_model_rms": 0.025,
                "physical_xyz5_max": 0.010,
                "physical_xyz5_rms": 0.005,
                "physical_action7_max": 0.050,
                "physical_action7_rms": 0.015,
            },
        )
        self.assertEqual(
            self.runner.REGISTERED_NORM_STATS_SHA256,
            "b3a44bb2810436fb62917decaea58bd4d9110255df527dea21e8fd40c960bd84",
        )
        decision = (
            ROOT / "docs/decisions/0010-freeze-r02-parity-and-directions.md"
        ).read_text(encoding="utf-8")
        for value in ("0.10", "0.025", "0.20", "0.050", "0.010", "0.005", "0.015"):
            self.assertIn(value, decision)

    def test_error_metrics_require_both_maximum_and_rms(self) -> None:
        passing = self.runner.comparison_metrics(
            [0.0, 0.0, 0.0, 0.0],
            [0.10, 0.0, 0.0, 0.0],
            max_limit=0.10,
            rms_limit=0.05,
            include_absolute_error=True,
        )
        self.assertTrue(passing["passed"])
        self.assertEqual(passing["absolute_error"], [0.1, 0.0, 0.0, 0.0])
        max_failure = self.runner.comparison_metrics(
            [0.0, 0.0],
            [0.11, 0.0],
            max_limit=0.10,
            rms_limit=1.0,
        )
        self.assertFalse(max_failure["passed"])
        rms_failure = self.runner.comparison_metrics(
            [0.0, 0.0],
            [0.06, 0.06],
            max_limit=0.10,
            rms_limit=0.05,
        )
        self.assertFalse(rms_failure["passed"])

    def test_within_backend_equivalence_is_array_equal(self) -> None:
        self.assertTrue(self.runner.exact_comparison([1.0], [1.0])["passed"])
        self.assertFalse(
            self.runner.exact_comparison([1.0], [1.0 + 1e-12])["passed"]
        )

    def _valid_artifact(self):
        action = [[[0.0 for _ in range(32)] for _ in range(10)]]
        physical = [[0.0 for _ in range(7)] for _ in range(10)]
        trace_value = [
            [[[0.0 for _ in range(32)] for _ in range(10)]]
            for _ in range(10)
        ]
        common_worker = {
            "schema_version": "1.0",
            "config_name": "pi05_libero",
            "asset_id": "physical-intelligence/libero",
            "num_steps": 10,
            "dt": -0.1,
            "noise_seed": 123,
            "raw_observation_sha256": "a" * 64,
            "transformed_observation_sha256": "b" * 64,
            "noise_sha256": "c" * 64,
            "noise_manifest": [
                {
                    "path": "noise",
                    "shape": [1, 10, 32],
                    "dtype": "<f4",
                }
            ],
            "output_sha256": "d" * 64,
            "default_normalized": action,
            "traced_normalized": action,
            "default_unnormalized": physical,
            "traced_unnormalized": physical,
            "trace": {
                "time": [1.0 - index / 10 for index in range(10)],
                "state_before": trace_value,
                "velocity": trace_value,
                "state_after": trace_value,
            },
            "exact_checks": {
                "instrumented_final_array_equal_ordinary_default": True,
            },
        }
        jax_worker = {**common_worker, "worker": "jax"}
        torch_worker = {
            **common_worker,
            "worker": "pytorch",
            "eager_normalized": action,
            "exact_checks": {
                **common_worker["exact_checks"],
                "compiled_default_array_equal_eager_trace_only": True,
            },
        }
        cross_x = self.runner._per_step_metrics(
            trace_value,
            trace_value,
            max_limit=0.10,
            rms_limit=0.025,
        )
        cross_v = self.runner._per_step_metrics(
            trace_value,
            trace_value,
            max_limit=0.20,
            rms_limit=0.050,
        )
        final_model = self.runner.comparison_metrics(
            action,
            action,
            max_limit=0.10,
            rms_limit=0.025,
            include_absolute_error=True,
        )
        xyz5 = self.runner.comparison_metrics(
            [row[:3] for row in physical[:5]],
            [row[:3] for row in physical[:5]],
            max_limit=0.010,
            rms_limit=0.005,
            include_absolute_error=True,
        )
        action7 = self.runner.comparison_metrics(
            physical,
            physical,
            max_limit=0.050,
            rms_limit=0.015,
            include_absolute_error=True,
        )
        exact = self.runner.exact_comparison(action, action)
        checks = {
            "real_frozen_branch_observation_shared": True,
            "transformed_observation_byte_identical": True,
            "explicit_noise_byte_identical": True,
            "checkpoint_norm_stats_byte_identical": True,
            "checkpoint_norm_stats_match_registered_asset": True,
            "jax_trace_reproduces_ordinary_final": True,
            "pytorch_trace_reproduces_ordinary_final": True,
            "pytorch_compiled_equals_eager_trace_only_final": True,
            "all_ten_cross_backend_pre_update_states_within_limits": True,
            "all_ten_cross_backend_velocities_within_limits": True,
            "cross_backend_final_normalized_within_tolerance": True,
            "cross_backend_final_physical_first_five_xyz_within_limits": True,
            "cross_backend_final_physical_first_seven_within_limits": True,
        }
        norm_hash = self.runner.REGISTERED_NORM_STATS_SHA256
        return {
            "schema_version": "1.0",
            "artifact_type": "sampler_parity",
            "gate": "R02",
            "status": "passed",
            "identity": {
                "fixture_id": self.runner.FIXTURE_ID,
                "case_id": "case-0",
                "case_index": 0,
                "noise_seed": 123,
                "config_name": "pi05_libero",
                "num_steps": 10,
                "action_horizon": 10,
                "action_dim": 32,
                "case_record_sha256": "1" * 64,
                "manifest_sha256": "2" * 64,
                "jax_params_sha256": "3" * 64,
                "pytorch_model_sha256": "4" * 64,
                "norm_stats_sha256": norm_hash,
                "runner_sha256": "5" * 64,
                "tracked_diff_sha256": "6" * 64,
                "git_commit": "commit",
                "git_dirty": True,
                "r01_summary_sha256": self.runner.R01_SUMMARY_SHA256,
            },
            "observation": {
                "fixture_id": self.runner.FIXTURE_ID,
                "raw_observation_sha256": "a" * 64,
                "transformed_observation_sha256": "b" * 64,
                "manifest_sha256": "2" * 64,
                "case_record_sha256": "1" * 64,
                "experiment_config_sha256": "7" * 64,
            },
            "acceptance": {
                "registered_tolerances": dict(self.runner.REGISTERED_LIMITS),
                "checks": checks,
                "passed": True,
            },
            "comparison": {
                "jax_trace_vs_ordinary_final": exact,
                "pytorch_trace_vs_ordinary_final": exact,
                "pytorch_compiled_vs_eager_trace_only_final": exact,
                "cross_backend_pre_update_x_t_per_step": cross_x,
                "cross_backend_pre_update_v_t_per_step": cross_v,
                "cross_backend_final_normalized": final_model,
                "cross_backend_final_physical_first_five_xyz": xyz5,
                "cross_backend_final_physical_first_seven": action7,
            },
            "checkpoints": {
                "public_jax": {
                    "params_directory": {"sha256": "3" * 64},
                    "norm_stats_sha256": norm_hash,
                },
                "converted_pytorch": {
                    "model_sha256": "4" * 64,
                    "norm_stats_sha256": norm_hash,
                },
            },
            "samplers": {"jax": jax_worker, "pytorch": torch_worker},
            "provenance": {
                "runner_sha256": "5" * 64,
                "git_commit": "commit",
                "git_dirty": True,
                "tracked_diff_sha256": "6" * 64,
                "slurm": {"job_id": "123"},
                "experiment_config_sha256": "7" * 64,
            },
        }

    def test_resume_validator_recomputes_comparisons(self) -> None:
        artifact = self._valid_artifact()
        self.assertEqual(self.runner.validate_parity_artifact(artifact), [])
        tampered = json.loads(json.dumps(artifact))
        tampered["samplers"]["pytorch"]["default_normalized"][0][0][0] = 1.0
        errors = self.runner.validate_parity_artifact(tampered)
        self.assertTrue(any("recomputed" in error for error in errors), errors)

    def test_resume_validator_rejects_hash_disagreement(self) -> None:
        artifact = self._valid_artifact()
        artifact["samplers"]["pytorch"]["noise_sha256"] = "e" * 64
        errors = self.runner.validate_parity_artifact(artifact)
        self.assertTrue(any("recomputed checks" in error for error in errors), errors)

    def test_runner_traces_ten_pre_update_states_and_velocities(self) -> None:
        source = RUNNER_PATH.read_text(encoding="utf-8")
        self.assertIn("jax.lax.scan", source)
        self.assertIn("length=NUM_STEPS", source)
        self.assertIn("for _ in range(NUM_STEPS)", source)
        self.assertIn('"state_before"', source)
        self.assertIn('"velocity"', source)
        self.assertIn("model.sample_actions(", source)
        self.assertIn("model.sample_actions_eager(", source)
        self.assertIn("nnx_utils.module_jit(model.sample_actions)", source)
        self.assertIn("exact_comparison", source)
        self.assertIn("cross_backend_pre_update_x_t_per_step", source)
        self.assertIn("cross_backend_pre_update_v_t_per_step", source)

    def test_real_branch_and_identical_transformed_tensor_path(self) -> None:
        source = RUNNER_PATH.read_text(encoding="utf-8")
        self.assertIn("SafeLiberoCase", source)
        self.assertIn("policy_observation", source)
        self.assertIn("_atomic_write_array_tree", source)
        self.assertIn("_load_array_tree", source)
        self.assertIn("applied once by public-JAX worker", source)
        self.assertIn("parity is apparatus evidence only", source)
        self.assertIn("R02 must separately reproduce all 20", source)
        first_case = json.loads(
            (ROOT / "manifests/oracle_h05_colliding.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()[0]
        )
        self.assertEqual(first_case["case_id"], "crfs-1069f29a8d76463a")
        self.assertEqual(first_case["policy_seed"], 1179198633)

    def test_runner_is_allocation_only_and_writes_atomically(self) -> None:
        source = RUNNER_PATH.read_text(encoding="utf-8")
        self.assertIn('os.environ.get("SLURM_JOB_ID")', source)
        self.assertIn('os.environ.get("CUDA_VISIBLE_DEVICES"', source)
        self.assertIn("atomic_write_json(args.output, artifact)", source)
        self.assertIn("file_sha256(pytorch_model)", source)
        self.assertIn("_directory_sha256(args.jax_checkpoint", source)
        self.assertIn("git_status_short", source)
        self.assertIn(self.runner.R01_SUMMARY_SHA256, source)

    def test_slurm_job_freezes_limits_and_stays_within_ceiling(self) -> None:
        source = (ROOT / "slurm/sampler_parity_mig.sbatch").read_text(
            encoding="utf-8"
        )
        self.assertIn("#SBATCH --gres=gpu:1", source)
        self.assertIn("#SBATCH --cpus-per-task=8", source)
        self.assertIn("#SBATCH --mem=128G", source)
        self.assertNotIn("#SBATCH --array", source)
        self.assertIn("XLA_PYTHON_CLIENT_PREALLOCATE=false", source)
        self.assertIn("MUJOCO_GL=osmesa", source)
        self.assertIn('"$OPENPI_PYTHON" "$RUNNER"', source)
        expected_arguments = {
            "--step-x-max": "0.10",
            "--step-x-rms": "0.025",
            "--step-v-max": "0.20",
            "--step-v-rms": "0.050",
            "--final-model-max": "0.10",
            "--final-model-rms": "0.025",
            "--physical-xyz5-max": "0.010",
            "--physical-xyz5-rms": "0.005",
            "--physical-action7-max": "0.050",
            "--physical-action7-rms": "0.015",
        }
        for argument, value in expected_arguments.items():
            self.assertIn(f"{argument} {value}", source)

    def test_submitter_preflights_and_only_submits_remote_compute(self) -> None:
        source = (ROOT / "scripts/hpc/submit_sampler_parity.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("scripts/hpc/preflight.sh", source)
        self.assertIn("sbatch --output=", source)
        self.assertNotIn("rsync", source)
        self.assertNotIn("git pull", source)
        self.assertNotIn("srun", source)
        self.assertNotIn("python -", source)

    def test_shell_files_parse(self) -> None:
        for relative in (
            "slurm/sampler_parity_mig.sbatch",
            "scripts/hpc/submit_sampler_parity.sh",
        ):
            subprocess.run(
                ["bash", "-n", str(ROOT / relative)],
                check=True,
                capture_output=True,
                text=True,
            )


if __name__ == "__main__":
    unittest.main()
