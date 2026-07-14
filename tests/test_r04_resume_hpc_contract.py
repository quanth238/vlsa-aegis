from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiments/r04_resume_parity.json"
RUN = ROOT / "scripts/hpc/run_r04_resume_parity.sh"
SUBMIT = ROOT / "scripts/hpc/submit_r04_resume_parity.sh"
SBATCH = ROOT / "slurm/r04_resume_parity_mig.sbatch"
EXPECTED_CONFIG_SHA256 = (
    "a0edb7edac86d2abd887218002d14e28e072cd472d631797d18d540f5140be33"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class R04ResumeHpcContractTest(unittest.TestCase):
    def test_frozen_config_is_apparatus_only_and_content_bound(self) -> None:
        self.assertEqual(_sha256(CONFIG), EXPECTED_CONFIG_SHA256)
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        settings = config["r04b"]
        self.assertIs(config["ready_to_run"], True)
        self.assertEqual(config["blocked_on"], [])
        self.assertEqual(settings["source_trace_steps"], [1, 2, 3, 4, 5])
        self.assertEqual(settings["source_eager_calls_per_step"], 2)
        self.assertEqual(settings["zero_resume_calls_per_step"], 2)
        self.assertEqual(settings["nonzero_resume_calls_per_step"], 2)
        self.assertEqual(settings["compiled_default_calls"], 2)
        self.assertEqual(
            settings["compiled_default_call_placement"],
            "one_before_and_one_after_eager_resume_sequence",
        )
        self.assertEqual(
            settings["compiled_default_request_contract"],
            "reserved_envelope_explicit_paired_noise_mode_none_no_trace_no_correction_or_resume",
        )
        self.assertEqual(
            settings["compiled_default_comparison_contract"],
            "exact_current_eager_R04A_golden_hashes_and_compiled_physical_within_frozen_R02_limits",
        )
        self.assertEqual(
            settings["compiled_default_physical_limits"],
            {
                "source": "docs/decisions/0011-use-eager-path-for-r02-parity.md",
                "source_sha256": "11646bec37bdb2d15e9507080156755300782e0a4eabf91449dcc5105ca10d36",
                "physical_xyz5_max": 0.010,
                "physical_xyz5_rms": 0.005,
                "physical_action7_max": 0.050,
                "physical_action7_rms": 0.015,
                "normalized_model_comparison_status": "not_exposed_by_ordinary_compiled_physical_policy_reply_validated_R02_predecessor_remains_authority",
            },
        )
        self.assertEqual(settings["total_policy_calls"], 32)
        self.assertEqual(settings["source_intervention_mode"], "none")
        self.assertEqual(
            settings["resume_intervention_mode"], "latent_resume_edit"
        )
        self.assertEqual(settings["latent_edit_space"], "model")
        self.assertEqual(settings["zero_edit"]["construction"], "all_zeros")
        self.assertEqual(settings["zero_edit"]["shape"], [10, 32])
        self.assertEqual(
            settings["nonzero_edit"],
            {
                "dtype": "float32",
                "construction": "single_model_coordinate",
                "shape": [10, 32],
                "index": [0, 0],
                "value": 0.03125,
                "role": "implementation_sentinel_only_never_support_or_dose_evidence",
            },
        )
        self.assertEqual(
            settings["simulator_setup"],
            "one_reset_plus_20_dummy_settle_control_steps",
        )
        self.assertEqual(settings["policy_generated_action_steps_executed"], 0)
        self.assertEqual(settings["efficacy_rollouts_executed"], 0)
        self.assertIs(settings["simulator_efficacy_evaluated"], False)
        self.assertIs(settings["training"], False)
        self.assertIs(settings["guidance"], False)
        self.assertIs(settings["learned_probe"], False)
        self.assertIs(settings["scientific_claim_authorized"], False)
        for path in (RUN, SUBMIT):
            self.assertIn(EXPECTED_CONFIG_SHA256, path.read_text(encoding="utf-8"))

    def test_allocation_runner_fails_atomically_and_validates_exact_job(self) -> None:
        source = RUN.read_text(encoding="utf-8")
        for required in (
            "SLURM_JOB_ID:?run_r04_resume_parity.sh must execute inside",
            "SLURM_ARRAY_JOB_ID:?R04B smoke must expose its exact Slurm array job id",
            "SLURM_ARRAY_TASK_ID:?R04B smoke must be a one-element Slurm array",
            "SLURM_JOB_PARTITION:?R04B smoke must record its Slurm partition",
            "CUDA_VISIBLE_DEVICES:?R04B smoke must run inside a GPU allocation",
            'test "$CASE_INDEX" = "$SLURM_ARRAY_TASK_ID"',
            "R04B is frozen to apparatus case index 0",
            "crfs-93365b8b851365f2",
            "main/run_r04_resume_parity.py",
            "r04-resume-parity.json",
            "trap cleanup EXIT",
            'kill "$SERVER_PID"',
            "REQUESTED_FAILURE_PYTHON=",
            "SYSTEM_PYTHON=",
            "no verified Python interpreter is available for atomic failure artifacts",
            '"$FAILURE_PYTHON" -',
            "launch-failure.json",
            "tempfile.NamedTemporaryFile",
            "os.replace(temporary, path)",
            "r04b_resume_parity_launch_failure",
            "one_reset_plus_20_dummy_settle_control_steps",
            "unknown_or_not_completed_due_failure",
            '"policy_generated_action_steps_executed": 0',
            '"efficacy_rollouts_executed": 0',
            'sha256sum "$MANIFEST"',
            'sha256sum "$EXPERIMENT_CONFIG"',
            'sha256sum "$R00_SUMMARY"',
            'sha256sum "$R03_SUMMARY"',
            'sha256sum "$PARITY_ARTIFACT"',
            'sha256sum "$R04A_ARTIFACT"',
            'sha256sum "$R04A_VALIDATION"',
            'sha256sum "$MODEL"',
            'test "$GIT_COMMIT" = "$EXPECTED_GIT_COMMIT"',
            'test "$GIT_DIRTY" = false',
            "FAILURE_STAGE=exact_job_artifact_validation",
            "FAILURE_STAGE=dependency_backed_sampler_policy_contract",
            "test_r04b_sampler_policy_contract.py",
            "--validate-artifact",
            "--expected-source-slurm-job-id",
            "--expected-source-slurm-array-job-id",
            "--expected-source-slurm-array-task-id",
            "from crfs_oracle.r04_resume import validate_r04_resume_result",
            '"slurm_job_id": os.environ["SLURM_JOB_ID"]',
            '"slurm_array_job_id": os.environ["SLURM_ARRAY_JOB_ID"]',
            '"slurm_array_task_id": os.environ["SLURM_ARRAY_TASK_ID"]',
            '"partition": os.environ["SLURM_JOB_PARTITION"]',
            '"device": os.environ["CUDA_VISIBLE_DEVICES"]',
            "provenance.get(key) != expected_value",
            "validation_error_count=0",
        ):
            self.assertIn(required, source)
        for forbidden in (
            "env.step",
            "--train",
            "--correction",
            "--simulator-repeats",
            "rm -rf",
            "scancel",
            "pkill",
            "rsync --delete",
        ):
            self.assertNotIn(forbidden, source)

    def test_submitter_guards_commit_hashes_capacity_and_live_nodes(self) -> None:
        source = SUBMIT.read_text(encoding="utf-8")
        for required in (
            "scripts/hpc/preflight.sh",
            "RUN_ID:?",
            "RUN_ID must contain only letters, digits, dot, underscore, or hyphen",
            "R04B source files must be committed before submission",
            "R04B submission requires a clean reviewed tracked worktree",
            "remote source is not synchronized to the reviewed R04B commit",
            "remote R04B source tree is dirty",
            "EXPECTED_R00_SUMMARY_SHA256=",
            "EXPECTED_R03_SUMMARY_SHA256=",
            "EXPECTED_PARITY_SHA256=",
            "EXPECTED_R04A_ARTIFACT_SHA256=",
            "EXPECTED_R04A_VALIDATION_SHA256=",
            "EXPECTED_CHECKPOINT_SHA256=",
            "squeue -h -u",
            "AllocTRES=",
            "projected_gpus=$((allocated_gpus + 1))",
            "projected_cpus=$((allocated_cpus + 6))",
            "projected_mem_mb=$((allocated_mem_mb + 80 * 1024))",
            "exceed the 2-GPU-equivalent user ceiling",
            "exceed the 16-CPU user ceiling",
            "exceed the 256-GiB user ceiling",
            "sinfo -h -p mig -N",
            "scontrol show node",
            "FreeMem=",
            "minimum_free_mem_mb=$((80 * 1024))",
            "*down*|*drain*|*drng*|*fail*|*maint*|*not_resp*",
            "no healthy MIG node has a fresh 80 GiB",
            "--nodelist=",
            "--exclude=",
            "sbatch --parsable",
            "sbatch did not return an exact numeric R04B job id",
            "submitted_exact_job_id=",
            "expected_array_task=",
            "standalone_validator_source_slurm_job_id=",
            "standalone_validator_source_slurm_array_job_id=",
            "standalone_validator_source_slurm_array_task_id=0",
            "r04-resume-parity.json",
            "slurm/r04_resume_parity_mig.sbatch",
        ):
            self.assertIn(required, source)
        self.assertLess(source.index("scripts/hpc/preflight.sh"), source.index("ssh \"$HOST\""))
        remote = source.split("<<'REMOTE'", 1)[1]
        self.assertNotIn("python3", remote)
        for forbidden in ("rm -rf", "scancel", "pkill", "rsync", "--delete"):
            self.assertNotIn(forbidden, source)

    def test_standalone_validator_requires_and_receives_all_source_ids(self) -> None:
        runner = RUN.read_text(encoding="utf-8")
        cli = (ROOT / "main/run_r04_resume_parity.py").read_text(encoding="utf-8")
        options = (
            "--expected-source-slurm-job-id",
            "--expected-source-slurm-array-job-id",
            "--expected-source-slurm-array-task-id",
        )
        for option in options:
            self.assertIn(option, runner)
            self.assertGreaterEqual(cli.count(option), 2)
        self.assertIn(
            "R04B validation mode requires all source Slurm identity arguments",
            cli,
        )

    def test_slurm_shape_is_one_element_mig_smoke(self) -> None:
        source = SBATCH.read_text(encoding="utf-8")
        for required in (
            "#SBATCH --partition=mig",
            "#SBATCH --gres=gpu:1",
            "#SBATCH --cpus-per-task=6",
            "#SBATCH --mem=80G",
            "#SBATCH --time=01:00:00",
            "#SBATCH --array=0-0%1",
            "scripts/hpc/run_r04_resume_parity.sh",
        ):
            self.assertIn(required, source)

    def test_shell_files_are_executable_and_parse(self) -> None:
        for path in (RUN, SUBMIT, SBATCH):
            self.assertTrue(os.access(path, os.X_OK), f"not executable: {path}")
            result = subprocess.run(
                ["bash", "-n", str(path)],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_prelaunch_failure_is_written_atomically_with_exact_job_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            remote = root / "repository"
            checkpoint = root / "checkpoint"
            inputs = (
                remote / "main/run_r04_resume_parity.py",
                remote / "evidence/r00/r00-summary.json",
                remote / "evidence/r03/r03-summary.json",
                root / "manifest.jsonl",
                root / "config.json",
                root / "parity.json",
                root / "r04a.json",
                root / "r04a-validation.json",
                checkpoint / "model.safetensors",
            )
            for path in inputs:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}\n", encoding="utf-8")
            environment = os.environ.copy()
            environment.update(
                {
                    "SLURM_JOB_ID": "99001",
                    "SLURM_ARRAY_JOB_ID": "99001",
                    "SLURM_ARRAY_TASK_ID": "0",
                    "SLURM_JOB_PARTITION": "mig",
                    "CUDA_VISIBLE_DEVICES": "MIG-test-device",
                    "REMOTE_REPO": str(remote),
                    "OPENPI_PYTHON": sys.executable,
                    "LIBERO_PYTHON": str(root / "missing-libero-python"),
                    "CHECKPOINT_DIR": str(checkpoint),
                    "EXPERIMENT_ROOT": str(root / "experiments"),
                    "RUN_ID": "r04b-atomic-failure-test",
                    "MANIFEST": str(root / "manifest.jsonl"),
                    "EXPERIMENT_CONFIG": str(root / "config.json"),
                    "PARITY_ARTIFACT": str(root / "parity.json"),
                    "R04A_ARTIFACT": str(root / "r04a.json"),
                    "R04A_VALIDATION": str(root / "r04a-validation.json"),
                    "EXPECTED_GIT_COMMIT": "0" * 40,
                }
            )
            result = subprocess.run(
                [str(RUN)],
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("missing LIBERO Python", result.stderr)
            failure = (
                root
                / "experiments/r04b-atomic-failure-test/failures/case-index-0.json"
            )
            self.assertTrue(failure.is_file())
            value = json.loads(failure.read_text(encoding="utf-8"))
            self.assertEqual(value["status"], "failed")
            self.assertEqual(value["stage"], "source_binding")
            self.assertEqual(value["artifact_role"], "r04b_resume_parity_launch_failure")
            self.assertIs(value["scientific_claim_allowed"], False)
            self.assertIs(value["probe_training_authorized"], False)
            self.assertEqual(
                value["planned_simulator_setup"],
                "one_reset_plus_20_dummy_settle_control_steps",
            )
            self.assertEqual(
                value["simulator_setup_completion"],
                "unknown_or_not_completed_due_failure",
            )
            self.assertNotIn("simulator_setup", value)
            self.assertEqual(value["policy_generated_action_steps_executed"], 0)
            self.assertEqual(value["efficacy_rollouts_executed"], 0)
            self.assertIs(value["simulator_efficacy_evaluated"], False)
            self.assertEqual(
                Path(value["failure_writer_python"]).resolve(),
                Path(sys.executable).resolve(),
            )
            self.assertEqual(
                value["provenance"],
                {
                    "slurm_job_id": "99001",
                    "slurm_array_job_id": "99001",
                    "slurm_array_task_id": "0",
                    "partition": "mig",
                    "device": "MIG-test-device",
                },
            )

    def test_submitter_rejects_nonimmutable_run_id_before_preflight(self) -> None:
        environment = os.environ.copy()
        environment["RUN_ID"] = "../mutable/run"
        result = subprocess.run(
            [str(SUBMIT), "unused.jsonl", "unused.json"],
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("RUN_ID must contain only", result.stderr)
        self.assertNotIn("preflight saved", result.stdout)


if __name__ == "__main__":
    unittest.main()
