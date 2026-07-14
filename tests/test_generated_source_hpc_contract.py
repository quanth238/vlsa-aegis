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
CONFIG = ROOT / "configs/experiments/task0_single_obstacle_generated_v1.json"
GENERATOR = ROOT / "main/generate_task0_single_obstacle_source.py"
VALIDATOR = ROOT / "main/validate_task0_single_obstacle_source.py"
RUN = ROOT / "scripts/hpc/run_generated_source_pilot.sh"
VALIDATE_RUN = ROOT / "scripts/hpc/run_generated_source_validation.sh"
SUBMIT = ROOT / "scripts/hpc/submit_generated_source_pilot.sh"
SBATCH = ROOT / "slurm/generated_source_pilot_main.sbatch"
VALIDATE_SBATCH = ROOT / "slurm/generated_source_validate_cpu.sbatch"
EXPECTED_CONFIG_SHA256 = "332c90fdb9e4560ab522e6333fbe5f0846d1c7caca9190f1730436036856f22a"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class GeneratedSourceHpcContractTest(unittest.TestCase):
    def test_frozen_config_and_source_canary_are_content_bound(self) -> None:
        value = json.loads(CONFIG.read_text(encoding="utf-8"))
        self.assertIs(value["ready_to_run"], True)
        self.assertEqual(value["blocked_on"], [])
        self.assertIs(value["pilot_only"], True)
        self.assertIs(value["ready_for_training"], False)
        self.assertIs(value["ready_for_claims"], False)
        self.assertEqual(_sha256(CONFIG), EXPECTED_CONFIG_SHA256)
        groups = value["source_groups"]
        self.assertEqual(len(groups), 10)
        self.assertEqual(
            [group["generation_request_id"] for group in groups],
            [
                f"greq-task0-single-obstacle-v1-{index:04d}"
                for index in range(10)
            ],
        )
        for path in (RUN, SUBMIT):
            self.assertIn(EXPECTED_CONFIG_SHA256, path.read_text(encoding="utf-8"))
        self.assertIn(
            "retired source-integrity pilot is frozen to group indices 0 through 9",
            RUN.read_text(encoding="utf-8"),
        )

    def test_allocation_runner_is_render_only_and_validates_exact_job(self) -> None:
        source = RUN.read_text(encoding="utf-8")
        for required in (
            "SLURM_JOB_ID:?run_generated_source_pilot.sh must execute inside",
            "SLURM_ARRAY_JOB_ID:?generated-source pilot must expose its exact Slurm array job id",
            "SLURM_ARRAY_TASK_ID:?generated-source pilot must expose its exact serialized-array task id",
            "SLURM_JOB_PARTITION:?generated-source pilot must record its Slurm partition",
            "CUDA_VISIBLE_DEVICES:?generated-source pilot requires a rendering-only MIG allocation",
            'test "$SLURM_JOB_PARTITION" = mig',
            'test "$GROUP_INDEX" = "$SLURM_ARRAY_TASK_ID"',
            "MUJOCO_GL=osmesa",
            "PYOPENGL_PLATFORM=osmesa",
            "OffScreenRenderEnv",
            "mujoco_offscreen_osmesa_render_and_observation_hashing_only",
            '"policy_server_started": False',
            '"policy_inference_calls": 0',
            '"model_checkpoint_loaded": False',
            '"training_executed": False',
            'config.get("ready_to_run") is not True',
            'config.get("blocked_on") != []',
            'config.get("pilot_only") is not True',
            'config.get("ready_for_training") is not False',
            "main/generate_task0_single_obstacle_source.py",
            "main/validate_task0_single_obstacle_source.py",
            "--config",
            "--output-root",
            "--run-id",
            "--group-index",
            "--expected-config-sha256",
            "--artifact",
            "--expected-source-slurm-job-id",
            "--expected-source-slurm-array-job-id",
            "--expected-source-slurm-array-task-id",
            "--expected-generation-request-id",
            "greq-task0-single-obstacle-v1-",
            "exact_job_artifact_validation",
            'sha256sum "$GENERATOR"',
            'sha256sum "$VALIDATOR"',
            'sha256sum "$RESULT"',
            'sha256sum "$VALIDATION_LOG"',
            "trap cleanup EXIT",
            "tempfile.NamedTemporaryFile",
            "os.replace(temporary, path)",
            "generated_source_pilot_launch_failure",
            'test "$GIT_COMMIT" = "$EXPECTED_GIT_COMMIT"',
            'test "$GIT_DIRTY" = false',
        ):
            self.assertIn(required, source)
        for forbidden in (
            "OPENPI_PYTHON",
            "CHECKPOINT_DIR",
            "serve_policy.py",
            "policy:checkpoint",
            "WebsocketClientPolicy",
            "--host",
            "--port",
            "--train",
            "rm -rf",
            "scancel",
            "pkill",
            "rsync --delete",
        ):
            self.assertNotIn(forbidden, source)

    def test_core_clis_expose_the_frozen_wrapper_interface(self) -> None:
        generator = GENERATOR.read_text(encoding="utf-8")
        validator = VALIDATOR.read_text(encoding="utf-8")
        for option in (
            "--config",
            "--output-root",
            "--run-id",
            "--group-index",
            "--expected-config-sha256",
        ):
            self.assertIn(option, generator)
        for option in (
            "--artifact",
            "--expected-config-sha256",
            "--expected-source-slurm-job-id",
            "--expected-source-slurm-array-job-id",
            "--expected-source-slurm-array-task-id",
            "--expected-generation-request-id",
        ):
            self.assertIn(option, validator)

    def test_submitter_guards_hash_commit_capacity_and_live_nodes(self) -> None:
        source = SUBMIT.read_text(encoding="utf-8")
        for required in (
            "scripts/hpc/preflight.sh",
            "RUN_ID:?",
            "generated-source pilot source files must be committed before submission",
            "generated-source submission requires a clean reviewed tracked worktree",
            "remote source is not synchronized to the reviewed generated-source commit",
            "remote generated-source source tree is dirty",
            "immutable generated-source RUN_ID already exists",
            'config.get("ready_to_run") is not True',
            'config.get("blocked_on") != []',
            'config.get("pilot_only") is not True',
            'config.get("ready_for_training") is not False',
            "main/crfs_oracle/generated_source.py",
            "schemas/generated-source-state.schema.json",
            "scripts/hpc/run_generated_source_validation.sh",
            "squeue -h -u",
            "AllocTRES=",
            "projected_gpus=$((allocated_gpus + 1))",
            "projected_cpus=$((allocated_cpus + 4))",
            "projected_mem_mb=$((allocated_mem_mb + 32 * 1024))",
            "exceed the 2-GPU-equivalent user ceiling",
            "exceed the 16-CPU user ceiling",
            "exceed the 256-GiB user ceiling",
            "sinfo -h -p mig -N",
            "scontrol show node",
            "FreeMem=",
            "minimum_free_mem_mb=$((32 * 1024))",
            "*down*|*drain*|*drng*|*fail*|*maint*|*not_resp*",
            "no healthy MIG node has a fresh 32 GiB",
            "--nodelist=",
            "--exclude=",
            "sbatch --parsable",
            "sbatch did not return an exact numeric generated-source job id",
            "submitted_exact_job_id=",
            "expected_array_task=",
            "pilot_phase=allocation_canary",
            "submitted_independent_validation_job_id=",
            'validation_dependency=afterok:',
            "slurm/generated_source_validate_cpu.sbatch",
            "source-bundle.json",
            "slurm/generated_source_pilot_main.sbatch",
        ):
            self.assertIn(required, source)
        self.assertLess(source.index("scripts/hpc/preflight.sh"), source.index('ssh "$HOST"'))
        remote = source.split("<<'REMOTE'", 1)[1]
        self.assertNotIn("python3", remote)
        for forbidden in ("rm -rf", "scancel", "pkill", "rsync", "--delete"):
            self.assertNotIn(forbidden, source)

    def test_slurm_shape_is_one_request_rendering_only_mig_canary(self) -> None:
        source = SBATCH.read_text(encoding="utf-8")
        for required in (
            "#SBATCH --partition=mig",
            "#SBATCH --gres=gpu:1",
            "#SBATCH --cpus-per-task=4",
            "#SBATCH --mem=32G",
            "#SBATCH --time=01:00:00",
            "#SBATCH --array=0-0%1",
            "scripts/hpc/run_generated_source_pilot.sh",
        ):
            self.assertIn(required, source)

    def test_independent_validator_is_dependency_backed_cpu_only(self) -> None:
        runner = VALIDATE_RUN.read_text(encoding="utf-8")
        batch = VALIDATE_SBATCH.read_text(encoding="utf-8")
        for required in (
            "SOURCE_JOB_ID:?SOURCE_JOB_ID is required",
            "SOURCE_ARTIFACT:?SOURCE_ARTIFACT is required",
            "EXPECTED_GIT_COMMIT:?EXPECTED_GIT_COMMIT is required",
            "--expected-source-slurm-job-id",
            "--expected-generation-request-id",
            ".cpu-validation.json",
            "set +e",
            "validation_status=$?",
            'test "$GIT_COMMIT" = "$EXPECTED_GIT_COMMIT"',
            'test "$GIT_DIRTY" = false',
        ):
            self.assertIn(required, runner)
        for required in (
            "#SBATCH --partition=main",
            "#SBATCH --cpus-per-task=1",
            "#SBATCH --mem=4G",
            "scripts/hpc/run_generated_source_validation.sh",
        ):
            self.assertIn(required, batch)
        for forbidden in ("cuda", "policy", "checkpoint", "train"):
            self.assertNotIn(forbidden, runner.lower())

    def test_independent_validator_preserves_a_failed_json_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            remote = root / "repository"
            validator = remote / "main/validate_task0_single_obstacle_source.py"
            validator.parent.mkdir(parents=True)
            validator.write_text(
                "import json\nprint(json.dumps({'validation_error_count': 1, 'validation_errors': ['forced']}))\nraise SystemExit(1)\n",
                encoding="utf-8",
            )
            subprocess.run(["git", "init", "-q", str(remote)], check=True)
            subprocess.run(["git", "-C", str(remote), "add", "."], check=True)
            subprocess.run(
                [
                    "git", "-C", str(remote), "-c", "user.name=CRFS Test",
                    "-c", "user.email=crfs-test@example.invalid", "commit", "-qm", "fixture",
                ],
                check=True,
            )
            commit = subprocess.check_output(
                ["git", "-C", str(remote), "rev-parse", "HEAD"], text=True
            ).strip()
            artifact = root / "source-bundle.json"
            artifact.write_text("{}\n", encoding="utf-8")
            environment = os.environ.copy()
            environment.update(
                {
                    "SLURM_JOB_ID": "99102",
                    "SLURM_JOB_PARTITION": "main",
                    "SOURCE_JOB_ID": "99101",
                    "SOURCE_ARRAY_JOB_ID": "99101",
                    "SOURCE_ARRAY_TASK_ID": "0",
                    "SOURCE_ARTIFACT": str(artifact),
                    "EXPECTED_CONFIG_SHA256": EXPECTED_CONFIG_SHA256,
                    "GENERATION_REQUEST_ID": "greq-task0-single-obstacle-v1-0000",
                    "EXPECTED_GIT_COMMIT": commit,
                    "REMOTE_REPO": str(remote),
                    "LIBERO_PYTHON": sys.executable,
                }
            )
            result = subprocess.run(
                [str(VALIDATE_RUN)], env=environment, check=False, capture_output=True, text=True
            )
            self.assertEqual(result.returncode, 1, result.stderr)
            report = root / "source-bundle.cpu-validation.json"
            self.assertTrue(report.is_file())
            value = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(value["validation_error_count"], 1)
            self.assertEqual(value["validation_errors"], ["forced"])

    def test_shell_files_are_executable_and_parse(self) -> None:
        for path in (RUN, VALIDATE_RUN, SUBMIT, SBATCH, VALIDATE_SBATCH):
            self.assertTrue(os.access(path, os.X_OK), f"not executable: {path}")
            result = subprocess.run(
                ["bash", "-n", str(path)],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_prelaunch_failure_is_atomic_and_records_render_only_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            remote = root / "repository"
            remote.mkdir()
            environment = os.environ.copy()
            environment.update(
                {
                    "SLURM_JOB_ID": "99101",
                    "SLURM_ARRAY_JOB_ID": "99101",
                    "SLURM_ARRAY_TASK_ID": "0",
                    "SLURM_JOB_PARTITION": "mig",
                    "CUDA_VISIBLE_DEVICES": "MIG-render-test-device",
                    "REMOTE_REPO": str(remote),
                    "LIBERO_PYTHON": sys.executable,
                    "FAILURE_PYTHON": sys.executable,
                    "OUTPUT_ROOT": str(root / "experiments"),
                    "RUN_ID": "generated-source-atomic-failure-test",
                    "EXPERIMENT_CONFIG": str(root / "missing-config.json"),
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
            failure = (
                root
                / "experiments/generated-source-atomic-failure-test/failures/group-index-0.json"
            )
            self.assertTrue(failure.is_file())
            value = json.loads(failure.read_text(encoding="utf-8"))
            self.assertEqual(value["status"], "failed")
            self.assertEqual(value["stage"], "source_binding")
            self.assertEqual(value["artifact_role"], "generated_source_pilot_launch_failure")
            self.assertIs(value["retired_pilot"], True)
            self.assertEqual(value["pilot_role"], "retired_source_integrity_only")
            self.assertEqual(
                value["generation_request_id"],
                "greq-task0-single-obstacle-v1-0000",
            )
            self.assertNotIn("source_state_id", value)
            self.assertIs(value["scientific_claim_allowed"], False)
            self.assertIs(value["probe_training_authorized"], False)
            self.assertEqual(value["policy_inference_calls"], 0)
            self.assertIs(value["policy_server_started"], False)
            self.assertIs(value["model_checkpoint_loaded"], False)
            self.assertIs(value["training_executed"], False)
            self.assertEqual(
                value["provenance"],
                {
                    "slurm_job_id": "99101",
                    "slurm_array_job_id": "99101",
                    "slurm_array_task_id": "0",
                    "partition": "mig",
                    "allocation_visible_gpu": "MIG-render-test-device",
                    "render_backend": "osmesa",
                    "mujoco_gl": "osmesa",
                },
            )

    def test_submitter_rejects_nonimmutable_run_id_before_preflight(self) -> None:
        environment = os.environ.copy()
        environment["RUN_ID"] = "../mutable/run"
        result = subprocess.run(
            [str(SUBMIT), "unused.json"],
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
