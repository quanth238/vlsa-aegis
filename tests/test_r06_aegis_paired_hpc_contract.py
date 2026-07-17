"""Dependency-light contracts for the R06 paired AEGIS canary path."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import stat
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "main/run_crfs_r06_aegis_paired_canary.py"
VALIDATOR = ROOT / "main/validate_crfs_r06_aegis_paired_canary.py"
WRAPPER = ROOT / "scripts/hpc/run_r06_aegis_paired_canary.sh"
WORKLOAD = ROOT / "scripts/hpc/run_r06_aegis_paired_workload.sh"
CPU_WRAPPER = ROOT / "scripts/hpc/validate_r06_aegis_paired_canary.sh"
SUBMITTER = ROOT / "scripts/hpc/submit_r06_aegis_paired_canary.sh"
GPU_SBATCH = ROOT / "slurm/r06_aegis_paired_canary_h100.sbatch"
CPU_SBATCH = ROOT / "slurm/r06_aegis_paired_canary_validate_cpu.sbatch"
CONFIG = ROOT / "configs/experiments/r06_aegis_collision_conditioned.json"


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _load_cli():
    spec = importlib.util.spec_from_file_location("r06_paired_cli", CLI)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load R06 paired CLI")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _paired_release(run_id: str = "r06-paired-test") -> dict:
    decision = "docs/decisions/0069-release-aegis-paired-canary.md"
    return {
        "artifact_role": "r06_aegis_paired_codex_label_canary_execution_release",
        "stage": "paired_codex_label_canary",
        "run_id": run_id,
        "accepted_implementation_commit": "a" * 40,
        "decision_artifact": decision,
        "allowed_release_diff_paths": [
            "configs/experiments/r06_aegis_collision_conditioned.json",
            decision,
        ],
        "single_case_index": 0,
        "case_id": "crfs-1069f29a8d76463a",
        "aegis_execution_allowed": True,
        "semantic_label_required": True,
        "groundingdino_execution_allowed": True,
        "qp_execution_allowed": True,
        "original_glm_execution_allowed": False,
        "robosuite_image_convention": "opengl",
        "robosuite_version": "1.4.1",
        "probe_or_mlp_training_authorized": False,
        "automatic_population_launch_authorized": False,
        "release_only_parent_required": True,
        "codex_label_manifest": {
            "path": "manifests/r06_codex_obstacle_labels_canary.jsonl",
            "sha256": "6a22b6d2f3705c008e338be6b217ce947fabfd4f56f70d3a8f442467694d996f",
            "freeze_commit": "d9cf569d619e014c9e6423cd9ddb40f592435a71",
            "capture_artifact_path": (
                "/mnt/data/quanth/experiments/crfs-oracle/"
                "r06-aegis-label-capture-canary-20260717b/"
                "crfs-1069f29a8d76463a/capture.json"
            ),
            "capture_artifact_sha256": (
                "f2d32024ac6fac5aa5d133b5138df72501f1590b8cb0927b732edde69dabaaac"
            ),
            "capture_completed_at": "2026-07-17T06:55:59Z",
        },
        "resources": {
            "partition": "main",
            "account": "normal",
            "qos": "normal",
            "gpus": 1,
            "cpus_per_task": 8,
            "host_memory_mib": 65536,
            "time_limit": "00:30:00",
            "array": "0-0%1",
            "source_host": "worker-1",
            "requeue": False,
            "validator_partition": "main",
            "validator_account": "normal",
            "validator_qos": "normal",
            "validator_cpus": 2,
            "validator_host_memory_mib": 8192,
            "validator_time_limit": "00:10:00",
            "validator_gpus": 0,
            "validator_dependency": "afterany",
        },
    }


class R06AegisPairedHpcContractTest(unittest.TestCase):
    def test_shell_files_parse_and_entrypoints_are_executable(self) -> None:
        shell_files = (
            WRAPPER,
            WORKLOAD,
            CPU_WRAPPER,
            SUBMITTER,
            GPU_SBATCH,
            CPU_SBATCH,
        )
        for path in shell_files:
            result = subprocess.run(
                ["bash", "-n", str(path)],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, msg=f"{path}: {result.stderr}")
        for path in (CLI, VALIDATOR, WRAPPER, WORKLOAD, CPU_WRAPPER, SUBMITTER):
            self.assertTrue(path.stat().st_mode & stat.S_IXUSR, path)

    def test_gpu_and_zero_gpu_validator_resources_are_exact(self) -> None:
        gpu = _text(GPU_SBATCH)
        for token in (
            "#SBATCH --gres=gpu:1",
            "#SBATCH --cpus-per-task=8",
            "#SBATCH --mem=64G",
            "#SBATCH --time=00:30:00",
            "#SBATCH --array=0-0%1",
            "#SBATCH --nodelist=worker-1",
            "#SBATCH --no-requeue",
        ):
            self.assertIn(token, gpu)
        cpu = _text(CPU_SBATCH)
        for token in (
            "#SBATCH --cpus-per-task=2",
            "#SBATCH --mem=8G",
            "#SBATCH --time=00:10:00",
            "#SBATCH --no-requeue",
        ):
            self.assertIn(token, cpu)
        self.assertNotIn("--gres=gpu", cpu)
        self.assertIn("NoDevFiles", _text(CPU_WRAPPER))

    def test_cli_accepts_only_exact_paired_release(self) -> None:
        module = _load_cli()
        release = _paired_release()
        value = {"ready_to_run": True, "blocked_on": [], "execution_release": release}
        self.assertEqual(
            module._require_paired_release(value, run_id="r06-paired-test"),
            release,
        )
        for key, replacement in (
            ("stage", "codex_label_capture"),
            ("aegis_execution_allowed", False),
            ("groundingdino_execution_allowed", False),
            ("qp_execution_allowed", False),
            ("original_glm_execution_allowed", True),
        ):
            changed = json.loads(json.dumps(value))
            changed["execution_release"][key] = replacement
            with self.assertRaises(ValueError):
                module._require_paired_release(changed, run_id="r06-paired-test")

    def test_config_is_fail_closed_or_exact_paired_release(self) -> None:
        value = json.loads(CONFIG.read_text(encoding="utf-8"))
        if value["ready_to_run"] is False:
            self.assertIsNone(value["execution_release"])
            self.assertTrue(value["blocked_on"])
            return
        release = value["execution_release"]
        module = _load_cli()
        self.assertEqual(
            module._require_paired_release(value, run_id=release["run_id"]),
            release,
        )

    def test_cli_calls_exact_runtime_provider_and_paired_api(self) -> None:
        value = _text(CLI)
        for token in (
            "SafeLiberoAegisRuntime(SafeLiberoCase(case, oracle))",
            "CodexFrozenLabelSafetyCoreProvider(",
            "run_aegis_case(",
            "validate_aegis_case_result(value)",
            "runtime.close()",
            'return 2 if status == "apparatus_invalid" else 0',
            '"merge-base", "--is-ancestor"',
        ):
            self.assertIn(token, value)
        self.assertNotIn("run_aegis_label_capture(", value)

    def test_workload_binds_every_static_input_and_preflights_full_stack(self) -> None:
        value = _text(WORKLOAD)
        for digest in (
            "b12319d3fed151bfee85e1615bd72254474a1480308389d747dce954c6131e41",
            "6a22b6d2f3705c008e338be6b217ce947fabfd4f56f70d3a8f442467694d996f",
            "c5be1b4759487f4d6541e49110b90fcf5de404bdaed5f389358db0abe4388f4e",
            "055fcf18781071c6c3575b42a1b44c32a76b474ac1911ad8c5443f78b9c42593",
            "f2d32024ac6fac5aa5d133b5138df72501f1590b8cb0927b732edde69dabaaac",
            "988055ccfd7032903c073a641f3c5f0f0541df444a315116a16f0bf4716d26ed",
            "5c2728c53f4b33ee16380140f303713fdb5df78a8d26ee1489ace374fc54327a",
            "b3a44bb2810436fb62917decaea58bd4d9110255df527dea21e8fd40c960bd84",
            "172e80017f9395668a9cb5d1b8bd9d061c0e360471c6ed673c83b69bb14399f1",
            "3b3ca2563c77c69f651d7bd133e97139c186df06231157a64c507099c52bc799",
        ):
            self.assertIn(digest, value)
        for package in (
            "import torch",
            "import groundingdino",
            "import open3d",
            "import cvxpy",
            "import osqp",
            "import scipy",
        ):
            self.assertIn(package, value)
        self.assertLess(value.index('cd "$REMOTE_REPO"'), value.index("import torch"))
        self.assertIn("focused_test_skips=0", value)
        self.assertIn("tests.test_aegis_runner", value)
        self.assertNotIn("ZHIPUAI_API_KEY", value)

    def test_submitter_registers_both_jobs_before_one_release(self) -> None:
        value = _text(SUBMITTER)
        ordered = (
            'mv "$source_tmp" "$source_contract"',
            "sbatch --parsable --hold",
            '--dependency="afterany:$gpu_id"',
            'mv "$submission_tmp" "$submission"',
            'scontrol release "$gpu_id"',
        )
        positions = [value.index(token) for token in ordered]
        self.assertEqual(positions, sorted(positions))
        self.assertEqual(value.count("sbatch --parsable --hold"), 1)
        self.assertEqual(value.count("scontrol release"), 1)
        self.assertIn('scancel "${gpu_id}_0"', value)
        self.assertIn('scancel "$cpu_id"', value)
        self.assertNotIn("scancel -u", value)
        self.assertNotIn("rsync --delete", value)
        self.assertIn("gpu_and_afterany_registered_before_release", value)

    def test_cpu_validator_retains_method_failure_but_forbids_claims(self) -> None:
        cli = _text(CLI)
        validator = _text(VALIDATOR)
        self.assertIn('status == "apparatus_invalid"', cli)
        self.assertNotIn('status == "method_failure"', cli)
        self.assertIn('"result_status": value.get("status")', validator)
        self.assertIn('"aegis_safety_result_allowed"', validator)
        self.assertIn('"general_benchmark_claim_allowed": False', validator)
        self.assertIn('"result_status": result_status', validator)

    def test_cpu_validator_cross_binds_result_to_exact_gpu_execution(self) -> None:
        validator = _text(VALIDATOR)
        for token in (
            'execution.get("run_id")',
            'execution.get("release_git_commit")',
            'execution.get("slurm_array_job_id")',
            'execution.get("slurm_array_task_id") == "0"',
            'execution.get("exact_gpu_task_id")',
            'submission.get("exact_gpu_task_id")',
            'execution.get("source_host")',
            'source.get("resources", {}).get("source_host")',
            'execution.get("cuda_visible_devices")',
            "source/submission/exact-task identity",
        ):
            self.assertIn(token, validator)


if __name__ == "__main__":
    unittest.main()
