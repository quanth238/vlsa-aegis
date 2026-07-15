from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "r05a-exact-array-task-afterany-regression-20260715a"
COMMIT = "a" * 40
HELPER = ROOT / "scripts" / "hpc" / "lib" / "slurm_exact_array_task_status.sh"
PRODUCTION_VALIDATOR = ROOT / "scripts" / "hpc" / "validate_r05a_sampled_current_canary.sh"
SOURCE = ROOT / "scripts" / "hpc" / "run_r05a_exact_task_status_source.sh"
VALIDATOR = ROOT / "scripts" / "hpc" / "validate_r05a_exact_task_status_regression.sh"
SUBMITTER = ROOT / "scripts" / "hpc" / "submit_r05a_exact_task_status_regression.sh"
SOURCE_SLURM = ROOT / "slurm" / "r05a_exact_task_status_source_cpu.sbatch"
VALIDATOR_SLURM = ROOT / "slurm" / "r05a_exact_task_status_validate_cpu.sbatch"
ADR0039 = (
    ROOT
    / "docs"
    / "decisions"
    / "0039-preserve-sampled-current-launch-b-and-repair-exact-task-accounting.md"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _fake_control_bin(root: Path, *, source_job_id: str = "9001") -> Path:
    fake_bin = root / "bin"
    fake_bin.mkdir()
    scripts = {
        "hostname": "#!/usr/bin/env bash\necho worker-1\n",
        "git": (
            "#!/usr/bin/env bash\n"
            "set -eu\n"
            'if [ "${1:-}" = -C ]; then shift 2; fi\n'
            'case "${1:-} ${2:-}" in\n'
            f'  "rev-parse HEAD") echo {COMMIT} ;;\n'
            '  "status --porcelain") : ;;\n'
            '  *) exit 2 ;;\n'
            "esac\n"
        ),
        "sacct": (
            "#!/usr/bin/env bash\n"
            "set -eu\n"
            f"echo '{source_job_id}_0|COMPLETED|0:0'\n"
        ),
    }
    for name, source in scripts.items():
        path = fake_bin / name
        path.write_text(source, encoding="utf-8")
        path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return fake_bin


def _fake_submitter_bin(root: Path) -> Path:
    fake_bin = root / "submitter-bin"
    fake_bin.mkdir()
    calls = root / "calls"
    calls.mkdir()
    scripts = {
        "git": (
            "#!/usr/bin/env bash\n"
            "set -eu\n"
            'if [ "${1:-}" = -C ]; then shift 2; fi\n'
            'case "${1:-}" in\n'
            "  ls-files) exit 0 ;;\n"
            "  status) exit 0 ;;\n"
            f"  rev-parse) echo {COMMIT} ;;\n"
            "  *) exit 2 ;;\n"
            "esac\n"
        ),
        "ssh": (
            "#!/usr/bin/env bash\n"
            "set -eu\n"
            'if [ "$#" -eq 2 ]; then\n'
            "  cat >/dev/null\n"
            "  echo host=login-restricted-1\n"
            "  echo '[paths]'\n"
            "  echo present /mnt/data/quanth/experiments\n"
            "  echo '[login-process-audit]'\n"
            "  exit 0\n"
            "fi\n"
            "shift\n"
            'exec "$1" "$2" "$3" "$4" "$5" "$6" "$FAKE_EXPERIMENT_ROOT" "$FAKE_SLURM_LOG_ROOT"\n'
        ),
        "squeue": "#!/usr/bin/env bash\nexit 0\n",
        "sbatch": (
            "#!/usr/bin/env bash\n"
            "set -eu\n"
            'count=0; test ! -f "$FAKE_SBATCH_COUNT" || read -r count <"$FAKE_SBATCH_COUNT"\n'
            'count=$((count + 1)); echo "$count" >"$FAKE_SBATCH_COUNT"\n'
            'printf "%s\\n" "$*" >>"$FAKE_SBATCH_CALLS"\n'
            'if [ "$count" -eq 1 ]; then echo 9101; else echo 9102; fi\n'
        ),
        "scontrol": (
            "#!/usr/bin/env bash\n"
            "set -eu\n"
            'printf "%s\\n" "$*" >>"$FAKE_SCONTROL_CALLS"\n'
            'if [ "${1:-}" = show ] && [ "${2:-}" = node ]; then\n'
            "  echo 'NodeName=worker-1 FreeMem=4096 State=IDLE'\n"
            "  exit 0\n"
            "fi\n"
            'if [ "${1:-}" = show ] && [ "${2:-}" = job ]; then\n'
            '  if [ "${3:-}" = 9101 ]; then\n'
            "    echo 'JobId=9101 JobState=PENDING Reason=JobHeldUser ReqNodeList=worker-1 Partition=main Account=normal QOS=normal TimeLimit=00:02:00 Requeue=0 ReqTRES=cpu=1,mem=256M,node=1'\n"
            "  else\n"
            "    echo 'JobId=9102 JobState=PENDING Reason=Dependency ReqNodeList=worker-1 Partition=main Account=normal QOS=normal TimeLimit=00:02:00 Requeue=0 Dependency=afterany:9101(unfulfilled) ReqTRES=cpu=1,mem=256M,node=1'\n"
            "  fi\n"
            "  exit 0\n"
            "fi\n"
            'if [ "${1:-}" = release ] && [ "${2:-}" = 9101 ]; then exit 0; fi\n'
            "exit 2\n"
        ),
    }
    for name, source in scripts.items():
        path = fake_bin / name
        path.write_text(source, encoding="utf-8")
        path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return fake_bin


class R05AExactTaskStatusRegressionTest(unittest.TestCase):
    def test_shell_surface_is_zero_gpu_shell_only_and_syntax_valid(self) -> None:
        for path in (HELPER, SOURCE, VALIDATOR, SUBMITTER, SOURCE_SLURM, VALIDATOR_SLURM):
            with self.subTest(path=path):
                self.assertTrue(path.stat().st_mode & stat.S_IXUSR)
                completed = subprocess.run(
                    ["bash", "-n", str(path)],
                    cwd=ROOT,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
        combined = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (SOURCE, VALIDATOR, SOURCE_SLURM, VALIDATOR_SLURM)
        )
        for forbidden in ("python3 ", "LIBERO_PYTHON", "OPENPI_PYTHON", "torch", "env.step"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, combined)
        self.assertNotIn("--gres", combined)
        self.assertNotIn("scancel", SUBMITTER.read_text(encoding="utf-8"))

    def test_exact_identity_resources_dependency_and_release_order_are_frozen(self) -> None:
        decision = ADR0039.read_text(encoding="utf-8")
        self.assertIn(RUN_ID, decision)
        for slurm_file in (SOURCE_SLURM, VALIDATOR_SLURM):
            source = slurm_file.read_text(encoding="utf-8")
            self.assertIn("#SBATCH --cpus-per-task=1", source)
            self.assertIn("#SBATCH --mem=256M", source)
            self.assertIn("#SBATCH --time=00:02:00", source)
            self.assertIn("#SBATCH --no-requeue", source)
            self.assertNotIn("#SBATCH --gres", source)
        self.assertIn("#SBATCH --array=0-0%1", SOURCE_SLURM.read_text(encoding="utf-8"))
        self.assertNotIn("#SBATCH --array", VALIDATOR_SLURM.read_text(encoding="utf-8"))
        submitter = SUBMITTER.read_text(encoding="utf-8")
        held = submitter.index("sbatch --parsable --hold")
        contract = submitter.index('mv "$temporary" "$source_contract"')
        afterany = submitter.index("dependency=afterany:$source_job_id")
        receipt = submitter.index('mv "$temporary" "$submission_receipt"')
        release = submitter.index('scontrol release "$source_job_id"')
        self.assertLess(held, contract)
        self.assertLess(contract, afterany)
        self.assertLess(afterany, receipt)
        self.assertLess(receipt, release)

    def test_fake_full_transaction_receipts_before_releasing_only_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fake_bin = _fake_submitter_bin(root)
            experiment_root = root / "experiments"
            log_root = root / "logs"
            experiment_root.mkdir()
            completed = subprocess.run(
                ["bash", str(SUBMITTER)],
                cwd=ROOT,
                env={
                    **os.environ,
                    "PATH": str(fake_bin) + os.pathsep + os.environ.get("PATH", ""),
                    "RUN_ID": RUN_ID,
                    "VINUNI_HOST": "vinuni",
                    "REMOTE_REPO": str(ROOT),
                    "PREFLIGHT_OUT": str(root / "preflight.txt"),
                    "FAKE_EXPERIMENT_ROOT": str(experiment_root),
                    "FAKE_SLURM_LOG_ROOT": str(log_root),
                    "FAKE_SBATCH_COUNT": str(root / "sbatch-count"),
                    "FAKE_SBATCH_CALLS": str(root / "calls" / "sbatch"),
                    "FAKE_SCONTROL_CALLS": str(root / "calls" / "scontrol"),
                },
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            run_root = experiment_root / RUN_ID
            for name in (
                "held-source-submission.json",
                "source-contract.json",
                "submission.json",
            ):
                self.assertTrue((run_root / name).is_file(), name)
            self.assertFalse((run_root / "submission-failure.json").exists())
            submission = json.loads(
                (run_root / "submission.json").read_text(encoding="utf-8")
            )
            self.assertEqual(submission["source_array_job_id"], "9101")
            self.assertEqual(submission["exact_source_task_id"], "9101_0")
            self.assertEqual(submission["validator_job_id"], "9102")
            self.assertEqual(submission["dependency"], "afterany:9101")
            self.assertFalse(submission["released_at_receipt_time"])
            sbatch_calls = (root / "calls" / "sbatch").read_text(
                encoding="utf-8"
            ).splitlines()
            self.assertEqual(len(sbatch_calls), 2)
            self.assertIn("--hold", sbatch_calls[0])
            self.assertIn("--array=0-0%1", sbatch_calls[0])
            self.assertNotIn("--dependency", sbatch_calls[0])
            self.assertIn("--dependency=afterany:9101", sbatch_calls[1])
            self.assertNotIn("--hold", sbatch_calls[1])
            self.assertNotIn("--gres", "\n".join(sbatch_calls))
            scontrol_calls = (root / "calls" / "scontrol").read_text(
                encoding="utf-8"
            ).splitlines()
            self.assertEqual(scontrol_calls[-1], "release 9101")
            self.assertNotIn("release 9102", scontrol_calls)

    def test_source_writes_only_a_pre_exit_marker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fake_bin = _fake_control_bin(root)
            source_contract = root / "source-contract.json"
            submission = root / "submission.json"
            marker = root / "source-marker.json"
            _write_json(
                source_contract,
                {
                    "status": "held_source_bound_before_afterany_submission",
                    "run_id": RUN_ID,
                    "git_commit": COMMIT,
                    "source_array_job_id": "9001",
                    "source_array_task_id": 0,
                    "source_node": "worker-1",
                    "resources": {
                        "partition": "main",
                        "account": "normal",
                        "qos": "normal",
                        "cpus": 1,
                        "host_memory_mib": 256,
                        "time_limit": "00:02:00",
                        "array": "0-0%1",
                        "gpus": 0,
                        "requeue": False,
                    },
                    "artifact_paths": {
                        "source_marker": str(marker),
                        "submission": str(submission),
                    },
                    "shell_only": True,
                    "scientific_claim_allowed": False,
                },
            )
            _write_json(submission, {"status": "held_source_and_afterany_registered"})
            completed = subprocess.run(
                ["bash", str(SOURCE)],
                cwd=ROOT,
                env={
                    **os.environ,
                    "PATH": str(fake_bin) + os.pathsep + os.environ.get("PATH", ""),
                    "SLURM_JOB_ID": "9001_0",
                    "SLURM_ARRAY_JOB_ID": "9001",
                    "SLURM_ARRAY_TASK_ID": "0",
                    "SLURM_JOB_PARTITION": "main",
                    "SLURM_CPUS_PER_TASK": "1",
                    "SLURM_MEM_PER_NODE": "256",
                    "CUDA_VISIBLE_DEVICES": "NoDevFiles",
                    "RUN_ID": RUN_ID,
                    "EXPECTED_GIT_COMMIT": COMMIT,
                    "SOURCE_CONTRACT": str(source_contract),
                    "SUBMISSION": str(submission),
                    "SOURCE_MARKER": str(marker),
                    "REMOTE_REPO": str(ROOT),
                },
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            value = json.loads(marker.read_text(encoding="utf-8"))
            self.assertEqual(value["status"], "body_completed_before_process_exit")
            self.assertEqual(value["exact_source_task_id"], "9001_0")
            self.assertFalse(value["gpu_allocated"])
            self.assertFalse(value["python_executed"])
            self.assertEqual(value["simulator_steps_executed"], 0)
            self.assertFalse(value["h100_retry_authorized"])

    def test_afterany_validator_uses_production_helper_and_publishes_exact_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fake_bin = _fake_control_bin(root)
            source_contract = root / "source-contract.json"
            submission = root / "submission.json"
            marker = root / "source-marker.json"
            result = root / "results.json"
            receipt = root / "cpu-afterany-validation.json"
            _write_json(
                source_contract,
                {
                    "bound_file_sha256": {
                        "scripts/hpc/lib/slurm_exact_array_task_status.sh": _sha256(HELPER),
                        "scripts/hpc/validate_r05a_sampled_current_canary.sh": _sha256(PRODUCTION_VALIDATOR),
                    }
                },
            )
            source_contract_sha = _sha256(source_contract)
            _write_json(
                submission,
                {
                    "status": "held_source_and_afterany_registered",
                    "run_id": RUN_ID,
                    "git_commit": COMMIT,
                    "source_array_job_id": "9001",
                    "source_array_task_id": 0,
                    "validator_job_id": "9002",
                    "dependency": "afterany:9001",
                    "source_contract_sha256": source_contract_sha,
                    "released_at_receipt_time": False,
                    "shell_only": True,
                    "scientific_claim_allowed": False,
                },
            )
            _write_json(
                marker,
                {
                    "status": "body_completed_before_process_exit",
                    "run_id": RUN_ID,
                    "git_commit": COMMIT,
                    "slurm_array_job_id": "9001",
                    "slurm_array_task_id": 0,
                    "exact_source_task_id": "9001_0",
                    "host": "worker-1",
                    "source_contract_path": str(source_contract),
                    "source_contract_sha256": source_contract_sha,
                    "submission_path": str(submission),
                    "submission_sha256": _sha256(submission),
                    "shell_only": True,
                    "gpu_allocated": False,
                    "python_executed": False,
                    "model_inference_executed": False,
                    "simulator_steps_executed": 0,
                    "teacher_searches_executed": 0,
                    "training_executed": False,
                    "scientific_claim_allowed": False,
                },
            )
            completed = subprocess.run(
                ["bash", str(VALIDATOR)],
                cwd=ROOT,
                env={
                    **os.environ,
                    "PATH": str(fake_bin) + os.pathsep + os.environ.get("PATH", ""),
                    "SLURM_JOB_ID": "9002",
                    "SLURM_JOB_PARTITION": "main",
                    "SLURM_CPUS_PER_TASK": "1",
                    "SLURM_MEM_PER_NODE": "256",
                    "CUDA_VISIBLE_DEVICES": "NoDevFiles",
                    "RUN_ID": RUN_ID,
                    "EXPECTED_GIT_COMMIT": COMMIT,
                    "SOURCE_JOB_ID": "9001",
                    "SOURCE_CONTRACT": str(source_contract),
                    "EXPECTED_SOURCE_CONTRACT_SHA256": source_contract_sha,
                    "SUBMISSION": str(submission),
                    "SOURCE_MARKER": str(marker),
                    "RESULT": str(result),
                    "VALIDATION_RECEIPT": str(receipt),
                    "REMOTE_REPO": str(ROOT),
                },
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            value = json.loads(result.read_text(encoding="utf-8"))
            self.assertEqual(value["status"], "passed")
            self.assertEqual(value["exact_source_task_id"], "9001_0")
            self.assertEqual(value["observed_source_state"], "COMPLETED")
            self.assertEqual(value["observed_source_exit_code"], "0:0")
            self.assertEqual(value["source_query_status"], 0)
            self.assertFalse(value["gpu_allocated"])
            self.assertFalse(value["python_executed"])
            self.assertFalse(value["h100_retry_authorized_by_this_artifact_alone"])
            receipt_value = json.loads(receipt.read_text(encoding="utf-8"))
            self.assertTrue(receipt_value["passed"])
            self.assertTrue(receipt_value["published"])
            self.assertEqual(receipt_value["source_task_id"], "9001_0")
            self.assertEqual(receipt_value["validator_job_id"], "9002")


if __name__ == "__main__":
    unittest.main()
