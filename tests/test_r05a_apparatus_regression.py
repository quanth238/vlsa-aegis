from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "hpc" / "run_r05a_apparatus_regression.sh"
SUBMITTER = ROOT / "scripts" / "hpc" / "submit_r05a_apparatus_regression.sh"
SBATCH = ROOT / "slurm" / "r05a_apparatus_regression_cpu.sbatch"


class R05AApparatusRegressionTest(unittest.TestCase):
    def test_cpu_regression_is_apparatus_only_and_uses_repaired_paths(self) -> None:
        for path in (RUNNER, SUBMITTER, SBATCH):
            subprocess.run(["bash", "-n", str(path)], check=True)
        runner = RUNNER.read_text(encoding="utf-8")
        submitter = SUBMITTER.read_text(encoding="utf-8")
        sbatch = SBATCH.read_text(encoding="utf-8")
        self.assertIn("#SBATCH --cpus-per-task=2", sbatch)
        self.assertIn("#SBATCH --mem=8G", sbatch)
        self.assertIn("#SBATCH --array=0-0%1", sbatch)
        self.assertIn("#SBATCH --no-requeue", sbatch)
        self.assertNotIn("#SBATCH --gres", sbatch)
        self.assertIn("crfs_run_r05a_allocation_tests", runner)
        self.assertIn("crfs_read_live_cgroup_memory_peak", runner)
        self.assertIn("_parse_allocation_test_log", runner)
        self.assertIn("_parse_cgroup_memory_diagnostic", runner)
        self.assertIn("validate_crfs_r05a_apparatus_regression.py", runner)
        independent_validation = runner.index('"$OPENPI_PYTHON" "$VALIDATOR_FILE"')
        publication = runner.index('mv "$RESULT_CANDIDATE" "$RESULT"')
        self.assertLess(independent_validation, publication)
        self.assertIn("checkpoint_loaded\": False", runner)
        self.assertIn("real_pi05_teacher_searches_executed\": 0", runner)
        self.assertIn("real_pi05_teacher_observations_produced\": 0", runner)
        self.assertIn("synthetic_unit_test_solver_calls_excluded_from_guard\": True", runner)
        for forbidden in (
            "serve_policy.py",
            "run_crfs_r05a_canary.py",
            "environment.rollout",
            "env.step",
            "nvidia-smi",
        ):
            self.assertNotIn(forbidden, runner)
        held = submitter.index("sbatch --parsable --hold")
        failure_trap = submitter.index("trap control_failure EXIT")
        receipt = submitter.index('mv "$receipt_tmp" "$run_root/submission.json"', held)
        release = submitter.index('scontrol release "$job_id"', receipt)
        self.assertLess(failure_trap, held)
        self.assertLess(held, receipt)
        self.assertLess(receipt, release)
        self.assertIn("--nodelist=worker-1", submitter)
        self.assertIn("scripts/hpc/preflight.sh", submitter)
        self.assertIn("submission-failure.json", submitter)
        self.assertIn("validator_sha256", submitter)
        self.assertNotIn("scancel", submitter)
        self.assertNotIn("python ", submitter)

    def test_fake_slurm_transaction_receipts_before_exact_release(self) -> None:
        source = SUBMITTER.read_text(encoding="utf-8")
        remote_start = source.index("<<'REMOTE'\n") + len("<<'REMOTE'\n")
        remote_end = source.rindex("\nREMOTE")
        remote_script = source[remote_start:remote_end]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            remote_repo = root / "repo"
            experiment_root = root / "experiments"
            slurm_logs = root / "logs"
            fake_bin = root / "bin"
            call_log = root / "calls.log"
            for path in (
                remote_repo / "slurm" / "r05a_apparatus_regression_cpu.sbatch",
                remote_repo / "scripts" / "hpc" / "run_r05a_apparatus_regression.sh",
                remote_repo / "main" / "crfs_oracle" / "r05a_allocation_tests.json",
                remote_repo / "main" / "validate_crfs_r05a_apparatus_regression.py",
                remote_repo / "scripts" / "hpc" / "lib" / "r05a_allocation_tests.sh",
                remote_repo / "scripts" / "hpc" / "lib" / "cgroup_memory.sh",
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
                "sbatch",
                'printf "sbatch %s\\n" "$*" >>"$FAKE_CALL_LOG"\n'
                'printf "9201\\n"\n',
            )
            executable(
                "scontrol",
                'case "$1 $2" in\n'
                '  "show node") printf "%s\\n" "NodeName=worker-1 FreeMem=9000 State=IDLE" ;;\n'
                '  "show job") printf "%s\\n" "JobId=9201 JobState=PENDING Reason=JobHeldUser ReqNodeList=worker-1 Partition=main Account=normal QOS=normal TimeLimit=00:20:00 Requeue=0 ReqTRES=cpu=2,mem=8G" ;;\n'
                '  "release 9201") printf "release 9201\\n" >>"$FAKE_CALL_LOG" ;;\n'
                '  *) exit 2 ;;\n'
                "esac\n",
            )

            run_id = "r05a-adr0031-apparatus-cpu-20260715a"
            commit = "b" * 40
            environment = os.environ.copy()
            environment.update(
                {
                    "PATH": f"{fake_bin}:{environment['PATH']}",
                    "FAKE_CALL_LOG": str(call_log),
                    "FAKE_COMMIT": commit,
                }
            )
            completed = subprocess.run(
                [
                    "bash",
                    "-s",
                    "--",
                    str(remote_repo),
                    run_id,
                    commit,
                    str(experiment_root),
                    str(slurm_logs),
                ],
                input=remote_script,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=environment,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            run_root = experiment_root / run_id
            receipt = json.loads((run_root / "submission.json").read_text(encoding="utf-8"))
            self.assertEqual(receipt["slurm_array_job_id"], "9201")
            self.assertEqual(receipt["source_node"], "worker-1")
            self.assertEqual(receipt["requested_gpus"], 0)
            self.assertFalse(receipt["scientific_claim_allowed"])
            calls = call_log.read_text(encoding="utf-8").splitlines()
            self.assertTrue(calls[0].startswith("sbatch "))
            self.assertEqual(calls[-1], "release 9201")

    def test_post_sbatch_contract_failure_records_held_job_without_release(self) -> None:
        source = SUBMITTER.read_text(encoding="utf-8")
        remote_start = source.index("<<'REMOTE'\n") + len("<<'REMOTE'\n")
        remote_end = source.rindex("\nREMOTE")
        remote_script = source[remote_start:remote_end]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            remote_repo = root / "repo"
            experiment_root = root / "experiments"
            slurm_logs = root / "logs"
            fake_bin = root / "bin"
            call_log = root / "calls.log"
            for path in (
                remote_repo / "slurm" / "r05a_apparatus_regression_cpu.sbatch",
                remote_repo / "scripts" / "hpc" / "run_r05a_apparatus_regression.sh",
                remote_repo / "main" / "crfs_oracle" / "r05a_allocation_tests.json",
                remote_repo / "main" / "validate_crfs_r05a_apparatus_regression.py",
                remote_repo / "scripts" / "hpc" / "lib" / "r05a_allocation_tests.sh",
                remote_repo / "scripts" / "hpc" / "lib" / "cgroup_memory.sh",
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
                "sbatch",
                'printf "sbatch %s\\n" "$*" >>"$FAKE_CALL_LOG"\n'
                'printf "9301\\n"\n',
            )
            executable(
                "scontrol",
                'case "$1 $2" in\n'
                '  "show node") printf "%s\\n" "NodeName=worker-1 FreeMem=9000 State=IDLE" ;;\n'
                '  "show job") printf "%s\\n" "JobId=9301 JobState=PENDING Reason=JobHeldUser ReqNodeList=worker-1 Partition=debug Account=normal QOS=normal TimeLimit=00:20:00 Requeue=0 ReqTRES=cpu=2,mem=8G" ;;\n'
                '  "release 9301") printf "release 9301\\n" >>"$FAKE_CALL_LOG" ;;\n'
                '  *) exit 2 ;;\n'
                "esac\n",
            )

            run_id = "r05a-adr0031-apparatus-cpu-20260715a"
            commit = "c" * 40
            environment = os.environ.copy()
            environment.update(
                {
                    "PATH": f"{fake_bin}:{environment['PATH']}",
                    "FAKE_CALL_LOG": str(call_log),
                    "FAKE_COMMIT": commit,
                }
            )
            completed = subprocess.run(
                [
                    "bash",
                    "-s",
                    "--",
                    str(remote_repo),
                    run_id,
                    commit,
                    str(experiment_root),
                    str(slurm_logs),
                ],
                input=remote_script,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=environment,
                check=False,
            )
            self.assertNotEqual(completed.returncode, 0)
            run_root = experiment_root / run_id
            failure = json.loads(
                (run_root / "submission-failure.json").read_text(encoding="utf-8")
            )
            self.assertEqual(failure["control_stage"], "held_job_contract_validation")
            self.assertEqual(failure["slurm_array_job_id"], "9301")
            self.assertFalse(failure["released"])
            self.assertTrue(failure["exact_job_must_be_inspected_before_cleanup"])
            self.assertTrue((run_root / "held-submission.json").is_file())
            self.assertFalse((run_root / "submission.json").exists())
            calls = call_log.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(calls), 1)
            self.assertTrue(calls[0].startswith("sbatch "))


if __name__ == "__main__":
    unittest.main()
