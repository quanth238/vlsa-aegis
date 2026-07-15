from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "experiments" / "r05a_runtime_identity_regression.json"
ADR0043 = ROOT / "docs" / "decisions" / "0043-preregister-runtime-identity-regression.md"
HELPER = ROOT / "scripts" / "hpc" / "lib" / "r05a_runtime_identity.sh"
RUNNER = ROOT / "scripts" / "hpc" / "run_r05a_runtime_identity_regression.sh"
SUBMITTER = ROOT / "scripts" / "hpc" / "submit_r05a_runtime_identity_regression.sh"
SLURM = ROOT / "slurm" / "r05a_runtime_identity_regression_cpu.sbatch"
TEST_FILE = Path(__file__).resolve()
RUN_ID = "r05a-runtime-identity-regression-test"
IMPLEMENTATION = "a" * 40
RELEASE = "b" * 40
BOUND_PATHS = (
    "configs/experiments/r05a_runtime_identity_regression.json",
    "docs/decisions/0043-preregister-runtime-identity-regression.md",
    "docs/decisions/0044-release-runtime-identity-regression.md",
    "scripts/hpc/lib/r05a_runtime_identity.sh",
    "scripts/hpc/run_r05a_runtime_identity_regression.sh",
    "scripts/hpc/submit_r05a_runtime_identity_regression.sh",
    "slurm/r05a_runtime_identity_regression_cpu.sbatch",
    "tests/test_r05a_runtime_identity_regression.py",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_executable(path: Path, source: str) -> None:
    path.write_text(source, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _released_config() -> tuple[dict, dict]:
    parent = json.loads(CONFIG.read_text(encoding="utf-8"))
    released = copy.deepcopy(parent)
    released["ready_to_run"] = True
    released["blocked_on"] = []
    released["execution_release"] = {
        "schema_version": "1.0",
        "artifact_role": "r05a_runtime_identity_regression_execution_release",
        "decision_artifact": "docs/decisions/0044-release-runtime-identity-regression.md",
        "accepted_implementation_commit": IMPLEMENTATION,
        "run_id": RUN_ID,
        "single_submission": True,
        "release_only_parent_required": True,
        "resources": {
            "partition": "main",
            "account": "normal",
            "qos": "normal",
            "source_host": "worker-1",
            "cpus_per_task": 1,
            "host_memory_mib": 256,
            "time_limit": "00:02:00",
            "gpus": 0,
            "requeue": False,
        },
        "allowed_release_diff_paths": [
            "configs/experiments/r05a_runtime_identity_regression.json",
            "docs/decisions/0044-release-runtime-identity-regression.md",
        ],
        "automatic_cancellation_allowed": False,
        "automatic_resubmission_allowed": False,
        "h100_submission_authorized": False,
        "automatic_next_gate_allowed": False,
    }
    return parent, released


class _SubmitterFixture:
    def __init__(
        self,
        root: Path,
        *,
        bad_job: bool = False,
        gpu_job: bool = False,
        release_fails: bool = False,
    ):
        self.root = root
        self.repo = root / "repo"
        self.experiments = root / "experiments"
        self.logs = root / "logs"
        self.calls = root / "calls"
        self.fake_bin = root / "bin"
        self.parent_config = root / "parent-config.json"
        self.bad_job = bad_job
        self.gpu_job = gpu_job
        self.release_fails = release_fails

    def prepare(self) -> None:
        self.repo.mkdir()
        self.experiments.mkdir()
        self.logs.mkdir()
        self.calls.mkdir()
        self.fake_bin.mkdir()
        parent, released = _released_config()
        self.parent_config.write_text(json.dumps(parent) + "\n", encoding="utf-8")
        sources = {
            BOUND_PATHS[0]: json.dumps(released, indent=2) + "\n",
            BOUND_PATHS[1]: ADR0043.read_text(encoding="utf-8"),
            BOUND_PATHS[2]: (
                "# 0044 — Release runtime identity regression\n\n"
                f"- Accepted implementation commit: `{IMPLEMENTATION}`.\n"
                f"- Immutable run ID: `{RUN_ID}`.\n"
                "- H100 submission authorized: `false`.\n"
            ),
            BOUND_PATHS[3]: HELPER.read_text(encoding="utf-8"),
            BOUND_PATHS[4]: RUNNER.read_text(encoding="utf-8"),
            BOUND_PATHS[5]: SUBMITTER.read_text(encoding="utf-8"),
            BOUND_PATHS[6]: SLURM.read_text(encoding="utf-8"),
            BOUND_PATHS[7]: TEST_FILE.read_text(encoding="utf-8"),
        }
        for relative, source in sources.items():
            path = self.repo / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(source, encoding="utf-8")
            if relative.endswith(".sh") or relative.endswith(".sbatch"):
                path.chmod(path.stat().st_mode | stat.S_IXUSR)
        preflight = self.repo / "scripts" / "hpc" / "preflight.sh"
        _write_executable(preflight, "#!/usr/bin/env bash\nset -euo pipefail\n")
        self._write_fake_commands()

    def _write_fake_commands(self) -> None:
        _write_executable(
            self.fake_bin / "git",
            """#!/usr/bin/env bash
set -euo pipefail
if [ "${1:-}" = -C ]; then shift 2; fi
command=${1:-}; shift || true
case "$command" in
  ls-files|status) exit 0 ;;
  rev-parse) echo "$FAKE_RELEASE" ;;
  rev-list) echo "$FAKE_RELEASE $FAKE_IMPLEMENTATION" ;;
  show) cat "$FAKE_PARENT_CONFIG" ;;
  diff)
    printf '%s\n' \
      configs/experiments/r05a_runtime_identity_regression.json \
      docs/decisions/0044-release-runtime-identity-regression.md
    ;;
  *) echo "unexpected fake git command: $command $*" >&2; exit 2 ;;
esac
""",
        )
        _write_executable(
            self.fake_bin / "ssh",
            """#!/usr/bin/env bash
set -euo pipefail
shift
exec "$1" "$2" "$3" "$4" "$5" "$6" "$7" \
  "$FAKE_EXPERIMENT_ROOT" "$FAKE_SLURM_LOG_ROOT"
""",
        )
        _write_executable(self.fake_bin / "squeue", "#!/usr/bin/env bash\nexit 0\n")
        _write_executable(
            self.fake_bin / "sbatch",
            """#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$*" >>"$FAKE_SBATCH_CALLS"
echo 9101
""",
        )
        memory = "512M" if self.bad_job else "256M"
        gpu_tres = ",gres/gpu=1" if self.gpu_job else ""
        release_status = "exit 1" if self.release_fails else "exit 0"
        _write_executable(
            self.fake_bin / "scontrol",
            f"""#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$*" >>"$FAKE_SCONTROL_CALLS"
if [ "${{1:-}} ${{2:-}} ${{3:-}}" = "show node worker-1" ]; then
  echo 'NodeName=worker-1 FreeMem=4096 State=IDLE'
  exit 0
fi
if [ "${{1:-}} ${{2:-}} ${{3:-}}" = "show job 9101" ]; then
  echo 'JobId=9101 JobState=PENDING Reason=JobHeldUser ReqNodeList=worker-1 Partition=main Account=normal QOS=normal TimeLimit=00:02:00 Requeue=0 ReqTRES=cpu=1,mem={memory},node=1{gpu_tres}'
  exit 0
fi
if [ "${{1:-}} ${{2:-}}" = "release 9101" ]; then {release_status}; fi
exit 2
""",
        )

    def run(self) -> subprocess.CompletedProcess[str]:
        env = {
            **os.environ,
            "PATH": str(self.fake_bin) + os.pathsep + os.environ.get("PATH", ""),
            "RUN_ID": RUN_ID,
            "EXPECTED_RELEASE_COMMIT": RELEASE,
            "VINUNI_HOST": "fake-vinuni",
            "REMOTE_REPO": str(self.repo),
            "FAKE_RELEASE": RELEASE,
            "FAKE_IMPLEMENTATION": IMPLEMENTATION,
            "FAKE_PARENT_CONFIG": str(self.parent_config),
            "FAKE_EXPERIMENT_ROOT": str(self.experiments),
            "FAKE_SLURM_LOG_ROOT": str(self.logs),
            "FAKE_SBATCH_CALLS": str(self.calls / "sbatch"),
            "FAKE_SCONTROL_CALLS": str(self.calls / "scontrol"),
        }
        return subprocess.run(
            ["bash", str(self.repo / BOUND_PATHS[5])],
            cwd=self.repo,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )


class R05ARuntimeIdentityRegressionTest(unittest.TestCase):
    def test_shell_surfaces_are_executable_syntax_valid_and_zero_gpu(self) -> None:
        for path in (HELPER, RUNNER, SUBMITTER, SLURM):
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
        slurm = SLURM.read_text(encoding="utf-8")
        self.assertIn("#SBATCH --nodelist=worker-1", slurm)
        self.assertIn("#SBATCH --cpus-per-task=1", slurm)
        self.assertIn("#SBATCH --mem=256M", slurm)
        self.assertIn("#SBATCH --time=00:02:00", slurm)
        self.assertIn("#SBATCH --no-requeue", slurm)
        self.assertNotIn("#SBATCH --array", slurm)
        self.assertNotIn("#SBATCH --gres", slurm)

    def test_config_is_fail_closed_or_exactly_released(self) -> None:
        value = json.loads(CONFIG.read_text(encoding="utf-8"))
        self.assertEqual(value["resource_contract"]["gpus"], 0)
        self.assertEqual(value["resource_contract"]["host_memory_mib"], 256)
        self.assertFalse(value["claim_boundary"]["h100_submission_authorized"])
        if not value["ready_to_run"]:
            self.assertTrue(value["blocked_on"])
            self.assertIsNone(value["execution_release"])
            return
        self.assertEqual(value["blocked_on"], [])
        release = value["execution_release"]
        self.assertEqual(
            set(release),
            {
                "schema_version",
                "artifact_role",
                "decision_artifact",
                "accepted_implementation_commit",
                "run_id",
                "single_submission",
                "release_only_parent_required",
                "resources",
                "allowed_release_diff_paths",
                "automatic_cancellation_allowed",
                "automatic_resubmission_allowed",
                "h100_submission_authorized",
                "automatic_next_gate_allowed",
            },
        )
        self.assertEqual(release["schema_version"], "1.0")
        self.assertEqual(
            release["artifact_role"],
            "r05a_runtime_identity_regression_execution_release",
        )
        self.assertEqual(len(release["accepted_implementation_commit"]), 40)
        int(release["accepted_implementation_commit"], 16)
        self.assertTrue(release["run_id"])
        self.assertTrue(release["single_submission"])
        self.assertTrue(release["release_only_parent_required"])
        self.assertEqual(release["resources"], value["resource_contract"])
        self.assertEqual(
            release["allowed_release_diff_paths"],
            [
                "configs/experiments/r05a_runtime_identity_regression.json",
                "docs/decisions/0044-release-runtime-identity-regression.md",
            ],
        )
        self.assertFalse(release["automatic_cancellation_allowed"])
        self.assertFalse(release["automatic_resubmission_allowed"])
        self.assertFalse(release["h100_submission_authorized"])
        self.assertFalse(release["automatic_next_gate_allowed"])

    def test_transaction_order_bound_set_and_no_automatic_gpu_path(self) -> None:
        submitter = SUBMITTER.read_text(encoding="utf-8")
        held_job = submitter.index("sbatch --parsable --hold")
        held_receipt = submitter.index('mv "$temporary" "$held"', held_job)
        source = submitter.index('mv "$temporary" "$source_contract"', held_receipt)
        submission = submitter.index('mv "$temporary" "$submission_receipt"', source)
        final = submitter.index("control_stage=final_pre_release_validation", submission)
        release = submitter.index('scontrol release "$held_job_id"', final)
        self.assertLess(held_job, held_receipt)
        self.assertLess(held_receipt, source)
        self.assertLess(source, submission)
        self.assertLess(submission, final)
        self.assertLess(final, release)
        self.assertEqual(submitter.count("sbatch --parsable"), 1)
        self.assertEqual(submitter.count("scontrol release"), 1)
        self.assertNotIn("scancel", submitter)
        self.assertNotIn("--array", submitter)
        self.assertNotIn("--gres", submitter)
        for relative in BOUND_PATHS:
            self.assertIn(relative, submitter)
            self.assertIn(relative, RUNNER.read_text(encoding="utf-8"))

    def test_login_submitter_contains_no_experiment_execution_command(self) -> None:
        submitter = SUBMITTER.read_text(encoding="utf-8")
        for forbidden in (
            "python3 ",
            "uv run",
            "torchrun",
            "nvidia-smi",
            "env.step",
            "mujoco",
            "--gres",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, submitter)
        self.assertIn("scripts/hpc/preflight.sh", submitter)
        self.assertIn('ssh "$HOST" bash -s --', submitter)

    def test_runner_rejects_array_gpu_and_arbitrary_artifact_paths(self) -> None:
        runner = RUNNER.read_text(encoding="utf-8")
        for token in (
            "SLURM_ARRAY_JOB_ID",
            "SLURM_ARRAY_TASK_ID",
            "CUDA_VISIBLE_DEVICES",
            "SLURM_JOB_GPUS",
            'SOURCE_CONTRACT" = "$RUN_ROOT/source-contract.json',
            'SUBMISSION" = "$RUN_ROOT/submission.json',
            'RESULT" = "$RUN_ROOT/results.json',
            'FAILURE" = "$RUN_ROOT/failure.json',
            "EXPECTED_BOUND_KEYS=",
            "source contract differs from final submission token",
            "held receipt differs from final submission token",
        ):
            with self.subTest(token=token):
                self.assertIn(token, runner)
        validation = runner.index("crfs_validate_r05a_openpi_python")
        validation2 = runner.index("crfs_validate_r05a_libero_python", validation)
        publication = runner.index("FAILURE_STAGE=result_publication", validation2)
        self.assertLess(validation2, publication)

    def test_helper_hashes_but_never_executes_a_launcher(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = root / "executed"
            resolved = root / "python-real"
            resolved.write_text(
                f"#!/usr/bin/env bash\ntouch {marker}\n", encoding="utf-8"
            )
            resolved.chmod(resolved.stat().st_mode | stat.S_IXUSR)
            intermediate = root / "python-middle"
            intermediate.symlink_to(resolved)
            public = root / "python"
            public.symlink_to(intermediate)
            command = f"""
set -euo pipefail
. {HELPER!s}
crfs_require_exact_interpreter_identity test {public!s} {public!s} {intermediate!s} {resolved.resolve()!s} {_sha256(resolved)}
"""
            completed = subprocess.run(
                ["bash", "-c", command],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertFalse(marker.exists())

    def test_fake_full_transaction_writes_exact_chain_then_releases_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _SubmitterFixture(Path(directory))
            fixture.prepare()
            completed = fixture.run()
            self.assertEqual(completed.returncode, 0, completed.stderr)
            run_root = fixture.experiments / RUN_ID
            held = run_root / "held-source-submission.json"
            source = run_root / "source-contract.json"
            submission = run_root / "submission.json"
            for path in (held, source, submission):
                self.assertTrue(path.is_file(), path)
            source_value = json.loads(source.read_text(encoding="utf-8"))
            submission_value = json.loads(submission.read_text(encoding="utf-8"))
            self.assertEqual(tuple(sorted(source_value["bound_file_sha256"])), BOUND_PATHS)
            self.assertEqual(source_value["held_submission_sha256"], _sha256(held))
            self.assertEqual(submission_value["held_submission_sha256"], _sha256(held))
            self.assertEqual(submission_value["source_contract_sha256"], _sha256(source))
            self.assertFalse(submission_value["released_at_receipt_time"])
            self.assertFalse(submission_value["gpu_allocated"])
            self.assertFalse(submission_value["h100_submission_authorized"])
            sbatch_calls = (fixture.calls / "sbatch").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(sbatch_calls), 1)
            self.assertIn("--hold", sbatch_calls[0])
            self.assertIn("--nodelist=worker-1", sbatch_calls[0])
            self.assertIn("--cpus-per-task=1", sbatch_calls[0])
            self.assertIn("--mem=256M", sbatch_calls[0])
            self.assertNotIn("--gres", sbatch_calls[0])
            self.assertNotIn("--array", sbatch_calls[0])
            control_calls = (fixture.calls / "scontrol").read_text(encoding="utf-8").splitlines()
            self.assertEqual(control_calls[-1], "release 9101")
            self.assertEqual(control_calls.count("release 9101"), 1)
            self.assertFalse((run_root / "submission-failure.json").exists())

    def test_bad_held_resource_is_fail_closed_without_release_or_resubmit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _SubmitterFixture(Path(directory), bad_job=True)
            fixture.prepare()
            completed = fixture.run()
            self.assertNotEqual(completed.returncode, 0)
            run_root = fixture.experiments / RUN_ID
            failure = json.loads(
                (run_root / "submission-failure.json").read_text(encoding="utf-8")
            )
            self.assertEqual(failure["slurm_job_id"], "9101")
            self.assertFalse(failure["release_attempted"])
            self.assertFalse(failure["release_confirmed"])
            self.assertFalse(failure["automatic_cancellation_allowed"])
            self.assertFalse(failure["automatic_resubmission_allowed"])
            self.assertEqual(
                len((fixture.calls / "sbatch").read_text(encoding="utf-8").splitlines()),
                1,
            )
            controls = (fixture.calls / "scontrol").read_text(encoding="utf-8")
            self.assertNotIn("release 9101", controls)

    def test_any_gpu_tres_is_fail_closed_without_release(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _SubmitterFixture(Path(directory), gpu_job=True)
            fixture.prepare()
            completed = fixture.run()
            self.assertNotEqual(completed.returncode, 0)
            run_root = fixture.experiments / RUN_ID
            failure = json.loads(
                (run_root / "submission-failure.json").read_text(encoding="utf-8")
            )
            self.assertEqual(failure["slurm_job_id"], "9101")
            self.assertFalse(failure["release_attempted"])
            self.assertFalse(failure["h100_submission_authorized"])
            controls = (fixture.calls / "scontrol").read_text(encoding="utf-8")
            self.assertNotIn("release 9101", controls)

    def test_ambiguous_release_failure_is_recorded_once_not_retried(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _SubmitterFixture(Path(directory), release_fails=True)
            fixture.prepare()
            completed = fixture.run()
            self.assertNotEqual(completed.returncode, 0)
            failure = json.loads(
                (
                    fixture.experiments
                    / RUN_ID
                    / "submission-failure.json"
                ).read_text(encoding="utf-8")
            )
            self.assertTrue(failure["release_attempted"])
            self.assertFalse(failure["release_confirmed"])
            controls = (fixture.calls / "scontrol").read_text(encoding="utf-8")
            self.assertEqual(controls.splitlines().count("release 9101"), 1)
            self.assertEqual(
                len((fixture.calls / "sbatch").read_text(encoding="utf-8").splitlines()),
                1,
            )


if __name__ == "__main__":
    unittest.main()
