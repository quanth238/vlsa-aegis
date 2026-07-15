from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "scripts" / "hpc" / "lib" / "cgroup_v2_current_capability.sh"
RUNNER = ROOT / "scripts" / "hpc" / "run_r05a_cgroup_v2_current_capability.sh"
SUBMITTER = ROOT / "scripts" / "hpc" / "submit_r05a_cgroup_v2_current_capability.sh"
SBATCH = ROOT / "slurm" / "r05a_cgroup_v2_current_capability_cpu.sbatch"
ADR = ROOT / "docs" / "decisions" / "0035-preregister-r05a-sampled-current-capability-gate.md"


def _summary(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split("\t")
        if len(fields) == 3 and fields[0] == "summary":
            if fields[1] in result:
                raise AssertionError(f"duplicate summary field {fields[1]}")
            result[fields[1]] = fields[2]
    return result


class CgroupV2CurrentCapabilityTest(unittest.TestCase):
    def _fixture(
        self,
        root: Path,
        *,
        maximum: str = "268435456",
        current: str = "4096",
        native_peak: str | None = None,
        local_events_mount: bool = False,
    ) -> tuple[Path, Path, Path, Path, str]:
        mount = root / "cgroup2"
        membership = "/kubepods.slice/system.slice/slurmstepd.scope/job_42/step_batch/user/task_0"
        logical = membership
        while True:
            directory = mount / logical.lstrip("/")
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "memory.current").write_text(current + "\n", encoding="utf-8")
            (directory / "memory.max").write_text(
                (maximum if logical.endswith("/job_42") else "268435456") + "\n",
                encoding="utf-8",
            )
            events = "low 0\nhigh 0\nmax 0\noom 0\noom_kill 0\n"
            (directory / "memory.events").write_text(events, encoding="utf-8")
            (directory / "memory.events.local").write_text(events, encoding="utf-8")
            if logical.endswith("/job_42"):
                if native_peak is not None:
                    (directory / "memory.peak").write_text(native_peak + "\n", encoding="utf-8")
                break
            logical = logical.rsplit("/", 1)[0]
        # A readable peak at the shared parent must never be selected for the
        # exact job-scoped capability decision.
        shared_parent = mount / membership.split("/job_42", 1)[0].lstrip("/")
        (shared_parent / "memory.peak").write_text("999999\n", encoding="utf-8")
        proc_cgroup = root / "cgroup"
        proc_cgroup.write_text(f"0::{membership}\n", encoding="utf-8")
        mountinfo = root / "mountinfo"
        super_options = "rw,memory_localevents" if local_events_mount else "rw"
        mountinfo.write_text(
            f"36 25 0:32 / {mount} rw,nosuid,nodev,noexec,relatime - cgroup2 cgroup {super_options}\n",
            encoding="utf-8",
        )
        osrelease = root / "osrelease"
        osrelease.write_text("5.15.0-130-generic\n", encoding="utf-8")
        output = root / "capability.tsv"
        return proc_cgroup, mountinfo, osrelease, output, membership

    def _run_helper(
        self,
        root: Path,
        *,
        maximum: str = "268435456",
        current: str = "4096",
        native_peak: str | None = None,
        local_events_mount: bool = False,
        timestamps_ns: list[int] | None = None,
        sample_delay: str = "0",
        event_delta_key: str | None = None,
    ) -> tuple[subprocess.CompletedProcess[str], Path, str]:
        proc_cgroup, mountinfo, osrelease, output, membership = self._fixture(
            root,
            maximum=maximum,
            current=current,
            native_peak=native_peak,
            local_events_mount=local_events_mount,
        )
        if timestamps_ns is None:
            timestamps_ns = [1_000_000_000] * 3
        clock = root / "monotonic-clock.txt"
        clock.write_text(
            "".join(f"{value}\n" for value in timestamps_ns), encoding="utf-8"
        )
        job_events = (
            root
            / "cgroup2"
            / membership.split("/step_batch", 1)[0].lstrip("/")
            / "memory.events"
        )
        environment = os.environ.copy()
        environment["CRFS_TEST_CLOCK"] = str(clock)
        if event_delta_key is not None:
            environment["CRFS_TEST_EVENT_FILE"] = str(job_events)
            environment["CRFS_TEST_EVENT_KEY"] = event_delta_key
        completed = subprocess.run(
            [
                "bash",
                "-c",
                '. "$1"; '
                '_crfs_cap_monotonic_ns() { '
                'IFS= read -r now <"$CRFS_TEST_CLOCK" || return 1; '
                'tail -n +2 "$CRFS_TEST_CLOCK" >"$CRFS_TEST_CLOCK.next" || return 1; '
                'mv "$CRFS_TEST_CLOCK.next" "$CRFS_TEST_CLOCK" || return 1; '
                'printf "%s\\n" "$now"; }; '
                'sleep() { '
                'if [ -n "${CRFS_TEST_EVENT_FILE:-}" ]; then '
                'case "$CRFS_TEST_EVENT_KEY" in '
                'max) max=1; oom=0; oom_kill=0 ;; '
                'oom) max=0; oom=1; oom_kill=0 ;; '
                'oom_kill) max=0; oom=0; oom_kill=1 ;; '
                '*) return 1 ;; esac; '
                'printf "low 0\\nhigh 0\\nmax %s\\noom %s\\noom_kill %s\\n" '
                '"$max" "$oom" "$oom_kill" >"$CRFS_TEST_EVENT_FILE"; '
                'unset CRFS_TEST_EVENT_FILE CRFS_TEST_EVENT_KEY; fi; }; '
                'crfs_probe_cgroup_v2_current_capability "$2" "$3" "$4" "$5" 42 "$6" "$7"',
                "capability-fixture",
                str(HELPER),
                str(proc_cgroup),
                str(mountinfo),
                str(osrelease),
                str(output),
                str(len(timestamps_ns)),
                sample_delay,
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment,
            check=False,
        )
        return completed, output, membership

    def test_bounded_fixture_supports_sampled_current_without_calling_it_peak(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            completed, output, membership = self._run_helper(Path(directory))
            self.assertEqual(completed.returncode, 0, completed.stderr)
            summary = _summary(output)
            self.assertEqual(summary["outcome"], "sampled_current_contract_supported")
            self.assertEqual(summary["sample_count_observed"], "3")
            self.assertEqual(summary["sampled_memory_current_high_water_bytes"], "4096")
            self.assertEqual(summary["memory_events_max_delta"], "0")
            self.assertEqual(summary["memory_events_oom_delta"], "0")
            self.assertEqual(summary["memory_events_oom_kill_delta"], "0")
            rows = [line.split("\t") for line in output.read_text(encoding="utf-8").splitlines()]
            scopes = [row for row in rows if row[0] == "scope"]
            self.assertEqual([row[2] for row in scopes], ["task", "intermediate", "step", "job"])
            self.assertEqual(scopes[0][3], membership)
            self.assertTrue(scopes[-1][3].endswith("/job_42"))
            self.assertFalse(any("system.slice" == row[3].rsplit("/", 1)[-1] for row in scopes))
            peak_rows = [row for row in rows if "memory_peak" in "\t".join(row)]
            self.assertEqual(len(peak_rows), 1)
            self.assertEqual(peak_rows[0][2], "missing")

    def test_unlimited_memory_max_is_completed_unsupported_outcome(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            completed, output, _ = self._run_helper(Path(directory), maximum="max")
            self.assertEqual(completed.returncode, 0, completed.stderr)
            summary = _summary(output)
            self.assertEqual(summary["job_memory_max_state"], "readable_unlimited")
            self.assertEqual(summary["outcome"], "sampled_current_contract_unsupported")

    def test_zero_ceiling_or_zero_only_current_is_completed_unsupported(self) -> None:
        for kwargs in ({"maximum": "0"}, {"current": "0"}):
            with self.subTest(**kwargs), tempfile.TemporaryDirectory() as directory:
                completed, output, _ = self._run_helper(Path(directory), **kwargs)
                self.assertEqual(completed.returncode, 0, completed.stderr)
                summary = _summary(output)
                self.assertEqual(summary["outcome"], "sampled_current_contract_unsupported")
                self.assertEqual(
                    summary["sampled_current_contract_capability_supported"], "false"
                )

    def test_exact_job_native_peak_is_reported_without_selecting_sampled_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            completed, output, _ = self._run_helper(
                Path(directory), native_peak="8192"
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            summary = _summary(output)
            self.assertEqual(summary["outcome"], "native_memory_peak_available")
            self.assertEqual(
                summary["sampled_current_contract_capability_supported"], "false"
            )

    def test_zero_native_peak_is_completed_unsupported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            completed, output, _ = self._run_helper(
                Path(directory), native_peak="0"
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            summary = _summary(output)
            self.assertEqual(summary["outcome"], "sampled_current_contract_unsupported")
            self.assertEqual(
                summary["sampled_current_contract_capability_supported"], "false"
            )

    def test_local_only_memory_events_mount_is_completed_unsupported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            completed, output, _ = self._run_helper(
                Path(directory), local_events_mount=True
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            summary = _summary(output)
            self.assertEqual(
                summary["memory_events_hierarchy_state"],
                "local_only_memory_localevents",
            )
            self.assertEqual(summary["outcome"], "sampled_current_contract_unsupported")
            self.assertEqual(
                summary["sampled_current_contract_capability_supported"], "false"
            )

    def test_production_sample_schedule_and_gap_threshold(self) -> None:
        start = 1_000_000_000
        good = [start + index * 100_000_000 for index in range(20)]
        with tempfile.TemporaryDirectory() as directory:
            completed, output, _ = self._run_helper(
                Path(directory), timestamps_ns=good, sample_delay="0.1"
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            summary = _summary(output)
            self.assertEqual(summary["sample_count_observed"], "20")
            self.assertEqual(summary["sample_max_gap_ns"], "100000000")
            self.assertEqual(summary["outcome"], "sampled_current_contract_supported")

        too_wide = good.copy()
        for index in range(10, len(too_wide)):
            too_wide[index] += 500_000_001
        with tempfile.TemporaryDirectory() as directory:
            completed, output, _ = self._run_helper(
                Path(directory), timestamps_ns=too_wide, sample_delay="0.1"
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            summary = _summary(output)
            self.assertEqual(summary["sample_max_gap_ns"], "600000001")
            self.assertEqual(summary["outcome"], "sampled_current_contract_unsupported")

    def test_new_limit_or_oom_event_is_completed_unsupported(self) -> None:
        timestamps = [1_000_000_000, 1_100_000_000, 1_200_000_000]
        summary_key = {
            "max": "memory_events_max_delta",
            "oom": "memory_events_oom_delta",
            "oom_kill": "memory_events_oom_kill_delta",
        }
        for key in summary_key:
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                completed, output, _ = self._run_helper(
                    Path(directory),
                    timestamps_ns=timestamps,
                    sample_delay="0.1",
                    event_delta_key=key,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
                summary = _summary(output)
                self.assertEqual(summary[summary_key[key]], "1")
                self.assertEqual(
                    summary["outcome"], "sampled_current_contract_unsupported"
                )

    def test_missing_exact_job_boundary_fails_without_partial_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            proc_cgroup, mountinfo, osrelease, output, _ = self._fixture(root)
            proc_cgroup.write_text("0::/not/the/registered/job\n", encoding="utf-8")
            completed = subprocess.run(
                [
                    "bash",
                    "-c",
                    '. "$1"; _crfs_cap_monotonic_ns() { printf "1000000000\\n"; }; '
                    'crfs_probe_cgroup_v2_current_capability "$2" "$3" "$4" "$5" 42 3 0',
                    "capability-fixture",
                    str(HELPER),
                    str(proc_cgroup),
                    str(mountinfo),
                    str(osrelease),
                    str(output),
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertFalse(output.exists())
            self.assertEqual(list(root.glob("capability.tsv.tmp.*")), [])

    def test_scripts_are_shell_only_bounded_atomic_and_exactly_resourced(self) -> None:
        for path in (HELPER, RUNNER, SUBMITTER, SBATCH):
            subprocess.run(["bash", "-n", str(path)], check=True)
        helper = HELPER.read_text(encoding="utf-8")
        runner = RUNNER.read_text(encoding="utf-8")
        submitter = SUBMITTER.read_text(encoding="utf-8")
        sbatch = SBATCH.read_text(encoding="utf-8")
        adr = ADR.read_text(encoding="utf-8")
        self.assertIn("#SBATCH --cpus-per-task=1", sbatch)
        self.assertIn("#SBATCH --mem=256M", sbatch)
        self.assertIn("#SBATCH --time=00:02:00", sbatch)
        self.assertIn("#SBATCH --array=0-0%1", sbatch)
        self.assertIn("#SBATCH --no-requeue", sbatch)
        self.assertNotIn("#SBATCH --gres", sbatch)
        self.assertIn("sampled_memory_current_is_lower_bound_not_peak", runner)
        self.assertIn("memory_max_is_job_scope_hard_limit_not_usage", runner)
        self.assertIn("job_$SLURM_ARRAY_JOB_ID", adr)
        self.assertIn("memory.events.local", helper)
        self.assertNotIn("date +%s", helper)
        self.assertIn('mv "$RESULT_CANDIDATE" "$RESULT"', runner)
        self.assertIn("failure.json", runner)
        held = submitter.index("sbatch --parsable --hold")
        receipt = submitter.index('mv "$temporary" "$run_root/submission.json"', held)
        release = submitter.index('scontrol release "$job_id"', receipt)
        self.assertLess(held, receipt)
        self.assertLess(receipt, release)
        self.assertNotIn("scancel", submitter)
        for source in (helper, runner, sbatch):
            for forbidden in (
                "python ",
                "nvidia-smi",
                "serve_policy.py",
                "run_crfs_r05a_canary.py",
                "env.step",
                "torch",
            ):
                self.assertNotIn(forbidden, source)

    def test_fake_slurm_is_held_receipted_then_released(self) -> None:
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
            for relative in (
                "docs/decisions/0035-preregister-r05a-sampled-current-capability-gate.md",
                "scripts/hpc/lib/cgroup_v2_current_capability.sh",
                "scripts/hpc/run_r05a_cgroup_v2_current_capability.sh",
                "scripts/hpc/submit_r05a_cgroup_v2_current_capability.sh",
                "slurm/r05a_cgroup_v2_current_capability_cpu.sbatch",
            ):
                path = remote_repo / relative
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
            executable("sbatch", 'printf "sbatch %s\\n" "$*" >>"$FAKE_CALL_LOG"\nprintf "9401\\n"\n')
            executable(
                "scontrol",
                'case "$1 $2" in\n'
                '  "show node") printf "%s\\n" "NodeName=worker-1 FreeMem=9000 State=IDLE" ;;\n'
                '  "show job") if [ "${FAKE_BAD_CONTRACT:-0}" = 1 ]; then cpus=2; else cpus=1; fi; printf "%s\\n" "JobId=9401 JobState=PENDING Reason=JobHeldUser ReqNodeList=worker-1 Partition=main Account=normal QOS=normal TimeLimit=00:02:00 Requeue=0 ReqTRES=cpu=$cpus,mem=256M" ;;\n'
                '  "release 9401") printf "release 9401\\n" >>"$FAKE_CALL_LOG" ;;\n'
                '  *) exit 2 ;;\n'
                "esac\n",
            )
            commit = "d" * 40
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
                    "r05a-cgroup-v2-current-capability-20260715a",
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
            receipt = json.loads(
                (
                    experiment_root
                    / "r05a-cgroup-v2-current-capability-20260715a"
                    / "submission.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(receipt["requested_host_memory_mib"], 256)
            self.assertEqual(receipt["requested_cpus"], 1)
            self.assertEqual(receipt["requested_gpus"], 0)
            self.assertFalse(receipt["scientific_claim_allowed"])
            calls = call_log.read_text(encoding="utf-8").splitlines()
            self.assertTrue(calls[0].startswith("sbatch "))
            for expected_flag in (
                "--hold",
                "--partition=main",
                "--account=normal",
                "--qos=normal",
                "--cpus-per-task=1",
                "--mem=256M",
                "--time=00:02:00",
                "--no-requeue",
                "--nodelist=worker-1",
                "--array=0-0%1",
            ):
                self.assertIn(expected_flag, calls[0])
            self.assertEqual(calls[-1], "release 9401")

            bad_experiment_root = root / "experiments-bad"
            bad_call_log = root / "bad-calls.log"
            bad_experiment_root.mkdir()
            bad_environment = environment.copy()
            bad_environment.update(
                {"FAKE_BAD_CONTRACT": "1", "FAKE_CALL_LOG": str(bad_call_log)}
            )
            rejected = subprocess.run(
                [
                    "bash",
                    "-s",
                    "--",
                    str(remote_repo),
                    "r05a-cgroup-v2-current-capability-20260715a",
                    commit,
                    str(bad_experiment_root),
                    str(slurm_logs),
                ],
                input=remote_script,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=bad_environment,
                check=False,
            )
            self.assertNotEqual(rejected.returncode, 0)
            bad_run_root = (
                bad_experiment_root
                / "r05a-cgroup-v2-current-capability-20260715a"
            )
            self.assertFalse((bad_run_root / "submission.json").exists())
            failure = json.loads(
                (bad_run_root / "submission-failure.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(failure["slurm_array_job_id"], "9401")
            self.assertEqual(failure["control_stage"], "held_job_contract_validation")
            self.assertFalse(failure["released"])
            self.assertFalse(
                any(line == "release 9401" for line in bad_call_log.read_text().splitlines())
            )


if __name__ == "__main__":
    unittest.main()
