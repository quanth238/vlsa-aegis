from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest


ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "scripts" / "hpc" / "lib" / "cgroup_v2_full_lifetime_monitor.sh"
sys.path.insert(0, str(ROOT / "main"))
PARSER_PATH = ROOT / "main" / "crfs_oracle" / "r05a_full_lifetime_telemetry.py"
_PARSER_SPEC = importlib.util.spec_from_file_location(
    "r05a_full_lifetime_telemetry_dependency_free", PARSER_PATH
)
assert _PARSER_SPEC is not None and _PARSER_SPEC.loader is not None
_PARSER_MODULE = importlib.util.module_from_spec(_PARSER_SPEC)
_PARSER_SPEC.loader.exec_module(_PARSER_MODULE)
parse_full_lifetime_telemetry = _PARSER_MODULE.parse_full_lifetime_telemetry


class FullLifetimeCgroupMonitorTest(unittest.TestCase):
    maxDiff = None

    def _fixture(
        self,
        root: Path,
        *,
        job_id: int = 42,
        membership: str | None = None,
        maximum: str = "68719476736",
        current: str = "6000000",
        native_peak: str | None = None,
        shared_parent_peak: str | None = "999999999",
        local_only_mount: bool = False,
    ) -> dict[str, Path | str | int]:
        mount = root / "cgroup2"
        if membership is None:
            membership = (
                "/kubepods.slice/system.slice/slurmstepd.scope/"
                f"job_{job_id}/step_batch/user/task_0"
            )
        job_component = f"/job_{job_id}"
        if job_component in membership:
            job_logical = membership.split(job_component, 1)[0] + job_component
            logical = membership
            while True:
                directory = mount / logical.lstrip("/")
                directory.mkdir(parents=True, exist_ok=True)
                (directory / "memory.current").write_text(current + "\n", encoding="utf-8")
                (directory / "memory.max").write_text(maximum + "\n", encoding="utf-8")
                events = "low 0\nhigh 0\nmax 0\noom 0\noom_kill 0\n"
                (directory / "memory.events").write_text(events, encoding="utf-8")
                (directory / "memory.events.local").write_text(events, encoding="utf-8")
                if logical == job_logical:
                    job_path = directory
                    if native_peak is not None:
                        (directory / "memory.peak").write_text(
                            native_peak + "\n", encoding="utf-8"
                        )
                    break
                logical = logical.rsplit("/", 1)[0]
            shared_parent = mount / job_logical.rsplit("/", 1)[0].lstrip("/")
            shared_parent.mkdir(parents=True, exist_ok=True)
            if shared_parent_peak is not None:
                (shared_parent / "memory.peak").write_text(
                    shared_parent_peak + "\n", encoding="utf-8"
                )
        else:
            (mount / membership.lstrip("/")).mkdir(parents=True, exist_ok=True)
            job_path = mount / "not-present"

        proc_cgroup = root / "proc-self-cgroup"
        proc_cgroup.write_text(f"0::{membership}\n", encoding="utf-8")
        mountinfo = root / "proc-self-mountinfo"
        super_options = "rw,memory_localevents" if local_only_mount else "rw"
        mountinfo.write_text(
            f"36 25 0:32 / {mount} rw,nosuid,nodev,noexec,relatime "
            f"- cgroup2 cgroup {super_options}\n",
            encoding="utf-8",
        )
        osrelease = root / "osrelease"
        osrelease.write_text("5.15.0-130-generic\n", encoding="utf-8")
        return {
            "mount": mount,
            "membership": membership,
            "job_path": job_path,
            "proc_cgroup": proc_cgroup,
            "mountinfo": mountinfo,
            "osrelease": osrelease,
            "job_id": job_id,
        }

    def _paths(self, root: Path) -> dict[str, Path]:
        return {
            "trace": root / "host-memory-current.tsv",
            "ready": root / "monitor-ready.ns",
            "stop": root / "monitor-stop.request",
            "policy_launch": root / "policy-launch.ns",
            "policy_cleanup": root / "policy-cleanup.ns",
            "gpu_cleanup": root / "gpu-cleanup.ns",
            "workload_cleanup": root / "workload-cleanup.ns",
            "clock": root / "test-monotonic-clock.ns",
        }

    def _start(
        self, root: Path, fixture: dict[str, Path | str | int]
    ) -> tuple[subprocess.Popen[str], dict[str, Path]]:
        paths = self._paths(root)
        paths["clock"].write_text(
            "".join(f"{1_000_000_000 + index * 100_000_000}\n" for index in range(200)),
            encoding="utf-8",
        )
        command = [
            "bash",
            "-c",
            '. "$1"; shift; '
            '_crfs_fl_monotonic_ns() { '
            'IFS= read -r now <"$CRFS_TEST_CLOCK" || return 1; '
            'tail -n +2 "$CRFS_TEST_CLOCK" >"$CRFS_TEST_CLOCK.next" || return 1; '
            'mv "$CRFS_TEST_CLOCK.next" "$CRFS_TEST_CLOCK" || return 1; '
            'printf "%s\\n" "$now"; }; '
            'crfs_monitor_cgroup_v2_full_lifetime "$@"',
            "monitor-fixture",
            str(HELPER),
            str(fixture["proc_cgroup"]),
            str(fixture["mountinfo"]),
            str(fixture["osrelease"]),
            str(paths["trace"]),
            str(paths["ready"]),
            str(paths["stop"]),
            str(paths["policy_launch"]),
            str(paths["policy_cleanup"]),
            str(paths["gpu_cleanup"]),
            str(paths["workload_cleanup"]),
            str(fixture["job_id"]),
            "0.1",
            "500000000",
        ]
        process = subprocess.Popen(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={**os.environ, "CRFS_TEST_CLOCK": str(paths["clock"])},
        )
        return process, paths

    def _wait_ready(
        self, process: subprocess.Popen[str], ready: Path, *, timeout: float = 4.0
    ) -> int:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if ready.is_file():
                return int(ready.read_text(encoding="utf-8").strip())
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                self.fail(
                    f"monitor exited before READY: rc={process.returncode}\n"
                    f"stdout={stdout}\nstderr={stderr}"
                )
            time.sleep(0.01)
        process.terminate()
        process.wait(timeout=2)
        self.fail("monitor did not publish READY")

    def _finish(
        self,
        process: subprocess.Popen[str],
        paths: dict[str, Path],
        *,
        marker_value: str | None = None,
        mutate_before_stop=None,
    ) -> subprocess.CompletedProcess[str]:
        ready = self._wait_ready(process, paths["ready"])
        value = str(ready) if marker_value is None else marker_value
        for key in (
            "policy_launch",
            "policy_cleanup",
            "gpu_cleanup",
            "workload_cleanup",
        ):
            paths[key].write_text(value + "\n", encoding="utf-8")
        if mutate_before_stop is not None:
            mutate_before_stop()
        paths["stop"].write_text("stop\n", encoding="utf-8")
        stdout, stderr = process.communicate(timeout=5)
        return subprocess.CompletedProcess(
            args=process.args,
            returncode=process.returncode,
            stdout=stdout,
            stderr=stderr,
        )

    def _successful_trace(
        self, root: Path, *, native_peak: str | None = None
    ) -> tuple[Path, dict]:
        fixture = self._fixture(root, native_peak=native_peak)
        process, paths = self._start(root, fixture)
        completed = self._finish(process, paths)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertTrue(paths["trace"].is_file())
        self.assertEqual(list(root.glob("host-memory-current.tsv.tmp.*")), [])
        parsed = parse_full_lifetime_telemetry(paths["trace"], expected_job_id="42")
        return paths["trace"], parsed

    def test_success_is_exact_job_scoped_ready_then_atomically_sealed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace, parsed = self._successful_trace(root)
            self.assertEqual(parsed["job_boundary_path"].rsplit("/", 1)[-1], "job_42")
            self.assertEqual(parsed["sample_interval_requested_ms"], 100)
            self.assertGreaterEqual(parsed["sample_count"], 2)
            self.assertEqual(
                parsed["host_cgroup_sampled_current_high_water_bytes"], 6_000_000
            )
            self.assertIs(
                parsed["host_cgroup_sampled_current_is_lower_bound_not_peak"], True
            )
            self.assertIsNone(parsed["native_memory_peak_value_bytes"])
            text = trace.read_text(encoding="utf-8")
            self.assertIn("capability\tjob_memory_peak\tmissing\t-\n", text)
            self.assertNotIn("999999999", text)  # shared-parent peak is forbidden.
            self.assertEqual(text.splitlines()[-1], "record\tstatus\tcompleted")

    def test_exact_job_native_peak_remains_separate_from_sampled_lower_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trace, parsed = self._successful_trace(
                Path(directory), native_peak="8000000"
            )
            self.assertEqual(parsed["native_memory_peak_value_bytes"], 8_000_000)
            self.assertEqual(
                parsed["host_cgroup_sampled_current_high_water_bytes"], 6_000_000
            )
            self.assertIn(
                "capability\tjob_memory_peak\treadable_integer\t8000000",
                trace.read_text(encoding="utf-8"),
            )

    def test_missing_job_boundary_and_local_only_events_fail_before_ready(self) -> None:
        cases = (
            {"membership": "/kubepods.slice/not-the-job/step_batch/task_0"},
            {"local_only_mount": True},
        )
        for kwargs in cases:
            with self.subTest(**kwargs), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                fixture = self._fixture(root, **kwargs)
                process, paths = self._start(root, fixture)
                stdout, stderr = process.communicate(timeout=3)
                self.assertNotEqual(process.returncode, 0, (stdout, stderr))
                self.assertFalse(paths["ready"].exists())
                self.assertFalse(paths["trace"].exists())
                self.assertEqual(list(root.glob("host-memory-current.tsv.tmp.*")), [])

    def test_duplicate_longest_prefix_mount_is_rejected_before_ready(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self._fixture(root)
            mountinfo = Path(fixture["mountinfo"])
            line = mountinfo.read_text(encoding="utf-8")
            mountinfo.write_text(line + line, encoding="utf-8")
            process, paths = self._start(root, fixture)
            stdout, stderr = process.communicate(timeout=3)
            self.assertNotEqual(process.returncode, 0, (stdout, stderr))
            self.assertFalse(paths["ready"].exists())
            self.assertFalse(paths["trace"].exists())

    def test_malformed_lifecycle_marker_seals_diagnostic_but_is_not_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self._fixture(root)
            process, paths = self._start(root, fixture)
            completed = self._finish(process, paths, marker_value="01")
            self.assertEqual(completed.returncode, 3, completed.stderr)
            self.assertTrue(paths["trace"].is_file())
            with self.assertRaises(ValueError):
                parse_full_lifetime_telemetry(paths["trace"], expected_job_id="42")

    def test_changed_memory_max_and_new_oom_event_are_diagnostic_failures(self) -> None:
        mutations = {
            "memory_max": lambda job: (job / "memory.max").write_text(
                "34359738368\n", encoding="utf-8"
            ),
            "oom_event": lambda job: (job / "memory.events").write_text(
                "low 0\nhigh 0\nmax 0\noom 1\noom_kill 0\n", encoding="utf-8"
            ),
            "zero_current": lambda job: (job / "memory.current").write_text(
                "0\n", encoding="utf-8"
            ),
        }
        for name, mutation in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                fixture = self._fixture(root)
                process, paths = self._start(root, fixture)
                completed = self._finish(
                    process,
                    paths,
                    mutate_before_stop=lambda: mutation(Path(fixture["job_path"])),
                )
                self.assertEqual(completed.returncode, 3, completed.stderr)
                self.assertTrue(paths["trace"].is_file())
                with self.assertRaises(ValueError):
                    parse_full_lifetime_telemetry(
                        paths["trace"], expected_job_id="42"
                    )

    def test_boundary_timestamp_event_max_and_summary_tampering_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original, _ = self._successful_trace(root)
            source = original.read_text(encoding="utf-8")
            lines = source.splitlines()
            sample_lines = [i for i, line in enumerate(lines) if line.startswith("sample\t")]
            self.assertGreaterEqual(len(sample_lines), 2)
            first_timestamp = lines[sample_lines[0]].split("\t")[2]

            replacements: dict[str, str] = {}
            replacements["boundary"] = source.replace(
                "record\tjob_boundary_path\t/kubepods.slice/system.slice/slurmstepd.scope/job_42",
                "record\tjob_boundary_path\t/kubepods.slice/system.slice/slurmstepd.scope",
                1,
            )
            timestamp_lines = lines.copy()
            fields = timestamp_lines[sample_lines[1]].split("\t")
            fields[2] = first_timestamp
            timestamp_lines[sample_lines[1]] = "\t".join(fields)
            replacements["timestamp"] = "\n".join(timestamp_lines) + "\n"
            replacements["event"] = source.replace(
                "event\tafter\toom\t0", "event\tafter\toom\t1", 1
            )
            replacements["maximum"] = source.replace(
                "snapshot\tafter\tmemory.max\treadable_integer\t68719476736",
                "snapshot\tafter\tmemory.max\treadable_integer\t34359738368",
                1,
            )
            replacements["summary"] = source.replace(
                "summary\tsampled_memory_current_high_water_bytes\t6000000",
                "summary\tsampled_memory_current_high_water_bytes\t6000001",
                1,
            )

            for name, text in replacements.items():
                with self.subTest(name=name):
                    tampered = root / f"tampered-{name}.tsv"
                    tampered.write_text(text, encoding="utf-8")
                    with self.assertRaises(ValueError):
                        parse_full_lifetime_telemetry(
                            tampered, expected_job_id="42"
                        )

    def test_helper_is_shell_only_bounded_and_never_enumerates_cgroups(self) -> None:
        subprocess.run(["bash", "-n", str(HELPER)], check=True)
        source = HELPER.read_text(encoding="utf-8")
        self.assertIn("crfs_monitor_cgroup_v2_full_lifetime()", source)
        self.assertIn("job_$expected_job_id", source)
        self.assertIn("sleep \"$sample_delay\"", source)
        self.assertIn("mv \"$temporary\" \"$trace\"", source)
        executable_source = "\n".join(
            line for line in source.splitlines() if not line.lstrip().startswith("#")
        ).lower()
        self.assertNotIn("python", executable_source)
        self.assertNotIn("nvidia-smi", source)
        self.assertNotIn("find ", source)
        self.assertNotIn("host_cgroup_peak_bytes", source)
        self.assertNotIn("scancel", source)


if __name__ == "__main__":
    unittest.main()
