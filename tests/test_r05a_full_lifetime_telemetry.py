from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "main" / "crfs_oracle" / "r05a_full_lifetime_telemetry.py"
SPEC = importlib.util.spec_from_file_location("r05a_full_lifetime_telemetry_tested", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
telemetry = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(telemetry)


class FullLifetimeTraceFixture:
    job_id = "42"
    membership = (
        "/kubepods.slice/system.slice/slurmstepd.scope/job_42/"
        "step_batch/user/task_0"
    )
    mount_root = "/"
    mount_point = "/sys/fs/cgroup"
    job_boundary = "/kubepods.slice/system.slice/slurmstepd.scope/job_42"
    job_scope = "/sys/fs/cgroup" + job_boundary
    start = 9_007_199_254_740_993  # deliberately greater than 2**53
    maximum = 68_719_476_736

    def __init__(self, root: Path) -> None:
        self.path = root / "host-cgroup-full-lifetime.tsv"
        self.samples = [
            (0, self.start, 4_000_000),
            (1, self.start + 100_000_000, 6_000_000),
            (2, self.start + 200_000_000, 5_000_000),
            (3, self.start + 300_000_000, 4_500_000),
        ]
        self.lifecycle = [
            (0, "monitor_ready", self.start + 1),
            (1, "policy_launch", self.start + 10),
            (2, "policy_cleanup_complete", self.start + 205_000_000),
            (3, "gpu_monitor_cleanup_complete", self.start + 215_000_000),
            (4, "workload_cleanup_complete", self.start + 220_000_000),
            (5, "monitor_stop_observed", self.start + 230_000_000),
        ]
        self.before = {
            "low": 0,
            "high": 0,
            "max": 7,
            "oom": 0,
            "oom_kill": 0,
            "oom_group_kill": 0,
        }
        self.after = dict(self.before)
        self.native_peak_state = "missing"
        self.native_peak_value = "-"
        self.write()

    @staticmethod
    def _escape(value: str) -> str:
        return value.replace("\\", "\\\\").replace("\t", "\\t").replace("\n", "\\n")

    def _summary(self) -> dict[str, str]:
        timestamps = [sample[1] for sample in self.samples]
        values = [sample[2] for sample in self.samples]
        maximum_gap = max(
            right - left for left, right in zip(timestamps, timestamps[1:])
        )
        return {
            "job_scope_path": self.job_scope,
            "job_memory_max_before_bytes": str(self.maximum),
            "job_memory_max_after_bytes": str(self.maximum),
            "sample_count_observed": str(len(self.samples)),
            "sample_interval_requested_seconds": "0.1",
            "first_sample_monotonic_ns": str(timestamps[0]),
            "last_sample_monotonic_ns": str(timestamps[-1]),
            "sample_max_gap_ns": str(maximum_gap),
            "sampled_memory_current_high_water_bytes": str(max(values)),
            "memory_events_max_delta": str(self.after["max"] - self.before["max"]),
            "memory_events_oom_delta": str(self.after["oom"] - self.before["oom"]),
            "memory_events_oom_kill_delta": str(
                self.after["oom_kill"] - self.before["oom_kill"]
            ),
            "native_memory_peak_state": self.native_peak_state,
            "native_memory_peak_value": self.native_peak_value,
            "first_sample_no_later_than_policy_launch": "true",
            "last_sample_no_earlier_than_workload_cleanup": "true",
            "host_cgroup_sampled_current_is_lower_bound_not_peak": "true",
            "outcome": "full_lifetime_sampled_current_supported",
        }

    def lines(self) -> list[str]:
        records = [
            ("schema_version", "1.0"),
            ("artifact_role", telemetry.TRACE_ARTIFACT_ROLE),
            ("slurm_array_job_id", self.job_id),
            ("slurm_array_task_id", "0"),
            ("proc_cgroup_line", f"0::{self.membership}"),
            ("membership_path", self.membership),
            (
                "selected_mountinfo_line",
                "4663 4516 0:29 / /sys/fs/cgroup rw,relatime - cgroup2 none rw",
            ),
            ("mount_root", self.mount_root),
            ("mount_point", self.mount_point),
            ("memory_events_hierarchy_state", "hierarchical"),
            ("membership_relative_to_mount_root", self.membership),
            ("job_boundary_path", self.job_boundary),
            ("kernel_osrelease", "5.15.0-130-generic"),
        ]
        lines = [f"record\t{key}\t{self._escape(value)}" for key, value in records]
        logical = self.membership
        index = 0
        while True:
            kind = (
                "task"
                if index == 0
                else "job"
                if logical == self.job_boundary
                else "step"
                if logical.endswith("/step_batch")
                else "intermediate"
            )
            lines.append(
                f"scope\t{index}\t{kind}\t{logical}\t{self.mount_point + logical}\tsearchable"
            )
            lines.extend(
                (
                    f"metric\t{index}\tmemory.current\treadable_integer\t4000000",
                    f"metric\t{index}\tmemory.max\treadable_integer\t{self.maximum}",
                    f"metric\t{index}\tmemory.events\treadable_flat_keys\t-",
                    f"metric\t{index}\tmemory.events.local\treadable_flat_keys\t-",
                )
            )
            for source in ("memory.events", "memory.events.local"):
                for key in self.before:
                    lines.append(f"counter\t{index}\t{source}\t{key}\t{self.before[key]}")
            if logical == self.job_boundary:
                break
            logical = logical.rsplit("/", 1)[0]
            index += 1
        lines.append(
            f"snapshot\tbefore\tmemory.max\treadable_integer\t{self.maximum}"
        )
        for key, value in self.before.items():
            lines.append(f"event\tbefore\t{key}\t{value}")
        for key, value in self.before.items():
            lines.append(f"local_event\tbefore\t{key}\t{value}")
        lines.extend(
            f"sample\t{index}\t{timestamp}\t{value}"
            for index, timestamp, value in self.samples
        )
        lines.extend(
            f"lifecycle\t{index}\t{name}\t{timestamp}"
            for index, name, timestamp in self.lifecycle
        )
        lines.append(
            f"snapshot\tafter\tmemory.max\treadable_integer\t{self.maximum}"
        )
        for key, value in self.after.items():
            lines.append(f"event\tafter\t{key}\t{value}")
        for key, value in self.after.items():
            lines.append(f"local_event\tafter\t{key}\t{value}")
        lines.append(
            f"capability\tjob_memory_peak\t{self.native_peak_state}\t{self.native_peak_value}"
        )
        lines.extend(f"summary\t{key}\t{value}" for key, value in self._summary().items())
        lines.append("record\tstatus\tcompleted")
        return lines

    def write(self, lines: list[str] | None = None, *, terminal_newline: bool = True) -> None:
        value = "\n".join(lines if lines is not None else self.lines())
        if terminal_newline:
            value += "\n"
        self.path.write_text(value, encoding="utf-8")

    def mutate_line(self, prefix: str, replacement: str) -> None:
        lines = self.path.read_text(encoding="utf-8").splitlines()
        matching = [index for index, line in enumerate(lines) if line.startswith(prefix)]
        if len(matching) != 1:
            raise AssertionError(f"fixture prefix is not unique: {prefix!r}")
        lines[matching[0]] = replacement
        self.write(lines)


class FullLifetimeTelemetryTest(unittest.TestCase):
    def test_valid_trace_preserves_exact_integers_above_two_to_the_53(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = FullLifetimeTraceFixture(Path(directory))
            summary = telemetry.parse_full_lifetime_telemetry(
                fixture.path, expected_job_id=fixture.job_id
            )
            self.assertEqual(summary["first_sample_monotonic_ns"], fixture.start)
            self.assertEqual(summary["last_sample_monotonic_ns"], fixture.start + 300_000_000)
            self.assertEqual(summary["maximum_adjacent_gap_ns"], 100_000_000)
            self.assertEqual(
                summary["host_cgroup_sampled_current_high_water_bytes"], 6_000_000
            )
            self.assertTrue(
                summary["host_cgroup_sampled_current_is_lower_bound_not_peak"]
            )
            self.assertEqual(
                summary["raw_trace_sha256"],
                hashlib.sha256(fixture.path.read_bytes()).hexdigest(),
            )

    def test_indices_timestamps_and_registered_gap_are_recomputed(self) -> None:
        mutations = (
            (
                "sample\t2\t",
                lambda fixture: f"sample\t1\t{fixture.start + 200_000_000}\t5000000",
                "duplicates sample 1",
            ),
            (
                "sample\t2\t",
                lambda fixture: f"sample\t2\t{fixture.start + 50_000_000}\t5000000",
                "not strictly increasing",
            ),
            (
                "sample\t1\t",
                lambda fixture: f"sample\t1\t0{fixture.start + 100_000_000}\t6000000",
                "canonical unsigned integer",
            ),
        )
        for prefix, replacement, error in mutations:
            with self.subTest(error=error), tempfile.TemporaryDirectory() as directory:
                fixture = FullLifetimeTraceFixture(Path(directory))
                fixture.mutate_line(prefix, replacement(fixture))
                with self.assertRaisesRegex(ValueError, error):
                    telemetry.parse_full_lifetime_telemetry(
                        fixture.path, expected_job_id=fixture.job_id
                    )
        with tempfile.TemporaryDirectory() as directory:
            fixture = FullLifetimeTraceFixture(Path(directory))
            fixture.samples = [
                (0, fixture.start, 4_000_000),
                (1, fixture.start + 600_000_001, 6_000_000),
                (2, fixture.start + 700_000_001, 5_000_000),
                (3, fixture.start + 800_000_001, 4_500_000),
            ]
            fixture.write()
            with self.assertRaisesRegex(ValueError, "maximum adjacent gap"):
                telemetry.parse_full_lifetime_telemetry(
                    fixture.path, expected_job_id=fixture.job_id
                )

    def test_raw_sample_and_summary_cannot_be_tampered_independently(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = FullLifetimeTraceFixture(Path(directory))
            fixture.mutate_line(
                "sample\t1\t",
                f"sample\t1\t{fixture.start + 100_000_000}\t7000000",
            )
            with self.assertRaisesRegex(ValueError, "summary differs"):
                telemetry.parse_full_lifetime_telemetry(
                    fixture.path, expected_job_id=fixture.job_id
                )
        with tempfile.TemporaryDirectory() as directory:
            fixture = FullLifetimeTraceFixture(Path(directory))
            fixture.mutate_line(
                "summary\tsampled_memory_current_high_water_bytes\t",
                "summary\tsampled_memory_current_high_water_bytes\t7000000",
            )
            with self.assertRaisesRegex(ValueError, "summary differs"):
                telemetry.parse_full_lifetime_telemetry(
                    fixture.path, expected_job_id=fixture.job_id
                )

    def test_lifecycle_order_and_both_window_boundaries_are_required(self) -> None:
        mutations = (
            (
                "lifecycle\t1\tpolicy_launch\t",
                lambda f: f"lifecycle\t1\twrong_phase\t{f.start + 10}",
                "marker order",
            ),
            (
                "lifecycle\t1\tpolicy_launch\t",
                lambda f: f"lifecycle\t1\tpolicy_launch\t{f.start - 1}",
                "out of order|first sample",
            ),
            (
                "lifecycle\t4\tworkload_cleanup_complete\t",
                lambda f: (
                    "lifecycle\t4\tworkload_cleanup_complete\t"
                    f"{f.start + 400_000_000}"
                ),
                "out of order|final sample",
            ),
        )
        for prefix, replacement, error in mutations:
            with self.subTest(error=error), tempfile.TemporaryDirectory() as directory:
                fixture = FullLifetimeTraceFixture(Path(directory))
                fixture.mutate_line(prefix, replacement(fixture))
                with self.assertRaisesRegex(ValueError, error):
                    telemetry.parse_full_lifetime_telemetry(
                        fixture.path, expected_job_id=fixture.job_id
                    )
        with tempfile.TemporaryDirectory() as directory:
            fixture = FullLifetimeTraceFixture(Path(directory))
            lines = fixture.path.read_text(encoding="utf-8").splitlines()
            left = next(
                index for index, line in enumerate(lines) if line.startswith("lifecycle\t2\t")
            )
            right = next(
                index for index, line in enumerate(lines) if line.startswith("lifecycle\t3\t")
            )
            lines[left], lines[right] = lines[right], lines[left]
            fixture.write(lines)
            with self.assertRaisesRegex(ValueError, "lifecycle indices|rows are out of order"):
                telemetry.parse_full_lifetime_telemetry(
                    fixture.path, expected_job_id=fixture.job_id
                )

    def test_raw_sample_row_reordering_cannot_be_hidden_by_valid_indices(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = FullLifetimeTraceFixture(Path(directory))
            lines = fixture.path.read_text(encoding="utf-8").splitlines()
            left = next(
                index for index, line in enumerate(lines) if line.startswith("sample\t1\t")
            )
            right = next(
                index for index, line in enumerate(lines) if line.startswith("sample\t2\t")
            )
            lines[left], lines[right] = lines[right], lines[left]
            fixture.write(lines)
            with self.assertRaisesRegex(ValueError, "sample indices"):
                telemetry.parse_full_lifetime_telemetry(
                    fixture.path, expected_job_id=fixture.job_id
                )

    def test_each_after_event_row_must_follow_the_final_sample(self) -> None:
        for prefix, error in (
            ("event\tafter\tmax\t", "post-workload memory.events row"),
            (
                "local_event\tafter\tmax\t",
                "post-workload memory.events.local row",
            ),
        ):
            with self.subTest(prefix=prefix), tempfile.TemporaryDirectory() as directory:
                fixture = FullLifetimeTraceFixture(Path(directory))
                lines = fixture.path.read_text(encoding="utf-8").splitlines()
                moved_index = next(
                    index for index, line in enumerate(lines) if line.startswith(prefix)
                )
                moved = lines.pop(moved_index)
                first_sample = next(
                    index for index, line in enumerate(lines) if line.startswith("sample\t0\t")
                )
                lines.insert(first_sample, moved)
                fixture.write(lines)
                with self.assertRaisesRegex(ValueError, error):
                    telemetry.parse_full_lifetime_telemetry(
                        fixture.path, expected_job_id=fixture.job_id
                    )

    def test_memory_max_is_positive_finite_unchanged_and_bounds_samples(self) -> None:
        mutations = (
            (
                "snapshot\tbefore\tmemory.max\t",
                "snapshot\tbefore\tmemory.max\treadable_unlimited\tmax",
                "not a finite integer",
            ),
            (
                "snapshot\tbefore\tmemory.max\t",
                "snapshot\tbefore\tmemory.max\treadable_integer\t0",
                "must be positive",
            ),
            (
                "snapshot\tafter\tmemory.max\t",
                f"snapshot\tafter\tmemory.max\treadable_integer\t{68_719_476_735}",
                "changed",
            ),
            (
                "sample\t1\t",
                lambda f: f"sample\t1\t{f.start + 100_000_000}\t{f.maximum + 1}",
                "outside the job-scope hard limit",
            ),
        )
        for prefix, replacement, error in mutations:
            with self.subTest(error=error), tempfile.TemporaryDirectory() as directory:
                fixture = FullLifetimeTraceFixture(Path(directory))
                value = replacement(fixture) if callable(replacement) else replacement
                fixture.mutate_line(prefix, value)
                with self.assertRaisesRegex(ValueError, error):
                    telemetry.parse_full_lifetime_telemetry(
                        fixture.path, expected_job_id=fixture.job_id
                    )

    def test_per_scope_metrics_counters_and_positive_samples_are_strict(self) -> None:
        cases = (
            (
                "metric\t0\tmemory.current\t",
                "metric\t0\tmemory.current_forged\treadable_integer\t4000000",
                "metric identities",
            ),
            (
                "metric\t3\tmemory.events.local\t",
                "metric\t3\tmemory.events.local\tmissing\t-",
                "unavailable memory.events.local",
            ),
            (
                "counter\t3\tmemory.events.local\toom_kill\t",
                "counter\t3\tmemory.events.local\toom_kill\t1",
                "local diagnostic differs",
            ),
            (
                "sample\t2\t",
                lambda fixture: f"sample\t2\t{fixture.start + 200_000_000}\t0",
                "sample 2 bytes must be positive",
            ),
        )
        for prefix, replacement, error in cases:
            with self.subTest(error=error), tempfile.TemporaryDirectory() as directory:
                fixture = FullLifetimeTraceFixture(Path(directory))
                value = replacement(fixture) if callable(replacement) else replacement
                fixture.mutate_line(prefix, value)
                with self.assertRaisesRegex(ValueError, error):
                    telemetry.parse_full_lifetime_telemetry(
                        fixture.path, expected_job_id=fixture.job_id
                    )
        with tempfile.TemporaryDirectory() as directory:
            fixture = FullLifetimeTraceFixture(Path(directory))
            lines = [
                line
                for line in fixture.path.read_text(encoding="utf-8").splitlines()
                if line != "counter\t3\tmemory.events\thigh\t0"
            ]
            fixture.write(lines)
            with self.assertRaisesRegex(ValueError, "counters are incomplete"):
                telemetry.parse_full_lifetime_telemetry(
                    fixture.path, expected_job_id=fixture.job_id
                )

    def test_hierarchical_events_are_complete_nondecreasing_and_zero_delta(self) -> None:
        cases = (
            ("event\tafter\tmax\t", "event\tafter\tmax\t8", "new max or OOM"),
            ("event\tafter\tmax\t", "event\tafter\tmax\t6", "counter max decreased"),
            ("event\tafter\toom\t", "event\tafter\toom\t1", "new max or OOM"),
            (
                "event\tafter\toom_kill\t",
                "event\tafter\toom_kill\t1",
                "new max or OOM",
            ),
        )
        for prefix, replacement, error in cases:
            with self.subTest(error=error), tempfile.TemporaryDirectory() as directory:
                fixture = FullLifetimeTraceFixture(Path(directory))
                fixture.mutate_line(prefix, replacement)
                with self.assertRaisesRegex(ValueError, error):
                    telemetry.parse_full_lifetime_telemetry(
                        fixture.path, expected_job_id=fixture.job_id
                    )
        with tempfile.TemporaryDirectory() as directory:
            fixture = FullLifetimeTraceFixture(Path(directory))
            lines = [
                line
                for line in fixture.path.read_text(encoding="utf-8").splitlines()
                if line != "event\tafter\thigh\t0"
            ]
            fixture.write(lines)
            with self.assertRaisesRegex(ValueError, "key sets differ|incomplete"):
                telemetry.parse_full_lifetime_telemetry(
                    fixture.path, expected_job_id=fixture.job_id
                )

    def test_exact_job_boundary_mount_mapping_and_hierarchy_are_recomputed(self) -> None:
        cases = (
            (
                "record\tjob_boundary_path\t",
                "record\tjob_boundary_path\t/kubepods.slice/system.slice/slurmstepd.scope",
                "job boundary differs",
            ),
            (
                "record\tmount_point\t",
                "record\tmount_point\t/sys/fs/wrong",
                "mountinfo mapping differs",
            ),
            (
                "record\tselected_mountinfo_line\t",
                (
                    "record\tselected_mountinfo_line\t"
                    "4663 4516 0:29 / /sys/fs/cgroup rw,relatime - cgroup2 none "
                    "rw,memory_localevents"
                ),
                "local-only",
            ),
            (
                "scope\t3\tjob\t",
                (
                    "scope\t3\tjob\t/kubepods.slice/system.slice/slurmstepd.scope"
                    "\t/sys/fs/cgroup/kubepods.slice/system.slice/slurmstepd.scope"
                    "\tsearchable"
                ),
                "task-to-job chain",
            ),
        )
        for prefix, replacement, error in cases:
            with self.subTest(error=error), tempfile.TemporaryDirectory() as directory:
                fixture = FullLifetimeTraceFixture(Path(directory))
                fixture.mutate_line(prefix, replacement)
                with self.assertRaisesRegex(ValueError, error):
                    telemetry.parse_full_lifetime_telemetry(
                        fixture.path, expected_job_id=fixture.job_id
                    )
        with tempfile.TemporaryDirectory() as directory:
            fixture = FullLifetimeTraceFixture(Path(directory))
            with self.assertRaisesRegex(ValueError, "job.*id|job boundary"):
                telemetry.parse_full_lifetime_telemetry(
                    fixture.path, expected_job_id="43"
                )

    def test_native_peak_is_separate_and_if_present_must_be_positive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = FullLifetimeTraceFixture(Path(directory))
            fixture.native_peak_state = "readable_integer"
            fixture.native_peak_value = "8000000"
            fixture.write()
            summary = telemetry.parse_full_lifetime_telemetry(
                fixture.path, expected_job_id=fixture.job_id
            )
            self.assertEqual(summary["native_memory_peak_value_bytes"], 8_000_000)
            self.assertEqual(
                summary["host_cgroup_sampled_current_high_water_bytes"], 6_000_000
            )
            self.assertTrue(
                summary["host_cgroup_sampled_current_is_lower_bound_not_peak"]
            )
        for value in ("0", "08", "sampled"):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as directory:
                fixture = FullLifetimeTraceFixture(Path(directory))
                fixture.mutate_line(
                    "capability\tjob_memory_peak\t",
                    f"capability\tjob_memory_peak\treadable_integer\t{value}",
                )
                with self.assertRaises(ValueError):
                    telemetry.parse_full_lifetime_telemetry(
                        fixture.path, expected_job_id=fixture.job_id
                    )

    def test_truncation_duplicate_unknown_and_noncanonical_files_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = FullLifetimeTraceFixture(Path(directory))
            fixture.write(fixture.lines()[:-1])
            with self.assertRaisesRegex(ValueError, "record keys|terminal status"):
                telemetry.parse_full_lifetime_telemetry(
                    fixture.path, expected_job_id=fixture.job_id
                )
        with tempfile.TemporaryDirectory() as directory:
            fixture = FullLifetimeTraceFixture(Path(directory))
            lines = fixture.lines()
            lines.insert(-1, "summary\toutcome\tfull_lifetime_sampled_current_supported")
            fixture.write(lines)
            with self.assertRaisesRegex(ValueError, "duplicates summary"):
                telemetry.parse_full_lifetime_telemetry(
                    fixture.path, expected_job_id=fixture.job_id
                )
        with tempfile.TemporaryDirectory() as directory:
            fixture = FullLifetimeTraceFixture(Path(directory))
            lines = fixture.lines()
            lines.insert(-1, "forged\trow")
            fixture.write(lines)
            with self.assertRaisesRegex(ValueError, "unknown shape"):
                telemetry.parse_full_lifetime_telemetry(
                    fixture.path, expected_job_id=fixture.job_id
                )
        with tempfile.TemporaryDirectory() as directory:
            fixture = FullLifetimeTraceFixture(Path(directory))
            fixture.write(terminal_newline=False)
            summary, errors = telemetry.validate_full_lifetime_telemetry(
                fixture.path, expected_job_id=fixture.job_id
            )
            self.assertIsNone(summary)
            self.assertEqual(len(errors), 1)
            self.assertIn("canonical newline-delimited", errors[0])


if __name__ == "__main__":
    unittest.main()
