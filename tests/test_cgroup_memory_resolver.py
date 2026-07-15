from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "scripts" / "hpc" / "lib" / "cgroup_memory.sh"
RUNNER = ROOT / "scripts" / "hpc" / "run_r05a_canary.sh"


def _mountinfo_escape(value: str) -> str:
    return (
        value.replace("\\", "\\134")
        .replace(" ", "\\040")
        .replace("\t", "\\011")
        .replace("\n", "\\012")
    )


def _mountinfo_line(
    mount_root: str,
    mount_point: Path,
    *,
    version: int,
    identifier: int,
) -> str:
    filesystem = "cgroup2" if version == 2 else "cgroup"
    super_options = "rw" if version == 2 else "rw,memory"
    return (
        f"{identifier} 1 0:{identifier} {_mountinfo_escape(mount_root)} "
        f"{_mountinfo_escape(str(mount_point))} rw,nosuid,nodev,noexec - "
        f"{filesystem} cgroup {super_options}\n"
    )


def _diagnostic(path: Path) -> dict[str, str]:
    def unescape(value: str) -> str:
        decoded: list[str] = []
        index = 0
        while index < len(value):
            if value[index] != "\\" or index + 1 >= len(value):
                decoded.append(value[index])
                index += 1
                continue
            marker = value[index + 1]
            decoded.append({"\\": "\\", "t": "\t", "n": "\n"}.get(marker, marker))
            index += 2
        return "".join(decoded)

    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, value = line.split("\t", 1)
        values[key] = unescape(value)
    return values


class CgroupMemoryResolverTest(unittest.TestCase):
    def _run(self, cgroup: Path, mountinfo: Path, diagnostic: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                "bash",
                "-c",
                '. "$1"; crfs_read_live_cgroup_memory_peak "$2" "$3" "$4"',
                "cgroup-memory-test",
                str(HELPER),
                str(cgroup),
                str(mountinfo),
                str(diagnostic),
            ],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def test_cgroup_v2_root_mount_resolves_live_positive_peak(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mount = root / "cgroup2"
            membership = "/slurm/uid_1073/job_27726/step_batch"
            peak = mount / membership.lstrip("/") / "memory.peak"
            peak.parent.mkdir(parents=True)
            peak.write_text("123456789\n", encoding="utf-8")
            cgroup = root / "cgroup"
            cgroup.write_text(f"0::{membership}\n", encoding="utf-8")
            mountinfo = root / "mountinfo"
            mountinfo.write_text(
                _mountinfo_line("/", mount, version=2, identifier=31),
                encoding="utf-8",
            )
            diagnostic = root / "diagnostic.tsv"

            completed = self._run(cgroup, mountinfo, diagnostic)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(completed.stdout, "123456789\n")
            record = _diagnostic(diagnostic)
            self.assertEqual(record["status"], "measured")
            self.assertEqual(record["cgroup_version"], "2")
            self.assertEqual(record["membership_path"], membership)
            self.assertEqual(record["mount_root"], "/")
            self.assertEqual(record["peak_file"], str(peak))
            self.assertEqual(record["peak_bytes"], "123456789")

    def test_cgroup_v2_subtree_mount_and_escaped_mountpoint_are_mapped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mount = root / "mounted cgroup\\subtree"
            membership = "/slurm/uid_1073/job_27726/step_batch"
            peak = mount / "uid_1073/job_27726/step_batch/memory.peak"
            peak.parent.mkdir(parents=True)
            peak.write_text("987654321\n", encoding="utf-8")
            cgroup = root / "cgroup"
            cgroup.write_text(f"0::{membership}\n", encoding="utf-8")
            mountinfo = root / "mountinfo"
            mountinfo.write_text(
                _mountinfo_line("/slurm", mount, version=2, identifier=32),
                encoding="utf-8",
            )
            diagnostic = root / "diagnostic.tsv"

            completed = self._run(cgroup, mountinfo, diagnostic)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(completed.stdout, "987654321\n")
            record = _diagnostic(diagnostic)
            self.assertEqual(record["mount_root"], "/slurm")
            self.assertEqual(record["mount_point"], str(mount))
            self.assertEqual(
                record["membership_relative_to_mount_root"],
                "/uid_1073/job_27726/step_batch",
            )

    def test_root_mountpoint_produces_canonical_single_slash_peak_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            membership_directory = root / "root-mounted-cgroup"
            peak = membership_directory / "memory.peak"
            peak.parent.mkdir(parents=True)
            peak.write_text("314159\n", encoding="utf-8")
            membership = str(membership_directory)
            cgroup = root / "cgroup"
            cgroup.write_text(f"0::{membership}\n", encoding="utf-8")
            mountinfo = root / "mountinfo"
            mountinfo.write_text(
                _mountinfo_line("/", Path("/"), version=2, identifier=39),
                encoding="utf-8",
            )
            diagnostic = root / "diagnostic.tsv"

            completed = self._run(cgroup, mountinfo, diagnostic)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(completed.stdout, "314159\n")
            record = _diagnostic(diagnostic)
            self.assertEqual(record["mount_point"], "/")
            self.assertEqual(record["peak_file"], str(peak))
            self.assertFalse(record["peak_file"].startswith("//"))

    def test_explicit_v1_memory_membership_wins_in_hybrid_layout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            unified_mount = root / "unified"
            memory_mount = root / "memory"
            unified_peak = unified_mount / "unified/job/memory.peak"
            memory_peak = memory_mount / "legacy/job/memory.max_usage_in_bytes"
            unified_peak.parent.mkdir(parents=True)
            memory_peak.parent.mkdir(parents=True)
            unified_peak.write_text("111\n", encoding="utf-8")
            memory_peak.write_text("222\n", encoding="utf-8")
            cgroup = root / "cgroup"
            cgroup.write_text("0::/unified/job\n5:cpu,memory:/legacy/job\n", encoding="utf-8")
            mountinfo = root / "mountinfo"
            mountinfo.write_text(
                _mountinfo_line("/", unified_mount, version=2, identifier=33)
                + _mountinfo_line("/", memory_mount, version=1, identifier=34),
                encoding="utf-8",
            )
            diagnostic = root / "diagnostic.tsv"

            completed = self._run(cgroup, mountinfo, diagnostic)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(completed.stdout, "222\n")
            self.assertEqual(_diagnostic(diagnostic)["cgroup_version"], "1")

    def test_longest_matching_readable_mount_root_is_selected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            broad = root / "broad"
            specific = root / "specific"
            membership = "/slurm/job/step"
            broad_peak = broad / "slurm/job/step/memory.peak"
            specific_peak = specific / "job/step/memory.peak"
            broad_peak.parent.mkdir(parents=True)
            specific_peak.parent.mkdir(parents=True)
            broad_peak.write_text("100\n", encoding="utf-8")
            specific_peak.write_text("200\n", encoding="utf-8")
            cgroup = root / "cgroup"
            cgroup.write_text(f"0::{membership}\n", encoding="utf-8")
            mountinfo = root / "mountinfo"
            mountinfo.write_text(
                _mountinfo_line("/", broad, version=2, identifier=35)
                + _mountinfo_line("/slurm", specific, version=2, identifier=36),
                encoding="utf-8",
            )
            diagnostic = root / "diagnostic.tsv"

            completed = self._run(cgroup, mountinfo, diagnostic)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(completed.stdout, "200\n")
            self.assertEqual(_diagnostic(diagnostic)["mount_root"], "/slurm")

    def test_unreadable_peak_fails_closed_with_path_diagnostic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mount = root / "cgroup2"
            mount.mkdir()
            membership = "/slurm/job/step"
            cgroup = root / "cgroup"
            cgroup.write_text(f"0::{membership}\n", encoding="utf-8")
            mountinfo = root / "mountinfo"
            mountinfo.write_text(
                _mountinfo_line("/slurm", mount, version=2, identifier=37),
                encoding="utf-8",
            )
            diagnostic = root / "diagnostic.tsv"

            completed = self._run(cgroup, mountinfo, diagnostic)
            self.assertNotEqual(completed.returncode, 0)
            self.assertEqual(completed.stdout, "")
            record = _diagnostic(diagnostic)
            self.assertEqual(record["status"], "unavailable")
            self.assertEqual(record["reason"], "peak_file_unreadable")
            self.assertEqual(
                record["peak_file"],
                str(mount / "job/step/memory.peak"),
            )

    def test_uncovered_membership_preserves_selected_hierarchy_in_diagnostic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mount = root / "unrelated-cgroup2"
            mount.mkdir()
            membership = "/slurm/job/step"
            cgroup = root / "cgroup"
            cgroup.write_text(f"0::{membership}\n", encoding="utf-8")
            mountinfo = root / "mountinfo"
            mountinfo.write_text(
                _mountinfo_line("/different-tree", mount, version=2, identifier=40),
                encoding="utf-8",
            )
            diagnostic = root / "diagnostic.tsv"

            completed = self._run(cgroup, mountinfo, diagnostic)
            self.assertNotEqual(completed.returncode, 0)
            self.assertEqual(completed.stdout, "")
            record = _diagnostic(diagnostic)
            self.assertEqual(record["status"], "unavailable")
            self.assertEqual(record["reason"], "peak_file_unreadable")
            self.assertEqual(record["cgroup_version"], "2")
            self.assertEqual(record["membership_path"], membership)
            self.assertEqual(record["mount_root"], "")
            self.assertEqual(record["peak_file"], "")

    def test_zero_or_nonnumeric_peak_is_never_accepted(self) -> None:
        for value in ("0\n", "max\n", "-1\n"):
            with self.subTest(value=value.strip()), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                mount = root / "cgroup2"
                peak = mount / "job/memory.peak"
                peak.parent.mkdir(parents=True)
                peak.write_text(value, encoding="utf-8")
                cgroup = root / "cgroup"
                cgroup.write_text("0::/job\n", encoding="utf-8")
                mountinfo = root / "mountinfo"
                mountinfo.write_text(
                    _mountinfo_line("/", mount, version=2, identifier=38),
                    encoding="utf-8",
                )
                diagnostic = root / "diagnostic.tsv"

                completed = self._run(cgroup, mountinfo, diagnostic)
                self.assertNotEqual(completed.returncode, 0)
                record = _diagnostic(diagnostic)
                self.assertEqual(record["status"], "invalid")
                self.assertEqual(record["reason"], "peak_not_positive_integer")

    def test_runner_sets_accurate_stage_and_records_resolver_provenance(self) -> None:
        helper_source = HELPER.read_text(encoding="utf-8")
        runner_source = RUNNER.read_text(encoding="utf-8")
        subprocess.run(["bash", "-n", str(HELPER)], check=True)
        subprocess.run(["bash", "-n", str(RUNNER)], check=True)
        self.assertIn("/proc/self/mountinfo", helper_source)
        self.assertIn("memory.peak", helper_source)
        self.assertIn("memory.max_usage_in_bytes", helper_source)
        stage = runner_source.index("FAILURE_STAGE=host_cgroup_memory_peak")
        read = runner_source.index("crfs_read_live_cgroup_memory_peak")
        self.assertLess(stage, read)
        self.assertIn("host-cgroup-memory.tsv", runner_source)
        self.assertIn("host_cgroup_diagnostic_sha256", runner_source)
        self.assertNotIn('"/sys/fs/cgroup${v2_path}/memory.peak"', runner_source)


if __name__ == "__main__":
    unittest.main()
