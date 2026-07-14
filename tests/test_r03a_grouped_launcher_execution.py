from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
import textwrap
from typing import Optional
import unittest


ROOT = Path(__file__).resolve().parents[1]
SUBMIT_JOB = ROOT / "scripts/hpc/submit_r03a_job.sh"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: str, *, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    if executable:
        path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _remote_body() -> str:
    value = SUBMIT_JOB.read_text(encoding="utf-8")
    opening = "<<'REMOTE'\n"
    closing = "\nREMOTE\n"
    if value.count(opening) != 1 or value.count(closing) != 1:
        raise AssertionError("R03A submitter must contain one exact remote heredoc")
    return value.split(opening, 1)[1].split(closing, 1)[0] + "\n"


class GroupedLauncherFixture:
    def __init__(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.remote_repo = self.root / "remote-repo"
        self.r02_root = self.root / "r02"
        self.output_root = self.root / "output"
        self.checkpoint = self.root / "checkpoint"
        self.fake_bin = self.root / "fake-bin"
        self.events_path = self.root / "events.jsonl"
        self.sbatch_count_path = self.root / "sbatch-count"
        self.run_id = "r03a-fake-grouped-execution"
        self.run_root = self.output_root / self.run_id
        self.case_ids = [f"case-{index:02d}" for index in range(17)]
        self.source_paths = []
        self._create_inputs()
        self._create_fake_commands()

    def cleanup(self) -> None:
        self.temporary.cleanup()

    def _create_inputs(self) -> None:
        self.output_root.mkdir(parents=True)
        self.checkpoint.mkdir(parents=True)
        (self.checkpoint / "model.safetensors").write_bytes(b"fixture")

        manifest = self.remote_repo / "manifest.jsonl"
        _write(
            manifest,
            "".join(
                json.dumps({"case_id": case_id}, sort_keys=True) + "\n"
                for case_id in self.case_ids
            ),
        )
        self.manifest = manifest
        self.config = self.remote_repo / "config.json"
        self.schema = self.remote_repo / "schema.json"
        self.decision = self.remote_repo / "decision.md"
        _write(self.config, "{}\n")
        _write(self.schema, "{}\n")
        _write(self.decision, "fixture decision\n")

        result_hashes = []
        for index, case_id in enumerate(self.case_ids):
            source = self.r02_root / case_id / "r02-paired.json"
            host = "worker-2" if index == 5 else "worker-1"
            _write(
                source,
                json.dumps({"provenance": {"host": host}}, sort_keys=True) + "\n",
            )
            self.source_paths.append(source)
            result_hashes.append({"case_id": case_id, "sha256": _sha256(source)})
        self.r03_summary = self.root / "r03-summary.json"
        _write(
            self.r03_summary,
            json.dumps({"result_hashes": result_hashes}, sort_keys=True) + "\n",
        )

        for relative in (
            "main/run_crfs_r03a.py",
            "main/summarize_r03a.py",
            "main/crfs_oracle/r03a_runner.py",
            "main/crfs_oracle/r03a_validation.py",
            "scripts/hpc/prepare_jsonschema_overlay.sh",
            "slurm/r03a_main_array.sbatch",
            "slurm/r03a_summary.sbatch",
        ):
            _write(self.remote_repo / relative, "fixture\n")
        _write(
            self.remote_repo / "scripts/hpc/run_r03a_case.sh",
            "#!/usr/bin/env bash\nexit 0\n",
            executable=True,
        )

        subprocess.run(
            ["git", "init", "-q", str(self.remote_repo)],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            ["git", "-C", str(self.remote_repo), "add", "."],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(self.remote_repo),
                "-c",
                "user.name=R03A fixture",
                "-c",
                "user.email=r03a-fixture@example.invalid",
                "commit",
                "-qm",
                "fixture",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        self.commit = subprocess.run(
            ["git", "-C", str(self.remote_repo), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    def _create_fake_commands(self) -> None:
        self.fake_bin.mkdir()
        dispatcher = self.fake_bin / "fake-command"
        dispatcher.write_text(
            "#!" + sys.executable + "\n" + textwrap.dedent(
                r'''
                import json
                import os
                from pathlib import Path
                import sys


                name = Path(sys.argv[0]).name
                args = sys.argv[1:]
                events_path = Path(os.environ["FAKE_SLURM_EVENTS"])


                def event(kind, **fields):
                    record = {"kind": kind}
                    record.update(fields)
                    with events_path.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(record, sort_keys=True) + "\n")


                if name == "sha256sum":
                    if args and args[-1].endswith("/r02-paired.json"):
                        event("source_hash", path=args[-1])
                    os.execv(os.environ["REAL_SHA256SUM"], ["sha256sum"] + args)

                if name == "jq":
                    if any(".provenance.host // empty" in value for value in args):
                        event("host_read", path=args[-1])
                    os.execv(os.environ["REAL_JQ"], ["jq"] + args)

                if name == "mkdir":
                    if args == ["-p", "/mnt/data/quanth/slurm_logs/crfs-oracle"]:
                        event("log_directory_request")
                        raise SystemExit(0)
                    os.execv(os.environ["REAL_MKDIR"], ["mkdir"] + args)

                if name == "squeue":
                    event("squeue", args=args)
                    raise SystemExit(0)

                if name == "sinfo":
                    scenario = os.environ.get("FAKE_SLURM_SCENARIO", "success")
                    worker_2_state = "drain" if scenario == "unhealthy" else "idle"
                    event("sinfo", args=args, scenario=scenario)
                    print("worker-1|idle")
                    print("worker-2|" + worker_2_state)
                    raise SystemExit(0)

                if name == "scontrol":
                    scenario = os.environ.get("FAKE_SLURM_SCENARIO", "success")
                    if len(args) >= 3 and args[:2] == ["show", "node"]:
                        node = args[2]
                        free_mem = 131071 if scenario == "low_memory" and node == "worker-2" else 262144
                        event("show_node", node=node, free_mem=free_mem)
                        print(f"NodeName={node} FreeMem={free_mem}")
                        raise SystemExit(0)
                    if len(args) >= 3 and args[:2] == ["show", "job"]:
                        job_id = args[2]
                        records = {
                            "9101": "JobId=9101 JobName=crfs-r03a-worker-1 ReqNodeList=worker-1 ArrayTaskId=0-4,6-16%1 ArrayTaskThrottle=1 JobState=PENDING Reason=JobHeldUser Priority=0",
                            "9102": "JobId=9102 JobName=crfs-r03a-worker-2 ReqNodeList=worker-2 ArrayTaskId=5%1 ArrayTaskThrottle=1 JobState=PENDING Reason=JobHeldUser Priority=0",
                            "9103": "JobId=9103 JobName=crfs-r03a-summary JobState=PENDING Partition=main ReqTRES=cpu=2,mem=16G,node=1,billing=2 Dependency=afterany:9101(unfulfilled),afterany:9102(unfulfilled)",
                        }
                        event("show_job", job_id=job_id)
                        if job_id not in records:
                            raise SystemExit(1)
                        print(records[job_id])
                        raise SystemExit(0)
                    if args and args[0] == "release":
                        run_root = Path(os.environ["FAKE_RUN_ROOT"])
                        event(
                            "release",
                            job_ids=args[1:],
                            worker_1_receipt_exists=(run_root / "worker-1-submission.json").is_file(),
                            worker_2_receipt_exists=(run_root / "worker-2-submission.json").is_file(),
                            grouped_launch_exists=(run_root / "grouped-launch.json").is_file(),
                            summary_receipt_exists=(run_root / "summary-submission.json").is_file(),
                        )
                        raise SystemExit(0)
                    raise SystemExit(2)

                if name == "sbatch":
                    count_path = Path(os.environ["FAKE_SBATCH_COUNT"])
                    count = int(count_path.read_text(encoding="utf-8")) + 1 if count_path.exists() else 1
                    count_path.write_text(str(count), encoding="utf-8")
                    event(
                        "sbatch",
                        call=count,
                        args=args,
                        expected_source_node=os.environ.get("EXPECTED_SOURCE_NODE"),
                        source_worker_1_job_id=os.environ.get("SOURCE_WORKER_1_SLURM_ARRAY_JOB_ID"),
                        source_worker_2_job_id=os.environ.get("SOURCE_WORKER_2_SLURM_ARRAY_JOB_ID"),
                    )
                    if os.environ.get("FAKE_FAIL_SBATCH_CALL") == str(count):
                        raise SystemExit(71)
                    print(9100 + count)
                    raise SystemExit(0)

                raise SystemExit("unexpected fake command: " + name)
                '''
            ).lstrip(),
            encoding="utf-8",
        )
        dispatcher.chmod(0o755)
        for name in (
            "jq",
            "mkdir",
            "sbatch",
            "scontrol",
            "sha256sum",
            "sinfo",
            "squeue",
        ):
            (self.fake_bin / name).symlink_to(dispatcher)

    def environment(self, *, scenario: str = "success", fail_call: int = 0):
        real_jq = shutil.which("jq")
        real_sha256sum = shutil.which("sha256sum")
        real_mkdir = shutil.which("mkdir")
        if not real_jq or not real_sha256sum or not real_mkdir:
            raise AssertionError("launcher execution test requires jq, sha256sum, and mkdir")
        value = dict(os.environ)
        value.update(
            {
                "PATH": str(self.fake_bin) + os.pathsep + value.get("PATH", ""),
                "FAKE_SLURM_EVENTS": str(self.events_path),
                "FAKE_SBATCH_COUNT": str(self.sbatch_count_path),
                "FAKE_SLURM_SCENARIO": scenario,
                "FAKE_RUN_ROOT": str(self.run_root),
                "REAL_JQ": real_jq,
                "REAL_SHA256SUM": real_sha256sum,
                "REAL_MKDIR": real_mkdir,
            }
        )
        if fail_call:
            value["FAKE_FAIL_SBATCH_CALL"] = str(fail_call)
        return value

    def arguments(self, *, run_id: Optional[str] = None):
        return [
            str(self.remote_repo),
            self.commit,
            "grouped_array",
            "main",
            "8",
            str(128 * 1024),
            "2",
            "17",
            run_id or self.run_id,
            str(self.manifest),
            _sha256(self.manifest),
            str(self.config),
            _sha256(self.config),
            str(self.schema),
            _sha256(self.schema),
            str(self.decision),
            _sha256(self.decision),
            str(self.r02_root),
            str(self.r03_summary),
            _sha256(self.r03_summary),
            str(self.checkpoint),
            "a" * 64,
            str(self.output_root),
            "slurm/r03a_main_array.sbatch",
        ]

    def run(
        self,
        *,
        scenario: str = "success",
        fail_call: int = 0,
        run_id: Optional[str] = None,
    ):
        return subprocess.run(
            ["/bin/bash", "-s", "--", *self.arguments(run_id=run_id)],
            input=_remote_body(),
            check=False,
            capture_output=True,
            text=True,
            env=self.environment(scenario=scenario, fail_call=fail_call),
            cwd=ROOT,
        )

    def events(self):
        if not self.events_path.exists():
            return []
        return [
            json.loads(line)
            for line in self.events_path.read_text(encoding="utf-8").splitlines()
            if line
        ]

    def clear_fake_state(self) -> None:
        for path in (self.events_path, self.sbatch_count_path):
            if path.exists():
                path.unlink()


class R03AGroupedLauncherExecutionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = GroupedLauncherFixture()

    def tearDown(self) -> None:
        self.fixture.cleanup()

    def assert_completed(self, completed: subprocess.CompletedProcess) -> None:
        self.assertEqual(
            completed.returncode,
            0,
            msg=f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
        )

    def test_success_hashes_before_hosts_and_registers_summary_before_release(self) -> None:
        completed = self.fixture.run()
        self.assert_completed(completed)
        events = self.fixture.events()

        source_hash_positions = [
            index for index, item in enumerate(events) if item["kind"] == "source_hash"
        ]
        host_read_positions = [
            index for index, item in enumerate(events) if item["kind"] == "host_read"
        ]
        self.assertEqual(len(source_hash_positions), 17)
        self.assertEqual(len(host_read_positions), 17)
        self.assertLess(max(source_hash_positions), min(host_read_positions))

        submissions = [item for item in events if item["kind"] == "sbatch"]
        self.assertEqual(len(submissions), 3)
        worker_1, worker_2, summary = submissions
        self.assertEqual(worker_1["expected_source_node"], "worker-1")
        self.assertIn("--hold", worker_1["args"])
        self.assertIn("--nodelist=worker-1", worker_1["args"])
        self.assertIn("--array=0-4,6-16%1", worker_1["args"])
        self.assertIn("--job-name=crfs-r03a-worker-1", worker_1["args"])
        self.assertEqual(worker_2["expected_source_node"], "worker-2")
        self.assertIn("--hold", worker_2["args"])
        self.assertIn("--nodelist=worker-2", worker_2["args"])
        self.assertIn("--array=5%1", worker_2["args"])
        self.assertIn("--job-name=crfs-r03a-worker-2", worker_2["args"])
        self.assertIn("--dependency=afterany:9101:9102", summary["args"])
        self.assertEqual(summary["source_worker_1_job_id"], "9101")
        self.assertEqual(summary["source_worker_2_job_id"], "9102")

        summary_position = next(
            index
            for index, item in enumerate(events)
            if item["kind"] == "sbatch" and item["call"] == 3
        )
        release_position = next(
            index for index, item in enumerate(events) if item["kind"] == "release"
        )
        self.assertLess(summary_position, release_position)
        release = events[release_position]
        self.assertEqual(release["job_ids"], ["9101", "9102"])
        self.assertTrue(release["worker_1_receipt_exists"])
        self.assertTrue(release["worker_2_receipt_exists"])
        self.assertTrue(release["grouped_launch_exists"])
        self.assertTrue(release["summary_receipt_exists"])

        summary_receipt = json.loads(
            (self.fixture.run_root / "summary-submission.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(summary_receipt["status"], "dependency_registered")
        self.assertEqual(summary_receipt["slurm_summary_job_id"], "9103")
        self.assertEqual(summary_receipt["dependency"], "afterany:9101:9102")

    def test_corrupt_final_source_hash_prevents_every_host_read_and_submission(self) -> None:
        with self.fixture.source_paths[-1].open("a", encoding="utf-8") as stream:
            stream.write("corrupt\n")
        completed = self.fixture.run()
        self.assertEqual(completed.returncode, 2)
        self.assertIn(
            "R02 source hash differs before source-node pinning",
            completed.stderr,
        )
        events = self.fixture.events()
        self.assertEqual(
            sum(item["kind"] == "source_hash" for item in events),
            17,
        )
        self.assertFalse(any(item["kind"] == "host_read" for item in events))
        self.assertFalse(any(item["kind"] == "sbatch" for item in events))
        self.assertFalse(self.fixture.run_root.exists())

    def test_second_array_submission_failure_leaves_first_held_and_receipted(self) -> None:
        completed = self.fixture.run(fail_call=2)
        self.assertNotEqual(completed.returncode, 0)
        events = self.fixture.events()
        submissions = [item for item in events if item["kind"] == "sbatch"]
        self.assertEqual(len(submissions), 2)
        self.assertIn("--hold", submissions[0]["args"])
        self.assertIn("--hold", submissions[1]["args"])
        self.assertFalse(any(item["kind"] == "release" for item in events))
        self.assertTrue((self.fixture.run_root / "launch-reservation.json").is_file())
        first_receipt = self.fixture.run_root / "worker-1-submission.json"
        self.assertTrue(first_receipt.is_file())
        self.assertEqual(
            json.loads(first_receipt.read_text(encoding="utf-8"))[
                "slurm_array_job_id"
            ],
            "9101",
        )
        self.assertFalse(
            (self.fixture.run_root / "worker-2-submission.json").exists()
        )
        self.assertFalse((self.fixture.run_root / "grouped-launch.json").exists())
        self.assertFalse(
            (self.fixture.run_root / "summary-submission.json").exists()
        )

    def test_existing_run_id_rejects_before_source_or_slurm_access(self) -> None:
        self.fixture.run_root.mkdir()
        sentinel = self.fixture.run_root / "owned"
        sentinel.write_text("preserve\n", encoding="utf-8")
        completed = self.fixture.run()
        self.assertEqual(completed.returncode, 2)
        self.assertIn("immutable R03A RUN_ID already exists", completed.stderr)
        events = self.fixture.events()
        self.assertFalse(any(item["kind"] == "source_hash" for item in events))
        self.assertFalse(any(item["kind"] == "host_read" for item in events))
        self.assertFalse(any(item["kind"] == "sbatch" for item in events))
        self.assertFalse(any(item["kind"] == "release" for item in events))
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "preserve\n")

    def test_unhealthy_or_low_memory_source_node_rejects_before_reservation(self) -> None:
        for scenario in ("unhealthy", "low_memory"):
            with self.subTest(scenario=scenario):
                run_id = f"{self.fixture.run_id}-{scenario}"
                completed = self.fixture.run(scenario=scenario, run_id=run_id)
                self.assertEqual(completed.returncode, 2)
                self.assertIn(
                    "required source node worker-2 is unhealthy or lacks requested live free memory",
                    completed.stderr,
                )
                events = self.fixture.events()
                self.assertFalse(any(item["kind"] == "sbatch" for item in events))
                self.assertFalse(any(item["kind"] == "release" for item in events))
                self.assertFalse((self.fixture.output_root / run_id).exists())
                self.fixture.clear_fake_state()


if __name__ == "__main__":
    unittest.main()
