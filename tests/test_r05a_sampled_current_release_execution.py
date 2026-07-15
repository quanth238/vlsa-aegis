from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[1]
SUBMITTER = ROOT / "scripts/hpc/submit_r05a_sampled_current_canary.sh"
CASE_ID = "crfs-1069f29a8d76463a"
RUN_ID = "r05a-sampled-current-release-fixture"
RELEASE_DECISION = (
    "docs/decisions/0037-require-exact-single-canary-release-identity.md"
)
APPARATUS_CONFIG = (
    "configs/experiments/r05a_sampled_current_canary_apparatus.json"
)
ALLOWED_RELEASE_DIFF_PATHS = [
    APPARATUS_CONFIG,
    RELEASE_DECISION,
]
RESOURCES = {
    "partition": "main",
    "account": "normal",
    "qos": "normal",
    "gpus": 1,
    "cpus_per_task": 8,
    "host_memory_mib": 65536,
    "time_limit": "02:00:00",
    "array": "0-0%1",
    "requeue": False,
    "validator_partition": "main",
    "validator_account": "normal",
    "validator_qos": "normal",
    "validator_cpus": 2,
    "validator_host_memory_mib": 8192,
    "validator_time_limit": "00:15:00",
    "validator_gpus": 0,
    "validator_dependency": "afterany",
}


def _release_decision_appendix(execution_release: dict) -> str:
    resources = json.dumps(
        execution_release["resources"], sort_keys=True, separators=(",", ":")
    )
    return (
        "\n## Exact execution release\n\n"
        "Execution authorization: one preregistered IFT-00A canary submission only.\n\n"
        f"- Accepted implementation commit: `{execution_release['accepted_implementation_commit']}`.\n"
        f"- Immutable run ID: `{execution_release['run_id']}`.\n"
        f"- Source host: `{execution_release['source_host']}`.\n"
        f"- Resources (canonical JSON): `{resources}`.\n"
        "- Single submission: `true`.\n"
        "- Automatic resubmission: `false`.\n"
        "- Automatic next experiment: `false`.\n"
        "- Simulator efficacy claim authorized: `false`.\n"
        "- Probe or MLP training authorized: `false`.\n\n"
        "This appendix authorizes only the frozen one-case mechanism canary. "
        "It does not authorize IFT-01, solver tuning, a simulator efficacy claim, "
        "label collection, probe training, or MLP training.\n"
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: str, *, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    if executable:
        path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _write_json(path: Path, value: object) -> None:
    _write(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _remote_body() -> str:
    value = SUBMITTER.read_text(encoding="utf-8")
    opening = "<<'REMOTE'\n"
    closing = "\nREMOTE\n"
    if value.count(opening) != 1 or value.count(closing) != 1:
        raise AssertionError("R05A submitter must contain one exact remote heredoc")
    return value.split(opening, 1)[1].split(closing, 1)[0] + "\n"


def _bound_repository_paths() -> list[str]:
    match = re.search(r"^bound_paths=\(\n(.*?)\n\)$", _remote_body(), re.MULTILINE | re.DOTALL)
    if match is None:
        raise AssertionError("R05A remote heredoc must declare bound_paths exactly once")
    paths = [line.strip() for line in match.group(1).splitlines() if line.strip()]
    if not paths or any(any(character.isspace() for character in path) for path in paths):
        raise AssertionError("R05A bound_paths must be nonempty plain relative paths")
    return paths


class SampledCurrentReleaseFixture:
    def __init__(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.remote_repo = self.root / "remote-repo"
        self.experiment_root = self.root / "experiments"
        self.r02_root = self.root / "r02"
        self.checkpoint_dir = self.root / "checkpoint"
        self.fake_bin = self.root / "fake-bin"
        self.events_path = self.root / "events.jsonl"
        self.sbatch_count_path = self.root / "sbatch-count"
        self.run_id = RUN_ID
        self.run_root = self.experiment_root / self.run_id
        self.bound_paths = _bound_repository_paths()
        self._create_inputs_and_release_history()
        self._create_fake_commands()

    def cleanup(self) -> None:
        self.temporary.cleanup()

    def _git(self, *arguments: str) -> str:
        return subprocess.run(
            ["git", "-C", str(self.remote_repo), *arguments],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    def _commit(self, message: str) -> str:
        self._git("add", ".")
        subprocess.run(
            [
                "git",
                "-C",
                str(self.remote_repo),
                "-c",
                "user.name=R05A release fixture",
                "-c",
                "user.email=r05a-release-fixture@example.invalid",
                "commit",
                "-qm",
                message,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return self._git("rev-parse", "HEAD")

    def _create_inputs_and_release_history(self) -> None:
        self.experiment_root.mkdir(parents=True)
        self.checkpoint_dir.mkdir(parents=True)
        self.r02_source = self.r02_root / CASE_ID / "r02-paired.json"
        _write(self.r02_source, '{"fixture":"r02-source"}\n')
        self.checkpoint = self.checkpoint_dir / "model.safetensors"
        self.checkpoint.write_bytes(b"r05a-checkpoint-fixture")

        for relative in self.bound_paths:
            _write(self.remote_repo / relative, f"fixture: {relative}\n")

        self.manifest = (
            self.remote_repo / "manifests/r05a_inverse_flow_teacher_smoke.jsonl"
        )
        self.scientific_config = (
            self.remote_repo / "configs/experiments/r05a_inverse_flow_canary.json"
        )
        self.apparatus_config = self.remote_repo / APPARATUS_CONFIG
        self.envelope_schema = (
            self.remote_repo
            / "schemas/r05a-sampled-current-canary-envelope.schema.json"
        )
        self.adr0036 = (
            self.remote_repo
            / "docs/decisions/0036-preregister-r05a-full-lifetime-sampled-current-canary.md"
        )
        self.adr0037 = self.remote_repo / RELEASE_DECISION

        _write(self.manifest, json.dumps({"case_id": CASE_ID}) + "\n")
        _write_json(self.scientific_config, {"fixture": "scientific-config"})
        _write_json(self.envelope_schema, {"fixture": "envelope-schema"})
        _write(self.adr0036, "fixture ADR-0036\n")
        parent_decision = "fixture ADR-0037 enforcement only\n"
        _write(self.adr0037, parent_decision)

        parent_config = {
            "schema_version": "1.0",
            "ready_to_run": False,
            "blocked_on": [
                "separate_h100_execution_release_decision_not_accepted",
                "immutable_h100_execution_identity_not_selected",
            ],
            "envelope_schema_sha256": _sha256(self.envelope_schema),
            "resource_contract": dict(RESOURCES, source_host="worker-1"),
            "fixture_non_release_content": {
                "science_is_unchanged": True,
                "probe_training_authorized": False,
            },
        }
        _write_json(self.apparatus_config, parent_config)

        self.remote_repo.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "init", "-q", str(self.remote_repo)],
            check=True,
            capture_output=True,
            text=True,
        )
        self.implementation_commit = self._commit("unreleased implementation")

        release_config = dict(parent_config)
        release_config["ready_to_run"] = True
        release_config["blocked_on"] = []
        execution_release = {
            "schema_version": "1.0",
            "artifact_role": "r05a_single_canary_execution_release",
            "decision_artifact": RELEASE_DECISION,
            "accepted_implementation_commit": self.implementation_commit,
            "run_id": self.run_id,
            "single_submission": True,
            "source_host": "worker-1",
            "resources": RESOURCES,
            "release_only_parent_required": True,
            "allowed_release_diff_paths": ALLOWED_RELEASE_DIFF_PATHS,
            "automatic_resubmission_allowed": False,
            "automatic_next_experiment_allowed": False,
        }
        release_config["execution_release"] = execution_release
        _write_json(self.apparatus_config, release_config)
        _write(
            self.adr0037,
            parent_decision + _release_decision_appendix(execution_release),
        )
        self.release_commit = self._commit("exact single-canary release")
        self._git(
            "update-ref",
            "refs/remotes/origin/agent/crfs-oracle-harness",
            self.release_commit,
        )

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
                run_root = Path(os.environ["FAKE_RUN_ROOT"])


                def event(kind, **fields):
                    record = {"kind": kind}
                    record.update(fields)
                    with events_path.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(record, sort_keys=True) + "\n")


                if name == "mkdir":
                    event("mkdir", args=args)
                    os.execv(os.environ["REAL_MKDIR"], ["mkdir"] + args)

                if name == "squeue":
                    event("squeue", args=args)
                    raise SystemExit(0)

                if name == "sbatch":
                    count_path = Path(os.environ["FAKE_SBATCH_COUNT"])
                    count = int(count_path.read_text(encoding="utf-8")) + 1 if count_path.exists() else 1
                    count_path.write_text(str(count), encoding="utf-8")
                    event(
                        "sbatch",
                        call=count,
                        args=args,
                        run_id=os.environ.get("RUN_ID"),
                        expected_git_commit=os.environ.get("EXPECTED_GIT_COMMIT"),
                        source_job_id=os.environ.get("SOURCE_JOB_ID"),
                        source_contract=os.environ.get("SOURCE_CONTRACT"),
                        expected_source_contract_sha256=os.environ.get("EXPECTED_SOURCE_CONTRACT_SHA256"),
                        reservation_exists=(run_root / "launch-reservation.json").is_file(),
                        held_receipt_exists=(run_root / "held-gpu-submission.json").is_file(),
                        source_contract_exists=(run_root / "source-contract.json").is_file(),
                        submission_exists=(run_root / "submission.json").is_file(),
                    )
                    if os.environ.get("FAKE_FAIL_SBATCH_CALL") == str(count):
                        raise SystemExit(71)
                    print(9200 + count)
                    raise SystemExit(0)

                if name == "scontrol":
                    scenario = os.environ.get("FAKE_SLURM_SCENARIO", "success")
                    if len(args) >= 3 and args[:2] == ["show", "node"]:
                        free_mem = 65535 if scenario == "low_memory" else 131072
                        event("show_node", node=args[2], free_mem=free_mem)
                        print(
                            "NodeName=worker-1 State=IDLE "
                            f"FreeMem={free_mem} "
                            "Gres=gpu:nvidia_h100_80gb_hbm3:8"
                        )
                        raise SystemExit(0)
                    if len(args) >= 3 and args[:2] == ["show", "job"]:
                        job_id = args[2]
                        gpu_state = "RUNNING" if scenario == "wrong_gpu_state" else "PENDING"
                        gpu_reason = "Resources" if scenario == "wrong_gpu_reason" else "JobHeldUser"
                        gpu_node = "worker-2" if scenario == "wrong_gpu_node" else "worker-1"
                        records = {
                            "9201": (
                                f"JobId=9201 JobState={gpu_state} "
                                f"Reason={gpu_reason} Dependency=(null) "
                                f"ReqNodeList={gpu_node} Partition=main Account=normal "
                                "QOS=normal TimeLimit=02:00:00 Requeue=0 "
                                "ReqTRES=cpu=8,mem=64G,node=1,gres/gpu=1"
                            ),
                            "9202": (
                                "JobId=9202 JobState=PENDING "
                                "Partition=main Account=normal QOS=normal "
                                "TimeLimit=00:15:00 "
                                "Requeue=0 ReqTRES=cpu=2,mem=8G,node=1 "
                                "Dependency=afterany:9201(unfulfilled)"
                            ),
                        }
                        event("show_job", job_id=job_id)
                        if job_id not in records:
                            raise SystemExit(1)
                        print(records[job_id])
                        raise SystemExit(0)
                    if args and args[0] == "release":
                        event(
                            "release",
                            job_ids=args[1:],
                            reservation_exists=(run_root / "launch-reservation.json").is_file(),
                            held_receipt_exists=(run_root / "held-gpu-submission.json").is_file(),
                            source_contract_exists=(run_root / "source-contract.json").is_file(),
                            submission_exists=(run_root / "submission.json").is_file(),
                        )
                        raise SystemExit(0)
                    raise SystemExit(2)

                raise SystemExit("unexpected fake command: " + name)
                '''
            ).lstrip(),
            encoding="utf-8",
        )
        dispatcher.chmod(0o755)
        for name in ("mkdir", "sbatch", "scontrol", "squeue"):
            (self.fake_bin / name).symlink_to(dispatcher)

    def environment(self, *, scenario: str = "success", fail_call: int = 0) -> dict[str, str]:
        real_mkdir = shutil.which("mkdir")
        for command in ("bash", "git", "jq", "sha256sum"):
            if shutil.which(command) is None:
                raise AssertionError(f"launcher execution test requires system {command}")
        if real_mkdir is None:
            raise AssertionError("launcher execution test requires system mkdir")
        value = dict(os.environ)
        value.update(
            {
                "PATH": str(self.fake_bin) + os.pathsep + value.get("PATH", ""),
                "FAKE_SLURM_EVENTS": str(self.events_path),
                "FAKE_SBATCH_COUNT": str(self.sbatch_count_path),
                "FAKE_SLURM_SCENARIO": scenario,
                "FAKE_RUN_ROOT": str(self.run_root),
                "REAL_MKDIR": real_mkdir,
            }
        )
        if fail_call:
            value["FAKE_FAIL_SBATCH_CALL"] = str(fail_call)
        return value

    def arguments(self, *, run_id: str | None = None) -> list[str]:
        return [
            str(self.remote_repo),
            run_id or self.run_id,
            self.release_commit,
            str(self.experiment_root),
            str(self.r02_root),
            str(self.checkpoint_dir),
            _sha256(self.manifest),
            _sha256(self.scientific_config),
            _sha256(self.r02_source),
            _sha256(self.checkpoint),
            _sha256(self.apparatus_config),
            _sha256(self.envelope_schema),
            _sha256(self.adr0036),
            _sha256(self.adr0037),
            self.implementation_commit,
        ]

    def run(
        self,
        *,
        scenario: str = "success",
        fail_call: int = 0,
        run_id: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["/bin/bash", "-s", "--", *self.arguments(run_id=run_id)],
            input=_remote_body(),
            check=False,
            capture_output=True,
            text=True,
            env=self.environment(scenario=scenario, fail_call=fail_call),
            cwd=ROOT,
        )

    def events(self) -> list[dict[str, object]]:
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

    def drift_origin_to_implementation(self) -> None:
        self._git(
            "update-ref",
            "refs/remotes/origin/agent/crfs-oracle-harness",
            self.implementation_commit,
        )

    def drift_head_to_implementation(self) -> None:
        self._git("checkout", "-q", "--detach", self.implementation_commit)


class R05ASampledCurrentReleaseExecutionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = SampledCurrentReleaseFixture()

    def tearDown(self) -> None:
        self.fixture.cleanup()

    def assert_completed(self, completed: subprocess.CompletedProcess[str]) -> None:
        self.assertEqual(
            completed.returncode,
            0,
            msg=f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
        )

    def assert_no_mutating_boundary(self) -> None:
        events = self.fixture.events()
        self.assertFalse(any(item["kind"] == "mkdir" for item in events))
        self.assertFalse(any(item["kind"] == "sbatch" for item in events))
        self.assertFalse(any(item["kind"] == "release" for item in events))

    def test_success_is_one_held_gpu_then_cpu_afterany_receipts_then_one_release(self) -> None:
        completed = self.fixture.run()
        self.assert_completed(completed)
        events = self.fixture.events()

        submissions = [item for item in events if item["kind"] == "sbatch"]
        self.assertEqual(len(submissions), 2)
        gpu, cpu = submissions
        self.assertEqual(gpu["call"], 1)
        self.assertIn("--hold", gpu["args"])
        self.assertIn("--nodelist=worker-1", gpu["args"])
        self.assertIn("--array=0-0%1", gpu["args"])
        self.assertIn("--gres=gpu:1", gpu["args"])
        self.assertIn("--cpus-per-task=8", gpu["args"])
        self.assertIn("--mem=64G", gpu["args"])
        self.assertEqual(gpu["run_id"], self.fixture.run_id)
        self.assertEqual(gpu["expected_git_commit"], self.fixture.release_commit)
        self.assertTrue(gpu["reservation_exists"])
        self.assertFalse(gpu["held_receipt_exists"])

        self.assertEqual(cpu["call"], 2)
        self.assertNotIn("--hold", cpu["args"])
        self.assertIn("--dependency=afterany:9201", cpu["args"])
        self.assertIn("--cpus-per-task=2", cpu["args"])
        self.assertIn("--mem=8G", cpu["args"])
        self.assertFalse(any(str(arg).startswith("--gres") for arg in cpu["args"]))
        self.assertEqual(cpu["source_job_id"], "9201")
        self.assertTrue(cpu["held_receipt_exists"])
        self.assertTrue(cpu["source_contract_exists"])
        self.assertFalse(cpu["submission_exists"])

        release_positions = [
            index for index, item in enumerate(events) if item["kind"] == "release"
        ]
        self.assertEqual(len(release_positions), 1)
        second_submission_position = next(
            index
            for index, item in enumerate(events)
            if item["kind"] == "sbatch" and item["call"] == 2
        )
        self.assertLess(second_submission_position, release_positions[0])
        release = events[release_positions[0]]
        self.assertEqual(release["job_ids"], ["9201"])
        self.assertTrue(release["reservation_exists"])
        self.assertTrue(release["held_receipt_exists"])
        self.assertTrue(release["source_contract_exists"])
        self.assertTrue(release["submission_exists"])

        reservation = json.loads(
            (self.fixture.run_root / "launch-reservation.json").read_text(
                encoding="utf-8"
            )
        )
        held = json.loads(
            (self.fixture.run_root / "held-gpu-submission.json").read_text(
                encoding="utf-8"
            )
        )
        source_contract_path = self.fixture.run_root / "source-contract.json"
        source_contract = json.loads(source_contract_path.read_text(encoding="utf-8"))
        submission = json.loads(
            (self.fixture.run_root / "submission.json").read_text(encoding="utf-8")
        )

        self.assertEqual(reservation["run_id"], self.fixture.run_id)
        self.assertEqual(reservation["git_commit"], self.fixture.release_commit)
        self.assertEqual(
            reservation["accepted_implementation_commit"],
            self.fixture.implementation_commit,
        )
        self.assertTrue(reservation["single_submission"])
        self.assertEqual(reservation["observed_free_mem_mib"], 131072)
        self.assertEqual(held["gpu_slurm_array_job_id"], "9201")
        self.assertEqual(held["exact_gpu_task_id"], "9201_0")
        self.assertFalse(held["released_at_receipt_time"])
        self.assertEqual(source_contract["run_id"], self.fixture.run_id)
        self.assertEqual(source_contract["git_commit"], self.fixture.release_commit)
        self.assertEqual(
            set(source_contract["repository_file_sha256"]),
            set(self.fixture.bound_paths),
        )
        self.assertEqual(
            source_contract["held_gpu_submission_sha256"],
            _sha256(self.fixture.run_root / "held-gpu-submission.json"),
        )
        self.assertEqual(submission["gpu_slurm_array_job_id"], "9201")
        self.assertEqual(submission["cpu_afterany_job_id"], "9202")
        self.assertEqual(submission["dependency"], "afterany:9201")
        self.assertEqual(
            submission["source_contract_sha256"], _sha256(source_contract_path)
        )
        self.assertIn("submitted_gpu_job_id=9201", completed.stdout)
        self.assertIn("submitted_cpu_publisher_job_id=9202", completed.stdout)

    def test_existing_run_root_rejects_before_mkdir_or_sbatch(self) -> None:
        self.fixture.run_root.mkdir()
        sentinel = self.fixture.run_root / "preserve"
        sentinel.write_text("do not overwrite\n", encoding="utf-8")
        completed = self.fixture.run()
        self.assertEqual(completed.returncode, 2)
        self.assertIn("immutable sampled-current run id is already used", completed.stderr)
        self.assert_no_mutating_boundary()
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "do not overwrite\n")

    def test_wrong_held_gpu_state_reason_or_node_fails_before_receipts(self) -> None:
        cases = {
            "wrong_gpu_state": "JobState=PENDING",
            "wrong_gpu_reason": "Reason=JobHeldUser",
            "wrong_gpu_node": "ReqNodeList=worker-1",
        }
        for scenario, rejected_field in cases.items():
            with self.subTest(scenario=scenario):
                completed = self.fixture.run(scenario=scenario)
                self.assertEqual(completed.returncode, 2)
                self.assertIn(
                    f"held GPU job field changed: {rejected_field}",
                    completed.stderr,
                )
                events = self.fixture.events()
                submissions = [item for item in events if item["kind"] == "sbatch"]
                self.assertEqual(len(submissions), 1)
                self.assertIn("--hold", submissions[0]["args"])
                self.assertFalse(any(item["kind"] == "release" for item in events))
                self.assertTrue(
                    (self.fixture.run_root / "launch-reservation.json").is_file()
                )
                self.assertFalse(
                    (self.fixture.run_root / "held-gpu-submission.json").exists()
                )
                self.assertFalse(
                    (self.fixture.run_root / "source-contract.json").exists()
                )
                self.assertFalse((self.fixture.run_root / "submission.json").exists())

                self.fixture.cleanup()
                self.fixture = SampledCurrentReleaseFixture()

    def test_low_memory_rejects_before_mkdir_or_sbatch(self) -> None:
        completed = self.fixture.run(scenario="low_memory")
        self.assertEqual(completed.returncode, 2)
        self.assertIn("worker-1 FreeMem=65535MiB < 65536MiB", completed.stderr)
        self.assert_no_mutating_boundary()
        self.assertFalse(self.fixture.run_root.exists())

    def test_registered_run_identity_drift_rejects_before_mkdir_or_sbatch(self) -> None:
        completed = self.fixture.run(run_id=self.fixture.run_id + "-drift")
        self.assertEqual(completed.returncode, 2)
        self.assertIn("remote registered run ID mismatch", completed.stderr)
        self.assert_no_mutating_boundary()
        self.assertFalse(
            (self.fixture.experiment_root / (self.fixture.run_id + "-drift")).exists()
        )

    def test_remote_head_or_origin_drift_rejects_before_mkdir_or_sbatch(self) -> None:
        self.fixture.drift_origin_to_implementation()
        completed = self.fixture.run()
        self.assertEqual(completed.returncode, 2)
        self.assertIn("remote origin release ref mismatch", completed.stderr)
        self.assert_no_mutating_boundary()
        self.assertFalse(self.fixture.run_root.exists())

        self.fixture.cleanup()
        self.fixture = SampledCurrentReleaseFixture()
        self.fixture.drift_head_to_implementation()
        completed = self.fixture.run()
        self.assertEqual(completed.returncode, 2)
        self.assertIn("remote commit mismatch", completed.stderr)
        self.assert_no_mutating_boundary()
        self.assertFalse(self.fixture.run_root.exists())

    def test_partial_cpu_submission_consumes_identity_and_cannot_be_reused(self) -> None:
        completed = self.fixture.run(fail_call=2)
        self.assertNotEqual(completed.returncode, 0)
        events = self.fixture.events()
        submissions = [item for item in events if item["kind"] == "sbatch"]
        self.assertEqual(len(submissions), 2)
        self.assertIn("--hold", submissions[0]["args"])
        self.assertIn("--dependency=afterany:9201", submissions[1]["args"])
        self.assertFalse(any(item["kind"] == "release" for item in events))
        self.assertTrue((self.fixture.run_root / "launch-reservation.json").is_file())
        self.assertTrue((self.fixture.run_root / "held-gpu-submission.json").is_file())
        self.assertTrue((self.fixture.run_root / "source-contract.json").is_file())
        self.assertFalse((self.fixture.run_root / "submission.json").exists())

        self.fixture.clear_fake_state()
        repeated = self.fixture.run()
        self.assertEqual(repeated.returncode, 2)
        self.assertIn("immutable sampled-current run id is already used", repeated.stderr)
        self.assert_no_mutating_boundary()
        self.assertTrue((self.fixture.run_root / "held-gpu-submission.json").is_file())
        self.assertTrue((self.fixture.run_root / "source-contract.json").is_file())


if __name__ == "__main__":
    unittest.main()
