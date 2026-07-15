from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
VALIDATOR_PATH = ROOT / "main" / "validate_crfs_r05a_apparatus_regression.py"
SPEC = importlib.util.spec_from_file_location("r05a_apparatus_validator_tested", VALIDATOR_PATH)
assert SPEC is not None and SPEC.loader is not None
validator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validator)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ApparatusFixture:
    def __init__(self, root: Path) -> None:
        # macOS exposes /var through /private/var; build artifact strings from
        # the canonical root so the production validator can reject aliases.
        self.root = root.resolve()
        root = self.root
        self.repo = root / "repo"
        self.run_root = root / "experiments" / validator.EXPECTED_RUN_ID
        self.result = self.run_root / ".results.pending.json"
        self.final_result = self.run_root / "results.json"
        self.submission = self.run_root / "submission.json"
        self.held = self.run_root / "held-submission.json"
        self.log = self.run_root / "allocation-focused-tests.log"
        self.cgroup = self.run_root / "host-cgroup-memory.tsv"
        self.commit = "a" * 40
        self.job_id = "9201_0"
        self.array_job_id = "9201"
        self.timestamp = "2026-07-15T07:00:00+00:00"
        self.run_root.mkdir(parents=True)
        self.repo.mkdir()
        self._write_sources()
        self._write_log()
        self._write_cgroup(123456789)
        self._write_held()
        self._write_submission()
        self._write_result()

    def _write_sources(self) -> None:
        registry = {
            "schema_version": "1.0",
            "suites": [
                {"pattern": name, "expected_tests": count}
                for name, count in validator.EXPECTED_COUNTS.items()
            ],
        }
        for relative in validator.SOURCE_HASH_PATHS.values():
            path = self.repo / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            if relative == validator.REGISTRY_RELATIVE_PATH:
                path.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
            else:
                path.write_text(f"fixture for {relative}\n", encoding="utf-8")

    def _write_log(self) -> None:
        lines: list[str] = []
        for suite, count in validator.EXPECTED_COUNTS.items():
            lines.extend(
                [
                    f"verified_test_suite={suite} expected={count} observed={count} "
                    "skips=0 status=passed",
                    f"[{suite}] test_contract ({suite}.Fixture.test_contract) ... ok",
                    f"[{suite}] Ran {count} tests in 0.001s",
                    f"[{suite}] OK",
                ]
            )
        self.log.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _diagnostic_record(self, peak: int) -> dict[str, object]:
        membership = "/slurm/uid_1073/job_9201/step_batch"
        return {
            "schema_version": "1.0",
            "artifact_role": "r05a_live_slurm_cgroup_memory_peak_diagnostic",
            "status": "measured",
            "reason": "live_positive_peak",
            "cgroup_version": "2",
            "membership_path": membership,
            "mount_root": "/",
            "mount_point": "/sys/fs/cgroup",
            "membership_relative_to_mount_root": membership,
            "peak_file": f"/sys/fs/cgroup{membership}/memory.peak",
            "peak_bytes": peak,
            "proc_cgroup_file": "/proc/self/cgroup",
            "mountinfo_file": "/proc/self/mountinfo",
        }

    def _write_cgroup(self, peak: int) -> None:
        record = self._diagnostic_record(peak)
        self.cgroup.write_text(
            "".join(
                f"{key}\t{record[key]}\n" for key in validator.CGROUP_KEYS
            ),
            encoding="utf-8",
        )

    def _write_held(self) -> None:
        value = {
            "schema_version": "1.0",
            "artifact_role": validator.EXPECTED_HELD_ROLE,
            "status": "sbatch_returned_held_job_id",
            "run_id": validator.EXPECTED_RUN_ID,
            "git_commit": self.commit,
            "slurm_array_job_id": self.array_job_id,
            "slurm_array_task_id": 0,
            "source_node": validator.EXPECTED_NODE,
            "partition": validator.EXPECTED_PARTITION,
            "account": validator.EXPECTED_ACCOUNT,
            "qos": validator.EXPECTED_QOS,
            "time_limit": validator.EXPECTED_TIME_LIMIT,
            "requeue": False,
            "requested_cpus": validator.EXPECTED_CPUS,
            "requested_host_memory_mib": validator.EXPECTED_MEMORY_MIB,
            "requested_gpus": 0,
            "released_at_receipt_time": False,
            "scientific_claim_allowed": False,
            "timestamp_utc": self.timestamp,
        }
        self.held.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")

    def submission_value(self) -> dict[str, object]:
        value: dict[str, object] = {
            "schema_version": "1.0",
            "artifact_role": validator.EXPECTED_SUBMISSION_ROLE,
            "status": "reviewed_job_held_and_receipted",
            "run_id": validator.EXPECTED_RUN_ID,
            "git_commit": self.commit,
            "slurm_array_job_id": self.array_job_id,
            "slurm_array_task_id": 0,
            "source_node": validator.EXPECTED_NODE,
            "partition": validator.EXPECTED_PARTITION,
            "account": validator.EXPECTED_ACCOUNT,
            "qos": validator.EXPECTED_QOS,
            "time_limit": validator.EXPECTED_TIME_LIMIT,
            "requeue": False,
            "array": "0-0%1",
            "requested_cpus": validator.EXPECTED_CPUS,
            "requested_host_memory_mib": validator.EXPECTED_MEMORY_MIB,
            "requested_gpus": 0,
            "held_submission_sha256": _sha(self.held),
            "expected_result": str(self.final_result),
            "scientific_claim_allowed": False,
            "timestamp_utc": self.timestamp,
        }
        for key, relative in validator.SOURCE_HASH_PATHS.items():
            value[key] = _sha(self.repo / relative)
        return value

    def _write_submission(self, value: dict[str, object] | None = None) -> None:
        if value is None:
            value = self.submission_value()
        self.submission.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")

    def result_value(self) -> dict[str, object]:
        cgroup_record = self._diagnostic_record(123456789)
        return {
            "schema_version": "1.0",
            "artifact_role": validator.EXPECTED_RESULT_ROLE,
            "status": "passed",
            "run_id": validator.EXPECTED_RUN_ID,
            "timestamp_utc": self.timestamp,
            "git_commit": self.commit,
            "git_dirty": False,
            "submission_receipt_path": str(self.submission),
            "submission_receipt_sha256": _sha(self.submission),
            "host": validator.EXPECTED_NODE,
            "slurm_job_id": self.job_id,
            "slurm_array_job_id": self.array_job_id,
            "slurm_array_task_id": 0,
            "requested_cpus": validator.EXPECTED_CPUS,
            "requested_host_memory_mib": validator.EXPECTED_MEMORY_MIB,
            "partition": validator.EXPECTED_PARTITION,
            "account": validator.EXPECTED_ACCOUNT,
            "qos": validator.EXPECTED_QOS,
            "time_limit": validator.EXPECTED_TIME_LIMIT,
            "requeue": False,
            "gpu_allocated": False,
            "allocation_tests": {
                "registry_path": validator.REGISTRY_RELATIVE_PATH,
                "registry_sha256": _sha(
                    self.repo / validator.REGISTRY_RELATIVE_PATH
                ),
                "expected_counts": dict(validator.EXPECTED_COUNTS),
                "observed_counts": dict(validator.EXPECTED_COUNTS),
                "zero_skips": True,
                "log_path": str(self.log),
                "log_sha256": _sha(self.log),
            },
            "host_cgroup_memory": {
                "diagnostic_path": str(self.cgroup),
                "diagnostic_sha256": _sha(self.cgroup),
                **cgroup_record,
            },
            "scientific_claim_allowed": False,
            "simulator_efficacy_evaluated": False,
            "checkpoint_loaded": False,
            "policy_server_started": False,
            "real_pi05_teacher_searches_executed": 0,
            "real_pi05_teacher_observations_produced": 0,
            "synthetic_unit_test_solver_calls_excluded_from_guard": True,
            "simulator_steps_executed": 0,
            "probe_training_authorized": False,
            "retry_c_authorized_by_this_artifact_alone": False,
        }

    def _write_result(self, value: dict[str, object] | None = None) -> None:
        if value is None:
            value = self.result_value()
        self.result.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")

    def validate(self, *, expected_submission_sha: str | None = None) -> list[str]:
        return validator.validate_r05a_apparatus_result(
            self.result,
            self.submission,
            expected_run_root=self.run_root,
            expected_result_path=self.final_result,
            repo_root=self.repo,
            expected_git_commit=self.commit,
            expected_job_id=self.job_id,
            expected_array_job_id=self.array_job_id,
            expected_task_id=0,
            expected_submission_sha256=expected_submission_sha or _sha(self.submission),
        )


class R05AApparatusResultValidatorTest(unittest.TestCase):
    def test_valid_fixture_and_cli_pass(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = ApparatusFixture(Path(directory))
            self.assertEqual(fixture.validate(), [])
            completed = subprocess.run(
                [
                    sys.executable,
                    str(VALIDATOR_PATH),
                    "--result",
                    str(fixture.result),
                    "--submission",
                    str(fixture.submission),
                    "--expected-run-root",
                    str(fixture.run_root),
                    "--expected-result-path",
                    str(fixture.final_result),
                    "--repo-root",
                    str(fixture.repo),
                    "--expected-git-commit",
                    fixture.commit,
                    "--expected-job-id",
                    fixture.job_id,
                    "--expected-array-job-id",
                    fixture.array_job_id,
                    "--expected-task-id",
                    "0",
                    "--expected-submission-sha256",
                    _sha(fixture.submission),
                ],
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            self.assertEqual(completed.stdout, "validation_error_count=0\n")

    def test_exact_result_identity_resources_and_no_science_claims_are_enforced(self) -> None:
        mutations = {
            "run_id": "wrong-run",
            "git_commit": "b" * 40,
            "host": "worker-2",
            "slurm_job_id": "9999_0",
            "slurm_array_task_id": 1,
            "requested_cpus": 4,
            "requested_host_memory_mib": 16384,
            "gpu_allocated": True,
            "real_pi05_teacher_searches_executed": 1,
            "real_pi05_teacher_observations_produced": 1,
            "simulator_steps_executed": 1,
            "probe_training_authorized": True,
        }
        with tempfile.TemporaryDirectory() as directory:
            fixture = ApparatusFixture(Path(directory))
            baseline = fixture.result_value()
            for key, changed in mutations.items():
                with self.subTest(key=key):
                    value = copy.deepcopy(baseline)
                    value[key] = changed
                    fixture._write_result(value)
                    self.assertTrue(fixture.validate(), key)

    def test_unknown_result_or_receipt_fields_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = ApparatusFixture(Path(directory))
            result = fixture.result_value()
            result["unreviewed"] = True
            fixture._write_result(result)
            self.assertTrue(any("keys changed" in item for item in fixture.validate()))

            fixture._write_result(fixture.result_value())
            submission = fixture.submission_value()
            submission["unreviewed"] = True
            fixture._write_submission(submission)
            result = fixture.result_value()
            fixture._write_result(result)
            self.assertTrue(
                any(
                    "submission receipt keys changed" in item
                    for item in fixture.validate(expected_submission_sha=_sha(fixture.submission))
                )
            )

    def test_externally_bound_submission_sha_and_receipt_fields_reject_coupled_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = ApparatusFixture(Path(directory))
            original_sha = _sha(fixture.submission)
            receipt = fixture.submission_value()
            receipt["partition"] = "debug"
            fixture._write_submission(receipt)
            result = fixture.result_value()
            fixture._write_result(result)
            errors = fixture.validate(expected_submission_sha=original_sha)
            self.assertTrue(any("externally bound SHA-256" in item for item in errors))

            errors = fixture.validate(expected_submission_sha=_sha(fixture.submission))
            self.assertTrue(any("submission receipt.partition changed" in item for item in errors))

    def test_bound_source_and_held_receipt_tampering_fail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = ApparatusFixture(Path(directory))
            helper = fixture.repo / validator.SOURCE_HASH_PATHS[
                "allocation_test_helper_sha256"
            ]
            helper.write_text("tampered helper\n", encoding="utf-8")
            self.assertTrue(any("bound source" in item for item in fixture.validate()))

        with tempfile.TemporaryDirectory() as directory:
            fixture = ApparatusFixture(Path(directory))
            held = json.loads(fixture.held.read_text(encoding="utf-8"))
            held["source_node"] = "worker-2"
            fixture.held.write_text(json.dumps(held, sort_keys=True) + "\n", encoding="utf-8")
            receipt = fixture.submission_value()
            fixture._write_submission(receipt)
            fixture._write_result(fixture.result_value())
            self.assertTrue(
                any("held submission receipt.source_node changed" in item for item in fixture.validate())
            )

    def test_allocation_registry_count_log_hash_and_skip_evidence_are_recomputed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = ApparatusFixture(Path(directory))
            result = fixture.result_value()
            result["allocation_tests"]["observed_counts"]["test_r05a_canary.py"] = 11
            fixture._write_result(result)
            self.assertTrue(any("observed_counts changed" in item for item in fixture.validate()))

        with tempfile.TemporaryDirectory() as directory:
            fixture = ApparatusFixture(Path(directory))
            fixture.log.write_text(
                fixture.log.read_text(encoding="utf-8")
                + "[test_r05a_canary.py] OK (skipped=1)\n",
                encoding="utf-8",
            )
            result = fixture.result_value()
            fixture._write_result(result)
            self.assertTrue(any("skipped or failed" in item for item in fixture.validate()))

        with tempfile.TemporaryDirectory() as directory:
            fixture = ApparatusFixture(Path(directory))
            registry_path = fixture.repo / validator.REGISTRY_RELATIVE_PATH
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
            registry["suites"][0]["expected_tests"] = 18
            registry_path.write_text(json.dumps(registry) + "\n", encoding="utf-8")
            receipt = fixture.submission_value()
            fixture._write_submission(receipt)
            result = fixture.result_value()
            fixture._write_result(result)
            self.assertTrue(any("17/8/10/12" in item for item in fixture.validate()))

    def test_cgroup_hash_mapping_and_exact_8gib_ceiling_are_recomputed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = ApparatusFixture(Path(directory))
            fixture._write_cgroup(validator.MAX_CGROUP_PEAK_BYTES + 1)
            result = fixture.result_value()
            high = fixture._diagnostic_record(validator.MAX_CGROUP_PEAK_BYTES + 1)
            result["host_cgroup_memory"] = {
                "diagnostic_path": str(fixture.cgroup),
                "diagnostic_sha256": _sha(fixture.cgroup),
                **high,
            }
            fixture._write_result(result)
            self.assertTrue(any("exceeds the exact 8 GiB" in item for item in fixture.validate()))

        with tempfile.TemporaryDirectory() as directory:
            fixture = ApparatusFixture(Path(directory))
            result = fixture.result_value()
            result["host_cgroup_memory"]["peak_file"] = "/sys/fs/cgroup/wrong.peak"
            fixture._write_result(result)
            errors = fixture.validate()
            self.assertTrue(any("independent TSV recomputation" in item for item in errors))

    def test_path_escape_and_wrong_external_job_binding_fail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = ApparatusFixture(Path(directory))
            outside = fixture.root / "outside-submission.json"
            outside.write_bytes(fixture.submission.read_bytes())
            result = fixture.result_value()
            result["submission_receipt_path"] = str(outside)
            fixture._write_result(result)
            errors = validator.validate_r05a_apparatus_result(
                fixture.result,
                outside,
                expected_run_root=fixture.run_root,
                expected_result_path=fixture.final_result,
                repo_root=fixture.repo,
                expected_git_commit=fixture.commit,
                expected_job_id="9999_0",
                expected_array_job_id=fixture.array_job_id,
                expected_task_id=0,
                expected_submission_sha256=_sha(outside),
            )
            self.assertTrue(any("escaped the immutable run root" in item for item in errors))
            self.assertTrue(any("slurm_job_id changed" in item for item in errors))


if __name__ == "__main__":
    unittest.main()
