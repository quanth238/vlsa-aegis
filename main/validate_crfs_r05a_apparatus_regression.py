#!/usr/bin/env python3
"""Strictly validate the preregistered R05A CPU apparatus result.

This validator is dependency-free on purpose.  It reparses the persisted test
log and cgroup sidecar independently of the allocation runner and refuses any
artifact whose receipt, source hashes, resource identity, or no-science claims
have changed.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any, Mapping, Optional


EXPECTED_RUN_ID = "r05a-adr0031-apparatus-cpu-20260715a"
EXPECTED_RESULT_ROLE = "r05a_adr0031_cpu_apparatus_regression"
EXPECTED_SUBMISSION_ROLE = "r05a_adr0031_cpu_apparatus_submission"
EXPECTED_HELD_ROLE = "r05a_adr0031_cpu_apparatus_held_submission"
EXPECTED_NODE = "worker-1"
EXPECTED_PARTITION = "main"
EXPECTED_ACCOUNT = "normal"
EXPECTED_QOS = "normal"
EXPECTED_TIME_LIMIT = "00:20:00"
EXPECTED_CPUS = 2
EXPECTED_MEMORY_MIB = 8192
MAX_CGROUP_PEAK_BYTES = EXPECTED_MEMORY_MIB * 1024 * 1024
EXPECTED_COUNTS = {
    "test_inverse_flow_control.py": 17,
    "test_inverse_flow_sampler.py": 8,
    "test_inverse_flow_policy.py": 10,
    "test_r05a_canary.py": 12,
}
REGISTRY_RELATIVE_PATH = "main/crfs_oracle/r05a_allocation_tests.json"
SOURCE_HASH_PATHS = {
    "registry_sha256": REGISTRY_RELATIVE_PATH,
    "allocation_test_helper_sha256": "scripts/hpc/lib/r05a_allocation_tests.sh",
    "cgroup_helper_sha256": "scripts/hpc/lib/cgroup_memory.sh",
    "runner_sha256": "scripts/hpc/run_r05a_apparatus_regression.sh",
    "slurm_sha256": "slurm/r05a_apparatus_regression_cpu.sbatch",
    "validator_sha256": "main/validate_crfs_r05a_apparatus_regression.py",
}
CGROUP_KEYS = (
    "schema_version",
    "artifact_role",
    "status",
    "reason",
    "cgroup_version",
    "membership_path",
    "mount_root",
    "mount_point",
    "membership_relative_to_mount_root",
    "peak_file",
    "peak_bytes",
    "proc_cgroup_file",
    "mountinfo_file",
)
RESULT_KEYS = {
    "schema_version",
    "artifact_role",
    "status",
    "run_id",
    "timestamp_utc",
    "git_commit",
    "git_dirty",
    "submission_receipt_path",
    "submission_receipt_sha256",
    "host",
    "slurm_job_id",
    "slurm_array_job_id",
    "slurm_array_task_id",
    "requested_cpus",
    "requested_host_memory_mib",
    "partition",
    "account",
    "qos",
    "time_limit",
    "requeue",
    "gpu_allocated",
    "allocation_tests",
    "host_cgroup_memory",
    "scientific_claim_allowed",
    "simulator_efficacy_evaluated",
    "checkpoint_loaded",
    "policy_server_started",
    "real_pi05_teacher_searches_executed",
    "real_pi05_teacher_observations_produced",
    "synthetic_unit_test_solver_calls_excluded_from_guard",
    "simulator_steps_executed",
    "probe_training_authorized",
    "retry_c_authorized_by_this_artifact_alone",
}
SUBMISSION_KEYS = {
    "schema_version",
    "artifact_role",
    "status",
    "run_id",
    "git_commit",
    "slurm_array_job_id",
    "slurm_array_task_id",
    "source_node",
    "partition",
    "account",
    "qos",
    "time_limit",
    "requeue",
    "array",
    "requested_cpus",
    "requested_host_memory_mib",
    "requested_gpus",
    "registry_sha256",
    "allocation_test_helper_sha256",
    "cgroup_helper_sha256",
    "runner_sha256",
    "slurm_sha256",
    "validator_sha256",
    "held_submission_sha256",
    "expected_result",
    "scientific_claim_allowed",
    "timestamp_utc",
}
HELD_KEYS = {
    "schema_version",
    "artifact_role",
    "status",
    "run_id",
    "git_commit",
    "slurm_array_job_id",
    "slurm_array_task_id",
    "source_node",
    "partition",
    "account",
    "qos",
    "time_limit",
    "requeue",
    "requested_cpus",
    "requested_host_memory_mib",
    "requested_gpus",
    "released_at_receipt_time",
    "scientific_claim_allowed",
    "timestamp_utc",
}
ALLOCATION_TEST_KEYS = {
    "registry_path",
    "registry_sha256",
    "expected_counts",
    "observed_counts",
    "zero_skips",
    "log_path",
    "log_sha256",
}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _is_git_commit(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{40}", value) is not None


def _load_object(path: Path, *, label: str, errors: list[str]) -> Optional[dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        errors.append(f"{label} cannot be read as JSON: {error}")
        return None
    if type(value) is not dict:
        errors.append(f"{label} must be a JSON object")
        return None
    return value


def _exact_keys(
    value: Mapping[str, Any], expected: set[str], *, label: str, errors: list[str]
) -> None:
    observed = set(value)
    if observed != expected:
        errors.append(
            f"{label} keys changed: missing={sorted(expected - observed)!r} "
            f"extra={sorted(observed - expected)!r}"
        )


def _check_equal(
    value: Mapping[str, Any], key: str, expected: Any, *, label: str, errors: list[str]
) -> None:
    if value.get(key) != expected or type(value.get(key)) is not type(expected):
        errors.append(f"{label}.{key} changed")


def _validate_utc_timestamp(value: Any, *, label: str, errors: list[str]) -> None:
    if not isinstance(value, str) or not value:
        errors.append(f"{label} is not a timestamp string")
        return
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        errors.append(f"{label} is not an ISO-8601 timestamp")
        return
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        errors.append(f"{label} is not UTC")


def _normalize_expected_path(path: Path, *, must_exist: bool) -> Path:
    if must_exist:
        return path.resolve(strict=True)
    parent = path.parent.resolve(strict=True)
    return parent / path.name


def _strict_registry(path: Path, *, errors: list[str]) -> Optional[dict[str, int]]:
    value = _load_object(path, label="allocation registry", errors=errors)
    if value is None:
        return None
    _exact_keys(value, {"schema_version", "suites"}, label="allocation registry", errors=errors)
    if value.get("schema_version") != "1.0":
        errors.append("allocation registry schema_version changed")
    suites = value.get("suites")
    if type(suites) is not list:
        errors.append("allocation registry suites must be a list")
        return None
    observed: dict[str, int] = {}
    for index, item in enumerate(suites):
        if type(item) is not dict or set(item) != {"pattern", "expected_tests"}:
            errors.append(f"allocation registry suite {index} changed shape")
            continue
        pattern = item.get("pattern")
        count = item.get("expected_tests")
        if type(pattern) is not str or re.fullmatch(r"test_[A-Za-z0-9_]+\.py", pattern) is None:
            errors.append(f"allocation registry suite {index} has an unsafe pattern")
            continue
        if type(count) is not int or count <= 0:
            errors.append(f"allocation registry suite {pattern} has a non-integer count")
            continue
        if pattern in observed:
            errors.append(f"allocation registry duplicates {pattern}")
            continue
        observed[pattern] = count
    if observed != EXPECTED_COUNTS or list(observed) != list(EXPECTED_COUNTS):
        errors.append("allocation registry differs from preregistered 17/8/10/12 suites")
    return observed


def _parse_allocation_log(
    path: Path, counts: Mapping[str, int], *, errors: list[str]
) -> Optional[dict[str, int]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        errors.append(f"allocation test log cannot be read: {error}")
        return None
    expected_markers = [
        f"verified_test_suite={suite} expected={count} observed={count} skips=0 status=passed"
        for suite, count in counts.items()
    ]
    markers = [line for line in lines if line.startswith("verified_test_suite=")]
    if markers != expected_markers:
        errors.append("allocation test markers changed in content, count, or order")
    for suite, count in counts.items():
        prefix = f"[{suite}] "
        suite_lines = [line for line in lines if line.startswith(prefix)]
        ran = re.compile(rf"^\[{re.escape(suite)}\] Ran {count} tests? in .+$")
        if sum(ran.fullmatch(line) is not None for line in suite_lines) != 1:
            errors.append(f"allocation test count evidence changed for {suite}")
        if suite_lines.count(f"[{suite}] OK") != 1:
            errors.append(f"allocation test success evidence changed for {suite}")
        if any(
            line.startswith(f"[{suite}] FAILED")
            or line.startswith(f"[{suite}] ERROR")
            or "skipped=" in line
            or re.search(r"\.\.\. skipped(?:\s|$)", line, flags=re.IGNORECASE) is not None
            for line in suite_lines
        ):
            errors.append(f"allocation test suite skipped or failed: {suite}")
    if any(
        line.startswith("FAILED") or line.startswith("ERROR") or "skipped=" in line
        for line in lines
        if not line.startswith("[")
    ):
        errors.append("allocation test log contains unscoped failure or skip evidence")
    return dict(counts)


def _decode_diagnostic_value(value: str) -> str:
    decoded: list[str] = []
    index = 0
    escapes = {"\\": "\\", "t": "\t", "n": "\n"}
    while index < len(value):
        if value[index] != "\\":
            decoded.append(value[index])
            index += 1
            continue
        if index + 1 >= len(value) or value[index + 1] not in escapes:
            raise ValueError("invalid diagnostic escape")
        decoded.append(escapes[value[index + 1]])
        index += 2
    return "".join(decoded)


def _parse_cgroup_diagnostic(path: Path, *, errors: list[str]) -> Optional[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        errors.append(f"cgroup diagnostic cannot be read: {error}")
        return None
    keys: list[str] = []
    record: dict[str, str] = {}
    for line in lines:
        if "\t" not in line:
            errors.append("cgroup diagnostic line has no tab separator")
            return None
        key, encoded = line.split("\t", 1)
        if key in record:
            errors.append(f"cgroup diagnostic duplicates {key}")
            return None
        keys.append(key)
        try:
            record[key] = _decode_diagnostic_value(encoded)
        except ValueError as error:
            errors.append(f"cgroup diagnostic {key}: {error}")
            return None
    if tuple(keys) != CGROUP_KEYS:
        errors.append("cgroup diagnostic keys or order changed")
        return None
    identity = {
        "schema_version": "1.0",
        "artifact_role": "r05a_live_slurm_cgroup_memory_peak_diagnostic",
        "status": "measured",
        "reason": "live_positive_peak",
        "proc_cgroup_file": "/proc/self/cgroup",
        "mountinfo_file": "/proc/self/mountinfo",
    }
    for key, expected in identity.items():
        if record.get(key) != expected:
            errors.append(f"cgroup diagnostic {key} changed")
    version = record.get("cgroup_version")
    if version not in {"1", "2"}:
        errors.append("cgroup diagnostic version changed")
        return None
    for key in ("membership_path", "mount_root", "mount_point", "peak_file"):
        value = record.get(key, "")
        if not value.startswith("/") or str(PurePosixPath(value)) != value:
            errors.append(f"cgroup diagnostic {key} is not a normalized absolute path")
    membership = record.get("membership_path", "")
    mount_root = record.get("mount_root", "")
    if mount_root == "/":
        expected_relative = "" if membership == "/" else membership
    elif membership == mount_root:
        expected_relative = ""
    elif membership.startswith(mount_root + "/"):
        expected_relative = membership[len(mount_root) :]
    else:
        errors.append("cgroup membership is outside the recorded mount root")
        expected_relative = None
    relative = record.get("membership_relative_to_mount_root")
    if expected_relative is not None and relative != expected_relative:
        errors.append("cgroup membership-relative path is inconsistent")
    metric = "memory.peak" if version == "2" else "memory.max_usage_in_bytes"
    if expected_relative is not None:
        expected_peak = str(
            PurePosixPath(record.get("mount_point", ""))
            / expected_relative.lstrip("/")
            / metric
        )
        if record.get("peak_file") != expected_peak:
            errors.append("cgroup peak path is inconsistent with mount provenance")
    peak_text = record.get("peak_bytes", "")
    try:
        peak = int(peak_text)
    except (TypeError, ValueError):
        errors.append("cgroup peak is not an integer")
        return None
    if peak <= 0 or str(peak) != peak_text:
        errors.append("cgroup peak is not a canonical positive integer")
    if peak > MAX_CGROUP_PEAK_BYTES:
        errors.append("cgroup peak exceeds the exact 8 GiB request")
    normalized: dict[str, Any] = dict(record)
    normalized["peak_bytes"] = peak
    return normalized


def validate_r05a_apparatus_result(
    result_path: str | Path,
    submission_path: str | Path,
    *,
    expected_run_root: str | Path,
    expected_result_path: str | Path,
    repo_root: str | Path,
    expected_git_commit: str,
    expected_job_id: str,
    expected_array_job_id: str,
    expected_submission_sha256: str,
    expected_task_id: int = 0,
) -> list[str]:
    """Return every strict validation error for one CPU apparatus result."""

    errors: list[str] = []
    if not _is_git_commit(expected_git_commit):
        errors.append("expected git commit is not a lowercase 40-hex commit")
    if re.fullmatch(r"[0-9]+(?:_[0-9]+)?", expected_job_id) is None:
        errors.append("expected Slurm job id is invalid")
    if re.fullmatch(r"[0-9]+", expected_array_job_id) is None:
        errors.append("expected Slurm array job id is invalid")
    if type(expected_task_id) is not int or expected_task_id != 0:
        errors.append("expected Slurm task id must be exact row zero")
    if not _is_sha256(expected_submission_sha256):
        errors.append("expected submission receipt SHA-256 is invalid")

    try:
        run_root = _normalize_expected_path(Path(expected_run_root), must_exist=True)
        result_file = _normalize_expected_path(Path(result_path), must_exist=True)
        submission_file = _normalize_expected_path(Path(submission_path), must_exist=True)
        repository = _normalize_expected_path(Path(repo_root), must_exist=True)
        final_result = _normalize_expected_path(Path(expected_result_path), must_exist=False)
    except OSError as error:
        return errors + [f"an expected validation path cannot be resolved: {error}"]
    if run_root.name != EXPECTED_RUN_ID:
        errors.append("run-root basename differs from the preregistered run id")
    if result_file.parent != run_root:
        errors.append("result candidate escaped the immutable run root")
    if submission_file != run_root / "submission.json":
        errors.append("submission receipt escaped the immutable run root")
    if final_result != run_root / "results.json":
        errors.append("expected final result path changed")

    result = _load_object(result_file, label="CPU apparatus result", errors=errors)
    submission = _load_object(submission_file, label="submission receipt", errors=errors)
    if result is None or submission is None:
        return errors
    _exact_keys(result, RESULT_KEYS, label="CPU apparatus result", errors=errors)
    _exact_keys(submission, SUBMISSION_KEYS, label="submission receipt", errors=errors)

    result_identity = {
        "schema_version": "1.0",
        "artifact_role": EXPECTED_RESULT_ROLE,
        "status": "passed",
        "run_id": EXPECTED_RUN_ID,
        "git_commit": expected_git_commit,
        "git_dirty": False,
        "submission_receipt_path": str(submission_file),
        "submission_receipt_sha256": expected_submission_sha256,
        "host": EXPECTED_NODE,
        "slurm_job_id": expected_job_id,
        "slurm_array_job_id": expected_array_job_id,
        "slurm_array_task_id": expected_task_id,
        "requested_cpus": EXPECTED_CPUS,
        "requested_host_memory_mib": EXPECTED_MEMORY_MIB,
        "partition": EXPECTED_PARTITION,
        "account": EXPECTED_ACCOUNT,
        "qos": EXPECTED_QOS,
        "time_limit": EXPECTED_TIME_LIMIT,
        "requeue": False,
        "gpu_allocated": False,
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
    for key, expected in result_identity.items():
        _check_equal(result, key, expected, label="CPU apparatus result", errors=errors)
    _validate_utc_timestamp(result.get("timestamp_utc"), label="result.timestamp_utc", errors=errors)

    try:
        actual_submission_sha = _file_sha256(submission_file)
    except OSError as error:
        errors.append(f"submission receipt cannot be hashed: {error}")
        actual_submission_sha = None
    if actual_submission_sha != expected_submission_sha256:
        errors.append("submission receipt differs from the externally bound SHA-256")

    submission_identity = {
        "schema_version": "1.0",
        "artifact_role": EXPECTED_SUBMISSION_ROLE,
        "status": "reviewed_job_held_and_receipted",
        "run_id": EXPECTED_RUN_ID,
        "git_commit": expected_git_commit,
        "slurm_array_job_id": expected_array_job_id,
        "slurm_array_task_id": expected_task_id,
        "source_node": EXPECTED_NODE,
        "partition": EXPECTED_PARTITION,
        "account": EXPECTED_ACCOUNT,
        "qos": EXPECTED_QOS,
        "time_limit": EXPECTED_TIME_LIMIT,
        "requeue": False,
        "array": "0-0%1",
        "requested_cpus": EXPECTED_CPUS,
        "requested_host_memory_mib": EXPECTED_MEMORY_MIB,
        "requested_gpus": 0,
        "expected_result": str(final_result),
        "scientific_claim_allowed": False,
    }
    for key, expected in submission_identity.items():
        _check_equal(submission, key, expected, label="submission receipt", errors=errors)
    _validate_utc_timestamp(
        submission.get("timestamp_utc"), label="submission.timestamp_utc", errors=errors
    )

    for receipt_key, relative in SOURCE_HASH_PATHS.items():
        source_path = repository / relative
        try:
            if source_path.resolve(strict=True) != source_path:
                errors.append(f"bound source {relative} is not a canonical in-repository file")
            observed_sha = _file_sha256(source_path)
        except OSError as error:
            errors.append(f"bound source {relative} cannot be hashed: {error}")
            continue
        if submission.get(receipt_key) != observed_sha:
            errors.append(f"submission receipt {receipt_key} differs from bound source {relative}")

    held_path = run_root / "held-submission.json"
    held = _load_object(held_path, label="held submission receipt", errors=errors)
    if held is not None:
        _exact_keys(held, HELD_KEYS, label="held submission receipt", errors=errors)
        held_identity = {
            "schema_version": "1.0",
            "artifact_role": EXPECTED_HELD_ROLE,
            "status": "sbatch_returned_held_job_id",
            "run_id": EXPECTED_RUN_ID,
            "git_commit": expected_git_commit,
            "slurm_array_job_id": expected_array_job_id,
            "slurm_array_task_id": expected_task_id,
            "source_node": EXPECTED_NODE,
            "partition": EXPECTED_PARTITION,
            "account": EXPECTED_ACCOUNT,
            "qos": EXPECTED_QOS,
            "time_limit": EXPECTED_TIME_LIMIT,
            "requeue": False,
            "requested_cpus": EXPECTED_CPUS,
            "requested_host_memory_mib": EXPECTED_MEMORY_MIB,
            "requested_gpus": 0,
            "released_at_receipt_time": False,
            "scientific_claim_allowed": False,
        }
        for key, expected in held_identity.items():
            _check_equal(held, key, expected, label="held submission receipt", errors=errors)
        _validate_utc_timestamp(
            held.get("timestamp_utc"), label="held_submission.timestamp_utc", errors=errors
        )
        try:
            held_sha = _file_sha256(held_path)
        except OSError as error:
            errors.append(f"held submission receipt cannot be hashed: {error}")
        else:
            if submission.get("held_submission_sha256") != held_sha:
                errors.append("held submission receipt SHA-256 changed")

    registry_path = repository / REGISTRY_RELATIVE_PATH
    registry_counts = _strict_registry(registry_path, errors=errors)
    registry_sha: Optional[str]
    try:
        registry_sha = _file_sha256(registry_path)
    except OSError as error:
        errors.append(f"allocation registry cannot be hashed: {error}")
        registry_sha = None

    allocation = result.get("allocation_tests")
    if type(allocation) is not dict:
        errors.append("CPU apparatus result allocation_tests must be an object")
    else:
        _exact_keys(allocation, ALLOCATION_TEST_KEYS, label="allocation_tests", errors=errors)
        _check_equal(
            allocation,
            "registry_path",
            REGISTRY_RELATIVE_PATH,
            label="allocation_tests",
            errors=errors,
        )
        if registry_sha is not None:
            _check_equal(
                allocation,
                "registry_sha256",
                registry_sha,
                label="allocation_tests",
                errors=errors,
            )
        _check_equal(
            allocation,
            "expected_counts",
            EXPECTED_COUNTS,
            label="allocation_tests",
            errors=errors,
        )
        _check_equal(
            allocation,
            "observed_counts",
            EXPECTED_COUNTS,
            label="allocation_tests",
            errors=errors,
        )
        _check_equal(allocation, "zero_skips", True, label="allocation_tests", errors=errors)
        log_path = run_root / "allocation-focused-tests.log"
        _check_equal(
            allocation, "log_path", str(log_path), label="allocation_tests", errors=errors
        )
        if registry_counts is not None:
            _parse_allocation_log(log_path, registry_counts, errors=errors)
        try:
            log_sha = _file_sha256(log_path)
        except OSError as error:
            errors.append(f"allocation test log cannot be hashed: {error}")
        else:
            _check_equal(
                allocation, "log_sha256", log_sha, label="allocation_tests", errors=errors
            )

    cgroup = result.get("host_cgroup_memory")
    if type(cgroup) is not dict:
        errors.append("CPU apparatus result host_cgroup_memory must be an object")
    else:
        expected_cgroup_keys = {"diagnostic_path", "diagnostic_sha256", *CGROUP_KEYS}
        _exact_keys(cgroup, expected_cgroup_keys, label="host_cgroup_memory", errors=errors)
        diagnostic_path = run_root / "host-cgroup-memory.tsv"
        _check_equal(
            cgroup,
            "diagnostic_path",
            str(diagnostic_path),
            label="host_cgroup_memory",
            errors=errors,
        )
        parsed_cgroup = _parse_cgroup_diagnostic(diagnostic_path, errors=errors)
        diagnostic_sha: Optional[str]
        try:
            diagnostic_sha = _file_sha256(diagnostic_path)
        except OSError as error:
            errors.append(f"cgroup diagnostic cannot be hashed: {error}")
            diagnostic_sha = None
        else:
            _check_equal(
                cgroup,
                "diagnostic_sha256",
                diagnostic_sha,
                label="host_cgroup_memory",
                errors=errors,
            )
        if parsed_cgroup is not None and diagnostic_sha is not None:
            expected_cgroup = {
                "diagnostic_path": str(diagnostic_path),
                "diagnostic_sha256": diagnostic_sha,
                **parsed_cgroup,
            }
            if cgroup != expected_cgroup:
                errors.append("host_cgroup_memory differs from independent TSV recomputation")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True)
    parser.add_argument("--submission", required=True)
    parser.add_argument("--expected-run-root", required=True)
    parser.add_argument("--expected-result-path", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--expected-git-commit", required=True)
    parser.add_argument("--expected-job-id", required=True)
    parser.add_argument("--expected-array-job-id", required=True)
    parser.add_argument("--expected-task-id", required=True, type=int)
    parser.add_argument("--expected-submission-sha256", required=True)
    args = parser.parse_args()
    errors = validate_r05a_apparatus_result(
        args.result,
        args.submission,
        expected_run_root=args.expected_run_root,
        expected_result_path=args.expected_result_path,
        repo_root=args.repo_root,
        expected_git_commit=args.expected_git_commit,
        expected_job_id=args.expected_job_id,
        expected_array_job_id=args.expected_array_job_id,
        expected_task_id=args.expected_task_id,
        expected_submission_sha256=args.expected_submission_sha256,
    )
    print(f"validation_error_count={len(errors)}")
    for error in errors:
        print(f"validation_error={error}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
