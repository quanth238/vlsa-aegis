#!/usr/bin/env python3
"""Verify and index the published VLSA/AEGIS population for transfer.

The population publisher is the scientific authority.  This post-publication
tool runs only in a CPU Slurm allocation whose dependency is the exact
successful publisher.  It reconstructs the publisher's v2 compact inventory,
rehashes every final result, video, gallery, and failure-analysis artifact,
then atomically publishes a small transfer bundle outside the immutable run
root.

The tool does not copy videos and does not make a scientific claim.
"""

from __future__ import annotations

import argparse
from collections import Counter
import ctypes
from datetime import datetime, timezone
import errno
import hashlib
import html
import json
import os
from pathlib import Path
import re
import socket
import stat
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple
from urllib.parse import quote

from analysis import aggregate_safelibero_aegis as aggregate
from analysis import validate_aegis_failure_diagnostics as failure_validation
from scripts import validate_aegis_run_artifacts as official


PUBLICATION_SCHEMA = "vlsa_table1_population_publication.v1"
PREPUBLISH_SCHEMA = "vlsa_table1_population_prepublish_validation.v2"
SUMMARY_SCHEMA = "vlsa_table1_population_summary.v1"
RESULT_SCHEMA = "vlsa_table1_episode_result.v1"
ROW_SCHEMA = "vlsa_table1_video_transfer_row.v2"
RECEIPT_SCHEMA = "vlsa_table1_video_transfer_receipt.v2"
FAILURE_CASE_SCHEMA = "vlsa_table1_aegis_failure_case.v1"
FAILURE_REPORT_SCHEMA = "vlsa_table1_aegis_failure_report.v1"

EXPECTED_CASES = 1600
EXPECTED_TASKS = 32
CASES_PER_TASK = 50
EXPECTED_RESULTS = 3200
EXPECTED_ARMS = (
    "pi05_translational",
    "pi05_plus_aegis_translational",
)
EXPECTED_MODES = ("pi05", "aegis")
ACCEPTED_STATUSES = {
    "complete",
    "method_failure",
    "method_failure_passthrough",
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


class TransferVerificationError(RuntimeError):
    """The immutable publication or transfer contract is inconsistent."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def require_equal(observed: Any, expected: Any, label: str) -> None:
    if observed != expected:
        raise TransferVerificationError(
            "{} differs: observed={!r}, expected={!r}".format(
                label, observed, expected
            )
        )


def require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise TransferVerificationError(
            "{} must be lowercase SHA-256".format(label)
        )
    return value


def require_commit(value: Any, label: str) -> str:
    if not isinstance(value, str) or GIT_COMMIT_RE.fullmatch(value) is None:
        raise TransferVerificationError(
            "{} must be a lowercase 40-hex commit".format(label)
        )
    return value


def _identity(file_stat: os.stat_result) -> Tuple[int, int, int, int]:
    return (
        int(file_stat.st_dev),
        int(file_stat.st_ino),
        int(file_stat.st_size),
        int(
            getattr(
                file_stat,
                "st_mtime_ns",
                int(file_stat.st_mtime * 1_000_000_000),
            )
        ),
    )


def stable_file_sha256_and_size(path: Path, label: str) -> Tuple[str, int]:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(str(path), flags)
    except OSError as error:
        raise TransferVerificationError(
            "{} cannot be opened safely: {}".format(label, path)
        ) from error
    digest = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise TransferVerificationError(
                "{} is not a regular file".format(label)
            )
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        path_stat = os.stat(str(path), follow_symlinks=False)
    except OSError as error:
        raise TransferVerificationError(
            "{} disappeared after hashing".format(label)
        ) from error
    if (
        _identity(before) != _identity(after)
        or _identity(after) != _identity(path_stat)
        or stat.S_ISLNK(path_stat.st_mode)
    ):
        raise TransferVerificationError(
            "{} changed while being hashed".format(label)
        )
    return digest.hexdigest(), int(after.st_size)


def stable_read_bytes(path: Path, label: str) -> Tuple[bytes, str]:
    expected_sha256, expected_size = stable_file_sha256_and_size(path, label)
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(str(path), flags)
    except OSError as error:
        raise TransferVerificationError(
            "{} cannot be reopened safely".format(label)
        ) from error
    chunks: List[bytes] = []
    try:
        before = os.fstat(descriptor)
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    payload = b"".join(chunks)
    if (
        _identity(before) != _identity(after)
        or len(payload) != expected_size
        or sha256_bytes(payload) != expected_sha256
    ):
        raise TransferVerificationError(
            "{} changed between hash and read".format(label)
        )
    return payload, expected_sha256


def load_json_with_sha(path: Path, label: str) -> Tuple[Dict[str, Any], str]:
    payload, digest = stable_read_bytes(path, label)
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TransferVerificationError(
            "{} is invalid JSON".format(label)
        ) from error
    if not isinstance(value, dict):
        raise TransferVerificationError(
            "{} must contain one JSON object".format(label)
        )
    return value, digest


def verify_payload_hash(
    value: Mapping[str, Any], field: str, label: str
) -> str:
    expected = require_sha256(
        value.get(field), "{}/{}".format(label, field)
    )
    payload = {key: item for key, item in value.items() if key != field}
    observed = sha256_bytes(canonical_json_bytes(payload))
    require_equal(observed, expected, "{} payload SHA-256".format(label))
    return observed


def lexical_absolute(path: Path) -> Path:
    return Path(os.path.abspath(str(path)))


def relative_under(root: Path, candidate: Path, label: str) -> Path:
    root = lexical_absolute(root)
    candidate = lexical_absolute(candidate)
    try:
        return candidate.relative_to(root)
    except ValueError as error:
        raise TransferVerificationError(
            "{} escapes root: {}".format(label, candidate)
        ) from error


def real_file_under(root: Path, candidate: Path, label: str) -> Path:
    relative = relative_under(root, candidate, label)
    current = lexical_absolute(root)
    if current.is_symlink() or not current.is_dir():
        raise TransferVerificationError("root is missing or symlinked")
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise TransferVerificationError(
                "{} traverses a symlink: {}".format(label, current)
            )
    if not current.is_file():
        raise TransferVerificationError(
            "{} is not a regular file: {}".format(label, current)
        )
    return current


def real_dir_under(root: Path, candidate: Path, label: str) -> Path:
    relative = relative_under(root, candidate, label)
    current = lexical_absolute(root)
    if current.is_symlink() or not current.is_dir():
        raise TransferVerificationError("root is missing or symlinked")
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise TransferVerificationError(
                "{} traverses a symlink: {}".format(label, current)
            )
    if not current.is_dir():
        raise TransferVerificationError(
            "{} is not a real directory: {}".format(label, current)
        )
    return current


def reject_symlinks(root: Path, label: str) -> None:
    for directory, dirnames, filenames in os.walk(
        str(root), followlinks=False
    ):
        base = Path(directory)
        for name in list(dirnames) + list(filenames):
            candidate = base / name
            if candidate.is_symlink():
                raise TransferVerificationError(
                    "{} contains a symlink: {}".format(label, candidate)
                )


def load_bound_artifact(
    run_root: Path,
    record: Any,
    label: str,
    *,
    json_object: bool,
) -> Tuple[Path, Optional[Dict[str, Any]], str]:
    if not isinstance(record, Mapping):
        raise TransferVerificationError("{} record is missing".format(label))
    raw_path = record.get("path")
    if not isinstance(raw_path, str) or not Path(raw_path).is_absolute():
        raise TransferVerificationError(
            "{} path must be absolute".format(label)
        )
    path = real_file_under(run_root, Path(raw_path), label)
    expected_sha256 = require_sha256(
        record.get("sha256"), "{} file".format(label)
    )
    if json_object:
        payload, observed_sha256 = stable_read_bytes(path, label)
    else:
        value = None
        observed_sha256 = stable_file_sha256_and_size(path, label)[0]
    require_equal(
        observed_sha256,
        expected_sha256,
        "{} file SHA-256".format(label),
    )
    if json_object:
        try:
            value = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise TransferVerificationError(
                "{} is invalid JSON".format(label)
            ) from error
        if not isinstance(value, dict):
            raise TransferVerificationError(
                "{} must contain one JSON object".format(label)
            )
    return path, value, observed_sha256


def validate_cpu_allocation(
    environment: Mapping[str, str],
    *,
    expected_publisher_job_id: str,
    actual_hostname: Optional[str] = None,
) -> Dict[str, Any]:
    required = (
        "SLURM_JOB_ID",
        "SLURMD_NODENAME",
        "SLURM_JOB_PARTITION",
        "SLURM_CPUS_PER_TASK",
        "SLURM_MEM_PER_NODE",
        "SLURM_JOB_DEPENDENCY",
    )
    missing = [key for key in required if not environment.get(key)]
    if missing:
        raise TransferVerificationError(
            "CPU Slurm allocation fields are missing: {}".format(missing)
        )
    node = str(environment["SLURMD_NODENAME"])
    if node == "worker-3" or node.startswith(("login", "login-restricted")):
        raise TransferVerificationError(
            "postpublication verifier cannot run on {}".format(node)
        )
    hostname = (actual_hostname or socket.gethostname()).split(".", 1)[0]
    require_equal(
        hostname,
        node.split(".", 1)[0],
        "Slurm node/actual host",
    )
    try:
        cpus = int(environment["SLURM_CPUS_PER_TASK"])
        memory = int(environment["SLURM_MEM_PER_NODE"])
    except ValueError as error:
        raise TransferVerificationError(
            "CPU or memory allocation is invalid"
        ) from error
    require_equal(cpus, 4, "transfer verifier CPU count")
    require_equal(memory, 32768, "transfer verifier memory MiB")
    require_equal(
        environment["SLURM_JOB_DEPENDENCY"],
        "afterok:{}".format(expected_publisher_job_id),
        "transfer verifier dependency",
    )
    gpu_fields = (
        "SLURM_JOB_GPUS",
        "SLURM_STEP_GPUS",
        "SLURM_GPUS",
        "SLURM_GPUS_ON_NODE",
        "CUDA_VISIBLE_DEVICES",
        "NVIDIA_VISIBLE_DEVICES",
        "SLURM_JOB_GRES",
        "SLURM_STEP_GRES",
        "SLURM_TRES_PER_TASK",
        "SLURM_JOB_TRES",
    )
    unsafe = {}
    for key in gpu_fields:
        value = environment.get(key)
        if value is None:
            continue
        normalized = str(value).strip()
        if normalized in {
            "",
            "-1",
            "none",
            "None",
            "void",
            "NoDevFiles",
        }:
            continue
        if normalized == "0" and key in {
            "SLURM_GPUS",
            "SLURM_GPUS_ON_NODE",
        }:
            continue
        if "gpu" in normalized.lower() or key in {
            "SLURM_JOB_GPUS",
            "SLURM_STEP_GPUS",
            "CUDA_VISIBLE_DEVICES",
            "NVIDIA_VISIBLE_DEVICES",
        }:
            unsafe[key] = value
    if unsafe:
        raise TransferVerificationError(
            "postpublication verifier must be CPU-only: {}".format(unsafe)
        )
    return {
        "job_id": str(environment["SLURM_JOB_ID"]),
        "node": node,
        "partition": str(environment["SLURM_JOB_PARTITION"]),
        "cpus_per_task": cpus,
        "memory_mib": memory,
        "dependency": str(environment["SLURM_JOB_DEPENDENCY"]),
        "gpu_allocation": False,
    }


def validate_allocation_record(
    value: Mapping[str, Any],
    *,
    expected_publisher_job_id: str,
) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TransferVerificationError(
            "transfer verifier allocation record is missing"
        )
    expected_dependency = "afterok:{}".format(expected_publisher_job_id)
    for field, expected in (
        ("partition", "main"),
        ("cpus_per_task", 4),
        ("memory_mib", 32768),
        ("dependency", expected_dependency),
        ("gpu_allocation", False),
    ):
        require_equal(
            value.get(field),
            expected,
            "transfer allocation/{}".format(field),
        )
    job_id = value.get("job_id")
    node = value.get("node")
    if not isinstance(job_id, str) or not job_id.isdigit():
        raise TransferVerificationError(
            "transfer allocation job ID must be numeric"
        )
    if (
        not isinstance(node, str)
        or not node
        or node == "worker-3"
        or node.startswith(("login", "login-restricted"))
    ):
        raise TransferVerificationError(
            "transfer allocation node is invalid: {!r}".format(node)
        )
    return {
        "job_id": job_id,
        "node": node,
        "partition": "main",
        "cpus_per_task": 4,
        "memory_mib": 32768,
        "dependency": expected_dependency,
        "gpu_allocation": False,
    }


def validate_verifier_identity(
    *,
    expected_sha256: str,
    expected_git_commit: str,
    script_path: Optional[Path] = None,
    repository_root: Optional[Path] = None,
) -> Dict[str, Any]:
    script = (script_path or Path(__file__)).resolve()
    repository = (
        repository_root or script.parent.parent
    ).resolve()
    observed_sha256 = stable_file_sha256_and_size(
        script, "postpublication verifier"
    )[0]
    require_equal(
        observed_sha256,
        require_sha256(expected_sha256, "expected verifier SHA-256"),
        "verifier SHA-256",
    )
    expected_commit = require_commit(
        expected_git_commit, "expected verifier git commit"
    )
    try:
        observed_commit = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            [
                "git",
                "-C",
                str(repository),
                "status",
                "--porcelain",
                "--untracked-files=all",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise TransferVerificationError(
            "cannot validate verifier git identity"
        ) from error
    require_equal(observed_commit, expected_commit, "verifier git commit")
    require_equal(dirty, "", "verifier source dirty state")
    try:
        script_relative = script.relative_to(repository).as_posix()
    except ValueError as error:
        raise TransferVerificationError(
            "verifier script is outside its repository"
        ) from error
    return {
        "path": str(script),
        "repository_root": str(repository),
        "repository_commit": observed_commit,
        "repository_dirty": False,
        "repository_relative_path": script_relative,
        "sha256": observed_sha256,
    }


def revalidate_verifier_identity(value: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TransferVerificationError(
            "postpublication verifier identity is missing"
        )
    raw_path = value.get("path")
    raw_repository = value.get("repository_root")
    if not isinstance(raw_path, str) or not isinstance(
        raw_repository, str
    ):
        raise TransferVerificationError(
            "postpublication verifier paths are missing"
        )
    observed = validate_verifier_identity(
        expected_sha256=require_sha256(
            value.get("sha256"), "recorded verifier SHA-256"
        ),
        expected_git_commit=require_commit(
            value.get("repository_commit"),
            "recorded verifier git commit",
        ),
        script_path=Path(raw_path),
        repository_root=Path(raw_repository),
    )
    require_equal(
        observed,
        dict(value),
        "postpublication verifier recorded identity",
    )
    return observed


def validate_failure_analysis(
    run_root: Path,
    record: Any,
) -> Dict[str, Any]:
    if not isinstance(record, Mapping):
        raise TransferVerificationError(
            "publication failure-analysis record is missing"
        )
    cases_path, _, cases_sha256 = load_bound_artifact(
        run_root,
        record.get("cases"),
        "failure case ledger",
        json_object=False,
    )
    report_path, report, report_sha256 = load_bound_artifact(
        run_root,
        record.get("report"),
        "failure report",
        json_object=True,
    )
    markdown_path, _, markdown_sha256 = load_bound_artifact(
        run_root,
        record.get("markdown"),
        "failure report Markdown",
        json_object=False,
    )
    assert report is not None
    require_equal(
        record["cases"].get("count"),
        EXPECTED_CASES,
        "failure case ledger count",
    )
    require_equal(
        report.get("schema_version"),
        FAILURE_REPORT_SCHEMA,
        "failure report schema",
    )
    require_equal(
        report.get("status"),
        "complete_population_failure_analysis",
        "failure report status",
    )
    report_payload_sha256 = verify_payload_hash(
        report, "report_payload_sha256", "failure report"
    )
    require_equal(
        record["report"].get("report_payload_sha256"),
        report_payload_sha256,
        "publication failure report payload",
    )
    rows: List[Dict[str, Any]] = []
    payload, _ = stable_read_bytes(cases_path, "failure case ledger")
    for line_number, raw_line in enumerate(payload.splitlines(), start=1):
        if not raw_line.strip():
            raise TransferVerificationError(
                "failure case ledger has a blank row"
            )
        try:
            row = json.loads(raw_line)
        except json.JSONDecodeError as error:
            raise TransferVerificationError(
                "failure case row {} is invalid JSON".format(line_number)
            ) from error
        if not isinstance(row, dict):
            raise TransferVerificationError(
                "failure case row {} is not an object".format(line_number)
            )
        require_equal(
            row.get("schema_version"),
            FAILURE_CASE_SCHEMA,
            "failure case row schema",
        )
        verify_payload_hash(
            row,
            "record_payload_sha256",
            "failure case row {}".format(line_number),
        )
        rows.append(row)
    require_equal(len(rows), EXPECTED_CASES, "failure case row count")
    case_ids = [row.get("case_id") for row in rows]
    if (
        any(not isinstance(case_id, str) or not case_id for case_id in case_ids)
        or len(set(case_ids)) != EXPECTED_CASES
    ):
        raise TransferVerificationError(
            "failure case identities are incomplete or duplicated"
        )
    population = report.get("population")
    counts = report.get("counts")
    if not isinstance(population, Mapping) or not isinstance(
        counts, Mapping
    ):
        raise TransferVerificationError(
            "failure report population or counts are missing"
        )
    for field, expected in (
        ("cases", EXPECTED_CASES),
        ("results", EXPECTED_RESULTS),
        ("no_cases_dropped", True),
        ("all_car_failures_classified", True),
        ("causal_limits_preserved", True),
    ):
        require_equal(
            population.get(field),
            expected,
            "failure report population/{}".format(field),
        )
    for field, expected in (
        ("case_count", EXPECTED_CASES),
        ("video_count", EXPECTED_RESULTS),
        ("unclassified_aegis_car_failures", 0),
        ("all_videos_hash_verified", True),
    ):
        require_equal(
            counts.get(field),
            expected,
            "failure report counts/{}".format(field),
        )
    case_ledger = [
        {
            "case_id": row["case_id"],
            "record_payload_sha256": row["record_payload_sha256"],
        }
        for row in sorted(
            rows, key=lambda row: int(row.get("case_ordinal", -1))
        )
    ]
    require_equal(
        report.get("case_record_ledger_sha256"),
        sha256_bytes(canonical_json_bytes(case_ledger)),
        "failure report case-record ledger SHA-256",
    )
    failure_root = cases_path.parent
    require_equal(
        report_path.parent,
        failure_root,
        "failure report directory",
    )
    require_equal(
        markdown_path.parent,
        failure_root,
        "failure Markdown directory",
    )
    reject_symlinks(failure_root, "failure-analysis directory")
    observed_entries = {path.name for path in failure_root.iterdir()}
    require_equal(
        observed_entries,
        {cases_path.name, report_path.name, markdown_path.name},
        "failure-analysis entry set",
    )
    markdown_size = stable_file_sha256_and_size(
        markdown_path, "failure report Markdown"
    )[1]
    if markdown_size < 1:
        raise TransferVerificationError("failure report Markdown is empty")
    return {
        "cases": {
            "path": str(cases_path),
            "run_relative_path": cases_path.relative_to(run_root).as_posix(),
            "sha256": cases_sha256,
            "rows": len(rows),
        },
        "report": {
            "path": str(report_path),
            "run_relative_path": report_path.relative_to(run_root).as_posix(),
            "sha256": report_sha256,
            "report_payload_sha256": report_payload_sha256,
        },
        "markdown": {
            "path": str(markdown_path),
            "run_relative_path": markdown_path.relative_to(run_root).as_posix(),
            "sha256": markdown_sha256,
            "bytes": markdown_size,
        },
        "report_value": report,
        "case_ids": set(case_ids),
    }


def validate_publication_chain(
    *,
    run_root: Path,
    publication_receipt: Path,
    expected_publication_receipt_sha256: str,
    expected_run_id: str,
    expected_source_commit: str,
    expected_population_array_job_id: str,
    expected_publisher_job_id: str,
    config_path: Path,
    manifest_path: Path,
    manifest_receipt_path: Path,
    labels_path: Path,
) -> Dict[str, Any]:
    expected_source_commit = require_commit(
        expected_source_commit, "expected outcome source commit"
    )
    if (
        not isinstance(expected_population_array_job_id, str)
        or not expected_population_array_job_id.isdigit()
    ):
        raise TransferVerificationError(
            "expected population array job ID must be numeric"
        )
    if (
        not isinstance(expected_publisher_job_id, str)
        or not expected_publisher_job_id.isdigit()
    ):
        raise TransferVerificationError(
            "expected publisher job ID must be numeric"
        )
    run_root = lexical_absolute(run_root)
    if run_root.is_symlink() or not run_root.is_dir():
        raise TransferVerificationError(
            "immutable run root is missing or symlinked"
        )
    run_root = run_root.resolve(strict=True)
    require_equal(run_root.name, expected_run_id, "run-root identity")
    expected_publication = run_root / "population-publication-receipt.json"
    try:
        observed_publication = publication_receipt.resolve(strict=True)
    except OSError as error:
        raise TransferVerificationError(
            "publication receipt cannot be resolved"
        ) from error
    require_equal(
        observed_publication,
        expected_publication,
        "publication receipt path",
    )
    publication_path = real_file_under(
        run_root, observed_publication, "publication receipt"
    )
    publication, publication_sha256 = load_json_with_sha(
        publication_path, "publication receipt"
    )
    require_equal(
        publication_sha256,
        require_sha256(
            expected_publication_receipt_sha256,
            "expected publication receipt SHA-256",
        ),
        "publication receipt SHA-256",
    )
    publication_payload_sha256 = verify_payload_hash(
        publication,
        "receipt_payload_sha256",
        "publication receipt",
    )
    for field, expected in (
        ("schema_version", PUBLICATION_SCHEMA),
        ("status", "published"),
        ("scientific_result", False),
        ("complete_paired_population", True),
        ("no_results_dropped", True),
        ("run_id", expected_run_id),
        ("source_git_commit", expected_source_commit),
    ):
        require_equal(
            publication.get(field),
            expected,
            "publication/{}".format(field),
        )
    publisher = publication.get("publisher_slurm")
    if not isinstance(publisher, Mapping):
        raise TransferVerificationError(
            "publication publisher allocation is missing"
        )
    publisher_host = publisher.get("host")
    if (
        not isinstance(publisher_host, str)
        or not publisher_host
        or publisher_host == "worker-3"
        or publisher_host.startswith(("login", "login-restricted"))
    ):
        raise TransferVerificationError(
            "publication publisher host is invalid: {!r}".format(
                publisher_host
            )
        )
    require_equal(
        publication.get("population_array_job_id"),
        expected_population_array_job_id,
        "publication population array job ID",
    )
    require_equal(
        publisher.get("job_id"),
        expected_publisher_job_id,
        "publication publisher job ID",
    )
    require_equal(
        publisher.get("dependency"),
        "afterany:{}".format(publication.get("population_array_job_id")),
        "publication publisher dependency",
    )

    prepublish_path, prepublish, prepublish_sha256 = load_bound_artifact(
        run_root,
        publication.get("prepublish_receipt"),
        "prepublish receipt",
        json_object=True,
    )
    summary_path, summary, summary_sha256 = load_bound_artifact(
        run_root,
        publication.get("summary"),
        "population summary",
        json_object=True,
    )
    gallery_path, _, gallery_sha256 = load_bound_artifact(
        run_root,
        publication.get("gallery"),
        "population gallery",
        json_object=False,
    )
    assert prepublish is not None
    assert summary is not None
    publication_attempt_root = (
        run_root
        / "publication-attempts"
        / "job-{}".format(expected_publisher_job_id)
    )
    for path, expected, label in (
        (
            prepublish_path,
            publication_attempt_root / "prepublish-validation.json",
            "prepublish receipt path",
        ),
        (
            summary_path,
            publication_attempt_root / "population-summary.json",
            "population summary path",
        ),
        (
            gallery_path,
            publication_attempt_root / "gallery" / "index.html",
            "population gallery path",
        ),
    ):
        require_equal(path, expected, label)
    prepublish_payload_sha256 = verify_payload_hash(
        prepublish,
        "receipt_payload_sha256",
        "prepublish receipt",
    )
    require_equal(
        publication["prepublish_receipt"].get("receipt_payload_sha256"),
        prepublish_payload_sha256,
        "publication/prepublish payload SHA-256",
    )
    for field, expected in (
        ("schema_version", PREPUBLISH_SCHEMA),
        ("status", "validated"),
        ("scientific_result", False),
        ("complete_paired_population", True),
        ("no_results_dropped", True),
        ("run_id", expected_run_id),
        ("source_git_commit", expected_source_commit),
    ):
        require_equal(
            prepublish.get(field),
            expected,
            "prepublish/{}".format(field),
        )
    require_equal(
        prepublish.get("population_array_job_id"),
        expected_population_array_job_id,
        "prepublish population array job ID",
    )
    require_equal(
        summary.get("schema_version"),
        SUMMARY_SCHEMA,
        "summary schema",
    )
    require_equal(
        summary.get("status"),
        "complete_population_validated",
        "summary status",
    )
    population = summary.get("population")
    if not isinstance(population, Mapping):
        raise TransferVerificationError("summary population is missing")
    for field, expected in (
        ("cases", EXPECTED_CASES),
        ("arms", len(EXPECTED_ARMS)),
        ("results", EXPECTED_RESULTS),
        ("task_level_groups", EXPECTED_TASKS),
        ("no_results_dropped", True),
    ):
        require_equal(
            population.get(field),
            expected,
            "summary population/{}".format(field),
        )

    try:
        contract, contract_record = official.validate_run_contract(
            run_root=run_root,
            expected_stage="population",
            expected_commit=expected_source_commit,
            config_path=config_path.resolve(),
            manifest_path=manifest_path.resolve(),
            manifest_receipt_path=manifest_receipt_path.resolve(),
            label_manifest_path=labels_path.resolve(),
        )
    except Exception as error:
        raise TransferVerificationError(
            "run contract validation failed: {}".format(error)
        ) from error
    require_equal(
        prepublish.get("run_contract"),
        contract_record,
        "prepublish run-contract record",
    )
    try:
        config, manifests = aggregate.load_protocol(
            config_path.resolve(),
            manifest_receipt_path.resolve(),
            manifest_path.resolve(),
        )
        labels = official._load_population_frozen_label_records(
            labels_path.resolve(),
            manifests=manifests,
        )
    except Exception as error:
        raise TransferVerificationError(
            "protocol or frozen-label validation failed: {}".format(error)
        ) from error
    require_equal(len(manifests), EXPECTED_CASES, "manifest case count")
    require_equal(tuple(config.get("arms", ())), EXPECTED_ARMS, "protocol arms")
    require_equal(len(labels), EXPECTED_CASES, "frozen label count")

    failure_analysis = validate_failure_analysis(
        run_root, publication.get("failure_analysis")
    )
    expected_failure_root = publication_attempt_root / "failure-analysis"
    require_equal(
        Path(failure_analysis["cases"]["path"]).parent,
        expected_failure_root,
        "failure-analysis directory",
    )
    failure_source = failure_analysis["report_value"].get("source")
    if not isinstance(failure_source, Mapping):
        raise TransferVerificationError(
            "failure report source binding is missing"
        )
    for field, expected in (
        ("population_summary_sha256", summary_sha256),
        (
            "population_validation_receipt_sha256",
            prepublish_sha256,
        ),
        (
            "accepted_result_payloads_sha256",
            summary.get("accepted_result_payloads_sha256"),
        ),
    ):
        require_equal(
            failure_source.get(field),
            expected,
            "failure report source/{}".format(field),
        )
    manifest_case_ids = {str(row["case_id"]) for row in manifests}
    require_equal(
        failure_analysis["case_ids"],
        manifest_case_ids,
        "failure-analysis case identities",
    )

    gallery_payload, _ = stable_read_bytes(
        gallery_path, "population gallery"
    )
    try:
        gallery_text = gallery_payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise TransferVerificationError(
            "population gallery is not UTF-8"
        ) from error
    for marker, expected in (
        ('<article class="case-card"', EXPECTED_CASES),
        ('<section class="arm-panel"', EXPECTED_RESULTS),
        ("<video controls", EXPECTED_RESULTS),
        ("Open MP4", EXPECTED_RESULTS),
    ):
        require_equal(
            gallery_text.count(marker),
            expected,
            "population gallery marker {}".format(marker),
        )

    result_artifacts = prepublish.get("result_artifacts")
    diagnostics = prepublish.get("failure_diagnostics")
    if not isinstance(result_artifacts, Mapping) or not isinstance(
        diagnostics, Mapping
    ):
        raise TransferVerificationError(
            "v2 prepublish inventory records are missing"
        )
    require_equal(
        result_artifacts.get("count"),
        EXPECTED_RESULTS,
        "prepublish result count",
    )
    require_equal(
        diagnostics.get("count"),
        EXPECTED_RESULTS,
        "prepublish diagnostic count",
    )
    require_equal(
        diagnostics.get("contact_schema_version"),
        official.CONTACT_SCHEMA_V3,
        "prepublish contact schema",
    )
    require_equal(
        diagnostics.get("contact_model_authority_schema_version"),
        official.CONTACT_MODEL_AUTHORITY_SCHEMA_V2,
        "prepublish model-authority schema",
    )
    return {
        "run_root": run_root,
        "publication_path": publication_path,
        "publication": publication,
        "publication_sha256": publication_sha256,
        "publication_payload_sha256": publication_payload_sha256,
        "prepublish_path": prepublish_path,
        "prepublish": prepublish,
        "prepublish_sha256": prepublish_sha256,
        "prepublish_payload_sha256": prepublish_payload_sha256,
        "summary_path": summary_path,
        "summary": summary,
        "summary_sha256": summary_sha256,
        "gallery_path": gallery_path,
        "gallery_sha256": gallery_sha256,
        "gallery_text": gallery_text,
        "failure_analysis": failure_analysis,
        "contract": contract,
        "contract_record": contract_record,
        "config_path": config_path.resolve(),
        "config": config,
        "manifest_path": manifest_path.resolve(),
        "manifest_receipt_path": manifest_receipt_path.resolve(),
        "labels_path": labels_path.resolve(),
        "manifests": manifests,
        "labels": labels,
    }


def scan_final_artifact_paths(
    run_root: Path, tasks_root: Path
) -> Tuple[Set[str], Set[str]]:
    reject_symlinks(tasks_root, "population tasks")
    observed_results = {
        path.relative_to(run_root).as_posix()
        for path in tasks_root.rglob("result.json")
        if path.is_file()
    }
    observed_videos = {
        path.relative_to(run_root).as_posix()
        for path in tasks_root.rglob("episode.mp4")
        if path.is_file()
    }
    return observed_results, observed_videos


def reconstruct_v2_inventory(chain: Mapping[str, Any]) -> Dict[str, Any]:
    run_root = chain["run_root"]
    tasks_root = real_dir_under(
        run_root, run_root / "tasks", "population tasks"
    )
    reject_symlinks(tasks_root, "population tasks")
    observed_tasks = {path.name for path in tasks_root.iterdir()}
    require_equal(
        observed_tasks,
        {"task-{}".format(index) for index in range(EXPECTED_TASKS)},
        "population task entry set",
    )
    if any(
        not (tasks_root / name).is_dir()
        for name in observed_tasks
    ):
        raise TransferVerificationError(
            "population task entries must all be real directories"
        )
    manifests = chain["manifests"]
    config = chain["config"]
    labels = chain["labels"]
    try:
        specs = official._population_result_specs(config)
    except Exception as error:
        raise TransferVerificationError(
            "population mode mapping is invalid: {}".format(error)
        ) from error
    require_equal(
        tuple(mode for mode, _ in specs),
        EXPECTED_MODES,
        "population mode order",
    )

    publisher_inventory: List[Dict[str, Any]] = []
    diagnostic_inventory: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []
    accepted_payloads: List[Dict[str, str]] = []
    status_counts: Counter = Counter()
    expected_result_paths = set()
    expected_video_paths = set()
    per_task: Dict[str, Dict[str, int]] = {}
    per_arm: Dict[str, Dict[str, int]] = {
        arm: {
            "rows": 0,
            "result_bytes": 0,
            "video_bytes": 0,
            "total_bytes": 0,
        }
        for arm in EXPECTED_ARMS
    }
    for task_index in range(EXPECTED_TASKS):
        per_task["task-{}".format(task_index)] = {
            "rows": 0,
            "result_bytes": 0,
            "video_bytes": 0,
            "total_bytes": 0,
        }

    for manifest in manifests:
        case_id = str(manifest["case_id"])
        ordinal = int(manifest["case_ordinal"])
        task_index = ordinal // CASES_PER_TASK
        task_name = "task-{}".format(task_index)
        result_root = tasks_root / task_name / "results"
        expected_label = labels.get(case_id)
        if not isinstance(expected_label, Mapping):
            raise TransferVerificationError(
                "{} frozen label is missing".format(case_id)
            )
        pair: Dict[Tuple[str, str], Dict[str, Any]] = {}
        pair_items: List[Tuple[Dict[str, Any], Dict[str, Any], Path, Path]] = []
        for policy_mode, arm in specs:
            result_path = result_root / policy_mode / case_id / "result.json"
            try:
                result, item = official._validate_population_result(
                    path=result_path,
                    task_results_root=result_root,
                    config=config,
                    manifest=manifest,
                    expected_commit=chain["publication"][
                        "source_git_commit"
                    ],
                )
                official._validate_population_result_binding(
                    result,
                    task_index=task_index,
                    policy_mode=policy_mode,
                    expected_arm=arm,
                    expected_label_record=expected_label,
                )
                diagnostic_validation = (
                    failure_validation.validate_diagnostic_result(
                        result,
                        output_root=result_root,
                        require_ready_geometry=False,
                    )
                )
                compact = official._compact_population_diagnostic_evidence(
                    result,
                    diagnostic_validation=diagnostic_validation,
                )
            except Exception as error:
                raise TransferVerificationError(
                    "{}/{}/v2 validation failed: {}".format(
                        case_id, policy_mode, error
                    )
                ) from error
            item["relative_result_path"] = str(
                result_path.relative_to(run_root)
            )
            item["case_ordinal"] = ordinal
            item["task_index"] = task_index
            item["policy_mode"] = policy_mode
            item["diagnostics"] = compact
            video = result.get("video")
            if not isinstance(video, Mapping):
                raise TransferVerificationError(
                    "{}/{} video record is missing".format(
                        case_id, policy_mode
                    )
                )
            raw_video_path = video.get("path")
            if (
                not isinstance(raw_video_path, str)
                or Path(raw_video_path).is_absolute()
                or ".." in Path(raw_video_path).parts
            ):
                raise TransferVerificationError(
                    "{}/{} video path is unsafe".format(
                        case_id, policy_mode
                    )
                )
            video_path = result_root / raw_video_path
            pair[(case_id, arm)] = result
            pair_items.append((result, item, result_path, video_path))
        try:
            aggregate.validate_pairs(
                config=config,
                manifests=[manifest],
                results=pair,
            )
        except Exception as error:
            raise TransferVerificationError(
                "{} paired result validation failed: {}".format(
                    case_id, error
                )
            ) from error

        for result, item, result_path, video_path in pair_items:
            arm = str(item["arm"])
            policy_mode = str(item["policy_mode"])
            result_relative = result_path.relative_to(run_root).as_posix()
            video_relative = video_path.relative_to(run_root).as_posix()
            result_sha256, result_bytes = stable_file_sha256_and_size(
                result_path, "{}/{} result".format(case_id, policy_mode)
            )
            video_sha256, video_bytes = stable_file_sha256_and_size(
                video_path, "{}/{} video".format(case_id, policy_mode)
            )
            require_equal(
                result_sha256,
                item["result_sha256"],
                "{}/{} result SHA-256".format(case_id, policy_mode),
            )
            require_equal(
                video_sha256,
                item["video_sha256"],
                "{}/{} video SHA-256".format(case_id, policy_mode),
            )
            if result_bytes < 1 or video_bytes < 1:
                raise TransferVerificationError(
                    "{}/{} result or video is empty".format(
                        case_id, policy_mode
                    )
                )
            expected_result_paths.add(result_relative)
            expected_video_paths.add(video_relative)
            publisher_inventory.append(item)
            diagnostic_inventory.append(
                {
                    "case_ordinal": ordinal,
                    "task_index": task_index,
                    **dict(item["diagnostics"]),
                }
            )
            payload_sha256 = require_sha256(
                item["result_payload_sha256"],
                "{}/{} result payload".format(case_id, policy_mode),
            )
            if result.get("status") not in ACCEPTED_STATUSES:
                raise TransferVerificationError(
                    "{}/{} has nonterminal result status {!r}".format(
                        case_id, policy_mode, result.get("status")
                    )
                )
            accepted_payloads.append(
                {
                    "case_id": case_id,
                    "arm": arm,
                    "result_payload_sha256": payload_sha256,
                }
            )
            video = result["video"]
            row = {
                "schema_version": ROW_SCHEMA,
                "case_ordinal": ordinal,
                "task_index": task_index,
                "task_level_group_id": manifest[
                    "task_level_group_id"
                ],
                "case_id": case_id,
                "arm": arm,
                "mode": policy_mode,
                "status": result["status"],
                "result": {
                    "path": result_relative,
                    "bytes": result_bytes,
                    "sha256": result_sha256,
                    "result_payload_sha256": payload_sha256,
                },
                "video": {
                    "path": video_relative,
                    "bytes": video_bytes,
                    "sha256": video_sha256,
                    "frames": video["frames"],
                    "fps": video["fps"],
                    "complete_episode": video["complete_episode"],
                },
                "publisher_v2_inventory_item": item,
            }
            rows.append(row)
            status_counts[str(result["status"])] += 1
            for summary in (per_task[task_name], per_arm[arm]):
                summary["rows"] += 1
                summary["result_bytes"] += result_bytes
                summary["video_bytes"] += video_bytes
                summary["total_bytes"] += result_bytes + video_bytes

        del pair
        del pair_items

    observed_results, observed_videos = scan_final_artifact_paths(
        run_root, tasks_root
    )
    require_equal(
        observed_results, expected_result_paths, "final result path set"
    )
    require_equal(
        observed_videos, expected_video_paths, "final video path set"
    )
    require_equal(len(rows), EXPECTED_RESULTS, "transfer row count")
    require_equal(
        len({row["video"]["path"] for row in rows}),
        EXPECTED_RESULTS,
        "unique transfer video count",
    )

    publisher_inventory_sha256 = sha256_bytes(
        canonical_json_bytes(publisher_inventory)
    )
    diagnostic_inventory_sha256 = sha256_bytes(
        canonical_json_bytes(diagnostic_inventory)
    )
    accepted_payloads.sort(key=lambda row: (row["case_id"], row["arm"]))
    accepted_payloads_sha256 = sha256_bytes(
        canonical_json_bytes(accepted_payloads)
    )
    prepublish = chain["prepublish"]
    require_equal(
        publisher_inventory_sha256,
        prepublish["result_artifacts"]["inventory_sha256"],
        "v2 publisher result inventory SHA-256",
    )
    require_equal(
        diagnostic_inventory_sha256,
        prepublish["failure_diagnostics"]["inventory_sha256"],
        "v2 publisher diagnostic inventory SHA-256",
    )
    require_equal(
        accepted_payloads_sha256,
        chain["summary"]["accepted_result_payloads_sha256"],
        "summary accepted-result payloads SHA-256",
    )
    require_equal(
        dict(sorted(status_counts.items())),
        dict(
            sorted(
                prepublish["result_artifacts"]["status_counts"].items()
            )
        ),
        "v2 publisher result status counts",
    )

    gallery_sources = re.findall(
        r'<source src="([^"]+)" type="video/mp4">', chain["gallery_text"]
    )
    gallery_links = re.findall(
        r'<a href="([^"]+)">Open MP4</a>', chain["gallery_text"]
    )
    expected_hrefs = []
    gallery_root = chain["gallery_path"].parent
    for row in rows:
        target = run_root / row["video"]["path"]
        relative = Path(os.path.relpath(target, start=gallery_root))
        expected_hrefs.append(
            html.escape(
                quote(relative.as_posix(), safe="/"), quote=True
            )
        )
    expected_href_counter = Counter(expected_hrefs)
    for name, observed_hrefs in (
        ("gallery video source inventory", gallery_sources),
        ("gallery Open-MP4 inventory", gallery_links),
    ):
        observed_counter = Counter(observed_hrefs)
        if observed_counter != expected_href_counter:
            missing = list(
                (expected_href_counter - observed_counter).elements()
            )[:5]
            extra = list(
                (observed_counter - expected_href_counter).elements()
            )[:5]
            raise TransferVerificationError(
                "{} mismatch: expected_count={} observed_count={} "
                "missing_sample={!r} extra_sample={!r}".format(
                    name,
                    sum(expected_href_counter.values()),
                    sum(observed_counter.values()),
                    missing,
                    extra,
                )
            )

    mode_order = {
        mode: index for index, mode in enumerate(EXPECTED_MODES)
    }
    rows.sort(
        key=lambda row: (
            row["case_ordinal"],
            mode_order[row["mode"]],
        )
    )
    total_result_bytes = sum(row["result"]["bytes"] for row in rows)
    total_video_bytes = sum(row["video"]["bytes"] for row in rows)
    return {
        "rows": rows,
        "publisher_inventory_sha256": publisher_inventory_sha256,
        "diagnostic_inventory_sha256": diagnostic_inventory_sha256,
        "accepted_payloads_sha256": accepted_payloads_sha256,
        "status_counts": dict(sorted(status_counts.items())),
        "bytes": {
            "result_json": total_result_bytes,
            "episode_mp4": total_video_bytes,
            "combined": total_result_bytes + total_video_bytes,
        },
        "per_task": per_task,
        "per_arm": per_arm,
        "expected_result_paths": expected_result_paths,
        "expected_video_paths": expected_video_paths,
    }


def revalidate_artifacts(
    chain: Mapping[str, Any],
    enumeration: Mapping[str, Any],
) -> None:
    frozen = [
        (chain["publication_path"], chain["publication_sha256"], "publication"),
        (chain["prepublish_path"], chain["prepublish_sha256"], "prepublish"),
        (chain["summary_path"], chain["summary_sha256"], "summary"),
        (chain["gallery_path"], chain["gallery_sha256"], "gallery"),
    ]
    failure = chain["failure_analysis"]
    for key in ("cases", "report", "markdown"):
        frozen.append(
            (
                Path(failure[key]["path"]),
                failure[key]["sha256"],
                "failure {}".format(key),
            )
        )
    frozen.extend(
        (
            (
                chain["config_path"],
                chain["contract"]["config_sha256"],
                "protocol config",
            ),
            (
                chain["manifest_path"],
                chain["contract"]["manifest_sha256"],
                "population manifest",
            ),
            (
                chain["manifest_receipt_path"],
                chain["contract"]["manifest_receipt_sha256"],
                "manifest receipt",
            ),
            (
                chain["labels_path"],
                chain["contract"]["label_manifest_sha256"],
                "label manifest",
            ),
        )
    )
    for path, expected_sha256, label in frozen:
        require_equal(
            stable_file_sha256_and_size(Path(path), label)[0],
            expected_sha256,
            "{} changed after enumeration".format(label),
        )
    run_root = chain["run_root"]
    tasks_root = real_dir_under(
        run_root, run_root / "tasks", "population tasks revalidation"
    )
    expected_task_entries = {
        "task-{}".format(index) for index in range(EXPECTED_TASKS)
    }
    require_equal(
        {path.name for path in tasks_root.iterdir()},
        expected_task_entries,
        "population task entry set changed after enumeration",
    )
    observed_results, observed_videos = scan_final_artifact_paths(
        run_root, tasks_root
    )
    require_equal(
        observed_results,
        enumeration["expected_result_paths"],
        "final result path set changed after enumeration",
    )
    require_equal(
        observed_videos,
        enumeration["expected_video_paths"],
        "final video path set changed after enumeration",
    )
    failure_root = Path(chain["failure_analysis"]["cases"]["path"]).parent
    reject_symlinks(failure_root, "failure-analysis revalidation")
    require_equal(
        {path.name for path in failure_root.iterdir()},
        {
            Path(chain["failure_analysis"][key]["path"]).name
            for key in ("cases", "report", "markdown")
        },
        "failure-analysis entry set changed after enumeration",
    )
    for row in enumeration["rows"]:
        for kind in ("result", "video"):
            record = row[kind]
            path = real_file_under(
                run_root,
                run_root / record["path"],
                "{}/{}/{} revalidation".format(
                    row["case_id"], row["mode"], kind
                ),
            )
            observed_sha256, observed_bytes = (
                stable_file_sha256_and_size(path, kind)
            )
            require_equal(
                observed_sha256,
                record["sha256"],
                "{}/{}/{} changed after enumeration".format(
                    row["case_id"], row["mode"], kind
                ),
            )
            require_equal(
                observed_bytes,
                record["bytes"],
                "{}/{}/{} size changed after enumeration".format(
                    row["case_id"], row["mode"], kind
                ),
            )


def write_exclusive(path: Path, payload: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def fsync_directory(path: Path) -> None:
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def rename_noreplace(source: Path, destination: Path) -> None:
    if not sys.platform.startswith("linux"):
        if destination.exists() or destination.is_symlink():
            raise TransferVerificationError(
                "output directory already exists: {}".format(destination)
            )
        os.rename(str(source), str(destination))
        return
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise TransferVerificationError(
            "renameat2 is required for no-replace publication"
        )
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(str(source)),
        -100,
        os.fsencode(str(destination)),
        1,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise TransferVerificationError(
            "output directory already exists: {}".format(destination)
        )
    raise TransferVerificationError(
        "renameat2 failed: {}".format(os.strerror(error_number))
    )


def publish_bundle(
    *,
    output_dir: Path,
    chain: Mapping[str, Any],
    enumeration: Mapping[str, Any],
    allocation: Mapping[str, Any],
    verifier_identity: Mapping[str, Any],
    generated_utc: Optional[str] = None,
) -> Dict[str, Any]:
    publisher = chain["publication"].get("publisher_slurm")
    if not isinstance(publisher, Mapping):
        raise TransferVerificationError(
            "publication publisher allocation is missing"
        )
    validated_allocation = validate_allocation_record(
        allocation,
        expected_publisher_job_id=str(publisher.get("job_id", "")),
    )
    validated_verifier = revalidate_verifier_identity(verifier_identity)
    output_dir = lexical_absolute(output_dir)
    if output_dir.exists() or output_dir.is_symlink():
        raise TransferVerificationError(
            "output directory already exists: {}".format(output_dir)
        )
    parent = output_dir.parent
    if parent.is_symlink() or not parent.is_dir():
        raise TransferVerificationError(
            "output parent must be an existing real directory"
        )
    physical_parent = parent.resolve(strict=True)
    physical_run_root = chain["run_root"].resolve(strict=True)
    physical_output = physical_parent / output_dir.name
    try:
        physical_output.relative_to(physical_run_root)
    except ValueError:
        pass
    else:
        raise TransferVerificationError(
            "output directory resolves inside immutable run root"
        )
    parent = physical_parent
    output_dir = physical_output

    revalidate_artifacts(chain, enumeration)
    revalidate_verifier_identity(validated_verifier)
    rows = enumeration["rows"]
    manifest_payload = b"".join(
        canonical_json_bytes(row) + b"\n" for row in rows
    )
    manifest_sha256 = sha256_bytes(manifest_payload)
    generated = generated_utc or (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
    failure = chain["failure_analysis"]
    receipt: Dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "status": "validated",
        "scientific_result": False,
        "complete_paired_population": True,
        "no_results_dropped": True,
        "generated_utc": generated,
        "allocation": validated_allocation,
        "verifier": validated_verifier,
        "source": {
            "run_id": chain["publication"]["run_id"],
            "source_git_commit": chain["publication"]["source_git_commit"],
            "run_root": str(chain["run_root"]),
            "publication_receipt": {
                "path": str(chain["publication_path"]),
                "sha256": chain["publication_sha256"],
                "receipt_payload_sha256": chain[
                    "publication_payload_sha256"
                ],
            },
            "prepublish_receipt": {
                "path": str(chain["prepublish_path"]),
                "sha256": chain["prepublish_sha256"],
                "receipt_payload_sha256": chain[
                    "prepublish_payload_sha256"
                ],
            },
            "summary": {
                "path": str(chain["summary_path"]),
                "sha256": chain["summary_sha256"],
            },
            "gallery": {
                "path": str(chain["gallery_path"]),
                "sha256": chain["gallery_sha256"],
                "video_controls": EXPECTED_RESULTS,
            },
            "failure_analysis": {
                key: {
                    field: value
                    for field, value in failure[key].items()
                    if field != "path"
                }
                for key in ("cases", "report", "markdown")
            },
        },
        "transfer_manifest": {
            "path": "video-transfer-manifest.jsonl",
            "rows": len(rows),
            "sha256": manifest_sha256,
            "canonical_rows_sha256": sha256_bytes(
                canonical_json_bytes(rows)
            ),
        },
        "population": {
            "cases": EXPECTED_CASES,
            "arms": len(EXPECTED_ARMS),
            "rows": EXPECTED_RESULTS,
            "task_groups": EXPECTED_TASKS,
            "status_counts": enumeration["status_counts"],
        },
        "cross_checks": {
            "publisher_v2_result_inventory_sha256": enumeration[
                "publisher_inventory_sha256"
            ],
            "publisher_v2_diagnostic_inventory_sha256": enumeration[
                "diagnostic_inventory_sha256"
            ],
            "accepted_result_payloads_sha256": enumeration[
                "accepted_payloads_sha256"
            ],
            "all_failure_artifacts_rehashed": True,
            "all_gallery_video_links_matched": True,
            "all_result_and_video_files_rehashed_twice": True,
        },
        "bytes": enumeration["bytes"],
        "per_task": enumeration["per_task"],
        "per_arm": enumeration["per_arm"],
        "transport": {
            "videos_copied": False,
            "login_node_bulk_transfer_authorized": False,
            "required_bulk_route": (
                "administrator-approved SFTP or object storage"
            ),
        },
    }
    receipt["receipt_payload_sha256"] = sha256_bytes(
        canonical_json_bytes(receipt)
    )
    receipt_payload = (
        json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")

    staging = Path(
        tempfile.mkdtemp(
            prefix=".{}.stage-".format(output_dir.name), dir=str(parent)
        )
    )
    manifest_path = staging / "video-transfer-manifest.jsonl"
    receipt_path = staging / "video-transfer-receipt.json"
    published = False
    try:
        write_exclusive(manifest_path, manifest_payload)
        write_exclusive(receipt_path, receipt_payload)
        require_equal(
            stable_file_sha256_and_size(
                manifest_path, "staged transfer manifest"
            ),
            (manifest_sha256, len(manifest_payload)),
            "staged transfer manifest identity",
        )
        require_equal(
            stable_file_sha256_and_size(
                receipt_path, "staged transfer receipt"
            ),
            (sha256_bytes(receipt_payload), len(receipt_payload)),
            "staged transfer receipt identity",
        )
        revalidate_artifacts(chain, enumeration)
        revalidate_verifier_identity(validated_verifier)
        fsync_directory(staging)
        rename_noreplace(staging, output_dir)
        published = True
        fsync_directory(parent)
    except Exception:
        if not published:
            for path in (receipt_path, manifest_path):
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
            try:
                staging.rmdir()
            except FileNotFoundError:
                pass
        raise
    return {
        "output_dir": str(output_dir),
        "manifest_path": str(output_dir / manifest_path.name),
        "manifest_sha256": manifest_sha256,
        "receipt_path": str(output_dir / receipt_path.name),
        "receipt_sha256": sha256_bytes(receipt_payload),
        "receipt_payload_sha256": receipt["receipt_payload_sha256"],
        "receipt": receipt,
    }


def generate_transfer_bundle(
    *,
    run_root: Path,
    publication_receipt: Path,
    expected_publication_receipt_sha256: str,
    expected_run_id: str,
    expected_source_commit: str,
    expected_population_array_job_id: str,
    expected_publisher_job_id: str,
    config_path: Path,
    manifest_path: Path,
    manifest_receipt_path: Path,
    labels_path: Path,
    output_dir: Path,
    allocation: Optional[Mapping[str, Any]] = None,
    verifier_identity: Optional[Mapping[str, Any]] = None,
    generated_utc: Optional[str] = None,
) -> Dict[str, Any]:
    selected_allocation = validate_allocation_record(
        (
            dict(allocation)
            if allocation is not None
            else validate_cpu_allocation(
                os.environ,
                expected_publisher_job_id=expected_publisher_job_id,
            )
        ),
        expected_publisher_job_id=expected_publisher_job_id,
    )
    if verifier_identity is None:
        raise TransferVerificationError(
            "a validated verifier identity is required"
        )
    selected_verifier_identity = revalidate_verifier_identity(
        verifier_identity
    )
    chain = validate_publication_chain(
        run_root=run_root,
        publication_receipt=publication_receipt,
        expected_publication_receipt_sha256=(
            expected_publication_receipt_sha256
        ),
        expected_run_id=expected_run_id,
        expected_source_commit=expected_source_commit,
        expected_population_array_job_id=(
            expected_population_array_job_id
        ),
        expected_publisher_job_id=expected_publisher_job_id,
        config_path=config_path,
        manifest_path=manifest_path,
        manifest_receipt_path=manifest_receipt_path,
        labels_path=labels_path,
    )
    enumeration = reconstruct_v2_inventory(chain)
    return publish_bundle(
        output_dir=output_dir,
        chain=chain,
        enumeration=enumeration,
        allocation=selected_allocation,
        verifier_identity=selected_verifier_identity,
        generated_utc=generated_utc,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify the published VLSA/AEGIS v2 population and atomically "
            "publish its exact 3,200-row video-transfer manifest"
        )
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--publication-receipt", type=Path, required=True)
    parser.add_argument(
        "--expected-publication-receipt-sha256", required=True
    )
    parser.add_argument("--expected-run-id", required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument(
        "--expected-population-array-job-id", required=True
    )
    parser.add_argument("--expected-publisher-job-id", required=True)
    parser.add_argument("--expected-verifier-sha256", required=True)
    parser.add_argument("--expected-verifier-git-commit", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-receipt", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    allocation = validate_cpu_allocation(
        os.environ,
        expected_publisher_job_id=args.expected_publisher_job_id,
    )
    verifier_identity = validate_verifier_identity(
        expected_sha256=args.expected_verifier_sha256,
        expected_git_commit=args.expected_verifier_git_commit,
    )
    outcome = generate_transfer_bundle(
        run_root=args.run_root,
        publication_receipt=args.publication_receipt,
        expected_publication_receipt_sha256=(
            args.expected_publication_receipt_sha256
        ),
        expected_run_id=args.expected_run_id,
        expected_source_commit=args.expected_source_commit,
        expected_population_array_job_id=(
            args.expected_population_array_job_id
        ),
        expected_publisher_job_id=args.expected_publisher_job_id,
        config_path=args.config,
        manifest_path=args.manifest,
        manifest_receipt_path=args.manifest_receipt,
        labels_path=args.labels,
        output_dir=args.output_dir,
        allocation=allocation,
        verifier_identity=verifier_identity,
    )
    print(
        json.dumps(
            {
                key: value
                for key, value in outcome.items()
                if key != "receipt"
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except TransferVerificationError as error:
        raise SystemExit("verification failed: {}".format(error))
