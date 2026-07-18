#!/usr/bin/env python3
"""Publish receipt-bound decision analysis over accepted analysis-v2.

This module never changes the immutable population or analysis-v2 artifacts.
It validates their exact byte identities first, derives the accepted v3 case
ledger and report into an unused directory, and writes the v3 receipt last.
An absent receipt therefore means the derivation is incomplete.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping, Sequence

try:
    from analysis import aggregate_safelibero_aegis as aggregate
    from analysis import build_aegis_failure_report_v3 as v3
    from analysis import render_safelibero_table1_comparison as table_report
except ImportError:  # pragma: no cover - direct script execution fallback
    import aggregate_safelibero_aegis as aggregate  # type: ignore[no-redef]
    import build_aegis_failure_report_v3 as v3  # type: ignore[no-redef]
    import render_safelibero_table1_comparison as table_report  # type: ignore[no-redef]


RECEIPT_SCHEMA = "vlsa_table1_postpublication_analysis_v3_receipt.v1"
RECEIPT_STATUS = "published_derived_analysis_v3"
EXPECTED_V3_IMPLEMENTATION_COMMIT = (
    "1c92370cb5b1278a4fcdda81332ec8eed9a49f8b"
)
EXPECTED_RUNTIME_COMMIT = (
    "1592aa59361f431ba96c6ddcbebcb596f6c20853"
)
EXPECTED_RUN_ID = "vlsa-table1-contact-authority-population-20260718a"
EXPECTED_POPULATION_ARRAY_JOB_ID = "28609"
EXPECTED_PUBLISHER_JOB_ID = "28610"
EXPECTED_V2_JOB_NAME = "vlsa-a2-p28610"
EXPECTED_V3_JOB_NAME = "vlsa-a3-p28610"
EXPECTED_CASES = 1600
EXPECTED_RESULTS = 3200
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


class PostpublicationV3Error(RuntimeError):
    """Raised when an accepted upstream identity or result is incomplete."""


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PostpublicationV3Error(f"{label} must be an object")
    return value


def _sha(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or SHA_RE.fullmatch(value) is None:
        raise PostpublicationV3Error(f"{label} must be lowercase SHA-256")
    return value


def _commit(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or COMMIT_RE.fullmatch(value) is None:
        raise PostpublicationV3Error(
            f"{label} must be a 40-character lowercase Git object ID"
        )
    return value


def _job_id(value: Any, *, label: str) -> str:
    text = str(value)
    if not text.isdigit() or int(text) <= 0:
        raise PostpublicationV3Error(f"{label} must be a positive job ID")
    return text


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _load_json(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        return table_report.load_json(path, label=label)
    except table_report.TableReportError as error:
        raise PostpublicationV3Error(str(error)) from error


def _require_file_hash(
    path: Path, expected_sha256: str, *, label: str
) -> None:
    expected = _sha(expected_sha256, label=f"{label} expected SHA")
    if path.is_symlink() or not path.is_file():
        raise PostpublicationV3Error(
            f"{label} is missing, non-regular, or symlinked"
        )
    if sha256_path(path) != expected:
        raise PostpublicationV3Error(f"{label} SHA-256 differs")


def _load_tsv(path: Path, *, label: str) -> dict[str, str]:
    if path.is_symlink() or not path.is_file():
        raise PostpublicationV3Error(
            f"{label} is missing, non-regular, or symlinked"
        )
    output: dict[str, str] = {}
    for line_number, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if raw.count("\t") != 1:
            raise PostpublicationV3Error(
                f"{label} line {line_number} is not exact two-column TSV"
            )
        key, value = raw.split("\t")
        if not key or key in output or not value:
            raise PostpublicationV3Error(
                f"{label} line {line_number} is ambiguous"
            )
        output[key] = value
    return output


def _validate_v2_control_receipts(
    *,
    analysis_job_id: str,
    analysis_git_commit: str,
    submission_path: Path,
    submission_sha256: str,
    release_path: Path,
    release_sha256: str,
) -> dict[str, Any]:
    job_id = _job_id(analysis_job_id, label="analysis-v2 job ID")
    git_commit = _commit(
        analysis_git_commit, label="analysis-v2 Git commit"
    )
    _require_file_hash(
        submission_path,
        submission_sha256,
        label="analysis-v2 submission receipt",
    )
    _require_file_hash(
        release_path,
        release_sha256,
        label="analysis-v2 release receipt",
    )
    submission = _load_tsv(
        submission_path, label="analysis-v2 submission receipt"
    )
    release = _load_tsv(
        release_path, label="analysis-v2 release receipt"
    )
    required_submission = {
        "schema_version": (
            "vlsa_postpublication_analysis_v2_submission_receipt.v1"
        ),
        "status": "held_validated",
        "job_id": job_id,
        "job_name": EXPECTED_V2_JOB_NAME,
        "population_array_job_id": EXPECTED_POPULATION_ARRAY_JOB_ID,
        "publisher_job_id": EXPECTED_PUBLISHER_JOB_ID,
        "dependency": f"afterok:{EXPECTED_PUBLISHER_JOB_ID}",
        "source_git_commit": EXPECTED_RUNTIME_COMMIT,
        "analysis_git_commit": git_commit,
        "cpu_only": "true",
        "held_before_release": "true",
        "output_unused": "true",
    }
    required_release = {
        "schema_version": (
            "vlsa_postpublication_analysis_v2_release_receipt.v1"
        ),
        "status": "released",
        "job_id": job_id,
        "job_name": EXPECTED_V2_JOB_NAME,
        "population_array_job_id": EXPECTED_POPULATION_ARRAY_JOB_ID,
        "publisher_job_id": EXPECTED_PUBLISHER_JOB_ID,
        "dependency": f"afterok:{EXPECTED_PUBLISHER_JOB_ID}",
        "output_unused_before_release": "true",
    }
    for key, expected in required_submission.items():
        if submission.get(key) != expected:
            raise PostpublicationV3Error(
                f"analysis-v2 submission receipt {key} differs"
            )
    for key, expected in required_release.items():
        if release.get(key) != expected:
            raise PostpublicationV3Error(
                f"analysis-v2 release receipt {key} differs"
            )
    if release.get("submission_receipt_sha256") != sha256_path(
        submission_path
    ):
        raise PostpublicationV3Error(
            "analysis-v2 release does not bind its submission receipt"
        )
    return {
        "analysis_job_id": job_id,
        "analysis_git_commit": git_commit,
        "submission_receipt_sha256": sha256_path(submission_path),
        "release_receipt_sha256": sha256_path(release_path),
    }


def _validate_v3_control_receipts(
    *,
    analysis_v2_job_id: str,
    analysis_v2_git_commit: str,
    analysis_v3_job_id: str,
    analysis_v3_git_commit: str,
    output_root: Path,
    submission_path: Path,
    submission_sha256: str,
    release_path: Path,
    release_sha256: str,
    expected_summary_v2_sha256: str,
    expected_analysis_v2_receipt_sha256: str,
    expected_v2_submission_receipt_sha256: str,
    expected_v2_release_receipt_sha256: str,
    expected_v1_publication_receipt_sha256: str,
    expected_prepublish_receipt_sha256: str,
    expected_config_sha256: str,
    expected_manifest_sha256: str,
    expected_manifest_receipt_sha256: str,
    expected_v3_module_sha256: str,
    expected_builder_sha256: str,
    expected_runner_sha256: str,
    expected_sbatch_sha256: str,
    expected_submit_helper_sha256: str,
) -> dict[str, Any]:
    v2_job_id = _job_id(
        analysis_v2_job_id, label="analysis-v2 job ID"
    )
    v3_job_id = _job_id(
        analysis_v3_job_id, label="analysis-v3 job ID"
    )
    v2_commit = _commit(
        analysis_v2_git_commit, label="analysis-v2 Git commit"
    )
    v3_commit = _commit(
        analysis_v3_git_commit, label="analysis-v3 Git commit"
    )
    _require_file_hash(
        submission_path,
        submission_sha256,
        label="analysis-v3 submission receipt",
    )
    _require_file_hash(
        release_path,
        release_sha256,
        label="analysis-v3 release receipt",
    )
    submission = _load_tsv(
        submission_path, label="analysis-v3 submission receipt"
    )
    release = _load_tsv(
        release_path, label="analysis-v3 release receipt"
    )
    required_submission = {
        "schema_version": (
            "vlsa_postpublication_analysis_v3_submission_receipt.v1"
        ),
        "status": "held_validated",
        "run_id": EXPECTED_RUN_ID,
        "job_id": v3_job_id,
        "job_name": EXPECTED_V3_JOB_NAME,
        "population_array_job_id": EXPECTED_POPULATION_ARRAY_JOB_ID,
        "publisher_job_id": EXPECTED_PUBLISHER_JOB_ID,
        "analysis_v2_job_id": v2_job_id,
        "dependency": f"afterok:{v2_job_id}",
        "source_git_commit": EXPECTED_RUNTIME_COMMIT,
        "analysis_v2_git_commit": v2_commit,
        "analysis_v3_git_commit": v3_commit,
        "summary_v2_sha256": _sha(
            expected_summary_v2_sha256, label="analysis-v2 summary SHA"
        ),
        "analysis_v2_receipt_sha256": _sha(
            expected_analysis_v2_receipt_sha256,
            label="analysis-v2 receipt SHA",
        ),
        "v2_submission_receipt_sha256": _sha(
            expected_v2_submission_receipt_sha256,
            label="analysis-v2 submission receipt SHA",
        ),
        "v2_release_receipt_sha256": _sha(
            expected_v2_release_receipt_sha256,
            label="analysis-v2 release receipt SHA",
        ),
        "v1_publication_receipt_sha256": _sha(
            expected_v1_publication_receipt_sha256,
            label="v1 publication receipt SHA",
        ),
        "prepublish_receipt_sha256": _sha(
            expected_prepublish_receipt_sha256,
            label="prepublish receipt SHA",
        ),
        "config_sha256": _sha(
            expected_config_sha256, label="config SHA"
        ),
        "manifest_sha256": _sha(
            expected_manifest_sha256, label="manifest SHA"
        ),
        "manifest_receipt_sha256": _sha(
            expected_manifest_receipt_sha256,
            label="manifest receipt SHA",
        ),
        "v3_module_sha256": _sha(
            expected_v3_module_sha256, label="analysis-v3 module SHA"
        ),
        "builder_sha256": _sha(
            expected_builder_sha256, label="analysis-v3 builder SHA"
        ),
        "runner_sha256": _sha(
            expected_runner_sha256, label="analysis-v3 runner SHA"
        ),
        "sbatch_sha256": _sha(
            expected_sbatch_sha256, label="analysis-v3 SBatch SHA"
        ),
        "submit_helper_sha256": _sha(
            expected_submit_helper_sha256,
            label="analysis-v3 submit helper SHA",
        ),
        "output_dir": str(output_root),
        "partition": "main",
        "account": "normal",
        "qos": "normal",
        "nodes": "1",
        "ntasks": "1",
        "cpus_per_task": "4",
        "memory": "32G",
        "time_limit": "04:00:00",
        "gpus": "0",
        "exclude": "worker-3",
        "cpu_only": "true",
        "held_before_release": "true",
        "output_unused": "true",
    }
    required_release = {
        "schema_version": (
            "vlsa_postpublication_analysis_v3_release_receipt.v1"
        ),
        "status": "released",
        "run_id": EXPECTED_RUN_ID,
        "job_id": v3_job_id,
        "job_name": EXPECTED_V3_JOB_NAME,
        "population_array_job_id": EXPECTED_POPULATION_ARRAY_JOB_ID,
        "publisher_job_id": EXPECTED_PUBLISHER_JOB_ID,
        "analysis_v2_job_id": v2_job_id,
        "dependency": f"afterok:{v2_job_id}",
        "output_unused_before_release": "true",
    }
    for key, expected in required_submission.items():
        if submission.get(key) != expected:
            raise PostpublicationV3Error(
                f"analysis-v3 submission receipt {key} differs"
            )
    for key, expected in required_release.items():
        if release.get(key) != expected:
            raise PostpublicationV3Error(
                f"analysis-v3 release receipt {key} differs"
            )
    if release.get("release_action") not in {
        "released_exact_held_job",
        "recovered_after_prior_release",
    }:
        raise PostpublicationV3Error(
            "analysis-v3 release receipt action is invalid"
        )
    for key in (
        "observed_state_before_action",
        "observed_reason_before_action",
        "observed_state_after_action",
        "observed_reason_after_action",
    ):
        if not release.get(key):
            raise PostpublicationV3Error(
                f"analysis-v3 release receipt {key} is missing"
            )
    if "Held" in release["observed_reason_after_action"]:
        raise PostpublicationV3Error(
            "analysis-v3 release receipt retains a held state"
        )
    if release.get("submission_receipt_sha256") != sha256_path(
        submission_path
    ):
        raise PostpublicationV3Error(
            "analysis-v3 release does not bind its submission receipt"
        )
    return {
        "analysis_job_id": v3_job_id,
        "analysis_git_commit": v3_commit,
        "job_name": EXPECTED_V3_JOB_NAME,
        "dependency": f"afterok:{v2_job_id}",
        "run_id": EXPECTED_RUN_ID,
        "population_array_job_id": EXPECTED_POPULATION_ARRAY_JOB_ID,
        "publisher_job_id": EXPECTED_PUBLISHER_JOB_ID,
        "population_runtime_git_commit": EXPECTED_RUNTIME_COMMIT,
        "output_root": str(output_root),
        "resources": {
            "partition": "main",
            "account": "normal",
            "qos": "normal",
            "nodes": 1,
            "ntasks": 1,
            "cpus_per_task": 4,
            "memory": "32G",
            "time_limit": "04:00:00",
            "gpus": 0,
            "exclude": "worker-3",
        },
        "reviewed_source_sha256": {
            "v3_module": expected_v3_module_sha256,
            "receipt_builder": expected_builder_sha256,
            "allocation_runner": expected_runner_sha256,
            "sbatch": expected_sbatch_sha256,
            "submit_helper": expected_submit_helper_sha256,
        },
        "submission_receipt": _artifact(submission_path),
        "release_receipt": _artifact(release_path),
    }


def _validate_v2_publication(
    *,
    summary_path: Path,
    summary_sha256: str,
    receipt_path: Path,
    receipt_sha256: str,
    v1_publication_receipt_path: Path,
    v1_publication_receipt_sha256: str,
    prepublish_receipt_path: Path,
    prepublish_receipt_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    _require_file_hash(
        summary_path, summary_sha256, label="analysis-v2 summary"
    )
    _require_file_hash(
        receipt_path, receipt_sha256, label="analysis-v2 receipt"
    )
    _require_file_hash(
        v1_publication_receipt_path,
        v1_publication_receipt_sha256,
        label="v1 publication receipt",
    )
    _require_file_hash(
        prepublish_receipt_path,
        prepublish_receipt_sha256,
        label="v1 prepublish receipt",
    )
    summary, summary_raw = _load_json(
        summary_path, label="analysis-v2 summary"
    )
    receipt, _ = _load_json(
        receipt_path, label="analysis-v2 receipt"
    )
    try:
        table_report.validate_analysis_receipt(
            receipt,
            summary=summary,
            summary_path=summary_path,
            summary_raw=summary_raw,
        )
    except table_report.TableReportError as error:
        raise PostpublicationV3Error(str(error)) from error
    if (
        summary.get("schema_version") != v3.SOURCE_SUMMARY_SCHEMA
        or summary.get("status") != v3.SOURCE_SUMMARY_STATUS
    ):
        raise PostpublicationV3Error(
            "analysis-v2 summary is not accepted and complete"
        )
    population = _mapping(
        summary.get("population"), label="analysis-v2 population"
    )
    if (
        population.get("cases") != EXPECTED_CASES
        or population.get("results") != EXPECTED_RESULTS
        or population.get("no_results_dropped") is not True
    ):
        raise PostpublicationV3Error(
            "analysis-v2 population is partial or drops results"
        )
    source_v1 = _mapping(
        summary.get("source_v1"), label="analysis-v2 source-v1"
    )
    if (
        source_v1.get("v1_source_git_commit") != EXPECTED_RUNTIME_COMMIT
        or source_v1.get("v1_run_id") != EXPECTED_RUN_ID
        or source_v1.get("v1_publication_receipt_sha256")
        != sha256_path(v1_publication_receipt_path)
        or source_v1.get("v1_prepublish_receipt_sha256")
        != sha256_path(prepublish_receipt_path)
    ):
        raise PostpublicationV3Error(
            "analysis-v2 source-v1 binding differs"
        )
    return summary, receipt, dict(source_v1)


def _write_atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        json.dump(
            value,
            stream,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _artifact(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise PostpublicationV3Error(f"derived artifact is missing: {path}")
    return {
        "path": str(path),
        "sha256": sha256_path(path),
        "bytes": path.stat().st_size,
    }


def _revalidate_frozen_files(
    entries: Sequence[tuple[Path, str, str]],
) -> None:
    for path, expected_sha256, label in entries:
        _require_file_hash(path, expected_sha256, label=label)


def build_postpublication_v3(
    *,
    config_path: Path,
    manifest_receipt_path: Path,
    manifest_path: Path,
    results_root: Path,
    v1_publication_receipt_path: Path,
    prepublish_receipt_path: Path,
    summary_v2_path: Path,
    analysis_v2_receipt_path: Path,
    analysis_v2_submission_receipt_path: Path,
    analysis_v2_release_receipt_path: Path,
    analysis_v3_submission_receipt_path: Path,
    analysis_v3_release_receipt_path: Path,
    output_root: Path,
    analysis_v2_job_id: str,
    analysis_v2_git_commit: str,
    analysis_v3_job_id: str,
    analysis_v3_git_commit: str,
    expected_summary_v2_sha256: str,
    expected_analysis_v2_receipt_sha256: str,
    expected_analysis_v2_submission_receipt_sha256: str,
    expected_analysis_v2_release_receipt_sha256: str,
    expected_analysis_v3_submission_receipt_sha256: str,
    expected_analysis_v3_release_receipt_sha256: str,
    expected_v1_publication_receipt_sha256: str,
    expected_prepublish_receipt_sha256: str,
    expected_config_sha256: str,
    expected_manifest_sha256: str,
    expected_manifest_receipt_sha256: str,
    expected_v3_module_sha256: str,
    expected_builder_sha256: str,
    expected_runner_sha256: str,
    expected_sbatch_sha256: str,
    expected_submit_helper_sha256: str,
    expected_cases: int = EXPECTED_CASES,
) -> dict[str, Any]:
    """Build a separate terminal v3 publication from accepted v2."""

    if output_root.exists() or output_root.is_symlink():
        raise PostpublicationV3Error(
            "analysis-v3 output root must be unused"
        )
    if expected_cases != EXPECTED_CASES:
        raise PostpublicationV3Error("analysis-v3 denominator is frozen")
    release_commit = _commit(
        analysis_v3_git_commit, label="analysis-v3 release Git commit"
    )
    builder_path = Path(__file__).resolve()
    v3_module_path = Path(v3.__file__).resolve()
    repository_root = builder_path.parents[1]
    runner_path = (
        repository_root / "slurm/run_vlsa_postpublication_analysis_v3.sh"
    )
    sbatch_path = (
        repository_root / "slurm/vlsa_postpublication_analysis_v3.sbatch"
    )
    submit_helper_path = (
        repository_root
        / "scripts/submit_vlsa_postpublication_analysis_v3.sh"
    )
    frozen_files = (
        (
            builder_path,
            expected_builder_sha256,
            "analysis-v3 builder",
        ),
        (
            v3_module_path,
            expected_v3_module_sha256,
            "accepted analysis-v3 module",
        ),
        (runner_path, expected_runner_sha256, "analysis-v3 runner"),
        (sbatch_path, expected_sbatch_sha256, "analysis-v3 SBatch"),
        (
            submit_helper_path,
            expected_submit_helper_sha256,
            "analysis-v3 submit helper",
        ),
        (config_path, expected_config_sha256, "protocol config"),
        (
            manifest_path,
            expected_manifest_sha256,
            "population manifest",
        ),
        (
            manifest_receipt_path,
            expected_manifest_receipt_sha256,
            "population manifest receipt",
        ),
        (
            v1_publication_receipt_path,
            expected_v1_publication_receipt_sha256,
            "v1 publication receipt",
        ),
        (
            prepublish_receipt_path,
            expected_prepublish_receipt_sha256,
            "v1 prepublish receipt",
        ),
        (
            summary_v2_path,
            expected_summary_v2_sha256,
            "analysis-v2 summary",
        ),
        (
            analysis_v2_receipt_path,
            expected_analysis_v2_receipt_sha256,
            "analysis-v2 receipt",
        ),
        (
            analysis_v2_submission_receipt_path,
            expected_analysis_v2_submission_receipt_sha256,
            "analysis-v2 submission receipt",
        ),
        (
            analysis_v2_release_receipt_path,
            expected_analysis_v2_release_receipt_sha256,
            "analysis-v2 release receipt",
        ),
        (
            analysis_v3_submission_receipt_path,
            expected_analysis_v3_submission_receipt_sha256,
            "analysis-v3 submission receipt",
        ),
        (
            analysis_v3_release_receipt_path,
            expected_analysis_v3_release_receipt_sha256,
            "analysis-v3 release receipt",
        ),
    )
    _revalidate_frozen_files(frozen_files)
    control_binding = _validate_v2_control_receipts(
        analysis_job_id=analysis_v2_job_id,
        analysis_git_commit=analysis_v2_git_commit,
        submission_path=analysis_v2_submission_receipt_path,
        submission_sha256=(
            expected_analysis_v2_submission_receipt_sha256
        ),
        release_path=analysis_v2_release_receipt_path,
        release_sha256=expected_analysis_v2_release_receipt_sha256,
    )
    launch_binding = _validate_v3_control_receipts(
        analysis_v2_job_id=analysis_v2_job_id,
        analysis_v2_git_commit=analysis_v2_git_commit,
        analysis_v3_job_id=analysis_v3_job_id,
        analysis_v3_git_commit=release_commit,
        output_root=output_root,
        submission_path=analysis_v3_submission_receipt_path,
        submission_sha256=(
            expected_analysis_v3_submission_receipt_sha256
        ),
        release_path=analysis_v3_release_receipt_path,
        release_sha256=expected_analysis_v3_release_receipt_sha256,
        expected_summary_v2_sha256=expected_summary_v2_sha256,
        expected_analysis_v2_receipt_sha256=(
            expected_analysis_v2_receipt_sha256
        ),
        expected_v2_submission_receipt_sha256=(
            expected_analysis_v2_submission_receipt_sha256
        ),
        expected_v2_release_receipt_sha256=(
            expected_analysis_v2_release_receipt_sha256
        ),
        expected_v1_publication_receipt_sha256=(
            expected_v1_publication_receipt_sha256
        ),
        expected_prepublish_receipt_sha256=(
            expected_prepublish_receipt_sha256
        ),
        expected_config_sha256=expected_config_sha256,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_manifest_receipt_sha256=(
            expected_manifest_receipt_sha256
        ),
        expected_v3_module_sha256=expected_v3_module_sha256,
        expected_builder_sha256=expected_builder_sha256,
        expected_runner_sha256=expected_runner_sha256,
        expected_sbatch_sha256=expected_sbatch_sha256,
        expected_submit_helper_sha256=expected_submit_helper_sha256,
    )
    summary_v2, receipt_v2, source_v1 = _validate_v2_publication(
        summary_path=summary_v2_path,
        summary_sha256=expected_summary_v2_sha256,
        receipt_path=analysis_v2_receipt_path,
        receipt_sha256=expected_analysis_v2_receipt_sha256,
        v1_publication_receipt_path=v1_publication_receipt_path,
        v1_publication_receipt_sha256=(
            expected_v1_publication_receipt_sha256
        ),
        prepublish_receipt_path=prepublish_receipt_path,
        prepublish_receipt_sha256=expected_prepublish_receipt_sha256,
    )
    if not results_root.is_dir() or results_root.is_symlink():
        raise PostpublicationV3Error(
            "immutable population results root is unavailable"
        )

    output_root.mkdir(parents=True)
    cases_path = output_root / "aegis-failure-cases-v3.jsonl"
    report_path = output_root / "aegis-failure-report-v3.json"
    markdown_path = output_root / "aegis-failure-report-v3.md"
    receipt_path = output_root / "analysis-v3-receipt.json"
    report = v3.build_population_failure_artifacts_v3(
        config_path=config_path,
        manifest_receipt_path=manifest_receipt_path,
        manifest_path=manifest_path,
        results_root=results_root,
        summary_path=summary_v2_path,
        validation_receipt_path=prepublish_receipt_path,
        source_publication_receipt_sha256=sha256_path(
            v1_publication_receipt_path
        ),
        cases_output_path=cases_path,
        report_output_path=report_path,
        markdown_output_path=markdown_path,
        expected_cases=expected_cases,
    )
    if (
        report.get("schema_version") != v3.REPORT_SCHEMA
        or report.get("status")
        != "complete_postpublication_failure_analysis_v3"
    ):
        raise PostpublicationV3Error(
            "analysis-v3 report is not complete"
        )
    report_population = _mapping(
        report.get("population"), label="analysis-v3 report population"
    )
    if (
        report_population.get("cases") != EXPECTED_CASES
        or report_population.get("results") != EXPECTED_RESULTS
        or report_population.get("no_cases_dropped") is not True
        or report_population.get("exact_decision_partition") is not True
    ):
        raise PostpublicationV3Error(
            "analysis-v3 report population is incomplete"
        )
    with cases_path.open("rb") as stream:
        case_rows = sum(1 for raw in stream if raw.strip())
    if case_rows != EXPECTED_CASES:
        raise PostpublicationV3Error(
            "analysis-v3 case ledger row count differs"
        )

    # Close the derivation-to-receipt TOCTOU window. Every externally pinned
    # file is re-read after all 1,600 rows are derived and immediately before
    # the terminal receipt is assembled.
    _revalidate_frozen_files(frozen_files)

    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "status": RECEIPT_STATUS,
        "scientific_result": False,
        "launch": launch_binding,
        "source_v1": source_v1,
        "source_v2": {
            **control_binding,
            "summary": _artifact(summary_v2_path),
            "analysis_receipt": _artifact(analysis_v2_receipt_path),
            "analysis_receipt_payload_sha256": receipt_v2[
                "receipt_payload_sha256"
            ],
        },
        "protocol_inputs": {
            "config": _artifact(config_path),
            "manifest": _artifact(manifest_path),
            "manifest_receipt": _artifact(manifest_receipt_path),
            "v1_publication_receipt": _artifact(
                v1_publication_receipt_path
            ),
            "v1_prepublish_receipt": _artifact(
                prepublish_receipt_path
            ),
            "results_root": str(results_root),
        },
        "implementation": {
            "analysis_v3_release_git_commit": release_commit,
            "accepted_v3_implementation_commit": (
                EXPECTED_V3_IMPLEMENTATION_COMMIT
            ),
            "accepted_v3_module": _artifact(v3_module_path),
            "receipt_builder": _artifact(builder_path),
        },
        "population": {
            "cases": EXPECTED_CASES,
            "results": EXPECTED_RESULTS,
            "no_cases_dropped": True,
            "exact_decision_partition": True,
            "accepted_result_payloads_sha256": summary_v2[
                "accepted_result_payloads_sha256"
            ],
        },
        "artifacts": {
            "failure_cases_v3": {
                **_artifact(cases_path),
                "count": case_rows,
            },
            "failure_report_v3": {
                **_artifact(report_path),
                "report_payload_sha256": report[
                    "report_payload_sha256"
                ],
            },
            "failure_markdown_v3": _artifact(markdown_path),
        },
        "claim_scope": summary_v2["claim_scope"],
        "interpretation_scope": report["interpretation_scope"],
    }
    receipt["receipt_payload_sha256"] = aggregate.sha256_bytes(
        canonical_json_bytes(receipt)
    )
    _write_atomic_json(receipt_path, receipt)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build receipt-bound postpublication analysis-v3"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest-receipt", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument(
        "--v1-publication-receipt", type=Path, required=True
    )
    parser.add_argument("--prepublish-receipt", type=Path, required=True)
    parser.add_argument("--summary-v2", type=Path, required=True)
    parser.add_argument("--analysis-v2-receipt", type=Path, required=True)
    parser.add_argument(
        "--analysis-v2-submission-receipt", type=Path, required=True
    )
    parser.add_argument(
        "--analysis-v2-release-receipt", type=Path, required=True
    )
    parser.add_argument(
        "--analysis-v3-submission-receipt", type=Path, required=True
    )
    parser.add_argument(
        "--analysis-v3-release-receipt", type=Path, required=True
    )
    parser.add_argument("--analysis-v2-job-id", required=True)
    parser.add_argument("--analysis-v2-git-commit", required=True)
    parser.add_argument("--analysis-v3-job-id", required=True)
    parser.add_argument("--analysis-v3-git-commit", required=True)
    parser.add_argument("--expected-summary-v2-sha256", required=True)
    parser.add_argument(
        "--expected-analysis-v2-receipt-sha256", required=True
    )
    parser.add_argument(
        "--expected-analysis-v2-submission-receipt-sha256",
        required=True,
    )
    parser.add_argument(
        "--expected-analysis-v2-release-receipt-sha256",
        required=True,
    )
    parser.add_argument(
        "--expected-analysis-v3-submission-receipt-sha256",
        required=True,
    )
    parser.add_argument(
        "--expected-analysis-v3-release-receipt-sha256",
        required=True,
    )
    parser.add_argument(
        "--expected-v1-publication-receipt-sha256", required=True
    )
    parser.add_argument(
        "--expected-prepublish-receipt-sha256", required=True
    )
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument(
        "--expected-manifest-receipt-sha256", required=True
    )
    parser.add_argument("--expected-v3-module-sha256", required=True)
    parser.add_argument("--expected-builder-sha256", required=True)
    parser.add_argument("--expected-runner-sha256", required=True)
    parser.add_argument("--expected-sbatch-sha256", required=True)
    parser.add_argument(
        "--expected-submit-helper-sha256", required=True
    )
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = build_postpublication_v3(
        config_path=args.config.resolve(),
        manifest_receipt_path=args.manifest_receipt.resolve(),
        manifest_path=args.manifest.resolve(),
        results_root=args.results_root.resolve(),
        v1_publication_receipt_path=(
            args.v1_publication_receipt.resolve()
        ),
        prepublish_receipt_path=args.prepublish_receipt.resolve(),
        summary_v2_path=args.summary_v2.resolve(),
        analysis_v2_receipt_path=args.analysis_v2_receipt.resolve(),
        analysis_v2_submission_receipt_path=(
            args.analysis_v2_submission_receipt.resolve()
        ),
        analysis_v2_release_receipt_path=(
            args.analysis_v2_release_receipt.resolve()
        ),
        analysis_v3_submission_receipt_path=(
            args.analysis_v3_submission_receipt.resolve()
        ),
        analysis_v3_release_receipt_path=(
            args.analysis_v3_release_receipt.resolve()
        ),
        output_root=args.output_root.resolve(),
        analysis_v2_job_id=args.analysis_v2_job_id,
        analysis_v2_git_commit=args.analysis_v2_git_commit,
        analysis_v3_job_id=args.analysis_v3_job_id,
        analysis_v3_git_commit=args.analysis_v3_git_commit,
        expected_summary_v2_sha256=args.expected_summary_v2_sha256,
        expected_analysis_v2_receipt_sha256=(
            args.expected_analysis_v2_receipt_sha256
        ),
        expected_analysis_v2_submission_receipt_sha256=(
            args.expected_analysis_v2_submission_receipt_sha256
        ),
        expected_analysis_v2_release_receipt_sha256=(
            args.expected_analysis_v2_release_receipt_sha256
        ),
        expected_analysis_v3_submission_receipt_sha256=(
            args.expected_analysis_v3_submission_receipt_sha256
        ),
        expected_analysis_v3_release_receipt_sha256=(
            args.expected_analysis_v3_release_receipt_sha256
        ),
        expected_v1_publication_receipt_sha256=(
            args.expected_v1_publication_receipt_sha256
        ),
        expected_prepublish_receipt_sha256=(
            args.expected_prepublish_receipt_sha256
        ),
        expected_config_sha256=args.expected_config_sha256,
        expected_manifest_sha256=args.expected_manifest_sha256,
        expected_manifest_receipt_sha256=(
            args.expected_manifest_receipt_sha256
        ),
        expected_v3_module_sha256=args.expected_v3_module_sha256,
        expected_builder_sha256=args.expected_builder_sha256,
        expected_runner_sha256=args.expected_runner_sha256,
        expected_sbatch_sha256=args.expected_sbatch_sha256,
        expected_submit_helper_sha256=(
            args.expected_submit_helper_sha256
        ),
    )
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "receipt_payload_sha256": receipt[
                    "receipt_payload_sha256"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
