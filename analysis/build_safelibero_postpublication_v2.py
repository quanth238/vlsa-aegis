#!/usr/bin/env python3
"""Derive immutable analysis-v2 artifacts from a validated v1 population.

This command never replaces or edits the accepted v1 publication.  It
revalidates the exact compact result population, regenerates the v1 summary,
then publishes a separate claim-scoped v2 directory.  The derived receipt is
written last, so an interrupted run cannot look like a complete publication.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

try:
    from analysis import aggregate_safelibero_aegis as aggregate
    from analysis import build_aegis_failure_report as failures
    from analysis import (
        build_safelibero_postpublication_gallery_v2 as gallery_v2,
    )
except ImportError:  # pragma: no cover - direct script fallback
    import aggregate_safelibero_aegis as aggregate  # type: ignore[no-redef]
    import build_aegis_failure_report as failures  # type: ignore[no-redef]
    import build_safelibero_postpublication_gallery_v2 as gallery_v2  # type: ignore[no-redef]


PUBLICATION_SCHEMA_V1 = "vlsa_table1_population_publication.v1"
PREPUBLISH_SCHEMA_V1 = "vlsa_table1_population_prepublish_validation.v2"
SUMMARY_SCHEMA_V1 = "vlsa_table1_population_summary.v1"
RECEIPT_SCHEMA = "vlsa_table1_postpublication_analysis_v2_receipt.v1"
EXPECTED_V1_SOURCE_COMMIT = (
    "1592aa59361f431ba96c6ddcbebcb596f6c20853"
)


class PostpublicationError(RuntimeError):
    """Raised when immutable v1 evidence or derived v2 output is invalid."""


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PostpublicationError(f"{label} must be an object")
    return value


def _sha(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PostpublicationError(f"{label} must be lowercase SHA-256")
    return value


def _git_commit(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) not in {40, 64}
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PostpublicationError(
            f"{label} must be a lowercase Git object ID"
        )
    return value


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise PostpublicationError(f"{label} is missing or symlinked")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PostpublicationError(f"{label} is not valid JSON") from error
    if not isinstance(value, dict):
        raise PostpublicationError(f"{label} must be a JSON object")
    return value


def _verify_payload_hash(
    record: Mapping[str, Any],
    *,
    field: str,
    label: str,
) -> str:
    expected = _sha(record.get(field), label=f"{label}/{field}")
    payload = dict(record)
    payload.pop(field, None)
    observed = aggregate.sha256_bytes(
        aggregate.canonical_json_bytes(payload)
    )
    if observed != expected:
        raise PostpublicationError(f"{label} payload hash differs")
    return expected


def validate_v1_source_binding(
    *,
    publication_receipt_path: Path,
    summary_path: Path,
    prepublish_receipt_path: Path,
    expected_source_commit: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate exact accepted v1 files and return their immutable binding."""

    publication = _load_json(
        publication_receipt_path, label="v1 publication receipt"
    )
    if (
        publication.get("schema_version") != PUBLICATION_SCHEMA_V1
        or publication.get("status") != "published"
        or publication.get("complete_paired_population") is not True
        or publication.get("no_results_dropped") is not True
    ):
        raise PostpublicationError(
            "v1 publication receipt is not complete and published"
        )
    publication_payload_sha = _verify_payload_hash(
        publication,
        field="receipt_payload_sha256",
        label="v1 publication receipt",
    )
    summary = _load_json(summary_path, label="v1 population summary")
    population = _mapping(
        summary.get("population"), label="v1 summary population"
    )
    if (
        summary.get("schema_version") != SUMMARY_SCHEMA_V1
        or summary.get("status") != "complete_population_validated"
        or population.get("no_results_dropped") is not True
    ):
        raise PostpublicationError("v1 population summary is not complete")
    accepted_ledger = _sha(
        summary.get("accepted_result_payloads_sha256"),
        label="v1 accepted-result ledger",
    )
    summary_descriptor = _mapping(
        publication.get("summary"),
        label="v1 publication summary descriptor",
    )
    summary_sha = aggregate.sha256_path(summary_path)
    if (
        _sha(
            summary_descriptor.get("sha256"),
            label="v1 publication summary SHA",
        )
        != summary_sha
    ):
        raise PostpublicationError(
            "supplied v1 summary differs from publication receipt"
        )

    prepublish = _load_json(
        prepublish_receipt_path,
        label="v1 prepublish validation receipt",
    )
    if (
        prepublish.get("schema_version") != PREPUBLISH_SCHEMA_V1
        or prepublish.get("status") != "validated"
        or prepublish.get("complete_paired_population") is not True
        or prepublish.get("no_results_dropped") is not True
    ):
        raise PostpublicationError(
            "v1 prepublish receipt is not complete and validated"
        )
    prepublish_payload_sha = _verify_payload_hash(
        prepublish,
        field="receipt_payload_sha256",
        label="v1 prepublish receipt",
    )
    prepublish_descriptor = _mapping(
        publication.get("prepublish_receipt"),
        label="v1 publication prepublish descriptor",
    )
    prepublish_sha = aggregate.sha256_path(prepublish_receipt_path)
    if (
        _sha(
            prepublish_descriptor.get("sha256"),
            label="v1 publication prepublish SHA",
        )
        != prepublish_sha
        or _sha(
            prepublish_descriptor.get("receipt_payload_sha256"),
            label="v1 publication prepublish payload SHA",
        )
        != prepublish_payload_sha
    ):
        raise PostpublicationError(
            "supplied prepublish receipt differs from publication receipt"
        )
    source_commit = _git_commit(
        publication.get("source_git_commit"),
        label="v1 source Git commit",
    )
    if (
        expected_source_commit is not None
        and source_commit
        != _git_commit(
            expected_source_commit,
            label="expected v1 source Git commit",
        )
    ):
        raise PostpublicationError(
            "v1 source Git commit differs from the accepted runtime"
        )
    run_id = publication.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise PostpublicationError("v1 run ID must be non-empty")
    gallery_descriptor = _mapping(
        publication.get("gallery"), label="v1 publication gallery"
    )
    failure_descriptor = _mapping(
        publication.get("failure_analysis"),
        label="v1 publication failure analysis",
    )
    failure_cases = _mapping(
        failure_descriptor.get("cases"),
        label="v1 publication failure cases",
    )
    failure_report = _mapping(
        failure_descriptor.get("report"),
        label="v1 publication failure report",
    )
    failure_markdown = _mapping(
        failure_descriptor.get("markdown"),
        label="v1 publication failure Markdown",
    )
    return (
        {
            "v1_publication_receipt_sha256": aggregate.sha256_path(
                publication_receipt_path
            ),
            "v1_publication_receipt_payload_sha256": (
                publication_payload_sha
            ),
            "v1_population_summary_sha256": summary_sha,
            "v1_prepublish_receipt_sha256": prepublish_sha,
            "v1_prepublish_receipt_payload_sha256": (
                prepublish_payload_sha
            ),
            "v1_accepted_result_payloads_sha256": accepted_ledger,
            "v1_gallery_sha256": _sha(
                gallery_descriptor.get("sha256"),
                label="v1 gallery SHA",
            ),
            "v1_failure_cases_sha256": _sha(
                failure_cases.get("sha256"),
                label="v1 failure cases SHA",
            ),
            "v1_failure_report_sha256": _sha(
                failure_report.get("sha256"),
                label="v1 failure report SHA",
            ),
            "v1_failure_markdown_sha256": _sha(
                failure_markdown.get("sha256"),
                label="v1 failure Markdown SHA",
            ),
            "v1_source_git_commit": source_commit,
            "v1_run_id": run_id,
        },
        summary,
    )


def _write_atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        delete=False,
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


def _artifact(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise PostpublicationError(f"derived artifact is missing: {path}")
    return {
        "path": str(path),
        "sha256": aggregate.sha256_path(path),
        "bytes": path.stat().st_size,
    }


def build_postpublication_v2(
    *,
    config_path: Path,
    manifest_receipt_path: Path,
    manifest_path: Path,
    results_root: Path,
    v1_publication_receipt_path: Path,
    v1_summary_path: Path,
    prepublish_receipt_path: Path,
    output_root: Path,
    expected_cases: int = failures.EXPECTED_CASES,
) -> dict[str, Any]:
    """Build a separate, receipt-bound post-publication analysis directory."""

    if output_root.exists() or output_root.is_symlink():
        raise PostpublicationError(
            "analysis-v2 output root must be unused"
        )
    source_binding, v1_summary = validate_v1_source_binding(
        publication_receipt_path=v1_publication_receipt_path,
        summary_path=v1_summary_path,
        prepublish_receipt_path=prepublish_receipt_path,
        expected_source_commit=EXPECTED_V1_SOURCE_COMMIT,
    )
    config, manifests = aggregate.load_protocol(
        config_path,
        manifest_receipt_path,
        manifest_path,
    )
    if len(manifests) != expected_cases:
        raise PostpublicationError(
            f"manifest has {len(manifests)} cases, expected {expected_cases}"
        )
    compact_results = aggregate.load_compact_results_streaming(
        [results_root],
        config=config,
        manifests=manifests,
        verify_video_files=False,
    )
    regenerated_v1 = aggregate.aggregate(
        config=config,
        manifests=manifests,
        results=compact_results,
    )
    if aggregate.canonical_json_bytes(
        regenerated_v1
    ) != aggregate.canonical_json_bytes(v1_summary):
        raise PostpublicationError(
            "exact result population does not regenerate immutable v1 summary"
        )
    summary_v2 = aggregate.aggregate_v2(
        config=config,
        manifests=manifests,
        results=compact_results,
        source_binding=source_binding,
    )
    del compact_results

    output_root.mkdir(parents=True)
    summary_v2_path = output_root / "population-summary-v2.json"
    cases_v2_path = output_root / "aegis-failure-cases-v2.jsonl"
    report_v2_path = output_root / "aegis-failure-report-v2.json"
    markdown_v2_path = output_root / "aegis-failure-report-v2.md"
    gallery_v2_path = output_root / "gallery-v2" / "index.html"
    receipt_path = output_root / "analysis-v2-receipt.json"
    _write_atomic_json(summary_v2_path, summary_v2)
    failure_report = failures.build_population_failure_artifacts_v2(
        config_path=config_path,
        manifest_receipt_path=manifest_receipt_path,
        manifest_path=manifest_path,
        results_root=results_root,
        summary_path=summary_v2_path,
        validation_receipt_path=prepublish_receipt_path,
        source_publication_receipt_sha256=source_binding[
            "v1_publication_receipt_sha256"
        ],
        cases_output_path=cases_v2_path,
        report_output_path=report_v2_path,
        markdown_output_path=markdown_v2_path,
        expected_cases=expected_cases,
    )
    gallery_summary, gallery_report, gallery_records = (
        gallery_v2.load_bound_inputs(
            summary_path=summary_v2_path,
            report_path=report_v2_path,
            cases_path=cases_v2_path,
        )
    )
    gallery_document, gallery_metadata = gallery_v2.build_gallery_v2(
        summary=gallery_summary,
        report=gallery_report,
        records=gallery_records,
        output_root=gallery_v2_path.parent,
    )
    gallery_v2.atomic_write_text(gallery_v2_path, gallery_document)
    del gallery_records
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "status": "published_derived_analysis_v2",
        "scientific_result": False,
        "source_v1": source_binding,
        "population": {
            "cases": expected_cases,
            "results": expected_cases * 2,
            "no_cases_dropped": True,
            "accepted_result_payloads_sha256": source_binding[
                "v1_accepted_result_payloads_sha256"
            ],
        },
        "artifacts": {
            "summary": _artifact(summary_v2_path),
            "failure_cases": {
                **_artifact(cases_v2_path),
                "count": expected_cases,
            },
            "failure_report": {
                **_artifact(report_v2_path),
                "report_payload_sha256": failure_report[
                    "report_payload_sha256"
                ],
            },
            "failure_markdown": _artifact(markdown_v2_path),
            "gallery": {
                **_artifact(gallery_v2_path),
                **gallery_metadata,
            },
        },
        "claim_scope": summary_v2["claim_scope"],
    }
    receipt["receipt_payload_sha256"] = aggregate.sha256_bytes(
        aggregate.canonical_json_bytes(receipt)
    )
    _write_atomic_json(receipt_path, receipt)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build receipt-bound post-publication analysis-v2 artifacts"
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest-receipt", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument(
        "--v1-publication-receipt", type=Path, required=True
    )
    parser.add_argument("--v1-summary", type=Path, required=True)
    parser.add_argument(
        "--prepublish-receipt", type=Path, required=True
    )
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = build_postpublication_v2(
        config_path=args.config,
        manifest_receipt_path=args.manifest_receipt,
        manifest_path=args.manifest,
        results_root=args.results_root,
        v1_publication_receipt_path=args.v1_publication_receipt,
        v1_summary_path=args.v1_summary,
        prepublish_receipt_path=args.prepublish_receipt,
        output_root=args.output_root,
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
