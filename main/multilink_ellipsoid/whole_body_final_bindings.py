"""Build immutable bindings for the final coverage and Q-only gates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from main.multilink_ellipsoid.whole_body_q_only_prediction import (
    BINDING_SCHEMA,
    file_sha256,
    load_protocol,
)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError("final binding source must be an object")
    return value


def bind_combined_audit(
    *, base_path: Path, targeted_cohort_path: Path,
    targeted_artifact_root: Path, targeted_artifact_commit: str,
    targeted_validation_path: Path,
) -> dict[str, Any]:
    """Extend the frozen two-source coverage config with one validated cohort."""

    from main.multilink_ellipsoid.exact_group_boundary import (
        VALIDATION_SCHEMA,
        payload_sha256 as validation_payload_sha256,
    )
    from main.multilink_ellipsoid.whole_body_combined_audit import CONFIG_SCHEMA

    base = _load(base_path)
    if (
        base.get("schema_version") != CONFIG_SCHEMA
        or len(base.get("sources", [])) != 2
    ):
        raise ValueError("combined base config differs")
    targeted = _load(targeted_cohort_path)
    split_counts = targeted.get("state_selection", {}).get("split_counts")
    if (
        split_counts is None
        or set(split_counts) != {"train", "validation", "test"}
        or any(int(count) <= 0 for count in split_counts.values())
    ):
        raise ValueError("targeted cohort split counts differ")
    validation = _load(targeted_validation_path)
    payload = validation.get("validation_payload_sha256")
    if (
        validation.get("schema_version") != VALIDATION_SCHEMA
        or payload != validation_payload_sha256(
            validation, key="validation_payload_sha256",
        )
        or not validation.get("independent_replay", {}).get(
            "exact_scientific_reproduction", False,
        )
    ):
        raise ValueError("targeted validation differs")

    output = json.loads(_canonical(base).decode("utf-8"))
    output["schema_version"] = "vlsa_distal_whole_body_combined_audit.v2"
    output["protocol_id"] = "vlsa-distal-whole-body-combined-audit-v2"
    output["claim_scope"] = (
        "Read-only combined whole-body future-risk coverage gate over immutable "
        "ADR-0177, ADR-0180, and the targeted raw-pi0.5 palm/L6 cohort. "
        "Initially unsafe roots remain recovery diagnostics and do not count "
        "toward prevention coverage or fit. This performs no model fitting, "
        "correction, QP, denoising, deployment, certification, or CBF claim."
    )
    repo_root = base_path.resolve().parent.parent
    existing_split_counts = {split: 0 for split in ("train", "validation", "test")}
    for source in base["sources"]:
        cohort_path = repo_root / source["cohort_config"]
        if file_sha256(cohort_path) != source["cohort_config_file_sha256"]:
            raise ValueError("combined base cohort config hash differs")
        cohort = _load(cohort_path)
        source_counts = cohort.get("state_selection", {}).get("split_counts")
        if source_counts is None:
            raise ValueError("combined base cohort split counts differ")
        for split in existing_split_counts:
            existing_split_counts[split] += int(source_counts[split])
    output["required_split_case_count"] = {
        split: int(existing_split_counts[split])
        + int(split_counts[split])
        for split in ("train", "validation", "test")
    }
    output["initially_unsafe_policy"] = (
        "retain_as_recovery_diagnostic_exclude_from_prevention_coverage_and_fit"
    )
    try:
        targeted_cohort_record = str(
            targeted_cohort_path.resolve().relative_to(repo_root)
        )
    except ValueError:
        targeted_cohort_record = str(targeted_cohort_path)
    output["sources"].append({
        "name": "ADR-0186-targeted-palm-L6",
        "cohort_config": targeted_cohort_record,
        "cohort_config_file_sha256": file_sha256(targeted_cohort_path),
        "artifact_root": str(targeted_artifact_root),
        "artifact_commit": str(targeted_artifact_commit),
        "validation_file_sha256": file_sha256(targeted_validation_path),
        "validation_payload_sha256": str(payload),
    })
    return output


def bind_q_prediction(
    *, protocol_path: Path, coverage_audit_path: Path,
) -> dict[str, Any]:
    """Bind a validated combined audit to the frozen Q-only protocol."""

    from main.multilink_ellipsoid.whole_body_combined_audit import (
        RESULT_SCHEMA,
        payload_sha256,
    )

    protocol = load_protocol(protocol_path)
    coverage = _load(coverage_audit_path)
    audit_payload = coverage.get("audit_payload_sha256")
    if (
        coverage.get("schema_version") != RESULT_SCHEMA
        or audit_payload != payload_sha256(
            coverage, key="audit_payload_sha256",
        )
        or len(coverage.get("combined_config", {}).get("sources", [])) != 3
    ):
        raise ValueError("combined coverage audit differs")
    return {
        "schema_version": BINDING_SCHEMA,
        "protocol_id": "vlsa-distal-whole-body-q-only-prediction-binding-v1",
        "claim_scope": (
            "Immutable binding of the final combined coverage audit to the "
            "frozen Q-only train/validation/test protocol. Training runs even "
            "when coverage fails, but such a result is diagnostic and cannot "
            "authorize correction."
        ),
        "protocol": {
            "path": "configs/vlsa_distal_whole_body_q_only_prediction_protocol.v1.json",
            "file_sha256": protocol["config_file_sha256"],
            "payload_sha256": protocol["config_payload_sha256"],
        },
        "coverage_audit": {
            "path": str(coverage_audit_path),
            "file_sha256": file_sha256(coverage_audit_path),
            "payload_sha256": str(audit_payload),
        },
        "sources": coverage["combined_config"]["sources"],
        "test_opened_once": True,
        "forbidden": list(protocol["forbidden"]),
    }


def binding_payload_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()
