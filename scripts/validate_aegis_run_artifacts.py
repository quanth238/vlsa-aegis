#!/usr/bin/env python3
"""Validate and bind AEGIS canary/population infrastructure artifacts.

This module does not define CAR/TSR/ETS semantics. It delegates per-result and
pair semantics to ``analysis.aggregate_safelibero_aegis`` and adds immutable
run, Slurm, source, checkpoint, preflight, result, and video provenance.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping
import json
import math
import os
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis import aggregate_safelibero_aegis as aggregate  # noqa: E402
from analysis import build_aegis_failure_report as failure_report_analysis  # noqa: E402
from analysis import build_safelibero_video_gallery as gallery_analysis  # noqa: E402
from analysis import validate_aegis_action_invariant_canary as action_canary  # noqa: E402
from analysis import validate_aegis_failure_diagnostics as failure_validation  # noqa: E402
from scripts import validate_aegis_assets as asset_validation  # noqa: E402
from scripts.aegis_receipt_utils import (  # noqa: E402
    ReceiptError,
    canonical_json_bytes,
    load_json_object,
    load_tsv_contract,
    require_sha256,
    sha256_bytes,
    sha256_path,
    verify_payload_sha256,
    write_json_exclusive,
)


CANARY_SCHEMA = (
    "vlsa_table1_action_invariant_paired_canary_validation.v2"
)
POPULATION_PREPUBLISH_SCHEMA = (
    "vlsa_table1_population_prepublish_validation.v2"
)
POPULATION_PUBLICATION_SCHEMA = "vlsa_table1_population_publication.v1"
RUN_CONTRACT_SCHEMA = "vlsa_table1_run_contract.v1"
PREFLIGHT_SCHEMA = "vlsa_table1_allocation_preflight.v1"
PI05_HASH_SCHEMA = "vlsa_table1_pi05_hash_receipt.v1"
RESULT_PAYLOAD_FIELD = "result_payload_sha256"
VALID_QP_STATUSES = {"optimal", "optimal_inaccurate"}
CONTACT_SCHEMA_V3 = "vlsa_table1_active_obstacle_contacts.v3"
CONTACT_MODEL_AUTHORITY_SCHEMA_V2 = (
    "vlsa_table1_contact_model_authority.v2"
)
PUBLISHER_RETRY_AUTHORITY_SCHEMA = (
    "vlsa_table1_publisher_retry_authority.v1"
)
CONTACT_ROLE_TAXONOMY = (
    "robot",
    "static_support",
    "dynamic_task_object",
    "dynamic_other",
    "unknown",
)
FULL_LABEL_MANIFEST_SHA256 = (
    "f9a862f28f168f02de4e0987e37d297de24b167ae50fb96c7f8243a76916880e"
)
CANARY_LABEL_ROW_SHA256 = (
    "2d4d1be5c0a4940c72eb452d00361f6a4935de3f5cbc1058c9671fff96a35a36"
)
CANARY_CASE_ORDINAL = 100
CANARY_CASE_ID = "vlsa-t1-spatial-i-t2-e00"


def _require_equal(observed: Any, expected: Any, *, label: str) -> None:
    if observed != expected:
        raise ReceiptError(
            f"{label} differs: observed={observed!r}, expected={expected!r}"
        )


def _require_bool(value: Any, *, label: str) -> bool:
    if not isinstance(value, bool):
        raise ReceiptError(f"{label} must be Boolean")
    return value


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _require_finite_sequence(
    value: Any,
    *,
    length: int,
    label: str,
) -> list[float]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != length
        or any(not _is_finite_number(item) for item in value)
    ):
        raise ReceiptError(
            f"paired-canary integration is inconclusive: {label} "
            "must be finite"
        )
    return [float(item) for item in value]


def validate_publisher_retry_authority(
    path: Path,
    *,
    run_root: Path,
    expected_population_commit: str,
    expected_run_id: str,
    expected_population_array_job_id: str,
) -> dict[str, Any]:
    run_root = run_root.resolve()
    authority = load_json_object(path, label="publisher retry authority")
    payload_sha256 = verify_payload_sha256(
        authority,
        field="receipt_payload_sha256",
        label="publisher retry authority",
    )
    for field, expected in (
        ("schema_version", PUBLISHER_RETRY_AUTHORITY_SCHEMA),
        ("status", "validated"),
        ("scientific_result", False),
        ("run_id", expected_run_id),
        ("population_array_job_id", expected_population_array_job_id),
        ("preserves_immutable_result_tree", True),
        ("permits_inference_or_simulation", False),
        ("failure_class", "publisher_validator_json_key_order"),
    ):
        _require_equal(
            authority.get(field),
            expected,
            label=f"publisher retry authority/{field}",
        )
    population_source = authority.get("population_source")
    publisher_source = authority.get("publisher_source")
    recovery_from = authority.get("recovery_from")
    publisher_slurm = authority.get("publisher_slurm")
    if not all(
        isinstance(value, Mapping)
        for value in (
            population_source,
            publisher_source,
            recovery_from,
            publisher_slurm,
        )
    ):
        raise ReceiptError("publisher retry authority records are incomplete")
    _require_equal(
        population_source.get("git_commit"),
        expected_population_commit,
        label="publisher retry population source commit",
    )
    publisher_commit = publisher_source.get("git_commit")
    if (
        not isinstance(publisher_commit, str)
        or len(publisher_commit) != 40
        or any(
            character not in "0123456789abcdef"
            for character in publisher_commit
        )
        or publisher_commit == expected_population_commit
    ):
        raise ReceiptError("publisher retry source commit is invalid")
    _require_equal(
        publisher_source.get("validator_sha256"),
        sha256_path(Path(__file__).resolve()),
        label="publisher retry validator source",
    )
    current_job_id = os.environ.get("SLURM_JOB_ID")
    current_host = os.environ.get("SLURMD_NODENAME")
    current_dependency = os.environ.get("SLURM_JOB_DEPENDENCY")
    for field, expected in (
        ("job_id", current_job_id),
        ("host", current_host),
        (
            "dependency",
            f"afterany:{expected_population_array_job_id}",
        ),
    ):
        _require_equal(
            publisher_slurm.get(field),
            expected,
            label=f"publisher retry Slurm/{field}",
        )
    expected_parent = (
        run_root / "publication-attempts" / f"job-{current_job_id}"
    )
    _require_equal(
        path.resolve().parent,
        expected_parent,
        label="publisher retry authority parent",
    )
    return {
        "path": str(path.resolve()),
        "sha256": sha256_path(path.resolve()),
        "receipt_payload_sha256": payload_sha256,
        "publisher_source_git_commit": publisher_commit,
        "previous_publisher_job_id": recovery_from.get(
            "publisher_job_id"
        ),
        "publisher_slurm": dict(publisher_slurm),
    }


def validate_full_label_manifest(
    path: Path,
    *,
    canary_case_id: str,
) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ReceiptError("full frozen label manifest is missing or symlinked")
    _require_equal(
        sha256_path(path),
        FULL_LABEL_MANIFEST_SHA256,
        label="full frozen label manifest SHA-256",
    )
    raw_lines = [
        line
        for line in path.read_bytes().splitlines(keepends=True)
        if line.strip()
    ]
    if len(raw_lines) != 1600:
        raise ReceiptError(
            "full frozen label manifest must contain exactly 1,600 rows"
        )
    selected: list[bytes] = []
    case_ids: set[str] = set()
    for line in raw_lines:
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ReceiptError(
                "full frozen label manifest contains invalid JSON"
            ) from error
        case_id = row.get("case_id") if isinstance(row, Mapping) else None
        if not isinstance(case_id, str) or not case_id or case_id in case_ids:
            raise ReceiptError(
                "full frozen label manifest has an invalid or duplicate case"
            )
        case_ids.add(case_id)
        if case_id == canary_case_id:
            selected.append(line)
    if len(selected) != 1:
        raise ReceiptError(
            "full frozen label manifest does not bind the canary exactly once"
        )
    canary_path = ROOT / "labels/vlsa_table1_canary_labels.jsonl"
    _require_equal(
        sha256_path(canary_path),
        CANARY_LABEL_ROW_SHA256,
        label="frozen one-row canary label SHA-256",
    )
    if selected[0] != canary_path.read_bytes():
        raise ReceiptError(
            "full label manifest canary row is not byte-identical to the "
            "frozen one-row label"
        )
    return {
        "path": str(path),
        "sha256": FULL_LABEL_MANIFEST_SHA256,
        "rows": 1600,
        "unique_case_ids": 1600,
        "canary_case_id": canary_case_id,
        "canary_row_sha256": CANARY_LABEL_ROW_SHA256,
        "canary_row_byte_identical": True,
    }


def _load_selected_frozen_label_record(
    path: Path,
    *,
    case_id: str,
) -> dict[str, Any]:
    selected: list[dict[str, Any]] = []
    for raw_line in path.read_bytes().splitlines():
        if not raw_line.strip():
            continue
        try:
            row = json.loads(raw_line)
        except json.JSONDecodeError as error:
            raise ReceiptError(
                "full frozen label manifest contains invalid JSON"
            ) from error
        if isinstance(row, dict) and row.get("case_id") == case_id:
            selected.append(row)
    if len(selected) != 1:
        raise ReceiptError(
            "full frozen label manifest does not bind the canary exactly once"
        )
    return selected[0]


def _load_population_frozen_label_records(
    path: Path,
    *,
    manifests: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    expected_case_ids = {str(row["case_id"]) for row in manifests}
    if len(expected_case_ids) != len(manifests):
        raise ReceiptError(
            "population manifest contains duplicate case identities"
        )
    records: dict[str, dict[str, Any]] = {}
    for raw_line in path.read_bytes().splitlines():
        if not raw_line.strip():
            continue
        try:
            row = json.loads(raw_line)
        except json.JSONDecodeError as error:
            raise ReceiptError(
                "full frozen label manifest contains invalid JSON"
            ) from error
        case_id = row.get("case_id") if isinstance(row, dict) else None
        if (
            not isinstance(case_id, str)
            or not case_id
            or case_id in records
        ):
            raise ReceiptError(
                "full frozen label manifest has an invalid or duplicate case"
            )
        records[case_id] = row
    _require_equal(
        set(records),
        expected_case_ids,
        label="full frozen label population identities",
    )
    return records


def _validate_canary_result_binding(
    result: Mapping[str, Any],
    *,
    diagnostics_mode: str,
    policy_mode: str,
    expected_arm: str,
    expected_label_record: Mapping[str, Any],
) -> None:
    label = f"{diagnostics_mode}/{policy_mode}"
    _require_equal(
        result.get("mode"),
        policy_mode,
        label=f"{label} result mode",
    )
    _require_equal(
        result.get("arm"),
        expected_arm,
        label=f"{label} result arm",
    )
    settled = result.get("settled_observation")
    label_record = (
        settled.get("label_record")
        if isinstance(settled, Mapping)
        else None
    )
    if not isinstance(label_record, Mapping):
        raise ReceiptError(f"{label} result lacks its settled label record")
    _require_equal(
        canonical_json_bytes(dict(label_record)),
        canonical_json_bytes(dict(expected_label_record)),
        label=f"{label} canonical frozen label-record bytes",
    )


def _validate_population_result_binding(
    result: Mapping[str, Any],
    *,
    task_index: int,
    policy_mode: str,
    expected_arm: str,
    expected_label_record: Mapping[str, Any],
) -> None:
    case_id = str(result.get("case_id", ""))
    label = f"task-{task_index}/{case_id}/{policy_mode}"
    _require_equal(
        result.get("mode"),
        policy_mode,
        label=f"{label} result mode",
    )
    _require_equal(
        result.get("arm"),
        expected_arm,
        label=f"{label} result arm",
    )
    settled = result.get("settled_observation")
    label_record = (
        settled.get("label_record")
        if isinstance(settled, Mapping)
        else None
    )
    if not isinstance(label_record, Mapping):
        raise ReceiptError(
            f"{label} result lacks its settled frozen label record"
        )
    _require_equal(
        canonical_json_bytes(dict(label_record)),
        canonical_json_bytes(dict(expected_label_record)),
        label=f"{label} canonical frozen label-record bytes",
    )


def _canary_result_specs(
    config: Mapping[str, Any],
) -> tuple[tuple[str, str, str], ...]:
    expected_arms = tuple(
        action_canary.EXPECTED_ARM_BY_POLICY_MODE[mode]
        for mode in action_canary.POLICY_MODES
    )
    _require_equal(
        tuple(config.get("arms", ())),
        expected_arms,
        label="paired-canary protocol mode/arm mapping",
    )
    return tuple(
        (
            diagnostics_mode,
            mode,
            action_canary.EXPECTED_ARM_BY_POLICY_MODE[mode],
        )
        for diagnostics_mode in action_canary.DIAGNOSTIC_MODES
        for mode in action_canary.POLICY_MODES
    )


def _validate_canary_action_reference_path(path: Path) -> Path:
    expected = (
        ROOT / "fixtures/vlsa_table1_canary_action_reference.json"
    )
    if path.is_symlink() or not path.is_file():
        raise ReceiptError(
            "paired-canary action reference must be a regular file"
        )
    if expected.is_symlink() or not expected.is_file():
        raise ReceiptError(
            "canonical paired-canary action-reference fixture is invalid"
        )
    resolved = path.resolve()
    _require_equal(
        resolved,
        expected.resolve(),
        label="paired-canary canonical action-reference fixture",
    )
    return resolved


def validate_aegis_canary_integration(
    result: Mapping[str, Any],
) -> dict[str, Any]:
    obstacle = result.get("obstacle")
    if (
        not isinstance(obstacle, Mapping)
        or obstacle.get("selector_matches_active") is not True
    ):
        raise ReceiptError(
            "paired-canary integration is inconclusive: frozen selector "
            "did not match the active obstacle"
        )
    perception = result.get("perception")
    if (
        not isinstance(perception, Mapping)
        or perception.get("status") != "ready"
    ):
        raise ReceiptError(
            "paired-canary integration is inconclusive: AEGIS perception "
            "did not reach ready"
        )
    actions = result.get("actions")
    if not isinstance(actions, list) or not actions:
        raise ReceiptError(
            "paired-canary integration is inconclusive: no action was "
            "executed through AEGIS"
        )
    qp_actions: list[dict[str, Any]] = []
    for action in actions:
        if (
            not isinstance(action, Mapping)
            or action.get("control_path") != "aegis_qp"
        ):
            continue
        qp = action.get("qp")
        if not isinstance(qp, Mapping):
            raise ReceiptError(
                "paired-canary integration is inconclusive: aegis_qp "
                "action lacks QP diagnostics"
            )
        solver_status = str(qp.get("solver_status", "")).strip().lower()
        if (
            qp.get("solver") != "OSQP"
            or solver_status not in VALID_QP_STATUSES
        ):
            raise ReceiptError(
                "paired-canary integration is inconclusive: QP solver "
                "diagnostics are not valid"
            )
        for field in ("objective", "barrier_h", "constraint_lhs"):
            if not _is_finite_number(qp.get(field)):
                raise ReceiptError(
                    "paired-canary integration is inconclusive: QP "
                    f"diagnostic {field} is not finite"
                )
        _require_finite_sequence(
            qp.get("u_solution"),
            length=6,
            label="QP u_solution",
        )
        _require_finite_sequence(
            qp.get("z_after"),
            length=3,
            label="QP z_after",
        )
        step = action.get("step")
        if not isinstance(step, int) or isinstance(step, bool) or step < 0:
            raise ReceiptError(
                "paired-canary integration is inconclusive: QP action "
                "step is invalid"
            )
        qp_actions.append(
            {
                "step": step,
                "solver": "OSQP",
                "solver_status": solver_status,
                "diagnostics_sha256": sha256_bytes(
                    canonical_json_bytes(dict(qp))
                ),
            }
        )
    if not qp_actions:
        raise ReceiptError(
            "paired-canary integration is inconclusive: actions used "
            "fail-open or another non-QP control path"
        )
    return {
        "status": "passed",
        "selector_match": True,
        "perception_status": "ready",
        "valid_qp_diagnostics": True,
        "executed_aegis_qp_actions": len(qp_actions),
        "first_aegis_qp_step": qp_actions[0]["step"],
        "qp_actions": qp_actions,
    }


def allocation_identity(environment: Mapping[str, str]) -> dict[str, str]:
    required = (
        "SLURM_JOB_ID",
        "SLURM_ARRAY_JOB_ID",
        "SLURM_ARRAY_TASK_ID",
        "SLURMD_NODENAME",
    )
    missing = [key for key in required if not environment.get(key)]
    if missing:
        raise ReceiptError(
            f"paired canary validation requires allocation fields {missing}"
        )
    host = str(environment["SLURMD_NODENAME"])
    if host == "worker-3" or host.startswith(("login", "login-restricted")):
        raise ReceiptError(f"validation cannot execute on {host}")
    return {
        "job_id": str(environment["SLURM_JOB_ID"]),
        "array_job_id": str(environment["SLURM_ARRAY_JOB_ID"]),
        "array_task_id": str(environment["SLURM_ARRAY_TASK_ID"]),
        "host": host,
    }


def validate_recorded_allocation_identity(
    value: Any,
    *,
    label: str,
    require_task_zero: bool,
) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ReceiptError(f"{label} is missing")
    required = ("job_id", "array_job_id", "array_task_id", "host")
    identity = {key: str(value.get(key, "")) for key in required}
    if any(not identity[key] for key in required):
        raise ReceiptError(f"{label} is incomplete")
    host = identity["host"]
    if host == "worker-3" or host.startswith(("login", "login-restricted")):
        raise ReceiptError(f"{label} records forbidden host {host}")
    if require_task_zero and identity["array_task_id"] != "0":
        raise ReceiptError(f"{label} must record array task zero")
    return identity


def validate_run_contract(
    *,
    run_root: Path,
    expected_stage: str,
    expected_commit: str,
    config_path: Path,
    manifest_path: Path,
    manifest_receipt_path: Path,
    label_manifest_path: Path,
) -> tuple[dict[str, str], dict[str, Any]]:
    contract_path = run_root / "run-contract.tsv"
    contract = load_tsv_contract(contract_path)
    required = {
        "schema_version",
        "run_id",
        "run_stage",
        "case_ordinal",
        "git_commit",
        "config_sha256",
        "manifest_sha256",
        "manifest_receipt_sha256",
        "label_manifest_sha256",
        "label_publication_receipt_sha256",
        "pi05_tree_sha256",
        "pi05_hash_receipt_sha256",
        "paired_canary_receipt_sha256",
        "groundingdino_device",
    }
    missing = sorted(required - set(contract))
    if missing:
        raise ReceiptError(f"run contract is missing fields: {missing}")
    _require_equal(
        contract["schema_version"],
        RUN_CONTRACT_SCHEMA,
        label="run contract schema",
    )
    _require_equal(
        contract["run_stage"], expected_stage, label="run contract stage"
    )
    _require_equal(
        contract["git_commit"], expected_commit, label="run source commit"
    )
    for path, field, label in (
        (config_path, "config_sha256", "protocol config"),
        (manifest_path, "manifest_sha256", "population manifest"),
        (
            manifest_receipt_path,
            "manifest_receipt_sha256",
            "manifest receipt",
        ),
        (
            label_manifest_path,
            "label_manifest_sha256",
            "frozen label manifest",
        ),
    ):
        _require_equal(
            sha256_path(path), contract[field], label=f"{label} SHA-256"
        )
    for field in (
        "config_sha256",
        "manifest_sha256",
        "manifest_receipt_sha256",
        "label_manifest_sha256",
        "label_publication_receipt_sha256",
        "pi05_tree_sha256",
        "pi05_hash_receipt_sha256",
    ):
        require_sha256(contract[field], label=f"run contract/{field}")
    _require_equal(
        contract["label_publication_receipt_sha256"],
        asset_validation.LABEL_PUBLICATION_RECEIPT_SHA256,
        label="run contract/frozen label-publication receipt",
    )
    _require_equal(
        contract["groundingdino_device"],
        "cpu",
        label="run contract/frozen GroundingDINO device",
    )
    if expected_stage == "population":
        require_sha256(
            contract["paired_canary_receipt_sha256"],
            label="run contract/paired_canary_receipt_sha256",
        )
    elif contract["paired_canary_receipt_sha256"] != "none":
        raise ReceiptError("paired canary cannot depend on a prior canary")
    return contract, {
        "path": str(contract_path),
        "sha256": sha256_path(contract_path),
    }


def validate_pi05_hash_receipt(
    path: Path,
    *,
    expected_file_sha256: str,
    expected_tree_sha256: str,
    expected_commit: str,
) -> dict[str, Any]:
    value = load_json_object(path, label="pi0.5 hash receipt")
    _require_equal(
        sha256_path(path),
        require_sha256(
            expected_file_sha256, label="pi0.5 hash receipt file"
        ),
        label="pi0.5 hash receipt file SHA-256",
    )
    _require_equal(
        value.get("schema_version"),
        PI05_HASH_SCHEMA,
        label="pi0.5 hash receipt schema",
    )
    _require_equal(value.get("status"), "passed", label="pi0.5 hash status")
    _require_equal(
        value.get("scientific_result"),
        False,
        label="pi0.5 hash scientific-result flag",
    )
    payload_sha256 = verify_payload_sha256(
        value,
        field="receipt_payload_sha256",
        label="pi0.5 hash receipt",
    )
    source = value.get("source")
    checkpoint = value.get("checkpoint")
    if not isinstance(source, Mapping) or not isinstance(checkpoint, Mapping):
        raise ReceiptError("pi0.5 hash receipt source/checkpoint is missing")
    _require_equal(
        source.get("git_commit"),
        expected_commit,
        label="pi0.5 hash source commit",
    )
    _require_equal(
        source.get("git_dirty"), False, label="pi0.5 hash source dirty state"
    )
    _require_equal(
        checkpoint.get("full_content_hash_verified"),
        True,
        label="pi0.5 full-content verification",
    )
    _require_equal(
        checkpoint.get("full_content_tree_sha256"),
        require_sha256(
            expected_tree_sha256, label="pi0.5 expected full tree"
        ),
        label="pi0.5 full tree SHA-256",
    )
    try:
        filesystem_identity = (
            asset_validation.validate_pi05_filesystem_identity_record(
                checkpoint.get("filesystem_identity")
            )
        )
    except asset_validation.PreflightError as error:
        raise ReceiptError(str(error)) from error
    if checkpoint.get("path") != filesystem_identity["checkpoint_path"]:
        raise ReceiptError(
            "pi0.5 hash receipt checkpoint path differs from stat identity"
        )
    execution = value.get("execution")
    execution_fields = (
        "policy_model_executed",
        "simulator_executed",
        "groundingdino_executed",
        "qp_executed",
        "training_executed",
    )
    if (
        not isinstance(execution, Mapping)
        or any(execution.get(field) is not False for field in execution_fields)
    ):
        raise ReceiptError(
            "pi0.5 hash receipt contains forbidden experiment execution"
        )
    slurm = value.get("slurm")
    if not isinstance(slurm, Mapping):
        raise ReceiptError("pi0.5 hash receipt lacks allocation identity")
    host = str(slurm.get("host", ""))
    if (
        not slurm.get("job_id")
        or not slurm.get("array_job_id")
        or str(slurm.get("array_task_id")) != "0"
        or not host
        or host == "worker-3"
        or host.startswith(("login", "login-restricted"))
    ):
        raise ReceiptError("pi0.5 hash receipt lacks valid compute identity")
    return {
        "path": str(path),
        "sha256": sha256_path(path),
        "receipt_payload_sha256": payload_sha256,
        "full_content_tree_sha256": expected_tree_sha256,
        "filesystem_identity_sha256": filesystem_identity[
            "identity_sha256"
        ],
        "slurm": dict(slurm),
    }


def validate_preflight_receipt(
    path: Path,
    *,
    contract: Mapping[str, str],
    expected_slurm: Mapping[str, str] | None,
    expected_array_job_id: str | None = None,
    expected_array_task_id: str | None = None,
) -> dict[str, Any]:
    value = load_json_object(path, label="allocation preflight receipt")
    _require_equal(
        value.get("schema_version"),
        PREFLIGHT_SCHEMA,
        label="preflight schema",
    )
    _require_equal(value.get("status"), "passed", label="preflight status")
    _require_equal(
        value.get("scientific_result"),
        False,
        label="preflight scientific-result flag",
    )
    _require_equal(
        value.get("profile"), "evaluation", label="preflight profile"
    )
    payload_sha256 = verify_payload_sha256(
        value,
        field="receipt_payload_sha256",
        label="allocation preflight",
    )
    source = value.get("source")
    protocol = value.get("protocol")
    assets = value.get("assets")
    slurm = value.get("slurm")
    if not all(
        isinstance(item, Mapping)
        for item in (source, protocol, assets, slurm)
    ):
        raise ReceiptError("preflight source/protocol/assets/Slurm is missing")
    for field in (
        "policy_model_executed",
        "groundingdino_executed",
        "qp_executed",
    ):
        _require_equal(
            assets.get(field),
            False,
            label=f"preflight forbidden execution/{field}",
        )
    _require_equal(
        source.get("git_commit"),
        contract["git_commit"],
        label="preflight source commit",
    )
    _require_equal(
        source.get("git_dirty"), False, label="preflight source dirty state"
    )
    for section, field, contract_field in (
        ("config", "sha256", "config_sha256"),
        ("manifest", "sha256", "manifest_sha256"),
        ("receipt", "sha256", "manifest_receipt_sha256"),
    ):
        record = protocol.get(section)
        if not isinstance(record, Mapping):
            raise ReceiptError(f"preflight protocol.{section} is missing")
        _require_equal(
            record.get(field),
            contract[contract_field],
            label=f"preflight protocol.{section} SHA-256",
        )
    checkpoint = assets.get("pi05_checkpoint")
    hash_receipt = assets.get("pi05_hash_receipt")
    labels = assets.get("frozen_codex_labels")
    grounding = assets.get("groundingdino")
    video_runtime = assets.get("video_runtime")
    label_publication = assets.get("label_publication_receipt")
    if not all(
        isinstance(item, Mapping)
        for item in (
            checkpoint,
            hash_receipt,
            labels,
            grounding,
            video_runtime,
            label_publication,
        )
    ):
        raise ReceiptError("preflight evaluation asset evidence is incomplete")
    _require_equal(
        checkpoint.get("full_content_hash_verified"),
        True,
        label="preflight checkpoint full verification",
    )
    _require_equal(
        checkpoint.get("full_content_tree_sha256"),
        contract["pi05_tree_sha256"],
        label="preflight checkpoint tree",
    )
    _require_equal(
        checkpoint.get("full_content_hash_verification"),
        "one_time_allocation_receipt_plus_exact_stat_identity",
        label="preflight checkpoint verification mode",
    )
    _require_equal(
        checkpoint.get("full_content_rehashed_in_this_allocation"),
        False,
        label="preflight checkpoint redundant full rehash",
    )
    _require_equal(
        checkpoint.get("filesystem_identity_sha256"),
        hash_receipt.get("filesystem_identity_sha256"),
        label="preflight checkpoint filesystem identity",
    )
    _require_equal(
        hash_receipt.get("sha256"),
        contract["pi05_hash_receipt_sha256"],
        label="preflight pi0.5 hash receipt",
    )
    _require_equal(
        hash_receipt.get("full_content_hash_verified"),
        True,
        label="preflight pi0.5 hash verification",
    )
    _require_equal(
        hash_receipt.get("full_content_tree_sha256"),
        contract["pi05_tree_sha256"],
        label="preflight pi0.5 hash tree",
    )
    _require_equal(
        labels.get("sha256"),
        contract["label_manifest_sha256"],
        label="preflight frozen labels",
    )
    _require_equal(
        labels.get("rows"),
        1600,
        label="preflight frozen-label row count",
    )
    _require_equal(
        labels.get("all_population_cases_bound"),
        True,
        label="preflight complete frozen-label population",
    )
    _require_equal(
        labels.get("required_cases"),
        1600 if contract["case_ordinal"] == "all" else 1,
        label="preflight required frozen-label cases",
    )
    _require_equal(
        label_publication.get("sha256"),
        contract["label_publication_receipt_sha256"],
        label="preflight label-publication receipt SHA-256",
    )
    _require_equal(
        label_publication.get("labels_sha256"),
        contract["label_manifest_sha256"],
        label="preflight label-publication manifest binding",
    )
    _require_equal(
        label_publication.get("case_count"),
        1600,
        label="preflight label-publication case count",
    )
    _require_equal(
        label_publication.get("outcome_blind"),
        True,
        label="preflight outcome-blind label publication",
    )
    _require_equal(
        grounding.get("device"),
        contract["groundingdino_device"],
        label="preflight GroundingDINO device",
    )
    aegis_python = video_runtime.get("aegis_python")
    imageio = video_runtime.get("imageio")
    imageio_ffmpeg = video_runtime.get("imageio_ffmpeg")
    ffmpeg = video_runtime.get("ffmpeg")
    if not all(
        isinstance(item, Mapping)
        for item in (aegis_python, imageio, imageio_ffmpeg, ffmpeg)
    ):
        raise ReceiptError("preflight video runtime evidence is incomplete")
    _require_equal(
        aegis_python.get("path"),
        str(asset_validation.DEFAULT_AEGIS_PYTHON),
        label="preflight AEGIS interpreter",
    )
    _require_equal(
        aegis_python.get("resolved_path"),
        str(asset_validation.DEFAULT_AEGIS_PYTHON_RESOLVED),
        label="preflight resolved AEGIS interpreter",
    )
    _require_equal(
        aegis_python.get("python_version"),
        asset_validation.AEGIS_PYTHON_VERSION,
        label="preflight AEGIS Python version",
    )
    _require_equal(
        imageio.get("version"),
        asset_validation.IMAGEIO_VERSION,
        label="preflight ImageIO version",
    )
    _require_equal(
        imageio_ffmpeg.get("version"),
        asset_validation.IMAGEIO_FFMPEG_VERSION,
        label="preflight imageio-ffmpeg version",
    )
    _require_equal(
        ffmpeg.get("path"),
        str(asset_validation.DEFAULT_IMAGEIO_FFMPEG_EXE),
        label="preflight FFmpeg path",
    )
    _require_equal(
        ffmpeg.get("resolved_path"),
        str(asset_validation.DEFAULT_IMAGEIO_FFMPEG_EXE),
        label="preflight resolved FFmpeg path",
    )
    _require_equal(
        ffmpeg.get("sha256"),
        asset_validation.IMAGEIO_FFMPEG_SHA256,
        label="preflight FFmpeg SHA-256",
    )
    _require_equal(
        ffmpeg.get("executable"),
        True,
        label="preflight FFmpeg executable flag",
    )
    observed_slurm = {key: str(value) for key, value in slurm.items()}
    if expected_slurm is not None:
        for key in ("job_id", "array_job_id", "array_task_id", "host"):
            _require_equal(
                observed_slurm.get(key),
                expected_slurm[key],
                label=f"preflight Slurm {key}",
            )
    if expected_array_job_id is not None:
        _require_equal(
            observed_slurm.get("array_job_id"),
            expected_array_job_id,
            label="preflight population array job",
        )
    if expected_array_task_id is not None:
        _require_equal(
            observed_slurm.get("array_task_id"),
            expected_array_task_id,
            label="preflight population array task",
        )
    host = observed_slurm.get("host", "")
    if host == "worker-3" or host.startswith(("login", "login-restricted")):
        raise ReceiptError(f"preflight records forbidden host {host!r}")
    return {
        "path": str(path),
        "sha256": sha256_path(path),
        "receipt_payload_sha256": payload_sha256,
        "slurm": observed_slurm,
    }


def _validate_result_artifact(
    *,
    result_path: Path,
    output_root: Path,
    config: dict[str, Any],
    manifest: dict[str, Any],
    expected_commit: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    output_root = output_root.resolve()
    try:
        result_path.resolve().relative_to(output_root)
    except ValueError as error:
        raise ReceiptError(
            f"{result_path}: result escaped its immutable output root"
        ) from error
    result = load_json_object(result_path, label="paired result")
    verify_payload_sha256(
        result, field=RESULT_PAYLOAD_FIELD, label=str(result_path)
    )
    validated = aggregate.validate_result(
        result,
        config=config,
        manifest=manifest,
    )
    source = result.get("source")
    git = source.get("git") if isinstance(source, Mapping) else None
    if not isinstance(git, Mapping):
        raise ReceiptError(f"{result_path}: source git identity is missing")
    _require_equal(
        git.get("commit"), expected_commit, label=f"{result_path}: source commit"
    )
    _require_equal(
        git.get("dirty"), False, label=f"{result_path}: source dirty state"
    )
    video = result.get("video")
    if not isinstance(video, Mapping):
        raise ReceiptError(f"{result_path}: video record is missing")
    _require_equal(
        video.get("complete_episode"),
        True,
        label=f"{result_path}: complete video",
    )
    relative = Path(str(video.get("path", "")))
    if not relative.name or relative.is_absolute() or ".." in relative.parts:
        raise ReceiptError(f"{result_path}: unsafe video path")
    video_candidate = output_root / relative
    if video_candidate.is_symlink() or not video_candidate.is_file():
        raise ReceiptError(f"{result_path}: video file is missing")
    video_path = video_candidate.resolve()
    try:
        video_path.relative_to(output_root)
    except ValueError as error:
        raise ReceiptError(f"{result_path}: video escaped its output root") from error
    video_sha256 = require_sha256(
        video.get("sha256"), label=f"{result_path}: video"
    )
    _require_equal(
        sha256_path(video_path),
        video_sha256,
        label=f"{result_path}: video file SHA-256",
    )
    return validated, {
        "path": str(result_path),
        "sha256": sha256_path(result_path),
        "result_payload_sha256": result[RESULT_PAYLOAD_FIELD],
        "video_path": str(video_path),
        "video_sha256": video_sha256,
        "case_id": result["case_id"],
        "arm": result["arm"],
        "status": result["status"],
    }


def validate_paired_canary(args: argparse.Namespace) -> dict[str, Any]:
    run_root = args.run_root.resolve()
    config_path = args.config.resolve()
    manifest_path = args.manifest.resolve()
    manifest_receipt_path = args.manifest_receipt.resolve()
    label_manifest_path = args.labels.resolve()
    action_reference_path = _validate_canary_action_reference_path(
        args.action_reference
    )
    contract, contract_record = validate_run_contract(
        run_root=run_root,
        expected_stage="paired-canary",
        expected_commit=args.expected_commit,
        config_path=config_path,
        manifest_path=manifest_path,
        manifest_receipt_path=manifest_receipt_path,
        label_manifest_path=label_manifest_path,
    )
    case_ordinal = int(contract["case_ordinal"])
    _require_equal(
        case_ordinal,
        CANARY_CASE_ORDINAL,
        label="frozen action-invariant canary ordinal",
    )
    if args.case_ordinal is not None and args.case_ordinal != case_ordinal:
        raise ReceiptError("validator case ordinal differs from run contract")
    config, manifests = aggregate.load_protocol(
        config_path, manifest_receipt_path, manifest_path
    )
    if not 0 <= case_ordinal < len(manifests):
        raise ReceiptError("canary case ordinal is outside the manifest")
    manifest = manifests[case_ordinal]
    _require_equal(
        manifest["case_id"],
        CANARY_CASE_ID,
        label="frozen action-invariant canary case",
    )
    full_label_manifest = validate_full_label_manifest(
        label_manifest_path,
        canary_case_id=manifest["case_id"],
    )
    expected_label_record = _load_selected_frozen_label_record(
        label_manifest_path,
        case_id=manifest["case_id"],
    )
    current_slurm = allocation_identity(dict(os.environ))
    _require_equal(
        current_slurm["array_task_id"], "0", label="canary array task"
    )
    task_root = run_root / "tasks/task-0"
    preflight = validate_preflight_receipt(
        task_root / "allocation-preflight.json",
        contract=contract,
        expected_slurm=current_slurm,
    )
    hash_receipt = validate_pi05_hash_receipt(
        args.pi05_hash_receipt.resolve(),
        expected_file_sha256=contract["pi05_hash_receipt_sha256"],
        expected_tree_sha256=contract["pi05_tree_sha256"],
        expected_commit=args.expected_commit,
    )
    output_root = task_root / "results"
    result_specs = _canary_result_specs(config)
    expected_result_paths = {
        (
            output_root
            / diagnostics_mode
            / mode
            / manifest["case_id"]
            / "result.json"
        ).resolve()
        for diagnostics_mode, mode, _ in result_specs
    }
    observed_result_paths = {
        path.resolve() for path in output_root.rglob("result.json")
    }
    _require_equal(
        observed_result_paths,
        expected_result_paths,
        label="paired-canary exact result inventory",
    )
    if (task_root / "runtime-failure.json").exists():
        raise ReceiptError("paired canary retains a runtime-failure artifact")
    results: dict[tuple[str, str], dict[str, Any]] = {}
    artifacts: list[dict[str, Any]] = []
    for diagnostics_mode, mode, arm in result_specs:
        diagnostic_output_root = output_root / diagnostics_mode
        result_path = (
            diagnostic_output_root
            / mode
            / manifest["case_id"]
            / "result.json"
        )
        validated, artifact = _validate_result_artifact(
            result_path=result_path,
            output_root=diagnostic_output_root,
            config=config,
            manifest=manifest,
            expected_commit=args.expected_commit,
        )
        _validate_canary_result_binding(
            validated,
            diagnostics_mode=diagnostics_mode,
            policy_mode=mode,
            expected_arm=arm,
            expected_label_record=expected_label_record,
        )
        results[(diagnostics_mode, mode)] = validated
        artifact["diagnostics_mode"] = diagnostics_mode
        artifact["policy_mode"] = mode
        artifacts.append(artifact)
    cross_arm_pairing: dict[str, bool] = {}
    for diagnostics_mode in action_canary.DIAGNOSTIC_MODES:
        paired_results = {
            (manifest["case_id"], config["arms"][0]): results[
                (diagnostics_mode, "pi05")
            ],
            (manifest["case_id"], config["arms"][1]): results[
                (diagnostics_mode, "aegis")
            ],
        }
        aggregate.validate_pairs(
            config=config,
            manifests=[manifest],
            results=paired_results,
        )
        cross_arm_pairing[diagnostics_mode] = True
    integration_gates = {
        diagnostics_mode: validate_aegis_canary_integration(
            results[(diagnostics_mode, "aegis")]
        )
        for diagnostics_mode in action_canary.DIAGNOSTIC_MODES
    }
    try:
        action_invariant_evidence = action_canary.validate_four_run_canary(
            results=results,
            output_roots={
                diagnostics_mode: output_root / diagnostics_mode
                for diagnostics_mode in action_canary.DIAGNOSTIC_MODES
            },
            reference_path=action_reference_path,
        )
    except (
        action_canary.ActionInvariantCanaryError,
        action_canary.failure_validation.DiagnosticValidationError,
    ) as error:
        raise ReceiptError(str(error)) from error
    receipt: dict[str, Any] = {
        "schema_version": CANARY_SCHEMA,
        "status": "validated",
        "scientific_result": False,
        "paired_result_valid": True,
        "action_invariance_valid": True,
        "failure_diagnostics_valid": True,
        "contact_schema_version": CONTACT_SCHEMA_V3,
        "contact_model_authority_schema_version": (
            CONTACT_MODEL_AUTHORITY_SCHEMA_V2
        ),
        "cross_arm_pairing": cross_arm_pairing,
        "aegis_integration_gates": integration_gates,
        "action_invariant_evidence": action_invariant_evidence,
        "source_git_commit": args.expected_commit,
        "pi05_tree_sha256": contract["pi05_tree_sha256"],
        "groundingdino_device": contract["groundingdino_device"],
        "run_id": contract["run_id"],
        "case_id": manifest["case_id"],
        "case_ordinal": case_ordinal,
        "run_contract": contract_record,
        "pi05_hash_receipt": hash_receipt,
        "allocation_preflight": preflight,
        "full_label_manifest": full_label_manifest,
        "results": artifacts,
        "slurm": current_slurm,
    }
    receipt["receipt_payload_sha256"] = sha256_bytes(
        canonical_json_bytes(receipt)
    )
    return receipt


def validate_paired_canary_receipt(
    path: Path,
    *,
    expected_file_sha256: str,
    expected_commit: str,
    expected_tree_sha256: str,
    expected_groundingdino_device: str,
    config_path: Path,
    manifest_path: Path,
    manifest_receipt_path: Path,
    config: dict[str, Any],
    manifests: list[dict[str, Any]],
) -> dict[str, Any]:
    value = load_json_object(path, label="paired-canary validation receipt")
    _require_equal(
        sha256_path(path),
        require_sha256(
            expected_file_sha256, label="paired-canary receipt file"
        ),
        label="paired-canary receipt file SHA-256",
    )
    _require_equal(
        value.get("schema_version"),
        CANARY_SCHEMA,
        label="paired-canary receipt schema",
    )
    _require_equal(
        value.get("status"), "validated", label="paired-canary status"
    )
    _require_equal(
        value.get("paired_result_valid"),
        True,
        label="paired-canary result validity",
    )
    _require_equal(
        value.get("scientific_result"),
        False,
        label="paired-canary scientific-result flag",
    )
    payload_sha256 = verify_payload_sha256(
        value,
        field="receipt_payload_sha256",
        label="paired-canary validation receipt",
    )
    _require_equal(
        value.get("source_git_commit"),
        expected_commit,
        label="paired-canary source commit",
    )
    _require_equal(
        value.get("pi05_tree_sha256"),
        expected_tree_sha256,
        label="paired-canary pi0.5 checkpoint",
    )
    _require_equal(
        value.get("groundingdino_device"),
        expected_groundingdino_device,
        label="paired-canary GroundingDINO device",
    )
    recorded_slurm = validate_recorded_allocation_identity(
        value.get("slurm"),
        label="paired-canary Slurm identity",
        require_task_zero=True,
    )
    run_contract_record = value.get("run_contract")
    preflight_record = value.get("allocation_preflight")
    hash_receipt_record = value.get("pi05_hash_receipt")
    full_label_record = value.get("full_label_manifest")
    result_records = value.get("results")
    if not all(
        isinstance(item, Mapping)
        for item in (
            run_contract_record,
            preflight_record,
            hash_receipt_record,
            full_label_record,
        )
    ) or not isinstance(result_records, list):
        raise ReceiptError(
            "paired-canary receipt lacks nested validation evidence"
        )
    run_contract_path = Path(
        str(run_contract_record.get("path", ""))
    ).resolve()
    if run_contract_path.name != "run-contract.tsv":
        raise ReceiptError("paired-canary run-contract path is invalid")
    canary_run_root = run_contract_path.parent
    _require_equal(
        path.resolve(),
        canary_run_root / "paired-canary-validation.json",
        label="paired-canary receipt location",
    )
    preflight_path = Path(str(preflight_record.get("path", ""))).resolve()
    expected_preflight_path = (
        canary_run_root / "tasks/task-0/allocation-preflight.json"
    )
    _require_equal(
        preflight_path,
        expected_preflight_path,
        label="paired-canary preflight location",
    )
    preflight_value = load_json_object(
        preflight_path, label="paired-canary allocation preflight"
    )
    assets = preflight_value.get("assets")
    labels = (
        assets.get("frozen_codex_labels")
        if isinstance(assets, Mapping)
        else None
    )
    if not isinstance(labels, Mapping) or not labels.get("path"):
        raise ReceiptError(
            "paired-canary preflight lacks its frozen label path"
        )
    label_manifest_path = Path(str(labels["path"])).resolve()
    contract, regenerated_contract_record = validate_run_contract(
        run_root=canary_run_root,
        expected_stage="paired-canary",
        expected_commit=expected_commit,
        config_path=config_path,
        manifest_path=manifest_path,
        manifest_receipt_path=manifest_receipt_path,
        label_manifest_path=label_manifest_path,
    )
    _require_equal(
        dict(run_contract_record),
        regenerated_contract_record,
        label="paired-canary nested run-contract record",
    )
    _require_equal(
        contract["run_id"], value.get("run_id"), label="paired-canary run ID"
    )
    try:
        case_ordinal = int(contract["case_ordinal"])
    except ValueError as error:
        raise ReceiptError(
            "paired-canary run contract has invalid case ordinal"
        ) from error
    if not 0 <= case_ordinal < len(manifests):
        raise ReceiptError("paired-canary case ordinal is outside the manifest")
    _require_equal(
        case_ordinal,
        CANARY_CASE_ORDINAL,
        label="paired-canary frozen ordinal",
    )
    manifest = manifests[case_ordinal]
    _require_equal(
        manifest["case_id"],
        CANARY_CASE_ID,
        label="paired-canary frozen case",
    )
    _require_equal(
        value.get("case_ordinal"),
        case_ordinal,
        label="paired-canary case ordinal",
    )
    _require_equal(
        value.get("case_id"),
        manifest["case_id"],
        label="paired-canary case ID",
    )
    regenerated_full_label_record = validate_full_label_manifest(
        label_manifest_path,
        canary_case_id=manifest["case_id"],
    )
    expected_label_record = _load_selected_frozen_label_record(
        label_manifest_path,
        case_id=manifest["case_id"],
    )
    _require_equal(
        dict(full_label_record),
        regenerated_full_label_record,
        label="paired-canary full frozen label manifest",
    )
    hash_receipt_path = Path(
        str(hash_receipt_record.get("path", ""))
    ).resolve()
    regenerated_hash_receipt = validate_pi05_hash_receipt(
        hash_receipt_path,
        expected_file_sha256=contract["pi05_hash_receipt_sha256"],
        expected_tree_sha256=contract["pi05_tree_sha256"],
        expected_commit=expected_commit,
    )
    _require_equal(
        dict(hash_receipt_record),
        regenerated_hash_receipt,
        label="paired-canary nested checkpoint receipt",
    )
    regenerated_preflight = validate_preflight_receipt(
        preflight_path,
        contract=contract,
        expected_slurm=recorded_slurm,
    )
    _require_equal(
        dict(preflight_record),
        regenerated_preflight,
        label="paired-canary nested preflight record",
    )
    output_root = canary_run_root / "tasks/task-0/results"
    if (canary_run_root / "tasks/task-0/runtime-failure.json").exists():
        raise ReceiptError(
            "paired-canary prerequisite retains a runtime-failure artifact"
        )
    result_specs = _canary_result_specs(config)
    expected_result_paths = {
        (
            output_root
            / diagnostics_mode
            / mode
            / manifest["case_id"]
            / "result.json"
        ).resolve()
        for diagnostics_mode, mode, _ in result_specs
    }
    observed_result_paths = {
        result_path.resolve()
        for result_path in output_root.rglob("result.json")
    }
    _require_equal(
        observed_result_paths,
        expected_result_paths,
        label="paired-canary retained result inventory",
    )
    if len(result_records) != 4:
        raise ReceiptError(
            "paired-canary receipt must bind exactly four result records"
        )
    validated_results: dict[tuple[str, str], dict[str, Any]] = {}
    regenerated_result_records: list[dict[str, Any]] = []
    for diagnostics_mode, mode, arm in result_specs:
        diagnostic_output_root = output_root / diagnostics_mode
        result_path = (
            diagnostic_output_root
            / mode
            / manifest["case_id"]
            / "result.json"
        )
        validated, artifact = _validate_result_artifact(
            result_path=result_path,
            output_root=diagnostic_output_root,
            config=config,
            manifest=manifest,
            expected_commit=expected_commit,
        )
        _validate_canary_result_binding(
            validated,
            diagnostics_mode=diagnostics_mode,
            policy_mode=mode,
            expected_arm=arm,
            expected_label_record=expected_label_record,
        )
        validated_results[(diagnostics_mode, mode)] = validated
        artifact["diagnostics_mode"] = diagnostics_mode
        artifact["policy_mode"] = mode
        regenerated_result_records.append(artifact)
    _require_equal(
        result_records,
        regenerated_result_records,
        label="paired-canary nested result records",
    )
    regenerated_cross_arm_pairing: dict[str, bool] = {}
    for diagnostics_mode in action_canary.DIAGNOSTIC_MODES:
        aggregate.validate_pairs(
            config=config,
            manifests=[manifest],
            results={
                (manifest["case_id"], config["arms"][0]): (
                    validated_results[(diagnostics_mode, "pi05")]
                ),
                (manifest["case_id"], config["arms"][1]): (
                    validated_results[(diagnostics_mode, "aegis")]
                ),
            },
        )
        regenerated_cross_arm_pairing[diagnostics_mode] = True
    _require_equal(
        value.get("cross_arm_pairing"),
        regenerated_cross_arm_pairing,
        label="paired-canary cross-arm pairing",
    )
    regenerated_integration_gates = {
        diagnostics_mode: validate_aegis_canary_integration(
            validated_results[(diagnostics_mode, "aegis")]
        )
        for diagnostics_mode in action_canary.DIAGNOSTIC_MODES
    }
    _require_equal(
        value.get("aegis_integration_gates"),
        regenerated_integration_gates,
        label="paired-canary AEGIS integration gates",
    )
    reference_path = _validate_canary_action_reference_path(
        ROOT / "fixtures/vlsa_table1_canary_action_reference.json"
    )
    try:
        regenerated_action_evidence = (
            action_canary.validate_four_run_canary(
                results=validated_results,
                output_roots={
                    diagnostics_mode: output_root / diagnostics_mode
                    for diagnostics_mode in action_canary.DIAGNOSTIC_MODES
                },
                reference_path=reference_path,
            )
        )
    except (
        action_canary.ActionInvariantCanaryError,
        action_canary.failure_validation.DiagnosticValidationError,
    ) as error:
        raise ReceiptError(str(error)) from error
    _require_equal(
        value.get("action_invariant_evidence"),
        regenerated_action_evidence,
        label="paired-canary action-invariant evidence",
    )
    for field in (
        "action_invariance_valid",
        "failure_diagnostics_valid",
    ):
        _require_equal(
            value.get(field),
            True,
            label=f"paired-canary {field}",
        )
    _require_equal(
        value.get("contact_schema_version"),
        CONTACT_SCHEMA_V3,
        label="paired-canary contact schema",
    )
    _require_equal(
        value.get("contact_model_authority_schema_version"),
        CONTACT_MODEL_AUTHORITY_SCHEMA_V2,
        label="paired-canary contact model-authority schema",
    )
    return {
        "path": str(path),
        "sha256": sha256_path(path),
        "receipt_payload_sha256": payload_sha256,
        "run_id": value.get("run_id"),
        "case_id": value.get("case_id"),
        "case_ordinal": value.get("case_ordinal"),
        "slurm": recorded_slurm,
        "run_contract_sha256": regenerated_contract_record["sha256"],
        "allocation_preflight_sha256": regenerated_preflight["sha256"],
        "result_artifact_sha256": [
            record["sha256"] for record in regenerated_result_records
        ],
        "aegis_integration_gates": regenerated_integration_gates,
        "contact_schema_version": CONTACT_SCHEMA_V3,
        "contact_model_authority_schema_version": (
            CONTACT_MODEL_AUTHORITY_SCHEMA_V2
        ),
        "action_invariant_evidence_sha256": sha256_bytes(
            canonical_json_bytes(regenerated_action_evidence)
        ),
    }


def validate_slurm_accounting(
    path: Path,
    *,
    population_array_job_id: str,
) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ReceiptError(f"Slurm accounting evidence is missing: {path}")
    states: dict[int, dict[str, str]] = {}
    raw_job_ids: set[str] = set()
    ignored = 0
    prefix = f"{population_array_job_id}_"
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.strip().split("|")
        if len(fields) < 3:
            if line.strip():
                ignored += 1
            continue
        job_id, job_id_raw, state = (
            fields[0].strip(),
            fields[1].strip(),
            fields[2].strip(),
        )
        if not job_id.startswith(prefix) or "." in job_id:
            ignored += 1
            continue
        suffix = job_id[len(prefix) :]
        if not suffix.isdigit():
            ignored += 1
            continue
        task = int(suffix)
        if task in states:
            raise ReceiptError(f"duplicate Slurm accounting row for task {task}")
        if not job_id_raw.isdigit() or job_id_raw in raw_job_ids:
            raise ReceiptError(
                f"invalid or duplicate raw Slurm job ID for task {task}"
            )
        raw_job_ids.add(job_id_raw)
        states[task] = {
            "job_id": job_id,
            "job_id_raw": job_id_raw,
            "state": state.split()[0].rstrip("+"),
        }
    expected_tasks = set(range(32))
    if set(states) != expected_tasks:
        raise ReceiptError(
            "Slurm accounting does not contain the exact 32 population tasks"
        )
    noncomplete = {
        task: record["state"]
        for task, record in states.items()
        if record["state"] != "COMPLETED"
    }
    if noncomplete:
        raise ReceiptError(
            f"population array contains non-COMPLETED tasks: {noncomplete}"
        )
    return {
        "path": str(path),
        "sha256": sha256_path(path),
        "population_array_job_id": population_array_job_id,
        "tasks": {str(key): states[key] for key in sorted(states)},
        "ignored_non_task_rows": ignored,
    }


def _validate_population_result(
    *,
    path: Path,
    task_results_root: Path,
    config: dict[str, Any],
    manifest: dict[str, Any],
    expected_commit: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    validated, artifact = _validate_result_artifact(
        result_path=path,
        output_root=task_results_root,
        config=config,
        manifest=manifest,
        expected_commit=expected_commit,
    )
    return validated, {
        "relative_result_path": path.as_posix(),
        "result_sha256": artifact["sha256"],
        "result_payload_sha256": artifact["result_payload_sha256"],
        "video_sha256": artifact["video_sha256"],
        "case_id": artifact["case_id"],
        "arm": artifact["arm"],
        "status": artifact["status"],
    }


def _population_result_specs(
    config: Mapping[str, Any],
) -> tuple[tuple[str, str], ...]:
    expected = tuple(
        action_canary.EXPECTED_ARM_BY_POLICY_MODE[mode]
        for mode in action_canary.POLICY_MODES
    )
    _require_equal(
        tuple(config.get("arms", ())),
        expected,
        label="population protocol mode/arm mapping",
    )
    return tuple(
        (
            mode,
            action_canary.EXPECTED_ARM_BY_POLICY_MODE[mode],
        )
        for mode in action_canary.POLICY_MODES
    )


def _compact_population_diagnostic_evidence(
    result: Mapping[str, Any],
    *,
    diagnostic_validation: Mapping[str, Any],
) -> dict[str, Any]:
    case_id = str(result.get("case_id", ""))
    mode = str(result.get("mode", ""))
    arm = str(result.get("arm", ""))
    label = f"{case_id}/{mode}"
    record = result.get("failure_diagnostics")
    if not isinstance(record, Mapping):
        raise ReceiptError(f"{label}: failure diagnostics are missing")
    terminal_frame = record.get("terminal_frame")
    contacts = record.get("contacts")
    geometry = record.get("geometry")
    obstacle = result.get("obstacle")
    active_obstacle_name = (
        obstacle.get("active_name")
        if isinstance(obstacle, Mapping)
        else None
    )
    if not isinstance(terminal_frame, Mapping) or not isinstance(
        contacts, Mapping
    ):
        raise ReceiptError(
            f"{label}: terminal-frame/contact diagnostics are missing"
        )
    if (
        not isinstance(active_obstacle_name, str)
        or not active_obstacle_name
        or contacts.get("active_obstacle_name") != active_obstacle_name
    ):
        raise ReceiptError(
            f"{label}: detailed contact active-obstacle binding changed"
        )
    terminal_frame_sha256 = require_sha256(
        terminal_frame.get("sha256"),
        label=f"{label}: terminal-frame artifact",
    )
    contacts_sha256 = require_sha256(
        contacts.get("sha256"),
        label=f"{label}: detailed contact artifact",
    )
    contact_payload_sha256 = require_sha256(
        contacts.get("uncompressed_payload_sha256"),
        label=f"{label}: detailed contact payload",
    )
    contact_model_authority_sha256 = require_sha256(
        contacts.get("model_authority_sha256"),
        label=f"{label}: contact model authority",
    )
    contact_task_context_sha256 = require_sha256(
        contacts.get("task_context_sha256"),
        label=f"{label}: contact task context",
    )
    active_obstacle_root_body_id = contacts.get(
        "active_obstacle_root_body_id"
    )
    if (
        type(active_obstacle_root_body_id) is not int
        or active_obstacle_root_body_id < 0
    ):
        raise ReceiptError(
            f"{label}: active-obstacle root-body authority changed"
        )
    _require_equal(
        contacts.get("schema_version"),
        CONTACT_SCHEMA_V3,
        label=f"{label}: detailed contact schema",
    )
    _require_equal(
        tuple(contacts.get("role_taxonomy", ())),
        CONTACT_ROLE_TAXONOMY,
        label=f"{label}: contact role taxonomy",
    )
    _require_equal(
        contacts.get("role_authority_complete"),
        True,
        label=f"{label}: contact role authority",
    )
    event_counts_by_role = contacts.get("event_counts_by_role")
    steps_by_role = contacts.get("steps_with_contact_by_role")
    first_steps_by_role = contacts.get("first_contact_step_by_role")
    if not all(
        isinstance(value, Mapping)
        for value in (
            event_counts_by_role,
            steps_by_role,
            first_steps_by_role,
        )
    ):
        raise ReceiptError(
            f"{label}: authoritative per-role contact summaries are missing"
        )
    for field_name, value in (
        ("event_counts_by_role", event_counts_by_role),
        ("steps_with_contact_by_role", steps_by_role),
        ("first_contact_step_by_role", first_steps_by_role),
    ):
        _require_equal(
            frozenset(value),
            frozenset(CONTACT_ROLE_TAXONOMY),
            label=f"{label}: contact {field_name} keys",
        )
    contact_event_counts_by_role: dict[str, int] = {}
    contact_steps_by_role: dict[str, list[int]] = {}
    contact_first_steps_by_role: dict[str, int | None] = {}
    for role in CONTACT_ROLE_TAXONOMY:
        count = event_counts_by_role[role]
        steps = steps_by_role[role]
        first_step = first_steps_by_role[role]
        if type(count) is not int or count < 0:
            raise ReceiptError(
                f"{label}: contact count for {role} is invalid"
            )
        if (
            not isinstance(steps, list)
            or any(type(step) is not int for step in steps)
            or steps != sorted(set(steps))
        ):
            raise ReceiptError(
                f"{label}: contact steps for {role} are invalid"
            )
        expected_first = None if not steps else steps[0]
        if first_step != expected_first:
            raise ReceiptError(
                f"{label}: first contact step for {role} is invalid"
            )
        if (count == 0) is not (not steps):
            raise ReceiptError(
                f"{label}: contact count/steps disagree for {role}"
            )
        contact_event_counts_by_role[role] = count
        contact_steps_by_role[role] = list(steps)
        contact_first_steps_by_role[role] = first_step
    if sum(contact_event_counts_by_role.values()) != contacts.get(
        "event_count"
    ):
        raise ReceiptError(
            f"{label}: per-role counts do not sum to all contact events"
        )
    _require_equal(
        contacts.get("robot_event_count"),
        contact_event_counts_by_role["robot"],
        label=f"{label}: legacy robot contact count",
    )
    _require_equal(
        contacts.get("nonrobot_event_count"),
        sum(
            contact_event_counts_by_role[role]
            for role in CONTACT_ROLE_TAXONOMY
            if role != "robot"
        ),
        label=f"{label}: legacy nonrobot contact count",
    )
    if (
        contact_event_counts_by_role["unknown"] != 0
        or contact_steps_by_role["unknown"]
        or contact_first_steps_by_role["unknown"] is not None
    ):
        raise ReceiptError(
            f"{label}: unknown contact roles are not publication-valid"
        )
    video_decode = diagnostic_validation.get("video_decode")
    ledger = diagnostic_validation.get("action_invariance_ledger")
    if not isinstance(video_decode, Mapping) or not isinstance(
        ledger, Mapping
    ):
        raise ReceiptError(
            f"{label}: compact diagnostic validation evidence is incomplete"
        )

    geometry_status = "not_applicable"
    geometry_artifact_sha256: str | None = None
    geometry_record_sha256: str | None = None
    geometry_view_statuses: dict[str, str] = {}
    geometry_detection_counts: dict[str, int | None] = {}
    filtering_status: str | None = None
    mvee_status: str | None = None
    if mode == "aegis":
        if not isinstance(geometry, Mapping):
            raise ReceiptError(f"{label}: AEGIS geometry evidence is missing")
        geometry_artifact_sha256 = require_sha256(
            geometry.get("sha256"),
            label=f"{label}: AEGIS geometry artifact",
        )
        geometry_record = geometry.get("record")
        if not isinstance(geometry_record, Mapping):
            raise ReceiptError(
                f"{label}: AEGIS geometry record is missing"
            )
        geometry_status = str(geometry_record.get("status", ""))
        geometry_record_sha256 = sha256_bytes(
            canonical_json_bytes(dict(geometry_record))
        )
        views = geometry_record.get("views")
        if isinstance(views, Mapping):
            for view_name in ("agentview", "backview"):
                view = views.get(view_name)
                if isinstance(view, Mapping):
                    geometry_view_statuses[view_name] = str(
                        view.get("status", "")
                    )
                    detections = view.get("detections")
                    geometry_detection_counts[view_name] = (
                        int(detections["count"])
                        if isinstance(detections, Mapping)
                        and type(detections.get("count")) is int
                        else None
                    )
        filtering = geometry_record.get("filtering")
        mvee = geometry_record.get("mvee")
        filtering_status = (
            str(filtering.get("status"))
            if isinstance(filtering, Mapping)
            else None
        )
        mvee_status = (
            str(mvee.get("status"))
            if isinstance(mvee, Mapping)
            else None
        )
    elif mode == "pi05":
        if not isinstance(geometry, Mapping) or (
            geometry.get("status"),
            geometry.get("reason"),
        ) != ("not_run", "pi05_baseline_arm"):
            raise ReceiptError(
                f"{label}: baseline geometry must be explicitly not run"
            )
        geometry_status = "not_run"
    else:
        raise ReceiptError(f"{label}: unsupported population policy mode")

    qp_rows: list[dict[str, Any]] = []
    solved_qp_actions = 0
    for action in result.get("actions", []):
        if (
            not isinstance(action, Mapping)
            or action.get("control_path") != "aegis_qp"
        ):
            continue
        qp = action.get("qp")
        context = qp.get("context") if isinstance(qp, Mapping) else None
        if not isinstance(qp, Mapping) or not isinstance(context, Mapping):
            raise ReceiptError(f"{label}: validated QP context is missing")
        solved_qp_actions += 1
        qp_rows.append(
            {
                "step": action.get("step"),
                "solver_status": qp.get("solver_status"),
                "qp_sha256": sha256_bytes(
                    canonical_json_bytes(dict(qp))
                ),
                "context_sha256": sha256_bytes(
                    canonical_json_bytes(dict(context))
                ),
            }
        )
    method_failure = result.get("method_failure")
    terminal_qp_failure = (
        isinstance(method_failure, Mapping)
        and (
            method_failure.get("component"),
            method_failure.get("phase"),
        )
        == ("aegis_qp", "control")
    )
    if terminal_qp_failure:
        qp_rows.append(
            {
                "step": method_failure.get("step"),
                "terminal_failure": True,
                "failure_type": (
                    method_failure.get("diagnostics", {}).get("failure_type")
                    if isinstance(
                        method_failure.get("diagnostics"), Mapping
                    )
                    else None
                ),
                "method_failure_sha256": sha256_bytes(
                    canonical_json_bytes(dict(method_failure))
                ),
            }
        )

    goal_progress = result.get("goal_progress")
    settled = result.get("settled_observation")
    label_record = (
        settled.get("label_record")
        if isinstance(settled, Mapping)
        else None
    )
    if not isinstance(goal_progress, Mapping) or not isinstance(
        label_record, Mapping
    ):
        raise ReceiptError(
            f"{label}: native goal or frozen label evidence is missing"
        )
    decoded_frame_count = video_decode.get("decoded_frame_count")
    if type(decoded_frame_count) is not int or decoded_frame_count < 1:
        raise ReceiptError(f"{label}: decoded video frame count is invalid")
    for field in (
        "snapshot_count",
        "event_count",
        "robot_event_count",
        "nonrobot_event_count",
    ):
        if type(contacts.get(field)) is not int or contacts[field] < 0:
            raise ReceiptError(
                f"{label}: detailed contact {field} is invalid"
            )

    compact = {
        "case_id": case_id,
        "mode": mode,
        "arm": arm,
        "result_status": result.get("status"),
        "diagnostic_validation_sha256": sha256_bytes(
            canonical_json_bytes(dict(diagnostic_validation))
        ),
        "failure_diagnostics_record_sha256": sha256_bytes(
            canonical_json_bytes(dict(record))
        ),
        "frozen_label_record_sha256": sha256_bytes(
            canonical_json_bytes(dict(label_record))
        ),
        "action_invariance_ledger_sha256": sha256_bytes(
            canonical_json_bytes(dict(ledger))
        ),
        "action_count": int(ledger["action_count"]),
        "policy_query_count": int(ledger["policy_query_count"]),
        "terminal_frame_sha256": terminal_frame_sha256,
        "geometry_status": geometry_status,
        "geometry_artifact_sha256": geometry_artifact_sha256,
        "geometry_record_sha256": geometry_record_sha256,
        "geometry_view_statuses": geometry_view_statuses,
        "geometry_detection_counts": geometry_detection_counts,
        "filtering_status": filtering_status,
        "mvee_status": mvee_status,
        "qp_solved_action_count": solved_qp_actions,
        "qp_terminal_failure": terminal_qp_failure,
        "qp_evidence_sha256": sha256_bytes(
            canonical_json_bytes(qp_rows)
        ),
        "contacts_sha256": contacts_sha256,
        "active_obstacle_name": active_obstacle_name,
        "contacts_uncompressed_payload_sha256": (
            contact_payload_sha256
        ),
        "contact_model_authority_sha256": (
            contact_model_authority_sha256
        ),
        "contact_task_context_sha256": contact_task_context_sha256,
        "active_obstacle_root_body_id": active_obstacle_root_body_id,
        "contact_schema_version": CONTACT_SCHEMA_V3,
        "contact_model_authority_schema_version": (
            CONTACT_MODEL_AUTHORITY_SCHEMA_V2
        ),
        "contact_role_taxonomy": list(CONTACT_ROLE_TAXONOMY),
        "contact_role_authority_complete": True,
        "contact_event_counts_by_role": contact_event_counts_by_role,
        "contact_steps_with_contact_by_role": contact_steps_by_role,
        "contact_first_contact_step_by_role": (
            contact_first_steps_by_role
        ),
        "contact_snapshot_count": int(contacts["snapshot_count"]),
        "contact_event_count": int(contacts["event_count"]),
        "contact_robot_event_count": int(contacts["robot_event_count"]),
        "contact_nonrobot_event_count": int(
            contacts["nonrobot_event_count"]
        ),
        "native_goal_progress_sha256": sha256_bytes(
            canonical_json_bytes(dict(goal_progress))
        ),
        "decoded_video_sha256": sha256_bytes(
            canonical_json_bytes(dict(video_decode))
        ),
        "decoded_video_frame_count": decoded_frame_count,
    }
    compact["compact_evidence_sha256"] = sha256_bytes(
        canonical_json_bytes(compact)
    )
    return compact


def _validate_population_task_results(
    *,
    task_index: int,
    task_root: Path,
    run_root: Path,
    config: dict[str, Any],
    task_manifests: list[dict[str, Any]],
    label_records: Mapping[str, Mapping[str, Any]],
    expected_commit: str,
) -> dict[str, Any]:
    if len(task_manifests) != 50:
        raise ReceiptError(
            f"task-{task_index} must bind exactly 50 manifest cases"
        )
    result_root = task_root / "results"
    specs = _population_result_specs(config)
    expected_paths = {
        (
            result_root
            / policy_mode
            / str(manifest["case_id"])
            / "result.json"
        ).resolve()
        for manifest in task_manifests
        for policy_mode, _ in specs
    }
    observed_paths = {
        path.resolve() for path in result_root.rglob("result.json")
    }
    _require_equal(
        observed_paths,
        expected_paths,
        label=f"task-{task_index} exact result inventory",
    )

    inventory: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    mode_counts: Counter[str] = Counter()
    arm_counts: Counter[str] = Counter()
    geometry_status_counts: Counter[str] = Counter()
    qp_solved_action_count = 0
    qp_terminal_failure_result_count = 0
    contact_snapshot_count = 0
    contact_event_count = 0
    contact_robot_event_count = 0
    contact_nonrobot_event_count = 0
    contact_event_counts_by_role: Counter[str] = Counter(
        {role: 0 for role in CONTACT_ROLE_TAXONOMY}
    )
    contact_case_step_counts_by_role: Counter[str] = Counter(
        {role: 0 for role in CONTACT_ROLE_TAXONOMY}
    )
    contact_results_with_role_by_role: Counter[str] = Counter(
        {role: 0 for role in CONTACT_ROLE_TAXONOMY}
    )
    contact_first_step_histograms_by_role: dict[str, Counter[str]] = {
        role: Counter() for role in CONTACT_ROLE_TAXONOMY
    }
    decoded_video_frame_count = 0

    # A single case pair is the largest collection of full result dictionaries
    # retained at any time.  Only compact, hash-bound evidence escapes this
    # loop.
    for manifest in task_manifests:
        case_id = str(manifest["case_id"])
        expected_label_record = label_records.get(case_id)
        if not isinstance(expected_label_record, Mapping):
            raise ReceiptError(
                f"task-{task_index}/{case_id}: frozen label is missing"
            )
        pair: dict[tuple[str, str], dict[str, Any]] = {}
        pair_items: list[dict[str, Any]] = []
        for policy_mode, arm in specs:
            result_path = (
                result_root / policy_mode / case_id / "result.json"
            )
            validated, item = _validate_population_result(
                path=result_path,
                task_results_root=result_root,
                config=config,
                manifest=manifest,
                expected_commit=expected_commit,
            )
            _validate_population_result_binding(
                validated,
                task_index=task_index,
                policy_mode=policy_mode,
                expected_arm=arm,
                expected_label_record=expected_label_record,
            )
            try:
                diagnostic_validation = (
                    failure_validation.validate_diagnostic_result(
                        validated,
                        output_root=result_root,
                        # Population must retain typed perception/geometry/QP
                        # method failures; ready geometry is only the canary's
                        # positive integration gate.
                        require_ready_geometry=False,
                    )
                )
            except failure_validation.DiagnosticValidationError as error:
                raise ReceiptError(
                    f"task-{task_index}/{case_id}/{policy_mode}: "
                    f"deep diagnostics invalid: {error}"
                ) from error
            compact = _compact_population_diagnostic_evidence(
                validated,
                diagnostic_validation=diagnostic_validation,
            )
            item["relative_result_path"] = str(
                result_path.relative_to(run_root)
            )
            item["case_ordinal"] = int(manifest["case_ordinal"])
            item["task_index"] = task_index
            item["policy_mode"] = policy_mode
            item["diagnostics"] = compact
            pair[(case_id, arm)] = validated
            pair_items.append(item)
        try:
            aggregate.validate_pairs(
                config=config,
                manifests=[manifest],
                results=pair,
            )
        except aggregate.AggregationError as error:
            raise ReceiptError(
                f"task-{task_index}/{case_id}: cross-arm pairing invalid: "
                f"{error}"
            ) from error

        for item in pair_items:
            compact = item["diagnostics"]
            inventory.append(item)
            status_counts[str(item.get("status", ""))] += 1
            mode_counts[str(compact["mode"])] += 1
            arm_counts[str(compact["arm"])] += 1
            geometry_status_counts[
                f"{compact['mode']}:{compact['geometry_status']}"
            ] += 1
            qp_solved_action_count += int(
                compact["qp_solved_action_count"]
            )
            qp_terminal_failure_result_count += int(
                compact["qp_terminal_failure"]
            )
            contact_snapshot_count += int(
                compact["contact_snapshot_count"]
            )
            contact_event_count += int(compact["contact_event_count"])
            contact_robot_event_count += int(
                compact["contact_robot_event_count"]
            )
            contact_nonrobot_event_count += int(
                compact["contact_nonrobot_event_count"]
            )
            for role in CONTACT_ROLE_TAXONOMY:
                role_count = int(
                    compact["contact_event_counts_by_role"][role]
                )
                role_steps = compact[
                    "contact_steps_with_contact_by_role"
                ][role]
                role_first = compact[
                    "contact_first_contact_step_by_role"
                ][role]
                contact_event_counts_by_role[role] += role_count
                contact_case_step_counts_by_role[role] += len(role_steps)
                contact_results_with_role_by_role[role] += int(
                    role_count > 0
                )
                histogram_key = (
                    "none" if role_first is None else str(role_first)
                )
                contact_first_step_histograms_by_role[role][
                    histogram_key
                ] += 1
            decoded_video_frame_count += int(
                compact["decoded_video_frame_count"]
            )
        del pair
        del pair_items
        del validated
        del diagnostic_validation

    for role in CONTACT_ROLE_TAXONOMY:
        _require_equal(
            sum(contact_first_step_histograms_by_role[role].values()),
            100,
            label=(
                f"task-{task_index} contact first-step accounting/{role}"
            ),
        )
    _require_equal(
        contact_event_counts_by_role["unknown"],
        0,
        label=f"task-{task_index} unknown-role contact events",
    )
    return {
        "inventory": inventory,
        "status_counts": status_counts,
        "mode_counts": mode_counts,
        "arm_counts": arm_counts,
        "geometry_status_counts": geometry_status_counts,
        "qp_solved_action_count": qp_solved_action_count,
        "qp_terminal_failure_result_count": (
            qp_terminal_failure_result_count
        ),
        "contact_snapshot_count": contact_snapshot_count,
        "contact_event_count": contact_event_count,
        "contact_robot_event_count": contact_robot_event_count,
        "contact_nonrobot_event_count": contact_nonrobot_event_count,
        "contact_event_counts_by_role": dict(
            contact_event_counts_by_role
        ),
        "contact_case_step_counts_by_role": dict(
            contact_case_step_counts_by_role
        ),
        "contact_results_with_role_by_role": dict(
            contact_results_with_role_by_role
        ),
        "contact_first_step_histograms_by_role": {
            role: dict(
                sorted(
                    contact_first_step_histograms_by_role[role].items(),
                    key=lambda item: (
                        item[0] != "none",
                        (
                            int(item[0])
                            if item[0] != "none"
                            else -1
                        ),
                    ),
                )
            )
            for role in CONTACT_ROLE_TAXONOMY
        },
        "decoded_video_frame_count": decoded_video_frame_count,
    }


def validate_population_prepublish(args: argparse.Namespace) -> dict[str, Any]:
    run_root = args.run_root.resolve()
    config_path = args.config.resolve()
    manifest_path = args.manifest.resolve()
    manifest_receipt_path = args.manifest_receipt.resolve()
    label_manifest_path = args.labels.resolve()
    contract, contract_record = validate_run_contract(
        run_root=run_root,
        expected_stage="population",
        expected_commit=args.expected_commit,
        config_path=config_path,
        manifest_path=manifest_path,
        manifest_receipt_path=manifest_receipt_path,
        label_manifest_path=label_manifest_path,
    )
    hash_receipt = validate_pi05_hash_receipt(
        args.pi05_hash_receipt.resolve(),
        expected_file_sha256=contract["pi05_hash_receipt_sha256"],
        expected_tree_sha256=contract["pi05_tree_sha256"],
        expected_commit=args.expected_commit,
    )
    config, manifests = aggregate.load_protocol(
        config_path, manifest_receipt_path, manifest_path
    )
    canary_receipt = validate_paired_canary_receipt(
        args.paired_canary_receipt.resolve(),
        expected_file_sha256=contract["paired_canary_receipt_sha256"],
        expected_commit=args.expected_commit,
        expected_tree_sha256=contract["pi05_tree_sha256"],
        expected_groundingdino_device=contract["groundingdino_device"],
        config_path=config_path,
        manifest_path=manifest_path,
        manifest_receipt_path=manifest_receipt_path,
        config=config,
        manifests=manifests,
    )
    accounting = validate_slurm_accounting(
        args.slurm_accounting.resolve(),
        population_array_job_id=args.population_array_job_id,
    )
    publisher_retry_authority = None
    if args.publisher_retry_authority is not None:
        publisher_retry_authority = validate_publisher_retry_authority(
            args.publisher_retry_authority.resolve(),
            run_root=run_root,
            expected_population_commit=args.expected_commit,
            expected_run_id=contract["run_id"],
            expected_population_array_job_id=args.population_array_job_id,
        )
    if len(manifests) != 1600:
        raise ReceiptError("population publisher requires exactly 1,600 cases")
    full_label_manifest = validate_full_label_manifest(
        label_manifest_path,
        canary_case_id=CANARY_CASE_ID,
    )
    label_records = _load_population_frozen_label_records(
        label_manifest_path,
        manifests=manifests,
    )
    tasks_root = run_root / "tasks"
    observed_task_names = {
        path.name
        for path in tasks_root.iterdir()
        if path.is_dir() and not path.is_symlink()
    }
    expected_task_names = {f"task-{index}" for index in range(32)}
    _require_equal(
        observed_task_names,
        expected_task_names,
        label="population task directories",
    )
    preflights: list[dict[str, Any]] = []
    inventory: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    mode_counts: Counter[str] = Counter()
    arm_counts: Counter[str] = Counter()
    geometry_status_counts: Counter[str] = Counter()
    qp_solved_action_count = 0
    qp_terminal_failure_result_count = 0
    contact_snapshot_count = 0
    contact_event_count = 0
    contact_robot_event_count = 0
    contact_nonrobot_event_count = 0
    contact_event_counts_by_role: Counter[str] = Counter(
        {role: 0 for role in CONTACT_ROLE_TAXONOMY}
    )
    contact_case_step_counts_by_role: Counter[str] = Counter(
        {role: 0 for role in CONTACT_ROLE_TAXONOMY}
    )
    contact_results_with_role_by_role: Counter[str] = Counter(
        {role: 0 for role in CONTACT_ROLE_TAXONOMY}
    )
    contact_first_step_histograms_by_role: dict[str, Counter[str]] = {
        role: Counter() for role in CONTACT_ROLE_TAXONOMY
    }
    task_diagnostic_summaries: list[dict[str, Any]] = []
    decoded_video_frame_count = 0
    for task_index in range(32):
        task_root = tasks_root / f"task-{task_index}"
        if (task_root / "runtime-failure.json").exists():
            raise ReceiptError(f"task-{task_index} has a runtime-failure artifact")
        preflight = validate_preflight_receipt(
            task_root / "allocation-preflight.json",
            contract=contract,
            expected_slurm=None,
            expected_array_job_id=args.population_array_job_id,
            expected_array_task_id=str(task_index),
        )
        preflights.append(preflight)
        first_ordinal = task_index * 50
        task_validation = _validate_population_task_results(
            task_index=task_index,
            task_root=task_root,
            run_root=run_root,
            config=config,
            task_manifests=manifests[
                first_ordinal : first_ordinal + 50
            ],
            label_records=label_records,
            expected_commit=args.expected_commit,
        )
        inventory.extend(task_validation["inventory"])
        status_counts.update(task_validation["status_counts"])
        mode_counts.update(task_validation["mode_counts"])
        arm_counts.update(task_validation["arm_counts"])
        geometry_status_counts.update(
            task_validation["geometry_status_counts"]
        )
        qp_solved_action_count += int(
            task_validation["qp_solved_action_count"]
        )
        qp_terminal_failure_result_count += int(
            task_validation["qp_terminal_failure_result_count"]
        )
        contact_snapshot_count += int(
            task_validation["contact_snapshot_count"]
        )
        contact_event_count += int(
            task_validation["contact_event_count"]
        )
        contact_robot_event_count += int(
            task_validation["contact_robot_event_count"]
        )
        contact_nonrobot_event_count += int(
            task_validation["contact_nonrobot_event_count"]
        )
        for role in CONTACT_ROLE_TAXONOMY:
            contact_event_counts_by_role[role] += int(
                task_validation["contact_event_counts_by_role"][role]
            )
            contact_case_step_counts_by_role[role] += int(
                task_validation["contact_case_step_counts_by_role"][role]
            )
            contact_results_with_role_by_role[role] += int(
                task_validation["contact_results_with_role_by_role"][role]
            )
            contact_first_step_histograms_by_role[role].update(
                task_validation[
                    "contact_first_step_histograms_by_role"
                ][role]
            )
        decoded_video_frame_count += int(
            task_validation["decoded_video_frame_count"]
        )
        task_diagnostic_summaries.append(
            {
                "task_index": task_index,
                "result_count": len(task_validation["inventory"]),
                "contact_schema_version": CONTACT_SCHEMA_V3,
                "contact_model_authority_schema_version": (
                    CONTACT_MODEL_AUTHORITY_SCHEMA_V2
                ),
                "contact_role_taxonomy": list(CONTACT_ROLE_TAXONOMY),
                "contact_role_authority_complete": True,
                "contact_event_counts_by_role": dict(
                    task_validation["contact_event_counts_by_role"]
                ),
                "contact_case_step_counts_by_role": dict(
                    task_validation[
                        "contact_case_step_counts_by_role"
                    ]
                ),
                "contact_results_with_role_by_role": dict(
                    task_validation[
                        "contact_results_with_role_by_role"
                    ]
                ),
                "contact_first_step_histograms_by_role": dict(
                    task_validation[
                        "contact_first_step_histograms_by_role"
                    ]
                ),
                "diagnostic_inventory_sha256": sha256_bytes(
                    canonical_json_bytes(
                        [
                            {
                                "case_ordinal": item["case_ordinal"],
                                "task_index": item["task_index"],
                                **dict(item["diagnostics"]),
                            }
                            for item in task_validation["inventory"]
                        ]
                    )
                ),
            }
        )
        del task_validation
    _require_equal(
        len(inventory),
        3200,
        label="complete compact population inventory",
    )
    expected_mode_counts = {"pi05": 1600, "aegis": 1600}
    _require_equal(
        dict(mode_counts),
        expected_mode_counts,
        label="population diagnostics mode counts",
    )
    _require_equal(
        dict(arm_counts),
        {str(arm): 1600 for arm in config["arms"]},
        label="population diagnostics arm counts",
    )
    _require_equal(
        contact_event_counts_by_role["unknown"],
        0,
        label="population unknown-role contact events",
    )
    for role in CONTACT_ROLE_TAXONOMY:
        _require_equal(
            sum(contact_first_step_histograms_by_role[role].values()),
            3200,
            label=f"population contact first-step accounting/{role}",
        )
    diagnostic_inventory = [
        {
            "case_ordinal": item["case_ordinal"],
            "task_index": item["task_index"],
            **dict(item["diagnostics"]),
        }
        for item in inventory
    ]
    receipt: dict[str, Any] = {
        "schema_version": POPULATION_PREPUBLISH_SCHEMA,
        "status": "validated",
        "scientific_result": False,
        "complete_paired_population": True,
        "no_results_dropped": True,
        "source_git_commit": args.expected_commit,
        "run_id": contract["run_id"],
        "population_array_job_id": args.population_array_job_id,
        "pi05_tree_sha256": contract["pi05_tree_sha256"],
        "groundingdino_device": contract["groundingdino_device"],
        "run_contract": contract_record,
        "pi05_hash_receipt": hash_receipt,
        "paired_canary_receipt": canary_receipt,
        "full_label_manifest": full_label_manifest,
        "slurm_accounting": accounting,
        "allocation_preflights": {
            "count": len(preflights),
            "inventory_sha256": sha256_bytes(
                canonical_json_bytes(preflights)
            ),
        },
        "result_artifacts": {
            "count": len(inventory),
            "status_counts": dict(sorted(status_counts.items())),
            "inventory_sha256": sha256_bytes(
                canonical_json_bytes(inventory)
            ),
        },
        "failure_diagnostics": {
            "validator": (
                "analysis.validate_aegis_failure_diagnostics."
                "validate_diagnostic_result"
            ),
            "count": len(diagnostic_inventory),
            "all_results_deep_validated": True,
            "require_ready_geometry": False,
            "mode_counts": dict(sorted(mode_counts.items())),
            "arm_counts": dict(sorted(arm_counts.items())),
            "geometry_status_counts": dict(
                sorted(geometry_status_counts.items())
            ),
            "qp_solved_action_count": qp_solved_action_count,
            "qp_terminal_failure_result_count": (
                qp_terminal_failure_result_count
            ),
            "contact_snapshot_count": contact_snapshot_count,
            "contact_event_count": contact_event_count,
            "contact_robot_event_count": (
                contact_robot_event_count
            ),
            "contact_nonrobot_event_count": (
                contact_nonrobot_event_count
            ),
            "contact_schema_version": CONTACT_SCHEMA_V3,
            "contact_model_authority_schema_version": (
                CONTACT_MODEL_AUTHORITY_SCHEMA_V2
            ),
            "contact_role_taxonomy": list(CONTACT_ROLE_TAXONOMY),
            "contact_role_authority_complete": True,
            "contact_event_counts_by_role": dict(
                contact_event_counts_by_role
            ),
            "contact_case_step_counts_by_role": dict(
                contact_case_step_counts_by_role
            ),
            "contact_results_with_role_by_role": dict(
                contact_results_with_role_by_role
            ),
            "contact_first_step_histograms_by_role": {
                role: dict(
                    sorted(
                        contact_first_step_histograms_by_role[
                            role
                        ].items(),
                        key=lambda item: (
                            item[0] != "none",
                            (
                                int(item[0])
                                if item[0] != "none"
                                else -1
                            ),
                        ),
                    )
                )
                for role in CONTACT_ROLE_TAXONOMY
            },
            "unknown_role_event_count": (
                contact_event_counts_by_role["unknown"]
            ),
            "task_summaries": task_diagnostic_summaries,
            "task_summaries_sha256": sha256_bytes(
                canonical_json_bytes(task_diagnostic_summaries)
            ),
            "decoded_video_frame_count": decoded_video_frame_count,
            "inventory_sha256": sha256_bytes(
                canonical_json_bytes(diagnostic_inventory)
            ),
        },
        "validation_memory_shape": {
            "streaming_unit": "one_case_pair",
            "maximum_live_full_result_records": 2,
            "full_result_records_retained": 0,
            "retained_records": "compact_hash_bound_summaries_only",
        },
    }
    if publisher_retry_authority is not None:
        receipt["publisher_retry_authority"] = publisher_retry_authority
    receipt["receipt_payload_sha256"] = sha256_bytes(
        canonical_json_bytes(receipt)
    )
    return receipt


def publisher_allocation_identity(
    environment: Mapping[str, str],
    *,
    expected_population_array_job_id: str,
) -> dict[str, str]:
    required = (
        "SLURM_JOB_ID",
        "SLURM_JOB_DEPENDENCY",
        "SLURMD_NODENAME",
    )
    missing = [key for key in required if not environment.get(key)]
    if missing:
        raise ReceiptError(
            f"population finalization requires allocation fields {missing}"
        )
    host = str(environment["SLURMD_NODENAME"])
    if host == "worker-3" or host.startswith(("login", "login-restricted")):
        raise ReceiptError(f"publisher cannot execute on {host}")
    expected_dependency = f"afterany:{expected_population_array_job_id}"
    _require_equal(
        environment["SLURM_JOB_DEPENDENCY"],
        expected_dependency,
        label="publisher Slurm dependency",
    )
    return {
        "job_id": str(environment["SLURM_JOB_ID"]),
        "host": host,
        "dependency": str(environment["SLURM_JOB_DEPENDENCY"]),
    }


def finalize_population_publication(args: argparse.Namespace) -> dict[str, Any]:
    run_root = args.run_root.resolve()
    prepublish_path = args.prepublish_receipt.resolve()
    prepublish = load_json_object(
        prepublish_path, label="population prepublish receipt"
    )
    _require_equal(
        prepublish.get("schema_version"),
        POPULATION_PREPUBLISH_SCHEMA,
        label="population prepublish schema",
    )
    _require_equal(
        prepublish.get("status"),
        "validated",
        label="population prepublish status",
    )
    _require_equal(
        prepublish.get("complete_paired_population"),
        True,
        label="population completeness",
    )
    prepublish_payload_sha256 = verify_payload_sha256(
        prepublish,
        field="receipt_payload_sha256",
        label="population prepublish receipt",
    )
    publisher_retry_authority = prepublish.get(
        "publisher_retry_authority"
    )
    if publisher_retry_authority is not None:
        if not isinstance(publisher_retry_authority, Mapping):
            raise ReceiptError(
                "population prepublish publisher retry authority is invalid"
            )
        authority_path = Path(
            str(publisher_retry_authority.get("path", ""))
        ).resolve()
        try:
            authority_path.relative_to(run_root)
        except ValueError as error:
            raise ReceiptError(
                "publisher retry authority escaped the immutable run root"
            ) from error
        _require_equal(
            sha256_path(authority_path),
            publisher_retry_authority.get("sha256"),
            label="population prepublish publisher retry authority file",
        )
        authority = load_json_object(
            authority_path,
            label="population publisher retry authority",
        )
        _require_equal(
            verify_payload_sha256(
                authority,
                field="receipt_payload_sha256",
                label="population publisher retry authority",
            ),
            publisher_retry_authority.get("receipt_payload_sha256"),
            label="population publisher retry authority payload",
        )
    summary_path = args.summary.resolve()
    summary = load_json_object(summary_path, label="population summary")
    _require_equal(
        summary.get("schema_version"),
        "vlsa_table1_population_summary.v1",
        label="population summary schema",
    )
    _require_equal(
        summary.get("status"),
        "complete_population_validated",
        label="population summary status",
    )
    population = summary.get("population")
    if not isinstance(population, Mapping):
        raise ReceiptError("population summary population record is missing")
    for field, expected in (
        ("cases", 1600),
        ("arms", 2),
        ("results", 3200),
        ("task_level_groups", 32),
        ("no_results_dropped", True),
    ):
        _require_equal(
            population.get(field),
            expected,
            label=f"population summary/{field}",
        )
    protocol_config, manifests = aggregate.load_protocol(
        args.config.resolve(),
        args.manifest_receipt.resolve(),
        args.manifest.resolve(),
    )
    results_root = args.results.resolve()
    _require_equal(
        results_root,
        run_root / "tasks",
        label="population finalizer results root",
    )
    recomputed_results = aggregate.load_compact_results_streaming(
        [results_root],
        config=protocol_config,
        manifests=manifests,
        # The independently regenerated failure rows below verify both video
        # files and hashes. Avoid reading all 3,200 videos twice here.
        verify_video_files=False,
    )
    recomputed_summary = aggregate.aggregate(
        config=protocol_config,
        manifests=manifests,
        results=recomputed_results,
    )
    if canonical_json_bytes(recomputed_summary) != canonical_json_bytes(
        summary
    ):
        raise ReceiptError(
            "population summary does not match the exact result population"
        )
    expected_case_ids = {
        str(manifest["case_id"]) for manifest in manifests
    }
    current_contract_path = run_root / "run-contract.tsv"
    current_contract = load_tsv_contract(current_contract_path)
    recorded_contract = prepublish.get("run_contract")
    if not isinstance(recorded_contract, Mapping):
        raise ReceiptError("population prepublish run contract is missing")
    _require_equal(
        sha256_path(current_contract_path),
        recorded_contract.get("sha256"),
        label="population prepublish run-contract file",
    )
    for field, expected in (
        ("schema_version", RUN_CONTRACT_SCHEMA),
        ("run_stage", "population"),
        ("run_id", prepublish.get("run_id")),
        ("git_commit", prepublish.get("source_git_commit")),
        ("config_sha256", sha256_path(args.config.resolve())),
        ("manifest_sha256", sha256_path(args.manifest.resolve())),
        (
            "manifest_receipt_sha256",
            sha256_path(args.manifest_receipt.resolve()),
        ),
    ):
        _require_equal(
            current_contract.get(field),
            expected,
            label=f"population finalizer run contract/{field}",
        )
    gallery_path = args.gallery.resolve()
    if gallery_path.is_symlink() or not gallery_path.is_file():
        raise ReceiptError("strict population gallery is missing")
    failure_cases_path = args.failure_cases.resolve()
    failure_report_path = args.failure_report.resolve()
    failure_markdown_path = args.failure_markdown.resolve()
    for path, label in (
        (failure_cases_path, "failure case ledger"),
        (failure_report_path, "failure report"),
        (failure_markdown_path, "failure report Markdown"),
    ):
        if path.is_symlink() or not path.is_file():
            raise ReceiptError(f"strict population {label} is missing")
    failure_report = load_json_object(
        failure_report_path,
        label="population failure report",
    )
    _require_equal(
        failure_report.get("schema_version"),
        "vlsa_table1_aegis_failure_report.v1",
        label="population failure report schema",
    )
    _require_equal(
        failure_report.get("status"),
        "complete_population_failure_analysis",
        label="population failure report status",
    )
    failure_population = failure_report.get("population")
    failure_counts = failure_report.get("counts")
    failure_source = failure_report.get("source")
    if (
        not isinstance(failure_population, Mapping)
        or not isinstance(failure_counts, Mapping)
        or not isinstance(failure_source, Mapping)
    ):
        raise ReceiptError(
            "population failure report lacks population/count/source records"
        )
    for field, expected in (
        ("cases", 1600),
        ("results", 3200),
        ("no_cases_dropped", True),
        ("all_car_failures_classified", True),
        ("causal_limits_preserved", True),
    ):
        _require_equal(
            failure_population.get(field),
            expected,
            label=f"population failure report/{field}",
        )
    for field, expected in (
        ("case_count", 1600),
        ("video_count", 3200),
        ("unclassified_aegis_car_failures", 0),
        ("all_videos_hash_verified", True),
    ):
        _require_equal(
            failure_counts.get(field),
            expected,
            label=f"population failure counts/{field}",
        )
    _require_equal(
        failure_source.get("population_summary_sha256"),
        sha256_path(summary_path),
        label="failure report population summary binding",
    )
    _require_equal(
        failure_source.get("population_validation_receipt_sha256"),
        sha256_path(prepublish_path),
        label="failure report prepublish binding",
    )
    _require_equal(
        failure_source.get("accepted_result_payloads_sha256"),
        summary.get("accepted_result_payloads_sha256"),
        label="failure report accepted-result binding",
    )
    failure_report_payload_sha256 = verify_payload_sha256(
        failure_report,
        field="report_payload_sha256",
        label="population failure report",
    )
    failure_case_ledger: list[dict[str, str]] = []
    failure_result_payload_ledger: list[dict[str, str]] = []
    failure_case_rows: list[dict[str, Any]] = []
    failure_case_ids: set[str] = set()
    with failure_cases_path.open("rb") as stream:
        for line_number, raw_line in enumerate(stream, 1):
            line = raw_line.strip()
            if not line:
                raise ReceiptError(
                    "population failure case ledger contains a blank row"
                )
            try:
                row = json.loads(line)
            except Exception as error:
                raise ReceiptError(
                    "population failure case ledger is not valid JSONL"
                ) from error
            if not isinstance(row, dict):
                raise ReceiptError(
                    "population failure case ledger row is not an object"
                )
            _require_equal(
                row.get("schema_version"),
                "vlsa_table1_aegis_failure_case.v1",
                label=f"failure case row {line_number} schema",
            )
            case_id = row.get("case_id")
            if not isinstance(case_id, str) or case_id in failure_case_ids:
                raise ReceiptError(
                    "population failure case identities are invalid"
                )
            failure_case_ids.add(case_id)
            row_payload_sha256 = verify_payload_sha256(
                row,
                field="record_payload_sha256",
                label=f"population failure case {case_id}",
            )
            failure_case_ledger.append(
                {
                    "case_id": case_id,
                    "record_payload_sha256": row_payload_sha256,
                }
            )
            failure_case_rows.append(row)
            outcomes = row.get("outcomes")
            if not isinstance(outcomes, Mapping):
                raise ReceiptError(
                    f"population failure case {case_id} lacks outcomes"
                )
            for arm in (
                "pi05_translational",
                "pi05_plus_aegis_translational",
            ):
                arm_outcome = outcomes.get(arm)
                if not isinstance(arm_outcome, Mapping):
                    raise ReceiptError(
                        f"population failure case {case_id}/{arm} is missing"
                    )
                failure_result_payload_ledger.append(
                    {
                        "case_id": case_id,
                        "arm": arm,
                        "result_payload_sha256": require_sha256(
                            arm_outcome.get("result_payload_sha256"),
                            label=(
                                "population failure case "
                                f"{case_id}/{arm}/result payload"
                            ),
                        ),
                    }
                )
                compact = recomputed_results.get((case_id, arm))
                if not isinstance(compact, Mapping):
                    raise ReceiptError(
                        "population failure case references an unknown "
                        f"result: {case_id}/{arm}"
                    )
                compact_metrics = compact.get("metrics")
                if not isinstance(compact_metrics, Mapping):
                    raise ReceiptError(
                        f"recomputed result metrics missing: {case_id}/{arm}"
                    )
                for row_field, compact_field in (
                    ("paper_collision", "public_collision"),
                    ("task_success", "task_success"),
                    ("legacy_ets_steps", "legacy_ets_steps"),
                    ("executed_action_count", "executed_action_count"),
                ):
                    _require_equal(
                        arm_outcome.get(row_field),
                        compact_metrics.get(compact_field),
                        label=(
                            "population failure outcome "
                            f"{case_id}/{arm}/{row_field}"
                        ),
                    )
                _require_equal(
                    arm_outcome.get("result_payload_sha256"),
                    compact.get("result_payload_sha256"),
                    label=(
                        "population failure result payload "
                        f"{case_id}/{arm}"
                    ),
                )
    _require_equal(
        len(failure_case_ledger),
        1600,
        label="population failure case count",
    )
    _require_equal(
        failure_case_ids,
        expected_case_ids,
        label="population failure case identities",
    )
    failure_rows_by_case = {
        str(row["case_id"]): row for row in failure_case_rows
    }
    group_size = int(
        protocol_config["population"][
            "expected_cases_per_task_level_group"
        ]
    )
    regenerated_diagnostic_inventory: list[dict[str, Any]] = []
    regenerated_result_inventory: list[dict[str, Any]] = []
    for manifest in manifests:
        case_id = str(manifest["case_id"])
        task_index = int(manifest["case_ordinal"]) // group_size
        task_results_root = (
            results_root / f"task-{task_index}" / "results"
        )
        baseline_path = (
            task_results_root / "pi05" / case_id / "result.json"
        )
        aegis_path = (
            task_results_root / "aegis" / case_id / "result.json"
        )
        baseline, baseline_item = _validate_population_result(
            path=baseline_path,
            task_results_root=task_results_root,
            config=protocol_config,
            manifest=manifest,
            expected_commit=str(prepublish["source_git_commit"]),
        )
        aegis, aegis_item = _validate_population_result(
            path=aegis_path,
            task_results_root=task_results_root,
            config=protocol_config,
            manifest=manifest,
            expected_commit=str(prepublish["source_git_commit"]),
        )
        for policy_mode, result, item, result_path in (
            ("pi05", baseline, baseline_item, baseline_path),
            ("aegis", aegis, aegis_item, aegis_path),
        ):
            diagnostic_validation = (
                failure_validation.validate_diagnostic_result(
                    result,
                    output_root=task_results_root,
                    require_ready_geometry=False,
                )
            )
            compact_diagnostics = (
                _compact_population_diagnostic_evidence(
                    result,
                    diagnostic_validation=diagnostic_validation,
                )
            )
            _require_equal(
                compact_diagnostics.get("mode"),
                policy_mode,
                label=(
                    "population finalizer diagnostic mode "
                    f"{case_id}/{policy_mode}"
                ),
            )
            regenerated_diagnostic_inventory.append(
                {
                    "case_ordinal": int(manifest["case_ordinal"]),
                    "task_index": task_index,
                    **compact_diagnostics,
                }
            )
            item["relative_result_path"] = str(
                result_path.relative_to(run_root)
            )
            item["case_ordinal"] = int(manifest["case_ordinal"])
            item["task_index"] = task_index
            item["policy_mode"] = policy_mode
            item["diagnostics"] = compact_diagnostics
            regenerated_result_inventory.append(item)
        try:
            aggregate.validate_pairs(
                config=protocol_config,
                manifests=[manifest],
                results={
                    (
                        case_id,
                        failure_report_analysis.BASELINE_ARM,
                    ): baseline,
                    (
                        case_id,
                        failure_report_analysis.AEGIS_ARM,
                    ): aegis,
                },
            )
            regenerated_case = (
                failure_report_analysis.build_case_record(
                    manifest=manifest,
                    baseline=baseline,
                    aegis=aegis,
                    baseline_artifact_root=task_results_root,
                    aegis_artifact_root=task_results_root,
                )
            )
        finally:
            del baseline
            del aegis
        if canonical_json_bytes(
            regenerated_case
        ) != canonical_json_bytes(failure_rows_by_case[case_id]):
            raise ReceiptError(
                "population failure case does not regenerate from exact "
                f"paired results: {case_id}"
            )
    prepublish_diagnostics = prepublish.get("failure_diagnostics")
    if not isinstance(prepublish_diagnostics, Mapping):
        raise ReceiptError(
            "population prepublish receipt lacks failure diagnostics"
        )
    _require_equal(
        sha256_bytes(
            canonical_json_bytes(regenerated_diagnostic_inventory)
        ),
        prepublish_diagnostics.get("inventory_sha256"),
        label="population prepublish diagnostic inventory",
    )
    prepublish_results = prepublish.get("result_artifacts")
    if not isinstance(prepublish_results, Mapping):
        raise ReceiptError(
            "population prepublish receipt lacks result artifacts"
        )
    _require_equal(
        sha256_bytes(canonical_json_bytes(regenerated_result_inventory)),
        prepublish_results.get("inventory_sha256"),
        label="population prepublish result inventory",
    )
    _require_equal(
        sha256_bytes(canonical_json_bytes(failure_case_ledger)),
        failure_report.get("case_record_ledger_sha256"),
        label="population failure case ledger binding",
    )
    _require_equal(
        sha256_bytes(
            canonical_json_bytes(
                sorted(
                    failure_result_payload_ledger,
                    key=lambda row: (row["case_id"], row["arm"]),
                )
            )
        ),
        summary.get("accepted_result_payloads_sha256"),
        label="population failure accepted-result ledger binding",
    )
    regenerated_counts = (
        failure_report_analysis.validate_report_against_summary(
            failure_case_rows,
            summary=summary,
            expected_cases=1600,
        )
    )
    _require_equal(
        regenerated_counts,
        failure_report.get("counts"),
        label="population failure report recomputed counts",
    )
    report_memory = failure_report.get("validation_memory_shape")
    if not isinstance(report_memory, Mapping):
        raise ReceiptError(
            "population failure report lacks its streaming memory shape"
        )
    regenerated_report = failure_report_analysis.build_report(
        failure_case_rows,
        summary=summary,
        summary_sha256=sha256_path(summary_path),
        validation_receipt_sha256=sha256_path(prepublish_path),
        expected_cases=1600,
        streaming_stats={
            "maximum_live_full_result_records": 2,
            "compact_case_records_retained": 1600,
        },
    )
    if canonical_json_bytes(regenerated_report) != canonical_json_bytes(
        failure_report
    ):
        raise ReceiptError(
            "population failure report does not regenerate from its rows"
        )
    try:
        markdown_text = failure_markdown_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise ReceiptError(
            "population failure Markdown is unreadable"
        ) from error
    _require_equal(
        markdown_text,
        failure_report_analysis.render_markdown(failure_report),
        label="population failure Markdown",
    )
    try:
        gallery_text = gallery_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise ReceiptError("strict population gallery is unreadable") from error
    for marker, expected in (
        ('<article class="case-card"', 1600),
        ('<section class="arm-panel"', 3200),
        ("<video controls", 3200),
        ("Open MP4", 3200),
    ):
        _require_equal(
            gallery_text.count(marker),
            expected,
            label=f"strict population gallery marker {marker}",
        )
    if any(
        gallery_text.count(f"<h2>{case_id}</h2>") != 1
        for case_id in expected_case_ids
    ):
        raise ReceiptError(
            "strict population gallery case identities are incomplete"
        )
    gallery_records = gallery_analysis.load_result_records(
        [results_root]
    )
    regenerated_gallery, gallery_warnings = (
        gallery_analysis.build_gallery(
            summary=summary,
            records=gallery_records,
            output_root=gallery_path.parent,
            allow_partial=False,
        )
    )
    if gallery_warnings:
        raise ReceiptError(
            "strict population gallery regeneration produced warnings: "
            f"{gallery_warnings[:5]}"
        )
    _require_equal(
        gallery_text,
        regenerated_gallery,
        label="strict population gallery exact regeneration",
    )
    try:
        gallery_path.relative_to(run_root)
        summary_path.relative_to(run_root)
        prepublish_path.relative_to(run_root)
        failure_cases_path.relative_to(run_root)
        failure_report_path.relative_to(run_root)
        failure_markdown_path.relative_to(run_root)
    except ValueError as error:
        raise ReceiptError("publication artifacts must remain under the run root") from error
    publisher_slurm = publisher_allocation_identity(
        dict(os.environ),
        expected_population_array_job_id=str(
            prepublish["population_array_job_id"]
        ),
    )
    receipt: dict[str, Any] = {
        "schema_version": POPULATION_PUBLICATION_SCHEMA,
        "status": "published",
        "scientific_result": False,
        "complete_paired_population": True,
        "no_results_dropped": True,
        "source_git_commit": prepublish["source_git_commit"],
        "run_id": prepublish["run_id"],
        "population_array_job_id": prepublish[
            "population_array_job_id"
        ],
        "publisher_slurm": publisher_slurm,
        "prepublish_receipt": {
            "path": str(prepublish_path),
            "sha256": sha256_path(prepublish_path),
            "receipt_payload_sha256": prepublish_payload_sha256,
        },
        "summary": {
            "path": str(summary_path),
            "sha256": sha256_path(summary_path),
        },
        "gallery": {
            "path": str(gallery_path),
            "sha256": sha256_path(gallery_path),
        },
        "failure_analysis": {
            "cases": {
                "path": str(failure_cases_path),
                "sha256": sha256_path(failure_cases_path),
                "count": len(failure_case_ledger),
            },
            "report": {
                "path": str(failure_report_path),
                "sha256": sha256_path(failure_report_path),
                "report_payload_sha256": (
                    failure_report_payload_sha256
                ),
            },
            "markdown": {
                "path": str(failure_markdown_path),
                "sha256": sha256_path(failure_markdown_path),
            },
        },
    }
    if publisher_retry_authority is not None:
        _require_equal(
            publisher_retry_authority.get("publisher_slurm"),
            publisher_slurm,
            label="population publisher retry allocation binding",
        )
        receipt["publisher_retry_authority"] = dict(
            publisher_retry_authority
        )
    receipt["receipt_payload_sha256"] = sha256_bytes(
        canonical_json_bytes(receipt)
    )
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate immutable AEGIS run infrastructure artifacts"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    canary = subparsers.add_parser("paired-canary")
    canary.add_argument("--run-root", type=Path, required=True)
    canary.add_argument("--expected-commit", required=True)
    canary.add_argument("--config", type=Path, required=True)
    canary.add_argument("--manifest", type=Path, required=True)
    canary.add_argument("--manifest-receipt", type=Path, required=True)
    canary.add_argument("--labels", type=Path, required=True)
    canary.add_argument("--pi05-hash-receipt", type=Path, required=True)
    canary.add_argument("--action-reference", type=Path, required=True)
    canary.add_argument("--case-ordinal", type=int)
    canary.add_argument("--output", type=Path, required=True)
    population = subparsers.add_parser("population-prepublish")
    population.add_argument("--run-root", type=Path, required=True)
    population.add_argument("--expected-commit", required=True)
    population.add_argument("--config", type=Path, required=True)
    population.add_argument("--manifest", type=Path, required=True)
    population.add_argument("--manifest-receipt", type=Path, required=True)
    population.add_argument("--labels", type=Path, required=True)
    population.add_argument("--pi05-hash-receipt", type=Path, required=True)
    population.add_argument(
        "--paired-canary-receipt", type=Path, required=True
    )
    population.add_argument("--population-array-job-id", required=True)
    population.add_argument("--slurm-accounting", type=Path, required=True)
    population.add_argument(
        "--publisher-retry-authority",
        type=Path,
    )
    population.add_argument("--output", type=Path, required=True)
    finalize = subparsers.add_parser("population-finalize")
    finalize.add_argument("--run-root", type=Path, required=True)
    finalize.add_argument("--prepublish-receipt", type=Path, required=True)
    finalize.add_argument("--config", type=Path, required=True)
    finalize.add_argument(
        "--manifest-receipt", type=Path, required=True
    )
    finalize.add_argument("--manifest", type=Path, required=True)
    finalize.add_argument("--results", type=Path, required=True)
    finalize.add_argument("--summary", type=Path, required=True)
    finalize.add_argument("--gallery", type=Path, required=True)
    finalize.add_argument("--failure-cases", type=Path, required=True)
    finalize.add_argument("--failure-report", type=Path, required=True)
    finalize.add_argument("--failure-markdown", type=Path, required=True)
    finalize.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "paired-canary":
            receipt = validate_paired_canary(args)
            expected_parent = args.run_root.resolve()
        elif args.command == "population-prepublish":
            receipt = validate_population_prepublish(args)
            expected_parent = args.output.resolve().parent
        elif args.command == "population-finalize":
            receipt = finalize_population_publication(args)
            expected_parent = args.run_root.resolve()
        else:
            raise ReceiptError(f"unsupported validation command {args.command}")
        output = args.output.resolve()
        _require_equal(output.parent, expected_parent, label="receipt parent")
        write_json_exclusive(output, receipt)
        message = {
            "status": receipt["status"],
            "output": str(output),
            "receipt_payload_sha256": receipt[
                "receipt_payload_sha256"
            ],
            "receipt_file_sha256": sha256_path(output),
        }
        if "case_id" in receipt:
            message["case_id"] = receipt["case_id"]
        print(
            json.dumps(message, sort_keys=True)
        )
        return 0
    except (
        KeyError,
        OSError,
        ValueError,
        ReceiptError,
        aggregate.AggregationError,
        failure_report_analysis.FailureReportError,
        failure_validation.DiagnosticValidationError,
        gallery_analysis.GalleryError,
    ) as error:
        print(f"AEGIS artifact validation failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
