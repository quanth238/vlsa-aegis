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
from analysis import validate_aegis_action_invariant_canary as action_canary  # noqa: E402
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
    "vlsa_table1_action_invariant_paired_canary_validation.v1"
)
POPULATION_PREPUBLISH_SCHEMA = (
    "vlsa_table1_population_prepublish_validation.v1"
)
POPULATION_PUBLICATION_SCHEMA = "vlsa_table1_population_publication.v1"
RUN_CONTRACT_SCHEMA = "vlsa_table1_run_contract.v1"
PREFLIGHT_SCHEMA = "vlsa_table1_allocation_preflight.v1"
PI05_HASH_SCHEMA = "vlsa_table1_pi05_hash_receipt.v1"
RESULT_PAYLOAD_FIELD = "result_payload_sha256"
VALID_QP_STATUSES = {"optimal", "optimal_inaccurate"}
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
    if len(manifests) != 1600:
        raise ReceiptError("population publisher requires exactly 1,600 cases")
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
    manifest_by_id = {row["case_id"]: row for row in manifests}
    validated_results: dict[tuple[str, str], dict[str, Any]] = {}
    preflights: list[dict[str, Any]] = []
    inventory: list[dict[str, str]] = []
    status_counts: Counter[str] = Counter()
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
        expected_case_ids = {
            manifests[ordinal]["case_id"]
            for ordinal in range(first_ordinal, first_ordinal + 50)
        }
        result_root = task_root / "results"
        result_paths = sorted(result_root.rglob("result.json"))
        if len(result_paths) != 100:
            raise ReceiptError(
                f"task-{task_index} has {len(result_paths)} results, expected 100"
            )
        observed_keys: set[tuple[str, str]] = set()
        for result_path in result_paths:
            raw = load_json_object(result_path, label="population result")
            case_id = raw.get("case_id")
            arm = raw.get("arm")
            if case_id not in expected_case_ids or arm not in config["arms"]:
                raise ReceiptError(
                    f"task-{task_index} contains an unexpected result "
                    f"{case_id!r}/{arm!r}"
                )
            key = (str(case_id), str(arm))
            if key in observed_keys or key in validated_results:
                raise ReceiptError(f"duplicate population result {key}")
            observed_keys.add(key)
            manifest = manifest_by_id[str(case_id)]
            validated, item = _validate_population_result(
                path=result_path,
                task_results_root=result_root,
                config=config,
                manifest=manifest,
                expected_commit=args.expected_commit,
            )
            validated_results[key] = validated
            item["relative_result_path"] = str(
                result_path.relative_to(run_root)
            )
            inventory.append(item)
            status_counts[str(validated["status"])] += 1
        expected_keys = {
            (case_id, arm)
            for case_id in expected_case_ids
            for arm in config["arms"]
        }
        _require_equal(
            observed_keys,
            expected_keys,
            label=f"task-{task_index} paired result keys",
        )
    expected_all = {
        (row["case_id"], arm)
        for row in manifests
        for arm in config["arms"]
    }
    _require_equal(
        set(validated_results),
        expected_all,
        label="complete paired population",
    )
    aggregate.validate_pairs(
        config=config,
        manifests=manifests,
        results=validated_results,
    )
    inventory.sort(
        key=lambda item: (item["case_id"], item["arm"])
    )
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
    }
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
    gallery_path = args.gallery.resolve()
    if gallery_path.is_symlink() or not gallery_path.is_file():
        raise ReceiptError("strict population gallery is missing")
    try:
        gallery_path.relative_to(run_root)
        summary_path.relative_to(run_root)
        prepublish_path.relative_to(run_root)
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
    }
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
    population.add_argument("--output", type=Path, required=True)
    finalize = subparsers.add_parser("population-finalize")
    finalize.add_argument("--run-root", type=Path, required=True)
    finalize.add_argument("--prepublish-receipt", type=Path, required=True)
    finalize.add_argument("--summary", type=Path, required=True)
    finalize.add_argument("--gallery", type=Path, required=True)
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
    ) as error:
        print(f"AEGIS artifact validation failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
