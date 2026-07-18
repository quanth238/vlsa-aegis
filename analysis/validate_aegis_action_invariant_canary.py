#!/usr/bin/env python3
"""Validate the four-run, action-invariant AEGIS paired canary."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

from analysis import validate_aegis_failure_diagnostics as failure_validation
from main import aegis_failure_diagnostics as diagnostics


CANARY_EVIDENCE_SCHEMA = (
    "vlsa_table1_action_invariant_canary_evidence.v1"
)
REFERENCE_SHA256 = (
    "1a06b4842b356eb0fd6671b214aaea7d63cb2d9878982aa6817d305a6489fdf1"
)
DIAGNOSTIC_MODES = ("diagnostics-off", "diagnostics-on")
POLICY_MODES = ("pi05", "aegis")
EXPECTED_ARM_BY_POLICY_MODE = {
    "pi05": "pi05_translational",
    "aegis": "pi05_plus_aegis_translational",
}
HISTORICAL_OUTCOME_FIELDS = (
    "status",
    "terminal_reason",
    "task_success",
    "paper_collision",
    "paper_collision_avoidance",
    "collision_first_step",
    "executed_action_count",
    "legacy_ets_steps",
    "policy_query_count",
)
HISTORICAL_METRIC_FIELDS = (
    "paper_collision",
    "paper_collision_avoidance",
    "collision_first_step",
    "executed_action_count",
    "legacy_ets_steps",
)


class ActionInvariantCanaryError(RuntimeError):
    pass


def _canonical_sha256(value: Any) -> str:
    return diagnostics.sha256_bytes(
        diagnostics.canonical_json_bytes(value)
    )


def _semantic_result(result: Mapping[str, Any]) -> dict[str, Any]:
    """Remove only observer/timing/file-encoding fields from one result."""

    normalized = copy.deepcopy(dict(result))
    for key in (
        "action_invariance_ledger",
        "failure_diagnostics",
        "result_payload_sha256",
        "timing",
    ):
        normalized.pop(key, None)
    if "prior_attempt_artifacts" in normalized:
        raise ActionInvariantCanaryError(
            "fresh canary result retained prior-attempt artifacts"
        )
    actions = normalized.get("actions")
    if not isinstance(actions, list):
        raise ActionInvariantCanaryError("canary action ledger is missing")
    for action in actions:
        if not isinstance(action, dict):
            raise ActionInvariantCanaryError(
                "canary action ledger contains a non-object"
            )
        for key in (
            "env_step_input",
            "post_step_controller_proxy",
            "step_elapsed_seconds",
        ):
            action.pop(key, None)
        qp = action.get("qp")
        if isinstance(qp, dict):
            for key in ("context", "status", "z_before"):
                qp.pop(key, None)
    queries = normalized.get("policy_queries")
    if not isinstance(queries, list):
        raise ActionInvariantCanaryError(
            "canary policy-query ledger is missing"
        )
    for query in queries:
        if not isinstance(query, dict):
            raise ActionInvariantCanaryError(
                "canary policy-query ledger contains a non-object"
            )
        query.pop("elapsed_seconds", None)
        query.pop("server_timing", None)
    contact = normalized.get("contact_telemetry")
    if isinstance(contact, dict):
        contact.pop("detailed_artifact", None)
    method_failure = normalized.get("method_failure")
    if isinstance(method_failure, dict):
        method_failure.pop("diagnostics", None)
        method_failure.pop("diagnostics_payload_sha256", None)
    video = normalized.get("video")
    if isinstance(video, dict):
        video.pop("sha256", None)
    return normalized


def validate_historical_outcome(
    result: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    """Pin a current run to the validated canary's discrete outcome."""

    if set(expected) != set(HISTORICAL_OUTCOME_FIELDS):
        raise ActionInvariantCanaryError(
            "historical canary outcome reference fields changed"
        )
    for key in ("status", "terminal_reason", "task_success"):
        if key not in result:
            raise ActionInvariantCanaryError(
                f"canary result lacks historical outcome field {key!r}"
            )
    metrics = result.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ActionInvariantCanaryError(
            "canary result lacks historical outcome metrics"
        )
    missing_metrics = [
        key for key in HISTORICAL_METRIC_FIELDS if key not in metrics
    ]
    if missing_metrics:
        raise ActionInvariantCanaryError(
            "canary result lacks historical outcome metrics: "
            f"{missing_metrics}"
        )
    queries = result.get("policy_queries")
    if not isinstance(queries, list):
        raise ActionInvariantCanaryError(
            "canary result lacks its policy-query ledger"
        )
    observed = {
        "status": result["status"],
        "terminal_reason": result["terminal_reason"],
        "task_success": result["task_success"],
        **{
            key: metrics[key]
            for key in HISTORICAL_METRIC_FIELDS
        },
        "policy_query_count": len(queries),
    }
    mismatches = [
        key
        for key in HISTORICAL_OUTCOME_FIELDS
        if diagnostics.canonical_json_bytes(observed[key])
        != diagnostics.canonical_json_bytes(expected[key])
    ]
    if mismatches:
        raise ActionInvariantCanaryError(
            "historical canary outcome mismatch: "
            f"{sorted(mismatches)}"
        )
    return observed


def validate_observational_pair(
    diagnostics_off: Mapping[str, Any],
    diagnostics_on: Mapping[str, Any],
) -> dict[str, Any]:
    action_receipt = failure_validation.validate_action_invariance_pair(
        diagnostics_off,
        diagnostics_on,
    )
    off_semantic = _semantic_result(diagnostics_off)
    on_semantic = _semantic_result(diagnostics_on)
    off_sha256 = _canonical_sha256(off_semantic)
    on_sha256 = _canonical_sha256(on_semantic)
    if off_sha256 != on_sha256:
        differing = sorted(
            key
            for key in set(off_semantic) | set(on_semantic)
            if diagnostics.canonical_json_bytes(off_semantic.get(key))
            != diagnostics.canonical_json_bytes(on_semantic.get(key))
        )
        raise ActionInvariantCanaryError(
            f"diagnostics changed outcome/state semantics: {differing}"
        )
    return {
        "case_id": diagnostics_on.get("case_id"),
        "arm": diagnostics_on.get("arm"),
        "status": "observationally_invariant",
        "semantic_result_sha256": on_sha256,
        "action_invariance": action_receipt,
    }


def _load_terminal_source(
    result: Mapping[str, Any],
    *,
    output_root: Path,
) -> Any:
    import numpy as np

    record = result.get("failure_diagnostics")
    descriptor = (
        record.get("terminal_frame")
        if isinstance(record, Mapping)
        else None
    )
    if not isinstance(descriptor, Mapping):
        raise ActionInvariantCanaryError(
            "diagnostics-on result lacks its terminal-frame artifact"
        )
    relative = Path(str(descriptor.get("path")))
    if relative.is_absolute() or ".." in relative.parts:
        raise ActionInvariantCanaryError(
            "terminal-frame artifact path is unsafe"
        )
    path = output_root / relative
    if not path.is_file():
        raise ActionInvariantCanaryError(
            "terminal-frame artifact is missing"
        )
    return np.load(path, allow_pickle=False)


def _decode_off_video(
    diagnostics_off: Mapping[str, Any],
    diagnostics_on: Mapping[str, Any],
    *,
    off_output_root: Path,
    on_output_root: Path,
) -> dict[str, Any]:
    video = diagnostics_off.get("video")
    actions = diagnostics_off.get("actions")
    if not isinstance(video, Mapping) or not isinstance(actions, list):
        raise ActionInvariantCanaryError(
            "diagnostics-off video/action evidence is missing"
        )
    relative = Path(str(video.get("path")))
    if relative.is_absolute() or ".." in relative.parts:
        raise ActionInvariantCanaryError(
            "diagnostics-off video path is unsafe"
        )
    source = _load_terminal_source(
        diagnostics_on,
        output_root=on_output_root,
    )
    return failure_validation._probe_episode_video(
        off_output_root / relative,
        expected_frame_count=len(actions) + 1,
        terminal_source=source,
    )


def validate_four_run_canary(
    *,
    results: Mapping[tuple[str, str], Mapping[str, Any]],
    output_roots: Mapping[str, Path],
    reference_path: Path,
) -> dict[str, Any]:
    expected_keys = {
        (diagnostic_mode, policy_mode)
        for diagnostic_mode in DIAGNOSTIC_MODES
        for policy_mode in POLICY_MODES
    }
    if set(results) != expected_keys:
        raise ActionInvariantCanaryError(
            "four-run canary result inventory changed"
        )
    if set(output_roots) != set(DIAGNOSTIC_MODES):
        raise ActionInvariantCanaryError(
            "four-run canary output-root inventory changed"
        )
    if diagnostics.sha256_path(reference_path) != REFERENCE_SHA256:
        raise ActionInvariantCanaryError(
            "frozen canary action reference hash changed"
        )
    reference = failure_validation._load_json(reference_path)
    case_ids = {
        result.get("case_id") for result in results.values()
    }
    if case_ids != {reference.get("case_id")}:
        raise ActionInvariantCanaryError(
            "four-run canary/reference case identity changed"
        )

    reference_receipts: dict[str, Any] = {}
    outcome_receipts: dict[str, Any] = {}
    diagnostic_receipts: dict[str, Any] = {}
    invariance_receipts: dict[str, Any] = {}
    off_video_receipts: dict[str, Any] = {}
    for policy_mode in POLICY_MODES:
        off = results[("diagnostics-off", policy_mode)]
        on = results[("diagnostics-on", policy_mode)]
        expected_arm = EXPECTED_ARM_BY_POLICY_MODE[policy_mode]
        arm_reference = reference.get("arms", {}).get(expected_arm)
        expected_outcome = (
            arm_reference.get("expected_outcome")
            if isinstance(arm_reference, Mapping)
            else None
        )
        if not isinstance(expected_outcome, Mapping):
            raise ActionInvariantCanaryError(
                f"historical canary outcome is missing for {expected_arm!r}"
            )
        for diagnostic_mode, result in (
            ("diagnostics-off", off),
            ("diagnostics-on", on),
        ):
            if (
                result.get("mode") != policy_mode
                or result.get("arm") != expected_arm
            ):
                raise ActionInvariantCanaryError(
                    f"{diagnostic_mode}/{policy_mode} result mode/arm "
                    "mapping changed"
                )
            key = f"{diagnostic_mode}/{policy_mode}"
            outcome_receipts[key] = validate_historical_outcome(
                result,
                expected_outcome,
            )
        invariance_receipts[policy_mode] = validate_observational_pair(
            off,
            on,
        )
        for diagnostic_mode, result in (
            ("diagnostics-off", off),
            ("diagnostics-on", on),
        ):
            key = f"{diagnostic_mode}/{policy_mode}"
            reference_receipts[key] = (
                failure_validation.validate_canary_reference(
                    result,
                    reference,
                )
            )
        diagnostic_receipts[policy_mode] = (
            failure_validation.validate_diagnostic_result(
                on,
                output_root=output_roots["diagnostics-on"],
                require_ready_geometry=policy_mode == "aegis",
            )
        )
        off_video_receipts[policy_mode] = _decode_off_video(
            off,
            on,
            off_output_root=output_roots["diagnostics-off"],
            on_output_root=output_roots["diagnostics-on"],
        )
    return {
        "schema_version": CANARY_EVIDENCE_SCHEMA,
        "status": "validated",
        "scientific_result": False,
        "case_id": reference["case_id"],
        "reference": {
            "path": str(reference_path.resolve()),
            "sha256": REFERENCE_SHA256,
            "ledgers": reference_receipts,
            "historical_outcomes": outcome_receipts,
        },
        "action_and_outcome_invariance": invariance_receipts,
        "diagnostics_on": diagnostic_receipts,
        "diagnostics_off_video_decode": off_video_receipts,
    }
