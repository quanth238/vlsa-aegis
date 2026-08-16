"""Contracts for the prospective grouped L5 Q-only diagnostic."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


CONFIG_SCHEMA = "vlsa_distal_prospective_l5_q_only_diagnostic.v1"
RESULT_SCHEMA = "vlsa_distal_prospective_l5_q_only_diagnostic_result.v1"
VALIDATION_SCHEMA = "vlsa_distal_prospective_l5_q_only_diagnostic_validation.v1"


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def payload_sha256(value: Mapping[str, Any], key: str) -> str:
    public = dict(value)
    public.pop(key, None)
    return hashlib.sha256(canonical(public)).hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    value = json.loads(raw)
    required = {
        "schema_version", "protocol_id", "claim_scope", "sources", "dataset",
        "features", "model", "metrics", "decision", "forbidden",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("prospective Q-only config keys differ")
    splits = value["dataset"]["split_case_ids"]
    flat = [case_id for split in ("train", "validation", "test") for case_id in splits[split]]
    if (
        value["schema_version"] != CONFIG_SCHEMA
        or value["protocol_id"] != "vlsa-distal-prospective-l5-q-only-diagnostic-v1"
        or [len(splits[name]) for name in ("train", "validation", "test")] != [6, 2, 2]
        or len(flat) != len(set(flat)) != 10
        or set(flat) != set(value["sources"]["case_artifacts"])
        or value["dataset"]["L5_row_indices"] != [0, 1, 2]
        or value["features"]["input_dimension"] != 9
        or value["features"]["translation_scale_m_per_action_unit"] != 0.05
        or value["model"]["input_dimension"] != 9
        or value["model"]["output_count"] != 3
        or value["model"]["hidden_widths"] != [32, 32]
        or value["model"]["checkpoint"] != "fixed_final_epoch_no_validation_model_selection"
        or "V_targets_or_loss" not in value["forbidden"]
        or "QP_or_gradient_correction" not in value["forbidden"]
    ):
        raise ValueError("prospective Q-only protocol differs")
    output = json.loads(canonical(value).decode("utf-8"))
    output["config_file_sha256"] = hashlib.sha256(raw).hexdigest()
    output["config_payload_sha256"] = hashlib.sha256(canonical(value)).hexdigest()
    return output


def classify_diagnostic(
    validation_metrics: Mapping[str, Any], test_metrics: Mapping[str, Any],
    decision: Mapping[str, Any],
) -> tuple[str, dict[str, bool]]:
    checks: dict[str, bool] = {}
    for split, metrics in (("validation", validation_metrics), ("test", test_metrics)):
        checks[f"{split}_zero_false_safes"] = (
            int(metrics["false_safe_count"]) <= int(decision["maximum_false_safe_count"])
        )
        recall = metrics["safe_recall"]
        checks[f"{split}_safe_recall"] = (
            recall is not None and float(recall) >= float(decision["minimum_safe_recall"])
        )
        checks[f"{split}_safe_support"] = (
            int(metrics["supported_state_count"])
            >= int(decision["required_safe_support_state_count"][split])
        )
        near = metrics["near_boundary_global_RMSE"]
        checks[f"{split}_near_boundary_RMSE"] = (
            near is not None
            and float(near) <= float(decision["maximum_near_boundary_global_RMSE"])
        )
    return (
        "diagnostic_transfer_signal_pass" if all(checks.values())
        else "diagnostic_transfer_signal_no_go",
        checks,
    )
