"""Contracts for the frozen 354D L5 row-0/1 weight-decay ablation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


CONFIG_SCHEMA = "vlsa_distal_l5_row01_weight_decay_ablation.v1"
RESULT_SCHEMA = "vlsa_distal_l5_row01_weight_decay_ablation_result.v1"
VALIDATION_SCHEMA = "vlsa_distal_l5_row01_weight_decay_ablation_validation.v1"


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def payload_sha256(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("result_payload_sha256", None)
    payload.pop("validation_payload_sha256", None)
    return hashlib.sha256(canonical(payload)).hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict) or set(value) != {
        "schema_version", "protocol_id", "claim_scope", "source", "arms",
        "matched", "decision", "forbidden",
    }:
        raise ValueError("L5 row01 weight-decay config keys differ")
    arms = value["arms"]
    if (
        value["schema_version"] != CONFIG_SCHEMA
        or value["protocol_id"]
        != "vlsa-distal-l5-row01-weight-decay-ablation-v1"
        or set(arms) != {"no_weight_decay", "established_weight_decay"}
        or float(arms["no_weight_decay"]["weight_decay"]) != 0.0
        or float(arms["established_weight_decay"]["weight_decay"]) != 1.0e-4
        or value["matched"]["input_dimension"] != 354
        or value["matched"]["seed"] != 20260814
    ):
        raise ValueError("L5 row01 weight-decay protocol differs")
    output = json.loads(canonical(value).decode("utf-8"))
    output["config_file_sha256"] = hashlib.sha256(raw).hexdigest()
    output["config_payload_sha256"] = hashlib.sha256(canonical(value)).hexdigest()
    return output


def classify_effect(
    no_decay: Mapping[str, Any], established: Mapping[str, Any],
    *, minimum_relative_validation_improvement: float,
) -> tuple[str, dict[str, Any]]:
    no_rmse = float(no_decay["rmse_m"])
    established_rmse = float(established["rmse_m"])
    relative = (no_rmse - established_rmse) / max(no_rmse, 1.0e-12)
    deltas = {
        "validation_RMSE_established_minus_no_decay_m": established_rmse - no_rmse,
        "relative_validation_RMSE_improvement": relative,
        "false_safe_count_established_minus_no_decay": int(
            established["row01_false_safe_count"]
        ) - int(no_decay["row01_false_safe_count"]),
        "supported_state_count_established_minus_no_decay": int(
            established["supported_recoverable_state_count"]
        ) - int(no_decay["supported_recoverable_state_count"]),
    }
    materially_helpful = (
        relative >= float(minimum_relative_validation_improvement)
        and int(established["row01_false_safe_count"])
        < int(no_decay["row01_false_safe_count"])
        and int(established["supported_recoverable_state_count"])
        >= int(no_decay["supported_recoverable_state_count"])
    )
    return (
        "ordinary_weight_decay_materially_helps_but_prediction_gate_still_separate"
        if materially_helpful else
        "ordinary_weight_decay_does_not_explain_grouped_generalization_gap"
    ), deltas
