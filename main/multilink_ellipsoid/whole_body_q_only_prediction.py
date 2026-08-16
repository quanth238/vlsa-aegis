"""Frozen contracts and gates for the held-out whole-body Q-only test."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence


PROTOCOL_SCHEMA = "vlsa_distal_whole_body_q_only_prediction_protocol.v1"
BINDING_SCHEMA = "vlsa_distal_whole_body_q_only_prediction_binding.v1"
RESULT_SCHEMA = "vlsa_distal_whole_body_q_only_prediction_result.v1"
VALIDATION_SCHEMA = "vlsa_distal_whole_body_q_only_prediction_validation.v1"


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def payload_sha256(value: Mapping[str, Any], key: str) -> str:
    public = dict(value)
    public.pop(key, None)
    return hashlib.sha256(canonical(public)).hexdigest()


def file_sha256(path: Any) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_protocol(path: Any) -> dict[str, Any]:
    raw = path.read_bytes()
    value = json.loads(raw)
    expected = {
        "schema_version", "protocol_id", "claim_scope", "dataset",
        "features", "arms", "model", "metrics", "prediction_gate",
        "timeout_rule", "forbidden",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("whole-body Q-only prediction protocol keys differ")
    if (
        value["schema_version"] != PROTOCOL_SCHEMA
        or value["protocol_id"]
        != "vlsa-distal-whole-body-q-only-prediction-protocol-v1"
        or value["dataset"]["fit_splits"] != ["train"]
        or value["dataset"]["evaluation_splits"]
        != ["train", "validation", "test"]
        or value["dataset"]["test_access"]
        != "one_time_after_fixed_final_epoch_model"
        or value["model"]["hidden_widths"] != [32, 32]
        or int(value["model"]["seed"]) != 20260814
        or float(value["model"]["weight_decay"]) != 0.0001
        or value["model"]["checkpoint"]
        != "fixed_final_epoch_no_validation_or_test_selection"
        or value["timeout_rule"] != "UNKNOWN_censored_from_fit_and_metrics"
        or value["prediction_gate"]["heldout_splits"]
        != ["validation", "test"]
        or int(value["prediction_gate"]["maximum_selected_false_safe_count"])
        != 0
        or float(value["prediction_gate"][
            "minimum_recoverable_state_safe_support_fraction"
        ]) != 1.0
    ):
        raise ValueError("whole-body Q-only prediction protocol differs")
    output = json.loads(canonical(value).decode("utf-8"))
    output["config_file_sha256"] = hashlib.sha256(raw).hexdigest()
    output["config_payload_sha256"] = hashlib.sha256(canonical(value)).hexdigest()
    return output


def load_binding(path: Any, protocol: Mapping[str, Any]) -> dict[str, Any]:
    raw = path.read_bytes()
    value = json.loads(raw)
    expected = {
        "schema_version", "protocol_id", "claim_scope", "protocol",
        "coverage_audit", "sources", "test_opened_once", "forbidden",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("whole-body Q-only prediction binding keys differ")
    if (
        value["schema_version"] != BINDING_SCHEMA
        or value["protocol_id"]
        != "vlsa-distal-whole-body-q-only-prediction-binding-v1"
        or value["protocol"] != {
            "path": "configs/vlsa_distal_whole_body_q_only_prediction_protocol.v1.json",
            "file_sha256": protocol["config_file_sha256"],
            "payload_sha256": protocol["config_payload_sha256"],
        }
        or not isinstance(value["sources"], list)
        or len(value["sources"]) < 3
        or value["test_opened_once"] is not True
    ):
        raise ValueError("whole-body Q-only prediction binding differs")
    output = json.loads(canonical(value).decode("utf-8"))
    output["config_file_sha256"] = hashlib.sha256(raw).hexdigest()
    output["config_payload_sha256"] = hashlib.sha256(canonical(value)).hexdigest()
    return output


def combined_training_config(
    protocol: Mapping[str, Any], binding: Mapping[str, Any],
) -> dict[str, Any]:
    output = {
        key: protocol[key]
        for key in ("claim_scope", "dataset", "features", "arms", "model", "metrics")
    }
    output["sources"] = binding["sources"]
    return output


def _split_gate(
    metrics: Mapping[str, Any], *, maximum_boundary_rmse: float,
    minimum_rank_spearman: float,
) -> dict[str, Any]:
    global_metrics = metrics["global"]
    states = global_metrics["states"]
    recoverable = [row for row in states if row["actual_safe_candidate_count"] > 0]
    selected_false_safe = sum(
        row["selected_actual_safe"] is False for row in recoverable
    )
    supported = sum(row["predicted_safe_support"] for row in recoverable)
    support_fraction = 1.0 if not recoverable else supported / len(recoverable)
    boundary_rmse = global_metrics["near_boundary_RMSE"]
    rank = global_metrics["rank_spearman"]
    return {
        "recoverable_state_count": len(recoverable),
        "supported_recoverable_state_count": supported,
        "recoverable_state_safe_support_fraction": support_fraction,
        "selected_false_safe_count": selected_false_safe,
        "near_boundary_RMSE": boundary_rmse,
        "rank_spearman": rank,
        "boundary_error_pass": (
            boundary_rmse is not None
            and float(boundary_rmse) <= float(maximum_boundary_rmse)
        ),
        "ordering_pass": (
            rank is not None and float(rank) >= float(minimum_rank_spearman)
        ),
    }


def evaluate_prediction_gate(
    split_metrics: Mapping[str, Mapping[str, Any]],
    protocol: Mapping[str, Any], *, source_replay_exact: bool,
) -> dict[str, Any]:
    gate = protocol["prediction_gate"]
    split_rows = {
        split: _split_gate(
            split_metrics[split],
            maximum_boundary_rmse=float(gate["maximum_near_boundary_RMSE"]),
            minimum_rank_spearman=float(gate["minimum_rank_spearman"]),
        )
        for split in gate["heldout_splits"]
    }
    total_recoverable = sum(
        row["recoverable_state_count"] for row in split_rows.values()
    )
    total_supported = sum(
        row["supported_recoverable_state_count"] for row in split_rows.values()
    )
    total_false_safe = sum(
        row["selected_false_safe_count"] for row in split_rows.values()
    )
    support_fraction = (
        1.0 if total_recoverable == 0 else total_supported / total_recoverable
    )
    passes = bool(
        source_replay_exact
        and total_recoverable > 0
        and total_false_safe
        <= int(gate["maximum_selected_false_safe_count"])
        and support_fraction
        >= float(gate["minimum_recoverable_state_safe_support_fraction"])
        and all(row["boundary_error_pass"] for row in split_rows.values())
        and all(row["ordering_pass"] for row in split_rows.values())
    )
    return {
        "heldout_splits": split_rows,
        "source_replay_exact": bool(source_replay_exact),
        "recoverable_state_count": total_recoverable,
        "supported_recoverable_state_count": total_supported,
        "recoverable_state_safe_support_fraction": support_fraction,
        "selected_false_safe_count": total_false_safe,
        "passes": passes,
    }
