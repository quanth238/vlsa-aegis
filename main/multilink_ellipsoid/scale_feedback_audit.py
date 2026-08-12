"""Contracts for the E05 compound-scale live-feedback audit."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


SCALE_FEEDBACK_AUDIT_SCHEMA = "vlsa_distal_scale_feedback_audit_e05.v1"


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def load_scale_feedback_audit_config(path: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    value = json.loads(raw)
    required = {
        "case_ids",
        "claim_scope",
        "gate",
        "internal_verification",
        "protocol_id",
        "registered_inputs",
        "schema_version",
        "state_protocol",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("scale-feedback audit config keys differ")
    if value["schema_version"] != SCALE_FEEDBACK_AUDIT_SCHEMA:
        raise ValueError("scale-feedback audit schema differs")
    if value["case_ids"] != ["vlsa-t1-goal-ii-t0-e05"]:
        raise ValueError("scale-feedback audit case differs")
    state = value["state_protocol"]
    expected_state = {
        "live_query_action_horizon": 5,
        "live_query_first_index": 41,
        "live_query_start_step": 206,
        "live_requery_interval_actions": 5,
        "policy_noise": "same_query_indices_and_seeds_for_both_scale_arms",
        "prelude_last_step": 205,
        "prelude_source": "immutable_registered_compound_scale_result",
        "released_aegis": "original_end_effector_proxy_only",
    }
    if state != expected_state:
        raise ValueError("scale-feedback audit state protocol differs")
    gate = value["gate"]
    expected_gate = {
        "internal_substep_clearance_buffer_m",
        "paper_car_threshold_m",
        "protected_raw_contact_count",
        "require_native_task_success",
    }
    if not isinstance(gate, Mapping) or set(gate) != expected_gate:
        raise ValueError("scale-feedback audit gate differs")
    for key in expected_gate - {"protected_raw_contact_count", "require_native_task_success"}:
        item = gate[key]
        if isinstance(item, bool) or not math.isfinite(float(item)) or float(item) < 0:
            raise ValueError("scale-feedback audit gate value differs: %s" % key)
    if int(gate["protected_raw_contact_count"]) != 0:
        raise ValueError("scale-feedback audit contact gate differs")
    registered = value["registered_inputs"]
    if not isinstance(registered, Mapping) or set(registered) != {"scale_1p01", "scale_1p02"}:
        raise ValueError("scale-feedback audit registered inputs differ")
    for label, expected_scale in (("scale_1p01", 1.01), ("scale_1p02", 1.02)):
        item = registered[label]
        if not isinstance(item, Mapping) or set(item) != {
            "correction_scale",
            "result_payload_sha256",
            "run_id",
            "source_commit",
        }:
            raise ValueError("scale-feedback audit input differs: %s" % label)
        if float(item["correction_scale"]) != expected_scale:
            raise ValueError("scale-feedback audit scale differs: %s" % label)
    output = json.loads(_canonical(value).decode("utf-8"))
    output["config_file_sha256"] = hashlib.sha256(raw).hexdigest()
    output["config_payload_sha256"] = hashlib.sha256(_canonical(value)).hexdigest()
    return output

