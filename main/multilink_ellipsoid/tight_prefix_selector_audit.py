"""Contracts for offline selection from frozen tight-prefix predictions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


CONFIG_SCHEMA = "vlsa_tight_prefix_selector_audit.v1"
RESULT_SCHEMA = "vlsa_tight_prefix_selector_audit_result.v1"
VALIDATION_SCHEMA = "vlsa_tight_prefix_selector_audit_validation.v1"
RULES = (
    "validation_margin_least_intervention",
    "zero_threshold_least_intervention",
    "minimum_predicted_primary_risk",
)


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def payload_sha256(value: Mapping[str, Any], key: str) -> str:
    public = dict(value)
    public.pop(key, None)
    return hashlib.sha256(canonical(public)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    value = json.loads(raw)
    expected = {
        "schema_version", "protocol_id", "claim_scope", "source",
        "selection", "evaluation", "forbidden",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("tight prefix selector config keys differ")
    if (
        value["schema_version"] != CONFIG_SCHEMA
        or value["protocol_id"] != "vlsa-tight-prefix-selector-audit-v1"
    ):
        raise ValueError("tight prefix selector protocol differs")
    selection = value["selection"]
    if (
        selection.get("rules") != list(RULES)
        or selection.get("rule_freeze_split") != "validation"
        or selection.get("test_open_after_rule_freeze") is not True
        or selection.get("primary_rows") != list(range(8))
        or selection.get("diagnostic_rows") != [8, 9]
    ):
        raise ValueError("tight prefix selector rules differ")
    gate = value["evaluation"]
    if gate != {
        "validation_required_safe_selected_states": 4,
        "validation_maximum_unsafe_selections": 0,
        "validation_maximum_abstentions": 0,
        "diagnostic_test_required_safe_selected_states": 6,
        "diagnostic_test_maximum_unsafe_selections": 0,
        "diagnostic_test_maximum_abstentions": 0,
        "require_zero_selected_L6_failures": True,
        "test_status": "already_opened_diagnostic",
    }:
        raise ValueError("tight prefix selector gate differs")
    output = json.loads(canonical(value).decode("utf-8"))
    output["config_file_sha256"] = hashlib.sha256(raw).hexdigest()
    output["config_payload_sha256"] = hashlib.sha256(canonical(value)).hexdigest()
    return output


def scientific_view(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in result.items()
        if key not in {"allocation", "result_payload_sha256", "source"}
    }
