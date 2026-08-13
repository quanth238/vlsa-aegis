"""Contracts for the immutable grouped action-risk coverage freeze."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


CONFIG_SCHEMA = "vlsa_distal_grouped_query_action_risk_freeze.v1"
VALIDATION_SCHEMA = "vlsa_distal_grouped_query_action_risk_validation.v2"


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def load_config(path: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    value = json.loads(raw)
    required = {
        "schema_version", "protocol_id", "source", "classification",
        "adequacy", "split", "freeze", "forbidden_until_prediction_gate",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("grouped query-risk freeze config keys differ")
    if value["schema_version"] != CONFIG_SCHEMA:
        raise ValueError("grouped query-risk freeze schema differs")
    if value["protocol_id"] != "vlsa-distal-grouped-query-action-risk-freeze-v1":
        raise ValueError("grouped query-risk freeze protocol differs")
    if value["classification"] != {
        "primary_state_precedence": [
            "proxy_invalid", "no_safe_candidate",
            "no_known_unsafe_candidate", "usable_mixed_support",
        ],
        "timeout_candidates_are_unknown_and_excluded_from_learning": True,
        "proxy_invalid_if_nonpositive_proxy_risk_has_physical_veto": True,
        "usable_requires_known_safe_and_known_unsafe_candidates": True,
    }:
        raise ValueError("grouped query-risk classification differs")
    if value["adequacy"]["claimed_rows"] != list(range(7)):
        raise ValueError("grouped query-risk claimed rows differ")
    per_row = value["adequacy"]["per_row"]
    if per_row != {
        "minimum_known_safe_candidates": 20,
        "minimum_known_unsafe_candidates": 20,
        "minimum_near_boundary_known_candidates": 20,
        "minimum_active_witness_candidates": 20,
        "minimum_train_episodes_with_unsafe": 3,
        "minimum_validation_episodes_with_unsafe": 1,
        "minimum_train_episodes_with_active_witness": 3,
        "minimum_validation_episodes_with_active_witness": 1,
    }:
        raise ValueError("grouped query-risk per-row adequacy differs")
    if value["split"]["unit"] != "complete_episode":
        raise ValueError("grouped query-risk split unit differs")
    if not value["freeze"]["unknown_candidates_never_enter_candidate_manifest"]:
        raise ValueError("grouped query-risk timeout freeze differs")
    if not value["freeze"]["coverage_snapshot_written_regardless_of_adequacy"]:
        raise ValueError("grouped query-risk coverage snapshot differs")
    if not value["freeze"]["training_authorized_only_when_all_adequacy_gates_pass"]:
        raise ValueError("grouped query-risk training authorization differs")
    output = json.loads(canonical(value).decode("utf-8"))
    output["config_file_sha256"] = hashlib.sha256(raw).hexdigest()
    output["config_payload_sha256"] = hashlib.sha256(canonical(value)).hexdigest()
    return output


def candidate_outcome(candidate: dict[str, Any]) -> str:
    """Return one disjoint learning outcome for a completed candidate record."""
    status = str(candidate["terminal_status"])
    if status == "UNKNOWN_TIMEOUT":
        if bool(candidate["exact_safe"]):
            raise ValueError("timeout candidate was labeled safe")
        return "unknown"
    if status not in {"SAFE_TERMINAL", "UNSAFE_CONTACT_OR_CAR"}:
        raise ValueError("candidate terminal status differs")
    risk = [float(item) for item in candidate["combined_risk"]]
    if len(risk) != 7:
        raise ValueError("candidate risk width differs")
    safe = bool(
        status == "SAFE_TERMINAL"
        and max(risk) <= 0.0
        and not bool(candidate["physical_veto"])
    )
    if bool(candidate["exact_safe"]) != safe:
        raise ValueError("candidate exact-safe label differs")
    return "safe" if safe else "unsafe"


def state_classification(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Classify a state without silently dropping unknown or proxy-invalid data."""
    outcomes = [candidate_outcome(item) for item in candidates]
    proxy_invalid_count = sum(
        max(float(value) for value in item["combined_risk"]) <= 0.0
        and bool(item["physical_veto"])
        for item in candidates
    )
    counts = {name: outcomes.count(name) for name in ("safe", "unsafe", "unknown")}
    if proxy_invalid_count:
        category = "proxy_invalid"
    elif counts["safe"] == 0:
        category = "no_safe_candidate"
    elif counts["unsafe"] == 0:
        category = "no_known_unsafe_candidate"
    else:
        category = "usable_mixed_support"
    return {
        "primary_category": category,
        "known_safe_candidate_count": counts["safe"],
        "known_unsafe_candidate_count": counts["unsafe"],
        "unknown_timeout_candidate_count": counts["unknown"],
        "contains_unknown_timeouts": counts["unknown"] > 0,
        "proxy_safe_physical_violation_count": proxy_invalid_count,
    }
