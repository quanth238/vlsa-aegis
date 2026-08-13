"""Contracts for grouped candidate-plus-complete-backup risk collection."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


CONFIG_SCHEMA = "vlsa_distal_grouped_query_action_risk.v1"
RESULT_SCHEMA = "vlsa_distal_grouped_query_action_risk_result.v1"
VALIDATION_SCHEMA = "vlsa_distal_grouped_query_action_risk_validation.v1"


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def load_config(path: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    value = json.loads(raw)
    required = {
        "schema_version", "protocol_id", "claim_scope", "coverage_source",
        "method", "population_gate", "forbidden",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("grouped query-risk config keys differ")
    if value["schema_version"] != CONFIG_SCHEMA:
        raise ValueError("grouped query-risk schema differs")
    if value["protocol_id"] != "vlsa-distal-grouped-query-action-risk-v1":
        raise ValueError("grouped query-risk protocol differs")
    if value["coverage_source"] != {
        "producer_root": (
            "/mnt/data/quanth/experiments/vlsa-distal-query-boundary-coverage/"
            "query-coverage-population-20260814a"
        ),
        "producer_commit": "4ac880c109540235bab43487e4139613821fe047",
        "validation_result_file_sha256": (
            "56ad37586f33f426e6fe920f6a5c5e67527acee81bb59f1d3140cf6d313dbf0c"
        ),
        "validation_payload_sha256": (
            "2c47f1bb17c5de676197ed58c320521c6f5a1d3cf0042f565dbf1ba0a06e14d7"
        ),
        "retained_state_count": 15,
        "one_state_per_episode": True,
        "included_splits": ["diagnostic", "train", "validation"],
        "excluded_splits": ["test"],
    }:
        raise ValueError("grouped query-risk coverage source differs")
    if value["method"] != {
        "base_config": "configs/vlsa_distal_query_action_risk_e05.v1.json",
        "base_config_file_sha256": (
            "fe29ac8353f34c4fe723d8f4e455fa2d0a333eba1f21ca154126b6e2a112f98c"
        ),
        "base_config_payload_sha256": (
            "a4a4b8d26268d62df085231b68b6ce37f5f3be6c37b975bfc21f04e4bc8f2cde"
        ),
        "candidate_family_unchanged": True,
        "backup_policy_unchanged": True,
        "risk_target_unchanged": True,
        "terminal_semantics_unchanged": True,
        "timeout_is_unknown_never_safe": True,
        "candidate_count_per_state": 37,
    }:
        raise ValueError("grouped query-risk method differs")
    if value["population_gate"] != {
        "retain_failed_states": True,
        "require_safe_and_unsafe_per_state": True,
        "require_no_proxy_safe_physical_collision": True,
        "near_boundary_absolute_risk_m": 0.005,
        "report_per_row": [
            "safe", "unsafe", "near_boundary", "active_witness",
            "state", "episode",
        ],
        "training_requires_all_claimed_rows_supported": True,
        "unsupported_rows_are_reported_not_filled": True,
    }:
        raise ValueError("grouped query-risk gate differs")
    if value["forbidden"] != {
        "model_training": True,
        "calibration": True,
        "QP": True,
        "closed_loop": True,
        "CBF_claim": True,
    }:
        raise ValueError("grouped query-risk forbidden set differs")
    output = json.loads(canonical(value).decode("utf-8"))
    output["config_file_sha256"] = hashlib.sha256(raw).hexdigest()
    output["config_payload_sha256"] = hashlib.sha256(canonical(value)).hexdigest()
    return output
