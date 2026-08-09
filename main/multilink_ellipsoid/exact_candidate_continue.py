"""Contracts for the unsafe suffix-continuation E05 diagnostic."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .exact_candidate_closed_loop import CONSTRAINT_ORDER


CONTINUE_CONFIG_SCHEMA = "vlsa_distal_exact_candidate_continue_e05.v1"
CONTINUE_RESULT_SCHEMA = "vlsa_distal_exact_candidate_continue_e05_result.v1"
CONTINUE_VALIDATION_SCHEMA = (
    "vlsa_distal_exact_candidate_continue_e05_validation.v1"
)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def load_continue_config(path: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    try:
        config = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("continue config is invalid JSON") from error
    required = {
        "schema_version", "protocol_id", "claim_scope", "primary_case",
        "prerequisite", "prefix", "continuation", "measurement",
        "decision_gate",
    }
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError("continue config keys differ")
    if (
        config["schema_version"] != CONTINUE_CONFIG_SCHEMA
        or config["protocol_id"]
        != "vlsa-distal-exact-candidate-continue-e05-v1"
    ):
        raise ValueError("continue protocol differs")
    if config["primary_case"] != {
        "case_id": "vlsa-t1-goal-ii-t0-e05",
        "expected_action_count": 237,
        "accepted_safe_prefix_action_count": 187,
        "unsafe_continuation_start_step": 187,
    }:
        raise ValueError("continue case differs")
    prerequisite = config["prerequisite"]
    if set(prerequisite) != {
        "exact_candidate_result_file_sha256",
        "exact_candidate_result_payload_sha256",
        "exact_candidate_validation_file_sha256",
        "require_validated_empty_safe_set_at_step",
    } or prerequisite["require_validated_empty_safe_set_at_step"] != 187:
        raise ValueError("continue prerequisite differs")
    for key in (
        "exact_candidate_result_file_sha256",
        "exact_candidate_result_payload_sha256",
        "exact_candidate_validation_file_sha256",
    ):
        if not isinstance(prerequisite[key], str) or len(prerequisite[key]) != 64:
            raise ValueError("continue prerequisite hash differs")
    if config["prefix"] != {
        "source": "validated_exact_candidate_executed_action_ledger",
        "require_every_dynamic_state_hash_match": True,
        "require_every_obstacle_displacement_match": True,
        "require_zero_prefix_protected_contact": True,
        "require_zero_prefix_paper_CAR": True,
    }:
        raise ValueError("continue prefix differs")
    if config["continuation"] != {
        "source": "immutable_successful_released_AEGIS_Table1_env_step_sequence",
        "execute_steps": "187_through_native_done_or_action_236",
        "safety_filter": "disabled_after_empty_safe_set",
        "candidate_search": False,
        "affine_QP_used": False,
        "stop_on_proxy_violation": False,
        "stop_on_raw_contact": False,
        "stop_on_paper_CAR": False,
        "orientation_gripper_channels": "unchanged",
    }:
        raise ValueError("continue behavior differs")
    if config["measurement"] != {
        "osc_internal_substeps": "all",
        "constraint_order": CONSTRAINT_ORDER,
        "paper_CAR_threshold_m": 1.0e-3,
    }:
        raise ValueError("continue measurement differs")
    if config["decision_gate"] != {
        "report": (
            "native_task_success_and_first_proxy_violation_contact_and_CAR_"
            "steps_under_unsafe_continuation"
        ),
        "safety_success_allowed": False,
        "neural_training_authorized": False,
        "interpretation": (
            "diagnostic_of_what_happens_if_fail_closed_stop_is_removed"
        ),
    }:
        raise ValueError("continue decision differs")
    output = json.loads(_canonical(config).decode("utf-8"))
    output["config_file_sha256"] = hashlib.sha256(raw).hexdigest()
    output["config_payload_sha256"] = hashlib.sha256(
        _canonical(config)
    ).hexdigest()
    return output
