"""Contracts for the focused post-detour live-continuation gate."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


POST_DETOUR_LIVE_SCHEMA = "vlsa_distal_post_detour_live_gate_e05_config.v1"


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def load_post_detour_live_config(path: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    value = json.loads(raw)
    if value.get("protocol_id") != "vlsa-distal-post-detour-live-gate-e05-v1":
        raise ValueError("post-detour live protocol differs")
    if value.get("case_ids") != ["vlsa-t1-goal-ii-t0-e05"]:
        raise ValueError("post-detour live case differs")
    if value["controls"]["arms"] != [
        "registered_compound_continuation",
        "fresh_pi05_released_aegis",
        "fresh_pi05_released_aegis_plus_smooth_field_if_needed",
    ]:
        raise ValueError("post-detour live arms differ")
    state = value["state_protocol"]
    if state["compound_prefix_steps"] != list(range(182, 187)):
        raise ValueError("post-detour prefix differs")
    if state["continuation_steps"] != list(range(187, 192)):
        raise ValueError("post-detour continuation differs")
    if int(state["policy_query_index"]) != 37 or int(state["policy_execute_prefix"]) != 5:
        raise ValueError("post-detour policy query differs")
    action = value["action_space"]
    if action != {
        "action_limit": 1.0,
        "corrected_dimensions": [0, 1, 2],
        "corrected_horizon": 5,
        "line_search_fractions": [1.0, 0.5, 0.25],
        "maximum_iterations": 10,
        "maximum_total_correction_l2_action": 1.0,
        "maximum_total_path_length_action": 1.0,
        "trust_radius_action": 0.1,
    }:
        raise ValueError("post-detour live action space differs")
    estimation = value["field_estimation"]
    if estimation != {
        "direction_seed": 26081251,
        "heldout_direction_count_per_iteration": 8,
        "heldout_direction_seed": 26081252,
        "paired_direction_count_per_iteration": 32,
        "paired_perturbation_action": 0.05,
        "ridge": 1e-06,
        "temperature_m": 0.002,
    }:
        raise ValueError("post-detour live field estimation differs")
    gate = value["gate"]
    if not math.isclose(float(gate["internal_substep_clearance_buffer_m"]), 0.001):
        raise ValueError("post-detour clearance gate differs")
    if not math.isclose(float(gate["paper_car_threshold_m"]), 0.001):
        raise ValueError("post-detour CAR gate differs")
    if int(gate["protected_raw_contact_count"]) != 0:
        raise ValueError("post-detour contact gate differs")
    output = json.loads(_canonical(value).decode())
    output["schema_version"] = POST_DETOUR_LIVE_SCHEMA
    output["config_file_sha256"] = hashlib.sha256(raw).hexdigest()
    output["config_payload_sha256"] = hashlib.sha256(_canonical(value)).hexdigest()
    return output


def verified_internal(record: Mapping[str, Any], gate: Mapping[str, Any]) -> bool:
    return bool(
        float(record["minimum_clearance_m"])
        >= float(gate["internal_substep_clearance_buffer_m"])
        and len(record["protected_contacts"]) == int(gate["protected_raw_contact_count"])
        and float(record["maximum_active_obstacle_l1_displacement_m"])
        <= float(gate["paper_car_threshold_m"])
    )
