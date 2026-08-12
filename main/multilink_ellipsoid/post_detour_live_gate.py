"""Contracts for the focused post-detour live-continuation gate."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


POST_DETOUR_LIVE_SCHEMA = "vlsa_distal_post_detour_live_gate_e05_config.v1"
POST_DETOUR_ROUTE_SCHEMA = "vlsa_distal_post_detour_route_oracle_e05_config.v1"
RECEDING_ROUTE_SCHEMA = "vlsa_distal_receding_route_oracle_e05_config.v1"


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


def persistent_route_directions(robot_center: Any, obstacle_center: Any) -> dict[str, Any]:
    """Return deterministic obstacle-normal and tangent route directions."""
    import numpy as np

    normal = np.asarray(robot_center, dtype=np.float64) - np.asarray(obstacle_center, dtype=np.float64)
    normal /= np.linalg.norm(normal)
    world_up = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    up = world_up - normal * float(np.dot(normal, world_up))
    if float(np.linalg.norm(up)) < 1.0e-8:
        fallback = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
        up = fallback - normal * float(np.dot(normal, fallback))
    up /= np.linalg.norm(up)
    side = np.cross(normal, up)
    side /= np.linalg.norm(side)
    return {"retreat": normal, "up": up, "left": side, "right": -side}


def persistent_route_correction(direction: Any, norm: float) -> Any:
    """Apply one spatial route direction persistently over five actions."""
    import numpy as np

    value = np.tile(np.asarray(direction, dtype=np.float64), 5)
    return float(norm) * value / np.linalg.norm(value)


def load_post_detour_route_config(path: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    value = json.loads(raw)
    if value.get("protocol_id") != "vlsa-distal-post-detour-route-oracle-e05-v1":
        raise ValueError("post-detour route protocol differs")
    if value.get("case_ids") != ["vlsa-t1-goal-ii-t0-e05"]:
        raise ValueError("post-detour route case differs")
    if value["route_search"]["modes"] != ["left", "right", "up", "retreat"]:
        raise ValueError("post-detour route modes differ")
    if value["route_search"]["coarse_norms_action"] != [0.5, 1.0, 1.5, 2.0]:
        raise ValueError("post-detour route scale grid differs")
    if value["route_search"]["basis"] != "constant_five_action_obstacle_normal_or_tangent":
        raise ValueError("post-detour route basis differs")
    if value["derivative_free"] != {
        "candidate_count_per_generation": 64,
        "elite_count": 8,
        "generations": 4,
        "maximum_correction_l2_action": 2.0,
        "seed": 26081261,
    }:
        raise ValueError("post-detour derivative-free control differs")
    state = value["state_protocol"]
    if state["compound_prefix_steps"] != list(range(182, 187)):
        raise ValueError("post-detour route prefix differs")
    if state["continuation_steps"] != list(range(187, 192)):
        raise ValueError("post-detour route continuation differs")
    gate = value["gate"]
    if not math.isclose(float(gate["internal_substep_clearance_buffer_m"]), 0.001):
        raise ValueError("post-detour route clearance gate differs")
    if not math.isclose(float(gate["paper_car_threshold_m"]), 0.001):
        raise ValueError("post-detour route CAR gate differs")
    if int(gate["protected_raw_contact_count"]) != 0:
        raise ValueError("post-detour route contact gate differs")
    output = json.loads(_canonical(value).decode())
    output["schema_version"] = POST_DETOUR_ROUTE_SCHEMA
    output["config_file_sha256"] = hashlib.sha256(raw).hexdigest()
    output["config_payload_sha256"] = hashlib.sha256(_canonical(value)).hexdigest()
    return output


def load_receding_route_config(path: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    value = json.loads(raw)
    if value.get("protocol_id") != "vlsa-distal-receding-route-oracle-e05-v1":
        raise ValueError("receding route protocol differs")
    if value.get("case_ids") != ["vlsa-t1-goal-ii-t0-e05"]:
        raise ValueError("receding route case differs")
    state = value["state_protocol"]
    expected_state = {
        "activation_step": 182,
        "initial_verified_prefix_steps": [],
        "live_receding_start_step": 183,
        "lookahead_actions": 5,
        "execute_prefix_actions": 1,
        "first_live_policy_query_index": 37,
        "replan_after_every_executed_action": True,
    }
    if state != expected_state:
        raise ValueError("receding route state protocol differs")
    route = value["route_search"]
    if route["modes"] != ["left", "right", "up", "retreat"]:
        raise ValueError("receding route modes differ")
    if route["coarse_norms_action"] != [0.5, 1.0, 1.5, 2.0]:
        raise ValueError("receding route scale grid differs")
    if route["selection"] != "smallest_verified_correction_then_persistent_mode_then_fixed_mode_order":
        raise ValueError("receding route selection differs")
    if not math.isclose(float(route["release_clearance_m"]), 0.005):
        raise ValueError("receding route release threshold differs")
    gate = value["gate"]
    if not math.isclose(float(gate["internal_substep_clearance_buffer_m"]), 0.001):
        raise ValueError("receding route clearance gate differs")
    if not math.isclose(float(gate["paper_car_threshold_m"]), 0.001):
        raise ValueError("receding route CAR gate differs")
    if int(gate["protected_raw_contact_count"]) != 0:
        raise ValueError("receding route contact gate differs")
    output = json.loads(_canonical(value).decode())
    output["schema_version"] = RECEDING_ROUTE_SCHEMA
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
