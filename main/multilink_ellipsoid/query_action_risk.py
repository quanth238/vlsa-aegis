"""Contracts for the query-aligned five-action Monte Carlo risk gate.

This module is intentionally independent of MuJoCo.  It freezes the candidate
family and terminal semantics before the allocation-backed diagnostic runs.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


CONFIG_SCHEMA = "vlsa_distal_query_action_risk_e05.v1"
RESULT_SCHEMA = "vlsa_distal_query_action_risk_e05_result.v1"


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def load_config(path: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    value = json.loads(raw)
    required = {
        "schema_version", "protocol_id", "case_id", "claim_scope",
        "state", "candidate_family", "backup_policy", "risk_target",
        "diagnostic_gate", "forbidden",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("query action-risk config keys differ")
    if value["schema_version"] != CONFIG_SCHEMA:
        raise ValueError("query action-risk schema differs")
    if value["protocol_id"] != "vlsa-distal-query-action-risk-e05-v1":
        raise ValueError("query action-risk protocol differs")
    if value["case_id"] != "vlsa-t1-goal-ii-t0-e05":
        raise ValueError("query action-risk case differs")
    if value["state"] != {
        "query_boundary_step": 185,
        "query_index": 37,
        "candidate_actions": 5,
        "action_coordinates": "released_AEGIS_output_before_OSC",
        "minimum_initial_proxy_clearance_m": 0.001,
        "maximum_initial_active_obstacle_l1_displacement_m": 0.001,
        "expected_mujoco_substeps_per_action": 25,
        "boundary_equivalence_tolerance_m": 1.0e-12,
    }:
        raise ValueError("query action-risk state protocol differs")
    if value["candidate_family"] != {
        "directions": ["normal", "tangent_up", "tangent_side"],
        "signed_directions": True,
        "temporal_profiles": ["constant", "front_loaded"],
        "correction_l2_action": [0.5, 1.0, 2.0],
        "include_nominal": True,
        "candidate_count": 37,
        "action_limit": 1.0,
        "preserve_rotation_and_gripper": True,
        "endpoint_preservation": False,
    }:
        raise ValueError("query action-risk candidate family differs")
    if value["backup_policy"] != {
        "policy": "receding_verified_registered_geometry_backup",
        "directions": "hold_plus_world_axes_plus_local_normal_tangents",
        "amplitudes_action": [0.5, 1.0],
        "candidate_count": 25,
        "selection_lookahead_hold_actions": 5,
        "execute_selected_actions": 1,
        "maximum_executed_actions": 10,
        "terminal_hold_actions": 10,
        "terminal_release_clearance_m": 0.005,
        "backup_rotation_and_gripper_action": 0.0,
        "vla_ledger_forbidden": True,
        "tie_break": "maximum_future_clearance_then_registered_order",
        "fallback": "maximum_future_clearance_then_registered_order",
    }:
        raise ValueError("query action-risk backup differs")
    if value["risk_target"] != {
        "output_count": 7,
        "positive_is_unsafe": True,
        "safety_buffer_m": 0.001,
        "paper_car_threshold_m": 0.001,
        "initial_state_is_eligibility_only": True,
        "candidate_prefix_excludes_fixed_k0": True,
        "definition": "max_candidate_prefix_and_complete_backup_violation_per_row",
        "physical_veto": "protected_MuJoCo_contact_or_paper_CAR",
        "terminal_statuses": [
            "SAFE_TERMINAL", "UNSAFE_CONTACT_OR_CAR", "UNKNOWN_TIMEOUT"
        ],
        "timeout_is_never_safe": True,
    }:
        raise ValueError("query action-risk target differs")
    output = json.loads(canonical(value).decode("utf-8"))
    output["config_file_sha256"] = hashlib.sha256(raw).hexdigest()
    output["config_payload_sha256"] = hashlib.sha256(canonical(value)).hexdigest()
    return output


def temporal_profile(name: str, action_count: int = 5) -> list[float]:
    """Return a unit-L2 temporal basis so radius equals correction norm."""

    if int(action_count) != 5:
        raise ValueError("query action-risk profile horizon differs")
    if name == "constant":
        values = [1.0] * 5
    elif name == "front_loaded":
        values = [5.0, 4.0, 3.0, 2.0, 1.0]
    else:
        raise ValueError("query action-risk profile differs")
    norm = math.sqrt(sum(item * item for item in values))
    return [item / norm for item in values]


def candidate_definitions(
    nominal_actions: Sequence[Sequence[float]],
    local_frame: Mapping[str, Sequence[float]],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Construct the frozen paired five-action translation candidate bank."""

    import numpy as np

    nominal = np.asarray(nominal_actions, dtype=np.float64)
    if nominal.shape != (5, 7) or not np.all(np.isfinite(nominal)):
        raise ValueError("query action-risk nominal chunk differs")
    family = config["candidate_family"]
    output = [{
        "name": "nominal",
        "order": 0,
        "direction": None,
        "sign": 0,
        "temporal_profile": None,
        "requested_correction_l2_action": 0.0,
        "applied_correction_l2_action": 0.0,
        "clipped": False,
        "actions": nominal.tolist(),
    }]
    for direction_name in family["directions"]:
        direction = np.asarray(local_frame[direction_name], dtype=np.float64)
        if direction.shape != (3,) or not np.all(np.isfinite(direction)):
            raise ValueError("query action-risk local direction differs")
        direction = direction / np.linalg.norm(direction)
        # Away/positive first makes the bounded apparatus canary exercise the
        # backup after a plausible recovery candidate without changing the
        # paired full-bank population.
        for sign in (1, -1):
            signed = direction * float(sign)
            for profile_name in family["temporal_profiles"]:
                profile = np.asarray(temporal_profile(profile_name), dtype=np.float64)
                for radius in family["correction_l2_action"]:
                    requested = profile[:, None] * signed[None, :] * float(radius)
                    actions = nominal.copy()
                    actions[:, :3] = np.clip(
                        actions[:, :3] + requested,
                        -float(family["action_limit"]),
                        float(family["action_limit"]),
                    )
                    applied = actions[:, :3] - nominal[:, :3]
                    output.append({
                        "name": "%s_%s_%s_r%s" % (
                            direction_name,
                            "pos" if sign > 0 else "neg",
                            profile_name,
                            radius,
                        ),
                        "order": len(output),
                        "direction": direction.tolist(),
                        "sign": int(sign),
                        "temporal_profile": profile_name,
                        "requested_correction_l2_action": float(radius),
                        "applied_correction_l2_action": float(np.linalg.norm(applied)),
                        "clipped": bool(np.max(np.abs(applied - requested)) > 1.0e-12),
                        "actions": actions.tolist(),
                    })
    if len(output) != int(family["candidate_count"]):
        raise ValueError("query action-risk candidate count differs")
    return output


def risk_from_row_minimum(
    row_minimum_clearance_m: Sequence[float], safety_buffer_m: float
) -> list[float]:
    values = [float(item) for item in row_minimum_clearance_m]
    if len(values) != 7 or any(not math.isfinite(item) for item in values):
        raise ValueError("query action-risk row minima differ")
    return [float(safety_buffer_m) - item for item in values]


def combine_row_minima(*parts: Sequence[float]) -> list[float]:
    if not parts:
        raise ValueError("query action-risk composition is empty")
    rows = [[float(item) for item in part] for part in parts]
    if any(len(row) != 7 for row in rows):
        raise ValueError("query action-risk composition row count differs")
    return [min(row[index] for row in rows) for index in range(7)]


def exact_safe(record: Mapping[str, Any]) -> bool:
    return bool(
        record["terminal_status"] == "SAFE_TERMINAL"
        and not record["physical_veto"]
        and max(float(item) for item in record["combined_risk"]) <= 0.0
    )
