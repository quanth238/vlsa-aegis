"""Frozen contracts for analytical-repulsion task generalization."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


CONFIG_SCHEMA = "vlsa_distal_analytical_repulsion_generalization.v1"
CASE_SCHEMA = "vlsa_distal_analytical_repulsion_generalization_case.v1"
RESULT_SCHEMA = "vlsa_distal_analytical_repulsion_generalization_result.v1"


def frame_integrity_metrics(frame: Any) -> dict[str, float]:
    """Detect the dense row/column corruption caused by a stale OSMesa buffer."""

    import numpy as np

    values = np.asarray(frame)
    if values.ndim != 3 or values.shape[2] != 3 or values.dtype != np.uint8:
        raise ValueError("simulation video frame must be HWC uint8 RGB")
    signed = values.astype(np.int16)
    horizontal = float(np.mean(np.abs(signed[:, 1:] - signed[:, :-1]))) / 255.0
    vertical = float(np.mean(np.abs(signed[1:] - signed[:-1]))) / 255.0
    return {
        "horizontal_neighbor_difference": horizontal,
        "vertical_neighbor_difference": vertical,
        "maximum_neighbor_difference": max(horizontal, vertical),
    }


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def load_config(path: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    value = json.loads(raw)
    required = {
        "schema_version", "protocol_id", "claim_scope", "population",
        "pairing", "warning_oracle", "fixed_repulsion", "execution",
        "measurement", "gate", "forbidden",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("analytical-repulsion config keys differ")
    if value["schema_version"] != CONFIG_SCHEMA:
        raise ValueError("analytical-repulsion config schema differs")
    if value["protocol_id"] != "vlsa-distal-analytical-repulsion-generalization-v1":
        raise ValueError("analytical-repulsion protocol differs")
    if value["population"]["case_count"] != 3:
        raise ValueError("analytical-repulsion case count differs")
    if value["pairing"]["replan_steps"] != 5:
        raise ValueError("analytical-repulsion replan stride differs")
    if value["warning_oracle"] != {
        "evaluation": "privileged_cloned_OSC_five_post_AEGIS_actions_all_internal_substeps_with_observed_execution_fidelity_audit",
        "buffer_m": 0.001,
        "protected_contact_limit": 0,
        "paper_car_threshold_m": 0.001,
        "include_initial_state": True,
        "trigger": "any_proxy_row_below_buffer_or_protected_contact_or_CAR",
        "deployment_claim": False,
    }:
        raise ValueError("analytical-repulsion warning rule differs")
    fixed = value["fixed_repulsion"]
    if fixed != {
        "spatial_direction": "outward_center_normal_of_current_closest_L5_L7_slab",
        "temporal_profile": "front_loaded_unit_L2_5_4_3_2_1",
        "requested_correction_l2_action": 2.0,
        "corrected_dimensions": [0, 1, 2],
        "action_limit": 1.0,
        "apply_after_nominal_released_AEGIS": True,
        "reapply_unchanged_released_AEGIS_sequentially": True,
        "execute_even_if_corrected_oracle_remains_unsafe": True,
        "adaptive_magnitude": False,
        "tangent_or_mode_search": False,
    }:
        raise ValueError("analytical-repulsion rule differs")
    output = json.loads(canonical(value).decode("utf-8"))
    output["config_file_sha256"] = hashlib.sha256(raw).hexdigest()
    output["config_payload_sha256"] = hashlib.sha256(canonical(value)).hexdigest()
    return output


def load_cases(path: Path) -> list[dict[str, Any]]:
    raw = Path(path).read_bytes()
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    if len(rows) != 3:
        raise ValueError("analytical-repulsion selection must have three cases")
    required = {
        "active_obstacle_name", "archived_result_file_sha256",
        "archived_result_payload_sha256", "archived_result_relative_path",
        "case_id", "case_ordinal", "episode_index",
        "first_relevant_contact_step", "paper_car_step",
        "protected_contact_body", "schema_version", "selection_rule",
        "source_split", "suite", "task_level_group_id",
    }
    expected = [
        ("vlsa-t1-goal-ii-t2-e42", "robot0_link5"),
        ("vlsa-t1-goal-ii-t3-e42", "robot0_link6"),
        ("vlsa-t1-goal-ii-t3-e44", "robot0_link6"),
    ]
    for row, identity in zip(rows, expected):
        if not isinstance(row, dict) or set(row) != required:
            raise ValueError("analytical-repulsion case keys differ")
        if row["schema_version"] != CASE_SCHEMA:
            raise ValueError("analytical-repulsion case schema differs")
        if (row["case_id"], row["protected_contact_body"]) != identity:
            raise ValueError("analytical-repulsion case identity differs")
        if row["source_split"] != "test":
            raise ValueError("analytical-repulsion case is not sealed test")
    return json.loads(canonical(rows).decode("utf-8"))


def front_loaded_profile() -> list[float]:
    values = [5.0, 4.0, 3.0, 2.0, 1.0]
    norm = math.sqrt(sum(value * value for value in values))
    return [value / norm for value in values]


def corrected_proposal(
    nominal_actions: Sequence[Sequence[float]],
    outward_normal: Sequence[float],
    *,
    radius: float = 2.0,
    action_limit: float = 1.0,
) -> dict[str, Any]:
    if len(nominal_actions) != 5 or any(len(row) != 7 for row in nominal_actions):
        raise ValueError("analytical-repulsion nominal chunk differs")
    direction = [float(value) for value in outward_normal]
    if len(direction) != 3 or not all(math.isfinite(value) for value in direction):
        raise ValueError("analytical-repulsion direction differs")
    direction_norm = math.sqrt(sum(value * value for value in direction))
    if direction_norm <= 0.0:
        raise ValueError("analytical-repulsion direction is degenerate")
    direction = [value / direction_norm for value in direction]
    profile = front_loaded_profile()
    actions = [[float(value) for value in row] for row in nominal_actions]
    applied_sq = 0.0
    clipped = False
    for time_index in range(5):
        for axis in range(3):
            requested = float(radius) * profile[time_index] * direction[axis]
            proposed = actions[time_index][axis] + requested
            bounded = min(float(action_limit), max(-float(action_limit), proposed))
            applied = bounded - actions[time_index][axis]
            applied_sq += applied * applied
            clipped = clipped or abs(applied - requested) > 1.0e-12
            actions[time_index][axis] = bounded
    return {
        "actions": actions,
        "direction": direction,
        "temporal_profile": profile,
        "requested_correction_l2_action": float(radius),
        "applied_correction_l2_action": math.sqrt(applied_sq),
        "clipped": clipped,
    }


def warning_trigger(
    rollout: Mapping[str, Any], *, buffer_m: float = 0.001,
    car_threshold_m: float = 0.001,
) -> bool:
    return bool(
        float(rollout["minimum_clearance_m"]) < float(buffer_m)
        or bool(rollout["protected_contacts"])
        or float(rollout["maximum_active_obstacle_l1_displacement_m"])
        > float(car_threshold_m)
    )


def aggregate_case_results(
    rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any],
) -> dict[str, Any]:
    if len(rows) != int(config["gate"]["required_case_count"]):
        raise ValueError("analytical-repulsion aggregate case count differs")
    case_ids = [str(row["case_id"]) for row in rows]
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("analytical-repulsion aggregate duplicates cases")
    contact_pass = [bool(row["raw_L5_L7_contact_pass"]) for row in rows]
    car_pass = [bool(row["paper_car_pass"]) for row in rows]
    task_pass = [bool(row["native_task_success"]) for row in rows]
    timeout_pass = [not bool(row["timeout"]) for row in rows]
    primary = [
        contact and car and task and timeout
        for contact, car, task, timeout in zip(
            contact_pass, car_pass, task_pass, timeout_pass
        )
    ]
    return {
        "case_count": len(rows),
        "case_ids": case_ids,
        "raw_L5_L7_contact_pass_count": sum(contact_pass),
        "paper_car_pass_count": sum(car_pass),
        "native_task_success_count": sum(task_pass),
        "no_timeout_count": sum(timeout_pass),
        "safe_task_success_count": sum(primary),
        "total_warning_count": sum(int(row["warning_count"]) for row in rows),
        "total_intervention_count": sum(int(row["intervention_count"]) for row in rows),
        "total_clipped_intervention_count": sum(
            int(row["clipped_intervention_count"]) for row in rows
        ),
        "strict_gate_pass": all(primary),
    }
