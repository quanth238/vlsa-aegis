"""Contracts for the E05 receding five-action exact oracle."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


RECEDING_FIVE_ACTION_SCHEMA = "vlsa_distal_receding_five_action_e05.v1"


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def load_receding_five_action_config(path: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    value = json.loads(raw)
    required = {
        "case_ids",
        "claim_scope",
        "finite_difference",
        "protected_geometry",
        "protocol_id",
        "schema_version",
        "sequential_qp",
        "state_protocol",
        "success_definition",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("receding five-action config keys differ")
    if value["schema_version"] != RECEDING_FIVE_ACTION_SCHEMA:
        raise ValueError("receding five-action schema differs")
    if value["case_ids"] != ["vlsa-t1-goal-ii-t0-e05"]:
        raise ValueError("receding five-action case differs")
    expected_state = {
        "activation_step": 182,
        "action_horizon": 5,
        "execute_prefix_actions": 1,
        "initial_window": "immutable_archived_aegis_actions_182_through_186",
        "later_windows": "fresh_frozen_pi05_translation_chunk_from_measured_state",
        "replan_after_every_executed_action": True,
        "endpoint_preservation": "sum_xyz_correction_over_five_actions_equals_zero",
    }
    if value["state_protocol"] != expected_state:
        raise ValueError("receding five-action state protocol differs")
    if value["finite_difference"] != {
        "action_dimensions": [0, 1, 2],
        "perturbation_action": 0.05,
        "scheme": "clipped_central_difference_at_every_sqp_iteration",
    }:
        raise ValueError("receding five-action finite difference differs")
    qp = value["sequential_qp"]
    expected_qp = {
        "action_limit",
        "clearance_buffer_m",
        "clearance_slack_weight",
        "iterations",
        "maximum_abs_total_correction_action",
        "maximum_abs_iteration_step_action",
        "maximum_terminal_eef_error_m",
        "minimum_progress_ratio",
        "nominal_deviation_weight",
        "smoothness_weight",
        "terminal_eef_weight",
    }
    if not isinstance(qp, Mapping) or set(qp) != expected_qp:
        raise ValueError("receding five-action QP keys differ")
    if int(qp["iterations"]) != 5:
        raise ValueError("receding five-action iteration count differs")
    for key in expected_qp - {"iterations"}:
        item = qp[key]
        if isinstance(item, bool) or not math.isfinite(float(item)) or float(item) < 0:
            raise ValueError("receding five-action QP value differs: %s" % key)
    output = json.loads(_canonical(value).decode("utf-8"))
    output["config_file_sha256"] = hashlib.sha256(raw).hexdigest()
    output["config_payload_sha256"] = hashlib.sha256(_canonical(value)).hexdigest()
    return output


def rollout_is_safe(
    rollout: Mapping[str, Any],
    *,
    clearance_buffer_m: float,
    paper_car_threshold_m: float,
) -> bool:
    """Use the same strict physical gate for nominal and corrected chunks."""

    displacements = rollout.get("active_obstacle_l1_displacement_m")
    return bool(
        float(rollout["minimum_clearance_m"]) >= float(clearance_buffer_m)
        and not rollout.get("protected_contacts")
        and not rollout.get("robot_contacts")
        and isinstance(displacements, list)
        and len(displacements) == 5
        and max(float(value) for value in displacements)
        <= float(paper_car_threshold_m)
    )


def receding_query_index(step: int) -> int:
    """Use query 37 at action 183, after archived query 36 supplies step 182."""

    if int(step) < 183:
        raise ValueError("fresh receding query starts at action 183")
    return 37 + int(step) - 183
