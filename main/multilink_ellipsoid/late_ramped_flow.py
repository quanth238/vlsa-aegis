"""Strict late-ramped fixed-repulsion timing experiment primitives."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .repulsive_force import normalized_direction
from .shadow import _numpy


LATE_RAMPED_FLOW_SCHEMA = "vlsa_late_ramped_repulsion_flow_e05.v1"


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def load_late_ramped_flow_config(path: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    value = json.loads(raw)
    required = {
        "case_ids",
        "claim_scope",
        "comparison",
        "flow_guidance",
        "nominal_action_source",
        "protected_geometry",
        "protocol_id",
        "repulsive_direction",
        "schema_version",
        "state_protocol",
        "success_definition",
        "verification",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("late-ramped config keys differ")
    if value["schema_version"] != LATE_RAMPED_FLOW_SCHEMA:
        raise ValueError("late-ramped schema differs")
    if value["case_ids"] != ["vlsa-t1-goal-ii-t0-e05"]:
        raise ValueError("late-ramped case differs")
    if value["state_protocol"] != {
        "activation_query_step": 180,
        "evaluated_action_steps": [180, 181, 182, 183, 184],
        "guided_executed_steps": [182, 183, 184],
    }:
        raise ValueError("late-ramped state protocol differs")
    flow = value["flow_guidance"]
    expected_schedules = {
        "final_step_only": [0.0] * 9 + [0.25],
        "late_linear_last_two": [0.0] * 8
        + [0.08333333333333333, 0.16666666666666666],
        "uniform_final_five": [0.0] * 5 + [0.05] * 5,
    }
    if flow != {
        "action_limit": 1.0,
        "guided_action_slots": [2, 3, 4],
        "injected_total_per_slot_action": 0.25,
        "schedules": expected_schedules,
    }:
        raise ValueError("late-ramped schedules differ")
    for schedule in flow["schedules"].values():
        if abs(sum(schedule) - float(flow["injected_total_per_slot_action"])) > 1e-12:
            raise ValueError("late-ramped injected budget differs")
    output = json.loads(_canonical(value).decode("utf-8"))
    output["config_file_sha256"] = hashlib.sha256(raw).hexdigest()
    output["config_payload_sha256"] = hashlib.sha256(_canonical(value)).hexdigest()
    return output


def build_schedule_envelope(
    nominal_actions: Any,
    direction: Any,
    *,
    guided_slots: list[int],
    schedule: list[float],
    action_limit: float,
) -> dict[str, Any]:
    np = _numpy()
    nominal = np.asarray(nominal_actions, dtype=np.float64)
    unit = normalized_direction(direction)
    strengths = np.asarray(schedule, dtype=np.float64)
    if nominal.shape != (10, 7):
        raise ValueError("late-ramped nominal chunk shape differs")
    if guided_slots != [2, 3, 4]:
        raise ValueError("late-ramped guided slots differ")
    if strengths.shape != (10,) or not np.all(np.isfinite(strengths)):
        raise ValueError("late-ramped schedule shape differs")
    if np.min(strengths) < 0.0 or float(np.sum(strengths)) <= 0.0:
        raise ValueError("late-ramped schedule is invalid")
    return {
        "schema_version": "crfs_scheduled_repulsive_flow_guidance.v1",
        "action_horizon": 10,
        "action_dimensions": [0, 1, 2],
        "physical_output_direction": unit.tolist(),
        "nominal_output_actions": nominal.tolist(),
        "guided_action_slots": list(guided_slots),
        "euler_step_strengths_action": strengths.tolist(),
        "action_limit": float(action_limit),
    }


def surviving_output_correction(
    nominal_actions: Any,
    guided_actions: Any,
    guided_slots: list[int],
) -> tuple[Any, float]:
    np = _numpy()
    nominal = np.asarray(nominal_actions, dtype=np.float64)
    guided = np.asarray(guided_actions, dtype=np.float64)
    if nominal.shape != (10, 7) or guided.shape != (10, 7):
        raise ValueError("surviving correction chunk shape differs")
    correction = guided[guided_slots, :3] - nominal[guided_slots, :3]
    return correction, float(np.linalg.norm(correction))


def norm_matched_posthoc_chunk(
    nominal_actions: Any,
    direction: Any,
    *,
    guided_slots: list[int],
    target_correction_l2: float,
    action_limit: float,
) -> tuple[Any, float]:
    """Apply post-hoc repulsion with the same surviving chunk correction norm."""

    np = _numpy()
    nominal = np.asarray(nominal_actions, dtype=np.float64)
    unit = normalized_direction(direction)
    if nominal.shape != (10, 7) or guided_slots != [2, 3, 4]:
        raise ValueError("matched posthoc chunk shape differs")
    target = float(target_correction_l2)
    if not np.isfinite(target) or target < 0.0:
        raise ValueError("matched posthoc target norm differs")

    def apply(per_slot: float) -> tuple[Any, float]:
        actions = nominal.copy()
        for slot in guided_slots:
            actions[slot, :3] = np.clip(
                actions[slot, :3] + per_slot * unit,
                -float(action_limit),
                float(action_limit),
            )
        correction = actions[guided_slots, :3] - nominal[guided_slots, :3]
        return actions, float(np.linalg.norm(correction))

    if target == 0.0:
        return apply(0.0)
    lower = 0.0
    upper = max(1.0, target)
    _, upper_norm = apply(upper)
    while upper_norm < target - 1e-12 and upper < 16.0:
        upper *= 2.0
        _, upper_norm = apply(upper)
    if upper_norm < target - 1e-10:
        raise ValueError("matched posthoc target is infeasible under clipping")
    for _ in range(80):
        midpoint = 0.5 * (lower + upper)
        _, norm = apply(midpoint)
        if norm < target:
            lower = midpoint
        else:
            upper = midpoint
    actions, observed = apply(0.5 * (lower + upper))
    if abs(observed - target) > 1e-8:
        raise ValueError("matched posthoc correction norm differs")
    return actions, observed
