"""Contracts for the terminalized late-flow task-success pilot."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


CONFIG_SCHEMA = "vlsa_terminalized_late_flow_task_success.v1"
RESULT_SCHEMA = "vlsa_terminalized_late_flow_task_success_result.v1"
VALIDATION_SCHEMA = "vlsa_terminalized_late_flow_task_success_validation.v1"
ARM_NAMES = (
    "nominal",
    "terminal_compact_selector",
    "late_flow_terminalized_compact_selector",
)


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def payload_sha256(value: Mapping[str, Any], key: str) -> str:
    public = dict(value)
    public.pop(key, None)
    return hashlib.sha256(canonical(public)).hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    value = json.loads(raw)
    if value.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("task-success config schema differs")
    if value.get("protocol_id") != "vlsa-terminalized-late-flow-task-success-v1":
        raise ValueError("task-success protocol differs")
    state = value.get("state_protocol", {})
    if int(state.get("archived_prefix_end_exclusive", -1)) != 180:
        raise ValueError("task-success archived prefix differs")
    if state.get("intervention_steps") != [180, 181, 182, 183, 184]:
        raise ValueError("task-success intervention horizon differs")
    if int(state.get("first_live_policy_query_index", -1)) != 37:
        raise ValueError("task-success first live query differs")
    if int(state.get("execute_actions_per_query", -1)) != 5:
        raise ValueError("task-success replan stride differs")
    if int(state.get("model_action_horizon", -1)) != 10:
        raise ValueError("task-success model horizon differs")
    if tuple(state.get("arms", ())) != ARM_NAMES:
        raise ValueError("task-success arm order differs")
    control = value.get("control", {})
    required_false = (
        "released_AEGIS_EE_QP_enabled",
        "learned_QP_enabled",
        "exact_rollout_verifier_at_inference",
        "additional_correction_after_first_chunk",
    )
    if any(control.get(key) is not False for key in required_false):
        raise ValueError("task-success forbidden control is enabled")
    if control.get("post_intervention_policy") != "raw_frozen_pi05_reobserve_requery":
        raise ValueError("task-success continuation policy differs")
    if control.get("OSC") != "unchanged_OSC_POSE":
        raise ValueError("task-success OSC differs")
    task = value.get("task_evaluation", {})
    if task.get("success_authority") != "native_environment_goal_done":
        raise ValueError("task-success authority differs")
    if task.get("timeout_authority") != "registered_suite_max_steps":
        raise ValueError("task-success timeout differs")
    if task.get("contact_authority") != "raw_MuJoCo_active_obstacle_robot_contacts":
        raise ValueError("task-success contact authority differs")
    if task.get("CAR_authority") != "active_obstacle_L1_displacement_gt_0.001_m":
        raise ValueError("task-success CAR authority differs")
    groups = value.get("physical_contact_groups", {})
    if groups.get("palm") != ["gripper0_hand_collision"]:
        raise ValueError("task-success palm contact binding differs")
    if groups.get("L5") != ["robot0_link5_collision"]:
        raise ValueError("task-success L5 contact binding differs")
    if groups.get("L6") != ["robot0_link6_collision"]:
        raise ValueError("task-success L6 contact binding differs")
    if groups.get("L7") != ["robot0_link7_collision"]:
        raise ValueError("task-success L7 contact binding differs")
    output = json.loads(canonical(value))
    output["config_file_sha256"] = hashlib.sha256(raw).hexdigest()
    output["config_payload_sha256"] = hashlib.sha256(canonical(value)).hexdigest()
    return output


def selected_arm_actions(
    pilot: Mapping[str, Any],
) -> list[dict[str, Any]]:
    candidates = pilot.get("fresh_selected_arm_execution", {}).get(
        "exact_case", {}
    ).get("candidates", [])
    if [row.get("name") for row in candidates] != list(ARM_NAMES):
        raise ValueError("task-success pilot arm order differs")
    outcomes = pilot.get("scientific_view", {}).get("outcome", {}).get(
        "records", []
    )
    by_arm = {str(row.get("arm")): row for row in outcomes}
    if set(by_arm) != set(ARM_NAMES):
        raise ValueError("task-success pilot outcomes differ")
    output = []
    for candidate in candidates:
        name = str(candidate["name"])
        actions = candidate.get("source_executed_actions")
        if (
            not isinstance(actions, list)
            or len(actions) != 5
            or any(not isinstance(row, list) or len(row) != 7 for row in actions)
        ):
            raise ValueError("task-success selected action shape differs")
        encoded = canonical(actions)
        source_hash = by_arm[name].get("executed_actions_sha256")
        # The older pilot hashes compact JSON without sorted keys. For a list,
        # canonical JSON is byte-identical to that encoding.
        if hashlib.sha256(encoded).hexdigest() != source_hash:
            raise ValueError("task-success selected action hash differs")
        output.append({
            "arm": name,
            "source_candidate": str(
                by_arm[name]["selection"]["source_candidate"]
            ),
            "actions": actions,
            "actions_sha256": source_hash,
            "selection": by_arm[name]["selection"],
        })
    return output


def contact_group(
    geom_name: str, groups: Mapping[str, Sequence[str]],
) -> str:
    matches = [
        str(group) for group, names in groups.items()
        if str(geom_name) in [str(name) for name in names]
    ]
    if len(matches) > 1:
        raise ValueError("robot contact geom maps to multiple groups")
    return matches[0] if matches else "other_robot"


def scientific_view(
    *, case_id: str, source_snapshot_sha256: str,
    arm_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if [row.get("arm") for row in arm_records] != list(ARM_NAMES):
        raise ValueError("task-success result arm order differs")
    arms = []
    for row in arm_records:
        arms.append({
            "arm": str(row["arm"]),
            "source_candidate": str(row["source_candidate"]),
            "intervention_actions_sha256": str(
                row["intervention_actions_sha256"]
            ),
            "complete_executed_actions_sha256": str(
                row["complete_executed_actions_sha256"]
            ),
            "action_count": int(row["action_count"]),
            "live_policy_query_count": int(row["live_policy_query_count"]),
            "native_task_success": bool(row["native_task_success"]),
            "native_task_success_step": row["native_task_success_step"],
            "timeout": bool(row["timeout"]),
            "raw_robot_contact_pass": bool(row["raw_robot_contact_pass"]),
            "first_raw_robot_contact": row["first_raw_robot_contact"],
            "robot_contact_sample_count": int(
                row["robot_contact_sample_count"]
            ),
            "robot_contact_sample_count_by_group": dict(
                row["robot_contact_sample_count_by_group"]
            ),
            "robot_contact_events_sha256": str(
                row["robot_contact_events_sha256"]
            ),
            "paper_CAR_pass": bool(row["paper_CAR_pass"]),
            "first_paper_CAR_step": row["first_paper_CAR_step"],
            "maximum_active_obstacle_l1_displacement_m": float(
                row["maximum_active_obstacle_l1_displacement_m"]
            ),
            "collision_free_task_success": bool(
                row["collision_free_task_success"]
            ),
            "terminal_dynamic_state_sha256": str(
                row["terminal_dynamic_state_sha256"]
            ),
            "goal_progress_summary_sha256": str(
                row["goal_progress_summary_sha256"]
            ),
        })
    return {
        "case_id": str(case_id),
        "source_snapshot_sha256": str(source_snapshot_sha256),
        "arm_count": len(arms),
        "task_success_count": sum(row["native_task_success"] for row in arms),
        "collision_free_task_success_count": sum(
            row["collision_free_task_success"] for row in arms
        ),
        "arms": arms,
    }
