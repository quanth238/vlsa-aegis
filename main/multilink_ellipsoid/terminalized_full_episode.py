"""Contracts for the true full-episode terminalized risk-selector pilot."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


CONFIG_SCHEMA = "vlsa_terminalized_full_episode.v1"
RESULT_SCHEMA = "vlsa_terminalized_full_episode_result.v1"
VALIDATION_SCHEMA = "vlsa_terminalized_full_episode_validation.v1"


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
        raise ValueError("full-episode config schema differs")
    if value.get("protocol_id") != "vlsa-terminalized-full-episode-v1":
        raise ValueError("full-episode protocol differs")
    if value.get("case_id") != "vlsa-t1-goal-ii-t0-e05":
        raise ValueError("full-episode case differs")
    method = value.get("method", {})
    if not (
        method.get("start_step") == 0
        and method.get("replan_stride") == 5
        and method.get("candidate_count") == 13
        and method.get("candidate_generation")
        == "single_batched_late_flow_terminalization"
        and method.get("critic") == "frozen_compact_shared_7D"
        and method.get("selection")
        == "least_modifying_predicted_safe_else_abstain"
        and float(method.get("predicted_safe_threshold", 1.0)) == 0.0
    ):
        raise ValueError("full-episode method differs")
    forbidden = value.get("forbidden", {})
    if any(forbidden.get(key) is not False for key in (
        "archived_action_prefix", "released_AEGIS_EE_QP",
        "learned_QP", "simulator_rollout_verification",
        "model_training", "new_label_collection",
    )):
        raise ValueError("full-episode forbidden component is enabled")
    output = json.loads(canonical(value))
    output["config_file_sha256"] = hashlib.sha256(raw).hexdigest()
    output["config_payload_sha256"] = hashlib.sha256(canonical(value)).hexdigest()
    return output


def scientific_view(result: Mapping[str, Any]) -> dict[str, Any]:
    episode = result["episode"]
    return {
        "case_id": str(result["case_id"]),
        "pairing_sha256": str(result["pairing_sha256"]),
        "complete_executed_actions_sha256": str(
            episode["complete_executed_actions_sha256"]
        ),
        "action_count": int(episode["action_count"]),
        "selection_count": int(episode["selection_count"]),
        "selection_history_sha256": str(episode["selection_history_sha256"]),
        "abstention_count": int(episode["abstention_count"]),
        "terminal_reason": str(episode["terminal_reason"]),
        "native_task_success": bool(episode["native_task_success"]),
        "native_task_success_step": episode["native_task_success_step"],
        "timeout": bool(episode["timeout"]),
        "raw_robot_contact_pass": bool(episode["raw_robot_contact_pass"]),
        "first_raw_robot_contact": episode["first_raw_robot_contact"],
        "robot_contact_sample_count": int(
            episode["robot_contact_sample_count"]
        ),
        "robot_contact_sample_count_by_group": dict(
            episode["robot_contact_sample_count_by_group"]
        ),
        "robot_contact_events_sha256": str(
            episode["robot_contact_events_sha256"]
        ),
        "paper_CAR_pass": bool(episode["paper_CAR_pass"]),
        "first_paper_CAR_step": episode["first_paper_CAR_step"],
        "maximum_active_obstacle_l1_displacement_m": float(
            episode["maximum_active_obstacle_l1_displacement_m"]
        ),
        "collision_free_task_success": bool(
            episode["collision_free_task_success"]
        ),
        "terminal_dynamic_state_sha256": str(
            episode["terminal_dynamic_state_sha256"]
        ),
        "goal_progress_summary_sha256": str(
            episode["goal_progress_summary_sha256"]
        ),
    }
