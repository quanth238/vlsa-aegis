"""Contracts for one opened full-episode tight-prefix selector pilot."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


CONFIG_SCHEMA = "vlsa_tight_prefix_full_episode.v1"
RESULT_SCHEMA = "vlsa_tight_prefix_full_episode_result.v1"
VALIDATION_SCHEMA = "vlsa_tight_prefix_full_episode_validation.v1"
CASE_ID = "vlsa-t1-spatial-i-t3-e15"


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def payload_sha256(value: Mapping[str, Any], key: str) -> str:
    public = dict(value)
    public.pop(key, None)
    return hashlib.sha256(canonical(public)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    value = json.loads(raw)
    expected = {
        "schema_version", "protocol_id", "claim_scope", "case_id",
        "source", "method", "controller", "termination",
        "physical_contact_groups", "verification", "forbidden",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("tight-prefix full-episode config keys differ")
    if (
        value.get("schema_version") != CONFIG_SCHEMA
        or value.get("protocol_id") != "vlsa-tight-prefix-full-episode-v1"
        or value.get("case_id") != CASE_ID
    ):
        raise ValueError("tight-prefix full-episode protocol differs")
    method = value["method"]
    if method != {
        "start_step": 0,
        "replan_stride": 5,
        "candidate_count": 13,
        "candidate_names": [
            "nominal",
            "grid_m1_m1_m1_front_loaded_r2.0",
            "grid_p1_p1_p1_front_loaded_r2.0",
            "grid_m1_m1_z0_front_loaded_r2.0",
            "grid_p1_p1_z0_front_loaded_r2.0",
            "grid_m1_z0_m1_front_loaded_r2.0",
            "grid_p1_z0_p1_front_loaded_r2.0",
            "grid_m1_p1_z0_front_loaded_r2.0",
            "grid_p1_m1_z0_front_loaded_r2.0",
            "grid_z0_m1_m1_front_loaded_r2.0",
            "grid_z0_p1_p1_front_loaded_r2.0",
            "grid_z0_z0_m1_front_loaded_r2.0",
            "grid_z0_z0_p1_front_loaded_r2.0",
        ],
        "candidate_generation": (
            "current_minimum_slack_tight_row_local_frame_frozen_13_bank"
        ),
        "critic": "frozen_ADR_0201_compact_shared_7D_tight_prefix",
        "model_rows": list(range(10)),
        "primary_rows": list(range(8)),
        "diagnostic_rows": [8, 9],
        "selection": "minimum_predicted_primary_risk",
        "tie_break": "minimum_effective_correction_then_registered_order",
        "always_execute_selected": True,
        "translation_scale_m_per_action_unit": 0.05,
    }:
        raise ValueError("tight-prefix full-episode method differs")
    source = value["source"]
    required_source = {
        "training_config", "training_config_file_sha256",
        "training_config_payload_sha256", "training_result",
        "training_result_file_sha256", "training_result_payload_sha256",
        "model_sha256", "selector_validation",
        "selector_validation_file_sha256",
        "selector_validation_payload_sha256", "tight_dataset_config",
        "tight_dataset_config_file_sha256",
        "tight_dataset_config_payload_sha256", "geometry_archive",
        "geometry_archive_file_sha256", "geometry_archive_payload_sha256",
        "raw_pi05_archive", "raw_pi05_archive_file_sha256",
        "raw_pi05_archive_payload_sha256",
    }
    if set(source) != required_source:
        raise ValueError("tight-prefix full-episode source keys differ")
    forbidden = value["forbidden"]
    if any(forbidden.get(key) is not False for key in (
        "released_AEGIS_EE_QP", "learned_QP", "late_denoising",
        "simulator_candidate_rollout", "model_training", "data_collection",
    )):
        raise ValueError("tight-prefix full-episode forbidden component enabled")
    output = json.loads(canonical(value).decode("utf-8"))
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
        "robot_contact_sample_count": int(episode["robot_contact_sample_count"]),
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
