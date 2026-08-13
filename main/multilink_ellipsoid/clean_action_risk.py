"""Contracts for the clean task-successful action-risk experiment.

The experiment is deliberately split into an exact oracle-label gate and a
later learned-prediction gate.  Nothing in this module changes released AEGIS.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


CONFIG_SCHEMA = "vlsa_distal_clean_action_risk.v1"
CASE_SCHEMA = "vlsa_distal_clean_action_risk_case.v1"
RESULT_SCHEMA = "vlsa_distal_clean_action_risk_dataset.v1"
VALIDATION_SCHEMA = "vlsa_distal_clean_action_risk_dataset_validation.v1"
MODEL_RESULT_SCHEMA = "vlsa_distal_clean_action_risk_model.v1"


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    value = json.loads(raw)
    required = {
        "schema_version", "protocol_id", "claim_scope", "source_evidence",
        "cohort", "state_sampling", "candidate_family", "risk_target",
        "model", "prediction_gate", "forbidden",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("clean action-risk config keys differ")
    if value["schema_version"] != CONFIG_SCHEMA:
        raise ValueError("clean action-risk schema differs")
    if value["protocol_id"] != "vlsa-distal-clean-action-risk-v1":
        raise ValueError("clean action-risk protocol differs")
    if value["state_sampling"] != {
        "lead_actions_before_first_protected_contact": [3],
        "include_k0_in_strict_safety": True,
        "minimum_initial_proxy_clearance_m": 0.001,
        "maximum_initial_active_obstacle_l1_displacement_m": 0.001,
        "expected_mujoco_substeps_per_action": 25,
        "boundary_equivalence_tolerance": 1.0e-12,
    }:
        raise ValueError("clean action-risk state sampling differs")
    candidate = value["candidate_family"]
    if candidate != {
        "proposal_members": "nominal_aegis_plus_hold_plus_world_axes_plus_local_normal_tangents",
        "backup_members": "hold_plus_world_axes_plus_local_normal_tangents",
        "amplitudes_action": [0.5, 1.0],
        "proposal_count": 26,
        "backup_candidate_count": 25,
        "execute_proposal_actions": 1,
        "execute_selected_backup_actions": 1,
        "terminal_hold_actions": 25,
        "rotation_action": [0.0, 0.0, 0.0],
        "backup_gripper_action": 0.0,
        "vla_ledger_forbidden_from_backup_candidates": True,
        "no_safe_backup_fallback": "maximum_future_clearance_then_fixed_registered_order",
    }:
        raise ValueError("clean action-risk candidate family differs")
    target = value["risk_target"]
    if target != {
        "output_count": 7,
        "definition": "safety_buffer_m_minus_rollout_minimum_proxy_clearance_m",
        "positive_is_unsafe": True,
        "safety_buffer_m": 0.001,
        "paper_car_threshold_m": 0.001,
        "physical_acceptance": "zero_protected_MuJoCo_contact_and_CAR_at_most_1mm",
        "proxy_acceptance": "all_seven_risks_nonpositive",
    }:
        raise ValueError("clean action-risk target differs")
    if value["cohort"]["split_unit"] != "complete_episode":
        raise ValueError("clean action-risk split unit differs")
    if value["model"].get("input_dimension") != 167:
        raise ValueError("clean action-risk model input dimension differs")
    if value["model"]["output_count"] != 7:
        raise ValueError("clean action-risk model output differs")
    if value["model"] != {
        "class": "single_action_conditioned_seven_output_MLP",
        "input_dimension": 167,
        "output_count": 7,
        "hidden_widths": [256, 256, 128],
        "activation": "silu",
        "input": "compact_robot_controller_state_candidate_action_obstacle_and_seven_current_geometry_rows",
        "loss": "boundary_weighted_Huber_on_seven_quantitative_risks",
        "seed": 20260813,
        "cpu_threads": 8,
        "epochs": 1500,
        "patience": 150,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "gradient_clip_norm": 10.0,
        "huber_beta_normalized": 0.25,
        "boundary_scale_m": 0.005,
        "boundary_weight_multiplier": 4.0,
        "minimum_target_scale_m": 0.001,
        "batching": "deterministic_full_batch",
        "checkpoint_metric": "validation_boundary_weighted_Huber",
        "early_stopping": "validation_episodes_only",
        "calibration": "none_in_prediction_gate",
    }:
        raise ValueError("clean action-risk model protocol differs")
    output = json.loads(canonical(value).decode("utf-8"))
    output["config_file_sha256"] = sha256(raw)
    output["config_payload_sha256"] = sha256(canonical(value))
    return output


def load_cases(path: Path, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = Path(path).read_bytes()
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    expected_counts = config["cohort"]["episode_counts"]
    counts = {name: sum(row.get("split") == name for row in rows) for name in expected_counts}
    if counts != expected_counts or len(rows) != sum(expected_counts.values()):
        raise ValueError("clean action-risk split counts differ")
    if sha256(raw) != config["cohort"]["selection_manifest_file_sha256"]:
        raise ValueError("clean action-risk selection manifest differs")
    required = {
        "schema_version", "case_id", "split", "suite", "task_level_group_id",
        "episode_index", "case_ordinal", "active_obstacle_name",
        "protected_contact_body", "first_relevant_contact_step",
        "paper_car_step", "archived_result_relative_path",
        "archived_result_file_sha256", "archived_result_payload_sha256",
        "source_static_eligibility",
    }
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != required:
            raise ValueError("clean action-risk case keys differ")
        if row["schema_version"] != CASE_SCHEMA:
            raise ValueError("clean action-risk case schema differs")
        if row["case_id"] in seen:
            raise ValueError("clean action-risk case repeats")
        seen.add(row["case_id"])
        eligibility = row["source_static_eligibility"]
        if not (
            eligibility["aegis_native_task_success"]
            and eligibility["aegis_paper_car_failure"]
            and eligibility["geometry_complete"]
            and eligibility["postcontrol_robot_contact"]
            and not eligibility["settled_relevant_contact"]
            and not eligibility["postcontrol_dynamic_task_contact"]
            and not eligibility["postcontrol_dynamic_other_contact"]
            and not eligibility["mvee_materially_invalid"]
            and not eligibility["mvee_detached_from_retained_cloud"]
            and not eligibility["mvee_solver_inaccurate"]
            and eligibility["refined_collision_mechanism"]
            == "positive_h_unmodelled_arm_link_contact"
        ):
            raise ValueError("clean action-risk static eligibility differs")
        if row["protected_contact_body"] not in ("robot0_link5", "robot0_link6", "robot0_link7"):
            raise ValueError("clean action-risk protected body differs")
        if int(row["first_relevant_contact_step"]) <= max(
            config["state_sampling"]["lead_actions_before_first_protected_contact"]
        ):
            raise ValueError("clean action-risk contact occurs too early")
    return rows


def decision_steps(case: Mapping[str, Any], config: Mapping[str, Any]) -> list[int]:
    contact = int(case["first_relevant_contact_step"])
    return [
        contact - int(lead)
        for lead in config["state_sampling"]["lead_actions_before_first_protected_contact"]
    ]


def select_proxy_boundary_lead_state(
    minimum_clearance_m: Sequence[float],
    physically_valid: Sequence[bool],
    *,
    safety_buffer_m: float,
    lead_actions: int,
    stop_before_step: int,
) -> dict[str, Any]:
    """Select a prevention state relative to the first modeled boundary crossing.

    Values describe states immediately before their same-indexed actions.  The
    selector depends on geometry and current physical validity, not raw
    time-to-contact.  It never skips an earlier proxy crossing to choose a more
    convenient later state.
    """

    clearance = [float(item) for item in minimum_clearance_m]
    valid = [bool(item) for item in physically_valid]
    if len(clearance) != len(valid) or not clearance:
        raise ValueError("warning-state audit traces differ")
    if any(not math.isfinite(item) for item in clearance):
        raise ValueError("warning-state audit clearance is nonfinite")
    if not 0 < int(stop_before_step) <= len(clearance):
        raise ValueError("warning-state audit stop step differs")
    if int(lead_actions) < 1:
        raise ValueError("warning-state audit lead must be positive")
    crossing = next(
        (
            index
            for index in range(int(stop_before_step))
            if clearance[index] < float(safety_buffer_m)
        ),
        None,
    )
    if crossing is None:
        return {"status": "no_proxy_boundary_crossing_before_contact"}
    selected = int(crossing) - int(lead_actions)
    if selected < 0:
        return {
            "status": "boundary_crossing_too_early",
            "boundary_crossing_step": int(crossing),
        }
    if clearance[selected] < float(safety_buffer_m) or not valid[selected]:
        return {
            "status": "selected_state_not_strictly_safe",
            "boundary_crossing_step": int(crossing),
            "selected_step": int(selected),
        }
    return {
        "status": "selected",
        "boundary_crossing_step": int(crossing),
        "selected_step": int(selected),
        "selected_minimum_clearance_m": clearance[selected],
        "lead_actions_before_proxy_boundary": int(lead_actions),
    }


def risk_from_row_minimum(
    row_minimum_clearance_m: Sequence[float], safety_buffer_m: float,
) -> list[float]:
    values = [float(item) for item in row_minimum_clearance_m]
    if len(values) != 7 or any(not math.isfinite(item) for item in values):
        raise ValueError("clean action-risk row minimum differs")
    return [float(safety_buffer_m) - item for item in values]


def exact_safe(record: Mapping[str, Any], config: Mapping[str, Any]) -> bool:
    return bool(
        max(float(item) for item in record["risk"]) <= 0.0
        and int(record["protected_contact_count"]) == 0
        and float(record["maximum_active_obstacle_l1_displacement_m"])
        <= float(config["risk_target"]["paper_car_threshold_m"])
    )


def compact_feature_vector(
    context: Mapping[str, Any], candidate_action: Sequence[float],
) -> list[float]:
    """Fixed-size model input; full dynamic state remains replay evidence only."""

    def flatten(value: Any) -> list[float]:
        if isinstance(value, (list, tuple)):
            output = []
            for item in value:
                output.extend(flatten(item))
            return output
        return [float(value)]

    values = []
    for key, expected in (
        ("arm_joint_position_rad", 7),
        ("arm_joint_velocity_rad_s", 7),
        ("eef_position_m", 3),
        ("eef_quaternion_xyzw", 4),
    ):
        group = [float(item) for item in context[key]]
        if len(group) != expected:
            raise ValueError("clean action-risk compact context differs")
        values.extend(group)
    controller = context["controller_snapshot"]
    for key, expected in (("goal_pos", 3), ("goal_ori", 9)):
        group = flatten(controller[key])
        if len(group) != expected:
            raise ValueError("clean action-risk controller context differs")
        values.extend(group)
    obstacle = context["obstacle"]
    values.extend(float(item) for item in obstacle["center_m"])
    values.extend(flatten(obstacle["rotation"]))
    values.extend(float(item) for item in obstacle["semiaxes_m"])
    rows = context["geometry_rows"]
    if len(rows) != 7:
        raise ValueError("clean action-risk geometry row count differs")
    for row in rows:
        values.extend(float(item) for item in row["center_m"])
        values.extend(flatten(row["rotation"]))
        values.extend(float(item) for item in row["semiaxes_m"])
        values.append(float(row["current_clearance_m"]))
    action = [float(item) for item in candidate_action]
    if len(action) != 7:
        raise ValueError("clean action-risk candidate action differs")
    values.extend(action)
    if len(values) != 167 or any(not math.isfinite(item) for item in values):
        raise ValueError("clean action-risk compact feature dimension differs")
    return values


def prediction_metrics(
    predicted_risk: Sequence[Sequence[float]],
    samples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    predicted = [[float(value) for value in row] for row in predicted_risk]
    exact = [[float(value) for value in sample["risk"]] for sample in samples]
    if len(predicted) != len(exact) or any(
        len(left) != 7 or len(right) != 7
        for left, right in zip(predicted, exact)
    ):
        raise ValueError("clean action-risk prediction arrays differ")
    predicted_safe = [max(row) <= 0.0 for row in predicted]
    exact_safe_mask = [bool(sample["exact_safe"]) for sample in samples]
    false_safe = [left and not right for left, right in zip(predicted_safe, exact_safe_mask)]
    accepted_true_safe = [left and right for left, right in zip(predicted_safe, exact_safe_mask)]
    recall = float(sum(accepted_true_safe) / max(1, sum(exact_safe_mask)))
    by_state: dict[str, dict[str, Any]] = {}
    for index, sample in enumerate(samples):
        key = str(sample["state_id"])
        entry = by_state.setdefault(key, {"exact_safe": 0, "predicted_safe": 0, "accepted_safe": 0})
        entry["exact_safe"] += int(exact_safe_mask[index])
        entry["predicted_safe"] += int(predicted_safe[index])
        entry["accepted_safe"] += int(accepted_true_safe[index])
    recoverable = [key for key, value in by_state.items() if value["exact_safe"] > 0]
    supported = [key for key in recoverable if by_state[key]["accepted_safe"] > 0]
    squared_error = sum(
        (left - right) ** 2
        for left_row, right_row in zip(predicted, exact)
        for left, right in zip(left_row, right_row)
    )
    element_count = max(1, len(exact) * 7)
    return {
        "sample_count": int(len(samples)),
        "risk_rmse_mm": float(math.sqrt(squared_error / element_count) * 1000.0),
        "false_safe_count": int(sum(false_safe)),
        "exact_safe_count": int(sum(exact_safe_mask)),
        "predicted_safe_count": int(sum(predicted_safe)),
        "accepted_exact_safe_count": int(sum(accepted_true_safe)),
        "safe_recall": recall,
        "recoverable_state_count": len(recoverable),
        "supported_recoverable_state_count": len(supported),
        "per_state": by_state,
    }
