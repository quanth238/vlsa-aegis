"""Contracts for privileged receding exact-candidate selection on E05."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


EXACT_CANDIDATE_CONFIG_SCHEMA = "vlsa_distal_exact_candidate_closed_loop_e05.v1"
EXACT_CANDIDATE_RESULT_SCHEMA = (
    "vlsa_distal_exact_candidate_closed_loop_e05_result.v1"
)
EXACT_CANDIDATE_VALIDATION_SCHEMA = (
    "vlsa_distal_exact_candidate_closed_loop_e05_validation.v1"
)

CONSTRAINT_ORDER = [
    "L5_part_0", "L5_part_1", "L5_part_2", "L6_part_0",
    "L6_part_1", "L7_part_0", "L7_part_1", "released_AEGIS_EE_proxy",
]


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_exact_candidate_config(path: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    try:
        config = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("exact-candidate config is invalid JSON") from error
    required = {
        "schema_version", "protocol_id", "claim_scope", "primary_case",
        "prerequisite", "constraint_order", "nominal_plan", "activation",
        "sampling", "candidate_selector", "execution", "decision_gate",
    }
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError("exact-candidate config keys differ")
    if (
        config["schema_version"] != EXACT_CANDIDATE_CONFIG_SCHEMA
        or config["protocol_id"]
        != "vlsa-distal-exact-candidate-closed-loop-e05-v1"
    ):
        raise ValueError("exact-candidate protocol differs")
    if config["primary_case"] != {
        "case_id": "vlsa-t1-goal-ii-t0-e05",
        "expected_action_count": 237,
        "original_first_L5_contact_step": 187,
        "original_first_paper_CAR_step": 188,
        "original_native_task_success_step": 236,
    }:
        raise ValueError("exact-candidate primary case differs")
    prerequisite = config["prerequisite"]
    if set(prerequisite) != {
        "eight_row_result_file_sha256", "eight_row_result_payload_sha256",
        "eight_row_validation_file_sha256", "require_observed_safe_grid_action",
    } or prerequisite["require_observed_safe_grid_action"] is not True:
        raise ValueError("exact-candidate prerequisite differs")
    for key in (
        "eight_row_result_file_sha256", "eight_row_result_payload_sha256",
        "eight_row_validation_file_sha256",
    ):
        if not isinstance(prerequisite[key], str) or len(prerequisite[key]) != 64:
            raise ValueError("exact-candidate prerequisite hash differs")
    if config["constraint_order"] != CONSTRAINT_ORDER:
        raise ValueError("exact-candidate constraint order differs")
    if config["nominal_plan"] != {
        "source": "immutable_successful_released_AEGIS_Table1_env_step_sequence",
        "orientation_gripper_channels": "unchanged",
        "second_action": "next_immutable_AEGIS_action",
        "terminal_horizon": "one_action_when_no_next_action_exists",
    }:
        raise ValueError("exact-candidate nominal plan differs")
    if config["activation"] != {
        "mode": "fresh_exact_cloned_OSC_horizon_check_before_every_execution",
        "horizon_actions": 2,
        "unsafe_if_any_eight_margin_negative": True,
        "unsafe_if_raw_L5_L6_L7_contact": True,
        "maximum_per_step_obstacle_l1_displacement_m": 1.0e-4,
    }:
        raise ValueError("exact-candidate activation differs")
    if config["sampling"] != {
        "vary": "first_action_normalized_XYZ_only",
        "action_limit": 1.0,
        "trust_region_linf_action": 0.5,
        "grid_points_per_dimension": 8,
        "expected_grid_action_count": 512,
        "candidate_order": "cartesian_product_lexicographic",
    }:
        raise ValueError("exact-candidate sampling differs")
    if config["candidate_selector"] != {
        "method": "exact_safe_then_minimum_L2_from_nominal_then_grid_index",
        "minimum_all_eight_margin_m": 1.0e-6,
        "require_raw_L5_L6_L7_contact_free": True,
        "require_obstacle_motion_gate": True,
        "fresh_exact_rollout_before_execution": True,
        "affine_QP_used": False,
        "fallback": "none_stop_if_no_exact_safe_candidate",
    }:
        raise ValueError("exact-candidate selector differs")
    if config["execution"] != {
        "execute": "verified_first_action_only_then_replan",
        "osc_internal_substeps": "all",
        "require_all_eight_margins_nonnegative": True,
        "require_zero_raw_L5_L6_L7_contact": True,
        "require_executed_next_state_hash_equals_clone": True,
        "paper_CAR_threshold_m": 1.0e-3,
    }:
        raise ValueError("exact-candidate execution differs")
    if config["decision_gate"] != {
        "success": (
            "all_executed_substeps_all_eight_proxy_safe_and_zero_raw_"
            "L5_L6_L7_contact_and_zero_paper_CAR_and_native_task_success"
        ),
        "go_interpretation": (
            "privileged_exact_candidate_receding_oracle_can_solve_primary_"
            "E05_not_QP_not_learned_not_deployable_safety"
        ),
        "no_go_interpretation": (
            "registered_two_step_grid_action_family_cannot_complete_E05_"
            "not_impossibility_of_longer_horizon_or_larger_action_search"
        ),
        "neural_training_authorized": False,
    }:
        raise ValueError("exact-candidate decision gate differs")
    output = json.loads(_canonical(config).decode("utf-8"))
    output["config_file_sha256"] = _sha256(raw)
    output["config_payload_sha256"] = _sha256(_canonical(config))
    return output


def select_exact_safe_candidate(
    candidate_xyz: Sequence[Sequence[float]],
    minimum_all_eight_margin_m: Sequence[Sequence[float]],
    raw_safe: Sequence[bool],
    grid_indexes: Sequence[int],
    nominal_xyz: Sequence[float],
    *,
    minimum_margin_m: float,
) -> dict[str, Any]:
    """Select the closest sampled action satisfying the complete exact gate."""

    import numpy as np

    xyz = np.asarray(candidate_xyz, dtype=np.float64)
    margins = np.asarray(minimum_all_eight_margin_m, dtype=np.float64)
    raw = np.asarray(raw_safe, dtype=bool)
    indexes = np.asarray(grid_indexes, dtype=np.int64)
    nominal = np.asarray(nominal_xyz, dtype=np.float64)
    threshold = float(minimum_margin_m)
    if (
        xyz.ndim != 2 or xyz.shape[1] != 3
        or margins.shape != (len(xyz), 8)
        or raw.shape != (len(xyz),) or indexes.shape != (len(xyz),)
        or nominal.shape != (3,) or len(set(indexes.tolist())) != len(indexes)
        or not math.isfinite(threshold) or threshold < 0.0
        or any(not np.all(np.isfinite(value)) for value in (xyz, margins, nominal))
    ):
        raise ValueError("exact-candidate arrays are invalid")
    eligible = np.flatnonzero(
        np.logical_and(np.all(margins >= threshold, axis=1), raw)
    )
    if not len(eligible):
        return {
            "valid": False,
            "reason": "no_exact_all_eight_safe_grid_candidate",
            "eligible_candidate_count": 0,
            "selected_candidate_xyz": None,
        }
    selected = min(
        eligible.tolist(),
        key=lambda item: (
            float(np.linalg.norm(xyz[item] - nominal)), int(indexes[item])
        ),
    )
    return {
        "valid": True,
        "reason": "exact_safe_candidate_selected",
        "eligible_candidate_count": int(len(eligible)),
        "selected_candidate_array_index": int(selected),
        "selected_grid_index": int(indexes[selected]),
        "selected_candidate_xyz": xyz[selected].tolist(),
        "selected_candidate_margin_m": margins[selected].tolist(),
        "selected_candidate_correction_l2": float(
            np.linalg.norm(xyz[selected] - nominal)
        ),
    }
