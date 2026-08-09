"""Contracts for the privileged receding affine-oracle E05 diagnostic.

This module is opt-in.  It does not alter released AEGIS execution and it is
not a learned or deployable safety method.  It only supports a deterministic
simulator-in-the-loop feasibility experiment.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


ORACLE_AFFINE_CLOSED_LOOP_CONFIG_SCHEMA = (
    "vlsa_distal_affine_oracle_closed_loop_e05.v1"
)
ORACLE_AFFINE_CLOSED_LOOP_RESULT_SCHEMA = (
    "vlsa_distal_affine_oracle_closed_loop_e05_result.v1"
)
ORACLE_AFFINE_CLOSED_LOOP_VALIDATION_SCHEMA = (
    "vlsa_distal_affine_oracle_closed_loop_e05_validation.v1"
)

CONSTRAINT_ORDER = [
    "L5_part_0", "L5_part_1", "L5_part_2", "L6_part_0",
    "L6_part_1", "L7_part_0", "L7_part_1",
]


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _finite_positive(value: Any, label: str) -> float:
    if isinstance(value, bool) or not math.isfinite(float(value)) or float(value) <= 0:
        raise ValueError("%s must be finite and positive" % label)
    return float(value)


def load_oracle_affine_closed_loop_config(path: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    try:
        config = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("affine-oracle closed-loop config is invalid JSON") from error
    required = {
        "schema_version", "protocol_id", "claim_scope", "primary_case",
        "prerequisite", "constraint_order", "nominal_plan", "activation",
        "sampling", "affine_certificate", "optimizer", "execution",
        "decision_gate",
    }
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError("affine-oracle closed-loop config keys differ")
    if (
        config["schema_version"] != ORACLE_AFFINE_CLOSED_LOOP_CONFIG_SCHEMA
        or config["protocol_id"]
        != "vlsa-distal-affine-oracle-closed-loop-e05-v1"
    ):
        raise ValueError("affine-oracle closed-loop protocol differs")
    if config["primary_case"] != {
        "case_id": "vlsa-t1-goal-ii-t0-e05",
        "expected_action_count": 237,
        "original_first_L5_contact_step": 187,
        "original_first_paper_CAR_step": 188,
        "original_native_task_success_step": 236,
    }:
        raise ValueError("affine-oracle closed-loop case differs")
    prerequisite = config["prerequisite"]
    if set(prerequisite) != {
        "representation_result_file_sha256",
        "representation_validation_file_sha256",
        "representation_result_payload_sha256",
        "require_representation_go",
    } or prerequisite["require_representation_go"] is not True:
        raise ValueError("affine-oracle prerequisite differs")
    for key in (
        "representation_result_file_sha256",
        "representation_validation_file_sha256",
        "representation_result_payload_sha256",
    ):
        if not isinstance(prerequisite[key], str) or len(prerequisite[key]) != 64:
            raise ValueError("affine-oracle prerequisite hash differs")
    if config["constraint_order"] != CONSTRAINT_ORDER:
        raise ValueError("affine-oracle constraint order differs")
    if config["nominal_plan"] != {
        "source": "immutable_successful_released_AEGIS_Table1_env_step_sequence",
        "orientation_gripper_channels": "unchanged",
        "second_action": "next_immutable_AEGIS_action",
        "terminal_horizon": "one_action_when_no_next_action_exists",
    }:
        raise ValueError("affine-oracle nominal plan differs")
    if config["activation"] != {
        "mode": "fresh_exact_cloned_OSC_horizon_check_before_every_execution",
        "horizon_actions": 2,
        "unsafe_if_any_seven_distal_margin_negative": True,
        "unsafe_if_released_AEGIS_EE_proxy_margin_negative": True,
        "unsafe_if_raw_L5_L6_L7_contact": True,
        "maximum_per_step_obstacle_l1_displacement_m": 1.0e-4,
    }:
        raise ValueError("affine-oracle activation differs")
    if config["sampling"] != {
        "vary": "first_action_normalized_XYZ_only",
        "action_limit": 1.0,
        "trust_region_linf_action": 0.5,
        "grid_points_per_dimension": 8,
        "expected_grid_action_count": 512,
        "candidate_order": "cartesian_product_lexicographic",
    }:
        raise ValueError("affine-oracle sampling differs")
    if config["affine_certificate"] != {
        "fit": "candidate_conditioned_minimum_l1_sampled_grid_lower_envelope",
        "one_sided_padding_m": 1.0e-6,
        "target_clearance_m": 0.0,
        "coefficient_postcheck_tolerance_m": 1.0e-8,
        "safe_candidate_requires_all_eight_proxy_and_raw_simulator_safety": True,
    }:
        raise ValueError("affine-oracle certificate differs")
    optimizer = config["optimizer"]
    if set(optimizer) != {
        "bound_tolerance_action", "eps_abs", "eps_rel", "max_iter",
        "residual_tolerance",
    }:
        raise ValueError("affine-oracle optimizer keys differ")
    for key in ("bound_tolerance_action", "eps_abs", "eps_rel", "residual_tolerance"):
        _finite_positive(optimizer[key], "optimizer.%s" % key)
    if isinstance(optimizer["max_iter"], bool) or int(optimizer["max_iter"]) < 1:
        raise ValueError("optimizer.max_iter must be positive")
    if config["execution"] != {
        "execute": "verified_first_action_only_then_replan",
        "osc_internal_substeps": "all",
        "require_all_seven_distal_margins_nonnegative": True,
        "require_released_AEGIS_EE_proxy_nonnegative": True,
        "require_zero_raw_L5_L6_L7_contact": True,
        "require_executed_next_state_hash_equals_clone": True,
        "paper_CAR_threshold_m": 1.0e-3,
        "video": "actual_executed_agentview_MP4_plus_final_JPG",
        "video_fps": 30,
    }:
        raise ValueError("affine-oracle execution differs")
    if config["decision_gate"] != {
        "success": (
            "all_executed_substeps_all_eight_proxy_safe_and_zero_raw_"
            "L5_L6_L7_contact_and_zero_paper_CAR_and_native_task_success"
        ),
        "go_interpretation": (
            "privileged_receding_affine_oracle_can_solve_primary_E05_not_"
            "learned_generalization_not_deployable_safety"
        ),
        "no_go_interpretation": (
            "registered_local_grid_affine_oracle_failed_closed_loop_E05_"
            "not_impossibility_of_other_oracles_or_action_chunk_methods"
        ),
        "neural_training_authorized": False,
    }:
        raise ValueError("affine-oracle decision gate differs")
    output = json.loads(_canonical(config).decode("utf-8"))
    output["config_file_sha256"] = _sha256(raw)
    output["config_payload_sha256"] = _sha256(_canonical(config))
    return output


def summarize_exact_chunk_all_eight(
    chunk: Mapping[str, Any], *, maximum_obstacle_displacement_m: float
) -> dict[str, Any]:
    """Summarize one or two exact cloned OSC transitions without hiding EE."""

    import numpy as np

    transitions = list(chunk.get("transitions", []))
    if len(transitions) not in (1, 2):
        raise ValueError("affine-oracle chunk must contain one or two transitions")
    margins = np.asarray(
        [item["minimum_substep_clearance_m"] for item in transitions],
        dtype=np.float64,
    )
    if margins.shape != (len(transitions), 8) or not np.all(np.isfinite(margins)):
        raise ValueError("affine-oracle chunk clearance shape differs")
    minimum = np.min(margins, axis=0)
    raw_contacts = int(sum(item["raw_protected_contact_count"] for item in transitions))
    obstacle_motion = float(max(
        item["maximum_within_step_obstacle_l1_displacement_m"]
        for item in transitions
    ))
    distal_safe = bool(np.all(minimum[:7] >= 0.0))
    ee_safe = bool(minimum[7] >= 0.0)
    raw_safe = bool(
        raw_contacts == 0 and obstacle_motion <= float(maximum_obstacle_displacement_m)
    )
    return {
        "action_count": len(transitions),
        "minimum_all_eight_substep_clearance_m": minimum.tolist(),
        "minimum_distal_substep_clearance_m": float(np.min(minimum[:7])),
        "minimum_released_AEGIS_EE_proxy_substep_clearance_m": float(minimum[7]),
        "D_opt_seven_distal_safe": distal_safe,
        "released_AEGIS_EE_proxy_safe": ee_safe,
        "D_sim_raw_safe": raw_safe,
        "raw_protected_contact_count": raw_contacts,
        "maximum_within_step_obstacle_l1_displacement_m": obstacle_motion,
        "safe_for_execution": bool(distal_safe and ee_safe and raw_safe),
        "first_transition_next_state_sha256": str(
            transitions[0]["next_state_sha256"]
        ),
        "final_next_state_sha256": str(transitions[-1]["next_state_sha256"]),
        "env_step_wall_seconds": float(sum(
            item["env_step_wall_seconds"] for item in transitions
        )),
    }
