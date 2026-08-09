"""Contracts for privileged waypoint Best-of-N selection on primary E05.

The module is opt-in and contains no simulator dependency.  It freezes the
small Cartesian chunk library and deterministic safe-candidate selector used
by the H100 feasibility diagnostic.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


WAYPOINT_CONFIG_SCHEMA = "vlsa_distal_waypoint_closed_loop_e05.v1"
WAYPOINT_RESULT_SCHEMA = "vlsa_distal_waypoint_closed_loop_e05_result.v1"
WAYPOINT_VALIDATION_SCHEMA = (
    "vlsa_distal_waypoint_closed_loop_e05_validation.v1"
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


def load_waypoint_config(path: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    try:
        config = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("waypoint config is invalid JSON") from error
    required = {
        "schema_version", "protocol_id", "claim_scope", "primary_case",
        "prerequisite", "constraint_order", "nominal_plan", "activation",
        "waypoint_search", "candidate_selector", "execution",
        "decision_gate",
    }
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError("waypoint config keys differ")
    if (
        config["schema_version"] != WAYPOINT_CONFIG_SCHEMA
        or config["protocol_id"]
        != "vlsa-distal-waypoint-closed-loop-e05-v1"
    ):
        raise ValueError("waypoint protocol differs")
    if config["primary_case"] != {
        "case_id": "vlsa-t1-goal-ii-t0-e05",
        "expected_action_count": 237,
        "original_first_L5_contact_step": 187,
        "original_first_paper_CAR_step": 188,
        "original_native_task_success_step": 236,
    }:
        raise ValueError("waypoint primary case differs")
    prerequisite = config["prerequisite"]
    if set(prerequisite) != {
        "exact_candidate_result_file_sha256",
        "exact_candidate_result_payload_sha256",
        "exact_candidate_validation_file_sha256",
        "require_observed_empty_one_step_safe_set",
    } or prerequisite["require_observed_empty_one_step_safe_set"] is not True:
        raise ValueError("waypoint prerequisite differs")
    for key in (
        "exact_candidate_result_file_sha256",
        "exact_candidate_result_payload_sha256",
        "exact_candidate_validation_file_sha256",
    ):
        if not isinstance(prerequisite[key], str) or len(prerequisite[key]) != 64:
            raise ValueError("waypoint prerequisite hash differs")
    if config["constraint_order"] != CONSTRAINT_ORDER:
        raise ValueError("waypoint constraint order differs")
    if config["nominal_plan"] != {
        "source": "immutable_successful_released_AEGIS_Table1_env_step_sequence",
        "orientation_gripper_channels": "unchanged_per_nominal_chunk_step",
        "task_waypoint": (
            "future_end_effector_position_from_parallel_immutable_AEGIS_replay"
        ),
        "terminal_horizon": "truncate_chunk_at_last_immutable_action",
    }:
        raise ValueError("waypoint nominal plan differs")
    if config["activation"] != {
        "mode": "fresh_exact_cloned_OSC_two_action_check_before_every_execution",
        "nominal_horizon_actions": 2,
        "unsafe_if_any_eight_margin_negative": True,
        "unsafe_if_raw_L5_L6_L7_contact": True,
        "maximum_per_step_obstacle_l1_displacement_m": 1.0e-4,
    }:
        raise ValueError("waypoint activation differs")
    if config["waypoint_search"] != {
        "horizon_actions": 4,
        "action_limit": 1.0,
        "lattice_values": [-1.0, 0.0, 1.0],
        "constant_full_lattice": True,
        "two_phase_primitives": [
            [0.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0], [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0], [0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0], [0.0, 0.0, 1.0],
        ],
        "diversion_prefix_lengths": [1, 2],
        "candidate_order": (
            "nominal_then_constant_lattice_then_two_phase_then_diversion"
        ),
        "expected_candidate_count_by_horizon": {
            "1": 28, "2": 97, "3": 124, "4": 124,
        },
    }:
        raise ValueError("waypoint search differs")
    if config["candidate_selector"] != {
        "method": (
            "exact_safe_then_task_completion_then_minimum_future_EE_waypoint_"
            "error_then_minimum_chunk_correction_L2_then_candidate_index"
        ),
        "minimum_all_eight_margin_m": 1.0e-6,
        "require_raw_L5_L6_L7_contact_free": True,
        "require_obstacle_motion_gate": True,
        "fresh_exact_rollout_before_execution": True,
        "QP_used": False,
        "learned_model_used": False,
        "fallback": "none_stop_if_no_exact_safe_waypoint_chunk",
    }:
        raise ValueError("waypoint selector differs")
    if config["execution"] != {
        "execute": "verified_first_action_only_then_replan",
        "osc_internal_substeps": "all",
        "require_all_eight_margins_nonnegative": True,
        "require_zero_raw_L5_L6_L7_contact": True,
        "require_executed_next_state_hash_equals_clone": True,
        "paper_CAR_threshold_m": 1.0e-3,
    }:
        raise ValueError("waypoint execution differs")
    if config["decision_gate"] != {
        "success": (
            "all_executed_substeps_all_eight_proxy_safe_and_zero_raw_"
            "L5_L6_L7_contact_and_zero_paper_CAR_and_native_task_success"
        ),
        "go_interpretation": (
            "privileged_receding_waypoint_Best_of_N_can_solve_primary_E05_"
            "not_QP_not_learned_not_deployable_safety"
        ),
        "no_go_interpretation": (
            "registered_four_action_waypoint_library_failed_closed_loop_E05_"
            "not_impossibility_of_live_VLA_or_joint_space_recovery"
        ),
        "neural_training_authorized": False,
    }:
        raise ValueError("waypoint decision gate differs")
    output = json.loads(_canonical(config).decode("utf-8"))
    output["config_file_sha256"] = _sha256(raw)
    output["config_payload_sha256"] = _sha256(_canonical(config))
    return output


def waypoint_chunks(
    nominal_chunk: Sequence[Sequence[float]], config: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Generate the frozen, deduplicated Cartesian waypoint chunk library."""

    import numpy as np

    nominal = np.asarray(nominal_chunk, dtype=np.float64)
    horizon = int(nominal.shape[0]) if nominal.ndim == 2 else -1
    if nominal.shape != (horizon, 7) or horizon < 1 or horizon > 4:
        raise ValueError("waypoint nominal chunk differs")
    if not np.all(np.isfinite(nominal)):
        raise ValueError("waypoint nominal chunk is not finite")
    settings = config["waypoint_search"]
    limit = float(settings["action_limit"])
    if np.max(np.abs(nominal[:, :3])) > limit + 1.0e-12:
        raise ValueError("waypoint nominal translation exceeds action limits")
    lattice = [
        np.asarray(value, dtype=np.float64)
        for value in itertools.product(settings["lattice_values"], repeat=3)
    ]
    candidates: list[tuple[str, Any]] = [("nominal", nominal.copy())]
    for direction in lattice:
        chunk = nominal.copy()
        chunk[:, :3] = direction
        candidates.append(("constant_full_lattice", chunk))
    if horizon >= 2:
        first_count = horizon // 2
        primitives = [
            np.asarray(item, dtype=np.float64)
            for item in settings["two_phase_primitives"]
        ]
        for first, second in itertools.product(primitives, repeat=2):
            chunk = nominal.copy()
            chunk[:first_count, :3] = first
            chunk[first_count:, :3] = second
            candidates.append(("two_phase_cardinal", chunk))
        for prefix_length in settings["diversion_prefix_lengths"]:
            count = int(prefix_length)
            if count >= horizon:
                continue
            for direction in lattice:
                chunk = nominal.copy()
                chunk[:count, :3] = direction
                candidates.append(("diversion_%d_then_nominal" % count, chunk))
    seen = set()
    output = []
    for source, chunk in candidates:
        array = np.asarray(chunk, dtype=np.float64)
        key = array[:, :3].tobytes()
        if key in seen:
            continue
        seen.add(key)
        correction = array[:, :3] - nominal[:, :3]
        output.append({
            "source": source,
            "chunk": array,
            "correction_l2": float(np.linalg.norm(correction)),
        })
    expected = int(
        settings["expected_candidate_count_by_horizon"][str(horizon)]
    )
    if len(output) != expected:
        raise ValueError(
            "waypoint candidate count differs: %d != %d" % (len(output), expected)
        )
    return output


def select_waypoint_candidate(
    candidate_records: Sequence[Mapping[str, Any]],
    *,
    minimum_margin_m: float,
) -> dict[str, Any]:
    """Select the registered closest-to-task exactly safe waypoint chunk."""

    threshold = float(minimum_margin_m)
    if not math.isfinite(threshold) or threshold < 0.0:
        raise ValueError("waypoint minimum margin is invalid")
    eligible = []
    for index, record in enumerate(candidate_records):
        margins = [float(item) for item in record["minimum_all_eight_margin_m"]]
        if len(margins) != 8 or not all(math.isfinite(item) for item in margins):
            raise ValueError("waypoint candidate margins differ")
        if (
            min(margins) >= threshold
            and record.get("raw_safe") is True
            and math.isfinite(float(record["terminal_eef_reference_error_m"]))
            and math.isfinite(float(record["chunk_correction_l2"]))
        ):
            eligible.append(index)
    if not eligible:
        return {
            "valid": False,
            "reason": "no_exact_all_eight_safe_waypoint_chunk",
            "eligible_candidate_count": 0,
            "selected_candidate_index": None,
        }
    selected = min(
        eligible,
        key=lambda index: (
            0 if candidate_records[index].get("task_success_in_chunk") else 1,
            float(candidate_records[index]["terminal_eef_reference_error_m"]),
            float(candidate_records[index]["chunk_correction_l2"]),
            index,
        ),
    )
    record = candidate_records[selected]
    return {
        "valid": True,
        "reason": "exact_safe_waypoint_chunk_selected",
        "eligible_candidate_count": len(eligible),
        "selected_candidate_index": int(selected),
        "selected_source": str(record["source"]),
        "selected_chunk": record["chunk"],
        "selected_minimum_all_eight_margin_m": list(
            record["minimum_all_eight_margin_m"]
        ),
        "selected_task_success_in_chunk": bool(
            record.get("task_success_in_chunk")
        ),
        "selected_terminal_eef_reference_error_m": float(
            record["terminal_eef_reference_error_m"]
        ),
        "selected_chunk_correction_l2": float(record["chunk_correction_l2"]),
    }
