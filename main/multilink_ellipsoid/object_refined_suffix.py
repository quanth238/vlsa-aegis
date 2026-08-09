"""Contracts for the object-aware continuous-refinement E05 suffix oracle."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


CONFIG_SCHEMA = "vlsa_distal_object_refined_suffix_e05.v1"
RESULT_SCHEMA = "vlsa_distal_object_refined_suffix_e05_result.v1"
VALIDATION_SCHEMA = "vlsa_distal_object_refined_suffix_e05_validation.v1"


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def load_config(path: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    try:
        config = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("object-refined config is invalid JSON") from error
    required = {
        "schema_version", "protocol_id", "claim_scope", "primary_case",
        "prerequisite", "activation", "search", "selector", "execution",
        "decision_gate",
    }
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError("object-refined config keys differ")
    if config["schema_version"] != CONFIG_SCHEMA:
        raise ValueError("object-refined config schema differs")
    if config["protocol_id"] != "vlsa-distal-object-refined-suffix-e05-v1":
        raise ValueError("object-refined protocol differs")
    if config["primary_case"] != {
        "case_id": "vlsa-t1-goal-ii-t0-e05",
        "expected_action_count": 237,
        "suffix_start_step": 230,
        "original_native_task_success_step": 236,
    }:
        raise ValueError("object-refined primary case differs")
    prerequisite = config["prerequisite"]
    if set(prerequisite) != {
        "waypoint_result_file_sha256", "waypoint_result_payload_sha256",
        "waypoint_validation_file_sha256", "require_exact_safe_prefix",
    } or prerequisite["require_exact_safe_prefix"] is not True:
        raise ValueError("object-refined prerequisite differs")
    for key in (
        "waypoint_result_file_sha256", "waypoint_result_payload_sha256",
        "waypoint_validation_file_sha256",
    ):
        if not isinstance(prerequisite[key], str) or len(prerequisite[key]) != 64:
            raise ValueError("object-refined prerequisite hash differs")
    if config["activation"] != {
        "nominal_exact_horizon_actions": 2,
        "intervene_if_any_eight_margin_below_m": 1.0e-6,
        "intervene_on_raw_L5_L6_L7_contact": True,
        "maximum_per_step_obstacle_l1_displacement_m": 1.0e-4,
    }:
        raise ValueError("object-refined activation differs")
    if config["search"] != {
        "coarse_library": "registered_four_action_waypoint_chunks",
        "horizon_actions": 4,
        "local_refinement_first_action_translation_only": True,
        "top_k_safe_anchors": 4,
        "refinement_magnitudes": [0.25, 0.1],
        "refinement_directions": "all_nonzero_cartesian_ternary_directions",
        "translation_action_bounds": [-1.0, 1.0],
        "deduplicate_full_translation_chunk": True,
    }:
        raise ValueError("object-refined search differs")
    if config["selector"] != {
        "ranking": [
            "exact_all_eight_safe", "native_task_success_in_chunk",
            "minimum_terminal_bowl_relative_to_plate_reference_error_m",
            "minimum_terminal_EE_reference_error_m",
            "minimum_chunk_correction_L2", "candidate_index",
        ],
        "object_reference": (
            "parallel_immutable_successful_AEGIS_replay_same_future_step"
        ),
        "QP_used": False,
        "learned_model_used": False,
    }:
        raise ValueError("object-refined selector differs")
    if config["execution"] != {
        "prefix": "exact_replay_of_validated_waypoint_actions_0_through_229",
        "suffix": "fresh_verified_first_action_only_then_replan",
        "require_executed_next_state_hash_equals_clone": True,
        "require_zero_raw_L5_L6_L7_contact": True,
        "require_all_eight_margins_nonnegative": True,
        "paper_CAR_threshold_m": 1.0e-3,
    }:
        raise ValueError("object-refined execution differs")
    if config["decision_gate"] != {
        "go": (
            "exact_safe_prefix_and_suffix_zero_contact_zero_CAR_and_native_"
            "task_success"
        ),
        "interpretation": (
            "privileged_local_object_aware_safe_suffix_exists_for_primary_E05"
        ),
        "not_claimed": [
            "deployable_method", "QP_success", "learned_model_success",
            "whole_body_safety",
        ],
        "neural_training_authorized": False,
    }:
        raise ValueError("object-refined decision gate differs")
    output = json.loads(_canonical(config).decode("utf-8"))
    output["config_file_sha256"] = hashlib.sha256(raw).hexdigest()
    output["config_payload_sha256"] = hashlib.sha256(_canonical(config)).hexdigest()
    return output


def candidate_rank(record: Mapping[str, Any], index: int) -> tuple[Any, ...]:
    return (
        0 if record.get("task_success_in_chunk") is True else 1,
        float(record["terminal_object_reference_error_m"]),
        float(record["terminal_eef_reference_error_m"]),
        float(record["chunk_correction_l2"]),
        int(index),
    )


def eligible_indices(
    records: Sequence[Mapping[str, Any]], minimum_margin_m: float,
) -> list[int]:
    threshold = float(minimum_margin_m)
    if not math.isfinite(threshold) or threshold < 0.0:
        raise ValueError("minimum margin differs")
    output = []
    for index, record in enumerate(records):
        margins = [float(value) for value in record["minimum_all_eight_margin_m"]]
        if len(margins) != 8 or not all(math.isfinite(value) for value in margins):
            raise ValueError("candidate margins differ")
        scores = (
            float(record["terminal_object_reference_error_m"]),
            float(record["terminal_eef_reference_error_m"]),
            float(record["chunk_correction_l2"]),
        )
        if (
            min(margins) >= threshold
            and record.get("raw_safe") is True
            and all(math.isfinite(value) for value in scores)
        ):
            output.append(index)
    return output


def select_candidate(
    records: Sequence[Mapping[str, Any]], minimum_margin_m: float,
) -> dict[str, Any]:
    eligible = eligible_indices(records, minimum_margin_m)
    if not eligible:
        return {
            "valid": False,
            "reason": "no_exact_all_eight_safe_object_refined_chunk",
            "eligible_candidate_count": 0,
            "selected_candidate_index": None,
        }
    selected = min(eligible, key=lambda index: candidate_rank(records[index], index))
    record = records[selected]
    return {
        "valid": True,
        "reason": "exact_safe_object_refined_chunk_selected",
        "eligible_candidate_count": len(eligible),
        "selected_candidate_index": int(selected),
        "selected_source": str(record["source"]),
        "selected_chunk": record["chunk"],
        "selected_task_success_in_chunk": bool(record["task_success_in_chunk"]),
        "selected_terminal_object_reference_error_m": float(
            record["terminal_object_reference_error_m"]
        ),
        "selected_terminal_eef_reference_error_m": float(
            record["terminal_eef_reference_error_m"]
        ),
        "selected_chunk_correction_l2": float(record["chunk_correction_l2"]),
    }


def refinement_chunks(
    anchors: Sequence[Sequence[Sequence[float]]], magnitude: float,
) -> list[Any]:
    """Perturb only the first Cartesian translation of each anchor."""

    import numpy as np

    value = float(magnitude)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("refinement magnitude differs")
    directions = [
        np.asarray(item, dtype=np.float64)
        for item in itertools.product((-1.0, 0.0, 1.0), repeat=3)
        if item != (0.0, 0.0, 0.0)
    ]
    output = []
    seen = set()
    for anchor in anchors:
        source = np.asarray(anchor, dtype=np.float64)
        if source.ndim != 2 or source.shape[1] != 7:
            raise ValueError("refinement anchor shape differs")
        for direction in directions:
            chunk = source.copy()
            chunk[0, :3] = np.clip(chunk[0, :3] + value * direction, -1.0, 1.0)
            key = chunk[:, :3].tobytes()
            if key in seen or np.array_equal(chunk[:, :3], source[:, :3]):
                continue
            seen.add(key)
            output.append(chunk)
    return output
