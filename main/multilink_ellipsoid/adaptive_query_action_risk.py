"""Pure contracts for adaptive post-AEGIS L5 boundary collection."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple


CONFIG_SCHEMA = "vlsa_distal_adaptive_query_action_risk.v1"
RESULT_SCHEMA = "vlsa_distal_adaptive_query_action_risk_result.v1"


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def load_config(path: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    value = json.loads(raw)
    required = {
        "schema_version", "protocol_id", "claim_scope", "cohort",
        "screening", "authoritative_labels", "feature_audit", "forbidden",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("adaptive query-risk config keys differ")
    if value["schema_version"] != CONFIG_SCHEMA:
        raise ValueError("adaptive query-risk config schema differs")
    if value["protocol_id"] != "vlsa-distal-adaptive-query-action-risk-v1":
        raise ValueError("adaptive query-risk protocol differs")
    if value["cohort"] != {
        "source_episode_manifest": "manifests/vlsa_distal_clean_action_risk.v1.jsonl",
        "available_clean_episode_groups": 18,
        "development_episode_groups": 15,
        "sealed_test_episode_groups": 3,
        "split_unit": "complete_episode",
        "note": (
            "The preregistered clean cohort contains 18 groups; known E38 proxy "
            "mismatch and gripper-contact cases remain excluded rather than used "
            "to manufacture 20 groups."
        ),
    }:
        raise ValueError("adaptive cohort differs")
    screening = value["screening"]
    if screening != {
        "coarse_candidate_count": 9,
        "coarse_candidates": [
            "nominal",
            "normal_pos_front_loaded_r1.0",
            "normal_neg_front_loaded_r1.0",
            "tangent_up_pos_front_loaded_r1.0",
            "tangent_up_neg_front_loaded_r1.0",
            "tangent_side_pos_front_loaded_r1.0",
            "tangent_side_neg_front_loaded_r1.0",
            "normal_pos_front_loaded_r2.0",
            "normal_neg_front_loaded_r2.0",
        ],
        "risk_rows": [0, 1, 2],
        "risk_scalar": "maximum_L5_prefix_risk",
        "safe_threshold_m": 0.0,
        "unsafe_threshold_m": 0.0,
        "physical_veto_is_unsafe": True,
        "stop_prefix_after_contact_or_CAR": True,
        "bracket_selection": "minimum_action_L2_then_boundary_distance_then_order",
        "bisection_iterations": 5,
        "retain_nominal": True,
        "maximum_authoritative_candidates": 8,
    }:
        raise ValueError("adaptive screening protocol differs")
    labels = value["authoritative_labels"]
    if labels != {
        "candidate": "exact_final_post_AEGIS_five_action_chunk",
        "rollout": "candidate_prefix_plus_complete_fixed_backup",
        "complete_backup_only_for_retained_candidates": True,
        "unknown_timeout_is_censored": True,
        "original_AEGIS_EE_QP_recorded": True,
        "L6_L7_contact_is_diagnostic": True,
    }:
        raise ValueError("adaptive authoritative-label protocol differs")
    if value["feature_audit"] != {
        "models": [
            "existing_86D", "compact_exact_action",
            "compact_exact_action_plus_complete_physical_state",
        ],
        "record": [
            "exact_final_post_AEGIS_action", "arm_q_and_dq",
            "L5_obstacle_relative_geometry", "current_clearances_and_normals",
            "EE_and_OSC_state", "AEGIS_EE_QP_records",
        ],
        "coverage_gate": {
            "minimum_train_known_candidates": 40,
            "minimum_validation_known_candidates": 12,
            "minimum_known_safe_candidates_per_L5_row": 10,
            "minimum_known_unsafe_candidates_per_L5_row": 10,
            "minimum_near_boundary_candidates_per_L5_row": 8,
            "minimum_train_useful_boundary_states_per_L5_row": 3,
            "minimum_validation_useful_boundary_states_per_L5_row": 1,
        },
        "no_training_in_collection_gate": True,
    }:
        raise ValueError("adaptive feature-audit protocol differs")
    if value["forbidden"] != {
        "Table1_artifact_change": True,
        "test_episode_opening": True,
        "model_training": True,
        "calibration": True,
        "QP": True,
        "closed_loop": True,
        "CBF_claim": True,
    }:
        raise ValueError("adaptive forbidden set differs")
    output = json.loads(canonical(value).decode("utf-8"))
    output["config_file_sha256"] = hashlib.sha256(raw).hexdigest()
    output["config_payload_sha256"] = hashlib.sha256(canonical(value)).hexdigest()
    return output


def coarse_candidate_definitions(
    nominal_actions: Sequence[Sequence[float]],
    local_frame: Mapping[str, Sequence[float]],
    base_config: Mapping[str, Any],
    adaptive_config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    from .query_action_risk import candidate_definitions

    bank = candidate_definitions(nominal_actions, local_frame, base_config)
    by_name = {record["name"]: record for record in bank}
    names = adaptive_config["screening"]["coarse_candidates"]
    if any(name not in by_name for name in names):
        raise ValueError("adaptive coarse candidate is absent from frozen bank")
    output = [{**by_name[name], "coarse_order": order}
              for order, name in enumerate(names)]
    if len(output) != int(adaptive_config["screening"]["coarse_candidate_count"]):
        raise ValueError("adaptive coarse candidate count differs")
    return output


def select_bracket(
    records: Sequence[Mapping[str, Any]],
) -> Optional[Tuple[int, int]]:
    """Return the closest safe/unsafe screening pair without hiding modes."""
    safe = [index for index, row in enumerate(records) if row["screen_safe"]]
    unsafe = [index for index, row in enumerate(records) if row["screen_unsafe"]]
    if not safe or not unsafe:
        return None
    ranked = []
    for safe_index in safe:
        left = records[safe_index]["definition"]["actions"]
        for unsafe_index in unsafe:
            right = records[unsafe_index]["definition"]["actions"]
            if (len(left) != 5 or len(right) != 5
                    or any(len(row) != 7 for row in list(left) + list(right))):
                raise ValueError("adaptive bracket action shape differs")
            distance = math.sqrt(sum(
                (float(a) - float(b)) ** 2
                for left_row, right_row in zip(left, right)
                for a, b in zip(left_row, right_row)
            ))
            ranked.append((
                distance,
                abs(float(records[safe_index]["L5_prefix_risk_m"]))
                + abs(float(records[unsafe_index]["L5_prefix_risk_m"])),
                int(records[safe_index]["definition"]["order"]),
                int(records[unsafe_index]["definition"]["order"]),
                safe_index,
                unsafe_index,
            ))
    best = min(ranked)
    return int(best[-2]), int(best[-1])


def midpoint_definition(
    safe_definition: Mapping[str, Any], unsafe_definition: Mapping[str, Any],
    *, iteration: int, order: int,
) -> dict[str, Any]:
    safe = safe_definition["actions"]
    unsafe = unsafe_definition["actions"]
    if (len(safe) != 5 or len(unsafe) != 5
            or any(len(row) != 7 for row in list(safe) + list(unsafe))):
        raise ValueError("adaptive bisection action shape differs")
    actions = [
        [0.5 * (float(a) + float(b)) for a, b in zip(left, right)]
        for left, right in zip(safe, unsafe)
    ]
    return {
        "name": "adaptive_bisection_%02d" % int(iteration),
        "order": int(order),
        "direction": None,
        "sign": 0,
        "temporal_profile": "safe_unsafe_action_segment",
        "requested_correction_l2_action": None,
        "applied_correction_l2_action": None,
        "clipped": False,
        "bisection_iteration": int(iteration),
        "safe_parent": str(safe_definition["name"]),
        "unsafe_parent": str(unsafe_definition["name"]),
        "actions": actions,
    }
