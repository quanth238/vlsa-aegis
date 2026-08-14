"""Per-row contracts for principled adaptive L5 boundary collection."""

from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

from .query_action_risk import temporal_profile


CONFIG_SCHEMA = "vlsa_distal_adaptive_query_action_risk.v2"
RESULT_SCHEMA = "vlsa_distal_adaptive_query_action_risk_result.v2"


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
        raise ValueError("adaptive v2 config keys differ")
    if value["schema_version"] != CONFIG_SCHEMA:
        raise ValueError("adaptive v2 config schema differs")
    if value["protocol_id"] != "vlsa-distal-adaptive-query-action-risk-v2":
        raise ValueError("adaptive v2 protocol differs")
    cohort = value["cohort"]
    if cohort != {
        "source_episode_manifest": "manifests/vlsa_distal_clean_action_risk.v1.jsonl",
        "available_clean_episode_groups": 18,
        "development_episode_groups": 15,
        "sealed_test_episode_groups": 3,
        "split_unit": "complete_episode",
        "mechanism_pilot": True,
    }:
        raise ValueError("adaptive v2 cohort differs")
    screening = value["screening"]
    if set(screening) != {
        "coarse_candidate_count", "mixture_seed", "candidate_recipes",
        "risk_rows", "two_sided_epsilon_m", "target_row", "safe_endpoint",
        "unsafe_endpoint", "coarse_stop_prefix_after_contact_or_CAR",
        "bisection_executes_complete_prefix", "bracket_selection",
        "bisection_iterations", "retain_nominal",
        "maximum_authoritative_candidates",
    }:
        raise ValueError("adaptive v2 screening keys differ")
    if (
        int(screening["coarse_candidate_count"]) != 12
        or int(screening["mixture_seed"]) != 20260814
        or len(screening["candidate_recipes"]) != 12
        or screening["risk_rows"] != [0, 1, 2]
        or float(screening["two_sided_epsilon_m"]) != 0.0005
        or int(screening["bisection_iterations"]) != 5
        or int(screening["maximum_authoritative_candidates"]) != 8
        or screening["coarse_stop_prefix_after_contact_or_CAR"] is not True
        or screening["bisection_executes_complete_prefix"] is not True
        or screening["retain_nominal"] is not True
        or screening["target_row"]
        != "highest_nominal_risk_among_rows_with_global_safe_and_row_positive_coarse_support"
        or screening["safe_endpoint"]
        != "globally_L5_safe_no_physical_veto_and_target_row_below_negative_epsilon"
        or screening["unsafe_endpoint"] != "target_row_above_positive_epsilon"
        or screening["bracket_selection"]
        != "minimum_action_L2_then_target_boundary_distance_then_order"
    ):
        raise ValueError("adaptive v2 screening protocol differs")
    recipes = screening["candidate_recipes"]
    if recipes[0] != {
        "name": "nominal", "source": "nominal", "coefficients": None,
        "temporal_profile": None, "radius": 0.0,
    }:
        raise ValueError("adaptive v2 nominal recipe differs")
    if len({str(item["name"]) for item in recipes}) != len(recipes):
        raise ValueError("adaptive v2 recipe names are not unique")
    for recipe in recipes[1:]:
        if (
            recipe["source"] not in ("axis", "fixed_seed_mixture")
            or len(recipe["coefficients"]) != 3
            or recipe["temporal_profile"] not in ("constant", "front_loaded")
            or float(recipe["radius"]) <= 0.0
        ):
            raise ValueError("adaptive v2 candidate recipe differs")
    generator = random.Random(int(screening["mixture_seed"]))
    generated = [
        [generator.uniform(0.25, 1.0), generator.uniform(-1.0, 1.0),
         generator.uniform(-1.0, 1.0)]
        for _ in range(3)
    ]
    observed = [
        [float(value) for value in recipe["coefficients"]]
        for recipe in recipes if recipe["source"] == "fixed_seed_mixture"
    ]
    if observed != generated:
        raise ValueError("adaptive v2 fixed-seed mixtures differ")
    if value["authoritative_labels"] != {
        "candidate": "exact_final_post_AEGIS_five_action_chunk",
        "rollout": "candidate_prefix_plus_complete_fixed_backup_unless_physical_veto",
        "complete_backup_only_for_retained_candidates": True,
        "unknown_timeout_is_censored": True,
        "original_AEGIS_EE_QP_recorded": True,
        "all_three_L5_rows_recorded": True,
        "L6_L7_contact_is_diagnostic": True,
    }:
        raise ValueError("adaptive v2 authoritative labels differ")
    feature = value["feature_audit"]
    if (
        feature.get("models") != [
            "existing_86D", "compact_exact_action",
            "compact_exact_action_plus_complete_physical_state",
        ]
        or feature.get("record") != [
            "exact_final_post_AEGIS_action", "arm_q_and_dq",
            "L5_obstacle_relative_geometry", "current_clearances_and_normals",
            "EE_and_OSC_state", "AEGIS_EE_QP_records",
        ]
        or feature.get("coverage_gate") != {
            "minimum_train_known_candidates": 40,
            "minimum_validation_known_candidates": 12,
            "minimum_known_safe_candidates_per_L5_row": 10,
            "minimum_known_unsafe_candidates_per_L5_row": 10,
            "minimum_near_boundary_candidates_per_L5_row": 8,
            "minimum_train_useful_boundary_states_per_L5_row": 3,
            "minimum_validation_useful_boundary_states_per_L5_row": 1,
        }
        or feature.get("no_training_in_collection_gate") is not True
    ):
        raise ValueError("adaptive v2 feature audit differs")
    if value["forbidden"] != {
        "Table1_artifact_change": True,
        "test_episode_opening": True,
        "model_training": True,
        "calibration": True,
        "QP": True,
        "closed_loop": True,
        "CBF_claim": True,
    }:
        raise ValueError("adaptive v2 forbidden set differs")
    output = json.loads(canonical(value).decode("utf-8"))
    output["config_file_sha256"] = hashlib.sha256(raw).hexdigest()
    output["config_payload_sha256"] = hashlib.sha256(canonical(value)).hexdigest()
    return output


def _unit(vector: Sequence[float]) -> list[float]:
    values = [float(item) for item in vector]
    if len(values) != 3 or any(not math.isfinite(item) for item in values):
        raise ValueError("adaptive v2 direction differs")
    norm = math.sqrt(sum(item * item for item in values))
    if norm <= 0.0:
        raise ValueError("adaptive v2 direction is degenerate")
    return [item / norm for item in values]


def coarse_candidate_definitions(
    nominal_actions: Sequence[Sequence[float]],
    local_frame: Mapping[str, Sequence[float]],
    base_config: Mapping[str, Any],
    adaptive_config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    nominal = [[float(item) for item in row] for row in nominal_actions]
    if len(nominal) != 5 or any(len(row) != 7 for row in nominal):
        raise ValueError("adaptive v2 nominal action shape differs")
    basis = [
        _unit(local_frame["normal"]),
        _unit(local_frame["tangent_up"]),
        _unit(local_frame["tangent_side"]),
    ]
    limit = float(base_config["candidate_family"]["action_limit"])
    output = []
    for order, recipe in enumerate(adaptive_config["screening"]["candidate_recipes"]):
        actions = [list(row) for row in nominal]
        coefficients = recipe["coefficients"]
        if coefficients is None:
            direction = None
            applied_norm = 0.0
            clipped = False
        else:
            combined = [
                sum(float(coefficients[index]) * basis[index][axis]
                    for index in range(3))
                for axis in range(3)
            ]
            direction = _unit(combined)
            profile = temporal_profile(str(recipe["temporal_profile"]))
            radius = float(recipe["radius"])
            applied_sq = 0.0
            clipped = False
            for time_index in range(5):
                for axis in range(3):
                    requested = profile[time_index] * direction[axis] * radius
                    value = min(limit, max(-limit, nominal[time_index][axis] + requested))
                    applied = value - nominal[time_index][axis]
                    applied_sq += applied * applied
                    clipped = clipped or abs(applied - requested) > 1.0e-12
                    actions[time_index][axis] = value
            applied_norm = math.sqrt(applied_sq)
        nonzero = [] if coefficients is None else [
            float(value) for value in coefficients if float(value) != 0.0
        ]
        sign = (
            0 if coefficients is None or len(nonzero) != 1
            else 1 if nonzero[0] > 0.0 else -1
        )
        output.append({
            "name": str(recipe["name"]),
            "order": order,
            "coarse_order": order,
            "direction": direction,
            "direction_source": str(recipe["source"]),
            "direction_coefficients": coefficients,
            "mixture_seed": (
                int(adaptive_config["screening"]["mixture_seed"])
                if recipe["source"] == "fixed_seed_mixture" else None
            ),
            "sign": sign,
            "temporal_profile": recipe["temporal_profile"],
            "requested_correction_l2_action": float(recipe["radius"]),
            "applied_correction_l2_action": applied_norm,
            "clipped": clipped,
            "actions": actions,
        })
    if len(output) != int(adaptive_config["screening"]["coarse_candidate_count"]):
        raise ValueError("adaptive v2 candidate count differs")
    return output


def row_support(
    records: Sequence[Mapping[str, Any]], epsilon_m: float,
) -> list[dict[str, Any]]:
    epsilon = float(epsilon_m)
    output = []
    globally_safe = [
        index for index, record in enumerate(records)
        if not bool(record["physical_veto"])
        and max(float(value) for value in record["prefix_risk"][:3]) < -epsilon
    ]
    for row in range(3):
        negative = [
            index for index, record in enumerate(records)
            if not bool(record["physical_veto"])
            and float(record["prefix_risk"][row]) < -epsilon
        ]
        positive = [
            index for index, record in enumerate(records)
            if float(record["prefix_risk"][row]) > epsilon
        ]
        safe_endpoints = sorted(set(negative).intersection(globally_safe))
        output.append({
            "row": row,
            "minimum_observed_prefix_risk_m": min(
                float(record["prefix_risk"][row]) for record in records
            ),
            "maximum_observed_prefix_risk_m": max(
                float(record["prefix_risk"][row]) for record in records
            ),
            "row_negative_indices": negative,
            "row_positive_indices": positive,
            "global_safe_endpoint_indices": safe_endpoints,
            "two_sided_observed": bool(negative and positive),
            "controllable_with_global_safe_endpoint": bool(
                safe_endpoints and positive
            ),
        })
    return output


def choose_target_row(
    records: Sequence[Mapping[str, Any]], support: Sequence[Mapping[str, Any]],
) -> Optional[int]:
    if not records:
        return None
    nominal = records[0]
    eligible = [
        int(item["row"]) for item in support
        if bool(item["controllable_with_global_safe_endpoint"])
    ]
    if not eligible:
        return None
    return max(eligible, key=lambda row: (float(nominal["prefix_risk"][row]), -row))


def select_row_bracket(
    records: Sequence[Mapping[str, Any]], *, row: int, epsilon_m: float,
) -> Optional[Tuple[int, int]]:
    epsilon = float(epsilon_m)
    safe = [
        index for index, record in enumerate(records)
        if not bool(record["physical_veto"])
        and max(float(value) for value in record["prefix_risk"][:3]) < -epsilon
        and float(record["prefix_risk"][row]) < -epsilon
    ]
    unsafe = [
        index for index, record in enumerate(records)
        if float(record["prefix_risk"][row]) > epsilon
    ]
    if not safe or not unsafe:
        return None
    ranked = []
    for safe_index in safe:
        left = records[safe_index]["definition"]["actions"]
        for unsafe_index in unsafe:
            right = records[unsafe_index]["definition"]["actions"]
            distance = math.sqrt(sum(
                (float(a) - float(b)) ** 2
                for left_row, right_row in zip(left, right)
                for a, b in zip(left_row, right_row)
            ))
            ranked.append((
                distance,
                abs(float(records[safe_index]["prefix_risk"][row]))
                + abs(float(records[unsafe_index]["prefix_risk"][row])),
                int(records[safe_index]["definition"]["order"]),
                int(records[unsafe_index]["definition"]["order"]),
                safe_index,
                unsafe_index,
            ))
    best = min(ranked)
    return int(best[-2]), int(best[-1])


def midpoint_definition(
    negative_definition: Mapping[str, Any],
    positive_definition: Mapping[str, Any], *, iteration: int, order: int,
) -> dict[str, Any]:
    negative = negative_definition["actions"]
    positive = positive_definition["actions"]
    if (
        len(negative) != 5 or len(positive) != 5
        or any(len(row) != 7 for row in list(negative) + list(positive))
    ):
        raise ValueError("adaptive v2 midpoint action shape differs")
    return {
        "name": "adaptive_v2_bisection_%02d" % int(iteration),
        "order": int(order),
        "direction": None,
        "direction_source": "target_row_action_segment",
        "direction_coefficients": None,
        "mixture_seed": None,
        "sign": 0,
        "temporal_profile": "target_row_negative_positive_action_segment",
        "requested_correction_l2_action": None,
        "applied_correction_l2_action": None,
        "clipped": False,
        "bisection_iteration": int(iteration),
        "negative_parent": str(negative_definition["name"]),
        "positive_parent": str(positive_definition["name"]),
        "actions": [
            [0.5 * (float(a) + float(b)) for a, b in zip(left, right)]
            for left, right in zip(negative, positive)
        ],
    }
