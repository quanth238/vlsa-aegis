"""Pure contracts for grouped query-boundary coverage discovery.

The allocation-backed producer replays immutable AEGIS ledgers.  This module
keeps the state rule and row accounting testable without MuJoCo.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


CONFIG_SCHEMA = "vlsa_distal_query_boundary_coverage.v1"
RESULT_SCHEMA = "vlsa_distal_query_boundary_coverage_result.v1"
VALIDATION_SCHEMA = "vlsa_distal_query_boundary_coverage_validation.v1"

ROW_IDENTITIES = (
    {"row": 0, "link": "L5", "body": "robot0_link5", "slab": 0},
    {"row": 1, "link": "L5", "body": "robot0_link5", "slab": 1},
    {"row": 2, "link": "L5", "body": "robot0_link5", "slab": 2},
    {"row": 3, "link": "L6", "body": "robot0_link6", "slab": 0},
    {"row": 4, "link": "L6", "body": "robot0_link6", "slab": 1},
    {"row": 5, "link": "L7", "body": "robot0_link7", "slab": 0},
    {"row": 6, "link": "L7", "body": "robot0_link7", "slab": 1},
)


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def load_config(path: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    value = json.loads(raw)
    required = {
        "schema_version", "protocol_id", "claim_scope", "source",
        "state_rule", "nominal_prefix", "coverage_report", "forbidden",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("query-boundary coverage config keys differ")
    if value["schema_version"] != CONFIG_SCHEMA:
        raise ValueError("query-boundary coverage schema differs")
    if value["protocol_id"] != "vlsa-distal-query-boundary-coverage-v1":
        raise ValueError("query-boundary coverage protocol differs")
    if value["source"] != {
        "selection_manifest": "manifests/vlsa_distal_clean_action_risk.v1.jsonl",
        "selection_manifest_file_sha256": (
            "8db2b300dc22198fcd2b6857ae4b68a04af4bde0fbdbca1a47d2d0598c47c326"
        ),
        "included_splits": ["diagnostic", "train", "validation"],
        "excluded_splits": ["test"],
        "episode_grouping_preserved": True,
        "diagnostic_cases_never_final_claim": [
            "vlsa-t1-goal-ii-t0-e05", "vlsa-t1-goal-ii-t0-e10"
        ],
        "final_test_episodes": "new_complete_episodes_reserved_after_coverage",
    }:
        raise ValueError("query-boundary coverage source differs")
    if value["state_rule"] != {
        "boundaries": "every_real_pi05_query_boundary_before_first_protected_contact",
        "query_stride_actions": 5,
        "minimum_initial_proxy_clearance_m": 0.001,
        "maximum_initial_active_obstacle_l1_displacement_m": 0.001,
        "require_zero_initial_protected_contact": True,
        "retention": "all_initially_safe_boundaries_with_unsafe_nominal_prefix",
    }:
        raise ValueError("query-boundary coverage state rule differs")
    if value["nominal_prefix"] != {
        "actions": 5,
        "source": "immutable_released_AEGIS_env_step_inputs_at_bound_query",
        "expected_mujoco_substeps_per_action": 25,
        "boundary_equivalence_tolerance_m": 1.0e-12,
        "safety_buffer_m": 0.001,
        "paper_car_threshold_m": 0.001,
        "initial_state_is_eligibility_only": True,
        "unsafe_if": "any_seven_row_violation_or_protected_contact_or_paper_CAR",
    }:
        raise ValueError("query-boundary coverage prefix differs")
    if value["coverage_report"] != {
        "row_identities": list(ROW_IDENTITIES),
        "near_boundary_band_m": 0.005,
        "counts": [
            "retained_state", "nominal_violated_row", "nominal_active_witness",
            "current_near_boundary",
        ],
    }:
        raise ValueError("query-boundary coverage report differs")
    if value["forbidden"] != {
        "candidate_execution": True,
        "model_training": True,
        "calibration": True,
        "QP": True,
        "closed_loop": True,
        "timeout_as_safe": True,
    }:
        raise ValueError("query-boundary coverage forbidden set differs")
    output = json.loads(canonical(value).decode("utf-8"))
    output["config_file_sha256"] = hashlib.sha256(raw).hexdigest()
    output["config_payload_sha256"] = hashlib.sha256(canonical(value)).hexdigest()
    return output


def initially_safe(
    row_clearance_m: Sequence[float], *, protected_contact_count: int,
    active_obstacle_l1_displacement_m: float, config: Mapping[str, Any],
) -> bool:
    values = [float(item) for item in row_clearance_m]
    if len(values) != 7 or any(not math.isfinite(item) for item in values):
        raise ValueError("query-boundary current clearances differ")
    rule = config["state_rule"]
    return bool(
        min(values) >= float(rule["minimum_initial_proxy_clearance_m"])
        and int(protected_contact_count) == 0
        and float(active_obstacle_l1_displacement_m)
        <= float(rule["maximum_initial_active_obstacle_l1_displacement_m"])
    )


def nominal_prefix_unsafe(
    row_minimum_clearance_m: Sequence[float], *, protected_contact_count: int,
    maximum_active_obstacle_l1_displacement_m: float,
    config: Mapping[str, Any],
) -> bool:
    values = [float(item) for item in row_minimum_clearance_m]
    if len(values) != 7 or any(not math.isfinite(item) for item in values):
        raise ValueError("query-boundary prefix clearances differ")
    prefix = config["nominal_prefix"]
    return bool(
        min(values) < float(prefix["safety_buffer_m"])
        or int(protected_contact_count) > 0
        or float(maximum_active_obstacle_l1_displacement_m)
        > float(prefix["paper_car_threshold_m"])
    )


def row_coverage(records: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> list[dict[str, Any]]:
    band = float(config["coverage_report"]["near_boundary_band_m"])
    buffer_m = float(config["nominal_prefix"]["safety_buffer_m"])
    output = []
    for identity in ROW_IDENTITIES:
        row = int(identity["row"])
        output.append({
            **identity,
            "retained_state_count": len(records),
            "nominal_violated_row_count": sum(
                float(item["nominal_prefix"]["row_minimum_clearance_m"][row]) < buffer_m
                for item in records
            ),
            "nominal_active_witness_count": sum(
                int(item["nominal_prefix"]["active_witness"]["row"]) == row
                for item in records
            ),
            "current_near_boundary_count": sum(
                abs(float(item["current"]["row_clearance_m"][row]) - buffer_m) <= band
                for item in records
            ),
        })
    return output
