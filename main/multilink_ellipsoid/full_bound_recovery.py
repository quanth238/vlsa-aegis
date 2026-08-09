"""Frozen helpers for the E05 full-action-bound recovery diagnostic."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA = "vlsa_distal_full_bound_recovery_e05.v1"
RESULT_SCHEMA = "vlsa_distal_full_bound_recovery_e05_result.v1"
VALIDATION_SCHEMA = "vlsa_distal_full_bound_recovery_e05_validation.v1"


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def load_full_bound_config(path: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    try:
        config = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("full-bound recovery config is invalid JSON") from error
    required = {
        "schema_version", "protocol_id", "case_id", "claim_scope", "source",
        "geometry", "action_space", "finite_difference", "optimizer",
        "verification", "decision_gate",
    }
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError("full-bound recovery config keys differ")
    if (
        config["schema_version"] != SCHEMA
        or config["protocol_id"] != "vlsa-distal-full-bound-recovery-e05-v1"
        or config["case_id"] != "vlsa-t1-goal-ii-t0-e05"
    ):
        raise ValueError("full-bound recovery identity differs")
    source = config["source"]
    if set(source) != {
        "table1_population_manifest_sha256", "archived_relative_path",
        "archived_file_sha256", "archived_payload_sha256",
        "executed_sequence_sha256", "action_count", "action_step",
        "prior_grouped_job_id", "prior_dataset_sha256",
        "prior_result_sha256", "prior_validation_sha256",
    } or source["action_count"] != 237 or source["action_step"] != 186:
        raise ValueError("full-bound recovery source differs")
    geometry = config["geometry"]
    if (
        geometry.get("constraint_count") != 7
        or geometry.get("constraint_order") != [
            "L5_part_0", "L5_part_1", "L5_part_2", "L6_part_0",
            "L6_part_1", "L7_part_0", "L7_part_1",
        ]
        or geometry.get("clearance_target_m") != 0.0
    ):
        raise ValueError("full-bound recovery geometry differs")
    action = config["action_space"]
    if action != {
        "action_limit": 1.0,
        "grid_points_per_dimension": 9,
        "expected_grid_count": 729,
        "include_nominal": True,
        "include_reverse_nominal": True,
        "include_stop": True,
    }:
        raise ValueError("full-bound recovery action space differs")
    if config["finite_difference"] != {
        "epsilon_action": 0.1,
        "scheme": "clipped_central_difference_at_nominal",
    }:
        raise ValueError("full-bound recovery finite difference differs")
    optimizer = config["optimizer"]
    if set(optimizer) != {
        "eps_abs", "eps_rel", "max_iter", "residual_tolerance",
        "bound_tolerance_action", "bounds",
    } or optimizer["bounds"] != "full_normalized_translation_box_minus1_plus1":
        raise ValueError("full-bound recovery optimizer differs")
    for key in ("eps_abs", "eps_rel", "residual_tolerance", "bound_tolerance_action"):
        if not math.isfinite(float(optimizer[key])) or float(optimizer[key]) <= 0.0:
            raise ValueError("full-bound recovery optimizer scalar differs")
    if not isinstance(optimizer["max_iter"], int) or optimizer["max_iter"] < 1:
        raise ValueError("full-bound recovery max_iter differs")
    if config["verification"] != {
        "raw_contact_distance_threshold_m": 0.0,
        "maximum_within_step_obstacle_l1_displacement_m": 0.0001,
        "clone_state_tolerance": 1.0e-10,
        "require_all_internal_osc_substeps": True,
        "execute_smallest_verified_recovery_once": True,
    }:
        raise ValueError("full-bound recovery verification differs")
    if config["decision_gate"] != {
        "physical_recovery": "at_least_one_full_bound_candidate_has_all_seven_exact_substep_gaps_nonnegative_zero_raw_L5_L7_contact_and_at_most_0p1mm_obstacle_motion",
        "qp_recovery": "finite_difference_seven_row_full_bound_qp_is_valid_and_its_exact_clone_passes_physical_recovery",
        "execution_fidelity": "executed_smallest_verified_recovery_next_state_hash_equals_clone",
        "larger_region_collection": "authorized_only_if_physical_recovery_and_execution_fidelity_pass",
        "closed_loop": "not_authorized_by_this_single_state_diagnostic",
    }:
        raise ValueError("full-bound recovery decision gate differs")
    output = json.loads(_canonical(config).decode("utf-8"))
    output["config_file_sha256"] = hashlib.sha256(raw).hexdigest()
    output["config_payload_sha256"] = hashlib.sha256(_canonical(config)).hexdigest()
    return output


def full_bound_candidates(
    nominal_xyz: Sequence[float], config: Mapping[str, Any]
) -> list[dict[str, Any]]:
    import numpy as np

    nominal = np.asarray(nominal_xyz, dtype=np.float64)
    if nominal.shape != (3,) or not np.all(np.isfinite(nominal)):
        raise ValueError("full-bound nominal action differs")
    settings = config["action_space"]
    limit = float(settings["action_limit"])
    count = int(settings["grid_points_per_dimension"])
    axis = np.linspace(-limit, limit, count)
    candidates = [
        ("global_grid", np.asarray(value, dtype=np.float64))
        for value in itertools.product(axis, repeat=3)
    ]
    if len(candidates) != int(settings["expected_grid_count"]):
        raise ValueError("full-bound grid count differs")
    if settings["include_nominal"]:
        candidates.append(("nominal", nominal.copy()))
    if settings["include_stop"]:
        candidates.append(("stop", np.zeros(3, dtype=np.float64)))
    if settings["include_reverse_nominal"]:
        candidates.append(("reverse_nominal", np.clip(-nominal, -limit, limit)))
    seen = set()
    output = []
    for source, value in candidates:
        key = tuple(float(item) for item in value)
        if key in seen:
            continue
        seen.add(key)
        output.append({"source": source, "xyz": value})
    return output


def transition_is_safe(transition: Mapping[str, Any], config: Mapping[str, Any]) -> bool:
    import numpy as np

    gaps = np.asarray(transition["minimum_substep_clearance_m"][:7], dtype=np.float64)
    return bool(
        gaps.shape == (7,)
        and np.all(gaps >= float(config["geometry"]["clearance_target_m"]))
        and int(transition["raw_protected_contact_count"]) == 0
        and float(transition["maximum_within_step_obstacle_l1_displacement_m"])
        <= float(config["verification"]["maximum_within_step_obstacle_l1_displacement_m"])
    )
