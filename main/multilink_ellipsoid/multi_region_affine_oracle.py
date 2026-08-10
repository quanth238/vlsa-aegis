"""Fixed overlapping multi-region affine safety oracle.

This module is opt-in.  It partitions the existing three-dimensional action
box into 27 fixed overlapping boxes and fits an independent conservative
ridge-Huber lower envelope in every box.  It never changes released AEGIS.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .ridge_huber_oracle import fit_ridge_huber_gradient


CONFIG_SCHEMA = "vlsa_distal_multi_region_affine_oracle_moka10.v1"
RESULT_SCHEMA = "vlsa_distal_multi_region_affine_oracle_moka10_result.v1"
VALIDATION_SCHEMA = "vlsa_distal_multi_region_affine_oracle_moka10_validation.v1"


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    try:
        config = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("multi-region affine config is invalid JSON") from error
    required = {
        "schema_version", "protocol_id", "claim_scope", "immutable_source",
        "source_population_manifest_sha256", "selected_manifest_sha256",
        "geometry_config_file_sha256", "exact_box_config_file_sha256",
        "split", "partition", "ridge_huber", "resampling", "comparison",
        "projection", "clearance_arms_m", "exact_verification", "oracle_gate",
    }
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError("multi-region affine config keys differ")
    if (
        config["schema_version"] != CONFIG_SCHEMA
        or config["protocol_id"]
        != "vlsa-distal-multi-region-affine-oracle-moka10-v1"
    ):
        raise ValueError("multi-region affine protocol differs")
    source = config["immutable_source"]
    if set(source) != {
        "dataset_file_sha256", "dataset_payload_sha256",
        "dataset_result_file_sha256", "dataset_validation_file_sha256",
        "off_grid_result_file_sha256", "off_grid_result_payload_sha256",
        "off_grid_validation_file_sha256", "single_affine_result_file_sha256",
        "single_affine_result_payload_sha256",
        "single_affine_validation_file_sha256", "dataset_source_commit",
        "expected_state_count", "fit_actions_per_state",
        "off_grid_actions_per_state",
    }:
        raise ValueError("multi-region immutable source keys differ")
    if (
        int(source["expected_state_count"]) != 50
        or int(source["fit_actions_per_state"]) != 125
        or int(source["off_grid_actions_per_state"]) != 96
    ):
        raise ValueError("multi-region immutable source counts differ")
    if config["split"] != {
        "unit": "complete_episode_and_task_level_group",
        "expected_state_counts": {"train": 30, "validation": 5, "test": 15},
        "primary_e05_use": "test_only",
    }:
        raise ValueError("multi-region split differs")
    if config["partition"] != {
        "coordinates": "per_state_action_box_normalized_minus1_plus1",
        "axis_intervals": [[-1.0, 0.0], [-0.5, 0.5], [0.0, 1.0]],
        "axis_centers": [-0.5, 0.0, 0.5],
        "region_count": 27,
        "fit_actions_per_region": 27,
        "inclusive_membership_tolerance": 1.0e-12,
    }:
        raise ValueError("multi-region partition differs")
    if config["ridge_huber"] != {
        "nominal_value": "exact_region_center_two_step_OSC_margin",
        "response_unit": "millimetres", "boundary_band_m": 0.005,
        "boundary_weight_multiplier": 9.0, "huber_delta_mm": 2.0,
        "ridge_strength": 0.0001, "optimizer": "LBFGSB",
        "maximum_iterations": 2000, "gradient_tolerance": 1.0e-10,
        "one_sided_padding_m": 1.0e-6,
    }:
        raise ValueError("multi-region ridge-Huber settings differ")
    if config["resampling"] != {
        "method": "deterministic_without_replacement",
        "replicate_count": 16, "candidate_fraction": 0.8,
        "seed": 20260810,
        "active_row": "unsafe_region_action_or_minimum_absolute_margin_at_most_5mm",
        "minimum_cosine": 0.9,
        "maximum_relative_norm_difference": 0.25,
        "near_zero_gradient_m_per_action": 1.0e-5,
    }:
        raise ValueError("multi-region resampling differs")
    if config["comparison"] != {
        "baseline": "immutable_single_affine_ridge_huber_job_37688",
        "off_grid_actions": "immutable_job_37649_fresh_uniform_cloned_OSC_actions",
        "same_states_actions_and_exact_labels": True,
    }:
        raise ValueError("multi-region comparison differs")
    if config["clearance_arms_m"] != [0.0, 0.001]:
        raise ValueError("multi-region clearance arms differ")
    projection = config["projection"]
    if set(projection) != {
        "eps_abs", "eps_rel", "max_iter", "residual_tolerance",
        "bound_tolerance_action",
    }:
        raise ValueError("multi-region projection keys differ")
    for key in ("eps_abs", "eps_rel", "residual_tolerance", "bound_tolerance_action"):
        if not math.isfinite(float(projection[key])) or float(projection[key]) <= 0:
            raise ValueError("multi-region projection value differs")
    if int(projection["max_iter"]) < 1:
        raise ValueError("multi-region max_iter differs")
    if config["exact_verification"] != {
        "horizon_actions": 2,
        "second_action": "immutable_released_AEGIS_nominal",
        "osc_internal_substeps": "all",
        "require_all_seven_distal_margins_at_clearance_arm": True,
        "require_zero_raw_L5_L6_L7_contact": True,
        "maximum_per_step_obstacle_l1_displacement_m": 1.0e-4,
        "released_AEGIS_EE_proxy": "diagnostic_only",
        "verify_every_valid_regional_QP_proposal": True,
        "selection": "verified_safe_then_minimum_L2_from_nominal_then_region_index",
    }:
        raise ValueError("multi-region exact verification differs")
    gate = config["oracle_gate"]
    if gate != {
        "off_grid_false_safe_action_count_per_arm": 0,
        "regional_QP_false_safe_proposal_count_per_arm": 0,
        "all_active_region_rows_resampling_stable": True,
        "minimum_accepted_safe_action_count_per_supported_state": 1,
        "required_zero_margin_recovery_state_indexes": [4, 19],
        "every_state_selected_exact_safe_QP_per_arm": True,
        "overall_pass": "both_clearance_arms_pass_every_gate",
        "mlp_training_executed": False,
        "closed_loop_e05_executed": False,
    }:
        raise ValueError("multi-region oracle gate differs")
    output = json.loads(_canonical(config).decode("utf-8"))
    output["config_file_sha256"] = _sha256(raw)
    output["config_payload_sha256"] = _sha256(_canonical(config))
    return output


def normalize_actions(values: Any, lower: Sequence[float], upper: Sequence[float]) -> Any:
    import numpy as np

    array = np.asarray(values, dtype=np.float64)
    low = np.asarray(lower, dtype=np.float64)
    high = np.asarray(upper, dtype=np.float64)
    if low.shape != (3,) or high.shape != (3,) or np.any(high <= low):
        raise ValueError("multi-region action bounds are invalid")
    return 2.0 * (array - low) / (high - low) - 1.0


def fixed_regions(
    candidate_xyz: Sequence[Sequence[float]], action_lower: Sequence[float],
    action_upper: Sequence[float], partition: Mapping[str, Any],
) -> list[dict[str, Any]]:
    import numpy as np

    xyz = np.asarray(candidate_xyz, dtype=np.float64)
    low = np.asarray(action_lower, dtype=np.float64)
    high = np.asarray(action_upper, dtype=np.float64)
    normalized = normalize_actions(xyz, low, high)
    intervals = [tuple(float(v) for v in item) for item in partition["axis_intervals"]]
    centers = [float(v) for v in partition["axis_centers"]]
    tolerance = float(partition["inclusive_membership_tolerance"])
    regions = []
    for region_index, cells in enumerate(itertools.product(range(3), repeat=3)):
        normalized_lower = np.asarray([intervals[cell][0] for cell in cells])
        normalized_upper = np.asarray([intervals[cell][1] for cell in cells])
        normalized_center = np.asarray([centers[cell] for cell in cells])
        mask = np.all(
            np.logical_and(
                normalized >= normalized_lower[None, :] - tolerance,
                normalized <= normalized_upper[None, :] + tolerance,
            ), axis=1,
        )
        indexes = np.flatnonzero(mask)
        center_distance = np.max(np.abs(normalized - normalized_center[None, :]), axis=1)
        center_indexes = np.flatnonzero(center_distance <= tolerance)
        if (
            len(indexes) != int(partition["fit_actions_per_region"])
            or len(center_indexes) != 1
        ):
            raise ValueError("fixed region does not match the immutable five-point grid")
        region_lower = low + 0.5 * (normalized_lower + 1.0) * (high - low)
        region_upper = low + 0.5 * (normalized_upper + 1.0) * (high - low)
        regions.append({
            "region_index": region_index, "axis_cells": list(cells),
            "normalized_lower": normalized_lower.tolist(),
            "normalized_upper": normalized_upper.tolist(),
            "normalized_center": normalized_center.tolist(),
            "action_lower": region_lower.tolist(),
            "action_upper": region_upper.tolist(),
            "fit_candidate_indexes": indexes.tolist(),
            "anchor_candidate_index": int(center_indexes[0]),
            "anchor_xyz": xyz[center_indexes[0]].tolist(),
        })
    if len(regions) != int(partition["region_count"]):
        raise ValueError("fixed region count differs")
    return regions


def containing_region_indexes(
    xyz: Sequence[float], regions: Sequence[Mapping[str, Any]], tolerance: float,
) -> list[int]:
    import numpy as np

    action = np.asarray(xyz, dtype=np.float64)
    output = []
    for region in regions:
        lower = np.asarray(region["action_lower"], dtype=np.float64)
        upper = np.asarray(region["action_upper"], dtype=np.float64)
        if np.all(action >= lower - tolerance) and np.all(action <= upper + tolerance):
            output.append(int(region["region_index"]))
    if not output:
        raise ValueError("action is outside every fixed region")
    return output


def fit_region_target(
    candidate_xyz: Sequence[Sequence[float]],
    minimum_margin_m: Sequence[Sequence[float]], region: Mapping[str, Any],
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    import numpy as np

    xyz = np.asarray(candidate_xyz, dtype=np.float64)
    margins = np.asarray(minimum_margin_m, dtype=np.float64)
    indexes = np.asarray(region["fit_candidate_indexes"], dtype=np.int64)
    anchor_index = int(region["anchor_candidate_index"])
    anchor = xyz[anchor_index]
    rows = [
        fit_ridge_huber_gradient(
            xyz[indexes], margins[indexes, row], anchor,
            float(margins[anchor_index, row]), settings,
        ) for row in range(7)
    ]
    return {
        "region_index": int(region["region_index"]),
        "anchor_candidate_index": anchor_index, "anchor_xyz": anchor.tolist(),
        "gradient_m_per_action": [item["gradient_m_per_action"] for item in rows],
        "one_sided_error_m": [item["one_sided_error_m"] for item in rows],
        "anchor_margin_m": margins[anchor_index].tolist(), "row_fits": rows,
    }


def region_affine_values(target: Mapping[str, Any], xyz: Sequence[float]) -> Any:
    import numpy as np

    return (
        np.asarray(target["anchor_margin_m"], dtype=np.float64)
        - np.asarray(target["one_sided_error_m"], dtype=np.float64)
        + np.asarray(target["gradient_m_per_action"], dtype=np.float64)
        @ (np.asarray(xyz, dtype=np.float64) - np.asarray(target["anchor_xyz"], dtype=np.float64))
    )


def certificate_at_nominal(
    target: Mapping[str, Any], nominal_xyz: Sequence[float], clearance_m: float,
) -> dict[str, Any]:
    import numpy as np

    nominal = np.asarray(nominal_xyz, dtype=np.float64)
    anchor = np.asarray(target["anchor_xyz"], dtype=np.float64)
    gradient = np.asarray(target["gradient_m_per_action"], dtype=np.float64)
    intercept = (
        np.asarray(target["anchor_margin_m"], dtype=np.float64)
        - np.asarray(target["one_sided_error_m"], dtype=np.float64)
        - float(clearance_m) + gradient @ (nominal - anchor)
    )
    return {
        "valid": True, "intercept_at_nominal_m": intercept.tolist(),
        "gradients_m_per_action": gradient.tolist(),
    }


def audit_region_resampling(
    candidate_xyz: Sequence[Sequence[float]],
    minimum_margin_m: Sequence[Sequence[float]], region: Mapping[str, Any],
    full_target: Mapping[str, Any], settings: Mapping[str, Any],
    resampling: Mapping[str, Any], seed: int,
) -> dict[str, Any]:
    import numpy as np

    xyz = np.asarray(candidate_xyz, dtype=np.float64)
    margins = np.asarray(minimum_margin_m, dtype=np.float64)
    indexes = np.asarray(region["fit_candidate_indexes"], dtype=np.int64)
    anchor_index = int(region["anchor_candidate_index"])
    anchor = xyz[anchor_index]
    full = np.asarray(full_target["gradient_m_per_action"], dtype=np.float64)
    count = int(round(len(indexes) * float(resampling["candidate_fraction"])))
    rng = np.random.RandomState(int(seed))
    row_audits = []
    for row in range(7):
        local_margins = margins[indexes, row]
        active = bool(
            np.any(local_margins < 0.0)
            or float(np.min(np.abs(local_margins))) <= 0.005
        )
        if not active:
            row_audits.append({
                "constraint_index": row, "active": False, "stable": True,
                "cosine_minimum": None,
                "relative_norm_difference_maximum": None, "gate_pass": True,
            })
            continue
        gradients = []
        selections = []
        for _ in range(int(resampling["replicate_count"])):
            selected_local = np.sort(rng.choice(len(indexes), size=count, replace=False))
            selected = indexes[selected_local]
            fitted = fit_ridge_huber_gradient(
                xyz[selected], margins[selected, row], anchor,
                float(margins[anchor_index, row]), settings,
            )
            gradients.append(fitted["gradient_m_per_action"])
            selections.append(selected.tolist())
        replicate = np.asarray(gradients, dtype=np.float64)
        norm = float(np.linalg.norm(full[row]))
        replicate_norm = np.linalg.norm(replicate, axis=1)
        differences = np.linalg.norm(replicate - full[row], axis=1)
        threshold = float(resampling["near_zero_gradient_m_per_action"])
        if norm <= threshold:
            cosine_minimum = None
            relative_maximum = None
            stable = bool(
                float(np.max(replicate_norm)) <= threshold
                and float(np.max(differences)) <= threshold
            )
        else:
            cosine = replicate @ full[row] / np.maximum(replicate_norm * norm, 1.0e-30)
            cosine_minimum = float(np.min(cosine))
            relative_maximum = float(np.max(np.abs(replicate_norm - norm) / norm))
            stable = bool(
                cosine_minimum >= float(resampling["minimum_cosine"])
                and relative_maximum
                <= float(resampling["maximum_relative_norm_difference"])
            )
        row_audits.append({
            "constraint_index": row, "active": True, "stable": stable,
            "full_gradient_norm_m_per_action": norm,
            "cosine_minimum": cosine_minimum,
            "relative_norm_difference_maximum": relative_maximum,
            "replicate_candidate_indexes": selections, "gate_pass": stable,
        })
    return {
        "region_index": int(region["region_index"]), "row_audits": row_audits,
        "all_active_rows_stable": bool(all(item["gate_pass"] for item in row_audits)),
    }
