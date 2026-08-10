"""Decision-level resampling audit for the fixed multi-region oracle."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .ridge_huber_oracle import fit_ridge_huber_gradient


CONFIG_SCHEMA = "vlsa_distal_multi_region_decision_stability_moka10.v1"
RESULT_SCHEMA = "vlsa_distal_multi_region_decision_stability_moka10_result.v1"
VALIDATION_SCHEMA = (
    "vlsa_distal_multi_region_decision_stability_moka10_validation.v1"
)


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
        raise ValueError("decision-stability config is invalid JSON") from error
    required = {
        "schema_version", "protocol_id", "claim_scope", "immutable_source",
        "source_population_manifest_sha256", "selected_manifest_sha256",
        "geometry_config_file_sha256", "exact_box_config_file_sha256",
        "clearance_target_m", "resampling", "ridge_huber", "projection",
        "exact_verification", "decision_gate",
    }
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError("decision-stability config keys differ")
    if (
        config["schema_version"] != CONFIG_SCHEMA
        or config["protocol_id"]
        != "vlsa-distal-multi-region-decision-stability-moka10-v1"
    ):
        raise ValueError("decision-stability protocol differs")
    source = config["immutable_source"]
    if set(source) != {
        "dataset_file_sha256", "dataset_payload_sha256",
        "off_grid_result_file_sha256", "off_grid_result_payload_sha256",
        "multi_region_result_file_sha256",
        "multi_region_result_payload_sha256",
        "multi_region_validation_file_sha256", "expected_state_count",
        "fit_actions_per_state", "off_grid_actions_per_state",
        "region_count",
    }:
        raise ValueError("decision-stability immutable source differs")
    if (
        int(source["expected_state_count"]) != 50
        or int(source["fit_actions_per_state"]) != 125
        or int(source["off_grid_actions_per_state"]) != 96
        or int(source["region_count"]) != 27
    ):
        raise ValueError("decision-stability source counts differ")
    if float(config["clearance_target_m"]) != 0.0:
        raise ValueError("decision-stability is zero-margin only")
    if config["resampling"] != {
        "method": "deterministic_without_replacement",
        "replicate_count": 16, "candidate_fraction": 0.8,
        "selected_actions_per_region": 22, "seed": 20260810,
        "shared_subset_across_all_seven_rows": True,
        "exact_region_center_anchor_retained": True,
        "one_sided_error_recalibrated_on_all_27_region_actions": True,
    }:
        raise ValueError("decision-stability resampling differs")
    if config["ridge_huber"] != {
        "response_unit": "millimetres", "boundary_band_m": 0.005,
        "boundary_weight_multiplier": 9.0, "huber_delta_mm": 2.0,
        "ridge_strength": 0.0001, "optimizer": "LBFGSB",
        "maximum_iterations": 2000, "gradient_tolerance": 1.0e-10,
        "one_sided_padding_m": 1.0e-6,
    }:
        raise ValueError("decision-stability ridge-Huber settings differ")
    projection = config["projection"]
    if set(projection) != {
        "eps_abs", "eps_rel", "max_iter", "residual_tolerance",
        "bound_tolerance_action",
    }:
        raise ValueError("decision-stability projection differs")
    if config["exact_verification"] != {
        "horizon_actions": 2,
        "second_action": "immutable_released_AEGIS_nominal",
        "osc_internal_substeps": "all",
        "selected_candidate_per_state_replicate": (
            "nearest_valid_regional_QP_then_region_index"
        ),
        "freshly_verify_selected_candidate_only": True,
        "require_all_seven_distal_margins_nonnegative": True,
        "require_zero_raw_L5_L6_L7_contact": True,
        "maximum_per_step_obstacle_l1_displacement_m": 1.0e-4,
        "released_AEGIS_EE_proxy": "diagnostic_only",
    }:
        raise ValueError("decision-stability exact verification differs")
    if config["decision_gate"] != {
        "replicated_off_grid_false_safe_action_count": 0,
        "minimum_global_accepted_set_jaccard_per_replicate": 0.98,
        "minimum_state_accepted_set_jaccard": 0.8,
        "retain_safe_support_in_all_baseline_supported_states": True,
        "expected_baseline_supported_state_count": 49,
        "every_state_replicate_valid_and_fresh_exact_safe_QP": True,
        "expected_selected_QP_rollout_count": 800,
        "selected_action_shift_l2_p95_maximum": 0.05,
        "selected_action_shift_l2_maximum": 0.15,
        "mlp_training_authorized": "only_if_every_decision_gate_passes",
        "mlp_training_executed": False,
        "closed_loop_e05_executed": False,
    }:
        raise ValueError("decision-stability gate differs")
    output = json.loads(_canonical(config).decode("utf-8"))
    output["config_file_sha256"] = _sha256(raw)
    output["config_payload_sha256"] = _sha256(_canonical(config))
    return output


def shared_region_resamples(
    candidate_indexes: Sequence[int], *, replicate_count: int,
    selected_count: int, seed: int,
) -> list[list[int]]:
    import numpy as np

    indexes = np.asarray(candidate_indexes, dtype=np.int64)
    if selected_count < 1 or selected_count > len(indexes):
        raise ValueError("decision-stability resample size is invalid")
    rng = np.random.RandomState(int(seed))
    return [
        indexes[np.sort(rng.choice(len(indexes), size=selected_count, replace=False))]
        .tolist()
        for _ in range(int(replicate_count))
    ]


def fit_resampled_region_target(
    candidate_xyz: Sequence[Sequence[float]],
    minimum_margin_m: Sequence[Sequence[float]], region: Mapping[str, Any],
    selected_indexes: Sequence[int], settings: Mapping[str, Any],
) -> dict[str, Any]:
    """Refit g on a subset, then recalibrate e on the complete local grid."""

    import numpy as np

    xyz = np.asarray(candidate_xyz, dtype=np.float64)
    margins = np.asarray(minimum_margin_m, dtype=np.float64)
    selected = np.asarray(selected_indexes, dtype=np.int64)
    full_indexes = np.asarray(region["fit_candidate_indexes"], dtype=np.int64)
    if not set(selected.tolist()).issubset(set(full_indexes.tolist())):
        raise ValueError("decision-stability resample leaves its region")
    anchor_index = int(region["anchor_candidate_index"])
    anchor = xyz[anchor_index]
    gradients = []
    errors = []
    for row in range(7):
        fitted = fit_ridge_huber_gradient(
            xyz[selected], margins[selected, row], anchor,
            float(margins[anchor_index, row]), settings,
        )
        gradient = np.asarray(fitted["gradient_m_per_action"], dtype=np.float64)
        predicted_full = (
            float(margins[anchor_index, row])
            + (xyz[full_indexes] - anchor[None, :]) @ gradient
        )
        error = max(
            float(np.max(predicted_full - margins[full_indexes, row])), 0.0
        ) + float(settings["one_sided_padding_m"])
        lower = predicted_full - error
        if float(np.max(lower - margins[full_indexes, row])) > 1.0e-10:
            raise RuntimeError("decision-stability recalibration failed")
        gradients.append(gradient.tolist())
        errors.append(error)
    return {
        "region_index": int(region["region_index"]),
        "selected_candidate_indexes": selected.tolist(),
        "anchor_xyz": anchor.tolist(),
        "anchor_margin_m": margins[anchor_index].tolist(),
        "gradient_m_per_action": gradients, "one_sided_error_m": errors,
    }


def jaccard(left: Sequence[bool], right: Sequence[bool]) -> float:
    import numpy as np

    a = np.asarray(left, dtype=bool)
    b = np.asarray(right, dtype=bool)
    if a.shape != b.shape:
        raise ValueError("decision-stability decision arrays differ")
    union = int(np.count_nonzero(np.logical_or(a, b)))
    if union == 0:
        return 1.0
    return float(np.count_nonzero(np.logical_and(a, b)) / union)
