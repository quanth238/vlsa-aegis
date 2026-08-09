"""Boundary-weighted ridge-Huber local affine safety targets.

This module is opt-in and does not alter released AEGIS.  It fixes the
nominal value b=m(u0), estimates only the action gradient, and calibrates a
one-sided error on the immutable fitted actions.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


CONFIG_SCHEMA = "vlsa_distal_ridge_huber_oracle_moka10.v1"
RESULT_SCHEMA = "vlsa_distal_ridge_huber_oracle_moka10_result.v1"
VALIDATION_SCHEMA = "vlsa_distal_ridge_huber_oracle_moka10_validation.v1"


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
        raise ValueError("ridge-Huber config is invalid JSON") from error
    required = {
        "schema_version", "protocol_id", "claim_scope", "immutable_source",
        "source_population_manifest_sha256", "selected_manifest_sha256",
        "geometry_config_file_sha256", "exact_box_config_file_sha256",
        "split", "ridge_huber", "resampling", "comparison",
        "projection", "exact_verification", "oracle_gate", "learned_gate",
    }
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError("ridge-Huber config keys differ")
    if (
        config["schema_version"] != CONFIG_SCHEMA
        or config["protocol_id"] != "vlsa-distal-ridge-huber-oracle-moka10-v1"
    ):
        raise ValueError("ridge-Huber protocol differs")
    source = config["immutable_source"]
    if set(source) != {
        "dataset_file_sha256", "dataset_payload_sha256",
        "dataset_result_file_sha256", "dataset_validation_file_sha256",
        "off_grid_result_file_sha256", "off_grid_result_payload_sha256",
        "off_grid_validation_file_sha256", "dataset_source_commit",
        "expected_state_count", "fit_actions_per_state",
        "off_grid_actions_per_state",
    }:
        raise ValueError("ridge-Huber immutable-source keys differ")
    if (
        int(source["expected_state_count"]) != 50
        or int(source["fit_actions_per_state"]) != 125
        or int(source["off_grid_actions_per_state"]) != 96
    ):
        raise ValueError("ridge-Huber immutable-source counts differ")
    split = config["split"]
    if split != {
        "unit": "complete_episode_and_task_level_group",
        "expected_state_counts": {"train": 30, "validation": 5, "test": 15},
        "primary_e05_use": "test_only",
    }:
        raise ValueError("ridge-Huber split differs")
    if config["ridge_huber"] != {
        "nominal_value": "exact_two_step_OSC_margin_b_equals_m_u0",
        "response_unit": "millimetres",
        "boundary_band_m": 0.005,
        "boundary_weight_multiplier": 9.0,
        "huber_delta_mm": 2.0,
        "ridge_strength": 0.0001,
        "optimizer": "LBFGSB",
        "maximum_iterations": 2000,
        "gradient_tolerance": 1.0e-10,
        "one_sided_padding_m": 1.0e-6,
    }:
        raise ValueError("ridge-Huber target contract differs")
    if config["resampling"] != {
        "method": "deterministic_without_replacement",
        "replicate_count": 16,
        "candidate_fraction": 0.8,
        "seed": 20260810,
        "active_row": "unsafe_fit_action_or_minimum_absolute_margin_at_most_5mm",
        "minimum_cosine": 0.9,
        "maximum_relative_norm_difference": 0.25,
        "near_zero_gradient_m_per_action": 1.0e-5,
    }:
        raise ValueError("ridge-Huber resampling contract differs")
    if config["comparison"] != {
        "baseline": "immutable_candidate_conditioned_minimum_L1_target",
        "off_grid_actions": "immutable_job_37649_fresh_uniform_cloned_OSC_actions",
        "same_actions_and_exact_labels_for_both_methods": True,
        "resampling": "same_16_deterministic_80_percent_subsets_for_both_targets",
    }:
        raise ValueError("ridge-Huber comparison contract differs")
    projection = config["projection"]
    if set(projection) != {
        "action_limit", "trust_region_linf_action", "clearance_target_m",
        "eps_abs", "eps_rel", "max_iter", "residual_tolerance",
        "bound_tolerance_action",
    }:
        raise ValueError("ridge-Huber projection keys differ")
    for key in (
        "action_limit", "trust_region_linf_action", "eps_abs", "eps_rel",
        "residual_tolerance", "bound_tolerance_action",
    ):
        if not math.isfinite(float(projection[key])) or float(projection[key]) <= 0:
            raise ValueError("ridge-Huber projection value differs")
    if config["exact_verification"] != {
        "horizon_actions": 2,
        "second_action": "immutable_released_AEGIS_nominal",
        "osc_internal_substeps": "all",
        "require_all_seven_distal_margins_nonnegative": True,
        "require_zero_raw_L5_L6_L7_contact": True,
        "maximum_per_step_obstacle_l1_displacement_m": 1.0e-4,
        "released_AEGIS_EE_proxy": "diagnostic_only",
    }:
        raise ValueError("ridge-Huber exact verification differs")
    if config["oracle_gate"] != {
        "off_grid_false_safe_action_count": 0,
        "minimum_accepted_safe_action_count_per_state": 1,
        "all_active_rows_resampling_stable": True,
        "every_state_valid_QP_and_fresh_exact_safe_rollout": True,
        "training_authorized": "ridge_huber_passes_every_oracle_gate",
    }:
        raise ValueError("ridge-Huber oracle gate differs")
    if config["learned_gate"] != {
        "test_false_safe_action_count": 0,
        "test_gradient_cosine_mean_minimum": 0.8,
        "every_test_state_feasible_QP_and_fresh_exact_safe_rollout": True,
        "closed_loop_e05_authorized": "only_if_every_learned_gate_passes",
    }:
        raise ValueError("ridge-Huber learned gate differs")
    output = json.loads(_canonical(config).decode("utf-8"))
    output["config_file_sha256"] = _sha256(raw)
    output["config_payload_sha256"] = _sha256(_canonical(config))
    return output


def boundary_weights(margins_m: Any, settings: Mapping[str, Any]) -> Any:
    import numpy as np

    margins = np.asarray(margins_m, dtype=np.float64)
    band = float(settings["boundary_band_m"])
    multiplier = float(settings["boundary_weight_multiplier"])
    if band <= 0.0 or multiplier < 0.0 or not np.all(np.isfinite(margins)):
        raise ValueError("ridge-Huber boundary weights are invalid")
    return 1.0 + multiplier * np.exp(-np.abs(margins) / band)


def fit_ridge_huber_gradient(
    candidate_xyz: Sequence[Sequence[float]],
    minimum_margin_m: Sequence[float],
    nominal_xyz: Sequence[float],
    exact_nominal_margin_m: float,
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    """Fit g in m/action while holding b exactly fixed at m(u0)."""

    import numpy as np
    from scipy.optimize import minimize

    xyz = np.asarray(candidate_xyz, dtype=np.float64)
    margins = np.asarray(minimum_margin_m, dtype=np.float64)
    nominal = np.asarray(nominal_xyz, dtype=np.float64)
    b = float(exact_nominal_margin_m)
    if (
        xyz.ndim != 2 or xyz.shape[1] != 3 or margins.shape != (len(xyz),)
        or nominal.shape != (3,) or not math.isfinite(b)
        or not np.all(np.isfinite(xyz)) or not np.all(np.isfinite(margins))
        or not np.all(np.isfinite(nominal))
    ):
        raise ValueError("ridge-Huber fit arrays are invalid")
    x = xyz - nominal[None, :]
    y_mm = (margins - b) * 1000.0
    weights = boundary_weights(margins, settings)
    weight_sum = float(np.sum(weights))
    delta = float(settings["huber_delta_mm"])
    ridge = float(settings["ridge_strength"])

    def objective(g_mm: Any) -> tuple[float, Any]:
        residual = x @ g_mm - y_mm
        absolute = np.abs(residual)
        quadratic = absolute <= delta
        losses = np.where(
            quadratic, 0.5 * residual * residual,
            delta * (absolute - 0.5 * delta),
        )
        psi = np.where(quadratic, residual, delta * np.sign(residual))
        loss = float(np.sum(weights * losses) / weight_sum)
        loss += 0.5 * ridge * float(g_mm @ g_mm)
        gradient = x.T @ (weights * psi) / weight_sum + ridge * g_mm
        return loss, gradient

    solution = minimize(
        objective, np.zeros(3, dtype=np.float64), jac=True, method="L-BFGS-B",
        options={
            "maxiter": int(settings["maximum_iterations"]),
            "gtol": float(settings["gradient_tolerance"]), "ftol": 1.0e-15,
        },
    )
    if solution.x is None or not np.all(np.isfinite(solution.x)):
        raise RuntimeError("ridge-Huber optimizer produced no finite solution")
    g = np.asarray(solution.x, dtype=np.float64) * 1.0e-3
    predicted = b + x @ g
    error = max(float(np.max(predicted - margins)), 0.0) + float(
        settings["one_sided_padding_m"]
    )
    lower = predicted - error
    maximum_overbound = float(np.max(lower - margins))
    if maximum_overbound > 1.0e-10:
        raise RuntimeError("ridge-Huber fitted lower bound postcheck failed")
    return {
        "valid": True, "optimizer_success": bool(solution.success),
        "optimizer_status": int(solution.status),
        "optimizer_message": str(solution.message),
        "optimizer_iterations": int(solution.nit),
        "exact_nominal_margin_m": b,
        "gradient_m_per_action": g.tolist(),
        "one_sided_error_m": error,
        "lower_intercept_at_nominal_m": b - error,
        "maximum_fit_overbound_m": maximum_overbound,
        "fit_false_safe_action_count": int(np.sum(np.logical_and(
            lower >= 0.0, margins < 0.0
        ))),
        "fit_safe_action_count": int(np.sum(margins >= 0.0)),
        "fit_accepted_safe_action_count": int(np.sum(np.logical_and(
            lower >= 0.0, margins >= 0.0
        ))),
    }


def fit_state_targets(
    candidate_xyz: Sequence[Sequence[float]],
    minimum_margin_m: Sequence[Sequence[float]],
    nominal_xyz: Sequence[float],
    exact_nominal_margin_m: Sequence[float],
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    import numpy as np

    margins = np.asarray(minimum_margin_m, dtype=np.float64)
    nominal_margin = np.asarray(exact_nominal_margin_m, dtype=np.float64)
    if margins.ndim != 2 or margins.shape[1] != 7 or nominal_margin.shape != (7,):
        raise ValueError("ridge-Huber state target arrays differ")
    rows = [
        fit_ridge_huber_gradient(
            candidate_xyz, margins[:, row], nominal_xyz,
            nominal_margin[row], settings,
        ) for row in range(7)
    ]
    return {
        "nominal_margin_m": nominal_margin.tolist(),
        "gradient_m_per_action": [item["gradient_m_per_action"] for item in rows],
        "one_sided_error_m": [item["one_sided_error_m"] for item in rows],
        "lower_intercept_m": [
            item["lower_intercept_at_nominal_m"] for item in rows
        ],
        "row_fits": rows,
    }


def audit_resampling(
    candidate_xyz: Sequence[Sequence[float]],
    minimum_margin_m: Sequence[Sequence[float]],
    nominal_xyz: Sequence[float],
    exact_nominal_margin_m: Sequence[float],
    full_target: Mapping[str, Any],
    settings: Mapping[str, Any],
    resampling: Mapping[str, Any],
    seed: int,
) -> dict[str, Any]:
    import numpy as np

    xyz = np.asarray(candidate_xyz, dtype=np.float64)
    margins = np.asarray(minimum_margin_m, dtype=np.float64)
    nominal_margin = np.asarray(exact_nominal_margin_m, dtype=np.float64)
    full = np.asarray(full_target["gradient_m_per_action"], dtype=np.float64)
    count = int(round(len(xyz) * float(resampling["candidate_fraction"])))
    rng = np.random.RandomState(int(seed))
    replicate_gradients = []
    indexes = []
    for _ in range(int(resampling["replicate_count"])):
        selected = np.sort(rng.choice(len(xyz), size=count, replace=False))
        target = fit_state_targets(
            xyz[selected], margins[selected], nominal_xyz, nominal_margin, settings
        )
        replicate_gradients.append(target["gradient_m_per_action"])
        indexes.append(selected.tolist())
    replicate = np.asarray(replicate_gradients, dtype=np.float64)
    threshold = float(resampling["near_zero_gradient_m_per_action"])
    row_audits = []
    for row in range(7):
        active = bool(
            np.any(margins[:, row] < 0.0)
            or float(np.min(np.abs(margins[:, row]))) <= 0.005
        )
        norm = float(np.linalg.norm(full[row]))
        replicate_norm = np.linalg.norm(replicate[:, row, :], axis=1)
        differences = np.linalg.norm(replicate[:, row, :] - full[row], axis=1)
        if norm <= threshold:
            cosine_minimum = None
            relative_norm_difference_maximum = None
            stable = bool(
                float(np.max(replicate_norm)) <= threshold
                and float(np.max(differences)) <= threshold
            )
        else:
            cosine = (
                replicate[:, row, :] @ full[row]
                / np.maximum(replicate_norm * norm, 1.0e-30)
            )
            cosine_minimum = float(np.min(cosine))
            relative_norm_difference_maximum = float(np.max(
                np.abs(replicate_norm - norm) / norm
            ))
            stable = bool(
                cosine_minimum >= float(resampling["minimum_cosine"])
                and relative_norm_difference_maximum
                <= float(resampling["maximum_relative_norm_difference"])
            )
        row_audits.append({
            "constraint_index": row, "active": active,
            "full_gradient_norm_m_per_action": norm,
            "replicate_gradient_norm_minimum_m_per_action": float(
                np.min(replicate_norm)
            ),
            "replicate_gradient_norm_maximum_m_per_action": float(
                np.max(replicate_norm)
            ),
            "cosine_minimum": cosine_minimum,
            "relative_norm_difference_maximum": relative_norm_difference_maximum,
            "stable": stable,
            "gate_applies": active,
            "gate_pass": bool((not active) or stable),
        })
    return {
        "seed": int(seed), "replicate_count": len(replicate),
        "candidate_count_per_replicate": count,
        "replicate_candidate_indexes": indexes,
        "row_audits": row_audits,
        "all_active_rows_stable": bool(all(item["gate_pass"] for item in row_audits)),
    }


def affine_values(
    nominal_margin_m: Sequence[float], gradient_m_per_action: Sequence[Sequence[float]],
    one_sided_error_m: Sequence[float], candidate_xyz: Sequence[float],
    nominal_xyz: Sequence[float],
) -> Any:
    import numpy as np

    b = np.asarray(nominal_margin_m, dtype=np.float64)
    g = np.asarray(gradient_m_per_action, dtype=np.float64)
    error = np.asarray(one_sided_error_m, dtype=np.float64)
    candidate = np.asarray(candidate_xyz, dtype=np.float64)
    nominal = np.asarray(nominal_xyz, dtype=np.float64)
    if b.shape != (7,) or g.shape != (7, 3) or error.shape != (7,):
        raise ValueError("ridge-Huber affine shapes differ")
    return b - error + g @ (candidate - nominal)
