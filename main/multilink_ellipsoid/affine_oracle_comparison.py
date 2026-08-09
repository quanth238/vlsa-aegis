"""No-training comparison of two local affine OSC safety representations.

This module is opt-in.  It compares a half-space fitted directly from
safe/unsafe rollout labels with a canonical finite-difference lower affine
bound.  It never changes the released AEGIS path or trains a neural model.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


CONFIG_SCHEMA = "vlsa_distal_affine_oracle_comparison_moka10.v1"
RESULT_SCHEMA = "vlsa_distal_affine_oracle_comparison_moka10_result.v1"
VALIDATION_SCHEMA = "vlsa_distal_affine_oracle_comparison_moka10_validation.v1"


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
        raise ValueError("affine oracle-comparison config is invalid JSON") from error
    required = {
        "schema_version", "protocol_id", "claim_scope", "immutable_source",
        "source_population_manifest_sha256", "selected_manifest_sha256",
        "geometry_config_file_sha256", "exact_box_config_file_sha256",
        "split", "direct_halfspace", "finite_difference", "fresh_actions",
        "exact_verification", "decision_gate",
    }
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError("affine oracle-comparison config keys differ")
    if (
        config["schema_version"] != CONFIG_SCHEMA
        or config["protocol_id"]
        != "vlsa-distal-affine-oracle-comparison-moka10-v1"
    ):
        raise ValueError("affine oracle-comparison protocol differs")
    source = config["immutable_source"]
    if set(source) != {
        "dataset_file_sha256", "dataset_payload_sha256",
        "dataset_result_file_sha256", "dataset_result_payload_sha256",
        "dataset_validation_file_sha256", "dataset_source_commit",
        "expected_state_count", "expected_fit_actions_per_state",
    }:
        raise ValueError("affine oracle-comparison source keys differ")
    if int(source["expected_state_count"]) != 50 or int(
        source["expected_fit_actions_per_state"]
    ) != 125:
        raise ValueError("affine oracle-comparison source counts differ")
    split = config["split"]
    groups = {
        name: set(split[name])
        for name in (
            "train_task_groups", "validation_task_groups", "test_task_groups"
        )
    }
    if (
        split.get("unit") != "complete_episode_and_task_level_group"
        or not all(groups.values())
        or groups["train_task_groups"] & groups["validation_task_groups"]
        or groups["train_task_groups"] & groups["test_task_groups"]
        or groups["validation_task_groups"] & groups["test_task_groups"]
    ):
        raise ValueError("affine oracle-comparison split differs")
    if config["direct_halfspace"] != {
        "fit_labels": "per_row_two_step_OSC_minimum_margin_nonnegative",
        "features": "nominal_centered_action_scaled_by_box_half_width",
        "optimizer": "class_balanced_L2_logistic_LBFGSB",
        "l2_regularization": 0.001,
        "maximum_iterations": 1000,
        "gradient_tolerance": 1.0e-10,
        "unsafe_score_padding": 1.0e-12,
        "conservative_threshold": "maximum_fitted_score_over_unsafe_fit_actions",
    }:
        raise ValueError("direct half-space contract differs")
    if config["finite_difference"] != {
        "nominal_value": "immutable_exact_nominal_two_step_margin",
        "central_epsilon_action": 0.02,
        "coefficient": "per_row_central_difference_of_exact_OSC_margin",
        "one_sided_error": "maximum_affine_overprediction_on_125_fit_actions",
        "one_sided_padding_m": 1.0e-6,
    }:
        raise ValueError("finite-difference contract differs")
    fresh = config["fresh_actions"]
    if fresh != {
        "distribution": "independent_uniform_inside_registered_action_box",
        "count_per_state": 96,
        "seed": 20260810,
        "second_action": "immutable_released_AEGIS_nominal",
    }:
        raise ValueError("fresh-action contract differs")
    if config["exact_verification"] != {
        "horizon_actions": 2,
        "osc_internal_substeps": "all",
        "true_safe_requires_all_seven_distal_margins_nonnegative": True,
        "true_safe_requires_zero_raw_L5_L6_L7_contact": True,
        "maximum_per_step_obstacle_l1_displacement_m": 1.0e-4,
        "released_AEGIS_EE_proxy": "diagnostic_only_not_a_fitted_row",
    }:
        raise ValueError("exact-verification contract differs")
    if config["decision_gate"] != {
        "false_safe_action_count": 0,
        "minimum_aggregate_safe_action_recall": 0.2,
        "minimum_safe_support_state_coverage": 0.8,
        "oracle_affine_assumption_supported": "either_method_passes_all_gates",
        "closed_loop_authorized": False,
    }:
        raise ValueError("affine oracle-comparison decision gate differs")
    output = json.loads(_canonical(config).decode("utf-8"))
    output["config_file_sha256"] = _sha256(raw)
    output["config_payload_sha256"] = _sha256(_canonical(config))
    return output


def _fit_logistic(features: Any, safe: Any, settings: Mapping[str, Any]) -> Any:
    import numpy as np
    from scipy.optimize import minimize

    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(safe, dtype=bool)
    design = np.column_stack((np.ones(len(x), dtype=np.float64), x))
    targets = np.where(y, 1.0, -1.0)
    positives = int(np.sum(y))
    negatives = int(np.sum(~y))
    weights = np.where(
        y, 0.5 / max(positives, 1), 0.5 / max(negatives, 1)
    )
    regularization = float(settings["l2_regularization"])

    def objective(parameters: Any) -> tuple[float, Any]:
        scores = design @ parameters
        signed = targets * scores
        loss = float(np.sum(weights * np.logaddexp(0.0, -signed)))
        loss += 0.5 * regularization * float(parameters[1:] @ parameters[1:])
        multiplier = -weights * targets / (1.0 + np.exp(np.clip(signed, -700, 700)))
        gradient = design.T @ multiplier
        gradient[1:] += regularization * parameters[1:]
        return loss, gradient

    solution = minimize(
        objective, np.zeros(4, dtype=np.float64), jac=True, method="L-BFGS-B",
        options={
            "maxiter": int(settings["maximum_iterations"]),
            "gtol": float(settings["gradient_tolerance"]), "ftol": 1.0e-15,
        },
    )
    if not solution.success or solution.x is None or not np.all(np.isfinite(solution.x)):
        raise RuntimeError("direct half-space logistic fit failed: %s" % solution.message)
    return np.asarray(solution.x, dtype=np.float64)


def fit_direct_halfspaces(
    candidate_xyz: Sequence[Sequence[float]],
    minimum_margin_m: Sequence[Sequence[float]],
    nominal_xyz: Sequence[float],
    action_lower: Sequence[float],
    action_upper: Sequence[float],
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    """Fit seven label-only half-spaces and tighten past every fit unsafe."""

    import numpy as np

    xyz = np.asarray(candidate_xyz, dtype=np.float64)
    margins = np.asarray(minimum_margin_m, dtype=np.float64)
    nominal = np.asarray(nominal_xyz, dtype=np.float64)
    lower = np.asarray(action_lower, dtype=np.float64)
    upper = np.asarray(action_upper, dtype=np.float64)
    if (
        xyz.ndim != 2 or xyz.shape[1] != 3 or margins.shape != (len(xyz), 7)
        or nominal.shape != (3,) or lower.shape != (3,) or upper.shape != (3,)
        or not all(np.all(np.isfinite(value)) for value in (
            xyz, margins, nominal, lower, upper
        )) or np.any(lower >= upper)
    ):
        raise ValueError("direct half-space arrays are invalid")
    scale = 0.5 * (upper - lower)
    features = (xyz - nominal[None, :]) / scale[None, :]
    intercepts = []
    gradients = []
    row_audits = []
    padding = float(settings["unsafe_score_padding"])
    for row in range(7):
        safe = margins[:, row] >= 0.0
        positive_count = int(np.sum(safe))
        negative_count = int(np.sum(~safe))
        if positive_count == 0:
            intercept = -1.0
            gradient = np.zeros(3, dtype=np.float64)
            status = "all_unsafe_reject_all"
        elif negative_count == 0:
            intercept = 1.0
            gradient = np.zeros(3, dtype=np.float64)
            status = "all_safe_accept_all"
        else:
            parameters = _fit_logistic(features, safe, settings)
            scores = parameters[0] + features @ parameters[1:]
            threshold = float(np.max(scores[~safe])) + padding
            intercept = float(parameters[0] - threshold)
            gradient = parameters[1:] / scale
            status = "mixed_labels_fitted"
        values = intercept + (xyz - nominal[None, :]) @ gradient
        false_safe = int(np.sum(np.logical_and(values >= 0.0, ~safe)))
        if false_safe != 0:
            raise RuntimeError("direct half-space tightening retained fit false-safe")
        intercepts.append(float(intercept))
        gradients.append(np.asarray(gradient, dtype=np.float64))
        row_audits.append({
            "constraint_index": row, "status": status,
            "safe_fit_action_count": positive_count,
            "unsafe_fit_action_count": negative_count,
            "accepted_fit_action_count": int(np.sum(values >= 0.0)),
            "false_safe_fit_action_count": false_safe,
        })
    return {
        "intercept_at_nominal": intercepts,
        "gradients_per_action": np.asarray(gradients).tolist(),
        "row_audits": row_audits,
        "all_fit_false_safe_count": int(sum(
            item["false_safe_fit_action_count"] for item in row_audits
        )),
    }


def fit_finite_difference_lower_bounds(
    nominal_margin_m: Sequence[float],
    plus_margin_m: Sequence[Sequence[float]],
    minus_margin_m: Sequence[Sequence[float]],
    epsilon_action: float,
    fit_candidate_xyz: Sequence[Sequence[float]],
    fit_minimum_margin_m: Sequence[Sequence[float]],
    nominal_xyz: Sequence[float],
    one_sided_padding_m: float,
) -> dict[str, Any]:
    """Build canonical central-FD rows and calibrate one-sided grid error."""

    import numpy as np

    b = np.asarray(nominal_margin_m, dtype=np.float64)
    plus = np.asarray(plus_margin_m, dtype=np.float64)
    minus = np.asarray(minus_margin_m, dtype=np.float64)
    xyz = np.asarray(fit_candidate_xyz, dtype=np.float64)
    margins = np.asarray(fit_minimum_margin_m, dtype=np.float64)
    nominal = np.asarray(nominal_xyz, dtype=np.float64)
    epsilon = float(epsilon_action)
    padding = float(one_sided_padding_m)
    if (
        b.shape != (7,) or plus.shape != (3, 7) or minus.shape != (3, 7)
        or xyz.ndim != 2 or xyz.shape[1] != 3
        or margins.shape != (len(xyz), 7) or nominal.shape != (3,)
        or epsilon <= 0.0 or padding <= 0.0
        or not all(np.all(np.isfinite(value)) for value in (
            b, plus, minus, xyz, margins, nominal
        ))
    ):
        raise ValueError("finite-difference lower-bound arrays are invalid")
    gradients = ((plus - minus) / (2.0 * epsilon)).T
    uncalibrated = b[None, :] + (xyz - nominal[None, :]) @ gradients.T
    error = np.maximum(np.max(uncalibrated - margins, axis=0), 0.0) + padding
    lower = uncalibrated - error[None, :]
    maximum_overbound = float(np.max(lower - margins))
    false_safe = int(np.sum(np.logical_and(lower >= 0.0, margins < 0.0)))
    if maximum_overbound > 1.0e-10 or false_safe != 0:
        raise RuntimeError("finite-difference fit-grid postcheck failed")
    return {
        "exact_nominal_margin_m": b.tolist(),
        "gradients_m_per_action": gradients.tolist(),
        "one_sided_error_m": error.tolist(),
        "intercept_at_nominal_m": (b - error).tolist(),
        "maximum_fit_overbound_m": maximum_overbound,
        "fit_false_safe_entry_count": false_safe,
    }


def affine_values(
    intercept_at_nominal: Sequence[float], gradients_per_action: Sequence[Sequence[float]],
    candidate_xyz: Sequence[float], nominal_xyz: Sequence[float],
) -> Any:
    import numpy as np

    intercept = np.asarray(intercept_at_nominal, dtype=np.float64)
    gradients = np.asarray(gradients_per_action, dtype=np.float64)
    candidate = np.asarray(candidate_xyz, dtype=np.float64)
    nominal = np.asarray(nominal_xyz, dtype=np.float64)
    if intercept.shape != (7,) or gradients.shape != (7, 3):
        raise ValueError("affine row shapes differ")
    return intercept + gradients @ (candidate - nominal)


def summarize_predictions(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total = len(records)
    true_safe = sum(bool(item["true_safe"]) for item in records)
    predicted_safe = sum(bool(item["predicted_safe"]) for item in records)
    accepted_safe = sum(
        bool(item["true_safe"]) and bool(item["predicted_safe"])
        for item in records
    )
    false_safe = sum(
        not bool(item["true_safe"]) and bool(item["predicted_safe"])
        for item in records
    )
    return {
        "fresh_action_count": total, "true_safe_action_count": true_safe,
        "predicted_safe_action_count": predicted_safe,
        "accepted_true_safe_action_count": accepted_safe,
        "false_safe_action_count": false_safe,
        "safe_action_recall": (accepted_safe / true_safe) if true_safe else None,
    }
