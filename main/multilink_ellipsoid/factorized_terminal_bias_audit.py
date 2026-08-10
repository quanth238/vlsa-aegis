"""Audit terminal execution bias before another factorized-model ablation.

This module contains only deterministic summaries and frozen decision rules.
The MuJoCo/FK reconstruction remains in the allocation-only audit script.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .complete_osc_margin import _canonical, _numpy


CONFIG_SCHEMA = "vlsa_distal_factorized_terminal_bias_audit_config.v1"
RESULT_SCHEMA = "vlsa_distal_factorized_terminal_bias_audit_result.v1"
VALIDATION_SCHEMA = "vlsa_distal_factorized_terminal_bias_audit_validation.v1"

LINK_ROWS = {"L5": (0, 1, 2), "L6": (3, 4), "L7": (5, 6)}


def load_terminal_bias_config(path: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    try:
        config = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("terminal-bias config is invalid JSON") from error
    required = {
        "schema_version", "protocol_id", "claim_scope", "immutable_source",
        "population", "residual_audit", "geometry_signal_audit",
        "ensemble_audit", "decision_gate", "final_evaluation_policy",
        "forbidden_actions",
    }
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError("terminal-bias config keys differ")
    if (
        config["schema_version"] != CONFIG_SCHEMA
        or config["protocol_id"]
        != "vlsa-distal-factorized-terminal-bias-audit-moka10-v1"
    ):
        raise ValueError("terminal-bias protocol differs")
    population = config["population"]
    if population != {
        "episode_count": 17,
        "state_count": 85,
        "state_split_counts": {"train": 60, "validation": 10, "test": 15},
        "candidate_scope": "64_out_of_fit_random_antithetic_actions_per_state",
        "expected_action_count": 5440,
        "joint_trace_state_count": 51,
        "constraint_count": 7,
        "link_row_groups": {"L5": [0, 1, 2], "L6": [3, 4], "L7": [5, 6]},
    }:
        raise ValueError("terminal-bias population differs")
    residual = config["residual_audit"]
    if residual != {
        "residual_sign": "predicted_minus_exact_static_ellipsoid_clearance",
        "near_boundary_absolute_margin_m": 0.005,
        "terminal_substep_index": 50,
        "material_bias_minimum_sample_count": 30,
        "material_optimistic_median_m": 0.0005,
        "material_optimistic_fraction": 0.6,
        "report_every_link_and_substep": True,
    }:
        raise ValueError("terminal-bias residual audit differs")
    geometry = config["geometry_signal_audit"]
    if geometry != {
        "method": "central_finite_difference_of_exact_FK_and_ellipsoid_geometry",
        "joint_step_rad": 1e-05,
        "random_action_ridge": 1e-08,
        "candidate_resample_count": 32,
        "candidate_resample_size": 48,
        "candidate_resample_seed": 2026081102,
        "minimum_nontrivial_clearance_delta_m": 0.0001,
        "maximum_near_boundary_linearization_RMSE_m": 0.0005,
        "minimum_signed_delta_cosine": 0.9,
        "minimum_signed_delta_sign_agreement": 0.9,
        "minimum_resampled_action_gradient_median_cosine": 0.95,
        "minimum_resampled_action_gradient_p05_cosine": 0.8,
    }:
        raise ValueError("terminal-bias geometry audit differs")
    if config["ensemble_audit"] != {
        "members": 5,
        "measurement": "exact_FK_ellipsoid_global_minimum_per_member",
        "population": "all_false_safes_and_same_state_nearest_margin_correct_controls",
        "high_disagreement_median_ratio": 2.0,
        "high_disagreement_AUROC": 0.8,
    }:
        raise ValueError("terminal-bias ensemble audit differs")
    if config["forbidden_actions"] != {
        "training": True, "new_rollout_labels": True,
        "one_sided_geometry_ablation": True, "surface_loss": True,
        "binary_classifier": True, "calibration": True, "QP": True,
        "closed_loop": True, "poisson_or_SDF": True,
    }:
        raise ValueError("terminal-bias forbidden actions differ")
    output = json.loads(_canonical(config).decode("utf-8"))
    output["config_file_sha256"] = hashlib.sha256(raw).hexdigest()
    output["config_payload_sha256"] = hashlib.sha256(_canonical(config)).hexdigest()
    return output


def _vector_summary(value: Any) -> dict[str, Any]:
    np = _numpy()
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    vector = vector[np.isfinite(vector)]
    if not len(vector):
        return {"count": 0, "mean": None, "median": None, "RMSE": None,
                "absolute_p95": None, "optimistic_fraction": None}
    return {
        "count": int(len(vector)), "mean": float(np.mean(vector)),
        "median": float(np.median(vector)),
        "RMSE": float(np.sqrt(np.mean(vector ** 2))),
        "absolute_p95": float(np.quantile(np.abs(vector), 0.95)),
        "optimistic_fraction": float(np.mean(vector > 0.0)),
    }


def residual_audit(
    *, predicted_h: Any, exact_h: Any, splits: Sequence[str],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Summarize r=predicted-exact for every split, link and substep."""

    np = _numpy()
    predicted = np.asarray(predicted_h, dtype=np.float64)
    exact = np.asarray(exact_h, dtype=np.float64)
    split = np.asarray(splits, dtype=object)
    if (
        predicted.shape != exact.shape or predicted.shape[1:] != (51, 7)
        or split.shape != (len(predicted),)
        or not np.all(np.isfinite(predicted))
        or not np.all(np.isfinite(exact))
    ):
        raise ValueError("terminal-bias residual arrays differ")
    residual = predicted - exact
    boundary = float(config["residual_audit"]["near_boundary_absolute_margin_m"])
    terminal = int(config["residual_audit"]["terminal_substep_index"])
    per_split: dict[str, Any] = {}
    material: dict[str, dict[str, bool]] = {}
    for name in ("train", "validation", "test"):
        selected = split == name
        if not np.any(selected):
            raise ValueError("terminal-bias split is empty")
        by_link_substep: dict[str, list[dict[str, Any]]] = {}
        terminal_links: dict[str, Any] = {}
        material[name] = {}
        for link, rows_tuple in LINK_ROWS.items():
            rows = np.asarray(rows_tuple, dtype=np.int64)
            exact_link = np.min(exact[selected][:, :, rows], axis=2)
            predicted_link = np.min(predicted[selected][:, :, rows], axis=2)
            link_residual = predicted_link - exact_link
            summaries = []
            for substep in range(51):
                near = np.abs(exact_link[:, substep]) <= boundary
                summaries.append({
                    "substep": int(substep),
                    "all": _vector_summary(link_residual[:, substep]),
                    "near_boundary": _vector_summary(link_residual[near, substep]),
                })
            by_link_substep[link] = summaries
            terminal_near = np.abs(exact_link[:, terminal]) <= boundary
            summary = _vector_summary(link_residual[terminal_near, terminal])
            terminal_links[link] = summary
            material[name][link] = bool(
                summary["count"] >= int(
                    config["residual_audit"]["material_bias_minimum_sample_count"]
                )
                and summary["median"] >= float(
                    config["residual_audit"]["material_optimistic_median_m"]
                )
                and summary["optimistic_fraction"] >= float(
                    config["residual_audit"]["material_optimistic_fraction"]
                )
            )
        exact_safe = np.all(exact[selected] >= 0.0, axis=(1, 2))
        predicted_safe = np.all(predicted[selected] >= 0.0, axis=(1, 2))
        false_safe = predicted_safe & ~exact_safe
        worst = np.argmin(exact[selected].reshape(np.count_nonzero(selected), -1), axis=1)
        worst_substep, worst_row = np.unravel_index(worst, (51, 7))
        per_split[name] = {
            "action_count": int(np.count_nonzero(selected)),
            "false_safe_action_count": int(np.count_nonzero(false_safe)),
            "false_safe_worst_substep_counts": {
                str(index): int(np.count_nonzero(false_safe & (worst_substep == index)))
                for index in sorted(set(worst_substep[false_safe].tolist()))
            },
            "false_safe_worst_row_counts": {
                str(index): int(np.count_nonzero(false_safe & (worst_row == index)))
                for index in sorted(set(worst_row[false_safe].tolist()))
            },
            "terminal_link": terminal_links,
            "by_link_substep": by_link_substep,
        }
    l5_train_validation = material["train"]["L5"] or material["validation"]["L5"]
    if l5_train_validation:
        l5_cause = "training_objective_or_flat_decoder_bias"
    elif material["test"]["L5"]:
        l5_cause = "unseen_state_generalization_bias"
    else:
        l5_cause = "no_material_terminal_L5_bias_detected"
    any_train_validation = any(
        material[name][link]
        for name in ("train", "validation") for link in LINK_ROWS
    )
    if any_train_validation:
        distal_cause = "training_objective_or_flat_decoder_terminal_bias"
    elif any(material["test"].values()):
        distal_cause = "unseen_state_generalization_terminal_bias"
    else:
        distal_cause = "no_material_terminal_distal_bias_detected"
    return {
        "residual_definition": "predicted_minus_exact_static_clearance",
        "per_split": per_split, "material_terminal_bias": material,
        "terminal_L5_cause": l5_cause,
        "terminal_distal_cause": distal_cause,
    }


def cosine_summary(exact: Any, predicted: Any) -> dict[str, Any]:
    np = _numpy()
    truth = np.asarray(exact, dtype=np.float64).reshape(-1)
    estimate = np.asarray(predicted, dtype=np.float64).reshape(-1)
    denominator = float(np.linalg.norm(truth) * np.linalg.norm(estimate))
    cosine = None if denominator <= 1e-15 else float(truth @ estimate / denominator)
    nontrivial = np.abs(truth) >= 1e-4
    return {
        "count": int(len(truth)), "cosine": cosine,
        "sign_agreement": None if not np.any(nontrivial) else float(np.mean(
            np.sign(truth[nontrivial]) == np.sign(estimate[nontrivial])
        )),
        "RMSE_m": float(np.sqrt(np.mean((estimate - truth) ** 2))),
    }


def geometry_signal_decision(metrics: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    gate = config["geometry_signal_audit"]
    tests = {
        "near_boundary_linearization_RMSE": float(
            metrics["near_boundary_linearization_RMSE_m"]
        ) <= float(gate["maximum_near_boundary_linearization_RMSE_m"]),
        "prediction_error_linearization_RMSE": float(
            metrics["near_boundary_prediction_error_linearization_RMSE_m"]
        ) <= float(gate["maximum_near_boundary_linearization_RMSE_m"]),
        "signed_delta_cosine": float(metrics["signed_delta_cosine"])
        >= float(gate["minimum_signed_delta_cosine"]),
        "signed_delta_sign_agreement": float(metrics["signed_delta_sign_agreement"])
        >= float(gate["minimum_signed_delta_sign_agreement"]),
        "resampled_gradient_median_cosine": float(
            metrics["resampled_action_gradient_median_cosine"]
        ) >= float(gate["minimum_resampled_action_gradient_median_cosine"]),
        "resampled_gradient_p05_cosine": float(
            metrics["resampled_action_gradient_p05_cosine"]
        ) >= float(gate["minimum_resampled_action_gradient_p05_cosine"]),
    }
    return {"tests": tests, "geometry_signal_valid": bool(all(tests.values()))}


def _rank_auc(positive: Any, negative: Any) -> float | None:
    np = _numpy()
    pos = np.asarray(positive, dtype=np.float64).reshape(-1)
    neg = np.asarray(negative, dtype=np.float64).reshape(-1)
    if not len(pos) or not len(neg):
        return None
    return float(np.mean(pos[:, None] > neg[None, :])
                 + 0.5 * np.mean(pos[:, None] == neg[None, :]))


def ensemble_disagreement_audit(
    *, member_minimum_m: Any, class_code: Any, config: Mapping[str, Any],
) -> dict[str, Any]:
    np = _numpy()
    member = np.asarray(member_minimum_m, dtype=np.float64)
    code = np.asarray(class_code, dtype=np.int8)
    valid = np.all(np.isfinite(member), axis=1)
    false = valid & (code == 1)
    control = valid & (code == 2)
    disagreement = np.std(member, axis=1)
    false_values = disagreement[false]
    control_values = disagreement[control]
    false_median = None if not len(false_values) else float(np.median(false_values))
    control_median = None if not len(control_values) else float(np.median(control_values))
    ratio = None
    if false_median is not None and control_median is not None:
        ratio = float(false_median / max(control_median, 1e-12))
    auc = _rank_auc(false_values, control_values)
    gate = config["ensemble_audit"]
    detectable = bool(
        ratio is not None and auc is not None
        and ratio >= float(gate["high_disagreement_median_ratio"])
        and auc >= float(gate["high_disagreement_AUROC"])
    )
    return {
        "false_safe_count": int(np.count_nonzero(false)),
        "matched_control_count": int(np.count_nonzero(control)),
        "false_safe_disagreement_median_m": false_median,
        "control_disagreement_median_m": control_median,
        "median_ratio": ratio, "AUROC": auc,
        "uncertainty_detectable": detectable,
        "interpretation": (
            "high_disagreement_may_support_uncertainty_calibration" if detectable
            else "low_or_overlapping_disagreement_indicates_systematic_bias"
        ),
    }


def audit_decision(
    *, residual: Mapping[str, Any], geometry: Mapping[str, Any],
    ensemble: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "training_submitted": False,
        "terminal_L5_cause": residual["terminal_L5_cause"],
        "terminal_distal_cause": residual["terminal_distal_cause"],
        "geometry_signal_valid": bool(geometry["geometry_signal_valid"]),
        "uncertainty_detectable": bool(ensemble["uncertainty_detectable"]),
        "matched_one_sided_ablation_authorized": bool(
            geometry["geometry_signal_valid"]
            and residual["terminal_distal_cause"]
            == "training_objective_or_flat_decoder_terminal_bias"
        ),
        "new_reserved_episode_evaluation_required": True,
    }
