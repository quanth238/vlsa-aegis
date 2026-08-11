"""Read-only audit utilities for the frozen direct-horizon OSC model."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .complete_osc_margin import _canonical, _numpy
from .factorized_execution_pilot import finite_difference_candidate_indexes


CONFIG_SCHEMA = "vlsa_distal_factorized_direct_horizon_root_cause_audit_config.v1"
CONFIG_SCHEMA_V2 = "vlsa_distal_factorized_direct_horizon_root_cause_audit_config.v2"
RESULT_SCHEMA = "vlsa_distal_factorized_direct_horizon_root_cause_audit_result.v1"
VALIDATION_SCHEMA = "vlsa_distal_factorized_direct_horizon_root_cause_audit_validation.v1"
LINK_ROWS = {"L5": (0, 1, 2), "L6": (3, 4), "L7": (5, 6)}


def load_audit_config(path: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    config = json.loads(raw)
    required = {
        "schema_version", "protocol_id", "claim_scope", "immutable_source",
        "populations", "audit", "interpretation", "forbidden_actions",
    }
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError("direct-horizon audit config keys differ")
    protocol = str(config["protocol_id"])
    if (config["schema_version"], protocol) not in {
        (CONFIG_SCHEMA,
         "vlsa-distal-factorized-direct-horizon-root-cause-audit-moka10-v1"),
        (CONFIG_SCHEMA_V2,
         "vlsa-distal-factorized-direct-horizon-root-cause-audit-moka10-v2"),
    }:
        raise ValueError("direct-horizon audit protocol differs")
    audit = config["audit"]
    expected_audit = {
        "near_boundary_absolute_margin_m": 0.005,
        "geometry_jacobian_fit_candidates": list(range(1, 29)),
        "geometry_jacobian_ridge": 1e-06,
        "sensitivity_cosine_threshold": 0.8,
        "boundary_RMSE_threshold_m": 0.002671122,
        "terminal_window_start": 45,
        "terminal_concentration_fraction": 0.5,
        "ensemble_detection_AUROC": 0.8,
        "input_distance": "RMS_train_normalized_state_only_nearest_training_state",
        "candidate_groups": [
            "nominal", "translation_FD", "rotation_FD", "gripper_FD",
            "mixed_random",
        ],
    }
    if protocol.endswith("-v2"):
        expected_audit.update({
            "sensitivity_median_norm_ratio_minimum": 0.5,
            "sensitivity_median_norm_ratio_maximum": 1.5,
            "sensitivity_mean_relative_norm_error_maximum": 0.5,
            "sensitivity_reporting": (
                "training_validation_diagnostic_reserved_by_14_action_"
                "dimensions_and_51_rollout_horizons"
            ),
        })
    if audit != expected_audit:
        raise ValueError("direct-horizon audit definition differs")
    forbidden = config["forbidden_actions"]
    if set(forbidden) != {
        "training", "new_rollout_labels", "calibration", "QP",
        "closed_loop", "poisson_or_SDF", "classifier",
    } or not all(bool(value) for value in forbidden.values()):
        raise ValueError("direct-horizon audit forbidden actions differ")
    output = json.loads(_canonical(config).decode("utf-8"))
    output["config_file_sha256"] = hashlib.sha256(raw).hexdigest()
    output["config_payload_sha256"] = hashlib.sha256(_canonical(config)).hexdigest()
    return output


def _summary(value: Any) -> dict[str, Any]:
    np = _numpy()
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    vector = vector[np.isfinite(vector)]
    if not len(vector):
        return {"count": 0, "mean": None, "median": None, "RMSE": None,
                "absolute_p95": None, "maximum_absolute": None}
    return {
        "count": int(len(vector)), "mean": float(np.mean(vector)),
        "median": float(np.median(vector)),
        "RMSE": float(np.sqrt(np.mean(vector ** 2))),
        "absolute_p95": float(np.quantile(np.abs(vector), 0.95)),
        "maximum_absolute": float(np.max(np.abs(vector))),
    }


def _cosine(exact: Any, predicted: Any) -> dict[str, Any]:
    np = _numpy()
    truth = np.asarray(exact, dtype=np.float64).reshape(len(exact), -1)
    estimate = np.asarray(predicted, dtype=np.float64).reshape(len(predicted), -1)
    truth_norm = np.linalg.norm(truth, axis=1)
    estimate_norm = np.linalg.norm(estimate, axis=1)
    valid = (truth_norm > 1.0e-12) & (estimate_norm > 1.0e-12)
    values = np.sum(truth[valid] * estimate[valid], axis=1) / (
        truth_norm[valid] * estimate_norm[valid]
    )
    ratio = estimate_norm[valid] / truth_norm[valid]
    return {
        "valid_count": int(np.count_nonzero(valid)),
        "mean_cosine": None if not len(values) else float(np.mean(values)),
        "median_cosine": None if not len(values) else float(np.median(values)),
        "cosine_p05": None if not len(values) else float(np.quantile(values, 0.05)),
        "median_norm_ratio": None if not len(ratio) else float(np.median(ratio)),
        "mean_relative_norm_error": None if not len(ratio) else float(
            np.mean(np.abs(ratio - 1.0))
        ),
    }


def candidate_group(candidate_index: int, source_code: int) -> str:
    if int(candidate_index) == 0:
        return "nominal"
    if int(source_code) == 2:
        return "mixed_random"
    for dimension in range(14):
        if int(candidate_index) in finite_difference_candidate_indexes(dimension):
            coordinate = dimension % 7
            if coordinate < 3:
                return "translation_FD"
            if coordinate < 6:
                return "rotation_FD"
            return "gripper_FD"
    raise ValueError("direct-horizon audit candidate identity differs")


def fit_diagnostic_geometry_jacobians(
    arrays: Mapping[str, Any], *, ridge: float,
) -> dict[str, Any]:
    """Fit per-state dh/dq only to measure signed joint error.

    This is an audit transform, not a learned deployment component.  Every
    state uses its own registered 28 central-FD candidates.
    """

    np = _numpy()
    q = np.asarray(arrays["joint_position_rad"], dtype=np.float64)
    h = np.asarray(arrays["ellipsoid_clearance_m"], dtype=np.float64)
    states = np.asarray(arrays["state_index"], dtype=np.int64)
    candidates = np.asarray(arrays["candidate_index"], dtype=np.int64)
    unique = sorted(set(states.tolist()))
    jacobian = np.full((max(unique) + 1, 51, 7, 7), np.nan, dtype=np.float64)
    fit_error = []
    for state in unique:
        rows = np.flatnonzero(states == state)
        by_candidate = {int(candidates[row]): int(row) for row in rows}
        if not all(index in by_candidate for index in range(29)):
            raise ValueError("direct-horizon audit FD population differs")
        nominal = by_candidate[0]
        fit_rows = np.asarray([by_candidate[index] for index in range(1, 29)])
        for k in range(51):
            x = q[fit_rows, k] - q[nominal, k]
            y = h[fit_rows, k] - h[nominal, k]
            coefficient = np.linalg.solve(
                x.T @ x + float(ridge) * np.eye(7), x.T @ y,
            )
            jacobian[state, k] = coefficient.T
            fit_error.append((x @ coefficient - y).reshape(-1))
    return {
        "jacobian_m_per_rad": jacobian,
        "fit_residual": _summary(np.concatenate(fit_error)),
        "state_count": int(len(unique)),
    }


def member_predictions(models: Sequence[Any], state: Mapping[str, Any], arrays: Mapping[str, Any]) -> Any:
    np = _numpy()
    import torch

    normalized = (
        np.asarray(arrays["features"], dtype=np.float64)
        - np.asarray(state["feature_mean"], dtype=np.float64)
    ) / np.asarray(state["feature_std"], dtype=np.float64)
    members = []
    with torch.no_grad():
        for model in models:
            parts = []
            for start in range(0, len(normalized), 512):
                parts.append(model(torch.as_tensor(
                    normalized[start:start + 512], dtype=torch.float32,
                )).cpu().numpy())
            members.append(np.concatenate(parts, axis=0))
    q0 = np.asarray(arrays["joint_position_rad"], dtype=np.float64)[:, :1]
    return q0[None, ...] + np.asarray(members, dtype=np.float64)


def state_input_distances(
    *, training_arrays: Mapping[str, Any], target_arrays: Mapping[str, Any],
    feature_mean: Any, feature_std: Any, retained_feature_count: int,
    training_state_to_episode: Optional[Mapping[int, str]] = None,
) -> dict[int, dict[str, Any]]:
    """Nearest state-only distance in the frozen model's normalized space."""

    np = _numpy()
    retained = int(retained_feature_count)
    action_start = retained - 14
    state_columns = np.r_[0:action_start, retained:retained + 6]
    mean = np.asarray(feature_mean, dtype=np.float64)[state_columns]
    std = np.asarray(feature_std, dtype=np.float64)[state_columns]
    train = np.asarray(training_arrays["features"], dtype=np.float64)
    train_state = np.asarray(training_arrays["state_index"], dtype=np.int64)
    train_split = np.asarray(training_arrays["split"], dtype=object)
    training_states = sorted(set(train_state[train_split == "train"].tolist()))
    train_rows = np.asarray([
        np.flatnonzero((train_state == state) & (train_split == "train"))[0]
        for state in training_states
    ])
    reference = (train[train_rows][:, state_columns] - mean) / std
    target = np.asarray(target_arrays["features"], dtype=np.float64)
    target_state = np.asarray(target_arrays["state_index"], dtype=np.int64)
    output = {}
    for state in sorted(set(target_state.tolist())):
        row = np.flatnonzero(target_state == state)[0]
        vector = (target[row, state_columns] - mean) / std
        difference = reference - vector[None, :]
        rms = np.sqrt(np.mean(difference ** 2, axis=1))
        maximum = np.max(np.abs(difference), axis=1)
        nearest = int(np.argmin(rms))
        output[int(state)] = {
            "nearest_training_state_index": int(training_states[nearest]),
            "nearest_training_episode": (
                None if training_state_to_episode is None else
                str(training_state_to_episode[int(training_states[nearest])])
            ),
            "nearest_RMS_z": float(rms[nearest]),
            "nearest_maximum_absolute_z": float(maximum[nearest]),
            "maximum_absolute_training_normalized_feature": float(
                np.max(np.abs(vector))
            ),
        }
    return output


def _roc_auc(labels: Any, score: Any) -> Optional[float]:
    np = _numpy()
    y = np.asarray(labels, dtype=bool)
    s = np.asarray(score, dtype=np.float64)
    positive = s[y]
    negative = s[~y]
    if not len(positive) or not len(negative):
        return None
    comparisons = positive[:, None] - negative[None, :]
    return float(np.mean(comparisons > 0) + 0.5 * np.mean(comparisons == 0))


def _population_metrics(
    *, name: str, arrays: Mapping[str, Any], predicted_q: Any,
    member_q: Any, predicted_h: Any, exact_static_h: Any,
    state_to_episode: Mapping[int, str], input_distance: Mapping[int, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    np = _numpy()
    exact_q = np.asarray(arrays["joint_position_rad"], dtype=np.float64)
    exact_h = np.asarray(arrays["ellipsoid_clearance_m"], dtype=np.float64)
    predicted_q = np.asarray(predicted_q, dtype=np.float64)
    predicted_h = np.asarray(predicted_h, dtype=np.float64)
    exact_static_h = np.asarray(exact_static_h, dtype=np.float64)
    state = np.asarray(arrays["state_index"], dtype=np.int64)
    candidate = np.asarray(arrays["candidate_index"], dtype=np.int64)
    source = np.asarray(arrays["source_code"], dtype=np.int8)
    if not (
        exact_q.shape == predicted_q.shape
        and exact_h.shape == predicted_h.shape == exact_static_h.shape
        and exact_q.shape == exact_h.shape
    ):
        raise ValueError("direct-horizon audit population shapes differ")
    jacobian_fit = fit_diagnostic_geometry_jacobians(
        arrays, ridge=float(config["audit"]["geometry_jacobian_ridge"]),
    )
    jacobian = jacobian_fit["jacobian_m_per_rad"][state]
    q_error = predicted_q - exact_q
    normal_error = np.einsum("bkrj,bkj->bkr", jacobian, q_error)
    exact_safe = np.all(exact_h >= 0.0, axis=(1, 2))
    predicted_safe = np.all(predicted_h >= 0.0, axis=(1, 2))
    false_safe = predicted_safe & ~exact_safe
    random = source == 2
    boundary = float(config["audit"]["near_boundary_absolute_margin_m"])
    exact_min = np.min(exact_h, axis=(1, 2))
    predicted_min = np.min(predicted_h, axis=(1, 2))
    boundary_rows = random & (np.abs(exact_min) <= boundary)
    member_q = np.asarray(member_q, dtype=np.float64)
    member_joint_disagreement = np.sqrt(np.mean(
        (member_q - np.mean(member_q, axis=0, keepdims=True)) ** 2,
        axis=(0, 2, 3),
    ))
    member_normal = np.einsum(
        "bkrj,mbkj->mbkr", jacobian,
        member_q - np.mean(member_q, axis=0, keepdims=True),
    )
    member_normal_disagreement = np.sqrt(np.mean(member_normal ** 2, axis=(0, 2, 3)))
    worst = np.argmin(exact_h.reshape(len(exact_h), -1), axis=1)
    worst_k, worst_row = np.unravel_index(worst, (51, 7))
    link_for_row = {
        row: link for link, rows in LINK_ROWS.items() for row in rows
    }
    false_localization = []
    for row in np.flatnonzero(false_safe):
        false_localization.append({
            "state_index": int(state[row]),
            "episode": str(state_to_episode[int(state[row])]),
            "candidate_index": int(candidate[row]),
            "candidate_group": candidate_group(candidate[row], source[row]),
            "active_link": link_for_row[int(worst_row[row])],
            "active_constraint_row": int(worst_row[row]),
            "active_substep": int(worst_k[row]),
            "exact_margin_m": float(exact_min[row]),
            "predicted_margin_m": float(predicted_min[row]),
            "joint_disagreement_rad": float(member_joint_disagreement[row]),
            "normal_disagreement_m": float(member_normal_disagreement[row]),
        })
    groups = np.asarray([
        candidate_group(index, code) for index, code in zip(candidate, source)
    ], dtype=object)
    by_group = {}
    for group in config["audit"]["candidate_groups"]:
        selected = groups == group
        by_group[group] = {
            "action_count": int(np.count_nonzero(selected)),
            "joint_error": _summary(q_error[selected]),
            "safety_normal_joint_error_m": _summary(normal_error[selected]),
            "minimum_margin_error_m": _summary(
                predicted_min[selected] - exact_min[selected]
            ),
            "false_safe_count": int(np.count_nonzero(false_safe & selected)),
            "false_safe_rate": float(np.mean(false_safe[selected])),
        }
    by_link_substep = {}
    for link, rows_tuple in LINK_ROWS.items():
        rows = np.asarray(rows_tuple, dtype=np.int64)
        link_exact = np.min(exact_h[:, :, rows], axis=2)
        link_predicted = np.min(predicted_h[:, :, rows], axis=2)
        link_normal = normal_error[:, :, rows]
        by_link_substep[link] = [{
            "substep": int(k),
            "joint_RMSE_rad": float(np.sqrt(np.mean(q_error[:, k] ** 2))),
            "margin_error_m": _summary(link_predicted[:, k] - link_exact[:, k]),
            "safety_normal_joint_error_m": _summary(link_normal[:, k]),
            "near_boundary_margin_error_m": _summary(
                (link_predicted[:, k] - link_exact[:, k])[
                    np.abs(link_exact[:, k]) <= boundary
                ]
            ),
        } for k in range(51)]
    episodes = {}
    for episode in sorted(set(state_to_episode.values())):
        selected = np.asarray([
            state_to_episode[int(item)] == episode for item in state
        ], dtype=bool)
        episodes[episode] = {
            "action_count": int(np.count_nonzero(selected)),
            "joint_error": _summary(q_error[selected]),
            "terminal_joint_error": _summary(q_error[selected, -1]),
            "safety_normal_joint_error_m": _summary(normal_error[selected]),
            "minimum_margin_error_m": _summary(
                predicted_min[selected] - exact_min[selected]
            ),
            "false_safe_count": int(np.count_nonzero(false_safe & selected)),
            "exact_safe_count": int(np.count_nonzero(exact_safe & selected)),
            "predicted_safe_count": int(np.count_nonzero(predicted_safe & selected)),
        }
    unsupported = []
    for state_id in sorted(set(state.tolist())):
        selected = state == state_id
        exact_count = int(np.count_nonzero(exact_safe & selected))
        predicted_count = int(np.count_nonzero(predicted_safe & selected))
        if exact_count > 0 and predicted_count == 0:
            unsupported.append({
                "state_index": int(state_id),
                "episode": str(state_to_episode[int(state_id)]),
                "exact_safe_count": exact_count,
                "predicted_safe_count": predicted_count,
                "false_safe_count": int(np.count_nonzero(false_safe & selected)),
                "best_exact_margin_m": float(np.max(exact_min[selected])),
                "best_predicted_margin_m": float(np.max(predicted_min[selected])),
                "input_distance": input_distance[int(state_id)],
            })
    exact_static_safe = np.all(exact_static_h >= 0.0, axis=(1, 2))
    exact_static_false_safe = exact_static_safe & ~exact_safe
    terminal_window = int(config["audit"]["terminal_window_start"])
    false_terminal_fraction = (
        0.0 if not np.any(false_safe)
        else float(np.mean(worst_k[false_safe] >= terminal_window))
    )
    return {
        "name": name,
        "action_count": int(len(exact_q)),
        "state_count": int(len(set(state.tolist()))),
        "joint_error_by_substep": [
            _summary(q_error[:, k]) for k in range(51)
        ],
        "joint_error_overall": _summary(q_error[:, 1:]),
        "joint_error_terminal": _summary(q_error[:, -1]),
        "safety_normal_joint_error_m": _summary(normal_error),
        "geometry_jacobian_fit": {
            "fit_residual": jacobian_fit["fit_residual"],
            "state_count": jacobian_fit["state_count"],
        },
        "safety": {
            "false_safe_count": int(np.count_nonzero(false_safe)),
            "exact_safe_count": int(np.count_nonzero(exact_safe)),
            "predicted_safe_count": int(np.count_nonzero(predicted_safe)),
            "safe_recall": float(
                np.count_nonzero(predicted_safe & exact_safe)
                / max(1, np.count_nonzero(exact_safe))
            ),
            "near_boundary_random_count": int(np.count_nonzero(boundary_rows)),
            "near_boundary_RMSE_m": float(np.sqrt(np.mean(
                (predicted_min[boundary_rows] - exact_min[boundary_rows]) ** 2
            ))) if np.any(boundary_rows) else None,
            "recoverable_state_count": int(sum(
                np.any(exact_safe[state == item]) for item in set(state.tolist())
            )),
            "supported_state_count": int(sum(
                np.any(exact_safe[state == item]) and np.any(predicted_safe[state == item])
                for item in set(state.tolist())
            )),
            "false_safe_terminal_window_fraction": false_terminal_fraction,
            "false_safe_localization": false_localization,
        },
        "exact_q_static_geometry": {
            "false_safe_count": int(np.count_nonzero(exact_static_false_safe)),
            "safe_recall": float(
                np.count_nonzero(exact_static_safe & exact_safe)
                / max(1, np.count_nonzero(exact_safe))
            ),
            "near_boundary_RMSE_m": float(np.sqrt(np.mean(
                (np.min(exact_static_h, axis=(1, 2))[boundary_rows]
                 - exact_min[boundary_rows]) ** 2
            ))) if np.any(boundary_rows) else None,
        },
        "by_candidate_group": by_group,
        "by_link_substep": by_link_substep,
        "by_episode": episodes,
        "input_distance_by_state": {
            str(key): value for key, value in input_distance.items()
        },
        "unsupported_recoverable_states": unsupported,
        "ensemble": {
            "joint_disagreement_rad": _summary(member_joint_disagreement),
            "normal_disagreement_m": _summary(member_normal_disagreement),
            "false_safe_joint_disagreement_rad": _summary(
                member_joint_disagreement[false_safe]
            ),
            "false_safe_normal_disagreement_m": _summary(
                member_normal_disagreement[false_safe]
            ),
            "normal_disagreement_false_safe_AUROC": _roc_auc(
                false_safe, member_normal_disagreement,
            ),
        },
        "_arrays": {
            "predicted_minimum_margin_m": predicted_min,
            "exact_minimum_margin_m": exact_min,
            "normal_error_m": normal_error,
            "false_safe": false_safe,
        },
    }


def sensitivity_audit(
    *, arrays: Mapping[str, Any], predicted_q: Any, predicted_h: Any,
    sensitivities: Mapping[str, Any], state_to_split: Mapping[int, str],
) -> dict[str, Any]:
    np = _numpy()
    negative = np.asarray(sensitivities["negative_row_index"], dtype=np.int64)
    positive = np.asarray(sensitivities["positive_row_index"], dtype=np.int64)
    dimensions = np.asarray(sensitivities["dimension_index"], dtype=np.int64)
    states = np.asarray(sensitivities["state_index"], dtype=np.int64)
    action = np.asarray(arrays["action_chunk"], dtype=np.float64).reshape(-1, 14)
    denominator = action[positive, dimensions] - action[negative, dimensions]
    q_secant = (
        np.asarray(predicted_q)[positive] - np.asarray(predicted_q)[negative]
    ) / denominator[:, None, None]
    margin = np.min(np.asarray(predicted_h), axis=1)
    h_secant = (margin[positive] - margin[negative]) / denominator[:, None]
    exact_trace = np.asarray(arrays["ellipsoid_clearance_m"], dtype=np.float64)
    predicted_trace = np.asarray(predicted_h, dtype=np.float64)
    exact_trace_secant = (
        exact_trace[positive] - exact_trace[negative]
    ) / denominator[:, None, None]
    predicted_trace_secant = (
        predicted_trace[positive] - predicted_trace[negative]
    ) / denominator[:, None, None]
    exact_q = np.asarray(sensitivities["joint_sensitivity_rad_per_action"])
    exact_h = np.asarray(sensitivities["margin_sensitivity_m_per_action"])
    family = np.asarray([
        "translation" if dim % 7 < 3 else "rotation" if dim % 7 < 6 else "gripper"
        for dim in dimensions
    ], dtype=object)
    split = np.asarray([state_to_split[int(item)] for item in states], dtype=object)
    output = {}
    coordinate_names = (
        "translation_x", "translation_y", "translation_z",
        "rotation_x", "rotation_y", "rotation_z", "gripper",
    )
    for split_name in sorted(set(split.tolist())):
        output[split_name] = {}
        split_selected = split == split_name
        for group in ("all", "translation", "rotation", "gripper"):
            selected = split_selected.copy()
            if group != "all":
                selected &= family == group
            output[split_name][group] = {
                "joint": _cosine(exact_q[selected], q_secant[selected]),
                "safety": _cosine(exact_h[selected], h_secant[selected]),
            }
        output[split_name]["all_horizon_trace"] = {
            "joint": _cosine(exact_q[split_selected], q_secant[split_selected]),
            "safety": _cosine(
                exact_trace_secant[split_selected],
                predicted_trace_secant[split_selected],
            ),
            "by_horizon": [{
                "substep": int(k),
                "joint": _cosine(
                    exact_q[split_selected, k], q_secant[split_selected, k]
                ),
                "safety": _cosine(
                    exact_trace_secant[split_selected, k],
                    predicted_trace_secant[split_selected, k],
                ),
            } for k in range(51)],
        }
        by_dimension = {}
        for dimension in range(14):
            selected = (split == split_name) & (dimensions == dimension)
            label = "action_%d_%s" % (
                dimension // 7, coordinate_names[dimension % 7]
            )
            by_dimension[label] = {
                "dimension_index": int(dimension),
                "joint_all_horizons": _cosine(
                    exact_q[selected], q_secant[selected]
                ),
                "safety_all_horizons": _cosine(
                    exact_trace_secant[selected],
                    predicted_trace_secant[selected],
                ),
                "by_horizon": [{
                    "substep": int(k),
                    "joint": _cosine(
                        exact_q[selected, k], q_secant[selected, k]
                    ),
                    "safety": _cosine(
                        exact_trace_secant[selected, k],
                        predicted_trace_secant[selected, k],
                    ),
                } for k in range(51)],
            }
        output[split_name]["by_dimension_and_horizon"] = by_dimension
    return output


def root_cause_decision(
    *, populations: Mapping[str, Mapping[str, Any]],
    sensitivities: Mapping[str, Any], config: Mapping[str, Any],
) -> dict[str, Any]:
    threshold = float(config["audit"]["boundary_RMSE_threshold_m"])
    training = populations["train"]
    validation = populations["validation"]
    reserved = populations["reserved"]
    training_sensitivity = sensitivities["source"]["train"]
    validation_sensitivity = sensitivities["source"]["validation"]
    reserved_sensitivity = sensitivities["reserved"]["reserved"]

    def sensitivity_gate(split_metrics: Mapping[str, Any]) -> dict[str, Any]:
        metrics = split_metrics.get("all_horizon_trace", split_metrics["all"])
        minimum_cosine = float(config["audit"]["sensitivity_cosine_threshold"])
        ratio_minimum = float(config["audit"].get(
            "sensitivity_median_norm_ratio_minimum", 0.0
        ))
        ratio_maximum = float(config["audit"].get(
            "sensitivity_median_norm_ratio_maximum", float("inf")
        ))
        relative_maximum = float(config["audit"].get(
            "sensitivity_mean_relative_norm_error_maximum", float("inf")
        ))
        tests = {}
        for name in ("joint", "safety"):
            item = metrics[name]
            tests[name + "_direction"] = bool(
                item["mean_cosine"] is not None
                and item["mean_cosine"] >= minimum_cosine
            )
            tests[name + "_magnitude"] = bool(
                item["median_norm_ratio"] is not None
                and ratio_minimum <= item["median_norm_ratio"] <= ratio_maximum
                and item["mean_relative_norm_error"] <= relative_maximum
            )
        return {"tests": tests, "pass": bool(all(tests.values())),
                "metrics": metrics}

    sensitivity_gates = {
        "train": sensitivity_gate(training_sensitivity),
        "validation": sensitivity_gate(validation_sensitivity),
        "reserved": sensitivity_gate(reserved_sensitivity),
    }
    training_fit_failure = bool(
        training["safety"]["false_safe_count"] > 0
        or training["safety"]["near_boundary_RMSE_m"] > threshold
        or not sensitivity_gates["train"]["pass"]
    )
    validation_fit_failure = bool(
        validation["safety"]["false_safe_count"] > 0
        or validation["safety"]["near_boundary_RMSE_m"] > threshold
        or not sensitivity_gates["validation"]["pass"]
    )
    geometry_failure = any(
        item["exact_q_static_geometry"]["false_safe_count"] > 0
        for item in populations.values()
    )
    terminal_concentration = bool(
        reserved["safety"]["false_safe_terminal_window_fraction"]
        >= float(config["audit"]["terminal_concentration_fraction"])
    )
    rotation = reserved["by_candidate_group"]["rotation_FD"]
    translation = reserved["by_candidate_group"]["translation_FD"]
    rotation_concentration = bool(
        rotation["false_safe_rate"] > 1.5 * max(translation["false_safe_rate"], 1e-12)
    )
    ensemble_detects = bool(
        (reserved["ensemble"]["normal_disagreement_false_safe_AUROC"] or 0.0)
        >= float(config["audit"]["ensemble_detection_AUROC"])
    )
    conclusion = (
        "geometry_audit_reopened" if geometry_failure else
        "model_capacity_objective_failure_already_present_in_training"
        if training_fit_failure else
        "validation_or_episode_generalization_failure"
        if validation_fit_failure or reserved["safety"]["false_safe_count"] else
        "no_failure_reproduced"
    )
    sensitivity_conclusion = (
        "loss_scaling_or_decoder_underfitting"
        if not (
            sensitivity_gates["train"]["pass"]
            and sensitivity_gates["validation"]["pass"]
        ) else
        "state_coverage_or_unseen_episode_generalization"
        if not sensitivity_gates["reserved"]["pass"] else
        "action_sensitivity_passes_all_splits"
    )
    return {
        "strict_model_NO_GO_preserved": True,
        "training_fit_failure": training_fit_failure,
        "validation_fit_failure": validation_fit_failure,
        "geometry_failure": geometry_failure,
        "terminal_error_concentration": terminal_concentration,
        "rotation_candidate_concentration": rotation_concentration,
        "ensemble_disagreement_detects_false_safes": ensemble_detects,
        "sensitivity_gates": sensitivity_gates,
        "sensitivity_root_cause": sensitivity_conclusion,
        "conclusion": conclusion,
        "authorized_next_action": (
            "reopen_geometry" if geometry_failure else
            "change_model_capacity_or_training_objective"
            if training_fit_failure else
            "collect_grouped_state_coverage_or_change_generalization_representation"
        ),
        "calibration_QP_closed_loop_authorized": False,
    }


def strip_private_arrays(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if not key.startswith("_")}
