"""No-training per-member audit for the explicit execution Jacobian."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

from .complete_osc_margin import _canonical, _numpy, _torch
from .factorized_direct_horizon_root_cause_audit import _cosine
from .factorized_execution_pilot import _arm_targets
from .factorized_explicit_execution_jacobian import nominal_row_mapping


CONFIG_SCHEMA = "vlsa_distal_factorized_explicit_jacobian_audit_config.v1"


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_explicit_jacobian_audit_config(path: Path) -> Dict[str, Any]:
    raw = Path(path).read_bytes()
    config = json.loads(raw)
    required = {
        "schema_version", "protocol_id", "claim_scope", "immutable_source",
        "population", "audit", "decision", "forbidden_actions",
    }
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError("explicit-J audit config keys differ")
    if (
        config["schema_version"] != CONFIG_SCHEMA
        or config["protocol_id"]
        != "vlsa-distal-factorized-explicit-jacobian-audit-moka10-v1"
    ):
        raise ValueError("explicit-J audit protocol differs")
    if config["audit"] != {
        "ensemble_member_count": 5,
        "splits": ["train", "validation", "test"],
        "action_families": ["all", "translation", "rotation", "gripper"],
        "report_every_action_dimension_and_horizon": True,
        "terminal_substep": 50,
        "minimum_joint_sensitivity_cosine": 0.8,
        "minimum_joint_sensitivity_median_norm_ratio": 0.5,
        "maximum_joint_sensitivity_median_norm_ratio": 1.5,
        "maximum_joint_sensitivity_mean_relative_norm_error": 0.5,
        "ensemble_cancellation_ratio_threshold": 0.5,
    }:
        raise ValueError("explicit-J audit settings differ")
    if set(config["forbidden_actions"]) != {
        "training", "simulation", "new_labels", "new_unseen_episodes",
        "residual_bound_calibration", "QP", "closed_loop",
        "poisson_or_SDF", "binary_classifier",
    } or not all(bool(value) for value in config["forbidden_actions"].values()):
        raise ValueError("explicit-J audit forbidden actions differ")
    output = json.loads(_canonical(config).decode("utf-8"))
    output["config_file_sha256"] = _sha256(raw)
    output["config_payload_sha256"] = _sha256(_canonical(config))
    return output


def member_predictions(
    models: Sequence[Any], state: Mapping[str, Any], arrays: Mapping[str, Any],
) -> Tuple[Any, Any, Any]:
    """Return per-member q, nominal displacement, and state-level explicit J."""

    np = _numpy()
    torch = _torch()
    features = np.asarray(arrays["features"], dtype=np.float64)
    normalized = (
        features - np.asarray(state["feature_mean"], dtype=np.float64)
    ) / np.asarray(state["feature_std"], dtype=np.float64)
    mapping = nominal_row_mapping(arrays)
    anchors = mapping["anchor_row_index"]
    row_to_anchor = mapping["row_to_anchor_ordinal"]
    actions = np.asarray(arrays["action_chunk"], dtype=np.float64).reshape(-1, 14)
    delta_action = actions - actions[anchors[row_to_anchor]]
    nominal_members = []
    jacobian_members = []
    with torch.no_grad():
        for model in models:
            nominal, jacobian = model(torch.as_tensor(
                normalized[anchors], dtype=torch.float32,
            ))
            nominal_members.append(nominal.cpu().numpy())
            jacobian_members.append(jacobian.cpu().numpy())
    nominal_array = np.asarray(nominal_members, dtype=np.float64)
    jacobian_array = np.asarray(jacobian_members, dtype=np.float64)
    displacement = (
        nominal_array[:, row_to_anchor]
        + np.einsum(
            "mskjd,sd->mskj", jacobian_array[:, row_to_anchor], delta_action,
        )
    )
    _, base, _ = _arm_targets(arrays, "factorized_execution")
    return base[None] + displacement, nominal_array, jacobian_array


def selected_member_sensitivities(
    *, arrays: Mapping[str, Any], sensitivities: Mapping[str, Any],
    state_indexes: Any, member_jacobian: Any,
) -> Any:
    np = _numpy()
    state_to_ordinal = {
        int(state): ordinal for ordinal, state in enumerate(
            np.asarray(state_indexes, dtype=np.int64)
        )
    }
    states = np.asarray(sensitivities["state_index"], dtype=np.int64)
    dimensions = np.asarray(sensitivities["dimension_index"], dtype=np.int64)
    jacobian = np.asarray(member_jacobian, dtype=np.float64)
    return np.asarray([
        jacobian[:, state_to_ordinal[int(state)], :, :, int(dimension)]
        for state, dimension in zip(states, dimensions)
    ]).transpose(1, 0, 2, 3)


def split_labels(
    arrays: Mapping[str, Any], sensitivities: Mapping[str, Any],
) -> Any:
    np = _numpy()
    state_to_split = {
        int(state): str(split) for state, split in zip(
            np.asarray(arrays["state_index"], dtype=np.int64),
            np.asarray(arrays["split"], dtype=object),
        )
    }
    return np.asarray([
        state_to_split[int(state)] for state in np.asarray(
            sensitivities["state_index"], dtype=np.int64,
        )
    ], dtype=object)


def _metric_pass(item: Mapping[str, Any], audit: Mapping[str, Any]) -> bool:
    return bool(
        item["mean_cosine"] is not None
        and float(item["mean_cosine"])
        >= float(audit["minimum_joint_sensitivity_cosine"])
        and item["median_norm_ratio"] is not None
        and float(audit["minimum_joint_sensitivity_median_norm_ratio"])
        <= float(item["median_norm_ratio"])
        <= float(audit["maximum_joint_sensitivity_median_norm_ratio"])
        and item["mean_relative_norm_error"] is not None
        and float(item["mean_relative_norm_error"])
        <= float(audit["maximum_joint_sensitivity_mean_relative_norm_error"])
    )


def _dimension_horizon_metrics(exact: Any, predicted: Any, dimensions: Any) -> Any:
    output = {}
    names = (
        "translation_x", "translation_y", "translation_z",
        "rotation_x", "rotation_y", "rotation_z", "gripper",
    )
    for dimension in range(14):
        selected = dimensions == dimension
        output["action_%d_%s" % (dimension // 7, names[dimension % 7])] = {
            "dimension_index": int(dimension),
            "all_horizons": _cosine(exact[selected], predicted[selected]),
            "by_horizon": [{
                "substep": int(k),
                "joint": _cosine(exact[selected, k], predicted[selected, k]),
            } for k in range(51)],
        }
    return output


def member_sensitivity_audit(
    *, arrays: Mapping[str, Any], sensitivities: Mapping[str, Any],
    selected: Any, config: Mapping[str, Any],
) -> Dict[str, Any]:
    np = _numpy()
    exact = np.asarray(
        sensitivities["joint_sensitivity_rad_per_action"], dtype=np.float64,
    )
    predicted = np.asarray(selected, dtype=np.float64)
    dimensions = np.asarray(sensitivities["dimension_index"], dtype=np.int64)
    family = np.asarray([
        "translation" if item % 7 < 3 else
        "rotation" if item % 7 < 6 else "gripper"
        for item in dimensions
    ], dtype=object)
    splits = split_labels(arrays, sensitivities)
    members = []
    for member in range(len(predicted)):
        member_result = {}
        for split in config["audit"]["splits"]:
            split_rows = splits == split
            groups = {}
            for group in config["audit"]["action_families"]:
                use = split_rows if group == "all" else split_rows & (family == group)
                groups[group] = {
                    "aggregate": _cosine(exact[use], predicted[member, use]),
                    "terminal": _cosine(
                        exact[use, 50], predicted[member, use, 50],
                    ),
                }
            groups["by_dimension_and_horizon"] = _dimension_horizon_metrics(
                exact[split_rows], predicted[member, split_rows],
                dimensions[split_rows],
            )
            member_result[split] = groups
        members.append(member_result)
    ensemble = np.mean(predicted, axis=0)
    ensemble_result = {}
    for split in config["audit"]["splits"]:
        use = splits == split
        ensemble_result[split] = {
            "aggregate": _cosine(exact[use], ensemble[use]),
            "terminal": _cosine(exact[use, 50], ensemble[use, 50]),
        }
    return {"members": members, "ensemble": ensemble_result}


def trajectory_audit(arrays: Mapping[str, Any], member_q: Any) -> Dict[str, Any]:
    np = _numpy()
    exact = np.asarray(arrays["joint_position_rad"], dtype=np.float64)
    split = np.asarray(arrays["split"], dtype=object)
    output = []
    for prediction in np.asarray(member_q, dtype=np.float64):
        item = {}
        for name in ("train", "validation", "test"):
            error = prediction[split == name] - exact[split == name]
            item[name] = {
                "overall_joint_RMSE_rad": float(np.sqrt(np.mean(error ** 2))),
                "terminal_joint_RMSE_rad": float(np.sqrt(np.mean(
                    error[:, 50] ** 2
                ))),
                "maximum_initial_joint_error_rad": float(np.max(
                    np.abs(error[:, 0])
                )),
            }
        output.append(item)
    return {"members": output}


def cancellation_audit(selected: Any, sensitivities: Mapping[str, Any], arrays: Mapping[str, Any]) -> Dict[str, Any]:
    np = _numpy()
    predicted = np.asarray(selected, dtype=np.float64)
    splits = split_labels(arrays, sensitivities)
    output = {}
    for split in ("train", "validation", "test"):
        use = splits == split
        vectors = predicted[:, use].reshape(len(predicted), np.count_nonzero(use), -1)
        ensemble = np.mean(vectors, axis=0)
        member_norm = np.mean(np.linalg.norm(vectors, axis=2), axis=0)
        ensemble_norm = np.linalg.norm(ensemble, axis=1)
        valid = member_norm > 1.0e-12
        ratio = ensemble_norm[valid] / member_norm[valid]
        pairwise = []
        for left in range(len(vectors)):
            for right in range(left + 1, len(vectors)):
                pairwise.append(_cosine(vectors[left], vectors[right])["mean_cosine"])
        output[split] = {
            "median_ensemble_norm_to_mean_member_norm_ratio": float(np.median(ratio)),
            "mean_ensemble_norm_to_mean_member_norm_ratio": float(np.mean(ratio)),
            "pairwise_member_mean_cosine_minimum": float(np.min(pairwise)),
            "pairwise_member_mean_cosine_mean": float(np.mean(pairwise)),
            "pairwise_member_mean_cosine_maximum": float(np.max(pairwise)),
        }
    return output


def audit_decision(
    *, sensitivity: Mapping[str, Any], cancellation: Mapping[str, Any],
    training_audits: Sequence[Mapping[str, Any]], config: Mapping[str, Any],
) -> Dict[str, Any]:
    audit = config["audit"]
    member_tests = []
    rows = []
    for member, metrics in enumerate(sensitivity["members"]):
        train = metrics["train"]["all"]
        validation = metrics["validation"]["all"]
        passed = bool(
            _metric_pass(train["aggregate"], audit)
            and _metric_pass(train["terminal"], audit)
            and _metric_pass(validation["aggregate"], audit)
            and _metric_pass(validation["terminal"], audit)
        )
        member_tests.append(passed)
        rows.append({
            "member_index": int(member),
            "seed": int(training_audits[member]["seed"]),
            "best_epoch": int(training_audits[member]["best_epoch"]),
            "train": train, "validation": validation,
            "train_and_validation_pass": passed,
        })
    passing = int(sum(member_tests))
    ratio = float(cancellation["validation"][
        "median_ensemble_norm_to_mean_member_norm_ratio"
    ])
    if passing >= 4 and ratio < float(
        audit["ensemble_cancellation_ratio_threshold"]
    ):
        classification = "ensemble_cancellation"
    elif passing <= 1:
        classification = "member_level_optimization_or_checkpoint_failure"
    else:
        classification = "heterogeneous_member_and_ensemble_failure"
    return {
        "member_rows": rows,
        "member_train_and_validation_pass_count": passing,
        "classification": classification,
        "retraining_authorized": False,
        "calibration_QP_closed_loop_authorized": False,
        "authorized_next_action": (
            "preregister_separate_intercept_and_J_optimization_or_checkpoint_ablation"
            if classification == "member_level_optimization_or_checkpoint_failure"
            else "preregister_ensemble_aggregation_ablation"
        ),
    }
