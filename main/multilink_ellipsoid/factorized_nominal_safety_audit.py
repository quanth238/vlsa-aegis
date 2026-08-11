"""Audit q prediction separately from exact-q FK/ellipsoid recomposition."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .execution_margin_nn import _canonical, _numpy


CONFIG_SCHEMA = "vlsa_distal_factorized_nominal_safety_formula_audit_config.v1"
RESULT_SCHEMA = "vlsa_distal_factorized_nominal_safety_formula_audit_result.v1"
VALIDATION_SCHEMA = "vlsa_distal_factorized_nominal_safety_formula_audit_validation.v1"


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def payload_sha256(value: Mapping[str, Any], field: str) -> str:
    payload = dict(value)
    payload.pop(field, None)
    return _sha256(_canonical(payload))


def load_formula_audit_config(path: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    config = json.loads(raw)
    required = {
        "schema_version", "protocol_id", "claim_scope", "immutable_source",
        "population", "decomposition", "decision_gate", "forbidden_actions",
    }
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError("nominal-safety formula config keys differ")
    if (
        config["schema_version"] != CONFIG_SCHEMA
        or config["protocol_id"]
        != "vlsa-distal-factorized-nominal-safety-formula-audit-moka10-v1"
    ):
        raise ValueError("nominal-safety formula protocol differs")
    if config["population"] != {
        "episode_count": 17, "state_count": 85,
        "random_action_count": 5440, "test_random_action_count": 960,
        "substep_count": 51, "joint_count": 7, "constraint_count": 7,
    }:
        raise ValueError("nominal-safety formula population differs")
    if not all(config["forbidden_actions"].values()):
        raise ValueError("nominal-safety formula forbidden actions differ")
    output = json.loads(_canonical(config).decode("utf-8"))
    output["config_file_sha256"] = _sha256(raw)
    output["config_payload_sha256"] = _sha256(_canonical(config))
    return output


def _rmse(value: Any) -> float:
    np = _numpy()
    array = np.asarray(value, dtype=np.float64)
    return float(np.sqrt(np.mean(array ** 2)))


def _minimum_metrics(
    target_trace: Any, estimate_trace: Any, *, boundary_m: float = 0.005,
) -> dict[str, Any]:
    np = _numpy()
    target = np.min(np.asarray(target_trace, dtype=np.float64), axis=1)
    estimate = np.min(np.asarray(estimate_trace, dtype=np.float64), axis=1)
    near = np.abs(target) <= float(boundary_m)
    target_safe = np.all(target >= 0.0, axis=1)
    estimate_safe = np.all(estimate >= 0.0, axis=1)
    accepted_safe = target_safe & estimate_safe
    return {
        "action_count": int(len(target)),
        "false_safe_action_count": int(np.count_nonzero(
            estimate_safe & ~target_safe
        )),
        "exact_safe_action_count": int(np.count_nonzero(target_safe)),
        "accepted_exact_safe_action_count": int(np.count_nonzero(accepted_safe)),
        "exact_safe_action_recall": (
            0.0 if not np.any(target_safe)
            else float(np.count_nonzero(accepted_safe) / np.count_nonzero(target_safe))
        ),
        "minimum_margin_RMSE_m": _rmse(estimate - target),
        "near_boundary_row_count": int(np.count_nonzero(near)),
        "near_boundary_RMSE_m": (
            0.0 if not np.any(near) else _rmse((estimate - target)[near])
        ),
        "maximum_absolute_minimum_margin_error_m": float(
            np.max(np.abs(estimate - target))
        ),
    }


def _trace_error(target: Any, estimate: Any) -> dict[str, Any]:
    np = _numpy()
    error = np.asarray(estimate, dtype=np.float64) - np.asarray(
        target, dtype=np.float64
    )
    by_substep = np.sqrt(np.mean(error ** 2, axis=(0, 2)))
    return {
        "trace_RMSE_m": _rmse(error),
        "k0_RMSE_m": float(by_substep[0]),
        "k0_maximum_absolute_error_m": float(np.max(np.abs(error[:, 0]))),
        "terminal_RMSE_m": float(by_substep[-1]),
        "maximum_absolute_error_m": float(np.max(np.abs(error))),
        "RMSE_by_substep_m": by_substep.tolist(),
    }


def _joint_metrics(exact: Any, predicted: Any) -> dict[str, Any]:
    np = _numpy()
    error = np.asarray(predicted, dtype=np.float64) - np.asarray(
        exact, dtype=np.float64
    )
    by_substep = np.sqrt(np.mean(error ** 2, axis=(0, 2)))
    return {
        "trajectory_RMSE_rad": _rmse(error),
        "k0_RMSE_rad": float(by_substep[0]),
        "terminal_RMSE_rad": float(by_substep[-1]),
        "maximum_absolute_error_rad": float(np.max(np.abs(error))),
        "RMSE_by_substep_rad": by_substep.tolist(),
    }


def audit_formula(
    *, arrays: Mapping[str, Any], records: Mapping[str, Any],
    structured_predictions: Mapping[str, Any], flat_output_shape: Any,
    time_output_shape: Any, config: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the registered error decomposition from immutable arrays."""

    np = _numpy()
    rows = np.asarray(records["row_index"], dtype=np.int64)
    split_code = np.asarray(records["split_code"], dtype=np.int8)
    exact_q = np.asarray(records["exact_joint_position_rad"], dtype=np.float64)
    predicted_q = np.asarray(
        records["predicted_joint_position_rad"], dtype=np.float64
    )
    dynamic_h = np.asarray(arrays["ellipsoid_clearance_m"], dtype=np.float64)[rows]
    exact_static_h = np.asarray(
        records["exact_static_clearance_m"], dtype=np.float64
    )
    predicted_static_h = np.asarray(
        records["predicted_static_clearance_m"], dtype=np.float64
    )
    expected = (int(config["population"]["random_action_count"]), 51, 7)
    if any(value.shape != expected for value in (
        exact_q, predicted_q, dynamic_h, exact_static_h, predicted_static_h,
    )):
        raise ValueError("nominal-safety formula trace shape differs")
    geometry_error = exact_static_h - dynamic_h
    joint_safety_error = predicted_static_h - exact_static_h
    total_error = predicted_static_h - dynamic_h
    identity_error = total_error - (geometry_error + joint_safety_error)
    if float(np.max(np.abs(identity_error))) > 1.0e-15:
        raise ValueError("nominal-safety formula decomposition identity differs")
    split_names = {0: "train", 1: "validation", 2: "test"}
    split_metrics = {}
    for code, name in split_names.items():
        selected = split_code == code
        split_metrics[name] = {
            "joint_prediction": _joint_metrics(
                exact_q[selected], predicted_q[selected]
            ),
            "exact_q_FK_phi_recomposition": {
                "trace": _trace_error(
                    dynamic_h[selected], exact_static_h[selected]
                ),
                "minimum": _minimum_metrics(
                    dynamic_h[selected], exact_static_h[selected]
                ),
            },
            "predicted_q_safety_error": {
                "trace": _trace_error(
                    exact_static_h[selected], predicted_static_h[selected]
                ),
                "minimum": _minimum_metrics(
                    exact_static_h[selected], predicted_static_h[selected]
                ),
            },
            "total_nominal_safety_error": {
                "trace": _trace_error(
                    dynamic_h[selected], predicted_static_h[selected]
                ),
                "minimum": _minimum_metrics(
                    dynamic_h[selected], predicted_static_h[selected]
                ),
            },
        }
    test = split_metrics["test"]
    geometry_boundary = float(
        test["exact_q_FK_phi_recomposition"]["minimum"]["near_boundary_RMSE_m"]
    )
    joint_boundary = float(
        test["predicted_q_safety_error"]["minimum"]["near_boundary_RMSE_m"]
    )
    ratio = joint_boundary / max(geometry_boundary, 1.0e-15)
    exact_static_minimum = np.asarray(
        structured_predictions["exact_q_static_minimum_margin_m"], dtype=np.float64
    )
    flat_margin = np.asarray(
        structured_predictions["flat_minimum_margin_m"], dtype=np.float64
    )
    time_margin = np.asarray(
        structured_predictions["time_minimum_margin_m"], dtype=np.float64
    )
    flat_q = np.asarray(
        structured_predictions["flat_joint_position_rad"], dtype=np.float64
    )
    time_q = np.asarray(
        structured_predictions["time_joint_position_rad"], dtype=np.float64
    )
    all_split = np.asarray(arrays["split"], dtype=object)
    source = np.asarray(arrays["source_code"], dtype=np.int8)
    test_random = (all_split == "test") & (source == 2)
    exact_test_q = np.asarray(arrays["joint_position_rad"], dtype=np.float64)[
        test_random
    ]
    current_models = {}
    for name, q_value, margin_value in (
        ("flat", flat_q, flat_margin), ("time", time_q, time_margin),
    ):
        current_models[name] = {
            "joint_prediction": _joint_metrics(
                exact_test_q, q_value[test_random]
            ),
            "predicted_q_static_vs_exact_q_static_minimum": _minimum_metrics(
                exact_static_minimum[test_random, None, :],
                margin_value[test_random, None, :],
            ),
            "artifact_output_shape": (
                np.asarray(flat_output_shape if name == "flat" else time_output_shape)
                .astype(int).tolist()
            ),
        }
    gate = config["decision_gate"]
    tests = {
        "artifact_outputs_are_51_by_7_joints": (
            current_models["flat"]["artifact_output_shape"] == [51, 7]
            and current_models["time"]["artifact_output_shape"] == [51, 7]
        ),
        "k0_exact_q_recomposition": float(
            test["exact_q_FK_phi_recomposition"]["trace"][
                "k0_maximum_absolute_error_m"
            ]
        ) <= float(gate["maximum_k0_exact_q_recomposition_absolute_error_m"]),
        "exact_q_boundary_RMSE": geometry_boundary
        <= float(gate["maximum_test_exact_q_minimum_boundary_RMSE_m"]),
        "exact_q_zero_false_safe": int(
            test["exact_q_FK_phi_recomposition"]["minimum"][
                "false_safe_action_count"
            ]
        ) <= int(gate["maximum_test_exact_q_false_safe_action_count"]),
        "exact_q_safe_recall": float(
            test["exact_q_FK_phi_recomposition"]["minimum"][
                "exact_safe_action_recall"
            ]
        ) >= float(gate["minimum_test_exact_q_safe_recall"]),
        "predicted_q_has_false_safes": int(
            test["predicted_q_safety_error"]["minimum"][
                "false_safe_action_count"
            ]
        ) > 0,
        "joint_prediction_error_dominates_boundary_recomposition": ratio
        >= float(gate["minimum_joint_prediction_to_geometry_RMSE_ratio"]),
    }
    passed = bool(all(tests.values()))
    return {
        "formula_receipt": {
            "F_theta_output": "51_by_7_joint_configuration_trace_rad",
            "p_i": "MuJoCo_sim_forward_then_L5_L7_ellipsoid_world_transforms",
            "phi": "minimum_union_support_gap_to_fixed_k0_exact_box_obstacle",
            "QP_or_h_linearization_used": False,
        },
        "split_metrics": split_metrics,
        "current_structured_models": current_models,
        "decomposition_identity_maximum_absolute_error_m": float(
            np.max(np.abs(identity_error))
        ),
        "test_joint_prediction_to_geometry_boundary_RMSE_ratio": float(ratio),
        "decision": {
            "gate_tests": tests,
            "joint_prediction_is_dominant": passed,
            "conclusion": (
                gate["interpretation_if_pass"] if passed
                else gate["interpretation_if_fail"]
            ),
            "QP_or_closed_loop_authorized": False,
        },
    }
