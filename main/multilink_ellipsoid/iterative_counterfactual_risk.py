"""Pure-risk helpers for the decisive local counterfactual-field gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .counterfactual_field import endpoint_projection_matrix, unit_direction
from .shadow import _numpy


ITERATIVE_RISK_SCHEMA = "vlsa_distal_iterative_counterfactual_risk_e05.v1"


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def load_iterative_risk_config(path: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    value = json.loads(raw)
    required = {
        "action_space",
        "case_ids",
        "claim_scope",
        "comparators",
        "field_estimation",
        "gate",
        "prior_counterfactual_result",
        "protected_geometry",
        "protocol_id",
        "reference_continuation",
        "risk",
        "schema_version",
        "search_modes",
        "state_protocol",
        "task_selection",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("iterative-risk config keys differ")
    if value["schema_version"] != ITERATIVE_RISK_SCHEMA:
        raise ValueError("iterative-risk schema differs")
    if value["case_ids"] != ["vlsa-t1-goal-ii-t0-e05"]:
        raise ValueError("iterative-risk case differs")
    if value["action_space"] != {
        "action_limit": 1.0,
        "corrected_dimensions": [0, 1, 2],
        "corrected_horizon": 5,
        "total_trust_radii_action": [0.1, 0.25, 0.5, 1.0],
    }:
        raise ValueError("iterative-risk action space differs")
    if value["risk"] != {
        "definition": "maximum_over_link_and_action_of_negative_ellipsoid_margin",
        "positive_part_disabled": True,
        "task_penalty_excluded_from_field_target": True,
    }:
        raise ValueError("iterative-risk target differs")
    estimation = value["field_estimation"]
    if estimation["paired_direction_count_per_radius"] != 32:
        raise ValueError("iterative-risk direction count differs")
    if estimation["paired_perturbation_action"] != 0.05:
        raise ValueError("iterative-risk perturbation differs")
    if estimation["inner_step_action"] != 0.1 or estimation["maximum_iterations"] != 10:
        raise ValueError("iterative-risk iteration schedule differs")
    if estimation["line_search_fractions"] != [1.0, 0.5, 0.25]:
        raise ValueError("iterative-risk line search differs")
    output = json.loads(_canonical(value).decode("utf-8"))
    output["config_file_sha256"] = hashlib.sha256(raw).hexdigest()
    output["config_payload_sha256"] = hashlib.sha256(_canonical(value)).hexdigest()
    return output


def active_witness(clearance_trace: Any) -> dict[str, Any]:
    np = _numpy()
    trace = np.asarray(clearance_trace, dtype=np.float64)
    if trace.ndim != 2 or trace.shape[1] != 7 or not np.all(np.isfinite(trace)):
        raise ValueError("iterative-risk clearance trace differs")
    flat = int(np.argmin(trace))
    step, row = np.unravel_index(flat, trace.shape)
    return {
        "action_offset": int(step),
        "ellipsoid_row": int(row),
        "margin_m": float(trace[step, row]),
    }


def fit_safety_direction(
    directions: Any,
    plus_hard_margins: Any,
    minus_hard_margins: Any,
    perturbation: float,
    ridge: float,
) -> dict[str, Any]:
    """Fit gradient of hard clearance, equivalent to negative risk gradient."""

    np = _numpy()
    design = np.asarray(directions, dtype=np.float64)
    positive = np.asarray(plus_hard_margins, dtype=np.float64).reshape(-1)
    negative = np.asarray(minus_hard_margins, dtype=np.float64).reshape(-1)
    if design.ndim != 2 or design.shape[1] != 15:
        raise ValueError("iterative-risk direction matrix differs")
    if positive.shape != (design.shape[0],) or negative.shape != positive.shape:
        raise ValueError("iterative-risk paired targets differ")
    target = (positive - negative) / (2.0 * float(perturbation))
    matrix = design.T.dot(design) + float(ridge) * np.eye(15, dtype=np.float64)
    gradient = np.linalg.solve(matrix, design.T.dot(target))
    prediction = design.dot(gradient)
    return {
        "gradient": gradient,
        "unit_direction": unit_direction(gradient),
        "directional_targets": target,
        "directional_predictions": prediction,
        "fit_rmse": float(np.sqrt(np.mean((prediction - target) ** 2))),
    }


def project_mode_direction(
    vector: Any,
    current_xyz: Any,
    *,
    action_limit: float,
    preserve_endpoint: bool,
) -> Any:
    np = _numpy()
    value = np.asarray(vector, dtype=np.float64).reshape(15)
    xyz = np.asarray(current_xyz, dtype=np.float64).reshape(15)
    headroom = float(action_limit) - np.abs(xyz)
    if float(np.min(headroom)) < -1.0e-10:
        raise ValueError("iterative-risk current action exceeds bounds")
    fixed = headroom <= 1.0e-10
    projection = (
        endpoint_projection_matrix(fixed)
        if preserve_endpoint
        else np.diag((~fixed).astype(np.float64))
    )
    return unit_direction(projection.dot(value))


def feasible_correction(
    base_xyz: Any,
    correction: Any,
    *,
    radius: float,
    action_limit: float,
    preserve_endpoint: bool,
) -> bool:
    np = _numpy()
    base = np.asarray(base_xyz, dtype=np.float64).reshape(5, 3)
    delta = np.asarray(correction, dtype=np.float64).reshape(5, 3)
    if not np.all(np.isfinite(delta)):
        return False
    if float(np.linalg.norm(delta)) > float(radius) + 1.0e-10:
        return False
    if float(np.max(np.abs(base + delta))) > float(action_limit) + 1.0e-10:
        return False
    if preserve_endpoint and float(np.max(np.abs(np.sum(delta, axis=0)))) > 1.0e-9:
        return False
    return True


def correction_selection_score(
    hard_margin_m: float,
    terminal_eef_error_m: float,
    correction: Any,
    config: dict[str, Any],
) -> float:
    np = _numpy()
    delta = np.asarray(correction, dtype=np.float64).reshape(5, 3)
    smoothness = float(np.sum((delta[1:] - delta[:-1]) ** 2))
    task = config["task_selection"]
    return float(
        float(hard_margin_m)
        - float(task["terminal_eef_weight_per_m"]) * float(terminal_eef_error_m) ** 2
        - float(task["correction_smoothness_weight_m_per_action2"]) * smoothness
    )
