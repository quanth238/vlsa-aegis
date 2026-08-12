"""Multi-witness counterfactual action-field helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .shadow import _numpy


MULTI_WITNESS_SCHEMA = "vlsa_distal_multi_witness_counterfactual_e05.v1"


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def load_multi_witness_config(path: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    value = json.loads(raw)
    required = {
        "action_space",
        "case_ids",
        "claim_scope",
        "comparators",
        "field_estimation",
        "gate",
        "multi_witness",
        "prior_iterative_result",
        "protected_geometry",
        "protocol_id",
        "reference_continuation",
        "risk",
        "smooth_max",
        "state_protocol",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("multi-witness config keys differ")
    if value.get("protocol_id") != "vlsa-distal-multi-witness-counterfactual-e05-v1":
        raise ValueError("multi-witness protocol differs")
    if value.get("case_ids") != ["vlsa-t1-goal-ii-t0-e05"]:
        raise ValueError("multi-witness case differs")
    if value.get("action_space") != {
        "action_limit": 1.0,
        "corrected_dimensions": [0, 1, 2],
        "corrected_horizon": 5,
        "endpoint_preservation": False,
        "line_search_fractions": [1.0, 0.5, 0.25],
        "maximum_iterations": 10,
        "maximum_total_correction_l2_action": 1.0,
        "maximum_total_path_length_action": 1.0,
        "trust_radius_action": 0.1,
    }:
        raise ValueError("multi-witness action space differs")
    if value.get("field_estimation") != {
        "direction_seed": 26081231,
        "paired_direction_count_per_iteration": 32,
        "paired_perturbation_action": 0.05,
        "ridge": 1e-06,
    }:
        raise ValueError("multi-witness field schedule differs")
    if value.get("multi_witness") != {
        "epigraph_regularization_m_per_action2": 0.01,
        "maximum_witness_count": 8,
        "near_active_threshold_m": 0.005,
        "solver": "CLARABEL",
    }:
        raise ValueError("multi-witness optimizer differs")
    if value.get("smooth_max") != {"temperature_m": 0.002}:
        raise ValueError("smooth-max setting differs")
    if value.get("risk") != {
        "definition": "maximum_over_link_and_action_of_negative_ellipsoid_margin",
        "evaluation_horizon_actions": 20,
        "task_penalty_excluded": True,
    }:
        raise ValueError("multi-witness risk differs")
    if value.get("gate") != {
        "minimum_exact_clearance_m": 0.0,
        "paper_car_threshold_m": 0.001,
        "protected_raw_contact_count": 0,
    }:
        raise ValueError("multi-witness gate differs")
    output = json.loads(_canonical(value).decode("utf-8"))
    output["schema_version"] = MULTI_WITNESS_SCHEMA
    output["config_file_sha256"] = hashlib.sha256(raw).hexdigest()
    output["config_payload_sha256"] = hashlib.sha256(_canonical(value)).hexdigest()
    return output


def select_near_active_witnesses(
    clearance_trace: Any,
    *,
    maximum_count: int,
    threshold_m: float,
) -> list[dict[str, Any]]:
    """Select deterministic top link/time rows without collapsing their identity."""

    np = _numpy()
    trace = np.asarray(clearance_trace, dtype=np.float64)
    if trace.ndim != 2 or trace.shape[1] != 7 or not np.all(np.isfinite(trace)):
        raise ValueError("multi-witness clearance trace differs")
    count = int(maximum_count)
    if count < 2 or count > trace.size:
        raise ValueError("multi-witness count differs")
    flattened = trace.reshape(-1)
    order = np.lexsort((np.arange(flattened.size), flattened))
    worst = float(flattened[order[0]])
    eligible = [int(index) for index in order if float(flattened[index]) <= worst + float(threshold_m)]
    selected = eligible[:count]
    if len(selected) < min(2, count):
        selected = [int(index) for index in order[:count]]
    return [
        {
            "flat_index": index,
            "action_offset": int(index // trace.shape[1]),
            "ellipsoid_row": int(index % trace.shape[1]),
            "margin_m": float(flattened[index]),
        }
        for index in selected
    ]


def fit_clearance_rows(
    directions: Any,
    plus_traces: Any,
    minus_traces: Any,
    perturbation: float,
    ridge: float,
) -> dict[str, Any]:
    """Fit one 15D action gradient for every explicit link/time clearance row."""

    np = _numpy()
    design = np.asarray(directions, dtype=np.float64)
    positive = np.asarray(plus_traces, dtype=np.float64)
    negative = np.asarray(minus_traces, dtype=np.float64)
    if design.ndim != 2 or design.shape[1] != 15:
        raise ValueError("multi-witness direction matrix differs")
    if positive.shape != negative.shape or positive.ndim != 3:
        raise ValueError("multi-witness paired traces differ")
    if positive.shape[0] != design.shape[0] or positive.shape[2] != 7:
        raise ValueError("multi-witness trace dimensions differ")
    targets = ((positive - negative) / (2.0 * float(perturbation))).reshape(
        design.shape[0], -1
    )
    matrix = design.T.dot(design) + float(ridge) * np.eye(15, dtype=np.float64)
    gradients = np.linalg.solve(matrix, design.T.dot(targets)).T
    predictions = design.dot(gradients.T)
    rmse = np.sqrt(np.mean((predictions - targets) ** 2, axis=0))
    return {
        "gradients": gradients,
        "directional_targets": targets,
        "directional_predictions": predictions,
        "row_fit_rmse": rmse,
    }


def smooth_min_direction(clearances: Any, gradients: Any, temperature_m: float) -> dict[str, Any]:
    """Return the gradient of a stable smooth minimum of explicit clearances."""

    np = _numpy()
    values = np.asarray(clearances, dtype=np.float64).reshape(-1)
    rows = np.asarray(gradients, dtype=np.float64)
    if rows.shape != (values.size, 15):
        raise ValueError("smooth-min clearance rows differ")
    temperature = float(temperature_m)
    logits = -(values - float(np.min(values))) / temperature
    logits -= float(np.max(logits))
    weights = np.exp(logits)
    weights /= float(np.sum(weights))
    gradient = weights.dot(rows)
    return {"gradient": gradient, "weights": weights}


def solve_multi_witness_epigraph(
    margins: Any,
    gradients: Any,
    current_xyz: Any,
    current_correction: Any,
    *,
    action_limit: float,
    trust_radius: float,
    maximum_total_correction: float,
    regularization: float,
    solver: str,
) -> dict[str, Any]:
    """Coordinate all selected risks with a convex local epigraph problem."""

    np = _numpy()
    try:
        import cvxpy as cp
    except ImportError as error:
        raise RuntimeError("multi-witness epigraph requires CVXPY") from error
    values = np.asarray(margins, dtype=np.float64).reshape(-1)
    rows = np.asarray(gradients, dtype=np.float64)
    xyz = np.asarray(current_xyz, dtype=np.float64).reshape(15)
    correction = np.asarray(current_correction, dtype=np.float64).reshape(15)
    if rows.shape != (values.size, 15) or values.size < 2:
        raise ValueError("multi-witness epigraph rows differ")
    delta = cp.Variable(15)
    epigraph = cp.Variable()
    risk = -values
    risk_rows = -rows
    constraints = [
        risk + risk_rows @ delta <= epigraph,
        cp.norm(delta, 2) <= float(trust_radius),
        cp.norm(correction + delta, 2) <= float(maximum_total_correction),
        xyz + delta <= float(action_limit),
        xyz + delta >= -float(action_limit),
    ]
    problem = cp.Problem(
        cp.Minimize(epigraph + float(regularization) * cp.sum_squares(delta)),
        constraints,
    )
    problem.solve(solver=str(solver))
    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or delta.value is None:
        raise RuntimeError("multi-witness epigraph did not solve: %s" % problem.status)
    result = np.asarray(delta.value, dtype=np.float64).reshape(15)
    predicted = values + rows.dot(result)
    return {
        "delta": result,
        "status": str(problem.status),
        "objective": float(problem.value),
        "predicted_margins_m": predicted,
        "predicted_worst_margin_m": float(np.min(predicted)),
    }
