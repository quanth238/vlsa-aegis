"""Paired-rollout action field for the local E05 mechanism gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Optional

from .shadow import _numpy


COUNTERFACTUAL_FIELD_SCHEMA = "vlsa_distal_counterfactual_field_e05.v1"


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def load_counterfactual_field_config(path: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    value = json.loads(raw)
    required = {
        "action_space",
        "case_ids",
        "claim_scope",
        "comparators",
        "gate",
        "protected_geometry",
        "protocol_id",
        "reference_continuation",
        "sampling",
        "schema_version",
        "score",
        "state_protocol",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("counterfactual-field config keys differ")
    if value["schema_version"] != COUNTERFACTUAL_FIELD_SCHEMA:
        raise ValueError("counterfactual-field schema differs")
    if value["case_ids"] != ["vlsa-t1-goal-ii-t0-e05"]:
        raise ValueError("counterfactual-field case differs")
    if value["state_protocol"] != {
        "archived_prefix_last_step": 181,
        "corrected_action_steps": [182, 183, 184, 185, 186],
        "fixed_evaluation_steps": list(range(182, 202)),
        "known_reference_collision_step": 197,
    }:
        raise ValueError("counterfactual-field state protocol differs")
    sampling = value["sampling"]
    if sampling["paired_direction_count"] != 64:
        raise ValueError("counterfactual-field pair count differs")
    if sampling["fit_direction_count"] != 48:
        raise ValueError("counterfactual-field fit count differs")
    if sampling["heldout_direction_count"] != 16:
        raise ValueError("counterfactual-field heldout count differs")
    if sampling["paired_perturbation_action"] != 0.1:
        raise ValueError("counterfactual-field perturbation differs")
    if sampling["comparison_correction_l2_action"] != 0.1:
        raise ValueError("counterfactual-field comparison radius differs")
    output = json.loads(_canonical(value).decode("utf-8"))
    output["config_file_sha256"] = hashlib.sha256(raw).hexdigest()
    output["config_payload_sha256"] = hashlib.sha256(_canonical(value)).hexdigest()
    return output


def endpoint_projection_matrix(fixed_coordinates: Optional[Any] = None) -> Any:
    """Project five XYZ corrections onto sum-over-time zero and fixed zeros."""

    np = _numpy()
    constraints = []
    for dimension in range(3):
        row = np.zeros(15, dtype=np.float64)
        row[dimension::3] = 1.0
        constraints.append(row)
    if fixed_coordinates is not None:
        fixed = np.asarray(fixed_coordinates, dtype=bool).reshape(15)
        for index in np.flatnonzero(fixed):
            row = np.zeros(15, dtype=np.float64)
            row[int(index)] = 1.0
            constraints.append(row)
    matrix = np.asarray(constraints, dtype=np.float64)
    projection = np.eye(15, dtype=np.float64) - matrix.T.dot(
        np.linalg.pinv(matrix.dot(matrix.T))
    ).dot(matrix)
    return 0.5 * (projection + projection.T)


def paired_chunk_directions(
    count: int,
    seed: int,
    nominal_xyz: Any,
    radius: float,
    action_limit: float,
    preserve_endpoint: bool = True,
) -> Any:
    """Sample smooth unit directions feasible with both signs and zero endpoint."""

    np = _numpy()
    nominal = np.asarray(nominal_xyz, dtype=np.float64)
    if nominal.shape != (5, 3) or not np.all(np.isfinite(nominal)):
        raise ValueError("counterfactual nominal action shape differs")
    if count < 1 or radius <= 0.0 or action_limit <= 0.0:
        raise ValueError("counterfactual direction request is invalid")
    headroom = float(action_limit) - np.abs(nominal.reshape(15))
    if np.min(headroom) < -1.0e-10:
        raise ValueError("counterfactual nominal action exceeds bounds")
    fixed = headroom <= 1.0e-10
    projection = (
        endpoint_projection_matrix(fixed)
        if bool(preserve_endpoint)
        else np.diag((~fixed).astype(np.float64))
    )
    rng = np.random.RandomState(int(seed))
    output = []
    attempts = 0
    while len(output) < int(count):
        attempts += 1
        if attempts > int(count) * 20000:
            raise ValueError("could not sample paired endpoint-preserving directions")
        raw = rng.normal(size=(5, 3))
        smooth = raw.copy()
        smooth[1:-1] = 0.25 * raw[:-2] + 0.5 * raw[1:-1] + 0.25 * raw[2:]
        direction = projection.dot(smooth.reshape(15))
        norm = float(np.linalg.norm(direction))
        if norm <= 1.0e-10:
            continue
        direction /= norm
        plus = nominal.reshape(15) + float(radius) * direction
        minus = nominal.reshape(15) - float(radius) * direction
        if max(float(np.max(np.abs(plus))), float(np.max(np.abs(minus)))) > float(action_limit) + 1.0e-10:
            continue
        if any(abs(float(np.dot(direction, prior))) > 0.999999 for prior in output):
            continue
        output.append(direction)
    return np.asarray(output, dtype=np.float64)


def fit_paired_field(
    directions: Any,
    plus_scores: Any,
    minus_scores: Any,
    perturbation: float,
    fit_count: int,
    ridge: float,
) -> dict[str, Any]:
    """Fit a local utility gradient and audit unseen paired directions."""

    np = _numpy()
    design = np.asarray(directions, dtype=np.float64)
    positive = np.asarray(plus_scores, dtype=np.float64).reshape(-1)
    negative = np.asarray(minus_scores, dtype=np.float64).reshape(-1)
    if design.ndim != 2 or design.shape[1] != 15:
        raise ValueError("counterfactual field design shape differs")
    if positive.shape != (design.shape[0],) or negative.shape != positive.shape:
        raise ValueError("counterfactual field score shape differs")
    if not 1 <= int(fit_count) < design.shape[0]:
        raise ValueError("counterfactual field split differs")
    target = (positive - negative) / (2.0 * float(perturbation))
    train = design[: int(fit_count)]
    train_target = target[: int(fit_count)]
    matrix = train.T.dot(train) + float(ridge) * np.eye(15, dtype=np.float64)
    gradient = np.linalg.solve(matrix, train.T.dot(train_target))
    heldout = design[int(fit_count) :]
    heldout_target = target[int(fit_count) :]
    heldout_prediction = heldout.dot(gradient)
    centered_target = heldout_target - float(np.mean(heldout_target))
    denominator = float(np.sum(centered_target * centered_target))
    r2 = float("nan") if denominator <= 1.0e-20 else float(
        1.0 - np.sum((heldout_prediction - heldout_target) ** 2) / denominator
    )
    if float(np.std(heldout_target)) <= 1.0e-12 or float(np.std(heldout_prediction)) <= 1.0e-12:
        correlation = float("nan")
    else:
        correlation = float(np.corrcoef(heldout_target, heldout_prediction)[0, 1])
    sign_mask = np.abs(heldout_target) > 1.0e-8
    sign_accuracy = (
        float("nan")
        if not np.any(sign_mask)
        else float(np.mean(np.sign(heldout_prediction[sign_mask]) == np.sign(heldout_target[sign_mask])))
    )
    return {
        "gradient": gradient,
        "directional_targets": target,
        "heldout_predictions": heldout_prediction,
        "heldout_targets": heldout_target,
        "heldout_rmse": float(np.sqrt(np.mean((heldout_prediction - heldout_target) ** 2))),
        "heldout_r2": r2,
        "heldout_pearson": correlation,
        "heldout_sign_accuracy": sign_accuracy,
    }


def unit_direction(vector: Any) -> Any:
    np = _numpy()
    value = np.asarray(vector, dtype=np.float64).reshape(15)
    norm = float(np.linalg.norm(value))
    if not np.all(np.isfinite(value)) or norm <= 1.0e-12:
        raise ValueError("counterfactual field direction is degenerate")
    return value / norm


def project_direction(vector: Any, direction_basis: Any) -> Any:
    """Project a comparator into the exact sampled correction subspace."""

    np = _numpy()
    value = np.asarray(vector, dtype=np.float64).reshape(15)
    basis = np.asarray(direction_basis, dtype=np.float64)
    if basis.ndim != 2 or basis.shape[1] != 15:
        raise ValueError("counterfactual direction basis shape differs")
    projected = basis.T.dot(np.linalg.pinv(basis.dot(basis.T))).dot(basis).dot(value)
    return unit_direction(projected)


def matched_random_p_value(learned_gain: float, random_gains: Any) -> float:
    np = _numpy()
    gains = np.asarray(random_gains, dtype=np.float64).reshape(-1)
    if gains.size < 1 or not np.all(np.isfinite(gains)):
        raise ValueError("counterfactual random gains are invalid")
    return float((1 + np.sum(gains >= float(learned_gain))) / (1 + gains.size))
