"""Inference-only smooth waypoint search through the terminal pi0.5 flow.

The diagnostic parameterizes a five-action correction with two cubic-Bezier
control points.  Both endpoints are fixed at zero, so the generated action
displacements sum to zero before the remaining frozen flow steps.  No
simulator transition is part of this module.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


CONFIG_SCHEMA = "vlsa_late_flow_waypoint_search.v1"
RESULT_SCHEMA = "vlsa_late_flow_waypoint_search_result.v1"
VALIDATION_SCHEMA = "vlsa_late_flow_waypoint_search_validation.v1"


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def payload_sha256(value: Mapping[str, Any], key: str) -> str:
    public = dict(value)
    public.pop(key, None)
    return hashlib.sha256(canonical(public)).hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    value = json.loads(raw)
    if value.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("late-flow waypoint config schema differs")
    if value.get("protocol_id") != "vlsa-late-flow-waypoint-search-v1":
        raise ValueError("late-flow waypoint protocol differs")
    source = value.get("source_query", {})
    if not (
        value.get("case_id") == "vlsa-t1-goal-ii-t0-e05"
        and source.get("query_index") == 2
        and source.get("replay_action_count") == 10
        and source.get("terminal_reason") == "all_predicted_unsafe_abstention"
    ):
        raise ValueError("late-flow waypoint source query differs")
    search = value.get("waypoint_search", {})
    if not (
        search.get("control_point_count") == 2
        and search.get("control_dimension") == 6
        and search.get("parameterization")
        == "zero_endpoint_cubic_bezier_cumulative_offset"
        and search.get("unique_candidate_count") == 505
        and search.get("nonzero_candidate_count") == 504
        and search.get("server_batch_count") == 42
        and search.get("branches_per_batch") == 13
        and search.get("nonzero_branches_per_batch") == 12
        and float(search.get("translation_scale_m_per_action_unit", 0.0))
        == 0.05
        and float(search.get("trust_region_action", 0.0)) == 0.25
        and search.get("sampling_seed") == 2026081701
    ):
        raise ValueError("late-flow waypoint search differs")
    scoring = value.get("scoring", {})
    if not (
        scoring.get("physical_constraint_rows") == [1, 2, 3, 4]
        and scoring.get("diagnostic_proxy_rows") == [0]
        and scoring.get("audit_rows") == [5, 6]
        and float(scoring.get("predicted_safe_threshold", 1.0)) == 0.0
        and scoring.get("selection")
        == "predicted_physical_safe_then_minimum_correction_then_smoothness"
    ):
        raise ValueError("late-flow waypoint scoring differs")
    forbidden = value.get("forbidden", {})
    if forbidden != {
        "new_data_collection": True,
        "model_training": True,
        "simulator_candidate_rollout": True,
        "action_execution": True,
        "QP": True,
        "released_AEGIS_EE_QP": True,
    }:
        raise ValueError("late-flow waypoint forbidden settings differ")
    output = json.loads(canonical(value))
    output["config_file_sha256"] = hashlib.sha256(raw).hexdigest()
    output["config_payload_sha256"] = hashlib.sha256(canonical(value)).hexdigest()
    return output


def bezier_action_basis(
    *, action_count: int = 5, translation_scale_m: float = 0.05,
) -> list[list[float]]:
    """Map two physical control points to five action displacements."""

    if int(action_count) != 5:
        raise ValueError("late-flow waypoint action count differs")
    scale = float(translation_scale_m)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("late-flow waypoint translation scale differs")

    cumulative = []
    for index in range(action_count + 1):
        t = float(index) / float(action_count)
        cumulative.append((
            3.0 * (1.0 - t) ** 2 * t,
            3.0 * (1.0 - t) * t ** 2,
        ))
    basis = []
    for index in range(action_count):
        basis.append([
            (cumulative[index + 1][column] - cumulative[index][column])
            / scale
            for column in range(2)
        ])
    if any(abs(sum(row[column] for row in basis)) > 1.0e-12 for column in range(2)):
        raise ValueError("late-flow waypoint basis does not preserve endpoint")
    return basis


def obstacle_frame(outward_world: Sequence[float]) -> list[list[float]]:
    """Return deterministic outward/tangent/up-like orthonormal columns."""

    import numpy as np

    outward = np.asarray(outward_world, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(outward))
    if not np.isfinite(norm) or norm <= 1.0e-12:
        raise ValueError("late-flow waypoint outward direction differs")
    outward /= norm
    reference = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(outward @ reference)) > 0.95:
        reference = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    tangent = np.cross(reference, outward)
    tangent /= np.linalg.norm(tangent)
    third = np.cross(outward, tangent)
    third /= np.linalg.norm(third)
    frame = np.stack([outward, tangent, third], axis=1)
    if float(np.max(np.abs(frame.T @ frame - np.eye(3)))) > 1.0e-12:
        raise ValueError("late-flow waypoint frame differs")
    return frame.tolist()


def waypoint_batches(
    *, outward_world: Sequence[float], seed: int, trust_region_action: float,
    translation_scale_m: float = 0.05,
) -> list[dict[str, Any]]:
    """Return 42 fixed server batches containing 504 unique smooth routes."""

    import numpy as np

    if int(seed) != 2026081701:
        raise ValueError("late-flow waypoint sampling seed differs")
    trust = float(trust_region_action)
    if not math.isfinite(trust) or trust != 0.25:
        raise ValueError("late-flow waypoint trust region differs")
    basis = np.asarray(
        bezier_action_basis(translation_scale_m=translation_scale_m),
        dtype=np.float64,
    )
    frame = np.asarray(obstacle_frame(outward_world), dtype=np.float64)
    rng = np.random.RandomState(int(seed))

    local_controls = []
    for coordinate in range(6):
        for sign in (-1.0, 1.0):
            value = np.zeros((2, 3), dtype=np.float64)
            value.reshape(6)[coordinate] = sign
            local_controls.append(value)
    while len(local_controls) < 504:
        raw = rng.normal(size=(2, 3))
        norm = float(np.linalg.norm(raw))
        if norm <= 1.0e-12:
            continue
        local_controls.append(raw / norm)

    batches = []
    record_index = 0
    radius_fractions = (0.25, 0.5, 0.75, 1.0)
    for batch_index in range(42):
        residuals = np.zeros((13, 10, 3), dtype=np.float64)
        records = []
        for branch_index in range(1, 13):
            local = np.asarray(local_controls[record_index], dtype=np.float64)
            world_controls = local @ frame.T
            raw_correction = basis @ world_controls
            maximum = float(np.max(np.abs(raw_correction)))
            if maximum <= 1.0e-12:
                raise ValueError("late-flow waypoint correction degenerates")
            target = trust * radius_fractions[record_index % len(radius_fractions)]
            correction = raw_correction * (target / maximum)
            residuals[branch_index, :5] = correction
            second = np.diff(correction, n=2, axis=0)
            records.append({
                "candidate_index": record_index + 1,
                "candidate_name": f"waypoint_{record_index + 1:04d}",
                "branch_index": branch_index,
                "control_points_world_m": (
                    world_controls * (target / maximum)
                ).tolist(),
                "preterminal_correction_l2_action": float(
                    np.linalg.norm(correction)
                ),
                "preterminal_correction_linf_action": float(
                    np.max(np.abs(correction))
                ),
                "preterminal_endpoint_sum_action": np.sum(
                    correction, axis=0,
                ).tolist(),
                "preterminal_smoothness": float(np.linalg.norm(second)),
            })
            record_index += 1
        batches.append({
            "batch_index": batch_index,
            "residuals": residuals.tolist(),
            "records": records,
        })
    if record_index != 504:
        raise ValueError("late-flow waypoint population differs")
    return batches


def summarize_predictions(
    *, nominal_risk: Sequence[float], records: Sequence[Mapping[str, Any]],
    physical_rows: Sequence[int], diagnostic_rows: Sequence[int],
    threshold: float,
) -> dict[str, Any]:
    """Apply the preregistered safety-first lexicographic selector."""

    import numpy as np

    nominal = np.asarray(nominal_risk, dtype=np.float64)
    if nominal.shape != (7,) or not np.all(np.isfinite(nominal)):
        raise ValueError("late-flow waypoint nominal risk differs")
    physical = tuple(int(row) for row in physical_rows)
    diagnostic = tuple(int(row) for row in diagnostic_rows)
    if physical != (1, 2, 3, 4) or diagnostic != (0,):
        raise ValueError("late-flow waypoint row set differs")
    limit = float(threshold)
    candidates = [{
        "candidate_index": 0,
        "candidate_name": "nominal",
        "predicted_risk_by_row": nominal.tolist(),
        "maximum_physical_risk": float(max(nominal[list(physical)])),
        "maximum_proxy_inclusive_risk": float(
            max(nominal[list(diagnostic + physical)])
        ),
        "effective_correction_l2_action": 0.0,
        "effective_smoothness": 0.0,
    }]
    for expected_index, source in enumerate(records, start=1):
        risk = np.asarray(source["predicted_risk_by_row"], dtype=np.float64)
        if (
            int(source["candidate_index"]) != expected_index
            or risk.shape != (7,) or not np.all(np.isfinite(risk))
        ):
            raise ValueError("late-flow waypoint prediction record differs")
        candidates.append({
            **dict(source),
            "maximum_physical_risk": float(max(risk[list(physical)])),
            "maximum_proxy_inclusive_risk": float(
                max(risk[list(diagnostic + physical)])
            ),
        })
    safe = [
        row for row in candidates
        if float(row["maximum_physical_risk"]) <= limit
    ]
    selected = None if not safe else min(safe, key=lambda row: (
        float(row["effective_correction_l2_action"]),
        float(row["effective_smoothness"]),
        int(row["candidate_index"]),
    ))
    best_physical = min(candidates, key=lambda row: (
        float(row["maximum_physical_risk"]),
        float(row["effective_correction_l2_action"]),
    ))
    best_proxy = min(candidates, key=lambda row: (
        float(row["maximum_proxy_inclusive_risk"]),
        float(row["effective_correction_l2_action"]),
    ))
    return {
        "candidate_count": len(candidates),
        "predicted_physical_safe_count": len(safe),
        "predicted_proxy_inclusive_safe_count": sum(
            float(row["maximum_proxy_inclusive_risk"]) <= limit
            for row in candidates
        ),
        "selected_physical_safe_minimum_change": selected,
        "minimum_physical_risk_candidate": best_physical,
        "minimum_proxy_inclusive_risk_candidate": best_proxy,
        "minimum_predicted_risk_by_row": [
            min(float(row["predicted_risk_by_row"][index]) for row in candidates)
            for index in range(7)
        ],
    }
