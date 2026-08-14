"""Pure contracts for the bounded row-targeted active-boundary search."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


CONFIG_SCHEMA = "vlsa_distal_active_boundary_search.v1"
RESULT_SCHEMA = "vlsa_distal_active_boundary_search_result.v1"


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def load_config(path: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    value = json.loads(raw)
    required = {
        "schema_version", "protocol_id", "claim_scope", "source",
        "target_rows", "target_jobs", "finite_search", "label_contract",
        "forbidden",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("active boundary search config keys differ")
    if value["schema_version"] != CONFIG_SCHEMA:
        raise ValueError("active boundary search schema differs")
    if value["protocol_id"] != "vlsa-distal-active-boundary-search-v1":
        raise ValueError("active boundary search protocol differs")
    if value["target_rows"] != [4, 5, 6]:
        raise ValueError("active boundary search rows differ")
    jobs = value["target_jobs"]
    if [item["array_index"] for item in jobs] != list(range(4)):
        raise ValueError("active boundary search array differs")
    if {item["case_index"] for item in jobs} != {10, 14}:
        raise ValueError("active boundary search cases differ")
    if [item["temporal_profile"] for item in jobs] != [
        "constant", "front_loaded", "constant", "front_loaded"
    ]:
        raise ValueError("active boundary search profiles differ")
    search = value["finite_search"]
    if search != {
        "action_coordinates": "released_AEGIS_output_before_OSC",
        "spatial_basis": ["normal", "tangent_up", "tangent_side"],
        "coefficient_grid": [-1, 0, 1],
        "exclude_all_zero": True,
        "spatial_direction_count": 26,
        "correction_l2_action": 2.0,
        "action_limit": 1.0,
        "include_nominal": True,
        "candidate_count_per_job": 27,
        "endpoint_preservation": False,
        "preserve_rotation_and_gripper": True,
        "robust_boundary_margin_m": 0.001,
        "interpretation": (
            "finite_witnesses_only_not_extrema_over_the_continuous_action_region"
        ),
    }:
        raise ValueError("active boundary search finite search differs")
    output = json.loads(canonical(value).decode("utf-8"))
    output["config_file_sha256"] = hashlib.sha256(raw).hexdigest()
    output["config_payload_sha256"] = hashlib.sha256(canonical(value)).hexdigest()
    return output


def target_job(config: Mapping[str, Any], array_index: int) -> dict[str, Any]:
    matches = [
        item for item in config["target_jobs"]
        if int(item["array_index"]) == int(array_index)
    ]
    if len(matches) != 1:
        raise ValueError("active boundary search job differs")
    return dict(matches[0])


def candidate_definitions(
    nominal_actions: Sequence[Sequence[float]],
    local_frame: Mapping[str, Sequence[float]],
    config: Mapping[str, Any],
    temporal_profile_name: str,
) -> list[dict[str, Any]]:
    """Cover local direction space with the normalized {-1,0,1}^3 grid."""

    import numpy as np
    from main.multilink_ellipsoid.query_action_risk import temporal_profile

    nominal = np.asarray(nominal_actions, dtype=np.float64)
    if nominal.shape != (5, 7) or not np.all(np.isfinite(nominal)):
        raise ValueError("active boundary search nominal differs")
    search = config["finite_search"]
    profile = np.asarray(temporal_profile(temporal_profile_name), dtype=np.float64)
    basis = np.stack([
        np.asarray(local_frame[name], dtype=np.float64)
        for name in search["spatial_basis"]
    ], axis=1)
    if basis.shape != (3, 3) or not np.all(np.isfinite(basis)):
        raise ValueError("active boundary search frame differs")
    if float(np.max(np.abs(basis.T @ basis - np.eye(3)))) > 1.0e-12:
        raise ValueError("active boundary search frame is not orthonormal")
    output = [{
        "name": "nominal",
        "order": 0,
        "direction": None,
        "spatial_coefficients": None,
        "sign": 0,
        "temporal_profile": None,
        "requested_correction_l2_action": 0.0,
        "applied_correction_l2_action": 0.0,
        "clipped": False,
        "actions": nominal.tolist(),
    }]
    grid = search["coefficient_grid"]
    for coefficients in itertools.product(grid, repeat=3):
        if all(int(item) == 0 for item in coefficients):
            continue
        coefficient_vector = np.asarray(coefficients, dtype=np.float64)
        direction = basis @ coefficient_vector
        direction /= np.linalg.norm(direction)
        requested = (
            profile[:, None] * direction[None, :]
            * float(search["correction_l2_action"])
        )
        actions = nominal.copy()
        actions[:, :3] = np.clip(
            actions[:, :3] + requested,
            -float(search["action_limit"]),
            float(search["action_limit"]),
        )
        applied = actions[:, :3] - nominal[:, :3]
        coefficient_name = "_".join(
            "p%d" % int(item) if int(item) > 0 else
            "m%d" % abs(int(item)) if int(item) < 0 else "z0"
            for item in coefficients
        )
        output.append({
            "name": "grid_%s_%s_r%s" % (
                coefficient_name, temporal_profile_name,
                search["correction_l2_action"],
            ),
            "order": len(output),
            "direction": direction.tolist(),
            "spatial_coefficients": [int(item) for item in coefficients],
            "sign": None,
            "temporal_profile": temporal_profile_name,
            "requested_correction_l2_action": float(
                search["correction_l2_action"]
            ),
            "applied_correction_l2_action": float(np.linalg.norm(applied)),
            "clipped": bool(np.max(np.abs(applied - requested)) > 1.0e-12),
            "actions": actions.tolist(),
        })
    if len(output) != int(search["candidate_count_per_job"]):
        raise ValueError("active boundary search candidate count differs")
    return output


def summarize_target_rows(
    candidates: Sequence[Mapping[str, Any]], target_rows: Sequence[int],
    robust_margin_m: float,
) -> list[dict[str, Any]]:
    known = [
        item for item in candidates
        if item["terminal_status"] != "UNKNOWN_TIMEOUT"
    ]
    output = []
    for row in target_rows:
        ranked = sorted(known, key=lambda item: float(item["combined_risk"][row]))
        maximum = None if not ranked else ranked[-1]
        minimum = None if not ranked else ranked[0]
        output.append({
            "row": int(row),
            "known_candidate_count": len(ranked),
            "witnessed_minimum_Q_m": None if minimum is None else float(
                minimum["combined_risk"][row]
            ),
            "minimum_candidate_name": None if minimum is None else minimum["name"],
            "witnessed_maximum_Q_m": None if maximum is None else float(
                maximum["combined_risk"][row]
            ),
            "maximum_candidate_name": None if maximum is None else maximum["name"],
            "robust_negative_side_observed": any(
                float(item["combined_risk"][row]) <= -float(robust_margin_m)
                for item in ranked
            ),
            "robust_positive_side_observed": any(
                float(item["combined_risk"][row]) >= float(robust_margin_m)
                for item in ranked
            ),
            "globally_safe_robust_negative_side_observed": any(
                bool(item["exact_safe"])
                and float(item["combined_risk"][row]) <= -float(robust_margin_m)
                for item in ranked
            ),
        })
    for row in output:
        row["useful_boundary_observed_within_job"] = bool(
            row["robust_positive_side_observed"]
            and row["globally_safe_robust_negative_side_observed"]
        )
    return output


def aggregate_search_rows(
    job_results: Sequence[Mapping[str, Any]], target_rows: Sequence[int]
) -> list[dict[str, Any]]:
    """Aggregate witnessed support across profiles without claiming extrema."""

    output = []
    for row in target_rows:
        observations = []
        for result in job_results:
            matches = [
                item for item in result["target_row_summary"]
                if int(item["row"]) == int(row)
            ]
            if len(matches) != 1:
                raise ValueError("active boundary aggregate row differs")
            observations.append((result["active_boundary_job"], matches[0]))
        known_maxima = [
            (float(item["witnessed_maximum_Q_m"]), job, item)
            for job, item in observations
            if item["witnessed_maximum_Q_m"] is not None
        ]
        maximum = None if not known_maxima else max(known_maxima, key=lambda item: item[0])
        useful_states = sorted({
            str(job["state_id"]) for job, item in observations
            if bool(item["useful_boundary_observed_combined"])
        })
        positive_states = sorted({
            str(job["state_id"]) for job, item in observations
            if bool(item["robust_positive_side_observed"])
        })
        output.append({
            "row": int(row),
            "job_count": len(observations),
            "state_count": len({str(job["state_id"]) for job, _ in observations}),
            "robust_positive_side_state_ids": positive_states,
            "useful_boundary_state_ids": useful_states,
            "useful_boundary_observed": bool(useful_states),
            "witnessed_maximum_Q_m": None if maximum is None else maximum[0],
            "maximum_witness": None if maximum is None else {
                "state_id": maximum[1]["state_id"],
                "split": maximum[1]["split"],
                "temporal_profile": maximum[1]["temporal_profile"],
                "candidate_name": maximum[2]["maximum_candidate_name"],
            },
        })
    return output
