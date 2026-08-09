"""Frozen candidate families for the E05 multi-step action-chunk oracle."""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA = "vlsa_distal_multistep_chunk_oracle_e05.v1"
RESULT_SCHEMA = "vlsa_distal_multistep_chunk_oracle_e05_result.v1"
VALIDATION_SCHEMA = "vlsa_distal_multistep_chunk_oracle_e05_validation.v1"


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def load_chunk_oracle_config(path: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    try:
        config = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("chunk-oracle config is invalid JSON") from error
    required = {
        "schema_version", "protocol_id", "case_id", "claim_scope", "source",
        "geometry", "candidate_family", "arms", "verification",
        "decision_gate",
    }
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError("chunk-oracle config keys differ")
    if (
        config["schema_version"] != SCHEMA
        or config["protocol_id"] != "vlsa-distal-multistep-chunk-oracle-e05-v1"
        or config["case_id"] != "vlsa-t1-goal-ii-t0-e05"
    ):
        raise ValueError("chunk-oracle identity differs")
    source = config["source"]
    if set(source) != {
        "table1_population_manifest_sha256", "archived_relative_path",
        "archived_file_sha256", "archived_payload_sha256",
        "executed_sequence_sha256", "action_count", "start_action_step",
        "prior_full_bound_job_id", "prior_full_bound_result_sha256",
        "prior_full_bound_validation_sha256",
        "prior_full_bound_result_payload_sha256",
    } or source["action_count"] != 237 or source["start_action_step"] != 185:
        raise ValueError("chunk-oracle source differs")
    if config["geometry"].get("constraint_count") != 7 or config["geometry"].get("clearance_target_m") != 0.0:
        raise ValueError("chunk-oracle geometry differs")
    family = config["candidate_family"]
    if family != {
        "action_limit": 1.0,
        "grid_points_per_dimension": 9,
        "one_shot_expected_candidate_count": 731,
        "distributed_expected_candidate_count": 730,
        "one_shot": "replace_only_first_chunk_translation_then_keep_immutable_aegis_suffix",
        "distributed": "add_weighted_residual_without_clipping_inside_jointly_feasible_delta_box_and_include_zero_residual",
        "two_step_residual_weights": [1.0, -1.0],
        "five_step_residual_weights": [1.0, 0.5, 0.0, -0.5, -1.0],
        "endpoint_preservation": "sum_of_translation_corrections_is_zero_per_dimension",
        "normalization": "normalized_action_displacements_use_scale_only_no_mean_subtraction",
    }:
        raise ValueError("chunk-oracle candidate family differs")
    expected_arms = [
        {"arm_id": "two_step_one_shot", "horizon": 2, "family": "one_shot"},
        {"arm_id": "two_step_distributed", "horizon": 2, "family": "distributed"},
        {"arm_id": "five_step_one_shot", "horizon": 5, "family": "one_shot"},
        {"arm_id": "five_step_distributed", "horizon": 5, "family": "distributed"},
    ]
    if config["arms"] != expected_arms:
        raise ValueError("chunk-oracle arms differ")
    if config["verification"] != {
        "raw_contact_distance_threshold_m": 0.0,
        "maximum_per_step_obstacle_l1_displacement_m": 0.0001,
        "clone_state_tolerance": 1.0e-10,
        "require_all_internal_osc_substeps": True,
        "execute_smallest_verified_safe_chunk_once": True,
    }:
        raise ValueError("chunk-oracle verification differs")
    if config["decision_gate"] != {
        "baseline": "immutable_two_and_five_step_chunks_must_reproduce_proxy_unsafe_crossing",
        "chunk_recovery": "at_least_one_registered_chunk_has_all_seven_exact_substep_gaps_nonnegative_zero_raw_contact_and_at_most_0p1mm_per_step_obstacle_motion",
        "execution_fidelity": "selected_safe_chunk_execution_matches_every_cloned_transition",
        "larger_region_collection": "authorized_if_any_arm_passes_chunk_recovery_and_execution_fidelity",
        "neural_training": "not_authorized_until_grouped_chunk_dataset_gate_passes",
        "closed_loop": "not_authorized_by_single_state_chunk_oracle",
    }:
        raise ValueError("chunk-oracle decision gate differs")
    output = json.loads(_canonical(config).decode("utf-8"))
    output["config_file_sha256"] = hashlib.sha256(raw).hexdigest()
    output["config_payload_sha256"] = hashlib.sha256(_canonical(config)).hexdigest()
    return output


def _axis(config: Mapping[str, Any]) -> Any:
    import numpy as np

    family = config["candidate_family"]
    return np.linspace(
        -float(family["action_limit"]),
        float(family["action_limit"]),
        int(family["grid_points_per_dimension"]),
    )


def one_shot_chunks(nominal_chunk: Any, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    import numpy as np

    nominal = np.asarray(nominal_chunk, dtype=np.float64)
    if nominal.ndim != 2 or nominal.shape[1] != 7:
        raise ValueError("one-shot nominal chunk differs")
    candidates = [
        ("global_first_action_grid", np.asarray(value, dtype=np.float64))
        for value in itertools.product(_axis(config), repeat=3)
    ]
    candidates.extend([
        ("nominal", nominal[0, :3].copy()),
        ("reverse_first_action", -nominal[0, :3].copy()),
    ])
    seen = set(); output = []
    for source, xyz in candidates:
        key = tuple(float(item) for item in xyz)
        if key in seen:
            continue
        seen.add(key)
        chunk = nominal.copy(); chunk[0, :3] = xyz
        correction = chunk[:, :3] - nominal[:, :3]
        output.append({
            "source": source,
            "parameter": xyz,
            "chunk": chunk,
            "correction": correction,
        })
    expected = int(config["candidate_family"]["one_shot_expected_candidate_count"])
    if len(output) != expected:
        raise ValueError("one-shot chunk count differs")
    return output


def distributed_delta_bounds(
    nominal_xyz: Any, weights: Sequence[float], action_limit: float
) -> tuple[Any, Any]:
    import numpy as np

    nominal = np.asarray(nominal_xyz, dtype=np.float64)
    values = np.asarray(weights, dtype=np.float64)
    if nominal.shape != (len(values), 3) or not np.isclose(np.sum(values), 0.0, atol=1.0e-15):
        raise ValueError("distributed residual inputs differ")
    lower = np.full(3, -np.inf, dtype=np.float64)
    upper = np.full(3, np.inf, dtype=np.float64)
    for step, weight in enumerate(values):
        if weight == 0.0:
            continue
        first = (-float(action_limit) - nominal[step]) / weight
        second = (float(action_limit) - nominal[step]) / weight
        lower = np.maximum(lower, np.minimum(first, second))
        upper = np.minimum(upper, np.maximum(first, second))
    if np.any(lower > upper):
        raise ValueError("distributed residual delta box is empty")
    return lower, upper


def distributed_chunks(nominal_chunk: Any, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    import numpy as np

    nominal = np.asarray(nominal_chunk, dtype=np.float64)
    horizon = int(nominal.shape[0]) if nominal.ndim == 2 else -1
    key = "two_step_residual_weights" if horizon == 2 else "five_step_residual_weights"
    weights = np.asarray(config["candidate_family"][key], dtype=np.float64)
    if nominal.shape != (len(weights), 7):
        raise ValueError("distributed nominal chunk differs")
    limit = float(config["candidate_family"]["action_limit"])
    lower, upper = distributed_delta_bounds(nominal[:, :3], weights, limit)
    count = int(config["candidate_family"]["grid_points_per_dimension"])
    axes = [np.linspace(lower[index], upper[index], count) for index in range(3)]
    deltas = [np.asarray(delta, dtype=np.float64) for delta in itertools.product(*axes)]
    deltas.append(np.zeros(3, dtype=np.float64))
    seen = set(); output = []
    for value in deltas:
        key_value = tuple(float(item) for item in value)
        if key_value in seen:
            continue
        seen.add(key_value)
        correction = weights[:, None] * value[None, :]
        chunk = nominal.copy(); chunk[:, :3] += correction
        if np.max(np.abs(chunk[:, :3])) > limit + 1.0e-12:
            raise ValueError("distributed chunk exceeds action bounds")
        if not np.allclose(np.sum(correction, axis=0), 0.0, rtol=0.0, atol=1.0e-12):
            raise ValueError("distributed correction does not preserve endpoint")
        output.append({
            "source": "endpoint_preserving_residual_grid",
            "parameter": value,
            "chunk": chunk,
            "correction": correction,
        })
    expected = int(config["candidate_family"]["distributed_expected_candidate_count"])
    if len(output) != expected:
        raise ValueError("distributed chunk count differs")
    return output
