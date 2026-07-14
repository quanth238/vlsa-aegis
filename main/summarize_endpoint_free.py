#!/usr/bin/env python3
"""Fail-closed population summary for the frozen R01 endpoint-free gate."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics
from typing import Any, Callable, List, Mapping

from crfs_harness.artifacts import (
    atomic_write_json,
    content_hash,
    file_sha256,
    load_json,
    validate_jsonl_unique,
)
from crfs_harness.manifest import validate_case


EXPECTED_CASES = 20
REQUIRED_CHANGED_ACTION_P_MIN_WITNESSES = 12
FROZEN_MANIFEST_RELATIVE_PATH = Path("manifests/oracle_h05_colliding.jsonl")
RESULT_FILENAME = "endpoint-free-feasibility.json"
EXECUTED_ACTIONS = 5
EXECUTED_ACTION_WIDTH = 7
TRANSLATION_DIMENSIONS = 3
EXPECTED_MEASUREMENT_SAMPLES = 126
REGISTERED_MAXIMUM_SCENE_MOTION_M = 0.001

Validator = Callable[[Mapping[str, Any]], List[str]]


def _is_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_git_oid(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) in {40, 64}
        and all(character in "0123456789abcdef" for character in value)
    )


def _load_endpoint_validator() -> Validator:
    # Keep module import dependency-free for local summary-contract tests. The
    # allocation environment imports the real runner and its NumPy dependency.
    from crfs_oracle.endpoint_free_runner import validate_endpoint_free_result

    return validate_endpoint_free_result


def _validated_manifest(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    try:
        cases, errors = validate_jsonl_unique(path, "case_id")
    except (OSError, json.JSONDecodeError) as error:
        return [], [f"cannot read manifest: {error}"]
    for case in cases:
        errors.extend(
            f"{case.get('case_id', '<unknown>')}: {error}"
            for error in validate_case(case)
        )
    case_ids = [str(case.get("case_id", "")) for case in cases]
    group_ids = [str(case.get("group_id", "")) for case in cases]
    if len(cases) != EXPECTED_CASES:
        errors.append(f"manifest must contain exactly {EXPECTED_CASES} cases, got {len(cases)}")
    if len(set(case_ids)) != EXPECTED_CASES:
        errors.append("manifest must contain exactly 20 unique case identities")
    if len(set(group_ids)) != EXPECTED_CASES:
        errors.append("manifest must contain exactly 20 unique group identities")
    return cases, errors


def _allocation_errors(provenance: Mapping[str, Any]) -> list[str]:
    errors = []
    if provenance.get("git_dirty") is not False:
        errors.append("provenance.git_dirty must be false")
    for key in ("git_commit", "baseline_commit"):
        if not _is_git_oid(provenance.get(key)):
            errors.append(f"provenance.{key} must be a full Git object ID")
    for key in ("partition", "host", "device"):
        if not isinstance(provenance.get(key), str) or not provenance.get(key):
            errors.append(f"provenance.{key} must be a non-empty string")
    for key in ("slurm_job_id", "slurm_array_task_id"):
        value = provenance.get(key)
        if not isinstance(value, str) or not value.isdigit():
            errors.append(f"provenance.{key} must be a numeric allocation identifier")
    return errors


def _identity_errors(
    value: Mapping[str, Any],
    *,
    folder_case_id: str,
    manifest_case: Mapping[str, Any] | None,
    expected_run_id: str,
    expected_config_hash: str,
    expected_manifest_sha256: str,
    expected_checkpoint_sha256: str,
    expected_r00_summary_sha256: str,
) -> list[str]:
    errors = []
    if value.get("case_id") != folder_case_id:
        errors.append("artifact case_id does not match its result directory")
    if value.get("run_id") != expected_run_id:
        errors.append("artifact run_id does not match the registered run")
    if value.get("config_hash") != expected_config_hash:
        errors.append("artifact config_hash does not match the registered config")

    provenance = value.get("provenance")
    if not isinstance(provenance, Mapping):
        return errors + ["artifact provenance is not an object"]
    errors.extend(_allocation_errors(provenance))
    expected_hashes = {
        "input_manifest_sha256": expected_manifest_sha256,
        "checkpoint_sha256": expected_checkpoint_sha256,
        "r00_summary_sha256": expected_r00_summary_sha256,
    }
    for key, expected in expected_hashes.items():
        if provenance.get(key) != expected:
            errors.append(f"provenance.{key} does not match the registered identity")

    calibration = value.get("calibration")
    if not isinstance(calibration, Mapping) or calibration.get(
        "summary_sha256"
    ) != expected_r00_summary_sha256:
        errors.append("calibration.summary_sha256 does not match the registered R00 artifact")

    if manifest_case is None:
        errors.append("artifact directory is not a frozen manifest case")
        return errors
    case_record = provenance.get("case_record")
    if not isinstance(case_record, Mapping):
        errors.append("provenance.case_record must be an object")
    elif dict(case_record) != dict(manifest_case):
        errors.append("provenance.case_record does not exactly match the frozen manifest record")
    for key in (
        "group_id",
        "task_suite",
        "safety_level",
        "task_index",
        "episode_index",
        "environment_seed",
        "policy_seed",
        "random_control_seed",
    ):
        if provenance.get(key) != manifest_case.get(key):
            errors.append(f"provenance.{key} does not match the frozen manifest")
    return errors


def _attempts(value: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    result = []
    verification = value.get("verification")
    if not isinstance(verification, Mapping):
        return result
    for search_name in ("p_min", "p_zero"):
        search = verification.get(search_name)
        if not isinstance(search, Mapping) or not isinstance(search.get("attempts"), list):
            continue
        result.extend(item for item in search["attempts"] if isinstance(item, Mapping))
    return result


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _vector3(value: Any) -> tuple[float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    converted = [_finite_number(item) for item in value]
    if any(item is None for item in converted):
        return None
    return tuple(float(item) for item in converted)  # type: ignore[arg-type,return-value]


def _rotation3(value: Any) -> tuple[tuple[float, float, float], ...] | None:
    if isinstance(value, (list, tuple)) and len(value) == 9:
        converted = [_finite_number(item) for item in value]
        if any(item is None for item in converted):
            return None
        flat = [float(item) for item in converted]
        return tuple(tuple(flat[3 * row : 3 * row + 3]) for row in range(3))
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    rows = [_vector3(row) for row in value]
    if any(row is None for row in rows):
        return None
    return tuple(rows)  # type: ignore[arg-type,return-value]


def _point_box_signed_distance(
    point: tuple[float, float, float],
    center: tuple[float, float, float],
    rotation: tuple[tuple[float, float, float], ...],
    half_size: tuple[float, float, float],
) -> float:
    """Match the registered conservative sphere/oriented-box primitive."""

    delta = tuple(point[axis] - center[axis] for axis in range(3))
    local = tuple(
        sum(rotation[row][column] * delta[row] for row in range(3))
        for column in range(3)
    )
    extent_delta = tuple(abs(local[axis]) - half_size[axis] for axis in range(3))
    outside = math.sqrt(sum(max(value, 0.0) ** 2 for value in extent_delta))
    inside = min(max(extent_delta), 0.0)
    return outside + inside


def _branch_clearance_m(trial: Mapping[str, Any]) -> float | None:
    """Recompute clearance at the immutable branch point from raw geometry."""

    point = _vector3(trial.get("start_eef_center_m"))
    boxes = trial.get("branch_obstacle_boxes")
    measurement = trial.get("measurement")
    if point is None or not isinstance(boxes, list) or not isinstance(measurement, Mapping):
        return None
    radius = _finite_number(measurement.get("conservative_eef_radius_m"))
    if radius is None or radius <= 0.0 or not boxes:
        return None

    clearances = []
    for box in boxes:
        if not isinstance(box, Mapping):
            return None
        center = _vector3(box.get("center_m"))
        half_size = _vector3(box.get("half_size_m"))
        rotation = _rotation3(box.get("rotation_world"))
        if (
            center is None
            or half_size is None
            or rotation is None
            or any(value <= 0.0 for value in half_size)
        ):
            return None
        clearances.append(
            _point_box_signed_distance(point, center, rotation, half_size) - radius
        )
    return min(clearances)


def _action_matrix(value: Any) -> list[list[float]] | None:
    if not isinstance(value, list) or len(value) != EXECUTED_ACTIONS:
        return None
    result = []
    for row in value:
        if not isinstance(row, list) or len(row) != EXECUTED_ACTION_WIDTH:
            return None
        converted = [_finite_number(item) for item in row]
        if any(item is None for item in converted):
            return None
        result.append([float(item) for item in converted])
    return result


def _trial_passes(
    trial: Mapping[str, Any],
    *,
    minimum_progress_m: float,
    simulator_safety_margin_m: float,
) -> bool:
    clearance = _finite_number(trial.get("clearance_m"))
    reach = trial.get("reach")
    if clearance is None or not isinstance(reach, Mapping):
        return False
    progress = _finite_number(reach.get("reach_progress_m"))
    target_motion = _finite_number(reach.get("maximum_target_displacement_m"))
    obstacle_motion = _finite_number(
        reach.get("maximum_active_obstacle_displacement_m")
    )
    samples = trial.get("measurement_samples")
    return bool(
        clearance >= simulator_safety_margin_m
        and trial.get("contact") is False
        and progress is not None
        and progress >= minimum_progress_m
        and target_motion is not None
        and target_motion <= REGISTERED_MAXIMUM_SCENE_MOTION_M
        and obstacle_motion is not None
        and obstacle_motion <= REGISTERED_MAXIMUM_SCENE_MOTION_M
        and isinstance(samples, int)
        and not isinstance(samples, bool)
        and samples == EXPECTED_MEASUREMENT_SAMPLES
    )


def _attempt_truth(
    attempt: Mapping[str, Any],
    *,
    nominal: list[list[float]],
    minimum_progress_m: float,
    p_min_m: float,
    simulator_safety_margin_m: float,
    expected_repeats: int,
) -> dict[str, Any] | None:
    actions = _action_matrix(attempt.get("actions"))
    trials_value = attempt.get("trials")
    if actions is None or not isinstance(trials_value, list) or not all(
        isinstance(trial, Mapping) for trial in trials_value
    ):
        return None
    trials = list(trials_value)
    replay_exact = bool(
        len(trials) == expected_repeats
        and trials
        and all(content_hash(trial) == content_hash(trials[0]) for trial in trials[1:])
    )
    nontranslation_preserved = all(
        actions[row][column] == nominal[row][column]
        for row in range(EXECUTED_ACTIONS)
        for column in range(TRANSLATION_DIMENSIONS, EXECUTED_ACTION_WIDTH)
    )
    action_changed = any(
        actions[row][column] != nominal[row][column]
        for row in range(EXECUTED_ACTIONS)
        for column in range(EXECUTED_ACTION_WIDTH)
    )
    translation_within_bounds = all(
        -1.0 <= actions[row][column] <= 1.0
        for row in range(EXECUTED_ACTIONS)
        for column in range(TRANSLATION_DIMENSIONS)
    )
    simulator_safety_pass = bool(
        replay_exact
        and all(
            _finite_number(trial.get("clearance_m")) is not None
            and float(trial["clearance_m"]) >= simulator_safety_margin_m
            and trial.get("contact") is False
            and trial.get("measurement_samples") == EXPECTED_MEASUREMENT_SAMPLES
            for trial in trials
        )
    )
    scene_stationary = bool(
        replay_exact
        and all(
            isinstance(trial.get("reach"), Mapping)
            and _finite_number(trial["reach"].get("maximum_target_displacement_m"))
            is not None
            and float(trial["reach"]["maximum_target_displacement_m"])
            <= REGISTERED_MAXIMUM_SCENE_MOTION_M
            and _finite_number(
                trial["reach"].get("maximum_active_obstacle_displacement_m")
            )
            is not None
            and float(trial["reach"]["maximum_active_obstacle_displacement_m"])
            <= REGISTERED_MAXIMUM_SCENE_MOTION_M
            for trial in trials
        )
    )
    simulator_progress_pass = bool(
        replay_exact
        and all(
            isinstance(trial.get("reach"), Mapping)
            and _finite_number(trial["reach"].get("reach_progress_m")) is not None
            and float(trial["reach"]["reach_progress_m"]) >= minimum_progress_m
            for trial in trials
        )
    )

    def verified_at(threshold: float) -> bool:
        return bool(
            nontranslation_preserved
            and translation_within_bounds
            and replay_exact
            and all(
                _trial_passes(
                    trial,
                    minimum_progress_m=threshold,
                    simulator_safety_margin_m=simulator_safety_margin_m,
                )
                for trial in trials
            )
        )

    verified = verified_at(minimum_progress_m)
    verified_at_p_min = verified_at(p_min_m)
    verified_at_p_zero = verified_at(0.0)
    correction = [
        actions[row][column] - nominal[row][column]
        for row in range(EXECUTED_ACTIONS)
        for column in range(TRANSLATION_DIMENSIONS)
    ]
    return {
        "actions": actions,
        "trials": trials,
        "direct_replay_exact": replay_exact,
        "nontranslation_preserved": nontranslation_preserved,
        "action_changed_from_nominal": action_changed,
        "translation_within_bounds": translation_within_bounds,
        "simulator_safety_pass": simulator_safety_pass,
        "simulator_progress_pass": simulator_progress_pass,
        "scene_stationary": scene_stationary,
        "verified": verified,
        "verified_at_p_min": verified_at_p_min,
        "verified_at_p_zero": verified_at_p_zero,
        "changed_witness_at_p_min": bool(action_changed and verified_at_p_min),
        "changed_witness_at_p_zero": bool(action_changed and verified_at_p_zero),
        "objective": 0.5 * sum(item * item for item in correction),
        "correction_l2": math.sqrt(sum(item * item for item in correction)),
        "correction_rms": math.sqrt(
            sum(item * item for item in correction) / len(correction)
        ),
        "correction_max_abs": max(abs(item) for item in correction),
        "saturated_translation_components": sum(
            abs(actions[row][column]) >= 1.0 - 1e-12
            for row in range(EXECUTED_ACTIONS)
            for column in range(TRANSLATION_DIMENSIONS)
        ),
    }


def _direct_evidence_errors(value: Mapping[str, Any]) -> list[str]:
    """Recompute the R01 numerator from raw actions and repeated rollouts."""

    errors = []
    provenance = value.get("provenance")
    calibration = value.get("calibration")
    nominal_value = value.get("nominal")
    verification = value.get("verification")
    outcome = value.get("outcome")
    if not all(
        isinstance(item, Mapping)
        for item in (provenance, calibration, nominal_value, verification, outcome)
    ):
        return ["cannot recompute direct evidence from malformed result sections"]
    nominal = _action_matrix(nominal_value.get("actions"))
    p_min_m = _finite_number(calibration.get("p_min_m"))
    simulator_margin = _finite_number(provenance.get("simulator_safety_margin_m"))
    expected_repeats = provenance.get("simulator_verification_repeats")
    if (
        nominal is None
        or p_min_m is None
        or simulator_margin is None
        or not isinstance(expected_repeats, int)
        or isinstance(expected_repeats, bool)
        or expected_repeats < 2
    ):
        return ["cannot recompute direct evidence from invalid registered thresholds"]

    nominal_repeats = nominal_value.get("repeats")
    nominal_replay_exact = bool(
        isinstance(nominal_repeats, list)
        and len(nominal_repeats) >= 2
        and all(isinstance(trial, Mapping) for trial in nominal_repeats)
        and all(
            content_hash(trial) == content_hash(nominal_repeats[0])
            for trial in nominal_repeats[1:]
        )
    )
    if not nominal_replay_exact:
        errors.append("nominal raw repeats are not exact")
        nominal_collision_reproduced = False
        nominal_fails_margin = False
    else:
        first_nominal = nominal_repeats[0]
        clearance = _finite_number(first_nominal.get("clearance_m"))
        contact = first_nominal.get("contact")
        samples = first_nominal.get("measurement_samples")
        if (
            clearance is None
            or not isinstance(contact, bool)
            or samples != EXPECTED_MEASUREMENT_SAMPLES
        ):
            errors.append("nominal raw repeat lacks valid clearance/contact/sample evidence")
            nominal_collision_reproduced = False
            nominal_fails_margin = False
        else:
            nominal_collision_reproduced = bool(clearance < 0.0 or contact)
            nominal_fails_margin = bool(clearance < simulator_margin or contact)
    for key, expected in (
        ("collision_reproduced", nominal_collision_reproduced),
        ("fails_registered_margin", nominal_fails_margin),
    ):
        if nominal_value.get(key) is not expected:
            errors.append(f"nominal.{key} conflicts with raw repeated rollout")

    pooled = {
        "p_min_any_action_verified": False,
        "p_zero_any_action_verified": False,
        "p_min_changed_witness_verified": False,
        "p_zero_changed_witness_verified": False,
    }
    for search_name, expected_threshold in (("p_min", p_min_m), ("p_zero", 0.0)):
        search = verification.get(search_name)
        if not isinstance(search, Mapping):
            errors.append(f"verification.{search_name} is not an object")
            continue
        threshold = _finite_number(search.get("minimum_progress_m"))
        if threshold is None or not math.isclose(
            threshold, expected_threshold, rel_tol=0.0, abs_tol=1e-12
        ):
            errors.append(f"verification.{search_name} has the wrong progress threshold")
            continue
        attempts = search.get("attempts")
        if not isinstance(attempts, list) or not all(
            isinstance(attempt, Mapping) for attempt in attempts
        ):
            errors.append(f"verification.{search_name}.attempts is malformed")
            continue
        truths = []
        for index, attempt in enumerate(attempts):
            truth = _attempt_truth(
                attempt,
                nominal=nominal,
                minimum_progress_m=threshold,
                p_min_m=p_min_m,
                simulator_safety_margin_m=simulator_margin,
                expected_repeats=expected_repeats,
            )
            if truth is None:
                errors.append(
                    f"verification.{search_name}.attempts[{index}] lacks raw replay evidence"
                )
                continue
            truths.append(truth)
            for key in (
                "direct_replay_exact",
                "nontranslation_preserved",
                "action_changed_from_nominal",
                "simulator_safety_pass",
                "simulator_progress_pass",
                "scene_stationary",
                "verified",
                "verified_at_p_min",
                "verified_at_p_zero",
                "changed_witness_at_p_min",
                "changed_witness_at_p_zero",
            ):
                if attempt.get(key) is not truth[key]:
                    errors.append(
                        f"verification.{search_name}.attempts[{index}].{key} "
                        "conflicts with raw replay evidence"
                    )
            objective = _finite_number(attempt.get("objective"))
            if objective is None or not math.isclose(
                objective, truth["objective"], rel_tol=0.0, abs_tol=1e-10
            ):
                errors.append(
                    f"verification.{search_name}.attempts[{index}].objective "
                    "conflicts with action displacement"
                )
            if not truth["translation_within_bounds"]:
                errors.append(
                    f"verification.{search_name}.attempts[{index}] violates action bounds"
                )
        aggregate = {
            "verified": any(item["verified"] for item in truths),
            "verified_at_p_min": any(item["verified_at_p_min"] for item in truths),
            "verified_at_p_zero": any(item["verified_at_p_zero"] for item in truths),
            "changed_witness_at_p_min": any(
                item["changed_witness_at_p_min"] for item in truths
            ),
            "changed_witness_at_p_zero": any(
                item["changed_witness_at_p_zero"] for item in truths
            ),
        }
        for key, expected in aggregate.items():
            if search.get(key) is not expected:
                errors.append(
                    f"verification.{search_name}.{key} conflicts with raw attempts"
                )
        pooled["p_min_any_action_verified"] |= aggregate["verified_at_p_min"]
        pooled["p_zero_any_action_verified"] |= aggregate["verified_at_p_zero"]
        pooled["p_min_changed_witness_verified"] |= aggregate[
            "changed_witness_at_p_min"
        ]
        pooled["p_zero_changed_witness_verified"] |= aggregate[
            "changed_witness_at_p_zero"
        ]

    pooled["p_min_verified"] = bool(
        nominal_collision_reproduced and pooled["p_min_changed_witness_verified"]
    )
    pooled["p_zero_verified"] = bool(
        nominal_collision_reproduced and pooled["p_zero_changed_witness_verified"]
    )
    pooled["nominal_collision_reproduced"] = nominal_collision_reproduced
    pooled["nominal_fails_registered_margin"] = nominal_fails_margin
    for key, expected in pooled.items():
        if outcome.get(key) is not expected:
            errors.append(f"outcome.{key} conflicts with recomputed direct evidence")
    return errors


def _witness_quality(records: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    case_metrics = []
    for case_id, value in sorted(records.items()):
        nominal = _action_matrix(value["nominal"]["actions"])
        p_min_m = float(value["calibration"]["p_min_m"])
        simulator_margin = float(value["provenance"]["simulator_safety_margin_m"])
        expected_repeats = int(value["provenance"]["simulator_verification_repeats"])
        if nominal is None:
            continue
        selected = None
        for search_name in ("p_min", "p_zero"):
            search = value["verification"][search_name]
            for attempt in search["attempts"]:
                truth = _attempt_truth(
                    attempt,
                    nominal=nominal,
                    minimum_progress_m=float(search["minimum_progress_m"]),
                    p_min_m=p_min_m,
                    simulator_safety_margin_m=simulator_margin,
                    expected_repeats=expected_repeats,
                )
                if truth is not None and truth["changed_witness_at_p_min"]:
                    selected = (search_name, attempt, truth)
                    break
            if selected is not None:
                break
        if selected is None:
            continue
        search_name, attempt, truth = selected
        first_trial = truth["trials"][0]
        reach = first_trial["reach"]
        case_metrics.append(
            {
                "case_id": case_id,
                "search": search_name,
                "candidate_index": attempt["candidate_index"],
                "source": attempt["source"],
                "within_h04_calibration_domain": bool(
                    attempt["within_h04_calibration_domain"]
                ),
                "correction_l2": truth["correction_l2"],
                "correction_rms": truth["correction_rms"],
                "correction_max_abs": truth["correction_max_abs"],
                "saturated_translation_components": truth[
                    "saturated_translation_components"
                ],
                "d_sim_m": float(first_trial["clearance_m"]),
                "reach_progress_m": float(reach["reach_progress_m"]),
                "target_max_displacement_m": float(
                    reach["maximum_target_displacement_m"]
                ),
                "obstacle_max_displacement_m": float(
                    reach["maximum_active_obstacle_displacement_m"]
                ),
            }
        )

    def distribution(key: str) -> dict[str, float] | None:
        values = [float(item[key]) for item in case_metrics]
        if not values:
            return None
        return {
            "minimum": min(values),
            "median": statistics.median(values),
            "maximum": max(values),
        }

    return {
        "selected_changed_p_min_witnesses": len(case_metrics),
        "within_h04_calibration_domain": sum(
            bool(item["within_h04_calibration_domain"]) for item in case_metrics
        ),
        "outside_h04_calibration_domain": sum(
            not bool(item["within_h04_calibration_domain"]) for item in case_metrics
        ),
        "with_saturated_translation_component": sum(
            int(item["saturated_translation_components"]) > 0 for item in case_metrics
        ),
        "correction_l2": distribution("correction_l2"),
        "correction_rms": distribution("correction_rms"),
        "d_sim_m": distribution("d_sim_m"),
        "reach_progress_m": distribution("reach_progress_m"),
        "case_metrics": case_metrics,
    }


def _branch_clearance_diagnostics(
    records: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    metrics = []
    for case_id, value in sorted(records.items()):
        repeats = value["nominal"].get("repeats")
        margin = _finite_number(value["provenance"].get("simulator_safety_margin_m"))
        if not isinstance(repeats, list) or not repeats or not isinstance(repeats[0], Mapping):
            continue
        clearance = _branch_clearance_m(repeats[0])
        if clearance is None or margin is None:
            continue
        metrics.append(
            {
                "case_id": case_id,
                "branch_clearance_m": clearance,
                "registered_margin_m": margin,
                "branch_in_penetration": clearance < 0.0,
                "branch_below_registered_margin": clearance < margin,
            }
        )

    clearances = [float(item["branch_clearance_m"]) for item in metrics]
    below_zero = [
        str(item["case_id"]) for item in metrics if item["branch_in_penetration"]
    ]
    below_margin = [
        str(item["case_id"])
        for item in metrics
        if item["branch_below_registered_margin"]
    ]
    return {
        "recomputed_cases": len(metrics),
        "branch_in_penetration": len(below_zero),
        "branch_below_registered_margin": len(below_margin),
        "branch_in_penetration_case_ids": below_zero,
        "branch_below_registered_margin_case_ids": below_margin,
        "clearance_m": (
            {
                "minimum": min(clearances),
                "median": statistics.median(clearances),
                "maximum": max(clearances),
            }
            if clearances
            else None
        ),
        "case_metrics": metrics,
        "interpretation": (
            "A case below the registered margin at the immutable branch point cannot "
            "satisfy the inclusive branch-plus-substep R01 safety predicate from that "
            "same instant, independent of the bounded search result."
        ),
    }


def _case_diagnostics(records: Mapping[str, Mapping[str, Any]]) -> tuple[dict[str, int], dict[str, list[str]]]:
    case_ids: dict[str, list[str]] = {
        "nominal_collision_reproduced": [],
        "nominal_collision_not_reproduced": [],
        "changed_action_p_min_rescues": [],
        "changed_action_p_zero_only_rescues": [],
        "any_action_simulator_safe": [],
        "any_action_safe_at_p_min": [],
        "any_action_safe_at_p_zero": [],
        "proxy_clearance_false_safe": [],
        "proxy_clearance_false_negative": [],
        "proxy_joint_false_negative": [],
        "bounded_search_miss_p_min": [],
        "bounded_search_miss_p_zero": [],
        "bounded_search_miss_both": [],
    }
    false_safe_attempts = 0
    false_negative_attempts = 0
    joint_false_negative_attempts = 0

    for case_id, value in sorted(records.items()):
        outcome = value["outcome"]
        nominal_reproduced = bool(outcome["nominal_collision_reproduced"])
        p_min_changed = bool(outcome["p_min_changed_witness_verified"])
        p_zero_changed = bool(outcome["p_zero_changed_witness_verified"])
        if nominal_reproduced:
            case_ids["nominal_collision_reproduced"].append(case_id)
        else:
            case_ids["nominal_collision_not_reproduced"].append(case_id)
        if nominal_reproduced and p_min_changed:
            case_ids["changed_action_p_min_rescues"].append(case_id)
        if nominal_reproduced and p_zero_changed and not p_min_changed:
            case_ids["changed_action_p_zero_only_rescues"].append(case_id)
        if bool(outcome["p_min_any_action_verified"]):
            case_ids["any_action_safe_at_p_min"].append(case_id)
        if bool(outcome["p_zero_any_action_verified"]):
            case_ids["any_action_safe_at_p_zero"].append(case_id)

        attempts = _attempts(value)
        if any(bool(item.get("simulator_safety_pass")) for item in attempts):
            case_ids["any_action_simulator_safe"].append(case_id)
        case_false_safe = sum(bool(item.get("proxy_clearance_false_safe")) for item in attempts)
        case_false_negative = sum(
            bool(item.get("proxy_clearance_false_negative")) for item in attempts
        )
        case_joint_false_negative = sum(
            bool(item.get("proxy_joint_false_negative_at_search_threshold"))
            for item in attempts
        )
        false_safe_attempts += case_false_safe
        false_negative_attempts += case_false_negative
        joint_false_negative_attempts += case_joint_false_negative
        if case_false_safe:
            case_ids["proxy_clearance_false_safe"].append(case_id)
        if case_false_negative:
            case_ids["proxy_clearance_false_negative"].append(case_id)
        if case_joint_false_negative:
            case_ids["proxy_joint_false_negative"].append(case_id)

        searches = value["searches"]
        p_min_miss = searches["p_min"]["outcome"] == "not_found_within_budget"
        p_zero_miss = searches["p_zero"]["outcome"] == "not_found_within_budget"
        if p_min_miss:
            case_ids["bounded_search_miss_p_min"].append(case_id)
        if p_zero_miss:
            case_ids["bounded_search_miss_p_zero"].append(case_id)
        if p_min_miss and p_zero_miss:
            case_ids["bounded_search_miss_both"].append(case_id)

    counts = {key: len(value) for key, value in case_ids.items()}
    counts.update(
        {
            "proxy_clearance_false_safe_attempts": false_safe_attempts,
            "proxy_clearance_false_negative_attempts": false_negative_attempts,
            "proxy_joint_false_negative_attempts": joint_false_negative_attempts,
        }
    )
    return counts, case_ids


def summarize_endpoint_free_population(
    manifest_path: str | Path,
    results_root: str | Path,
    *,
    expected_run_id: str,
    expected_config_hash: str,
    expected_checkpoint_sha256: str,
    expected_r00_summary_sha256: str,
    repo_root: str | Path | None = None,
    validator: Validator | None = None,
) -> dict[str, Any]:
    """Validate and summarize exactly the immutable 20-case R01 population."""

    if not expected_run_id:
        raise ValueError("expected_run_id must be non-empty")
    for name, value in (
        ("expected_config_hash", expected_config_hash),
        ("expected_checkpoint_sha256", expected_checkpoint_sha256),
        ("expected_r00_summary_sha256", expected_r00_summary_sha256),
    ):
        if not _is_sha256(value):
            raise ValueError(f"{name} must be a lowercase SHA-256 digest")

    root = Path(repo_root).resolve() if repo_root is not None else Path(__file__).resolve().parents[1]
    manifest = Path(manifest_path).resolve()
    run_root = Path(results_root).resolve()
    frozen_manifest = root / FROZEN_MANIFEST_RELATIVE_PATH
    cases, manifest_errors = _validated_manifest(manifest)
    frozen_cases, frozen_errors = _validated_manifest(frozen_manifest)
    if frozen_errors:
        raise RuntimeError("checked-in frozen R01 manifest is invalid: " + "; ".join(frozen_errors))

    expected_manifest_sha256 = file_sha256(manifest)
    frozen_manifest_sha256 = file_sha256(frozen_manifest)
    if expected_manifest_sha256 != frozen_manifest_sha256:
        manifest_errors.append("input manifest bytes differ from the checked-in frozen R01 manifest")
    frozen_identities = {
        (str(case["case_id"]), str(case["group_id"])) for case in frozen_cases
    }
    supplied_identities = {
        (str(case.get("case_id", "")), str(case.get("group_id", ""))) for case in cases
    }
    if supplied_identities != frozen_identities:
        manifest_errors.append("input manifest case/group identities differ from the frozen population")

    expected_by_case = {str(case["case_id"]): case for case in frozen_cases}
    final_paths = sorted(run_root.glob(f"*/{RESULT_FILENAME}"))
    found_directory_ids = {path.parent.name for path in final_paths}
    expected_ids = set(expected_by_case)
    missing_case_ids = sorted(expected_ids - found_directory_ids)
    unexpected_case_ids = sorted(found_directory_ids - expected_ids)
    endpoint_validator = validator if validator is not None else _load_endpoint_validator()

    invalid_artifacts = []
    valid_records: dict[str, Mapping[str, Any]] = {}
    result_hashes = []
    runner_validated_artifacts = 0
    for path in final_paths:
        folder_case_id = path.parent.name
        errors = []
        try:
            value = load_json(path)
        except (OSError, json.JSONDecodeError) as error:
            invalid_artifacts.append(
                {"case_id": folder_case_id, "path": str(path), "errors": [f"invalid JSON: {error}"]}
            )
            continue
        try:
            runner_errors = endpoint_validator(value)  # type: ignore[arg-type]
        except Exception as error:  # A malformed final artifact must fail closed.
            runner_errors = [f"endpoint_free_runner validator raised: {error}"]
        if runner_errors:
            errors.extend(f"runner: {error}" for error in runner_errors)
        else:
            runner_validated_artifacts += 1
        if not isinstance(value, Mapping):
            errors.append("final artifact must be a JSON object")
        else:
            errors.extend(
                _identity_errors(
                    value,
                    folder_case_id=folder_case_id,
                    manifest_case=expected_by_case.get(folder_case_id),
                    expected_run_id=expected_run_id,
                    expected_config_hash=expected_config_hash,
                    expected_manifest_sha256=expected_manifest_sha256,
                    expected_checkpoint_sha256=expected_checkpoint_sha256,
                    expected_r00_summary_sha256=expected_r00_summary_sha256,
                )
            )
            errors.extend(
                f"direct evidence: {error}" for error in _direct_evidence_errors(value)
            )
        if errors:
            invalid_artifacts.append(
                {"case_id": folder_case_id, "path": str(path), "errors": errors}
            )
            continue
        if folder_case_id in valid_records:
            invalid_artifacts.append(
                {
                    "case_id": folder_case_id,
                    "path": str(path),
                    "errors": ["duplicate final artifact for one frozen case"],
                }
            )
            continue
        valid_records[folder_case_id] = value
        result_hashes.append({"case_id": folder_case_id, "sha256": file_sha256(path)})

    population_errors = list(manifest_errors)
    if missing_case_ids:
        population_errors.append(f"missing frozen case artifacts: {missing_case_ids}")
    if unexpected_case_ids:
        population_errors.append(f"unexpected case artifacts: {unexpected_case_ids}")
    if invalid_artifacts:
        population_errors.append(f"{len(invalid_artifacts)} final artifacts failed validation")
    if len(valid_records) != EXPECTED_CASES:
        population_errors.append(
            f"validated population is {len(valid_records)}/{EXPECTED_CASES}"
        )

    git_commits = sorted(
        {str(value["provenance"]["git_commit"]) for value in valid_records.values()}
    )
    baseline_commits = sorted(
        {str(value["provenance"]["baseline_commit"]) for value in valid_records.values()}
    )
    p_min_values = sorted(
        {float(value["calibration"]["p_min_m"]) for value in valid_records.values()}
    )
    if valid_records and len(git_commits) != 1:
        population_errors.append(f"population mixes git commits: {git_commits}")
    if valid_records and len(baseline_commits) != 1:
        population_errors.append(f"population mixes baseline commits: {baseline_commits}")
    if valid_records and len(p_min_values) != 1:
        population_errors.append(f"population mixes frozen p_min values: {p_min_values}")

    observed_counts, case_ids = _case_diagnostics(valid_records)
    witness_quality = _witness_quality(valid_records)
    branch_clearance = _branch_clearance_diagnostics(valid_records)
    population_valid = not population_errors
    observed_numerator = observed_counts["changed_action_p_min_rescues"]
    gate_numerator = observed_numerator if population_valid else 0
    gate_passed = bool(
        population_valid
        and gate_numerator >= REQUIRED_CHANGED_ACTION_P_MIN_WITNESSES
    )
    status = (
        "passed"
        if gate_passed
        else "failed_threshold"
        if population_valid
        else "invalid_population"
    )
    if gate_passed:
        decision = "The registered R01 count threshold was met; later steering gates remain unevaluated."
    elif population_valid:
        decision = (
            "The registered R01 count threshold was not met; bounded search misses are not "
            "non-existence certificates."
        )
    else:
        decision = "R01 population is not validation-ready; repair the listed identity or provenance errors."

    ordered_hashes = sorted(result_hashes, key=lambda item: item["case_id"])
    counts = {
        "expected_population": EXPECTED_CASES,
        "validated_population": len(valid_records),
        **observed_counts,
        "changed_action_p_min_rescues_observed": observed_numerator,
        "changed_action_p_min_gate_numerator": gate_numerator,
        "required_changed_action_p_min_witnesses": REQUIRED_CHANGED_ACTION_P_MIN_WITNESSES,
    }
    return {
        "schema_version": "1.0",
        "gate": "R01",
        "status": status,
        "gate_passed": gate_passed,
        "decision": decision,
        "analysis_scope": "registered population counts only; no bootstrap analysis",
        "population": {
            "valid": population_valid,
            "expected_cases": EXPECTED_CASES,
            "expected_unique_groups": EXPECTED_CASES,
            "final_artifacts_found": len(final_paths),
            "runner_validated_artifacts": runner_validated_artifacts,
            "fully_validated_expected_artifacts": len(valid_records),
            "missing_case_ids": missing_case_ids,
            "unexpected_case_ids": unexpected_case_ids,
            "invalid_artifacts": invalid_artifacts,
            "errors": population_errors,
        },
        "identities": {
            "run_id": expected_run_id,
            "config_hash": expected_config_hash,
            "manifest": str(manifest),
            "manifest_sha256": expected_manifest_sha256,
            "frozen_manifest_sha256": frozen_manifest_sha256,
            "checkpoint_sha256": expected_checkpoint_sha256,
            "r00_summary_sha256": expected_r00_summary_sha256,
            "git_commit": git_commits[0] if len(git_commits) == 1 else None,
            "baseline_commit": baseline_commits[0] if len(baseline_commits) == 1 else None,
            "p_min_m": p_min_values[0] if len(p_min_values) == 1 else None,
        },
        "counts": counts,
        "case_ids": case_ids,
        "status_counts": dict(
            sorted(Counter(str(value["status"]) for value in valid_records.values()).items())
        ),
        "witness_quality": witness_quality,
        "branch_clearance": branch_clearance,
        "ordered_result_set_digest": content_hash(ordered_hashes),
        "result_hashes": ordered_hashes,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--config-hash", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--r00-summary-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    summary = summarize_endpoint_free_population(
        args.manifest,
        args.results_root,
        expected_run_id=args.run_id,
        expected_config_hash=args.config_hash,
        expected_checkpoint_sha256=args.checkpoint_sha256,
        expected_r00_summary_sha256=args.r00_summary_sha256,
    )
    atomic_write_json(args.output, summary)
    printable = {
        key: value
        for key, value in summary.items()
        if key not in {"case_ids", "result_hashes"}
    }
    print(json.dumps(printable, sort_keys=True))
    return 0 if summary["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
