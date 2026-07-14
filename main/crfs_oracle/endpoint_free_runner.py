"""Allocation-backed R01 endpoint-free reach-feasibility runner.

The stopped H05 runner remains unchanged.  R01 replays the same immutable case
and policy noise, performs two bounded ``D_opt`` searches, and only then
directly verifies nominated actions with repeated ``D_sim`` rollouts.  A search
miss is a completed bounded experiment, not a non-existence certificate.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import socket
from typing import Any, Mapping, Sequence

import numpy as np

from crfs_harness.artifacts import (
    atomic_write_json,
    content_hash,
    file_sha256,
    load_json,
    scientific_config,
)

from .endpoint_free_projection import solve_endpoint_free_projection
from .progress_calibration import _target_contact_at_branch
from .reach_progress import (
    EXECUTED_REACH_ACTIONS,
    TARGET_OBJECT_NAME,
    ReachSnapshot,
    annotate_reach_rollout,
    capture_reach_snapshot,
)
from .runner import (
    OracleConfig,
    SafeLiberoCase,
    _array_hash,
    _determinism_check,
    _git_state,
    _infer,
    policy_observation,
)


FINAL_STATUSES = {
    "verified_safe_progress",
    "no_verified_safe_progress",
    "nominal_collision_not_reproduced",
}
SEARCH_KEYS = {"p_min", "p_zero"}
EXPECTED_MEASUREMENT_SAMPLES = 1 + EXECUTED_REACH_ACTIONS * 25
RESULT_FIELDS = {
    "schema_version",
    "gate",
    "case_id",
    "run_id",
    "status",
    "config_hash",
    "provenance",
    "calibration",
    "nominal",
    "searches",
    "verification",
    "outcome",
}


@dataclass(frozen=True)
class EndpointFreeConfig:
    oracle: OracleConfig
    target_name: str
    p_min_m: float
    r00_summary_path: str
    r00_summary_sha256: str
    r00_summary: Mapping[str, Any]
    optimizer_safety_margin_m: float
    simulator_safety_margin_m: float
    maximum_target_displacement_m: float
    maximum_obstacle_displacement_m: float
    simulator_verification_candidates: int
    simulator_verification_repeats: int
    planner_seed: int
    translation_action_low: float
    translation_action_high: float
    response_matrix_m_per_action: tuple[tuple[float, ...], ...]
    search_kwargs: Mapping[str, Any]


def _finite_number(value: Any, *, name: str, positive: bool = False) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric") from error
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "positive and finite" if positive else "finite"
        raise ValueError(f"{name} must be {qualifier}")
    return result


def _resolve_artifact(path_value: str, repo_root: str | Path) -> Path:
    path = Path(path_value).expanduser()
    return path if path.is_absolute() else Path(repo_root).resolve() / path


def endpoint_free_config_from_mapping(
    value: Mapping[str, Any],
    oracle: OracleConfig,
    *,
    repo_root: str | Path,
    evaluation_manifest_sha256: str,
) -> EndpointFreeConfig:
    """Validate R01 settings and load the content-addressed frozen R00 summary."""
    if not bool(value.get("ready_to_run", False)):
        raise ValueError("R01 config is not ready_to_run")
    if oracle.action_horizon != 10 or oracle.executed_prefix != EXECUTED_REACH_ACTIONS:
        raise ValueError("R01 requires a ten-action model horizon and five executed actions")

    progress = value.get("progress")
    planner = value.get("planner")
    if not isinstance(progress, Mapping) or not isinstance(planner, Mapping):
        raise ValueError("R01 config requires progress and planner objects")
    if progress.get("phase") != "pregrasp_reach":
        raise ValueError("R01 is restricted to the pregrasp_reach phase")
    if progress.get("scene_motion_measurement") != "maximum_substep_displacement":
        raise ValueError("R01 requires maximum-substep target and obstacle motion")
    target_name = str(progress.get("target_object", ""))
    if target_name != TARGET_OBJECT_NAME:
        raise ValueError(f"R01 target must be {TARGET_OBJECT_NAME!r}")
    method = str(planner.get("method", ""))
    if method not in {"deterministic_cem_multistart", "cem_deterministic_multistart"}:
        raise ValueError("R01 planner.method must identify the deterministic CEM implementation")
    if bool(planner.get("endpoint_preservation", True)):
        raise ValueError("R01 must not preserve the nominal endpoint")
    action_bounds = planner.get("translation_action_bounds")
    if not (
        isinstance(action_bounds, Sequence)
        and not isinstance(action_bounds, (str, bytes))
        and len(action_bounds) == 2
    ):
        raise ValueError("planner.translation_action_bounds must be [low, high]")
    action_low = _finite_number(action_bounds[0], name="translation action lower bound")
    action_high = _finite_number(action_bounds[1], name="translation action upper bound")
    if not math.isclose(action_low, -1.0, rel_tol=0.0, abs_tol=1e-12) or not math.isclose(
        action_high, 1.0, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("R01 requires the frozen LIBERO translation action bounds [-1, 1]")

    artifact_value = progress.get("calibration_artifact")
    expected_sha = str(progress.get("calibration_sha256", ""))
    if not isinstance(artifact_value, str) or not artifact_value:
        raise ValueError("progress.calibration_artifact must name the frozen R00 summary")
    if len(expected_sha) != 64 or any(character not in "0123456789abcdef" for character in expected_sha):
        raise ValueError("progress.calibration_sha256 must be a lowercase SHA-256 digest")
    artifact = _resolve_artifact(artifact_value, repo_root)
    actual_sha = file_sha256(artifact)
    if actual_sha != expected_sha:
        raise ValueError(f"R00 summary hash mismatch: expected {expected_sha}, got {actual_sha}")
    summary = load_json(artifact)
    if not isinstance(summary, Mapping):
        raise ValueError("R00 summary must be a JSON object")
    if summary.get("schema_version") != "1.0" or summary.get("gate") != "R00":
        raise ValueError("progress calibration artifact is not an R00 schema_version 1.0 summary")
    if summary.get("status") != "passed" or not summary.get(
        "calibration_and_evaluation_groups_disjoint", False
    ):
        raise ValueError("R00 must pass with disjoint calibration and evaluation groups")
    if summary.get("scene_motion_measurement") != (
        "maximum direct MuJoCo body displacement over 125 substeps"
    ):
        raise ValueError("R00 must measure maximum target and obstacle motion over all substeps")
    if int(summary.get("eligible_positive_examples", 0)) < 50:
        raise ValueError("R00 must contain at least 50 eligible positive examples")
    if int(summary.get("calibration_groups", 0)) != 30 or int(
        summary.get("evaluation_groups", 0)
    ) != 20:
        raise ValueError("R00 must retain the registered 30/20 disjoint group split")
    if summary.get("evaluation_manifest_sha256") != evaluation_manifest_sha256:
        raise ValueError("R00 was not frozen against the supplied R01 evaluation manifest")
    if summary.get("checkpoint_sha256") != oracle.checkpoint_sha256:
        raise ValueError("R00 and R01 checkpoint hashes differ")
    calibration = summary.get("calibration")
    if not isinstance(calibration, Mapping):
        raise ValueError("passed R00 summary has no calibration object")
    p_min_m = _finite_number(calibration.get("p_min_m"), name="R00 p_min_m", positive=True)
    if calibration.get("quantile_method") != "inverted_cdf" or not math.isclose(
        _finite_number(calibration.get("quantile"), name="R00 quantile"),
        0.25,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("R00 must use the preregistered inverted_cdf lower quartile")
    declared_p_min = progress.get("minimum_progress_m")
    if declared_p_min is not None and not math.isclose(
        _finite_number(declared_p_min, name="progress.minimum_progress_m"),
        p_min_m,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("config minimum_progress_m differs from the frozen R00 summary")

    target_limit = _finite_number(
        progress.get("reject_target_motion_above_m", 0.001),
        name="target motion limit",
    )
    obstacle_limit = _finite_number(
        progress.get("reject_obstacle_motion_above_m", target_limit),
        name="obstacle motion limit",
    )
    optimizer_margin = _finite_number(
        planner.get("optimizer_safety_margin_m"), name="optimizer safety margin", positive=True
    )
    simulator_margin = _finite_number(
        planner.get("simulator_safety_margin_m"), name="simulator safety margin", positive=True
    )
    verification_candidates = int(planner.get("simulator_verification_candidates", 3))
    verification_repeats = int(planner.get("simulator_verification_repeats", oracle.measurement_repeats))
    if target_limit < 0.0 or obstacle_limit < 0.0:
        raise ValueError("scene-motion limits must be non-negative")
    if target_limit > 0.001 or obstacle_limit > 0.001:
        raise ValueError("R01 scene-motion rejection limits cannot exceed 1 mm")
    if not math.isclose(simulator_margin, 0.005, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("R01 direct simulator verification requires a fixed 5 mm margin")
    if not math.isclose(oracle.safety_margin_m, 0.005, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("R01 top-level safety_margin_m must remain fixed at 5 mm")
    if verification_candidates < 1 or verification_repeats < 2:
        raise ValueError("R01 requires at least one candidate and two direct verification repeats")
    if oracle.response_matrix_m_per_action is None:
        raise ValueError("R01 requires the frozen H04 response matrix")

    search_defaults = {
        "population_size": 256,
        "generations": 10,
        "restarts": 4,
        "elite_fraction": 0.10,
        "initial_std_fraction": 0.25,
        "std_floor_fraction": 0.02,
        "template_amplitude": 0.15,
        "max_candidates": max(verification_candidates, 12),
        "samples_per_segment": 26,
        "h04_calibration_limit": 0.15,
    }
    search = planner.get("search")
    if search is not None and not isinstance(search, Mapping):
        raise ValueError("planner.search must be an object")
    if isinstance(search, Mapping):
        for key in search_defaults:
            if key in search:
                search_defaults[key] = search[key]

    return EndpointFreeConfig(
        oracle=oracle,
        target_name=target_name,
        p_min_m=p_min_m,
        r00_summary_path=str(artifact),
        r00_summary_sha256=actual_sha,
        r00_summary=summary,
        optimizer_safety_margin_m=optimizer_margin,
        simulator_safety_margin_m=simulator_margin,
        maximum_target_displacement_m=target_limit,
        maximum_obstacle_displacement_m=obstacle_limit,
        simulator_verification_candidates=verification_candidates,
        simulator_verification_repeats=verification_repeats,
        planner_seed=int(planner.get("planner_seed", 20260714)),
        translation_action_low=action_low,
        translation_action_high=action_high,
        response_matrix_m_per_action=oracle.response_matrix_m_per_action,
        search_kwargs=search_defaults,
    )


def _normalized_config(config: EndpointFreeConfig) -> dict[str, Any]:
    """Return the portable, content-addressed runtime configuration."""
    return {
        **scientific_config(config.oracle.__dict__),
        "target_name": config.target_name,
        "p_min_m": config.p_min_m,
        "r00_summary_sha256": config.r00_summary_sha256,
        "optimizer_safety_margin_m": config.optimizer_safety_margin_m,
        "simulator_safety_margin_m": config.simulator_safety_margin_m,
        "maximum_target_displacement_m": config.maximum_target_displacement_m,
        "maximum_obstacle_displacement_m": config.maximum_obstacle_displacement_m,
        "scene_motion_measurement": "maximum_substep_displacement",
        "simulator_verification_candidates": config.simulator_verification_candidates,
        "simulator_verification_repeats": config.simulator_verification_repeats,
        "planner_seed": config.planner_seed,
        "translation_action_bounds": [
            config.translation_action_low,
            config.translation_action_high,
        ],
        "search_kwargs": dict(config.search_kwargs),
    }


def _is_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_finite_json_number(value: Any) -> bool:
    return bool(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _nontranslation_preserved(candidate: Sequence[Sequence[float]], nominal: np.ndarray) -> bool:
    value = np.asarray(candidate, dtype=np.float64)
    return bool(value.shape == nominal.shape and np.array_equal(value[:, 3:], nominal[:, 3:]))


def _action_changed(candidate: Sequence[Sequence[float]], nominal: np.ndarray) -> bool:
    value = np.asarray(candidate, dtype=np.float64)
    return bool(value.shape == nominal.shape and not np.array_equal(value, nominal))


def _simulator_trial_pass(
    trial: Mapping[str, Any],
    *,
    minimum_progress_m: float,
    simulator_safety_margin_m: float,
    maximum_target_displacement_m: float,
    maximum_obstacle_displacement_m: float,
) -> tuple[bool, list[str]]:
    reasons = []
    reach = trial.get("reach")
    if not isinstance(reach, Mapping):
        return False, ["missing_reach_annotation"]
    if float(trial.get("clearance_m", float("-inf"))) < simulator_safety_margin_m:
        reasons.append("clearance_below_margin")
    if bool(trial.get("contact", True)):
        reasons.append("physical_contact")
    if float(reach.get("reach_progress_m", float("-inf"))) < minimum_progress_m:
        reasons.append("progress_below_threshold")
    if (
        float(reach.get("maximum_target_displacement_m", float("inf")))
        > maximum_target_displacement_m
    ):
        reasons.append("target_moved")
    if (
        float(reach.get("maximum_active_obstacle_displacement_m", float("inf")))
        > maximum_obstacle_displacement_m
    ):
        reasons.append("obstacle_moved")
    if int(trial.get("measurement_samples", -1)) != EXPECTED_MEASUREMENT_SAMPLES:
        reasons.append("wrong_measurement_sample_count")
    return not reasons, reasons


def _annotated_rollout(
    environment: SafeLiberoCase,
    actions: np.ndarray,
    initial: ReachSnapshot,
    target_name: str,
) -> dict[str, Any]:
    obstacle_name = environment.obstacle_name
    if obstacle_name is None:
        raise RuntimeError("active obstacle was not resolved")
    rollout, reach = annotate_reach_rollout(
        environment,
        actions,
        target_name=target_name,
        obstacle_name=obstacle_name,
        initial_snapshot=initial,
    )
    return {**rollout, "reach": reach}


def _exact_repeated_rollout(values: Sequence[Mapping[str, Any]]) -> bool:
    return bool(values and all(content_hash(value) == content_hash(values[0]) for value in values[1:]))


def _verification_shortlist(
    search: Mapping[str, Any],
    *,
    limit: int,
) -> list[Mapping[str, Any]]:
    """Keep proxy witnesses and reserve slots for proxy-error controls.

    Restricting direct replay to proxy-feasible actions would make a negative
    R01 result circular: a D_opt false negative could never be observed.  The
    shortlist therefore always attempts the proxy best point plus stationary,
    maximum-clearance, and maximum-progress controls when available.
    """
    candidates = list(search.get("candidates", []))
    controls = list(search.get("controls", []))
    best_attempt = search.get("best_attempt")
    maximum_clearance_attempt = search.get("maximum_clearance_attempt")
    maximum_progress_attempt = search.get("maximum_progress_attempt")
    reserved: list[Mapping[str, Any]] = []
    if isinstance(best_attempt, Mapping):
        reserved.append(best_attempt)
    if isinstance(maximum_clearance_attempt, Mapping):
        reserved.append(maximum_clearance_attempt)
    if isinstance(maximum_progress_attempt, Mapping):
        reserved.append(maximum_progress_attempt)
    stationary = next(
        (item for item in controls if isinstance(item, Mapping) and item.get("source") == "stationary"),
        None,
    )
    if stationary is not None:
        reserved.append(stationary)
    numeric_controls = [item for item in controls if isinstance(item, Mapping)]
    ordered = candidates[: max(0, limit - len(reserved))] + reserved + candidates + numeric_controls
    selected: list[Mapping[str, Any]] = []
    seen = set()
    for item in ordered:
        actions = item.get("actions") if isinstance(item, Mapping) else None
        if not isinstance(actions, list):
            continue
        identity = content_hash(actions)
        if identity in seen:
            continue
        seen.add(identity)
        selected.append(item)
        if len(selected) == limit:
            break
    return selected


def _verify_candidates(
    environment: SafeLiberoCase,
    candidates: Sequence[Mapping[str, Any]],
    nominal: np.ndarray,
    initial: ReachSnapshot,
    config: EndpointFreeConfig,
    *,
    minimum_progress_m: float,
) -> dict[str, Any]:
    attempts = []
    selected = None
    for candidate_index, candidate in enumerate(
        candidates[: config.simulator_verification_candidates]
    ):
        actions = np.asarray(candidate["actions"], dtype=np.float64)
        nontranslation_preserved = _nontranslation_preserved(actions, nominal)
        repeats = [
            _annotated_rollout(environment, actions, initial, config.target_name)
            for _ in range(config.simulator_verification_repeats)
        ]
        replay_exact = _exact_repeated_rollout(repeats)
        trial_checks = [
            _simulator_trial_pass(
                trial,
                minimum_progress_m=minimum_progress_m,
                simulator_safety_margin_m=config.simulator_safety_margin_m,
                maximum_target_displacement_m=config.maximum_target_displacement_m,
                maximum_obstacle_displacement_m=config.maximum_obstacle_displacement_m,
            )
            for trial in repeats
        ]
        p_min_trial_checks = [
            _simulator_trial_pass(
                trial,
                minimum_progress_m=config.p_min_m,
                simulator_safety_margin_m=config.simulator_safety_margin_m,
                maximum_target_displacement_m=config.maximum_target_displacement_m,
                maximum_obstacle_displacement_m=config.maximum_obstacle_displacement_m,
            )
            for trial in repeats
        ]
        p_zero_trial_checks = [
            _simulator_trial_pass(
                trial,
                minimum_progress_m=0.0,
                simulator_safety_margin_m=config.simulator_safety_margin_m,
                maximum_target_displacement_m=config.maximum_target_displacement_m,
                maximum_obstacle_displacement_m=config.maximum_obstacle_displacement_m,
            )
            for trial in repeats
        ]
        reasons = sorted({reason for _, item_reasons in trial_checks for reason in item_reasons})
        if not nontranslation_preserved:
            reasons.append("nontranslation_changed")
        if not replay_exact:
            reasons.append("direct_replay_not_exact")
        action_changed = _action_changed(actions, nominal)
        verified = bool(nontranslation_preserved and replay_exact and all(item[0] for item in trial_checks))
        verified_at_p_min = bool(
            nontranslation_preserved
            and replay_exact
            and all(item[0] for item in p_min_trial_checks)
        )
        verified_at_p_zero = bool(
            nontranslation_preserved
            and replay_exact
            and all(item[0] for item in p_zero_trial_checks)
        )
        simulator_safety_pass = bool(
            replay_exact
            and all(
                float(trial["clearance_m"]) >= config.simulator_safety_margin_m
                and not bool(trial["contact"])
                and int(trial["measurement_samples"]) == EXPECTED_MEASUREMENT_SAMPLES
                for trial in repeats
            )
        )
        simulator_progress_pass = bool(
            replay_exact
            and all(
                float(trial["reach"]["reach_progress_m"]) >= minimum_progress_m
                for trial in repeats
            )
        )
        scene_stationary = bool(
            replay_exact
            and all(
                float(trial["reach"]["maximum_target_displacement_m"])
                <= config.maximum_target_displacement_m
                and float(trial["reach"]["maximum_active_obstacle_displacement_m"])
                <= config.maximum_obstacle_displacement_m
                for trial in repeats
            )
        )
        proxy_clearance_pass = bool(
            float(candidate["d_opt_m"]) >= config.optimizer_safety_margin_m
        )
        proxy_progress_pass = bool(float(candidate["progress_opt"]) >= minimum_progress_m)
        attempt = {
            "candidate_index": candidate_index,
            "source": candidate["source"],
            "actions": candidate["actions"],
            "d_opt_m": candidate["d_opt_m"],
            "progress_opt_m": candidate["progress_opt"],
            "objective": candidate["objective"],
            "within_h04_calibration_domain": candidate["within_h04_calibration_domain"],
            "proxy_clearance_pass": proxy_clearance_pass,
            "proxy_progress_pass": proxy_progress_pass,
            "simulator_safety_pass": simulator_safety_pass,
            "simulator_progress_pass": simulator_progress_pass,
            "scene_stationary": scene_stationary,
            "proxy_clearance_false_safe": bool(
                proxy_clearance_pass and not simulator_safety_pass
            ),
            "proxy_clearance_false_negative": bool(
                not proxy_clearance_pass and simulator_safety_pass
            ),
            "proxy_joint_false_negative_at_search_threshold": bool(
                not (proxy_clearance_pass and proxy_progress_pass) and verified
            ),
            "action_changed_from_nominal": action_changed,
            "nontranslation_preserved": nontranslation_preserved,
            "direct_replay_exact": replay_exact,
            "verified": verified,
            "verified_at_p_min": verified_at_p_min,
            "verified_at_p_zero": verified_at_p_zero,
            "changed_witness_at_p_min": bool(action_changed and verified_at_p_min),
            "changed_witness_at_p_zero": bool(action_changed and verified_at_p_zero),
            "rejection_reasons": sorted(set(reasons)),
            "trials": repeats,
        }
        attempts.append(attempt)
        if verified and selected is None:
            selected = attempt
    return {
        "minimum_progress_m": float(minimum_progress_m),
        "attempted_candidates": len(attempts),
        "verified": selected is not None,
        "selected": selected,
        "verified_at_p_min": any(bool(item["verified_at_p_min"]) for item in attempts),
        "verified_at_p_zero": any(bool(item["verified_at_p_zero"]) for item in attempts),
        "changed_witness_at_p_min": any(
            bool(item["changed_witness_at_p_min"]) for item in attempts
        ),
        "changed_witness_at_p_zero": any(
            bool(item["changed_witness_at_p_zero"]) for item in attempts
        ),
        "proxy_clearance_false_safe_attempts": sum(
            bool(item["proxy_clearance_false_safe"]) for item in attempts
        ),
        "proxy_clearance_false_negative_attempts": sum(
            bool(item["proxy_clearance_false_negative"]) for item in attempts
        ),
        "proxy_joint_false_negative_attempts": sum(
            bool(item["proxy_joint_false_negative_at_search_threshold"]) for item in attempts
        ),
        "attempts": attempts,
    }


def _first_verified_attempt(
    verification: Mapping[str, Mapping[str, Any]],
    field: str,
) -> dict[str, Any] | None:
    for search_name in ("p_min", "p_zero"):
        for attempt in verification[search_name].get("attempts", []):
            if bool(attempt.get(field, False)):
                return {
                    "search": search_name,
                    "candidate_index": attempt["candidate_index"],
                    "source": attempt["source"],
                    "actions_sha256": content_hash(attempt["actions"]),
                }
    return None


def _pooled_witness_outcome(
    verification: Mapping[str, Mapping[str, Any]],
    *,
    nominal_collision_reproduced: bool,
) -> dict[str, Any]:
    """Pool direct replays across both searches before deciding either threshold."""
    p_min_any_action_verified = any(
        bool(verification[name]["verified_at_p_min"]) for name in SEARCH_KEYS
    )
    p_zero_any_action_verified = any(
        bool(verification[name]["verified_at_p_zero"]) for name in SEARCH_KEYS
    )
    p_min_changed_witness_verified = any(
        bool(verification[name]["changed_witness_at_p_min"]) for name in SEARCH_KEYS
    )
    p_zero_changed_witness_verified = any(
        bool(verification[name]["changed_witness_at_p_zero"]) for name in SEARCH_KEYS
    )
    return {
        "p_min_any_action_verified": p_min_any_action_verified,
        "p_zero_any_action_verified": p_zero_any_action_verified,
        "p_min_changed_witness_verified": p_min_changed_witness_verified,
        "p_zero_changed_witness_verified": p_zero_changed_witness_verified,
        "p_min_verified": bool(
            nominal_collision_reproduced and p_min_changed_witness_verified
        ),
        "p_zero_verified": bool(
            nominal_collision_reproduced and p_zero_changed_witness_verified
        ),
        "selected_p_min_changed_witness": _first_verified_attempt(
            verification, "changed_witness_at_p_min"
        ),
        "selected_p_zero_changed_witness": _first_verified_attempt(
            verification, "changed_witness_at_p_zero"
        ),
    }


def validate_endpoint_free_result(value: Mapping[str, Any]) -> list[str]:
    """Dependency-free validation used both before commit and during resume."""
    errors: list[str] = []
    missing = RESULT_FIELDS - set(value)
    if missing:
        errors.append(f"missing required fields: {sorted(missing)}")
    unexpected = set(value) - RESULT_FIELDS
    if unexpected:
        errors.append(f"unexpected top-level fields: {sorted(unexpected)}")
    if value.get("schema_version") != "1.0" or value.get("gate") != "R01":
        errors.append("result must be an R01 schema_version 1.0 artifact")
    if value.get("status") not in FINAL_STATUSES:
        errors.append(f"invalid R01 final status: {value.get('status')!r}")
    for key in ("case_id", "run_id"):
        if not isinstance(value.get(key), str) or not value.get(key):
            errors.append(f"{key} must be a non-empty string")
    config_hash = value.get("config_hash")
    if not _is_sha256(config_hash):
        errors.append("config_hash must be a lowercase SHA-256 digest")
    provenance = value.get("provenance")
    provenance_required = {
        "git_commit",
        "git_dirty",
        "baseline_commit",
        "task_suite",
        "safety_level",
        "task_index",
        "episode_index",
        "group_id",
        "environment_seed",
        "policy_seed",
        "random_control_seed",
        "planner_seed_p_min",
        "planner_seed_p_zero",
        "case_record",
        "checkpoint_sha256",
        "checkpoint_id",
        "noise_sha256",
        "input_manifest_sha256",
        "sampler_steps",
        "intervention_step",
        "model_action_horizon",
        "executed_action_horizon",
        "model_action_dimension",
        "policy_determinism",
        "nominal_simulator_replay_exact",
        "slurm_job_id",
        "slurm_array_task_id",
        "partition",
        "host",
        "device",
        "case_record_sha256",
        "r00_summary_sha256",
        "action_frame",
        "normalization_space",
        "translation_action_bounds",
        "d_opt_model",
        "d_sim_model",
        "scene_motion_measurement",
        "optimizer_safety_margin_m",
        "simulator_safety_margin_m",
        "simulator_verification_repeats",
        "measurement_samples_per_trial",
        "search_registration",
    }
    if not isinstance(provenance, Mapping):
        errors.append("provenance must be an object")
    else:
        absent = provenance_required - set(provenance)
        if absent:
            errors.append(f"provenance missing fields: {sorted(absent)}")
        for key in ("checkpoint_sha256", "noise_sha256", "input_manifest_sha256", "case_record_sha256", "r00_summary_sha256"):
            if key in provenance and not _is_sha256(provenance[key]):
                errors.append(f"provenance.{key} must be a lowercase SHA-256 digest")
        case_record = provenance.get("case_record")
        if not isinstance(case_record, Mapping):
            errors.append("provenance.case_record must be an object")
        else:
            if provenance.get("case_record_sha256") != content_hash(dict(case_record)):
                errors.append("provenance.case_record_sha256 does not match case_record")
            if case_record.get("case_id") != value.get("case_id"):
                errors.append("provenance.case_record does not match the result case_id")
        for key in (
            "environment_seed",
            "policy_seed",
            "random_control_seed",
            "planner_seed_p_min",
            "planner_seed_p_zero",
        ):
            if not isinstance(provenance.get(key), int) or isinstance(provenance.get(key), bool):
                errors.append(f"provenance.{key} must be an integer")
        if provenance.get("planner_seed_p_min") != provenance.get("planner_seed_p_zero"):
            errors.append("p_min and p_zero searches must use the paired planner seed")
        if not isinstance(provenance.get("search_registration"), Mapping):
            errors.append("provenance.search_registration must be an object")
        for key in ("slurm_job_id", "slurm_array_task_id", "partition", "host", "device"):
            if not isinstance(provenance.get(key), str) or not provenance.get(key):
                errors.append(f"provenance.{key} must be a non-empty allocation value")
        if provenance.get("model_action_horizon") != 10:
            errors.append("provenance.model_action_horizon must be 10")
        if provenance.get("executed_action_horizon") != EXECUTED_REACH_ACTIONS:
            errors.append("provenance.executed_action_horizon must be 5")
        if provenance.get("translation_action_bounds") != [-1.0, 1.0]:
            errors.append("provenance.translation_action_bounds must be [-1, 1]")
        if provenance.get("scene_motion_measurement") != (
            "maximum direct MuJoCo body displacement over 125 substeps"
        ):
            errors.append("provenance.scene_motion_measurement must cover all 125 substeps")
        if provenance.get("measurement_samples_per_trial") != EXPECTED_MEASUREMENT_SAMPLES:
            errors.append("provenance.measurement_samples_per_trial must be 126")
        if provenance.get("simulator_verification_repeats", 0) < 2:
            errors.append("provenance.simulator_verification_repeats must be at least two")
        determinism = provenance.get("policy_determinism")
        if not isinstance(determinism, Mapping) or determinism.get("passed") is not True:
            errors.append("provenance.policy_determinism must record an exact passing replay")
        if provenance.get("nominal_simulator_replay_exact") is not True:
            errors.append("provenance.nominal_simulator_replay_exact must be true")

    nominal = value.get("nominal")
    if not isinstance(nominal, Mapping):
        errors.append("nominal must be an object")
    else:
        nominal_required = {
            "actions",
            "actions_sha256",
            "full_model_actions_sha256",
            "branch_snapshot",
            "collision_reproduced",
            "fails_registered_margin",
            "repeats",
        }
        absent = nominal_required - set(nominal)
        if absent:
            errors.append(f"nominal missing fields: {sorted(absent)}")
        actions = nominal.get("actions")
        if not (
            isinstance(actions, list)
            and len(actions) == EXECUTED_REACH_ACTIONS
            and all(
                isinstance(row, list)
                and len(row) == 7
                and all(_is_finite_json_number(item) for item in row)
                for row in actions
            )
        ):
            errors.append("nominal.actions must be a finite 5x7 array")
        elif nominal.get("actions_sha256") != _array_hash(
            np.asarray(actions, dtype=np.float64)
        ):
            errors.append("nominal.actions_sha256 does not match nominal.actions")
        for key in ("actions_sha256", "full_model_actions_sha256"):
            if key in nominal and not _is_sha256(nominal[key]):
                errors.append(f"nominal.{key} must be a lowercase SHA-256 digest")
        if "branch_snapshot" in nominal and not isinstance(nominal["branch_snapshot"], Mapping):
            errors.append("nominal.branch_snapshot must be an object")
        for key in ("collision_reproduced", "fails_registered_margin"):
            if not isinstance(nominal.get(key), bool):
                errors.append(f"nominal.{key} must be boolean")
        repeats = nominal.get("repeats")
        if not isinstance(repeats, list) or len(repeats) < 2 or not all(
            isinstance(repeat, Mapping) for repeat in repeats
        ):
            errors.append("nominal.repeats must contain at least two rollout objects")

    calibration = value.get("calibration")
    if not isinstance(calibration, Mapping):
        errors.append("calibration must be an object")
    else:
        calibration_required = {"gate", "summary_sha256", "p_min_m"}
        absent = calibration_required - set(calibration)
        if absent:
            errors.append(f"calibration missing fields: {sorted(absent)}")
        if calibration.get("gate") != "R00":
            errors.append("calibration.gate must be R00")
        if "summary_sha256" in calibration and not _is_sha256(calibration["summary_sha256"]):
            errors.append("calibration.summary_sha256 must be a lowercase SHA-256 digest")
        if isinstance(provenance, Mapping) and calibration.get(
            "summary_sha256"
        ) != provenance.get("r00_summary_sha256"):
            errors.append("calibration summary hash conflicts with provenance")
        if not _is_finite_json_number(calibration.get("p_min_m")) or float(
            calibration.get("p_min_m", 0.0)
        ) <= 0.0:
            errors.append("calibration.p_min_m must be positive and finite")

    searches = value.get("searches")
    if not isinstance(searches, Mapping) or set(searches) != SEARCH_KEYS:
        errors.append("searches must contain exactly p_min and p_zero")
    else:
        for name in sorted(SEARCH_KEYS):
            search = searches[name]
            if not isinstance(search, Mapping):
                errors.append(f"searches.{name} must be an object")
                continue
            search_required = {
                "outcome",
                "candidates",
                "best_attempt",
                "maximum_clearance_attempt",
                "maximum_progress_attempt",
                "evaluations",
                "evaluation_budget",
            }
            absent = search_required - set(search)
            if absent:
                errors.append(f"searches.{name} missing fields: {sorted(absent)}")
            if search.get("outcome") not in {"candidate_found", "not_found_within_budget"}:
                errors.append(f"searches.{name} has an invalid bounded-search outcome")
            candidates = search.get("candidates")
            if not isinstance(candidates, list) or not all(
                isinstance(candidate, Mapping) for candidate in candidates
            ):
                errors.append(f"searches.{name}.candidates must be an array of objects")
            elif (search.get("outcome") == "candidate_found") != bool(candidates):
                errors.append(f"searches.{name} outcome conflicts with its candidate list")
            for key in (
                "best_attempt",
                "maximum_clearance_attempt",
                "maximum_progress_attempt",
            ):
                if key in search and not isinstance(search[key], Mapping):
                    errors.append(f"searches.{name}.{key} must be an object")
            for key in ("evaluations", "evaluation_budget"):
                item = search.get(key)
                if not isinstance(item, int) or isinstance(item, bool) or item < 1:
                    errors.append(f"searches.{name}.{key} must be a positive integer")

    verification = value.get("verification")
    if not isinstance(verification, Mapping) or set(verification) != SEARCH_KEYS:
        errors.append("verification must contain exactly p_min and p_zero")
    else:
        for name in sorted(SEARCH_KEYS):
            item = verification[name]
            if not isinstance(item, Mapping):
                errors.append(f"verification.{name} must be an object")
                continue
            verification_required = {
                "minimum_progress_m",
                "attempted_candidates",
                "verified",
                "verified_at_p_min",
                "verified_at_p_zero",
                "changed_witness_at_p_min",
                "changed_witness_at_p_zero",
                "selected",
                "attempts",
            }
            absent = verification_required - set(item)
            if absent:
                errors.append(f"verification.{name} missing fields: {sorted(absent)}")
            if not _is_finite_json_number(item.get("minimum_progress_m")):
                errors.append(f"verification.{name}.minimum_progress_m must be finite")
            attempted = item.get("attempted_candidates")
            attempts = item.get("attempts")
            if not isinstance(attempted, int) or isinstance(attempted, bool) or attempted < 0:
                errors.append(f"verification.{name}.attempted_candidates must be non-negative")
            if not isinstance(attempts, list) or not all(
                isinstance(attempt, Mapping) for attempt in attempts
            ):
                errors.append(f"verification.{name}.attempts must be an array of objects")
            elif isinstance(attempted, int) and not isinstance(attempted, bool) and attempted != len(attempts):
                errors.append(f"verification.{name}.attempted_candidates conflicts with attempts")
            if not isinstance(item.get("verified"), bool):
                errors.append(f"verification.{name}.verified must be boolean")
            elif item["verified"] and not isinstance(item.get("selected"), Mapping):
                errors.append(f"verification.{name}.selected must record the verified witness")
            elif not item["verified"] and item.get("selected") is not None:
                errors.append(f"verification.{name}.selected must be null without a witness")
            for key in (
                "verified_at_p_min",
                "verified_at_p_zero",
                "changed_witness_at_p_min",
                "changed_witness_at_p_zero",
            ):
                if not isinstance(item.get(key), bool):
                    errors.append(f"verification.{name}.{key} must be boolean")
                elif isinstance(attempts, list) and all(
                    isinstance(attempt, Mapping) for attempt in attempts
                ):
                    expected = any(bool(attempt.get(key, False)) for attempt in attempts)
                    if item[key] != expected:
                        errors.append(
                            f"verification.{name}.{key} conflicts with its attempts"
                        )
            if item.get("changed_witness_at_p_min") and not item.get("verified_at_p_min"):
                errors.append(
                    f"verification.{name} changed p_min witness must also be verified"
                )
            if item.get("changed_witness_at_p_zero") and not item.get("verified_at_p_zero"):
                errors.append(
                    f"verification.{name} changed p_zero witness must also be verified"
                )

    outcome = value.get("outcome")
    if not isinstance(outcome, Mapping):
        errors.append("outcome must be an object")
    else:
        for key in (
            "p_min_verified",
            "p_zero_verified",
            "nominal_collision_reproduced",
            "nominal_fails_registered_margin",
            "p_min_any_action_verified",
            "p_zero_any_action_verified",
            "p_min_changed_witness_verified",
            "p_zero_changed_witness_verified",
        ):
            if not isinstance(outcome.get(key), bool):
                errors.append(f"outcome.{key} must be boolean")
        if not isinstance(outcome.get("interpretation"), str) or not outcome.get("interpretation"):
            errors.append("outcome.interpretation must be a non-empty string")
        if isinstance(nominal, Mapping) and isinstance(
            nominal.get("collision_reproduced"), bool
        ):
            if outcome.get("nominal_collision_reproduced") != nominal["collision_reproduced"]:
                errors.append("outcome nominal collision flag conflicts with nominal")
            if outcome.get("nominal_fails_registered_margin") != nominal.get(
                "fails_registered_margin"
            ):
                errors.append("outcome nominal margin flag conflicts with nominal")
            if (
                isinstance(verification, Mapping)
                and set(verification) == SEARCH_KEYS
                and all(isinstance(verification[name], Mapping) for name in SEARCH_KEYS)
            ):
                expected_outcome = _pooled_witness_outcome(
                    verification,
                    nominal_collision_reproduced=nominal["collision_reproduced"],
                )
                for key in (
                    "p_min_verified",
                    "p_zero_verified",
                    "p_min_any_action_verified",
                    "p_zero_any_action_verified",
                    "p_min_changed_witness_verified",
                    "p_zero_changed_witness_verified",
                ):
                    if outcome.get(key) != expected_outcome[key]:
                        errors.append(f"outcome.{key} conflicts with pooled direct verification")
        if isinstance(outcome.get("p_min_verified"), bool):
            if value.get("status") == "verified_safe_progress" and not outcome["p_min_verified"]:
                errors.append("verified_safe_progress status requires a verified p_min witness")
            elif value.get("status") == "no_verified_safe_progress" and outcome["p_min_verified"]:
                errors.append("no_verified_safe_progress status conflicts with its outcome")
            elif value.get("status") == "nominal_collision_not_reproduced" and outcome.get(
                "nominal_collision_reproduced"
            ):
                errors.append("nominal_collision_not_reproduced status conflicts with its outcome")
            elif value.get("status") != "nominal_collision_not_reproduced" and outcome.get(
                "nominal_collision_reproduced"
            ) is False:
                errors.append("R01 rescue statuses require the nominal collision to reproduce")
    return errors


def valid_endpoint_free_completion(
    path: str | Path,
    *,
    case_id: str | None = None,
    run_id: str | None = None,
    config_hash: str | None = None,
    input_manifest_sha256: str | None = None,
) -> bool:
    try:
        value = load_json(path)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return False
    if not isinstance(value, Mapping) or validate_endpoint_free_result(value):
        return False
    if case_id is not None and value.get("case_id") != case_id:
        return False
    if run_id is not None and value.get("run_id") != run_id:
        return False
    if config_hash is not None and value.get("config_hash") != config_hash:
        return False
    if input_manifest_sha256 is not None:
        provenance = value.get("provenance")
        if not isinstance(provenance, Mapping) or provenance.get(
            "input_manifest_sha256"
        ) != input_manifest_sha256:
            return False
    return True


def run_endpoint_free_case(
    case: Mapping[str, Any],
    config: EndpointFreeConfig,
    *,
    repo_root: str | Path,
    input_manifest_sha256: str,
    client=None,
    environment: SafeLiberoCase | None = None,
) -> tuple[Path, str]:
    """Run one frozen H05 identity through the two R01 searches and D_sim."""
    oracle = config.oracle
    root = Path(repo_root).resolve()
    config_hash = content_hash(_normalized_config(config))
    output = (
        Path(oracle.output_root)
        / oracle.run_id
        / str(case["case_id"])
        / "endpoint-free-feasibility.json"
    )
    if valid_endpoint_free_completion(
        output,
        case_id=str(case["case_id"]),
        run_id=oracle.run_id,
        config_hash=config_hash,
        input_manifest_sha256=input_manifest_sha256,
    ):
        return output, "skipped_valid_completion"

    if str(case.get("task_suite")) != "safelibero_spatial" or str(case.get("safety_level")) != "II":
        raise ValueError("R01 case is outside the frozen SafeLIBERO Spatial Level-II population")
    if int(case.get("task_index", -1)) != 0:
        raise ValueError("R01 case must be task 0")

    git_commit, git_dirty = _git_state(root)
    noise = np.random.default_rng(int(case["policy_seed"])).normal(
        size=(oracle.action_horizon, oracle.action_dim)
    ).astype(np.float32)
    if client is None:
        from openpi_client import websocket_client_policy

        client = websocket_client_policy.WebsocketClientPolicy(oracle.host, oracle.port)
    owns_environment = environment is None
    if environment is None:
        environment = SafeLiberoCase(dict(case), oracle)
    else:
        environment.configure_case(dict(case))

    try:
        initial_observation = environment.reset_and_settle()
        obstacle_name = environment.obstacle_name
        if obstacle_name is None:
            raise RuntimeError("failed to resolve the active obstacle")
        initial = capture_reach_snapshot(environment, config.target_name, obstacle_name)
        if _target_contact_at_branch(environment, config.target_name) or environment.env.check_success():
            raise RuntimeError("frozen R01 branch is not a valid pre-grasp reach state")

        policy_input = policy_observation(initial_observation, environment.prompt, oracle.resize_size)
        first_reply = _infer(client, policy_input, noise, oracle, intervention_mode="none")
        second_reply = _infer(client, policy_input, noise, oracle, intervention_mode="none")
        policy_determinism = _determinism_check(first_reply, second_reply)
        if not policy_determinism["passed"]:
            raise RuntimeError(f"fixed observation/noise policy replay is not exact: {policy_determinism}")
        full_actions = np.asarray(first_reply["actions"], dtype=np.float64)
        if full_actions.shape[0] != 10 or full_actions.shape[1] < 7:
            raise RuntimeError(f"expected a 10x>=7 model action chunk, got {full_actions.shape}")
        nominal = full_actions[:EXECUTED_REACH_ACTIONS, :7]

        nominal_repeats = [
            _annotated_rollout(environment, nominal, initial, config.target_name)
            for _ in range(max(2, oracle.measurement_repeats))
        ]
        nominal_replay_exact = _exact_repeated_rollout(nominal_repeats)
        if not nominal_replay_exact:
            raise RuntimeError("paired nominal endpoint-free branch replay is not exact")
        if any(
            int(trial["measurement_samples"]) != EXPECTED_MEASUREMENT_SAMPLES
            for trial in nominal_repeats
        ):
            raise RuntimeError("five-action nominal rollout must contain 126 measurements")
        nominal_rollout = nominal_repeats[0]
        nominal_collision_reproduced = bool(
            float(nominal_rollout["clearance_m"]) < 0.0
            or bool(nominal_rollout["contact"])
        )
        nominal_fails_registered_margin = bool(
            float(nominal_rollout["clearance_m"]) < config.simulator_safety_margin_m
            or bool(nominal_rollout["contact"])
        )

        start_center = np.asarray(nominal_rollout["start_eef_center_m"], dtype=np.float64)
        center_to_site = np.asarray(initial.eef_world_m, dtype=np.float64) - start_center
        branch_target = np.asarray(initial.target_world_m, dtype=np.float64)
        start_distance = float(np.linalg.norm(np.asarray(initial.eef_world_m) - branch_target))

        def progress_fn(points: tuple[tuple[float, float, float], ...]) -> float:
            predicted_site = np.asarray(points[-1], dtype=np.float64) + center_to_site
            return start_distance - float(np.linalg.norm(predicted_site - branch_target))

        common_search = {
            "nominal_prefix": nominal,
            "start_eef_center_m": start_center,
            "response_matrix": config.response_matrix_m_per_action,
            "obstacle_boxes": nominal_rollout["branch_obstacle_boxes"],
            "eef_radius_m": oracle.eef_radius_m,
            "progress_fn": progress_fn,
            "optimizer_clearance_margin_m": config.optimizer_safety_margin_m,
            "action_low": config.translation_action_low,
            "action_high": config.translation_action_high,
            **dict(config.search_kwargs),
        }
        p_min_search = solve_endpoint_free_projection(
            **common_search,
            minimum_progress=config.p_min_m,
            seed=config.planner_seed,
        )
        p_zero_search = solve_endpoint_free_projection(
            **common_search,
            minimum_progress=0.0,
            seed=config.planner_seed,
        )

        searches = {
            "p_min": p_min_search.to_dict(),
            "p_zero": p_zero_search.to_dict(),
        }
        verification = {
            "p_min": _verify_candidates(
                environment,
                _verification_shortlist(
                    searches["p_min"], limit=config.simulator_verification_candidates
                ),
                nominal,
                initial,
                config,
                minimum_progress_m=config.p_min_m,
            ),
            "p_zero": _verify_candidates(
                environment,
                _verification_shortlist(
                    searches["p_zero"], limit=config.simulator_verification_candidates
                ),
                nominal,
                initial,
                config,
                minimum_progress_m=0.0,
            ),
        }
        pooled_outcome = _pooled_witness_outcome(
            verification,
            nominal_collision_reproduced=nominal_collision_reproduced,
        )
        p_min_verified = bool(pooled_outcome["p_min_verified"])
        p_zero_verified = bool(pooled_outcome["p_zero_verified"])
        if not nominal_collision_reproduced:
            status = "nominal_collision_not_reproduced"
        elif p_min_verified:
            status = "verified_safe_progress"
        else:
            status = "no_verified_safe_progress"

        provenance = {
            "evidence_tier": "real_safelibero_endpoint_free_preliminary",
            "git_commit": git_commit,
            "git_dirty": git_dirty,
            "baseline_commit": "57b1aef306f212aea3574b0a3b64aa1a3d8f5e4b",
            "python_version": platform.python_version(),
            "host": socket.gethostname(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "task_suite": case["task_suite"],
            "safety_level": case["safety_level"],
            "task_index": case["task_index"],
            "episode_index": case["episode_index"],
            "case_record": dict(case),
            "case_record_sha256": content_hash(dict(case)),
            "environment_seed": case["environment_seed"],
            "policy_seed": case["policy_seed"],
            "random_control_seed": case["random_control_seed"],
            "planner_seed_p_min": config.planner_seed,
            "planner_seed_p_zero": config.planner_seed,
            "group_id": case["group_id"],
            "checkpoint_id": oracle.checkpoint_id,
            "checkpoint_sha256": oracle.checkpoint_sha256,
            "noise_sha256": _array_hash(noise),
            "input_manifest_sha256": input_manifest_sha256,
            "r00_summary_sha256": config.r00_summary_sha256,
            "sampler_steps": oracle.sampler_steps,
            "intervention_step": oracle.intervention_step,
            "intervention_mode": "none; direct candidate action execution only",
            "model_action_horizon": oracle.action_horizon,
            "executed_action_horizon": oracle.executed_prefix,
            "model_action_dimension": oracle.action_dim,
            "action_frame": "world-frame OSC translation delta",
            "policy_action_space": "unnormalized LIBERO controller action",
            "normalization_space": (
                "unnormalized policy action output; physical displacement is scale-only; "
                "no normalization mean is subtracted"
            ),
            "controlled_dimensions": "translation only; rotation and gripper copied exactly from nominal",
            "translation_action_bounds": [
                config.translation_action_low,
                config.translation_action_high,
            ],
            "d_opt_model": "frozen H04 response matrix plus static branch oriented-box geometry",
            "d_sim_model": "physics-substep EEF-sphere/oriented-box clearance plus contact",
            "scene_motion_measurement": "maximum direct MuJoCo body displacement over 125 substeps",
            "response_matrix_m_per_action": [list(row) for row in config.response_matrix_m_per_action],
            "optimizer_safety_margin_m": config.optimizer_safety_margin_m,
            "simulator_safety_margin_m": config.simulator_safety_margin_m,
            "simulator_verification_repeats": config.simulator_verification_repeats,
            "measurement_samples_per_trial": EXPECTED_MEASUREMENT_SAMPLES,
            "search_registration": {
                "method": "deterministic_cem_multistart",
                "paired_seed": config.planner_seed,
                "translation_action_bounds": [
                    config.translation_action_low,
                    config.translation_action_high,
                ],
                "p_min_m": config.p_min_m,
                "p_zero_m": 0.0,
                "optimizer_safety_margin_m": config.optimizer_safety_margin_m,
                "simulator_safety_margin_m": config.simulator_safety_margin_m,
                "simulator_verification_candidates": config.simulator_verification_candidates,
                "simulator_verification_repeats": config.simulator_verification_repeats,
                "search_kwargs": dict(config.search_kwargs),
            },
            "policy_determinism": policy_determinism,
            "nominal_simulator_replay_exact": nominal_replay_exact,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "partition": os.environ.get("SLURM_JOB_PARTITION"),
            "device": os.environ.get("CUDA_VISIBLE_DEVICES"),
        }
        result = {
            "schema_version": "1.0",
            "gate": "R01",
            "case_id": case["case_id"],
            "run_id": oracle.run_id,
            "status": status,
            "config_hash": config_hash,
            "provenance": provenance,
            "calibration": {
                "gate": "R00",
                "summary_path": config.r00_summary_path,
                "summary_sha256": config.r00_summary_sha256,
                "p_min_m": config.p_min_m,
                "quantile": config.r00_summary["calibration"]["quantile"],
                "eligible_positive_examples": config.r00_summary["eligible_positive_examples"],
                "calibration_groups": config.r00_summary["calibration_groups"],
                "evaluation_groups": config.r00_summary["evaluation_groups"],
            },
            "nominal": {
                "actions": nominal.tolist(),
                "actions_sha256": _array_hash(nominal),
                "full_model_actions_sha256": _array_hash(full_actions),
                "branch_snapshot": initial.to_dict(),
                "collision_reproduced": nominal_collision_reproduced,
                "fails_registered_margin": nominal_fails_registered_margin,
                "repeats": nominal_repeats,
            },
            "searches": searches,
            "verification": verification,
            "outcome": {
                "p_min_verified": p_min_verified,
                "p_zero_verified": p_zero_verified,
                "nominal_collision_reproduced": nominal_collision_reproduced,
                "nominal_fails_registered_margin": nominal_fails_registered_margin,
                **pooled_outcome,
                "proxy_clearance_false_safe_attempts": sum(
                    int(verification[name]["proxy_clearance_false_safe_attempts"])
                    for name in SEARCH_KEYS
                ),
                "proxy_clearance_false_negative_attempts": sum(
                    int(verification[name]["proxy_clearance_false_negative_attempts"])
                    for name in SEARCH_KEYS
                ),
                "proxy_joint_false_negative_attempts": sum(
                    int(verification[name]["proxy_joint_false_negative_attempts"])
                    for name in SEARCH_KEYS
                ),
                "interpretation": (
                    "nominal collision did not reproduce; case is excluded from the rescue numerator"
                    if not nominal_collision_reproduced
                    else "simulator-verified changed-action endpoint-free safe-progress witness found"
                    if p_min_verified
                    else "no safe-progress witness was verified within the registered bounded searches"
                ),
            },
        }
        errors = validate_endpoint_free_result(result)
        if errors:
            raise RuntimeError("refusing invalid R01 final artifact: " + "; ".join(errors))
        atomic_write_json(output, result)
        return output, status
    finally:
        if owns_environment:
            environment.close()


__all__ = [
    "EndpointFreeConfig",
    "endpoint_free_config_from_mapping",
    "run_endpoint_free_case",
    "valid_endpoint_free_completion",
    "validate_endpoint_free_result",
]
