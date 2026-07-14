"""Paired strong-analytic R03A necessity-test case runner.

R03A is deliberately a development-population kill test.  It consumes the
immutable, validated R02/R03 artifacts and adds two trajectory-wide analytic
collision-field arms.  It does not train a probe and it does not manufacture a
new test population.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import socket
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from crfs_harness.artifacts import (
    atomic_write_json,
    content_hash,
    file_sha256,
    load_json,
    scientific_config,
)

from .progress_calibration import _target_contact_at_branch
from .r02_runner import (
    BASELINE_COMMIT,
    EXPECTED_MEASUREMENT_SAMPLES,
    NORMALIZATION_ASSET_SHA256,
    REGISTERED_DISTANCE_LIMIT_M,
    REGISTERED_EEF_RADIUS_M,
    REGISTERED_RESPONSE_MATRIX_M_PER_ACTION,
    REGISTERED_TRANSLATION_ACTION_SCALE,
    _array_from_record,
    _array_record,
    _bounds_check,
    _evaluated_arm,
    _json_compatible,
    _observation_fingerprint,
    _policy_timing_control,
    _reply_policy_timing,
    _trace_record,
    _validate_array_record,
    validate_r02_result,
)
from .r04_labels import MAXIMUM_OBBS, _capture_branch_geometry
from .reach_progress import TARGET_OBJECT_NAME, capture_reach_snapshot
from .runner import OracleConfig, SafeLiberoCase, _git_state, policy_observation


SCHEMA_VERSION = "1.0"
GATE = "R03A"
ARTIFACT_TYPE = "r03a_analytic_kill_test_case"
ACCEPTED_R03_SUMMARY_SHA256 = (
    "dea3e66c854faa0b659777bfed4b651b19d8d4d0710696ff7bfe52c44d4ee76e"
)
ACCEPTED_R03_ORDERED_RESULTS_SHA256 = (
    "fa79a132f2fecbedab5917111ef71f022e52c933d8e535aaca118add5f5b7895"
)
ACCEPTED_R02_CONFIG_SHA256 = (
    "c5be1b4759487f4d6541e49110b90fcf5de404bdaed5f389358db0abe4388f4e"
)
ACCEPTED_ORIGINAL_MANIFEST_SHA256 = (
    "b12319d3fed151bfee85e1615bd72254474a1480308389d747dce954c6131e41"
)
ACCEPTED_ELIGIBLE_MANIFEST_SHA256 = (
    "241f1b94a5434973dcfb235b43b9386ad15046ff87d35d5b0c34e295c2132916"
)
DECISION_ARTIFACT = "docs/decisions/0023-run-strong-analytic-kill-test.md"
DECISION_SHA256 = (
    "b1b717856ebee8d3e93a21b35606e62ad6cc4baa2b9346f7bd1f415c84b25ead"
)
NEW_ARMS = ("analytic_trajectory_mid", "analytic_trajectory_early")
ARMS = ("frozen",) + NEW_ARMS
INTERVENTION_STEPS = {
    "analytic_trajectory_mid": 5,
    "analytic_trajectory_early": 1,
}
ENERGY_MARGIN_M = 0.005
ENERGY_TEMPERATURE_M = 0.005
SAMPLES_PER_SEGMENT = 26
TIMING_WARMUP_REPEATS = 2
TIMING_MEASURED_REPEATS = 2
SATURATION_CRITERION = (
    "any_executed_first_five_xyz_exactly_equals_registered_inclusive_bound"
)
SATURATION_COMPARISON = "exact_float_equality_no_epsilon_no_clipping"
TERMINAL_FAILURE_CRITERION = (
    "any_simulator_task_success_during_prefix_is_true"
)
EVALUATED_FAILURE_PRECEDENCE = (
    "bounds_failure",
    "saturation_failure",
    "budget_failure",
    "terminal_failure",
    "zero_gradient_failure",
    "ordinary_gate",
)
FINAL_STATUSES = {"completed", "nominal_collision_not_reconfirmed"}


class R03ASourceError(ValueError):
    """Raised when an immutable predecessor or pairing identity is invalid."""


class R03APolicyInferenceError(RuntimeError):
    """Expected policy-server/transport failure before an auditable reply."""


@dataclass(frozen=True)
class R03AConfig:
    """Content-bound scientific and runtime settings for R03A."""

    oracle: OracleConfig
    target_name: str
    p_min_m: float
    simulator_safety_margin_m: float
    maximum_target_displacement_m: float
    maximum_obstacle_displacement_m: float
    simulator_repeats: int
    translation_action_low: float
    translation_action_high: float
    action_scale: Tuple[float, float, float]
    samples_per_segment: int
    source_r03_summary_path: str
    source_r03_summary_sha256: str
    source_r03_summary: Mapping[str, Any]
    source_r03_ordered_result_set_digest: str
    source_r02_results_root: str
    source_r02_result_hashes: Mapping[str, str]
    eligible_case_ids: Tuple[str, ...]
    eligible_manifest_sha256: str
    original_manifest_sha256: str
    r02_config_sha256: str
    decision_artifact: str
    decision_sha256: str
    config_file_sha256: str
    scientific_config_hash: str
    timing_warmup_repeats: int
    timing_measured_repeats: int
    saturation_criterion: str
    saturation_comparison: str
    terminal_failure_criterion: str
    evaluated_failure_precedence: Tuple[str, ...]


def _is_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _finite(value: Any, *, name: str, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric") from error
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "positive and finite" if positive else "finite"
        raise ValueError(f"{name} must be {qualifier}")
    return result


def _resolve(path_value: str, root: str | Path) -> Path:
    path = Path(path_value).expanduser()
    return path.resolve() if path.is_absolute() else (Path(root).resolve() / path).resolve()


def _summary_result_hashes(summary: Mapping[str, Any]) -> Dict[str, str]:
    records = summary.get("result_hashes")
    if not isinstance(records, list):
        raise R03ASourceError("R03 summary result_hashes must be a list")
    result: Dict[str, str] = {}
    for item in records:
        if not isinstance(item, Mapping):
            raise R03ASourceError("R03 result hash entry must be an object")
        case_id = item.get("case_id")
        digest = item.get("sha256")
        if not isinstance(case_id, str) or not case_id or not _is_sha256(digest):
            raise R03ASourceError("R03 result hash entry is invalid")
        if case_id in result:
            raise R03ASourceError(f"duplicate R03 result hash for {case_id}")
        result[case_id] = str(digest)
    return result


def r03a_config_from_mapping(
    value: Mapping[str, Any],
    oracle: OracleConfig,
    *,
    repo_root: str | Path,
    config_file_sha256: Optional[str] = None,
    scientific_config_hash: Optional[str] = None,
) -> R03AConfig:
    """Load R03A only after its protocol and predecessor hashes verify."""

    if not bool(value.get("ready_to_run", False)):
        raise ValueError("R03A config is not ready_to_run")
    settings = value.get("r03a")
    if not isinstance(settings, Mapping):
        raise ValueError("R03A config requires an r03a object")
    if oracle.action_horizon != 10 or oracle.action_dim != 32:
        raise ValueError("R03A requires a 10x32 model action")
    if oracle.executed_prefix != 5 or oracle.sampler_steps != 10:
        raise ValueError("R03A requires five executed actions and ten Euler steps")
    if oracle.response_matrix_m_per_action is None:
        raise ValueError("R03A requires the frozen H04 response matrix")
    response = tuple(
        tuple(float(item) for item in row)
        for row in oracle.response_matrix_m_per_action
    )
    if response != REGISTERED_RESPONSE_MATRIX_M_PER_ACTION:
        raise ValueError("R03A response matrix differs from H04")
    if not math.isclose(
        oracle.eef_radius_m, REGISTERED_EEF_RADIUS_M, rel_tol=0.0, abs_tol=1e-15
    ):
        raise ValueError("R03A EEF radius must remain 6 cm")
    if not math.isclose(
        oracle.distance_limit_m,
        REGISTERED_DISTANCE_LIMIT_M,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError("R03A simulator distance limit must remain 1 m")
    if settings.get("phase") != "pregrasp_reach":
        raise ValueError("R03A is restricted to pregrasp_reach")
    if tuple(settings.get("required_arms", ())) != NEW_ARMS:
        raise ValueError(f"r03a.required_arms must equal {list(NEW_ARMS)!r}")
    observed_steps = settings.get("intervention_steps")
    if not isinstance(observed_steps, Mapping) or {
        str(key): int(item) for key, item in observed_steps.items()
    } != INTERVENTION_STEPS:
        raise ValueError("R03A intervention steps differ from ADR-0023")
    if settings.get("clipping_policy") != "fail_without_clipping":
        raise ValueError("R03A must report rather than clip sampled actions")
    if settings.get("budget_source") != "r02.directions.delta_star_model:first_five_xyz_l2":
        raise ValueError("R03A budget source differs from ADR-0023")
    if not math.isclose(
        _finite(settings.get("energy_margin_m"), name="energy margin", positive=True),
        ENERGY_MARGIN_M,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError("R03A energy margin must remain 5 mm")
    if not math.isclose(
        _finite(
            settings.get("energy_temperature_m"),
            name="energy temperature",
            positive=True,
        ),
        ENERGY_TEMPERATURE_M,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError("R03A softplus temperature must remain 5 mm")
    samples = int(settings.get("samples_per_segment", -1))
    if samples != SAMPLES_PER_SEGMENT:
        raise ValueError("R03A must use 26 equally spaced samples per segment")

    root = Path(repo_root).resolve()
    decision = settings.get("decision_artifact")
    decision_sha = settings.get("decision_sha256")
    if decision != DECISION_ARTIFACT or decision_sha != DECISION_SHA256:
        raise ValueError("R03A must bind the accepted ADR-0023")
    decision_path = _resolve(str(decision), root)
    if file_sha256(decision_path) != decision_sha:
        raise ValueError("R03A ADR-0023 content hash differs")

    summary_value = settings.get("source_r03_summary_artifact")
    summary_sha = settings.get("source_r03_summary_sha256")
    if not isinstance(summary_value, str) or not summary_value:
        raise ValueError("r03a.source_r03_summary_artifact must be a path")
    if summary_sha != ACCEPTED_R03_SUMMARY_SHA256:
        raise ValueError("R03A must bind the accepted R03 summary")
    summary_path = _resolve(summary_value, root)
    if file_sha256(summary_path) != summary_sha:
        raise ValueError("R03A R03 summary content hash differs")
    summary = load_json(summary_path)
    if not isinstance(summary, Mapping):
        raise ValueError("R03 summary must be an object")
    if not (
        summary.get("gate") == "R03"
        and summary.get("status") == "passed"
        and summary.get("gate_passed") is True
        and summary.get("r02_apparatus_passed") is True
    ):
        raise ValueError("R03A source summary is not a passed R03 artifact")
    digest = settings.get("source_r03_ordered_result_set_digest")
    if (
        digest != ACCEPTED_R03_ORDERED_RESULTS_SHA256
        or summary.get("ordered_result_set_digest") != digest
    ):
        raise ValueError("R03A R03 ordered result-set digest differs")
    population = summary.get("population")
    if not isinstance(population, Mapping) or population.get("valid") is not True:
        raise ValueError("R03A source population is not valid")
    eligible_value = population.get("eligible_case_ids")
    if not isinstance(eligible_value, list) or not all(
        isinstance(item, str) and item for item in eligible_value
    ):
        raise ValueError("R03A source has no valid eligible-case list")
    eligible = tuple(str(item) for item in eligible_value)
    if len(eligible) != 17 or len(set(eligible)) != 17:
        raise ValueError("R03A requires the complete 17-case eligible population")
    all_hashes = _summary_result_hashes(summary)
    if not set(eligible).issubset(all_hashes):
        raise ValueError("R03A eligible cases are absent from R03 result hashes")

    identities = summary.get("identities")
    if not isinstance(identities, Mapping):
        raise ValueError("R03 summary has no identities object")
    if identities.get("baseline_commit") != BASELINE_COMMIT:
        raise ValueError("R03A baseline revision differs from R03")
    if identities.get("checkpoint_sha256") != oracle.checkpoint_sha256:
        raise ValueError("R03A checkpoint differs from R03")
    original_manifest_sha = settings.get("source_original_manifest_sha256")
    if (
        original_manifest_sha != ACCEPTED_ORIGINAL_MANIFEST_SHA256
        or identities.get("manifest_sha256") != original_manifest_sha
    ):
        raise ValueError("R03A original manifest binding differs from R03")
    r02_config_sha = settings.get("source_r02_config_sha256")
    if (
        r02_config_sha != ACCEPTED_R02_CONFIG_SHA256
        or identities.get("config_file_sha256") != r02_config_sha
    ):
        raise ValueError("R03A R02 config binding differs from R03")

    results_root_value = settings.get("source_r02_results_root")
    if not isinstance(results_root_value, str) or not results_root_value:
        raise ValueError("r03a.source_r02_results_root must locate raw R02 artifacts")
    results_root = _resolve(results_root_value, root)
    eligible_manifest_sha = settings.get("eligible_manifest_sha256")
    if eligible_manifest_sha != ACCEPTED_ELIGIBLE_MANIFEST_SHA256:
        raise ValueError("R03A eligible manifest hash differs from the frozen subset")

    target_name = str(settings.get("target_object", ""))
    if target_name != TARGET_OBJECT_NAME:
        raise ValueError(f"R03A target must be {TARGET_OBJECT_NAME!r}")
    p_min = _finite(settings.get("minimum_progress_m"), name="R03A p_min", positive=True)
    safety_margin = _finite(
        settings.get("simulator_safety_margin_m"),
        name="R03A simulator margin",
        positive=True,
    )
    if not math.isclose(safety_margin, 0.005, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("R03A simulator margin must remain 5 mm")
    if not math.isclose(
        oracle.safety_margin_m, safety_margin, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("R03A top-level and simulator margins differ")
    target_limit = _finite(
        settings.get("maximum_target_displacement_m"), name="target motion limit"
    )
    obstacle_limit = _finite(
        settings.get("maximum_obstacle_displacement_m"),
        name="obstacle motion limit",
    )
    if not (0.0 <= target_limit <= 0.001 and 0.0 <= obstacle_limit <= 0.001):
        raise ValueError("R03A scene-motion limits must lie in [0,1 mm]")
    repeats = int(settings.get("simulator_repeats", oracle.measurement_repeats))
    if repeats < 2 or repeats != oracle.measurement_repeats:
        raise ValueError("R03A requires at least two declared simulator repeats")
    bounds = settings.get("translation_action_bounds")
    if not isinstance(bounds, Sequence) or isinstance(bounds, (str, bytes)) or len(bounds) != 2:
        raise ValueError("R03A translation_action_bounds must be [low,high]")
    low = _finite(bounds[0], name="translation action low")
    high = _finite(bounds[1], name="translation action high")
    if low != -1.0 or high != 1.0:
        raise ValueError("R03A translation action bounds must remain [-1,1]")
    saturation_criterion = str(settings.get("saturation_criterion", ""))
    saturation_comparison = str(settings.get("saturation_comparison", ""))
    terminal_failure_criterion = str(settings.get("terminal_failure_criterion", ""))
    precedence_value = settings.get("evaluated_failure_precedence")
    if not isinstance(precedence_value, list) or not all(
        isinstance(item, str) for item in precedence_value
    ):
        raise ValueError("R03A evaluated-failure precedence must be a string list")
    evaluated_failure_precedence = tuple(precedence_value)
    if saturation_criterion != SATURATION_CRITERION:
        raise ValueError("R03A saturation criterion differs from ADR-0023")
    if saturation_comparison != SATURATION_COMPARISON:
        raise ValueError("R03A saturation comparison differs from ADR-0023")
    if terminal_failure_criterion != TERMINAL_FAILURE_CRITERION:
        raise ValueError("R03A terminal-failure criterion differs from ADR-0023")
    if evaluated_failure_precedence != EVALUATED_FAILURE_PRECEDENCE:
        raise ValueError("R03A evaluated-failure precedence differs from ADR-0023")
    scale_value = settings.get("normalization_action_scale")
    if not isinstance(scale_value, Sequence) or isinstance(scale_value, (str, bytes)):
        raise ValueError("R03A action scale must contain three values")
    scale = tuple(
        _finite(item, name="normalization action scale", positive=True)
        for item in scale_value
    )
    if len(scale) != 3 or any(
        not math.isclose(item, expected, rel_tol=0.0, abs_tol=1e-12)
        for item, expected in zip(scale, REGISTERED_TRANSLATION_ACTION_SCALE)
    ):
        raise ValueError("R03A action scale differs from the frozen normalization")
    if settings.get("normalization_asset_sha256") != NORMALIZATION_ASSET_SHA256:
        raise ValueError("R03A normalization asset hash differs")
    latency = settings.get("latency_protocol")
    if not isinstance(latency, Mapping):
        raise ValueError("R03A requires a frozen latency_protocol object")
    warmups = int(
        latency.get(
            "warmup_policy_calls_before_measurement", TIMING_WARMUP_REPEATS
        )
    )
    measured = int(
        latency.get(
            "measured_policy_calls_per_arm_per_case", TIMING_MEASURED_REPEATS
        )
    )
    if warmups != TIMING_WARMUP_REPEATS or measured != TIMING_MEASURED_REPEATS:
        raise ValueError("R03A timing repetitions differ from the frozen protocol")
    if not (
        latency.get("mode") == "warmed_batch_one"
        and latency.get("timer_scope")
        == "client_round_trip_batch_one_infer_including_analytic_field"
        and latency.get("timer_unit") == "seconds"
        and latency.get("quantile_method") == "inverted_cdf"
    ):
        raise ValueError("R03A latency boundary differs from the frozen protocol")
    if config_file_sha256 is None:
        config_file_sha256 = value.get("_config_file_sha256")
    if not _is_sha256(config_file_sha256):
        raise ValueError("R03A requires the immutable config-file SHA-256")
    if scientific_config_hash is None:
        from .r03a_validation import r03a_config_hash

        scientific_config_hash = r03a_config_hash(value)
    if not _is_sha256(scientific_config_hash):
        raise ValueError("R03A scientific config hash must be SHA-256")

    return R03AConfig(
        oracle=oracle,
        target_name=target_name,
        p_min_m=p_min,
        simulator_safety_margin_m=safety_margin,
        maximum_target_displacement_m=target_limit,
        maximum_obstacle_displacement_m=obstacle_limit,
        simulator_repeats=repeats,
        translation_action_low=low,
        translation_action_high=high,
        action_scale=scale,  # type: ignore[arg-type]
        samples_per_segment=samples,
        source_r03_summary_path=str(summary_path),
        source_r03_summary_sha256=str(summary_sha),
        source_r03_summary=summary,
        source_r03_ordered_result_set_digest=str(digest),
        source_r02_results_root=str(results_root),
        source_r02_result_hashes={case_id: all_hashes[case_id] for case_id in eligible},
        eligible_case_ids=eligible,
        eligible_manifest_sha256=str(eligible_manifest_sha),
        original_manifest_sha256=str(original_manifest_sha),
        r02_config_sha256=str(r02_config_sha),
        decision_artifact=str(decision),
        decision_sha256=str(decision_sha),
        config_file_sha256=str(config_file_sha256),
        scientific_config_hash=scientific_config_hash,
        timing_warmup_repeats=warmups,
        timing_measured_repeats=measured,
        saturation_criterion=saturation_criterion,
        saturation_comparison=saturation_comparison,
        terminal_failure_criterion=terminal_failure_criterion,
        evaluated_failure_precedence=evaluated_failure_precedence,
    )


def _normalized_config(config: R03AConfig) -> Dict[str, Any]:
    return {
        **scientific_config(config.oracle.__dict__),
        "target_name": config.target_name,
        "minimum_progress_m": config.p_min_m,
        "simulator_safety_margin_m": config.simulator_safety_margin_m,
        "maximum_target_displacement_m": config.maximum_target_displacement_m,
        "maximum_obstacle_displacement_m": config.maximum_obstacle_displacement_m,
        "simulator_repeats": config.simulator_repeats,
        "translation_action_bounds": [
            config.translation_action_low,
            config.translation_action_high,
        ],
        "normalization_action_scale": list(config.action_scale),
        "normalization_asset_sha256": NORMALIZATION_ASSET_SHA256,
        "required_new_arms": list(NEW_ARMS),
        "intervention_steps": dict(INTERVENTION_STEPS),
        "energy_margin_m": ENERGY_MARGIN_M,
        "energy_temperature_m": ENERGY_TEMPERATURE_M,
        "samples_per_segment": SAMPLES_PER_SEGMENT,
        "budget_source": "r02.directions.delta_star_model:first_five_xyz_l2",
        "clipping_policy": "fail_without_clipping",
        "timing_warmup_repeats": config.timing_warmup_repeats,
        "timing_measured_repeats": config.timing_measured_repeats,
        "saturation_criterion": config.saturation_criterion,
        "saturation_comparison": config.saturation_comparison,
        "terminal_failure_criterion": config.terminal_failure_criterion,
        "evaluated_failure_precedence": list(config.evaluated_failure_precedence),
        "source_r03_summary_sha256": config.source_r03_summary_sha256,
        "source_r03_ordered_result_set_digest": (
            config.source_r03_ordered_result_set_digest
        ),
        "source_r02_result_hashes": dict(sorted(config.source_r02_result_hashes.items())),
        "eligible_case_ids": list(config.eligible_case_ids),
        "eligible_manifest_sha256": config.eligible_manifest_sha256,
        "original_manifest_sha256": config.original_manifest_sha256,
        "r02_config_sha256": config.r02_config_sha256,
        "decision_artifact": config.decision_artifact,
        "decision_sha256": config.decision_sha256,
    }


def _load_source_r02(
    case_id: str, config: R03AConfig
) -> Tuple[Path, Mapping[str, Any], str]:
    expected = config.source_r02_result_hashes.get(case_id)
    if expected is None:
        raise R03ASourceError(f"case {case_id!r} is outside the R03 eligible population")
    path = Path(config.source_r02_results_root) / case_id / "r02-paired.json"
    actual = file_sha256(path)
    if actual != expected:
        raise R03ASourceError(
            f"raw R02 hash mismatch for {case_id}: expected {expected}, got {actual}"
        )
    value = load_json(path)
    if not isinstance(value, Mapping):
        raise R03ASourceError(f"raw R02 result for {case_id} is not an object")
    errors = validate_r02_result(value)
    if errors:
        raise R03ASourceError(
            f"raw R02 result for {case_id} is invalid: {'; '.join(errors)}"
        )
    if value.get("case_id") != case_id or value.get("status") != "completed":
        raise R03ASourceError("R03A source is not the eligible completed R02 case")
    outcome = value.get("outcome")
    if not isinstance(outcome, Mapping) or not (
        outcome.get("r01_feasible_conditioned") is True
        and outcome.get("nominal_collision_reproduced") is True
        and outcome.get("direct_witness_reconfirmed") is True
    ):
        raise R03ASourceError("R03A source case is outside the witness-confirmed population")
    return path, value, actual


def _trace_from_record(value: Any, *, name: str) -> Mapping[str, np.ndarray]:
    if not isinstance(value, Mapping) or not isinstance(value.get("leaves"), Mapping):
        raise R03ASourceError(f"{name} is not a trace record")
    leaves = value["leaves"]
    if value.get("sha256") != content_hash(dict(leaves)):
        raise R03ASourceError(f"{name} content hash differs")
    result: Dict[str, np.ndarray] = {}
    for key, record in leaves.items():
        array, errors = _validate_array_record(record, name=f"{name}.{key}")
        if errors or array is None:
            raise R03ASourceError(f"invalid {name}.{key}: {'; '.join(errors)}")
        result[str(key)] = array
    return result


def _source_budget(raw_r02: Mapping[str, Any]) -> Tuple[Dict[str, Any], float]:
    directions = raw_r02.get("directions")
    if not isinstance(directions, Mapping):
        raise R03ASourceError("raw R02 result has no direction bundle")
    arrays = directions.get("arrays")
    if not isinstance(arrays, Mapping):
        raise R03ASourceError("raw R02 result has no direction arrays")
    record = arrays.get("delta_star_model")
    array, errors = _validate_array_record(
        record, name="directions.arrays.delta_star_model", shape=(10, 32)
    )
    if errors or array is None or not isinstance(record, Mapping):
        raise R03ASourceError("invalid source delta_star_model: " + "; ".join(errors))
    mask = np.zeros((10, 32), dtype=np.bool_)
    mask[:5, :3] = True
    if np.any(np.asarray(array)[~mask] != 0.0):
        raise R03ASourceError("source delta_star_model violates the registered translation mask")
    budget = float(np.linalg.norm(np.asarray(array, dtype=np.float64)[:5, :3]))
    if not math.isfinite(budget) or budget <= 0.0:
        raise R03ASourceError("source delta_star_model has no positive finite budget")
    norms = directions.get("l2_norms")
    reported = norms.get("delta_star_model") if isinstance(norms, Mapping) else None
    if not isinstance(reported, (int, float)) or not math.isclose(
        budget, float(reported), rel_tol=0.0, abs_tol=1e-12
    ):
        raise R03ASourceError("source delta_star_model L2 conflicts with its record")
    return (
        {
            "definition": "r02.directions.delta_star_model:first_five_xyz_l2",
            "source_direction_key": "delta_star_model",
            "source_delta_star_model": dict(record),
            "source_array_sha256": record.get("sha256"),
            "first_five_xyz_model_l2": budget,
            "source_reported_full_model_l2": float(reported),
        },
        budget,
    )


def _request(
    client: Any,
    observation: Mapping[str, Any],
    controls: Mapping[str, Any],
) -> Tuple[Mapping[str, Any], float]:
    request = copy.deepcopy(dict(observation))
    request["__crfs__"] = copy.deepcopy(dict(controls))
    start = time.perf_counter_ns()
    try:
        reply = client.infer(request)
    except Exception as error:  # external policy-server/transport boundary
        raise R03APolicyInferenceError(f"{type(error).__name__}: {error}") from error
    elapsed_seconds = (time.perf_counter_ns() - start) / 1_000_000_000.0
    if not isinstance(reply, Mapping) or "actions" not in reply:
        raise RuntimeError("policy reply has no actions")
    trace = reply.get("crfs_trace")
    if not isinstance(trace, Mapping):
        raise RuntimeError("R03A policy reply has no CRFS trace")
    return reply, float(elapsed_seconds)


def _frozen_controls(noise: np.ndarray, *, intervention_step: int) -> Mapping[str, Any]:
    return {
        "noise": np.asarray(noise, dtype=np.float32).copy(),
        "intervention_step": int(intervention_step),
        "intervention_mode": "none",
        "return_trace": True,
    }


def _analytic_controls(
    noise: np.ndarray,
    *,
    intervention_step: int,
    budget: float,
    geometry: Mapping[str, Any],
    config: R03AConfig,
) -> Mapping[str, Any]:
    count = int(geometry.get("num_valid_obbs", -1))
    if not 1 <= count <= MAXIMUM_OBBS:
        raise ValueError("R03A branch geometry has no valid OBBs")
    centers = np.asarray(geometry.get("centers_m"), dtype=np.float64)[:count]
    rotations = np.asarray(geometry.get("rotations_world"), dtype=np.float64)[:count]
    half_sizes = np.asarray(geometry.get("half_sizes_m"), dtype=np.float64)[:count]
    return {
        "noise": np.asarray(noise, dtype=np.float32).copy(),
        "intervention_mode": "analytic_trajectory_field",
        "intervention_step": int(intervention_step),
        "return_trace": True,
        "return_normalized_final": True,
        "model_l2_path_budget": float(budget),
        "branch_eef_center_m": np.asarray(
            geometry.get("branch_eef_center_m"), dtype=np.float64
        ),
        "response_matrix_m_per_action": np.asarray(
            config.oracle.response_matrix_m_per_action, dtype=np.float64
        ),
        "obstacle_centers_m": centers,
        "obstacle_rotations_world": rotations.reshape(count, 3, 3),
        "obstacle_half_sizes_m": half_sizes,
        "eef_radius_m": float(config.oracle.eef_radius_m),
    }


def _science_trace(trace: Mapping[str, Any]) -> Mapping[str, Any]:
    """Exclude runtime telemetry while retaining every deterministic leaf."""

    return {
        str(key): value
        for key, value in trace.items()
        if str(key) != "analytic_gradient_ms"
    }


def _trace_pairing_diagnostics(
    source: Mapping[str, Any], fresh: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Bind the exact trace predicate to the same bounded evidence it reports."""

    source_items = list(source.items())
    fresh_items = list(fresh.items())
    source_keys_are_strings = all(isinstance(key, str) for key, _ in source_items)
    fresh_keys_are_strings = all(isinstance(key, str) for key, _ in fresh_items)
    source_raw_keys = {key for key, _ in source_items}
    fresh_raw_keys = {key for key, _ in fresh_items}
    raw_key_sets_equal = source_raw_keys == fresh_raw_keys
    source_snapshot = {
        key: np.ascontiguousarray(np.asarray(value)).copy()
        for key, value in source_items
        if isinstance(key, str)
    }
    fresh_snapshot = {
        key: np.ascontiguousarray(np.asarray(value)).copy()
        for key, value in fresh_items
        if isinstance(key, str)
    }
    source_keys = set(source_snapshot)
    fresh_keys = set(fresh_snapshot)
    leaves: Dict[str, Any] = {}
    for key in sorted(source_keys | fresh_keys):
        record: Dict[str, Any] = {
            "present_in_source": key in source_snapshot,
            "present_in_fresh": key in fresh_snapshot,
        }
        if key in source_snapshot and key in fresh_snapshot:
            source_array = source_snapshot[key]
            fresh_array = fresh_snapshot[key]
            same_shape = source_array.shape == fresh_array.shape
            same_dtype = source_array.dtype == fresh_array.dtype
            source_supported_dtype = source_array.dtype.kind in "biuf"
            fresh_supported_dtype = fresh_array.dtype.kind in "biuf"
            source_finite = bool(
                source_supported_dtype and np.all(np.isfinite(source_array))
            )
            fresh_finite = bool(
                fresh_supported_dtype and np.all(np.isfinite(fresh_array))
            )
            record.update(
                {
                    "source_dtype": str(source_array.dtype),
                    "fresh_dtype": str(fresh_array.dtype),
                    "source_shape": list(source_array.shape),
                    "fresh_shape": list(fresh_array.shape),
                    "same_dtype": bool(same_dtype),
                    "same_shape": bool(same_shape),
                    "source_supported_dtype": bool(source_supported_dtype),
                    "fresh_supported_dtype": bool(fresh_supported_dtype),
                    "source_finite": source_finite,
                    "fresh_finite": fresh_finite,
                    "array_equal": bool(np.array_equal(source_array, fresh_array)),
                    "native_bytes_equal": bool(
                        same_shape
                        and same_dtype
                        and source_array.tobytes() == fresh_array.tobytes()
                    ),
                    "source_sha256": hashlib.sha256(source_array.tobytes()).hexdigest(),
                    "fresh_sha256": hashlib.sha256(fresh_array.tobytes()).hexdigest(),
                    "maximum_absolute_error": None,
                    "rms_absolute_error": None,
                }
            )
            if (
                same_shape
                and source_array.dtype.kind in "iuf"
                and fresh_array.dtype.kind in "iuf"
                and source_finite
                and fresh_finite
            ):
                difference = np.asarray(fresh_array, dtype=np.float64) - np.asarray(
                    source_array, dtype=np.float64
                )
                if np.all(np.isfinite(difference)):
                    absolute = np.abs(difference)
                    record["maximum_absolute_error"] = (
                        float(np.max(absolute)) if absolute.size else 0.0
                    )
                    record["rms_absolute_error"] = (
                        float(np.sqrt(np.mean(np.square(difference))))
                        if difference.size
                        else 0.0
                    )
        leaves[key] = record
    exact_leaves = bool(
        source_keys_are_strings
        and fresh_keys_are_strings
        and raw_key_sets_equal
        and source_keys == fresh_keys
        and bool(leaves)
        and all(
            record.get("present_in_source") is True
            and record.get("present_in_fresh") is True
            and record.get("same_dtype") is True
            and record.get("same_shape") is True
            and record.get("source_supported_dtype") is True
            and record.get("fresh_supported_dtype") is True
            and record.get("source_finite") is True
            and record.get("fresh_finite") is True
            and record.get("array_equal") is True
            and record.get("native_bytes_equal") is True
            and record.get("source_sha256") == record.get("fresh_sha256")
            for record in leaves.values()
        )
    )
    source_recordable = bool(
        source_keys_are_strings
        and source_snapshot
        and all(
            value.dtype.kind in "biuf" and np.all(np.isfinite(value))
            for value in source_snapshot.values()
        )
    )
    fresh_recordable = bool(
        fresh_keys_are_strings
        and fresh_snapshot
        and all(
            value.dtype.kind in "biuf" and np.all(np.isfinite(value))
            for value in fresh_snapshot.values()
        )
    )
    source_record_sha256 = (
        _trace_record(source_snapshot)["sha256"] if source_recordable else None
    )
    fresh_record_sha256 = (
        _trace_record(fresh_snapshot)["sha256"] if fresh_recordable else None
    )
    canonical_record_equal = bool(
        source_recordable
        and fresh_recordable
        and source_record_sha256 == fresh_record_sha256
    )
    return {
        "source_keys_are_strings": source_keys_are_strings,
        "fresh_keys_are_strings": fresh_keys_are_strings,
        "raw_key_sets_equal": raw_key_sets_equal,
        "source_recordable": source_recordable,
        "fresh_recordable": fresh_recordable,
        "source_keys": sorted(source_keys),
        "fresh_keys": sorted(fresh_keys),
        "missing_from_fresh": sorted(source_keys - fresh_keys),
        "extra_in_fresh": sorted(fresh_keys - source_keys),
        "source_trace_record_sha256": source_record_sha256,
        "fresh_trace_record_sha256": fresh_record_sha256,
        "canonical_record_equal": canonical_record_equal,
        "exact_native_leaf_pairing": bool(exact_leaves and canonical_record_equal),
        "leaves": leaves,
    }


def _quantiles(values: Sequence[float]) -> Mapping[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.all(np.isfinite(array)):
        raise ValueError("timing samples must be a non-empty finite vector")
    ordered = np.sort(array)

    def inverted_cdf(probability: float) -> float:
        index = max(0, min(len(ordered) - 1, math.ceil(probability * len(ordered)) - 1))
        return float(ordered[index])

    return {
        "values": array.tolist(),
        "p50": inverted_cdf(0.50),
        "p95": inverted_cdf(0.95),
    }


def _empty_timing_series() -> Mapping[str, Any]:
    return {"values": [], "p50": None, "p95": None}


def _timed_replies(
    client: Any,
    observation: Mapping[str, Any],
    controls: Mapping[str, Any],
    *,
    warmups: int,
    measured: int,
) -> Tuple[List[Mapping[str, Any]], Mapping[str, Any]]:
    warmup_wall: List[float] = []
    for _ in range(warmups):
        _, elapsed = _request(client, observation, controls)
        warmup_wall.append(elapsed)
    replies: List[Mapping[str, Any]] = []
    wall: List[float] = []
    for _ in range(measured):
        reply, elapsed = _request(client, observation, controls)
        replies.append(reply)
        wall.append(elapsed)
    reference_actions = np.asarray(replies[0]["actions"])
    reference_trace = _science_trace(replies[0]["crfs_trace"])
    for reply in replies[1:]:
        candidate_actions = np.asarray(reply["actions"])
        if (
            candidate_actions.dtype != reference_actions.dtype
            or not np.array_equal(reference_actions, candidate_actions)
            or np.ascontiguousarray(reference_actions).tobytes()
            != np.ascontiguousarray(candidate_actions).tobytes()
        ):
            raise RuntimeError("R03A repeated policy actions are not exact")
        repeat_trace_diagnostic = _trace_pairing_diagnostics(
            reference_trace, _science_trace(reply["crfs_trace"])
        )
        if not repeat_trace_diagnostic["exact_native_leaf_pairing"]:
            raise RuntimeError("R03A repeated deterministic trace leaves are not exact")
    analytic_seconds: Mapping[str, Any] = _empty_timing_series()
    if controls.get("intervention_mode") == "analytic_trajectory_field":
        per_step = []
        for reply in replies:
            gradient_ms = np.asarray(
                reply["crfs_trace"].get("analytic_gradient_ms"), dtype=np.float64
            )
            evaluated = np.asarray(
                reply["crfs_trace"].get("field_evaluated"), dtype=np.bool_
            )
            if gradient_ms.shape != (10,) or evaluated.shape != (10,):
                raise RuntimeError("analytic timing trace has an invalid shape")
            per_step.extend((gradient_ms[evaluated] / 1000.0).tolist())
        analytic_seconds = _quantiles(per_step)
    return replies, {
        "descriptive_only": True,
        "warmed_batch_one": True,
        "timer_scope": "client_round_trip_batch_one_infer_including_analytic_field",
        "quantile_method": "inverted_cdf",
        "warmup_repeats": warmups,
        "measured_repeats": measured,
        "warmup_policy_seconds": warmup_wall,
        "policy_seconds": _quantiles(wall),
        "analytic_gradient_seconds": analytic_seconds,
        "server_policy_timing": [
            _reply_policy_timing(reply) for reply in replies
        ],
    }


def _validate_analytic_trace(
    trace: Mapping[str, Any],
    *,
    reply_actions: np.ndarray,
    intervention_step: int,
    budget: float,
    config: R03AConfig,
    frozen_reference_trace: Mapping[str, Any],
) -> Mapping[str, Any]:
    required_shapes = {
        "step_index": (10,),
        "time": (10,),
        "active": (10,),
        "field_evaluated": (10,),
        "x_t_steps": (10, 10, 32),
        "v_base_steps": (10, 10, 32),
        "predicted_clean_steps": (10, 10, 32),
        "physical_predicted_clean_xyz_steps": (10, 5, 3),
        "predicted_eef_centers_m_steps": (10, 5, 3),
        "hard_min_clearance_m": (10,),
        "energy": (10,),
        "energy_gradient_steps": (10, 10, 32),
        "normalized_energy_gradient_steps": (10, 10, 32),
        "gradient_l2": (10,),
        "margin_satisfied": (10,),
        "gradient_finite": (10,),
        "gradient_valid": (10,),
        "applied": (10,),
        "guidance_velocity_steps": (10, 10, 32),
        "guidance_velocity_l2": (10,),
        "path_increment_l2": (10,),
        "cumulative_integrated_field_l2": (10,),
        "analytic_gradient_ms": (10,),
        "final_normalized": (10, 32),
        "final_normalized_physical": (10, 7),
    }
    expected_dtypes = {
        "step_index": np.dtype(np.int64),
        "time": np.dtype(np.float32),
        "active": np.dtype(np.bool_),
        "field_evaluated": np.dtype(np.bool_),
        "x_t_steps": np.dtype(np.float32),
        "v_base_steps": np.dtype(np.float32),
        "predicted_clean_steps": np.dtype(np.float32),
        "physical_predicted_clean_xyz_steps": np.dtype(np.float32),
        "predicted_eef_centers_m_steps": np.dtype(np.float32),
        "hard_min_clearance_m": np.dtype(np.float32),
        "energy": np.dtype(np.float32),
        "energy_gradient_steps": np.dtype(np.float32),
        "normalized_energy_gradient_steps": np.dtype(np.float32),
        "gradient_l2": np.dtype(np.float32),
        "margin_satisfied": np.dtype(np.bool_),
        "gradient_finite": np.dtype(np.bool_),
        "gradient_valid": np.dtype(np.bool_),
        "applied": np.dtype(np.bool_),
        "guidance_velocity_steps": np.dtype(np.float32),
        "guidance_velocity_l2": np.dtype(np.float32),
        "path_increment_l2": np.dtype(np.float32),
        "cumulative_integrated_field_l2": np.dtype(np.float32),
        "analytic_gradient_ms": np.dtype(np.float32),
        "final_normalized": np.dtype(np.float32),
    }
    arrays: Dict[str, np.ndarray] = {}
    missing = set(required_shapes) - set(trace)
    if missing:
        raise RuntimeError(f"analytic trace missing fields: {sorted(missing)}")
    for key, shape in required_shapes.items():
        array = np.asarray(trace[key])
        if array.shape != shape:
            raise RuntimeError(f"analytic trace {key} must have shape {shape}, got {array.shape}")
        if array.dtype.kind not in "biuf" or not np.all(np.isfinite(array)):
            raise RuntimeError(f"analytic trace {key} must be finite numeric/bool")
        expected_dtype = expected_dtypes.get(key)
        if expected_dtype is not None and array.dtype != expected_dtype:
            raise RuntimeError(
                f"analytic trace {key} must have dtype {expected_dtype}, got {array.dtype}"
            )
        arrays[key] = array
    scalar_fields = {
        "intervention_step",
        "dt",
        "active_horizon",
        "model_l2_path_budget",
        "velocity_gain",
        "safety_margin_m",
        "softplus_tau_m",
        "samples_per_segment",
        "integrated_field_l2",
    }
    if not scalar_fields.issubset(trace):
        raise RuntimeError(
            f"analytic trace missing scalar fields: {sorted(scalar_fields - set(trace))}"
        )
    scalar_dtypes = {
        "intervention_step": np.dtype(np.int64),
        "dt": np.dtype(np.float32),
        "active_horizon": np.dtype(np.float32),
        "model_l2_path_budget": np.dtype(np.float32),
        "velocity_gain": np.dtype(np.float32),
        "safety_margin_m": np.dtype(np.float32),
        "softplus_tau_m": np.dtype(np.float32),
        "samples_per_segment": np.dtype(np.int64),
        "integrated_field_l2": np.dtype(np.float32),
    }
    for key, expected_dtype in scalar_dtypes.items():
        scalar = np.asarray(trace[key])
        if scalar.shape != () or scalar.dtype != expected_dtype:
            raise RuntimeError(
                f"analytic trace scalar {key} must have scalar dtype {expected_dtype}"
            )
    if not np.array_equal(arrays["step_index"], np.arange(10)):
        raise RuntimeError("analytic trace step indices differ from the ten-step sampler")
    expected_active = np.arange(10) >= intervention_step
    if not np.array_equal(arrays["active"].astype(bool), expected_active):
        raise RuntimeError("analytic trace active mask differs from the registered start")
    if int(np.asarray(trace["intervention_step"]).item()) != intervention_step:
        raise RuntimeError("analytic trace intervention_step differs")
    # These are fixed protocol constants serialized by the sampler as native
    # float32 scalars.  Construct their expected values in that same recorded
    # dtype and compare exactly; a binary64 literal plus a fitted allowance is
    # not their identity (notably for the early horizon, float32(0.9)).
    dt_scalar = np.asarray(trace["dt"])
    expected_dt_scalar = np.asarray(np.float32(-0.1))
    if not np.array_equal(dt_scalar, expected_dt_scalar):
        raise RuntimeError("analytic trace reverse-time Euler step differs from -0.1")
    dt = float(dt_scalar.item())
    expected_horizon = (10 - intervention_step) / 10.0
    active_horizon_scalar = np.asarray(trace["active_horizon"])
    expected_horizon_scalar = np.asarray(np.float32(expected_horizon))
    if not np.array_equal(active_horizon_scalar, expected_horizon_scalar):
        raise RuntimeError("analytic trace active horizon differs")
    reported_budget = float(np.asarray(trace["model_l2_path_budget"]).item())
    if not math.isclose(reported_budget, budget, rel_tol=2e-6, abs_tol=2e-6):
        raise RuntimeError("analytic trace path budget differs from R02")
    velocity_gain = float(np.asarray(trace["velocity_gain"]).item())
    if not math.isclose(
        velocity_gain, budget / expected_horizon, rel_tol=2e-6, abs_tol=2e-6
    ):
        raise RuntimeError("analytic trace velocity gain does not integrate to the budget")
    if not np.array_equal(
        np.asarray(trace["safety_margin_m"]),
        np.asarray(np.float32(ENERGY_MARGIN_M)),
    ):
        raise RuntimeError("analytic trace safety margin differs")
    if not np.array_equal(
        np.asarray(trace["softplus_tau_m"]),
        np.asarray(np.float32(ENERGY_TEMPERATURE_M)),
    ):
        raise RuntimeError("analytic trace softplus temperature differs")
    if int(np.asarray(trace["samples_per_segment"]).item()) != SAMPLES_PER_SEGMENT:
        raise RuntimeError("analytic trace trajectory sampling density differs")

    active_mask = arrays["active"].astype(bool)
    if not np.array_equal(arrays["field_evaluated"].astype(bool), active_mask):
        raise RuntimeError("analytic field-evaluated mask differs from the active mask")
    predicted_clean_expected = np.asarray(arrays["x_t_steps"], dtype=np.float32) - (
        np.asarray(arrays["time"], dtype=np.float32)[:, None, None]
        * np.asarray(arrays["v_base_steps"], dtype=np.float32)
    )
    predicted_clean_error = np.abs(
        predicted_clean_expected
        - np.asarray(arrays["predicted_clean_steps"], dtype=np.float32)
    )
    if not np.allclose(
        predicted_clean_expected,
        arrays["predicted_clean_steps"],
        rtol=2e-6,
        atol=2e-6,
    ):
        raise RuntimeError("analytic approximate-clean trace differs from x_t - t * v_base")

    inactive = ~active_mask
    for key in (
        "physical_predicted_clean_xyz_steps",
        "predicted_eef_centers_m_steps",
        "hard_min_clearance_m",
        "energy",
        "analytic_gradient_ms",
    ):
        if np.any(arrays[key][inactive] != 0):
            raise RuntimeError(f"analytic trace {key} must be zero before field activation")
    if np.any(arrays["energy"][active_mask] < 0.0):
        raise RuntimeError("analytic collision energy must be nonnegative")
    if np.any(arrays["analytic_gradient_ms"][active_mask] < 0.0):
        raise RuntimeError("analytic gradient timing must be nonnegative")

    # The full output affine is policy-owned; its offset cancels when every
    # active physical prediction is compared with the first paired field.
    first_model_xyz = arrays["predicted_clean_steps"][intervention_step, :5, :3]
    first_physical_xyz = arrays["physical_predicted_clean_xyz_steps"][intervention_step]
    expected_physical_xyz = first_physical_xyz[None, ...] + (
        arrays["predicted_clean_steps"][active_mask, :5, :3]
        - first_model_xyz[None, ...]
    ) * np.asarray(config.action_scale, dtype=np.float32)[None, None, :]
    if not np.allclose(
        expected_physical_xyz,
        arrays["physical_predicted_clean_xyz_steps"][active_mask],
        rtol=2e-6,
        atol=2e-6,
    ):
        raise RuntimeError("analytic physical approximate-clean trace violates the frozen affine")

    energy_gradient = np.asarray(arrays["energy_gradient_steps"], dtype=np.float32)
    gradient_l2_expected = np.linalg.norm(
        energy_gradient[:, :5, :3].reshape(10, -1), axis=1
    ).astype(np.float32)
    if not np.allclose(
        arrays["gradient_l2"], gradient_l2_expected, rtol=2e-6, atol=2e-6
    ):
        raise RuntimeError("analytic gradient L2 does not match the recorded energy gradient")
    finite_expected = active_mask.copy()
    if not np.array_equal(arrays["gradient_finite"].astype(bool), finite_expected):
        raise RuntimeError("analytic gradient-finite flags differ from the finite active field")
    valid_expected = finite_expected & (gradient_l2_expected > 0.0)
    if not np.array_equal(arrays["gradient_valid"].astype(bool), valid_expected):
        raise RuntimeError("analytic gradient-valid flags differ from finite positive L2")
    margin_expected = active_mask & (
        np.asarray(arrays["hard_min_clearance_m"], dtype=np.float32)
        >= np.float32(ENERGY_MARGIN_M)
    )
    if not np.array_equal(arrays["margin_satisfied"].astype(bool), margin_expected):
        raise RuntimeError("analytic margin-stop flags differ from hard predicted clearance")
    applied_expected = active_mask & ~margin_expected & valid_expected
    if not np.array_equal(arrays["applied"].astype(bool), applied_expected):
        raise RuntimeError("analytic applied flags differ from active unsafe valid gradients")

    normalized_expected = np.zeros_like(energy_gradient)
    if np.any(applied_expected):
        denominator = np.maximum(
            gradient_l2_expected[applied_expected], np.finfo(np.float32).tiny
        )[:, None, None]
        normalized_expected[applied_expected] = (
            energy_gradient[applied_expected] / denominator
        )
    if not np.allclose(
        arrays["normalized_energy_gradient_steps"],
        normalized_expected,
        rtol=2e-6,
        atol=2e-6,
    ):
        raise RuntimeError("analytic normalized gradient does not match the energy gradient")
    normalized_l2 = np.linalg.norm(
        normalized_expected[:, :5, :3].reshape(10, -1), axis=1
    )
    if np.any(applied_expected) and not np.allclose(
        normalized_l2[applied_expected], 1.0, rtol=2e-6, atol=2e-6
    ):
        raise RuntimeError("analytic applied normalized gradient is not unit L2")

    guidance_expected = normalized_expected * np.float32(velocity_gain)
    if not np.allclose(
        arrays["guidance_velocity_steps"],
        guidance_expected,
        rtol=2e-6,
        atol=2e-6,
    ):
        raise RuntimeError("analytic guidance velocity differs from normalized gradient times gain")
    guidance_l2_expected = np.linalg.norm(
        guidance_expected[:, :5, :3].reshape(10, -1), axis=1
    ).astype(np.float32)
    if not np.allclose(
        arrays["guidance_velocity_l2"],
        guidance_l2_expected,
        rtol=2e-6,
        atol=2e-6,
    ):
        raise RuntimeError("analytic guidance L2 differs from the applied guidance tensor")
    path_increment_expected = (
        np.abs(np.float32(dt)) * guidance_l2_expected
    ).astype(np.float32)
    if not np.allclose(
        arrays["path_increment_l2"],
        path_increment_expected,
        rtol=2e-6,
        atol=2e-6,
    ):
        raise RuntimeError("analytic path increment differs from Euler step times guidance L2")
    cumulative_expected = np.cumsum(path_increment_expected, dtype=np.float32)
    if not np.allclose(
        arrays["cumulative_integrated_field_l2"],
        cumulative_expected,
        rtol=2e-6,
        atol=2e-6,
    ):
        raise RuntimeError("analytic cumulative path differs from recomputed guidance increments")

    # Audit the complete sampler recurrence, including every inactive and
    # active step, then bind its terminal state to the physical action reply.
    # This prevents a valid-looking field trace from being disconnected from
    # the action sequence that is actually sent to the simulator.
    expected_times = np.asarray(
        [np.float32(1.0) + np.float32(index) * np.float32(dt) for index in range(10)],
        dtype=np.float32,
    )
    if not np.allclose(
        np.asarray(arrays["time"], dtype=np.float32),
        expected_times,
        rtol=0.0,
        atol=2e-7,
    ):
        raise RuntimeError("analytic trace reverse-time grid differs from ten Euler steps")
    total_velocity = np.asarray(arrays["v_base_steps"], dtype=np.float32) + np.asarray(
        arrays["guidance_velocity_steps"], dtype=np.float32
    )
    reconstructed_next = np.asarray(arrays["x_t_steps"], dtype=np.float32) + np.float32(
        dt
    ) * total_velocity
    recurrence_error = np.abs(
        reconstructed_next[:-1]
        - np.asarray(arrays["x_t_steps"][1:], dtype=np.float32)
    )
    terminal_error = np.abs(
        reconstructed_next[-1]
        - np.asarray(arrays["final_normalized"], dtype=np.float32)
    )
    if not np.allclose(
        reconstructed_next[:-1],
        np.asarray(arrays["x_t_steps"][1:], dtype=np.float32),
        rtol=2e-6,
        atol=2e-6,
    ):
        raise RuntimeError("analytic trace does not satisfy the recorded Euler recurrence")
    if not np.allclose(
        reconstructed_next[-1],
        np.asarray(arrays["final_normalized"], dtype=np.float32),
        rtol=2e-6,
        atol=2e-6,
    ):
        raise RuntimeError("analytic final normalized action differs from the Euler recurrence")
    physical_reply = np.asarray(reply_actions)
    if physical_reply.shape != (10, 7) or not np.all(np.isfinite(physical_reply)):
        raise RuntimeError("analytic policy reply must contain finite (10,7) physical actions")
    physical_final = np.asarray(arrays["final_normalized_physical"])
    if physical_final.dtype != physical_reply.dtype:
        raise RuntimeError(
            "analytic physical terminal trace dtype differs from the native policy reply"
        )
    physical_reply_error = np.abs(physical_final - physical_reply)
    if not np.array_equal(physical_final, physical_reply) or (
        np.ascontiguousarray(physical_final).tobytes()
        != np.ascontiguousarray(physical_reply).tobytes()
    ):
        raise RuntimeError(
            "analytic final normalized trace does not exactly reproduce the physical action reply"
        )

    mask = np.zeros((10, 32), dtype=np.bool_)
    mask[:5, :3] = True
    for key in (
        "energy_gradient_steps",
        "normalized_energy_gradient_steps",
        "guidance_velocity_steps",
    ):
        if np.any(arrays[key][:, ~mask] != 0.0):
            raise RuntimeError(f"analytic trace {key} changed outside first-five XYZ")
    if np.any(arrays["guidance_velocity_steps"][~expected_active] != 0.0):
        raise RuntimeError("analytic guidance was applied before its registered start")
    increments = np.asarray(arrays["path_increment_l2"], dtype=np.float64)
    cumulative = np.asarray(
        arrays["cumulative_integrated_field_l2"], dtype=np.float64
    )
    if np.any(increments < -1e-12) or not np.allclose(
        cumulative, np.cumsum(increments), rtol=2e-6, atol=2e-6
    ):
        raise RuntimeError("analytic integrated path accounting is inconsistent")
    integrated = float(np.asarray(trace["integrated_field_l2"]).item())
    if not math.isclose(integrated, float(cumulative[-1]), rel_tol=2e-6, abs_tol=2e-6):
        raise RuntimeError("analytic final integrated path total is inconsistent")
    budget_exceeded = bool(
        integrated > budget
        and not math.isclose(integrated, budget, rel_tol=2e-6, abs_tol=2e-6)
    )

    source_step = int(np.asarray(frozen_reference_trace["step_index"]).reshape(-1)[0])
    if source_step != intervention_step:
        raise RuntimeError("frozen reference trace is from the wrong intervention step")
    comparisons = {
        "x_t": "x_t_steps",
        "v_base": "v_base_steps",
        "predicted_clean": "predicted_clean_steps",
    }
    pre_exact = all(
        np.array_equal(
            np.asarray(frozen_reference_trace[source_key]),
            arrays[analytic_key][intervention_step],
        )
        for source_key, analytic_key in comparisons.items()
    )
    physical_reference = np.asarray(
        frozen_reference_trace["predicted_clean_physical"]
    )[:5, :3]
    physical_candidate = arrays["physical_predicted_clean_xyz_steps"][
        intervention_step
    ]
    physical_max_abs_error = float(
        np.max(np.abs(physical_reference - physical_candidate))
    )
    pre_exact = bool(
        pre_exact
        and np.allclose(
            physical_reference,
            physical_candidate,
            rtol=0.0,
            atol=1.0e-6,
        )
    )
    if not pre_exact:
        raise RuntimeError("analytic arm diverged before computing its first field")

    active = arrays["active"].astype(bool)
    evaluated = arrays["field_evaluated"].astype(bool)
    applied = arrays["applied"].astype(bool)
    margin = arrays["margin_satisfied"].astype(bool)
    valid = arrays["gradient_valid"].astype(bool)
    gradient_ms = np.asarray(arrays["analytic_gradient_ms"], dtype=np.float64)
    evaluated_gradient_ms = gradient_ms[evaluated]
    if not len(evaluated_gradient_ms):
        raise RuntimeError("analytic arm evaluated no collision field")
    invalid_active = active & ~valid & ~margin
    d_opt_trace = [
        {
            "step_index": index,
            "active": bool(arrays["active"][index]),
            "field_evaluated": bool(arrays["field_evaluated"][index]),
            "hard_min_clearance_m": float(arrays["hard_min_clearance_m"][index]),
            "energy": float(arrays["energy"][index]),
            "margin_satisfied": bool(arrays["margin_satisfied"][index]),
            "gradient_valid": bool(arrays["gradient_valid"][index]),
            "applied": bool(arrays["applied"][index]),
        }
        for index in range(10)
    ]
    zero_gradient_count = int(
        np.count_nonzero(active & ~valid & arrays["gradient_finite"].astype(bool) & ~margin)
    )
    nonfinite_gradient_count = int(
        np.count_nonzero(active & ~arrays["gradient_finite"].astype(bool))
    )
    if nonfinite_gradient_count:
        raise RuntimeError(
            "analytic server returned an auditable reply after flagging a nonfinite gradient"
        )
    return {
        "preintervention_trace_exact_to_frozen": pre_exact,
        "preintervention_physical_max_abs_error": physical_max_abs_error,
        "predicted_clean_max_abs_error": float(np.max(predicted_clean_error)),
        "euler_recurrence_max_abs_error": float(np.max(recurrence_error)),
        "euler_terminal_max_abs_error": float(np.max(terminal_error)),
        "physical_reply_max_abs_error": float(np.max(physical_reply_error)),
        "physical_reply_exact": True,
        "budget_model_l2": budget,
        "integrated_field_model_l2": integrated,
        "budget_exceeded_materially": budget_exceeded,
        "active_step_count": int(np.count_nonzero(active)),
        "applied_step_count": int(np.count_nonzero(applied)),
        "margin_stop_step_count": int(np.count_nonzero(active & margin)),
        "zero_gradient_step_count": zero_gradient_count,
        "nonfinite_gradient_step_count": nonfinite_gradient_count,
        "d_opt_trace": d_opt_trace,
        "active_steps": int(np.count_nonzero(active)),
        "field_evaluated_steps": int(np.count_nonzero(evaluated)),
        "applied_steps": int(np.count_nonzero(applied)),
        "margin_satisfied_steps": int(np.count_nonzero(active & margin)),
        "zero_or_invalid_gradient_steps": int(np.count_nonzero(invalid_active)),
        "all_active_gradients_finite": bool(
            np.all(arrays["gradient_finite"].astype(bool)[active])
        ),
        "minimum_predicted_d_opt_clearance_m": float(
            np.min(np.asarray(arrays["hard_min_clearance_m"], dtype=np.float64)[evaluated])
        ),
        "final_predicted_d_opt_clearance_m": float(
            np.asarray(arrays["hard_min_clearance_m"], dtype=np.float64)[-1]
        ),
        "integrated_field_l2": integrated,
        "path_budget_model_l2": budget,
        "budget_fraction_used": integrated / budget,
        "analytic_gradient_ms": _quantiles(evaluated_gradient_ms.tolist()),
        "d_opt_definition": (
            "frozen H04 response; 26 samples/segment; branch OBBs; 6cm EEF sphere"
        ),
        "d_sim_definition": (
            "branch-inclusive hidden-substep simulator EEF-sphere/active-OBB clearance"
        ),
    }


def _realized_correction(
    frozen_actions: np.ndarray,
    candidate_actions: np.ndarray,
    config: R03AConfig,
) -> Mapping[str, Any]:
    frozen = np.asarray(frozen_actions, dtype=np.float64)
    candidate = np.asarray(candidate_actions, dtype=np.float64)
    if frozen.shape != candidate.shape or frozen.shape[0] < 5 or frozen.shape[1] < 3:
        raise ValueError("realized correction requires paired physical actions")
    physical = candidate[:5, :3] - frozen[:5, :3]
    model = physical / np.asarray(config.action_scale, dtype=np.float64)[None, :]
    return {
        "physical_l2": float(np.linalg.norm(physical)),
        "physical_rms": float(np.sqrt(np.mean(np.square(physical)))),
        "model_l2": float(np.linalg.norm(model)),
        "model_rms": float(np.sqrt(np.mean(np.square(model)))),
    }


def _not_evaluated_arm(name: str, reason: str) -> Mapping[str, Any]:
    return {
        "mechanism": name,
        "applicable": True,
        "status": "not_evaluated_after_nominal_collision_not_reconfirmed",
        "controls": {
            "intervention_mode": "analytic_trajectory_field",
            "intervention_step": INTERVENTION_STEPS[name],
            "clipping_policy": "fail_without_clipping",
            "policy_timing": _policy_timing_control(
                policy_inference=False, reason=reason
            ),
        },
        "correction": None,
        "full_actions": None,
        "executed_actions": None,
        "bounds": {
            "checked": False,
            "passed": None,
            "violations": [],
            "clipped": False,
        },
        "trace_sha256": None,
        "preintervention_trace_exact_to_frozen": None,
        "policy_replay_exact": None,
        "policy_duplicate_actions": None,
        "policy_duplicate_trace": None,
        "replay_exact": None,
        "repeats": [],
        "gate": {"passed": None, "trial_checks": [], "reason": reason},
        "trace": None,
        "timing": {
            "descriptive_only": True,
            "warmed_batch_one": False,
            "timer_scope": "client_round_trip_batch_one_infer_including_analytic_field",
            "quantile_method": "inverted_cdf",
            "policy_seconds": _empty_timing_series(),
            "analytic_gradient_seconds": _empty_timing_series(),
            "reason": reason,
        },
        "diagnostics": None,
    }


def _analytic_failure_status(error: Exception) -> str:
    message = f"{type(error).__name__}: {error}".lower()
    nonfinite_markers = (
        "nonfinite",
        "non-finite",
        "not finite",
        "must be finite",
        "contains nan",
        "contains inf",
    )
    return (
        "nonfinite_failure"
        if any(marker in message for marker in nonfinite_markers)
        else "policy_failure"
    )


def _unrun_analytic_failure_arm(
    name: str,
    *,
    status: str,
    reason: str,
    budget: float,
) -> Mapping[str, Any]:
    """Retain a fixed-denominator arm when inference yields no auditable reply."""

    if status not in {"policy_failure", "nonfinite_failure"}:
        raise ValueError("unrun analytic failure must be policy_failure or nonfinite_failure")
    if not reason:
        raise ValueError("unrun analytic failure requires a reason")
    return {
        "mechanism": (
            "recomputed approximate-clean signed-softplus trajectory collision field"
        ),
        "applicable": True,
        "status": status,
        "failure_reason": reason,
        "controls": {
            "intervention_mode": "analytic_trajectory_field",
            "intervention_step": INTERVENTION_STEPS[name],
            "model_l2_path_budget": float(budget),
            "energy_margin_m": ENERGY_MARGIN_M,
            "energy_temperature_m": ENERGY_TEMPERATURE_M,
            "samples_per_segment": SAMPLES_PER_SEGMENT,
            "clipping_policy": "fail_without_clipping",
            "policy_timing": _policy_timing_control(
                policy_inference=False, reason=reason
            ),
        },
        "correction": None,
        "full_actions": None,
        "executed_actions": None,
        "bounds": {
            "checked": False,
            "passed": None,
            "violations": [],
            "clipped": False,
        },
        "trace_sha256": None,
        "preintervention_trace_exact_to_frozen": None,
        "policy_replay_exact": None,
        "policy_duplicate_actions": None,
        "policy_duplicate_trace": None,
        "replay_exact": None,
        "repeats": [],
        "gate": {"passed": False, "trial_checks": [], "reason": reason},
        "trace": None,
        "timing": {
            "descriptive_only": True,
            "warmed_batch_one": False,
            "timer_scope": "client_round_trip_batch_one_infer_including_analytic_field",
            "quantile_method": "inverted_cdf",
            "policy_seconds": _empty_timing_series(),
            "analytic_gradient_seconds": _empty_timing_series(),
            "reason": reason,
        },
        "diagnostics": None,
    }


def _pre_rollout_analytic_failure_arm(
    name: str,
    *,
    status: str,
    reason: str,
    actions: np.ndarray,
    trace: Mapping[str, Any],
    duplicate_actions: np.ndarray,
    duplicate_trace: Mapping[str, Any],
    timing: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    budget: float,
    config: R03AConfig,
) -> Mapping[str, Any]:
    """Record a fully audited policy reply that is invalid before simulation."""

    if status not in {"saturation_failure", "budget_failure"}:
        raise ValueError("pre-rollout analytic failure status is invalid")
    full_actions = np.asarray(actions, dtype=np.float64)
    duplicate = np.asarray(duplicate_actions, dtype=np.float64)
    bounds = _bounds_check(full_actions, config)
    if not bounds["passed"]:
        raise ValueError("saturation/budget failure helper received out-of-bound actions")
    return {
        "mechanism": (
            "recomputed approximate-clean signed-softplus trajectory collision field"
        ),
        "applicable": True,
        "status": status,
        "controls": {
            "intervention_mode": "analytic_trajectory_field",
            "intervention_step": INTERVENTION_STEPS[name],
            "model_l2_path_budget": float(budget),
            "energy_margin_m": ENERGY_MARGIN_M,
            "energy_temperature_m": ENERGY_TEMPERATURE_M,
            "samples_per_segment": SAMPLES_PER_SEGMENT,
            "clipping_policy": "fail_without_clipping",
            "policy_timing": timing,
        },
        "correction": None,
        "full_actions": _array_record(full_actions, dtype=np.float64),
        "executed_actions": _array_record(full_actions[:5, :7], dtype=np.float64),
        "bounds": bounds,
        "trace_sha256": _trace_record(trace)["sha256"],
        "preintervention_trace_exact_to_frozen": diagnostics[
            "preintervention_trace_exact_to_frozen"
        ],
        "policy_replay_exact": True,
        "policy_duplicate_actions": _array_record(duplicate, dtype=np.float64),
        "policy_duplicate_trace": _trace_record(duplicate_trace),
        "replay_exact": None,
        "repeats": [],
        "gate": {"passed": False, "trial_checks": [], "reason": reason},
        "trace": _trace_record(trace),
        "timing": timing,
        "diagnostics": dict(diagnostics),
    }


def _exact_translation_saturation(actions: np.ndarray, config: R03AConfig) -> bool:
    value = np.asarray(actions, dtype=np.float64)
    if value.ndim != 2 or value.shape[0] < 5 or value.shape[1] < 3:
        return False
    xyz = value[:5, :3]
    return bool(
        np.any(xyz == config.translation_action_low)
        or np.any(xyz == config.translation_action_high)
    )


def _pairing_binding(
    source: Any,
    fresh: Any,
    *,
    source_sha256: Optional[str] = None,
    fresh_sha256: Optional[str] = None,
) -> Mapping[str, Any]:
    """Retain both records and bind their exact canonical JSON identities."""

    source_sha = source_sha256 or content_hash(source)
    fresh_sha = fresh_sha256 or content_hash(fresh)
    if not _is_sha256(source_sha) or not _is_sha256(fresh_sha):
        raise ValueError("pairing bindings require valid SHA-256 identities")
    return {
        "source": source,
        "fresh": fresh,
        "source_sha256": source_sha,
        "fresh_sha256": fresh_sha,
    }


def _provenance(
    case: Mapping[str, Any],
    config: R03AConfig,
    *,
    root: Path,
    input_manifest_sha256: str,
    raw_path: Path,
    raw_sha256: str,
    noise: np.ndarray,
) -> Mapping[str, Any]:
    git_commit, git_dirty = _git_state(root)
    return {
        "evidence_tier": "real_safelibero_r03a_strong_analytic_development_diagnostic",
        "training": False,
        "confirmatory_test": False,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "baseline_commit": BASELINE_COMMIT,
        "python_version": platform.python_version(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "device": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID")
        or os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID") or "0",
        "partition": os.environ.get("SLURM_JOB_PARTITION") or "unknown",
        "case_record": dict(case),
        "case_record_sha256": content_hash(dict(case)),
        "input_manifest_sha256": input_manifest_sha256,
        "config_file_sha256": config.config_file_sha256,
        "checkpoint_id": config.oracle.checkpoint_id,
        "checkpoint_sha256": config.oracle.checkpoint_sha256,
        "source_r03_summary_path": config.source_r03_summary_path,
        "source_r03_summary_sha256": config.source_r03_summary_sha256,
        "source_r02_case_path": str(raw_path),
        "source_r02_case_sha256": raw_sha256,
        "noise": _array_record(noise, dtype=np.float32),
        "sampler_steps": config.oracle.sampler_steps,
        "model_action_horizon": config.oracle.action_horizon,
        "executed_action_horizon": config.oracle.executed_prefix,
        "model_action_dimension": config.oracle.action_dim,
        "action_frame": "world-frame OSC translation delta",
        "policy_action_space": "unnormalized LIBERO controller actions",
        "normalization_space": (
            "absolute approximate-clean uses full inverse affine; displacement uses scale only"
        ),
        "normalization_asset_sha256": NORMALIZATION_ASSET_SHA256,
        "normalization_action_scale": list(config.action_scale),
        "translation_action_bounds": [
            config.translation_action_low,
            config.translation_action_high,
        ],
        "clipping_policy": "fail_without_clipping",
        "d_opt_model": (
            "differentiable signed softplus margin field on frozen H04 trajectory proxy"
        ),
        "d_sim_model": (
            "physics-substep 6cm EEF-sphere/active-OBB clearance plus contact"
        ),
        "eef_radius_m": config.oracle.eef_radius_m,
        "distance_limit_m": config.oracle.distance_limit_m,
        "response_matrix_m_per_action": [
            list(row) for row in config.oracle.response_matrix_m_per_action or ()
        ],
        "energy_margin_m": ENERGY_MARGIN_M,
        "energy_temperature_m": ENERGY_TEMPERATURE_M,
        "samples_per_segment": SAMPLES_PER_SEGMENT,
        "simulator_safety_margin_m": config.simulator_safety_margin_m,
        "minimum_progress_m": config.p_min_m,
        "maximum_target_displacement_m": config.maximum_target_displacement_m,
        "maximum_obstacle_displacement_m": config.maximum_obstacle_displacement_m,
        "simulator_repeats": config.simulator_repeats,
        "measurement_samples_per_trial": EXPECTED_MEASUREMENT_SAMPLES,
        "required_arms": list(ARMS),
        "timing_protocol": {
            "warmup_repeats": config.timing_warmup_repeats,
            "measured_repeats": config.timing_measured_repeats,
            "batch_size": 1,
        },
    }


def _valid_completion(
    path: Path, *, case_id: str, run_id: str, config_hash: str
) -> bool:
    try:
        value = load_json(path)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return False
    if not isinstance(value, Mapping):
        return False
    try:
        from .r03a_validation import validate_r03a_result, validate_r03a_schema
    except ImportError:
        return False
    return bool(
        not validate_r03a_result(value)
        and not validate_r03a_schema(value, require_jsonschema=True)
        and value.get("case_id") == case_id
        and value.get("run_id") == run_id
        and value.get("config_hash") == config_hash
    )


def _validate_before_write(result: Mapping[str, Any]) -> None:
    from .r03a_validation import validate_r03a_result, validate_r03a_schema

    errors = validate_r03a_result(result) + validate_r03a_schema(
        result, require_jsonschema=True
    )
    if errors:
        raise RuntimeError("refusing invalid R03A result: " + "; ".join(errors))


def run_r03a_case(
    case: Mapping[str, Any],
    config: R03AConfig,
    *,
    repo_root: str | Path,
    input_manifest_sha256: str,
    client: Any = None,
    environment: Optional[SafeLiberoCase] = None,
) -> Tuple[Path, str]:
    """Run one paired R03A case inside a GPU-backed Slurm allocation."""

    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("R03A real cases must execute inside a Slurm allocation")
    visible_device = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible_device or visible_device == "NoDevFiles":
        raise RuntimeError("R03A real cases require an allocation-visible GPU")
    root = Path(repo_root).resolve()
    case_id = str(case.get("case_id", ""))
    if case_id not in config.eligible_case_ids:
        raise ValueError("R03A case is outside the frozen 17-case eligible population")
    if input_manifest_sha256 != config.eligible_manifest_sha256:
        raise R03ASourceError("R03A input manifest differs from its frozen eligible subset")
    config_hash = config.scientific_config_hash
    output = (
        Path(config.oracle.output_root)
        / config.oracle.run_id
        / case_id
        / "r03a-analytic-kill-test.json"
    )
    if _valid_completion(
        output, case_id=case_id, run_id=config.oracle.run_id, config_hash=config_hash
    ):
        return output, "skipped_valid_completion"

    raw_path, raw_r02, raw_sha = _load_source_r02(case_id, config)
    raw_case = raw_r02.get("provenance", {}).get("case_record")
    if raw_case != dict(case):
        raise R03ASourceError("R03A manifest record differs from the immutable R02 case")
    budget_record, budget = _source_budget(raw_r02)
    source_noise_record = raw_r02.get("provenance", {}).get("noise")
    source_noise, noise_errors = _validate_array_record(
        source_noise_record, name="source R02 noise", shape=(10, 32)
    )
    if noise_errors or source_noise is None:
        raise R03ASourceError("source R02 noise is invalid: " + "; ".join(noise_errors))
    noise = np.random.default_rng(int(case["policy_seed"])).normal(
        size=(config.oracle.action_horizon, config.oracle.action_dim)
    ).astype(np.float32)
    noise_exact = bool(
        np.array_equal(noise, np.asarray(source_noise, dtype=np.float32))
    )
    if not noise_exact:
        raise R03ASourceError("R03A reconstructed noise differs from immutable R02")

    provenance = _provenance(
        case,
        config,
        root=root,
        input_manifest_sha256=input_manifest_sha256,
        raw_path=raw_path,
        raw_sha256=raw_sha,
        noise=noise,
    )
    if client is None:
        from openpi_client import websocket_client_policy

        client = websocket_client_policy.WebsocketClientPolicy(
            config.oracle.host, config.oracle.port
        )
    owns_environment = environment is None
    if environment is None:
        environment = SafeLiberoCase(dict(case), config.oracle)
    else:
        environment.configure_case(dict(case))

    try:
        initial_observation = environment.reset_and_settle()
        if environment.obstacle_name is None:
            raise RuntimeError("failed to resolve active obstacle")
        initial = capture_reach_snapshot(
            environment, config.target_name, environment.obstacle_name
        )
        if _target_contact_at_branch(environment, config.target_name) or environment.env.check_success():
            raise RuntimeError("R03A branch is no longer a valid pregrasp reach state")
        policy_input = policy_observation(
            initial_observation, environment.prompt, config.oracle.resize_size
        )
        observation_fingerprint = _observation_fingerprint(policy_input)
        branch_snapshot = _json_compatible(initial.to_dict())
        geometry = _capture_branch_geometry(
            environment, maximum_obbs=MAXIMUM_OBBS
        )

        source_pairing = raw_r02.get("pairing")
        if not isinstance(source_pairing, Mapping):
            raise R03ASourceError("raw R02 result has no pairing record")
        source_frozen = raw_r02.get("arms", {}).get("frozen")
        if not isinstance(source_frozen, Mapping):
            raise R03ASourceError("raw R02 result has no frozen arm")
        source_frozen_actions_record = source_pairing.get("eager_actions")
        source_frozen_actions, action_errors = _validate_array_record(
            source_frozen_actions_record,
            name="source frozen actions",
            shape=(10, 7),
        )
        if action_errors or source_frozen_actions is None:
            raise R03ASourceError("invalid source frozen actions: " + "; ".join(action_errors))
        source_mid_trace_record = source_pairing.get("eager_trace")
        source_mid_trace = _trace_from_record(
            source_mid_trace_record, name="source frozen midpoint trace"
        )
        source_rollouts = source_frozen.get("repeats")
        if not isinstance(source_rollouts, list) or not source_rollouts:
            raise R03ASourceError("source R02 frozen arm has no simulator repeats")
        source_rollout = source_rollouts[0]
        if not isinstance(source_rollout, Mapping):
            raise R03ASourceError("source R02 frozen rollout is invalid")
        from .r04_labels import _rollout_branch_geometry

        source_geometry = _rollout_branch_geometry(
            source_rollout, maximum_obbs=MAXIMUM_OBBS
        )
        checks: Dict[str, bool] = {
            "case_record_exact_to_source": raw_case == dict(case),
            "observation_exact_to_source": (
                observation_fingerprint == source_pairing.get("policy_observation")
            ),
            "branch_snapshot_exact_to_source": (
                branch_snapshot == source_pairing.get("branch_snapshot")
            ),
            "geometry_exact_to_source": geometry == source_geometry,
            "noise_exact_to_source": noise_exact,
            "fresh_frozen_actions_exact_to_source": False,
            "fresh_frozen_trace_exact_to_source": False,
        }
        if not all(list(checks.values())[:5]):
            raise R03ASourceError(f"R03A fresh branch pairing failed: {checks}")

        frozen_replies, frozen_timing = _timed_replies(
            client,
            policy_input,
            _frozen_controls(noise, intervention_step=5),
            warmups=config.timing_warmup_repeats,
            measured=config.timing_measured_repeats,
        )
        frozen_reply = frozen_replies[0]
        frozen_actions = np.asarray(frozen_reply["actions"], dtype=np.float64)
        frozen_mid_trace = frozen_reply["crfs_trace"]
        checks["fresh_frozen_actions_exact_to_source"] = bool(
            np.array_equal(frozen_actions, np.asarray(source_frozen_actions))
        )
        trace_diagnostics = _trace_pairing_diagnostics(
            source_mid_trace, frozen_mid_trace
        )
        checks["fresh_frozen_trace_exact_to_source"] = bool(
            trace_diagnostics["exact_native_leaf_pairing"]
        )
        if not all(checks.values()):
            raise R03ASourceError(
                "R03A policy pairing failed: "
                f"{checks}; trace_diagnostics="
                + json.dumps(trace_diagnostics, sort_keys=True, separators=(",", ":"))
            )

        early_reference_reply, _ = _request(
            client,
            policy_input,
            _frozen_controls(noise, intervention_step=1),
        )
        early_reference_actions = np.asarray(
            early_reference_reply["actions"], dtype=np.float64
        )
        if not np.array_equal(early_reference_actions, frozen_actions):
            raise RuntimeError("frozen action changed when requesting the early trace")
        early_reference_trace = early_reference_reply["crfs_trace"]

        frozen_arm = _evaluated_arm(
            "frozen",
            frozen_actions,
            environment,
            initial,
            config,  # R03AConfig intentionally satisfies the R02 rollout interface.
            mechanism="fresh exact R02-paired frozen eager sampler",
            controls={
                "intervention_mode": "none",
                "intervention_step": 5,
                "policy_timing": frozen_timing,
            },
            trace=frozen_mid_trace,
            policy_replay_exact=True,
            duplicate_actions=np.asarray(frozen_replies[1]["actions"], dtype=np.float64),
            duplicate_trace=frozen_replies[1]["crfs_trace"],
        )
        frozen_arm["trace"] = _trace_record(frozen_mid_trace)
        frozen_arm["timing"] = frozen_timing
        frozen_arm["diagnostics"] = {
            "source_r02_actions_exact": True,
            "source_r02_trace_exact": True,
            "early_reference_trace": _trace_record(early_reference_trace),
        }
        if frozen_arm.get("replay_exact") is not True:
            raise RuntimeError("R03A frozen simulator repeats are not exact")
        frozen_rollout = frozen_arm["repeats"][0]
        nominal_collision = bool(
            float(frozen_rollout["clearance_m"]) < 0.0
            or bool(frozen_rollout["contact"])
        )

        source_outcome = raw_r02.get("outcome", {})
        source_pass_map = dict(source_outcome.get("arm_gate_pass", {}))
        source_evidence = {
            "r03_summary_path": config.source_r03_summary_path,
            "r03_summary_sha256": config.source_r03_summary_sha256,
            "r03_ordered_result_set_digest": config.source_r03_ordered_result_set_digest,
            "r02_case_path": str(raw_path),
            "r02_case_sha256": raw_sha,
            "r02_case_validator": "validate_r02_result:passed",
            "r02_config_sha256": config.r02_config_sha256,
            "decision_artifact": config.decision_artifact,
            "decision_sha256": config.decision_sha256,
            "eligible_manifest_sha256": config.eligible_manifest_sha256,
        }
        fresh_noise_record = _array_record(noise, dtype=np.float32)
        fresh_frozen_actions_record = _array_record(
            frozen_actions, dtype=np.float64
        )
        fresh_frozen_trace_record = _trace_record(frozen_mid_trace)
        pairing = {
            "passed": True,
            "policy_observation": _pairing_binding(
                source_pairing.get("policy_observation"),
                observation_fingerprint,
                source_sha256=str(
                    source_pairing.get("policy_observation", {}).get("sha256")
                ),
                fresh_sha256=str(observation_fingerprint["sha256"]),
            ),
            "branch_snapshot": _pairing_binding(
                source_pairing.get("branch_snapshot"), branch_snapshot
            ),
            "branch_geometry": _pairing_binding(
                # Both records already passed the exact normalized geometry
                # comparison above.  Bind those canonical, geom-name-sorted
                # representations so XML/contact enumeration order (notably
                # g7 versus g10) cannot create a false source mismatch.
                source_geometry,
                geometry,
            ),
            "noise": _pairing_binding(
                dict(source_noise_record),
                fresh_noise_record,
                source_sha256=str(source_noise_record.get("sha256")),
                fresh_sha256=str(fresh_noise_record["sha256"]),
            ),
            "source_frozen_actions": _pairing_binding(
                dict(source_frozen_actions_record),
                fresh_frozen_actions_record,
                source_sha256=str(source_frozen_actions_record.get("sha256")),
                fresh_sha256=str(fresh_frozen_actions_record["sha256"]),
            ),
            "source_frozen_trace": _pairing_binding(
                dict(source_mid_trace_record),
                fresh_frozen_trace_record,
                source_sha256=str(source_mid_trace_record.get("sha256")),
                fresh_sha256=str(fresh_frozen_trace_record["sha256"]),
            ),
            "checks": checks,
        }

        if not nominal_collision:
            reason = "fresh frozen branch no longer reproduces the registered collision"
            arms = {
                "frozen": frozen_arm,
                "analytic_trajectory_mid": _not_evaluated_arm(
                    "analytic_trajectory_mid", reason
                ),
                "analytic_trajectory_early": _not_evaluated_arm(
                    "analytic_trajectory_early", reason
                ),
            }
            result = {
                "schema_version": SCHEMA_VERSION,
                "artifact_type": ARTIFACT_TYPE,
                "gate": GATE,
                "case_id": case_id,
                "run_id": config.oracle.run_id,
                "status": "nominal_collision_not_reconfirmed",
                "config_hash": config_hash,
                "source_evidence": source_evidence,
                "provenance": provenance,
                "pairing": pairing,
                "budget": budget_record,
                "arms": arms,
                "outcome": {
                    "population": "r03_witness_confirmed_development_diagnostic",
                    "source_arm_gate_pass": source_pass_map,
                    "fresh_nominal_collision_reproduced": False,
                    "arm_gate_pass": {
                        "frozen": bool(frozen_arm["gate"]["passed"]),
                        "analytic_trajectory_mid": None,
                        "analytic_trajectory_early": None,
                    },
                    "arm_status": {name: arms[name]["status"] for name in ARMS},
                    "privileged_reference_sps_count": 9,
                    "privileged_reference_population_size": 17,
                    "kill_threshold_sps_count": 9,
                    "population_decision": "deferred_to_population_summary",
                },
            }
            _validate_before_write(result)
            atomic_write_json(output, result)
            return output, "nominal_collision_not_reconfirmed"

        reference_traces = {
            5: frozen_mid_trace,
            1: early_reference_trace,
        }
        arms: Dict[str, Any] = {"frozen": frozen_arm}
        for name in NEW_ARMS:
            step = INTERVENTION_STEPS[name]
            controls = _analytic_controls(
                noise,
                intervention_step=step,
                budget=budget,
                geometry=geometry,
                config=config,
            )
            try:
                replies, timing = _timed_replies(
                    client,
                    policy_input,
                    controls,
                    warmups=config.timing_warmup_repeats,
                    measured=config.timing_measured_repeats,
                )
            except R03APolicyInferenceError as error:
                reason = f"{type(error).__name__}: {error}"
                arms[name] = _unrun_analytic_failure_arm(
                    name,
                    status=_analytic_failure_status(error),
                    reason=reason,
                    budget=budget,
                )
                continue
            # Everything after a reply is an apparatus invariant.  A malformed
            # trace, failed deterministic replay, disconnected Euler path, or
            # wrong physical action must abort the case rather than masquerade
            # as evidence that the analytic method failed scientifically.
            reply = replies[0]
            native_actions = np.asarray(reply["actions"])
            trace = reply["crfs_trace"]
            diagnostics = dict(
                _validate_analytic_trace(
                    trace,
                    reply_actions=native_actions,
                    intervention_step=step,
                    budget=budget,
                    config=config,
                    frozen_reference_trace=reference_traces[step],
                )
            )
            actions = np.asarray(native_actions, dtype=np.float64)
            diagnostics["realized_final_correction"] = _realized_correction(
                frozen_actions, actions, config
            )
            action_bounds = _bounds_check(actions, config)
            if action_bounds["passed"] and _exact_translation_saturation(actions, config):
                arms[name] = _pre_rollout_analytic_failure_arm(
                    name,
                    status="saturation_failure",
                    reason="executed first-five XYZ exactly equals a registered inclusive bound",
                    actions=actions,
                    trace=trace,
                    duplicate_actions=np.asarray(
                        replies[1]["actions"], dtype=np.float64
                    ),
                    duplicate_trace=replies[1]["crfs_trace"],
                    timing=timing,
                    diagnostics=diagnostics,
                    budget=budget,
                    config=config,
                )
                continue
            if action_bounds["passed"] and diagnostics["budget_exceeded_materially"]:
                arms[name] = _pre_rollout_analytic_failure_arm(
                    name,
                    status="budget_failure",
                    reason="integrated analytic field materially exceeds immutable R02 budget",
                    actions=actions,
                    trace=trace,
                    duplicate_actions=np.asarray(
                        replies[1]["actions"], dtype=np.float64
                    ),
                    duplicate_trace=replies[1]["crfs_trace"],
                    timing=timing,
                    diagnostics=diagnostics,
                    budget=budget,
                    config=config,
                )
                continue
            arm = _evaluated_arm(
                name,
                actions,
                environment,
                initial,
                config,
                mechanism=(
                    "recomputed approximate-clean signed-softplus trajectory collision field"
                ),
                controls={
                    "intervention_mode": "analytic_trajectory_field",
                    "intervention_step": step,
                    "model_l2_path_budget": budget,
                    "energy_margin_m": ENERGY_MARGIN_M,
                    "energy_temperature_m": ENERGY_TEMPERATURE_M,
                    "samples_per_segment": SAMPLES_PER_SEGMENT,
                    "clipping_policy": "fail_without_clipping",
                    "policy_timing": timing,
                },
                trace=trace,
                policy_replay_exact=True,
                duplicate_actions=np.asarray(replies[1]["actions"], dtype=np.float64),
                duplicate_trace=replies[1]["crfs_trace"],
            )
            arm["preintervention_trace_exact_to_frozen"] = diagnostics[
                "preintervention_trace_exact_to_frozen"
            ]
            arm["trace"] = _trace_record(trace)
            arm["timing"] = timing
            arm["diagnostics"] = diagnostics
            if arm["status"] == "bounds_failure":
                pass
            elif any(
                repeat.get("task_success_during_prefix") is True
                for repeat in arm.get("repeats", [])
                if isinstance(repeat, Mapping)
            ):
                arm["status"] = "terminal_failure"
                arm["gate"] = {
                    **dict(arm["gate"]),
                    "passed": False,
                    "reason": "task success occurred during the pregrasp five-action prefix",
                }
            elif diagnostics["zero_gradient_step_count"] > 0:
                arm["status"] = "zero_gradient_failure"
                arm["gate"] = {
                    **dict(arm["gate"]),
                    "passed": False,
                    "reason": "unsafe analytic field had a zero gradient",
                }
            arms[name] = arm

        arms = {name: arms[name] for name in ARMS}
        result = {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": ARTIFACT_TYPE,
            "gate": GATE,
            "case_id": case_id,
            "run_id": config.oracle.run_id,
            "status": "completed",
            "config_hash": config_hash,
            "source_evidence": source_evidence,
            "provenance": provenance,
            "pairing": pairing,
            "budget": budget_record,
            "arms": arms,
            "outcome": {
                "population": "r03_witness_confirmed_development_diagnostic",
                "source_arm_gate_pass": source_pass_map,
                "fresh_nominal_collision_reproduced": True,
                "arm_gate_pass": {
                    name: bool(arms[name]["gate"]["passed"]) for name in ARMS
                },
                "arm_status": {name: arms[name]["status"] for name in ARMS},
                "privileged_reference_sps_count": 9,
                "privileged_reference_population_size": 17,
                "kill_threshold_sps_count": 9,
                "population_decision": "deferred_to_population_summary",
            },
        }
        _validate_before_write(result)
        atomic_write_json(output, result)
        return output, "completed"
    finally:
        if owns_environment:
            environment.close()


__all__ = [
    "ARMS",
    "NEW_ARMS",
    "R03AConfig",
    "R03ASourceError",
    "r03a_config_from_mapping",
    "run_r03a_case",
]
