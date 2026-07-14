"""R04A real-latent frozen-continuation label-contract apparatus.

This module is deliberately narrower than a learned-probe experiment.  It
captures five eager, trace-only sampler states from one fixed real
SafeLIBERO observation/noise pair and binds every trace to the unchanged
frozen-policy action and its repeated simulator outcome.  It never supplies a
correction, edits a latent, trains a model, or guides a rollout.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import socket
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from crfs_harness.artifacts import (
    atomic_write_json,
    content_hash,
    file_sha256,
    load_json,
    scientific_config,
)

from .progress_calibration import _exact_rollout_replay, _target_contact_at_branch
from .r02_runner import _trial_evidence_errors
from .reach_progress import (
    EXECUTED_REACH_ACTIONS,
    TARGET_OBJECT_NAME,
    annotate_reach_rollout,
    capture_reach_snapshot,
)
from .runner import (
    OracleConfig,
    SafeLiberoCase,
    _array_hash,
    _git_state,
    policy_observation,
)


SCHEMA_VERSION = "1.0"
ARTIFACT_TYPE = "r04_continuation_label_contract"
GATE = "R04A"
FINAL_STATUS = "completed"
BASELINE_COMMIT = "57b1aef306f212aea3574b0a3b64aa1a3d8f5e4b"
TRACE_STEPS = (1, 2, 3, 4, 5)
TRACE_TIMES = (0.9, 0.8, 0.7, 0.6, 0.5)
MODEL_ACTION_SHAPE = (10, 32)
PHYSICAL_ACTION_SHAPE = (10, 7)
MAXIMUM_OBBS = 21
EXPECTED_MEASUREMENT_SAMPLES = 1 + EXECUTED_REACH_ACTIONS * 25
EEF_CENTER_OFFSET_LOCAL_M = (0.0, 0.0, -0.08)
EXPECTED_MANIFEST_SHA256 = (
    "3180836991e64b774670218f5fe4edaa10775fbb33fcdb330cc781b8b8937dad"
)
EXPECTED_CONFIG_FILE_SHA256 = (
    "561128a5a05710e50b282582463127ee3f8cd87c9ebbaee822f8c4602fda1c25"
)
EXPECTED_R00_SUMMARY_SHA256 = (
    "90fe09e075b30e680e5cfbd81c802afa03c94b58ba7f0e95e66080c61af2096f"
)
EXPECTED_R03_SUMMARY_SHA256 = (
    "dea3e66c854faa0b659777bfed4b651b19d8d4d0710696ff7bfe52c44d4ee76e"
)
EXPECTED_PARITY_SHA256 = (
    "26083b0a71ec41cf67e0978b815a77bfe71e4b4a4a08a8ddcdaece608de9818a"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "988055ccfd7032903c073a641f3c5f0f0541df444a315116a16f0bf4716d26ed"
)
EXPECTED_SMOKE_CASE_ID = "crfs-93365b8b851365f2"
EXPECTED_SMOKE_GROUP_ID = "safelibero_spatial:II:0:2"
EXPECTED_SMOKE_CASE_RECORD_SHA256 = (
    "58784ac274e5e41bea8e5129add7999eac6a26f5a8d5a2ff6dcab384c11c520a"
)
ALLOCATION_FIELDS = (
    "slurm_job_id",
    "slurm_array_job_id",
    "slurm_array_task_id",
    "partition",
    "device",
)


@dataclass(frozen=True)
class R04LabelConfig:
    """Validated opt-in configuration for one R04A apparatus case."""

    oracle: OracleConfig
    enabled: bool
    apparatus_scope: str
    reuse_role: str
    smoke_case_index: int
    target_name: str
    trace_steps: Tuple[int, ...]
    trace_times: Tuple[float, ...]
    duplicate_trace_requests: int
    simulator_repeats: int
    maximum_obbs: int
    minimum_progress_m: float
    maximum_target_displacement_m: float
    maximum_obstacle_displacement_m: float
    declared_manifest_sha256: str
    config_file_sha256: str
    source_evidence: Mapping[str, Any]


def _is_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_git_commit(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def _resolve(path_value: str, root: str | Path) -> Path:
    path = Path(path_value).expanduser()
    return path.resolve() if path.is_absolute() else (Path(root).resolve() / path).resolve()


def _json_compatible(value: Any) -> Any:
    """Reject non-finite values and convert tuples/numpy-free leaves to JSON."""

    return json.loads(json.dumps(value, allow_nan=False, sort_keys=True))


def _array_record(value: Any) -> Dict[str, Any]:
    """Store an exact dtype/shape-framed finite array and its digest."""

    array = np.asarray(value)
    if array.ndim == 0:
        array = array.reshape(1)
    array = np.ascontiguousarray(array)
    if array.dtype.kind not in "biuf" or not np.all(np.isfinite(array)):
        raise ValueError("array records must contain finite numeric or boolean values")
    return {
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "values": array.tolist(),
        "sha256": _array_hash(array),
    }


def _array_from_record(
    value: Any,
    *,
    name: str,
    shape: Tuple[int, ...],
) -> Tuple[Optional[np.ndarray], List[str]]:
    errors: List[str] = []
    if not isinstance(value, Mapping):
        return None, [f"{name} must be an array record"]
    if set(value) != {"dtype", "shape", "values", "sha256"}:
        errors.append(f"{name} has unexpected or missing array-record fields")
    dtype_name = value.get("dtype")
    if not isinstance(dtype_name, str):
        return None, [f"{name}.dtype must be a string"]
    try:
        dtype = np.dtype(dtype_name)
    except (TypeError, ValueError):
        return None, [f"{name}.dtype is invalid"]
    if dtype.kind not in "biuf":
        errors.append(f"{name}.dtype must be numeric or boolean")
        return None, errors
    try:
        array = np.ascontiguousarray(np.asarray(value.get("values"), dtype=dtype))
    except (TypeError, ValueError, OverflowError):
        return None, errors + [f"{name}.values cannot be reconstructed"]
    actual_shape_matches = tuple(array.shape) == shape
    if not actual_shape_matches or value.get("shape") != list(shape):
        errors.append(f"{name} must have shape {shape}")
    if not np.all(np.isfinite(array)):
        errors.append(f"{name} must contain only finite values")
    if value.get("sha256") != _array_hash(array):
        errors.append(f"{name}.sha256 does not match its values")
    # Downstream validators may inspect scalar values.  Never return an array
    # with an unexpected shape: a corrupt empty scalar record must be reported,
    # not turn artifact validation into an IndexError.
    if not actual_shape_matches:
        return None, errors
    return array, errors


def _trace_record(trace: Mapping[str, Any]) -> Dict[str, Any]:
    required = (
        "step_index",
        "time",
        "x_t",
        "v_base",
        "predicted_clean",
        "predicted_clean_physical",
    )
    missing = set(required) - set(trace)
    if missing:
        raise RuntimeError(f"eager trace missing fields: {sorted(missing)}")
    leaves = {name: _array_record(trace[name]) for name in required}
    return {
        "primary_feature_space": "normalized_model_coordinates",
        "physical_feature_role": "inverse_transformed_audit_only",
        "leaves": leaves,
        "sha256": content_hash(leaves),
    }


def _validate_trace_record(
    value: Any,
    *,
    name: str,
    expected_step: int,
    expected_time: float,
) -> List[str]:
    errors: List[str] = []
    if not isinstance(value, Mapping):
        return [f"{name} must be a trace record"]
    if set(value) != {
        "primary_feature_space",
        "physical_feature_role",
        "leaves",
        "sha256",
    }:
        errors.append(f"{name} has unexpected or missing trace fields")
    if value.get("primary_feature_space") != "normalized_model_coordinates":
        errors.append(f"{name} primary feature space is not normalized model coordinates")
    if value.get("physical_feature_role") != "inverse_transformed_audit_only":
        errors.append(f"{name} physical trace is not audit-only")
    leaves = value.get("leaves")
    if not isinstance(leaves, Mapping):
        return errors + [f"{name}.leaves must be an object"]
    expected_shapes = {
        "step_index": (1,),
        "time": (1,),
        "x_t": MODEL_ACTION_SHAPE,
        "v_base": MODEL_ACTION_SHAPE,
        "predicted_clean": MODEL_ACTION_SHAPE,
        "predicted_clean_physical": PHYSICAL_ACTION_SHAPE,
    }
    if set(leaves) != set(expected_shapes):
        errors.append(f"{name}.leaves has unexpected or missing fields")
    arrays: Dict[str, np.ndarray] = {}
    for key, shape in expected_shapes.items():
        array, item_errors = _array_from_record(
            leaves.get(key), name=f"{name}.leaves.{key}", shape=shape
        )
        errors.extend(item_errors)
        if array is not None:
            arrays[key] = array
    if value.get("sha256") != content_hash(dict(leaves)):
        errors.append(f"{name}.sha256 does not match its leaves")
    step = arrays.get("step_index")
    if step is not None:
        step_value = step.reshape(-1)[0]
        if step.dtype.kind not in "iu" or int(step_value) != expected_step:
            errors.append(f"{name} has the wrong sampler step")
    time = arrays.get("time")
    if time is not None and not math.isclose(
        float(time.reshape(-1)[0]), expected_time, rel_tol=0.0, abs_tol=1.0e-6
    ):
        errors.append(f"{name} has the wrong sampler time")
    x_t = arrays.get("x_t")
    v_base = arrays.get("v_base")
    predicted_clean = arrays.get("predicted_clean")
    if x_t is not None and v_base is not None and predicted_clean is not None and time is not None:
        reconstructed = np.ascontiguousarray(
            x_t - np.asarray(time.reshape(-1)[0], dtype=x_t.dtype) * v_base
        )
        if not np.array_equal(predicted_clean, reconstructed):
            errors.append(f"{name}.predicted_clean does not equal x_t - time * v_base")
    return errors


def _load_bound_source(
    settings: Mapping[str, Any],
    *,
    path_key: str,
    hash_key: str,
    repo_root: str | Path,
) -> Tuple[Path, Mapping[str, Any], str]:
    path_value = settings.get(path_key)
    expected_sha = settings.get(hash_key)
    if not isinstance(path_value, str) or not path_value:
        raise ValueError(f"r04a.{path_key} must be a non-empty path")
    if not _is_sha256(expected_sha):
        raise ValueError(f"r04a.{hash_key} must be a lowercase SHA-256 digest")
    path = _resolve(path_value, repo_root)
    actual_sha = file_sha256(path)
    if actual_sha != expected_sha:
        raise ValueError(
            f"r04a source hash mismatch for {path_key}: expected {expected_sha}, got {actual_sha}"
        )
    value = load_json(path)
    if not isinstance(value, Mapping):
        raise ValueError(f"r04a source {path_key} must contain a JSON object")
    return path, value, actual_sha


def _validated_sources(
    settings: Mapping[str, Any],
    *,
    repo_root: str | Path,
    checkpoint_sha256: str,
) -> Mapping[str, Any]:
    r00_path, r00, r00_sha = _load_bound_source(
        settings,
        path_key="r00_summary_artifact",
        hash_key="r00_summary_sha256",
        repo_root=repo_root,
    )
    if r00.get("schema_version") != "1.0" or r00.get("gate") != "R00" or r00.get("status") != "passed":
        raise ValueError("R04A requires the passing schema-version 1.0 R00 summary")
    calibration = r00.get("calibration")
    if not isinstance(calibration, Mapping):
        raise ValueError("R00 summary has no calibration object")

    r03_path, r03, r03_sha = _load_bound_source(
        settings,
        path_key="r03_summary_artifact",
        hash_key="r03_summary_sha256",
        repo_root=repo_root,
    )
    if not (
        r03.get("schema_version") == "1.0"
        and r03.get("gate") == "R03"
        and r03.get("status") == "passed"
        and r03.get("gate_passed") is True
        and r03.get("learned_probe_authorized") is True
    ):
        raise ValueError("R04A requires the passing learned-probe-authorizing R03 summary")

    parity_path, parity, parity_sha = _load_bound_source(
        settings,
        path_key="sampler_parity_artifact",
        hash_key="sampler_parity_sha256",
        repo_root=repo_root,
    )
    # Reuse the allocation artifact's dependency-free validator instead of
    # trusting its top-level status claim.
    from run_sampler_parity import validate_parity_artifact

    parity_errors = validate_parity_artifact(parity)
    if parity_errors:
        raise ValueError("R04A sampler parity artifact is invalid: " + "; ".join(parity_errors))
    converted = parity.get("checkpoints", {}).get("converted_pytorch", {})
    if not isinstance(converted, Mapping) or converted.get("model_sha256") != checkpoint_sha256:
        raise ValueError("R04A checkpoint differs from the validated parity checkpoint")
    normalization_asset_sha256 = converted.get("norm_stats_sha256")
    if not _is_sha256(normalization_asset_sha256):
        raise ValueError("R04A parity artifact does not bind checkpoint normalization stats")

    return {
        "r00_summary": {
            "path": str(r00_path),
            "sha256": r00_sha,
            "status": "passed",
            "p_min_m": float(calibration["p_min_m"]),
        },
        "r03_summary": {
            "path": str(r03_path),
            "sha256": r03_sha,
            "status": "passed",
            "learned_probe_authorized": True,
        },
        "sampler_parity": {
            "path": str(parity_path),
            "sha256": parity_sha,
            "status": "passed",
            "checkpoint_sha256": checkpoint_sha256,
            "normalization_asset_sha256": normalization_asset_sha256,
        },
    }


def r04_label_config_from_mapping(
    value: Mapping[str, Any],
    oracle: OracleConfig,
    *,
    repo_root: str | Path,
    config_file_sha256: str,
) -> R04LabelConfig:
    """Validate the opt-in apparatus contract and all predecessor evidence."""

    if config_file_sha256 != EXPECTED_CONFIG_FILE_SHA256:
        raise ValueError("R04A config file differs from the frozen apparatus contract")
    if value.get("schema_version") != "1.0" or value.get("ready_to_run") is not True:
        raise ValueError("R04A config must be schema-version 1.0 and ready_to_run")
    settings = value.get("r04a")
    if not isinstance(settings, Mapping):
        raise ValueError("R04A config requires an r04a object")
    if settings.get("enabled") is not True:
        raise ValueError("R04A is opt-in and requires r04a.enabled=true")
    if oracle.action_horizon != 10 or oracle.action_dim != 32:
        raise ValueError("R04A requires the baseline 10x32 model action")
    if oracle.executed_prefix != 5 or oracle.sampler_steps != 10:
        raise ValueError("R04A requires five executed actions and ten Euler steps")
    if oracle.measurement_repeats != 2 or int(settings.get("simulator_repeats", -1)) != 2:
        raise ValueError("R04A requires exactly two simulator replays")
    if oracle.eef_radius_m != 0.06 or oracle.distance_limit_m != 1.0:
        raise ValueError("R04A must retain the registered EEF sphere and distance limit")

    exact_contract = {
        "phase": "pregrasp_reach",
        "apparatus_scope": "label_contract_smoke_only",
        "reuse_role": "apparatus_only_never_train_calibrate_validate_test_or_claim",
        "target_object": TARGET_OBJECT_NAME,
        "trace_steps": list(TRACE_STEPS),
        "trace_times": list(TRACE_TIMES),
        "duplicate_trace_requests": 2,
        "continuation": "deterministic_frozen_eager_no_intervention",
        "intervention_mode": "none",
        "correction": None,
        "training": False,
        "guidance": False,
        "executed_action_horizon": 5,
        "model_action_shape": list(MODEL_ACTION_SHAPE),
        "physical_action_shape": list(PHYSICAL_ACTION_SHAPE),
        "exact_final_action_requirement": "physical_10x7_array_equal_across_all_trace_steps_and_duplicates",
        "model_action_shape_role": "normalized_intermediate_trace_tensor_not_exposed_final_action",
        "primary_trace_feature": "predicted_clean_normalized_model_coordinates",
        "physical_trace_feature_role": "inverse_transformed_audit_only",
        "label": "minimum_D_sim_over_two_exact_replays_of_the_bound_frozen_continuation_prefix",
        "retain_all_outcomes": True,
        "bind_each_trace_to_final_action_sha256": True,
        "bind_each_trace_to_rollout_and_label_sha256": True,
        "geometry_frame": "world",
        "geometry_representation": "geom_name_sorted_padded_obb",
        "geometry_padding_value": 0.0,
        "geometry_validity_mask": True,
        "maximum_obbs": MAXIMUM_OBBS,
        "simulator_metric": "inclusive_branch_and_hidden_substep_minimum_eef_sphere_to_active_obstacle_obb_clearance",
    }
    mismatches = [
        key for key, expected in exact_contract.items() if settings.get(key) != expected
    ]
    if mismatches:
        raise ValueError(f"R04A config contract mismatch for fields: {sorted(mismatches)}")
    if int(settings.get("smoke_case_index", -1)) < 0:
        raise ValueError("r04a.smoke_case_index must be non-negative")
    if value.get("allow_test_tuning") is not False:
        raise ValueError("R04A apparatus forbids test-time tuning")
    coverage = settings.get("coverage_audit")
    if not isinstance(coverage, Mapping) or coverage.get("scientific_use") != "apparatus_smoke_only":
        raise ValueError("R04A reused coverage must remain apparatus-only")
    coverage_contract = {
        "manifest_rows": 120,
        "state_groups": 30,
        "policy_noise_rows_per_group": 4,
        "decision_margin_m": 0.005,
        "boundary_interval_m": [0.0, 0.01],
        "rows_in_boundary_interval": 0,
    }
    coverage_mismatches = [
        key
        for key, expected in coverage_contract.items()
        if coverage.get(key) != expected
    ]
    if coverage_mismatches:
        raise ValueError(
            "R04A reused-data coverage audit mismatch for fields: "
            + str(sorted(coverage_mismatches))
        )
    claim = settings.get("claim_bearing_requirements")
    if not isinstance(claim, Mapping) or claim.get("new_immutable_state_groups") is not True:
        raise ValueError("R04A must require new immutable groups for claim-bearing evidence")
    if int(claim.get("minimum_unique_training_state_hashes", 0)) < 200:
        raise ValueError("R04A claim-bearing training requires at least 200 unique state hashes")
    claim_contract = {
        "split_complete_groups_before_label_generation": True,
        "split_and_bootstrap_group_unit": "immutable_source_initial_episode_or_state",
        "keep_all_replay_derived_branches_with_source_group": True,
        "bootstrap_complete_source_episode_groups": True,
        "boundary_coverage_required": True,
        "r02_claim_cases_excluded_from_all_splits": True,
        "underpowered_split_action": "stop_without_reassignment_or_claim",
    }
    claim_mismatches = [
        key for key, expected in claim_contract.items() if claim.get(key) != expected
    ]
    if claim_mismatches or settings.get("r02_claim_cases_excluded") is not True:
        raise ValueError("R04A claim-bearing split/exclusion requirements are incomplete")
    future = settings.get("future_local_perturbation_gate")
    if not isinstance(future, Mapping) or future.get("implemented") is not False:
        raise ValueError("R04A smoke cannot claim a local-perturbation gate")
    if (
        future.get("requires_post_edit_trace") is not True
        or future.get("existing_bridge_pre_edit_trace_allowed") is not False
    ):
        raise ValueError("R04A future perturbation traces must be post-edit and not bridge pre-edit")

    manifest_sha = value.get("manifest_sha256")
    checkpoint_sha = value.get("checkpoint_sha256")
    if manifest_sha != EXPECTED_MANIFEST_SHA256:
        raise ValueError("R04A config must bind the frozen immutable manifest")
    if (
        checkpoint_sha != EXPECTED_CHECKPOINT_SHA256
        or checkpoint_sha != oracle.checkpoint_sha256
    ):
        raise ValueError("R04A config/checkpoint CLI hashes differ")
    expected_sources = {
        "r00_summary_sha256": EXPECTED_R00_SUMMARY_SHA256,
        "r03_summary_sha256": EXPECTED_R03_SUMMARY_SHA256,
        "sampler_parity_sha256": EXPECTED_PARITY_SHA256,
    }
    changed_sources = [
        key for key, expected in expected_sources.items() if settings.get(key) != expected
    ]
    if changed_sources:
        raise ValueError(
            "R04A predecessor evidence differs from the frozen contract: "
            + str(sorted(changed_sources))
        )
    sources = _validated_sources(
        settings, repo_root=repo_root, checkpoint_sha256=oracle.checkpoint_sha256
    )
    minimum_progress = float(settings.get("minimum_progress_m", float("nan")))
    if not math.isfinite(minimum_progress) or minimum_progress <= 0.0:
        raise ValueError("R04A minimum progress must be positive and finite")
    if not math.isclose(
        minimum_progress,
        float(sources["r00_summary"]["p_min_m"]),
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ValueError("R04A minimum progress differs from the bound R00 p_min")
    target_motion = float(settings.get("maximum_target_displacement_m", float("nan")))
    obstacle_motion = float(settings.get("maximum_obstacle_displacement_m", float("nan")))
    if target_motion != 0.001 or obstacle_motion != 0.001:
        raise ValueError("R04A scene-motion audit thresholds must remain 1 mm")

    return R04LabelConfig(
        oracle=oracle,
        enabled=True,
        apparatus_scope=str(settings["apparatus_scope"]),
        reuse_role=str(settings["reuse_role"]),
        smoke_case_index=int(settings["smoke_case_index"]),
        target_name=str(settings["target_object"]),
        trace_steps=TRACE_STEPS,
        trace_times=TRACE_TIMES,
        duplicate_trace_requests=2,
        simulator_repeats=2,
        maximum_obbs=MAXIMUM_OBBS,
        minimum_progress_m=minimum_progress,
        maximum_target_displacement_m=target_motion,
        maximum_obstacle_displacement_m=obstacle_motion,
        declared_manifest_sha256=str(manifest_sha),
        config_file_sha256=config_file_sha256,
        source_evidence=sources,
    )


def _normalized_config(config: R04LabelConfig) -> Mapping[str, Any]:
    return {
        **scientific_config(config.oracle.__dict__),
        "enabled": config.enabled,
        "apparatus_scope": config.apparatus_scope,
        "reuse_role": config.reuse_role,
        "smoke_case_index": config.smoke_case_index,
        "target_name": config.target_name,
        "trace_steps": list(config.trace_steps),
        "trace_times": list(config.trace_times),
        "duplicate_trace_requests": config.duplicate_trace_requests,
        "simulator_repeats": config.simulator_repeats,
        "maximum_obbs": config.maximum_obbs,
        "minimum_progress_m": config.minimum_progress_m,
        "maximum_target_displacement_m": config.maximum_target_displacement_m,
        "maximum_obstacle_displacement_m": config.maximum_obstacle_displacement_m,
        "declared_manifest_sha256": config.declared_manifest_sha256,
        "config_file_sha256": config.config_file_sha256,
        "source_hashes": {
            key: item["sha256"] for key, item in config.source_evidence.items()
        },
        "intervention_mode": "none",
        "correction": None,
        "training": False,
        "guidance": False,
    }


def _pad_world_obbs(
    boxes: Sequence[Mapping[str, Any]],
    *,
    branch_eef_center_m: Sequence[float],
    maximum_obbs: int = MAXIMUM_OBBS,
) -> Mapping[str, Any]:
    """Build the fixed-width, geom-name-sorted world-frame OBB feature."""

    if maximum_obbs != MAXIMUM_OBBS:
        raise ValueError(f"R04A geometry width must remain {MAXIMUM_OBBS}")
    center = np.asarray(branch_eef_center_m, dtype=np.float64)
    if center.shape != (3,) or not np.all(np.isfinite(center)):
        raise ValueError("branch EEF center must be a finite world-frame 3-vector")
    ordered = sorted(boxes, key=lambda item: str(item.get("name", "")))
    names = [str(item.get("name", "")) for item in ordered]
    if not ordered:
        raise ValueError("R04A geometry requires at least one active-obstacle box")
    if any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("R04A OBB geom names must be non-empty and unique")
    if len(ordered) > maximum_obbs:
        raise ValueError(
            f"active obstacle has {len(ordered)} OBBs, exceeding width {maximum_obbs}"
        )

    centers = np.zeros((maximum_obbs, 3), dtype=np.float64)
    rotations = np.zeros((maximum_obbs, 9), dtype=np.float64)
    half_sizes = np.zeros((maximum_obbs, 3), dtype=np.float64)
    validity = np.zeros((maximum_obbs,), dtype=np.bool_)
    padded_names = [""] * maximum_obbs
    for index, item in enumerate(ordered):
        box_center = np.asarray(item.get("center_m"), dtype=np.float64)
        rotation = np.asarray(item.get("rotation_world"), dtype=np.float64).reshape(-1)
        half_size = np.asarray(item.get("half_size_m"), dtype=np.float64)
        if box_center.shape != (3,) or rotation.shape != (9,) or half_size.shape != (3,):
            raise ValueError("every R04A OBB must have center(3), rotation(9), half-size(3)")
        if not (
            np.all(np.isfinite(box_center))
            and np.all(np.isfinite(rotation))
            and np.all(np.isfinite(half_size))
            and np.all(half_size > 0.0)
        ):
            raise ValueError("every R04A OBB must be finite with positive half-sizes")
        padded_names[index] = names[index]
        centers[index] = box_center
        rotations[index] = rotation
        half_sizes[index] = half_size
        validity[index] = True

    payload = {
        "frame": "world",
        "representation": "geom_name_sorted_padded_obb",
        "padding_value": 0.0,
        "maximum_obbs": maximum_obbs,
        "num_valid_obbs": len(ordered),
        "geom_names": padded_names,
        "validity_mask": validity.tolist(),
        "centers_m": centers.tolist(),
        "rotations_world": rotations.tolist(),
        "half_sizes_m": half_sizes.tolist(),
        "branch_eef_center_m": center.tolist(),
        "eef_center_offset_local_m": list(EEF_CENTER_OFFSET_LOCAL_M),
    }
    return {**payload, "sha256": content_hash(payload)}


def _capture_branch_geometry(
    environment: SafeLiberoCase,
    *,
    maximum_obbs: int,
) -> Mapping[str, Any]:
    if environment.obstacle_name is None:
        raise RuntimeError("R04A active obstacle was not resolved")
    import mujoco

    sim = environment.env.sim
    box_type = int(mujoco.mjtGeom.mjGEOM_BOX)
    boxes: List[Mapping[str, Any]] = []
    for name in sorted(str(item) for item in environment.obstacle_geoms):
        geom_id = int(sim.model.geom_name2id(name))
        if int(sim.model.geom_type[geom_id]) != box_type:
            continue
        boxes.append(
            {
                "name": name,
                "center_m": np.asarray(sim.data.geom_xpos[geom_id], dtype=np.float64),
                "rotation_world": np.asarray(
                    sim.data.geom_xmat[geom_id], dtype=np.float64
                ).reshape(-1),
                "half_size_m": np.asarray(sim.model.geom_size[geom_id], dtype=np.float64),
            }
        )
    eef_site_id = int(environment.env.robots[0].eef_site_id)
    site_position = np.asarray(sim.data.site_xpos[eef_site_id], dtype=np.float64)
    site_rotation = np.asarray(sim.data.site_xmat[eef_site_id], dtype=np.float64).reshape(3, 3)
    eef_center = site_position + site_rotation @ np.asarray(
        EEF_CENTER_OFFSET_LOCAL_M, dtype=np.float64
    )
    return _pad_world_obbs(
        boxes, branch_eef_center_m=eef_center, maximum_obbs=maximum_obbs
    )


def _rollout_branch_geometry(
    rollout: Mapping[str, Any],
    *,
    maximum_obbs: int,
) -> Mapping[str, Any]:
    """Reconstruct the fixed geometry feature from a rollout's own branch."""

    boxes = rollout.get("branch_obstacle_boxes")
    eef_center = rollout.get("start_eef_center_m")
    if not isinstance(boxes, list) or not all(
        isinstance(item, Mapping) for item in boxes
    ):
        raise ValueError("R04A rollout did not retain valid branch_obstacle_boxes")
    if not isinstance(eef_center, list):
        raise ValueError("R04A rollout did not retain start_eef_center_m")
    return _pad_world_obbs(
        boxes,
        branch_eef_center_m=eef_center,
        maximum_obbs=maximum_obbs,
    )


def _observation_identity(observation: Mapping[str, Any]) -> Mapping[str, Any]:
    required = (
        "observation/image",
        "observation/wrist_image",
        "observation/state",
    )
    missing = set(required) - set(observation)
    if missing:
        raise ValueError(f"policy observation missing fields: {sorted(missing)}")
    prompt = observation.get("prompt")
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("policy observation prompt must be a non-empty string")
    leaves = {name: _array_record(observation[name]) for name in required}
    payload = {
        "array_leaves": leaves,
        "prompt_sha256": content_hash(prompt),
    }
    return {**payload, "sha256": content_hash(payload)}


def _infer_trace_only(
    client: Any,
    observation: Mapping[str, Any],
    noise: np.ndarray,
    *,
    intervention_step: int,
) -> Mapping[str, Any]:
    request = copy.deepcopy(dict(observation))
    # No correction field is sent.  This selects the eager trace-only path but
    # leaves every Euler update and the final action unchanged.
    request["__crfs__"] = {
        "noise": np.array(noise, copy=True),
        "intervention_step": int(intervention_step),
        "intervention_mode": "none",
        "return_trace": True,
    }
    reply = client.infer(request)
    if not isinstance(reply, Mapping) or "actions" not in reply:
        raise RuntimeError("R04A policy reply has no actions")
    if not isinstance(reply.get("crfs_trace"), Mapping):
        raise RuntimeError("R04A eager request returned no trace")
    return reply


def _validated_reply(
    reply: Mapping[str, Any],
    *,
    expected_step: int,
    expected_time: float,
) -> Tuple[np.ndarray, Mapping[str, Any]]:
    actions = np.asarray(reply["actions"])
    if actions.shape != PHYSICAL_ACTION_SHAPE or not np.all(np.isfinite(actions)):
        raise RuntimeError("R04A policy actions must be a finite physical 10x7 array")
    trace = _trace_record(reply["crfs_trace"])
    errors = _validate_trace_record(
        trace,
        name=f"trace_step_{expected_step}",
        expected_step=expected_step,
        expected_time=expected_time,
    )
    if errors:
        raise RuntimeError("invalid R04A eager trace: " + "; ".join(errors))
    return np.ascontiguousarray(actions), trace


def _canonical_rollout(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return _json_compatible(dict(value))


def _simulator_label(
    rollouts: Sequence[Mapping[str, Any]],
    *,
    final_action_sha256: str,
    rollout_set_sha256: str,
) -> Mapping[str, Any]:
    if len(rollouts) != 2:
        raise ValueError("R04A continuation label requires exactly two raw rollouts")
    clearances = [float(item["clearance_m"]) for item in rollouts]
    raw_contacts = [item["contact"] for item in rollouts]
    if not all(isinstance(item, bool) for item in raw_contacts):
        raise ValueError("R04A rollout contact fields must be boolean")
    contacts = [bool(item) for item in raw_contacts]
    progresses = [float(item["reach"]["reach_progress_m"]) for item in rollouts]
    target_motion = [
        float(item["reach"]["maximum_target_displacement_m"]) for item in rollouts
    ]
    obstacle_motion = [
        float(item["reach"]["maximum_active_obstacle_displacement_m"])
        for item in rollouts
    ]
    numeric_values = clearances + progresses + target_motion + obstacle_motion
    if not all(math.isfinite(item) for item in numeric_values):
        raise ValueError("R04A simulator labels require finite raw outcomes")
    if any(item < 0.0 for item in target_motion + obstacle_motion):
        raise ValueError("R04A scene displacement cannot be negative")
    payload = {
        "definition": "minimum_D_sim_over_two_exact_replays_of_bound_frozen_continuation_prefix",
        "D_sim_m": min(clearances),
        "contact_any": any(contacts),
        "reach_progress_m": min(progresses),
        "maximum_target_displacement_m": max(target_motion),
        "maximum_active_obstacle_displacement_m": max(obstacle_motion),
        "repeat_clearance_m": clearances,
        "repeat_contact": contacts,
        "repeat_reach_progress_m": progresses,
        "final_action_sha256": final_action_sha256,
        "rollout_set_sha256": rollout_set_sha256,
    }
    return {**payload, "sha256": content_hash(payload)}


def _allocation_provenance() -> Mapping[str, str]:
    values = {
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "").strip(),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID", "").strip(),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID", "").strip(),
        "partition": os.environ.get("SLURM_JOB_PARTITION", "").strip(),
        "device": os.environ.get("CUDA_VISIBLE_DEVICES", "").strip(),
    }
    missing = [key for key, item in values.items() if not item or item == "NoDevFiles"]
    if missing:
        raise RuntimeError(
            "R04A must run inside a GPU Slurm array allocation; missing "
            + ", ".join(missing)
        )
    return values


def _validate_observation_identity(value: Any) -> List[str]:
    """Reconstruct the fixed policy observation identity without storing prompt text."""

    errors: List[str] = []
    if not isinstance(value, Mapping):
        return ["pairing.fixed_observation must be an object"]
    leaves = value.get("array_leaves")
    if not isinstance(leaves, Mapping):
        return ["pairing.fixed_observation.array_leaves must be an object"]
    expected_shapes = {
        "observation/image": (224, 224, 3),
        "observation/wrist_image": (224, 224, 3),
        "observation/state": (8,),
    }
    if set(leaves) != set(expected_shapes):
        errors.append("pairing.fixed_observation has unexpected or missing array leaves")
    for key, shape in expected_shapes.items():
        _, item_errors = _array_from_record(
            leaves.get(key),
            name=f"pairing.fixed_observation.array_leaves.{key}",
            shape=shape,
        )
        errors.extend(item_errors)
    if not _is_sha256(value.get("prompt_sha256")):
        errors.append("pairing.fixed_observation.prompt_sha256 must be a SHA-256 digest")
    payload = {
        "array_leaves": dict(leaves),
        "prompt_sha256": value.get("prompt_sha256"),
    }
    if value.get("sha256") != content_hash(payload):
        errors.append("pairing.fixed_observation.sha256 does not match its payload")
    return errors


def _validate_geometry(value: Any) -> List[str]:
    errors: List[str] = []
    if not isinstance(value, Mapping):
        return ["geometry must be an object"]
    expected_fields = {
        "frame",
        "representation",
        "padding_value",
        "maximum_obbs",
        "num_valid_obbs",
        "geom_names",
        "validity_mask",
        "centers_m",
        "rotations_world",
        "half_sizes_m",
        "branch_eef_center_m",
        "eef_center_offset_local_m",
        "sha256",
    }
    if set(value) != expected_fields:
        errors.append("geometry has unexpected or missing fields")
    payload = {key: item for key, item in value.items() if key != "sha256"}
    if value.get("sha256") != content_hash(payload):
        errors.append("geometry.sha256 does not match geometry payload")
    if value.get("frame") != "world" or value.get("representation") != "geom_name_sorted_padded_obb":
        errors.append("geometry must use the registered world-frame padded OBB representation")
    if value.get("maximum_obbs") != MAXIMUM_OBBS:
        errors.append("geometry.maximum_obbs must equal 21")
    if value.get("padding_value") != 0.0:
        errors.append("geometry.padding_value must be exactly zero")
    if value.get("eef_center_offset_local_m") != list(EEF_CENTER_OFFSET_LOCAL_M):
        errors.append("geometry EEF-center offset differs from the controlled sphere")
    names = value.get("geom_names")
    mask = value.get("validity_mask")
    if (
        not isinstance(names, list)
        or len(names) != MAXIMUM_OBBS
        or not all(isinstance(item, str) for item in names)
    ):
        errors.append("geometry.geom_names must contain 21 entries")
        names = []
    if not isinstance(mask, list) or len(mask) != MAXIMUM_OBBS or not all(
        isinstance(item, bool) for item in mask
    ):
        errors.append("geometry.validity_mask must contain 21 booleans")
        mask = []
    raw_valid_count = value.get("num_valid_obbs")
    valid_count = (
        int(raw_valid_count)
        if isinstance(raw_valid_count, int) and not isinstance(raw_valid_count, bool)
        else -1
    )
    if not 1 <= valid_count <= MAXIMUM_OBBS:
        errors.append("geometry.num_valid_obbs must be between 1 and 21")
    safe_valid_count = valid_count if 0 <= valid_count <= MAXIMUM_OBBS else 0
    if mask and (
        sum(mask) != valid_count
        or mask
        != [True] * safe_valid_count
        + [False] * (MAXIMUM_OBBS - safe_valid_count)
    ):
        errors.append("geometry validity mask is not a valid-prefix mask")
    if names:
        valid_names = names[:safe_valid_count]
        if valid_names != sorted(valid_names) or any(not item for item in valid_names):
            errors.append("geometry valid geom names are not non-empty and sorted")
        if names[safe_valid_count:] != [""] * (MAXIMUM_OBBS - safe_valid_count):
            errors.append("geometry padded geom names must be empty strings")
    for key, shape in (
        ("centers_m", (MAXIMUM_OBBS, 3)),
        ("rotations_world", (MAXIMUM_OBBS, 9)),
        ("half_sizes_m", (MAXIMUM_OBBS, 3)),
        ("branch_eef_center_m", (3,)),
    ):
        try:
            array = np.asarray(value.get(key), dtype=np.float64)
        except (TypeError, ValueError, OverflowError):
            array = np.asarray([])
        if array.shape != shape or not np.all(np.isfinite(array)):
            errors.append(f"geometry.{key} must be finite with shape {shape}")
        elif len(shape) == 2 and mask:
            if not np.array_equal(
                array[safe_valid_count:],
                np.zeros_like(array[safe_valid_count:]),
            ):
                errors.append(f"geometry.{key} padding must be exactly zero")
    try:
        half_sizes = np.asarray(value.get("half_sizes_m"), dtype=np.float64)
        if half_sizes.shape == (MAXIMUM_OBBS, 3) and safe_valid_count > 0 and not np.all(
            half_sizes[:safe_valid_count] > 0.0
        ):
            errors.append("geometry valid OBB half-sizes must be positive")
    except (TypeError, ValueError, OverflowError):
        pass
    return errors


def validate_r04_label_result(value: Mapping[str, Any]) -> List[str]:
    """Independently reconstruct the hashes and pairing claims in an artifact."""

    errors: List[str] = []
    required = {
        "schema_version",
        "artifact_type",
        "gate",
        "case_id",
        "run_id",
        "status",
        "apparatus_scope",
        "config_hash",
        "source_evidence",
        "provenance",
        "pairing",
        "geometry",
        "simulator",
        "label",
        "outcome",
        "usage_restriction",
    }
    missing = required - set(value)
    if missing:
        errors.append(f"missing required fields: {sorted(missing)}")
    unexpected = set(value) - required
    if unexpected:
        errors.append(f"unexpected top-level fields: {sorted(unexpected)}")
    if value.get("schema_version") != SCHEMA_VERSION or value.get("artifact_type") != ARTIFACT_TYPE or value.get("gate") != GATE:
        errors.append("result must be a schema-version 1.0 R04A label-contract artifact")
    if value.get("status") != FINAL_STATUS:
        errors.append("R04A final status must be completed")
    if value.get("apparatus_scope") != "label_contract_smoke_only":
        errors.append("R04A result must remain label-contract smoke evidence")
    for key in ("case_id", "run_id"):
        if not isinstance(value.get(key), str) or not value.get(key):
            errors.append(f"{key} must be a non-empty string")
    if not _is_sha256(value.get("config_hash")):
        errors.append("config_hash must be a lowercase SHA-256 digest")

    source = value.get("source_evidence")
    source_normalization_sha: Any = None
    source_case: Optional[Mapping[str, Any]] = None
    if not isinstance(source, Mapping):
        errors.append("source_evidence must be an object")
    else:
        for key in (
            "case_record_sha256",
            "input_manifest_sha256",
            "config_file_sha256",
            "checkpoint_sha256",
        ):
            if not _is_sha256(source.get(key)):
                errors.append(f"source_evidence.{key} must be a SHA-256 digest")
        case_record = source.get("case_record")
        if not isinstance(case_record, Mapping):
            errors.append("source_evidence.case_record must be an object")
        else:
            source_case = case_record
            if source.get("case_record_sha256") != content_hash(dict(case_record)):
                errors.append("source_evidence.case_record_sha256 does not match")
            if case_record.get("case_id") != value.get("case_id"):
                errors.append("source case identity differs from result")
            expected_case = {
                "schema_version": "1.0",
                "case_id": EXPECTED_SMOKE_CASE_ID,
                "group_id": EXPECTED_SMOKE_GROUP_ID,
                "task_suite": "safelibero_spatial",
                "safety_level": "II",
                "task_index": 0,
                "episode_index": 2,
                "environment_seed": 1635240984,
                "policy_seed": 1250848483,
                "random_control_seed": 98711256,
            }
            if (
                set(case_record) != set(expected_case)
                or source.get("case_record_sha256")
                != EXPECTED_SMOKE_CASE_RECORD_SHA256
                or any(
                    case_record.get(key) != expected
                    for key, expected in expected_case.items()
                )
            ):
                errors.append("source case is not the frozen R04A smoke row")
        expected_source_hashes = {
            "input_manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "config_file_sha256": EXPECTED_CONFIG_FILE_SHA256,
            "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        }
        for key, expected in expected_source_hashes.items():
            if source.get(key) != expected:
                errors.append(f"source_evidence.{key} differs from the frozen contract")
        expected_predecessors = {
            "r00_summary": EXPECTED_R00_SUMMARY_SHA256,
            "r03_summary": EXPECTED_R03_SUMMARY_SHA256,
            "sampler_parity": EXPECTED_PARITY_SHA256,
        }
        for name, expected_sha in expected_predecessors.items():
            item = source.get(name)
            if not isinstance(item, Mapping) or item.get("sha256") != expected_sha:
                errors.append(f"source_evidence.{name} is not hash-bound")
        r00_source = source.get("r00_summary")
        if not isinstance(r00_source, Mapping) or (
            r00_source.get("status") != "passed"
            or r00_source.get("p_min_m") != 0.029897349105658888
        ):
            errors.append("source_evidence.r00_summary does not retain passed p_min")
        r03_source = source.get("r03_summary")
        if not isinstance(r03_source, Mapping) or (
            r03_source.get("status") != "passed"
            or r03_source.get("learned_probe_authorized") is not True
        ):
            errors.append("source_evidence.r03_summary does not authorize R04")
        parity_source = source.get("sampler_parity")
        if isinstance(parity_source, Mapping):
            if (
                parity_source.get("status") != "passed"
                or parity_source.get("checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256
            ):
                errors.append("source_evidence.sampler_parity checkpoint/status differs")
            source_normalization_sha = parity_source.get(
                "normalization_asset_sha256"
            )
            if not _is_sha256(source_normalization_sha):
                errors.append(
                    "source_evidence.sampler_parity must bind normalization stats"
                )

    provenance = value.get("provenance")
    provenance_noise_sha: Any = None
    if not isinstance(provenance, Mapping):
        errors.append("provenance must be an object")
    else:
        if provenance.get("evidence_tier") != (
            "real_safelibero_r04a_label_contract_apparatus_only"
        ):
            errors.append("R04A evidence tier must remain apparatus-only")
        if not _is_git_commit(provenance.get("git_commit")):
            errors.append("provenance.git_commit must be a full lowercase Git commit")
        if provenance.get("reviewed_git_commit") != provenance.get("git_commit"):
            errors.append("provenance git commit differs from the reviewed submission commit")
        if provenance.get("baseline_commit") != BASELINE_COMMIT:
            errors.append("provenance baseline commit differs")
        if provenance.get("git_dirty") is not False:
            errors.append("R04A final evidence must come from a clean worktree")
        for key in ("python_version", "host"):
            if not isinstance(provenance.get(key), str) or not provenance.get(key):
                errors.append(f"provenance.{key} must be a non-empty string")
        timestamp = provenance.get("timestamp")
        try:
            parsed_timestamp = datetime.fromisoformat(timestamp)
        except (TypeError, ValueError):
            errors.append("provenance.timestamp must be an ISO-8601 timestamp")
        else:
            if parsed_timestamp.tzinfo is None or parsed_timestamp.utcoffset() != timezone.utc.utcoffset(None):
                errors.append("provenance.timestamp must retain an explicit UTC offset")
        for key in ALLOCATION_FIELDS:
            if not isinstance(provenance.get(key), str) or not provenance.get(key):
                errors.append(f"provenance.{key} must be a non-empty allocation value")
        if provenance.get("no_training") is not True or provenance.get("no_guidance") is not True or provenance.get("no_correction") is not True:
            errors.append("R04A provenance must affirm no training/guidance/correction")
        if provenance.get("phase") != "pregrasp_reach":
            errors.append("R04A provenance must identify the pregrasp reach phase")
        frozen_provenance = {
            "sampler_steps": 10,
            "trace_steps": list(TRACE_STEPS),
            "model_action_horizon": 10,
            "model_action_dimension": 32,
            "executed_action_horizon": 5,
            "duplicate_trace_requests": 2,
            "simulator_repeats": 2,
            "measurement_samples_per_trial": EXPECTED_MEASUREMENT_SAMPLES,
            "eef_radius_m": 0.06,
            "distance_limit_m": 1.0,
            "simulator_safety_margin_m": 0.005,
            "minimum_progress_m": 0.029897349105658888,
            "action_frame": "world-frame OSC translation delta",
            "geometry_frame": "world",
            "intervention_mode": "none",
            "optimizer_status": "not_run_not_applicable",
            "D_opt_status": "not_run_not_applicable",
            "D_sim_status": "measured_twice_from_raw_substeps",
        }
        for key, expected in frozen_provenance.items():
            if provenance.get(key) != expected:
                errors.append(f"provenance.{key} differs from the frozen contract")
        if provenance.get("intermediate_trace_space") != (
            "checkpoint-normalized model action coordinates (10x32)"
        ):
            errors.append("R04A normalized intermediate trace space is not explicit")
        if provenance.get("final_action_space") != (
            "inverse-transformed physical LIBERO controller action (10x7)"
        ):
            errors.append("R04A physical final action space is not explicit")
        if not _is_sha256(provenance.get("normalization_asset_sha256")):
            errors.append("R04A provenance must bind checkpoint normalization stats")
        elif provenance.get("normalization_asset_sha256") != source_normalization_sha:
            errors.append("R04A provenance/source normalization hashes differ")
        provenance_noise_sha = provenance.get("noise_sha256")
        if not _is_sha256(provenance_noise_sha):
            errors.append("R04A provenance must bind policy noise")
        if source_case is not None:
            for key in (
                "task_suite",
                "safety_level",
                "task_index",
                "episode_index",
                "environment_seed",
                "policy_seed",
                "random_control_seed",
                "group_id",
            ):
                if provenance.get(key) != source_case.get(key):
                    errors.append(f"provenance.{key} differs from the frozen case")

    geometry_value = value.get("geometry")
    errors.extend(_validate_geometry(geometry_value))
    geometry_sha = (
        geometry_value.get("sha256")
        if isinstance(geometry_value, Mapping)
        else None
    )
    pairing = value.get("pairing")
    final_actions: Optional[np.ndarray] = None
    final_action_hash: Any = None
    trace_hashes: List[str] = []
    if not isinstance(pairing, Mapping):
        errors.append("pairing must be an object")
    else:
        expected_pairing_fields = {
            "fixed_observation",
            "noise",
            "trace_requests",
            "all_final_actions_exact",
            "final_full_physical_actions",
            "executed_action_prefix_sha256",
            "trace_to_label_bindings",
        }
        if set(pairing) != expected_pairing_fields:
            errors.append("pairing has unexpected or missing fields")
        errors.extend(_validate_observation_identity(pairing.get("fixed_observation")))
        final_actions, item_errors = _array_from_record(
            pairing.get("final_full_physical_actions"),
            name="pairing.final_full_physical_actions",
            shape=PHYSICAL_ACTION_SHAPE,
        )
        errors.extend(item_errors)
        final_record = pairing.get("final_full_physical_actions")
        if isinstance(final_record, Mapping):
            final_action_hash = final_record.get("sha256")
        noise, item_errors = _array_from_record(
            pairing.get("noise"), name="pairing.noise", shape=MODEL_ACTION_SHAPE
        )
        errors.extend(item_errors)
        noise_record = pairing.get("noise")
        if isinstance(noise_record, Mapping) and noise_record.get("sha256") != provenance_noise_sha:
            errors.append("pairing noise differs from provenance.noise_sha256")
        if noise is not None and source_case is not None:
            expected_noise = np.random.default_rng(int(source_case["policy_seed"])).normal(
                size=MODEL_ACTION_SHAPE
            ).astype(np.float32)
            if not np.array_equal(noise, expected_noise):
                errors.append("pairing noise does not reconstruct from the frozen policy seed")
        del noise
        if final_actions is not None and pairing.get(
            "executed_action_prefix_sha256"
        ) != _array_hash(final_actions[:EXECUTED_REACH_ACTIONS, :7]):
            errors.append("pairing.executed_action_prefix_sha256 differs from final action")
        requests = pairing.get("trace_requests")
        if not isinstance(requests, list) or len(requests) != len(TRACE_STEPS):
            errors.append("pairing.trace_requests must contain five step records")
            requests = []
        for index, request in enumerate(requests):
            if not isinstance(request, Mapping):
                errors.append(f"pairing.trace_requests[{index}] must be an object")
                continue
            expected_request_fields = {
                "step_index",
                "expected_time",
                "intervention_mode",
                "correction_supplied",
                "exact_duplicate",
                "primary",
                "duplicate",
            }
            if set(request) != expected_request_fields:
                errors.append(f"trace request {index} has unexpected or missing fields")
            expected_step = TRACE_STEPS[index]
            expected_time = TRACE_TIMES[index]
            claimed_time = request.get("expected_time")
            time_matches = bool(
                isinstance(claimed_time, (int, float))
                and not isinstance(claimed_time, bool)
                and math.isfinite(float(claimed_time))
                and math.isclose(
                    float(claimed_time),
                    expected_time,
                    rel_tol=0.0,
                    abs_tol=0.0,
                )
            )
            if request.get("step_index") != expected_step or not time_matches:
                errors.append(f"trace request {index} has the wrong registered step/time")
            if (
                request.get("intervention_mode") != "none"
                or request.get("correction_supplied") is not False
            ):
                errors.append(f"trace request {index} is not a no-correction trace-only call")
            if request.get("exact_duplicate") is not True:
                errors.append(f"trace request {index} did not pass exact duplication")
            for copy_name in ("primary", "duplicate"):
                copy_value = request.get(copy_name)
                if not isinstance(copy_value, Mapping):
                    errors.append(f"trace request {index}.{copy_name} must be an object")
                    continue
                if set(copy_value) != {"trace", "final_action_sha256", "policy_timing"}:
                    errors.append(
                        f"trace request {index}.{copy_name} has unexpected or missing fields"
                    )
                errors.extend(
                    _validate_trace_record(
                        copy_value.get("trace"),
                        name=f"pairing.trace_requests[{index}].{copy_name}.trace",
                        expected_step=expected_step,
                        expected_time=expected_time,
                    )
                )
                trace = copy_value.get("trace")
                if isinstance(trace, Mapping) and _is_sha256(trace.get("sha256")):
                    trace_hashes.append(str(trace["sha256"]))
                if copy_value.get("final_action_sha256") != final_action_hash:
                    errors.append(f"trace request {index}.{copy_name} final action is not bound")
            primary = request.get("primary")
            duplicate = request.get("duplicate")
            if isinstance(primary, Mapping) and isinstance(duplicate, Mapping):
                first_trace = primary.get("trace")
                second_trace = duplicate.get("trace")
                if not isinstance(first_trace, Mapping) or not isinstance(second_trace, Mapping) or first_trace.get("sha256") != second_trace.get("sha256"):
                    errors.append(f"trace request {index} primary/duplicate traces differ")
                if primary.get("final_action_sha256") != duplicate.get("final_action_sha256"):
                    errors.append(f"trace request {index} primary/duplicate actions differ")
        if pairing.get("all_final_actions_exact") is not True:
            errors.append("all eager calls must return the exact same final action")

    simulator = value.get("simulator")
    rollout_hashes: List[str] = []
    raw_rollouts: List[Mapping[str, Any]] = []
    rollout_set_sha: Any = None
    if not isinstance(simulator, Mapping):
        errors.append("simulator must be an object")
    else:
        if set(simulator) != {
            "measurement",
            "repeats",
            "exact_replay",
            "rollout_set_sha256",
        }:
            errors.append("simulator has unexpected or missing fields")
        if simulator.get("measurement") != "inclusive branch plus every hidden MuJoCo substep":
            errors.append("simulator measurement semantics differ from the registered metric")
        repeats = simulator.get("repeats")
        if not isinstance(repeats, list) or len(repeats) != 2:
            errors.append("simulator.repeats must contain two raw rollouts")
            repeats = []
        for index, repeat in enumerate(repeats):
            if not isinstance(repeat, Mapping) or not isinstance(repeat.get("rollout"), Mapping):
                errors.append(f"simulator repeat {index} must contain a rollout")
                continue
            rollout = repeat["rollout"]
            if repeat.get("repeat_index") != index:
                errors.append(f"simulator repeat {index} has the wrong repeat_index")
            if repeat.get("rollout_sha256") != content_hash(dict(rollout)):
                errors.append(f"simulator repeat {index} rollout hash differs")
            if repeat.get("final_action_sha256") != final_action_hash:
                errors.append(f"simulator repeat {index} action binding differs")
            if repeat.get("branch_geometry_sha256") != geometry_sha:
                errors.append(f"simulator repeat {index} geometry binding differs")
            if rollout.get("measurement_samples") != EXPECTED_MEASUREMENT_SAMPLES:
                errors.append(f"simulator repeat {index} must include branch plus 125 substeps")
            if not isinstance(rollout.get("reach"), Mapping):
                errors.append(f"simulator repeat {index} has no retained reach outcome")
            errors.extend(
                f"simulator repeat {index} raw evidence: {error}"
                for error in _trial_evidence_errors(rollout)
            )
            try:
                reconstructed_geometry = _rollout_branch_geometry(
                    rollout, maximum_obbs=MAXIMUM_OBBS
                )
            except (TypeError, ValueError, AttributeError, OverflowError) as error:
                errors.append(
                    f"simulator repeat {index} branch geometry is invalid: {error}"
                )
            else:
                if reconstructed_geometry.get("sha256") != geometry_sha:
                    errors.append(
                        f"simulator repeat {index} branch geometry differs from feature geometry"
                    )
            rollout_hashes.append(str(repeat.get("rollout_sha256")))
            raw_rollouts.append(rollout)
        rollout_set_sha = content_hash(rollout_hashes)
        if simulator.get("rollout_set_sha256") != rollout_set_sha:
            errors.append("simulator.rollout_set_sha256 differs from raw repeats")
        if simulator.get("exact_replay") is not True or (
            len(rollout_hashes) == 2 and rollout_hashes[0] != rollout_hashes[1]
        ):
            errors.append("simulator repeats are not exact")

    label = value.get("label")
    label_sha: Any = None
    reconstructed_label: Optional[Mapping[str, Any]] = None
    if not isinstance(label, Mapping):
        errors.append("label must be an object")
    else:
        label_payload = {key: item for key, item in label.items() if key != "sha256"}
        label_sha = label.get("sha256")
        if label_sha != content_hash(label_payload):
            errors.append("label.sha256 does not match its payload")
        if label.get("final_action_sha256") != final_action_hash or label.get("rollout_set_sha256") != rollout_set_sha:
            errors.append("label is not bound to the final action and rollout set")
        if len(raw_rollouts) == 2:
            try:
                clearances = [float(item["clearance_m"]) for item in raw_rollouts]
                claimed_d_sim = float(label.get("D_sim_m"))
            except (KeyError, TypeError, ValueError, OverflowError):
                errors.append("label/raw repeat clearances must be numeric")
            else:
                if not all(math.isfinite(item) for item in clearances) or not math.isfinite(
                    claimed_d_sim
                ):
                    errors.append("label/raw repeat clearances must be finite")
                elif claimed_d_sim != min(clearances):
                    errors.append("label D_sim is not the minimum raw repeat clearance")
                if label.get("repeat_clearance_m") != clearances:
                    errors.append("label repeat clearances differ from raw outcomes")
            try:
                reconstructed_label = _simulator_label(
                    raw_rollouts,
                    final_action_sha256=str(final_action_hash),
                    rollout_set_sha256=str(rollout_set_sha),
                )
            except (KeyError, TypeError, ValueError, OverflowError) as error:
                errors.append(f"raw simulator outcomes cannot reconstruct the label: {error}")
            else:
                if dict(label) != dict(reconstructed_label):
                    errors.append("label differs from the independently reconstructed raw outcomes")

    if isinstance(pairing, Mapping):
        bindings = pairing.get("trace_to_label_bindings")
        if not isinstance(bindings, list) or len(bindings) != 10:
            errors.append("pairing must bind all ten trace requests to the simulator label")
        else:
            for index, binding in enumerate(bindings):
                if not isinstance(binding, Mapping):
                    errors.append(f"trace binding {index} must be an object")
                    continue
                if index >= len(trace_hashes):
                    errors.append(f"trace binding {index} has no validated trace record")
                elif binding.get("trace_sha256") != trace_hashes[index]:
                    errors.append(f"trace binding {index} trace hash differs")
                if binding.get("final_action_sha256") != final_action_hash:
                    errors.append(f"trace binding {index} action hash differs")
                if binding.get("rollout_set_sha256") != rollout_set_sha:
                    errors.append(f"trace binding {index} rollout hash differs")
                if binding.get("simulator_label_sha256") != label_sha:
                    errors.append(f"trace binding {index} label hash differs")
                expected_request_index = index // 2
                expected_copy = "primary" if index % 2 == 0 else "duplicate"
                if (
                    binding.get("request_index") != expected_request_index
                    or binding.get("copy") != expected_copy
                    or binding.get("step_index") != TRACE_STEPS[expected_request_index]
                ):
                    errors.append(f"trace binding {index} ordering metadata differs")

    outcome = value.get("outcome")
    if not isinstance(outcome, Mapping) or outcome.get("retained_regardless_of_safety_or_progress") is not True:
        errors.append("R04A must retain every safety/progress outcome")
    elif (
        outcome.get("phase") != "pregrasp_reach"
        or not isinstance(outcome.get("phase_valid"), bool)
        or not isinstance(outcome.get("initial_target_contact"), bool)
        or not isinstance(outcome.get("initial_task_success"), bool)
    ):
        errors.append("R04A outcome must retain the descriptive phase audit")
    if isinstance(outcome, Mapping) and reconstructed_label is not None:
        expected_outcome_values = {
            "D_sim_m": reconstructed_label["D_sim_m"],
            "contact": reconstructed_label["contact_any"],
            "reach_progress_m": reconstructed_label["reach_progress_m"],
            "maximum_target_displacement_m": reconstructed_label[
                "maximum_target_displacement_m"
            ],
            "maximum_active_obstacle_displacement_m": reconstructed_label[
                "maximum_active_obstacle_displacement_m"
            ],
        }
        if any(outcome.get(key) != expected for key, expected in expected_outcome_values.items()):
            errors.append("outcome differs from the independently reconstructed label")
        expected_checks = {
            "clearance_at_registered_margin": bool(
                reconstructed_label["D_sim_m"] >= 0.005
                and not reconstructed_label["contact_any"]
            ),
            "progress_at_r00_p_min": bool(
                reconstructed_label["reach_progress_m"] >= 0.029897349105658888
            ),
            "target_stationary": bool(
                reconstructed_label["maximum_target_displacement_m"] <= 0.001
            ),
            "active_obstacle_stationary": bool(
                reconstructed_label["maximum_active_obstacle_displacement_m"]
                <= 0.001
            ),
        }
        if outcome.get("descriptive_checks") != expected_checks:
            errors.append("outcome descriptive checks differ from frozen thresholds")
        if source_case is not None:
            expected_phase_valid = bool(
                source_case.get("task_suite") == "safelibero_spatial"
                and source_case.get("safety_level") == "II"
                and source_case.get("task_index") == 0
                and outcome.get("initial_target_contact") is False
                and outcome.get("initial_task_success") is False
            )
            if outcome.get("phase_valid") is not expected_phase_valid:
                errors.append("outcome.phase_valid differs from the retained phase audit")
    if value.get("usage_restriction") != "apparatus_only_never_train_calibrate_validate_test_or_claim":
        errors.append("R04A usage restriction is missing or weakened")
    return errors


def valid_r04_label_completion(
    path: str | Path,
    *,
    case_id: Optional[str] = None,
    run_id: Optional[str] = None,
    config_hash: Optional[str] = None,
    input_manifest_sha256: Optional[str] = None,
    config_file_sha256: Optional[str] = None,
    checkpoint_sha256: Optional[str] = None,
    git_commit: Optional[str] = None,
) -> bool:
    try:
        value = load_json(path)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return False
    if not isinstance(value, Mapping):
        return False
    try:
        validation_errors = validate_r04_label_result(value)
    except (AttributeError, IndexError, KeyError, TypeError, ValueError, OverflowError):
        # Resume must never trust or choke on a malformed prior artifact.  Any
        # reconstruction failure means rerun and atomically replace it.
        return False
    if validation_errors:
        return False
    comparisons = {
        "case_id": case_id,
        "run_id": run_id,
        "config_hash": config_hash,
    }
    if any(expected is not None and value.get(key) != expected for key, expected in comparisons.items()):
        return False
    source = value.get("source_evidence", {})
    provenance = value.get("provenance", {})
    return bool(
        (input_manifest_sha256 is None or source.get("input_manifest_sha256") == input_manifest_sha256)
        and (config_file_sha256 is None or source.get("config_file_sha256") == config_file_sha256)
        and (checkpoint_sha256 is None or source.get("checkpoint_sha256") == checkpoint_sha256)
        and (git_commit is None or provenance.get("git_commit") == git_commit)
    )


def run_r04_label_case(
    case: Mapping[str, Any],
    config: R04LabelConfig,
    *,
    repo_root: str | Path,
    input_manifest_sha256: str,
    client: Any = None,
    environment: Optional[SafeLiberoCase] = None,
) -> Tuple[Path, str]:
    """Capture one R04A trace/continuation label and atomically finalize it."""

    if input_manifest_sha256 != config.declared_manifest_sha256:
        raise ValueError("runtime manifest hash differs from the R04A frozen config")
    allocation = _allocation_provenance()
    oracle = config.oracle
    normalized_config = _normalized_config(config)
    config_hash = content_hash(normalized_config)
    git_commit, git_dirty = _git_state(Path(repo_root).resolve())
    if git_dirty:
        raise RuntimeError("R04A allocation worktree must be clean before evidence capture")
    reviewed_git_commit = os.environ.get("EXPECTED_GIT_COMMIT", "").strip()
    if not _is_git_commit(reviewed_git_commit) or git_commit != reviewed_git_commit:
        raise RuntimeError("R04A source commit differs from the reviewed submission commit")
    output = (
        Path(oracle.output_root)
        / oracle.run_id
        / str(case["case_id"])
        / "r04-label-contract.json"
    )
    if valid_r04_label_completion(
        output,
        case_id=str(case["case_id"]),
        run_id=oracle.run_id,
        config_hash=config_hash,
        input_manifest_sha256=input_manifest_sha256,
        config_file_sha256=config.config_file_sha256,
        checkpoint_sha256=oracle.checkpoint_sha256,
        git_commit=git_commit,
    ):
        return output, "skipped_valid_completion"
    noise = np.random.default_rng(int(case["policy_seed"])).normal(
        size=MODEL_ACTION_SHAPE
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
            raise RuntimeError("R04A failed to resolve the active obstacle")
        initial_snapshot = capture_reach_snapshot(
            environment, config.target_name, obstacle_name
        )
        initial_target_contact = _target_contact_at_branch(
            environment, config.target_name
        )
        initial_task_success = bool(environment.env.check_success())
        phase_valid = bool(
            str(case["task_suite"]) == "safelibero_spatial"
            and str(case["safety_level"]) == "II"
            and int(case["task_index"]) == 0
            and not initial_target_contact
            and not initial_task_success
        )
        geometry = _capture_branch_geometry(
            environment, maximum_obbs=config.maximum_obbs
        )
        policy_input = policy_observation(
            initial_observation, environment.prompt, oracle.resize_size
        )
        observation_identity = _observation_identity(policy_input)

        trace_requests: List[Mapping[str, Any]] = []
        returned_actions: List[np.ndarray] = []
        for step, expected_time in zip(config.trace_steps, config.trace_times):
            replies = [
                _infer_trace_only(
                    client,
                    policy_input,
                    noise,
                    intervention_step=step,
                )
                for _ in range(config.duplicate_trace_requests)
            ]
            first_actions, first_trace = _validated_reply(
                replies[0], expected_step=step, expected_time=expected_time
            )
            second_actions, second_trace = _validated_reply(
                replies[1], expected_step=step, expected_time=expected_time
            )
            exact_duplicate = bool(
                np.array_equal(first_actions, second_actions)
                and first_trace["sha256"] == second_trace["sha256"]
            )
            if not exact_duplicate:
                raise RuntimeError(f"R04A eager trace/action replay differs at step {step}")
            returned_actions.extend((first_actions, second_actions))
            trace_requests.append(
                {
                    "step_index": step,
                    "expected_time": expected_time,
                    "intervention_mode": "none",
                    "correction_supplied": False,
                    "exact_duplicate": True,
                    "primary": {
                        "trace": first_trace,
                        "final_action_sha256": _array_hash(first_actions),
                        "policy_timing": _json_compatible(
                            replies[0].get("policy_timing", {})
                        ),
                    },
                    "duplicate": {
                        "trace": second_trace,
                        "final_action_sha256": _array_hash(second_actions),
                        "policy_timing": _json_compatible(
                            replies[1].get("policy_timing", {})
                        ),
                    },
                }
            )
        final_actions = returned_actions[0]
        if not all(np.array_equal(final_actions, item) for item in returned_actions[1:]):
            raise RuntimeError("R04A trace steps changed the final frozen-policy action")
        final_action_record = _array_record(final_actions)
        final_action_sha = str(final_action_record["sha256"])

        raw_rollouts: List[Mapping[str, Any]] = []
        repeats: List[Mapping[str, Any]] = []
        executed_actions = np.asarray(final_actions[:EXECUTED_REACH_ACTIONS, :7])
        for repeat_index in range(config.simulator_repeats):
            rollout, reach = annotate_reach_rollout(
                environment,
                executed_actions,
                target_name=config.target_name,
                obstacle_name=obstacle_name,
                initial_snapshot=initial_snapshot,
            )
            canonical = _canonical_rollout({**rollout, "reach": reach})
            if canonical.get("measurement_samples") != EXPECTED_MEASUREMENT_SAMPLES:
                raise RuntimeError("R04A five-action rollout did not measure branch plus 125 substeps")
            rollout_geometry = _rollout_branch_geometry(
                canonical, maximum_obbs=config.maximum_obbs
            )
            if rollout_geometry["sha256"] != geometry["sha256"]:
                raise RuntimeError(
                    "R04A rollout branch geometry differs from the captured probe geometry"
                )
            raw_rollouts.append(canonical)
            repeats.append(
                {
                    "repeat_index": repeat_index,
                    "final_action_sha256": final_action_sha,
                    "branch_geometry_sha256": geometry["sha256"],
                    "rollout_sha256": content_hash(canonical),
                    "rollout": canonical,
                }
            )
        exact_replay = bool(
            _exact_rollout_replay(raw_rollouts[0], raw_rollouts[1])
            and content_hash(raw_rollouts[0]) == content_hash(raw_rollouts[1])
        )
        if not exact_replay:
            raise RuntimeError("R04A repeated frozen-continuation simulator rollout is not exact")
        rollout_hashes = [str(item["rollout_sha256"]) for item in repeats]
        rollout_set_sha = content_hash(rollout_hashes)
        label = _simulator_label(
            raw_rollouts,
            final_action_sha256=final_action_sha,
            rollout_set_sha256=rollout_set_sha,
        )

        bindings: List[Mapping[str, Any]] = []
        for request_index, request in enumerate(trace_requests):
            for copy_name in ("primary", "duplicate"):
                trace = request[copy_name]["trace"]
                bindings.append(
                    {
                        "request_index": request_index,
                        "copy": copy_name,
                        "step_index": request["step_index"],
                        "trace_sha256": trace["sha256"],
                        "final_action_sha256": final_action_sha,
                        "rollout_set_sha256": rollout_set_sha,
                        "simulator_label_sha256": label["sha256"],
                    }
                )

        source_evidence = {
            "case_record": _json_compatible(dict(case)),
            "case_record_sha256": content_hash(dict(case)),
            "input_manifest_sha256": input_manifest_sha256,
            "config_file_sha256": config.config_file_sha256,
            "checkpoint_id": oracle.checkpoint_id,
            "checkpoint_sha256": oracle.checkpoint_sha256,
            **_json_compatible(dict(config.source_evidence)),
        }
        provenance = {
            "evidence_tier": "real_safelibero_r04a_label_contract_apparatus_only",
            "git_commit": git_commit,
            "reviewed_git_commit": reviewed_git_commit,
            "git_dirty": git_dirty,
            "baseline_commit": BASELINE_COMMIT,
            "python_version": platform.python_version(),
            "host": socket.gethostname(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "task_suite": case["task_suite"],
            "safety_level": case["safety_level"],
            "task_index": case["task_index"],
            "episode_index": case["episode_index"],
            "environment_seed": case["environment_seed"],
            "policy_seed": case["policy_seed"],
            "random_control_seed": case["random_control_seed"],
            "group_id": case["group_id"],
            "noise_sha256": _array_hash(noise),
            "sampler_steps": oracle.sampler_steps,
            "trace_steps": list(config.trace_steps),
            "model_action_horizon": oracle.action_horizon,
            "model_action_dimension": oracle.action_dim,
            "executed_action_horizon": oracle.executed_prefix,
            "duplicate_trace_requests": config.duplicate_trace_requests,
            "simulator_repeats": config.simulator_repeats,
            "measurement_samples_per_trial": EXPECTED_MEASUREMENT_SAMPLES,
            "eef_radius_m": oracle.eef_radius_m,
            "distance_limit_m": oracle.distance_limit_m,
            "simulator_safety_margin_m": oracle.safety_margin_m,
            "minimum_progress_m": config.minimum_progress_m,
            "action_frame": "world-frame OSC translation delta",
            "geometry_frame": "world",
            "phase": "pregrasp_reach",
            "intervention_mode": "none",
            "optimizer_status": "not_run_not_applicable",
            "D_opt_status": "not_run_not_applicable",
            "D_sim_status": "measured_twice_from_raw_substeps",
            "intermediate_trace_space": "checkpoint-normalized model action coordinates (10x32)",
            "intermediate_trace_semantics": "x_t, v_base, and predicted_clean only; final normalized x_0 is not exposed",
            "final_action_space": "inverse-transformed physical LIBERO controller action (10x7)",
            "normalization_asset_sha256": config.source_evidence["sampler_parity"][
                "normalization_asset_sha256"
            ],
            "no_training": True,
            "no_guidance": True,
            "no_correction": True,
            **allocation,
        }
        clearance = float(label["D_sim_m"])
        progress = float(label["reach_progress_m"])
        target_motion = float(label["maximum_target_displacement_m"])
        obstacle_motion = float(label["maximum_active_obstacle_displacement_m"])
        outcome = {
            "retained_regardless_of_safety_or_progress": True,
            "phase": "pregrasp_reach",
            "phase_valid": phase_valid,
            "initial_target_contact": initial_target_contact,
            "initial_task_success": initial_task_success,
            "D_sim_m": clearance,
            "contact": bool(label["contact_any"]),
            "reach_progress_m": progress,
            "maximum_target_displacement_m": target_motion,
            "maximum_active_obstacle_displacement_m": obstacle_motion,
            "descriptive_checks": {
                "clearance_at_registered_margin": bool(
                    clearance >= oracle.safety_margin_m and not label["contact_any"]
                ),
                "progress_at_r00_p_min": bool(progress >= config.minimum_progress_m),
                "target_stationary": bool(
                    target_motion <= config.maximum_target_displacement_m
                ),
                "active_obstacle_stationary": bool(
                    obstacle_motion <= config.maximum_obstacle_displacement_m
                ),
            },
        }
        result = {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": ARTIFACT_TYPE,
            "gate": GATE,
            "case_id": case["case_id"],
            "run_id": oracle.run_id,
            "status": FINAL_STATUS,
            "apparatus_scope": config.apparatus_scope,
            "config_hash": config_hash,
            "source_evidence": source_evidence,
            "provenance": provenance,
            "pairing": {
                "fixed_observation": observation_identity,
                "noise": _array_record(noise),
                "trace_requests": trace_requests,
                "all_final_actions_exact": True,
                "final_full_physical_actions": final_action_record,
                "executed_action_prefix_sha256": _array_hash(executed_actions),
                "trace_to_label_bindings": bindings,
            },
            "geometry": geometry,
            "simulator": {
                "measurement": "inclusive branch plus every hidden MuJoCo substep",
                "repeats": repeats,
                "exact_replay": True,
                "rollout_set_sha256": rollout_set_sha,
            },
            "label": label,
            "outcome": outcome,
            "usage_restriction": config.reuse_role,
        }
        errors = validate_r04_label_result(result)
        if errors:
            raise RuntimeError(
                "refusing invalid R04A final artifact: " + "; ".join(errors)
            )
        atomic_write_json(output, result)
        return output, FINAL_STATUS
    finally:
        if owns_environment:
            environment.close()
