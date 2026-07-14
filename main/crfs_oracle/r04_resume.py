"""R04B exact saved-latent resume/edit parity apparatus.

This is a deliberately non-scientific, one-case transport smoke.  It performs
one simulator reset plus the registered 20 dummy settling controls solely to
obtain the fixed observation.  It then captures real eager sampler states,
resumes the frozen sampler from those exact float32 states, and validates
byte-exact zero-edit feature and final-action parity.  One ordinary compiled
default call before and after that sequence provides a current-commit baseline
regression under the frozen R02 physical tolerances.  It executes no
policy-generated action and no efficacy rollout, trains no probe, applies no
guidance, and authorizes no research claim.
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

from .r04_baseline_reference import (
    COMPILED_CALL_PLACEMENT,
    COMPILED_COMPARISON_CONTRACT,
    COMPILED_DEFAULT_CALLS,
    COMPILED_REQUEST_CONTRACT,
    MODEL_SPACE_COMPARISON_STATUS,
    R02_PHYSICAL_LIMITS,
    R02_TOLERANCE_SOURCE,
    R02_TOLERANCE_SOURCE_SHA256,
    build_baseline_reference,
    expected_r04a_golden_contract,
    infer_compiled_default,
    validate_baseline_reference,
    validate_r04a_golden_source,
    validated_compiled_reply,
)
from .r04_labels import (
    BASELINE_COMMIT,
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_PARITY_SHA256,
    EXPECTED_R00_SUMMARY_SHA256,
    EXPECTED_R03_SUMMARY_SHA256,
    EXPECTED_SMOKE_CASE_ID,
    EXPECTED_SMOKE_CASE_RECORD_SHA256,
    EXPECTED_SMOKE_GROUP_ID,
    MODEL_ACTION_SHAPE,
    PHYSICAL_ACTION_SHAPE,
    _array_from_record,
    _array_record,
    _is_git_commit,
    _is_sha256,
    _json_compatible,
    _observation_identity,
    _trace_record,
    _validate_observation_identity,
    _validate_trace_record,
    validate_r04_label_result,
)
from .runner import (
    OracleConfig,
    SafeLiberoCase,
    _array_hash,
    _git_state,
    policy_observation,
)


SCHEMA_VERSION = "1.0"
ARTIFACT_TYPE = "r04_saved_latent_resume_parity"
GATE = "R04B"
FINAL_STATUS = "completed"
TRACE_STEPS = (1, 2, 3, 4, 5)
SOURCE_CALLS_PER_STEP = 2
ZERO_RESUME_CALLS_PER_STEP = 2
NONZERO_RESUME_CALLS_PER_STEP = 2
EAGER_RESUME_POLICY_CALLS = len(TRACE_STEPS) * (
    SOURCE_CALLS_PER_STEP
    + ZERO_RESUME_CALLS_PER_STEP
    + NONZERO_RESUME_CALLS_PER_STEP
)
TOTAL_POLICY_CALLS = EAGER_RESUME_POLICY_CALLS + COMPILED_DEFAULT_CALLS
NONZERO_EDIT_INDEX = (0, 0)
NONZERO_EDIT_VALUE = np.float32(0.03125)
EXPECTED_CONFIG_FILE_SHA256 = (
    "a0edb7edac86d2abd887218002d14e28e072cd472d631797d18d540f5140be33"
)
EXPECTED_R04A_VALIDATION_SUMMARY_SHA256 = (
    "cf821650c48d30f6ebf2bbf7afe06b21938e0b137aa5acf1921f571e409e05ea"
)
EXPECTED_R04A_RAW_ARTIFACT_SHA256 = (
    "b820793ec42a5228c858e297d13806d8ae7f02e5cc7a765769316473d795f285"
)
APPARATUS_SCOPE = "exact_saved_latent_resume_edit_parity_smoke_only"
USAGE_RESTRICTION = "apparatus_only_never_train_calibrate_validate_test_or_claim"
EVIDENCE_TIER = "real_safelibero_r04b_resume_edit_apparatus_only"
TIME_CONTRACT = (
    "captured_float32_source_reply_is_authoritative_no_schedule_reconstruction"
)
NOISE_CONTRACT = (
    "same_explicit_fixed_policy_noise_carried_by_every_source_and_resume_request"
)
NONZERO_EDIT_ROLE = "implementation_sentinel_only_never_support_or_dose_evidence"
EXACT_EQUALITY_CONTRACT = "dtype_shape_and_c_contiguous_array_bytes_sha256"
SIMULATOR_SETUP = "one_reset_plus_20_dummy_settle_control_steps"
ALLOCATION_FIELDS = (
    "slurm_job_id",
    "slurm_array_job_id",
    "slurm_array_task_id",
    "partition",
    "device",
)


@dataclass(frozen=True)
class R04ResumeConfig:
    """Validated opt-in configuration for one R04B apparatus case."""

    oracle: OracleConfig
    enabled: bool
    apparatus_scope: str
    reuse_role: str
    smoke_case_index: int
    trace_steps: Tuple[int, ...]
    source_calls_per_step: int
    zero_resume_calls_per_step: int
    nonzero_resume_calls_per_step: int
    nonzero_edit_index: Tuple[int, int]
    nonzero_edit_value: float
    declared_manifest_sha256: str
    config_file_sha256: str
    source_evidence: Mapping[str, Any]


def _resolve(path_value: str, root: str | Path) -> Path:
    path = Path(path_value).expanduser()
    return path.resolve() if path.is_absolute() else (Path(root).resolve() / path).resolve()


def _load_bound_json(
    settings: Mapping[str, Any],
    *,
    path_key: str,
    hash_key: str,
    repo_root: str | Path,
) -> Tuple[Path, Mapping[str, Any], str]:
    path_value = settings.get(path_key)
    expected_sha = settings.get(hash_key)
    if not isinstance(path_value, str) or not path_value:
        raise ValueError(f"r04b.{path_key} must be a non-empty path")
    if not _is_sha256(expected_sha):
        raise ValueError(f"r04b.{hash_key} must be a lowercase SHA-256 digest")
    path = _resolve(path_value, repo_root)
    actual_sha = file_sha256(path)
    if actual_sha != expected_sha:
        raise ValueError(
            f"R04B source hash mismatch for {path_key}: expected {expected_sha}, got {actual_sha}"
        )
    value = load_json(path)
    if not isinstance(value, Mapping):
        raise ValueError(f"R04B source {path_key} must contain a JSON object")
    return path, value, actual_sha


def _validated_sources(
    settings: Mapping[str, Any],
    *,
    repo_root: str | Path,
    checkpoint_sha256: str,
) -> Mapping[str, Any]:
    """Validate every predecessor rather than trusting top-level pass claims."""

    r00_path, r00, r00_sha = _load_bound_json(
        settings,
        path_key="r00_summary_artifact",
        hash_key="r00_summary_sha256",
        repo_root=repo_root,
    )
    if not (
        r00.get("schema_version") == "1.0"
        and r00.get("gate") == "R00"
        and r00.get("status") == "passed"
    ):
        raise ValueError("R04B requires the passing schema-version 1.0 R00 summary")
    calibration = r00.get("calibration")
    if not isinstance(calibration, Mapping):
        raise ValueError("R04B R00 predecessor has no calibration object")

    r03_path, r03, r03_sha = _load_bound_json(
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
        raise ValueError("R04B requires the passing R03 predecessor")

    parity_path, parity, parity_sha = _load_bound_json(
        settings,
        path_key="sampler_parity_artifact",
        hash_key="sampler_parity_sha256",
        repo_root=repo_root,
    )
    from run_sampler_parity import validate_parity_artifact

    parity_errors = validate_parity_artifact(parity)
    if parity_errors:
        raise ValueError(
            "R04B sampler parity artifact is invalid: " + "; ".join(parity_errors)
        )
    converted = parity.get("checkpoints", {}).get("converted_pytorch", {})
    if not isinstance(converted, Mapping) or converted.get("model_sha256") != checkpoint_sha256:
        raise ValueError("R04B checkpoint differs from the validated parity checkpoint")
    normalization_sha = converted.get("norm_stats_sha256")
    if not _is_sha256(normalization_sha):
        raise ValueError("R04B parity predecessor does not bind normalization stats")

    summary_path, summary, summary_sha = _load_bound_json(
        settings,
        path_key="r04a_validation_summary",
        hash_key="r04a_validation_summary_sha256",
        repo_root=repo_root,
    )
    summary_source = summary.get("source")
    independent = summary.get("independent_validation")
    if not (
        summary.get("schema_version") == "1.0"
        and summary.get("artifact_type") == "r04a_label_contract_validation_summary"
        and summary.get("status") == "passed_apparatus_only"
        and summary.get("case_id") == EXPECTED_SMOKE_CASE_ID
        and summary.get("usage_restriction") == USAGE_RESTRICTION
        and summary.get("scientific_claim_authorized") is False
        and summary.get("probe_training_authorized") is False
        and isinstance(summary_source, Mapping)
        and summary_source.get("artifact_sha256") == EXPECTED_R04A_RAW_ARTIFACT_SHA256
        and isinstance(independent, Mapping)
        and independent.get("validation_error_count") == 0
        and independent.get("validated_artifact_sha256")
        == EXPECTED_R04A_RAW_ARTIFACT_SHA256
    ):
        raise ValueError("R04B requires the independently validated apparatus-only R04A summary")

    raw_path, raw, raw_sha = _load_bound_json(
        settings,
        path_key="r04a_raw_artifact",
        hash_key="r04a_raw_artifact_sha256",
        repo_root=repo_root,
    )
    raw_errors = validate_r04_label_result(raw)
    if raw_errors:
        raise ValueError("R04B raw R04A predecessor is invalid: " + "; ".join(raw_errors))
    if not (
        raw.get("case_id") == EXPECTED_SMOKE_CASE_ID
        and raw.get("usage_restriction") == USAGE_RESTRICTION
        and raw.get("source_evidence", {}).get("checkpoint_sha256")
        == checkpoint_sha256
        and summary_source.get("artifact_path") == str(raw_path)
        and summary_source.get("artifact_sha256") == raw_sha
    ):
        raise ValueError("R04B R04A summary/raw predecessor binding differs")
    golden_baseline = validate_r04a_golden_source(raw)

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
            "normalization_asset_sha256": normalization_sha,
        },
        "r04a_validation_summary": {
            "path": str(summary_path),
            "sha256": summary_sha,
            "status": "passed_apparatus_only",
            "raw_artifact_sha256": raw_sha,
            "independent_validation_error_count": 0,
        },
        "r04a_raw_artifact": {
            "path": str(raw_path),
            "sha256": raw_sha,
            "status": "validated_apparatus_only",
            "config_hash": str(raw["config_hash"]),
            "git_commit": str(raw["provenance"]["git_commit"]),
            "golden_baseline": golden_baseline,
        },
    }


def r04_resume_config_from_mapping(
    value: Mapping[str, Any],
    oracle: OracleConfig,
    *,
    repo_root: str | Path,
    config_file_sha256: str,
) -> R04ResumeConfig:
    """Validate the immutable, opt-in R04B apparatus contract."""

    if config_file_sha256 != EXPECTED_CONFIG_FILE_SHA256:
        raise ValueError("R04B config file differs from the frozen apparatus contract")
    if value.get("schema_version") != "1.0" or value.get("ready_to_run") is not True:
        raise ValueError("R04B config must be schema-version 1.0 and ready_to_run")
    settings = value.get("r04b")
    if not isinstance(settings, Mapping) or settings.get("enabled") is not True:
        raise ValueError("R04B is opt-in and requires r04b.enabled=true")
    if oracle.action_horizon != 10 or oracle.action_dim != 32:
        raise ValueError("R04B requires the baseline 10x32 model action")
    if oracle.executed_prefix != 5 or oracle.sampler_steps != 10:
        raise ValueError("R04B requires five-action baseline output and ten Euler steps")

    exact_contract = {
        "apparatus_scope": APPARATUS_SCOPE,
        "reuse_role": USAGE_RESTRICTION,
        "smoke_case_index": 0,
        "source_trace_steps": list(TRACE_STEPS),
        "source_eager_calls_per_step": SOURCE_CALLS_PER_STEP,
        "zero_resume_calls_per_step": ZERO_RESUME_CALLS_PER_STEP,
        "nonzero_resume_calls_per_step": NONZERO_RESUME_CALLS_PER_STEP,
        "compiled_default_calls": COMPILED_DEFAULT_CALLS,
        "compiled_default_call_placement": COMPILED_CALL_PLACEMENT,
        "compiled_default_request_contract": COMPILED_REQUEST_CONTRACT,
        "compiled_default_comparison_contract": COMPILED_COMPARISON_CONTRACT,
        "compiled_default_physical_limits": {
            "source": R02_TOLERANCE_SOURCE,
            "source_sha256": R02_TOLERANCE_SOURCE_SHA256,
            **R02_PHYSICAL_LIMITS,
            "normalized_model_comparison_status": MODEL_SPACE_COMPARISON_STATUS,
        },
        "total_policy_calls": TOTAL_POLICY_CALLS,
        "source_intervention_mode": "none",
        "resume_intervention_mode": "latent_resume_edit",
        "latent_edit_space": "model",
        "resume_source": "primary_source_x_t_and_captured_float32_time",
        "resume_noise_contract": NOISE_CONTRACT,
        "time_contract": TIME_CONTRACT,
        "model_action_shape": list(MODEL_ACTION_SHAPE),
        "physical_action_shape": list(PHYSICAL_ACTION_SHAPE),
        "source_duplicate_requirement": "captured_trace_and_normalized_and_physical_final_arrays_exact",
        "zero_resume_requirement": "captured_time_and_latent_exact_post_edit_velocity_predicted_clean_physical_predicted_clean_and_normalized_and_physical_finals_equal_source",
        "nonzero_resume_requirement": "captured_time_and_latent_exact_direct_edit_algebra_exact_and_duplicate_continuation_exact",
        "exact_equality_contract": EXACT_EQUALITY_CONTRACT,
        "return_normalized_final": True,
        "simulator_setup": SIMULATOR_SETUP,
        "policy_generated_action_steps_executed": 0,
        "efficacy_rollouts_executed": 0,
        "simulator_efficacy_evaluated": False,
        "training": False,
        "guidance": False,
        "learned_probe": False,
        "scientific_claim_authorized": False,
    }
    mismatches = [
        key for key, expected in exact_contract.items() if settings.get(key) != expected
    ]
    if mismatches:
        raise ValueError(f"R04B config contract mismatch for fields: {sorted(mismatches)}")
    expected_zero = {
        "dtype": "float32",
        "construction": "all_zeros",
        "shape": list(MODEL_ACTION_SHAPE),
    }
    expected_nonzero = {
        "dtype": "float32",
        "construction": "single_model_coordinate",
        "shape": list(MODEL_ACTION_SHAPE),
        "index": list(NONZERO_EDIT_INDEX),
        "value": float(NONZERO_EDIT_VALUE),
        "role": NONZERO_EDIT_ROLE,
    }
    if settings.get("zero_edit") != expected_zero or settings.get("nonzero_edit") != expected_nonzero:
        raise ValueError("R04B edit construction differs from the frozen contract")
    if value.get("allow_test_tuning") is not False:
        raise ValueError("R04B apparatus forbids test-time tuning")
    if value.get("manifest_sha256") != EXPECTED_MANIFEST_SHA256:
        raise ValueError("R04B config must bind the frozen immutable manifest")
    if (
        value.get("checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256
        or oracle.checkpoint_sha256 != EXPECTED_CHECKPOINT_SHA256
    ):
        raise ValueError("R04B config/CLI checkpoint differs from the frozen checkpoint")
    expected_source_hashes = {
        "r00_summary_sha256": EXPECTED_R00_SUMMARY_SHA256,
        "r03_summary_sha256": EXPECTED_R03_SUMMARY_SHA256,
        "sampler_parity_sha256": EXPECTED_PARITY_SHA256,
        "r04a_validation_summary_sha256": EXPECTED_R04A_VALIDATION_SUMMARY_SHA256,
        "r04a_raw_artifact_sha256": EXPECTED_R04A_RAW_ARTIFACT_SHA256,
    }
    changed = [
        key for key, expected in expected_source_hashes.items() if settings.get(key) != expected
    ]
    if changed:
        raise ValueError(f"R04B predecessor evidence differs: {sorted(changed)}")
    sources = _validated_sources(
        settings,
        repo_root=repo_root,
        checkpoint_sha256=oracle.checkpoint_sha256,
    )
    return R04ResumeConfig(
        oracle=oracle,
        enabled=True,
        apparatus_scope=APPARATUS_SCOPE,
        reuse_role=USAGE_RESTRICTION,
        smoke_case_index=0,
        trace_steps=TRACE_STEPS,
        source_calls_per_step=SOURCE_CALLS_PER_STEP,
        zero_resume_calls_per_step=ZERO_RESUME_CALLS_PER_STEP,
        nonzero_resume_calls_per_step=NONZERO_RESUME_CALLS_PER_STEP,
        nonzero_edit_index=NONZERO_EDIT_INDEX,
        nonzero_edit_value=float(NONZERO_EDIT_VALUE),
        declared_manifest_sha256=EXPECTED_MANIFEST_SHA256,
        config_file_sha256=config_file_sha256,
        source_evidence=sources,
    )


def _normalized_config(config: R04ResumeConfig) -> Mapping[str, Any]:
    return {
        **scientific_config(config.oracle.__dict__),
        "enabled": config.enabled,
        "apparatus_scope": config.apparatus_scope,
        "reuse_role": config.reuse_role,
        "smoke_case_index": config.smoke_case_index,
        "trace_steps": list(config.trace_steps),
        "source_calls_per_step": config.source_calls_per_step,
        "zero_resume_calls_per_step": config.zero_resume_calls_per_step,
        "nonzero_resume_calls_per_step": config.nonzero_resume_calls_per_step,
        "compiled_default_calls": COMPILED_DEFAULT_CALLS,
        "compiled_default_call_placement": COMPILED_CALL_PLACEMENT,
        "compiled_default_request_contract": COMPILED_REQUEST_CONTRACT,
        "compiled_default_comparison_contract": COMPILED_COMPARISON_CONTRACT,
        "compiled_default_physical_limits": {
            "source": R02_TOLERANCE_SOURCE,
            "source_sha256": R02_TOLERANCE_SOURCE_SHA256,
            **R02_PHYSICAL_LIMITS,
            "normalized_model_comparison_status": MODEL_SPACE_COMPARISON_STATUS,
        },
        "total_policy_calls": TOTAL_POLICY_CALLS,
        "nonzero_edit_index": list(config.nonzero_edit_index),
        "nonzero_edit_value": config.nonzero_edit_value,
        "declared_manifest_sha256": config.declared_manifest_sha256,
        "config_file_sha256": config.config_file_sha256,
        "source_hashes": {
            key: item["sha256"] for key, item in config.source_evidence.items()
        },
        "source_intervention_mode": "none",
        "resume_intervention_mode": "latent_resume_edit",
        "latent_edit_space": "model",
        "nonzero_edit_role": NONZERO_EDIT_ROLE,
        "exact_equality_contract": EXACT_EQUALITY_CONTRACT,
        "return_normalized_final": True,
        "simulator_setup": SIMULATOR_SETUP,
        "policy_generated_action_steps_executed": 0,
        "efficacy_rollouts_executed": 0,
        "simulator_efficacy_evaluated": False,
        "training": False,
        "guidance": False,
        "learned_probe": False,
        "scientific_claim_authorized": False,
    }


def _nonzero_edit() -> np.ndarray:
    edit = np.zeros(MODEL_ACTION_SHAPE, dtype=np.float32)
    edit[NONZERO_EDIT_INDEX] = NONZERO_EDIT_VALUE
    return edit


def _arrays_byte_exact(left: Any, right: Any) -> bool:
    """Compare dtype, shape, and canonical contiguous bytes, including -0.0."""

    try:
        left_array = np.ascontiguousarray(np.asarray(left))
        right_array = np.ascontiguousarray(np.asarray(right))
    except (TypeError, ValueError, OverflowError):
        return False
    return bool(
        left_array.dtype == right_array.dtype
        and left_array.shape == right_array.shape
        and _array_hash(left_array) == _array_hash(right_array)
    )


def _array_byte_identity(value: Any) -> Tuple[str, Tuple[int, ...], str]:
    array = np.ascontiguousarray(np.asarray(value))
    return str(array.dtype), tuple(array.shape), _array_hash(array)


def _array_records_byte_exact(left: Any, right: Any) -> bool:
    """Compare validated array records by their byte-framed identity."""

    return bool(
        isinstance(left, Mapping)
        and isinstance(right, Mapping)
        and left.get("dtype") == right.get("dtype")
        and left.get("shape") == right.get("shape")
        and left.get("sha256") == right.get("sha256")
    )


def _trace_records_byte_exact(left: Any, right: Any) -> bool:
    if not isinstance(left, Mapping) or not isinstance(right, Mapping):
        return False
    left_leaves = left.get("leaves")
    right_leaves = right.get("leaves")
    if not isinstance(left_leaves, Mapping) or not isinstance(right_leaves, Mapping):
        return False
    return bool(
        set(left_leaves) == set(right_leaves)
        and all(
            _array_records_byte_exact(left_leaves[key], right_leaves[key])
            for key in left_leaves
        )
    )


def _resume_input_record(
    resume_latent: np.ndarray,
    resume_time: np.ndarray,
    latent_edit: np.ndarray,
    noise_sha256: str,
) -> Mapping[str, Any]:
    payload = {
        "resume_latent": _array_record(resume_latent),
        "resume_time": _array_record(resume_time),
        "latent_edit": _array_record(latent_edit),
        "latent_edit_space": "model",
        "noise_sha256": noise_sha256,
    }
    return {**payload, "sha256": content_hash(payload)}


def _resume_trace_record(trace: Mapping[str, Any]) -> Mapping[str, Any]:
    required = (
        "step_index",
        "time",
        "x_t_pre_edit",
        "latent_edit",
        "x_t_post_edit",
        "v_post_edit",
        "predicted_clean_post_edit",
        "predicted_clean_post_edit_physical",
        "final_normalized",
    )
    missing = set(required) - set(trace)
    if missing:
        raise RuntimeError(f"R04B resume trace missing fields: {sorted(missing)}")
    leaves = {name: _array_record(trace[name]) for name in required}
    return {
        "primary_feature_space": "normalized_model_coordinates",
        "physical_feature_role": "inverse_transformed_audit_only",
        "leaves": leaves,
        "sha256": content_hash(leaves),
    }


def _validate_resume_trace_record(
    value: Any,
    *,
    name: str,
    expected_step: int,
    expected_time: np.ndarray,
    expected_latent: np.ndarray,
    expected_edit: np.ndarray,
) -> List[str]:
    errors: List[str] = []
    if not isinstance(value, Mapping):
        return [f"{name} must be a resume trace record"]
    expected_fields = {
        "primary_feature_space",
        "physical_feature_role",
        "leaves",
        "sha256",
    }
    if set(value) != expected_fields:
        errors.append(f"{name} has unexpected or missing trace fields")
    if value.get("primary_feature_space") != "normalized_model_coordinates":
        errors.append(f"{name} is not in normalized model coordinates")
    if value.get("physical_feature_role") != "inverse_transformed_audit_only":
        errors.append(f"{name} physical feature is not audit-only")
    leaves = value.get("leaves")
    if not isinstance(leaves, Mapping):
        return errors + [f"{name}.leaves must be an object"]
    shapes = {
        "step_index": (1,),
        "time": (1,),
        "x_t_pre_edit": MODEL_ACTION_SHAPE,
        "latent_edit": MODEL_ACTION_SHAPE,
        "x_t_post_edit": MODEL_ACTION_SHAPE,
        "v_post_edit": MODEL_ACTION_SHAPE,
        "predicted_clean_post_edit": MODEL_ACTION_SHAPE,
        "predicted_clean_post_edit_physical": PHYSICAL_ACTION_SHAPE,
        "final_normalized": MODEL_ACTION_SHAPE,
    }
    if set(leaves) != set(shapes):
        errors.append(f"{name}.leaves has unexpected or missing fields")
    arrays: Dict[str, np.ndarray] = {}
    for key, shape in shapes.items():
        array, item_errors = _array_from_record(
            leaves.get(key), name=f"{name}.leaves.{key}", shape=shape
        )
        errors.extend(item_errors)
        if array is not None:
            arrays[key] = array
    if value.get("sha256") != content_hash(dict(leaves)):
        errors.append(f"{name}.sha256 does not match its leaves")
    step = arrays.get("step_index")
    if step is not None and (
        step.dtype.kind not in "iu" or int(step.reshape(-1)[0]) != expected_step
    ):
        errors.append(f"{name} has the wrong sampler step")
    time = arrays.get("time")
    if time is not None:
        if time.dtype != np.dtype("float32"):
            errors.append(f"{name}.time must retain captured float32 dtype")
        if not _arrays_byte_exact(
            time, np.asarray(expected_time, dtype=np.float32).reshape(1)
        ):
            errors.append(f"{name}.time differs from the exact captured source time")
    for key in (
        "x_t_pre_edit",
        "latent_edit",
        "x_t_post_edit",
        "v_post_edit",
        "predicted_clean_post_edit",
        "final_normalized",
    ):
        if key in arrays and arrays[key].dtype != np.dtype("float32"):
            errors.append(f"{name}.{key} must retain float32 model coordinates")
    pre = arrays.get("x_t_pre_edit")
    edit = arrays.get("latent_edit")
    post = arrays.get("x_t_post_edit")
    velocity = arrays.get("v_post_edit")
    predicted = arrays.get("predicted_clean_post_edit")
    if pre is not None and not _arrays_byte_exact(pre, expected_latent):
        errors.append(f"{name}.x_t_pre_edit differs from the exact saved latent")
    if edit is not None and not _arrays_byte_exact(edit, expected_edit):
        errors.append(f"{name}.latent_edit differs from the requested direct edit")
    if pre is not None and edit is not None and post is not None:
        registered_zero = np.zeros(MODEL_ACTION_SHAPE, dtype=np.float32)
        if _arrays_byte_exact(expected_edit, registered_zero):
            # Zero-edit parity is a true byte-preserving no-op.  Arithmetic
            # `-0.0 + +0.0` may flip the sign bit, so it is not an acceptable
            # implementation of this arm.
            reconstructed_post = np.ascontiguousarray(pre.copy())
            algebra_description = "byte-exact x_t_pre_edit for the zero edit"
        else:
            reconstructed_post = np.ascontiguousarray(pre + edit)
            algebra_description = "x_t_pre_edit + latent_edit"
        if not _arrays_byte_exact(post, reconstructed_post):
            errors.append(
                f"{name}.x_t_post_edit does not equal {algebra_description}"
            )
    if post is not None and velocity is not None and predicted is not None and time is not None:
        scalar = np.asarray(time.reshape(-1)[0], dtype=post.dtype)
        reconstructed_prediction = np.ascontiguousarray(post - scalar * velocity)
        if not _arrays_byte_exact(predicted, reconstructed_prediction):
            errors.append(
                f"{name}.predicted_clean_post_edit does not equal x_t_post_edit - time * v_post_edit"
            )
    return errors


def _infer_source(
    client: Any,
    observation: Mapping[str, Any],
    noise: np.ndarray,
    *,
    intervention_step: int,
) -> Mapping[str, Any]:
    request = copy.deepcopy(dict(observation))
    request["__crfs__"] = {
        "noise": np.array(noise, copy=True),
        "intervention_step": int(intervention_step),
        "intervention_mode": "none",
        "return_trace": True,
        "return_normalized_final": True,
    }
    reply = client.infer(request)
    if not isinstance(reply, Mapping) or "actions" not in reply:
        raise RuntimeError("R04B source reply has no physical actions")
    if not isinstance(reply.get("crfs_trace"), Mapping):
        raise RuntimeError("R04B source reply has no eager trace")
    return reply


def _infer_resume(
    client: Any,
    observation: Mapping[str, Any],
    noise: np.ndarray,
    *,
    intervention_step: int,
    resume_latent: np.ndarray,
    resume_time: np.ndarray,
    latent_edit: np.ndarray,
) -> Mapping[str, Any]:
    time_array = np.asarray(resume_time)
    if time_array.dtype != np.dtype("float32") or time_array.size != 1:
        raise ValueError("R04B resume_time transport must be one exact float32 scalar")
    scalar_time = np.asarray(time_array.reshape(-1)[0], dtype=np.float32).reshape(())
    request = copy.deepcopy(dict(observation))
    request["__crfs__"] = {
        "noise": np.array(noise, copy=True),
        "intervention_step": int(intervention_step),
        "intervention_mode": "latent_resume_edit",
        "resume_latent": np.array(resume_latent, copy=True),
        "resume_time": np.array(scalar_time, copy=True),
        "latent_edit": np.array(latent_edit, copy=True),
        "latent_edit_space": "model",
        "return_trace": True,
        "return_normalized_final": True,
    }
    reply = client.infer(request)
    if not isinstance(reply, Mapping) or "actions" not in reply:
        raise RuntimeError("R04B resume reply has no physical actions")
    if not isinstance(reply.get("crfs_trace"), Mapping):
        raise RuntimeError("R04B resume reply has no post-edit trace")
    return reply


def _validated_source_reply(
    reply: Mapping[str, Any], *, expected_step: int
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Mapping[str, np.ndarray],
    Mapping[str, Any],
]:
    actions = np.ascontiguousarray(np.asarray(reply["actions"]))
    if actions.shape != PHYSICAL_ACTION_SHAPE or not np.all(np.isfinite(actions)):
        raise RuntimeError("R04B source physical actions must be finite 10x7")
    raw_trace = reply["crfs_trace"]
    final_normalized = np.ascontiguousarray(np.asarray(raw_trace.get("final_normalized")))
    if (
        final_normalized.shape != MODEL_ACTION_SHAPE
        or final_normalized.dtype != np.dtype("float32")
        or not np.all(np.isfinite(final_normalized))
    ):
        raise RuntimeError("R04B source final_normalized must be finite float32 10x32")
    raw_time = np.asarray(raw_trace.get("time"))
    raw_latent = np.asarray(raw_trace.get("x_t"))
    if raw_time.dtype != np.dtype("float32") or raw_time.size != 1:
        raise RuntimeError("R04B source time must be an exact float32 scalar")
    if (
        raw_latent.dtype != np.dtype("float32")
        or raw_latent.shape != MODEL_ACTION_SHAPE
        or not np.all(np.isfinite(raw_latent))
    ):
        raise RuntimeError("R04B source x_t must be finite float32 10x32")
    time = np.ascontiguousarray(raw_time.reshape(1))
    features: Dict[str, np.ndarray] = {}
    for key, shape in (
        ("x_t", MODEL_ACTION_SHAPE),
        ("v_base", MODEL_ACTION_SHAPE),
        ("predicted_clean", MODEL_ACTION_SHAPE),
        ("predicted_clean_physical", PHYSICAL_ACTION_SHAPE),
    ):
        array = np.ascontiguousarray(np.asarray(raw_trace.get(key)))
        if array.shape != shape or not np.all(np.isfinite(array)):
            raise RuntimeError(f"R04B source {key} must be a finite {shape} array")
        if key != "predicted_clean_physical" and array.dtype != np.dtype("float32"):
            raise RuntimeError(f"R04B source {key} must retain float32 model coordinates")
        features[key] = array
    time_value = float(time[0])
    if not 0.0 < time_value < 1.0:
        raise RuntimeError("R04B captured source time must lie strictly inside (0, 1)")
    trace = _trace_record(raw_trace)
    errors = _validate_trace_record(
        trace,
        name=f"source_trace_step_{expected_step}",
        expected_step=expected_step,
        expected_time=time_value,
    )
    if errors:
        raise RuntimeError("invalid R04B source trace: " + "; ".join(errors))
    return actions, final_normalized, time, features, trace


def _validated_resume_reply(
    reply: Mapping[str, Any],
    *,
    expected_step: int,
    expected_time: np.ndarray,
    expected_latent: np.ndarray,
    expected_edit: np.ndarray,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    Mapping[str, np.ndarray],
    Mapping[str, Any],
]:
    actions = np.ascontiguousarray(np.asarray(reply["actions"]))
    if actions.shape != PHYSICAL_ACTION_SHAPE or not np.all(np.isfinite(actions)):
        raise RuntimeError("R04B resume physical actions must be finite 10x7")
    trace = _resume_trace_record(reply["crfs_trace"])
    errors = _validate_resume_trace_record(
        trace,
        name=f"resume_trace_step_{expected_step}",
        expected_step=expected_step,
        expected_time=expected_time,
        expected_latent=expected_latent,
        expected_edit=expected_edit,
    )
    if errors:
        raise RuntimeError("invalid R04B resume trace: " + "; ".join(errors))
    leaves = trace["leaves"]
    final_normalized, reconstruction_errors = _array_from_record(
        leaves["final_normalized"],
        name="resume.final_normalized",
        shape=MODEL_ACTION_SHAPE,
    )
    if reconstruction_errors or final_normalized is None:
        raise RuntimeError(
            "invalid R04B normalized resume final: " + "; ".join(reconstruction_errors)
        )
    features: Dict[str, np.ndarray] = {}
    for key, shape in (
        ("x_t_pre_edit", MODEL_ACTION_SHAPE),
        ("latent_edit", MODEL_ACTION_SHAPE),
        ("x_t_post_edit", MODEL_ACTION_SHAPE),
        ("v_post_edit", MODEL_ACTION_SHAPE),
        ("predicted_clean_post_edit", MODEL_ACTION_SHAPE),
        ("predicted_clean_post_edit_physical", PHYSICAL_ACTION_SHAPE),
    ):
        array, item_errors = _array_from_record(
            leaves[key], name=f"resume.{key}", shape=shape
        )
        if item_errors or array is None:
            raise RuntimeError(
                f"invalid R04B resume {key}: " + "; ".join(item_errors)
            )
        features[key] = array
    return actions, final_normalized, features, trace


def _source_copy_record(
    reply: Mapping[str, Any],
    actions: np.ndarray,
    final_normalized: np.ndarray,
    trace: Mapping[str, Any],
) -> Mapping[str, Any]:
    return {
        "trace": trace,
        "final_normalized": _array_record(final_normalized),
        "final_physical": _array_record(actions),
        "policy_timing": _json_compatible(reply.get("policy_timing", {})),
    }


def _resume_copy_record(
    reply: Mapping[str, Any],
    actions: np.ndarray,
    trace: Mapping[str, Any],
) -> Mapping[str, Any]:
    return {
        "trace": trace,
        "final_physical": _array_record(actions),
        "policy_timing": _json_compatible(reply.get("policy_timing", {})),
    }


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
            "R04B must run inside a GPU Slurm array allocation; missing "
            + ", ".join(missing)
        )
    return values


def _expected_case() -> Mapping[str, Any]:
    return {
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


def _record_array(
    record: Any,
    *,
    name: str,
    shape: Tuple[int, ...],
    errors: List[str],
) -> Optional[np.ndarray]:
    array, item_errors = _array_from_record(record, name=name, shape=shape)
    errors.extend(item_errors)
    return array


def _validate_resume_input(
    value: Any,
    *,
    name: str,
    expected_latent: np.ndarray,
    expected_time: np.ndarray,
    expected_edit: np.ndarray,
    expected_noise_sha256: str,
) -> List[str]:
    errors: List[str] = []
    if not isinstance(value, Mapping):
        return [f"{name} must be a resume request record"]
    expected_fields = {
        "resume_latent",
        "resume_time",
        "latent_edit",
        "latent_edit_space",
        "noise_sha256",
        "sha256",
    }
    if set(value) != expected_fields:
        errors.append(f"{name} has unexpected or missing fields")
    payload = {key: item for key, item in value.items() if key != "sha256"}
    if value.get("sha256") != content_hash(payload):
        errors.append(f"{name}.sha256 does not match its request payload")
    latent = _record_array(
        value.get("resume_latent"),
        name=f"{name}.resume_latent",
        shape=MODEL_ACTION_SHAPE,
        errors=errors,
    )
    time = _record_array(
        value.get("resume_time"),
        name=f"{name}.resume_time",
        shape=(1,),
        errors=errors,
    )
    edit = _record_array(
        value.get("latent_edit"),
        name=f"{name}.latent_edit",
        shape=MODEL_ACTION_SHAPE,
        errors=errors,
    )
    for key, array, expected in (
        ("resume_latent", latent, expected_latent),
        ("resume_time", time, expected_time),
        ("latent_edit", edit, expected_edit),
    ):
        if array is not None:
            if array.dtype != np.dtype("float32"):
                errors.append(f"{name}.{key} must retain float32 dtype")
            if not _arrays_byte_exact(
                array, np.asarray(expected, dtype=np.float32)
            ):
                errors.append(f"{name}.{key} differs from the exact registered input")
    if value.get("noise_sha256") != expected_noise_sha256:
        errors.append(f"{name}.noise_sha256 differs from the paired fixed noise")
    if value.get("latent_edit_space") != "model":
        errors.append(f"{name}.latent_edit_space must be explicit model space")
    return errors


def _validate_source_copy(
    value: Any,
    *,
    name: str,
    expected_step: int,
    expected_time: np.ndarray,
) -> Tuple[
    List[str],
    Optional[Mapping[str, np.ndarray]],
    Optional[np.ndarray],
    Optional[np.ndarray],
]:
    errors: List[str] = []
    if not isinstance(value, Mapping):
        return [f"{name} must be a source copy"], None, None, None
    if set(value) != {"trace", "final_normalized", "final_physical", "policy_timing"}:
        errors.append(f"{name} has unexpected or missing fields")
    errors.extend(
        _validate_trace_record(
            value.get("trace"),
            name=f"{name}.trace",
            expected_step=expected_step,
            expected_time=float(expected_time.reshape(-1)[0]),
        )
    )
    features: Dict[str, np.ndarray] = {}
    trace = value.get("trace")
    if isinstance(trace, Mapping) and isinstance(trace.get("leaves"), Mapping):
        leaves = trace["leaves"]
        time = _record_array(
            leaves.get("time"),
            name=f"{name}.trace.leaves.time",
            shape=(1,),
            errors=errors,
        )
        if time is not None and (
            time.dtype != np.dtype("float32")
            or not _arrays_byte_exact(time, expected_time)
        ):
            errors.append(f"{name} does not retain the exact captured float32 time")
        for key, shape in (
            ("x_t", MODEL_ACTION_SHAPE),
            ("v_base", MODEL_ACTION_SHAPE),
            ("predicted_clean", MODEL_ACTION_SHAPE),
            ("predicted_clean_physical", PHYSICAL_ACTION_SHAPE),
        ):
            array = _record_array(
                leaves.get(key),
                name=f"{name}.trace.leaves.{key}",
                shape=shape,
                errors=errors,
            )
            if array is not None:
                features[key] = array
                if key != "predicted_clean_physical" and array.dtype != np.dtype(
                    "float32"
                ):
                    errors.append(
                        f"{name}.trace.leaves.{key} must retain float32 model coordinates"
                    )
    normalized = _record_array(
        value.get("final_normalized"),
        name=f"{name}.final_normalized",
        shape=MODEL_ACTION_SHAPE,
        errors=errors,
    )
    physical = _record_array(
        value.get("final_physical"),
        name=f"{name}.final_physical",
        shape=PHYSICAL_ACTION_SHAPE,
        errors=errors,
    )
    if normalized is not None and normalized.dtype != np.dtype("float32"):
        errors.append(f"{name}.final_normalized must retain float32 dtype")
    if not isinstance(value.get("policy_timing"), Mapping):
        errors.append(f"{name}.policy_timing must be an object")
    return errors, features if len(features) == 4 else None, normalized, physical


def _validate_resume_copy(
    value: Any,
    *,
    name: str,
    expected_step: int,
    expected_time: np.ndarray,
    expected_latent: np.ndarray,
    expected_edit: np.ndarray,
) -> Tuple[
    List[str],
    Optional[Mapping[str, np.ndarray]],
    Optional[np.ndarray],
    Optional[np.ndarray],
]:
    errors: List[str] = []
    if not isinstance(value, Mapping):
        return [f"{name} must be a resume copy"], None, None, None
    if set(value) != {"trace", "final_physical", "policy_timing"}:
        errors.append(f"{name} has unexpected or missing fields")
    errors.extend(
        _validate_resume_trace_record(
            value.get("trace"),
            name=f"{name}.trace",
            expected_step=expected_step,
            expected_time=expected_time,
            expected_latent=expected_latent,
            expected_edit=expected_edit,
        )
    )
    normalized: Optional[np.ndarray] = None
    features: Dict[str, np.ndarray] = {}
    trace = value.get("trace")
    if isinstance(trace, Mapping) and isinstance(trace.get("leaves"), Mapping):
        leaves = trace["leaves"]
        normalized = _record_array(
            leaves.get("final_normalized"),
            name=f"{name}.trace.leaves.final_normalized",
            shape=MODEL_ACTION_SHAPE,
            errors=errors,
        )
        for key, shape in (
            ("x_t_pre_edit", MODEL_ACTION_SHAPE),
            ("latent_edit", MODEL_ACTION_SHAPE),
            ("x_t_post_edit", MODEL_ACTION_SHAPE),
            ("v_post_edit", MODEL_ACTION_SHAPE),
            ("predicted_clean_post_edit", MODEL_ACTION_SHAPE),
            ("predicted_clean_post_edit_physical", PHYSICAL_ACTION_SHAPE),
        ):
            array = _record_array(
                leaves.get(key),
                name=f"{name}.trace.leaves.{key}",
                shape=shape,
                errors=errors,
            )
            if array is not None:
                features[key] = array
    physical = _record_array(
        value.get("final_physical"),
        name=f"{name}.final_physical",
        shape=PHYSICAL_ACTION_SHAPE,
        errors=errors,
    )
    if not isinstance(value.get("policy_timing"), Mapping):
        errors.append(f"{name}.policy_timing must be an object")
    return errors, features if len(features) == 6 else None, normalized, physical


def _zero_source_parity(
    source_features: Mapping[str, np.ndarray],
    resume_features: Mapping[str, np.ndarray],
    source_normalized: np.ndarray,
    resume_normalized: np.ndarray,
    source_physical: np.ndarray,
    resume_physical: np.ndarray,
) -> Mapping[str, bool]:
    """Reconstruct every zero-edit source equality using canonical bytes."""

    return {
        "x_t_post_edit_equals_source_x_t": _arrays_byte_exact(
            resume_features["x_t_post_edit"], source_features["x_t"]
        ),
        "v_post_edit_equals_source_v_base": _arrays_byte_exact(
            resume_features["v_post_edit"], source_features["v_base"]
        ),
        "predicted_clean_post_edit_equals_source_predicted_clean": _arrays_byte_exact(
            resume_features["predicted_clean_post_edit"],
            source_features["predicted_clean"],
        ),
        "predicted_clean_post_edit_physical_equals_source_predicted_clean_physical": _arrays_byte_exact(
            resume_features["predicted_clean_post_edit_physical"],
            source_features["predicted_clean_physical"],
        ),
        "final_normalized_equals_source": _arrays_byte_exact(
            resume_normalized, source_normalized
        ),
        "final_physical_equals_source": _arrays_byte_exact(
            resume_physical, source_physical
        ),
    }


def _all_exact(parity: Mapping[str, bool]) -> bool:
    return bool(parity and all(value is True for value in parity.values()))


def validate_r04_resume_result(value: Mapping[str, Any]) -> List[str]:
    """Independently reconstruct every exact-capture, parity, and algebra claim."""

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
        "execution_boundaries",
        "usage_restriction",
    }
    missing = required - set(value)
    if missing:
        errors.append(f"missing required fields: {sorted(missing)}")
    unexpected = set(value) - required
    if unexpected:
        errors.append(f"unexpected top-level fields: {sorted(unexpected)}")
    if not (
        value.get("schema_version") == SCHEMA_VERSION
        and value.get("artifact_type") == ARTIFACT_TYPE
        and value.get("gate") == GATE
    ):
        errors.append("result must be a schema-version 1.0 R04B resume-parity artifact")
    if value.get("status") != FINAL_STATUS:
        errors.append("R04B final status must be completed")
    if value.get("case_id") != EXPECTED_SMOKE_CASE_ID:
        errors.append("R04B result does not use the frozen apparatus case")
    if not isinstance(value.get("run_id"), str) or not value.get("run_id"):
        errors.append("run_id must be a non-empty string")
    if value.get("apparatus_scope") != APPARATUS_SCOPE:
        errors.append("R04B apparatus scope differs")
    if not _is_sha256(value.get("config_hash")):
        errors.append("config_hash must be a lowercase SHA-256 digest")

    expected_case = _expected_case()
    source = value.get("source_evidence")
    source_case: Optional[Mapping[str, Any]] = None
    source_normalization_sha: Any = None
    if not isinstance(source, Mapping):
        errors.append("source_evidence must be an object")
    else:
        expected_source_fields = {
            "case_record",
            "case_record_sha256",
            "input_manifest_sha256",
            "config_file_sha256",
            "checkpoint_id",
            "checkpoint_sha256",
            "r00_summary",
            "r03_summary",
            "sampler_parity",
            "r04a_validation_summary",
            "r04a_raw_artifact",
        }
        if set(source) != expected_source_fields:
            errors.append("source_evidence has unexpected or missing fields")
        case_record = source.get("case_record")
        if not isinstance(case_record, Mapping):
            errors.append("source_evidence.case_record must be an object")
        else:
            source_case = case_record
            if (
                dict(case_record) != expected_case
                or source.get("case_record_sha256") != EXPECTED_SMOKE_CASE_RECORD_SHA256
                or source.get("case_record_sha256") != content_hash(dict(case_record))
            ):
                errors.append("source_evidence case is not the exact frozen R00/R04A row")
        expected_hashes = {
            "input_manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "config_file_sha256": EXPECTED_CONFIG_FILE_SHA256,
            "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        }
        for key, expected in expected_hashes.items():
            if source.get(key) != expected:
                errors.append(f"source_evidence.{key} differs from the frozen contract")
        expected_predecessors = {
            "r00_summary": EXPECTED_R00_SUMMARY_SHA256,
            "r03_summary": EXPECTED_R03_SUMMARY_SHA256,
            "sampler_parity": EXPECTED_PARITY_SHA256,
            "r04a_validation_summary": EXPECTED_R04A_VALIDATION_SUMMARY_SHA256,
            "r04a_raw_artifact": EXPECTED_R04A_RAW_ARTIFACT_SHA256,
        }
        for name, expected_sha in expected_predecessors.items():
            item = source.get(name)
            if not isinstance(item, Mapping) or item.get("sha256") != expected_sha:
                errors.append(f"source_evidence.{name} is not content-bound")
        r00 = source.get("r00_summary")
        if not isinstance(r00, Mapping) or r00.get("status") != "passed":
            errors.append("source_evidence.r00_summary is not passing")
        r03 = source.get("r03_summary")
        if not isinstance(r03, Mapping) or not (
            r03.get("status") == "passed"
            and r03.get("learned_probe_authorized") is True
        ):
            errors.append("source_evidence.r03_summary is not passing")
        parity = source.get("sampler_parity")
        if isinstance(parity, Mapping):
            if not (
                parity.get("status") == "passed"
                and parity.get("checkpoint_sha256") == EXPECTED_CHECKPOINT_SHA256
            ):
                errors.append("source_evidence.sampler_parity checkpoint/status differs")
            source_normalization_sha = parity.get("normalization_asset_sha256")
            if not _is_sha256(source_normalization_sha):
                errors.append("source_evidence.sampler_parity does not bind normalization")
        summary = source.get("r04a_validation_summary")
        if not isinstance(summary, Mapping) or not (
            summary.get("status") == "passed_apparatus_only"
            and summary.get("raw_artifact_sha256")
            == EXPECTED_R04A_RAW_ARTIFACT_SHA256
            and summary.get("independent_validation_error_count") == 0
        ):
            errors.append("source_evidence R04A compact validation binding differs")
        raw = source.get("r04a_raw_artifact")
        if not isinstance(raw, Mapping) or not (
            raw.get("status") == "validated_apparatus_only"
            and raw.get("golden_baseline") == expected_r04a_golden_contract()
        ):
            errors.append("source_evidence raw R04A artifact is not validated apparatus evidence")

    provenance = value.get("provenance")
    provenance_noise_sha: Any = None
    if not isinstance(provenance, Mapping):
        errors.append("provenance must be an object")
    else:
        if provenance.get("evidence_tier") != EVIDENCE_TIER:
            errors.append("R04B evidence tier must remain apparatus-only")
        if not _is_git_commit(provenance.get("git_commit")):
            errors.append("provenance.git_commit must be a full lowercase commit")
        if provenance.get("reviewed_git_commit") != provenance.get("git_commit"):
            errors.append("provenance commit differs from reviewed submission commit")
        if provenance.get("git_dirty") is not False:
            errors.append("R04B allocation evidence requires a clean worktree")
        if provenance.get("baseline_commit") != BASELINE_COMMIT:
            errors.append("provenance baseline commit differs")
        for key in ("python_version", "host"):
            if not isinstance(provenance.get(key), str) or not provenance.get(key):
                errors.append(f"provenance.{key} must be a non-empty string")
        try:
            timestamp = datetime.fromisoformat(provenance.get("timestamp"))
        except (TypeError, ValueError):
            errors.append("provenance.timestamp must be ISO-8601")
        else:
            if timestamp.tzinfo is None or timestamp.utcoffset() != timezone.utc.utcoffset(None):
                errors.append("provenance.timestamp must retain explicit UTC")
        for key in ALLOCATION_FIELDS:
            if not isinstance(provenance.get(key), str) or not provenance.get(key):
                errors.append(f"provenance.{key} must be a non-empty allocation value")
        frozen = {
            "sampler_steps": 10,
            "source_trace_steps": list(TRACE_STEPS),
            "source_eager_calls": 10,
            "zero_resume_calls": 10,
            "nonzero_resume_calls": 10,
            "compiled_default_calls": COMPILED_DEFAULT_CALLS,
            "total_policy_calls": TOTAL_POLICY_CALLS,
            "model_action_shape": list(MODEL_ACTION_SHAPE),
            "physical_action_shape": list(PHYSICAL_ACTION_SHAPE),
            "source_intervention_mode": "none",
            "resume_intervention_mode": "latent_resume_edit",
            "latent_edit_space": "model",
            "time_contract": TIME_CONTRACT,
            "noise_contract": NOISE_CONTRACT,
            "exact_equality_contract": EXACT_EQUALITY_CONTRACT,
            "nonzero_edit_role": NONZERO_EDIT_ROLE,
            "simulator_setup": SIMULATOR_SETUP,
            "policy_generated_action_steps_executed": 0,
            "efficacy_rollouts_executed": 0,
            "simulator_efficacy_evaluated": False,
            "no_training": True,
            "no_guidance": True,
            "no_learned_probe": True,
            "scientific_claim_authorized": False,
            "D_opt_status": "not_run_not_applicable",
            "D_sim_status": "not_run_not_applicable",
            "r04a_raw_artifact_sha256": EXPECTED_R04A_RAW_ARTIFACT_SHA256,
        }
        for key, expected in frozen.items():
            if provenance.get(key) != expected:
                errors.append(f"provenance.{key} differs from the frozen contract")
        provenance_noise_sha = provenance.get("noise_sha256")
        if not _is_sha256(provenance_noise_sha):
            errors.append("provenance.noise_sha256 must bind fixed policy noise")
        if provenance.get("normalization_asset_sha256") != source_normalization_sha:
            errors.append("provenance/source normalization hashes differ")
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

    pairing = value.get("pairing")
    if not isinstance(pairing, Mapping):
        errors.append("pairing must be an object")
    else:
        expected_pairing_fields = {
            "fixed_observation",
            "noise",
            "step_records",
            "baseline_reference",
            "source_call_count",
            "zero_resume_call_count",
            "nonzero_resume_call_count",
            "compiled_default_call_count",
            "total_policy_call_count",
            "all_source_duplicates_exact",
            "all_source_finals_exact_across_steps",
            "all_zero_resume_duplicates_exact",
            "all_zero_resume_source_parity_exact",
            "all_nonzero_resume_duplicates_exact",
            "all_nonzero_edit_algebra_exact",
        }
        if set(pairing) != expected_pairing_fields:
            errors.append("pairing has unexpected or missing fields")
        errors.extend(_validate_observation_identity(pairing.get("fixed_observation")))
        noise = _record_array(
            pairing.get("noise"),
            name="pairing.noise",
            shape=MODEL_ACTION_SHAPE,
            errors=errors,
        )
        noise_record = pairing.get("noise")
        noise_sha: Any = None
        if isinstance(noise_record, Mapping):
            noise_sha = noise_record.get("sha256")
        if noise_sha != provenance_noise_sha:
            errors.append("pairing noise differs from provenance.noise_sha256")
        if noise is not None:
            if noise.dtype != np.dtype("float32"):
                errors.append("pairing.noise must retain float32 dtype")
            expected_noise = np.random.default_rng(1250848483).normal(
                size=MODEL_ACTION_SHAPE
            ).astype(np.float32)
            if not _arrays_byte_exact(noise, expected_noise):
                errors.append("pairing.noise does not reconstruct from frozen policy seed")

        step_records = pairing.get("step_records")
        if not isinstance(step_records, list) or len(step_records) != len(TRACE_STEPS):
            errors.append("pairing.step_records must contain five records")
            step_records = []
        source_final_hashes: List[Tuple[Any, Any]] = []
        captured_times: List[float] = []
        for index, step_record in enumerate(step_records):
            if not isinstance(step_record, Mapping):
                errors.append(f"step record {index} must be an object")
                continue
            if set(step_record) != {
                "step_index",
                "captured_time",
                "source",
                "zero_resume",
                "nonzero_resume",
            }:
                errors.append(f"step record {index} has unexpected or missing fields")
            expected_step = TRACE_STEPS[index]
            if step_record.get("step_index") != expected_step:
                errors.append(f"step record {index} has the wrong step index")
            captured_time = _record_array(
                step_record.get("captured_time"),
                name=f"step_records[{index}].captured_time",
                shape=(1,),
                errors=errors,
            )
            if captured_time is None:
                continue
            if captured_time.dtype != np.dtype("float32"):
                errors.append(f"step record {index} captured time is not float32")
            captured_times.append(float(captured_time[0]))

            source_arm = step_record.get("source")
            if not isinstance(source_arm, Mapping):
                errors.append(f"step record {index}.source must be an object")
                continue
            if set(source_arm) != {"exact_duplicate", "primary", "duplicate"}:
                errors.append(f"step record {index}.source fields differ")
            primary_errors, source_features, source_norm, source_phys = _validate_source_copy(
                source_arm.get("primary"),
                name=f"step_records[{index}].source.primary",
                expected_step=expected_step,
                expected_time=captured_time,
            )
            duplicate_errors, duplicate_features, duplicate_norm, duplicate_phys = _validate_source_copy(
                source_arm.get("duplicate"),
                name=f"step_records[{index}].source.duplicate",
                expected_step=expected_step,
                expected_time=captured_time,
            )
            errors.extend(primary_errors)
            errors.extend(duplicate_errors)
            source_duplicate_exact = bool(
                source_features is not None
                and duplicate_features is not None
                and source_norm is not None
                and duplicate_norm is not None
                and source_phys is not None
                and duplicate_phys is not None
                and all(
                    _arrays_byte_exact(source_features[key], duplicate_features[key])
                    for key in source_features
                )
                and _arrays_byte_exact(source_norm, duplicate_norm)
                and _arrays_byte_exact(source_phys, duplicate_phys)
                and isinstance(source_arm.get("primary"), Mapping)
                and isinstance(source_arm.get("duplicate"), Mapping)
                and _trace_records_byte_exact(
                    source_arm["primary"].get("trace"),
                    source_arm["duplicate"].get("trace"),
                )
            )
            if source_arm.get("exact_duplicate") is not source_duplicate_exact or not source_duplicate_exact:
                errors.append(f"step record {index} source duplicate is not exact")
            if source_norm is None or source_phys is None or source_features is None:
                continue
            source_latent = source_features["x_t"]
            source_final_hashes.append(
                (_array_byte_identity(source_norm), _array_byte_identity(source_phys))
            )

            zero_edit = np.zeros(MODEL_ACTION_SHAPE, dtype=np.float32)
            nonzero_edit = _nonzero_edit()
            for arm_name, expected_edit, require_parity in (
                ("zero_resume", zero_edit, True),
                ("nonzero_resume", nonzero_edit, False),
            ):
                arm = step_record.get(arm_name)
                if not isinstance(arm, Mapping):
                    errors.append(f"step record {index}.{arm_name} must be an object")
                    continue
                expected_fields = {
                    "request",
                    "exact_duplicate",
                    "algebra_exact",
                    "primary",
                    "duplicate",
                }
                if require_parity:
                    expected_fields.add("source_parity")
                if set(arm) != expected_fields:
                    errors.append(f"step record {index}.{arm_name} fields differ")
                errors.extend(
                    _validate_resume_input(
                        arm.get("request"),
                        name=f"step_records[{index}].{arm_name}.request",
                        expected_latent=source_latent,
                        expected_time=captured_time,
                        expected_edit=expected_edit,
                        expected_noise_sha256=str(noise_sha),
                    )
                )
                first_errors, first_features, first_norm, first_phys = _validate_resume_copy(
                    arm.get("primary"),
                    name=f"step_records[{index}].{arm_name}.primary",
                    expected_step=expected_step,
                    expected_time=captured_time,
                    expected_latent=source_latent,
                    expected_edit=expected_edit,
                )
                second_errors, second_features, second_norm, second_phys = _validate_resume_copy(
                    arm.get("duplicate"),
                    name=f"step_records[{index}].{arm_name}.duplicate",
                    expected_step=expected_step,
                    expected_time=captured_time,
                    expected_latent=source_latent,
                    expected_edit=expected_edit,
                )
                errors.extend(first_errors)
                errors.extend(second_errors)
                duplicate_exact = bool(
                    first_features is not None
                    and second_features is not None
                    and first_norm is not None
                    and second_norm is not None
                    and first_phys is not None
                    and second_phys is not None
                    and all(
                        _arrays_byte_exact(first_features[key], second_features[key])
                        for key in first_features
                    )
                    and _arrays_byte_exact(first_norm, second_norm)
                    and _arrays_byte_exact(first_phys, second_phys)
                    and isinstance(arm.get("primary"), Mapping)
                    and isinstance(arm.get("duplicate"), Mapping)
                    and _trace_records_byte_exact(
                        arm["primary"].get("trace"),
                        arm["duplicate"].get("trace"),
                    )
                )
                if arm.get("exact_duplicate") is not duplicate_exact or not duplicate_exact:
                    errors.append(f"step record {index}.{arm_name} duplicate is not exact")
                if arm.get("algebra_exact") is not True:
                    errors.append(f"step record {index}.{arm_name} algebra flag is false")
                if require_parity:
                    if (
                        first_features is None
                        or first_norm is None
                        or first_phys is None
                    ):
                        reconstructed_parity: Mapping[str, bool] = {}
                    else:
                        reconstructed_parity = _zero_source_parity(
                            source_features,
                            first_features,
                            source_norm,
                            first_norm,
                            source_phys,
                            first_phys,
                        )
                    if arm.get("source_parity") != reconstructed_parity or not _all_exact(
                        reconstructed_parity
                    ):
                        errors.append(
                            f"step record {index} zero resume source feature/final byte parity failed"
                        )

        if len(captured_times) == len(TRACE_STEPS):
            if not all(
                math.isfinite(item) and 0.0 < item < 1.0 for item in captured_times
            ):
                errors.append("captured source times must be finite inside (0, 1)")
            if not all(
                captured_times[index] > captured_times[index + 1]
                for index in range(len(captured_times) - 1)
            ):
                errors.append("captured source times must be strictly decreasing")
        all_source_finals_exact = bool(
            len(source_final_hashes) == len(TRACE_STEPS)
            and len(set(source_final_hashes)) == 1
        )
        errors.extend(
            validate_baseline_reference(
                pairing.get("baseline_reference"),
                fixed_observation=pairing.get("fixed_observation"),
                noise_sha256=noise_sha,
                step_records=step_records,
            )
        )
        summary_flags = {
            "source_call_count": 10,
            "zero_resume_call_count": 10,
            "nonzero_resume_call_count": 10,
            "compiled_default_call_count": COMPILED_DEFAULT_CALLS,
            "total_policy_call_count": TOTAL_POLICY_CALLS,
            "all_source_duplicates_exact": True,
            "all_source_finals_exact_across_steps": all_source_finals_exact,
            "all_zero_resume_duplicates_exact": True,
            "all_zero_resume_source_parity_exact": True,
            "all_nonzero_resume_duplicates_exact": True,
            "all_nonzero_edit_algebra_exact": True,
        }
        for key, expected in summary_flags.items():
            if pairing.get(key) != expected:
                errors.append(f"pairing.{key} differs from independently reconstructed result")

    expected_boundaries = {
        "simulator_setup": SIMULATOR_SETUP,
        "policy_generated_action_steps_executed": 0,
        "efficacy_rollouts_executed": 0,
        "simulator_efficacy_evaluated": False,
        "training_executed": False,
        "guidance_executed": False,
        "learned_probe_executed": False,
        "scientific_claim_authorized": False,
        "D_opt_status": "not_run_not_applicable",
        "D_sim_status": "not_run_not_applicable",
        "nonzero_edit_role": NONZERO_EDIT_ROLE,
    }
    if value.get("execution_boundaries") != expected_boundaries:
        errors.append("R04B execution boundaries are missing or weakened")
    if value.get("usage_restriction") != USAGE_RESTRICTION:
        errors.append("R04B usage restriction is missing or weakened")
    return errors


def valid_r04_resume_completion(
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
        errors = validate_r04_resume_result(value)
    except (AttributeError, IndexError, KeyError, TypeError, ValueError, OverflowError):
        return False
    if errors:
        return False
    comparisons = {
        "case_id": case_id,
        "run_id": run_id,
        "config_hash": config_hash,
    }
    if any(
        expected is not None and value.get(key) != expected
        for key, expected in comparisons.items()
    ):
        return False
    source = value.get("source_evidence", {})
    provenance = value.get("provenance", {})
    return bool(
        (
            input_manifest_sha256 is None
            or source.get("input_manifest_sha256") == input_manifest_sha256
        )
        and (
            config_file_sha256 is None
            or source.get("config_file_sha256") == config_file_sha256
        )
        and (
            checkpoint_sha256 is None
            or source.get("checkpoint_sha256") == checkpoint_sha256
        )
        and (git_commit is None or provenance.get("git_commit") == git_commit)
    )


def run_r04_resume_case(
    case: Mapping[str, Any],
    config: R04ResumeConfig,
    *,
    repo_root: str | Path,
    input_manifest_sha256: str,
    client: Any = None,
    environment: Optional[SafeLiberoCase] = None,
) -> Tuple[Path, str]:
    """Run the one-case R04B apparatus and atomically finalize only if valid."""

    if input_manifest_sha256 != config.declared_manifest_sha256:
        raise ValueError("runtime manifest hash differs from the R04B frozen config")
    if dict(case) != _expected_case():
        raise ValueError("R04B may run only the frozen apparatus-only R00/R04A case")
    allocation = _allocation_provenance()
    oracle = config.oracle
    normalized_config = _normalized_config(config)
    config_hash = content_hash(normalized_config)
    git_commit, git_dirty = _git_state(Path(repo_root).resolve())
    if git_dirty:
        raise RuntimeError("R04B allocation worktree must be clean before evidence capture")
    reviewed_git_commit = os.environ.get("EXPECTED_GIT_COMMIT", "").strip()
    if not _is_git_commit(reviewed_git_commit) or git_commit != reviewed_git_commit:
        raise RuntimeError("R04B source commit differs from the reviewed submission commit")
    output = (
        Path(oracle.output_root)
        / oracle.run_id
        / str(case["case_id"])
        / "r04-resume-parity.json"
    )
    if valid_r04_resume_completion(
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
    noise_sha = _array_hash(noise)
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
        if environment.obstacle_name is None:
            raise RuntimeError("R04B frozen state did not resolve exactly one active obstacle")
        policy_input = policy_observation(
            initial_observation, environment.prompt, oracle.resize_size
        )
        observation_identity = _observation_identity(policy_input)
        golden_baseline = expected_r04a_golden_contract()
        if observation_identity.get("sha256") != golden_baseline["observation_sha256"]:
            raise RuntimeError("R04B fixed observation differs from the R04A golden observation")
        if noise_sha != golden_baseline["noise_sha256"]:
            raise RuntimeError("R04B fixed policy noise differs from the R04A golden noise")
        compiled_pre_reply = infer_compiled_default(client, policy_input, noise)
        compiled_pre_actions, compiled_pre_record = validated_compiled_reply(
            compiled_pre_reply,
            name="R04B pre-sequence compiled default",
        )
        step_records: List[Mapping[str, Any]] = []
        source_finals: List[Tuple[np.ndarray, np.ndarray]] = []
        captured_times: List[float] = []
        zero_edit = np.zeros(MODEL_ACTION_SHAPE, dtype=np.float32)
        nonzero_edit = _nonzero_edit()

        for step in config.trace_steps:
            source_replies = [
                _infer_source(
                    client,
                    policy_input,
                    noise,
                    intervention_step=step,
                )
                for _ in range(config.source_calls_per_step)
            ]
            source_values = [
                _validated_source_reply(reply, expected_step=step)
                for reply in source_replies
            ]
            first_actions, first_normalized, captured_time, first_features, first_trace = source_values[0]
            second_actions, second_normalized, second_time, second_features, second_trace = source_values[1]
            captured_latent = first_features["x_t"]
            source_duplicate_exact = bool(
                _arrays_byte_exact(captured_time, second_time)
                and all(
                    _arrays_byte_exact(first_features[key], second_features[key])
                    for key in first_features
                )
                and _arrays_byte_exact(first_normalized, second_normalized)
                and _arrays_byte_exact(first_actions, second_actions)
                and _trace_records_byte_exact(first_trace, second_trace)
            )
            if not source_duplicate_exact:
                raise RuntimeError(f"R04B source eager duplicate differs at step {step}")
            captured_times.append(float(captured_time[0]))
            source_finals.append((first_normalized, first_actions))

            arm_records: Dict[str, Mapping[str, Any]] = {}
            for arm_name, edit in (("zero_resume", zero_edit), ("nonzero_resume", nonzero_edit)):
                replies = [
                    _infer_resume(
                        client,
                        policy_input,
                        noise,
                        intervention_step=step,
                        resume_latent=captured_latent,
                        resume_time=captured_time,
                        latent_edit=edit,
                    )
                    for _ in range(2)
                ]
                values = [
                    _validated_resume_reply(
                        reply,
                        expected_step=step,
                        expected_time=captured_time,
                        expected_latent=captured_latent,
                        expected_edit=edit,
                    )
                    for reply in replies
                ]
                first_resume_actions, first_resume_normalized, first_resume_features, first_resume_trace = values[0]
                second_resume_actions, second_resume_normalized, second_resume_features, second_resume_trace = values[1]
                exact_duplicate = bool(
                    _arrays_byte_exact(first_resume_actions, second_resume_actions)
                    and _arrays_byte_exact(
                        first_resume_normalized, second_resume_normalized
                    )
                    and all(
                        _arrays_byte_exact(
                            first_resume_features[key], second_resume_features[key]
                        )
                        for key in first_resume_features
                    )
                    and _trace_records_byte_exact(
                        first_resume_trace, second_resume_trace
                    )
                )
                if not exact_duplicate:
                    raise RuntimeError(f"R04B {arm_name} duplicate differs at step {step}")
                arm_record: Dict[str, Any] = {
                    "request": _resume_input_record(
                        captured_latent, captured_time, edit, noise_sha
                    ),
                    "exact_duplicate": True,
                    "algebra_exact": True,
                    "primary": _resume_copy_record(
                        replies[0], first_resume_actions, first_resume_trace
                    ),
                    "duplicate": _resume_copy_record(
                        replies[1], second_resume_actions, second_resume_trace
                    ),
                }
                if arm_name == "zero_resume":
                    source_parity = _zero_source_parity(
                        first_features,
                        first_resume_features,
                        first_normalized,
                        first_resume_normalized,
                        first_actions,
                        first_resume_actions,
                    )
                    if not _all_exact(source_parity):
                        raise RuntimeError(
                            f"R04B zero resume does not byte-exactly reproduce source features/finals at step {step}: {source_parity}"
                        )
                    arm_record["source_parity"] = source_parity
                arm_records[arm_name] = arm_record

            step_records.append(
                {
                    "step_index": step,
                    "captured_time": _array_record(captured_time),
                    "source": {
                        "exact_duplicate": True,
                        "primary": _source_copy_record(
                            source_replies[0], first_actions, first_normalized, first_trace
                        ),
                        "duplicate": _source_copy_record(
                            source_replies[1], second_actions, second_normalized, second_trace
                        ),
                    },
                    "zero_resume": arm_records["zero_resume"],
                    "nonzero_resume": arm_records["nonzero_resume"],
                }
            )

        if not all(
            captured_times[index] > captured_times[index + 1]
            for index in range(len(captured_times) - 1)
        ):
            raise RuntimeError("R04B captured source times are not strictly decreasing")
        reference_normalized, reference_physical = source_finals[0]
        if not all(
            _arrays_byte_exact(reference_normalized, normalized)
            and _arrays_byte_exact(reference_physical, physical)
            for normalized, physical in source_finals[1:]
        ):
            raise RuntimeError("R04B eager trace requests changed the frozen source final")
        compiled_post_reply = infer_compiled_default(client, policy_input, noise)
        compiled_post_actions, compiled_post_record = validated_compiled_reply(
            compiled_post_reply,
            name="R04B post-sequence compiled default",
        )
        baseline_reference = build_baseline_reference(
            observation_identity=observation_identity,
            noise_sha256=noise_sha,
            step_records=step_records,
            compiled_pre_actions=compiled_pre_actions,
            compiled_pre_record=compiled_pre_record,
            compiled_post_actions=compiled_post_actions,
            compiled_post_record=compiled_post_record,
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
            "evidence_tier": EVIDENCE_TIER,
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
            "noise_sha256": noise_sha,
            "normalization_asset_sha256": config.source_evidence["sampler_parity"][
                "normalization_asset_sha256"
            ],
            "sampler_steps": oracle.sampler_steps,
            "source_trace_steps": list(config.trace_steps),
            "source_eager_calls": len(config.trace_steps) * config.source_calls_per_step,
            "zero_resume_calls": len(config.trace_steps) * config.zero_resume_calls_per_step,
            "nonzero_resume_calls": len(config.trace_steps)
            * config.nonzero_resume_calls_per_step,
            "compiled_default_calls": COMPILED_DEFAULT_CALLS,
            "total_policy_calls": TOTAL_POLICY_CALLS,
            "model_action_shape": list(MODEL_ACTION_SHAPE),
            "physical_action_shape": list(PHYSICAL_ACTION_SHAPE),
            "source_intervention_mode": "none",
            "resume_intervention_mode": "latent_resume_edit",
            "latent_edit_space": "model",
            "time_contract": TIME_CONTRACT,
            "noise_contract": NOISE_CONTRACT,
            "exact_equality_contract": EXACT_EQUALITY_CONTRACT,
            "nonzero_edit_role": NONZERO_EDIT_ROLE,
            "simulator_setup": SIMULATOR_SETUP,
            "policy_generated_action_steps_executed": 0,
            "efficacy_rollouts_executed": 0,
            "simulator_efficacy_evaluated": False,
            "no_training": True,
            "no_guidance": True,
            "no_learned_probe": True,
            "scientific_claim_authorized": False,
            "D_opt_status": "not_run_not_applicable",
            "D_sim_status": "not_run_not_applicable",
            "r04a_raw_artifact_sha256": EXPECTED_R04A_RAW_ARTIFACT_SHA256,
            **allocation,
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
                "step_records": step_records,
                "baseline_reference": baseline_reference,
                "source_call_count": 10,
                "zero_resume_call_count": 10,
                "nonzero_resume_call_count": 10,
                "compiled_default_call_count": COMPILED_DEFAULT_CALLS,
                "total_policy_call_count": TOTAL_POLICY_CALLS,
                "all_source_duplicates_exact": True,
                "all_source_finals_exact_across_steps": True,
                "all_zero_resume_duplicates_exact": True,
                "all_zero_resume_source_parity_exact": True,
                "all_nonzero_resume_duplicates_exact": True,
                "all_nonzero_edit_algebra_exact": True,
            },
            "execution_boundaries": {
                "simulator_setup": SIMULATOR_SETUP,
                "policy_generated_action_steps_executed": 0,
                "efficacy_rollouts_executed": 0,
                "simulator_efficacy_evaluated": False,
                "training_executed": False,
                "guidance_executed": False,
                "learned_probe_executed": False,
                "scientific_claim_authorized": False,
                "D_opt_status": "not_run_not_applicable",
                "D_sim_status": "not_run_not_applicable",
                "nonzero_edit_role": NONZERO_EDIT_ROLE,
            },
            "usage_restriction": config.reuse_role,
        }
        errors = validate_r04_resume_result(result)
        if errors:
            raise RuntimeError(
                "refusing invalid R04B final artifact: " + "; ".join(errors)
            )
        atomic_write_json(output, result)
        return output, FINAL_STATUS
    finally:
        if owns_environment:
            environment.close()
