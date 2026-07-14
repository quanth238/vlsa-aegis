"""Fail-closed paired R02 oracle-flow case runner.

R02 does not search for a new repair.  It consumes the immutable R01 result
set, reconstructs the same SafeLIBERO branch and explicit policy noise, and
compares six preregistered arms.  The direct witness remains a physical upper
bound; distributed residual and one-shot bridge editing remain distinct flow
mechanisms.

The module intentionally writes one final artifact for every member of the
original 20-case population.  The three R01 cases without a changed-action
``p_min`` witness are explicit ``not_applicable_no_r01_witness`` artifacts,
not missing data.  For eligible cases, sampled actions outside the registered
translation bounds are recorded as bounds failures and are never clipped or
sent to the simulator.
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
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from crfs_harness.artifacts import (
    atomic_write_json,
    content_hash,
    file_sha256,
    load_json,
    scientific_config,
)

from .endpoint_free_runner import validate_endpoint_free_result
from .measurement import signed_distance_point_to_oriented_box
from .progress_calibration import _target_contact_at_branch
from .reach_progress import (
    EXECUTED_REACH_ACTIONS,
    TARGET_OBJECT_NAME,
    ReachSnapshot,
    annotate_reach_rollout,
    annotate_reach_snapshots,
    capture_reach_snapshot,
    _tracked_reach_snapshots,
)
from .runner import (
    OracleConfig,
    SafeLiberoCase,
    _array_hash,
    _git_state,
    policy_observation,
)


SCHEMA_VERSION = "1.0"
GATE = "R02"
ARTIFACT_TYPE = "paired_oracle_flow_case"
BASELINE_COMMIT = "57b1aef306f212aea3574b0a3b64aa1a3d8f5e4b"
ACCEPTED_R01_SUMMARY_SHA256 = (
    "715d9326f02df531e0c6d7aa7b94629569b320800b4283c36fff41335d9b8bc5"
)
ACCEPTED_R01_ORDERED_RESULTS_SHA256 = (
    "6cc9bcf435dbe06396b90e34a0f4930994039538a6cc6ecad82876a538513140"
)
NORMALIZATION_ASSET_SHA256 = (
    "b3a44bb2810436fb62917decaea58bd4d9110255df527dea21e8fd40c960bd84"
)
PARITY_SEMANTICS_DECISION = "docs/decisions/0011-use-eager-path-for-r02-parity.md"
DIRECTION_SEMANTICS_DECISION = (
    "docs/decisions/0013-reference-oracle-to-fresh-paired-baseline.md"
)
DIRECTION_SEMANTICS_DECISION_SHA256 = (
    "9af853d339059d8bbfade06f7d43f5ffe3303d89d98cbfa24158016813251b0c"
)
DIRECTION_REFERENCE = (
    "immutable R01 witness translation minus fresh paired eager translation"
)
HISTORICAL_TRANSLATION_MAX_ABS_LIMIT = 0.010
HISTORICAL_TRANSLATION_RMS_LIMIT = 0.005
HISTORICAL_EXECUTED_MAX_ABS_LIMIT = 0.050
HISTORICAL_EXECUTED_RMS_LIMIT = 0.015
REGISTERED_TRANSLATION_ACTION_SCALE = (0.8422505, 0.827813, 0.937313)
REGISTERED_EEF_RADIUS_M = 0.06
REGISTERED_DISTANCE_LIMIT_M = 1.0
REGISTERED_RESPONSE_MATRIX_M_PER_ACTION = (
    (0.009370281145853376, 0.00014623337157483132, -0.0009171200705674251),
    (0.0000039560765073778535, 0.012040711768393353, 0.0000006816739267718219),
    (-0.002845391485346128, 0.000025633541617775525, 0.011826427878652547),
)
EXPECTED_MEASUREMENT_SAMPLES = 1 + EXECUTED_REACH_ACTIONS * 25
ARMS = (
    "frozen",
    "direct_witness",
    "random_residual",
    "analytic_geometry_residual",
    "oracle_residual",
    "bridge_diagnostic",
)
FLOW_ARMS = (
    "random_residual",
    "analytic_geometry_residual",
    "oracle_residual",
    "bridge_diagnostic",
)
NOMINAL_RECONFIRMATION_FAILURE = "nominal_collision_not_reconfirmed"
DIRECT_RECONFIRMATION_FAILURE = "direct_witness_not_reconfirmed"
NOT_EVALUATED_AFTER_RECONFIRMATION_FAILURE = (
    "not_evaluated_after_reconfirmation_failure"
)
FINAL_STATUSES = {
    "completed",
    "not_applicable_no_r01_witness",
    NOMINAL_RECONFIRMATION_FAILURE,
    DIRECT_RECONFIRMATION_FAILURE,
}
EVALUATED_ARM_STATUSES = {"passed_gate", "failed_gate"}
NONROLLOUT_ARM_STATUSES = {
    "bounds_failure",
    "direction_failure",
    "not_applicable_no_r01_witness",
    NOT_EVALUATED_AFTER_RECONFIRMATION_FAILURE,
}
RESULT_FIELDS = {
    "schema_version",
    "artifact_type",
    "gate",
    "case_id",
    "run_id",
    "status",
    "config_hash",
    "source_evidence",
    "provenance",
    "pairing",
    "directions",
    "arms",
    "outcome",
}


@dataclass(frozen=True)
class R02Config:
    """Validated, content-addressed configuration for one R02 population."""

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
    r01_summary_path: str
    r01_summary_sha256: str
    r01_summary: Mapping[str, Any]
    r01_results_root: str
    r01_result_hashes: Mapping[str, str]
    eligible_case_ids: Tuple[str, ...]
    no_witness_case_ids: Tuple[str, ...]
    parity_artifact_path: str
    parity_artifact_sha256: str
    parity_artifact: Mapping[str, Any]
    direction_reference: str
    direction_semantics_decision: str
    direction_semantics_decision_sha256: str


class R02SourceError(ValueError):
    """Raised when immutable predecessor evidence does not bind cleanly."""


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


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _validate_parity_artifact(
    value: Mapping[str, Any],
    *,
    r01_summary_sha256: str,
    checkpoint_sha256: str,
) -> List[str]:
    """Validate raw parity arrays, then enforce R02-specific identity bindings."""

    # Do not trust a top-level ``status=passed`` claim.  The allocation parity
    # validator independently reconstructs every within-/cross-backend metric
    # from the stored worker arrays and rejects malformed provenance, fixture,
    # tolerance, checkpoint, and ADR-0011 records.
    from run_sampler_parity import validate_parity_artifact as validate_raw_parity

    errors: List[str] = [
        f"authoritative parity validation: {error}"
        for error in validate_raw_parity(value)
    ]
    if value.get("schema_version") != "1.0":
        errors.append("parity schema_version must be 1.0")
    if value.get("artifact_type") != "sampler_parity" or value.get("gate") != "R02":
        errors.append("parity artifact must be the R02 sampler_parity artifact")
    if value.get("status") != "passed":
        errors.append("sampler parity must pass before R02 case execution")
    identity = value.get("identity")
    if not isinstance(identity, Mapping):
        errors.append("parity identity must be an object")
    else:
        if identity.get("r01_summary_sha256") != r01_summary_sha256:
            errors.append("parity artifact is bound to a different R01 summary")
        if identity.get("num_steps") != 10:
            errors.append("parity artifact must compare ten Euler steps")
        if identity.get("action_horizon") != 10 or identity.get("action_dim") != 32:
            errors.append("parity artifact must compare a 10x32 model action")
        if identity.get("git_dirty") is not False:
            errors.append("sampler parity must come from a clean allocation worktree")
    acceptance = value.get("acceptance")
    if not isinstance(acceptance, Mapping) or acceptance.get("passed") is not True:
        errors.append("parity acceptance must be explicitly passed")
    else:
        checks = acceptance.get("checks")
        if not isinstance(checks, Mapping) or not checks or not all(
            item is True for item in checks.values()
        ):
            errors.append("every parity acceptance check must pass")
        if acceptance.get("path_identity_decision") != PARITY_SEMANTICS_DECISION:
            errors.append("sampler parity must use the ADR-0011 eager-path identity")
        if acceptance.get("primary_path") != (
            "public JAX default versus PyTorch eager trace-only"
        ):
            errors.append("sampler parity primary path must be PyTorch eager trace-only")
    comparison = value.get("comparison")
    if not isinstance(comparison, Mapping):
        errors.append("parity comparison must be an object")
    else:
        latent_steps = comparison.get("cross_backend_pre_update_x_t_per_step")
        velocity_steps = comparison.get("cross_backend_pre_update_v_t_per_step")
        if not isinstance(latent_steps, list) or len(latent_steps) != 10:
            errors.append("parity comparison must contain all ten pre-update Euler latents")
        elif [item.get("step_index") for item in latent_steps if isinstance(item, Mapping)] != list(range(10)) or not all(
            isinstance(item, Mapping) and item.get("passed") is True for item in latent_steps
        ):
            errors.append("all ten indexed parity latent comparisons must pass")
        if not isinstance(velocity_steps, list) or len(velocity_steps) != 10:
            errors.append("parity comparison must contain all ten pre-update Euler velocities")
        elif [item.get("step_index") for item in velocity_steps if isinstance(item, Mapping)] != list(range(10)) or not all(
            isinstance(item, Mapping) and item.get("passed") is True for item in velocity_steps
        ):
            errors.append("all ten indexed parity velocity comparisons must pass")
    checkpoints = value.get("checkpoints")
    if not isinstance(checkpoints, Mapping):
        errors.append("parity checkpoints must be an object")
    else:
        converted = checkpoints.get("converted_pytorch")
        if not isinstance(converted, Mapping):
            errors.append("parity artifact has no converted PyTorch checkpoint")
        else:
            if converted.get("model_sha256") != checkpoint_sha256:
                errors.append("parity and R02 converted-checkpoint hashes differ")
            if converted.get("norm_stats_sha256") != NORMALIZATION_ASSET_SHA256:
                errors.append("parity normalization asset differs from the registered R02 asset")
    return errors


def _validate_r01_summary(value: Mapping[str, Any]) -> Tuple[
    Dict[str, str], Tuple[str, ...], Tuple[str, ...]
]:
    if value.get("schema_version") != "1.0" or value.get("gate") != "R01":
        raise R02SourceError("R01 summary must be a schema_version 1.0 R01 artifact")
    if value.get("status") != "passed" or value.get("gate_passed") is not True:
        raise R02SourceError("R01 summary must pass")
    if value.get("ordered_result_set_digest") != ACCEPTED_R01_ORDERED_RESULTS_SHA256:
        raise R02SourceError("R01 ordered raw-result digest differs from the accepted population")
    counts = value.get("counts")
    case_ids = value.get("case_ids")
    if not isinstance(counts, Mapping) or not isinstance(case_ids, Mapping):
        raise R02SourceError("R01 summary is missing population counts or case identities")
    if counts.get("expected_population") != 20 or counts.get("validated_population") != 20:
        raise R02SourceError("R02 requires the complete validated 20-case R01 population")
    eligible = case_ids.get("changed_action_p_min_rescues")
    if not isinstance(eligible, list) or len(eligible) != 17 or len(set(eligible)) != 17:
        raise R02SourceError("R01 summary must contain exactly 17 unique changed p_min witnesses")
    branch = value.get("branch_clearance")
    negatives = branch.get("branch_below_registered_margin_case_ids") if isinstance(branch, Mapping) else None
    if not isinstance(negatives, list) or len(negatives) != 3 or len(set(negatives)) != 3:
        raise R02SourceError("R01 summary must retain the three explicit branch-margin negatives")
    if set(eligible) & set(negatives):
        raise R02SourceError("R01 eligible and no-witness identities overlap")
    result_hashes = value.get("result_hashes")
    if not isinstance(result_hashes, list) or len(result_hashes) != 20:
        raise R02SourceError("R01 summary must bind all 20 raw result hashes")
    hashes: Dict[str, str] = {}
    for item in result_hashes:
        if not isinstance(item, Mapping):
            raise R02SourceError("R01 result_hashes entries must be objects")
        case_id = item.get("case_id")
        sha256 = item.get("sha256")
        if not isinstance(case_id, str) or not case_id or not _is_sha256(sha256):
            raise R02SourceError("R01 result_hashes contains an invalid identity or digest")
        if case_id in hashes:
            raise R02SourceError(f"duplicate R01 result hash for {case_id}")
        hashes[case_id] = str(sha256)
    if set(hashes) != set(eligible) | set(negatives):
        raise R02SourceError("R01 result hashes do not equal the 17+3 registered population")
    return hashes, tuple(sorted(eligible)), tuple(sorted(negatives))


def r02_config_from_mapping(
    value: Mapping[str, Any],
    oracle: OracleConfig,
    *,
    repo_root: str | Path,
) -> R02Config:
    """Load R02 only after all predecessor artifacts verify by content."""

    if not bool(value.get("ready_to_run", False)):
        raise ValueError("R02 config is not ready_to_run")
    settings = value.get("r02")
    if not isinstance(settings, Mapping):
        raise ValueError("R02 config requires an r02 object")
    if oracle.action_horizon != 10 or oracle.action_dim != 32:
        raise ValueError("R02 requires a 10x32 model action")
    if oracle.executed_prefix != 5 or oracle.sampler_steps != 10 or oracle.intervention_step != 5:
        raise ValueError("R02 requires five executed actions and midpoint step 5 of 10")
    if oracle.response_matrix_m_per_action is None:
        raise ValueError("R02 analytic geometry requires the frozen H04 response matrix")
    observed_response = tuple(
        tuple(float(item) for item in row)
        for row in oracle.response_matrix_m_per_action
    )
    if observed_response != REGISTERED_RESPONSE_MATRIX_M_PER_ACTION:
        raise ValueError("R02 response matrix differs from the frozen H04 calibration")
    if not math.isclose(
        oracle.eef_radius_m, REGISTERED_EEF_RADIUS_M, rel_tol=0.0, abs_tol=1e-15
    ):
        raise ValueError("R02 D_sim/D_opt EEF radius must remain 6 cm")
    if not math.isclose(
        oracle.distance_limit_m,
        REGISTERED_DISTANCE_LIMIT_M,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError("R02 MuJoCo distance limit must remain 1 m")
    if settings.get("phase") != "pregrasp_reach":
        raise ValueError("R02 is restricted to the frozen pregrasp_reach population")
    if tuple(settings.get("required_arms", ())) != ARMS:
        raise ValueError(f"r02.required_arms must equal {list(ARMS)!r} in order")
    if settings.get("clipping_policy") != "fail_without_clipping":
        raise ValueError("R02 sampled-action bounds failures must never be clipped")

    root = Path(repo_root).resolve()
    direction_reference = settings.get("direction_reference")
    direction_decision = settings.get("direction_semantics_decision")
    direction_decision_sha = settings.get("direction_semantics_decision_sha256")
    if direction_reference != DIRECTION_REFERENCE:
        raise ValueError(
            "R02 direction_reference must use the fresh paired eager baseline"
        )
    if direction_decision != DIRECTION_SEMANTICS_DECISION:
        raise ValueError("R02 direction semantics must bind ADR-0013")
    if direction_decision_sha != DIRECTION_SEMANTICS_DECISION_SHA256:
        raise ValueError("R02 direction semantics ADR-0013 SHA-256 differs")
    direction_decision_path = _resolve(str(direction_decision), root)
    if not _inside(direction_decision_path, root / "docs" / "decisions"):
        raise ValueError("R02 direction semantics decision must be checked in")
    if file_sha256(direction_decision_path) != direction_decision_sha:
        raise ValueError("R02 direction semantics ADR-0013 content hash differs")

    summary_value = settings.get("r01_summary_artifact")
    summary_sha = settings.get("r01_summary_sha256")
    if not isinstance(summary_value, str) or not summary_value:
        raise ValueError("r02.r01_summary_artifact must be a path")
    if summary_sha != ACCEPTED_R01_SUMMARY_SHA256:
        raise ValueError("R02 must bind the accepted checked-in R01 summary SHA-256")
    summary_path = _resolve(summary_value, root)
    if not _inside(summary_path, root / "evidence" / "r01"):
        raise ValueError("R01 summary must be the checked-in evidence/r01 artifact")
    actual_summary_sha = file_sha256(summary_path)
    if actual_summary_sha != summary_sha:
        raise ValueError(
            f"R01 summary hash mismatch: expected {summary_sha}, got {actual_summary_sha}"
        )
    summary = load_json(summary_path)
    if not isinstance(summary, Mapping):
        raise ValueError("R01 summary must be an object")
    result_hashes, eligible, no_witness = _validate_r01_summary(summary)
    identities = summary.get("identities")
    if not isinstance(identities, Mapping):
        raise ValueError("R01 summary has no immutable identities object")
    if identities.get("baseline_commit") != BASELINE_COMMIT:
        raise ValueError("R01 summary baseline revision differs from R02")
    if identities.get("checkpoint_sha256") != oracle.checkpoint_sha256:
        raise ValueError("R01 and R02 checkpoint hashes differ")
    if not _is_sha256(identities.get("manifest_sha256")):
        raise ValueError("R01 summary has no valid frozen manifest hash")

    results_root_value = settings.get("r01_results_root")
    if not isinstance(results_root_value, str) or not results_root_value:
        raise ValueError("r02.r01_results_root must locate immutable raw R01 artifacts")
    results_root = _resolve(results_root_value, root)

    parity_value = settings.get("sampler_parity_artifact")
    parity_sha = settings.get("sampler_parity_sha256")
    if not isinstance(parity_value, str) or not parity_value:
        raise ValueError("r02.sampler_parity_artifact must be a path")
    if not _is_sha256(parity_sha):
        raise ValueError("r02.sampler_parity_sha256 must be a lowercase SHA-256")
    parity_path = _resolve(parity_value, root)
    actual_parity_sha = file_sha256(parity_path)
    if actual_parity_sha != parity_sha:
        raise ValueError(
            f"sampler parity hash mismatch: expected {parity_sha}, got {actual_parity_sha}"
        )
    parity = load_json(parity_path)
    if not isinstance(parity, Mapping):
        raise ValueError("sampler parity artifact must be an object")
    parity_errors = _validate_parity_artifact(
        parity,
        r01_summary_sha256=actual_summary_sha,
        checkpoint_sha256=oracle.checkpoint_sha256,
    )
    if parity_errors:
        raise ValueError("invalid or failed sampler parity artifact: " + "; ".join(parity_errors))

    target_name = str(settings.get("target_object", ""))
    if target_name != TARGET_OBJECT_NAME:
        raise ValueError(f"R02 target must be {TARGET_OBJECT_NAME!r}")
    p_min_m = _finite(summary.get("identities", {}).get("p_min_m"), name="R01 p_min", positive=True)
    declared_p_min = _finite(settings.get("minimum_progress_m"), name="R02 p_min", positive=True)
    if not math.isclose(p_min_m, declared_p_min, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError("R02 minimum progress differs from the accepted R01 summary")
    safety_margin = _finite(
        settings.get("simulator_safety_margin_m"), name="R02 simulator margin", positive=True
    )
    if not math.isclose(safety_margin, 0.005, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("R02 simulator safety margin must remain 5 mm")
    if not math.isclose(oracle.safety_margin_m, safety_margin, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("R02 top-level and arm safety margins differ")
    target_limit = _finite(settings.get("maximum_target_displacement_m"), name="target motion limit")
    obstacle_limit = _finite(
        settings.get("maximum_obstacle_displacement_m"), name="obstacle motion limit"
    )
    if not (0.0 <= target_limit <= 0.001 and 0.0 <= obstacle_limit <= 0.001):
        raise ValueError("R02 scene-motion limits must lie in [0, 1 mm]")
    repeats = int(settings.get("simulator_repeats", oracle.measurement_repeats))
    if repeats < 2 or repeats != oracle.measurement_repeats:
        raise ValueError("R02 requires the same declared simulator repeat count, at least two")
    bounds = settings.get("translation_action_bounds")
    if not isinstance(bounds, Sequence) or isinstance(bounds, (str, bytes)) or len(bounds) != 2:
        raise ValueError("R02 translation_action_bounds must be [low, high]")
    low = _finite(bounds[0], name="translation action low")
    high = _finite(bounds[1], name="translation action high")
    if low != -1.0 or high != 1.0:
        raise ValueError("R02 translation action bounds must remain [-1, 1]")
    scale_value = settings.get("normalization_action_scale")
    if not isinstance(scale_value, Sequence) or isinstance(scale_value, (str, bytes)):
        raise ValueError("R02 normalization_action_scale must contain three values")
    scale = tuple(_finite(item, name="normalization action scale", positive=True) for item in scale_value)
    if len(scale) != 3 or any(
        not math.isclose(item, expected, rel_tol=0.0, abs_tol=1e-12)
        for item, expected in zip(scale, REGISTERED_TRANSLATION_ACTION_SCALE)
    ):
        raise ValueError("R02 normalization action scale differs from decision 0010")
    if settings.get("normalization_asset_sha256") != NORMALIZATION_ASSET_SHA256:
        raise ValueError("R02 normalization asset hash differs from decision 0010")
    samples = int(settings.get("analytic_samples_per_segment", 26))
    if samples != 26:
        raise ValueError("R02 analytic D_opt must use 26 samples per segment")

    return R02Config(
        oracle=oracle,
        target_name=target_name,
        p_min_m=p_min_m,
        simulator_safety_margin_m=safety_margin,
        maximum_target_displacement_m=target_limit,
        maximum_obstacle_displacement_m=obstacle_limit,
        simulator_repeats=repeats,
        translation_action_low=low,
        translation_action_high=high,
        action_scale=scale,  # type: ignore[arg-type]
        samples_per_segment=samples,
        r01_summary_path=str(summary_path),
        r01_summary_sha256=actual_summary_sha,
        r01_summary=summary,
        r01_results_root=str(results_root),
        r01_result_hashes=result_hashes,
        eligible_case_ids=eligible,
        no_witness_case_ids=no_witness,
        parity_artifact_path=str(parity_path),
        parity_artifact_sha256=actual_parity_sha,
        parity_artifact=parity,
        direction_reference=str(direction_reference),
        direction_semantics_decision=str(direction_decision),
        direction_semantics_decision_sha256=str(direction_decision_sha),
    )


def _normalized_config(config: R02Config) -> Dict[str, Any]:
    """Remove transport paths while retaining every scientific content hash."""

    return {
        **scientific_config(config.oracle.__dict__),
        "target_name": config.target_name,
        "p_min_m": config.p_min_m,
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
        "analytic_samples_per_segment": config.samples_per_segment,
        "required_arms": list(ARMS),
        "clipping_policy": "fail_without_clipping",
        "r01_summary_sha256": config.r01_summary_sha256,
        "r01_ordered_result_set_digest": ACCEPTED_R01_ORDERED_RESULTS_SHA256,
        "r01_result_hashes": dict(sorted(config.r01_result_hashes.items())),
        "eligible_case_ids": list(config.eligible_case_ids),
        "no_witness_case_ids": list(config.no_witness_case_ids),
        "sampler_parity_sha256": config.parity_artifact_sha256,
        "direction_reference": config.direction_reference,
        "direction_semantics_decision": config.direction_semantics_decision,
        "direction_semantics_decision_sha256": (
            config.direction_semantics_decision_sha256
        ),
    }


def _array_record(value: Any, *, dtype: Optional[np.dtype] = None) -> Dict[str, Any]:
    array = np.asarray(value, dtype=dtype)
    if not np.all(np.isfinite(array)):
        raise ValueError("array record contains a non-finite value")
    contiguous = np.ascontiguousarray(array)
    return {
        "dtype": str(contiguous.dtype),
        "shape": list(contiguous.shape),
        "sha256": _array_hash(contiguous),
        "values": contiguous.tolist(),
    }


def _array_from_record(value: Mapping[str, Any]) -> np.ndarray:
    return np.asarray(value["values"], dtype=np.dtype(str(value["dtype"])))


def _validate_array_record(
    value: Any,
    *,
    name: str,
    shape: Optional[Tuple[int, ...]] = None,
) -> Tuple[Optional[np.ndarray], List[str]]:
    errors: List[str] = []
    if not isinstance(value, Mapping):
        return None, [f"{name} must be an array record"]
    if set(value) != {"dtype", "shape", "sha256", "values"}:
        errors.append(f"{name} has incorrect array-record fields")
    try:
        dtype = np.dtype(str(value.get("dtype")))
        array = np.asarray(value.get("values"), dtype=dtype)
    except (TypeError, ValueError) as error:
        return None, [f"{name} cannot be reconstructed: {error}"]
    if not np.all(np.isfinite(array)):
        errors.append(f"{name} contains non-finite values")
    if value.get("shape") != list(array.shape):
        errors.append(f"{name}.shape conflicts with values")
    if shape is not None and array.shape != shape:
        errors.append(f"{name} must have shape {shape}, got {array.shape}")
    if value.get("sha256") != _array_hash(np.ascontiguousarray(array)):
        errors.append(f"{name}.sha256 conflicts with values")
    return array, errors


def _trace_record(trace: Mapping[str, Any]) -> Dict[str, Any]:
    leaves = {str(key): _array_record(value) for key, value in sorted(trace.items())}
    return {"sha256": content_hash(leaves), "leaves": leaves}


def _validate_trace_record(value: Any, *, name: str) -> List[str]:
    errors: List[str] = []
    if not isinstance(value, Mapping) or set(value) != {"sha256", "leaves"}:
        return [f"{name} must be a trace record"]
    leaves = value.get("leaves")
    if not isinstance(leaves, Mapping) or not leaves:
        return [f"{name}.leaves must be a non-empty object"]
    for key, leaf in leaves.items():
        _, item_errors = _validate_array_record(leaf, name=f"{name}.leaves.{key}")
        errors.extend(item_errors)
    if value.get("sha256") != content_hash(dict(leaves)):
        errors.append(f"{name}.sha256 conflicts with leaves")
    return errors


def _observation_fingerprint(value: Mapping[str, Any]) -> Dict[str, Any]:
    digest = hashlib.sha256()
    leaves = []
    for key in sorted(value):
        item = value[key]
        if isinstance(item, str):
            payload = item.encode("utf-8")
            record = {
                "key": key,
                "kind": "string",
                "sha256": hashlib.sha256(payload).hexdigest(),
                "value": item,
            }
        else:
            array = np.ascontiguousarray(np.asarray(item))
            payload = array.tobytes()
            record = {
                "key": key,
                "kind": "array",
                "dtype": str(array.dtype),
                "shape": list(array.shape),
                "sha256": _array_hash(array),
            }
        framed = json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
        digest.update(len(framed).to_bytes(8, "big"))
        digest.update(framed)
        leaves.append(record)
    return {"sha256": digest.hexdigest(), "leaves": leaves}


def _load_r01_case(
    case_id: str,
    config: R02Config,
) -> Tuple[Path, Mapping[str, Any], str, bool]:
    if case_id not in config.r01_result_hashes:
        raise R02SourceError(f"case {case_id!r} is outside the bound R01 population")
    path = Path(config.r01_results_root) / case_id / "endpoint-free-feasibility.json"
    actual_sha = file_sha256(path)
    expected_sha = config.r01_result_hashes[case_id]
    if actual_sha != expected_sha:
        raise R02SourceError(
            f"raw R01 hash mismatch for {case_id}: expected {expected_sha}, got {actual_sha}"
        )
    value = load_json(path)
    if not isinstance(value, Mapping):
        raise R02SourceError(f"raw R01 result for {case_id} is not an object")
    errors = validate_endpoint_free_result(value)
    if errors:
        raise R02SourceError(f"raw R01 result for {case_id} is invalid: {'; '.join(errors)}")
    if value.get("case_id") != case_id:
        raise R02SourceError("raw R01 result case identity differs")
    eligible = case_id in config.eligible_case_ids
    if eligible != (value.get("status") == "verified_safe_progress"):
        raise R02SourceError("R01 summary eligibility conflicts with the validated raw result")
    if not eligible and case_id not in config.no_witness_case_ids:
        raise R02SourceError("R01 case is neither eligible nor an explicit no-witness case")
    return path, value, actual_sha, eligible


def _witness_pointer(witness: Any, witness_actions: np.ndarray) -> Dict[str, Any]:
    values = np.asarray(witness_actions, dtype=np.float64)
    return {
        "selection_rule": "first raw changed p_min witness in p_min then p_zero attempt order",
        "search": str(witness.search),
        "raw_attempt_index": int(witness.raw_attempt_index),
        "candidate_index": int(witness.candidate_index),
        "source": str(witness.source),
        # R01's outcome pointer hashes the JSON action values.  Keep that hash
        # verbatim and add the dtype/shape-framed array hash used by R02.
        "actions_sha256": content_hash(values.tolist()),
        "actions_array_sha256": _array_hash(values),
    }


def _infer(
    client: Any,
    observation: Mapping[str, Any],
    noise: np.ndarray,
    config: R02Config,
    *,
    intervention_mode: str,
    return_trace: bool,
    correction: Optional[np.ndarray] = None,
) -> Mapping[str, Any]:
    request = copy.deepcopy(dict(observation))
    controls: Dict[str, Any] = {
        "noise": np.array(noise, copy=True),
        "intervention_step": config.oracle.intervention_step,
        "intervention_mode": intervention_mode,
        "return_trace": return_trace,
    }
    if correction is not None:
        model_correction = np.asarray(correction, dtype=np.float32)
        expected = (config.oracle.action_horizon, config.oracle.action_dim)
        if model_correction.shape != expected:
            raise ValueError(f"model correction must have shape {expected}, got {model_correction.shape}")
        if not np.all(np.isfinite(model_correction)):
            raise ValueError("model correction must be finite")
        controls["correction"] = model_correction
        controls["correction_space"] = "model"
    request["__crfs__"] = controls
    reply = client.infer(request)
    if not isinstance(reply, Mapping) or "actions" not in reply:
        raise RuntimeError("policy reply has no actions")
    return reply


def _trace_arrays(reply: Mapping[str, Any]) -> Mapping[str, Any]:
    trace = reply.get("crfs_trace")
    if not isinstance(trace, Mapping):
        raise RuntimeError("eager R02 request did not return a CRFS trace")
    required = {"step_index", "time", "x_t", "v_base", "predicted_clean", "predicted_clean_physical"}
    if not required.issubset(trace):
        raise RuntimeError(f"CRFS trace missing fields: {sorted(required - set(trace))}")
    return trace


def _same_trace(first: Mapping[str, Any], second: Mapping[str, Any]) -> bool:
    return bool(
        set(first) == set(second)
        and all(np.array_equal(np.asarray(first[key]), np.asarray(second[key])) for key in first)
    )


def _path_diagnostic(reference: np.ndarray, candidate: np.ndarray) -> Dict[str, Any]:
    first = np.asarray(reference, dtype=np.float64)
    second = np.asarray(candidate, dtype=np.float64)
    if first.shape != second.shape:
        return {
            "shape_equal": False,
            "array_equal": False,
            "maximum_absolute_error": None,
            "rms_absolute_error": None,
        }
    error = np.abs(first - second)
    return {
        "shape_equal": True,
        "array_equal": bool(np.array_equal(first, second)),
        "maximum_absolute_error": float(np.max(error)) if error.size else 0.0,
        "rms_absolute_error": (
            float(np.sqrt(np.mean(np.square(error)))) if error.size else 0.0
        ),
    }


def _json_compatible(value: Any) -> Any:
    """Canonicalize tuples and other JSON sequences without numeric tolerance."""

    return json.loads(json.dumps(value, allow_nan=False, sort_keys=True))


def _bounded_drift_diagnostic(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    maximum_absolute_error_limit: float,
    rms_absolute_error_limit: float,
) -> Dict[str, Any]:
    diagnostic = _path_diagnostic(reference, candidate)
    maximum = diagnostic["maximum_absolute_error"]
    rms = diagnostic["rms_absolute_error"]
    return {
        **diagnostic,
        "maximum_absolute_error_limit": maximum_absolute_error_limit,
        "rms_absolute_error_limit": rms_absolute_error_limit,
        "passed": bool(
            diagnostic["shape_equal"]
            and isinstance(maximum, float)
            and isinstance(rms, float)
            and maximum <= maximum_absolute_error_limit
            and rms <= rms_absolute_error_limit
        ),
    }


def _delta_comparison_diagnostic(
    nominal_actions: Any,
    fresh_eager_actions: Any,
    witness_actions: Any,
) -> Dict[str, float]:
    """Compare historical and fresh witness deltas in scale-only model space."""

    nominal = np.asarray(nominal_actions, dtype=np.float64)
    fresh = np.asarray(fresh_eager_actions, dtype=np.float64)
    witness = np.asarray(witness_actions, dtype=np.float64)
    if nominal.shape != (5, 7) or fresh.shape != (5, 7) or witness.shape != (5, 7):
        raise ValueError("Delta comparison requires nominal, fresh, and witness 5x7 actions")
    if not all(np.all(np.isfinite(item)) for item in (nominal, fresh, witness)):
        raise ValueError("Delta comparison actions must be finite")
    scale = np.asarray(REGISTERED_TRANSLATION_ACTION_SCALE, dtype=np.float64)
    raw_model = np.zeros((10, 32), dtype=np.float64)
    fresh_model = np.zeros((10, 32), dtype=np.float64)
    raw_model[:5, :3] = (witness[:5, :3] - nominal[:5, :3]) / scale[None, :]
    fresh_model[:5, :3] = (witness[:5, :3] - fresh[:5, :3]) / scale[None, :]
    raw_l2 = float(np.linalg.norm(raw_model))
    fresh_l2 = float(np.linalg.norm(fresh_model))
    if raw_l2 <= 0.0 or fresh_l2 <= 0.0:
        raise ValueError("historical and fresh model-space Delta norms must be nonzero")
    cosine = float(
        np.dot(raw_model.reshape(-1), fresh_model.reshape(-1))
        / (raw_l2 * fresh_l2)
    )
    cosine = max(-1.0, min(1.0, cosine))
    return {
        "raw_model_l2": raw_l2,
        "fresh_model_l2": fresh_l2,
        "fresh_minus_raw_model_l2": float(np.linalg.norm(fresh_model - raw_model)),
        "cosine": cosine,
        "angle_degrees": float(math.degrees(math.acos(cosine))),
    }


def _historical_r01_diagnostic(
    nominal_actions: Any,
    fresh_eager_actions: Any,
    *,
    witness_actions: Optional[Any] = None,
) -> Dict[str, Any]:
    """Keep the stale R01 action relation visible without applying it.

    The raw R01 nominal and witness came from a different allocation.  Their
    difference remains useful source evidence, while the applied R02 oracle is
    constructed separately against the fresh eager baseline below.
    """

    nominal = np.asarray(nominal_actions, dtype=np.float64)
    fresh = np.asarray(fresh_eager_actions, dtype=np.float64)
    if nominal.shape != (5, 7) or fresh.shape != (5, 7):
        raise ValueError("historical R01 diagnostic requires paired 5x7 actions")
    if not np.all(np.isfinite(nominal)) or not np.all(np.isfinite(fresh)):
        raise ValueError("historical R01 diagnostic actions must be finite")
    translation_drift = _bounded_drift_diagnostic(
        nominal[:, :3],
        fresh[:, :3],
        maximum_absolute_error_limit=HISTORICAL_TRANSLATION_MAX_ABS_LIMIT,
        rms_absolute_error_limit=HISTORICAL_TRANSLATION_RMS_LIMIT,
    )
    executed_drift = _bounded_drift_diagnostic(
        nominal,
        fresh,
        maximum_absolute_error_limit=HISTORICAL_EXECUTED_MAX_ABS_LIMIT,
        rms_absolute_error_limit=HISTORICAL_EXECUTED_RMS_LIMIT,
    )
    result: Dict[str, Any] = {
        "raw_r01_nominal_actions_content_sha256": content_hash(nominal.tolist()),
        "raw_r01_nominal_actions_array_sha256": _array_hash(nominal),
        "fresh_eager_actions_content_sha256": content_hash(fresh.tolist()),
        "fresh_eager_actions_array_sha256": _array_hash(fresh),
        "fresh_eager_vs_raw_r01_nominal": _path_diagnostic(nominal, fresh),
        "first_five_translation_drift": translation_drift,
        "executed_first_five_action_drift": executed_drift,
        "passed": bool(translation_drift["passed"] and executed_drift["passed"]),
        "delta_comparison": None,
        "raw_r01_witness_actions": None,
        "raw_r01_witness_actions_content_sha256": None,
        "raw_r01_delta_star_physical": None,
    }
    if witness_actions is None:
        return result
    witness = np.asarray(witness_actions, dtype=np.float64)
    if witness.shape != (5, 7) or not np.all(np.isfinite(witness)):
        raise ValueError("historical R01 witness must contain finite 5x7 actions")
    raw_delta = np.zeros((10, 32), dtype=np.float64)
    raw_delta[:5, :3] = witness[:5, :3] - nominal[:5, :3]
    result.update(
        {
            "raw_r01_witness_actions": _array_record(witness, dtype=np.float64),
            "raw_r01_witness_actions_content_sha256": content_hash(witness.tolist()),
            "raw_r01_delta_star_physical": _array_record(
                raw_delta, dtype=np.float64
            ),
            "delta_comparison": _delta_comparison_diagnostic(
                nominal, fresh, witness
            ),
        }
    )
    return result


def _reply_policy_timing(reply: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    timing = reply.get("policy_timing")
    if not isinstance(timing, Mapping):
        return None
    record: Dict[str, Any] = {}
    for key, value in timing.items():
        if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
            value, bool
        ):
            record[str(key)] = float(value)
        elif value is None or isinstance(value, (str, bool)):
            record[str(key)] = value
        else:
            # Timing is descriptive only. Preserve an inspectable representation
            # without allowing an auxiliary telemetry type to gate the science.
            record[str(key)] = repr(value)
    return record


def _policy_timing_control(
    *,
    policy_inference: bool,
    primary_reply: Optional[Mapping[str, Any]] = None,
    duplicate_reply: Optional[Mapping[str, Any]] = None,
    reason: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "descriptive_only": True,
        "policy_inference": policy_inference,
        "primary": (
            _reply_policy_timing(primary_reply)
            if primary_reply is not None
            else None
        ),
        "duplicate": (
            _reply_policy_timing(duplicate_reply)
            if duplicate_reply is not None
            else None
        ),
        "reason": reason,
    }


def _provenance_with_policy_timing(
    provenance: Mapping[str, Any], arms: Mapping[str, Mapping[str, Any]]
) -> Dict[str, Any]:
    result = dict(provenance)
    result["policy_timing"] = {
        "descriptive_only": True,
        "source": "per-reply policy_timing returned by the policy server",
        "arms": {
            name: dict(arms[name].get("controls", {}).get("policy_timing", {}))
            for name in ARMS
        },
    }
    return result


def _bounds_check(actions: np.ndarray, config: R02Config) -> Dict[str, Any]:
    value = np.asarray(actions, dtype=np.float64)
    violations = []
    if value.ndim != 2 or value.shape[0] < 5 or value.shape[1] < 3:
        return {
            "checked": True,
            "passed": False,
            "low": config.translation_action_low,
            "high": config.translation_action_high,
            "violations": [{"reason": f"invalid action shape {value.shape}"}],
            "clipped": False,
        }
    for row in range(5):
        for axis in range(3):
            item = float(value[row, axis])
            if not math.isfinite(item) or item < config.translation_action_low or item > config.translation_action_high:
                violations.append(
                    {"action_index": row, "translation_axis": axis, "value": item}
                )
    return {
        "checked": True,
        "passed": not violations,
        "low": config.translation_action_low,
        "high": config.translation_action_high,
        "violations": violations,
        "clipped": False,
    }


def _trial_evidence_errors(trial: Mapping[str, Any]) -> List[str]:
    """Cross-check gate fields against raw simulator measurement/tracking."""

    errors: List[str] = []
    measurement = trial.get("measurement")
    if not isinstance(measurement, Mapping):
        return ["missing raw simulator measurement"]
    reported_pairs = (
        ("clearance_m", "conservative_clearance_m"),
        ("raw_mujoco_clearance_m", "min_clearance_m"),
    )
    for top_key, raw_key in reported_pairs:
        try:
            top_value = float(trial[top_key])
            raw_value = float(measurement[raw_key])
        except (KeyError, TypeError, ValueError, OverflowError):
            errors.append(f"{top_key} or measurement.{raw_key} is invalid")
            continue
        if not (
            math.isfinite(top_value)
            and math.isfinite(raw_value)
            and math.isclose(top_value, raw_value, rel_tol=0.0, abs_tol=1e-12)
        ):
            errors.append(f"{top_key} conflicts with measurement.{raw_key}")
    if trial.get("contact") is not measurement.get("contact"):
        errors.append("contact conflicts with measurement.contact")
    try:
        top_samples = int(trial.get("measurement_samples", -1))
        raw_samples = int(measurement.get("samples", -1))
    except (TypeError, ValueError, OverflowError):
        top_samples = raw_samples = -1
    if top_samples != raw_samples or raw_samples != EXPECTED_MEASUREMENT_SAMPLES:
        errors.append("measurement sample count conflicts with raw measurement")
    if measurement.get("distance_limit_m") != REGISTERED_DISTANCE_LIMIT_M:
        errors.append("raw measurement distance limit differs from 1 m")
    if measurement.get("conservative_eef_radius_m") != REGISTERED_EEF_RADIUS_M:
        errors.append("raw conservative EEF radius differs from 6 cm")
    try:
        recomputed_clearance = signed_distance_point_to_oriented_box(
            measurement["conservative_eef_center_m"],
            measurement["conservative_obstacle_center_m"],
            np.asarray(
                measurement["conservative_obstacle_rotation_world"],
                dtype=np.float64,
            ).reshape(3, 3),
            measurement["conservative_obstacle_half_size_m"],
        ) - REGISTERED_EEF_RADIUS_M
        stored_conservative = float(measurement["conservative_clearance_m"])
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        errors.append(f"conservative sphere/OBB witness is invalid: {error}")
    else:
        if not math.isclose(
            recomputed_clearance,
            stored_conservative,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            errors.append("conservative clearance does not match stored sphere/OBB witness")
    expected_pair = (
        "crfs_eef_sphere",
        measurement.get("conservative_obstacle_geom"),
    )
    if content_hash(trial.get("minimum_geom_pair")) != content_hash(expected_pair):
        errors.append("minimum_geom_pair conflicts with conservative measurement")
    if content_hash(trial.get("raw_mujoco_minimum_geom_pair")) != content_hash(
        measurement.get("min_pair")
    ):
        errors.append("raw minimum geom pair conflicts with measurement")

    reach = trial.get("reach")
    if not isinstance(reach, Mapping):
        return errors + ["missing raw reach annotation"]
    target_name = reach.get("target_object_name")
    obstacle_name = reach.get("active_obstacle_name")
    if not isinstance(target_name, str) or not isinstance(obstacle_name, str):
        return errors + ["reach target/obstacle identity is invalid"]
    try:
        branch, end, target_maximum, obstacle_maximum = _tracked_reach_snapshots(
            trial,
            target_name=target_name,
            obstacle_name=obstacle_name,
        )
        recomputed_reach = annotate_reach_snapshots(
            branch,
            end,
            executed_actions=EXECUTED_REACH_ACTIONS,
            maximum_target_displacement_m=target_maximum,
            maximum_active_obstacle_displacement_m=obstacle_maximum,
        ).to_dict()
    except (KeyError, TypeError, ValueError, RuntimeError, OverflowError) as error:
        errors.append(f"tracked reach evidence is invalid: {error}")
    else:
        if content_hash(reach) != content_hash(recomputed_reach):
            errors.append("reach annotation conflicts with tracked body motion")
    return errors


def _trial_gate(trial: Mapping[str, Any], config: R02Config) -> Dict[str, Any]:
    reach = trial.get("reach")
    reasons: List[str] = []
    evidence_errors = _trial_evidence_errors(trial)
    if evidence_errors:
        reasons.extend(f"inconsistent_raw_evidence:{error}" for error in evidence_errors)
    if not isinstance(reach, Mapping):
        return {"passed": False, "reasons": ["missing_reach_annotation"]}
    if float(trial.get("clearance_m", float("-inf"))) < config.simulator_safety_margin_m:
        reasons.append("clearance_below_5mm")
    if bool(trial.get("contact", True)):
        reasons.append("forbidden_contact")
    if float(reach.get("reach_progress_m", float("-inf"))) < config.p_min_m:
        reasons.append("progress_below_p_min")
    if float(reach.get("maximum_target_displacement_m", float("inf"))) > config.maximum_target_displacement_m:
        reasons.append("target_motion_above_limit")
    if float(reach.get("maximum_active_obstacle_displacement_m", float("inf"))) > config.maximum_obstacle_displacement_m:
        reasons.append("obstacle_motion_above_limit")
    if int(trial.get("measurement_samples", -1)) != EXPECTED_MEASUREMENT_SAMPLES:
        reasons.append("wrong_measurement_sample_count")
    return {"passed": not reasons, "reasons": reasons}


def _annotated_rollout(
    environment: SafeLiberoCase,
    actions: np.ndarray,
    initial: ReachSnapshot,
    config: R02Config,
) -> Dict[str, Any]:
    if environment.obstacle_name is None:
        raise RuntimeError("active obstacle was not resolved")
    rollout, reach = annotate_reach_rollout(
        environment,
        np.asarray(actions, dtype=np.float64)[:5, :7],
        target_name=config.target_name,
        obstacle_name=environment.obstacle_name,
        initial_snapshot=initial,
    )
    return {**rollout, "reach": reach}


def _evaluated_arm(
    name: str,
    actions: np.ndarray,
    environment: SafeLiberoCase,
    initial: ReachSnapshot,
    config: R02Config,
    *,
    mechanism: str,
    controls: Mapping[str, Any],
    correction: Optional[np.ndarray] = None,
    trace: Optional[Mapping[str, Any]] = None,
    frozen_trace: Optional[Mapping[str, Any]] = None,
    policy_replay_exact: Optional[bool] = None,
    duplicate_actions: Optional[np.ndarray] = None,
    duplicate_trace: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    full_actions = np.asarray(actions, dtype=np.float64)
    bounds = _bounds_check(full_actions, config)
    base = {
        "mechanism": mechanism,
        "applicable": True,
        "controls": dict(controls),
        "correction": _array_record(correction, dtype=np.float32) if correction is not None else None,
        "full_actions": _array_record(full_actions),
        "executed_actions": _array_record(full_actions[:5, :7], dtype=np.float64),
        "bounds": bounds,
        "trace_sha256": _trace_record(trace)["sha256"] if trace is not None else None,
        "preintervention_trace_exact_to_frozen": (
            _same_trace(trace, frozen_trace)
            if trace is not None and frozen_trace is not None
            else None
        ),
        "policy_replay_exact": policy_replay_exact,
        "policy_duplicate_actions": (
            _array_record(duplicate_actions, dtype=np.float64)
            if duplicate_actions is not None
            else None
        ),
        "policy_duplicate_trace": (
            _trace_record(duplicate_trace) if duplicate_trace is not None else None
        ),
    }
    if not bounds["passed"]:
        return {
            **base,
            "status": "bounds_failure",
            "replay_exact": None,
            "repeats": [],
            "gate": {"passed": False, "trial_checks": [], "reason": "sampled_action_out_of_bounds"},
        }
    repeats = [
        _annotated_rollout(environment, full_actions, initial, config)
        for _ in range(config.simulator_repeats)
    ]
    replay_exact = bool(
        repeats
        and all(content_hash(item) == content_hash(repeats[0]) for item in repeats[1:])
    )
    checks = [_trial_gate(trial, config) for trial in repeats]
    passed = bool(replay_exact and all(item["passed"] for item in checks))
    return {
        **base,
        "status": "passed_gate" if passed else "failed_gate",
        "replay_exact": replay_exact,
        "repeats": repeats,
        "gate": {
            "passed": passed,
            "trial_checks": checks,
            "minimum_clearance_m": min(float(item["clearance_m"]) for item in repeats),
            "minimum_progress_m": min(float(item["reach"]["reach_progress_m"]) for item in repeats),
            "maximum_target_displacement_m": max(
                float(item["reach"]["maximum_target_displacement_m"]) for item in repeats
            ),
            "maximum_obstacle_displacement_m": max(
                float(item["reach"]["maximum_active_obstacle_displacement_m"]) for item in repeats
            ),
            "any_contact": any(bool(item["contact"]) for item in repeats),
        },
    }


def _nonrollout_arm(
    *,
    mechanism: str,
    status: str,
    reason: str,
    controls: Optional[Mapping[str, Any]] = None,
    correction: Optional[np.ndarray] = None,
    applicable: Optional[bool] = None,
) -> Dict[str, Any]:
    arm_controls = dict(controls or {})
    arm_controls.setdefault(
        "policy_timing",
        _policy_timing_control(
            policy_inference=False,
            reason=reason,
        ),
    )
    return {
        "mechanism": mechanism,
        "applicable": (
            status != "not_applicable_no_r01_witness"
            if applicable is None
            else bool(applicable)
        ),
        "status": status,
        "controls": arm_controls,
        "correction": _array_record(correction, dtype=np.float32) if correction is not None else None,
        "full_actions": None,
        "executed_actions": None,
        "bounds": {"checked": False, "passed": None, "violations": [], "clipped": False},
        "trace_sha256": None,
        "preintervention_trace_exact_to_frozen": None,
        "policy_replay_exact": None,
        "policy_duplicate_actions": None,
        "policy_duplicate_trace": None,
        "replay_exact": None,
        "repeats": [],
        "gate": {"passed": None, "trial_checks": [], "reason": reason},
    }


def _not_evaluated_arms(
    frozen_arm: Mapping[str, Any],
    *,
    eligible: bool,
    direct_arm: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Retain observed arms and make every post-failure arm explicit."""

    mechanisms = {
        "direct_witness": "R01 direct action witness",
        "random_residual": "endpoint-free equal-L2 random distributed residual",
        "analytic_geometry_residual": "analytic sphere/OBB distributed residual",
        "oracle_residual": "Delta_star distributed residual",
        "bridge_diagnostic": "Delta_star one-shot bridge edit diagnostic",
    }
    arms: Dict[str, Any] = {"frozen": dict(frozen_arm)}
    if direct_arm is not None:
        arms["direct_witness"] = dict(direct_arm)
        unrun = FLOW_ARMS
    else:
        unrun = ARMS[1:]
    for name in unrun:
        arms[name] = _nonrollout_arm(
            mechanism=mechanisms[name],
            status=NOT_EVALUATED_AFTER_RECONFIRMATION_FAILURE,
            reason="R02 stopped after a registered reconfirmation failure",
            applicable=eligible,
        )
    return {name: arms[name] for name in ARMS}


def _no_witness_arms(frozen_arm: Mapping[str, Any]) -> Dict[str, Any]:
    arms: Dict[str, Any] = {"frozen": dict(frozen_arm)}
    for name in ARMS[1:]:
        arms[name] = _nonrollout_arm(
            mechanism={
                "direct_witness": "R01 direct action witness",
                "random_residual": "endpoint-free equal-L2 random distributed residual",
                "analytic_geometry_residual": "analytic sphere/OBB distributed residual",
                "oracle_residual": "Delta_star distributed residual",
                "bridge_diagnostic": "Delta_star one-shot bridge edit diagnostic",
            }[name],
            status="not_applicable_no_r01_witness",
            reason="validated R01 case has no changed-action p_min witness",
        )
    return arms


def _direction_bundle(
    r01: Mapping[str, Any],
    config: R02Config,
    fresh_eager_physical: np.ndarray,
    predicted_clean_physical: np.ndarray,
    branch_rollout: Mapping[str, Any],
) -> Tuple[
    Any,
    Dict[str, Any],
    Dict[str, Optional[np.ndarray]],
    Dict[str, Optional[str]],
    Dict[str, Any],
]:
    """Construct the registered physical/model directions from raw evidence."""

    from .r02_directions import (
        analytic_d_opt_ascent_direction,
        deterministic_equal_l2_random_direction,
        select_r01_changed_p_min_witness,
    )

    witness = select_r01_changed_p_min_witness(r01)
    witness_actions = np.asarray(witness.witness_prefix, dtype=np.float64)
    fresh_eager = np.asarray(fresh_eager_physical, dtype=np.float64)
    if witness_actions.shape != (5, 7) or fresh_eager.shape != (10, 7):
        raise ValueError(
            "fresh-paired oracle requires a 5x7 R01 witness and 10x7 eager actions"
        )
    if not np.all(np.isfinite(witness_actions)) or not np.all(np.isfinite(fresh_eager)):
        raise ValueError("fresh-paired oracle actions must be finite")
    delta_physical = np.zeros(
        (config.oracle.action_horizon, config.oracle.action_dim), dtype=np.float64
    )
    delta_physical[:5, :3] = (
        witness_actions[:5, :3] - fresh_eager[:5, :3]
    )
    delta_model = np.zeros_like(delta_physical)
    delta_model[:5, :3] = (
        delta_physical[:5, :3]
        / np.asarray(config.action_scale, dtype=np.float64)[None, :]
    )
    if float(np.linalg.norm(delta_model)) <= 0.0:
        raise ValueError("fresh-paired Delta_star_model must have nonzero norm")
    directions: Dict[str, Optional[np.ndarray]] = {
        "delta_star_physical": delta_physical,
        "delta_star_model": delta_model,
        "random_model": None,
        "analytic_geometry_model": None,
    }
    failures: Dict[str, Optional[str]] = {"random_model": None, "analytic_geometry_model": None}
    diagnostics: Dict[str, Any] = {"analytic_geometry": None}
    try:
        directions["random_model"] = np.asarray(
            deterministic_equal_l2_random_direction(
                delta_model,
                base_physical_prefix=predicted_clean_physical[:5, :7],
                action_scale=config.action_scale,
                seed=witness.random_seed,
                action_low=witness.action_low,
                action_high=witness.action_high,
            ),
            dtype=np.float64,
        )
    except ValueError as error:
        failures["random_model"] = str(error)

    predicted = np.asarray(predicted_clean_physical, dtype=np.float64)
    if predicted.ndim != 2 or predicted.shape[0] < 5 or predicted.shape[1] < 3:
        failures["analytic_geometry_model"] = (
            f"predicted_clean_physical has invalid shape {predicted.shape}"
        )
    else:
        try:
            analytic = analytic_d_opt_ascent_direction(
                predicted[:5, :7],
                start_eef_center_m=branch_rollout["start_eef_center_m"],
                response_matrix_m_per_action=config.oracle.response_matrix_m_per_action,
                obstacle_boxes=branch_rollout["branch_obstacle_boxes"],
                eef_radius_m=config.oracle.eef_radius_m,
                action_scale=config.action_scale,
                model_l2_budget=float(np.linalg.norm(delta_model)),
                samples_per_segment=config.samples_per_segment,
                model_action_horizon=config.oracle.action_horizon,
                model_action_dimension=config.oracle.action_dim,
            )
            direction_value = getattr(analytic, "direction_model", analytic)
            directions["analytic_geometry_model"] = np.asarray(direction_value, dtype=np.float64)
            diagnostics["analytic_geometry"] = (
                analytic.to_dict() if hasattr(analytic, "to_dict") else None
            )
        except (KeyError, TypeError, ValueError) as error:
            failures["analytic_geometry_model"] = str(error)

    pointer = _witness_pointer(witness, np.asarray(witness.witness_prefix, dtype=np.float64))
    return witness, pointer, directions, failures, diagnostics


def _direction_records(
    directions: Mapping[str, Optional[np.ndarray]],
    failures: Mapping[str, Optional[str]],
    *,
    witness_pointer: Optional[Mapping[str, Any]],
    config: R02Config,
    diagnostics: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    records = {
        key: _array_record(value, dtype=np.float64) if value is not None else None
        for key, value in directions.items()
    }
    norms = {
        key: float(np.linalg.norm(value)) if value is not None else None
        for key, value in directions.items()
    }
    return {
        "witness_pointer": dict(witness_pointer) if witness_pointer is not None else None,
        "direction_reference": config.direction_reference,
        "direction_semantics_decision": config.direction_semantics_decision,
        "direction_semantics_decision_sha256": (
            config.direction_semantics_decision_sha256
        ),
        "translation_mask": "first five actions x first three translation channels; all other entries zero",
        "normalization": "scale-only displacement conversion; no normalization mean subtraction",
        "action_scale_physical_per_model": list(config.action_scale),
        "arrays": records,
        "l2_norms": norms,
        "construction_failures": dict(failures),
        "diagnostics": dict(diagnostics or {"analytic_geometry": None}),
    }


def _base_provenance(
    case: Mapping[str, Any],
    config: R02Config,
    *,
    root: Path,
    input_manifest_sha256: str,
    raw_r01_path: Path,
    raw_r01_sha256: str,
    noise: np.ndarray,
) -> Dict[str, Any]:
    git_commit, git_dirty = _git_state(root)
    return {
        "evidence_tier": "real_safelibero_r02_oracle_flow_preliminary",
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "baseline_commit": BASELINE_COMMIT,
        "python_version": platform.python_version(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "device": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "case_record": dict(case),
        "case_record_sha256": content_hash(dict(case)),
        "input_manifest_sha256": input_manifest_sha256,
        "task_suite": case.get("task_suite"),
        "safety_level": case.get("safety_level"),
        "task_index": case.get("task_index"),
        "episode_index": case.get("episode_index"),
        "group_id": case.get("group_id"),
        "environment_seed": case.get("environment_seed"),
        "policy_seed": case.get("policy_seed"),
        "random_control_seed": case.get("random_control_seed"),
        "checkpoint_id": config.oracle.checkpoint_id,
        "checkpoint_sha256": config.oracle.checkpoint_sha256,
        "r01_summary_sha256": config.r01_summary_sha256,
        "r01_case_result_path": str(raw_r01_path),
        "r01_case_result_sha256": raw_r01_sha256,
        "sampler_parity_path": config.parity_artifact_path,
        "sampler_parity_sha256": config.parity_artifact_sha256,
        "direction_reference": config.direction_reference,
        "direction_semantics_decision": config.direction_semantics_decision,
        "direction_semantics_decision_sha256": (
            config.direction_semantics_decision_sha256
        ),
        "noise": _array_record(noise, dtype=np.float32),
        "sampler_steps": config.oracle.sampler_steps,
        "intervention_step": config.oracle.intervention_step,
        "model_action_horizon": config.oracle.action_horizon,
        "executed_action_horizon": config.oracle.executed_prefix,
        "model_action_dimension": config.oracle.action_dim,
        "action_frame": "world-frame OSC translation delta",
        "policy_action_space": "unnormalized LIBERO controller actions",
        "normalization_space": (
            "model correction uses scale only; absolute predicted-clean action uses full inverse transform"
        ),
        "normalization_asset_sha256": NORMALIZATION_ASSET_SHA256,
        "normalization_action_scale": list(config.action_scale),
        "translation_action_bounds": [
            config.translation_action_low,
            config.translation_action_high,
        ],
        "clipping_policy": "fail_without_clipping",
        "d_opt_model": "frozen H04 response plus sampled EEF-sphere/branch-OBB geometry",
        "d_sim_model": "physics-substep 6cm EEF-sphere/active-OBB clearance plus contact",
        "eef_radius_m": config.oracle.eef_radius_m,
        "distance_limit_m": config.oracle.distance_limit_m,
        "response_matrix_m_per_action": [
            list(row) for row in config.oracle.response_matrix_m_per_action or ()
        ],
        "simulator_safety_margin_m": config.simulator_safety_margin_m,
        "minimum_progress_m": config.p_min_m,
        "maximum_target_displacement_m": config.maximum_target_displacement_m,
        "maximum_obstacle_displacement_m": config.maximum_obstacle_displacement_m,
        "simulator_repeats": config.simulator_repeats,
        "measurement_samples_per_trial": EXPECTED_MEASUREMENT_SAMPLES,
        "required_arms": list(ARMS),
    }


def _no_witness_result(
    case: Mapping[str, Any],
    config: R02Config,
    *,
    config_hash: str,
    provenance: Mapping[str, Any],
    raw_r01_path: Path,
    raw_r01: Mapping[str, Any],
    raw_r01_sha256: str,
    pairing: Mapping[str, Any],
    frozen_arm: Mapping[str, Any],
) -> Dict[str, Any]:
    arms = _no_witness_arms(frozen_arm)
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "gate": GATE,
        "case_id": case["case_id"],
        "run_id": config.oracle.run_id,
        "status": "not_applicable_no_r01_witness",
        "config_hash": config_hash,
        "source_evidence": {
            "r01_summary_sha256": config.r01_summary_sha256,
            "r01_ordered_result_set_digest": ACCEPTED_R01_ORDERED_RESULTS_SHA256,
            "r01_case_path": str(raw_r01_path),
            "r01_case_sha256": raw_r01_sha256,
            "r01_case_validator": "validate_endpoint_free_result:passed",
            "r01_case_status": raw_r01["status"],
            "r01_selected_p_min_changed_witness": None,
            "selected_witness_pointer": None,
            "sampler_parity_sha256": config.parity_artifact_sha256,
            "sampler_parity_status": "passed",
            "direction_reference": config.direction_reference,
            "direction_semantics_decision": config.direction_semantics_decision,
            "direction_semantics_decision_sha256": (
                config.direction_semantics_decision_sha256
            ),
        },
        "provenance": _provenance_with_policy_timing(provenance, arms),
        "pairing": dict(pairing),
        "directions": {
            "witness_pointer": None,
            "direction_reference": config.direction_reference,
            "direction_semantics_decision": config.direction_semantics_decision,
            "direction_semantics_decision_sha256": (
                config.direction_semantics_decision_sha256
            ),
            "translation_mask": "not applicable: no R01 changed-action p_min witness",
            "normalization": "not applicable",
            "action_scale_physical_per_model": list(config.action_scale),
            "arrays": {
                "delta_star_physical": None,
                "delta_star_model": None,
                "random_model": None,
                "analytic_geometry_model": None,
            },
            "l2_norms": {
                "delta_star_physical": None,
                "delta_star_model": None,
                "random_model": None,
                "analytic_geometry_model": None,
            },
            "construction_failures": {
                "random_model": "not applicable: no witness",
                "analytic_geometry_model": "not applicable: no witness",
            },
            "diagnostics": {"analytic_geometry": None},
        },
        "arms": arms,
        "outcome": {
            "population": "original_20_collision_population_no_r01_witness",
            "r01_feasible_conditioned": False,
            "nominal_collision_reproduced": True,
            "direct_witness_reconfirmed": None,
            "population_mismatch_reason": None,
            "arm_gate_pass": {
                "frozen": bool(frozen_arm["gate"]["passed"]),
                **{name: None for name in ARMS[1:]},
            },
            "arm_status": {
                "frozen": frozen_arm["status"],
                **{
                    name: "not_applicable_no_r01_witness" for name in ARMS[1:]
                },
            },
        },
    }


def _unconstructed_direction_records(
    config: R02Config,
    *,
    witness_pointer: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    reason = "not constructed after nominal collision reconfirmation failure"
    return {
        "witness_pointer": (
            dict(witness_pointer) if witness_pointer is not None else None
        ),
        "direction_reference": config.direction_reference,
        "direction_semantics_decision": config.direction_semantics_decision,
        "direction_semantics_decision_sha256": (
            config.direction_semantics_decision_sha256
        ),
        "translation_mask": (
            "first five actions x first three translation channels; all other entries zero"
            if witness_pointer is not None
            else "not applicable: no R01 changed-action p_min witness"
        ),
        "normalization": (
            "scale-only displacement conversion; no normalization mean subtraction"
            if witness_pointer is not None
            else "not applicable"
        ),
        "action_scale_physical_per_model": list(config.action_scale),
        "arrays": {
            "delta_star_physical": None,
            "delta_star_model": None,
            "random_model": None,
            "analytic_geometry_model": None,
        },
        "l2_norms": {
            "delta_star_physical": None,
            "delta_star_model": None,
            "random_model": None,
            "analytic_geometry_model": None,
        },
        "construction_failures": {
            "random_model": reason,
            "analytic_geometry_model": reason,
        },
        "diagnostics": {"analytic_geometry": None},
    }


def _reconfirmation_failure_result(
    case: Mapping[str, Any],
    config: R02Config,
    *,
    status: str,
    eligible: bool,
    config_hash: str,
    provenance: Mapping[str, Any],
    raw_r01_path: Path,
    raw_r01: Mapping[str, Any],
    raw_r01_sha256: str,
    pairing: Mapping[str, Any],
    frozen_arm: Mapping[str, Any],
    r01_pointer: Optional[Mapping[str, Any]],
    witness_pointer: Optional[Mapping[str, Any]],
    direction_records: Optional[Mapping[str, Any]] = None,
    direct_arm: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a terminal, non-passing artifact from all observations made so far."""

    if status not in {NOMINAL_RECONFIRMATION_FAILURE, DIRECT_RECONFIRMATION_FAILURE}:
        raise ValueError(f"unsupported R02 reconfirmation failure status: {status!r}")
    if (status == DIRECT_RECONFIRMATION_FAILURE) != (direct_arm is not None):
        raise ValueError("direct reconfirmation status and observed direct arm must agree")
    arms = _not_evaluated_arms(
        frozen_arm,
        eligible=eligible,
        direct_arm=direct_arm,
    )
    gate_pass = {
        name: (
            bool(arms[name]["gate"]["passed"])
            if arms[name]["status"] in EVALUATED_ARM_STATUSES
            else None
        )
        for name in ARMS
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "gate": GATE,
        "case_id": case["case_id"],
        "run_id": config.oracle.run_id,
        "status": status,
        "config_hash": config_hash,
        "source_evidence": {
            "r01_summary_sha256": config.r01_summary_sha256,
            "r01_ordered_result_set_digest": ACCEPTED_R01_ORDERED_RESULTS_SHA256,
            "r01_case_path": str(raw_r01_path),
            "r01_case_sha256": raw_r01_sha256,
            "r01_case_validator": "validate_endpoint_free_result:passed",
            "r01_case_status": raw_r01["status"],
            "r01_selected_p_min_changed_witness": (
                dict(r01_pointer) if r01_pointer is not None else None
            ),
            "selected_witness_pointer": (
                dict(witness_pointer) if witness_pointer is not None else None
            ),
            "sampler_parity_sha256": config.parity_artifact_sha256,
            "sampler_parity_status": "passed",
            "direction_reference": config.direction_reference,
            "direction_semantics_decision": config.direction_semantics_decision,
            "direction_semantics_decision_sha256": (
                config.direction_semantics_decision_sha256
            ),
        },
        "provenance": _provenance_with_policy_timing(provenance, arms),
        "pairing": dict(pairing),
        "directions": (
            dict(direction_records)
            if direction_records is not None
            else _unconstructed_direction_records(
                config, witness_pointer=witness_pointer
            )
        ),
        "arms": arms,
        "outcome": {
            "population": (
                "r01_changed_action_p_min_witness_conditioned"
                if eligible
                else "original_20_collision_population_no_r01_witness"
            ),
            "r01_feasible_conditioned": eligible,
            "nominal_collision_reproduced": (
                status != NOMINAL_RECONFIRMATION_FAILURE
            ),
            "direct_witness_reconfirmed": (
                False if status == DIRECT_RECONFIRMATION_FAILURE else None
            ),
            "population_mismatch_reason": status,
            "arm_gate_pass": gate_pass,
            "arm_status": {name: arms[name]["status"] for name in ARMS},
        },
    }


def _validate_exact_pairing(
    pairing: Mapping[str, Any],
) -> Tuple[
    List[str],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
]:
    errors: List[str] = []
    if pairing.get("applicable") is not True or pairing.get("passed") is not True:
        errors.append("R02 artifact requires passing exact policy pairing")
    for key in (
        "duplicate_eager_actions_exact",
        "duplicate_eager_trace_exact",
        "branch_snapshot_equals_r01_exact",
        "historical_r01_drift_within_frozen_limits",
        "order_contamination_check_exact",
        "eager_order_contamination_check_exact",
    ):
        if pairing.get(key) is not True:
            errors.append(f"pairing.{key} must be true")
    branch_snapshot = pairing.get("branch_snapshot")
    r01_branch_snapshot = pairing.get("r01_branch_snapshot")
    if not isinstance(branch_snapshot, Mapping):
        errors.append("pairing.branch_snapshot must be an object")
    if not isinstance(r01_branch_snapshot, Mapping):
        errors.append("pairing.r01_branch_snapshot must be an object")
    if (
        isinstance(branch_snapshot, Mapping)
        and isinstance(r01_branch_snapshot, Mapping)
        and dict(branch_snapshot) != dict(r01_branch_snapshot)
    ):
        errors.append("R02 branch snapshot differs from the immutable R01 branch")
    compiled, item_errors = _validate_array_record(
        pairing.get("compiled_actions"),
        name="pairing.compiled_actions",
        shape=(10, 7),
    )
    errors.extend(item_errors)
    eager, item_errors = _validate_array_record(
        pairing.get("eager_actions"), name="pairing.eager_actions", shape=(10, 7)
    )
    errors.extend(item_errors)
    duplicate, item_errors = _validate_array_record(
        pairing.get("duplicate_eager_actions"),
        name="pairing.duplicate_eager_actions",
        shape=(10, 7),
    )
    errors.extend(item_errors)
    r01_nominal, item_errors = _validate_array_record(
        pairing.get("r01_nominal_actions"),
        name="pairing.r01_nominal_actions",
        shape=(5, 7),
    )
    errors.extend(item_errors)
    historical_witness: Optional[np.ndarray] = None
    historical_delta: Optional[np.ndarray] = None
    historical = pairing.get("historical_r01_diagnostic")
    expected_historical_fields = {
        "raw_r01_nominal_actions_content_sha256",
        "raw_r01_nominal_actions_array_sha256",
        "fresh_eager_actions_content_sha256",
        "fresh_eager_actions_array_sha256",
        "fresh_eager_vs_raw_r01_nominal",
        "first_five_translation_drift",
        "executed_first_five_action_drift",
        "passed",
        "delta_comparison",
        "raw_r01_witness_actions",
        "raw_r01_witness_actions_content_sha256",
        "raw_r01_delta_star_physical",
    }
    if not isinstance(historical, Mapping) or set(historical) != expected_historical_fields:
        errors.append("pairing historical R01 diagnostic has incorrect fields")
        historical = {}
    if r01_nominal is not None:
        nominal_content_sha = content_hash(
            np.asarray(r01_nominal, dtype=np.float64).tolist()
        )
        if historical.get("raw_r01_nominal_actions_content_sha256") != nominal_content_sha:
            errors.append("historical R01 nominal content hash conflicts with raw actions")
        if historical.get("raw_r01_nominal_actions_array_sha256") != _array_hash(
            np.asarray(r01_nominal, dtype=np.float64)
        ):
            errors.append("historical R01 nominal array hash conflicts with raw actions")
    if eager is not None:
        fresh_prefix = np.asarray(eager[:5, :7], dtype=np.float64)
        if historical.get("fresh_eager_actions_content_sha256") != content_hash(
            fresh_prefix.tolist()
        ):
            errors.append("historical fresh eager content hash conflicts with actions")
        if historical.get("fresh_eager_actions_array_sha256") != _array_hash(
            fresh_prefix
        ):
            errors.append("historical fresh eager array hash conflicts with actions")
    witness_record = historical.get("raw_r01_witness_actions")
    witness_content_sha = historical.get("raw_r01_witness_actions_content_sha256")
    delta_record = historical.get("raw_r01_delta_star_physical")
    delta_comparison = historical.get("delta_comparison")
    if witness_record is None:
        if (
            witness_content_sha is not None
            or delta_record is not None
            or delta_comparison is not None
        ):
            errors.append("historical R01 witness diagnostics must be jointly null")
    else:
        historical_witness, item_errors = _validate_array_record(
            witness_record,
            name="pairing.historical_r01_diagnostic.raw_r01_witness_actions",
            shape=(5, 7),
        )
        errors.extend(item_errors)
        historical_delta, item_errors = _validate_array_record(
            delta_record,
            name="pairing.historical_r01_diagnostic.raw_r01_delta_star_physical",
            shape=(10, 32),
        )
        errors.extend(item_errors)
        if historical_witness is not None:
            if witness_content_sha != content_hash(historical_witness.tolist()):
                errors.append("historical R01 witness content hash conflicts with actions")
            if r01_nominal is not None:
                expected_historical_delta = np.zeros((10, 32), dtype=np.float64)
                expected_historical_delta[:5, :3] = (
                    historical_witness[:5, :3] - r01_nominal[:5, :3]
                )
                if not np.array_equal(historical_delta, expected_historical_delta):
                    errors.append(
                        "historical R01 Delta_star is not witness minus R01 nominal"
                    )
                if not np.array_equal(
                    historical_witness[:5, 3:], r01_nominal[:5, 3:]
                ):
                    errors.append("historical R01 witness changes nontranslation actions")
            expected_comparison_fields = {
                "raw_model_l2",
                "fresh_model_l2",
                "fresh_minus_raw_model_l2",
                "cosine",
                "angle_degrees",
            }
            if (
                not isinstance(delta_comparison, Mapping)
                or set(delta_comparison) != expected_comparison_fields
            ):
                errors.append("historical versus fresh Delta comparison has incorrect fields")
            elif r01_nominal is not None and eager is not None:
                try:
                    expected_comparison = _delta_comparison_diagnostic(
                        r01_nominal,
                        eager[:5, :7],
                        historical_witness,
                    )
                except ValueError as error:
                    errors.append(f"historical versus fresh Delta comparison failed: {error}")
                else:
                    for key, expected in expected_comparison.items():
                        claim = delta_comparison.get(key)
                        if (
                            not isinstance(claim, (int, float))
                            or isinstance(claim, bool)
                            or not math.isfinite(float(claim))
                            or not math.isclose(
                                float(claim),
                                expected,
                                rel_tol=0.0,
                                abs_tol=1e-12,
                            )
                        ):
                            errors.append(
                                f"historical versus fresh Delta comparison {key} conflicts with raw actions"
                            )
    final_compiled, item_errors = _validate_array_record(
        pairing.get("final_compiled_actions"),
        name="pairing.final_compiled_actions",
        shape=(10, 7),
    )
    errors.extend(item_errors)
    final_eager, item_errors = _validate_array_record(
        pairing.get("final_eager_actions"),
        name="pairing.final_eager_actions",
        shape=(10, 7),
    )
    errors.extend(item_errors)
    errors.extend(
        _validate_trace_record(pairing.get("eager_trace"), name="pairing.eager_trace")
    )
    errors.extend(
        _validate_trace_record(
            pairing.get("duplicate_eager_trace"), name="pairing.duplicate_eager_trace"
        )
    )
    errors.extend(
        _validate_trace_record(
            pairing.get("final_eager_trace"), name="pairing.final_eager_trace"
        )
    )
    primary_trace = pairing.get("eager_trace")
    duplicate_trace = pairing.get("duplicate_eager_trace")
    if isinstance(primary_trace, Mapping) and isinstance(duplicate_trace, Mapping):
        if primary_trace.get("sha256") != duplicate_trace.get("sha256"):
            errors.append("duplicate eager traces differ despite pairing claim")
        leaves = primary_trace.get("leaves")
        required_trace = {
            # np.ascontiguousarray promotes scalar leaves to length-one arrays.
            "step_index": (1,),
            "time": (1,),
            "x_t": (10, 32),
            "v_base": (10, 32),
            "predicted_clean": (10, 32),
            "predicted_clean_physical": (10, 7),
        }
        if not isinstance(leaves, Mapping) or not set(required_trace).issubset(leaves):
            errors.append("eager trace lacks the registered midpoint fields")
        else:
            trace_arrays: Dict[str, np.ndarray] = {}
            for key, shape in required_trace.items():
                array, trace_errors = _validate_array_record(
                    leaves[key], name=f"pairing.eager_trace.leaves.{key}", shape=shape
                )
                errors.extend(trace_errors)
                if array is not None:
                    trace_arrays[key] = array
            if "step_index" in trace_arrays and int(
                trace_arrays["step_index"].reshape(-1)[0]
            ) != 5:
                errors.append("eager trace must be captured at sampler step five")
            if "time" in trace_arrays and not math.isclose(
                float(trace_arrays["time"].reshape(-1)[0]),
                0.5,
                rel_tol=0.0,
                abs_tol=1e-6,
            ):
                errors.append("eager trace must be captured at t=0.5")
    if eager is not None and duplicate is not None and not np.array_equal(eager, duplicate):
        errors.append("duplicate eager actions differ despite pairing claim")
    drift = historical.get("fresh_eager_vs_raw_r01_nominal")
    if not isinstance(drift, Mapping):
        errors.append("fresh eager versus historical R01 drift diagnostic is missing")
    elif eager is not None and r01_nominal is not None:
        expected_drift = _path_diagnostic(r01_nominal, eager[:5, :7])
        if dict(drift) != expected_drift:
            errors.append("historical R01 drift diagnostic conflicts with raw actions")
        expected_translation_drift = _bounded_drift_diagnostic(
            r01_nominal[:, :3],
            eager[:5, :3],
            maximum_absolute_error_limit=HISTORICAL_TRANSLATION_MAX_ABS_LIMIT,
            rms_absolute_error_limit=HISTORICAL_TRANSLATION_RMS_LIMIT,
        )
        expected_executed_drift = _bounded_drift_diagnostic(
            r01_nominal,
            eager[:5, :7],
            maximum_absolute_error_limit=HISTORICAL_EXECUTED_MAX_ABS_LIMIT,
            rms_absolute_error_limit=HISTORICAL_EXECUTED_RMS_LIMIT,
        )
        if historical.get("first_five_translation_drift") != expected_translation_drift:
            errors.append("historical translation drift diagnostic conflicts with raw actions")
        if historical.get("executed_first_five_action_drift") != expected_executed_drift:
            errors.append("historical executed-action drift diagnostic conflicts with raw actions")
        expected_historical_pass = bool(
            expected_translation_drift["passed"]
            and expected_executed_drift["passed"]
        )
        if historical.get("passed") is not expected_historical_pass:
            errors.append("historical R01 drift pass claim conflicts with raw actions")
        if pairing.get("historical_r01_drift_within_frozen_limits") is not expected_historical_pass:
            errors.append("historical R01 pairing gate conflicts with raw actions")
        if not expected_historical_pass:
            errors.append("historical R01 drift exceeds frozen ADR-0010 limits")
    if compiled is not None and final_compiled is not None and not np.array_equal(
        compiled, final_compiled
    ):
        errors.append("final compiled sampler differs after intervention requests")
    if eager is not None and final_eager is not None and not np.array_equal(
        eager, final_eager
    ):
        errors.append("final eager sampler differs after intervention requests")
    final_eager_trace = pairing.get("final_eager_trace")
    if isinstance(primary_trace, Mapping) and isinstance(final_eager_trace, Mapping):
        if primary_trace.get("sha256") != final_eager_trace.get("sha256"):
            errors.append("final eager trace differs after intervention requests")
    diagnostic = pairing.get("compiled_vs_eager_diagnostic")
    if not isinstance(diagnostic, Mapping):
        errors.append("compiled/eager path seam must be retained as a diagnostic")
    elif compiled is not None and eager is not None:
        expected_diagnostic = _path_diagnostic(compiled, eager)
        if diagnostic != expected_diagnostic:
            errors.append("compiled/eager diagnostic conflicts with stored actions")
    return (
        errors,
        compiled,
        eager,
        r01_nominal,
        historical_witness,
        historical_delta,
    )


def _recompute_serialized_arm_gate(
    arm: Mapping[str, Any],
    *,
    repeat_count: int,
    margin: float,
    p_min: float,
    target_limit: float,
    obstacle_limit: float,
) -> Tuple[Optional[bool], List[str]]:
    errors: List[str] = []
    repeats = arm.get("repeats")
    if not isinstance(repeats, list) or len(repeats) != repeat_count:
        return None, [f"arm must contain exactly {repeat_count} repeats"]
    exact = bool(
        repeats
        and all(content_hash(item) == content_hash(repeats[0]) for item in repeats[1:])
    )
    if arm.get("replay_exact") != exact:
        errors.append("arm replay_exact conflicts with raw repeats")
    trial_passes = []
    for index, trial in enumerate(repeats):
        if not isinstance(trial, Mapping):
            errors.append(f"repeat {index} must be an object")
            trial_passes.append(False)
            continue
        reach = trial.get("reach")
        evidence_errors = _trial_evidence_errors(trial)
        errors.extend(
            f"repeat {index} raw evidence: {error}" for error in evidence_errors
        )
        passed = bool(
            not evidence_errors
            and isinstance(reach, Mapping)
            and float(trial.get("clearance_m", float("-inf"))) >= margin
            and not bool(trial.get("contact", True))
            and int(trial.get("measurement_samples", -1))
            == EXPECTED_MEASUREMENT_SAMPLES
            and float(reach.get("reach_progress_m", float("-inf"))) >= p_min
            and float(reach.get("maximum_target_displacement_m", float("inf")))
            <= target_limit
            and float(
                reach.get("maximum_active_obstacle_displacement_m", float("inf"))
            )
            <= obstacle_limit
        )
        trial_passes.append(passed)
    return bool(exact and all(trial_passes)), errors


def _validate_not_evaluated_arm(
    arm: Any,
    *,
    name: str,
    applicable: bool,
) -> List[str]:
    errors: List[str] = []
    if not isinstance(arm, Mapping):
        return [f"arm {name} must be an object"]
    if arm.get("status") != NOT_EVALUATED_AFTER_RECONFIRMATION_FAILURE:
        errors.append(
            f"arm {name} must be {NOT_EVALUATED_AFTER_RECONFIRMATION_FAILURE}"
        )
    if arm.get("applicable") is not applicable:
        errors.append(f"arm {name} applicability conflicts with the R01 subset")
    for field in (
        "correction",
        "full_actions",
        "executed_actions",
        "trace_sha256",
        "preintervention_trace_exact_to_frozen",
        "policy_replay_exact",
        "policy_duplicate_actions",
        "policy_duplicate_trace",
        "replay_exact",
    ):
        if arm.get(field) is not None:
            errors.append(f"not-evaluated arm {name} cannot contain {field}")
    if arm.get("repeats") != []:
        errors.append(f"not-evaluated arm {name} cannot contain simulator evidence")
    bounds = arm.get("bounds")
    if not isinstance(bounds, Mapping) or dict(bounds) != {
        "checked": False,
        "passed": None,
        "violations": [],
        "clipped": False,
    }:
        errors.append(f"not-evaluated arm {name} must retain an unchecked bounds record")
    gate = arm.get("gate")
    if (
        not isinstance(gate, Mapping)
        or gate.get("passed") is not None
        or gate.get("trial_checks") != []
        or not isinstance(gate.get("reason"), str)
        or not gate.get("reason")
    ):
        errors.append(f"not-evaluated arm {name} must retain a reason and no gate claim")
    return errors


def validate_r02_result(value: Mapping[str, Any]) -> List[str]:
    """Recompute completion and simulator gates from the serialized evidence."""

    errors: List[str] = []
    missing = RESULT_FIELDS - set(value)
    unexpected = set(value) - RESULT_FIELDS
    if missing:
        errors.append(f"missing top-level fields: {sorted(missing)}")
    if unexpected:
        errors.append(f"unexpected top-level fields: {sorted(unexpected)}")
    if value.get("schema_version") != SCHEMA_VERSION or value.get("gate") != GATE:
        errors.append("result must be an R02 schema_version 1.0 artifact")
    if value.get("artifact_type") != ARTIFACT_TYPE:
        errors.append(f"artifact_type must be {ARTIFACT_TYPE}")
    if value.get("status") not in FINAL_STATUSES:
        errors.append(f"invalid R02 final status: {value.get('status')!r}")
    for key in ("case_id", "run_id"):
        if not isinstance(value.get(key), str) or not value.get(key):
            errors.append(f"{key} must be a non-empty string")
    if not _is_sha256(value.get("config_hash")):
        errors.append("config_hash must be a lowercase SHA-256")

    source = value.get("source_evidence")
    if not isinstance(source, Mapping):
        errors.append("source_evidence must be an object")
    else:
        for key in ("r01_summary_sha256", "r01_case_sha256", "sampler_parity_sha256"):
            if not _is_sha256(source.get(key)):
                errors.append(f"source_evidence.{key} must be a lowercase SHA-256")
        if source.get("r01_summary_sha256") != ACCEPTED_R01_SUMMARY_SHA256:
            errors.append("source evidence is not bound to accepted R01")
        if source.get("r01_ordered_result_set_digest") != ACCEPTED_R01_ORDERED_RESULTS_SHA256:
            errors.append("source evidence has the wrong R01 ordered result digest")
        if source.get("r01_case_validator") != "validate_endpoint_free_result:passed":
            errors.append("raw R01 validator must pass")
        if source.get("sampler_parity_status") != "passed":
            errors.append("sampler parity must pass")
        if source.get("direction_reference") != DIRECTION_REFERENCE:
            errors.append("source evidence direction reference differs from ADR-0013")
        if source.get("direction_semantics_decision") != DIRECTION_SEMANTICS_DECISION:
            errors.append("source evidence direction semantics decision differs")
        if (
            source.get("direction_semantics_decision_sha256")
            != DIRECTION_SEMANTICS_DECISION_SHA256
        ):
            errors.append("source evidence direction semantics hash differs")

    provenance = value.get("provenance")
    required_provenance = {
        "git_commit",
        "git_dirty",
        "baseline_commit",
        "host",
        "device",
        "slurm_job_id",
        "slurm_array_task_id",
        "partition",
        "case_record",
        "case_record_sha256",
        "input_manifest_sha256",
        "environment_seed",
        "policy_seed",
        "random_control_seed",
        "checkpoint_id",
        "checkpoint_sha256",
        "r01_summary_sha256",
        "r01_case_result_sha256",
        "sampler_parity_sha256",
        "direction_reference",
        "direction_semantics_decision",
        "direction_semantics_decision_sha256",
        "noise",
        "sampler_steps",
        "intervention_step",
        "model_action_horizon",
        "executed_action_horizon",
        "model_action_dimension",
        "action_frame",
        "normalization_space",
        "normalization_asset_sha256",
        "normalization_action_scale",
        "translation_action_bounds",
        "clipping_policy",
        "d_opt_model",
        "d_sim_model",
        "eef_radius_m",
        "distance_limit_m",
        "response_matrix_m_per_action",
        "simulator_safety_margin_m",
        "minimum_progress_m",
        "maximum_target_displacement_m",
        "maximum_obstacle_displacement_m",
        "simulator_repeats",
        "measurement_samples_per_trial",
        "required_arms",
    }
    if not isinstance(provenance, Mapping):
        errors.append("provenance must be an object")
        provenance = {}
    else:
        absent = required_provenance - set(provenance)
        if absent:
            errors.append(f"provenance missing fields: {sorted(absent)}")
        for key in ("host", "device", "slurm_job_id", "slurm_array_task_id", "partition"):
            if not isinstance(provenance.get(key), str) or not provenance.get(key):
                errors.append(f"provenance.{key} must be a non-empty allocation value")
        case_record = provenance.get("case_record")
        if not isinstance(case_record, Mapping) or provenance.get("case_record_sha256") != content_hash(dict(case_record or {})):
            errors.append("provenance case record/hash is invalid")
        if isinstance(case_record, Mapping) and case_record.get("case_id") != value.get("case_id"):
            errors.append("provenance case identity differs from the result")
        for key in ("checkpoint_sha256", "input_manifest_sha256", "r01_case_result_sha256", "sampler_parity_sha256"):
            if not _is_sha256(provenance.get(key)):
                errors.append(f"provenance.{key} must be a lowercase SHA-256")
        noise_array, noise_errors = _validate_array_record(
            provenance.get("noise"), name="provenance.noise", shape=(10, 32)
        )
        errors.extend(noise_errors)
        for key in ("environment_seed", "policy_seed", "random_control_seed"):
            if not isinstance(provenance.get(key), int) or isinstance(provenance.get(key), bool):
                errors.append(f"provenance.{key} must be an integer")
        policy_seed = provenance.get("policy_seed")
        if noise_array is not None and isinstance(policy_seed, int) and not isinstance(policy_seed, bool):
            expected_noise = np.random.default_rng(policy_seed).normal(
                size=(10, 32)
            ).astype(np.float32)
            if not np.array_equal(noise_array, expected_noise):
                errors.append("serialized explicit noise does not reconstruct from policy_seed")
        if provenance.get("sampler_steps") != 10 or provenance.get("intervention_step") != 5:
            errors.append("provenance must record intervention step 5 of ten")
        if provenance.get("model_action_horizon") != 10 or provenance.get("model_action_dimension") != 32:
            errors.append("provenance model action must be 10x32")
        if provenance.get("executed_action_horizon") != 5:
            errors.append("provenance executed horizon must be five")
        if provenance.get("normalization_asset_sha256") != NORMALIZATION_ASSET_SHA256:
            errors.append("provenance normalization asset is not registered")
        if provenance.get("normalization_action_scale") != list(REGISTERED_TRANSLATION_ACTION_SCALE):
            errors.append("provenance normalization action scale is not registered")
        try:
            recorded_response = tuple(
                tuple(float(item) for item in row)
                for row in provenance.get("response_matrix_m_per_action", ())
            )
        except (TypeError, ValueError, OverflowError):
            recorded_response = ()
        if recorded_response != REGISTERED_RESPONSE_MATRIX_M_PER_ACTION:
            errors.append("provenance response matrix is not the frozen H04 calibration")
        if provenance.get("eef_radius_m") != REGISTERED_EEF_RADIUS_M:
            errors.append("provenance EEF radius must remain 6 cm")
        if provenance.get("distance_limit_m") != REGISTERED_DISTANCE_LIMIT_M:
            errors.append("provenance distance limit must remain 1 m")
        if provenance.get("translation_action_bounds") != [-1.0, 1.0]:
            errors.append("provenance action bounds must be [-1, 1]")
        if provenance.get("clipping_policy") != "fail_without_clipping":
            errors.append("provenance must forbid clipping")
        if provenance.get("measurement_samples_per_trial") != EXPECTED_MEASUREMENT_SAMPLES:
            errors.append("provenance must require 126 measurements per trial")
        if provenance.get("simulator_repeats", 0) < 2:
            errors.append("provenance must require at least two simulator repeats")
        if provenance.get("required_arms") != list(ARMS):
            errors.append("provenance arm registration differs")
        if provenance.get("direction_reference") != DIRECTION_REFERENCE:
            errors.append("provenance direction reference differs from ADR-0013")
        if provenance.get("direction_semantics_decision") != DIRECTION_SEMANTICS_DECISION:
            errors.append("provenance direction semantics decision differs")
        if (
            provenance.get("direction_semantics_decision_sha256")
            != DIRECTION_SEMANTICS_DECISION_SHA256
        ):
            errors.append("provenance direction semantics hash differs")
        if isinstance(source, Mapping):
            for source_key, provenance_key in (
                ("r01_summary_sha256", "r01_summary_sha256"),
                ("r01_case_sha256", "r01_case_result_sha256"),
                ("sampler_parity_sha256", "sampler_parity_sha256"),
            ):
                if source.get(source_key) != provenance.get(provenance_key):
                    errors.append(
                        f"source_evidence.{source_key} conflicts with provenance.{provenance_key}"
                    )

    arms = value.get("arms")
    if not isinstance(arms, Mapping) or set(arms) != set(ARMS):
        errors.append(f"arms must contain exactly {list(ARMS)!r}")
        arms = {}
    status = value.get("status")
    pairing = value.get("pairing")
    directions = value.get("directions")
    outcome = value.get("outcome")
    if not isinstance(pairing, Mapping):
        errors.append("pairing must be an object")
        pairing = {}
    if not isinstance(directions, Mapping):
        errors.append("directions must be an object")
        directions = {}
    else:
        if directions.get("direction_reference") != DIRECTION_REFERENCE:
            errors.append("directions use a stale oracle reference")
        if directions.get("direction_semantics_decision") != DIRECTION_SEMANTICS_DECISION:
            errors.append("directions semantics decision differs from ADR-0013")
        if (
            directions.get("direction_semantics_decision_sha256")
            != DIRECTION_SEMANTICS_DECISION_SHA256
        ):
            errors.append("directions semantics decision hash differs")
    if not isinstance(outcome, Mapping):
        errors.append("outcome must be an object")
        outcome = {}
    (
        pairing_errors,
        compiled,
        eager_frozen,
        r01_nominal_actions,
        historical_witness_actions,
        historical_r01_delta,
    ) = _validate_exact_pairing(pairing)
    errors.extend(pairing_errors)

    if status == NOMINAL_RECONFIRMATION_FAILURE:
        eligible = outcome.get("r01_feasible_conditioned")
        if not isinstance(eligible, bool):
            errors.append("nominal mismatch must identify its R01 subset")
            eligible = False
        if outcome.get("population_mismatch_reason") != status:
            errors.append("nominal mismatch reason must equal its terminal status")
        if outcome.get("nominal_collision_reproduced") is not False:
            errors.append("nominal mismatch cannot claim collision reproduction")
        if outcome.get("direct_witness_reconfirmed") is not None:
            errors.append("nominal mismatch cannot claim a direct-witness outcome")

        source_pointer = (
            source.get("selected_witness_pointer")
            if isinstance(source, Mapping)
            else None
        )
        direction_pointer = directions.get("witness_pointer")
        r01_pointer = (
            source.get("r01_selected_p_min_changed_witness")
            if isinstance(source, Mapping)
            else None
        )
        if eligible:
            if (
                not isinstance(source_pointer, Mapping)
                or not _is_sha256(source_pointer.get("actions_sha256"))
                or not _is_sha256(source_pointer.get("actions_array_sha256"))
            ):
                errors.append("eligible nominal mismatch must bind its raw R01 witness")
            if direction_pointer != source_pointer:
                errors.append("nominal mismatch source and direction witness pointers differ")
            if isinstance(source_pointer, Mapping):
                expected_r01_pointer = {
                    "search": source_pointer.get("search"),
                    "candidate_index": source_pointer.get("candidate_index"),
                    "source": source_pointer.get("source"),
                    "actions_sha256": source_pointer.get("actions_sha256"),
                }
                if r01_pointer != expected_r01_pointer:
                    errors.append("nominal mismatch witness differs from the R01 outcome pointer")
            if isinstance(source, Mapping) and source.get("r01_case_status") != "verified_safe_progress":
                errors.append("eligible nominal mismatch must come from an eligible R01 case")
            if historical_witness_actions is None or historical_r01_delta is None:
                errors.append("eligible nominal mismatch must retain raw R01 action diagnostics")
            elif float(np.linalg.norm(historical_r01_delta)) <= 0.0:
                errors.append("eligible nominal mismatch has a zero historical R01 Delta")
            elif isinstance(source_pointer, Mapping):
                if source_pointer.get("actions_array_sha256") != _array_hash(
                    np.asarray(historical_witness_actions, dtype=np.float64)
                ):
                    errors.append("nominal mismatch witness diagnostic differs from pointer")
                if source_pointer.get("actions_sha256") != content_hash(
                    historical_witness_actions.tolist()
                ):
                    errors.append("nominal mismatch witness content hash differs from pointer")
        else:
            if source_pointer is not None or direction_pointer is not None or r01_pointer is not None:
                errors.append("no-witness nominal mismatch cannot bind a witness")
            if isinstance(source, Mapping) and source.get("r01_case_status") == "verified_safe_progress":
                errors.append("no-witness nominal mismatch conflicts with the R01 status")
            if historical_witness_actions is not None or historical_r01_delta is not None:
                errors.append("no-witness nominal mismatch cannot retain a witness diagnostic")

        arrays = directions.get("arrays")
        l2_norms = directions.get("l2_norms")
        expected_direction_keys = {
            "delta_star_physical",
            "delta_star_model",
            "random_model",
            "analytic_geometry_model",
        }
        if (
            not isinstance(arrays, Mapping)
            or set(arrays) != expected_direction_keys
            or any(item is not None for item in arrays.values())
        ):
            errors.append("nominal mismatch cannot claim constructed directions")
        if (
            not isinstance(l2_norms, Mapping)
            or set(l2_norms) != expected_direction_keys
            or any(item is not None for item in l2_norms.values())
        ):
            errors.append("nominal mismatch direction norms must all be null")
        construction_failures = directions.get("construction_failures")
        if (
            not isinstance(construction_failures, Mapping)
            or set(construction_failures)
            != {"random_model", "analytic_geometry_model"}
            or not all(
                isinstance(item, str) and item
                for item in construction_failures.values()
            )
        ):
            errors.append("nominal mismatch must explain why directions were not constructed")
        if directions.get("diagnostics") != {"analytic_geometry": None}:
            errors.append("nominal mismatch cannot claim an analytic diagnostic")

        frozen = arms.get("frozen")
        frozen_gate: Optional[bool] = None
        nominal_collision = False
        if not isinstance(frozen, Mapping) or frozen.get("status") not in EVALUATED_ARM_STATUSES:
            errors.append("nominal mismatch must retain the evaluated frozen arm")
        else:
            frozen_full, item_errors = _validate_array_record(
                frozen.get("full_actions"), name="arms.frozen.full_actions"
            )
            errors.extend(item_errors)
            _, item_errors = _validate_array_record(
                frozen.get("executed_actions"),
                name="arms.frozen.executed_actions",
                shape=(5, 7),
            )
            errors.extend(item_errors)
            if (
                frozen_full is not None
                and eager_frozen is not None
                and not np.array_equal(frozen_full, eager_frozen)
            ):
                errors.append("nominal mismatch frozen actions differ from eager output")
            bounds = frozen.get("bounds")
            if not isinstance(bounds, Mapping) or bounds.get("passed") is not True:
                errors.append("nominal mismatch frozen arm must pass bounds")
            frozen_gate, gate_errors = _recompute_serialized_arm_gate(
                frozen,
                repeat_count=int(provenance.get("simulator_repeats", 0)),
                margin=float(provenance.get("simulator_safety_margin_m", float("nan"))),
                p_min=float(provenance.get("minimum_progress_m", float("nan"))),
                target_limit=float(
                    provenance.get("maximum_target_displacement_m", float("nan"))
                ),
                obstacle_limit=float(
                    provenance.get("maximum_obstacle_displacement_m", float("nan"))
                ),
            )
            errors.extend(f"arms.frozen: {item}" for item in gate_errors)
            if not isinstance(frozen.get("gate"), Mapping) or frozen["gate"].get("passed") != frozen_gate:
                errors.append("nominal mismatch frozen gate conflicts with raw repeats")
            if frozen.get("status") != ("passed_gate" if frozen_gate else "failed_gate"):
                errors.append("nominal mismatch frozen status conflicts with raw repeats")
            repeats = frozen.get("repeats")
            first_repeat = (
                repeats[0]
                if isinstance(repeats, list)
                and repeats
                and isinstance(repeats[0], Mapping)
                else None
            )
            nominal_collision = bool(
                first_repeat is not None
                and (
                    float(first_repeat.get("clearance_m", float("inf"))) < 0.0
                    or bool(first_repeat.get("contact", False))
                )
            )
            if nominal_collision:
                errors.append("nominal mismatch status conflicts with frozen collision evidence")

        for name in ARMS[1:]:
            errors.extend(
                _validate_not_evaluated_arm(
                    arms.get(name), name=name, applicable=eligible
                )
            )
        expected_gates = {"frozen": frozen_gate, **{name: None for name in ARMS[1:]}}
        if outcome.get("arm_gate_pass") != expected_gates:
            errors.append("nominal mismatch arm_gate_pass conflicts with observed evidence")
        expected_statuses = {
            "frozen": frozen.get("status") if isinstance(frozen, Mapping) else None,
            **{
                name: NOT_EVALUATED_AFTER_RECONFIRMATION_FAILURE
                for name in ARMS[1:]
            },
        }
        if outcome.get("arm_status") != expected_statuses:
            errors.append("nominal mismatch arm_status conflicts with arms")
        return errors

    if status == "not_applicable_no_r01_witness":
        if not isinstance(source, Mapping) or source.get("selected_witness_pointer") is not None:
            errors.append("no-witness artifact cannot bind a selected witness")
        if isinstance(source, Mapping) and source.get("r01_selected_p_min_changed_witness") is not None:
            errors.append("no-witness artifact cannot have an R01 witness outcome pointer")
        if historical_witness_actions is not None or historical_r01_delta is not None:
            errors.append("no-witness artifact cannot retain a witness diagnostic")
        arrays = directions.get("arrays")
        if not isinstance(arrays, Mapping) or any(item is not None for item in arrays.values()):
            errors.append("no-witness direction arrays must all be null")
        frozen = arms.get("frozen")
        if not isinstance(frozen, Mapping) or frozen.get("status") not in EVALUATED_ARM_STATUSES:
            errors.append("no-witness artifact must still evaluate the frozen arm")
            frozen = {}
        else:
            frozen_full, frozen_errors = _validate_array_record(
                frozen.get("full_actions"), name="arms.frozen.full_actions"
            )
            errors.extend(frozen_errors)
            _, executed_errors = _validate_array_record(
                frozen.get("executed_actions"),
                name="arms.frozen.executed_actions",
                shape=(5, 7),
            )
            errors.extend(executed_errors)
            if eager_frozen is not None and frozen_full is not None and not np.array_equal(
                eager_frozen, frozen_full
            ):
                errors.append("no-witness frozen actions differ from eager trace-only output")
            if not isinstance(frozen.get("bounds"), Mapping) or frozen["bounds"].get("passed") is not True:
                errors.append("no-witness frozen action must pass bounds before replay")
            gate, gate_errors = _recompute_serialized_arm_gate(
                frozen,
                repeat_count=int(provenance.get("simulator_repeats", 0)),
                margin=float(provenance.get("simulator_safety_margin_m", float("nan"))),
                p_min=float(provenance.get("minimum_progress_m", float("nan"))),
                target_limit=float(provenance.get("maximum_target_displacement_m", float("nan"))),
                obstacle_limit=float(provenance.get("maximum_obstacle_displacement_m", float("nan"))),
            )
            errors.extend(f"arms.frozen: {item}" for item in gate_errors)
            if not isinstance(frozen.get("gate"), Mapping) or frozen["gate"].get("passed") != gate:
                errors.append("no-witness frozen gate conflicts with raw repeats")
            if frozen.get("status") != ("passed_gate" if gate else "failed_gate"):
                errors.append("no-witness frozen status conflicts with raw repeats")
            repeats = frozen.get("repeats")
            nominal_collision = bool(
                isinstance(repeats, list)
                and repeats
                and (
                    float(repeats[0].get("clearance_m", float("inf"))) < 0.0
                    or bool(repeats[0].get("contact", False))
                )
            )
            if not nominal_collision:
                errors.append("no-witness artifact must reconfirm the nominal collision")
            if outcome.get("nominal_collision_reproduced") != nominal_collision:
                errors.append("no-witness outcome nominal flag conflicts with frozen replay")
            expected_gates = {"frozen": gate, **{name: None for name in ARMS[1:]}}
            if outcome.get("arm_gate_pass") != expected_gates:
                errors.append("no-witness arm_gate_pass conflicts with frozen replay")
        for name in ARMS[1:]:
            arm = arms.get(name)
            if not isinstance(arm, Mapping) or arm.get("status") != "not_applicable_no_r01_witness":
                errors.append(f"no-witness arm {name} must be explicit N/A")
            elif arm.get("repeats") != [] or arm.get("full_actions") is not None:
                errors.append(f"no-witness arm {name} cannot contain an execution")
        if outcome.get("r01_feasible_conditioned") is not False:
            errors.append("no-witness outcome cannot enter the feasible-conditioned population")
        expected_statuses = {
            "frozen": frozen.get("status") if isinstance(frozen, Mapping) else None,
            **{name: "not_applicable_no_r01_witness" for name in ARMS[1:]},
        }
        if outcome.get("arm_status") != expected_statuses:
            errors.append("no-witness arm_status conflicts with arms")
        if outcome.get("direct_witness_reconfirmed") is not None:
            errors.append("no-witness direct witness outcome must be null")
        if outcome.get("population_mismatch_reason") is not None:
            errors.append("no-witness completion cannot claim a population mismatch")
        return errors

    pointer = directions.get("witness_pointer")
    if (
        not isinstance(pointer, Mapping)
        or not _is_sha256(pointer.get("actions_sha256"))
        or not _is_sha256(pointer.get("actions_array_sha256"))
    ):
        errors.append("eligible directions must bind a selected raw witness pointer")
    if isinstance(source, Mapping) and source.get("selected_witness_pointer") != pointer:
        errors.append("source and direction witness pointers differ")
    if isinstance(source, Mapping) and isinstance(pointer, Mapping):
        expected_r01_pointer = {
            "search": pointer.get("search"),
            "candidate_index": pointer.get("candidate_index"),
            "source": pointer.get("source"),
            "actions_sha256": pointer.get("actions_sha256"),
        }
        if source.get("r01_selected_p_min_changed_witness") != expected_r01_pointer:
            errors.append("recomputed witness pointer differs from the bound R01 outcome pointer")
    if historical_witness_actions is None or historical_r01_delta is None:
        errors.append("eligible result must retain raw R01 witness and Delta diagnostics")
    elif float(np.linalg.norm(historical_r01_delta)) <= 0.0:
        errors.append("historical R01 Delta diagnostic must remain nonzero")
    elif isinstance(pointer, Mapping):
        if pointer.get("actions_array_sha256") != _array_hash(
            np.asarray(historical_witness_actions, dtype=np.float64)
        ):
            errors.append("historical R01 witness diagnostic differs from pointer")
        if pointer.get("actions_sha256") != content_hash(
            historical_witness_actions.tolist()
        ):
            errors.append("historical R01 witness content hash differs from pointer")
    arrays = directions.get("arrays")
    direction_values: Dict[str, Optional[np.ndarray]] = {}
    if not isinstance(arrays, Mapping) or set(arrays) != {
        "delta_star_physical", "delta_star_model", "random_model", "analytic_geometry_model"
    }:
        errors.append("directions.arrays has incorrect fields")
    else:
        for key, record in arrays.items():
            if record is None:
                direction_values[key] = None
                continue
            array, item_errors = _validate_array_record(
                record, name=f"directions.arrays.{key}", shape=(10, 32)
            )
            errors.extend(item_errors)
            direction_values[key] = array
            if array is not None and (
                np.any(array[5:, :] != 0.0) or np.any(array[:5, 3:] != 0.0)
            ):
                errors.append(f"direction {key} is nonzero outside the registered mask")
    delta_model = direction_values.get("delta_star_model")
    delta_physical = direction_values.get("delta_star_physical")
    if delta_physical is None:
        errors.append("eligible result requires Delta_star_physical")
    if delta_model is None or float(np.linalg.norm(delta_model)) <= 0.0:
        errors.append("eligible result requires nonzero Delta_star_model")
    if delta_model is not None and delta_physical is not None:
        scale = np.asarray(REGISTERED_TRANSLATION_ACTION_SCALE, dtype=np.float64)
        expected_model = np.zeros((10, 32), dtype=np.float64)
        expected_model[:5, :3] = delta_physical[:5, :3] / scale[None, :]
        if not np.array_equal(delta_model, expected_model):
            errors.append("Delta_star_model is not the scale-only conversion of Delta_star_physical")

    if (
        eager_frozen is not None
        and historical_witness_actions is not None
        and delta_physical is not None
    ):
        expected_delta_physical = np.zeros((10, 32), dtype=np.float64)
        expected_delta_physical[:5, :3] = (
            historical_witness_actions[:5, :3] - eager_frozen[:5, :3]
        )
        if not np.array_equal(delta_physical, expected_delta_physical):
            errors.append(
                "Delta_star_physical is not immutable witness minus fresh paired eager"
            )

    l2_claims = directions.get("l2_norms")
    if not isinstance(l2_claims, Mapping) or set(l2_claims) != {
        "delta_star_physical",
        "delta_star_model",
        "random_model",
        "analytic_geometry_model",
    }:
        errors.append("directions.l2_norms has incorrect fields")
    else:
        for key, direction in direction_values.items():
            expected_norm = float(np.linalg.norm(direction)) if direction is not None else None
            claim = l2_claims.get(key)
            if expected_norm is None:
                if claim is not None:
                    errors.append(f"directions.l2_norms.{key} must be null")
            elif not isinstance(claim, (int, float)) or isinstance(claim, bool) or not math.isclose(
                float(claim), expected_norm, rel_tol=0.0, abs_tol=1e-12
            ):
                errors.append(f"directions.l2_norms.{key} conflicts with its raw array")

    construction_failures = directions.get("construction_failures")
    if not isinstance(construction_failures, Mapping) or set(construction_failures) != {
        "random_model",
        "analytic_geometry_model",
    }:
        errors.append("directions.construction_failures has incorrect fields")
        construction_failures = {}

    predicted_clean_physical: Optional[np.ndarray] = None
    trace_record = pairing.get("eager_trace")
    trace_leaves = trace_record.get("leaves") if isinstance(trace_record, Mapping) else None
    if isinstance(trace_leaves, Mapping):
        predicted_clean_physical, item_errors = _validate_array_record(
            trace_leaves.get("predicted_clean_physical"),
            name="pairing.eager_trace.predicted_clean_physical_for_direction",
            shape=(10, 7),
        )
        errors.extend(item_errors)

    if delta_model is not None and predicted_clean_physical is not None:
        from .r02_directions import (
            R02DirectionError,
            analytic_d_opt_ascent_direction,
            deterministic_equal_l2_random_direction,
        )

        try:
            expected_random = np.asarray(
                deterministic_equal_l2_random_direction(
                    delta_model,
                    base_physical_prefix=predicted_clean_physical[:5, :7],
                    action_scale=REGISTERED_TRANSLATION_ACTION_SCALE,
                    seed=int(provenance.get("random_control_seed")),
                    action_low=-1.0,
                    action_high=1.0,
                ),
                dtype=np.float64,
            )
        except (R02DirectionError, TypeError, ValueError, OverflowError) as error:
            if direction_values.get("random_model") is not None:
                errors.append("stored random direction exists although reconstruction failed")
            if construction_failures.get("random_model") != str(error):
                errors.append("random direction failure differs from deterministic reconstruction")
        else:
            if not np.array_equal(direction_values.get("random_model"), expected_random):
                errors.append("random direction differs from seeded equal-L2 reconstruction")
            if construction_failures.get("random_model") is not None:
                errors.append("successful random direction cannot claim construction failure")

        frozen_arm_for_geometry = arms.get("frozen")
        frozen_repeats = (
            frozen_arm_for_geometry.get("repeats")
            if isinstance(frozen_arm_for_geometry, Mapping)
            else None
        )
        branch_rollout = (
            frozen_repeats[0]
            if isinstance(frozen_repeats, list)
            and frozen_repeats
            and isinstance(frozen_repeats[0], Mapping)
            else None
        )
        try:
            if branch_rollout is None:
                raise R02DirectionError("frozen rollout is unavailable for analytic reconstruction")
            expected_analytic_result = analytic_d_opt_ascent_direction(
                predicted_clean_physical[:5, :7],
                start_eef_center_m=branch_rollout["start_eef_center_m"],
                response_matrix_m_per_action=REGISTERED_RESPONSE_MATRIX_M_PER_ACTION,
                obstacle_boxes=branch_rollout["branch_obstacle_boxes"],
                eef_radius_m=REGISTERED_EEF_RADIUS_M,
                action_scale=REGISTERED_TRANSLATION_ACTION_SCALE,
                model_l2_budget=float(np.linalg.norm(delta_model)),
                samples_per_segment=26,
                model_action_horizon=10,
                model_action_dimension=32,
            )
        except (R02DirectionError, KeyError, TypeError, ValueError, OverflowError) as error:
            if direction_values.get("analytic_geometry_model") is not None:
                errors.append("stored analytic direction exists although reconstruction failed")
            if construction_failures.get("analytic_geometry_model") != str(error):
                errors.append("analytic failure differs from exact D_opt reconstruction")
        else:
            expected_analytic = np.asarray(
                expected_analytic_result.direction_model, dtype=np.float64
            )
            if not np.array_equal(
                direction_values.get("analytic_geometry_model"), expected_analytic
            ):
                errors.append("analytic direction differs from exact D_opt reconstruction")
            if construction_failures.get("analytic_geometry_model") is not None:
                errors.append("successful analytic direction cannot claim construction failure")
            stored_diagnostic = directions.get("diagnostics", {}).get("analytic_geometry") if isinstance(
                directions.get("diagnostics"), Mapping
            ) else None
            if content_hash(stored_diagnostic) != content_hash(
                expected_analytic_result.to_dict()
            ):
                errors.append("analytic diagnostic differs from exact D_opt reconstruction")
    for key in ("random_model", "analytic_geometry_model"):
        direction = direction_values.get(key)
        failure = directions.get("construction_failures", {}).get(key) if isinstance(
            directions.get("construction_failures"), Mapping
        ) else None
        if direction is None and not isinstance(failure, str):
            errors.append(f"missing direction {key} requires an explicit construction failure")
        if direction is not None and delta_model is not None and not math.isclose(
            float(np.linalg.norm(direction)),
            float(np.linalg.norm(delta_model)),
            rel_tol=1e-6,
            abs_tol=1e-6,
        ):
            errors.append(f"direction {key} does not have the oracle model-space L2 budget")
    diagnostics = directions.get("diagnostics")
    if not isinstance(diagnostics, Mapping) or set(diagnostics) != {"analytic_geometry"}:
        errors.append("directions.diagnostics must contain analytic_geometry")
    else:
        analytic_diagnostic = diagnostics.get("analytic_geometry")
        analytic_direction = direction_values.get("analytic_geometry_model")
        if analytic_direction is None:
            if analytic_diagnostic is not None:
                errors.append("failed analytic direction cannot claim a diagnostic result")
        elif not isinstance(analytic_diagnostic, Mapping):
            errors.append("successful analytic direction must retain its exact geometry diagnostic")
        else:
            diagnostic_direction = np.asarray(
                analytic_diagnostic.get("direction_model"), dtype=np.float64
            )
            if not np.array_equal(diagnostic_direction, analytic_direction):
                errors.append("analytic diagnostic direction differs from the applied direction")

    repeat_count = int(provenance.get("simulator_repeats", 0)) if isinstance(provenance, Mapping) else 0
    margin = float(provenance.get("simulator_safety_margin_m", float("nan"))) if isinstance(provenance, Mapping) else float("nan")
    p_min = float(provenance.get("minimum_progress_m", float("nan"))) if isinstance(provenance, Mapping) else float("nan")
    target_limit = float(provenance.get("maximum_target_displacement_m", float("nan"))) if isinstance(provenance, Mapping) else float("nan")
    obstacle_limit = float(provenance.get("maximum_obstacle_displacement_m", float("nan"))) if isinstance(provenance, Mapping) else float("nan")
    recomputed_gate: Dict[str, Optional[bool]] = {}
    for name in ARMS:
        arm = arms.get(name)
        if not isinstance(arm, Mapping):
            errors.append(f"arm {name} must be an object")
            continue
        arm_status = arm.get("status")
        if arm_status not in EVALUATED_ARM_STATUSES | NONROLLOUT_ARM_STATUSES:
            errors.append(f"arm {name} has invalid status {arm_status!r}")
            continue
        bounds = arm.get("bounds")
        if not isinstance(bounds, Mapping) or bounds.get("clipped") is not False:
            errors.append(f"arm {name} must record no clipping")
        if arm_status in EVALUATED_ARM_STATUSES:
            full, item_errors = _validate_array_record(arm.get("full_actions"), name=f"arms.{name}.full_actions")
            errors.extend(item_errors)
            executed, item_errors = _validate_array_record(
                arm.get("executed_actions"), name=f"arms.{name}.executed_actions", shape=(5, 7)
            )
            errors.extend(item_errors)
            if full is not None and executed is not None and not np.array_equal(
                np.asarray(full[:5, :7], dtype=np.float64),
                np.asarray(executed, dtype=np.float64),
            ):
                errors.append(f"arm {name} executed prefix differs from full actions")
            if not isinstance(bounds, Mapping) or bounds.get("passed") is not True:
                errors.append(f"evaluated arm {name} must pass bounds before rollout")
            repeats = arm.get("repeats")
            if not isinstance(repeats, list) or len(repeats) != repeat_count:
                errors.append(f"arm {name} must contain exactly {repeat_count} repeats")
                recomputed_gate[name] = None
                continue
            exact = bool(
                repeats and all(content_hash(item) == content_hash(repeats[0]) for item in repeats[1:])
            )
            if arm.get("replay_exact") != exact:
                errors.append(f"arm {name} replay_exact conflicts with raw repeats")
            trial_passes = []
            for index, trial in enumerate(repeats):
                if not isinstance(trial, Mapping):
                    errors.append(f"arm {name} repeat {index} must be an object")
                    trial_passes.append(False)
                    continue
                reach = trial.get("reach")
                evidence_errors = _trial_evidence_errors(trial)
                errors.extend(
                    f"arm {name} repeat {index} raw evidence: {error}"
                    for error in evidence_errors
                )
                passed = bool(
                    not evidence_errors
                    and isinstance(reach, Mapping)
                    and float(trial.get("clearance_m", float("-inf"))) >= margin
                    and not bool(trial.get("contact", True))
                    and int(trial.get("measurement_samples", -1)) == EXPECTED_MEASUREMENT_SAMPLES
                    and float(reach.get("reach_progress_m", float("-inf"))) >= p_min
                    and float(reach.get("maximum_target_displacement_m", float("inf"))) <= target_limit
                    and float(reach.get("maximum_active_obstacle_displacement_m", float("inf"))) <= obstacle_limit
                )
                trial_passes.append(passed)
            gate_pass = bool(exact and all(trial_passes))
            recomputed_gate[name] = gate_pass
            if not isinstance(arm.get("gate"), Mapping) or arm["gate"].get("passed") != gate_pass:
                errors.append(f"arm {name} gate conflicts with raw D_sim repeats")
            expected_status = "passed_gate" if gate_pass else "failed_gate"
            if arm_status != expected_status:
                errors.append(f"arm {name} status conflicts with recomputed gate")
        else:
            recomputed_gate[name] = None
            if arm.get("repeats") != [] or arm.get("replay_exact") is not None:
                errors.append(f"non-rollout arm {name} cannot contain simulator evidence")
            if arm_status == "bounds_failure":
                full, item_errors = _validate_array_record(
                    arm.get("full_actions"), name=f"arms.{name}.full_actions"
                )
                errors.extend(item_errors)
                if full is not None:
                    values = np.asarray(full[:5, :3], dtype=np.float64)
                    if np.all(np.isfinite(values)) and np.all(values >= -1.0) and np.all(values <= 1.0):
                        errors.append(f"arm {name} claims bounds failure without a violating action")
                if not isinstance(bounds, Mapping) or bounds.get("passed") is not False:
                    errors.append(f"bounds-failure arm {name} must fail its bounds record")
            elif arm_status == "direction_failure" and arm.get("full_actions") is not None:
                errors.append(f"direction-failure arm {name} cannot contain sampled actions")
            elif arm_status == NOT_EVALUATED_AFTER_RECONFIRMATION_FAILURE:
                errors.extend(
                    _validate_not_evaluated_arm(
                        arm, name=name, applicable=True
                    )
                )

    direction_for_arm = {
        "random_residual": "random_model",
        "analytic_geometry_residual": "analytic_geometry_model",
        "oracle_residual": "delta_star_model",
        "bridge_diagnostic": "delta_star_model",
    }
    for name, direction_key in direction_for_arm.items():
        arm = arms.get(name) if isinstance(arms, Mapping) else None
        if not isinstance(arm, Mapping):
            continue
        if arm.get("status") == NOT_EVALUATED_AFTER_RECONFIRMATION_FAILURE:
            continue
        direction = direction_values.get(direction_key)
        if direction is None:
            if arm.get("status") != "direction_failure":
                errors.append(f"arm {name} must report its missing registered direction")
            continue
        if arm.get("status") == "direction_failure":
            errors.append(f"arm {name} cannot claim direction failure with a direction present")
            continue
        correction, correction_errors = _validate_array_record(
            arm.get("correction"), name=f"arms.{name}.correction", shape=(10, 32)
        )
        errors.extend(correction_errors)
        if correction is not None and not np.array_equal(
            correction, np.asarray(direction, dtype=np.float32)
        ):
            errors.append(f"arm {name} correction differs from direction {direction_key}")
        if arm.get("preintervention_trace_exact_to_frozen") is not True:
            errors.append(f"arm {name} must share the exact pre-intervention frozen trace")
        if arm.get("policy_replay_exact") is not True:
            errors.append(f"arm {name} must have an exact duplicate policy replay")
        primary_actions, primary_errors = _validate_array_record(
            arm.get("full_actions"), name=f"arms.{name}.full_actions"
        )
        errors.extend(primary_errors)
        duplicate_actions, duplicate_errors = _validate_array_record(
            arm.get("policy_duplicate_actions"),
            name=f"arms.{name}.policy_duplicate_actions",
        )
        errors.extend(duplicate_errors)
        if (
            primary_actions is not None
            and duplicate_actions is not None
            and not np.array_equal(primary_actions, duplicate_actions)
        ):
            errors.append(f"arm {name} duplicate policy actions differ")
        duplicate_trace = arm.get("policy_duplicate_trace")
        errors.extend(
            _validate_trace_record(
                duplicate_trace, name=f"arms.{name}.policy_duplicate_trace"
            )
        )
        if isinstance(pairing, Mapping):
            baseline_trace = pairing.get("eager_trace")
            expected_trace_sha = baseline_trace.get("sha256") if isinstance(baseline_trace, Mapping) else None
            if arm.get("trace_sha256") != expected_trace_sha:
                errors.append(f"arm {name} trace hash differs from the frozen eager trace")
            if not isinstance(duplicate_trace, Mapping) or duplicate_trace.get("sha256") != expected_trace_sha:
                errors.append(f"arm {name} duplicate trace hash differs from frozen eager trace")

    frozen = arms.get("frozen") if isinstance(arms, Mapping) else None
    direct = arms.get("direct_witness") if isinstance(arms, Mapping) else None
    if not isinstance(frozen, Mapping) or frozen.get("status") not in EVALUATED_ARM_STATUSES:
        errors.append("eligible completion must evaluate the frozen arm")
    if not isinstance(direct, Mapping) or direct.get("status") not in EVALUATED_ARM_STATUSES:
        errors.append("eligible completion must evaluate the direct witness arm")
    if isinstance(frozen, Mapping) and eager_frozen is not None:
        frozen_full, frozen_errors = _validate_array_record(
            frozen.get("full_actions"), name="arms.frozen.full_actions"
        )
        errors.extend(frozen_errors)
        if frozen_full is not None and not np.array_equal(frozen_full, eager_frozen):
            errors.append("frozen arm actions differ from the exact eager trace-only output")
    if isinstance(direct, Mapping) and isinstance(pointer, Mapping):
        direct_executed, direct_errors = _validate_array_record(
            direct.get("executed_actions"), name="arms.direct_witness.executed_actions", shape=(5, 7)
        )
        errors.extend(direct_errors)
        if direct_executed is not None and pointer.get("actions_array_sha256") != _array_hash(
            np.asarray(direct_executed, dtype=np.float64)
        ):
            errors.append("direct arm actions differ from the selected raw witness pointer")
        if direct_executed is not None and pointer.get("actions_sha256") != content_hash(
            np.asarray(direct_executed, dtype=np.float64).tolist()
        ):
            errors.append("direct arm content differs from the selected raw witness pointer")
        if (
            direct_executed is not None
            and historical_witness_actions is not None
            and not np.array_equal(direct_executed, historical_witness_actions)
        ):
            errors.append("direct arm differs from the retained immutable R01 witness")
    nominal_collision = False
    if isinstance(frozen, Mapping) and isinstance(frozen.get("repeats"), list) and frozen["repeats"]:
        nominal_collision = bool(
            float(frozen["repeats"][0].get("clearance_m", float("inf"))) < 0.0
            or bool(frozen["repeats"][0].get("contact", False))
        )
    if outcome.get("nominal_collision_reproduced") != nominal_collision:
        errors.append("outcome nominal collision conflicts with frozen raw rollout")
    if not nominal_collision:
        errors.append("eligible result requires nominal collision reproduction")
    if outcome.get("direct_witness_reconfirmed") != recomputed_gate.get("direct_witness"):
        errors.append("outcome direct witness flag conflicts with raw rollout")
    if outcome.get("arm_gate_pass") != recomputed_gate:
        errors.append("outcome arm_gate_pass conflicts with recomputed gates")
    expected_statuses = {
        name: arms[name].get("status") for name in ARMS if isinstance(arms.get(name), Mapping)
    }
    if outcome.get("arm_status") != expected_statuses:
        errors.append("outcome arm_status conflicts with arms")
    if outcome.get("r01_feasible_conditioned") is not True:
        errors.append("eligible completion must enter the R01-feasible-conditioned population")
    if status == DIRECT_RECONFIRMATION_FAILURE:
        if recomputed_gate.get("direct_witness") is not False:
            errors.append("direct mismatch requires a failed observed direct-witness gate")
        if outcome.get("population_mismatch_reason") != status:
            errors.append("direct mismatch reason must equal its terminal status")
        for name in FLOW_ARMS:
            errors.extend(
                _validate_not_evaluated_arm(
                    arms.get(name), name=name, applicable=True
                )
            )
    else:
        if status != "completed":
            errors.append("eligible artifact reached an unsupported terminal state")
        if recomputed_gate.get("direct_witness") is not True:
            errors.append("completed eligible result requires direct witness reconfirmation")
        if outcome.get("population_mismatch_reason") is not None:
            errors.append("completed eligible result cannot claim a population mismatch")
        for name in FLOW_ARMS:
            arm = arms.get(name)
            if (
                isinstance(arm, Mapping)
                and arm.get("status")
                == NOT_EVALUATED_AFTER_RECONFIRMATION_FAILURE
            ):
                errors.append(f"completed eligible arm {name} cannot be not-evaluated")
    return errors


def valid_r02_completion(
    path: str | Path,
    *,
    case_id: Optional[str] = None,
    run_id: Optional[str] = None,
    config_hash: Optional[str] = None,
) -> bool:
    try:
        value = load_json(path)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return False
    if not isinstance(value, Mapping) or validate_r02_result(value):
        return False
    return bool(
        (case_id is None or value.get("case_id") == case_id)
        and (run_id is None or value.get("run_id") == run_id)
        and (config_hash is None or value.get("config_hash") == config_hash)
    )


def _finalize_order_contamination_check(
    client: Any,
    policy_input: Mapping[str, Any],
    noise: np.ndarray,
    config: R02Config,
    pairing: Dict[str, Any],
    *,
    initial_compiled_actions: np.ndarray,
    initial_eager_actions: np.ndarray,
    initial_eager_trace: Mapping[str, np.ndarray],
) -> None:
    """Re-run both public paths before any terminal artifact is committed."""

    final_compiled = _infer(
        client,
        policy_input,
        noise,
        config,
        intervention_mode="none",
        return_trace=False,
    )
    final_actions = np.asarray(final_compiled["actions"], dtype=np.float64)
    if not np.array_equal(initial_compiled_actions, final_actions):
        raise RuntimeError("R02 final compiled sampler changed after trace/intervention requests")
    final_eager = _infer(
        client,
        policy_input,
        noise,
        config,
        intervention_mode="none",
        return_trace=True,
    )
    final_eager_actions = np.asarray(final_eager["actions"], dtype=np.float64)
    final_eager_trace = _trace_arrays(final_eager)
    if not (
        np.array_equal(initial_eager_actions, final_eager_actions)
        and _same_trace(initial_eager_trace, final_eager_trace)
    ):
        raise RuntimeError("R02 final eager sampler changed after trace/intervention requests")
    pairing["final_compiled_actions"] = _array_record(final_actions)
    pairing["order_contamination_check_exact"] = True
    pairing["final_eager_actions"] = _array_record(final_eager_actions)
    pairing["final_eager_trace"] = _trace_record(final_eager_trace)
    pairing["eager_order_contamination_check_exact"] = True


def run_r02_case(
    case: Mapping[str, Any],
    config: R02Config,
    *,
    repo_root: str | Path,
    input_manifest_sha256: str,
    client: Any = None,
    environment: Optional[SafeLiberoCase] = None,
) -> Tuple[Path, str]:
    """Execute one immutable case or write its explicit no-witness artifact."""

    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("R02 real cases must execute inside a Slurm allocation")
    visible_device = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible_device or visible_device == "NoDevFiles":
        raise RuntimeError("R02 real cases require an allocation-visible GPU")
    root = Path(repo_root).resolve()
    case_id = str(case.get("case_id", ""))
    config_hash = content_hash(_normalized_config(config))
    output = Path(config.oracle.output_root) / config.oracle.run_id / case_id / "r02-paired.json"
    if valid_r02_completion(
        output, case_id=case_id, run_id=config.oracle.run_id, config_hash=config_hash
    ):
        return output, "skipped_valid_completion"
    if str(case.get("task_suite")) != "safelibero_spatial" or str(case.get("safety_level")) != "II" or int(case.get("task_index", -1)) != 0:
        raise ValueError("R02 case is outside the frozen SafeLIBERO Spatial Level-II task-0 population")
    raw_path, raw_r01, raw_sha, eligible = _load_r01_case(case_id, config)
    expected_manifest_sha = config.r01_summary.get("identities", {}).get(
        "manifest_sha256"
    )
    if input_manifest_sha256 != expected_manifest_sha:
        raise R02SourceError("R02 manifest content differs from the accepted R01 manifest")
    raw_case = raw_r01.get("provenance", {}).get("case_record")
    if raw_case != dict(case):
        raise R02SourceError("R02 manifest record differs from the immutable raw R01 case record")
    noise = np.random.default_rng(int(case["policy_seed"])).normal(
        size=(config.oracle.action_horizon, config.oracle.action_dim)
    ).astype(np.float32)
    provenance = _base_provenance(
        case,
        config,
        root=root,
        input_manifest_sha256=input_manifest_sha256,
        raw_r01_path=raw_path,
        raw_r01_sha256=raw_sha,
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
            raise RuntimeError("R02 branch is no longer a valid pregrasp reach state")
        policy_input = policy_observation(
            initial_observation, environment.prompt, config.oracle.resize_size
        )
        compiled = _infer(
            client,
            policy_input,
            noise,
            config,
            intervention_mode="none",
            return_trace=False,
        )
        eager = _infer(
            client,
            policy_input,
            noise,
            config,
            intervention_mode="none",
            return_trace=True,
        )
        duplicate_eager = _infer(
            client,
            policy_input,
            noise,
            config,
            intervention_mode="none",
            return_trace=True,
        )
        compiled_actions = np.asarray(compiled["actions"], dtype=np.float64)
        eager_actions = np.asarray(eager["actions"], dtype=np.float64)
        duplicate_actions = np.asarray(duplicate_eager["actions"], dtype=np.float64)
        eager_trace = _trace_arrays(eager)
        duplicate_trace = _trace_arrays(duplicate_eager)
        r01_nominal = np.asarray(raw_r01["nominal"]["actions"], dtype=np.float64)
        r01_branch_snapshot = raw_r01.get("nominal", {}).get("branch_snapshot")
        if not isinstance(r01_branch_snapshot, Mapping):
            raise R02SourceError("immutable raw R01 result has no branch snapshot")
        # ReachSnapshot stores coordinates as tuples, while the content-bound
        # R01 artifact was loaded from JSON and therefore contains lists.  A
        # JSON round trip canonicalizes representation without introducing a
        # numerical state tolerance.
        branch_snapshot = _json_compatible(initial.to_dict())
        registered_witness = None
        registered_witness_actions: Optional[np.ndarray] = None
        if eligible:
            from .r02_directions import select_r01_changed_p_min_witness

            registered_witness = select_r01_changed_p_min_witness(raw_r01)
            registered_witness_actions = np.asarray(
                registered_witness.witness_prefix, dtype=np.float64
            )
        historical_r01 = _historical_r01_diagnostic(
            r01_nominal,
            eager_actions[:5, :7],
            witness_actions=registered_witness_actions,
        )
        pairing_checks = {
            "duplicate_eager_actions_exact": bool(np.array_equal(eager_actions, duplicate_actions)),
            "duplicate_eager_trace_exact": _same_trace(eager_trace, duplicate_trace),
            "branch_snapshot_equals_r01_exact": bool(
                branch_snapshot == dict(r01_branch_snapshot)
            ),
            "historical_r01_drift_within_frozen_limits": bool(
                historical_r01["passed"]
            ),
        }
        if not all(pairing_checks.values()):
            raise RuntimeError(f"R02 exact fresh-allocation pairing failed: {pairing_checks}")
        pairing = {
            "applicable": True,
            "passed": True,
            "policy_observation": _observation_fingerprint(policy_input),
            "branch_snapshot": branch_snapshot,
            "r01_branch_snapshot": dict(r01_branch_snapshot),
            "r01_nominal_actions": _array_record(r01_nominal, dtype=np.float64),
            "historical_r01_diagnostic": historical_r01,
            "compiled_actions": _array_record(compiled_actions),
            "eager_actions": _array_record(eager_actions),
            "duplicate_eager_actions": _array_record(duplicate_actions),
            "eager_trace": _trace_record(eager_trace),
            "duplicate_eager_trace": _trace_record(duplicate_trace),
            "compiled_vs_eager_diagnostic": _path_diagnostic(
                compiled_actions, eager_actions
            ),
            # Filled only after every intervention request to detect mutable
            # server/order contamination.
            "final_compiled_actions": None,
            "final_eager_actions": None,
            "final_eager_trace": None,
            "order_contamination_check_exact": None,
            "eager_order_contamination_check_exact": None,
            **pairing_checks,
        }

        frozen_arm = _evaluated_arm(
            "frozen",
            eager_actions,
            environment,
            initial,
            config,
            mechanism="eager explicit-noise trace-only frozen sampler",
            controls={
                "intervention_mode": "none",
                "return_trace": True,
                "policy_timing": _policy_timing_control(
                    policy_inference=True,
                    primary_reply=eager,
                    duplicate_reply=duplicate_eager,
                ),
            },
        )
        if frozen_arm["status"] == "bounds_failure":
            raise RuntimeError("frozen R01-reproducing policy unexpectedly violates action bounds")
        if frozen_arm.get("replay_exact") is not True:
            raise RuntimeError("R02 frozen simulator repeats are not exact")
        frozen_rollout = frozen_arm["repeats"][0]
        nominal_collision = bool(
            float(frozen_rollout["clearance_m"]) < 0.0
            or bool(frozen_rollout["contact"])
        )
        if not nominal_collision:
            r01_pointer: Optional[Mapping[str, Any]] = None
            pointer: Optional[Mapping[str, Any]] = None
            if eligible:
                if registered_witness is None or registered_witness_actions is None:
                    raise RuntimeError("eligible R02 case lost its bound R01 witness")
                pointer = _witness_pointer(
                    registered_witness, registered_witness_actions
                )
                r01_pointer = raw_r01.get("outcome", {}).get(
                    "selected_p_min_changed_witness"
                )
                expected_pointer = {
                    "search": pointer["search"],
                    "candidate_index": pointer["candidate_index"],
                    "source": pointer["source"],
                    "actions_sha256": pointer["actions_sha256"],
                }
                if r01_pointer != expected_pointer:
                    raise RuntimeError(
                        "recomputed R02 witness differs from R01 selected_p_min_changed_witness"
                    )
            _finalize_order_contamination_check(
                client,
                policy_input,
                noise,
                config,
                pairing,
                initial_compiled_actions=compiled_actions,
                initial_eager_actions=eager_actions,
                initial_eager_trace=eager_trace,
            )
            result = _reconfirmation_failure_result(
                case,
                config,
                status=NOMINAL_RECONFIRMATION_FAILURE,
                eligible=eligible,
                config_hash=config_hash,
                provenance=provenance,
                raw_r01_path=raw_path,
                raw_r01=raw_r01,
                raw_r01_sha256=raw_sha,
                pairing=pairing,
                frozen_arm=frozen_arm,
                r01_pointer=r01_pointer,
                witness_pointer=pointer,
            )
            errors = validate_r02_result(result)
            if errors:
                raise RuntimeError(
                    "refusing invalid R02 nominal-mismatch artifact: "
                    + "; ".join(errors)
                )
            atomic_write_json(output, result)
            return output, NOMINAL_RECONFIRMATION_FAILURE

        if not eligible:
            _finalize_order_contamination_check(
                client,
                policy_input,
                noise,
                config,
                pairing,
                initial_compiled_actions=compiled_actions,
                initial_eager_actions=eager_actions,
                initial_eager_trace=eager_trace,
            )
            result = _no_witness_result(
                case,
                config,
                config_hash=config_hash,
                provenance=provenance,
                raw_r01_path=raw_path,
                raw_r01=raw_r01,
                raw_r01_sha256=raw_sha,
                pairing=pairing,
                frozen_arm=frozen_arm,
            )
            errors = validate_r02_result(result)
            if errors:
                raise RuntimeError("refusing invalid R02 N/A artifact: " + "; ".join(errors))
            atomic_write_json(output, result)
            return output, "not_applicable_no_r01_witness"

        witness, pointer, direction_values, direction_failures, direction_diagnostics = _direction_bundle(
            raw_r01,
            config,
            eager_actions,
            np.asarray(eager_trace["predicted_clean_physical"], dtype=np.float64),
            frozen_rollout,
        )
        witness_actions = np.asarray(witness.witness_prefix, dtype=np.float64)
        if pointer["actions_array_sha256"] != _array_hash(witness_actions):
            raise RuntimeError("selected witness pointer does not bind its raw actions")
        r01_pointer = raw_r01.get("outcome", {}).get("selected_p_min_changed_witness")
        expected_pointer = {
            "search": pointer["search"],
            "candidate_index": pointer["candidate_index"],
            "source": pointer["source"],
            "actions_sha256": pointer["actions_sha256"],
        }
        if r01_pointer != expected_pointer:
            raise RuntimeError(
                "recomputed R02 witness differs from R01 selected_p_min_changed_witness"
            )
        direct_arm = _evaluated_arm(
            "direct_witness",
            witness_actions,
            environment,
            initial,
            config,
            mechanism="validated R01 direct physical action witness",
            controls={
                "intervention_mode": "direct_action_execution",
                "flow_arm": False,
                "policy_timing": _policy_timing_control(
                    policy_inference=False,
                    reason="direct arm executes the immutable R01 action witness",
                ),
            },
        )
        if direct_arm["status"] != "passed_gate":
            if direct_arm["status"] != "failed_gate":
                raise RuntimeError(
                    "R02 direct R01 witness could not be evaluated under the registered bounds"
                )
            _finalize_order_contamination_check(
                client,
                policy_input,
                noise,
                config,
                pairing,
                initial_compiled_actions=compiled_actions,
                initial_eager_actions=eager_actions,
                initial_eager_trace=eager_trace,
            )
            direction_records = _direction_records(
                direction_values,
                direction_failures,
                witness_pointer=pointer,
                config=config,
                diagnostics=direction_diagnostics,
            )
            result = _reconfirmation_failure_result(
                case,
                config,
                status=DIRECT_RECONFIRMATION_FAILURE,
                eligible=True,
                config_hash=config_hash,
                provenance=provenance,
                raw_r01_path=raw_path,
                raw_r01=raw_r01,
                raw_r01_sha256=raw_sha,
                pairing=pairing,
                frozen_arm=frozen_arm,
                r01_pointer=r01_pointer,
                witness_pointer=pointer,
                direction_records=direction_records,
                direct_arm=direct_arm,
            )
            errors = validate_r02_result(result)
            if errors:
                raise RuntimeError(
                    "refusing invalid R02 direct-mismatch artifact: "
                    + "; ".join(errors)
                )
            atomic_write_json(output, result)
            return output, DIRECT_RECONFIRMATION_FAILURE

        arms: Dict[str, Any] = {
            "frozen": frozen_arm,
            "direct_witness": direct_arm,
        }
        arm_specs = {
            "random_residual": (
                "endpoint-free equal-L2 random distributed residual",
                "residual",
                "random_model",
            ),
            "analytic_geometry_residual": (
                "analytic sphere/OBB equal-L2 distributed residual",
                "residual",
                "analytic_geometry_model",
            ),
            "oracle_residual": (
                "Delta_star distributed residual over remaining Euler interval",
                "residual",
                "delta_star_model",
            ),
            "bridge_diagnostic": (
                "Delta_star one-shot bridge edit diagnostic",
                "bridge_edit",
                "delta_star_model",
            ),
        }
        for name, (mechanism, mode, direction_key) in arm_specs.items():
            correction = direction_values.get(direction_key)
            if correction is None:
                arms[name] = _nonrollout_arm(
                    mechanism=mechanism,
                    status="direction_failure",
                    reason=direction_failures.get(direction_key) or "direction construction failed",
                    controls={
                        "intervention_mode": mode,
                        "intervention_step": config.oracle.intervention_step,
                        "correction_space": "model",
                    },
                )
                continue
            correction32 = np.asarray(correction, dtype=np.float32)
            reply = _infer(
                client,
                policy_input,
                noise,
                config,
                intervention_mode=mode,
                return_trace=True,
                correction=correction32,
            )
            reply_trace = _trace_arrays(reply)
            duplicate_reply = _infer(
                client,
                policy_input,
                noise,
                config,
                intervention_mode=mode,
                return_trace=True,
                correction=correction32,
            )
            duplicate_reply_trace = _trace_arrays(duplicate_reply)
            reply_actions = np.asarray(reply["actions"], dtype=np.float64)
            duplicate_reply_actions = np.asarray(
                duplicate_reply["actions"], dtype=np.float64
            )
            policy_replay_exact = bool(
                np.array_equal(reply_actions, duplicate_reply_actions)
                and _same_trace(reply_trace, duplicate_reply_trace)
            )
            if not policy_replay_exact:
                raise RuntimeError(f"{name} duplicate policy action/trace replay is not exact")
            if not _same_trace(reply_trace, eager_trace) or not _same_trace(
                duplicate_reply_trace, eager_trace
            ):
                raise RuntimeError(f"{name} diverged before the registered intervention")
            arms[name] = _evaluated_arm(
                name,
                reply_actions,
                environment,
                initial,
                config,
                mechanism=mechanism,
                controls={
                    "intervention_mode": mode,
                    "intervention_step": config.oracle.intervention_step,
                    "correction_space": "model",
                    "direction_key": direction_key,
                    "policy_timing": _policy_timing_control(
                        policy_inference=True,
                        primary_reply=reply,
                        duplicate_reply=duplicate_reply,
                    ),
                },
                correction=correction32,
                trace=reply_trace,
                frozen_trace=eager_trace,
                policy_replay_exact=policy_replay_exact,
                duplicate_actions=duplicate_reply_actions,
                duplicate_trace=duplicate_reply_trace,
            )
        _finalize_order_contamination_check(
            client,
            policy_input,
            noise,
            config,
            pairing,
            initial_compiled_actions=compiled_actions,
            initial_eager_actions=eager_actions,
            initial_eager_trace=eager_trace,
        )
        arms = {name: arms[name] for name in ARMS}
        direction_records = _direction_records(
            direction_values,
            direction_failures,
            witness_pointer=pointer,
            config=config,
            diagnostics=direction_diagnostics,
        )
        gate_pass = {
            name: (
                bool(arms[name]["gate"]["passed"])
                if arms[name]["status"] in EVALUATED_ARM_STATUSES
                else None
            )
            for name in ARMS
        }
        result = {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": ARTIFACT_TYPE,
            "gate": GATE,
            "case_id": case_id,
            "run_id": config.oracle.run_id,
            "status": "completed",
            "config_hash": config_hash,
            "source_evidence": {
                "r01_summary_sha256": config.r01_summary_sha256,
                "r01_ordered_result_set_digest": ACCEPTED_R01_ORDERED_RESULTS_SHA256,
                "r01_case_path": str(raw_path),
                "r01_case_sha256": raw_sha,
                "r01_case_validator": "validate_endpoint_free_result:passed",
                "r01_case_status": raw_r01["status"],
                "r01_selected_p_min_changed_witness": r01_pointer,
                "selected_witness_pointer": pointer,
                "sampler_parity_sha256": config.parity_artifact_sha256,
                "sampler_parity_status": "passed",
                "direction_reference": config.direction_reference,
                "direction_semantics_decision": config.direction_semantics_decision,
                "direction_semantics_decision_sha256": (
                    config.direction_semantics_decision_sha256
                ),
            },
            "provenance": _provenance_with_policy_timing(provenance, arms),
            "pairing": pairing,
            "directions": direction_records,
            "arms": arms,
            "outcome": {
                "population": "r01_changed_action_p_min_witness_conditioned",
                "r01_feasible_conditioned": True,
                "nominal_collision_reproduced": nominal_collision,
                "direct_witness_reconfirmed": gate_pass["direct_witness"],
                "population_mismatch_reason": None,
                "arm_gate_pass": gate_pass,
                "arm_status": {name: arms[name]["status"] for name in ARMS},
            },
        }
        errors = validate_r02_result(result)
        if errors:
            raise RuntimeError("refusing invalid R02 final artifact: " + "; ".join(errors))
        atomic_write_json(output, result)
        return output, "completed"
    finally:
        if owns_environment:
            environment.close()


__all__ = [
    "ARMS",
    "R02Config",
    "R02SourceError",
    "r02_config_from_mapping",
    "run_r02_case",
    "valid_r02_completion",
    "validate_r02_result",
]
