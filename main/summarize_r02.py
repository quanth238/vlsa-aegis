#!/usr/bin/env python3
"""Fail-closed R02 population validation and preregistered R03 analysis.

The summary is allocation-independent: it reads final artifacts and performs
no policy, simulator, CUDA, or rendering work.  The command-line entry point is
intended to run in the CPU Slurm verifier defined in ``slurm/r02_summary.sbatch``.
Synthetic fixtures may exercise the implementation through the Python API,
but they can never make the scientific gate pass.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
import re
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from crfs_harness.artifacts import (
    atomic_write_json,
    content_hash,
    file_sha256,
    load_json,
    validate_jsonl_unique,
)
from crfs_harness.manifest import validate_case


EXPECTED_CASES = 20
EXPECTED_FEASIBLE_CASES = 17
EXPECTED_STRUCTURAL_NO_WITNESS_CASES = 3
RESULT_FILENAME = "r02-paired.json"
FROZEN_MANIFEST_RELATIVE_PATH = Path("manifests/oracle_h05_colliding.jsonl")
REAL_EVIDENCE_TIER = "real_safelibero_r02_oracle_flow_preliminary"
SYNTHETIC_EVIDENCE_TIER = "synthetic_implementation_evidence_only"
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
REGISTERED_TRANSLATION_ACTION_SCALE = (0.8422505, 0.827813, 0.937313)
REGISTERED_EEF_RADIUS_M = 0.06
REGISTERED_DISTANCE_LIMIT_M = 1.0
REGISTERED_RESPONSE_MATRIX_M_PER_ACTION = (
    (0.009370281145853376, 0.00014623337157483132, -0.0009171200705674251),
    (0.0000039560765073778535, 0.012040711768393353, 0.0000006816739267718219),
    (-0.002845391485346128, 0.000025633541617775525, 0.011826427878652547),
)
PARITY_SEMANTICS_DECISION = "docs/decisions/0011-use-eager-path-for-r02-parity.md"
PROBE_AUTHORIZATION_DECISION = (
    "docs/decisions/0012-separate-r03-from-probe-authorization.md"
)
PROBE_AUTHORIZATION_DECISION_SHA256 = (
    "de04c05c38a47c1febd45d8f80e087e69adaca1f805c3844fca75ac79582912f"
)

BOOTSTRAP_SEED = 20260714
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_LOWER_QUANTILE = 0.025

ORACLE_MINIMUM_SPSR = 0.50
ORACLE_RANDOM_MINIMUM_DIFFERENCE = 0.20

RAW_ARMS = (
    "frozen",
    "direct_witness",
    "random_residual",
    "analytic_geometry_residual",
    "oracle_residual",
    "bridge_diagnostic",
)
ARM_LABELS = {
    "oracle": "oracle_residual",
    "random": "random_residual",
    "analytic": "analytic_geometry_residual",
    "frozen": "frozen",
    "bridge": "bridge_diagnostic",
    "direct": "direct_witness",
}
ROLLOUT_STATUSES = {"passed_gate", "failed_gate"}
NOT_EVALUATED_STATUS = "not_evaluated_after_reconfirmation_failure"
ELIGIBLE_FINAL_STATUSES = {
    "completed",
    "nominal_collision_not_reconfirmed",
    "direct_witness_not_reconfirmed",
}
NO_WITNESS_FINAL_STATUSES = {
    "not_applicable_no_r01_witness",
    "nominal_collision_not_reconfirmed",
}

Validator = Callable[[Mapping[str, Any]], List[str]]


@dataclass(frozen=True)
class SummaryContract:
    """Content-bound identities shared by every final R02 artifact."""

    manifest_cases: Tuple[Mapping[str, Any], ...]
    manifest_sha256: str
    frozen_manifest_sha256: str
    config_path: str
    config_file_sha256: str
    config_hash: str
    r01_summary_path: str
    r01_summary_sha256: str
    r01_ordered_result_set_digest: str
    r01_result_hashes: Mapping[str, str]
    r01_results_root: str
    eligible_case_ids: Tuple[str, ...]
    no_witness_case_ids: Tuple[str, ...]
    run_id: str
    checkpoint_id: str
    checkpoint_sha256: str
    sampler_parity_sha256: str
    probe_authorization_decision_path: str
    probe_authorization_decision_sha256: str


class SummaryContractError(ValueError):
    """Raised when frozen inputs cannot define one R02 result identity."""


def _is_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and re.fullmatch(r"[0-9a-f]{64}", value) is not None
    )


def _is_git_oid(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) in {40, 64}
        and re.fullmatch(r"[0-9a-f]+", value) is not None
    )


def _resolve(path_value: str, repo_root: Path) -> Path:
    path = Path(path_value).expanduser()
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _load_r02_validator() -> Validator:
    """Load the NumPy-backed authoritative validator only for real artifacts."""

    # validate_r02_result is the authoritative per-case validator.  It
    # independently recomputes bounds, repeats, D_sim gates, and arm outcomes.
    from crfs_oracle.r02_runner import (  # pylint: disable=import-outside-toplevel
        validate_r02_result,
    )

    return validate_r02_result


def _load_r01_validator() -> Validator:
    """Load the authoritative R01 validator in the allocation environment."""

    from crfs_oracle.endpoint_free_runner import (  # pylint: disable=import-outside-toplevel
        validate_endpoint_free_result,
    )

    return validate_endpoint_free_result


def _finite(value: Any, *, name: str, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise SummaryContractError(f"{name} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise SummaryContractError(f"{name} must be numeric") from error
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "positive and finite" if positive else "finite"
        raise SummaryContractError(f"{name} must be {qualifier}")
    return result


def _validate_r01_summary(value: Mapping[str, Any], actual_sha256: str) -> Tuple[
    Dict[str, str], Tuple[str, ...], Tuple[str, ...], float, str
]:
    if actual_sha256 != ACCEPTED_R01_SUMMARY_SHA256:
        raise SummaryContractError("R01 summary is not the accepted checked-in artifact")
    if value.get("schema_version") != "1.0" or value.get("gate") != "R01":
        raise SummaryContractError("R01 summary must be a schema_version 1.0 R01 artifact")
    if value.get("status") != "passed" or value.get("gate_passed") is not True:
        raise SummaryContractError("R01 summary must pass")
    ordered_digest = value.get("ordered_result_set_digest")
    if ordered_digest != ACCEPTED_R01_ORDERED_RESULTS_SHA256:
        raise SummaryContractError("R01 ordered result-set digest is not accepted")
    counts = value.get("counts")
    case_ids = value.get("case_ids")
    branch = value.get("branch_clearance")
    if not isinstance(counts, Mapping) or not isinstance(case_ids, Mapping):
        raise SummaryContractError("R01 summary is missing counts or case identities")
    if counts.get("expected_population") != 20 or counts.get("validated_population") != 20:
        raise SummaryContractError("R01 summary must validate the full 20-case population")
    eligible_value = case_ids.get("changed_action_p_min_rescues")
    no_witness_value = (
        branch.get("branch_below_registered_margin_case_ids")
        if isinstance(branch, Mapping)
        else None
    )
    if (
        not isinstance(eligible_value, list)
        or len(eligible_value) != EXPECTED_FEASIBLE_CASES
        or len(set(eligible_value)) != EXPECTED_FEASIBLE_CASES
    ):
        raise SummaryContractError("R01 summary must define 17 unique feasible cases")
    if (
        not isinstance(no_witness_value, list)
        or len(no_witness_value) != EXPECTED_STRUCTURAL_NO_WITNESS_CASES
        or len(set(no_witness_value)) != EXPECTED_STRUCTURAL_NO_WITNESS_CASES
    ):
        raise SummaryContractError("R01 summary must retain three structural negatives")
    if set(eligible_value) & set(no_witness_value):
        raise SummaryContractError("R01 feasible and structural identities overlap")
    result_hashes_value = value.get("result_hashes")
    if not isinstance(result_hashes_value, list) or len(result_hashes_value) != 20:
        raise SummaryContractError("R01 summary must bind all 20 raw result hashes")
    result_hashes: Dict[str, str] = {}
    for item in result_hashes_value:
        if not isinstance(item, Mapping):
            raise SummaryContractError("R01 result hash entries must be objects")
        case_id = item.get("case_id")
        sha256 = item.get("sha256")
        if not isinstance(case_id, str) or not case_id or not _is_sha256(sha256):
            raise SummaryContractError("R01 result hash entry is invalid")
        if case_id in result_hashes:
            raise SummaryContractError(f"duplicate R01 result hash for {case_id}")
        result_hashes[case_id] = str(sha256)
    if set(result_hashes) != set(eligible_value) | set(no_witness_value):
        raise SummaryContractError("R01 result hashes do not equal the registered 17+3 partition")
    identities = value.get("identities")
    p_min = _finite(
        identities.get("p_min_m") if isinstance(identities, Mapping) else None,
        name="R01 p_min",
        positive=True,
    )
    return (
        result_hashes,
        tuple(sorted(str(item) for item in eligible_value)),
        tuple(sorted(str(item) for item in no_witness_value)),
        p_min,
        str(ordered_digest),
    )


def _validate_parity_artifact(
    value: Mapping[str, Any],
    *,
    r01_summary_sha256: str,
    checkpoint_sha256: str,
) -> None:
    # Recompute the complete allocation gate from raw sampler arrays rather
    # than admitting a shallow, self-asserted ``status=passed`` artifact.
    from run_sampler_parity import (  # pylint: disable=import-outside-toplevel
        validate_parity_artifact as validate_raw_parity,
    )

    errors: List[str] = [
        f"authoritative parity validation: {error}"
        for error in validate_raw_parity(value)
    ]
    if value.get("schema_version") != "1.0":
        errors.append("schema_version must be 1.0")
    if value.get("artifact_type") != "sampler_parity" or value.get("gate") != "R02":
        errors.append("artifact must be the R02 sampler parity artifact")
    if value.get("status") != "passed":
        errors.append("status must be passed")
    identity = value.get("identity")
    if not isinstance(identity, Mapping):
        errors.append("identity must be an object")
    else:
        if identity.get("r01_summary_sha256") != r01_summary_sha256:
            errors.append("R01 summary binding differs")
        if identity.get("num_steps") != 10:
            errors.append("ten Euler steps are required")
        if identity.get("action_horizon") != 10 or identity.get("action_dim") != 32:
            errors.append("model action shape must be 10x32")
        if identity.get("git_dirty") is not False:
            errors.append("sampler parity must come from a clean allocation worktree")
    acceptance = value.get("acceptance")
    if not isinstance(acceptance, Mapping) or acceptance.get("passed") is not True:
        errors.append("acceptance must pass")
    else:
        checks = acceptance.get("checks")
        if not isinstance(checks, Mapping) or not checks or not all(
            check is True for check in checks.values()
        ):
            errors.append("every parity acceptance check must pass")
        if acceptance.get("path_identity_decision") != PARITY_SEMANTICS_DECISION:
            errors.append("parity artifact must use the ADR-0011 eager-path identity")
        if acceptance.get("primary_path") != (
            "public JAX default versus PyTorch eager trace-only"
        ):
            errors.append("parity primary path must be PyTorch eager trace-only")
    comparison = value.get("comparison")
    if not isinstance(comparison, Mapping):
        errors.append("comparison must be an object")
    else:
        for key, description in (
            ("cross_backend_pre_update_x_t_per_step", "latents"),
            ("cross_backend_pre_update_v_t_per_step", "velocities"),
        ):
            steps = comparison.get(key)
            if not isinstance(steps, list) or len(steps) != 10:
                errors.append(f"comparison must contain ten Euler {description}")
            elif [
                item.get("step_index") for item in steps if isinstance(item, Mapping)
            ] != list(range(10)) or not all(
                isinstance(item, Mapping) and item.get("passed") is True
                for item in steps
            ):
                errors.append(f"all ten indexed parity {description} must pass")
    checkpoints = value.get("checkpoints")
    converted = (
        checkpoints.get("converted_pytorch")
        if isinstance(checkpoints, Mapping)
        else None
    )
    if not isinstance(converted, Mapping):
        errors.append("converted PyTorch checkpoint record is missing")
    else:
        if converted.get("model_sha256") != checkpoint_sha256:
            errors.append("converted checkpoint hash differs")
        if converted.get("norm_stats_sha256") != NORMALIZATION_ASSET_SHA256:
            errors.append("normalization asset hash differs")
    if errors:
        raise SummaryContractError("invalid sampler parity artifact: " + "; ".join(errors))


def _validated_manifest(path: Path, repo_root: Path) -> Tuple[List[Mapping[str, Any]], str, str]:
    try:
        cases, errors = validate_jsonl_unique(path, "case_id")
    except (OSError, json.JSONDecodeError) as error:
        raise SummaryContractError(f"cannot read manifest: {error}") from error
    for case in cases:
        errors.extend(
            f"{case.get('case_id', '<unknown>')}: {item}"
            for item in validate_case(case)
        )
    case_ids = [str(case.get("case_id", "")) for case in cases]
    group_ids = [str(case.get("group_id", "")) for case in cases]
    if len(cases) != EXPECTED_CASES:
        errors.append(f"manifest must contain exactly {EXPECTED_CASES} cases")
    if len(set(case_ids)) != EXPECTED_CASES:
        errors.append("manifest must contain exactly 20 unique case identities")
    if len(set(group_ids)) != EXPECTED_CASES:
        errors.append("manifest must preserve exactly 20 unique state/episode groups")

    frozen_path = (repo_root / FROZEN_MANIFEST_RELATIVE_PATH).resolve()
    if not frozen_path.is_file():
        errors.append(f"checked-in frozen manifest is missing: {frozen_path}")
        frozen_sha256 = ""
    else:
        frozen_sha256 = file_sha256(frozen_path)
    observed_sha256 = file_sha256(path)
    if frozen_sha256 and observed_sha256 != frozen_sha256:
        errors.append("input manifest content differs from the frozen 20-case manifest")
    if errors:
        raise SummaryContractError("invalid frozen manifest: " + "; ".join(errors))
    return list(cases), observed_sha256, frozen_sha256


def _runtime_config_value(value: Mapping[str, Any]) -> Dict[str, Any]:
    """Apply exactly the defaults used by ``main/run_crfs_r02.py``."""

    runtime = dict(value)
    settings = runtime.get("r02")
    if not isinstance(settings, Mapping):
        raise SummaryContractError("config requires an r02 object")
    runtime.setdefault("intervention_step", 5)
    runtime.setdefault("optimizer_max_iterations", 1)
    runtime.setdefault("measurement_repeats", int(settings.get("simulator_repeats", 2)))
    runtime.setdefault("stop_after_measurement", False)
    return runtime


def load_summary_contract(
    manifest: str | Path,
    config: str | Path,
    r01_summary: str | Path,
    *,
    run_id: str,
    checkpoint_id: str,
    checkpoint_sha256: str,
    repo_root: str | Path,
) -> SummaryContract:
    """Validate all frozen inputs and reproduce the runner's config hash."""

    root = Path(repo_root).resolve()
    manifest_path = Path(manifest).resolve()
    config_path = Path(config).resolve()
    r01_summary_path = Path(r01_summary).resolve()
    if not re.fullmatch(r"[A-Za-z0-9._-]+", run_id):
        raise SummaryContractError("run_id must use only letters, digits, dot, underscore, or hyphen")
    if not checkpoint_id:
        raise SummaryContractError("checkpoint_id must be non-empty")
    if not _is_sha256(checkpoint_sha256):
        raise SummaryContractError("checkpoint_sha256 must be a lowercase SHA-256")

    cases, manifest_sha256, frozen_manifest_sha256 = _validated_manifest(
        manifest_path, root
    )
    try:
        config_value = load_json(config_path)
    except (OSError, json.JSONDecodeError) as error:
        raise SummaryContractError(f"cannot read R02 config: {error}") from error
    if not isinstance(config_value, Mapping):
        raise SummaryContractError("R02 config must be an object")
    if config_value.get("ready_to_run") is not True or config_value.get("blocked_on", []) != []:
        raise SummaryContractError("R02 config must be ready_to_run with no blocked dependencies")
    declared_manifest = config_value.get("manifest")
    if not isinstance(declared_manifest, str) or _resolve(declared_manifest, root) != manifest_path:
        raise SummaryContractError("config manifest does not resolve to the supplied frozen manifest")
    settings = config_value.get("r02")
    if not isinstance(settings, Mapping):
        raise SummaryContractError("R02 config has no r02 object")
    declared_summary = settings.get("r01_summary_artifact")
    if not isinstance(declared_summary, str) or _resolve(declared_summary, root) != r01_summary_path:
        raise SummaryContractError("config R01 summary does not resolve to the supplied artifact")
    try:
        r01_summary_path.relative_to((root / "evidence" / "r01").resolve())
    except ValueError as error:
        raise SummaryContractError("R01 summary must be checked-in under evidence/r01") from error
    try:
        summary_value = load_json(r01_summary_path)
    except (OSError, json.JSONDecodeError) as error:
        raise SummaryContractError(f"cannot read R01 summary: {error}") from error
    if not isinstance(summary_value, Mapping):
        raise SummaryContractError("R01 summary must be an object")
    actual_r01_sha256 = file_sha256(r01_summary_path)
    if settings.get("r01_summary_sha256") != actual_r01_sha256:
        raise SummaryContractError("config R01 summary hash differs from the supplied artifact")
    (
        r01_result_hashes,
        eligible_case_ids,
        no_witness_case_ids,
        p_min_m,
        ordered_digest,
    ) = _validate_r01_summary(summary_value, actual_r01_sha256)

    manifest_ids = {str(case["case_id"]) for case in cases}
    if set(r01_result_hashes) != manifest_ids:
        raise SummaryContractError("R01 result identities differ from the frozen R02 manifest")
    if set(eligible_case_ids) | set(no_witness_case_ids) != manifest_ids:
        raise SummaryContractError("R01 feasible and structural populations do not partition the manifest")

    runtime_value = _runtime_config_value(config_value)
    if runtime_value.get("response_matrix_m_per_action") is None:
        raise SummaryContractError("R02 analytic geometry requires the frozen response matrix")
    try:
        response_matrix = tuple(
            tuple(float(item) for item in row)
            for row in runtime_value["response_matrix_m_per_action"]
        )
        oracle_config = {
            "resize_size": int(runtime_value["resize_size"]),
            "settle_steps": int(runtime_value["settle_steps"]),
            "executed_prefix": int(runtime_value["executed_prefix"]),
            "action_horizon": int(runtime_value["action_horizon"]),
            "action_dim": int(runtime_value["action_dim"]),
            "sampler_steps": int(runtime_value["sampler_steps"]),
            "intervention_step": int(runtime_value["intervention_step"]),
            "safety_margin_m": float(runtime_value["safety_margin_m"]),
            "distance_limit_m": float(runtime_value["distance_limit_m"]),
            "eef_radius_m": float(runtime_value["eef_radius_m"]),
            "measurement_repeats": int(runtime_value.get("measurement_repeats", 2)),
            "stop_after_measurement": bool(runtime_value.get("stop_after_measurement", False)),
            "response_matrix_m_per_action": response_matrix,
            "optimizer_max_iterations": int(runtime_value["optimizer_max_iterations"]),
            "checkpoint_id": checkpoint_id,
            "checkpoint_sha256": checkpoint_sha256,
        }
    except (KeyError, TypeError, ValueError) as error:
        raise SummaryContractError(f"invalid top-level R02 sampler config: {error}") from error
    if (
        oracle_config["action_horizon"] != 10
        or oracle_config["action_dim"] != 32
        or oracle_config["executed_prefix"] != 5
        or oracle_config["sampler_steps"] != 10
        or oracle_config["intervention_step"] != 5
    ):
        raise SummaryContractError("R02 requires 10x32 actions, five executed actions, and step 5 of 10")
    if response_matrix != REGISTERED_RESPONSE_MATRIX_M_PER_ACTION:
        raise SummaryContractError("R02 response matrix differs from the frozen H04 calibration")
    if not math.isclose(
        float(oracle_config["eef_radius_m"]),
        REGISTERED_EEF_RADIUS_M,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise SummaryContractError("R02 D_sim/D_opt EEF radius must remain 6 cm")
    if not math.isclose(
        float(oracle_config["distance_limit_m"]),
        REGISTERED_DISTANCE_LIMIT_M,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise SummaryContractError("R02 MuJoCo distance limit must remain 1 m")
    if settings.get("phase") != "pregrasp_reach":
        raise SummaryContractError("R02 phase must remain pregrasp_reach")
    if tuple(settings.get("required_arms", ())) != RAW_ARMS:
        raise SummaryContractError("R02 required arms differ from the frozen ordered arms")
    if settings.get("clipping_policy") != "fail_without_clipping":
        raise SummaryContractError("R02 must fail sampled bounds without clipping")
    if settings.get("target_object") != "akita_black_bowl_1":
        raise SummaryContractError("R02 target object differs from the frozen reach target")
    declared_p_min = _finite(
        settings.get("minimum_progress_m"), name="R02 minimum progress", positive=True
    )
    if not math.isclose(declared_p_min, p_min_m, rel_tol=0.0, abs_tol=1e-15):
        raise SummaryContractError("R02 minimum progress differs from the accepted R01 summary")
    simulator_margin = _finite(
        settings.get("simulator_safety_margin_m"),
        name="R02 simulator safety margin",
        positive=True,
    )
    if not math.isclose(simulator_margin, 0.005, rel_tol=0.0, abs_tol=1e-12):
        raise SummaryContractError("R02 simulator safety margin must remain 5 mm")
    if not math.isclose(
        float(oracle_config["safety_margin_m"]), simulator_margin, rel_tol=0.0, abs_tol=1e-12
    ):
        raise SummaryContractError("R02 top-level and arm safety margins differ")
    target_limit = _finite(
        settings.get("maximum_target_displacement_m"), name="target motion limit"
    )
    obstacle_limit = _finite(
        settings.get("maximum_obstacle_displacement_m"), name="obstacle motion limit"
    )
    if not (0.0 <= target_limit <= 0.001 and 0.0 <= obstacle_limit <= 0.001):
        raise SummaryContractError("R02 scene-motion limits must lie in [0, 1 mm]")
    repeats = int(settings.get("simulator_repeats", oracle_config["measurement_repeats"]))
    if repeats < 2 or repeats != oracle_config["measurement_repeats"]:
        raise SummaryContractError("R02 simulator repeats must agree and be at least two")
    bounds = settings.get("translation_action_bounds")
    if not isinstance(bounds, Sequence) or isinstance(bounds, (str, bytes)) or len(bounds) != 2:
        raise SummaryContractError("R02 translation action bounds must be [low, high]")
    low = _finite(bounds[0], name="translation action low")
    high = _finite(bounds[1], name="translation action high")
    if low != -1.0 or high != 1.0:
        raise SummaryContractError("R02 translation action bounds must remain [-1, 1]")
    scale_value = settings.get("normalization_action_scale")
    if (
        not isinstance(scale_value, Sequence)
        or isinstance(scale_value, (str, bytes))
        or len(scale_value) != 3
    ):
        raise SummaryContractError("R02 normalization action scale must have three values")
    scale = tuple(
        _finite(item, name="normalization action scale", positive=True)
        for item in scale_value
    )
    if any(
        not math.isclose(item, expected, rel_tol=0.0, abs_tol=1e-12)
        for item, expected in zip(scale, REGISTERED_TRANSLATION_ACTION_SCALE)
    ):
        raise SummaryContractError("R02 normalization action scale is not registered")
    if settings.get("normalization_asset_sha256") != NORMALIZATION_ASSET_SHA256:
        raise SummaryContractError("R02 normalization asset hash is not registered")
    samples_per_segment = int(settings.get("analytic_samples_per_segment", 26))
    if samples_per_segment != 26:
        raise SummaryContractError("R02 analytic geometry must use 26 samples per segment")

    r01_results_root = settings.get("r01_results_root")
    if not isinstance(r01_results_root, str) or not r01_results_root:
        raise SummaryContractError("R02 config must locate the immutable raw R01 results")
    resolved_r01_results_root = _resolve(r01_results_root, root)
    parity_value = settings.get("sampler_parity_artifact")
    parity_sha256 = settings.get("sampler_parity_sha256")
    if not isinstance(parity_value, str) or not parity_value or not _is_sha256(parity_sha256):
        raise SummaryContractError("R02 config must bind a sampler parity path and SHA-256")
    parity_path = _resolve(parity_value, root)
    try:
        actual_parity_sha256 = file_sha256(parity_path)
        parity_artifact = load_json(parity_path)
    except (OSError, json.JSONDecodeError) as error:
        raise SummaryContractError(f"cannot read sampler parity artifact: {error}") from error
    if actual_parity_sha256 != parity_sha256:
        raise SummaryContractError("sampler parity file hash differs from the R02 config")
    if not isinstance(parity_artifact, Mapping):
        raise SummaryContractError("sampler parity artifact must be an object")
    _validate_parity_artifact(
        parity_artifact,
        r01_summary_sha256=actual_r01_sha256,
        checkpoint_sha256=checkpoint_sha256,
    )

    authorization_path = (root / PROBE_AUTHORIZATION_DECISION).resolve()
    try:
        authorization_sha256 = file_sha256(authorization_path)
    except OSError as error:
        raise SummaryContractError(f"cannot read ADR-0012: {error}") from error
    if authorization_sha256 != PROBE_AUTHORIZATION_DECISION_SHA256:
        raise SummaryContractError("checked-in ADR-0012 content differs from the frozen authorization rule")

    normalized_config = {
        **oracle_config,
        "target_name": "akita_black_bowl_1",
        "p_min_m": p_min_m,
        "simulator_safety_margin_m": simulator_margin,
        "maximum_target_displacement_m": target_limit,
        "maximum_obstacle_displacement_m": obstacle_limit,
        "simulator_repeats": repeats,
        "translation_action_bounds": [low, high],
        "normalization_action_scale": list(scale),
        "normalization_asset_sha256": NORMALIZATION_ASSET_SHA256,
        "analytic_samples_per_segment": samples_per_segment,
        "required_arms": list(RAW_ARMS),
        "clipping_policy": "fail_without_clipping",
        "r01_summary_sha256": actual_r01_sha256,
        "r01_ordered_result_set_digest": ordered_digest,
        "r01_result_hashes": dict(sorted(r01_result_hashes.items())),
        "eligible_case_ids": list(eligible_case_ids),
        "no_witness_case_ids": list(no_witness_case_ids),
        "sampler_parity_sha256": actual_parity_sha256,
    }
    return SummaryContract(
        manifest_cases=tuple(dict(case) for case in cases),
        manifest_sha256=manifest_sha256,
        frozen_manifest_sha256=frozen_manifest_sha256,
        config_path=str(config_path),
        config_file_sha256=file_sha256(config_path),
        config_hash=content_hash(normalized_config),
        r01_summary_path=str(r01_summary_path),
        r01_summary_sha256=actual_r01_sha256,
        r01_ordered_result_set_digest=str(ordered_digest),
        r01_result_hashes=dict(r01_result_hashes),
        r01_results_root=str(resolved_r01_results_root),
        eligible_case_ids=eligible_case_ids,
        no_witness_case_ids=no_witness_case_ids,
        run_id=run_id,
        checkpoint_id=checkpoint_id,
        checkpoint_sha256=checkpoint_sha256,
        sampler_parity_sha256=actual_parity_sha256,
        probe_authorization_decision_path=str(authorization_path),
        probe_authorization_decision_sha256=authorization_sha256,
    )


def _identity_errors(
    value: Mapping[str, Any],
    *,
    folder_case_id: str,
    manifest_case: Optional[Mapping[str, Any]],
    contract: SummaryContract,
    allow_synthetic_implementation_evidence: bool,
) -> List[str]:
    errors: List[str] = []
    if value.get("case_id") != folder_case_id:
        errors.append("artifact case_id differs from its result directory")
    if value.get("run_id") != contract.run_id:
        errors.append("artifact run_id differs from the registered run")
    if value.get("config_hash") != contract.config_hash:
        errors.append("artifact config_hash differs from the independently reproduced config hash")
    if manifest_case is None:
        return errors + ["artifact directory is not a frozen manifest case"]

    expected_case_id = str(manifest_case["case_id"])
    expected_eligible = expected_case_id in set(contract.eligible_case_ids)
    allowed_statuses = (
        ELIGIBLE_FINAL_STATUSES if expected_eligible else NO_WITNESS_FINAL_STATUSES
    )
    status = value.get("status")
    if status not in allowed_statuses:
        errors.append(
            f"artifact status {status!r} is not a terminal status for its R01 population"
        )
    outcome = value.get("outcome")
    if isinstance(outcome, Mapping):
        if status == "nominal_collision_not_reconfirmed" and outcome.get(
            "nominal_collision_reproduced"
        ) is not False:
            errors.append("nominal mismatch status requires a false nominal outcome")
        if status == "direct_witness_not_reconfirmed" and (
            outcome.get("nominal_collision_reproduced") is not True
            or outcome.get("direct_witness_reconfirmed") is not False
        ):
            errors.append("direct mismatch status requires nominal=true and direct=false")
        if status in {
            "nominal_collision_not_reconfirmed",
            "direct_witness_not_reconfirmed",
        } and outcome.get("population_mismatch_reason") != status:
            errors.append("population_mismatch_reason must equal the terminal mismatch status")
        if status not in {
            "nominal_collision_not_reconfirmed",
            "direct_witness_not_reconfirmed",
        } and outcome.get("population_mismatch_reason") is not None:
            errors.append("non-mismatch completion cannot claim a population mismatch reason")

    source = value.get("source_evidence")
    if not isinstance(source, Mapping):
        errors.append("source_evidence must be an object")
    else:
        expected_source = {
            "r01_summary_sha256": contract.r01_summary_sha256,
            "r01_ordered_result_set_digest": contract.r01_ordered_result_set_digest,
            "r01_case_sha256": contract.r01_result_hashes.get(expected_case_id),
            "sampler_parity_sha256": contract.sampler_parity_sha256,
        }
        for key, expected in expected_source.items():
            if source.get(key) != expected:
                errors.append(f"source_evidence.{key} differs from frozen input evidence")

    provenance = value.get("provenance")
    if not isinstance(provenance, Mapping):
        return errors + ["provenance must be an object"]
    evidence_tier = provenance.get("evidence_tier")
    if evidence_tier != REAL_EVIDENCE_TIER:
        if not (
            allow_synthetic_implementation_evidence
            and evidence_tier == SYNTHETIC_EVIDENCE_TIER
        ):
            errors.append("provenance evidence tier is not real allocation-backed R02 evidence")
    if provenance.get("git_dirty") is not False:
        errors.append("provenance.git_dirty must be false")
    if not _is_git_oid(provenance.get("git_commit")):
        errors.append("provenance.git_commit must be a full Git object ID")
    if not _is_git_oid(provenance.get("baseline_commit")):
        errors.append("provenance.baseline_commit must be a full Git object ID")
    elif provenance.get("baseline_commit") != BASELINE_COMMIT:
        errors.append("provenance.baseline_commit differs from the frozen baseline")
    if provenance.get("case_record") != dict(manifest_case):
        errors.append("provenance.case_record differs from the frozen manifest record")
    for key in (
        "case_id",
        "group_id",
        "task_suite",
        "safety_level",
        "task_index",
        "episode_index",
        "environment_seed",
        "policy_seed",
        "random_control_seed",
    ):
        if key == "case_id":
            continue
        if provenance.get(key) != manifest_case.get(key):
            errors.append(f"provenance.{key} differs from the frozen manifest")
    expected_hashes = {
        "input_manifest_sha256": contract.manifest_sha256,
        "checkpoint_sha256": contract.checkpoint_sha256,
        "r01_summary_sha256": contract.r01_summary_sha256,
        "r01_case_result_sha256": contract.r01_result_hashes.get(expected_case_id),
        "sampler_parity_sha256": contract.sampler_parity_sha256,
    }
    for key, expected in expected_hashes.items():
        if provenance.get(key) != expected:
            errors.append(f"provenance.{key} differs from the registered identity")
    if provenance.get("checkpoint_id") != contract.checkpoint_id:
        errors.append("provenance.checkpoint_id differs from the registered checkpoint")
    for key in ("slurm_job_id", "slurm_array_task_id"):
        item = provenance.get(key)
        if not isinstance(item, str) or not item.isdigit():
            errors.append(f"provenance.{key} must be a numeric Slurm allocation identifier")
    for key in ("host", "partition", "device"):
        if not isinstance(provenance.get(key), str) or not provenance.get(key):
            errors.append(f"provenance.{key} must be a non-empty allocation value")
    return errors


def _load_raw_r01_population(
    contract: SummaryContract,
    manifest_cases: Mapping[str, Mapping[str, Any]],
    validator: Validator,
) -> Tuple[Dict[str, Mapping[str, Any]], Dict[str, List[str]], List[Dict[str, str]]]:
    """Hash and authoritatively validate every immutable raw R01 source."""

    records: Dict[str, Mapping[str, Any]] = {}
    invalid: Dict[str, List[str]] = {}
    hashes: List[Dict[str, str]] = []
    root = Path(contract.r01_results_root)
    eligible = set(contract.eligible_case_ids)
    for case_id, manifest_case in sorted(manifest_cases.items()):
        path = root / case_id / "endpoint-free-feasibility.json"
        errors: List[str] = []
        try:
            actual_sha256 = file_sha256(path)
            value = load_json(path)
        except (OSError, json.JSONDecodeError) as error:
            invalid[case_id] = [f"cannot read immutable raw R01 artifact: {error}"]
            continue
        expected_sha256 = contract.r01_result_hashes.get(case_id)
        if actual_sha256 != expected_sha256:
            errors.append(
                f"raw R01 SHA-256 differs: expected {expected_sha256}, got {actual_sha256}"
            )
        if not isinstance(value, Mapping):
            errors.append("raw R01 artifact must be an object")
        else:
            errors.extend(f"authoritative R01 validation: {item}" for item in validator(value))
            if value.get("case_id") != case_id:
                errors.append("raw R01 case_id differs from the frozen manifest")
            provenance = value.get("provenance")
            if not isinstance(provenance, Mapping) or provenance.get("case_record") != dict(
                manifest_case
            ):
                errors.append("raw R01 provenance.case_record differs from the frozen manifest")
            expected_status = (
                "verified_safe_progress" if case_id in eligible else "no_verified_safe_progress"
            )
            if value.get("status") != expected_status:
                errors.append(
                    f"raw R01 status must be {expected_status!r} for its frozen population"
                )
        hashes.append({"case_id": case_id, "path": str(path), "sha256": actual_sha256})
        if errors:
            invalid[case_id] = sorted(set(errors))
        else:
            records[case_id] = value
    return records, invalid, hashes


def _raw_geometry(value: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(value, Mapping):
        return None
    if "start_eef_center_m" not in value or "branch_obstacle_boxes" not in value:
        return None
    return {
        "start_eef_center_m": value["start_eef_center_m"],
        "branch_obstacle_boxes": value["branch_obstacle_boxes"],
    }


def _raw_r01_binding_errors(
    r02_value: Mapping[str, Any],
    raw_r01: Mapping[str, Any],
) -> List[str]:
    """Compare duplicated pairing/geometry fields to content-bound raw R01."""

    errors: List[str] = []
    nominal = raw_r01.get("nominal")
    pairing = r02_value.get("pairing")
    if not isinstance(nominal, Mapping) or not isinstance(pairing, Mapping):
        return ["raw R01 nominal evidence and R02 pairing must be objects"]
    raw_branch = nominal.get("branch_snapshot")
    if not isinstance(raw_branch, Mapping):
        errors.append("raw R01 nominal branch_snapshot must be an object")
    else:
        for key in ("r01_branch_snapshot", "branch_snapshot"):
            if pairing.get(key) != dict(raw_branch):
                errors.append(f"pairing.{key} differs from content-bound raw R01")
    raw_actions = nominal.get("actions")
    r01_actions_record = pairing.get("r01_nominal_actions")
    if not isinstance(r01_actions_record, Mapping):
        errors.append("pairing.r01_nominal_actions must be an array record")
    elif r01_actions_record.get("values") != raw_actions:
        errors.append("pairing.r01_nominal_actions differs from content-bound raw R01")

    raw_repeats = nominal.get("repeats")
    if not isinstance(raw_repeats, list) or not raw_repeats:
        errors.append("raw R01 nominal repeats are missing")
        reference_geometry = None
    else:
        reference_geometry = _raw_geometry(raw_repeats[0])
        if reference_geometry is None:
            errors.append("raw R01 nominal repeat has no branch geometry")
        elif any(_raw_geometry(item) != reference_geometry for item in raw_repeats[1:]):
            errors.append("raw R01 nominal repeat geometry is not exact across repeats")
    arms = r02_value.get("arms")
    frozen = arms.get("frozen") if isinstance(arms, Mapping) else None
    frozen_repeats = frozen.get("repeats") if isinstance(frozen, Mapping) else None
    if not isinstance(frozen_repeats, list) or not frozen_repeats:
        errors.append("R02 frozen arm has no simulator repeats")
    elif reference_geometry is not None:
        for index, repeat in enumerate(frozen_repeats):
            if _raw_geometry(repeat) != reference_geometry:
                errors.append(
                    f"R02 frozen repeat {index} geometry differs from content-bound raw R01"
                )
    return errors


def _arm_status(value: Mapping[str, Any], raw_name: str) -> str:
    arms = value.get("arms")
    if not isinstance(arms, Mapping) or not isinstance(arms.get(raw_name), Mapping):
        return "invalid"
    return str(arms[raw_name].get("status", "invalid"))


def _arm_success(value: Mapping[str, Any], raw_name: str) -> int:
    return int(_arm_status(value, raw_name) == "passed_gate")


def _arm_population_metrics(
    records: Mapping[str, Mapping[str, Any]],
    eligible_case_ids: Sequence[str],
    no_witness_case_ids: Sequence[str],
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    feasible: Dict[str, Any] = {}
    intent_to_treat: Dict[str, Any] = {}
    observed: Dict[str, Any] = {}
    failures: Dict[str, Any] = {}
    eligible = list(eligible_case_ids)
    all_case_ids = eligible + list(no_witness_case_ids)
    structural_count = len(no_witness_case_ids)
    for label, raw_name in ARM_LABELS.items():
        eligible_statuses = {
            case_id: _arm_status(records[case_id], raw_name)
            for case_id in eligible
        }
        feasible_successes = sum(
            status == "passed_gate" for status in eligible_statuses.values()
        )
        # The frozen arm is executed for all 20 cases. Every other arm needs an
        # R01 witness, so its three no-witness cases are zeros only in ITT.
        analysis_ids = all_case_ids if label == "frozen" else eligible
        analysis_statuses = {
            case_id: _arm_status(records[case_id], raw_name)
            for case_id in analysis_ids
        }
        intent_successes = sum(
            status == "passed_gate" for status in analysis_statuses.values()
        )
        arm_structural_count = 0 if label == "frozen" else structural_count
        evaluated_ids = sorted(
            case_id
            for case_id, status in analysis_statuses.items()
            if status in ROLLOUT_STATUSES
        )
        bounds_ids = sorted(
            case_id
            for case_id, status in analysis_statuses.items()
            if status == "bounds_failure"
        )
        direction_ids = sorted(
            case_id
            for case_id, status in analysis_statuses.items()
            if status == "direction_failure"
        )
        not_evaluated_ids = sorted(
            case_id
            for case_id in all_case_ids
            if _arm_status(records[case_id], raw_name) == NOT_EVALUATED_STATUS
        )
        feasible[label] = {
            "raw_arm": raw_name,
            "successes": feasible_successes,
            "denominator": len(eligible),
            "spsr": feasible_successes / len(eligible),
            "structural_no_witness_zeroes": 0,
        }
        intent_to_treat[label] = {
            "raw_arm": raw_name,
            "successes": intent_successes,
            "denominator": len(eligible) + structural_count,
            "spsr": intent_successes / (len(eligible) + structural_count),
            "structural_no_witness_zeroes": arm_structural_count,
        }
        observed[label] = {
            "scope": (
                "all 20 actually rolled-out frozen cases"
                if label == "frozen"
                else "actually rolled-out R01-feasible cases only"
            ),
            "successes": intent_successes,
            "evaluated_rollouts": len(evaluated_ids),
            "success_rate": intent_successes / len(evaluated_ids) if evaluated_ids else None,
            "evaluated_case_ids": evaluated_ids,
            "structural_no_witness_cases_in_denominator": 0,
        }
        failures[label] = {
            "bounds_failure_count": len(bounds_ids),
            "bounds_failure_case_ids": bounds_ids,
            "direction_failure_count": len(direction_ids),
            "direction_failure_case_ids": direction_ids,
            "not_evaluated_after_reconfirmation_failure_count": len(
                not_evaluated_ids
            ),
            "not_evaluated_after_reconfirmation_failure_case_ids": not_evaluated_ids,
            "failed_gate_count": sum(
                status == "failed_gate" for status in analysis_statuses.values()
            ),
            "failed_gate_case_ids": sorted(
                case_id
                for case_id, status in analysis_statuses.items()
                if status == "failed_gate"
            ),
        }
    return feasible, intent_to_treat, observed, failures


def _inverted_cdf(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must lie in [0, 1]")
    ordered = sorted(float(value) for value in values)
    rank = max(0, min(len(ordered) - 1, math.ceil(probability * len(ordered)) - 1))
    return ordered[rank]


def _launch_failure_audit(
    results_root: Path,
    *,
    expected_case_ids: Sequence[str],
    final_case_ids: Sequence[str],
) -> Dict[str, Any]:
    """Expose allocation failures without treating them as final artifacts."""

    expected = set(expected_case_ids)
    finals = set(final_case_ids)
    records: List[Dict[str, Any]] = []
    invalid_records: List[Dict[str, Any]] = []
    for path in sorted(results_root.rglob("launch-failure.json")) if results_root.exists() else []:
        try:
            value = load_json(path)
        except (OSError, json.JSONDecodeError) as error:
            invalid_records.append({"path": str(path), "errors": [f"cannot read: {error}"]})
            continue
        if not isinstance(value, Mapping):
            invalid_records.append({"path": str(path), "errors": ["must be an object"]})
            continue
        errors: List[str] = []
        case_id = value.get("case_id")
        stage = value.get("stage")
        exit_code = value.get("exit_code")
        policy_server_log = value.get("policy_server_log")
        client_log = value.get("client_log")
        if not isinstance(case_id, str) or not case_id:
            errors.append("case_id must be non-empty")
        elif case_id not in expected:
            errors.append("case_id is outside the frozen manifest")
        if not isinstance(stage, str) or not stage:
            errors.append("stage must be non-empty")
        if isinstance(exit_code, bool) or not isinstance(exit_code, int) or exit_code == 0:
            errors.append("exit_code must be a nonzero integer")
        for key, item in (
            ("policy_server_log", policy_server_log),
            ("client_log", client_log),
        ):
            if not isinstance(item, str) or not item:
                errors.append(f"{key} must be a non-empty path")
        record = {
            "path": str(path),
            "case_id": case_id,
            "stage": stage,
            "exit_code": exit_code,
            "policy_server_log": policy_server_log,
            "client_log": client_log,
            "run_id": value.get("run_id"),
            "case_index": value.get("case_index"),
            "job_id": value.get("job_id"),
            "array_job_id": value.get("array_job_id"),
            "array_task_id": value.get("array_task_id"),
            "has_final_artifact": isinstance(case_id, str) and case_id in finals,
        }
        if errors:
            invalid_records.append({**record, "errors": errors})
        else:
            records.append(record)
    valid_case_ids = sorted(
        {str(record["case_id"]) for record in records if record.get("case_id")}
    )
    return {
        "found": len(records) + len(invalid_records),
        "valid_records": records,
        "invalid_records": invalid_records,
        "case_ids": valid_case_ids,
        "missing_final_case_ids_with_launch_failure": sorted(
            case_id for case_id in valid_case_ids if case_id not in finals
        ),
        "case_ids_with_both_failure_and_final_artifact": sorted(
            case_id for case_id in valid_case_ids if case_id in finals
        ),
        "population_credit": 0,
        "interpretation": "launch failures are audit evidence only and never satisfy a final case",
    }


def grouped_paired_bootstrap_lcb(
    pairs: Sequence[Tuple[str, int, int]],
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
    lower_quantile: float = BOOTSTRAP_LOWER_QUANTILE,
) -> Dict[str, Any]:
    """Bootstrap complete state/episode groups, never trials or frames."""

    if replicates < 10_000:
        raise ValueError("registered grouped bootstrap requires at least 10,000 replicates")
    grouped: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    for group_id, oracle_success, random_success in pairs:
        if not group_id:
            raise ValueError("bootstrap group_id must be non-empty")
        if oracle_success not in {0, 1} or random_success not in {0, 1}:
            raise ValueError("paired outcomes must be binary")
        grouped[group_id].append((oracle_success, random_success))
    group_ids = sorted(grouped)
    if not group_ids:
        raise ValueError("grouped bootstrap requires at least one complete group")

    rng = random.Random(seed)
    estimates: List[float] = []
    for _ in range(replicates):
        oracle_total = 0
        random_total = 0
        case_total = 0
        for _group_index in range(len(group_ids)):
            sampled_id = group_ids[rng.randrange(len(group_ids))]
            members = grouped[sampled_id]
            oracle_total += sum(item[0] for item in members)
            random_total += sum(item[1] for item in members)
            case_total += len(members)
        estimates.append((oracle_total - random_total) / case_total)
    lower_bound = _inverted_cdf(estimates, lower_quantile)
    return {
        "unit": "complete state/episode group",
        "group_count": len(group_ids),
        "group_ids": group_ids,
        "replicates": replicates,
        "seed": seed,
        "method": "fixed-seed nonparametric percentile bootstrap; complete groups resampled with replacement",
        "confidence_level": 0.95,
        "lower_quantile": lower_quantile,
        "quantile_convention": "inverted_cdf (ceil(p*n)-1)",
        "lower_confidence_bound": lower_bound,
    }


def exact_one_sided_paired_discordance(
    pairs: Sequence[Tuple[str, int, int]],
) -> Dict[str, Any]:
    """Exact one-sided paired label-swap tail for first-arm superiority."""

    first_only = 0
    second_only = 0
    for group_id, first_success, second_success in pairs:
        if not group_id:
            raise ValueError("paired discordance group_id must be non-empty")
        if first_success not in {0, 1} or second_success not in {0, 1}:
            raise ValueError("paired discordance outcomes must be binary")
        first_only += int(first_success == 1 and second_success == 0)
        second_only += int(first_success == 0 and second_success == 1)
    discordances = first_only + second_only
    if discordances == 0:
        p_value = 1.0
    else:
        numerator = sum(
            math.comb(discordances, successes)
            for successes in range(first_only, discordances + 1)
        )
        p_value = numerator / (2 ** discordances)
    return {
        "test": "exact one-sided paired label-swap discordance tail",
        "first_only_wins_b": first_only,
        "second_only_wins_c": second_only,
        "discordant_groups": discordances,
        "p_value": p_value,
        "formula": "sum(k=b..b+c, choose(b+c,k)) / 2^(b+c); p=1 if b+c=0",
    }


def _invalid_summary(
    *,
    manifest: str | Path,
    config: str | Path,
    r01_summary: str | Path,
    results_root: str | Path,
    error: str,
) -> Dict[str, Any]:
    return {
        "schema_version": "1.0",
        "artifact_type": "r02_r03_population_summary",
        "gate": "R03",
        "status": "invalid_frozen_inputs",
        "gate_passed": False,
        "r02_apparatus_passed": False,
        "r03_primary_criteria_met": False,
        "learned_probe_authorized": False,
        "scientific_evidence": False,
        "decision": "Frozen inputs are invalid; no R02/R03 result is interpretable.",
        "population": {
            "valid": False,
            "errors": [error],
            "manifest": str(manifest),
            "config": str(config),
            "r01_summary": str(r01_summary),
            "results_root": str(results_root),
        },
        "populations": {},
        "observed_rollouts": {},
        "failures": {},
        "paired_oracle_minus_random": None,
        "paired_analytic_minus_random": None,
        "paired_oracle_minus_analytic": None,
        "primary": {
            "oracle_spsr_at_least_0_50": False,
            "oracle_minus_random_at_least_0_20": False,
            "grouped_bootstrap_lcb_above_zero": False,
            "criteria_met": False,
        },
        "analytic_matched_gate": {"criteria_met": False, "scientific_gate_passed": False},
        "probe_authorization": {
            "decision": PROBE_AUTHORIZATION_DECISION,
            "decision_sha256": PROBE_AUTHORIZATION_DECISION_SHA256,
            "outcome_conditions_met_before_evidence_tier": False,
            "learned_probe_authorized": False,
        },
    }


def summarize_r02_population(
    manifest: str | Path,
    config: str | Path,
    r01_summary: str | Path,
    results_root: str | Path,
    *,
    run_id: str,
    checkpoint_id: str,
    checkpoint_sha256: str,
    repo_root: str | Path,
    validator: Optional[Validator] = None,
    r01_validator: Optional[Validator] = None,
    contract_override: Optional[SummaryContract] = None,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    allow_synthetic_implementation_evidence: bool = False,
) -> Dict[str, Any]:
    """Validate the complete R02 population and compute the frozen R03 test."""

    implementation_override = any(
        item is not None for item in (validator, r01_validator, contract_override)
    )
    if implementation_override and not allow_synthetic_implementation_evidence:
        return _invalid_summary(
            manifest=manifest,
            config=config,
            r01_summary=r01_summary,
            results_root=results_root,
            error="test-only validator or contract overrides require explicit synthetic implementation mode",
        )
    if contract_override is not None:
        contract = contract_override
    else:
        try:
            contract = load_summary_contract(
                manifest,
                config,
                r01_summary,
                run_id=run_id,
                checkpoint_id=checkpoint_id,
                checkpoint_sha256=checkpoint_sha256,
                repo_root=repo_root,
            )
        except (SummaryContractError, OSError, json.JSONDecodeError, ValueError) as error:
            return _invalid_summary(
                manifest=manifest,
                config=config,
                r01_summary=r01_summary,
                results_root=results_root,
                error=str(error),
            )
    if bootstrap_replicates < 10_000:
        return _invalid_summary(
            manifest=manifest,
            config=config,
            r01_summary=r01_summary,
            results_root=results_root,
            error="registered grouped bootstrap requires at least 10,000 replicates",
        )

    if validator is None:
        validator = _load_r02_validator()
    if r01_validator is None:
        r01_validator = _load_r01_validator()
    root = Path(results_root).resolve()
    expected_cases = {str(case["case_id"]): case for case in contract.manifest_cases}
    expected_ids = set(expected_cases)
    raw_r01_records, raw_r01_errors, raw_r01_hashes = _load_raw_r01_population(
        contract, expected_cases, r01_validator
    )
    final_paths = sorted(root.rglob(RESULT_FILENAME)) if root.exists() else []
    paths_by_folder: Dict[str, List[Path]] = defaultdict(list)
    for path in final_paths:
        paths_by_folder[path.parent.name].append(path)

    launch_failures = _launch_failure_audit(
        root,
        expected_case_ids=sorted(expected_ids),
        final_case_ids=sorted(paths_by_folder),
    )

    missing_case_ids = sorted(expected_ids - set(paths_by_folder))
    unexpected_case_ids = sorted(set(paths_by_folder) - expected_ids)
    duplicate_case_ids = sorted(
        case_id for case_id, paths in paths_by_folder.items() if len(paths) != 1
    )
    invalid_artifacts: Dict[str, List[str]] = {}
    valid_records: Dict[str, Mapping[str, Any]] = {}
    result_hashes: List[Dict[str, str]] = []
    synthetic_case_ids: List[str] = []

    for folder_case_id, paths in sorted(paths_by_folder.items()):
        for path_index, path in enumerate(paths):
            key = folder_case_id if len(paths) == 1 else f"{folder_case_id}#{path_index}"
            errors: List[str] = []
            try:
                value = load_json(path)
            except (OSError, json.JSONDecodeError) as error:
                invalid_artifacts[key] = [f"cannot read final artifact: {error}"]
                continue
            if not isinstance(value, Mapping):
                invalid_artifacts[key] = ["final artifact must be an object"]
                continue
            errors.extend(validator(value))
            errors.extend(raw_r01_errors.get(folder_case_id, []))
            raw_r01 = raw_r01_records.get(folder_case_id)
            if raw_r01 is not None:
                errors.extend(_raw_r01_binding_errors(value, raw_r01))
            errors.extend(
                _identity_errors(
                    value,
                    folder_case_id=folder_case_id,
                    manifest_case=expected_cases.get(folder_case_id),
                    contract=contract,
                    allow_synthetic_implementation_evidence=allow_synthetic_implementation_evidence,
                )
            )
            provenance = value.get("provenance")
            if (
                isinstance(provenance, Mapping)
                and provenance.get("evidence_tier") == SYNTHETIC_EVIDENCE_TIER
            ):
                synthetic_case_ids.append(folder_case_id)
            if errors:
                invalid_artifacts[key] = sorted(set(errors))
                continue
            valid_records[folder_case_id] = value
            result_hashes.append({"case_id": folder_case_id, "sha256": file_sha256(path)})

    valid_ids = set(valid_records)
    population_errors: List[str] = []
    if missing_case_ids:
        population_errors.append("missing final artifacts")
    if unexpected_case_ids:
        population_errors.append("unexpected final artifacts")
    if duplicate_case_ids:
        population_errors.append("duplicate final artifacts")
    if invalid_artifacts:
        population_errors.append("invalid final artifacts")
    if raw_r01_errors:
        population_errors.append("invalid or mismatched immutable raw R01 source artifacts")
    if valid_ids != expected_ids:
        population_errors.append("fully validated identities do not equal the frozen population")

    git_commits = sorted(
        {
            str(value["provenance"]["git_commit"])
            for value in valid_records.values()
            if isinstance(value.get("provenance"), Mapping)
        }
    )
    baseline_commits = sorted(
        {
            str(value["provenance"]["baseline_commit"])
            for value in valid_records.values()
            if isinstance(value.get("provenance"), Mapping)
        }
    )
    if valid_records and len(git_commits) != 1:
        population_errors.append("case artifacts do not share one code revision")
    if valid_records and len(baseline_commits) != 1:
        population_errors.append("case artifacts do not share one baseline revision")

    artifact_population_valid = not population_errors
    eligible_ids = list(contract.eligible_case_ids)
    no_witness_ids = list(contract.no_witness_case_ids)
    nominal_mismatch_ids: List[str] = []
    direct_mismatch_ids: List[str] = []
    if artifact_population_valid:
        for case_id in eligible_ids + no_witness_ids:
            outcome = valid_records[case_id].get("outcome")
            if not isinstance(outcome, Mapping) or outcome.get("nominal_collision_reproduced") is not True:
                nominal_mismatch_ids.append(case_id)
        for case_id in eligible_ids:
            outcome = valid_records[case_id].get("outcome")
            if (
                not isinstance(outcome, Mapping)
                or (
                    outcome.get("nominal_collision_reproduced") is True
                    and outcome.get("direct_witness_reconfirmed") is not True
                )
            ):
                direct_mismatch_ids.append(case_id)
    nominal_mismatch_ids.sort()
    direct_mismatch_ids.sort()
    population_match = bool(
        artifact_population_valid and not nominal_mismatch_ids and not direct_mismatch_ids
    )

    feasible_metrics: Dict[str, Any] = {}
    itt_metrics: Dict[str, Any] = {}
    observed_metrics: Dict[str, Any] = {}
    failure_metrics: Dict[str, Any] = {}
    paired: Optional[Dict[str, Any]] = None
    analytic_paired: Optional[Dict[str, Any]] = None
    oracle_analytic: Optional[Dict[str, Any]] = None
    criteria = {
        "oracle_spsr_at_least_0_50": False,
        "oracle_minus_random_at_least_0_20": False,
        "grouped_bootstrap_lcb_above_zero": False,
        "criteria_met": False,
    }
    analytic_criteria = {
        "analytic_spsr_at_least_0_50": False,
        "analytic_minus_random_at_least_0_20": False,
        "grouped_bootstrap_lcb_above_zero": False,
        "criteria_met": False,
    }
    if artifact_population_valid:
        feasible_metrics, itt_metrics, observed_metrics, failure_metrics = _arm_population_metrics(
            valid_records, eligible_ids, no_witness_ids
        )
        case_to_group = {
            str(case["case_id"]): str(case["group_id"])
            for case in contract.manifest_cases
        }
        oracle_random_pairs = [
            (
                case_to_group[case_id],
                _arm_success(valid_records[case_id], ARM_LABELS["oracle"]),
                _arm_success(valid_records[case_id], ARM_LABELS["random"]),
            )
            for case_id in eligible_ids
        ]
        oracle_spsr = float(feasible_metrics["oracle"]["spsr"])
        random_spsr = float(feasible_metrics["random"]["spsr"])
        difference = oracle_spsr - random_spsr
        bootstrap = grouped_paired_bootstrap_lcb(
            oracle_random_pairs,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed,
        )
        paired = {
            "scope": "R01-feasible-conditioned complete groups",
            "oracle_spsr": oracle_spsr,
            "random_spsr": random_spsr,
            "point_difference": difference,
            "case_count": len(oracle_random_pairs),
            "structural_no_witness_cases_in_denominator": 0,
            "bootstrap": bootstrap,
        }
        criteria = {
            "oracle_spsr_at_least_0_50": oracle_spsr >= ORACLE_MINIMUM_SPSR,
            "oracle_minus_random_at_least_0_20": difference >= ORACLE_RANDOM_MINIMUM_DIFFERENCE,
            "grouped_bootstrap_lcb_above_zero": bootstrap["lower_confidence_bound"] > 0.0,
            "criteria_met": False,
        }
        criteria["criteria_met"] = all(
            bool(value) for key, value in criteria.items() if key != "criteria_met"
        )

        analytic_spsr = float(feasible_metrics["analytic"]["spsr"])
        analytic_random_difference = analytic_spsr - random_spsr
        analytic_random_pairs = [
            (
                case_to_group[case_id],
                _arm_success(valid_records[case_id], ARM_LABELS["analytic"]),
                _arm_success(valid_records[case_id], ARM_LABELS["random"]),
            )
            for case_id in eligible_ids
        ]
        analytic_bootstrap = grouped_paired_bootstrap_lcb(
            analytic_random_pairs,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed,
        )
        analytic_criteria = {
            "analytic_spsr_at_least_0_50": analytic_spsr >= ORACLE_MINIMUM_SPSR,
            "analytic_minus_random_at_least_0_20": (
                analytic_random_difference >= ORACLE_RANDOM_MINIMUM_DIFFERENCE
            ),
            "grouped_bootstrap_lcb_above_zero": (
                analytic_bootstrap["lower_confidence_bound"] > 0.0
            ),
            "criteria_met": False,
        }
        analytic_criteria["criteria_met"] = all(
            bool(value)
            for key, value in analytic_criteria.items()
            if key != "criteria_met"
        )
        analytic_paired = {
            "scope": "R01-feasible-conditioned complete groups",
            "analytic_spsr": analytic_spsr,
            "random_spsr": random_spsr,
            "point_difference": analytic_random_difference,
            "case_count": len(analytic_random_pairs),
            "structural_no_witness_cases_in_denominator": 0,
            "bootstrap": analytic_bootstrap,
            "matched_gate_criteria": analytic_criteria,
        }

        oracle_analytic_pairs = [
            (
                case_to_group[case_id],
                _arm_success(valid_records[case_id], ARM_LABELS["oracle"]),
                _arm_success(valid_records[case_id], ARM_LABELS["analytic"]),
            )
            for case_id in eligible_ids
        ]
        oracle_analytic_bootstrap = grouped_paired_bootstrap_lcb(
            oracle_analytic_pairs,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed,
        )
        oracle_analytic_exact = exact_one_sided_paired_discordance(
            oracle_analytic_pairs
        )
        oracle_analytic = {
            "scope": "R01-feasible-conditioned complete groups",
            "oracle_spsr": oracle_spsr,
            "analytic_spsr": analytic_spsr,
            "point_difference": oracle_spsr - analytic_spsr,
            "case_count": len(oracle_analytic_pairs),
            "structural_no_witness_cases_in_denominator": 0,
            "bootstrap": oracle_analytic_bootstrap,
            "exact_one_sided_paired_test": oracle_analytic_exact,
        }

    synthetic_only = bool(synthetic_case_ids)
    scientific_evidence = bool(
        artifact_population_valid
        and not synthetic_only
        and not implementation_override
    )
    r03_primary_criteria_met = bool(criteria["criteria_met"])
    scientific_gate_passed = bool(
        scientific_evidence and population_match and r03_primary_criteria_met
    )
    analytic_matched_criteria_met = bool(analytic_criteria["criteria_met"])
    analytic_matched_scientific_gate_passed = bool(
        scientific_evidence and population_match and analytic_matched_criteria_met
    )
    oracle_analytic_lcb_positive = bool(
        oracle_analytic is not None
        and oracle_analytic["bootstrap"]["lower_confidence_bound"] > 0.0
    )
    oracle_analytic_exact_p_below_0_05 = bool(
        oracle_analytic is not None
        and oracle_analytic["exact_one_sided_paired_test"]["p_value"] < 0.05
    )
    authorization_outcome_conditions_met = bool(
        r03_primary_criteria_met
        and not analytic_matched_criteria_met
        and oracle_analytic_lcb_positive
        and oracle_analytic_exact_p_below_0_05
    )
    learned_probe_authorized = bool(
        scientific_gate_passed
        and not analytic_matched_scientific_gate_passed
        and oracle_analytic_lcb_positive
        and oracle_analytic_exact_p_below_0_05
    )
    if not artifact_population_valid:
        status = "invalid_population"
        decision = "R02 population is not validation-ready; repair the listed artifact errors."
    elif not population_match:
        status = "population_mismatch"
        decision = "Nominal collision or direct-witness reconfirmation failed; R03 is not interpretable."
    elif synthetic_only:
        status = "implementation_evidence_only"
        decision = "Synthetic artifacts validate implementation only and cannot pass R03."
    elif scientific_gate_passed:
        status = "passed"
        decision = "The preregistered distributed-oracle R03 criteria passed."
    else:
        status = "failed_threshold"
        decision = "The preregistered distributed-oracle R03 criteria did not all pass."

    ordered_hashes = sorted(result_hashes, key=lambda item: item["case_id"])
    return {
        "schema_version": "1.0",
        "artifact_type": "r02_r03_population_summary",
        "gate": "R03",
        "status": status,
        "gate_passed": scientific_gate_passed,
        "r02_apparatus_passed": artifact_population_valid,
        "r03_primary_criteria_met": r03_primary_criteria_met,
        "learned_probe_authorized": learned_probe_authorized,
        "scientific_evidence": scientific_evidence,
        "decision": decision,
        "analysis_scope": (
            "pregrasp R01-feasible-conditioned oracle-flow test; synthetic inputs, when explicitly "
            "enabled through the Python test API, remain implementation evidence only"
        ),
        "population": {
            "valid": artifact_population_valid,
            "paired_baseline_reconfirmed": population_match,
            "expected_cases": EXPECTED_CASES,
            "expected_feasible_conditioned_cases": EXPECTED_FEASIBLE_CASES,
            "expected_structural_no_witness_cases": EXPECTED_STRUCTURAL_NO_WITNESS_CASES,
            "final_artifacts_found": len(final_paths),
            "fully_validated_artifacts": len(valid_records),
            "missing_case_ids": missing_case_ids,
            "unexpected_case_ids": unexpected_case_ids,
            "duplicate_case_ids": duplicate_case_ids,
            "invalid_artifacts": invalid_artifacts,
            "errors": population_errors,
            "nominal_collision_not_reconfirmed_case_ids": nominal_mismatch_ids,
            "direct_witness_not_reconfirmed_case_ids": direct_mismatch_ids,
            "eligible_case_ids": eligible_ids,
            "structural_no_witness_case_ids": no_witness_ids,
            "synthetic_implementation_case_ids": sorted(set(synthetic_case_ids)),
            "terminal_status_counts": dict(
                sorted(
                    Counter(
                        str(value.get("status"))
                        for value in valid_records.values()
                    ).items()
                )
            ),
            "launch_failures": launch_failures,
            "raw_r01_source": {
                "results_root": contract.r01_results_root,
                "validated_artifacts": len(raw_r01_records),
                "invalid_artifacts": raw_r01_errors,
                "result_hashes": raw_r01_hashes,
            },
        },
        "identities": {
            "run_id": contract.run_id,
            "manifest": str(Path(manifest).resolve()),
            "manifest_sha256": contract.manifest_sha256,
            "frozen_manifest_sha256": contract.frozen_manifest_sha256,
            "config": contract.config_path,
            "config_file_sha256": contract.config_file_sha256,
            "config_hash": contract.config_hash,
            "r01_summary": contract.r01_summary_path,
            "r01_summary_sha256": contract.r01_summary_sha256,
            "r01_ordered_result_set_digest": contract.r01_ordered_result_set_digest,
            "sampler_parity_sha256": contract.sampler_parity_sha256,
            "checkpoint_id": contract.checkpoint_id,
            "checkpoint_sha256": contract.checkpoint_sha256,
            "git_commit": git_commits[0] if len(git_commits) == 1 else None,
            "baseline_commit": baseline_commits[0] if len(baseline_commits) == 1 else None,
            "probe_authorization_decision": contract.probe_authorization_decision_path,
            "probe_authorization_decision_sha256": contract.probe_authorization_decision_sha256,
            "test_only_contract_or_validator_override": implementation_override,
        },
        "populations": {
            "feasible_conditioned_17": {
                "definition": "17 R01 changed-action p_min witness groups",
                "population_size": EXPECTED_FEASIBLE_CASES,
                "arm_spsr": feasible_metrics,
            },
            "original_20_intent_to_treat": {
                "definition": (
                    "original frozen collision population; the frozen arm is observed on all 20, "
                    "while three R01 no-witness cases are structural zeros for witness-dependent arms"
                ),
                "population_size": EXPECTED_CASES,
                "structural_no_witness_cases": EXPECTED_STRUCTURAL_NO_WITNESS_CASES,
                "structural_zero_policy": "direct/random/analytic/oracle/bridge only; frozen is observed",
                "arm_spsr": itt_metrics,
            },
        },
        "observed_rollouts": {
            "definition": "actually executed eligible rollouts; structural no-witness cases never enter this denominator",
            "arms": observed_metrics,
        },
        "failures": failure_metrics,
        "status_counts": {
            label: dict(
                sorted(
                    Counter(
                        _arm_status(valid_records[case_id], raw_name)
                        for case_id in eligible_ids + no_witness_ids
                        if case_id in valid_records
                    ).items()
                )
            )
            for label, raw_name in ARM_LABELS.items()
        },
        "paired_oracle_minus_random": paired,
        "paired_analytic_minus_random": analytic_paired,
        "paired_oracle_minus_analytic": oracle_analytic,
        "primary": {
            "registered_thresholds": {
                "oracle_spsr_minimum": ORACLE_MINIMUM_SPSR,
                "oracle_minus_random_minimum": ORACLE_RANDOM_MINIMUM_DIFFERENCE,
                "grouped_bootstrap_lcb_strictly_above": 0.0,
            },
            **criteria,
            "scientific_gate_passed": scientific_gate_passed,
        },
        "analytic_matched_gate": {
            "registered_thresholds": {
                "analytic_spsr_minimum": ORACLE_MINIMUM_SPSR,
                "analytic_minus_random_minimum": ORACLE_RANDOM_MINIMUM_DIFFERENCE,
                "grouped_bootstrap_lcb_strictly_above": 0.0,
            },
            **analytic_criteria,
            "scientific_gate_passed": analytic_matched_scientific_gate_passed,
        },
        "probe_authorization": {
            "decision": PROBE_AUTHORIZATION_DECISION,
            "decision_sha256": PROBE_AUTHORIZATION_DECISION_SHA256,
            "conditions": {
                "r03_oracle_vs_random_scientific_gate_passed": scientific_gate_passed,
                "analytic_vs_random_matched_gate_did_not_pass": (
                    not analytic_matched_scientific_gate_passed
                ),
                "oracle_minus_analytic_grouped_lcb_above_zero": (
                    oracle_analytic_lcb_positive
                ),
                "oracle_vs_analytic_exact_one_sided_p_below_0_05": (
                    oracle_analytic_exact_p_below_0_05
                ),
            },
            "outcome_conditions_met_before_evidence_tier": (
                authorization_outcome_conditions_met
            ),
            "learned_probe_authorized": learned_probe_authorized,
            "interpretation": (
                "authorization permits R04 only; it is not evidence that ECG is effective or novel"
            ),
        },
        "ordered_result_set_digest": content_hash(ordered_hashes),
        "result_hashes": ordered_hashes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--r01-summary", required=True)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--checkpoint-id", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    summary = summarize_r02_population(
        args.manifest,
        args.config,
        args.r01_summary,
        args.results_root,
        run_id=args.run_id,
        checkpoint_id=args.checkpoint_id,
        checkpoint_sha256=args.checkpoint_sha256,
        repo_root=repo_root,
    )
    atomic_write_json(args.output, summary)
    printable = {
        key: value
        for key, value in summary.items()
        if key not in {"result_hashes", "status_counts"}
    }
    print(json.dumps(printable, sort_keys=True))
    return 0 if summary["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
