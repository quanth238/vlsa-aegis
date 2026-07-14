#!/usr/bin/env python3
"""Validate and summarize the 17-case R03A strong-analytic kill test.

This entry point is CPU-only: it reads immutable R02 source artifacts and final
R03A artifacts, performs no policy/simulator/CUDA work, and publishes a summary
atomically only after the complete fixed-denominator population validates.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import importlib.util
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from crfs_harness.artifacts import (  # noqa: E402
    atomic_write_json,
    content_hash,
    file_sha256,
    load_json,
)


def _load_validation_module():
    path = REPO_ROOT / "main/crfs_oracle/r03a_validation.py"
    spec = importlib.util.spec_from_file_location("crfs_r03a_validation", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the R03A validator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


VALIDATION = _load_validation_module()
EXPECTED_CASES = VALIDATION.EXPECTED_CASES
EXPECTED_ARMS = tuple(VALIDATION.EXPECTED_ARMS)
ANALYTIC_ARMS = tuple(VALIDATION.ANALYTIC_ARMS)
SOURCE_ARM_KEYS = tuple(VALIDATION.SOURCE_ARM_KEYS)
MAXIMUM_OBBS = 21
EEF_CENTER_OFFSET_LOCAL_M = [0.0, 0.0, -0.08]
RESULT_FILENAME = "r03a-analytic-kill-test.json"
SUMMARY_ARTIFACT_TYPE = "r03a_analytic_kill_test_population_summary"
REAL_EVIDENCE_TIER = "real_safelibero_r03a_strong_analytic_development_diagnostic"
SYNTHETIC_EVIDENCE_TIER = "synthetic_implementation_evidence_only"

Validator = Callable[[Mapping[str, Any]], List[str]]


@dataclass(frozen=True)
class SummaryContract:
    manifest_cases: Tuple[Mapping[str, Any], ...]
    manifest_sha256: str
    config_path: str
    config_file_sha256: str
    config_hash: str
    r03_summary_path: str
    r03_summary_sha256: str
    r03_ordered_result_set_digest: str
    r02_result_hashes: Mapping[str, str]
    r02_results_root: str
    run_id: str
    checkpoint_id: str
    checkpoint_sha256: str
    expected_source_slurm_array_job_id: Optional[str]
    expected_git_commit: str


class SummaryContractError(ValueError):
    """Raised when frozen inputs or final artifacts violate the R03A contract."""


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _resolve(path_value: Union[str, Path], repo_root: Path) -> Path:
    path = Path(path_value).expanduser()
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _read_jsonl(path: Path) -> List[Mapping[str, Any]]:
    records: List[Mapping[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise SummaryContractError(f"cannot read R03A manifest: {error}") from error
    for index, line in enumerate(lines, start=1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise SummaryContractError(f"invalid manifest JSON at line {index}: {error}") from error
        if not isinstance(value, Mapping):
            raise SummaryContractError(f"manifest line {index} must be an object")
        records.append(value)
    return records


def _validate_manifest(records: Sequence[Mapping[str, Any]]) -> Tuple[str, ...]:
    if len(records) != EXPECTED_CASES:
        raise SummaryContractError("R03A requires exactly 17 manifest cases")
    expected_keys = {
        "schema_version",
        "case_id",
        "group_id",
        "task_suite",
        "safety_level",
        "task_index",
        "episode_index",
        "environment_seed",
        "policy_seed",
        "random_control_seed",
    }
    identities: List[str] = []
    groups: List[str] = []
    for index, case in enumerate(records):
        if set(case) != expected_keys:
            raise SummaryContractError(f"manifest case {index} has incorrect fields")
        if case.get("schema_version") != "1.0":
            raise SummaryContractError("R03A manifest must use schema_version 1.0")
        if (
            case.get("task_suite") != "safelibero_spatial"
            or case.get("safety_level") != "II"
            or case.get("task_index") != 0
        ):
            raise SummaryContractError("R03A manifest differs from frozen Spatial Level-II task 0")
        case_id = case.get("case_id")
        group_id = case.get("group_id")
        if not isinstance(case_id, str) or not case_id:
            raise SummaryContractError("R03A manifest case_id must be nonempty")
        if not isinstance(group_id, str) or not group_id:
            raise SummaryContractError("R03A manifest group_id must be nonempty")
        identities.append(case_id)
        groups.append(group_id)
    if len(set(identities)) != EXPECTED_CASES or len(set(groups)) != EXPECTED_CASES:
        raise SummaryContractError("R03A manifest identities/groups must be unique")
    return tuple(identities)


def _validate_r03_summary(
    value: Mapping[str, Any],
    *,
    actual_sha256: str,
    manifest_case_ids: Sequence[str],
) -> Dict[str, str]:
    if actual_sha256 != VALIDATION.R03_SUMMARY_SHA256:
        raise SummaryContractError("R03 summary is not the accepted checked-in artifact")
    if (
        value.get("schema_version") != "1.0"
        or value.get("gate") != "R03"
        or value.get("status") != "passed"
        or value.get("gate_passed") is not True
        or value.get("scientific_evidence") is not True
    ):
        raise SummaryContractError("R03 summary must be the passed scientific artifact")
    if value.get("ordered_result_set_digest") != VALIDATION.R03_ORDERED_RESULT_SET_DIGEST:
        raise SummaryContractError("R03 ordered result-set digest differs")
    population = value.get("population")
    eligible = population.get("eligible_case_ids") if isinstance(population, Mapping) else None
    if not isinstance(eligible, list) or set(eligible) != set(manifest_case_ids):
        raise SummaryContractError("R03 eligible identities differ from the R03A manifest")
    populations = value.get("populations")
    feasible = populations.get("feasible_conditioned_17") if isinstance(populations, Mapping) else None
    arm_spsr = feasible.get("arm_spsr") if isinstance(feasible, Mapping) else None
    expected_source_counts = {
        "frozen": 0,
        "direct": 17,
        "random": 0,
        "analytic": 0,
        "oracle": 9,
        "bridge": 0,
    }
    if not isinstance(arm_spsr, Mapping):
        raise SummaryContractError("R03 summary lacks the feasible arm population")
    for name, count in expected_source_counts.items():
        item = arm_spsr.get(name)
        if not isinstance(item, Mapping) or item.get("denominator") != 17 or item.get("successes") != count:
            raise SummaryContractError(f"R03 source count differs for {name}")
    hashes = value.get("result_hashes")
    if not isinstance(hashes, list) or len(hashes) != 20:
        raise SummaryContractError("R03 summary must bind all 20 source artifacts")
    result: Dict[str, str] = {}
    for item in hashes:
        if not isinstance(item, Mapping):
            raise SummaryContractError("R03 result_hashes entries must be objects")
        case_id = item.get("case_id")
        digest = item.get("sha256")
        if not isinstance(case_id, str) or not _is_sha256(digest) or case_id in result:
            raise SummaryContractError("R03 result_hashes contains invalid/duplicate identities")
        result[case_id] = digest
    if not set(manifest_case_ids).issubset(result):
        raise SummaryContractError("R03 source hashes omit an eligible case")
    return {case_id: result[case_id] for case_id in manifest_case_ids}


def load_summary_contract(
    manifest_path: Union[str, Path],
    config_path: Union[str, Path],
    r03_summary_path: Union[str, Path],
    *,
    r02_results_root: Union[str, Path],
    run_id: str,
    checkpoint_id: str,
    checkpoint_sha256: str,
    expected_git_commit: str,
    expected_source_slurm_array_job_id: Optional[str] = None,
    repo_root: Path = REPO_ROOT,
) -> SummaryContract:
    manifest = _resolve(manifest_path, repo_root)
    config_file = _resolve(config_path, repo_root)
    r03_file = _resolve(r03_summary_path, repo_root)
    if not isinstance(run_id, str) or not run_id:
        raise SummaryContractError("run_id must be an explicit immutable identifier")
    if not isinstance(checkpoint_id, str) or not checkpoint_id:
        raise SummaryContractError("checkpoint_id must be nonempty")
    if checkpoint_sha256 != VALIDATION.CHECKPOINT_SHA256:
        raise SummaryContractError("checkpoint hash differs from accepted R03 source")
    if not isinstance(expected_git_commit, str) or re.fullmatch(
        r"[0-9a-f]{40}([0-9a-f]{24})?", expected_git_commit
    ) is None:
        raise SummaryContractError("expected_git_commit must be an immutable Git object id")

    cases = _read_jsonl(manifest)
    case_ids = _validate_manifest(cases)
    manifest_sha = file_sha256(manifest)
    if manifest_sha != VALIDATION.ELIGIBLE_MANIFEST_SHA256:
        raise SummaryContractError("R03A manifest bytes differ from the frozen eligible subset")
    config = load_json(config_file)
    if not isinstance(config, Mapping):
        raise SummaryContractError("R03A config must be an object")
    if config.get("ready_to_run") is not True or config.get("blocked_on") != []:
        raise SummaryContractError("R03A config must be ready_to_run with no blocked dependencies")
    if config.get("manifest") != "manifests/r03a_analytic_kill_test_eligible.jsonl":
        raise SummaryContractError("R03A config is not bound to the eligible manifest")
    settings = config.get("r03a")
    if not isinstance(settings, Mapping):
        raise SummaryContractError("R03A config has no r03a object")
    fixed_config_values = {
        "eligible_case_count": EXPECTED_CASES,
        "eligible_manifest_sha256": manifest_sha,
        "artifact_schema_sha256": "2f0f9db28ae9d62582e88bc70afe40ef4e37245a60190863e2f3419a135d70c3",
        "source_r03_summary_sha256": VALIDATION.R03_SUMMARY_SHA256,
        "source_r03_ordered_result_set_digest": VALIDATION.R03_ORDERED_RESULT_SET_DIGEST,
        "source_r02_config_sha256": VALIDATION.R02_CONFIG_SHA256,
        "decision_artifact": VALIDATION.DECISION_ARTIFACT,
        "decision_sha256": VALIDATION.DECISION_SHA256,
        "budget_source": VALIDATION.BUDGET_DEFINITION,
        "policy_intervention_mode": "analytic_trajectory_field",
        "clipping_policy": "fail_without_clipping",
        "saturation_criterion": "any_executed_first_five_xyz_exactly_equals_registered_inclusive_bound",
        "saturation_comparison": "exact_float_equality_no_epsilon_no_clipping",
        "terminal_failure_criterion": "any_simulator_task_success_during_prefix_is_true",
        "evaluated_failure_precedence": [
            "bounds_failure",
            "saturation_failure",
            "budget_failure",
            "terminal_failure",
            "zero_gradient_failure",
            "ordinary_gate",
        ],
        "kill_threshold_privileged_sps_count": 9,
        "probe_training_authorized": False,
        "confirmatory_r04_unblocked_by_r03a_alone": False,
    }
    for key, expected in fixed_config_values.items():
        if settings.get(key) != expected:
            raise SummaryContractError(f"R03A config field {key} differs from the frozen contract")
    if settings.get("required_arms") != list(ANALYTIC_ARMS):
        raise SummaryContractError("R03A config analytic arms/order differ")
    if settings.get("intervention_steps") != VALIDATION.INTERVENTION_STEPS:
        raise SummaryContractError("R03A intervention steps differ")
    if settings.get("expected_active_steps") != VALIDATION.EXPECTED_ACTIVE_STEPS:
        raise SummaryContractError("R03A active-step counts differ")
    source_root = _resolve(r02_results_root, repo_root)
    configured_root = _resolve(str(settings.get("source_r02_results_root", "")), repo_root)
    if source_root != configured_root:
        raise SummaryContractError("R02 source root differs from the frozen R03A config")

    r03 = load_json(r03_file)
    if not isinstance(r03, Mapping):
        raise SummaryContractError("R03 summary must be an object")
    r03_sha = file_sha256(r03_file)
    r02_hashes = _validate_r03_summary(
        r03,
        actual_sha256=r03_sha,
        manifest_case_ids=case_ids,
    )
    return SummaryContract(
        manifest_cases=tuple(cases),
        manifest_sha256=manifest_sha,
        config_path=str(config_file),
        config_file_sha256=file_sha256(config_file),
        config_hash=VALIDATION.r03a_config_hash(config),
        r03_summary_path=str(r03_file),
        r03_summary_sha256=r03_sha,
        r03_ordered_result_set_digest=VALIDATION.R03_ORDERED_RESULT_SET_DIGEST,
        r02_result_hashes=r02_hashes,
        r02_results_root=str(source_root),
        run_id=run_id,
        checkpoint_id=checkpoint_id,
        checkpoint_sha256=checkpoint_sha256,
        expected_source_slurm_array_job_id=expected_source_slurm_array_job_id,
        expected_git_commit=expected_git_commit,
    )


def _load_r02_validator() -> Validator:
    from crfs_oracle.r02_runner import validate_r02_result

    return validate_r02_result


def _source_arm_pass(source: Mapping[str, Any]) -> Dict[str, bool]:
    arms = source.get("arms")
    if not isinstance(arms, Mapping) or set(arms) != set(SOURCE_ARM_KEYS):
        raise SummaryContractError("source R02 arm set differs")
    result: Dict[str, bool] = {}
    for name in SOURCE_ARM_KEYS:
        gate = arms[name].get("gate") if isinstance(arms[name], Mapping) else None
        if not isinstance(gate, Mapping) or gate.get("passed") not in {True, False}:
            raise SummaryContractError(f"source R02 arm {name} has no binary gate")
        result[name] = bool(gate["passed"])
    return result


def _flat_finite_vector(value: Any, *, length: int, name: str) -> List[float]:
    flattened: List[float] = []

    def visit(item: Any) -> None:
        if isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
            return
        if isinstance(item, bool):
            raise SummaryContractError(f"{name} must be finite numeric")
        try:
            number = float(item)
        except (TypeError, ValueError) as error:
            raise SummaryContractError(f"{name} must be finite numeric") from error
        if not math.isfinite(number):
            raise SummaryContractError(f"{name} must be finite numeric")
        flattened.append(number)

    visit(value)
    if len(flattened) != length:
        raise SummaryContractError(f"{name} must contain exactly {length} values")
    return flattened


def _canonical_rollout_geometry(rollout: Mapping[str, Any]) -> Mapping[str, Any]:
    """Dependency-free mirror of R04's sorted, padded world-OBB identity."""

    boxes = rollout.get("branch_obstacle_boxes")
    if not isinstance(boxes, list) or not boxes or not all(
        isinstance(item, Mapping) for item in boxes
    ):
        raise SummaryContractError("source R02 frozen geometry has invalid obstacle boxes")
    if len(boxes) > MAXIMUM_OBBS:
        raise SummaryContractError("source R02 frozen geometry exceeds the fixed OBB width")
    branch_center = _flat_finite_vector(
        rollout.get("start_eef_center_m"), length=3, name="branch EEF center"
    )
    ordered = sorted(boxes, key=lambda item: str(item.get("name", "")))
    names = [str(item.get("name", "")) for item in ordered]
    if any(not name for name in names) or len(set(names)) != len(names):
        raise SummaryContractError("source R02 OBB names must be nonempty and unique")

    padded_names = [""] * MAXIMUM_OBBS
    validity = [False] * MAXIMUM_OBBS
    centers = [[0.0, 0.0, 0.0] for _ in range(MAXIMUM_OBBS)]
    rotations = [[0.0] * 9 for _ in range(MAXIMUM_OBBS)]
    half_sizes = [[0.0, 0.0, 0.0] for _ in range(MAXIMUM_OBBS)]
    for index, box in enumerate(ordered):
        center = _flat_finite_vector(
            box.get("center_m"), length=3, name=f"OBB {names[index]} center"
        )
        rotation = _flat_finite_vector(
            box.get("rotation_world"), length=9, name=f"OBB {names[index]} rotation"
        )
        half_size = _flat_finite_vector(
            box.get("half_size_m"), length=3, name=f"OBB {names[index]} half-size"
        )
        if any(item <= 0.0 for item in half_size):
            raise SummaryContractError("source R02 OBB half-sizes must be positive")
        padded_names[index] = names[index]
        validity[index] = True
        centers[index] = center
        rotations[index] = rotation
        half_sizes[index] = half_size
    payload = {
        "frame": "world",
        "representation": "geom_name_sorted_padded_obb",
        "padding_value": 0.0,
        "maximum_obbs": MAXIMUM_OBBS,
        "num_valid_obbs": len(ordered),
        "geom_names": padded_names,
        "validity_mask": validity,
        "centers_m": centers,
        "rotations_world": rotations,
        "half_sizes_m": half_sizes,
        "branch_eef_center_m": branch_center,
        "eef_center_offset_local_m": list(EEF_CENTER_OFFSET_LOCAL_M),
    }
    return {**payload, "sha256": content_hash(payload)}


def _source_pairing_hashes(source: Mapping[str, Any]) -> Dict[str, str]:
    pairing = source.get("pairing")
    provenance = source.get("provenance")
    arms = source.get("arms")
    if not isinstance(pairing, Mapping) or not isinstance(provenance, Mapping) or not isinstance(arms, Mapping):
        raise SummaryContractError("source R02 pairing/provenance/arms are missing")
    frozen = arms.get("frozen")
    repeats = frozen.get("repeats") if isinstance(frozen, Mapping) else None
    if not isinstance(repeats, list) or not repeats or not isinstance(repeats[0], Mapping):
        raise SummaryContractError("source R02 frozen geometry witness is missing")
    first = repeats[0]
    # Bind the same geom-name-sorted, fixed-width OBB representation as the
    # allocation runner.  Raw simulator/XML box enumeration is not an identity.
    geometry = _canonical_rollout_geometry(first)
    records = {
        "policy_observation": pairing.get("policy_observation"),
        "noise": provenance.get("noise"),
        "source_frozen_actions": pairing.get("eager_actions"),
        "source_frozen_trace": pairing.get("eager_trace"),
    }
    hashes: Dict[str, str] = {
        "branch_snapshot": content_hash(pairing.get("branch_snapshot")),
        "branch_geometry": content_hash(geometry),
    }
    for name, record in records.items():
        digest = record.get("sha256") if isinstance(record, Mapping) else None
        if not _is_sha256(digest):
            raise SummaryContractError(f"source R02 {name} hash is missing")
        hashes[name] = digest
    return hashes


def _validate_source_case(
    source: Mapping[str, Any],
    *,
    case: Mapping[str, Any],
    validator: Validator,
) -> None:
    validation_errors = validator(source)
    if validation_errors:
        raise SummaryContractError(
            f"source R02 case {case['case_id']} is invalid: {'; '.join(validation_errors)}"
        )
    if source.get("case_id") != case.get("case_id") or source.get("status") != "completed":
        raise SummaryContractError("R03A source must be the completed matching R02 case")
    provenance = source.get("provenance")
    if not isinstance(provenance, Mapping) or provenance.get("case_record") != dict(case):
        raise SummaryContractError("source R02 case record differs from the eligible manifest")
    outcome = source.get("outcome")
    if not isinstance(outcome, Mapping):
        raise SummaryContractError("source R02 outcome is missing")
    if (
        outcome.get("r01_feasible_conditioned") is not True
        or outcome.get("nominal_collision_reproduced") is not True
        or outcome.get("direct_witness_reconfirmed") is not True
    ):
        raise SummaryContractError("R03A source is not a witness-confirmed R02 collision case")


def _cross_validate_case(
    result: Mapping[str, Any],
    source: Mapping[str, Any],
    *,
    case: Mapping[str, Any],
    source_path: Path,
    source_sha256: str,
    contract: SummaryContract,
) -> None:
    if result.get("case_id") != case.get("case_id") or result.get("run_id") != contract.run_id:
        raise SummaryContractError("R03A result identity/run differs")
    if result.get("config_hash") != contract.config_hash:
        raise SummaryContractError("R03A scientific config hash differs")
    provenance = result.get("provenance")
    if not isinstance(provenance, Mapping):
        raise SummaryContractError("R03A result provenance is missing")
    if provenance.get("case_record") != dict(case):
        raise SummaryContractError("R03A result case record differs from manifest")
    if provenance.get("config_file_sha256") != contract.config_file_sha256:
        raise SummaryContractError("R03A config file hash differs")
    if provenance.get("checkpoint_id") != contract.checkpoint_id:
        raise SummaryContractError("R03A checkpoint path differs")
    if provenance.get("checkpoint_sha256") != contract.checkpoint_sha256:
        raise SummaryContractError("R03A checkpoint hash differs")
    if (
        contract.expected_source_slurm_array_job_id is not None
        and provenance.get("slurm_array_job_id") != contract.expected_source_slurm_array_job_id
    ):
        raise SummaryContractError("R03A source Slurm array job differs")
    source_evidence = result.get("source_evidence")
    if not isinstance(source_evidence, Mapping):
        raise SummaryContractError("R03A source_evidence is missing")
    if _resolve(source_evidence.get("r03_summary_path", ""), REPO_ROOT) != Path(
        contract.r03_summary_path
    ).resolve():
        raise SummaryContractError("R03A source R03 summary path differs")
    if _resolve(source_evidence.get("r02_case_path", ""), REPO_ROOT) != source_path.resolve():
        raise SummaryContractError("R03A source R02 case path differs")
    if source_evidence.get("r02_case_sha256") != source_sha256:
        raise SummaryContractError("R03A source R02 case hash differs")

    expected_source_pass = _source_arm_pass(source)
    outcome = result.get("outcome")
    if not isinstance(outcome, Mapping) or outcome.get("source_arm_gate_pass") != expected_source_pass:
        raise SummaryContractError("R03A retained source arm outcomes differ from R02")
    budget = result.get("budget")
    directions = source.get("directions")
    arrays = directions.get("arrays") if isinstance(directions, Mapping) else None
    l2_norms = directions.get("l2_norms") if isinstance(directions, Mapping) else None
    source_delta = arrays.get("delta_star_model") if isinstance(arrays, Mapping) else None
    source_l2 = l2_norms.get("delta_star_model") if isinstance(l2_norms, Mapping) else None
    if not isinstance(budget, Mapping) or budget.get("source_delta_star_model") != source_delta:
        raise SummaryContractError("R03A budget array differs from immutable R02 delta_star_model")
    if budget.get("source_reported_full_model_l2") != source_l2:
        raise SummaryContractError("R03A source reported budget norm differs from R02")

    expected_pairing = _source_pairing_hashes(source)
    pairing = result.get("pairing")
    if not isinstance(pairing, Mapping):
        raise SummaryContractError("R03A pairing is missing")
    for name, expected_hash in expected_pairing.items():
        binding = pairing.get(name)
        if not isinstance(binding, Mapping) or binding.get("source_sha256") != expected_hash:
            raise SummaryContractError(f"R03A pairing source hash differs for {name}")


def _inverted_cdf(values: Sequence[float], probability: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(float(item) for item in values)
    index = max(0, min(len(ordered) - 1, math.ceil(probability * len(ordered)) - 1))
    return ordered[index]


def _numeric_summary(values: Sequence[float]) -> Dict[str, Any]:
    clean = [float(item) for item in values if math.isfinite(float(item))]
    return {
        "count": len(clean),
        "minimum": min(clean) if clean else None,
        "median": _inverted_cdf(clean, 0.50),
        "p95": _inverted_cdf(clean, 0.95),
        "maximum": max(clean) if clean else None,
    }


def _arm_population(records: Sequence[Mapping[str, Any]], arm_name: str) -> Dict[str, Any]:
    status_counts: Counter[str] = Counter()
    success_ids: List[str] = []
    failure_ids: List[str] = []
    clearance: List[float] = []
    progress: List[float] = []
    policy_seconds: List[float] = []
    gradient_seconds: List[float] = []
    integrated: List[float] = []
    corrections: Dict[str, List[float]] = {
        "model_l2": [],
        "model_rms": [],
        "physical_l2": [],
        "physical_rms": [],
    }
    contacts = 0
    for result in records:
        case_id = str(result["case_id"])
        arm = result["arms"][arm_name]
        status = str(arm["status"])
        status_counts[status] += 1
        if arm.get("gate", {}).get("passed") is True:
            success_ids.append(case_id)
        else:
            failure_ids.append(case_id)
        gate = arm.get("gate")
        if isinstance(gate, Mapping):
            if isinstance(gate.get("minimum_clearance_m"), (int, float)):
                clearance.append(float(gate["minimum_clearance_m"]))
            if isinstance(gate.get("minimum_progress_m"), (int, float)):
                progress.append(float(gate["minimum_progress_m"]))
            if gate.get("any_contact") is True:
                contacts += 1
        timing = arm.get("timing")
        if isinstance(timing, Mapping):
            policy = timing.get("policy_seconds")
            gradient = timing.get("analytic_gradient_seconds")
            if isinstance(policy, Mapping) and isinstance(policy.get("values"), list):
                policy_seconds.extend(float(item) for item in policy["values"])
            if isinstance(gradient, Mapping) and isinstance(gradient.get("values"), list):
                gradient_seconds.extend(float(item) for item in gradient["values"])
        diagnostics = arm.get("diagnostics")
        if isinstance(diagnostics, Mapping):
            if isinstance(diagnostics.get("integrated_field_model_l2"), (int, float)):
                integrated.append(float(diagnostics["integrated_field_model_l2"]))
            correction = diagnostics.get("realized_final_correction")
            if isinstance(correction, Mapping):
                for key in corrections:
                    if isinstance(correction.get(key), (int, float)):
                        corrections[key].append(float(correction[key]))
    successes = len(success_ids)
    return {
        "denominator": EXPECTED_CASES,
        "successes": successes,
        "spsr": successes / EXPECTED_CASES,
        "success_case_ids": success_ids,
        "failure_case_ids": failure_ids,
        "status_counts": dict(sorted(status_counts.items())),
        "minimum_clearance_m": _numeric_summary(clearance),
        "minimum_progress_m": _numeric_summary(progress),
        "any_contact_case_count": contacts,
        "policy_seconds": _numeric_summary(policy_seconds),
        "analytic_gradient_seconds": _numeric_summary(gradient_seconds),
        "integrated_field_model_l2": _numeric_summary(integrated),
        "realized_final_correction": {
            key: _numeric_summary(values) for key, values in corrections.items()
        },
    }


def summarize_r03a_population(
    manifest_path: Union[str, Path],
    config_path: Union[str, Path],
    r02_results_root: Union[str, Path],
    r03_summary_path: Union[str, Path],
    results_root: Union[str, Path],
    *,
    run_id: str,
    checkpoint_id: str,
    checkpoint_sha256: str,
    expected_git_commit: str,
    expected_source_slurm_array_job_id: Optional[str] = None,
    repo_root: Path = REPO_ROOT,
    validator: Validator = VALIDATION.validate_r03a_result,
    source_validator: Optional[Validator] = None,
    contract_override: Optional[SummaryContract] = None,
    allow_synthetic_implementation_evidence: bool = False,
    require_schema_dependency: bool = True,
) -> Dict[str, Any]:
    contract = contract_override or load_summary_contract(
        manifest_path,
        config_path,
        r03_summary_path,
        r02_results_root=r02_results_root,
        run_id=run_id,
        checkpoint_id=checkpoint_id,
        checkpoint_sha256=checkpoint_sha256,
        expected_git_commit=expected_git_commit,
        expected_source_slurm_array_job_id=expected_source_slurm_array_job_id,
        repo_root=repo_root,
    )
    source_validator = source_validator or _load_r02_validator()
    output_root = _resolve(results_root, repo_root)
    source_root = Path(contract.r02_results_root)
    expected_ids = [str(case["case_id"]) for case in contract.manifest_cases]
    found = sorted(output_root.rglob(RESULT_FILENAME)) if output_root.exists() else []
    if len(found) != EXPECTED_CASES:
        raise SummaryContractError(
            f"R03A requires exactly {EXPECTED_CASES} final artifacts, found {len(found)}"
        )
    if {path.parent.name for path in found} != set(expected_ids):
        raise SummaryContractError("R03A result directories differ from the frozen population")

    records: List[Mapping[str, Any]] = []
    result_hashes: List[Dict[str, str]] = []
    source_hashes: List[Dict[str, str]] = []
    evidence_tiers: set[str] = set()
    git_commits: set[str] = set()
    git_dirty_values: set[bool] = set()
    source_oracle_successes = 0
    case_by_id = {str(case["case_id"]): case for case in contract.manifest_cases}
    for case_id in expected_ids:
        case = case_by_id[case_id]
        result_path = output_root / case_id / RESULT_FILENAME
        source_path = source_root / case_id / "r02-paired.json"
        if not result_path.is_file() or not source_path.is_file():
            raise SummaryContractError(f"missing R03A/source R02 artifact for {case_id}")
        result = load_json(result_path)
        source = load_json(source_path)
        if not isinstance(result, Mapping) or not isinstance(source, Mapping):
            raise SummaryContractError(f"R03A/source artifact for {case_id} must be an object")
        errors = validator(result)
        errors.extend(
            VALIDATION.validate_r03a_schema(
                result, require_jsonschema=require_schema_dependency
            )
        )
        if errors:
            raise SummaryContractError(f"invalid R03A case {case_id}: {'; '.join(errors)}")
        source_sha = file_sha256(source_path)
        if source_sha != contract.r02_result_hashes[case_id]:
            raise SummaryContractError(f"source R02 hash differs for {case_id}")
        _validate_source_case(source, case=case, validator=source_validator)
        _cross_validate_case(
            result,
            source,
            case=case,
            source_path=source_path,
            source_sha256=source_sha,
            contract=contract,
        )
        provenance = result["provenance"]
        evidence_tiers.add(str(provenance["evidence_tier"]))
        git_commits.add(str(provenance["git_commit"]))
        git_dirty_values.add(bool(provenance["git_dirty"]))
        source_oracle_successes += int(result["outcome"]["source_arm_gate_pass"]["oracle_residual"])
        records.append(result)
        result_hashes.append({"case_id": case_id, "sha256": file_sha256(result_path)})
        source_hashes.append({"case_id": case_id, "sha256": source_sha})

    if len(evidence_tiers) != 1:
        raise SummaryContractError("R03A evidence tiers differ across cases")
    evidence_tier = next(iter(evidence_tiers))
    if evidence_tier == SYNTHETIC_EVIDENCE_TIER and not allow_synthetic_implementation_evidence:
        raise SummaryContractError("synthetic R03A fixtures cannot produce a scientific summary")
    if evidence_tier not in {REAL_EVIDENCE_TIER, SYNTHETIC_EVIDENCE_TIER}:
        raise SummaryContractError("R03A evidence tier is unregistered")
    if len(git_commits) != 1 or len(git_dirty_values) != 1:
        raise SummaryContractError("R03A code/dirty provenance differs across cases")
    if git_commits != {contract.expected_git_commit}:
        raise SummaryContractError(
            "R03A case provenance commit differs from the reviewed summary checkout"
        )
    if evidence_tier == REAL_EVIDENCE_TIER and git_dirty_values != {False}:
        raise SummaryContractError("real R03A evidence requires a clean reviewed commit")
    if source_oracle_successes != 9:
        raise SummaryContractError("retained source oracle outcomes no longer total 9/17")

    arm_populations = {name: _arm_population(records, name) for name in EXPECTED_ARMS}
    nominal_mismatch_ids = [
        str(result["case_id"])
        for result in records
        if result["status"] == "nominal_collision_not_reconfirmed"
    ]
    policy_failure_case_ids_by_arm = {
        name: [
            str(result["case_id"])
            for result in records
            if result["arms"][name]["status"] == "policy_failure"
        ]
        for name in ANALYTIC_ARMS
    }
    policy_failure_ids = sorted(
        {
            case_id
            for case_ids in policy_failure_case_ids_by_arm.values()
            for case_id in case_ids
        }
    )
    population_valid = not nominal_mismatch_ids and not policy_failure_ids
    maximum_analytic_successes = max(
        arm_populations[name]["successes"] for name in ANALYTIC_ARMS
    )
    if not population_valid:
        decision = (
            "not_evaluable_population_mismatch"
            if nominal_mismatch_ids
            else "not_evaluable_policy_failure"
        )
        necessity_rejected: Optional[bool] = None
    elif maximum_analytic_successes >= VALIDATION.KILL_THRESHOLD_SPS_COUNT:
        decision = "pure_learned_clearance_probe_necessity_rejected"
        necessity_rejected = True
    else:
        decision = "strong_analytic_below_privileged_reference_mandatory_future_baseline"
        necessity_rejected = False
    scientific_evidence = evidence_tier == REAL_EVIDENCE_TIER
    return {
        "schema_version": "1.0",
        "artifact_type": SUMMARY_ARTIFACT_TYPE,
        "gate": "R03A",
        "status": "completed" if population_valid else "failed_population_integrity",
        "scientific_evidence": scientific_evidence,
        "analysis_scope": "17 witness-confirmed R03 development groups; learned-necessity diagnostic only",
        "identities": {
            "run_id": contract.run_id,
            "manifest_sha256": contract.manifest_sha256,
            "config_file_sha256": contract.config_file_sha256,
            "config_hash": contract.config_hash,
            "r03_summary_sha256": contract.r03_summary_sha256,
            "r03_ordered_result_set_digest": contract.r03_ordered_result_set_digest,
            "checkpoint_id": contract.checkpoint_id,
            "checkpoint_sha256": contract.checkpoint_sha256,
            "git_commit": next(iter(git_commits)),
            "git_dirty": next(iter(git_dirty_values)),
            "source_slurm_array_job_id": contract.expected_source_slurm_array_job_id,
        },
        "population": {
            "expected_cases": EXPECTED_CASES,
            "validated_cases": len(records),
            "case_ids": expected_ids,
            "nominal_collision_not_reconfirmed_case_ids": nominal_mismatch_ids,
            "policy_failure_case_ids": policy_failure_ids,
            "policy_failure_case_ids_by_arm": policy_failure_case_ids_by_arm,
            "fixed_denominator_preserved": True,
            "population_integrity_passed": population_valid,
        },
        "source_r03_reference": {
            "privileged_sps_count": 9,
            "population_size": 17,
            "privileged_spsr": 9 / 17,
            "source_oracle_outcomes_revalidated": source_oracle_successes,
        },
        "arms": arm_populations,
        "kill_test": {
            "threshold_sps_count": 9,
            "maximum_analytic_sps_count": maximum_analytic_successes,
            "either_analytic_reaches_privileged_reference": (
                maximum_analytic_successes >= 9 if population_valid else None
            ),
            "pure_learned_clearance_probe_necessity_rejected": necessity_rejected,
            "decision": decision,
            "both_arms_reported_without_selection": True,
        },
        "probe_training_authorized": False,
        "confirmatory_r04_unblocked": False,
        "interpretation": (
            "R03A is a development-population learned-necessity kill test; it cannot support "
            "probe efficacy, transport, novelty, or deployment claims."
        ),
        "ordered_result_set_digest": content_hash(result_hashes),
        "result_hashes": result_hashes,
        "source_r02_result_hashes": source_hashes,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--r02-raw-root", required=True)
    parser.add_argument("--r03-summary", required=True)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--checkpoint-id", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--expected-git-commit", required=True)
    parser.add_argument("--source-slurm-array-job-id", required=True)
    args = parser.parse_args(argv)
    summary = summarize_r03a_population(
        args.manifest,
        args.config,
        args.r02_raw_root,
        args.r03_summary,
        args.results_root,
        run_id=args.run_id,
        checkpoint_id=args.checkpoint_id,
        checkpoint_sha256=args.checkpoint_sha256,
        expected_git_commit=args.expected_git_commit,
        expected_source_slurm_array_job_id=args.source_slurm_array_job_id,
    )
    atomic_write_json(Path(args.output), summary)
    print(json.dumps(summary["kill_test"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
