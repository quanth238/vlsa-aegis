#!/usr/bin/env python3
"""Fail-closed population summary for the frozen R01 endpoint-free gate."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any, Callable, Mapping

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

Validator = Callable[[Mapping[str, Any]], list[str]]


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
