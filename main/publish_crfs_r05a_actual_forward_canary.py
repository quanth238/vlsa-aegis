#!/usr/bin/env python3
"""Independently validate and publish one terminal AF-00A artifact."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any

import numpy as np

from crfs_harness.artifacts import file_sha256, validate_jsonl_unique
from crfs_harness.manifest import validate_case
from crfs_oracle.r05a_actual_forward_validation import (
    POLICY_REQUESTS,
    validate_actual_forward_npz,
)
from crfs_oracle.r02_runner import _validate_array_record, validate_r02_result
from crfs_oracle.r05a_canary import REGISTERED_XYZ_SCALE
from crfs_oracle.r05a_full_lifetime_telemetry import parse_full_lifetime_telemetry


CASE_ID = "crfs-1069f29a8d76463a"
EXPERIMENT_ROOT = Path("/mnt/data/quanth/experiments/crfs-oracle")
CHECKPOINT_MODEL = Path(
    "/mnt/data/quanth/cache/openpi/openpi-assets/checkpoints/"
    "pi05_libero_pytorch/model.safetensors"
)
CHECKPOINT_CONFIG = CHECKPOINT_MODEL.parent / "config.json"
NORMALIZATION_ASSET = (
    CHECKPOINT_MODEL.parent
    / "assets"
    / "physical-intelligence"
    / "libero"
    / "norm_stats.json"
)
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
JOB_ID_PATTERN = re.compile(r"[1-9][0-9]*")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--legacy-config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--source-contract", required=True)
    parser.add_argument("--expected-source-contract-sha256", required=True)
    parser.add_argument("--held-gpu-submission", required=True)
    parser.add_argument("--expected-held-gpu-submission-sha256", required=True)
    parser.add_argument("--submission", required=True)
    parser.add_argument("--expected-submission-sha256", required=True)
    parser.add_argument("--release-fingerprint", required=True)
    parser.add_argument("--expected-release-fingerprint-sha256", required=True)
    parser.add_argument("--source-job-id", required=True)
    parser.add_argument("--publisher-job-id", required=True)
    return parser


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, sort_keys=True)
        handle.write("\n")
        temporary = handle.name
    os.replace(temporary, path)


def _object(path: Path, *, name: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{name} must be one regular non-symlinked file")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _array_exact(left: Any, right: Any) -> bool:
    first = np.ascontiguousarray(np.asarray(left))
    second = np.ascontiguousarray(np.asarray(right))
    return bool(
        first.shape == second.shape
        and first.dtype == second.dtype
        and first.tobytes() == second.tobytes()
    )


def _digest(path: Path, *, name: str) -> str:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{name} must be one regular non-symlinked file")
    return file_sha256(path)


def _require_sha256(value: str, *, name: str) -> str:
    if SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} is not a lowercase SHA-256 digest")
    return value


def _require_false_claims(value: dict[str, Any], *, name: str) -> None:
    for key in (
        "scientific_claim_allowed",
        "simulator_efficacy_claim_allowed",
        "infeasibility_claim_allowed",
        "probe_training_authorized",
    ):
        if value.get(key) is not False:
            raise ValueError(f"{name} changed its fail-closed claim: {key}")


def _allocation_test_summary(path: Path) -> dict[str, Any]:
    test_log = path.read_text(encoding="utf-8")
    observed = re.findall(
        r"^Ran ([1-9][0-9]*) tests? in ", test_log, flags=re.MULTILINE
    )
    if len(observed) != 1 or test_log.count("\nOK\n") != 1:
        raise ValueError("AF-00A allocation-focused tests are incomplete")
    if re.search(r"skipped=|^FAILED|^ERROR", test_log, flags=re.MULTILINE):
        raise ValueError("AF-00A allocation-focused tests skipped or failed")
    return {
        "passed": True,
        "test_count": int(observed[0]),
        "skipped_test_count": 0,
    }


def _host_telemetry_summary(path: Path, *, source_job_id: str) -> dict[str, Any]:
    telemetry = parse_full_lifetime_telemetry(path, expected_job_id=source_job_id)
    if telemetry.get("contract_passed") is not True:
        raise ValueError("AF-00A full-lifetime host telemetry did not pass")
    if telemetry.get("raw_trace_sha256") != _digest(
        path, name="AF-00A host telemetry"
    ):
        raise ValueError("AF-00A host telemetry parser digest changed")
    return {
        "contract_passed": True,
        "raw_trace_sha256": telemetry["raw_trace_sha256"],
        "sample_count": telemetry["sample_count"],
        "first_sample_monotonic_ns": telemetry["first_sample_monotonic_ns"],
        "last_sample_monotonic_ns": telemetry["last_sample_monotonic_ns"],
        "maximum_adjacent_gap_ns": telemetry["maximum_adjacent_gap_ns"],
        "host_cgroup_sampled_current_high_water_bytes": telemetry[
            "host_cgroup_sampled_current_high_water_bytes"
        ],
        "host_cgroup_sampled_current_is_lower_bound_not_peak": telemetry[
            "host_cgroup_sampled_current_is_lower_bound_not_peak"
        ],
        "memory_max_job_scope_hard_limit_bytes": telemetry[
            "memory_max_job_scope_hard_limit_bytes"
        ],
        "memory_events_deltas": telemetry["memory_events_deltas"],
        "native_memory_peak_state": telemetry["native_memory_peak_state"],
        "native_memory_peak_value_bytes": telemetry[
            "native_memory_peak_value_bytes"
        ],
        "full_lifetime_window_passed": telemetry["full_lifetime_window_passed"],
    }


def _canonical_uint(text: str, *, name: str, positive: bool = False) -> int:
    if re.fullmatch(r"0|[1-9][0-9]*", text) is None:
        raise ValueError(f"{name} is not a canonical unsigned integer")
    value = int(text)
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _gpu_telemetry_summary(path: Path) -> dict[str, Any]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != [
            "timestamp_ns",
            "gpu_uuid",
            "compute_mib",
            "device_mib",
        ]:
            raise ValueError("AF-00A GPU telemetry header changed")
        rows = list(reader)
    if not rows:
        raise ValueError("AF-00A GPU telemetry is empty")
    timestamps: list[int] = []
    compute_values: list[int] = []
    device_values: list[int] = []
    gpu_uuid: str | None = None
    for index, row in enumerate(rows):
        if set(row) != {
            "timestamp_ns",
            "gpu_uuid",
            "compute_mib",
            "device_mib",
        } or None in row:
            raise ValueError(f"AF-00A GPU telemetry row {index} is malformed")
        timestamp = _canonical_uint(
            row["timestamp_ns"], name=f"GPU timestamp row {index}", positive=True
        )
        compute = _canonical_uint(
            row["compute_mib"], name=f"GPU compute MiB row {index}"
        )
        device = _canonical_uint(
            row["device_mib"], name=f"GPU device MiB row {index}"
        )
        observed_uuid = row["gpu_uuid"]
        if not observed_uuid.startswith("GPU-"):
            raise ValueError(f"AF-00A GPU UUID row {index} is invalid")
        if gpu_uuid is None:
            gpu_uuid = observed_uuid
        elif observed_uuid != gpu_uuid:
            raise ValueError("AF-00A GPU telemetry mixed allocation devices")
        if timestamps and timestamp <= timestamps[-1]:
            raise ValueError("AF-00A GPU telemetry timestamps are not increasing")
        timestamps.append(timestamp)
        compute_values.append(compute)
        device_values.append(device)
    if max(compute_values) <= 0 or max(device_values) <= 0:
        raise ValueError("AF-00A GPU telemetry never observed the policy allocation")
    assert gpu_uuid is not None
    return {
        "gpu_uuid": gpu_uuid,
        "sample_count": len(rows),
        "first_timestamp_ns": timestamps[0],
        "last_timestamp_ns": timestamps[-1],
        "maximum_compute_mib": max(compute_values),
        "maximum_device_mib": max(device_values),
    }


def main() -> int:
    args = _parser().parse_args()
    output = Path(args.output)
    receipt = Path(args.receipt)
    try:
        run_root = Path(args.run_root)
        if not run_root.is_absolute() or run_root.parent != EXPERIMENT_ROOT:
            raise ValueError("AF-00A run root is not the exact immutable experiment root")
        if JOB_ID_PATTERN.fullmatch(args.source_job_id) is None:
            raise ValueError("AF-00A source job id is invalid")
        if JOB_ID_PATTERN.fullmatch(args.publisher_job_id) is None:
            raise ValueError("AF-00A publisher job id is invalid")
        _require_sha256(
            args.expected_source_contract_sha256,
            name="expected AF-00A source-contract digest",
        )
        _require_sha256(
            args.expected_held_gpu_submission_sha256,
            name="expected AF-00A held-GPU digest",
        )
        _require_sha256(
            args.expected_submission_sha256,
            name="expected AF-00A submission digest",
        )
        _require_sha256(
            args.expected_release_fingerprint_sha256,
            name="expected AF-00A release-fingerprint digest",
        )
        case_id = CASE_ID
        case_dir = run_root / case_id
        payload_path = case_dir / "af00a-raw-payload.json"
        tensors_path = case_dir / "af00a-tensors.npz"
        expected_output = case_dir / "results.json"
        expected_receipt = run_root / "cpu-afterany-validation.json"
        repo_root = Path(__file__).resolve().parents[1]
        actual_config_path = (
            repo_root / "configs" / "experiments" / "r05a_actual_forward_canary.json"
        )
        legacy_config_path = (
            repo_root / "configs" / "experiments" / "r05a_inverse_flow_canary.json"
        )
        source_contract_path = run_root / "source-contract.json"
        held_gpu_path = run_root / "held-gpu-submission.json"
        submission_path = run_root / "submission.json"
        release_fingerprint_path = run_root / "final-pre-release-fingerprint.json"
        if output != expected_output:
            raise ValueError("AF-00A output must be the exact immutable case results.json")
        if receipt != expected_receipt:
            raise ValueError("AF-00A receipt must be the exact immutable CPU receipt")
        supplied_paths = {
            "config": (Path(args.config), actual_config_path),
            "legacy config": (Path(args.legacy_config), legacy_config_path),
            "source contract": (Path(args.source_contract), source_contract_path),
            "held GPU submission": (Path(args.held_gpu_submission), held_gpu_path),
            "submission": (Path(args.submission), submission_path),
            "release fingerprint": (
                Path(args.release_fingerprint),
                release_fingerprint_path,
            ),
        }
        for label, (supplied, expected) in supplied_paths.items():
            if supplied != expected:
                raise ValueError(f"AF-00A {label} path is not the frozen path")
        payload = _object(payload_path, name="AF-00A raw payload")
        config = _object(actual_config_path, name="AF-00A config")
        legacy_config = _object(legacy_config_path, name="legacy R05A config")
        source_contract = _object(source_contract_path, name="AF-00A source contract")
        source_contract_sha256 = _digest(
            source_contract_path, name="AF-00A source contract"
        )
        if source_contract_sha256 != args.expected_source_contract_sha256:
            raise ValueError("AF-00A source contract differs from its external digest")
        held_gpu = _object(held_gpu_path, name="AF-00A held GPU submission")
        held_gpu_sha256 = _digest(held_gpu_path, name="AF-00A held GPU submission")
        if held_gpu_sha256 != args.expected_held_gpu_submission_sha256:
            raise ValueError("AF-00A held GPU receipt differs from its external digest")
        submission = _object(submission_path, name="AF-00A atomic submission")
        submission_sha256 = _digest(submission_path, name="AF-00A atomic submission")
        if submission_sha256 != args.expected_submission_sha256:
            raise ValueError("AF-00A submission differs from its external digest")
        release_fingerprint = _object(
            release_fingerprint_path, name="AF-00A final release fingerprint"
        )
        release_fingerprint_sha256 = _digest(
            release_fingerprint_path, name="AF-00A final release fingerprint"
        )
        if release_fingerprint_sha256 != args.expected_release_fingerprint_sha256:
            raise ValueError("AF-00A release fingerprint differs from its external digest")
        expected_dependency = f"afterany:{args.source_job_id}"
        source_contract_identity = {
            "schema_version": "1.0",
            "artifact_role": "r05a_actual_forward_canary_source_contract",
            "status": "gpu_held_sources_bound_before_cpu_submission",
            "run_id": run_root.name,
            "git_dirty": False,
            "source_node": "worker-1",
            "gpu_slurm_array_job_id": str(args.source_job_id),
            "gpu_slurm_array_task_id": 0,
            "exact_gpu_task_id": f"{args.source_job_id}_0",
        }
        for key, expected in source_contract_identity.items():
            if source_contract.get(key) != expected:
                raise ValueError(f"AF-00A source-contract identity changed: {key}")
        git_commit = source_contract.get("git_commit")
        if not isinstance(git_commit, str) or re.fullmatch(r"[0-9a-f]{40}", git_commit) is None:
            raise ValueError("AF-00A source contract release commit is invalid")
        _require_false_claims(source_contract, name="AF-00A source contract")
        artifact_paths = source_contract.get("artifact_paths")
        if not isinstance(artifact_paths, dict):
            raise ValueError("AF-00A source contract lacks artifact paths")
        preflight_path = run_root / "vinuni-preflight.txt"
        expected_paths = {
            "run_root": str(run_root),
            "case_dir": str(case_dir),
            "payload": str(payload_path),
            "raw_tensors": str(tensors_path),
            "query_ledger": str(case_dir / "query-ledger.json"),
            "host_telemetry": str(case_dir / "host-cgroup-sampled-current.tsv"),
            "gpu_samples": str(case_dir / "gpu-memory-samples.csv"),
            "allocation_tests_log": str(case_dir / "allocation-focused-tests.log"),
            "hidden_candidate": str(case_dir / ".results.candidate.json"),
            "result": str(output),
            "validation_receipt": str(receipt),
            "held_gpu_submission": str(held_gpu_path),
            "submission": str(submission_path),
            "release_fingerprint": str(release_fingerprint_path),
            "live_preflight": str(preflight_path),
        }
        if artifact_paths != expected_paths:
            raise ValueError("AF-00A source-contract artifact paths changed")
        if (
            source_contract.get("held_gpu_submission_path") != str(held_gpu_path)
            or source_contract.get("held_gpu_submission_sha256") != held_gpu_sha256
        ):
            raise ValueError("AF-00A source contract does not bind the held GPU receipt")
        expected_submission = {
            "schema_version": "1.0",
            "artifact_role": "r05a_actual_forward_canary_atomic_submission",
            "status": "cpu_afterany_registered_gpu_held",
            "run_id": run_root.name,
            "git_commit": git_commit,
            "source_node": "worker-1",
            "gpu_slurm_array_job_id": str(args.source_job_id),
            "gpu_slurm_array_task_id": 0,
            "exact_gpu_task_id": f"{args.source_job_id}_0",
            "cpu_afterany_job_id": str(args.publisher_job_id),
            "dependency": expected_dependency,
            "source_contract_path": str(source_contract_path),
            "source_contract_sha256": args.expected_source_contract_sha256,
            "held_gpu_submission_path": str(held_gpu_path),
            "held_gpu_submission_sha256": held_gpu_sha256,
            "release_fingerprint_path": str(release_fingerprint_path),
            "expected_result": str(output),
            "expected_validation_receipt": str(receipt),
            "released_at_receipt_time": False,
        }
        for key, expected in expected_submission.items():
            if submission.get(key) != expected:
                raise ValueError(f"AF-00A atomic submission changed: {key}")
        _require_false_claims(submission, name="AF-00A atomic submission")
        expected_held = {
            "schema_version": "1.0",
            "artifact_role": "r05a_actual_forward_canary_held_gpu_submission",
            "status": "sbatch_returned_held_gpu_id",
            "run_id": run_root.name,
            "git_commit": git_commit,
            "gpu_slurm_array_job_id": str(args.source_job_id),
            "gpu_slurm_array_task_id": 0,
            "exact_gpu_task_id": f"{args.source_job_id}_0",
            "source_node": "worker-1",
            "released_at_receipt_time": False,
        }
        for key, expected in expected_held.items():
            if held_gpu.get(key) != expected:
                raise ValueError(f"AF-00A held GPU receipt changed: {key}")
        _require_false_claims(held_gpu, name="AF-00A held GPU receipt")
        expected_fingerprint = {
            "schema_version": "1.0",
            "artifact_role": "r05a_actual_forward_canary_final_pre_release_fingerprint",
            "status": "all_receipts_bound_gpu_still_held",
            "run_id": run_root.name,
            "git_commit": git_commit,
            "source_node": "worker-1",
            "gpu_slurm_array_job_id": str(args.source_job_id),
            "gpu_slurm_array_task_id": 0,
            "exact_gpu_task_id": f"{args.source_job_id}_0",
            "cpu_afterany_job_id": str(args.publisher_job_id),
            "dependency": expected_dependency,
            "source_contract_path": str(source_contract_path),
            "source_contract_sha256": args.expected_source_contract_sha256,
            "held_gpu_submission_path": str(held_gpu_path),
            "held_gpu_submission_sha256": held_gpu_sha256,
            "submission_path": str(submission_path),
            "submission_sha256": submission_sha256,
            "expected_result": str(output),
            "expected_validation_receipt": str(receipt),
            "released_at_fingerprint_time": False,
        }
        for key, expected in expected_fingerprint.items():
            if release_fingerprint.get(key) != expected:
                raise ValueError(f"AF-00A final release fingerprint changed: {key}")
        _require_false_claims(
            release_fingerprint, name="AF-00A final release fingerprint"
        )
        repository_hashes = source_contract.get("repository_file_sha256")
        if not isinstance(repository_hashes, dict) or not repository_hashes:
            raise ValueError("AF-00A source contract lacks repository hashes")
        for relative, digest in repository_hashes.items():
            if not isinstance(relative, str) or not isinstance(digest, str):
                raise ValueError("AF-00A repository hash binding is malformed")
            relative_path = Path(relative)
            if (
                relative_path.is_absolute()
                or relative in {"", "."}
                or any(part in {"", ".", ".."} for part in relative_path.parts)
            ):
                raise ValueError("AF-00A repository hash path is not normalized")
            _require_sha256(digest, name=f"repository digest for {relative}")
            bound = repo_root / relative
            if not bound.is_file() or bound.is_symlink() or file_sha256(bound) != digest:
                raise ValueError(f"AF-00A repository source changed: {relative}")
        frozen_bindings = source_contract.get("frozen_bindings")
        if not isinstance(frozen_bindings, dict):
            raise ValueError("AF-00A source contract lacks frozen bindings")
        actual_config_sha256 = _digest(actual_config_path, name="AF-00A config")
        legacy_config_sha256 = _digest(legacy_config_path, name="legacy R05A config")
        if frozen_bindings.get("actual_config_sha256") != actual_config_sha256:
            raise ValueError("AF-00A config differs from its source binding")
        if frozen_bindings.get("legacy_config_sha256") != legacy_config_sha256:
            raise ValueError("legacy config differs from its source binding")
        preflight_sha256 = _digest(preflight_path, name="AF-00A live preflight")
        if frozen_bindings.get("live_preflight_sha256") != preflight_sha256:
            raise ValueError("AF-00A live preflight differs from its source binding")
        if config.get("ready_to_run") is not True or config.get("blocked_on") != []:
            raise ValueError("AF-00A config is not released")
        release = config.get("execution_release")
        if not isinstance(release, dict) or release.get("run_id") != run_root.name:
            raise ValueError("AF-00A release/run-root identity changed")
        released_resources = release.get("resources")
        if (
            not isinstance(released_resources, dict)
            or source_contract.get("resources") != released_resources
            or held_gpu.get("resources") != released_resources
        ):
            raise ValueError("AF-00A held/source resources differ from the release")
        if legacy_config.get("ready_to_run") is not True:
            raise ValueError("legacy R05A config is not released")

        source_bindings = config.get("frozen_source_bindings")
        frozen_case = config.get("frozen_case")
        if not isinstance(source_bindings, dict) or not isinstance(frozen_case, dict):
            raise ValueError("AF-00A config lacks frozen source/case bindings")
        manifest_binding = source_bindings.get("manifest")
        r02_binding = source_bindings.get("source_r02")
        r02_config_binding = source_bindings.get("source_r02_config")
        if (
            not isinstance(manifest_binding, dict)
            or not isinstance(r02_binding, dict)
            or not isinstance(r02_config_binding, dict)
        ):
            raise ValueError("AF-00A config source bindings are malformed")
        manifest_path = repo_root / str(manifest_binding.get("path"))
        if not manifest_path.is_file() or manifest_path.is_symlink():
            raise ValueError("AF-00A manifest is missing or symlinked")
        manifest_sha256 = file_sha256(manifest_path)
        if manifest_sha256 != manifest_binding.get("sha256"):
            raise ValueError("AF-00A manifest differs from the released config")
        if frozen_bindings.get("manifest_sha256") != manifest_sha256:
            raise ValueError("AF-00A manifest differs from the source contract")
        r02_config_path = repo_root / str(r02_config_binding.get("path"))
        r02_config_sha256 = _digest(r02_config_path, name="AF-00A R02 config")
        if (
            r02_config_sha256 != r02_config_binding.get("sha256")
            or r02_config_sha256 != frozen_bindings.get("source_r02_config_sha256")
        ):
            raise ValueError("AF-00A R02 config differs from its released binding")
        manifest_cases, manifest_errors = validate_jsonl_unique(manifest_path, "case_id")
        for item in manifest_cases:
            manifest_errors.extend(
                f"{item.get('case_id', '<unknown>')}: {error}"
                for error in validate_case(item)
            )
        row_index = manifest_binding.get("row_index_zero_based")
        if manifest_errors or row_index != 0 or len(manifest_cases) != 3:
            raise ValueError("AF-00A manifest failed independent validation")
        manifest_case = manifest_cases[0]
        for key in (
            "case_id",
            "group_id",
            "environment_seed",
            "policy_seed",
            "task_suite",
            "safety_level",
            "task_index",
            "episode_index",
        ):
            if manifest_case.get(key) != frozen_case.get(key):
                raise ValueError(f"AF-00A frozen case differs from manifest: {key}")

        r02_path = Path(str(r02_binding.get("path")))
        if not r02_path.is_file() or r02_path.is_symlink():
            raise ValueError("AF-00A immutable R02 source is missing or symlinked")
        r02_sha256 = file_sha256(r02_path)
        if r02_sha256 != r02_binding.get("sha256"):
            raise ValueError("AF-00A R02 source differs from the released config")
        if frozen_bindings.get("source_r02_sha256") != r02_sha256:
            raise ValueError("AF-00A R02 source differs from the source contract")
        raw_r02 = _object(r02_path, name="immutable AF-00A R02 source")
        r02_errors = validate_r02_result(raw_r02)
        if r02_errors:
            raise ValueError("AF-00A R02 source failed validation: " + "; ".join(r02_errors))
        if raw_r02.get("case_id") != frozen_case.get("case_id") or raw_r02.get("status") != "completed":
            raise ValueError("AF-00A R02 source is not the completed frozen case")

        checkpoint_path = CHECKPOINT_MODEL
        checkpoint_sha256 = _digest(checkpoint_path, name="AF-00A checkpoint model")
        checkpoint_model_binding = source_bindings.get("checkpoint_model")
        if (
            not isinstance(checkpoint_model_binding, dict)
            or checkpoint_model_binding.get("path") != str(CHECKPOINT_MODEL)
            or checkpoint_model_binding.get("sha256") != checkpoint_sha256
            or checkpoint_sha256 != source_bindings.get("checkpoint_sha256")
        ):
            raise ValueError("AF-00A checkpoint differs from the released config")
        if frozen_bindings.get("checkpoint_sha256") != checkpoint_sha256:
            raise ValueError("AF-00A checkpoint differs from the source contract")
        checkpoint_config_binding = source_bindings.get("checkpoint_config")
        normalization_binding = source_bindings.get("normalization_asset")
        if not isinstance(checkpoint_config_binding, dict) or not isinstance(
            normalization_binding, dict
        ):
            raise ValueError("AF-00A runtime checkpoint assets are not fully bound")
        checkpoint_config_path = Path(str(checkpoint_config_binding.get("path")))
        normalization_path = Path(str(normalization_binding.get("path")))
        if checkpoint_config_path != CHECKPOINT_CONFIG:
            raise ValueError("AF-00A checkpoint config path changed")
        if normalization_path != NORMALIZATION_ASSET:
            raise ValueError("AF-00A normalization asset path changed")
        checkpoint_config_sha256: str | None = None
        normalization_sha256: str | None = None
        for label, path, binding, source_key in (
            (
                "checkpoint config",
                checkpoint_config_path,
                checkpoint_config_binding,
                "checkpoint_config_sha256",
            ),
            (
                "normalization asset",
                normalization_path,
                normalization_binding,
                "normalization_asset_sha256",
            ),
        ):
            digest = _digest(path, name=f"AF-00A {label}")
            if digest != binding.get("sha256") or digest != frozen_bindings.get(source_key):
                raise ValueError(f"AF-00A {label} differs from its frozen bindings")
            if source_key == "checkpoint_config_sha256":
                checkpoint_config_sha256 = digest
            else:
                normalization_sha256 = digest
        assert checkpoint_config_sha256 is not None
        assert normalization_sha256 is not None
        if normalization_binding.get("sha256") != source_bindings.get(
            "normalization_asset_sha256"
        ):
            raise ValueError("AF-00A normalization scalar/object bindings disagree")

        if payload.get("run_id") != run_root.name or payload.get("case_id") != case_id:
            raise ValueError("AF-00A raw payload identity changed")
        payload_source = payload.get("source")
        payload_configs = payload.get("configs")
        if not isinstance(payload_source, dict) or not isinstance(payload_configs, dict):
            raise ValueError("AF-00A raw payload lacks source/config bindings")
        if payload_source.get("r02_path") != str(r02_path) or payload_source.get("r02_sha256") != r02_sha256:
            raise ValueError("AF-00A raw payload R02 binding changed")
        source_delta_record = raw_r02.get("directions", {}).get("arrays", {}).get(
            "delta_star_model"
        )
        if payload_source.get("delta_star_model") != source_delta_record:
            raise ValueError("AF-00A raw payload Delta* record changed")
        if payload_source.get("source_budget_float64") != raw_r02.get("directions", {}).get(
            "l2_norms", {}
        ).get("delta_star_model"):
            raise ValueError("AF-00A raw payload source budget changed")
        payload_budget32 = np.float32(payload_source.get("source_budget_float32"))
        released_budget32 = np.float32(
            config.get("target_contract", {}).get("source_budget_float32")
        )
        if payload_budget32.tobytes() != released_budget32.tobytes():
            raise ValueError("AF-00A raw payload float32 budget changed")
        if payload_configs.get("actual_sha256") != file_sha256(args.config):
            raise ValueError("AF-00A raw payload config digest changed")
        if payload_configs.get("legacy_sha256") != file_sha256(args.legacy_config):
            raise ValueError("AF-00A raw payload legacy-config digest changed")
        ledger = payload.get("request_ledger")
        if not isinstance(ledger, list) or len(ledger) != POLICY_REQUESTS:
            raise ValueError(f"AF-00A raw payload must contain {POLICY_REQUESTS} request rows")
        query_ledger_path = Path(expected_paths["query_ledger"])
        query_ledger = _object(query_ledger_path, name="AF-00A query ledger")
        if (
            query_ledger.get("case_id") != case_id
            or query_ledger.get("exact_policy_request_count") != POLICY_REQUESTS
            or query_ledger.get("rows") != ledger
        ):
            raise ValueError("AF-00A query-ledger sidecar differs from raw payload")
        tensor_artifact = payload.get("tensor_archive")
        if not isinstance(tensor_artifact, dict):
            raise ValueError("AF-00A raw payload lacks its tensor artifact binding")
        if tensor_artifact.get("path") != str(tensors_path):
            raise ValueError("AF-00A raw payload tensor path changed")
        tensors_sha256 = file_sha256(tensors_path)
        if tensor_artifact.get("sha256") != tensors_sha256:
            raise ValueError("AF-00A raw payload tensor digest changed")
        payload_artifacts = payload.get("artifacts")
        if not isinstance(payload_artifacts, dict):
            raise ValueError("AF-00A raw payload lacks sidecar artifact bindings")
        if (
            payload_artifacts.get("query_ledger") != query_ledger_path.name
            or payload_artifacts.get("query_ledger_sha256") != file_sha256(query_ledger_path)
            or query_ledger.get("tensor_archive_sha256") != tensors_sha256
        ):
            raise ValueError("AF-00A query-ledger artifact binding changed")
        provenance = payload.get("provenance")
        if not isinstance(provenance, dict):
            raise ValueError("AF-00A raw payload lacks provenance")
        if str(provenance.get("slurm_array_job_id")) != str(args.source_job_id):
            raise ValueError("AF-00A raw payload source job changed")
        if str(provenance.get("slurm_array_task_id")) != "0":
            raise ValueError("AF-00A raw payload is not singleton task zero")
        if str(provenance.get("slurm_job_id")) != str(args.source_job_id):
            raise ValueError("AF-00A raw payload exact Slurm job changed")
        if provenance.get("host") != "worker-1":
            raise ValueError("AF-00A raw payload source host changed")

        source_pairing = raw_r02.get("pairing")
        pairing_records = payload.get("source_pairing_records")
        if not isinstance(source_pairing, dict) or not isinstance(pairing_records, dict):
            raise ValueError("AF-00A source pairing records are missing")
        pairing_keys = {
            "policy_observation_source": "policy_observation",
            "branch_snapshot_source": "branch_snapshot",
            "eager_actions_source": "eager_actions",
            "eager_trace_source": "eager_trace",
        }
        for payload_key, r02_key in pairing_keys.items():
            if pairing_records.get(payload_key) != source_pairing.get(r02_key):
                raise ValueError(f"AF-00A serialized source pairing changed: {payload_key}")
        if pairing_records.get("policy_observation_current") != source_pairing.get("policy_observation"):
            raise ValueError("AF-00A current observation differs from R02")
        if pairing_records.get("branch_snapshot_current") != source_pairing.get("branch_snapshot"):
            raise ValueError("AF-00A current branch differs from R02")
        for key in ("eager_actions_before_current", "eager_actions_after_current"):
            if pairing_records.get(key) != source_pairing.get("eager_actions"):
                raise ValueError(f"AF-00A current source actions differ from R02: {key}")
        for key in ("eager_trace_before_current", "eager_trace_after_current"):
            if pairing_records.get(key) != source_pairing.get("eager_trace"):
                raise ValueError(f"AF-00A current source trace differs from R02: {key}")

        source_delta, delta_errors = _validate_array_record(
            raw_r02.get("directions", {}).get("arrays", {}).get("delta_star_model"),
            name="directions.arrays.delta_star_model",
            shape=(10, 32),
        )
        source_noise, noise_errors = _validate_array_record(
            raw_r02.get("provenance", {}).get("noise"),
            name="provenance.noise",
            shape=(10, 32),
        )
        source_actions, action_errors = _validate_array_record(
            source_pairing.get("eager_actions"),
            name="pairing.eager_actions",
            shape=(10, 7),
        )
        if delta_errors or noise_errors or action_errors:
            raise ValueError("AF-00A R02 source arrays failed reconstruction")
        if source_delta is None or source_noise is None or source_actions is None:
            raise ValueError("AF-00A R02 source arrays are missing")
        expected_noise = np.random.default_rng(int(frozen_case["policy_seed"])).normal(
            size=(10, 32)
        ).astype(np.float32)
        if not _array_exact(np.asarray(source_noise, dtype=np.float32), expected_noise):
            raise ValueError("AF-00A R02 noise does not reconstruct from policy seed")
        with np.load(tensors_path, allow_pickle=False) as archive:
            expected_delta32 = np.asarray(source_delta, dtype=np.float32)
            if not _array_exact(archive["source_delta_model_f32"], expected_delta32):
                raise ValueError("AF-00A tensor Delta* differs from R02 exact float32 cast")
            if not _array_exact(archive["source_noise_f32"], expected_noise):
                raise ValueError("AF-00A tensor noise differs from the frozen policy seed")
            expected_actions64 = np.asarray(source_actions, dtype=np.float64)
            expected_action_pair = np.stack((expected_actions64, expected_actions64))
            if not _array_exact(archive["reference_source_actions_f64"], expected_action_pair):
                raise ValueError("AF-00A tensor source actions differ from R02")
            expected_scale = np.ones((10, 32), dtype=np.float32)
            expected_scale[:, :3] = np.asarray(REGISTERED_XYZ_SCALE, dtype=np.float32)
            if not _array_exact(archive["source_model_to_physical_scale_f32"], expected_scale):
                raise ValueError("AF-00A model-to-physical scale changed")
        accounting = payload.get("request_accounting")
        if not isinstance(accounting, dict) or accounting.get("exact") is not True:
            raise ValueError("AF-00A raw payload request accounting is not exact")
        if (
            accounting.get("total") != POLICY_REQUESTS
            or accounting.get("cem") != 520
            or accounting.get("trace_rows") != 528
        ):
            raise ValueError("AF-00A raw payload request counts changed")
        expected_search_runtime = {
            "numpy_version": np.__version__,
            "bit_generator": "PCG64",
            "seed": 20260716,
            "generations": 8,
            "population_size": 65,
            "search_requests": 520,
        }
        if payload.get("search_runtime") != expected_search_runtime:
            raise ValueError("AF-00A search runtime identity changed")
        raw_boundary = payload.get("execution_boundary")
        if not isinstance(raw_boundary, dict):
            raise ValueError("AF-00A raw payload lacks its execution boundary")
        expected_raw_boundary = {
            "policy_requests": POLICY_REQUESTS,
            "policy_generated_action_steps_executed": 0,
            "teacher_generated_action_steps_executed": 0,
            "efficacy_rollouts_executed": 0,
            "simulator_efficacy_evaluated": False,
            "training": False,
        }
        if raw_boundary != expected_raw_boundary:
            raise ValueError("AF-00A raw execution boundary changed")

        allocation_tests_path = Path(expected_paths["allocation_tests_log"])
        telemetry_path = Path(expected_paths["host_telemetry"])
        gpu_telemetry_path = Path(expected_paths["gpu_samples"])
        allocation_test_summary = _allocation_test_summary(allocation_tests_path)
        host_telemetry_summary = _host_telemetry_summary(
            telemetry_path, source_job_id=str(args.source_job_id)
        )
        gpu_telemetry_summary = _gpu_telemetry_summary(gpu_telemetry_path)
        measured_telemetry = {
            "allocation_tests": allocation_test_summary,
            "host": host_telemetry_summary,
            "gpu": gpu_telemetry_summary,
        }

        validation = validate_actual_forward_npz(tensors_path, ledger)
        outcome = validation.get("outcome")
        allowed = {
            "mechanism_pass",
            "baseline_sufficient_no_incremental_support",
            "frozen_cem_negative",
        }
        if outcome not in allowed:
            raise ValueError(f"AF-00A validator returned an unknown outcome: {outcome!r}")
        if payload.get("status") != outcome:
            raise ValueError("AF-00A GPU status differs from independent validation")
        raw_evidence = {
            "source_contract_path": str(source_contract_path),
            "source_contract_sha256": source_contract_sha256,
            "held_gpu_submission_path": str(held_gpu_path),
            "held_gpu_submission_sha256": held_gpu_sha256,
            "submission_path": str(submission_path),
            "submission_sha256": submission_sha256,
            "release_fingerprint_path": str(release_fingerprint_path),
            "release_fingerprint_sha256": release_fingerprint_sha256,
            "payload_path": str(payload_path),
            "payload_sha256": _digest(payload_path, name="AF-00A raw payload"),
            "tensor_path": str(tensors_path),
            "tensor_sha256": tensors_sha256,
            "query_ledger_path": str(query_ledger_path),
            "query_ledger_sha256": _digest(
                query_ledger_path, name="AF-00A query ledger"
            ),
            "allocation_tests_log_path": str(allocation_tests_path),
            "allocation_tests_log_sha256": _digest(
                allocation_tests_path, name="AF-00A allocation tests"
            ),
            "host_telemetry_path": str(telemetry_path),
            "host_telemetry_sha256": _digest(
                telemetry_path, name="AF-00A host telemetry"
            ),
            "gpu_telemetry_path": str(gpu_telemetry_path),
            "gpu_telemetry_sha256": _digest(
                gpu_telemetry_path, name="AF-00A GPU telemetry"
            ),
            "live_preflight_path": str(preflight_path),
            "live_preflight_sha256": preflight_sha256,
            "actual_config_path": str(actual_config_path),
            "actual_config_sha256": actual_config_sha256,
            "legacy_config_path": str(legacy_config_path),
            "legacy_config_sha256": legacy_config_sha256,
            "manifest_path": str(manifest_path),
            "manifest_sha256": manifest_sha256,
            "source_r02_path": str(r02_path),
            "source_r02_sha256": r02_sha256,
            "source_r02_config_path": str(r02_config_path),
            "source_r02_config_sha256": r02_config_sha256,
            "checkpoint_model_path": str(checkpoint_path),
            "checkpoint_model_sha256": checkpoint_sha256,
            "checkpoint_config_path": str(checkpoint_config_path),
            "checkpoint_config_sha256": checkpoint_config_sha256,
            "normalization_asset_path": str(normalization_path),
            "normalization_asset_sha256": normalization_sha256,
        }
        result_value = {
            "schema_version": "1.0",
            "artifact_type": "r05a_actual_forward_canary_result",
            "gate": "R05A",
            "experiment_identity": "AF-00A",
            "run_id": run_root.name,
            "case_id": case_id,
            "status": outcome,
            "source_job_id": f"{args.source_job_id}_0",
            "publisher_job_id": str(args.publisher_job_id),
            "execution_identity": {
                "git_commit": git_commit,
                "source_node": "worker-1",
                "gpu_slurm_array_job_id": str(args.source_job_id),
                "gpu_slurm_array_task_id": 0,
                "exact_gpu_task_id": f"{args.source_job_id}_0",
                "cpu_afterany_job_id": str(args.publisher_job_id),
                "dependency": expected_dependency,
            },
            "raw_evidence": raw_evidence,
            "measured_telemetry": measured_telemetry,
            "validation": validation,
            "execution_boundary": {
                "policy_generated_action_steps_executed": 0,
                "teacher_generated_action_steps_executed": 0,
                "efficacy_rollouts_executed": 0,
                "simulator_efficacy_evaluated": False,
            },
            "claims": {
                "scientific_claim_allowed": False,
                "simulator_efficacy_claim_allowed": False,
                "collision_or_progress_claim_allowed": False,
                "infeasibility_claim_allowed": False,
                "probe_training_authorized": False,
                "automatic_next_gate_authorized": False,
            },
        }
        candidate = output.parent / ".results.candidate.json"
        if output.exists() or candidate.exists() or receipt.exists():
            raise ValueError("AF-00A publication target is not fresh")
        _atomic_json(candidate, result_value)
        # Re-read the candidate before the sole atomic publication rename.
        candidate_value = _object(candidate, name="AF-00A result candidate")
        if candidate_value != result_value:
            raise ValueError("AF-00A result candidate round trip changed")
        schema_path = repo_root / "schemas" / "r05a-actual-forward-canary-envelope.schema.json"
        schema_relative = "schemas/r05a-actual-forward-canary-envelope.schema.json"
        receipt_schema_path = (
            repo_root
            / "schemas"
            / "r05a-actual-forward-canary-publication-receipt.schema.json"
        )
        receipt_schema_relative = (
            "schemas/r05a-actual-forward-canary-publication-receipt.schema.json"
        )
        artifact_contract = config.get("artifact_contract")
        if not isinstance(artifact_contract, dict):
            raise ValueError("AF-00A config lacks its artifact contract")
        if (
            artifact_contract.get("raw_payload_name") != payload_path.name
            or artifact_contract.get("tensor_archive_name") != tensors_path.name
            or artifact_contract.get("published_result_name") != output.name
            or artifact_contract.get("publication_receipt_name") != receipt.name
            or artifact_contract.get("envelope_schema_path") != schema_relative
            or artifact_contract.get("envelope_schema_sha256")
            != file_sha256(schema_path)
            or artifact_contract.get("publication_receipt_schema_path")
            != receipt_schema_relative
            or artifact_contract.get("publication_receipt_schema_sha256")
            != file_sha256(receipt_schema_path)
        ):
            raise ValueError("AF-00A publication schemas differ from released config")
        if repository_hashes.get(schema_relative) != file_sha256(schema_path):
            raise ValueError("AF-00A envelope schema differs from its source binding")
        if repository_hashes.get(receipt_schema_relative) != file_sha256(
            receipt_schema_path
        ):
            raise ValueError("AF-00A receipt schema differs from its source binding")
        schema = _object(schema_path, name="AF-00A envelope schema")
        receipt_schema = _object(
            receipt_schema_path, name="AF-00A publication-receipt schema"
        )
        import jsonschema

        jsonschema.Draft202012Validator.check_schema(schema)
        jsonschema.Draft202012Validator.check_schema(receipt_schema)
        jsonschema.Draft202012Validator(schema).validate(candidate_value)
        candidate_sha256 = _digest(candidate, name="AF-00A result candidate")
        receipt_value = {
            "schema_version": "1.0",
            "artifact_role": "r05a_actual_forward_canary_cpu_publication",
            "run_id": run_root.name,
            "case_id": case_id,
            "git_commit": git_commit,
            "source_node": "worker-1",
            "source_job_id": str(args.source_job_id),
            "source_task_id": f"{args.source_job_id}_0",
            "publisher_job_id": str(args.publisher_job_id),
            "dependency": expected_dependency,
            "result_path": str(output),
            "result_sha256": candidate_sha256,
            "published": True,
            "passed": True,
            "outcome": outcome,
            "evidence": raw_evidence,
            "measured_telemetry": measured_telemetry,
            "scientific_claim_allowed": False,
            "simulator_efficacy_claim_allowed": False,
            "infeasibility_claim_allowed": False,
            "probe_training_authorized": False,
            "automatic_next_gate_authorized": False,
        }
        jsonschema.Draft202012Validator(receipt_schema).validate(receipt_value)
        os.replace(candidate, output)
        if _digest(output, name="AF-00A published result") != candidate_sha256:
            raise ValueError("AF-00A published result digest changed after rename")
        _atomic_json(receipt, receipt_value)
    except Exception as error:
        output.parent.joinpath(".results.candidate.json").unlink(missing_ok=True)
        if not receipt.exists():
            _atomic_json(
                receipt,
                {
                    "schema_version": "1.0",
                    "artifact_role": "r05a_actual_forward_canary_cpu_publication",
                    "source_job_id": str(args.source_job_id),
                    "source_task_id": f"{args.source_job_id}_0",
                    "publisher_job_id": str(args.publisher_job_id),
                    "result_path": str(output),
                    "result_sha256": None,
                    "published": False,
                    "passed": False,
                    "errors": [str(error)],
                    "simulator_efficacy_claim_allowed": False,
                    "infeasibility_claim_allowed": False,
                    "probe_training_authorized": False,
                    "automatic_next_gate_authorized": False,
                },
            )
        print(f"AF-00A publication failed: {error}")
        return 1
    print(f"published {output}")
    print(f"receipt {receipt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
