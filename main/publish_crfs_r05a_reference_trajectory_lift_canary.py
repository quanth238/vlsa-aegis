#!/usr/bin/env python3
"""Independently validate and publish one finite TRL-00A canary artifact.

The GPU allocation is a raw-evidence producer only. This CPU afterany
publisher accepts exactly one complete 18-request finite ledger, independently
reconstructs every scientific quantity through the NumPy validator, and then
atomically publishes results.json. A missing artifact, typed terminal,
nonfinite leaf, producer exception, or validation error is apparatus-
inconclusive and can create only a failure receipt.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping

CASE_ID = "crfs-1069f29a8d76463a"
GROUP_ID = "safelibero_spatial:II:0:46"
RESULT_SCHEMA = "schemas/r05a-reference-trajectory-lift-canary-envelope.schema.json"
RECEIPT_SCHEMA = (
    "schemas/r05a-reference-trajectory-lift-canary-publication-receipt.schema.json"
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


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


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_digest(value: str, *, name: str) -> None:
    _require(SHA256_RE.fullmatch(value) is not None, f"{name} is not SHA-256")


def _regular(path: Path, *, name: str) -> Path:
    _require(path.exists(), f"{name} is missing")
    _require(path.is_file() and not path.is_symlink(), f"{name} is not a regular file")
    return path


def _object(path: Path, *, name: str) -> dict[str, Any]:
    _regular(path, name=name)
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"{name} must be a JSON object")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _exact_paths(
    source: Mapping[str, Any],
    *,
    run_root: Path,
    output: Path,
    receipt: Path,
) -> dict[str, Path]:
    case_dir = run_root / CASE_ID
    expected = {
        "run_root": run_root,
        "case_dir": case_dir,
        "payload": case_dir / "trl00a-raw-payload.json",
        "raw_tensors": case_dir / "trl00a-tensors.npz",
        "request_ledger": case_dir / "request-ledger.json",
        "host_telemetry": case_dir / "host-cgroup-sampled-current.tsv",
        "gpu_samples": case_dir / "gpu-memory-samples.csv",
        "allocation_tests_log": case_dir / "allocation-focused-tests.log",
        "hidden_candidate": case_dir / ".results.candidate.json",
        "result": output,
        "validation_receipt": receipt,
        "held_gpu_submission": run_root / "held-gpu-submission.json",
        "submission": run_root / "submission.json",
        "release_fingerprint": run_root / "final-pre-release-fingerprint.json",
        "live_preflight": run_root / "vinuni-preflight.txt",
    }
    observed = source.get("artifact_paths")
    _require(isinstance(observed, Mapping), "source contract lacks artifact paths")
    _require(set(observed) == set(expected), "source artifact path keys changed")
    for key, path in expected.items():
        _require(observed.get(key) == str(path), f"artifact path changed: {key}")
    return expected


def _validate_release(config: Mapping[str, Any], run_id: str) -> Mapping[str, Any]:
    _require(config.get("config_status") == "released_exact_single_canary", "config is not released")
    _require(config.get("ready_to_run") is True, "config is not ready")
    _require(config.get("blocked_on") == [], "config still has blockers")
    preregistration = config.get("preregistration")
    _require(isinstance(preregistration, Mapping), "config lacks preregistration")
    _require(
        preregistration.get("h100_submission_authorized") is True,
        "H100 submission was not authorized",
    )
    release = config.get("execution_release")
    _require(isinstance(release, Mapping), "config lacks execution release")
    _require(release.get("run_id") == run_id, "execution release run ID changed")
    _require(
        release.get("artifact_role")
        == "r05a_reference_trajectory_lift_canary_execution_release",
        "execution release role changed",
    )
    _require(release.get("single_submission") is True, "release is not single submission")
    _require(release.get("source_host") == "worker-1", "release source host changed")
    expected_resources = {
        "partition": "main",
        "account": "normal",
        "qos": "normal",
        "gpus": 1,
        "cpus_per_task": 8,
        "host_memory_mib": 65536,
        "time_limit": "00:30:00",
        "array": "0-0%1",
        "requeue": False,
        "validator_partition": "main",
        "validator_account": "normal",
        "validator_qos": "normal",
        "validator_cpus": 2,
        "validator_host_memory_mib": 8192,
        "validator_time_limit": "00:15:00",
        "validator_gpus": 0,
        "validator_dependency": "afterany",
    }
    _require(release.get("resources") == expected_resources, "released resources changed")
    _require(
        release.get("automatic_resubmission_allowed") is False
        and release.get("automatic_next_experiment_allowed") is False,
        "release permits an automatic next action",
    )
    ledger = config.get("request_ledger")
    _require(
        isinstance(ledger, Mapping)
        and ledger.get("complete_finite_exact_policy_request_count") == 18,
        "released finite request count changed",
    )
    lift = config.get("reference_lift_contract")
    _require(
        isinstance(lift, Mapping)
        and lift.get("raw_nonfinite_result")
        == "apparatus_inconclusive_no_scientific_results_json",
        "raw-nonfinite outcome changed",
    )
    artifact = config.get("artifact_contract")
    _require(isinstance(artifact, Mapping), "config lacks artifact contract")
    expected_artifacts = {
        "raw_payload_name": "trl00a-raw-payload.json",
        "tensor_archive_name": "trl00a-tensors.npz",
        "request_ledger_name": "request-ledger.json",
        "published_result_name": "results.json",
        "publication_receipt_name": "cpu-afterany-validation.json",
        "gpu_may_publish_results_json": False,
        "cpu_afterany_is_sole_publisher": True,
        "npz_allow_pickle": False,
        "envelope_schema_path": RESULT_SCHEMA,
        "publication_receipt_schema_path": RECEIPT_SCHEMA,
    }
    for key, expected in expected_artifacts.items():
        _require(artifact.get(key) == expected, f"artifact contract changed: {key}")
    for key in ("envelope_schema_sha256", "publication_receipt_schema_sha256"):
        value = artifact.get(key)
        _require(isinstance(value, str), f"artifact contract lacks {key}")
        _require_digest(value, name=key)
    return release


def _validate_transaction(
    args: argparse.Namespace,
    *,
    run_root: Path,
    config: Mapping[str, Any],
    repo_root: Path,
) -> tuple[dict[str, Any], dict[str, Path], str]:
    for name in (
        "expected_source_contract_sha256",
        "expected_held_gpu_submission_sha256",
        "expected_submission_sha256",
        "expected_release_fingerprint_sha256",
    ):
        _require_digest(getattr(args, name), name=name)
    _require(str(args.source_job_id).isdigit(), "source job ID is not numeric")
    _require(str(args.publisher_job_id).isdigit(), "publisher job ID is not numeric")
    source_path = Path(args.source_contract)
    held_path = Path(args.held_gpu_submission)
    submission_path = Path(args.submission)
    fingerprint_path = Path(args.release_fingerprint)
    expected_transaction_paths = {
        source_path: run_root / "source-contract.json",
        held_path: run_root / "held-gpu-submission.json",
        submission_path: run_root / "submission.json",
        fingerprint_path: run_root / "final-pre-release-fingerprint.json",
    }
    for observed, expected in expected_transaction_paths.items():
        _require(observed == expected, f"transaction path changed: {observed.name}")
    expected_digests = {
        source_path: args.expected_source_contract_sha256,
        held_path: args.expected_held_gpu_submission_sha256,
        submission_path: args.expected_submission_sha256,
        fingerprint_path: args.expected_release_fingerprint_sha256,
    }
    for path, expected in expected_digests.items():
        _regular(path, name=path.name)
        _require(_sha256(path) == expected, f"transaction digest changed: {path.name}")

    source = _object(source_path, name="TRL source contract")
    held = _object(held_path, name="held GPU receipt")
    submission = _object(submission_path, name="atomic submission receipt")
    fingerprint = _object(fingerprint_path, name="release fingerprint")
    git_commit = source.get("git_commit")
    _require(
        isinstance(git_commit, str) and re.fullmatch(r"[0-9a-f]{40}", git_commit),
        "source commit changed",
    )
    common = {
        "run_id": run_root.name,
        "git_commit": git_commit,
        "source_node": "worker-1",
        "gpu_slurm_array_job_id": str(args.source_job_id),
        "gpu_slurm_array_task_id": 0,
        "exact_gpu_task_id": f"{args.source_job_id}_0",
    }
    _require(
        source.get("artifact_role")
        == "r05a_reference_trajectory_lift_canary_source_contract"
        and source.get("status") == "gpu_held_sources_bound_before_cpu_submission"
        and source.get("git_dirty") is False,
        "source contract identity changed",
    )
    _require(
        held.get("artifact_role")
        == "r05a_reference_trajectory_lift_canary_held_gpu_submission",
        "held GPU receipt role changed",
    )
    _require(
        submission.get("artifact_role")
        == "r05a_reference_trajectory_lift_canary_atomic_submission",
        "submission receipt role changed",
    )
    _require(
        fingerprint.get("artifact_role")
        == "r05a_reference_trajectory_lift_canary_final_pre_release_fingerprint",
        "release fingerprint role changed",
    )
    for name, value in (
        ("source", source),
        ("held", held),
        ("submission", submission),
        ("fingerprint", fingerprint),
    ):
        for key, expected in common.items():
            _require(value.get(key) == expected, f"{name} field changed: {key}")
    dependency = f"afterany:{args.source_job_id}"
    for name, value in (("submission", submission), ("fingerprint", fingerprint)):
        _require(
            value.get("cpu_afterany_job_id") == str(args.publisher_job_id),
            f"{name} publisher changed",
        )
        _require(value.get("dependency") == dependency, f"{name} dependency changed")
    _require(
        submission.get("source_contract_path") == str(source_path)
        and submission.get("source_contract_sha256")
        == args.expected_source_contract_sha256,
        "submission source binding changed",
    )
    _require(
        submission.get("held_gpu_submission_path") == str(held_path)
        and submission.get("held_gpu_submission_sha256")
        == args.expected_held_gpu_submission_sha256,
        "submission held-GPU binding changed",
    )
    _require(
        fingerprint.get("submission_path") == str(submission_path)
        and fingerprint.get("submission_sha256") == args.expected_submission_sha256,
        "release fingerprint submission binding changed",
    )
    _require(
        fingerprint.get("source_contract_path") == str(source_path)
        and fingerprint.get("source_contract_sha256")
        == args.expected_source_contract_sha256,
        "release fingerprint source binding changed",
    )
    _require(
        fingerprint.get("held_gpu_submission_path") == str(held_path)
        and fingerprint.get("held_gpu_submission_sha256")
        == args.expected_held_gpu_submission_sha256,
        "release fingerprint held-GPU binding changed",
    )
    _require(
        held.get("released_at_receipt_time") is False
        and submission.get("released_at_receipt_time") is False
        and fingerprint.get("released_at_fingerprint_time") is False,
        "transaction receipts claim pre-fingerprint release",
    )
    for value in (source, held, submission, fingerprint):
        for key in (
            "scientific_claim_allowed",
            "simulator_efficacy_claim_allowed",
            "infeasibility_claim_allowed",
            "probe_training_authorized",
        ):
            if key in value:
                _require(value[key] is False, f"transaction claim changed: {key}")

    bindings = fingerprint.get("bindings")
    _require(isinstance(bindings, Mapping), "release fingerprint lacks bindings")
    expected_binding_keys = {
        "provisional_gpu_job_id",
        "held_gpu_submission",
        "source_contract",
        "provisional_cpu_job_id",
        "atomic_submission",
        "held_gpu_task_job_record",
        "held_gpu_parent_job_record",
        "cpu_afterany_job_record",
    }
    _require(set(bindings) == expected_binding_keys, "fingerprint binding keys changed")
    for key, binding in bindings.items():
        _require(isinstance(binding, Mapping), f"release binding {key} is invalid")
        path = Path(str(binding.get("path")))
        _regular(path, name=f"release binding {key}")
        digest = binding.get("sha256")
        _require(isinstance(digest, str), f"release binding {key} lacks digest")
        _require_digest(digest, name=f"release binding {key}")
        _require(_sha256(path) == digest, f"release binding changed: {key}")

    paths = _exact_paths(
        source,
        run_root=run_root,
        output=Path(args.output),
        receipt=Path(args.receipt),
    )
    hashes = source.get("repository_file_sha256")
    _require(isinstance(hashes, Mapping) and hashes, "source lacks repository hashes")
    for relative, expected in hashes.items():
        _require(
            isinstance(relative, str)
            and not relative.startswith("/")
            and ".." not in Path(relative).parts,
            "unsafe repository binding",
        )
        _require(isinstance(expected, str), f"binding {relative} lacks digest")
        _require_digest(expected, name=f"repository binding {relative}")
        path = repo_root / relative
        _regular(path, name=f"repository binding {relative}")
        _require(_sha256(path) == expected, f"repository source changed: {relative}")

    artifact = config["artifact_contract"]
    result_schema = repo_root / RESULT_SCHEMA
    receipt_schema = repo_root / RECEIPT_SCHEMA
    result_schema_sha = _sha256(_regular(result_schema, name="TRL result schema"))
    receipt_schema_sha = _sha256(_regular(receipt_schema, name="TRL receipt schema"))
    _require(
        result_schema_sha == artifact["envelope_schema_sha256"],
        "result schema digest changed",
    )
    _require(
        receipt_schema_sha == artifact["publication_receipt_schema_sha256"],
        "receipt schema digest changed",
    )
    _require(hashes.get(RESULT_SCHEMA) == result_schema_sha, "result schema not source-bound")
    _require(
        hashes.get(RECEIPT_SCHEMA) == receipt_schema_sha,
        "receipt schema not source-bound",
    )
    return source, paths, git_commit


def _validate_allocation_tests(path: Path) -> dict[str, Any]:
    _regular(path, name="allocation focused-test log")
    text = path.read_text(encoding="utf-8")
    matches = re.findall(r"^Ran ([1-9][0-9]*) tests? in ", text, flags=re.MULTILINE)
    _require(
        text.count("[openpi-python]") == 1
        and text.count("[libero-python]") == 1
        and text.index("[openpi-python]") < text.index("[libero-python]"),
        "focused-test interpreter groups changed",
    )
    _require(len(matches) == 2, "focused-test suite counts are missing or ambiguous")
    _require(
        len(re.findall(r"^OK$", text, flags=re.MULTILINE)) == 2,
        "allocation tests did not pass",
    )
    _require(
        re.search(r"^(?:FAILED|ERROR)|skipped=", text, flags=re.MULTILINE) is None,
        "allocation tests failed or skipped",
    )
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "tests_run": sum(int(value) for value in matches),
        "skipped": 0,
        "passed": True,
    }


def _validate_gpu_samples(path: Path) -> dict[str, Any]:
    _regular(path, name="GPU memory samples")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    _require(len(rows) >= 2, "GPU memory samples are incomplete")
    _require(
        set(rows[0]) == {"timestamp_ns", "gpu_uuid", "compute_mib", "device_mib"},
        "GPU memory sample columns changed",
    )
    timestamps: list[int] = []
    compute: list[int] = []
    device: list[int] = []
    uuids: set[str] = set()
    for row in rows:
        timestamps.append(int(row["timestamp_ns"]))
        compute.append(int(row["compute_mib"]))
        device.append(int(row["device_mib"]))
        uuids.add(row["gpu_uuid"])
    _require(
        all(right > left for left, right in zip(timestamps, timestamps[1:])),
        "GPU timestamps are not increasing",
    )
    _require(
        len(uuids) == 1 and next(iter(uuids)).startswith("GPU-"),
        "GPU UUID changed",
    )
    _require(all(value >= 0 for value in compute + device), "GPU memory is negative")
    _require(max(device) > 0, "GPU telemetry never observed the model")
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "sample_count": len(rows),
        "gpu_uuid": next(iter(uuids)),
        "compute_high_water_mib": max(compute),
        "device_high_water_mib": max(device),
    }


def _source_bindings(
    config: Mapping[str, Any],
    source: Mapping[str, Any],
    config_path: Path,
) -> dict[str, Any]:
    from crfs_oracle.r05a_reference_trajectory_lift_validation import (
        EXPECTED_SOURCE_BINDINGS,
    )

    frozen = config.get("frozen_source_bindings")
    case = config.get("frozen_case")
    source_frozen = source.get("frozen_bindings")
    _require(
        isinstance(frozen, Mapping) and isinstance(case, Mapping),
        "config source bindings missing",
    )
    _require(isinstance(source_frozen, Mapping), "source frozen bindings missing")
    observed = {
        "case_id": case.get("case_id"),
        "group_id": case.get("group_id"),
        "environment_seed": case.get("environment_seed"),
        "policy_seed": case.get("policy_seed"),
        "source_host": case.get("source_host"),
        "manifest_sha256": frozen.get("manifest", {}).get("sha256"),
        "source_r02_sha256": frozen.get("source_r02", {}).get("sha256"),
        "source_r02_config_sha256": frozen.get("source_r02_config", {}).get("sha256"),
        "checkpoint_model_sha256": frozen.get("checkpoint_model", {}).get("sha256"),
        "checkpoint_config_sha256": frozen.get("checkpoint_config", {}).get("sha256"),
        "normalization_asset_sha256": frozen.get("normalization_asset", {}).get("sha256"),
        "baseline_revision": frozen.get("baseline_revision"),
        "af00a_run_id": frozen.get("af00a_source_run", {}).get("run_id"),
        "af00a_tensor_sha256": frozen.get("af00a_source_run", {}).get(
            "tensor_archive_sha256"
        ),
        "af00a_results_sha256": frozen.get("af00a_source_run", {}).get(
            "results_json_sha256"
        ),
    }
    _require(observed == EXPECTED_SOURCE_BINDINGS, "external source bindings changed")
    source_expected = {
        "actual_config_sha256": _sha256(config_path),
        "manifest_sha256": observed["manifest_sha256"],
        "source_r02_sha256": observed["source_r02_sha256"],
        "source_r02_config_sha256": observed["source_r02_config_sha256"],
        "checkpoint_sha256": observed["checkpoint_model_sha256"],
        "checkpoint_config_sha256": observed["checkpoint_config_sha256"],
        "normalization_asset_sha256": observed["normalization_asset_sha256"],
    }
    for key, expected in source_expected.items():
        _require(
            source_frozen.get(key) == expected,
            f"source frozen binding changed: {key}",
        )

    external_paths = {
        "manifest_sha256": Path(str(frozen["manifest"]["path"])),
        "source_r02_sha256": Path(str(frozen["source_r02"]["path"])),
        "source_r02_config_sha256": Path(str(frozen["source_r02_config"]["path"])),
        "checkpoint_model_sha256": Path(str(frozen["checkpoint_model"]["path"])),
        "checkpoint_config_sha256": Path(str(frozen["checkpoint_config"]["path"])),
        "normalization_asset_sha256": Path(str(frozen["normalization_asset"]["path"])),
        "af00a_tensor_sha256": Path(str(frozen["af00a_source_run"]["tensor_archive_path"])),
        "af00a_results_sha256": Path(str(frozen["af00a_source_run"]["results_json_path"])),
    }
    repo_root = Path(__file__).resolve().parents[1]
    for key, raw_path in external_paths.items():
        path = raw_path if raw_path.is_absolute() else repo_root / raw_path
        _regular(path, name=f"external source {key}")
        _require(_sha256(path) == observed[key], f"external source changed: {key}")
    return observed


def _validate_raw(
    *,
    paths: Mapping[str, Path],
    source: Mapping[str, Any],
    config: Mapping[str, Any],
    config_path: Path,
    legacy_path: Path,
    source_job_id: str,
) -> tuple[dict[str, Any], list[Mapping[str, Any]], dict[str, Any]]:
    payload_path = paths["payload"]
    tensor_path = paths["raw_tensors"]
    ledger_path = paths["request_ledger"]
    payload = _object(payload_path, name="TRL raw payload")
    ledger = _object(ledger_path, name="TRL request ledger")
    _regular(tensor_path, name="TRL tensor archive")
    _require(
        payload.get("schema_version") == "1.0"
        and payload.get("payload_type")
        == "r05a_reference_trajectory_lift_raw_payload"
        and payload.get("status") == "raw_evidence_complete"
        and payload.get("branch") == "finite",
        "raw payload is not the complete finite branch",
    )
    _require(
        payload.get("run_id") == paths["run_root"].name
        and payload.get("case_id") == CASE_ID
        and payload.get("group_id") == GROUP_ID,
        "raw payload identity changed",
    )
    _require(
        payload.get("request_accounting")
        == {"exact": True, "total": 18, "finite_complete": True},
        "raw request accounting changed",
    )
    rows = payload.get("request_ledger")
    _require(
        isinstance(rows, list) and len(rows) == 18,
        "raw payload lacks the exact finite ledger",
    )
    _require(
        ledger.get("schema_version") == "1.0"
        and ledger.get("payload_type")
        == "r05a_reference_trajectory_lift_request_ledger"
        and ledger.get("case_id") == CASE_ID
        and ledger.get("branch") == "finite"
        and ledger.get("exact_policy_request_count") == 18
        and ledger.get("rows") == rows,
        "ledger sidecar differs from finite payload",
    )
    tensor_sha = _sha256(tensor_path)
    _require(
        ledger.get("tensor_archive_sha256") == tensor_sha,
        "ledger tensor binding changed",
    )
    artifacts = payload.get("artifacts")
    _require(
        artifacts
        == {
            "tensor_archive": "trl00a-tensors.npz",
            "tensor_archive_sha256": tensor_sha,
            "request_ledger": "request-ledger.json",
            "request_ledger_sha256": _sha256(ledger_path),
            "results_json_published": False,
        },
        "raw artifact bindings changed",
    )
    raw = payload.get("raw")
    _require(
        isinstance(raw, Mapping)
        and set(raw)
        == {
            "branch",
            "evaluations",
            "replay_envelope",
            "duplicate_exact",
            "ordinary_replay_exact",
        }
        and raw.get("branch") == "finite"
        and isinstance(raw.get("evaluations"), list)
        and len(raw["evaluations"]) == 2
        and raw.get("duplicate_exact") is True
        and raw.get("ordinary_replay_exact") is True,
        "raw payload contains a terminal or nonfinite branch",
    )
    provenance = payload.get("provenance")
    _require(isinstance(provenance, Mapping), "raw payload lacks provenance")
    _require(
        str(provenance.get("slurm_array_job_id")) == source_job_id
        and str(provenance.get("slurm_array_task_id")) == "0"
        and provenance.get("host") == "worker-1",
        "raw Slurm/source host changed",
    )
    expected_boundary = {
        "policy_requests": 18,
        "policy_generated_action_steps_executed": 0,
        "teacher_generated_action_steps_executed": 0,
        "efficacy_rollouts_executed": 0,
        "simulator_efficacy_evaluated": False,
        "geometry_or_planner_queries": 0,
        "training": False,
    }
    _require(
        payload.get("execution_boundary") == expected_boundary,
        "raw execution boundary changed",
    )
    post_checks = payload.get("post_checks")
    _require(
        isinstance(post_checks, Mapping)
        and post_checks
        and all(post_checks.values()),
        "raw post checks failed",
    )
    source_payload = payload.get("source")
    _require(isinstance(source_payload, Mapping), "raw payload lacks source checks")
    source_bindings = _source_bindings(config, source, config_path)
    _require(
        payload.get("source_bindings") == source_bindings,
        "raw payload source bindings changed",
    )
    frozen = config["frozen_source_bindings"]
    r02_path = Path(str(frozen["source_r02"]["path"]))
    af_tensor_path = Path(
        str(frozen["af00a_source_run"]["tensor_archive_path"])
    )
    af_result_path = Path(
        str(frozen["af00a_source_run"]["results_json_path"])
    )
    _require(
        Path(str(source_payload.get("r02_path"))).resolve() == r02_path.resolve()
        and source_payload.get("r02_sha256") == source_bindings["source_r02_sha256"],
        "raw payload R02 binding changed",
    )
    _require(
        Path(str(source_payload.get("af00a_tensor_path"))).resolve()
        == af_tensor_path.resolve()
        and source_payload.get("af00a_tensor_sha256")
        == source_bindings["af00a_tensor_sha256"]
        and Path(str(source_payload.get("af00a_results_path"))).resolve()
        == af_result_path.resolve()
        and source_payload.get("af00a_results_sha256")
        == source_bindings["af00a_results_sha256"],
        "raw payload AF binding changed",
    )
    payload_configs = payload.get("configs")
    _require(isinstance(payload_configs, Mapping), "raw payload lacks config bindings")
    _require(
        Path(str(payload_configs.get("actual_path"))).resolve()
        == config_path.resolve()
        and payload_configs.get("actual_sha256") == _sha256(config_path)
        and Path(str(payload_configs.get("legacy_path"))).resolve()
        == legacy_path.resolve()
        and payload_configs.get("legacy_sha256") == _sha256(legacy_path),
        "raw payload config binding changed",
    )
    _require(
        Path(str(payload_configs.get("repo_root"))).resolve()
        == Path(__file__).resolve().parents[1],
        "raw payload repository root changed",
    )

    r02 = _object(r02_path, name="immutable R02 source")
    r02_delta = r02.get("directions", {}).get("arrays", {}).get("delta_star_model")
    r02_budget = r02.get("directions", {}).get("l2_norms", {}).get(
        "delta_star_model"
    )
    _require(
        source_payload.get("delta_star_model") == r02_delta
        and source_payload.get("source_budget_float64") == r02_budget
        and source_payload.get("source_budget_float64")
        == config["target_contract"]["source_reported_budget_float64"]
        and source_payload.get("source_budget_float32")
        == config["target_contract"]["source_budget_float32"],
        "raw payload Delta-star or budget differs from immutable R02",
    )
    r02_pairing = r02.get("pairing")
    records = payload.get("source_pairing_records")
    _require(
        isinstance(r02_pairing, Mapping) and isinstance(records, Mapping),
        "source pairing records are missing",
    )
    expected_record_keys = {
        "policy_observation_source",
        "policy_observation_current",
        "branch_snapshot_source",
        "branch_snapshot_current",
        "eager_actions_source",
        "eager_actions_before_current",
        "eager_actions_after_current",
        "eager_trace_source",
        "eager_trace_before_current",
        "eager_trace_after_current",
    }
    _require(set(records) == expected_record_keys, "source pairing record keys changed")
    immutable_pairs = {
        "policy_observation_source": r02_pairing.get("policy_observation"),
        "branch_snapshot_source": r02_pairing.get("branch_snapshot"),
        "eager_actions_source": r02_pairing.get("eager_actions"),
        "eager_trace_source": r02_pairing.get("eager_trace"),
    }
    for key, expected in immutable_pairs.items():
        _require(records.get(key) == expected, f"pairing source differs from R02: {key}")
    _require(
        records["policy_observation_current"]
        == records["policy_observation_source"],
        "current policy observation differs from R02",
    )
    _require(
        records["branch_snapshot_current"] == records["branch_snapshot_source"],
        "current branch snapshot differs from R02",
    )
    _require(
        records["eager_actions_before_current"]
        == records["eager_actions_source"]
        == records["eager_actions_after_current"],
        "current eager actions differ from R02 or post reference",
    )
    _require(
        records["eager_trace_before_current"]
        == records["eager_trace_source"]
        == records["eager_trace_after_current"],
        "current eager trace differs from R02 or post reference",
    )
    from crfs_oracle.r05a_reference_trajectory_lift_validation import (
        validate_reference_trajectory_lift_npz,
    )

    validation = validate_reference_trajectory_lift_npz(
        tensor_path,
        rows,
        source_bindings,
    )
    metric_names = ("xyz_max_abs", "xyz_rms", "full_max_abs", "full_rms")

    def payload_evaluation(summary: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "objective": summary["objective"],
            "metrics": [summary["metrics"][name] for name in metric_names],
            "gates": [summary["gates"][name] for name in metric_names],
        }

    _require(
        payload.get("arm_a", {}).get("evaluation")
        == payload_evaluation(validation["arm_a"]),
        "GPU Arm A summary differs from independent validation",
    )
    for payload_key, validation_key in (("budgeted", "budgeted"), ("raw", "raw")):
        evaluations = payload.get(payload_key, {}).get("evaluations")
        expected = payload_evaluation(validation[validation_key])
        _require(
            isinstance(evaluations, list)
            and len(evaluations) == 2
            and evaluations == [expected, expected],
            f"GPU {payload_key} summaries differ from independent validation",
        )
    af_result = _object(af_result_path, name="immutable AF result")
    historical_arm_a = af_result.get("validation", {}).get("arm_a")
    _require(isinstance(historical_arm_a, Mapping), "AF result lacks Arm A validation")
    _require(
        validation["arm_a"]["objective"] == historical_arm_a.get("objective")
        and validation["arm_a"]["metrics"] == historical_arm_a.get("metrics")
        and validation["arm_a"]["full_pass"] is bool(historical_arm_a.get("passed")),
        "independently validated Arm A differs from immutable AF result",
    )
    _require(
        validation.get("branch") == "finite"
        and validation.get("request_count") == 18
        and validation.get("outcome")
        in {
            "same_budget_lift_pass",
            "canonical_budget_bottleneck",
            "canonical_mask_coupling",
            "canonical_lift_negative",
        },
        "independent validator returned an invalid finite outcome",
    )
    return payload, rows, validation


def _publish(args: argparse.Namespace) -> tuple[Path, Path]:
    run_root = Path(args.run_root)
    output = Path(args.output)
    receipt = Path(args.receipt)
    _require(run_root.is_dir() and not run_root.is_symlink(), "run root is invalid")
    _require(output == run_root / CASE_ID / "results.json", "result path changed")
    _require(
        receipt == run_root / "cpu-afterany-validation.json",
        "receipt path changed",
    )
    _require(not output.exists(), "results.json already exists")
    _require(not receipt.exists(), "publication receipt already exists")
    config_path = Path(args.config)
    legacy_path = Path(args.legacy_config)
    config = _object(config_path, name="released TRL config")
    _object(legacy_path, name="legacy R05A config")
    release = _validate_release(config, run_root.name)
    repo_root = Path(__file__).resolve().parents[1]
    source, paths, git_commit = _validate_transaction(
        args,
        run_root=run_root,
        config=config,
        repo_root=repo_root,
    )
    frozen = source.get("frozen_bindings")
    _require(
        isinstance(frozen, Mapping)
        and frozen.get("legacy_config_sha256") == _sha256(legacy_path),
        "legacy config source binding changed",
    )
    payload, rows, validation = _validate_raw(
        paths=paths,
        source=source,
        config=config,
        config_path=config_path,
        legacy_path=legacy_path,
        source_job_id=str(args.source_job_id),
    )
    allocation_tests = _validate_allocation_tests(paths["allocation_tests_log"])
    from crfs_oracle.r05a_full_lifetime_telemetry import (
        parse_full_lifetime_telemetry,
    )

    host_telemetry = parse_full_lifetime_telemetry(
        paths["host_telemetry"], expected_job_id=str(args.source_job_id)
    )
    _require(
        host_telemetry.get("contract_passed") is True,
        "host telemetry contract failed",
    )
    gpu_telemetry = _validate_gpu_samples(paths["gpu_samples"])
    measured = {
        "allocation_tests": allocation_tests,
        "host": host_telemetry,
        "gpu": gpu_telemetry,
    }
    raw_evidence = {
        "payload_path": str(paths["payload"]),
        "payload_sha256": _sha256(paths["payload"]),
        "tensor_path": str(paths["raw_tensors"]),
        "tensor_sha256": _sha256(paths["raw_tensors"]),
        "request_ledger_path": str(paths["request_ledger"]),
        "request_ledger_sha256": _sha256(paths["request_ledger"]),
        "request_count": len(rows),
        "source_contract_path": str(Path(args.source_contract)),
        "source_contract_sha256": args.expected_source_contract_sha256,
        "held_gpu_submission_path": str(Path(args.held_gpu_submission)),
        "held_gpu_submission_sha256": args.expected_held_gpu_submission_sha256,
        "submission_path": str(Path(args.submission)),
        "submission_sha256": args.expected_submission_sha256,
        "release_fingerprint_path": str(Path(args.release_fingerprint)),
        "release_fingerprint_sha256": args.expected_release_fingerprint_sha256,
    }
    claims = {
        "scientific_claim_allowed": False,
        "simulator_efficacy_claim_allowed": False,
        "collision_or_progress_claim_allowed": False,
        "infeasibility_claim_allowed": False,
        "raw_authority_is_minimum_required_claim_allowed": False,
        "probe_training_authorized": False,
        "automatic_next_gate_authorized": False,
    }
    result = {
        "schema_version": "1.0",
        "artifact_type": "r05a_reference_trajectory_lift_canary_result",
        "gate": "R05A",
        "experiment_identity": "TRL-00A",
        "run_id": run_root.name,
        "case_id": CASE_ID,
        "group_id": GROUP_ID,
        "status": validation["outcome"],
        "branch": "finite",
        "source_job_id": f"{args.source_job_id}_0",
        "publisher_job_id": str(args.publisher_job_id),
        "execution_identity": {
            "git_commit": git_commit,
            "accepted_implementation_commit": release[
                "accepted_implementation_commit"
            ],
            "source_node": "worker-1",
            "gpu_slurm_array_job_id": str(args.source_job_id),
            "gpu_slurm_array_task_id": 0,
            "exact_gpu_task_id": f"{args.source_job_id}_0",
            "cpu_afterany_job_id": str(args.publisher_job_id),
            "dependency": f"afterany:{args.source_job_id}",
        },
        "raw_evidence": raw_evidence,
        "measured_telemetry": measured,
        "validation": validation,
        "execution_boundary": payload["execution_boundary"],
        "claims": claims,
    }

    import jsonschema

    result_schema_path = repo_root / RESULT_SCHEMA
    receipt_schema_path = repo_root / RECEIPT_SCHEMA
    result_schema = _object(result_schema_path, name="TRL result schema")
    receipt_schema = _object(receipt_schema_path, name="TRL receipt schema")
    jsonschema.Draft202012Validator.check_schema(result_schema)
    jsonschema.Draft202012Validator.check_schema(receipt_schema)
    jsonschema.Draft202012Validator(result_schema).validate(result)

    candidate = paths["hidden_candidate"]
    _require(not candidate.exists(), "hidden result candidate already exists")
    _atomic_json(candidate, result)
    result_sha = _sha256(candidate)
    receipt_value = {
        "schema_version": "1.0",
        "artifact_role": "r05a_reference_trajectory_lift_canary_cpu_publication",
        "status": "published",
        "outcome": validation["outcome"],
        "run_id": run_root.name,
        "case_id": CASE_ID,
        "git_commit": git_commit,
        "source_node": "worker-1",
        "source_job_id": str(args.source_job_id),
        "source_task_id": f"{args.source_job_id}_0",
        "source_job_state": "COMPLETED",
        "source_exit_code": "0:0",
        "publisher_job_id": str(args.publisher_job_id),
        "dependency": f"afterany:{args.source_job_id}",
        "result_path": str(output),
        "result_sha256": result_sha,
        "published": True,
        "passed": True,
        "errors": [],
        "evidence": raw_evidence,
        "measured_telemetry": measured,
        **claims,
    }
    jsonschema.Draft202012Validator(receipt_schema).validate(receipt_value)
    published = False
    try:
        os.replace(candidate, output)
        published = True
        _require(_sha256(output) == result_sha, "published result digest changed")
        _atomic_json(receipt, receipt_value)
    except BaseException:
        candidate.unlink(missing_ok=True)
        if published:
            output.unlink(missing_ok=True)
        raise
    return output, receipt


def _failure_receipt(args: argparse.Namespace, error: BaseException) -> None:
    run_root = Path(args.run_root)
    receipt = Path(args.receipt)
    output = Path(args.output)
    if receipt.exists():
        return
    value = {
        "schema_version": "1.0",
        "artifact_role": "r05a_reference_trajectory_lift_canary_cpu_publication",
        "status": "apparatus_inconclusive",
        "outcome": "apparatus_inconclusive",
        "run_id": run_root.name,
        "case_id": CASE_ID,
        "git_commit": None,
        "source_node": "worker-1",
        "source_job_id": str(args.source_job_id),
        "source_task_id": f"{args.source_job_id}_0",
        "source_job_state": "COMPLETED",
        "source_exit_code": "0:0",
        "publisher_job_id": str(args.publisher_job_id),
        "dependency": f"afterany:{args.source_job_id}",
        "failure_stage": "independent_cpu_publication",
        "result_path": str(output),
        "result_sha256": None,
        "published": False,
        "passed": False,
        "errors": [f"{type(error).__name__}: {error}"],
        "scientific_claim_allowed": False,
        "simulator_efficacy_claim_allowed": False,
        "collision_or_progress_claim_allowed": False,
        "infeasibility_claim_allowed": False,
        "raw_authority_is_minimum_required_claim_allowed": False,
        "probe_training_authorized": False,
        "automatic_next_gate_authorized": False,
    }
    _atomic_json(receipt, value)


def main() -> int:
    args = _parser().parse_args()
    try:
        output, receipt = _publish(args)
    except BaseException as error:
        Path(args.output).parent.joinpath(".results.candidate.json").unlink(
            missing_ok=True
        )
        _failure_receipt(args, error)
        print(f"TRL-00A apparatus-inconclusive: {type(error).__name__}: {error}")
        return 1
    print(f"published {output}")
    print(f"receipt {receipt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
