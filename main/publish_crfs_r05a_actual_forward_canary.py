#!/usr/bin/env python3
"""Independently validate and publish one terminal AF-00A artifact."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
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
from crfs_oracle.r05a_canary import REGISTERED_XYZ_SCALE, _source_delta
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
RECOVERY_RUN_ID = "r05a-actual-forward-cem-canary-20260716c"
RECOVERY_SOURCE_COMMIT = "bd14f97eeffafd20525454db4d7a52614e1146c4"
RECOVERY_SOURCE_JOB_ID = "28281"
RECOVERY_ORIGINAL_PUBLISHER_JOB_ID = "28282"


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
    parser.add_argument("--recovery-config")
    parser.add_argument("--original-publisher-job-id")
    parser.add_argument("--original-receipt")
    parser.add_argument("--expected-original-receipt-sha256")
    parser.add_argument("--recovery-source-contract")
    parser.add_argument("--expected-recovery-source-contract-sha256")
    parser.add_argument("--recovery-submission")
    parser.add_argument("--expected-recovery-submission-sha256")
    parser.add_argument("--recovery-release-fingerprint")
    parser.add_argument("--expected-recovery-release-fingerprint-sha256")
    return parser


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, sort_keys=True)
        handle.write("\n")
        temporary = handle.name
    os.replace(temporary, path)


def _commit_publication(
    candidate: Path,
    output: Path,
    receipt: Path,
    receipt_value: dict[str, Any],
    *,
    expected_output_sha256: str,
) -> None:
    """Commit result+receipt or remove only the result created by this call."""

    os.replace(candidate, output)
    try:
        if file_sha256(output) != expected_output_sha256:
            raise ValueError("AF-00A published result digest changed after rename")
        _atomic_json(receipt, receipt_value)
    except Exception:
        output.unlink(missing_ok=True)
        raise


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


def _git_blob_sha256(repo_root: Path, commit: str, relative: str) -> str:
    """Hash one source-bound blob without substituting the recovery checkout."""

    completed = subprocess.run(
        ["git", "-C", str(repo_root), "cat-file", "blob", f"{commit}:{relative}"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        raise ValueError(f"AF-00A source commit blob is unavailable: {relative}")
    return hashlib.sha256(completed.stdout).hexdigest()


def _validate_source_budget(
    payload_source: dict[str, Any],
    raw_r02: dict[str, Any],
    config: dict[str, Any],
) -> None:
    """Validate the producer's recomputed binary64 norm and frozen float32 budget."""

    _, _, recomputed_budget64 = _source_delta(raw_r02)
    reported_budget64 = raw_r02.get("directions", {}).get("l2_norms", {}).get(
        "delta_star_model"
    )
    frozen_reported64 = config.get("target_contract", {}).get(
        "source_reported_budget_float64"
    )
    if (
        isinstance(reported_budget64, bool)
        or not isinstance(reported_budget64, (int, float))
        or not math.isfinite(float(reported_budget64))
        or reported_budget64 != frozen_reported64
    ):
        raise ValueError("AF-00A R02 reported source budget changed")
    payload_budget64 = payload_source.get("source_budget_float64")
    if (
        isinstance(payload_budget64, bool)
        or not isinstance(payload_budget64, (int, float))
        or not math.isfinite(float(payload_budget64))
        or float(payload_budget64) != recomputed_budget64
    ):
        raise ValueError("AF-00A raw payload recomputed source budget changed")
    payload_budget32 = np.float32(payload_source.get("source_budget_float32"))
    released_budget32 = np.float32(
        config.get("target_contract", {}).get("source_budget_float32")
    )
    if (
        payload_budget32.tobytes() != released_budget32.tobytes()
        or np.float32(reported_budget64).tobytes() != released_budget32.tobytes()
        or np.float32(recomputed_budget64).tobytes() != released_budget32.tobytes()
    ):
        raise ValueError("AF-00A raw payload float32 budget changed")


def _recovery_mode(args: argparse.Namespace) -> bool:
    names = (
        "recovery_config",
        "original_publisher_job_id",
        "original_receipt",
        "expected_original_receipt_sha256",
        "recovery_source_contract",
        "expected_recovery_source_contract_sha256",
        "recovery_submission",
        "expected_recovery_submission_sha256",
        "recovery_release_fingerprint",
        "expected_recovery_release_fingerprint_sha256",
    )
    supplied = [getattr(args, name) is not None for name in names]
    if any(supplied) and not all(supplied):
        raise ValueError("AF-00A recovery arguments must be supplied as one complete set")
    return all(supplied)


def _validate_recovery_transaction(
    *,
    args: argparse.Namespace,
    repo_root: Path,
    run_root: Path,
    output: Path,
    receipt: Path,
    source_git_commit: str,
) -> dict[str, Any]:
    """Validate the fresh CPU-only recovery chain without mutating run-C evidence."""

    recovery_config_path = (
        repo_root
        / "configs"
        / "experiments"
        / "r05a_actual_forward_recovery_20260716c.json"
    )
    original_receipt_path = run_root / "cpu-afterany-validation.json"
    recovery_source_contract_path = run_root / "republication-source-contract.json"
    recovery_submission_path = run_root / "republication-submission.json"
    recovery_fingerprint_path = (
        run_root / "republication-final-pre-release-fingerprint.json"
    )
    expected_paths = {
        "recovery config": (Path(args.recovery_config), recovery_config_path),
        "original receipt": (Path(args.original_receipt), original_receipt_path),
        "recovery source contract": (
            Path(args.recovery_source_contract),
            recovery_source_contract_path,
        ),
        "recovery submission": (Path(args.recovery_submission), recovery_submission_path),
        "recovery fingerprint": (
            Path(args.recovery_release_fingerprint),
            recovery_fingerprint_path,
        ),
    }
    for label, (supplied, expected) in expected_paths.items():
        if supplied != expected:
            raise ValueError(f"AF-00A {label} path is not the frozen recovery path")
    for value, label in (
        (args.expected_original_receipt_sha256, "original failure receipt"),
        (args.expected_recovery_source_contract_sha256, "recovery source contract"),
        (args.expected_recovery_submission_sha256, "recovery submission"),
        (args.expected_recovery_release_fingerprint_sha256, "recovery fingerprint"),
    ):
        _require_sha256(value, name=f"expected AF-00A {label} digest")
    if run_root.name != RECOVERY_RUN_ID or source_git_commit != RECOVERY_SOURCE_COMMIT:
        raise ValueError("AF-00A recovery source run/commit changed")
    if (
        args.source_job_id != RECOVERY_SOURCE_JOB_ID
        or args.original_publisher_job_id != RECOVERY_ORIGINAL_PUBLISHER_JOB_ID
        or args.publisher_job_id == args.original_publisher_job_id
    ):
        raise ValueError("AF-00A recovery job identities changed or were conflated")
    if receipt != run_root / "cpu-republication-validation.json":
        raise ValueError("AF-00A recovery receipt path changed")

    original_receipt = _object(original_receipt_path, name="AF-00A original failure receipt")
    if _digest(original_receipt_path, name="AF-00A original failure receipt") != args.expected_original_receipt_sha256:
        raise ValueError("AF-00A original failure receipt digest changed")
    expected_original_failure = {
        "artifact_role": "r05a_actual_forward_canary_cpu_publication",
        "source_job_id": RECOVERY_SOURCE_JOB_ID,
        "source_task_id": f"{RECOVERY_SOURCE_JOB_ID}_0",
        "publisher_job_id": RECOVERY_ORIGINAL_PUBLISHER_JOB_ID,
        "result_path": str(output),
        "result_sha256": None,
        "published": False,
        "passed": False,
        "errors": ["AF-00A raw payload source budget changed"],
    }
    for key, expected in expected_original_failure.items():
        if original_receipt.get(key) != expected:
            raise ValueError(f"AF-00A original failure receipt changed: {key}")

    recovery_source_contract = _object(
        recovery_source_contract_path, name="AF-00A recovery source contract"
    )
    recovery_source_sha = _digest(
        recovery_source_contract_path, name="AF-00A recovery source contract"
    )
    if recovery_source_sha != args.expected_recovery_source_contract_sha256:
        raise ValueError("AF-00A recovery source contract digest changed")
    recovery_git_commit = recovery_source_contract.get("recovery_git_commit")
    if not isinstance(recovery_git_commit, str) or re.fullmatch(r"[0-9a-f]{40}", recovery_git_commit) is None:
        raise ValueError("AF-00A recovery release commit is invalid")
    recovery_config = _object(recovery_config_path, name="AF-00A recovery config")
    release = recovery_config.get("execution_release")
    if (
        recovery_config.get("ready_to_run") is not True
        or recovery_config.get("blocked_on") != []
        or not isinstance(release, dict)
    ):
        raise ValueError("AF-00A recovery config is not released")
    accepted_implementation_commit = release.get("accepted_implementation_commit")
    if not isinstance(accepted_implementation_commit, str) or re.fullmatch(
        r"[0-9a-f]{40}", accepted_implementation_commit
    ) is None:
        raise ValueError("AF-00A recovery implementation commit is invalid")
    if recovery_config.get("source_binding") != {
        "run_id": RECOVERY_RUN_ID,
        "source_git_commit": RECOVERY_SOURCE_COMMIT,
        "source_job_id": RECOVERY_SOURCE_JOB_ID,
        "source_task_id": f"{RECOVERY_SOURCE_JOB_ID}_0",
        "source_state": "COMPLETED",
        "source_exit_code": "0:0",
        "source_node": "worker-1",
        "original_publisher_job_id": RECOVERY_ORIGINAL_PUBLISHER_JOB_ID,
        "original_publisher_state": "FAILED",
        "original_publisher_exit_code": "1:0",
    }:
        raise ValueError("AF-00A recovery source binding changed")
    resources = {
        "partition": "main",
        "account": "normal",
        "qos": "normal",
        "cpus_per_task": 2,
        "host_memory_mib": 8192,
        "time_limit": "00:15:00",
        "gpus": 0,
        "requeue": False,
        "dependency": f"afterany:{RECOVERY_ORIGINAL_PUBLISHER_JOB_ID}",
    }
    if recovery_config.get("resources") != resources:
        raise ValueError("AF-00A recovery resources changed")
    if recovery_config.get("budget_contract") != {
        "reported_float64": 3.6398398429065115,
        "recomputed_float64": 3.639839842906512,
        "float32": 3.6398398876190186,
        "float32_hex": "0x4068f323",
        "existing_source_consistency_tolerance": 1e-12,
        "tolerance_change_allowed": False,
    }:
        raise ValueError("AF-00A recovery budget contract changed")
    if release.get("single_submission") is not True or release.get(
        "resources"
    ) != resources or release.get("automatic_resubmission_allowed") is not False or release.get(
        "automatic_next_experiment_allowed"
    ) is not False:
        raise ValueError("AF-00A recovery execution release changed")
    decision_artifact = release.get("decision_artifact")
    allowed_release_diff_paths = release.get("allowed_release_diff_paths")
    if (
        release.get("release_only_parent_required") is not True
        or not isinstance(decision_artifact, str)
        or Path(decision_artifact).is_absolute()
        or not (repo_root / decision_artifact).is_file()
        or allowed_release_diff_paths
        != [
            "configs/experiments/r05a_actual_forward_recovery_20260716c.json",
            decision_artifact,
        ]
    ):
        raise ValueError("AF-00A recovery release scope changed")
    if recovery_config.get("claims") != {
        "scientific_claim_allowed": False,
        "simulator_efficacy_claim_allowed": False,
        "infeasibility_claim_allowed": False,
        "probe_training_authorized": False,
        "automatic_next_gate_authorized": False,
    }:
        raise ValueError("AF-00A recovery claim boundary changed")
    immutable_artifacts = recovery_config.get("immutable_artifact_sha256")
    if not isinstance(immutable_artifacts, dict) or not immutable_artifacts:
        raise ValueError("AF-00A recovery config lacks immutable artifact hashes")
    for relative, digest in immutable_artifacts.items():
        relative_path = Path(relative)
        if (
            not isinstance(relative, str)
            or not isinstance(digest, str)
            or relative_path.is_absolute()
            or any(part in {"", ".", ".."} for part in relative_path.parts)
        ):
            raise ValueError("AF-00A immutable recovery artifact binding is malformed")
        if _digest(run_root / relative_path, name=f"AF-00A immutable {relative}") != digest:
            raise ValueError(f"AF-00A immutable run-C artifact changed: {relative}")

    recovery_submission = _object(
        recovery_submission_path, name="AF-00A recovery submission"
    )
    recovery_submission_sha = _digest(
        recovery_submission_path, name="AF-00A recovery submission"
    )
    if recovery_submission_sha != args.expected_recovery_submission_sha256:
        raise ValueError("AF-00A recovery submission digest changed")
    recovery_fingerprint = _object(
        recovery_fingerprint_path, name="AF-00A recovery fingerprint"
    )
    recovery_fingerprint_sha = _digest(
        recovery_fingerprint_path, name="AF-00A recovery fingerprint"
    )
    if recovery_fingerprint_sha != args.expected_recovery_release_fingerprint_sha256:
        raise ValueError("AF-00A recovery fingerprint digest changed")

    expected_identity = {
        "run_id": RECOVERY_RUN_ID,
        "source_git_commit": RECOVERY_SOURCE_COMMIT,
        "recovery_git_commit": recovery_git_commit,
        "source_job_id": RECOVERY_SOURCE_JOB_ID,
        "source_task_id": f"{RECOVERY_SOURCE_JOB_ID}_0",
        "original_publisher_job_id": RECOVERY_ORIGINAL_PUBLISHER_JOB_ID,
        "recovery_publisher_job_id": str(args.publisher_job_id),
        "dependency": f"afterany:{RECOVERY_ORIGINAL_PUBLISHER_JOB_ID}",
    }
    for label, value, role in (
        (
            "source contract",
            recovery_source_contract,
            "r05a_actual_forward_republication_source_contract",
        ),
        (
            "submission",
            recovery_submission,
            "r05a_actual_forward_republication_submission",
        ),
        (
            "fingerprint",
            recovery_fingerprint,
            "r05a_actual_forward_republication_final_pre_release_fingerprint",
        ),
    ):
        if value.get("artifact_role") != role:
            raise ValueError(f"AF-00A recovery {label} role changed")
        for key, expected in expected_identity.items():
            if value.get(key) != expected:
                raise ValueError(f"AF-00A recovery {label} changed: {key}")
        _require_false_claims(value, name=f"AF-00A recovery {label}")
    if recovery_source_contract.get("resources") != resources:
        raise ValueError("AF-00A recovery source-contract resources changed")
    if recovery_source_contract.get("original_receipt_sha256") != args.expected_original_receipt_sha256:
        raise ValueError("AF-00A recovery contract does not bind the original receipt")
    if recovery_submission.get("recovery_source_contract_sha256") != recovery_source_sha:
        raise ValueError("AF-00A recovery submission source binding changed")
    if recovery_fingerprint.get("recovery_source_contract_sha256") != recovery_source_sha:
        raise ValueError("AF-00A recovery fingerprint source binding changed")
    if recovery_fingerprint.get("recovery_submission_sha256") != recovery_submission_sha:
        raise ValueError("AF-00A recovery fingerprint submission binding changed")
    if (
        recovery_submission.get("expected_result") != str(output)
        or recovery_submission.get("expected_recovery_receipt") != str(receipt)
        or recovery_fingerprint.get("expected_result") != str(output)
        or recovery_fingerprint.get("expected_recovery_receipt") != str(receipt)
    ):
        raise ValueError("AF-00A recovery publication paths changed")
    expected_binding_paths = {
        "provisional_cpu_job_id": run_root
        / "republication-provisional-cpu-job-id.json",
        "held_cpu_job_record": run_root / "republication-held-cpu-job-record.txt",
        "source_terminal_job_record": run_root
        / "republication-source-terminal-job-record.txt",
        "original_publisher_terminal_job_record": run_root
        / "republication-original-publisher-terminal-job-record.txt",
    }
    fingerprint_bindings = recovery_fingerprint.get("bindings")
    if not isinstance(fingerprint_bindings, dict) or set(
        fingerprint_bindings
    ) != set(expected_binding_paths):
        raise ValueError("AF-00A recovery fingerprint binding set changed")
    for key, expected_path in expected_binding_paths.items():
        binding = fingerprint_bindings.get(key)
        if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
            raise ValueError(f"AF-00A recovery fingerprint binding is malformed: {key}")
        if binding.get("path") != str(expected_path):
            raise ValueError(f"AF-00A recovery fingerprint path changed: {key}")
        digest = _digest(expected_path, name=f"AF-00A recovery {key}")
        if binding.get("sha256") != digest:
            raise ValueError(f"AF-00A recovery fingerprint digest changed: {key}")
    provisional = _object(
        expected_binding_paths["provisional_cpu_job_id"],
        name="AF-00A recovery provisional CPU receipt",
    )
    for key, expected in {
        "artifact_role": "r05a_actual_forward_republication_provisional_cpu_job_id",
        **expected_identity,
    }.items():
        if provisional.get(key) != expected:
            raise ValueError(f"AF-00A recovery provisional CPU receipt changed: {key}")
    if provisional.get("status") != "sbatch_returned_held_numeric_id_before_inspection":
        raise ValueError("AF-00A recovery provisional CPU receipt status changed")
    _require_false_claims(provisional, name="AF-00A recovery provisional CPU receipt")
    held_record = expected_binding_paths["held_cpu_job_record"].read_text(
        encoding="utf-8"
    )
    for token in (
        f"JobId={args.publisher_job_id}",
        "JobState=PENDING",
        "Reason=JobHeldUser",
        "Partition=main",
        "Account=normal",
        "QOS=normal",
        "TimeLimit=00:15:00",
        "Requeue=0",
        "NumNodes=1",
        "NumCPUs=2",
        "CPUs/Task=2",
    ):
        if f" {token} " not in f" {held_record.strip()} ":
            raise ValueError(f"AF-00A recovery held CPU record changed: {token}")
    if "gres/gpu" in held_record or "ArrayJobId=" in held_record:
        raise ValueError("AF-00A recovery held CPU record gained GPU/array resources")
    req_tres_matches = re.findall(r"(?:^| )ReqTRES=([^ ]+)(?: |$)", held_record)
    if len(req_tres_matches) != 1:
        raise ValueError("AF-00A recovery held CPU record lacks one ReqTRES")
    req_tres: dict[str, str] = {}
    for token in req_tres_matches[0].split(","):
        if token.count("=") != 1:
            raise ValueError("AF-00A recovery held CPU ReqTRES is malformed")
        key, value = token.split("=", 1)
        if key in req_tres:
            raise ValueError("AF-00A recovery held CPU ReqTRES has duplicates")
        req_tres[key] = value
    req_tres.pop("billing", None)
    if (
        set(req_tres) != {"cpu", "mem", "node"}
        or req_tres.get("cpu") != "2"
        or req_tres.get("mem") not in {"8G", "8192M"}
        or req_tres.get("node") != "1"
    ):
        raise ValueError("AF-00A recovery held CPU ReqTRES changed")
    source_terminal = expected_binding_paths["source_terminal_job_record"].read_text(
        encoding="utf-8"
    )
    for token in (
        "ArrayJobId=28281",
        "ArrayTaskId=0",
        "JobState=COMPLETED",
        "ExitCode=0:0",
        "NodeList=worker-1",
    ):
        if f" {token} " not in f" {source_terminal.strip()} ":
            raise ValueError(f"AF-00A recovery source terminal record changed: {token}")
    original_terminal = expected_binding_paths[
        "original_publisher_terminal_job_record"
    ].read_text(encoding="utf-8")
    for token in ("JobId=28282", "JobState=FAILED", "ExitCode=1:0"):
        if f" {token} " not in f" {original_terminal.strip()} ":
            raise ValueError(f"AF-00A original publisher terminal record changed: {token}")
    terminal_job_records = recovery_source_contract.get("terminal_job_records")
    if not isinstance(terminal_job_records, dict) or set(terminal_job_records) != {
        "source",
        "original_publisher",
    }:
        raise ValueError("AF-00A recovery terminal-job binding set changed")
    for source_key, fingerprint_key in (
        ("source", "source_terminal_job_record"),
        ("original_publisher", "original_publisher_terminal_job_record"),
    ):
        if terminal_job_records.get(source_key) != fingerprint_bindings.get(
            fingerprint_key
        ):
            raise ValueError(f"AF-00A terminal-job bindings disagree: {source_key}")

    repository_hashes = recovery_source_contract.get("repository_file_sha256")
    if not isinstance(repository_hashes, dict) or not repository_hashes:
        raise ValueError("AF-00A recovery contract lacks repository hashes")
    expected_repository_paths = recovery_config.get("recovery_repository_paths")
    if (
        not isinstance(expected_repository_paths, list)
        or not expected_repository_paths
        or len(expected_repository_paths) != len(set(expected_repository_paths))
        or set(repository_hashes) != set(expected_repository_paths) | {decision_artifact}
    ):
        raise ValueError("AF-00A recovery repository binding set changed")
    for relative, digest in repository_hashes.items():
        relative_path = Path(relative)
        if (
            not isinstance(relative, str)
            or not isinstance(digest, str)
            or relative_path.is_absolute()
            or relative in {"", "."}
            or any(part in {"", ".", ".."} for part in relative_path.parts)
        ):
            raise ValueError("AF-00A recovery repository binding is malformed")
        _require_sha256(digest, name=f"recovery repository digest for {relative}")
        bound = repo_root / relative
        if not bound.is_file() or bound.is_symlink() or file_sha256(bound) != digest:
            raise ValueError(f"AF-00A recovery source changed: {relative}")
    current_commit = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.strip()
    if current_commit != recovery_git_commit:
        raise ValueError("AF-00A recovery checkout differs from its release")
    release_parents = subprocess.run(
        ["git", "-C", str(repo_root), "rev-list", "--parents", "-n", "1", current_commit],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.strip()
    if release_parents != f"{current_commit} {accepted_implementation_commit}":
        raise ValueError("AF-00A recovery release is not the direct implementation child")
    return {
        "config": recovery_config,
        "source_contract_path": str(recovery_source_contract_path),
        "source_contract_sha256": recovery_source_sha,
        "submission_path": str(recovery_submission_path),
        "submission_sha256": recovery_submission_sha,
        "release_fingerprint_path": str(recovery_fingerprint_path),
        "release_fingerprint_sha256": recovery_fingerprint_sha,
        "original_receipt_path": str(original_receipt_path),
        "original_receipt_sha256": args.expected_original_receipt_sha256,
        "source_git_commit": source_git_commit,
        "recovery_git_commit": recovery_git_commit,
        "resources": resources,
    }


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
        recovery_mode = _recovery_mode(args)
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
        original_receipt_path = run_root / "cpu-afterany-validation.json"
        expected_receipt = (
            run_root / "cpu-republication-validation.json"
            if recovery_mode
            else original_receipt_path
        )
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
        transaction_publisher_job_id = (
            str(args.original_publisher_job_id)
            if recovery_mode
            else str(args.publisher_job_id)
        )
        transaction_dependency = f"afterany:{args.source_job_id}"
        publication_dependency = (
            f"afterany:{args.original_publisher_job_id}"
            if recovery_mode
            else transaction_dependency
        )
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
            "validation_receipt": str(original_receipt_path),
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
            "cpu_afterany_job_id": transaction_publisher_job_id,
            "dependency": transaction_dependency,
            "source_contract_path": str(source_contract_path),
            "source_contract_sha256": args.expected_source_contract_sha256,
            "held_gpu_submission_path": str(held_gpu_path),
            "held_gpu_submission_sha256": held_gpu_sha256,
            "release_fingerprint_path": str(release_fingerprint_path),
            "expected_result": str(output),
            "expected_validation_receipt": str(original_receipt_path),
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
            "cpu_afterany_job_id": transaction_publisher_job_id,
            "dependency": transaction_dependency,
            "source_contract_path": str(source_contract_path),
            "source_contract_sha256": args.expected_source_contract_sha256,
            "held_gpu_submission_path": str(held_gpu_path),
            "held_gpu_submission_sha256": held_gpu_sha256,
            "submission_path": str(submission_path),
            "submission_sha256": submission_sha256,
            "expected_result": str(output),
            "expected_validation_receipt": str(original_receipt_path),
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
            observed_digest = (
                _git_blob_sha256(repo_root, git_commit, relative)
                if recovery_mode
                else (
                    file_sha256(bound)
                    if bound.is_file() and not bound.is_symlink()
                    else None
                )
            )
            if observed_digest != digest:
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
        recovery_evidence = (
            _validate_recovery_transaction(
                args=args,
                repo_root=repo_root,
                run_root=run_root,
                output=output,
                receipt=receipt,
                source_git_commit=git_commit,
            )
            if recovery_mode
            else None
        )

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
        _validate_source_budget(payload_source, raw_r02, config)
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
                "dependency": publication_dependency,
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
        if recovery_mode:
            assert recovery_evidence is not None
            result_value["recovery_identity"] = {
                "source_git_commit": git_commit,
                "recovery_git_commit": recovery_evidence["recovery_git_commit"],
                "original_publisher_job_id": str(args.original_publisher_job_id),
                "original_publisher_state": "FAILED",
                "original_publisher_exit_code": "1:0",
                "recovery_publisher_job_id": str(args.publisher_job_id),
                "dependency": publication_dependency,
                "original_receipt_path": recovery_evidence["original_receipt_path"],
                "original_receipt_sha256": recovery_evidence[
                    "original_receipt_sha256"
                ],
                "recovery_source_contract_path": recovery_evidence[
                    "source_contract_path"
                ],
                "recovery_source_contract_sha256": recovery_evidence[
                    "source_contract_sha256"
                ],
                "recovery_submission_path": recovery_evidence["submission_path"],
                "recovery_submission_sha256": recovery_evidence[
                    "submission_sha256"
                ],
                "recovery_release_fingerprint_path": recovery_evidence[
                    "release_fingerprint_path"
                ],
                "recovery_release_fingerprint_sha256": recovery_evidence[
                    "release_fingerprint_sha256"
                ],
            }
        candidate = output.parent / ".results.candidate.json"
        if output.exists() or candidate.exists() or receipt.exists():
            raise ValueError("AF-00A publication target is not fresh")
        _atomic_json(candidate, result_value)
        # Re-read the candidate before the sole atomic publication rename.
        candidate_value = _object(candidate, name="AF-00A result candidate")
        if candidate_value != result_value:
            raise ValueError("AF-00A result candidate round trip changed")
        schema_relative = (
            "schemas/r05a-actual-forward-recovery-envelope.schema.json"
            if recovery_mode
            else "schemas/r05a-actual-forward-canary-envelope.schema.json"
        )
        schema_path = repo_root / schema_relative
        if recovery_mode:
            receipt_schema_relative = (
                "schemas/r05a-actual-forward-recovery-publication-receipt.schema.json"
            )
        else:
            receipt_schema_relative = (
                "schemas/r05a-actual-forward-canary-publication-receipt.schema.json"
            )
        receipt_schema_path = repo_root / receipt_schema_relative
        artifact_contract = config.get("artifact_contract")
        if not isinstance(artifact_contract, dict):
            raise ValueError("AF-00A config lacks its artifact contract")
        if (
            artifact_contract.get("raw_payload_name") != payload_path.name
            or artifact_contract.get("tensor_archive_name") != tensors_path.name
            or artifact_contract.get("published_result_name") != output.name
            or artifact_contract.get("publication_receipt_name")
            != original_receipt_path.name
            or (
                not recovery_mode
                and artifact_contract.get("envelope_schema_path") != schema_relative
            )
            or (
                not recovery_mode
                and artifact_contract.get("envelope_schema_sha256")
                != file_sha256(schema_path)
            )
            or (
                not recovery_mode
                and artifact_contract.get("publication_receipt_schema_path")
                != receipt_schema_relative
            )
            or (
                not recovery_mode
                and artifact_contract.get("publication_receipt_schema_sha256")
                != file_sha256(receipt_schema_path)
            )
        ):
            raise ValueError("AF-00A publication schemas differ from released config")
        if recovery_mode:
            assert recovery_evidence is not None
            recovery_repository_hashes = _object(
                Path(recovery_evidence["source_contract_path"]),
                name="AF-00A recovery source contract",
            ).get("repository_file_sha256", {})
            if recovery_repository_hashes.get(schema_relative) != file_sha256(
                schema_path
            ):
                raise ValueError("AF-00A recovery envelope schema differs from its binding")
            if recovery_repository_hashes.get(receipt_schema_relative) != file_sha256(
                receipt_schema_path
            ):
                raise ValueError("AF-00A recovery receipt schema differs from its binding")
        else:
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
        if recovery_mode:
            base_schema_path = (
                repo_root
                / "schemas"
                / "r05a-actual-forward-canary-envelope.schema.json"
            )
            base_schema = _object(
                base_schema_path, name="AF-00A original result envelope schema"
            )
            base_candidate = dict(candidate_value)
            base_candidate.pop("recovery_identity", None)
            jsonschema.Draft202012Validator.check_schema(base_schema)
            jsonschema.Draft202012Validator(base_schema).validate(base_candidate)
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
            "dependency": publication_dependency,
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
        if recovery_mode:
            assert recovery_evidence is not None
            receipt_value = {
                "schema_version": "1.0",
                "artifact_role": "r05a_actual_forward_canary_cpu_republication",
                "run_id": run_root.name,
                "case_id": case_id,
                "source_git_commit": git_commit,
                "recovery_git_commit": recovery_evidence["recovery_git_commit"],
                "source_job_id": str(args.source_job_id),
                "source_task_id": f"{args.source_job_id}_0",
                "source_job_state": "COMPLETED",
                "source_exit_code": "0:0",
                "source_node": "worker-1",
                "original_publisher_job_id": str(args.original_publisher_job_id),
                "original_publisher_state": "FAILED",
                "original_publisher_exit_code": "1:0",
                "original_publisher_published": False,
                "recovery_publisher_job_id": str(args.publisher_job_id),
                "recovery_publisher_host": os.uname().nodename.split(".", 1)[0],
                "recovery_resources": recovery_evidence["resources"],
                "dependency": publication_dependency,
                "result_path": str(output),
                "result_sha256": candidate_sha256,
                "recovery_receipt_path": str(receipt),
                "published": True,
                "passed": True,
                "outcome": outcome,
                "original_receipt_preserved": True,
                "recovery_cpu_is_sole_results_json_publisher": True,
                "original_evidence": raw_evidence,
                "recovery_evidence": {
                    key: value
                    for key, value in recovery_evidence.items()
                    if key not in {"config", "resources"}
                },
                "measured_telemetry": measured_telemetry,
                "scientific_claim_allowed": False,
                "simulator_efficacy_claim_allowed": False,
                "infeasibility_claim_allowed": False,
                "probe_training_authorized": False,
                "automatic_next_gate_authorized": False,
            }
        jsonschema.Draft202012Validator(receipt_schema).validate(receipt_value)
        _commit_publication(
            candidate,
            output,
            receipt,
            receipt_value,
            expected_output_sha256=candidate_sha256,
        )
    except Exception as error:
        output.parent.joinpath(".results.candidate.json").unlink(missing_ok=True)
        if not receipt.exists():
            failure_role = (
                "r05a_actual_forward_canary_cpu_republication"
                if args.recovery_config is not None
                else "r05a_actual_forward_canary_cpu_publication"
            )
            _atomic_json(
                receipt,
                {
                    "schema_version": "1.0",
                    "artifact_role": failure_role,
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
