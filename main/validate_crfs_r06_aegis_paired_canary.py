#!/usr/bin/env python3
"""CPU-afterany validator for the exact R06 paired AEGIS canary."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping


CASE_ID = "crfs-1069f29a8d76463a"
SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "run-root",
        "config",
        "result",
        "receipt",
        "source-contract",
        "source-contract-sha256",
        "submission",
        "submission-sha256",
        "source-job-id",
        "validator-job-id",
        "source-state",
        "source-exit-code",
    ):
        parser.add_argument(f"--{name}", required=True)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _object(path: Path, *, name: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{name} is missing, symlinked, or not regular")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _validate_transaction(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    run_root = Path(args.run_root).resolve()
    result_path = Path(args.result).resolve()
    receipt_path = Path(args.receipt).resolve()
    source_path = Path(args.source_contract).resolve()
    submission_path = Path(args.submission).resolve()
    expected_paths = {
        result_path: run_root / CASE_ID / "results.json",
        receipt_path: run_root / "cpu-afterany-validation.json",
        source_path: run_root / "source-contract.json",
        submission_path: run_root / "submission.json",
    }
    for observed, expected in expected_paths.items():
        if observed != expected:
            raise ValueError(f"transaction path changed: {observed.name}")
    for digest_name in ("source_contract_sha256", "submission_sha256"):
        if SHA256.fullmatch(getattr(args, digest_name)) is None:
            raise ValueError(f"{digest_name} is not SHA-256")
    if _sha256(source_path) != args.source_contract_sha256:
        raise ValueError("source contract digest changed")
    if _sha256(submission_path) != args.submission_sha256:
        raise ValueError("submission digest changed")
    source = _object(source_path, name="source contract")
    submission = _object(submission_path, name="submission")
    if not (
        source.get("artifact_role") == "r06_aegis_paired_canary_source_contract"
        and source.get("run_id") == run_root.name
        and source.get("stage") == "paired_codex_label_canary"
        and source.get("collision_conditioned_only") is True
        and source.get("probe_or_mlp_training_authorized") is False
    ):
        raise ValueError("source contract identity changed")
    if not (
        submission.get("artifact_role")
        == "r06_aegis_paired_canary_atomic_submission"
        and submission.get("run_id") == run_root.name
        and submission.get("gpu_slurm_array_job_id") == args.source_job_id
        and submission.get("exact_gpu_task_id") == f"{args.source_job_id}_0"
        and submission.get("cpu_afterany_job_id") == args.validator_job_id
        and submission.get("dependency") == f"afterany:{args.source_job_id}"
        and submission.get("source_contract_sha256")
        == args.source_contract_sha256
        and submission.get("released_at_receipt_time") is False
    ):
        raise ValueError("atomic submission binding changed")
    frozen_sources = {
        "manifest_sha256": "b12319d3fed151bfee85e1615bd72254474a1480308389d747dce954c6131e41",
        "r02_config_sha256": "c5be1b4759487f4d6541e49110b90fcf5de404bdaed5f389358db0abe4388f4e",
        "source_r02_pair_sha256": "055fcf18781071c6c3575b42a1b44c32a76b474ac1911ad8c5443f78b9c42593",
        "checkpoint_sha256": "988055ccfd7032903c073a641f3c5f0f0541df444a315116a16f0bf4716d26ed",
        "checkpoint_config_sha256": "5c2728c53f4b33ee16380140f303713fdb5df78a8d26ee1489ace374fc54327a",
        "normalization_asset_sha256": "b3a44bb2810436fb62917decaea58bd4d9110255df527dea21e8fd40c960bd84",
        "groundingdino_config_sha256": "172e80017f9395668a9cb5d1b8bd9d061c0e360471c6ed673c83b69bb14399f1",
        "groundingdino_checkpoint_sha256": "3b3ca2563c77c69f651d7bd133e97139c186df06231157a64c507099c52bc799",
        "codex_label_manifest_sha256": "6a22b6d2f3705c008e338be6b217ce947fabfd4f56f70d3a8f442467694d996f",
        "capture_artifact_sha256": "f2d32024ac6fac5aa5d133b5138df72501f1590b8cb0927b732edde69dabaaac",
    }
    if any(source.get(key) != expected for key, expected in frozen_sources.items()):
        raise ValueError("source contract frozen input binding changed")
    if args.source_state != "COMPLETED" or args.source_exit_code != "0:0":
        raise ValueError(
            f"source GPU task was not successful: "
            f"{args.source_state}/{args.source_exit_code}"
        )
    return source, submission, source.get("git_commit", "")


def main() -> int:
    args = _parser().parse_args()
    receipt = Path(args.receipt).resolve()
    base = {
        "schema_version": "1.0",
        "artifact_role": "r06_aegis_paired_canary_cpu_validation",
        "run_id": Path(args.run_root).resolve().name,
        "case_id": CASE_ID,
        "source_job_id": args.source_job_id,
        "exact_source_task_id": f"{args.source_job_id}_0",
        "validator_job_id": args.validator_job_id,
        "source_state": args.source_state,
        "source_exit_code": args.source_exit_code,
        "probe_or_mlp_training_authorized": False,
        "population_launch_authorized": False,
    }
    try:
        source, submission, git_commit = _validate_transaction(args)
        config_value = _object(Path(args.config), name="R06 config")
        from run_crfs_r06_aegis_paired_canary import _require_paired_release

        release = _require_paired_release(config_value, run_id=base["run_id"])
        if (
            source.get("config_sha256") != _sha256(Path(args.config))
            or source.get("accepted_implementation_commit")
            != release.get("accepted_implementation_commit")
        ):
            raise ValueError("live config digest differs from source contract")
        result_path = Path(args.result).resolve()
        value = _object(result_path, name="paired result")
        from crfs_oracle.aegis_runner import validate_aegis_case_result

        errors = list(validate_aegis_case_result(value))
        if errors:
            raise ValueError("paired result failed validation: " + "; ".join(errors))
        execution = value.get("execution")
        if not (
            value.get("config", {}).get("sha256") == source.get("config_sha256")
            and value.get("implementation_identity", {}).get("source_git_commit")
            == git_commit
            and isinstance(execution, Mapping)
            and execution.get("run_id")
            == source.get("run_id")
            == submission.get("run_id")
            and execution.get("release_git_commit")
            == git_commit
            == submission.get("git_commit")
            and execution.get("slurm_array_job_id")
            == args.source_job_id
            == submission.get("gpu_slurm_array_job_id")
            and execution.get("slurm_array_task_id") == "0"
            and execution.get("exact_gpu_task_id")
            == f"{args.source_job_id}_0"
            == submission.get("exact_gpu_task_id")
            and execution.get("source_host")
            == source.get("resources", {}).get("source_host")
            == submission.get("source_host")
            == "worker-1"
            and isinstance(execution.get("cuda_visible_devices"), str)
            and bool(execution["cuda_visible_devices"])
        ):
            raise ValueError(
                "paired result differs from source/submission/exact-task identity"
            )
        result_sha = _sha256(result_path)
        _atomic(
            receipt,
            {
                **base,
                "status": "validated_terminal_result",
                "git_commit": git_commit,
                "source_contract_sha256": args.source_contract_sha256,
                "submission_sha256": args.submission_sha256,
                "result_path": str(result_path),
                "result_sha256": result_sha,
                "result_status": value.get("status"),
                "canary_apparatus_valid": value.get("canary_apparatus_valid"),
                "paired_execution": dict(execution),
                "aegis_safety_result_allowed": bool(
                    value.get("status") == "complete"
                    and value.get("canary_apparatus_valid") is True
                ),
                "general_benchmark_claim_allowed": False,
            },
        )
        print(f"validated_result={result_path}")
        print(f"validated_result_sha256={result_sha}")
        print(f"validation_receipt={receipt}")
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as error:
        result_status = None
        result_path = Path(args.result).resolve()
        if result_path.is_file() and not result_path.is_symlink():
            try:
                candidate = _object(result_path, name="paired result")
                result_status = candidate.get("status")
            except (OSError, ValueError, json.JSONDecodeError):
                result_status = "unreadable"
        _atomic(
            receipt,
            {
                **base,
                "status": "apparatus_inconclusive",
                "error": str(error),
                "result_status": result_status,
                "aegis_safety_result_allowed": False,
                "general_benchmark_claim_allowed": False,
            },
        )
        print(str(error))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
