#!/usr/bin/env python3
"""Run the exact row-0 pi0.5 versus pi0.5+AEGIS paired canary."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping


CANARY_CASE_ID = "crfs-1069f29a8d76463a"
FROZEN_MANIFEST_SHA256 = (
    "b12319d3fed151bfee85e1615bd72254474a1480308389d747dce954c6131e41"
)
R02_CONFIG_SHA256 = (
    "c5be1b4759487f4d6541e49110b90fcf5de404bdaed5f389358db0abe4388f4e"
)
SOURCE_R02_SHA256 = (
    "055fcf18781071c6c3575b42a1b44c32a76b474ac1911ad8c5443f78b9c42593"
)
CHECKPOINT_SHA256 = (
    "988055ccfd7032903c073a641f3c5f0f0541df444a315116a16f0bf4716d26ed"
)
CHECKPOINT_CONFIG_SHA256 = (
    "5c2728c53f4b33ee16380140f303713fdb5df78a8d26ee1489ace374fc54327a"
)
NORMALIZATION_ASSET_SHA256 = (
    "b3a44bb2810436fb62917decaea58bd4d9110255df527dea21e8fd40c960bd84"
)
CODEX_LABEL_SHA256 = (
    "6a22b6d2f3705c008e338be6b217ce947fabfd4f56f70d3a8f442467694d996f"
)
LABEL_FREEZE_COMMIT = "d9cf569d619e014c9e6423cd9ddb40f592435a71"
CAPTURE_SHA256 = (
    "f2d32024ac6fac5aa5d133b5138df72501f1590b8cb0927b732edde69dabaaac"
)
CAPTURE_COMPLETED_AT = "2026-07-17T06:55:59Z"
DINO_CONFIG_SHA256 = (
    "172e80017f9395668a9cb5d1b8bd9d061c0e360471c6ed673c83b69bb14399f1"
)
DINO_CHECKPOINT_SHA256 = (
    "3b3ca2563c77c69f651d7bd133e97139c186df06231157a64c507099c52bc799"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "manifest",
        "config",
        "r02-config",
        "source-r02",
        "output",
        "perception-output-dir",
        "run-id",
        "checkpoint-id",
        "checkpoint-model",
        "checkpoint-config",
        "normalization-asset",
        "label-manifest",
        "capture-artifact",
        "dino-config",
        "dino-checkpoint",
    ):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--case-index", type=int, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    return parser


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _object(path: str | Path, *, name: str) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _rows(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, raw in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw.strip():
            raise ValueError(f"manifest line {line_number} is blank")
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError(f"manifest line {line_number} is not an object")
        rows.append(value)
    return rows


def _require_paired_release(
    value: Mapping[str, Any], *, run_id: str
) -> Mapping[str, Any]:
    if value.get("ready_to_run") is not True or value.get("blocked_on") != []:
        raise ValueError("R06 paired canary has not been released")
    release = value.get("execution_release")
    if not isinstance(release, Mapping):
        raise ValueError("R06 paired canary has no execution_release")
    required = {
        "artifact_role": "r06_aegis_paired_codex_label_canary_execution_release",
        "stage": "paired_codex_label_canary",
        "run_id": run_id,
        "single_case_index": 0,
        "case_id": CANARY_CASE_ID,
        "aegis_execution_allowed": True,
        "semantic_label_required": True,
        "groundingdino_execution_allowed": True,
        "qp_execution_allowed": True,
        "original_glm_execution_allowed": False,
        "robosuite_image_convention": "opengl",
        "robosuite_version": "1.4.1",
        "probe_or_mlp_training_authorized": False,
        "automatic_population_launch_authorized": False,
        "release_only_parent_required": True,
    }
    for key, expected in required.items():
        if release.get(key) != expected:
            raise ValueError(f"R06 paired release field changed: {key}")
    resources = {
        "partition": "main",
        "account": "normal",
        "qos": "normal",
        "gpus": 1,
        "cpus_per_task": 8,
        "host_memory_mib": 65536,
        "time_limit": "00:30:00",
        "array": "0-0%1",
        "source_host": "worker-1",
        "requeue": False,
        "validator_partition": "main",
        "validator_account": "normal",
        "validator_qos": "normal",
        "validator_cpus": 2,
        "validator_host_memory_mib": 8192,
        "validator_time_limit": "00:10:00",
        "validator_gpus": 0,
        "validator_dependency": "afterany",
    }
    if release.get("resources") != resources:
        raise ValueError("R06 paired release resources changed")
    accepted = release.get("accepted_implementation_commit")
    if not (
        isinstance(accepted, str)
        and len(accepted) == 40
        and all(character in "0123456789abcdef" for character in accepted)
    ):
        raise ValueError("R06 paired release implementation commit is invalid")
    decision = release.get("decision_artifact")
    if not (
        isinstance(decision, str)
        and decision.startswith("docs/decisions/")
        and decision.endswith("-release-aegis-paired-canary.md")
    ):
        raise ValueError("R06 paired release decision path changed")
    if release.get("allowed_release_diff_paths") != [
        "configs/experiments/r06_aegis_collision_conditioned.json",
        decision,
    ]:
        raise ValueError("R06 paired release diff allowlist changed")
    label = release.get("codex_label_manifest")
    expected_label = {
        "path": "manifests/r06_codex_obstacle_labels_canary.jsonl",
        "sha256": CODEX_LABEL_SHA256,
        "freeze_commit": LABEL_FREEZE_COMMIT,
        "capture_artifact_path": (
            "/mnt/data/quanth/experiments/crfs-oracle/"
            "r06-aegis-label-capture-canary-20260717b/"
            f"{CANARY_CASE_ID}/capture.json"
        ),
        "capture_artifact_sha256": CAPTURE_SHA256,
        "capture_completed_at": CAPTURE_COMPLETED_AT,
    }
    if label != expected_label:
        raise ValueError("R06 paired Codex label/capture binding changed")
    return release


def _validate_live_release(
    release: Mapping[str, Any], *, repo_root: Path, config_path: Path
) -> None:
    expected = os.environ.get("EXPECTED_GIT_COMMIT", "").strip()
    if len(expected) != 40 or any(c not in "0123456789abcdef" for c in expected):
        raise ValueError("R06 paired canary requires EXPECTED_GIT_COMMIT")
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
    ).strip()
    if head != expected:
        raise ValueError("R06 paired live HEAD differs from EXPECTED_GIT_COMMIT")
    accepted = str(release["accepted_implementation_commit"])
    parents = subprocess.check_output(
        ["git", "rev-list", "--parents", "-n", "1", expected],
        cwd=repo_root,
        text=True,
    ).strip()
    if parents != f"{expected} {accepted}":
        raise ValueError("R06 paired release is not the direct implementation child")
    observed = subprocess.check_output(
        ["git", "diff", "--name-only", accepted, expected],
        cwd=repo_root,
        text=True,
    ).splitlines()
    config_relative = str(config_path.resolve().relative_to(repo_root))
    if sorted(observed) != sorted(
        [config_relative, str(release["decision_artifact"])]
    ):
        raise ValueError("R06 paired release diff is not exact config plus release ADR")
    label = release["codex_label_manifest"]
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", label["freeze_commit"], expected],
        cwd=repo_root,
        check=False,
    ).returncode:
        raise ValueError("Codex label freeze commit is not a release ancestor")
    frozen = subprocess.check_output(
        [
            "git",
            "show",
            f"{label['freeze_commit']}:{label['path']}",
        ],
        cwd=repo_root,
    )
    if hashlib.sha256(frozen).hexdigest() != CODEX_LABEL_SHA256:
        raise ValueError("Codex label differs from its freeze commit")


def _validate_static_inputs(
    args: argparse.Namespace, *, repo_root: Path
) -> tuple[dict[str, Any], dict[str, Any], Mapping[str, Any]]:
    if args.case_index != 0:
        raise ValueError("R06 paired canary is fixed to manifest row zero")
    expected_hashes = {
        args.manifest: FROZEN_MANIFEST_SHA256,
        args.r02_config: R02_CONFIG_SHA256,
        args.source_r02: SOURCE_R02_SHA256,
        args.checkpoint_model: CHECKPOINT_SHA256,
        args.checkpoint_config: CHECKPOINT_CONFIG_SHA256,
        args.normalization_asset: NORMALIZATION_ASSET_SHA256,
        args.label_manifest: CODEX_LABEL_SHA256,
        args.capture_artifact: CAPTURE_SHA256,
        args.dino_config: DINO_CONFIG_SHA256,
        args.dino_checkpoint: DINO_CHECKPOINT_SHA256,
    }
    for path, expected in expected_hashes.items():
        if Path(path).is_symlink() or _sha256(path) != expected:
            raise ValueError(f"R06 paired static input changed: {path}")
    rows = _rows(args.manifest)
    if len(rows) != 20 or rows[0].get("case_id") != CANARY_CASE_ID:
        raise ValueError("R06 paired canary manifest identity changed")
    config_value = _object(args.config, name="R06 config")
    release = _require_paired_release(config_value, run_id=args.run_id)
    label_release = release["codex_label_manifest"]
    label_path = (repo_root / str(label_release["path"])).resolve()
    if Path(args.label_manifest).resolve() != label_path:
        raise ValueError("R06 paired label-manifest path changed")
    if Path(args.capture_artifact).resolve() != Path(
        str(label_release["capture_artifact_path"])
    ).resolve():
        raise ValueError("R06 paired capture-artifact path changed")
    checkpoint_root = Path(args.checkpoint_id).resolve()
    expected_checkpoint_paths = {
        Path(args.checkpoint_model).resolve(): checkpoint_root / "model.safetensors",
        Path(args.checkpoint_config).resolve(): checkpoint_root / "config.json",
        Path(args.normalization_asset).resolve(): (
            checkpoint_root
            / "assets/physical-intelligence/libero/norm_stats.json"
        ),
    }
    if any(observed != expected for observed, expected in expected_checkpoint_paths.items()):
        raise ValueError("R06 paired checkpoint path binding changed")
    return rows[0], _object(args.r02_config, name="R02 config"), release


def main() -> int:
    args = _parser().parse_args()
    root = Path(__file__).resolve().parents[1]
    try:
        case, r02_value, release = _validate_static_inputs(args, repo_root=root)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error

    from crfs_oracle.aegis_runner import (
        CodexFrozenLabelSafetyCoreProvider,
        SafeLiberoAegisRuntime,
        load_aegis_experiment_config,
        run_aegis_case,
        validate_aegis_case_result,
    )
    from crfs_oracle.runner import SafeLiberoCase, oracle_config_from_mapping
    from openpi_client import websocket_client_policy

    config = load_aegis_experiment_config(
        args.config, repo_root=root, require_execution_release=True
    )
    _validate_live_release(release, repo_root=root, config_path=Path(args.config))
    oracle = oracle_config_from_mapping(
        r02_value,
        host=args.host,
        port=args.port,
        checkpoint_id=args.checkpoint_id,
        checkpoint_sha256=CHECKPOINT_SHA256,
        output_root=str(Path(args.output).parent),
        run_id=args.run_id,
    )
    runtime = SafeLiberoAegisRuntime(SafeLiberoCase(case, oracle))
    provider = CodexFrozenLabelSafetyCoreProvider(
        label_jsonl_path=args.label_manifest,
        label_jsonl_sha256=CODEX_LABEL_SHA256,
        output_dir=args.perception_output_dir,
        repo_root=root,
    )
    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    try:
        output, status = run_aegis_case(
            case,
            config,
            output_path=args.output,
            client=client,
            runtime=runtime,
            aegis_provider=provider,
        )
    finally:
        runtime.close()
    value = _object(output, name="R06 paired result")
    errors = list(validate_aegis_case_result(value))
    if errors:
        raise SystemExit("R06 paired result is invalid: " + "; ".join(errors))
    print(f"{status} {output}")
    # A dependency-complete method failure is a retained scientific outcome.
    # Apparatus invalidity is also persisted, but must fail the GPU task so the
    # afterany validator cannot mistake it for a successfully run method.
    return 2 if status == "apparatus_invalid" else 0


if __name__ == "__main__":
    raise SystemExit(main())
