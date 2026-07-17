#!/usr/bin/env python3
"""Capture the exact R06 canary branch before any AEGIS execution."""

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
CHECKPOINT_SHA256 = (
    "988055ccfd7032903c073a641f3c5f0f0541df444a315116a16f0bf4716d26ed"
)
CHECKPOINT_CONFIG_SHA256 = (
    "5c2728c53f4b33ee16380140f303713fdb5df78a8d26ee1489ace374fc54327a"
)
NORMALIZATION_ASSET_SHA256 = (
    "b3a44bb2810436fb62917decaea58bd4d9110255df527dea21e8fd40c960bd84"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--r02-config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--asset-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--case-index", type=int, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--checkpoint-id", required=True)
    parser.add_argument("--checkpoint-model", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--checkpoint-config", required=True)
    parser.add_argument("--checkpoint-config-sha256", required=True)
    parser.add_argument("--normalization-asset", required=True)
    parser.add_argument("--normalization-asset-sha256", required=True)
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
    rows: list[dict[str, Any]] = []
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


def _require_capture_release(
    value: Mapping[str, Any],
    *,
    run_id: str,
) -> Mapping[str, Any]:
    """Fail closed unless this is the separately reviewed capture-only release."""

    if value.get("ready_to_run") is not True or value.get("blocked_on") != []:
        raise ValueError("R06 capture config has not been released")
    release = value.get("execution_release")
    if not isinstance(release, Mapping):
        raise ValueError("R06 capture config has no execution_release")
    required = {
        "artifact_role": "r06_aegis_codex_label_capture_execution_release",
        "stage": "codex_label_capture",
        "run_id": run_id,
        "single_case_index": 0,
        "case_id": CANARY_CASE_ID,
        "aegis_execution_allowed": False,
        "semantic_label_required": False,
        "groundingdino_execution_allowed": False,
        "qp_execution_allowed": False,
        "robosuite_image_convention": "opengl",
        "robosuite_version": "1.4.1",
        "probe_or_mlp_training_authorized": False,
        "automatic_population_launch_authorized": False,
        "release_only_parent_required": True,
    }
    for key, expected in required.items():
        if release.get(key) != expected:
            raise ValueError(f"R06 capture release field changed: {key}")
    accepted = release.get("accepted_implementation_commit")
    if (
        not isinstance(accepted, str)
        or len(accepted) != 40
        or any(character not in "0123456789abcdef" for character in accepted)
    ):
        raise ValueError("R06 capture release has no accepted implementation commit")
    decision = release.get("decision_artifact")
    if not (
        isinstance(decision, str)
        and decision.startswith("docs/decisions/")
        and decision.endswith("-release-aegis-label-capture-canary.md")
    ):
        raise ValueError("R06 capture release decision path changed")
    if release.get("allowed_release_diff_paths") != [
        "configs/experiments/r06_aegis_collision_conditioned.json",
        decision,
    ]:
        raise ValueError("R06 capture release diff allowlist changed")
    resources = release.get("resources")
    expected_resources = {
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
    }
    if resources != expected_resources:
        raise ValueError("R06 capture release resources changed")
    return release


def _validate_live_release(
    release: Mapping[str, Any],
    *,
    repo_root: Path,
    config_path: Path,
) -> None:
    """Bind live source to the direct-child config-plus-ADR release commit."""

    expected = os.environ.get("EXPECTED_GIT_COMMIT", "").strip()
    if (
        len(expected) != 40
        or any(character not in "0123456789abcdef" for character in expected)
    ):
        raise ValueError("R06 capture requires EXPECTED_GIT_COMMIT")
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
    ).strip()
    if head != expected:
        raise ValueError("R06 capture live HEAD differs from EXPECTED_GIT_COMMIT")
    accepted = str(release["accepted_implementation_commit"])
    parents = subprocess.check_output(
        ["git", "rev-list", "--parents", "-n", "1", expected],
        cwd=repo_root,
        text=True,
    ).strip()
    if parents != f"{expected} {accepted}":
        raise ValueError("R06 capture release is not the direct child of its implementation")
    observed = subprocess.check_output(
        ["git", "diff", "--name-only", accepted, expected],
        cwd=repo_root,
        text=True,
    ).splitlines()
    config_relative = str(config_path.resolve().relative_to(repo_root))
    expected_paths = sorted(
        [config_relative, str(release["decision_artifact"])]
    )
    if sorted(observed) != expected_paths:
        raise ValueError("R06 capture release diff is not exact config plus release ADR")


def _validate_static_inputs(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    if args.case_index != 0:
        raise ValueError("R06 capture canary is fixed to manifest row zero")
    if _sha256(args.manifest) != FROZEN_MANIFEST_SHA256:
        raise ValueError("R06 collision manifest hash changed")
    rows = _rows(args.manifest)
    if len(rows) != 20 or len({row.get("case_id") for row in rows}) != 20:
        raise ValueError("R06 requires the immutable 20-case manifest")
    if rows[0].get("case_id") != CANARY_CASE_ID:
        raise ValueError("R06 capture canary identity changed")
    if _sha256(args.r02_config) != R02_CONFIG_SHA256:
        raise ValueError("R06 source R02 config hash changed")
    if args.checkpoint_sha256 != CHECKPOINT_SHA256:
        raise ValueError("R06 checkpoint digest argument changed")
    if _sha256(args.checkpoint_model) != CHECKPOINT_SHA256:
        raise ValueError("R06 live checkpoint model hash changed")
    if args.checkpoint_config_sha256 != CHECKPOINT_CONFIG_SHA256:
        raise ValueError("R06 checkpoint-config digest argument changed")
    if _sha256(args.checkpoint_config) != CHECKPOINT_CONFIG_SHA256:
        raise ValueError("R06 live checkpoint config hash changed")
    if args.normalization_asset_sha256 != NORMALIZATION_ASSET_SHA256:
        raise ValueError("R06 normalization digest argument changed")
    if _sha256(args.normalization_asset) != NORMALIZATION_ASSET_SHA256:
        raise ValueError("R06 live normalization asset hash changed")
    config_value = _object(args.config, name="R06 config")
    _require_capture_release(config_value, run_id=args.run_id)
    source = config_value.get("source_evidence")
    checkpoint = source.get("checkpoint") if isinstance(source, Mapping) else None
    if not isinstance(checkpoint, Mapping):
        raise ValueError("R06 config has no checkpoint source binding")
    checkpoint_root = Path(str(checkpoint.get("id", ""))).resolve()
    if Path(args.checkpoint_id).resolve() != checkpoint_root:
        raise ValueError("R06 checkpoint directory differs from the config")
    expected_paths = {
        "checkpoint model": (args.checkpoint_model, checkpoint_root / "model.safetensors"),
        "checkpoint config": (args.checkpoint_config, checkpoint_root / "config.json"),
        "normalization asset": (
            args.normalization_asset,
            checkpoint_root
            / "assets"
            / "physical-intelligence"
            / "libero"
            / "norm_stats.json",
        ),
    }
    for label, (observed, expected) in expected_paths.items():
        if Path(observed).resolve() != expected:
            raise ValueError(f"R06 {label} path differs from the frozen checkpoint")
    return rows[0], _object(args.r02_config, name="R02 config")


def main() -> int:
    args = _parser().parse_args()
    try:
        case, r02_value = _validate_static_inputs(args)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error

    # Heavy simulator, NumPy, and WebSocket dependencies are allocation-only.
    from crfs_oracle.aegis_runner import (
        SafeLiberoAegisRuntime,
        load_aegis_experiment_config,
        run_aegis_label_capture,
    )
    from crfs_oracle.runner import (
        SafeLiberoCase,
        oracle_config_from_mapping,
    )
    from openpi_client import websocket_client_policy

    root = Path(__file__).resolve().parents[1]
    config = load_aegis_experiment_config(
        args.config,
        repo_root=root,
        require_execution_release=True,
    )
    _validate_live_release(
        config.execution_release or {},
        repo_root=root,
        config_path=Path(args.config),
    )
    oracle = oracle_config_from_mapping(
        r02_value,
        host=args.host,
        port=args.port,
        checkpoint_id=args.checkpoint_id,
        checkpoint_sha256=args.checkpoint_sha256,
        output_root=str(Path(args.output).parent),
        run_id=args.run_id,
    )
    safe_case = SafeLiberoCase(case, oracle)
    runtime = SafeLiberoAegisRuntime(safe_case)
    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    try:
        output, status = run_aegis_label_capture(
            case,
            config,
            output_path=args.output,
            asset_dir=args.asset_dir,
            client=client,
            runtime=runtime,
        )
    finally:
        runtime.close()
    if status != "capture_complete":
        raise SystemExit(f"R06 capture returned unexpected status: {status}")
    print(f"{status} {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
