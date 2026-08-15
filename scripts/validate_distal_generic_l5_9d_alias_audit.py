#!/usr/bin/env python3
"""Independently validate the no-training 9D L5 alias audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional, Sequence

from scripts.audit_distal_generic_l5_9d_alias import compute_audit, load_sources
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _git_identity, _load, _require,
)


def validate(
    *, repo_root: Path, result_path: Path, config_path: Path,
    producer_commit: str, validator_commit: str,
) -> dict[str, Any]:
    from main.multilink_ellipsoid.generic_l5_9d_alias_audit import (
        RESULT_SCHEMA, VALIDATION_SCHEMA, load_config, payload_sha256,
    )

    _git_identity(repo_root, validator_commit)
    config = load_config(config_path)
    result = _load(result_path)
    _require(result["schema_version"] == RESULT_SCHEMA, "alias-audit result schema differs")
    _require(result["source"]["commit"] == producer_commit, "alias-audit producer differs")
    _require(result["config"] == config, "alias-audit embedded config differs")
    _require(
        result["result_payload_sha256"]
        == payload_sha256(result, "result_payload_sha256"),
        "alias-audit result payload differs",
    )
    capacity_result, capacity_config, grouped = load_sources(
        repo_root=repo_root, audit_config=config,
    )
    replay = compute_audit(
        repo_root=repo_root, audit_config=config, result=capacity_result,
        capacity_config=capacity_config, grouped=grouped,
    )
    _require(replay == result["audit"], "alias-audit independent replay differs")
    output = {
        "schema_version": VALIDATION_SCHEMA, "status": "validated",
        "scientific_result": True, "producer_commit": producer_commit,
        "validator_commit": validator_commit,
        "result_file_sha256": _file_sha256(result_path),
        "result_payload_sha256": result["result_payload_sha256"],
        "audit": replay,
        "independent_replay_exact": True,
        "interpretation": replay["interpretation"],
        "correction_authorized": False, "QP_authorized": False,
        "calibration_authorized": False, "closed_loop_authorized": False,
    }
    output["validation_payload_sha256"] = payload_sha256(
        output, "validation_payload_sha256",
    )
    return output


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--producer-commit", required=True)
    parser.add_argument("--validator-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = validate(
        repo_root=args.repo_root.resolve(), result_path=args.result.resolve(),
        config_path=args.config.resolve(), producer_commit=args.producer_commit,
        validator_commit=args.validator_commit,
    )
    _atomic_write(args.output.resolve(), output)
    print(json.dumps({
        "interpretation": output["interpretation"],
        "comparison": output["audit"]["oracle_anchor_comparison"],
        "validation_payload_sha256": output["validation_payload_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
