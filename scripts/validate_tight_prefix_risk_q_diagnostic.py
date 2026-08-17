#!/usr/bin/env python3
"""Validate two independent compact tight-prefix Q training results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _git_identity, _load, _require,
)


def validate(
    *, repo_root: Path, config_path: Path, producer_path: Path,
    replay_path: Path, expected_commit: str,
) -> dict:
    from main.multilink_ellipsoid.shadow import allocation_record
    from main.multilink_ellipsoid.tight_prefix_risk_q_diagnostic import (
        RESULT_SCHEMA, VALIDATION_SCHEMA, canonical, load_config,
        payload_sha256, scientific_view,
    )

    config = load_config(config_path)
    producer = _load(producer_path)
    replay = _load(replay_path)
    for result in (producer, replay):
        _require(
            result.get("schema_version") == RESULT_SCHEMA
            and result.get("source", {}).get("commit") == expected_commit
            and result.get("result_payload_sha256")
            == payload_sha256(result, "result_payload_sha256")
            and result.get("config") == config,
            "tight prefix Q training result differs",
        )
    exact = canonical(scientific_view(producer)) == canonical(
        scientific_view(replay)
    )
    _require(exact, "tight prefix Q independent training differs")
    value = {
        "schema_version": VALIDATION_SCHEMA,
        "status": "validated_exact_independent_training",
        "scientific_result": True,
        "validator_source": _git_identity(repo_root, expected_commit),
        "allocation": allocation_record(),
        "producer": {
            "path": str(producer_path),
            "file_sha256": _file_sha256(producer_path),
            "payload_sha256": producer["result_payload_sha256"],
        },
        "replay": {
            "path": str(replay_path),
            "file_sha256": _file_sha256(replay_path),
            "payload_sha256": replay["result_payload_sha256"],
        },
        "independent_training_exact": exact,
        "model_sha256": producer["compact_shared_7D"]["model"][
            "model_sha256"
        ],
        "metrics": producer["compact_shared_7D"]["metrics"],
        "interpretation": producer["interpretation"],
        "correction_or_QP_authorized": False,
    }
    value["validation_payload_sha256"] = payload_sha256(
        value, "validation_payload_sha256",
    )
    return value


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--producer", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    value = validate(
        repo_root=args.repo_root.resolve(), config_path=args.config.resolve(),
        producer_path=args.producer.resolve(), replay_path=args.replay.resolve(),
        expected_commit=args.expected_commit,
    )
    _atomic_write(args.output.resolve(), value)
    print(json.dumps({
        "status": value["status"],
        "independent_training_exact": value["independent_training_exact"],
        "model_sha256": value["model_sha256"],
        "validation_payload_sha256": value["validation_payload_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
