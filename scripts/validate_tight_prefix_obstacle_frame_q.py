#!/usr/bin/env python3
"""Validate independent matched 7D/17D obstacle-frame critic training."""

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
    replay_path: Path, expected_commit: str, accepted_training_commit: str,
) -> dict:
    from main.multilink_ellipsoid.pi05_palm_l6_source_selection import (
        cpu_allocation_record,
    )
    from main.multilink_ellipsoid.tight_prefix_obstacle_frame_q import (
        RESULT_SCHEMA, VALIDATION_SCHEMA, canonical, load_config,
        payload_sha256, scientific_view,
    )

    config = load_config(config_path)
    producer = _load(producer_path)
    replay = _load(replay_path)
    for result in (producer, replay):
        _require(
            result.get("schema_version") == RESULT_SCHEMA
            and result.get("source", {}).get("commit")
            == accepted_training_commit
            and result.get("result_payload_sha256")
            == payload_sha256(result, "result_payload_sha256")
            and result.get("config") == config
            and result.get("test_split_accessed") is False
            and result.get("directional_loss_used") is False,
            "obstacle-frame Q training result differs",
        )
    exact = canonical(scientific_view(producer)) == canonical(
        scientific_view(replay)
    )
    _require(exact, "obstacle-frame Q independent training differs")
    value = {
        "schema_version": VALIDATION_SCHEMA,
        "status": "validated_exact_matched_obstacle_frame_Q_training",
        "scientific_result": True,
        "validator_source": _git_identity(repo_root, expected_commit),
        "allocation": cpu_allocation_record(),
        "accepted_training_commit": accepted_training_commit,
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
        "model_sha256": {
            arm: producer["arms"][arm]["model"]["model_sha256"]
            for arm in ("compact_7D", "obstacle_frame_17D")
        },
        "gate": producer["gate"],
        "directional_observability_validation": producer[
            "directional_observability_validation"
        ],
        "metrics": {
            arm: producer["arms"][arm]["metrics"]
            for arm in ("compact_7D", "obstacle_frame_17D")
        },
        "test_split_accessed": False,
        "exact_gradient_probe_authorized": bool(
            producer["gate"]["exact_gradient_probe_authorized"]
        ),
        "full_episode_authorized": False,
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
    parser.add_argument("--accepted-training-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    value = validate(
        repo_root=args.repo_root.resolve(), config_path=args.config.resolve(),
        producer_path=args.producer.resolve(), replay_path=args.replay.resolve(),
        expected_commit=args.expected_commit,
        accepted_training_commit=args.accepted_training_commit,
    )
    _atomic_write(args.output.resolve(), value)
    print(json.dumps({
        "status": value["status"],
        "independent_training_exact": value["independent_training_exact"],
        "gate": value["gate"],
        "validation_payload_sha256": value["validation_payload_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
