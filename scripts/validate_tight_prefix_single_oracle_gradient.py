#!/usr/bin/env python3
"""Validate the producer-generated/exact-action CPU-replayed E15 decision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _git_identity, _load, _require,
)


def _maximum_numeric_difference(left: Any, right: Any) -> float:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if set(left) != set(right):
            return 0.0
        return max(
            (_maximum_numeric_difference(left[key], right[key]) for key in left),
            default=0.0,
        )
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return 0.0
        return max(
            (_maximum_numeric_difference(a, b) for a, b in zip(left, right)),
            default=0.0,
        )
    if (
        isinstance(left, (int, float)) and not isinstance(left, bool)
        and isinstance(right, (int, float)) and not isinstance(right, bool)
    ):
        return abs(float(left) - float(right))
    return 0.0


def validate(
    *, repo_root: Path, config_path: Path, producer_path: Path,
    replay_path: Path, expected_commit: str, accepted_commit: str,
) -> dict[str, Any]:
    from main.multilink_ellipsoid.tight_prefix_single_oracle_gradient import (
        RESULT_SCHEMA, VALIDATION_SCHEMA, file_sha256, load_config,
        payload_sha256, scientific_view,
    )

    config = load_config(config_path)
    producer = _load(producer_path)
    replay = _load(replay_path)
    producer_view = scientific_view(producer)
    replay_view = scientific_view(replay)
    maximum_error = _maximum_numeric_difference(producer_view, replay_view)
    checks = {
        "producer_schema": producer.get("schema_version") == RESULT_SCHEMA,
        "replay_schema": replay.get("schema_version") == RESULT_SCHEMA,
        "producer_status": producer.get("status") == "complete_single_oracle_gradient",
        "replay_status": replay.get("status") == "complete_single_oracle_gradient",
        "producer_replica": producer.get("replica") == "producer",
        "replay_replica": replay.get("replica") == "replay",
        "producer_source": producer.get("source", {}).get("commit") == accepted_commit,
        "replay_source": replay.get("source", {}).get("commit") == accepted_commit,
        "producer_payload": producer.get("result_payload_sha256")
        == payload_sha256(producer, "result_payload_sha256"),
        "replay_payload": replay.get("result_payload_sha256")
        == payload_sha256(replay, "result_payload_sha256"),
        "producer_config": producer.get("config_payload_sha256")
        == config["config_payload_sha256"],
        "replay_config": replay.get("config_payload_sha256")
        == config["config_payload_sha256"],
        "frozen_action_bundle_exact": producer.get("action_bundle")
        == replay.get("action_bundle"),
        "scientific_view_exact": producer_view == replay_view,
        "numeric_tolerance": maximum_error
        <= float(config["gate"]["maximum_exact_replay_absolute_error"]),
        "zero_physical_false_safes": int(
            producer.get("summary", {}).get("physical_false_safe_count", -1)
        ) <= int(config["gate"]["maximum_physical_false_safe_count"])
        and producer.get("summary", {}).get("physical_false_safe_count")
        == replay.get("summary", {}).get("physical_false_safe_count"),
    }
    binding_keys = {
        "producer_schema", "replay_schema", "producer_status", "replay_status",
        "producer_replica", "replay_replica", "producer_source", "replay_source",
        "producer_payload", "replay_payload", "producer_config", "replay_config",
    }
    _require(
        all(checks[key] for key in binding_keys),
        "single-oracle-gradient validation binding failed",
    )
    exact_replay = bool(all(checks.values()))
    decision = str(producer["summary"]["decision"])
    if not exact_replay:
        status = "apparatus_no_go"
    elif decision == "unsuitable_raw_terminal_not_unsafe":
        status = "unsuitable_case_stop"
    elif decision in {"oracle_local_correction_no_go", "oracle_gradient_unavailable"}:
        status = "oracle_local_no_go"
    elif decision == "oracle_succeeds_MLP_fails":
        status = "critic_gradient_no_go"
    elif decision == "oracle_and_MLP_succeed":
        status = "ready_tiny_full_episode"
    else:
        status = "apparatus_no_go"
    full_episode = bool(exact_replay and decision == "oracle_and_MLP_succeed")
    result = {
        "schema_version": VALIDATION_SCHEMA,
        "status": status,
        "scientific_result": exact_replay,
        "source": _git_identity(repo_root, expected_commit),
        "accepted_experiment_commit": accepted_commit,
        "config_file_sha256": config["config_file_sha256"],
        "config_payload_sha256": config["config_payload_sha256"],
        "producer": {
            "path": str(producer_path), "file_sha256": file_sha256(producer_path),
            "payload_sha256": producer["result_payload_sha256"],
        },
        "replay": {
            "path": str(replay_path), "file_sha256": file_sha256(replay_path),
            "payload_sha256": replay["result_payload_sha256"],
        },
        "checks": checks,
        "maximum_exact_replay_absolute_error": float(maximum_error),
        "decision": decision,
        "summary": producer["summary"],
        "phase_records": producer["phase_records"],
        "exact_gradient": producer["exact_gradient"],
        "learned_gradient": producer["learned_gradient"],
        "selected_exact_safe_oracle_radius_l2": producer[
            "selected_exact_safe_oracle_radius_l2"
        ],
        "comparison_metrics": producer["comparison_metrics"],
        "tiny_full_episode_authorized": full_episode,
        "retraining_or_collection_authorized": False,
        "QP_authorized": False,
    }
    result["validation_payload_sha256"] = payload_sha256(
        result, "validation_payload_sha256",
    )
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--producer", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--accepted-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = validate(
        repo_root=args.repo_root.resolve(), config_path=args.config.resolve(),
        producer_path=args.producer.resolve(), replay_path=args.replay.resolve(),
        expected_commit=args.expected_commit, accepted_commit=args.accepted_commit,
    )
    _atomic_write(args.output.resolve(), result)
    print(json.dumps({
        "status": result["status"], "decision": result["decision"],
        "tiny_full_episode_authorized": result["tiny_full_episode_authorized"],
        "validation_payload_sha256": result["validation_payload_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
