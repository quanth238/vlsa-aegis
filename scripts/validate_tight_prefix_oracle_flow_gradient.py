#!/usr/bin/env python3
"""Independently validate the paired oracle-flow-gradient canary."""

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
            return float("inf")
        return max(
            (_maximum_numeric_difference(left[key], right[key]) for key in left),
            default=0.0,
        )
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return float("inf")
        return max(
            (_maximum_numeric_difference(a, b) for a, b in zip(left, right)),
            default=0.0,
        )
    if (
        isinstance(left, (int, float)) and not isinstance(left, bool)
        and isinstance(right, (int, float)) and not isinstance(right, bool)
    ):
        return abs(float(left) - float(right))
    return 0.0 if left == right else float("inf")


def validate(
    *, repo_root: Path, config_path: Path, producer_path: Path,
    replay_path: Path, expected_commit: str, accepted_commit: str,
) -> dict[str, Any]:
    from main.multilink_ellipsoid.tight_prefix_oracle_flow_gradient import (
        CASE_SCHEMA, VALIDATION_SCHEMA, file_sha256, load_config,
        payload_sha256, scientific_view,
    )

    config = load_config(config_path)
    producer = _load(producer_path)
    replay = _load(replay_path)
    checks = {
        "producer_schema": producer.get("schema_version") == CASE_SCHEMA,
        "replay_schema": replay.get("schema_version") == CASE_SCHEMA,
        "producer_status": producer.get("status")
        == "complete_oracle_flow_gradient_canary",
        "replay_status": replay.get("status")
        == "complete_oracle_flow_gradient_canary",
        "producer_replica": producer.get("replica") == "producer",
        "replay_replica": replay.get("replica") == "replay",
        "producer_source": producer.get("source", {}).get("commit") == accepted_commit,
        "replay_source": replay.get("source", {}).get("commit") == accepted_commit,
        "producer_payload": producer.get("result_payload_sha256")
        == payload_sha256(producer, "result_payload_sha256"),
        "replay_payload": replay.get("result_payload_sha256")
        == payload_sha256(replay, "result_payload_sha256"),
        "producer_config": producer.get("config_file_sha256")
        == config["config_file_sha256"]
        and producer.get("config_payload_sha256")
        == config["config_payload_sha256"],
        "replay_config": replay.get("config_file_sha256")
        == config["config_file_sha256"]
        and replay.get("config_payload_sha256")
        == config["config_payload_sha256"],
    }
    producer_view = scientific_view(producer)
    replay_view = scientific_view(replay)
    maximum_error = _maximum_numeric_difference(producer_view, replay_view)
    checks["scientific_view_exact"] = producer_view == replay_view
    checks["independent_replay_tolerance"] = maximum_error <= float(
        config["gate"]["maximum_independent_replay_absolute_error"]
    )
    checks["all_case_gates"] = bool(
        producer.get("summary", {}).get("checks")
        == replay.get("summary", {}).get("checks")
    )
    _require(all(checks.values()), "oracle-flow-gradient validation failed")
    canary_pass = bool(producer["summary"]["oracle_steering_canary_pass"])
    result = {
        "schema_version": VALIDATION_SCHEMA,
        "status": "passing" if canary_pass else "scientific_no_go",
        "scientific_result": True,
        "source": _git_identity(repo_root, expected_commit),
        "accepted_experiment_commit": accepted_commit,
        "config_file_sha256": config["config_file_sha256"],
        "config_payload_sha256": config["config_payload_sha256"],
        "producer": {
            "path": str(producer_path),
            "file_sha256": file_sha256(producer_path),
            "payload_sha256": producer["result_payload_sha256"],
        },
        "replay": {
            "path": str(replay_path),
            "file_sha256": file_sha256(replay_path),
            "payload_sha256": replay["result_payload_sha256"],
        },
        "checks": checks,
        "maximum_independent_replay_absolute_error": float(maximum_error),
        "summary": producer["summary"],
        "case_results": producer["case_results"],
        "oracle_steering_canary_pass": canary_pass,
        "expanded_oracle_study_authorized": canary_pass,
        "critic_retraining_authorized": False,
        "full_episode_or_online_guidance_authorized": False,
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
        "status": result["status"],
        "eligible_boundary_root_count": result["summary"]["eligible_boundary_root_count"],
        "oracle_safe_conversion_count": result["summary"]["oracle_safe_conversion_count"],
        "oracle_steering_canary_pass": result["oracle_steering_canary_pass"],
        "validation_payload_sha256": result["validation_payload_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
