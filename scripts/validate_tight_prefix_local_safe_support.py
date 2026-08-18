#!/usr/bin/env python3
"""Validate two independent immutable local-safe-support audits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional, Sequence

from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _git_identity, _load, _require,
)


def validate(
    *, repo_root: Path, config_path: Path, producer_result: Path,
    replay_result: Path, expected_commit: str, accepted_result_commit: str,
) -> dict[str, Any]:
    from main.multilink_ellipsoid.tight_prefix_local_safe_support import (
        RESULT_SCHEMA, VALIDATION_SCHEMA, canonical, load_config,
        payload_sha256, scientific_view,
    )

    config = load_config(config_path)
    producer = _load(producer_result)
    replay = _load(replay_result)
    for record in (producer, replay):
        _require(
            record.get("schema_version") == RESULT_SCHEMA
            and record.get("source", {}).get("commit")
            == accepted_result_commit
            and record.get("config_file_sha256")
            == config["config_file_sha256"]
            and record.get("config_payload_sha256")
            == config["config_payload_sha256"]
            and record.get("result_payload_sha256")
            == payload_sha256(record, "result_payload_sha256")
            and record.get("training_performed") is False
            and int(record.get("simulator_rollout_count", -1)) == 0
            and int(record.get("policy_query_count", -1)) == 0,
            "local safe-support result differs",
        )
    exact = canonical(scientific_view(producer)) == canonical(
        scientific_view(replay)
    )
    _require(exact, "local safe-support independent audit differs")
    result = {
        "schema_version": VALIDATION_SCHEMA,
        "status": "complete_independent_local_safe_support_validation",
        "scientific_result": True,
        "claim_scope": config["claim_scope"],
        "source": _git_identity(repo_root, expected_commit),
        "accepted_result_commit": accepted_result_commit,
        "config": config,
        "producer_result_payload_sha256": producer["result_payload_sha256"],
        "replay_result_payload_sha256": replay["result_payload_sha256"],
        "independent_scientific_audit_exact": exact,
        "case_count": producer["case_count"],
        "candidate_count": producer["candidate_count"],
        "split_summary": producer["split_summary"],
        "prior_gradient_probe_anchor_records": producer[
            "prior_gradient_probe_anchor_records"
        ],
        "decision": producer["decision"],
        "training_performed": False,
        "simulator_rollout_count": 0,
        "policy_query_count": 0,
        "paper_or_safety_claim_authorized": False,
    }
    result["validation_payload_sha256"] = payload_sha256(
        result, "validation_payload_sha256",
    )
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--producer-result", type=Path, required=True)
    parser.add_argument("--replay-result", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--accepted-result-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = validate(
        repo_root=args.repo_root.resolve(), config_path=args.config.resolve(),
        producer_result=args.producer_result.resolve(),
        replay_result=args.replay_result.resolve(),
        expected_commit=args.expected_commit,
        accepted_result_commit=args.accepted_result_commit,
    )
    _atomic_write(args.output.resolve(), result)
    print(json.dumps({
        "independent_scientific_audit_exact": result[
            "independent_scientific_audit_exact"
        ],
        "split_summary": result["split_summary"],
        "decision": result["decision"],
        "validation_payload_sha256": result["validation_payload_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
