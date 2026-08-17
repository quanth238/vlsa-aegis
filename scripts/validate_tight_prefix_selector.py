#!/usr/bin/env python3
"""Validate paired offline tight-prefix selector audits."""

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
    from main.multilink_ellipsoid.pi05_palm_l6_source_selection import (
        cpu_allocation_record,
    )
    from main.multilink_ellipsoid.tight_prefix_selector_audit import (
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
            "tight prefix selector result differs",
        )
    exact = canonical(scientific_view(producer)) == canonical(
        scientific_view(replay)
    )
    _require(exact, "tight prefix selector independent audit differs")
    value = {
        "schema_version": VALIDATION_SCHEMA,
        "status": "validated_exact_offline_selector_audit",
        "scientific_result": True,
        "validator_source": _git_identity(repo_root, expected_commit),
        "allocation": cpu_allocation_record(),
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
        "independent_audit_exact": exact,
        "validation_margin": producer["validation_margin"],
        "validation_rules": producer["validation_rules"],
        "frozen_rule_from_validation_only": producer[
            "frozen_rule_from_validation_only"
        ],
        "diagnostic_test_rules": producer["diagnostic_test_rules"],
        "opened_full_episode_pilot_supported": producer[
            "opened_full_episode_pilot_supported"
        ],
        "paper_or_safety_claim_authorized": False,
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
        "independent_audit_exact": value["independent_audit_exact"],
        "frozen_rule": value["frozen_rule_from_validation_only"],
        "opened_full_episode_pilot_supported": value[
            "opened_full_episode_pilot_supported"
        ],
        "validation_payload_sha256": value["validation_payload_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
