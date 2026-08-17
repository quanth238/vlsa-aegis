#!/usr/bin/env python3
"""Validate independent tight prefix-risk producer/replay populations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _git_identity, _load, _require,
)


def validate(
    *, repo_root: Path, config_path: Path, producer_dir: Path,
    replay_dir: Path, expected_commit: str,
) -> dict:
    from main.multilink_ellipsoid.tight_prefix_risk_dataset import (
        CASE_SCHEMA, VALIDATION_SCHEMA, canonical, load_config,
        payload_sha256, scientific_view, summarize_records,
    )

    config = load_config(config_path, repo_root=repo_root)
    producers = []
    replays = []
    equality = {}
    for item in config["cases"]:
        case_id = str(item["case_id"])
        producer_path = producer_dir / (case_id + ".json")
        replay_path = replay_dir / (case_id + ".json")
        _require(producer_path.is_file(), "tight prefix-risk producer is missing")
        _require(replay_path.is_file(), "tight prefix-risk replay is missing")
        producer = _load(producer_path)
        replay = _load(replay_path)
        for record in (producer, replay):
            _require(record["schema_version"] == CASE_SCHEMA,
                     "tight prefix-risk case schema differs")
            _require(record["case_id"] == case_id,
                     "tight prefix-risk case identity differs")
            _require(record["source"]["commit"] == expected_commit,
                     "tight prefix-risk source commit differs")
            _require(record["result_payload_sha256"] == payload_sha256(record),
                     "tight prefix-risk payload differs")
        equality[case_id] = bool(
            canonical(scientific_view(producer))
            == canonical(scientific_view(replay))
        )
        producers.append(producer)
        replays.append(replay)
    _require(all(equality.values()),
             "tight prefix-risk independent replay differs")
    producer_summary = summarize_records(producers, config)
    replay_summary = summarize_records(replays, config)
    _require(canonical(producer_summary) == canonical(replay_summary),
             "tight prefix-risk summary replay differs")
    value = {
        "schema_version": VALIDATION_SCHEMA,
        "status": "complete",
        "scientific_result": True,
        "claim_scope": config["claim_scope"],
        "validator_source": _git_identity(repo_root, expected_commit),
        "config_file_sha256": _file_sha256(config_path),
        "config_payload_sha256": config["config_payload_sha256"],
        "producer_dir": str(producer_dir),
        "replay_dir": str(replay_dir),
        "independent_scientific_view_equal_by_case": equality,
        "summary": producer_summary,
        "dataset_gate_pass": producer_summary["dataset_gate_pass"],
        "diagnostic_Q_only_training_authorized": producer_summary[
            "diagnostic_Q_only_training_authorized"
        ],
        "paper_scale_or_untouched_test_claim_authorized": False,
        "correction_or_QP_authorized": False,
    }
    value["validation_payload_sha256"] = payload_sha256(value)
    return value


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--producer-dir", type=Path, required=True)
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    value = validate(
        repo_root=args.repo_root.resolve(),
        config_path=args.config.resolve(),
        producer_dir=args.producer_dir.resolve(),
        replay_dir=args.replay_dir.resolve(),
        expected_commit=args.expected_commit,
    )
    _atomic_write(args.output.resolve(), value)
    print(json.dumps({
        "dataset_gate_pass": value["dataset_gate_pass"],
        "diagnostic_Q_only_training_authorized": value[
            "diagnostic_Q_only_training_authorized"
        ],
        "summary": value["summary"],
        "validation_payload_sha256": value["validation_payload_sha256"],
    }, sort_keys=True), flush=True)
    return 0 if value["dataset_gate_pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
