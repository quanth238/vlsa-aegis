#!/usr/bin/env python3
"""Validate independent natural pi0.5 palm/L6 source audits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional, Sequence

from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _git_identity, _load, _require,
)


VALIDATION_SCHEMA = "vlsa_distal_pi05_palm_l6_source_audit_validation.v1"


def run(
    *, repo_root: Path, producer_path: Path, replay_path: Path,
    expected_commit: str,
) -> dict[str, Any]:
    from main.multilink_ellipsoid.pi05_palm_l6_source_selection import (
        RESULT_SCHEMA, file_sha256, payload_sha256, scientific_view,
    )
    from main.multilink_ellipsoid.shadow import allocation_record

    identity = _git_identity(repo_root, expected_commit)
    producer = _load(producer_path)
    replay = _load(replay_path)
    for label, value in (("producer", producer), ("replay", replay)):
        _require(value["schema_version"] == RESULT_SCHEMA, f"{label} source schema differs")
        _require(value["status"] == "complete", f"{label} source status differs")
        _require(value["scientific_result"] is True, f"{label} source result differs")
        _require(value["candidate_outcomes_accessed"] is False, f"{label} accessed outcomes")
        _require(value["new_simulation_performed"] is False, f"{label} ran simulation")
        _require(payload_sha256(value) == value["result_payload_sha256"], f"{label} payload differs")
    equal = scientific_view(producer) == scientific_view(replay)
    _require(equal, "pi05 palm/L6 source audits differ")
    summary = producer["summary"]
    groups = summary["eligible_by_target_group"]
    apparatus_pass = bool(
        equal and int(summary["record_count"]) == 1600
        and int(groups["palm"]["eligible_episode_count"]) > 0
        and int(groups["L6"]["eligible_episode_count"]) > 0
        and summary["candidate_outcomes_accessed"] is False
    )
    value = {
        "schema_version": VALIDATION_SCHEMA,
        "status": "complete",
        "scientific_result": True,
        "source": identity,
        "allocation": allocation_record(),
        "producer": {
            "path": str(producer_path),
            "file_sha256": file_sha256(producer_path),
            "result_payload_sha256": producer["result_payload_sha256"],
        },
        "replay": {
            "path": str(replay_path),
            "file_sha256": file_sha256(replay_path),
            "result_payload_sha256": replay["result_payload_sha256"],
        },
        "scientific_view_equal": equal,
        "summary": summary,
        "source_pool_apparatus_pass": apparatus_pass,
        "strict_split_manifest_frozen": False,
        "boundary_collection_authorized": False,
        "training_authorized": False,
        "correction_authorized": False,
    }
    value["validation_payload_sha256"] = payload_sha256(value)
    return value


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--producer", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    value = run(
        repo_root=args.repo_root.resolve(), producer_path=args.producer.resolve(),
        replay_path=args.replay.resolve(), expected_commit=args.expected_commit,
    )
    _atomic_write(args.output.resolve(), value)
    print(json.dumps({
        "summary": value["summary"],
        "source_pool_apparatus_pass": value["source_pool_apparatus_pass"],
        "validation_payload_sha256": value["validation_payload_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

