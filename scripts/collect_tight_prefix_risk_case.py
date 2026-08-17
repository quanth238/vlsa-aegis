#!/usr/bin/env python3
"""Replay one immutable 13-chunk state with tight five-action risk labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from scripts.audit_distal_compiled_box_risk_target import _evaluate_case
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _git_identity, _load, _require,
)


def collect(
    *, repo_root: Path, config_path: Path, case_index: int,
    expected_commit: str,
) -> dict:
    from main.multilink_ellipsoid.tight_prefix_risk_dataset import (
        CASE_SCHEMA, load_config, payload_sha256,
    )

    config = load_config(config_path, repo_root=repo_root)
    _require(0 <= int(case_index) < len(config["cases"]),
             "tight prefix-risk case index differs")
    validation_binding = config["tight_geometry_validation"]
    validation_path = Path(validation_binding["path"])
    _require(_file_sha256(validation_path) == validation_binding["file_sha256"],
             "tight prefix-risk geometry validation file differs")
    validation = _load(validation_path)
    _require(
        validation["validation_payload_sha256"]
        == validation_binding["validation_payload_sha256"],
        "tight prefix-risk geometry validation payload differs",
    )
    _require(
        validation[validation_binding["required_gate"]] is True,
        "tight prefix-risk geometry validation gate failed",
    )
    selected = dict(config["cases"][int(case_index)])
    selected["candidate_names"] = list(
        config["candidate_bank"]["candidate_names"]
    )
    selected["slab_initialization"] = "query_state_matching_source"
    source = config["source"]
    case = _evaluate_case(
        repo_root=repo_root,
        population_manifest=repo_root / source["population_manifest"],
        geometry_config_path=(
            repo_root / source["legacy_replay_geometry_config"]
        ),
        case_config=selected,
        audit_config={
            "rollout_scope": config["rollout_scope"],
            "gate": config["gate"],
            "exact_group_target": config["exact_group_target"],
        },
    )
    value = {
        "schema_version": CASE_SCHEMA,
        "status": "complete",
        "scientific_result": True,
        "claim_scope": config["claim_scope"],
        "source": _git_identity(repo_root, expected_commit),
        "case_index": int(case_index),
        "case_id": selected["case_id"],
        "split": selected["split"],
        "config_file_sha256": config["config_file_sha256"],
        "config_payload_sha256": config["config_payload_sha256"],
        "case": case,
        "training_scope": "diagnostic_Q_only_only",
        "correction_or_QP_authorized": False,
    }
    value["result_payload_sha256"] = payload_sha256(value)
    return value


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--case-index", type=int, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    value = collect(
        repo_root=args.repo_root.resolve(),
        config_path=args.config.resolve(),
        case_index=args.case_index,
        expected_commit=args.expected_commit,
    )
    _atomic_write(args.output.resolve(), value)
    print(json.dumps({
        "case_id": value["case_id"],
        "split": value["split"],
        "source_replay_exact": value["case"]["source_replay_exact"],
        "candidate_count": len(value["case"]["candidates"]),
        "result_payload_sha256": value["result_payload_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
