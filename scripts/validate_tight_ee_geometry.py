#!/usr/bin/env python3
"""Validate independent tight-EE geometry replays and raw contacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional, Sequence

from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _git_identity, _load, _require,
)


def validate(
    *, repo_root: Path, config_path: Path, producer_dir: Path,
    replay_dir: Path, producer_commit: str, validator_commit: str,
) -> dict[str, Any]:
    from main.multilink_ellipsoid.tight_ee_geometry import (
        RESULT_SCHEMA, VALIDATION_SCHEMA, canonical, load_config,
        payload_sha256, scientific_view, summarize_records,
    )

    validator_source = _git_identity(repo_root, validator_commit)
    config = load_config(config_path, repo_root=repo_root)
    case_ids = [str(item["case_id"]) for item in config["cohort"]["cases"]]
    producer_records = []
    replay_records = []
    equality = {}
    for case_id in case_ids:
        producer = _load(producer_dir / (case_id + ".json"))
        replay = _load(replay_dir / (case_id + ".json"))
        for name, record in (("producer", producer), ("replay", replay)):
            _require(record.get("schema_version") == RESULT_SCHEMA, f"{name} schema differs")
            _require(record.get("status") == "complete", f"{name} is incomplete")
            _require(record.get("case_id") == case_id, f"{name} case differs")
            _require(
                record["config"]["config_payload_sha256"]
                == config["config_payload_sha256"],
                f"{name} config differs",
            )
            _require(
                record["source"]["commit"] == producer_commit,
                f"{name} producer commit differs",
            )
            _require(
                record["result_payload_sha256"] == payload_sha256(record),
                f"{name} payload differs",
            )
        equal = canonical(scientific_view(producer)) == canonical(scientific_view(replay))
        equality[case_id] = equal
        _require(equal, "tight EE independent scientific replay differs for " + case_id)
        producer_records.append(producer)
        replay_records.append(replay)
    producer_summary = summarize_records(producer_records, config)
    replay_summary = summarize_records(replay_records, config)
    _require(
        canonical(producer_summary) == canonical(replay_summary),
        "tight EE independent summaries differ",
    )
    visualization_case = str(config["visualization"]["case_id"])
    visual_record = next(
        record for record in producer_records if record["case_id"] == visualization_case
    )
    visualization = visual_record.get("visualization")
    _require(visualization is not None, "tight EE producer visualization is missing")
    for key in ("base_image", "overlay", "preview"):
        artifact = visualization[key]
        path = Path(visualization["metadata_path"]).parent / artifact["path"]
        _require(path.is_file(), "tight EE visualization artifact is missing")
    gate_pass = bool(
        all(equality.values())
        and producer_summary["tight_ee_geometry_gate_pass"]
    )
    result = {
        "schema_version": VALIDATION_SCHEMA,
        "status": "pass" if gate_pass else "scientific_no_go",
        "scientific_result": True,
        "claim_scope": config["claim_scope"],
        "validator_source": validator_source,
        "producer_commit": producer_commit,
        "config": config,
        "producer_dir": str(producer_dir),
        "replay_dir": str(replay_dir),
        "independent_scientific_view_equal_by_case": equality,
        "summary": producer_summary,
        "visualization": visualization,
        "tight_ee_geometry_gate_pass": gate_pass,
        "released_proxy_replacement_authorized": gate_pass,
        "model_training_authorized": False,
        "QP_authorized": False,
    }
    result["validation_payload_sha256"] = payload_sha256(result)
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--producer-dir", type=Path, required=True)
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--producer-commit", required=True)
    parser.add_argument("--validator-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = validate(
        repo_root=args.repo_root.resolve(),
        config_path=args.config.resolve(),
        producer_dir=args.producer_dir.resolve(),
        replay_dir=args.replay_dir.resolve(),
        producer_commit=args.producer_commit,
        validator_commit=args.validator_commit,
    )
    _atomic_write(args.output.resolve(), result)
    print(json.dumps({
        "status": result["status"],
        "gate_pass": result["tight_ee_geometry_gate_pass"],
        "payload_sha256": result["validation_payload_sha256"],
    }, sort_keys=True), flush=True)
    return 0 if result["tight_ee_geometry_gate_pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
