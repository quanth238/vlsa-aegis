#!/usr/bin/env python3
"""Validate independent top-five exact-verification executions."""

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
    replay_dir: Path, expected_commit: str,
) -> dict[str, Any]:
    from main.multilink_ellipsoid.top5_exact_verified_pilot import (
        RESULT_SCHEMA, VALIDATION_SCHEMA, load_config, payload_sha256,
        scientific_view,
    )

    config = load_config(config_path)
    rows = []
    for case in config["cases"]:
        case_id = case["case_id"]
        producer = _load(producer_dir / (case_id + ".json"))
        replay = _load(replay_dir / (case_id + ".json"))
        for result in (producer, replay):
            _require(
                result.get("schema_version") == RESULT_SCHEMA
                and result.get("case_id") == case_id
                and result.get("source", {}).get("commit") == expected_commit
                and result.get("result_payload_sha256")
                == payload_sha256(result, "result_payload_sha256"),
                "top-five pilot result payload differs",
            )
        exact = scientific_view(producer) == scientific_view(replay)
        rows.append({
            "case_id": case_id,
            "split": case["split"],
            "independent_replay_exact": exact,
            "top_one_freshly_unsafe": bool(producer["top_one_freshly_unsafe"]),
            "verified_safe_selection": bool(producer["verified_safe_selection"]),
            "method_avoids_top_one_collision": bool(
                producer["method_avoids_top_one_collision"]
            ),
            "selected_rank": producer["selected_rank"],
            "selected_candidate": producer["selected_candidate"],
            "attempt_count": len(producer["attempts"]),
        })
    gates = {
        "case_count": len(rows) == 3,
        "independent_replay_exact": all(
            row["independent_replay_exact"] for row in rows
        ),
        "top_one_freshly_unsafe_every_case": all(
            row["top_one_freshly_unsafe"] for row in rows
        ),
        "verified_safe_selection_every_case": all(
            row["verified_safe_selection"] for row in rows
        ),
        "method_avoids_top_one_collision_every_case": all(
            row["method_avoids_top_one_collision"] for row in rows
        ),
        "selected_within_top_five": all(
            row["selected_rank"] is not None
            and int(row["selected_rank"]) <= 5 for row in rows
        ),
    }
    value = {
        "schema_version": VALIDATION_SCHEMA,
        "status": "complete",
        "scientific_result": True,
        "claim_scope": config["claim_scope"],
        "source": _git_identity(repo_root, expected_commit),
        "config_file_sha256": config["config_file_sha256"],
        "config_payload_sha256": config["config_payload_sha256"],
        "cases": rows,
        "gates": gates,
        "strict_gate_pass": all(gates.values()),
        "interpretation": (
            "top5_exact_verified_correction_mechanism_pass"
            if all(gates.values()) else "top5_exact_verified_correction_no_go"
        ),
        "model_only_correction_authorized": False,
        "closed_loop_authorized": False,
    }
    value["validation_payload_sha256"] = payload_sha256(
        value, "validation_payload_sha256",
    )
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
        repo_root=args.repo_root.resolve(), config_path=args.config.resolve(),
        producer_dir=args.producer_dir.resolve(),
        replay_dir=args.replay_dir.resolve(), expected_commit=args.expected_commit,
    )
    _atomic_write(args.output.resolve(), value)
    print(json.dumps({
        "strict_gate_pass": value["strict_gate_pass"],
        "interpretation": value["interpretation"],
        "cases": value["cases"],
        "validation_payload_sha256": value["validation_payload_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
