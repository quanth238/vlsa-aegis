#!/usr/bin/env python3
"""Validate exact replay for the opened tight-prefix full-episode pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from main.multilink_ellipsoid.tight_prefix_full_episode import (
    RESULT_SCHEMA, VALIDATION_SCHEMA, file_sha256, payload_sha256,
)
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _load, _require,
)


def validate(producer_path: Path, replay_path: Path) -> dict:
    producer = _load(producer_path)
    replay = _load(replay_path)
    for value, replica in ((producer, "producer"), (replay, "replay")):
        _require(
            value.get("schema_version") == RESULT_SCHEMA
            and value.get("status") == "complete"
            and value.get("replica") == replica
            and value.get("result_payload_sha256")
            == payload_sha256(value, "result_payload_sha256"),
            "tight-prefix full-episode result differs",
        )
    _require(
        producer["scientific_view"] == replay["scientific_view"],
        "tight-prefix full-episode scientific view differs",
    )
    episode = producer["episode"]
    _require(
        int(episode["selection_count"]) > 0
        and int(episode["abstention_count"]) == 0,
        "tight-prefix full-episode selector invocation differs",
    )
    _require(
        all(
            row.get("selected_candidate") is not None
            and row.get("abstained") is False
            and int(row.get("candidate_count", 0)) == 13
            for row in episode["selection_history"]
        ),
        "tight-prefix full-episode selection history differs",
    )
    value = {
        "schema_version": VALIDATION_SCHEMA,
        "status": "validated_exact_opened_full_episode",
        "scientific_result": True,
        "producer_file_sha256": file_sha256(producer_path),
        "producer_payload_sha256": producer["result_payload_sha256"],
        "replay_file_sha256": file_sha256(replay_path),
        "replay_payload_sha256": replay["result_payload_sha256"],
        "scientific_view_exact": True,
        "case_id": producer["case_id"],
        "action_count": int(episode["action_count"]),
        "selection_count": int(episode["selection_count"]),
        "native_task_success": bool(episode["native_task_success"]),
        "raw_robot_contact_pass": bool(episode["raw_robot_contact_pass"]),
        "paper_CAR_pass": bool(episode["paper_CAR_pass"]),
        "collision_free_task_success": bool(
            episode["collision_free_task_success"]
        ),
        "terminal_reason": str(episode["terminal_reason"]),
        "opened_single_case_mechanism_only": True,
        "population_claim_authorized": False,
        "formal_safety_claim": False,
    }
    value["validation_payload_sha256"] = payload_sha256(
        value, "validation_payload_sha256",
    )
    return value


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--producer", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    value = validate(args.producer.resolve(), args.replay.resolve())
    _atomic_write(args.output.resolve(), value)
    print(json.dumps(value, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
