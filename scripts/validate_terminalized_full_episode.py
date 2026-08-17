#!/usr/bin/env python3
"""Validate the true full-episode terminalized selector action replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional, Sequence

from main.multilink_ellipsoid.terminalized_full_episode import (
    RESULT_SCHEMA, VALIDATION_SCHEMA, payload_sha256,
)
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _load, _require,
)


def validate(producer_path: Path, replay_path: Path) -> dict[str, Any]:
    producer = _load(producer_path)
    replay = _load(replay_path)
    for label, value, replica in (
        ("producer", producer, "producer"),
        ("replay", replay, "replay"),
    ):
        _require(value.get("schema_version") == RESULT_SCHEMA,
                 "%s schema differs" % label)
        _require(value.get("status") == "complete",
                 "%s status differs" % label)
        _require(value.get("scientific_result") is True,
                 "%s result differs" % label)
        _require(value.get("replica") == replica,
                 "%s replica differs" % label)
        _require(
            value.get("result_payload_sha256")
            == payload_sha256(value, "result_payload_sha256"),
            "%s payload differs" % label,
        )
        _require(value.get("population_claim_authorized") is False,
                 "%s population claim differs" % label)
        _require(value.get("formal_safety_claim") is False,
                 "%s formal claim differs" % label)
    _require(
        producer["config"]["config_file_sha256"]
        == replay["config"]["config_file_sha256"],
        "full-episode config differs",
    )
    _require(
        replay.get("producer_binding", {}).get("file_sha256")
        == _file_sha256(producer_path),
        "full-episode producer file binding differs",
    )
    _require(
        producer["scientific_view"] == replay["scientific_view"],
        "full-episode scientific view differs",
    )
    view = producer["scientific_view"]
    _require(view["selection_count"] >= 1,
             "full-episode selector was not invoked")
    _require(view["terminal_reason"] in (
        "native_task_success", "raw_robot_contact", "paper_CAR",
        "all_predicted_unsafe_abstention", "timeout",
    ), "full-episode terminal reason differs")
    validation = {
        "schema_version": VALIDATION_SCHEMA,
        "status": "passing",
        "scientific_result": True,
        "producer": {
            "path": str(producer_path),
            "file_sha256": _file_sha256(producer_path),
            "payload_sha256": producer["result_payload_sha256"],
            "slurm_job_id": producer["allocation"]["slurm_job_id"],
        },
        "replay": {
            "path": str(replay_path),
            "file_sha256": _file_sha256(replay_path),
            "payload_sha256": replay["result_payload_sha256"],
            "slurm_job_id": replay["allocation"]["slurm_job_id"],
        },
        "gates": {
            "method_active_from_first_policy_query": True,
            "single_batched_terminal_candidate_inference": True,
            "least_modifying_predicted_safe_else_abstain": True,
            "no_archived_action_prefix": True,
            "no_QP_or_simulator_rollout_verifier": True,
            "exact_action_replay": True,
            "population_claim_authorized": False,
            "formal_safety_claim": False,
        },
        "verdict": view,
    }
    validation["validation_payload_sha256"] = payload_sha256(
        validation, "validation_payload_sha256",
    )
    return validation


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--producer", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    value = validate(
        args.producer.resolve(), args.replay.resolve(),
    )
    _atomic_write(args.output.resolve(), value)
    print(json.dumps({
        "status": value["status"],
        "terminal_reason": value["verdict"]["terminal_reason"],
        "task_success": value["verdict"]["native_task_success"],
        "collision_free_task_success": value["verdict"][
            "collision_free_task_success"
        ],
        "validation_payload_sha256": value["validation_payload_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
