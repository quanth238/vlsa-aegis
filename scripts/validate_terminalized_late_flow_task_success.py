#!/usr/bin/env python3
"""Validate exact action replay for the late-flow task-success pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from main.multilink_ellipsoid.terminalized_task_success import (
    ARM_NAMES,
    RESULT_SCHEMA,
    VALIDATION_SCHEMA,
    payload_sha256,
)
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write,
    _file_sha256,
    _load,
    _require,
)


def validate(
    producer_path: Path, replay_path: Path,
) -> dict[str, Any]:
    producer = _load(producer_path)
    replay = _load(replay_path)
    for label, value, replica in (
        ("producer", producer, "producer"),
        ("replay", replay, "replay"),
    ):
        _require(value.get("schema_version") == RESULT_SCHEMA, "%s schema differs" % label)
        _require(value.get("status") == "complete", "%s status differs" % label)
        _require(value.get("scientific_result") is True, "%s result is not scientific" % label)
        _require(value.get("replica") == replica, "%s replica differs" % label)
        _require(
            value.get("result_payload_sha256")
            == payload_sha256(value, "result_payload_sha256"),
            "%s payload differs" % label,
        )
        _require(value.get("task_success_is_distinct_from_safe_terminal") is True,
                 "%s task-success semantics differ" % label)
        _require(value.get("population_claim_authorized") is False,
                 "%s population claim differs" % label)
        _require(value.get("formal_safety_claim") is False,
                 "%s formal claim differs" % label)
    _require(
        producer["config"]["config_file_sha256"]
        == replay["config"]["config_file_sha256"],
        "task-success config file differs",
    )
    _require(
        producer["config"]["config_payload_sha256"]
        == replay["config"]["config_payload_sha256"],
        "task-success config payload differs",
    )
    _require(
        producer["registered_selected_action_pilot"]
        == replay["registered_selected_action_pilot"],
        "task-success selected-action pilot binding differs",
    )
    _require(
        replay.get("producer_binding", {}).get("file_sha256")
        == _file_sha256(producer_path),
        "task-success producer file binding differs",
    )
    _require(
        replay.get("producer_binding", {}).get("payload_sha256")
        == producer["result_payload_sha256"],
        "task-success producer payload binding differs",
    )
    _require(
        producer["scientific_view"] == replay["scientific_view"],
        "task-success scientific view differs",
    )
    view = producer["scientific_view"]
    arms = view["arms"]
    _require(view["arm_count"] == 3, "task-success arm count differs")
    _require([row["arm"] for row in arms] == list(ARM_NAMES), "task-success arm order differs")
    _require(all(isinstance(row["native_task_success"], bool) for row in arms),
             "task-success outcome is missing")
    _require(all(isinstance(row["raw_robot_contact_pass"], bool) for row in arms),
             "task-success contact outcome is missing")
    _require(all(isinstance(row["paper_CAR_pass"], bool) for row in arms),
             "task-success CAR outcome is missing")
    _require(all(row["action_count"] >= 185 for row in arms),
             "task-success continuation was not executed")
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
            "exact_scientific_view_replay": True,
            "three_frozen_arms_complete": True,
            "native_task_outcome_recorded_for_every_arm": True,
            "raw_contact_and_CAR_recorded_for_every_arm": True,
            "task_success_not_inferred_from_safe_terminal": True,
            "population_claim_authorized": False,
            "formal_safety_claim": False,
        },
        "verdict": {
            "task_success_count": int(view["task_success_count"]),
            "collision_free_task_success_count": int(
                view["collision_free_task_success_count"]
            ),
            "arms": arms,
        },
    }
    validation["validation_payload_sha256"] = payload_sha256(
        validation, "validation_payload_sha256"
    )
    return validation


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--producer", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    value = validate(args.producer.resolve(), args.replay.resolve())
    _atomic_write(args.output.resolve(), value)
    print(json.dumps({
        "status": value["status"],
        "task_success_count": value["verdict"]["task_success_count"],
        "collision_free_task_success_count": value["verdict"]["collision_free_task_success_count"],
        "validation_payload_sha256": value["validation_payload_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
