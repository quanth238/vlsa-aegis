#!/usr/bin/env python3
"""Validate exact replay for the nominal-first full-episode pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from main.multilink_ellipsoid.tight_prefix_full_episode import (
    file_sha256,
    payload_sha256,
)
from main.multilink_ellipsoid.tight_prefix_nominal_first_full_episode import (
    RESULT_SCHEMA,
    VALIDATION_SCHEMA,
)
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write,
    _load,
    _require,
)


def _validated_result(value: Mapping[str, Any], replica: str) -> None:
    _require(
        value.get("schema_version") == RESULT_SCHEMA
        and value.get("status") == "complete"
        and value.get("scientific_result") is True
        and value.get("replica") == replica
        and value.get("result_payload_sha256")
        == payload_sha256(value, "result_payload_sha256"),
        "nominal-first validated result differs",
    )


def validate(producer_path: Path, replay_path: Path) -> dict[str, Any]:
    producer = _load(producer_path)
    replay = _load(replay_path)
    _validated_result(producer, "producer")
    _validated_result(replay, "replay")
    _require(
        producer["scientific_view"] == replay["scientific_view"],
        "nominal-first scientific view differs",
    )
    episode = producer["episode"]
    history = episode["selection_history"]
    _require(
        len(history) == int(episode["selection_count"])
        and int(episode["action_count"]) <= 300,
        "nominal-first episode count differs",
    )
    warning_count = 0
    critic_count = 0
    intervention_count = 0
    nominal_count = 0
    for row in history:
        warning = bool(row["warning_triggered"])
        critic = bool(row["critic_invoked"])
        _require(warning is critic,
                 "nominal-first warning/critic receipt differs")
        warning_count += int(warning)
        critic_count += int(critic)
        if not warning:
            _require(
                int(row["candidate_count"]) == 1
                and row["selected_candidate"] == "nominal"
                and row["abstained"] is False
                and row["intervened"] is False
                and float(row["selected_effective_correction_l2_action"])
                == 0.0,
                "nominal-first outside-warning selection differs",
            )
        else:
            _require(int(row["candidate_count"]) == 13,
                     "nominal-first warning bank differs")
            if row["abstained"] is False:
                _require(
                    row["selected_endpoint_preserved"] is True
                    and max(abs(float(value)) for value in row[
                        "selected_effective_correction_sum_xyz"
                    ]) <= 1.0e-12,
                    "nominal-first endpoint receipt differs",
                )
        intervention_count += int(bool(row["intervened"]))
        nominal_count += int(row["selected_candidate"] == "nominal")
    view = producer["scientific_view"]
    result = {
        "schema_version": VALIDATION_SCHEMA,
        "status": "validated_exact_opened_nominal_first_full_episode",
        "scientific_result": True,
        "case_id": view["case_id"],
        "scientific_view_exact": True,
        "producer_file_sha256": file_sha256(producer_path),
        "producer_payload_sha256": producer["result_payload_sha256"],
        "replay_file_sha256": file_sha256(replay_path),
        "replay_payload_sha256": replay["result_payload_sha256"],
        "terminal_reason": view["terminal_reason"],
        "action_count": view["action_count"],
        "selection_count": view["selection_count"],
        "warning_count": warning_count,
        "critic_invocation_count": critic_count,
        "intervention_count": intervention_count,
        "nominal_selection_count": nominal_count,
        "localized_intervention": bool(
            intervention_count < int(view["selection_count"])
        ),
        "raw_robot_contact_pass": view["raw_robot_contact_pass"],
        "paper_CAR_pass": view["paper_CAR_pass"],
        "native_task_success": view["native_task_success"],
        "collision_free_task_success": view["collision_free_task_success"],
        "opened_single_case_mechanism_only": True,
        "population_claim_authorized": False,
        "formal_safety_claim": False,
    }
    result["validation_payload_sha256"] = payload_sha256(
        result, "validation_payload_sha256",
    )
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--producer", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = validate(args.producer.resolve(), args.replay.resolve())
    _atomic_write(args.output.resolve(), result)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
