#!/usr/bin/env python3
"""Validate exact replay of the inference-only learned-risk SQP diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional, Sequence

from main.multilink_ellipsoid.learned_risk_sqp import (
    RESULT_SCHEMA, VALIDATION_SCHEMA, payload_sha256,
)
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _load, _require,
)


def validate(producer_path: Path, replay_path: Path) -> dict[str, Any]:
    producer = _load(producer_path)
    replay = _load(replay_path)
    for label, value, replica in (
        ("producer", producer, "producer"), ("replay", replay, "replay"),
    ):
        _require(value.get("schema_version") == RESULT_SCHEMA,
                 "%s schema differs" % label)
        _require(value.get("status") == "complete",
                 "%s status differs" % label)
        _require(value.get("scientific_result") is True,
                 "%s science differs" % label)
        _require(value.get("replica") == replica,
                 "%s replica differs" % label)
        _require(
            value.get("result_payload_sha256")
            == payload_sha256(value, "result_payload_sha256"),
            "%s payload differs" % label,
        )
        _require(value.get("candidate_future_rollout_count") == 0,
                 "%s future rollout count differs" % label)
        _require(value.get("QP_action_executed") is False,
                 "%s QP execution differs" % label)
        _require(value.get("control_authorized") is False,
                 "%s control authorization differs" % label)
    _require(
        replay.get("producer_binding", {}).get("file_sha256")
        == _file_sha256(producer_path),
        "learned-risk SQP producer binding differs",
    )
    _require(producer["scientific_view"] == replay["scientific_view"],
             "learned-risk SQP scientific view differs")
    arms = producer["arms"]
    _require(set(arms) == {"EE_only", "EE_palm_L5"},
             "learned-risk SQP arms differ")
    _require(arms["EE_only"]["constraint_rows"] == [0],
             "learned-risk SQP EE rows differ")
    _require(arms["EE_palm_L5"]["constraint_rows"] == [0, 1, 2, 3, 4],
             "learned-risk SQP multi rows differ")
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
            "exact_query_state_replay": True,
            "exact_serialized_critic_and_SQP_replay": True,
            "same_terminal_nominal": True,
            "EE_only_and_multi_constraint_arms": True,
            "candidate_future_rollout_count_zero": True,
            "QP_action_execution_false": True,
            "diagnostic_only": True,
        },
        "verdict": {
            "nominal_predicted_risk_by_row": producer[
                "nominal_predicted_risk_by_row"
            ],
            "arms": arms,
        },
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
    value = validate(args.producer.resolve(), args.replay.resolve())
    _atomic_write(args.output.resolve(), value)
    print(json.dumps({
        "status": value["status"],
        "EE_only_feasible": value["verdict"]["arms"]["EE_only"][
            "nonlinear_constraints_satisfied"
        ],
        "multi_feasible": value["verdict"]["arms"]["EE_palm_L5"][
            "nonlinear_constraints_satisfied"
        ],
        "validation_payload_sha256": value["validation_payload_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
