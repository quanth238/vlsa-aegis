#!/usr/bin/env python3
"""Validate paired H100 late-flow waypoint-search replicas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional, Sequence

from main.multilink_ellipsoid.late_flow_waypoint_search import (
    RESULT_SCHEMA, VALIDATION_SCHEMA, payload_sha256,
)
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _load, _require,
)


def validate(producer_path: Path, replica_path: Path) -> dict[str, Any]:
    producer = _load(producer_path)
    replica = _load(replica_path)
    for label, value, expected_replica in (
        ("producer", producer, "producer"),
        ("replica", replica, "replica"),
    ):
        _require(value.get("schema_version") == RESULT_SCHEMA,
                 "%s schema differs" % label)
        _require(value.get("status") == "complete",
                 "%s status differs" % label)
        _require(value.get("diagnostic_only") is True,
                 "%s scope differs" % label)
        _require(value.get("replica") == expected_replica,
                 "%s replica differs" % label)
        _require(
            value.get("result_payload_sha256")
            == payload_sha256(value, "result_payload_sha256"),
            "%s payload differs" % label,
        )
        _require(value.get("new_label_count") == 0,
                 "%s label count differs" % label)
        _require(value.get("simulator_candidate_rollout_count") == 0,
                 "%s simulator rollout differs" % label)
        _require(value.get("action_executed") is False,
                 "%s action execution differs" % label)
        _require(value.get("control_authorized") is False,
                 "%s control authorization differs" % label)
        _require(value["summary"]["candidate_count"] == 505,
                 "%s candidate count differs" % label)
    _require(producer["scientific_view"] == replica["scientific_view"],
             "late-flow waypoint scientific view differs")
    summary = producer["summary"]
    validation = {
        "schema_version": VALIDATION_SCHEMA,
        "status": "passing",
        "diagnostic_only": True,
        "producer": {
            "path": str(producer_path),
            "file_sha256": _file_sha256(producer_path),
            "payload_sha256": producer["result_payload_sha256"],
            "slurm_job_id": producer["allocation"]["slurm_job_id"],
        },
        "replica": {
            "path": str(replica_path),
            "file_sha256": _file_sha256(replica_path),
            "payload_sha256": replica["result_payload_sha256"],
            "slurm_job_id": replica["allocation"]["slurm_job_id"],
        },
        "gates": {
            "exact_query_state": True,
            "exact_terminal_nominal": True,
            "exact_candidate_population_and_predictions": True,
            "candidate_count_505": True,
            "simulator_candidate_rollout_count_zero": True,
            "action_execution_false": True,
        },
        "verdict": {
            "predicted_physical_safe_count": summary[
                "predicted_physical_safe_count"
            ],
            "predicted_proxy_inclusive_safe_count": summary[
                "predicted_proxy_inclusive_safe_count"
            ],
            "minimum_predicted_risk_by_row": summary[
                "minimum_predicted_risk_by_row"
            ],
            "minimum_physical_risk_candidate": summary[
                "minimum_physical_risk_candidate"
            ],
            "minimum_proxy_inclusive_risk_candidate": summary[
                "minimum_proxy_inclusive_risk_candidate"
            ],
            "selected_physical_safe_minimum_change": summary[
                "selected_physical_safe_minimum_change"
            ],
        },
    }
    validation["validation_payload_sha256"] = payload_sha256(
        validation, "validation_payload_sha256",
    )
    return validation


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--producer", type=Path, required=True)
    parser.add_argument("--replica", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    value = validate(
        args.producer.resolve(), args.replica.resolve(),
    )
    _atomic_write(args.output.resolve(), value)
    print(json.dumps({
        "status": value["status"],
        "physical_safe_count": value["verdict"]["predicted_physical_safe_count"],
        "proxy_inclusive_safe_count": value["verdict"][
            "predicted_proxy_inclusive_safe_count"
        ],
        "validation_payload_sha256": value["validation_payload_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
