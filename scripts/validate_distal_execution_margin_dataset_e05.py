#!/usr/bin/env python3
"""Validate the pretraining E05 execution-margin oracle dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from main.multilink_ellipsoid.execution_margin_nn import load_execution_margin_config
from scripts.collect_distal_execution_margin_dataset_e05 import DATASET_RESULT_SCHEMA
from scripts.replay_distal_three_ellipsoid_multicbf import (
    CASE_ID,
    _atomic_write,
    _file_sha256,
    _load,
    _require,
)


VALIDATION_SCHEMA = "vlsa_distal_execution_margin_dataset_e05_validation.v1"


def _hash_without(value: Mapping[str, Any], key: str) -> str:
    payload = dict(value)
    payload.pop(key, None)
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def validate(
    result_path: Path,
    *,
    expected_commit: str,
    expected_config: Mapping[str, Any],
) -> dict[str, Any]:
    result = _load(result_path)
    _require(
        result.get("schema_version") == DATASET_RESULT_SCHEMA
        and result.get("status") == "complete"
        and result.get("scientific_result") is True
        and result.get("case_id") == CASE_ID
        and result.get("failure") is None
        and result.get("result_payload_sha256")
        == _hash_without(result, "result_payload_sha256"),
        "execution-margin dataset result identity differs",
    )
    _require(result.get("config") == expected_config, "dataset config differs")
    source = result.get("source", {})
    allocation = result.get("allocation", {})
    _require(
        source.get("commit") == expected_commit
        and source.get("dirty") is False
        and str(allocation.get("slurm_job_id", "")).isdigit()
        and "H100" in str(allocation.get("device", {}).get("name", "")),
        "dataset source or H100 allocation differs",
    )
    geometry = result.get("geometry", {})
    _require(
        geometry.get("accepted_robot_and_released_ee", {}).get(
            "distal_ellipsoid_count"
        )
        == 7
        and geometry.get("exact_obstacle_boxes", {}).get("exact_box_count") == 15
        and geometry.get("exact_obstacle_boxes", {}).get("geometric_inflation")
        == 0.0,
        "dataset geometry differs",
    )
    identity = result.get("dataset", {})
    dataset_path = Path(identity.get("path", "")).resolve()
    _require(
        dataset_path.is_file()
        and not dataset_path.is_symlink()
        and dataset_path.parent == result_path.parent.resolve()
        and identity.get("file_sha256") == _file_sha256(dataset_path),
        "dataset file identity differs",
    )
    dataset = _load(dataset_path)
    _require(
        dataset.get("dataset_payload_sha256")
        == _hash_without(dataset, "dataset_payload_sha256")
        == identity.get("payload_sha256")
        and dataset.get("source_commit") == expected_commit
        and len(dataset.get("states", [])) == 13
        and len(dataset.get("records", [])) == identity.get("sample_count")
        == 1125,
        "dataset payload or dimensions differ",
    )
    for step in range(180, 193):
        selected = [item for item in dataset["records"] if item.get("state_step") == step]
        expected_count = expected_config["candidate_set"][
            "expected_candidate_count_by_state"
        ][str(step)]
        _require(
            len(selected) == expected_count
            and {item.get("split") for item in selected}
            <= {"train", "validation", "test"}
            and all(
                len(item.get("feature_vector", [])) == 33
                and len(item.get("current_clearance_m", [])) == 7
                and len(item.get("minimum_substep_clearance_m", [])) == 7
                and isinstance(item.get("D_opt_proxy_safe"), bool)
                and isinstance(item.get("D_sim_raw_safe"), bool)
                for item in selected
            ),
            "dataset state group is incomplete",
        )
    summary = result.get("dataset_summary", {})
    decision = result.get("decision", {})
    expected_gate = bool(
        summary.get("geometry_authority_pass")
        and summary.get("recoverable_exact_proxy_crossing_steps")
        and expected_config["state_groups"]["primary_projection_step"]
        in summary.get("recoverable_exact_proxy_crossing_steps", [])
    )
    _require(
        summary.get("state_count") == 13
        and summary.get("candidate_count_by_state")
        == expected_config["candidate_set"]["expected_candidate_count_by_state"]
        and summary.get("sample_count") == 1125
        and summary.get("oracle_analysis_gate_pass") is expected_gate
        and decision.get("geometry_authority_pass")
        is bool(summary.get("geometry_authority_pass"))
        and decision.get("oracle_analysis_gate_pass") is expected_gate
        and decision.get("neural_training_authorized") is expected_gate,
        "dataset oracle decision differs",
    )
    return {
        "schema_version": VALIDATION_SCHEMA,
        "status": "valid",
        "scientific_result": False,
        "case_id": CASE_ID,
        "source_commit": expected_commit,
        "slurm_job_id": allocation["slurm_job_id"],
        "result_file_sha256": _file_sha256(result_path),
        "result_payload_sha256": result["result_payload_sha256"],
        "dataset_file_sha256": identity["file_sha256"],
        "oracle_analysis_gate_pass": expected_gate,
        "neural_training_authorized": expected_gate,
        "checks": {
            "clean_H100_source": True,
            "immutable_pairing_and_action_ledger": True,
            "thirteen_complete_state_groups": True,
            "D_opt_D_sim_separation": True,
            "geometry_and_recoverable_crossing_gate": True,
        },
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    record = validate(
        args.result.resolve(),
        expected_commit=args.expected_commit,
        expected_config=load_execution_margin_config(args.config.resolve()),
    )
    _atomic_write(args.output.resolve(), record)
    print(json.dumps(record, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
