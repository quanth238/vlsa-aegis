#!/usr/bin/env python3
"""Validate the paired E05 multi-step action-chunk oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from main.multilink_ellipsoid.chunk_oracle import (
    RESULT_SCHEMA, VALIDATION_SCHEMA, load_chunk_oracle_config,
)
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _load, _require,
)


def _hash_without(value, key):
    payload = dict(value); payload.pop(key, None)
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")).hexdigest()


def main() -> int:
    import numpy as np

    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_chunk_oracle_config(args.config.resolve())
    result = _load(args.result.resolve())
    _require(
        result.get("schema_version") == RESULT_SCHEMA
        and result.get("status") == "complete"
        and result.get("scientific_result") is True,
        "chunk-oracle result contract differs",
    )
    _require(
        result.get("result_payload_sha256")
        == _hash_without(result, "result_payload_sha256"),
        "chunk-oracle result payload differs",
    )
    _require(
        result.get("source", {}).get("commit") == args.expected_commit
        and result.get("source", {}).get("dirty") is False
        and result.get("config", {}).get("config_file_sha256")
        == config["config_file_sha256"],
        "chunk-oracle source binding differs",
    )
    _require(
        "H100" in result.get("allocation", {}).get("device", {}).get("name", "")
        and result.get("allocation", {}).get("slurm_job_id"),
        "chunk-oracle allocation differs",
    )
    baselines = result.get("baselines", {})
    _require(
        set(baselines) == {"2", "5"}
        and all(item.get("verified_safe") is False for item in baselines.values()),
        "chunk-oracle baseline crossing differs",
    )
    arms = result.get("arms", {})
    expected_arms = {item["arm_id"]: item for item in config["arms"]}
    _require(set(arms) == set(expected_arms), "chunk-oracle arm set differs")
    passing = []
    for arm_id, registered in expected_arms.items():
        arm = arms[arm_id]
        expected_count = int(config["candidate_family"][
            "one_shot_expected_candidate_count"
            if registered["family"] == "one_shot"
            else "distributed_expected_candidate_count"
        ])
        records = arm.get("candidates", [])
        _require(
            arm.get("horizon") == registered["horizon"]
            and arm.get("family") == registered["family"]
            and arm.get("candidate_count") == expected_count
            and len(records) == expected_count,
            "chunk-oracle candidate population differs: %s" % arm_id,
        )
        safe = []
        baseline_chunk = np.asarray(
            baselines[str(registered["horizon"])]["chunk"], dtype=np.float64
        )
        for index, record in enumerate(records):
            margins = np.asarray(record["minimum_substep_clearance_m"], dtype=np.float64)
            recomputed = bool(
                margins.shape == (7,)
                and np.all(margins >= 0.0)
                and int(record["raw_protected_contact_count"]) == 0
                and float(record["maximum_per_step_obstacle_l1_displacement_m"])
                <= float(config["verification"]["maximum_per_step_obstacle_l1_displacement_m"])
            )
            _require(
                record["candidate_index"] == index
                and record["verified_safe"] is recomputed,
                "chunk-oracle candidate decision differs: %s" % arm_id,
            )
            chunk = np.asarray(record["chunk"], dtype=np.float64)
            _require(
                chunk.shape == baseline_chunk.shape
                and np.max(np.abs(chunk[:, :3])) <= 1.0 + 1.0e-12,
                "chunk-oracle action bounds differ: %s" % arm_id,
            )
            correction = chunk[:, :3] - baseline_chunk[:, :3]
            if registered["family"] == "one_shot":
                _require(
                    np.allclose(correction[1:], 0.0, rtol=0.0, atol=1.0e-12),
                    "one-shot suffix was modified",
                )
            else:
                _require(
                    np.allclose(np.sum(correction, axis=0), 0.0, rtol=0.0, atol=1.0e-12),
                    "distributed endpoint preservation differs",
                )
            if recomputed:
                safe.append(record)
        _require(
            len(safe) == arm["verified_safe_candidate_count"],
            "chunk-oracle safe count differs: %s" % arm_id,
        )
        if safe:
            passing.append(arm_id)
            smallest = min(
                safe,
                key=lambda item: (
                    item["objective_chunk_correction_l2"], item["candidate_index"]
                ),
            )
            _require(
                arm["smallest_verified_safe_chunk"]["candidate_index"]
                == smallest["candidate_index"],
                "chunk-oracle smallest safe candidate differs",
            )
        else:
            _require(
                arm["smallest_verified_safe_chunk"] is None,
                "chunk-oracle empty safe arm has a selected candidate",
            )
    passing = sorted(passing)
    selected = result.get("selected_safe_chunk")
    execution = result.get("executed_selected_chunk")
    execution_pass = bool(
        selected is not None
        and isinstance(execution, dict)
        and execution.get("all_steps_match") is True
        and all(execution.get("per_step_clone_execution_match", []))
        and execution.get("clone_next_state_sha256")
        == execution.get("executed_next_state_sha256")
    )
    any_safe = bool(passing)
    decision = {
        "verified_safe_chunk_exists": any_safe,
        "passing_arm_ids": passing,
        "execution_fidelity_pass": execution_pass,
        "larger_chunk_region_collection_authorized": bool(any_safe and execution_pass),
        "neural_training_authorized": False,
        "closed_loop_e05_authorized": False,
        "registered_chunk_families_exhausted_without_recovery": bool(not any_safe),
    }
    _require(result.get("decision") == decision, "chunk-oracle decision differs")
    validation = {
        "schema_version": VALIDATION_SCHEMA,
        "status": "valid",
        "source_commit": args.expected_commit,
        "result_file_sha256": _file_sha256(args.result.resolve()),
        "result_payload_sha256": result["result_payload_sha256"],
        "arm_candidate_counts": {
            key: value["candidate_count"] for key, value in arms.items()
        },
        "arm_safe_candidate_counts": {
            key: value["verified_safe_candidate_count"] for key, value in arms.items()
        },
        "passing_arm_ids": passing,
        "execution_fidelity_pass": execution_pass,
        "larger_chunk_region_collection_authorized": bool(any_safe and execution_pass),
        "neural_training_authorized": False,
        "closed_loop_e05_authorized": False,
    }
    _atomic_write(args.output.resolve(), validation)
    print(json.dumps(validation, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
