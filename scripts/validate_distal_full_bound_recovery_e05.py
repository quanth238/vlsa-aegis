#!/usr/bin/env python3
"""Independently validate the E05 full-bound recovery receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from main.multilink_ellipsoid.full_bound_recovery import (
    RESULT_SCHEMA, VALIDATION_SCHEMA, load_full_bound_config,
)
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _load, _require,
)


def _hash_without(value, key):
    payload = dict(value)
    payload.pop(key, None)
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_full_bound_config(args.config.resolve())
    result = _load(args.result.resolve())
    _require(
        result.get("schema_version") == RESULT_SCHEMA
        and result.get("status") == "complete"
        and result.get("scientific_result") is True,
        "full-bound result contract differs",
    )
    _require(
        result.get("result_payload_sha256")
        == _hash_without(result, "result_payload_sha256"),
        "full-bound result payload differs",
    )
    _require(
        result.get("source", {}).get("commit") == args.expected_commit
        and result.get("source", {}).get("dirty") is False,
        "full-bound source differs",
    )
    _require(
        result.get("config", {}).get("config_file_sha256")
        == config["config_file_sha256"],
        "full-bound config binding differs",
    )
    allocation = result.get("allocation", {})
    _require(
        allocation.get("slurm_job_id")
        and "H100" in allocation.get("device", {}).get("name", ""),
        "full-bound allocation differs",
    )
    search = result.get("global_search", {})
    _require(
        search.get("candidate_count") == 731
        and isinstance(search.get("candidates"), list)
        and len(search["candidates"]) == 731,
        "full-bound candidate population differs",
    )
    recomputed_safe = [
        item for item in search["candidates"]
        if item.get("D_opt_proxy_safe") is True
        and item.get("D_sim_raw_safe") is True
    ]
    _require(
        len(recomputed_safe) == search.get("verified_safe_candidate_count"),
        "full-bound safe count differs",
    )
    decision = result.get("decision", {})
    physical = bool(recomputed_safe)
    qp_candidate = result.get("qp", {}).get("candidate")
    qp_pass = bool(
        result.get("qp", {}).get("valid") is True
        and isinstance(qp_candidate, dict)
        and qp_candidate.get("D_opt_proxy_safe") is True
        and qp_candidate.get("D_sim_raw_safe") is True
    )
    executed = result.get("executed_verified_recovery")
    execution = bool(
        isinstance(executed, dict)
        and executed.get("exact_clone_execution_match") is True
        and executed.get("clone_next_state_sha256")
        == executed.get("executed_next_state_sha256")
    )
    _require(
        decision == {
            "physical_full_bound_recovery_exists": physical,
            "finite_difference_qp_recovery_pass": qp_pass,
            "execution_fidelity_pass": execution,
            "larger_region_collection_authorized": bool(physical and execution),
            "closed_loop_e05_authorized": False,
            "multi_step_required_before_larger_region_collection": bool(not physical),
        },
        "full-bound decision differs",
    )
    validation = {
        "schema_version": VALIDATION_SCHEMA,
        "status": "valid",
        "source_commit": args.expected_commit,
        "result_file_sha256": _file_sha256(args.result.resolve()),
        "result_payload_sha256": result["result_payload_sha256"],
        "candidate_count": len(search["candidates"]),
        "verified_safe_candidate_count": len(recomputed_safe),
        "physical_full_bound_recovery_exists": physical,
        "finite_difference_qp_recovery_pass": qp_pass,
        "execution_fidelity_pass": execution,
        "larger_region_collection_authorized": bool(physical and execution),
        "closed_loop_e05_authorized": False,
    }
    _atomic_write(args.output.resolve(), validation)
    print(json.dumps(validation, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
