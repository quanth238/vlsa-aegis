#!/usr/bin/env python3
"""Independently validate the E05 query-aligned Monte Carlo risk artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Optional, Sequence

from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _git_identity, _load, _require, _sha256,
)


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def validate(
    *, repo_root: Path, result_path: Path, archived_path: Path,
    experiment_config_path: Path, producer_commit: str, validator_commit: str,
) -> dict[str, Any]:
    import numpy as np

    from main.multilink_ellipsoid.pure_backup import orthonormal_local_frame
    from main.multilink_ellipsoid.query_action_risk import (
        RESULT_SCHEMA, candidate_definitions, combine_row_minima, exact_safe,
        load_config, risk_from_row_minimum,
    )

    validator_source = _git_identity(repo_root, validator_commit)
    config = load_config(experiment_config_path)
    result = _load(result_path)
    archived = _load(archived_path)
    _require(result["schema_version"] == RESULT_SCHEMA, "query-risk result schema differs")
    _require(result["status"] == "complete", "query-risk result is incomplete")
    _require(result["scientific_result"] is True, "query-risk result is not scientific")
    _require(result["execution_mode"] == "full_diagnostic", "query-risk execution mode differs")
    _require(result["source"]["commit"] == producer_commit, "query-risk producer commit differs")
    _require(result["config"]["config_payload_sha256"] == config["config_payload_sha256"],
             "query-risk config binding differs")
    _require(_file_sha256(archived_path) == result["archived_table1"]["file_sha256"],
             "query-risk Table-1 file differs")
    _require(archived["result_payload_sha256"] == result["archived_table1"]["result_payload_sha256"],
             "query-risk Table-1 payload differs")
    _require(result["archived_table1"]["read_only"] is True, "query-risk Table-1 mutability differs")
    _require(result["policy_query"] == archived["policy_queries"][37],
             "query-risk policy query differs")
    nominal = np.asarray(
        [archived["actions"][step]["executed"] for step in range(185, 190)],
        dtype=np.float64,
    )
    _require(np.array_equal(nominal, np.asarray(result["nominal_five_action_chunk"])),
             "query-risk nominal five-action chunk differs")
    _require(hashlib.sha256(nominal.tobytes()).hexdigest()
             == result["nominal_five_action_chunk_sha256"],
             "query-risk nominal hash differs")
    expected = candidate_definitions(
        nominal, orthonormal_local_frame(result["state"]["local_frame"]["normal"]), config
    )
    candidates = result["candidates"]
    _require(len(candidates) == len(expected) == 37, "query-risk candidate count differs")
    buffer_m = float(config["risk_target"]["safety_buffer_m"])
    car_limit = float(config["risk_target"]["paper_car_threshold_m"])
    safe_count = 0
    timeout_count = 0
    physical_count = 0
    witness_counts = [0] * 7
    for observed, definition in zip(candidates, expected):
        for key in (
            "name", "order", "direction", "sign", "temporal_profile",
            "requested_correction_l2_action", "applied_correction_l2_action",
            "clipped", "actions",
        ):
            _require(observed[key] == definition[key],
                     "query-risk candidate identity differs: %s" % key)
        prefix_trace = np.asarray(observed["prefix"]["clearance_trace_m"], dtype=np.float64)
        _require(prefix_trace.shape == (126, 7), "query-risk prefix trace shape differs")
        prefix_row = np.min(prefix_trace[1:], axis=0)
        _require(np.array_equal(prefix_row, np.asarray(observed["prefix"]["row_minimum_clearance_m"])),
                 "query-risk prefix row minimum differs")
        _require(observed["candidate_prefix_risk"]
                 == risk_from_row_minimum(prefix_row, buffer_m),
                 "query-risk prefix target differs")
        parts = [prefix_row.tolist()]
        backup_row = observed["backup"]["row_minimum_clearance_m"]
        if backup_row is not None:
            parts.append(backup_row)
            _require(observed["backup_risk"]
                     == risk_from_row_minimum(backup_row, buffer_m),
                     "query-risk backup target differs")
        else:
            _require(observed["backup_risk"] is None, "query-risk absent backup differs")
        combined = combine_row_minima(*parts)
        _require(combined == observed["combined_row_minimum_clearance_m"],
                 "query-risk combined row minimum differs")
        _require(observed["combined_risk"] == risk_from_row_minimum(combined, buffer_m),
                 "query-risk combined target differs")
        status = observed["terminal_status"]
        _require(status in config["risk_target"]["terminal_statuses"],
                 "query-risk terminal status differs")
        if status == "SAFE_TERMINAL":
            hold = observed["backup"]["terminal_hold"]
            _require(hold is not None, "query-risk safe terminal lacks hold")
            _require(hold["sample_count"] == 251, "query-risk terminal hold length differs")
            _require(hold["minimum_clearance_m"] >= buffer_m,
                     "query-risk terminal hold violates proxy boundary")
            _require(hold["protected_contact_count"] == 0,
                     "query-risk terminal hold has protected contact")
            _require(hold["maximum_active_obstacle_l1_displacement_m"] <= car_limit,
                     "query-risk terminal hold fails CAR")
        if status == "UNKNOWN_TIMEOUT":
            timeout_count += 1
            _require(observed["exact_safe"] is False,
                     "query-risk timeout was labeled safe")
        prefix_physical = bool(
            observed["prefix"]["protected_contact_count"] > 0
            or observed["prefix"]["maximum_active_obstacle_l1_displacement_m"] > car_limit
        )
        backup_physical = bool(
            observed["backup"]["protected_contact_count"] > 0
            or observed["backup"]["maximum_active_obstacle_l1_displacement_m"] > car_limit
        )
        _require(observed["physical_veto"] == (prefix_physical or backup_physical),
                 "query-risk physical veto differs")
        _require(observed["exact_safe"] == exact_safe(observed),
                 "query-risk exact-safe label differs")
        safe_count += int(observed["exact_safe"])
        physical_count += int(observed["physical_veto"])
        witness_counts[int(np.argmin(combined))] += 1
        _require(math.isfinite(max(observed["combined_risk"])),
                 "query-risk target is nonfinite")
    summary = result["summary"]
    _require(summary["safe_candidate_count"] == safe_count == 2,
             "query-risk safe count differs")
    _require(summary["unsafe_candidate_count"] == 35,
             "query-risk unsafe count differs")
    _require(summary["unknown_timeout_count"] == timeout_count == 3,
             "query-risk timeout count differs")
    _require(summary["row_active_witness_counts"] == witness_counts == [0, 37, 0, 0, 0, 0, 0],
             "query-risk witness population differs")
    _require(all(result["gates"].values()), "query-risk producer gates did not pass")
    _require(result["training_authorized"] is False,
             "query-risk diagnostic improperly authorizes training")
    _require(result["interpretation"] == "query_aligned_five_action_risk_diagnostic_pass",
             "query-risk interpretation differs")
    validation = {
        "schema_version": "vlsa_distal_query_action_risk_e05_validation.v1",
        "status": "complete",
        "scientific_result": True,
        "producer": {
            "path": str(result_path),
            "file_sha256": _file_sha256(result_path),
            "result_payload_sha256": result["result_payload_sha256"],
            "commit": producer_commit,
            "slurm_job_id": result["allocation"]["slurm_job_id"],
            "host": result["allocation"]["host"],
            "execution_mode": result["allocation"]["execution_mode"],
        },
        "validator_source": validator_source,
        "checks": {
            "immutable_Table1_binding": True,
            "policy_query_and_nominal_chunk_binding": True,
            "candidate_identity_and_order": True,
            "prefix_trace_and_risk_composition": True,
            "registered_terminal_semantics": True,
            "physical_veto_recomputed": True,
            "timeout_never_safe": True,
            "mixed_support": safe_count > 0 and safe_count < len(candidates),
            "all_active_witnesses_are_L5_row_1": witness_counts == [0, 37, 0, 0, 0, 0, 0],
        },
        "counts": {
            "candidates": len(candidates),
            "safe": safe_count,
            "unsafe_or_timeout": len(candidates) - safe_count,
            "timeouts": timeout_count,
            "physical_veto": physical_count,
        },
        "interpretation": "validated_nonvacuous_E05_L5_query_aligned_risk_boundary",
        "training_authorized": False,
        "next_gate": "grouped_episode_collection_requires_new_query_boundary_discovery_and_per_row_coverage",
    }
    validation["validation_payload_sha256"] = _sha256(canonical(validation))
    return validation


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--archived", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--producer-commit", required=True)
    parser.add_argument("--validator-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = validate(
        repo_root=args.repo_root.resolve(), result_path=args.result.resolve(),
        archived_path=args.archived.resolve(),
        experiment_config_path=args.experiment_config.resolve(),
        producer_commit=args.producer_commit, validator_commit=args.validator_commit,
    )
    _atomic_write(args.output.resolve(), output)
    print(json.dumps(output, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
