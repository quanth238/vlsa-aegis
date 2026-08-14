#!/usr/bin/env python3
"""Validate that L5 Monte-Carlo labels bind to final original-AEGIS actions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write,
    _file_sha256,
    _git_identity,
    _load,
    _require,
    _sha256,
)


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _array_equal(left: Any, right: Any) -> bool:
    import numpy as np

    return bool(np.array_equal(
        np.asarray(left, dtype=np.float64),
        np.asarray(right, dtype=np.float64),
    ))


def _check_projection(record: Mapping[str, Any], expected_actions: Any) -> None:
    projection = record["aegis_consistency"]
    _require(projection["enabled"] is True, "released AEGIS projection is disabled")
    _require(_array_equal(projection["proposed_actions"], record["proposed_actions"]),
             "released AEGIS proposal binding differs")
    _require(_array_equal(projection["executed_actions"], expected_actions),
             "released AEGIS executed-action binding differs")
    qps = projection["qp_records"]
    _require(len(qps) == len(expected_actions), "released AEGIS QP count differs")
    for qp, action in zip(qps, expected_actions):
        _require(qp["solver_status"] in ("optimal", "optimal_inaccurate"),
                 "released AEGIS QP did not solve")
        _require(_array_equal(qp["context"]["executed_action"], action),
                 "released AEGIS QP output differs from executed action")


def validate(
    *, repo_root: Path, result_path: Path, producer_commit: str,
    validator_commit: str, grouped_collection: bool = False,
) -> dict[str, Any]:
    import numpy as np

    from main.multilink_ellipsoid.query_action_risk import (
        RESULT_SCHEMA as QUERY_RESULT_SCHEMA,
        combine_row_minima,
        exact_safe,
        risk_from_row_minimum,
    )
    from main.multilink_ellipsoid.grouped_query_action_risk import (
        RESULT_SCHEMA as GROUPED_RESULT_SCHEMA,
    )

    validator_source = _git_identity(repo_root, validator_commit)
    result = _load(result_path)
    expected_schema = (
        GROUPED_RESULT_SCHEMA if grouped_collection else QUERY_RESULT_SCHEMA
    )
    _require(result["schema_version"] == expected_schema,
             "risk result schema differs")
    _require(result["status"] == "complete", "risk result is incomplete")
    _require(result["scientific_result"] is True, "risk result is not full protocol")
    _require(result["execution_mode"] == "full_diagnostic", "risk execution mode differs")
    _require(result["source"]["commit"] == producer_commit, "producer commit differs")
    payload = dict(result)
    claimed_payload = payload.pop("result_payload_sha256")
    _require(_sha256(canonical(payload)) == claimed_payload,
             "risk result payload self-hash differs")

    contract = result["action_contract"]
    _require(contract["released_aegis_ee_enabled"] is True,
             "original AEGIS EE filter is not enabled")
    _require(contract["risk_label_coordinates"]
             == "exact_final_released_AEGIS_output_executed_by_OSC",
             "risk label is not bound to final executed action")
    determinism = result["determinism_replay"]
    for key in (
        "next_state_maximum_absolute_error",
        "clearance_trace_maximum_absolute_error_m",
        "CAR_maximum_absolute_error_m",
        "executed_action_maximum_absolute_error",
        "nominal_recomputed_action_maximum_absolute_error",
    ):
        _require(float(determinism[key]) == 0.0,
                 "determinism or recomputed baseline differs: %s" % key)
    _require(determinism["contacts_identical"] is True,
             "determinism contact trace differs")
    _require(result["gates"][
        "zero_L5_residual_reproduces_recomputed_released_aegis"
    ] is True, "zero L5 residual does not reproduce original AEGIS")

    nominal = np.asarray(result["nominal_five_action_chunk"], dtype=np.float64)
    nominal_raw = np.asarray(
        result["nominal_raw_translational_five_action_chunk"], dtype=np.float64
    )
    _require(nominal.shape == nominal_raw.shape == (5, 7),
             "nominal action chunks differ")
    _require(hashlib.sha256(nominal.tobytes()).hexdigest()
             == result["nominal_five_action_chunk_sha256"],
             "recomputed nominal hash differs")
    _require(hashlib.sha256(nominal_raw.tobytes()).hexdigest()
             == result["nominal_raw_translational_five_action_chunk_sha256"],
             "raw nominal hash differs")

    candidates = result["candidates"]
    _require(len(candidates) == result["candidate_count"] == 37,
             "full candidate population differs")
    risk_config = (
        result["base_method_config"] if grouped_collection else result["config"]
    )
    buffer_m = float(risk_config["risk_target"]["safety_buffer_m"])
    car_limit = float(risk_config["risk_target"]["paper_car_threshold_m"])
    safe_count = timeout_count = proxy_collision_count = 0
    witness_counts = [0] * 7
    for index, candidate in enumerate(candidates):
        actions = np.asarray(candidate["actions"], dtype=np.float64)
        _require(actions.shape == (5, 7), "final candidate action shape differs")
        _check_projection(candidate, actions)
        residual = candidate["residual_binding"]
        _require(residual is not None, "L5 residual binding is absent")
        _require(_array_equal(residual["proposed_pre_aegis_actions"],
                              candidate["proposed_actions"]),
                 "pre-AEGIS action binding differs")
        applied = np.asarray(residual["applied_residual"], dtype=np.float64)
        _require(applied.shape == (5, 7)
                 and float(np.max(np.abs(applied[:, 3:]))) == 0.0,
                 "L5 residual changes rotation or gripper")
        if index == 0:
            _require(float(residual["applied_residual_l2_action"]) == 0.0,
                     "nominal candidate has nonzero L5 residual")
            _require(np.array_equal(actions, nominal),
                     "nominal candidate differs from recomputed original AEGIS")

        trace = np.asarray(candidate["prefix"]["clearance_trace_m"], dtype=np.float64)
        _require(trace.shape == (126, 7), "prefix trace shape differs")
        prefix_row = np.min(trace[1:], axis=0)
        _require(_array_equal(prefix_row, candidate["prefix"]["row_minimum_clearance_m"]),
                 "prefix row minima differ")
        _require(candidate["candidate_prefix_risk"]
                 == risk_from_row_minimum(prefix_row, buffer_m),
                 "prefix risk differs")
        parts = [prefix_row.tolist()]
        backup_row = candidate["backup"]["row_minimum_clearance_m"]
        if backup_row is not None:
            parts.append(backup_row)
        combined = combine_row_minima(*parts)
        _require(combined == candidate["combined_row_minimum_clearance_m"],
                 "combined row minimum differs")
        _require(candidate["combined_risk"]
                 == risk_from_row_minimum(combined, buffer_m),
                 "combined risk differs")
        _require(candidate["exact_safe"] == exact_safe(candidate),
                 "exact-safe target differs")
        if candidate["terminal_status"] == "UNKNOWN_TIMEOUT":
            timeout_count += 1
            _require(candidate["exact_safe"] is False, "timeout labeled safe")
        safe_count += int(candidate["exact_safe"])
        witness_counts[int(np.argmin(combined))] += 1
        proxy_safe_physical = bool(
            max(candidate["combined_risk"]) <= 0.0 and candidate["physical_veto"]
        )
        proxy_collision_count += int(proxy_safe_physical)
        for decision in candidate["backup"]["decisions"]:
            selected = decision["selected_aegis_consistency"]
            _require(selected["enabled"] is True,
                     "backup action bypassed original AEGIS")
            _require(_array_equal(selected["executed_actions"][0],
                                  decision["selected_action"]),
                     "selected backup action differs from AEGIS output")
            _require(_array_equal(selected["proposed_actions"][0],
                                  decision["selected_proposed_action"]),
                     "selected backup proposal differs")
        hold = candidate["backup"]["terminal_hold"]
        if hold is not None:
            _require(hold["aegis_consistency"]["enabled"] is True,
                     "terminal hold bypassed original AEGIS")
            _require(hold["maximum_active_obstacle_l1_displacement_m"] <= car_limit,
                     "terminal hold fails CAR")

    summary = result["summary"]
    _require(summary["safe_candidate_count"] == safe_count,
             "safe candidate count differs")
    _require(summary["unknown_timeout_count"] == timeout_count,
             "timeout count differs")
    _require(summary["row_active_witness_counts"] == witness_counts,
             "active witness count differs")
    _require(summary["proxy_safe_physical_collision_count"]
             == proxy_collision_count == 0,
             "proxy-safe physical collision exists")
    mixed_support = bool(safe_count > 0 and safe_count < len(candidates))
    if not grouped_collection:
        _require(mixed_support,
                 "corrected population lacks mixed safe/unsafe support")
        _require(all(result["gates"].values()), "producer gates did not all pass")
    else:
        for key in (
            "exact_snapshot_replay",
            "all_replays_boundary_exact",
            "timeout_never_labeled_safe",
            "current_state_safe",
            "no_proxy_safe_physical_collision",
            "zero_L5_residual_reproduces_recomputed_released_aegis",
        ):
            _require(result["gates"][key] is True,
                     "grouped producer apparatus gate failed: %s" % key)

    output = {
        "schema_version": "vlsa_distal_l5_aegis_consistent_risk_validation.v1",
        "status": "complete",
        "scientific_result": True,
        "producer": {
            "path": str(result_path),
            "file_sha256": _file_sha256(result_path),
            "result_payload_sha256": claimed_payload,
            "commit": producer_commit,
            "slurm_job_id": result["allocation"]["slurm_job_id"],
            "host": result["allocation"]["host"],
        },
        "validator_source": validator_source,
        "checks": {
            "payload_self_hash": True,
            "original_AEGIS_enabled": True,
            "zero_residual_baseline_exact": True,
            "every_prefix_label_bound_to_final_AEGIS_output": True,
            "every_backup_action_passes_original_AEGIS": True,
            "seven_row_geometry_and_L6_L7_diagnostics_present": True,
            "mixed_safe_unsafe_support": mixed_support,
            "no_proxy_safe_physical_collision": True,
        },
        "counts": {
            "candidates": len(candidates),
            "safe": safe_count,
            "unsafe_or_timeout": len(candidates) - safe_count,
            "timeouts": timeout_count,
            "active_witness_by_row": witness_counts,
        },
        "archived_replay_diagnostic": {
            "recomputed_vs_archived_AEGIS_maximum_action_error": determinism[
                "nominal_archived_action_maximum_absolute_error"
            ],
            "used_as_gate": False,
            "reason": "counterfactuals_are_bound_to_the_exact_recomputed_replay_state",
        },
        "interpretation": (
            "validated_grouped_L5_AEGIS_action_contract_"
            + ("mixed_support" if mixed_support else "support_failure_retained")
            if grouped_collection else
            "validated_L5_residual_before_original_AEGIS_nonvacuous_risk_population"
        ),
        "training_authorized": False,
        "next_gate": (
            "aggregate_grouped_action_contract_and_boundary_coverage"
            if grouped_collection else
            "collect_more_episode_grouped_recoverable_L5_boundary_states_with_the_same_action_contract"
        ),
    }
    output["validation_payload_sha256"] = _sha256(canonical(output))
    return output


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--producer-commit", required=True)
    parser.add_argument("--validator-commit", required=True)
    parser.add_argument(
        "--grouped-collection",
        action="store_true",
        help="Validate a grouped artifact while retaining states without mixed support.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = validate(
        repo_root=args.repo_root.resolve(),
        result_path=args.result.resolve(),
        producer_commit=args.producer_commit,
        validator_commit=args.validator_commit,
        grouped_collection=bool(args.grouped_collection),
    )
    _atomic_write(args.output.resolve(), output)
    print(json.dumps({
        "status": output["status"],
        "interpretation": output["interpretation"],
        "counts": output["counts"],
        "checks": output["checks"],
        "validation_payload_sha256": output["validation_payload_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
