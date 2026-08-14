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


def _validate_proxy_validity(
    *, grouped_collection: bool, reported_count: int, observed_count: int,
    producer_zero_gate: bool,
) -> None:
    """Separate a retained scientific proxy failure from apparatus corruption."""
    _require(int(reported_count) == int(observed_count),
             "proxy-safe physical collision count differs")
    _require(bool(producer_zero_gate) == (int(observed_count) == 0),
             "proxy-validity gate differs from observed count")
    if not grouped_collection:
        _require(int(observed_count) == 0,
                 "proxy-safe physical collision exists")


def validate(
    *, repo_root: Path, result_path: Path, producer_commit: str,
    validator_commit: str, grouped_collection: bool = False,
    adaptive_collection: bool = False,
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
    from main.multilink_ellipsoid.adaptive_query_action_risk import (
        RESULT_SCHEMA as ADAPTIVE_RESULT_SCHEMA,
        load_config as load_adaptive_config,
    )

    validator_source = _git_identity(repo_root, validator_commit)
    result = _load(result_path)
    _require(not (grouped_collection and adaptive_collection),
             "risk validation collection mode is ambiguous")
    grouped_semantics = bool(grouped_collection or adaptive_collection)
    expected_schema = (
        ADAPTIVE_RESULT_SCHEMA if adaptive_collection else
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
    if adaptive_collection:
        adaptive = result["adaptive_boundary_sampling"]
        adaptive_config = load_adaptive_config(
            repo_root / "configs/vlsa_distal_adaptive_query_action_risk.v1.json"
        )
        _require(adaptive["config"] == adaptive_config,
                 "adaptive embedded config differs")
        _require(result["population_binding"]["adaptive_config"] == adaptive_config,
                 "adaptive population config binding differs")
        _require(
            len(candidates) == result["candidate_count"]
            == adaptive["retained_authoritative_candidate_count"]
            and 1 <= len(candidates)
            <= int(adaptive_config["screening"][
                "maximum_authoritative_candidates"
            ]),
            "adaptive authoritative candidate population differs",
        )
    else:
        _require(len(candidates) == result["candidate_count"] == 37,
                 "full candidate population differs")
    risk_config = (
        result["base_method_config"] if grouped_semantics else result["config"]
    )
    buffer_m = float(risk_config["risk_target"]["safety_buffer_m"])
    car_limit = float(risk_config["risk_target"]["paper_car_threshold_m"])
    if adaptive_collection:
        from main.multilink_ellipsoid.adaptive_query_action_risk import (
            midpoint_definition, select_bracket,
        )

        context = result["state"]["physical_context"]
        _require(
            len(context["arm_joint_position_rad"]) == 7
            and len(context["arm_joint_velocity_rad_s"]) == 7
            and len(context["eef_position_m"]) == 3
            and len(context["eef_quaternion_xyzw"]) == 4
            and len(context["geometry_rows"]) == 7,
            "adaptive physical context differs",
        )
        for row in context["geometry_rows"]:
            _require(
                len(row["obstacle_relative_center_m"]) == 3
                and len(row["outward_normal"]) == 3,
                "adaptive relative geometry differs",
            )
        coarse = adaptive["coarse_screening_records"]
        _require(len(coarse) == int(adaptive_config["screening"][
            "coarse_candidate_count"
        ]), "adaptive screening count differs")
        from main.multilink_ellipsoid.adaptive_query_action_risk import (
            coarse_candidate_definitions,
        )
        expected_coarse = coarse_candidate_definitions(
            nominal,
            result["state"]["local_frame"],
            risk_config,
            adaptive_config,
        )
        _require(
            [record["definition"] for record in coarse] == expected_coarse,
            "adaptive coarse candidate bank differs",
        )

        def validate_screen(record: Mapping[str, Any]) -> None:
            executed = np.asarray(
                record["exact_final_post_aegis_actions"], dtype=np.float64
            )
            _require(executed.shape == (5, 7),
                     "adaptive screened action shape differs")
            _check_projection(record, executed)
            trace = np.asarray(
                record["prefix"]["clearance_trace_m"], dtype=np.float64
            )
            substeps = [int(value) for value in record["prefix"]["substep_counts"]]
            _require(
                trace.shape == (1 + sum(substeps), 7)
                and all(value == 25 for value in substeps)
                and len(substeps) == int(record["executed_prefix_action_count"]),
                "adaptive screening prefix shape differs",
            )
            row_minimum = np.min(trace[1:], axis=0)
            risk = risk_from_row_minimum(row_minimum, buffer_m)
            _require(
                _array_equal(row_minimum, record["prefix"][
                    "row_minimum_clearance_m"
                ])
                and risk == record["prefix_risk"],
                "adaptive screening risk differs",
            )
            scalar = max(float(risk[row]) for row in (0, 1, 2))
            physical = bool(
                record["prefix"]["protected_contact_count"] > 0
                or record["prefix"][
                    "maximum_active_obstacle_l1_displacement_m"
                ] > car_limit
            )
            _require(
                float(record["L5_prefix_risk_m"]) == scalar
                and bool(record["physical_veto"]) == physical
                and bool(record["screen_safe"]) == (
                    not physical and scalar <= 0.0
                )
                and bool(record["screen_unsafe"]) == (
                    physical or scalar > 0.0
                ),
                "adaptive screening classification differs",
            )
            _require("backup" not in record,
                     "screening-only candidate contains backup label")

        for record in coarse:
            validate_screen(record)
        expected_bracket = select_bracket(coarse)
        _require(
            bool(adaptive["bracket_found"]) == (expected_bracket is not None)
            and adaptive["initial_bracket_indices"] == (
                None if expected_bracket is None else list(expected_bracket)
            ),
            "adaptive bracket selection differs",
        )
        bisected = adaptive["bisection_records"]
        if expected_bracket is None:
            _require(not bisected and len(candidates) == 1,
                     "adaptive unbracketed state retained extra labels")
        else:
            _require(len(bisected) == int(adaptive_config["screening"][
                "bisection_iterations"
            ]), "adaptive bisection count differs")
            safe_record = coarse[int(expected_bracket[0])]
            unsafe_record = coarse[int(expected_bracket[1])]
            for iteration, record in enumerate(bisected):
                validate_screen(record)
                expected = midpoint_definition(
                    safe_record["definition"], unsafe_record["definition"],
                    iteration=iteration,
                    order=int(record["definition"]["order"]),
                )
                _require(
                    _array_equal(expected["actions"], record["definition"]["actions"]),
                    "adaptive bisection midpoint differs",
                )
                if record["screen_safe"]:
                    safe_record = record
                else:
                    unsafe_record = record
        _require(
            adaptive["retained_authoritative_candidate_names"]
            == [candidate["name"] for candidate in candidates]
            and adaptive[
                "complete_backup_not_run_for_screening_only_candidates"
            ] is True,
            "adaptive retained candidate binding differs",
        )
    safe_count = known_unsafe_count = timeout_count = proxy_collision_count = 0
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
        substeps = [int(value) for value in candidate["prefix"]["substep_counts"]]
        _require(
            trace.shape == (1 + sum(substeps), 7)
            and all(value == 25 for value in substeps),
            "prefix trace shape differs",
        )
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
        elif not bool(candidate["exact_safe"]):
            known_unsafe_count += 1
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
    _validate_proxy_validity(
        grouped_collection=grouped_semantics,
        reported_count=summary["proxy_safe_physical_collision_count"],
        observed_count=proxy_collision_count,
        producer_zero_gate=result["gates"][
            "proxy_safe_physical_collision_count_zero"
        ],
    )
    mixed_support = bool(safe_count > 0 and known_unsafe_count > 0)
    if not grouped_semantics:
        _require(mixed_support,
                 "corrected population lacks mixed safe/unsafe support")
        _require(all(result["gates"].values()), "producer gates did not all pass")
    else:
        for key in (
            "exact_snapshot_replay",
            "all_replays_boundary_exact",
            "no_timeout_labeled_safe",
            "query_boundary_is_initially_safe",
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
            "no_proxy_safe_physical_collision": proxy_collision_count == 0,
            "adaptive_boundary_protocol": bool(adaptive_collection),
        },
        "counts": {
            "candidates": len(candidates),
            "safe": safe_count,
            "known_unsafe": known_unsafe_count,
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
            if grouped_semantics else
            "validated_L5_residual_before_original_AEGIS_nonvacuous_risk_population"
        ),
        "training_authorized": False,
        "next_gate": (
            "aggregate_grouped_action_contract_and_boundary_coverage"
            if grouped_semantics else
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
    parser.add_argument(
        "--adaptive-collection",
        action="store_true",
        help="Validate variable-count adaptive boundary artifacts.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = validate(
        repo_root=args.repo_root.resolve(),
        result_path=args.result.resolve(),
        producer_commit=args.producer_commit,
        validator_commit=args.validator_commit,
        grouped_collection=bool(args.grouped_collection),
        adaptive_collection=bool(args.adaptive_collection),
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
