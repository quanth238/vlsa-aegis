#!/usr/bin/env python3
"""Independently validate the fixed multi-region affine oracle artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from main.multilink_ellipsoid.multi_region_affine_oracle import (
    RESULT_SCHEMA, VALIDATION_SCHEMA, containing_region_indexes, fixed_regions,
    load_config, region_affine_values,
)
from main.multilink_ellipsoid.ridge_huber_oracle import affine_values
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _load, _require,
)


METHODS = ("multi_region", "single_affine")


def _hash_without(value: Mapping[str, Any], key: str) -> str:
    payload = dict(value)
    payload.pop(key, None)
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")).hexdigest()


def _arm_key(clearance_m: float) -> str:
    return "%dmm" % round(float(clearance_m) * 1000.0)


def _summary(pairs: list[tuple[bool, bool]]) -> dict[str, Any]:
    true_count = sum(true for true, _ in pairs)
    predicted_count = sum(predicted for _, predicted in pairs)
    accepted = sum(true and predicted for true, predicted in pairs)
    false_safe = sum((not true) and predicted for true, predicted in pairs)
    return {
        "action_count": len(pairs), "true_safe_action_count": true_count,
        "predicted_safe_action_count": predicted_count,
        "accepted_true_safe_action_count": accepted,
        "false_safe_action_count": false_safe,
        "safe_action_recall": accepted / true_count if true_count else None,
    }


def _true_safe(record: Mapping[str, Any], clearance: float, motion: float) -> bool:
    import numpy as np

    return bool(
        np.all(np.asarray(record["minimum_distal_margin_m"], dtype=np.float64)
               >= float(clearance))
        and record["D_sim_raw_safe"]
        and int(record["raw_protected_contact_count"]) == 0
        and float(record["maximum_within_step_obstacle_l1_displacement_m"]) <= motion
    )


def _exact_safe(record: Any, clearance: float, motion: float) -> bool:
    import numpy as np

    return bool(
        record is not None
        and np.all(np.asarray(record["minimum_distal_margin_m"], dtype=np.float64)
                   >= float(clearance))
        and record["D_opt_seven_distal_at_target"]
        and record["D_sim_raw_safe"]
        and int(record["raw_protected_contact_count"]) == 0
        and float(record["maximum_within_step_obstacle_l1_displacement_m"]) <= motion
        and record["true_safe"]
    )


def main() -> int:
    import numpy as np

    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--off-grid-result", type=Path, required=True)
    parser.add_argument("--single-affine-result", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = {name: path.resolve() for name, path in {
        "result": args.result, "dataset": args.dataset,
        "off_grid": args.off_grid_result,
        "single": args.single_affine_result, "config": args.config,
        "output": args.output,
    }.items()}
    config = load_config(paths["config"])
    result = _load(paths["result"])
    dataset = _load(paths["dataset"])
    off_grid = _load(paths["off_grid"])
    single = _load(paths["single"])
    source = config["immutable_source"]
    _require(
        result.get("schema_version") == RESULT_SCHEMA
        and result.get("source", {}).get("commit") == args.expected_commit
        and result.get("result_payload_sha256")
        == _hash_without(result, "result_payload_sha256")
        and result.get("config", {}).get("config_file_sha256")
        == config["config_file_sha256"]
        and _file_sha256(paths["dataset"]) == source["dataset_file_sha256"]
        and _file_sha256(paths["off_grid"]) == source["off_grid_result_file_sha256"]
        and _file_sha256(paths["single"]) == source["single_affine_result_file_sha256"],
        "multi-region result identity differs",
    )
    source_states = {int(item["state_index"]): item for item in dataset["state_records"]}
    fresh_states = {int(item["state_index"]): item for item in off_grid["state_results"]}
    single_states = {int(item["state_index"]): item for item in single["state_results"]}
    states = result.get("state_results", [])
    expected_count = int(source["expected_state_count"])
    _require(
        len(states) == expected_count
        and [item["state_index"] for item in states] == list(range(expected_count))
        and set(source_states) == set(fresh_states) == set(single_states)
        == set(range(expected_count)),
        "multi-region state set differs",
    )
    motion = float(config["exact_verification"][
        "maximum_per_step_obstacle_l1_displacement_m"
    ])
    tolerance = float(config["partition"]["inclusive_membership_tolerance"])
    pairs = {
        _arm_key(clearance): {method: [] for method in METHODS}
        for clearance in config["clearance_arms_m"]
    }
    supported = {
        arm: {method: 0 for method in METHODS} for arm in pairs
    }
    accepted_states = {
        arm: {method: 0 for method in METHODS} for arm in pairs
    }
    selected_qp = {
        arm: {method: 0 for method in METHODS} for arm in pairs
    }
    qp_false_safe = {
        arm: {method: 0 for method in METHODS} for arm in pairs
    }
    stability_gate = True
    counted_rollouts = 0
    for state in states:
        index = int(state["state_index"])
        source_state = source_states[index]
        fresh_source = fresh_states[index]
        single_source = single_states[index]
        _require(
            state["case_id"] == source_state["case_id"] == fresh_source["case_id"]
            == single_source["case_id"]
            and int(state["state_step"]) == int(source_state["state_step"])
            and state["feature_max_abs_error"] <= 1.0e-8
            and state["clearance_max_abs_error_m"] <= 1.0e-8,
            "multi-region state identity/receipt differs",
        )
        fit_xyz = np.asarray(source_state["candidate_first_xyz"], dtype=np.float64)
        fit_margins = np.asarray(
            source_state["candidate_minimum_distal_margin_m"], dtype=np.float64
        )
        nominal = np.asarray(source_state["nominal_first_action"][:3], dtype=np.float64)
        regions = fixed_regions(
            fit_xyz, source_state["action_lower"], source_state["action_upper"],
            config["partition"],
        )
        _require(state["regions"] == regions and len(state["regional_targets"]) == 27,
                 "multi-region fixed partition differs")
        for region, target in zip(regions, state["regional_targets"]):
            indexes = region["fit_candidate_indexes"]
            lower = np.asarray([
                region_affine_values(target, fit_xyz[item]) for item in indexes
            ])
            _require(
                int(target["region_index"]) == int(region["region_index"])
                and float(np.max(lower - fit_margins[indexes])) <= 1.0e-10,
                "multi-region fit lower bound differs",
            )
        local_stable = bool(all(
            item["all_active_rows_stable"]
            and all(row["gate_pass"] for row in item["row_audits"])
            for item in state["regional_resampling_stability"]
        ))
        _require(
            len(state["regional_resampling_stability"]) == 27
            and state["all_active_region_rows_stable"] is local_stable,
            "multi-region stability decision differs",
        )
        stability_gate = bool(stability_gate and local_stable)
        local_pairs = {
            arm: {method: [] for method in METHODS} for arm in pairs
        }
        _require(
            len(state["fresh_actions"]) == len(fresh_source["fresh_actions"])
            == int(source["off_grid_actions_per_state"]),
            "multi-region fresh action count differs",
        )
        for record, immutable in zip(state["fresh_actions"], fresh_source["fresh_actions"]):
            _require(
                record["fresh_index"] == immutable["fresh_index"]
                and record["candidate_xyz"] == immutable["candidate_xyz"]
                and record["minimum_distal_margin_m"]
                == immutable["minimum_distal_margin_m"],
                "multi-region immutable off-grid label differs",
            )
            xyz = np.asarray(record["candidate_xyz"], dtype=np.float64)
            containing = containing_region_indexes(xyz, regions, tolerance)
            _require(record["containing_region_indexes"] == containing,
                     "multi-region membership differs")
            regional_minima = {
                str(region_index): float(np.min(region_affine_values(
                    state["regional_targets"][region_index], xyz
                ))) for region_index in containing
            }
            _require(all(abs(
                regional_minima[key]
                - float(record["regional_minimum_lower_bounds_m"][key])
            ) <= 1.0e-12 for key in regional_minima),
                     "multi-region stored lower bound differs")
            target = state["single_affine_target"]
            single_values = affine_values(
                target["nominal_margin_m"], target["gradient_m_per_action"],
                target["one_sided_error_m"], xyz, nominal,
            )
            _require(np.allclose(single_values, record["single_affine_values_m"],
                                 atol=1.0e-12, rtol=0.0),
                     "single-affine stored value differs")
            for clearance in config["clearance_arms_m"]:
                arm = _arm_key(clearance)
                true = _true_safe(immutable, clearance, motion)
                flags = {
                    "multi_region": any(
                        regional_minima[str(region_index)] >= float(clearance)
                        for region_index in containing
                    ),
                    "single_affine": bool(np.all(single_values >= float(clearance))),
                }
                _require(
                    record["arms"][arm]["true_safe"] is true
                    and record["arms"][arm]["multi_region_predicted_safe"]
                    is flags["multi_region"]
                    and record["arms"][arm]["single_affine_predicted_safe"]
                    is flags["single_affine"],
                    "multi-region off-grid arm decision differs",
                )
                for method in METHODS:
                    pair = (true, flags[method])
                    pairs[arm][method].append(pair)
                    local_pairs[arm][method].append(pair)
        for clearance in config["clearance_arms_m"]:
            arm = _arm_key(clearance)
            for method in METHODS:
                local_summary = _summary(local_pairs[arm][method])
                _require(
                    state["off_grid_summaries"][arm][method] == local_summary,
                    "multi-region per-state off-grid summary differs",
                )
                has_support = local_summary["true_safe_action_count"] > 0
                supported[arm][method] += int(has_support)
                accepted_states[arm][method] += int(
                    has_support and local_summary["accepted_true_safe_action_count"] > 0
                )
            qp = state["QP"][arm]
            proposals = qp["multi_region_proposals"]
            _require(len(proposals) == 27, "regional QP proposal count differs")
            valid = 0
            false = 0
            verified = []
            for proposal in proposals:
                solution = proposal["solution"]
                exact_safe = _exact_safe(
                    proposal["fresh_exact_two_step"], clearance, motion
                )
                valid += int(solution["valid"])
                counted_rollouts += int(solution["valid"])
                false += int(solution["valid"] and not exact_safe)
                if solution["valid"] and exact_safe:
                    verified.append(proposal)
                _require(
                    proposal["false_safe"] is bool(solution["valid"] and not exact_safe)
                    and proposal["verified_safe"] is bool(solution["valid"] and exact_safe),
                    "regional QP exact flag differs",
                )
            verified.sort(key=lambda item: (
                float(item["solution"]["correction_l2"]), int(item["region_index"])
            ))
            chosen = None if not verified else verified[0]
            _require(
                qp["valid_proposal_count"] == valid
                and qp["false_safe_proposal_count"] == false
                and qp["verified_safe_proposal_count"] == len(verified)
                and qp["selected_region_index"]
                == (None if chosen is None else int(chosen["region_index"]))
                and qp["selected_gate_pass"] is (chosen is not None),
                "regional QP selection differs",
            )
            selected_qp[arm]["multi_region"] += int(chosen is not None)
            qp_false_safe[arm]["multi_region"] += false
            single_qp = qp["single_affine"]
            single_safe = _exact_safe(
                single_qp["fresh_exact_two_step"], clearance, motion
            )
            counted_rollouts += int(single_qp["solution"]["valid"])
            single_false = bool(single_qp["solution"]["valid"] and not single_safe)
            single_gate = bool(single_qp["solution"]["valid"] and single_safe)
            _require(
                single_qp["false_safe"] is single_false
                and single_qp["gate_pass"] is single_gate,
                "single-affine QP flag differs",
            )
            selected_qp[arm]["single_affine"] += int(single_gate)
            qp_false_safe[arm]["single_affine"] += int(single_false)
    _require(result["new_cloned_OSC_rollout_count"] == counted_rollouts,
             "multi-region rollout count differs")
    aggregates = {}
    decisions = {}
    required = set(config["oracle_gate"]["required_zero_margin_recovery_state_indexes"])
    for clearance in config["clearance_arms_m"]:
        arm = _arm_key(clearance)
        aggregates[arm] = {}
        for method in METHODS:
            aggregate = _summary(pairs[arm][method])
            aggregate.update({
                "states_with_exact_off_grid_safe_support": supported[arm][method],
                "supported_states_with_accepted_safe_action": accepted_states[arm][method],
                "states_with_selected_exact_safe_QP": selected_qp[arm][method],
                "QP_false_safe_proposal_count": qp_false_safe[arm][method],
            })
            aggregates[arm][method] = aggregate
        recovered = sorted(
            index for index in required
            if states[index]["off_grid_summaries"][arm]["multi_region"][
                "accepted_true_safe_action_count"
            ] > 0 and states[index]["QP"][arm]["selected_gate_pass"]
        )
        multi = aggregates[arm]["multi_region"]
        gate = bool(
            multi["false_safe_action_count"] == 0
            and multi["QP_false_safe_proposal_count"] == 0
            and multi["supported_states_with_accepted_safe_action"]
            == multi["states_with_exact_off_grid_safe_support"]
            and multi["states_with_selected_exact_safe_QP"] == expected_count
            and stability_gate
            and (float(clearance) != 0.0 or set(recovered) == required)
        )
        decisions[arm] = {
            "clearance_target_m": float(clearance), "gate_pass": gate,
            "required_recovery_state_indexes": sorted(required),
            "recovered_required_state_indexes": recovered,
        }
    overall = bool(all(item["gate_pass"] for item in decisions.values()))
    expected_decision = {
        "clearance_arm_decisions": decisions,
        "all_active_region_rows_resampling_stable": stability_gate,
        "multi_region_oracle_gate_pass": overall,
        "mlp_training_authorized": overall,
        "mlp_training_executed": False,
        "closed_loop_e05_authorized": False,
        "closed_loop_e05_executed": False,
        "stop_reason": None if overall else "one_or_more_multi_region_oracle_gates_failed",
    }
    _require(result["arm_aggregates"] == aggregates
             and result["decision"] == expected_decision,
             "multi-region aggregate/decision differs")
    output = {
        "schema_version": VALIDATION_SCHEMA, "status": "valid",
        "scientific_result": False, "expected_commit": args.expected_commit,
        "result_file_sha256": _file_sha256(paths["result"]),
        "result_payload_sha256": result["result_payload_sha256"],
        "dataset_file_sha256": _file_sha256(paths["dataset"]),
        "off_grid_result_file_sha256": _file_sha256(paths["off_grid"]),
        "single_affine_result_file_sha256": _file_sha256(paths["single"]),
        "arm_aggregates": aggregates,
        "all_active_region_rows_resampling_stable": stability_gate,
        "multi_region_oracle_gate_pass": overall,
        "mlp_training_authorized": overall,
        "closed_loop_e05_authorized": False,
    }
    _atomic_write(paths["output"], output)
    print(json.dumps(output, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
