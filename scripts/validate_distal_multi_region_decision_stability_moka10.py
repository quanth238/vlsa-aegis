#!/usr/bin/env python3
"""Independently validate the multi-region decision-stability artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from main.multilink_ellipsoid.multi_region_affine_oracle import (
    containing_region_indexes, region_affine_values,
)
from main.multilink_ellipsoid.multi_region_decision_stability import (
    RESULT_SCHEMA, VALIDATION_SCHEMA, jaccard, load_config,
)
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _load, _require,
)


def _hash_without(value: Mapping[str, Any], key: str) -> str:
    payload = dict(value)
    payload.pop(key, None)
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")).hexdigest()


def _exact_safe(record: Any, maximum_motion: float) -> bool:
    import numpy as np

    return bool(
        record is not None
        and np.all(np.asarray(record["minimum_distal_margin_m"], dtype=np.float64)
                   >= 0.0)
        and record["D_opt_seven_distal_safe"]
        and record["D_sim_raw_safe"]
        and int(record["raw_protected_contact_count"]) == 0
        and float(record["maximum_within_step_obstacle_l1_displacement_m"])
        <= maximum_motion
        and record["true_safe"]
    )


def main() -> int:
    import numpy as np

    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--off-grid-result", type=Path, required=True)
    parser.add_argument("--multi-region-result", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = {name: path.resolve() for name, path in {
        "result": args.result, "dataset": args.dataset,
        "off_grid": args.off_grid_result, "multi": args.multi_region_result,
        "config": args.config, "output": args.output,
    }.items()}
    config = load_config(paths["config"])
    result = _load(paths["result"])
    dataset = _load(paths["dataset"])
    off_grid = _load(paths["off_grid"])
    baseline = _load(paths["multi"])
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
        and _file_sha256(paths["multi"]) == source["multi_region_result_file_sha256"],
        "decision-stability result identity differs",
    )
    source_states = {int(item["state_index"]): item for item in dataset["state_records"]}
    fresh_states = {int(item["state_index"]): item for item in off_grid["state_results"]}
    baseline_states = {int(item["state_index"]): item for item in baseline["state_results"]}
    states = result.get("state_results", [])
    state_count = int(source["expected_state_count"])
    replicate_count = int(config["resampling"]["replicate_count"])
    _require(
        len(states) == state_count
        and [item["state_index"] for item in states] == list(range(state_count))
        and set(source_states) == set(fresh_states) == set(baseline_states)
        == set(range(state_count)),
        "decision-stability state set differs",
    )
    maximum_motion = float(config["exact_verification"][
        "maximum_per_step_obstacle_l1_displacement_m"
    ])
    tolerance = float(baseline["config"]["partition"][
        "inclusive_membership_tolerance"
    ])
    rollout_count = 0
    exact_safe_count = 0
    region_match_count = 0
    false_safe_total = 0
    all_shifts = []
    replicate_base = [[] for _ in range(replicate_count)]
    replicate_sampled = [[] for _ in range(replicate_count)]
    replicate_false = [0] * replicate_count
    replicate_support = [0] * replicate_count
    replicate_retained = [0] * replicate_count
    minimum_state_jaccard = 1.0
    for state in states:
        index = int(state["state_index"])
        source_state = source_states[index]
        fresh_state = fresh_states[index]
        base_state = baseline_states[index]
        _require(
            state["case_id"] == source_state["case_id"] == fresh_state["case_id"]
            == base_state["case_id"]
            and int(state["state_step"]) == int(source_state["state_step"])
            and state["feature_max_abs_error"] <= 1.0e-8
            and state["clearance_max_abs_error_m"] <= 1.0e-8
            and state["regions"] == base_state["regions"]
            and len(state["replicates"]) == replicate_count,
            "decision-stability state pairing differs",
        )
        xyz = np.asarray(source_state["candidate_first_xyz"], dtype=np.float64)
        margins = np.asarray(
            source_state["candidate_minimum_distal_margin_m"], dtype=np.float64
        )
        regions = state["regions"]
        baseline_flags = np.asarray([
            bool(item["arms"]["0mm"]["multi_region_predicted_safe"])
            for item in base_state["fresh_actions"]
        ], dtype=bool)
        true_flags = np.asarray([
            bool(item["true_safe"]) for item in fresh_state["fresh_actions"]
        ], dtype=bool)
        baseline_indexes = np.flatnonzero(baseline_flags).tolist()
        baseline_accepted_true = int(np.count_nonzero(
            np.logical_and(baseline_flags, true_flags)
        ))
        _require(
            state["baseline_accepted_fresh_indexes"] == baseline_indexes
            and state["baseline_accepted_true_safe_action_count"]
            == baseline_accepted_true
            and state["baseline_selected_xyz"]
            == base_state["QP"]["0mm"]["selected_solution"]["candidate_xyz"],
            "decision-stability baseline decision differs",
        )
        baseline_selected = np.asarray(state["baseline_selected_xyz"], dtype=np.float64)
        for replicate in state["replicates"]:
            replicate_index = int(replicate["replicate_index"])
            targets = replicate["regional_targets"]
            _require(len(targets) == len(regions) == 27,
                     "decision-stability regional target count differs")
            for region, target in zip(regions, targets):
                selected = target["selected_candidate_indexes"]
                full_indexes = region["fit_candidate_indexes"]
                anchor = np.asarray(target["anchor_xyz"], dtype=np.float64)
                gradient = np.asarray(target["gradient_m_per_action"], dtype=np.float64)
                lower = (
                    np.asarray(target["anchor_margin_m"], dtype=np.float64)
                    - np.asarray(target["one_sided_error_m"], dtype=np.float64)
                    + (xyz[full_indexes] - anchor) @ gradient.T
                )
                _require(
                    len(selected) == int(config["resampling"][
                        "selected_actions_per_region"
                    ])
                    and set(selected).issubset(set(full_indexes))
                    and float(np.max(lower - margins[full_indexes])) <= 1.0e-10,
                    "decision-stability regional recalibration differs",
                )
            flags = []
            accepted_indexes = []
            false_indexes = []
            for fresh_index, fresh in enumerate(fresh_state["fresh_actions"]):
                action = np.asarray(fresh["candidate_xyz"], dtype=np.float64)
                containing = containing_region_indexes(action, regions, tolerance)
                accepted = bool(any(
                    np.all(region_affine_values(targets[region_index], action) >= 0.0)
                    for region_index in containing
                ))
                flags.append(accepted)
                if accepted:
                    accepted_indexes.append(fresh_index)
                    if not true_flags[fresh_index]:
                        false_indexes.append(fresh_index)
            flags_array = np.asarray(flags, dtype=bool)
            accepted_true = int(np.count_nonzero(
                np.logical_and(flags_array, true_flags)
            ))
            state_jaccard = jaccard(baseline_flags, flags_array)
            _require(
                replicate["accepted_fresh_indexes"] == accepted_indexes
                and replicate["false_safe_fresh_indexes"] == false_indexes
                and replicate["accepted_true_safe_action_count"] == accepted_true
                and abs(replicate["accepted_set_jaccard_to_full_fit"]
                        - state_jaccard) <= 1.0e-15,
                "decision-stability accepted set differs",
            )
            minimum_state_jaccard = min(minimum_state_jaccard, state_jaccard)
            false_safe_total += len(false_indexes)
            replicate_false[replicate_index] += len(false_indexes)
            replicate_base[replicate_index].extend(baseline_flags.tolist())
            replicate_sampled[replicate_index].extend(flags_array.tolist())
            if baseline_accepted_true > 0:
                replicate_support[replicate_index] += 1
                replicate_retained[replicate_index] += int(accepted_true > 0)
            qp = replicate["QP"]
            proposals = qp["regional_proposals"]
            valid = [item for item in proposals if item["valid"]]
            valid.sort(key=lambda item: (
                float(item["correction_l2"]), int(item["region_index"])
            ))
            chosen = None if not valid else valid[0]
            exact_safe = _exact_safe(qp["fresh_exact_two_step"], maximum_motion)
            if chosen is not None:
                rollout_count += 1
                shift = float(np.linalg.norm(
                    np.asarray(chosen["candidate_xyz"], dtype=np.float64)
                    - baseline_selected
                ))
                all_shifts.append(shift)
            else:
                shift = None
            exact_safe_count += int(chosen is not None and exact_safe)
            region_match = bool(
                chosen is not None
                and int(chosen["region_index"])
                == int(state["baseline_selected_region_index"])
            )
            region_match_count += int(region_match)
            _require(
                len(proposals) == 27
                and qp["selected_region_index"]
                == (None if chosen is None else int(chosen["region_index"]))
                and qp["selected_xyz"]
                == (None if chosen is None else chosen["candidate_xyz"])
                and qp["selected_action_shift_l2_from_full_fit"] == shift
                and qp["selected_region_matches_full_fit"] is region_match
                and qp["gate_pass"] is bool(chosen is not None and exact_safe),
                "decision-stability selected QP differs",
            )
    global_jaccards = [
        jaccard(replicate_base[index], replicate_sampled[index])
        for index in range(replicate_count)
    ]
    summaries = [{
        "replicate_index": index,
        "global_accepted_set_jaccard": global_jaccards[index],
        "false_safe_action_count": replicate_false[index],
        "baseline_supported_state_count": replicate_support[index],
        "retained_supported_state_count": replicate_retained[index],
    } for index in range(replicate_count)]
    _require(result["replicate_summaries"] == summaries,
             "decision-stability replicate summaries differ")
    shifts = np.asarray(all_shifts, dtype=np.float64)
    p95 = float(np.percentile(shifts, 95)) if len(shifts) else None
    maximum = float(np.max(shifts)) if len(shifts) else None
    aggregates = {
        "state_count": state_count, "replicate_count": replicate_count,
        "replicated_off_grid_decision_count": state_count * replicate_count * 96,
        "replicated_off_grid_false_safe_action_count": false_safe_total,
        "minimum_global_accepted_set_jaccard": min(global_jaccards),
        "minimum_state_accepted_set_jaccard": minimum_state_jaccard,
        "selected_QP_rollout_count": rollout_count,
        "fresh_exact_safe_selected_QP_count": exact_safe_count,
        "selected_region_match_count": region_match_count,
        "selected_action_shift_l2_mean": float(np.mean(shifts)),
        "selected_action_shift_l2_p95": p95,
        "selected_action_shift_l2_maximum": maximum,
    }
    gate = config["decision_gate"]
    component = {
        "zero_replicated_false_safes": false_safe_total
        == int(gate["replicated_off_grid_false_safe_action_count"]),
        "global_jaccard_gate": min(global_jaccards)
        >= float(gate["minimum_global_accepted_set_jaccard_per_replicate"]),
        "state_jaccard_gate": minimum_state_jaccard
        >= float(gate["minimum_state_accepted_set_jaccard"]),
        "support_retention_gate": all(
            item["baseline_supported_state_count"]
            == int(gate["expected_baseline_supported_state_count"])
            and item["retained_supported_state_count"]
            == item["baseline_supported_state_count"] for item in summaries
        ),
        "fresh_exact_selected_QP_gate": bool(
            rollout_count == int(gate["expected_selected_QP_rollout_count"])
            and exact_safe_count == rollout_count
        ),
        "selected_action_p95_gate": bool(
            p95 is not None
            and p95 <= float(gate["selected_action_shift_l2_p95_maximum"])
        ),
        "selected_action_maximum_gate": bool(
            maximum is not None
            and maximum <= float(gate["selected_action_shift_l2_maximum"])
        ),
    }
    overall = bool(all(component.values()))
    decision = {
        "component_gates": component,
        "decision_stability_gate_pass": overall,
        "coefficient_nonuniqueness_decision_harmless": overall,
        "mlp_training_authorized": overall,
        "mlp_training_executed": False,
        "closed_loop_e05_authorized": False,
        "closed_loop_e05_executed": False,
        "stop_reason": None if overall else "one_or_more_decision_stability_gates_failed",
    }
    _require(result["aggregates"] == aggregates and result["decision"] == decision,
             "decision-stability aggregate differs")
    output = {
        "schema_version": VALIDATION_SCHEMA, "status": "valid",
        "scientific_result": False, "expected_commit": args.expected_commit,
        "result_file_sha256": _file_sha256(paths["result"]),
        "result_payload_sha256": result["result_payload_sha256"],
        "dataset_file_sha256": _file_sha256(paths["dataset"]),
        "off_grid_result_file_sha256": _file_sha256(paths["off_grid"]),
        "multi_region_result_file_sha256": _file_sha256(paths["multi"]),
        "aggregates": aggregates,
        "decision_stability_gate_pass": overall,
        "mlp_training_authorized": overall,
        "closed_loop_e05_authorized": False,
    }
    _atomic_write(paths["output"], output)
    print(json.dumps(output, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
