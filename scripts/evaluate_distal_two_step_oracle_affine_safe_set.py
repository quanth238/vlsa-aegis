#!/usr/bin/env python3
"""Evaluate the privileged two-step conservative affine representation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping

from main.multilink_ellipsoid.oracle_affine_safe_set import (
    ORACLE_AFFINE_SAFE_SET_RESULT_SCHEMA,
    fit_candidate_conditioned_affine_certificate,
    load_oracle_affine_safe_set_config,
    solve_affine_certificate_qp,
)
from main.multilink_ellipsoid.two_step_margin import (
    TWO_STEP_DATASET_SCHEMA,
    feature_context,
    load_selected_manifest,
    load_two_step_config,
    summarize_chunk,
)
from scripts.collect_distal_boundary_generalization_moka10 import _canonical_action
from scripts.evaluate_distal_execution_margin_nn_e05 import _build_pair, _geometry
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write,
    _file_sha256,
    _git_identity,
    _load,
    _require,
)


def _hash_without(value: Mapping[str, Any], key: str) -> str:
    payload = dict(value)
    payload.pop(key, None)
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def _case_grid(dataset: Mapping[str, Any], case_id: str, expected: int) -> dict[str, Any]:
    import numpy as np

    records = [
        item for item in dataset["records"]
        if item["case_id"] == case_id and item["source"] == "grid"
    ]
    records.sort(key=lambda item: int(item["grid_index"]))
    _require(
        len(records) == expected
        and [int(item["grid_index"]) for item in records] == list(range(expected)),
        "oracle affine safe-set grid identity differs: %s" % case_id,
    )
    first = np.asarray(records[0]["nominal_first_action"], dtype=np.float64)
    second = np.asarray(records[0]["nominal_second_action"], dtype=np.float64)
    steps = {int(item["state_step"]) for item in records}
    _require(
        first.shape == (7,) and second.shape == (7,) and len(steps) == 1
        and all(
            np.array_equal(
                np.asarray(item["nominal_first_action"], dtype=np.float64), first
            )
            and np.array_equal(
                np.asarray(item["nominal_second_action"], dtype=np.float64), second
            )
            for item in records
        ),
        "oracle affine safe-set nominal chunk differs: %s" % case_id,
    )
    xyz = np.asarray([item["candidate_first_xyz"] for item in records], dtype=np.float64)
    margins = np.asarray(
        [item["minimum_substep_clearance_m"] for item in records], dtype=np.float64
    )
    return {
        "records": records,
        "nominal_first": first,
        "nominal_second": second,
        "state_step": next(iter(steps)),
        "xyz": xyz,
        "margins": margins,
        "raw_safe": [bool(item["D_sim_raw_safe"]) for item in records],
        "grid_indexes": [int(item["grid_index"]) for item in records],
        "action_lower": np.min(xyz, axis=0),
        "action_upper": np.max(xyz, axis=0),
        "current_clearance_m": np.asarray(
            records[0]["current_clearance_m"], dtype=np.float64
        ),
    }


def _exact_two_step_verification(
    *, runtime: Mapping[str, Any], case: Mapping[str, Any],
    archived: Mapping[str, Any], geometry_placeholder: Mapping[str, Any],
    geometry_config: Mapping[str, Any], exact_box_config: Mapping[str, Any],
    state_step: int, nominal_first: Any, nominal_second: Any,
    expected_current_clearance_m: Any, candidate_xyz: Any,
) -> dict[str, Any]:
    import numpy as np
    from main.multilink_ellipsoid.oracle_affine import SubstepEightConstraintProbe

    env = probe_env = None
    try:
        env, probe_env, _, _, setup = _build_pair(runtime, case)
        geometry, exact_boxes = _geometry(
            geometry_config=geometry_config, exact_box_config=exact_box_config,
            archived=geometry_placeholder, env=env,
            obstacle_name=setup["obstacle_name"],
        )
        for index in range(state_step):
            env.step(_canonical_action(archived["actions"][index], index).tolist())
        probe = SubstepEightConstraintProbe(
            probe_env, geometry, active_obstacle_name=setup["obstacle_name"],
            quadratic_tolerance=1.0e-6, contact_distance_threshold_m=0.0,
            obstacle_primitive_union=exact_boxes,
        )
        context = feature_context(env, probe)
        _require(
            np.allclose(
                np.asarray(context["current_clearance_m"], dtype=np.float64),
                np.asarray(expected_current_clearance_m, dtype=np.float64),
                rtol=0.0, atol=1.0e-12,
            ),
            "oracle affine safe-set reconstructed state differs",
        )
        archived_first = _canonical_action(archived["actions"][state_step], state_step)
        archived_second = _canonical_action(
            archived["actions"][state_step + 1], state_step + 1
        )
        _require(
            np.array_equal(archived_first, np.asarray(nominal_first, dtype=np.float64))
            and np.array_equal(
                archived_second, np.asarray(nominal_second, dtype=np.float64)
            ),
            "oracle affine safe-set archived chunk differs",
        )
        nominal_chunk = probe.rollout_chunk(env, [archived_first, archived_second])
        nominal_summary = summarize_chunk(nominal_chunk)
        action = archived_first.copy()
        action[:3] = np.asarray(candidate_xyz, dtype=np.float64)
        candidate_chunk = probe.rollout_chunk(env, [action, archived_second])
        summary = summarize_chunk(candidate_chunk)
        all_eight = np.min(
            np.asarray(
                [
                    item["minimum_substep_clearance_m"]
                    for item in candidate_chunk["transitions"]
                ],
                dtype=np.float64,
            ),
            axis=0,
        )
        _require(all_eight.shape == (8,), "oracle affine safe-set row count differs")
        return {
            "nominal_minimum_distal_clearance_m": float(
                min(nominal_summary["minimum_substep_clearance_m"])
            ),
            "minimum_all_eight_substep_clearance_m": all_eight.tolist(),
            "minimum_distal_substep_clearance_m": float(np.min(all_eight[:7])),
            "minimum_released_AEGIS_EE_proxy_substep_clearance_m": float(all_eight[7]),
            "D_opt_seven_distal_safe": bool(np.all(all_eight[:7] >= 0.0)),
            "released_AEGIS_EE_proxy_safe": bool(all_eight[7] >= 0.0),
            "D_sim_raw_safe": bool(summary["D_sim_raw_safe"]),
            "raw_protected_contact_count": int(summary["raw_protected_contact_count"]),
            "maximum_within_step_obstacle_l1_displacement_m": float(
                summary["maximum_within_step_obstacle_l1_displacement_m"]
            ),
            "next_state_sha256": summary["next_state_sha256"],
            "minimum_substep_witnesses": summary["minimum_substep_witnesses"],
        }
    finally:
        if probe_env is not None:
            probe_env.close()
        if env is not None:
            env.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--population-manifest", type=Path, required=True)
    parser.add_argument("--selected-manifest", type=Path, required=True)
    parser.add_argument("--archived-root", type=Path, required=True)
    parser.add_argument("--geometry-config", type=Path, required=True)
    parser.add_argument("--exact-box-config", type=Path, required=True)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dataset-result", type=Path, required=True)
    parser.add_argument("--dataset-validation", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter_ns()

    from main.evaluate_safelibero_aegis import _runtime_imports, read_jsonl
    from main.multilink_ellipsoid.obstacle_primitives import load_obstacle_primitive_config
    from main.multilink_ellipsoid.shadow import allocation_record, load_shadow_config

    config = load_oracle_affine_safe_set_config(args.config.resolve())
    source_config = load_two_step_config(args.source_config.resolve())
    dataset = _load(args.dataset.resolve())
    dataset_result = _load(args.dataset_result.resolve())
    dataset_validation = _load(args.dataset_validation.resolve())
    source = config["source_dataset"]
    _require(
        _file_sha256(args.dataset.resolve()) == source["dataset_file_sha256"]
        and _file_sha256(args.dataset_result.resolve())
        == source["dataset_result_file_sha256"]
        and _file_sha256(args.dataset_validation.resolve())
        == source["dataset_validation_file_sha256"],
        "oracle affine safe-set source artifact differs",
    )
    _require(
        dataset.get("schema_version") == TWO_STEP_DATASET_SCHEMA
        and dataset.get("source_commit") == source["dataset_source_commit"]
        and dataset.get("dataset_payload_sha256")
        == _hash_without(dataset, "dataset_payload_sha256")
        and dataset_result.get("decision", {}).get("dataset_gate_pass") is True
        and dataset_validation.get("neural_training_authorized") is True,
        "oracle affine safe-set source dataset gate differs",
    )
    selected = load_selected_manifest(args.selected_manifest.resolve(), source_config)
    _require(
        dataset.get("selected_manifest_file_sha256")
        == _file_sha256(args.selected_manifest.resolve()),
        "oracle affine safe-set selected manifest differs",
    )
    selected_by_case = {item["case_id"]: item for item in selected}
    _require(
        set(config["test_case_ids"]).issubset(selected_by_case),
        "oracle affine safe-set selected cases missing",
    )
    population = {
        item["case_id"]: item for item in read_jsonl(args.population_manifest.resolve())
    }
    geometry_config = load_shadow_config(args.geometry_config.resolve())
    exact_box_config = load_obstacle_primitive_config(args.exact_box_config.resolve())
    primary_row = selected_by_case["vlsa-t1-goal-ii-t0-e05"]
    placeholder_path = args.archived_root.resolve() / primary_row["archived_relative_path"]
    geometry_placeholder = _load(placeholder_path)
    _require(
        _file_sha256(placeholder_path) == primary_row["archived_file_sha256"]
        and geometry_placeholder.get("result_payload_sha256")
        == primary_row["archived_payload_sha256"],
        "oracle affine safe-set geometry placeholder differs",
    )
    runtime = _runtime_imports(include_aegis=False)
    case_results = {}
    certificate_settings = config["affine_certificate"]
    for case_id in config["test_case_ids"]:
        grid = _case_grid(
            dataset, case_id, int(source["expected_grid_count_per_case"])
        )
        certificate = fit_candidate_conditioned_affine_certificate(
            grid["xyz"], grid["margins"], grid["nominal_first"][:3],
            grid["raw_safe"], grid["grid_indexes"],
            one_sided_padding_m=float(
                certificate_settings["one_sided_padding_m"]
            ),
            target_clearance_m=float(
                certificate_settings["target_clearance_m"]
            ),
            postcheck_tolerance_m=float(
                certificate_settings["coefficient_postcheck_tolerance_m"]
            ),
        )
        qp = solve_affine_certificate_qp(
            grid["nominal_first"][:3], grid["action_lower"], grid["action_upper"],
            certificate, config["optimizer"],
        )
        exact = None
        if qp["valid"]:
            row = selected_by_case[case_id]
            archived_path = args.archived_root.resolve() / row["archived_relative_path"]
            archived = _load(archived_path)
            _require(
                _file_sha256(archived_path) == row["archived_file_sha256"]
                and archived.get("result_payload_sha256")
                == row["archived_payload_sha256"],
                "oracle affine safe-set archived result differs: %s" % case_id,
            )
            exact = _exact_two_step_verification(
                runtime=runtime, case=population[case_id], archived=archived,
                geometry_placeholder=geometry_placeholder,
                geometry_config=geometry_config, exact_box_config=exact_box_config,
                state_step=int(grid["state_step"]),
                nominal_first=grid["nominal_first"],
                nominal_second=grid["nominal_second"],
                expected_current_clearance_m=grid["current_clearance_m"],
                candidate_xyz=qp["candidate_xyz"],
            )
        seven_input_rows = bool(
            qp.get("diagnostics", {}).get("input_constraint_count") == 7
        )
        exact_pass = bool(
            exact is not None
            and exact["D_opt_seven_distal_safe"]
            and exact["released_AEGIS_EE_proxy_safe"]
            and exact["D_sim_raw_safe"]
            and exact["maximum_within_step_obstacle_l1_displacement_m"]
            <= float(
                config["exact_verification"]
                ["maximum_per_step_obstacle_l1_displacement_m"]
            )
        )
        case_results[case_id] = {
            "state_step": int(grid["state_step"]),
            "nominal_first_action": grid["nominal_first"].tolist(),
            "nominal_second_action": grid["nominal_second"].tolist(),
            "action_lower": grid["action_lower"].tolist(),
            "action_upper": grid["action_upper"].tolist(),
            "certificate": certificate,
            "qp": qp,
            "valid_seven_input_row_qp": seven_input_rows,
            "exact_two_step_verification": exact,
            "case_gate_pass": bool(
                certificate["valid"] and qp["valid"] and seven_input_rows and exact_pass
            ),
        }
    representation_go = bool(
        all(item["case_gate_pass"] for item in case_results.values())
    )
    result = {
        "schema_version": ORACLE_AFFINE_SAFE_SET_RESULT_SCHEMA,
        "status": "complete", "scientific_result": True,
        "claim_scope": config["claim_scope"],
        "source": _git_identity(args.repo_root.resolve(), args.expected_commit),
        "allocation": allocation_record(), "config": config,
        "dataset": {
            "path": str(args.dataset.resolve()),
            "file_sha256": _file_sha256(args.dataset.resolve()),
            "payload_sha256": dataset["dataset_payload_sha256"],
        },
        "case_results": case_results,
        "decision": {
            "representation_go": representation_go,
            "affine_coefficient_target_collection_authorized": representation_go,
            "neural_training_authorized": False,
            "closed_loop_e05_authorized": False,
            "stop_reason": (
                None if representation_go
                else "one_or_more_registered_affine_representation_cases_failed"
            ),
        },
        "wall_seconds": (time.perf_counter_ns() - started) * 1.0e-9,
    }
    result["result_payload_sha256"] = _hash_without(result, "result_payload_sha256")
    _atomic_write(args.output.resolve(), result)
    print(json.dumps(result["decision"], sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
