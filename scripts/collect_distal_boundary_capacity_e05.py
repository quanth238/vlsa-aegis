#!/usr/bin/env python3
"""Collect the no-training action-188 boundary-capacity dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any, Mapping, Optional, Sequence

from main.multilink_ellipsoid.boundary_capacity import (
    BOUNDARY_DATASET_RESULT_SCHEMA,
    BOUNDARY_DATASET_SCHEMA,
    assign_grouped_splits,
    critical_category,
    load_boundary_capacity_config,
    select_gradient_anchor_indexes,
    structured_grid_actions,
)
from main.multilink_ellipsoid.execution_margin_nn import feature_vector
from scripts.evaluate_distal_execution_margin_nn_e05 import (
    _build_pair,
    _geometry,
    _replay_prefix,
    _source_action,
    _verify_sources,
)
from scripts.replay_distal_three_ellipsoid_multicbf import (
    CASE_ID,
    _atomic_write,
    _file_sha256,
    _git_identity,
    _load,
    _require,
    _sha256,
)


def _record_transition(
    *,
    index: int,
    source: str,
    group_id: str,
    grid_index: Optional[int],
    anchor_grid_index: Optional[int],
    probe_dimension: Optional[int],
    probe_sign: Optional[int],
    nominal_xyz: Any,
    candidate_xyz: Any,
    transition: Mapping[str, Any],
    critical_index: int,
    boundary_band_m: float,
) -> dict[str, Any]:
    import numpy as np

    start = transition["substeps"][0]
    minimum = np.asarray(
        transition["minimum_substep_clearance_m"][:7], dtype=np.float64
    )
    current = np.asarray(start["clearance_m"][:7], dtype=np.float64)
    raw_safe = bool(
        transition["raw_protected_contact_count"] == 0
        and transition["maximum_within_step_obstacle_l1_displacement_m"]
        <= 1.0e-4
    )
    return {
        "record_index": int(index),
        "source": str(source),
        "group_id": str(group_id),
        "grid_index": None if grid_index is None else int(grid_index),
        "anchor_grid_index": (
            None if anchor_grid_index is None else int(anchor_grid_index)
        ),
        "probe_dimension": (
            None if probe_dimension is None else int(probe_dimension)
        ),
        "probe_sign": None if probe_sign is None else int(probe_sign),
        "candidate_xyz": np.asarray(candidate_xyz, dtype=np.float64).tolist(),
        "feature_vector": feature_vector(
            start, nominal_xyz, candidate_xyz
        ).tolist(),
        "current_clearance_m": current.tolist(),
        "minimum_substep_clearance_m": minimum.tolist(),
        "minimum_substep_witnesses": list(
            transition["minimum_substep_witnesses"][:7]
        ),
        "D_opt_proxy_safe": bool(np.all(minimum >= 0.0)),
        "D_sim_raw_safe": raw_safe,
        "raw_protected_contact_count": int(
            transition["raw_protected_contact_count"]
        ),
        "maximum_within_step_obstacle_l1_displacement_m": float(
            transition["maximum_within_step_obstacle_l1_displacement_m"]
        ),
        "geometry_consistency": dict(transition["geometry_consistency"]),
        "next_state_sha256": str(transition["next_state_sha256"]),
        "env_step_wall_seconds": float(transition["env_step_wall_seconds"]),
        "category": critical_category(
            float(minimum[critical_index]), boundary_band_m
        ),
        "split": None,
        "gradient_m_per_action": None,
        "gradient_valid_rows": None,
    }


def collect(
    *,
    repo_root: Path,
    manifest_path: Path,
    archived_path: Path,
    false_safe_result_path: Path,
    obstacle_discovery_result_path: Path,
    baseline_nn_result_path: Path,
    geometry_config_path: Path,
    exact_box_config_path: Path,
    config_path: Path,
    expected_commit: str,
    dataset_path: Path,
) -> dict[str, Any]:
    import numpy as np

    from main.evaluate_safelibero_aegis import (
        _runtime_imports,
        pairing_record,
        read_jsonl,
        validate_case_row,
    )
    from main.multilink_ellipsoid.obstacle_primitives import (
        EXACT_BOX_OBSTACLE_SCHEMA,
        load_obstacle_primitive_config,
    )
    from main.multilink_ellipsoid.oracle_affine import SubstepEightConstraintProbe
    from main.multilink_ellipsoid.shadow import allocation_record, load_shadow_config

    started = time.perf_counter_ns()
    config = load_boundary_capacity_config(config_path)
    geometry_config = load_shadow_config(geometry_config_path)
    exact_box_config = load_obstacle_primitive_config(exact_box_config_path)
    _require(
        exact_box_config["schema_version"] == EXACT_BOX_OBSTACLE_SCHEMA
        and exact_box_config["config_payload_sha256"]
        == config["immutable_sources"]["exact_box_config_payload_sha256"],
        "boundary-capacity exact-box config differs",
    )
    archived, false_safe, discovery = _verify_sources(
        archived_path=archived_path,
        false_safe_result_path=false_safe_result_path,
        obstacle_discovery_result_path=obstacle_discovery_result_path,
        geometry_config_path=geometry_config_path,
        exact_box_config_path=exact_box_config_path,
        config=config,
    )
    baseline_nn = _load(baseline_nn_result_path)
    _require(
        _file_sha256(baseline_nn_result_path)
        == config["immutable_sources"]["baseline_nn_result_file_sha256"]
        and baseline_nn.get("result_payload_sha256")
        == config["immutable_sources"]["baseline_nn_result_payload_sha256"]
        and baseline_nn.get("config", {}).get("config_file_sha256")
        == config["immutable_sources"]["previous_nn_config_file_sha256"]
        and baseline_nn.get("research_direction_go") is False
        and baseline_nn.get("stop_reason")
        == "residual_model_failed_held_out_conservative_learnability_gate",
        "boundary-capacity baseline neural result differs",
    )
    matches = [
        row for row in read_jsonl(manifest_path) if row.get("case_id") == CASE_ID
    ]
    _require(len(matches) == 1, "boundary-capacity manifest row is not unique")
    case = matches[0]
    validate_case_row(case, repo_root)
    source = _git_identity(repo_root, expected_commit)
    allocation = allocation_record()
    runtime = _runtime_imports(include_aegis=False)
    step = int(config["state"]["step"])
    critical = int(config["state"]["critical_constraint_index"])
    settings = config["sampling"]
    band = float(settings["boundary_band_m"])
    epsilon = float(settings["finite_difference_epsilon_action"])
    env = probe_env = None
    try:
        env, probe_env, task, observation, setup = _build_pair(runtime, case)
        _require(
            setup["obstacle_name"] == archived["obstacle"]["active_name"],
            "boundary-capacity active obstacle differs",
        )
        pairing = pairing_record(
            case=case,
            selected_initial_state=setup["selected_initial_state"],
            settled_observation=observation,
            task_description=str(task.language),
            active_obstacle_name=setup["obstacle_name"],
            settled_simulator_state=np.asarray(
                env.sim.get_state().flatten(), dtype=np.float64
            ),
        )
        for key in (
            "manifest_row_sha256",
            "initial_state_sha256",
            "initial_observation_sha256",
            "settled_simulator_state_sha256",
            "settled_active_obstacle_position_sha256",
            "policy_noise_schedule_sha256",
        ):
            _require(
                pairing[key] == archived["pairing"][key],
                "boundary-capacity pairing differs: %s" % key,
            )
        geometry, exact_boxes = _geometry(
            geometry_config=geometry_config,
            exact_box_config=exact_box_config,
            archived=archived,
            env=env,
            obstacle_name=setup["obstacle_name"],
        )
        geometry_record = geometry.geometry_record(env)
        exact_box_record = exact_boxes.geometry_record(env)
        _replay_prefix(env, false_safe["actions"], step)
        nominal = _source_action(false_safe["actions"][step], step)
        _require(
            np.allclose(nominal[3:6], 0.0, rtol=0.0, atol=1.0e-12),
            "boundary-capacity requires translational AEGIS action",
        )
        lower, upper, grid_actions = structured_grid_actions(nominal[:3], config)
        probe = SubstepEightConstraintProbe(
            probe_env,
            geometry,
            active_obstacle_name=setup["obstacle_name"],
            quadratic_tolerance=1.0e-6,
            contact_distance_threshold_m=0.0,
            obstacle_primitive_union=exact_boxes,
        )
        nominal_transition = probe.transition(env, nominal)
        nominal_start = np.asarray(
            nominal_transition["substeps"][0]["clearance_m"][:7],
            dtype=np.float64,
        )
        nominal_minimum = np.asarray(
            nominal_transition["minimum_substep_clearance_m"][:7],
            dtype=np.float64,
        )
        _require(
            np.all(nominal_start >= 0.0)
            and nominal_minimum[critical] < 0.0,
            "boundary-capacity primary nominal transition is not the registered crossing",
        )
        nominal_summary = {
            "start_clearance_m": nominal_start.tolist(),
            "minimum_substep_clearance_m": nominal_minimum.tolist(),
            "minimum_substep_witnesses": nominal_transition[
                "minimum_substep_witnesses"
            ][:7],
            "D_opt_proxy_safe": bool(np.all(nominal_minimum >= 0.0)),
            "D_sim_raw_safe": bool(
                nominal_transition["raw_protected_contact_count"] == 0
                and nominal_transition[
                    "maximum_within_step_obstacle_l1_displacement_m"
                ]
                <= 1.0e-4
            ),
            "raw_protected_contact_count": int(
                nominal_transition["raw_protected_contact_count"]
            ),
            "next_state_sha256": nominal_transition["next_state_sha256"],
        }
        grid_records = []
        current_reference = None
        for grid_index, xyz in enumerate(grid_actions):
            action = nominal.copy()
            action[:3] = xyz
            transition = probe.transition(env, action)
            current = np.asarray(
                transition["substeps"][0]["clearance_m"][:7], dtype=np.float64
            )
            if current_reference is None:
                current_reference = current
            else:
                _require(
                    np.array_equal(current, current_reference),
                    "boundary-capacity grid start differs",
                )
            grid_records.append(
                _record_transition(
                    index=len(grid_records),
                    source="grid",
                    group_id="grid_%04d" % grid_index,
                    grid_index=grid_index,
                    anchor_grid_index=None,
                    probe_dimension=None,
                    probe_sign=None,
                    nominal_xyz=nominal[:3],
                    candidate_xyz=xyz,
                    transition=transition,
                    critical_index=critical,
                    boundary_band_m=band,
                )
            )
        anchor_indexes = select_gradient_anchor_indexes(
            grid_records, lower, upper, config
        )
        anchor_set = set(anchor_indexes)
        grid_by_index = {int(item["grid_index"]): item for item in grid_records}
        records = list(grid_records)
        probes: dict[tuple[int, int, int], dict[str, Any]] = {}
        for anchor_index in anchor_indexes:
            anchor = grid_by_index[anchor_index]
            group_id = "anchor_%04d" % anchor_index
            anchor["group_id"] = group_id
            anchor_category = str(anchor["category"])
            base_xyz = np.asarray(anchor["candidate_xyz"], dtype=np.float64)
            for dimension in range(3):
                for sign in (-1, 1):
                    xyz = base_xyz.copy()
                    xyz[dimension] += sign * epsilon
                    _require(
                        np.all(xyz >= lower - 1.0e-12)
                        and np.all(xyz <= upper + 1.0e-12),
                        "boundary-capacity finite difference leaves trust bounds",
                    )
                    action = nominal.copy()
                    action[:3] = xyz
                    transition = probe.transition(env, action)
                    record = _record_transition(
                        index=len(records),
                        source="gradient_probe",
                        group_id=group_id,
                        grid_index=None,
                        anchor_grid_index=anchor_index,
                        probe_dimension=dimension,
                        probe_sign=sign,
                        nominal_xyz=nominal[:3],
                        candidate_xyz=xyz,
                        transition=transition,
                        critical_index=critical,
                        boundary_band_m=band,
                    )
                    record["category"] = anchor_category
                    records.append(record)
                    probes[(anchor_index, dimension, sign)] = record
        _require(
            len(probes) == int(settings["expected_gradient_probe_count"]),
            "boundary-capacity gradient probe count differs",
        )
        for anchor_index in anchor_indexes:
            anchor = grid_by_index[anchor_index]
            gradients = np.zeros((7, 3), dtype=np.float64)
            valid = np.ones(7, dtype=bool)
            anchor_witness = anchor["minimum_substep_witnesses"]
            for dimension in range(3):
                minus = probes[(anchor_index, dimension, -1)]
                plus = probes[(anchor_index, dimension, 1)]
                gradients[:, dimension] = (
                    np.asarray(
                        plus["minimum_substep_clearance_m"], dtype=np.float64
                    )
                    - np.asarray(
                        minus["minimum_substep_clearance_m"], dtype=np.float64
                    )
                ) / (2.0 * epsilon)
                if settings["require_matching_substep_and_obstacle_witness"]:
                    for row in range(7):
                        valid[row] = bool(
                            valid[row]
                            and anchor_witness[row]
                            == minus["minimum_substep_witnesses"][row]
                            == plus["minimum_substep_witnesses"][row]
                        )
            anchor["gradient_m_per_action"] = gradients.tolist()
            anchor["gradient_valid_rows"] = valid.tolist()
        split_by_group = assign_grouped_splits(records, config)
        for record in records:
            record["split"] = split_by_group[str(record["group_id"])]
        _require(
            all(
                record["grid_index"] in anchor_set
                for record in grid_records
                if record["gradient_m_per_action"] is not None
            ),
            "boundary-capacity gradient leaked to non-anchor",
        )
    finally:
        if probe_env is not None:
            probe_env.close()
        if env is not None:
            env.close()

    categories = {
        name: sum(item["category"] == name for item in grid_records)
        for name in (
            "boundary_safe",
            "boundary_unsafe",
            "far_safe",
            "far_unsafe",
        )
    }
    selected_anchor_records = [grid_by_index[index] for index in anchor_indexes]
    stable_by_split = {
        name: sum(
            item["split"] == name
            and bool(item["gradient_valid_rows"][critical])
            for item in selected_anchor_records
        )
        for name in ("train", "validation", "test")
    }
    split_group_counts = {
        name: len(
            {
                item["group_id"] for item in records if item["split"] == name
            }
        )
        for name in ("train", "validation", "test")
    }
    dataset_gate = bool(
        categories["boundary_safe"]
        >= int(settings["gradient_anchor_safe_count"])
        and categories["boundary_unsafe"]
        >= int(settings["gradient_anchor_unsafe_count"])
        and min(stable_by_split.values()) >= 1
    )
    summary = {
        "state_step": step,
        "grid_action_count": len(grid_records),
        "gradient_anchor_count": len(anchor_indexes),
        "gradient_probe_count": len(probes),
        "record_count": len(records),
        "critical_constraint_index": critical,
        "critical_constraint_name": config["state"]["critical_constraint_name"],
        "nominal_transition": nominal_summary,
        "grid_category_counts": categories,
        "stable_critical_gradient_anchor_count_by_split": stable_by_split,
        "split_group_counts": split_group_counts,
        "proxy_safe_grid_count": sum(item["D_opt_proxy_safe"] for item in grid_records),
        "raw_safe_grid_count": sum(item["D_sim_raw_safe"] for item in grid_records),
        "dataset_capacity_gate_pass": dataset_gate,
        "total_clone_env_step_wall_seconds": float(
            sum(item["env_step_wall_seconds"] for item in records)
        ),
    }
    dataset = {
        "schema_version": BOUNDARY_DATASET_SCHEMA,
        "case_id": CASE_ID,
        "source_commit": source["commit"],
        "config_file_sha256": config["config_file_sha256"],
        "config_payload_sha256": config["config_payload_sha256"],
        "constraint_order": [
            "L5_part_0",
            "L5_part_1",
            "L5_part_2",
            "L6_part_0",
            "L6_part_1",
            "L7_part_0",
            "L7_part_1",
        ],
        "nominal_action": nominal.tolist(),
        "action_lower": lower.tolist(),
        "action_upper": upper.tolist(),
        "gradient_anchor_grid_indexes": anchor_indexes,
        "summary": summary,
        "records": records,
    }
    dataset["dataset_payload_sha256"] = _sha256(
        json.dumps(
            dataset, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    )
    _atomic_write(dataset_path, dataset)
    result = {
        "schema_version": BOUNDARY_DATASET_RESULT_SCHEMA,
        "status": "complete",
        "scientific_result": True,
        "case_id": CASE_ID,
        "claim_scope": config["claim_scope"],
        "source": source,
        "allocation": allocation,
        "config": config,
        "pairing": pairing,
        "immutable_sources": {
            "archived_table1": str(archived_path),
            "false_safe_action_ledger": str(false_safe_result_path),
            "exact_box_discovery": str(obstacle_discovery_result_path),
            "baseline_nn_result": str(baseline_nn_result_path),
            "discovery_schema_version": discovery.get("schema_version"),
            "read_only": True,
        },
        "geometry": {
            "accepted_robot_and_released_ee": geometry_record,
            "exact_obstacle_boxes": exact_box_record,
        },
        "dataset": {
            "path": str(dataset_path),
            "file_sha256": _file_sha256(dataset_path),
            "payload_sha256": dataset["dataset_payload_sha256"],
            "record_count": len(records),
        },
        "dataset_summary": summary,
        "decision": {
            "dataset_capacity_gate_pass": dataset_gate,
            "neural_training_authorized": dataset_gate,
            "stop_reason": (
                None
                if dataset_gate
                else "insufficient_balanced_or_stable_boundary_labels"
            ),
        },
        "wall_seconds": (time.perf_counter_ns() - started) * 1.0e-9,
        "failure": None,
    }
    result["result_payload_sha256"] = _sha256(
        json.dumps(
            result, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    )
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--archived", type=Path, required=True)
    parser.add_argument("--false-safe-result", type=Path, required=True)
    parser.add_argument("--obstacle-discovery-result", type=Path, required=True)
    parser.add_argument("--baseline-nn-result", type=Path, required=True)
    parser.add_argument("--geometry-config", type=Path, required=True)
    parser.add_argument("--exact-box-config", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = collect(
        repo_root=args.repo_root.resolve(),
        manifest_path=args.manifest.resolve(),
        archived_path=args.archived.resolve(),
        false_safe_result_path=args.false_safe_result.resolve(),
        obstacle_discovery_result_path=args.obstacle_discovery_result.resolve(),
        baseline_nn_result_path=args.baseline_nn_result.resolve(),
        geometry_config_path=args.geometry_config.resolve(),
        exact_box_config_path=args.exact_box_config.resolve(),
        config_path=args.config.resolve(),
        expected_commit=args.expected_commit,
        dataset_path=args.dataset.resolve(),
    )
    _atomic_write(args.output.resolve(), result)
    print(json.dumps(result["decision"], sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
