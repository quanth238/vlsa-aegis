#!/usr/bin/env python3
"""Collect grouped two-step L5--L7 execution-margin supervision on H100."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping, Optional, Sequence

from main.multilink_ellipsoid.execution_margin_nn import CONSTRAINT_ORDER
from main.multilink_ellipsoid.two_step_margin import (
    TWO_STEP_DATASET_RESULT_SCHEMA,
    TWO_STEP_DATASET_SCHEMA,
    boundary_category,
    feature_context,
    feature_vectors,
    grid_actions,
    load_selected_manifest,
    load_two_step_config,
    select_balanced_anchors,
    summarize_chunk,
)
from scripts.collect_distal_boundary_generalization_moka10 import (
    _canonical_action,
    _restore_env,
    _snapshot_env,
)
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


def _record(
    *, case_row: Mapping[str, Any], source: str, group_id: str,
    grid_index: Optional[int], anchor_grid_index: Optional[int],
    probe_dimension: Optional[int], probe_sign: Optional[int],
    state_step: int, nominal_first: Any, nominal_second: Any,
    candidate_first_xyz: Any, context: Mapping[str, Any], chunk: Mapping[str, Any],
    band_m: float,
) -> dict[str, Any]:
    import numpy as np

    summary = summarize_chunk(chunk)
    global_feature, pair_features = feature_vectors(
        context, nominal_first[:3], candidate_first_xyz, nominal_second[:3]
    )
    minimum = np.asarray(summary["minimum_substep_clearance_m"], dtype=np.float64)
    current = np.asarray(context["current_clearance_m"], dtype=np.float64)
    category = boundary_category(minimum, band_m)
    return {
        "record_index": -1,
        "case_id": str(case_row["case_id"]),
        "task_level_group_id": str(case_row["task_level_group_id"]),
        "episode_split": str(case_row["split"]),
        "split": str(case_row["split"]) if category.startswith("boundary_") else "excluded_far",
        "state_step": int(state_step),
        "source": str(source),
        "group_id": str(group_id),
        "grid_index": grid_index,
        "anchor_grid_index": anchor_grid_index,
        "probe_dimension": probe_dimension,
        "probe_sign": probe_sign,
        "nominal_first_action": np.asarray(nominal_first, dtype=np.float64).tolist(),
        "nominal_second_action": np.asarray(nominal_second, dtype=np.float64).tolist(),
        "candidate_first_xyz": np.asarray(candidate_first_xyz, dtype=np.float64).tolist(),
        "global_feature_vector": global_feature.tolist(),
        "pair_feature_vectors": pair_features.tolist(),
        "current_clearance_m": current.tolist(),
        "start_obstacle_witness_indexes": np.asarray(
            context["start_obstacle_witness_indexes"], dtype=np.int64
        ).tolist(),
        "minimum_substep_clearance_m": minimum.tolist(),
        "minimum_substep_witnesses": list(summary["minimum_substep_witnesses"]),
        "active_constraint_index": int(np.argmin(minimum)),
        "category": category,
        "D_opt_proxy_safe": bool(summary["D_opt_proxy_safe"]),
        "D_sim_raw_safe": bool(summary["D_sim_raw_safe"]),
        "raw_protected_contact_count": int(summary["raw_protected_contact_count"]),
        "maximum_within_step_obstacle_l1_displacement_m": float(
            summary["maximum_within_step_obstacle_l1_displacement_m"]
        ),
        "next_state_sha256": str(summary["next_state_sha256"]),
        "env_step_wall_seconds": float(summary["env_step_wall_seconds"]),
        "gradient_m_per_action": None,
        "gradient_valid_rows": None,
    }


def collect(
    *, repo_root: Path, population_manifest_path: Path,
    selected_manifest_path: Path, archived_root: Path,
    geometry_config_path: Path, exact_box_config_path: Path,
    config_path: Path, expected_commit: str, dataset_path: Path,
) -> dict[str, Any]:
    import numpy as np

    from main.evaluate_safelibero_aegis import (
        _runtime_imports,
        pairing_record,
        read_jsonl,
        validate_case_row,
    )
    from main.multilink_ellipsoid.obstacle_primitives import (
        load_obstacle_primitive_config,
    )
    from main.multilink_ellipsoid.oracle_affine import SubstepEightConstraintProbe
    from main.multilink_ellipsoid.shadow import allocation_record, load_shadow_config

    started = time.perf_counter_ns()
    config = load_two_step_config(config_path)
    selected = load_selected_manifest(selected_manifest_path, config)
    _require(
        _file_sha256(population_manifest_path)
        == config["source_population_manifest_sha256"],
        "two-step population manifest differs",
    )
    _require(
        _file_sha256(geometry_config_path)
        == config["geometry_config_file_sha256"],
        "two-step geometry config differs",
    )
    _require(
        _file_sha256(exact_box_config_path)
        == config["exact_box_config_file_sha256"],
        "two-step exact-box config differs",
    )
    geometry_config = load_shadow_config(geometry_config_path)
    exact_box_config = load_obstacle_primitive_config(exact_box_config_path)
    population = {item["case_id"]: item for item in read_jsonl(population_manifest_path)}
    source = _git_identity(repo_root, expected_commit)
    allocation = allocation_record()
    runtime = _runtime_imports(include_aegis=False)
    primary_row = next(
        item for item in selected if item["case_id"] == "vlsa-t1-goal-ii-t0-e05"
    )
    geometry_placeholder_path = archived_root / primary_row["archived_relative_path"]
    geometry_placeholder = _load(geometry_placeholder_path)
    _require(
        _file_sha256(geometry_placeholder_path)
        == primary_row["archived_file_sha256"]
        and geometry_placeholder.get("result_payload_sha256")
        == primary_row["archived_payload_sha256"],
        "two-step geometry placeholder differs",
    )
    settings = config["sampling"]
    band = float(settings["boundary_band_m"])
    epsilon = float(settings["finite_difference_epsilon_action"])
    all_records: list[dict[str, Any]] = []
    episode_results = []
    pairings = {}

    for selected_row in selected:
        case_id = str(selected_row["case_id"])
        archived_path = archived_root / selected_row["archived_relative_path"]
        archived = _load(archived_path)
        _require(
            archived_path.is_file()
            and not archived_path.is_symlink()
            and _file_sha256(archived_path) == selected_row["archived_file_sha256"]
            and archived.get("result_payload_sha256")
            == selected_row["archived_payload_sha256"]
            and archived.get("case_id") == case_id
            and archived.get("obstacle", {}).get("active_name")
            == "moka_pot_obstacle_1"
            and len(archived.get("actions", [])) == int(selected_row["action_count"])
            and archived.get("action_invariance_ledger", {}).get(
                "executed_sequence_sha256"
            )
            == selected_row["executed_sequence_sha256"],
            "two-step archived case differs: %s" % case_id,
        )
        _require(case_id in population, "two-step population case missing")
        case = population[case_id]
        validate_case_row(case, repo_root)
        env = probe_env = None
        try:
            env, probe_env, task, observation, setup = _build_pair(runtime, case)
            _require(
                setup["obstacle_name"] == "moka_pot_obstacle_1",
                "two-step active obstacle differs",
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
                "manifest_row_sha256", "initial_state_sha256",
                "initial_observation_sha256", "settled_simulator_state_sha256",
                "settled_active_obstacle_position_sha256",
                "policy_noise_schedule_sha256",
            ):
                _require(
                    pairing[key] == archived["pairing"][key],
                    "two-step pairing differs for %s: %s" % (case_id, key),
                )
            pairings[case_id] = pairing
            geometry, exact_boxes = _geometry(
                geometry_config=geometry_config,
                exact_box_config=exact_box_config,
                archived=geometry_placeholder,
                env=env,
                obstacle_name=setup["obstacle_name"],
            )
            probe = SubstepEightConstraintProbe(
                probe_env, geometry, active_obstacle_name=setup["obstacle_name"],
                quadratic_tolerance=1.0e-6, contact_distance_threshold_m=0.0,
                obstacle_primitive_union=exact_boxes,
            )
            actions = archived["actions"]
            collision_step = int(selected_row["collision_first_step"])
            search_start = max(
                0, collision_step - int(config["state"]["search_start_offset_actions"])
            )
            for step in range(search_start):
                env.step(_canonical_action(actions[step], step).tolist())
            crossing_step = None
            searched = []
            snapshots = {}
            for step in range(search_start, min(collision_step + 1, len(actions) - 1)):
                snapshots[step] = _snapshot_env(env)
                first = _canonical_action(actions[step], step)
                second = _canonical_action(actions[step + 1], step + 1)
                chunk = probe.rollout_chunk(env, [first, second])
                summary = summarize_chunk(chunk)
                context = feature_context(env, probe)
                current = np.asarray(context["current_clearance_m"], dtype=np.float64)
                minimum = np.asarray(summary["minimum_substep_clearance_m"], dtype=np.float64)
                crossing = bool(np.all(current >= 0.0) and np.any(minimum < 0.0))
                searched.append(
                    {
                        "step": step,
                        "minimum_start_m": float(np.min(current)),
                        "minimum_two_step_m": float(np.min(minimum)),
                        "crossing": crossing,
                    }
                )
                if crossing:
                    crossing_step = step
                    break
                env.step(first.tolist())
            if crossing_step is None:
                episode_results.append(
                    {
                        "case_id": case_id, "split": selected_row["split"],
                        "task_level_group_id": selected_row["task_level_group_id"],
                        "eligible": False,
                        "reason": "no_two_step_crossing_in_registered_window",
                        "search": searched,
                    }
                )
                continue
            selected_step = None
            selected_nominal = None
            selected_context = None
            grid_records: list[dict[str, Any]] = []
            anchors: list[int] = []
            boundary_search = []
            last_error = "no_registered_two_step_boundary_state"
            for offset in config["state"]["candidate_state_offsets_from_first_crossing"]:
                candidate_step = crossing_step + int(offset)
                if candidate_step not in snapshots or candidate_step + 1 >= len(actions):
                    continue
                _restore_env(env, snapshots[candidate_step])
                first = _canonical_action(actions[candidate_step], candidate_step)
                second = _canonical_action(actions[candidate_step + 1], candidate_step + 1)
                context = feature_context(env, probe)
                _, _, candidates = grid_actions(first[:3], config)
                candidate_records = []
                for grid_index, xyz in enumerate(candidates):
                    action = first.copy()
                    action[:3] = xyz
                    candidate_records.append(
                        _record(
                            case_row=selected_row, source="grid",
                            group_id="%s::grid_%04d" % (case_id, grid_index),
                            grid_index=grid_index, anchor_grid_index=None,
                            probe_dimension=None, probe_sign=None,
                            state_step=candidate_step, nominal_first=first,
                            nominal_second=second, candidate_first_xyz=xyz,
                            context=context,
                            chunk=probe.rollout_chunk(env, [action, second]),
                            band_m=band,
                        )
                    )
                counts = {
                    name: sum(item["category"] == name for item in candidate_records)
                    for name in (
                        "boundary_safe", "boundary_unsafe", "far_safe", "far_unsafe"
                    )
                }
                try:
                    candidate_anchors = select_balanced_anchors(candidate_records, config)
                    error = None
                except ValueError as exception:
                    candidate_anchors = []
                    error = str(exception)
                    last_error = error
                boundary_search.append(
                    {
                        "state_offset_from_first_crossing": int(offset),
                        "state_step": candidate_step,
                        "grid_category_counts": counts,
                        "balanced_anchor_gate_pass": bool(candidate_anchors),
                        "reason": error,
                    }
                )
                grid_records = candidate_records
                if candidate_anchors:
                    selected_step = candidate_step
                    selected_nominal = (first, second)
                    selected_context = context
                    anchors = candidate_anchors
                    break
            if selected_step is None or selected_nominal is None or selected_context is None:
                offset_index = len(all_records)
                for index, item in enumerate(grid_records):
                    item["record_index"] = offset_index + index
                all_records.extend(grid_records)
                episode_results.append(
                    {
                        "case_id": case_id, "split": selected_row["split"],
                        "task_level_group_id": selected_row["task_level_group_id"],
                        "eligible": False, "reason": last_error,
                        "first_two_step_crossing_step": crossing_step,
                        "search": searched, "boundary_state_search": boundary_search,
                    }
                )
                continue
            _restore_env(env, snapshots[selected_step])
            first, second = selected_nominal
            grid_by_index = {int(item["grid_index"]): item for item in grid_records}
            probes = {}
            episode_records = list(grid_records)
            for anchor_index in anchors:
                anchor = grid_by_index[anchor_index]
                group_id = "%s::anchor_%04d" % (case_id, anchor_index)
                anchor["group_id"] = group_id
                anchor["split"] = str(selected_row["split"])
                base = np.asarray(anchor["candidate_first_xyz"], dtype=np.float64)
                for dimension in range(3):
                    for sign in (-1, 1):
                        xyz = base.copy()
                        xyz[dimension] += sign * epsilon
                        action = first.copy()
                        action[:3] = xyz
                        item = _record(
                            case_row=selected_row, source="gradient_probe",
                            group_id=group_id, grid_index=None,
                            anchor_grid_index=anchor_index,
                            probe_dimension=dimension, probe_sign=sign,
                            state_step=selected_step, nominal_first=first,
                            nominal_second=second, candidate_first_xyz=xyz,
                            context=selected_context,
                            chunk=probe.rollout_chunk(env, [action, second]),
                            band_m=band,
                        )
                        item["category"] = anchor["category"]
                        item["split"] = str(selected_row["split"])
                        episode_records.append(item)
                        probes[(anchor_index, dimension, sign)] = item
            _require(
                len(probes) == int(settings["expected_gradient_probe_count_per_episode"]),
                "two-step gradient probe count differs",
            )
            stable_active = 0
            for anchor_index in anchors:
                anchor = grid_by_index[anchor_index]
                gradients = np.zeros((7, 3), dtype=np.float64)
                valid = np.ones(7, dtype=bool)
                for dimension in range(3):
                    minus = probes[(anchor_index, dimension, -1)]
                    plus = probes[(anchor_index, dimension, 1)]
                    gradients[:, dimension] = (
                        np.asarray(plus["minimum_substep_clearance_m"], dtype=np.float64)
                        - np.asarray(minus["minimum_substep_clearance_m"], dtype=np.float64)
                    ) / (2.0 * epsilon)
                    for row in range(7):
                        valid[row] = bool(
                            valid[row]
                            and anchor["minimum_substep_witnesses"][row]
                            == minus["minimum_substep_witnesses"][row]
                            == plus["minimum_substep_witnesses"][row]
                        )
                anchor["gradient_m_per_action"] = gradients.tolist()
                anchor["gradient_valid_rows"] = valid.tolist()
                stable_active += int(valid[int(anchor["active_constraint_index"])])
            record_offset = len(all_records)
            for index, item in enumerate(episode_records):
                item["record_index"] = record_offset + index
            all_records.extend(episode_records)
            counts = {
                name: sum(item["category"] == name for item in grid_records)
                for name in (
                    "boundary_safe", "boundary_unsafe", "far_safe", "far_unsafe"
                )
            }
            episode_results.append(
                {
                    "case_id": case_id, "split": selected_row["split"],
                    "task_level_group_id": selected_row["task_level_group_id"],
                    "eligible": bool(stable_active > 0),
                    "reason": None if stable_active > 0 else "no_stable_active_gradient_anchor",
                    "first_two_step_crossing_step": crossing_step,
                    "selected_step": selected_step,
                    "nominal_first_action": first.tolist(),
                    "nominal_second_action": second.tolist(),
                    "search": searched, "boundary_state_search": boundary_search,
                    "grid_category_counts": counts,
                    "gradient_anchor_count": len(anchors),
                    "stable_active_gradient_anchor_count": stable_active,
                    "record_count": len(episode_records),
                }
            )
        finally:
            if probe_env is not None:
                probe_env.close()
            if env is not None:
                env.close()

    dataset_gate = bool(
        len(episode_results) == len(selected)
        and all(item["eligible"] for item in episode_results)
    )
    dataset = {
        "schema_version": TWO_STEP_DATASET_SCHEMA,
        "source_commit": source["commit"],
        "config_file_sha256": config["config_file_sha256"],
        "config_payload_sha256": config["config_payload_sha256"],
        "selected_manifest_file_sha256": _file_sha256(selected_manifest_path),
        "constraint_order": list(CONSTRAINT_ORDER),
        "episode_results": episode_results,
        "summary": {
            "selected_case_count": len(selected),
            "eligible_case_count": sum(item["eligible"] for item in episode_results),
            "record_count": len(all_records),
            "learning_record_count": sum(
                item["split"] in {"train", "validation", "test"}
                for item in all_records
            ),
            "dataset_gate_pass": dataset_gate,
        },
        "records": all_records,
    }
    dataset["dataset_payload_sha256"] = _hash_without(
        dataset, "dataset_payload_sha256"
    )
    _atomic_write(dataset_path, dataset)
    result = {
        "schema_version": TWO_STEP_DATASET_RESULT_SCHEMA,
        "status": "complete", "scientific_result": True,
        "claim_scope": config["claim_scope"], "source": source,
        "allocation": allocation, "config": config,
        "selected_manifest": {
            "path": str(selected_manifest_path),
            "file_sha256": _file_sha256(selected_manifest_path),
        },
        "archived_table1_root": {"path": str(archived_root), "read_only": True},
        "geometry_placeholder": {
            "path": str(geometry_placeholder_path),
            "file_sha256": _file_sha256(geometry_placeholder_path),
            "use": (
                "proper_rotation_constructor_placeholder_only_all_labels_use_"
                "live_exact_15_box_union"
            ),
        },
        "pairings": pairings,
        "dataset": {
            "path": str(dataset_path), "file_sha256": _file_sha256(dataset_path),
            "payload_sha256": dataset["dataset_payload_sha256"],
        },
        "dataset_summary": dataset["summary"],
        "decision": {
            "dataset_gate_pass": dataset_gate,
            "neural_training_authorized": dataset_gate,
            "stop_reason": None if dataset_gate else "one_or_more_two_step_episode_gates_failed",
        },
        "wall_seconds": (time.perf_counter_ns() - started) * 1.0e-9,
    }
    result["result_payload_sha256"] = _hash_without(result, "result_payload_sha256")
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--population-manifest", type=Path, required=True)
    parser.add_argument("--selected-manifest", type=Path, required=True)
    parser.add_argument("--archived-root", type=Path, required=True)
    parser.add_argument("--geometry-config", type=Path, required=True)
    parser.add_argument("--exact-box-config", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = collect(
        repo_root=args.repo_root.resolve(),
        population_manifest_path=args.population_manifest.resolve(),
        selected_manifest_path=args.selected_manifest.resolve(),
        archived_root=args.archived_root.resolve(),
        geometry_config_path=args.geometry_config.resolve(),
        exact_box_config_path=args.exact_box_config.resolve(),
        config_path=args.config.resolve(), expected_commit=args.expected_commit,
        dataset_path=args.dataset.resolve(),
    )
    _atomic_write(args.output.resolve(), result)
    print(json.dumps(result["decision"], sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
