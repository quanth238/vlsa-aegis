#!/usr/bin/env python3
"""Collect the registered grouped multi-task distal boundary dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping, Optional, Sequence

from main.multilink_ellipsoid.boundary_generalization import (
    GENERALIZATION_DATASET_RESULT_SCHEMA,
    GENERALIZATION_DATASET_SCHEMA,
    grid_actions,
    load_generalization_config,
    load_selected_manifest,
    select_balanced_anchors,
    worst_category,
)
from main.multilink_ellipsoid.execution_margin_nn import CONSTRAINT_ORDER, feature_vector
from scripts.evaluate_distal_execution_margin_nn_e05 import _build_pair, _geometry
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _git_identity, _load, _require,
)


def _hash_without(value: Mapping[str, Any], key: str) -> str:
    payload = dict(value)
    payload.pop(key, None)
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")).hexdigest()


def _canonical_action(item: Mapping[str, Any], step: int) -> Any:
    import numpy as np

    _require(int(item.get("step", -1)) == step, "canonical action step differs")
    action = np.asarray(item.get("env_step_input"), dtype=np.float64)
    executed = np.asarray(item.get("executed"), dtype=np.float64)
    _require(
        action.shape == (7,) and np.all(np.isfinite(action))
        and np.array_equal(action, executed),
        "canonical Table-1 env.step action differs",
    )
    return action


def _snapshot_env(env: Any) -> dict[str, Any]:
    import numpy as np
    from main.multilink_ellipsoid.rollout import (
        _auxiliary_sim_snapshot, _base_env, _controller_snapshot,
    )

    base = _base_env(env)
    return {
        "simulator_state": np.asarray(
            env.sim.get_state().flatten(), dtype=np.float64
        ).copy(),
        "auxiliary": _auxiliary_sim_snapshot(env),
        "controllers": _controller_snapshot(env),
        "timestep": int(base.timestep),
        "cur_time": float(base.cur_time),
        "done": bool(base.done),
    }


def _restore_env(env: Any, snapshot: Mapping[str, Any]) -> None:
    from main.multilink_ellipsoid.rollout import (
        _base_env, _restore_auxiliary_sim_snapshot,
        _restore_controller_snapshot,
    )

    base = _base_env(env)
    env.sim.set_state_from_flattened(snapshot["simulator_state"])
    _restore_auxiliary_sim_snapshot(env, snapshot["auxiliary"])
    base.timestep = int(snapshot["timestep"])
    base.cur_time = float(snapshot["cur_time"])
    base.done = bool(snapshot["done"])
    _restore_controller_snapshot(env, snapshot["controllers"])
    env.sim.forward()
    _restore_auxiliary_sim_snapshot(env, snapshot["auxiliary"])


def _transition_record(
    *, record_index: int, case_row: Mapping[str, Any], source: str,
    group_id: str, grid_index: Optional[int], anchor_grid_index: Optional[int],
    probe_dimension: Optional[int], probe_sign: Optional[int], nominal: Any,
    candidate_xyz: Any, transition: Mapping[str, Any], band_m: float,
) -> dict[str, Any]:
    import numpy as np

    start = transition["substeps"][0]
    minimum = np.asarray(transition["minimum_substep_clearance_m"][:7], dtype=np.float64)
    current = np.asarray(start["clearance_m"][:7], dtype=np.float64)
    category = worst_category(minimum, band_m)
    return {
        "record_index": int(record_index),
        "case_id": str(case_row["case_id"]),
        "task_level_group_id": str(case_row["task_level_group_id"]),
        "episode_split": str(case_row["split"]),
        "split": str(case_row["split"]) if category.startswith("boundary_") else "excluded_far",
        "source": source,
        "group_id": group_id,
        "grid_index": grid_index,
        "anchor_grid_index": anchor_grid_index,
        "probe_dimension": probe_dimension,
        "probe_sign": probe_sign,
        "nominal_xyz": np.asarray(nominal[:3], dtype=np.float64).tolist(),
        "candidate_xyz": np.asarray(candidate_xyz, dtype=np.float64).tolist(),
        "feature_vector": feature_vector(start, nominal[:3], candidate_xyz).tolist(),
        "current_clearance_m": current.tolist(),
        "minimum_substep_clearance_m": minimum.tolist(),
        "minimum_substep_witnesses": list(transition["minimum_substep_witnesses"][:7]),
        "active_constraint_index": int(np.argmin(minimum)),
        "category": category,
        "D_opt_proxy_safe": bool(np.all(minimum >= 0.0)),
        "D_sim_raw_safe": bool(
            transition["raw_protected_contact_count"] == 0
            and transition["maximum_within_step_obstacle_l1_displacement_m"] <= 1e-4
        ),
        "raw_protected_contact_count": int(transition["raw_protected_contact_count"]),
        "maximum_within_step_obstacle_l1_displacement_m": float(
            transition["maximum_within_step_obstacle_l1_displacement_m"]
        ),
        "next_state_sha256": str(transition["next_state_sha256"]),
        "env_step_wall_seconds": float(transition["env_step_wall_seconds"]),
        "gradient_m_per_action": None,
        "gradient_valid_rows": None,
    }


def collect(
    *, repo_root: Path, population_manifest_path: Path,
    selected_manifest_path: Path, archived_root: Path, geometry_config_path: Path,
    exact_box_config_path: Path, config_path: Path, expected_commit: str,
    dataset_path: Path,
) -> dict[str, Any]:
    import numpy as np

    from main.evaluate_safelibero_aegis import (
        _runtime_imports, pairing_record, read_jsonl, validate_case_row,
    )
    from main.multilink_ellipsoid.obstacle_primitives import load_obstacle_primitive_config
    from main.multilink_ellipsoid.oracle_affine import SubstepEightConstraintProbe
    from main.multilink_ellipsoid.shadow import allocation_record, load_shadow_config

    started = time.perf_counter_ns()
    config = load_generalization_config(config_path)
    selected = load_selected_manifest(selected_manifest_path, config)
    _require(_file_sha256(geometry_config_path) == config["geometry_config_file_sha256"], "geometry config differs")
    _require(_file_sha256(exact_box_config_path) == config["exact_box_config_file_sha256"], "exact-box config differs")
    geometry_config = load_shadow_config(geometry_config_path)
    exact_box_config = load_obstacle_primitive_config(exact_box_config_path)
    population_rows = {row["case_id"]: row for row in read_jsonl(population_manifest_path)}
    source = _git_identity(repo_root, expected_commit)
    allocation = allocation_record()
    runtime = _runtime_imports(include_aegis=False)
    primary_row = next(
        row for row in selected
        if row["case_id"] == "vlsa-t1-goal-ii-t0-e05"
    )
    geometry_placeholder_path = archived_root / str(
        primary_row["archived_relative_path"]
    )
    geometry_placeholder_archived = _load(geometry_placeholder_path)
    _require(
        _file_sha256(geometry_placeholder_path)
        == primary_row["archived_file_sha256"]
        and geometry_placeholder_archived.get("result_payload_sha256")
        == primary_row["archived_payload_sha256"],
        "exact-box geometry placeholder identity differs",
    )
    settings = config["sampling"]
    band = float(settings["boundary_band_m"])
    epsilon = float(settings["finite_difference_epsilon_action"])
    all_records: list[dict[str, Any]] = []
    episode_results: list[dict[str, Any]] = []
    all_pairings: dict[str, Any] = {}

    for selected_row in selected:
        case_id = str(selected_row["case_id"])
        archived_path = archived_root / str(selected_row["archived_relative_path"])
        _require(archived_path.is_file() and not archived_path.is_symlink(), "archived case missing: %s" % case_id)
        archived = _load(archived_path)
        _require(
            _file_sha256(archived_path) == selected_row["archived_file_sha256"]
            and archived.get("result_payload_sha256") == selected_row["archived_payload_sha256"]
            and archived.get("case_id") == case_id
            and archived.get("obstacle", {}).get("active_name") == "moka_pot_obstacle_1"
            and len(archived.get("actions", [])) == int(selected_row["action_count"])
            and archived.get("action_invariance_ledger", {}).get("executed_sequence_sha256")
            == selected_row["executed_sequence_sha256"],
            "archived case identity differs: %s" % case_id,
        )
        _require(case_id in population_rows, "population case missing: %s" % case_id)
        case = population_rows[case_id]
        validate_case_row(case, repo_root)
        _require(case["task_level_group_id"] == selected_row["task_level_group_id"], "selected task group differs")
        env = probe_env = None
        try:
            env, probe_env, task, observation, setup = _build_pair(runtime, case)
            _require(setup["obstacle_name"] == "moka_pot_obstacle_1", "active obstacle differs")
            pairing = pairing_record(
                case=case,
                selected_initial_state=setup["selected_initial_state"],
                settled_observation=observation,
                task_description=str(task.language),
                active_obstacle_name=setup["obstacle_name"],
                settled_simulator_state=np.asarray(env.sim.get_state().flatten(), dtype=np.float64),
            )
            for key in (
                "manifest_row_sha256", "initial_state_sha256", "initial_observation_sha256",
                "settled_simulator_state_sha256", "settled_active_obstacle_position_sha256",
                "policy_noise_schedule_sha256",
            ):
                _require(pairing[key] == archived["pairing"][key], "pairing differs for %s: %s" % (case_id, key))
            all_pairings[case_id] = pairing
            geometry, exact_boxes = _geometry(
                geometry_config=geometry_config, exact_box_config=exact_box_config,
                archived=geometry_placeholder_archived, env=env,
                obstacle_name=setup["obstacle_name"],
            )
            probe = SubstepEightConstraintProbe(
                probe_env, geometry, active_obstacle_name=setup["obstacle_name"],
                quadratic_tolerance=1e-6, contact_distance_threshold_m=0.0,
                obstacle_primitive_union=exact_boxes,
            )
            actions = archived["actions"]
            collision_step = int(selected_row["collision_first_step"])
            search_start = max(0, collision_step - int(config["state"]["search_start_offset_actions"]))
            for step in range(search_start):
                env.step(_canonical_action(actions[step], step).tolist())
            crossing_step = None
            searched = []
            state_snapshots = {}
            for step in range(search_start, collision_step + 1):
                state_snapshots[step] = _snapshot_env(env)
                nominal = _canonical_action(actions[step], step)
                transition = probe.transition(env, nominal)
                start_gap = np.asarray(transition["substeps"][0]["clearance_m"][:7], dtype=np.float64)
                minimum = np.asarray(transition["minimum_substep_clearance_m"][:7], dtype=np.float64)
                crossing = bool(np.all(start_gap >= 0.0) and np.any(minimum < 0.0))
                searched.append({"step": step, "minimum_start_m": float(np.min(start_gap)), "minimum_interval_m": float(np.min(minimum)), "crossing": crossing})
                if crossing:
                    crossing_step = step
                    break
                env.step(nominal.tolist())
            if crossing_step is None:
                episode_results.append({
                    "case_id": case_id, "split": selected_row["split"],
                    "task_level_group_id": selected_row["task_level_group_id"],
                    "eligible": False, "reason": "no_exact_recoverable_crossing_in_registered_window",
                    "search": searched,
                })
                continue
            state_offsets = config["state"].get(
                "candidate_state_offsets_from_first_crossing", [0]
            )
            selected_step = None
            selected_transition = None
            nominal = None
            grid_records = []
            anchors = []
            boundary_state_search = []
            last_error = "no_registered_boundary_state"
            for state_offset in state_offsets:
                candidate_step = crossing_step + int(state_offset)
                if candidate_step < search_start or candidate_step not in state_snapshots:
                    continue
                _restore_env(env, state_snapshots[candidate_step])
                candidate_nominal = _canonical_action(
                    actions[candidate_step], candidate_step
                )
                candidate_transition = probe.transition(env, candidate_nominal)
                lower, upper, candidates = grid_actions(
                    candidate_nominal[:3], config
                )
                candidate_records = []
                for grid_index, xyz in enumerate(candidates):
                    action = candidate_nominal.copy(); action[:3] = xyz
                    candidate_records.append(_transition_record(
                        record_index=len(all_records) + len(candidate_records),
                        case_row=selected_row, source="grid",
                        group_id="%s::grid_%04d" % (case_id, grid_index),
                        grid_index=grid_index, anchor_grid_index=None,
                        probe_dimension=None, probe_sign=None,
                        nominal=candidate_nominal, candidate_xyz=xyz,
                        transition=probe.transition(env, action), band_m=band,
                    ))
                counts = {
                    name: sum(r["category"] == name for r in candidate_records)
                    for name in (
                        "boundary_safe", "boundary_unsafe",
                        "far_safe", "far_unsafe",
                    )
                }
                try:
                    candidate_anchors = select_balanced_anchors(
                        candidate_records, lower, upper, config
                    )
                    candidate_error = None
                except ValueError as error:
                    candidate_anchors = []
                    candidate_error = str(error)
                    last_error = candidate_error
                boundary_state_search.append({
                    "state_offset_from_first_crossing": int(state_offset),
                    "state_step": candidate_step,
                    "grid_category_counts": counts,
                    "balanced_anchor_gate_pass": bool(candidate_anchors),
                    "reason": candidate_error,
                })
                if candidate_anchors:
                    selected_step = candidate_step
                    selected_transition = candidate_transition
                    nominal = candidate_nominal
                    grid_records = candidate_records
                    anchors = candidate_anchors
                    break
                grid_records = candidate_records
            if selected_step is None:
                episode_results.append({
                    "case_id": case_id, "split": selected_row["split"],
                    "task_level_group_id": selected_row["task_level_group_id"],
                    "eligible": False, "reason": last_error,
                    "first_crossing_step": crossing_step,
                    "search": searched,
                    "boundary_state_search": boundary_state_search,
                    "grid_category_counts": {name: sum(r["category"] == name for r in grid_records) for name in ("boundary_safe", "boundary_unsafe", "far_safe", "far_unsafe")},
                })
                all_records.extend(grid_records)
                continue
            _restore_env(env, state_snapshots[selected_step])
            grid_by_index = {int(item["grid_index"]): item for item in grid_records}
            probes: dict[tuple[int, int, int], dict[str, Any]] = {}
            episode_records = list(grid_records)
            for anchor_index in anchors:
                anchor = grid_by_index[anchor_index]
                group_id = "%s::anchor_%04d" % (case_id, anchor_index)
                anchor["group_id"] = group_id
                anchor["split"] = str(selected_row["split"])
                base = np.asarray(anchor["candidate_xyz"], dtype=np.float64)
                for dimension in range(3):
                    for sign in (-1, 1):
                        xyz = base.copy(); xyz[dimension] += sign * epsilon
                        action = nominal.copy(); action[:3] = xyz
                        item = _transition_record(
                            record_index=len(all_records) + len(episode_records), case_row=selected_row,
                            source="gradient_probe", group_id=group_id, grid_index=None,
                            anchor_grid_index=anchor_index, probe_dimension=dimension,
                            probe_sign=sign, nominal=nominal, candidate_xyz=xyz,
                            transition=probe.transition(env, action), band_m=band,
                        )
                        item["category"] = anchor["category"]
                        item["split"] = str(selected_row["split"])
                        episode_records.append(item)
                        probes[(anchor_index, dimension, sign)] = item
            _require(len(probes) == int(settings["expected_gradient_probe_count_per_episode"]), "gradient probe count differs")
            stable_active = 0
            for anchor_index in anchors:
                anchor = grid_by_index[anchor_index]
                gradients = np.zeros((7, 3), dtype=np.float64)
                valid = np.ones(7, dtype=bool)
                for dimension in range(3):
                    minus = probes[(anchor_index, dimension, -1)]
                    plus = probes[(anchor_index, dimension, 1)]
                    gradients[:, dimension] = (
                        np.asarray(plus["minimum_substep_clearance_m"]) - np.asarray(minus["minimum_substep_clearance_m"])
                    ) / (2.0 * epsilon)
                    for row in range(7):
                        valid[row] = bool(valid[row] and anchor["minimum_substep_witnesses"][row] == minus["minimum_substep_witnesses"][row] == plus["minimum_substep_witnesses"][row])
                anchor["gradient_m_per_action"] = gradients.tolist()
                anchor["gradient_valid_rows"] = valid.tolist()
                stable_active += int(valid[int(anchor["active_constraint_index"])])
            offset = len(all_records)
            for index, record in enumerate(episode_records):
                record["record_index"] = offset + index
            all_records.extend(episode_records)
            counts = {name: sum(r["category"] == name for r in grid_records) for name in ("boundary_safe", "boundary_unsafe", "far_safe", "far_unsafe")}
            episode_results.append({
                "case_id": case_id, "split": selected_row["split"],
                "task_level_group_id": selected_row["task_level_group_id"],
                "eligible": bool(stable_active > 0),
                "reason": None if stable_active > 0 else "no_stable_active_gradient_anchor",
                "first_crossing_step": crossing_step,
                "selected_step": selected_step,
                "nominal_action": nominal.tolist(),
                "nominal_minimum_substep_clearance_m": list(selected_transition["minimum_substep_clearance_m"][:7]),
                "search": searched,
                "boundary_state_search": boundary_state_search,
                "grid_category_counts": counts,
                "gradient_anchor_count": len(anchors),
                "stable_active_gradient_anchor_count": stable_active,
                "record_count": len(episode_records),
            })
        finally:
            if probe_env is not None: probe_env.close()
            if env is not None: env.close()

    dataset_gate = bool(len(episode_results) == len(selected) and all(item["eligible"] for item in episode_results))
    split_case_counts = {name: sum(item["split"] == name for item in episode_results) for name in ("train", "validation", "test")}
    dataset = {
        "schema_version": GENERALIZATION_DATASET_SCHEMA,
        "source_commit": source["commit"],
        "config_file_sha256": config["config_file_sha256"],
        "config_payload_sha256": config["config_payload_sha256"],
        "selected_manifest_file_sha256": _file_sha256(selected_manifest_path),
        "constraint_order": list(CONSTRAINT_ORDER),
        "episode_results": episode_results,
        "summary": {
            "selected_case_count": len(selected), "eligible_case_count": sum(item["eligible"] for item in episode_results),
            "split_case_counts": split_case_counts, "record_count": len(all_records),
            "learning_record_count": sum(item["split"] in {"train", "validation", "test"} for item in all_records),
            "dataset_gate_pass": dataset_gate,
        },
        "records": all_records,
    }
    dataset["dataset_payload_sha256"] = _hash_without(dataset, "dataset_payload_sha256")
    _atomic_write(dataset_path, dataset)
    result = {
        "schema_version": GENERALIZATION_DATASET_RESULT_SCHEMA,
        "status": "complete", "scientific_result": True,
        "claim_scope": config["claim_scope"], "source": source, "allocation": allocation,
        "config": config, "selected_manifest": {"path": str(selected_manifest_path), "file_sha256": _file_sha256(selected_manifest_path)},
        "archived_table1_root": {"path": str(archived_root), "read_only": True},
        "geometry_placeholder": {
            "path": str(geometry_placeholder_path),
            "file_sha256": _file_sha256(geometry_placeholder_path),
            "use": (
                "proper_rotation_constructor_placeholder_only; all_recorded_"
                "clearances_use_live_exact_15_box_union"
            ),
        },
        "pairings": all_pairings,
        "dataset": {"path": str(dataset_path), "file_sha256": _file_sha256(dataset_path), "payload_sha256": dataset["dataset_payload_sha256"]},
        "dataset_summary": dataset["summary"],
        "decision": {"dataset_gate_pass": dataset_gate, "neural_training_authorized": dataset_gate, "stop_reason": None if dataset_gate else "one_or_more_episode_boundary_gates_failed"},
        "wall_seconds": (time.perf_counter_ns() - started) * 1e-9,
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
        repo_root=args.repo_root.resolve(), population_manifest_path=args.population_manifest.resolve(),
        selected_manifest_path=args.selected_manifest.resolve(), archived_root=args.archived_root.resolve(),
        geometry_config_path=args.geometry_config.resolve(), exact_box_config_path=args.exact_box_config.resolve(),
        config_path=args.config.resolve(), expected_commit=args.expected_commit, dataset_path=args.dataset.resolve(),
    )
    _atomic_write(args.output.resolve(), result)
    print(json.dumps(result["decision"], sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
