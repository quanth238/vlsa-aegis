#!/usr/bin/env python3
"""Collect multi-state conservative affine-row targets using cloned OSC."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping

from main.multilink_ellipsoid.affine_coefficient_model import (
    AFFINE_COEFFICIENT_DATASET_RESULT_SCHEMA,
    AFFINE_COEFFICIENT_DATASET_SCHEMA,
    coefficient_grid_actions,
    coefficient_targets,
    load_affine_coefficient_config,
)
from main.multilink_ellipsoid.execution_margin_nn import CONSTRAINT_ORDER
from main.multilink_ellipsoid.oracle_affine_safe_set import (
    fit_candidate_conditioned_affine_certificate,
    solve_affine_certificate_qp,
)
from main.multilink_ellipsoid.two_step_margin import (
    feature_context,
    feature_vectors,
    load_selected_manifest,
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
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")).hexdigest()


def _chunk_values(chunk: Mapping[str, Any]) -> tuple[dict[str, Any], float]:
    import numpy as np

    summary = summarize_chunk(chunk)
    ee = min(
        float(np.asarray(item["minimum_substep_clearance_m"], dtype=np.float64)[7])
        for item in chunk["transitions"]
    )
    return summary, ee


def collect(
    *, repo_root: Path, population_manifest_path: Path,
    selected_manifest_path: Path, archived_root: Path,
    geometry_config_path: Path, exact_box_config_path: Path,
    config_path: Path, expected_commit: str, dataset_path: Path,
) -> dict[str, Any]:
    import numpy as np
    from main.evaluate_safelibero_aegis import (
        _runtime_imports, pairing_record, read_jsonl, validate_case_row,
    )
    from main.multilink_ellipsoid.obstacle_primitives import (
        load_obstacle_primitive_config,
    )
    from main.multilink_ellipsoid.oracle_affine import SubstepEightConstraintProbe
    from main.multilink_ellipsoid.shadow import allocation_record, load_shadow_config

    started = time.perf_counter_ns()
    config = load_affine_coefficient_config(config_path)
    selected = load_selected_manifest(selected_manifest_path, config)
    for path, expected, label in (
        (population_manifest_path, config["source_population_manifest_sha256"], "population"),
        (selected_manifest_path, config["selected_manifest_sha256"], "selected"),
        (geometry_config_path, config["geometry_config_file_sha256"], "geometry"),
        (exact_box_config_path, config["exact_box_config_file_sha256"], "exact-box"),
    ):
        _require(_file_sha256(path) == expected, "affine-coefficient %s differs" % label)
    geometry_config = load_shadow_config(geometry_config_path)
    exact_box_config = load_obstacle_primitive_config(exact_box_config_path)
    population = {item["case_id"]: item for item in read_jsonl(population_manifest_path)}
    source = _git_identity(repo_root, expected_commit)
    runtime = _runtime_imports(include_aegis=False)
    primary = next(item for item in selected if item["case_id"].endswith("e05"))
    placeholder_path = archived_root / primary["archived_relative_path"]
    placeholder = _load(placeholder_path)
    _require(
        _file_sha256(placeholder_path) == primary["archived_file_sha256"]
        and placeholder.get("result_payload_sha256") == primary["archived_payload_sha256"],
        "affine-coefficient geometry placeholder differs",
    )
    target_settings = config["coefficient_target"]
    offsets = list(config["state_sampling"]["state_offsets_from_crossing"])
    maximum_motion = 1.0e-4
    state_records = []
    episode_results = []
    pairings = {}

    for selected_row in selected:
        case_id = str(selected_row["case_id"])
        archived_path = archived_root / selected_row["archived_relative_path"]
        archived = _load(archived_path)
        _require(
            archived_path.is_file() and not archived_path.is_symlink()
            and _file_sha256(archived_path) == selected_row["archived_file_sha256"]
            and archived.get("result_payload_sha256") == selected_row["archived_payload_sha256"]
            and archived.get("case_id") == case_id,
            "affine-coefficient archived case differs: %s" % case_id,
        )
        case = population[case_id]
        validate_case_row(case, repo_root)
        env = probe_env = None
        case_states = []
        try:
            env, probe_env, task, observation, setup = _build_pair(runtime, case)
            pairing = pairing_record(
                case=case, selected_initial_state=setup["selected_initial_state"],
                settled_observation=observation, task_description=str(task.language),
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
                    "affine-coefficient pairing differs for %s: %s" % (case_id, key),
                )
            pairings[case_id] = pairing
            geometry, exact_boxes = _geometry(
                geometry_config=geometry_config, exact_box_config=exact_box_config,
                archived=placeholder, env=env, obstacle_name=setup["obstacle_name"],
            )
            probe = SubstepEightConstraintProbe(
                probe_env, geometry, active_obstacle_name=setup["obstacle_name"],
                quadratic_tolerance=1.0e-6, contact_distance_threshold_m=0.0,
                obstacle_primitive_union=exact_boxes,
            )
            actions = archived["actions"]
            collision_step = int(selected_row["collision_first_step"])
            search_start = max(
                0, collision_step
                - int(config["state_sampling"]["search_start_offset_actions"])
            )
            for step in range(search_start):
                env.step(_canonical_action(actions[step], step).tolist())
            crossing_step = None
            snapshots = {}
            search = []
            for step in range(search_start, min(collision_step + 1, len(actions) - 1)):
                snapshots[step] = _snapshot_env(env)
                first = _canonical_action(actions[step], step)
                second = _canonical_action(actions[step + 1], step + 1)
                nominal_summary, _ = _chunk_values(probe.rollout_chunk(env, [first, second]))
                current = np.asarray(
                    feature_context(env, probe)["current_clearance_m"], dtype=np.float64
                )
                minimum = np.asarray(
                    nominal_summary["minimum_substep_clearance_m"], dtype=np.float64
                )
                crossing = bool(np.all(current >= 0.0) and np.any(minimum < 0.0))
                search.append({
                    "step": int(step), "minimum_start_m": float(np.min(current)),
                    "minimum_two_step_m": float(np.min(minimum)), "crossing": crossing,
                })
                if crossing:
                    crossing_step = step
                    break
                env.step(first.tolist())
            if crossing_step is None:
                episode_results.append({
                    "case_id": case_id, "split": selected_row["split"],
                    "task_level_group_id": selected_row["task_level_group_id"],
                    "gate_pass": False, "reason": "no_two_step_crossing", "search": search,
                })
                continue

            for offset in offsets:
                step = int(crossing_step + offset)
                if step not in snapshots or step + 1 >= len(actions):
                    case_states.append({
                        "case_id": case_id, "state_step": step,
                        "state_offset_from_crossing": int(offset), "gate_pass": False,
                        "reason": "registered_state_snapshot_missing",
                    })
                    continue
                _restore_env(env, snapshots[step])
                first = _canonical_action(actions[step], step)
                second = _canonical_action(actions[step + 1], step + 1)
                context = feature_context(env, probe)
                _, pair_features = feature_vectors(
                    context, first[:3], first[:3], second[:3]
                )
                nominal_summary, nominal_ee = _chunk_values(
                    probe.rollout_chunk(env, [first, second])
                )
                nominal_margin = np.asarray(
                    nominal_summary["minimum_substep_clearance_m"], dtype=np.float64
                )
                lower, upper, candidates = coefficient_grid_actions(first[:3], config)
                margins = []
                raw_safe = []
                contact_counts = []
                obstacle_motion = []
                for xyz in candidates:
                    action = first.copy()
                    action[:3] = xyz
                    summary, _ = _chunk_values(probe.rollout_chunk(env, [action, second]))
                    margins.append(summary["minimum_substep_clearance_m"])
                    contact_counts.append(summary["raw_protected_contact_count"])
                    obstacle_motion.append(
                        summary["maximum_within_step_obstacle_l1_displacement_m"]
                    )
                    raw_safe.append(bool(
                        summary["D_sim_raw_safe"]
                        and summary["maximum_within_step_obstacle_l1_displacement_m"]
                        <= maximum_motion
                    ))
                certificate = fit_candidate_conditioned_affine_certificate(
                    candidates, margins, first[:3], raw_safe, list(range(len(candidates))),
                    one_sided_padding_m=float(target_settings["one_sided_padding_m"]),
                    target_clearance_m=float(target_settings["target_clearance_m"]),
                    postcheck_tolerance_m=float(
                        target_settings["coefficient_postcheck_tolerance_m"]
                    ),
                )
                qp = solve_affine_certificate_qp(
                    first[:3], lower, upper, certificate, config["projection"]
                )
                exact_qp = None
                target = None
                if certificate["valid"]:
                    target = coefficient_targets(nominal_margin, certificate)
                if qp["valid"]:
                    qp_action = first.copy()
                    qp_action[:3] = np.asarray(qp["candidate_xyz"], dtype=np.float64)
                    qp_summary, qp_ee = _chunk_values(
                        probe.rollout_chunk(env, [qp_action, second])
                    )
                    exact_qp = {
                        "minimum_distal_margin_m": qp_summary["minimum_substep_clearance_m"],
                        "minimum_released_AEGIS_EE_margin_m": float(qp_ee),
                        "D_opt_seven_distal_safe": bool(qp_summary["D_opt_proxy_safe"]),
                        "released_AEGIS_EE_proxy_safe": bool(qp_ee >= 0.0),
                        "D_sim_raw_safe": bool(qp_summary["D_sim_raw_safe"]),
                        "raw_protected_contact_count": int(
                            qp_summary["raw_protected_contact_count"]
                        ),
                        "maximum_within_step_obstacle_l1_displacement_m": float(
                            qp_summary["maximum_within_step_obstacle_l1_displacement_m"]
                        ),
                        "next_state_sha256": qp_summary["next_state_sha256"],
                    }
                gate = bool(
                    certificate["valid"] and qp["valid"]
                    and qp.get("diagnostics", {}).get("input_constraint_count") == 7
                    and exact_qp is not None
                    and exact_qp["D_opt_seven_distal_safe"]
                    and exact_qp["released_AEGIS_EE_proxy_safe"]
                    and exact_qp["D_sim_raw_safe"]
                    and exact_qp["maximum_within_step_obstacle_l1_displacement_m"]
                    <= maximum_motion
                )
                record = {
                    "state_index": len(state_records), "case_id": case_id,
                    "task_level_group_id": selected_row["task_level_group_id"],
                    "split": selected_row["split"], "state_step": step,
                    "state_offset_from_crossing": int(offset),
                    "crossing_step": int(crossing_step),
                    "nominal_first_action": first.tolist(),
                    "nominal_second_action": second.tolist(),
                    "action_lower": lower.tolist(), "action_upper": upper.tolist(),
                    "pair_state_feature_vectors": pair_features.tolist(),
                    "current_clearance_m": np.asarray(
                        context["current_clearance_m"], dtype=np.float64
                    ).tolist(),
                    "nominal_minimum_distal_margin_m": nominal_margin.tolist(),
                    "nominal_minimum_released_AEGIS_EE_margin_m": float(nominal_ee),
                    "candidate_first_xyz": np.asarray(candidates).tolist(),
                    "candidate_minimum_distal_margin_m": margins,
                    "candidate_raw_safe": raw_safe,
                    "candidate_raw_contact_count": contact_counts,
                    "candidate_maximum_obstacle_motion_m": obstacle_motion,
                    "certificate": certificate, "coefficient_target": target,
                    "oracle_qp": qp, "oracle_qp_exact_verification": exact_qp,
                    "gate_pass": gate,
                    "reason": None if gate else "affine_target_or_exact_oracle_gate_failed",
                }
                state_records.append(record)
                case_states.append({
                    "state_index": record["state_index"], "state_step": step,
                    "state_offset_from_crossing": int(offset), "gate_pass": gate,
                    "safe_candidate_count": certificate[
                        "exact_proxy_raw_safe_grid_candidate_count"
                    ],
                    "certificate_valid": certificate["valid"],
                    "qp_valid": qp["valid"], "reason": record["reason"],
                })
            episode_gate = bool(
                len(case_states) == int(
                    config["state_sampling"]["expected_states_per_episode"]
                ) and all(item["gate_pass"] for item in case_states)
            )
            episode_results.append({
                "case_id": case_id, "split": selected_row["split"],
                "task_level_group_id": selected_row["task_level_group_id"],
                "first_two_step_crossing_step": int(crossing_step),
                "registered_state_count": len(case_states),
                "gate_pass": episode_gate,
                "reason": None if episode_gate else "one_or_more_registered_states_failed",
                "search": search, "states": case_states,
            })
        finally:
            if probe_env is not None:
                probe_env.close()
            if env is not None:
                env.close()

    expected_state_count = len(selected) * int(
        config["state_sampling"]["expected_states_per_episode"]
    )
    dataset_gate = bool(
        len(episode_results) == len(selected)
        and len(state_records) == expected_state_count
        and all(item["gate_pass"] for item in episode_results)
        and all(item["gate_pass"] for item in state_records)
    )
    dataset = {
        "schema_version": AFFINE_COEFFICIENT_DATASET_SCHEMA,
        "source_commit": source["commit"],
        "config_file_sha256": config["config_file_sha256"],
        "config_payload_sha256": config["config_payload_sha256"],
        "selected_manifest_file_sha256": _file_sha256(selected_manifest_path),
        "constraint_order": list(CONSTRAINT_ORDER),
        "episode_results": episode_results,
        "state_records": state_records,
        "summary": {
            "episode_count": len(episode_results),
            "state_count": len(state_records),
            "expected_state_count": expected_state_count,
            "candidate_rollout_count": sum(
                len(item["candidate_first_xyz"]) for item in state_records
            ),
            "dataset_gate_pass": dataset_gate,
        },
    }
    dataset["dataset_payload_sha256"] = _hash_without(
        dataset, "dataset_payload_sha256"
    )
    _atomic_write(dataset_path, dataset)
    result = {
        "schema_version": AFFINE_COEFFICIENT_DATASET_RESULT_SCHEMA,
        "status": "complete", "scientific_result": True,
        "claim_scope": config["claim_scope"], "source": source,
        "allocation": allocation_record(), "config": config,
        "archived_table1_root": {"path": str(archived_root), "read_only": True},
        "pairings": pairings,
        "dataset": {
            "path": str(dataset_path), "file_sha256": _file_sha256(dataset_path),
            "payload_sha256": dataset["dataset_payload_sha256"],
        },
        "dataset_summary": dataset["summary"],
        "decision": {
            "dataset_gate_pass": dataset_gate,
            "neural_training_authorized": dataset_gate,
            "stop_reason": None if dataset_gate else "one_or_more_affine_target_states_failed",
        },
        "wall_seconds": (time.perf_counter_ns() - started) * 1.0e-9,
    }
    result["result_payload_sha256"] = _hash_without(result, "result_payload_sha256")
    return result


def main() -> int:
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
    args = parser.parse_args()
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
