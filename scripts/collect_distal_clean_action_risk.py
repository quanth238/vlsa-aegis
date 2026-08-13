#!/usr/bin/env python3
"""Collect one clean episode's exact candidate-plus-pure-backup risks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.evaluate_distal_smooth_field_attribution_e05 import (
    InstrumentedContinuationProbe,
    _disable_images,
)
from scripts.evaluate_distal_counterfactual_field_e05 import FixedContinuationProbe
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _git_identity, _load, _require, _sha256,
)


def _public(value: Any) -> Any:
    try:
        import numpy as np
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
    except ImportError:
        pass
    if isinstance(value, Mapping):
        return {str(key): _public(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_public(item) for item in value]
    return value


def _context(env: Any, observation: Mapping[str, Any], geometry: Any, current: Any) -> dict[str, Any]:
    import numpy as np

    from main.multilink_ellipsoid.rollout import _controller_snapshot, _dynamic_state_vector

    robot = env.robots[0]
    qpos_indexes = getattr(robot, "_ref_joint_pos_indexes", None)
    qvel_indexes = getattr(robot, "_ref_joint_vel_indexes", None)
    _require(qpos_indexes is not None and len(qpos_indexes) == 7, "arm qpos indexes differ")
    _require(qvel_indexes is not None and len(qvel_indexes) == 7, "arm qvel indexes differ")
    links = geometry._slabbed_links(env)
    rows = []
    for clearance, link in zip(np.asarray(current), links):
        rows.append({
            "body_name": str(link.body_name),
            "center_m": np.asarray(link.center).tolist(),
            "rotation": np.asarray(link.rotation).tolist(),
            "semiaxes_m": np.asarray(link.semiaxes_m).tolist(),
            "current_clearance_m": float(clearance),
        })
    return {
        "dynamic_state": np.asarray(_dynamic_state_vector(env), dtype=np.float64).tolist(),
        "arm_joint_position_rad": np.asarray(env.sim.data.qpos[list(qpos_indexes)], dtype=np.float64).tolist(),
        "arm_joint_velocity_rad_s": np.asarray(env.sim.data.qvel[list(qvel_indexes)], dtype=np.float64).tolist(),
        "eef_position_m": np.asarray(observation["robot0_eef_pos"], dtype=np.float64).tolist(),
        "eef_quaternion_xyzw": np.asarray(observation["robot0_eef_quat"], dtype=np.float64).tolist(),
        "controller_snapshot": _public(_controller_snapshot(env)[0]),
        "obstacle": {
            "center_m": np.asarray(geometry.obstacle.center).tolist(),
            "rotation": np.asarray(geometry.obstacle.rotation).tolist(),
            "semiaxes_m": np.asarray(geometry.obstacle.semiaxes_m).tolist(),
        },
        "geometry_rows": rows,
    }


def collect(
    *, repo_root: Path, population_manifest_path: Path, selection_manifest_path: Path,
    experiment_config_path: Path, geometry_config_path: Path,
    table1_root: Path, case_index: int, expected_commit: str,
) -> dict[str, Any]:
    import numpy as np

    from main.evaluate_safelibero_aegis import (
        TABLE_SETTLE_ACTIONS, _active_obstacle, _build_environment,
        _runtime_imports, _settle, read_jsonl, validate_case_row,
    )
    from main.multilink_ellipsoid.clean_action_risk import (
        RESULT_SCHEMA, canonical, compact_feature_vector, decision_steps, exact_safe, load_cases,
        load_config, risk_from_row_minimum,
    )
    from main.multilink_ellipsoid.pure_backup import (
        registered_directions, select_backup_with_fallback,
    )
    from main.multilink_ellipsoid.rollout import _dynamic_state_vector
    from main.multilink_ellipsoid.shadow import (
        MultilinkEllipsoidShadow, allocation_record, load_shadow_config,
    )
    from main.multilink_ellipsoid.sitl_candidate import (
        SlabbedEightConstraintProbe, _protected_contact_evidence,
    )

    config = load_config(experiment_config_path)
    cases = load_cases(selection_manifest_path, config)
    _require(0 <= int(case_index) < len(cases), "clean action-risk case index differs")
    selected = cases[int(case_index)]
    source_path = table1_root / selected["archived_result_relative_path"]
    _require(_file_sha256(source_path) == selected["archived_result_file_sha256"], "Table-1 source file differs")
    archived = _load(source_path)
    _require(archived["result_payload_sha256"] == selected["archived_result_payload_sha256"], "Table-1 source payload differs")
    _require(archived["task_success"] is True, "Table-1 source task did not succeed")
    _require(archived["metrics"]["paper_collision"] is True, "Table-1 source CAR did not fail")
    all_rows = [row for row in read_jsonl(population_manifest_path) if row.get("case_id") == selected["case_id"]]
    _require(len(all_rows) == 1, "Table-1 population case differs")
    case = all_rows[0]
    validate_case_row(case, repo_root)
    source = _git_identity(repo_root, expected_commit)
    runtime = _runtime_imports(include_aegis=True)
    geometry_config = load_shadow_config(geometry_config_path)
    actions = {
        int(item["step"]): np.asarray(item["executed"], dtype=np.float64)
        for item in archived["actions"]
    }
    _require(set(actions) == set(range(len(archived["actions"]))), "Table-1 action ledger differs")
    targets = decision_steps(selected, config)
    _require(max(targets) < len(actions), "clean action-risk state exceeds action ledger")
    env = probe_env = backup_env = None
    try:
        env, task, observation, _ = _build_environment(runtime, case, render_resolution=32)
        observation = _settle(env, observation, TABLE_SETTLE_ACTIONS)
        probe_env, probe_task, probe_observation, _ = _build_environment(runtime, case, render_resolution=32)
        probe_observation = _settle(probe_env, probe_observation, TABLE_SETTLE_ACTIONS)
        backup_env, backup_task, backup_observation, _ = _build_environment(runtime, case, render_resolution=32)
        backup_observation = _settle(backup_env, backup_observation, TABLE_SETTLE_ACTIONS)
        _require(str(task.language) == str(probe_task.language), "probe task differs")
        _require(str(task.language) == str(backup_task.language), "backup probe task differs")
        obstacle_name, _ = _active_obstacle(env, observation)
        _require(obstacle_name == selected["active_obstacle_name"], "active obstacle differs")
        obstacle_reference = np.asarray(observation["%s_pos" % obstacle_name], dtype=np.float64).copy()
        perception = archived["perception"]
        geometry = MultilinkEllipsoidShadow.from_aegis_geometry(
            geometry_config,
            {
                "p2": perception["mvee_center"], "R2": perception["mvee_rotation"],
                "Q2_diag": perception["mvee_semiaxes"],
                "record": {"label": perception["obstacle_label"]},
            },
        )
        one_step = SlabbedEightConstraintProbe(
            probe_env, geometry, clearance_m=0.0, active_obstacle_name=obstacle_name
        )
        _disable_images(probe_env)
        instrumented = InstrumentedContinuationProbe(
            one_step, obstacle_name, obstacle_reference
        )
        backup_one_step = SlabbedEightConstraintProbe(
            backup_env, geometry, clearance_m=0.0, active_obstacle_name=obstacle_name
        )
        _disable_images(backup_env)
        backup_instrumented = InstrumentedContinuationProbe(
            backup_one_step, obstacle_name, obstacle_reference
        )
        buffer_m = float(config["risk_target"]["safety_buffer_m"])
        terminal = int(config["candidate_family"]["terminal_hold_actions"])
        expected_substeps = int(config["state_sampling"]["expected_mujoco_substeps_per_action"])
        tolerance = float(config["state_sampling"]["boundary_equivalence_tolerance"])
        state_records = []
        rejected_states = []

        def rollout_summary(source_env: Any, command: Any, step: int) -> dict[str, Any]:
            commands = np.zeros((1 + terminal, 7), dtype=np.float64)
            commands[0] = np.asarray(command, dtype=np.float64)
            rollout = backup_instrumented.rollout_internal(
                source_env, commands, expected_substeps=expected_substeps,
                boundary_tolerance=tolerance, step_base=step + 1,
            )
            trace = np.asarray(rollout["clearance_trace_m"], dtype=np.float64)[:, :7]
            row_min = np.min(trace, axis=0)
            return {
                "row_minimum_clearance_m": row_min.tolist(),
                "minimum_clearance_m": float(np.min(row_min)),
                "future_minimum_clearance_m": float(np.min(trace[1:])),
                "protected_contact_count": len(rollout["protected_contacts"]),
                "maximum_active_obstacle_l1_displacement_m": float(rollout["maximum_active_obstacle_l1_displacement_m"]),
                "sample_count": int(trace.shape[0]),
                "maximum_boundary_equivalence_error_m": float(rollout["maximum_boundary_equivalence_error_m"]),
            }

        def evaluate_candidate(name: str, order: int, action: Any, step: int) -> dict[str, Any]:
            proposal = np.asarray(action, dtype=np.float64)
            one_step.synchronize(env)
            probe_env.step(proposal.tolist())
            successor = np.asarray(backup_one_step.clearances(probe_env)[:7], dtype=np.float64)
            links = geometry._slabbed_links(probe_env)
            active_row = int(np.argmin(successor))
            normal = np.asarray(links[active_row].center) - np.asarray(geometry.obstacle.center)
            normal /= float(np.linalg.norm(normal))
            backup_definitions: list[tuple[str, Any]] = [("hold", np.zeros(7, dtype=np.float64))]
            for direction_name, direction in registered_directions(normal):
                for amplitude in config["candidate_family"]["amplitudes_action"]:
                    command = np.zeros(7, dtype=np.float64)
                    command[:3] = np.asarray(direction, dtype=np.float64) * float(amplitude)
                    backup_definitions.append(("%s_amp_%s" % (direction_name, amplitude), command))
            _require(
                len(backup_definitions) == int(config["candidate_family"]["backup_candidate_count"]),
                "backup candidate count differs",
            )
            backup_candidates = []
            for backup_order, (backup_name, backup_action) in enumerate(backup_definitions):
                backup_candidates.append({
                    "name": backup_name, "order": backup_order,
                    "first_action": np.asarray(backup_action, dtype=np.float64).tolist(),
                    "record": rollout_summary(probe_env, backup_action, step),
                })
            selected, selection_mode = select_backup_with_fallback(
                backup_candidates, safety_buffer_m=buffer_m,
                paper_car_threshold_m=float(config["risk_target"]["paper_car_threshold_m"]),
            )
            full_commands = np.zeros((2 + terminal, 7), dtype=np.float64)
            full_commands[0] = proposal
            full_commands[1] = np.asarray(selected["first_action"], dtype=np.float64)
            rollout = instrumented.rollout_internal(
                env, full_commands, expected_substeps=expected_substeps,
                boundary_tolerance=tolerance, step_base=step,
            )
            trace = np.asarray(rollout["clearance_trace_m"], dtype=np.float64)[:, :7]
            row_min = np.min(trace, axis=0)
            successor_index = expected_substeps
            successor_replay_error = float(np.max(np.abs(
                trace[successor_index] - successor
            )))
            selected_row_min = np.asarray(
                selected["record"]["row_minimum_clearance_m"], dtype=np.float64
            )
            selected_backup_replay_error = float(np.max(np.abs(
                np.min(trace[successor_index:], axis=0) - selected_row_min
            )))
            selected_contact_count = sum(
                int(item["action_offset"]) >= 1
                or (
                    int(item["action_offset"]) == 0
                    and int(item["substep"]) == expected_substeps - 1
                )
                for item in rollout["protected_contacts"]
            )
            selected_contact_count_error = abs(
                selected_contact_count
                - int(selected["record"]["protected_contact_count"])
            )
            displacement = np.asarray(
                rollout["active_obstacle_l1_displacement_trace_m"], dtype=np.float64
            )
            selected_displacement_error = abs(
                float(np.max(displacement[successor_index:]))
                - float(selected["record"]["maximum_active_obstacle_l1_displacement_m"])
            )
            _require(successor_replay_error <= tolerance, "proposal successor replay differs")
            _require(selected_backup_replay_error <= tolerance, "selected backup replay differs")
            _require(selected_contact_count_error == 0, "selected backup contact replay differs")
            _require(selected_displacement_error <= tolerance, "selected backup CAR replay differs")
            record = {
                "name": name, "order": int(order), "first_action": proposal.tolist(),
                "first_action_sha256": hashlib.sha256(proposal.tobytes()).hexdigest(),
                "successor_clearance_m": successor.tolist(),
                "backup_state_derived_normal": normal.tolist(),
                "backup_candidate_count": len(backup_candidates),
                "backup_safe_candidate_count": sum(
                    item["record"]["minimum_clearance_m"] >= buffer_m
                    and item["record"]["protected_contact_count"] == 0
                    and item["record"]["maximum_active_obstacle_l1_displacement_m"]
                    <= float(config["risk_target"]["paper_car_threshold_m"])
                    for item in backup_candidates
                ),
                "selected_backup": {
                    "name": selected["name"], "order": selected["order"],
                    "first_action": selected["first_action"],
                    "selection_mode": selection_mode,
                },
                "proposal_successor_replay_error_m": successor_replay_error,
                "selected_backup_row_replay_error_m": selected_backup_replay_error,
                "selected_backup_contact_count_replay_error": selected_contact_count_error,
                "selected_backup_car_replay_error_m": selected_displacement_error,
                "backup_candidates": backup_candidates,
                "row_minimum_clearance_m": row_min.tolist(),
                "risk": risk_from_row_minimum(row_min, buffer_m),
                "minimum_clearance_m": float(np.min(row_min)),
                "future_minimum_clearance_m": float(np.min(trace[1:])),
                "protected_contact_count": len(rollout["protected_contacts"]),
                "protected_contacts": rollout["protected_contacts"],
                "maximum_active_obstacle_l1_displacement_m": float(rollout["maximum_active_obstacle_l1_displacement_m"]),
                "sample_count": int(trace.shape[0]),
                "maximum_boundary_equivalence_error_m": float(rollout["maximum_boundary_equivalence_error_m"]),
            }
            record["exact_safe"] = exact_safe(record, config)
            return record

        for step in range(max(targets) + 1):
            if step in targets:
                current = np.asarray(one_step.clearances(env)[:7], dtype=np.float64)
                contacts = _protected_contact_evidence(env, obstacle_name)["events"]
                current_obstacle_displacement = float(np.sum(np.abs(
                    np.asarray(observation["%s_pos" % obstacle_name], dtype=np.float64)
                    - obstacle_reference
                )))
                state_id = "%s-step-%03d" % (selected["case_id"], step)
                if (
                    float(np.min(current)) < float(config["state_sampling"]["minimum_initial_proxy_clearance_m"])
                    or contacts
                    or current_obstacle_displacement
                    > float(config["state_sampling"]["maximum_initial_active_obstacle_l1_displacement_m"])
                ):
                    rejected_states.append({
                        "state_id": state_id, "step": step,
                        "initial_clearance_m": current.tolist(),
                        "initial_protected_contacts": contacts,
                        "initial_active_obstacle_l1_displacement_m": current_obstacle_displacement,
                        "reason": "initial_state_not_strictly_safe",
                    })
                else:
                    source_state_before = np.asarray(_dynamic_state_vector(env), dtype=np.float64).copy()
                    links = geometry._slabbed_links(env)
                    active_row = int(np.argmin(current))
                    normal = np.asarray(links[active_row].center) - np.asarray(geometry.obstacle.center)
                    normal /= float(np.linalg.norm(normal))
                    definitions: list[tuple[str, Any]] = [("nominal_aegis", actions[step])]
                    definitions.append(("hold", np.zeros(7, dtype=np.float64)))
                    for direction_name, direction in registered_directions(normal):
                        for amplitude in config["candidate_family"]["amplitudes_action"]:
                            command = np.zeros(7, dtype=np.float64)
                            command[:3] = np.asarray(direction, dtype=np.float64) * float(amplitude)
                            definitions.append(("%s_amp_%s" % (direction_name, amplitude), command))
                    _require(len(definitions) == int(config["candidate_family"]["proposal_count"]), "proposal count differs")
                    candidates = [
                        evaluate_candidate(name, order, command, step)
                        for order, (name, command) in enumerate(definitions)
                    ]
                    source_state_after = np.asarray(_dynamic_state_vector(env), dtype=np.float64)
                    source_state_mutation = float(np.max(np.abs(source_state_after - source_state_before)))
                    context = _context(env, observation, geometry, current)
                    for candidate in candidates:
                        candidate["feature_vector"] = compact_feature_vector(
                            context, candidate["first_action"]
                        )
                    safe_count = sum(bool(item["exact_safe"]) for item in candidates)
                    record = {
                        "state_id": state_id, "step": step,
                        "lead_actions_before_contact": int(selected["first_relevant_contact_step"]) - step,
                        "initial_clearance_m": current.tolist(), "active_row": active_row,
                        "initial_active_obstacle_l1_displacement_m": current_obstacle_displacement,
                        "state_derived_normal": normal.tolist(), "context": context,
                        "candidate_count": len(candidates), "exact_safe_candidate_count": safe_count,
                        "source_state_maximum_mutation": source_state_mutation,
                        "candidates": candidates,
                    }
                    if safe_count == 0:
                        rejected_states.append({
                            "state_id": state_id, "step": step,
                            "initial_clearance_m": current.tolist(),
                            "initial_protected_contacts": [],
                            "reason": "no_exact_safe_candidate_plus_backup_support",
                        })
                    state_records.append(record)
            if step < max(targets):
                observation, _, _, _ = env.step(actions[step].tolist())
        accepted = [item for item in state_records if item["exact_safe_candidate_count"] > 0]
        gates = {
            "source_task_success_and_car_failure": True,
            "all_requested_states_strictly_initially_safe": len(rejected_states) == 0,
            "all_requested_states_have_exact_safe_support": len(accepted) == len(targets),
            "candidate_count_and_order_exact": all(item["candidate_count"] == 26 for item in state_records),
            "all_candidate_labels_finite": all(
                np.all(np.isfinite(candidate["risk"]))
                for state in state_records for candidate in state["candidates"]
            ),
            "source_states_unmodified_by_cloned_rollouts": all(
                state["source_state_maximum_mutation"] == 0.0 for state in state_records
            ),
            "proposal_and_selected_backup_replay_consistent": all(
                candidate["proposal_successor_replay_error_m"] <= tolerance
                and candidate["selected_backup_row_replay_error_m"] <= tolerance
                and candidate["selected_backup_contact_count_replay_error"] == 0
                and candidate["selected_backup_car_replay_error_m"] <= tolerance
                for state in state_records for candidate in state["candidates"]
            ),
        }
        result = {
            "schema_version": RESULT_SCHEMA, "status": "complete",
            "scientific_result": True, "case_index": int(case_index),
            "case": selected, "source": source, "allocation": allocation_record(),
            "config": config,
            "source_result": {
                "path": str(source_path), "file_sha256": _file_sha256(source_path),
                "result_payload_sha256": archived["result_payload_sha256"],
                "native_task_success": archived["task_success"],
                "paper_car_failure": archived["metrics"]["paper_collision"],
            },
            "requested_state_count": len(targets), "accepted_state_count": len(accepted),
            "states": state_records, "rejected_states": rejected_states,
            "gates": gates,
            "interpretation": (
                "clean_exact_action_risk_case_pass" if all(gates.values())
                else "clean_exact_action_risk_case_no_go"
            ),
            "training_authorized_for_case": all(gates.values()),
        }
        result["result_payload_sha256"] = _sha256(canonical(result))
        return result
    finally:
        if backup_env is not None:
            backup_env.close()
        if probe_env is not None:
            probe_env.close()
        if env is not None:
            env.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--population-manifest", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--geometry-config", type=Path, required=True)
    parser.add_argument("--table1-root", type=Path, required=True)
    parser.add_argument("--case-index", type=int, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = collect(
        repo_root=args.repo_root.resolve(), population_manifest_path=args.population_manifest.resolve(),
        selection_manifest_path=args.selection_manifest.resolve(),
        experiment_config_path=args.experiment_config.resolve(),
        geometry_config_path=args.geometry_config.resolve(), table1_root=args.table1_root.resolve(),
        case_index=args.case_index, expected_commit=args.expected_commit,
    )
    _atomic_write(args.output.resolve(), result)
    print(json.dumps({
        "case_id": result["case"]["case_id"], "interpretation": result["interpretation"],
        "gates": result["gates"], "result_payload_sha256": result["result_payload_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
