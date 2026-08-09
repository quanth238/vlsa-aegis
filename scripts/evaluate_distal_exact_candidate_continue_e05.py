#!/usr/bin/env python3
"""Continue E05 nominal execution after the exact-candidate empty safe set."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping, Optional, Sequence

from main.multilink_ellipsoid.exact_candidate_closed_loop import (
    EXACT_CANDIDATE_RESULT_SCHEMA,
    EXACT_CANDIDATE_VALIDATION_SCHEMA,
)
from main.multilink_ellipsoid.exact_candidate_continue import (
    CONTINUE_RESULT_SCHEMA,
    load_continue_config,
)
from main.multilink_ellipsoid.two_step_margin import (
    load_selected_manifest,
    load_two_step_config,
)
from scripts.collect_distal_boundary_generalization_moka10 import _canonical_action
from scripts.evaluate_distal_affine_oracle_closed_loop_e05 import (
    _hash_without,
    _measured_env_step,
)
from scripts.evaluate_distal_execution_margin_nn_e05 import _build_pair, _geometry
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write,
    _file_sha256,
    _git_identity,
    _load,
    _require,
)


def _verify_prerequisite(
    config: Mapping[str, Any], result_path: Path, validation_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    prerequisite = config["prerequisite"]
    result = _load(result_path)
    validation = _load(validation_path)
    records = result.get("actions", [])
    final = records[-1] if isinstance(records, list) and records else {}
    _require(
        _file_sha256(result_path)
        == prerequisite["exact_candidate_result_file_sha256"]
        and result.get("result_payload_sha256")
        == prerequisite["exact_candidate_result_payload_sha256"]
        and _file_sha256(validation_path)
        == prerequisite["exact_candidate_validation_file_sha256"]
        and result.get("schema_version") == EXACT_CANDIDATE_RESULT_SCHEMA
        and validation.get("schema_version")
        == EXACT_CANDIDATE_VALIDATION_SCHEMA
        and validation.get("status") == "validated"
        and validation.get("all_executed_substeps_all_eight_safe") is True
        and validation.get("zero_protected_contact") is True
        and validation.get("zero_paper_CAR") is True
        and result.get("closed_loop", {}).get("action_count") == 187
        and result.get("closed_loop", {}).get("failure", {}).get("step") == 187
        and final.get("step") == 187
        and final.get("executed") is False
        and final.get("selector", {}).get("reason")
        == "no_exact_all_eight_safe_grid_candidate",
        "exact-candidate continuation prerequisite differs",
    )
    return result, {
        "result_path": str(result_path),
        "result_file_sha256": _file_sha256(result_path),
        "result_payload_sha256": result["result_payload_sha256"],
        "validation_path": str(validation_path),
        "validation_file_sha256": _file_sha256(validation_path),
        "validated_empty_safe_set_step": 187,
    }


def evaluate(
    *, repo_root: Path, population_manifest: Path, selected_manifest: Path,
    archived_root: Path, geometry_config_path: Path, exact_box_config_path: Path,
    source_config_path: Path, config_path: Path, prerequisite_result_path: Path,
    prerequisite_validation_path: Path, expected_commit: str, output_path: Path,
) -> dict[str, Any]:
    import numpy as np
    from main.evaluate_safelibero_aegis import (
        _goal_progress_definition,
        _goal_progress_snapshot,
        _goal_progress_summary,
        _runtime_imports,
        pairing_record,
        read_jsonl,
    )
    from main.multilink_ellipsoid.obstacle_primitives import (
        load_obstacle_primitive_config,
    )
    from main.multilink_ellipsoid.oracle_affine import SubstepEightConstraintProbe
    from main.multilink_ellipsoid.rollout import _dynamic_state_vector
    from main.multilink_ellipsoid.shadow import allocation_record, load_shadow_config

    started = time.perf_counter_ns()
    config = load_continue_config(config_path)
    source_config = load_two_step_config(source_config_path)
    accepted, prerequisite = _verify_prerequisite(
        config, prerequisite_result_path, prerequisite_validation_path
    )
    selected = load_selected_manifest(selected_manifest, source_config)
    case_id = config["primary_case"]["case_id"]
    selected_rows = [item for item in selected if item["case_id"] == case_id]
    _require(len(selected_rows) == 1, "primary selected row differs")
    selected_row = selected_rows[0]
    population_rows = {
        item["case_id"]: item for item in read_jsonl(population_manifest)
    }
    _require(case_id in population_rows, "primary population row missing")
    case = population_rows[case_id]
    archived_path = archived_root / selected_row["archived_relative_path"]
    archived = _load(archived_path)
    _require(
        _file_sha256(archived_path) == selected_row["archived_file_sha256"]
        and archived.get("result_payload_sha256")
        == selected_row["archived_payload_sha256"],
        "immutable Table-1 E05 result differs",
    )
    nominal_actions = archived.get("actions")
    expected_count = config["primary_case"]["expected_action_count"]
    prefix_count = config["primary_case"]["accepted_safe_prefix_action_count"]
    _require(
        isinstance(nominal_actions, list) and len(nominal_actions) == expected_count,
        "immutable action count differs",
    )
    accepted_prefix = [
        item for item in accepted.get("actions", []) if item.get("executed") is True
    ]
    _require(
        len(accepted_prefix) == prefix_count
        and [item["step"] for item in accepted_prefix] == list(range(prefix_count)),
        "accepted prefix ledger differs",
    )
    source = _git_identity(repo_root, expected_commit)
    allocation = allocation_record()
    runtime = _runtime_imports(include_aegis=False)
    geometry_config = load_shadow_config(geometry_config_path)
    exact_box_config = load_obstacle_primitive_config(exact_box_config_path)
    env = probe_env = None
    suffix_records: list[dict[str, Any]] = []
    try:
        env, probe_env, task, observation, setup = _build_pair(runtime, case)
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
                pairing[key] == accepted["pairing"][key],
                "pairing differs: %s" % key,
            )
        geometry, exact_boxes = _geometry(
            geometry_config=geometry_config,
            exact_box_config=exact_box_config,
            archived=archived,
            env=env,
            obstacle_name=setup["obstacle_name"],
        )
        probe = SubstepEightConstraintProbe(
            probe_env,
            geometry,
            active_obstacle_name=setup["obstacle_name"],
            quadratic_tolerance=1.0e-6,
            contact_distance_threshold_m=0.0,
            obstacle_primitive_union=exact_boxes,
        )
        initial_obstacle = np.asarray(
            observation["%s_pos" % setup["obstacle_name"]], dtype=np.float64
        ).copy()
        goal_definition, goal_atoms = _goal_progress_definition(env)
        initial_goal = _goal_progress_snapshot(
            env, goal_atoms, step=-1, previous_values=None
        )
        previous_goal_values = initial_goal["values"]
        maximum_prefix_obstacle_error = 0.0
        for record in accepted_prefix:
            step = int(record["step"])
            action = np.asarray(record["executed_action"], dtype=np.float64)
            observation, reward, done, _ = env.step(action.tolist())
            measurement = record["executed_measurement"]
            _require(
                float(reward) == float(measurement["reward"])
                and bool(done) is bool(measurement["done"]),
                "prefix reward/done differs",
            )
            state_hash = hashlib.sha256(
                _dynamic_state_vector(env).tobytes()
            ).hexdigest()
            _require(
                state_hash == measurement["next_state_sha256"],
                "prefix state hash differs at step %d" % step,
            )
            displacement = float(np.sum(np.abs(
                np.asarray(
                    observation["%s_pos" % setup["obstacle_name"]],
                    dtype=np.float64,
                ) - initial_obstacle
            )))
            error = abs(
                displacement - float(record["active_obstacle_l1_displacement_m"])
            )
            maximum_prefix_obstacle_error = max(
                maximum_prefix_obstacle_error, error
            )
            _require(error <= 1.0e-10, "prefix obstacle trace differs")
            previous_goal_values = record["goal_progress"]["values"]
        _require(not bool(accepted_prefix[-1]["executed_measurement"]["done"]),
                 "prefix unexpectedly completed task")
        first_proxy_violation_step = None
        first_contact_step = None
        first_car_step = None
        maximum_obstacle_displacement = float(
            accepted["closed_loop"]["maximum_active_obstacle_l1_displacement_m"]
        )
        for step in range(prefix_count, expected_count):
            action = _canonical_action(nominal_actions[step], step)
            measured = _measured_env_step(env, action, probe)
            observation = measured.pop("observation")
            margins = measured["minimum_all_eight_substep_clearance_m"]
            if first_proxy_violation_step is None and min(margins) < 0.0:
                first_proxy_violation_step = step
            if first_contact_step is None and measured["raw_protected_contact_count"]:
                first_contact_step = step
            displacement = float(np.sum(np.abs(
                np.asarray(
                    observation["%s_pos" % setup["obstacle_name"]],
                    dtype=np.float64,
                ) - initial_obstacle
            )))
            maximum_obstacle_displacement = max(
                maximum_obstacle_displacement, displacement
            )
            if first_car_step is None and displacement > 1.0e-3:
                first_car_step = step
            goal = _goal_progress_snapshot(
                env, goal_atoms, step=step, previous_values=previous_goal_values
            )
            previous_goal_values = goal["values"]
            _require(
                goal["all_satisfied"] is measured["done"],
                "native goal vector differs",
            )
            suffix_records.append({
                "step": step,
                "nominal_action": action.tolist(),
                "executed_action": action.tolist(),
                "modified": False,
                "intervention": False,
                "safety_filter_bypassed": True,
                "unsafe_continuation": True,
                "continuation_reason": "prior_empty_safe_set_at_step_187",
                "candidate_search_used": False,
                "affine_QP_used": False,
                "executed_measurement": measured,
                "active_obstacle_l1_displacement_m": displacement,
                "goal_progress": goal,
                "executed": True,
            })
            if measured["done"]:
                break
        combined = [dict(item) for item in accepted_prefix] + suffix_records
        goal_summary = _goal_progress_summary(initial_goal, combined)
        native_success_step = goal_summary["first_all_satisfied_step"]
        native_success = native_success_step is not None
        all_margins = np.asarray([
            item["executed_measurement"][
                "minimum_all_eight_substep_clearance_m"
            ] for item in combined
        ], dtype=np.float64)
        minimum_all_eight = np.min(all_margins, axis=0)
        result = {
            "schema_version": CONTINUE_RESULT_SCHEMA,
            "status": "complete",
            "scientific_result": True,
            "claim_scope": config["claim_scope"],
            "source": source,
            "allocation": allocation,
            "config": config,
            "prerequisite": prerequisite,
            "case_id": case_id,
            "archived_table1": {
                "path": str(archived_path),
                "file_sha256": _file_sha256(archived_path),
                "payload_sha256": archived["result_payload_sha256"],
                "read_only": True,
            },
            "pairing": pairing,
            "goal_progress_definition": goal_definition,
            "goal_progress_summary": goal_summary,
            "prefix_replay": {
                "executed_action_count": prefix_count,
                "all_dynamic_state_hashes_match": True,
                "maximum_obstacle_displacement_error_m": (
                    maximum_prefix_obstacle_error
                ),
            },
            "continuation": {
                "start_step": prefix_count,
                "action_count": len(suffix_records),
                "total_executed_action_count": len(combined),
                "safety_filter_bypassed": True,
                "candidate_search_used": False,
                "affine_QP_used": False,
                "first_proxy_violation_step": first_proxy_violation_step,
                "first_protected_contact_step": first_contact_step,
                "first_paper_CAR_step": first_car_step,
                "maximum_active_obstacle_l1_displacement_m": (
                    maximum_obstacle_displacement
                ),
                "minimum_all_eight_executed_substep_clearance_m": (
                    minimum_all_eight.tolist()
                ),
                "minimum_distal_executed_substep_clearance_m": float(
                    np.min(minimum_all_eight[:7])
                ),
                "minimum_released_AEGIS_EE_proxy_substep_clearance_m": float(
                    minimum_all_eight[7]
                ),
                "native_task_success": native_success,
                "native_task_success_step": native_success_step,
                "ended_by": "native_task_success" if native_success else "action_horizon",
            },
            "actions": combined,
            "visual_replay": {
                "status": "pending_single_context_replay",
            },
            "decision": {
                "continued_after_empty_safe_set": True,
                "native_task_success": native_success,
                "collision_free": bool(
                    first_contact_step is None and first_car_step is None
                ),
                "safety_success": False,
                "deployable_method_demonstrated": False,
                "neural_training_authorized": False,
            },
            "wall_seconds": (time.perf_counter_ns() - started) * 1.0e-9,
        }
        result["result_payload_sha256"] = _hash_without(
            result, "result_payload_sha256"
        )
        return result
    finally:
        if probe_env is not None:
            probe_env.close()
        if env is not None:
            env.close()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--population-manifest", type=Path, required=True)
    parser.add_argument("--selected-manifest", type=Path, required=True)
    parser.add_argument("--archived-root", type=Path, required=True)
    parser.add_argument("--geometry-config", type=Path, required=True)
    parser.add_argument("--exact-box-config", type=Path, required=True)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prerequisite-result", type=Path, required=True)
    parser.add_argument("--prerequisite-validation", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = evaluate(
        repo_root=args.repo_root.resolve(),
        population_manifest=args.population_manifest.resolve(),
        selected_manifest=args.selected_manifest.resolve(),
        archived_root=args.archived_root.resolve(),
        geometry_config_path=args.geometry_config.resolve(),
        exact_box_config_path=args.exact_box_config.resolve(),
        source_config_path=args.source_config.resolve(),
        config_path=args.config.resolve(),
        prerequisite_result_path=args.prerequisite_result.resolve(),
        prerequisite_validation_path=args.prerequisite_validation.resolve(),
        expected_commit=args.expected_commit,
        output_path=args.output.resolve(),
    )
    _atomic_write(args.output.resolve(), result)
    print(json.dumps(result["decision"], sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
