#!/usr/bin/env python3
"""Evaluate unseen episodes and conditionally run receding two-step E05."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping, Optional

from main.multilink_ellipsoid.two_step_margin import (
    TWO_STEP_DATASET_SCHEMA,
    TWO_STEP_RESULT_SCHEMA,
    TWO_STEP_TRAINING_SCHEMA,
    feature_context,
    load_model,
    load_selected_manifest,
    load_two_step_config,
    project_first_action,
    summarize_chunk,
)
from scripts.collect_distal_boundary_generalization_moka10 import _canonical_action
from scripts.evaluate_distal_execution_margin_nn_e05 import _build_pair, _geometry
from scripts.replay_distal_three_ellipsoid_multicbf import (
    PAPER_CAR_THRESHOLD_M,
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


def _measured_env_step(env: Any, action: Any, probe: Any) -> dict[str, Any]:
    """Execute once while recording exact geometry/contact at every substep."""

    import numpy as np
    from main.multilink_ellipsoid.rollout import _base_env, _dynamic_state_vector

    base = _base_env(env)
    trace = [probe._capture(env, 0, "interval_start")]
    original_update = base._update_observables

    def traced_update(*args: Any, **kwargs: Any) -> Any:
        value = original_update(*args, **kwargs)
        trace.append(
            probe._capture(env, len(trace), "after_internal_mujoco_step")
        )
        return value

    base._update_observables = traced_update
    started = time.perf_counter_ns()
    try:
        observation, reward, done, info = env.step(action.tolist())
    finally:
        base._update_observables = original_update
    elapsed = (time.perf_counter_ns() - started) * 1.0e-9
    expected = int(base.control_timestep / base.model_timestep)
    _require(len(trace) == expected + 1, "two-step closed-loop substep count differs")
    clearance = np.asarray(
        [item["clearance_m"][:7] for item in trace], dtype=np.float64
    )
    contacts = [
        {**event, "substep_index": int(substep["substep_index"])}
        for substep in trace
        for event in substep["contact_events"]
    ]
    obstacle_positions = np.asarray(
        [item["obstacle_position_m"] for item in trace], dtype=np.float64
    )
    displacement = np.sum(np.abs(obstacle_positions - obstacle_positions[0]), axis=1)
    vector = _dynamic_state_vector(env)
    return {
        "observation": observation,
        "reward": float(reward), "done": bool(done), "info": info,
        "minimum_substep_clearance_m": np.min(clearance, axis=0).tolist(),
        "raw_protected_contact_count": len(contacts),
        "raw_protected_contact_events": contacts,
        "maximum_within_step_obstacle_l1_displacement_m": float(np.max(displacement)),
        "next_state_sha256": hashlib.sha256(vector.tobytes()).hexdigest(),
        "env_step_wall_seconds": elapsed,
    }


def _held_out_projection(
    *, runtime: Mapping[str, Any], case: Mapping[str, Any],
    archived: Mapping[str, Any], geometry_placeholder: Mapping[str, Any],
    step: int, geometry_config: Mapping[str, Any],
    exact_box_config: Mapping[str, Any], config: Mapping[str, Any],
    model: Any, model_state: Mapping[str, Any],
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
        for index in range(step):
            env.step(_canonical_action(archived["actions"][index], index).tolist())
        probe = SubstepEightConstraintProbe(
            probe_env, geometry, active_obstacle_name=setup["obstacle_name"],
            quadratic_tolerance=1.0e-6, contact_distance_threshold_m=0.0,
            obstacle_primitive_union=exact_boxes,
        )
        first = _canonical_action(archived["actions"][step], step)
        second = _canonical_action(archived["actions"][step + 1], step + 1)
        context = feature_context(env, probe)
        nominal_summary = summarize_chunk(probe.rollout_chunk(env, [first, second]))
        projection = project_first_action(
            model, model_state, context, first[:3], second[:3], config
        )
        exact = None
        if projection["valid"]:
            action = first.copy()
            action[:3] = projection["projected_xyz"]
            exact = summarize_chunk(probe.rollout_chunk(env, [action, second]))
        qps = [
            item["qp"] for item in projection["iterations"] if item.get("qp") is not None
        ]
        seven_row_qp = bool(
            qps
            and all(
                item["valid"] and len(item["lower"]) == 7 for item in qps
            )
        )
        exact_safe = bool(
            exact is not None and exact["D_opt_proxy_safe"] and exact["D_sim_raw_safe"]
        )
        return {
            "state_step": int(step),
            "nominal_exact_minimum_margin_m": float(
                min(nominal_summary["minimum_substep_clearance_m"])
            ),
            "projection": projection,
            "valid_seven_row_qp": seven_row_qp,
            "projected_exact_summary": exact,
            "projection_gate_pass": bool(
                projection["valid"] and seven_row_qp and exact_safe
            ),
        }
    finally:
        if probe_env is not None:
            probe_env.close()
        if env is not None:
            env.close()


def _closed_loop_e05(
    *, runtime: Mapping[str, Any], case: Mapping[str, Any],
    archived: Mapping[str, Any], geometry_placeholder: Mapping[str, Any],
    geometry_config: Mapping[str, Any], exact_box_config: Mapping[str, Any],
    config: Mapping[str, Any], model: Any, model_state: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute the learned two-step filter after every executed action.

    This fastest diagnostic retains the immutable successful AEGIS action
    sequence as the nominal plan.  It performs no online cloned rollout or
    candidate search; the simulator is used only to execute and measure the
    selected action.
    """

    import numpy as np
    from main.multilink_ellipsoid.oracle_affine import SubstepEightConstraintProbe

    env = measurement_env = None
    records = []
    failure: Optional[dict[str, Any]] = None
    try:
        env, measurement_env, _, observation, setup = _build_pair(runtime, case)
        geometry, exact_boxes = _geometry(
            geometry_config=geometry_config, exact_box_config=exact_box_config,
            archived=geometry_placeholder, env=env,
            obstacle_name=setup["obstacle_name"],
        )
        probe = SubstepEightConstraintProbe(
            measurement_env, geometry,
            active_obstacle_name=setup["obstacle_name"],
            quadratic_tolerance=1.0e-6, contact_distance_threshold_m=0.0,
            obstacle_primitive_union=exact_boxes,
        )
        initial_obstacle = np.asarray(
            observation["%s_pos" % setup["obstacle_name"]], dtype=np.float64
        ).copy()
        first_car_step = None
        first_contact_step = None
        maximum_displacement = 0.0
        maximum_filter_seconds = 0.0
        actions = archived["actions"]
        for step in range(len(actions)):
            first = _canonical_action(actions[step], step)
            second = (
                _canonical_action(actions[step + 1], step + 1)
                if step + 1 < len(actions)
                else first.copy()
            )
            context = feature_context(env, probe)
            projection = project_first_action(
                model, model_state, context, first[:3], second[:3], config
            )
            maximum_filter_seconds = max(
                maximum_filter_seconds, float(projection["total_wall_seconds"])
            )
            if not projection["valid"]:
                failure = {
                    "step": step, "reason": projection["reason"],
                    "component": "learned_two_step_projection",
                }
                break
            executed = first.copy()
            executed[:3] = projection["projected_xyz"]
            measured = _measured_env_step(env, executed, probe)
            observation = measured.pop("observation")
            displacement = float(
                np.sum(
                    np.abs(
                        np.asarray(
                            observation["%s_pos" % setup["obstacle_name"]],
                            dtype=np.float64,
                        )
                        - initial_obstacle
                    )
                )
            )
            maximum_displacement = max(maximum_displacement, displacement)
            if first_car_step is None and displacement > PAPER_CAR_THRESHOLD_M:
                first_car_step = step
            if (
                first_contact_step is None
                and measured["raw_protected_contact_count"] > 0
            ):
                first_contact_step = step
            records.append(
                {
                    "step": step,
                    "nominal_action": first.tolist(),
                    "nominal_second_action": second.tolist(),
                    "executed_action": executed.tolist(),
                    "modified": bool(
                        not np.array_equal(executed[:3], first[:3])
                    ),
                    "projection": projection,
                    "measurement": measured,
                    "active_obstacle_l1_displacement_m": displacement,
                }
            )
            if measured["done"]:
                break
        task_success = bool(records and records[-1]["measurement"]["done"])
        contact_free = first_contact_step is None
        car_pass = first_car_step is None
        problem_solved = bool(
            failure is None and task_success and contact_free and car_pass
        )
        return {
            "status": "complete" if failure is None else "method_failure",
            "nominal_source": (
                "immutable_successful_released_AEGIS_action_sequence_recomputed_"
                "from_each_executed_state"
            ),
            "online_cloned_simulator_oracle_used": False,
            "action_count": len(records),
            "modified_action_count": sum(item["modified"] for item in records),
            "first_modified_step": next(
                (item["step"] for item in records if item["modified"]), None
            ),
            "first_protected_contact_step": first_contact_step,
            "first_paper_car_step": first_car_step,
            "maximum_active_obstacle_l1_displacement_m": maximum_displacement,
            "minimum_exact_substep_clearance_m": (
                None
                if not records
                else float(
                    min(
                        min(item["measurement"]["minimum_substep_clearance_m"])
                        for item in records
                    )
                )
            ),
            "native_task_success": task_success,
            "native_task_success_step": (
                None if not task_success else int(records[-1]["step"])
            ),
            "maximum_filter_wall_seconds": maximum_filter_seconds,
            "primary_problem_solved": problem_solved,
            "failure": failure,
            "actions": records,
        }
    finally:
        if measurement_env is not None:
            measurement_env.close()
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
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dataset-result", type=Path, required=True)
    parser.add_argument("--dataset-validation", type=Path, required=True)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter_ns()
    from main.evaluate_safelibero_aegis import _runtime_imports, read_jsonl
    from main.multilink_ellipsoid.obstacle_primitives import load_obstacle_primitive_config
    from main.multilink_ellipsoid.shadow import allocation_record, load_shadow_config

    config = load_two_step_config(args.config.resolve())
    selected = load_selected_manifest(args.selected_manifest.resolve(), config)
    dataset = _load(args.dataset.resolve())
    training = _load(args.training.resolve())
    validation = _load(args.dataset_validation.resolve())
    _require(
        dataset.get("schema_version") == TWO_STEP_DATASET_SCHEMA
        and dataset.get("source_commit") == args.expected_commit,
        "two-step evaluation dataset differs",
    )
    _require(
        training.get("schema_version") == TWO_STEP_TRAINING_SCHEMA
        and training.get("source", {}).get("commit") == args.expected_commit
        and training.get("training_payload_sha256")
        == _hash_without(training, "training_payload_sha256"),
        "two-step evaluation training differs",
    )
    _require(
        validation.get("neural_training_authorized") is True
        and validation.get("dataset_file_sha256")
        == _file_sha256(args.dataset.resolve()),
        "two-step evaluation dataset gate differs",
    )
    geometry_config = load_shadow_config(args.geometry_config.resolve())
    exact_box_config = load_obstacle_primitive_config(args.exact_box_config.resolve())
    population = {
        item["case_id"]: item for item in read_jsonl(args.population_manifest.resolve())
    }
    episode_state = {item["case_id"]: item for item in dataset["episode_results"]}
    test_rows = [item for item in selected if item["split"] == "test"]
    primary_row = next(
        item for item in selected if item["case_id"] == "vlsa-t1-goal-ii-t0-e05"
    )
    placeholder_path = args.archived_root.resolve() / primary_row["archived_relative_path"]
    geometry_placeholder = _load(placeholder_path)
    _require(
        _file_sha256(placeholder_path) == primary_row["archived_file_sha256"]
        and geometry_placeholder.get("result_payload_sha256")
        == primary_row["archived_payload_sha256"],
        "two-step evaluation geometry placeholder differs",
    )
    runtime = _runtime_imports(include_aegis=False)
    arm_results = {}
    loaded_models = {}
    for arm in config["training"]["arms"]:
        identity = training["arms"][arm]["model_artifact"]
        model_path = Path(identity["path"]).resolve()
        _require(
            model_path.is_file()
            and _file_sha256(model_path) == identity["file_sha256"]
            and model_path.parent == args.training.resolve().parent,
            "two-step model artifact differs",
        )
        model, model_state = load_model(model_path, device="cpu")
        loaded_models[arm] = (model, model_state)
        projections = {}
        for row in test_rows:
            case_id = row["case_id"]
            archived_path = args.archived_root.resolve() / row["archived_relative_path"]
            archived = _load(archived_path)
            projections[case_id] = _held_out_projection(
                runtime=runtime, case=population[case_id], archived=archived,
                geometry_placeholder=geometry_placeholder,
                step=int(episode_state[case_id]["selected_step"]),
                geometry_config=geometry_config, exact_box_config=exact_box_config,
                config=config, model=model, model_state=model_state,
            )
        model_gate = bool(training["arms"][arm]["training"]["held_out_model_gate_pass"])
        projection_gate = bool(
            all(item["projection_gate_pass"] for item in projections.values())
        )
        arm_results[arm] = {
            "model_artifact": identity,
            "training": training["arms"][arm]["training"],
            "held_out_exact_projections": projections,
            "decision": {
                "held_out_model_gate_pass": model_gate,
                "every_test_episode_projection_gate_pass": projection_gate,
                "arm_generalization_gate_pass": bool(model_gate and projection_gate),
            },
        }
    factorized_go = bool(
        arm_results["factorized"]["decision"]["arm_generalization_gate_pass"]
    )
    closed_loop = None
    if factorized_go:
        primary_archived = _load(placeholder_path)
        model, model_state = loaded_models["factorized"]
        closed_loop = _closed_loop_e05(
            runtime=runtime, case=population[primary_row["case_id"]],
            archived=primary_archived, geometry_placeholder=geometry_placeholder,
            geometry_config=geometry_config, exact_box_config=exact_box_config,
            config=config, model=model, model_state=model_state,
        )
    validated = bool(
        factorized_go
        and closed_loop is not None
        and closed_loop["primary_problem_solved"]
    )
    result = {
        "schema_version": TWO_STEP_RESULT_SCHEMA,
        "status": "complete", "scientific_result": True,
        "claim_scope": config["claim_scope"],
        "source": _git_identity(args.repo_root.resolve(), args.expected_commit),
        "allocation": allocation_record(), "config": config,
        "test_case_ids": [item["case_id"] for item in test_rows],
        "dataset": {
            "path": str(args.dataset.resolve()),
            "file_sha256": _file_sha256(args.dataset.resolve()),
            "payload_sha256": dataset["dataset_payload_sha256"],
        },
        "training": {
            "path": str(args.training.resolve()),
            "file_sha256": _file_sha256(args.training.resolve()),
            "payload_sha256": training["training_payload_sha256"],
        },
        "arms": arm_results,
        "closed_loop_e05": closed_loop,
        "decision": {
            "factorized_grouped_generalization_go": factorized_go,
            "closed_loop_e05_executed": closed_loop is not None,
            "research_direction_validated": validated,
            "stop_reason": (
                None
                if validated
                else (
                    "factorized_held_out_gate_failed"
                    if not factorized_go
                    else "factorized_closed_loop_e05_failed"
                )
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
