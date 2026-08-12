#!/usr/bin/env python3
"""Run the matched direction-first ExecGrad mechanism experiment on H100."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping

from main.multilink_ellipsoid.execgrad import (
    PREPROCESS_SCHEMA,
    action_gradient_from_pairs,
    direction_metrics,
    execgrad_fitted_decision,
    load_execgrad_config,
    predict_execgrad_models,
    save_execgrad_weights,
    soft_min_trace,
    split_temporal_metrics,
    train_matched_execgrad_models,
)
from main.multilink_ellipsoid.factorized_execution_pilot import (
    cosine_summary,
    payload_sha256,
    safety_metrics,
)
from main.multilink_ellipsoid.shadow import (
    _eef_jacobian,
    _geom_jacobians,
    _raw_model_data,
)
from scripts.collect_distal_boundary_generalization_moka10 import (
    _canonical_action,
    _restore_env,
    _snapshot_env,
)
from scripts.collect_distal_complete_osc_margin_moka10 import (
    _GEOMETRY_PLACEHOLDER_CASE_ID,
    _geometry_placeholder_row,
)
from scripts.evaluate_distal_direct_horizon_displacement_moka10 import (
    _hash_array,
    add_direct_horizon_arguments,
    direct_paths,
)
from scripts.evaluate_distal_execution_margin_nn_e05 import _build_pair, _geometry
from scripts.evaluate_distal_factorized_cumulative_delta_q_moka10 import prepare
from scripts.evaluate_distal_factorized_one_sided_geometry_moka10 import (
    geometry_kwargs,
)
from scripts.evaluate_distal_native_geom_inventory_moka10 import _read_manifest
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write,
    _file_sha256,
    _git_identity,
    _load,
    _require,
)


RESULT_SCHEMA = "vlsa_distal_execgrad_direction_result.v1"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    add_direct_horizon_arguments(parser)
    parser.add_argument("--execgrad-config", type=Path, required=True)
    parser.add_argument("--normalized-direct-run", type=Path, required=True)
    parser.add_argument("--normalized-direct-validation", type=Path, required=True)
    parser.add_argument("--future-untouched-manifest", type=Path, required=True)
    parser.add_argument("--preprocess", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)


def extra_paths(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "execgrad_config": args.execgrad_config.resolve(),
        "normalized_direct_run": args.normalized_direct_run.resolve(),
        "normalized_direct_validation": args.normalized_direct_validation.resolve(),
        "future_untouched_manifest": args.future_untouched_manifest.resolve(),
        "preprocess": args.preprocess.resolve(),
        "model": args.model.resolve(),
        "predictions": args.predictions.resolve(),
        "output": args.output.resolve(),
    }


def validate_sources(paths: Mapping[str, Path], config: Mapping[str, Any]) -> None:
    source = config["immutable_source"]
    checks = (
        (paths["complete_dataset"], "complete_dataset_file_sha256"),
        (paths["complete_collection"], "complete_collection_file_sha256"),
        (paths["array"], "trajectory_array_file_sha256"),
        (paths["metadata"], "trajectory_metadata_file_sha256"),
        (paths["collection"], "trajectory_collection_file_sha256"),
        (paths["normalized_direct_run"] / "model.npz", "normalized_direct_model_file_sha256"),
        (paths["normalized_direct_run"] / "predictions.npz", "normalized_direct_predictions_file_sha256"),
        (paths["normalized_direct_run"] / "result.json", "normalized_direct_result_file_sha256"),
        (paths["normalized_direct_validation"], "normalized_direct_validation_file_sha256"),
        (paths["future_untouched_manifest"], "future_untouched_manifest_file_sha256"),
    )
    for path, key in checks:
        _require(_file_sha256(path) == source[key], "ExecGrad immutable source differs: %s" % key)
    direct = _load(paths["normalized_direct_run"] / "result.json")
    validation = _load(paths["normalized_direct_validation"])
    _require(
        direct.get("result_payload_sha256")
        == payload_sha256(direct, "result_payload_sha256")
        and validation.get("valid") is True
        and validation.get("result_payload_sha256") == direct["result_payload_sha256"],
        "ExecGrad direct baseline receipt differs",
    )


def _resolved_rate_nominal_trace(
    env: Any,
    q0: Any,
    action_chunk: Any,
    position_indexes: Any,
    velocity_indexes: Any,
    config: Mapping[str, Any],
) -> Any:
    """Return a deterministic, controller-scale-aware kinematic baseline."""

    import numpy as np

    q_start = np.asarray(q0, dtype=np.float64).copy()
    actions = np.asarray(action_chunk, dtype=np.float64)
    if q_start.shape != (7,) or actions.shape != (2, 7):
        raise ValueError("ExecGrad nominal rollout input differs")
    controller = env.robots[0].controller
    model, _ = _raw_model_data(env.sim)
    joint_ids = [int(model.dof_jntid[int(index)]) for index in velocity_indexes]
    lower = np.asarray(
        [model.jnt_range[index][0] for index in joint_ids], dtype=np.float64
    )
    upper = np.asarray(
        [model.jnt_range[index][1] for index in joint_ids], dtype=np.float64
    )
    if lower.shape != (7,) or upper.shape != (7,) or not np.all(lower < upper):
        raise ValueError("authoritative Panda joint ranges differ")
    trace = [q_start.copy()]
    damping = float(config["nominal_rollout"]["damping"])
    maximum = float(
        config["nominal_rollout"]["maximum_absolute_joint_delta_per_segment_rad"]
    )
    for action in actions:
        env.sim.data.qpos[position_indexes] = q_start
        env.sim.forward()
        jacobian = _eef_jacobian(env, velocity_indexes)
        scaled = np.asarray(controller.scale_action(action[:6].copy()), dtype=np.float64)
        if scaled.shape != (6,) or not np.all(np.isfinite(scaled)):
            raise ValueError("authoritative OSC action scaling differs")
        regularized = jacobian @ jacobian.T + damping * damping * np.eye(6)
        delta = jacobian.T @ np.linalg.solve(regularized, scaled)
        delta = np.clip(delta, -maximum, maximum)
        target = np.clip(q_start + delta, lower, upper)
        for substep in range(1, 26):
            alpha = substep / 25.0
            trace.append(q_start + alpha * (target - q_start))
        q_start = target
    output = np.asarray(trace, dtype=np.float64)
    if output.shape != (51, 7) or not np.all(np.isfinite(output)):
        raise ValueError("ExecGrad nominal q trace differs")
    return output


def _scaled_pose_basis(env: Any) -> dict[str, Any]:
    import numpy as np

    controller = env.robots[0].controller
    positive = []
    negative = []
    for dimension in range(6):
        basis = np.zeros(6, dtype=np.float64)
        basis[dimension] = 1.0
        positive.append(np.asarray(controller.scale_action(basis.copy()), dtype=np.float64).tolist())
        negative.append(np.asarray(controller.scale_action(-basis.copy()), dtype=np.float64).tolist())
    return {"positive_unit_action": positive, "negative_unit_action": negative}


def _ellipsoid_centers(probe: Any, env: Any) -> Any:
    import numpy as np

    output = np.asarray(
        [item.center for item in probe._ellipsoids(env)[:7]], dtype=np.float64
    )
    if output.shape != (7, 3) or not np.all(np.isfinite(output)):
        raise ValueError("ExecGrad link-center trace differs")
    return output


def build_preprocess(
    *,
    paths: Mapping[str, Path],
    config: Mapping[str, Any],
    factorized_config: Mapping[str, Any],
    complete_dataset: Mapping[str, Any],
    complete_collection: Mapping[str, Any],
    arrays: Mapping[str, Any],
    sensitivities: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build Q_nom and exact link-center JVP labels in the H100 allocation."""

    import numpy as np
    from main.evaluate_safelibero_aegis import (
        _runtime_imports,
        read_jsonl,
        validate_case_row,
    )
    from main.multilink_ellipsoid.obstacle_primitives import (
        load_obstacle_primitive_config,
    )
    from main.multilink_ellipsoid.oracle_affine import SubstepEightConstraintProbe
    from main.multilink_ellipsoid.rollout import _dynamic_state_vector
    from main.multilink_ellipsoid.shadow import load_shadow_config

    source_rows = []
    for path in (paths["selected"], paths["same_task"], paths["targeted"]):
        source_rows.extend(_read_manifest(path, _file_sha256(path)))
    by_case = {str(row["case_id"]): row for row in source_rows}
    placeholder_row = _geometry_placeholder_row(by_case)
    placeholder_path = paths["archived"] / placeholder_row["archived_relative_path"]
    placeholder = _load(placeholder_path)
    _require(
        _file_sha256(placeholder_path) == placeholder_row["archived_file_sha256"]
        and placeholder.get("result_payload_sha256") == placeholder_row["archived_payload_sha256"]
        and placeholder.get("case_id") == _GEOMETRY_PLACEHOLDER_CASE_ID,
        "ExecGrad geometry placeholder differs",
    )
    population = {
        row["case_id"]: row for row in read_jsonl(paths["population"])
    }
    runtime = _runtime_imports(include_aegis=False)
    geometry_config = load_shadow_config(paths["geometry"])
    exact_box_config = load_obstacle_primitive_config(paths["exact_box"])
    exact_q = np.asarray(arrays["joint_position_rad"], dtype=np.float64)
    action = np.asarray(arrays["action_chunk"], dtype=np.float64)
    state_index = np.asarray(arrays["state_index"], dtype=np.int64)
    candidate_index = np.asarray(arrays["candidate_index"], dtype=np.int64)
    state_count = int(config["population"]["existing_state_count"])
    nominal_q = np.full_like(exact_q, np.nan)
    center_jacobian = np.full((state_count, 51, 7, 3, 7), np.nan, dtype=np.float64)
    exact_center_jvp = np.full(
        (len(sensitivities["state_index"]), 51, 7, 3), np.nan, dtype=np.float64
    )
    row_by_identity = {
        (int(state), int(candidate)): int(row)
        for row, (state, candidate) in enumerate(zip(state_index, candidate_index))
    }
    pairs_by_state: dict[int, list[int]] = {state: [] for state in range(state_count)}
    for pair, state in enumerate(np.asarray(sensitivities["state_index"], dtype=np.int64)):
        pairs_by_state[int(state)].append(int(pair))
    scaled_basis = None
    deterministic_maximum = 0.0
    evaluated_states = 0
    for case_id in sorted(set(str(record["case_id"]) for record in complete_dataset["state_records"])):
        case = population[case_id]
        validate_case_row(case, paths["repo"])
        archived_row = by_case[case_id]
        archived_path = paths["archived"] / archived_row["archived_relative_path"]
        archived = _load(archived_path)
        _require(
            _file_sha256(archived_path) == archived_row["archived_file_sha256"]
            and archived.get("result_payload_sha256") == archived_row["archived_payload_sha256"],
            "ExecGrad archived episode differs",
        )
        env = probe_env = None
        try:
            env, probe_env, _, _, setup = _build_pair(runtime, case)
            geometry, exact_boxes = _geometry(
                geometry_config=geometry_config,
                exact_box_config=exact_box_config,
                archived=placeholder,
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
            records = sorted(
                (record for record in complete_dataset["state_records"] if record["case_id"] == case_id),
                key=lambda record: int(record["state_step"]),
            )
            by_step = {int(record["state_step"]): record for record in records}
            maximum_step = max(by_step)
            robot = env.robots[0]
            position_indexes = np.asarray(robot._ref_joint_pos_indexes, dtype=np.int64)
            velocity_indexes = np.asarray(robot._ref_joint_vel_indexes, dtype=np.int64)
            for step in range(maximum_step + 1):
                if step in by_step:
                    record = by_step[step]
                    state = int(record["state_index"])
                    _require(
                        hashlib.sha256(_dynamic_state_vector(env).tobytes()).hexdigest()
                        == record["dynamic_state_sha256"],
                        "ExecGrad replayed state differs",
                    )
                    snapshot = _snapshot_env(env)
                    if scaled_basis is None:
                        scaled_basis = _scaled_pose_basis(env)
                    state_rows = np.flatnonzero(state_index == state)
                    if len(state_rows) != 93:
                        raise ValueError("ExecGrad state candidate count differs")
                    q0 = exact_q[state_rows[0], 0]
                    for row in state_rows:
                        nominal_q[row] = _resolved_rate_nominal_trace(
                            env,
                            q0,
                            action[row],
                            position_indexes,
                            velocity_indexes,
                            config,
                        )
                    repeated = _resolved_rate_nominal_trace(
                        env,
                        q0,
                        action[state_rows[0]],
                        position_indexes,
                        velocity_indexes,
                        config,
                    )
                    deterministic_maximum = max(
                        deterministic_maximum,
                        float(np.max(np.abs(repeated - nominal_q[state_rows[0]]))),
                    )
                    nominal_row = row_by_identity[(state, 0)]
                    for horizon, q in enumerate(exact_q[nominal_row]):
                        env.sim.data.qpos[position_indexes] = q
                        env.sim.forward()
                        links = probe._ellipsoids(env)[:7]
                        for link_index, link in enumerate(links):
                            center_jacobian[state, horizon, link_index] = _geom_jacobians(
                                env, link, velocity_indexes
                            )[0]
                    for pair in pairs_by_state[state]:
                        negative_row = int(sensitivities["negative_row_index"][pair])
                        positive_row = int(sensitivities["positive_row_index"][pair])
                        dimension = int(sensitivities["dimension_index"][pair])
                        flat = action.reshape(len(action), -1)
                        denominator = flat[positive_row, dimension] - flat[negative_row, dimension]
                        negative_center = np.empty((51, 7, 3), dtype=np.float64)
                        positive_center = np.empty_like(negative_center)
                        for horizon in range(51):
                            env.sim.data.qpos[position_indexes] = exact_q[negative_row, horizon]
                            env.sim.forward()
                            negative_center[horizon] = _ellipsoid_centers(probe, env)
                            env.sim.data.qpos[position_indexes] = exact_q[positive_row, horizon]
                            env.sim.forward()
                            positive_center[horizon] = _ellipsoid_centers(probe, env)
                        exact_center_jvp[pair] = (
                            positive_center - negative_center
                        ) / denominator
                    _restore_env(env, snapshot)
                    evaluated_states += 1
                if step < maximum_step:
                    env.step(_canonical_action(archived["actions"][step], step).tolist())
        finally:
            if probe_env is not None:
                probe_env.close()
            if env is not None:
                env.close()
    if (
        evaluated_states != state_count
        or not np.all(np.isfinite(nominal_q))
        or not np.all(np.isfinite(center_jacobian))
        or not np.all(np.isfinite(exact_center_jvp))
        or scaled_basis is None
    ):
        raise ValueError("ExecGrad kinematic preprocessing is incomplete")
    predicted_center_jvp = np.einsum(
        "skpcj,skj->skpc",
        center_jacobian[np.asarray(sensitivities["state_index"], dtype=np.int64)],
        np.asarray(sensitivities["joint_sensitivity_rad_per_action"], dtype=np.float64),
    )
    state_split = {
        int(state): str(split)
        for state, split in zip(state_index, np.asarray(arrays["split"], dtype=object))
    }
    linearization = {}
    for split_name in ("train", "validation", "test"):
        selected = np.asarray(
            [state_split[int(state)] == split_name for state in sensitivities["state_index"]],
            dtype=bool,
        )
        error = predicted_center_jvp[selected] - exact_center_jvp[selected]
        cosine = cosine_summary(exact_center_jvp[selected], predicted_center_jvp[selected])
        linearization[split_name] = {
            "scalar_RMSE_m_per_action": float(np.sqrt(np.mean(error ** 2))),
            "vector_cosine": cosine,
        }
    validation_contract = config["nominal_rollout"]["validation"]
    gate = {
        "deterministic_repeat": deterministic_maximum == 0.0,
        "finite_nominal_q": bool(np.all(np.isfinite(nominal_q))),
        "validation_center_JVP_RMSE": bool(
            linearization["validation"]["scalar_RMSE_m_per_action"]
            <= float(validation_contract["maximum_center_JVP_linearization_RMSE_m_per_action"])
        ),
        "validation_center_JVP_cosine": bool(
            linearization["validation"]["vector_cosine"]["mean_cosine"] is not None
            and float(linearization["validation"]["vector_cosine"]["mean_cosine"])
            >= float(validation_contract["minimum_center_JVP_cosine"])
        ),
    }
    audit = {
        "schema_version": PREPROCESS_SCHEMA,
        "evaluated_state_count": evaluated_states,
        "evaluated_action_count": int(len(nominal_q)),
        "deterministic_repeat_maximum_absolute_joint_error_rad": deterministic_maximum,
        "authoritative_scaled_pose_basis": scaled_basis,
        "center_JVP_linearization": linearization,
        "gate_tests": gate,
        "apparatus_gate_pass": bool(all(gate.values())),
        "nominal_temporal_joint": split_temporal_metrics(
            nominal_q, exact_q, arrays["split"]
        ),
    }
    arrays_to_save = {
        "schema_version": np.asarray(PREPROCESS_SCHEMA),
        "nominal_joint_position_rad": nominal_q,
        "center_jacobian_m_per_rad": center_jacobian,
        "exact_center_JVP_m_per_action": exact_center_jvp,
    }
    paths["preprocess"].parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(paths["preprocess"], **arrays_to_save)
    audit["artifact"] = {
        "path": str(paths["preprocess"]),
        "file_sha256": _file_sha256(paths["preprocess"]),
        "nominal_q_sha256": _hash_array(nominal_q),
        "center_jacobian_sha256": _hash_array(center_jacobian),
        "exact_center_JVP_sha256": _hash_array(exact_center_jvp),
    }
    return arrays_to_save, audit


def _direction_suite(
    *,
    traces: Mapping[str, Any],
    arrays: Mapping[str, Any],
    sensitivities: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    import numpy as np

    dimensions = config["training"]["steering_action_dimensions"]
    state_split = {
        int(state): str(split)
        for state, split in zip(
            np.asarray(arrays["state_index"], dtype=np.int64),
            np.asarray(arrays["split"], dtype=object),
        )
    }
    gradients = {}
    state_ids = None
    temperature = float(config["direction_score"]["soft_min_temperature_m"])
    for name, trace in traces.items():
        score = soft_min_trace(trace, temperature)
        states, gradient = action_gradient_from_pairs(
            score, sensitivities, arrays, dimensions
        )
        if state_ids is None:
            state_ids = states
        elif not np.array_equal(state_ids, states):
            raise ValueError("ExecGrad direction state identity differs")
        gradients[name] = gradient
    exact = gradients["exact_OSC"]
    return {
        "gradient_sha256": {
            name: _hash_array(value) for name, value in gradients.items()
        },
        "comparisons": {
            name: direction_metrics(
                exact_gradient=exact,
                predicted_gradient=gradient,
                state_ids=state_ids,
                state_splits=state_split,
                config=config,
            )
            for name, gradient in gradients.items()
            if name != "exact_OSC"
        },
    }


def main() -> int:
    import numpy as np
    import torch
    from main.multilink_ellipsoid.shadow import allocation_record

    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()
    started = time.perf_counter_ns()
    (
        paths,
        _,
        _,
        factorized_config,
        complete_dataset,
        complete_collection,
        arrays,
        sensitivities,
        normalization,
        representation,
        local_geometry,
    ) = prepare(args)
    paths.update(direct_paths(args))
    paths.update(extra_paths(args))
    config = load_execgrad_config(paths["execgrad_config"])
    validate_sources(paths, config)
    preprocess, apparatus = build_preprocess(
        paths=paths,
        config=config,
        factorized_config=factorized_config,
        complete_dataset=complete_dataset,
        complete_collection=complete_collection,
        arrays=arrays,
        sensitivities=sensitivities,
    )
    _require(apparatus["apparatus_gate_pass"], "ExecGrad preprocessing apparatus gate failed")
    torch.set_num_threads(8)
    models, model_state, training = train_matched_execgrad_models(
        arrays=arrays,
        sensitivities=sensitivities,
        local_geometry=local_geometry,
        nominal_q=preprocess["nominal_joint_position_rad"],
        center_jacobian=preprocess["center_jacobian_m_per_rad"],
        exact_center_jvp=preprocess["exact_center_JVP_m_per_action"],
        config=config,
        normalization=normalization,
    )
    predictions = predict_execgrad_models(
        models,
        model_state,
        arrays,
        preprocess["nominal_joint_position_rad"],
    )
    model_receipt = save_execgrad_weights(paths["model"], model_state)
    traces = {"exact_OSC": np.asarray(arrays["ellipsoid_clearance_m"], dtype=np.float64)}
    geometry_receipts = {}
    for name, predicted_q in {
        "kinematic_nominal": preprocess["nominal_joint_position_rad"],
        **predictions,
    }.items():
        evaluated = geometry_kwargs(
            paths,
            factorized_config,
            complete_dataset,
            complete_collection,
            arrays,
            predicted_q,
            ("train", "validation", "test"),
            return_trace=True,
        )
        traces[name] = evaluated.pop("predicted_clearance_trace_m")
        evaluated.pop("exact_q_static_clearance_trace_m")
        evaluated.pop("predicted_minimum_margin_m")
        evaluated.pop("exact_q_static_minimum_margin_m")
        geometry_receipts[name] = evaluated
    direct_archive = np.load(
        paths["normalized_direct_run"] / "predictions.npz", allow_pickle=False
    )
    direct_q = direct_archive["joint_position_rad"].astype(np.float64)
    direct_trace = direct_archive["predicted_clearance_trace_m"].astype(np.float64)
    traces["frozen_direct_normalized_joint_secant"] = direct_trace
    direction = _direction_suite(
        traces=traces,
        arrays=arrays,
        sensitivities=sensitivities,
        config=config,
    )
    exact_q = np.asarray(arrays["joint_position_rad"], dtype=np.float64)
    temporal = {
        name: split_temporal_metrics(q, exact_q, arrays["split"])
        for name, q in {
            "kinematic_nominal": preprocess["nominal_joint_position_rad"],
            "frozen_direct_normalized_joint_secant": direct_q,
            **predictions,
        }.items()
    }
    safety = {
        name: {
            split: safety_metrics(
                arrays["minimum_margin_m"],
                np.min(trace, axis=1),
                arrays,
                factorized_config,
                split_name=split,
                evaluation_only=True,
            )
            for split in ("train", "validation", "test")
        }
        for name, trace in traces.items()
        if name != "exact_OSC"
    }
    decision = execgrad_fitted_decision(
        trajectory_only=direction["comparisons"]["trajectory_only"],
        execgrad=direction["comparisons"]["trajectory_plus_link_JVP"],
        temporal=temporal,
        config=config,
    )
    paths["predictions"].parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        paths["predictions"],
        nominal_joint_position_rad=preprocess["nominal_joint_position_rad"],
        trajectory_only_joint_position_rad=predictions["trajectory_only"],
        execgrad_joint_position_rad=predictions["trajectory_plus_link_JVP"],
        trajectory_only_clearance_trace_m=traces["trajectory_only"],
        execgrad_clearance_trace_m=traces["trajectory_plus_link_JVP"],
    )
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "complete",
        "scientific_result": True,
        "claim_scope": config["claim_scope"],
        "source": _git_identity(paths["repo"], args.expected_commit),
        "allocation": allocation_record(),
        "config": config,
        "representation": representation,
        "apparatus": apparatus,
        "training": training,
        "model": model_receipt,
        "predictions": {
            "path": str(paths["predictions"]),
            "file_sha256": _file_sha256(paths["predictions"]),
            "joint_sha256": {
                name: _hash_array(value) for name, value in predictions.items()
            },
        },
        "geometry": geometry_receipts,
        "temporal_joint": temporal,
        "absolute_safety_diagnostic": safety,
        "direction": direction,
        "decision": decision,
        "future_untouched_episode_receipt": {
            "manifest_file_sha256": _file_sha256(paths["future_untouched_manifest"]),
            "opened_or_evaluated": False,
            "opening_authorized": bool(decision["fitted_direction_gate_pass"]),
        },
        "forbidden_action_receipt": {
            "new_unseen_episode_opened": False,
            "fresh_cloned_OSC_direction_rollouts_executed": False,
            "flow_guidance_executed": False,
            "calibration_executed": False,
            "QP_executed": False,
            "closed_loop_executed": False,
            "poisson_or_SDF_executed": False,
            "binary_classifier_executed": False,
        },
        "wall_seconds": (time.perf_counter_ns() - started) * 1.0e-9,
    }
    result["result_payload_sha256"] = payload_sha256(
        result, "result_payload_sha256"
    )
    _atomic_write(paths["output"], result)
    print(
        json.dumps(
            {
                "decision": decision,
                "validation_directions": {
                    name: value["validation"]
                    for name, value in direction["comparisons"].items()
                },
                "wall_seconds": result["wall_seconds"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
