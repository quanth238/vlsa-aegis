#!/usr/bin/env python3
"""Collect, train, and exactly audit the E05 execution-margin residual MLP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time
from typing import Any, Mapping, Optional, Sequence

from scripts.evaluate_distal_sitl_candidate_e05 import _disable_probe_images
from scripts.replay_distal_three_ellipsoid_multicbf import (
    ARCHIVED_FILE_SHA256,
    ARCHIVED_PAYLOAD_SHA256,
    CASE_ID,
    EXPECTED_ACTION_HORIZON,
    _atomic_write,
    _file_sha256,
    _git_identity,
    _load,
    _require,
    _sha256,
)


def _verify_sources(
    *,
    archived_path: Path,
    false_safe_result_path: Path,
    obstacle_discovery_result_path: Path,
    geometry_config_path: Path,
    exact_box_config_path: Path,
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    archived = _load(archived_path)
    false_safe = _load(false_safe_result_path)
    discovery = _load(obstacle_discovery_result_path)
    identities = config["immutable_sources"]
    _require(
        _file_sha256(archived_path) == identities["archived_table1_file_sha256"]
        == ARCHIVED_FILE_SHA256
        and archived.get("result_payload_sha256")
        == identities["archived_table1_payload_sha256"]
        == ARCHIVED_PAYLOAD_SHA256,
        "archived Table-1 identity differs",
    )
    _require(
        _file_sha256(false_safe_result_path)
        == identities["false_safe_action_ledger_file_sha256"]
        and false_safe.get("result_payload_sha256")
        == identities["false_safe_action_ledger_payload_sha256"],
        "false-safe action ledger identity differs",
    )
    _require(
        _file_sha256(obstacle_discovery_result_path)
        == identities["exact_box_discovery_file_sha256"]
        and discovery.get("result_payload_sha256")
        == identities["exact_box_discovery_payload_sha256"],
        "exact-box discovery identity differs",
    )
    _require(
        _file_sha256(geometry_config_path)
        == identities["geometry_config_file_sha256"],
        "distal geometry config identity differs",
    )
    _require(
        _file_sha256(exact_box_config_path)
        == identities["exact_box_config_file_sha256"],
        "exact-box config file identity differs",
    )
    actions = false_safe.get("actions")
    _require(
        false_safe.get("schema_version")
        == "vlsa_distal_sitl_candidate_e05_result.v1"
        and false_safe.get("status") == "complete"
        and false_safe.get("scientific_result") is True
        and false_safe.get("case_id") == CASE_ID
        and isinstance(actions, list)
        and len(actions) == EXPECTED_ACTION_HORIZON,
        "false-safe action ledger contract differs",
    )
    return archived, false_safe, discovery


def _build_pair(runtime: Any, case: Mapping[str, Any]) -> tuple[Any, Any, Any, Any, Any]:
    import numpy as np

    from main.evaluate_safelibero_aegis import (
        TABLE_RENDER_RESOLUTION,
        TABLE_SETTLE_ACTIONS,
        _active_obstacle,
        _build_environment,
        _settle,
    )

    env, task, observation, selected_initial_state = _build_environment(
        runtime, case, render_resolution=TABLE_RENDER_RESOLUTION
    )
    observation = _settle(env, observation, TABLE_SETTLE_ACTIONS)
    probe_env, probe_task, probe_observation, probe_initial_state = _build_environment(
        runtime, case, render_resolution=32
    )
    probe_observation = _settle(probe_env, probe_observation, TABLE_SETTLE_ACTIONS)
    _require(str(probe_task.language) == str(task.language), "probe task differs")
    _require(
        np.array_equal(np.asarray(probe_initial_state), np.asarray(selected_initial_state)),
        "probe initial state differs",
    )
    _require(
        np.array_equal(
            np.asarray(probe_env.sim.get_state().flatten()),
            np.asarray(env.sim.get_state().flatten()),
        ),
        "settled probe simulator state differs",
    )
    obstacle_name, _ = _active_obstacle(env, observation)
    probe_obstacle, _ = _active_obstacle(probe_env, probe_observation)
    _require(probe_obstacle == obstacle_name, "probe active obstacle differs")
    disabled = {
        "main": _disable_probe_images(env),
        "probe": _disable_probe_images(probe_env),
    }
    return env, probe_env, task, observation, {
        "selected_initial_state": selected_initial_state,
        "obstacle_name": obstacle_name,
        "disabled_images": disabled,
    }


def _source_action(source_action: Mapping[str, Any], step: int) -> Any:
    import numpy as np

    _require(int(source_action.get("step", -1)) == int(step), "source action step differs")
    action = np.asarray(source_action.get("executed_sitl_action"), dtype=np.float64)
    recorded = np.asarray(
        source_action.get("filter", {}).get("executed_action"), dtype=np.float64
    )
    _require(
        action.shape == (7,)
        and np.all(np.isfinite(action))
        and np.array_equal(action, recorded),
        "source executed action binding differs",
    )
    return action


def _geometry(
    *,
    geometry_config: Mapping[str, Any],
    exact_box_config: Mapping[str, Any],
    archived: Mapping[str, Any],
    env: Any,
    obstacle_name: str,
) -> tuple[Any, Any]:
    from main.multilink_ellipsoid.obstacle_primitives import ExactObstacleBoxUnion
    from main.multilink_ellipsoid.shadow import MultilinkEllipsoidShadow

    perception = archived["perception"]
    geometry = MultilinkEllipsoidShadow.from_aegis_geometry(
        geometry_config,
        {
            "p2": perception["mvee_center"],
            "R2": perception["mvee_rotation"],
            "Q2_diag": perception["mvee_semiaxes"],
            "record": {"label": perception["obstacle_label"]},
        },
    )
    record = geometry.geometry_record(env)
    _require(
        record["distal_ellipsoid_count"] == 7
        and record["total_constraint_geometry_count"] == 8,
        "accepted distal geometry count differs",
    )
    return geometry, ExactObstacleBoxUnion(exact_box_config, env, obstacle_name)


def _replay_prefix(env: Any, actions: Sequence[Mapping[str, Any]], stop: int) -> list[str]:
    hashes = []
    for step in range(int(stop)):
        action = _source_action(actions[step], step)
        env.step(action.tolist())
        hashes.append(_sha256(json.dumps(action.tolist(), separators=(",", ":")).encode("utf-8")))
    return hashes


def _collect_dataset(
    *,
    env: Any,
    probe_env: Any,
    geometry: Any,
    exact_boxes: Any,
    obstacle_name: str,
    actions: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    import numpy as np

    from main.multilink_ellipsoid.execution_margin_nn import feature_vector
    from main.multilink_ellipsoid.oracle_affine import (
        SubstepEightConstraintProbe,
        oracle_candidate_xyz,
    )
    from main.multilink_ellipsoid.rollout import _dynamic_state_vector

    groups = config["state_groups"]
    first = int(groups["collect_steps"][0])
    prefix_hashes = _replay_prefix(env, actions, first)
    probe = SubstepEightConstraintProbe(
        probe_env,
        geometry,
        active_obstacle_name=obstacle_name,
        quadratic_tolerance=1.0e-6,
        contact_distance_threshold_m=0.0,
        obstacle_primitive_union=exact_boxes,
    )
    candidate_config = {"candidate_set": dict(config["candidate_set"])}
    expected_candidate_counts = candidate_config["candidate_set"].pop(
        "expected_candidate_count_by_state"
    )
    records = []
    state_records = []
    for step in groups["collect_steps"]:
        nominal = _source_action(actions[step], step)
        _require(
            np.allclose(nominal[3:6], 0.0, rtol=0.0, atol=1.0e-12),
            "execution-margin audit requires translational AEGIS actions",
        )
        state_before = np.asarray(_dynamic_state_vector(env), dtype=np.float64)
        candidates = oracle_candidate_xyz(nominal[:3], candidate_config)
        candidate_count = int(expected_candidate_counts[str(step)])
        _require(len(candidates) == candidate_count, "candidate count differs")
        nominal_transition = None
        current_reference = None
        jointly_safe = []
        for candidate_index, candidate in enumerate(candidates):
            action = nominal.copy()
            action[:3] = candidate["xyz"]
            transition = probe.transition(env, action)
            start = transition["substeps"][0]
            current = np.asarray(start["clearance_m"][:7], dtype=np.float64)
            minimum = np.asarray(
                transition["minimum_substep_clearance_m"][:7], dtype=np.float64
            )
            if current_reference is None:
                current_reference = current
            else:
                _require(
                    np.array_equal(current, current_reference),
                    "candidate interval-start clearance differs",
                )
            raw_safe = bool(
                transition["raw_protected_contact_count"] == 0
                and transition["maximum_within_step_obstacle_l1_displacement_m"]
                <= 1.0e-4
            )
            proxy_safe = bool(np.all(minimum >= 0.0))
            inside_trust = bool(
                np.max(np.abs(np.asarray(candidate["xyz"]) - nominal[:3]))
                <= float(config["projection"]["trust_region_linf_action"]) + 1.0e-12
            )
            if raw_safe and proxy_safe and inside_trust:
                jointly_safe.append(candidate_index)
            records.append(
                {
                    "record_index": len(records),
                    "state_step": int(step),
                    "split": next(
                        name
                        for name, selected in (
                            ("train", groups["train_steps"]),
                            ("validation", groups["validation_steps"]),
                            ("test", groups["test_steps"]),
                        )
                        if step in selected
                    ),
                    "candidate_index": int(candidate_index),
                    "candidate_source": str(candidate["source"]),
                    "nominal_xyz": nominal[:3].tolist(),
                    "candidate_xyz": np.asarray(candidate["xyz"], dtype=np.float64).tolist(),
                    "feature_vector": feature_vector(
                        start, nominal[:3], candidate["xyz"]
                    ).tolist(),
                    "current_clearance_m": current.tolist(),
                    "minimum_substep_clearance_m": minimum.tolist(),
                    "endpoint_clearance_m": list(transition["endpoint_clearance_m"][:7]),
                    "D_opt_proxy_safe": proxy_safe,
                    "D_sim_raw_safe": raw_safe,
                    "raw_protected_contact_count": int(
                        transition["raw_protected_contact_count"]
                    ),
                    "geometry_consistency": dict(
                        transition["geometry_consistency"]
                    ),
                    "maximum_within_step_obstacle_l1_displacement_m": float(
                        transition["maximum_within_step_obstacle_l1_displacement_m"]
                    ),
                    "next_state_sha256": transition["next_state_sha256"],
                    "env_step_wall_seconds": float(transition["env_step_wall_seconds"]),
                }
            )
            if candidate["source"] == "nominal":
                nominal_transition = transition
        _require(nominal_transition is not None, "state lacks nominal transition")
        state_records.append(
            {
                "state_step": int(step),
                "dynamic_state_sha256": _sha256(state_before.tobytes()),
                "nominal_action": nominal.tolist(),
                "current_clearance_m": list(
                    nominal_transition["substeps"][0]["clearance_m"][:7]
                ),
                "nominal_minimum_substep_clearance_m": list(
                    nominal_transition["minimum_substep_clearance_m"][:7]
                ),
                "nominal_raw_protected_contact_count": int(
                    nominal_transition["raw_protected_contact_count"]
                ),
                "nominal_geometry_consistency": dict(
                    nominal_transition["geometry_consistency"]
                ),
                "nominal_maximum_within_step_obstacle_l1_displacement_m": float(
                    nominal_transition["maximum_within_step_obstacle_l1_displacement_m"]
                ),
                "jointly_raw_and_proxy_safe_candidate_indexes_inside_trust_region": jointly_safe,
                "nominal_substep_trace": nominal_transition["substeps"],
            }
        )
        env.step(nominal.tolist())
        actual_next = _sha256(
            np.asarray(_dynamic_state_vector(env), dtype=np.float64).tobytes()
        )
        _require(
            actual_next == nominal_transition["next_state_sha256"],
            "collected nominal clone does not match immutable replay execution",
        )
    representation_state = next(
        item
        for item in state_records
        if item["state_step"] == groups["primary_projection_step"]
    )
    geometry_authority_pass = bool(
        all(
            item["raw_protected_contact_count"] == 0
            or item["geometry_consistency"]["status"] == "passed"
            for item in records
        )
    )
    recoverable_crossings = [
        int(item["state_step"])
        for item in state_records
        if np.all(np.asarray(item["current_clearance_m"], dtype=np.float64) >= 0.0)
        and np.any(
            np.asarray(
                item["nominal_minimum_substep_clearance_m"], dtype=np.float64
            )
            < 0.0
        )
        and bool(
            item[
                "jointly_raw_and_proxy_safe_candidate_indexes_inside_trust_region"
            ]
        )
    ]
    summary = {
        "state_count": len(state_records),
        "candidate_count_by_state": {
            str(item["state_step"]): sum(
                record["state_step"] == item["state_step"] for record in records
            )
            for item in state_records
        },
        "sample_count": len(records),
        "immutable_prefix_action_count": first,
        "immutable_prefix_action_sha256": prefix_hashes,
        "primary_projection_step": groups["primary_projection_step"],
        "primary_local_jointly_safe_candidate_exists": bool(
            representation_state[
                "jointly_raw_and_proxy_safe_candidate_indexes_inside_trust_region"
            ]
        ),
        "primary_local_jointly_safe_candidate_indexes": representation_state[
            "jointly_raw_and_proxy_safe_candidate_indexes_inside_trust_region"
        ],
        "geometry_authority_pass": geometry_authority_pass,
        "recoverable_exact_proxy_crossing_steps": recoverable_crossings,
        "oracle_analysis_gate_pass": bool(
            geometry_authority_pass
            and recoverable_crossings
            and groups["primary_projection_step"] in recoverable_crossings
        ),
        "total_clone_env_step_wall_seconds": float(
            sum(item["env_step_wall_seconds"] for item in records)
        ),
    }
    return records, state_records, summary


def _projection_audit(
    *,
    runtime: Any,
    case: Mapping[str, Any],
    archived: Mapping[str, Any],
    actions: Sequence[Mapping[str, Any]],
    geometry_config: Mapping[str, Any],
    exact_box_config: Mapping[str, Any],
    config: Mapping[str, Any],
    model: Any,
    model_state: Mapping[str, Any],
) -> dict[str, Any]:
    import numpy as np

    from main.multilink_ellipsoid.execution_margin_nn import (
        predict_margin_and_jacobian,
        project_action_with_model,
    )
    from main.multilink_ellipsoid.oracle_affine import SubstepEightConstraintProbe

    env = probe_env = None
    try:
        env, probe_env, _, _, setup = _build_pair(runtime, case)
        geometry, exact_boxes = _geometry(
            geometry_config=geometry_config,
            exact_box_config=exact_box_config,
            archived=archived,
            env=env,
            obstacle_name=setup["obstacle_name"],
        )
        step = int(config["state_groups"]["primary_projection_step"])
        _replay_prefix(env, actions, step)
        probe = SubstepEightConstraintProbe(
            probe_env,
            geometry,
            active_obstacle_name=setup["obstacle_name"],
            quadratic_tolerance=1.0e-6,
            contact_distance_threshold_m=0.0,
            obstacle_primitive_union=exact_boxes,
        )
        nominal = _source_action(actions[step], step)
        nominal_transition = probe.transition(env, nominal)
        start = nominal_transition["substeps"][0]
        nominal_margin, nominal_jacobian, nominal_inference = predict_margin_and_jacobian(
            model, model_state, start, nominal[:3], nominal[:3]
        )
        calibration = np.asarray(model_state["calibration_m"], dtype=np.float64)
        nominal_lower = nominal_margin - calibration
        activated = bool(
            np.min(nominal_lower)
            < float(config["projection"]["activation_warning_m"])
        )
        projection = project_action_with_model(
            model, model_state, start, nominal[:3], config
        )
        exact = None
        exact_raw_safe = False
        exact_proxy_safe = False
        if projection["valid"]:
            proposed = nominal.copy()
            proposed[:3] = np.asarray(projection["projected_xyz"], dtype=np.float64)
            exact = probe.transition(env, proposed)
            exact_raw_safe = bool(
                exact["raw_protected_contact_count"] == 0
                and exact["maximum_within_step_obstacle_l1_displacement_m"] <= 1.0e-4
            )
            exact_proxy_safe = bool(
                np.all(
                    np.asarray(exact["minimum_substep_clearance_m"][:7], dtype=np.float64)
                    >= float(config["projection"]["clearance_target_m"])
                )
            )
        return {
            "state_step": step,
            "activation_warning_m": float(config["projection"]["activation_warning_m"]),
            "activated": activated,
            "nominal_action": nominal.tolist(),
            "nominal_predicted_margin_m": nominal_margin.tolist(),
            "nominal_conservative_margin_m": nominal_lower.tolist(),
            "nominal_jacobian_m_per_action": nominal_jacobian.tolist(),
            "nominal_inference_and_jacobian_wall_seconds": nominal_inference,
            "nominal_exact_transition": nominal_transition,
            "neural_projection": projection,
            "projected_exact_transition": exact,
            "projected_exact_raw_safe": exact_raw_safe,
            "projected_exact_proxy_safe": exact_proxy_safe,
            "projection_gate_pass": bool(
                activated
                and projection["valid"]
                and exact_raw_safe
                and exact_proxy_safe
            ),
        }
    finally:
        if probe_env is not None:
            probe_env.close()
        if env is not None:
            env.close()


def evaluate(
    *,
    repo_root: Path,
    manifest_path: Path,
    archived_path: Path,
    false_safe_result_path: Path,
    obstacle_discovery_result_path: Path,
    geometry_config_path: Path,
    exact_box_config_path: Path,
    config_path: Path,
    expected_commit: str,
    openpi_python: Path,
    dataset_path: Path,
    model_path: Path,
    training_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    import numpy as np

    from main.evaluate_safelibero_aegis import (
        _runtime_imports,
        pairing_record,
        read_jsonl,
        validate_case_row,
    )
    from main.multilink_ellipsoid.execution_margin_nn import (
        EXECUTION_MARGIN_RESULT_SCHEMA,
        load_model_artifact,
        load_execution_margin_config,
    )
    from main.multilink_ellipsoid.obstacle_primitives import (
        EXACT_BOX_OBSTACLE_SCHEMA,
        load_obstacle_primitive_config,
    )
    from main.multilink_ellipsoid.shadow import allocation_record, load_shadow_config

    started = time.perf_counter_ns()
    config = load_execution_margin_config(config_path)
    geometry_config = load_shadow_config(geometry_config_path)
    exact_box_config = load_obstacle_primitive_config(exact_box_config_path)
    _require(
        exact_box_config["schema_version"] == EXACT_BOX_OBSTACLE_SCHEMA
        and exact_box_config["config_payload_sha256"]
        == config["immutable_sources"]["exact_box_config_payload_sha256"],
        "exact-box config payload identity differs",
    )
    archived, false_safe, discovery = _verify_sources(
        archived_path=archived_path,
        false_safe_result_path=false_safe_result_path,
        obstacle_discovery_result_path=obstacle_discovery_result_path,
        geometry_config_path=geometry_config_path,
        exact_box_config_path=exact_box_config_path,
        config=config,
    )
    rows = read_jsonl(manifest_path)
    matches = [row for row in rows if row.get("case_id") == CASE_ID]
    _require(len(matches) == 1, "primary execution-margin manifest row is not unique")
    case = matches[0]
    validate_case_row(case, repo_root)
    source = _git_identity(repo_root, expected_commit)
    allocation = allocation_record()
    runtime = _runtime_imports(include_aegis=False)
    actions = false_safe["actions"]
    env = probe_env = None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        env, probe_env, task, observation, setup = _build_pair(runtime, case)
        _require(
            setup["obstacle_name"] == archived["obstacle"]["active_name"],
            "active obstacle differs from Table 1",
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
            _require(pairing[key] == archived["pairing"][key], "pairing field differs: %s" % key)
        geometry, exact_boxes = _geometry(
            geometry_config=geometry_config,
            exact_box_config=exact_box_config,
            archived=archived,
            env=env,
            obstacle_name=setup["obstacle_name"],
        )
        geometry_record = geometry.geometry_record(env)
        exact_box_record = exact_boxes.geometry_record(env)
        records, state_records, dataset_summary = _collect_dataset(
            env=env,
            probe_env=probe_env,
            geometry=geometry,
            exact_boxes=exact_boxes,
            obstacle_name=setup["obstacle_name"],
            actions=actions,
            config=config,
        )
    finally:
        if probe_env is not None:
            probe_env.close()
        if env is not None:
            env.close()

    dataset = {
        "schema_version": "vlsa_distal_execution_margin_nn_e05_dataset.v1",
        "case_id": CASE_ID,
        "config_file_sha256": config["config_file_sha256"],
        "config_payload_sha256": config["config_payload_sha256"],
        "source_commit": source["commit"],
        "constraint_order": [
            "L5_part_0",
            "L5_part_1",
            "L5_part_2",
            "L6_part_0",
            "L6_part_1",
            "L7_part_0",
            "L7_part_1",
        ],
        "summary": dataset_summary,
        "states": state_records,
        "records": records,
    }
    dataset["dataset_payload_sha256"] = _sha256(
        json.dumps(dataset, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    )
    _atomic_write(dataset_path, dataset)
    dataset_identity = {
        "path": str(dataset_path),
        "file_sha256": _file_sha256(dataset_path),
        "payload_sha256": dataset["dataset_payload_sha256"],
        "sample_count": len(records),
    }
    _require(
        dataset_summary["oracle_analysis_gate_pass"] is True,
        "pretraining oracle-analysis gate did not authorize neural training",
    )
    _require(
        openpi_python.is_file() and not openpi_python.is_symlink(),
        "OpenPI training Python is unavailable",
    )
    subprocess.run(
        [
            str(openpi_python),
            str(repo_root / "scripts/train_distal_execution_margin_nn_e05.py"),
            "--repo-root",
            str(repo_root),
            "--config",
            str(config_path),
            "--dataset",
            str(dataset_path),
            "--expected-commit",
            str(expected_commit),
            "--model",
            str(model_path),
            "--output",
            str(training_path),
        ],
        check=True,
        cwd=str(repo_root),
    )
    training_record = _load(training_path)
    _require(
        training_record.get("schema_version")
        == "vlsa_distal_execution_margin_nn_e05_training.v1"
        and training_record.get("status") == "complete"
        and training_record.get("source", {}).get("commit") == expected_commit
        and training_record.get("dataset") == dataset_identity,
        "execution-margin training artifact differs",
    )
    model_identity = training_record["model_artifact"]
    _require(
        model_identity.get("file_sha256") == _file_sha256(model_path),
        "execution-margin model identity differs after training",
    )
    training_audit = training_record["training"]
    model, model_state = load_model_artifact(model_path, device="cpu")
    projection = _projection_audit(
        runtime=runtime,
        case=case,
        archived=archived,
        actions=actions,
        geometry_config=geometry_config,
        exact_box_config=exact_box_config,
        config=config,
        model=model,
        model_state=model_state,
    )
    model_gate = bool(training_audit["model_learnability_gate_pass"])
    representation_gate = bool(dataset_summary["primary_local_jointly_safe_candidate_exists"])
    projection_gate = bool(projection["projection_gate_pass"])
    research_direction_go = bool(model_gate and representation_gate and projection_gate)
    if not representation_gate:
        stop_reason = "no_exact_proxy_and_raw_safe_local_action_at_primary_state"
    elif not model_gate:
        stop_reason = "residual_model_failed_held_out_conservative_learnability_gate"
    elif not projection_gate:
        stop_reason = "neural_projection_failed_exact_cloned_OSC_verification"
    else:
        stop_reason = None
    result = {
        "schema_version": EXECUTION_MARGIN_RESULT_SCHEMA,
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
            "discovery_schema_version": discovery.get("schema_version"),
            "read_only": True,
        },
        "probe_environment": {
            "osc_controller": "OSC_POSE",
            "control_frequency_hz": 20,
            "internal_substep_capture": "interval_start_and_every_internal_MuJoCo_step",
            "disabled_image_observable_count": setup["disabled_images"],
        },
        "geometry": {
            "accepted_robot_and_released_ee": geometry_record,
            "exact_obstacle_boxes": exact_box_record,
        },
        "dataset": dataset_identity,
        "dataset_summary": dataset_summary,
        "model_artifact": model_identity,
        "training_artifact": {
            "path": str(training_path),
            "file_sha256": _file_sha256(training_path),
            "payload_sha256": training_record["training_payload_sha256"],
        },
        "training": training_audit,
        "primary_projection": projection,
        "decision": {
            "model_learnability_gate_pass": model_gate,
            "representation_gate_pass": representation_gate,
            "projection_gate_pass": projection_gate,
            "research_direction_go": research_direction_go,
            "stop_reason": stop_reason,
        },
        "research_direction_go": research_direction_go,
        "stop_reason": stop_reason,
        "wall_seconds": (time.perf_counter_ns() - started) * 1.0e-9,
        "failure": None,
    }
    result["result_payload_sha256"] = _sha256(
        json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    )
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--archived", type=Path, required=True)
    parser.add_argument("--false-safe-result", type=Path, required=True)
    parser.add_argument("--obstacle-discovery-result", type=Path, required=True)
    parser.add_argument("--geometry-config", type=Path, required=True)
    parser.add_argument("--exact-box-config", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--openpi-python", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = evaluate(
        repo_root=args.repo_root.resolve(),
        manifest_path=args.manifest.resolve(),
        archived_path=args.archived.resolve(),
        false_safe_result_path=args.false_safe_result.resolve(),
        obstacle_discovery_result_path=args.obstacle_discovery_result.resolve(),
        geometry_config_path=args.geometry_config.resolve(),
        exact_box_config_path=args.exact_box_config.resolve(),
        config_path=args.config.resolve(),
        expected_commit=args.expected_commit,
        openpi_python=args.openpi_python.resolve(),
        dataset_path=args.dataset.resolve(),
        model_path=args.model.resolve(),
        training_path=args.training.resolve(),
        output_path=args.output.resolve(),
    )
    _atomic_write(args.output.resolve(), result)
    print(json.dumps(result["decision"], sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
