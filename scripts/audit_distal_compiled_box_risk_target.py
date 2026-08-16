#!/usr/bin/env python3
"""Replay fixed L6 candidates and audit a compiled-box future-risk target."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from scripts.evaluate_distal_query_action_risk_e05 import _allocation_record
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write,
    _file_sha256,
    _git_identity,
    _load,
    _require,
    _sha256,
)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


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


def _candidate_source_car(candidate: Mapping[str, Any]) -> float:
    values = [float(candidate["prefix"]["maximum_active_obstacle_l1_displacement_m"])]
    values.append(float(candidate["backup"]["maximum_active_obstacle_l1_displacement_m"] or 0.0))
    terminal = candidate["backup"].get("terminal_hold")
    if terminal is not None:
        values.append(float(terminal["maximum_active_obstacle_l1_displacement_m"]))
    return max(values)


def _candidate_source_contacts(candidate: Mapping[str, Any]) -> int:
    return int(candidate["prefix"]["protected_contact_count"]) + int(
        candidate["backup"]["protected_contact_count"]
    )


def _evaluate_case(
    *, repo_root: Path, population_manifest: Path, geometry_config_path: Path,
    case_config: Mapping[str, Any], audit_config: Mapping[str, Any],
) -> dict[str, Any]:
    import numpy as np

    from main.multilink_ellipsoid.pncbf_policy_value import (
        exact_group_action_boundary_values,
    )

    from main.evaluate_safelibero_aegis import (
        TABLE_SETTLE_ACTIONS,
        _active_obstacle,
        _build_environment,
        _contact_model_authority,
        _detailed_active_obstacle_contacts,
        _eef_site_id,
        _runtime_imports,
        _settle,
        read_jsonl,
        validate_case_row,
    )
    from main.multilink_ellipsoid.compiled_box_risk_target_audit import (
        candidate_action_sequence,
    )
    from main.multilink_ellipsoid.obstacle_proxy_audit import (
        compiled_obstacle_boxes,
        evaluate_obstacle_representations,
        minimum_ellipsoid_quadratics_over_boxes,
    )
    from main.multilink_ellipsoid.palm_primitive_audit import (
        fit_compiled_mesh_geom,
        world_ellipsoid,
    )
    from main.multilink_ellipsoid.geometry import Ellipsoid
    from main.multilink_ellipsoid.l6_proxy_scale_audit import rescale_row_slacks
    from main.multilink_ellipsoid.rollout import (
        _auxiliary_sim_snapshot,
        _base_env,
        _controller_snapshot,
        _dynamic_state_vector,
        _restore_auxiliary_sim_snapshot,
        _restore_controller_snapshot,
    )
    from main.multilink_ellipsoid.shadow import (
        MultilinkEllipsoidShadow,
        load_shadow_config,
    )
    from main.multilink_ellipsoid.sitl_candidate import (
        SlabbedEightConstraintProbe,
        _obstacle_root_body_id,
        _protected_contact_evidence,
    )

    source_path = Path(case_config["source_result"])
    _require(_file_sha256(source_path) == case_config["source_result_file_sha256"],
             "compiled-box source result file differs")
    source = _load(source_path)
    _require(source["result_payload_sha256"] == case_config["source_result_payload_sha256"],
             "compiled-box source result payload differs")
    _require(source["population_binding"]["selection"]["case_id"] == case_config["case_id"],
             "compiled-box source case differs")
    _require(int(source["state"]["step"]) == int(case_config["state_step"]),
             "compiled-box source state differs")
    archived_path = Path(source["archived_table1"]["path"])
    _require(_file_sha256(archived_path) == source["archived_table1"]["file_sha256"],
             "compiled-box archived ledger differs")
    archived = _load(archived_path)
    _require(archived["result_payload_sha256"] == source["archived_table1"]["result_payload_sha256"],
             "compiled-box archived payload differs")

    rows = [
        row for row in read_jsonl(population_manifest)
        if row.get("case_id") == case_config["case_id"]
    ]
    _require(len(rows) == 1, "compiled-box population case differs")
    case = dict(rows[0])
    validate_case_row(case, repo_root)
    runtime = _runtime_imports(include_aegis=True)
    geometry_config = load_shadow_config(geometry_config_path)
    env = None
    try:
        env, _, observation, _ = _build_environment(
            runtime, case, render_resolution=32
        )
        observation = _settle(env, observation, TABLE_SETTLE_ACTIONS)
        obstacle_name, _ = _active_obstacle(env, observation)
        _require(
            obstacle_name == source["population_binding"]["selection"]["active_obstacle_name"],
            "compiled-box active obstacle differs",
        )
        obstacle_reference = np.asarray(
            observation[obstacle_name + "_pos"], dtype=np.float64
        ).copy()
        perception = archived["perception"]
        from scripts.audit_distal_palm_primitive_case import (
            _canonicalize_perception_ellipsoid_rotation,
        )
        perception_rotation, perception_rotation_record = (
            _canonicalize_perception_ellipsoid_rotation(
                perception["mvee_rotation"]
            )
        )
        geometry = MultilinkEllipsoidShadow.from_aegis_geometry(
            geometry_config,
            {
                "p2": perception["mvee_center"],
                "R2": perception_rotation,
                "Q2_diag": perception["mvee_semiaxes"],
                "record": {"label": perception["obstacle_label"]},
            },
        )
        probe = SlabbedEightConstraintProbe(
            env, geometry, clearance_m=0.0, active_obstacle_name=obstacle_name
        )
        if case_config["slab_initialization"] == "settled_initial_state_matching_source":
            geometry._slabbed_links(env, include_certificates=True)
        elif case_config["slab_initialization"] != "query_state_matching_source":
            raise ValueError("compiled-box slab initialization differs")

        exact_target = audit_config.get("exact_group_target")
        exact_shadow = None
        exact_palm_template = None
        exact_group_order: list[str] = []
        exact_group_rows: dict[str, list[int]] = {}
        exact_geom_to_group: dict[str, str] = {}
        exact_certificate_pass = None
        exact_include_released_ee = False
        exact_distal_row_count = 5
        contact_authority = None
        if exact_target is not None:
            exact_geometry_path = repo_root / str(
                exact_target["robot_geometry_config"]
            )
            exact_geometry_config = load_shadow_config(exact_geometry_path)
            exact_shadow = MultilinkEllipsoidShadow.from_aegis_geometry(
                exact_geometry_config,
                {
                    "p2": perception["mvee_center"],
                    "R2": perception_rotation,
                    "Q2_diag": perception["mvee_semiaxes"],
                    "record": {"label": perception["obstacle_label"]},
                },
            )
            palm_fit = exact_target["palm_fit"]
            exact_palm_template = fit_compiled_mesh_geom(
                env,
                str(exact_target["palm_geom_name"]),
                relative_padding=float(palm_fit["relative_padding"]),
                tolerance=float(palm_fit["khachiyan_tolerance"]),
                max_iterations=int(palm_fit["khachiyan_max_iterations"]),
            )
            exact_include_released_ee = bool(
                exact_target.get("include_released_aegis_end_effector_proxy", False)
            )
            exact_distal_row_count = int(exact_target.get("distal_row_count", 5))
            exact_distal = exact_shadow._slabbed_links(
                env, include_certificates=True
            )[:exact_distal_row_count]
            exact_certificate_pass = bool(
                (exact_palm_template.enclosure_certificate or {}).get("verified")
                and all(
                    bool((row.enclosure_certificate or {}).get("verified"))
                    for row in exact_distal
                )
            )
            exact_group_order = list(exact_target["group_order"])
            exact_group_rows = {
                str(group): [int(index) for index in indices]
                for group, indices in exact_target["robot_rows"].items()
            }
            exact_geom_to_group = {
                str(geom_name): str(group)
                for group, geom_names in exact_target["groups"].items()
                for geom_name in geom_names
            }
            compiled_obstacle_boxes(env, obstacle_name)
            contact_authority = _contact_model_authority(env, obstacle_name)

        action_rows = {int(row["step"]): row for row in archived["actions"]}
        state_step = int(case_config["state_step"])
        _require(set(range(state_step)).issubset(action_rows),
                 "compiled-box action history differs")
        for step in range(state_step):
            observation, _, done, _ = env.step(action_rows[step]["executed"])
            _require(not done, "compiled-box episode completed before audit state")

        source_dynamic = np.asarray(_dynamic_state_vector(env), dtype=np.float64).copy()
        source_hash = hashlib.sha256(source_dynamic.tobytes()).hexdigest()
        current = np.asarray(probe.clearances(env)[:7], dtype=np.float64)
        initial_clearance_error = float(np.max(np.abs(
            current - np.asarray(source["state"]["initial_clearance_m"], dtype=np.float64)
        )))
        state_hash_matches = source_hash == source["state"]["source_snapshot_sha256"]
        base = _base_env(env)
        simulator_state = np.asarray(env.sim.get_state().flatten(), dtype=np.float64).copy()
        auxiliary = _auxiliary_sim_snapshot(env)
        controllers = _controller_snapshot(env)
        clock = (int(base.timestep), float(base.cur_time), bool(base.done))

        empirical_proxy = audit_config.get("empirical_l6_proxy")
        trajectory_value_config = audit_config.get("trajectory_policy_value")
        artifact_superset = audit_config.get("artifact_superset") or {}
        capture_action_boundaries = bool(
            (
                trajectory_value_config
                and trajectory_value_config.get("capture_action_boundaries") is True
            )
            or artifact_superset.get("capture_action_boundary_context") is True
        )

        def empirical_slacks(values: Sequence[float]) -> list[float]:
            if empirical_proxy is None:
                return [float(item) for item in values]
            return rescale_row_slacks(
                values,
                scale=float(empirical_proxy["uniform_semiaxis_scale"]),
                scaled_rows=empirical_proxy["scaled_rows"],
            )

        def released_ee_row() -> tuple[Any, dict[str, Any]]:
            """Build the unchanged released AEGIS EE proxy from live MuJoCo state."""

            site_id = int(_eef_site_id(env))
            site_position = np.asarray(
                env.sim.data.site_xpos[site_id], dtype=np.float64
            ).copy()
            eef_body_name = str(env.robots[0].robot_model.eef_name)
            quaternion_wxyz = np.asarray(
                env.sim.data.get_body_xquat(eef_body_name), dtype=np.float64
            )
            quaternion_xyzw = quaternion_wxyz[[1, 2, 3, 0]]
            rotation = runtime["Rotation"].from_quat(quaternion_xyzw).as_matrix()
            center = site_position + rotation @ np.asarray(
                [0.0, 0.0, -0.08], dtype=np.float64
            )
            row = Ellipsoid(
                center=center,
                rotation=rotation,
                semiaxes_m=np.asarray([0.06, 0.12, 0.11], dtype=np.float64),
                body_name="robot0_end_effector",
                geom_name="released_aegis_end_effector_proxy",
                bound_source="released_aegis_end_effector_proxy",
                source_body_names=(eef_body_name,),
            )
            return row, {
                "eef_site_position_m": site_position.tolist(),
                "eef_body_quaternion_xyzw": quaternion_xyzw.tolist(),
                "ee_proxy_center_m": center.tolist(),
                "ee_proxy_rotation": rotation.tolist(),
                "ee_proxy_semiaxes_m": [0.06, 0.12, 0.11],
            }

        def exact_robot_rows() -> tuple[list[Any], Optional[dict[str, Any]]]:
            palm = world_ellipsoid(env, exact_palm_template)
            distal = exact_shadow._slabbed_links(env)[:exact_distal_row_count]
            rows = [palm] + distal
            ee_pose = None
            if exact_include_released_ee:
                ee, ee_pose = released_ee_row()
                rows = [ee] + rows
            return rows, ee_pose

        def robot_row_record(row: Any) -> dict[str, Any]:
            return {
                "body_name": str(row.body_name),
                "geom_name": str(row.geom_name),
                "bound_source": str(row.bound_source),
                "center_m": np.asarray(row.center, dtype=np.float64).tolist(),
                "rotation": np.asarray(row.rotation, dtype=np.float64).tolist(),
                "semiaxes_m": np.asarray(row.semiaxes_m, dtype=np.float64).tolist(),
            }

        def exact_group_measurement(step: int) -> Optional[dict[str, Any]]:
            if exact_target is None:
                return None
            robot_rows, ee_pose = exact_robot_rows()
            boxes = compiled_obstacle_boxes(env, obstacle_name)
            pair_quadratics = minimum_ellipsoid_quadratics_over_boxes(
                robot_rows, boxes,
            )
            row_slack = np.sqrt(np.min(pair_quadratics, axis=1)) - 1.0
            group_slack = {
                group: float(min(row_slack[index] for index in indices))
                for group, indices in exact_group_rows.items()
            }
            contacts = _detailed_active_obstacle_contacts(
                env,
                obstacle_name,
                step=int(step),
                contact_authority=contact_authority,
            )
            _require(
                contacts["status"] == "available",
                "exact-group contact evidence unavailable",
            )
            group_events = {group: [] for group in exact_group_order}
            for event in contacts["events"]:
                if event.get("other", {}).get("classification") != "robot":
                    continue
                group = exact_geom_to_group.get(
                    str(event.get("other", {}).get("geom_name"))
                )
                if group is not None:
                    group_events[group].append(event)
            return {
                "row_normalized_radial_slack": row_slack.tolist(),
                "group_normalized_radial_slack": group_slack,
                "group_contact_events": group_events,
                "compiled_box_count": len(boxes),
                "ee_pose": ee_pose,
                "palm_pose": {
                    "center_m": np.asarray(
                        robot_rows[1 if exact_include_released_ee else 0].center,
                        dtype=np.float64,
                    ).tolist(),
                    "rotation": np.asarray(
                        robot_rows[1 if exact_include_released_ee else 0].rotation,
                        dtype=np.float64,
                    ).tolist(),
                },
            }

        initial_links = geometry._slabbed_links(env)
        initial_boxes = compiled_obstacle_boxes(env, obstacle_name)
        initial_representation = evaluate_obstacle_representations(
            initial_links,
            geometry.obstacle,
            initial_boxes,
            overlap_tolerance=float(audit_config["gate"]["compiled_overlap_tolerance"]),
        )
        initial_slacks = empirical_slacks(
            initial_representation[
                "compiled_box_union_row_minimum_normalized_radial_slack"
            ]
        )
        initial_contacts = _protected_contact_evidence(env, obstacle_name)
        initial_exact_group = exact_group_measurement(state_step)
        initial_exact_robot_rows = None
        initial_ee_pose = None
        if exact_target is not None:
            initial_rows, initial_ee_pose = exact_robot_rows()
            initial_exact_robot_rows = [robot_row_record(row) for row in initial_rows]

        def restore_source() -> None:
            env.sim.set_state_from_flattened(simulator_state)
            env.sim.forward()
            _restore_auxiliary_sim_snapshot(env, auxiliary)
            _restore_controller_snapshot(env, controllers)
            base.timestep, base.cur_time, base.done = clock
            _require(np.array_equal(_dynamic_state_vector(env), source_dynamic),
                     "compiled-box source restore differs")

        obstacle_id = _obstacle_root_body_id(env.sim.model, obstacle_name)
        expected_substeps = int(
            source["base_method_config"]["state"]["expected_mujoco_substeps_per_action"]
        )
        overlap_tolerance = float(audit_config["gate"]["compiled_overlap_tolerance"])
        car_limit = float(audit_config["gate"]["paper_car_threshold_m"])
        by_name = {candidate["name"]: candidate for candidate in source["candidates"]}
        _require(set(case_config["candidate_names"]).issubset(by_name),
                 "compiled-box candidate names differ")
        candidate_records = []
        for candidate_name in case_config["candidate_names"]:
            candidate = by_name[candidate_name]
            actions, phases = candidate_action_sequence(candidate)
            restore_source()
            observation = base._get_observations()
            proxy_trace = []
            exact_slack_trace = []
            exact_overlap_trace = []
            contacts = []
            displacements = []
            sample_phases = []
            exact_group_trace = []
            exact_group_contacts = {group: [] for group in exact_group_order}
            action_boundaries = []

            def measure(phase: str, action_offset: int, substep: int) -> None:
                links = geometry._slabbed_links(env)
                boxes = compiled_obstacle_boxes(env, obstacle_name)
                audit = evaluate_obstacle_representations(
                    links, geometry.obstacle, boxes,
                    overlap_tolerance=overlap_tolerance,
                )
                proxy_trace.append(np.asarray(
                    audit["perceived_mvee_row_clearance_m"], dtype=np.float64
                ))
                scaled_slacks = np.asarray(
                    empirical_slacks(
                        audit[
                            "compiled_box_union_row_minimum_normalized_radial_slack"
                        ]
                    ),
                    dtype=np.float64,
                )
                exact_slack_trace.append(scaled_slacks)
                exact_overlap_trace.append(
                    np.asarray(
                        audit["compiled_box_union_row_any_exact_solid_overlap"],
                        dtype=bool,
                    )
                    if empirical_proxy is None
                    else scaled_slacks <= 0.0
                )
                evidence = _protected_contact_evidence(env, obstacle_name)
                for event in evidence["events"]:
                    contacts.append({
                        "phase": phase,
                        "action_offset": int(action_offset),
                        "substep": int(substep),
                        **event,
                    })
                exact_sample = exact_group_measurement(
                    state_step + int(action_offset)
                )
                if exact_sample is not None:
                    trace_record = {
                        "phase": phase,
                        "action_offset": int(action_offset),
                        "substep": int(substep),
                        "row_normalized_radial_slack": exact_sample[
                            "row_normalized_radial_slack"
                        ],
                        "group_normalized_radial_slack": exact_sample[
                            "group_normalized_radial_slack"
                        ],
                        "compiled_box_count": exact_sample["compiled_box_count"],
                    }
                    if artifact_superset.get("capture_internal_substep_ee_pose") is True:
                        trace_record["ee_pose"] = exact_sample["ee_pose"]
                    if artifact_superset.get("capture_internal_substep_palm_pose") is True:
                        trace_record["palm_pose"] = exact_sample["palm_pose"]
                    exact_group_trace.append(trace_record)
                    for group, events in exact_sample[
                        "group_contact_events"
                    ].items():
                        for event in events:
                            exact_group_contacts[group].append({
                                "phase": phase,
                                "action_offset": int(action_offset),
                                "substep": int(substep),
                                **event,
                            })
                obstacle = np.asarray(env.sim.data.xpos[obstacle_id], dtype=np.float64)
                displacements.append(float(np.sum(np.abs(obstacle - obstacle_reference))))
                sample_phases.append(phase)

            for action_offset, (action, phase) in enumerate(zip(actions, phases)):
                if capture_action_boundaries:
                    _require(exact_target is not None, "trajectory-value exact target is absent")
                    current_exact = exact_group_measurement(
                        state_step + int(action_offset)
                    )
                    robot = env.robots[0]
                    qpos_indexes = getattr(robot, "_ref_joint_pos_indexes", None)
                    qvel_indexes = getattr(robot, "_ref_joint_vel_indexes", None)
                    _require(
                        qpos_indexes is not None and len(qpos_indexes) == 7,
                        "trajectory-value arm qpos indexes differ",
                    )
                    _require(
                        qvel_indexes is not None and len(qvel_indexes) == 7,
                        "trajectory-value arm qvel indexes differ",
                    )
                    boundary_robot_rows, boundary_ee_pose = exact_robot_rows()
                    action_public = np.asarray(action, dtype=np.float64).tolist()
                    action_boundaries.append({
                        "action_offset": int(action_offset),
                        "state_step": state_step + int(action_offset),
                        "phase": str(phase),
                        "executed_action": action_public,
                        "executed_action_sha256": hashlib.sha256(
                            np.asarray(action, dtype=np.float64).tobytes()
                        ).hexdigest(),
                        "dynamic_state": np.asarray(
                            _dynamic_state_vector(env), dtype=np.float64
                        ).tolist(),
                        "arm_joint_position_rad": np.asarray(
                            env.sim.data.qpos[list(qpos_indexes)], dtype=np.float64
                        ).tolist(),
                        "arm_joint_velocity_rad_s": np.asarray(
                            env.sim.data.qvel[list(qvel_indexes)], dtype=np.float64
                        ).tolist(),
                        "eef_position_m": np.asarray(
                            observation["robot0_eef_pos"], dtype=np.float64
                        ).tolist(),
                        "eef_quaternion_xyzw": np.asarray(
                            observation["robot0_eef_quat"], dtype=np.float64
                        ).tolist(),
                        "controller_snapshot": _public(_controller_snapshot(env)[0]),
                        "exact_robot_rows": [
                            robot_row_record(row) for row in boundary_robot_rows
                        ],
                        "ee_pose": boundary_ee_pose,
                        "compiled_obstacle_boxes": [
                            box.to_record()
                            for box in compiled_obstacle_boxes(env, obstacle_name)
                        ],
                        "row_normalized_radial_slack": current_exact[
                            "row_normalized_radial_slack"
                        ],
                        "group_normalized_radial_slack": current_exact[
                            "group_normalized_radial_slack"
                        ],
                    })
                count_before = len(proxy_trace)
                original_step = env.sim.step

                def instrumented_step(*args: Any, **kwargs: Any) -> Any:
                    value = original_step(*args, **kwargs)
                    measure(phase, action_offset, len(proxy_trace) - count_before)
                    return value

                env.sim.step = instrumented_step
                try:
                    observation, _, _, _ = env.step(action)
                finally:
                    env.sim.step = original_step
                _require(
                    len(proxy_trace) - count_before == expected_substeps,
                    "compiled-box internal substep count differs",
                )

            proxy = np.asarray(proxy_trace, dtype=np.float64)
            slack = np.asarray(exact_slack_trace, dtype=np.float64)
            overlap = np.asarray(exact_overlap_trace, dtype=bool)
            replayed_row_minimum = np.min(proxy, axis=0)
            source_row_minimum = np.asarray(
                candidate["combined_row_minimum_clearance_m"], dtype=np.float64
            )
            row_error = float(np.max(np.abs(replayed_row_minimum - source_row_minimum)))
            replay_car = float(max(displacements))
            source_car = _candidate_source_car(candidate)
            source_contacts = _candidate_source_contacts(candidate)
            replay_physical_veto = bool(contacts or replay_car > car_limit)
            source_proxy_nonoverlap = bool(float(np.min(source_row_minimum)) >= 0.0)
            source_proxy_buffer_safe = bool(max(candidate["combined_risk"]) <= 0.0)
            any_exact_overlap = bool(np.any(overlap))
            compiled_safe_terminal = bool(
                candidate["terminal_status"] == "SAFE_TERMINAL"
                and not any_exact_overlap
                and not contacts
                and replay_car <= car_limit
            )
            exact_group_record = None
            if exact_target is not None:
                _require(
                    len(exact_group_trace) == len(proxy_trace),
                    "exact-group trace length differs",
                )
                group_minimum = {
                    group: min(
                        float(sample["group_normalized_radial_slack"][group])
                        for sample in exact_group_trace
                    )
                    for group in exact_group_order
                }
                phase_minimum = {
                    phase: {
                        group: min(
                            float(sample["group_normalized_radial_slack"][group])
                            for sample in exact_group_trace
                            if sample["phase"] == phase
                        )
                        for group in exact_group_order
                    }
                    for phase in ("prefix", "backup", "terminal_hold")
                    if any(sample["phase"] == phase for sample in exact_group_trace)
                }
                group_contact_count = {
                    group: len(events)
                    for group, events in exact_group_contacts.items()
                }
                exact_group_record = {
                    "group_order": exact_group_order,
                    "group_representation": exact_target.get(
                        "group_representation", {}
                    ),
                    "group_minimum_normalized_radial_slack": group_minimum,
                    "group_future_violation": {
                        group: -float(value)
                        for group, value in group_minimum.items()
                    },
                    "phase_group_minimum_normalized_radial_slack": phase_minimum,
                    "group_contact_sample_count": group_contact_count,
                    "group_contact_events": exact_group_contacts,
                    "known_outcome": candidate["terminal_status"] != "UNKNOWN_TIMEOUT",
                    "safe_terminal": bool(
                        candidate["terminal_status"] == "SAFE_TERMINAL"
                        and min(group_minimum.values()) > 0.0
                        and sum(group_contact_count.values()) == 0
                        and replay_car <= car_limit
                    ),
                    "trace": exact_group_trace,
                }
                if capture_action_boundaries:
                    trajectory_value = exact_group_action_boundary_values(
                        action_boundaries,
                        exact_group_trace,
                        group_order=exact_group_order,
                    )
                    eligible_phases = set(
                        str(item)
                        for item in trajectory_value_config[
                            "value_training_phases"
                        ]
                    )
                    for record in trajectory_value["records"]:
                        record["training_sample_eligible"] = bool(
                            candidate["terminal_status"] != "UNKNOWN_TIMEOUT"
                            and record["phase"] in eligible_phases
                        )
                    exact_group_record["action_boundaries"] = action_boundaries
                    exact_group_record["trajectory_policy_value"] = trajectory_value
            candidate_records.append({
                "case_id": case_config["case_id"],
                "name": candidate_name,
                "requested_alpha": float(candidate["requested_alpha"]),
                "source_effective_post_AEGIS_correction_l2_action": float(
                    candidate.get("effective_post_AEGIS_correction_l2_action", 0.0)
                ),
                "source_executed_actions": candidate["actions"],
                "source_terminal_status": candidate["terminal_status"],
                "source_physical_veto": bool(candidate["physical_veto"]),
                "source_raw_protected_contact_count": source_contacts,
                "source_proxy_nonoverlap": source_proxy_nonoverlap,
                "source_proxy_buffer_safe": source_proxy_buffer_safe,
                "source_row_minimum_clearance_m": source_row_minimum.tolist(),
                "replayed_row_minimum_clearance_m": replayed_row_minimum.tolist(),
                "proxy_replay_maximum_error_m": row_error,
                "source_maximum_CAR_m": source_car,
                "replayed_maximum_CAR_m": replay_car,
                "CAR_replay_error_m": abs(replay_car - source_car),
                "replayed_physical_veto": replay_physical_veto,
                "raw_protected_contact_sample_count": len(contacts),
                "raw_protected_contacts": contacts,
                "compiled_box_row_minimum_normalized_radial_slack": np.min(
                    slack, axis=0
                ).tolist(),
                "compiled_box_row_exact_overlap_sample_count": np.sum(
                    overlap, axis=0
                ).astype(int).tolist(),
                "compiled_box_minimum_normalized_radial_slack": float(np.min(slack)),
                "compiled_box_risk": float(-np.min(slack)),
                "compiled_box_any_exact_overlap": any_exact_overlap,
                "compiled_box_safe_terminal": compiled_safe_terminal,
                "exact_group_target": exact_group_record,
                "empirical_l6_proxy": empirical_proxy,
                "sample_count": int(len(proxy_trace)),
                "action_count": len(actions),
                "phase_sample_counts": {
                    phase: sum(value == phase for value in sample_phases)
                    for phase in ("prefix", "backup", "terminal_hold")
                },
            })

        tolerance = float(audit_config["gate"]["source_replay_tolerance_m"])
        source_replay_exact = bool(
            state_hash_matches
            and initial_clearance_error <= tolerance
            and all(
                candidate["proxy_replay_maximum_error_m"] <= tolerance
                and candidate["CAR_replay_error_m"] <= tolerance
                and candidate["replayed_physical_veto"]
                == candidate["source_physical_veto"]
                for candidate in candidate_records
            )
        )
        return {
            "case_id": case_config["case_id"],
            "state_step": state_step,
            "source_result": str(source_path),
            "source_result_file_sha256": case_config["source_result_file_sha256"],
            "source_result_payload_sha256": case_config["source_result_payload_sha256"],
            "source_snapshot_sha256": source["state"]["source_snapshot_sha256"],
            "physical_context": source["state"].get("physical_context"),
            "artifact_superset_contract": artifact_superset or None,
            "source_nominal_five_action_chunk": source["nominal_five_action_chunk"],
            "initial_compiled_obstacle_boxes": [box.to_record() for box in initial_boxes],
            "initial_empirical_robot_rows": [
                {
                    "body_name": str(link.body_name),
                    "center_m": np.asarray(link.center, dtype=np.float64).tolist(),
                    "rotation": np.asarray(link.rotation, dtype=np.float64).tolist(),
                    "semiaxes_m": (
                        np.asarray(link.semiaxes_m, dtype=np.float64)
                        * (
                            float(empirical_proxy["uniform_semiaxis_scale"])
                            if empirical_proxy is not None
                            and index in set(int(value) for value in empirical_proxy["scaled_rows"])
                            else 1.0
                        )
                    ).tolist(),
                }
                for index, link in enumerate(initial_links)
            ],
            "initial_exact_robot_rows": initial_exact_robot_rows,
            "initial_ee_pose": initial_ee_pose,
            "replayed_snapshot_sha256": source_hash,
            "state_hash_matches": state_hash_matches,
            "initial_clearance_replay_error_m": initial_clearance_error,
            "initial_compiled_box_minimum_normalized_radial_slack": float(
                min(initial_slacks)
            ),
            "initial_compiled_box_any_exact_overlap": bool(
                initial_representation[
                    "compiled_box_union_any_exact_solid_overlap"
                ]
                if empirical_proxy is None
                else min(initial_slacks) <= 0.0
            ),
            "initial_raw_protected_contact_count": int(
                initial_contacts["nonpositive_protected_contact_count"]
            ),
            "perception_rotation_canonicalization": perception_rotation_record,
            "exact_group_target": (
                None
                if exact_target is None
                else {
                    "group_order": exact_group_order,
                    "group_representation": exact_target.get(
                        "group_representation", {}
                    ),
                    "robot_primitive_certificate_pass": exact_certificate_pass,
                    "initial_row_normalized_radial_slack": initial_exact_group[
                        "row_normalized_radial_slack"
                    ],
                    "initial_group_normalized_radial_slack": initial_exact_group[
                        "group_normalized_radial_slack"
                    ],
                    "initial_group_contact_sample_count": {
                        group: len(events)
                        for group, events in initial_exact_group[
                            "group_contact_events"
                        ].items()
                    },
                    "compiled_box_count": initial_exact_group[
                        "compiled_box_count"
                    ],
                }
            ),
            "source_replay_exact": source_replay_exact,
            "candidates": candidate_records,
        }
    finally:
        if env is not None:
            env.close()


def audit(
    *, repo_root: Path, config_path: Path, expected_commit: str, output_path: Path,
) -> dict[str, Any]:
    from main.multilink_ellipsoid.compiled_box_risk_target_audit import (
        RESULT_SCHEMA,
        classify,
        load_config,
    )

    config = load_config(config_path)
    population = repo_root / config["population_manifest"]
    geometry = repo_root / config["geometry_config"]
    _require(_file_sha256(population) == config["population_manifest_file_sha256"],
             "compiled-box population manifest differs")
    _require(_file_sha256(geometry) == config["geometry_config_file_sha256"],
             "compiled-box geometry config differs")
    cases = [
        _evaluate_case(
            repo_root=repo_root,
            population_manifest=population,
            geometry_config_path=geometry,
            case_config=case_config,
            audit_config=config,
        )
        for case_config in config["cases"]
    ]
    classification = classify(cases, config["gate"])
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "complete",
        "scientific_result": True,
        "claim_scope": config["claim_scope"],
        "source": _git_identity(repo_root, expected_commit),
        "allocation": _allocation_record(),
        "config": config,
        "cases": cases,
        "classification": classification,
        "strict_gate_pass": classification["strict_gate_pass"],
        "interpretation": (
            "compiled_box_future_risk_target_mechanism_pass"
            if classification["strict_gate_pass"]
            else "compiled_box_future_risk_target_mechanism_no_go"
        ),
        "training_authorized": False,
    }
    result["result_payload_sha256"] = _sha256(_canonical(result))
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = audit(
        repo_root=args.repo_root.resolve(),
        config_path=args.config.resolve(),
        expected_commit=args.expected_commit,
        output_path=args.output.resolve(),
    )
    _atomic_write(args.output.resolve(), result)
    print(json.dumps({
        "strict_gate_pass": result["strict_gate_pass"],
        "interpretation": result["interpretation"],
        "classification": result["classification"],
        "result_payload_sha256": result["result_payload_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
