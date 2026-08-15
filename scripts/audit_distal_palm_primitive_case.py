#!/usr/bin/env python3
"""Replay one frozen palm audit case at every internal MuJoCo substep."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Optional, Sequence

from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _git_identity, _load, _require,
)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def _expected_palm_steps(contact_path: Path, geom_name: str) -> set[int]:
    with gzip.open(contact_path, "rt", encoding="utf-8") as stream:
        artifact = json.load(stream)
    if artifact.get("schema_version") != "vlsa_table1_active_obstacle_contacts.v3":
        raise ValueError("palm audit source contact schema differs")
    return {
        int(event["step"])
        for snapshot in artifact["snapshots"]
        for event in snapshot["events"]
        if event.get("other", {}).get("classification") == "robot"
        and event.get("other", {}).get("geom_name") == geom_name
    }


def evaluate_case(
    *, repo_root: Path, config_path: Path, expected_commit: str, case_index: int,
) -> dict[str, Any]:
    import numpy as np

    from main.evaluate_safelibero_aegis import (
        TABLE_RENDER_RESOLUTION,
        TABLE_SETTLE_ACTIONS,
        _active_obstacle,
        _build_environment,
        _contact_model_authority,
        _detailed_active_obstacle_contacts,
        _runtime_imports,
        _settle,
        pairing_record,
        read_jsonl,
        validate_case_row,
    )
    from main.multilink_ellipsoid.barrier import support_gap
    from main.multilink_ellipsoid.geometry import Ellipsoid
    from main.multilink_ellipsoid.palm_primitive_audit import (
        RESULT_SCHEMA,
        RESULT_SCHEMA_V2,
        RESULT_SCHEMA_V3,
        compiled_obstacle_templates,
        file_sha256,
        fit_compiled_mesh_geom,
        load_config,
        payload_sha256,
        point_quadratic,
        world_ellipsoid,
    )
    from main.multilink_ellipsoid.shadow import (
        _released_aegis_end_effector_ellipsoid,
        allocation_record,
    )

    started = time.perf_counter_ns()
    source_identity = _git_identity(repo_root, expected_commit)
    config = load_config(config_path, repo_root=repo_root)
    source = config["source"]
    availability_result_path = Path(source["availability_result"])
    availability_validation_path = Path(source["availability_validation"])
    _require(
        file_sha256(availability_result_path)
        == source["availability_result_file_sha256"],
        "palm audit availability result file differs",
    )
    _require(
        file_sha256(availability_validation_path)
        == source["availability_validation_file_sha256"],
        "palm audit availability validation file differs",
    )
    availability = _load(availability_result_path)
    validation = _load(availability_validation_path)
    _require(
        availability["result_payload_sha256"]
        == source["availability_result_payload_sha256"],
        "palm audit availability result payload differs",
    )
    _require(
        validation["validation_payload_sha256"]
        == source["availability_validation_payload_sha256"],
        "palm audit availability validation payload differs",
    )
    cohort = (
        list(config["cohort"]["contact_case_ids"])
        + list(config["cohort"]["control_case_ids"])
    )
    if case_index < 0 or case_index >= len(cohort):
        raise ValueError("palm audit case index is outside the frozen cohort")
    case_id = str(cohort[case_index])
    expected_class = (
        "contact" if case_index < len(config["cohort"]["contact_case_ids"])
        else "control"
    )
    availability_by_id = {
        str(item["case_id"]): item for item in availability["records"]
    }
    _require(case_id in availability_by_id, "palm audit case lacks availability record")
    availability_record = availability_by_id[case_id]
    if expected_class == "contact":
        _require(
            "palm" in availability_record["eligible_clean_contact_groups"],
            "palm audit positive is not an eligible palm contact",
        )
    else:
        _require(
            bool(availability_record["eligible_clean_contact_free_control"]),
            "palm audit control is not eligible contact-free",
        )

    archived_path = Path(availability_record["result_path"])
    _require(
        file_sha256(archived_path) == availability_record["result_file_sha256"],
        "palm audit archived result file differs",
    )
    archived = _load(archived_path)
    _require(
        archived["result_payload_sha256"]
        == availability_record["result_payload_sha256"],
        "palm audit archived result payload differs",
    )
    contact_record = archived["failure_diagnostics"]["contacts"]
    contact_path = archived_path.parents[2] / str(contact_record["path"])
    _require(
        file_sha256(contact_path) == availability_record["contact_file_sha256"],
        "palm audit source contact artifact differs",
    )
    geom_name = str(config["primitive"]["geom_name"])
    expected_palm_steps = _expected_palm_steps(contact_path, geom_name)

    rows = [
        row for row in read_jsonl(repo_root / source["population_manifest"])
        if row.get("case_id") == case_id
    ]
    _require(len(rows) == 1, "palm audit manifest case is not unique")
    case = rows[0]
    validate_case_row(case, repo_root)
    runtime = _runtime_imports(include_aegis=False)
    env = None
    try:
        env, task, observation, selected_initial_state = _build_environment(
            runtime, case, render_resolution=TABLE_RENDER_RESOLUTION,
        )
        observation = _settle(env, observation, TABLE_SETTLE_ACTIONS)
        obstacle_name, _ = _active_obstacle(env, observation)
        _require(
            obstacle_name == archived["obstacle"]["active_name"],
            "palm audit active obstacle differs",
        )
        obstacle_reference = np.asarray(
            observation[obstacle_name + "_pos"], dtype=np.float64,
        ).copy()
        pairing = pairing_record(
            case=case,
            selected_initial_state=selected_initial_state,
            settled_observation=observation,
            task_description=str(task.language),
            active_obstacle_name=obstacle_name,
            settled_simulator_state=np.asarray(
                env.sim.get_state().flatten(), dtype=np.float64,
            ),
        )
        pairing_keys = (
            "manifest_row_sha256", "initial_state_sha256",
            "initial_observation_sha256", "settled_simulator_state_sha256",
            "settled_active_obstacle_position_sha256",
            "policy_noise_schedule_sha256",
        )
        pairing_errors = [
            key for key in pairing_keys
            if pairing[key] != archived["pairing"][key]
        ]

        fit = config["primitive"]["fit"]
        template = fit_compiled_mesh_geom(
            env,
            geom_name,
            relative_padding=float(fit["relative_padding"]),
            tolerance=float(fit["khachiyan_tolerance"]),
            max_iterations=int(fit["khachiyan_max_iterations"]),
        )
        template_record = template.to_record()
        certificate = template_record["enclosure_certificate"]
        certificate_pass = bool(
            certificate["verified"]
            and certificate["maximum_normalized_quadratic"]
            <= 1.0 + float(config["gate"]["vertex_containment_tolerance"])
        )
        perception = archived["perception"]
        obstacle = Ellipsoid(
            center=perception["mvee_center"],
            rotation=perception["mvee_rotation"],
            semiaxes_m=perception["mvee_semiaxes"],
            body_name=obstacle_name,
            geom_name="released_aegis_perception_mvee",
            bound_source="released_aegis_perception_mvee",
        )
        contact_authority = _contact_model_authority(env, obstacle_name)
        tracked_obstacle_template = None
        if "tracked_obstacle" in config:
            root_id = int(contact_authority["active_obstacle_root_body_id"])
            root_rotation = np.asarray(
                env.sim.data.xmat[root_id], dtype=np.float64,
            ).reshape(3, 3)
            root_position = np.asarray(
                env.sim.data.xpos[root_id], dtype=np.float64,
            )
            tracked_obstacle_template = {
                "root_body_id": root_id,
                "center_local_m": (
                    root_rotation.T @ (obstacle.center - root_position)
                ),
                "rotation_local": root_rotation.T @ obstacle.rotation,
                "semiaxes_m": obstacle.semiaxes_m.copy(),
            }
        compiled_templates = None
        if "compiled_obstacle" in config:
            compiled_fit = config["compiled_obstacle"]["fit"]
            compiled_templates = compiled_obstacle_templates(
                env,
                obstacle_name,
                relative_padding=float(compiled_fit["relative_padding"]),
                tolerance=float(compiled_fit["khachiyan_tolerance"]),
                max_iterations=int(compiled_fit["khachiyan_max_iterations"]),
            )
        overlap_tolerance = float(config["gate"]["support_gap_tolerance_m"])
        point_tolerance = float(config["gate"]["contact_point_tolerance"])
        trace_rows: list[list[Any]] = []
        contact_witnesses: list[dict[str, Any]] = []
        action_boundary_palm_steps: set[int] = set()
        maximum_obstacle_displacement = 0.0

        def measure(action_index: int, substep_index: int) -> None:
            nonlocal maximum_obstacle_displacement
            tight = world_ellipsoid(env, template)
            released = _released_aegis_end_effector_ellipsoid(env)
            tight_gap = float(support_gap(tight, obstacle))
            released_gap = float(support_gap(released, obstacle))
            if tracked_obstacle_template is not None:
                root_id = int(tracked_obstacle_template["root_body_id"])
                root_rotation = np.asarray(
                    env.sim.data.xmat[root_id], dtype=np.float64,
                ).reshape(3, 3)
                root_position = np.asarray(
                    env.sim.data.xpos[root_id], dtype=np.float64,
                )
                primary_obstacle = Ellipsoid(
                    center=(
                        root_position
                        + root_rotation
                        @ tracked_obstacle_template["center_local_m"]
                    ),
                    rotation=(
                        root_rotation
                        @ tracked_obstacle_template["rotation_local"]
                    ),
                    semiaxes_m=tracked_obstacle_template["semiaxes_m"],
                    body_id=root_id,
                    body_name=obstacle_name,
                    geom_name="pose_tracked_released_aegis_perception_mvee",
                    bound_source="released_shape_exact_simulator_pose_tracking",
                )
                primary_gap = float(support_gap(tight, primary_obstacle))
            elif compiled_templates is not None:
                primary_gap = min(
                    float(support_gap(tight, world_ellipsoid(env, item)))
                    for item in compiled_templates
                )
            else:
                primary_gap = tight_gap
            contacts = _detailed_active_obstacle_contacts(
                env,
                obstacle_name,
                step=int(action_index),
                contact_authority=contact_authority,
            )
            _require(contacts["status"] == "available", "palm contact evidence unavailable")
            palm_events = [
                event for event in contacts["events"]
                if event.get("other", {}).get("classification") == "robot"
                and event.get("other", {}).get("geom_name") == geom_name
            ]
            palm_contact = bool(palm_events)
            displacement = float(np.sum(np.abs(
                np.asarray(env.sim.data.xpos[contact_authority["active_obstacle_root_body_id"]], dtype=np.float64)
                - obstacle_reference
            )))
            maximum_obstacle_displacement = max(maximum_obstacle_displacement, displacement)
            trace_rows.append([
                int(action_index), int(substep_index), tight_gap, released_gap,
                primary_gap, palm_contact, len(palm_events), displacement,
            ])
            for event in palm_events:
                position = [float(item) for item in event["position"]]
                quadratic = point_quadratic(tight, position)
                contact_witnesses.append({
                    "action_index": int(action_index),
                    "substep_index": int(substep_index),
                    "distance_m": float(event["distance"]),
                    "position_m": position,
                    "obstacle_geom_name": str(event["obstacle"]["geom_name"]),
                    "tight_support_gap_m": tight_gap,
                    "primary_obstacle_support_gap_m": primary_gap,
                    "released_support_gap_m": released_gap,
                    "tight_contact_point_quadratic": quadratic,
                    "tight_contact_point_inside": bool(quadratic <= 1.0 + point_tolerance),
                })

        measure(-1, -1)
        replay_errors: list[str] = []
        expected_substeps = int(config["replay"]["expected_mujoco_substeps_per_action"])
        for action_index, action_record in enumerate(archived["actions"]):
            if int(action_record["step"]) != action_index:
                replay_errors.append("action_index")
                break
            action = np.asarray(action_record["env_step_input"], dtype=np.float64)
            if action.shape != (7,) or not np.array_equal(
                action, np.asarray(action_record["executed"], dtype=np.float64),
            ):
                replay_errors.append("executed_action")
                break
            before = len(trace_rows)
            original_step = env.sim.step

            def instrumented_step(*args: Any, **kwargs: Any) -> Any:
                value = original_step(*args, **kwargs)
                measure(action_index, len(trace_rows) - before)
                return value

            env.sim.step = instrumented_step
            try:
                observation, reward, done, _ = env.step(action.tolist())
            finally:
                env.sim.step = original_step
            if len(trace_rows) - before != expected_substeps:
                replay_errors.append("internal_substep_count")
            boundary_contacts = _detailed_active_obstacle_contacts(
                env,
                obstacle_name,
                step=action_index,
                contact_authority=contact_authority,
            )
            boundary_palm = any(
                event.get("other", {}).get("classification") == "robot"
                and event.get("other", {}).get("geom_name") == geom_name
                for event in boundary_contacts["events"]
            )
            if boundary_palm:
                action_boundary_palm_steps.add(action_index)
            if bool(done) is not bool(action_record["done"]):
                replay_errors.append("done")
            if not math.isclose(
                float(reward), float(action_record["reward"]),
                rel_tol=0.0, abs_tol=float(config["replay"]["scalar_tolerance"]),
            ):
                replay_errors.append("reward")
            replay_eef = np.asarray(observation["robot0_eef_pos"], dtype=np.float64)
            source_eef = np.asarray(
                action_record["post_step_controller_proxy"]["eef_position"],
                dtype=np.float64,
            )
            if not np.allclose(
                replay_eef, source_eef, rtol=0.0,
                atol=float(config["replay"]["state_tolerance"]),
            ):
                replay_errors.append("eef_position")
            displacement = float(np.sum(np.abs(
                np.asarray(observation[obstacle_name + "_pos"], dtype=np.float64)
                - obstacle_reference
            )))
            if not math.isclose(
                displacement,
                float(action_record["obstacle_l1_displacement_m"]),
                rel_tol=0.0,
                abs_tol=float(config["replay"]["state_tolerance"]),
            ):
                replay_errors.append("obstacle_displacement")

        replay_errors = sorted(set(replay_errors))
        boundary_steps_match = action_boundary_palm_steps == expected_palm_steps
        if not boundary_steps_match:
            replay_errors.append("action_boundary_palm_steps")
        tight_gaps = np.asarray([float(row[2]) for row in trace_rows], dtype=np.float64)
        released_gaps = np.asarray([float(row[3]) for row in trace_rows], dtype=np.float64)
        primary_gaps = np.asarray([float(row[4]) for row in trace_rows], dtype=np.float64)
        contact_flags = np.asarray([bool(row[5]) for row in trace_rows], dtype=bool)
        tight_false_safe = int(np.count_nonzero(
            contact_flags & (primary_gaps > overlap_tolerance)
        ))
        static_obstacle_false_safe = int(np.count_nonzero(
            contact_flags & (tight_gaps > overlap_tolerance)
        ))
        released_false_safe = int(np.count_nonzero(
            contact_flags & (released_gaps > overlap_tolerance)
        ))
        tight_false_unsafe = int(np.count_nonzero(
            (~contact_flags) & (primary_gaps <= 0.0)
        ))
        static_obstacle_false_unsafe = int(np.count_nonzero(
            (~contact_flags) & (tight_gaps <= 0.0)
        ))
        released_false_unsafe = int(np.count_nonzero(
            (~contact_flags) & (released_gaps <= 0.0)
        ))
        released_volume = float(4.0 * math.pi * np.prod([0.06, 0.12, 0.11]) / 3.0)
        trace_sha = hashlib.sha256(_canonical(trace_rows)).hexdigest()
        result = {
            "schema_version": (
                RESULT_SCHEMA_V3 if tracked_obstacle_template is not None else
                RESULT_SCHEMA_V2 if compiled_templates is not None else RESULT_SCHEMA
            ),
            "status": "complete",
            "scientific_result": True,
            "claim_scope": config["claim_scope"],
            "case_id": case_id,
            "case_index": int(case_index),
            "expected_class": expected_class,
            "source": source_identity,
            "allocation": allocation_record(),
            "config": config,
            "availability_binding": {
                "result_path": str(availability_result_path),
                "result_file_sha256": source["availability_result_file_sha256"],
                "result_payload_sha256": source["availability_result_payload_sha256"],
                "validation_path": str(availability_validation_path),
                "validation_file_sha256": source["availability_validation_file_sha256"],
                "validation_payload_sha256": source["availability_validation_payload_sha256"],
            },
            "archived_table1": {
                "path": str(archived_path),
                "file_sha256": availability_record["result_file_sha256"],
                "payload_sha256": availability_record["result_payload_sha256"],
                "contact_path": str(contact_path),
                "contact_file_sha256": availability_record["contact_file_sha256"],
                "action_count": len(archived["actions"]),
            },
            "pairing": pairing,
            "primitive_fit": {
                **template_record,
                "certificate_pass": certificate_pass,
                "fit_uses_contact_outcomes": False,
                "released_proxy_volume_m3": released_volume,
                "tight_to_released_volume_ratio": (
                    float(template_record["volume_m3"]) / released_volume
                ),
            },
            "compiled_obstacle_fit": (
                {
                    "geom_count": len(compiled_templates),
                    "templates": [item.to_record() for item in compiled_templates],
                    "fit_uses_contact_outcomes": False,
                    "privileged_simulation_geometry": True,
                }
                if compiled_templates is not None else None
            ),
            "tracked_obstacle_binding": (
                {
                    "root_body_id": int(tracked_obstacle_template["root_body_id"]),
                    "center_local_m": tracked_obstacle_template["center_local_m"].tolist(),
                    "rotation_local": tracked_obstacle_template["rotation_local"].tolist(),
                    "semiaxes_m": tracked_obstacle_template["semiaxes_m"].tolist(),
                    "shape_or_threshold_tuning_from_contacts": False,
                    "privileged_simulation_pose": True,
                }
                if tracked_obstacle_template is not None else None
            ),
            "replay": {
                "fidelity_pass": bool(not pairing_errors and not replay_errors),
                "pairing_errors": pairing_errors,
                "replay_errors": sorted(set(replay_errors)),
                "expected_action_boundary_palm_steps": sorted(expected_palm_steps),
                "replayed_action_boundary_palm_steps": sorted(action_boundary_palm_steps),
                "action_boundary_palm_steps_match": boundary_steps_match,
                "sample_count": len(trace_rows),
                "trace_sha256": trace_sha,
                "maximum_active_obstacle_l1_displacement_m": maximum_obstacle_displacement,
            },
            "physical_contact": {
                "authority": "raw_MuJoCo_active_obstacle_contact_at_every_internal_substep",
                "internal_palm_contact_sample_count": int(np.count_nonzero(contact_flags)),
                "internal_palm_contact_event_count": len(contact_witnesses),
                "contact_witnesses": contact_witnesses,
                "contact_point_outside_tight_count": sum(
                    int(not item["tight_contact_point_inside"])
                    for item in contact_witnesses
                ),
            },
            "tight_primitive": {
                "obstacle_representation": (
                    "released_shape_live_exact_obstacle_root_pose"
                    if tracked_obstacle_template is not None else
                    "live_certified_compiled_obstacle_geom_union"
                    if compiled_templates is not None else
                    "static_released_AEGIS_perception_MVEE"
                ),
                "episode_minimum_support_gap_m": float(np.min(primary_gaps)),
                "overlap_sample_count": int(np.count_nonzero(primary_gaps <= 0.0)),
                "physical_false_safe_sample_count": tight_false_safe,
                "contact_free_overlap_sample_count": tight_false_unsafe,
            },
            "tight_with_released_obstacle_proxy": {
                "episode_minimum_support_gap_m": float(np.min(tight_gaps)),
                "overlap_sample_count": int(np.count_nonzero(tight_gaps <= 0.0)),
                "physical_false_safe_sample_count": static_obstacle_false_safe,
                "contact_free_overlap_sample_count": static_obstacle_false_unsafe,
            },
            "released_proxy": {
                "episode_minimum_support_gap_m": float(np.min(released_gaps)),
                "overlap_sample_count": int(np.count_nonzero(released_gaps <= 0.0)),
                "physical_false_safe_sample_count": released_false_safe,
                "contact_free_overlap_sample_count": released_false_unsafe,
            },
            "new_actions_generated": False,
            "new_policy_queries": False,
            "boundary_collection_authorized": False,
            "training_authorized": False,
            "QP_authorized": False,
            "closed_loop_authorized": False,
            "wall_seconds": (time.perf_counter_ns() - started) * 1.0e-9,
        }
        result["result_payload_sha256"] = payload_sha256(result)
        return result
    finally:
        if env is not None:
            env.close()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--case-index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = evaluate_case(
        repo_root=args.repo_root.resolve(),
        config_path=args.config.resolve(),
        expected_commit=args.expected_commit,
        case_index=args.case_index,
    )
    _atomic_write(args.output.resolve(), result)
    print(json.dumps({
        "case_id": result["case_id"],
        "expected_class": result["expected_class"],
        "fidelity_pass": result["replay"]["fidelity_pass"],
        "palm_contact_samples": result["physical_contact"]["internal_palm_contact_sample_count"],
        "tight_false_safes": result["tight_primitive"]["physical_false_safe_sample_count"],
        "tight_minimum_gap_m": result["tight_primitive"]["episode_minimum_support_gap_m"],
        "released_minimum_gap_m": result["released_proxy"]["episode_minimum_support_gap_m"],
        "result_payload_sha256": result["result_payload_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
