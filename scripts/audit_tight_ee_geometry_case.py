#!/usr/bin/env python3
"""Replay one frozen case and audit five tight EE collision primitives."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
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


def _expected_group_steps(
    path: Path, geom_to_group: Mapping[str, str],
) -> dict[str, set[int]]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        artifact = json.load(stream)
    if artifact.get("schema_version") != "vlsa_table1_active_obstacle_contacts.v3":
        raise ValueError("tight EE source contact schema differs")
    output = {group: set() for group in geom_to_group.values()}
    for snapshot in artifact["snapshots"]:
        for event in snapshot["events"]:
            if event.get("other", {}).get("classification") != "robot":
                continue
            group = geom_to_group.get(str(event["other"]["geom_name"]))
            if group is not None:
                output[group].add(int(event["step"]))
    return output


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _render_overlay(
    *, env: Any, templates: Mapping[str, Any], config: Mapping[str, Any],
    output_dir: Path, contact_record: Mapping[str, Any],
) -> dict[str, Any]:
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    from robosuite.utils.camera_utils import (
        get_camera_extrinsic_matrix, get_camera_intrinsic_matrix,
    )

    from main.multilink_ellipsoid.palm_primitive_audit import world_ellipsoid
    from main.multilink_ellipsoid.shadow import _released_aegis_end_effector_ellipsoid
    from scripts.render_multilink_ellipsoid_overlay import (
        _frame_arm_camera, _project, _rotate_projection_180, _segments,
        _wire_loops,
    )

    tight = {group: world_ellipsoid(env, template) for group, template in templates.items()}
    released = _released_aegis_end_effector_ellipsoid(env)
    camera_name = str(config["visualization"]["camera_name"])
    resolution = int(config["visualization"]["resolution"])
    camera_record = _frame_arm_camera(
        env, [*tight.values(), released], camera_name,
    )
    observation = env.env._get_observations()
    image = np.ascontiguousarray(
        np.asarray(observation[camera_name + "_image"], dtype=np.uint8)[::-1, ::-1]
    )
    if image.shape != (resolution, resolution, 3):
        raise ValueError("tight EE simulator image shape differs")
    intrinsic = get_camera_intrinsic_matrix(
        env.sim, camera_name, resolution, resolution,
    )
    world_to_camera = np.linalg.inv(
        get_camera_extrinsic_matrix(env.sim, camera_name)
    )
    colors = {
        "palm": (46, 204, 113, 255),
        "finger1_base": (255, 152, 0, 255),
        "finger1_pad": (255, 235, 59, 255),
        "finger2_base": (233, 30, 99, 255),
        "finger2_pad": (63, 81, 181, 255),
        "released_proxy": (0, 188, 212, 150),
    }
    labels = {
        "palm": "Palm",
        "finger1_base": "F1 base",
        "finger1_pad": "F1 pad",
        "finger2_base": "F2 base",
        "finger2_pad": "F2 pad",
        "released_proxy": "Old EE proxy",
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    base = output_dir / "real-simulator-frame.jpg"
    Image.fromarray(image).save(base, quality=92, optimize=True)
    overlay = Image.new("RGBA", (resolution, resolution), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()
    rows = [("released_proxy", released), *tight.items()]
    primitive_records = []
    for name, ellipsoid in rows:
        color = colors[name]
        width = 3 if name == "released_proxy" else 2
        visible = 0
        for loop in _wire_loops(ellipsoid):
            projected = _rotate_projection_180(
                _project(loop, world_to_camera, intrinsic, resolution),
                resolution, resolution,
            )
            visible += sum(point is not None for point in projected)
            for segment in _segments(projected):
                draw.line(segment, fill=color, width=width, joint="curve")
        center = _rotate_projection_180(
            _project(
                np.asarray([ellipsoid.center]), world_to_camera, intrinsic,
                resolution,
            ),
            resolution, resolution,
        )[0]
        if center is None:
            raise ValueError("tight EE ellipsoid center is behind camera")
        x, y = center
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color)
        draw.text(
            (x + 6, y - 7), labels[name], fill=(255, 255, 255, 255),
            font=font, stroke_width=2, stroke_fill=(0, 0, 0, 230),
        )
        primitive_records.append({
            "name": name,
            "color_rgba": list(color),
            "center_px": [float(x), float(y)],
            "visible_wire_points": visible,
            "ellipsoid": ellipsoid.to_record(),
        })
    legend_x, legend_y = 16, 16
    legend_h = 20 * len(rows) + 18
    draw.rounded_rectangle(
        (legend_x, legend_y, legend_x + 170, legend_y + legend_h),
        radius=8, fill=(0, 0, 0, 175), outline=(255, 255, 255, 160),
    )
    draw.text(
        (legend_x + 9, legend_y + 6), "Real MuJoCo contact frame",
        fill=(255, 255, 255, 255), font=font,
    )
    for index, (name, _) in enumerate(rows):
        y = legend_y + 26 + index * 20
        draw.line(
            (legend_x + 10, y + 5, legend_x + 30, y + 5),
            fill=colors[name], width=4,
        )
        draw.text(
            (legend_x + 38, y), labels[name], fill=(255, 255, 255, 255),
            font=font,
        )
    overlay_path = output_dir / "tight-ee-overlay.png"
    overlay.save(overlay_path, optimize=True)
    preview = Image.alpha_composite(
        Image.fromarray(image).convert("RGBA"), overlay,
    ).convert("RGB")
    preview_path = output_dir / "tight-ee-vs-released.jpg"
    preview.save(preview_path, quality=94, optimize=True)
    record = {
        "schema_version": "vlsa_tight_ee_geometry_visualization.v1",
        "simulator": "SafeLIBERO MuJoCo",
        "frame": config["visualization"]["frame"],
        "contact": dict(contact_record),
        "camera": camera_record,
        "display_transform": "rotate_180_matching_released_AEGIS_camera_preprocessing",
        "primitive_records": primitive_records,
        "base_image": {"path": base.name, "sha256": _sha256_path(base)},
        "overlay": {"path": overlay_path.name, "sha256": _sha256_path(overlay_path)},
        "preview": {"path": preview_path.name, "sha256": _sha256_path(preview_path)},
    }
    record["payload_sha256"] = hashlib.sha256(_canonical(record)).hexdigest()
    metadata = output_dir / "visualization.json"
    metadata.write_bytes(json.dumps(record, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    return {**record, "metadata_path": str(metadata)}


def evaluate_case(
    *, repo_root: Path, config_path: Path, expected_commit: str,
    case_index: int, visualization_root: Path | None,
) -> dict[str, Any]:
    import numpy as np

    from main.evaluate_safelibero_aegis import (
        TABLE_RENDER_RESOLUTION, TABLE_SETTLE_ACTIONS, _active_obstacle,
        _build_environment, _contact_model_authority,
        _detailed_active_obstacle_contacts, _runtime_imports, _settle,
        pairing_record, read_jsonl, validate_case_row,
    )
    from main.multilink_ellipsoid.obstacle_proxy_audit import (
        compiled_obstacle_boxes, minimum_ellipsoid_quadratics_over_boxes,
    )
    from main.multilink_ellipsoid.palm_primitive_audit import (
        fit_compiled_contact_geom, point_quadratic, world_ellipsoid,
    )
    from main.multilink_ellipsoid.shadow import (
        _released_aegis_end_effector_ellipsoid, allocation_record,
    )
    from main.multilink_ellipsoid.tight_ee_geometry import (
        RESULT_SCHEMA, file_sha256, load_config, payload_sha256,
    )

    started = time.perf_counter_ns()
    source_identity = _git_identity(repo_root, expected_commit)
    config = load_config(config_path, repo_root=repo_root)
    cases = config["cohort"]["cases"]
    if case_index < 0 or case_index >= len(cases):
        raise ValueError("tight EE case index is outside frozen cohort")
    case_spec = cases[case_index]
    case_id = str(case_spec["case_id"])
    source = config["source"]
    availability_path = Path(source["availability_result"])
    availability_validation_path = Path(source["availability_validation"])
    _require(
        file_sha256(availability_path) == source["availability_result_file_sha256"],
        "tight EE availability result differs",
    )
    _require(
        file_sha256(availability_validation_path)
        == source["availability_validation_file_sha256"],
        "tight EE availability validation differs",
    )
    availability = _load(availability_path)
    availability_validation = _load(availability_validation_path)
    _require(
        availability["result_payload_sha256"]
        == source["availability_result_payload_sha256"],
        "tight EE availability payload differs",
    )
    _require(
        availability_validation["validation_payload_sha256"]
        == source["availability_validation_payload_sha256"],
        "tight EE availability validation payload differs",
    )
    availability_by_id = {
        str(item["case_id"]): item for item in availability["records"]
    }
    _require(case_id in availability_by_id, "tight EE case lacks availability record")
    availability_record = availability_by_id[case_id]
    _require(
        set(case_spec["expected_contact_groups"])
        <= set(availability_record["contact_groups"]),
        "tight EE expected raw contact is absent from immutable audit",
    )
    archived_path = Path(availability_record["result_path"])
    _require(
        file_sha256(archived_path) == availability_record["result_file_sha256"],
        "tight EE archived result differs",
    )
    archived = _load(archived_path)
    _require(
        archived["result_payload_sha256"] == availability_record["result_payload_sha256"],
        "tight EE archived payload differs",
    )
    contact_path = archived_path.parents[2] / archived["failure_diagnostics"]["contacts"]["path"]
    _require(
        file_sha256(contact_path) == availability_record["contact_file_sha256"],
        "tight EE contact artifact differs",
    )
    group_to_geom = {
        str(group): str(geom)
        for group, geom in config["tight_ee_primitives"].items()
    }
    geom_to_group = {geom: group for group, geom in group_to_geom.items()}
    expected_boundary_steps = _expected_group_steps(contact_path, geom_to_group)
    rows = [
        row for row in read_jsonl(repo_root / source["population_manifest"])
        if row.get("case_id") == case_id
    ]
    _require(len(rows) == 1, "tight EE manifest case is not unique")
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
            "tight EE active obstacle differs",
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
        fit = config["fit"]
        templates = {
            group: fit_compiled_contact_geom(
                env, geom,
                relative_padding=float(fit["relative_padding"]),
                tolerance=float(fit["khachiyan_tolerance"]),
                max_iterations=int(fit["khachiyan_max_iterations"]),
            )
            for group, geom in group_to_geom.items()
        }
        template_records = {
            group: template.to_record() for group, template in templates.items()
        }
        certificate_pass = all(
            bool(record["enclosure_certificate"].get("verified"))
            for record in template_records.values()
        )
        compiled_obstacle_boxes(env, obstacle_name)
        contact_authority = _contact_model_authority(env, obstacle_name)
        overlap_tolerance = float(config["gate"]["radial_slack_contact_tolerance"])
        point_tolerance = float(config["gate"]["contact_point_quadratic_tolerance"])
        trace: list[dict[str, Any]] = []
        replayed_boundary_steps = {group: set() for group in group_to_geom}
        first_contact_state = None
        first_contact_record = None

        def measure(action_index: int, substep_index: int) -> None:
            nonlocal first_contact_state, first_contact_record
            ellipsoids = [
                world_ellipsoid(env, templates[group]) for group in group_to_geom
            ]
            boxes = compiled_obstacle_boxes(env, obstacle_name)
            pair = minimum_ellipsoid_quadratics_over_boxes(ellipsoids, boxes)
            slacks = np.sqrt(np.min(pair, axis=1)) - 1.0
            group_slack = {
                group: float(slacks[index])
                for index, group in enumerate(group_to_geom)
            }
            released = _released_aegis_end_effector_ellipsoid(env)
            released_pair = minimum_ellipsoid_quadratics_over_boxes([released], boxes)
            released_slack = float(np.sqrt(np.min(released_pair)) - 1.0)
            contacts = _detailed_active_obstacle_contacts(
                env, obstacle_name, step=int(action_index),
                contact_authority=contact_authority,
            )
            _require(contacts["status"] == "available", "tight EE raw contacts unavailable")
            event_count = {group: 0 for group in group_to_geom}
            witnesses = []
            for event in contacts["events"]:
                if event.get("other", {}).get("classification") != "robot":
                    continue
                group = geom_to_group.get(str(event["other"]["geom_name"]))
                if group is None:
                    continue
                event_count[group] += 1
                ellipsoid = ellipsoids[list(group_to_geom).index(group)]
                quadratic = point_quadratic(ellipsoid, event["position"])
                witnesses.append({
                    "group": group,
                    "robot_geom_name": str(event["other"]["geom_name"]),
                    "obstacle_geom_name": str(event["obstacle"]["geom_name"]),
                    "distance_m": float(event["distance"]),
                    "position_m": [float(value) for value in event["position"]],
                    "contact_point_quadratic": float(quadratic),
                    "contact_point_inside_tight_primitive": bool(
                        quadratic <= 1.0 + point_tolerance
                    ),
                })
            if witnesses and first_contact_state is None:
                first_contact_state = np.asarray(
                    env.sim.get_state().flatten(), dtype=np.float64,
                ).copy()
                first_contact_record = {
                    "action_index": int(action_index),
                    "substep_index": int(substep_index),
                    "witnesses": witnesses,
                }
            trace.append({
                "action_index": int(action_index),
                "substep_index": int(substep_index),
                "radial_slack_by_group": group_slack,
                "released_proxy_radial_slack": released_slack,
                "raw_contact_event_count_by_group": event_count,
                "contact_witnesses": witnesses,
                "compiled_box_count": len(boxes),
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
            before = len(trace)
            original_step = env.sim.step

            def instrumented_step(*args: Any, **kwargs: Any) -> Any:
                value = original_step(*args, **kwargs)
                measure(action_index, len(trace) - before)
                return value

            env.sim.step = instrumented_step
            try:
                observation, reward, done, _ = env.step(action.tolist())
            finally:
                env.sim.step = original_step
            if len(trace) - before != expected_substeps:
                replay_errors.append("internal_substep_count")
            boundary = _detailed_active_obstacle_contacts(
                env, obstacle_name, step=action_index,
                contact_authority=contact_authority,
            )
            for event in boundary["events"]:
                if event.get("other", {}).get("classification") != "robot":
                    continue
                group = geom_to_group.get(str(event["other"]["geom_name"]))
                if group is not None:
                    replayed_boundary_steps[group].add(action_index)
            if bool(done) is not bool(action_record["done"]):
                replay_errors.append("done")
            if not math.isclose(
                float(reward), float(action_record["reward"]), rel_tol=0.0,
                abs_tol=float(config["replay"]["scalar_tolerance"]),
            ):
                replay_errors.append("reward")
            if not np.allclose(
                np.asarray(observation["robot0_eef_pos"], dtype=np.float64),
                np.asarray(
                    action_record["post_step_controller_proxy"]["eef_position"],
                    dtype=np.float64,
                ),
                rtol=0.0, atol=float(config["replay"]["state_tolerance"]),
            ):
                replay_errors.append("eef_position")
            displacement = float(np.sum(np.abs(
                np.asarray(observation[obstacle_name + "_pos"], dtype=np.float64)
                - obstacle_reference
            )))
            if not math.isclose(
                displacement, float(action_record["obstacle_l1_displacement_m"]),
                rel_tol=0.0, abs_tol=float(config["replay"]["state_tolerance"]),
            ):
                replay_errors.append("obstacle_displacement")
        boundary_match = {
            group: replayed_boundary_steps[group] == expected_boundary_steps[group]
            for group in group_to_geom
        }
        if not all(boundary_match.values()):
            replay_errors.append("action_boundary_group_steps")
        sample_count_by_group = {
            group: sum(
                int(row["raw_contact_event_count_by_group"][group] > 0)
                for row in trace
            )
            for group in group_to_geom
        }
        event_count_by_group = {
            group: sum(
                int(row["raw_contact_event_count_by_group"][group])
                for row in trace
            )
            for group in group_to_geom
        }
        group_minimum = {
            group: min(float(row["radial_slack_by_group"][group]) for row in trace)
            for group in group_to_geom
        }
        false_safes = {
            group: sum(int(
                row["raw_contact_event_count_by_group"][group] > 0
                and row["radial_slack_by_group"][group] > overlap_tolerance
            ) for row in trace)
            for group in group_to_geom
        }
        point_outside = {
            group: sum(
                int(not witness["contact_point_inside_tight_primitive"])
                for row in trace for witness in row["contact_witnesses"]
                if witness["group"] == group
            )
            for group in group_to_geom
        }
        released_minimum = min(
            float(row["released_proxy_radial_slack"]) for row in trace
        )
        released_contact_free_overlap_samples = sum(int(
            not any(row["raw_contact_event_count_by_group"].values())
            and row["released_proxy_radial_slack"] <= 0.0
        ) for row in trace)
        visualization = None
        if (
            visualization_root is not None
            and case_id == config["visualization"]["case_id"]
        ):
            _require(first_contact_state is not None, "visualization case has no tight EE contact")
            env.sim.set_state_from_flattened(first_contact_state)
            env.sim.forward()
            env.env._update_observables(force=True)
            visualization = _render_overlay(
                env=env, templates=templates, config=config,
                output_dir=visualization_root / case_id,
                contact_record=first_contact_record,
            )
        released_volume = float(4.0 * math.pi * np.prod([0.06, 0.12, 0.11]) / 3.0)
        tight_total_volume = float(sum(
            record["volume_m3"] for record in template_records.values()
        ))
        result = {
            "schema_version": RESULT_SCHEMA,
            "status": "complete",
            "scientific_result": True,
            "claim_scope": config["claim_scope"],
            "case_id": case_id,
            "case_index": int(case_index),
            "case_classification": case_spec["classification"],
            "expected_contact_groups": list(case_spec["expected_contact_groups"]),
            "source": source_identity,
            "allocation": allocation_record(),
            "config": config,
            "archived_table1": {
                "path": str(archived_path),
                "file_sha256": availability_record["result_file_sha256"],
                "payload_sha256": availability_record["result_payload_sha256"],
                "contact_path": str(contact_path),
                "contact_file_sha256": availability_record["contact_file_sha256"],
                "action_count": len(archived["actions"]),
            },
            "pairing": pairing,
            "primitive_templates": template_records,
            "replay": {
                "fidelity_pass": bool(
                    not pairing_errors and not replay_errors
                ),
                "pairing_errors": pairing_errors,
                "replay_errors": sorted(set(replay_errors)),
                "expected_action_boundary_steps_by_group": {
                    group: sorted(values)
                    for group, values in expected_boundary_steps.items()
                },
                "replayed_action_boundary_steps_by_group": {
                    group: sorted(values)
                    for group, values in replayed_boundary_steps.items()
                },
                "action_boundary_steps_match_by_group": boundary_match,
                "sample_count": len(trace),
                "trace_sha256": hashlib.sha256(_canonical(trace)).hexdigest(),
            },
            "raw_contact": {
                "authority": "raw_MuJoCo_active_obstacle_contacts_at_every_internal_substep",
                "observed_groups": sorted(
                    group for group, count in sample_count_by_group.items()
                    if count > 0
                ),
                "sample_count_by_group": sample_count_by_group,
                "event_count_by_group": event_count_by_group,
                "contact_point_outside_tight_primitive_count_by_group": point_outside,
            },
            "geometry": {
                "representation": "five_separate_contact_aligned_compiled_geom_ellipsoids_vs_exact_compiled_obstacle_boxes",
                "units": config["geometry"]["units"],
                "primitive_certificate_pass": certificate_pass,
                "fit_uses_contact_outcomes": False,
                "episode_minimum_radial_slack_by_group": group_minimum,
                "physical_false_safe_sample_count_by_group": false_safes,
                "tight_union_total_volume_m3": tight_total_volume,
                "released_proxy_volume_m3": released_volume,
                "tight_union_to_released_volume_ratio": tight_total_volume / released_volume,
                "released_proxy_episode_minimum_radial_slack": released_minimum,
                "released_proxy_contact_free_overlap_sample_count": released_contact_free_overlap_samples,
                "trace_sha256": hashlib.sha256(_canonical(trace)).hexdigest(),
            },
            "visualization": visualization,
            "new_actions_generated": False,
            "new_policy_queries": False,
            "model_training_authorized": False,
            "QP_authorized": False,
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
    parser.add_argument("--visualization-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = evaluate_case(
        repo_root=args.repo_root.resolve(),
        config_path=args.config.resolve(),
        expected_commit=args.expected_commit,
        case_index=args.case_index,
        visualization_root=(
            args.visualization_root.resolve()
            if args.visualization_root is not None else None
        ),
    )
    _atomic_write(args.output.resolve(), result)
    print(json.dumps({
        "status": "complete", "case_id": result["case_id"],
        "payload_sha256": result["result_payload_sha256"],
    }, sort_keys=True), flush=True)
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
