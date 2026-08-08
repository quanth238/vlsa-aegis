#!/usr/bin/env python3
"""Replay the accepted predictive-flow ledger with verified ellipsoid video."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from scripts.replay_distal_three_ellipsoid_multicbf import (
    CASE_ID,
    PAPER_CAR_THRESHOLD_M,
    _atomic_write,
    _file_sha256,
    _git_identity,
    _load,
    _require,
)


EXPECTED_RESULT_SCHEMA = "vlsa_predictive_flow_guidance_e05_result.v1"
PAIRING_KEYS = (
    "manifest_row_sha256",
    "initial_state_sha256",
    "initial_observation_sha256",
    "settled_simulator_state_sha256",
    "settled_active_obstacle_position_sha256",
    "policy_noise_schedule_sha256",
)
BODY_COLORS = (
    (244, 67, 54, 245),
    (255, 152, 0, 245),
    (205, 220, 57, 245),
    (0, 229, 255, 245),
)
OBSTACLE_COLOR = (66, 133, 244, 230)
BODY_LABELS = ("L5", "L6", "L7", "EE")


def _state_sha256(value: Any) -> str:
    import numpy as np

    return hashlib.sha256(
        np.ascontiguousarray(value, dtype=np.float64).tobytes()
    ).hexdigest()


def _draw_ellipsoid(
    draw: Any,
    ellipsoid: Any,
    *,
    color: tuple[int, int, int, int],
    label: str,
    world_to_camera: Any,
    intrinsic: Any,
    width: int,
    height: int,
    font: Any,
) -> int:
    import numpy as np

    from scripts.render_multilink_ellipsoid_overlay import (
        _project,
        _rotate_projection_180,
        _segments,
        _wire_loops,
    )

    visible = 0
    for loop in _wire_loops(ellipsoid, samples=96):
        projected = _project(loop, world_to_camera, intrinsic, height)
        projected = _rotate_projection_180(projected, width, height)
        visible += sum(point is not None for point in projected)
        for segment in _segments(projected):
            draw.line(segment, fill=color, width=3, joint="curve")
    center = _project(
        np.asarray([ellipsoid.center], dtype=np.float64),
        world_to_camera,
        intrinsic,
        height,
    )
    center = _rotate_projection_180(center, width, height)[0]
    if center is not None:
        x, y = center
        if -20.0 <= x <= width + 20.0 and -20.0 <= y <= height + 20.0:
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color)
            draw.text(
                (x + 8, y - 10),
                label,
                fill=(255, 255, 255, 255),
                font=font,
                stroke_width=2,
                stroke_fill=(0, 0, 0, 245),
            )
    return visible


def _annotated_frame(
    image: Any,
    *,
    env: Any,
    geometry: Any,
    world_to_camera: Any,
    intrinsic: Any,
    clearances: Any,
    frame_index: int,
    action_step: Optional[int],
    guidance_active: bool,
    maximum_obstacle_displacement_m: float,
) -> tuple[Any, dict[str, int]]:
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    source = np.ascontiguousarray(np.asarray(image, dtype=np.uint8))
    _require(
        source.ndim == 3 and source.shape[2] == 3,
        "predictive video source frame is not RGB",
    )
    height, width = source.shape[:2]
    base = Image.fromarray(source).convert("RGBA")
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    try:
        label_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 22)
        text_font = ImageFont.truetype("DejaVuSans.ttf", 22)
        strong_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 24)
    except OSError:
        label_font = ImageFont.load_default()
        text_font = label_font
        strong_font = label_font

    visible: dict[str, int] = {}
    for ellipsoid, color, label in zip(
        geometry.ellipsoids(env), BODY_COLORS, BODY_LABELS
    ):
        visible[label] = _draw_ellipsoid(
            draw,
            ellipsoid,
            color=color,
            label=label,
            world_to_camera=world_to_camera,
            intrinsic=intrinsic,
            width=width,
            height=height,
            font=label_font,
        )
    visible["obstacle"] = _draw_ellipsoid(
        draw,
        geometry.obstacle,
        color=OBSTACLE_COLOR,
        label="obstacle MVEE",
        world_to_camera=world_to_camera,
        intrinsic=intrinsic,
        width=width,
        height=height,
        font=label_font,
    )
    rendered = Image.alpha_composite(base, overlay).convert("RGB")
    banner_height = 128
    canvas = Image.new("RGB", (width, height + banner_height), "black")
    canvas.paste(rendered, (0, 0))
    banner = ImageDraw.Draw(canvas)
    step_label = "initial" if action_step is None else "action %03d / 299" % action_step
    mode = "PREDICTIVE GUIDANCE ACTIVE" if guidance_active else "nominal pi0.5"
    banner.text(
        (16, height + 8),
        "OSC-consistent L5/L6/L7/EE replay | %s | %s" % (step_label, mode),
        fill="#ffcc66" if guidance_active else "white",
        font=strong_font,
    )
    clearance_mm = [1000.0 * float(value) for value in clearances]
    banner.text(
        (16, height + 43),
        "h_opt [mm]  L5 %+7.2f | L6 %+7.2f | L7 %+7.2f | EE %+7.2f"
        % tuple(clearance_mm),
        fill="white",
        font=text_font,
    )
    banner.text(
        (16, height + 78),
        "Raw robot contact: none | obstacle L1 displacement: %.6f mm | frame %03d"
        % (1000.0 * maximum_obstacle_displacement_m, frame_index),
        fill="#66dd99",
        font=text_font,
    )
    return np.ascontiguousarray(np.asarray(canvas)), visible


def _verify_video(
    imageio: Any,
    video_path: Path,
    source_samples: Mapping[int, Any],
    expected_frames: int,
) -> dict[str, Any]:
    import numpy as np

    reader = imageio.get_reader(str(video_path))
    try:
        decoded_frames = int(reader.count_frames())
        _require(decoded_frames == expected_frames, "decoded video frame count differs")
        sample_records = []
        maximum_mae = 0.0
        for index in sorted(source_samples):
            decoded = np.asarray(reader.get_data(index), dtype=np.uint8)
            source = np.asarray(source_samples[index], dtype=np.uint8)
            _require(decoded.shape == source.shape, "decoded video shape differs")
            mae = float(
                np.mean(
                    np.abs(decoded.astype(np.int16) - source.astype(np.int16))
                )
            )
            maximum_mae = max(maximum_mae, mae)
            _require(mae <= 12.0, "decoded video visual fidelity differs")
            _require(float(np.std(decoded)) >= 8.0, "decoded video frame is degenerate")
            sample_records.append(
                {
                    "frame_index": index,
                    "mean_absolute_pixel_error": mae,
                    "decoded_rgb_stddev": float(np.std(decoded)),
                }
            )
    finally:
        reader.close()
    return {
        "decoded_frames": decoded_frames,
        "sample_count": len(sample_records),
        "maximum_mean_absolute_pixel_error": maximum_mae,
        "samples": sample_records,
    }


def render(
    *,
    repo_root: Path,
    manifest_path: Path,
    config_path: Path,
    accepted_result_path: Path,
    expected_result_sha256: str,
    expected_commit: str,
    video_path: Path,
    preview_path: Path,
    receipt_path: Path,
) -> dict[str, Any]:
    import numpy as np
    from PIL import Image
    from robosuite.utils.camera_utils import (
        get_camera_extrinsic_matrix,
        get_camera_intrinsic_matrix,
    )

    from main.evaluate_safelibero_aegis import (
        TABLE_RENDER_RESOLUTION,
        TABLE_SETTLE_ACTIONS,
        TABLE_VIDEO_FPS,
        _active_obstacle,
        _build_environment,
        _contact_model_authority,
        _detailed_active_obstacle_contacts,
        _processed_image,
        _runtime_imports,
        _settle,
        pairing_record,
        read_jsonl,
        validate_case_row,
    )
    from main.multilink_ellipsoid.predictive_flow import (
        PredictiveFullBodyGeometry,
        load_predictive_flow_config,
    )
    from main.multilink_ellipsoid.rollout import _dynamic_state_vector
    from main.multilink_ellipsoid.shadow import allocation_record

    _require(video_path.suffix == ".mp4", "predictive replay video must be MP4")
    _require(preview_path.suffix.lower() == ".jpg", "preview must be JPEG")
    _require(receipt_path.suffix == ".json", "receipt must be JSON")
    for output in (video_path, preview_path, receipt_path):
        _require(not output.exists(), "predictive replay output already exists")
    partial = video_path.with_name(".%s.partial.mp4" % video_path.stem)
    _require(not partial.exists(), "predictive replay partial video exists")

    accepted = _load(accepted_result_path)
    accepted_sha = _file_sha256(accepted_result_path)
    _require(accepted_sha == expected_result_sha256, "accepted result hash differs")
    _require(accepted.get("schema_version") == EXPECTED_RESULT_SCHEMA, "result schema differs")
    _require(accepted.get("status") == "complete", "accepted result is incomplete")
    _require(accepted.get("case_id") == CASE_ID, "accepted result case differs")
    _require(accepted.get("primary_problem_solved") is False, "accepted outcome differs")
    actions = accepted.get("actions")
    _require(isinstance(actions, list) and len(actions) == 300, "action ledger differs")
    source = _git_identity(repo_root, expected_commit)
    allocation = allocation_record()
    config = load_predictive_flow_config(config_path)
    runtime = _runtime_imports(include_aegis=False)
    rows = read_jsonl(manifest_path)
    matches = [row for row in rows if row.get("case_id") == CASE_ID]
    _require(len(matches) == 1, "predictive replay manifest row is not unique")
    case = matches[0]
    validate_case_row(case, repo_root)

    video_path.parent.mkdir(parents=True, exist_ok=True)
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    env = None
    writer = None
    source_samples: dict[int, Any] = {}
    minimum_clearances = np.full(4, np.inf, dtype=np.float64)
    maximum_state_error = 0.0
    maximum_eef_error = 0.0
    maximum_displacement_error = 0.0
    maximum_displacement = 0.0
    maximum_visible = {label: 0 for label in (*BODY_LABELS, "obstacle")}
    minimum_visible = {label: 10**9 for label in (*BODY_LABELS, "obstacle")}
    try:
        env, task, observation, selected_initial_state = _build_environment(
            runtime,
            case,
            render_resolution=TABLE_RENDER_RESOLUTION,
        )
        observation = _settle(env, observation, TABLE_SETTLE_ACTIONS)
        obstacle_name, _ = _active_obstacle(env, observation)
        initial_obstacle_position = np.asarray(
            observation["%s_pos" % obstacle_name], dtype=np.float64
        ).copy()
        pairing = pairing_record(
            case=case,
            selected_initial_state=selected_initial_state,
            settled_observation=observation,
            task_description=str(task.language),
            active_obstacle_name=obstacle_name,
            settled_simulator_state=np.asarray(
                env.sim.get_state().flatten(), dtype=np.float64
            ),
        )
        for key in PAIRING_KEYS:
            _require(pairing[key] == accepted["pairing"][key], "pairing differs: %s" % key)
        obstacle = accepted["geometry"]["obstacle"]
        geometry = PredictiveFullBodyGeometry.from_aegis_geometry(
            config,
            {
                "p2": obstacle["center_m"],
                "R2": obstacle["rotation"],
                "Q2_diag": obstacle["semiaxes_m"],
                "record": {"label": "blue moka pot"},
            },
        )
        initial_h = geometry.clearances(env)
        _require(
            float(np.max(np.abs(initial_h - accepted["policy_queries"][0]["current_h_opt_m"])))
            <= 1.0e-10,
            "initial predictive clearances differ",
        )
        contact_authority = _contact_model_authority(env, obstacle_name)
        camera_name = "agentview"
        intrinsic = get_camera_intrinsic_matrix(
            env.sim,
            camera_name,
            TABLE_RENDER_RESOLUTION,
            TABLE_RENDER_RESOLUTION,
        )
        world_to_camera = np.linalg.inv(
            get_camera_extrinsic_matrix(env.sim, camera_name)
        )
        writer = runtime["imageio"].get_writer(
            str(partial),
            fps=TABLE_VIDEO_FPS,
            codec="libx264",
            macro_block_size=None,
            pixelformat="yuv420p",
            output_params=["-crf", "18", "-movflags", "+faststart"],
        )

        def append_frame(
            *, frame_index: int, action_step: Optional[int], active: bool
        ) -> None:
            current_h = geometry.clearances(env)
            minimum_clearances[:] = np.minimum(minimum_clearances, current_h)
            frame, visible = _annotated_frame(
                _processed_image(observation, "agentview_image"),
                env=env,
                geometry=geometry,
                world_to_camera=world_to_camera,
                intrinsic=intrinsic,
                clearances=current_h,
                frame_index=frame_index,
                action_step=action_step,
                guidance_active=active,
                maximum_obstacle_displacement_m=maximum_displacement,
            )
            for label, value in visible.items():
                maximum_visible[label] = max(maximum_visible[label], value)
                minimum_visible[label] = min(minimum_visible[label], value)
            writer.append_data(frame)
            if frame_index % 60 == 0:
                source_samples[frame_index] = frame.copy()
            if frame_index == 60:
                Image.fromarray(frame).save(preview_path, quality=92, optimize=True)

        append_frame(frame_index=0, action_step=None, active=True)
        for index, record in enumerate(actions):
            _require(int(record["step"]) == index, "action index differs")
            executed = np.asarray(record["executed_action"], dtype=np.float64)
            _require(executed.shape == (7,), "executed action shape differs")
            observation, reward, done, _ = env.step(executed.tolist())
            _require(float(reward) == float(record["reward"]), "reward trace differs")
            _require(bool(done) is bool(record["done"]), "done trace differs")
            actual_state = _dynamic_state_vector(env)
            _require(
                _state_sha256(actual_state) == record["main_next_state_sha256"],
                "dynamic-state hash trace differs",
            )
            state_error = float(record["main_vs_verified_clone_max_abs_error"])
            maximum_state_error = max(maximum_state_error, state_error)
            observed_eef = np.asarray(observation["robot0_eef_pos"], dtype=np.float64)
            recorded_eef = np.asarray(record["eef_position_m"], dtype=np.float64)
            eef_error = float(np.max(np.abs(observed_eef - recorded_eef)))
            maximum_eef_error = max(maximum_eef_error, eef_error)
            _require(eef_error <= 1.0e-10, "end-effector trace differs")
            displacement = float(
                np.sum(
                    np.abs(
                        np.asarray(
                            observation["%s_pos" % obstacle_name], dtype=np.float64
                        )
                        - initial_obstacle_position
                    )
                )
            )
            displacement_error = abs(
                displacement - float(record["active_obstacle_l1_displacement_m"])
            )
            maximum_displacement_error = max(
                maximum_displacement_error, displacement_error
            )
            _require(displacement_error <= 1.0e-10, "obstacle trace differs")
            maximum_displacement = max(maximum_displacement, displacement)
            contacts = _detailed_active_obstacle_contacts(
                env,
                obstacle_name,
                step=index,
                contact_authority=contact_authority,
            )
            robot_events = [
                event
                for event in contacts["events"]
                if event.get("other", {}).get("classification") == "robot"
            ]
            _require(not robot_events, "replay produced robot contact")
            if (index + 1) % 5 == 0 and index < 295:
                next_query = (index + 1) // 5
                expected_h = np.asarray(
                    accepted["policy_queries"][next_query]["current_h_opt_m"],
                    dtype=np.float64,
                )
                _require(
                    float(np.max(np.abs(geometry.clearances(env) - expected_h)))
                    <= 1.0e-10,
                    "policy-query clearance trace differs",
                )
            query_index = min(index // 5, len(accepted["policy_queries"]) - 1)
            append_frame(
                frame_index=index + 1,
                action_step=index,
                active=bool(accepted["policy_queries"][query_index]["activation"]),
            )
        _require(maximum_displacement <= PAPER_CAR_THRESHOLD_M, "replay produced CAR")
    finally:
        try:
            if writer is not None:
                writer.close()
        finally:
            if env is not None:
                env.close()

    _require(partial.is_file() and partial.stat().st_size > 0, "encoder produced no video")
    os.replace(partial, video_path)
    _require(preview_path.is_file(), "predictive replay preview is missing")
    verification = _verify_video(
        runtime["imageio"],
        video_path,
        source_samples,
        expected_frames=301,
    )
    _require(all(value >= 96 for value in maximum_visible.values()), "ellipsoid overlay is not visible")
    receipt = {
        "schema_version": "vlsa_predictive_flow_guidance_video.v1",
        "status": "verified",
        "case_id": CASE_ID,
        "source": source,
        "allocation": allocation,
        "accepted_result": {
            "path": str(accepted_result_path),
            "file_sha256": accepted_sha,
            "payload_sha256": accepted["result_payload_sha256"],
            "read_only": True,
        },
        "pairing_sha256_fields_verified": list(PAIRING_KEYS),
        "trace_equivalence": {
            "action_count": len(actions),
            "maximum_state_error_m": maximum_state_error,
            "maximum_eef_coordinate_error_m": maximum_eef_error,
            "maximum_obstacle_displacement_error_m": maximum_displacement_error,
            "maximum_active_obstacle_l1_displacement_m": maximum_displacement,
            "robot_contact": False,
            "paper_car": False,
            "native_task_success": False,
        },
        "clearance": {
            "body_names": list(BODY_LABELS),
            "minimum_h_opt_m": minimum_clearances.tolist(),
        },
        "overlay": {
            "semantics": "projected live L5/L6/L7/EE ellipsoid wireframes plus frozen obstacle MVEE",
            "minimum_visible_wire_points": minimum_visible,
            "maximum_visible_wire_points": maximum_visible,
        },
        "video": {
            "path": str(video_path),
            "sha256": _file_sha256(video_path),
            "source_frames": 301,
            "fps": TABLE_VIDEO_FPS,
            "codec": "libx264_yuv420p_crf18_faststart",
            "decoded_fidelity": verification,
        },
        "preview": {
            "path": str(preview_path),
            "sha256": _file_sha256(preview_path),
            "source_frame_index": 60,
        },
    }
    _atomic_write(receipt_path, receipt)
    return receipt


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--accepted-result", type=Path, required=True)
    parser.add_argument("--expected-result-sha256", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--preview", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = render(
        repo_root=args.repo_root.resolve(),
        manifest_path=args.manifest.resolve(),
        config_path=args.config.resolve(),
        accepted_result_path=args.accepted_result.resolve(),
        expected_result_sha256=args.expected_result_sha256,
        expected_commit=args.expected_commit,
        video_path=args.video.resolve(),
        preview_path=args.preview.resolve(),
        receipt_path=args.receipt.resolve(),
    )
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "video": receipt["video"]["path"],
                "video_sha256": receipt["video"]["sha256"],
                "preview": receipt["preview"]["path"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    # The legacy OSMesa/MuJoCo stack can double-free its already-closed context
    # during interpreter teardown. All writers, readers, simulator resources,
    # and the atomic receipt are closed before this point, so bypass only the
    # process-global C-extension destructor path.
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
