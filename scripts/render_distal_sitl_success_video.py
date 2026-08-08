#!/usr/bin/env python3
"""Render a clean, exact replay of the accepted L5--L7 SITL success ledger."""

from __future__ import annotations

import argparse
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


EXPECTED_RESULT_SCHEMA = "vlsa_distal_sitl_candidate_e05_result.v1"
EXPECTED_ACTION_COUNT = 193
PAIRING_KEYS = (
    "manifest_row_sha256",
    "initial_state_sha256",
    "initial_observation_sha256",
    "settled_simulator_state_sha256",
    "settled_active_obstacle_position_sha256",
    "policy_noise_schedule_sha256",
)


def _event_signature(event: Mapping[str, Any]) -> tuple[str, str, str]:
    obstacle = event.get("obstacle", {})
    other = event.get("other", {})
    return (
        str(obstacle.get("geom_name")),
        str(other.get("body_name")),
        str(other.get("geom_name")),
    )


def _compare_events(
    actual: Sequence[Mapping[str, Any]],
    expected: Sequence[Mapping[str, Any]],
) -> float:
    actual_sorted = sorted(actual, key=_event_signature)
    expected_sorted = sorted(expected, key=_event_signature)
    _require(len(actual_sorted) == len(expected_sorted), "raw contact count differs")
    maximum_distance_error = 0.0
    for observed, recorded in zip(actual_sorted, expected_sorted):
        _require(
            _event_signature(observed) == _event_signature(recorded),
            "raw contact geom identity differs",
        )
        error = abs(float(observed["distance"]) - float(recorded["distance"]))
        maximum_distance_error = max(maximum_distance_error, error)
        _require(error <= 1.0e-10, "raw contact distance differs")
    return maximum_distance_error


def _frame_quality(image: Any) -> dict[str, float]:
    import numpy as np

    value = np.ascontiguousarray(np.asarray(image, dtype=np.uint8))
    _require(value.ndim == 3 and value.shape[2] == 3, "source frame is not RGB")
    numeric = value.astype(np.float32)
    horizontal = float(np.mean(np.abs(np.diff(numeric, axis=1))))
    vertical = float(np.mean(np.abs(np.diff(numeric, axis=0))))
    quality = {
        "horizontal_adjacent_mad": horizontal,
        "vertical_adjacent_mad": vertical,
        "maximum_adjacent_mad": max(horizontal, vertical),
        "rgb_stddev": float(np.std(numeric)),
    }
    _require(quality["maximum_adjacent_mad"] <= 8.0, "source frame has striped pixel corruption")
    _require(quality["rgb_stddev"] >= 8.0, "source frame is visually degenerate")
    return quality


def _annotated_frame(
    image: Any,
    *,
    step: Optional[int],
    action_record: Optional[Mapping[str, Any]],
    slow_repeat: int,
) -> Any:
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    source = np.ascontiguousarray(np.asarray(image, dtype=np.uint8))
    _require(source.ndim == 3 and source.shape[2] == 3, "agentview frame is not RGB")
    banner_height = 126
    canvas = Image.new("RGB", (source.shape[1], source.shape[0] + banner_height), "black")
    canvas.paste(Image.fromarray(source), (0, 0))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 21)
        strong = ImageFont.truetype("DejaVuSans-Bold.ttf", 23)
    except OSError:
        font = ImageFont.load_default()
        strong = font

    if action_record is None:
        lines = (
            ("Verified L5--L7 simulator-in-the-loop replay", "white", strong),
            ("Primary E05 SafeLIBERO case | exact accepted action ledger", "white", font),
            ("Seven distal ellipsoids + original AEGIS EE | no contact", "#66dd99", font),
        )
    else:
        filter_record = action_record["filter"]
        modified = bool(filter_record["modified"])
        correction = float(filter_record["correction_l2"])
        displacement_mm = 1000.0 * float(
            action_record["active_obstacle_l1_displacement_m"]
        )
        slow = " | %dx SLOW" % slow_repeat if slow_repeat > 1 else ""
        task_success = bool(action_record["goal_progress"]["all_satisfied"])
        lines = (
            (
                "L5--L7 verified replay | action %03d / 192%s" % (step, slow),
                "#ffcc66" if modified else "white",
                strong,
            ),
            (
                "%s | correction L2 %.4f | 8 QP rows"
                % ("SITL CORRECTION" if modified else "nominal AEGIS", correction),
                "#ffcc66" if modified else "white",
                font,
            ),
            (
                "%s | no raw robot contact | obstacle move %.4f mm"
                % ("TASK SUCCESS" if task_success else "task active", displacement_mm),
                "#66dd99",
                strong if task_success else font,
            ),
        )
    top = source.shape[0] + 8
    for index, (line, color, selected_font) in enumerate(lines):
        draw.text((16, top + index * 36), line, fill=color, font=selected_font)
    return np.ascontiguousarray(np.asarray(canvas))


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
        records = []
        maximum_mae = 0.0
        maximum_adjacent_mad = 0.0
        for index in sorted(source_samples):
            decoded = np.asarray(reader.get_data(index), dtype=np.uint8)
            source = np.asarray(source_samples[index], dtype=np.uint8)
            _require(decoded.shape == source.shape, "decoded frame shape differs")
            mae = float(np.mean(np.abs(decoded.astype(np.int16) - source.astype(np.int16))))
            numeric = decoded.astype(np.float32)
            adjacent = max(
                float(np.mean(np.abs(np.diff(numeric, axis=1)))),
                float(np.mean(np.abs(np.diff(numeric, axis=0)))),
            )
            maximum_mae = max(maximum_mae, mae)
            maximum_adjacent_mad = max(maximum_adjacent_mad, adjacent)
            _require(mae <= 12.0, "decoded video visual fidelity differs")
            _require(adjacent <= 12.0, "decoded video has striped pixel corruption")
            _require(float(np.std(decoded)) >= 8.0, "decoded video frame is degenerate")
            records.append(
                {
                    "frame_index": index,
                    "mean_absolute_pixel_error": mae,
                    "adjacent_pixel_mad": adjacent,
                }
            )
    finally:
        reader.close()
    return {
        "decoded_frames": decoded_frames,
        "sample_count": len(records),
        "maximum_mean_absolute_pixel_error": maximum_mae,
        "maximum_adjacent_pixel_mad": maximum_adjacent_mad,
        "samples": records,
    }


def render(
    *,
    repo_root: Path,
    manifest_path: Path,
    accepted_result_path: Path,
    expected_result_sha256: str,
    expected_commit: str,
    video_path: Path,
    preview_path: Path,
    final_jpg_path: Path,
    receipt_path: Path,
    fps: int,
    slow_start: int,
    slow_end: int,
    slow_repeat: int,
) -> dict[str, Any]:
    import numpy as np
    from PIL import Image

    from main.evaluate_safelibero_aegis import (
        TABLE_RENDER_RESOLUTION,
        TABLE_SETTLE_ACTIONS,
        _active_obstacle,
        _build_environment,
        _contact_model_authority,
        _detailed_active_obstacle_contacts,
        _goal_progress_definition,
        _goal_progress_snapshot,
        _processed_image,
        _runtime_imports,
        _settle,
        array_sha256,
        pairing_record,
        read_jsonl,
        validate_case_row,
    )
    from main.multilink_ellipsoid.shadow import allocation_record

    outputs = (video_path, preview_path, final_jpg_path, receipt_path)
    _require(video_path.suffix == ".mp4", "video output must be MP4")
    _require(preview_path.suffix.lower() == ".jpg", "preview output must be JPG")
    _require(final_jpg_path.suffix.lower() == ".jpg", "final output must be JPG")
    _require(receipt_path.suffix == ".json", "receipt output must be JSON")
    for output in outputs:
        _require(not output.exists(), "video replay output already exists")
    partial = video_path.with_name(".%s.partial.mp4" % video_path.stem)
    _require(not partial.exists(), "partial video output already exists")

    accepted = _load(accepted_result_path)
    accepted_sha = _file_sha256(accepted_result_path)
    _require(accepted_sha == expected_result_sha256, "accepted result file hash differs")
    _require(accepted.get("schema_version") == EXPECTED_RESULT_SCHEMA, "result schema differs")
    _require(accepted.get("status") == "complete", "accepted result is incomplete")
    _require(accepted.get("case_id") == CASE_ID, "accepted result case differs")
    _require(accepted.get("primary_problem_solved") is True, "accepted success differs")
    actions = accepted.get("actions")
    _require(
        isinstance(actions, list) and len(actions) == EXPECTED_ACTION_COUNT,
        "accepted action ledger length differs",
    )
    _require(isinstance(fps, int) and fps > 0, "fps must be positive")
    _require(0 <= slow_start <= slow_end < len(actions), "slow interval is invalid")
    _require(isinstance(slow_repeat, int) and slow_repeat >= 1, "slow repeat is invalid")

    source = _git_identity(repo_root, expected_commit)
    allocation = allocation_record()
    runtime = _runtime_imports(include_aegis=False)
    rows = read_jsonl(manifest_path)
    matches = [row for row in rows if row.get("case_id") == CASE_ID]
    _require(len(matches) == 1, "video replay manifest row is not unique")
    case = matches[0]
    validate_case_row(case, repo_root)

    for output in outputs:
        output.parent.mkdir(parents=True, exist_ok=True)
    env = None
    writer = None
    source_frames = 0
    encoded_frames = 0
    source_samples: dict[int, Any] = {}
    maximum_source_adjacent_mad = 0.0
    minimum_source_rgb_stddev = float("inf")
    maximum_eef_error_m = 0.0
    maximum_displacement_error_m = 0.0
    maximum_contact_distance_error_m = 0.0
    maximum_displacement_m = 0.0
    native_success_step: Optional[int] = None
    final_frame = None
    try:
        # Keep exactly one rendered MuJoCo environment alive. Creating the probe
        # environments used by the online search can corrupt legacy OSMesa frames.
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
            settled_simulator_state=np.asarray(env.sim.get_state().flatten(), dtype=np.float64),
        )
        for key in PAIRING_KEYS:
            _require(pairing[key] == accepted["pairing"][key], "pairing differs: %s" % key)
        contact_authority = _contact_model_authority(env, obstacle_name)
        _, goal_atoms = _goal_progress_definition(env)
        previous_goal_values = accepted["goal_progress"]["initial"]["values"]
        writer = runtime["imageio"].get_writer(
            str(partial),
            fps=fps,
            codec="libx264",
            macro_block_size=None,
            pixelformat="yuv420p",
            output_params=["-crf", "18", "-movflags", "+faststart"],
        )

        initial_source = _processed_image(observation, "agentview_image")
        initial_quality = _frame_quality(initial_source)
        maximum_source_adjacent_mad = initial_quality["maximum_adjacent_mad"]
        minimum_source_rgb_stddev = initial_quality["rgb_stddev"]
        initial_frame = _annotated_frame(
            initial_source,
            step=None,
            action_record=None,
            slow_repeat=1,
        )
        writer.append_data(initial_frame)
        source_samples[0] = initial_frame.copy()
        source_frames += 1
        encoded_frames += 1

        for index, record in enumerate(actions):
            _require(int(record["step"]) == index, "accepted action index differs")
            executed = np.asarray(record["executed_sitl_action"], dtype=np.float64)
            _require(executed.shape == (7,) and np.all(np.isfinite(executed)), "action differs")
            observation, reward, done, _ = env.step(executed.tolist())
            _require(float(reward) == float(record["reward"]), "reward trace differs")
            _require(bool(done) is bool(record["done"]), "done trace differs")
            _require(
                array_sha256(env.sim.get_state().flatten())
                == record["goal_progress"]["simulator_state_sha256_after"],
                "simulator-state trace differs",
            )

            observed_eef = np.asarray(observation["robot0_eef_pos"], dtype=np.float64)
            recorded_eef = np.asarray(record["eef_position_m"], dtype=np.float64)
            eef_error = float(np.max(np.abs(observed_eef - recorded_eef)))
            maximum_eef_error_m = max(maximum_eef_error_m, eef_error)
            _require(eef_error <= 1.0e-10, "end-effector trace differs")

            displacement = float(
                np.sum(
                    np.abs(
                        np.asarray(observation["%s_pos" % obstacle_name], dtype=np.float64)
                        - initial_obstacle_position
                    )
                )
            )
            displacement_error = abs(
                displacement - float(record["active_obstacle_l1_displacement_m"])
            )
            maximum_displacement_error_m = max(
                maximum_displacement_error_m, displacement_error
            )
            _require(displacement_error <= 1.0e-10, "obstacle displacement trace differs")
            maximum_displacement_m = max(maximum_displacement_m, displacement)

            contacts = _detailed_active_obstacle_contacts(
                env,
                obstacle_name,
                step=index,
                contact_authority=contact_authority,
            )
            _require(contacts["status"] == "available", "raw contact evidence unavailable")
            actual_robot = [
                event
                for event in contacts["events"]
                if event.get("other", {}).get("classification") == "robot"
            ]
            maximum_contact_distance_error_m = max(
                maximum_contact_distance_error_m,
                _compare_events(actual_robot, record["robot_contact_events"]),
            )

            goal = _goal_progress_snapshot(
                env,
                goal_atoms,
                step=index,
                previous_values=previous_goal_values,
            )
            _require(goal["values"] == record["goal_progress"]["values"], "goal trace differs")
            _require(
                goal["simulator_state_sha256_after"]
                == record["goal_progress"]["simulator_state_sha256_after"],
                "goal-state trace differs",
            )
            previous_goal_values = goal["values"]
            if goal["all_satisfied"] and native_success_step is None:
                native_success_step = index

            source_frame = _processed_image(observation, "agentview_image")
            quality = _frame_quality(source_frame)
            maximum_source_adjacent_mad = max(
                maximum_source_adjacent_mad, quality["maximum_adjacent_mad"]
            )
            minimum_source_rgb_stddev = min(
                minimum_source_rgb_stddev, quality["rgb_stddev"]
            )
            repeats = slow_repeat if slow_start <= index <= slow_end else 1
            frame = _annotated_frame(
                source_frame,
                step=index,
                action_record=record,
                slow_repeat=repeats,
            )
            if index == slow_start:
                Image.fromarray(frame).save(preview_path, quality=94, optimize=True)
            for _ in range(repeats):
                writer.append_data(frame)
                if encoded_frames in {60, 120, 180, 200}:
                    source_samples[encoded_frames] = frame.copy()
                encoded_frames += 1
            source_frames += 1
            final_frame = frame

        expected_raw = accepted["raw_simulation_evidence"]
        _require(native_success_step == expected_raw["native_task_success_step"], "success step differs")
        _require(maximum_displacement_m <= PAPER_CAR_THRESHOLD_M, "replay produced CAR")
        _require(
            abs(maximum_displacement_m - expected_raw["maximum_active_obstacle_l1_displacement_m"])
            <= 1.0e-10,
            "maximum obstacle displacement differs",
        )
        _require(final_frame is not None, "final replay frame is missing")
        source_samples[encoded_frames - 1] = final_frame.copy()
        Image.fromarray(final_frame).save(final_jpg_path, quality=94, optimize=True)
    finally:
        try:
            if writer is not None:
                writer.close()
        finally:
            if env is not None:
                env.close()

    _require(partial.is_file() and partial.stat().st_size > 0, "video encoder produced no file")
    os.replace(partial, video_path)
    verification = _verify_video(
        runtime["imageio"],
        video_path,
        source_samples,
        encoded_frames,
    )
    receipt = {
        "schema_version": "vlsa_distal_sitl_success_video.v1",
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
            "maximum_eef_coordinate_error_m": maximum_eef_error_m,
            "maximum_obstacle_displacement_error_m": maximum_displacement_error_m,
            "maximum_contact_distance_error_m": maximum_contact_distance_error_m,
            "maximum_active_obstacle_l1_displacement_m": maximum_displacement_m,
            "robot_contact": False,
            "protected_link_contact": False,
            "paper_car": False,
            "native_task_success": True,
            "native_task_success_step": native_success_step,
        },
        "visual_integrity": {
            "single_rendered_environment": True,
            "maximum_source_adjacent_pixel_mad": maximum_source_adjacent_mad,
            "minimum_source_rgb_stddev": minimum_source_rgb_stddev,
            "decoded_fidelity": verification,
        },
        "video": {
            "path": str(video_path),
            "sha256": _file_sha256(video_path),
            "source_frames": source_frames,
            "encoded_frames": encoded_frames,
            "fps": fps,
            "codec": "libx264_yuv420p_crf18_faststart",
            "slow_motion": {
                "action_start": slow_start,
                "action_end": slow_end,
                "repeat_each_source_frame": slow_repeat,
            },
        },
        "preview": {"path": str(preview_path), "sha256": _file_sha256(preview_path)},
        "final_jpg": {"path": str(final_jpg_path), "sha256": _file_sha256(final_jpg_path)},
    }
    _atomic_write(receipt_path, receipt)
    return receipt


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--accepted-result", type=Path, required=True)
    parser.add_argument("--expected-result-sha256", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--preview", type=Path, required=True)
    parser.add_argument("--final-jpg", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--slow-start", type=int, default=182)
    parser.add_argument("--slow-end", type=int, default=192)
    parser.add_argument("--slow-repeat", type=int, default=5)
    args = parser.parse_args(argv)
    receipt = render(
        repo_root=args.repo_root.resolve(),
        manifest_path=args.manifest.resolve(),
        accepted_result_path=args.accepted_result.resolve(),
        expected_result_sha256=args.expected_result_sha256,
        expected_commit=args.expected_commit,
        video_path=args.video.resolve(),
        preview_path=args.preview.resolve(),
        final_jpg_path=args.final_jpg.resolve(),
        receipt_path=args.receipt.resolve(),
        fps=args.fps,
        slow_start=args.slow_start,
        slow_end=args.slow_end,
        slow_repeat=args.slow_repeat,
    )
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "video": receipt["video"]["path"],
                "video_sha256": receipt["video"]["sha256"],
                "preview": receipt["preview"]["path"],
                "final_jpg": receipt["final_jpg"]["path"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    # Avoid legacy OSMesa process-global destructor corruption after all files
    # and the validation receipt have already been closed atomically.
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
