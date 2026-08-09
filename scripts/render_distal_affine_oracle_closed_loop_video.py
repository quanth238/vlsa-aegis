#!/usr/bin/env python3
"""Render a single-context, state-hash-verified affine-oracle replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from main.multilink_ellipsoid.oracle_affine_closed_loop import (
    ORACLE_AFFINE_CLOSED_LOOP_RESULT_SCHEMA_V2,
)
from main.multilink_ellipsoid.exact_candidate_closed_loop import (
    EXACT_CANDIDATE_RESULT_SCHEMA,
)
from main.multilink_ellipsoid.exact_candidate_continue import (
    CONTINUE_RESULT_SCHEMA,
)
from main.multilink_ellipsoid.waypoint_candidate_closed_loop import (
    WAYPOINT_RESULT_SCHEMA,
)
from main.multilink_ellipsoid.object_refined_suffix import (
    RESULT_SCHEMA as OBJECT_REFINED_SUFFIX_RESULT_SCHEMA,
)
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _git_identity, _load, _require,
)


CASE_ID = "vlsa-t1-goal-ii-t0-e05"
PAIRING_KEYS = (
    "manifest_row_sha256", "initial_state_sha256", "initial_observation_sha256",
    "settled_simulator_state_sha256",
    "settled_active_obstacle_position_sha256", "policy_noise_schedule_sha256",
)


def _state_sha256(value: Any) -> str:
    return hashlib.sha256(value.tobytes()).hexdigest()


def _frame_statistics(frame: Any) -> dict[str, float]:
    import numpy as np

    array = np.asarray(frame, dtype=np.uint8)
    _require(
        array.ndim == 3 and array.shape[2] == 3,
        "affine-oracle replay source frame is not RGB",
    )
    numeric = array.astype(np.float32)
    adjacent = max(
        float(np.mean(np.abs(np.diff(numeric, axis=1)))),
        float(np.mean(np.abs(np.diff(numeric, axis=0)))),
    )
    standard_deviation = float(np.std(array))
    _require(adjacent <= 12.0, "source frame has striped pixel corruption")
    _require(standard_deviation >= 8.0, "source frame is degenerate")
    return {"adjacent_pixel_mad": adjacent, "rgb_stddev": standard_deviation}


def _verify_video(
    imageio: Any, video_path: Path, source_samples: Mapping[int, Any],
    expected_frames: int,
) -> dict[str, Any]:
    import numpy as np

    reader = imageio.get_reader(str(video_path))
    records = []
    maximum_mae = 0.0
    maximum_adjacent = 0.0
    try:
        decoded_frames = int(reader.count_frames())
        _require(decoded_frames == expected_frames, "decoded frame count differs")
        for index in sorted(source_samples):
            decoded = np.asarray(reader.get_data(index), dtype=np.uint8)
            source = np.asarray(source_samples[index], dtype=np.uint8)
            _require(decoded.shape == source.shape, "decoded frame shape differs")
            mae = float(np.mean(
                np.abs(decoded.astype(np.int16) - source.astype(np.int16))
            ))
            statistics = _frame_statistics(decoded)
            maximum_mae = max(maximum_mae, mae)
            maximum_adjacent = max(
                maximum_adjacent, statistics["adjacent_pixel_mad"]
            )
            _require(mae <= 12.0, "decoded video fidelity differs")
            records.append({
                "frame_index": index,
                "mean_absolute_pixel_error": mae,
                **statistics,
            })
    finally:
        reader.close()
    return {
        "decoded_frame_count": decoded_frames,
        "sample_count": len(records),
        "maximum_mean_absolute_pixel_error": maximum_mae,
        "maximum_adjacent_pixel_mad": maximum_adjacent,
        "samples": records,
    }


def render(
    *, repo_root: Path, manifest_path: Path, accepted_result_path: Path,
    expected_result_sha256: str, expected_commit: str, video_path: Path,
    final_jpg_path: Path, receipt_path: Path, video_fps: int,
    prefix_result_path: Optional[Path] = None,
) -> dict[str, Any]:
    import numpy as np
    from main.evaluate_safelibero_aegis import (
        TABLE_RENDER_RESOLUTION, TABLE_SETTLE_ACTIONS, TABLE_VIDEO_FPS,
        _active_obstacle, _build_environment, _processed_image,
        _runtime_imports, _settle, pairing_record, read_jsonl,
        validate_case_row,
    )
    from main.multilink_ellipsoid.rollout import _dynamic_state_vector
    from main.multilink_ellipsoid.shadow import allocation_record

    accepted = _load(accepted_result_path)
    schema = accepted.get("schema_version")
    exact_candidate = schema == EXACT_CANDIDATE_RESULT_SCHEMA
    unsafe_continuation = schema == CONTINUE_RESULT_SCHEMA
    waypoint_candidate = schema == WAYPOINT_RESULT_SCHEMA
    object_refined_suffix = schema == OBJECT_REFINED_SUFFIX_RESULT_SCHEMA
    accepted_case_id = (
        accepted.get("config", {}).get("primary_case", {}).get("case_id")
        if object_refined_suffix else accepted.get("case_id")
    )
    decision_key = (
        "privileged_object_refined_suffix_e05_go"
        if object_refined_suffix else (
            "privileged_waypoint_best_of_n_closed_loop_e05_go"
            if waypoint_candidate else (
                "privileged_exact_candidate_closed_loop_e05_go"
                if exact_candidate else "privileged_closed_loop_e05_go"
            )
        )
    )
    _require(
        _file_sha256(accepted_result_path) == expected_result_sha256
        and schema in (
            ORACLE_AFFINE_CLOSED_LOOP_RESULT_SCHEMA_V2,
            EXACT_CANDIDATE_RESULT_SCHEMA,
            CONTINUE_RESULT_SCHEMA,
            WAYPOINT_RESULT_SCHEMA,
            OBJECT_REFINED_SUFFIX_RESULT_SCHEMA,
        )
        and accepted.get("status") in ("method_failure", "complete")
        and accepted_case_id == CASE_ID
        and (
            accepted.get("decision", {}).get(
                "continued_after_empty_safe_set"
            ) is True
            if unsafe_continuation
            else accepted.get("decision", {}).get(decision_key) is False
        ),
        "accepted closed-loop result differs",
    )
    suffix_actions = [
        item for item in accepted.get("actions", []) if item.get("executed") is True
    ]
    prefix_accepted = None
    if object_refined_suffix:
        _require(prefix_result_path is not None, "suffix visual prefix is missing")
        prefix_accepted = _load(prefix_result_path)
        prefix_end = int(accepted["suffix"]["start_step"])
        prefix_actions = [
            item for item in prefix_accepted.get("actions", [])
            if item.get("executed") is True and int(item["step"]) < prefix_end
        ]
        _require(
            prefix_accepted.get("schema_version") == WAYPOINT_RESULT_SCHEMA
            and _file_sha256(prefix_result_path)
            == accepted["config"]["prerequisite"]["waypoint_result_file_sha256"]
            and prefix_accepted.get("result_payload_sha256")
            == accepted["config"]["prerequisite"][
                "waypoint_result_payload_sha256"
            ]
            and len(prefix_actions) == prefix_end,
            "suffix visual prefix result differs",
        )
        actions = prefix_actions + suffix_actions
        expected_action_count = prefix_end + int(
            accepted["suffix"]["executed_action_count"]
        )
    else:
        _require(prefix_result_path is None, "unexpected visual prefix result")
        actions = suffix_actions
        expected_action_count = (
            accepted["continuation"]["total_executed_action_count"]
            if unsafe_continuation else accepted["closed_loop"]["action_count"]
        )
    _require(
        len(actions) == expected_action_count
        and len(actions) > 0
        and [int(item["step"]) for item in actions] == list(range(len(actions))),
        "accepted closed-loop action ledger differs",
    )
    _require(
        isinstance(video_fps, int) and 1 <= video_fps <= TABLE_VIDEO_FPS,
        "visual replay FPS differs",
    )
    rows = read_jsonl(manifest_path)
    matches = [item for item in rows if item.get("case_id") == CASE_ID]
    _require(len(matches) == 1, "primary manifest row differs")
    case = matches[0]
    validate_case_row(case, repo_root)
    source = _git_identity(repo_root, expected_commit)
    allocation = allocation_record()
    runtime = _runtime_imports(include_aegis=False)
    for output in (video_path, final_jpg_path, receipt_path):
        _require(not output.exists(), "visual replay output already exists")
        output.parent.mkdir(parents=True, exist_ok=True)
    partial = video_path.with_name(".%s.partial.mp4" % video_path.stem)
    _require(not partial.exists(), "visual replay partial already exists")
    env = writer = None
    source_samples = {}
    source_statistics = []
    maximum_state_error = 0.0
    maximum_obstacle_displacement_error = 0.0
    terminal_frame = None
    try:
        env, task, observation, selected_initial_state = _build_environment(
            runtime, case, render_resolution=TABLE_RENDER_RESOLUTION
        )
        observation = _settle(env, observation, TABLE_SETTLE_ACTIONS)
        obstacle_name, _ = _active_obstacle(env, observation)
        initial_obstacle = np.asarray(
            observation["%s_pos" % obstacle_name], dtype=np.float64
        ).copy()
        pairing = pairing_record(
            case=case, selected_initial_state=selected_initial_state,
            settled_observation=observation, task_description=str(task.language),
            active_obstacle_name=obstacle_name,
            settled_simulator_state=np.asarray(
                env.sim.get_state().flatten(), dtype=np.float64
            ),
        )
        for key in PAIRING_KEYS:
            _require(pairing[key] == accepted["pairing"][key], "pairing differs: %s" % key)
        writer = runtime["imageio"].get_writer(
            str(partial), fps=video_fps, codec="libx264",
            macro_block_size=None, pixelformat="yuv420p",
            output_params=["-crf", "18", "-movflags", "+faststart"],
        )

        def append_frame(index: int) -> None:
            nonlocal terminal_frame
            terminal_frame = _processed_image(observation, "agentview_image")
            statistics = _frame_statistics(terminal_frame)
            source_statistics.append({"frame_index": index, **statistics})
            writer.append_data(terminal_frame)
            sample_indexes = {
                0, min(15, len(actions)), min(60, len(actions)),
                min(120, len(actions)), min(180, len(actions)), len(actions),
            }
            if index in sample_indexes:
                source_samples[index] = terminal_frame.copy()

        append_frame(0)
        for index, record in enumerate(actions):
            action = np.asarray(record["executed_action"], dtype=np.float64)
            _require(action.shape == (7,), "replay action shape differs")
            observation, reward, done, _ = env.step(action.tolist())
            measurement = record["executed_measurement"]
            _require(
                float(reward) == float(measurement["reward"])
                and bool(done) is bool(measurement["done"]),
                "replay reward/done differs",
            )
            state_hash = _state_sha256(_dynamic_state_vector(env))
            _require(
                state_hash == measurement["next_state_sha256"],
                "replay dynamic-state hash differs at action %d" % index,
            )
            displacement = float(np.sum(np.abs(
                np.asarray(
                    observation["%s_pos" % obstacle_name], dtype=np.float64
                ) - initial_obstacle
            )))
            displacement_error = abs(
                displacement - float(record["active_obstacle_l1_displacement_m"])
            )
            maximum_obstacle_displacement_error = max(
                maximum_obstacle_displacement_error, displacement_error
            )
            _require(displacement_error <= 1.0e-10, "replay obstacle trace differs")
            maximum_state_error = max(maximum_state_error, 0.0)
            append_frame(index + 1)
    finally:
        try:
            if writer is not None:
                writer.close()
        finally:
            if env is not None:
                env.close()
    _require(partial.is_file() and partial.stat().st_size > 0, "encoder wrote no video")
    os.replace(partial, video_path)
    _require(terminal_frame is not None, "terminal replay frame missing")
    runtime["imageio"].imwrite(str(final_jpg_path), terminal_frame)
    video_validation = _verify_video(
        runtime["imageio"], video_path, source_samples, len(actions) + 1
    )
    receipt = {
        "schema_version": (
            "vlsa_distal_exact_candidate_continue_e05_video.v1"
            if unsafe_continuation else (
                "vlsa_distal_object_refined_suffix_e05_video.v1"
                if object_refined_suffix else (
                    "vlsa_distal_waypoint_closed_loop_e05_video.v1"
                    if waypoint_candidate else (
                        "vlsa_distal_exact_candidate_closed_loop_e05_video.v1"
                        if exact_candidate
                        else "vlsa_distal_affine_oracle_closed_loop_video.v1"
                    )
                )
            )
        ),
        "status": "verified",
        "scientific_result": False,
        "case_id": CASE_ID,
        "source": source,
        "allocation": allocation,
        "accepted_result": {
            "path": str(accepted_result_path),
            "file_sha256": _file_sha256(accepted_result_path),
            "payload_sha256": accepted["result_payload_sha256"],
            "read_only": True,
        },
        "accepted_prefix_result": (
            None if prefix_accepted is None else {
                "path": str(prefix_result_path),
                "file_sha256": _file_sha256(prefix_result_path),
                "payload_sha256": prefix_accepted["result_payload_sha256"],
                "read_only": True,
            }
        ),
        "pairing_fields_verified": list(PAIRING_KEYS),
        "trace_equivalence": {
            "executed_action_count": len(actions),
            "maximum_dynamic_state_error": maximum_state_error,
            "maximum_obstacle_displacement_error_m": (
                maximum_obstacle_displacement_error
            ),
            "native_task_success": bool(
                accepted["suffix"]["native_task_success"]
                if object_refined_suffix else accepted[
                    "continuation" if unsafe_continuation else "closed_loop"
                ]["native_task_success"]
            ),
            "stopped_before_action": (
                None if unsafe_continuation or object_refined_suffix else (
                    accepted["closed_loop"].get("failure", {}).get("step")
                    if accepted["closed_loop"].get("failure") else None
                )
            ),
        },
        "source_frame_maximum_adjacent_pixel_mad": max(
            item["adjacent_pixel_mad"] for item in source_statistics
        ),
        "video_validation": video_validation,
        "video": {
            "path": str(video_path), "file_sha256": _file_sha256(video_path),
            "frame_count": len(actions) + 1, "fps": video_fps,
        },
        "final_image": {
            "path": str(final_jpg_path),
            "file_sha256": _file_sha256(final_jpg_path),
        },
    }
    _atomic_write(receipt_path, receipt)
    return receipt


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--accepted-result", type=Path, required=True)
    parser.add_argument("--prefix-result", type=Path)
    parser.add_argument("--expected-result-sha256", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--final-jpg", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--video-fps", type=int, default=30)
    args = parser.parse_args(argv)
    receipt = render(
        repo_root=args.repo_root.resolve(), manifest_path=args.manifest.resolve(),
        accepted_result_path=args.accepted_result.resolve(),
        expected_result_sha256=args.expected_result_sha256,
        expected_commit=args.expected_commit, video_path=args.video.resolve(),
        final_jpg_path=args.final_jpg.resolve(), receipt_path=args.receipt.resolve(),
        video_fps=args.video_fps,
        prefix_result_path=(
            None if args.prefix_result is None else args.prefix_result.resolve()
        ),
    )
    print(json.dumps({
        "status": receipt["status"],
        "video": receipt["video"],
        "video_validation": receipt["video_validation"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
