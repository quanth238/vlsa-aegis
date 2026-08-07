#!/usr/bin/env python3
"""Render and verify the accepted active three-CBF action ledger on H100."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence


CASE_ID = "vlsa-t1-goal-ii-t0-e05"
EXPECTED_RESULT_SCHEMA = "vlsa_distal_three_ellipsoid_multicbf_replay.v1"
EXPECTED_ACTION_COUNT = 237
PAPER_CAR_THRESHOLD_M = 0.001
PROTECTED_BODY_NAMES = {"robot0_link5", "robot0_link6", "robot0_link7"}
PAIRING_KEYS = (
    "manifest_row_sha256",
    "initial_state_sha256",
    "initial_observation_sha256",
    "settled_simulator_state_sha256",
    "settled_active_obstacle_position_sha256",
    "policy_noise_schedule_sha256",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), "input JSON is missing or symlinked")
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), "input JSON must contain one object")
    return value


def _atomic_write(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode(
        "utf-8"
    ) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".%s.%d.tmp" % (path.name, os.getpid()))
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _git_identity(root: Path, expected_commit: str) -> dict[str, Any]:
    def run(*arguments: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(root)] + list(arguments),
            stderr=subprocess.STDOUT,
            text=True,
        ).strip()

    commit = run("rev-parse", "HEAD")
    _require(commit == expected_commit, "video replay source commit differs")
    _require(not run("status", "--short"), "video replay source tree is dirty")
    return {
        "commit": commit,
        "dirty": False,
        "branch": run("branch", "--show-current"),
    }


def _event_signature(event: Mapping[str, Any]) -> tuple[str, str, str]:
    obstacle = event.get("obstacle", {})
    other = event.get("other", {})
    return (
        str(obstacle.get("geom_name")),
        str(other.get("body_name")),
        str(other.get("geom_name")),
    )


def _protected_events(events: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [
        event
        for event in events
        if event.get("other", {}).get("body_name") in PROTECTED_BODY_NAMES
    ]


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


def _annotated_frame(
    image: Any,
    *,
    step: int | None,
    action_record: Mapping[str, Any] | None,
    slow_repeat: int,
) -> Any:
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    source = np.asarray(image, dtype=np.uint8)
    _require(source.ndim == 3 and source.shape[2] == 3, "agentview frame is not RGB")
    banner_height = 112
    canvas = Image.new("RGB", (source.shape[1], source.shape[0] + banner_height), "black")
    canvas.paste(Image.fromarray(source), (0, 0))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 22)
        strong = ImageFont.truetype("DejaVuSans-Bold.ttf", 24)
    except OSError:
        font = ImageFont.load_default()
        strong = font

    if action_record is None:
        lines = [
            ("Three-ellipsoid multi-CBF | accepted H100 action-ledger replay", "white", strong),
            ("Initial paired SafeLIBERO state | task: put the bowl on the plate", "white", font),
            ("L5 / L6 / L7 protected | optimizer clearance D_opt = 10 mm", "white", font),
        ]
    else:
        multicbf = action_record["multicbf"]
        h_mm = 1000.0 * float(multicbf["minimum_h_opt_m"])
        correction = float(multicbf["active_correction_l2"])
        displacement_mm = 1000.0 * float(
            action_record["active_obstacle_l1_displacement_m"]
        )
        contacts = action_record["protected_link_contact_events"]
        slow = " | %dx SLOW" % slow_repeat if slow_repeat > 1 else ""
        lines = [
            (
                "Three-ellipsoid multi-CBF | action %03d / 236%s" % (step, slow),
                "white",
                strong,
            ),
            (
                "min h_opt = %+7.2f mm | XYZ correction L2 = %.4f" % (h_mm, correction),
                "#ffcc66" if h_mm < 0.0 else "white",
                font,
            ),
        ]
        if contacts:
            depth_mm = 1000.0 * min(float(item["distance"]) for item in contacts)
            lines.append(
                (
                    "RAW MUJOCO LINK-5 CONTACT = %+.3f mm | obstacle L1 move = %.2f mm"
                    % (depth_mm, displacement_mm),
                    "#ff5555",
                    strong,
                )
            )
        elif displacement_mm > 1.0:
            lines.append(
                (
                    "PAPER CAR FAIL | obstacle L1 move = %.2f mm (> 1.00 mm)"
                    % displacement_mm,
                    "#ff5555",
                    strong,
                )
            )
        else:
            lines.append(
                (
                    "No raw L5-L7 contact | obstacle L1 move = %.2f mm" % displacement_mm,
                    "#66dd99",
                    font,
                )
            )
    top = source.shape[0] + 8
    for index, (line, color, selected_font) in enumerate(lines):
        draw.text((16, top + index * 33), line, fill=color, font=selected_font)
    return np.ascontiguousarray(np.asarray(canvas))


def render(
    *,
    repo_root: Path,
    manifest_path: Path,
    accepted_result_path: Path,
    expected_result_sha256: str,
    expected_commit: str,
    video_path: Path,
    receipt_path: Path,
    fps: int,
    slow_start: int,
    slow_end: int,
    slow_repeat: int,
) -> dict[str, Any]:
    import numpy as np

    from main.evaluate_safelibero_aegis import (
        TABLE_RENDER_RESOLUTION,
        TABLE_SETTLE_ACTIONS,
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
    from main.multilink_ellipsoid.shadow import allocation_record

    _require(video_path.suffix == ".mp4", "video output must be MP4")
    _require(receipt_path.suffix == ".json", "receipt output must be JSON")
    _require(not video_path.exists(), "video output already exists")
    _require(not receipt_path.exists(), "video receipt already exists")
    partial = video_path.with_name(".%s.partial.mp4" % video_path.stem)
    _require(not partial.exists(), "partial video output already exists")
    accepted = _load(accepted_result_path)
    accepted_sha = _file_sha256(accepted_result_path)
    _require(accepted_sha == expected_result_sha256, "accepted result file hash differs")
    _require(accepted.get("schema_version") == EXPECTED_RESULT_SCHEMA, "result schema differs")
    _require(accepted.get("status") == "complete", "accepted result is not complete")
    _require(accepted.get("case_id") == CASE_ID, "accepted result case differs")
    _require(accepted.get("primary_problem_solved") is False, "accepted outcome differs")
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

    video_path.parent.mkdir(parents=True, exist_ok=True)
    env = None
    writer = None
    source_frames = 0
    encoded_frames = 0
    maximum_eef_error_m = 0.0
    maximum_displacement_error_m = 0.0
    maximum_contact_distance_error_m = 0.0
    first_protected_contact_step: int | None = None
    first_car_step: int | None = None
    native_success_step: int | None = None
    maximum_displacement_m = 0.0
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
        contact_authority = _contact_model_authority(env, obstacle_name)
        writer = runtime["imageio"].get_writer(
            str(partial),
            fps=fps,
            codec="libx264",
            macro_block_size=None,
        )
        writer.append_data(
            _annotated_frame(
                _processed_image(observation, "agentview_image"),
                step=None,
                action_record=None,
                slow_repeat=1,
            )
        )
        source_frames += 1
        encoded_frames += 1

        for index, record in enumerate(actions):
            _require(int(record["step"]) == index, "accepted action index differs")
            executed = np.asarray(record["executed_multicbf_action"], dtype=np.float64)
            _require(executed.shape == (7,) and np.all(np.isfinite(executed)), "action differs")
            observation, reward, done, _ = env.step(executed)
            _require(float(reward) == float(record["reward"]), "reward trace differs")
            _require(bool(done) is bool(record["done"]), "done trace differs")
            if done and native_success_step is None:
                native_success_step = index

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
            if first_car_step is None and displacement > PAPER_CAR_THRESHOLD_M:
                first_car_step = index

            contacts = _detailed_active_obstacle_contacts(
                env,
                obstacle_name,
                step=index,
                contact_authority=contact_authority,
            )
            _require(contacts["status"] == "available", "raw contact evidence unavailable")
            actual_protected = _protected_events(contacts["events"])
            expected_protected = record["protected_link_contact_events"]
            maximum_contact_distance_error_m = max(
                maximum_contact_distance_error_m,
                _compare_events(actual_protected, expected_protected),
            )
            if actual_protected and first_protected_contact_step is None:
                first_protected_contact_step = index

            repeats = slow_repeat if slow_start <= index <= slow_end else 1
            frame = _annotated_frame(
                _processed_image(observation, "agentview_image"),
                step=index,
                action_record=record,
                slow_repeat=repeats,
            )
            for _ in range(repeats):
                writer.append_data(frame)
                encoded_frames += 1
            source_frames += 1

        expected_raw = accepted["raw_simulation_evidence"]
        _require(
            first_protected_contact_step == expected_raw["first_protected_link_contact_step"],
            "first protected contact step differs",
        )
        _require(first_car_step == expected_raw["first_paper_car_step"], "first CAR step differs")
        _require(
            native_success_step == expected_raw["native_task_success_step"],
            "native task success step differs",
        )
        _require(
            abs(maximum_displacement_m - expected_raw["maximum_active_obstacle_l1_displacement_m"])
            <= 1.0e-10,
            "maximum obstacle displacement differs",
        )
    finally:
        try:
            if writer is not None:
                writer.close()
        finally:
            if env is not None:
                env.close()

    _require(partial.is_file() and partial.stat().st_size > 0, "video encoder produced no file")
    os.replace(partial, video_path)
    reader = runtime["imageio"].get_reader(str(video_path))
    try:
        decoded_frames = int(reader.count_frames())
    finally:
        reader.close()
    _require(decoded_frames == encoded_frames, "decoded video frame count differs")
    receipt: dict[str, Any] = {
        "schema_version": "vlsa_distal_three_ellipsoid_multicbf_video.v1",
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
            "first_protected_link_contact_step": first_protected_contact_step,
            "first_paper_car_step": first_car_step,
            "native_task_success_step": native_success_step,
            "maximum_active_obstacle_l1_displacement_m": maximum_displacement_m,
        },
        "video": {
            "path": str(video_path),
            "sha256": _file_sha256(video_path),
            "source_frames": source_frames,
            "encoded_frames": encoded_frames,
            "decoded_frames": decoded_frames,
            "fps": fps,
            "slow_motion": {
                "action_start": slow_start,
                "action_end": slow_end,
                "repeat_each_source_frame": slow_repeat,
            },
            "annotation_source": "accepted_result_plus_verified_post_step_raw_simulator_trace",
        },
    }
    _atomic_write(receipt_path, receipt)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--accepted-result", type=Path, required=True)
    parser.add_argument("--expected-result-sha256", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--slow-start", type=int, default=178)
    parser.add_argument("--slow-end", type=int, default=196)
    parser.add_argument("--slow-repeat", type=int, default=4)
    args = parser.parse_args(argv)
    receipt = render(
        repo_root=args.repo_root.resolve(),
        manifest_path=args.manifest.resolve(),
        accepted_result_path=args.accepted_result.resolve(),
        expected_result_sha256=args.expected_result_sha256,
        expected_commit=args.expected_commit,
        video_path=args.video.resolve(),
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
                "receipt": str(args.receipt.resolve()),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
