#!/usr/bin/env python3
"""Render qualitative E39 videos where frozen gradient beats/loses to random."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from scripts.replay_distal_three_ellipsoid_multicbf import (
    _file_sha256, _git_identity, _load, _require,
)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def _atomic_write(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode()
    temporary = path.with_name(".%s.tmp" % path.name)
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def comparison_specs(risks: Mapping[str, float]) -> list[dict[str, Any]]:
    """Freeze one clear random win and one clear gradient win."""
    gradient = float(risks["gradient_down"])
    specs = [
        {
            "comparison_id": "gradient_beats_random",
            "left": "gradient_down", "right": "matched_random_3",
            "winner": "gradient_down",
        },
        {
            "comparison_id": "gradient_loses_to_random",
            "left": "gradient_down", "right": "matched_random_1",
            "winner": "matched_random_1",
        },
    ]
    _require(
        gradient < float(risks["matched_random_3"])
        and gradient > float(risks["matched_random_1"]),
        "gradient video comparison ordering differs",
    )
    return specs


def _font(size: int, *, bold: bool = False) -> Any:
    from PIL import ImageFont

    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(name, size)
    except OSError:
        return ImageFont.load_default()


def _compose(
    left: Any, right: Any, *, comparison: Mapping[str, Any],
    risks: Mapping[str, float], frame_index: int, context_frame_count: int,
    total_frame_count: int,
) -> Any:
    import numpy as np
    from PIL import Image, ImageDraw

    left_image = Image.fromarray(np.asarray(left, dtype=np.uint8)).convert("RGB")
    right_image = Image.fromarray(np.asarray(right, dtype=np.uint8)).convert("RGB")
    _require(left_image.size == right_image.size, "gradient video frame sizes differ")
    width, height = left_image.size
    top = 76
    bottom = 94
    canvas = Image.new("RGB", (2 * width, top + height + bottom), "#101318")
    canvas.paste(left_image, (0, top))
    canvas.paste(right_image, (width, top))
    draw = ImageDraw.Draw(canvas)
    title_font = _font(21, bold=True)
    text_font = _font(17)
    small_font = _font(15)
    phase = (
        "shared archived context"
        if frame_index < context_frame_count
        else "counterfactual action %d / 5"
        % (frame_index - context_frame_count + 1)
    )
    draw.text(
        (14, 8),
        "%s | %s" % (comparison["comparison_id"].replace("_", " ").upper(), phase),
        fill="white", font=title_font,
    )
    draw.text(
        (14, 41),
        "Same E39 state, same OSC, equal correction norm | lower Q is safer",
        fill="#c9d1d9", font=text_font,
    )
    names = (str(comparison["left"]), str(comparison["right"]))
    for panel, name in enumerate(names):
        x0 = panel * width
        winner = name == comparison["winner"]
        color = "#2ecc71" if winner else "#f1c40f"
        draw.rectangle(
            (x0 + 2, top + 2, x0 + width - 3, top + height - 3),
            outline=color, width=5,
        )
        draw.rectangle(
            (x0, top + height, x0 + width, top + height + bottom),
            fill="#151a21",
        )
        label = "NEGATIVE GRADIENT" if name == "gradient_down" else name.upper()
        draw.text((x0 + 12, top + height + 9), label, fill=color, font=title_font)
        draw.text(
            (x0 + 12, top + height + 40),
            "exact Q = %+0.6f  |  SAFE" % float(risks[name]),
            fill="white", font=text_font,
        )
        draw.text(
            (x0 + 12, top + height + 68),
            "winner" if winner else "higher exact risk",
            fill=color, font=small_font,
        )
    draw.line((width, top, width, top + height + bottom), fill="#59636e", width=2)
    draw.text(
        (2 * width - 92, 45), "%02d/%02d" % (frame_index + 1, total_frame_count),
        fill="#8b949e", font=small_font,
    )
    return np.ascontiguousarray(np.asarray(canvas, dtype=np.uint8))


def _video_fidelity(imageio: Any, path: Path, expected_frames: int) -> dict[str, Any]:
    import numpy as np

    reader = imageio.get_reader(str(path))
    try:
        count = int(reader.count_frames())
        _require(count == expected_frames, "gradient video decoded frame count differs")
        samples = []
        for index in sorted({0, expected_frames // 2, expected_frames - 1}):
            frame = np.asarray(reader.get_data(index), dtype=np.uint8)
            _require(float(np.std(frame)) >= 8.0, "gradient video frame is degenerate")
            samples.append({
                "frame_index": int(index),
                "rgb_stddev": float(np.std(frame)),
                "sha256": hashlib.sha256(frame.tobytes()).hexdigest(),
            })
    finally:
        reader.close()
    return {"decoded_frame_count": count, "samples": samples}


def render(
    *, repo_root: Path, config_path: Path, manifest_path: Path,
    prep_path: Path, result_path: Path, validation_path: Path,
    expected_prep_sha256: str, expected_result_sha256: str,
    expected_validation_sha256: str, accepted_result_commit: str,
    expected_commit: str, output_root: Path,
) -> dict[str, Any]:
    import numpy as np
    from PIL import Image

    from main.evaluate_safelibero_aegis import (
        TABLE_RENDER_RESOLUTION, TABLE_SETTLE_ACTIONS, _active_obstacle,
        _build_environment,
        _processed_image, _runtime_imports, _settle, pairing_record,
        read_jsonl, validate_case_row,
    )
    from main.multilink_ellipsoid.rollout import (
        _auxiliary_sim_snapshot, _base_env, _controller_snapshot,
        _dynamic_state_vector, _restore_auxiliary_sim_snapshot,
        _restore_controller_snapshot,
    )
    from main.multilink_ellipsoid.shadow import allocation_record
    from main.multilink_ellipsoid.tight_prefix_early_gradient_probe import (
        CASE_SCHEMA, PREP_SCHEMA, VALIDATION_SCHEMA, load_config, payload_sha256,
    )

    config = load_config(config_path)
    selected = config["cases"][0]
    case_id = str(selected["case_id"])
    _require(case_id == "vlsa-t1-goal-ii-t0-e39", "gradient video case differs")
    for path, expected, label in (
        (prep_path, expected_prep_sha256, "prep"),
        (result_path, expected_result_sha256, "result"),
        (validation_path, expected_validation_sha256, "validation"),
    ):
        _require(_file_sha256(path) == expected, "gradient video %s file differs" % label)
    prep = _load(prep_path)
    result = _load(result_path)
    validation = _load(validation_path)
    _require(
        prep.get("schema_version") == PREP_SCHEMA
        and result.get("schema_version") == CASE_SCHEMA
        and validation.get("schema_version") == VALIDATION_SCHEMA
        and prep.get("source", {}).get("commit") == accepted_result_commit
        and result.get("source", {}).get("commit") == accepted_result_commit
        and validation.get("accepted_result_commit") == accepted_result_commit
        and prep.get("result_payload_sha256")
        == payload_sha256(prep, "result_payload_sha256")
        and result.get("result_payload_sha256")
        == payload_sha256(result, "result_payload_sha256")
        and validation.get("validation_payload_sha256")
        == payload_sha256(validation, "validation_payload_sha256")
        and result.get("case_id") == case_id
        and result.get("record", {}).get("eligible") is True,
        "gradient video artifact binding differs",
    )
    risks = {
        str(name): float(value)
        for name, value in result["record"]["exact_probe_primary_risk"].items()
    }
    comparisons = comparison_specs(risks)
    exact_by_name = {
        str(item["name"]): item for item in result["exact_rollout"]["candidates"]
    }
    candidate_names = sorted({
        str(spec[key]) for spec in comparisons for key in ("left", "right")
    })
    _require(set(candidate_names).issubset(exact_by_name), "gradient video action set differs")

    table1_root = Path(config["source"]["table1_root"])
    archived_path = table1_root / selected["archived_result_relative_path"]
    _require(
        _file_sha256(archived_path) == selected["archived_result_file_sha256"],
        "gradient video archive differs",
    )
    archived = _load(archived_path)
    rows = [row for row in read_jsonl(manifest_path) if row.get("case_id") == case_id]
    _require(len(rows) == 1, "gradient video manifest case differs")
    case = rows[0]
    validate_case_row(case, repo_root)
    output_root.mkdir(parents=True, exist_ok=False)
    runtime = _runtime_imports(include_aegis=False)
    source = _git_identity(repo_root, expected_commit)
    allocation = allocation_record()
    env = None
    writers = []
    try:
        env, task, observation, initial_state = _build_environment(
            runtime, case, render_resolution=TABLE_RENDER_RESOLUTION,
        )
        observation = _settle(env, observation, TABLE_SETTLE_ACTIONS)
        obstacle_name, _ = _active_obstacle(env, observation)
        pairing = pairing_record(
            case=case, selected_initial_state=initial_state,
            settled_observation=observation, task_description=str(task.language),
            active_obstacle_name=obstacle_name,
            settled_simulator_state=np.asarray(env.sim.get_state().flatten()),
        )
        for key in (
            "manifest_row_sha256", "initial_state_sha256",
            "initial_observation_sha256", "settled_simulator_state_sha256",
            "settled_active_obstacle_position_sha256",
            "policy_noise_schedule_sha256",
        ):
            _require(pairing[key] == archived["pairing"][key],
                     "gradient video pairing differs: %s" % key)
        action_rows = {int(item["step"]): item for item in archived["actions"]}
        state_step = int(selected["early_state_step"])
        context_start = state_step - 10
        _require(context_start >= 0, "gradient video context start differs")
        for step in range(context_start):
            observation, _, _, _ = env.step(action_rows[step]["executed"])
        context_frames = [_processed_image(observation, "agentview_image")]
        for step in range(context_start, state_step):
            observation, _, _, _ = env.step(action_rows[step]["executed"])
            context_frames.append(_processed_image(observation, "agentview_image"))
        source_dynamic = np.asarray(_dynamic_state_vector(env), dtype=np.float64)
        source_hash = hashlib.sha256(source_dynamic.tobytes()).hexdigest()
        _require(
            source_hash == prep["exact_case"]["source_snapshot_sha256"],
            "gradient video source snapshot differs",
        )
        source_state = np.asarray(env.sim.get_state().flatten(), dtype=np.float64).copy()
        source_auxiliary = _auxiliary_sim_snapshot(env)
        source_controller = _controller_snapshot(env)
        base = _base_env(env)
        source_clock = (int(base.timestep), float(base.cur_time), bool(base.done))
        source_frame = np.asarray(context_frames[-1], dtype=np.uint8)

        def restore() -> None:
            env.sim.set_state_from_flattened(source_state)
            env.sim.forward()
            _restore_auxiliary_sim_snapshot(env, source_auxiliary)
            _restore_controller_snapshot(env, source_controller)
            base.timestep, base.cur_time, base.done = source_clock
            _require(
                np.array_equal(_dynamic_state_vector(env), source_dynamic),
                "gradient video state restore differs",
            )

        candidate_frames = {}
        candidate_state_hashes = {}
        for name in candidate_names:
            restore()
            frames = [source_frame]
            hashes = [source_hash]
            actions = np.asarray(
                exact_by_name[name]["source_executed_actions"], dtype=np.float64,
            )
            _require(actions.shape == (5, 7), "gradient video action shape differs")
            for action in actions:
                observation, _, _, _ = env.step(action.tolist())
                frames.append(_processed_image(observation, "agentview_image"))
                state = np.asarray(_dynamic_state_vector(env), dtype=np.float64)
                hashes.append(hashlib.sha256(state.tobytes()).hexdigest())
            candidate_frames[name] = frames
            candidate_state_hashes[name] = hashes

        video_records = []
        repeat = 3
        fps = 9
        for comparison in comparisons:
            comparison_id = str(comparison["comparison_id"])
            left_name = str(comparison["left"])
            right_name = str(comparison["right"])
            raw_pairs = [
                (frame, frame) for frame in context_frames
            ] + list(zip(
                candidate_frames[left_name][1:], candidate_frames[right_name][1:],
            ))
            annotated = [
                _compose(
                    left, right, comparison=comparison, risks=risks,
                    frame_index=index,
                    context_frame_count=len(context_frames),
                    total_frame_count=len(raw_pairs),
                )
                for index, (left, right) in enumerate(raw_pairs)
            ]
            path = output_root / (comparison_id + ".mp4")
            partial = output_root / (comparison_id + ".partial.mp4")
            preview = output_root / (comparison_id + ".jpg")
            writer = runtime["imageio"].get_writer(
                str(partial), fps=fps, codec="libx264", macro_block_size=None,
                pixelformat="yuv420p",
                output_params=["-crf", "18", "-movflags", "+faststart"],
            )
            writers.append(writer)
            for frame in annotated:
                for _ in range(repeat):
                    writer.append_data(frame)
            writer.close()
            writers.pop()
            partial.replace(path)
            Image.fromarray(annotated[-1]).save(preview, quality=94, optimize=True)
            expected_frames = len(annotated) * repeat
            video_records.append({
                **comparison,
                "left_exact_risk": risks[left_name],
                "right_exact_risk": risks[right_name],
                "context_action_count": 10,
                "counterfactual_action_count": 5,
                "source_frame_count": len(annotated),
                "encoded_frame_count": expected_frames,
                "fps": fps,
                "video_path": str(path),
                "video_file_sha256": _file_sha256(path),
                "preview_path": str(preview),
                "preview_file_sha256": _file_sha256(preview),
                "fidelity": _video_fidelity(runtime["imageio"], path, expected_frames),
            })
        receipt = {
            "schema_version": "vlsa_tight_prefix_early_gradient_videos.v1",
            "status": "complete_qualitative_gradient_comparison_videos",
            "scientific_result": False,
            "claim_scope": (
                "Qualitative exact-action visualization of the already validated "
                "ADR-0209 E39 probes; no new scientific comparison."
            ),
            "source": source, "allocation": allocation,
            "accepted_result_commit": accepted_result_commit,
            "accepted_prep_file_sha256": expected_prep_sha256,
            "accepted_result_file_sha256": expected_result_sha256,
            "accepted_validation_file_sha256": expected_validation_sha256,
            "case_id": case_id, "early_state_step": state_step,
            "source_snapshot_sha256": source_hash,
            "candidate_state_hashes": candidate_state_hashes,
            "exact_primary_risks": risks,
            "videos": video_records,
            "new_policy_query_count": 0, "training_performed": False,
            "scientific_probe_rollout_performed": False,
        }
        receipt["receipt_payload_sha256"] = hashlib.sha256(
            _canonical(receipt)
        ).hexdigest()
        _atomic_write(output_root / "video_receipt.json", receipt)
        return receipt
    finally:
        for writer in writers:
            writer.close()
        if env is not None:
            env.close()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--prep", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--expected-prep-sha256", required=True)
    parser.add_argument("--expected-result-sha256", required=True)
    parser.add_argument("--expected-validation-sha256", required=True)
    parser.add_argument("--accepted-result-commit", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = render(
        repo_root=args.repo_root.resolve(), config_path=args.config.resolve(),
        manifest_path=args.manifest.resolve(), prep_path=args.prep.resolve(),
        result_path=args.result.resolve(), validation_path=args.validation.resolve(),
        expected_prep_sha256=args.expected_prep_sha256,
        expected_result_sha256=args.expected_result_sha256,
        expected_validation_sha256=args.expected_validation_sha256,
        accepted_result_commit=args.accepted_result_commit,
        expected_commit=args.expected_commit,
        output_root=args.output_root.resolve(),
    )
    print(json.dumps({
        "videos": receipt["videos"],
        "receipt_payload_sha256": receipt["receipt_payload_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
