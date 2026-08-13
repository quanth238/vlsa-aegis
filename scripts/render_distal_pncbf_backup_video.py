#!/usr/bin/env python3
"""Render a validated backup-policy action ledger in one MuJoCo environment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from scripts.replay_distal_three_ellipsoid_multicbf import (
    CASE_ID, _file_sha256, _git_identity, _require,
)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _atomic_write(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode()
    temporary = path.with_name(".%s.tmp" % path.name)
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def render(
    *, repo_root: Path, manifest_path: Path, result_path: Path,
    expected_result_sha256: str, expected_commit: str, output_root: Path,
) -> dict[str, Any]:
    import numpy as np
    from PIL import Image

    from main.evaluate_safelibero_aegis import (
        TABLE_RENDER_RESOLUTION, TABLE_SETTLE_ACTIONS, TABLE_VIDEO_FPS,
        _active_obstacle, _build_environment, _processed_image,
        _runtime_imports, _settle, array_sha256, pairing_record, read_jsonl,
        validate_case_row,
    )
    from main.multilink_ellipsoid.rollout import _dynamic_state_vector
    from main.multilink_ellipsoid.shadow import allocation_record

    _require(_file_sha256(result_path) == expected_result_sha256, "validated result file differs")
    result = json.loads(result_path.read_text())
    _require(result["schema_version"] == "vlsa_distal_pncbf_backup_oracle_e05_result.v1", "result schema differs")
    _require(result["case_id"] == CASE_ID and result["scientific_result"] is True, "result identity differs")
    actions = result["actions"]
    _require(actions and [item["step"] for item in actions] == list(range(len(actions))), "action ledger differs")
    rows = [row for row in read_jsonl(manifest_path) if row.get("case_id") == CASE_ID]
    _require(len(rows) == 1, "manifest case differs")
    case = rows[0]
    validate_case_row(case, repo_root)
    source = _git_identity(repo_root, expected_commit)
    allocation = allocation_record()
    runtime = _runtime_imports(include_aegis=False)
    output_root.mkdir(parents=True, exist_ok=False)
    partial = output_root / "episode.partial.mp4"
    video = output_root / "episode.mp4"
    preview = output_root / "preview.jpg"
    receipt_path = output_root / "video_receipt.json"
    env = writer = None
    try:
        env, task, observation, initial_state = _build_environment(
            runtime, case, render_resolution=TABLE_RENDER_RESOLUTION
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
            _require(pairing[key] == result["pairing"][key], "render pairing differs: %s" % key)
        writer = runtime["imageio"].get_writer(
            str(partial), fps=TABLE_VIDEO_FPS, codec="libx264",
            macro_block_size=None, pixelformat="yuv420p",
            output_params=["-crf", "18", "-movflags", "+faststart"],
        )
        frame = _processed_image(observation, "agentview_image")
        writer.append_data(frame)
        frame_hashes = [array_sha256(frame)]
        maximum_state_error = 0.0
        for record in actions:
            command = np.asarray(record["action"], dtype=np.float64)
            observation, _, done, _ = env.step(command.tolist())
            state_hash = array_sha256(_dynamic_state_vector(env))
            _require(state_hash == record["next_state_sha256"], "render replay state differs")
            _require(bool(done) == bool(record["done"]), "render replay task signal differs")
            frame = _processed_image(observation, "agentview_image")
            writer.append_data(frame)
            if int(record["step"]) % 60 == 0:
                frame_hashes.append(array_sha256(frame))
        writer.close(); writer = None
        partial.replace(video)
        Image.fromarray(frame).save(preview, quality=92, optimize=True)
        receipt = {
            "schema_version": "vlsa_distal_pncbf_backup_video_receipt.v1",
            "status": "validated",
            "scientific_result": False,
            "source": source,
            "allocation": allocation,
            "accepted_result_file_sha256": expected_result_sha256,
            "accepted_result_payload_sha256": result["result_payload_sha256"],
            "action_count": len(actions),
            "frames_written": len(actions) + 1,
            "state_hash_match_count": len(actions),
            "maximum_state_error": maximum_state_error,
            "sampled_frame_sha256": frame_hashes,
            "video": {"path": str(video), "file_sha256": _file_sha256(video)},
            "preview": {"path": str(preview), "file_sha256": _file_sha256(preview)},
        }
        receipt["receipt_payload_sha256"] = hashlib.sha256(_canonical(receipt)).hexdigest()
        _atomic_write(receipt_path, receipt)
        return receipt
    finally:
        if writer is not None:
            writer.close()
        if env is not None:
            env.close()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--expected-result-sha256", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = render(
        repo_root=args.repo_root.resolve(), manifest_path=args.manifest.resolve(),
        result_path=args.result.resolve(),
        expected_result_sha256=args.expected_result_sha256,
        expected_commit=args.expected_commit,
        output_root=args.output_root.resolve(),
    )
    print(json.dumps(receipt, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
