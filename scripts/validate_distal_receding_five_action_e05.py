#!/usr/bin/env python3
"""Independently validate the E05 receding five-action oracle result."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Sequence


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_atomic(path: Path, value: Any) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".%s.tmp" % path.name)
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def validate(
    *,
    run_root: Path,
    archived_path: Path,
    producer_commit: str,
    validator_commit: str,
) -> dict[str, Any]:
    import imageio.v2 as imageio
    import numpy as np

    result = json.loads((run_root / "result.json").read_text())
    published = result["result_payload_sha256"]
    payload = dict(result)
    payload.pop("result_payload_sha256")
    _require(
        hashlib.sha256(_canonical(payload)).hexdigest() == published,
        "result payload hash differs",
    )
    _require(
        result["source"]["commit"] == producer_commit
        and not result["source"]["dirty"],
        "producer source differs",
    )
    _require(
        result["schema_version"] == "vlsa_distal_receding_five_action_e05_result.v1"
        and result["status"] == "method_failure"
        and result["scientific_result"] is True
        and result["primary_problem_solved"] is False,
        "terminal classification differs",
    )
    archived = json.loads(archived_path.read_text())
    actions = result["actions"]
    for step in range(182):
        _require(
            np.array_equal(
                np.asarray(actions[step]["action"], dtype=np.float64),
                np.asarray(
                    archived["actions"][step]["env_step_input"], dtype=np.float64
                ),
            ),
            "archived prefix differs at %d" % step,
        )
    windows = result["windows"]
    _require(
        [int(record["step"]) for record in windows] == [182, 183, 184, 185],
        "receding window schedule differs",
    )
    _require(
        all(int(record["executed_prefix_actions"]) == 1 for record in windows[:3])
        and int(windows[3]["executed_prefix_actions"]) == 0,
        "execute-one contract differs",
    )
    first = windows[0]
    _require(
        first["nominal_safe"] is False
        and first["search"]["gate_pass"] is True
        and first["selected_source"] == "fresh_exact_task_rejoining_detour_window"
        and abs(float(first["nominal"]["minimum_clearance_m"]) + 0.009313103935574219)
        <= 5.0e-10
        and float(first["selected"]["minimum_clearance_m"]) >= 0.001,
        "initial corrected window differs",
    )
    _require(
        windows[1]["nominal_safe"] is True
        and windows[2]["nominal_safe"] is True
        and windows[1]["selected_source"] == "fresh_exact_safe_nominal_window"
        and windows[2]["selected_source"] == "fresh_exact_safe_nominal_window",
        "safe nominal windows differ",
    )
    failure = windows[3]
    exact_minima = [
        float(record["exact"]["exact"]["minimum_clearance_m"])
        for record in failure["search"]["history"]
        if record["exact"] is not None
    ]
    _require(
        failure["nominal_safe"] is False
        and failure["search"]["gate_pass"] is False
        and failure["selected_source"] is None
        and len(exact_minima) == 5
        and max(exact_minima) < 0.001,
        "action-185 no-support evidence differs",
    )
    _require(
        result["failure"]
        == {
            "component": "receding_five_action_oracle",
            "reason": "no_verified_task_rejoining_five_action_candidate",
            "step": 185,
        },
        "failure record differs",
    )
    evidence = result["raw_simulation_evidence"]
    _require(
        evidence["first_robot_contact_step"] is None
        and evidence["first_protected_link_contact_step"] is None
        and evidence["first_paper_car_step"] is None
        and evidence["native_task_success"] is False,
        "physical outcome differs",
    )
    _require(
        all(
            float(record.get("clone_state_max_abs_error", 0.0)) <= 1.0e-10
            for record in actions[182:]
        ),
        "clone/execution equality differs",
    )
    video_path = run_root / "episode.mp4"
    _require(
        _file_sha256(video_path) == result["video"]["file_sha256"],
        "video hash differs",
    )
    reader = imageio.get_reader(str(video_path))
    frame_count = 0
    shape = None
    try:
        for frame in reader:
            array = np.asarray(frame)
            if shape is None:
                shape = tuple(array.shape)
            _require(tuple(array.shape) == shape, "video shape differs")
            frame_count += 1
    finally:
        reader.close()
    _require(
        frame_count == len(actions) + 1 == int(result["video"]["frames_written"]),
        "video frame count differs",
    )
    return {
        "schema_version": "vlsa_distal_receding_five_action_e05_validation.v1",
        "status": "valid",
        "scientific_result": True,
        "validator_source_commit": validator_commit,
        "producer_result_payload_sha256": published,
        "video_sha256": result["video"]["file_sha256"],
        "decoded_video_frame_count": frame_count,
        "initial_nominal_minimum_mm": 1000.0
        * float(first["nominal"]["minimum_clearance_m"]),
        "initial_corrected_minimum_mm": 1000.0
        * float(first["selected"]["minimum_clearance_m"]),
        "deadlock_step": 185,
        "deadlock_nominal_minimum_mm": 1000.0
        * float(failure["nominal"]["minimum_clearance_m"]),
        "best_deadlock_candidate_minimum_mm": 1000.0 * max(exact_minima),
        "window_count": len(windows),
        "action_count": len(actions),
        "primary_problem_solved": False,
        "interpretation": "receding_five_action_exact_oracle_has_no_buffered_support_at_action185",
        "next_gate": "intervene_before_182_or_expand_task_rejoining_horizon_or_candidate_family_before_learning",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--archived", type=Path, required=True)
    parser.add_argument("--producer-commit", required=True)
    parser.add_argument("--validator-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = validate(
        run_root=args.run_root.resolve(),
        archived_path=args.archived.resolve(),
        producer_commit=args.producer_commit,
        validator_commit=args.validator_commit,
    )
    receipt["validation_payload_sha256"] = hashlib.sha256(
        _canonical(receipt)
    ).hexdigest()
    _write_atomic(args.output.resolve(), receipt)
    print(json.dumps(receipt, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
