#!/usr/bin/env python3
"""Validate the early E05 five-action detour and one-step composition."""

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


def _validate_result(
    root: Path,
    archived: dict[str, Any],
    expected_commit: str,
    *,
    expected_mode: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import imageio.v2 as imageio
    import numpy as np

    result = json.loads((root / "result.json").read_text())
    published = result["result_payload_sha256"]
    payload = dict(result)
    payload.pop("result_payload_sha256")
    _require(
        hashlib.sha256(_canonical(payload)).hexdigest() == published,
        "%s result hash differs" % expected_mode,
    )
    _require(
        result["source"]["commit"] == expected_commit
        and not result["source"]["dirty"],
        "%s source differs" % expected_mode,
    )
    actions = result["actions"]
    for step in range(182):
        _require(
            np.array_equal(
                np.asarray(actions[step]["action"], dtype=np.float64),
                np.asarray(
                    archived["actions"][step]["env_step_input"], dtype=np.float64
                ),
            ),
            "%s prefix differs at %d" % (expected_mode, step),
        )
    best = result["detour"]["best"]
    _require(result["detour"]["gate_pass"], "%s detour gate failed" % expected_mode)
    _require(
        abs(float(result["detour"]["nominal"]["minimum_clearance_m"]) + 0.009313103935574219)
        <= 5.0e-10,
        "%s nominal margin differs" % expected_mode,
    )
    _require(
        float(best["exact"]["minimum_clearance_m"]) >= 0.001,
        "%s detour buffer differs" % expected_mode,
    )
    _require(
        np.max(np.abs(np.asarray(best["endpoint_correction_sum"]))) <= 1.0e-10,
        "%s endpoint correction differs" % expected_mode,
    )
    _require(
        float(best["terminal_eef_error_m"]) <= 0.015
        and float(best["task_progress_ratio"]) >= 0.5,
        "%s task rejoin gate differs" % expected_mode,
    )
    for step in range(182, 187):
        record = actions[step]
        _require(
            record["source"] == "verified_five_action_task_rejoining_detour"
            and float(record["clone_state_max_abs_error"]) <= 1.0e-10,
            "%s executed detour clone differs" % expected_mode,
        )
    video_path = root / "episode.mp4"
    _require(
        _file_sha256(video_path) == result["video"]["file_sha256"],
        "%s video hash differs" % expected_mode,
    )
    reader = imageio.get_reader(str(video_path))
    frame_count = 0
    shape = None
    try:
        for frame in reader:
            array = np.asarray(frame)
            if shape is None:
                shape = tuple(array.shape)
            _require(tuple(array.shape) == shape, "%s video shape differs" % expected_mode)
            frame_count += 1
    finally:
        reader.close()
    _require(
        frame_count == len(actions) + 1 == int(result["video"]["frames_written"]),
        "%s frame count differs" % expected_mode,
    )
    record = {
        "result_payload_sha256": published,
        "video_sha256": result["video"]["file_sha256"],
        "decoded_video_frame_count": frame_count,
        "nominal_minimum_mm": 1000.0
        * float(result["detour"]["nominal"]["minimum_clearance_m"]),
        "detour_minimum_mm": 1000.0
        * float(best["exact"]["minimum_clearance_m"]),
        "correction_l2": float(best["correction_l2"]),
        "terminal_eef_error_mm": 1000.0 * float(best["terminal_eef_error_m"]),
        "task_progress_ratio": float(best["task_progress_ratio"]),
        "action_count": len(actions),
        "first_robot_contact_step": result["raw_simulation_evidence"][
            "first_robot_contact_step"
        ],
        "first_paper_car_step": result["raw_simulation_evidence"][
            "first_paper_car_step"
        ],
        "native_task_success": result["raw_simulation_evidence"][
            "native_task_success"
        ],
        "native_task_success_step": result["raw_simulation_evidence"][
            "native_task_success_step"
        ],
        "failure": result["failure"],
    }
    return result, record


def validate(
    *,
    detour_only_root: Path,
    receding_root: Path,
    archived_path: Path,
    detour_only_commit: str,
    receding_commit: str,
    validator_commit: str,
) -> dict[str, Any]:
    archived = json.loads(archived_path.read_text())
    detour_result, detour = _validate_result(
        detour_only_root,
        archived,
        detour_only_commit,
        expected_mode="detour_only",
    )
    receding_result, receding = _validate_result(
        receding_root,
        archived,
        receding_commit,
        expected_mode="detour_plus_one_step_filter",
    )
    _require(
        detour["native_task_success"]
        and detour["native_task_success_step"] == 283
        and detour["first_robot_contact_step"] == 197
        and detour["first_paper_car_step"] == 199,
        "detour-only outcome differs",
    )
    _require(
        not receding["native_task_success"]
        and receding["first_robot_contact_step"] is None
        and receding["first_paper_car_step"] is None
        and receding["failure"] == {
            "component": "post_detour_receding_filter",
            "reason": "no_exactly_verified_candidate",
            "step": 187,
        },
        "receding outcome differs",
    )
    _require(
        detour_result["detour"]["best"]["correction"]
        == receding_result["detour"]["best"]["correction"],
        "paired detour differs",
    )
    return {
        "schema_version": "vlsa_distal_five_action_detour_e05_validation.v1",
        "status": "valid",
        "scientific_result": True,
        "validator_source_commit": validator_commit,
        "detour_only": detour,
        "detour_plus_one_step_filter": receding,
        "primary_problem_solved": False,
        "interpretation": (
            "early_five_action_detour_preserves_task_but_collision_recurs_"
            "while_one_step_filter_has_no_post_detour_support"
        ),
        "next_gate": (
            "receding_five_action_task_rejoining_detour_not_learning_or_"
            "stronger_one_step_repulsion"
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detour-only-root", type=Path, required=True)
    parser.add_argument("--receding-root", type=Path, required=True)
    parser.add_argument("--archived", type=Path, required=True)
    parser.add_argument("--detour-only-commit", required=True)
    parser.add_argument("--receding-commit", required=True)
    parser.add_argument("--validator-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = validate(
        detour_only_root=args.detour_only_root.resolve(),
        receding_root=args.receding_root.resolve(),
        archived_path=args.archived.resolve(),
        detour_only_commit=args.detour_only_commit,
        receding_commit=args.receding_commit,
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
