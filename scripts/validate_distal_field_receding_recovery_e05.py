#!/usr/bin/env python3
"""Independently validate the four-arm executable E05 field recovery run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Sequence


ARMS = (
    "fixed_analytical_softmin_repulsion",
    "nonnegative_normal_field_mixture",
    "normal_plus_task_tangent_mixture",
    "unrestricted_six_dimensional_correction",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


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


def validate(root: Path, archived_path: Path, expected_commit: str) -> dict[str, Any]:
    import imageio.v2 as imageio
    import numpy as np

    archived = json.loads(archived_path.read_text())
    archived_actions = archived["actions"]
    records = {}
    for arm in ARMS:
        arm_root = root / arm
        result_path = arm_root / "result.json"
        result = json.loads(result_path.read_text())
        published_payload = result["result_payload_sha256"]
        payload = dict(result)
        payload.pop("result_payload_sha256")
        _require(
            hashlib.sha256(_canonical(payload)).hexdigest() == published_payload,
            "%s result payload hash differs" % arm,
        )
        _require(
            result["source"]["git_commit"] == expected_commit
            and not result["source"]["dirty"],
            "%s source identity differs" % arm,
        )
        _require(
            result["hybrid_recovery"]["activation_step"] == 185,
            "%s did not activate at action 185" % arm,
        )
        _require(
            result["initial_field_recovery"]["record"]["arm"] == arm,
            "%s field record differs" % arm,
        )
        _require(
            abs(
                float(
                    result["initial_field_recovery"]["record"][
                        "nominal_two_action_rollout"
                    ]["minimum_row_m"]
                )
                + 0.009313103935574107
            )
            <= 5.0e-10,
            "%s dangerous nominal margin differs" % arm,
        )
        _require(
            float(
                result["initial_field_recovery"]["record"][
                    "fresh_selected_verification"
                ]["minimum_row_m"]
            )
            >= 0.0,
            "%s fresh field verification is unsafe" % arm,
        )
        actions = result["actions"]
        _require(len(actions) == int(result["action_count"]), "%s action count differs" % arm)
        for step in range(185):
            _require(
                np.array_equal(
                    np.asarray(actions[step]["executed_sitl_action"]),
                    np.asarray(archived_actions[step]["env_step_input"]),
                ),
                "%s immutable prefix differs at %d" % (arm, step),
            )
        summary = result["filter_summary"]
        _require(
            summary["all_executed_transitions_match_clone"],
            "%s executed clone verification differs" % arm,
        )
        evidence = result["raw_simulation_evidence"]
        _require(
            evidence["all_robot_active_obstacle_contact_pass"]
            and evidence["protected_link_contact_pass"]
            and evidence["paper_car_pass"],
            "%s raw safety evidence failed" % arm,
        )
        _require(
            not evidence["native_task_success"]
            and not result["primary_problem_solved"],
            "%s scientific verdict differs" % arm,
        )
        video_path = Path(result["video"]["path"])
        _require(video_path == arm_root / "episode.mp4", "%s video path differs" % arm)
        _require(
            _file_sha256(video_path) == result["video"]["file_sha256"],
            "%s video hash differs" % arm,
        )
        reader = imageio.get_reader(str(video_path))
        frame_count = 0
        first_shape = None
        try:
            for frame in reader:
                array = np.asarray(frame)
                if first_shape is None:
                    first_shape = list(array.shape)
                _require(
                    list(array.shape) == first_shape and array.ndim == 3,
                    "%s video frame shape differs" % arm,
                )
                frame_count += 1
        finally:
            reader.close()
        _require(
            frame_count == int(result["video"]["frames_written"])
            == len(actions) + 1,
            "%s decoded frame count differs" % arm,
        )
        records[arm] = {
            "status": result["status"],
            "action_count": len(actions),
            "failure_step": (
                None if result["failure"] is None else int(result["failure"]["step"])
            ),
            "nominal_minimum_mm": 1000.0
            * float(
                result["initial_field_recovery"]["record"][
                    "nominal_two_action_rollout"
                ]["minimum_row_m"]
            ),
            "fresh_verified_minimum_mm": 1000.0
            * float(
                result["initial_field_recovery"]["record"][
                    "fresh_selected_verification"
                ]["minimum_row_m"]
            ),
            "correction_l2": float(
                result["initial_field_recovery"]["record"]["search"]["best"][
                    "correction_l2"
                ]
            ),
            "task_progress_ratio": float(
                result["initial_field_recovery"]["record"]["search"]["best"][
                    "task_progress_ratio"
                ]
            ),
            "contact_free": True,
            "paper_car_pass": True,
            "native_task_success": False,
            "video_sha256": result["video"]["file_sha256"],
            "decoded_video_frame_count": frame_count,
            "result_payload_sha256": published_payload,
        }
    return {
        "schema_version": "vlsa_distal_field_receding_recovery_e05_validation.v1",
        "status": "valid",
        "scientific_result": True,
        "source_commit": expected_commit,
        "producer_root": str(root),
        "arms": records,
        "all_arms_prevented_robot_obstacle_contact": True,
        "all_arms_failed_native_task_completion": True,
        "normal_plus_tangent_beats_fixed_task_success": False,
        "mlp_training_gate_pass": False,
        "interpretation": (
            "late_two_action_posthoc_repairs_prevent_collision_but_destroy_"
            "task_completion_no_field_mixer_training_justified"
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--archived", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = validate(
        args.root.resolve(), args.archived.resolve(), args.expected_commit
    )
    receipt["validation_payload_sha256"] = hashlib.sha256(
        _canonical(receipt)
    ).hexdigest()
    _write_atomic(args.output.resolve(), receipt)
    print(json.dumps(receipt, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
