#!/usr/bin/env python3
"""Validate paired EmbodiSteer baseline result and decode every video frame."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from scripts.evaluate_embodisteer_joint_baselines_e05 import (
    CASE_ID,
    _atomic_write,
    _canonical,
    _file_sha256,
    _require,
    _sha256,
)


def validate(result_path: Path, output_path: Path) -> dict[str, Any]:
    import imageio.v2 as imageio
    import numpy as np

    result = json.loads(result_path.read_text(encoding="utf-8"))
    _require(
        result.get("schema_version")
        == "vlsa_embodisteer_joint_baseline_pair_result.v1",
        "pair schema differs",
    )
    _require(result.get("status") == "complete", "pair result incomplete")
    _require(result.get("case_id") == CASE_ID, "pair case differs")
    claimed_payload = result.pop("result_payload_sha256")
    _require(_sha256(_canonical(result)) == claimed_payload, "pair payload hash differs")
    result["result_payload_sha256"] = claimed_payload
    isolation = result["geometry_isolation"]
    _require(
        isolation
        == {
            "l5_l6_l7_ellipsoids_constructed": False,
            "collision_sdf_queried": False,
            "barrier_constraints_built": False,
            "qp_solved": False,
            "physical_obstacle_remains_in_scene": True,
        },
        "geometry isolation differs",
    )
    _require(
        result["sampler_regression_preflight"]["status"] == "passing",
        "sampler regression did not pass",
    )
    config = result["config"]
    _require(
        config["nominal_policy"]["same_checkpoint_both_arms"] is True,
        "baseline checkpoints are not paired",
    )
    arms = result["arms"]
    _require(set(arms) == set(config["arms"]), "baseline arms differ")
    settled_hashes = set()
    video_receipts = {}
    for arm_name in config["arms"]:
        arm = arms[arm_name]
        _require(arm["status"] == "complete", "%s arm incomplete" % arm_name)
        _require(
            arm["barrier_projection_enabled"] is False
            and arm["ellipsoid_constraints_enabled"] is False
            and arm["qp_enabled"] is False,
            "%s arm applied forbidden guidance" % arm_name,
        )
        _require(arm["action_count"] == len(arm["actions"]), "action count differs")
        _require(0 < arm["action_count"] <= 300, "action horizon differs")
        settled_hashes.add(arm["pairing"]["settled_simulator_state_sha256"])
        for query in arm["policy_queries"]:
            _require(query["collision_geometry_queried"] is False, "query used geometry")
            _require(query["barrier_qp_solved"] is False, "query solved barrier QP")
            if arm_name == "joint_denoising_no_guidance":
                _require(len(query["flow_steps"]) == 10, "joint flow horizon differs")
        evidence = arm["raw_simulation_evidence"]
        observed_protected = next(
            (
                action["step"]
                for action in arm["actions"]
                if action["protected_link_contact_events"]
            ),
            None,
        )
        _require(
            observed_protected == evidence["first_protected_link_contact_step"],
            "protected contact summary differs",
        )
        _require(
            (arm["goal_progress"]["summary"]["first_all_satisfied_step"] is not None)
            == evidence["native_task_success"],
            "native success summary differs",
        )
        video_path = result_path.parent / arm["video"]["path"]
        jpg_path = result_path.parent / arm["final_jpg"]["path"]
        _require(_file_sha256(video_path) == arm["video"]["sha256"], "video hash differs")
        _require(_file_sha256(jpg_path) == arm["final_jpg"]["sha256"], "JPG hash differs")
        decoded = 0
        last_shape = None
        reader = imageio.get_reader(str(video_path))
        try:
            for frame in reader:
                value = np.asarray(frame)
                _require(value.ndim == 3 and value.shape[2] == 3, "decoded frame differs")
                decoded += 1
                last_shape = list(value.shape)
        finally:
            reader.close()
        _require(decoded == arm["video"]["frames"], "decoded frame count differs")
        jpg = np.asarray(imageio.imread(str(jpg_path)))
        _require(jpg.ndim == 3 and jpg.shape[2] == 3, "decoded JPG differs")
        video_receipts[arm_name] = {
            "video_sha256": arm["video"]["sha256"],
            "decoded_frames": decoded,
            "decoded_frame_shape": last_shape,
            "jpg_sha256": arm["final_jpg"]["sha256"],
            "jpg_shape": list(jpg.shape),
        }
    _require(len(settled_hashes) == 1, "paired settled states differ")
    receipt = {
        "schema_version": "vlsa_embodisteer_joint_baseline_validation.v1",
        "status": "passing",
        "case_id": CASE_ID,
        "result_path": str(result_path.resolve()),
        "result_file_sha256": _file_sha256(result_path),
        "result_payload_sha256": claimed_payload,
        "all_guidance_disabled": True,
        "same_checkpoint_and_settled_state": True,
        "videos": video_receipts,
        "comparison": result["comparison"],
    }
    receipt["receipt_payload_sha256"] = _sha256(_canonical(receipt))
    _atomic_write(output_path, receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = validate(args.result.resolve(), args.output.resolve())
    print(json.dumps(receipt["comparison"], sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
