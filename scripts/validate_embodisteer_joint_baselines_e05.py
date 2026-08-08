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
    _frame_orientation,
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
    config = result["config"]
    guided = config["schema_version"] == "vlsa_embodisteer_aegis_ee_pair.v1"
    isolation = result["geometry_isolation"]
    expected_isolation = {
            "l5_l6_l7_ellipsoids_constructed": False,
            "collision_sdf_queried": False,
            "barrier_constraints_built": guided,
            "qp_solved": guided,
            "physical_obstacle_remains_in_scene": True,
    }
    if guided:
        expected_isolation.update(
            {
                "barrier_constraint_count_per_action": 1,
                "protected_geometry": "released_aegis_end_effector_proxy_only",
            }
        )
    _require(isolation == expected_isolation, "geometry isolation differs")
    _require(
        result["sampler_regression_preflight"]["status"] == "passing",
        "sampler regression did not pass",
    )
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
            arm["barrier_projection_enabled"] is guided
            and arm["ellipsoid_constraints_enabled"] is guided
            and arm["qp_enabled"] is guided,
            "%s arm guidance state differs" % arm_name,
        )
        _require(arm["action_count"] == len(arm["actions"]), "action count differs")
        _require(0 < arm["action_count"] <= 300, "action horizon differs")
        _require(
            arm["visual_integrity"]["all_frames_passing"] is True,
            "%s source frames failed visual integrity" % arm_name,
        )
        settled_hashes.add(arm["pairing"]["settled_simulator_state_sha256"])
        for query in arm["policy_queries"]:
            _require(
                query["collision_geometry_queried"] is guided,
                "query geometry state differs",
            )
            _require(
                query["barrier_qp_solved"] is guided,
                "query QP state differs",
            )
            if arm_name in {
                "joint_denoising_no_guidance",
                "joint_denoising_with_aegis_ee",
            }:
                _require(len(query["flow_steps"]) == 10, "joint flow horizon differs")
        absolute_joint_arm = (
            config["schema_version"]
            in {
                "vlsa_embodisteer_joint_baselines.v2",
                "vlsa_embodisteer_joint_baselines.v3",
            }
            and arm_name == "joint_denoising_no_guidance"
        ) or (guided and arm_name == "joint_denoising_with_aegis_ee")
        if absolute_joint_arm:
            _require(
                arm["joint_target_execution"][
                    "steps_with_delta_encoding_saturation"
                ]
                == 0,
                "v2 absolute joint targets saturated",
            )
        if guided:
            geometry = arm["aegis_ee_geometry"]
            _require(
                geometry["constraint_count"] == 1
                and geometry["protected_body_names"]
                == ["robot0_end_effector"]
                and geometry["l5_l6_l7_ellipsoids_constructed"] is False
                and geometry["l5_l6_l7_constraints_built"] is False,
                "AEGIS-EE geometry isolation differs",
            )
            binding = geometry["source_binding"]
            _require(
                binding["settled_simulator_state_sha256_match"] is True
                and binding["camera_bytes_expected_to_match"] is False
                and float(binding["active_obstacle_position_max_error_m"])
                <= float(binding["active_obstacle_position_tolerance_m"]),
                "AEGIS-EE archived geometry source binding differs",
            )
            _require(
                arm["aegis_ee_qp_timing"]["qp_count"] == arm["action_count"],
                "AEGIS-EE QP count differs",
            )
            for action in arm["actions"]:
                safety = action["aegis_ee_constraint"]
                _require(
                    safety["constraint_count"] == 1
                    and safety["l5_l6_l7_constraints_enabled"] is False,
                    "action did not use exactly one EE constraint",
                )
                _require(
                    safety["qp"]["solver_status"]
                    in {"optimal", "optimal_inaccurate"},
                    "AEGIS-EE QP did not solve",
                )
                _require(
                    float(safety["qp"]["constraint_lhs"]) >= -1.0e-5,
                    "AEGIS-EE solution violates its optimizer row",
                )
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
        maximum_decoded_adjacent_mad = 0.0
        minimum_decoded_upright_margin = None
        upright_reference = None
        reader = imageio.get_reader(str(video_path))
        try:
            for frame in reader:
                value = np.asarray(frame)
                _require(value.ndim == 3 and value.shape[2] == 3, "decoded frame differs")
                numeric = value.astype(np.float32)
                adjacent = max(
                    float(np.mean(np.abs(np.diff(numeric, axis=1)))),
                    float(np.mean(np.abs(np.diff(numeric, axis=0)))),
                )
                maximum_decoded_adjacent_mad = max(
                    maximum_decoded_adjacent_mad, adjacent
                )
                if upright_reference is None:
                    upright_reference = value.copy()
                orientation = _frame_orientation(value, upright_reference)
                margin = (
                    orientation["rotated_reference_mad"]
                    - orientation["upright_reference_mad"]
                )
                minimum_decoded_upright_margin = (
                    margin
                    if minimum_decoded_upright_margin is None
                    else min(minimum_decoded_upright_margin, margin)
                )
                _require(
                    orientation["passing"],
                    "decoded video has a flipped or ambiguous camera orientation",
                )
                decoded += 1
                last_shape = list(value.shape)
        finally:
            reader.close()
        _require(decoded == arm["video"]["frames"], "decoded frame count differs")
        _require(
            maximum_decoded_adjacent_mad <= 8.0,
            "decoded video has high-frequency pixel corruption",
        )
        jpg = np.asarray(imageio.imread(str(jpg_path)))
        _require(jpg.ndim == 3 and jpg.shape[2] == 3, "decoded JPG differs")
        jpg_numeric = jpg.astype(np.float32)
        jpg_adjacent = max(
            float(np.mean(np.abs(np.diff(jpg_numeric, axis=1)))),
            float(np.mean(np.abs(np.diff(jpg_numeric, axis=0)))),
        )
        _require(jpg_adjacent <= 8.0, "JPG has high-frequency pixel corruption")
        video_receipts[arm_name] = {
            "video_sha256": arm["video"]["sha256"],
            "decoded_frames": decoded,
            "decoded_frame_shape": last_shape,
            "maximum_decoded_adjacent_mad": maximum_decoded_adjacent_mad,
            "minimum_decoded_upright_margin": minimum_decoded_upright_margin,
            "jpg_sha256": arm["final_jpg"]["sha256"],
            "jpg_shape": list(jpg.shape),
            "jpg_adjacent_mad": jpg_adjacent,
        }
    _require(len(settled_hashes) == 1, "paired settled states differ")
    receipt = {
        "schema_version": "vlsa_embodisteer_joint_baseline_validation.v1",
        "status": "passing",
        "case_id": CASE_ID,
        "result_path": str(result_path.resolve()),
        "result_file_sha256": _file_sha256(result_path),
        "result_payload_sha256": claimed_payload,
        "all_guidance_disabled": not guided,
        "exactly_one_aegis_ee_constraint": guided,
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
