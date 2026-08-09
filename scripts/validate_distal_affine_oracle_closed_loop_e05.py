#!/usr/bin/env python3
"""Independently validate the privileged affine-oracle closed-loop artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from main.multilink_ellipsoid.oracle_affine_closed_loop import (
    ORACLE_AFFINE_CLOSED_LOOP_RESULT_SCHEMA,
    ORACLE_AFFINE_CLOSED_LOOP_RESULT_SCHEMA_V2,
    ORACLE_AFFINE_CLOSED_LOOP_VALIDATION_SCHEMA,
    ORACLE_AFFINE_CLOSED_LOOP_VALIDATION_SCHEMA_V2,
    load_oracle_affine_closed_loop_config,
)
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _load, _require,
)


def _hash_without(value: Mapping[str, Any], key: str) -> str:
    payload = dict(value)
    payload.pop(key, None)
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _video_frame_count(path: Path) -> tuple[int, list[int]]:
    import imageio.v2 as imageio
    import numpy as np

    reader = imageio.get_reader(str(path))
    count = 0
    shape = None
    try:
        for frame in reader:
            array = np.asarray(frame)
            if shape is None:
                shape = list(array.shape)
            _require(list(array.shape) == shape, "video frame shape changed")
            count += 1
    finally:
        reader.close()
    _require(shape is not None and len(shape) == 3, "video has no RGB frames")
    return count, shape


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    config = load_oracle_affine_closed_loop_config(args.config.resolve())
    qp_constraint_count = len(config["constraint_order"])
    result_schema = (
        ORACLE_AFFINE_CLOSED_LOOP_RESULT_SCHEMA_V2
        if qp_constraint_count == 8 else ORACLE_AFFINE_CLOSED_LOOP_RESULT_SCHEMA
    )
    validation_schema = (
        ORACLE_AFFINE_CLOSED_LOOP_VALIDATION_SCHEMA_V2
        if qp_constraint_count == 8
        else ORACLE_AFFINE_CLOSED_LOOP_VALIDATION_SCHEMA
    )
    result = _load(args.result.resolve())
    _require(
        result.get("schema_version") == result_schema
        and result.get("status") in ("complete", "method_failure")
        and result.get("scientific_result") is True
        and result.get("source", {}).get("commit") == args.expected_commit
        and result.get("source", {}).get("dirty") is False
        and result.get("result_payload_sha256")
        == _hash_without(result, "result_payload_sha256"),
        "closed-loop result identity differs",
    )
    _require(
        result.get("config", {}).get("config_payload_sha256")
        == config["config_payload_sha256"],
        "closed-loop result config differs",
    )
    allocation = result.get("allocation", {})
    _require(
        "H100" in str(allocation.get("device", {}).get("name"))
        and str(allocation.get("slurm_job_id", "")).isdigit(),
        "closed-loop result is not H100 allocation-backed",
    )
    prerequisite = result.get("prerequisite", {})
    _require(
        prerequisite.get("representation_go") is True
        and prerequisite.get("result_file_sha256")
        == config["prerequisite"]["representation_result_file_sha256"]
        and prerequisite.get("validation_file_sha256")
        == config["prerequisite"]["representation_validation_file_sha256"]
        and prerequisite.get("result_payload_sha256")
        == config["prerequisite"]["representation_result_payload_sha256"],
        "closed-loop representation prerequisite differs",
    )
    records = result.get("actions", [])
    _require(isinstance(records, list), "closed-loop actions are invalid")
    executed = [item for item in records if item.get("executed") is True]
    _require(
        [int(item["step"]) for item in executed] == list(range(len(executed))),
        "closed-loop executed action indexes differ",
    )
    maximum_step_motion = float(
        config["activation"]["maximum_per_step_obstacle_l1_displacement_m"]
    )
    executed_safe = True
    clone_match = True
    intervention_valid = True
    for item in records:
        nominal = item.get("nominal_exact_summary", {})
        intervention = bool(item.get("intervention"))
        _require(
            intervention is (not bool(nominal.get("safe_for_execution"))),
            "closed-loop activation decision differs at step %s" % item.get("step"),
        )
        if intervention:
            certificate = item.get("certificate", {})
            qp = item.get("qp", {})
            grid = item.get("grid", {})
            valid = bool(
                grid.get("candidate_count")
                == config["sampling"]["expected_grid_action_count"]
                and grid.get("qp_constraint_count") == qp_constraint_count
                and certificate.get("valid") is True
                and certificate.get("sampled_grid_false_safe_candidate_count") == 0
                and qp.get("valid") is True
                and qp.get("diagnostics", {}).get("input_constraint_count")
                == qp_constraint_count
            )
            if item.get("executed") is True:
                exact = item.get("selected_exact_summary", {})
                valid = bool(valid and exact.get("safe_for_execution") is True)
            intervention_valid = bool(intervention_valid and valid)
        if item.get("executed") is not True:
            continue
        measurement = item.get("executed_measurement", {})
        margins = measurement.get("minimum_all_eight_substep_clearance_m")
        safe = bool(
            isinstance(margins, list) and len(margins) == 8
            and min(float(value) for value in margins) >= 0.0
            and measurement.get("D_opt_seven_distal_safe") is True
            and measurement.get("released_AEGIS_EE_proxy_safe") is True
            and measurement.get("raw_protected_contact_count") == 0
            and float(measurement.get(
                "maximum_within_step_obstacle_l1_displacement_m", 1.0
            )) <= maximum_step_motion
        )
        executed_safe = bool(executed_safe and safe)
        clone_match = bool(
            clone_match and item.get("executed_next_state_matches_exact_clone") is True
            and measurement.get("next_state_sha256")
            == item.get("selected_exact_summary", {}).get(
                "first_transition_next_state_sha256"
            )
        )
    closed = result.get("closed_loop", {})
    _require(
        closed.get("action_count") == len(executed)
        and closed.get("intervention_count")
        == sum(bool(item.get("intervention")) for item in records)
        and closed.get("modified_action_count")
        == sum(bool(item.get("modified")) for item in executed),
        "closed-loop count summary differs",
    )
    video = result.get("video", {})
    video_path = Path(str(video.get("path", ""))).resolve()
    image = result.get("final_image", {})
    image_path = Path(str(image.get("path", ""))).resolve()
    _require(
        video_path.is_file() and _file_sha256(video_path) == video.get("file_sha256")
        and image_path.is_file() and _file_sha256(image_path) == image.get("file_sha256"),
        "closed-loop visual artifact identity differs",
    )
    frame_count, frame_shape = _video_frame_count(video_path)
    _require(
        frame_count == len(executed) + 1 == video.get("frame_count"),
        "closed-loop video frame count differs",
    )
    task_success = bool(closed.get("native_task_success"))
    no_contact = closed.get("first_protected_contact_step") is None
    no_car = closed.get("first_paper_CAR_step") is None
    expected_go = bool(
        result.get("closed_loop", {}).get("failure") is None
        and task_success and no_contact and no_car and executed_safe
        and clone_match and intervention_valid
    )
    _require(
        closed.get("primary_problem_solved") is expected_go
        and result.get("decision", {}).get("privileged_closed_loop_e05_go")
        is expected_go
        and result.get("decision", {}).get("neural_training_authorized") is False
        and result.get("decision", {}).get("deployable_method_demonstrated") is False,
        "closed-loop final decision differs",
    )
    output = {
        "schema_version": validation_schema,
        "status": "validated",
        "scientific_result": False,
        "expected_commit": args.expected_commit,
        "result_file_sha256": _file_sha256(args.result.resolve()),
        "result_payload_sha256": result["result_payload_sha256"],
        "executed_action_count": len(executed),
        "intervention_count": sum(bool(item.get("intervention")) for item in records),
        "all_executed_substeps_all_eight_safe": executed_safe,
        "all_executed_next_states_match_clone": clone_match,
        "all_completed_interventions_have_valid_registered_row_qp": intervention_valid,
        "qp_constraint_count": qp_constraint_count,
        "native_task_success": task_success,
        "zero_protected_contact": no_contact,
        "zero_paper_CAR": no_car,
        "video_frame_count": frame_count,
        "video_frame_shape": frame_shape,
        "privileged_closed_loop_e05_go": expected_go,
        "neural_training_authorized": False,
        "deployable_method_demonstrated": False,
    }
    _atomic_write(args.output.resolve(), output)
    print(json.dumps(output, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
