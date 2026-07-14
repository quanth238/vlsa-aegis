#!/usr/bin/env python3
"""Validate complete R00 artifacts and freeze the reach-progress threshold."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from crfs_harness.artifacts import (
    atomic_write_json,
    content_hash,
    file_sha256,
    load_json,
    validate_jsonl_unique,
)
from crfs_harness.manifest import validate_case
from crfs_oracle.progress_calibration import validate_reach_calibration_result
from crfs_oracle.reach_progress import (
    ReachRolloutAnnotation,
    calibrate_reach_p_min,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--evaluation-manifest", required=True)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--minimum-positive-examples", type=int, default=50)
    args = parser.parse_args()

    cases, manifest_errors = validate_jsonl_unique(args.manifest, "case_id")
    evaluation, evaluation_errors = validate_jsonl_unique(args.evaluation_manifest, "case_id")
    for case in cases:
        manifest_errors.extend(f"{case.get('case_id', '<unknown>')}: {error}" for error in validate_case(case))
    for case in evaluation:
        evaluation_errors.extend(
            f"{case.get('case_id', '<unknown>')}: {error}" for error in validate_case(case)
        )
    if manifest_errors or evaluation_errors:
        raise SystemExit("invalid manifest: " + "; ".join(manifest_errors + evaluation_errors))

    calibration_groups = {str(case["group_id"]) for case in cases}
    evaluation_groups = {str(case["group_id"]) for case in evaluation}
    overlapping_groups = sorted(calibration_groups & evaluation_groups)
    if overlapping_groups:
        raise SystemExit(f"calibration/evaluation group leakage: {overlapping_groups}")

    root = Path(args.results_root)
    records = []
    missing = []
    invalid = []
    result_hashes = []
    for case in cases:
        path = root / str(case["case_id"]) / "reach-calibration.json"
        if not path.is_file():
            missing.append(str(case["case_id"]))
            continue
        value = load_json(path)
        errors = validate_reach_calibration_result(value)
        if value.get("case_id") != case["case_id"]:
            errors.append("artifact case_id does not match manifest")
        if value.get("provenance", {}).get("group_id") != case["group_id"]:
            errors.append("artifact group_id does not match manifest")
        if errors:
            invalid.append({"case_id": case["case_id"], "errors": errors})
            continue
        records.append(value)
        result_hashes.append({"case_id": case["case_id"], "sha256": file_sha256(path)})
    if missing or invalid or len(records) != len(cases):
        raise SystemExit(
            f"R00 results are not validation-ready: missing={missing}, invalid={invalid}, "
            f"valid={len(records)}/{len(cases)}"
        )

    config_hashes = sorted({str(value["config_hash"]) for value in records})
    checkpoint_hashes = sorted(
        {str(value["provenance"]["checkpoint_sha256"]) for value in records}
    )
    git_commits = sorted({str(value["provenance"]["git_commit"]) for value in records})
    if len(config_hashes) != 1 or len(checkpoint_hashes) != 1 or len(git_commits) != 1:
        raise SystemExit(
            "R00 artifacts mix config/checkpoint/code identities: "
            f"config={config_hashes}, checkpoint={checkpoint_hashes}, code={git_commits}"
        )

    eligible = [value for value in records if value["trial"]["eligible_for_p_min"]]
    annotations = [ReachRolloutAnnotation(**value["trial"]["reach"]) for value in eligible]
    calibration = None
    failure = None
    try:
        calibration = calibrate_reach_p_min(
            annotations,
            minimum_positive_examples=args.minimum_positive_examples,
        )
    except ValueError as error:
        failure = str(error)

    statuses = Counter(str(value["status"]) for value in records)
    group_positive_counts = Counter(
        str(value["provenance"]["group_id"]) for value in eligible
    )
    passed = calibration is not None
    summary = {
        "schema_version": "1.0",
        "gate": "R00",
        "status": "passed" if passed else "failed_underpowered",
        "decision": (
            "Freeze p_min and advance to endpoint-free planning."
            if passed
            else "Do not run endpoint-free planning until calibration is adequately powered."
        ),
        "failure": failure,
        "manifest": str(args.manifest),
        "manifest_sha256": file_sha256(args.manifest),
        "evaluation_manifest": str(args.evaluation_manifest),
        "evaluation_manifest_sha256": file_sha256(args.evaluation_manifest),
        "calibration_and_evaluation_groups_disjoint": True,
        "calibration_groups": len(calibration_groups),
        "evaluation_groups": len(evaluation_groups),
        "cases": len(records),
        "status_counts": dict(sorted(statuses.items())),
        "eligible_positive_examples": len(eligible),
        "eligible_positive_groups": len(group_positive_counts),
        "positive_examples_per_group": dict(sorted(group_positive_counts.items())),
        "minimum_positive_examples": args.minimum_positive_examples,
        "calibration": calibration.to_dict() if calibration is not None else None,
        "config_hash": config_hashes[0],
        "checkpoint_sha256": checkpoint_hashes[0],
        "git_commit": git_commits[0],
        "ordered_result_set_digest": content_hash(result_hashes),
        "result_hashes": result_hashes,
        "case_records": [
            {
                "case_id": value["case_id"],
                "group_id": value["provenance"]["group_id"],
                "status": value["status"],
                "clearance_m": value["trial"]["clearance_m"],
                "reach_progress_m": value["trial"]["reach"]["reach_progress_m"],
                "target_displacement_m": value["trial"]["reach"]["target_displacement_m"],
                "active_obstacle_displacement_m": value["trial"]["reach"][
                    "active_obstacle_displacement_m"
                ],
                "eligible_for_p_min": value["trial"]["eligible_for_p_min"],
            }
            for value in records
        ],
    }
    atomic_write_json(args.output, summary)
    print(json.dumps({key: value for key, value in summary.items() if key not in {"result_hashes", "case_records"}}, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
