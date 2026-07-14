#!/usr/bin/env python3
"""Summarize H03 artifacts without allowing partial cases to count."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from crfs_harness.artifacts import atomic_write_json, validate_jsonl_unique
from crfs_harness.manifest import validate_case


def summarize(
    manifest_path: Path, run_root: Path, primitive_path: Path, visualization_path: Path
) -> dict[str, Any]:
    cases, manifest_errors = validate_jsonl_unique(manifest_path, "case_id")
    for case in cases:
        manifest_errors.extend(
            f"{case.get('case_id', '<unknown>')}: {error}" for error in validate_case(case)
        )
    primitive = json.loads(primitive_path.read_text(encoding="utf-8"))
    audits = []
    missing = []
    invalid = []
    for case in cases:
        path = run_root / str(case["case_id"]) / "measurement-audit.json"
        if not path.exists():
            missing.append(str(case["case_id"]))
            continue
        value = json.loads(path.read_text(encoding="utf-8"))
        reasons = []
        if value.get("case_id") != case["case_id"]:
            reasons.append("case identity mismatch")
        if value.get("status") != "passed":
            reasons.append(f"status={value.get('status')!r}")
        if int(value.get("repeats", 0)) < 5:
            reasons.append("fewer than five repeats")
        if not value.get("sample_counts_correct"):
            reasons.append("physics-substep sample count mismatch")
        if not value.get("contact_conservative"):
            reasons.append("positive proxy clearance during physical contact")
        if float(value.get("max_clearance_variation_m", float("inf"))) >= 1e-4:
            reasons.append("clearance repeatability is not below 1e-4 m")
        if reasons:
            invalid.append({"case_id": case["case_id"], "reasons": reasons})
        else:
            audits.append((case, value))

    clearances = [float(audit["runs"][0]["clearance_m"]) for _, audit in audits]
    contacts = [bool(audit["runs"][0]["contact"]) for _, audit in audits]
    raw_inconsistent = [
        not bool(audit["runs"][0]["measurement"]["raw_mj_contact_consistent"])
        for _, audit in audits
    ]
    unique_episodes = len({int(case["episode_index"]) for case, _ in audits})
    passed = bool(
        not manifest_errors
        and len(cases) >= 50
        and unique_episodes >= 50
        and not missing
        and not invalid
        and primitive.get("status") == "passed"
        and visualization_path.exists()
    )
    return {
        "schema_version": "1.0",
        "gate": "H03",
        "status": "passed" if passed else "failed",
        "manifest": str(manifest_path),
        "run_root": str(run_root),
        "primitive_calibration": str(primitive_path),
        "primitive_status": primitive.get("status"),
        "minimum_pair_visualization": str(visualization_path),
        "visualization_exists": visualization_path.exists(),
        "expected_cases": len(cases),
        "valid_cases": len(audits),
        "unique_saved_states": unique_episodes,
        "missing_case_ids": missing,
        "invalid_cases": invalid,
        "manifest_errors": manifest_errors,
        "max_clearance_variation_m": max(
            (float(audit["max_clearance_variation_m"]) for _, audit in audits), default=None
        ),
        "max_endpoint_coordinate_variation_m": max(
            (float(audit["max_endpoint_coordinate_variation_m"]) for _, audit in audits), default=None
        ),
        "proxy_colliding_cases": sum(clearance < 0.0 for clearance in clearances),
        "proxy_safe_cases": sum(clearance >= 0.0 for clearance in clearances),
        "physical_contact_cases": sum(contacts),
        "raw_mujoco_inconsistent_cases": sum(raw_inconsistent),
        "minimum_proxy_clearance_m": min(clearances, default=None),
        "maximum_proxy_clearance_m": max(clearances, default=None),
        "decision": (
            "Advance to H04 calibration; raw mesh-box mj_geomDistance remains advisory only."
            if passed
            else "Do not advance; repair or complete the listed H03 evidence."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--primitive-calibration", required=True)
    parser.add_argument("--visualization", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = summarize(
        Path(args.manifest),
        Path(args.run_root),
        Path(args.primitive_calibration),
        Path(args.visualization),
    )
    atomic_write_json(args.output, result)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
