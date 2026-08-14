#!/usr/bin/env python3
"""Aggregate and independently validate the sealed repulsion gate."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from scripts.replay_distal_three_ellipsoid_multicbf import (
    _file_sha256,
    _load,
    _require,
    _sha256,
)


SUMMARY_SCHEMA = "vlsa_distal_analytical_repulsion_generalization_summary.v1"
VALIDATION_SCHEMA = "vlsa_distal_analytical_repulsion_generalization_validation.v1"


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _atomic_write(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".%s.tmp" % path.name)
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def validate_results(
    *, config_path: Path, selection_path: Path, result_paths: Sequence[Path],
    expected_commit: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from main.multilink_ellipsoid.analytical_repulsion_generalization import (
        RESULT_SCHEMA,
        aggregate_case_results,
        load_cases,
        load_config,
        warning_trigger,
    )

    config = load_config(config_path)
    cases = load_cases(selection_path)
    _require(len(result_paths) == len(cases), "result count differs")
    results = []
    checks = []
    for expected_case, path in zip(cases, result_paths):
        result = _load(path)
        _require(result["schema_version"] == RESULT_SCHEMA, "result schema differs")
        _require(result["status"] == "complete", "result is incomplete")
        _require(result["scientific_result"] is True, "result is not scientific")
        _require(result["case_id"] == expected_case["case_id"], "case order differs")
        _require(result["source"]["commit"] == expected_commit, "source commit differs")
        payload = dict(result)
        claimed = payload.pop("result_payload_sha256")
        _require(_sha256(_canonical(payload)) == claimed, "result self-hash differs")
        _require(result["config"] == config, "embedded config differs")
        _require(result["selection"] == expected_case, "embedded selection differs")
        raw = result["sealed_raw_AEGIS"]
        _require(raw["native_task_success"] is True, "raw comparator task differs")
        _require(raw["raw_L5_L7_contact_pass"] is False, "raw comparator contact differs")
        _require(raw["paper_car_pass"] is False, "raw comparator CAR differs")
        for action in result["actions"]:
            _require(
                int(action["internal"]["substep_count"])
                == int(config["execution"]["expected_mujoco_substeps_per_action"]),
                "internal substep count differs",
            )
        _require(result["warning_count"] == len(result["interventions"]),
                 "warning count differs")
        clone_error = result["clone_execution_maximum_absolute_error"]
        _require(
            result["clone_execution_mismatch_count"]
            == sum(
                float(row["full_dynamic_state_maximum_absolute_error"])
                > float(config["execution"]["boundary_equivalence_tolerance_m"])
                for row in result["clone_execution_records"]
            ),
            "clone/execution mismatch count differs",
        )
        for intervention in result["interventions"]:
            _require(
                warning_trigger(
                    intervention["nominal_prediction"],
                    buffer_m=float(config["warning_oracle"]["buffer_m"]),
                    car_threshold_m=float(config["warning_oracle"]["paper_car_threshold_m"]),
                ),
                "intervention lacks registered warning",
            )
            _require(
                float(intervention["proposal"]["requested_correction_l2_action"])
                == float(config["fixed_repulsion"]["requested_correction_l2_action"]),
                "intervention radius differs",
            )
        _require(sum(result["contact_samples_by_link"].values()) ==
                 int(result["physical_safety"]["protected_contact_sample_count"]),
                 "contact-link accounting differs")
        results.append(result)
        checks.append({
            "case_id": result["case_id"],
            "result_file_sha256": _file_sha256(path),
            "result_payload_sha256": claimed,
            "warning_count": result["warning_count"],
            "intervention_count": result["intervention_count"],
            "clipped_intervention_count": result["clipped_intervention_count"],
            "raw_L5_L7_contact_pass": result["raw_L5_L7_contact_pass"],
            "paper_car_pass": result["paper_car_pass"],
            "native_task_success": result["native_task_success"],
            "timeout": result["timeout"],
            "primary_problem_solved": result["primary_problem_solved"],
            "clone_execution_maximum_absolute_error": clone_error,
            "clone_execution_mismatch_count": result["clone_execution_mismatch_count"],
            "clone_execution_maximum_boundary_clearance_error_m": result[
                "clone_execution_maximum_boundary_clearance_error_m"
            ],
            "contact_samples_by_link": result["contact_samples_by_link"],
        })
    aggregate = aggregate_case_results(results, config)
    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "status": "complete",
        "scientific_result": True,
        "claim_scope": config["claim_scope"],
        "source_commit": expected_commit,
        "config": config,
        "selection_manifest": {
            "path": str(selection_path),
            "file_sha256": _file_sha256(selection_path),
            "case_ids": [row["case_id"] for row in cases],
        },
        "cases": checks,
        "aggregate": aggregate,
        "interpretation": (
            "fixed_analytical_repulsion_passes_three_case_safe_task_gate"
            if aggregate["strict_gate_pass"]
            else "fixed_analytical_repulsion_generalization_no_go"
        ),
        "limitations": [
            "warning_trigger_uses_privileged_cloned_OSC_with_measured_execution_divergence",
            "actual_all_substep_MuJoCo_contact_and_CAR_are_acceptance_authorities",
            "three_preselected_task_successful_collision_cases_not_population_sample",
            "two_tasks_one_obstacle_class",
            "no_formal_or_real_robot_safety_claim",
        ],
    }
    summary["result_payload_sha256"] = _sha256(_canonical(summary))
    validation = {
        "schema_version": VALIDATION_SCHEMA,
        "status": "passed",
        "scientific_result": True,
        "source_commit": expected_commit,
        "summary_payload_sha256": summary["result_payload_sha256"],
        "case_checks": checks,
        "aggregate": aggregate,
    }
    validation["validation_payload_sha256"] = _sha256(_canonical(validation))
    return summary, validation


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--result", type=Path, action="append", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    args = parser.parse_args(argv)
    summary, validation = validate_results(
        config_path=args.config.resolve(),
        selection_path=args.selection_manifest.resolve(),
        result_paths=[path.resolve() for path in args.result],
        expected_commit=args.expected_commit,
    )
    _atomic_write(args.summary.resolve(), summary)
    _atomic_write(args.validation.resolve(), validation)
    print(json.dumps({
        "interpretation": summary["interpretation"],
        "strict_gate_pass": summary["aggregate"]["strict_gate_pass"],
        "result_payload_sha256": summary["result_payload_sha256"],
        "validation_payload_sha256": validation["validation_payload_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
