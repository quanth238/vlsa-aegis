#!/usr/bin/env python3
"""Independently validate the clean warning-state selector population."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _git_identity, _load, _require, _sha256,
)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _payload_sha256(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("result_payload_sha256", None)
    return _sha256(_canonical(payload))


def validate(
    *, repo_root: Path, producer_root: Path, selection_manifest: Path,
    experiment_config: Path, expected_producer_commit: str,
    expected_validation_commit: str,
) -> dict[str, Any]:
    from main.multilink_ellipsoid.clean_action_risk import (
        load_cases, load_config, select_proxy_boundary_lead_state,
    )
    from main.multilink_ellipsoid.shadow import allocation_record

    config = load_config(experiment_config)
    cases = [
        case for case in load_cases(selection_manifest, config)
        if case["split"] in {"diagnostic", "train", "validation"}
    ]
    _require(len(cases) == 15, "warning-state validation case count differs")
    records = []
    failures = []
    for case_index, expected_case in enumerate(cases):
        path = producer_root / ("case-%02d" % case_index) / "result.json"
        result = _load(path)
        _require(
            result.get("schema_version")
            == "vlsa_distal_clean_warning_state_audit_result.v1",
            "warning-state result schema differs",
        )
        _require(result.get("status") == "complete", "warning-state result incomplete")
        _require(result.get("case_index") == case_index, "warning-state case index differs")
        _require(result.get("case") == expected_case, "warning-state case differs")
        _require(
            result.get("result_payload_sha256") == _payload_sha256(result),
            "warning-state result payload differs",
        )
        _require(
            result.get("source", {}).get("commit") == expected_producer_commit
            and result.get("source", {}).get("dirty") is False,
            "warning-state producer source differs",
        )
        _require(
            result.get("scientific_result") is True
            and result.get("candidate_evaluation_performed") is False
            and result.get("MLP_training_authorized") is False
            and result.get("QP_authorized") is False
            and result.get("closed_loop_authorized") is False,
            "warning-state authorization differs",
        )
        trace = result.get("trace", [])
        contact_step = int(expected_case["first_relevant_contact_step"])
        _require(len(trace) == contact_step, "warning-state trace length differs")
        _require(
            [item.get("step") for item in trace] == list(range(contact_step)),
            "warning-state trace steps differ",
        )
        audit_cfg = result["audit_config"]
        recomputed = select_proxy_boundary_lead_state(
            [item["minimum_clearance_m"] for item in trace],
            [item["physically_valid"] for item in trace],
            safety_buffer_m=float(audit_cfg["safety_buffer_m"]),
            lead_actions=int(audit_cfg["lead_actions_before_proxy_boundary"]),
            stop_before_step=contact_step,
        )
        recorded = dict(result["selection"])
        selected_state = recorded.pop("selected_state", None)
        _require(recorded == recomputed, "warning-state selector reproduction differs")
        passed = recomputed["status"] == "selected"
        if passed:
            selected_trace = trace[int(recomputed["selected_step"])]
            _require(selected_state == selected_trace, "warning-state selected trace differs")
            passed = bool(
                selected_trace["minimum_clearance_m"]
                >= float(audit_cfg["safety_buffer_m"])
                and selected_trace["protected_contact_count"] == 0
                and selected_trace["active_obstacle_l1_displacement_m"]
                <= float(audit_cfg["maximum_initial_active_obstacle_l1_displacement_m"])
                and selected_trace["physically_valid"]
            )
        if not passed:
            failures.append({
                "case_id": expected_case["case_id"],
                "split": expected_case["split"],
                "selection": result["selection"],
            })
        records.append({
            "case_id": expected_case["case_id"],
            "split": expected_case["split"],
            "source_result_file_sha256": _file_sha256(path),
            "source_result_payload_sha256": result["result_payload_sha256"],
            "selection": result["selection"],
            "strict_selected_state_pass": passed,
        })
    split_summary = {}
    for split in ("diagnostic", "train", "validation"):
        split_records = [item for item in records if item["split"] == split]
        split_summary[split] = {
            "case_count": len(split_records),
            "selected_count": sum(item["strict_selected_state_pass"] for item in split_records),
        }
    selector_pass = not failures
    output = {
        "schema_version": "vlsa_distal_clean_warning_state_audit_validation.v1",
        "status": "validated",
        "source": _git_identity(repo_root, expected_validation_commit),
        "allocation": allocation_record(),
        "producer": {
            "root": str(producer_root),
            "expected_commit": expected_producer_commit,
        },
        "case_count": len(records),
        "split_summary": split_summary,
        "records": records,
        "failures": failures,
        "gates": {
            "all_requested_artifacts_complete": True,
            "all_selected_states_strictly_safe": selector_pass,
            "geometry_relative_selector_pass": selector_pass,
            "exact_candidate_support_evaluated": False,
        },
        "interpretation": (
            "geometry_relative_selector_pass_candidate_dataset_still_required"
            if selector_pass else "geometry_relative_selector_no_go"
        ),
        "MLP_training_authorized": False,
        "QP_authorized": False,
        "closed_loop_authorized": False,
    }
    output["result_payload_sha256"] = _sha256(_canonical(output))
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--producer-root", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--expected-producer-commit", required=True)
    parser.add_argument("--expected-validation-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = validate(
        repo_root=args.repo_root.resolve(),
        producer_root=args.producer_root.resolve(),
        selection_manifest=args.selection_manifest.resolve(),
        experiment_config=args.experiment_config.resolve(),
        expected_producer_commit=args.expected_producer_commit,
        expected_validation_commit=args.expected_validation_commit,
    )
    _atomic_write(args.output.resolve(), result)
    print(json.dumps({
        "interpretation": result["interpretation"],
        "split_summary": result["split_summary"],
        "result_payload_sha256": result["result_payload_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
