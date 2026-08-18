#!/usr/bin/env python3
"""Validate independent symmetric normal/tangent observability replays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _git_identity, _load, _require,
)


def _case_report(result: Mapping[str, Any], config: Mapping[str, Any]) -> dict:
    from main.multilink_ellipsoid.tight_prefix_action_observability import (
        row_future_risk,
    )

    candidates = {row["name"]: row for row in result["case"]["candidates"]}
    _require(
        set(candidates) == {
            "nominal", "normal_pos", "normal_neg", "tangent_up_pos",
            "tangent_up_neg", "tangent_side_pos", "tangent_side_neg",
        },
        "action-observability candidate set differs",
    )
    active_row = int(result["active_frame"]["active_row"])
    primary_rows = tuple(int(row) for row in config["perturbation"]["primary_rows"])
    risks = {
        name: {
            "active_row": row_future_risk(candidate, active_row),
            "global_primary": max(
                row_future_risk(candidate, row) for row in primary_rows
            ),
        }
        for name, candidate in candidates.items()
    }
    nominal = risks["nominal"]
    deltas = {
        name: {
            key: float(value - nominal[key]) for key, value in values.items()
        }
        for name, values in risks.items() if name != "nominal"
    }
    tangent_names = (
        "tangent_up_pos", "tangent_up_neg",
        "tangent_side_pos", "tangent_side_neg",
    )
    best_tangent = min(
        tangent_names, key=lambda name: deltas[name]["active_row"],
    )
    maximum_absolute_tangent_change = max(
        abs(deltas[name]["active_row"]) for name in tangent_names
    )
    material = float(config["gate"]["material_absolute_risk_change"])
    blind = bool(
        float(result["maximum_active_row_tangent_7D_feature_change"])
        <= float(config["gate"]["maximum_tangent_7D_feature_change"])
    )
    physical_false_safes = []
    for name, candidate in candidates.items():
        exact = candidate["exact_group_target"]
        for group in exact["group_order"]:
            if (
                float(exact["group_minimum_normalized_radial_slack"][group]) > 0.0
                and int(exact["group_contact_sample_count"][group]) > 0
            ):
                physical_false_safes.append({
                    "candidate": name, "group": group,
                    "slack": exact["group_minimum_normalized_radial_slack"][group],
                    "contact_count": exact["group_contact_sample_count"][group],
                })
    radius = float(result["symmetry_audit"]["common_translation_l2_action"])
    slopes = {
        axis: {
            "active_row": float(
                (risks[axis + "_pos"]["active_row"]
                 - risks[axis + "_neg"]["active_row"]) / (2.0 * radius)
            ),
            "global_primary": float(
                (risks[axis + "_pos"]["global_primary"]
                 - risks[axis + "_neg"]["global_primary"]) / (2.0 * radius)
            ),
        }
        for axis in ("normal", "tangent_up", "tangent_side")
    }
    return {
        "case_id": result["case_id"],
        "active_row": active_row,
        "active_initial_slack": result["active_frame"]["active_initial_slack"],
        "common_translation_l2_action": radius,
        "risks": risks,
        "risk_changes_from_nominal": deltas,
        "finite_difference_slopes": slopes,
        "best_tangent_direction": best_tangent,
        "best_tangent_active_row_risk_change": deltas[best_tangent]["active_row"],
        "maximum_absolute_tangent_active_row_risk_change": (
            maximum_absolute_tangent_change
        ),
        "active_row_7D_tangent_blind": blind,
        "structural_counterexample": bool(
            blind and maximum_absolute_tangent_change >= material
        ),
        "material_tangent_descent": bool(
            blind and deltas[best_tangent]["active_row"] <= -material
        ),
        "physical_false_safes": physical_false_safes,
    }


def validate(
    *, repo_root: Path, config_path: Path, producer_dir: Path,
    replay_dir: Path, expected_commit: str, accepted_commit: str,
) -> dict:
    from main.multilink_ellipsoid.pi05_palm_l6_source_selection import (
        cpu_allocation_record,
    )
    from main.multilink_ellipsoid.tight_prefix_action_observability import (
        CASE_SCHEMA, VALIDATION_SCHEMA, canonical, load_config,
        payload_sha256, scientific_view,
    )

    config = load_config(config_path)
    reports = []
    exact_by_case = {}
    producer_records = []
    replay_records = []
    for selected in config["cases"]:
        case_id = selected["case_id"]
        producer_path = producer_dir / (case_id + ".json")
        replay_path = replay_dir / (case_id + ".json")
        producer = _load(producer_path)
        replay = _load(replay_path)
        for result in (producer, replay):
            _require(
                result.get("schema_version") == CASE_SCHEMA
                and result.get("case_id") == case_id
                and result.get("source", {}).get("commit") == accepted_commit
                and result.get("config") == config
                and result.get("result_payload_sha256")
                == payload_sha256(result, "result_payload_sha256"),
                "action-observability result differs",
            )
        exact = canonical(scientific_view(producer)) == canonical(
            scientific_view(replay)
        )
        _require(exact, "action-observability independent replay differs")
        exact_by_case[case_id] = exact
        reports.append(_case_report(producer, config))
        producer_records.append({
            "case_id": case_id, "path": str(producer_path),
            "file_sha256": _file_sha256(producer_path),
            "payload_sha256": producer["result_payload_sha256"],
        })
        replay_records.append({
            "case_id": case_id, "path": str(replay_path),
            "file_sha256": _file_sha256(replay_path),
            "payload_sha256": replay["result_payload_sha256"],
        })
    structural = sum(row["structural_counterexample"] for row in reports)
    tangent_descent = sum(row["material_tangent_descent"] for row in reports)
    physical_false_safes = sum(
        len(row["physical_false_safes"]) for row in reports
    )
    maximum_symmetry = max(
        float(_load(producer_dir / (row["case_id"] + ".json"))[
            "symmetry_audit"
        ]["maximum_pair_symmetry_error"])
        for row in reports
    )
    gate = config["gate"]
    apparatus_pass = bool(
        len(reports) == int(gate["required_case_count"])
        and all(exact_by_case.values())
        and maximum_symmetry
        <= float(gate["maximum_post_clipping_symmetry_error"])
        and physical_false_safes
        <= int(gate["maximum_physical_false_safe_count"])
    )
    representation_pass = bool(
        apparatus_pass
        and structural >= int(gate["minimum_structural_counterexample_count"])
        and tangent_descent
        >= int(gate["minimum_material_tangent_descent_case_count"])
    )
    value = {
        "schema_version": VALIDATION_SCHEMA,
        "status": "validated_action_observability_audit",
        "scientific_result": True,
        "claim_scope": config["claim_scope"],
        "validator_source": _git_identity(repo_root, expected_commit),
        "allocation": cpu_allocation_record(),
        "accepted_experiment_commit": accepted_commit,
        "config": config,
        "producer_records": producer_records,
        "replay_records": replay_records,
        "independent_scientific_view_equal_by_case": exact_by_case,
        "case_reports": reports,
        "summary": {
            "case_count": len(reports),
            "structural_counterexample_count": int(structural),
            "material_tangent_descent_case_count": int(tangent_descent),
            "physical_false_safe_count": int(physical_false_safes),
            "maximum_post_clipping_symmetry_error": float(maximum_symmetry),
            "apparatus_gate_pass": apparatus_pass,
            "seven_D_action_observability_rejected": bool(structural > 0),
            "seventeen_D_ablation_authorized": representation_pass,
        },
        "model_training_or_policy_query_performed": False,
        "test_split_accessed": False,
        "full_episode_authorized": False,
    }
    value["validation_payload_sha256"] = payload_sha256(
        value, "validation_payload_sha256",
    )
    return value


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--producer-dir", type=Path, required=True)
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--accepted-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    value = validate(
        repo_root=args.repo_root.resolve(), config_path=args.config.resolve(),
        producer_dir=args.producer_dir.resolve(), replay_dir=args.replay_dir.resolve(),
        expected_commit=args.expected_commit, accepted_commit=args.accepted_commit,
    )
    _atomic_write(args.output.resolve(), value)
    print(json.dumps({
        "status": value["status"],
        "summary": value["summary"],
        "validation_payload_sha256": value["validation_payload_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
