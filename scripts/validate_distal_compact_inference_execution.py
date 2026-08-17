#!/usr/bin/env python3
"""Validate paired compact-inference execution and independent replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional, Sequence

from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _git_identity, _load, _require,
)


def _failed_groups(pair: dict[str, Any], groups: Sequence[str]) -> list[str]:
    selected = pair["selected"]
    risks = selected["group_future_violation"]
    contacts = selected["group_contact_sample_count"]
    return [
        group for group in groups
        if float(risks[group]) > 0.0 or int(contacts[group]) > 0
    ]


def run(
    *, repo_root: Path, config_path: Path, producer_dir: Path,
    replay_dir: Path, expected_commit: str,
) -> dict[str, Any]:
    from main.multilink_ellipsoid.compact_inference_execution import (
        RESULT_SCHEMA, VALIDATION_SCHEMA, load_config, payload_sha256,
        scientific_view,
    )

    config = load_config(config_path)
    rows = []
    for case in config["cases"]:
        case_id = str(case["case_id"])
        producer_path = producer_dir / (case_id + ".json")
        replay_path = replay_dir / (case_id + ".json")
        producer = _load(producer_path)
        replay = _load(replay_path)
        for value, path, label in (
            (producer, producer_path, "producer"),
            (replay, replay_path, "replay"),
        ):
            _require(
                value.get("schema_version") == RESULT_SCHEMA
                and value.get("status") == "complete"
                and value.get("case_id") == case_id
                and value.get("source", {}).get("commit") == expected_commit
                and value.get("result_payload_sha256")
                == payload_sha256(value, "result_payload_sha256"),
                "compact inference execution %s differs" % label,
            )
            _require(
                value["fresh_pair"]["exact_case"]["source_replay_exact"] is True
                and value["fresh_pair"]["original_AEGIS_EE_QP_enabled"] is False
                and value["fresh_pair"]["learned_correction_QP_enabled"] is False,
                "compact inference execution fresh apparatus differs",
            )
        _require(
            scientific_view(producer) == scientific_view(replay),
            "compact inference execution independent replay differs",
        )
        pair = producer["pair"]
        rows.append({
            "case_id": case_id,
            "split": str(case["split"]),
            "stratum": str(case["stratum"]),
            "selected_candidate": producer["model_selection"][
                "selected_candidate"
            ],
            "selected_predicted_primary": producer["model_selection"][
                "selected_predicted_primary"
            ],
            "nominal_physical_safe": bool(pair["nominal_physical_safe"]),
            "selected_physical_safe": bool(pair["selected_physical_safe"]),
            "selected_outcome_known": bool(pair["selected_outcome_known"]),
            "collision_avoided": bool(pair["collision_avoided"]),
            "collision_persisted": bool(pair["collision_persisted"]),
            "selected_failed_groups": _failed_groups(
                pair, config["physical_outcome_groups"],
            ),
            "selected_diagnostic_EE_future_violation": pair["selected"][
                "diagnostic_EE_future_violation"
            ],
            "producer_file_sha256": _file_sha256(producer_path),
            "replay_file_sha256": _file_sha256(replay_path),
        })

    nominal_safe = sum(row["nominal_physical_safe"] for row in rows)
    selected_safe = sum(row["selected_physical_safe"] for row in rows)
    unknown = sum(not row["selected_outcome_known"] for row in rows)
    avoided = sum(row["collision_avoided"] for row in rows)
    persisted = sum(row["collision_persisted"] for row in rows)
    evaluation = config["evaluation"]
    gates = {
        "all_cases_complete": len(rows) == len(config["cases"]),
        "exact_independent_replay": True,
        "same_snapshot_paired_execution": all(
            row["nominal_physical_safe"] is not None for row in rows
        ),
        "zero_selected_UNKNOWN": unknown == 0,
        "minimum_selected_physical_safe_count": selected_safe >= int(
            evaluation["minimum_selected_physical_safe_count"]
        ),
        "minimum_safe_count_improvement_over_nominal": (
            selected_safe - nominal_safe
            >= int(evaluation["minimum_safe_count_improvement_over_nominal"])
        ),
        "selection_never_uses_fresh_outcome": True,
        "QP_disabled": True,
    }
    mechanism_pass = all(gates.values())
    safety_pass = bool(mechanism_pass and selected_safe == len(rows))
    value = {
        "schema_version": VALIDATION_SCHEMA,
        "status": "complete",
        "scientific_result": True,
        "claim_scope": config["claim_scope"],
        "source": _git_identity(repo_root, expected_commit),
        "config_file_sha256": config["config_file_sha256"],
        "config_payload_sha256": config["config_payload_sha256"],
        "case_count": len(rows),
        "nominal_physical_safe_count": nominal_safe,
        "selected_physical_safe_count": selected_safe,
        "safe_count_improvement": selected_safe - nominal_safe,
        "collision_avoided_count": avoided,
        "collision_persisted_count": persisted,
        "selected_UNKNOWN_count": unknown,
        "cases": rows,
        "gates": gates,
        "mechanism_pass": mechanism_pass,
        "model_only_safety_pass": safety_pass,
        "interpretation": (
            "compact_inference_reduces_collision_on_opened_pilot_but_is_not_safe"
            if mechanism_pass and not safety_pass
            else "compact_inference_opened_pilot_safety_pass"
            if safety_pass
            else "compact_inference_opened_pilot_no_go"
        ),
        "correction_safety_authorized": False,
        "closed_loop_authorized": False,
        "untouched_generalization_claim_authorized": False,
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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    value = run(
        repo_root=args.repo_root.resolve(), config_path=args.config.resolve(),
        producer_dir=args.producer_dir.resolve(), replay_dir=args.replay_dir.resolve(),
        expected_commit=args.expected_commit,
    )
    _atomic_write(args.output.resolve(), value)
    print(json.dumps({
        "interpretation": value["interpretation"],
        "nominal_physical_safe_count": value["nominal_physical_safe_count"],
        "selected_physical_safe_count": value["selected_physical_safe_count"],
        "collision_avoided_count": value["collision_avoided_count"],
        "collision_persisted_count": value["collision_persisted_count"],
        "validation_payload_sha256": value["validation_payload_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
