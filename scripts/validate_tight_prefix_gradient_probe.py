#!/usr/bin/env python3
"""Validate paired exact compact-critic gradient probes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _git_identity, _load, _require,
)


def validate(
    *, repo_root: Path, config_path: Path, producer_dir: Path,
    replay_dir: Path, expected_commit: str, accepted_result_commit: str,
) -> dict[str, Any]:
    from main.multilink_ellipsoid.tight_prefix_gradient_probe import (
        CASE_SCHEMA, VALIDATION_SCHEMA, aggregate_anchor_records,
        canonical, feasibility_verdict, load_config, payload_sha256,
        scientific_view,
    )

    config = load_config(config_path)
    producers = []
    replays = []
    exact_replay = True
    for index, case in enumerate(config["cases"]):
        case_id = str(case["case_id"])
        producer = _load(producer_dir / (case_id + ".json"))
        replay = _load(replay_dir / (case_id + ".json"))
        for record in (producer, replay):
            _require(
                record.get("schema_version") == CASE_SCHEMA
                and record.get("case_index") == index
                and record.get("case_id") == case_id
                and record.get("split") == "validation"
                and record.get("source", {}).get("commit")
                == accepted_result_commit
                and record.get("config_file_sha256")
                == config["config_file_sha256"]
                and record.get("config_payload_sha256")
                == config["config_payload_sha256"]
                and record.get("result_payload_sha256")
                == payload_sha256(record, "result_payload_sha256"),
                "tight gradient-probe case artifact differs",
            )
        exact_replay = exact_replay and bool(
            canonical(scientific_view(producer))
            == canonical(scientific_view(replay))
        )
        producers.append(producer)
        replays.append(replay)
    anchors = [
        {
            "case_id": record["case_id"],
            "anchor_name": anchor["anchor_name"],
            "anchor_true_primary_risk": anchor["anchor_true_primary_risk"],
            "exact_probe_primary_risk": anchor["exact_probe_primary_risk"],
            "direction_correct": anchor["direction_correct"],
            "gradient_down_descends": anchor["gradient_down_descends"],
            "gradient_down_beats_random": anchor[
                "gradient_down_beats_random"
            ],
            "gradient_down_converts_safe": anchor[
                "gradient_down_converts_safe"
            ],
            "down_risk_change": anchor["down_risk_change"],
            "anchor_prediction_absolute_error": anchor[
                "anchor_prediction_absolute_error"
            ],
            "post_clipping_symmetry_error": anchor[
                "post_clipping_symmetry_error"
            ],
            "radius_l2_action": anchor["probe_construction"][
                "radius_l2_action"
            ],
            "free_translation_coordinate_count": anchor[
                "probe_construction"
            ]["free_translation_coordinate_count"],
        }
        for record in producers for anchor in record["anchor_records"]
    ]
    metrics = aggregate_anchor_records(anchors)
    physical_false_safes = 0
    exact_state_restore = True
    certified = True
    for record in producers:
        exact = record["exact_rollout"]
        exact_state_restore = exact_state_restore and bool(
            exact["source_replay_exact"]
        )
        certified = certified and bool(
            exact["exact_group_target"]["robot_primitive_certificate_pass"]
        )
        for candidate in exact["candidates"]:
            target = candidate["exact_group_target"]
            for row_group in (
                "palm", "finger1_base", "finger1_pad", "finger2_base",
                "finger2_pad", "L5",
            ):
                physical_false_safes += int(
                    float(target["group_future_violation"][row_group]) <= 0.0
                    and int(target["group_contact_sample_count"][row_group]) > 0
                )
    scientific = feasibility_verdict(metrics, config["gate"])
    apparatus_checks = {
        "independent_scientific_replay_exact": exact_replay,
        "source_state_restore_exact": exact_state_restore,
        "robot_primitives_certified": certified,
        "zero_represented_geometry_physical_false_safes": (
            physical_false_safes == 0
        ),
        "H100_model_prediction_matches_frozen_inference": (
            float(metrics["maximum_anchor_prediction_absolute_error"])
            <= float(config["gate"][
                "maximum_anchor_prediction_absolute_error"
            ])
        ),
    }
    all_pass = bool(
        all(apparatus_checks.values()) and scientific["all_checks_pass"]
    )
    result = {
        "schema_version": VALIDATION_SCHEMA,
        "status": "complete_independent_gradient_probe_validation",
        "scientific_result": True,
        "claim_scope": config["claim_scope"],
        "source": _git_identity(repo_root, expected_commit),
        "accepted_result_commit": accepted_result_commit,
        "config": config,
        "producer_result_payload_sha256": [
            item["result_payload_sha256"] for item in producers
        ],
        "replay_result_payload_sha256": [
            item["result_payload_sha256"] for item in replays
        ],
        "apparatus_checks": apparatus_checks,
        "physical_false_safe_count": int(physical_false_safes),
        "anchor_records": anchors,
        "metrics": metrics,
        "feasibility_verdict": {
            **scientific,
            "all_checks_pass": all_pass,
            "negative_gradient_exact_risk_feasible": all_pass,
            "flow_guidance_or_QP_authorized": False,
        },
        "new_simulator_rollout_count_per_replica": sum(
            int(item["new_simulator_rollout_count"]) for item in producers
        ),
        "training_or_policy_query_performed": False,
        "test_split_opened": False,
        "paper_or_safety_claim_authorized": False,
    }
    result["validation_payload_sha256"] = payload_sha256(
        result, "validation_payload_sha256",
    )
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--producer-dir", type=Path, required=True)
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--accepted-result-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = validate(
        repo_root=args.repo_root.resolve(), config_path=args.config.resolve(),
        producer_dir=args.producer_dir.resolve(),
        replay_dir=args.replay_dir.resolve(),
        expected_commit=args.expected_commit,
        accepted_result_commit=args.accepted_result_commit,
    )
    _atomic_write(args.output.resolve(), result)
    print(json.dumps({
        "apparatus_checks": result["apparatus_checks"],
        "metrics": result["metrics"],
        "feasibility_verdict": result["feasibility_verdict"],
        "validation_payload_sha256": result["validation_payload_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
