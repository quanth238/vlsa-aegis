#!/usr/bin/env python3
"""Validate paired immediately-earlier-query frozen-critic probes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _git_identity, _load, _require,
)


PRIMARY_GROUPS = (
    "palm", "finger1_base", "finger1_pad", "finger2_base",
    "finger2_pad", "L5",
)


def _artifact_valid(
    record: Mapping[str, Any], *, schema: str, case_index: int,
    case_id: str, config: Mapping[str, Any], accepted_commit: str,
    payload_key: str,
) -> None:
    from main.multilink_ellipsoid.tight_prefix_early_gradient_probe import (
        payload_sha256,
    )

    _require(
        record.get("schema_version") == schema
        and record.get("case_index") == int(case_index)
        and record.get("case_id") == case_id
        and record.get("split") == "validation"
        and record.get("source", {}).get("commit") == accepted_commit
        and record.get("config_file_sha256")
        == config["config_file_sha256"]
        and record.get("config_payload_sha256")
        == config["config_payload_sha256"]
        and record.get(payload_key) == payload_sha256(record, payload_key),
        "early gradient-probe artifact differs",
    )


def _audit_exact_case(case: Mapping[str, Any]) -> dict[str, Any]:
    false_safes = 0
    for candidate in case["candidates"]:
        exact = candidate["exact_group_target"]
        for group in PRIMARY_GROUPS:
            false_safes += int(
                float(exact["group_future_violation"][group]) <= 0.0
                and int(exact["group_contact_sample_count"][group]) > 0
            )
    return {
        "source_replay_exact": bool(case["source_replay_exact"]),
        "robot_primitive_certificate_pass": bool(
            case["exact_group_target"]["robot_primitive_certificate_pass"]
        ),
        "physical_false_safe_count": int(false_safes),
    }


def validate(
    *, repo_root: Path, config_path: Path,
    producer_prep_dir: Path, replay_prep_dir: Path,
    producer_dir: Path, replay_dir: Path,
    expected_commit: str, accepted_result_commit: str,
) -> dict[str, Any]:
    from main.multilink_ellipsoid.tight_prefix_early_gradient_probe import (
        CASE_SCHEMA, PREP_SCHEMA, VALIDATION_SCHEMA, canonical,
        hypothesis_verdict, load_config, payload_sha256, scientific_view,
        summarize_cases,
    )

    config = load_config(config_path)
    producer_records = []
    replay_records = []
    prep_exact = True
    result_exact = True
    exact_audits = []
    maximum_prediction_error = 0.0
    maximum_symmetry_error = 0.0
    no_forbidden_execution = True
    for index, selected in enumerate(config["cases"]):
        case_id = str(selected["case_id"])
        producer_prep = _load(producer_prep_dir / (case_id + ".json"))
        replay_prep = _load(replay_prep_dir / (case_id + ".json"))
        for prep in (producer_prep, replay_prep):
            _artifact_valid(
                prep, schema=PREP_SCHEMA, case_index=index, case_id=case_id,
                config=config, accepted_commit=accepted_result_commit,
                payload_key="result_payload_sha256",
            )
            _require(
                prep.get("early_state_step") == selected["early_state_step"]
                and prep.get("new_policy_query_count") == 0
                and prep.get("model_training_performed") is False
                and prep.get("QP_or_analytical_guidance_performed") is False,
                "early gradient-probe prep protocol differs",
            )
        prep_exact = prep_exact and bool(
            canonical(producer_prep["scientific_view"])
            == canonical(replay_prep["scientific_view"])
        )
        exact_audits.extend((
            _audit_exact_case(producer_prep["exact_case"]),
            _audit_exact_case(replay_prep["exact_case"]),
        ))

        producer = _load(producer_dir / (case_id + ".json"))
        replay = _load(replay_dir / (case_id + ".json"))
        for record in (producer, replay):
            _artifact_valid(
                record, schema=CASE_SCHEMA, case_index=index, case_id=case_id,
                config=config, accepted_commit=accepted_result_commit,
                payload_key="result_payload_sha256",
            )
            _require(
                record.get("model_sha256") == config["source"]["model_sha256"]
                and record.get("new_policy_query_count") == 0
                and record.get("training_performed") is False
                and record.get("QP_or_analytical_guidance_performed") is False
                and record.get("test_split_opened") is False,
                "early gradient-probe case protocol differs",
            )
            no_forbidden_execution = no_forbidden_execution and bool(
                record["new_policy_query_count"] == 0
                and record["training_performed"] is False
                and record["QP_or_analytical_guidance_performed"] is False
                and record["test_split_opened"] is False
            )
        result_exact = result_exact and bool(
            canonical(scientific_view(producer))
            == canonical(scientific_view(replay))
        )
        producer_records.append(producer["record"])
        replay_records.append(replay["record"])
        for record in (producer, replay):
            maximum_prediction_error = max(
                maximum_prediction_error,
                float(record["record"]["critic"][
                    "maximum_stored_prediction_absolute_error"
                ]),
            )
            construction = record["record"]["probe_construction"]
            if construction is not None:
                maximum_symmetry_error = max(
                    maximum_symmetry_error,
                    float(construction["post_clipping_symmetry_error"]),
                    float(construction["equal_norm_maximum_error"]),
                )
            if record["exact_rollout"] is not None:
                exact_audits.append(_audit_exact_case(record["exact_rollout"]))

    summary = summarize_cases(producer_records)
    scientific = hypothesis_verdict(summary, config["gate"])
    apparatus_checks = {
        "prep_independent_scientific_replay_exact": prep_exact,
        "probe_independent_scientific_replay_exact": result_exact,
        "source_state_restore_exact": all(
            item["source_replay_exact"] for item in exact_audits
        ),
        "robot_primitives_certified": all(
            item["robot_primitive_certificate_pass"] for item in exact_audits
        ),
        "zero_represented_geometry_physical_false_safes": sum(
            item["physical_false_safe_count"] for item in exact_audits
        ) == 0,
        "H100_model_prediction_matches_frozen_inference": (
            maximum_prediction_error
            <= float(config["gate"][
                "maximum_model_prediction_absolute_error"
            ])
        ),
        "symmetric_equal_norm_probes": (
            maximum_symmetry_error
            <= float(config["gate"][
                "maximum_post_clipping_symmetry_error"
            ])
        ),
        "no_forbidden_execution": no_forbidden_execution,
    }
    all_pass = bool(
        all(apparatus_checks.values()) and scientific["all_checks_pass"]
    )
    result = {
        "schema_version": VALIDATION_SCHEMA,
        "status": "complete_early_query_gradient_probe_validation",
        "scientific_result": True,
        "claim_scope": config["claim_scope"],
        "source": _git_identity(repo_root, expected_commit),
        "accepted_result_commit": accepted_result_commit,
        "config": config,
        "producer_result_payload_sha256": [
            item["result_payload_sha256"]
            for item in (
                _load(producer_dir / (str(case["case_id"]) + ".json"))
                for case in config["cases"]
            )
        ],
        "replay_result_payload_sha256": [
            item["result_payload_sha256"]
            for item in (
                _load(replay_dir / (str(case["case_id"]) + ".json"))
                for case in config["cases"]
            )
        ],
        "apparatus_checks": apparatus_checks,
        "maximum_model_prediction_absolute_error": maximum_prediction_error,
        "maximum_action_symmetry_or_norm_error": maximum_symmetry_error,
        "physical_false_safe_count": sum(
            item["physical_false_safe_count"] for item in exact_audits
        ),
        "case_records": producer_records,
        "summary": summary,
        "hypothesis_verdict": {
            **scientific,
            "all_checks_pass": all_pass,
            "early_bounded_gradient_hypothesis_confirmed": all_pass,
            "retraining_authorized": False,
            "full_episode_or_deployment_authorized": False,
        },
        "new_simulator_rollout_count_per_replica": sum(
            int(_load(producer_dir / (str(case["case_id"]) + ".json"))[
                "new_simulator_rollout_count"
            ]) for case in config["cases"]
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
    parser.add_argument("--producer-prep-dir", type=Path, required=True)
    parser.add_argument("--replay-prep-dir", type=Path, required=True)
    parser.add_argument("--producer-dir", type=Path, required=True)
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--accepted-result-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = validate(
        repo_root=args.repo_root.resolve(), config_path=args.config.resolve(),
        producer_prep_dir=args.producer_prep_dir.resolve(),
        replay_prep_dir=args.replay_prep_dir.resolve(),
        producer_dir=args.producer_dir.resolve(),
        replay_dir=args.replay_dir.resolve(),
        expected_commit=args.expected_commit,
        accepted_result_commit=args.accepted_result_commit,
    )
    _atomic_write(args.output.resolve(), result)
    print(json.dumps({
        "apparatus_checks": result["apparatus_checks"],
        "summary": result["summary"],
        "hypothesis_verdict": result["hypothesis_verdict"],
        "validation_payload_sha256": result["validation_payload_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
