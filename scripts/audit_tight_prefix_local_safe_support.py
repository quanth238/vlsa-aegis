#!/usr/bin/env python3
"""Audit immutable exact-safe action support inside the frozen correction ball."""

from __future__ import annotations

import argparse
import json
import os
import socket
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _git_identity, _load, _require,
)


PRIMARY_CONTACT_GROUPS = (
    "palm", "finger1_base", "finger1_pad", "finger2_base",
    "finger2_pad", "L5",
)


def _allocation_record() -> dict[str, Any]:
    _require(os.environ.get("SLURM_JOB_ID") is not None,
             "local safe-support audit requires Slurm")
    _require(
        not os.environ.get("SLURM_JOB_GPUS")
        and not os.environ.get("CUDA_VISIBLE_DEVICES"),
        "local safe-support audit must be CPU-only",
    )
    return {
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "host": socket.gethostname(),
        "device": "CPU",
        "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        "slurm_mem_per_node": os.environ.get("SLURM_MEM_PER_NODE"),
    }


def _candidate_record(
    candidate: Mapping[str, Any], *, primary_rows: Sequence[int],
    car_threshold: float,
) -> dict[str, Any]:
    from main.multilink_ellipsoid.tight_prefix_risk_q_diagnostic import (
        row_future_risks,
    )

    row_q = row_future_risks(candidate, primary_rows)
    exact = candidate["exact_group_target"]
    contact_count = sum(
        int(exact["group_contact_sample_count"][group])
        for group in PRIMARY_CONTACT_GROUPS
    )
    car = float(candidate["replayed_maximum_CAR_m"])
    maximum_q = float(max(row_q))
    safe = bool(maximum_q <= 0.0 and contact_count == 0 and car <= car_threshold)
    return {
        "name": str(candidate["name"]),
        "actions": candidate["source_executed_actions"],
        "maximum_primary_exact_row_Q": maximum_q,
        "primary_exact_row_Q": [float(value) for value in row_q],
        "primary_contact_sample_count": int(contact_count),
        "maximum_CAR_m": car,
        "exact_safe": safe,
    }


def audit(
    *, repo_root: Path, config_path: Path, expected_commit: str,
) -> dict[str, Any]:
    from main.multilink_ellipsoid.tight_prefix_local_safe_support import (
        RESULT_SCHEMA, file_sha256, load_config, nearest_safe_support,
        payload_sha256, summarize,
    )
    from main.multilink_ellipsoid.tight_prefix_risk_dataset import (
        CASE_SCHEMA as DATASET_CASE_SCHEMA,
        VALIDATION_SCHEMA as DATASET_VALIDATION_SCHEMA,
        load_config as load_dataset_config,
        payload_sha256 as dataset_payload_sha256,
    )

    config = load_config(config_path)
    source = config["source"]
    dataset_config_path = repo_root / source["dataset_config"]
    _require(
        file_sha256(dataset_config_path) == source["dataset_config_file_sha256"],
        "local safe-support dataset config file differs",
    )
    dataset_config = load_dataset_config(dataset_config_path, repo_root=repo_root)
    _require(
        dataset_config["config_payload_sha256"]
        == source["dataset_config_payload_sha256"],
        "local safe-support dataset config payload differs",
    )
    validation_path = Path(source["dataset_validation"])
    _require(
        _file_sha256(validation_path) == source["dataset_validation_file_sha256"],
        "local safe-support dataset validation file differs",
    )
    validated = _load(validation_path)
    _require(
        validated.get("schema_version") == DATASET_VALIDATION_SCHEMA
        and validated.get("validation_payload_sha256")
        == source["dataset_validation_payload_sha256"]
        == dataset_payload_sha256(validated)
        and validated.get("dataset_gate_pass") is True,
        "local safe-support validated dataset differs",
    )

    producer_dir = Path(source["dataset_producer_dir"])
    primary_rows = config["audit"]["primary_rows"]
    radius = float(config["audit"]["translation_correction_radius_l2_action"])
    tolerance = float(config["audit"]["distance_tolerance"])
    non_translation_tolerance = float(
        config["audit"]["non_translation_action_tolerance"]
    )
    car_threshold = float(dataset_config["gate"]["paper_car_threshold_m"])
    records = []
    cases = []
    by_anchor = {}
    for case_config in dataset_config["cases"]:
        case_id = str(case_config["case_id"])
        artifact = _load(producer_dir / (case_id + ".json"))
        _require(
            artifact.get("schema_version") == DATASET_CASE_SCHEMA
            and artifact.get("case_id") == case_id
            and artifact.get("split") == case_config["split"]
            and artifact.get("source", {}).get("commit")
            == source["dataset_artifact_commit"]
            and artifact.get("result_payload_sha256")
            == dataset_payload_sha256(artifact),
            "local safe-support dataset case differs",
        )
        candidates = [
            _candidate_record(
                candidate, primary_rows=primary_rows,
                car_threshold=car_threshold,
            )
            for candidate in artifact["case"]["candidates"]
        ]
        safe_candidates = [item for item in candidates if item["exact_safe"]]
        case_records = []
        for anchor in candidates:
            if float(anchor["maximum_primary_exact_row_Q"]) <= 0.0:
                continue
            support = nearest_safe_support(
                anchor_name=anchor["name"], anchor_actions=anchor["actions"],
                safe_candidates=safe_candidates, radius=radius,
                tolerance=tolerance,
                non_translation_tolerance=non_translation_tolerance,
            )
            record = {
                "case_id": case_id,
                "split": str(case_config["split"]),
                "anchor_name": anchor["name"],
                "anchor_maximum_primary_exact_row_Q": anchor[
                    "maximum_primary_exact_row_Q"
                ],
                "anchor_primary_contact_sample_count": anchor[
                    "primary_contact_sample_count"
                ],
                "anchor_maximum_CAR_m": anchor["maximum_CAR_m"],
                **support,
            }
            records.append(record)
            case_records.append(record)
            by_anchor[(case_id, anchor["name"])] = record
        cases.append({
            "case_id": case_id,
            "split": str(case_config["split"]),
            "candidate_count": len(candidates),
            "exact_safe_candidate_count": len(safe_candidates),
            "primary_risk_unsafe_anchor_count": len(case_records),
            "local_safe_supported_anchor_count": sum(
                bool(item["safe_candidate_within_radius"])
                for item in case_records
            ),
        })

    split_summary = {
        split: summarize([item for item in records if item["split"] == split])
        for split in config["audit"]["report_splits"]
    }
    prior_probe = []
    for anchor in config["prior_probe_anchors"]:
        key = (str(anchor["case_id"]), str(anchor["candidate_name"]))
        _require(key in by_anchor, "local safe-support prior anchor is absent")
        prior_probe.append(by_anchor[key])
    decision = split_summary[config["audit"]["decision_split"]]
    supported = int(decision["local_safe_supported_anchor_count"])
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "complete_immutable_local_safe_support_audit",
        "scientific_result": True,
        "claim_scope": config["claim_scope"],
        "source": _git_identity(repo_root, expected_commit),
        "allocation": _allocation_record(),
        "config_file_sha256": config["config_file_sha256"],
        "config_payload_sha256": config["config_payload_sha256"],
        "dataset_validation_payload_sha256": validated[
            "validation_payload_sha256"
        ],
        "case_count": len(cases),
        "candidate_count": sum(item["candidate_count"] for item in cases),
        "cases": cases,
        "unsafe_anchor_records": records,
        "split_summary": split_summary,
        "prior_gradient_probe_anchor_records": prior_probe,
        "decision": {
            "decision_split": config["audit"]["decision_split"],
            "validation_local_safe_supported_anchor_count": supported,
            "frozen_critic_local_support_test_authorized": supported > 0,
            "immediate_retraining_authorized": False,
            "earlier_query_intervention_required_if_unsupported": supported == 0,
            "new_simulation_or_collection_performed": False,
            "test_used_for_decision": False,
        },
        "training_performed": False,
        "simulator_rollout_count": 0,
        "policy_query_count": 0,
    }
    result["result_payload_sha256"] = payload_sha256(
        result, "result_payload_sha256",
    )
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = audit(
        repo_root=args.repo_root.resolve(), config_path=args.config.resolve(),
        expected_commit=args.expected_commit,
    )
    _atomic_write(args.output.resolve(), result)
    print(json.dumps({
        "split_summary": result["split_summary"],
        "prior_gradient_probe_anchor_records": result[
            "prior_gradient_probe_anchor_records"
        ],
        "decision": result["decision"],
        "result_payload_sha256": result["result_payload_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
