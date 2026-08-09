#!/usr/bin/env python3
"""Independently validate the grouped two-step no-training dataset gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from main.multilink_ellipsoid.two_step_margin import (
    TWO_STEP_DATASET_RESULT_SCHEMA,
    TWO_STEP_DATASET_SCHEMA,
    load_selected_manifest,
    load_two_step_config,
)
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write,
    _file_sha256,
    _load,
    _require,
)


def _hash_without(value: Mapping[str, Any], key: str) -> str:
    payload = dict(value)
    payload.pop(key, None)
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--selected-manifest", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_two_step_config(args.config.resolve())
    selected = load_selected_manifest(args.selected_manifest.resolve(), config)
    result = _load(args.result.resolve())
    dataset = _load(args.dataset.resolve())
    _require(
        result.get("schema_version") == TWO_STEP_DATASET_RESULT_SCHEMA
        and dataset.get("schema_version") == TWO_STEP_DATASET_SCHEMA,
        "two-step dataset schemas differ",
    )
    _require(
        result.get("source", {}).get("commit") == args.expected_commit
        and dataset.get("source_commit") == args.expected_commit,
        "two-step dataset source differs",
    )
    _require(
        result.get("result_payload_sha256")
        == _hash_without(result, "result_payload_sha256")
        and dataset.get("dataset_payload_sha256")
        == _hash_without(dataset, "dataset_payload_sha256"),
        "two-step dataset payload hash differs",
    )
    _require(
        result.get("dataset", {}).get("file_sha256")
        == _file_sha256(args.dataset.resolve())
        and result.get("dataset", {}).get("payload_sha256")
        == dataset.get("dataset_payload_sha256"),
        "two-step dataset result binding differs",
    )
    episodes = dataset.get("episode_results", [])
    _require(
        len(episodes) == len(selected)
        and {item["case_id"] for item in episodes}
        == {item["case_id"] for item in selected},
        "two-step episode set differs",
    )
    selected_by_case = {item["case_id"]: item for item in selected}
    gate = True
    for episode in episodes:
        row = selected_by_case[episode["case_id"]]
        _require(
            episode["split"] == row["split"]
            and episode["task_level_group_id"] == row["task_level_group_id"],
            "two-step grouped episode identity differs",
        )
        if not episode.get("eligible"):
            gate = False
            continue
        _require(
            episode.get("gradient_anchor_count")
            == config["sampling"]["gradient_anchor_count"]
            and episode.get("stable_active_gradient_anchor_count", 0) > 0,
            "two-step eligible episode gradient gate differs",
        )
        counts = episode.get("grid_category_counts", {})
        _require(
            counts.get("boundary_safe", 0)
            >= config["sampling"]["gradient_anchor_safe_count"]
            and counts.get("boundary_unsafe", 0)
            >= config["sampling"]["gradient_anchor_unsafe_count"],
            "two-step eligible episode boundary balance differs",
        )
    _require(
        bool(dataset.get("summary", {}).get("dataset_gate_pass")) is gate
        and bool(result.get("decision", {}).get("dataset_gate_pass")) is gate
        and bool(result.get("decision", {}).get("neural_training_authorized")) is gate,
        "two-step dataset gate is inconsistent",
    )
    records = dataset.get("records", [])
    _require(
        [item.get("record_index") for item in records] == list(range(len(records))),
        "two-step record indexes differ",
    )
    learning = [
        item for item in records if item.get("split") in {"train", "validation", "test"}
    ]
    _require(
        all(item["episode_split"] == item["split"] for item in learning),
        "two-step complete episode split leaked",
    )
    output = {
        "schema_version": "vlsa_distal_two_step_margin_moka10_dataset_validation.v1",
        "status": "valid",
        "scientific_result": False,
        "expected_commit": args.expected_commit,
        "result_file_sha256": _file_sha256(args.result.resolve()),
        "dataset_file_sha256": _file_sha256(args.dataset.resolve()),
        "config_file_sha256": _file_sha256(args.config.resolve()),
        "selected_manifest_file_sha256": _file_sha256(
            args.selected_manifest.resolve()
        ),
        "eligible_episode_count": sum(item.get("eligible") is True for item in episodes),
        "learning_record_count": len(learning),
        "neural_training_authorized": gate,
    }
    _atomic_write(args.output.resolve(), output)
    print(json.dumps({"status": "valid", "neural_training_authorized": gate}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
