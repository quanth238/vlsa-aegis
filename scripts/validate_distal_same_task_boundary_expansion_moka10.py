#!/usr/bin/env python3
"""Independently recompute the same-task expansion and support decision."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from main.multilink_ellipsoid.same_task_boundary_expansion import (
    RESULT_SCHEMA, VALIDATION_SCHEMA, evaluate_expansion, load_config,
)
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _load, _require,
)


def _hash_without(value: Mapping[str, Any], key: str) -> str:
    payload = dict(value)
    payload.pop(key, None)
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--original-dataset", type=Path, required=True)
    parser.add_argument("--additional-dataset", type=Path, required=True)
    parser.add_argument("--collection-result", type=Path, required=True)
    parser.add_argument("--original-multi-region", type=Path, required=True)
    parser.add_argument("--expanded-dataset", type=Path, required=True)
    parser.add_argument("--expanded-oracle", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = _load(args.result.resolve())
    config = load_config(args.config.resolve())
    original = _load(args.original_dataset.resolve())
    additional = _load(args.additional_dataset.resolve())
    collection_result = _load(args.collection_result.resolve())
    original_multi = _load(args.original_multi_region.resolve())
    expanded, oracle, analysis = evaluate_expansion(
        original, additional, original_multi, config
    )
    _require(
        result.get("schema_version") == RESULT_SCHEMA
        and result.get("status") == "complete"
        and result.get("scientific_result") is True
        and result.get("source", {}).get("commit") == args.expected_commit
        and result.get("source", {}).get("dirty") is False
        and result.get("config", {}).get("config_file_sha256")
        == config["config_file_sha256"]
        and result.get("result_payload_sha256")
        == _hash_without(result, "result_payload_sha256"),
        "same-task expansion result identity differs",
    )
    _require(
        collection_result.get("source", {}).get("commit") == args.expected_commit
        and collection_result.get("source", {}).get("dirty") is False
        and collection_result.get("result_payload_sha256")
        == _hash_without(collection_result, "result_payload_sha256")
        and result.get("additional_collection", {}).get("result_file_sha256")
        == _file_sha256(args.collection_result.resolve()),
        "same-task expansion collection result identity differs",
    )
    _require(
        _load(args.expanded_dataset.resolve()) == expanded
        and _load(args.expanded_oracle.resolve()) == oracle,
        "same-task expansion derived artifact recomputation differs",
    )
    for key in (
        "feature_shift", "reference", "validation_state_results",
        "test_state_results", "aggregates", "decision",
    ):
        _require(result.get(key) == analysis[key],
                 "same-task expansion analysis differs: %s" % key)
    output = {
        "schema_version": VALIDATION_SCHEMA, "status": "valid",
        "scientific_result": True, "expected_commit": args.expected_commit,
        "result_file_sha256": _file_sha256(args.result.resolve()),
        "result_payload_sha256": result["result_payload_sha256"],
        "expanded_dataset_file_sha256": _file_sha256(
            args.expanded_dataset.resolve()
        ),
        "expanded_oracle_file_sha256": _file_sha256(
            args.expanded_oracle.resolve()
        ),
        "aggregates": result["aggregates"], "decision": result["decision"],
    }
    _atomic_write(args.output.resolve(), output)
    print(json.dumps(output, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
