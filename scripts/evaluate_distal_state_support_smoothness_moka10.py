#!/usr/bin/env python3
"""Evaluate grouped feature support and regional-oracle smoothness."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping

from main.multilink_ellipsoid.shadow import allocation_record
from main.multilink_ellipsoid.state_support_smoothness import (
    RESULT_SCHEMA, analyze, load_config,
)
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _git_identity, _load, _require,
)


def _hash_without(value: Mapping[str, Any], key: str) -> str:
    payload = dict(value)
    payload.pop(key, None)
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--multi-region-result", type=Path, required=True)
    parser.add_argument("--multi-region-validation", type=Path, required=True)
    parser.add_argument("--learned-result", type=Path, required=True)
    parser.add_argument("--learned-validation", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter_ns()
    paths = {
        name: value.resolve() for name, value in {
            "repo": args.repo_root, "dataset": args.dataset,
            "multi": args.multi_region_result,
            "multi_validation": args.multi_region_validation,
            "learned": args.learned_result,
            "learned_validation": args.learned_validation,
            "config": args.config, "output": args.output,
        }.items()
    }
    config = load_config(paths["config"])
    source = config["immutable_source"]
    for key, path_key in (
        ("dataset_file_sha256", "dataset"),
        ("multi_region_result_file_sha256", "multi"),
        ("multi_region_validation_file_sha256", "multi_validation"),
        ("learned_result_file_sha256", "learned"),
        ("learned_validation_file_sha256", "learned_validation"),
    ):
        _require(_file_sha256(paths[path_key]) == source[key],
                 "support/smoothness immutable file hash differs: %s" % key)
    dataset = _load(paths["dataset"])
    multi = _load(paths["multi"])
    multi_validation = _load(paths["multi_validation"])
    learned = _load(paths["learned"])
    learned_validation = _load(paths["learned_validation"])
    _require(
        dataset.get("dataset_payload_sha256")
        == source["dataset_payload_sha256"]
        and multi.get("result_payload_sha256")
        == source["multi_region_result_payload_sha256"]
        and learned.get("result_payload_sha256")
        == source["learned_result_payload_sha256"]
        and multi_validation.get("status") == "valid"
        and learned_validation.get("status") == "valid"
        and learned_validation.get("closed_loop_e05_authorized") is False,
        "support/smoothness immutable payload or prior decision differs",
    )
    analysis = analyze(dataset, multi, config)
    output = {
        "schema_version": RESULT_SCHEMA, "status": "complete",
        "scientific_result": True, "claim_scope": config["claim_scope"],
        "source": _git_identity(paths["repo"], args.expected_commit),
        "allocation": allocation_record(), "config": config,
        "immutable_inputs": {
            "dataset_file_sha256": _file_sha256(paths["dataset"]),
            "multi_region_result_file_sha256": _file_sha256(paths["multi"]),
            "multi_region_validation_file_sha256": _file_sha256(
                paths["multi_validation"]
            ),
            "learned_result_file_sha256": _file_sha256(paths["learned"]),
            "learned_validation_file_sha256": _file_sha256(
                paths["learned_validation"]
            ),
        },
        "feature_shift": analysis["feature_shift"],
        "reference": analysis["reference"],
        "validation_state_results": analysis["validation_state_results"],
        "test_state_results": analysis["test_state_results"],
        "aggregates": analysis["aggregates"],
        "decision": analysis["decision"],
        "wall_seconds": (time.perf_counter_ns() - started) * 1.0e-9,
    }
    output["result_payload_sha256"] = _hash_without(
        output, "result_payload_sha256"
    )
    _atomic_write(paths["output"], output)
    print(json.dumps({
        "feature_shift": {
            "global_test_maximum_abs_z": output["feature_shift"][
                "global_test_maximum_abs_z"
            ],
            "top_feature": output["feature_shift"]["top_features"][0],
        },
        "aggregates": output["aggregates"],
        "decision": output["decision"], "output": str(paths["output"]),
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
