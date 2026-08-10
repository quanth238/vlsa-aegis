#!/usr/bin/env python3
"""Evaluate ellipsoid/contact confusion on immutable cloned-OSC rollouts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from main.multilink_ellipsoid.proxy_contact_boundary_audit import (
    RESULT_SCHEMA, analyze, load_config, paired_records,
)
from main.multilink_ellipsoid.supported_region_aware_mlp import fresh_payload
from main.multilink_ellipsoid.targeted_boundary_expansion import (
    DATASET_SCHEMA, ORACLE_SCHEMA,
)
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _git_identity, _load, _require,
)


def _hash_without(value: Mapping[str, Any], key: str) -> str:
    payload = dict(value)
    payload.pop(key, None)
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


def main() -> int:
    from main.multilink_ellipsoid.shadow import allocation_record

    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expanded-dataset", type=Path, required=True)
    parser.add_argument("--expanded-oracle", type=Path, required=True)
    parser.add_argument("--validation-fresh", type=Path, required=True)
    parser.add_argument("--target-result", type=Path, required=True)
    parser.add_argument("--target-validation", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = {key: value.resolve() for key, value in {
        "repo": args.repo_root, "config": args.config,
        "dataset": args.expanded_dataset, "oracle": args.expanded_oracle,
        "fresh": args.validation_fresh, "target_result": args.target_result,
        "target_validation": args.target_validation, "output": args.output,
    }.items()}
    config = load_config(paths["config"])
    source = config["immutable_source"]
    for path, expected, label in (
        (paths["dataset"], source["expanded_dataset_file_sha256"], "dataset"),
        (paths["oracle"], source["expanded_oracle_file_sha256"], "oracle"),
        (paths["fresh"], source["validation_fresh_file_sha256"], "fresh"),
        (paths["target_result"], source["target_authority_result_file_sha256"], "target-result"),
        (paths["target_validation"], source["target_authority_validation_file_sha256"], "target-validation"),
    ):
        _require(_file_sha256(path) == expected, f"immutable {label} differs")
    dataset = _load(paths["dataset"])
    oracle = _load(paths["oracle"])
    fresh = _load(paths["fresh"])
    target_result = _load(paths["target_result"])
    target_validation = _load(paths["target_validation"])
    _require(
        dataset.get("schema_version") == DATASET_SCHEMA
        and dataset.get("dataset_payload_sha256")
        == source["expanded_dataset_payload_sha256"]
        == _hash_without(dataset, "dataset_payload_sha256")
        and oracle.get("schema_version") == ORACLE_SCHEMA
        and oracle.get("oracle_payload_sha256")
        == source["expanded_oracle_payload_sha256"]
        == _hash_without(oracle, "oracle_payload_sha256")
        and fresh.get("validation_fresh_payload_sha256")
        == source["validation_fresh_payload_sha256"] == fresh_payload(fresh)
        and target_result.get("result_payload_sha256")
        == source["target_authority_result_payload_sha256"]
        == _hash_without(target_result, "result_payload_sha256")
        and target_validation.get("status") == "valid"
        and target_validation.get("target_authority_gate_pass") is True,
        "proxy-contact authorization differs",
    )
    analysis = analyze(
        paired_records(dataset, oracle["state_results"], fresh, config), config
    )
    output = {
        "schema_version": RESULT_SCHEMA, "status": "complete",
        "scientific_result": True, "claim_scope": config["claim_scope"],
        "source": _git_identity(paths["repo"], args.expected_commit),
        "allocation": allocation_record(), "config": config,
        "immutable_inputs": {
            "expanded_dataset_file_sha256": _file_sha256(paths["dataset"]),
            "expanded_oracle_file_sha256": _file_sha256(paths["oracle"]),
            "validation_fresh_file_sha256": _file_sha256(paths["fresh"]),
            "target_authority_result_file_sha256": _file_sha256(paths["target_result"]),
            "target_authority_validation_file_sha256": _file_sha256(paths["target_validation"]),
        },
        **analysis,
    }
    output["result_payload_sha256"] = _hash_without(
        output, "result_payload_sha256"
    )
    _atomic_write(paths["output"], output)
    print(json.dumps({
        "zero_threshold_confusion": output["zero_threshold_confusion"],
        "decision": output["decision"], "output": str(paths["output"]),
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
