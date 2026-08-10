#!/usr/bin/env python3
"""Audit immutable ellipsoid margins against raw distal MuJoCo contact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from main.multilink_ellipsoid.margin_target_authority import (
    RESULT_SCHEMA, analyze, load_config,
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
    parser.add_argument("--action-conditioned-result", type=Path, required=True)
    parser.add_argument("--action-conditioned-validation", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = {key: value.resolve() for key, value in {
        "repo": args.repo_root, "config": args.config,
        "dataset": args.expanded_dataset, "oracle": args.expanded_oracle,
        "validation_fresh": args.validation_fresh,
        "action_result": args.action_conditioned_result,
        "action_validation": args.action_conditioned_validation,
        "output": args.output,
    }.items()}
    config = load_config(paths["config"])
    source = config["immutable_source"]
    for path, expected, label in (
        (paths["dataset"], source["expanded_dataset_file_sha256"], "dataset"),
        (paths["oracle"], source["expanded_oracle_file_sha256"], "oracle"),
        (paths["validation_fresh"], source["validation_fresh_file_sha256"], "validation-fresh"),
        (paths["action_result"], source["action_conditioned_result_file_sha256"], "action-result"),
        (paths["action_validation"], source["action_conditioned_validation_file_sha256"], "action-validation"),
    ):
        _require(
            _file_sha256(path) == expected,
            "margin-target immutable %s differs" % label,
        )
    dataset = _load(paths["dataset"])
    oracle = _load(paths["oracle"])
    fresh = _load(paths["validation_fresh"])
    action_result = _load(paths["action_result"])
    action_validation = _load(paths["action_validation"])
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
        == source["validation_fresh_payload_sha256"]
        == fresh_payload(fresh)
        and action_result.get("result_payload_sha256")
        == source["action_conditioned_result_payload_sha256"]
        == _hash_without(action_result, "result_payload_sha256")
        and action_result.get("decision", {}).get("learned_gate_pass") is False
        and action_validation.get("status") == "valid"
        and action_validation.get("plain_action_conditioned_MLP_rejected") is True,
        "margin-target authorization differs",
    )
    analysis = analyze(
        dataset, oracle["state_results"], fresh, config
    )
    output = {
        "schema_version": RESULT_SCHEMA, "status": "complete",
        "scientific_result": True, "claim_scope": config["claim_scope"],
        "source": _git_identity(paths["repo"], args.expected_commit),
        "allocation": allocation_record(), "config": config,
        "immutable_inputs": {
            "expanded_dataset_file_sha256": _file_sha256(paths["dataset"]),
            "expanded_oracle_file_sha256": _file_sha256(paths["oracle"]),
            "validation_fresh_file_sha256": _file_sha256(paths["validation_fresh"]),
            "action_conditioned_result_file_sha256": _file_sha256(paths["action_result"]),
        },
        **analysis,
    }
    output["result_payload_sha256"] = _hash_without(
        output, "result_payload_sha256"
    )
    _atomic_write(paths["output"], output)
    print(json.dumps({
        "aggregates": output["aggregates"],
        "decision": output["decision"], "output": str(paths["output"]),
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
