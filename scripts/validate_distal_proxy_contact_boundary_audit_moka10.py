#!/usr/bin/env python3
"""Independently validate the ellipsoid/contact boundary audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from main.multilink_ellipsoid.proxy_contact_boundary_audit import (
    RESULT_SCHEMA, VALIDATION_SCHEMA, analyze, load_config, paired_records,
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
    parser.add_argument("--expanded-dataset", type=Path, required=True)
    parser.add_argument("--expanded-oracle", type=Path, required=True)
    parser.add_argument("--validation-fresh", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = {key: value.resolve() for key, value in {
        "result": args.result, "config": args.config,
        "dataset": args.expanded_dataset, "oracle": args.expanded_oracle,
        "fresh": args.validation_fresh, "output": args.output,
    }.items()}
    result = _load(paths["result"])
    config = load_config(paths["config"])
    source = config["immutable_source"]
    _require(
        _file_sha256(paths["dataset"])
        == source["expanded_dataset_file_sha256"]
        and _file_sha256(paths["oracle"])
        == source["expanded_oracle_file_sha256"]
        and _file_sha256(paths["fresh"])
        == source["validation_fresh_file_sha256"],
        "proxy-contact validation immutable inputs differ",
    )
    recomputed = analyze(
        paired_records(
            _load(paths["dataset"]), _load(paths["oracle"])["state_results"],
            _load(paths["fresh"]), config,
        ),
        config,
    )
    recorded = {
        key: result[key] for key in (
            "population", "zero_threshold_confusion", "threshold_sweep",
            "absolute_proxy_boundary_bands", "split_zero_threshold_confusion",
            "decision",
        )
    }
    _require(
        result.get("schema_version") == RESULT_SCHEMA
        and result.get("status") == "complete"
        and result.get("scientific_result") is True
        and result.get("source", {}).get("commit") == args.expected_commit
        and result.get("source", {}).get("dirty") is False
        and result.get("config", {}).get("config_file_sha256")
        == config["config_file_sha256"]
        and result.get("result_payload_sha256")
        == _hash_without(result, "result_payload_sha256")
        and recorded == recomputed
        and result["decision"]["training_executed"] is False
        and result["decision"]["QP_executed"] is False
        and result["decision"]["new_simulation_executed"] is False
        and result["decision"]["closed_loop_E05_remains_blocked"] is True,
        "proxy-contact result differs",
    )
    output = {
        "schema_version": VALIDATION_SCHEMA, "status": "valid",
        "scientific_result": True, "expected_commit": args.expected_commit,
        "result_file_sha256": _file_sha256(paths["result"]),
        "result_payload_sha256": result["result_payload_sha256"],
        "zero_threshold_confusion": result["zero_threshold_confusion"],
        "audit_complete": result["decision"]["audit_complete"],
        "initial_state_audit_authorized": result["decision"][
            "initial_state_audit_authorized"
        ],
        "closed_loop_E05_authorized": False,
    }
    _atomic_write(paths["output"], output)
    print(json.dumps(output, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
