#!/usr/bin/env python3
"""Independently validate the local residual-bound safety gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from main.multilink_ellipsoid.local_residual_bound import (
    OOF_SCHEMA, RESULT_SCHEMA, VALIDATION_SCHEMA, load_config, target_values,
)
from scripts.evaluate_distal_local_residual_bound_moka10 import _aggregates
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
    import numpy as np

    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--oof-artifact", type=Path, required=True)
    parser.add_argument("--expanded-dataset", type=Path, required=True)
    parser.add_argument("--expanded-oracle", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = _load(args.result.resolve())
    dataset = _load(args.expanded_dataset.resolve())
    oracle = _load(args.expanded_oracle.resolve())
    config = load_config(args.config.resolve())
    source = config["immutable_source"]
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
        and result.get("oof_artifact", {}).get("file_sha256")
        == _file_sha256(args.oof_artifact.resolve())
        and _file_sha256(args.expanded_dataset.resolve())
        == source["expanded_dataset_file_sha256"]
        and _file_sha256(args.expanded_oracle.resolve())
        == source["expanded_oracle_file_sha256"],
        "local-residual result identity differs",
    )
    with np.load(args.oof_artifact.resolve(), allow_pickle=False) as archive:
        metadata = json.loads(bytes(
            archive["metadata_utf8"].tolist()
        ).decode("utf-8"))
        residual = np.asarray(archive["residual_m"], dtype=np.float64)
        features = np.asarray(archive["features"], dtype=np.float64)
        state_index = np.asarray(archive["state_index"], dtype=np.int64)
        constraint_index = np.asarray(
            archive["constraint_index"], dtype=np.int64
        )
    train_indexes = {
        int(state["state_index"]) for state in dataset["state_records"]
        if state["split"] == "train"
    }
    _require(
        metadata.get("schema_version") == OOF_SCHEMA
        and metadata.get("source_commit") == args.expected_commit
        and len(residual) == int(config["leave_one_episode_out"][
            "expected_residual_count"
        ])
        and features.shape == (len(residual), 46)
        and set(state_index.tolist()) == train_indexes
        and set(constraint_index.tolist()) == set(range(7))
        and np.all(np.isfinite(features)) and np.all(np.isfinite(residual)),
        "local-residual OOF artifact differs",
    )
    oracle_by_index = {
        int(item["state_index"]): item for item in oracle["state_results"]
    }
    for state in result["test_state_results"]:
        source_state = oracle_by_index[int(state["state_index"])]
        targets = state["predicted_targets"]
        learned = []
        oracle_flags = []
        true_flags = []
        for action in source_state["fresh_actions"]:
            learned.append(bool(any(
                all(value >= 0.0 for value in target_values(
                    targets[int(region_index)], action["candidate_xyz"]
                ))
                for region_index in action["containing_region_indexes"]
            )))
            oracle_flags.append(bool(
                action["arms"]["0mm"]["multi_region_predicted_safe"]
            ))
            true_flags.append(bool(action["arms"]["0mm"]["true_safe"]))
        false_indexes = [
            index for index, (accepted, safe)
            in enumerate(zip(learned, true_flags))
            if accepted and not safe
        ]
        accepted_true = sum(
            accepted and safe for accepted, safe in zip(learned, true_flags)
        )
        _require(
            learned == state["learned_accepted_flags"]
            and oracle_flags == state["oracle_accepted_flags"]
            and true_flags == state["true_safe_flags"]
            and false_indexes == state["false_safe_fresh_indexes"]
            and accepted_true == int(state["accepted_true_safe_action_count"]),
            "local-residual state decision differs",
        )
    aggregates = _aggregates(result["test_state_results"])
    for key, value in aggregates.items():
        recorded = result["test_aggregates"].get(key)
        if isinstance(value, float):
            _require(
                abs(value - float(recorded)) <= 1.0e-15,
                "local-residual aggregate differs: %s" % key,
            )
        else:
            _require(
                value == recorded,
                "local-residual aggregate differs: %s" % key,
            )
    gates = config["learned_gate"]
    preliminary = bool(
        aggregates["off_grid_false_safe_action_count"]
        == int(gates["test_off_grid_false_safe_action_count"])
        and aggregates["safe_support_state_count"]
        == int(gates["required_test_state_safe_support_count"])
    )
    learned_pass = bool(
        preliminary
        and aggregates["valid_selected_QP_count"]
        == int(gates["required_valid_selected_QP_count"])
        and aggregates["fresh_exact_safe_selected_QP_count"]
        == int(gates["required_fresh_exact_safe_selected_QP_count"])
    )
    _require(
        result["decision"]["off_grid_preliminary_gate_pass"] is preliminary
        and result["decision"]["learned_gate_pass"] is learned_pass
        and result["decision"][
            "receding_closed_loop_E05_preregistration_authorized"
        ] is learned_pass
        and result["decision"]["closed_loop_e05_executed"] is False,
        "local-residual decision differs",
    )
    output = {
        "schema_version": VALIDATION_SCHEMA, "status": "valid",
        "scientific_result": True, "expected_commit": args.expected_commit,
        "result_file_sha256": _file_sha256(args.result.resolve()),
        "result_payload_sha256": result["result_payload_sha256"],
        "oof_artifact_file_sha256": _file_sha256(
            args.oof_artifact.resolve()
        ),
        "aggregates": aggregates, "learned_gate_pass": learned_pass,
        "receding_closed_loop_E05_preregistration_authorized": learned_pass,
    }
    _atomic_write(args.output.resolve(), output)
    print(json.dumps(output, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
