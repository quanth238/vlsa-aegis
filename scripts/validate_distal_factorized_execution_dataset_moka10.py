#!/usr/bin/env python3
"""Independently validate the factorized-execution trajectory dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from main.multilink_ellipsoid.factorized_execution_pilot import (
    COLLECTION_SCHEMA, DATASET_SCHEMA, DATASET_VALIDATION_SCHEMA,
    candidate_chunks, dataset_arrays, load_config, payload_sha256,
    sensitivity_arrays,
)
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _load, _require,
)


def main() -> int:
    import numpy as np

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--array-dataset", type=Path, required=True)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config.resolve())
    metadata = _load(args.metadata.resolve())
    collection = _load(args.collection.resolve())
    _require(
        metadata.get("schema_version") == DATASET_SCHEMA
        and metadata.get("dataset_payload_sha256")
        == payload_sha256(metadata, "dataset_payload_sha256")
        and metadata.get("config_file_sha256") == config["config_file_sha256"]
        and metadata["array_dataset"]["file_sha256"]
        == _file_sha256(args.array_dataset.resolve())
        and collection.get("schema_version") == COLLECTION_SCHEMA
        and collection.get("result_payload_sha256")
        == payload_sha256(collection, "result_payload_sha256")
        and collection["trajectory_dataset"]["metadata_file_sha256"]
        == _file_sha256(args.metadata.resolve())
        and collection["trajectory_dataset"]["array_file_sha256"]
        == _file_sha256(args.array_dataset.resolve()),
        "factorized-execution dataset identity differs",
    )
    archive = np.load(args.array_dataset.resolve(), allow_pickle=False)
    arrays = dataset_arrays(metadata, archive)
    sensitivity = sensitivity_arrays(arrays)
    state_ids = sorted(set(np.asarray(arrays["state_index"], dtype=np.int64).tolist()))
    candidate_pairing = True
    for state in state_ids:
        rows = np.flatnonzero(np.asarray(arrays["state_index"]) == state)
        order = rows[np.argsort(np.asarray(arrays["candidate_index"])[rows])]
        nominal = np.asarray(arrays["action_chunk"])[order[0]]
        expected = candidate_chunks(nominal, state, config)
        observed = np.asarray(arrays["action_chunk"])[order]
        regenerated = np.asarray([
            item["full_two_action_commands"] for item in expected
        ], dtype=np.float64)
        candidate_pairing = bool(candidate_pairing and np.array_equal(
            observed, regenerated
        ))
    split_counts = {
        name: len(set(np.asarray(arrays["state_index"])[
            np.asarray(arrays["split"], dtype=object) == name
        ].tolist()))
        for name in ("train", "validation", "test")
    }
    expected_sensitivity_rows = int(config["immutable_source"]["expected_state_count"]) * 14
    valid = bool(
        collection["decision"]["collection_gate_pass"]
        and not collection["determinism"]["repeat_mismatches"]
        and not collection["pairing"]["nominal_mismatches"]
        and int(metadata["summary"]["rollout_count"])
        == int(config["candidate_design"]["expected_rollout_count"])
        and split_counts == config["immutable_source"]["expected_state_split_counts"]
        and candidate_pairing
        and len(sensitivity["state_index"]) == expected_sensitivity_rows
        and np.all(np.isfinite(sensitivity["joint_sensitivity_rad_per_action"]))
        and np.all(np.isfinite(sensitivity["margin_sensitivity_m_per_action"]))
    )
    result = {
        "schema_version": DATASET_VALIDATION_SCHEMA,
        "status": "complete", "valid": valid,
        "config_file_sha256": config["config_file_sha256"],
        "metadata_file_sha256": _file_sha256(args.metadata.resolve()),
        "metadata_payload_sha256": metadata["dataset_payload_sha256"],
        "array_file_sha256": _file_sha256(args.array_dataset.resolve()),
        "collection_file_sha256": _file_sha256(args.collection.resolve()),
        "collection_payload_sha256": collection["result_payload_sha256"],
        "audit": {
            "split_state_counts": split_counts,
            "candidate_actions_exactly_regenerated": candidate_pairing,
            "sensitivity_row_count": int(len(sensitivity["state_index"])),
            "expected_sensitivity_row_count": expected_sensitivity_rows,
            "joint_trace_sha256": __import__("hashlib").sha256(
                np.asarray(arrays["joint_position_rad"]).tobytes()
            ).hexdigest(),
            "margin_trace_sha256": __import__("hashlib").sha256(
                np.asarray(arrays["ellipsoid_clearance_m"]).tobytes()
            ).hexdigest(),
        },
    }
    result["validation_payload_sha256"] = payload_sha256(
        result, "validation_payload_sha256"
    )
    _atomic_write(args.output.resolve(), result)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
