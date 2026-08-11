#!/usr/bin/env python3
"""Independently replay the frozen input-representation audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from main.multilink_ellipsoid.execution_margin_nn import _canonical
from main.multilink_ellipsoid.factorized_input_representation_audit import (
    VALIDATION_SCHEMA, payload_sha256, run_audit,
)
from scripts.audit_distal_factorized_input_representation_moka10 import (
    add_arguments, load_inputs, resolved_paths,
)
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _git_identity, _load, _require,
)


def records_equal(path: Path, expected: Mapping[str, Any]) -> bool:
    import numpy as np

    archive = np.load(path, allow_pickle=False)
    if set(archive.files) != set(expected):
        return False
    return all(np.array_equal(archive[key], expected[key]) for key in expected)


def main() -> int:
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = resolved_paths(args)
    paths.update({
        "records": args.records.resolve(), "result": args.result.resolve(),
        "output": args.output.resolve(),
    })
    config, complete_dataset, metadata, archive = load_inputs(paths)
    result = _load(paths["result"])
    _require(
        result.get("result_payload_sha256")
        == payload_sha256(result, "result_payload_sha256"),
        "input-representation result payload differs",
    )
    fresh, expected_records = run_audit(
        config=config, complete_dataset=complete_dataset, metadata=metadata,
        archive=archive, flat_model_path=paths["flat_model"],
        flat_predictions_path=paths["flat_predictions"],
        time_model_path=paths["time_model"],
        time_predictions_path=paths["time_predictions"],
    )
    tests = {
        "source_identity_exact": result.get("source", {}).get("commit")
        == args.expected_commit,
        "audit_exact": _canonical(result.get("audit", {})) == _canonical(fresh),
        "records_file_hash_exact": result.get("records", {}).get("file_sha256")
        == _file_sha256(paths["records"]),
        "records_arrays_exact": records_equal(paths["records"], expected_records),
        "immutable_prediction_replay_exact": bool(
            fresh["prediction"]["immutable_flat_prediction_exact"]
            and fresh["prediction"]["immutable_time_prediction_exact"]
        ),
        "normalization_reproduction_exact": bool(
            fresh["normalization"][
                "train_mean_and_std_exactly_match_both_models"
            ]
        ),
        "forbidden_actions_respected": result.get("forbidden_action_receipt")
        == {key: False for key in config["forbidden_actions"]},
    }
    validation = {
        "schema_version": VALIDATION_SCHEMA,
        "source": _git_identity(paths["repo"], args.expected_commit),
        "result_file_sha256": _file_sha256(paths["result"]),
        "records_file_sha256": _file_sha256(paths["records"]),
        "tests": tests, "valid": bool(all(tests.values())),
        "decision": fresh["decision"],
    }
    validation["validation_payload_sha256"] = payload_sha256(
        validation, "validation_payload_sha256"
    )
    _atomic_write(paths["output"], validation)
    print(json.dumps(validation, sort_keys=True), flush=True)
    return 0 if validation["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
