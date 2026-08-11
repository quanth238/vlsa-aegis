#!/usr/bin/env python3
"""Run the frozen factorized input-representation audit on H100."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any, Mapping

from main.multilink_ellipsoid.factorized_input_representation_audit import (
    RESULT_SCHEMA, load_audit_config, payload_sha256, run_audit,
)
from main.multilink_ellipsoid.shadow import allocation_record
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _git_identity, _load, _require,
)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--complete-dataset", type=Path, required=True)
    parser.add_argument("--trajectory-metadata", type=Path, required=True)
    parser.add_argument("--trajectory-array", type=Path, required=True)
    parser.add_argument("--flat-model", type=Path, required=True)
    parser.add_argument("--flat-predictions", type=Path, required=True)
    parser.add_argument("--flat-result", type=Path, required=True)
    parser.add_argument("--flat-validation", type=Path, required=True)
    parser.add_argument("--time-model", type=Path, required=True)
    parser.add_argument("--time-predictions", type=Path, required=True)
    parser.add_argument("--time-result", type=Path, required=True)
    parser.add_argument("--time-validation", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)


def resolved_paths(args: argparse.Namespace) -> dict[str, Path]:
    return {key: value.resolve() for key, value in {
        "repo": args.repo_root, "config": args.config,
        "complete_dataset": args.complete_dataset,
        "metadata": args.trajectory_metadata, "array": args.trajectory_array,
        "flat_model": args.flat_model, "flat_predictions": args.flat_predictions,
        "flat_result": args.flat_result, "flat_validation": args.flat_validation,
        "time_model": args.time_model, "time_predictions": args.time_predictions,
        "time_result": args.time_result, "time_validation": args.time_validation,
    }.items()}


def load_inputs(paths: Mapping[str, Path]) -> tuple[Any, ...]:
    import numpy as np

    config = load_audit_config(paths["config"])
    source = config["immutable_source"]
    for source_key, path_key in (
        ("complete_dataset_file_sha256", "complete_dataset"),
        ("trajectory_metadata_file_sha256", "metadata"),
        ("trajectory_array_file_sha256", "array"),
        ("flat_model_file_sha256", "flat_model"),
        ("flat_predictions_file_sha256", "flat_predictions"),
        ("flat_result_file_sha256", "flat_result"),
        ("flat_validation_file_sha256", "flat_validation"),
        ("time_model_file_sha256", "time_model"),
        ("time_predictions_file_sha256", "time_predictions"),
        ("time_result_file_sha256", "time_result"),
        ("time_validation_file_sha256", "time_validation"),
    ):
        _require(
            _file_sha256(paths[path_key]) == source[source_key],
            "input-representation immutable %s differs" % path_key,
        )
    complete_dataset = _load(paths["complete_dataset"])
    metadata = _load(paths["metadata"])
    flat_result = _load(paths["flat_result"])
    flat_validation = _load(paths["flat_validation"])
    time_result = _load(paths["time_result"])
    time_validation = _load(paths["time_validation"])
    population = config["population"]
    _require(
        int(complete_dataset["complete_input_dimension"])
        == int(population["state_input_dimension"])
        and int(metadata["summary"]["rollout_count"])
        == int(population["rollout_count"])
        and int(metadata["summary"]["state_count"])
        == int(population["state_count"])
        and flat_result.get("decision", {}).get("one_sided_geometry_GO") is False
        and flat_validation.get("valid") is True
        and time_result.get("decision", {}).get(
            "time_conditioned_mechanism_GO"
        ) is False
        and time_validation.get("valid") is True,
        "input-representation validated source contract differs",
    )
    archive = np.load(paths["array"], allow_pickle=False)
    return config, complete_dataset, metadata, archive


def write_records(path: Path, records: Mapping[str, Any]) -> dict[str, Any]:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".%s.tmp.npz" % path.stem)
    np.savez_compressed(temporary, **records)
    temporary.replace(path)
    return {"path": str(path), "file_sha256": _file_sha256(path)}


def main() -> int:
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter_ns()
    paths = resolved_paths(args)
    paths.update({"records": args.records.resolve(), "output": args.output.resolve()})
    config, complete_dataset, metadata, archive = load_inputs(paths)
    core, records = run_audit(
        config=config, complete_dataset=complete_dataset, metadata=metadata,
        archive=archive, flat_model_path=paths["flat_model"],
        flat_predictions_path=paths["flat_predictions"],
        time_model_path=paths["time_model"],
        time_predictions_path=paths["time_predictions"],
    )
    records_receipt = write_records(paths["records"], records)
    result = {
        "schema_version": RESULT_SCHEMA, "status": "complete",
        "scientific_result": True, "claim_scope": config["claim_scope"],
        "source": _git_identity(paths["repo"], args.expected_commit),
        "allocation": allocation_record(), "config": config,
        "immutable_artifacts": {
            key: {"path": str(paths[key]), "file_sha256": _file_sha256(paths[key])}
            for key in (
                "complete_dataset", "metadata", "array", "flat_model",
                "flat_predictions", "flat_result", "flat_validation",
                "time_model", "time_predictions", "time_result", "time_validation",
            )
        },
        "audit": core, "records": records_receipt,
        "forbidden_action_receipt": {
            key: False for key in config["forbidden_actions"]
        },
        "wall_seconds": (time.perf_counter_ns() - started) * 1.0e-9,
    }
    result["result_payload_sha256"] = payload_sha256(
        result, "result_payload_sha256"
    )
    _atomic_write(paths["output"], result)
    print(json.dumps({
        "normalization": core["normalization"],
        "causal_group_mask": core["causal_group_mask"],
        "decision": core["decision"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
