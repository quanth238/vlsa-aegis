#!/usr/bin/env python3
"""Collect three same-task episodes and rerun the frozen support audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping

from main.multilink_ellipsoid.affine_coefficient_model import (
    load_affine_coefficient_config,
)
from main.multilink_ellipsoid.multi_region_affine_oracle import (
    load_config as load_multi_region_config,
)
from main.multilink_ellipsoid.same_task_boundary_expansion import (
    RESULT_SCHEMA, collection_config, evaluate_expansion, load_config,
    load_selected_manifest,
)
from main.multilink_ellipsoid.shadow import allocation_record
from scripts.collect_distal_affine_coefficient_moka10 import collect
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--population-manifest", type=Path, required=True)
    parser.add_argument("--selected-manifest", type=Path, required=True)
    parser.add_argument("--archived-root", type=Path, required=True)
    parser.add_argument("--geometry-config", type=Path, required=True)
    parser.add_argument("--exact-box-config", type=Path, required=True)
    parser.add_argument("--affine-config", type=Path, required=True)
    parser.add_argument("--multi-region-config", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--original-dataset", type=Path, required=True)
    parser.add_argument("--original-multi-region", type=Path, required=True)
    parser.add_argument("--original-multi-region-validation", type=Path, required=True)
    parser.add_argument("--prior-support-result", type=Path, required=True)
    parser.add_argument("--prior-support-validation", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--additional-dataset", type=Path, required=True)
    parser.add_argument("--collection-result", type=Path, required=True)
    parser.add_argument("--expanded-dataset", type=Path, required=True)
    parser.add_argument("--expanded-oracle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter_ns()
    paths = {
        name: value.resolve() for name, value in {
            "repo": args.repo_root, "population": args.population_manifest,
            "selected": args.selected_manifest, "archived": args.archived_root,
            "geometry": args.geometry_config, "exact_box": args.exact_box_config,
            "affine_config": args.affine_config,
            "multi_config": args.multi_region_config, "config": args.config,
            "original_dataset": args.original_dataset,
            "original_multi": args.original_multi_region,
            "original_multi_validation": args.original_multi_region_validation,
            "prior_support": args.prior_support_result,
            "prior_support_validation": args.prior_support_validation,
            "additional_dataset": args.additional_dataset,
            "collection_result": args.collection_result,
            "expanded_dataset": args.expanded_dataset,
            "expanded_oracle": args.expanded_oracle, "output": args.output,
        }.items()
    }
    config = load_config(paths["config"])
    source = config["immutable_source"]
    for path, key, label in (
        (paths["population"], "source_population_manifest_sha256", "population"),
        (paths["selected"], "selected_manifest_sha256", "selected"),
        (paths["geometry"], "geometry_config_file_sha256", "geometry"),
        (paths["exact_box"], "exact_box_config_file_sha256", "exact-box"),
        (paths["affine_config"], "affine_collection_config_file_sha256", "affine-config"),
        (paths["multi_config"], "multi_region_config_file_sha256", "multi-config"),
        (paths["original_dataset"], "original_dataset_file_sha256", "dataset"),
        (paths["original_multi"], "original_multi_region_file_sha256", "multi-region"),
        (paths["original_multi_validation"], "original_multi_region_validation_file_sha256", "multi-validation"),
        (paths["prior_support"], "prior_support_result_file_sha256", "prior-support"),
        (paths["prior_support_validation"], "prior_support_validation_file_sha256", "prior-validation"),
    ):
        _require(_file_sha256(path) == source[key],
                 "same-task expansion immutable %s differs" % label)
    original_dataset = _load(paths["original_dataset"])
    original_multi = _load(paths["original_multi"])
    original_multi_validation = _load(paths["original_multi_validation"])
    prior_support = _load(paths["prior_support"])
    prior_support_validation = _load(paths["prior_support_validation"])
    _require(
        original_dataset.get("dataset_payload_sha256")
        == source["original_dataset_payload_sha256"]
        == _hash_without(original_dataset, "dataset_payload_sha256")
        and original_multi.get("result_payload_sha256")
        == source["original_multi_region_payload_sha256"]
        == _hash_without(original_multi, "result_payload_sha256")
        and original_multi_validation.get("status") == "valid"
        and prior_support.get("result_payload_sha256")
        == source["prior_support_result_payload_sha256"]
        == _hash_without(prior_support, "result_payload_sha256")
        and prior_support_validation.get("status") == "valid"
        and prior_support.get("aggregates", {}).get("supported_test_state_count") == 0,
        "same-task expansion prior payload or decision differs",
    )
    affine_config = load_affine_coefficient_config(paths["affine_config"])
    multi_config = load_multi_region_config(paths["multi_config"])
    _require(
        config["multi_region"]["partition"] == multi_config["partition"]
        and config["multi_region"]["ridge_huber"] == multi_config["ridge_huber"],
        "same-task expansion regional settings differ from validated oracle",
    )
    selected = load_selected_manifest(paths["selected"], config)
    adapted = collection_config(config, affine_config)
    collection_result = collect(
        repo_root=paths["repo"], population_manifest_path=paths["population"],
        selected_manifest_path=paths["selected"], archived_root=paths["archived"],
        geometry_config_path=paths["geometry"],
        exact_box_config_path=paths["exact_box"], config_path=paths["config"],
        expected_commit=args.expected_commit,
        dataset_path=paths["additional_dataset"],
        config_override=adapted, selected_override=selected,
    )
    _atomic_write(paths["collection_result"], collection_result)
    additional = _load(paths["additional_dataset"])
    expanded, oracle, analysis = evaluate_expansion(
        original_dataset, additional, original_multi, config
    )
    _atomic_write(paths["expanded_dataset"], expanded)
    _atomic_write(paths["expanded_oracle"], oracle)
    output = {
        "schema_version": RESULT_SCHEMA, "status": "complete",
        "scientific_result": True, "claim_scope": config["claim_scope"],
        "source": _git_identity(paths["repo"], args.expected_commit),
        "allocation": allocation_record(), "config": config,
        "additional_collection": {
            "result_file_sha256": _file_sha256(paths["collection_result"]),
            "result_payload_sha256": collection_result[
                "result_payload_sha256"
            ],
            "dataset_file_sha256": _file_sha256(paths["additional_dataset"]),
            "dataset_payload_sha256": additional["dataset_payload_sha256"],
            "summary": additional["summary"],
            "collector_decision": collection_result["decision"],
        },
        "expanded_dataset": {
            "file_sha256": _file_sha256(paths["expanded_dataset"]),
            "payload_sha256": expanded["dataset_payload_sha256"],
            "summary": expanded["summary"],
        },
        "expanded_oracle": {
            "file_sha256": _file_sha256(paths["expanded_oracle"]),
            "payload_sha256": oracle["oracle_payload_sha256"],
            "summary": oracle["summary"],
        },
        "feature_shift": analysis["feature_shift"],
        "reference": analysis["reference"],
        "validation_state_results": analysis["validation_state_results"],
        "test_state_results": analysis["test_state_results"],
        "aggregates": analysis["aggregates"], "decision": analysis["decision"],
        "wall_seconds": (time.perf_counter_ns() - started) * 1.0e-9,
    }
    output["result_payload_sha256"] = _hash_without(
        output, "result_payload_sha256"
    )
    _atomic_write(paths["output"], output)
    print(json.dumps({
        "additional": output["additional_collection"]["summary"],
        "aggregates": output["aggregates"], "decision": output["decision"],
        "output": str(paths["output"]),
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
