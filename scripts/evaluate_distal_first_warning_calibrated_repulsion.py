#!/usr/bin/env python3
"""Run one first-warning calibrated complete-episode diagnostic case."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from scripts.evaluate_distal_analytical_repulsion_generalization import evaluate
from scripts.replay_distal_three_ellipsoid_multicbf import _atomic_write, _sha256


def _canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    from main.multilink_ellipsoid.first_warning_calibrated_repulsion import (
        RESULT_SCHEMA, load_config,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--population-manifest", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--table1-root", type=Path, required=True)
    parser.add_argument("--geometry-config", type=Path, required=True)
    parser.add_argument("--base-experiment-config", type=Path, required=True)
    parser.add_argument("--calibrated-config", type=Path, required=True)
    parser.add_argument("--case-index", type=int, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    config = load_config(args.calibrated_config.resolve())
    if not 0 <= int(args.case_index) < len(config["case_ids"]):
        raise ValueError("calibrated-repulsion case index differs")
    radius = float(config["requested_radius_by_case"][int(args.case_index)])
    binding = {
        "calibrated_config": config,
        "case_id": config["case_ids"][int(args.case_index)],
        "requested_radius": radius,
        "diagnostic_outcome_tuned": True,
        "online_adaptation": False,
    }
    result = evaluate(
        repo_root=args.repo_root.resolve(),
        population_manifest_path=args.population_manifest.resolve(),
        selection_manifest_path=args.selection_manifest.resolve(),
        table1_root=args.table1_root.resolve(),
        geometry_config_path=args.geometry_config.resolve(),
        experiment_config_path=args.base_experiment_config.resolve(),
        case_index=int(args.case_index),
        expected_commit=args.expected_commit,
        host=args.host,
        port=int(args.port),
        output_path=args.output.resolve(),
        requested_radius_override=radius,
        result_schema_override=RESULT_SCHEMA,
        claim_scope_override=config["claim_scope"],
        controller_binding=binding,
    )
    if result["case_id"] != binding["case_id"]:
        raise ValueError("calibrated-repulsion case binding differs")
    result.pop("result_payload_sha256", None)
    result["result_payload_sha256"] = _sha256(_canonical(result))
    _atomic_write(args.output.resolve(), result)
    print(json.dumps({
        "case_id": result["case_id"],
        "requested_radius": radius,
        "intervention_count": result["intervention_count"],
        "native_task_success": result["native_task_success"],
        "raw_L5_L7_contact_pass": result["raw_L5_L7_contact_pass"],
        "paper_car_pass": result["paper_car_pass"],
        "result_payload_sha256": result["result_payload_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
