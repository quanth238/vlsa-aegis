#!/usr/bin/env python3
"""Collect one exact post-AEGIS outward-normal risk curve."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from scripts.evaluate_distal_query_action_risk_e05 import evaluate
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _load, _sha256,
)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def collect(
    *, repo_root: Path, population_manifest: Path, selection_manifest: Path,
    table1_root: Path, geometry_config: Path, base_risk_config: Path,
    curve_config_path: Path, case_index: int, expected_commit: str,
    output_path: Path,
) -> dict[str, Any]:
    import numpy as np

    from main.multilink_ellipsoid.normal_risk_curve import (
        RESULT_SCHEMA, candidate_definitions, curve_summary, load_config,
    )

    curve_config = load_config(curve_config_path)
    selection = [json.loads(line) for line in selection_manifest.read_text().splitlines() if line]
    if not 0 <= int(case_index) < len(selection):
        raise ValueError("normal risk-curve case index differs")
    selected = selection[int(case_index)]
    archived = table1_root / selected["archived_result_relative_path"]
    if _file_sha256(archived) != selected["archived_result_file_sha256"]:
        raise ValueError("normal risk-curve archived result differs")
    result = evaluate(
        repo_root=repo_root,
        population_manifest_path=population_manifest,
        archived_path=archived,
        geometry_config_path=geometry_config,
        experiment_config_path=base_risk_config,
        expected_commit=expected_commit,
        output_path=output_path,
        case_id_override=selected["case_id"],
        state_step_override=int(curve_config["warning_steps"][int(case_index)]),
        query_index_override=int(curve_config["query_indices"][int(case_index)]),
        result_schema_override=RESULT_SCHEMA,
        claim_scope_override=curve_config["claim_scope"],
        population_binding={
            "case_index": int(case_index),
            "selection": selected,
            "selection_manifest": str(selection_manifest),
            "selection_manifest_sha256": _file_sha256(selection_manifest),
            "diagnostic_after_opening": True,
        },
        candidate_definitions_override=lambda nominal, frame, _base: candidate_definitions(
            nominal, frame, curve_config
        ),
        candidate_protocol_binding=curve_config,
        apply_released_aegis_ee_to_all_proposed_actions=True,
    )
    nominal = np.asarray(result["nominal_five_action_chunk"], dtype=np.float64)
    for candidate in result["candidates"]:
        exact = np.asarray(candidate["actions"], dtype=np.float64)
        candidate["effective_post_AEGIS_correction_l2_action"] = float(
            np.linalg.norm(exact[:, :3] - nominal[:, :3])
        )
        candidate["active_combined_row"] = int(np.argmax(candidate["combined_risk"]))
        row = int(candidate["active_combined_row"])
        prefix = float(candidate["candidate_prefix_risk"][row])
        backup = candidate["backup_risk"]
        backup_value = None if backup is None else float(backup[row])
        candidate["active_phase"] = (
            "prefix" if backup_value is None or prefix >= backup_value else "backup"
        )
    result["curve_config"] = curve_config
    result["curve_summary"] = curve_summary(
        result["candidates"],
        tolerance_m=float(curve_config["monotonicity_tolerance_m"]),
    )
    result["interpretation"] = (
        "exact_normal_risk_curve_has_safe_support"
        if result["curve_summary"]["safe_candidate_count"] > 0
        else "exact_normal_risk_curve_no_safe_support"
    )
    result.pop("result_payload_sha256", None)
    result["result_payload_sha256"] = _sha256(_canonical(result))
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--population-manifest", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--table1-root", type=Path, required=True)
    parser.add_argument("--geometry-config", type=Path, required=True)
    parser.add_argument("--base-risk-config", type=Path, required=True)
    parser.add_argument("--curve-config", type=Path, required=True)
    parser.add_argument("--case-index", type=int, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = collect(
        repo_root=args.repo_root.resolve(),
        population_manifest=args.population_manifest.resolve(),
        selection_manifest=args.selection_manifest.resolve(),
        table1_root=args.table1_root.resolve(),
        geometry_config=args.geometry_config.resolve(),
        base_risk_config=args.base_risk_config.resolve(),
        curve_config_path=args.curve_config.resolve(),
        case_index=args.case_index,
        expected_commit=args.expected_commit,
        output_path=args.output.resolve(),
    )
    _atomic_write(args.output.resolve(), result)
    print(json.dumps({
        "case_id": result["population_binding"]["selection"]["case_id"],
        "interpretation": result["interpretation"],
        "curve_summary": result["curve_summary"],
        "result_payload_sha256": result["result_payload_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
