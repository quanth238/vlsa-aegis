#!/usr/bin/env python3
"""Evaluate the exact registered normal candidate bank at E42 step 105."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional, Sequence

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
    table1_root: Path, geometry_config: Path, bank_config_path: Path,
    expected_commit: str, output_path: Path,
) -> dict[str, Any]:
    import numpy as np

    from main.multilink_ellipsoid.later_warning_candidate_bank import (
        RESULT_SCHEMA, derive_replay_archive, load_config,
        minimum_actual_intervention,
    )
    from main.multilink_ellipsoid.normal_risk_curve import (
        candidate_definitions, curve_summary, load_config as load_curve_config,
    )

    config = load_config(bank_config_path)
    if _file_sha256(selection_manifest) != config["selection_manifest_file_sha256"]:
        raise ValueError("later-warning selection manifest differs")
    base_risk = repo_root / config["base_risk_config"]
    normal_curve = repo_root / config["normal_curve_config"]
    if _file_sha256(base_risk) != config["base_risk_config_file_sha256"]:
        raise ValueError("later-warning base-risk config differs")
    if _file_sha256(normal_curve) != config["normal_curve_config_file_sha256"]:
        raise ValueError("later-warning normal-curve config differs")
    curve_config = load_curve_config(normal_curve)

    source_path = Path(config["source_result"])
    if _file_sha256(source_path) != config["source_result_file_sha256"]:
        raise ValueError("later-warning source result file differs")
    source = _load(source_path)
    if source["result_payload_sha256"] != config["source_result_payload_sha256"]:
        raise ValueError("later-warning source result payload differs")
    selection = [
        json.loads(line) for line in selection_manifest.read_text().splitlines()
        if line
    ]
    selected = selection[int(config["case_index"])]
    if selected["case_id"] != config["case_id"]:
        raise ValueError("later-warning selection case differs")
    sealed_path = table1_root / selected["archived_result_relative_path"]
    if _file_sha256(sealed_path) != selected["archived_result_file_sha256"]:
        raise ValueError("later-warning sealed result file differs")
    sealed = _load(sealed_path)
    if sealed["result_payload_sha256"] != selected["archived_result_payload_sha256"]:
        raise ValueError("later-warning sealed result payload differs")

    derived = derive_replay_archive(source, sealed, config)
    derived_path = output_path.with_name("derived-replay-ledger.json")
    _atomic_write(derived_path, derived)
    result = evaluate(
        repo_root=repo_root,
        population_manifest_path=population_manifest,
        archived_path=derived_path,
        geometry_config_path=geometry_config,
        experiment_config_path=base_risk,
        expected_commit=expected_commit,
        output_path=output_path,
        case_id_override=config["case_id"],
        state_step_override=int(config["state_step"]),
        query_index_override=int(config["query_index"]),
        result_schema_override=RESULT_SCHEMA,
        claim_scope_override=config["claim_scope"],
        population_binding={
            "source_result": str(source_path),
            "source_result_file_sha256": config["source_result_file_sha256"],
            "source_result_payload_sha256": config["source_result_payload_sha256"],
            "derived_replay_ledger": str(derived_path),
            "derived_replay_ledger_file_sha256": _file_sha256(derived_path),
            "derived_replay_ledger_payload_sha256": derived["result_payload_sha256"],
            "selection": selected,
        },
        candidate_definitions_override=lambda nominal, frame, _base: candidate_definitions(
            nominal, frame, curve_config
        ),
        candidate_protocol_binding=config,
        apply_released_aegis_ee_to_all_proposed_actions=True,
        prime_slabbed_geometry_at_initial_state=True,
    )

    nominal = np.asarray(result["nominal_five_action_chunk"], dtype=np.float64)
    for candidate in result["candidates"]:
        actions = np.asarray(candidate["actions"], dtype=np.float64)
        candidate["effective_post_AEGIS_correction_l2_action"] = float(
            np.linalg.norm(actions - nominal)
        )
        candidate["active_combined_row"] = int(np.argmax(candidate["combined_risk"]))
        row = int(candidate["active_combined_row"])
        prefix = float(candidate["candidate_prefix_risk"][row])
        backup = candidate["backup_risk"]
        backup_value = None if backup is None else float(backup[row])
        candidate["active_phase"] = (
            "prefix" if backup_value is None or prefix >= backup_value else "backup"
        )

    source_intervention = [
        row for row in source["interventions"]
        if int(row["step"]) == int(config["state_step"])
    ]
    if len(source_intervention) != 1:
        raise ValueError("later-warning source intervention count differs")
    source_intervention = source_intervention[0]
    clearance_error = float(np.max(np.abs(
        np.asarray(result["state"]["initial_clearance_m"], dtype=np.float64)
        - np.asarray(source_intervention["current_row_clearance_m"], dtype=np.float64)
    )))
    source_query = [
        row for row in source["policy_queries"]
        if int(row["query_index"]) == int(config["query_index"])
    ][0]
    source_post = np.asarray(
        [row["context"]["executed_action"]
         for row in source_query["nominal_qp_records"]],
        dtype=np.float64,
    )
    nominal_projection_error = float(np.max(np.abs(nominal - source_post)))
    tolerance = float(config["source_state_replay_tolerance_m"])
    curve = curve_summary(result["candidates"], tolerance_m=1.0e-9)
    selected_candidate = minimum_actual_intervention(result["candidates"])
    result["later_warning_config"] = config
    result["source_reproduction"] = {
        "source_current_clearance_m": source_intervention["current_row_clearance_m"],
        "replayed_current_clearance_m": result["state"]["initial_clearance_m"],
        "maximum_clearance_error_m": clearance_error,
        "nominal_post_AEGIS_maximum_action_error": nominal_projection_error,
        "source_corrected_prediction_safe": bool(
            source_intervention["corrected_prediction_safe"]
        ),
        "source_corrected_prediction_minimum_clearance_m": float(
            source_intervention["corrected_prediction"]["minimum_clearance_m"]
        ),
    }
    result["curve_summary"] = curve
    result["selected_minimum_actual_intervention_candidate"] = selected_candidate
    result["later_warning_gates"] = {
        "source_state_replay": clearance_error <= tolerance,
        "nominal_projection_replay": nominal_projection_error <= tolerance,
        "candidate_count": len(result["candidates"]) == int(
            config["gate"]["require_candidate_count"]
        ),
        "nominal_unsafe": not bool(result["candidates"][0]["exact_safe"]),
        "known_exact_safe_support": selected_candidate is not None,
        "zero_proxy_safe_physical_collision": (
            int(result["summary"]["proxy_safe_physical_collision_count"]) == 0
        ),
        "minimum_actual_intervention_selection": (
            selected_candidate is None
            or selected_candidate == minimum_actual_intervention(result["candidates"])
        ),
    }
    result["strict_gate_pass"] = bool(all(result["later_warning_gates"].values()))
    result["interpretation"] = (
        "later_warning_normal_candidate_bank_recoverable"
        if result["strict_gate_pass"]
        else "later_warning_normal_candidate_bank_no_go"
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
    parser.add_argument("--bank-config", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = collect(
        repo_root=args.repo_root.resolve(),
        population_manifest=args.population_manifest.resolve(),
        selection_manifest=args.selection_manifest.resolve(),
        table1_root=args.table1_root.resolve(),
        geometry_config=args.geometry_config.resolve(),
        bank_config_path=args.bank_config.resolve(),
        expected_commit=args.expected_commit,
        output_path=args.output.resolve(),
    )
    _atomic_write(args.output.resolve(), result)
    print(json.dumps({
        "case_id": result["population_binding"]["selection"]["case_id"],
        "strict_gate_pass": result["strict_gate_pass"],
        "interpretation": result["interpretation"],
        "curve_summary": result["curve_summary"],
        "selected": result["selected_minimum_actual_intervention_candidate"],
        "result_payload_sha256": result["result_payload_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
