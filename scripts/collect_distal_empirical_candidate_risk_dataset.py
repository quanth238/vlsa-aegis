#!/usr/bin/env python3
"""Collect one grouped normal bank and relabel it with empirical geometry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional, Sequence

from scripts.audit_distal_compiled_box_risk_target import _evaluate_case
from scripts.evaluate_distal_query_action_risk_e05 import evaluate
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _git_identity, _sha256,
)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def collect(
    *, repo_root: Path, table1_root: Path, config_path: Path,
    case_index: int, expected_commit: str, run_root: Path,
) -> dict[str, Any]:
    import numpy as np

    from main.multilink_ellipsoid.empirical_candidate_risk_dataset import (
        CASE_RESULT_SCHEMA, load_cases, load_config, warning_step,
    )
    from main.multilink_ellipsoid.l6_proxy_scale_audit import (
        load_empirical_proxy_config,
    )
    from main.multilink_ellipsoid.normal_risk_curve import (
        RESULT_SCHEMA as CURVE_RESULT_SCHEMA,
        candidate_definitions,
        curve_summary,
    )

    config = load_config(config_path)
    population_path = repo_root / config["population_manifest"]
    selection_path = repo_root / config["selection_manifest"]
    geometry_path = repo_root / config["geometry_config"]
    base_risk_path = repo_root / config["base_risk_config"]
    proxy_path = repo_root / config["empirical_l6_proxy_config"]
    for path, expected, label in (
        (population_path, config["population_manifest_file_sha256"], "population"),
        (geometry_path, config["geometry_config_file_sha256"], "geometry"),
        (base_risk_path, config["base_risk_config_file_sha256"], "base risk"),
        (proxy_path, config["empirical_l6_proxy_config_file_sha256"], "proxy"),
    ):
        if _file_sha256(path) != expected:
            raise ValueError("empirical candidate-risk %s differs" % label)
    cases = load_cases(selection_path, config)
    if not 0 <= int(case_index) < len(cases):
        raise ValueError("empirical candidate-risk case index differs")
    selected = cases[int(case_index)]
    state_step = warning_step(selected, config)
    archived_path = table1_root / selected["archived_result_relative_path"]
    if _file_sha256(archived_path) != selected["archived_result_file_sha256"]:
        raise ValueError("empirical candidate-risk Table-1 source differs")

    curve_binding = {
        "requested_alpha": config["candidate_bank"]["requested_alpha"],
        "direction": config["candidate_bank"]["direction"],
        "temporal_profile": config["candidate_bank"]["temporal_profile"],
        "action_limit": config["candidate_bank"]["action_limit"],
        "risk_definition": config["risk_target"]["primary"],
    }
    source_path = run_root / "source-curve.json"
    raw = evaluate(
        repo_root=repo_root,
        population_manifest_path=population_path,
        archived_path=archived_path,
        geometry_config_path=geometry_path,
        experiment_config_path=base_risk_path,
        expected_commit=expected_commit,
        output_path=source_path,
        case_id_override=selected["case_id"],
        state_step_override=state_step,
        query_index_override=state_step // 5,
        result_schema_override=CURVE_RESULT_SCHEMA,
        claim_scope_override=config["claim_scope"],
        population_binding={
            "case_index": int(case_index),
            "selection": selected,
            "selection_manifest": str(selection_path),
            "selection_manifest_sha256": _file_sha256(selection_path),
            "split": selected["split"],
            "test_cases_excluded": True,
        },
        candidate_definitions_override=lambda nominal, frame, _base: candidate_definitions(
            nominal, frame, config["candidate_bank"]
        ),
        candidate_protocol_binding=curve_binding,
        apply_released_aegis_ee_to_all_proposed_actions=True,
        capture_physical_context=True,
    )
    nominal = np.asarray(raw["nominal_five_action_chunk"], dtype=np.float64)
    for candidate in raw["candidates"]:
        actions = np.asarray(candidate["actions"], dtype=np.float64)
        candidate["effective_post_AEGIS_correction_l2_action"] = float(
            np.linalg.norm(actions[:, :3] - nominal[:, :3])
        )
        candidate["active_combined_row"] = int(np.argmax(candidate["combined_risk"]))
        row = int(candidate["active_combined_row"])
        prefix = float(candidate["candidate_prefix_risk"][row])
        backup = candidate["backup_risk"]
        backup_value = None if backup is None else float(backup[row])
        candidate["active_phase"] = (
            "prefix" if backup_value is None or prefix >= backup_value else "backup"
        )
    raw["curve_summary"] = curve_summary(raw["candidates"], tolerance_m=1.0e-9)
    raw["dataset_config_payload_sha256"] = config["config_payload_sha256"]
    raw.pop("result_payload_sha256", None)
    raw["result_payload_sha256"] = _sha256(_canonical(raw))
    _atomic_write(source_path, raw)

    source_case = {
        "case_id": selected["case_id"],
        "state_step": state_step,
        "source_result": str(source_path),
        "source_result_file_sha256": _file_sha256(source_path),
        "source_result_payload_sha256": raw["result_payload_sha256"],
        "slab_initialization": "query_state_matching_source",
        "candidate_names": [item["name"] for item in raw["candidates"]],
    }
    proxy = load_empirical_proxy_config(proxy_path)
    empirical = _evaluate_case(
        repo_root=repo_root,
        population_manifest=population_path,
        geometry_config_path=geometry_path,
        case_config=source_case,
        audit_config={"gate": config["gate"], "empirical_l6_proxy": proxy},
    )
    value = {
        "schema_version": CASE_RESULT_SCHEMA,
        "status": "complete",
        "scientific_result": True,
        "claim_scope": config["claim_scope"],
        "source": _git_identity(repo_root, expected_commit),
        "allocation": raw["allocation"],
        "config_file_sha256": config["config_file_sha256"],
        "config_payload_sha256": config["config_payload_sha256"],
        "case_index": int(case_index),
        "case_id": selected["case_id"],
        "split": selected["split"],
        "state_step": state_step,
        "query_index": state_step // 5,
        "selection": selected,
        "source_curve": {
            "path": str(source_path),
            "file_sha256": _file_sha256(source_path),
            "result_payload_sha256": raw["result_payload_sha256"],
        },
        "empirical_l6_proxy": proxy,
        "empirical_case": empirical,
        "training_authorized_for_case": False,
    }
    value["result_payload_sha256"] = _sha256(_canonical(value))
    return value


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--table1-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--case-index", type=int, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    value = collect(
        repo_root=args.repo_root.resolve(),
        table1_root=args.table1_root.resolve(),
        config_path=args.config.resolve(),
        case_index=args.case_index,
        expected_commit=args.expected_commit,
        run_root=args.run_root.resolve(),
    )
    _atomic_write(args.output.resolve(), value)
    print(json.dumps({
        "case_id": value["case_id"],
        "split": value["split"],
        "state_step": value["state_step"],
        "source_replay_exact": value["empirical_case"]["source_replay_exact"],
        "result_payload_sha256": value["result_payload_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
