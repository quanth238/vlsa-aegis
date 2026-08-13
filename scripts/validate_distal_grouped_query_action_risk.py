#!/usr/bin/env python3
"""Validate, classify, aggregate, and freeze grouped query-risk artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.evaluate_distal_query_action_risk_e05 import _allocation_record
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _git_identity, _load, _require, _sha256,
)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _payload_sha256(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("result_payload_sha256", None)
    return _sha256(_canonical(payload))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = b"".join(_canonical(record) + b"\n" for record in records)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def _split_template(cases: Sequence[Mapping[str, Any]], freeze: Mapping[str, Any]) -> dict[str, Any]:
    output: dict[str, list[str]] = {name: [] for name in ("train", "validation", "test", "diagnostic")}
    for case in cases:
        output[str(case["split"])].append(str(case["case_id"]))
    _require(output["diagnostic"] == freeze["split"]["diagnostic_cases"],
             "grouped query-risk diagnostic split differs")
    _require(output["test"] == freeze["split"]["reserved_test_cases"],
             "grouped query-risk reserved test split differs")
    return output


def validate(
    *, repo_root: Path, producer_root: Path, coverage_root: Path,
    grouped_config_path: Path, base_config_path: Path, freeze_config_path: Path,
    selection_manifest_path: Path, table1_root: Path,
    expected_producer_commit: str, expected_validation_commit: str,
    freeze_root: Path,
) -> dict[str, Any]:
    import numpy as np

    from main.multilink_ellipsoid.grouped_query_action_risk import RESULT_SCHEMA, load_config
    from main.multilink_ellipsoid.grouped_query_action_risk_freeze import (
        VALIDATION_SCHEMA, candidate_outcome, load_config as load_freeze_config,
        state_classification,
    )
    from main.multilink_ellipsoid.pure_backup import orthonormal_local_frame
    from main.multilink_ellipsoid.query_boundary_coverage import ROW_IDENTITIES
    from main.multilink_ellipsoid.query_action_risk import (
        candidate_definitions, combine_row_minima, exact_safe,
        load_config as load_base_config, risk_from_row_minimum,
    )

    config = load_config(grouped_config_path)
    freeze = load_freeze_config(freeze_config_path)
    base = load_base_config(base_config_path)
    _require(_file_sha256(grouped_config_path)
             == freeze["source"]["grouped_config_file_sha256"],
             "grouped query-risk freeze grouped-config file differs")
    _require(config["config_payload_sha256"]
             == freeze["source"]["grouped_config_payload_sha256"],
             "grouped query-risk freeze grouped-config payload differs")
    _require(_file_sha256(selection_manifest_path)
             == freeze["source"]["episode_manifest_file_sha256"],
             "grouped query-risk episode manifest differs")
    cases = _read_jsonl(selection_manifest_path)
    split_template = _split_template(cases, freeze)
    selected_cases = [case for case in cases if case["split"] in config["coverage_source"]["included_splits"]]
    _require(len(selected_cases) == int(config["coverage_source"]["retained_state_count"]),
             "grouped query-risk selected case count differs")

    near = float(config["population_gate"]["near_boundary_absolute_risk_m"])
    row_records = [{
        **identity,
        "known_safe_candidate_count": 0,
        "known_unsafe_candidate_count": 0,
        "unknown_timeout_candidate_count": 0,
        "near_boundary_known_candidate_count": 0,
        "active_witness_known_candidate_count": 0,
        "train_episode_with_unsafe_count": 0,
        "validation_episode_with_unsafe_count": 0,
        "train_episode_with_active_witness_count": 0,
        "validation_episode_with_active_witness_count": 0,
        "all_split_episode_with_safe_count": 0,
        "all_split_episode_with_unsafe_count": 0,
        "all_split_episode_with_active_witness_count": 0,
    } for identity in ROW_IDENTITIES]
    state_records: list[dict[str, Any]] = []
    candidate_manifest: list[dict[str, Any]] = []
    artifact_records: list[dict[str, Any]] = []
    replay_failures: list[dict[str, Any]] = []

    for case_index, expected_case in enumerate(selected_cases):
        result_path = producer_root / ("case-%02d" % case_index) / "result.json"
        coverage_path = coverage_root / ("case-%02d" % case_index) / "result.json"
        result = _load(result_path)
        coverage = _load(coverage_path)
        _require(result.get("schema_version") == RESULT_SCHEMA,
                 "grouped query-risk result schema differs")
        _require(result.get("status") == "complete"
                 and result.get("scientific_result") is True,
                 "grouped query-risk result incomplete")
        _require(result.get("execution_mode") == "full_diagnostic",
                 "grouped query-risk execution mode differs")
        _require(result.get("source", {}).get("commit") == expected_producer_commit
                 and result.get("source", {}).get("dirty") is False,
                 "grouped query-risk producer source differs")
        _require(result.get("result_payload_sha256") == _payload_sha256(result),
                 "grouped query-risk result payload differs")
        _require(result.get("base_method_config") == base,
                 "grouped query-risk base method changed")
        binding = result["population_binding"]
        _require(binding["grouped_config"] == config,
                 "grouped query-risk population config differs")
        _require(binding["coverage_result_file_sha256"] == _file_sha256(coverage_path)
                 and binding["coverage_result_payload_sha256"]
                 == coverage["result_payload_sha256"],
                 "grouped query-risk coverage binding differs")
        _require(binding["case_index"] == case_index
                 and binding["case"] == coverage["case"] == expected_case
                 and binding["retained_state"] == coverage["retained_states"][0],
                 "grouped query-risk retained state differs")
        retained = binding["retained_state"]
        _require(result["state"]["step"] == retained["step"]
                 and result["policy_query"]["query_index"] == retained["query_index"]
                 and result["policy_query"]["rng_seed"] == retained["query_rng_seed"],
                 "grouped query-risk query identity differs")
        archived_path = table1_root / expected_case["archived_result_relative_path"]
        archived = _load(archived_path)
        _require(_file_sha256(archived_path) == result["archived_table1"]["file_sha256"]
                 and archived["result_payload_sha256"]
                 == result["archived_table1"]["result_payload_sha256"]
                 and result["archived_table1"]["read_only"] is True,
                 "grouped query-risk immutable Table-1 binding differs")
        _require(result["policy_query"] == archived["policy_queries"][int(retained["query_index"])],
                 "grouped query-risk policy query differs")
        nominal = np.asarray(result["nominal_five_action_chunk"], dtype=np.float64)
        archived_nominal = np.asarray([
            archived["actions"][step]["executed"]
            for step in range(int(retained["step"]), int(retained["step"]) + 5)
        ], dtype=np.float64)
        _require(np.array_equal(nominal, archived_nominal),
                 "grouped query-risk nominal chunk differs")
        expected = candidate_definitions(
            nominal,
            orthonormal_local_frame(result["state"]["local_frame"]["normal"]),
            base,
        )
        candidates = result["candidates"]
        _require(len(candidates) == len(expected) == 37,
                 "grouped query-risk candidate count differs")
        replay = result["determinism_replay"]
        replay_ok = bool(
            replay["next_state_maximum_absolute_error"] == 0.0
            and replay["clearance_trace_maximum_absolute_error_m"] == 0.0
            and replay["contacts_identical"] is True
            and replay["CAR_maximum_absolute_error_m"] == 0.0
            and len(set(replay["next_state_sha256"])) == 1
            and result["gates"]["exact_snapshot_replay"] is True
            and result["gates"]["all_replays_boundary_exact"] is True
        )
        if not replay_ok:
            replay_failures.append({"case_id": expected_case["case_id"], "state_id": retained["state_id"]})
        _require(replay_ok, "grouped query-risk deterministic replay differs")

        row_flags = [{"safe": False, "unsafe": False, "active": False} for _ in range(7)]
        state_candidate_manifest = []
        for observed, definition in zip(candidates, expected):
            for key in (
                "name", "order", "direction", "sign", "temporal_profile",
                "requested_correction_l2_action", "applied_correction_l2_action",
                "clipped", "actions",
            ):
                _require(observed[key] == definition[key],
                         "grouped query-risk candidate identity differs")
            prefix_trace = np.asarray(observed["prefix"]["clearance_trace_m"], dtype=np.float64)
            _require(prefix_trace.shape == (126, 7),
                     "grouped query-risk prefix trace differs")
            prefix_row = np.min(prefix_trace[1:], axis=0).tolist()
            _require(prefix_row == observed["prefix"]["row_minimum_clearance_m"],
                     "grouped query-risk prefix rows differ")
            parts = [prefix_row]
            if observed["backup"]["row_minimum_clearance_m"] is not None:
                parts.append(observed["backup"]["row_minimum_clearance_m"])
            combined = combine_row_minima(*parts)
            risk = risk_from_row_minimum(
                combined, float(base["risk_target"]["safety_buffer_m"])
            )
            _require(combined == observed["combined_row_minimum_clearance_m"]
                     and risk == observed["combined_risk"]
                     and observed["exact_safe"] == exact_safe(observed),
                     "grouped query-risk composed target differs")
            outcome = candidate_outcome(observed)
            _require(observed["source_restore_maximum_error"] == 0.0,
                     "grouped query-risk candidate source restore differs")
            active = int(np.argmax(np.asarray(risk)))
            if outcome != "unknown":
                row_records[active]["active_witness_known_candidate_count"] += 1
                row_flags[active]["active"] = True
            for row, value in enumerate(risk):
                if outcome == "safe":
                    row_records[row]["known_safe_candidate_count"] += 1
                    row_flags[row]["safe"] = True
                elif outcome == "unsafe" and float(value) > 0.0:
                    row_records[row]["known_unsafe_candidate_count"] += 1
                    row_flags[row]["unsafe"] = True
                elif outcome == "unknown":
                    row_records[row]["unknown_timeout_candidate_count"] += 1
                if outcome != "unknown" and abs(float(value)) <= near:
                    row_records[row]["near_boundary_known_candidate_count"] += 1
            if outcome != "unknown":
                state_candidate_manifest.append({
                    "case_id": expected_case["case_id"],
                    "split": expected_case["split"],
                    "state_id": retained["state_id"],
                    "candidate_name": observed["name"],
                    "candidate_order": observed["order"],
                    "actions": observed["actions"],
                    "risk": risk,
                    "outcome": outcome,
                    "exact_safe": observed["exact_safe"],
                    "physical_veto": observed["physical_veto"],
                    "active_witness_row": active,
                    "source_result_payload_sha256": result["result_payload_sha256"],
                })
        classification = state_classification(candidates)
        if classification["primary_category"] == "usable_mixed_support":
            candidate_manifest.extend(state_candidate_manifest)
        for row, flags in enumerate(row_flags):
            if flags["safe"]:
                row_records[row]["all_split_episode_with_safe_count"] += 1
            if flags["unsafe"]:
                row_records[row]["all_split_episode_with_unsafe_count"] += 1
            if flags["active"]:
                row_records[row]["all_split_episode_with_active_witness_count"] += 1
            split = str(expected_case["split"])
            if split in {"train", "validation"}:
                if flags["unsafe"]:
                    row_records[row]["%s_episode_with_unsafe_count" % split] += 1
                if flags["active"]:
                    row_records[row]["%s_episode_with_active_witness_count" % split] += 1
        _require(result["summary"]["unknown_timeout_count"]
                 == classification["unknown_timeout_candidate_count"],
                 "grouped query-risk timeout summary differs")
        _require(result["summary"]["safe_candidate_count"]
                 == classification["known_safe_candidate_count"],
                 "grouped query-risk safe summary differs")
        state_record = {
            "case_id": expected_case["case_id"],
            "split": expected_case["split"],
            "state_id": retained["state_id"],
            "source_result_file_sha256": _file_sha256(result_path),
            "source_result_payload_sha256": result["result_payload_sha256"],
            "deterministic_replay_pass": replay_ok,
            **classification,
        }
        state_records.append(state_record)
        artifact_records.append({
            "case_id": expected_case["case_id"],
            "split": expected_case["split"],
            "path": str(result_path),
            "file_sha256": _file_sha256(result_path),
            "result_payload_sha256": result["result_payload_sha256"],
            "archived_table1_file_sha256": result["archived_table1"]["file_sha256"],
        })

    thresholds = freeze["adequacy"]
    per_row = thresholds["per_row"]
    row_failures: dict[str, list[str]] = {}
    for row in row_records:
        failures = []
        for metric, threshold_name in (
            ("known_safe_candidate_count", "minimum_known_safe_candidates"),
            ("known_unsafe_candidate_count", "minimum_known_unsafe_candidates"),
            ("near_boundary_known_candidate_count", "minimum_near_boundary_known_candidates"),
            ("active_witness_known_candidate_count", "minimum_active_witness_candidates"),
            ("train_episode_with_unsafe_count", "minimum_train_episodes_with_unsafe"),
            ("validation_episode_with_unsafe_count", "minimum_validation_episodes_with_unsafe"),
            ("train_episode_with_active_witness_count", "minimum_train_episodes_with_active_witness"),
            ("validation_episode_with_active_witness_count", "minimum_validation_episodes_with_active_witness"),
        ):
            if int(row[metric]) < int(per_row[threshold_name]):
                failures.append(metric)
        row["adequate"] = not failures
        row["failed_metrics"] = failures
        if failures:
            row_failures[str(row["row"])] = failures

    usable = [item for item in state_records if item["primary_category"] == "usable_mixed_support"]
    usable_train = [item for item in usable if item["split"] == "train"]
    usable_validation = [item for item in usable if item["split"] == "validation"]
    gates = {
        "all_15_artifacts_complete_and_hash_valid": len(artifact_records) == 15,
        "all_replays_deterministic_and_bound": not replay_failures,
        "timeout_candidates_excluded_from_candidate_manifest": (
            len(candidate_manifest)
            == sum(item["known_safe_candidate_count"] + item["known_unsafe_candidate_count"]
                   for item in usable)
        ),
        "minimum_usable_train_states": len(usable_train)
        >= int(thresholds["minimum_usable_train_states"]),
        "minimum_usable_validation_states": len(usable_validation)
        >= int(thresholds["minimum_usable_validation_states"]),
        "all_claimed_rows_adequate": not row_failures,
        "no_proxy_invalid_states": all(item["primary_category"] != "proxy_invalid"
                                    for item in state_records),
    }
    passed = all(gates.values())
    targeted_rows = []
    for row in row_records:
        if row["adequate"]:
            continue
        shortfall = {}
        for metric, threshold_name in (
            ("known_safe_candidate_count", "minimum_known_safe_candidates"),
            ("known_unsafe_candidate_count", "minimum_known_unsafe_candidates"),
            ("near_boundary_known_candidate_count", "minimum_near_boundary_known_candidates"),
            ("active_witness_known_candidate_count", "minimum_active_witness_candidates"),
            ("train_episode_with_unsafe_count", "minimum_train_episodes_with_unsafe"),
            ("validation_episode_with_unsafe_count", "minimum_validation_episodes_with_unsafe"),
            ("train_episode_with_active_witness_count", "minimum_train_episodes_with_active_witness"),
            ("validation_episode_with_active_witness_count", "minimum_validation_episodes_with_active_witness"),
        ):
            shortfall[metric] = max(
                0, int(per_row[threshold_name]) - int(row[metric])
            )
        targeted_rows.append({
            "row": row["row"], "link": row["link"], "slab": row["slab"],
            "failed_metrics": row["failed_metrics"], "minimum_shortfall": shortfall,
        })
    split_manifest = {
        "schema_version": "vlsa_distal_grouped_query_action_risk_split.v1",
        "unit": "complete_episode",
        "source_episode_manifest_file_sha256": _file_sha256(selection_manifest_path),
        "source_artifacts": artifact_records,
        "episode_groups": split_template,
        "usable_fit_cases": {
            split: [item["case_id"] for item in usable if item["split"] == split]
            for split in ("train", "validation")
        },
        "diagnostic_excluded_from_fit_and_final_claim": True,
        "reserved_test_labels_absent": True,
        "coverage_adequate_for_training": passed,
        "training_authorized": passed,
    }
    output = {
        "schema_version": VALIDATION_SCHEMA,
        "status": "validated",
        "source": _git_identity(repo_root, expected_validation_commit),
        "allocation": _allocation_record(),
        "producer": {"root": str(producer_root), "commit": expected_producer_commit},
        "configs": {"grouped": config, "freeze": freeze},
        "artifact_records": artifact_records,
        "state_classifications": state_records,
        "classification_counts": {
            name: sum(item["primary_category"] == name for item in state_records)
            for name in freeze["classification"]["primary_state_precedence"]
        },
        "states_with_unknown_timeouts": [
            item for item in state_records if item["contains_unknown_timeouts"]
        ],
        "row_coverage": row_records,
        "row_adequacy_failures": row_failures,
        "targeted_collection_plan": {
            "required": not passed,
            "target_rows": targeted_rows,
            "retain_same_candidate_and_backup_protocol": True,
            "collect_only_initially_safe_real_query_boundaries": True,
            "require_known_safe_and_known_unsafe_support": True,
            "reject_proxy_invalid_states_from_learning": True,
            "preserve_complete_episode_groups": True,
            "unknown_timeouts_remain_censored": True,
        },
        "split_manifest": split_manifest,
        "gates": gates,
        "interpretation": (
            "grouped_query_action_risk_dataset_frozen_training_preregistration_may_begin"
            if passed else "grouped_query_action_risk_coverage_no_go_targeted_collection_required"
        ),
        "MLP_training_authorized": passed,
        "calibration_authorized": False,
        "QP_authorized": False,
        "closed_loop_authorized": False,
    }
    freeze_root.mkdir(parents=True, exist_ok=False)
    _write_jsonl(freeze_root / freeze["freeze"]["candidate_manifest"], candidate_manifest)
    _write_jsonl(freeze_root / freeze["freeze"]["state_manifest"], state_records)
    _atomic_write(freeze_root / freeze["freeze"]["split_manifest"], split_manifest)
    output["frozen_dataset"] = {
        "root": str(freeze_root),
        "status": "training_ready" if passed else "coverage_no_go_not_for_training",
        "usable_known_candidate_count": len(candidate_manifest),
        "candidate_manifest_file_sha256": _file_sha256(
            freeze_root / freeze["freeze"]["candidate_manifest"]
        ),
        "state_manifest_file_sha256": _file_sha256(
            freeze_root / freeze["freeze"]["state_manifest"]
        ),
        "split_manifest_file_sha256": _file_sha256(
            freeze_root / freeze["freeze"]["split_manifest"]
        ),
    }
    output["result_payload_sha256"] = _sha256(_canonical(output))
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--producer-root", type=Path, required=True)
    parser.add_argument("--coverage-root", type=Path, required=True)
    parser.add_argument("--grouped-config", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--freeze-config", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--table1-root", type=Path, required=True)
    parser.add_argument("--expected-producer-commit", required=True)
    parser.add_argument("--expected-validation-commit", required=True)
    parser.add_argument("--freeze-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = validate(
        repo_root=args.repo_root.resolve(), producer_root=args.producer_root.resolve(),
        coverage_root=args.coverage_root.resolve(),
        grouped_config_path=args.grouped_config.resolve(),
        base_config_path=args.base_config.resolve(),
        freeze_config_path=args.freeze_config.resolve(),
        selection_manifest_path=args.selection_manifest.resolve(),
        table1_root=args.table1_root.resolve(),
        expected_producer_commit=args.expected_producer_commit,
        expected_validation_commit=args.expected_validation_commit,
        freeze_root=args.freeze_root.resolve(),
    )
    _atomic_write(args.output.resolve(), result)
    print(json.dumps({
        "interpretation": result["interpretation"],
        "gates": result["gates"],
        "classification_counts": result["classification_counts"],
        "row_adequacy_failures": result["row_adequacy_failures"],
        "MLP_training_authorized": result["MLP_training_authorized"],
        "result_payload_sha256": result["result_payload_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
