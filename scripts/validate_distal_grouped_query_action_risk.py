#!/usr/bin/env python3
"""Validate and aggregate grouped query action-risk artifacts."""

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


def validate(
    *, repo_root: Path, producer_root: Path, coverage_root: Path,
    grouped_config_path: Path, base_config_path: Path,
    expected_producer_commit: str, expected_validation_commit: str,
) -> dict[str, Any]:
    import numpy as np

    from main.multilink_ellipsoid.grouped_query_action_risk import (
        RESULT_SCHEMA, VALIDATION_SCHEMA, load_config,
    )
    from main.multilink_ellipsoid.pure_backup import orthonormal_local_frame
    from main.multilink_ellipsoid.query_action_risk import (
        candidate_definitions, combine_row_minima, exact_safe,
        load_config as load_base_config, risk_from_row_minimum,
    )

    config = load_config(grouped_config_path)
    base = load_base_config(base_config_path)
    _require(base["config_payload_sha256"] == config["method"]["base_config_payload_sha256"],
             "grouped query-risk base config differs")
    rows = [{
        "row": row, "safe_candidate_count": 0, "unsafe_candidate_count": 0,
        "unknown_candidate_count": 0, "near_boundary_candidate_count": 0,
        "active_witness_count": 0, "state_with_safe_count": 0,
        "state_with_unsafe_count": 0, "state_with_active_witness_count": 0,
        "episode_with_safe_count": 0, "episode_with_unsafe_count": 0,
        "episode_with_active_witness_count": 0,
    } for row in range(7)]
    records = []
    unsupported_states = []
    proxy_safe_physical = 0
    near = float(config["population_gate"]["near_boundary_absolute_risk_m"])
    for case_index in range(int(config["coverage_source"]["retained_state_count"])):
        result_path = producer_root / ("case-%02d" % case_index) / "result.json"
        coverage_path = coverage_root / ("case-%02d" % case_index) / "result.json"
        result = _load(result_path)
        coverage = _load(coverage_path)
        _require(result.get("schema_version") == RESULT_SCHEMA,
                 "grouped query-risk result schema differs")
        _require(result.get("status") == "complete", "grouped query-risk result incomplete")
        _require(result.get("scientific_result") is True,
                 "grouped query-risk result is not scientific")
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
                 and binding["case"] == coverage["case"]
                 and binding["retained_state"] == coverage["retained_states"][0],
                 "grouped query-risk retained state differs")
        retained = binding["retained_state"]
        _require(result["state"]["step"] == retained["step"]
                 and result["policy_query"]["query_index"] == retained["query_index"]
                 and result["policy_query"]["rng_seed"] == retained["query_rng_seed"],
                 "grouped query-risk query identity differs")
        nominal = np.asarray(result["nominal_five_action_chunk"], dtype=np.float64)
        expected = candidate_definitions(
            nominal,
            orthonormal_local_frame(result["state"]["local_frame"]["normal"]),
            base,
        )
        candidates = result["candidates"]
        _require(len(candidates) == len(expected) == 37,
                 "grouped query-risk candidate count differs")
        row_flags = [{"safe": False, "unsafe": False, "active": False}
                     for _ in range(7)]
        safe_count = 0
        unsafe_count = 0
        timeout_count = 0
        state_proxy_physical = 0
        for observed, definition in zip(candidates, expected):
            for key in (
                "name", "order", "direction", "sign", "temporal_profile",
                "requested_correction_l2_action", "applied_correction_l2_action",
                "clipped", "actions",
            ):
                _require(observed[key] == definition[key],
                         "grouped query-risk candidate identity differs")
            prefix_trace = np.asarray(
                observed["prefix"]["clearance_trace_m"], dtype=np.float64
            )
            _require(prefix_trace.shape == (126, 7),
                     "grouped query-risk prefix trace differs")
            prefix_row = np.min(prefix_trace[1:], axis=0).tolist()
            _require(prefix_row == observed["prefix"]["row_minimum_clearance_m"],
                     "grouped query-risk prefix rows differ")
            parts = [prefix_row]
            backup_row = observed["backup"]["row_minimum_clearance_m"]
            if backup_row is not None:
                parts.append(backup_row)
            combined = combine_row_minima(*parts)
            risk = risk_from_row_minimum(
                combined, float(base["risk_target"]["safety_buffer_m"])
            )
            _require(combined == observed["combined_row_minimum_clearance_m"]
                     and risk == observed["combined_risk"],
                     "grouped query-risk composed target differs")
            _require(observed["exact_safe"] == exact_safe(observed),
                     "grouped query-risk exact-safe label differs")
            status = observed["terminal_status"]
            known_safe_terminal = status == "SAFE_TERMINAL"
            is_timeout = status == "UNKNOWN_TIMEOUT"
            timeout_count += int(is_timeout)
            safe_count += int(observed["exact_safe"])
            candidate_unsafe = bool(max(risk) > 0.0 or observed["physical_veto"])
            unsafe_count += int(candidate_unsafe)
            if max(risk) <= 0.0 and observed["physical_veto"]:
                proxy_safe_physical += 1
                state_proxy_physical += 1
            active = int(np.argmax(np.asarray(risk)))
            rows[active]["active_witness_count"] += 1
            row_flags[active]["active"] = True
            for row, value in enumerate(risk):
                if known_safe_terminal and value <= 0.0 and not observed["physical_veto"]:
                    rows[row]["safe_candidate_count"] += 1
                    row_flags[row]["safe"] = True
                elif value > 0.0:
                    rows[row]["unsafe_candidate_count"] += 1
                    row_flags[row]["unsafe"] = True
                elif is_timeout:
                    rows[row]["unknown_candidate_count"] += 1
                if abs(float(value)) <= near:
                    rows[row]["near_boundary_candidate_count"] += 1
        mixed = safe_count > 0 and unsafe_count > 0
        if not mixed:
            unsupported_states.append({
                "case_id": coverage["case"]["case_id"],
                "split": coverage["case"]["split"],
                "state_id": retained["state_id"],
                "safe_candidate_count": safe_count,
                "unsafe_candidate_count": unsafe_count,
                "timeout_count": timeout_count,
            })
        for row, flags in enumerate(row_flags):
            for name in ("safe", "unsafe", "active"):
                rows[row]["state_with_%s_count" % name] += int(flags[name])
                rows[row]["episode_with_%s_count" % name] += int(flags[name])
        _require(result["summary"]["safe_candidate_count"] == safe_count
                 and result["summary"]["unknown_timeout_count"] == timeout_count
                 and result["summary"]["proxy_safe_physical_collision_count"]
                 == state_proxy_physical,
                 "grouped query-risk summary differs")
        records.append({
            "case_id": coverage["case"]["case_id"],
            "split": coverage["case"]["split"],
            "state_id": retained["state_id"],
            "source_result_file_sha256": _file_sha256(result_path),
            "source_result_payload_sha256": result["result_payload_sha256"],
            "safe_candidate_count": safe_count,
            "unsafe_candidate_count": unsafe_count,
            "unknown_timeout_count": timeout_count,
            "safe_and_unsafe_support": mixed,
        })
    supported_rows = [
        item["row"] for item in rows
        if item["safe_candidate_count"] > 0
        and item["unsafe_candidate_count"] > 0
        and item["active_witness_count"] > 0
    ]
    output = {
        "schema_version": VALIDATION_SCHEMA,
        "status": "validated",
        "source": _git_identity(repo_root, expected_validation_commit),
        "allocation": _allocation_record(),
        "producer": {"root": str(producer_root), "commit": expected_producer_commit},
        "state_count": len(records),
        "candidate_count": len(records) * 37,
        "records": records,
        "unsupported_states": unsupported_states,
        "row_coverage": rows,
        "supported_rows": supported_rows,
        "unsupported_rows": [row for row in range(7) if row not in supported_rows],
        "proxy_safe_physical_collision_count": proxy_safe_physical,
        "gates": {
            "all_15_states_complete": len(records) == 15,
            "safe_and_unsafe_support_every_state": not unsupported_states,
            "no_proxy_safe_physical_collision": proxy_safe_physical == 0,
            "all_seven_rows_supported": len(supported_rows) == 7,
        },
        "interpretation": (
            "grouped_query_action_risk_support_pass_training_may_be_preregistered"
            if not unsupported_states and proxy_safe_physical == 0
            and len(supported_rows) == 7
            else "grouped_query_action_risk_support_no_go_do_not_train"
        ),
        "MLP_training_authorized": False,
        "calibration_authorized": False,
        "QP_authorized": False,
        "closed_loop_authorized": False,
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
    parser.add_argument("--expected-producer-commit", required=True)
    parser.add_argument("--expected-validation-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = validate(
        repo_root=args.repo_root.resolve(), producer_root=args.producer_root.resolve(),
        coverage_root=args.coverage_root.resolve(),
        grouped_config_path=args.grouped_config.resolve(),
        base_config_path=args.base_config.resolve(),
        expected_producer_commit=args.expected_producer_commit,
        expected_validation_commit=args.expected_validation_commit,
    )
    _atomic_write(args.output.resolve(), result)
    print(json.dumps({
        "interpretation": result["interpretation"],
        "gates": result["gates"], "supported_rows": result["supported_rows"],
        "unsupported_rows": result["unsupported_rows"],
        "unsupported_states": result["unsupported_states"],
        "result_payload_sha256": result["result_payload_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
