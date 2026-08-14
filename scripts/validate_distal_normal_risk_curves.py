#!/usr/bin/env python3
"""Independently validate and aggregate exact normal risk curves."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional, Sequence

from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _load, _require, _sha256,
)


SUMMARY_SCHEMA = "vlsa_distal_normal_risk_curve_summary.v1"
VALIDATION_SCHEMA = "vlsa_distal_normal_risk_curve_validation.v1"


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def validate(
    *, config_path: Path, selection_path: Path, result_paths: Sequence[Path],
    expected_commit: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from main.multilink_ellipsoid.normal_risk_curve import (
        RESULT_SCHEMA, curve_summary, load_config,
    )

    config = load_config(config_path)
    selected = [json.loads(line) for line in selection_path.read_text().splitlines() if line]
    _require(len(selected) == int(config["gate"]["required_case_count"]),
             "normal risk-curve selection count differs")
    _require(len(result_paths) == len(selected), "normal risk-curve result count differs")
    checks = []
    for case_index, (expected, path) in enumerate(zip(selected, result_paths)):
        result = _load(path)
        _require(result["schema_version"] == RESULT_SCHEMA, "normal risk-curve result schema differs")
        _require(result["source"]["commit"] == expected_commit, "normal risk-curve source differs")
        _require(result["population_binding"]["selection"] == expected,
                 "normal risk-curve selection differs")
        _require(result["curve_config"] == config, "normal risk-curve config differs")
        payload = dict(result)
        claimed = payload.pop("result_payload_sha256")
        _require(_sha256(_canonical(payload)) == claimed, "normal risk-curve self-hash differs")
        _require(int(result["state"]["step"]) == int(config["warning_steps"][case_index]),
                 "normal risk-curve warning step differs")
        _require(result["determinism_replay"]["next_state_maximum_absolute_error"] == 0.0,
                 "normal risk-curve replay differs")
        _require(result["gates"]["no_timeout_labeled_safe"],
                 "normal risk-curve timeout handling differs")
        _require(result["gates"]["proxy_safe_physical_collision_count_zero"],
                 "normal risk-curve proxy false-safe differs")
        recomputed = curve_summary(
            result["candidates"],
            tolerance_m=float(config["monotonicity_tolerance_m"]),
        )
        _require(recomputed == result["curve_summary"],
                 "normal risk-curve summary replay differs")
        requested = [float(row["requested_alpha"]) for row in result["candidates"]]
        _require(requested == [float(item) for item in config["requested_alpha"]],
                 "normal risk-curve alpha order differs")
        checks.append({
            "case_id": expected["case_id"],
            "warning_step": int(result["state"]["step"]),
            "result_file_sha256": _file_sha256(path),
            "result_payload_sha256": claimed,
            "curve_summary": recomputed,
            "active_rows": [int(row["active_combined_row"]) for row in result["candidates"]],
            "active_phases": [str(row["active_phase"]) for row in result["candidates"]],
            "requested_to_effective": [{
                "requested_alpha": float(row["requested_alpha"]),
                "pre_AEGIS_applied_l2": float(row["applied_correction_l2_action"]),
                "post_AEGIS_effective_l2": float(row["effective_post_AEGIS_correction_l2_action"]),
                "clipped": bool(row["clipped"]),
            } for row in result["candidates"]],
        })
    gate = {
        "case_count_pass": len(checks) == int(config["gate"]["required_case_count"]),
        "exact_replay_pass": True,
        "nominal_unsafe_pass": all(
            not bool(_load(path)["candidates"][0]["exact_safe"])
            for path in result_paths
        ),
        "known_safe_alpha_support_pass": all(
            int(row["curve_summary"]["safe_candidate_count"]) > 0 for row in checks
        ),
        "proxy_safe_physical_collision_pass": True,
    }
    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "status": "complete",
        "scientific_result": True,
        "source_commit": expected_commit,
        "claim_scope": config["claim_scope"],
        "config": config,
        "selection_manifest": {
            "path": str(selection_path),
            "file_sha256": _file_sha256(selection_path),
        },
        "cases": checks,
        "gate": gate,
        "mechanism_gate_pass": all(gate.values()),
        "interpretation": (
            "exact_normal_magnitude_curves_support_adaptive_episode_test"
            if all(gate.values())
            else "exact_normal_magnitude_curve_no_go"
        ),
        "limitations": [
            "opened_diagnostic_cases_not_unseen_generalization",
            "first_warning_state_only_not_complete_adaptive_episode",
            "no_learning_no_denoising_guidance_no_QP_change",
            "timeouts_censored_as_unknown",
        ],
    }
    summary["result_payload_sha256"] = _sha256(_canonical(summary))
    validation = {
        "schema_version": VALIDATION_SCHEMA,
        "status": "passed",
        "scientific_result": True,
        "source_commit": expected_commit,
        "summary_payload_sha256": summary["result_payload_sha256"],
        "case_checks": checks,
        "gate": gate,
    }
    validation["validation_payload_sha256"] = _sha256(_canonical(validation))
    return summary, validation


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--result", type=Path, action="append", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    args = parser.parse_args(argv)
    summary, validation = validate(
        config_path=args.config.resolve(),
        selection_path=args.selection_manifest.resolve(),
        result_paths=[path.resolve() for path in args.result],
        expected_commit=args.expected_commit,
    )
    _atomic_write(args.summary.resolve(), summary)
    _atomic_write(args.validation.resolve(), validation)
    print(json.dumps({
        "interpretation": summary["interpretation"],
        "gate": summary["gate"],
        "result_payload_sha256": summary["result_payload_sha256"],
        "validation_payload_sha256": validation["validation_payload_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
