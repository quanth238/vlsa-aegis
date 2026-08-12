#!/usr/bin/env python3
"""Validate the decisive iterative counterfactual-risk diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Optional, Sequence


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _write_atomic(path: Path, value: Any) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".%s.tmp" % path.name)
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _any_safe(value: Any, buffer_key: str = "buffer_0mm") -> bool:
    if isinstance(value, dict):
        if value.get("gates", {}).get(buffer_key) is True:
            return True
        return any(_any_safe(item, buffer_key) for item in value.values())
    if isinstance(value, list):
        return any(_any_safe(item, buffer_key) for item in value)
    return False


def validate(path: Path, expected_commit: str, validator_commit: str) -> dict[str, Any]:
    import numpy as np

    result = json.loads(path.read_text())
    claimed = result["result_payload_sha256"]
    payload = dict(result)
    payload.pop("result_payload_sha256")
    _require(hashlib.sha256(_canonical(payload)).hexdigest() == claimed, "result hash differs")
    _require(result["source"]["commit"] == expected_commit, "source commit differs")
    _require(not result["source"]["dirty"], "source tree was dirty")
    _require(result["status"] == "complete" and result["scientific_result"], "result incomplete")
    nominal = result["nominal"]
    _require(
        197 in {int(event["step"]) for event in nominal["rollout"]["protected_contacts"]},
        "nominal action-197 contact differs",
    )
    for mode_name in ("exact_endpoint_primary", "soft_endpoint_secondary"):
        mode = result[mode_name]
        if mode is None:
            continue
        preserve = mode_name == "exact_endpoint_primary"
        for method_name in ("field", "analytical"):
            method = mode[method_name]
            for candidate in method["accepted_path"]:
                delta = np.asarray(candidate["correction"], dtype=np.float64).reshape(5, 3)
                if preserve:
                    _require(
                        float(np.max(np.abs(np.sum(delta, axis=0)))) <= 1.0e-9,
                        "endpoint-preserving path differs",
                    )
                _require(
                    abs(float(candidate["pure_risk_m"]) + float(candidate["hard_margin_m"]))
                    <= 1.0e-12,
                    "pure risk differs",
                )
            for iteration in method["iterations"]:
                selected = iteration["selected"]
                if selected is not None:
                    _require(
                        float(selected["hard_margin_m"]) > float(iteration["center"]["hard_margin_m"]),
                        "accepted iterative step did not improve exact risk",
                    )
        search = mode["search"]["by_radius"]
        for radius, record in search.items():
            _require(
                int(record["derivative_free_candidate_count"])
                == int(result["config"]["comparators"]["derivative_free_candidates_per_radius"])
                * int(result["config"]["comparators"]["derivative_free_generations"]),
                "derivative-free budget differs",
            )
            for key in ("random_best", "derivative_free_best"):
                candidate = record[key]
                _require(
                    float(candidate["correction_l2_action"]) <= float(radius) + 1.0e-9,
                    "search correction radius differs",
                )
    primary = result["exact_endpoint_primary"]
    secondary = result["soft_endpoint_secondary"]
    learned = _any_safe(primary["field"]) or bool(secondary and _any_safe(secondary["field"]))
    empirical = _any_safe(primary["search"]) or bool(secondary and _any_safe(secondary["search"]))
    _require(learned == bool(result["learned_field_safe_support"]), "learned support differs")
    _require(empirical == bool(result["empirical_search_safe_support"]), "search support differs")
    audit = result["existing_data_audit"]
    return {
        "schema_version": "vlsa_distal_iterative_counterfactual_risk_e05_validation.v1",
        "status": "valid",
        "scientific_result": True,
        "source_result_path": str(path),
        "source_result_payload_sha256": claimed,
        "source_commit": expected_commit,
        "validator_commit": validator_commit,
        "rollout_count": int(result["rollout_count"]),
        "nominal_hard_margin_mm": 1000.0 * float(nominal["hard_margin_m"]),
        "pure_risk_refit_cosine": float(
            audit["pure_risk_vs_previous_task_penalized_direction_cosine"]
        ),
        "analytical_sign_interpretation": audit["analytical_sign_interpretation"],
        "primary_safe_support": bool(result["primary_safe_support"]),
        "secondary_safe_support": bool(result["secondary_safe_support"]),
        "learned_field_safe_support": learned,
        "empirical_search_safe_support": empirical,
        "interpretation": result["interpretation"],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--validator-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = validate(args.result.resolve(), args.expected_commit, args.validator_commit)
    receipt["validation_payload_sha256"] = hashlib.sha256(_canonical(receipt)).hexdigest()
    _write_atomic(args.output.resolve(), receipt)
    print(json.dumps(receipt, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
