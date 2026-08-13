#!/usr/bin/env python3
"""Independent validator for the compact-input Moka memorization ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


SCHEMA = "vlsa_distal_moka_compact_input_ablation_e05_validation.v1"


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _atomic_write(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".%s.tmp" % path.name)
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def validate(result_path: Path, frozen_audit_path: Path, expected_commit: str) -> dict[str, Any]:
    result = json.loads(result_path.read_text())
    frozen = json.loads(frozen_audit_path.read_text())
    _require(result.get("schema_version") == "vlsa_distal_moka_compact_input_ablation_e05_result.v1", "schema differs")
    _require(result.get("status") == "complete", "result is incomplete")
    _require(result.get("source", {}).get("commit") == expected_commit, "commit differs")
    _require(result.get("rollout_count") == 65, "rollout count differs")
    _require(result.get("correction_or_qp_attempted") is False, "control was attempted")
    _require(result.get("state", {}).get("step") == 182, "state differs")
    inputs = result.get("input_ablation", {})
    _require(inputs.get("context_dimension") == 15, "context dimension differs")
    _require(inputs.get("model_input_dimension") == 25, "model dimension differs")
    _require(inputs.get("original_model_input_dimension") == 1055, "original dimension differs")
    _require(inputs.get("only_changed_factor") == "input_representation", "changed factor differs")
    _require(result.get("frozen_audit", {}).get("result_file_sha256") == _file_sha256(frozen_audit_path), "frozen file differs")
    _require(result.get("frozen_audit", {}).get("result_payload_sha256") == frozen.get("result_payload_sha256"), "frozen payload differs")
    compact = result["compact_mlp"]
    ridge = result["local_ridge"]
    for report in (compact["fit"], compact["heldout"], ridge["fit"], ridge["heldout"]):
        for key in ("cosine", "sign_accuracy", "rmse_m_per_action"):
            _require(math.isfinite(float(report[key])), "response metric is nonfinite")
    _require(ridge.get("design_rank") == 15, "fit directions are not full rank")
    config_gate = result["config"]["gate"]
    fit_ratio = float(compact["fit"]["rmse_m_per_action"]) / max(float(ridge["fit"]["rmse_m_per_action"]), 1.0e-12)
    held_ratio = float(compact["heldout"]["rmse_m_per_action"]) / max(float(ridge["heldout"]["rmse_m_per_action"]), 1.0e-12)
    held_delta = float(compact["heldout"]["cosine"]) - float(ridge["heldout"]["cosine"])
    checks = {
        "fit_response_cosine": float(compact["fit"]["cosine"]) >= float(config_gate["minimum_fit_response_cosine"]),
        "fit_sign_accuracy": float(compact["fit"]["sign_accuracy"]) >= float(config_gate["minimum_fit_sign_accuracy"]),
        "fit_rmse_relative_to_ridge": fit_ratio <= float(config_gate["maximum_fit_rmse_ratio_to_local_ridge"]),
        "row_gradient_cosine_to_ridge": float(compact["row_gradient_cosine_to_ridge"]["mean"]) >= float(config_gate["minimum_row_gradient_cosine_to_local_ridge"]),
        "heldout_cosine_relative_to_ridge": held_delta >= float(config_gate["minimum_heldout_cosine_relative_to_local_ridge"]),
        "heldout_rmse_relative_to_ridge": held_ratio <= float(config_gate["maximum_heldout_rmse_ratio_to_local_ridge"]),
    }
    _require(result["gate"]["checks"] == checks, "gate checks differ")
    _require(result["gate"]["pass"] is bool(all(checks.values())), "gate verdict differs")
    _require(abs(float(result["gate"]["fit_rmse_ratio_to_ridge"]) - fit_ratio) <= 1.0e-12, "fit ratio differs")
    _require(abs(float(result["gate"]["heldout_rmse_ratio_to_ridge"]) - held_ratio) <= 1.0e-12, "heldout ratio differs")
    _require(abs(float(result["gate"]["heldout_cosine_delta_from_ridge"]) - held_delta) <= 1.0e-12, "heldout cosine delta differs")
    payload = dict(result)
    recorded = payload.pop("result_payload_sha256", None)
    _require(hashlib.sha256(_canonical(payload)).hexdigest() == recorded, "payload hash differs")
    output = {
        "schema_version": SCHEMA,
        "status": "validated",
        "expected_commit": expected_commit,
        "result_file_sha256": _file_sha256(result_path),
        "result_payload_sha256": recorded,
        "frozen_audit_file_sha256": _file_sha256(frozen_audit_path),
        "gate_pass": result["gate"]["pass"],
        "interpretation": result["interpretation"],
        "rollout_count": result["rollout_count"],
    }
    output["validation_payload_sha256"] = hashlib.sha256(_canonical(output)).hexdigest()
    return output


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--frozen-audit", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = validate(args.result.resolve(), args.frozen_audit.resolve(), args.expected_commit)
    _atomic_write(args.output.resolve(), output)
    print(json.dumps(output, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
