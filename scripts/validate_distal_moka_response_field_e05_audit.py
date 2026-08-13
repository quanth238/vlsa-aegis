#!/usr/bin/env python3
"""Independent structural and arithmetic validator for the frozen-model audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


SCHEMA = "vlsa_distal_moka_response_field_e05_audit_validation.v1"


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_write(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".%s.tmp" % path.name)
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate(result_path: Path, frozen_result_path: Path, expected_commit: str) -> dict[str, Any]:
    result = json.loads(result_path.read_text())
    frozen = json.loads(frozen_result_path.read_text())
    _require(result.get("schema_version") == "vlsa_distal_moka_response_field_e05_audit_result.v1", "audit schema differs")
    _require(result.get("status") == "complete", "audit is incomplete")
    _require(result.get("source", {}).get("commit") == expected_commit, "audit commit differs")
    _require(result.get("rollout_count") == 585, "audit rollout count differs")
    model = result.get("frozen_model", {})
    _require(model.get("parameters_modified") is False, "audit modified model")
    _require(model.get("training_called") is False, "audit trained model")
    _require(model.get("producer_result_file_sha256") == _file_sha256(frozen_result_path), "frozen file hash differs")
    _require(model.get("producer_result_payload_sha256") == frozen.get("result_payload_sha256"), "frozen payload differs")
    _require(model.get("model_sha256") == frozen.get("model", {}).get("model_sha256"), "model hash differs")
    _require(model.get("model_sha256") == model.get("reconstructed_model_payload_sha256"), "reconstructed model differs")
    states = result.get("state_reports", [])
    _require([row.get("step") for row in states] == list(range(178, 187)), "audit states differ")
    _require([row.get("split") for row in states] == ["train"] * 5 + ["validation"] * 2 + ["test"] * 2, "audit splits differ")
    reports = result.get("split_reports", {})
    _require(set(reports) == {"train", "validation", "test"}, "audit split reports differ")
    for split, count in (("train", 5), ("validation", 2), ("test", 2)):
        report = reports[split]
        _require(report.get("state_count") == count, "audit state count differs")
        for key in ("value_rmse_m", "value_bias_m", "value_maximum_optimism_m"):
            _require(math.isfinite(float(report[key])), "audit metric is nonfinite")
        for section in (
            "fit_directional",
            "heldout_directional",
            "local_ridge_heldout_directional",
        ):
            for key in ("cosine", "sign_accuracy", "rmse_m_per_action"):
                _require(math.isfinite(float(report[section][key])), "audit response metric is nonfinite")
    support = result.get("state_support", {}).get("states", [])
    _require(len(support) == 9, "audit support state count differs")
    expected_groups = {
        "simulator_state",
        "auxiliary_state",
        "environment_clock",
        "controller_state",
        "controller_snapshot_duplicate",
        "nominal_action_chunk",
        "obstacle_geometry",
    }
    for state in support:
        _require(set(state.get("groups", {})) == expected_groups, "audit support groups differ")
    near = result.get("near_active_witnesses", {})
    _require(set(near) == {"0.002", "0.005", "physical_boundary_abs_0.005"}, "audit witness reports differ")
    for threshold in ("0.002", "0.005"):
        rows = near[threshold].get("rows", [])
        _require(near[threshold].get("count") == len(rows), "audit witness count differs")
        for row in rows:
            _require(row.get("link_name") in {"robot0_link5", "robot0_link6", "robot0_link7"}, "audit link differs")
            _require(0 <= int(row.get("action_offset")) < 20, "audit horizon differs")
            _require(0 <= int(row.get("primitive_index")) < 15, "audit primitive differs")
            _require(
                0.0 <= float(row.get("paired_primitive_switch_fraction")) <= 1.0,
                "audit primitive switch fraction differs",
            )
    diagnosis = result.get("diagnosis", {})
    _require(diagnosis.get("correction_or_qp_attempted") is False, "audit attempted control")
    payload = dict(result)
    recorded = payload.pop("result_payload_sha256", None)
    recomputed = hashlib.sha256(_canonical(payload)).hexdigest()
    _require(recorded == recomputed, "audit payload hash differs")
    output = {
        "schema_version": SCHEMA,
        "status": "validated",
        "expected_commit": expected_commit,
        "result_file_sha256": _file_sha256(result_path),
        "result_payload_sha256": recorded,
        "frozen_result_file_sha256": _file_sha256(frozen_result_path),
        "diagnosis": diagnosis.get("classification"),
        "rollout_count": result.get("rollout_count"),
    }
    output["validation_payload_sha256"] = hashlib.sha256(_canonical(output)).hexdigest()
    return output


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--frozen-result", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = validate(args.result.resolve(), args.frozen_result.resolve(), args.expected_commit)
    _atomic_write(args.output.resolve(), output)
    print(json.dumps(output, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
