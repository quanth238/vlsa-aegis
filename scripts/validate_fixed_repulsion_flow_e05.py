#!/usr/bin/env python3
"""Independently validate the immutable fixed-repulsion flow result."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


RESULT_SCHEMA = "vlsa_fixed_repulsion_flow_e05_result.v1"
VALIDATION_SCHEMA = "vlsa_fixed_repulsion_flow_e05_validation.v1"


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _atomic_write(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".%s.%d.tmp" % (path.name, os.getpid()))
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def validate(result_path: Path, expected_producer_commit: str) -> dict[str, Any]:
    import numpy as np

    from main.multilink_ellipsoid.repulsive_flow import load_repulsive_flow_config

    result = json.loads(result_path.read_text(encoding="utf-8"))
    _require(result.get("schema_version") == RESULT_SCHEMA, "result schema differs")
    _require(result.get("status") == "complete", "result is incomplete")
    _require(result.get("scientific_result") is True, "result is not scientific")
    _require(result["source"]["commit"] == expected_producer_commit, "producer commit differs")
    _require(result["source"]["dirty"] is False, "producer source was dirty")
    _require("H100" in result["allocation"]["device"]["name"], "producer was not H100")
    payload_hash = result.pop("result_payload_sha256")
    _require(_sha256(_canonical(result)) == payload_hash, "result payload hash differs")
    result["result_payload_sha256"] = payload_hash
    config = load_repulsive_flow_config(
        Path(__file__).resolve().parents[1]
        / "configs/vlsa_fixed_repulsion_flow_e05.v1.json"
    )
    for key in ("config_file_sha256", "config_payload_sha256"):
        _require(result["config"][key] == config[key], "config hash differs")
    _require(result["query"]["query_index"] == 36, "query index differs")
    _require(result["query"]["step"] == 180, "query step differs")
    _require(
        result["query"]["scientific_arm_pairing"]
        == "same_live_state_observation_rng_seed_and_horizon",
        "arm pairing differs",
    )
    _require(
        result["query"]["live_vs_archived_status"]
        == "diagnostic_only_no_late_query_equivalence_claim",
        "late-query diagnostic differs",
    )
    _require(result["repulsive_model"]["rollout_count"] == 31, "probe count differs")
    _require(result["repulsive_model"]["jacobian_shape"] == [5, 7, 15], "Jacobian shape differs")
    arm_minima = {}
    for name in (
        "ordinary_pi05_chunk",
        "posthoc_fixed_repulsion",
        "fixed_repulsion_inside_final_flow_steps",
    ):
        arm = result["arms"][name]
        trace = np.asarray(arm["rollout"]["h_opt_m"], dtype=np.float64)
        _require(trace.shape == (10, 7), "%s trace shape differs" % name)
        minimum = float(np.min(trace[:5]))
        row_minimum = np.min(trace[:5], axis=0)
        _require(abs(minimum - float(arm["hard_minimum_m"])) <= 1.0e-12, "%s minimum differs" % name)
        _require(np.max(np.abs(row_minimum - np.asarray(arm["row_minimum_m"]))) <= 1.0e-12, "%s row minima differ" % name)
        _require(
            float(arm["rollout"]["synchronization"]["maximum_absolute_error"])
            <= float(config["verification"]["clone_state_tolerance"]),
            "%s clone synchronization differs" % name,
        )
        arm_minima[name] = minimum
    ordinary = arm_minima["ordinary_pi05_chunk"]
    posthoc = arm_minima["posthoc_fixed_repulsion"]
    guided = arm_minima["fixed_repulsion_inside_final_flow_steps"]
    expected_gate = bool(guided >= 0.0 and guided > ordinary and guided > posthoc)
    _require(result["direction_gate_pass"] is expected_gate, "direction gate differs")
    _require(abs((guided - ordinary) - result["guided_minus_ordinary_minimum_m"]) <= 1.0e-12, "guided gain differs")
    _require(abs((guided - posthoc) - result["guided_minus_posthoc_minimum_m"]) <= 1.0e-12, "posthoc comparison differs")
    _require(guided >= 0.0, "guided prefix was not safe")
    _require(guided > ordinary, "guided prefix did not improve ordinary")
    _require(guided < posthoc, "guided prefix did not underperform posthoc")
    _require(result["execution"]["attempted"] is False, "failed comparative gate executed")
    _require(result["primary_problem_solved"] is False, "unexpected solved verdict")
    corrections = np.asarray(
        result["query"]["guided_sampler"]["guided_slot_output_corrections"],
        dtype=np.float64,
    )
    _require(corrections.shape == (3, 3), "guided correction shape differs")
    receipt = {
        "schema_version": VALIDATION_SCHEMA,
        "status": "validated",
        "result_path": str(result_path),
        "result_file_sha256": _file_sha256(result_path),
        "result_payload_sha256": payload_hash,
        "producer_commit": expected_producer_commit,
        "producer_job_id": result["allocation"]["slurm_job_id"],
        "arm_minimum_m": arm_minima,
        "guided_minus_ordinary_m": guided - ordinary,
        "guided_minus_posthoc_m": guided - posthoc,
        "guided_slot_correction_norms": np.linalg.norm(corrections, axis=1).tolist(),
        "direction_gate_pass": expected_gate,
        "execution_attempted": False,
        "primary_problem_solved": False,
        "stored_interpretation": result["interpretation"],
        "corrected_interpretation": "fixed_repulsion_inside_flow_worse_than_posthoc",
    }
    receipt["receipt_payload_sha256"] = _sha256(_canonical(receipt))
    return receipt


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--expected-producer-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = validate(args.result.resolve(), args.expected_producer_commit)
    _atomic_write(args.output.resolve(), receipt)
    print(json.dumps(receipt, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
