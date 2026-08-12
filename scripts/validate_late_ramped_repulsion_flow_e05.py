#!/usr/bin/env python3
"""Independently validate the immutable late-ramped flow timing result."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


RESULT_SCHEMA = "vlsa_late_ramped_repulsion_flow_e05_result.v1"
VALIDATION_SCHEMA = "vlsa_late_ramped_repulsion_flow_e05_validation.v1"


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
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
    temporary = path.with_name(".%s.%d.tmp" % (path.name, os.getpid()))
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def validate(result_path: Path, expected_producer_commit: str) -> dict[str, Any]:
    import numpy as np

    from main.multilink_ellipsoid.late_ramped_flow import (
        load_late_ramped_flow_config,
    )

    result = json.loads(result_path.read_text(encoding="utf-8"))
    _require(result.get("schema_version") == RESULT_SCHEMA, "result schema differs")
    _require(result.get("status") == "complete", "result is incomplete")
    _require(result.get("scientific_result") is True, "result is not scientific")
    _require(result["source"]["commit"] == expected_producer_commit, "producer commit differs")
    _require(result["source"]["dirty"] is False, "producer source was dirty")
    _require("H100" in str(result["allocation"]["device"]["name"]), "producer was not H100")
    _require(result["archived_table1"]["read_only"] is True, "Table 1 was not read-only")
    _require(
        result["archived_table1"]["file_sha256"]
        == "273d77cba3fd5b1e457817ad20f8572b5628aeaa8b5b9f768b32e4f095fbad8b",
        "Table-1 hash differs",
    )
    payload_hash = result.pop("result_payload_sha256")
    _require(_sha256(_canonical(result)) == payload_hash, "result payload hash differs")
    result["result_payload_sha256"] = payload_hash
    config = load_late_ramped_flow_config(
        Path(__file__).resolve().parents[1]
        / "configs/vlsa_late_ramped_repulsion_flow_e05.v1.json"
    )
    for key in ("config_file_sha256", "config_payload_sha256"):
        _require(result["config"][key] == config[key], "config hash differs")
    _require(result["query"]["query_index"] == 36, "query index differs")
    _require(result["query"]["step"] == 180, "query step differs")
    _require(
        result["query"]["scientific_arm_pairing"]
        == "same_live_state_observation_rng_seed_physical_direction_and_horizon",
        "pairing differs",
    )
    _require(result["repulsive_model"]["rollout_count"] == 31, "probe count differs")
    _require(result["repulsive_model"]["jacobian_shape"] == [5, 7, 15], "Jacobian shape differs")
    summaries = {}
    schedule_names = (
        "uniform_final_five",
        "late_linear_last_two",
        "final_step_only",
    )
    tolerance = float(config["success_definition"]["matching_tolerance_action_l2"])
    for name in schedule_names:
        arm = result["arms"][name]
        trace = np.asarray(arm["rollout"]["h_opt_m"], dtype=np.float64)
        _require(trace.shape == (10, 7), "%s rollout shape differs" % name)
        minimum = float(np.min(trace[:5]))
        _require(abs(minimum - float(arm["hard_minimum_m"])) <= 1e-12, "%s minimum differs" % name)
        correction = np.asarray(arm["surviving_output_correction"], dtype=np.float64)
        _require(correction.shape == (3, 3), "%s correction shape differs" % name)
        norm = float(np.linalg.norm(correction))
        _require(
            abs(norm - float(arm["surviving_output_correction_l2"])) <= 1e-10,
            "%s correction norm differs" % name,
        )
        _require(
            arm["schedule_action"] == config["flow_guidance"]["schedules"][name],
            "%s schedule differs" % name,
        )
        matched = result["arms"]["norm_matched_posthoc_%s" % name]
        matched_trace = np.asarray(matched["rollout"]["h_opt_m"], dtype=np.float64)
        _require(matched_trace.shape == (10, 7), "%s matched rollout shape differs" % name)
        matched_minimum = float(np.min(matched_trace[:5]))
        _require(abs(matched_minimum - float(matched["hard_minimum_m"])) <= 1e-12, "%s matched minimum differs" % name)
        matched_norm = float(matched["surviving_output_correction_l2"])
        _require(abs(norm - matched_norm) <= tolerance, "%s posthoc norm is not matched" % name)
        summaries[name] = {
            "surviving_output_correction_l2": norm,
            "hard_minimum_m": minimum,
            "posthoc_hard_minimum_m": matched_minimum,
            "task_progress_ratio": float(arm["task_progress_ratio"]),
            "direction_cosine": float(arm["surviving_direction_cosine"]),
        }
    uniform = summaries["uniform_final_five"]["surviving_output_correction_l2"]
    late = summaries["late_linear_last_two"]["surviving_output_correction_l2"]
    final = summaries["final_step_only"]["surviving_output_correction_l2"]
    gate = bool(late > uniform and final > uniform)
    _require(result["timing_gate_pass"] is gate, "timing gate differs")
    _require(result["execution"]["attempted"] is False, "timing diagnostic executed")
    _require(result["primary_problem_solved"] is False, "timing diagnostic claims solved")
    interpretation = (
        "late_guidance_reduces_denoiser_cancellation"
        if gate
        else "late_guidance_does_not_reduce_denoiser_cancellation"
    )
    _require(result["interpretation"] == interpretation, "interpretation differs")
    receipt = {
        "schema_version": VALIDATION_SCHEMA,
        "status": "validated",
        "result_path": str(result_path),
        "result_file_sha256": _file_sha256(result_path),
        "result_payload_sha256": payload_hash,
        "producer_commit": expected_producer_commit,
        "producer_job_id": result["allocation"]["slurm_job_id"],
        "arms": summaries,
        "timing_gate_pass": gate,
        "execution_attempted": False,
        "primary_problem_solved": False,
        "interpretation": interpretation,
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
