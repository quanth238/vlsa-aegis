#!/usr/bin/env python3
"""Validate and summarize the grouped empirical candidate-risk dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional, Sequence

from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _load, _sha256,
)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def validate(
    *, repo_root: Path, run_root: Path, config_path: Path,
    expected_commit: str,
) -> dict[str, Any]:
    from main.multilink_ellipsoid.empirical_candidate_risk_dataset import (
        CASE_RESULT_SCHEMA, SUMMARY_SCHEMA, classify, load_cases, load_config,
        warning_step,
    )

    config = load_config(config_path)
    cases = load_cases(repo_root / config["selection_manifest"], config)
    results = []
    artifacts = []
    for index, selected in enumerate(cases):
        path = run_root / ("case-%02d" % index) / "result.json"
        result = _load(path)
        if result.get("schema_version") != CASE_RESULT_SCHEMA:
            raise ValueError("empirical candidate-risk case schema differs")
        if result.get("source", {}).get("commit") != expected_commit:
            raise ValueError("empirical candidate-risk producer commit differs")
        if result.get("case_index") != index or result.get("case_id") != selected["case_id"]:
            raise ValueError("empirical candidate-risk case identity differs")
        if result.get("split") != selected["split"]:
            raise ValueError("empirical candidate-risk split differs")
        if int(result.get("state_step")) != warning_step(selected, config):
            raise ValueError("empirical candidate-risk state step differs")
        if result.get("config_payload_sha256") != config["config_payload_sha256"]:
            raise ValueError("empirical candidate-risk config binding differs")
        payload = result.get("result_payload_sha256")
        copy = dict(result)
        copy.pop("result_payload_sha256", None)
        if payload != _sha256(_canonical(copy)):
            raise ValueError("empirical candidate-risk case payload differs")
        source_path = Path(result["source_curve"]["path"])
        if _file_sha256(source_path) != result["source_curve"]["file_sha256"]:
            raise ValueError("empirical candidate-risk source curve differs")
        source = _load(source_path)
        if source.get("result_payload_sha256") != result["source_curve"]["result_payload_sha256"]:
            raise ValueError("empirical candidate-risk source payload differs")
        empirical = result["empirical_case"]
        if empirical.get("physical_context") is None:
            raise ValueError("empirical candidate-risk physical context missing")
        if len(empirical.get("initial_empirical_robot_rows", [])) != 7:
            raise ValueError("empirical candidate-risk robot rows differ")
        if not empirical.get("initial_compiled_obstacle_boxes"):
            raise ValueError("empirical candidate-risk obstacle boxes missing")
        if len(empirical.get("candidates", [])) != 9:
            raise ValueError("empirical candidate-risk candidate bank differs")
        if any(len(candidate.get("source_executed_actions", [])) != 5
               for candidate in empirical["candidates"]):
            raise ValueError("empirical candidate-risk executed chunk differs")
        results.append(result)
        artifacts.append({
            "case_id": result["case_id"],
            "split": result["split"],
            "state_step": result["state_step"],
            "file_sha256": _file_sha256(path),
            "result_payload_sha256": payload,
            "source_curve_file_sha256": result["source_curve"]["file_sha256"],
            "source_curve_payload_sha256": result["source_curve"]["result_payload_sha256"],
        })
    classification = classify(results, config)
    value = {
        "schema_version": SUMMARY_SCHEMA,
        "status": "validated",
        "scientific_result": True,
        "claim_scope": config["claim_scope"],
        "validator_commit": expected_commit,
        "config": config,
        "case_artifacts": artifacts,
        "classification": classification,
        "strict_gate_pass": classification["strict_gate_pass"],
        "MLP_training_authorized": classification["training_authorized"],
        "QP_authorized": False,
        "closed_loop_authorized": False,
        "interpretation": (
            "grouped_empirical_candidate_risk_dataset_pass"
            if classification["strict_gate_pass"]
            else "grouped_empirical_candidate_risk_dataset_no_go"
        ),
    }
    value["validation_payload_sha256"] = _sha256(_canonical(value))
    return value


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    value = validate(
        repo_root=args.repo_root.resolve(),
        run_root=args.run_root.resolve(),
        config_path=args.config.resolve(),
        expected_commit=args.expected_commit,
    )
    _atomic_write(args.output.resolve(), value)
    print(json.dumps({
        "strict_gate_pass": value["strict_gate_pass"],
        "MLP_training_authorized": value["MLP_training_authorized"],
        "classification": value["classification"],
        "validation_payload_sha256": value["validation_payload_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
