#!/usr/bin/env python3
"""Audit target availability separately from unrelated EE/deployment gates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from main.multilink_ellipsoid.affine_coefficient_model import (
    AFFINE_COEFFICIENT_DATASET_SCHEMA, affine_values,
)
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _load, _require,
)


SCHEMA = "vlsa_distal_affine_coefficient_target_analysis.v1"
RESULT_SCHEMA = "vlsa_distal_affine_coefficient_target_analysis_result.v1"


def _hash_without(value: Mapping[str, Any], key: str) -> str:
    payload = dict(value)
    payload.pop(key, None)
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")).hexdigest()


def main() -> int:
    import numpy as np

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dataset-result", type=Path, required=True)
    parser.add_argument("--dataset-validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config_raw = args.config.resolve().read_bytes()
    config = json.loads(config_raw)
    _require(
        config.get("schema_version") == SCHEMA
        and config.get("protocol_id")
        == "vlsa-distal-affine-coefficient-target-analysis-v1",
        "affine target-analysis config differs",
    )
    dataset = _load(args.dataset.resolve())
    result = _load(args.dataset_result.resolve())
    validation = _load(args.dataset_validation.resolve())
    source = config["immutable_source"]
    for path, key in (
        (args.dataset, "dataset_file_sha256"),
        (args.dataset_result, "result_file_sha256"),
        (args.dataset_validation, "validation_file_sha256"),
    ):
        _require(
            _file_sha256(path.resolve()) == source[key],
            "affine target-analysis immutable source differs: %s" % key,
        )
    _require(
        dataset.get("schema_version") == AFFINE_COEFFICIENT_DATASET_SCHEMA
        and dataset.get("dataset_payload_sha256")
        == source["dataset_payload_sha256"]
        == _hash_without(dataset, "dataset_payload_sha256")
        and dataset.get("source_commit") == source["source_commit"]
        and result.get("result_payload_sha256") == source["result_payload_sha256"]
        and result.get("decision", {}).get("neural_training_authorized") is False
        and validation.get("neural_training_authorized") is False,
        "affine target-analysis source decision differs",
    )
    states = dataset["state_records"]
    gate_config = config["target_gate"]
    split_counts = {
        name: sum(item["split"] == name for item in states)
        for name in ("train", "validation", "test")
    }
    _require(
        len(states) == int(gate_config["expected_state_count"])
        and split_counts == gate_config["expected_split_state_counts"],
        "affine target-analysis state groups differ",
    )
    audits = []
    for state in states:
        certificate = state["certificate"]
        target = state.get("coefficient_target")
        support = int(certificate["exact_proxy_raw_safe_grid_candidate_count"])
        valid = bool(certificate["valid"] and target is not None and support > 0)
        maximum_overbound = None
        nonnegative_error = False
        if valid:
            candidates = np.asarray(state["candidate_first_xyz"], dtype=np.float64)
            margins = np.asarray(
                state["candidate_minimum_distal_margin_m"], dtype=np.float64
            )
            predicted = np.asarray([
                affine_values(
                    target["nominal_margin_m"], target["gradient_m_per_action"],
                    target["state_conditioned_error_m"], xyz,
                    state["nominal_first_action"][:3],
                ) for xyz in candidates
            ])
            maximum_overbound = float(np.max(predicted - margins))
            nonnegative_error = bool(np.all(
                np.asarray(target["state_conditioned_error_m"], dtype=np.float64)
                >= 0.0
            ))
            valid = bool(
                maximum_overbound
                <= float(gate_config["maximum_lower_envelope_overbound_m"])
                and nonnegative_error
            )
        audits.append({
            "state_index": state["state_index"], "case_id": state["case_id"],
            "split": state["split"], "state_step": state["state_step"],
            "safe_support_count": support,
            "certificate_valid": bool(certificate["valid"]),
            "maximum_grid_overbound_m": maximum_overbound,
            "nonnegative_state_conditioned_error": nonnegative_error,
            "target_gate_pass": valid,
        })
    target_gate = bool(len(audits) == len(states) and all(
        item["target_gate_pass"] for item in audits
    ))
    diagnostic = {
        "strict_state_gate_failure_count": sum(
            item["gate_pass"] is False for item in states
        ),
        "oracle_qp_failure_count": sum(
            item["oracle_qp"]["valid"] is False for item in states
        ),
        "released_AEGIS_EE_failure_count": sum(
            item["oracle_qp_exact_verification"] is not None
            and item["oracle_qp_exact_verification"][
                "released_AEGIS_EE_proxy_safe"
            ] is False for item in states
        ),
        "test_strict_state_gate_pass": all(
            item["gate_pass"] for item in states if item["split"] == "test"
        ),
    }
    output = {
        "schema_version": RESULT_SCHEMA, "status": "valid",
        "scientific_result": False,
        "claim_scope": config["claim_scope"],
        "config_file_sha256": hashlib.sha256(config_raw).hexdigest(),
        "dataset_file_sha256": source["dataset_file_sha256"],
        "dataset_payload_sha256": source["dataset_payload_sha256"],
        "split_state_counts": split_counts, "state_audits": audits,
        "diagnostic_only": diagnostic,
        "decision": {
            "all_50_coefficient_targets_valid": target_gate,
            "neural_training_authorized": target_gate,
            "strict_dataset_no_go_unchanged": True,
            "closed_loop_e05_authorized": False,
        },
    }
    output["analysis_payload_sha256"] = _hash_without(
        output, "analysis_payload_sha256"
    )
    _atomic_write(args.output.resolve(), output)
    print(json.dumps(output["decision"], sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
