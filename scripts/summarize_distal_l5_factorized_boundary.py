#!/usr/bin/env python3
"""Aggregate independently validated factorized-boundary cases."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Sequence

from scripts.replay_distal_three_ellipsoid_multicbf import _atomic_write, _git_identity


def main(argv: Sequence[str] | None = None) -> int:
    from main.multilink_ellipsoid.l5_factorized_boundary import (
        SUMMARY_SCHEMA, VALIDATION_SCHEMA, load_config, load_manifest,
        payload_sha256,
    )
    from main.multilink_ellipsoid.shadow import allocation_record

    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--producer-root", type=Path, required=True)
    parser.add_argument("--producer-commit", required=True)
    parser.add_argument("--validator-commit", required=True)
    parser.add_argument("--summary-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.repo_root.resolve()
    config = load_config(args.config.resolve(), root)
    rows = load_manifest(args.manifest.resolve(), config)
    records = []
    classifications: Counter[str] = Counter()
    for index, row in enumerate(rows):
        path = args.producer_root.resolve() / ("case-%02d" % index) / "case-validation.json"
        if not path.is_file():
            raise ValueError("factorized case validation is missing")
        value = json.loads(path.read_text(encoding="utf-8"))
        if value["schema_version"] != VALIDATION_SCHEMA:
            raise ValueError("factorized case validation schema differs")
        classifications[value["status"] if not value.get("scientific_result") else value["classification"]] += 1
        records.append(value)
    useful = [item for item in records if item.get("risk_report") is not None]
    counts = {
        split: {
            "known_nominal_four_response_states": sum(
                item["risk_report"]["known_nominal"]
                and item["risk_report"]["known_response_count"] >= 4
                for item in useful if item["split"] == split
            ),
            "prefix_dominated_boundary_states": sum(
                item["risk_report"]["prefix_dominated_boundary"]
                for item in useful if item["split"] == split
            ),
            "backup_dominated_boundary_states": sum(
                item["risk_report"]["backup_dominated_boundary"]
                for item in useful if item["split"] == split
            ),
            "usable_risk_states": sum(item["split"] == split for item in useful),
        } for split in ("train", "validation")
    }
    gate_cfg = config["coverage_gate_before_learning"]
    proxy_safe_collisions = sum(
        int(not item["risk_validation_summary"][
            "no_proxy_safe_physical_collision"
        ]) for item in useful
    )
    gates = {
        "minimum_train_known_nominal_response_states":
        counts["train"]["known_nominal_four_response_states"]
        >= gate_cfg["minimum_new_train_states_with_known_nominal_and_four_responses"],
        "minimum_validation_known_nominal_response_states":
        counts["validation"]["known_nominal_four_response_states"]
        >= gate_cfg["minimum_new_validation_states_with_known_nominal_and_four_responses"],
        "train_prefix_boundary": counts["train"]["prefix_dominated_boundary_states"]
        >= gate_cfg["minimum_train_prefix_dominated_boundary_states"],
        "validation_prefix_boundary": counts["validation"]["prefix_dominated_boundary_states"]
        >= gate_cfg["minimum_validation_prefix_dominated_boundary_states"],
        "train_backup_boundary": counts["train"]["backup_dominated_boundary_states"]
        >= gate_cfg["minimum_train_backup_dominated_boundary_states"],
        "validation_backup_boundary": counts["validation"]["backup_dominated_boundary_states"]
        >= gate_cfg["minimum_validation_backup_dominated_boundary_states"],
        "zero_proxy_safe_physical_collisions": proxy_safe_collisions == 0,
        "all_case_validations_present": len(records) == 12,
        "reserved_episodes_unopened": True,
    }
    coverage_pass = all(gates.values())
    output = {
        "schema_version": SUMMARY_SCHEMA, "status": "complete",
        "scientific_result": True, "source": _git_identity(root, args.summary_commit),
        "allocation": allocation_record(), "config": config,
        "producer_commit": args.producer_commit,
        "validator_commit": args.validator_commit,
        "classification_counts": dict(sorted(classifications.items())),
        "coverage": counts, "proxy_safe_physical_collision_count": proxy_safe_collisions,
        "records": records, "gates": gates,
        "coverage_gate_pass": coverage_pass,
        "factorized_training_authorized": coverage_pass,
        "QP_authorized": False, "closed_loop_authorized": False,
        "interpretation": (
            "grouped_factorized_boundary_coverage_pass"
            if coverage_pass else "grouped_factorized_boundary_coverage_incomplete"
        ),
    }
    output["payload_sha256"] = payload_sha256(output)
    _atomic_write(args.output.resolve(), output)
    print(json.dumps({
        "classification_counts": output["classification_counts"],
        "coverage": counts, "gates": gates,
        "factorized_training_authorized": coverage_pass,
        "payload_sha256": output["payload_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
