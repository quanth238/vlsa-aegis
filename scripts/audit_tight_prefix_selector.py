#!/usr/bin/env python3
"""Audit frozen tight-prefix candidate selection without simulation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _git_identity, _load, _require,
)
from scripts.train_tight_prefix_risk_q_diagnostic import load_samples


def _candidate_records(
    samples: Sequence[Mapping[str, Any]],
    predictions: Sequence[Sequence[float]],
    *, primary_rows: Sequence[int], diagnostic_rows: Sequence[int],
    group_rows: Mapping[str, Sequence[int]],
) -> list[dict[str, Any]]:
    _require(len(samples) == len(predictions),
             "tight prefix selector prediction count differs")
    required = sorted(set(int(row) for row in primary_rows + diagnostic_rows))
    grouped = {}
    for sample, prediction in zip(samples, predictions):
        _require(len(prediction) == 1, "tight prefix selector prediction differs")
        key = (str(sample["state_id"]), str(sample["candidate_name"]))
        record = grouped.setdefault(key, {
            "state_id": key[0], "candidate_name": key[1],
            "candidate_order": int(sample["candidate_order"]),
            "correction": float(sample["applied_correction_l2_action"]),
            "actual_by_row": {}, "predicted_by_row": {},
        })
        row = int(sample["row_index"])
        _require(row not in record["actual_by_row"],
                 "tight prefix selector row repeats")
        record["actual_by_row"][row] = float(sample["risk"])
        record["predicted_by_row"][row] = float(prediction[0])
    output = []
    for record in grouped.values():
        _require(
            sorted(record["actual_by_row"]) == required
            and sorted(record["predicted_by_row"]) == required,
            "tight prefix selector candidate rows differ",
        )
        record["actual_primary"] = max(
            record["actual_by_row"][int(row)] for row in primary_rows
        )
        record["predicted_primary"] = max(
            record["predicted_by_row"][int(row)] for row in primary_rows
        )
        record["actual_L6"] = max(
            record["actual_by_row"][int(row)] for row in diagnostic_rows
        )
        record["failed_groups"] = sorted(
            group for group, rows in group_rows.items()
            if max(record["actual_by_row"][int(row)] for row in rows) > 0.0
        )
        output.append(record)
    return sorted(output, key=lambda item: (
        item["state_id"], item["candidate_order"], item["candidate_name"],
    ))


def _select(
    rows: Sequence[Mapping[str, Any]], *, rule: str, margin: float,
) -> Optional[Mapping[str, Any]]:
    if rule == "minimum_predicted_primary_risk":
        return min(rows, key=lambda item: (
            float(item["predicted_primary"]), float(item["correction"]),
            int(item["candidate_order"]),
        ))
    threshold = 0.0
    if rule == "validation_margin_least_intervention":
        threshold = -float(margin)
    elif rule != "zero_threshold_least_intervention":
        raise ValueError("tight prefix selector rule differs")
    accepted = [
        item for item in rows
        if float(item["predicted_primary"]) <= threshold
    ]
    if not accepted:
        return None
    return min(accepted, key=lambda item: (
        float(item["correction"]), int(item["candidate_order"]),
    ))


def _evaluate(
    records: Sequence[Mapping[str, Any]], *, rule: str, margin: float,
) -> dict[str, Any]:
    by_state = {}
    for record in records:
        by_state.setdefault(str(record["state_id"]), []).append(record)
    states = []
    for state_id, rows in sorted(by_state.items()):
        selected = _select(rows, rule=rule, margin=margin)
        states.append({
            "state_id": state_id,
            "actual_safe_candidate_count": sum(
                float(item["actual_primary"]) <= 0.0 for item in rows
            ),
            "selected_candidate": None if selected is None else selected[
                "candidate_name"
            ],
            "selected_predicted_primary": None if selected is None else float(
                selected["predicted_primary"]
            ),
            "selected_actual_primary": None if selected is None else float(
                selected["actual_primary"]
            ),
            "selected_primary_safe": None if selected is None else bool(
                float(selected["actual_primary"]) <= 0.0
            ),
            "selected_L6_safe": None if selected is None else bool(
                float(selected["actual_L6"]) <= 0.0
            ),
            "selected_failed_groups": [] if selected is None else selected[
                "failed_groups"
            ],
            "selected_correction": None if selected is None else float(
                selected["correction"]
            ),
        })
    selected = [item for item in states if item["selected_candidate"] is not None]
    return {
        "rule": rule,
        "margin": float(margin),
        "state_count": len(states),
        "safe_selection_count": sum(
            item["selected_primary_safe"] is True for item in states
        ),
        "unsafe_selection_count": sum(
            item["selected_primary_safe"] is False for item in states
        ),
        "abstention_count": sum(
            item["selected_candidate"] is None for item in states
        ),
        "selected_L6_failure_count": sum(
            item["selected_L6_safe"] is False for item in states
        ),
        "mean_selected_correction": None if not selected else sum(
            float(item["selected_correction"]) for item in selected
        ) / len(selected),
        "states": states,
    }


def _passes(summary: Mapping[str, Any], *, config: Mapping[str, Any], split: str) -> bool:
    gate = config["evaluation"]
    if split == "validation":
        safe = int(gate["validation_required_safe_selected_states"])
        unsafe = int(gate["validation_maximum_unsafe_selections"])
        abstain = int(gate["validation_maximum_abstentions"])
    elif split == "test":
        safe = int(gate["diagnostic_test_required_safe_selected_states"])
        unsafe = int(gate["diagnostic_test_maximum_unsafe_selections"])
        abstain = int(gate["diagnostic_test_maximum_abstentions"])
    else:
        raise ValueError("tight prefix selector gate split differs")
    return bool(
        int(summary["safe_selection_count"]) == safe
        and int(summary["unsafe_selection_count"]) <= unsafe
        and int(summary["abstention_count"]) <= abstain
        and (
            not gate["require_zero_selected_L6_failures"]
            or int(summary["selected_L6_failure_count"]) == 0
        )
    )


def run(*, repo_root: Path, config_path: Path, expected_commit: str) -> dict[str, Any]:
    from main.multilink_ellipsoid.pi05_palm_l6_source_selection import (
        cpu_allocation_record,
    )
    from main.multilink_ellipsoid.tight_prefix_risk_q_diagnostic import (
        RESULT_SCHEMA as TRAINING_RESULT_SCHEMA,
        VALIDATION_SCHEMA as TRAINING_VALIDATION_SCHEMA,
        load_config as load_training_config,
        payload_sha256 as training_payload,
    )
    from main.multilink_ellipsoid.tight_prefix_selector_audit import (
        RESULT_SCHEMA, RULES, file_sha256, load_config, payload_sha256,
    )

    config = load_config(config_path)
    source = config["source"]
    training_config_path = repo_root / source["training_config"]
    _require(
        file_sha256(training_config_path)
        == source["training_config_file_sha256"],
        "tight prefix selector training config file differs",
    )
    training_config = load_training_config(training_config_path)
    _require(
        training_config["config_payload_sha256"]
        == source["training_config_payload_sha256"],
        "tight prefix selector training config payload differs",
    )
    result_path = Path(source["training_result"])
    validation_path = Path(source["training_validation"])
    _require(
        _file_sha256(result_path) == source["training_result_file_sha256"]
        and _file_sha256(validation_path)
        == source["training_validation_file_sha256"],
        "tight prefix selector source file differs",
    )
    trained = _load(result_path)
    validated = _load(validation_path)
    _require(
        trained.get("schema_version") == TRAINING_RESULT_SCHEMA
        and trained.get("result_payload_sha256")
        == source["training_result_payload_sha256"]
        == training_payload(trained, "result_payload_sha256")
        and validated.get("schema_version") == TRAINING_VALIDATION_SCHEMA
        and validated.get("validation_payload_sha256")
        == source["training_validation_payload_sha256"]
        == training_payload(validated, "validation_payload_sha256")
        and validated.get("independent_training_exact") is True
        and validated.get("model_sha256") == source["model_sha256"]
        == trained["compact_shared_7D"]["model"]["model_sha256"],
        "tight prefix selector validated model differs",
    )
    samples, sample_source = load_samples(
        repo_root=repo_root, config=training_config,
    )
    predictions = trained["compact_shared_7D"]["predictions"]
    primary_rows = config["selection"]["primary_rows"]
    diagnostic_rows = config["selection"]["diagnostic_rows"]
    records = {
        split: _candidate_records(
            samples[split], predictions[split],
            primary_rows=primary_rows, diagnostic_rows=diagnostic_rows,
            group_rows=training_config["group_rows"],
        )
        for split in ("validation", "test")
    }
    validation_margin = max(
        0.0,
        max(
            float(item["actual_primary"]) - float(item["predicted_primary"])
            for item in records["validation"]
        ),
    )
    validation_rules = {
        rule: _evaluate(
            records["validation"], rule=rule, margin=validation_margin,
        )
        for rule in RULES
    }
    frozen_rule = next(
        (
            rule for rule in RULES
            if _passes(validation_rules[rule], config=config, split="validation")
        ),
        None,
    )
    test_rules = {
        rule: _evaluate(records["test"], rule=rule, margin=validation_margin)
        for rule in RULES
    }
    frozen_test_pass = bool(
        frozen_rule is not None
        and _passes(test_rules[frozen_rule], config=config, split="test")
    )
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "complete_offline_selector_audit",
        "scientific_result": True,
        "claim_scope": config["claim_scope"],
        "source": _git_identity(repo_root, expected_commit),
        "allocation": cpu_allocation_record(),
        "config": config,
        "source_artifacts": {
            "training_result": source["training_result"],
            "training_validation": source["training_validation"],
            "model_sha256": source["model_sha256"],
            "sample_source": sample_source,
        },
        "validation_margin": float(validation_margin),
        "validation_rules": validation_rules,
        "frozen_rule_from_validation_only": frozen_rule,
        "diagnostic_test_rules": test_rules,
        "frozen_rule_diagnostic_test_pass": frozen_test_pass,
        "opened_full_episode_pilot_supported": frozen_test_pass,
        "new_simulator_rollout_count": 0,
        "model_retraining_performed": False,
        "correction_or_QP_executed": False,
        "paper_or_safety_claim_authorized": False,
    }
    result["result_payload_sha256"] = payload_sha256(
        result, "result_payload_sha256",
    )
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run(
        repo_root=args.repo_root.resolve(), config_path=args.config.resolve(),
        expected_commit=args.expected_commit,
    )
    _atomic_write(args.output.resolve(), result)
    print(json.dumps({
        "status": result["status"],
        "validation_margin": result["validation_margin"],
        "validation_rules": result["validation_rules"],
        "frozen_rule": result["frozen_rule_from_validation_only"],
        "diagnostic_test_rules": result["diagnostic_test_rules"],
        "opened_full_episode_pilot_supported": result[
            "opened_full_episode_pilot_supported"
        ],
        "result_payload_sha256": result["result_payload_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
