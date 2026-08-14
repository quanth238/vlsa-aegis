#!/usr/bin/env python3
"""Audit existing exact substep traces against raw protected contacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _git_identity, _load, _require,
)


def _action_sha(actions: Any) -> str:
    return hashlib.sha256(json.dumps(
        actions, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")).hexdigest()


def _prefix_records(result: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    sampling = result.get("adaptive_boundary_sampling") or {}
    records = []
    records.extend(sampling.get("coarse_screening_records", []))
    records.extend(sampling.get("bisection_records", []))
    return records


def extract_records(
    result: Mapping[str, Any], *, state_id: str, classification: str,
) -> list[dict[str, Any]]:
    from main.multilink_ellipsoid.row_contact_alignment import segment_samples

    state_step = int(result["state"]["step"])
    seen_prefixes = set()
    output = []

    def add_prefix(source: str, actions: Any, prefix: Mapping[str, Any]) -> None:
        digest = _action_sha(actions)
        if digest in seen_prefixes:
            return
        seen_prefixes.add(digest)
        output.append({
            "state_id": state_id,
            "classification": classification,
            "phase": source,
            "samples": segment_samples(prefix, start_step=state_step),
        })

    for item in _prefix_records(result):
        add_prefix(
            "exact_screening_prefix",
            item["exact_final_post_aegis_actions"], item["prefix"],
        )
    for candidate_index, candidate in enumerate(result["candidates"]):
        add_prefix(
            "authoritative_candidate_prefix",
            candidate["actions"], candidate["prefix"],
        )
        horizon = len(candidate["prefix"]["substep_counts"])
        for backup in candidate["backup"]["decisions"]:
            backup_step = int(backup["backup_step"])
            output.append({
                "state_id": state_id,
                "classification": classification,
                "phase": "executed_backup_action",
                "candidate_index": candidate_index,
                "backup_step": backup_step,
                "samples": segment_samples(
                    backup["actual"],
                    start_step=state_step + horizon + backup_step,
                ),
            })
        hold = candidate["backup"].get("terminal_hold")
        if hold is not None:
            output.append({
                "state_id": state_id,
                "classification": classification,
                "phase": "terminal_hold",
                "candidate_index": candidate_index,
                "samples": segment_samples(
                    hold,
                    start_step=state_step + horizon
                    + len(candidate["backup"]["decisions"]),
                ),
            })
    return output


def analyze(config: Mapping[str, Any], summary_paths: Sequence[Path]) -> dict[str, Any]:
    from main.multilink_ellipsoid.row_contact_alignment import decision, summarize_samples

    _require(len(summary_paths) == len(config["source_populations"]),
             "row-contact summary count differs")
    records = []
    sources = []
    proxy_valid_records = []
    for expected, path in zip(config["source_populations"], summary_paths):
        summary = _load(path)
        _require(summary["schema_version"] == expected["expected_summary_schema"],
                 "row-contact summary schema differs")
        _require(summary["result_payload_sha256"]
                 == expected["expected_summary_payload_sha256"],
                 "row-contact summary payload differs")
        _require(len(summary["states"]) == int(expected["expected_state_count"]),
                 "row-contact state count differs")
        source_states = []
        for state in summary["states"]:
            _require(state["split"] in expected["allowed_splits"],
                     "row-contact split differs")
            result_path = Path(state["result_path"])
            validation_path = Path(state["validation_path"])
            _require(_file_sha256(result_path) == state["result_file_sha256"],
                     "row-contact result file differs")
            _require(_file_sha256(validation_path) == state["validation_file_sha256"],
                     "row-contact validation file differs")
            result = _load(result_path)
            validation = _load(validation_path)
            _require(result["status"] == "complete", "row-contact result is incomplete")
            _require(validation["status"] in ("complete", "validated"),
                     "row-contact validation is incomplete")
            extracted = extract_records(
                result, state_id=state["state_id"],
                classification=state["classification"],
            )
            records.extend(extracted)
            if state["classification"] != "proxy_invalid":
                proxy_valid_records.extend(extracted)
            source_states.append({
                "state_id": state["state_id"],
                "split": state["split"],
                "classification": state["classification"],
                "result_file_sha256": state["result_file_sha256"],
                "validation_file_sha256": state["validation_file_sha256"],
                "trace_record_count": len(extracted),
            })
        sources.append({
            "name": expected["name"],
            "summary_path": str(path),
            "summary_file_sha256": _file_sha256(path),
            "summary_payload_sha256": summary["result_payload_sha256"],
            "states": source_states,
        })
    all_summary = summarize_samples(records, config)
    proxy_valid_summary = summarize_samples(proxy_valid_records, config)
    return {
        "sources": sources,
        "all_observed_states_including_proxy_invalid": all_summary,
        "proxy_valid_states": proxy_valid_summary,
        "decision": decision(proxy_valid_summary),
        "timeouts_used_as_complete_labels": False,
        "observed_timeout_prefixes_retained_for_geometry_alignment": True,
        "sealed_test_episodes_opened": False,
        "new_simulation_performed": False,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    from main.multilink_ellipsoid.row_contact_alignment import (
        AUDIT_SCHEMA, load_config, payload_sha256,
    )
    from main.multilink_ellipsoid.shadow import allocation_record

    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--summary", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    source = _git_identity(args.repo_root.resolve(), args.expected_commit)
    config = load_config(args.config.resolve())
    analysis = analyze(config, [item.resolve() for item in args.summary])
    output = {
        "schema_version": AUDIT_SCHEMA,
        "status": "complete",
        "scientific_result": True,
        "claim_scope": config["claim_scope"],
        "source": source,
        "allocation": allocation_record(),
        "config": config,
        "analysis": analysis,
    }
    output["result_payload_sha256"] = payload_sha256(
        output, "result_payload_sha256"
    )
    _atomic_write(args.output.resolve(), output)
    print(json.dumps({
        "decision": analysis["decision"],
        "proxy_valid": analysis["proxy_valid_states"],
        "result_payload_sha256": output["result_payload_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
