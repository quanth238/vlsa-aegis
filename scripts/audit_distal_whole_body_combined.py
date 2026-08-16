#!/usr/bin/env python3
"""Audit the immutable ADR-0177 + ADR-0180 whole-body cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Optional, Sequence

from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _git_identity, _load, _require,
)
from scripts.validate_distal_exact_group_boundary import _scientific_view


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(
    *, repo_root: Path, combined_config_path: Path,
    expected_audit_commit: str,
) -> dict[str, Any]:
    from main.multilink_ellipsoid.exact_group_boundary import (
        CASE_SCHEMA, canonical as artifact_canonical, load_cases, load_config,
        payload_sha256 as artifact_payload_sha256,
    )
    from main.multilink_ellipsoid.whole_body_combined_audit import (
        RESULT_SCHEMA, evaluate_gate, load_config as load_combined_config,
        payload_sha256,
    )
    from main.multilink_ellipsoid.whole_body_support_audit import (
        audit_cases, load_audit_config,
    )

    combined = load_combined_config(combined_config_path)
    support_config_path = repo_root / combined["support_audit_config"]
    _require(
        _file_sha256(support_config_path)
        == combined["support_audit_config_file_sha256"],
        "support audit config hash differs",
    )
    support_config = load_audit_config(support_config_path)

    all_records = []
    bindings = []
    replay_exact = True
    state_hash_exact = True
    physical_false_safes = 0
    context_complete = True
    maximum_bellman_residual = 0.0
    seen_case_ids = set()
    retained_rejections = []
    for source in combined["sources"]:
        cohort_path = repo_root / source["cohort_config"]
        _require(
            _file_sha256(cohort_path) == source["cohort_config_file_sha256"],
            "combined cohort config hash differs",
        )
        cohort = load_config(cohort_path)
        selections = load_cases(repo_root / cohort["selection_manifest"], cohort)
        root = Path(source["artifact_root"])
        validation_path = root / "validation.json"
        validation = _load(validation_path)
        _require(
            _file_sha256(validation_path) == source["validation_file_sha256"],
            "combined source validation file hash differs",
        )
        _require(
            validation.get("validation_payload_sha256")
            == source["validation_payload_sha256"],
            "combined source validation payload differs",
        )
        _require(
            validation.get("validation_payload_sha256")
            == artifact_payload_sha256(
                validation, key="validation_payload_sha256",
            ),
            "combined source validation payload is invalid",
        )

        producer_records = []
        replay_records = []
        progressive_commits = validation.get(
            "progressive_case_source_commits", {}
        )
        for selection in selections:
            case_id = selection["case_id"]
            _require(case_id not in seen_case_ids, "combined case identity repeats")
            seen_case_ids.add(case_id)
            pair = []
            for directory in ("producer", "replay"):
                record = _load(root / directory / (case_id + ".json"))
                _require(record.get("schema_version") == CASE_SCHEMA,
                         "combined case schema differs")
                _require(record.get("case_id") == case_id,
                         "combined case identity differs")
                expected_case_commit = str(
                    progressive_commits.get(case_id, source["artifact_commit"])
                )
                _require(record.get("source", {}).get("commit")
                         == expected_case_commit,
                         "combined case artifact commit differs")
                _require(record.get("result_payload_sha256")
                         == artifact_payload_sha256(record),
                         "combined case payload differs")
                pair.append(record)
            producer_records.append(pair[0])
            replay_records.append(pair[1])
        mismatches = [
            left["case_id"] for left, right in zip(
                producer_records, replay_records,
            )
            if artifact_canonical(_scientific_view(left))
            != artifact_canonical(_scientific_view(right))
        ]
        _require(not mismatches, "combined independent replay differs")
        for record in producer_records:
            if record.get("status") == "complete":
                all_records.append(record)
            else:
                rejection = record.get("rejection") or {}
                _require(
                    record.get("status")
                    == "retained_scientific_rejection_initial_CAR"
                    and rejection.get("retained") is True
                    and rejection.get("candidate_outcomes_observed") is False,
                    "combined retained scientific rejection differs",
                )
                retained_rejections.append({
                    "case_id": record["case_id"],
                    "split": record["selection"]["split"],
                    "target_group": record["selection"]["target_group"],
                    "code": rejection["code"],
                    "reason": rejection["reason"],
                    "source": source["name"],
                })
        summary = validation["summary"]
        independent = validation.get("independent_replay", {})
        replay_exact = bool(
            replay_exact
            and independent.get("exact_scientific_reproduction", False)
        )
        state_hash_exact = bool(
            state_hash_exact and summary.get("source_state_hash_exact", False)
        )
        physical_false_safes += int(summary["physical_false_safe_count"])
        context_complete = bool(
            context_complete
            and summary.get("artifact_superset_complete", False)
            and summary.get("trajectory_context_complete", False)
        )
        maximum_bellman_residual = max(
            maximum_bellman_residual,
            float(summary.get("trajectory_maximum_bellman_residual", 0.0)),
        )
        bindings.append({
            "name": source["name"],
            "case_count": len(selections),
            "artifact_root": str(root),
            "artifact_commit": source["artifact_commit"],
            "validation_file_sha256": _file_sha256(validation_path),
            "validation_payload_sha256": validation[
                "validation_payload_sha256"
            ],
            "independent_replay_exact": not mismatches,
            "usable_complete_case_count": sum(
                record.get("status") == "complete"
                for record in producer_records
            ),
            "retained_scientific_rejection_count": sum(
                record.get("status") != "complete"
                for record in producer_records
            ),
        })

    summary = audit_cases(all_records, support_config)
    gate = evaluate_gate(
        summary, combined, source_replay_exact=replay_exact,
        source_state_hash_exact=state_hash_exact,
        physical_false_safe_count=physical_false_safes,
        context_complete=context_complete,
        maximum_bellman_residual=maximum_bellman_residual,
    )
    output = {
        "schema_version": RESULT_SCHEMA,
        "status": "training_authorized" if gate["training_authorized"]
        else "scientific_no_go",
        "scientific_result": bool(gate["training_authorized"]),
        "claim_scope": combined["claim_scope"],
        "source": _git_identity(repo_root, expected_audit_commit),
        "combined_config": combined,
        "source_bindings": bindings,
        "retained_scientific_rejections": retained_rejections,
        "retained_scientific_rejection_count": len(retained_rejections),
        "summary": summary,
        "gate": gate,
        "training_authorized": bool(gate["training_authorized"]),
        "correction_authorized": False,
        "QP_authorized": False,
        "closed_loop_authorized": False,
    }
    output["audit_payload_sha256"] = payload_sha256(
        output, key="audit_payload_sha256",
    )
    return output


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--combined-config", type=Path, required=True)
    parser.add_argument("--expected-audit-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = audit(
        repo_root=args.repo_root.resolve(),
        combined_config_path=args.combined_config.resolve(),
        expected_audit_commit=args.expected_audit_commit,
    )
    _atomic_write(args.output.resolve(), result)
    print(json.dumps({
        "status": result["status"],
        "gate": result["gate"],
        "audit_payload_sha256": result["audit_payload_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
