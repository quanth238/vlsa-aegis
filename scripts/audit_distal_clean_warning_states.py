#!/usr/bin/env python3
"""Audit a geometry-relative prevention-state rule without candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _git_identity, _load, _require, _sha256,
)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def audit(
    *, repo_root: Path, population_manifest: Path, selection_manifest: Path,
    experiment_config: Path, audit_config: Path, geometry_config: Path,
    table1_root: Path, case_index: int, expected_commit: str,
) -> dict[str, Any]:
    import numpy as np

    from main.evaluate_safelibero_aegis import (
        TABLE_SETTLE_ACTIONS, _active_obstacle, _build_environment,
        _runtime_imports, _settle, read_jsonl, validate_case_row,
    )
    from main.multilink_ellipsoid.clean_action_risk import (
        load_cases, load_config, select_proxy_boundary_lead_state,
    )
    from main.multilink_ellipsoid.shadow import (
        MultilinkEllipsoidShadow, allocation_record, load_shadow_config,
    )
    from main.multilink_ellipsoid.sitl_candidate import (
        SlabbedEightConstraintProbe, _protected_contact_evidence,
    )

    config = load_config(experiment_config)
    cases = load_cases(selection_manifest, config)
    audit_cfg = json.loads(audit_config.read_text())
    _require(
        audit_cfg == {
            "schema_version": "vlsa_distal_clean_warning_state_audit.v1",
            "protocol_id": "vlsa-distal-clean-warning-state-audit-v1",
            "source_manifest": "manifests/vlsa_distal_clean_action_risk.v1.jsonl",
            "included_splits": ["diagnostic", "train", "validation"],
            "excluded_splits": ["test"],
            "selector": "two_actions_before_first_strict_proxy_boundary_crossing",
            "lead_actions_before_proxy_boundary": 2,
            "safety_buffer_m": 0.001,
            "maximum_initial_active_obstacle_l1_displacement_m": 0.001,
            "require_zero_protected_contact_at_selected_state": True,
            "candidate_evaluation": False,
            "model_training": False,
            "claim_scope": "state_selection_audit_only_not_action_risk_not_control_not_qp_not_cbf_not_generalization",
        },
        "warning-state audit config differs",
    )
    eligible = [case for case in cases if case["split"] in audit_cfg["included_splits"]]
    _require(len(eligible) == 15, "warning-state audit case count differs")
    _require(0 <= int(case_index) < len(eligible), "warning-state audit case index differs")
    selected = eligible[int(case_index)]
    source_path = table1_root / selected["archived_result_relative_path"]
    _require(_file_sha256(source_path) == selected["archived_result_file_sha256"], "source file differs")
    archived = _load(source_path)
    _require(archived["result_payload_sha256"] == selected["archived_result_payload_sha256"], "source payload differs")
    rows = [row for row in read_jsonl(population_manifest) if row.get("case_id") == selected["case_id"]]
    _require(len(rows) == 1, "population case differs")
    validate_case_row(rows[0], repo_root)
    actions = {
        int(item["step"]): np.asarray(item["executed"], dtype=np.float64)
        for item in archived["actions"]
    }
    contact_step = int(selected["first_relevant_contact_step"])
    _require(all(step in actions for step in range(contact_step)), "action ledger ends before contact")
    runtime = _runtime_imports(include_aegis=True)
    env = None
    try:
        env, task, observation, _ = _build_environment(runtime, rows[0], render_resolution=32)
        observation = _settle(env, observation, TABLE_SETTLE_ACTIONS)
        obstacle_name, _ = _active_obstacle(env, observation)
        _require(obstacle_name == selected["active_obstacle_name"], "active obstacle differs")
        obstacle_reference = np.asarray(observation[obstacle_name + "_pos"], dtype=np.float64).copy()
        perception = archived["perception"]
        geometry = MultilinkEllipsoidShadow.from_aegis_geometry(
            load_shadow_config(geometry_config),
            {
                "p2": perception["mvee_center"],
                "R2": perception["mvee_rotation"],
                "Q2_diag": perception["mvee_semiaxes"],
                "record": {"label": perception["obstacle_label"]},
            },
        )
        probe = SlabbedEightConstraintProbe(
            env, geometry, clearance_m=0.0, active_obstacle_name=obstacle_name
        )
        trace = []
        for step in range(contact_step):
            clearance = np.asarray(probe.clearances(env)[:7], dtype=np.float64)
            contacts = _protected_contact_evidence(env, obstacle_name)["events"]
            displacement = float(np.sum(np.abs(
                np.asarray(observation[obstacle_name + "_pos"], dtype=np.float64)
                - obstacle_reference
            )))
            trace.append({
                "step": step,
                "row_clearance_m": clearance.tolist(),
                "minimum_clearance_m": float(np.min(clearance)),
                "protected_contact_count": len(contacts),
                "active_obstacle_l1_displacement_m": displacement,
                "physically_valid": bool(
                    not contacts
                    and displacement <= float(audit_cfg["maximum_initial_active_obstacle_l1_displacement_m"])
                ),
            })
            if step < contact_step - 1:
                observation, _, _, _ = env.step(actions[step].tolist())
        selection = select_proxy_boundary_lead_state(
            [item["minimum_clearance_m"] for item in trace],
            [item["physically_valid"] for item in trace],
            safety_buffer_m=float(audit_cfg["safety_buffer_m"]),
            lead_actions=int(audit_cfg["lead_actions_before_proxy_boundary"]),
            stop_before_step=contact_step,
        )
        if selection["status"] == "selected":
            selection["selected_state"] = trace[int(selection["selected_step"])]
        result = {
            "schema_version": "vlsa_distal_clean_warning_state_audit_result.v1",
            "status": "complete",
            "scientific_result": True,
            "case_index": int(case_index),
            "case": selected,
            "source": _git_identity(repo_root, expected_commit),
            "allocation": allocation_record(),
            "audit_config": audit_cfg,
            "source_result": {
                "path": str(source_path),
                "file_sha256": _file_sha256(source_path),
                "result_payload_sha256": archived["result_payload_sha256"],
            },
            "first_relevant_contact_step": contact_step,
            "trace": trace,
            "selection": selection,
            "candidate_evaluation_performed": False,
            "MLP_training_authorized": False,
            "QP_authorized": False,
            "closed_loop_authorized": False,
        }
        result["result_payload_sha256"] = _sha256(_canonical(result))
        return result
    finally:
        if env is not None:
            env.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--population-manifest", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--audit-config", type=Path, required=True)
    parser.add_argument("--geometry-config", type=Path, required=True)
    parser.add_argument("--table1-root", type=Path, required=True)
    parser.add_argument("--case-index", type=int, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = audit(
        repo_root=args.repo_root.resolve(),
        population_manifest=args.population_manifest.resolve(),
        selection_manifest=args.selection_manifest.resolve(),
        experiment_config=args.experiment_config.resolve(),
        audit_config=args.audit_config.resolve(),
        geometry_config=args.geometry_config.resolve(),
        table1_root=args.table1_root.resolve(),
        case_index=args.case_index,
        expected_commit=args.expected_commit,
    )
    _atomic_write(args.output.resolve(), result)
    print(json.dumps({
        "case_id": result["case"]["case_id"],
        "selection": result["selection"],
        "result_payload_sha256": result["result_payload_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
