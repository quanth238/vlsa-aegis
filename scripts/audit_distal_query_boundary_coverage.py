#!/usr/bin/env python3
"""Discover initially safe, real-query warning states on grouped episodes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.evaluate_distal_query_action_risk_e05 import (
    _allocation_record, _disable_images,
)
from scripts.evaluate_distal_smooth_field_attribution_e05 import (
    InstrumentedContinuationProbe,
)
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _git_identity, _load, _require, _sha256,
)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _prefix_summary(record: Mapping[str, Any]) -> dict[str, Any]:
    import numpy as np

    trace = np.asarray(record["clearance_trace_m"], dtype=np.float64)[:, :7]
    evaluated = trace[1:]
    identities = list(record["sample_identities"])[1:]
    _require(evaluated.shape[0] == len(identities), "prefix identities differ")
    flat = int(np.argmin(evaluated))
    sample_index, row = np.unravel_index(flat, evaluated.shape)
    return {
        "row_minimum_clearance_m": np.min(evaluated, axis=0).tolist(),
        "minimum_clearance_m": float(np.min(evaluated)),
        "active_witness": {
            "row": int(row),
            "action_offset": int(identities[int(sample_index)]["action_offset"]),
            "substep": int(identities[int(sample_index)]["substep"]),
            "clearance_m": float(evaluated[int(sample_index), int(row)]),
        },
        "protected_contact_count": len(record["protected_contacts"]),
        "protected_contacts": list(record["protected_contacts"]),
        "maximum_active_obstacle_l1_displacement_m": float(
            record["maximum_active_obstacle_l1_displacement_m"]
        ),
        "sample_count": int(evaluated.shape[0]),
        "substep_counts": list(record["substep_counts"]),
        "maximum_boundary_equivalence_error_m": float(
            record["maximum_boundary_equivalence_error_m"]
        ),
    }


def audit(
    *, repo_root: Path, population_manifest: Path, selection_manifest: Path,
    experiment_config: Path, coverage_config: Path, geometry_config: Path,
    table1_root: Path, case_index: int, expected_commit: str,
    selected_case_override: Mapping[str, Any] | None = None,
    targeted_extension_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    import numpy as np

    from main.evaluate_safelibero_aegis import (
        TABLE_SETTLE_ACTIONS, _active_obstacle, _build_environment,
        _runtime_imports, _settle, read_jsonl, validate_case_row,
    )
    from main.multilink_ellipsoid.clean_action_risk import load_cases, load_config
    from main.multilink_ellipsoid.query_boundary_coverage import (
        RESULT_SCHEMA, ROW_IDENTITIES, initially_safe, load_config as load_coverage_config,
        nominal_prefix_unsafe, row_coverage,
    )
    from main.multilink_ellipsoid.rollout import _dynamic_state_vector
    from main.multilink_ellipsoid.shadow import (
        MultilinkEllipsoidShadow, load_shadow_config,
    )
    from main.multilink_ellipsoid.sitl_candidate import (
        SlabbedEightConstraintProbe, _protected_contact_evidence,
    )

    selection_cfg = load_config(experiment_config)
    config = load_coverage_config(coverage_config)
    if selected_case_override is None:
        cases = [
            item for item in load_cases(selection_manifest, selection_cfg)
            if item["split"] in config["source"]["included_splits"]
        ]
        expected_count = sum(
            int(selection_cfg["cohort"]["episode_counts"][split])
            for split in config["source"]["included_splits"]
        )
        _require(
            len(cases) == expected_count,
            "query-boundary coverage case count differs",
        )
        _require(
            0 <= int(case_index) < len(cases),
            "query-boundary case index differs",
        )
        selected = cases[int(case_index)]
    else:
        selected = dict(selected_case_override)
        _require(
            selected["split"] in ("diagnostic", "train", "validation"),
            "targeted query-boundary split differs",
        )
    source_path = table1_root / selected["archived_result_relative_path"]
    _require(_file_sha256(source_path) == selected["archived_result_file_sha256"],
             "query-boundary source file differs")
    archived = _load(source_path)
    _require(archived["result_payload_sha256"] == selected["archived_result_payload_sha256"],
             "query-boundary source payload differs")
    _require(archived.get("task_success") is True, "query-boundary source task failed")
    rows = [row for row in read_jsonl(population_manifest)
            if row.get("case_id") == selected["case_id"]]
    _require(len(rows) == 1, "query-boundary population case differs")
    validate_case_row(rows[0], repo_root)
    case = dict(rows[0])
    if selected_case_override is not None and "policy_noise_seed" in selected:
        _require(
            int(archived["policy_queries"][0]["rng_seed"])
            == int(selected["policy_noise_seed"]),
            "query-boundary overridden policy seed differs from source ledger",
        )
        case["policy_noise_seed"] = int(selected["policy_noise_seed"])
    action_rows = {int(item["step"]): item for item in archived["actions"]}
    contact_step = int(selected["first_relevant_contact_step"])
    horizon = int(config["nominal_prefix"]["actions"])
    _require(set(range(contact_step + horizon)).issubset(action_rows),
             "query-boundary action ledger ends before prefix")
    queries = {int(item["query_index"]): item for item in archived["policy_queries"]}
    stride = int(config["state_rule"]["query_stride_actions"])

    runtime = _runtime_imports(include_aegis=True)
    env = probe_env = None
    try:
        env, task, observation, _ = _build_environment(runtime, case, render_resolution=32)
        observation = _settle(env, observation, TABLE_SETTLE_ACTIONS)
        probe_env, probe_task, probe_observation, _ = _build_environment(
            runtime, case, render_resolution=32
        )
        probe_observation = _settle(probe_env, probe_observation, TABLE_SETTLE_ACTIONS)
        _require(str(task.language) == str(probe_task.language),
                 "query-boundary probe task differs")
        obstacle_name, _ = _active_obstacle(env, observation)
        _require(obstacle_name == selected["active_obstacle_name"],
                 "query-boundary obstacle differs")
        obstacle_reference = np.asarray(
            observation[obstacle_name + "_pos"], dtype=np.float64
        ).copy()
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
        one_step = SlabbedEightConstraintProbe(
            probe_env, geometry, clearance_m=0.0,
            active_obstacle_name=obstacle_name,
        )
        _disable_images(probe_env)
        instrumented = InstrumentedContinuationProbe(
            one_step, obstacle_name, obstacle_reference
        )
        links = geometry._slabbed_links(env)
        _require(
            [item.body_name for item in links]
            == [item["body"] for item in ROW_IDENTITIES],
            "query-boundary row identity differs",
        )
        boundary_records = []
        retained = []
        expected_substeps = int(
            config["nominal_prefix"]["expected_mujoco_substeps_per_action"]
        )
        tolerance = float(
            config["nominal_prefix"]["boundary_equivalence_tolerance_m"]
        )
        for step in range(contact_step):
            if step % stride == 0:
                query_index = step // stride
                _require(query_index in queries, "query-boundary policy query missing")
                query = queries[query_index]
                _require(int(query["rng_seed"]) == int(case["policy_noise_seed"]) + query_index,
                         "query-boundary policy seed differs")
                current_rows = np.asarray(
                    one_step.clearances(env)[:7], dtype=np.float64
                )
                current_contacts = _protected_contact_evidence(
                    env, obstacle_name
                )["events"]
                current_car = float(np.sum(np.abs(
                    np.asarray(observation[obstacle_name + "_pos"], dtype=np.float64)
                    - obstacle_reference
                )))
                current_ok = initially_safe(
                    current_rows,
                    protected_contact_count=len(current_contacts),
                    active_obstacle_l1_displacement_m=current_car,
                    config=config,
                )
                nominal_actions = np.asarray(
                    [action_rows[offset]["executed"]
                     for offset in range(step, step + horizon)],
                    dtype=np.float64,
                )
                prefix = instrumented.rollout_internal(
                    env, nominal_actions, expected_substeps=expected_substeps,
                    boundary_tolerance=tolerance, step_base=step,
                )
                summary = _prefix_summary(prefix)
                prefix_unsafe = nominal_prefix_unsafe(
                    summary["row_minimum_clearance_m"],
                    protected_contact_count=summary["protected_contact_count"],
                    maximum_active_obstacle_l1_displacement_m=(
                        summary["maximum_active_obstacle_l1_displacement_m"]
                    ),
                    config=config,
                )
                record = {
                    "state_id": "%s:q%03d:s%03d" % (
                        selected["case_id"], query_index, step
                    ),
                    "step": step,
                    "query_index": query_index,
                    "query_rng_seed": int(query["rng_seed"]),
                    "query_returned_actions_sha256": query["returned_actions_sha256"],
                    "source_state_sha256": hashlib.sha256(
                        np.asarray(_dynamic_state_vector(env), dtype=np.float64).tobytes()
                    ).hexdigest(),
                    "current": {
                        "row_clearance_m": current_rows.tolist(),
                        "minimum_clearance_m": float(np.min(current_rows)),
                        "active_row": int(np.argmin(current_rows)),
                        "protected_contact_count": len(current_contacts),
                        "protected_contacts": current_contacts,
                        "active_obstacle_l1_displacement_m": current_car,
                        "strictly_safe": current_ok,
                    },
                    "nominal_prefix": summary,
                    "nominal_prefix_unsafe": prefix_unsafe,
                    "retained": bool(current_ok and prefix_unsafe),
                }
                boundary_records.append(record)
                if record["retained"]:
                    retained.append(record)
            observation, _, _, _ = env.step(action_rows[step]["executed"])

        result = {
            "schema_version": RESULT_SCHEMA,
            "status": "complete",
            "scientific_result": True,
            "case_index": int(case_index),
            "case": selected,
            "source": _git_identity(repo_root, expected_commit),
            "allocation": _allocation_record(),
            "config": config,
            "source_result": {
                "path": str(source_path),
                "file_sha256": _file_sha256(source_path),
                "result_payload_sha256": archived["result_payload_sha256"],
            },
            "first_relevant_contact_step": contact_step,
            "query_boundary_count": len(boundary_records),
            "boundary_records": boundary_records,
            "retained_state_count": len(retained),
            "retained_states": retained,
            "row_coverage": row_coverage(retained, config),
            "candidate_execution_performed": False,
            "MLP_training_authorized": False,
            "calibration_authorized": False,
            "QP_authorized": False,
            "closed_loop_authorized": False,
        }
        if targeted_extension_binding is not None:
            result["targeted_extension_binding"] = dict(
                targeted_extension_binding
            )
        result["result_payload_sha256"] = _sha256(_canonical(result))
        return result
    finally:
        if env is not None:
            env.close()
        if probe_env is not None:
            probe_env.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--population-manifest", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--coverage-config", type=Path, required=True)
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
        coverage_config=args.coverage_config.resolve(),
        geometry_config=args.geometry_config.resolve(),
        table1_root=args.table1_root.resolve(), case_index=args.case_index,
        expected_commit=args.expected_commit,
    )
    _atomic_write(args.output.resolve(), result)
    print(json.dumps({
        "case_id": result["case"]["case_id"],
        "retained_state_count": result["retained_state_count"],
        "active_witness_counts": [
            item["nominal_active_witness_count"] for item in result["row_coverage"]
        ],
        "result_payload_sha256": result["result_payload_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
