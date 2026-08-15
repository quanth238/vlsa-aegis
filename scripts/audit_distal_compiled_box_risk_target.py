#!/usr/bin/env python3
"""Replay fixed L6 candidates and audit a compiled-box future-risk target."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from scripts.evaluate_distal_query_action_risk_e05 import _allocation_record
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write,
    _file_sha256,
    _git_identity,
    _load,
    _require,
    _sha256,
)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _public(value: Any) -> Any:
    try:
        import numpy as np
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
    except ImportError:
        pass
    if isinstance(value, Mapping):
        return {str(key): _public(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_public(item) for item in value]
    return value


def _candidate_source_car(candidate: Mapping[str, Any]) -> float:
    values = [float(candidate["prefix"]["maximum_active_obstacle_l1_displacement_m"])]
    values.append(float(candidate["backup"]["maximum_active_obstacle_l1_displacement_m"] or 0.0))
    terminal = candidate["backup"].get("terminal_hold")
    if terminal is not None:
        values.append(float(terminal["maximum_active_obstacle_l1_displacement_m"]))
    return max(values)


def _candidate_source_contacts(candidate: Mapping[str, Any]) -> int:
    return int(candidate["prefix"]["protected_contact_count"]) + int(
        candidate["backup"]["protected_contact_count"]
    )


def _evaluate_case(
    *, repo_root: Path, population_manifest: Path, geometry_config_path: Path,
    case_config: Mapping[str, Any], audit_config: Mapping[str, Any],
) -> dict[str, Any]:
    import numpy as np

    from main.evaluate_safelibero_aegis import (
        TABLE_SETTLE_ACTIONS,
        _active_obstacle,
        _build_environment,
        _runtime_imports,
        _settle,
        read_jsonl,
        validate_case_row,
    )
    from main.multilink_ellipsoid.compiled_box_risk_target_audit import (
        candidate_action_sequence,
    )
    from main.multilink_ellipsoid.obstacle_proxy_audit import (
        compiled_obstacle_boxes,
        evaluate_obstacle_representations,
    )
    from main.multilink_ellipsoid.l6_proxy_scale_audit import rescale_row_slacks
    from main.multilink_ellipsoid.rollout import (
        _auxiliary_sim_snapshot,
        _base_env,
        _controller_snapshot,
        _dynamic_state_vector,
        _restore_auxiliary_sim_snapshot,
        _restore_controller_snapshot,
    )
    from main.multilink_ellipsoid.shadow import (
        MultilinkEllipsoidShadow,
        load_shadow_config,
    )
    from main.multilink_ellipsoid.sitl_candidate import (
        SlabbedEightConstraintProbe,
        _obstacle_root_body_id,
        _protected_contact_evidence,
    )

    source_path = Path(case_config["source_result"])
    _require(_file_sha256(source_path) == case_config["source_result_file_sha256"],
             "compiled-box source result file differs")
    source = _load(source_path)
    _require(source["result_payload_sha256"] == case_config["source_result_payload_sha256"],
             "compiled-box source result payload differs")
    _require(source["population_binding"]["selection"]["case_id"] == case_config["case_id"],
             "compiled-box source case differs")
    _require(int(source["state"]["step"]) == int(case_config["state_step"]),
             "compiled-box source state differs")
    archived_path = Path(source["archived_table1"]["path"])
    _require(_file_sha256(archived_path) == source["archived_table1"]["file_sha256"],
             "compiled-box archived ledger differs")
    archived = _load(archived_path)
    _require(archived["result_payload_sha256"] == source["archived_table1"]["result_payload_sha256"],
             "compiled-box archived payload differs")

    rows = [
        row for row in read_jsonl(population_manifest)
        if row.get("case_id") == case_config["case_id"]
    ]
    _require(len(rows) == 1, "compiled-box population case differs")
    case = dict(rows[0])
    validate_case_row(case, repo_root)
    runtime = _runtime_imports(include_aegis=True)
    geometry_config = load_shadow_config(geometry_config_path)
    env = None
    try:
        env, _, observation, _ = _build_environment(
            runtime, case, render_resolution=32
        )
        observation = _settle(env, observation, TABLE_SETTLE_ACTIONS)
        obstacle_name, _ = _active_obstacle(env, observation)
        _require(
            obstacle_name == source["population_binding"]["selection"]["active_obstacle_name"],
            "compiled-box active obstacle differs",
        )
        obstacle_reference = np.asarray(
            observation[obstacle_name + "_pos"], dtype=np.float64
        ).copy()
        perception = archived["perception"]
        geometry = MultilinkEllipsoidShadow.from_aegis_geometry(
            geometry_config,
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
        if case_config["slab_initialization"] == "settled_initial_state_matching_source":
            geometry._slabbed_links(env, include_certificates=True)
        elif case_config["slab_initialization"] != "query_state_matching_source":
            raise ValueError("compiled-box slab initialization differs")

        action_rows = {int(row["step"]): row for row in archived["actions"]}
        state_step = int(case_config["state_step"])
        _require(set(range(state_step)).issubset(action_rows),
                 "compiled-box action history differs")
        for step in range(state_step):
            observation, _, done, _ = env.step(action_rows[step]["executed"])
            _require(not done, "compiled-box episode completed before audit state")

        source_dynamic = np.asarray(_dynamic_state_vector(env), dtype=np.float64).copy()
        source_hash = hashlib.sha256(source_dynamic.tobytes()).hexdigest()
        current = np.asarray(probe.clearances(env)[:7], dtype=np.float64)
        initial_clearance_error = float(np.max(np.abs(
            current - np.asarray(source["state"]["initial_clearance_m"], dtype=np.float64)
        )))
        state_hash_matches = source_hash == source["state"]["source_snapshot_sha256"]
        base = _base_env(env)
        simulator_state = np.asarray(env.sim.get_state().flatten(), dtype=np.float64).copy()
        auxiliary = _auxiliary_sim_snapshot(env)
        controllers = _controller_snapshot(env)
        clock = (int(base.timestep), float(base.cur_time), bool(base.done))

        empirical_proxy = audit_config.get("empirical_l6_proxy")

        def empirical_slacks(values: Sequence[float]) -> list[float]:
            if empirical_proxy is None:
                return [float(item) for item in values]
            return rescale_row_slacks(
                values,
                scale=float(empirical_proxy["uniform_semiaxis_scale"]),
                scaled_rows=empirical_proxy["scaled_rows"],
            )

        initial_representation = evaluate_obstacle_representations(
            geometry._slabbed_links(env),
            geometry.obstacle,
            compiled_obstacle_boxes(env, obstacle_name),
            overlap_tolerance=float(audit_config["gate"]["compiled_overlap_tolerance"]),
        )
        initial_slacks = empirical_slacks(
            initial_representation[
                "compiled_box_union_row_minimum_normalized_radial_slack"
            ]
        )
        initial_contacts = _protected_contact_evidence(env, obstacle_name)

        def restore_source() -> None:
            env.sim.set_state_from_flattened(simulator_state)
            env.sim.forward()
            _restore_auxiliary_sim_snapshot(env, auxiliary)
            _restore_controller_snapshot(env, controllers)
            base.timestep, base.cur_time, base.done = clock
            _require(np.array_equal(_dynamic_state_vector(env), source_dynamic),
                     "compiled-box source restore differs")

        obstacle_id = _obstacle_root_body_id(env.sim.model, obstacle_name)
        expected_substeps = int(
            source["base_method_config"]["state"]["expected_mujoco_substeps_per_action"]
        )
        overlap_tolerance = float(audit_config["gate"]["compiled_overlap_tolerance"])
        car_limit = float(audit_config["gate"]["paper_car_threshold_m"])
        by_name = {candidate["name"]: candidate for candidate in source["candidates"]}
        _require(set(case_config["candidate_names"]).issubset(by_name),
                 "compiled-box candidate names differ")
        candidate_records = []
        for candidate_name in case_config["candidate_names"]:
            candidate = by_name[candidate_name]
            actions, phases = candidate_action_sequence(candidate)
            restore_source()
            proxy_trace = []
            exact_slack_trace = []
            exact_overlap_trace = []
            contacts = []
            displacements = []
            sample_phases = []

            def measure(phase: str, action_offset: int, substep: int) -> None:
                links = geometry._slabbed_links(env)
                boxes = compiled_obstacle_boxes(env, obstacle_name)
                audit = evaluate_obstacle_representations(
                    links, geometry.obstacle, boxes,
                    overlap_tolerance=overlap_tolerance,
                )
                proxy_trace.append(np.asarray(
                    audit["perceived_mvee_row_clearance_m"], dtype=np.float64
                ))
                scaled_slacks = np.asarray(
                    empirical_slacks(
                        audit[
                            "compiled_box_union_row_minimum_normalized_radial_slack"
                        ]
                    ),
                    dtype=np.float64,
                )
                exact_slack_trace.append(scaled_slacks)
                exact_overlap_trace.append(
                    np.asarray(
                        audit["compiled_box_union_row_any_exact_solid_overlap"],
                        dtype=bool,
                    )
                    if empirical_proxy is None
                    else scaled_slacks <= 0.0
                )
                evidence = _protected_contact_evidence(env, obstacle_name)
                for event in evidence["events"]:
                    contacts.append({
                        "phase": phase,
                        "action_offset": int(action_offset),
                        "substep": int(substep),
                        **event,
                    })
                obstacle = np.asarray(env.sim.data.xpos[obstacle_id], dtype=np.float64)
                displacements.append(float(np.sum(np.abs(obstacle - obstacle_reference))))
                sample_phases.append(phase)

            for action_offset, (action, phase) in enumerate(zip(actions, phases)):
                count_before = len(proxy_trace)
                original_step = env.sim.step

                def instrumented_step(*args: Any, **kwargs: Any) -> Any:
                    value = original_step(*args, **kwargs)
                    measure(phase, action_offset, len(proxy_trace) - count_before)
                    return value

                env.sim.step = instrumented_step
                try:
                    env.step(action)
                finally:
                    env.sim.step = original_step
                _require(
                    len(proxy_trace) - count_before == expected_substeps,
                    "compiled-box internal substep count differs",
                )

            proxy = np.asarray(proxy_trace, dtype=np.float64)
            slack = np.asarray(exact_slack_trace, dtype=np.float64)
            overlap = np.asarray(exact_overlap_trace, dtype=bool)
            replayed_row_minimum = np.min(proxy, axis=0)
            source_row_minimum = np.asarray(
                candidate["combined_row_minimum_clearance_m"], dtype=np.float64
            )
            row_error = float(np.max(np.abs(replayed_row_minimum - source_row_minimum)))
            replay_car = float(max(displacements))
            source_car = _candidate_source_car(candidate)
            source_contacts = _candidate_source_contacts(candidate)
            replay_physical_veto = bool(contacts or replay_car > car_limit)
            source_proxy_nonoverlap = bool(float(np.min(source_row_minimum)) >= 0.0)
            source_proxy_buffer_safe = bool(max(candidate["combined_risk"]) <= 0.0)
            any_exact_overlap = bool(np.any(overlap))
            compiled_safe_terminal = bool(
                candidate["terminal_status"] == "SAFE_TERMINAL"
                and not any_exact_overlap
                and not contacts
                and replay_car <= car_limit
            )
            candidate_records.append({
                "case_id": case_config["case_id"],
                "name": candidate_name,
                "requested_alpha": float(candidate["requested_alpha"]),
                "source_effective_post_AEGIS_correction_l2_action": float(
                    candidate.get("effective_post_AEGIS_correction_l2_action", 0.0)
                ),
                "source_terminal_status": candidate["terminal_status"],
                "source_physical_veto": bool(candidate["physical_veto"]),
                "source_raw_protected_contact_count": source_contacts,
                "source_proxy_nonoverlap": source_proxy_nonoverlap,
                "source_proxy_buffer_safe": source_proxy_buffer_safe,
                "source_row_minimum_clearance_m": source_row_minimum.tolist(),
                "replayed_row_minimum_clearance_m": replayed_row_minimum.tolist(),
                "proxy_replay_maximum_error_m": row_error,
                "source_maximum_CAR_m": source_car,
                "replayed_maximum_CAR_m": replay_car,
                "CAR_replay_error_m": abs(replay_car - source_car),
                "replayed_physical_veto": replay_physical_veto,
                "raw_protected_contact_sample_count": len(contacts),
                "raw_protected_contacts": contacts,
                "compiled_box_row_minimum_normalized_radial_slack": np.min(
                    slack, axis=0
                ).tolist(),
                "compiled_box_row_exact_overlap_sample_count": np.sum(
                    overlap, axis=0
                ).astype(int).tolist(),
                "compiled_box_minimum_normalized_radial_slack": float(np.min(slack)),
                "compiled_box_risk": float(-np.min(slack)),
                "compiled_box_any_exact_overlap": any_exact_overlap,
                "compiled_box_safe_terminal": compiled_safe_terminal,
                "empirical_l6_proxy": empirical_proxy,
                "sample_count": int(len(proxy_trace)),
                "action_count": len(actions),
                "phase_sample_counts": {
                    phase: sum(value == phase for value in sample_phases)
                    for phase in ("prefix", "backup", "terminal_hold")
                },
            })

        tolerance = float(audit_config["gate"]["source_replay_tolerance_m"])
        source_replay_exact = bool(
            state_hash_matches
            and initial_clearance_error <= tolerance
            and all(
                candidate["proxy_replay_maximum_error_m"] <= tolerance
                and candidate["CAR_replay_error_m"] <= tolerance
                and candidate["replayed_physical_veto"]
                == candidate["source_physical_veto"]
                for candidate in candidate_records
            )
        )
        return {
            "case_id": case_config["case_id"],
            "state_step": state_step,
            "source_result": str(source_path),
            "source_result_file_sha256": case_config["source_result_file_sha256"],
            "source_result_payload_sha256": case_config["source_result_payload_sha256"],
            "source_snapshot_sha256": source["state"]["source_snapshot_sha256"],
            "replayed_snapshot_sha256": source_hash,
            "state_hash_matches": state_hash_matches,
            "initial_clearance_replay_error_m": initial_clearance_error,
            "initial_compiled_box_minimum_normalized_radial_slack": float(
                min(initial_slacks)
            ),
            "initial_compiled_box_any_exact_overlap": bool(
                initial_representation[
                    "compiled_box_union_any_exact_solid_overlap"
                ]
                if empirical_proxy is None
                else min(initial_slacks) <= 0.0
            ),
            "initial_raw_protected_contact_count": int(
                initial_contacts["nonpositive_protected_contact_count"]
            ),
            "source_replay_exact": source_replay_exact,
            "candidates": candidate_records,
        }
    finally:
        if env is not None:
            env.close()


def audit(
    *, repo_root: Path, config_path: Path, expected_commit: str, output_path: Path,
) -> dict[str, Any]:
    from main.multilink_ellipsoid.compiled_box_risk_target_audit import (
        RESULT_SCHEMA,
        classify,
        load_config,
    )

    config = load_config(config_path)
    population = repo_root / config["population_manifest"]
    geometry = repo_root / config["geometry_config"]
    _require(_file_sha256(population) == config["population_manifest_file_sha256"],
             "compiled-box population manifest differs")
    _require(_file_sha256(geometry) == config["geometry_config_file_sha256"],
             "compiled-box geometry config differs")
    cases = [
        _evaluate_case(
            repo_root=repo_root,
            population_manifest=population,
            geometry_config_path=geometry,
            case_config=case_config,
            audit_config=config,
        )
        for case_config in config["cases"]
    ]
    classification = classify(cases, config["gate"])
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "complete",
        "scientific_result": True,
        "claim_scope": config["claim_scope"],
        "source": _git_identity(repo_root, expected_commit),
        "allocation": _allocation_record(),
        "config": config,
        "cases": cases,
        "classification": classification,
        "strict_gate_pass": classification["strict_gate_pass"],
        "interpretation": (
            "compiled_box_future_risk_target_mechanism_pass"
            if classification["strict_gate_pass"]
            else "compiled_box_future_risk_target_mechanism_no_go"
        ),
        "training_authorized": False,
    }
    result["result_payload_sha256"] = _sha256(_canonical(result))
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = audit(
        repo_root=args.repo_root.resolve(),
        config_path=args.config.resolve(),
        expected_commit=args.expected_commit,
        output_path=args.output.resolve(),
    )
    _atomic_write(args.output.resolve(), result)
    print(json.dumps({
        "strict_gate_pass": result["strict_gate_pass"],
        "interpretation": result["interpretation"],
        "classification": result["classification"],
        "result_payload_sha256": result["result_payload_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
