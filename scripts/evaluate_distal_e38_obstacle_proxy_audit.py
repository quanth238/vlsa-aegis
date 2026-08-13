#!/usr/bin/env python3
"""Replay the successful E38 ledger while changing only obstacle geometry."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Optional, Sequence

from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write,
    _file_sha256,
    _git_identity,
    _load,
    _require,
    _sha256,
)


RESULT_SCHEMA = "vlsa_distal_e38_obstacle_proxy_audit_result.v1"


def _load_config(path: Path) -> dict[str, Any]:
    config = _load(path)
    _require(
        config.get("schema_version") == "vlsa_distal_e38_obstacle_proxy_audit.v1",
        "E38 obstacle proxy audit schema differs",
    )
    _require(
        config.get("protocol_id") == "vlsa-distal-e38-obstacle-proxy-audit-v1",
        "E38 obstacle proxy audit protocol differs",
    )
    _require(config.get("case_id") == "vlsa-t1-goal-ii-t3-e38", "E38 case differs")
    _require(config["measurement"]["start_action"] == 108, "E38 start differs")
    _require(
        config["measurement"]["expected_mujoco_substeps_per_action"] == 25,
        "E38 substep contract differs",
    )
    return config


def _box_vertices(boxes: Sequence[Any]) -> Any:
    import itertools
    import numpy as np

    values = []
    for box in boxes:
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            local = np.asarray(signs, dtype=np.float64) * box.half_extents_m
            values.append(box.center + box.rotation @ local)
    return np.asarray(values, dtype=np.float64)


def _canonical_obstacle_rotation(value: Any) -> tuple[Any, str]:
    import numpy as np

    rotation = np.asarray(value, dtype=np.float64)
    if float(np.linalg.det(rotation)) >= 0.0:
        return rotation, "none"
    output = rotation.copy()
    output[:, -1] *= -1.0
    return output, "flip_last_eigenvector_preserves_centered_ellipsoid"


def evaluate(
    *,
    repo_root: Path,
    table1_manifest_path: Path,
    selection_manifest_path: Path,
    geometry_config_path: Path,
    experiment_config_path: Path,
    receding_result_path: Path,
    expected_commit: str,
) -> dict[str, Any]:
    import numpy as np

    from main.evaluate_safelibero_aegis import (
        TABLE_RENDER_RESOLUTION,
        TABLE_SETTLE_ACTIONS,
        _active_obstacle,
        _build_environment,
        _runtime_imports,
        _settle,
        pairing_record,
        read_jsonl,
        validate_case_row,
    )
    from main.multilink_ellipsoid.barrier import support_gap
    from main.multilink_ellipsoid.geometry import Ellipsoid, minimum_volume_enclosing_ellipsoid
    from main.multilink_ellipsoid.obstacle_proxy_audit import (
        compiled_obstacle_boxes,
        evaluate_obstacle_representations,
    )
    from main.multilink_ellipsoid.shadow import (
        MultilinkEllipsoidShadow,
        _raw_model_data,
        allocation_record,
        load_shadow_config,
    )
    from main.multilink_ellipsoid.sitl_candidate import _protected_contact_evidence
    from main.multilink_ellipsoid.sitl_candidate import _obstacle_root_body_id

    started = time.perf_counter_ns()
    config = _load_config(experiment_config_path)
    source = _git_identity(repo_root, expected_commit)
    allocation = allocation_record()
    registered = config["registered_input"]
    _require(
        _file_sha256(receding_result_path)
        == registered["receding_result_file_sha256"],
        "registered E38 receding result file differs",
    )
    receding = _load(receding_result_path)
    _require(
        receding.get("result_payload_sha256")
        == registered["receding_result_payload_sha256"],
        "registered E38 receding result payload differs",
    )
    _require(
        receding.get("summary", {}).get("physical_collision_free_task_completion") is True,
        "registered E38 receding run is not physical safe task completion",
    )
    selection = [
        json.loads(line)
        for line in selection_manifest_path.read_text().splitlines()
        if line.strip()
    ]
    _require(len(selection) == 1 and selection[0]["case_id"] == config["case_id"], "selection differs")
    rows = [row for row in read_jsonl(table1_manifest_path) if row["case_id"] == config["case_id"]]
    _require(len(rows) == 1, "Table-1 row differs")
    case = rows[0]
    validate_case_row(case, repo_root)
    geometry_config = load_shadow_config(geometry_config_path)
    runtime = _runtime_imports(include_aegis=False)
    env = None
    try:
        env, task, observation, selected_initial_state = _build_environment(
            runtime, case, render_resolution=TABLE_RENDER_RESOLUTION
        )
        observation = _settle(env, observation, TABLE_SETTLE_ACTIONS)
        obstacle_name, _ = _active_obstacle(env, observation)
        _require(obstacle_name == selection[0]["active_obstacle_name"], "obstacle differs")
        pairing = pairing_record(
            case=case,
            selected_initial_state=selected_initial_state,
            settled_observation=observation,
            task_description=str(task.language),
            active_obstacle_name=obstacle_name,
            settled_simulator_state=np.asarray(env.sim.get_state().flatten()),
        )
        for key in (
            "manifest_row_sha256",
            "initial_state_sha256",
            "initial_observation_sha256",
            "settled_simulator_state_sha256",
            "settled_active_obstacle_position_sha256",
            "policy_noise_schedule_sha256",
        ):
            _require(pairing[key] == receding["pairing"][key], "pairing differs: %s" % key)

        perceived_rotation, rotation_rule = _canonical_obstacle_rotation(
            receding["geometry"]["obstacle"]["rotation"]
        )
        perceived = Ellipsoid(
            center=np.asarray(receding["geometry"]["obstacle"]["center_m"], dtype=np.float64),
            rotation=perceived_rotation,
            semiaxes_m=np.asarray(
                receding["geometry"]["obstacle"]["semiaxes_m"], dtype=np.float64
            ),
            body_name=obstacle_name,
            geom_name="released_aegis_perception_mvee",
            bound_source="released_aegis_frozen_perception_mvee",
        )
        geometry = MultilinkEllipsoidShadow(geometry_config, perceived)
        initial_boxes = compiled_obstacle_boxes(env, obstacle_name)
        compiled_vertices = _box_vertices(initial_boxes)
        compiled_mvee_initial = minimum_volume_enclosing_ellipsoid(
            compiled_vertices,
            body_name=obstacle_name,
            geom_name="compiled_obstacle_box_vertex_mvee",
            source_body_names=tuple(sorted({box.body_name for box in initial_boxes})),
            source_geom_names=tuple(box.geom_name for box in initial_boxes),
            relative_padding=1.0e-9,
            tolerance=1.0e-4,
            max_iterations=50000,
        )
        model, data = _raw_model_data(env.sim)
        root_id = _obstacle_root_body_id(env.sim.model, obstacle_name)
        root_rotation = np.asarray(data.xmat[root_id], dtype=np.float64).reshape(3, 3)
        root_position = np.asarray(data.xpos[root_id], dtype=np.float64)
        compiled_center_body = root_rotation.T @ (compiled_mvee_initial.center - root_position)
        compiled_rotation_body = root_rotation.T @ compiled_mvee_initial.rotation

        perception_local = (compiled_vertices - perceived.center) @ perceived.rotation
        perception_vertex_quadratic = np.sum(
            (perception_local / perceived.semiaxes_m) ** 2, axis=1
        )
        ledger = receding["actions"]
        _require(len(ledger) == receding["action_count"], "receding action ledger differs")
        start_action = int(config["measurement"]["start_action"])
        samples = []
        contacts = []
        substep_counts = []

        def measure(step: int, substep: int) -> None:
            links = geometry._slabbed_links(env)
            boxes = compiled_obstacle_boxes(env, obstacle_name)
            live_root_rotation = np.asarray(data.xmat[root_id], dtype=np.float64).reshape(3, 3)
            live_root_position = np.asarray(data.xpos[root_id], dtype=np.float64)
            compiled_mvee = Ellipsoid(
                center=live_root_position + live_root_rotation @ compiled_center_body,
                rotation=live_root_rotation @ compiled_rotation_body,
                semiaxes_m=compiled_mvee_initial.semiaxes_m,
                body_id=root_id,
                body_name=obstacle_name,
                geom_name="compiled_obstacle_box_vertex_mvee",
                bound_source="compiled_collision_box_vertex_mvee",
            )
            representations = evaluate_obstacle_representations(links, perceived, boxes)
            compiled_mvee_gaps = np.asarray(
                [support_gap(link, compiled_mvee) for link in links], dtype=np.float64
            )
            evidence = _protected_contact_evidence(env, obstacle_name)
            for event in evidence["events"]:
                contacts.append({"step": step, "substep": substep, **event})
            samples.append(
                {
                    "step": int(step),
                    "substep": int(substep),
                    "perceived_mvee_minimum_support_gap_m": representations[
                        "perceived_mvee_minimum_clearance_m"
                    ],
                    "compiled_box_vertex_mvee_minimum_support_gap_m": float(
                        np.min(compiled_mvee_gaps)
                    ),
                    "compiled_box_union_minimum_normalized_radial_slack": representations[
                        "compiled_box_union_minimum_normalized_radial_slack"
                    ],
                    "compiled_box_union_any_exact_solid_overlap": representations[
                        "compiled_box_union_any_exact_solid_overlap"
                    ],
                    "compiled_box_union_exact_overlap_count": representations[
                        "compiled_box_union_exact_overlap_count"
                    ],
                    "compiled_box_loewner_minimum_support_gap_m": representations[
                        "compiled_box_loewner_minimum_support_gap_m"
                    ],
                    "compiled_box_union_closest_pair": representations[
                        "compiled_box_union_closest_pair"
                    ],
                    "raw_protected_contact_count": evidence[
                        "nonpositive_protected_contact_count"
                    ],
                }
            )

        for record in ledger:
            step = int(record["step"])
            action = np.asarray(record["action"], dtype=np.float64)
            _require(action.shape == (7,), "receding action shape differs")
            if step == start_action and config["measurement"]["include_pre_action_108_state"]:
                env.sim.forward()
                measure(step, -1)
            count_before = len(samples)
            original_step = env.sim.step

            def instrumented_step(*args: Any, **kwargs: Any) -> Any:
                output = original_step(*args, **kwargs)
                if step >= start_action:
                    measure(step, len(samples) - count_before)
                return output

            env.sim.step = instrumented_step
            try:
                _, _, done, _ = env.step(action.tolist())
            finally:
                env.sim.step = original_step
            if step >= start_action:
                count = len(samples) - count_before
                substep_counts.append(count)
                _require(count == 25, "E38 replay substep count differs")
            _require(bool(done) == bool(record["done"]), "E38 replay done flag differs")
        _require(samples, "E38 proxy audit measured no samples")
        perceived_min = min(item["perceived_mvee_minimum_support_gap_m"] for item in samples)
        compiled_mvee_min = min(
            item["compiled_box_vertex_mvee_minimum_support_gap_m"] for item in samples
        )
        loewner_min = min(item["compiled_box_loewner_minimum_support_gap_m"] for item in samples)
        exact_overlap_samples = sum(
            bool(item["compiled_box_union_any_exact_solid_overlap"]) for item in samples
        )
        raw_contact_samples = sum(bool(item["raw_protected_contact_count"]) for item in samples)
        partial = bool(perceived_min < 0.0 and raw_contact_samples == 0 and exact_overlap_samples == 0)
        strong = bool(partial and compiled_mvee_min >= 0.0)
        unresolved_robot = bool(exact_overlap_samples > 0 and raw_contact_samples == 0)
        if strong:
            interpretation = "perception_mvee_error_strongly_isolated"
        elif partial:
            interpretation = "perceived_proxy_mismatch_isolated_but_single_mvee_or_gap_remains_conservative"
        elif unresolved_robot:
            interpretation = "robot_ellipsoid_empty_space_or_contact_geometry_mismatch_remains"
        else:
            interpretation = "obstacle_proxy_hypothesis_not_isolated"
        perception_volume = float(4.0 * math.pi * np.prod(perceived.semiaxes_m) / 3.0)
        compiled_volume = float(
            4.0 * math.pi * np.prod(compiled_mvee_initial.semiaxes_m) / 3.0
        )
        result = {
            "schema_version": RESULT_SCHEMA,
            "status": "complete",
            "scientific_result": True,
            "case_id": config["case_id"],
            "claim_scope": config["claim_scope"],
            "source": source,
            "allocation": allocation,
            "config": config,
            "registered_receding_result": {
                "path": str(receding_result_path),
                "file_sha256": registered["receding_result_file_sha256"],
                "payload_sha256": registered["receding_result_payload_sha256"],
                "read_only": True,
            },
            "pairing": pairing,
            "obstacle_rotation_canonicalization": rotation_rule,
            "fixed_robot_geometry": geometry.geometry_record(env),
            "obstacle_geometry": {
                "perceived_mvee": perceived.to_record(),
                "compiled_collision_box_count": len(initial_boxes),
                "compiled_collision_boxes_initial": [box.to_record() for box in initial_boxes],
                "compiled_collision_box_vertex_mvee_initial": compiled_mvee_initial.to_record(),
                "compiled_box_vertex_count": int(len(compiled_vertices)),
                "perceived_mvee_contains_all_compiled_box_vertices": bool(
                    float(np.max(perception_vertex_quadratic)) <= 1.0 + 1.0e-10
                ),
                "perceived_mvee_maximum_compiled_vertex_quadratic": float(
                    np.max(perception_vertex_quadratic)
                ),
                "perceived_mvee_volume_m3": perception_volume,
                "compiled_box_vertex_mvee_volume_m3": compiled_volume,
                "perceived_to_compiled_mvee_volume_ratio": perception_volume
                / compiled_volume,
                "center_error_m": float(
                    np.linalg.norm(perceived.center - compiled_mvee_initial.center)
                ),
            },
            "measurement": {
                "sample_count": len(samples),
                "substep_counts": substep_counts,
                "samples": samples,
                "raw_protected_contacts": contacts,
            },
            "summary": {
                "perceived_mvee_minimum_support_gap_m": perceived_min,
                "compiled_box_vertex_mvee_minimum_support_gap_m": compiled_mvee_min,
                "compiled_box_loewner_minimum_support_gap_m": loewner_min,
                "compiled_box_union_exact_overlap_sample_count": exact_overlap_samples,
                "raw_protected_contact_sample_count": raw_contact_samples,
                "partial_proxy_mismatch": partial,
                "strong_perception_mvee_isolation": strong,
                "unresolved_robot_proxy": unresolved_robot,
            },
            "interpretation": interpretation,
            "limitations": [
                "compiled geometry is privileged simulation information",
                "normalized radial slack is not metric clearance",
                "single-case matched diagnostic is not population evidence",
                "no controller action or repulsion output is changed by this audit",
            ],
            "wall_seconds": (time.perf_counter_ns() - started) * 1.0e-9,
        }
        payload = json.dumps(
            result, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        result["result_payload_sha256"] = _sha256(payload)
        return result
    finally:
        if env is not None:
            env.close()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--table1-manifest", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--geometry-config", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--receding-result", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = evaluate(
        repo_root=args.repo_root.resolve(),
        table1_manifest_path=args.table1_manifest.resolve(),
        selection_manifest_path=args.selection_manifest.resolve(),
        geometry_config_path=args.geometry_config.resolve(),
        experiment_config_path=args.experiment_config.resolve(),
        receding_result_path=args.receding_result.resolve(),
        expected_commit=args.expected_commit,
    )
    _atomic_write(args.output.resolve(), result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "summary": result["summary"],
                "interpretation": result["interpretation"],
                "result_payload_sha256": result["result_payload_sha256"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
