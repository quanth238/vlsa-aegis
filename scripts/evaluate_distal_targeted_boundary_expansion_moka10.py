#!/usr/bin/env python3
"""Collect test-label-free boundary episodes and rerun the frozen audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping

from main.multilink_ellipsoid.affine_coefficient_model import (
    load_affine_coefficient_config,
)
from main.multilink_ellipsoid.multi_region_affine_oracle import (
    load_config as load_multi_region_config,
)
from main.multilink_ellipsoid.targeted_boundary_expansion import (
    RESULT_SCHEMA, SCAN_SCHEMA, SELECTION_SCHEMA, collection_config, evaluate,
    load_config, load_selected_manifest, select_boundary_steps,
)
from main.multilink_ellipsoid.two_step_margin import (
    feature_context, feature_vectors,
)
from main.multilink_ellipsoid.shadow import allocation_record
from scripts.collect_distal_affine_coefficient_moka10 import collect
from scripts.evaluate_distal_execution_margin_nn_e05 import _build_pair, _geometry
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _git_identity, _load, _require,
)
from scripts.collect_distal_boundary_generalization_moka10 import (
    _canonical_action,
)


def _hash_without(value: Mapping[str, Any], key: str) -> str:
    payload = dict(value)
    payload.pop(key, None)
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _scan_complete_episodes(
    *, repo_root: Path, population_path: Path, selected: list[dict[str, Any]],
    archived_root: Path, geometry_path: Path, exact_box_path: Path,
    expected_commit: str, scan_start_step: int,
) -> dict[str, Any]:
    import numpy as np
    from main.evaluate_safelibero_aegis import (
        _runtime_imports, pairing_record, read_jsonl, validate_case_row,
    )
    from main.multilink_ellipsoid.obstacle_primitives import (
        load_obstacle_primitive_config,
    )
    from main.multilink_ellipsoid.oracle_affine import SubstepEightConstraintProbe
    from main.multilink_ellipsoid.shadow import load_shadow_config

    population = {item["case_id"]: item for item in read_jsonl(population_path)}
    geometry_config = load_shadow_config(geometry_path)
    exact_box_config = load_obstacle_primitive_config(exact_box_path)
    runtime = _runtime_imports(include_aegis=False)
    placeholder_path = archived_root / selected[0]["archived_relative_path"]
    placeholder = _load(placeholder_path)
    _require(
        _file_sha256(placeholder_path) == selected[0]["archived_file_sha256"]
        and placeholder.get("result_payload_sha256")
        == selected[0]["archived_payload_sha256"],
        "targeted expansion geometry placeholder differs",
    )
    records: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    pairings: dict[str, Any] = {}
    for selected_row in selected:
        case_id = str(selected_row["case_id"])
        archived_path = archived_root / selected_row["archived_relative_path"]
        archived = _load(archived_path)
        _require(
            _file_sha256(archived_path) == selected_row["archived_file_sha256"]
            and archived.get("result_payload_sha256")
            == selected_row["archived_payload_sha256"]
            and archived.get("case_id") == case_id,
            "targeted expansion archived episode differs: %s" % case_id,
        )
        case = population[case_id]
        validate_case_row(case, repo_root)
        env = probe_env = None
        try:
            env, probe_env, task, observation, setup = _build_pair(runtime, case)
            pairing = pairing_record(
                case=case, selected_initial_state=setup["selected_initial_state"],
                settled_observation=observation, task_description=str(task.language),
                active_obstacle_name=setup["obstacle_name"],
                settled_simulator_state=np.asarray(
                    env.sim.get_state().flatten(), dtype=np.float64,
                ),
            )
            for key in (
                "manifest_row_sha256", "initial_state_sha256",
                "initial_observation_sha256", "settled_simulator_state_sha256",
                "settled_active_obstacle_position_sha256",
                "policy_noise_schedule_sha256",
            ):
                _require(
                    pairing[key] == archived["pairing"][key],
                    "targeted expansion pairing differs for %s: %s"
                    % (case_id, key),
                )
            pairings[case_id] = pairing
            geometry, exact_boxes = _geometry(
                geometry_config=geometry_config,
                exact_box_config=exact_box_config, archived=placeholder, env=env,
                obstacle_name=setup["obstacle_name"],
            )
            probe = SubstepEightConstraintProbe(
                probe_env, geometry,
                active_obstacle_name=setup["obstacle_name"],
                quadratic_tolerance=1.0e-6,
                contact_distance_threshold_m=0.0,
                obstacle_primitive_union=exact_boxes,
            )
            actions = archived["actions"]
            episode_start = len(records)
            for step in range(len(actions) - 1):
                first = _canonical_action(actions[step], step)
                second = _canonical_action(actions[step + 1], step + 1)
                if step >= int(scan_start_step):
                    context = feature_context(env, probe)
                    _, pair_features = feature_vectors(
                        context, first[:3], first[:3], second[:3],
                    )
                    start = context["start_substep"]
                    clearance = np.asarray(
                        context["current_clearance_m"], dtype=np.float64,
                    )
                    records.append({
                        "case_id": case_id, "split": "train",
                        "state_step": int(step),
                        "nominal_first_action": first.tolist(),
                        "nominal_second_action": second.tolist(),
                        "robot_joint_position_rad": list(
                            start["robot_joint_position_rad"]
                        ),
                        "robot_joint_velocity_rad_s": list(
                            start["robot_joint_velocity_rad_s"]
                        ),
                        "controller_goal_position_m": list(
                            start["controller_goal_position_m"]
                        ),
                        "current_clearance_m": clearance.tolist(),
                        "minimum_current_clearance_m": float(np.min(clearance)),
                        "pair_state_feature_vectors": pair_features.tolist(),
                    })
                env.step(first.tolist())
            episodes.append({
                "case_id": case_id, "split": "train",
                "archived_action_count": len(actions),
                "scanned_state_count": len(records) - episode_start,
                "task_success": bool(selected_row["task_success"]),
                "first_robot_contact_step": selected_row["first_robot_contact_step"],
            })
        finally:
            if probe_env is not None:
                probe_env.close()
            if env is not None:
                env.close()
    output = {
        "schema_version": SCAN_SCHEMA,
        "source": _git_identity(repo_root, expected_commit),
        "selection_uses_test_episode_features_or_labels": False,
        "pairings": pairings, "episodes": episodes, "records": records,
        "summary": {
            "complete_episode_count": len(episodes),
            "scanned_state_count": len(records),
            "q_qdot_and_cartesian_action_recorded": True,
        },
    }
    output["scan_payload_sha256"] = _hash_without(output, "scan_payload_sha256")
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--population-manifest", type=Path, required=True)
    parser.add_argument("--selected-manifest", type=Path, required=True)
    parser.add_argument("--archived-root", type=Path, required=True)
    parser.add_argument("--geometry-config", type=Path, required=True)
    parser.add_argument("--exact-box-config", type=Path, required=True)
    parser.add_argument("--affine-config", type=Path, required=True)
    parser.add_argument("--multi-region-config", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--base-expanded-dataset", type=Path, required=True)
    parser.add_argument("--base-expanded-oracle", type=Path, required=True)
    parser.add_argument("--prior-result", type=Path, required=True)
    parser.add_argument("--prior-validation", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--scan-records", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--additional-dataset", type=Path, required=True)
    parser.add_argument("--collection-result", type=Path, required=True)
    parser.add_argument("--expanded-dataset", type=Path, required=True)
    parser.add_argument("--expanded-oracle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter_ns()
    paths = {
        key: value.resolve() for key, value in {
            "repo": args.repo_root, "population": args.population_manifest,
            "selected": args.selected_manifest, "archived": args.archived_root,
            "geometry": args.geometry_config, "exact_box": args.exact_box_config,
            "affine_config": args.affine_config,
            "multi_config": args.multi_region_config, "config": args.config,
            "base_dataset": args.base_expanded_dataset,
            "base_oracle": args.base_expanded_oracle,
            "prior_result": args.prior_result,
            "prior_validation": args.prior_validation,
            "scan": args.scan_records, "selection": args.selection,
            "additional": args.additional_dataset,
            "collection": args.collection_result,
            "expanded_dataset": args.expanded_dataset,
            "expanded_oracle": args.expanded_oracle, "output": args.output,
        }.items()
    }
    config = load_config(paths["config"])
    immutable = config["immutable_source"]
    for path, key, label in (
        (paths["population"], "source_population_manifest_sha256", "population"),
        (paths["selected"], "selected_manifest_sha256", "selected"),
        (paths["geometry"], "geometry_config_file_sha256", "geometry"),
        (paths["exact_box"], "exact_box_config_file_sha256", "exact-box"),
        (paths["affine_config"], "affine_collection_config_file_sha256", "affine-config"),
        (paths["multi_config"], "multi_region_config_file_sha256", "multi-config"),
        (paths["base_dataset"], "base_expanded_dataset_file_sha256", "base-dataset"),
        (paths["base_oracle"], "base_expanded_oracle_file_sha256", "base-oracle"),
        (paths["prior_result"], "prior_result_file_sha256", "prior-result"),
        (paths["prior_validation"], "prior_validation_file_sha256", "prior-validation"),
    ):
        _require(
            _file_sha256(path) == immutable[key],
            "targeted expansion immutable %s differs" % label,
        )
    base_dataset = _load(paths["base_dataset"])
    base_oracle = _load(paths["base_oracle"])
    prior_result = _load(paths["prior_result"])
    prior_validation = _load(paths["prior_validation"])
    _require(
        base_dataset.get("dataset_payload_sha256")
        == immutable["base_expanded_dataset_payload_sha256"]
        == _hash_without(base_dataset, "dataset_payload_sha256")
        and base_oracle.get("oracle_payload_sha256")
        == immutable["base_expanded_oracle_payload_sha256"]
        == _hash_without(base_oracle, "oracle_payload_sha256")
        and prior_result.get("result_payload_sha256")
        == immutable["prior_result_payload_sha256"]
        == _hash_without(prior_result, "result_payload_sha256")
        and prior_validation.get("status") == "valid"
        and prior_result.get("aggregates", {}).get("supported_test_state_count") == 7,
        "targeted expansion base evidence differs",
    )
    affine = load_affine_coefficient_config(paths["affine_config"])
    multi = load_multi_region_config(paths["multi_config"])
    _require(
        config["multi_region"]["partition"] == multi["partition"]
        and config["multi_region"]["ridge_huber"] == multi["ridge_huber"],
        "targeted expansion regional oracle settings differ",
    )
    selected = load_selected_manifest(paths["selected"], config)
    scan = _scan_complete_episodes(
        repo_root=paths["repo"], population_path=paths["population"],
        selected=selected, archived_root=paths["archived"],
        geometry_path=paths["geometry"], exact_box_path=paths["exact_box"],
        expected_commit=args.expected_commit,
        scan_start_step=int(config["state_selection"]["scan_start_step"]),
    )
    _atomic_write(paths["scan"], scan)
    selection = select_boundary_steps(scan["records"], config)
    selection.update({
        "schema_version": SELECTION_SCHEMA,
        "source_scan_payload_sha256": scan["scan_payload_sha256"],
    })
    selection["selection_payload_sha256"] = _hash_without(
        selection, "selection_payload_sha256"
    )
    _atomic_write(paths["selection"], selection)
    if not selection["all_episode_windows_valid"]:
        output = {
            "schema_version": RESULT_SCHEMA, "status": "complete",
            "scientific_result": True, "claim_scope": config["claim_scope"],
            "source": _git_identity(paths["repo"], args.expected_commit),
            "allocation": allocation_record(), "config": config,
            "complete_episode_scan": {
                "file_sha256": _file_sha256(paths["scan"]),
                "payload_sha256": scan["scan_payload_sha256"],
                "summary": scan["summary"],
            },
            "state_selection": {
                "file_sha256": _file_sha256(paths["selection"]),
                "payload_sha256": selection["selection_payload_sha256"],
                "episode_selections": selection["episode_selections"],
                "uses_test_episode_features_or_labels": False,
            },
            "aggregates": {
                "all_episode_windows_valid": False,
                "current_region_aware_MLP_retraining_preregistration_authorized": False,
                "action_conditioned_model_preregistration_authorized": False,
            },
            "decision": {
                "boundary_collection_simulation_executed": False,
                "current_region_aware_MLP_retraining_authorized": False,
                "action_conditioned_model_authorized": False,
                "closed_loop_E05_remains_blocked": True,
                "stop_reason": "one_or_more_training_episodes_lack_a_registered_boundary_window",
            },
            "wall_seconds": (time.perf_counter_ns() - started) * 1.0e-9,
        }
        output["result_payload_sha256"] = _hash_without(
            output, "result_payload_sha256"
        )
        _atomic_write(paths["output"], output)
        print(json.dumps({
            "selection": selection["episode_selections"],
            "decision": output["decision"], "output": str(paths["output"]),
        }, sort_keys=True), flush=True)
        return 0
    adapted = collection_config(config, affine)
    collection_result = collect(
        repo_root=paths["repo"], population_manifest_path=paths["population"],
        selected_manifest_path=paths["selected"],
        archived_root=paths["archived"], geometry_config_path=paths["geometry"],
        exact_box_config_path=paths["exact_box"], config_path=paths["config"],
        expected_commit=args.expected_commit, dataset_path=paths["additional"],
        config_override=adapted, selected_override=selected,
        registered_state_steps_override=selection[
            "registered_state_steps_by_case"
        ],
    )
    _atomic_write(paths["collection"], collection_result)
    additional = _load(paths["additional"])
    dataset, oracle, audit = evaluate(
        base_dataset, additional, base_oracle, config,
    )
    _atomic_write(paths["expanded_dataset"], dataset)
    _atomic_write(paths["expanded_oracle"], oracle)
    output = {
        "schema_version": RESULT_SCHEMA, "status": "complete",
        "scientific_result": True, "claim_scope": config["claim_scope"],
        "source": _git_identity(paths["repo"], args.expected_commit),
        "allocation": allocation_record(), "config": config,
        "complete_episode_scan": {
            "file_sha256": _file_sha256(paths["scan"]),
            "payload_sha256": scan["scan_payload_sha256"],
            "summary": scan["summary"],
        },
        "state_selection": {
            "file_sha256": _file_sha256(paths["selection"]),
            "payload_sha256": selection["selection_payload_sha256"],
            "episode_selections": selection["episode_selections"],
            "uses_test_episode_features_or_labels": False,
        },
        "additional_collection": {
            "result_file_sha256": _file_sha256(paths["collection"]),
            "result_payload_sha256": collection_result["result_payload_sha256"],
            "dataset_file_sha256": _file_sha256(paths["additional"]),
            "dataset_payload_sha256": additional["dataset_payload_sha256"],
            "summary": additional["summary"],
            "collector_decision": collection_result["decision"],
        },
        "expanded_dataset": {
            "file_sha256": _file_sha256(paths["expanded_dataset"]),
            "payload_sha256": dataset["dataset_payload_sha256"],
            "summary": dataset["summary"],
        },
        "expanded_oracle": {
            "file_sha256": _file_sha256(paths["expanded_oracle"]),
            "payload_sha256": oracle["oracle_payload_sha256"],
            "summary": oracle["summary"],
        },
        "feature_shift": audit["feature_shift"],
        "reference": audit["reference"],
        "validation_state_results": audit["validation_state_results"],
        "test_state_results": audit["test_state_results"],
        "aggregates": audit["aggregates"], "decision": audit["decision"],
        "wall_seconds": (time.perf_counter_ns() - started) * 1.0e-9,
    }
    output["result_payload_sha256"] = _hash_without(
        output, "result_payload_sha256"
    )
    _atomic_write(paths["output"], output)
    print(json.dumps({
        "selection": selection["episode_selections"],
        "aggregates": output["aggregates"], "decision": output["decision"],
        "output": str(paths["output"]),
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
