#!/usr/bin/env python3
"""Collect one palm/L5/L6 exact-geometry boundary-canary action bank."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from scripts.audit_distal_compiled_box_risk_target import _evaluate_case
from scripts.evaluate_distal_query_action_risk_e05 import evaluate
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _git_identity, _load, _require, _sha256,
)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def _retained_initial_car_rejection(
    *, repo_root: Path, expected_commit: str, config: Mapping[str, Any],
    case_index: int, selected: Mapping[str, Any], state_step: int,
    reason: str,
) -> dict[str, Any]:
    """Serialize a frozen warning state rejected before candidate execution."""

    from main.multilink_ellipsoid.exact_group_boundary import (
        CASE_SCHEMA, TARGETED_PI05_CONFIG_SCHEMA, payload_sha256,
    )
    from scripts.evaluate_distal_query_action_risk_e05 import _allocation_record

    # The shared evaluator intentionally reports its registered apparatus label
    # (E05), not the prospective episode id. Keep this match exact so only the
    # known pre-candidate CAR rejection is serialized; every other ValueError
    # remains a hard apparatus failure.
    expected_reason = "E05 query boundary already fails CAR"
    if (
        config.get("schema_version") != TARGETED_PI05_CONFIG_SCHEMA
        or reason != expected_reason
    ):
        raise ValueError(reason)
    bank = config["candidate_bank"]
    value = {
        "schema_version": CASE_SCHEMA,
        "status": "retained_scientific_rejection_initial_CAR",
        "scientific_result": False,
        "claim_scope": config["claim_scope"],
        "source": _git_identity(repo_root, expected_commit),
        "allocation": _allocation_record(),
        "config_file_sha256": config["config_file_sha256"],
        "config_payload_sha256": config["config_payload_sha256"],
        "case_index": int(case_index),
        "case_id": selected["case_id"],
        "selection": dict(selected),
        "state_step": int(state_step),
        "query_index": int(state_step) // 5,
        "source_curve": None,
        "exact_case": None,
        "rejection": {
            "code": "initial_active_obstacle_CAR_exceeds_registered_limit",
            "reason": reason,
            "stage": "restored_source_state_before_candidate_execution",
            "candidate_outcomes_observed": False,
            "candidate_count": 0,
            "retained": True,
            "prevention_eligible": False,
            "training_eligible": False,
        },
        "original_AEGIS_EE_QP_enabled": bool(
            bank["released_AEGIS_EE_applied_to_every_candidate"]
        ),
        "learned_correction_QP_enabled": False,
        "training_authorized_for_case": False,
    }
    value["result_payload_sha256"] = payload_sha256(value)
    return value


def _frozen_grid_subset_candidates(
    nominal: Sequence[Sequence[float]], frame: Mapping[str, Sequence[float]],
    bank: Mapping[str, Any],
) -> list[dict[str, Any]]:
    from main.multilink_ellipsoid.active_boundary_search import (
        candidate_definitions as grid_candidate_definitions,
    )

    generation_bank = dict(bank)
    generation_bank["candidate_count_per_job"] = 27
    rows = grid_candidate_definitions(
        nominal, frame, {"finite_search": generation_bank}, bank["temporal_profile"]
    )
    selected = list(bank["selected_candidate_names"])
    by_name = {row["name"]: row for row in rows}
    if set(selected) - set(by_name):
        raise ValueError("frozen candidate name is unavailable")
    output = [by_name[name] for name in selected]
    if len(output) != int(bank["candidate_count_per_job"]):
        raise ValueError("frozen candidate subset count differs")
    return output


def collect(
    *, repo_root: Path, table1_root: Path, config_path: Path,
    case_index: int, expected_commit: str, run_root: Path,
) -> dict[str, Any]:
    import numpy as np

    from main.multilink_ellipsoid.exact_group_boundary import (
        CASE_SCHEMA, DEVELOPMENT_EXCITATION_CONFIG_SCHEMA,
        DEVELOPMENT_L5_CONFIG_SCHEMA, GENERIC_L5_CONFIG_SCHEMA,
        TRAJECTORY_VALUE_CONFIG_SCHEMA, PROSPECTIVE_L5_CONFIG_SCHEMA,
        WHOLE_BODY_SUPERSET_CONFIG_SCHEMA,
        WHOLE_BODY_PROSPECTIVE_CONFIG_SCHEMA,
        WHOLE_BODY_EXTENSION_CONFIG_SCHEMA,
        TARGETED_PI05_CONFIG_SCHEMA,
        load_cases, load_config,
        payload_sha256, warning_step,
    )
    from main.multilink_ellipsoid.generic_action_boundary import (
        candidate_definitions as generic_candidate_definitions,
    )
    from main.multilink_ellipsoid.active_boundary_search import (
        candidate_definitions as grid_candidate_definitions,
    )
    from main.multilink_ellipsoid.normal_risk_curve import (
        RESULT_SCHEMA as CURVE_RESULT_SCHEMA, candidate_definitions,
    )
    from main.multilink_ellipsoid.obstacle_proxy_audit import (
        compiled_obstacle_boxes, minimum_ellipsoid_quadratics_over_boxes,
    )
    from main.multilink_ellipsoid.palm_primitive_audit import (
        fit_compiled_mesh_geom, world_ellipsoid,
    )
    from main.multilink_ellipsoid.shadow import (
        MultilinkEllipsoidShadow, _released_aegis_end_effector_ellipsoid,
        load_shadow_config,
    )

    config = load_config(config_path)
    population_path = repo_root / config["population_manifest"]
    selection_path = repo_root / config["selection_manifest"]
    source_geometry_path = repo_root / config["source_geometry_config"]
    base_risk_path = repo_root / config["base_risk_config"]
    exact_geometry_path = repo_root / config["exact_group_target"]["robot_geometry_config"]
    for path, expected, label in (
        (population_path, config["population_manifest_file_sha256"], "population"),
        (selection_path, config["selection_manifest_file_sha256"], "selection"),
        (source_geometry_path, config["source_geometry_config_file_sha256"], "source geometry"),
        (base_risk_path, config["base_risk_config_file_sha256"], "base risk"),
        (exact_geometry_path, config["exact_group_target"]["robot_geometry_config_file_sha256"], "exact geometry"),
    ):
        _require(_file_sha256(path) == expected, "exact-group %s differs" % label)
    validation = config["exact_geometry_validation"]
    validation_path = Path(validation["path"])
    _require(_file_sha256(validation_path) == validation["file_sha256"],
             "exact geometry validation file differs")
    validation_value = _load(validation_path)
    _require(validation_value["validation_payload_sha256"] == validation["validation_payload_sha256"],
             "exact geometry validation payload differs")
    _require(validation_value["boundary_collection_authorized"] is True,
             "exact geometry did not authorize boundary collection")
    bank_binding = config["candidate_bank"]
    if bank_binding.get("selection_artifact") is not None:
        bank_selection_path = Path(bank_binding["selection_artifact"])
        _require(
            _file_sha256(bank_selection_path)
            == bank_binding["selection_artifact_file_sha256"],
            "frozen candidate-bank selection file differs",
        )
        bank_selection = _load(bank_selection_path)
        _require(
            bank_selection.get("selection_payload_sha256")
            == bank_binding["selection_artifact_payload_sha256"],
            "frozen candidate-bank selection payload differs",
        )

    cases = load_cases(selection_path, config)
    _require(0 <= int(case_index) < len(cases), "exact-group case index differs")
    selected = cases[int(case_index)]
    state_step = warning_step(selected, config)
    archived_path = table1_root / selected["archived_result_relative_path"]
    _require(_file_sha256(archived_path) == selected["archived_result_file_sha256"],
             "exact-group archived Table-1 source differs")
    archived = _load(archived_path)
    _require(archived["result_payload_sha256"] == selected["archived_result_payload_sha256"],
             "exact-group archived payload differs")
    perception = archived.get("perception")
    perception_source_binding = None
    geometry_relative = selected.get("aegis_geometry_result_relative_path")
    if geometry_relative is not None:
        geometry_path = table1_root / str(geometry_relative)
        _require(
            _file_sha256(geometry_path)
            == selected["aegis_geometry_result_file_sha256"],
            "exact-group paired AEGIS geometry source differs",
        )
        geometry_archived = _load(geometry_path)
        _require(
            geometry_archived["result_payload_sha256"]
            == selected["aegis_geometry_result_payload_sha256"],
            "exact-group paired AEGIS geometry payload differs",
        )
        _require(
            geometry_archived["case_id"] == archived["case_id"],
            "exact-group paired AEGIS geometry case differs",
        )
        perception = geometry_archived["perception"]
        perception_source_binding = {
            "role": "fixed_backup_and_diagnostic_EE_geometry_only",
            "state_or_action_source": False,
            "path": str(geometry_path),
            "file_sha256": selected["aegis_geometry_result_file_sha256"],
            "result_payload_sha256": selected[
                "aegis_geometry_result_payload_sha256"
            ],
        }
    _require(isinstance(perception, dict), "exact-group perception source differs")

    exact_cfg = config["exact_group_target"]
    exact_shadow_config = load_shadow_config(exact_geometry_path)
    provider_cache: dict[str, Any] = {}

    def local_frame_provider(env: Any, obstacle_name: str, source_geometry: Any) -> dict[str, Any]:
        del source_geometry
        if not provider_cache:
            from scripts.audit_distal_palm_primitive_case import (
                _canonicalize_perception_ellipsoid_rotation,
            )
            rotation, _ = _canonicalize_perception_ellipsoid_rotation(
                perception["mvee_rotation"]
            )
            provider_cache["shadow"] = MultilinkEllipsoidShadow.from_aegis_geometry(
                exact_shadow_config,
                {
                    "p2": perception["mvee_center"],
                    "R2": rotation,
                    "Q2_diag": perception["mvee_semiaxes"],
                    "record": {"label": perception["obstacle_label"]},
                },
            )
            provider_cache["palm"] = fit_compiled_mesh_geom(
                env, exact_cfg["palm_geom_name"],
                relative_padding=float(exact_cfg["palm_fit"]["relative_padding"]),
                tolerance=float(exact_cfg["palm_fit"]["khachiyan_tolerance"]),
                max_iterations=int(exact_cfg["palm_fit"]["khachiyan_max_iterations"]),
            )
            provider_cache["shadow"]._slabbed_links(env, include_certificates=True)
        palm = world_ellipsoid(env, provider_cache["palm"])
        distal = provider_cache["shadow"]._slabbed_links(env)[
            :int(exact_cfg.get("distal_row_count", 5))
        ]
        rows = [palm] + distal
        if exact_cfg.get("include_released_aegis_end_effector_proxy") is True:
            rows = [_released_aegis_end_effector_ellipsoid(env)] + rows
        indices = exact_cfg["robot_rows"][selected["target_group"]]
        target_rows = [rows[int(index)] for index in indices]
        boxes = compiled_obstacle_boxes(env, obstacle_name)
        quadratics = minimum_ellipsoid_quadratics_over_boxes(target_rows, boxes)
        local_index, box_index = np.unravel_index(int(np.argmin(quadratics)), quadratics.shape)
        row = target_rows[int(local_index)]
        box = boxes[int(box_index)]
        normal = np.asarray(row.center, dtype=np.float64) - np.asarray(box.center, dtype=np.float64)
        norm = float(np.linalg.norm(normal))
        _require(norm > 1.0e-12, "exact-group outward normal is degenerate")
        normal = normal / norm
        tangent_up = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
        tangent_up = tangent_up - float(tangent_up @ normal) * normal
        tangent_norm = float(np.linalg.norm(tangent_up))
        if tangent_norm <= 1.0e-12:
            fallback = np.eye(3, dtype=np.float64)[int(np.argmin(np.abs(normal)))]
            tangent_up = fallback - float(fallback @ normal) * normal
            tangent_norm = float(np.linalg.norm(tangent_up))
        _require(tangent_norm > 1.0e-12, "exact-group upward tangent is degenerate")
        tangent_up = tangent_up / tangent_norm
        tangent_side = np.cross(normal, tangent_up)
        tangent_side = tangent_side / float(np.linalg.norm(tangent_side))
        return {
            "normal": normal.tolist(),
            "tangent_up": tangent_up.tolist(),
            "tangent_side": tangent_side.tolist(),
            "source": "target_group_closest_exact_compiled_box_center_outward_normal",
            "target_group": selected["target_group"],
            "closest_robot_row": int(indices[int(local_index)]),
            "closest_compiled_box": int(box_index),
            "initial_pair_normalized_radial_slack": float(np.sqrt(quadratics[local_index, box_index]) - 1.0),
        }

    bank = config["candidate_bank"]
    generic_bank = config["schema_version"] in (
        GENERIC_L5_CONFIG_SCHEMA, TRAJECTORY_VALUE_CONFIG_SCHEMA,
        PROSPECTIVE_L5_CONFIG_SCHEMA, DEVELOPMENT_L5_CONFIG_SCHEMA,
    )
    grid_bank = config["schema_version"] in (
        DEVELOPMENT_EXCITATION_CONFIG_SCHEMA, WHOLE_BODY_SUPERSET_CONFIG_SCHEMA,
        WHOLE_BODY_PROSPECTIVE_CONFIG_SCHEMA,
        WHOLE_BODY_EXTENSION_CONFIG_SCHEMA,
        TARGETED_PI05_CONFIG_SCHEMA,
    )

    def definitions(nominal, frame, _base):
        if grid_bank:
            if bank.get("selected_candidate_names") is not None:
                return _frozen_grid_subset_candidates(nominal, frame, bank)
            rows = grid_candidate_definitions(
                nominal, frame, {"finite_search": bank}, bank["temporal_profile"]
            )
            return rows
        if generic_bank:
            return generic_candidate_definitions(nominal, bank)
        return candidate_definitions(nominal, frame, bank)

    source_path = run_root / "source-curve.json"
    try:
        raw = evaluate(
            repo_root=repo_root,
            population_manifest_path=population_path,
            archived_path=archived_path,
            geometry_config_path=source_geometry_path,
            experiment_config_path=base_risk_path,
            expected_commit=expected_commit,
            output_path=source_path,
            case_id_override=selected["case_id"],
            state_step_override=state_step,
            query_index_override=state_step // 5,
            result_schema_override=CURVE_RESULT_SCHEMA,
            claim_scope_override=config["claim_scope"],
            population_binding={
                "case_index": int(case_index),
                "selection": selected,
                "selection_manifest": str(selection_path),
                "selection_manifest_sha256": _file_sha256(selection_path),
                "split": selected["split"],
                "sealed_test_access": False,
            },
            candidate_definitions_override=definitions,
            candidate_protocol_binding={
                "candidate_basis": (
                    "geometry_conditioned_normal_tangent_grid"
                    if grid_bank else "symmetric_world_Cartesian_axes"
                    if generic_bank else bank["direction"]
                ),
                "radii_or_alpha": (
                    [bank["correction_l2_action"]]
                    if grid_bank else bank["radii"]
                    if generic_bank else bank["requested_alpha"]
                ),
                "temporal_profile": bank["temporal_profile"],
                "released_AEGIS_EE_applied_to_every_candidate": bool(
                    bank["released_AEGIS_EE_applied_to_every_candidate"]
                ),
                "learned_correction_QP_enabled": False,
            },
            apply_released_aegis_ee_to_all_proposed_actions=bool(
                bank["released_AEGIS_EE_applied_to_every_candidate"]
            ),
            capture_physical_context=True,
            local_frame_provider=local_frame_provider,
            nominal_action_source=config["state_selection"].get(
                "nominal_action_source"
            ),
            perception_override=(
                perception if perception_source_binding is not None else None
            ),
            perception_source_binding=perception_source_binding,
            allow_initial_proxy_unsafe_for_empirical_relabel=True,
            require_archived_task_success=bool(
                config["state_selection"].get("require_archived_task_success", True)
            ),
        )
    except ValueError as error:
        return _retained_initial_car_rejection(
            repo_root=repo_root, expected_commit=expected_commit, config=config,
            case_index=case_index, selected=selected, state_step=state_step,
            reason=str(error),
        )
    nominal = np.asarray(raw["nominal_five_action_chunk"], dtype=np.float64)
    for candidate in raw["candidates"]:
        actions = np.asarray(candidate["actions"], dtype=np.float64)
        candidate["effective_post_AEGIS_correction_l2_action"] = float(
            np.linalg.norm(actions[:, :3] - nominal[:, :3])
        )
    raw["dataset_config_payload_sha256"] = config["config_payload_sha256"]
    raw["scientific_result"] = False
    raw["execution_mode"] = "three_group_apparatus_canary"
    raw.pop("result_payload_sha256", None)
    raw["result_payload_sha256"] = _sha256(_canonical(raw))
    _atomic_write(source_path, raw)

    exact_case = _evaluate_case(
        repo_root=repo_root,
        population_manifest=population_path,
        geometry_config_path=source_geometry_path,
        case_config={
            "case_id": selected["case_id"],
            "state_step": state_step,
            "source_result": str(source_path),
            "source_result_file_sha256": _file_sha256(source_path),
            "source_result_payload_sha256": raw["result_payload_sha256"],
            "slab_initialization": "query_state_matching_source",
            "candidate_names": [row["name"] for row in raw["candidates"]],
        },
        audit_config={
            "gate": config["gate"],
            "exact_group_target": exact_cfg,
            "trajectory_policy_value": config.get("trajectory_policy_value"),
            "artifact_superset": config.get("artifact_superset"),
        },
    )
    value = {
        "schema_version": CASE_SCHEMA,
        "status": "complete",
        "scientific_result": False,
        "claim_scope": config["claim_scope"],
        "source": _git_identity(repo_root, expected_commit),
        "allocation": raw["allocation"],
        "config_file_sha256": config["config_file_sha256"],
        "config_payload_sha256": config["config_payload_sha256"],
        "case_index": int(case_index),
        "case_id": selected["case_id"],
        "selection": selected,
        "state_step": state_step,
        "query_index": state_step // 5,
        "candidate_frame_binding": raw["state"]["candidate_frame_binding"],
        "source_curve": {
            "path": str(source_path),
            "file_sha256": _file_sha256(source_path),
            "result_payload_sha256": raw["result_payload_sha256"],
        },
        "exact_case": exact_case,
        "original_AEGIS_EE_QP_enabled": bool(
            bank["released_AEGIS_EE_applied_to_every_candidate"]
        ),
        "learned_correction_QP_enabled": False,
        "training_authorized_for_case": False,
    }
    value["result_payload_sha256"] = payload_sha256(value)
    return value


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--table1-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--case-index", type=int, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    value = collect(
        repo_root=args.repo_root.resolve(),
        table1_root=args.table1_root.resolve(),
        config_path=args.config.resolve(),
        case_index=args.case_index,
        expected_commit=args.expected_commit,
        run_root=args.run_root.resolve(),
    )
    _atomic_write(args.output.resolve(), value)
    summary = {
        "status": value["status"],
        "case_id": value["case_id"],
        "target_group": value["selection"]["target_group"],
        "state_step": value["state_step"],
        "result_payload_sha256": value["result_payload_sha256"],
    }
    if value.get("exact_case") is not None:
        summary["source_replay_exact"] = value["exact_case"]["source_replay_exact"]
    else:
        summary["rejection"] = value["rejection"]
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
