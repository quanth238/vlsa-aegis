"""Contracts for exact-geometry grouped boundary collection."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


CONFIG_SCHEMA = "vlsa_distal_exact_group_boundary_canary.v1"
NO_QP_L5_CONFIG_SCHEMA = "vlsa_distal_no_qp_l5_boundary_canary.v1"
GENERIC_L5_CONFIG_SCHEMA = "vlsa_distal_generic_l5_boundary_canary.v1"
TRAJECTORY_VALUE_CONFIG_SCHEMA = "vlsa_distal_generic_l5_trajectory_value_canary.v1"
PROSPECTIVE_L5_CONFIG_SCHEMA = "vlsa_distal_prospective_l5_boundary_population.v1"
DEVELOPMENT_L5_CONFIG_SCHEMA = "vlsa_distal_exact_group_boundary_development_search.v1"
DEVELOPMENT_EXCITATION_CONFIG_SCHEMA = "vlsa_distal_exact_group_boundary_development_excitation.v1"
WHOLE_BODY_SUPERSET_CONFIG_SCHEMA = "vlsa_distal_whole_body_superset_canary.v1"
WHOLE_BODY_PROSPECTIVE_CONFIG_SCHEMA = "vlsa_distal_whole_body_prospective_population.v1"
WHOLE_BODY_EXTENSION_CONFIG_SCHEMA = "vlsa_distal_whole_body_progressive_extension.v1"
TARGETED_PI05_CONFIG_SCHEMA = "vlsa_distal_targeted_pi05_palm_l6_extension.v1"
CASE_SCHEMA = "vlsa_distal_exact_group_boundary_case_result.v1"
VALIDATION_SCHEMA = "vlsa_distal_exact_group_boundary_validation.v1"
GROUPS = ("palm", "L5", "L6")
WHOLE_BODY_GROUPS = ("end_effector", "palm", "L5", "L6", "L7")


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def payload_sha256(value: Mapping[str, Any], key: str = "result_payload_sha256") -> str:
    public = dict(value)
    public.pop(key, None)
    return hashlib.sha256(canonical(public)).hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    value = json.loads(raw)
    schema = value.get("schema_version")
    if schema == TARGETED_PI05_CONFIG_SCHEMA:
        required = {
            "schema_version", "protocol_id", "claim_scope", "base_config",
            "base_config_file_sha256", "source_audit", "selection_manifest",
            "selection_manifest_file_sha256", "target_group_sequence",
            "state_selection", "gate", "training_authorization_mode",
            "learned_correction_QP_enabled", "forbidden",
        }
        allowed = required | {"progressive_historical_source_commits"}
        if not required.issubset(value) or set(value) - allowed:
            raise ValueError("targeted pi05 exact-group config keys differ")
        base_path = Path(path).resolve().parents[1] / str(value["base_config"])
        base_raw = base_path.read_bytes()
        if hashlib.sha256(base_raw).hexdigest() != value["base_config_file_sha256"]:
            raise ValueError("targeted pi05 base config differs")
        base = json.loads(base_raw)
        if base.get("schema_version") != WHOLE_BODY_EXTENSION_CONFIG_SCHEMA:
            raise ValueError("targeted pi05 base schema differs")
        expanded = dict(base)
        for key in (
            "schema_version", "protocol_id", "claim_scope",
            "selection_manifest", "selection_manifest_file_sha256",
            "target_group_sequence", "state_selection", "gate",
            "training_authorization_mode", "learned_correction_QP_enabled",
            "forbidden",
        ):
            expanded[key] = value[key]
        expanded["base_config_binding"] = {
            "path": value["base_config"],
            "file_sha256": value["base_config_file_sha256"],
        }
        expanded["source_audit"] = value["source_audit"]
        if value.get("progressive_historical_source_commits") is not None:
            historical = value["progressive_historical_source_commits"]
            if (
                not isinstance(historical, dict)
                or not all(
                    isinstance(case_id, str)
                    and isinstance(commit, str)
                    and len(commit) == 40
                    for case_id, commit in historical.items()
                )
            ):
                raise ValueError("targeted pi05 historical source commits differ")
            expanded["progressive_historical_source_commits"] = historical
        value = expanded
    if schema not in (
        CONFIG_SCHEMA, NO_QP_L5_CONFIG_SCHEMA, GENERIC_L5_CONFIG_SCHEMA,
        TRAJECTORY_VALUE_CONFIG_SCHEMA, PROSPECTIVE_L5_CONFIG_SCHEMA,
        DEVELOPMENT_L5_CONFIG_SCHEMA, DEVELOPMENT_EXCITATION_CONFIG_SCHEMA,
        WHOLE_BODY_SUPERSET_CONFIG_SCHEMA, WHOLE_BODY_PROSPECTIVE_CONFIG_SCHEMA,
        WHOLE_BODY_EXTENSION_CONFIG_SCHEMA,
        TARGETED_PI05_CONFIG_SCHEMA,
    ):
        raise ValueError("exact-group boundary config schema differs")
    expected_protocol = {
        CONFIG_SCHEMA: "vlsa-distal-exact-group-boundary-canary-v1",
        NO_QP_L5_CONFIG_SCHEMA: "vlsa-distal-no-qp-l5-boundary-canary-v1",
        GENERIC_L5_CONFIG_SCHEMA: "vlsa-distal-generic-l5-boundary-canary-v1",
        TRAJECTORY_VALUE_CONFIG_SCHEMA: "vlsa-distal-generic-l5-trajectory-value-canary-v1",
        PROSPECTIVE_L5_CONFIG_SCHEMA: "vlsa-distal-prospective-l5-boundary-population-v1",
        DEVELOPMENT_L5_CONFIG_SCHEMA: "vlsa-distal-exact-group-boundary-development-search-v1",
        DEVELOPMENT_EXCITATION_CONFIG_SCHEMA: "vlsa-distal-exact-group-boundary-development-excitation-v1",
        WHOLE_BODY_SUPERSET_CONFIG_SCHEMA: "vlsa-distal-whole-body-superset-canary-v1",
        WHOLE_BODY_PROSPECTIVE_CONFIG_SCHEMA: "vlsa-distal-whole-body-prospective-population-v1",
        WHOLE_BODY_EXTENSION_CONFIG_SCHEMA: "vlsa-distal-whole-body-progressive-extension-v1",
        TARGETED_PI05_CONFIG_SCHEMA: "vlsa-distal-targeted-pi05-palm-l6-extension-v1",
    }[schema]
    if value.get("protocol_id") != expected_protocol:
        raise ValueError("exact-group boundary protocol differs")
    bank = value["candidate_bank"]
    if schema in (
        DEVELOPMENT_EXCITATION_CONFIG_SCHEMA, WHOLE_BODY_SUPERSET_CONFIG_SCHEMA,
        WHOLE_BODY_PROSPECTIVE_CONFIG_SCHEMA,
    ):
        expected_grid = {
            "kind": "local_frame_grid",
            "spatial_basis": ["normal", "tangent_up", "tangent_side"],
            "coefficient_grid": [-1, 0, 1],
            "exclude_all_zero": True,
            "spatial_direction_count": 26,
            "correction_l2_action": 2.0,
            "action_limit": 1.0,
            "include_nominal": True,
            "candidate_count_per_job": 27,
            "endpoint_preservation": False,
            "preserve_rotation_and_gripper": True,
            "temporal_profile": "front_loaded",
            "released_AEGIS_EE_applied_to_every_candidate": False,
            "candidate_execution": "direct_clipped_Cartesian_VLA_action_to_unchanged_OSC",
            "selection_cost": "direct_five_action_L2_from_nominal_VLA_chunk",
        }
        if bank != expected_grid:
            raise ValueError("development excitation candidate bank differs")
    elif schema in (
        WHOLE_BODY_EXTENSION_CONFIG_SCHEMA, TARGETED_PI05_CONFIG_SCHEMA,
    ):
        names = list(bank.get("selected_candidate_names", []))
        if (
            bank.get("kind") != "frozen_local_frame_grid_subset"
            or bank.get("spatial_basis") != ["normal", "tangent_up", "tangent_side"]
            or bank.get("coefficient_grid") != [-1, 0, 1]
            or int(bank.get("spatial_direction_count", 0)) != 12
            or int(bank.get("candidate_count_per_job", 0)) != 13
            or len(names) != 13 or names[0] != "nominal"
            or len(set(names)) != 13
        ):
            raise ValueError("whole-body extension candidate bank differs")
    elif schema in (
        GENERIC_L5_CONFIG_SCHEMA, TRAJECTORY_VALUE_CONFIG_SCHEMA,
        PROSPECTIVE_L5_CONFIG_SCHEMA, DEVELOPMENT_L5_CONFIG_SCHEMA,
    ):
        if [float(item) for item in bank["radii"]] != [0.5, 1.5]:
            raise ValueError("generic exact-group boundary radii differ")
        if bank["axis_order"] != ["x", "y", "z"]:
            raise ValueError("generic exact-group boundary axes differ")
        if bank["sign_order"] != [-1, 1]:
            raise ValueError("generic exact-group boundary signs differ")
    elif [float(item) for item in bank["requested_alpha"]] != [
        0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0,
    ]:
        raise ValueError("exact-group boundary alpha bank differs")
    expected_aegis = schema == CONFIG_SCHEMA
    if bank["released_AEGIS_EE_applied_to_every_candidate"] is not expected_aegis:
        raise ValueError("released AEGIS EE-QP execution mode differs")
    if value["learned_correction_QP_enabled"] is not False:
        raise ValueError("learned correction QP must remain disabled")
    expected_group_order = (
        list(WHOLE_BODY_GROUPS)
        if schema in (
            WHOLE_BODY_SUPERSET_CONFIG_SCHEMA,
            WHOLE_BODY_PROSPECTIVE_CONFIG_SCHEMA,
            WHOLE_BODY_EXTENSION_CONFIG_SCHEMA,
            TARGETED_PI05_CONFIG_SCHEMA,
        )
        else list(GROUPS)
    )
    if value["exact_group_target"]["group_order"] != expected_group_order:
        raise ValueError("exact-group order differs")
    if schema in (
        WHOLE_BODY_SUPERSET_CONFIG_SCHEMA,
        WHOLE_BODY_PROSPECTIVE_CONFIG_SCHEMA,
        WHOLE_BODY_EXTENSION_CONFIG_SCHEMA,
        TARGETED_PI05_CONFIG_SCHEMA,
    ):
        exact = value["exact_group_target"]
        if (
            exact.get("include_released_aegis_end_effector_proxy") is not True
            or int(exact.get("distal_row_count", 0)) != 7
            or exact.get("robot_rows") != {
                "end_effector": [0], "palm": [1],
                "L5": [2, 3, 4], "L6": [5, 6], "L7": [7, 8],
            }
            or exact.get("groups", {}).get("end_effector") != []
        ):
            raise ValueError("whole-body superset geometry contract differs")
        trace = value.get("artifact_superset", {})
        if (
            trace.get("capture_internal_substep_ee_pose") is not True
            or trace.get("capture_internal_substep_palm_pose") is not True
            or trace.get("capture_action_boundary_context") is not True
            or trace.get("store_all_group_risks") != list(WHOLE_BODY_GROUPS)
        ):
            raise ValueError("whole-body superset trace contract differs")
    if value.get("trajectory_policy_value") is not None:
        trajectory = value["trajectory_policy_value"]
        if trajectory.get("capture_action_boundaries") is not True:
            raise ValueError("trajectory-value action-boundary capture differs")
        if trajectory.get("value_training_phases") != ["backup", "terminal_hold"]:
            raise ValueError("trajectory-value fixed-policy phases differ")
        if trajectory.get("internal_substeps") != "label_authority_only":
            raise ValueError("trajectory-value substep sampling differs")
    output = json.loads(canonical(value).decode("utf-8"))
    output["config_file_sha256"] = hashlib.sha256(raw).hexdigest()
    output["config_payload_sha256"] = hashlib.sha256(canonical(value)).hexdigest()
    return output


def load_cases(path: Path, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    cases = [json.loads(line) for line in Path(path).read_text().splitlines() if line]
    if len(cases) != int(config["gate"]["required_case_count"]):
        raise ValueError("exact-group boundary case count differs")
    expected_groups = list(config.get("target_group_sequence", GROUPS))
    if [case["target_group"] for case in cases] != expected_groups:
        raise ValueError("exact-group boundary target cases differ")
    grouping_key = (
        "episode_group_id"
        if config["schema_version"] in (
            PROSPECTIVE_L5_CONFIG_SCHEMA, DEVELOPMENT_L5_CONFIG_SCHEMA,
            DEVELOPMENT_EXCITATION_CONFIG_SCHEMA,
            WHOLE_BODY_SUPERSET_CONFIG_SCHEMA,
            WHOLE_BODY_PROSPECTIVE_CONFIG_SCHEMA,
            WHOLE_BODY_EXTENSION_CONFIG_SCHEMA,
            TARGETED_PI05_CONFIG_SCHEMA,
        )
        else "task_level_group_id"
    )
    if len({case[grouping_key] for case in cases}) != len(cases):
        raise ValueError("exact-group boundary episode groups are not independent")
    if config["schema_version"] in (
        PROSPECTIVE_L5_CONFIG_SCHEMA,
        WHOLE_BODY_PROSPECTIVE_CONFIG_SCHEMA,
        WHOLE_BODY_EXTENSION_CONFIG_SCHEMA,
        TARGETED_PI05_CONFIG_SCHEMA,
    ):
        if not isinstance(
            config["state_selection"].get("require_archived_task_success", True),
            bool,
        ):
            raise ValueError("prospective archived task-success precondition differs")
        required = config["gate"]["required_split_case_count"]
        observed = {
            split: sum(case["split"] == split for case in cases)
            for split in ("train", "validation", "test")
        }
        if observed != {key: int(value) for key, value in required.items()}:
            raise ValueError("prospective exact-group split counts differ")
        if not all(
            case.get("prospective_split_frozen_before_candidate_outcomes") is True
            for case in cases
        ):
            raise ValueError("prospective exact-group split was not frozen")
    return cases


def warning_step(case: Mapping[str, Any], config: Mapping[str, Any]) -> int:
    if config.get("schema_version") in (
        GENERIC_L5_CONFIG_SCHEMA, TRAJECTORY_VALUE_CONFIG_SCHEMA,
        PROSPECTIVE_L5_CONFIG_SCHEMA, DEVELOPMENT_L5_CONFIG_SCHEMA,
        DEVELOPMENT_EXCITATION_CONFIG_SCHEMA,
        WHOLE_BODY_SUPERSET_CONFIG_SCHEMA,
        WHOLE_BODY_PROSPECTIVE_CONFIG_SCHEMA,
        WHOLE_BODY_EXTENSION_CONFIG_SCHEMA,
        TARGETED_PI05_CONFIG_SCHEMA,
    ):
        step = int(case["state_step"])
        if step < 0 or step % 5:
            raise ValueError("generic exact-group state step differs")
        return step
    lead = int(config["state_selection"]["lead_actions_before_first_target_contact"])
    step = int(math.floor((int(case["first_target_contact_step"]) - lead) / 5.0) * 5)
    if step < 0 or step % 5:
        raise ValueError("exact-group warning step differs")
    return step


def summarize_cases(cases: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    near = float(config["gate"]["near_boundary_abs_slack"])
    summaries = []
    proxy_false_safe = 0
    replay_pass = True
    source_state_hash_pass = True
    trajectory_mode = config.get("trajectory_policy_value") is not None
    trajectory_candidate_count = 0
    trajectory_action_boundary_count = 0
    trajectory_eligible_value_state_count = 0
    trajectory_maximum_bellman_residual = 0.0
    trajectory_context_complete = True
    artifact_superset_mode = (
        config.get("schema_version") in (
            WHOLE_BODY_SUPERSET_CONFIG_SCHEMA,
            WHOLE_BODY_PROSPECTIVE_CONFIG_SCHEMA,
            WHOLE_BODY_EXTENSION_CONFIG_SCHEMA,
            TARGETED_PI05_CONFIG_SCHEMA,
        )
    )
    artifact_superset_complete = True
    retained_rejections = []
    successful_cases = []
    for case in cases:
        target = str(case["selection"]["target_group"])
        if case.get("status", "complete") != "complete":
            rejection = case.get("rejection") or {}
            if (
                case.get("status") != "retained_scientific_rejection_initial_CAR"
                or rejection.get("code")
                != "initial_active_obstacle_CAR_exceeds_registered_limit"
                or rejection.get("candidate_outcomes_observed") is not False
                or rejection.get("retained") is not True
            ):
                raise ValueError("exact-group retained rejection differs")
            retained_rejections.append({
                "case_id": case["case_id"],
                "split": str(case["selection"].get("split", "development")),
                "target_group": target,
                "status": case["status"],
                "code": rejection["code"],
                "reason": rejection["reason"],
            })
            replay_pass = False
            source_state_hash_pass = False
            artifact_superset_complete = False
            trajectory_context_complete = False
            summaries.append({
                "case_id": case["case_id"],
                "split": str(case["selection"].get("split", "development")),
                "target_group": target,
                "status": case["status"],
                "candidate_count": 0,
                "known_candidate_count": 0,
                "unknown_timeout_count": 0,
                "target_safe_candidate_count": 0,
                "target_unsafe_candidate_count": 0,
                "target_near_boundary_candidate_count": 0,
                "target_two_sided_support": False,
                "initial_target_slack": None,
                "initial_target_contact_count": None,
                "initial_physical_groups_safe": False,
                "robot_primitive_certificate_pass": False,
                "physical_false_safe_count": 0,
                "retained_rejection": rejection["code"],
            })
            continue
        successful_cases.append(case)
        initial = case["exact_case"]["exact_group_target"]
        candidates = case["exact_case"]["candidates"]
        known = [
            row for row in candidates
            if bool(row["exact_group_target"]["known_outcome"])
        ]
        safe = [
            row for row in known
            if float(row["exact_group_target"]["group_future_violation"][target]) <= 0.0
            and int(row["exact_group_target"]["group_contact_sample_count"][target]) == 0
        ]
        unsafe = [row for row in known if row not in safe]
        boundary = [
            row for row in known
            if abs(float(row["exact_group_target"]["group_minimum_normalized_radial_slack"][target])) <= near
        ]
        configured_groups = tuple(config["exact_group_target"]["group_order"])
        false_safe = sum(
            int(row["exact_group_target"]["group_contact_sample_count"][group]) > 0
            and float(row["exact_group_target"]["group_minimum_normalized_radial_slack"][group]) > 0.0
            for row in known for group in configured_groups
        )
        proxy_false_safe += false_safe
        replay_pass = replay_pass and bool(case["exact_case"]["source_replay_exact"])
        source_state_hash_pass = source_state_hash_pass and bool(
            case["exact_case"]["state_hash_matches"]
        )
        if artifact_superset_mode:
            exact_case = case["exact_case"]
            configured = set(config["exact_group_target"]["group_order"])
            artifact_superset_complete = bool(
                artifact_superset_complete
                and exact_case.get("artifact_superset_contract")
                == config["artifact_superset"]
                and len(exact_case.get("initial_exact_robot_rows") or []) == 9
                and exact_case.get("initial_ee_pose") is not None
                and all(
                    set(row["exact_group_target"]["group_future_violation"])
                    == configured
                    and all(
                        sample.get("ee_pose") is not None
                        and sample.get("palm_pose") is not None
                        and set(sample["group_normalized_radial_slack"])
                        == configured
                        for sample in row["exact_group_target"]["trace"]
                    )
                    for row in candidates
                )
            )
        if trajectory_mode:
            required_context = {
                "dynamic_state", "arm_joint_position_rad",
                "arm_joint_velocity_rad_s", "eef_position_m",
                "eef_quaternion_xyzw", "controller_snapshot",
                "exact_robot_rows", "compiled_obstacle_boxes",
                "group_normalized_radial_slack", "executed_action",
            }
            if artifact_superset_mode:
                required_context.add("ee_pose")
            for candidate in candidates:
                exact = candidate["exact_group_target"]
                boundaries = exact.get("action_boundaries", [])
                trajectory = exact.get("trajectory_policy_value", {})
                records = trajectory.get("records", [])
                trajectory_candidate_count += 1
                trajectory_action_boundary_count += len(boundaries)
                trajectory_eligible_value_state_count += sum(
                    bool(record.get("training_sample_eligible"))
                    for record in records
                )
                trajectory_maximum_bellman_residual = max(
                    trajectory_maximum_bellman_residual,
                    float(trajectory.get("maximum_bellman_residual", math.inf)),
                )
                trajectory_context_complete = bool(
                    trajectory_context_complete
                    and len(boundaries) == int(candidate["action_count"])
                    and len(records) == len(boundaries)
                    and all(required_context.issubset(boundary) for boundary in boundaries)
                    and all(
                        bool(record.get("training_sample_eligible"))
                        == bool(
                            exact["known_outcome"]
                            and record["phase"] in config[
                                "trajectory_policy_value"
                            ]["value_training_phases"]
                        )
                        for record in records
                    )
                )
        summaries.append({
            "case_id": case["case_id"],
            "split": str(case["selection"].get("split", "development")),
            "target_group": target,
            "candidate_count": len(candidates),
            "known_candidate_count": len(known),
            "unknown_timeout_count": len(candidates) - len(known),
            "target_safe_candidate_count": len(safe),
            "target_unsafe_candidate_count": len(unsafe),
            "target_near_boundary_candidate_count": len(boundary),
            "target_two_sided_support": bool(safe and unsafe),
            "initial_target_slack": float(initial["initial_group_normalized_radial_slack"][target]),
            "initial_target_contact_count": int(initial["initial_group_contact_sample_count"][target]),
            "initial_physical_groups_safe": (
                all(
                    float(initial["initial_group_normalized_radial_slack"][group]) > 0.0
                    and int(initial["initial_group_contact_sample_count"][group]) == 0
                    for group in ("palm", "L5", "L6", "L7")
                )
                if config.get("schema_version") in (
                    WHOLE_BODY_EXTENSION_CONFIG_SCHEMA, TARGETED_PI05_CONFIG_SCHEMA,
                )
                else True
            ),
            "robot_primitive_certificate_pass": bool(initial["robot_primitive_certificate_pass"]),
            "physical_false_safe_count": false_safe,
        })
    replay_gate = (
        source_state_hash_pass
        if config["gate"].get("legacy_source_proxy_replay") == "diagnostic_only"
        else replay_pass
    )
    trajectory_apparatus_pass = bool(
        not retained_rejections
        and (
        not trajectory_mode
        or (
            trajectory_candidate_count == sum(
                len(case["exact_case"]["candidates"]) for case in successful_cases
            )
            and trajectory_context_complete
            and trajectory_maximum_bellman_residual <= float(
                config["gate"]["required_maximum_bellman_residual"]
            )
            and trajectory_eligible_value_state_count > 0
        )
        )
    )
    apparatus = bool(
        not retained_rejections
        and
        len(cases) == int(config["gate"]["required_case_count"])
        and all(item["candidate_count"] == int(config["gate"]["required_candidates_per_case"])
                for item in summaries)
        and all(item["robot_primitive_certificate_pass"] for item in summaries)
        and all(item["initial_target_slack"] > 0.0 and item["initial_target_contact_count"] == 0
                for item in summaries)
        and (
            config.get("schema_version") not in (
                WHOLE_BODY_EXTENSION_CONFIG_SCHEMA, TARGETED_PI05_CONFIG_SCHEMA,
            )
            or all(item["initial_physical_groups_safe"] for item in summaries)
        )
        and replay_gate and proxy_false_safe == 0
        and trajectory_apparatus_pass
        and (not artifact_superset_mode or artifact_superset_complete)
    )
    same_bank = bool(apparatus and all(item["target_two_sided_support"] for item in summaries))
    two_sided_case_count = sum(
        bool(item["target_two_sided_support"]) for item in summaries
    )
    targeted_required = int(config["gate"].get(
        "required_two_sided_case_count", len(summaries),
    ))
    targeted_coverage = bool(
        apparatus and two_sided_case_count >= targeted_required
    )
    prospective_mode = (
        config.get("schema_version") in (
            PROSPECTIVE_L5_CONFIG_SCHEMA,
            WHOLE_BODY_PROSPECTIVE_CONFIG_SCHEMA,
            WHOLE_BODY_EXTENSION_CONFIG_SCHEMA,
            TARGETED_PI05_CONFIG_SCHEMA,
        )
    )
    per_split: dict[str, Any] = {}
    prospective_gate = False
    if prospective_mode:
        required_two_sided = config["gate"]["required_two_sided_by_split"]
        required_initially_safe = config["gate"]["required_initially_safe_by_split"]
        for split in ("train", "validation", "test"):
            rows = [item for item in summaries if item["split"] == split]
            per_split[split] = {
                "case_count": len(rows),
                "initially_safe_case_count": sum(
                    (
                        bool(item["initial_physical_groups_safe"])
                        if config.get("schema_version") in (
                            WHOLE_BODY_EXTENSION_CONFIG_SCHEMA,
                            TARGETED_PI05_CONFIG_SCHEMA,
                        )
                        else (
                            item["initial_target_slack"] is not None
                            and item["initial_target_slack"] > 0.0
                            and item["initial_target_contact_count"] == 0
                        )
                    )
                    for item in rows
                ),
                "two_sided_case_count": sum(
                    bool(item["target_two_sided_support"]) for item in rows
                ),
                "known_candidate_count": sum(
                    int(item["known_candidate_count"]) for item in rows
                ),
                "safe_candidate_count": sum(
                    int(item["target_safe_candidate_count"]) for item in rows
                ),
                "unsafe_candidate_count": sum(
                    int(item["target_unsafe_candidate_count"]) for item in rows
                ),
                "unknown_timeout_count": sum(
                    int(item["unknown_timeout_count"]) for item in rows
                ),
            }
        prospective_gate = bool(
            apparatus
            and proxy_false_safe <= int(config["gate"][
                "required_maximum_physical_false_safe_count"
            ])
            and all(
                per_split[split]["initially_safe_case_count"]
                >= int(required_initially_safe[split])
                and per_split[split]["two_sided_case_count"]
                >= int(required_two_sided[split])
                for split in ("train", "validation", "test")
            )
        )
        if config.get("training_authorization_mode") == "combined_24_state_external_audit_only":
            prospective_gate = False
    return {
        "case_count": len(cases),
        "candidate_count": sum(item["candidate_count"] for item in summaries),
        "source_replay_exact": replay_pass,
        "source_state_hash_exact": source_state_hash_pass,
        "legacy_source_proxy_replay": config["gate"].get(
            "legacy_source_proxy_replay", "strict"
        ),
        "physical_false_safe_count": proxy_false_safe,
        "trajectory_policy_value_enabled": trajectory_mode,
        "trajectory_policy_value_apparatus_pass": trajectory_apparatus_pass,
        "trajectory_candidate_count": trajectory_candidate_count,
        "trajectory_action_boundary_count": trajectory_action_boundary_count,
        "trajectory_eligible_value_state_count": trajectory_eligible_value_state_count,
        "trajectory_context_complete": trajectory_context_complete,
        "trajectory_maximum_bellman_residual": trajectory_maximum_bellman_residual,
        "artifact_superset_enabled": artifact_superset_mode,
        "artifact_superset_complete": artifact_superset_complete,
        "retained_scientific_rejection_count": len(retained_rejections),
        "retained_scientific_rejections": retained_rejections,
        "two_sided_case_count": two_sided_case_count,
        "required_two_sided_case_count": targeted_required,
        "per_case": summaries,
        "prospective_split_gate_enabled": prospective_mode,
        "prospective_split_summary": per_split,
        "q_only_prediction_gate_authorized": prospective_gate,
        "apparatus_pass": apparatus,
        "targeted_coverage_canary_pass": targeted_coverage,
        "same_bank_grouped_collection_authorized": same_bank,
        "training_authorized": prospective_gate,
        "QP_authorized": False,
        "closed_loop_authorized": False,
    }
