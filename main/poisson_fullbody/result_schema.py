"""Semantic validation for final Poisson feasibility episode results."""

import math
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Optional, Sequence

from main.poisson_fullbody.contracts import (
    ArtifactContractError,
    attach_payload_hash,
    canonical_json_bytes,
    publish_hashed_json,
    require_sha256,
    sha256_bytes,
    validate_artifact_reference,
    verify_payload_hash,
)


RESULT_SCHEMA_VERSION = "vlsa_poisson_link56_episode_result.v2"
PROTOCOL_ID = "vlsa-poisson-link56-aegis-car-109-v1"
CODE_REPOSITORY = "quanth238/vlsa-aegis"
BASELINE_REPOSITORY = "THU-RCSCT/vlsa-aegis"
BASELINE_COMMIT = "1592aa59361f431ba96c6ddcbebcb596f6c20853"
COMPLETION_CLASSES = {
    "executed",
    "safe_start_inadmissible",
    "static_obstacle_inadmissible",
    "field_invalid",
    "barrier_invariance_lost",
    "qp_infeasible",
    "qp_solver_failure",
    "controller_tracking_invalid",
    "fail_closed_runtime",
}
ARMS = {
    "joint_velocity_adapter_only",
    "joint_velocity_psf_eef_only",
    "joint_velocity_psf_link56",
    "joint_velocity_psf_all_moving_links",
}
SUITES = {
    "safelibero_spatial",
    "safelibero_goal",
    "safelibero_object",
    "safelibero_long",
}
MEASUREMENT_SCHEMA_VERSION = "vlsa_poisson_simulator_measurement.v1"
D_SIM_SEMANTICS = {
    "contact_authority_plus_exact_primitive_clearance",
    "contact_authority_plus_coverage_lower_bound",
    "contact_only_positive_clearance_unavailable",
    "union_of_settled_live_solver_and_forwarded_post_state_nonpositive_contacts_plus_exact_obb_coverage_lower_bound",
}
TOP_LEVEL_KEYS = {
    "schema_version",
    "result_payload_sha256",
    "status",
    "scientific_result",
    "completion_class",
    "run_id",
    "protocol_id",
    "case_id",
    "arm",
    "provenance",
    "case_identity",
    "seeds",
    "pairing",
    "runtime",
    "optimizer",
    "allocation",
    "execution",
    "endpoints",
    "artifact_references",
}


def _error(label: str, message: str) -> None:
    raise ArtifactContractError("%s %s" % (label, message))


def _object(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        _error(label, "must be an object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: Sequence[str], label: str) -> None:
    expected_set = set(expected)
    missing = expected_set.difference(value)
    extra = set(value).difference(expected_set)
    if missing:
        _error(label, "is missing required fields: %s" % ", ".join(sorted(missing)))
    if extra:
        _error(label, "contains unsupported fields: %s" % ", ".join(sorted(extra)))


def _bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        _error(label, "must be Boolean")
    return value


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        _error(label, "must be a non-empty string")
    return value


def _enum(value: Any, allowed: Sequence[str], label: str) -> str:
    result = _string(value, label)
    if result not in set(allowed):
        _error(label, "is unsupported")
    return result


def _nullable_string(value: Any, label: str) -> Optional[str]:
    if value is None:
        return None
    return _string(value, label)


def _digit_string(value: Any, label: str) -> str:
    result = _string(value, label)
    if not result.isdigit():
        _error(label, "must contain decimal digits only")
    return result


def _relative_posix_path(value: Any, label: str) -> str:
    result = _string(value, label)
    path = PurePosixPath(result)
    if (
        path.is_absolute()
        or str(path) != result
        or any(part in ("", ".", "..") for part in path.parts)
    ):
        _error(label, "must be a normalized relative POSIX path")
    return result


def _string_list(
    value: Any,
    label: str,
    *,
    minimum: int = 0,
    unique: bool = True,
) -> Sequence[str]:
    if not isinstance(value, list) or len(value) < minimum:
        _error(label, "must be a list with at least %d entries" % minimum)
    result = [_string(item, "%s[%d]" % (label, index)) for index, item in enumerate(value)]
    if unique and len(set(result)) != len(result):
        _error(label, "must not contain duplicates")
    return result


def _integer(value: Any, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _error(label, "must be an integer >= %d" % minimum)
    return value


def _git_oid(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) not in (40, 64)
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _error(label, "must be a lowercase Git object ID")
    return value


def _number(value: Any, label: str, *, minimum: Optional[float] = None) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        _error(label, "must be a finite number")
    result = float(value)
    if minimum is not None and result < minimum:
        _error(label, "must be >= %s" % minimum)
    return result


def _reject_nonfinite(value: Any, label: str = "result") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        _error(label, "contains a non-finite number")
    if isinstance(value, dict):
        for key, item in value.items():
            _reject_nonfinite(item, "%s.%s" % (label, key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_nonfinite(item, "%s[%d]" % (label, index))


def _measurement(
    value: Any,
    label: str,
    *,
    units: str,
    allow_negative: bool,
) -> Optional[float]:
    record = _object(value, label)
    _exact_keys(record, ("available", "value", "units", "reason"), label)
    available = _bool(record.get("available"), label + ".available")
    if record.get("units") != units:
        _error(label + ".units", "must equal %r" % units)
    if available:
        result = _number(record.get("value"), label + ".value")
        if not allow_negative and result < 0.0:
            _error(label + ".value", "must be nonnegative")
        if record.get("reason") is not None:
            _error(label + ".reason", "must be null when available")
        return result
    if record.get("value") is not None:
        _error(label + ".value", "must be null when unavailable")
    _string(record.get("reason"), label + ".reason")
    return None


_CONTACT_RECORD_KEYS = (
    "source_phase",
    "observation_index",
    "high_level_index",
    "inner_control_index",
    "physics_substep_index",
    "mujoco_contact_index",
    "is_physical_nonpositive_distance_contact",
    "within_registered_near_contact_tolerance",
    "solver_constraint_active",
    "efc_address",
    "mujoco_geom1_id",
    "mujoco_geom1_name",
    "mujoco_geom2_id",
    "mujoco_geom2_name",
    "robot_geom_id",
    "robot_geom_name",
    "robot_body_id",
    "robot_body_name",
    "obstacle_geom_id",
    "obstacle_geom_name",
    "obstacle_body_id",
    "obstacle_body_name",
    "contact_distance_m",
    "contact_includemargin_m",
    "robot_geom_margin_m",
    "robot_geom_gap_m",
    "obstacle_geom_margin_m",
    "obstacle_geom_gap_m",
    "explicit_pair_id",
    "explicit_pair_margin_m",
    "explicit_pair_gap_m",
    "position_world_m",
    "frame_normal_mujoco_geom1_to_geom2_world",
    "force_available",
    "force_semantics",
    "force_contact_frame_n",
    "force_world_n",
    "normal_force_n",
    "impulse_estimate_contact_frame_ns",
    "impulse_estimate_world_ns",
    "force_unavailable_reason",
)


def _contact_record(value: Any, label: str) -> Mapping[str, Any]:
    record = _object(value, label)
    _exact_keys(record, _CONTACT_RECORD_KEYS, label)
    phase = _enum(
        record.get("source_phase"),
        (
            "settled_post_integration_recomputed",
            "live_solver_phase_preintegration_geometry",
            "post_integration_recomputed",
        ),
        label + ".source_phase",
    )
    if not _bool(
        record.get("is_physical_nonpositive_distance_contact"),
        label + ".is_physical_nonpositive_distance_contact",
    ):
        _error(label, "must bind to an observed physical nonpositive contact")
    if _number(record.get("contact_distance_m"), label + ".contact_distance_m") > 0.0:
        _error(label + ".contact_distance_m", "must be nonpositive")
    for field in (
        "mujoco_contact_index",
        "mujoco_geom1_id",
        "mujoco_geom2_id",
        "robot_geom_id",
        "robot_body_id",
        "obstacle_geom_id",
        "obstacle_body_id",
    ):
        _integer(record.get(field), label + "." + field)
    for field in (
        "mujoco_geom1_name",
        "mujoco_geom2_name",
        "robot_geom_name",
        "robot_body_name",
        "obstacle_geom_name",
        "obstacle_body_name",
        "force_semantics",
    ):
        _string(record.get(field), label + "." + field)
    for field in (
        "contact_includemargin_m",
        "robot_geom_margin_m",
        "robot_geom_gap_m",
        "obstacle_geom_margin_m",
        "obstacle_geom_gap_m",
    ):
        _number(record.get(field), label + "." + field)
    for field in (
        "within_registered_near_contact_tolerance",
        "solver_constraint_active",
        "force_available",
    ):
        _bool(record.get(field), label + "." + field)
    for field in ("position_world_m", "frame_normal_mujoco_geom1_to_geom2_world"):
        vector = record.get(field)
        if not isinstance(vector, (list, tuple)) or len(vector) != 3:
            _error(label + "." + field, "must be a length-three vector")
        for index, component in enumerate(vector):
            _number(component, "%s.%s[%d]" % (label, field, index))
    if phase == "settled_post_integration_recomputed":
        for field in (
            "observation_index",
            "high_level_index",
            "inner_control_index",
            "physics_substep_index",
        ):
            if record.get(field) is not None:
                _error(label + "." + field, "must be null for settled contact")
    else:
        for field in (
            "observation_index",
            "high_level_index",
            "inner_control_index",
            "physics_substep_index",
        ):
            _integer(record.get(field), label + "." + field)
    return record


def validate_episode_result(result: Mapping[str, Any]) -> None:
    """Validate units, denominators, contact truth, and fail-closed semantics."""

    if not isinstance(result, dict):
        _error("result", "must be an object")
    _reject_nonfinite(result)
    _exact_keys(result, TOP_LEVEL_KEYS, "result")
    if result.get("schema_version") != RESULT_SCHEMA_VERSION:
        _error("schema_version", "is unsupported")
    verify_payload_hash(result)
    if result.get("status") != "complete" or result.get("scientific_result") is not True:
        _error("status", "must be a schema-valid complete scientific outcome")
    completion = result.get("completion_class")
    if completion not in COMPLETION_CLASSES:
        _error("completion_class", "is unsupported")
    arm = result.get("arm")
    if arm not in ARMS:
        _error("arm", "is unsupported")
    _string(result.get("run_id"), "run_id")
    if result.get("protocol_id") != PROTOCOL_ID:
        _error("protocol_id", "must equal %r" % PROTOCOL_ID)
    case_id = _string(result.get("case_id"), "case_id")

    provenance = _object(result.get("provenance"), "provenance")
    _exact_keys(
        provenance,
        (
            "code_repository",
            "code_commit",
            "code_dirty",
            "code_dirty_state_sha256",
            "code_dirty_state_semantics",
            "baseline_repository",
            "baseline_commit",
            "run_contract_sha256",
            "manifest_sha256",
            "manifest_record_sha256",
            "protocol_config_sha256",
            "runtime_protocol_raw_sha256",
            "runtime_protocol_semantic_sha256",
        ),
        "provenance",
    )
    if provenance.get("code_repository") != CODE_REPOSITORY:
        _error("provenance.code_repository", "must equal %r" % CODE_REPOSITORY)
    if provenance.get("baseline_repository") != BASELINE_REPOSITORY:
        _error(
            "provenance.baseline_repository",
            "must equal %r" % BASELINE_REPOSITORY,
        )
    for field in (
        "code_dirty_state_sha256",
        "run_contract_sha256",
        "manifest_sha256",
        "manifest_record_sha256",
        "protocol_config_sha256",
        "runtime_protocol_raw_sha256",
        "runtime_protocol_semantic_sha256",
    ):
        require_sha256(provenance.get(field), "provenance.%s" % field)
    _git_oid(provenance.get("code_commit"), "provenance.code_commit")
    if provenance.get("baseline_commit") != BASELINE_COMMIT:
        _error(
            "provenance.baseline_commit", "must equal the registered Table-1 base"
        )
    if _bool(provenance.get("code_dirty"), "provenance.code_dirty"):
        _error("provenance.code_dirty", "must be false for a final scientific result")
    if provenance.get("code_dirty_state_sha256") != sha256_bytes(b""):
        _error(
            "provenance.code_dirty_state_sha256",
            "must equal SHA-256(empty bytes) for the required clean worktree",
        )
    if (
        provenance.get("code_dirty_state_semantics")
        != "sha256_empty_bytes_for_required_clean_git_worktree_v1"
    ):
        _error(
            "provenance.code_dirty_state_semantics",
            "must name the registered clean-worktree identity",
        )

    identity = _object(result.get("case_identity"), "case_identity")
    _exact_keys(
        identity,
        (
            "dataset",
            "manifest_case_id",
            "manifest_case_ordinal",
            "pair_group_id",
            "suite",
            "safety_level",
            "logical_task_index",
            "resolved_task_index",
            "episode_index",
            "task_name",
            "task_family_id",
            "task_level_group_id",
            "bddl_relative_path",
            "bddl_sha256",
            "initial_states_relative_path",
            "initial_states_sha256",
            "initial_state_record_sha256",
            "active_obstacle_name",
            "split",
            "stratum",
            "suite_horizon_high_level_steps",
            "registered_source_exposure_high_level_steps",
        ),
        "case_identity",
    )
    if identity.get("dataset") != "SafeLIBERO":
        _error("case_identity.dataset", "must equal 'SafeLIBERO'")
    if identity.get("manifest_case_id") != case_id:
        _error("case_identity.manifest_case_id", "must equal case_id")
    _integer(identity.get("manifest_case_ordinal"), "case_identity.manifest_case_ordinal")
    pair_group_id = _string(identity.get("pair_group_id"), "case_identity.pair_group_id")
    _enum(identity.get("suite"), SUITES, "case_identity.suite")
    _enum(identity.get("safety_level"), ("I", "II"), "case_identity.safety_level")
    for field in ("logical_task_index", "resolved_task_index", "episode_index"):
        _integer(identity.get(field), "case_identity.%s" % field)
    for field in (
        "task_name",
        "task_level_group_id",
        "active_obstacle_name",
    ):
        _string(identity.get(field), "case_identity.%s" % field)
    require_sha256(identity.get("task_family_id"), "case_identity.task_family_id")
    for prefix in ("bddl", "initial_states"):
        _relative_posix_path(
            identity.get(prefix + "_relative_path"),
            "case_identity.%s_relative_path" % prefix,
        )
        require_sha256(
            identity.get(prefix + "_sha256"),
            "case_identity.%s_sha256" % prefix,
        )
    require_sha256(
        identity.get("initial_state_record_sha256"),
        "case_identity.initial_state_record_sha256",
    )
    _enum(
        identity.get("split"),
        ("bringup_canary", "parameter_freeze", "heldout_evaluation"),
        "case_identity.split",
    )
    _enum(
        identity.get("stratum"),
        (
            "primary_static_positive_aegis_barrier",
            "support_contact_after_settling_stress",
            "nonpositive_aegis_barrier_recovery_stress",
        ),
        "case_identity.stratum",
    )
    suite_horizon_steps = _integer(
        identity.get("suite_horizon_high_level_steps"),
        "case_identity.suite_horizon_high_level_steps",
        1,
    )
    registered_source_exposure_steps = _integer(
        identity.get("registered_source_exposure_high_level_steps"),
        "case_identity.registered_source_exposure_high_level_steps",
        1,
    )
    if registered_source_exposure_steps > suite_horizon_steps:
        _error(
            "case_identity.registered_source_exposure_high_level_steps",
            "cannot exceed the suite horizon",
        )

    seeds = _object(result.get("seeds"), "seeds")
    _exact_keys(seeds, ("inventory_complete", "entries", "ledger_sha256"), "seeds")
    if not _bool(seeds.get("inventory_complete"), "seeds.inventory_complete"):
        _error("seeds.inventory_complete", "must be true")
    entries = seeds.get("entries")
    if not isinstance(entries, list) or len(entries) < 2:
        _error("seeds.entries", "must contain the complete seed inventory")
    seen_seed_names = set()
    seen_seed_scopes = set()
    for index, entry_value in enumerate(entries):
        label = "seeds.entries[%d]" % index
        entry = _object(entry_value, label)
        _exact_keys(entry, ("scope", "name", "value"), label)
        scope = _enum(
            entry.get("scope"),
            ("python", "numpy", "simulator", "policy", "policy_noise"),
            label + ".scope",
        )
        name = _string(entry.get("name"), label + ".name")
        _integer(entry.get("value"), label + ".value")
        key = (scope, name)
        if key in seen_seed_names:
            _error("seeds.entries", "must not contain duplicate scope/name pairs")
        seen_seed_names.add(key)
        seen_seed_scopes.add(scope)
    if "simulator" not in seen_seed_scopes or not {
        "policy",
        "policy_noise",
    }.intersection(seen_seed_scopes):
        _error("seeds.entries", "must record simulator and policy-side seeds")
    seed_ledger_hash = require_sha256(seeds.get("ledger_sha256"), "seeds.ledger_sha256")
    if seed_ledger_hash != sha256_bytes(canonical_json_bytes(entries)):
        _error("seeds.ledger_sha256", "does not match the ordered seed entries")

    pairing = _object(result.get("pairing"), "pairing")
    _exact_keys(
        pairing,
        (
            "pair_group_id",
            "pairing_key_sha256",
            "restored_settled_state_sha256",
            "source_settled_observation_sha256",
            "active_joint_velocity_initial_observation_sha256",
            "policy_noise_schedule_sha256",
            "policy_query_schedule_sha256",
            "nominal_high_level_action_ledger_sha256",
            "settling_action_ledger_sha256",
            "controller_initial_state_sha256",
            "compiled_physical_model_sha256",
            "compiled_mjb_sha256",
            "field_sha256",
            "source_exposure_high_level_steps",
            "inner_updates_per_high_level_action",
            "high_level_dt_seconds",
            "inner_dt_seconds",
            "physics_dt_seconds",
        ),
        "pairing",
    )
    if pairing.get("pair_group_id") != pair_group_id:
        _error("pairing.pair_group_id", "must equal case_identity.pair_group_id")
    for field in (
        "pairing_key_sha256",
        "restored_settled_state_sha256",
        "source_settled_observation_sha256",
        "active_joint_velocity_initial_observation_sha256",
        "policy_noise_schedule_sha256",
        "policy_query_schedule_sha256",
        "nominal_high_level_action_ledger_sha256",
        "settling_action_ledger_sha256",
        "controller_initial_state_sha256",
        "compiled_physical_model_sha256",
        "compiled_mjb_sha256",
        "field_sha256",
    ):
        require_sha256(pairing.get(field), "pairing.%s" % field)
    source_exposure_steps = _integer(
        pairing.get("source_exposure_high_level_steps"),
        "pairing.source_exposure_high_level_steps",
        1,
    )
    if source_exposure_steps != registered_source_exposure_steps:
        _error(
            "pairing.source_exposure_high_level_steps",
            "must equal case_identity.registered_source_exposure_high_level_steps",
        )
    if pairing.get("inner_updates_per_high_level_action") != 5:
        _error("pairing.inner_updates_per_high_level_action", "must equal 5")
    if _number(pairing.get("high_level_dt_seconds"), "pairing.high_level_dt_seconds") != 0.05:
        _error("pairing.high_level_dt_seconds", "must equal 0.05")
    if _number(pairing.get("inner_dt_seconds"), "pairing.inner_dt_seconds") != 0.01:
        _error("pairing.inner_dt_seconds", "must equal 0.01")
    if _number(pairing.get("physics_dt_seconds"), "pairing.physics_dt_seconds") != 0.002:
        _error("pairing.physics_dt_seconds", "must equal 0.002")

    runtime = _object(result.get("runtime"), "runtime")
    _exact_keys(
        runtime,
        (
            "protocol_parameter_block_sha256",
            "model",
            "sampler",
            "intervention",
            "controller",
            "action_space",
            "measurement",
        ),
        "runtime",
    )
    require_sha256(
        runtime.get("protocol_parameter_block_sha256"),
        "runtime.protocol_parameter_block_sha256",
    )

    model = _object(runtime.get("model"), "runtime.model")
    _exact_keys(
        model,
        (
            "policy_name",
            "model_configuration_sha256",
            "checkpoint_identifier",
            "checkpoint_sha256",
            "checkpoint_hash_semantics",
            "normalization_statistics_sha256",
        ),
        "runtime.model",
    )
    if model.get("policy_name") != "pi0.5":
        _error("runtime.model.policy_name", "must equal 'pi0.5'")
    _string(
        model.get("checkpoint_identifier"), "runtime.model.checkpoint_identifier"
    )
    if model.get("checkpoint_hash_semantics") != "sha256_content_tree_v1":
        _error(
            "runtime.model.checkpoint_hash_semantics",
            "must equal 'sha256_content_tree_v1'",
        )
    for field in (
        "model_configuration_sha256",
        "checkpoint_sha256",
        "normalization_statistics_sha256",
    ):
        require_sha256(model.get(field), "runtime.model.%s" % field)

    sampler = _object(runtime.get("sampler"), "runtime.sampler")
    _exact_keys(
        sampler,
        (
            "implementation",
            "configuration_sha256",
            "source_action_horizon",
            "source_replan_steps",
            "source_policy_query_count",
            "active_policy_query_count",
            "active_policy_rng_exercised",
            "source_action_semantics",
            "deterministic_replay",
        ),
        "runtime.sampler",
    )
    if (
        sampler.get("implementation")
        != "frozen_historical_pi05_plus_aegis_executed_action_replay"
    ):
        _error(
            "runtime.sampler.implementation",
            "must name the frozen historical pi0.5+AEGIS executed-action replay",
        )
    require_sha256(
        sampler.get("configuration_sha256"),
        "runtime.sampler.configuration_sha256",
    )
    action_horizon = _integer(
        sampler.get("source_action_horizon"),
        "runtime.sampler.source_action_horizon",
        1,
    )
    replan_steps = _integer(
        sampler.get("source_replan_steps"),
        "runtime.sampler.source_replan_steps",
        1,
    )
    if replan_steps > action_horizon:
        _error("runtime.sampler.replan_steps", "cannot exceed action_horizon")
    if action_horizon != 10 or replan_steps != 5:
        _error(
            "runtime.sampler",
            "must preserve the registered 10-action horizon and 5-action replanning",
        )
    source_policy_query_count = _integer(
        sampler.get("source_policy_query_count"),
        "runtime.sampler.source_policy_query_count",
    )
    active_policy_query_count = _integer(
        sampler.get("active_policy_query_count"),
        "runtime.sampler.active_policy_query_count",
    )
    if active_policy_query_count != 0:
        _error(
            "runtime.sampler.active_policy_query_count",
            "must be zero because this canary replays recorded actions",
        )
    if _bool(
        sampler.get("active_policy_rng_exercised"),
        "runtime.sampler.active_policy_rng_exercised",
    ):
        _error(
            "runtime.sampler.active_policy_rng_exercised",
            "must be false because no active policy query or RNG is executed",
        )
    expected_source_queries = (
        source_exposure_steps + replan_steps - 1
    ) // replan_steps
    if source_policy_query_count != expected_source_queries:
        _error(
            "runtime.sampler.source_policy_query_count",
            "does not match the recorded five-action source schedule",
        )
    if sampler.get("source_action_semantics") != (
        "historical_pi05_plus_aegis_translational_exact_actions_executed_not_nominal_raw"
    ):
        _error(
            "runtime.sampler.source_action_semantics",
            "must disclose that Poisson is applied after recorded AEGIS actions",
        )
    if not _bool(
        sampler.get("deterministic_replay"),
        "runtime.sampler.deterministic_replay",
    ):
        _error(
            "runtime.sampler.deterministic_replay", "must be true"
        )

    intervention = _object(runtime.get("intervention"), "runtime.intervention")
    _exact_keys(
        intervention,
        (
            "enabled",
            "mode",
            "configuration_sha256",
            "protected_robot_bodies",
            "static_selected_obstacle_only",
            "fail_closed",
        ),
        "runtime.intervention",
    )
    intervention_enabled = _bool(
        intervention.get("enabled"), "runtime.intervention.enabled"
    )
    if intervention.get("mode") != arm:
        _error("runtime.intervention.mode", "must equal the result arm")
    require_sha256(
        intervention.get("configuration_sha256"),
        "runtime.intervention.configuration_sha256",
    )
    protected_bodies = _string_list(
        intervention.get("protected_robot_bodies"),
        "runtime.intervention.protected_robot_bodies",
    )
    if arm == "joint_velocity_adapter_only":
        if intervention_enabled or protected_bodies:
            _error(
                "runtime.intervention",
                "adapter-only results must disable intervention and protect no bodies",
            )
    else:
        if not intervention_enabled or not protected_bodies:
            _error(
                "runtime.intervention",
                "Poisson arms must enable intervention and name protected bodies",
            )
    if arm == "joint_velocity_psf_link56" and set(protected_bodies) != {
        "robot0_link5",
        "robot0_link6",
    }:
        _error(
            "runtime.intervention.protected_robot_bodies",
            "must contain exactly robot0_link5 and robot0_link6 for the link56 arm",
        )
    if not _bool(
        intervention.get("static_selected_obstacle_only"),
        "runtime.intervention.static_selected_obstacle_only",
    ):
        _error(
            "runtime.intervention.static_selected_obstacle_only",
            "must be true for this protocol",
        )
    if not _bool(intervention.get("fail_closed"), "runtime.intervention.fail_closed"):
        _error("runtime.intervention.fail_closed", "must be true")

    controller = _object(runtime.get("controller"), "runtime.controller")
    _exact_keys(
        controller,
        (
            "name",
            "implementation_version",
            "configuration_sha256",
            "controlled_joint_count",
            "command_dimension",
            "normalized_limit_abs",
            "physical_velocity_limit_rad_s",
        ),
        "runtime.controller",
    )
    if controller.get("name") != "JOINT_VELOCITY":
        _error("runtime.controller.name", "must equal 'JOINT_VELOCITY'")
    _string(
        controller.get("implementation_version"),
        "runtime.controller.implementation_version",
    )
    require_sha256(
        controller.get("configuration_sha256"),
        "runtime.controller.configuration_sha256",
    )
    if _integer(
        controller.get("controlled_joint_count"),
        "runtime.controller.controlled_joint_count",
        1,
    ) != 7:
        _error("runtime.controller.controlled_joint_count", "must equal 7")
    if _integer(
        controller.get("command_dimension"),
        "runtime.controller.command_dimension",
        1,
    ) != 8:
        _error("runtime.controller.command_dimension", "must equal 8")
    if _number(
        controller.get("normalized_limit_abs"),
        "runtime.controller.normalized_limit_abs",
        minimum=0.0,
    ) != 1.0:
        _error("runtime.controller.normalized_limit_abs", "must equal 1.0")
    if _number(
        controller.get("physical_velocity_limit_rad_s"),
        "runtime.controller.physical_velocity_limit_rad_s",
        minimum=0.0,
    ) != 0.5:
        _error(
            "runtime.controller.physical_velocity_limit_rad_s", "must equal 0.5"
        )

    action_space = _object(runtime.get("action_space"), "runtime.action_space")
    _exact_keys(
        action_space,
        (
            "source_policy_frame",
            "controller_command_frame",
            "source_policy_normalization_space",
            "controller_normalization_space",
            "physical_displacement_conversion",
            "gripper_semantics",
        ),
        "runtime.action_space",
    )
    expected_action_literals = {
        "source_policy_frame": "mujoco_world_frame_cartesian_delta",
        "controller_command_frame": "panda_joint_velocity_coordinates",
        "source_policy_normalization_space": "openpi_output_after_libero_action_unnormalization",
        "controller_normalization_space": "robosuite_joint_velocity_normalized_minus1_plus1",
        "physical_displacement_conversion": "scale_only_never_subtract_normalization_mean",
        "gripper_semantics": "unchanged_vla_command",
    }
    for field, expected in expected_action_literals.items():
        if action_space.get(field) != expected:
            _error("runtime.action_space.%s" % field, "must equal %r" % expected)

    measurement = _object(runtime.get("measurement"), "runtime.measurement")
    _exact_keys(
        measurement,
        (
            "schema_version",
            "implementation_sha256",
            "configuration_sha256",
            "simulator",
            "simulator_version",
            "robosuite_version",
            "physics_timestep_seconds",
            "contact_definition",
            "contact_sources",
            "D_opt_semantics",
            "D_sim_semantics",
            "optimizer_and_simulator_clearance_distinct",
        ),
        "runtime.measurement",
    )
    if measurement.get("schema_version") != MEASUREMENT_SCHEMA_VERSION:
        _error("runtime.measurement.schema_version", "is unsupported")
    for field in ("implementation_sha256", "configuration_sha256"):
        require_sha256(measurement.get(field), "runtime.measurement.%s" % field)
    if measurement.get("simulator") != "MuJoCo":
        _error("runtime.measurement.simulator", "must equal 'MuJoCo'")
    _string(measurement.get("simulator_version"), "runtime.measurement.simulator_version")
    _string(measurement.get("robosuite_version"), "runtime.measurement.robosuite_version")
    if _number(
        measurement.get("physics_timestep_seconds"),
        "runtime.measurement.physics_timestep_seconds",
    ) != 0.002:
        _error("runtime.measurement.physics_timestep_seconds", "must equal 0.002")
    if measurement.get("contact_definition") != "mujoco_contact_dist_le_zero":
        _error(
            "runtime.measurement.contact_definition",
            "must equal 'mujoco_contact_dist_le_zero'",
        )
    contact_sources = _string_list(
        measurement.get("contact_sources"),
        "runtime.measurement.contact_sources",
        minimum=3,
    )
    if set(contact_sources) != {
        "settled_state",
        "live_solver_state",
        "forwarded_post_integration_state",
    }:
        _error(
            "runtime.measurement.contact_sources",
            "must contain the three registered contact authorities",
        )
    if (
        measurement.get("D_opt_semantics")
        != "minimum_registered_surface_sample_signed_distance_to_selected_obstacle_obb_union"
    ):
        _error(
            "runtime.measurement.D_opt_semantics",
            "must name the registered optimizer-clearance diagnostic",
        )
    measured_d_sim_semantics = _enum(
        measurement.get("D_sim_semantics"),
        D_SIM_SEMANTICS,
        "runtime.measurement.D_sim_semantics",
    )
    if not _bool(
        measurement.get("optimizer_and_simulator_clearance_distinct"),
        "runtime.measurement.optimizer_and_simulator_clearance_distinct",
    ):
        _error(
            "runtime.measurement.optimizer_and_simulator_clearance_distinct",
            "must be true",
        )

    optimizer = _object(result.get("optimizer"), "optimizer")
    _exact_keys(
        optimizer,
        (
            "enabled",
            "solver",
            "solver_version",
            "configuration_sha256",
            "attempt_count",
            "solved_count",
            "infeasible_count",
            "solver_failure_count",
            "postcheck_failure_count",
            "terminal_status",
            "minimum_normalized_cbf_residual",
            "minimum_raw_cbf_residual_m2_per_s",
            "normalized_cbf_postcheck_tolerance",
        ),
        "optimizer",
    )
    optimizer_enabled = _bool(optimizer.get("enabled"), "optimizer.enabled")
    if optimizer.get("solver") != "osqp":
        _error("optimizer.solver", "must equal 'osqp'")
    _string(optimizer.get("solver_version"), "optimizer.solver_version")
    require_sha256(
        optimizer.get("configuration_sha256"), "optimizer.configuration_sha256"
    )
    optimizer_counts = {}
    for field in (
        "attempt_count",
        "solved_count",
        "infeasible_count",
        "solver_failure_count",
        "postcheck_failure_count",
    ):
        optimizer_counts[field] = _integer(optimizer.get(field), "optimizer.%s" % field)
    if optimizer_counts["attempt_count"] != sum(
        optimizer_counts[field]
        for field in (
            "solved_count",
            "infeasible_count",
            "solver_failure_count",
            "postcheck_failure_count",
        )
    ):
        _error("optimizer.attempt_count", "must equal all classified attempt outcomes")
    optimizer_terminal_status = _enum(
        optimizer.get("terminal_status"),
        (
            "not_applicable",
            "not_reached",
            "solved",
            "infeasible",
            "solver_failure",
            "postcheck_failure",
        ),
        "optimizer.terminal_status",
    )
    normalized_postcheck_tolerance = _number(
        optimizer.get("normalized_cbf_postcheck_tolerance"),
        "optimizer.normalized_cbf_postcheck_tolerance",
        minimum=0.0,
    )
    if normalized_postcheck_tolerance != 5e-7:
        _error(
            "optimizer.normalized_cbf_postcheck_tolerance",
            "must equal the registered normalized-row tolerance",
        )
    minimum_normalized_optimizer_residual = optimizer.get(
        "minimum_normalized_cbf_residual"
    )
    minimum_raw_optimizer_residual = optimizer.get(
        "minimum_raw_cbf_residual_m2_per_s"
    )
    if optimizer_counts["solved_count"] > 0:
        minimum_normalized_optimizer_residual = _number(
            minimum_normalized_optimizer_residual,
            "optimizer.minimum_normalized_cbf_residual",
        )
        minimum_raw_optimizer_residual = _number(
            minimum_raw_optimizer_residual,
            "optimizer.minimum_raw_cbf_residual_m2_per_s",
        )
    elif (
        minimum_normalized_optimizer_residual is not None
        or minimum_raw_optimizer_residual is not None
    ):
        _error(
            "optimizer",
            "cannot report solved-QP residuals without a solved attempt",
        )
    if arm == "joint_velocity_adapter_only":
        if (
            optimizer_enabled
            or optimizer_counts["attempt_count"] != 0
            or optimizer_terminal_status != "not_applicable"
        ):
            _error("optimizer", "must be disabled and not applicable for adapter-only")
    elif not optimizer_enabled or optimizer_terminal_status == "not_applicable":
        _error("optimizer", "must be enabled for every Poisson arm")
    terminal_status_count = {
        "solved": "solved_count",
        "infeasible": "infeasible_count",
        "solver_failure": "solver_failure_count",
        "postcheck_failure": "postcheck_failure_count",
    }.get(optimizer_terminal_status)
    if (
        terminal_status_count is not None
        and optimizer_counts[terminal_status_count] == 0
    ):
        _error(
            "optimizer.terminal_status",
            "requires a positive %s" % terminal_status_count,
        )

    allocation = _object(result.get("allocation"), "allocation")
    _exact_keys(
        allocation,
        (
            "execution_environment",
            "slurm_job_id",
            "slurm_array_job_id",
            "slurm_array_task_id",
            "slurm_step_id",
            "host_name",
            "process_id",
            "device",
        ),
        "allocation",
    )
    if allocation.get("execution_environment") != "slurm_allocation":
        _error(
            "allocation.execution_environment", "must equal 'slurm_allocation'"
        )
    _digit_string(allocation.get("slurm_job_id"), "allocation.slurm_job_id")
    array_job_id = _nullable_string(
        allocation.get("slurm_array_job_id"), "allocation.slurm_array_job_id"
    )
    array_task_id = _nullable_string(
        allocation.get("slurm_array_task_id"), "allocation.slurm_array_task_id"
    )
    if (array_job_id is None) != (array_task_id is None):
        _error(
            "allocation",
            "slurm_array_job_id and slurm_array_task_id must both be set or both be null",
        )
    if array_job_id is not None:
        _digit_string(array_job_id, "allocation.slurm_array_job_id")
        _digit_string(array_task_id, "allocation.slurm_array_task_id")
    slurm_step_id = _nullable_string(
        allocation.get("slurm_step_id"), "allocation.slurm_step_id"
    )
    if (
        slurm_step_id is not None
        and slurm_step_id not in {"batch", "extern", "interactive"}
        and not all(part.isdigit() for part in slurm_step_id.split("."))
    ):
        _error(
            "allocation.slurm_step_id",
            "must be a Slurm named step, dot-separated decimal identifiers, or null",
        )
    _string(allocation.get("host_name"), "allocation.host_name")
    _integer(allocation.get("process_id"), "allocation.process_id", 1)
    device = _object(allocation.get("device"), "allocation.device")
    _exact_keys(
        device,
        (
            "device_type",
            "visible_device_ids",
            "model",
            "uuid",
            "driver_version",
            "cuda_runtime_version",
        ),
        "allocation.device",
    )
    if device.get("device_type") != "cuda":
        _error("allocation.device.device_type", "must equal 'cuda'")
    _string_list(
        device.get("visible_device_ids"),
        "allocation.device.visible_device_ids",
        minimum=1,
    )
    for field in ("model", "uuid", "driver_version", "cuda_runtime_version"):
        _string(device.get(field), "allocation.device.%s" % field)
    if "H100" not in device.get("model"):
        _error("allocation.device.model", "must identify an H100 allocation")

    execution = _object(result.get("execution"), "execution")
    _exact_keys(
        execution,
        (
            "high_level_steps",
            "inner_control_steps",
            "physics_substeps",
            "physics_exposure_seconds",
            "exposure_complete",
            "fail_closed_no_further_physics",
            "terminal_reason",
            "completed_high_level_steps",
            "terminal_entered_high_level_index",
            "terminal_entered_inner_control_index",
            "completed_physics_substeps_in_terminal_inner_control",
            "source_action_prefix_semantics",
            "planned_source_action_ledger_sha256",
            "entered_source_action_prefix_sha256",
            "completed_high_level_source_action_prefix_sha256",
            "nominal_joint_velocity_ledger_sha256",
            "executed_controller_action_ledger_sha256",
            "terminal_simulator_state_sha256",
            "terminal_simulator_state_semantics",
            "terminal_observation_sha256",
            "terminal_observation_semantics",
        ),
        "execution",
    )
    high_steps = _integer(execution.get("high_level_steps"), "execution.high_level_steps")
    inner_steps = _integer(execution.get("inner_control_steps"), "execution.inner_control_steps")
    physics_steps = _integer(execution.get("physics_substeps"), "execution.physics_substeps")
    physics_exposure = _number(
        execution.get("physics_exposure_seconds"),
        "execution.physics_exposure_seconds",
        minimum=0.0,
    )
    if not math.isclose(
        physics_exposure,
        physics_steps * 0.002,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        _error(
            "execution.physics_exposure_seconds",
            "must equal physics_substeps times 0.002 seconds",
        )
    _string(execution.get("terminal_reason"), "execution.terminal_reason")
    completed_high_steps = _integer(
        execution.get("completed_high_level_steps"),
        "execution.completed_high_level_steps",
    )
    terminal_high_index = execution.get("terminal_entered_high_level_index")
    terminal_inner_index = execution.get("terminal_entered_inner_control_index")
    if terminal_high_index is not None:
        _integer(
            terminal_high_index,
            "execution.terminal_entered_high_level_index",
        )
    if terminal_inner_index is not None:
        terminal_inner_index = _integer(
            terminal_inner_index,
            "execution.terminal_entered_inner_control_index",
        )
        if terminal_inner_index > 4:
            _error(
                "execution.terminal_entered_inner_control_index",
                "must be in 0..4",
            )
    completed_terminal_substeps = _integer(
        execution.get("completed_physics_substeps_in_terminal_inner_control"),
        "execution.completed_physics_substeps_in_terminal_inner_control",
    )
    if completed_terminal_substeps > 5:
        _error(
            "execution.completed_physics_substeps_in_terminal_inner_control",
            "must be in 0..5",
        )
    if execution.get("source_action_prefix_semantics") != (
        "entered_hash_counts_provider_entered_actions_completed_hash_counts_only_"
        "fully_returned_high_level_post_steps"
    ):
        _error(
            "execution.source_action_prefix_semantics",
            "must distinguish entered actions from completed high-level post-steps",
        )
    for field in (
        "planned_source_action_ledger_sha256",
        "entered_source_action_prefix_sha256",
        "completed_high_level_source_action_prefix_sha256",
        "nominal_joint_velocity_ledger_sha256",
        "executed_controller_action_ledger_sha256",
        "terminal_simulator_state_sha256",
        "terminal_observation_sha256",
    ):
        require_sha256(execution.get(field), "execution.%s" % field)
    if execution.get("terminal_simulator_state_semantics") != (
        "complete_official_mujoco_mjSTATE_INTEGRATION"
    ):
        _error(
            "execution.terminal_simulator_state_semantics",
            "must bind the complete official MuJoCo integration state",
        )
    if execution.get("terminal_observation_semantics") != (
        "current_state_synchronized_without_additional_physics"
    ):
        _error(
            "execution.terminal_observation_semantics",
            "must describe a current-state observation",
        )
    if (
        execution.get("planned_source_action_ledger_sha256")
        != pairing.get("nominal_high_level_action_ledger_sha256")
    ):
        _error(
            "execution.planned_source_action_ledger_sha256",
            "must equal the paired frozen source-action ledger",
        )
    exposure_complete = _bool(execution.get("exposure_complete"), "execution.exposure_complete")
    fail_closed = _bool(
        execution.get("fail_closed_no_further_physics"),
        "execution.fail_closed_no_further_physics",
    )
    if completion == "executed":
        if not exposure_complete:
            _error("execution.exposure_complete", "must be true for executed results")
        if fail_closed:
            _error("execution.fail_closed_no_further_physics", "must be false for executed results")
        if (
            execution.get("entered_source_action_prefix_sha256")
            != execution.get("planned_source_action_ledger_sha256")
            or execution.get("completed_high_level_source_action_prefix_sha256")
            != execution.get("planned_source_action_ledger_sha256")
        ):
            _error(
                "execution",
                "complete exposure must enter and execute the full frozen source-action ledger",
            )
        if high_steps != source_exposure_steps or inner_steps != 5 * high_steps or physics_steps != 5 * inner_steps:
            _error("execution", "does not preserve the registered 20/100/500 Hz exposure")
        if completed_high_steps != high_steps:
            _error("execution.completed_high_level_steps", "must equal complete exposure")
    else:
        if exposure_complete:
            _error("execution.exposure_complete", "must be false for fail-closed results")
        if not fail_closed:
            _error("execution.fail_closed_no_further_physics", "must be true for fail-closed results")

        # Counters use prefix semantics: high_level_steps and
        # inner_control_steps count entered updates, while physics_substeps
        # counts completed integration steps.  This admits stopping before the
        # first physics step of the current inner update, but rejects impossible
        # traces such as ten entered inner updates with zero physics.
        if high_steps == 0:
            if inner_steps != 0 or physics_steps != 0:
                _error("execution", "has inner or physics work without a high-level update")
        else:
            if not (5 * (high_steps - 1) < inner_steps <= 5 * high_steps):
                _error("execution", "inner updates are not an exact high-level prefix")
            if not (5 * (inner_steps - 1) <= physics_steps <= 5 * inner_steps):
                _error("execution", "physics substeps are not an exact inner-update prefix")
        if high_steps > source_exposure_steps:
            _error("execution.high_level_steps", "exceeds the registered horizon")

    if completed_high_steps > high_steps:
        _error(
            "execution.completed_high_level_steps",
            "cannot exceed entered high-level steps",
        )
    if high_steps == 0:
        if terminal_high_index is not None or terminal_inner_index is not None:
            _error("execution", "zero entered steps require null terminal indexes")
        if completed_terminal_substeps != 0 or completed_high_steps != 0:
            _error("execution", "zero entered steps require zero completed prefixes")
    else:
        if terminal_high_index != high_steps - 1:
            _error(
                "execution.terminal_entered_high_level_index",
                "must identify the last entered high-level action",
            )
        if inner_steps == 0:
            if terminal_inner_index is not None or completed_terminal_substeps != 0:
                _error("execution", "no entered inner control permits no inner coordinate")
        else:
            if terminal_inner_index != (inner_steps - 1) % 5:
                _error(
                    "execution.terminal_entered_inner_control_index",
                    "must identify the last entered inner control",
                )
            if completed_terminal_substeps != physics_steps - 5 * (inner_steps - 1):
                _error(
                    "execution.completed_physics_substeps_in_terminal_inner_control",
                    "does not match the exact completed physics prefix",
                )

    endpoints = _object(result.get("endpoints"), "endpoints")
    _exact_keys(endpoints, ("safety", "task", "usefulness", "validity"), "endpoints")
    safety = _object(endpoints.get("safety"), "endpoints.safety")
    _exact_keys(
        safety,
        (
            "h_min",
            "minimum_nominal_cbf_residual",
            "minimum_safe_cbf_residual",
            "minimum_realized_cbf_residual",
            "D_opt_min_m",
            "D_sim_min_m",
            "D_sim_semantics",
            "any_robot_obstacle_contact",
            "link56_obstacle_contact",
            "contact_observation_complete",
            "contact_absence_right_censored",
            "total_physical_contact_point_record_count",
            "settled_physical_contact_point_record_count",
            "rollout_phase_physical_contact_point_record_count",
            "live_solver_nonpositive_contact_point_record_count",
            "post_state_physical_contact_point_record_count",
            "first_physical_contact_point_record",
            "first_settled_physical_contact_point_record",
            "first_live_solver_physical_contact_point_record",
            "first_post_state_physical_contact_point_record",
            "paper_car",
        ),
        "endpoints.safety",
    )
    h_minimum = _measurement(
        safety.get("h_min"),
        "endpoints.safety.h_min",
        units="poisson_field_m2",
        allow_negative=True,
    )
    minimum_nominal_residual = _measurement(
        safety.get("minimum_nominal_cbf_residual"),
        "endpoints.safety.minimum_nominal_cbf_residual",
        units="poisson_cbf_m2_per_s",
        allow_negative=True,
    )
    minimum_safe_residual = _measurement(
        safety.get("minimum_safe_cbf_residual"),
        "endpoints.safety.minimum_safe_cbf_residual",
        units="poisson_cbf_m2_per_s",
        allow_negative=True,
    )
    minimum_realized_residual = _measurement(
        safety.get("minimum_realized_cbf_residual"),
        "endpoints.safety.minimum_realized_cbf_residual",
        units="poisson_cbf_m2_per_s",
        allow_negative=True,
    )
    d_optimum = _measurement(
        safety.get("D_opt_min_m"),
        "endpoints.safety.D_opt_min_m",
        units="m",
        allow_negative=True,
    )
    d_sim = _measurement(
        safety.get("D_sim_min_m"),
        "endpoints.safety.D_sim_min_m",
        units="m",
        allow_negative=True,
    )
    if safety.get("D_sim_semantics") not in D_SIM_SEMANTICS:
        _error("endpoints.safety.D_sim_semantics", "is unsupported")
    if safety.get("D_sim_semantics") != measured_d_sim_semantics:
        _error(
            "endpoints.safety.D_sim_semantics",
            "must equal runtime.measurement.D_sim_semantics",
        )
    any_contact = _bool(safety.get("any_robot_obstacle_contact"), "endpoints.safety.any_robot_obstacle_contact")
    link56_contact = _bool(
        safety.get("link56_obstacle_contact"),
        "endpoints.safety.link56_obstacle_contact",
    )
    contact_observation_complete = _bool(
        safety.get("contact_observation_complete"),
        "endpoints.safety.contact_observation_complete",
    )
    contact_absence_right_censored = _bool(
        safety.get("contact_absence_right_censored"),
        "endpoints.safety.contact_absence_right_censored",
    )
    if contact_observation_complete != exposure_complete:
        _error(
            "endpoints.safety.contact_observation_complete",
            "must equal fixed-exposure completeness",
        )
    if contact_absence_right_censored != (
        not exposure_complete and not any_contact
    ):
        _error(
            "endpoints.safety.contact_absence_right_censored",
            "must be true exactly for an unobserved-contact partial prefix",
        )
    if any_contact and d_sim is not None and d_sim > 1e-6:
        _error("endpoints.safety.D_sim_min_m", "cannot be positive when contact was observed")
    total_contact_count = _integer(
        safety.get("total_physical_contact_point_record_count"),
        "endpoints.safety.total_physical_contact_point_record_count",
    )
    settled_contact_count = _integer(
        safety.get("settled_physical_contact_point_record_count"),
        "endpoints.safety.settled_physical_contact_point_record_count",
    )
    rollout_contact_count = _integer(
        safety.get("rollout_phase_physical_contact_point_record_count"),
        "endpoints.safety.rollout_phase_physical_contact_point_record_count",
    )
    live_solver_contact_count = _integer(
        safety.get("live_solver_nonpositive_contact_point_record_count"),
        "endpoints.safety.live_solver_nonpositive_contact_point_record_count",
    )
    post_state_contact_count = _integer(
        safety.get("post_state_physical_contact_point_record_count"),
        "endpoints.safety.post_state_physical_contact_point_record_count",
    )
    first_contact = safety.get("first_physical_contact_point_record")
    first_settled = safety.get("first_settled_physical_contact_point_record")
    first_live_solver = safety.get(
        "first_live_solver_physical_contact_point_record"
    )
    first_post_state = safety.get(
        "first_post_state_physical_contact_point_record"
    )
    for label, record in (
        ("first_physical_contact_point_record", first_contact),
        ("first_settled_physical_contact_point_record", first_settled),
        ("first_live_solver_physical_contact_point_record", first_live_solver),
        ("first_post_state_physical_contact_point_record", first_post_state),
    ):
        if not isinstance(record, (dict, type(None))):
            _error("endpoints.safety." + label, "must be an object or null")
        if record is not None:
            _contact_record(record, "endpoints.safety." + label)
    if total_contact_count != settled_contact_count + rollout_contact_count:
        _error(
            "endpoints.safety.total_physical_contact_point_record_count",
            "must equal settled plus rollout-phase contact-point records",
        )
    if rollout_contact_count != live_solver_contact_count + post_state_contact_count:
        _error(
            "endpoints.safety.rollout_phase_physical_contact_point_record_count",
            "must equal live-solver plus post-state contact-point records",
        )
    if any_contact != (total_contact_count > 0):
        _error(
            "endpoints.safety.total_physical_contact_point_record_count",
            "must be positive exactly when any_robot_obstacle_contact is true",
        )
    if (first_contact is not None) != any_contact:
        _error(
            "endpoints.safety.first_physical_contact_point_record",
            "must be present exactly when contact was observed",
        )
    for label, record, count in (
        ("first_settled_physical_contact_point_record", first_settled, settled_contact_count),
        ("first_live_solver_physical_contact_point_record", first_live_solver, live_solver_contact_count),
        ("first_post_state_physical_contact_point_record", first_post_state, post_state_contact_count),
    ):
        if (record is not None) != (count > 0):
            _error(
                "endpoints.safety." + label,
                "must be present exactly when its phase has contact-point records",
            )
    if link56_contact and not any_contact:
        _error(
            "endpoints.safety.link56_obstacle_contact",
            "cannot be true without robot-obstacle contact",
        )
    phase_first_records = tuple(
        record
        for record in (first_settled, first_live_solver, first_post_state)
        if record is not None
    )
    if first_contact is not None and first_contact not in phase_first_records:
        _error(
            "endpoints.safety.first_physical_contact_point_record",
            "must equal one of the bound first phase records",
        )

    car = _object(safety.get("paper_car"), "endpoints.safety.paper_car")
    _exact_keys(
        car,
        (
            "available",
            "maximum_active_obstacle_l1_displacement_m",
            "threshold_m",
            "collision",
            "avoidance",
            "reason",
            "exposure_scope",
            "position_ledger_sha256",
        ),
        "endpoints.safety.paper_car",
    )
    car_available = _bool(
        car.get("available"), "endpoints.safety.paper_car.available"
    )
    if car.get("exposure_scope") != (
        "registered_historical_source_exposure_not_full_table1_suite_horizon"
    ):
        _error(
            "endpoints.safety.paper_car.exposure_scope",
            "must distinguish the source exposure from full Table-1 horizon",
        )
    require_sha256(
        car.get("position_ledger_sha256"),
        "endpoints.safety.paper_car.position_ledger_sha256",
    )
    displacement = _number(
        car.get("maximum_active_obstacle_l1_displacement_m"),
        "endpoints.safety.paper_car.maximum_active_obstacle_l1_displacement_m",
        minimum=0.0,
    )
    threshold = _number(car.get("threshold_m"), "endpoints.safety.paper_car.threshold_m")
    if threshold != 0.001:
        _error("endpoints.safety.paper_car.threshold_m", "must equal 0.001")
    if car_available:
        if not exposure_complete:
            _error(
                "endpoints.safety.paper_car.available",
                "cannot be true for a partial or fail-closed exposure",
            )
        collided = _bool(
            car.get("collision"), "endpoints.safety.paper_car.collision"
        )
        avoided = _bool(
            car.get("avoidance"), "endpoints.safety.paper_car.avoidance"
        )
        if car.get("reason") is not None:
            _error(
                "endpoints.safety.paper_car.reason",
                "must be null when CAR is available",
            )
        if collided != (displacement > threshold) or avoided != (not collided):
            _error(
                "endpoints.safety.paper_car",
                "does not use the registered strict > threshold",
            )
    else:
        if exposure_complete:
            _error(
                "endpoints.safety.paper_car.available",
                "must be true after a complete fixed exposure",
            )
        if car.get("collision") is not None or car.get("avoidance") is not None:
            _error(
                "endpoints.safety.paper_car",
                "partial exposure collision/avoidance values must be null",
            )
        _string(car.get("reason"), "endpoints.safety.paper_car.reason")

    task = _object(endpoints.get("task"), "endpoints.task")
    _exact_keys(
        task,
        (
            "fixed_exposure_available",
            "outcome_scope",
            "ever_task_success_within_registered_source_exposure",
            "terminal_task_success_within_registered_source_exposure",
            "first_task_success_high_level_index",
            "initial_task_success",
            "prefix_task_success_latched",
            "prefix_first_task_success_high_level_index",
            "prefix_terminal_task_success",
            "terminal_goal_fraction",
            "maximum_goal_fraction",
            "goal_regression_count",
            "unavailable_reason",
        ),
        "endpoints.task",
    )
    fixed_task_available = _bool(
        task.get("fixed_exposure_available"),
        "endpoints.task.fixed_exposure_available",
    )
    if task.get("outcome_scope") != (
        "registered_source_outcome_dependent_exposure_not_full_table1_suite_horizon"
    ):
        _error(
            "endpoints.task.outcome_scope",
            "must distinguish the source exposure from the Table-1 suite horizon",
        )
    initial_task_success = _bool(
        task.get("initial_task_success"), "endpoints.task.initial_task_success"
    )
    prefix_task_latched = _bool(
        task.get("prefix_task_success_latched"),
        "endpoints.task.prefix_task_success_latched",
    )
    prefix_terminal_task = _bool(
        task.get("prefix_terminal_task_success"),
        "endpoints.task.prefix_terminal_task_success",
    )
    prefix_first_success = task.get("prefix_first_task_success_high_level_index")
    if prefix_first_success is not None:
        prefix_first_success = _integer(
            prefix_first_success,
            "endpoints.task.prefix_first_task_success_high_level_index",
        )
        if prefix_first_success >= high_steps:
            _error(
                "endpoints.task.prefix_first_task_success_high_level_index",
                "must identify an entered high-level action",
            )
    if initial_task_success:
        if not prefix_task_latched or prefix_first_success is not None:
            _error(
                "endpoints.task",
                "initial success must be latched and has no post-action first-success index",
            )
    elif prefix_task_latched != (prefix_first_success is not None):
        _error(
            "endpoints.task",
            "noninitial latched success must have exactly one first-success index",
        )
    if prefix_terminal_task and not prefix_task_latched:
        _error(
            "endpoints.task.prefix_terminal_task_success",
            "terminal prefix success implies latched prefix success",
        )
    if fixed_task_available:
        if not exposure_complete:
            _error(
                "endpoints.task.fixed_exposure_available",
                "cannot be true for a partial exposure",
            )
        ever_success = _bool(
            task.get("ever_task_success_within_registered_source_exposure"),
            "endpoints.task.ever_task_success_within_registered_source_exposure",
        )
        terminal_success = _bool(
            task.get("terminal_task_success_within_registered_source_exposure"),
            "endpoints.task.terminal_task_success_within_registered_source_exposure",
        )
        first_success = task.get("first_task_success_high_level_index")
        if first_success is not None:
            _integer(
                first_success,
                "endpoints.task.first_task_success_high_level_index",
            )
        if first_success != prefix_first_success:
            _error(
                "endpoints.task.first_task_success_high_level_index",
                "must equal the complete prefix diagnostic",
            )
        if ever_success != prefix_task_latched:
            _error(
                "endpoints.task.ever_task_success_within_registered_source_exposure",
                "must equal the complete latched prefix diagnostic",
            )
        if terminal_success != prefix_terminal_task:
            _error(
                "endpoints.task.terminal_task_success_within_registered_source_exposure",
                "must equal the complete terminal prefix diagnostic",
            )
        if terminal_success and not ever_success:
            _error(
                "endpoints.task",
                "terminal task success implies task success was achieved",
            )
        if task.get("unavailable_reason") is not None:
            _error(
                "endpoints.task.unavailable_reason",
                "must be null for a complete fixed exposure",
            )
    else:
        if exposure_complete:
            _error(
                "endpoints.task.fixed_exposure_available",
                "must be true for a complete fixed exposure",
            )
        for field in (
            "ever_task_success_within_registered_source_exposure",
            "terminal_task_success_within_registered_source_exposure",
            "first_task_success_high_level_index",
        ):
            if task.get(field) is not None:
                _error(
                    "endpoints.task." + field,
                    "must be null for a right-censored partial exposure",
                )
        _string(
            task.get("unavailable_reason"),
            "endpoints.task.unavailable_reason",
        )
    terminal_fraction = _number(
        task.get("terminal_goal_fraction"),
        "endpoints.task.terminal_goal_fraction",
    )
    fraction = _number(task.get("maximum_goal_fraction"), "endpoints.task.maximum_goal_fraction")
    if not 0.0 <= terminal_fraction <= 1.0 or not 0.0 <= fraction <= 1.0:
        _error("endpoints.task.maximum_goal_fraction", "must lie in [0,1]")
    if fraction < terminal_fraction:
        _error(
            "endpoints.task.maximum_goal_fraction",
            "must be at least the terminal goal fraction",
        )
    if prefix_terminal_task and terminal_fraction != 1.0:
        _error(
            "endpoints.task.terminal_goal_fraction",
            "must equal one when the native goal conjunction is satisfied",
        )
    _integer(task.get("goal_regression_count"), "endpoints.task.goal_regression_count")

    usefulness = _object(endpoints.get("usefulness"), "endpoints.usefulness")
    _exact_keys(
        usefulness,
        (
            "nominal_joint_motion_integral_rad",
            "safe_joint_motion_integral_rad",
            "measured_joint_motion_integral_rad",
            "eef_path_length_m",
            "correction_integral_rad",
            "motion_retention_ratio",
            "zero_motion_fraction",
            "all_issued_arm_joint_commands_zero",
            "safety_by_no_execution",
        ),
        "endpoints.usefulness",
    )
    for field in (
        "nominal_joint_motion_integral_rad",
        "safe_joint_motion_integral_rad",
        "measured_joint_motion_integral_rad",
        "eef_path_length_m",
        "correction_integral_rad",
    ):
        _number(usefulness.get(field), "endpoints.usefulness.%s" % field, minimum=0.0)
    nominal_joint_motion = float(usefulness.get("nominal_joint_motion_integral_rad"))
    safe_joint_motion = float(usefulness.get("safe_joint_motion_integral_rad"))
    correction_integral = float(usefulness.get("correction_integral_rad"))
    retention = _number(usefulness.get("motion_retention_ratio"), "endpoints.usefulness.motion_retention_ratio")
    if retention < 0.0:
        _error("endpoints.usefulness.motion_retention_ratio", "must be nonnegative")
    zero_fraction = _number(usefulness.get("zero_motion_fraction"), "endpoints.usefulness.zero_motion_fraction")
    if not 0.0 <= zero_fraction <= 1.0:
        _error("endpoints.usefulness.zero_motion_fraction", "must lie in [0,1]")
    all_arm_commands_zero = _bool(
        usefulness.get("all_issued_arm_joint_commands_zero"),
        "endpoints.usefulness.all_issued_arm_joint_commands_zero",
    )
    safety_by_no_execution = _bool(
        usefulness.get("safety_by_no_execution"),
        "endpoints.usefulness.safety_by_no_execution",
    )
    expected_retention = (
        safe_joint_motion / nominal_joint_motion
        if nominal_joint_motion > 0.0
        else 0.0
    )
    if not math.isclose(
        retention, expected_retention, rel_tol=0.0, abs_tol=1e-12
    ):
        _error(
            "endpoints.usefulness.motion_retention_ratio",
            "must equal safe divided by nominal motion (or zero for zero nominal motion)",
        )
    if safety_by_no_execution != (physics_steps == 0):
        _error(
            "endpoints.usefulness.safety_by_no_execution",
            "must be true exactly when no physics substep completed",
        )
    if all_arm_commands_zero:
        if physics_steps == 0 or safe_joint_motion != 0.0 or zero_fraction != 1.0:
            _error(
                "endpoints.usefulness.all_issued_arm_joint_commands_zero",
                "requires issued physics, zero safe-arm motion, and zero-motion fraction one",
            )
    elif physics_steps > 0 and safe_joint_motion == 0.0:
        _error(
            "endpoints.usefulness.all_issued_arm_joint_commands_zero",
            "must be true when completed physics used only zero arm-joint commands",
        )
    if safety_by_no_execution and (
        nominal_joint_motion != 0.0
        or safe_joint_motion != 0.0
        or correction_integral != 0.0
        or zero_fraction != 1.0
        or all_arm_commands_zero
    ):
        _error(
            "endpoints.usefulness",
            "no-execution outcome must have zero motion/correction and no issued-command stop claim",
        )
    if arm == "joint_velocity_adapter_only" and (
        correction_integral != 0.0 or safe_joint_motion != nominal_joint_motion
    ):
        _error(
            "endpoints.usefulness",
            "adapter-only outcome cannot contain a filter correction, including partial outcomes",
        )

    validity = _object(endpoints.get("validity"), "endpoints.validity")
    _exact_keys(
        validity,
        (
            "poisson_safe_start",
            "static_admissible",
            "coverage_audit_passed",
            "obstacle_translation_drift_m",
            "obstacle_rotation_drift_rad",
            "invalid_field_query_count",
            "nonpositive_runtime_field_query_count",
            "negative_realized_cbf_residual_count",
            "realized_cbf_observed_physics_substep_count",
            "realized_cbf_residual_evaluation_count",
            "first_negative_realized_cbf_residual",
            "qp_infeasible_count",
            "qp_solver_failure_count",
            "qp_postcheck_failure_count",
            "maximum_velocity_tracking_error_rad_s",
            "velocity_tracking_rmse_rad_s",
            "velocity_tracking_linf_threshold_rad_s",
            "velocity_tracking_rmse_threshold_rad_s",
            "velocity_tracking_gate_semantics",
            "velocity_tracking_observed_physics_substep_count",
            "first_velocity_tracking_threshold_crossing",
        ),
        "endpoints.validity",
    )
    poisson_safe_start = _bool(
        validity.get("poisson_safe_start"),
        "endpoints.validity.poisson_safe_start",
    )
    static_admissible = _bool(
        validity.get("static_admissible"),
        "endpoints.validity.static_admissible",
    )
    coverage_audit_passed = _bool(
        validity.get("coverage_audit_passed"),
        "endpoints.validity.coverage_audit_passed",
    )
    _number(validity.get("obstacle_translation_drift_m"), "endpoints.validity.obstacle_translation_drift_m", minimum=0.0)
    _number(validity.get("obstacle_rotation_drift_rad"), "endpoints.validity.obstacle_rotation_drift_rad", minimum=0.0)
    invalid_field_count = _integer(
        validity.get("invalid_field_query_count"),
        "endpoints.validity.invalid_field_query_count",
    )
    nonpositive_runtime_field_count = _integer(
        validity.get("nonpositive_runtime_field_query_count"),
        "endpoints.validity.nonpositive_runtime_field_query_count",
    )
    negative_realized_residual_count = _integer(
        validity.get("negative_realized_cbf_residual_count"),
        "endpoints.validity.negative_realized_cbf_residual_count",
    )
    realized_observed_substeps = _integer(
        validity.get("realized_cbf_observed_physics_substep_count"),
        "endpoints.validity.realized_cbf_observed_physics_substep_count",
    )
    realized_evaluation_count = _integer(
        validity.get("realized_cbf_residual_evaluation_count"),
        "endpoints.validity.realized_cbf_residual_evaluation_count",
    )
    first_negative_realized = validity.get(
        "first_negative_realized_cbf_residual"
    )
    if first_negative_realized is not None:
        first_negative_realized = _object(
            first_negative_realized,
            "endpoints.validity.first_negative_realized_cbf_residual",
        )
        _exact_keys(
            first_negative_realized,
            (
                "high_level_index",
                "inner_control_index",
                "physics_substep_index",
                "minimum_realized_cbf_residual_m2_per_s",
                "minimum_sample_index",
                "minimum_h_m2",
            ),
            "endpoints.validity.first_negative_realized_cbf_residual",
        )
        _integer(first_negative_realized.get("high_level_index"), "first_negative_realized.high_level_index")
        inner_crossing = _integer(first_negative_realized.get("inner_control_index"), "first_negative_realized.inner_control_index")
        physics_crossing = _integer(first_negative_realized.get("physics_substep_index"), "first_negative_realized.physics_substep_index")
        if inner_crossing > 4 or physics_crossing > 4:
            _error("endpoints.validity.first_negative_realized_cbf_residual", "indexes must be in 0..4")
        if _number(first_negative_realized.get("minimum_realized_cbf_residual_m2_per_s"), "first_negative_realized.minimum_realized_cbf_residual_m2_per_s") >= 0.0:
            _error("endpoints.validity.first_negative_realized_cbf_residual", "must record a negative residual")
        _integer(first_negative_realized.get("minimum_sample_index"), "first_negative_realized.minimum_sample_index")
        _number(first_negative_realized.get("minimum_h_m2"), "first_negative_realized.minimum_h_m2")
        negative_global_substep = (
            _integer(
                first_negative_realized.get("high_level_index"),
                "first_negative_realized.high_level_index",
            )
            * 25
            + inner_crossing * 5
            + physics_crossing
        )
        if negative_global_substep >= physics_steps:
            _error(
                "endpoints.validity.first_negative_realized_cbf_residual",
                "must identify a completed physics substep inside the exact prefix",
            )
    if (first_negative_realized is None) != (negative_realized_residual_count == 0):
        _error(
            "endpoints.validity.first_negative_realized_cbf_residual",
            "must exist exactly when a negative realized residual was observed",
        )
    if negative_realized_residual_count > realized_observed_substeps:
        _error(
            "endpoints.validity.negative_realized_cbf_residual_count",
            "cannot exceed observed realized-CBF physics callbacks",
        )
    if (minimum_realized_residual is None) != (realized_evaluation_count == 0):
        _error(
            "endpoints.safety.minimum_realized_cbf_residual",
            "availability must match whether any realized residual was evaluated",
        )
    qp_infeasible_count = _integer(
        validity.get("qp_infeasible_count"),
        "endpoints.validity.qp_infeasible_count",
    )
    qp_solver_failure_count = _integer(
        validity.get("qp_solver_failure_count"),
        "endpoints.validity.qp_solver_failure_count",
    )
    qp_postcheck_count = _integer(
        validity.get("qp_postcheck_failure_count"),
        "endpoints.validity.qp_postcheck_failure_count",
    )
    maximum_tracking_error = _number(
        validity.get("maximum_velocity_tracking_error_rad_s"),
        "endpoints.validity.maximum_velocity_tracking_error_rad_s",
        minimum=0.0,
    )
    tracking_rmse = _number(
        validity.get("velocity_tracking_rmse_rad_s"),
        "endpoints.validity.velocity_tracking_rmse_rad_s",
        minimum=0.0,
    )
    tracking_linf_threshold = _number(
        validity.get("velocity_tracking_linf_threshold_rad_s"),
        "endpoints.validity.velocity_tracking_linf_threshold_rad_s",
        minimum=0.0,
    )
    tracking_rmse_threshold = _number(
        validity.get("velocity_tracking_rmse_threshold_rad_s"),
        "endpoints.validity.velocity_tracking_rmse_threshold_rad_s",
        minimum=0.0,
    )
    if validity.get("velocity_tracking_gate_semantics") != (
        "empirical_controller_fidelity_gate_not_cbf_safety_certificate"
    ):
        _error(
            "endpoints.validity.velocity_tracking_gate_semantics",
            "must label tracking as an empirical fidelity gate, not a CBF certificate",
        )
    tracking_observed_substeps = _integer(
        validity.get("velocity_tracking_observed_physics_substep_count"),
        "endpoints.validity.velocity_tracking_observed_physics_substep_count",
    )
    if tracking_observed_substeps != physics_steps:
        _error(
            "endpoints.validity.velocity_tracking_observed_physics_substep_count",
            "must equal every completed 2 ms physics substep",
        )
    if tracking_observed_substeps == 0 and (
        maximum_tracking_error != 0.0 or tracking_rmse != 0.0
    ):
        _error(
            "endpoints.validity",
            "zero tracking observations require zero Linf and RMSE diagnostics",
        )
    first_tracking_crossing = validity.get(
        "first_velocity_tracking_threshold_crossing"
    )
    if first_tracking_crossing is not None:
        first_tracking_crossing = _object(
            first_tracking_crossing,
            "endpoints.validity.first_velocity_tracking_threshold_crossing",
        )
        _exact_keys(
            first_tracking_crossing,
            (
                "high_level_index",
                "inner_control_index",
                "physics_substep_index",
                "error_linf_rad_s",
                "cumulative_rmse_rad_s",
                "linf_threshold_rad_s",
                "rmse_threshold_rad_s",
                "command_rad_s",
                "measured_rad_s",
            ),
            "endpoints.validity.first_velocity_tracking_threshold_crossing",
        )
        tracking_high = _integer(
            first_tracking_crossing.get("high_level_index"),
            "first_tracking_crossing.high_level_index",
        )
        tracking_inner = _integer(
            first_tracking_crossing.get("inner_control_index"),
            "first_tracking_crossing.inner_control_index",
        )
        tracking_physics = _integer(
            first_tracking_crossing.get("physics_substep_index"),
            "first_tracking_crossing.physics_substep_index",
        )
        if tracking_inner > 4 or tracking_physics > 4:
            _error(
                "endpoints.validity.first_velocity_tracking_threshold_crossing",
                "indexes must be in 0..4",
            )
        for field in ("error_linf_rad_s", "cumulative_rmse_rad_s", "linf_threshold_rad_s", "rmse_threshold_rad_s"):
            _number(first_tracking_crossing.get(field), "first_tracking_crossing." + field, minimum=0.0)
        for field in ("command_rad_s", "measured_rad_s"):
            values = first_tracking_crossing.get(field)
            if not isinstance(values, list) or len(values) != 7:
                _error("first_tracking_crossing." + field, "must be a length-7 vector")
            for index, value in enumerate(values):
                _number(value, "first_tracking_crossing.%s[%d]" % (field, index))
        tracking_global_substep = (
            tracking_high * 25 + tracking_inner * 5 + tracking_physics
        )
        if tracking_global_substep >= physics_steps:
            _error(
                "endpoints.validity.first_velocity_tracking_threshold_crossing",
                "must identify a completed physics substep inside the exact prefix",
            )
    if tracking_linf_threshold != 0.05 or tracking_rmse_threshold != 0.02:
        _error(
            "endpoints.validity",
            "tracking thresholds must equal the registered 0.05 Linf / 0.02 RMSE limits",
        )
    threshold_crossed = bool(
        maximum_tracking_error > tracking_linf_threshold
        or tracking_rmse > tracking_rmse_threshold
    )
    if arm == "joint_velocity_psf_link56":
        if realized_observed_substeps != physics_steps:
            _error(
                "endpoints.validity.realized_cbf_observed_physics_substep_count",
                "must audit every completed PSF physics substep",
            )
    elif (
        realized_observed_substeps != 0
        or realized_evaluation_count != 0
        or negative_realized_residual_count != 0
        or minimum_realized_residual is not None
    ):
        _error(
            "endpoints.validity.realized_cbf_observed_physics_substep_count",
            "adapter-only arm must not attribute Poisson realized-residual audits",
        )
    if qp_infeasible_count != optimizer_counts["infeasible_count"]:
        _error(
            "endpoints.validity.qp_infeasible_count",
            "must equal optimizer.infeasible_count",
        )
    if qp_solver_failure_count != optimizer_counts["solver_failure_count"]:
        _error(
            "endpoints.validity.qp_solver_failure_count",
            "must equal optimizer.solver_failure_count",
        )
    if qp_postcheck_count != optimizer_counts["postcheck_failure_count"]:
        _error(
            "endpoints.validity.qp_postcheck_failure_count",
            "must equal optimizer.postcheck_failure_count",
        )

    if completion == "executed":
        if threshold_crossed:
            _error(
                "completion_class",
                "executed results cannot contain a registered joint-velocity "
                "tracking-threshold crossing",
            )
        if arm != "joint_velocity_adapter_only" and not (
            poisson_safe_start and static_admissible and coverage_audit_passed
        ):
            _error(
                "endpoints.validity",
                "executed Poisson results require all static-field validity gates",
            )
        if (
            invalid_field_count
            or nonpositive_runtime_field_count
            or qp_infeasible_count
            or qp_solver_failure_count
            or qp_postcheck_count
        ):
            _error("endpoints.validity", "executed results cannot contain fail-closed events")
        if arm != "joint_velocity_adapter_only":
            if (
                optimizer_counts["attempt_count"] == 0
                or optimizer_terminal_status != "solved"
            ):
                _error(
                    "optimizer",
                    "executed Poisson results require at least one solved attempt",
                )
            if (
                minimum_normalized_optimizer_residual is None
                or minimum_normalized_optimizer_residual
                < -normalized_postcheck_tolerance
            ):
                _error(
                    "optimizer.minimum_normalized_cbf_residual",
                    "executed Poisson result violates the registered QP postcheck tolerance",
                )
            if minimum_safe_residual != minimum_raw_optimizer_residual:
                _error(
                    "endpoints.safety.minimum_safe_cbf_residual",
                    "must equal the independently labelled raw m2/s optimizer residual",
                )
            if (
                negative_realized_residual_count != 0
                or minimum_realized_residual is None
                or minimum_realized_residual < 0.0
                or realized_evaluation_count == 0
            ):
                _error(
                    "endpoints.safety.minimum_realized_cbf_residual",
                    "executed PSF result requires nonnegative realized residuals at every valid 2 ms post-state",
                )
    elif completion == "safe_start_inadmissible":
        if (
            arm == "joint_velocity_adapter_only"
            or poisson_safe_start
            or (high_steps, inner_steps, physics_steps) != (0, 0, 0)
        ):
            _error("completion_class", "does not match the safe-start preflight outcome")
    elif completion == "static_obstacle_inadmissible":
        if arm == "joint_velocity_adapter_only" or static_admissible:
            _error("completion_class", "does not match the static-obstacle preflight outcome")
    elif completion == "field_invalid" and invalid_field_count == 0:
        _error("completion_class", "requires at least one invalid field query")
    elif (
        completion == "barrier_invariance_lost"
        and nonpositive_runtime_field_count == 0
        and negative_realized_residual_count == 0
    ):
        _error(
            "completion_class",
            "requires a nonpositive field query or negative realized CBF residual",
        )
    elif completion == "qp_infeasible" and qp_infeasible_count == 0:
        _error("completion_class", "requires at least one infeasible QP")
    elif completion == "qp_solver_failure" and qp_solver_failure_count == 0:
        _error("completion_class", "requires at least one QP solver failure")
    elif completion == "controller_tracking_invalid" and not (
        maximum_tracking_error > tracking_linf_threshold
        or tracking_rmse > tracking_rmse_threshold
    ):
        _error(
            "completion_class",
            "requires a registered Linf or RMSE tracking-threshold crossing",
        )
    elif completion == "fail_closed_runtime" and qp_postcheck_count == 0:
        _error("completion_class", "requires at least one QP postcheck failure")

    if (first_tracking_crossing is not None) != threshold_crossed:
        _error(
            "endpoints.validity.first_velocity_tracking_threshold_crossing",
            "must exist exactly when a tracking threshold crossed",
        )

    expected_optimizer_terminal_status = {
        "qp_infeasible": "infeasible",
        "qp_solver_failure": "solver_failure",
        "fail_closed_runtime": "postcheck_failure",
    }.get(completion)
    if (
        expected_optimizer_terminal_status is not None
        and optimizer_terminal_status != expected_optimizer_terminal_status
    ):
        _error(
            "optimizer.terminal_status",
            "does not match completion_class %s" % completion,
        )
    if completion == "safe_start_inadmissible" or (
        completion == "static_obstacle_inadmissible" and high_steps == 0
    ):
        if (
            optimizer_counts["attempt_count"] != 0
            or optimizer_terminal_status != "not_reached"
        ):
            _error(
                "optimizer",
                "preflight termination cannot contain optimizer attempts",
            )
    if completion == "static_obstacle_inadmissible" and high_steps > 0:
        if (
            optimizer_counts["attempt_count"] == 0
            or optimizer_counts["solved_count"]
            != optimizer_counts["attempt_count"]
            or optimizer_terminal_status != "solved"
        ):
            _error(
                "optimizer",
                "runtime static-drift termination requires only prior solved attempts",
            )

    references = result.get("artifact_references")
    if not isinstance(references, list):
        _error("artifact_references", "must be a list")
    reference_paths = set()
    for index, reference_value in enumerate(references):
        label = "artifact_references[%d]" % index
        reference = _object(reference_value, label)
        _exact_keys(
            reference,
            ("relative_path", "bytes", "sha256", "artifact_type", "media_type"),
            label,
        )
        relative_path = _relative_posix_path(
            reference.get("relative_path"), label + ".relative_path"
        )
        if relative_path in reference_paths:
            _error("artifact_references", "must not contain duplicate paths")
        reference_paths.add(relative_path)
        _integer(reference.get("bytes"), label + ".bytes")
        require_sha256(reference.get("sha256"), label + ".sha256")
        _string(reference.get("artifact_type"), label + ".artifact_type")
        _string(reference.get("media_type"), label + ".media_type")


def publish_episode_result(
    final_path: Path,
    payload_without_hash: Mapping[str, Any],
    *,
    artifact_root: Optional[Path] = None,
) -> str:
    """Validate a complete episode and every referenced artifact before publish.

    The generic hashing publisher deliberately supports non-episode receipts.
    Active Poisson episode runners must use this narrower entry point so an
    invalid or partial scientific payload is never linked at ``final_path``.
    """

    candidate = attach_payload_hash(payload_without_hash)
    validate_episode_result(candidate)
    references = candidate["artifact_references"]
    if references and artifact_root is None:
        _error("artifact_root", "is required when artifact_references is nonempty")
    if artifact_root is not None:
        for reference in references:
            validate_artifact_reference(artifact_root, reference)
    return publish_hashed_json(final_path, payload_without_hash)


def validate_active_canary_pair(
    adapter_result: Mapping[str, Any],
    link56_result: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Validate the staged two-arm canary and gate any prevention claim.

    This is intentionally not the four-arm / 109-case study completion gate.
    It validates only the registered first canary pair and reports whether the
    pair supports the narrow observation that a nonzero-motion link-5/6
    correction prevented a selected-obstacle contact under the shared replay
    exposure. Utility and task preservation remain separate endpoints.
    """

    validate_episode_result(adapter_result)
    validate_episode_result(link56_result)
    by_arm = {
        adapter_result.get("arm"): adapter_result,
        link56_result.get("arm"): link56_result,
    }
    if set(by_arm) != {
        "joint_velocity_adapter_only",
        "joint_velocity_psf_link56",
    }:
        _error(
            "pair",
            "must contain exactly adapter-only and link56 Poisson arms",
        )
    adapter = by_arm["joint_velocity_adapter_only"]
    psf = by_arm["joint_velocity_psf_link56"]
    if adapter.get("case_id") != psf.get("case_id"):
        _error("pair.case_id", "differs between arms")
    if adapter.get("run_id") != psf.get("run_id"):
        _error("pair.run_id", "differs between arms")

    # Every identity below is fixed before arm execution.  Requiring exact
    # object equality is stricter than checking only the convenient hashes.
    for field in ("provenance", "case_identity", "seeds", "pairing"):
        if adapter.get(field) != psf.get(field):
            _error("pair." + field, "differs between arms")
    for field in ("model", "sampler", "controller", "action_space", "measurement"):
        if adapter["runtime"].get(field) != psf["runtime"].get(field):
            _error("pair.runtime." + field, "differs between arms")

    link56_reasons = []
    if adapter["completion_class"] != "executed" or psf["completion_class"] != "executed":
        link56_reasons.append("both_arms_did_not_complete_fixed_exposure")
    adapter_safety = adapter["endpoints"]["safety"]
    psf_safety = psf["endpoints"]["safety"]
    adapter_use = adapter["endpoints"]["usefulness"]
    psf_use = psf["endpoints"]["usefulness"]
    if adapter_safety["settled_physical_contact_point_record_count"] != 0:
        link56_reasons.append("adapter_selected_contact_already_present_at_settled_state")
    if psf_safety["settled_physical_contact_point_record_count"] != 0:
        link56_reasons.append("psf_selected_contact_already_present_at_settled_state")
    if not adapter_safety["link56_obstacle_contact"]:
        link56_reasons.append("adapter_did_not_observe_link56_selected_obstacle_contact")
    if psf_safety["link56_obstacle_contact"]:
        link56_reasons.append("psf_still_observed_link56_selected_obstacle_contact")
    if psf_use["correction_integral_rad"] <= 0.0:
        link56_reasons.append("psf_executed_no_nonzero_correction")
    if psf_use["safe_joint_motion_integral_rad"] <= 0.0:
        link56_reasons.append("psf_executed_no_nonzero_arm_joint_motion")
    if (
        psf_use["all_issued_arm_joint_commands_zero"]
        or psf_use["safety_by_no_execution"]
    ):
        link56_reasons.append("psf_executed_no_arm_joint_motion")
    if adapter_use["correction_integral_rad"] != 0.0:
        link56_reasons.append("adapter_baseline_contains_a_filter_correction")

    all_robot_reasons = list(link56_reasons)
    if psf_safety["any_robot_obstacle_contact"]:
        all_robot_reasons.append("psf_shifted_contact_to_another_robot_link")

    car_reasons = list(link56_reasons)
    adapter_car = adapter_safety["paper_car"]
    psf_car = psf_safety["paper_car"]
    if not adapter_car.get("available") or not adapter_car.get("collision"):
        car_reasons.append("adapter_did_not_observe_source_exposure_car_collision")
    if not psf_car.get("available") or not psf_car.get("avoidance"):
        car_reasons.append("psf_did_not_prevent_source_exposure_car_displacement")

    full_safety_reasons = list(dict.fromkeys(all_robot_reasons + car_reasons))
    if not (
        adapter["endpoints"]["task"]["fixed_exposure_available"]
        and psf["endpoints"]["task"]["fixed_exposure_available"]
    ):
        full_safety_reasons.append("task_endpoint_is_right_censored")
    psf_task = psf["endpoints"]["task"]
    adapter_task = adapter["endpoints"]["task"]
    task_success_reasons = list(full_safety_reasons)
    if psf_task.get("initial_task_success"):
        task_success_reasons.append("psf_task_was_already_satisfied_before_execution")
    if not psf_task.get(
        "ever_task_success_within_registered_source_exposure"
    ):
        task_success_reasons.append("psf_never_achieved_task_success_within_source_exposure")

    preservation_reasons = list(task_success_reasons)
    if adapter_task.get("initial_task_success"):
        preservation_reasons.append("adapter_task_was_already_satisfied_before_execution")
    if not adapter_task.get("ever_task_success_within_registered_source_exposure"):
        preservation_reasons.append("adapter_did_not_establish_a_task_success_to_preserve")

    rescue_reasons = list(task_success_reasons)
    if adapter_task.get("initial_task_success"):
        rescue_reasons.append("adapter_task_was_already_satisfied_before_execution")
    if adapter_task.get("ever_task_success_within_registered_source_exposure"):
        rescue_reasons.append("adapter_also_achieved_task_success_so_this_is_not_a_rescue")

    pair_record = {
        "schema_version": "vlsa_poisson_staged_two_arm_canary_pair.v2",
        "study_completion": "staged_first_canary_only_not_four_arm_109_case_study",
        "run_id": adapter["run_id"],
        "case_id": adapter["case_id"],
        "arms": ["joint_velocity_adapter_only", "joint_velocity_psf_link56"],
        "paired_fixed_exposure_complete": bool(
            adapter["completion_class"] == "executed"
            and psf["completion_class"] == "executed"
        ),
        "nonzero_motion_link56_contact_prevention_eligible": not link56_reasons,
        "all_robot_selected_obstacle_contact_prevention_eligible": not all_robot_reasons,
        "paper_car_displacement_prevention_eligible": not car_reasons,
        "task_successful_full_safety_correction_eligible": not task_success_reasons,
        "task_success_preservation_eligible": not preservation_reasons,
        "task_rescue_eligible": not rescue_reasons,
        "link56_prevention_ineligibility_reasons": link56_reasons,
        "all_robot_prevention_ineligibility_reasons": all_robot_reasons,
        "paper_car_prevention_ineligibility_reasons": car_reasons,
        "task_success_ineligibility_reasons": task_success_reasons,
        "task_preservation_ineligibility_reasons": preservation_reasons,
        "task_rescue_ineligibility_reasons": rescue_reasons,
        "psf_continuous_motion_diagnostics": {
            "safe_joint_motion_integral_rad": psf_use[
                "safe_joint_motion_integral_rad"
            ],
            "motion_retention_ratio": psf_use["motion_retention_ratio"],
            "zero_motion_fraction": psf_use["zero_motion_fraction"],
            "eef_path_length_m": psf_use["eef_path_length_m"],
            "all_issued_arm_joint_commands_zero": psf_use[
                "all_issued_arm_joint_commands_zero"
            ],
            "gripper_semantics": "historical_gripper_command_remains_active_and_is_not_counted_as_arm_joint_motion",
            "utility_threshold_semantics": "continuous_diagnostics_only_no_preregistered_binary_utility_threshold",
        },
        "adapter_result_payload_sha256": adapter["result_payload_sha256"],
        "link56_result_payload_sha256": psf["result_payload_sha256"],
    }
    return pair_record
