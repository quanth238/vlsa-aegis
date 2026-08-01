#!/usr/bin/env python3
"""Allocation-only SafeLIBERO static-Poisson shadow identification.

This stage is deliberately non-interventional.  It replays the first frozen
canary's 237 historical AEGIS Cartesian actions through the unchanged released
OSC environment.  A post-integration callback measures authoritative MuJoCo
contact and evaluates the settled static link-5/6 Poisson field, but it never
changes an action, control, model, or live simulator state.

An already-passed exact shadow-parity artifact from the same clean commit is a
required input.  The final result is useful only for identifying whether the
registered field signal precedes the known link contact; it is not evidence of
active collision avoidance or task-preserving correction.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import socket
import sys
import time
import traceback
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


SCHEMA_VERSION = "vlsa_poisson_shadow_identification.v4"
DEFAULT_CASE_ID = "vlsa-t1-goal-ii-t0-e05"
EXPECTED_PARITY_SCHEMA = "vlsa_poisson_shadow_parity.v3"
CONTACT_DEFINITION = "mujoco_contact_dist_le_0"
INNER_UPDATES_PER_HIGH_LEVEL_ACTION = 5
PHYSICS_SUBSTEPS_PER_INNER_UPDATE = 5
PHYSICS_SUBSTEPS_PER_HIGH_LEVEL_ACTION = (
    INNER_UPDATES_PER_HIGH_LEVEL_ACTION * PHYSICS_SUBSTEPS_PER_INNER_UPDATE
)
DIFFERENTIAL_AUDIT_FAILURE_EVIDENCE_SCHEMA = (
    "vlsa_poisson_shadow_identification_differential_audit_failure_evidence.v2"
)
DIFFERENTIAL_AUDIT_FAILURE_PHASE = (
    "settled_link56_protected_sample_differential_audit_validation"
)


class ShadowIdentificationRunnerError(RuntimeError):
    """A required allocation, pairing, or exact-replay invariant failed."""


class ShadowIdentificationDifferentialAuditFailure(
    ShadowIdentificationRunnerError
):
    """A complete, deeply validated differential audit genuinely failed."""

    def __init__(self, failure_evidence: Mapping[str, Any]) -> None:
        super().__init__(
            "settled link-5/6 protected-sample differential audit did not pass"
        )
        self.phase = DIFFERENTIAL_AUDIT_FAILURE_PHASE
        self.failure_evidence = dict(failure_evidence)


def _differential_audit_failure_evidence(
    audit: Mapping[str, Any], validation_receipt: Mapping[str, Any]
) -> Dict[str, Any]:
    """Retain one genuine failed audit without making it success evidence."""

    receipt_keys = {
        "schema_version",
        "audit_payload_sha256",
        "ordered_sample_identity_sha256",
        "integration_state_sha256",
        "specification_sha256",
        "binding_sha256",
        "classification_ledger_sha256",
        "counts",
        "passed",
    }
    if not isinstance(audit, Mapping) or not isinstance(
        validation_receipt, Mapping
    ):
        raise ShadowIdentificationRunnerError(
            "differential-audit failure evidence must contain mappings"
        )
    if set(validation_receipt) != receipt_keys:
        raise ShadowIdentificationRunnerError(
            "differential-audit failed validation receipt has invalid keys"
        )
    integration_state = audit.get("integration_state")
    matching_fields = (
        ("schema_version", "schema_version"),
        ("audit_payload_sha256", "audit_payload_sha256"),
        ("ordered_sample_identity_sha256", "ordered_sample_identity_sha256"),
        ("specification_sha256", "specification_sha256"),
        ("binding_sha256", "binding_sha256"),
        ("classification_ledger_sha256", "classification_ledger_sha256"),
        ("counts", "counts"),
        ("passed", "passed"),
    )
    if any(
        validation_receipt[receipt_key] != audit.get(audit_key)
        for receipt_key, audit_key in matching_fields
    ) or not isinstance(integration_state, Mapping) or validation_receipt[
        "integration_state_sha256"
    ] != integration_state.get("source_initial_sha256"):
        raise ShadowIdentificationRunnerError(
            "differential-audit failed validation receipt differs from audit"
        )
    if validation_receipt["passed"] is not False:
        raise ShadowIdentificationRunnerError(
            "differential-audit failure evidence requires a failed validation receipt"
        )
    return {
        "schema_version": DIFFERENTIAL_AUDIT_FAILURE_EVIDENCE_SCHEMA,
        "phase": DIFFERENTIAL_AUDIT_FAILURE_PHASE,
        "failure_kind": "registered_protected_sample_differential_audit_failed",
        "scientific_result": False,
        "active_physics_authorized": False,
        "settled_link56_differential_audit": dict(audit),
        "settled_link56_differential_audit_validation": dict(
            validation_receipt
        ),
        "interpretation": (
            "complete registered numerical audit failure; this is retained "
            "diagnostic evidence and cannot authorize shadow interpretation "
            "or active physics"
        ),
    }


def _failure_authority_binding(provenance: Mapping[str, Any]) -> Dict[str, Any]:
    """Select the source, immutable-input, and Slurm identities for a failure."""

    source = provenance.get("source")
    if (
        not isinstance(source, Mapping)
        or set(source) != {"commit", "branch", "status_short"}
        or not isinstance(source.get("commit"), str)
        or len(source["commit"]) != 40
        or any(character not in "0123456789abcdef" for character in source["commit"])
        or not isinstance(source.get("branch"), str)
        or not source["branch"]
        or source.get("status_short") != []
    ):
        raise ShadowIdentificationRunnerError(
            "failure evidence lacks exact clean source provenance"
        )
    hash_fields = (
        "manifest_sha256",
        "manifest_row_sha256",
        "selection_config_sha256",
        "runtime_protocol_raw_sha256",
        "runtime_protocol_semantic_sha256",
        "runtime_parameter_block_sha256",
        "historical_result_file_sha256",
        "historical_result_payload_sha256",
        "upstream_parity_payload_sha256",
    )
    required = hash_fields + (
        "host",
        "slurm_job_id",
        "slurm_job_name",
    )
    if any(
        not isinstance(provenance.get(key), str)
        or len(provenance[key]) != 64
        or any(
            character not in "0123456789abcdef"
            for character in provenance[key]
        )
        for key in hash_fields
    ) or any(
        not isinstance(provenance.get(key), str) or not provenance[key]
        for key in ("host", "slurm_job_name")
    ) or (
        not isinstance(provenance.get("slurm_job_id"), str)
        or not provenance["slurm_job_id"].isdigit()
    ):
        raise ShadowIdentificationRunnerError(
            "failure evidence has invalid immutable input or Slurm provenance"
        )
    return {
        "source": dict(source),
        **{key: provenance[key] for key in required},
    }


def _record_failed_result(
    payload: Dict[str, Any], error: Exception, traceback_text: str
) -> None:
    """Record a terminal failure, preserving validated numerical evidence."""

    payload["status"] = "failed"
    payload["scientific_result"] = False
    if isinstance(error, ShadowIdentificationDifferentialAuditFailure):
        failure_evidence = dict(error.failure_evidence)
        provenance = payload.get("provenance")
        if not isinstance(provenance, Mapping):
            raise ShadowIdentificationRunnerError(
                "differential-audit failure occurred before provenance was bound"
            )
        failure_evidence["authority_binding"] = _failure_authority_binding(
            provenance
        )
        payload["phase"] = error.phase
        payload["failure_evidence"] = failure_evidence
    payload["failure"] = {
        "type": type(error).__name__,
        "message": str(error),
        "traceback": traceback_text,
    }


def _package_versions(names: Sequence[str]) -> Dict[str, Any]:
    result = {}
    for name in names:
        try:
            result[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            result[name] = None
    return result


def _load_json_object(path: Path, label: str) -> Dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ShadowIdentificationRunnerError("%s is missing or symlinked" % label)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ShadowIdentificationRunnerError("%s is invalid JSON" % label) from error
    if not isinstance(payload, dict):
        raise ShadowIdentificationRunnerError("%s must contain one JSON object" % label)
    return payload


def _model_body_id(model: Any, name: str) -> int:
    method = getattr(model, "body_name2id", None)
    if method is not None:
        try:
            value = int(method(name))
        except Exception as error:
            raise ShadowIdentificationRunnerError(
                "required MuJoCo body %s is absent" % name
            ) from error
    else:
        mujoco = importlib.import_module("mujoco")
        raw_model = getattr(model, "_model", model)
        value = int(
            mujoco.mj_name2id(raw_model, mujoco.mjtObj.mjOBJ_BODY, name)
        )
    if value < 0:
        raise ShadowIdentificationRunnerError(
            "required MuJoCo body %s is absent" % name
        )
    return value


def _official_integration_state(sim: Any, np: Any) -> Any:
    """Read MuJoCo's complete official integration state without forwarding."""

    mujoco = importlib.import_module("mujoco")
    model_candidate = getattr(sim, "model", None)
    data_candidate = getattr(sim, "data", None)
    model = getattr(model_candidate, "_model", model_candidate)
    data = getattr(data_candidate, "_data", data_candidate)
    if not isinstance(model, mujoco.MjModel) or not isinstance(data, mujoco.MjData):
        raise ShadowIdentificationRunnerError(
            "simulator does not expose official MuJoCo model/data"
        )
    specification = int(mujoco.mjtState.mjSTATE_INTEGRATION)
    state = np.empty(
        int(mujoco.mj_stateSize(model, specification)), dtype=np.float64
    )
    mujoco.mj_getState(model, data, state, specification)
    if state.ndim != 1 or state.size <= 0 or not np.all(np.isfinite(state)):
        raise ShadowIdentificationRunnerError(
            "official MuJoCo integration state is empty or non-finite"
        )
    return state


def _load_bound_runtime_protocol(
    *,
    root: Path,
    case: Mapping[str, Any],
    selection_config_path: Path,
) -> Tuple[Mapping[str, Any], Any, Mapping[str, Any], Path]:
    from main.poisson_fullbody.feasibility_protocol import load_feasibility_protocol
    from scripts.run_poisson_shadow_parity import _file_sha256

    selection = _load_json_object(selection_config_path, "selection protocol")
    if _file_sha256(selection_config_path) != case.get("protocol_config_sha256"):
        raise ShadowIdentificationRunnerError(
            "case selection-protocol SHA-256 binding differs"
        )
    runtime_record = selection.get("runtime_protocol")
    if not isinstance(runtime_record, dict):
        raise ShadowIdentificationRunnerError(
            "selection protocol lacks runtime_protocol binding"
        )
    relative = runtime_record.get("relative_path")
    if (
        not isinstance(relative, str)
        or not relative
        or Path(relative).is_absolute()
        or ".." in Path(relative).parts
    ):
        raise ShadowIdentificationRunnerError("runtime protocol path is invalid")
    runtime_path = root / relative
    if _file_sha256(runtime_path) != runtime_record.get("raw_file_sha256"):
        raise ShadowIdentificationRunnerError("runtime protocol raw SHA-256 differs")
    protocol, hashes = load_feasibility_protocol(
        runtime_path,
        expected_protocol_sha256=runtime_record.get(
            "semantic_protocol_sha256"
        ),
    )
    if (
        hashes.parameter_block_sha256
        != runtime_record.get("parameter_block_sha256")
    ):
        raise ShadowIdentificationRunnerError(
            "runtime protocol parameter-block binding differs"
        )
    return protocol, hashes, selection, runtime_path


def _require_upstream_parity(
    parity_path: Path,
    *,
    case_id: str,
    source_commit: str,
    historical_payload_sha256: str,
    manifest_sha256: str,
    manifest_row_sha256: str,
    expected_callback_count: int,
    historical_provenance: Mapping[str, Any],
    expected_action_state_sha256_ledger: Sequence[str],
) -> Mapping[str, Any]:
    from main.poisson_fullbody.contracts import load_hashed_json
    from scripts.run_poisson_shadow_parity import _canonical, _sha256

    def valid_sha256(value: Any) -> bool:
        return (
            isinstance(value, str)
            and len(value) == 64
            and all(character in "0123456789abcdef" for character in value)
        )

    parity = load_hashed_json(parity_path)
    if (
        parity.get("schema_version") != EXPECTED_PARITY_SCHEMA
        or parity.get("status") != "passed"
        or parity.get("scientific_result") is not False
        or parity.get("case_id") != case_id
    ):
        raise ShadowIdentificationRunnerError(
            "upstream exact-parity artifact has wrong schema, status, or case"
        )
    provenance = parity.get("provenance")
    ordinary = parity.get("ordinary_replay")
    callback = parity.get("callback_replay")
    acceptance = parity.get("acceptance")
    if (
        not isinstance(provenance, dict)
        or not isinstance(ordinary, dict)
        or not isinstance(callback, dict)
    ):
        raise ShadowIdentificationRunnerError("upstream exact-parity provenance is incomplete")
    if not isinstance(acceptance, dict) or not all(
        acceptance.get(field) is True
        for field in (
            "all_historical_post_step_states_exact",
            "ordinary_and_callback_states_exact",
            "ordinary_and_callback_observations_exact",
            "ordinary_and_callback_boundary_0_mjstate_integration_exact",
            "reward_done_goal_exact",
            "full_callback_exposure",
            "ordinary_step_path_unmodified",
        )
    ):
        raise ShadowIdentificationRunnerError("upstream exact-parity acceptance is incomplete")
    source = provenance.get("source")
    if (
        not isinstance(source, dict)
        or source.get("commit") != source_commit
        or source.get("status_short") != []
    ):
        raise ShadowIdentificationRunnerError(
            "upstream exact-parity source is not this exact clean commit"
        )
    if (
        provenance.get("historical_result_payload_sha256")
        != historical_payload_sha256
        or provenance.get("manifest_sha256") != manifest_sha256
        or provenance.get("manifest_row_sha256") != manifest_row_sha256
    ):
        raise ShadowIdentificationRunnerError(
            "upstream exact-parity evidence binding differs"
        )
    if parity.get("historical") != dict(historical_provenance):
        raise ShadowIdentificationRunnerError(
            "upstream exact-parity historical replay identity differs"
        )
    if (
        isinstance(expected_callback_count, bool)
        or not isinstance(expected_callback_count, int)
        or expected_callback_count <= 0
        or expected_callback_count
        % PHYSICS_SUBSTEPS_PER_HIGH_LEVEL_ACTION
        != 0
    ):
        raise ShadowIdentificationRunnerError(
            "expected exact-parity callback count is invalid"
        )
    expected_action_count = (
        expected_callback_count // PHYSICS_SUBSTEPS_PER_HIGH_LEVEL_ACTION
    )
    if (
        not isinstance(expected_action_state_sha256_ledger, (list, tuple))
        or len(expected_action_state_sha256_ledger) != expected_action_count
        or not all(
            valid_sha256(value)
            for value in expected_action_state_sha256_ledger
        )
    ):
        raise ShadowIdentificationRunnerError(
            "historical action-boundary state ledger is invalid"
        )
    expected_action_states = list(expected_action_state_sha256_ledger)
    expected_state_sequence_sha256 = _sha256(
        _canonical(expected_action_states)
    )
    if (
        historical_provenance.get("terminal_simulator_state_sha256")
        != expected_action_states[-1]
    ):
        raise ShadowIdentificationRunnerError(
            "historical terminal state differs from its action-state ledger"
        )
    if (
        ordinary.get("executed_action_count") != expected_action_count
        or ordinary.get("expected_state_match_count")
        != expected_action_count
        or callback.get("executed_action_count") != expected_action_count
        or callback.get("callback_count") != expected_callback_count
        or callback.get("expected_callback_count") != expected_callback_count
    ):
        raise ShadowIdentificationRunnerError(
            "upstream exact-parity callback exposure differs"
        )
    if (
        ordinary.get("action_boundary_state_sha256_ledger")
        != expected_action_states
        or callback.get("action_boundary_state_sha256_ledger")
        != expected_action_states
        or ordinary.get("state_sequence_sha256")
        != expected_state_sequence_sha256
        or callback.get("state_sequence_sha256")
        != expected_state_sequence_sha256
        or ordinary.get("terminal_simulator_state_sha256")
        != expected_action_states[-1]
        or callback.get("terminal_simulator_state_sha256")
        != expected_action_states[-1]
    ):
        raise ShadowIdentificationRunnerError(
            "upstream exact-parity action-boundary states differ from history"
        )
    for field in (
        "state_sequence_sha256",
        "observation_sequence_sha256",
        "terminal_simulator_state_sha256",
    ):
        if (
            not valid_sha256(ordinary.get(field))
            or ordinary.get(field) != callback.get(field)
        ):
            raise ShadowIdentificationRunnerError(
                "upstream ordinary/callback parity differs at %s" % field
            )
    official_ledger = callback.get(
        "official_integration_state_sha256_ledger"
    )
    ordinary_settled_official_state = ordinary.get(
        "settled_official_integration_state"
    )
    callback_settled_official_state = callback.get(
        "settled_official_integration_state"
    )
    if (
        not isinstance(ordinary_settled_official_state, dict)
        or set(ordinary_settled_official_state)
        != {
            "physical_boundary",
            "mujoco_state_specification",
            "state_vector_length",
            "sha256",
        }
        or isinstance(
            ordinary_settled_official_state.get("physical_boundary"), bool
        )
        or not isinstance(
            ordinary_settled_official_state.get("physical_boundary"), int
        )
        or ordinary_settled_official_state.get("physical_boundary") != 0
        or ordinary_settled_official_state.get("mujoco_state_specification")
        != "mjSTATE_INTEGRATION"
        or isinstance(
            ordinary_settled_official_state.get("state_vector_length"), bool
        )
        or not isinstance(
            ordinary_settled_official_state.get("state_vector_length"), int
        )
        or ordinary_settled_official_state.get("state_vector_length") <= 0
        or not valid_sha256(ordinary_settled_official_state.get("sha256"))
        or ordinary_settled_official_state != callback_settled_official_state
    ):
        raise ShadowIdentificationRunnerError(
            "upstream exact-parity settled official integration state differs"
        )
    if (
        not isinstance(official_ledger, list)
        or len(official_ledger) != expected_callback_count
        or callback.get("official_integration_state_count")
        != expected_callback_count
        or not all(valid_sha256(value) for value in official_ledger)
        or callback.get("official_integration_state_sequence_sha256")
        != _sha256(_canonical(official_ledger))
    ):
        raise ShadowIdentificationRunnerError(
            "upstream exact-parity official integration-state ledger differs"
        )
    first_substep = callback.get("first_substep")
    last_substep = callback.get("last_substep")
    substep_trace = callback.get("substep_trace")
    if (
        not isinstance(substep_trace, list)
        or len(substep_trace) != expected_action_count
        or not isinstance(first_substep, dict)
        or not isinstance(last_substep, dict)
    ):
        raise ShadowIdentificationRunnerError(
            "upstream exact-parity callback trace is incomplete"
        )
    for high_level_index, raw_action_trace in enumerate(substep_trace):
        if (
            not isinstance(raw_action_trace, list)
            or len(raw_action_trace)
            != PHYSICS_SUBSTEPS_PER_HIGH_LEVEL_ACTION
        ):
            raise ShadowIdentificationRunnerError(
                "upstream exact-parity callback action trace differs"
            )
        for substep_index, raw_substep in enumerate(raw_action_trace):
            callback_index = (
                high_level_index * PHYSICS_SUBSTEPS_PER_HIGH_LEVEL_ACTION
                + substep_index
            )
            if (
                not isinstance(raw_substep, dict)
                or set(raw_substep)
                != {
                    "substep",
                    "official_mjstate_integration_sha256",
                }
                or raw_substep.get("substep") != substep_index
                or raw_substep.get(
                    "official_mjstate_integration_sha256"
                )
                != official_ledger[callback_index]
            ):
                raise ShadowIdentificationRunnerError(
                    "upstream exact-parity callback trace differs at %d"
                    % callback_index
                )
    if (
        callback.get("substep_trace_sha256")
        != _sha256(_canonical(substep_trace))
        or first_substep != substep_trace[0][0]
        or last_substep != substep_trace[-1][-1]
    ):
        raise ShadowIdentificationRunnerError(
            "upstream exact-parity callback trace hash or endpoints differ"
        )
    return parity


def _first_link56_contact(
    measurement: Any, link56_geom_ids: Sequence[int]
) -> Optional[Dict[str, Any]]:
    link_geoms = set(int(value) for value in link56_geom_ids)
    records = list(measurement.settled_state.physical_contact_point_records)
    records.extend(
        record
        for record in measurement.live_solver_phase_contact_point_records
        if record.is_physical_nonpositive_distance_contact
    )
    records.extend(measurement.post_state_physical_contact_point_records)
    records = [record for record in records if record.robot_geom_id in link_geoms]
    if not records:
        return None

    def order(record: Any) -> Tuple[int, int, int]:
        observation = (
            -1 if record.observation_index is None else int(record.observation_index)
        )
        phase = {
            "settled_post_integration_recomputed": 0,
            "live_solver_phase_preintegration_geometry": 1,
            "post_integration_recomputed": 2,
        }.get(str(record.source_phase), 3)
        return observation, phase, int(record.mujoco_contact_index)

    return min(records, key=order).to_dict()


def _require_complete_measurement_exposure(
    measurement: Any, *, action_count: int
) -> int:
    """Validate the monitor's typed three-level callback cadence.

    ``FullRobotObstacleMeasurement.first_index`` and ``last_index`` are
    ``(high_level, inner_control, physics_substep)`` tuples.  They are not
    scalar global callback indices.  Keep this check executable and separate
    from the expensive replay so a type/semantic regression fails in unit
    tests instead of after the final H100 callback.
    """

    if (
        isinstance(action_count, bool)
        or not isinstance(action_count, int)
        or action_count <= 0
    ):
        raise ShadowIdentificationRunnerError(
            "measurement exposure action_count must be a positive integer"
        )
    expected_count = action_count * PHYSICS_SUBSTEPS_PER_HIGH_LEVEL_ACTION
    expected_first = (0, 0, 0)
    expected_last = (
        action_count - 1,
        INNER_UPDATES_PER_HIGH_LEVEL_ACTION - 1,
        PHYSICS_SUBSTEPS_PER_INNER_UPDATE - 1,
    )
    observed_count = measurement.observed_physics_substeps
    if isinstance(observed_count, bool) or not isinstance(observed_count, int):
        raise ShadowIdentificationRunnerError(
            "MuJoCo contact measurement callback count is not an integer"
        )
    first_index = measurement.first_index
    last_index = measurement.last_index
    if not all(
        isinstance(index, tuple)
        and len(index) == 3
        and all(type(value) is int for value in index)
        for index in (first_index, last_index)
    ):
        raise ShadowIdentificationRunnerError(
            "MuJoCo contact measurement cadence indices must be strict "
            "(high_level, inner_control, physics_substep) integer tuples"
        )
    if (
        observed_count != expected_count
        or first_index != expected_first
        or last_index != expected_last
    ):
        raise ShadowIdentificationRunnerError(
            "MuJoCo contact measurement did not cover every physics callback: "
            "expected count=%d first=%r last=%r; observed count=%r first=%r last=%r"
            % (
                expected_count,
                expected_first,
                expected_last,
                measurement.observed_physics_substeps,
                first_index,
                last_index,
            )
        )
    return expected_count


def _prepare_shadow_runtime(
    *,
    evaluator: Any,
    runtime: Mapping[str, Any],
    case: Mapping[str, Any],
    replay: Any,
    protocol: Mapping[str, Any],
    protocol_hashes: Any,
) -> Tuple[Any, Any, Mapping[str, Any], Sequence[Sequence[str]], Sequence[bool], Any, Any, Any]:
    from main.poisson_fullbody.field_bundle import build_static_field_bundle
    from main.poisson_fullbody.jacobians import (
        DifferentialAuditError,
        audit_protected_sample_differentials,
        inspect_protected_sample_differential_audit,
    )
    from main.poisson_fullbody.measurement import (
        FullRobotObstacleMonitor,
        clone_forwarded_state,
        resolve_collision_geom_sets,
    )
    from main.poisson_fullbody.robot_samples import validate_rigid_roundtrip
    from main.poisson_fullbody.shadow_identification import (
        StaticDriftThresholds,
        StaticPoissonShadowObserver,
    )
    from main.poisson_fullbody.surface_sampling import (
        build_robot_collision_samples,
        validate_robot_sample_evidence,
    )
    from scripts.run_poisson_shadow_parity import _prepare_environment

    env, task, observation, goal_atoms, previous_goal = _prepare_environment(
        evaluator, runtime, case, replay
    )
    construction_state_before = _official_integration_state(
        env.sim, runtime["np"]
    )
    obstacle_name, _ = evaluator._active_obstacle(env, observation)
    if obstacle_name != case.get("active_obstacle_name"):
        env.close()
        raise ShadowIdentificationRunnerError("active selected obstacle differs from manifest")
    authority = evaluator._contact_model_authority(env, obstacle_name)
    raw_model = getattr(env.sim.model, "_model", env.sim.model)
    raw_data = getattr(env.sim.data, "_data", env.sim.data)
    if len(env.robots) != 1:
        env.close()
        raise ShadowIdentificationRunnerError(
            "shadow identification requires exactly one robot"
        )
    robot_root_name = getattr(env.robots[0].robot_model, "root_body", None)
    if not isinstance(robot_root_name, str) or not robot_root_name:
        env.close()
        raise ShadowIdentificationRunnerError(
            "robosuite robot model does not expose its authoritative root body"
        )
    robot_roots = (_model_body_id(env.sim.model, robot_root_name),)
    if robot_roots[0] not in set(int(value) for value in authority["robot_body_ids"]):
        env.close()
        raise ShadowIdentificationRunnerError(
            "robosuite robot root is absent from frozen contact-model authority"
        )
    link_ids = tuple(
        _model_body_id(env.sim.model, name)
        for name in protocol["claim_scope"]["protected_robot_bodies"]
    )
    resolved = resolve_collision_geom_sets(
        raw_model,
        robot_root_body_ids=robot_roots,
        obstacle_root_body_ids=(int(authority["active_obstacle_root_body_id"]),),
        link56_body_ids=link_ids,
    )
    bundle = build_static_field_bundle(
        raw_model,
        raw_data,
        resolved=resolved,
        protocol=protocol,
        protocol_hashes=protocol_hashes,
    )
    forwarded = clone_forwarded_state(raw_model, raw_data)
    full_samples = build_robot_collision_samples(
        env.sim.model,
        forwarded,
        geom_ids=resolved.robot_geom_ids,
        epsilon_m=float(protocol["coverage"]["epsilon_m"]),
    )
    sampled_geom_ids = tuple(
        int(record["geom_id"]) for record in full_samples.geom_records
    )
    if sampled_geom_ids != tuple(int(value) for value in resolved.robot_geom_ids):
        env.close()
        raise ShadowIdentificationRunnerError(
            "full-robot measurement samples differ from authoritative resolved geoms"
        )
    roundtrip = validate_rigid_roundtrip(full_samples.samples, forwarded)
    if not roundtrip["passed"]:
        env.close()
        raise ShadowIdentificationRunnerError("full-robot sample transform audit failed")
    full_sampling_evidence = {
        "sample_count": len(full_samples.samples),
        "sample_ledger_sha256": full_samples.sample_ledger_sha256,
        "geom_records": list(full_samples.geom_records),
        "epsilon_m": full_samples.epsilon_m,
        "maximum_surface_cover_radius_m": (
            full_samples.maximum_surface_cover_radius_m
        ),
        "coverage_semantics": full_samples.coverage_semantics,
        "rigid_roundtrip": roundtrip,
    }
    try:
        type_counts = validate_robot_sample_evidence(
            full_sampling_evidence,
            resolved_geom_ids=resolved.robot_geom_ids,
            resolved_geom_names=resolved.robot_geom_names,
            resolved_body_ids=resolved.robot_body_ids,
            roundtrip_field="rigid_roundtrip",
        )
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        env.close()
        raise ShadowIdentificationRunnerError(
            "full-robot surface-sampling evidence is invalid: %s" % error
        ) from error
    if type_counts != {"mesh": 11, "box": 4, "cylinder": 1}:
        env.close()
        raise ShadowIdentificationRunnerError(
            "first-canary robot collision geometry type counts changed"
        )
    cylinder_record = next(
        record
        for record in full_samples.geom_records
        if record["geom_type_name"] == "cylinder"
    )
    if (
        cylinder_record["geom_id"] != 84
        or cylinder_record["geom_name"] != "mount0_pedestal_col"
        or cylinder_record["geom_size"] != [0.18, 0.31, 0.0]
    ):
        env.close()
        raise ShadowIdentificationRunnerError(
            "first-canary pedestal-cylinder identity or dimensions changed"
        )
    admissibility = protocol["admissibility"]
    safety = protocol["safety"]
    monitor = FullRobotObstacleMonitor(
        env.sim,
        resolved,
        full_samples.samples,
        certified_coverage_radius_m=(
            full_samples.maximum_surface_cover_radius_m
        ),
        max_selected_geom_surface_drift_m=float(
            admissibility["max_selected_geom_surface_drift_m"]
        ),
        max_selected_geom_translation_drift_m=float(
            admissibility["max_selected_geom_translation_drift_m"]
        ),
        max_selected_geom_rotation_drift_rad=float(
            admissibility["max_selected_geom_rotation_drift_rad"]
        ),
        max_settled_obstacle_linear_speed_m_per_s=float(
            admissibility["max_selected_body_linear_speed_m_s"]
        ),
        max_settled_obstacle_angular_speed_rad_per_s=float(
            admissibility["max_selected_body_angular_speed_rad_s"]
        ),
        require_settled_static_motion=True,
        terminate_on_static_drift=False,
        near_contact_tolerance_m=float(safety["contact_margin_m"]),
        inner_updates_per_high_level_action=INNER_UPDATES_PER_HIGH_LEVEL_ACTION,
        physics_substeps_per_inner_update=PHYSICS_SUBSTEPS_PER_INNER_UPDATE,
    )
    monitor.require_settled_obstacle_motion_admissible()
    arm_dof_indices = tuple(int(value) for value in env.robots[0]._ref_joint_vel_indexes)
    construction_state_before_hash = evaluator.array_sha256(
        construction_state_before
    )
    try:
        differential_audit = audit_protected_sample_differentials(
            raw_model,
            raw_data,
            bundle.protected_samples.samples,
            arm_dof_indices,
            bundle.field,
            protocol["differential_audit"],
        )
        differential_audit_validation = (
            inspect_protected_sample_differential_audit(
                differential_audit,
                expected_samples=bundle.protected_samples.samples,
                expected_arm_dof_indices=arm_dof_indices,
                expected_integration_state_sha256=(
                    construction_state_before_hash
                ),
                expected_differential_audit_config=protocol[
                    "differential_audit"
                ],
            )
        )
    except (DifferentialAuditError, RuntimeError, TypeError, ValueError) as error:
        env.close()
        raise ShadowIdentificationRunnerError(
            "settled link-5/6 differential audit failed: %s" % error
        ) from error
    if differential_audit_validation["passed"] is not True:
        failure_evidence = _differential_audit_failure_evidence(
            differential_audit, differential_audit_validation
        )
        env.close()
        raise ShadowIdentificationDifferentialAuditFailure(failure_evidence)
    observer = StaticPoissonShadowObserver(
        field=bundle.field,
        samples=bundle.protected_samples.samples,
        settled_obstacle_boxes=bundle.obstacle_boxes,
        arm_dof_indices=arm_dof_indices,
        alpha_gain_per_s=float(protocol["cbf"]["alpha_gain_per_s"]),
        physics_timestep_s=float(protocol["cadence"]["physics_timestep_s"]),
        drift_thresholds=StaticDriftThresholds(
            translation_m=float(
                admissibility["max_selected_geom_translation_drift_m"]
            ),
            rotation_rad=float(
                admissibility["max_selected_geom_rotation_drift_rad"]
            ),
            surface_m=float(admissibility["max_selected_geom_surface_drift_m"]),
        ),
        physics_substeps_per_high_level_action=(
            PHYSICS_SUBSTEPS_PER_HIGH_LEVEL_ACTION
        ),
    )
    from main.poisson_fullbody.controller_bridge import model_physics_contract

    try:
        physical_model = model_physics_contract(env.sim)
    except (RuntimeError, TypeError, ValueError) as error:
        env.close()
        raise ShadowIdentificationRunnerError(
            "compiled physical-model authority construction failed: %s" % error
        ) from error
    construction_state_after = _official_integration_state(
        env.sim, runtime["np"]
    )
    construction_state_after_hash = evaluator.array_sha256(
        construction_state_after
    )
    if not runtime["np"].array_equal(
        construction_state_after, construction_state_before
    ) or construction_state_after_hash != construction_state_before_hash:
        env.close()
        raise ShadowIdentificationRunnerError(
            "field and monitor construction changed complete MuJoCo integration state"
        )
    construction = {
        "active_obstacle_name": obstacle_name,
        "contact_model_authority_sha256": authority["authority_sha256"],
        "physical_model": physical_model,
        "robot_root_body_name": robot_root_name,
        "robot_root_body_ids": list(robot_roots),
        "arm_dof_indices": list(arm_dof_indices),
        "resolved_geometry": resolved.to_dict(),
        "field_bundle": {
            "protocol_id": bundle.protocol_id,
            "protected_body_ids": list(bundle.protected_body_ids),
            "protected_body_names": list(bundle.protected_body_names),
            "diagnostics": asdict(bundle.diagnostics),
            "hashes": asdict(bundle.hashes),
            "surface_components": [
                asdict(value) for value in bundle.protected_samples.components
            ],
            "protected_sample_count": len(bundle.protected_samples.samples),
            "protected_sampling_epsilon_m": float(
                bundle.protected_samples.epsilon_m
            ),
            "protected_sampling_maximum_surface_cover_radius_m": float(
                bundle.protected_samples.maximum_surface_cover_radius_m
            ),
            "protected_sampling_coverage_semantics": (
                bundle.protected_samples.coverage_semantics
            ),
            "protected_samples": [
                sample.to_dict() for sample in bundle.protected_samples.samples
            ],
        },
        "static_field_drift_thresholds": {
            "translation_m": float(
                admissibility["max_selected_geom_translation_drift_m"]
            ),
            "rotation_rad": float(
                admissibility["max_selected_geom_rotation_drift_rad"]
            ),
            "surface_m": float(
                admissibility["max_selected_geom_surface_drift_m"]
            ),
        },
        "cbf_alpha_gain_per_s": float(protocol["cbf"]["alpha_gain_per_s"]),
        "settled_link56_differential_audit": differential_audit,
        "settled_link56_differential_audit_validation": (
            differential_audit_validation
        ),
        "full_robot_measurement_sampling": full_sampling_evidence,
        "settled_measurement": monitor.settled_state.to_dict(),
        "measurement_drift_mode": (
            "diagnostic_continue_contact_measurement_after_static_field_invalidation"
        ),
        "complete_integration_state_read_only_audit": {
            "mujoco_state_specification": "mjSTATE_INTEGRATION",
            "state_vector_length": int(construction_state_before.size),
            "before_sha256": construction_state_before_hash,
            "after_sha256": construction_state_after_hash,
            "exact_array_equal": True,
            "semantics": (
                "the complete official MuJoCo integration state was bitwise "
                "unchanged across field, sampling, monitor construction, and "
                "the exhaustive clone-only differential audit"
            ),
        },
    }
    return (
        env,
        task,
        observation,
        goal_atoms,
        previous_goal,
        monitor,
        observer,
        construction,
    )


def _run_shadow(
    *,
    evaluator: Any,
    runtime: Mapping[str, Any],
    case: Mapping[str, Any],
    replay: Any,
    protocol: Mapping[str, Any],
    protocol_hashes: Any,
    upstream_parity: Mapping[str, Any],
) -> Dict[str, Any]:
    from main.poisson_fullbody.measurement import StaticObstacleDriftInadmissible
    from main.poisson_fullbody.shadow_identification import (
        validate_shadow_replay_record,
    )
    from scripts.run_poisson_shadow_parity import _canonical, _check_step, _sha256

    env = None
    try:
        (
            env,
            _,
            observation,
            goal_atoms,
            previous_goal,
            monitor,
            observer,
            construction,
        ) = _prepare_shadow_runtime(
            evaluator=evaluator,
            runtime=runtime,
            case=case,
            replay=replay,
            protocol=protocol,
            protocol_hashes=protocol_hashes,
        )
        external_settled_state = upstream_parity["callback_replay"][
            "settled_official_integration_state"
        ]
        construction_read_only = construction[
            "complete_integration_state_read_only_audit"
        ]
        if (
            construction_read_only["before_sha256"]
            != external_settled_state["sha256"]
            or construction_read_only["after_sha256"]
            != external_settled_state["sha256"]
            or construction_read_only["state_vector_length"]
            != external_settled_state["state_vector_length"]
        ):
            raise ShadowIdentificationRunnerError(
                "shadow construction settled state differs from upstream exact parity"
            )
        state_hashes = []
        observation_hashes = []
        callback_state_hashes = []
        callback_state_read_only_ledger = []
        monitor_drift_exceptions = []
        for expected_step in replay.steps:
            current_callback_hashes = []
            current_callback_state_read_only_records = []

            def callback(sim: Any, substep_index: int) -> None:
                before_state = _official_integration_state(sim, runtime["np"])
                before_state_hash = evaluator.array_sha256(before_state)
                inner = (
                    int(substep_index) // PHYSICS_SUBSTEPS_PER_INNER_UPDATE
                )
                physics = (
                    int(substep_index) % PHYSICS_SUBSTEPS_PER_INNER_UPDATE
                )
                try:
                    monitor.observe_post_integration(
                        sim,
                        high_level_index=int(expected_step.step),
                        inner_control_index=inner,
                        physics_substep_index=physics,
                    )
                except StaticObstacleDriftInadmissible as error:
                    # The monitor has already recorded this substep.  Static
                    # field invalidation is handled independently by observer;
                    # exact OSC replay and contact measurement must continue.
                    monitor_drift_exceptions.append(
                        {
                            "observation_index": (
                                int(expected_step.step)
                                * PHYSICS_SUBSTEPS_PER_HIGH_LEVEL_ACTION
                                + int(substep_index)
                            ),
                            "message": str(error),
                        }
                    )
                observer.observe(
                    sim,
                    high_level_index=int(expected_step.step),
                    physics_substep_index=int(substep_index),
                )
                after_state = _official_integration_state(sim, runtime["np"])
                after_state_hash = evaluator.array_sha256(after_state)
                if not runtime["np"].array_equal(
                    after_state, before_state
                ) or after_state_hash != before_state_hash:
                    raise ShadowIdentificationRunnerError(
                        "read-only shadow callback mutated complete MuJoCo integration state"
                    )
                current_callback_state_read_only_records.append(
                    {
                        "observation_index": (
                            int(expected_step.step)
                            * PHYSICS_SUBSTEPS_PER_HIGH_LEVEL_ACTION
                            + int(substep_index)
                        ),
                        "high_level_index": int(expected_step.step),
                        "inner_control_index": inner,
                        "physics_substep_index": physics,
                        "before_sha256": before_state_hash,
                        "after_sha256": after_state_hash,
                        "exact_array_equal": True,
                    }
                )
                current_callback_hashes.append(after_state_hash)

            observation, reward, done, _ = env.step_with_substep_callback(
                expected_step.action,
                callback,
                expected_substeps=PHYSICS_SUBSTEPS_PER_HIGH_LEVEL_ACTION,
            )
            if (
                len(current_callback_hashes)
                != PHYSICS_SUBSTEPS_PER_HIGH_LEVEL_ACTION
            ):
                raise ShadowIdentificationRunnerError(
                    "shadow callback cadence differs at step %d" % expected_step.step
                )
            state_hash, observation_hash, previous_goal = _check_step(
                evaluator=evaluator,
                env=env,
                observation=observation,
                reward=reward,
                done=done,
                expected_step=expected_step,
                goal_atoms=goal_atoms,
                previous_goal_values=previous_goal,
                np=runtime["np"],
            )
            state_hashes.append(state_hash)
            observation_hashes.append(observation_hash)
            callback_state_hashes.extend(current_callback_hashes)
            callback_state_read_only_ledger.extend(
                current_callback_state_read_only_records
            )
            proxy = evaluator._eef_proxy(runtime, observation)
            evaluator._update_eef_marker(env, proxy)
            if done and expected_step.step != len(replay.steps) - 1:
                raise ShadowIdentificationRunnerError(
                    "shadow replay terminated before the registered horizon"
                )
        measurement = monitor.result()
        expected_callback_count = _require_complete_measurement_exposure(
            measurement, action_count=len(replay.steps)
        )
        if measurement.physical_contact_distance_semantics != CONTACT_DEFINITION:
            raise ShadowIdentificationRunnerError(
                "MuJoCo physical-contact authority semantics changed"
            )
        resolved = construction["resolved_geometry"]
        first_link_contact = _first_link56_contact(
            measurement, resolved["link56_geom_ids"]
        )
        identification = observer.result(
            expected_callback_count=expected_callback_count,
            first_link56_contact=first_link_contact,
        )
        state_sequence_hash = _sha256(_canonical(state_hashes))
        observation_sequence_hash = _sha256(_canonical(observation_hashes))
        upstream_callback = upstream_parity["callback_replay"]
        expected_action_states = [
            step.simulator_state_sha256 for step in replay.steps
        ]
        if (
            state_hashes != expected_action_states
            or upstream_callback.get(
                "action_boundary_state_sha256_ledger"
            )
            != expected_action_states
            or state_sequence_hash
            != upstream_callback["state_sequence_sha256"]
        ):
            raise ShadowIdentificationRunnerError(
                "shadow state sequence differs from upstream exact parity"
            )
        if observation_sequence_hash != upstream_callback["observation_sequence_sha256"]:
            raise ShadowIdentificationRunnerError(
                "shadow observation sequence differs from upstream exact parity"
            )
        if callback_state_hashes != upstream_callback[
            "official_integration_state_sha256_ledger"
        ] or _sha256(_canonical(callback_state_hashes)) != upstream_callback[
            "official_integration_state_sequence_sha256"
        ]:
            raise ShadowIdentificationRunnerError(
                "shadow callback integration-state sequence differs from upstream exact parity"
            )
        if state_hashes[-1] != upstream_callback[
            "terminal_simulator_state_sha256"
        ]:
            raise ShadowIdentificationRunnerError(
                "shadow terminal state differs from upstream exact parity"
            )
        shadow = {
            "executed_action_count": len(state_hashes),
            "callback_count": len(callback_state_hashes),
            "expected_callback_count": expected_callback_count,
            "action_boundary_state_sha256_ledger": state_hashes,
            "state_sequence_sha256": state_sequence_hash,
            "observation_sequence_sha256": observation_sequence_hash,
            "callback_state_read_only_ledger": (
                callback_state_read_only_ledger
            ),
            "callback_state_read_only_ledger_sha256": _sha256(
                _canonical(callback_state_read_only_ledger)
            ),
            "callback_state_sequence_sha256": _sha256(
                _canonical(callback_state_hashes)
            ),
            "terminal_simulator_state_sha256": state_hashes[-1],
            "construction": construction,
            "measurement": measurement.to_dict(),
            "monitor_static_drift_exception_count": len(monitor_drift_exceptions),
            "monitor_static_drift_exceptions": monitor_drift_exceptions,
            "poisson_identification": identification,
        }
        validate_shadow_replay_record(
            shadow,
            action_count=len(replay.steps),
            inner_updates_per_high_level_action=(
                INNER_UPDATES_PER_HIGH_LEVEL_ACTION
            ),
            physics_substeps_per_inner_update=(
                PHYSICS_SUBSTEPS_PER_INNER_UPDATE
            ),
            physics_timestep_s=0.002,
            contact_definition=CONTACT_DEFINITION,
            alpha_gain_per_s=float(protocol["cbf"]["alpha_gain_per_s"]),
            static_drift_thresholds={
                "translation_m": float(
                    protocol["admissibility"][
                        "max_selected_geom_translation_drift_m"
                    ]
                ),
                "rotation_rad": float(
                    protocol["admissibility"][
                        "max_selected_geom_rotation_drift_rad"
                    ]
                ),
                "surface_m": float(
                    protocol["admissibility"][
                        "max_selected_geom_surface_drift_m"
                    ]
                ),
            },
            differential_audit_config=protocol["differential_audit"],
            expected_settled_integration_state_sha256=(
                external_settled_state["sha256"]
            ),
            expected_settled_integration_state_length=(
                external_settled_state["state_vector_length"]
            ),
        )
        return shadow
    finally:
        if env is not None:
            env.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("manifests/vlsa_poisson_link56_aegis_car_109.v1.jsonl"),
    )
    parser.add_argument(
        "--selection-config",
        type=Path,
        default=Path("configs/vlsa_poisson_link56_feasibility.v1.json"),
    )
    parser.add_argument("--historical-result-root", type=Path, required=True)
    parser.add_argument("--parity-result", type=Path, required=True)
    parser.add_argument("--case-id", default=DEFAULT_CASE_ID)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    root = arguments.repo_root.resolve()
    manifest = arguments.manifest
    if not manifest.is_absolute():
        manifest = root / manifest
    selection_config = arguments.selection_config
    if not selection_config.is_absolute():
        selection_config = root / selection_config
    output = arguments.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "safelibero"))
    sys.path.insert(0, str(root / "main"))

    from main.poisson_fullbody.contracts import publish_hashed_json
    from main.poisson_fullbody.shadow_replay import load_historical_action_replay
    from scripts.run_poisson_shadow_parity import (
        _file_sha256,
        _git_record,
        _gpu_inventory,
        _load_case,
    )

    started = time.time()
    payload: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "failed",
        "scientific_result": False,
        "evidence_tier": "allocation_backed_exact_replay_shadow_identification",
        "claim_limit": (
            "read-only temporal identification only; no action was corrected and "
            "no active safety or utility efficacy was tested"
        ),
        "case_id": arguments.case_id,
        "phase": "initial_authority_validation",
        "timing": {"started_unix": started},
    }
    try:
        if arguments.case_id != DEFAULT_CASE_ID:
            raise ShadowIdentificationRunnerError(
                "shadow identification is registered only for the first canary %s"
                % DEFAULT_CASE_ID
            )
        source = _git_record(root)
        if source["status_short"]:
            raise ShadowIdentificationRunnerError(
                "shadow identification requires a clean source tree"
            )
        if not os.environ.get("SLURM_JOB_ID"):
            raise ShadowIdentificationRunnerError(
                "shadow identification must run inside a Slurm allocation"
            )
        gpu = _gpu_inventory()
        line_number, case, row_hash = _load_case(manifest, arguments.case_id)
        historical_record = case.get("historical_aegis_result")
        if not isinstance(historical_record, dict):
            raise ShadowIdentificationRunnerError(
                "manifest lacks historical AEGIS binding"
            )
        relative = historical_record.get("source_relative_path")
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
        ):
            raise ShadowIdentificationRunnerError(
                "historical result relative path is invalid"
            )
        historical_path = arguments.historical_result_root.resolve() / relative
        if _file_sha256(historical_path) != historical_record.get("raw_file_sha256"):
            raise ShadowIdentificationRunnerError("historical result raw SHA-256 differs")
        replay = load_historical_action_replay(
            historical_path, expected_case_id=arguments.case_id
        )
        if replay.result_payload_sha256 != historical_record.get(
            "result_payload_sha256"
        ):
            raise ShadowIdentificationRunnerError(
                "historical result payload binding differs"
            )
        if len(replay.steps) != 237:
            raise ShadowIdentificationRunnerError(
                "registered first canary must expose exactly 237 historical actions"
            )
        manifest_hash = _file_sha256(manifest)
        upstream_parity = _require_upstream_parity(
            arguments.parity_result.resolve(),
            case_id=arguments.case_id,
            source_commit=source["commit"],
            historical_payload_sha256=replay.result_payload_sha256,
            manifest_sha256=manifest_hash,
            manifest_row_sha256=row_hash,
            expected_callback_count=(
                PHYSICS_SUBSTEPS_PER_HIGH_LEVEL_ACTION * len(replay.steps)
            ),
            historical_provenance=replay.provenance(),
            expected_action_state_sha256_ledger=[
                step.simulator_state_sha256 for step in replay.steps
            ],
        )
        protocol, protocol_hashes, selection, runtime_path = (
            _load_bound_runtime_protocol(
                root=root,
                case=case,
                selection_config_path=selection_config,
            )
        )
        provenance = {
            "source": source,
            "manifest_path": str(manifest),
            "manifest_sha256": manifest_hash,
            "manifest_line_number": line_number,
            "manifest_row_sha256": row_hash,
            "selection_config_path": str(selection_config),
            "selection_config_sha256": _file_sha256(selection_config),
            "selection_protocol_id": selection["protocol_id"],
            "runtime_protocol_path": str(runtime_path),
            "runtime_protocol_raw_sha256": _file_sha256(runtime_path),
            "runtime_protocol_semantic_sha256": (
                protocol_hashes.protocol_sha256
            ),
            "runtime_parameter_block_sha256": (
                protocol_hashes.parameter_block_sha256
            ),
            "historical_result_path": str(historical_path),
            "historical_result_file_sha256": replay.result_file_sha256,
            "historical_result_payload_sha256": (
                replay.result_payload_sha256
            ),
            "upstream_parity_path": str(arguments.parity_result.resolve()),
            "upstream_parity_payload_sha256": upstream_parity[
                "result_payload_sha256"
            ],
            "host": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_job_name": os.environ.get("SLURM_JOB_NAME"),
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "packages": _package_versions(
                ("numpy", "mujoco", "robosuite", "scipy")
            ),
            "gpu": gpu,
        }
        payload.update(
            {
                "phase": "prepare_shadow_runtime_and_exact_replay",
                "provenance": provenance,
                "historical": replay.provenance(),
            }
        )
        evaluator = importlib.import_module("evaluate_safelibero_aegis")
        runtime = evaluator._runtime_imports(include_aegis=False)
        shadow = _run_shadow(
            evaluator=evaluator,
            runtime=runtime,
            case=case,
            replay=replay,
            protocol=protocol,
            protocol_hashes=protocol_hashes,
            upstream_parity=upstream_parity,
        )
        payload["phase"] = "final_shadow_validation"
        expected_state_hash = upstream_parity["callback_replay"][
            "state_sequence_sha256"
        ]
        expected_observation_hash = upstream_parity["callback_replay"][
            "observation_sequence_sha256"
        ]
        identification = shadow["poisson_identification"]
        payload.update(
            {
                "status": "passed",
                "shadow_replay": shadow,
                "acceptance": {
                    "upstream_exact_parity_same_clean_commit": True,
                    "all_237_historical_actions_executed": (
                        shadow["executed_action_count"] == 237
                    ),
                    "all_5925_callbacks_observed": (
                        shadow["callback_count"] == 5925
                        and identification["observed_callback_count"] == 5925
                        and shadow["measurement"]["observed_physics_substeps"]
                        == 5925
                    ),
                    "historical_state_reward_done_goal_exact": (
                        shadow["state_sequence_sha256"] == expected_state_hash
                    ),
                    "upstream_observation_sequence_exact": (
                        shadow["observation_sequence_sha256"]
                        == expected_observation_hash
                    ),
                    "settled_mjstate_integration_matches_upstream_exact_parity": (
                        shadow["construction"][
                            "complete_integration_state_read_only_audit"
                        ]["before_sha256"]
                        == upstream_parity["callback_replay"][
                            "settled_official_integration_state"
                        ]["sha256"]
                    ),
                    "complete_mujoco_integration_state_unchanged_by_construction": (
                        shadow["construction"][
                            "complete_integration_state_read_only_audit"
                        ]["exact_array_equal"]
                    ),
                    "complete_mujoco_integration_state_unchanged_by_callback": (
                        len(shadow["callback_state_read_only_ledger"])
                        == 5925
                        and all(
                            record["exact_array_equal"] is True
                            and record["before_sha256"]
                            == record["after_sha256"]
                            for record in shadow[
                                "callback_state_read_only_ledger"
                            ]
                        )
                    ),
                    "all_link56_protected_sample_point_jacobians_validated": (
                        shadow["construction"][
                            "settled_link56_differential_audit_validation"
                        ]["counts"][
                            "point_jacobian_passed_sample_count"
                        ]
                        == shadow["construction"][
                            "settled_link56_differential_audit_validation"
                        ]["counts"]["sample_count"]
                    ),
                    "all_link56_protected_sample_field_chain_rules_validated": (
                        shadow["construction"][
                            "settled_link56_differential_audit_validation"
                        ]["counts"]["passed_coupled_direction_count"]
                        == shadow["construction"][
                            "settled_link56_differential_audit_validation"
                        ]["counts"]["required_coupled_direction_count"]
                    ),
                    "static_queries_stop_at_first_registered_drift": True,
                    "contact_authority_is_mujoco_nonpositive_distance": True,
                    "no_action_or_control_mutation": True,
                    "no_active_safety_efficacy_claim": True,
                },
            }
        )
        if not all(payload["acceptance"].values()):
            raise ShadowIdentificationRunnerError(
                "one or more final shadow acceptance checks failed"
            )
        payload["phase"] = "complete"
    except Exception as error:
        _record_failed_result(payload, error, traceback.format_exc())
    payload["timing"].update(
        {
            "finished_unix": time.time(),
            "elapsed_seconds": time.time() - started,
        }
    )
    publish_hashed_json(output, payload)
    print(json.dumps({"status": payload["status"], "output": str(output)}, sort_keys=True))
    return 0 if payload["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
