#!/usr/bin/env python3
"""Independent consumer for the paired live-policy Poisson canary.

This module deliberately does not import the producer classifier.  It accepts
only an atomically published, hashed terminal result and reconstructs the
closed-loop action/observation chains, controller cadence, MuJoCo contacts,
CBF intervention, useful motion, native task outcome, videos, and final label.
A complete collision or task-failure outcome is a valid scientific negative;
only ``SAFE_TASK_SUCCESS_USEFUL_CORRECTION`` is feasible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import stat
import struct
import subprocess
import sys
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main.poisson_fullbody.closed_loop_canary import (  # noqa: E402
    CLASSIFICATION_SCHEMA,
    RESULT_SCHEMA,
    validate_closed_loop_canary_protocol,
)
from main.poisson_fullbody.contracts import (  # noqa: E402
    ArtifactContractError,
    load_hashed_json,
    sha256_file,
)
from main.poisson_fullbody.shadow_replay import (  # noqa: E402
    load_historical_action_replay,
)
from scripts import validate_poisson_fast_feasibility_artifact as narrow  # noqa: E402
from scripts import validate_poisson_full_episode_feasibility_artifact as full  # noqa: E402


SUMMARY_SCHEMA = "vlsa_poisson_closed_loop_canary_consumer.v1"
PROTOCOL_SCHEMA = "vlsa_poisson_closed_loop_canary_protocol.v1"
PROTOCOL_ID = "vlsa-poisson-link56-closed-loop-suffix-v1"
EXPECTED_CASE = "vlsa-t1-goal-ii-t0-e05"
EXPECTED_BRANCH = "codex/poisson-closed-loop-canary"
EXPECTED_SOURCE_ARM = "pi05_plus_aegis_translational"
EXPECTED_CASE_ROW_SHA256 = (
    "ee42bf1e8587acf2fd4fe73cb05d7974de836a37f53acc8035e0995b1d1fca9e"
)
EXPECTED_HISTORICAL_FILE_SHA256 = (
    "273d77cba3fd5b1e457817ad20f8572b5628aeaa8b5b9f768b32e4f095fbad8b"
)
EXPECTED_HISTORICAL_PAYLOAD_SHA256 = (
    "ec58c8581297751de33756e76efbf5b18e24a5f836aa6a8f147d0caebdd92d1c"
)
EXPECTED_ACTION_SEQUENCE_SHA256 = (
    "e5e2df4efee667fee0b0b7596dbc203a3cce84e67cb3507cb839132d2ee67481"
)
EXPECTED_BOUNDARY_STATE_SHA256 = (
    "2b4d25f12da2e1c110883b9f2194c9790270f36911f449d8c4f058a039c90335"
)
EXPECTED_BOUNDARY_RAW_SHA256 = (
    "ff1a9a80d03178c9ac85e5dc6750aac98f920a050462df0b17d7ca1cdbb069c7"
)
EXPECTED_FIRST_CHUNK_SHA256 = (
    "572e21359bcd19f6ccaf491bc393b9f0be7f7e1886de39bf483e1036f45a97af"
)
EXPECTED_FIELD_BUNDLE_SHA256 = (
    "979ebc41198293cec1a4d95e1c9df67d121767824e5b0d4a6c089bdb030d7f77"
)
EXPECTED_FIELD_PROTOCOL_SHA256 = (
    "2125989269a2ffeeb8d3408d56e4aaa74e1dc5816256d8210bf9ee686c5f5a30"
)
EXPECTED_FIELD_PARAMETER_BLOCK_SHA256 = (
    "11de833ab8b1586948e7eb039c449a66a0c1a4107fa3232b236ad1ecb9ceeabe"
)
EXPECTED_FULL_ROBOT_SAMPLING_SHA256 = (
    "063a56c76c4d554f35523e7e171f6076c950d147f702d5a6eff1b59acdd78ba1"
)
EXPECTED_CLAIM_SCOPE = (
    "one_case_static_simulator_oracle_hybrid_prefix_live_closed_loop_suffix_"
    "feasibility_not_population_safety_not_realtime_or_tracking_certified"
)
EXPECTED_GPU = "NVIDIA H100 80GB HBM3"

METRIC_KEYS = (
    "exact_paired_start",
    "shared_prefix_complete",
    "baseline_exposure_complete",
    "psf_exposure_complete",
    "live_policy_contract_valid",
    "aegis_contract_valid",
    "videos_complete_and_decodable",
    "psf_qp_contract_valid",
    "static_selected_obstacle_admissible",
    "literal_contact_checked_at_every_physics_substep",
    "released_eef_marker_update_contract_valid",
    "first_live_query_identical",
    "only_psf_query_36_reused_paired_cache",
    "historical_first_live_query_action_chunk_matches_diagnostic",
    "first_current_action_180_aegis_inputs_identical",
    "first_current_action_180_aegis_outputs_identical",
    "post_divergence_own_observations_used",
    "post_divergence_policy_inputs_differ",
    "fresh_policy_query_after_material_correction",
    "fresh_policy_query_indexes_after_material_correction",
    "own_observation_chain_valid",
    "no_recorded_suffix_action_replay",
    "first_live_aegis_action_matches_historical",
    "historical_action_180_full_output_matches_diagnostic",
    "both_nominal_commands_within_dynamic_joint_bounds",
    "baseline_link56_contact_present",
    "baseline_first_selected_obstacle_contact_is_link56",
    "psf_link56_contact_present",
    "psf_any_robot_selected_obstacle_contact_present",
    "psf_clearance_certified",
    "psf_periodic_clearance_diagnostic_positive",
    "material_correction_before_baseline_contact",
    "first_material_correction_physical_boundary",
    "baseline_first_link56_contact_physical_boundary",
    "material_correction_update_count",
    "maximum_correction_norm_rad_s",
    "filter_correction_integral_rad",
    "post_correction_measured_joint_motion_integral_rad",
    "post_correction_cartesian_path_length_m",
    "post_correction_executed_command_integral_rad",
    "post_correction_zero_command_fraction",
    "baseline_task_success_ever",
    "baseline_terminal_task_success",
    "psf_task_success_ever",
    "psf_terminal_task_success",
    "psf_task_success_after_material_correction",
    "psf_successful_source_action_indexes_after_material_correction",
)

_Audit = narrow._Audit
_canonical = narrow._canonical
_close = narrow._close
_sha256 = narrow._sha256
_vector = narrow._vector
_norm = narrow._norm


def _load_plain_json(path: Path, label: str) -> Mapping[str, Any]:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ArtifactContractError("%s is missing, nonregular, or symlinked" % label)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ArtifactContractError("%s must contain one object" % label)
    return value


def _array_sha256(value: Any) -> str:
    """Independently reproduce the Table-1 ndarray hash for float64 matrices."""

    rows = list(value)
    if not rows or any(
        isinstance(row, (str, bytes)) or not isinstance(row, Sequence)
        for row in rows
    ):
        raise ArtifactContractError("policy action array is not a nonempty matrix")
    width = len(rows[0])
    if width <= 0 or any(len(row) != width for row in rows):
        raise ArtifactContractError("policy action array is not rectangular")
    flattened: List[float] = []
    for row in rows:
        for item in row:
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise ArtifactContractError("policy action array is not numeric")
            item = float(item)
            if not math.isfinite(item):
                raise ArtifactContractError("policy action array is nonfinite")
            flattened.append(item)
    byte_order = "<" if sys.byteorder == "little" else ">"
    raw = struct.pack("%s%dd" % (byte_order, len(flattened)), *flattened)
    digest = hashlib.sha256()
    digest.update(b"vlsa-table1-array-v1\0")
    digest.update(_canonical({"dtype": byte_order + "f8", "shape": [len(rows), width]}))
    digest.update(b"\0")
    digest.update(raw)
    return digest.hexdigest()


def _numpy_array_sha256(value: Any) -> str:
    """Hash an arbitrary ndarray without importing producer hash code."""

    import numpy as np

    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(b"vlsa-table1-array-v1\0")
    digest.update(_canonical({"dtype": array.dtype.str, "shape": list(array.shape)}))
    digest.update(b"\0")
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _protocol_expectations(protocol: Mapping[str, Any]) -> Dict[str, Any]:
    """Fail closed on the frozen protocol and derive primitive populations."""

    try:
        derived = validate_closed_loop_canary_protocol(protocol)
    except Exception as error:
        raise ArtifactContractError("closed-loop protocol invalid: %s" % error) from error
    if protocol.get("schema_version") != PROTOCOL_SCHEMA or protocol.get("protocol_id") != PROTOCOL_ID:
        raise ArtifactContractError("closed-loop protocol identity differs")
    source = protocol.get("source")
    if not isinstance(source, Mapping):
        raise ArtifactContractError("protocol source block is absent")
    frozen = {
        "historical_result_file_sha256": EXPECTED_HISTORICAL_FILE_SHA256,
        "historical_result_payload_sha256": EXPECTED_HISTORICAL_PAYLOAD_SHA256,
        "historical_executed_action_sequence_sha256": EXPECTED_ACTION_SEQUENCE_SHA256,
        "post_action_179_flattened_state_sha256": EXPECTED_BOUNDARY_STATE_SHA256,
        "historical_first_live_query_action_chunk_sha256_diagnostic": EXPECTED_FIRST_CHUNK_SHA256,
    }
    for field, expected in frozen.items():
        if source.get(field) != expected:
            raise ArtifactContractError("protocol source %s differs" % field)
    cadence = protocol["cadence"]
    episode = protocol["episode"]
    expectation = {
        "prefix_start": 0,
        "prefix_end": 179,
        "prefix_action_count": 180,
        "suffix_start": int(derived["start_action"]),
        "suffix_end": int(derived["end_action"]),
        "source_actions": tuple(range(int(derived["start_action"]), int(derived["end_action"]) + 1)),
        "action_count": int(derived["action_count"]),
        "controls_per_action": int(cadence["inner_updates_per_high_level_action"]),
        "substeps_per_control": int(cadence["physics_substeps_per_inner_update"]),
        "filter_updates": int(derived["expected_updates"]),
        "physics_substeps": int(derived["expected_substeps"]),
        "start_boundary": int(episode["branch_physical_boundary"]),
        "end_boundary": int(episode["terminal_physical_boundary"]),
        "no_event_boundary": int(episode["terminal_physical_boundary"]) + 1,
        "control_dt_s": 1.0 / float(cadence["filter_frequency_hz"]),
        "physics_dt_s": float(cadence["physics_timestep_s"]),
        "expected_boundary_goal_values": tuple(episode["expected_boundary_goal_values"]),
        "first_query_index": int(derived["first_query_index"]),
        "queries_per_arm": int(derived["queries_per_arm"]),
        "replan_steps": int(derived["replan_steps"]),
        "clearance_stride": int(derived["clearance_diagnostic_stride"]),
        "thresholds": dict(derived["thresholds"]),
    }
    if (
        expectation["source_actions"] != tuple(range(180, 237))
        or expectation["filter_updates"] != 285
        or expectation["physics_substeps"] != 1425
        or expectation["start_boundary"] != 4500
        or expectation["end_boundary"] != 5925
        or expectation["clearance_stride"] != 25
    ):
        raise ArtifactContractError("derived closed-loop population differs")
    return expectation


def _historical_path(root: Path) -> Path:
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ArtifactContractError("historical result root is missing or symlinked")
    matches = list(root.glob("tasks/task-*/results/aegis/%s/result.json" % EXPECTED_CASE))
    if len(matches) != 1:
        raise ArtifactContractError("historical result population is ambiguous")
    path = matches[0]
    if path.is_symlink() or not path.is_file():
        raise ArtifactContractError("historical result is missing or symlinked")
    if sha256_file(path) != EXPECTED_HISTORICAL_FILE_SHA256:
        raise ArtifactContractError("historical result file hash differs")
    return path


def _same_numeric(left: Any, right: Any, tolerance: float = 1.0e-12) -> bool:
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return isinstance(left, Mapping) and isinstance(right, Mapping) and set(left) == set(right) and all(
            _same_numeric(left[key], right[key], tolerance) for key in left
        )
    if isinstance(left, Sequence) and not isinstance(left, (str, bytes)):
        return isinstance(right, Sequence) and not isinstance(right, (str, bytes)) and len(left) == len(right) and all(
            _same_numeric(a, b, tolerance) for a, b in zip(left, right)
        )
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isfinite(float(left)) and math.isfinite(float(right)) and math.isclose(
            float(left), float(right), rel_tol=0.0, abs_tol=tolerance
        )
    return left == right


_OBSERVATION_FIELDS = (
    "native_observation_sha256",
    "official_integration_state_raw_bytes_sha256",
    "agentview_policy_array_sha256",
    "wrist_policy_array_sha256",
    "policy_state_array_sha256",
    "policy_state",
    "prompt",
)

_HISTORICAL_REQUIRED_AEGIS_INPUT_KEYS = (
    "p1",
    "R1",
    "q1_diag",
    "p2",
    "R2",
    "Q2_diag",
    "z_before",
)
_CURRENT_AEGIS_INPUT_KEYS = (
    *_HISTORICAL_REQUIRED_AEGIS_INPUT_KEYS,
    "nominal_translational",
)
_AEGIS_OUTPUT_KEYS = (
    "solver",
    "solver_status",
    "objective",
    "barrier_h",
    "constraint_lhs",
    "u_solution",
    "z_after",
    "status",
)
_AEGIS_CONTEXT_OUTPUT_KEYS = (
    "status",
    "solver_status",
    "solver_stats",
    "objective",
    "u_solution",
    "solution_lhs",
    "solution_slack",
    "solution_violation",
    "constraint_dual",
    "z_after",
    "executed_action",
    "executed_action_array_sha256",
    "executed_action_canonical_sha256",
)


def _without_timing(value: Any) -> Any:
    """Remove runtime-only timing leaves from a frozen comparison projection."""

    if isinstance(value, Mapping):
        return {
            str(key): _without_timing(item)
            for key, item in value.items()
            if "time" not in str(key).lower()
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_without_timing(item) for item in value]
    return value


def _aegis_input_projection(
    row: Mapping[str, Any], *, historical: bool, include_nominal: bool
) -> Dict[str, Any]:
    qp = row.get("qp" if historical else "aegis_qp")
    context = qp.get("context") if isinstance(qp, Mapping) else None
    if not isinstance(context, Mapping):
        return {}
    keys = (
        _CURRENT_AEGIS_INPUT_KEYS
        if include_nominal
        else _HISTORICAL_REQUIRED_AEGIS_INPUT_KEYS
    )
    return {key: context.get(key) for key in keys}


def _aegis_output_projection(
    row: Mapping[str, Any], *, historical: bool
) -> Dict[str, Any]:
    qp = row.get("qp" if historical else "aegis_qp")
    context = qp.get("context") if isinstance(qp, Mapping) else None
    if not isinstance(qp, Mapping) or not isinstance(context, Mapping):
        return {}
    return {
        "executed_action": row.get(
            "executed" if historical else "aegis_executed"
        ),
        "qp": _without_timing(
            {key: qp.get(key) for key in _AEGIS_OUTPUT_KEYS}
        ),
        "context": _without_timing(
            {key: context.get(key) for key in _AEGIS_CONTEXT_OUTPUT_KEYS}
        ),
    }


def _validate_query_execution(
    audit: _Audit,
    query: Mapping[str, Any],
    label: str,
    query_index: int,
) -> None:
    if label == "psf" and query_index == 36:
        expected = (
            "paired_cache_reuse",
            False,
            "pi05_plus_aegis_joint_velocity_adapter",
            36,
            query.get("returned_actions_sha256"),
        )
    else:
        expected = ("live_inference", True, None, None, None)
    observed = (
        query.get("query_execution"),
        query.get("inference_performed"),
        query.get("paired_cache_source_arm"),
        query.get("paired_cache_source_query_index"),
        query.get("paired_cache_source_returned_actions_sha256"),
    )
    audit.check(
        observed == expected,
        "%s_query_%d_execution_provenance_differs" % (label, query_index),
    )
    diagnostic_hash = query.get(
        "historical_returned_actions_sha256_diagnostic"
    )
    diagnostic_match = query.get(
        "matches_historical_returned_actions_sha256_diagnostic"
    )
    if query_index == 36:
        audit.check(
            diagnostic_hash == EXPECTED_FIRST_CHUNK_SHA256,
            "%s_query_36_historical_diagnostic_hash_differs" % label,
        )
        audit.check(
            diagnostic_match
            is (query.get("returned_actions_sha256") == EXPECTED_FIRST_CHUNK_SHA256),
            "%s_query_36_historical_diagnostic_flag_differs" % label,
        )
    else:
        audit.check(
            diagnostic_hash is None and diagnostic_match is None,
            "%s_query_%d_unexpected_historical_diagnostic" % (label, query_index),
        )


def _validate_observation_record(audit: _Audit, row: Mapping[str, Any], label: str) -> None:
    record: Dict[str, Any] = {}
    for field in _OBSERVATION_FIELDS:
        value = row.get(field)
        if field.endswith("sha256"):
            value = _sha256(audit, value, "%s_%s" % (label, field))
        elif field == "policy_state":
            values = audit.sequence(value, "%s_policy_state" % label)
            for index, item in enumerate(values):
                audit.number(item, "%s_policy_state_%d" % (label, index))
            value = list(values)
        elif field == "prompt":
            audit.check(isinstance(value, str) and bool(value), "%s_prompt_invalid" % label)
        record[field] = value
    fingerprint = _sha256(audit, row.get("policy_input_fingerprint_sha256"), "%s_fingerprint" % label)
    audit.check(hashlib.sha256(_canonical(record)).hexdigest() == fingerprint, "%s_fingerprint_differs" % label)


def _validate_provider(
    audit: _Audit,
    provider_value: Any,
    arm: Mapping[str, Any],
    label: str,
    expectation: Mapping[str, Any],
    historical: Mapping[str, Any],
    *,
    expected_actions: int,
) -> Dict[str, Any]:
    provider = audit.mapping(provider_value, "%s_provider" % label)
    actions = [
        audit.mapping(value, "%s_action_%d" % (label, index))
        for index, value in enumerate(audit.sequence(provider.get("high_level_action_trace"), "%s_actions" % label))
    ]
    queries = [
        audit.mapping(value, "%s_query_%d" % (label, index))
        for index, value in enumerate(audit.sequence(provider.get("policy_queries"), "%s_queries" % label))
    ]
    expected_queries = 0 if expected_actions == 0 else (expected_actions + 4) // 5
    expected_arm = (
        "pi05_plus_aegis_joint_velocity_adapter"
        if label == "baseline"
        else "pi05_plus_aegis_joint_velocity_adapter_plus_link56_psf"
    )
    expected_first_query_execution = (
        "live_inference_and_cache"
        if label == "baseline"
        else "paired_cache_reuse"
    )
    expected_paired_source_arm = (
        None
        if label == "baseline"
        else "pi05_plus_aegis_joint_velocity_adapter"
    )
    audit.check(len(actions) == expected_actions, "%s_action_count_differs" % label)
    audit.check(len(queries) == expected_queries, "%s_query_count_differs" % label)
    audit.check(
        provider.get("source")
        == "live_pi05_with_shared_current_q36_then_per_arm_own_observations",
        "%s_source_differs" % label,
    )
    audit.check(provider.get("recorded_suffix_actions_executed") is False, "%s_recorded_action_replay" % label)
    audit.check(
        provider.get("first_query_execution")
        == expected_first_query_execution,
        "%s_first_query_execution_differs" % label,
    )
    audit.check(
        provider.get("paired_first_query_source_arm")
        == expected_paired_source_arm,
        "%s_paired_first_query_source_differs" % label,
    )
    audit.check(
        provider.get(
            "historical_first_live_query_action_chunk_sha256_diagnostic"
        )
        == EXPECTED_FIRST_CHUNK_SHA256,
        "%s_historical_query_hash_diagnostic_differs" % label,
    )
    audit.check(provider.get("first_query_index") == 36, "%s_first_query_differs" % label)
    audit.check(provider.get("arm") == expected_arm, "%s_provider_arm_differs" % label)

    query_map: Dict[int, Mapping[str, Any]] = {}
    for offset, query in enumerate(queries):
        query_index = 36 + offset
        query_map[query_index] = query
        audit.check(query.get("arm") == expected_arm, "%s_query_%d_arm_differs" % (label, query_index))
        _validate_observation_record(audit, query, "%s_query_%d" % (label, query_index))
        returned = audit.sequence(query.get("returned_actions"), "%s_query_%d_returned" % (label, query_index))
        valid_matrix = len(returned) == 10 and all(
            isinstance(row, Sequence) and not isinstance(row, (str, bytes)) and len(row) == 7
            for row in returned
        )
        audit.check(valid_matrix, "%s_query_%d_shape_invalid" % (label, query_index))
        observed_hash = query.get("returned_actions_sha256")
        try:
            expected_hash = _array_sha256(returned) if valid_matrix else None
        except ArtifactContractError:
            expected_hash = None
        audit.check(observed_hash == expected_hash, "%s_query_%d_action_hash_differs" % (label, query_index))
        local_index = offset * 5
        for field, expected in (
            ("query_index", query_index),
            ("local_action_index", local_index),
            ("source_action_index", 180 + local_index),
            ("rng_seed", 2026691220 + query_index),
            ("returned_action_shape", [10, 7]),
        ):
            audit.check(query.get(field) == expected, "%s_query_%d_%s_differs" % (label, query_index, field))
        _validate_query_execution(audit, query, label, query_index)
    audit.check(sorted(query_map) == list(range(36, 36 + expected_queries)), "%s_query_population_differs" % label)

    returned_observations = audit.sequence(
        audit.mapping(arm.get("task"), "%s_task_provider" % label).get("returned_observation_sha256_ledger"),
        "%s_returned_observations" % label,
    )
    initial_z = _vector(audit, provider.get("initial_aegis_z_fixed_from_historical_action_179"), 3, "%s_initial_z" % label)
    historical_action_179 = historical["actions"][179]
    historical_action_180 = historical["actions"][180]
    expected_initial_z = historical_action_179["qp"]["z_after"]
    audit.check(_same_numeric(initial_z, expected_initial_z), "%s_initial_z_historical_differs" % label)
    previous_z: Any = initial_z
    by_source: Dict[int, Mapping[str, Any]] = {}
    for local_index, action in enumerate(actions):
        action_label = "%s_action_%d" % (label, local_index)
        audit.check(action.get("arm") == expected_arm, "%s_arm_differs" % action_label)
        _validate_observation_record(audit, action, action_label)
        query_index = 36 + local_index // 5
        chunk_offset = local_index % 5
        query = query_map.get(query_index, {})
        returned = query.get("returned_actions", ()) if isinstance(query, Mapping) else ()
        expected_raw = returned[chunk_offset] if len(returned) > chunk_offset else None
        for field, expected in (
            ("local_action_index", local_index),
            ("source_action_index", 180 + local_index),
            ("query_index", query_index),
            ("query_chunk_offset", chunk_offset),
        ):
            audit.check(action.get(field) == expected, "%s_%s_differs" % (action_label, field))
        audit.check(_same_numeric(action.get("nominal_raw"), expected_raw), "%s_nominal_raw_differs" % action_label)
        if chunk_offset == 0 and isinstance(query, Mapping):
            for field in _OBSERVATION_FIELDS + ("policy_input_fingerprint_sha256",):
                audit.check(_canonical(action.get(field)) == _canonical(query.get(field)), "%s_query_observation_%s_differs" % (action_label, field))
        if local_index > 0:
            preceding = returned_observations[local_index - 1] if local_index - 1 < len(returned_observations) else None
            audit.check(action.get("native_observation_sha256") == preceding, "%s_stale_observation" % action_label)
        before = _vector(audit, action.get("aegis_z_before"), 3, "%s_z_before" % action_label)
        after = _vector(audit, action.get("aegis_z_after"), 3, "%s_z_after" % action_label)
        audit.check(_same_numeric(before, previous_z), "%s_z_chain_differs" % action_label)
        audit.check(math.isclose(_norm(after), 1.0, rel_tol=0.0, abs_tol=1.0e-10), "%s_z_after_not_unit" % action_label)
        nominal = _vector(audit, action.get("nominal_translational"), 7, "%s_nominal" % action_label)
        executed = _vector(audit, action.get("aegis_executed"), 7, "%s_executed" % action_label)
        correction = _norm(tuple(a - b for a, b in zip(executed, nominal)))
        audit.check(_close(audit.number(action.get("aegis_correction_l2"), "%s_correction" % action_label), correction), "%s_correction_differs" % action_label)
        qp = audit.mapping(action.get("aegis_qp"), "%s_qp" % action_label)
        context = audit.mapping(qp.get("context"), "%s_qp_context" % action_label)
        audit.check(qp.get("status") == "solved", "%s_qp_not_solved" % action_label)
        audit.check(qp.get("solver_status") in ("optimal", "optimal_inaccurate"), "%s_qp_solver_status_differs" % action_label)
        for observed, expected, suffix in (
            (qp.get("z_before"), before, "qp_z_before"),
            (qp.get("z_after"), after, "qp_z_after"),
            (context.get("z_before"), before, "context_z_before"),
            (context.get("z_after"), after, "context_z_after"),
            (context.get("q1_diag"), [0.06, 0.12, 0.11], "q1_diag"),
            (context.get("nominal_translational"), nominal, "context_nominal"),
            (context.get("executed_action"), executed, "context_executed"),
        ):
            audit.check(_same_numeric(observed, expected), "%s_%s_differs" % (action_label, suffix))
        previous_z = after
        by_source[180 + local_index] = action

    # Historical action 180 remains branch/proxy authority only.  The nominal
    # policy output and all AEGIS outputs are live-run diagnostics, because a
    # fresh policy-server inference need not be bitwise identical to the old
    # Table-1 server process.
    expected_historical_inputs = _aegis_input_projection(
        historical_action_180,
        historical=True,
        include_nominal=False,
    )
    historical_outputs_diagnostic = _aegis_output_projection(
        historical_action_180,
        historical=True,
    )
    stored_historical_inputs = audit.mapping(
        provider.get("historical_action_180_required_aegis_inputs"),
        "%s_historical_inputs" % label,
    )
    audit.check(
        _canonical(stored_historical_inputs)
        == _canonical(expected_historical_inputs),
        "%s_historical_required_inputs_differs" % label,
    )
    # These records prove that the producer disclosed the old comparison; they
    # are not used to accept the current live output.
    stored_historical_outputs = provider.get(
        "historical_action_180_aegis_full_output_diagnostic"
    )
    audit.check(
        _canonical(stored_historical_outputs)
        == _canonical(historical_outputs_diagnostic),
        "%s_historical_output_diagnostic_differs" % label,
    )
    current_first_input = (
        _aegis_input_projection(
            actions[0], historical=False, include_nominal=False
        )
        if actions
        else {}
    )
    current_first_output = (
        _aegis_output_projection(actions[0], historical=False)
        if actions
        else {}
    )
    expected_input_match = bool(
        current_first_input
        and all(
            _same_numeric(current_first_input.get(key), expected)
            for key, expected in expected_historical_inputs.items()
        )
    )
    expected_action_match = bool(
        actions
        and _same_numeric(
            actions[0].get("aegis_executed"),
            historical_action_180.get("executed"),
        )
    )
    expected_internal_match = bool(
        current_first_output
        and _canonical(
            {
                key: value
                for key, value in current_first_output.items()
                if key != "executed_action"
            }
        )
        == _canonical(
            {
                key: value
                for key, value in historical_outputs_diagnostic.items()
                if key != "executed_action"
            }
        )
    )
    expected_full_output_match = bool(
        current_first_output
        and _canonical(current_first_output)
        == _canonical(historical_outputs_diagnostic)
    )
    for field, expected in (
        ("first_aegis_input_binding_matches_historical", expected_input_match),
        ("first_aegis_action_matches_historical", expected_action_match),
        (
            "first_aegis_internal_output_matches_historical",
            expected_internal_match,
        ),
        (
            "first_aegis_full_output_matches_historical_diagnostic",
            expected_full_output_match,
        ),
    ):
        audit.check(
            provider.get(field) is expected,
            "%s_%s_flag_differs" % (label, field),
        )
    # Only the seven historical branch/proxy inputs are authoritative.  The
    # historical nominal and output/action comparisons above are diagnostics.
    audit.check(expected_input_match, "%s_first_input_binding_false" % label)
    if actions:
        first_qp = actions[0].get("aegis_qp", {})
        first_context = first_qp.get("context", {})
        for key, expected in expected_historical_inputs.items():
            audit.check(
                _same_numeric(first_context.get(key), expected),
                "%s_action180_context_%s_historical_differs" % (label, key),
            )
    commands = audit.sequence(arm.get("command_trace"), "%s_provider_commands" % label)
    for index, value in enumerate(commands):
        command = audit.mapping(value, "%s_provider_command_%d" % (label, index))
        source = audit.integer(command.get("source_action_index"), "%s_provider_command_source" % label)
        action = by_source.get(source)
        audit.check(action is not None, "%s_command_without_live_action" % label)
        if action is not None:
            audit.check(_same_numeric(command.get("source_action"), action.get("aegis_executed")), "%s_command_source_action_differs" % label)
    audit.check(
        provider.get("live_aegis_action_sequence_sha256")
        == hashlib.sha256(
            _canonical([row.get("aegis_executed") for row in actions])
        ).hexdigest(),
        "%s_live_aegis_sequence_hash_differs" % label,
    )
    return {"provider": provider, "actions": actions, "queries": queries}


def _live_provider_contract(
    record: Mapping[str, Any],
    *,
    expected_actions: int,
    policy_noise_seed: int,
) -> bool:
    """Independently reconstruct the producer's live-query/action contract."""

    try:
        provider = record["provider"]
        actions = record["actions"]
        queries = record["queries"]
        expected_queries = (
            0 if expected_actions == 0 else (expected_actions + 4) // 5
        )
        if len(actions) != expected_actions or len(queries) != expected_queries:
            return False
        first_execution = provider.get("first_query_execution")
        paired_source = provider.get("paired_first_query_source_arm")
        if first_execution not in (
            "live_inference_and_cache",
            "paired_cache_reuse",
        ):
            return False
        if (
            first_execution == "live_inference_and_cache"
            and paired_source is not None
        ) or (
            first_execution == "paired_cache_reuse"
            and not isinstance(paired_source, str)
        ):
            return False
        query_map = {int(row["query_index"]): row for row in queries}
        if sorted(query_map) != list(range(36, 36 + expected_queries)):
            return False
        for offset, query_index in enumerate(range(36, 36 + expected_queries)):
            query = query_map[query_index]
            returned = query.get("returned_actions")
            valid_returned = bool(
                isinstance(returned, Sequence)
                and not isinstance(returned, (str, bytes))
                and len(returned) == 10
                and all(
                    isinstance(row, Sequence)
                    and not isinstance(row, (str, bytes))
                    and len(row) == 7
                    and all(
                        not isinstance(value, bool)
                        and isinstance(value, (int, float))
                        and math.isfinite(float(value))
                        for value in row
                    )
                    for row in returned
                )
            )
            expected_execution = (
                "paired_cache_reuse"
                if query_index == 36
                and first_execution == "paired_cache_reuse"
                else "live_inference"
            )
            execution_valid = bool(
                query.get("query_execution") == expected_execution
                and query.get("inference_performed")
                is (expected_execution == "live_inference")
            )
            if expected_execution == "live_inference":
                execution_valid = bool(
                    execution_valid
                    and query.get("paired_cache_source_arm") is None
                    and query.get("paired_cache_source_query_index") is None
                    and query.get(
                        "paired_cache_source_returned_actions_sha256"
                    )
                    is None
                )
            else:
                execution_valid = bool(
                    execution_valid
                    and query.get("paired_cache_source_arm") == paired_source
                    and query.get("paired_cache_source_query_index") == 36
                    and query.get(
                        "paired_cache_source_returned_actions_sha256"
                    )
                    == query.get("returned_actions_sha256")
                )
            if (
                query.get("query_index") != query_index
                or query.get("local_action_index") != offset * 5
                or query.get("source_action_index") != 180 + offset * 5
                or query.get("rng_seed") != policy_noise_seed + query_index
                or query.get("returned_action_shape") != [10, 7]
                or not valid_returned
                or query.get("returned_actions_sha256")
                != (_array_sha256(returned) if valid_returned else None)
                or not execution_valid
            ):
                return False
        for local_index, row in enumerate(actions):
            query_index = 36 + local_index // 5
            chunk_offset = local_index % 5
            query = query_map.get(query_index)
            returned = query.get("returned_actions") if query else None
            if (
                query is None
                or row.get("local_action_index") != local_index
                or row.get("source_action_index") != 180 + local_index
                or row.get("query_index") != query_index
                or row.get("query_chunk_offset") != chunk_offset
                or not isinstance(returned, Sequence)
                or chunk_offset >= len(returned)
                or _canonical(row.get("nominal_raw"))
                != _canonical(returned[chunk_offset])
                or (
                    chunk_offset == 0
                    and row.get("native_observation_sha256")
                    != query.get("native_observation_sha256")
                )
            ):
                return False
        return bool(
            provider.get("recorded_suffix_actions_executed") is False
            and provider.get("first_query_index") == 36
            and provider.get("source")
            == "live_pi05_with_shared_current_q36_then_per_arm_own_observations"
        )
    except (KeyError, TypeError, ValueError, ArtifactContractError):
        return False


def _own_observation_chain_valid(
    record: Mapping[str, Any], arm: Mapping[str, Any]
) -> bool:
    returned = arm.get("task", {}).get(
        "returned_observation_sha256_ledger"
    )
    if not isinstance(returned, Sequence):
        return False
    for row in list(record["actions"]) + list(record["queries"]):
        local_index = row.get("local_action_index")
        if isinstance(local_index, bool) or not isinstance(local_index, int):
            return False
        if local_index == 0:
            continue
        preceding = local_index - 1
        if (
            preceding >= len(returned)
            or row.get("native_observation_sha256") != returned[preceding]
        ):
            return False
    return bool(record["queries"])


def _aegis_state_and_command_chain_valid(
    record: Mapping[str, Any], arm: Mapping[str, Any]
) -> bool:
    actions = record["actions"]
    commands = arm.get("command_trace")
    initial_z = record["provider"].get(
        "initial_aegis_z_fixed_from_historical_action_179"
    )
    if (
        not actions
        or not isinstance(commands, Sequence)
        or not isinstance(initial_z, Sequence)
    ):
        return False
    previous_z = initial_z
    by_source: Dict[int, Mapping[str, Any]] = {}
    try:
        for row in actions:
            before = row.get("aegis_z_before")
            after = row.get("aegis_z_after")
            qp = row.get("aegis_qp")
            context = qp.get("context") if isinstance(qp, Mapping) else None
            if (
                not _same_numeric(before, previous_z)
                or len(before) != 3
                or len(after) != 3
                or not math.isclose(
                    _norm(after), 1.0, rel_tol=0.0, abs_tol=1.0e-10
                )
                or not isinstance(context, Mapping)
                or qp.get("status") != "solved"
                or not _same_numeric(qp.get("z_before"), before)
                or not _same_numeric(qp.get("z_after"), after)
                or not _same_numeric(context.get("z_before"), before)
                or not _same_numeric(context.get("z_after"), after)
                or not _same_numeric(
                    context.get("q1_diag"), [0.06, 0.12, 0.11]
                )
                or not _same_numeric(
                    context.get("nominal_translational"),
                    row.get("nominal_translational"),
                )
                or not _same_numeric(
                    context.get("executed_action"),
                    row.get("aegis_executed"),
                )
            ):
                return False
            previous_z = after
            by_source[int(row["source_action_index"])] = row
        for command in commands:
            action = by_source.get(int(command["source_action_index"]))
            if action is None or not _same_numeric(
                command.get("source_action"), action.get("aegis_executed")
            ):
                return False
    except (KeyError, TypeError, ValueError):
        return False
    return True


def _first_action_pair_metrics(
    baseline: Mapping[str, Any], psf: Mapping[str, Any]
) -> Dict[str, Any]:
    if not baseline["actions"] or not psf["actions"]:
        return {
            "first_current_action_180_aegis_inputs_identical": False,
            "first_current_action_180_aegis_outputs_identical": False,
            "first_current_action_180_aegis_pair_contract_valid": False,
            "baseline_first_current_aegis_input_sha256": None,
            "psf_first_current_aegis_input_sha256": None,
            "baseline_first_current_aegis_output_sha256": None,
            "psf_first_current_aegis_output_sha256": None,
        }
    baseline_row = baseline["actions"][0]
    psf_row = psf["actions"][0]
    baseline_input = _aegis_input_projection(
        baseline_row, historical=False, include_nominal=True
    )
    psf_input = _aegis_input_projection(
        psf_row, historical=False, include_nominal=True
    )
    baseline_output = _aegis_output_projection(
        baseline_row, historical=False
    )
    psf_output = _aegis_output_projection(psf_row, historical=False)
    baseline_input_bytes = _canonical(baseline_input)
    psf_input_bytes = _canonical(psf_input)
    baseline_output_bytes = _canonical(baseline_output)
    psf_output_bytes = _canonical(psf_output)
    indexed = bool(
        baseline_row.get("local_action_index") == 0
        and baseline_row.get("source_action_index") == 180
        and psf_row.get("local_action_index") == 0
        and psf_row.get("source_action_index") == 180
    )
    inputs_identical = bool(
        indexed and baseline_input and baseline_input_bytes == psf_input_bytes
    )
    outputs_identical = bool(
        indexed and baseline_output and baseline_output_bytes == psf_output_bytes
    )
    return {
        "first_current_action_180_aegis_inputs_identical": inputs_identical,
        "first_current_action_180_aegis_outputs_identical": outputs_identical,
        "first_current_action_180_aegis_pair_contract_valid": bool(
            inputs_identical and outputs_identical
        ),
        "baseline_first_current_aegis_input_sha256": hashlib.sha256(
            baseline_input_bytes
        ).hexdigest(),
        "psf_first_current_aegis_input_sha256": hashlib.sha256(
            psf_input_bytes
        ).hexdigest(),
        "baseline_first_current_aegis_output_sha256": hashlib.sha256(
            baseline_output_bytes
        ).hexdigest(),
        "psf_first_current_aegis_output_sha256": hashlib.sha256(
            psf_output_bytes
        ).hexdigest(),
    }


def _feedback(
    audit: _Audit,
    baseline: Mapping[str, Any],
    psf: Mapping[str, Any],
) -> Dict[str, Any]:
    baseline_actions = baseline["actions"]
    psf_actions = psf["actions"]
    divergence = None
    for index, (left, right) in enumerate(zip(baseline_actions, psf_actions)):
        if left.get("official_integration_state_raw_bytes_sha256") != right.get("official_integration_state_raw_bytes_sha256"):
            divergence = index
            break
    bq = {int(row["query_index"]): row for row in baseline["queries"]}
    pq = {int(row["query_index"]): row for row in psf["queries"]}
    first_baseline = bq.get(36, {})
    first_psf = pq.get(36, {})
    baseline_inferred = [
        int(row["query_index"])
        for row in baseline["queries"]
        if row.get("query_execution") == "live_inference"
        and row.get("inference_performed") is True
    ]
    baseline_cached = [
        int(row["query_index"])
        for row in baseline["queries"]
        if row.get("query_execution") == "paired_cache_reuse"
        or row.get("inference_performed") is False
    ]
    psf_inferred = [
        int(row["query_index"])
        for row in psf["queries"]
        if row.get("query_execution") == "live_inference"
        and row.get("inference_performed") is True
    ]
    psf_cached = [
        int(row["query_index"])
        for row in psf["queries"]
        if row.get("query_execution") == "paired_cache_reuse"
        or row.get("inference_performed") is False
    ]
    baseline_provider = baseline["provider"]
    psf_provider = psf["provider"]
    only_psf_query_36_reused = bool(
        baseline_provider.get("first_query_execution")
        == "live_inference_and_cache"
        and psf_provider.get("first_query_execution") == "paired_cache_reuse"
        and baseline_cached == []
        and baseline_inferred
        == [int(row["query_index"]) for row in baseline["queries"]]
        and psf_cached == [36]
        and psf_inferred
        == [
            int(row["query_index"])
            for row in psf["queries"]
            if int(row["query_index"]) > 36
        ]
        and first_psf.get("paired_cache_source_arm")
        == baseline_provider.get("arm")
        and first_psf.get("paired_cache_source_query_index") == 36
        and first_psf.get("paired_cache_source_returned_actions_sha256")
        == first_baseline.get("returned_actions_sha256")
    )
    first_identical = bool(
        36 in bq
        and 36 in pq
        and first_baseline.get("policy_input_fingerprint_sha256")
        == first_psf.get("policy_input_fingerprint_sha256")
        and first_baseline.get("rng_seed")
        == first_psf.get("rng_seed")
        == 2026691256
        and first_baseline.get("source_action_index")
        == first_psf.get("source_action_index")
        == 180
        and first_baseline.get("local_action_index")
        == first_psf.get("local_action_index")
        == 0
        and first_baseline.get("returned_actions_sha256")
        == first_psf.get("returned_actions_sha256")
        and _canonical(first_baseline.get("returned_actions"))
        == _canonical(first_psf.get("returned_actions"))
        and only_psf_query_36_reused
    )
    scheduled: List[int] = []
    distinct: List[int] = []
    if divergence is not None:
        divergence_source = 180 + divergence
        for query_index in sorted(set(bq).intersection(pq)):
            if int(bq[query_index]["source_action_index"]) <= divergence_source:
                continue
            scheduled.append(query_index)
            if bq[query_index].get("policy_input_fingerprint_sha256") != pq[query_index].get("policy_input_fingerprint_sha256"):
                distinct.append(query_index)
    current_first_chunk_sha256 = first_baseline.get(
        "returned_actions_sha256"
    )
    cache_producer = {
        "producer_arm": baseline_provider.get("arm"),
        "query_index": 36,
        "rng_seed": first_baseline.get("rng_seed"),
        "policy_input_fingerprint_sha256": first_baseline.get(
            "policy_input_fingerprint_sha256"
        ),
        "returned_actions": first_baseline.get("returned_actions"),
        "returned_actions_sha256": current_first_chunk_sha256,
    }
    cache_reuse = {
        "consumer_arm": psf_provider.get("arm"),
        "query_index": 36,
        "rng_seed": first_psf.get("rng_seed"),
        "policy_input_fingerprint_sha256": first_psf.get(
            "policy_input_fingerprint_sha256"
        ),
    }
    cache_contract_valid = bool(
        first_baseline
        and first_psf
        and cache_producer["query_index"] == cache_reuse["query_index"] == 36
        and cache_producer["rng_seed"] == cache_reuse["rng_seed"]
        and cache_producer["policy_input_fingerprint_sha256"]
        == cache_reuse["policy_input_fingerprint_sha256"]
        and cache_producer["producer_arm"] != cache_reuse["consumer_arm"]
    )
    result = {
        "first_divergent_action_input_local_index": divergence,
        "first_divergent_action_input_source_index": None if divergence is None else 180 + divergence,
        "first_live_query_identical": first_identical,
        "only_psf_query_36_reused_paired_cache": only_psf_query_36_reused,
        "baseline_live_inference_query_indexes": baseline_inferred,
        "baseline_paired_cache_reuse_query_indexes": baseline_cached,
        "psf_live_inference_query_indexes": psf_inferred,
        "psf_paired_cache_reuse_query_indexes": psf_cached,
        "current_first_live_query_action_chunk_sha256": (
            current_first_chunk_sha256
        ),
        "historical_first_live_query_action_chunk_sha256_diagnostic": (
            EXPECTED_FIRST_CHUNK_SHA256
        ),
        "historical_first_live_query_action_chunk_matches_diagnostic": bool(
            current_first_chunk_sha256 == EXPECTED_FIRST_CHUNK_SHA256
        ),
        "scheduled_query_indexes_after_divergence": scheduled,
        "post_divergence_query_indexes_with_distinct_policy_inputs": distinct,
        "post_divergence_policy_inputs_differ": bool(distinct),
        "post_divergence_own_observations_used": bool(divergence is not None and scheduled),
    }
    result.update(_first_action_pair_metrics(baseline, psf))
    result["paired_first_query_cache"] = {
        "schema_version": "vlsa_poisson_paired_first_query_cache.v1",
        "first_query_index": 36,
        "store_count": 1 if first_baseline else 0,
        "reuse_count": 1 if first_psf else 0,
        "producer": cache_producer if first_baseline else None,
        "reuse": cache_reuse if first_psf else None,
        "contract_valid": cache_contract_valid,
    }
    audit.check(first_identical, "first_live_query_not_identical")
    audit.check(
        only_psf_query_36_reused,
        "paired_query36_cache_execution_contract_differs",
    )
    return result


def _validate_first_live_pair(
    audit: _Audit,
    baseline: Mapping[str, Any],
    psf: Mapping[str, Any],
) -> bool:
    """Use the paired live q36/action-180 result as current-run authority."""

    if not baseline["queries"] or not psf["queries"] or not baseline["actions"] or not psf["actions"]:
        audit.check(False, "paired_q36_or_action180_missing")
        return False
    baseline_query = baseline["queries"][0]
    psf_query = psf["queries"][0]
    audit.check(
        _canonical(baseline_query.get("returned_actions"))
        == _canonical(psf_query.get("returned_actions")),
        "paired_q36_action_chunk_differs",
    )
    for field in _OBSERVATION_FIELDS + (
        "policy_input_fingerprint_sha256",
        "rng_seed",
        "source_action_index",
        "local_action_index",
    ):
        audit.check(
            _canonical(baseline_query.get(field))
            == _canonical(psf_query.get(field)),
            "paired_q36_%s_differs" % field,
        )
    baseline_action = baseline["actions"][0]
    psf_action = psf["actions"][0]
    for field in ("nominal_raw", "nominal_translational", "aegis_executed"):
        audit.check(
            _canonical(baseline_action.get(field))
            == _canonical(psf_action.get(field)),
            "paired_action180_%s_differs" % field,
        )
    pair_metrics = _first_action_pair_metrics(baseline, psf)
    audit.check(
        pair_metrics["first_current_action_180_aegis_inputs_identical"],
        "paired_action180_aegis_inputs_differ",
    )
    audit.check(
        pair_metrics["first_current_action_180_aegis_outputs_identical"],
        "paired_action180_aegis_outputs_differ",
    )
    return not any(item.startswith("paired_q36") or item.startswith("paired_action180") for item in audit.discrepancies)


def _safe_child(root: Path, relative: Any, label: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ArtifactContractError("%s path is not a safe relative path" % label)
    parts = Path(relative).parts
    if any(part in ("", ".", "..") for part in parts):
        raise ArtifactContractError("%s path traverses outside the run" % label)
    current = root
    for part in parts:
        current = current / part
        if current.is_symlink():
            raise ArtifactContractError("%s path traverses a symlink" % label)
    if not current.is_file():
        raise ArtifactContractError("%s file is missing" % label)
    return current


def _validate_video(
    audit: _Audit,
    value: Any,
    run_root: Path,
    label: str,
    *,
    completed_actions: int,
    contact_terminal: bool,
) -> bool:
    video = audit.mapping(value, "%s_video" % label)
    expected_frames = completed_actions + (2 if contact_terminal else 1)
    for field, expected in (
        ("fps", 30),
        ("codec", "libx264"),
        ("coverage_scope", "branch_boundary_through_live_suffix_terminal"),
        ("playback_not_wall_clock", True),
        ("real_simulation_frames", True),
        ("two_dimensional_safety_overlay", False),
        ("decoded_successfully", True),
        ("frame_count", expected_frames),
        ("decoded_frame_count", expected_frames),
        ("decoded_frame_shape", [1024, 1024, 3]),
        ("source_action_index_start", 179),
        ("source_action_index_end", 179 + completed_actions + (1 if contact_terminal else 0)),
    ):
        audit.check(video.get(field) == expected, "%s_video_%s_differs" % (label, field))
    snapshots = [
        audit.mapping(row, "%s_snapshot_%d" % (label, index))
        for index, row in enumerate(audit.sequence(video.get("snapshot_trace"), "%s_snapshots" % label))
    ]
    audit.check(len(snapshots) == expected_frames, "%s_snapshot_count_differs" % label)
    for index, row in enumerate(snapshots):
        audit.check(row.get("frame_index") == index, "%s_snapshot_index_differs" % label)
        audit.check(row.get("height") == 1024 and row.get("width") == 1024, "%s_snapshot_shape_differs" % label)
        _sha256(audit, row.get("source_array_sha256"), "%s_snapshot_hash" % label)
        if index == 0:
            expected = ("branch_boundary_pre_action", None, 179)
        elif contact_terminal and index == expected_frames - 1:
            expected = ("partial_action_terminal_state", completed_actions, 180 + completed_actions)
        else:
            local = index - 1
            expected = ("completed_high_level_post_step", local, 180 + local)
        audit.check(
            (row.get("snapshot_kind"), row.get("local_action_index"), row.get("source_action_index")) == expected,
            "%s_snapshot_cadence_differs" % label,
        )
    try:
        video_path = _safe_child(run_root, video.get("path"), "%s video" % label)
        audit.check(sha256_file(video_path) == video.get("sha256"), "%s_video_hash_differs" % label)
        audit.check(video_path.stat().st_size == video.get("byte_count"), "%s_video_size_differs" % label)
        import imageio.v2 as imageio
        import numpy as np

        lossless_arrays = {}
        for endpoint, snapshot_index in (("branch", 0), ("terminal", -1)):
            record = audit.mapping(video.get(endpoint + "_lossless"), "%s_%s_lossless" % (label, endpoint))
            path = _safe_child(run_root, record.get("path"), "%s %s lossless" % (label, endpoint))
            audit.check(sha256_file(path) == record.get("sha256"), "%s_%s_lossless_hash_differs" % (label, endpoint))
            array = np.load(path, allow_pickle=False)
            audit.check(list(array.shape) == [1024, 1024, 3] and array.dtype == np.uint8, "%s_%s_lossless_shape_differs" % (label, endpoint))
            observed_array_hash = _numpy_array_sha256(array)
            audit.check(observed_array_hash == record.get("array_sha256"), "%s_%s_array_hash_differs" % (label, endpoint))
            if snapshots:
                audit.check(observed_array_hash == snapshots[snapshot_index].get("source_array_sha256"), "%s_%s_snapshot_hash_differs" % (label, endpoint))
            lossless_arrays[endpoint] = array
        decoded: List[Any] = []
        reader = imageio.get_reader(str(video_path))
        try:
            for frame in reader:
                decoded.append(np.asarray(frame))
        finally:
            reader.close()
        audit.check(len(decoded) == expected_frames, "%s_decoded_count_differs" % label)
        audit.check(all(list(frame.shape) == [1024, 1024, 3] for frame in decoded), "%s_decoded_shape_differs" % label)
        hashes = [_numpy_array_sha256(frame) for frame in decoded]
        audit.check(hashlib.sha256(_canonical(hashes)).hexdigest() == video.get("decoded_frame_sequence_sha256"), "%s_decoded_sequence_hash_differs" % label)
        diagnostics = audit.mapping(video.get("decoded_endpoint_diagnostics"), "%s_endpoint_diagnostics" % label)
        for endpoint, decoded_frame in (("branch", decoded[0] if decoded else None), ("terminal", decoded[-1] if decoded else None)):
            if decoded_frame is None or endpoint not in lossless_arrays:
                continue
            difference = decoded_frame.astype(np.float64) - lossless_arrays[endpoint].astype(np.float64)
            mae = float(np.mean(np.abs(difference)))
            mse = float(np.mean(np.square(difference)))
            psnr = float("inf") if mse == 0.0 else 20.0 * math.log10(255.0 / math.sqrt(mse))
            stored = audit.mapping(diagnostics.get(endpoint), "%s_%s_diagnostic" % (label, endpoint))
            audit.check(_close(audit.number(stored.get("mae"), "%s_%s_mae" % (label, endpoint)), mae), "%s_%s_mae_differs" % (label, endpoint))
            stored_psnr = stored.get("psnr_db")
            psnr_matches = (math.isinf(psnr) and stored_psnr == float("inf")) or (
                not math.isinf(psnr) and _close(audit.number(stored_psnr, "%s_%s_psnr" % (label, endpoint)), psnr)
            )
            audit.check(psnr_matches and mae <= 12.0 and psnr >= 20.0, "%s_%s_endpoint_decode_failed" % (label, endpoint))
    except Exception as error:
        audit.check(False, "%s_video_file_validation_failed:%s" % (label, error))
    return not any(item.startswith(label + "_video") or item.startswith(label + "_snapshot") or item.startswith(label + "_decoded") for item in audit.discrepancies)


def _validate_periodic_clearance(
    audit: _Audit,
    arm: Mapping[str, Any],
    physics: Sequence[Mapping[str, Any]],
    label: str,
    expectation: Mapping[str, Any],
) -> bool:
    stride = int(expectation["clearance_stride"])
    flags: List[bool] = []
    values: List[float] = []
    for index, row in enumerate(physics):
        flag = audit.boolean(row.get("full_surface_clearance_observed_this_substep"), "%s_clearance_flag_%d" % (label, index))
        expected_flag = (index + 1) % stride == 0 or (index + 1) == int(expectation["physics_substeps"])
        audit.check(flag is expected_flag, "%s_clearance_stride_differs" % label)
        flags.append(flag)
        values.append(audit.number(row.get("cumulative_full_robot_surface_clearance_lower_bound_m"), "%s_clearance_%d" % (label, index)))
    for previous, current in zip(values, values[1:]):
        audit.check(current <= previous + 1.0e-10, "%s_clearance_not_cumulative" % label)
    record = audit.mapping(arm.get("conservative_full_robot_clearance"), "%s_clearance_record" % label)
    audit.check(record.get("observation_stride_physics_substeps") == stride, "%s_clearance_record_stride_differs" % label)
    audit.check(record.get("observation_count_including_branch") == 1 + sum(flags), "%s_clearance_observation_count_differs" % label)
    audit.check(record.get("continuous_every_substep_certificate") is False, "%s_clearance_falsely_continuous" % label)
    if values:
        audit.check(_close(audit.number(record.get("minimum_full_surface_lower_bound_m"), "%s_min_clearance" % label), min(values)), "%s_min_clearance_differs" % label)
    return bool(record.get("available") is True and values and min(values) > 0.0)


def _post_correction_motion(
    audit: _Audit,
    commands: Sequence[Mapping[str, Any]],
    physics: Sequence[Mapping[str, Any]],
    first_correction: Optional[int],
    expectation: Mapping[str, Any],
) -> Dict[str, Any]:
    """Reconstruct producer motion semantics, including partial-contact rows."""

    selected_commands = [] if first_correction is None else [
        row
        for row in commands
        if audit.integer(row.get("physical_boundary"), "motion_boundary")
        >= int(first_correction)
    ]
    keys = {
        (
            audit.integer(row.get("source_action_index"), "motion_source"),
            audit.integer(row.get("inner_control_index"), "motion_inner"),
        )
        for row in selected_commands
    }
    selected_physics = [
        row
        for row in physics
        if (row.get("source_action_index"), row.get("inner_control_index")) in keys
    ]
    corrections = [
        audit.number(row.get("correction_l2_rad_s"), "motion_correction")
        for row in selected_commands
    ]
    executed = [
        _vector(audit, row.get("executed_qdot_rad_s"), 7, "motion_executed")
        for row in selected_commands
    ]
    correction_integral = sum(
        value * float(expectation["control_dt_s"]) for value in corrections
    )
    command_integral = sum(
        _norm(value) * float(expectation["control_dt_s"]) for value in executed
    )
    measured_integral = sum(
        _norm(_vector(audit, row.get("measured_qvel_rad_s"), 7, "motion_measured"))
        * float(expectation["physics_dt_s"])
        for row in selected_physics
    )
    cartesian_path = 0.0
    complete_intervals = 0
    for command in selected_commands:
        key = (command.get("source_action_index"), command.get("inner_control_index"))
        rows = sorted(
            (
                row
                for row in selected_physics
                if (row.get("source_action_index"), row.get("inner_control_index")) == key
                and row.get("eef_position_world_m") is not None
            ),
            key=lambda row: audit.integer(row.get("physics_substep_index"), "motion_substep"),
        )
        if len(rows) != int(expectation["substeps_per_control"]):
            continue
        previous = _vector(
            audit,
            command.get("eef_position_before_update_world_m"),
            3,
            "motion_eef_start",
        )
        for row in rows:
            current = _vector(
                audit, row.get("eef_position_world_m"), 3, "motion_eef"
            )
            cartesian_path += math.sqrt(
                sum((right - left) ** 2 for left, right in zip(previous, current))
            )
            previous = current
        complete_intervals += 1
    zero_count = sum(_norm(value) <= 1.0e-8 for value in executed)
    return {
        "first_correction_physical_boundary": first_correction,
        "command_update_count": len(selected_commands),
        "physics_substep_count": len(selected_physics),
        "complete_command_interval_count": complete_intervals,
        "maximum_correction_norm_rad_s": max(corrections, default=0.0),
        "filter_correction_integral_rad": correction_integral,
        "executed_command_integral_rad": command_integral,
        "measured_joint_motion_integral_rad": measured_integral,
        "cartesian_path_length_m": cartesian_path,
        "zero_command_fraction": (
            1.0 if not executed else float(zero_count) / len(executed)
        ),
    }


def _independent_classification(metrics: Mapping[str, Any], expectation: Mapping[str, Any]) -> Dict[str, Any]:
    """Reconstruct classifier-v1 without calling the producer classifier."""

    thresholds = expectation["thresholds"]
    apparatus_complete = all(
        metrics.get(field) is True
        for field in (
            "exact_paired_start",
            "shared_prefix_complete",
            "live_policy_contract_valid",
            "aegis_contract_valid",
            "both_nominal_commands_within_dynamic_joint_bounds",
            "videos_complete_and_decodable",
            "psf_qp_contract_valid",
            "static_selected_obstacle_admissible",
            "literal_contact_checked_at_every_physics_substep",
            "released_eef_marker_update_contract_valid",
        )
    )
    feedback_valid = all(
        metrics.get(field) is True
        for field in ("first_live_query_identical", "own_observation_chain_valid", "no_recorded_suffix_action_replay")
    )
    post_divergence_feedback = metrics.get("post_divergence_own_observations_used") is True
    baseline_reproduced = bool(metrics.get("baseline_link56_contact_present") is True and metrics.get("baseline_first_selected_obstacle_contact_is_link56") is True)
    contact_prevented = metrics.get("psf_any_robot_selected_obstacle_contact_present") is False
    corrected = bool(
        metrics.get("material_correction_before_baseline_contact") is True
        and float(metrics.get("maximum_correction_norm_rad_s", 0.0)) >= thresholds["minimum_filter_correction_norm_rad_s"]
        and float(metrics.get("filter_correction_integral_rad", 0.0)) >= thresholds["minimum_filter_correction_integral_rad"]
    )
    useful = bool(
        corrected
        and float(metrics.get("post_correction_measured_joint_motion_integral_rad", 0.0)) >= thresholds["minimum_post_correction_measured_joint_motion_integral_rad"]
        and float(metrics.get("post_correction_cartesian_path_length_m", 0.0)) >= thresholds["minimum_post_correction_cartesian_path_length_m"]
        and float(metrics.get("post_correction_executed_command_integral_rad", 0.0)) >= thresholds["minimum_post_correction_executed_command_integral_rad"]
        and float(metrics.get("post_correction_zero_command_fraction", 1.0)) <= thresholds["maximum_post_correction_zero_command_fraction"]
    )
    task_success = bool(
        metrics.get("psf_task_success_ever") is True
        and metrics.get("psf_task_success_after_material_correction") is True
    )
    psf_contact = metrics.get("psf_any_robot_selected_obstacle_contact_present") is True
    positive_exposure = bool(metrics.get("baseline_exposure_complete") is True and metrics.get("psf_exposure_complete") is True)
    pair_complete = bool(apparatus_complete and metrics.get("baseline_exposure_complete") is True and (metrics.get("psf_exposure_complete") is True or psf_contact))
    if not pair_complete or not feedback_valid:
        label = "INCONCLUSIVE_APPARATUS"
    elif not baseline_reproduced:
        label = "BASELINE_CONTACT_NOT_REPRODUCED"
    elif psf_contact:
        label = "CONTACT_REMAINS_OR_SHIFTED"
    elif not post_divergence_feedback:
        label = "INCONCLUSIVE_APPARATUS"
    elif metrics.get("fresh_policy_query_after_material_correction") is not True:
        label = "INCONCLUSIVE_APPARATUS"
    elif not positive_exposure or not contact_prevented:
        label = "INCONCLUSIVE_APPARATUS"
    elif not useful:
        label = "STOP_ONLY"
    elif not task_success:
        label = "CONTACT_PREVENTED_TASK_FAILED"
    else:
        label = "SAFE_TASK_SUCCESS_USEFUL_CORRECTION"
    return {
        "schema_version": CLASSIFICATION_SCHEMA,
        "classification": label,
        "feasible": label == "SAFE_TASK_SUCCESS_USEFUL_CORRECTION",
        "pair_complete": pair_complete,
        "closed_loop_feedback_valid": feedback_valid,
        "post_divergence_feedback_observed": post_divergence_feedback,
        "baseline_reproduced": baseline_reproduced,
        "contact_prevented": contact_prevented,
        "useful_correction": useful,
        "stop_only": bool(contact_prevented and not useful),
        "task_successful": task_success,
    }


def _compare_mapping(audit: _Audit, observed: Any, expected: Mapping[str, Any], label: str) -> None:
    value = audit.mapping(observed, label)
    audit.check(set(value) == set(expected), "%s_key_set_differs" % label)
    for field, expected_value in expected.items():
        observed_value = value.get(field)
        if isinstance(expected_value, float):
            audit.check(_close(audit.number(observed_value, "%s_%s" % (label, field)), expected_value), "%s_%s_differs" % (label, field))
        else:
            audit.check(_canonical(observed_value) == _canonical(expected_value), "%s_%s_differs" % (label, field))


def _validate_checkpoint(audit: _Audit, protocol: Mapping[str, Any], provenance: Mapping[str, Any]) -> None:
    online = audit.mapping(protocol.get("online_policy"), "protocol_online_policy")
    identity = audit.mapping(provenance.get("policy_checkpoint_identity"), "checkpoint_identity")
    for field, expected in (
        ("checkpoint_path", online.get("checkpoint")),
        ("full_content_tree_sha256", online.get("checkpoint_tree_sha256")),
        ("receipt_file_sha256", online.get("checkpoint_receipt_sha256")),
    ):
        audit.check(identity.get(field) == expected, "checkpoint_%s_differs" % field)
    audit.check(identity.get("current_filesystem_identity_matches_receipt") is True, "checkpoint_filesystem_identity_false")
    try:
        from scripts.validate_aegis_assets import (
            pi05_checkpoint_filesystem_identity,
            validate_pi05_filesystem_identity_record,
        )

        receipt_path = Path(str(online.get("checkpoint_receipt")))
        receipt = _load_plain_json(receipt_path, "checkpoint receipt")
        audit.check(sha256_file(receipt_path) == online.get("checkpoint_receipt_sha256"), "checkpoint_receipt_file_hash_differs")
        payload = {key: value for key, value in receipt.items() if key != "receipt_payload_sha256"}
        audit.check(hashlib.sha256(_canonical(payload)).hexdigest() == receipt.get("receipt_payload_sha256"), "checkpoint_receipt_payload_hash_differs")
        checkpoint = audit.mapping(receipt.get("checkpoint"), "checkpoint_receipt_checkpoint")
        audit.check(receipt.get("schema_version") == online.get("checkpoint_receipt_schema_version"), "checkpoint_receipt_schema_differs")
        audit.check(receipt.get("status") == "passed" and checkpoint.get("full_content_hash_verified") is True, "checkpoint_receipt_not_passed")
        audit.check(checkpoint.get("full_content_tree_sha256") == online.get("checkpoint_tree_sha256"), "checkpoint_tree_hash_differs")
        expected_filesystem = validate_pi05_filesystem_identity_record(checkpoint.get("filesystem_identity"))
        current = pi05_checkpoint_filesystem_identity(
            Path(str(online.get("checkpoint"))),
            tuple(str(row["relative_path"]) for row in expected_filesystem["files"]),
        )
        audit.check(current == expected_filesystem, "checkpoint_current_filesystem_differs")
        audit.check(identity.get("filesystem_identity_sha256") == current.get("identity_sha256"), "checkpoint_identity_hash_differs")
        audit.check(identity.get("receipt_payload_sha256") == receipt.get("receipt_payload_sha256"), "checkpoint_provenance_payload_hash_differs")
    except Exception as error:
        audit.check(False, "checkpoint_validation_failed:%s" % error)


def validate_payload(
    payload_value: Mapping[str, Any],
    protocol_value: Mapping[str, Any],
    historical_value: Mapping[str, Any],
    *,
    protocol_file_sha256: str,
    historical_file_sha256: str,
    expected_producer_commit: str,
    expected_producer_job_id: Optional[str] = None,
    expected_producer_host: Optional[str] = None,
    expected_producer_device: str = EXPECTED_GPU,
    result_path: Optional[Path] = None,
) -> Dict[str, Any]:
    expectation = _protocol_expectations(protocol_value)
    audit = _Audit()
    payload = audit.mapping(payload_value, "result")
    audit.check(payload.get("schema_version") == RESULT_SCHEMA, "result_schema_differs")
    audit.check(payload.get("status") == "complete", "result_status_not_complete")
    audit.check(payload.get("case_id") == EXPECTED_CASE, "case_id_differs")
    audit.check(payload.get("protocol_id") == PROTOCOL_ID, "protocol_id_differs")
    audit.check(payload.get("producer_classification_is_preliminary") is True, "producer_classification_not_preliminary")
    audit.check(payload.get("scientific_interpretation_requires_independent_consumer") is True, "independent_consumer_not_required")
    audit.check(payload.get("partial_output_interpreted") is False, "partial_output_interpreted")
    audit.check(payload.get("claim_scope") == EXPECTED_CLAIM_SCOPE, "claim_scope_differs")
    if result_path is not None:
        audit.check(payload.get("run_id") == result_path.parent.name, "run_id_path_differs")

    provenance = audit.mapping(payload.get("provenance"), "provenance")
    source = audit.mapping(provenance.get("source"), "provenance_source")
    allocation = audit.mapping(provenance.get("allocation"), "provenance_allocation")
    audit.check(source.get("commit") == expected_producer_commit, "producer_commit_differs")
    audit.check(source.get("branch") == EXPECTED_BRANCH, "producer_branch_differs")
    audit.check(provenance.get("run_id") == payload.get("run_id"), "provenance_run_id_differs")
    audit.check(provenance.get("case_id") == EXPECTED_CASE, "provenance_case_differs")
    audit.check(provenance.get("case_row_sha256") == EXPECTED_CASE_ROW_SHA256, "case_row_hash_differs")
    audit.check(provenance.get("protocol_file_sha256") == protocol_file_sha256, "protocol_file_hash_differs")
    for field, expected in (
        ("historical_result_file_sha256", EXPECTED_HISTORICAL_FILE_SHA256),
        ("historical_result_payload_sha256", EXPECTED_HISTORICAL_PAYLOAD_SHA256),
        ("historical_executed_action_sequence_sha256", EXPECTED_ACTION_SEQUENCE_SHA256),
        ("policy_checkpoint_tree_sha256", protocol_value["online_policy"]["checkpoint_tree_sha256"]),
        ("policy_checkpoint_receipt_sha256", protocol_value["online_policy"]["checkpoint_receipt_sha256"]),
        ("execution", "live_pi05_feedback_after_exact_shared_prefix"),
    ):
        audit.check(provenance.get(field) == expected, "provenance_%s_differs" % field)
    audit.check(historical_file_sha256 == EXPECTED_HISTORICAL_FILE_SHA256, "historical_file_hash_differs")
    audit.check(historical_value.get("result_payload_sha256") == EXPECTED_HISTORICAL_PAYLOAD_SHA256, "historical_payload_hash_differs")
    audit.check(allocation.get("gpu_name") == expected_producer_device, "producer_device_differs")
    audit.check(str(allocation.get("slurm_job_id", "")).isdigit(), "producer_job_id_invalid")
    audit.check(isinstance(allocation.get("host"), str) and bool(allocation.get("host")), "producer_host_missing")
    if expected_producer_job_id is not None:
        audit.check(str(allocation.get("slurm_job_id")) == str(expected_producer_job_id), "producer_job_id_differs")
    if expected_producer_host is not None:
        audit.check(allocation.get("host") == expected_producer_host, "producer_host_differs")
    _validate_checkpoint(audit, protocol_value, provenance)
    server = audit.mapping(payload.get("policy_server"), "policy_server")
    audit.check(server.get("status") == "available", "policy_server_unavailable")
    audit.check(server.get("metadata") is not None, "policy_server_metadata_missing")

    episode = audit.mapping(payload.get("episode"), "episode")
    audit.check(list(episode.get("shared_prefix_action_indexes", ())) == list(range(180)), "shared_prefix_indexes_differ")
    audit.check(episode.get("shared_prefix_execution") == "exact_historical_aegis_actions_under_unchanged_osc", "shared_prefix_execution_differs")
    audit.check(episode.get("branch_flattened_state_sha256") == EXPECTED_BOUNDARY_STATE_SHA256, "branch_state_hash_differs")
    boundary_raw = _sha256(audit, episode.get("branch_official_integration_state_raw_bytes_sha256"), "branch_official_hash")
    audit.check(boundary_raw == EXPECTED_BOUNDARY_RAW_SHA256, "branch_official_hash_differs")
    audit.check(list(episode.get("live_suffix_action_indexes", ())) == list(range(180, 237)), "live_suffix_indexes_differ")
    audit.check(episode.get("fixed_live_suffix_after_success") is True, "fixed_suffix_after_success_false")
    historical_suffix_hash = hashlib.sha256(
        _canonical(
            [
                [float(value) for value in row["executed"]]
                for row in historical_value["actions"][180:237]
            ]
        )
    ).hexdigest()
    audit.check(episode.get("historical_suffix_reference_sha256_never_executed") == historical_suffix_hash, "historical_suffix_reference_hash_differs")
    audit.check(provenance.get("historical_suffix_reference_sha256_never_executed") == historical_suffix_hash, "provenance_suffix_reference_hash_differs")

    field = audit.mapping(payload.get("field"), "field")
    resolved = audit.mapping(field.get("resolved_geometry"), "resolved_geometry")
    link_ids = [audit.integer(value, "link_geom_id") for value in audit.sequence(resolved.get("link56_geom_ids"), "link_geom_ids")]
    robot_ids = [audit.integer(value, "robot_geom_id") for value in audit.sequence(resolved.get("robot_geom_ids"), "robot_geom_ids")]
    obstacle_ids = [audit.integer(value, "obstacle_geom_id") for value in audit.sequence(resolved.get("obstacle_geom_ids"), "obstacle_geom_ids")]
    audit.check(bool(link_ids) and bool(robot_ids) and bool(obstacle_ids), "resolved_geometry_empty")
    bundle_hashes = audit.mapping(field.get("bundle_hashes"), "field_bundle_hashes")
    for name, expected in (
        ("bundle_sha256", EXPECTED_FIELD_BUNDLE_SHA256),
        ("protocol_sha256", EXPECTED_FIELD_PROTOCOL_SHA256),
        ("parameter_block_sha256", EXPECTED_FIELD_PARAMETER_BLOCK_SHA256),
    ):
        audit.check(bundle_hashes.get(name) == expected, "field_%s_differs" % name)
    sampling = audit.mapping(field.get("full_robot_sampling"), "full_robot_sampling")
    audit.check(hashlib.sha256(_canonical(sampling)).hexdigest() == EXPECTED_FULL_ROBOT_SAMPLING_SHA256, "full_robot_sampling_hash_differs")
    static = audit.mapping(field.get("static_assumption"), "static_assumption")
    shared_prefix_complete = bool(
        episode.get("branch_flattened_state_sha256") == EXPECTED_BOUNDARY_STATE_SHA256
        and list(episode.get("shared_prefix_action_indexes", ())) == list(range(180))
        and static.get("field_construction_boundary") == 0
        and static.get("field_construction_state") == "after_20_settling_actions"
        and static.get("prefix_action_boundary_count") == 180
        and static.get("admissible") is True
    )

    arms = audit.mapping(payload.get("arms"), "arms")
    baseline = audit.mapping(arms.get("baseline"), "baseline")
    psf = audit.mapping(arms.get("psf"), "psf")
    audit.check(baseline.get("arm_name") == "joint_velocity_adapter_only", "baseline_arm_name_differs")
    audit.check(psf.get("arm_name") == "joint_velocity_adapter_plus_link56_psf", "psf_arm_name_differs")
    narrow._validate_pairing(audit, baseline, psf, boundary_raw)
    baseline_contact = full._reconstruct_contacts(audit, baseline, link_ids, robot_ids, obstacle_ids, "baseline", expectation)
    psf_contact = full._reconstruct_contacts(audit, psf, link_ids, robot_ids, obstacle_ids, "psf", expectation)
    baseline_commands, baseline_physics = full._validate_cadence(audit, baseline, "baseline", expectation, contact_present=baseline_contact["any_present"], must_complete=True)
    psf_commands_raw, psf_physics = full._validate_cadence(audit, psf, "psf", expectation, contact_present=psf_contact["any_present"], must_complete=not psf_contact["any_present"])
    narrow._validate_adapter_execution(audit, baseline, baseline_commands)
    psf_commands = narrow._validate_qp(audit, baseline, psf, psf_commands_raw)
    baseline_task = full._validate_task(audit, baseline, "baseline", expectation, allow_contact_terminal=False)
    psf_task = full._validate_task(audit, psf, "psf", expectation, allow_contact_terminal=psf_contact["any_present"])
    audit.check(_canonical(baseline_task["definition"]) == _canonical(psf_task["definition"]), "paired_goal_definition_differs")

    expected_psf_actions = psf_task["completed_action_count"] + (1 if psf_contact["any_present"] else 0)
    providers = audit.mapping(payload.get("providers"), "providers")
    baseline_provider = _validate_provider(audit, providers.get("baseline"), baseline, "baseline", expectation, historical_value, expected_actions=57)
    psf_provider = _validate_provider(audit, providers.get("psf"), psf, "psf", expectation, historical_value, expected_actions=expected_psf_actions)
    paired_first_live_action = _validate_first_live_pair(
        audit, baseline_provider, psf_provider
    )
    feedback = _feedback(audit, baseline_provider, psf_provider)
    _compare_mapping(audit, payload.get("closed_loop_feedback"), feedback, "closed_loop_feedback")

    _baseline_periodic_positive = _validate_periodic_clearance(audit, baseline, baseline_physics, "baseline", expectation)
    psf_periodic_positive = _validate_periodic_clearance(audit, psf, psf_physics, "psf", expectation)
    # This helper independently checks the static sampling certificate formula,
    # signed-contact clamp, and diagnostic minimum.  Its boolean is periodic
    # diagnostic positivity here, never a continuous safety certificate.
    psf_clearance_formula_positive = full._validate_field_and_clearance(
        audit,
        field,
        robot_ids,
        psf,
        psf_physics,
        contact_present=psf_contact["any_present"],
        minimum_contact_distance_m=psf_contact["minimum_contact_distance_m"],
    )
    # Periodic full-surface clearance is diagnostic for both arms.  In
    # particular, a nonpositive baseline sample cannot manufacture contact:
    # selected-obstacle MuJoCo contacts observed at every 2 ms substep are the
    # sole collision authority under this protocol.

    material_cutoff = baseline_contact["first_link_boundary"] if baseline_contact["first_link_boundary"] is not None else expectation["no_event_boundary"]
    material = narrow._material_activation(audit, psf_commands, psf, int(material_cutoff))
    first_correction = material[0]["boundary"] if material else None
    post_motion = _post_correction_motion(
        audit, psf_commands_raw, psf_physics, first_correction, expectation
    )
    _compare_mapping(audit, payload.get("post_correction_motion"), post_motion, "post_correction_motion")
    fresh_queries = [
        int(row["query_index"])
        for row in psf_provider["queries"]
        if first_correction is not None and int(row["source_action_index"]) * 25 > int(first_correction)
    ]
    success_rows = [
        row
        for row in psf["task"]["goal_progress_ledger"]
        if first_correction is not None
        and row.get("snapshot_kind") == "completed_high_level_post_step"
        and row.get("all_satisfied") is True
        and (int(row["source_action_index"]) + 1) * 25 > int(first_correction)
    ]
    baseline_video_ok = _validate_video(audit, audit.mapping(payload.get("videos"), "videos").get("baseline"), result_path.parent if result_path else Path("."), "baseline", completed_actions=baseline_task["completed_action_count"], contact_terminal=False)
    psf_video_ok = _validate_video(audit, payload["videos"].get("psf"), result_path.parent if result_path else Path("."), "psf", completed_actions=psf_task["completed_action_count"], contact_terminal=psf_contact["any_present"])

    marker_contract = bool(
        all(
            arm.get("released_eef_marker_update_count") == task["completed_action_count"]
            and arm.get("released_eef_marker_update_semantics") == "one_model_update_after_each_completed_high_level_env_step"
            for arm, task in ((baseline, baseline_task), (psf, psf_task))
        )
    )
    literal_every_substep = bool(
        baseline.get("monitor_observed_physics_substep_count") == len(baseline_physics)
        and psf.get("monitor_observed_physics_substep_count") == len(psf_physics)
        and all("literal_contact_observed" in row for row in list(baseline_physics) + list(psf_physics))
    )
    exact_pair = payload.get("metrics", {}).get("exact_paired_start") is True
    # ``narrow._validate_pairing`` supplied the fail-closed discrepancies.  The
    # boolean below is independently reconstructed for the metric comparison.
    pair_fields = (
        "model_topology_sha256", "physical_model_sha256", "compiled_mjb_sha256",
        "official_integration_state_sha256", "target_official_integration_state_sha256",
        "settled_state_sha256", "target_state_sha256", "controller",
        "controller_software_state", "pid_memory_reset",
    )
    exact_pair = bool(
        all(_canonical(baseline["restore"].get(field)) == _canonical(psf["restore"].get(field)) for field in pair_fields)
        and baseline.get("start_official_raw_bytes_sha256") == psf.get("start_official_raw_bytes_sha256") == boundary_raw
        and baseline.get("fresh_adapter") is True
        and psf.get("fresh_adapter") is True
    )
    baseline_first = baseline_contact["first_link_boundary"]
    correction_before = bool(first_correction is not None and baseline_first is not None and int(first_correction) < int(baseline_first))
    psf_qp_contract = bool(
        psf.get("precontact_execution_valid") is True
        and psf.get("qp_solve_count") == len(psf_commands_raw)
        and psf.get("qp_postcheck_count") == len(psf_commands_raw)
        and psf.get("joint_limit_postcheck_count") == len(psf_commands_raw)
    )
    baseline_live_provider_valid = _live_provider_contract(
        baseline_provider,
        expected_actions=expectation["action_count"],
        policy_noise_seed=2026691220,
    )
    psf_live_provider_valid = _live_provider_contract(
        psf_provider,
        expected_actions=expected_psf_actions,
        policy_noise_seed=2026691220,
    )
    live_policy_contract_valid = bool(
        baseline_live_provider_valid
        and psf_live_provider_valid
        and feedback["paired_first_query_cache"]["contract_valid"]
        and feedback["only_psf_query_36_reused_paired_cache"]
    )
    aegis_contract_valid = bool(
        paired_first_live_action
        and feedback[
            "first_current_action_180_aegis_pair_contract_valid"
        ]
        and all(
            record["actions"]
            and record["provider"].get(
                "first_aegis_input_binding_matches_historical"
            )
            is True
            and all(
                row.get("aegis_qp", {}).get("status") == "solved"
                and row.get("aegis_qp", {}).get("solver_status")
                in ("optimal", "optimal_inaccurate")
                for row in record["actions"]
            )
            for record in (baseline_provider, psf_provider)
        )
        and _aegis_state_and_command_chain_valid(
            baseline_provider, baseline
        )
        and _aegis_state_and_command_chain_valid(psf_provider, psf)
    )
    reconstructed_metrics = {
        "exact_paired_start": exact_pair,
        "shared_prefix_complete": shared_prefix_complete,
        "baseline_exposure_complete": baseline.get("exposure_complete") is True,
        "psf_exposure_complete": psf.get("exposure_complete") is True,
        "live_policy_contract_valid": live_policy_contract_valid,
        "aegis_contract_valid": aegis_contract_valid,
        "videos_complete_and_decodable": bool(baseline_video_ok and psf_video_ok),
        "psf_qp_contract_valid": psf_qp_contract,
        "static_selected_obstacle_admissible": bool(static.get("admissible") is True and baseline.get("static_precontact_admissible") is True and psf.get("static_precontact_admissible") is True),
        "literal_contact_checked_at_every_physics_substep": literal_every_substep,
        "released_eef_marker_update_contract_valid": marker_contract,
        "first_live_query_identical": feedback["first_live_query_identical"],
        "only_psf_query_36_reused_paired_cache": feedback[
            "only_psf_query_36_reused_paired_cache"
        ],
        "historical_first_live_query_action_chunk_matches_diagnostic": (
            feedback[
                "historical_first_live_query_action_chunk_matches_diagnostic"
            ]
        ),
        "first_current_action_180_aegis_inputs_identical": feedback[
            "first_current_action_180_aegis_inputs_identical"
        ],
        "first_current_action_180_aegis_outputs_identical": feedback[
            "first_current_action_180_aegis_outputs_identical"
        ],
        "post_divergence_own_observations_used": feedback["post_divergence_own_observations_used"],
        "post_divergence_policy_inputs_differ": feedback["post_divergence_policy_inputs_differ"],
        "fresh_policy_query_after_material_correction": bool(fresh_queries),
        "fresh_policy_query_indexes_after_material_correction": fresh_queries,
        "own_observation_chain_valid": bool(
            _own_observation_chain_valid(baseline_provider, baseline)
            and _own_observation_chain_valid(psf_provider, psf)
        ),
        "no_recorded_suffix_action_replay": bool(baseline_provider["provider"].get("recorded_suffix_actions_executed") is False and psf_provider["provider"].get("recorded_suffix_actions_executed") is False),
        "first_live_aegis_action_matches_historical": bool(baseline_provider["actions"] and psf_provider["actions"] and _same_numeric(baseline_provider["actions"][0].get("aegis_executed"), historical_value["actions"][180]["executed"]) and _same_numeric(psf_provider["actions"][0].get("aegis_executed"), historical_value["actions"][180]["executed"])),
        "historical_action_180_full_output_matches_diagnostic": bool(
            all(
                record["provider"].get(
                    "first_aegis_full_output_matches_historical_diagnostic"
                )
                is True
                for record in (baseline_provider, psf_provider)
            )
        ),
        "both_nominal_commands_within_dynamic_joint_bounds": bool(baseline.get("all_nominal_commands_within_dynamic_joint_bounds") is True and psf.get("all_nominal_commands_within_dynamic_joint_bounds") is True),
        "baseline_link56_contact_present": baseline_contact["link_present"],
        "baseline_first_selected_obstacle_contact_is_link56": bool(baseline_contact["first_link_boundary"] is not None and baseline_contact["first_link_boundary"] == baseline_contact["first_any_boundary"]),
        "psf_link56_contact_present": psf_contact["link_present"],
        "psf_any_robot_selected_obstacle_contact_present": psf_contact["any_present"],
        "psf_clearance_certified": False,
        "psf_periodic_clearance_diagnostic_positive": bool(psf_periodic_positive and psf_clearance_formula_positive),
        "material_correction_before_baseline_contact": correction_before,
        "first_material_correction_physical_boundary": first_correction,
        "baseline_first_link56_contact_physical_boundary": baseline_first,
        "material_correction_update_count": len(material),
        "maximum_correction_norm_rad_s": post_motion["maximum_correction_norm_rad_s"],
        "filter_correction_integral_rad": post_motion["filter_correction_integral_rad"],
        "post_correction_measured_joint_motion_integral_rad": post_motion["measured_joint_motion_integral_rad"],
        "post_correction_cartesian_path_length_m": post_motion["cartesian_path_length_m"],
        "post_correction_executed_command_integral_rad": post_motion["executed_command_integral_rad"],
        "post_correction_zero_command_fraction": post_motion["zero_command_fraction"],
        "baseline_task_success_ever": baseline_task["ever_success"],
        "baseline_terminal_task_success": baseline_task["terminal_success"],
        "psf_task_success_ever": psf_task["ever_success"],
        "psf_terminal_task_success": psf_task["terminal_success"],
        "psf_task_success_after_material_correction": bool(success_rows),
        "psf_successful_source_action_indexes_after_material_correction": [int(row["source_action_index"]) for row in success_rows],
    }
    metrics = audit.mapping(payload.get("metrics"), "metrics")
    audit.check(set(metrics) == set(METRIC_KEYS), "metric_key_set_differs")
    for field, expected in reconstructed_metrics.items():
        observed = metrics.get(field)
        if isinstance(expected, float):
            audit.check(_close(audit.number(observed, "metric_%s" % field), expected), "metric_%s_differs" % field)
        else:
            audit.check(_canonical(observed) == _canonical(expected), "metric_%s_differs" % field)

    classification = _independent_classification(reconstructed_metrics, expectation)
    producer_classification = audit.mapping(payload.get("classification"), "producer_classification")
    audit.check(_canonical(producer_classification) == _canonical(classification), "producer_classification_differs")
    valid = not audit.discrepancies
    return {
        "schema_version": SUMMARY_SCHEMA,
        "status": "validated" if valid else "rejected",
        "artifact_valid": valid,
        "run_id": payload.get("run_id"),
        "case_id": payload.get("case_id"),
        "classification": classification["classification"] if valid else None,
        "feasible": classification["feasible"] if valid else None,
        "pair_complete": classification["pair_complete"] if valid else None,
        "baseline_reproduced": classification["baseline_reproduced"] if valid else None,
        "contact_prevented": classification["contact_prevented"] if valid else None,
        "useful_correction": classification["useful_correction"] if valid else None,
        "stop_only": classification["stop_only"] if valid else None,
        "task_successful": classification["task_successful"] if valid else None,
        "baseline_first_link56_contact_physical_boundary": baseline_first,
        "psf_any_robot_selected_obstacle_contact_present": psf_contact["any_present"],
        "first_material_correction_physical_boundary": first_correction,
        "fresh_policy_query_indexes_after_material_correction": fresh_queries,
        "post_correction_motion": post_motion,
        "baseline_task_success_ever": baseline_task["ever_success"],
        "psf_task_success_ever": psf_task["ever_success"],
        "producer_source_commit": source.get("commit"),
        "producer_slurm_job_id": allocation.get("slurm_job_id"),
        "producer_host": allocation.get("host"),
        "producer_device": allocation.get("gpu_name"),
        "result_payload_sha256": payload.get("result_payload_sha256"),
        "protocol_file_sha256": protocol_file_sha256,
        "historical_result_file_sha256": historical_file_sha256,
        "partial_output_interpreted": False,
        "discrepancies": audit.discrepancies,
    }


def validate_artifact(
    result_path: Path,
    protocol_path: Path,
    historical_result_root: Path,
    *,
    expected_producer_commit: str,
    expected_producer_job_id: Optional[str] = None,
    expected_producer_host: Optional[str] = None,
    expected_producer_device: str = EXPECTED_GPU,
) -> Dict[str, Any]:
    result_path = Path(result_path)
    if result_path.name != "result.json":
        raise ArtifactContractError("only terminal result.json is accepted")
    try:
        mode = result_path.lstat().st_mode
    except OSError as error:
        raise ArtifactContractError("terminal result is unavailable") from error
    if result_path.is_symlink() or not stat.S_ISREG(mode):
        raise ArtifactContractError("terminal result must be a nonsymlink regular file")
    protocol_path = Path(protocol_path)
    historical_path = _historical_path(historical_result_root)
    historical_replay = load_historical_action_replay(
        historical_path,
        expected_case_id=EXPECTED_CASE,
        expected_arm=EXPECTED_SOURCE_ARM,
    )
    if historical_replay.result_payload_sha256 != EXPECTED_HISTORICAL_PAYLOAD_SHA256 or historical_replay.executed_sequence_sha256 != EXPECTED_ACTION_SEQUENCE_SHA256:
        raise ArtifactContractError("historical replay authority differs")
    payload = load_hashed_json(result_path)
    protocol = _load_plain_json(protocol_path, "protocol")
    historical = _load_plain_json(historical_path, "historical result")
    summary = validate_payload(
        payload,
        protocol,
        historical,
        protocol_file_sha256=sha256_file(protocol_path),
        historical_file_sha256=sha256_file(historical_path),
        expected_producer_commit=expected_producer_commit,
        expected_producer_job_id=expected_producer_job_id,
        expected_producer_host=expected_producer_host,
        expected_producer_device=expected_producer_device,
        result_path=result_path,
    )
    summary["result_file_sha256"] = sha256_file(result_path)
    return summary


def _commit(value: str, label: str) -> str:
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise ArtifactContractError("%s must be one lowercase 40-character SHA" % label)
    return value


def _validate_consumer_repo(repo_root: Path, consumer_commit: str) -> None:
    root = Path(repo_root)
    if root.is_symlink() or not root.is_dir():
        raise ArtifactContractError("consumer repo root is missing or symlinked")

    def git(*arguments: str) -> str:
        try:
            return subprocess.check_output(
                ["git", "-C", str(root)] + list(arguments),
                stderr=subprocess.STDOUT,
                text=True,
            ).strip()
        except subprocess.CalledProcessError as error:
            raise ArtifactContractError("consumer git identity is unavailable") from error

    if git("status", "--short"):
        raise ArtifactContractError("consumer worktree is dirty")
    if git("rev-parse", "HEAD") != consumer_commit:
        raise ArtifactContractError("consumer checkout commit differs")
    if git("branch", "--show-current") != EXPECTED_BRANCH:
        raise ArtifactContractError("consumer checkout branch differs")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--historical-result-root", required=True, type=Path)
    parser.add_argument("--expected-producer-commit", required=True)
    parser.add_argument("--consumer-commit", required=True)
    parser.add_argument("--expected-producer-job-id")
    parser.add_argument("--expected-producer-host")
    parser.add_argument("--expected-producer-device", default=EXPECTED_GPU)
    arguments = parser.parse_args()
    try:
        producer_commit = _commit(arguments.expected_producer_commit, "producer commit")
        consumer_commit = _commit(arguments.consumer_commit, "consumer commit")
        _validate_consumer_repo(arguments.repo_root, consumer_commit)
        summary = validate_artifact(
            arguments.result,
            arguments.protocol,
            arguments.historical_result_root,
            expected_producer_commit=producer_commit,
            expected_producer_job_id=arguments.expected_producer_job_id,
            expected_producer_host=arguments.expected_producer_host,
            expected_producer_device=arguments.expected_producer_device,
        )
        summary = dict(summary)
        summary["consumer_source_commit"] = consumer_commit
    except Exception as error:
        summary = {
            "schema_version": SUMMARY_SCHEMA,
            "status": "rejected",
            "artifact_valid": False,
            "partial_output_interpreted": False,
            "consumer_source_commit": arguments.consumer_commit,
            "discrepancies": ["artifact_load_failed:%s" % error],
        }
    print(json.dumps(summary, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0 if summary.get("artifact_valid") is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
