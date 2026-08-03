#!/usr/bin/env python3
"""Independent consumer for the direct joint-velocity Poisson-CBF experiment.

The producer is deliberately not imported.  This consumer reconstructs the
claim-bearing pairing, controller, CBF-QP, contact, tracking, motion, and task
facts from the serialized arm traces.  Producer metrics and classification
are checked only after the independent reconstruction.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import sys
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main.poisson_fullbody.contracts import (  # noqa: E402
    ArtifactContractError,
    load_hashed_json,
    sha256_file,
)


RESULT_SCHEMA = "vlsa_poisson_direct_joint_velocity_result.v1"
PROTOCOL_ID = "vlsa-poisson-direct-joint-velocity-action0-link56-v1"
CASE_HORIZONS = {
    "vlsa-t1-goal-ii-t0-e05": 237,
    "vlsa-t1-goal-ii-t3-e42": 120,
}
SUMMARY_SCHEMA = (
    "vlsa_poisson_direct_joint_velocity_feasibility_independent_validation.v1"
)
PRODUCER_MODULE = "scripts.run_poisson_direct_joint_velocity_feasibility"
PROTOCOL_MODULE = "main.poisson_fullbody.direct_joint_velocity_feasibility"
ARM_RUNNER_MODULE = "scripts.run_poisson_fast_feasibility"
CONTACT_MONITOR_MODULE = "main.poisson_fullbody.registered_contact_monitor"
CBF_MODULE = "main.poisson_fullbody.cbf_qp"
ADAPTER_MODULE = "main.poisson_fullbody.joint_velocity_adapter"

PROTECTED_BODY_NAMES = ("robot0_link5", "robot0_link6")
CONTROLLER_NAME = "JOINT_VELOCITY"
UPDATES_PER_ACTION = 5
SUBSTEPS_PER_UPDATE = 5
CONTROL_DT_S = 0.01
PHYSICS_DT_S = 0.002

CBF_TOLERANCE = 5.0e-7
BOUND_TOLERANCE = 5.0e-8
FLOAT_TOLERANCE = 1.0e-10
MATERIAL_CORRECTION_MIN_RAD_S = 1.0e-4
MATERIAL_CORRECTION_INTEGRAL_MIN_RAD = 1.0e-6
POST_CORRECTION_JOINT_MOTION_MIN_RAD = 1.0e-4
POST_CORRECTION_EEF_PATH_MIN_M = 1.0e-3
POST_CORRECTION_EXECUTED_INTEGRAL_MIN_RAD = 1.0e-4
POST_CORRECTION_ZERO_COMMAND_FRACTION_MAX = 0.99
TRACKING_LINF_MAX_RAD_S = 0.05
TRACKING_RMSE_MAX_RAD_S = 0.02

APPARATUS_METRIC_KEYS = (
    "exact_settled_pair_start",
    "baseline_exposure_complete",
    "treatment_exposure_complete",
    "first_live_policy_query_paired",
    "live_policy_contract_valid",
    "released_aegis_contract_valid",
    "own_observation_chains_valid",
    "direct_joint_velocity_controller_used",
    "fixed_link56_protected_set_used",
    "hard_psf_rows_enforced",
    "shared_nominal_joint_bounds_inactive",
    "joint_velocity_tracking_valid",
    "no_unregistered_fallback_executed",
    "every_physics_substep_contact_checked",
    "paper_car_endpoint_ledger_complete",
)
OUTCOME_BOOLEAN_METRIC_KEYS = (
    "baseline_link56_contact_present",
    "treatment_any_robot_selected_obstacle_contact_present",
    "treatment_link56_shifted_external_contact_present",
    "treatment_method_stop",
    "treatment_stalled_after_correction",
    "treatment_native_task_success",
    "treatment_task_success_after_correction",
    "treatment_paper_car_avoided",
    "material_correction_present",
    "material_correction_before_baseline_contact",
)
OUTCOME_NUMERIC_METRIC_KEYS = (
    "maximum_correction_norm_rad_s",
    "correction_integral_rad",
    "post_correction_measured_joint_motion_integral_rad",
    "post_correction_eef_path_length_m",
    "post_correction_executed_command_integral_rad",
    "post_correction_zero_command_fraction",
)
METRIC_KEYS = (
    APPARATUS_METRIC_KEYS
    + OUTCOME_BOOLEAN_METRIC_KEYS
    + OUTCOME_NUMERIC_METRIC_KEYS
)

# Generic low-level joint-velocity controllers legitimately expose actuator
# torque state.  Reject only the post-OSC torque-shield semantics that changed
# the method, not ordinary controller-internal torque bookkeeping.
FORBIDDEN_SEMANTIC_TOKENS = (
    "postosctorque",
    "torquesensitivity",
    "torqueshield",
    "deltatau",
    "deltatorque",
    "torquetonextqvel",
    "sampleddatapostosc",
)
FORBIDDEN_IMPORT_MODULES = (
    "main.poisson_fullbody.post_osc_torque_sensitivity",
    "main.poisson_fullbody.post_osc_torque_shield",
    "scripts.run_poisson_osc_arm_link_canary",
)


class _Audit:
    def __init__(self) -> None:
        self.discrepancies: List[str] = []

    def check(self, condition: bool, message: str) -> None:
        if not condition and message not in self.discrepancies:
            self.discrepancies.append(message)

    def mapping(self, value: Any, label: str) -> Mapping[str, Any]:
        if not isinstance(value, Mapping):
            self.check(False, "%s_not_object" % label)
            return {}
        return value

    def sequence(self, value: Any, label: str) -> Sequence[Any]:
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            self.check(False, "%s_not_array" % label)
            return ()
        return value

    def integer(self, value: Any, label: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            self.check(False, "%s_not_integer" % label)
            return 0
        return int(value)

    def number(self, value: Any, label: str) -> float:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            self.check(False, "%s_not_finite" % label)
            return 0.0
        return float(value)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and bool(re.fullmatch(r"[0-9a-f]{64}", value))


def _is_git_commit(value: Any) -> bool:
    return isinstance(value, str) and bool(re.fullmatch(r"[0-9a-f]{40}", value))


def _close(left: float, right: float, tolerance: float = FLOAT_TOLERANCE) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)


def _vector(
    audit: _Audit, value: Any, length: int, label: str
) -> Tuple[float, ...]:
    rows = audit.sequence(value, label)
    audit.check(len(rows) == length, "%s_length_differs" % label)
    if len(rows) != length:
        return tuple(0.0 for _ in range(length))
    return tuple(audit.number(item, "%s_%d" % (label, index)) for index, item in enumerate(rows))


def _matrix(
    audit: _Audit,
    value: Any,
    row_count: int,
    column_count: int,
    label: str,
) -> Tuple[Tuple[float, ...], ...]:
    rows = audit.sequence(value, label)
    audit.check(len(rows) == row_count, "%s_row_count_differs" % label)
    if len(rows) != row_count:
        return tuple(
            tuple(0.0 for _ in range(column_count))
            for _ in range(row_count)
        )
    return tuple(
        _vector(
            audit,
            row,
            column_count,
            "%s_%d" % (label, row_index),
        )
        for row_index, row in enumerate(rows)
    )


def _nested_close(left: Any, right: Any, tolerance: float = 1.0e-12) -> bool:
    if isinstance(left, tuple) and isinstance(right, tuple):
        return len(left) == len(right) and all(
            _nested_close(a, b, tolerance) for a, b in zip(left, right)
        )
    return _close(float(left), float(right), tolerance)


def _norm(value: Sequence[float]) -> float:
    return math.sqrt(sum(float(item) * float(item) for item in value))


def _distance(left: Sequence[float], right: Sequence[float]) -> float:
    return _norm(tuple(float(a) - float(b) for a, b in zip(left, right)))


def _float64_matrix_sha256(audit: _Audit, value: Any, label: str) -> str:
    rows_value = audit.sequence(value, label)
    rows = [audit.sequence(row, "%s_%d" % (label, index)) for index, row in enumerate(rows_value)]
    audit.check(bool(rows), "%s_empty" % label)
    width = len(rows[0]) if rows else 0
    audit.check(
        width > 0 and all(len(row) == width for row in rows),
        "%s_not_rectangular" % label,
    )
    numbers = [
        audit.number(item, "%s_%d_%d" % (label, row_index, column_index))
        for row_index, row in enumerate(rows)
        for column_index, item in enumerate(row)
    ]
    byte_order = "<" if sys.byteorder == "little" else ">"
    raw = struct.pack(
        "%s%d" % (byte_order, len(numbers)) + "d", *numbers
    ) if numbers else b""
    digest = hashlib.sha256()
    digest.update(b"vlsa-table1-array-v1\0")
    digest.update(
        _canonical({"dtype": "%sf8" % byte_order, "shape": [len(rows), width]})
    )
    digest.update(b"\0")
    digest.update(raw)
    return digest.hexdigest()


def _normalized_semantic_token(value: str) -> str:
    return "".join(character for character in value.lower() if character.isalnum())


def _forbidden_semantic(value: str) -> Optional[str]:
    normalized = _normalized_semantic_token(value)
    return next(
        (token for token in FORBIDDEN_SEMANTIC_TOKENS if token in normalized),
        None,
    )


def _scan_forbidden_artifact_semantics(
    value: Any, *, path: str = "result"
) -> List[str]:
    findings: List[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            token = _forbidden_semantic(key_text)
            if token is not None:
                findings.append("%s.%s:key:%s" % (path, key_text, token))
            findings.extend(
                _scan_forbidden_artifact_semantics(
                    item, path="%s.%s" % (path, key_text)
                )
            )
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, item in enumerate(value):
            findings.extend(
                _scan_forbidden_artifact_semantics(
                    item, path="%s[%d]" % (path, index)
                )
            )
    elif isinstance(value, str):
        token = _forbidden_semantic(value)
        if token is not None:
            findings.append("%s:value:%s" % (path, token))
    return findings


def _audit_source_text(source: str) -> Dict[str, Any]:
    """Audit executable import/name semantics without importing the producer."""

    tree = ast.parse(source)
    imports: List[str] = []
    forbidden: List[str] = []
    names: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            imports.append(module)
            imports.extend(
                "%s.%s" % (module, alias.name) if module else alias.name
                for alias in node.names
            )
        elif isinstance(node, ast.Name):
            names.append(node.id)
        elif isinstance(node, ast.Attribute):
            names.append(node.attr)
    for value in imports + names:
        token = _forbidden_semantic(value)
        if token is not None:
            forbidden.append("%s:%s" % (value, token))
    for imported in imports:
        if any(
            imported == forbidden_module
            or imported.startswith(forbidden_module + ".")
            for forbidden_module in FORBIDDEN_IMPORT_MODULES
        ):
            forbidden.append("%s:forbidden_module" % imported)
    direct_modules = {
        name
        for name in imports
        if name in (CBF_MODULE, ADAPTER_MODULE, ARM_RUNNER_MODULE)
        or name.startswith(CBF_MODULE + ".")
        or name.startswith(ADAPTER_MODULE + ".")
    }
    return {
        "imports": sorted(set(imports)),
        "forbidden_semantics": sorted(set(forbidden)),
        "has_direct_joint_velocity_dependencies": bool(direct_modules),
        "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
    }


def _validate_source(
    audit: _Audit,
    source_path: Path,
    runner_source_path: Path,
    arm_runner_source_path: Path,
    contact_monitor_source_path: Path,
    cbf_source_path: Path,
    adapter_source_path: Path,
    providers: Mapping[str, Any],
) -> Dict[str, Any]:
    records: Dict[str, Any] = {}
    source_hashes = audit.mapping(
        providers.get("source_sha256"), "provider_source_sha256"
    )
    specifications = (
        ("protocol", source_path, "protocol", False),
        ("producer", runner_source_path, "producer", True),
        ("arm_runner", arm_runner_source_path, "arm_runner", True),
        ("contact_monitor", contact_monitor_source_path, "contact_monitor", False),
        ("cbf_qp", cbf_source_path, "cbf_qp", False),
        ("joint_velocity_adapter", adapter_source_path, "joint_velocity_adapter", False),
    )
    for label, path, hash_key, require_direct in specifications:
        try:
            source = path.read_text(encoding="utf-8")
            record = _audit_source_text(source)
        except (OSError, UnicodeDecodeError, SyntaxError) as error:
            audit.check(
                False,
                "%s_source_unavailable_or_invalid:%s" % (label, error),
            )
            continue
        records[label] = record
        audit.check(
            not record["forbidden_semantics"],
            "%s_source_has_post_osc_torque_semantics" % label,
        )
        if require_direct:
            audit.check(
                record["has_direct_joint_velocity_dependencies"],
                "%s_source_lacks_direct_joint_velocity_dependency" % label,
            )
        audit.check(
            source_hashes.get(hash_key) == record["source_sha256"],
            "provider_%s_differs" % hash_key,
        )
    return records


def _validate_providers(audit: _Audit, value: Any) -> Mapping[str, Any]:
    providers = audit.mapping(value, "providers")
    audit.check(
        set(providers)
        == {"implementation", "adapter_only", "adapter_plus_psf"},
        "provider_names_differ",
    )
    implementation = audit.mapping(
        providers.get("implementation"), "provider_implementation"
    )
    expected = {
        "producer_module": PRODUCER_MODULE,
        "protocol_module": PROTOCOL_MODULE,
        "cbf_qp_module": CBF_MODULE,
        "joint_velocity_adapter_module": ADAPTER_MODULE,
        "contact_monitor_module": CONTACT_MONITOR_MODULE,
        "arm_runner_module": ARM_RUNNER_MODULE,
        "controller": CONTROLLER_NAME,
        "decision_variable": "joint_velocity_rad_s",
    }
    for key, expected_value in expected.items():
        audit.check(
            implementation.get(key) == expected_value,
            "provider_%s_differs" % key,
        )
    source_hashes = audit.mapping(
        implementation.get("source_sha256"), "provider_source_sha256"
    )
    for key in (
        "producer",
        "protocol",
        "arm_runner",
        "cbf_qp",
        "joint_velocity_adapter",
        "contact_monitor",
    ):
        audit.check(
            _is_sha256(source_hashes.get(key)),
            "provider_source_sha256_%s_invalid" % key,
        )
    declared_imports = audit.sequence(
        implementation.get("runtime_imports"), "provider_runtime_imports"
    )
    audit.check(
        CBF_MODULE in declared_imports,
        "provider_cbf_import_absent",
    )
    audit.check(
        ADAPTER_MODULE in declared_imports,
        "provider_adapter_import_absent",
    )
    audit.check(
        CONTACT_MONITOR_MODULE in declared_imports,
        "provider_contact_monitor_import_absent",
    )
    audit.check(
        set(declared_imports) == {CBF_MODULE, ADAPTER_MODULE, CONTACT_MONITOR_MODULE},
        "provider_runtime_imports_differ",
    )
    audit.check(
        not _scan_forbidden_artifact_semantics(declared_imports, path="runtime_imports"),
        "provider_forbidden_import_present",
    )
    proxy_bindings = []
    for arm_key in ("adapter_only", "adapter_plus_psf"):
        provider = audit.mapping(
            providers.get(arm_key), "provider_%s" % arm_key
        )
        audit.check(
            provider.get("source_start_action") == 0,
            "provider_%s_source_start_differs" % arm_key,
        )
        audit.check(
            provider.get("recorded_suffix_actions_executed") is False,
            "provider_%s_used_recorded_actions" % arm_key,
        )
        audit.check(
            provider.get("first_aegis_proxy_source")
            == "released_pre_settle_proxy_from_historical_action0_context",
            "provider_%s_released_action0_proxy_source_differs" % arm_key,
        )
        proxy_diagnostic = audit.mapping(
            provider.get("first_fresh_proxy_diagnostic"),
            "provider_%s_first_fresh_proxy_diagnostic" % arm_key,
        )
        fresh_p1 = _vector(
            audit,
            proxy_diagnostic.get("fresh_settled_p1"),
            3,
            "provider_%s_fresh_settled_p1" % arm_key,
        )
        released_p1 = _vector(
            audit,
            proxy_diagnostic.get("released_pre_settle_p1"),
            3,
            "provider_%s_released_pre_settle_p1" % arm_key,
        )
        fresh_R1 = _matrix(
            audit,
            proxy_diagnostic.get("fresh_settled_R1"),
            3,
            3,
            "provider_%s_fresh_settled_R1" % arm_key,
        )
        released_R1 = _matrix(
            audit,
            proxy_diagnostic.get("released_pre_settle_R1"),
            3,
            3,
            "provider_%s_released_pre_settle_R1" % arm_key,
        )
        reported_difference = audit.number(
            proxy_diagnostic.get("p1_l2_difference_m"),
            "provider_%s_p1_l2_difference_m" % arm_key,
        )
        reconstructed_difference = _distance(fresh_p1, released_p1)
        audit.check(
            reported_difference >= 0.0
            and _close(reported_difference, reconstructed_difference, 1.0e-12),
            "provider_%s_p1_difference_reconstruction_failed" % arm_key,
        )
        historical_inputs = audit.mapping(
            provider.get("historical_first_action_required_aegis_inputs"),
            "provider_%s_historical_first_aegis_inputs" % arm_key,
        )
        historical_p1 = _vector(
            audit,
            historical_inputs.get("p1"),
            3,
            "provider_%s_historical_first_p1" % arm_key,
        )
        historical_R1 = _matrix(
            audit,
            historical_inputs.get("R1"),
            3,
            3,
            "provider_%s_historical_first_R1" % arm_key,
        )
        audit.check(
            _nested_close(released_p1, historical_p1)
            and _nested_close(released_R1, historical_R1),
            "provider_%s_released_proxy_not_historical_authority" % arm_key,
        )
        audit.check(
            isinstance(provider.get("policy_queries"), Sequence)
            and not isinstance(provider.get("policy_queries"), (str, bytes)),
            "provider_%s_policy_queries_invalid" % arm_key,
        )
        audit.check(
            isinstance(provider.get("high_level_action_trace"), Sequence)
            and not isinstance(
                provider.get("high_level_action_trace"), (str, bytes)
            ),
            "provider_%s_action_trace_invalid" % arm_key,
        )
        actions = audit.sequence(
            provider.get("high_level_action_trace"),
            "provider_%s_proxy_action_trace" % arm_key,
        )
        proxy_source_trace_valid = bool(actions)
        for action_index, action_value in enumerate(actions):
            action = audit.mapping(
                action_value,
                "provider_%s_proxy_action_%d" % (arm_key, action_index),
            )
            expected_source = (
                "released_pre_settle_proxy_from_historical_action0_context"
                if action_index == 0
                else "current_observation_proxy"
            )
            proxy_source_trace_valid = bool(
                proxy_source_trace_valid
                and action.get("aegis_proxy_source") == expected_source
            )
        audit.check(
            proxy_source_trace_valid,
            "provider_%s_proxy_source_trace_differs" % arm_key,
        )
        first_action = (
            audit.mapping(actions[0], "provider_%s_first_proxy_action" % arm_key)
            if actions
            else {}
        )
        first_qp = audit.mapping(
            first_action.get("aegis_qp"),
            "provider_%s_first_proxy_qp" % arm_key,
        )
        first_context = audit.mapping(
            first_qp.get("context"),
            "provider_%s_first_proxy_context" % arm_key,
        )
        context_p1 = _vector(
            audit,
            first_context.get("p1"),
            3,
            "provider_%s_first_proxy_context_p1" % arm_key,
        )
        context_R1 = _matrix(
            audit,
            first_context.get("R1"),
            3,
            3,
            "provider_%s_first_proxy_context_R1" % arm_key,
        )
        audit.check(
            _nested_close(context_p1, released_p1)
            and _nested_close(context_R1, released_R1),
            "provider_%s_action0_context_not_released_proxy" % arm_key,
        )
        proxy_bindings.append(
            (fresh_p1, fresh_R1, released_p1, released_R1)
        )
    audit.check(
        len(proxy_bindings) == 2
        and _nested_close(proxy_bindings[0], proxy_bindings[1]),
        "paired_action0_proxy_diagnostics_differ",
    )
    return implementation


def _validate_controller_contract(audit: _Audit, value: Any) -> Mapping[str, Any]:
    contract = audit.mapping(value, "controller_contract")
    expected = {
        "controller": CONTROLLER_NAME,
        "system_model": "qdot_equals_v",
        "decision_variable": "joint_velocity_rad_s",
        "cbf_constraint": "grad_h_dot_J_qdot_plus_alpha_h_ge_0",
        "static_obstacle_partial_t_h": 0.0,
        "control_frequency_hz": 100,
        "physics_frequency_hz": 500,
        "controller_updates_per_high_level_action": UPDATES_PER_ACTION,
        "physics_substeps_per_controller_update": SUBSTEPS_PER_UPDATE,
        "hard_constraints": True,
        "safety_slack_used": False,
        "fallback_policy": "none_fail_closed_as_negative",
        "zero_velocity_is_feasibility_witness_not_acceptance": True,
    }
    for key, expected_value in expected.items():
        observed = contract.get(key)
        if isinstance(expected_value, float):
            audit.check(
                not isinstance(observed, bool)
                and isinstance(observed, (int, float))
                and _close(float(observed), expected_value),
                "controller_contract_%s_differs" % key,
            )
        else:
            audit.check(
                observed == expected_value,
                "controller_contract_%s_differs" % key,
            )
    registered_thresholds = {
        "cbf_postcheck_tolerance_m2_per_s": CBF_TOLERANCE,
        "velocity_bound_tolerance_rad_s": BOUND_TOLERANCE,
        "material_correction_minimum_rad_s": MATERIAL_CORRECTION_MIN_RAD_S,
        "material_correction_integral_minimum_rad": (
            MATERIAL_CORRECTION_INTEGRAL_MIN_RAD
        ),
        "post_correction_joint_motion_minimum_rad": (
            POST_CORRECTION_JOINT_MOTION_MIN_RAD
        ),
        "post_correction_eef_path_minimum_m": POST_CORRECTION_EEF_PATH_MIN_M,
        "post_correction_executed_integral_minimum_rad": (
            POST_CORRECTION_EXECUTED_INTEGRAL_MIN_RAD
        ),
        "post_correction_zero_command_fraction_maximum": (
            POST_CORRECTION_ZERO_COMMAND_FRACTION_MAX
        ),
        "tracking_linf_maximum_rad_s": TRACKING_LINF_MAX_RAD_S,
        "tracking_rmse_maximum_rad_s": TRACKING_RMSE_MAX_RAD_S,
    }
    thresholds = audit.mapping(contract.get("acceptance_thresholds"), "acceptance_thresholds")
    for key, expected_value in registered_thresholds.items():
        observed = audit.number(thresholds.get(key), "threshold_%s" % key)
        audit.check(_close(observed, expected_value), "threshold_%s_differs" % key)
    return contract


def _terminal_kind(audit: _Audit, arm: Mapping[str, Any], label: str) -> Optional[str]:
    """Return the only two scientifically complete early-terminal kinds.

    A QP method stop is allowed only before its next physics transition.  A
    contact terminal is allowed only after the contact-bearing physics row is
    serialized.  Every other shortened treatment remains an apparatus error.
    """

    contact = arm.get("contact_terminated_early") is True
    method = arm.get("method_terminated_early") is True
    audit.check(not (contact and method), "%s_multiple_terminal_kinds" % label)
    if method:
        audit.check(
            isinstance(arm.get("method_stop"), Mapping),
            "%s_method_stop_record_missing" % label,
        )
        return "method_stop"
    audit.check(
        arm.get("method_stop") is None,
        "%s_unregistered_method_stop_record" % label,
    )
    if contact:
        return "contact"
    return None


def _validate_field(audit: _Audit, value: Any) -> Dict[str, Any]:
    field = audit.mapping(value, "field")
    resolved = audit.mapping(field.get("resolved_geometry"), "resolved_geometry")
    resolved_sets = {
        key: tuple(
            audit.integer(item, "resolved_%s" % key)
            for item in audit.sequence(resolved.get(key), "resolved_%s" % key)
        )
        for key in ("robot_geom_ids", "obstacle_geom_ids", "link56_geom_ids")
    }
    audit.check(
        bool(resolved_sets["robot_geom_ids"])
        and bool(resolved_sets["obstacle_geom_ids"])
        and bool(resolved_sets["link56_geom_ids"]),
        "resolved_geometry_sets_empty",
    )
    names = audit.sequence(
        field.get("protected_robot_body_names"), "protected_robot_body_names"
    )
    audit.check(
        tuple(names) == PROTECTED_BODY_NAMES,
        "protected_robot_body_names_not_fixed_link5_link6",
    )
    audit.check(
        field.get("selection_rule")
        == "fixed_shared_link5_link6_surface_union_for_all_cases",
        "protected_surface_selection_rule_differs",
    )
    audit.check(
        field.get("task_conditioned_link_selection") is False,
        "protected_surface_selection_is_task_conditioned",
    )
    evidence = audit.mapping(field.get("protected_samples"), "protected_samples")
    rows = audit.sequence(evidence.get("samples"), "protected_sample_rows")
    audit.check(bool(rows), "protected_sample_rows_empty")
    observed_names = set()
    for index, value_row in enumerate(rows):
        row = audit.mapping(value_row, "protected_sample_%d" % index)
        audit.check(row.get("sample_id") == index, "protected_sample_id_%d_differs" % index)
        name = row.get("body_name")
        audit.check(
            name in PROTECTED_BODY_NAMES,
            "protected_sample_%d_outside_link5_link6" % index,
        )
        if isinstance(name, str):
            observed_names.add(name)
        audit.check(
            isinstance(row.get("geom_name"), str) and bool(row.get("geom_name")),
            "protected_sample_%d_geom_name_invalid" % index,
        )
        point = _vector(
            audit,
            row.get("point_body_local_m"),
            3,
            "protected_sample_%d_point" % index,
        )
        del point
    audit.check(
        observed_names == set(PROTECTED_BODY_NAMES),
        "protected_samples_do_not_cover_both_link5_link6",
    )
    audit.check(
        evidence.get("sample_count") == len(rows),
        "protected_sample_count_differs",
    )
    audit.check(
        evidence.get("sample_ledger_sha256") == _canonical_sha256(rows),
        "protected_sample_ledger_sha256_differs",
    )
    return {
        "sample_count": len(rows),
        "sample_ledger_sha256": _canonical_sha256(rows),
        "body_names": list(PROTECTED_BODY_NAMES),
        "resolved_sets": resolved_sets,
    }


def _validate_restore_and_cadence(
    audit: _Audit,
    arm: Mapping[str, Any],
    label: str,
) -> None:
    restore = audit.mapping(arm.get("restore"), "%s_restore" % label)
    audit.check(
        restore.get("target_controller") == CONTROLLER_NAME,
        "%s_restore_target_controller_differs" % label,
    )
    for key in (
        "official_integration_state_sha256",
        "target_official_integration_state_sha256",
        "settled_state_sha256",
        "target_state_sha256",
    ):
        audit.check(_is_sha256(restore.get(key)), "%s_restore_%s_invalid" % (label, key))
    audit.check(
        restore.get("official_integration_state_sha256")
        == restore.get("target_official_integration_state_sha256"),
        "%s_restore_official_state_not_exact" % label,
    )
    audit.check(
        restore.get("settled_state_sha256") == restore.get("target_state_sha256"),
        "%s_restore_flat_state_not_exact" % label,
    )
    audit.check(
        restore.get("exact_flattened_state") is True,
        "%s_restore_not_exact" % label,
    )
    controller = audit.mapping(restore.get("controller"), "%s_controller" % label)
    audit.check(
        controller.get("controller_name") == CONTROLLER_NAME,
        "%s_controller_name_differs" % label,
    )
    for key, expected in (
        ("controller_class_qualname", "JointVelocityController"),
        ("environment_action_dim", 8),
        ("arm_control_dim", 7),
    ):
        audit.check(
            controller.get(key) == expected,
            "%s_controller_%s_differs" % (label, key),
        )
    audit.check(
        controller.get("control_frequency_hz") == 100,
        "%s_controller_frequency_differs" % label,
    )
    audit.check(
        controller.get("physics_substeps_per_control") == SUBSTEPS_PER_UPDATE,
        "%s_controller_substeps_differs" % label,
    )
    audit.check(
        _close(
            audit.number(
                controller.get("control_timestep_s"),
                "%s_controller_control_dt" % label,
            ),
            CONTROL_DT_S,
        ),
        "%s_controller_control_dt_differs" % label,
    )
    audit.check(
        _close(
            audit.number(
                controller.get("physics_timestep_s"),
                "%s_controller_physics_dt" % label,
            ),
            PHYSICS_DT_S,
        ),
        "%s_controller_physics_dt_differs" % label,
    )
    cadence = audit.mapping(arm.get("execution_cadence"), "%s_cadence" % label)
    for key, expected in (
        ("control_timestep_s", CONTROL_DT_S),
        ("wrapper_model_timestep_s", PHYSICS_DT_S),
        ("mujoco_model_timestep_s", PHYSICS_DT_S),
    ):
        audit.check(
            _close(audit.number(cadence.get(key), "%s_%s" % (label, key)), expected),
            "%s_%s_differs" % (label, key),
        )


def _validate_command_trace(
    audit: _Audit,
    arm: Mapping[str, Any],
    *,
    label: str,
    psf_enabled: bool,
    sample_count: int,
    terminal_kind: Optional[str] = None,
) -> List[Dict[str, Any]]:
    trace = audit.sequence(arm.get("command_trace"), "%s_command_trace" % label)
    audit.check(
        bool(trace) or terminal_kind == "method_stop",
        "%s_command_trace_empty" % label,
    )
    output: List[Dict[str, Any]] = []
    first_source = None
    action_vectors: Dict[int, Tuple[float, ...]] = {}
    for index, value in enumerate(trace):
        row = audit.mapping(value, "%s_command_%d" % (label, index))
        source = audit.integer(row.get("source_action_index"), "%s_command_%d_source" % (label, index))
        local = audit.integer(row.get("local_action_index"), "%s_command_%d_local" % (label, index))
        inner = audit.integer(row.get("inner_control_index"), "%s_command_%d_inner" % (label, index))
        if first_source is None:
            first_source = source
        expected_local = index // UPDATES_PER_ACTION
        expected_inner = index % UPDATES_PER_ACTION
        expected_source = int(first_source or 0) + expected_local
        audit.check(local == expected_local, "%s_command_%d_local_differs" % (label, index))
        audit.check(inner == expected_inner, "%s_command_%d_inner_differs" % (label, index))
        audit.check(source == expected_source, "%s_command_%d_source_differs" % (label, index))
        audit.check(
            row.get("physical_boundary") == source * 25 + inner * 5,
            "%s_command_%d_boundary_differs" % (label, index),
        )
        source_action = _vector(audit, row.get("source_action"), 7, "%s_command_%d_action" % (label, index))
        if source in action_vectors:
            audit.check(
                source_action == action_vectors[source],
                "%s_action_%d_changed_within_updates" % (label, source),
            )
        else:
            action_vectors[source] = source_action
        nominal = _vector(audit, row.get("nominal_qdot_rad_s"), 7, "%s_command_%d_nominal" % (label, index))
        executed = _vector(audit, row.get("executed_qdot_rad_s"), 7, "%s_command_%d_executed" % (label, index))
        lower = _vector(audit, row.get("nominal_dynamic_lower_bound_rad_s"), 7, "%s_command_%d_lower" % (label, index))
        upper = _vector(audit, row.get("nominal_dynamic_upper_bound_rad_s"), 7, "%s_command_%d_upper" % (label, index))
        correction = _norm(tuple(safe - raw for safe, raw in zip(executed, nominal)))
        audit.check(
            _close(
                audit.number(row.get("correction_l2_rad_s"), "%s_command_%d_correction" % (label, index)),
                correction,
            ),
            "%s_command_%d_correction_differs" % (label, index),
        )
        bounds_valid = all(low <= high for low, high in zip(lower, upper))
        audit.check(bounds_valid, "%s_command_%d_bounds_contradictory" % (label, index))
        audit.check(
            all(low <= BOUND_TOLERANCE and high >= -BOUND_TOLERANCE for low, high in zip(lower, upper)),
            "%s_command_%d_zero_not_in_dynamic_bounds" % (label, index),
        )
        audit.check(
            all(
                low - BOUND_TOLERANCE <= value_safe <= high + BOUND_TOLERANCE
                for value_safe, low, high in zip(executed, lower, upper)
            ),
            "%s_command_%d_executed_outside_bounds" % (label, index),
        )
        nominal_within_bounds = all(
            low - BOUND_TOLERANCE <= value_nominal <= high + BOUND_TOLERANCE
            for value_nominal, low, high in zip(nominal, lower, upper)
        )
        audit.check(
            row.get("nominal_within_dynamic_joint_bounds")
            is nominal_within_bounds,
            "%s_command_%d_nominal_bound_flag_differs" % (label, index),
        )
        qp = row.get("qp")
        if psf_enabled:
            qp_record = audit.mapping(qp, "%s_command_%d_qp" % (label, index))
            status = str(qp_record.get("status", "")).lower()
            audit.check(
                status in ("solved", "solved inaccurate"),
                "%s_command_%d_qp_not_solved" % (label, index),
            )
            audit.check(
                qp_record.get("input_constraint_count") == sample_count,
                "%s_command_%d_qp_sample_count_differs" % (label, index),
            )
            solved_count = audit.integer(
                qp_record.get("solved_constraint_count"),
                "%s_command_%d_qp_solved_count" % (label, index),
            )
            audit.check(
                0 <= solved_count <= sample_count,
                "%s_command_%d_qp_solved_count_out_of_range" % (label, index),
            )
            safe_residual = audit.number(
                row.get("safe_cbf_residual_minimum_m2_per_s"),
                "%s_command_%d_safe_residual" % (label, index),
            )
            qp_residual = audit.number(
                qp_record.get("minimum_raw_cbf_residual_m2_per_s"),
                "%s_command_%d_qp_residual" % (label, index),
            )
            audit.check(
                _close(safe_residual, qp_residual),
                "%s_command_%d_qp_residual_differs" % (label, index),
            )
            audit.check(
                safe_residual >= -CBF_TOLERANCE,
                "%s_command_%d_cbf_postcheck_failed" % (label, index),
            )
            bound_violation = audit.number(
                qp_record.get("maximum_velocity_bound_violation_rad_s"),
                "%s_command_%d_qp_bound_violation" % (label, index),
            )
            audit.check(
                bound_violation <= BOUND_TOLERANCE,
                "%s_command_%d_qp_bound_postcheck_failed" % (label, index),
            )
            nominal_residual_value = row.get("nominal_cbf_residual_minimum_m2_per_s")
            nominal_residual = audit.number(
                nominal_residual_value,
                "%s_command_%d_nominal_residual" % (label, index),
            )
            sample = row.get("nominal_argmin_protected_sample")
            if sample is not None:
                sample_record = audit.mapping(sample, "%s_command_%d_argmin_sample" % (label, index))
                audit.check(
                    sample_record.get("body_name") in PROTECTED_BODY_NAMES,
                    "%s_command_%d_argmin_outside_link5_link6" % (label, index),
                )
        else:
            nominal_residual = None
            audit.check(qp is None, "%s_command_%d_unexpected_qp" % (label, index))
            audit.check(
                correction <= FLOAT_TOLERANCE and all(_close(a, b) for a, b in zip(nominal, executed)),
                "%s_command_%d_adapter_not_passthrough" % (label, index),
            )
            audit.check(
                row.get("safe_cbf_residual_minimum_m2_per_s") is None
                and row.get("nominal_cbf_residual_minimum_m2_per_s") is None,
                "%s_command_%d_adapter_has_cbf_fields" % (label, index),
            )
        output.append(
            {
                "source": source,
                "local": local,
                "inner": inner,
                "boundary": source * 25 + inner * 5,
                "source_action": source_action,
                "nominal": nominal,
                "executed": executed,
                "correction": correction,
                "nominal_residual": nominal_residual,
                "nominal_within_bounds": nominal_within_bounds,
                "eef_before": _vector(
                    audit,
                    row.get("eef_position_before_update_world_m"),
                    3,
                    "%s_command_%d_eef" % (label, index),
                ),
                "raw": row,
            }
        )
    if terminal_kind is None:
        audit.check(
            len(trace) % UPDATES_PER_ACTION == 0,
            "%s_command_count_not_complete_actions" % label,
        )
    expected_qp = len(trace) if psf_enabled else 0
    for field, expected in (
        ("filter_update_count", len(trace)),
        ("qp_solve_count", expected_qp),
        ("qp_postcheck_count", expected_qp),
        ("joint_limit_postcheck_count", expected_qp),
        ("issued_command_bound_check_count", len(trace)),
    ):
        audit.check(arm.get(field) == expected, "%s_%s_differs" % (label, field))
    audit.check(
        arm.get("all_issued_commands_within_physical_bounds")
        is bool(len(trace) > 0),
        "%s_issued_command_bound_summary_failed" % label,
    )
    if psf_enabled:
        audit.check(
            arm.get("protected_sample_count") == sample_count,
            "%s_protected_sample_count_differs" % label,
        )
        audit.check(
            arm.get("invalid_field_query_count") == 0,
            "%s_invalid_field_query_present" % label,
        )
        audit.check(
            arm.get("pre_filter_field_observation_count") == len(trace),
            "%s_field_observation_count_differs" % label,
        )
        audit.check(
            arm.get("pre_filter_field_query_count") == len(trace) * sample_count,
            "%s_field_query_count_differs" % label,
        )
    nominal_violation_count = sum(
        not row["nominal_within_bounds"] for row in output
    )
    audit.check(
        arm.get("nominal_dynamic_bound_check_count") == len(output),
        "%s_nominal_dynamic_bound_check_count_differs" % label,
    )
    audit.check(
        arm.get("nominal_dynamic_bound_violation_count")
        == nominal_violation_count,
        "%s_nominal_dynamic_bound_violation_count_differs" % label,
    )
    audit.check(
        arm.get("all_nominal_commands_within_dynamic_joint_bounds")
        is bool(output and nominal_violation_count == 0),
        "%s_nominal_dynamic_bound_summary_differs" % label,
    )
    return output


def _validate_physics_trace(
    audit: _Audit,
    arm: Mapping[str, Any],
    commands: Sequence[Mapping[str, Any]],
    *,
    label: str,
    terminal_kind: Optional[str] = None,
) -> List[Dict[str, Any]]:
    trace = audit.sequence(arm.get("physics_trace"), "%s_physics_trace" % label)
    expected_count = len(commands) * SUBSTEPS_PER_UPDATE
    if terminal_kind == "contact":
        minimum_count = (
            0
            if not commands
            else (len(commands) - 1) * SUBSTEPS_PER_UPDATE + 1
        )
        audit.check(
            bool(commands) and minimum_count <= len(trace) <= expected_count,
            "%s_contact_terminal_physics_count_differs" % label,
        )
    else:
        audit.check(len(trace) == expected_count, "%s_physics_count_differs" % label)
    output: List[Dict[str, Any]] = []
    for index, value in enumerate(trace):
        row = audit.mapping(value, "%s_physics_%d" % (label, index))
        command_index = index // SUBSTEPS_PER_UPDATE
        local_substep = index % SUBSTEPS_PER_UPDATE
        if command_index >= len(commands):
            break
        command = commands[command_index]
        audit.check(row.get("observation_index") == index, "%s_physics_%d_observation_differs" % (label, index))
        for field, expected in (
            ("local_action_index", command["local"]),
            ("source_action_index", command["source"]),
            ("inner_control_index", command["inner"]),
            ("physics_substep_index", local_substep),
            ("post_state_physical_boundary", command["boundary"] + local_substep + 1),
        ):
            audit.check(row.get(field) == expected, "%s_physics_%d_%s_differs" % (label, index, field))
        measured = _vector(audit, row.get("measured_qvel_rad_s"), 7, "%s_physics_%d_measured" % (label, index))
        issued = _vector(audit, row.get("issued_qvel_rad_s"), 7, "%s_physics_%d_issued" % (label, index))
        stored_error = _vector(audit, row.get("tracking_error_rad_s"), 7, "%s_physics_%d_error" % (label, index))
        expected_error = tuple(a - b for a, b in zip(measured, issued))
        audit.check(
            all(_close(a, b) for a, b in zip(issued, command["executed"])),
            "%s_physics_%d_issued_command_differs" % (label, index),
        )
        audit.check(
            all(_close(a, b) for a, b in zip(stored_error, expected_error)),
            "%s_physics_%d_tracking_error_differs" % (label, index),
        )
        terminal_contact_row = bool(
            terminal_kind == "contact" and index == len(trace) - 1
        )
        eef_value = row.get("eef_position_world_m")
        if terminal_contact_row and eef_value is None:
            eef = None
        else:
            eef = _vector(
                audit,
                eef_value,
                3,
                "%s_physics_%d_eef" % (label, index),
            )
        output.append(
            {
                "index": index,
                "transition_boundary": command["boundary"] + local_substep,
                "source": command["source"],
                "inner": command["inner"],
                "substep": local_substep,
                "measured": measured,
                "issued": issued,
                "error": expected_error,
                "eef": eef,
                "raw": row,
            }
        )
    for field, expected in (
        ("physics_substep_count", len(trace)),
        ("physics_trace_row_count", len(trace)),
        ("monitor_observed_physics_substep_count", len(trace)),
    ):
        audit.check(arm.get(field) == expected, "%s_%s_differs" % (label, field))
    audit.check(
        arm.get("physics_monitor_trace_counts_match") is True,
        "%s_physics_monitor_count_mismatch" % label,
    )
    audit.check(
        arm.get("exposure_complete") is (terminal_kind is None),
        "%s_exposure_terminal_summary_differs" % label,
    )
    return output


def _validate_method_stop(
    audit: _Audit,
    arm: Mapping[str, Any],
    commands: Sequence[Mapping[str, Any]],
    provider: Mapping[str, Any],
    *,
    label: str,
) -> None:
    record = audit.mapping(arm.get("method_stop"), "%s_method_stop" % label)
    next_index = len(commands)
    expected_local = next_index // UPDATES_PER_ACTION
    expected_inner = next_index % UPDATES_PER_ACTION
    expected_boundary = expected_local * UPDATES_PER_ACTION * SUBSTEPS_PER_UPDATE + expected_inner * SUBSTEPS_PER_UPDATE
    for field, expected in (
        ("local_action_index", expected_local),
        ("source_action_index", expected_local),
        ("inner_control_index", expected_inner),
        ("physical_boundary", expected_boundary),
        ("valid", False),
    ):
        audit.check(
            record.get(field) == expected,
            "%s_method_stop_%s_differs" % (label, field),
        )
    audit.check(
        isinstance(record.get("reason"), str) and bool(record.get("reason")),
        "%s_method_stop_reason_invalid" % label,
    )
    audit.mapping(record.get("diagnostics"), "%s_method_stop_diagnostics" % label)
    actions = audit.sequence(
        provider.get("high_level_action_trace"), "%s_method_stop_provider_actions" % label
    )
    audit.check(
        len(actions) == expected_local + 1,
        "%s_method_stop_provider_action_count_differs" % label,
    )
    if expected_local < len(actions):
        provider_row = audit.mapping(
            actions[expected_local], "%s_method_stop_provider_action" % label
        )
        expected_action = _vector(
            audit,
            provider_row.get("aegis_executed"),
            7,
            "%s_method_stop_provider_executed" % label,
        )
        observed_action = _vector(
            audit,
            record.get("source_action"),
            7,
            "%s_method_stop_source_action" % label,
        )
        audit.check(
            observed_action == expected_action,
            "%s_method_stop_source_action_differs" % label,
        )


def _validate_contact_terminal(
    audit: _Audit,
    physics: Sequence[Mapping[str, Any]],
    registered: Mapping[str, Any],
    *,
    label: str,
) -> None:
    audit.check(bool(physics), "%s_contact_terminal_without_physics" % label)
    if not physics:
        return
    last = physics[-1]
    audit.check(
        last["raw"].get("literal_contact_observed") is True,
        "%s_contact_terminal_last_row_not_contact" % label,
    )
    audit.check(
        last["raw"].get("registered_forbidden_contact_observed") is True,
        "%s_contact_terminal_last_row_not_registered" % label,
    )
    audit.check(
        registered.get("last_boundary") == last["transition_boundary"],
        "%s_contact_terminal_boundary_differs" % label,
    )


def _selected_contacts(
    audit: _Audit,
    arm: Mapping[str, Any],
    *,
    label: str,
) -> Dict[str, Any]:
    measurement = audit.mapping(arm.get("measurement"), "%s_measurement" % label)
    settled = audit.mapping(measurement.get("settled_state"), "%s_settled" % label)
    settled_records = audit.sequence(
        settled.get("physical_contact_point_records"),
        "%s_settled_physical_records" % label,
    )
    audit.check(not settled_records, "%s_settled_selected_contact_present" % label)
    audit.check(
        settled.get("any_robot_obstacle_contact") is False,
        "%s_settled_selected_contact_flag" % label,
    )
    live_rows = audit.sequence(
        measurement.get("live_solver_phase_contact_point_records"),
        "%s_live_contact_rows" % label,
    )
    post_candidates = audit.sequence(
        measurement.get("post_state_candidate_contact_point_records"),
        "%s_post_candidate_rows" % label,
    )
    physical: List[Mapping[str, Any]] = []
    for phase, rows in (("live", live_rows), ("post", post_candidates)):
        for index, value in enumerate(rows):
            row = audit.mapping(value, "%s_%s_contact_%d" % (label, phase, index))
            distance = audit.number(row.get("contact_distance_m"), "%s_%s_contact_%d_distance" % (label, phase, index))
            is_physical = distance <= 0.0
            audit.check(
                row.get("is_physical_nonpositive_distance_contact") is is_physical,
                "%s_%s_contact_%d_physical_flag_differs" % (label, phase, index),
            )
            if is_physical:
                physical.append(row)
    live_physical = sum(
        1
        for row in live_rows
        if isinstance(row, Mapping)
        and isinstance(row.get("contact_distance_m"), (int, float))
        and not isinstance(row.get("contact_distance_m"), bool)
        and float(row["contact_distance_m"]) <= 0.0
    )
    post_physical = len(physical) - live_physical
    audit.check(
        measurement.get("live_solver_nonpositive_contact_point_record_count") == live_physical,
        "%s_live_physical_count_differs" % label,
    )
    audit.check(
        measurement.get("post_state_physical_contact_point_record_count") == post_physical,
        "%s_post_physical_count_differs" % label,
    )
    audit.check(
        measurement.get("total_physical_contact_point_record_count") == len(physical),
        "%s_total_physical_count_differs" % label,
    )
    any_present = bool(physical)
    link_rows = [
        row for row in physical if row.get("robot_body_name") in PROTECTED_BODY_NAMES
    ]
    link_present = bool(link_rows)
    boundaries = []
    for row in physical:
        observation = audit.integer(
            row.get("observation_index"), "%s_contact_observation" % label
        )
        phase = row.get("source_phase")
        audit.check(
            phase
            in (
                "live_solver_phase_preintegration_geometry",
                "post_integration_recomputed",
            ),
            "%s_contact_source_phase_invalid" % label,
        )
        boundaries.append(
            observation
            + (1 if phase == "post_integration_recomputed" else 0)
        )
    first_any = min(boundaries) if boundaries else None
    link_boundaries = []
    for row in link_rows:
        observation = audit.integer(
            row.get("observation_index"), "%s_link_contact_observation" % label
        )
        link_boundaries.append(
            observation
            + (1 if row.get("source_phase") == "post_integration_recomputed" else 0)
        )
    first_link = min(link_boundaries) if link_boundaries else None
    audit.check(
        measurement.get("any_robot_obstacle_contact") is any_present,
        "%s_selected_contact_flag_differs" % label,
    )
    audit.check(
        measurement.get("link56_obstacle_contact") is link_present,
        "%s_link56_contact_flag_differs" % label,
    )
    literal = audit.mapping(arm.get("literal_contact"), "%s_literal_contact" % label)
    audit.check(
        literal.get("any_robot_selected_obstacle_present") is any_present,
        "%s_literal_any_contact_differs" % label,
    )
    audit.check(
        literal.get("link56_present") is link_present,
        "%s_literal_link56_contact_differs" % label,
    )
    audit.check(
        literal.get("first_any_robot_physical_boundary") == first_any,
        "%s_literal_first_any_boundary_differs" % label,
    )
    audit.check(
        literal.get("first_link56_physical_boundary") == first_link,
        "%s_literal_first_link_boundary_differs" % label,
    )
    return {
        "any_present": any_present,
        "link56_present": link_present,
        "first_any_boundary": first_any,
        "first_link56_boundary": first_link,
    }


def _validate_registered_contacts(
    audit: _Audit,
    arm: Mapping[str, Any],
    physics_count: int,
    *,
    label: str,
) -> Dict[str, Any]:
    measurement = audit.mapping(
        arm.get("registered_forbidden_contact"),
        "%s_registered_contact_measurement" % label,
    )
    audit.check(
        measurement.get("schema_version")
        == "vlsa_poisson_registered_contact_measurement.v2",
        "%s_registered_contact_schema_differs" % label,
    )
    scope = audit.mapping(
        measurement.get("scope"), "%s_registered_contact_scope" % label
    )
    scope_unhashed = dict(scope)
    scope_hash = scope_unhashed.pop("identity_sha256", None)
    audit.check(
        scope.get("schema_version") == "vlsa_poisson_registered_contact_scope.v1",
        "%s_registered_contact_scope_schema_differs" % label,
    )
    audit.check(
        scope_hash == _canonical_sha256(scope_unhashed),
        "%s_registered_contact_scope_hash_differs" % label,
    )
    audit.check(
        measurement.get("scope_identity_sha256") == scope_hash,
        "%s_registered_contact_scope_identity_differs" % label,
    )
    robot_ids = {
        audit.integer(value, "%s_scope_robot_geom" % label)
        for value in audit.sequence(
            scope.get("robot_geom_ids"), "%s_scope_robot_geom_ids" % label
        )
    }
    selected_ids = {
        audit.integer(value, "%s_scope_selected_geom" % label)
        for value in audit.sequence(
            scope.get("selected_obstacle_geom_ids"),
            "%s_scope_selected_geom_ids" % label,
        )
    }
    link56_ids = {
        audit.integer(value, "%s_scope_link56_geom" % label)
        for value in audit.sequence(
            scope.get("link56_geom_ids"), "%s_scope_link56_geom_ids" % label
        )
    }
    external_ids = {
        audit.integer(value, "%s_scope_external_geom" % label)
        for value in audit.sequence(
            scope.get("external_nonrobot_geom_ids"),
            "%s_scope_external_geom_ids" % label,
        )
    }
    audit.check(
        bool(robot_ids)
        and bool(selected_ids)
        and bool(link56_ids)
        and bool(external_ids)
        and link56_ids <= robot_ids
        and selected_ids <= external_ids
        and not (robot_ids & external_ids),
        "%s_registered_contact_scope_sets_invalid" % label,
    )
    audit.check(
        measurement.get("observed_physics_substeps") == physics_count,
        "%s_registered_contact_observation_count_differs" % label,
    )
    audit.check(
        measurement.get("callback_cadence")
        == {
            "controller_updates_per_action": UPDATES_PER_ACTION,
            "physics_substeps_per_controller_update": SUBSTEPS_PER_UPDATE,
            "physics_substeps_per_action": UPDATES_PER_ACTION * SUBSTEPS_PER_UPDATE,
        },
        "%s_registered_contact_cadence_differs" % label,
    )
    settled = audit.sequence(
        measurement.get("settled_contact_records"),
        "%s_registered_settled_records" % label,
    )
    rollout = audit.sequence(
        measurement.get("rollout_contact_records"),
        "%s_registered_rollout_records" % label,
    )
    audit.check(not settled, "%s_registered_settled_contact_present" % label)
    selected_any = False
    shifted_any = False
    link_external_any = False
    contact_boundaries: List[int] = []
    for index, value in enumerate(rollout):
        row = audit.mapping(value, "%s_registered_contact_%d" % (label, index))
        unhashed = dict(row)
        observed_hash = unhashed.pop("record_sha256", None)
        audit.check(
            observed_hash == _canonical_sha256(unhashed),
            "%s_registered_contact_%d_hash_differs" % (label, index),
        )
        distance = audit.number(
            row.get("contact_distance_m"),
            "%s_registered_contact_%d_distance" % (label, index),
        )
        audit.check(distance <= 0.0, "%s_registered_contact_%d_not_literal" % (label, index))
        boundary = audit.integer(
            row.get("physical_boundary"),
            "%s_registered_contact_%d_boundary" % (label, index),
        )
        audit.check(
            0 <= boundary < physics_count,
            "%s_registered_contact_%d_boundary_out_of_range" % (label, index),
        )
        expected_source = boundary // (UPDATES_PER_ACTION * SUBSTEPS_PER_UPDATE)
        expected_substep = boundary % (UPDATES_PER_ACTION * SUBSTEPS_PER_UPDATE)
        expected_inner = expected_substep // SUBSTEPS_PER_UPDATE
        expected_local_substep = expected_substep % SUBSTEPS_PER_UPDATE
        source_phase = row.get("source_phase")
        audit.check(
            source_phase
            in (
                "live_solver_phase_preintegration_geometry",
                "post_integration_recomputed",
            ),
            "%s_registered_contact_%d_source_phase_invalid" % (label, index),
        )
        for field, expected in (
            ("executed_transition_start_boundary", boundary),
            (
                "observed_state_boundary",
                boundary
                if source_phase == "live_solver_phase_preintegration_geometry"
                else boundary + 1,
            ),
            ("source_action_index", expected_source),
            ("physics_substep_index", expected_substep),
            ("controller_update_index", expected_inner),
            ("physics_substep_within_controller_update", expected_local_substep),
            (
                "callback_endpoint_action_inner_substep",
                [expected_source, expected_inner, expected_local_substep],
            ),
        ):
            audit.check(
                row.get(field) == expected,
                "%s_registered_contact_%d_%s_differs" % (label, index, field),
            )
        categories = audit.sequence(
            row.get("contact_categories"),
            "%s_registered_contact_%d_categories" % (label, index),
        )
        robot_geom_id = audit.integer(
            row.get("robot_geom_id"),
            "%s_registered_contact_%d_robot_geom" % (label, index),
        )
        external_geom_id = audit.integer(
            row.get("external_geom_id"),
            "%s_registered_contact_%d_external_geom" % (label, index),
        )
        selected = bool(
            robot_geom_id in robot_ids and external_geom_id in selected_ids
        )
        link_external = bool(
            robot_geom_id in link56_ids and external_geom_id in external_ids
        )
        shifted = bool(link_external and external_geom_id not in selected_ids)
        expected_categories = []
        if selected:
            expected_categories.append("any_robot_vs_selected_obstacle")
        if link_external:
            expected_categories.append("link56_vs_external_nonrobot")
        audit.check(
            list(categories) == expected_categories,
            "%s_registered_contact_%d_categories_differ" % (label, index),
        )
        audit.check(
            selected or link_external,
            "%s_registered_contact_%d_outside_registered_union" % (label, index),
        )
        audit.check(
            row.get("any_robot_selected_obstacle_contact") is selected,
            "%s_registered_contact_%d_selected_flag_differs" % (label, index),
        )
        audit.check(
            row.get("link56_external_nonrobot_contact") is link_external,
            "%s_registered_contact_%d_link_external_flag_differs" % (label, index),
        )
        audit.check(
            row.get("link56_nonselected_external_contact") is shifted,
            "%s_registered_contact_%d_shifted_flag_differs" % (label, index),
        )
        selected_any = selected_any or selected
        link_external_any = link_external_any or link_external
        shifted_any = shifted_any or shifted
        contact_boundaries.append(boundary)
    audit.check(
        measurement.get("settled_forbidden_contact") is False,
        "%s_registered_settled_flag_differs" % label,
    )
    audit.check(
        measurement.get("rollout_forbidden_contact") is bool(rollout),
        "%s_registered_rollout_flag_differs" % label,
    )
    audit.check(
        measurement.get("any_registered_forbidden_contact") is bool(rollout),
        "%s_registered_any_flag_differs" % label,
    )
    audit.check(
        measurement.get("any_robot_selected_obstacle_contact") is selected_any,
        "%s_registered_selected_summary_differs" % label,
    )
    audit.check(
        measurement.get("any_link56_external_nonrobot_contact") is link_external_any,
        "%s_registered_link_external_summary_differs" % label,
    )
    audit.check(
        measurement.get("any_link56_nonselected_external_contact") is shifted_any,
        "%s_registered_shifted_summary_differs" % label,
    )
    return {
        "any_forbidden": bool(rollout),
        "selected_contact": selected_any,
        "link56_external_contact": link_external_any,
        "shifted_contact": shifted_any,
        "first_boundary": min(contact_boundaries) if contact_boundaries else None,
        "last_boundary": max(contact_boundaries) if contact_boundaries else None,
        "scope_sets": {
            "robot_geom_ids": tuple(sorted(robot_ids)),
            "obstacle_geom_ids": tuple(sorted(selected_ids)),
            "link56_geom_ids": tuple(sorted(link56_ids)),
        },
    }


def _validate_task(
    audit: _Audit,
    arm: Mapping[str, Any],
    label: str,
    *,
    terminal_kind: Optional[str] = None,
) -> Dict[str, Any]:
    task = audit.mapping(arm.get("task"), "%s_task" % label)
    audit.check(
        task.get("source") == "native_bddl_goal_predicates",
        "%s_task_source_differs" % label,
    )
    ledger = audit.sequence(task.get("goal_progress_ledger"), "%s_goal_ledger" % label)
    success_rows: List[Mapping[str, Any]] = []
    completed_success_rows: List[Mapping[str, Any]] = []
    completed_rows = 0
    completed_hashes: List[str] = []
    completed_sources: List[int] = []
    audit.check(bool(ledger), "%s_goal_ledger_empty" % label)
    for index, value in enumerate(ledger):
        row = audit.mapping(value, "%s_goal_%d" % (label, index))
        values = audit.sequence(row.get("values"), "%s_goal_%d_values" % (label, index))
        audit.check(
            bool(values) and all(isinstance(item, bool) for item in values),
            "%s_goal_%d_values_invalid" % (label, index),
        )
        all_satisfied = bool(values) and all(values)
        fraction = 0.0 if not values else sum(bool(item) for item in values) / len(values)
        audit.check(
            row.get("all_satisfied") is all_satisfied,
            "%s_goal_%d_all_satisfied_differs" % (label, index),
        )
        audit.check(
            _close(audit.number(row.get("fraction"), "%s_goal_%d_fraction" % (label, index)), fraction),
            "%s_goal_%d_fraction_differs" % (label, index),
        )
        if row.get("snapshot_kind") == "completed_high_level_post_step":
            completed_rows += 1
            completed_sources.append(
                audit.integer(
                    row.get("source_action_index"),
                    "%s_goal_%d_source" % (label, index),
                )
            )
            audit.check(
                row.get("returned_done") is all_satisfied,
                "%s_goal_%d_done_differs" % (label, index),
            )
            audit.check(
                _is_sha256(row.get("returned_observation_sha256")),
                "%s_goal_%d_observation_hash_invalid" % (label, index),
            )
            if isinstance(row.get("returned_observation_sha256"), str):
                completed_hashes.append(row["returned_observation_sha256"])
            if all_satisfied:
                success_rows.append(row)
                completed_success_rows.append(row)
        elif index == 0:
            audit.check(
                row.get("snapshot_kind") == "branch_boundary_pre_action"
                and row.get("source_action_index") == -1,
                "%s_goal_branch_row_differs" % label,
            )
        elif terminal_kind is not None and index == len(ledger) - 1:
            expected_kind = (
                "method_stop_prephysics_terminal_state_diagnostic"
                if terminal_kind == "method_stop"
                else "partial_action_terminal_state_diagnostic"
            )
            audit.check(
                row.get("snapshot_kind") == expected_kind
                and row.get("source_action_index") == completed_rows,
                "%s_goal_terminal_diagnostic_differs" % label,
            )
            audit.check(
                row.get("returned_done") is None,
                "%s_goal_terminal_done_not_none" % label,
            )
            if all_satisfied:
                success_rows.append(row)
        else:
            audit.check(False, "%s_goal_%d_snapshot_kind_invalid" % (label, index))
    audit.check(
        completed_sources == list(range(completed_rows)),
        "%s_goal_completed_sources_not_contiguous" % label,
    )
    first_success = (
        min(int(row["source_action_index"]) for row in success_rows)
        if success_rows
        else None
    )
    terminal_success = bool(ledger and ledger[-1].get("all_satisfied") is True)
    audit.check(
        task.get("completed_source_action_count") == completed_rows,
        "%s_task_completed_count_differs" % label,
    )
    audit.check(
        task.get("initial_task_success_at_branch") is False,
        "%s_task_already_successful_at_start" % label,
    )
    audit.check(
        task.get("ever_task_success_at_or_after_branch") is bool(success_rows),
        "%s_task_ever_success_differs" % label,
    )
    audit.check(
        task.get("first_task_success_source_action_index") == first_success,
        "%s_task_first_success_differs" % label,
    )
    audit.check(
        task.get("terminal_task_success") is terminal_success,
        "%s_task_terminal_success_differs" % label,
    )
    returned_ledger = audit.sequence(
        task.get("returned_observation_sha256_ledger"),
        "%s_returned_observation_hashes" % label,
    )
    audit.check(
        list(returned_ledger) == completed_hashes,
        "%s_returned_observation_hash_ledger_differs" % label,
    )
    return {
        "success": bool(success_rows) and terminal_success,
        "first_success_source_action_index": first_success,
        "completed_action_count": completed_rows,
        "registered_action_count": audit.integer(
            task.get("registered_source_action_count"),
            "%s_task_registered_action_count" % label,
        ),
        "completed_success_sources": [
            int(row["source_action_index"]) for row in completed_success_rows
        ],
    }


def _validate_paper_car(
    audit: _Audit,
    arm: Mapping[str, Any],
    *,
    label: str,
    completed_action_count: int,
    expected_action_count: int,
    terminal_kind: Optional[str] = None,
) -> Dict[str, Any]:
    car = audit.mapping(arm.get("paper_car"), "%s_paper_car" % label)
    audit.check(car.get("enabled") is True, "%s_paper_car_not_enabled" % label)
    initial = _vector(
        audit,
        car.get("initial_active_obstacle_position_m"),
        3,
        "%s_paper_car_initial" % label,
    )
    ledger = audit.sequence(car.get("endpoint_ledger"), "%s_paper_car_ledger" % label)
    audit.check(
        len(ledger) == completed_action_count,
        "%s_paper_car_endpoint_count_differs" % label,
    )
    maximum = 0.0
    first_collision = None
    threshold = None
    for index, value in enumerate(ledger):
        row = audit.mapping(value, "%s_paper_car_%d" % (label, index))
        source = audit.integer(
            row.get("source_action_index"),
            "%s_paper_car_%d_source" % (label, index),
        )
        audit.check(source == index, "%s_paper_car_%d_source_differs" % (label, index))
        position = _vector(
            audit,
            row.get("active_obstacle_position_m"),
            3,
            "%s_paper_car_%d_position" % (label, index),
        )
        displacement = sum(abs(a - b) for a, b in zip(position, initial))
        stored = audit.number(
            row.get("l1_displacement_from_settled_m"),
            "%s_paper_car_%d_displacement" % (label, index),
        )
        audit.check(
            _close(stored, displacement),
            "%s_paper_car_%d_displacement_differs" % (label, index),
        )
        row_threshold = audit.number(
            row.get("paper_collision_threshold_m"),
            "%s_paper_car_%d_threshold" % (label, index),
        )
        if threshold is None:
            threshold = row_threshold
        audit.check(
            threshold is not None and _close(row_threshold, threshold),
            "%s_paper_car_%d_threshold_differs" % (label, index),
        )
        collision = bool(displacement > row_threshold)
        audit.check(
            row.get("paper_collision") is collision,
            "%s_paper_car_%d_collision_flag_differs" % (label, index),
        )
        audit.check(
            _is_sha256(row.get("returned_observation_sha256")),
            "%s_paper_car_%d_observation_hash_invalid" % (label, index),
        )
        maximum = max(maximum, displacement)
        if collision and first_collision is None:
            first_collision = source
    if threshold is None:
        threshold = audit.number(
            car.get("paper_collision_threshold_m"),
            "%s_paper_car_threshold" % label,
        )
    audit.check(
        car.get("expected_endpoint_count") == expected_action_count
        and car.get("endpoint_ledger_complete")
        is bool(completed_action_count == expected_action_count),
        "%s_paper_car_completion_differs" % label,
    )
    if terminal_kind is None:
        audit.check(
            completed_action_count == expected_action_count,
            "%s_paper_car_full_horizon_incomplete" % label,
        )
    audit.check(
        _close(
            audit.number(
                car.get("maximum_active_obstacle_l1_displacement_m"),
                "%s_paper_car_maximum" % label,
            ),
            maximum,
        ),
        "%s_paper_car_maximum_differs" % label,
    )
    audit.check(
        car.get("collision_first_source_action_index") == first_collision,
        "%s_paper_car_first_collision_differs" % label,
    )
    avoided = bool(first_collision is None and maximum <= threshold)
    audit.check(
        car.get("paper_collision_avoidance") is avoided,
        "%s_paper_car_avoidance_differs" % label,
    )
    return {
        "avoided": avoided,
        "maximum_displacement_m": maximum,
        "first_collision_source_action_index": first_collision,
        "complete": completed_action_count == expected_action_count,
    }


def _tracking(
    physics: Sequence[Mapping[str, Any]],
    *,
    end_transition_boundary_exclusive: Optional[int] = None,
) -> Dict[str, Any]:
    rows = [
        row
        for row in physics
        if end_transition_boundary_exclusive is None
        or int(row["transition_boundary"]) < end_transition_boundary_exclusive
    ]
    absolute = [abs(value) for row in rows for value in row["error"]]
    squared = [value * value for row in rows for value in row["error"]]
    linf = max(absolute, default=0.0)
    rmse = math.sqrt(sum(squared) / len(squared)) if squared else 0.0
    return {
        "observation_count": len(rows),
        "linf_rad_s": linf,
        "rmse_rad_s": rmse,
        "valid": bool(
            rows
            and linf <= TRACKING_LINF_MAX_RAD_S
            and rmse <= TRACKING_RMSE_MAX_RAD_S
        ),
    }


def _post_correction_motion(
    commands: Sequence[Mapping[str, Any]],
    physics: Sequence[Mapping[str, Any]],
    *,
    first_correction_boundary: Optional[int],
    first_success_source_action_index: Optional[int],
) -> Dict[str, Any]:
    if first_correction_boundary is None:
        return {
            "command_count": 0,
            "physics_substep_count": 0,
            "maximum_correction_norm_rad_s": 0.0,
            "filter_correction_integral_rad": 0.0,
            "executed_command_integral_rad": 0.0,
            "measured_joint_motion_integral_rad": 0.0,
            "cartesian_path_length_m": 0.0,
            "zero_command_fraction": 1.0,
            "complete_command_interval_count": 0,
        }
    end_boundary = (
        None
        if first_success_source_action_index is None
        else (int(first_success_source_action_index) + 1) * 25
    )
    command_rows = [
        row
        for row in commands
        if int(first_correction_boundary) <= int(row["boundary"])
        and (end_boundary is None or int(row["boundary"]) < end_boundary)
    ]
    command_keys = {(row["source"], row["inner"]) for row in command_rows}
    physics_rows = [
        row
        for row in physics
        if (row["source"], row["inner"]) in command_keys
    ]
    executed_norms = [_norm(row["executed"]) for row in command_rows]
    measured_integral = sum(_norm(row["measured"]) * PHYSICS_DT_S for row in physics_rows)
    path = 0.0
    complete_intervals = 0
    for command in command_rows:
        rows = sorted(
            (
                row
                for row in physics_rows
                if row["source"] == command["source"]
                and row["inner"] == command["inner"]
                and row["eef"] is not None
            ),
            key=lambda row: int(row["substep"]),
        )
        if len(rows) != SUBSTEPS_PER_UPDATE:
            continue
        previous = command["eef_before"]
        for row in rows:
            path += _distance(previous, row["eef"])
            previous = row["eef"]
        complete_intervals += 1
    return {
        "command_count": len(command_rows),
        "physics_substep_count": len(physics_rows),
        "maximum_correction_norm_rad_s": max(
            (float(row["correction"]) for row in command_rows), default=0.0
        ),
        "filter_correction_integral_rad": sum(
            float(row["correction"]) for row in command_rows
        )
        * CONTROL_DT_S,
        "executed_command_integral_rad": sum(executed_norms) * CONTROL_DT_S,
        "measured_joint_motion_integral_rad": measured_integral,
        "cartesian_path_length_m": path,
        "complete_command_interval_count": complete_intervals,
        "zero_command_fraction": (
            1.0
            if not executed_norms
            else sum(value <= MATERIAL_CORRECTION_MIN_RAD_S for value in executed_norms)
            / len(executed_norms)
        ),
    }


def _validate_pairing(
    audit: _Audit,
    payload: Mapping[str, Any],
    adapter: Mapping[str, Any],
    treatment: Mapping[str, Any],
    adapter_commands: Sequence[Mapping[str, Any]],
    treatment_commands: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    pairing = audit.mapping(payload.get("pairing"), "pairing")
    initial_hash = pairing.get("settled_state_sha256")
    audit.check(_is_sha256(initial_hash), "pair_initial_state_hash_invalid")
    audit.check(
        _is_sha256(pairing.get("historical_table1_settled_state_sha256")),
        "historical_table1_settled_state_hash_invalid",
    )
    adapter_hash = adapter.get("start_official_raw_bytes_sha256")
    treatment_hash = treatment.get("start_official_raw_bytes_sha256")
    state_exact = bool(
        _is_sha256(adapter_hash)
        and adapter_hash == treatment_hash
        and adapter.get("restore", {}).get("settled_state_sha256")
        == treatment.get("restore", {}).get("settled_state_sha256")
        == initial_hash
    )
    audit.check(state_exact, "pair_initial_state_differs")
    audit.check(
        pairing.get("same_initial_state") is state_exact,
        "pair_initial_state_summary_differs",
    )
    cache = audit.mapping(pairing.get("first_query_cache"), "first_query_cache")
    audit.check(
        cache.get("schema_version") == "vlsa_poisson_paired_first_query_cache.v1",
        "first_query_cache_schema_differs",
    )
    audit.check(cache.get("first_query_index") == 0, "first_query_index_differs")
    audit.check(
        cache.get("store_count") == 1 and cache.get("reuse_count") == 1,
        "first_query_store_reuse_count_differs",
    )
    producer = audit.mapping(cache.get("producer"), "first_query_producer")
    reuse = audit.mapping(cache.get("reuse"), "first_query_reuse")
    returned_actions = producer.get("returned_actions")
    returned_hash = _float64_matrix_sha256(
        audit, returned_actions, "first_query_returned_actions"
    )
    audit.check(
        producer.get("returned_actions_sha256") == returned_hash,
        "first_query_returned_actions_hash_differs",
    )
    query_binding_exact = bool(
        producer.get("query_index") == reuse.get("query_index") == 0
        and producer.get("rng_seed") == reuse.get("rng_seed")
        and _is_sha256(producer.get("policy_input_fingerprint_sha256"))
        and producer.get("policy_input_fingerprint_sha256")
        == reuse.get("policy_input_fingerprint_sha256")
        and isinstance(producer.get("producer_arm"), str)
        and isinstance(reuse.get("consumer_arm"), str)
        and producer.get("producer_arm") != reuse.get("consumer_arm")
    )
    audit.check(query_binding_exact, "first_query_binding_differs")
    audit.check(cache.get("contract_valid") is True, "first_query_contract_invalid")

    providers = audit.mapping(payload.get("providers"), "pair_providers")
    baseline_provider = audit.mapping(
        providers.get("adapter_only"), "pair_adapter_provider"
    )
    treatment_provider = audit.mapping(
        providers.get("adapter_plus_psf"), "pair_treatment_provider"
    )
    baseline_queries = audit.sequence(
        baseline_provider.get("policy_queries"), "pair_adapter_policy_queries"
    )
    treatment_queries = audit.sequence(
        treatment_provider.get("policy_queries"), "pair_treatment_policy_queries"
    )
    first_provider_query_exact = False
    if baseline_queries and treatment_queries:
        baseline_query = audit.mapping(
            baseline_queries[0], "pair_adapter_first_query"
        )
        treatment_query = audit.mapping(
            treatment_queries[0], "pair_treatment_first_query"
        )
        first_provider_query_exact = bool(
            baseline_query.get("query_index")
            == treatment_query.get("query_index")
            == 0
            and baseline_query.get("rng_seed")
            == treatment_query.get("rng_seed")
            == producer.get("rng_seed")
            and baseline_query.get("policy_input_fingerprint_sha256")
            == treatment_query.get("policy_input_fingerprint_sha256")
            == producer.get("policy_input_fingerprint_sha256")
            and baseline_query.get("returned_actions_sha256")
            == treatment_query.get("returned_actions_sha256")
            == returned_hash
            and baseline_query.get("returned_actions") == returned_actions
            and treatment_query.get("returned_actions") == returned_actions
            and baseline_query.get("inference_performed") is True
            and treatment_query.get("inference_performed") is False
            and treatment_query.get("paired_cache_source_returned_actions_sha256")
            == returned_hash
        )
    audit.check(first_provider_query_exact, "pair_first_provider_query_differs")

    baseline_actions = audit.sequence(
        baseline_provider.get("high_level_action_trace"),
        "pair_adapter_action_trace",
    )
    treatment_actions = audit.sequence(
        treatment_provider.get("high_level_action_trace"),
        "pair_treatment_action_trace",
    )
    first_action_exact = False
    if (
        baseline_actions
        and treatment_actions
        and adapter_commands
    ):
        baseline_action = audit.mapping(
            baseline_actions[0], "pair_adapter_first_action"
        )
        treatment_action = audit.mapping(
            treatment_actions[0], "pair_treatment_first_action"
        )
        baseline_executed = _vector(
            audit,
            baseline_action.get("aegis_executed"),
            7,
            "pair_adapter_first_aegis_action",
        )
        treatment_executed = _vector(
            audit,
            treatment_action.get("aegis_executed"),
            7,
            "pair_treatment_first_aegis_action",
        )
        first_action_exact = bool(
            baseline_action.get("source_action_index")
            == treatment_action.get("source_action_index")
            == 0
            and baseline_executed
            == treatment_executed
            == adapter_commands[0]["source_action"]
            and (
                not treatment_commands
                or treatment_executed == treatment_commands[0]["source_action"]
            )
        )
    audit.check(first_action_exact, "pair_first_executed_action_differs")
    first_query_exact = bool(
        first_action_exact
        and first_provider_query_exact
        and query_binding_exact
        and cache.get("contract_valid") is True
    )
    audit.check(
        pairing.get("same_first_policy_query") is first_query_exact,
        "pair_first_query_summary_differs",
    )
    return {
        "initial_state_exact": state_exact,
        "first_query_exact": first_query_exact,
    }


def _validate_policy_provider(
    audit: _Audit,
    provider: Mapping[str, Any],
    commands: Sequence[Mapping[str, Any]],
    *,
    label: str,
    terminal_kind: Optional[str] = None,
    expected_horizon_action_count: Optional[int] = None,
) -> Dict[str, Any]:
    if terminal_kind == "method_stop":
        action_count = len(commands) // UPDATES_PER_ACTION + 1
    elif terminal_kind == "contact":
        action_count = 0 if not commands else int(commands[-1]["source"]) + 1
    else:
        action_count = len(commands) // UPDATES_PER_ACTION
    actions = audit.sequence(
        provider.get("high_level_action_trace"), "%s_provider_actions" % label
    )
    queries = audit.sequence(
        provider.get("policy_queries"), "%s_provider_queries" % label
    )
    action_trace_valid = len(actions) == action_count
    for index, value in enumerate(actions):
        row = audit.mapping(value, "%s_provider_action_%d" % (label, index))
        command_for_action = next(
            (row for row in commands if int(row["source"]) == index), None
        )
        executed = _vector(
            audit,
            row.get("aegis_executed"),
            7,
            "%s_provider_action_%d_executed" % (label, index),
        )
        row_valid = bool(
            row.get("local_action_index") == index
            and row.get("source_action_index") == index
            and (command_for_action is None or executed == command_for_action["source_action"])
            and row.get("query_index") == index // 5
            and row.get("query_chunk_offset") == index % 5
            and audit.mapping(
                row.get("aegis_qp"),
                "%s_provider_action_%d_aegis_qp" % (label, index),
            ).get("status")
            == "solved"
        )
        action_trace_valid = action_trace_valid and row_valid
    audit.check(action_trace_valid, "%s_provider_action_trace_differs" % label)

    expected_query_count = (action_count + 4) // 5
    query_trace_valid = len(queries) == expected_query_count
    for index, value in enumerate(queries):
        row = audit.mapping(value, "%s_provider_query_%d" % (label, index))
        returned = row.get("returned_actions")
        returned_hash = _float64_matrix_sha256(
            audit, returned, "%s_provider_query_%d_actions" % (label, index)
        )
        query_trace_valid = bool(
            query_trace_valid
            and row.get("query_index") == index
            and row.get("source_action_index") == index * 5
            and row.get("local_action_index") == index * 5
            and _is_sha256(row.get("policy_input_fingerprint_sha256"))
            and row.get("returned_action_shape") == [10, 7]
            and row.get("returned_actions_sha256") == returned_hash
        )
    audit.check(query_trace_valid, "%s_provider_query_trace_differs" % label)
    return {
        "action_trace_valid": action_trace_valid,
        "query_trace_valid": query_trace_valid,
        "structurally_valid": bool(action_trace_valid and query_trace_valid),
        "complete": bool(
            action_trace_valid
            and query_trace_valid
            and expected_horizon_action_count is not None
            and action_count == expected_horizon_action_count
        ),
        "action_count": action_count,
    }


def _own_observation_chain_valid(
    audit: _Audit,
    provider: Mapping[str, Any],
    arm: Mapping[str, Any],
    *,
    label: str,
) -> bool:
    task = audit.mapping(arm.get("task"), "%s_own_chain_task" % label)
    returned = audit.sequence(
        task.get("returned_observation_sha256_ledger"),
        "%s_own_chain_returned" % label,
    )
    valid = True
    for kind, values in (
        ("action", provider.get("high_level_action_trace")),
        ("query", provider.get("policy_queries")),
    ):
        rows = audit.sequence(values, "%s_own_chain_%s_rows" % (label, kind))
        for index, value in enumerate(rows):
            row = audit.mapping(value, "%s_own_chain_%s_%d" % (label, kind, index))
            local = row.get("local_action_index")
            if isinstance(local, bool) or not isinstance(local, int) or local < 0:
                valid = False
                continue
            if local == 0:
                continue
            prior = local - 1
            if (
                prior >= len(returned)
                or row.get("native_observation_sha256") != returned[prior]
            ):
                valid = False
    return valid


def _validate_video(
    audit: _Audit,
    value: Any,
    *,
    label: str,
    expected_action_count: int,
    completed_action_count: int,
    terminal_kind: Optional[str],
) -> Dict[str, Any]:
    video = audit.mapping(value, "%s_video" % label)
    for key, expected in (
        ("decoded_successfully", True),
        ("real_simulation_frames", True),
        ("two_dimensional_safety_overlay", False),
        ("source_action_index_start", -1),
    ):
        audit.check(video.get(key) == expected, "%s_video_%s_differs" % (label, key))
    expected_frames = (
        expected_action_count + 1
        if terminal_kind is None
        else completed_action_count + 2
    )
    expected_end = (
        expected_action_count - 1
        if terminal_kind is None
        else completed_action_count
    )
    audit.check(
        video.get("source_action_index_end") == expected_end,
        "%s_video_end_action_differs" % label,
    )
    audit.check(
        video.get("frame_count") == expected_frames
        and video.get("decoded_frame_count") == expected_frames,
        "%s_video_frame_count_differs" % label,
    )
    snapshots = audit.sequence(video.get("snapshot_trace"), "%s_video_snapshots" % label)
    audit.check(len(snapshots) == expected_frames, "%s_video_snapshot_count_differs" % label)
    for index, value_row in enumerate(snapshots):
        row = audit.mapping(value_row, "%s_video_snapshot_%d" % (label, index))
        audit.check(row.get("frame_index") == index, "%s_video_snapshot_%d_index_differs" % (label, index))
        audit.check(
            _is_sha256(row.get("source_array_sha256")),
            "%s_video_snapshot_%d_hash_invalid" % (label, index),
        )
        audit.check(
            row.get("height") == 1024 and row.get("width") == 1024,
            "%s_video_snapshot_%d_shape_differs" % (label, index),
        )
    complete = bool(
        terminal_kind is None
        and completed_action_count == expected_action_count
        and video.get("decoded_successfully") is True
        and video.get("real_simulation_frames") is True
        and video.get("two_dimensional_safety_overlay") is False
        and video.get("source_action_index_start") == -1
        and video.get("source_action_index_end") == expected_action_count - 1
        and video.get("frame_count") == expected_action_count + 1
        and video.get("decoded_frame_count") == expected_action_count + 1
    )
    return {"complete": complete, "frame_count": expected_frames}


def _metric_equal(observed: Any, expected: Any) -> bool:
    if isinstance(expected, float):
        return (
            not isinstance(observed, bool)
            and isinstance(observed, (int, float))
            and math.isfinite(float(observed))
            and _close(float(observed), expected)
        )
    return observed == expected


def validate_payload(
    payload_value: Mapping[str, Any],
    *,
    expected_commit: Optional[str] = None,
    expected_job_id: Optional[str] = None,
    result_path: Optional[Path] = None,
    source_path: Optional[Path] = None,
    runner_source_path: Optional[Path] = None,
    arm_runner_source_path: Optional[Path] = None,
    contact_monitor_source_path: Optional[Path] = None,
    cbf_source_path: Optional[Path] = None,
    adapter_source_path: Optional[Path] = None,
    inspect_source: bool = True,
    _test_expected_action_count_override: Optional[int] = None,
) -> Dict[str, Any]:
    audit = _Audit()
    payload = audit.mapping(payload_value, "result")
    audit.check(payload.get("schema_version") == RESULT_SCHEMA, "result_schema_differs")
    audit.check(payload.get("status") == "complete", "result_status_not_complete")
    case_id = payload.get("case_id")
    if _test_expected_action_count_override is None:
        audit.check(payload.get("protocol_id") == PROTOCOL_ID, "protocol_id_differs")
        audit.check(case_id in CASE_HORIZONS, "case_id_not_frozen")
        frozen_action_count = CASE_HORIZONS.get(case_id, 0)
    else:
        frozen_action_count = _test_expected_action_count_override
    audit.check(
        payload.get("partial_output_interpreted") is False,
        "partial_output_interpreted",
    )
    if result_path is not None:
        audit.check(
            payload.get("run_id") == result_path.parent.name,
            "run_id_path_differs",
        )
    forbidden = _scan_forbidden_artifact_semantics(payload)
    audit.check(not forbidden, "artifact_contains_post_osc_torque_semantics")

    providers = _validate_providers(audit, payload.get("providers"))
    source_record: Dict[str, Any] = {}
    if inspect_source:
        source_record = _validate_source(
            audit,
            source_path
            or ROOT / "main/poisson_fullbody/direct_joint_velocity_feasibility.py",
            runner_source_path
            or ROOT / "scripts/run_poisson_direct_joint_velocity_feasibility.py",
            arm_runner_source_path
            or ROOT / "scripts/run_poisson_fast_feasibility.py",
            contact_monitor_source_path
            or ROOT / "main/poisson_fullbody/registered_contact_monitor.py",
            cbf_source_path or ROOT / "main/poisson_fullbody/cbf_qp.py",
            adapter_source_path
            or ROOT / "main/poisson_fullbody/joint_velocity_adapter.py",
            providers,
        )
    _validate_controller_contract(audit, payload.get("controller_contract"))
    field = _validate_field(audit, payload.get("field"))

    provenance = audit.mapping(payload.get("provenance"), "provenance")
    source = audit.mapping(provenance.get("source"), "provenance_source")
    allocation = audit.mapping(provenance.get("allocation"), "provenance_allocation")
    audit.check(source.get("dirty", False) is False, "source_dirty")
    audit.check(_is_git_commit(source.get("commit")), "source_commit_invalid")
    if expected_commit is not None:
        audit.check(source.get("commit") == expected_commit, "source_commit_differs")
    if expected_job_id is not None:
        audit.check(
            str(allocation.get("slurm_job_id")) == str(expected_job_id),
            "allocation_job_id_differs",
        )

    arms = audit.mapping(payload.get("arms"), "arms")
    audit.check(
        set(arms) == {"adapter_only", "adapter_plus_psf"},
        "arm_names_differ",
    )
    adapter = audit.mapping(arms.get("adapter_only"), "adapter_arm")
    treatment = audit.mapping(arms.get("adapter_plus_psf"), "treatment_arm")
    audit.check(
        adapter.get("arm_name") == "joint_velocity_adapter_only",
        "adapter_arm_name_differs",
    )
    audit.check(
        treatment.get("arm_name")
        == "joint_velocity_adapter_plus_link56_psf",
        "treatment_arm_name_differs",
    )
    adapter_terminal = _terminal_kind(audit, adapter, "adapter")
    treatment_terminal = _terminal_kind(audit, treatment, "treatment")
    audit.check(adapter_terminal is None, "adapter_terminated_early")
    registered_action_count = audit.integer(
        audit.mapping(adapter.get("task"), "adapter_task_registration").get(
            "registered_source_action_count"
        ),
        "expected_action_count",
    )
    expected_action_count = int(frozen_action_count)
    audit.check(
        registered_action_count == expected_action_count,
        "registered_action_count_differs_from_frozen_horizon",
    )
    audit.check(expected_action_count > 0, "expected_action_count_not_positive")
    audit.check(
        audit.mapping(treatment.get("task"), "treatment_task_registration").get(
            "registered_source_action_count"
        )
        == expected_action_count,
        "paired_registered_action_count_differs",
    )
    for arm, label in ((adapter, "adapter"), (treatment, "treatment")):
        _validate_restore_and_cadence(audit, arm, label)

    adapter_commands = _validate_command_trace(
        audit,
        adapter,
        label="adapter",
        psf_enabled=False,
        sample_count=field["sample_count"],
    )
    treatment_commands = _validate_command_trace(
        audit,
        treatment,
        label="treatment",
        psf_enabled=True,
        sample_count=field["sample_count"],
        terminal_kind=treatment_terminal,
    )
    adapter_physics = _validate_physics_trace(
        audit, adapter, adapter_commands, label="adapter"
    )
    treatment_physics = _validate_physics_trace(
        audit,
        treatment,
        treatment_commands,
        label="treatment",
        terminal_kind=treatment_terminal,
    )
    pairing = _validate_pairing(
        audit,
        payload,
        adapter,
        treatment,
        adapter_commands,
        treatment_commands,
    )
    provider_records = audit.mapping(payload.get("providers"), "provider_records")
    adapter_provider = _validate_policy_provider(
        audit,
        audit.mapping(
            provider_records.get("adapter_only"), "adapter_policy_provider"
        ),
        adapter_commands,
        label="adapter",
        expected_horizon_action_count=expected_action_count,
    )
    treatment_provider = _validate_policy_provider(
        audit,
        audit.mapping(
            provider_records.get("adapter_plus_psf"),
            "treatment_policy_provider",
        ),
        treatment_commands,
        label="treatment",
        terminal_kind=treatment_terminal,
        expected_horizon_action_count=expected_action_count,
    )
    treatment_provider_raw = audit.mapping(
        provider_records.get("adapter_plus_psf"), "treatment_policy_provider_terminal"
    )
    if treatment_terminal == "method_stop":
        _validate_method_stop(
            audit,
            treatment,
            treatment_commands,
            treatment_provider_raw,
            label="treatment",
        )

    adapter_selected = _selected_contacts(audit, adapter, label="adapter")
    treatment_selected = _selected_contacts(audit, treatment, label="treatment")
    adapter_registered = _validate_registered_contacts(
        audit, adapter, len(adapter_physics), label="adapter"
    )
    treatment_registered = _validate_registered_contacts(
        audit, treatment, len(treatment_physics), label="treatment"
    )
    audit.check(
        adapter_registered["scope_sets"] == treatment_registered["scope_sets"],
        "paired_registered_contact_scopes_differ",
    )
    audit.check(
        adapter_registered["scope_sets"] == field["resolved_sets"],
        "registered_contact_scope_differs_from_resolved_geometry",
    )
    if treatment_terminal == "contact":
        _validate_contact_terminal(
            audit,
            treatment_physics,
            treatment_registered,
            label="treatment",
        )
    elif treatment_terminal == "method_stop":
        audit.check(
            not treatment_selected["any_present"]
            and not treatment_registered["any_forbidden"],
            "treatment_method_stop_after_registered_contact",
        )
    audit.check(
        adapter_selected["any_present"] == adapter_registered["selected_contact"],
        "adapter_contact_monitors_disagree",
    )
    audit.check(
        treatment_selected["any_present"]
        == treatment_registered["selected_contact"],
        "treatment_contact_monitors_disagree",
    )

    adapter_task = _validate_task(audit, adapter, "adapter")
    treatment_task = _validate_task(
        audit, treatment, "treatment", terminal_kind=treatment_terminal
    )
    audit.check(
        adapter_task["completed_action_count"] == expected_action_count
        and len(adapter_commands) == expected_action_count * UPDATES_PER_ACTION,
        "adapter_task_action_exposure_differs",
    )
    expected_treatment_completed = (
        expected_action_count
        if treatment_terminal is None
        else (
            len(treatment_commands) // UPDATES_PER_ACTION
            if treatment_terminal == "method_stop"
            else (
                int(treatment_commands[-1]["source"])
                if treatment_commands
                else 0
            )
        )
    )
    audit.check(
        treatment_task["completed_action_count"]
        == expected_treatment_completed,
        "treatment_task_action_exposure_differs",
    )
    adapter_car = _validate_paper_car(
        audit,
        adapter,
        label="adapter",
        completed_action_count=adapter_task["completed_action_count"],
        expected_action_count=expected_action_count,
    )
    treatment_car = _validate_paper_car(
        audit,
        treatment,
        label="treatment",
        completed_action_count=treatment_task["completed_action_count"],
        expected_action_count=expected_action_count,
        terminal_kind=treatment_terminal,
    )
    videos = audit.mapping(payload.get("videos"), "videos")
    audit.check(set(videos) == {"baseline", "psf"}, "video_arm_names_differ")
    adapter_video = _validate_video(
        audit,
        videos.get("baseline"),
        label="adapter",
        expected_action_count=expected_action_count,
        completed_action_count=adapter_task["completed_action_count"],
        terminal_kind=None,
    )
    treatment_video = _validate_video(
        audit,
        videos.get("psf"),
        label="treatment",
        expected_action_count=expected_action_count,
        completed_action_count=treatment_task["completed_action_count"],
        terminal_kind=treatment_terminal,
    )
    adapter_own_chain = _own_observation_chain_valid(
        audit,
        audit.mapping(provider_records.get("adapter_only"), "adapter_own_provider"),
        adapter,
        label="adapter",
    )
    treatment_own_chain = _own_observation_chain_valid(
        audit,
        treatment_provider_raw,
        treatment,
        label="treatment",
    )
    baseline_contact_boundary = adapter_selected["first_link56_boundary"]
    material_rows = [
        row
        for row in treatment_commands
        if row["correction"] >= MATERIAL_CORRECTION_MIN_RAD_S
        and row["nominal_within_bounds"]
        and row["nominal_residual"] is not None
        and float(row["nominal_residual"]) <= -CBF_TOLERANCE
        and (
            baseline_contact_boundary is None
            or int(row["boundary"]) < int(baseline_contact_boundary)
        )
    ]
    first_correction = material_rows[0]["boundary"] if material_rows else None
    motion = _post_correction_motion(
        treatment_commands,
        treatment_physics,
        first_correction_boundary=first_correction,
        first_success_source_action_index=treatment_task[
            "first_success_source_action_index"
        ],
    )
    useful_motion = bool(
        first_correction is not None
        and motion["measured_joint_motion_integral_rad"]
        >= POST_CORRECTION_JOINT_MOTION_MIN_RAD
        and motion["filter_correction_integral_rad"]
        >= MATERIAL_CORRECTION_INTEGRAL_MIN_RAD
        and motion["cartesian_path_length_m"] >= POST_CORRECTION_EEF_PATH_MIN_M
        and motion["executed_command_integral_rad"]
        >= POST_CORRECTION_EXECUTED_INTEGRAL_MIN_RAD
        and motion["zero_command_fraction"]
        <= POST_CORRECTION_ZERO_COMMAND_FRACTION_MAX
    )
    adapter_tracking = _tracking(
        adapter_physics,
        end_transition_boundary_exclusive=adapter_selected[
            "first_any_boundary"
        ],
    )
    treatment_tracking = _tracking(
        treatment_physics,
    )
    tracking_valid = bool(
        adapter_tracking["valid"]
        and treatment_tracking["valid"]
        and len(treatment_physics)
        == expected_action_count * UPDATES_PER_ACTION * SUBSTEPS_PER_UPDATE
    )

    baseline_contact = bool(
        adapter_selected["link56_present"]
        and adapter_registered["selected_contact"]
        and adapter_selected["first_link56_boundary"]
        == adapter_selected["first_any_boundary"]
    )
    treatment_selected_contact = bool(
        treatment_selected["any_present"]
        or treatment_registered["selected_contact"]
    )
    treatment_shifted_contact = bool(treatment_registered["shifted_contact"])
    treatment_collision_free = not treatment_selected_contact and not treatment_shifted_contact
    correction_before_contact = bool(
        first_correction is not None
        and adapter_selected["first_link56_boundary"] is not None
        and int(first_correction)
        < int(adapter_selected["first_link56_boundary"])
    )
    task_success_after_correction = bool(
        first_correction is not None
        and any(
            (source + 1) * 25 > int(first_correction)
            for source in treatment_task["completed_success_sources"]
        )
    )
    expected_updates = expected_action_count * UPDATES_PER_ACTION
    expected_substeps = expected_updates * SUBSTEPS_PER_UPDATE
    qp_complete = bool(
        treatment.get("qp_solve_count") == expected_updates
        and treatment.get("qp_postcheck_count") == expected_updates
        and treatment.get("joint_limit_postcheck_count") == expected_updates
        and treatment.get("precontact_execution_valid") is True
    )
    adapter_exposure = bool(
        adapter.get("exposure_complete") is True
        and len(adapter_commands) == expected_updates
        and len(adapter_physics) == expected_substeps
        and adapter_task["completed_action_count"] == expected_action_count
    )
    treatment_exposure = bool(
        treatment.get("exposure_complete") is True
        and len(treatment_commands) == expected_updates
        and len(treatment_physics) == expected_substeps
        and treatment_task["completed_action_count"] == expected_action_count
    )
    controller_direct = bool(
        adapter.get("restore", {}).get("controller", {}).get(
            "controller_class_qualname"
        )
        == "JointVelocityController"
        and treatment.get("restore", {}).get("controller", {}).get(
            "controller_class_qualname"
        )
        == "JointVelocityController"
    )
    raw_field = audit.mapping(payload.get("field"), "classification_field")
    fixed_set = bool(
        raw_field.get("protected_robot_body_names")
        == list(PROTECTED_BODY_NAMES)
        and raw_field.get("task_conditioned_link_selection") is False
        and field["sample_count"] > 0
    )
    hard_rows = bool(
        qp_complete
        and treatment.get("pre_filter_field_observation_count") == expected_updates
        and treatment.get("invalid_field_query_count") == 0
        and treatment.get("nonpositive_post_state_field_query_count") == 0
        and all(
            row["raw"].get("safe_cbf_residual_minimum_m2_per_s")
            is not None
            and float(row["raw"]["safe_cbf_residual_minimum_m2_per_s"])
            >= -CBF_TOLERANCE
            for row in treatment_commands
        )
    )
    shared_bounds = bool(
        adapter.get("all_nominal_commands_within_dynamic_joint_bounds") is True
        and treatment.get("all_nominal_commands_within_dynamic_joint_bounds") is True
        and all(row["nominal_within_bounds"] for row in adapter_commands)
        and all(row["nominal_within_bounds"] for row in treatment_commands)
    )
    no_fallback = bool(
        adapter_exposure
        and treatment_exposure
        and provider_records.get("adapter_only", {}).get(
            "recorded_suffix_actions_executed"
        )
        is False
        and provider_records.get("adapter_plus_psf", {}).get(
            "recorded_suffix_actions_executed"
        )
        is False
    )
    contact_checked = bool(
        adapter.get("registered_forbidden_contact", {}).get(
            "observed_physics_substeps"
        )
        == expected_substeps
        and treatment.get("registered_forbidden_contact", {}).get(
            "observed_physics_substeps"
        )
        == expected_substeps
        and all(
            "registered_forbidden_contact_observed" in row["raw"]
            for row in adapter_physics + treatment_physics
        )
    )
    car_complete = bool(
        adapter_car["complete"] and treatment_car["complete"]
    )
    material_correction = bool(
        first_correction is not None
        and correction_before_contact
        and motion["maximum_correction_norm_rad_s"]
        >= MATERIAL_CORRECTION_MIN_RAD_S
        and motion["filter_correction_integral_rad"]
        >= MATERIAL_CORRECTION_INTEGRAL_MIN_RAD
    )
    method_stop = treatment_terminal == "method_stop"
    provider_complete = bool(
        adapter_provider["complete"] and treatment_provider["complete"]
    )
    released_aegis_valid = bool(
        provider_complete
        and provider_records.get("adapter_only", {}).get(
            "first_aegis_input_binding_matches_historical"
        )
        is True
        and provider_records.get("adapter_plus_psf", {}).get(
            "first_aegis_input_binding_matches_historical"
        )
        is True
        and pairing["first_query_exact"]
    )
    metrics = {
        "exact_settled_pair_start": pairing["initial_state_exact"],
        "baseline_exposure_complete": adapter_exposure,
        "treatment_exposure_complete": treatment_exposure,
        "first_live_policy_query_paired": pairing["first_query_exact"],
        "live_policy_contract_valid": bool(
            provider_complete and adapter_video["complete"] and treatment_video["complete"]
        ),
        "released_aegis_contract_valid": released_aegis_valid,
        "own_observation_chains_valid": bool(adapter_own_chain and treatment_own_chain),
        "direct_joint_velocity_controller_used": controller_direct,
        "fixed_link56_protected_set_used": fixed_set,
        "hard_psf_rows_enforced": hard_rows,
        "shared_nominal_joint_bounds_inactive": shared_bounds,
        "joint_velocity_tracking_valid": tracking_valid,
        "no_unregistered_fallback_executed": no_fallback,
        "every_physics_substep_contact_checked": contact_checked,
        "paper_car_endpoint_ledger_complete": car_complete,
        "baseline_link56_contact_present": baseline_contact,
        "treatment_any_robot_selected_obstacle_contact_present": treatment_selected_contact,
        "treatment_link56_shifted_external_contact_present": treatment_shifted_contact,
        "treatment_method_stop": method_stop,
        "treatment_stalled_after_correction": not useful_motion,
        "treatment_native_task_success": treatment_task["success"],
        "treatment_task_success_after_correction": task_success_after_correction,
        "treatment_paper_car_avoided": treatment_car["avoided"],
        "material_correction_present": bool(material_rows),
        "material_correction_before_baseline_contact": bool(
            baseline_contact_boundary is not None and material_rows
        ),
        "maximum_correction_norm_rad_s": motion["maximum_correction_norm_rad_s"],
        "correction_integral_rad": motion["filter_correction_integral_rad"],
        "post_correction_measured_joint_motion_integral_rad": motion[
            "measured_joint_motion_integral_rad"
        ],
        "post_correction_eef_path_length_m": motion["cartesian_path_length_m"],
        "post_correction_executed_command_integral_rad": motion[
            "executed_command_integral_rad"
        ],
        "post_correction_zero_command_fraction": motion["zero_command_fraction"],
    }
    apparatus_valid = all(bool(metrics[key]) for key in APPARATUS_METRIC_KEYS)
    typed_terminal_negative = bool(
        baseline_contact and (method_stop or not treatment_collision_free)
    )
    if not apparatus_valid and not typed_terminal_negative:
        outcome = "INCONCLUSIVE_APPARATUS"
    elif not baseline_contact:
        outcome = "BASELINE_CONTACT_NOT_REPRODUCED"
    elif method_stop:
        outcome = "METHOD_STOP"
    elif not treatment_collision_free:
        outcome = "CONTACT_REMAINS_OR_SHIFTED"
    elif not material_correction:
        outcome = "NO_MATERIAL_CORRECTION"
    elif not useful_motion:
        outcome = "STOP_ONLY"
    elif not treatment_car["avoided"]:
        outcome = "CAR_FAILURE"
    elif not (treatment_task["success"] and task_success_after_correction):
        outcome = "CONTACT_PREVENTED_TASK_FAILED"
    else:
        outcome = "SAFE_TASK_SUCCESS_USEFUL_CORRECTION"
    positive = outcome == "SAFE_TASK_SUCCESS_USEFUL_CORRECTION"

    diagnostics = {
        "protected_sample_count": field["sample_count"],
        "adapter_filter_update_count": len(adapter_commands),
        "treatment_filter_update_count": len(treatment_commands),
        "treatment_qp_solve_count": treatment.get("qp_solve_count"),
        "treatment_qp_postcheck_count": treatment.get("qp_postcheck_count"),
        "treatment_joint_limit_postcheck_count": treatment.get(
            "joint_limit_postcheck_count"
        ),
        "adapter_first_link56_contact_physical_boundary": adapter_selected[
            "first_link56_boundary"
        ],
        "first_material_correction_physical_boundary": first_correction,
        "adapter_precontact_tracking_linf_rad_s": adapter_tracking["linf_rad_s"],
        "adapter_precontact_tracking_rmse_rad_s": adapter_tracking["rmse_rad_s"],
        "treatment_presuccess_tracking_linf_rad_s": treatment_tracking[
            "linf_rad_s"
        ],
        "treatment_presuccess_tracking_rmse_rad_s": treatment_tracking[
            "rmse_rad_s"
        ],
        "treatment_paper_car_maximum_displacement_m": treatment_car[
            "maximum_displacement_m"
        ],
    }
    producer_metrics = audit.mapping(payload.get("metrics"), "producer_metrics")
    audit.check(set(producer_metrics) == set(METRIC_KEYS), "producer_metric_keys_differ")
    for key in METRIC_KEYS:
        expected = metrics[key]
        audit.check(
            _metric_equal(producer_metrics.get(key), expected),
            "producer_metric_%s_differs" % key,
        )

    classification = {
        "schema_version": "vlsa_poisson_direct_joint_velocity_classification.v1",
        "classification": outcome,
        "feasible": positive,
        "pair_complete": apparatus_valid,
        "typed_terminal_negative": typed_terminal_negative,
        "baseline_reproduced": baseline_contact,
        "contact_prevented": treatment_collision_free,
        "material_correction": material_correction,
        "useful_motion": useful_motion,
        "car_safe": treatment_car["avoided"],
        "task_success": bool(
            treatment_task["success"] and task_success_after_correction
        ),
        "method_stop": method_stop,
        "safety_by_stopping": bool(method_stop or not useful_motion),
    }
    producer_classification = audit.mapping(
        payload.get("classification"), "producer_classification"
    )
    for key, expected in classification.items():
        audit.check(
            producer_classification.get(key) == expected,
            "producer_classification_%s_differs" % key,
        )

    valid = not audit.discrepancies
    return {
        "schema_version": SUMMARY_SCHEMA,
        "status": "validated" if valid else "rejected",
        "artifact_valid": valid,
        "run_id": payload.get("run_id"),
        "case_id": payload.get("case_id"),
        "result_payload_sha256": payload.get("result_payload_sha256"),
        "feasible": positive if valid else None,
        "outcome": outcome if valid else None,
        "recomputed_metrics": metrics,
        "recomputed_diagnostics": diagnostics,
        "recomputed_classification": classification,
        "source_audit": source_record,
        "forbidden_artifact_semantics": forbidden,
        "discrepancies": audit.discrepancies,
    }


def validate_artifact(
    result_path: Path,
    *,
    expected_commit: Optional[str] = None,
    expected_job_id: Optional[str] = None,
    source_path: Optional[Path] = None,
    runner_source_path: Optional[Path] = None,
    arm_runner_source_path: Optional[Path] = None,
    contact_monitor_source_path: Optional[Path] = None,
    cbf_source_path: Optional[Path] = None,
    adapter_source_path: Optional[Path] = None,
    inspect_source: bool = True,
) -> Dict[str, Any]:
    path = Path(result_path)
    payload = load_hashed_json(path)
    summary = validate_payload(
        payload,
        expected_commit=expected_commit,
        expected_job_id=expected_job_id,
        result_path=path,
        source_path=source_path,
        runner_source_path=runner_source_path,
        arm_runner_source_path=arm_runner_source_path,
        contact_monitor_source_path=contact_monitor_source_path,
        cbf_source_path=cbf_source_path,
        adapter_source_path=adapter_source_path,
        inspect_source=inspect_source,
    )
    summary["result_file_sha256"] = sha256_file(path)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--expected-commit")
    parser.add_argument("--expected-job-id")
    parser.add_argument("--producer-source", type=Path)
    parser.add_argument("--runner-source", type=Path)
    parser.add_argument("--arm-runner-source", type=Path)
    parser.add_argument("--contact-monitor-source", type=Path)
    parser.add_argument("--cbf-source", type=Path)
    parser.add_argument("--adapter-source", type=Path)
    parser.add_argument(
        "--skip-source-inspection",
        action="store_true",
        help="Fixture-only escape hatch; never use for a claim-bearing artifact.",
    )
    arguments = parser.parse_args()
    try:
        summary = validate_artifact(
            arguments.result,
            expected_commit=arguments.expected_commit,
            expected_job_id=arguments.expected_job_id,
            source_path=arguments.producer_source,
            runner_source_path=arguments.runner_source,
            arm_runner_source_path=arguments.arm_runner_source,
            contact_monitor_source_path=arguments.contact_monitor_source,
            cbf_source_path=arguments.cbf_source,
            adapter_source_path=arguments.adapter_source,
            inspect_source=not arguments.skip_source_inspection,
        )
    except (ArtifactContractError, OSError, TypeError, ValueError) as error:
        summary = {
            "schema_version": SUMMARY_SCHEMA,
            "status": "rejected",
            "artifact_valid": False,
            "discrepancies": ["artifact_load_failed:%s" % error],
        }
    print(json.dumps(summary, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0 if summary.get("artifact_valid") is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
