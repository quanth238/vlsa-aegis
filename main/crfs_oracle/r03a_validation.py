"""Dependency-free semantic validation for R03A strong-analytic case artifacts.

The R03A runner is deliberately separate.  This module validates the immutable
source bindings, pairing, budget, arm accounting, and kill-test semantics
without importing a policy, simulator, NumPy, or the allocation runner.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import struct
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from crfs_harness.artifacts import content_hash


SCHEMA_VERSION = "1.0"
ARTIFACT_TYPE = "r03a_analytic_kill_test_case"
GATE = "R03A"
EXPECTED_CASES = 17
EXPECTED_ARMS = (
    "frozen",
    "analytic_trajectory_mid",
    "analytic_trajectory_early",
)
ANALYTIC_ARMS = EXPECTED_ARMS[1:]
INTERVENTION_STEPS = {
    "analytic_trajectory_mid": 5,
    "analytic_trajectory_early": 1,
}
EXPECTED_ACTIVE_STEPS = {
    "analytic_trajectory_mid": 5,
    "analytic_trajectory_early": 9,
}
REALIZED_CORRECTION_KEYS = {"model_l2", "model_rms", "physical_l2", "physical_rms"}
SOURCE_ARM_KEYS = (
    "frozen",
    "direct_witness",
    "random_residual",
    "analytic_geometry_residual",
    "oracle_residual",
    "bridge_diagnostic",
)
FINAL_STATUSES = {"completed", "nominal_collision_not_reconfirmed"}
EVALUATED_STATUSES = {"passed_gate", "failed_gate"}
EXPLICIT_FAILURE_STATUSES = {
    "bounds_failure",
    "saturation_failure",
    "zero_gradient_failure",
    "nonfinite_failure",
    "budget_failure",
    "policy_failure",
    "terminal_failure",
}
UNRUN_FAILURE_STATUSES = {"policy_failure", "nonfinite_failure"}
EVALUATED_FAILURE_STATUSES = EXPLICIT_FAILURE_STATUSES - UNRUN_FAILURE_STATUSES
PRE_ROLLOUT_FAILURE_STATUSES = {
    "bounds_failure",
    "saturation_failure",
    "budget_failure",
}
ROLLOUT_EVIDENCE_STATUSES = EVALUATED_STATUSES | {
    "zero_gradient_failure",
    "terminal_failure",
}
NOT_EVALUATED_STATUS = "not_evaluated_after_nominal_collision_not_reconfirmed"
ARM_STATUSES = EVALUATED_STATUSES | EXPLICIT_FAILURE_STATUSES | {NOT_EVALUATED_STATUS}

ELIGIBLE_MANIFEST_SHA256 = (
    "241f1b94a5434973dcfb235b43b9386ad15046ff87d35d5b0c34e295c2132916"
)
R03_SUMMARY_SHA256 = (
    "dea3e66c854faa0b659777bfed4b651b19d8d4d0710696ff7bfe52c44d4ee76e"
)
R03_ORDERED_RESULT_SET_DIGEST = (
    "fa79a132f2fecbedab5917111ef71f022e52c933d8e535aaca118add5f5b7895"
)
R02_CONFIG_SHA256 = (
    "c5be1b4759487f4d6541e49110b90fcf5de404bdaed5f389358db0abe4388f4e"
)
DECISION_ARTIFACT = "docs/decisions/0023-run-strong-analytic-kill-test.md"
DECISION_SHA256 = (
    "b1b717856ebee8d3e93a21b35606e62ad6cc4baa2b9346f7bd1f415c84b25ead"
)
BASELINE_COMMIT = "57b1aef306f212aea3574b0a3b64aa1a3d8f5e4b"
CHECKPOINT_SHA256 = (
    "988055ccfd7032903c073a641f3c5f0f0541df444a315116a16f0bf4716d26ed"
)
NORMALIZATION_ASSET_SHA256 = (
    "b3a44bb2810436fb62917decaea58bd4d9110255df527dea21e8fd40c960bd84"
)
BUDGET_DEFINITION = "r02.directions.delta_star_model:first_five_xyz_l2"
PRIVILEGED_REFERENCE_SPS_COUNT = 9
PRIVILEGED_REFERENCE_POPULATION_SIZE = 17
KILL_THRESHOLD_SPS_COUNT = 9
ARTIFACT_SCHEMA_SHA256 = "2f0f9db28ae9d62582e88bc70afe40ef4e37245a60190863e2f3419a135d70c3"
NUMERICAL_AUDIT_RTOL = 2e-6
NUMERICAL_AUDIT_ATOL = 2e-6
TRANSLATION_ACTION_LOW = -1.0
TRANSLATION_ACTION_HIGH = 1.0

TRANSPORT_TOP_LEVEL_CONFIG_KEYS = {"ready_to_run", "blocked_on", "manifest"}
TRANSPORT_R03A_CONFIG_KEYS = {
    "artifact_schema",
    "source_original_manifest",
    "source_r02_config",
    "source_r02_results_root",
    "source_r02_result_filename",
    "source_r03_summary_artifact",
    "decision_artifact",
}

TOP_LEVEL_KEYS = {
    "schema_version",
    "artifact_type",
    "gate",
    "case_id",
    "run_id",
    "status",
    "config_hash",
    "source_evidence",
    "provenance",
    "pairing",
    "budget",
    "arms",
    "outcome",
}
SOURCE_EVIDENCE_KEYS = {
    "r03_summary_path",
    "r03_summary_sha256",
    "r03_ordered_result_set_digest",
    "r02_case_path",
    "r02_case_sha256",
    "r02_case_validator",
    "r02_config_sha256",
    "decision_artifact",
    "decision_sha256",
    "eligible_manifest_sha256",
}
PAIRING_KEYS = {
    "passed",
    "policy_observation",
    "branch_snapshot",
    "branch_geometry",
    "noise",
    "source_frozen_actions",
    "source_frozen_trace",
    "checks",
}
PAIRING_CHECK_KEYS = {
    "case_record_exact_to_source",
    "observation_exact_to_source",
    "branch_snapshot_exact_to_source",
    "geometry_exact_to_source",
    "noise_exact_to_source",
    "fresh_frozen_actions_exact_to_source",
    "fresh_frozen_trace_exact_to_source",
}
BUDGET_KEYS = {
    "definition",
    "source_direction_key",
    "source_delta_star_model",
    "source_array_sha256",
    "first_five_xyz_model_l2",
    "source_reported_full_model_l2",
}
OUTCOME_KEYS = {
    "population",
    "source_arm_gate_pass",
    "fresh_nominal_collision_reproduced",
    "arm_gate_pass",
    "arm_status",
    "privileged_reference_sps_count",
    "privileged_reference_population_size",
    "kill_threshold_sps_count",
    "population_decision",
}
ARM_REQUIRED_KEYS = {
    "mechanism",
    "applicable",
    "status",
    "controls",
    "full_actions",
    "executed_actions",
    "bounds",
    "trace",
    "timing",
    "diagnostics",
    "policy_replay_exact",
    "replay_exact",
    "repeats",
    "gate",
}


def validate_r03a_schema(
    value: Mapping[str, Any],
    *,
    schema_path: Optional[Path] = None,
    require_jsonschema: bool = False,
) -> List[str]:
    """Validate the frozen JSON Schema without importing policy/simulator code."""

    path = schema_path or (
        Path(__file__).resolve().parents[2]
        / "schemas/r03a-analytic-kill-test.schema.json"
    )
    try:
        observed_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        return [f"cannot read frozen R03A schema: {error}"]
    if observed_sha != ARTIFACT_SCHEMA_SHA256:
        return ["frozen R03A schema content hash differs"]
    try:
        import jsonschema
    except ImportError:
        if require_jsonschema:
            return ["jsonschema is required for production R03A artifact validation"]
        return []
    try:
        schema = json.loads(path.read_text(encoding="utf-8"))
        validator = jsonschema.Draft202012Validator(schema)
    except (OSError, json.JSONDecodeError, jsonschema.SchemaError) as error:
        return [f"cannot load frozen R03A schema: {error}"]
    return [
        "schema: " + error.message
        for error in sorted(
            validator.iter_errors(value),
            key=lambda item: tuple(str(part) for part in item.absolute_path),
        )
    ]


def normalized_r03a_config(value: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the scientific R03A identity with transport paths removed."""

    if not isinstance(value, Mapping):
        raise TypeError("R03A config must be an object")
    normalized = json.loads(json.dumps(value))
    for key in TRANSPORT_TOP_LEVEL_CONFIG_KEYS:
        normalized.pop(key, None)
    settings = normalized.get("r03a")
    if not isinstance(settings, dict):
        raise ValueError("R03A config must contain an r03a object")
    for key in TRANSPORT_R03A_CONFIG_KEYS:
        settings.pop(key, None)
    return normalized


def r03a_config_hash(value: Mapping[str, Any]) -> str:
    """Content hash for the frozen scientific R03A configuration."""

    return content_hash(normalized_r03a_config(value))
ANALYTIC_DIAGNOSTIC_KEYS = {
    "budget_model_l2",
    "integrated_field_model_l2",
    "active_step_count",
    "applied_step_count",
    "margin_stop_step_count",
    "zero_gradient_step_count",
    "nonfinite_gradient_step_count",
    "d_opt_trace",
    "realized_final_correction",
}


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _is_git_oid(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) in {40, 64}
        and re.fullmatch(r"[0-9a-f]+", value) is not None
    )


def _finite(value: Any, *, nonnegative: bool = False, positive: bool = False) -> bool:
    if isinstance(value, bool):
        return False
    try:
        result = float(value)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(result):
        return False
    if positive and result <= 0.0:
        return False
    return not (nonnegative and result < 0.0)


def _nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _shape_and_values(value: Any) -> Tuple[Tuple[int, ...], List[float]]:
    if not isinstance(value, list):
        if not _finite(value):
            raise ValueError("array value must be finite numeric")
        return (), [float(value)]
    if not value:
        return (0,), []
    child = [_shape_and_values(item) for item in value]
    first_shape = child[0][0]
    if any(shape != first_shape for shape, _flat in child[1:]):
        raise ValueError("array values must be rectangular")
    flattened: List[float] = []
    for _shape, flat in child:
        flattened.extend(flat)
    return (len(value),) + first_shape, flattened


def _array_hash(dtype: str, shape: Tuple[int, ...], flat: Sequence[float]) -> str:
    formats = {"float32": "<f", "float64": "<d"}
    if dtype not in formats:
        raise ValueError(f"unsupported dependency-free array dtype {dtype!r}")
    digest = hashlib.sha256()
    digest.update(dtype.encode("utf-8"))
    digest.update(str(shape).encode("utf-8"))
    for item in flat:
        digest.update(struct.pack(formats[dtype], float(item)))
    return digest.hexdigest()


def _validate_array_record(
    value: Any,
    *,
    name: str,
    shape: Optional[Tuple[int, ...]] = None,
) -> Tuple[Optional[List[float]], List[str]]:
    if not isinstance(value, Mapping):
        return None, [f"{name} must be an array record"]
    errors: List[str] = []
    if set(value) != {"dtype", "shape", "sha256", "values"}:
        errors.append(f"{name} has incorrect array-record fields")
    dtype = value.get("dtype")
    if dtype not in {"float32", "float64"}:
        errors.append(f"{name}.dtype must be float32 or float64")
    try:
        actual_shape, flat = _shape_and_values(value.get("values"))
    except (TypeError, ValueError) as error:
        return None, errors + [f"{name}.values are invalid: {error}"]
    if value.get("shape") != list(actual_shape):
        errors.append(f"{name}.shape conflicts with values")
    if shape is not None and actual_shape != shape:
        errors.append(f"{name} must have shape {shape}, got {actual_shape}")
    if isinstance(dtype, str) and dtype in {"float32", "float64"}:
        try:
            expected_hash = _array_hash(dtype, actual_shape, flat)
        except (ValueError, OverflowError) as error:
            errors.append(f"{name} cannot be hashed: {error}")
        else:
            if value.get("sha256") != expected_hash:
                errors.append(f"{name}.sha256 conflicts with dtype/shape/values")
    elif not _is_sha256(value.get("sha256")):
        errors.append(f"{name}.sha256 must be SHA-256")
    return flat, errors


def _validate_hash_binding(value: Any, *, name: str) -> List[str]:
    if not isinstance(value, Mapping) or not {"source_sha256", "fresh_sha256"}.issubset(value):
        return [f"pairing.{name} must contain source_sha256/fresh_sha256 fields"]
    source = value.get("source_sha256")
    fresh = value.get("fresh_sha256")
    errors: List[str] = []
    if not _is_sha256(source) or not _is_sha256(fresh):
        errors.append(f"pairing.{name} hashes must be SHA-256")
    if source != fresh:
        errors.append(f"pairing.{name} source/fresh hashes differ")
    return errors


def _inverted_cdf(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(item) for item in values)
    index = max(0, min(len(ordered) - 1, math.ceil(probability * len(ordered)) - 1))
    return ordered[index]


def _validate_timing_series(
    value: Any,
    *,
    name: str,
    allow_empty: bool,
    expected_count: Optional[int] = None,
) -> List[str]:
    if not isinstance(value, Mapping) or set(value) != {"values", "p50", "p95"}:
        return [f"{name} must contain exact values/p50/p95 fields"]
    raw = value.get("values")
    errors: List[str] = []
    if not isinstance(raw, list) or any(not _finite(item, nonnegative=True) for item in (raw or [])):
        return [f"{name}.values must contain finite nonnegative seconds"]
    if expected_count is not None and len(raw) != expected_count:
        errors.append(f"{name}.values must contain exactly {expected_count} measurements")
    if not raw:
        if not allow_empty:
            errors.append(f"{name}.values must be nonempty")
        if value.get("p50") is not None or value.get("p95") is not None:
            errors.append(f"{name} empty series must have null p50/p95")
        return errors
    for key, probability in (("p50", 0.50), ("p95", 0.95)):
        if not _finite(value.get(key), nonnegative=True):
            errors.append(f"{name}.{key} must be finite nonnegative")
        elif not math.isclose(
            float(value[key]), _inverted_cdf(raw, probability), rel_tol=0.0, abs_tol=1e-15
        ):
            errors.append(f"{name}.{key} conflicts with inverted-CDF raw values")
    return errors


def _validate_timing(
    value: Any,
    *,
    name: str,
    analytic: bool,
    evaluated: bool = True,
    expected_gradient_count: Optional[int] = None,
) -> List[str]:
    if not isinstance(value, Mapping):
        return [f"{name}.timing must be an object"]
    required = {
        "descriptive_only",
        "warmed_batch_one",
        "timer_scope",
        "quantile_method",
        "policy_seconds",
        "analytic_gradient_seconds",
    }
    errors: List[str] = []
    if not required.issubset(value):
        errors.append(f"{name}.timing is missing required fields")
        return errors
    if value.get("descriptive_only") is not True:
        errors.append(f"{name}.timing must remain descriptive only")
    if evaluated and value.get("warmed_batch_one") is not True:
        errors.append(f"{name}.timing evaluated arm must use warmed batch-one timing")
    if not evaluated and value.get("warmed_batch_one") is not False:
        errors.append(f"{name}.timing unrun arm must record warmed_batch_one false")
    if value.get("timer_scope") != "client_round_trip_batch_one_infer_including_analytic_field":
        errors.append(f"{name}.timing timer scope differs from the frozen protocol")
    if value.get("quantile_method") != "inverted_cdf":
        errors.append(f"{name}.timing quantile method must be inverted_cdf")
    policy = value.get("policy_seconds")
    gradient = value.get("analytic_gradient_seconds")
    errors.extend(
        _validate_timing_series(
            policy,
            name=f"{name}.timing.policy_seconds",
            allow_empty=not evaluated,
            expected_count=2 if evaluated else 0,
        )
    )
    errors.extend(
        _validate_timing_series(
            gradient,
            name=f"{name}.timing.analytic_gradient_seconds",
            allow_empty=not (analytic and evaluated),
            expected_count=(
                expected_gradient_count if analytic and evaluated else 0
            ),
        )
    )
    if (not analytic or not evaluated) and isinstance(gradient, Mapping) and gradient.get("values"):
        errors.append(f"{name}.timing frozen/unrun arm cannot report analytic-gradient timing")
    return errors


def _validate_arm(name: str, value: Any, *, terminal: bool, budget_l2: float) -> List[str]:
    if not isinstance(value, Mapping):
        return [f"arms.{name} must be an object"]
    errors: List[str] = []
    missing = ARM_REQUIRED_KEYS - set(value)
    if missing:
        errors.append(f"arms.{name} missing required fields: {sorted(missing)}")
        return errors
    status = value.get("status")
    if status not in ARM_STATUSES:
        errors.append(f"arms.{name}.status is invalid")
    analytic = name in ANALYTIC_ARMS
    full_is_null = value.get("full_actions") is None
    executed_is_null = value.get("executed_actions") is None
    if full_is_null != executed_is_null:
        errors.append(f"arms.{name} cannot retain only one of full/executed actions")
    has_actions = not full_is_null and not executed_is_null
    unrun_failure = analytic and status in UNRUN_FAILURE_STATUSES
    applicable = value.get("applicable")
    if applicable is not True:
        errors.append(f"arms.{name} must remain applicable on every eligible R03A case")
    controls = value.get("controls")
    if not isinstance(controls, Mapping):
        errors.append(f"arms.{name}.controls must be an object")
    elif analytic:
        if controls.get("intervention_mode") != "analytic_trajectory_field":
            errors.append(f"arms.{name} must use analytic_trajectory_field mode")
        if controls.get("intervention_step") != INTERVENTION_STEPS[name]:
            errors.append(f"arms.{name} intervention step differs from ADR-0023")
        if controls.get("clipping_policy") != "fail_without_clipping":
            errors.append(f"arms.{name} must fail without clipping")
    elif controls.get("intervention_mode") != "none":
        errors.append("arms.frozen must have no intervention")

    errors.extend(
        _validate_timing(
            value.get("timing"),
            name=f"arms.{name}",
            analytic=analytic,
            evaluated=not (terminal and analytic) and not unrun_failure,
            expected_gradient_count=(
                2 * EXPECTED_ACTIVE_STEPS[name] if analytic else 0
            ),
        )
    )
    bounds = value.get("bounds")
    if not isinstance(bounds, Mapping) or bounds.get("clipped") is not False:
        errors.append(f"arms.{name}.bounds must explicitly record no clipping")
    gate = value.get("gate")
    if not isinstance(gate, Mapping) or gate.get("passed") not in {True, False, None}:
        errors.append(f"arms.{name}.gate must contain a tri-state passed field")
    elif status == "passed_gate" and gate.get("passed") is not True:
        errors.append(f"arms.{name} passed status conflicts with gate")
    elif status in (EVALUATED_STATUSES - {"passed_gate"}) | EXPLICIT_FAILURE_STATUSES and gate.get("passed") is not False:
        errors.append(f"arms.{name} failure status conflicts with gate")
    elif status == NOT_EVALUATED_STATUS and gate.get("passed") is not None:
        errors.append(f"arms.{name} not-evaluated status conflicts with gate")

    if terminal and analytic:
        if status != NOT_EVALUATED_STATUS:
            errors.append(f"arms.{name} must be explicit not-evaluated after nominal mismatch")
        for field in ("full_actions", "executed_actions", "trace"):
            if value.get(field) is not None:
                errors.append(f"arms.{name}.{field} must be null after nominal mismatch")
        if value.get("repeats") != []:
            errors.append(f"arms.{name}.repeats must be empty after nominal mismatch")
        return errors

    failure_reason = value.get("failure_reason")
    if failure_reason is not None and (
        not isinstance(failure_reason, str) or not failure_reason.strip()
    ):
        errors.append(f"arms.{name}.failure_reason must be a nonempty string when present")

    if unrun_failure:
        if not isinstance(failure_reason, str) or not failure_reason.strip():
            errors.append(f"arms.{name} unrun failure must retain failure_reason")
        for field in ("full_actions", "executed_actions", "trace", "diagnostics"):
            if value.get(field) is not None:
                errors.append(f"arms.{name}.{field} must be null for a pre-reply failure")
        if value.get("policy_replay_exact") is not None:
            errors.append(f"arms.{name}.policy_replay_exact must be null before a reply")
        if value.get("replay_exact") is not None:
            errors.append(f"arms.{name}.replay_exact must be null before a rollout")
        if value.get("repeats") != []:
            errors.append(f"arms.{name}.repeats must be empty before a rollout")
        if has_actions:
            errors.append(f"arms.{name} pre-reply failure cannot retain sampled actions")
        return errors

    if analytic and status in UNRUN_FAILURE_STATUSES:
        # Kept separate from `unrun_failure` for type-checker/readability: these
        # statuses are never allowed to masquerade as evaluated evidence.
        errors.append(f"arms.{name} policy/nonfinite failures must occur before a reply")

    executed_flat: Optional[List[float]] = None
    out_of_bounds = False
    exactly_saturated = False
    if status in EVALUATED_STATUSES | EVALUATED_FAILURE_STATUSES or not analytic:
        _full, item_errors = _validate_array_record(
            value.get("full_actions"), name=f"arms.{name}.full_actions", shape=(10, 7)
        )
        errors.extend(item_errors)
        executed_flat, item_errors = _validate_array_record(
            value.get("executed_actions"), name=f"arms.{name}.executed_actions", shape=(5, 7)
        )
        errors.extend(item_errors)
        if not isinstance(value.get("trace"), Mapping) or not value.get("trace"):
            errors.append(f"arms.{name}.trace must retain the complete policy trace")
        if value.get("policy_replay_exact") is not True:
            errors.append(f"arms.{name} policy duplicate must be exact")
        repeats = value.get("repeats")
        if status in ROLLOUT_EVIDENCE_STATUSES:
            if value.get("replay_exact") is not True:
                errors.append(f"arms.{name} simulator repeats must be exact")
            if not isinstance(repeats, list) or len(repeats) != 2:
                errors.append(f"arms.{name} must retain exactly two simulator repeats")
            elif any(
                not isinstance(repeat, Mapping)
                or not isinstance(repeat.get("task_success"), bool)
                or not isinstance(repeat.get("task_success_during_prefix"), bool)
                for repeat in repeats
            ):
                errors.append(
                    f"arms.{name} repeats must retain Boolean task_success and task_success_during_prefix"
                )
            elif any(
                repeat.get("task_success") is True
                and repeat.get("task_success_during_prefix") is not True
                for repeat in repeats
            ):
                errors.append(
                    f"arms.{name} final task_success implies task_success_during_prefix"
                )
        elif status in PRE_ROLLOUT_FAILURE_STATUSES:
            if value.get("replay_exact") is not None:
                errors.append(f"arms.{name} pre-rollout failure must have null replay_exact")
            if repeats != []:
                errors.append(f"arms.{name} pre-rollout failure must have no simulator repeats")

    if analytic and executed_flat is not None and len(executed_flat) == 35:
        xyz = [executed_flat[row * 7 + column] for row in range(5) for column in range(3)]
        out_of_bounds = any(
            item < TRANSLATION_ACTION_LOW or item > TRANSLATION_ACTION_HIGH
            for item in xyz
        )
        exactly_saturated = any(
            item == TRANSLATION_ACTION_LOW or item == TRANSLATION_ACTION_HIGH
            for item in xyz
        )

    repeats = value.get("repeats")
    task_success = bool(
        isinstance(repeats, list)
        and any(
            isinstance(repeat, Mapping)
            and repeat.get("task_success_during_prefix") is True
            for repeat in repeats
        )
    )
    diagnostics = value.get("diagnostics")
    materially_above_budget = False
    zero_gradient_count = 0
    if analytic:
        if not isinstance(diagnostics, Mapping):
            errors.append(f"arms.{name}.diagnostics must be an object")
        else:
            missing_diagnostics = ANALYTIC_DIAGNOSTIC_KEYS - set(diagnostics)
            if missing_diagnostics:
                errors.append(
                    f"arms.{name}.diagnostics missing fields: {sorted(missing_diagnostics)}"
                )
            if not _finite(diagnostics.get("budget_model_l2"), positive=True):
                errors.append(f"arms.{name}.diagnostics budget must be positive")
            elif not math.isclose(
                float(diagnostics["budget_model_l2"]),
                budget_l2,
                rel_tol=NUMERICAL_AUDIT_RTOL,
                abs_tol=NUMERICAL_AUDIT_ATOL,
            ):
                errors.append(f"arms.{name}.diagnostics budget differs from immutable source")
            if not _finite(
                diagnostics.get("integrated_field_model_l2"), nonnegative=True
            ):
                errors.append(
                    f"arms.{name}.diagnostics.integrated_field_model_l2 must be finite nonnegative"
                )
            for field in (
                "active_step_count",
                "applied_step_count",
                "margin_stop_step_count",
                "zero_gradient_step_count",
                "nonfinite_gradient_step_count",
            ):
                if not _nonnegative_int(diagnostics.get(field)):
                    errors.append(f"arms.{name}.diagnostics.{field} must be a nonnegative integer")
            if not isinstance(diagnostics.get("d_opt_trace"), list):
                errors.append(f"arms.{name}.diagnostics.d_opt_trace must be a list")
            if not isinstance(diagnostics.get("realized_final_correction"), Mapping):
                errors.append(f"arms.{name}.diagnostics.realized_final_correction must be an object")
            else:
                correction = diagnostics["realized_final_correction"]
                if set(correction) != REALIZED_CORRECTION_KEYS:
                    errors.append(
                        f"arms.{name}.diagnostics.realized_final_correction has incorrect fields"
                    )
                elif any(not _finite(correction.get(key), nonnegative=True) for key in REALIZED_CORRECTION_KEYS):
                    errors.append(
                        f"arms.{name}.diagnostics.realized_final_correction values must be finite nonnegative"
                    )
            active_steps = diagnostics.get("active_step_count")
            if active_steps != EXPECTED_ACTIVE_STEPS[name]:
                errors.append(f"arms.{name}.diagnostics active-step count differs from protocol")
            applied = diagnostics.get("applied_step_count")
            stopped = diagnostics.get("margin_stop_step_count")
            zero_gradient = diagnostics.get("zero_gradient_step_count")
            nonfinite_gradient = diagnostics.get("nonfinite_gradient_step_count")
            counts = (active_steps, applied, stopped, zero_gradient, nonfinite_gradient)
            if all(_nonnegative_int(item) for item in counts):
                if applied + stopped + zero_gradient + nonfinite_gradient != active_steps:
                    errors.append(
                        f"arms.{name}.diagnostics applied+margin-stop+zero-gradient+"
                        "nonfinite-gradient counts must equal active steps"
                    )
            integrated = diagnostics.get("integrated_field_model_l2")
            if _finite(integrated, nonnegative=True) and math.isfinite(budget_l2):
                integrated_value = float(integrated)
                materially_above_budget = integrated_value > budget_l2 and not math.isclose(
                    integrated_value,
                    budget_l2,
                    rel_tol=NUMERICAL_AUDIT_RTOL,
                    abs_tol=NUMERICAL_AUDIT_ATOL,
                )
            zero_count = diagnostics.get("zero_gradient_step_count")
            nonfinite_count = diagnostics.get("nonfinite_gradient_step_count")
            if _nonnegative_int(zero_count):
                zero_gradient_count = int(zero_count)
            if _nonnegative_int(nonfinite_count) and nonfinite_count > 0:
                errors.append(f"arms.{name} evaluated evidence cannot retain a nonfinite gradient")
    elif diagnostics not in ({}, None):
        # Frozen diagnostics may retain baseline-only telemetry, but must never
        # masquerade as an analytic field record.
        if isinstance(diagnostics, Mapping) and any(
            key in diagnostics for key in ANALYTIC_DIAGNOSTIC_KEYS
        ):
            errors.append("arms.frozen cannot contain analytic diagnostics")

    if analytic:
        if out_of_bounds:
            expected_status: Any = "bounds_failure"
        elif exactly_saturated:
            expected_status = "saturation_failure"
        elif materially_above_budget:
            expected_status = "budget_failure"
        elif task_success:
            expected_status = "terminal_failure"
        elif zero_gradient_count > 0:
            expected_status = "zero_gradient_failure"
        else:
            expected_status = EVALUATED_STATUSES
        if isinstance(expected_status, str) and status != expected_status:
            errors.append(
                f"arms.{name} evaluated failure precedence requires {expected_status}"
            )
        elif isinstance(expected_status, set) and status not in expected_status:
            errors.append(f"arms.{name} has an explicit failure status without its cause")
    return errors


def validate_r03a_result(value: Mapping[str, Any]) -> List[str]:
    """Return all allocation-independent semantic errors for one final case."""

    if not isinstance(value, Mapping):
        return ["result must be an object"]
    errors: List[str] = []
    if set(value) != TOP_LEVEL_KEYS:
        errors.append("result has incorrect top-level fields")
    if value.get("schema_version") != SCHEMA_VERSION:
        errors.append("schema_version must be 1.0")
    if value.get("artifact_type") != ARTIFACT_TYPE or value.get("gate") != GATE:
        errors.append("result must be an R03A analytic kill-test case")
    if not isinstance(value.get("case_id"), str) or not value.get("case_id"):
        errors.append("case_id must be nonempty")
    if not isinstance(value.get("run_id"), str) or not value.get("run_id"):
        errors.append("run_id must be nonempty")
    if value.get("status") not in FINAL_STATUSES:
        errors.append("status must be a registered final R03A status")
    if not _is_sha256(value.get("config_hash")):
        errors.append("config_hash must be SHA-256")

    source = value.get("source_evidence")
    if not isinstance(source, Mapping) or set(source) != SOURCE_EVIDENCE_KEYS:
        errors.append("source_evidence has incorrect fields")
    else:
        expected = {
            "r03_summary_sha256": R03_SUMMARY_SHA256,
            "r03_ordered_result_set_digest": R03_ORDERED_RESULT_SET_DIGEST,
            "r02_config_sha256": R02_CONFIG_SHA256,
            "decision_artifact": DECISION_ARTIFACT,
            "decision_sha256": DECISION_SHA256,
            "eligible_manifest_sha256": ELIGIBLE_MANIFEST_SHA256,
            "r02_case_validator": "validate_r02_result:passed",
        }
        for key, expected_value in expected.items():
            if source.get(key) != expected_value:
                errors.append(f"source_evidence.{key} differs from the accepted source")
        if not isinstance(source.get("r03_summary_path"), str) or not source.get("r03_summary_path"):
            errors.append("source_evidence.r03_summary_path must be nonempty")
        if not isinstance(source.get("r02_case_path"), str) or not source.get("r02_case_path"):
            errors.append("source_evidence.r02_case_path must be nonempty")
        if not _is_sha256(source.get("r02_case_sha256")):
            errors.append("source_evidence.r02_case_sha256 must be SHA-256")

    provenance = value.get("provenance")
    required_provenance = {
        "evidence_tier",
        "git_commit",
        "git_dirty",
        "baseline_commit",
        "host",
        "device",
        "slurm_job_id",
        "slurm_array_job_id",
        "slurm_array_task_id",
        "partition",
        "case_record",
        "input_manifest_sha256",
        "config_file_sha256",
        "checkpoint_id",
        "checkpoint_sha256",
    }
    if not isinstance(provenance, Mapping) or not required_provenance.issubset(provenance):
        errors.append("provenance is missing required allocation/source fields")
    else:
        if not _is_git_oid(provenance.get("git_commit")):
            errors.append("provenance.git_commit must be an immutable git oid")
        if not isinstance(provenance.get("git_dirty"), bool):
            errors.append("provenance.git_dirty must be Boolean")
        if provenance.get("baseline_commit") != BASELINE_COMMIT:
            errors.append("provenance baseline revision differs")
        if provenance.get("input_manifest_sha256") != ELIGIBLE_MANIFEST_SHA256:
            errors.append("provenance input manifest differs")
        if not _is_sha256(provenance.get("config_file_sha256")):
            errors.append("provenance.config_file_sha256 must be SHA-256")
        if provenance.get("checkpoint_sha256") != CHECKPOINT_SHA256:
            errors.append("provenance checkpoint differs")
        case = provenance.get("case_record")
        if not isinstance(case, Mapping) or case.get("case_id") != value.get("case_id"):
            errors.append("provenance.case_record differs from result case")

    pairing = value.get("pairing")
    if not isinstance(pairing, Mapping) or set(pairing) != PAIRING_KEYS:
        errors.append("pairing has incorrect fields")
    else:
        for name in (
            "policy_observation",
            "branch_snapshot",
            "branch_geometry",
            "noise",
            "source_frozen_actions",
            "source_frozen_trace",
        ):
            errors.extend(_validate_hash_binding(pairing.get(name), name=name))
        checks = pairing.get("checks")
        if not isinstance(checks, Mapping) or set(checks) != PAIRING_CHECK_KEYS:
            errors.append("pairing.checks has incorrect fields")
        elif any(checks.get(key) is not True for key in PAIRING_CHECK_KEYS):
            errors.append("every R03A pairing check must pass")
        if pairing.get("passed") is not True:
            errors.append("R03A pairing must pass before a trusted final artifact")

    budget = value.get("budget")
    budget_l2 = float("nan")
    if not isinstance(budget, Mapping) or set(budget) != BUDGET_KEYS:
        errors.append("budget has incorrect fields")
    else:
        if budget.get("definition") != BUDGET_DEFINITION:
            errors.append("budget.definition differs from ADR-0023")
        if budget.get("source_direction_key") != "delta_star_model":
            errors.append("budget.source_direction_key must be delta_star_model")
        flat, item_errors = _validate_array_record(
            budget.get("source_delta_star_model"),
            name="budget.source_delta_star_model",
            shape=(10, 32),
        )
        errors.extend(item_errors)
        record = budget.get("source_delta_star_model")
        if isinstance(record, Mapping) and budget.get("source_array_sha256") != record.get("sha256"):
            errors.append("budget.source_array_sha256 differs from source array record")
        if flat is not None and len(flat) == 320:
            budget_l2 = math.sqrt(
                sum(flat[row * 32 + column] ** 2 for row in range(5) for column in range(3))
            )
        if not _finite(budget.get("first_five_xyz_model_l2"), positive=True):
            errors.append("budget.first_five_xyz_model_l2 must be positive")
        elif math.isfinite(budget_l2) and not math.isclose(
            float(budget["first_five_xyz_model_l2"]), budget_l2, rel_tol=0.0, abs_tol=1e-12
        ):
            errors.append("budget first-five XYZ L2 conflicts with source array")
        if not _finite(budget.get("source_reported_full_model_l2"), positive=True):
            errors.append("budget.source_reported_full_model_l2 must be positive")
        elif _finite(budget.get("first_five_xyz_model_l2"), positive=True) and not math.isclose(
            float(budget["source_reported_full_model_l2"]),
            float(budget["first_five_xyz_model_l2"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            errors.append("source full model L2 differs from first-five XYZ budget")

    arms = value.get("arms")
    terminal = value.get("status") == "nominal_collision_not_reconfirmed"
    if not isinstance(arms, Mapping) or set(arms) != set(EXPECTED_ARMS):
        errors.append("arms must contain the exact registered set")
    else:
        for name in EXPECTED_ARMS:
            errors.extend(_validate_arm(name, arms[name], terminal=terminal, budget_l2=budget_l2))

    outcome = value.get("outcome")
    if not isinstance(outcome, Mapping) or set(outcome) != OUTCOME_KEYS:
        errors.append("outcome has incorrect fields")
    else:
        if outcome.get("population") != "r03_witness_confirmed_development_diagnostic":
            errors.append("outcome population differs from the registered diagnostic")
        source_pass = outcome.get("source_arm_gate_pass")
        if not isinstance(source_pass, Mapping) or set(source_pass) != set(SOURCE_ARM_KEYS):
            errors.append("outcome.source_arm_gate_pass must retain every source R02 arm")
        elif source_pass.get("direct_witness") is not True:
            errors.append("eligible R03A source must retain a passing direct witness")
        fresh_collision = outcome.get("fresh_nominal_collision_reproduced")
        if fresh_collision is not (not terminal):
            errors.append("fresh nominal collision conflicts with final status")
        arm_pass = outcome.get("arm_gate_pass")
        arm_status = outcome.get("arm_status")
        if not isinstance(arm_pass, Mapping) or set(arm_pass) != set(EXPECTED_ARMS):
            errors.append("outcome.arm_gate_pass has incorrect arm set")
        if not isinstance(arm_status, Mapping) or set(arm_status) != set(EXPECTED_ARMS):
            errors.append("outcome.arm_status has incorrect arm set")
        if isinstance(arms, Mapping) and set(arms) == set(EXPECTED_ARMS):
            for name in EXPECTED_ARMS:
                if isinstance(arm_status, Mapping) and arm_status.get(name) != arms[name].get("status"):
                    errors.append(f"outcome arm status differs for {name}")
                expected_pass = arms[name].get("gate", {}).get("passed")
                if isinstance(arm_pass, Mapping) and arm_pass.get(name) is not expected_pass:
                    errors.append(f"outcome arm pass differs for {name}")
        if outcome.get("privileged_reference_sps_count") != PRIVILEGED_REFERENCE_SPS_COUNT:
            errors.append("outcome privileged reference count must remain 9")
        if outcome.get("privileged_reference_population_size") != PRIVILEGED_REFERENCE_POPULATION_SIZE:
            errors.append("outcome privileged reference population must remain 17")
        if outcome.get("kill_threshold_sps_count") != KILL_THRESHOLD_SPS_COUNT:
            errors.append("outcome kill threshold must remain 9")
        if outcome.get("population_decision") != "deferred_to_population_summary":
            errors.append("per-case artifact cannot make the population decision")
    return errors


__all__ = [
    "ANALYTIC_ARMS",
    "ARTIFACT_SCHEMA_SHA256",
    "ARTIFACT_TYPE",
    "EXPECTED_ARMS",
    "EXPECTED_CASES",
    "GATE",
    "normalized_r03a_config",
    "r03a_config_hash",
    "validate_r03a_result",
    "validate_r03a_schema",
]
