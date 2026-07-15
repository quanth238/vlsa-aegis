"""Independent semantic validation for CFS-00A raw and final artifacts.

The GPU/client payload contains useful summaries, but no ``passed`` boolean is
trusted here.  Arrays are reconstructed from their byte hashes, the frozen
linear algebra is rerun through the NumPy audit in the canary module, every
selected schedule is checked against its canonical recurrence, and the
three-way outcome is recomputed.  A terminal apparatus payload is accepted
only as ``apparatus_inconclusive`` and never promoted to a method result.
"""

from __future__ import annotations

from collections.abc import Mapping
import copy
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from crfs_harness.artifacts import content_hash, file_sha256

from .r02_runner import _array_from_record, _validate_array_record
from .r05a_canary import (
    CASE_ID,
    PAYLOAD_TYPE as LEGACY_PAYLOAD_TYPE,
    SOLVER_CONFIG,
    _artifact_replay_checks,
    _array_exact,
    _schedule_diagnostics,
    _trace_from_record,
    validate_flow_recurrence,
)
from . import r05a_constrained_flow_canary as cfs


SUCCESS_KEYS = {
    "schema_version",
    "payload_type",
    "payload_variant",
    "status",
    "case_id",
    "run_id",
    "legacy_payload_path",
    "legacy_payload_sha256",
    "config",
    "source_pairing",
    "arms",
    "duplicates",
    "canonical_replays",
    "apparatus",
    "outcome",
    "simulator_use",
}
FAILURE_KEYS = {
    "schema_version",
    "payload_type",
    "payload_variant",
    "status",
    "case_id",
    "run_id",
    "config",
    "failure",
    "legacy_payload",
    "provenance",
    "outcome",
    "simulator_use",
}
SIMULATOR_USE = {
    "policy_generated_action_steps_executed": 0,
    "teacher_generated_action_steps_executed": 0,
    "efficacy_rollouts_executed": 0,
    "simulator_efficacy_evaluated": False,
}
SOURCE_PAIRING_KEYS = {
    "manifest_sha256",
    "source_r02_sha256",
    "checkpoint_sha256",
    "case_record_sha256",
    "target_normalized",
    "noise",
    "budget_float32",
    "checkpoint_model_to_physical_scale",
    "comparison_checks",
    "comparison_checks_passed",
    "comparison_request_records",
}


class ConstrainedFlowValidationError(ValueError):
    """One fail-closed CFS semantic reconstruction failed."""


def _array(value: Any, *, name: str, shape: tuple[int, ...], dtype: Any) -> np.ndarray:
    array, errors = _validate_array_record(value, name=name, shape=shape)
    if errors or array is None:
        raise ConstrainedFlowValidationError("; ".join(errors or [f"{name} is missing"]))
    if array.dtype != np.dtype(dtype):
        raise ConstrainedFlowValidationError(
            f"{name} must preserve {np.dtype(dtype)}, got {array.dtype}"
        )
    return np.ascontiguousarray(array)


def _decode_tree(value: Any, *, path: str = "tree") -> Any:
    if isinstance(value, Mapping):
        if set(value) == {"dtype", "shape", "sha256", "values"}:
            array, errors = _validate_array_record(value, name=path)
            if errors or array is None:
                raise ConstrainedFlowValidationError("; ".join(errors))
            return np.ascontiguousarray(array)
        return {
            str(key): _decode_tree(item, path=f"{path}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_decode_tree(item, path=f"{path}[]") for item in value]
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ConstrainedFlowValidationError(f"{path} is nonfinite")
        return value
    raise ConstrainedFlowValidationError(
        f"{path} has unsupported type {type(value).__name__}"
    )


def _all_true(value: Any, *, name: str) -> bool:
    if not isinstance(value, Mapping) or not value:
        raise ConstrainedFlowValidationError(f"{name} must be a nonempty mapping")
    if any(type(item) is not bool for item in value.values()):
        raise ConstrainedFlowValidationError(f"{name} must contain only booleans")
    return all(value.values())


def _metrics_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return bool(
        all(float(left.get(key, float("nan"))) == float(right.get(key, float("inf"))) for key in cfs.FIDELITY_LIMITS)
        and left.get("passed") is right.get("passed")
        and left.get("checks") == right.get("checks")
    )


def _validate_config_binding(
    payload_config: Any,
    constrained_config: Mapping[str, Any],
    *,
    constrained_config_path: str | Path | None,
    legacy_config_path: str | Path | None,
) -> None:
    if not isinstance(payload_config, Mapping):
        raise ConstrainedFlowValidationError("payload config binding is missing")
    config_errors = cfs.validate_constrained_flow_config(constrained_config)
    if config_errors:
        raise ConstrainedFlowValidationError(
            "frozen constrained config failed: " + "; ".join(config_errors)
        )
    if payload_config.get(
        "constrained_flow_scientific_hash"
    ) != cfs.constrained_flow_scientific_config_hash(constrained_config):
        raise ConstrainedFlowValidationError("constrained scientific hash changed")
    if constrained_config_path is not None:
        path = Path(constrained_config_path)
        if payload_config.get("constrained_flow_path") != str(path):
            raise ConstrainedFlowValidationError("constrained config path changed")
        if payload_config.get("constrained_flow_sha256") != file_sha256(path):
            raise ConstrainedFlowValidationError("constrained config bytes changed")
    if legacy_config_path is not None:
        path = Path(legacy_config_path)
        if payload_config.get("legacy_path") != str(path):
            raise ConstrainedFlowValidationError("legacy config path changed")
        if payload_config.get("legacy_sha256") != file_sha256(path):
            raise ConstrainedFlowValidationError("legacy config bytes changed")


def _validate_failure_payload(
    value: Mapping[str, Any],
    constrained_config: Mapping[str, Any],
    *,
    expected_run_id: str | None,
    constrained_config_path: str | Path | None,
    legacy_config_path: str | Path | None,
) -> str:
    if set(value) != FAILURE_KEYS:
        raise ConstrainedFlowValidationError("terminal failure payload keys changed")
    if value.get("status") != "apparatus_inconclusive":
        raise ConstrainedFlowValidationError("terminal failure was promoted")
    _validate_config_binding(
        value.get("config"),
        constrained_config,
        constrained_config_path=constrained_config_path,
        legacy_config_path=legacy_config_path,
    )
    failure = value.get("failure")
    if not isinstance(failure, Mapping) or not isinstance(failure.get("message"), str):
        raise ConstrainedFlowValidationError("terminal failure record is incomplete")
    if set(failure) != {
        "stage",
        "error_type",
        "message",
        "finite_failure_is_infeasibility",
        "numeric_failure_is_method_negative",
    }:
        raise ConstrainedFlowValidationError("terminal failure record keys changed")
    if failure.get("stage") != "paired_transport_execution" or not all(
        isinstance(failure.get(key), str) and bool(failure.get(key))
        for key in ("error_type", "message")
    ):
        raise ConstrainedFlowValidationError("terminal failure identity is incomplete")
    if failure.get("finite_failure_is_infeasibility") is not False or failure.get(
        "numeric_failure_is_method_negative"
    ) is not False:
        raise ConstrainedFlowValidationError("terminal failure inflated a method claim")
    outcome = value.get("outcome")
    expected_outcome = {
        "status": "apparatus_inconclusive",
        "scientific_claim_allowed": False,
        "infeasibility_claim_allowed": False,
        "collision_or_progress_claim_allowed": False,
        "probe_training_authorized": False,
        "automatic_next_gate_authorized": False,
    }
    if outcome != expected_outcome or value.get("simulator_use") != SIMULATOR_USE:
        raise ConstrainedFlowValidationError("terminal failure claim boundary changed")
    if expected_run_id is not None and value.get("run_id") != expected_run_id:
        raise ConstrainedFlowValidationError("terminal failure run ID changed")
    if not isinstance(value.get("run_id"), str) or not value["run_id"]:
        raise ConstrainedFlowValidationError("terminal failure run ID is missing")
    legacy = value.get("legacy_payload")
    if not isinstance(legacy, Mapping) or set(legacy) != {"path", "exists", "sha256"}:
        raise ConstrainedFlowValidationError("terminal legacy-payload receipt changed")
    legacy_path = Path(str(legacy.get("path")))
    legacy_exists = legacy.get("exists")
    if type(legacy_exists) is not bool:
        raise ConstrainedFlowValidationError("terminal legacy-payload existence is not boolean")
    if legacy_exists:
        if not legacy_path.is_file() or legacy.get("sha256") != file_sha256(legacy_path):
            raise ConstrainedFlowValidationError("terminal legacy-payload hash changed")
    elif legacy.get("sha256") is not None or legacy_path.exists():
        raise ConstrainedFlowValidationError(
            "absent legacy payload record disagrees with filesystem"
        )
    provenance = value.get("provenance")
    if not isinstance(provenance, Mapping) or set(provenance) != {
        "git_commit",
        "source_node",
        "slurm_job_id",
        "slurm_array_job_id",
        "slurm_array_task_id",
        "timestamp_utc",
    }:
        raise ConstrainedFlowValidationError("terminal provenance keys changed")
    if provenance.get("source_node") != "worker-1" or str(
        provenance.get("slurm_array_task_id")
    ) != "0":
        raise ConstrainedFlowValidationError("terminal provenance source identity changed")
    for key in ("git_commit", "slurm_job_id", "slurm_array_job_id", "timestamp_utc"):
        if not isinstance(provenance.get(key), str) or not provenance[key]:
            raise ConstrainedFlowValidationError(f"terminal provenance {key} is missing")
    return "apparatus_inconclusive"


def _source_arrays(
    source: Mapping[str, Any],
    legacy_result: Mapping[str, Any],
    constrained_config: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.float32]:
    if set(source) != SOURCE_PAIRING_KEYS:
        raise ConstrainedFlowValidationError("CFS source-pairing keys changed")
    frozen = constrained_config.get("frozen_source_bindings")
    if not isinstance(frozen, Mapping):
        raise ConstrainedFlowValidationError("frozen source bindings are missing")
    expected_source = {
        "manifest_sha256": frozen.get("manifest", {}).get("sha256"),
        "source_r02_sha256": frozen.get("source_r02", {}).get("sha256"),
        "checkpoint_sha256": frozen.get("checkpoint_sha256"),
        "case_record_sha256": legacy_result.get("provenance", {}).get(
            "case_record_sha256"
        ),
    }
    legacy_source = legacy_result.get("source_evidence", {})
    legacy_provenance = legacy_result.get("provenance", {})
    if expected_source != {
        "manifest_sha256": legacy_source.get("manifest_sha256"),
        "source_r02_sha256": legacy_source.get("r02_case_sha256"),
        "checkpoint_sha256": legacy_provenance.get("checkpoint_sha256"),
        "case_record_sha256": legacy_provenance.get("case_record_sha256"),
    }:
        raise ConstrainedFlowValidationError("legacy source differs from frozen CFS binding")
    if any(source.get(key) != expected for key, expected in expected_source.items()):
        raise ConstrainedFlowValidationError("CFS source identity differs from legacy binding")
    target = _array(
        source.get("target_normalized"),
        name="source target",
        shape=(10, 32),
        dtype=np.float32,
    )
    noise = _array(
        source.get("noise"),
        name="source noise",
        shape=(10, 32),
        dtype=np.float32,
    )
    scale = _array(
        source.get("checkpoint_model_to_physical_scale"),
        name="source checkpoint scale",
        shape=(10, 32),
        dtype=np.float32,
    )
    baseline = _array(
        legacy_result["target"]["fresh_frozen_normalized"],
        name="source frozen baseline",
        shape=(10, 32),
        dtype=np.float32,
    )
    legacy_target = _array_from_record(legacy_result["target"]["target_normalized"])
    legacy_noise = _array_from_record(legacy_result["provenance"]["noise"])
    legacy_scale = _array_from_record(
        legacy_result["solver"]["first"]["checkpoint_model_to_physical_scale"]
    )
    if not (
        _array_exact(target, legacy_target)
        and _array_exact(noise, legacy_noise)
        and _array_exact(scale, legacy_scale)
    ):
        raise ConstrainedFlowValidationError(
            "CFS source target/noise/scale differs from legacy paired source"
        )
    budget_raw = np.asarray(source.get("budget_float32"), dtype=np.float64)
    if budget_raw.shape != () or not bool(np.isfinite(budget_raw)):
        raise ConstrainedFlowValidationError("CFS source budget is not a finite scalar")
    budget = np.float32(budget_raw)
    expected_budget = np.float32(3.6398398876190186)
    if budget.tobytes() != expected_budget.tobytes() or float(budget_raw) != float(
        expected_budget
    ):
        raise ConstrainedFlowValidationError("CFS source budget changed")
    legacy_budget_raw = np.asarray(
        legacy_result.get("target", {}).get("source_budget_float32"), dtype=np.float64
    )
    legacy_budget = np.float32(legacy_budget_raw)
    if (
        legacy_budget_raw.shape != ()
        or not bool(np.isfinite(legacy_budget_raw))
        or legacy_budget.tobytes() != budget.tobytes()
        or float(legacy_budget_raw) != float(budget_raw)
    ):
        raise ConstrainedFlowValidationError("CFS budget differs from legacy paired source")
    return target, noise, scale, baseline, budget


def _validate_request_records(
    source: Mapping[str, Any], *, target: np.ndarray, noise: np.ndarray, budget: np.float32
) -> Mapping[str, bool]:
    records = source.get("comparison_request_records")
    if not isinstance(records, Mapping) or set(records) != {"first", "duplicate"}:
        raise ConstrainedFlowValidationError("comparison request records changed")
    decoded: dict[str, Mapping[str, Any]] = {}
    decoded_observations: dict[str, Any] = {}
    checks: dict[str, bool] = {}
    for replicate in ("first", "duplicate"):
        record = records[replicate]
        if not isinstance(record, Mapping):
            raise ConstrainedFlowValidationError(f"{replicate} request record is missing")
        ordinary_observation = _decode_tree(
            record.get("ordinary_observation"), path=f"{replicate}.ordinary_observation"
        )
        comparison_observation = _decode_tree(
            record.get("comparison_observation"), path=f"{replicate}.comparison_observation"
        )
        if content_hash(record.get("ordinary_observation")) != record.get(
            "ordinary_observation_sha256"
        ) or content_hash(record.get("comparison_observation")) != record.get(
            "comparison_observation_sha256"
        ):
            raise ConstrainedFlowValidationError("comparison observation hash changed")
        if content_hash(_tree_json(ordinary_observation)) != content_hash(
            _tree_json(comparison_observation)
        ):
            raise ConstrainedFlowValidationError("comparison observation changed")
        checks[f"{replicate}_request_controls_present"] = True
        ordinary = _decode_tree(
            record.get("ordinary_controls"), path=f"{replicate}.ordinary_controls"
        )
        comparison = _decode_tree(
            record.get("comparison_controls"), path=f"{replicate}.comparison_controls"
        )
        if not isinstance(ordinary, Mapping) or not isinstance(comparison, Mapping):
            raise ConstrainedFlowValidationError("comparison controls are malformed")
        if set(ordinary) != {
            "noise",
            "intervention_mode",
            "intervention_step",
            "return_trace",
            "return_normalized_final",
            "target",
            "target_space",
            "model_l2_path_budget",
            "solver_config",
        }:
            raise ConstrainedFlowValidationError("ordinary teacher control keys changed")
        if not (
            ordinary.get("intervention_mode") == "inverse_flow_teacher"
            and ordinary.get("intervention_step") == 5
            and ordinary.get("return_trace") is True
            and ordinary.get("return_normalized_final") is True
            and ordinary.get("target_space") == "model"
            and ordinary.get("solver_config") == dict(SOLVER_CONFIG)
        ):
            raise ConstrainedFlowValidationError("ordinary teacher control contract changed")
        sanitized = dict(comparison)
        if sanitized.pop("experiment_arm", None) != cfs.EXPERIMENT_ARM:
            raise ConstrainedFlowValidationError("comparison arm identity changed")
        if content_hash(_tree_json(sanitized)) != content_hash(_tree_json(ordinary)):
            raise ConstrainedFlowValidationError(
                "comparison request differs by more than experiment_arm"
            )
        checks[f"{replicate}_request_diff_only_experiment_arm"] = True
        if not _array_exact(ordinary.get("target"), target) or not _array_exact(
            ordinary.get("noise"), noise
        ):
            raise ConstrainedFlowValidationError("comparison controls lost target/noise pairing")
        checks[f"{replicate}_request_target_exact"] = True
        checks[f"{replicate}_request_noise_exact"] = True
        observed_budget = np.asarray(ordinary.get("model_l2_path_budget"))
        if (
            observed_budget.size != 1
            or observed_budget.dtype != np.dtype(np.float32)
            or observed_budget.reshape(()).tobytes() != budget.tobytes()
        ):
            raise ConstrainedFlowValidationError("comparison controls lost budget pairing")
        checks[f"{replicate}_request_budget_exact"] = True
        decoded[replicate] = ordinary
        decoded_observations[replicate] = ordinary_observation
    if content_hash(_tree_json(decoded["first"])) != content_hash(
        _tree_json(decoded["duplicate"])
    ):
        raise ConstrainedFlowValidationError("duplicate ordinary requests changed")
    if content_hash(_tree_json(decoded_observations["first"])) != content_hash(
        _tree_json(decoded_observations["duplicate"])
    ):
        raise ConstrainedFlowValidationError("duplicate ordinary observations changed")
    return checks


def _validate_comparison_trace_pairing(
    arm_b: Mapping[str, tuple[Mapping[str, Any], np.ndarray, np.ndarray, Mapping[str, Any]]],
    arm_c: Mapping[str, tuple[Mapping[str, Any], np.ndarray, np.ndarray, Mapping[str, Any]]],
    *,
    target: np.ndarray,
    noise: np.ndarray,
    scale: np.ndarray,
    baseline: np.ndarray,
    budget: np.float32,
) -> Mapping[str, bool]:
    checks: dict[str, bool] = {}
    for replicate in ("first", "duplicate"):
        nested = arm_b[replicate][3]
        outer = arm_c[replicate][3]
        nested_target = np.asarray(nested.get("target"))
        nested_baseline = np.asarray(nested.get("baseline_final"))
        if (
            nested_target.shape != (1, 10, 32)
            or nested_target.dtype != np.dtype(np.float32)
            or nested_baseline.shape != (1, 10, 32)
            or nested_baseline.dtype != np.dtype(np.float32)
        ):
            raise ConstrainedFlowValidationError("comparison Arm-B source tensors changed")
        outer_target = np.asarray(outer.get("inverse_target"))
        outer_noise = np.asarray(outer.get("initial_noise"))
        outer_scale = np.asarray(outer.get("model_to_physical_scale"))
        outer_baseline = np.asarray(outer.get("solver_baseline_final"))
        for name, array in (
            ("outer target", outer_target),
            ("outer noise", outer_noise),
            ("outer scale", outer_scale),
            ("outer baseline", outer_baseline),
        ):
            if array.shape != (10, 32) or array.dtype != np.dtype(np.float32):
                raise ConstrainedFlowValidationError(
                    f"comparison {replicate} {name} changed shape/dtype"
                )
        outer_budget = np.asarray(outer.get("source_control_budget"))
        nested_budget = np.asarray(nested.get("budget"))
        if (
            outer_budget.size != 1
            or outer_budget.dtype != np.dtype(np.float32)
            or nested_budget.size != 1
            or nested_budget.dtype != np.dtype(np.float32)
        ):
            raise ConstrainedFlowValidationError("comparison source budget dtype changed")
        expected_budget = np.asarray(budget, dtype=np.float32).tobytes()
        values = {
            f"{replicate}_comparison_trace_fields_valid": True,
            f"{replicate}_outer_target_exact": _array_exact(outer_target, target),
            f"{replicate}_nested_target_exact": _array_exact(nested_target[0], target),
            f"{replicate}_outer_noise_exact": _array_exact(outer_noise, noise),
            f"{replicate}_outer_scale_exact": _array_exact(outer_scale, scale),
            f"{replicate}_outer_baseline_exact": _array_exact(outer_baseline, baseline),
            f"{replicate}_nested_baseline_exact": _array_exact(
                nested_baseline[0], baseline
            ),
            f"{replicate}_outer_budget_exact": outer_budget.reshape(()).tobytes()
            == expected_budget,
            f"{replicate}_nested_budget_exact": nested_budget.reshape(()).tobytes()
            == expected_budget,
        }
        if not all(values.values()):
            raise ConstrainedFlowValidationError(
                f"comparison {replicate} trace lost legacy source pairing"
            )
        checks.update(values)
    return checks


def _tree_json(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "bytes": value.tobytes().hex(),
        }
    if isinstance(value, Mapping):
        return {str(key): _tree_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_tree_json(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _validate_arm_b(
    record: Mapping[str, Any],
    *,
    budget: np.float32,
    source_target: np.ndarray,
    source_baseline: np.ndarray,
    source_scale: np.ndarray,
) -> tuple[Mapping[str, Any], np.ndarray, np.ndarray, Mapping[str, Any]]:
    diagnostic_record = record.get("diagnostic")
    if not isinstance(diagnostic_record, Mapping):
        raise ConstrainedFlowValidationError("Arm B diagnostic is missing")
    if content_hash(diagnostic_record) != record.get("diagnostic_sha256"):
        raise ConstrainedFlowValidationError("Arm B diagnostic hash changed")
    diagnostic = _decode_tree(diagnostic_record, path="arm_b.diagnostic")
    if not isinstance(diagnostic, Mapping):
        raise ConstrainedFlowValidationError("Arm B diagnostic did not decode")
    recomputed, schedule, _error = cfs._arm_b_summary(
        diagnostic,
        budget=budget,
        source_target=source_target,
        source_baseline=source_baseline,
        source_scale=source_scale,
    )
    if recomputed.get("passed") is not True or not _all_true(
        recomputed.get("checks"), name="recomputed Arm B checks"
    ):
        raise ConstrainedFlowValidationError("independent Arm B reconstruction failed")
    if content_hash(dict(record)) != content_hash(dict(recomputed)):
        raise ConstrainedFlowValidationError("Arm B stored summary differs from reconstruction")
    final_raw = diagnostic.get("rollout", {}).get("final")
    final_array = np.asarray(final_raw)
    if final_array.shape != (1, 10, 32) or final_array.dtype != np.dtype(np.float32):
        raise ConstrainedFlowValidationError("Arm B selected final changed shape/dtype")
    return recomputed, schedule, np.ascontiguousarray(final_array[0]), diagnostic


def _validate_arm_c(
    record: Mapping[str, Any], *, budget: np.float32, action_dtype: np.dtype[Any]
) -> tuple[Mapping[str, Any], np.ndarray, np.ndarray, Mapping[str, Any]]:
    trace_record = record.get("comparison_trace")
    if not isinstance(trace_record, Mapping):
        raise ConstrainedFlowValidationError("Arm C comparison trace is missing")
    trace = _decode_tree(trace_record, path="arm_c.comparison_trace")
    if not isinstance(trace, Mapping):
        raise ConstrainedFlowValidationError("Arm C comparison trace did not decode")
    returned_actions = _array(
        record.get("returned_actions"),
        name="Arm C comparison reply actions",
        shape=(10, 7),
        dtype=action_dtype,
    )
    recomputed, schedule, final = cfs._arm_c_summary(
        trace, budget=budget, returned_actions=returned_actions
    )
    if recomputed.get("passed") is not True or not _all_true(
        recomputed.get("checks"), name="recomputed Arm C checks"
    ):
        raise ConstrainedFlowValidationError("independent Arm C reconstruction failed")
    if content_hash(dict(record)) != content_hash(dict(recomputed)):
        raise ConstrainedFlowValidationError("Arm C stored summary differs from reconstruction")
    return recomputed, schedule, final, trace


def _recompute_duplicate_checks(
    arms_b: Mapping[
        str, tuple[Mapping[str, Any], np.ndarray, np.ndarray, Mapping[str, Any]]
    ],
    arms_c: Mapping[
        str, tuple[Mapping[str, Any], np.ndarray, np.ndarray, Mapping[str, Any]]
    ],
) -> Mapping[str, bool]:
    first_b, duplicate_b = arms_b["first"], arms_b["duplicate"]
    first_c, duplicate_c = arms_c["first"], arms_c["duplicate"]
    first_nested, duplicate_nested = first_b[3], duplicate_b[3]
    first_outer, duplicate_outer = first_c[3], duplicate_c[3]
    checks = {
        "arm_b_full_diagnostic_exact": first_b[0]["diagnostic_sha256"]
        == duplicate_b[0]["diagnostic_sha256"],
        "arm_b_jacobian_exact": first_b[0]["jacobian"]["sha256"]
        == duplicate_b[0]["jacobian"]["sha256"],
        "arm_b_candidate_exact": first_b[0]["schedule"]["sha256"]
        == duplicate_b[0]["schedule"]["sha256"],
        "arm_b_selected_iteration_exact": cfs._scalar(
            first_nested.get("fista", {}).get("selected_iteration"),
            name="first selected iteration",
        )
        == cfs._scalar(
            duplicate_nested.get("fista", {}).get("selected_iteration"),
            name="duplicate selected iteration",
        ),
        "arm_c_status_exact": _array_exact(
            first_outer.get("solver_status"), duplicate_outer.get("solver_status")
        ),
        "arm_c_schedule_exact": _array_exact(first_c[1], duplicate_c[1]),
        "arm_c_selected_final_exact": _array_exact(first_c[2], duplicate_c[2]),
        "arm_c_fidelity_exact": _array_exact(
            first_outer.get("solver_fidelity_error"),
            duplicate_outer.get("solver_fidelity_error"),
        ),
        "arm_c_returned_actions_exact": _array_exact(
            first_outer.get("final_normalized_physical"),
            duplicate_outer.get("final_normalized_physical"),
        ),
    }
    if not all(checks.values()):
        raise ConstrainedFlowValidationError("B/C raw duplicate diagnostics changed")
    return checks


def _validate_replay(
    record: Mapping[str, Any],
    *,
    schedule: np.ndarray,
    selected_final: np.ndarray,
    target: np.ndarray,
    noise: np.ndarray,
    scale: np.ndarray,
    budget: np.float32,
    action_dtype: np.dtype[Any],
) -> Mapping[str, Any]:
    expected_keys = {
        "actions",
        "final_normalized",
        "trace",
        "schedule_diagnostics",
        "recurrence_errors",
        "schedule_errors",
        "checks",
        "passed",
        "elapsed_seconds",
        "fidelity_error",
        "metrics",
        "returned_actions",
    }
    if set(record) != expected_keys:
        raise ConstrainedFlowValidationError("canonical replay summary keys changed")
    trace_record = record.get("trace")
    trace = _trace_from_record(trace_record, name="canonical replay trace")
    recurrence_errors = validate_flow_recurrence(trace)
    if recurrence_errors:
        raise ConstrainedFlowValidationError(
            "canonical recurrence failed: " + "; ".join(recurrence_errors)
        )
    trace_schedule = np.asarray(trace.get("control_velocity_steps"))
    if trace_schedule.dtype != np.dtype(np.float32) or not _array_exact(
        trace_schedule, schedule
    ):
        raise ConstrainedFlowValidationError("canonical replay schedule changed")
    diagnostics, schedule_errors = _schedule_diagnostics(
        trace_schedule, source_budget_float32=budget
    )
    if schedule_errors:
        raise ConstrainedFlowValidationError(
            "canonical replay constraints failed: " + "; ".join(schedule_errors)
        )
    final = np.asarray(trace.get("final_normalized"))
    if final.dtype != np.dtype(np.float32) or not _array_exact(final, selected_final):
        raise ConstrainedFlowValidationError("canonical replay selected final changed")
    if not _array_exact(trace.get("initial_noise"), noise):
        raise ConstrainedFlowValidationError("canonical replay noise changed")
    actions = _array(
        record.get("actions"),
        name="canonical actions",
        shape=(10, 7),
        dtype=action_dtype,
    )
    returned_actions = _array(
        record.get("returned_actions"),
        name="canonical returned actions",
        shape=(10, 7),
        dtype=action_dtype,
    )
    if not _array_exact(actions, returned_actions):
        raise ConstrainedFlowValidationError("canonical action records differ")
    # Current checkpoint unnormalization returns float64.  The exact trace
    # comparison, not a cast, owns this dtype contract.
    if not _array_exact(trace.get("final_normalized_physical"), actions):
        raise ConstrainedFlowValidationError("canonical returned actions changed")
    recorded_final = _array(
        record.get("final_normalized"),
        name="canonical recorded final",
        shape=(10, 32),
        dtype=np.float32,
    )
    if not _array_exact(recorded_final, final):
        raise ConstrainedFlowValidationError("canonical final summary changed")
    error = np.asarray((final - target) * scale, dtype=np.float32)
    recorded_error = _array(
        record.get("fidelity_error"),
        name="canonical fidelity error",
        shape=(10, 32),
        dtype=np.float32,
    )
    if not _array_exact(recorded_error, error):
        raise ConstrainedFlowValidationError("canonical fidelity error changed")
    metrics = cfs._fidelity_metrics(error)
    if not _metrics_equal(record.get("metrics", {}), metrics):
        raise ConstrainedFlowValidationError("canonical replay metrics changed")
    raw_checks = dict(
        _artifact_replay_checks(trace, actions, noise, schedule, budget)
    )
    raw_checks.update(
        {
            "selected_final_exact": _array_exact(final, selected_final),
            "returned_actions_exact_audited_physical": _array_exact(
                actions, trace.get("final_normalized_physical")
            ),
            "zero_simulator_steps": True,
        }
    )
    if record.get("checks") != raw_checks or record.get("passed") is not all(
        raw_checks.values()
    ):
        raise ConstrainedFlowValidationError("canonical replay checks changed")
    if not all(raw_checks.values()):
        raise ConstrainedFlowValidationError("canonical replay raw invariants failed")
    if record.get("recurrence_errors") != [] or record.get("schedule_errors") != []:
        raise ConstrainedFlowValidationError("canonical replay stored errors changed")
    if record.get("schedule_diagnostics") != diagnostics:
        raise ConstrainedFlowValidationError("canonical schedule diagnostics changed")
    elapsed = record.get("elapsed_seconds")
    if not isinstance(elapsed, (int, float)) or not math.isfinite(float(elapsed)) or elapsed < 0:
        raise ConstrainedFlowValidationError("canonical replay elapsed time is invalid")
    return {
        "schedule": schedule,
        "selected_final": selected_final,
        "actions": actions,
        "trace": trace_record,
        "metrics": metrics,
        "schedule_diagnostics": diagnostics,
    }


def _validate_success_payload(
    value: Mapping[str, Any],
    constrained_config: Mapping[str, Any],
    legacy_payload: Mapping[str, Any],
    *,
    expected_run_id: str | None,
    constrained_config_path: str | Path | None,
    legacy_config_path: str | Path | None,
) -> str:
    if set(value) != SUCCESS_KEYS:
        raise ConstrainedFlowValidationError("complete comparison payload keys changed")
    _validate_config_binding(
        value.get("config"),
        constrained_config,
        constrained_config_path=constrained_config_path,
        legacy_config_path=legacy_config_path,
    )
    if legacy_payload.get("payload_type") != LEGACY_PAYLOAD_TYPE:
        raise ConstrainedFlowValidationError("legacy payload identity changed")
    if set(legacy_payload) != {
        "schema_version",
        "payload_type",
        "complete_result_requires_memory_finalization",
        "result_without_memory",
    } or legacy_payload.get("schema_version") != "1.0" or legacy_payload.get(
        "complete_result_requires_memory_finalization"
    ) is not True:
        raise ConstrainedFlowValidationError("legacy pre-memory envelope changed")
    legacy_result = legacy_payload.get("result_without_memory")
    if not isinstance(legacy_result, Mapping):
        raise ConstrainedFlowValidationError("legacy result is missing")
    if expected_run_id is not None and value.get("run_id") != expected_run_id:
        raise ConstrainedFlowValidationError("CFS run ID changed")
    if value.get("run_id") != legacy_result.get("run_id") or value.get("case_id") != CASE_ID:
        raise ConstrainedFlowValidationError("CFS/legacy run or case identity differs")
    legacy_path = Path(str(value.get("legacy_payload_path")))
    if not legacy_path.is_file() or file_sha256(legacy_path) != value.get(
        "legacy_payload_sha256"
    ):
        raise ConstrainedFlowValidationError("legacy payload bytes changed")
    try:
        loaded_legacy = json.loads(legacy_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ConstrainedFlowValidationError(
            f"legacy payload cannot be reloaded: {error}"
        ) from error
    if content_hash(loaded_legacy) != content_hash(dict(legacy_payload)):
        raise ConstrainedFlowValidationError("supplied legacy payload differs from exact file")

    source = value.get("source_pairing")
    if not isinstance(source, Mapping):
        raise ConstrainedFlowValidationError("source pairing record is missing")
    target, noise, scale, baseline, budget = _source_arrays(
        source, legacy_result, constrained_config
    )
    request_pairing_checks = _validate_request_records(
        source, target=target, noise=noise, budget=budget
    )

    arm_records = value.get("arms")
    if not isinstance(arm_records, Mapping) or set(arm_records) != {
        "A_historical_run_b",
        "B_linearized_candidate",
        "C_linearized_then_historical_adam",
    }:
        raise ConstrainedFlowValidationError("arm set changed")
    legacy_status = str(legacy_result.get("status"))
    arm_a = cfs._legacy_arm_a_summary(legacy_payload, constrained_config, legacy_status)
    if arm_a.get("passed") is not True or not _all_true(
        arm_a.get("checks"), name="recomputed Arm A checks"
    ):
        raise ConstrainedFlowValidationError("historical Arm A did not reproduce")
    if content_hash(dict(arm_records["A_historical_run_b"])) != content_hash(
        dict(arm_a)
    ):
        raise ConstrainedFlowValidationError("stored Arm A differs from legacy reconstruction")
    legacy_first = legacy_result["solver"]["first"]
    legacy_duplicate = legacy_result["solver"]["duplicate"]
    legacy_first_actions = _array_from_record(legacy_first["actions"])
    legacy_duplicate_actions = _array_from_record(legacy_duplicate["actions"])
    if (
        legacy_first_actions.shape != (10, 7)
        or legacy_first_actions.dtype != legacy_duplicate_actions.dtype
        or not _array_exact(legacy_first_actions, legacy_duplicate_actions)
    ):
        raise ConstrainedFlowValidationError("legacy action dtype/duplicate binding changed")
    action_dtype = legacy_first_actions.dtype

    b_records = arm_records["B_linearized_candidate"]
    c_records = arm_records["C_linearized_then_historical_adam"]
    if not isinstance(b_records, Mapping) or set(b_records) != {"first", "duplicate"}:
        raise ConstrainedFlowValidationError("Arm B duplicate set changed")
    if not isinstance(c_records, Mapping) or set(c_records) != {"first", "duplicate"}:
        raise ConstrainedFlowValidationError("Arm C duplicate set changed")
    arms_b: dict[
        str, tuple[Mapping[str, Any], np.ndarray, np.ndarray, Mapping[str, Any]]
    ] = {}
    arms_c: dict[
        str, tuple[Mapping[str, Any], np.ndarray, np.ndarray, Mapping[str, Any]]
    ] = {}
    for replicate in ("first", "duplicate"):
        arms_b[replicate] = _validate_arm_b(
            b_records[replicate],
            budget=budget,
            source_target=target,
            source_baseline=baseline,
            source_scale=scale,
        )
        arms_c[replicate] = _validate_arm_c(
            c_records[replicate], budget=budget, action_dtype=action_dtype
        )
    if not (
        _array_exact(arms_b["first"][1], arms_b["duplicate"][1])
        and _array_exact(arms_c["first"][1], arms_c["duplicate"][1])
    ):
        raise ConstrainedFlowValidationError("B/C selected schedules are nondeterministic")
    trace_pairing_checks = _validate_comparison_trace_pairing(
        arms_b,
        arms_c,
        target=target,
        noise=noise,
        scale=scale,
        baseline=baseline,
        budget=budget,
    )
    comparison_pairing_checks = {
        **request_pairing_checks,
        **trace_pairing_checks,
    }
    if (
        source.get("comparison_checks") != comparison_pairing_checks
        or source.get("comparison_checks_passed") is not True
        or not all(comparison_pairing_checks.values())
    ):
        raise ConstrainedFlowValidationError("comparison source-pairing summary changed")
    duplicate_checks = _recompute_duplicate_checks(arms_b, arms_c)
    duplicates = value.get("duplicates")
    if not isinstance(duplicates, Mapping) or set(duplicates) != {
        "checks",
        "passed",
        "comparison_elapsed_seconds",
    }:
        raise ConstrainedFlowValidationError("duplicate summary keys changed")
    elapsed_values = duplicates.get("comparison_elapsed_seconds")
    if (
        not isinstance(elapsed_values, list)
        or len(elapsed_values) != 2
        or any(
            not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            or float(item) < 0.0
            for item in elapsed_values
        )
    ):
        raise ConstrainedFlowValidationError("comparison elapsed-time summary changed")
    if duplicates.get("checks") != duplicate_checks or duplicates.get("passed") is not True:
        raise ConstrainedFlowValidationError("duplicate summary differs from raw diagnostics")

    schedules = {
        ("A_historical_run_b", "first"): _array_from_record(legacy_first["schedule"]),
        ("A_historical_run_b", "duplicate"): _array_from_record(
            legacy_duplicate["schedule"]
        ),
        ("B_linearized_candidate", "first"): arms_b["first"][1],
        ("B_linearized_candidate", "duplicate"): arms_b["duplicate"][1],
        ("C_linearized_then_historical_adam", "first"): arms_c["first"][1],
        ("C_linearized_then_historical_adam", "duplicate"): arms_c["duplicate"][1],
    }
    legacy_first_trace = _trace_from_record(legacy_first["trace"], name="Arm A first trace")
    legacy_duplicate_trace = _trace_from_record(
        legacy_duplicate["trace"], name="Arm A duplicate trace"
    )
    finals = {
        ("A_historical_run_b", "first"): np.asarray(
            legacy_first_trace["solver_internal_replay_final"]
        ),
        ("A_historical_run_b", "duplicate"): np.asarray(
            legacy_duplicate_trace["solver_internal_replay_final"]
        ),
        ("B_linearized_candidate", "first"): arms_b["first"][2],
        ("B_linearized_candidate", "duplicate"): arms_b["duplicate"][2],
        ("C_linearized_then_historical_adam", "first"): arms_c["first"][2],
        ("C_linearized_then_historical_adam", "duplicate"): arms_c["duplicate"][2],
    }

    replays = value.get("canonical_replays")
    if not isinstance(replays, Mapping) or set(replays) != {
        "A_historical_run_b",
        "B_linearized_candidate",
        "C_linearized_then_historical_adam",
    }:
        raise ConstrainedFlowValidationError("canonical replay arm set changed")
    replay_records: dict[tuple[str, str], Mapping[str, Any]] = {}
    for arm in replays:
        arm_replays = replays[arm]
        if not isinstance(arm_replays, Mapping) or set(arm_replays) != {
            "first",
            "duplicate",
            "duplicate_checks",
            "passed",
        }:
            raise ConstrainedFlowValidationError(f"{arm} canonical replay shape changed")
        for replicate in ("first", "duplicate"):
            replay_records[(arm, replicate)] = _validate_replay(
                arm_replays[replicate],
                schedule=schedules[(arm, replicate)],
                selected_final=finals[(arm, replicate)],
                target=target,
                noise=noise,
                scale=scale,
                budget=budget,
                action_dtype=action_dtype,
            )
        first = replay_records[(arm, "first")]
        duplicate = replay_records[(arm, "duplicate")]
        independently_duplicate = {
            "schedule_exact": _array_exact(first["schedule"], duplicate["schedule"]),
            "selected_final_exact": _array_exact(
                first["selected_final"], duplicate["selected_final"]
            ),
            "returned_actions_exact": _array_exact(first["actions"], duplicate["actions"]),
            "replay_final_exact": _array_exact(
                first["selected_final"], duplicate["selected_final"]
            ),
            "recurrence_trace_exact": content_hash(first["trace"])
            == content_hash(duplicate["trace"]),
        }
        if arm_replays.get("duplicate_checks") != independently_duplicate or not all(
            independently_duplicate.values()
        ):
            raise ConstrainedFlowValidationError(f"{arm} duplicate replay changed")
        if arm_replays.get("passed") is not True:
            raise ConstrainedFlowValidationError(f"{arm} canonical replay pass changed")

    legacy_determinism = legacy_result.get("determinism", {})
    if legacy_determinism.get("passed") is not True:
        raise ConstrainedFlowValidationError("legacy default/zero path regression failed")

    arm_b_pass = bool(replay_records[("B_linearized_candidate", "first")]["metrics"]["passed"])
    arm_c_pass = bool(
        replay_records[("C_linearized_then_historical_adam", "first")]["metrics"]["passed"]
    )
    paired_request_checks = {
        "exactly_two_comparison_calls": True,
        "ordinary_teacher_requests_exact": True,
        "comparison_adapter_duplicates_exact": all(duplicate_checks.values()),
        "comparison_source_pairing_exact": all(comparison_pairing_checks.values()),
    }
    apparatus_checks = {
        "arm_a_historical_reproduced": True,
        "arm_b_valid": all(arms_b[item][0]["passed"] is True for item in ("first", "duplicate")),
        "arm_c_valid": all(arms_c[item][0]["passed"] is True for item in ("first", "duplicate")),
        "arm_a_canonical_replays": replays["A_historical_run_b"]["passed"] is True,
        "arm_b_canonical_replays": replays["B_linearized_candidate"]["passed"] is True,
        "arm_c_canonical_replays": replays[
            "C_linearized_then_historical_adam"
        ]["passed"]
        is True,
        "duplicates_valid": all(duplicate_checks.values()),
        "paired_requests_valid": all(paired_request_checks.values()),
        "zero_policy_generated_simulator_steps": legacy_result["simulator_use"].get(
            "policy_generated_action_steps_executed"
        )
        == 0,
        "zero_teacher_generated_simulator_steps": legacy_result["simulator_use"].get(
            "teacher_generated_action_steps_executed"
        )
        == 0,
        "zero_efficacy_rollouts": legacy_result["simulator_use"].get(
            "efficacy_rollouts_executed"
        )
        == 0,
    }
    recomputed_apparatus = all(apparatus_checks.values())
    apparatus = value.get("apparatus")
    if (
        not isinstance(apparatus, Mapping)
        or set(apparatus) != {"checks", "paired_request_checks", "passed"}
        or apparatus.get("checks") != apparatus_checks
        or apparatus.get("paired_request_checks") != paired_request_checks
        or apparatus.get("passed") is not recomputed_apparatus
        or not recomputed_apparatus
    ):
        raise ConstrainedFlowValidationError("apparatus summary differs from raw evidence")
    status = cfs._classify_outcome(
        apparatus_valid=recomputed_apparatus,
        arm_b_nonlinear_pass=arm_b_pass,
        arm_c_nonlinear_pass=arm_c_pass,
    )
    if value.get("status") != status or value.get("outcome", {}).get("status") != status:
        raise ConstrainedFlowValidationError("CFS outcome classification changed")
    outcome = value.get("outcome", {})
    expected_outcome = {
        "status": status,
        "arm_b_linear_prediction_passed": bool(
            arms_b["first"][0]["linear_metrics"]["passed"]
        ),
        "arm_b_nonlinear_replay_passed": arm_b_pass,
        "arm_c_nonlinear_replay_passed": arm_c_pass,
        "finite_nonconvergence_is_infeasibility": False,
        "optimality_certificate": False,
        "infeasibility_certificate": False,
        "simulator_efficacy_evaluated": False,
        "collision_or_progress_claim_allowed": False,
        "student_or_generalization_claim_allowed": False,
        "probe_training_authorized": False,
        "automatic_next_gate_authorized": False,
    }
    if outcome != expected_outcome:
        raise ConstrainedFlowValidationError("CFS outcome claim boundary changed")
    if value.get("simulator_use") != {"setup_only": True, **SIMULATOR_USE}:
        raise ConstrainedFlowValidationError("CFS simulator-use contract changed")
    return status


def validate_constrained_flow_payload_or_raise(
    value: Mapping[str, Any],
    constrained_config: Mapping[str, Any],
    *,
    legacy_payload: Mapping[str, Any] | None = None,
    expected_run_id: str | None = None,
    constrained_config_path: str | Path | None = None,
    legacy_config_path: str | Path | None = None,
) -> str:
    if not isinstance(value, Mapping):
        raise ConstrainedFlowValidationError("CFS payload must be an object")
    if value.get("schema_version") != "1.0" or value.get("payload_type") != cfs.PAYLOAD_TYPE:
        raise ConstrainedFlowValidationError("CFS payload identity changed")
    if value.get("case_id") != CASE_ID:
        raise ConstrainedFlowValidationError("CFS case identity changed")
    variant = value.get("payload_variant")
    if variant == "terminal_apparatus_failure":
        return _validate_failure_payload(
            value,
            constrained_config,
            expected_run_id=expected_run_id,
            constrained_config_path=constrained_config_path,
            legacy_config_path=legacy_config_path,
        )
    if variant != "complete_comparison" or legacy_payload is None:
        raise ConstrainedFlowValidationError("complete CFS payload lacks its legacy payload")
    return _validate_success_payload(
        value,
        constrained_config,
        legacy_payload,
        expected_run_id=expected_run_id,
        constrained_config_path=constrained_config_path,
        legacy_config_path=legacy_config_path,
    )


def validate_constrained_flow_payload(
    value: Mapping[str, Any],
    constrained_config: Mapping[str, Any],
    **kwargs: Any,
) -> list[str]:
    try:
        validate_constrained_flow_payload_or_raise(value, constrained_config, **kwargs)
    except (ConstrainedFlowValidationError, ValueError, TypeError, KeyError, OSError) as error:
        return [str(error)]
    return []


__all__ = [
    "ConstrainedFlowValidationError",
    "validate_constrained_flow_payload",
    "validate_constrained_flow_payload_or_raise",
]
