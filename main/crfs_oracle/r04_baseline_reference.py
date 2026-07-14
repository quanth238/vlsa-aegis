"""Frozen ordinary-baseline regression used by the R04B apparatus.

The R04B resume/edit seam is opt-in, but it changes code next to the ordinary
PyTorch sampler.  This module therefore checks the ordinary compiled policy
path in the same allocation, with the same observation and explicit noise as
the eager resume apparatus.  It is apparatus evidence only: no returned action
is sent to the simulator.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from .r04_labels import (
    PHYSICAL_ACTION_SHAPE,
    _array_from_record,
    _array_record,
    _is_sha256,
    _json_compatible,
)
from .runner import _array_hash


COMPILED_DEFAULT_CALLS = 2
R02_TOLERANCE_SOURCE = "docs/decisions/0011-use-eager-path-for-r02-parity.md"
R02_TOLERANCE_SOURCE_SHA256 = (
    "11646bec37bdb2d15e9507080156755300782e0a4eabf91449dcc5105ca10d36"
)
R02_PHYSICAL_LIMITS = {
    "physical_xyz5_max": 0.010,
    "physical_xyz5_rms": 0.005,
    "physical_action7_max": 0.050,
    "physical_action7_rms": 0.015,
}
MODEL_SPACE_COMPARISON_STATUS = (
    "not_exposed_by_ordinary_compiled_physical_policy_reply_"
    "validated_R02_predecessor_remains_authority"
)
COMPILED_REQUEST_CONTRACT = (
    "reserved_envelope_explicit_paired_noise_mode_none_no_trace_"
    "no_correction_or_resume"
)
COMPILED_CALL_PLACEMENT = "one_before_and_one_after_eager_resume_sequence"
COMPILED_COMPARISON_CONTRACT = (
    "exact_current_eager_R04A_golden_hashes_and_compiled_physical_"
    "within_frozen_R02_limits"
)

# These are content from the independently validated, immutable R04A raw
# apparatus artifact.  Exact equality is required only after the new run has
# reproduced the same observation and noise identities.  It is a regression
# anchor, not a claim that eager BF16 is portable to a different device class.
R04A_RAW_ARTIFACT_SHA256 = (
    "b820793ec42a5228c858e297d13806d8ae7f02e5cc7a765769316473d795f285"
)
R04A_GOLDEN_OBSERVATION_SHA256 = (
    "c27069074f08dbeb4ae5c0b8c6afe41222e6a305f8784d3284c9865944700082"
)
R04A_GOLDEN_NOISE_SHA256 = (
    "8b3e29cce20eb702936bd4d72509bb08117f19b7bea65d49f4232a60e0d86d00"
)
R04A_GOLDEN_FINAL_PHYSICAL_SHA256 = (
    "f620a67d1979413812d171f6617033086e5fcf27d87855c91a551827fb03db60"
)
R04A_GOLDEN_TRACE_SHA256_BY_STEP = (
    "45e65ccc4c765c5ab42bbe434cc407c34d4f172d7d3b874c645f3926fa5692c6",
    "418c2d0e3714ec891a587b33bbbb144bd811e3b8168cc2871dc184992443502b",
    "f09e9807b997e1e317df0de52d74adf4b35774f8755630b39714e77738c8a1e6",
    "77d6f235485495250b1a10357f3e1ad4ddffcdc3086d54d929ac07cf3e6233c1",
    "a5ae6bd49cc2825feb2ccd42d41a493062e49b211f4fd1e8a6d54ee7faffb438",
)


def _golden_contract() -> Mapping[str, Any]:
    return {
        "raw_artifact_sha256": R04A_RAW_ARTIFACT_SHA256,
        "observation_sha256": R04A_GOLDEN_OBSERVATION_SHA256,
        "noise_sha256": R04A_GOLDEN_NOISE_SHA256,
        "source_trace_sha256_by_step": [
            {"step_index": step, "sha256": digest}
            for step, digest in enumerate(R04A_GOLDEN_TRACE_SHA256_BY_STEP, start=1)
        ],
        "final_physical_sha256": R04A_GOLDEN_FINAL_PHYSICAL_SHA256,
        "device_conditioning": (
            "exact_same_observation_noise_checkpoint_path_regression_only_"
            "not_cross_device_portability_evidence"
        ),
    }


def expected_r04a_golden_contract() -> Mapping[str, Any]:
    """Return a fresh copy of the immutable R04A baseline anchor metadata."""

    return _golden_contract()


def validate_r04a_golden_source(raw: Mapping[str, Any]) -> Mapping[str, Any]:
    """Extract and independently bind the exact R04A baseline anchors."""

    pairing = raw.get("pairing")
    provenance = raw.get("provenance")
    if not isinstance(pairing, Mapping) or not isinstance(provenance, Mapping):
        raise ValueError("R04A golden source lacks pairing or provenance")
    observation = pairing.get("fixed_observation")
    if not isinstance(observation, Mapping):
        raise ValueError("R04A golden source lacks its fixed observation")
    if observation.get("sha256") != R04A_GOLDEN_OBSERVATION_SHA256:
        raise ValueError("R04A golden observation hash differs")
    if provenance.get("noise_sha256") != R04A_GOLDEN_NOISE_SHA256:
        raise ValueError("R04A golden noise hash differs")

    final, final_errors = _array_from_record(
        pairing.get("final_full_physical_actions"),
        name="r04a.pairing.final_full_physical_actions",
        shape=PHYSICAL_ACTION_SHAPE,
    )
    if final_errors or final is None:
        raise ValueError("R04A golden physical final is invalid: " + "; ".join(final_errors))
    if _array_hash(final) != R04A_GOLDEN_FINAL_PHYSICAL_SHA256:
        raise ValueError("R04A golden physical final hash differs")

    requests = pairing.get("trace_requests")
    if not isinstance(requests, list) or len(requests) != 5:
        raise ValueError("R04A golden source must have five trace requests")
    observed: List[str] = []
    for index, request in enumerate(requests):
        if not isinstance(request, Mapping) or request.get("step_index") != index + 1:
            raise ValueError("R04A golden trace order differs")
        primary = request.get("primary")
        duplicate = request.get("duplicate")
        if not isinstance(primary, Mapping) or not isinstance(duplicate, Mapping):
            raise ValueError("R04A golden trace pair is missing")
        primary_trace = primary.get("trace")
        duplicate_trace = duplicate.get("trace")
        primary_sha = primary_trace.get("sha256") if isinstance(primary_trace, Mapping) else None
        duplicate_sha = duplicate_trace.get("sha256") if isinstance(duplicate_trace, Mapping) else None
        if primary_sha != duplicate_sha or not _is_sha256(primary_sha):
            raise ValueError("R04A golden trace duplicates differ")
        observed.append(str(primary_sha))
    if tuple(observed) != R04A_GOLDEN_TRACE_SHA256_BY_STEP:
        raise ValueError("R04A golden trace hashes differ")
    return _golden_contract()


def infer_compiled_default(
    client: Any,
    observation: Mapping[str, Any],
    noise: np.ndarray,
) -> Mapping[str, Any]:
    """Call the compiled ordinary sampler through an opt-in transport envelope."""

    request = copy.deepcopy(dict(observation))
    request["__crfs__"] = {
        "noise": np.array(noise, copy=True),
        "intervention_mode": "none",
        "return_trace": False,
    }
    reply = client.infer(request)
    if not isinstance(reply, Mapping) or "actions" not in reply:
        raise RuntimeError("R04B compiled baseline reply has no physical actions")
    if "crfs_trace" in reply:
        raise RuntimeError("R04B compiled baseline unexpectedly returned an eager trace")
    return reply


def validated_compiled_reply(
    reply: Mapping[str, Any],
    *,
    name: str,
) -> Tuple[np.ndarray, Mapping[str, Any]]:
    actions = np.ascontiguousarray(np.asarray(reply["actions"]))
    if actions.shape != PHYSICAL_ACTION_SHAPE or not np.all(np.isfinite(actions)):
        raise RuntimeError(f"{name} actions must be finite physical 10x7")
    timing = reply.get("policy_timing", {})
    if not isinstance(timing, Mapping):
        raise RuntimeError(f"{name} policy_timing must be an object")
    return actions, {
        "final_physical": _array_record(actions),
        "policy_timing": _json_compatible(dict(timing)),
    }


def _exact_array(left: np.ndarray, right: np.ndarray) -> bool:
    return bool(
        left.dtype == right.dtype
        and left.shape == right.shape
        and _array_hash(left) == _array_hash(right)
        and np.array_equal(left, right)
    )


def _error_metrics(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    max_limit: float,
    rms_limit: float,
) -> Mapping[str, Any]:
    first = np.asarray(reference, dtype=np.float64)
    second = np.asarray(candidate, dtype=np.float64)
    if first.shape != second.shape or first.size == 0:
        raise ValueError("baseline comparison arrays must have one equal nonempty shape")
    if not np.all(np.isfinite(first)) or not np.all(np.isfinite(second)):
        raise ValueError("baseline comparison arrays must be finite")
    flat = np.ascontiguousarray(np.abs(second - first)).reshape(-1).tolist()
    maximum = 0.0
    squared = 0.0
    for item in flat:
        value = float(item)
        if value > maximum:
            maximum = value
        squared += value * value
    rms = math.sqrt(squared / len(flat))
    return {
        "shape": list(first.shape),
        "max_absolute_error": maximum,
        "rmse": rms,
        "max_limit": max_limit,
        "rms_limit": rms_limit,
        "passed": bool(maximum <= max_limit and rms <= rms_limit),
    }


def _physical_comparison(
    eager: np.ndarray,
    compiled: np.ndarray,
) -> Mapping[str, Any]:
    if eager.shape != PHYSICAL_ACTION_SHAPE or compiled.shape != PHYSICAL_ACTION_SHAPE:
        raise ValueError("R04B physical baseline comparison requires 10x7 actions")
    xyz = _error_metrics(
        eager[:5, :3],
        compiled[:5, :3],
        max_limit=R02_PHYSICAL_LIMITS["physical_xyz5_max"],
        rms_limit=R02_PHYSICAL_LIMITS["physical_xyz5_rms"],
    )
    action7 = _error_metrics(
        eager,
        compiled,
        max_limit=R02_PHYSICAL_LIMITS["physical_action7_max"],
        rms_limit=R02_PHYSICAL_LIMITS["physical_action7_rms"],
    )
    return {
        "physical_first_five_xyz": xyz,
        "physical_full_10x7": action7,
        "all_registered_physical_limits_pass": bool(
            xyz["passed"] and action7["passed"]
        ),
    }


def _source_reference(
    step_records: Sequence[Mapping[str, Any]],
) -> Tuple[np.ndarray, List[Mapping[str, Any]]]:
    if len(step_records) != 5:
        raise ValueError("baseline reference requires five eager source records")
    trace_bindings: List[Mapping[str, Any]] = []
    reference: np.ndarray | None = None
    for index, record in enumerate(step_records):
        source = record.get("source") if isinstance(record, Mapping) else None
        primary = source.get("primary") if isinstance(source, Mapping) else None
        trace = primary.get("trace") if isinstance(primary, Mapping) else None
        trace_sha = trace.get("sha256") if isinstance(trace, Mapping) else None
        if record.get("step_index") != index + 1 or not _is_sha256(trace_sha):
            raise ValueError("baseline reference source trace record differs")
        trace_bindings.append({"step_index": index + 1, "sha256": trace_sha})
        physical, physical_errors = _array_from_record(
            primary.get("final_physical") if isinstance(primary, Mapping) else None,
            name=f"baseline_reference.step_records[{index}].source.primary.final_physical",
            shape=PHYSICAL_ACTION_SHAPE,
        )
        if physical_errors or physical is None:
            raise ValueError("baseline reference source final is invalid: " + "; ".join(physical_errors))
        if reference is None:
            reference = physical
        elif not _exact_array(reference, physical):
            raise ValueError("baseline reference eager finals differ across trace steps")
    if reference is None:
        raise ValueError("baseline reference has no eager physical final")
    return reference, trace_bindings


def build_baseline_reference(
    *,
    observation_identity: Mapping[str, Any],
    noise_sha256: str,
    step_records: Sequence[Mapping[str, Any]],
    compiled_pre_actions: np.ndarray,
    compiled_pre_record: Mapping[str, Any],
    compiled_post_actions: np.ndarray,
    compiled_post_record: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Build a fail-closed current-commit baseline regression record."""

    eager, traces = _source_reference(step_records)
    if observation_identity.get("sha256") != R04A_GOLDEN_OBSERVATION_SHA256:
        raise RuntimeError("R04B fixed observation differs from the R04A golden observation")
    if noise_sha256 != R04A_GOLDEN_NOISE_SHA256:
        raise RuntimeError("R04B explicit noise differs from the R04A golden noise")
    expected_traces = list(_golden_contract()["source_trace_sha256_by_step"])
    if traces != expected_traces:
        raise RuntimeError("R04B eager source traces differ from the R04A golden traces")
    eager_sha = _array_hash(eager)
    if eager_sha != R04A_GOLDEN_FINAL_PHYSICAL_SHA256:
        raise RuntimeError("R04B eager physical final differs from the R04A golden final")
    if not _exact_array(compiled_pre_actions, compiled_post_actions):
        raise RuntimeError("R04B compiled default changed across the eager/resume sequence")

    pre_comparison = _physical_comparison(eager, compiled_pre_actions)
    post_comparison = _physical_comparison(eager, compiled_post_actions)
    if not (
        pre_comparison["all_registered_physical_limits_pass"]
        and post_comparison["all_registered_physical_limits_pass"]
    ):
        raise RuntimeError("R04B compiled default exceeds the frozen R02 physical limits")

    return {
        "request_contract": {
            "semantics": COMPILED_REQUEST_CONTRACT,
            "placement": COMPILED_CALL_PLACEMENT,
            "call_count": COMPILED_DEFAULT_CALLS,
            "explicit_noise_sha256": noise_sha256,
            "intervention_mode": "none",
            "return_trace": False,
            "correction_supplied": False,
            "resume_controls_supplied": False,
        },
        "r04a_golden": _golden_contract(),
        "current_eager_golden_binding": {
            "observation_sha256": observation_identity["sha256"],
            "noise_sha256": noise_sha256,
            "source_trace_sha256_by_step": traces,
            "final_physical_sha256": eager_sha,
            "all_exact": True,
        },
        "compiled_default": {
            "pre_sequence": dict(compiled_pre_record),
            "post_sequence": dict(compiled_post_record),
            "exact_dtype_shape_and_canonical_bytes_across_sequence": True,
        },
        "compiled_vs_current_eager": {
            "tolerance_source": R02_TOLERANCE_SOURCE,
            "tolerance_source_sha256": R02_TOLERANCE_SOURCE_SHA256,
            "registered_physical_limits": dict(R02_PHYSICAL_LIMITS),
            "normalized_model_comparison_status": MODEL_SPACE_COMPARISON_STATUS,
            "pre_sequence": pre_comparison,
            "post_sequence": post_comparison,
            "all_registered_physical_limits_pass": True,
        },
    }


def _record_array(
    value: Any,
    *,
    name: str,
    errors: List[str],
) -> np.ndarray | None:
    array, item_errors = _array_from_record(
        value,
        name=name,
        shape=PHYSICAL_ACTION_SHAPE,
    )
    errors.extend(item_errors)
    return array


def validate_baseline_reference(
    value: Any,
    *,
    fixed_observation: Any,
    noise_sha256: Any,
    step_records: Any,
) -> List[str]:
    """Independently reconstruct every baseline and golden comparison."""

    errors: List[str] = []
    if not isinstance(value, Mapping):
        return ["pairing.baseline_reference must be an object"]
    expected_fields = {
        "request_contract",
        "r04a_golden",
        "current_eager_golden_binding",
        "compiled_default",
        "compiled_vs_current_eager",
    }
    if set(value) != expected_fields:
        errors.append("pairing.baseline_reference has unexpected or missing fields")

    request = value.get("request_contract")
    expected_request = {
        "semantics": COMPILED_REQUEST_CONTRACT,
        "placement": COMPILED_CALL_PLACEMENT,
        "call_count": COMPILED_DEFAULT_CALLS,
        "explicit_noise_sha256": noise_sha256,
        "intervention_mode": "none",
        "return_trace": False,
        "correction_supplied": False,
        "resume_controls_supplied": False,
    }
    if request != expected_request:
        errors.append("pairing.baseline_reference request contract differs")
    if value.get("r04a_golden") != _golden_contract():
        errors.append("pairing.baseline_reference R04A golden contract differs")

    observation_sha = (
        fixed_observation.get("sha256") if isinstance(fixed_observation, Mapping) else None
    )
    try:
        eager, traces = _source_reference(step_records if isinstance(step_records, list) else [])
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        errors.append(f"pairing.baseline_reference eager source is invalid: {error}")
        eager = None
        traces = []
    eager_sha = _array_hash(eager) if eager is not None else None
    expected_binding = {
        "observation_sha256": observation_sha,
        "noise_sha256": noise_sha256,
        "source_trace_sha256_by_step": traces,
        "final_physical_sha256": eager_sha,
        "all_exact": True,
    }
    if value.get("current_eager_golden_binding") != expected_binding:
        errors.append("pairing.baseline_reference current eager binding is not reconstructed")
    if not (
        observation_sha == R04A_GOLDEN_OBSERVATION_SHA256
        and noise_sha256 == R04A_GOLDEN_NOISE_SHA256
        and traces == list(_golden_contract()["source_trace_sha256_by_step"])
        and eager_sha == R04A_GOLDEN_FINAL_PHYSICAL_SHA256
    ):
        errors.append("pairing.baseline_reference current eager path differs from R04A golden")

    compiled = value.get("compiled_default")
    pre: np.ndarray | None = None
    post: np.ndarray | None = None
    if not isinstance(compiled, Mapping) or set(compiled) != {
        "pre_sequence",
        "post_sequence",
        "exact_dtype_shape_and_canonical_bytes_across_sequence",
    }:
        errors.append("pairing.baseline_reference compiled_default fields differ")
    else:
        for position in ("pre_sequence", "post_sequence"):
            item = compiled.get(position)
            if not isinstance(item, Mapping) or set(item) != {
                "final_physical",
                "policy_timing",
            }:
                errors.append(f"pairing.baseline_reference compiled {position} fields differ")
                continue
            if not isinstance(item.get("policy_timing"), Mapping):
                errors.append(f"pairing.baseline_reference compiled {position} timing differs")
            array = _record_array(
                item.get("final_physical"),
                name=f"pairing.baseline_reference.compiled_default.{position}.final_physical",
                errors=errors,
            )
            if position == "pre_sequence":
                pre = array
            else:
                post = array
        exact = bool(pre is not None and post is not None and _exact_array(pre, post))
        if compiled.get("exact_dtype_shape_and_canonical_bytes_across_sequence") is not exact or not exact:
            errors.append("pairing.baseline_reference compiled calls are not exact")

    comparison = value.get("compiled_vs_current_eager")
    expected_comparison_fields = {
        "tolerance_source",
        "tolerance_source_sha256",
        "registered_physical_limits",
        "normalized_model_comparison_status",
        "pre_sequence",
        "post_sequence",
        "all_registered_physical_limits_pass",
    }
    if not isinstance(comparison, Mapping) or set(comparison) != expected_comparison_fields:
        errors.append("pairing.baseline_reference compiled/eager comparison fields differ")
    else:
        if not (
            comparison.get("tolerance_source") == R02_TOLERANCE_SOURCE
            and comparison.get("tolerance_source_sha256") == R02_TOLERANCE_SOURCE_SHA256
            and comparison.get("registered_physical_limits") == R02_PHYSICAL_LIMITS
            and comparison.get("normalized_model_comparison_status")
            == MODEL_SPACE_COMPARISON_STATUS
        ):
            errors.append("pairing.baseline_reference frozen tolerance contract differs")
        reconstructed: Dict[str, Mapping[str, Any]] = {}
        if eager is not None:
            for position, candidate in (("pre_sequence", pre), ("post_sequence", post)):
                if candidate is not None:
                    reconstructed[position] = _physical_comparison(eager, candidate)
                    if comparison.get(position) != reconstructed[position]:
                        errors.append(
                            f"pairing.baseline_reference {position} metrics differ"
                        )
        all_pass = bool(
            len(reconstructed) == 2
            and all(
                item.get("all_registered_physical_limits_pass") is True
                for item in reconstructed.values()
            )
        )
        if comparison.get("all_registered_physical_limits_pass") is not all_pass or not all_pass:
            errors.append("pairing.baseline_reference compiled/eager physical limits failed")
    return errors
