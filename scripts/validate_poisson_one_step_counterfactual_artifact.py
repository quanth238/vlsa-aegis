#!/usr/bin/env python3
"""Independent consumer for one-step Poisson counterfactual artifacts.

This process is deliberately separate from the allocation-side producer.  It
accepts the immutable result, its final receipt, and every prerequisite used
as an authority.  It writes nothing: success is reported as one JSON object
on stdout and failure as one diagnostic on stderr.

This validator establishes artifact and arithmetic consistency only.  The
exact producer Slurm job must still be checked externally to be
``COMPLETED|0:0`` before the result is interpreted scientifically.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


RESULT_HASH_FIELD = "result_payload_sha256"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
EXPECTED_PROTOCOL_SCHEMA = "vlsa_poisson_one_step_counterfactual_protocol.v2"
EXPECTED_RESULT_SCHEMA = "vlsa_poisson_one_step_counterfactual_result.v1"
EXPECTED_RECEIPT_SCHEMA = (
    "vlsa_poisson_one_step_counterfactual_validation_receipt.v1"
)
RECEIPT_FIELDS = {
    "schema_version",
    "status",
    "scientific_result",
    "run_id",
    "case_id",
    "result",
    "protocol_binding_sha256",
    "provenance_sha256",
    "authority_sha256",
    "nominal_velocity_estimate_sha256",
    "boundary_B_filter_sha256",
    "arm_ledger_sha256",
    "classification_ledger_sha256",
    "core_validation",
    "producer",
    "independent_consumer_required",
    "external_slurm_requirement",
    RESULT_HASH_FIELD,
}
RECEIPT_RESULT_FIELDS = {
    "relative_path",
    "file_sha256",
    "payload_sha256",
    "schema_version",
}
RECEIPT_PRODUCER_FIELDS = {
    "code_commit",
    "slurm_job_id",
    "host",
    "device",
}
PROJECTION_HASH_FIELDS = (
    "protocol_binding_sha256",
    "provenance_sha256",
    "authority_sha256",
    "nominal_velocity_estimate_sha256",
    "boundary_B_filter_sha256",
    "arm_ledger_sha256",
    "classification_ledger_sha256",
)
PREREQUISITE_SCHEMAS = {
    "numeric": "vlsa_poisson_numeric_validation.v1",
    "parity": "vlsa_poisson_shadow_parity.v3",
    "identification": "vlsa_poisson_shadow_identification.v4",
}
PARITY_ACCEPTANCE_FIELDS = frozenset(
    (
        "all_historical_post_step_states_exact",
        "ordinary_and_callback_states_exact",
        "ordinary_and_callback_observations_exact",
        "ordinary_and_callback_boundary_0_mjstate_integration_exact",
        "reward_done_goal_exact",
        "full_callback_exposure",
        "ordinary_step_path_unmodified",
    )
)
IDENTIFICATION_ACCEPTANCE_FIELDS = frozenset(
    (
        "upstream_exact_parity_same_clean_commit",
        "all_237_historical_actions_executed",
        "all_5925_callbacks_observed",
        "historical_state_reward_done_goal_exact",
        "upstream_observation_sequence_exact",
        "settled_mjstate_integration_matches_upstream_exact_parity",
        "complete_mujoco_integration_state_unchanged_by_construction",
        "complete_mujoco_integration_state_unchanged_by_callback",
        "all_link56_protected_sample_point_jacobians_validated",
        "all_link56_protected_sample_field_chain_rules_validated",
        "static_queries_stop_at_first_registered_drift",
        "contact_authority_is_mujoco_nonpositive_distance",
        "no_action_or_control_mutation",
        "no_active_safety_efficacy_claim",
    )
)
EXTERNAL_SLURM_REQUIREMENT = (
    "exact_producer_job_must_be_independently_confirmed_COMPLETED_with_exit_0_0"
)


class OneStepArtifactValidationError(RuntimeError):
    """Raised when a final Stage-13 artifact or authority is inconsistent."""


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as error:
        raise OneStepArtifactValidationError(
            "artifact contains a non-canonical or non-finite value"
        ) from error


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _historical_canonical(value: Any) -> bytes:
    """Match the frozen Table-1 result/action canonicalization exactly."""

    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as error:
        raise OneStepArtifactValidationError(
            "historical result contains a non-canonical or non-finite value"
        ) from error


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise OneStepArtifactValidationError(
            "cannot hash artifact: %s" % path
        ) from error
    return digest.hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise OneStepArtifactValidationError(
            "%s must be one lowercase SHA-256" % label
        )
    return value


def _require_commit(value: Any, label: str) -> str:
    if not isinstance(value, str) or COMMIT_PATTERN.fullmatch(value) is None:
        raise OneStepArtifactValidationError(
            "%s must be one lowercase 40-character Git commit" % label
        )
    return value


def _require_real_file(path: Path, label: str) -> Path:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise OneStepArtifactValidationError(
            "%s must be an existing nonsymlink regular file" % label
        )
    return candidate.resolve()


def _reject_duplicate_pairs(
    pairs: Sequence[Tuple[str, Any]],
) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("duplicate JSON key %r" % key)
        output[key] = value
    return output


def _load_strict_json(path: Path, label: str) -> Tuple[Path, Dict[str, Any]]:
    real = _require_real_file(path, label)
    try:
        value = json.loads(
            real.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda item: (_ for _ in ()).throw(
                ValueError("non-finite JSON constant %r" % item)
            ),
        )
    except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as error:
        raise OneStepArtifactValidationError(
            "%s is not strict finite duplicate-free JSON" % label
        ) from error
    if not isinstance(value, dict):
        raise OneStepArtifactValidationError("%s must contain one object" % label)
    # This recursive walk catches Python infinities constructed by unusual
    # decoders and gives a direct invariant independent of json.dumps.
    _require_finite_tree(value, label)
    return real, value


def _require_finite_tree(value: Any, label: str) -> None:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise OneStepArtifactValidationError("%s contains a non-finite number" % label)
        return
    if value is None or isinstance(value, (bool, int, str)):
        return
    if isinstance(value, list):
        for item in value:
            _require_finite_tree(item, label)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise OneStepArtifactValidationError(
                    "%s contains a non-string object key" % label
                )
            _require_finite_tree(item, label)
        return
    raise OneStepArtifactValidationError(
        "%s contains unsupported JSON value type %s" % (label, type(value).__name__)
    )


def _verify_payload_hash(value: Mapping[str, Any], label: str) -> str:
    expected = _require_sha256(value.get(RESULT_HASH_FIELD), label + "." + RESULT_HASH_FIELD)
    unhashed = {
        key: item for key, item in value.items() if key != RESULT_HASH_FIELD
    }
    observed = _sha256_bytes(_canonical(unhashed))
    if observed != expected:
        raise OneStepArtifactValidationError(
            "%s payload hash differs from its content" % label
        )
    return observed


def _validate_physical_model_contract(value: Any, label: str) -> Dict[str, Any]:
    """Validate the exact opaque contract emitted from the compiled MuJoCo model."""

    contract = _mapping(value, label)
    expected_fields = {
        "schema_version",
        "sha256",
        "field_count",
        "option_field_count",
        "compiled_mjb_sha256",
        "compiled_mjb_bytes",
        "nq",
        "nv",
        "na",
        "mjstate_integration_size",
        "robosuite_flattened_state_size",
        "robosuite_flattened_state_layout",
    }
    if set(contract) != expected_fields:
        raise OneStepArtifactValidationError(
            "%s fields differ from model_physics_contract" % label
        )
    if contract.get("schema_version") != "vlsa_poisson_physical_model.v3":
        raise OneStepArtifactValidationError("%s schema differs" % label)
    _require_sha256(contract.get("sha256"), label + ".sha256")
    _require_sha256(
        contract.get("compiled_mjb_sha256"),
        label + ".compiled_mjb_sha256",
    )
    for field in ("field_count", "option_field_count", "compiled_mjb_bytes"):
        observed = contract.get(field)
        if isinstance(observed, bool) or not isinstance(observed, int) or observed < 1:
            raise OneStepArtifactValidationError(
                "%s.%s must be a positive integer" % (label, field)
            )
    dimensions = {}
    for field, minimum in (("nq", 1), ("nv", 1), ("na", 0)):
        observed = contract.get(field)
        if (
            isinstance(observed, bool)
            or not isinstance(observed, int)
            or observed < minimum
        ):
            raise OneStepArtifactValidationError(
                "%s.%s has an invalid dimension" % (label, field)
            )
        dimensions[field] = int(observed)
    common_size = 1 + dimensions["nq"] + dimensions["nv"] + dimensions["na"]
    integration_size = contract.get("mjstate_integration_size")
    flattened_size = contract.get("robosuite_flattened_state_size")
    if (
        isinstance(integration_size, bool)
        or not isinstance(integration_size, int)
        or integration_size < common_size
        or isinstance(flattened_size, bool)
        or not isinstance(flattened_size, int)
        or flattened_size != common_size
        or contract.get("robosuite_flattened_state_layout")
        != "time_qpos_qvel_act_no_udd_tail"
    ):
        raise OneStepArtifactValidationError(
            "%s state layout differs" % label
        )
    return dict(contract)


def _validate_installed_controller_source_identity(
    *, result: Mapping[str, Any], protocol: Mapping[str, Any]
) -> Dict[str, Any]:
    """Independently re-hash Robosuite's JV source/config when installed."""

    authority = _mapping(
        protocol.get("joint_velocity_controller_authority"),
        "protocol joint-velocity controller authority",
    )
    if authority.get("independent_installed_source_rehash_required") is not True:
        raise OneStepArtifactValidationError(
            "protocol does not require installed controller source re-hashing"
        )
    expected = _mapping(
        authority.get("expected_restore_controller_contract"),
        "protocol expected restore.controller",
    )
    provenance = _mapping(result.get("provenance"), "result.provenance")
    observed = _mapping(
        provenance.get("joint_velocity_controller"),
        "result.provenance.joint_velocity_controller",
    )
    for field in (
        "controller_implementation_file_sha256",
        "controller_configuration_file_sha256",
        "panda_robot_xml_file_sha256",
    ):
        expected_digest = _require_sha256(expected.get(field), "protocol." + field)
        if _require_sha256(observed.get(field), "result." + field) != expected_digest:
            raise OneStepArtifactValidationError(
                "recorded joint-velocity %s differs from protocol" % field
            )
    module_name = expected.get("controller_class_module")
    if not isinstance(module_name, str) or not module_name:
        raise OneStepArtifactValidationError(
            "protocol joint-velocity module identity is missing"
        )
    try:
        specification = importlib.util.find_spec(module_name)
    except (ImportError, ModuleNotFoundError, ValueError):
        specification = None
    origin = None if specification is None else specification.origin
    if not origin:
        raise OneStepArtifactValidationError(
            "installed joint-velocity controller module is unavailable"
        )
    source_path = Path(origin)
    config_path = source_path.parent / "config" / "joint_velocity.json"
    panda_xml_path = (
        source_path.parent.parent
        / "models"
        / "assets"
        / "robots"
        / "panda"
        / "robot.xml"
    )
    if (
        source_path.is_symlink()
        or not source_path.is_file()
        or config_path.is_symlink()
        or not config_path.is_file()
        or panda_xml_path.is_symlink()
        or not panda_xml_path.is_file()
    ):
        raise OneStepArtifactValidationError(
            "installed joint-velocity source/config is not a real file"
        )
    source_sha256 = _sha256_file(source_path)
    config_sha256 = _sha256_file(config_path)
    panda_xml_sha256 = _sha256_file(panda_xml_path)
    if (
        source_sha256 != expected["controller_implementation_file_sha256"]
        or config_sha256 != expected["controller_configuration_file_sha256"]
        or panda_xml_sha256 != expected["panda_robot_xml_file_sha256"]
    ):
        raise OneStepArtifactValidationError(
            "installed joint-velocity source/config hash differs from protocol"
        )
    return {
        "status": "installed_source_rehashed",
        "module": module_name,
        "implementation_file_sha256": source_sha256,
        "configuration_file_sha256": config_sha256,
        "panda_robot_xml_file_sha256": panda_xml_sha256,
        "protocol_and_result_hashes_bound": True,
    }


def _validate_historical_action_binding(
    *,
    historical_path: Path,
    result: Mapping[str, Any],
    dynamic_authority: Mapping[str, Any],
) -> Dict[str, Any]:
    """Bind the Stage-13 gripper to one exact immutable Table-1 action."""

    real, historical = _load_strict_json(historical_path, "historical result")
    expected = _mapping(
        dynamic_authority.get("historical"), "dynamic authority historical"
    )
    observed_file_sha256 = _sha256_file(real)
    observed_payload_sha256 = _require_sha256(
        historical.get(RESULT_HASH_FIELD),
        "historical result payload hash",
    )
    unhashed = {
        key: item for key, item in historical.items() if key != RESULT_HASH_FIELD
    }
    if _sha256_bytes(_historical_canonical(unhashed)) != observed_payload_sha256:
        raise OneStepArtifactValidationError(
            "historical result payload hash differs from its content"
        )
    if (
        observed_file_sha256
        != _require_sha256(
            expected.get("result_file_sha256"),
            "dynamic historical result file hash",
        )
        or observed_payload_sha256
        != _require_sha256(
            expected.get("result_payload_sha256"),
            "dynamic historical result payload hash",
        )
    ):
        raise OneStepArtifactValidationError(
            "supplied historical result differs from dynamic authority"
        )
    case_id = result.get("case_id")
    if (
        historical.get("case_id") != case_id
        or historical.get("arm") != "pi05_plus_aegis_translational"
        or historical.get("status") != "complete"
        or historical.get("scientific_result") is not True
    ):
        raise OneStepArtifactValidationError(
            "historical result is not the complete registered AEGIS case"
        )
    records = historical.get("actions")
    ledger = _mapping(
        historical.get("action_invariance_ledger"),
        "historical action invariance ledger",
    )
    if not isinstance(records, list) or not records:
        raise OneStepArtifactValidationError(
            "historical result has no executed-action ledger"
        )
    actions = []
    for index, raw in enumerate(records):
        record = _mapping(raw, "historical action %d" % index)
        if record.get("step") != index:
            raise OneStepArtifactValidationError(
                "historical action step order differs"
            )
        action = _finite_vector(
            record.get("executed"), 7, "historical executed action %d" % index
        )
        if "env_step_input" in record and _canonical(
            record["env_step_input"]
        ) != _canonical(record.get("executed")):
            raise OneStepArtifactValidationError(
                "historical env-step input differs from executed action"
            )
        actions.append(action)
    action_count = len(actions)
    action_sequence_sha256 = _sha256_bytes(_historical_canonical(actions))
    if (
        ledger.get("action_count") != action_count
        or action_count != expected.get("action_count")
        or ledger.get("executed_sequence_sha256") != action_sequence_sha256
        or action_sequence_sha256
        != expected.get("executed_action_sequence_sha256")
    ):
        raise OneStepArtifactValidationError(
            "historical executed-action sequence differs from dynamic authority"
        )
    pairing = _mapping(historical.get("pairing"), "historical pairing")
    policy_noise_sha256 = _require_sha256(
        pairing.get("policy_noise_schedule_sha256"),
        "historical policy-noise schedule hash",
    )
    if policy_noise_sha256 != expected.get("policy_noise_schedule_sha256"):
        raise OneStepArtifactValidationError(
            "historical policy-noise schedule differs from dynamic authority"
        )

    source = _mapping(result.get("source_boundary"), "result source boundary")
    boundary = source.get("filter_boundary_B")
    if isinstance(boundary, bool) or not isinstance(boundary, int) or boundary < 0:
        raise OneStepArtifactValidationError(
            "result filter boundary is invalid for historical action binding"
        )
    source_action_index = boundary // 25
    if source_action_index >= action_count:
        raise OneStepArtifactValidationError(
            "result source action index exceeds historical exposure"
        )
    gripper = _mapping(
        source.get("gripper_evidence"), "result gripper evidence"
    )
    if gripper.get("source_action_index") != source_action_index:
        raise OneStepArtifactValidationError(
            "result gripper action index differs from historical cadence"
        )
    observed_source_action = _finite_vector(
        gripper.get("source_action_7d"), 7, "result gripper source action"
    )
    if _canonical(observed_source_action) != _canonical(actions[source_action_index]):
        raise OneStepArtifactValidationError(
            "result gripper source action differs from exact historical action"
        )
    return {
        "path": str(real),
        "file_sha256": observed_file_sha256,
        "payload_sha256": observed_payload_sha256,
        "action_count": action_count,
        "executed_action_sequence_sha256": action_sequence_sha256,
        "source_action_index": source_action_index,
        "source_action_7d": list(actions[source_action_index]),
        "passed": True,
    }


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise OneStepArtifactValidationError("%s must be an object" % label)
    return value


def _finite_vector(value: Any, count: int, label: str) -> list:
    if not isinstance(value, list) or len(value) != count:
        raise OneStepArtifactValidationError(
            "%s must contain exactly %d values" % (label, count)
        )
    output = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise OneStepArtifactValidationError("%s contains a non-number" % label)
        numeric = float(item)
        if not math.isfinite(numeric):
            raise OneStepArtifactValidationError("%s contains a non-finite number" % label)
        output.append(numeric)
    return output


def _exact_keys(value: Mapping[str, Any], expected: set, label: str) -> None:
    observed = set(value)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise OneStepArtifactValidationError(
            "%s keys differ (missing=%r extra=%r)" % (label, missing, extra)
        )


def _artifact_identity(path: Path, value: Mapping[str, Any], label: str) -> Dict[str, Any]:
    return {
        "path": str(path),
        "file_sha256": _sha256_file(path),
        "payload_sha256": _verify_payload_hash(value, label),
        "schema_version": value.get("schema_version"),
        "status": value.get("status"),
    }


def _load_prerequisite(path: Path, kind: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    real, value = _load_strict_json(path, kind + " prerequisite")
    identity = _artifact_identity(real, value, kind + " prerequisite")
    if identity["schema_version"] != PREREQUISITE_SCHEMAS[kind]:
        raise OneStepArtifactValidationError(
            "%s prerequisite schema differs" % kind
        )
    if identity["status"] != "passed":
        raise OneStepArtifactValidationError(
            "%s prerequisite is not a complete passing artifact" % kind
        )
    acceptance = _mapping(
        value.get("acceptance"), kind + " prerequisite.acceptance"
    )
    expected_acceptance = (
        PARITY_ACCEPTANCE_FIELDS
        if kind == "parity"
        else IDENTIFICATION_ACCEPTANCE_FIELDS
        if kind == "identification"
        else None
    )
    if not acceptance or any(item is not True for item in acceptance.values()):
        raise OneStepArtifactValidationError(
            "%s prerequisite acceptance must be nonempty and literally all true"
            % kind
        )
    if expected_acceptance is not None and set(acceptance) != set(
        expected_acceptance
    ):
        raise OneStepArtifactValidationError(
            "%s prerequisite acceptance fields differ" % kind
        )
    return value, identity


def _protocol_identity(path: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    real, protocol = _load_strict_json(path, "Stage-13 protocol")
    if protocol.get("schema_version") != EXPECTED_PROTOCOL_SCHEMA:
        raise OneStepArtifactValidationError("Stage-13 protocol schema differs")
    contract = _mapping(protocol.get("result_contract"), "protocol.result_contract")
    if contract.get("schema_version") != EXPECTED_RESULT_SCHEMA:
        raise OneStepArtifactValidationError("protocol result schema binding differs")
    prerequisites = _mapping(protocol.get("prerequisites"), "protocol.prerequisites")
    for kind, field in (
        ("numeric", "numeric_validation_schema_version"),
        ("parity", "exact_parity_schema_version"),
        ("identification", "shadow_identification_schema_version"),
    ):
        if prerequisites.get(field) != PREREQUISITE_SCHEMAS[kind]:
            raise OneStepArtifactValidationError(
                "protocol %s prerequisite schema binding differs" % kind
            )
    return protocol, {
        "path": str(real),
        "file_sha256": _sha256_file(real),
        "semantic_sha256": _sha256_bytes(_canonical(protocol)),
        "schema_version": protocol["schema_version"],
        "protocol_id": protocol.get("protocol_id"),
    }


def _registered_relative_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise OneStepArtifactValidationError("%s must be nonempty" % label)
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise OneStepArtifactValidationError("%s must be a safe relative path" % label)
    return relative


def _path_has_registered_suffix(path: Path, relative: Path, label: str) -> None:
    parts = relative.parts
    if not parts or tuple(path.parts[-len(parts) :]) != tuple(parts):
        raise OneStepArtifactValidationError(
            "%s does not occupy its registered relative path" % label
        )


def _load_selection_protocol(
    path: Path, protocol: Mapping[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    real, selection = _load_strict_json(path, "selection protocol")
    frozen = _mapping(protocol.get("prerequisites"), "protocol.prerequisites")
    relative = _registered_relative_path(
        frozen.get("selection_protocol_relative_path"),
        "protocol.prerequisites.selection_protocol_relative_path",
    )
    _path_has_registered_suffix(real, relative, "selection protocol")
    file_sha256 = _sha256_file(real)
    expected_sha256 = _require_sha256(
        frozen.get("selection_raw_file_sha256"),
        "protocol.prerequisites.selection_raw_file_sha256",
    )
    identity = {
        "relative_path": str(relative),
        "schema_version": frozen.get("selection_schema_version"),
        "file_sha256": file_sha256,
        "protocol_id": frozen.get("selection_protocol_id"),
    }
    if (
        file_sha256 != expected_sha256
        or selection.get("schema_version") != identity["schema_version"]
        or selection.get("protocol_id") != identity["protocol_id"]
        or not isinstance(identity["schema_version"], str)
        or not isinstance(identity["protocol_id"], str)
        or not identity["schema_version"]
        or not identity["protocol_id"]
    ):
        raise OneStepArtifactValidationError(
            "selection file differs from the Stage-13 frozen identity"
        )
    return selection, identity


def _load_runtime_protocol(
    path: Path, protocol: Mapping[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    real, runtime = _load_strict_json(path, "runtime v3 protocol")
    frozen = _mapping(protocol.get("prerequisites"), "protocol.prerequisites")
    relative = _registered_relative_path(
        frozen.get("runtime_protocol_relative_path"),
        "protocol.prerequisites.runtime_protocol_relative_path",
    )
    _path_has_registered_suffix(real, relative, "runtime v3 protocol")
    expected_raw = _require_sha256(
        frozen.get("runtime_raw_file_sha256"),
        "protocol.prerequisites.runtime_raw_file_sha256",
    )
    if _sha256_file(real) != expected_raw:
        raise OneStepArtifactValidationError(
            "runtime v3 raw file differs from the Stage-13 protocol"
        )

    from main.poisson_fullbody.feasibility_protocol import (  # noqa: E402
        FeasibilityProtocolError,
        validate_feasibility_protocol,
    )

    try:
        hashes = validate_feasibility_protocol(runtime)
    except (FeasibilityProtocolError, KeyError, TypeError, ValueError) as error:
        raise OneStepArtifactValidationError(
            "runtime v3 protocol fails pure semantic validation"
        ) from error
    expected_semantic = _require_sha256(
        frozen.get("runtime_semantic_protocol_sha256"),
        "protocol.prerequisites.runtime_semantic_protocol_sha256",
    )
    expected_parameters = _require_sha256(
        frozen.get("runtime_parameter_block_sha256"),
        "protocol.prerequisites.runtime_parameter_block_sha256",
    )
    if (
        runtime.get("schema_version") != frozen.get("runtime_schema_version")
        or runtime.get("protocol_id") != frozen.get("runtime_protocol_id")
        or hashes.protocol_sha256 != expected_semantic
        or hashes.parameter_block_sha256 != expected_parameters
    ):
        raise OneStepArtifactValidationError(
            "runtime v3 semantic identity differs from the Stage-13 protocol"
        )
    return runtime, {
        "relative_path": str(relative),
        "raw_file_sha256": expected_raw,
        "semantic_sha256": expected_semantic,
        "parameter_block_sha256": expected_parameters,
        "schema_version": runtime["schema_version"],
        "protocol_id": runtime["protocol_id"],
    }


def _source_identity(value: Mapping[str, Any], label: str) -> Tuple[str, bool]:
    provenance_value = value.get("provenance")
    provenance = (
        _mapping(provenance_value, label + ".provenance")
        if provenance_value is not None
        else {}
    )
    source = value.get("source") if value.get("source") is not None else provenance.get("source")
    if isinstance(source, Mapping):
        commit = source.get("commit")
        status_short = source.get("status_short")
        clean = isinstance(status_short, list) and not status_short
    else:
        commit = provenance.get("code_commit")
        clean = provenance.get("code_dirty") is False
    return _require_commit(commit, label + " source commit"), bool(clean)


def _cross_validate_prerequisite_commits(
    prerequisites: Mapping[str, Mapping[str, Any]], expected_commit: str
) -> None:
    for kind, value in prerequisites.items():
        commit, clean = _source_identity(value, kind + " prerequisite")
        if commit != expected_commit or not clean:
            raise OneStepArtifactValidationError(
                "%s prerequisite is not from the expected clean commit" % kind
            )


def _cross_validate_prerequisite_chain(
    *,
    protocol: Mapping[str, Any],
    parity: Mapping[str, Any],
    parity_identity: Mapping[str, Any],
    identification: Mapping[str, Any],
) -> None:
    """Bind the external replay artifacts to each other and runtime v3."""

    case_id = _mapping(protocol.get("case"), "protocol.case").get("case_id")
    if parity.get("case_id") != case_id or identification.get("case_id") != case_id:
        raise OneStepArtifactValidationError(
            "parity/identification case differs from the Stage-13 protocol"
        )
    provenance = _mapping(
        identification.get("provenance"), "identification.provenance"
    )
    if provenance.get("upstream_parity_payload_sha256") != parity_identity.get(
        "payload_sha256"
    ):
        raise OneStepArtifactValidationError(
            "identification does not bind the supplied parity payload"
        )
    prerequisites = _mapping(protocol.get("prerequisites"), "protocol.prerequisites")
    runtime_bindings = {
        "runtime_protocol_raw_sha256": prerequisites.get(
            "runtime_raw_file_sha256"
        ),
        "runtime_protocol_semantic_sha256": prerequisites.get(
            "runtime_semantic_protocol_sha256"
        ),
        "runtime_parameter_block_sha256": prerequisites.get(
            "runtime_parameter_block_sha256"
        ),
    }
    for field, expected in runtime_bindings.items():
        expected_digest = _require_sha256(expected, "protocol.prerequisites." + field)
        if provenance.get(field) != expected_digest:
            raise OneStepArtifactValidationError(
                "identification runtime binding differs at %s" % field
            )
    selection_sha256 = _require_sha256(
        prerequisites.get("selection_raw_file_sha256"),
        "protocol.prerequisites.selection_raw_file_sha256",
    )
    selection_protocol_id = prerequisites.get("selection_protocol_id")
    if not isinstance(selection_protocol_id, str) or not selection_protocol_id:
        raise OneStepArtifactValidationError(
            "protocol selection protocol ID is missing"
        )
    if (
        provenance.get("selection_config_sha256") != selection_sha256
        or provenance.get("selection_protocol_id") != selection_protocol_id
    ):
        raise OneStepArtifactValidationError(
            "identification selection binding differs from the Stage-13 protocol"
        )


def _validate_identification_differential_audit(
    *,
    construction: Mapping[str, Any],
    runtime_protocol: Mapping[str, Any],
) -> Dict[str, Any]:
    """Purely reconstruct the claim-bearing identification differential audit."""

    field_bundle = _mapping(
        construction.get("field_bundle"), "identification construction.field_bundle"
    )
    protected_samples = field_bundle.get("protected_samples")
    if not isinstance(protected_samples, list) or not protected_samples:
        raise OneStepArtifactValidationError(
            "identification protected-sample ledger is absent"
        )
    arm_dof_indices = construction.get("arm_dof_indices")
    if not isinstance(arm_dof_indices, list) or len(arm_dof_indices) != 7:
        raise OneStepArtifactValidationError(
            "identification arm-DOF authority is incomplete"
        )
    read_only = _mapping(
        construction.get("complete_integration_state_read_only_audit"),
        "identification construction state audit",
    )
    settled_sha256 = _require_sha256(
        read_only.get("before_sha256"), "identification settled integration state"
    )
    differential_config = _mapping(
        runtime_protocol.get("differential_audit"),
        "runtime v3 differential_audit",
    )
    audit = _mapping(
        construction.get("settled_link56_differential_audit"),
        "identification differential audit",
    )
    serialized = _mapping(
        construction.get("settled_link56_differential_audit_validation"),
        "identification differential validation",
    )

    from main.poisson_fullbody.jacobians import (  # noqa: E402
        DifferentialAuditError,
        validate_protected_sample_differential_audit,
    )

    try:
        reconstructed = validate_protected_sample_differential_audit(
            audit,
            expected_samples=protected_samples,
            expected_arm_dof_indices=arm_dof_indices,
            expected_integration_state_sha256=settled_sha256,
            expected_differential_audit_config=differential_config,
        )
    except (DifferentialAuditError, KeyError, TypeError, ValueError) as error:
        raise OneStepArtifactValidationError(
            "identification differential audit fails pure reconstruction"
        ) from error
    if (
        not isinstance(reconstructed, Mapping)
        or reconstructed.get("passed") is not True
        or _canonical(reconstructed) != _canonical(serialized)
    ):
        raise OneStepArtifactValidationError(
            "identification serialized differential validation differs from reconstruction"
        )
    return dict(reconstructed)


def _validate_full_robot_measurement_sampling(
    *, construction: Mapping[str, Any]
) -> Dict[str, Any]:
    """Validate the full-robot D_sim surface population without MuJoCo."""

    evidence = _mapping(
        construction.get("full_robot_measurement_sampling"),
        "construction.full_robot_measurement_sampling",
    )
    _exact_keys(
        evidence,
        {
            "sample_count",
            "sample_ledger_sha256",
            "geom_records",
            "epsilon_m",
            "maximum_surface_cover_radius_m",
            "coverage_semantics",
            "rigid_roundtrip",
        },
        "construction.full_robot_measurement_sampling",
    )
    count = evidence.get("sample_count")
    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or count <= 0
    ):
        raise OneStepArtifactValidationError(
            "full-robot sample count is invalid"
        )
    _require_sha256(
        evidence.get("sample_ledger_sha256"), "full-robot sample ledger hash"
    )

    resolved = _mapping(
        construction.get("resolved_geometry"), "construction.resolved_geometry"
    )
    try:
        from main.poisson_fullbody.surface_sampling import (  # noqa: E402
            validate_robot_sample_evidence,
        )

        validate_robot_sample_evidence(
            evidence,
            resolved_geom_ids=resolved["robot_geom_ids"],
            resolved_geom_names=resolved["robot_geom_names"],
            resolved_body_ids=resolved["robot_body_ids"],
            roundtrip_field="rigid_roundtrip",
        )
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise OneStepArtifactValidationError(
            "full-robot sampling certificate fails pure validation"
        ) from error
    return dict(evidence)


def _dynamic_authority(
    *,
    protocol: Mapping[str, Any],
    selection_identity: Mapping[str, Any],
    runtime_protocol: Mapping[str, Any],
    parity: Mapping[str, Any],
    identification: Mapping[str, Any],
) -> Dict[str, Any]:
    """Derive claim-bearing authority from the supplied upstream payloads."""

    case = _mapping(protocol.get("case"), "protocol.case")
    parity_provenance = _mapping(parity.get("provenance"), "parity.provenance")
    identification_provenance = _mapping(
        identification.get("provenance"), "identification.provenance"
    )
    for field in (
        "manifest_sha256",
        "manifest_row_sha256",
        "historical_result_file_sha256",
        "historical_result_payload_sha256",
    ):
        if parity_provenance.get(field) != identification_provenance.get(field):
            raise OneStepArtifactValidationError(
                "parity/identification provenance differs at %s" % field
            )

    parity_historical = _mapping(parity.get("historical"), "parity.historical")
    identification_historical = _mapping(
        identification.get("historical"), "identification.historical"
    )
    if _canonical(parity_historical) != _canonical(identification_historical):
        raise OneStepArtifactValidationError(
            "parity and identification historical replay authorities differ"
        )

    ordinary = _mapping(parity.get("ordinary_replay"), "parity.ordinary_replay")
    callback = _mapping(parity.get("callback_replay"), "parity.callback_replay")
    ordinary_settled_state = _mapping(
        ordinary.get("settled_official_integration_state"),
        "parity ordinary boundary-0 official integration state",
    )
    callback_settled_state = _mapping(
        callback.get("settled_official_integration_state"),
        "parity callback boundary-0 official integration state",
    )
    if _canonical(ordinary_settled_state) != _canonical(callback_settled_state):
        raise OneStepArtifactValidationError(
            "parity ordinary/callback boundary-0 integration states differ"
        )
    shadow = _mapping(
        identification.get("shadow_replay"), "identification.shadow_replay"
    )
    action_states = callback.get("action_boundary_state_sha256_ledger")
    official_states = callback.get("official_integration_state_sha256_ledger")
    if not isinstance(action_states, list) or not action_states:
        raise OneStepArtifactValidationError("parity action-state ledger is absent")
    if not isinstance(official_states, list) or not official_states:
        raise OneStepArtifactValidationError("parity official-state ledger is absent")
    for label, ledger in (
        ("parity action state", action_states),
        ("parity official state", official_states),
    ):
        for index, digest in enumerate(ledger):
            _require_sha256(digest, "%s %d" % (label, index))
    if _sha256_bytes(_canonical(action_states)) != callback.get(
        "state_sequence_sha256"
    ):
        raise OneStepArtifactValidationError(
            "parity action-state sequence hash differs from its ledger"
        )
    if _sha256_bytes(_canonical(official_states)) != callback.get(
        "official_integration_state_sequence_sha256"
    ):
        raise OneStepArtifactValidationError(
            "parity official-state sequence hash differs from its ledger"
        )

    identification_callback_rows = shadow.get("callback_state_read_only_ledger")
    identification_callback_count = shadow.get("callback_count")
    if (
        not isinstance(identification_callback_rows, list)
        or not identification_callback_rows
        or isinstance(identification_callback_count, bool)
        or not isinstance(identification_callback_count, int)
        or identification_callback_count != len(identification_callback_rows)
    ):
        raise OneStepArtifactValidationError(
            "identification callback state population differs"
        )
    identification_after_states = []
    for index, row_value in enumerate(identification_callback_rows):
        row = _mapping(
            row_value, "identification callback state row %d" % index
        )
        identification_after_states.append(
            _require_sha256(
                row.get("after_sha256"),
                "identification callback after-state %d" % index,
            )
        )
    if shadow.get("callback_state_read_only_ledger_sha256") != _sha256_bytes(
        _canonical(identification_callback_rows)
    ):
        raise OneStepArtifactValidationError(
            "identification callback read-only ledger hash differs"
        )
    identification_callback_sequence = _sha256_bytes(
        _canonical(identification_after_states)
    )
    if shadow.get("callback_state_sequence_sha256") != identification_callback_sequence:
        raise OneStepArtifactValidationError(
            "identification callback after-state sequence hash differs"
        )
    if (
        shadow.get("action_boundary_state_sha256_ledger") != action_states
        or shadow.get("state_sequence_sha256") != callback.get("state_sequence_sha256")
        or shadow.get("callback_state_sequence_sha256")
        != callback.get("official_integration_state_sequence_sha256")
    ):
        raise OneStepArtifactValidationError(
            "identification replay state authority differs from parity"
        )
    if (
        identification_after_states != official_states
        or identification_callback_sequence
        != callback.get("official_integration_state_sequence_sha256")
    ):
        raise OneStepArtifactValidationError(
            "identification callback state authority differs from parity"
        )

    construction = _mapping(shadow.get("construction"), "identification construction")
    physical_model = _validate_physical_model_contract(
        construction.get("physical_model"),
        "identification construction physical_model",
    )
    field_bundle = _mapping(construction.get("field_bundle"), "construction.field_bundle")
    field_hashes = _mapping(field_bundle.get("hashes"), "construction field hashes")
    read_only = _mapping(
        construction.get("complete_integration_state_read_only_audit"),
        "construction complete integration-state audit",
    )
    parity_settled_state = _mapping(
        callback.get("settled_official_integration_state"),
        "parity boundary-0 official integration state",
    )
    if (
        set(parity_settled_state)
        != {
            "physical_boundary",
            "mujoco_state_specification",
            "state_vector_length",
            "sha256",
        }
        or isinstance(parity_settled_state.get("physical_boundary"), bool)
        or not isinstance(parity_settled_state.get("physical_boundary"), int)
        or parity_settled_state.get("physical_boundary") != 0
        or parity_settled_state.get("mujoco_state_specification")
        != "mjSTATE_INTEGRATION"
        or isinstance(parity_settled_state.get("state_vector_length"), bool)
        or not isinstance(parity_settled_state.get("state_vector_length"), int)
        or parity_settled_state.get("state_vector_length") <= 0
    ):
        raise OneStepArtifactValidationError(
            "parity boundary-0 integration-state authority is invalid"
        )
    parity_settled_state_sha256 = _require_sha256(
        parity_settled_state.get("sha256"),
        "parity boundary-0 integration-state hash",
    )
    if (
        set(read_only)
        != {
            "mujoco_state_specification",
            "state_vector_length",
            "before_sha256",
            "after_sha256",
            "exact_array_equal",
            "semantics",
        }
        or read_only.get("mujoco_state_specification") != "mjSTATE_INTEGRATION"
        or read_only.get("exact_array_equal") is not True
        or not isinstance(read_only.get("semantics"), str)
        or not read_only.get("semantics")
        or read_only.get("before_sha256") != parity_settled_state_sha256
        or read_only.get("after_sha256") != parity_settled_state_sha256
        or read_only.get("state_vector_length")
        != parity_settled_state.get("state_vector_length")
        or read_only.get("state_vector_length")
        != physical_model["mjstate_integration_size"]
    ):
        raise OneStepArtifactValidationError(
            "identification settled state authority differs from parity"
        )
    differential = _mapping(
        construction.get("settled_link56_differential_audit"),
        "construction differential audit",
    )
    differential_validation = _mapping(
        construction.get("settled_link56_differential_audit_validation"),
        "construction differential validation",
    )
    _validate_identification_differential_audit(
        construction=construction,
        runtime_protocol=runtime_protocol,
    )
    full_robot_measurement_sampling = _validate_full_robot_measurement_sampling(
        construction=construction
    )
    poisson = _mapping(
        shadow.get("poisson_identification"), "shadow Poisson identification"
    )
    assessment = _mapping(
        poisson.get("contact_prediction_assessment"),
        "identification contact assessment",
    )
    first_contact = _mapping(
        assessment.get("first_link56_contact"), "identification first link contact"
    )
    primary_warning = _mapping(
        assessment.get("primary_registered_warning"),
        "identification primary registered warning",
    )
    prerequisites = _mapping(protocol.get("prerequisites"), "protocol.prerequisites")

    digests = {
        "manifest_file_sha256": parity_provenance.get("manifest_sha256"),
        "manifest_row_sha256": parity_provenance.get("manifest_row_sha256"),
        "result_file_sha256": parity_historical.get(
            "historical_result_file_sha256"
        ),
        "result_payload_sha256": parity_historical.get(
            "historical_result_payload_sha256"
        ),
        "executed_action_sequence_sha256": parity_historical.get(
            "executed_sequence_sha256"
        ),
        "policy_noise_schedule_sha256": parity_historical.get(
            "policy_noise_schedule_sha256"
        ),
        "runtime_raw_file_sha256": prerequisites.get("runtime_raw_file_sha256"),
        "runtime_semantic_sha256": prerequisites.get(
            "runtime_semantic_protocol_sha256"
        ),
        "runtime_parameter_block_sha256": prerequisites.get(
            "runtime_parameter_block_sha256"
        ),
        "parity_state_sequence_sha256": callback.get("state_sequence_sha256"),
        "parity_official_sequence_sha256": callback.get(
            "official_integration_state_sequence_sha256"
        ),
        "contact_model_authority_sha256": construction.get(
            "contact_model_authority_sha256"
        ),
        "protected_sample_ledger_sha256": field_hashes.get(
            "protected_samples_sha256"
        ),
        "settled_state_sha256": read_only.get("before_sha256"),
        "differential_binding_sha256": differential.get("binding_sha256"),
        "differential_classification_sha256": differential.get(
            "classification_ledger_sha256"
        ),
    }


    for field, digest in digests.items():
        _require_sha256(digest, "dynamic authority " + field)
    if (
        read_only.get("after_sha256") != digests["settled_state_sha256"]
        or read_only.get("exact_array_equal") is not True
    ):
        raise OneStepArtifactValidationError(
            "identification settled construction state is not read-only exact"
        )

    action_count = parity_historical.get("action_count")
    if isinstance(action_count, bool) or not isinstance(action_count, int) or action_count < 1:
        raise OneStepArtifactValidationError("historical action count is invalid")
    if callback.get("executed_action_count") != action_count:
        raise OneStepArtifactValidationError("parity action exposure differs from history")
    official_count = callback.get("official_integration_state_count")
    if (
        isinstance(official_count, bool)
        or not isinstance(official_count, int)
        or official_count != len(official_states)
    ):
        raise OneStepArtifactValidationError("parity official-state count differs")

    return {
        "case": {
            "case_id": case.get("case_id"),
            "manifest_file_sha256": digests["manifest_file_sha256"],
            "manifest_row_sha256": digests["manifest_row_sha256"],
        },
        "selection": {
            "relative_path": selection_identity.get("relative_path"),
            "schema_version": selection_identity.get("schema_version"),
            "file_sha256": selection_identity.get("file_sha256"),
            "protocol_id": selection_identity.get("protocol_id"),
        },
        "historical": {
            "result_file_sha256": digests["result_file_sha256"],
            "result_payload_sha256": digests["result_payload_sha256"],
            "action_count": action_count,
            "executed_action_sequence_sha256": digests[
                "executed_action_sequence_sha256"
            ],
            "policy_noise_schedule_sha256": digests[
                "policy_noise_schedule_sha256"
            ],
        },
        "runtime": {
            "raw_file_sha256": digests["runtime_raw_file_sha256"],
            "semantic_sha256": digests["runtime_semantic_sha256"],
            "parameter_block_sha256": digests["runtime_parameter_block_sha256"],
            "schema_version": prerequisites.get("runtime_schema_version"),
            "protocol_id": prerequisites.get("runtime_protocol_id"),
            "registered_parameters": {
                section: dict(
                    _mapping(
                        runtime_protocol.get(section),
                        "runtime v3.%s" % section,
                    )
                )
                for section in ("admissibility", "coverage", "cbf", "qp", "cadence")
            },
        },
        "parity": {
            "settled_official_integration_state": dict(parity_settled_state),
            "executed_action_count": callback.get("executed_action_count"),
            "action_boundary_state_sha256_ledger": action_states,
            "state_sequence_sha256": digests["parity_state_sequence_sha256"],
            "official_integration_state_count": official_count,
            "official_integration_state_sha256_ledger": official_states,
            "official_integration_state_sequence_sha256": digests[
                "parity_official_sequence_sha256"
            ],
        },
        "identification": {
            "callback_state_read_only_count": identification_callback_count,
            "callback_state_read_only_after_sha256_ledger": (
                identification_after_states
            ),
            "callback_state_sequence_sha256": identification_callback_sequence,
            "first_link56_contact": dict(first_contact),
            "primary_registered_warning": dict(primary_warning),
            "physical_model": physical_model,
            "contact_model_authority_sha256": digests[
                "contact_model_authority_sha256"
            ],
            "resolved_geometry": construction.get("resolved_geometry"),
            "full_robot_measurement_sampling": full_robot_measurement_sampling,
            "field_bundle_hashes": dict(field_hashes),
            "ordered_protected_sample_ledger_sha256": digests[
                "protected_sample_ledger_sha256"
            ],
            "protected_sample_count": field_bundle.get("protected_sample_count"),
            "arm_dof_indices": construction.get("arm_dof_indices"),
            "settled_mjstate_integration_sha256": digests["settled_state_sha256"],
            "differential_binding_sha256": digests[
                "differential_binding_sha256"
            ],
            "differential_classification_ledger_sha256": digests[
                "differential_classification_sha256"
            ],
            "differential_validation": dict(differential_validation),
        },
    }


def _validate_independent_qp_reference(
    *,
    result: Mapping[str, Any],
    protocol: Mapping[str, Any],
    runtime_protocol: Mapping[str, Any],
) -> Dict[str, Any]:
    """Re-solve the serialized hard QP after producer exit with CVXPY/OSQP."""

    execution = _mapping(protocol.get("qp_execution"), "protocol.qp_execution")
    reference = _mapping(
        execution.get("independent_postproducer_reference_check"),
        "protocol independent QP reference check",
    )
    expected_reference = {
        "execution": "separate_artifact_consumer_after_producer_exit",
        "solver": "cvxpy_osqp",
        "eps_abs": 1e-9,
        "eps_rel": 1e-9,
        "max_iterations": 20000,
        "qdot_linf_tolerance_rad_s": 2e-5,
        "required_before_scientific_interpretation": True,
    }
    _exact_keys(reference, set(expected_reference), "independent QP reference")
    if _canonical(reference) != _canonical(expected_reference):
        raise OneStepArtifactValidationError(
            "independent QP reference configuration differs from the frozen gate"
        )

    result_execution = _mapping(result.get("execution"), "result.execution")
    outcome_kind = result_execution.get("outcome_kind")
    if outcome_kind == "preflight_inadmissible_nominal_velocity":
        return _validate_preflight_inadmissible_nominal_velocity(
            result=result,
            protocol=protocol,
            runtime_protocol=runtime_protocol,
        )
    if outcome_kind != "paired_counterfactual_complete":
        raise OneStepArtifactValidationError(
            "result execution outcome is invalid for independent QP validation"
        )

    nominal_record = _mapping(
        result.get("nominal_velocity_estimate"), "result.nominal_velocity_estimate"
    )
    nominal = _finite_vector(
        nominal_record.get("qdot_nom_arm_slice"), 7, "nominal arm qdot"
    )
    boundary_filter = _mapping(
        result.get("boundary_B_filter"), "result.boundary_B_filter"
    )
    cbf = _mapping(
        boundary_filter.get("one_CBF_row_per_exact_bound_sample"),
        "result serialized CBF rows",
    )
    raw_rows = cbf.get("rows_m_per_rad")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise OneStepArtifactValidationError(
            "independent QP reference has no serialized CBF rows"
        )
    rows = [
        _finite_vector(row, 7, "serialized CBF row %d" % index)
        for index, row in enumerate(raw_rows)
    ]
    lower = _finite_vector(
        cbf.get("lower_bounds_m2_per_s"),
        len(rows),
        "serialized CBF lower bounds",
    )
    velocity = _mapping(
        boundary_filter.get("joint_velocity_bound_rows"),
        "result serialized velocity bounds",
    )
    final_lower = _finite_vector(
        velocity.get("final_lower_rad_s"), 7, "serialized final lower bounds"
    )
    final_upper = _finite_vector(
        velocity.get("final_upper_rad_s"), 7, "serialized final upper bounds"
    )
    recorded = _finite_vector(
        boundary_filter.get("qdot_safe"), 7, "producer safe qdot"
    )
    runtime_qp = _mapping(runtime_protocol.get("qp"), "runtime v3.qp")
    weights = _finite_vector(
        runtime_qp.get("weight_diagonal"), 7, "runtime QP weights"
    )
    if any(weight <= 0.0 for weight in weights):
        raise OneStepArtifactValidationError(
            "runtime QP weights must be strictly positive"
        )

    from main.poisson_fullbody.cbf_qp import solve_reference_cvxpy  # noqa: E402

    try:
        solution = solve_reference_cvxpy(
            nominal,
            rows,
            lower,
            final_lower,
            final_upper,
            weight_diagonal=weights,
        )
        reference_qdot = _finite_vector(
            list(solution), 7, "independent reference qdot"
        )
    except (ImportError, RuntimeError, TypeError, ValueError, OverflowError) as error:
        raise OneStepArtifactValidationError(
            "independent CVXPY/OSQP reference QP failed"
        ) from error

    linf_error = max(
        abs(reference_qdot[index] - recorded[index]) for index in range(7)
    )
    tolerance = float(reference["qdot_linf_tolerance_rad_s"])
    if linf_error > tolerance:
        raise OneStepArtifactValidationError(
            "producer safe qdot differs from the independent QP optimum"
        )
    reference_residuals = [
        sum(row[index] * reference_qdot[index] for index in range(7))
        - lower_value
        for row, lower_value in zip(rows, lower)
    ]
    reference_objective = 0.5 * sum(
        weights[index] * (reference_qdot[index] - nominal[index]) ** 2
        for index in range(7)
    )
    producer_objective = 0.5 * sum(
        weights[index] * (recorded[index] - nominal[index]) ** 2
        for index in range(7)
    )
    return {
        "status": "passed",
        "solver": reference["solver"],
        "eps_abs": reference["eps_abs"],
        "eps_rel": reference["eps_rel"],
        "max_iterations": reference["max_iterations"],
        "constraint_count": len(rows),
        "reference_qdot_safe_rad_s": reference_qdot,
        "producer_qdot_safe_rad_s": recorded,
        "qdot_linf_error_rad_s": linf_error,
        "qdot_linf_tolerance_rad_s": tolerance,
        "minimum_reference_constraint_margin_m2_per_s": min(reference_residuals),
        "reference_objective": reference_objective,
        "producer_objective": producer_objective,
        "passed": True,
    }


def _validate_preflight_inadmissible_nominal_velocity(
    *,
    result: Mapping[str, Any],
    protocol: Mapping[str, Any],
    runtime_protocol: Mapping[str, Any],
) -> Dict[str, Any]:
    """Independently reconstruct the registered no-execution OOB outcome."""

    execution = _mapping(result.get("execution"), "result.execution")
    _exact_keys(
        execution,
        {
            "outcome_kind",
            "source_prefix_complete",
            "qp_executed",
            "paired_joint_velocity_physics_executed",
            "no_hidden_clipping",
        },
        "result.execution",
    )
    expected_execution = {
        "outcome_kind": "preflight_inadmissible_nominal_velocity",
        "source_prefix_complete": True,
        "qp_executed": False,
        "paired_joint_velocity_physics_executed": False,
        "no_hidden_clipping": True,
    }
    if _canonical(execution) != _canonical(expected_execution):
        raise OneStepArtifactValidationError(
            "inadmissible nominal execution semantics differ"
        )
    if (
        result.get("boundary_B_filter") is not None
        or result.get("arms") != []
        or result.get("diagnostics") is not None
    ):
        raise OneStepArtifactValidationError(
            "inadmissible nominal outcome contains forbidden QP or arm evidence"
        )

    source = _mapping(result.get("source_boundary"), "result.source_boundary")
    prefix = _mapping(source.get("exact_prefix"), "result exact source prefix")
    state_B = _mapping(prefix.get("state_at_B"), "result state at B")
    state_B5 = _mapping(prefix.get("state_at_B_plus_5"), "result state at B+5")

    qpos_B_raw = state_B.get("qpos")
    qpos_B5_raw = state_B5.get("qpos")
    if (
        not isinstance(qpos_B_raw, list)
        or not qpos_B_raw
        or not isinstance(qpos_B5_raw, list)
        or len(qpos_B5_raw) != len(qpos_B_raw)
    ):
        raise OneStepArtifactValidationError(
            "inadmissible nominal endpoint qpos arrays differ"
        )
    qpos_B = _finite_vector(qpos_B_raw, len(qpos_B_raw), "state-B qpos")
    qpos_B5 = _finite_vector(qpos_B5_raw, len(qpos_B_raw), "state-B+5 qpos")
    for state, qpos, label in (
        (state_B, qpos_B, "state B"),
        (state_B5, qpos_B5, "state B+5"),
    ):
        integration_raw = state.get("integration_state")
        if not isinstance(integration_raw, list):
            raise OneStepArtifactValidationError(
                "%s integration state is absent" % label
            )
        integration = _finite_vector(
            integration_raw, len(integration_raw), label + " integration state"
        )
        if len(integration) < 1 + len(qpos) or integration[1 : 1 + len(qpos)] != qpos:
            raise OneStepArtifactValidationError(
                "%s qpos is not the mjSTATE_INTEGRATION projection" % label
            )

    nominal = _mapping(
        result.get("nominal_velocity_estimate"), "result.nominal_velocity_estimate"
    )
    if (
        nominal.get("method") != "mujoco_mj_differentiatePos_full_nv"
        or nominal.get("arm_joint_types") != ["hinge"] * 7
        or nominal.get("arm_joint_qpos_widths") != [1] * 7
        or nominal.get("out_of_bounds_policy")
        != "inadmissible_no_hidden_clipping"
    ):
        raise OneStepArtifactValidationError(
            "inadmissible nominal scalar-hinge estimator semantics differ"
        )
    indices_raw = nominal.get("arm_qpos_indices")
    if not isinstance(indices_raw, list) or len(indices_raw) != 7:
        raise OneStepArtifactValidationError(
            "inadmissible nominal arm-qpos mapping is incomplete"
        )
    indices = []
    for raw in indices_raw:
        if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
            raise OneStepArtifactValidationError(
                "inadmissible nominal arm-qpos mapping is invalid"
            )
        indices.append(int(raw))
    if len(set(indices)) != 7 or any(index >= len(qpos_B) for index in indices):
        raise OneStepArtifactValidationError(
            "inadmissible nominal arm-qpos mapping exceeds endpoint state"
        )

    counterfactual = _mapping(
        protocol.get("counterfactual_boundary"),
        "protocol.counterfactual_boundary",
    )
    interval_s = float(counterfactual.get("counterfactual_horizon_s"))
    if (
        not math.isfinite(interval_s)
        or interval_s <= 0.0
        or not math.isclose(
            float(nominal.get("interval_s")),
            interval_s,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
    ):
        raise OneStepArtifactValidationError(
            "inadmissible nominal endpoint interval differs"
        )
    reconstructed = [
        (qpos_B5[index] - qpos_B[index]) / interval_s for index in indices
    ]
    recorded = _finite_vector(
        nominal.get("qdot_nom_arm_slice"), 7, "inadmissible nominal arm qdot"
    )
    if any(
        not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-12)
        for observed, expected in zip(recorded, reconstructed)
    ):
        raise OneStepArtifactValidationError(
            "inadmissible nominal arm qdot does not reconstruct from endpoints"
        )

    estimator = _mapping(
        protocol.get("nominal_velocity_estimator"),
        "protocol.nominal_velocity_estimator",
    )
    if indices != estimator.get("expected_arm_qpos_indices"):
        raise OneStepArtifactValidationError(
            "inadmissible nominal arm-qpos mapping differs from protocol"
        )
    runtime_qp = _mapping(runtime_protocol.get("qp"), "runtime v3.qp")
    lower = _finite_vector(
        estimator.get("arm_velocity_lower_rad_s"),
        7,
        "protocol nominal lower bounds",
    )
    upper = _finite_vector(
        estimator.get("arm_velocity_upper_rad_s"),
        7,
        "protocol nominal upper bounds",
    )
    if (
        lower
        != _finite_vector(
            runtime_qp.get("velocity_lower_rad_s"), 7, "runtime QP lower bounds"
        )
        or upper
        != _finite_vector(
            runtime_qp.get("velocity_upper_rad_s"), 7, "runtime QP upper bounds"
        )
        or lower
        != _finite_vector(
            nominal.get("registered_lower_rad_s"), 7, "recorded nominal lower bounds"
        )
        or upper
        != _finite_vector(
            nominal.get("registered_upper_rad_s"), 7, "recorded nominal upper bounds"
        )
    ):
        raise OneStepArtifactValidationError(
            "inadmissible nominal velocity bounds differ from authority"
        )
    lower_excess = [max(lower[index] - value, 0.0) for index, value in enumerate(recorded)]
    upper_excess = [max(value - upper[index], 0.0) for index, value in enumerate(recorded)]
    violations = [
        index
        for index in range(7)
        if lower_excess[index] > 0.0 or upper_excess[index] > 0.0
    ]
    if not violations:
        raise OneStepArtifactValidationError(
            "inadmissible nominal outcome has no reconstructed bound violation"
        )
    if nominal.get("bound_violation_indices") != violations:
        raise OneStepArtifactValidationError(
            "inadmissible nominal violation indices differ"
        )
    for field, expected in (
        ("lower_bound_excess_rad_s", lower_excess),
        ("upper_bound_excess_rad_s", upper_excess),
    ):
        observed = _finite_vector(nominal.get(field), 7, "nominal " + field)
        if any(
            not math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12)
            for left, right in zip(observed, expected)
        ):
            raise OneStepArtifactValidationError(
                "inadmissible nominal %s differs" % field
            )
    maximum_excess = max(lower_excess + upper_excess)
    raw_maximum = nominal.get("maximum_bound_excess_rad_s")
    if (
        isinstance(raw_maximum, bool)
        or not isinstance(raw_maximum, (int, float))
        or not math.isclose(
            float(raw_maximum), maximum_excess, rel_tol=0.0, abs_tol=1e-12
        )
        or nominal.get("finite") is not True
        or nominal.get("arm_velocity_within_registered_bounds") is not False
        or nominal.get("hidden_clipping_applied") is not False
    ):
        raise OneStepArtifactValidationError(
            "inadmissible nominal bound classification differs"
        )
    return {
        "status": "not_run_by_preregistered_inadmissibility",
        "outcome_kind": "preflight_inadmissible_nominal_velocity",
        "solver_executed": False,
        "paired_joint_velocity_physics_executed": False,
        "endpoint_velocity_independently_reconstructed": True,
        "qdot_nom_arm_rad_s": recorded,
        "bound_violation_indices": violations,
        "lower_bound_excess_rad_s": lower_excess,
        "upper_bound_excess_rad_s": upper_excess,
        "maximum_bound_excess_rad_s": maximum_excess,
        "no_hidden_clipping": True,
    }


def validate_one_step_artifacts(
    *,
    result_path: Path,
    receipt_path: Path,
    historical_path: Path,
    protocol_path: Path,
    selection_path: Path,
    runtime_path: Path,
    numeric_path: Path,
    parity_path: Path,
    identification_path: Path,
    expected_code_commit: str,
    expected_run_id: str,
    expected_slurm_job_id: str,
    expected_host: str,
    expected_device: str,
) -> Mapping[str, Any]:
    """Validate one immutable Stage-13 result tree without producer imports."""

    expected_commit = _require_commit(expected_code_commit, "expected code commit")
    if not isinstance(expected_run_id, str) or RUN_ID_PATTERN.fullmatch(expected_run_id) is None:
        raise OneStepArtifactValidationError("expected run ID is invalid")
    for value, label in (
        (expected_slurm_job_id, "expected Slurm job ID"),
        (expected_host, "expected host"),
        (expected_device, "expected device"),
    ):
        if not isinstance(value, str) or not value:
            raise OneStepArtifactValidationError("%s must be nonempty" % label)

    protocol, protocol_identity = _protocol_identity(protocol_path)
    numeric, numeric_identity = _load_prerequisite(numeric_path, "numeric")
    parity, parity_identity = _load_prerequisite(parity_path, "parity")
    identification, identification_identity = _load_prerequisite(
        identification_path, "identification"
    )
    prerequisite_values = {
        "numeric": numeric,
        "parity": parity,
        "identification": identification,
    }
    _cross_validate_prerequisite_commits(prerequisite_values, expected_commit)
    _cross_validate_prerequisite_chain(
        protocol=protocol,
        parity=parity,
        parity_identity=parity_identity,
        identification=identification,
    )
    _, selection_identity = _load_selection_protocol(selection_path, protocol)
    runtime_protocol, _ = _load_runtime_protocol(runtime_path, protocol)

    result_real, result = _load_strict_json(result_path, "Stage-13 result")
    receipt_real, receipt = _load_strict_json(receipt_path, "Stage-13 receipt")
    result_payload_sha256 = _verify_payload_hash(result, "Stage-13 result")
    receipt_payload_sha256 = _verify_payload_hash(receipt, "Stage-13 receipt")
    result_file_sha256 = _sha256_file(result_real)
    receipt_file_sha256 = _sha256_file(receipt_real)
    if (
        result_real.name != "result.json"
        or receipt_real.name != "validation_receipt.json"
        or result_real.parent != receipt_real.parent
        or result_real.parent.name != expected_run_id
    ):
        raise OneStepArtifactValidationError(
            "result and receipt do not occupy the canonical immutable run layout"
        )

    authorities = {
        "expected_code_commit": expected_commit,
        "expected_run_id": expected_run_id,
        "expected_slurm_job_id": expected_slurm_job_id,
        "expected_host": expected_host,
        "expected_device": expected_device,
        "protocol": protocol_identity,
        "numeric_prerequisite": numeric_identity,
        "parity_prerequisite": parity_identity,
        "identification_prerequisite": identification_identity,
        "dynamic_authority": _dynamic_authority(
            protocol=protocol,
            selection_identity=selection_identity,
            runtime_protocol=runtime_protocol,
            parity=parity,
            identification=identification,
        ),
    }
    historical_binding = _validate_historical_action_binding(
        historical_path=historical_path,
        result=result,
        dynamic_authority=authorities["dynamic_authority"],
    )

    # Import only the pure, simulator-free consumer.  Importing the producer
    # here would collapse the independent validation boundary.
    from main.poisson_fullbody.one_step_counterfactual import (  # noqa: E402
        validate_one_step_counterfactual_result,
    )

    core = validate_one_step_counterfactual_result(
        result,
        protocol=protocol,
        protocol_raw_sha256=protocol_identity["file_sha256"],
        expected_authority=authorities,
    )
    if not isinstance(core, Mapping):
        raise OneStepArtifactValidationError(
            "pure Stage-13 validator returned no validation record"
        )
    installed_controller_source = _validate_installed_controller_source_identity(
        result=result, protocol=protocol
    )
    qp_reference = _validate_independent_qp_reference(
        result=result,
        protocol=protocol,
        runtime_protocol=runtime_protocol,
    )

    # Receipt field validation is finalized against the producer-independent
    # core contract.  Keep this consumer strict and exact once that API lands.
    _validate_receipt_binding(
        receipt=receipt,
        result=result,
        result_file_sha256=result_file_sha256,
        result_payload_sha256=result_payload_sha256,
        protocol_identity=protocol_identity,
        authorities=authorities,
        core=core,
        expected_code_commit=expected_commit,
        expected_slurm_job_id=expected_slurm_job_id,
        expected_host=expected_host,
        expected_device=expected_device,
    )
    return {
        "status": "valid",
        "schema_version": EXPECTED_RESULT_SCHEMA,
        "run_id": expected_run_id,
        "case_id": result.get("case_id"),
        "classification": result.get("classification"),
        "result_file_sha256": result_file_sha256,
        "result_payload_sha256": result_payload_sha256,
        "receipt_file_sha256": receipt_file_sha256,
        "receipt_payload_sha256": receipt_payload_sha256,
        "historical_action_binding": historical_binding,
        "pure_core_validation": "passed",
        "independent_artifact_consumer": "passed",
        "independent_qp_reference": qp_reference,
        "installed_controller_source_identity": installed_controller_source,
        "slurm_terminal_state_validation": "external_not_checked_by_this_consumer",
        "external_slurm_requirement": EXTERNAL_SLURM_REQUIREMENT,
    }


def _validate_receipt_binding(
    *,
    receipt: Mapping[str, Any],
    result: Mapping[str, Any],
    result_file_sha256: str,
    result_payload_sha256: str,
    protocol_identity: Mapping[str, Any],
    authorities: Mapping[str, Any],
    core: Mapping[str, Any],
    expected_code_commit: str,
    expected_slurm_job_id: str,
    expected_host: str,
    expected_device: str,
) -> None:
    """Validate the final receipt after the pure result consumer passes."""

    _exact_keys(receipt, RECEIPT_FIELDS, "receipt")
    if receipt.get("schema_version") != EXPECTED_RECEIPT_SCHEMA:
        raise OneStepArtifactValidationError("receipt schema differs")
    if receipt.get("status") != "validated":
        raise OneStepArtifactValidationError("receipt is not final and validated")
    if receipt.get("scientific_result") is not False:
        raise OneStepArtifactValidationError(
            "receipt must label Stage-13 as non-population scientific evidence"
        )
    if (
        receipt.get("run_id") != result.get("run_id")
        or receipt.get("case_id") != result.get("case_id")
    ):
        raise OneStepArtifactValidationError("receipt result identity differs")

    result_binding = _mapping(receipt.get("result"), "receipt.result")
    _exact_keys(result_binding, RECEIPT_RESULT_FIELDS, "receipt.result")
    expected_result_binding = {
        "relative_path": "result.json",
        "file_sha256": result_file_sha256,
        "payload_sha256": result_payload_sha256,
        "schema_version": EXPECTED_RESULT_SCHEMA,
    }
    if _canonical(result_binding) != _canonical(expected_result_binding):
        raise OneStepArtifactValidationError(
            "receipt result file/payload/schema binding differs"
        )

    for field in PROJECTION_HASH_FIELDS:
        receipt_value = _require_sha256(receipt.get(field), "receipt." + field)
        result_value = _require_sha256(result.get(field), "result." + field)
        if receipt_value != result_value:
            raise OneStepArtifactValidationError(
                "receipt projection differs from result at %s" % field
            )
    if _canonical(receipt.get("core_validation")) != _canonical(core):
        raise OneStepArtifactValidationError(
            "receipt core validation differs from fresh pure validation"
        )
    producer = _mapping(receipt.get("producer"), "receipt.producer")
    _exact_keys(producer, RECEIPT_PRODUCER_FIELDS, "receipt.producer")
    expected_producer = {
        "code_commit": expected_code_commit,
        "slurm_job_id": expected_slurm_job_id,
        "host": expected_host,
        "device": expected_device,
    }
    if _canonical(producer) != _canonical(expected_producer):
        raise OneStepArtifactValidationError(
            "receipt producer differs from external authority"
        )
    if receipt.get("independent_consumer_required") is not True:
        raise OneStepArtifactValidationError(
            "receipt does not require the independent artifact consumer"
        )
    if receipt.get("external_slurm_requirement") != EXTERNAL_SLURM_REQUIREMENT:
        raise OneStepArtifactValidationError(
            "receipt external Slurm acceptance requirement differs"
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--historical-result", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--selection-protocol", required=True, type=Path)
    parser.add_argument("--runtime-protocol", required=True, type=Path)
    parser.add_argument("--numeric-prerequisite", required=True, type=Path)
    parser.add_argument("--parity-prerequisite", required=True, type=Path)
    parser.add_argument("--identification-prerequisite", required=True, type=Path)
    parser.add_argument("--expected-code-commit", required=True)
    parser.add_argument("--expected-run-id", required=True)
    parser.add_argument("--expected-slurm-job-id", required=True)
    parser.add_argument("--expected-host", required=True)
    parser.add_argument("--expected-device", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        summary = validate_one_step_artifacts(
            result_path=arguments.result,
            receipt_path=arguments.receipt,
            historical_path=arguments.historical_result,
            protocol_path=arguments.protocol,
            selection_path=arguments.selection_protocol,
            runtime_path=arguments.runtime_protocol,
            numeric_path=arguments.numeric_prerequisite,
            parity_path=arguments.parity_prerequisite,
            identification_path=arguments.identification_prerequisite,
            expected_code_commit=arguments.expected_code_commit,
            expected_run_id=arguments.expected_run_id,
            expected_slurm_job_id=arguments.expected_slurm_job_id,
            expected_host=arguments.expected_host,
            expected_device=arguments.expected_device,
        )
    except (
        KeyError,
        TypeError,
        ValueError,
        OverflowError,
        RuntimeError,
        OneStepArtifactValidationError,
    ) as error:
        print("INVALID: %s" % error, file=sys.stderr)
        return 1
    print(json.dumps(summary, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
