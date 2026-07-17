r"""Paired collision-conditioned :math:`\pi_{0.5}` versus AEGIS runner.

The module is intentionally importable without NumPy, MuJoCo, LIBERO,
GroundingDINO, CVXPY, or the OpenPI client.  Those dependencies are loaded only
inside an allocation-backed execution path.  Pure config and result validators
therefore remain available on the login node and in the dependency-light local
test gate.

The high-level runner accepts a small duck-typed AEGIS provider:

``prepare_perception(...)``
    Run the public GLM/GroundingDINO/point-cloud/MVEE stages once and return a
    finite, serializable perception record.  A no/wrong detection is raised as
    :class:`AegisMethodFailure`; a missing credential or package is raised as
    :class:`AegisApparatusError`.

``new_core(perception=..., literal_pre_settle_geometry=...)``
    Return a fresh controller implementing ``filter_action`` and
    ``observe_executed_state``.  The former returns a mapping with a finite
    seven-dimensional ``action`` and a serializable ``qp`` record.  The latter
    updates the public controller's robot ellipsoid only *after* an action has
    executed.  This preserves the public first-step use of pre-settle ``p1``
    and ``R1``.

The concrete :class:`SafeLiberoAegisRuntime` handles only released frozen
states.  Every arm is reconstructed by reset plus the selected number of
complete dummy controls; hidden physics substeps are measured but never used
as branch states.
"""

from __future__ import annotations

import copy
import ast
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import socket
import struct
import subprocess
import zlib
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

from .aegis_pairing import (
    AEGIS_MAIN_SHA256,
    AEGIS_TRANSLATIONAL_SHA256,
    AEGIS_UTILS_SHA256,
    BASELINE_COMMIT,
    EXECUTED_ACTION_HORIZON,
    EXECUTED_ACTION_SHAPE,
    FROZEN_CASE_IDS,
    FROZEN_MANIFEST_SHA256,
    LATE_INTERVENTION_CASE_IDS,
    POLICY_NOISE_SHAPE,
    PRIMARY_ELIGIBLE_CASE_IDS,
    REGISTERED_SAFETY_MARGIN_M,
    build_pairing_record,
    case_stratum,
    exact_array_record,
    exact_observation_record,
    reduce_case_metrics,
    select_latest_safe_settle_boundary,
    stopping_diagnostic,
    validate_exact_pairing,
)


SCHEMA_VERSION = "1.0"
EXPERIMENT_IDENTITY = "AEGIS-00A"
CANARY_CASE_ID = "crfs-1069f29a8d76463a"
R01_SUMMARY_SHA256 = "715d9326f02df531e0c6d7aa7b94629569b320800b4283c36fff41335d9b8bc5"
R02_CONFIG_SHA256 = "c5be1b4759487f4d6541e49110b90fcf5de404bdaed5f389358db0abe4388f4e"
ELIGIBLE_MANIFEST_SHA256 = "241f1b94a5434973dcfb235b43b9386ad15046ff87d35d5b0c34e295c2132916"
CHECKPOINT_SHA256 = "988055ccfd7032903c073a641f3c5f0f0541df444a315116a16f0bf4716d26ed"
CHECKPOINT_CONFIG_SHA256 = "5c2728c53f4b33ee16380140f303713fdb5df78a8d26ee1489ace374fc54327a"
NORMALIZATION_ASSET_SHA256 = "b3a44bb2810436fb62917decaea58bd4d9110255df527dea21e8fd40c960bd84"
CANARY_R02_SHA256 = "055fcf18781071c6c3575b42a1b44c32a76b474ac1911ad8c5443f78b9c42593"
CHECKPOINT_MODEL_PATH = Path(
    "/mnt/data/quanth/cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch/model.safetensors"
)
CHECKPOINT_CONFIG_PATH = CHECKPOINT_MODEL_PATH.parent / "config.json"
NORMALIZATION_ASSET_PATH = (
    CHECKPOINT_MODEL_PATH.parent
    / "assets/physical-intelligence/libero/norm_stats.json"
)
R02_RESULTS_ROOT = Path(
    "/mnt/data/quanth/experiments/crfs-oracle/r02-oracle-flow-population-20260714a"
)
POLICY_RENDER_SIZE = 224
AEGIS_RENDER_SIZE = 1024
POLICY_ACTION_SHAPE = (10, 7)
SETTLE_BOUNDARIES = tuple(range(21))
PHYSICS_SUBSTEPS_PER_ACTION = 25
MEASUREMENT_SAMPLES_PER_ARM = 1 + EXECUTED_ACTION_HORIZON * PHYSICS_SUBSTEPS_PER_ACTION
TARGET_OBJECT_NAME = "akita_black_bowl_1"
EEF_CENTER_OFFSET_LOCAL_M = (0.0, 0.0, -0.08)
EEF_RADIUS_M = 0.06
DISTANCE_LIMIT_M = 1.0
INTERVENTION_STEP = 5
LIBERO_DUMMY_ACTION = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0)
R06_ROBOT_CLASS = "SingleArm"
R06_ROBOT_NAME = "MountedPanda"
R06_CONTROLLER_CLASS = "OperationalSpaceController"
R06_CONTROLLER_NAME = "OSC_POSE"
R06_GRIPPER_CLASS = "PandaGripper"
CONTROLLER_ARRAY_ATTRIBUTES = (
    ("input_min", "input_min"),
    ("input_max", "input_max"),
    ("output_min", "output_min"),
    ("output_max", "output_max"),
    ("kp", "kp"),
    ("kd", "kd"),
    ("ee_position", "ee_pos"),
    ("ee_orientation_matrix", "ee_ori_mat"),
    ("ee_linear_velocity", "ee_pos_vel"),
    ("ee_angular_velocity", "ee_ori_vel"),
    ("joint_position", "joint_pos"),
    ("joint_velocity", "joint_vel"),
    ("jacobian_position", "J_pos"),
    ("jacobian_orientation", "J_ori"),
    ("jacobian_full", "J_full"),
    ("mass_matrix", "mass_matrix"),
    ("initial_joint", "initial_joint"),
    ("goal_position", "goal_pos"),
    ("goal_orientation_matrix", "goal_ori"),
)


class AegisRunnerError(RuntimeError):
    """Base class for fail-closed R06 execution failures."""


class AegisApparatusError(AegisRunnerError):
    """The registered experiment could not be executed as specified."""

    def __init__(self, message: str, *, evidence: Optional[Mapping[str, Any]] = None):
        super().__init__(message)
        self.evidence = dict(evidence or {})


class AegisMethodFailure(AegisRunnerError):
    """The dependency-complete public method failed on the retained case."""

    def __init__(
        self,
        kind: str,
        message: str,
        *,
        evidence: Optional[Mapping[str, Any]] = None,
    ):
        if not kind or kind == "apparatus_invalid":
            raise ValueError("method failure kind must be explicit and non-apparatus")
        super().__init__(message)
        self.kind = str(kind)
        self.evidence = dict(evidence or {})


def _translate_core_error(error: Exception, *, stage: str) -> AegisRunnerError:
    """Translate the additive baseline adapter's typed error without importing it.

    Duck typing keeps this module dependency-light and lets focused tests use a
    fake safety core.  Runtime/source/input failures mean the registered
    apparatus was not instantiated.  A finite dependency-complete geometry or
    QP failure is a retained method outcome.
    """

    code = str(getattr(error, "code", "")).strip()
    details = getattr(error, "details", {})
    evidence = {
        "stage": str(stage),
        "adapter_failure_code": code or type(error).__name__,
        "adapter_details": dict(details) if isinstance(details, Mapping) else {},
    }
    if code in {
        "runtime_dependency_failure",
        "upstream_source_drift",
        "input_failure",
        "initialization_failure",
    }:
        return AegisApparatusError(str(error), evidence=evidence)
    if code:
        return AegisMethodFailure(code, str(error), evidence=evidence)
    return AegisMethodFailure(
        f"{stage}_failure",
        str(error),
        evidence=evidence,
    )


def _numpy():
    try:
        import numpy as np
    except ModuleNotFoundError as error:  # pragma: no cover - allocation dependency
        raise AegisApparatusError("R06 execution requires NumPy") from error
    return np


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _content_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _read_json(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _read_jsonl(path: Path) -> Tuple[Mapping[str, Any], ...]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, raw in enumerate(stream, start=1):
            line = raw.strip()
            if not line:
                raise ValueError(f"{path}:{line_number} is blank")
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError(f"{path}:{line_number} must be a JSON object")
            rows.append(value)
    return tuple(rows)


def _resolve_inside(root: Path, value: str, *, label: str) -> Path:
    candidate = (root / str(value)).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} escapes the repository root") from error
    if not candidate.is_file():
        raise ValueError(f"{label} does not exist: {candidate}")
    return candidate


def _require_exact(value: Any, expected: Any, *, label: str) -> None:
    if value != expected:
        raise ValueError(f"{label} differs from the frozen R06 contract")


@dataclass(frozen=True)
class AegisExperimentConfig:
    """Validated R06 config plus its exact repository bindings."""

    path: str
    sha256: str
    raw: Mapping[str, Any]
    manifest_path: str
    manifest_sha256: str
    cases: Tuple[Mapping[str, Any], ...]
    ready_to_run: bool
    execution_release: Optional[Mapping[str, Any]]


def load_aegis_experiment_config(
    config_path: str | Path,
    *,
    repo_root: str | Path,
    require_execution_release: bool = True,
) -> AegisExperimentConfig:
    """Load the preregistration and revalidate every frozen source/stratum.

    ``require_execution_release=False`` supports dependency-light review of the
    accepted draft.  The execution runner always requires the default ``True``.
    """

    root = Path(repo_root).resolve()
    path = Path(config_path)
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError("R06 config must be inside the repository") from error
    raw = _read_json(path)

    _require_exact(raw.get("schema_version"), SCHEMA_VERSION, label="schema_version")
    _require_exact(raw.get("experiment_identity"), EXPERIMENT_IDENTITY, label="experiment_identity")
    _require_exact(raw.get("gate"), "R06", label="gate")
    manifest = raw.get("manifest")
    if not isinstance(manifest, Mapping):
        raise ValueError("manifest must be an object")
    _require_exact(manifest.get("sha256"), FROZEN_MANIFEST_SHA256, label="manifest.sha256")
    _require_exact(manifest.get("ordered_cases"), 20, label="manifest.ordered_cases")
    _require_exact(manifest.get("case_removal_forbidden"), True, label="manifest.case_removal_forbidden")
    manifest_path = _resolve_inside(root, str(manifest.get("path", "")), label="manifest.path")
    _require_exact(_sha256_path(manifest_path), FROZEN_MANIFEST_SHA256, label="live manifest hash")
    cases = _read_jsonl(manifest_path)
    _require_exact(
        tuple(str(row.get("case_id")) for row in cases),
        FROZEN_CASE_IDS,
        label="ordered manifest case IDs",
    )

    canary = raw.get("canary")
    if not isinstance(canary, Mapping):
        raise ValueError("canary must be an object")
    _require_exact(canary.get("row_index_zero_based"), 0, label="canary row")
    _require_exact(canary.get("case_id"), CANARY_CASE_ID, label="canary case")
    _require_exact(
        canary.get("automatic_population_launch_authorized"),
        False,
        label="automatic population launch",
    )

    source = raw.get("source_evidence")
    if not isinstance(source, Mapping):
        raise ValueError("source_evidence must be an object")
    _require_exact(source.get("baseline_revision"), BASELINE_COMMIT, label="baseline revision")
    for key, frozen_hash in (
        ("r01_summary", R01_SUMMARY_SHA256),
        ("r02_config", R02_CONFIG_SHA256),
    ):
        record = source.get(key)
        if not isinstance(record, Mapping):
            raise ValueError(f"source_evidence.{key} must be an object")
        _require_exact(record.get("sha256"), frozen_hash, label=f"{key}.sha256")
        source_path = _resolve_inside(root, str(record.get("path", "")), label=f"{key}.path")
        _require_exact(_sha256_path(source_path), frozen_hash, label=f"live {key} hash")
    for prefix in ("base", "current"):
        decision_path = _resolve_inside(
            root,
            str(source.get(f"{prefix}_decision", "")),
            label=f"{prefix}_decision",
        )
        _require_exact(
            _sha256_path(decision_path),
            source.get(f"{prefix}_decision_sha256"),
            label=f"live {prefix} decision hash",
        )
    checkpoint = source.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise ValueError("source_evidence.checkpoint must be an object")
    _require_exact(checkpoint.get("sha256"), CHECKPOINT_SHA256, label="checkpoint hash")
    _require_exact(
        source.get("normalization_asset_sha256"),
        NORMALIZATION_ASSET_SHA256,
        label="normalization asset hash",
    )

    strata = raw.get("strata")
    if not isinstance(strata, Mapping):
        raise ValueError("strata must be an object")
    primary = strata.get("primary_standard_settled")
    late = strata.get("pre_settle_late_intervention")
    if not isinstance(primary, Mapping) or not isinstance(late, Mapping):
        raise ValueError("both frozen strata must be objects")
    _require_exact(primary.get("count"), 17, label="primary count")
    _require_exact(primary.get("manifest_sha256"), ELIGIBLE_MANIFEST_SHA256, label="eligible manifest hash")
    eligible_path = _resolve_inside(root, str(primary.get("manifest", "")), label="eligible manifest")
    _require_exact(_sha256_path(eligible_path), ELIGIBLE_MANIFEST_SHA256, label="live eligible hash")
    eligible_rows = _read_jsonl(eligible_path)
    _require_exact(
        tuple(str(row.get("case_id")) for row in eligible_rows),
        PRIMARY_ELIGIBLE_CASE_IDS,
        label="primary stratum order",
    )
    _require_exact(late.get("count"), 3, label="late count")
    _require_exact(tuple(late.get("case_ids", ())), LATE_INTERVENTION_CASE_IDS, label="late stratum IDs")
    _require_exact(late.get("pooled_with_primary"), False, label="stratum pooling")

    pairing = raw.get("pairing")
    if not isinstance(pairing, Mapping):
        raise ValueError("pairing must be an object")
    for key, expected in (
        ("policy_render_size", POLICY_RENDER_SIZE),
        ("aegis_perception_render_size", AEGIS_RENDER_SIZE),
        ("policy_noise_shape", list(POLICY_NOISE_SHAPE)),
        ("policy_noise_dtype", "float32"),
        ("sampler_path", "eager_explicit_noise_trace_only"),
        ("duplicate_policy_inference_required", True),
        ("returned_action_byte_equality_required", True),
        ("executed_actions", EXECUTED_ACTION_HORIZON),
        ("executed_action_dimension", EXECUTED_ACTION_SHAPE[1]),
        ("physics_substeps_per_action", PHYSICS_SUBSTEPS_PER_ACTION),
    ):
        _require_exact(pairing.get(key), expected, label=f"pairing.{key}")
    _require_exact(tuple(pairing.get("settle_action", ())), LIBERO_DUMMY_ACTION, label="settle action")

    branch = raw.get("branch_selection")
    if not isinstance(branch, Mapping):
        raise ValueError("branch_selection must be an object")
    _require_exact(branch.get("primary_requires_boundary"), 20, label="primary boundary")
    _require_exact(branch.get("hidden_physics_substep_branching"), "forbidden_measurement_only", label="branching semantics")
    _require_exact(branch.get("no_admissible_branch_is_retained"), True, label="no-branch retention")

    arms = raw.get("arms")
    if not isinstance(arms, Mapping):
        raise ValueError("arms must be an object")
    public = arms.get("pi05_plus_aegis_codex_label")
    if not isinstance(public, Mapping):
        raise ValueError("public AEGIS arm must be an object")
    for key, expected in (
        ("authoritative_entrypoint_sha256", AEGIS_MAIN_SHA256),
        ("translational_reference_sha256", AEGIS_TRANSLATIONAL_SHA256),
        ("authoritative_utils_sha256", AEGIS_UTILS_SHA256),
        ("controller", "canonical_public_full_9_variable_translation_rotation_cbf_qp"),
        ("perception_mode", "codex_frozen_semantic_label_plus_original_groundingdino"),
        ("oracle_label_or_geometry_substitution", "forbidden"),
        ("translational_only_substitution", "forbidden"),
        ("qp_fallback", "forbidden_record_qp_failure"),
    ):
        _require_exact(public.get(key), expected, label=f"public AEGIS {key}")
    for key, frozen_hash in (
        ("authoritative_entrypoint", AEGIS_MAIN_SHA256),
        ("authoritative_utils", AEGIS_UTILS_SHA256),
    ):
        source_path = _resolve_inside(root, str(public.get(key, "")), label=key)
        _require_exact(_sha256_path(source_path), frozen_hash, label=f"live {key} hash")

    label_protocol = raw.get("codex_label_protocol")
    if not isinstance(label_protocol, Mapping):
        raise ValueError("codex_label_protocol must be an object")
    _require_exact(label_protocol.get("reviewer"), "codex", label="Codex reviewer")
    _require_exact(
        label_protocol.get("simulator_object_name_or_geometry_allowed"),
        False,
        label="Codex label geometry policy",
    )
    _require_exact(
        label_protocol.get("relabel_after_aegis_outcome"),
        "forbidden",
        label="Codex relabel policy",
    )
    assets = raw.get("groundingdino_assets")
    if not isinstance(assets, Mapping):
        raise ValueError("groundingdino_assets must be an object")
    setup_path = _resolve_inside(
        root, str(assets.get("setup_evidence", "")), label="GroundingDINO setup evidence"
    )
    _require_exact(
        _sha256_path(setup_path),
        assets.get("setup_evidence_sha256"),
        label="GroundingDINO setup evidence hash",
    )
    _require_exact(
        assets.get("checkpoint_sha256"),
        "3b3ca2563c77c69f651d7bd133e97139c186df06231157a64c507099c52bc799",
        label="GroundingDINO checkpoint hash",
    )
    _require_exact(assets.get("glm_api_key_required"), False, label="GLM key policy")

    ready = raw.get("ready_to_run")
    if not isinstance(ready, bool):
        raise ValueError("ready_to_run must be boolean")
    release = raw.get("execution_release")
    if require_execution_release:
        if not ready:
            raise AegisApparatusError("R06 config is preregistered but not released for execution")
        if raw.get("blocked_on") not in ([], ()):
            raise AegisApparatusError("released R06 config still contains execution blockers")
        if not isinstance(release, Mapping):
            raise AegisApparatusError("released R06 config has no content-bound execution_release")

    return AegisExperimentConfig(
        path=str(path.relative_to(root)),
        sha256=_sha256_path(path),
        raw=copy.deepcopy(dict(raw)),
        manifest_path=str(manifest_path.relative_to(root)),
        manifest_sha256=FROZEN_MANIFEST_SHA256,
        cases=tuple(copy.deepcopy(dict(row)) for row in cases),
        ready_to_run=ready,
        execution_release=(copy.deepcopy(dict(release)) if isinstance(release, Mapping) else None),
    )


def _assert_allocation() -> None:
    if not os.environ.get("SLURM_JOB_ID"):
        raise AegisApparatusError("real R06 execution requires a Slurm allocation")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible or visible == "NoDevFiles":
        raise AegisApparatusError("real R06 execution requires an allocation-visible GPU")


def fixed_policy_noise(policy_seed: int):
    """Reconstruct the registered 10x32 float32 policy noise exactly."""

    if isinstance(policy_seed, bool):
        raise ValueError("policy_seed must be an integer")
    np = _numpy()
    return np.random.default_rng(int(policy_seed)).normal(size=POLICY_NOISE_SHAPE).astype(np.float32)


def _policy_reply_actions(reply: Any, *, label: str):
    np = _numpy()
    if not isinstance(reply, Mapping) or "actions" not in reply:
        raise AegisApparatusError(f"{label} policy reply has no actions")
    actions = np.asarray(reply["actions"])
    if actions.shape != POLICY_ACTION_SHAPE:
        raise AegisApparatusError(
            f"{label} policy actions must have shape {POLICY_ACTION_SHAPE}, got {actions.shape}"
        )
    if not np.issubdtype(actions.dtype, np.number) or not bool(np.isfinite(actions).all()):
        raise AegisApparatusError(f"{label} policy actions must be finite numeric values")
    return np.ascontiguousarray(actions)


def infer_duplicate_eager_actions(
    client: Any,
    observation: Mapping[str, Any],
    noise: Any,
) -> Tuple[Any, Mapping[str, Any]]:
    """Query the unchanged eager sampler twice and require exact action bytes."""

    np = _numpy()
    fixed_noise = np.asarray(noise)
    if fixed_noise.shape != POLICY_NOISE_SHAPE or fixed_noise.dtype != np.float32:
        raise ValueError("policy noise must retain exact registered shape and float32 dtype")

    replies = []
    actions = []
    for duplicate_index in range(2):
        request = copy.deepcopy(dict(observation))
        request["__crfs__"] = {
            "noise": np.array(fixed_noise, copy=True),
            "intervention_step": INTERVENTION_STEP,
            "intervention_mode": "none",
            "return_trace": True,
        }
        reply = client.infer(request)
        trace = reply.get("crfs_trace") if isinstance(reply, Mapping) else None
        if not isinstance(trace, Mapping):
            raise AegisApparatusError(
                f"duplicate eager policy reply {duplicate_index} has no CRFS trace"
            )
        replies.append(reply)
        actions.append(_policy_reply_actions(reply, label=f"duplicate {duplicate_index}"))

    exact = bool(
        actions[0].dtype == actions[1].dtype
        and actions[0].shape == actions[1].shape
        and actions[0].tobytes(order="C") == actions[1].tobytes(order="C")
    )
    if not exact:
        raise AegisApparatusError("duplicate eager policy returned different action bytes")
    return np.ascontiguousarray(actions[0][:EXECUTED_ACTION_HORIZON, :7]), {
        "duplicate_eager_actions_exact": True,
        "full_actions": exact_array_record(actions[0], label="full_policy_actions"),
        "duplicate_full_actions": exact_array_record(actions[1], label="duplicate_full_policy_actions"),
        "full_actions_values": _json_value(actions[0]),
        "trace_keys": sorted(str(key) for key in replies[0]["crfs_trace"]),
    }


def _json_value(value: Any) -> Any:
    """Convert numerical scalar/array records without accepting non-finites."""

    np = _numpy()
    if isinstance(value, np.ndarray):
        if not bool(np.isfinite(value).all()):
            raise ValueError("artifact array contains non-finite values")
        return value.tolist()
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("artifact contains a non-finite float")
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "to_dict"):
        return _json_value(value.to_dict())
    if hasattr(value, "__dataclass_fields__"):
        return _json_value(asdict(value))
    raise TypeError(f"artifact contains unsupported type {type(value).__name__}")


def _robot_geometry_from_observation(observation: Mapping[str, Any]) -> Mapping[str, Any]:
    np = _numpy()
    try:
        from scipy.spatial.transform import Rotation
    except ModuleNotFoundError as error:  # pragma: no cover - allocation dependency
        raise AegisApparatusError("literal public robot geometry requires SciPy Rotation") from error
    position = np.asarray(observation["robot0_eef_pos"], dtype=np.float64)
    quaternion_xyzw = np.asarray(observation["robot0_eef_quat"], dtype=np.float64)
    if position.shape != (3,) or quaternion_xyzw.shape != (4,):
        raise AegisApparatusError("robot observation has invalid EEF pose shape")
    rotation = Rotation.from_quat(quaternion_xyzw).as_matrix()
    center = position + rotation @ np.asarray(EEF_CENTER_OFFSET_LOCAL_M, dtype=np.float64)
    return {
        "eef_position_m": position.copy(),
        "eef_quaternion_xyzw": quaternion_xyzw.copy(),
        "R1": np.asarray(rotation, dtype=np.float64),
        "p1": np.asarray(center, dtype=np.float64),
    }


def _native_integration_state(environment: Any):
    np = _numpy()
    try:
        import mujoco
    except ModuleNotFoundError as error:  # pragma: no cover - allocation dependency
        raise AegisApparatusError("R06 execution requires modern MuJoCo") from error
    sim = environment.sim
    model = getattr(sim.model, "_model", sim.model)
    data = getattr(sim.data, "_data", sim.data)
    try:
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        size = int(mujoco.mj_stateSize(model, spec))
        state = np.empty(size, dtype=np.float64)
        mujoco.mj_getState(model, data, state, spec)
    except (AttributeError, TypeError) as error:
        raise AegisApparatusError(
            "MuJoCo runtime does not expose full mjSTATE_INTEGRATION capture"
        ) from error
    if size <= 0 or not bool(np.isfinite(state).all()):
        raise AegisApparatusError("captured MuJoCo integration state is invalid")
    return np.ascontiguousarray(state)


def _read_only_controller_state(environment: Any) -> Mapping[str, Any]:
    """Fingerprint live robosuite controller state without mutating it."""

    np = _numpy()
    inner = getattr(environment, "env", None)
    robots = getattr(inner, "robots", None)
    if not isinstance(robots, (list, tuple)) or len(robots) != 1:
        raise AegisApparatusError(
            "controller-state fingerprint requires one live robot"
        )
    robot = robots[0]
    controller = getattr(robot, "controller", None)
    gripper = getattr(robot, "gripper", None)
    if controller is None or gripper is None:
        raise AegisApparatusError(
            "controller-state fingerprint cannot resolve controller/gripper"
        )
    arrays = {}
    for record_name, attribute_name in CONTROLLER_ARRAY_ATTRIBUTES:
        raw = getattr(controller, attribute_name, None)
        if raw is None:
            raise AegisApparatusError(
                f"controller-state fingerprint is missing {attribute_name}"
            )
        arrays[record_name] = exact_array_record(
            np.asarray(raw), label=f"controller.{attribute_name}"
        )
    optional_arrays = {}
    for name in (
        "action_scale",
        "action_input_transform",
        "action_output_transform",
    ):
        raw = getattr(controller, name, None)
        optional_arrays[name] = (
            None
            if raw is None
            else exact_array_record(np.asarray(raw), label=f"controller.{name}")
        )
    record = {
        "schema_version": SCHEMA_VERSION,
        "source": "read_only_live_robosuite_controller_snapshot",
        "robot_class": type(robot).__name__,
        "robot_name": getattr(robot, "name", None),
        "controller_class": type(controller).__name__,
        "controller_name": getattr(controller, "name", None),
        "gripper_class": type(gripper).__name__,
        "new_update": getattr(controller, "new_update", None),
        "arrays": arrays,
        "optional_action_scaling_arrays": optional_arrays,
        "gripper_current_action": exact_array_record(
            np.asarray(getattr(gripper, "current_action", None)),
            label="gripper.current_action",
        ),
    }
    return {**record, "fingerprint_sha256": _content_sha256(record)}


@dataclass
class BranchContext:
    boundary_index: int
    integration_state: Any
    controller_state: Mapping[str, Any]
    camera_observation: Mapping[str, Any]
    policy_observation: Mapping[str, Any]
    prompt: str
    active_obstacle: str
    literal_pre_settle_geometry: Mapping[str, Any]
    branch_robot_geometry: Mapping[str, Any]
    branch_reach_snapshot: Any


class SafeLiberoAegisRuntime:
    """Allocation-only adapter around the additive :class:`SafeLiberoCase`.

    The wrapped environment must use 224-pixel camera observables.  Separate
    calls to ``sim.render`` provide the public 1024-pixel AEGIS views without
    modifying simulator integration state.
    """

    def __init__(self, safe_libero_case: Any) -> None:
        if int(safe_libero_case.config.resize_size) != POLICY_RENDER_SIZE:
            raise ValueError("R06 SafeLiberoCase must preserve the 224-pixel policy camera")
        if int(safe_libero_case.config.executed_prefix) != EXECUTED_ACTION_HORIZON:
            raise ValueError("R06 executes exactly five actions")
        if getattr(safe_libero_case, "_generated_source", True):
            raise ValueError("R06 frozen manifest requires released saved states")
        self.case = safe_libero_case
        self.env = safe_libero_case.env
        self._current: Optional[BranchContext] = None

    def _reset_to_boundary(self, boundary_index: int) -> Tuple[Mapping[str, Any], Mapping[str, Any]]:
        if boundary_index not in SETTLE_BOUNDARIES:
            raise ValueError("settle boundary must be in 0..20")
        environment = self.case
        environment.env.seed(environment._environment_seed)
        environment.env.reset()
        initial_observation = environment.env.set_init_state(environment._init_state.copy())
        literal = _robot_geometry_from_observation(initial_observation)
        for _ in range(boundary_index):
            environment.env.step_with_substep_callback(
                list(LIBERO_DUMMY_ACTION),
                lambda _sim, _substep: None,
                update_observables=False,
                collect_observations=False,
            )
        integration_before_render = _native_integration_state(environment.env)
        environment.env._update_observables(force=True)
        observation = environment.env.env._get_observations()
        integration_after_render = _native_integration_state(environment.env)
        np = _numpy()
        if not bool(np.array_equal(integration_before_render, integration_after_render)):
            raise AegisApparatusError("policy observation render changed simulator integration state")
        return observation, literal

    def _bind_active_obstacle(self, observation: Mapping[str, Any]) -> None:
        from .measurement import resolve_crfs_geom_groups
        from .runner import _active_obstacle

        obstacle = _active_obstacle(self.env, dict(observation))
        eef_geoms, obstacle_geoms = resolve_crfs_geom_groups(self.env, obstacle)
        self.case.obstacle_name = obstacle
        self.case.eef_geoms = eef_geoms
        self.case.obstacle_geoms = obstacle_geoms

    def _boundary_measurement(self) -> Mapping[str, Any]:
        from .measurement import GeomClearanceMonitor

        monitor = GeomClearanceMonitor(
            self.env.sim,
            self.case.eef_geoms,
            self.case.obstacle_geoms,
            distance_limit_m=DISTANCE_LIMIT_M,
            eef_site_id=int(self.env.robots[0].eef_site_id),
            eef_center_offset_local_m=EEF_CENTER_OFFSET_LOCAL_M,
            eef_radius_m=EEF_RADIUS_M,
        )
        monitor.observe(self.env.sim, -1)
        measurement = monitor.result()
        if measurement.conservative_clearance_m is None:
            raise AegisApparatusError("boundary measurement found no active obstacle box")
        return {
            "D_sim_m": float(measurement.conservative_clearance_m),
            "contact": bool(measurement.contact),
            "raw_mujoco_clearance_m": float(measurement.min_clearance_m),
            "measurement": measurement.to_dict(),
        }

    def build_settle_ledger(self) -> Mapping[str, Any]:
        """Reconstruct and measure every complete control boundary 0..20."""

        observation, literal = self._reset_to_boundary(0)
        self._bind_active_obstacle(observation)
        obstacle = str(self.case.obstacle_name)
        ledger = []
        for index in SETTLE_BOUNDARIES:
            if index > 0:
                self.env.step_with_substep_callback(
                    list(LIBERO_DUMMY_ACTION),
                    lambda _sim, _substep: None,
                    update_observables=False,
                    collect_observations=False,
                )
            measurement = self._boundary_measurement()
            state = _native_integration_state(self.env)
            ledger.append(
                {
                    "boundary_index": index,
                    **measurement,
                    "integration_state": exact_array_record(
                        state, label=f"settle_boundary_{index}_integration_state"
                    ),
                }
            )
        return {
            "ledger": ledger,
            "literal_pre_settle_geometry": _json_value(literal),
            "active_obstacle": obstacle,
        }

    def reconstruct_branch(self, boundary_index: int) -> BranchContext:
        from .reach_progress import capture_reach_snapshot
        from .runner import policy_observation

        observation, literal = self._reset_to_boundary(int(boundary_index))
        self._bind_active_obstacle(observation)
        if self.case.obstacle_name is None:
            raise AegisApparatusError("failed to resolve active obstacle")
        integration_before_controller = _native_integration_state(self.env)
        controller_state = _read_only_controller_state(self.env)
        controller_state_after = _read_only_controller_state(self.env)
        integration_state = _native_integration_state(self.env)
        np = _numpy()
        if dict(controller_state) != dict(controller_state_after):
            raise AegisApparatusError(
                "read-only controller-state fingerprint mutated live controller state"
            )
        if not bool(np.array_equal(integration_before_controller, integration_state)):
            raise AegisApparatusError(
                "read-only controller-state fingerprint changed MuJoCo integration state"
            )
        prompt = str(self.case.prompt)
        camera_observation = {}
        for camera in ("agentview", "backview"):
            for kind in ("image", "depth"):
                key = f"{camera}_{kind}"
                if key not in observation:
                    raise AegisApparatusError(
                        f"public camera observable is missing {key}"
                    )
                camera_observation[key] = np.ascontiguousarray(observation[key])
        context = BranchContext(
            boundary_index=int(boundary_index),
            integration_state=integration_state,
            controller_state=copy.deepcopy(controller_state),
            camera_observation=camera_observation,
            policy_observation=policy_observation(
                dict(observation), prompt, POLICY_RENDER_SIZE
            ),
            prompt=prompt,
            active_obstacle=str(self.case.obstacle_name),
            literal_pre_settle_geometry=literal,
            branch_robot_geometry=_robot_geometry_from_observation(observation),
            branch_reach_snapshot=capture_reach_snapshot(
                self.case, TARGET_OBJECT_NAME, str(self.case.obstacle_name)
            ),
        )
        self._current = context
        return context

    def measure_current_boundary(self) -> Mapping[str, Any]:
        if self._current is None:
            raise RuntimeError("reconstruct_branch must precede boundary measurement")
        return self._boundary_measurement()

    def render_aegis_views_1024(self) -> Mapping[str, Any]:
        """Render the exact public agent/back RGB-depth inputs without physics.

        Robosuite camera observables first apply the live convention mapping to
        the raw OpenGL render.  The public AEGIS entrypoint then applies
        ``[::-1, ::-1]``.  We reproduce both operations and fail closed unless a
        same-state 224 render maps byte-for-byte to the live observables (and
        the agent image maps to the exact policy image).
        """

        if self._current is None:
            raise RuntimeError("reconstruct_branch must precede AEGIS rendering")
        np = _numpy()
        try:
            import robosuite
            import robosuite.macros as robosuite_macros
            from importlib.metadata import version as package_version
            from robosuite.utils.mjcf_utils import IMAGE_CONVENTION_MAPPING
        except (ImportError, ModuleNotFoundError) as error:
            raise AegisApparatusError(
                "cannot resolve the robosuite camera convention mapping"
            ) from error
        convention_name = str(getattr(robosuite_macros, "IMAGE_CONVENTION", ""))
        robosuite_version = str(package_version("robosuite"))
        convention_step = IMAGE_CONVENTION_MAPPING.get(convention_name)
        declared_convention = os.environ.get(
            "R06_ROBOSUITE_IMAGE_CONVENTION", ""
        ).strip()
        if (
            robosuite_version != "1.4.1"
            or os.environ.get("R06_ROBOSUITE_VERSION", "").strip()
            != robosuite_version
            or convention_name != "opengl"
            or convention_step != 1
            or declared_convention != convention_name
        ):
            raise AegisApparatusError(
                "R06 release requires the allocation-verified robosuite opengl convention",
                evidence={
                    "live_convention": convention_name,
                    "live_robosuite_version": robosuite_version,
                    "declared_robosuite_version": os.environ.get(
                        "R06_ROBOSUITE_VERSION"
                    ),
                    "mapping_step": convention_step,
                    "declared_convention": declared_convention,
                },
            )
        before = _native_integration_state(self.env)
        views: Dict[str, Any] = {}
        anchor_views: Dict[str, Any] = {}
        for camera in ("agentview", "backview"):
            mapped_by_size = {}
            for size in (POLICY_RENDER_SIZE, AEGIS_RENDER_SIZE):
                rendered = self.env.sim.render(
                    camera_name=camera,
                    width=size,
                    height=size,
                    depth=True,
                )
                if not isinstance(rendered, tuple) or len(rendered) != 2:
                    raise AegisApparatusError(
                        f"{camera} {size} render did not return RGB and depth"
                    )
                rgb, depth = (np.asarray(item) for item in rendered)
                if rgb.shape != (size, size, 3):
                    raise AegisApparatusError(
                        f"{camera} RGB render has wrong shape {rgb.shape}"
                    )
                if depth.shape != (size, size):
                    raise AegisApparatusError(
                        f"{camera} raw depth render has wrong shape {depth.shape}"
                    )
                if not bool(np.isfinite(depth).all()):
                    raise AegisApparatusError(
                        f"{camera} depth render contains non-finite values"
                    )
                # RobotEnv.camera_rgb applies only the first-axis convention and
                # stores depth with a trailing singleton channel.
                mapped_by_size[size] = (
                    np.ascontiguousarray(rgb[::convention_step]),
                    np.ascontiguousarray(
                        np.expand_dims(depth[::convention_step], axis=-1)
                    ),
                )
            mapped_224_image, mapped_224_depth = mapped_by_size[POLICY_RENDER_SIZE]
            observable_image = np.asarray(
                self._current.camera_observation[f"{camera}_image"]
            )
            observable_depth = np.asarray(
                self._current.camera_observation[f"{camera}_depth"]
            )
            image_equal = bool(
                mapped_224_image.dtype == observable_image.dtype
                and mapped_224_image.shape == observable_image.shape
                and mapped_224_image.tobytes(order="C")
                == np.ascontiguousarray(observable_image).tobytes(order="C")
            )
            depth_equal = bool(
                mapped_224_depth.dtype == observable_depth.dtype
                and mapped_224_depth.shape == observable_depth.shape
                and mapped_224_depth.tobytes(order="C")
                == np.ascontiguousarray(observable_depth).tobytes(order="C")
            )
            if not image_equal or not depth_equal:
                raise AegisApparatusError(
                    f"{camera} raw render does not reproduce public camera observables",
                    evidence={
                        "image_equal": image_equal,
                        "depth_equal": depth_equal,
                        "convention": convention_name,
                    },
                )
            anchor_views[camera] = {
                "mapped_raw_image": exact_array_record(
                    mapped_224_image, label=f"{camera}_mapped_raw_224_image"
                ),
                "observable_image": exact_array_record(
                    observable_image, label=f"{camera}_observable_224_image"
                ),
                "mapped_raw_depth": exact_array_record(
                    mapped_224_depth, label=f"{camera}_mapped_raw_224_depth"
                ),
                "observable_depth": exact_array_record(
                    observable_depth, label=f"{camera}_observable_224_depth"
                ),
                "image_bytes_equal": True,
                "depth_bytes_equal": True,
            }
            mapped_1024_image, mapped_1024_depth = mapped_by_size[
                AEGIS_RENDER_SIZE
            ]
            views[f"{camera}_image"] = np.ascontiguousarray(
                mapped_1024_image[::-1, ::-1]
            )
            views[f"{camera}_depth"] = np.ascontiguousarray(
                mapped_1024_depth[::-1, ::-1]
            )
        policy_agent = np.asarray(
            self._current.policy_observation["observation/image"]
        )
        public_agent_224 = np.ascontiguousarray(
            self._current.camera_observation["agentview_image"][::-1, ::-1]
        )
        policy_anchor_equal = bool(
            public_agent_224.dtype == policy_agent.dtype
            and public_agent_224.shape == policy_agent.shape
            and public_agent_224.tobytes(order="C")
            == np.ascontiguousarray(policy_agent).tobytes(order="C")
        )
        if not policy_anchor_equal:
            raise AegisApparatusError(
                "same-state public agent image does not match policy observation bytes"
            )
        after = _native_integration_state(self.env)
        if not bool(np.array_equal(before, after)):
            raise AegisApparatusError("1024 AEGIS rendering changed simulator integration state")
        return {
            **views,
            "integration_state_before": exact_array_record(before, label="perception_state_before"),
            "integration_state_after": exact_array_record(after, label="perception_state_after"),
            "render_advanced_physics": False,
            "public_double_axis_flip_applied": True,
            "camera_orientation": {
                "robosuite_version": robosuite_version,
                "live_image_convention": convention_name,
                "image_convention_mapping_step": int(convention_step),
                "observable_mapping_applied_before_public_flip": True,
                "same_state_224_observable_anchor_passed": True,
                "same_state_224_policy_agent_anchor_passed": True,
                "anchors": anchor_views,
                "policy_agent_image": exact_array_record(
                    policy_agent, label="policy_agent_224_image"
                ),
                "public_agent_image": exact_array_record(
                    public_agent_224, label="public_agent_224_image"
                ),
                "controller_state_fingerprint_sha256": self._current.controller_state[
                    "fingerprint_sha256"
                ],
            },
        }

    def _current_robot_observation(self) -> Mapping[str, Any]:
        before = _native_integration_state(self.env)
        self.env._update_observables(force=True)
        observation = self.env.env._get_observations()
        after = _native_integration_state(self.env)
        np = _numpy()
        if not bool(np.array_equal(before, after)):
            raise AegisApparatusError("post-action observation changed integration state")
        return observation

    def execute_actions(
        self,
        nominal_actions: Any,
        *,
        controller: Optional[Any] = None,
    ) -> Mapping[str, Any]:
        """Execute one already-reconstructed branch with 126 measurement samples."""

        if self._current is None:
            raise RuntimeError("reconstruct_branch must precede action execution")
        np = _numpy()
        actions = np.asarray(nominal_actions)
        if actions.shape != EXECUTED_ACTION_SHAPE or not bool(np.isfinite(actions).all()):
            raise ValueError(f"nominal_actions must be finite with shape {EXECUTED_ACTION_SHAPE}")

        from .measurement import GeomClearanceMonitor
        from .reach_progress import annotate_reach_snapshots, capture_reach_snapshot

        target_body_id = int(self.env.env.obj_body_id[TARGET_OBJECT_NAME])
        obstacle_name = str(self._current.active_obstacle)
        obstacle_body_id = int(self.env.env.obj_body_id[obstacle_name])
        branch_target = np.asarray(self.env.sim.data.body_xpos[target_body_id], dtype=np.float64).copy()
        branch_obstacle = np.asarray(self.env.sim.data.body_xpos[obstacle_body_id], dtype=np.float64).copy()
        branch_snapshot = capture_reach_snapshot(self.case, TARGET_OBJECT_NAME, obstacle_name)
        eef_site_id = int(self.env.robots[0].eef_site_id)
        eef_substep_trajectory = [
            np.asarray(self.env.sim.data.site_xpos[eef_site_id], dtype=np.float64).copy()
        ]
        eef_control_boundary_trajectory = [eef_substep_trajectory[0].copy()]
        target_max = 0.0
        obstacle_max = 0.0
        legacy_obstacle_l1_max = 0.0

        monitor = GeomClearanceMonitor(
            self.env.sim,
            self.case.eef_geoms,
            self.case.obstacle_geoms,
            distance_limit_m=DISTANCE_LIMIT_M,
            eef_site_id=eef_site_id,
            eef_center_offset_local_m=EEF_CENTER_OFFSET_LOCAL_M,
            eef_radius_m=EEF_RADIUS_M,
        )
        monitor.observe(self.env.sim, -1)
        executed = []
        controller_records = []
        task_success_during = False
        task_success_end = False

        for action_index, nominal in enumerate(actions):
            action = np.asarray(nominal, dtype=np.float64).copy()
            controller_record: Optional[Mapping[str, Any]] = None
            if controller is not None:
                try:
                    parameters = inspect.signature(
                        controller.filter_action
                    ).parameters
                    if "step_index" in parameters:
                        filtered = controller.filter_action(
                            nominal_action=action.copy(), step_index=action_index
                        )
                    else:
                        filtered = controller.filter_action(action.copy())
                except AegisMethodFailure:
                    raise
                except Exception as error:
                    translated = _translate_core_error(error, stage="filter_action")
                    translated.evidence.update(
                        {
                            "failed_action_index": action_index,
                            "executed_actions_before_failure": _json_value(executed),
                            "controller_records_before_failure": _json_value(controller_records),
                        }
                    )
                    raise translated from error
                if isinstance(filtered, Mapping):
                    filtered_action = filtered.get("action")
                    telemetry = filtered.get("telemetry", filtered.get("qp"))
                else:
                    filtered_action = getattr(filtered, "action", None)
                    telemetry = getattr(filtered, "telemetry", None)
                if filtered_action is None or not isinstance(telemetry, Mapping):
                    raise AegisMethodFailure(
                        "qp_failure",
                        "AEGIS filter_action did not return action and telemetry records",
                        evidence={"failed_action_index": action_index},
                    )
                action = np.asarray(filtered_action, dtype=np.float64)
                if action.shape != (7,) or not bool(np.isfinite(action).all()):
                    raise AegisMethodFailure(
                        "qp_failure",
                        "AEGIS returned a nonfinite or wrong-shaped action",
                        evidence={"failed_action_index": action_index},
                    )
                if float(action[6]) != float(nominal[6]):
                    raise AegisMethodFailure(
                        "gripper_changed",
                        "public AEGIS must preserve the nominal gripper command",
                        evidence={"failed_action_index": action_index},
                    )
                controller_record = {
                    "qp": dict(telemetry),
                }

            def observe_substep(sim: Any, substep_index: int, *, offset=action_index * PHYSICS_SUBSTEPS_PER_ACTION):
                nonlocal target_max, obstacle_max
                monitor.observe(sim, offset + int(substep_index))
                eef_substep_trajectory.append(
                    np.asarray(
                        sim.data.site_xpos[eef_site_id], dtype=np.float64
                    ).copy()
                )
                target_position = np.asarray(sim.data.body_xpos[target_body_id], dtype=np.float64)
                obstacle_position = np.asarray(sim.data.body_xpos[obstacle_body_id], dtype=np.float64)
                target_max = max(target_max, float(np.linalg.norm(target_position - branch_target)))
                obstacle_max = max(
                    obstacle_max, float(np.linalg.norm(obstacle_position - branch_obstacle))
                )

            _, _, done, _ = self.env.step_with_substep_callback(
                action.tolist(),
                observe_substep,
                update_observables=False,
                collect_observations=False,
            )
            observation = self._current_robot_observation()
            task_success_end = bool(done or self.env.check_success())
            task_success_during = bool(task_success_during or task_success_end)
            executed.append(action.copy())
            eef_control_boundary_trajectory.append(
                np.asarray(self.env.sim.data.site_xpos[eef_site_id], dtype=np.float64).copy()
            )
            legacy_obstacle_l1_max = max(
                legacy_obstacle_l1_max,
                float(
                    np.abs(
                        np.asarray(self.env.sim.data.body_xpos[obstacle_body_id], dtype=np.float64)
                        - branch_obstacle
                    ).sum()
                ),
            )
            if controller is not None:
                geometry = _robot_geometry_from_observation(observation)
                try:
                    if hasattr(controller, "observe_executed_state"):
                        refresh = controller.observe_executed_state(
                            robot_geometry=copy.deepcopy(geometry),
                            step_index=action_index,
                        )
                    elif hasattr(controller, "refresh_after_action"):
                        refresh = controller.refresh_after_action(
                            eef_position=copy.deepcopy(geometry["eef_position_m"]),
                            eef_rotation_matrix=copy.deepcopy(geometry["R1"]),
                        )
                    else:
                        raise AttributeError(
                            "controller exposes no post-action robot-geometry refresh"
                        )
                except Exception as error:
                    translated = _translate_core_error(
                        error, stage="controller_state_update"
                    )
                    translated.evidence.update({"failed_action_index": action_index})
                    raise translated from error
                controller_records.append(
                    {
                        "step_index": action_index,
                        "nominal_action": _json_value(nominal),
                        "executed_action": _json_value(action),
                        "post_action_refresh": _json_value(refresh),
                        **_json_value(controller_record or {}),
                    }
                )

        measurement = monitor.result()
        if measurement.samples != MEASUREMENT_SAMPLES_PER_ARM:
            raise AegisApparatusError(
                f"arm measured {measurement.samples}, expected {MEASUREMENT_SAMPLES_PER_ARM} samples"
            )
        if len(eef_substep_trajectory) != MEASUREMENT_SAMPLES_PER_ARM:
            raise AegisApparatusError(
                "arm did not record the exact initial-plus-125-substep EEF trajectory"
            )
        if measurement.conservative_clearance_m is None:
            raise AegisApparatusError("arm has no registered D_sim measurement")
        final_snapshot = capture_reach_snapshot(self.case, TARGET_OBJECT_NAME, obstacle_name)
        reach = annotate_reach_snapshots(
            branch_snapshot,
            final_snapshot,
            executed_actions=EXECUTED_ACTION_HORIZON,
            maximum_target_displacement_m=target_max,
            maximum_active_obstacle_displacement_m=obstacle_max,
        ).to_dict()
        result = {
            "nominal_actions": _json_value(actions),
            "executed_actions": _json_value(np.asarray(executed)),
            "eef_trajectory_m": _json_value(
                np.asarray(eef_substep_trajectory)
            ),
            "eef_control_boundary_trajectory_m": _json_value(
                np.asarray(eef_control_boundary_trajectory)
            ),
            "measurement_samples": int(measurement.samples),
            "contact": bool(measurement.contact),
            "minimum_D_sim_m": float(measurement.conservative_clearance_m),
            "raw_mujoco_minimum_clearance_m": float(measurement.min_clearance_m),
            "measurement": measurement.to_dict(),
            "reach": reach,
            "task_completed_during_prefix": bool(task_success_during),
            "task_completed_at_end": bool(task_success_end),
            "legacy_public_obstacle_l1_max_m": legacy_obstacle_l1_max,
            "legacy_public_collision_flag": bool(legacy_obstacle_l1_max > 0.001),
            "controller_steps": controller_records,
        }
        return _json_value(result)

    def close(self) -> None:
        self.case.close()


def _branch_pairing(context: BranchContext, noise: Any, nominal_actions: Any) -> Mapping[str, Any]:
    return {
        **build_pairing_record(
        branch_state=context.integration_state,
        observation=context.policy_observation,
        policy_noise=noise,
        nominal_actions=nominal_actions,
        executed_action_horizon=EXECUTED_ACTION_HORIZON,
        ),
        "controller_state": copy.deepcopy(dict(context.controller_state)),
    }


def _r02_array_hash(value: Any) -> str:
    np = _numpy()
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _r02_observation_fingerprint(value: Mapping[str, Any]) -> Mapping[str, Any]:
    np = _numpy()
    digest = hashlib.sha256()
    leaves = []
    for key in sorted(value):
        item = value[key]
        if isinstance(item, str):
            payload = item.encode("utf-8")
            record = {
                "key": key,
                "kind": "string",
                "sha256": hashlib.sha256(payload).hexdigest(),
                "value": item,
            }
        else:
            array = np.ascontiguousarray(np.asarray(item))
            record = {
                "key": key,
                "kind": "array",
                "dtype": str(array.dtype),
                "shape": list(array.shape),
                "sha256": _r02_array_hash(array),
            }
        framed = _canonical_json_bytes(record)
        digest.update(len(framed).to_bytes(8, "big"))
        digest.update(framed)
        leaves.append(record)
    return {"sha256": digest.hexdigest(), "leaves": leaves}


def _validate_prior_r02_bytes(
    case_id: str,
    *,
    observation: Mapping[str, Any],
    nominal_actions: Any,
) -> Mapping[str, Any]:
    """Bind the standard-settled branch to the accepted raw R02 bytes."""

    np = _numpy()
    path = R02_RESULTS_ROOT / str(case_id) / "r02-paired.json"
    if not path.is_file():
        raise AegisApparatusError(
            "accepted raw R02 pairing artifact is missing",
            evidence={"path": str(path)},
        )
    value = _read_json(path)
    live_sha256 = _sha256_path(path)
    if case_id == CANARY_CASE_ID and live_sha256 != CANARY_R02_SHA256:
        raise AegisApparatusError(
            "accepted canary R02 artifact hash changed",
            evidence={"expected": CANARY_R02_SHA256, "actual": live_sha256},
        )
    provenance = value.get("provenance")
    pairing = value.get("pairing")
    if not isinstance(provenance, Mapping) or not isinstance(pairing, Mapping):
        raise AegisApparatusError("accepted raw R02 artifact lacks provenance/pairing")
    checks = {
        "case_id": value.get("case_id") == case_id,
        "checkpoint": provenance.get("checkpoint_sha256") == CHECKPOINT_SHA256,
        "normalization": (
            provenance.get("normalization_asset_sha256")
            == NORMALIZATION_ASSET_SHA256
        ),
        "observation": (
            pairing.get("policy_observation")
            == _r02_observation_fingerprint(observation)
        ),
    }
    eager = pairing.get("eager_actions")
    if isinstance(eager, Mapping) and "values" in eager:
        historical = np.asarray(eager["values"], dtype=np.dtype(str(eager["dtype"])))
        current = np.asarray(nominal_actions)
        historical_prefix = np.ascontiguousarray(historical[:5, :7])
        current = np.ascontiguousarray(current)
        checks["nominal_actions"] = bool(
            historical.shape == POLICY_ACTION_SHAPE
            and eager.get("shape") == list(POLICY_ACTION_SHAPE)
            and eager.get("sha256") == _r02_array_hash(historical)
            and historical_prefix.shape == current.shape
            and historical_prefix.dtype == current.dtype
            and historical_prefix.tobytes(order="C") == current.tobytes(order="C")
        )
    else:
        checks["nominal_actions"] = False
    if not all(checks.values()):
        raise AegisApparatusError(
            "current branch observation/action bytes differ from accepted R02",
            evidence={"path": str(path), "checks": checks},
        )
    return {
        "path": str(path),
        "sha256": live_sha256,
        "checks": checks,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "normalization_asset_sha256": NORMALIZATION_ASSET_SHA256,
        "policy_observation_sha256": pairing["policy_observation"]["sha256"],
        "eager_actions_sha256": eager["sha256"],
    }


def _validate_runtime_model_files() -> Mapping[str, Any]:
    records = {}
    for label, path, expected in (
        ("checkpoint_model", CHECKPOINT_MODEL_PATH, CHECKPOINT_SHA256),
        ("checkpoint_config", CHECKPOINT_CONFIG_PATH, CHECKPOINT_CONFIG_SHA256),
        ("normalization_asset", NORMALIZATION_ASSET_PATH, NORMALIZATION_ASSET_SHA256),
    ):
        if not path.is_file():
            raise AegisApparatusError(
                f"{label} is missing", evidence={"path": str(path)}
            )
        actual = _sha256_path(path)
        if actual != expected:
            raise AegisApparatusError(
                f"{label} hash changed",
                evidence={"path": str(path), "expected": expected, "actual": actual},
            )
        records[label] = {"path": str(path), "sha256": actual}
    return records


def _implementation_identity(
    *,
    runtime: Any,
    provider: Optional[Any],
    release: Any,
) -> Mapping[str, Any]:
    root = Path(__file__).resolve().parents[2]
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root, text=True
        ).strip()
    )
    expected_commit = os.environ.get("EXPECTED_GIT_COMMIT", "").strip()
    if (
        len(expected_commit) != 40
        or any(character not in "0123456789abcdef" for character in expected_commit)
    ):
        raise AegisApparatusError(
            "R06 workload must export its exact EXPECTED_GIT_COMMIT"
        )
    if not isinstance(release, Mapping):
        raise AegisApparatusError("R06 execution release must be a mapping")
    accepted = str(release.get("accepted_implementation_commit", ""))
    allowed_paths = release.get("allowed_release_diff_paths")
    if (
        not _is_lower_hex(accepted, 40)
        or release.get("release_only_parent_required") is not True
        or not isinstance(allowed_paths, list)
        or len(allowed_paths) < 2
        or any(not isinstance(path, str) or not path for path in allowed_paths)
        or len(set(allowed_paths)) != len(allowed_paths)
    ):
        raise AegisApparatusError(
            "R06 release lacks the registered two-commit release identity"
        )
    parents = subprocess.check_output(
        ["git", "rev-list", "--parents", "-n", "1", expected_commit],
        cwd=root,
        text=True,
    ).strip().split()
    diff_paths = sorted(
        line
        for line in subprocess.check_output(
            ["git", "diff", "--name-only", accepted, expected_commit],
            cwd=root,
            text=True,
        ).splitlines()
        if line
    )
    if (
        commit != expected_commit
        or dirty
        or parents != [expected_commit, accepted]
        or diff_paths != sorted(allowed_paths)
    ):
        raise AegisApparatusError(
            "R06 execution source differs from its clean release commit",
            evidence={
                "expected_commit": expected_commit,
                "accepted_implementation_commit": accepted,
                "actual_commit": commit,
                "dirty": dirty,
                "release_parents": parents,
                "release_diff_paths": diff_paths,
                "allowed_release_diff_paths": sorted(allowed_paths),
            },
        )
    source_paths = {
        "runner": Path(__file__).resolve(),
        "pairing": Path(__file__).with_name("aegis_pairing.py"),
        "baseline_adapter": Path(__file__).with_name("aegis_baseline.py"),
        "perception_adapter": Path(__file__).with_name("aegis_perception.py"),
    }
    return {
        "source_git_commit": commit,
        "expected_git_commit": expected_commit,
        "accepted_implementation_commit": accepted,
        "release_direct_child_verified": True,
        "release_diff_paths": diff_paths,
        "allowed_release_diff_paths": sorted(allowed_paths),
        "git_dirty": False,
        "runtime_type": f"{type(runtime).__module__}.{type(runtime).__qualname__}",
        "provider_type": (
            None
            if provider is None
            else f"{type(provider).__module__}.{type(provider).__qualname__}"
        ),
        "source_sha256": {
            name: _sha256_path(path) for name, path in source_paths.items()
        },
    }


def _implementation_identity_valid(value: Any, *, capture: bool) -> bool:
    if not isinstance(value, Mapping) or value.get("git_dirty") is not False:
        return False
    expected_runtime = f"{SafeLiberoAegisRuntime.__module__}.{SafeLiberoAegisRuntime.__qualname__}"
    expected_provider = (
        None
        if capture
        else (
            f"{CodexFrozenLabelSafetyCoreProvider.__module__}."
            f"{CodexFrozenLabelSafetyCoreProvider.__qualname__}"
        )
    )
    if (
        value.get("runtime_type") != expected_runtime
        or value.get("provider_type") != expected_provider
        or not _is_lower_hex(value.get("source_git_commit"), 40)
        or value.get("expected_git_commit") != value.get("source_git_commit")
        or not _is_lower_hex(value.get("accepted_implementation_commit"), 40)
        or value.get("accepted_implementation_commit")
        == value.get("source_git_commit")
        or value.get("release_direct_child_verified") is not True
        or not isinstance(value.get("release_diff_paths"), list)
        or value.get("release_diff_paths")
        != value.get("allowed_release_diff_paths")
    ):
        return False
    sources = value.get("source_sha256")
    if not isinstance(sources, Mapping):
        return False
    expected_paths = {
        "runner": Path(__file__).resolve(),
        "pairing": Path(__file__).with_name("aegis_pairing.py"),
        "baseline_adapter": Path(__file__).with_name("aegis_baseline.py"),
        "perception_adapter": Path(__file__).with_name("aegis_perception.py"),
    }
    return all(sources.get(name) == _sha256_path(path) for name, path in expected_paths.items())


def _validate_selected_ledger_reference(
    settle_evidence: Mapping[str, Any],
    selection: Any,
    reference: BranchContext,
    current_measurement: Mapping[str, Any],
) -> Mapping[str, Any]:
    rows = settle_evidence.get("ledger")
    if not isinstance(rows, Sequence):
        raise AegisApparatusError("settle evidence has no complete ledger")
    selected_rows = [
        row
        for row in rows
        if isinstance(row, Mapping)
        and int(row.get("boundary_index", -1)) == selection.selected_boundary_index
    ]
    if len(selected_rows) != 1:
        raise AegisApparatusError("settle ledger has no unique selected row")
    row = selected_rows[0]
    checks = {
        "integration_state": (
            row.get("integration_state")
            == exact_array_record(
                reference.integration_state,
                label=f"settle_boundary_{selection.selected_boundary_index}_integration_state",
            )
        ),
        "active_obstacle": (
            settle_evidence.get("active_obstacle") == reference.active_obstacle
        ),
        "clearance": (
            float(row.get("D_sim_m")) == float(current_measurement.get("D_sim_m"))
        ),
        "contact": (
            bool(row.get("contact")) == bool(current_measurement.get("contact"))
        ),
    }
    if not all(checks.values()):
        raise AegisApparatusError(
            "selected settle ledger row differs from fresh branch reconstruction",
            evidence={"checks": checks},
        )
    return {
        "passed": True,
        "checks": checks,
        "selected_integration_state": row["integration_state"],
        "selected_active_obstacle": reference.active_obstacle,
        "selected_D_sim_m": float(row["D_sim_m"]),
        "selected_contact": bool(row["contact"]),
    }


def _validate_branch_identity(reference: BranchContext, actual: BranchContext) -> Sequence[str]:
    errors = []
    if actual.boundary_index != reference.boundary_index:
        errors.append("settle boundary differs")
    if actual.prompt != reference.prompt:
        errors.append("instruction differs")
    if actual.active_obstacle != reference.active_obstacle:
        errors.append("active obstacle differs")
    if dict(actual.controller_state) != dict(reference.controller_state):
        errors.append("canonical controller-state fingerprint differs")
    if _json_value(actual.literal_pre_settle_geometry) != _json_value(
        reference.literal_pre_settle_geometry
    ):
        errors.append("literal pre-settle p1/R1 differs")
    return errors


def _arm_metrics(case_id: str, execution: Mapping[str, Any], baseline: Mapping[str, Any]) -> Mapping[str, Any]:
    nominal = execution["nominal_actions"]
    executed = execution["executed_actions"]
    baseline_eef = baseline["eef_trajectory_m"]
    arm_eef = execution["eef_trajectory_m"]
    progress = float(execution["reach"]["reach_progress_m"])
    stop = stopping_diagnostic(
        nominal_actions=nominal,
        executed_actions=executed,
        baseline_eef_trajectory_m=baseline_eef,
        arm_eef_trajectory_m=arm_eef,
        reach_progress_m=progress,
    )
    reduced = reduce_case_metrics(
        case_id,
        contact=bool(execution["contact"]),
        minimum_clearance_m=float(execution["minimum_D_sim_m"]),
        reach_progress_m=progress,
        task_completed_during_prefix=bool(execution["task_completed_during_prefix"]),
        task_completed_at_end=bool(execution["task_completed_at_end"]),
        stopping=stop,
    )
    return {
        **execution,
        "outcomes": reduced,
        "frozen_collision_definition": bool(
            execution["contact"] or float(execution["minimum_D_sim_m"]) < 0.0
        ),
    }


def _exact_executions(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return _content_sha256(left) == _content_sha256(right)


def _perception_array_records(views: Mapping[str, Any]) -> Mapping[str, Any]:
    required = (
        "agentview_image",
        "agentview_depth",
        "backview_image",
        "backview_depth",
    )
    records = {}
    for name in required:
        if name not in views:
            raise AegisApparatusError(f"AEGIS perception render is missing {name}")
        records[name] = exact_array_record(views[name], label=name)
    return {
        "render_size": AEGIS_RENDER_SIZE,
        "views": records,
        "integration_state_before": views.get("integration_state_before"),
        "integration_state_after": views.get("integration_state_after"),
        "render_advanced_physics": views.get("render_advanced_physics"),
        "public_double_axis_flip_applied": views.get("public_double_axis_flip_applied"),
        "camera_orientation": copy.deepcopy(views.get("camera_orientation")),
    }


class AegisSafetyCoreProvider:
    """Strict bridge from an end-to-end perception backend to AegisSafetyCore.

    The callback is intentionally small and injectable because the public
    perception code has heavyweight allocation-only dependencies.  It must run
    GLM-4.5V, two-view GroundingDINO, public filtering, and MVEE and return the
    resulting obstacle ellipsoid plus audit records.  This bridge never accepts
    a fixed label or simulator geometry.
    """

    def __init__(
        self,
        perception_backend: Callable[..., Mapping[str, Any]],
        *,
        repo_root: str | Path,
        expected_dino_config_sha256: str,
        expected_dino_checkpoint_sha256: str,
    ) -> None:
        if not callable(perception_backend):
            raise TypeError("perception_backend must be callable")
        for label, digest in (
            ("expected_dino_config_sha256", expected_dino_config_sha256),
            ("expected_dino_checkpoint_sha256", expected_dino_checkpoint_sha256),
        ):
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError(f"{label} must be a lowercase SHA-256 digest")
        self._backend = perception_backend
        self._repo_root = Path(repo_root).resolve()
        self._expected_dino_config_sha256 = expected_dino_config_sha256
        self._expected_dino_checkpoint_sha256 = expected_dino_checkpoint_sha256

    def prepare_perception(self, **kwargs: Any) -> Mapping[str, Any]:
        from .aegis_baseline import verify_upstream_sources
        from .aegis_perception import (
            resolve_original_perception_dependencies,
        )

        dependencies = resolve_original_perception_dependencies(
            os.environ,
            expected_config_sha256=self._expected_dino_config_sha256,
            expected_checkpoint_sha256=self._expected_dino_checkpoint_sha256,
            require_packages=True,
            require_cuda=True,
        )
        sources = verify_upstream_sources(self._repo_root)
        result = self._backend(
            dependencies=dependencies,
            repo_root=self._repo_root,
            **kwargs,
        )
        if not isinstance(result, Mapping):
            raise AegisApparatusError("end-to-end perception backend returned no record")
        if result.get("status") != "ok":
            raise AegisMethodFailure(
                "perception_failure",
                "public AEGIS perception did not produce a usable obstacle ellipsoid",
                evidence={"perception": _json_value(result)},
            )
        required = {
            "semantic_label_source": "zhipu_glm_4_5v",
            "oracle_substitution_used": False,
        }
        for key, expected in required.items():
            if result.get(key) != expected:
                raise AegisApparatusError(
                    f"perception backend violates {key}={expected!r}"
                )
        label = result.get("obstacle_label")
        if not isinstance(label, str) or not label.strip():
            raise AegisMethodFailure(
                "perception_failure",
                "GLM-4.5V did not return a non-empty obstacle label",
                evidence={"perception": _json_value(result)},
            )
        glm = result.get("glm")
        dino = result.get("groundingdino")
        ellipsoid = result.get("ellipsoid")
        if not isinstance(glm, Mapping) or not isinstance(dino, Mapping):
            raise AegisApparatusError("perception record lacks GLM/GroundingDINO audit data")
        for key, expected in (
            ("model", "glm-4.5v"),
            ("temperature", 0.1),
            ("top_p", 0.1),
        ):
            if glm.get(key) != expected:
                raise AegisApparatusError(f"GLM audit record has wrong {key}")
        if glm.get("thinking") not in ("enabled", {"type": "enabled"}):
            raise AegisApparatusError("GLM audit record has wrong thinking setting")
        for key, expected in (
            ("box_threshold", 0.35),
            ("text_threshold", 0.25),
        ):
            if dino.get(key) != expected:
                raise AegisApparatusError(f"GroundingDINO audit record has wrong {key}")
        if not isinstance(ellipsoid, Mapping):
            raise AegisMethodFailure(
                "perception_failure",
                "public filtering/MVEE produced no ellipsoid",
                evidence={"perception": _json_value(result)},
            )
        for key in ("p2", "q2_diag", "r2"):
            if key not in ellipsoid:
                raise AegisMethodFailure(
                    "perception_failure",
                    f"public MVEE result is missing {key}",
                    evidence={"perception": _json_value(result)},
                )
        backend_name = (
            f"{getattr(self._backend, '__module__', type(self._backend).__module__)}."
            f"{getattr(self._backend, '__qualname__', type(self._backend).__qualname__)}"
        )
        return {
            **_json_value(result),
            "perception_dependencies": dependencies.to_dict(),
            "upstream_source_sha256": dict(sources),
            "backend": backend_name,
            "backend_kind": "original_end_to_end_only",
        }

    def new_core(
        self,
        *,
        perception: Mapping[str, Any],
        literal_pre_settle_geometry: Mapping[str, Any],
        branch_robot_geometry: Mapping[str, Any],
        instruction: str,
    ) -> Any:
        from .aegis_baseline import (
            AegisSafetyCore,
            INITIALIZATION_PRE_SETTLE_UPSTREAM,
            PerceptionContract,
            PERCEPTION_ORIGINAL_END_TO_END,
            upstream_eef_q_diag,
        )

        if perception.get("backend_kind") != "original_end_to_end_only":
            raise AegisApparatusError("controller received non-original perception")
        ellipsoid = perception.get("ellipsoid")
        if not isinstance(ellipsoid, Mapping):
            raise AegisMethodFailure(
                "perception_failure", "controller has no public MVEE ellipsoid"
            )
        dependency = perception.get("perception_dependencies")
        if not isinstance(dependency, Mapping):
            raise AegisApparatusError("controller has no bound perception dependencies")
        contract = PerceptionContract(
            mode=PERCEPTION_ORIGINAL_END_TO_END,
            semantic_label_source="zhipu_glm_4_5v",
            obstacle_label=str(perception.get("obstacle_label", "")),
            dino_config_path=str(dependency.get("config_path", "")),
            dino_checkpoint_path=str(dependency.get("checkpoint_path", "")),
            zhipu_key_present=bool(dependency.get("api_key_present")),
            runtime_dependencies_checked=bool(
                dependency.get("packages")
                and all(bool(value) for value in dependency["packages"].values())
            ),
        )
        return AegisSafetyCore(
            pre_settle_p1=literal_pre_settle_geometry["p1"],
            pre_settle_r1=literal_pre_settle_geometry["R1"],
            branch_p1=branch_robot_geometry["p1"],
            branch_r1=branch_robot_geometry["R1"],
            q1_diag=upstream_eef_q_diag(instruction),
            p2=ellipsoid["p2"],
            q2_diag=ellipsoid["q2_diag"],
            r2=ellipsoid["r2"],
            perception=contract,
            initialization_source=INITIALIZATION_PRE_SETTLE_UPSTREAM,
            repo_root=self._repo_root,
        )


class CodexFrozenLabelSafetyCoreProvider:
    """Release provider for the ADR-0065 Codex-label diagnostic arm."""

    def __init__(
        self,
        *,
        label_jsonl_path: str | Path,
        label_jsonl_sha256: str,
        output_dir: str | Path,
        repo_root: str | Path,
    ) -> None:
        if (
            not isinstance(label_jsonl_sha256, str)
            or len(label_jsonl_sha256) != 64
            or any(character not in "0123456789abcdef" for character in label_jsonl_sha256)
        ):
            raise ValueError("label_jsonl_sha256 must be a lowercase SHA-256")
        self._label_path = Path(label_jsonl_path).expanduser().resolve()
        self._label_sha256 = label_jsonl_sha256
        self._output_dir = Path(output_dir).expanduser().resolve()
        self._repo_root = Path(repo_root).resolve()

    def prepare_perception(self, **kwargs: Any) -> Mapping[str, Any]:
        from .aegis_baseline import verify_upstream_sources
        from .aegis_perception import (
            AegisPerceptionApparatusError,
            AegisPerceptionMethodError,
            run_codex_frozen_label_groundingdino_backend,
        )

        try:
            sources = verify_upstream_sources(self._repo_root)
            result = run_codex_frozen_label_groundingdino_backend(
                label_jsonl_path=self._label_path,
                expected_label_jsonl_sha256=self._label_sha256,
                case_id=str(kwargs.pop("case_id")),
                views=kwargs.pop("views"),
                instruction=str(kwargs.pop("instruction")),
                task_suite=str(kwargs.pop("task_suite")),
                runtime=kwargs.pop("runtime"),
                environment=os.environ,
                output_dir=self._output_dir,
                repo_root=self._repo_root,
            )
            if result.get("public_utils", {}).get("sha256") != sources.get(
                "main/utils.py"
            ):
                raise AegisApparatusError(
                    "perception did not bind the verified upstream utils source"
                )
            result["upstream_source_sha256"] = sources
            return result
        except AegisPerceptionMethodError as error:
            raise AegisMethodFailure(
                error.code, str(error), evidence=error.to_dict()
            ) from error
        except AegisPerceptionApparatusError as error:
            raise AegisApparatusError(
                str(error), evidence=error.to_dict()
            ) from error

    @property
    def label_jsonl_sha256(self) -> str:
        return self._label_sha256

    @property
    def label_jsonl_path(self) -> str:
        return str(self._label_path)

    def new_core(
        self,
        *,
        perception: Mapping[str, Any],
        literal_pre_settle_geometry: Mapping[str, Any],
        branch_robot_geometry: Mapping[str, Any],
        instruction: str,
    ) -> Any:
        from .aegis_baseline import (
            AegisSafetyCore,
            INITIALIZATION_PRE_SETTLE_UPSTREAM,
            PERCEPTION_FROZEN_LABEL_CORE,
            resolve_perception_mode,
            upstream_eef_q_diag,
        )
        from .aegis_perception import CODEX_FROZEN_LABEL_BACKEND_KIND

        if (
            perception.get("backend_kind") != CODEX_FROZEN_LABEL_BACKEND_KIND
            or perception.get("release_runtime_backend") is not True
            or perception.get("test_injection_used") is not False
            or perception.get("original_glm_executed") is not False
        ):
            raise AegisApparatusError(
                "controller requires the release Codex-label GroundingDINO backend"
            )
        ellipsoid = perception.get("ellipsoid")
        if not isinstance(ellipsoid, Mapping):
            raise AegisMethodFailure("perception_failure", "public MVEE ellipsoid is missing")
        contract = resolve_perception_mode(
            PERCEPTION_FROZEN_LABEL_CORE,
            obstacle_label=str(perception.get("obstacle_label", "")),
            repo_root=self._repo_root,
        )
        return AegisSafetyCore(
            pre_settle_p1=literal_pre_settle_geometry["p1"],
            pre_settle_r1=literal_pre_settle_geometry["R1"],
            branch_p1=branch_robot_geometry["p1"],
            branch_r1=branch_robot_geometry["R1"],
            q1_diag=upstream_eef_q_diag(instruction),
            p2=ellipsoid["p2"],
            q2_diag=ellipsoid["q2_diag"],
            r2=ellipsoid["r2"],
            perception=contract,
            initialization_source=INITIALIZATION_PRE_SETTLE_UPSTREAM,
            repo_root=self._repo_root,
        )


def _method_failure_payload(
    base: Mapping[str, Any],
    error: AegisMethodFailure,
) -> Mapping[str, Any]:
    return {
        **base,
        "status": "method_failure",
        "failure": {
            "classification": "retained_method_failure",
            "kind": error.kind,
            "message": str(error),
            "evidence": _json_value(error.evidence),
            "silent_fallback_used": False,
        },
        "canary_apparatus_valid": False,
        "claim_scope": "collision_conditioned_diagnostic_only",
    }


def _apparatus_failure_payload(
    base: Mapping[str, Any],
    error: Exception,
) -> Mapping[str, Any]:
    evidence = error.evidence if isinstance(error, AegisApparatusError) else {}
    return {
        **base,
        "status": "apparatus_invalid",
        "failure": {
            "classification": "apparatus_invalid",
            "kind": type(error).__name__,
            "message": str(error),
            "evidence": _json_value(evidence),
            "silent_fallback_used": False,
        },
        "canary_apparatus_valid": False,
        "claim_scope": "no_scientific_result",
    }


def _validate_codex_perception_record(
    value: Any,
    *,
    case_id: str,
    instruction: str,
    image_sha256: str,
    label_manifest_sha256: Optional[str] = None,
    capture_completed_at: Optional[str] = None,
) -> bool:
    """Independently validate release Codex-label/DINO/filter/MVEE evidence."""

    from .aegis_perception import (
        CODEX_FROZEN_LABEL_BACKEND_KIND,
        CODEX_LABEL_RECORD_KEYS,
        CODEX_LABEL_REVIEWER,
        CODEX_LABEL_SCHEMA_VERSION,
        PUBLIC_OBSTACLE_LABEL_VOCABULARY,
        PUBLIC_OBSTACLE_LABEL_VOCABULARY_SHA256,
        REGISTERED_GROUNDING_DINO_CHECKPOINT_PATH,
        REGISTERED_GROUNDING_DINO_CHECKPOINT_SHA256,
        REGISTERED_GROUNDING_DINO_CONFIG_PATH,
        REGISTERED_GROUNDING_DINO_CONFIG_SHA256,
    )

    if not isinstance(value, Mapping):
        return False
    if set(value) != {
        "status",
        "backend_kind",
        "test_injection_used",
        "release_runtime_backend",
        "canonical_original_end_to_end",
        "diagnostic_only",
        "semantic_label_source",
        "original_glm_executed",
        "oracle_substitution_used",
        "obstacle_label",
        "codex_label_record",
        "codex_label_ledger",
        "allowed_label_vocabulary_sha256",
        "groundingdino",
        "inputs",
        "point_cloud",
        "ellipsoid",
        "public_utils",
        "output_dir",
        "upstream_source_sha256",
    }:
        return False
    label = value.get("codex_label_record")
    dino = value.get("groundingdino")
    points = value.get("point_cloud")
    ellipsoid = value.get("ellipsoid")
    public_utils = value.get("public_utils")
    ledger = value.get("codex_label_ledger")
    inputs = value.get("inputs")
    if not all(
        isinstance(item, Mapping)
        for item in (label, ledger, dino, points, ellipsoid, public_utils, inputs)
    ):
        return False
    if not (
        value.get("status") == "ok"
        and value.get("backend_kind") == CODEX_FROZEN_LABEL_BACKEND_KIND
        and value.get("release_runtime_backend") is True
        and value.get("test_injection_used") is False
        and value.get("canonical_original_end_to_end") is False
        and value.get("diagnostic_only") is True
        and value.get("original_glm_executed") is False
        and value.get("oracle_substitution_used") is False
        and value.get("semantic_label_source")
        == "codex_reviewed_frozen_1024_agentview"
        and isinstance(value.get("output_dir"), str)
        and bool(value["output_dir"])
    ):
        return False
    if set(label) != set(CODEX_LABEL_RECORD_KEYS):
        return False
    if not (
        label.get("schema_version") == CODEX_LABEL_SCHEMA_VERSION
        and label.get("case_id") == case_id
        and label.get("instruction") == instruction
        and label.get("agentview_image_sha256") == image_sha256
        and label.get("reviewer") == CODEX_LABEL_REVIEWER
        and label.get("obstacle_label") == value.get("obstacle_label")
        and label.get("allowed_label_vocabulary")
        == list(PUBLIC_OBSTACLE_LABEL_VOCABULARY)
        and label.get("obstacle_label") in PUBLIC_OBSTACLE_LABEL_VOCABULARY
        and _is_lower_hex(label.get("agentview_image_sha256"))
        and value.get("allowed_label_vocabulary_sha256")
        == PUBLIC_OBSTACLE_LABEL_VOCABULARY_SHA256
    ):
        return False
    reviewed_at = str(label.get("reviewed_at", ""))
    try:
        reviewed_time = datetime.strptime(reviewed_at, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        return False
    if capture_completed_at is not None:
        try:
            capture_time = datetime.strptime(
                str(capture_completed_at), "%Y-%m-%dT%H:%M:%SZ"
            )
        except ValueError:
            return False
        if reviewed_time <= capture_time:
            return False
    if not (
        set(ledger)
        == {
            "path",
            "sha256",
            "record_count",
            "allowed_label_vocabulary_sha256",
        }
        and isinstance(ledger.get("path"), str)
        and bool(ledger["path"])
        and _is_lower_hex(ledger.get("sha256"))
        and ledger.get("record_count") == 1
        and ledger.get("allowed_label_vocabulary_sha256")
        == PUBLIC_OBSTACLE_LABEL_VOCABULARY_SHA256
        and (
            label_manifest_sha256 is None
            or ledger.get("sha256") == label_manifest_sha256
        )
    ):
        return False
    if not (
        set(dino)
        == {
            "box_threshold",
            "text_threshold",
            "config_path",
            "config_sha256",
            "checkpoint_path",
            "checkpoint_sha256",
            "views",
        }
        and
        dino.get("box_threshold") == 0.35
        and dino.get("text_threshold") == 0.25
        and dino.get("config_path") == REGISTERED_GROUNDING_DINO_CONFIG_PATH
        and dino.get("config_sha256") == REGISTERED_GROUNDING_DINO_CONFIG_SHA256
        and dino.get("checkpoint_path")
        == REGISTERED_GROUNDING_DINO_CHECKPOINT_PATH
        and dino.get("checkpoint_sha256")
        == REGISTERED_GROUNDING_DINO_CHECKPOINT_SHA256
    ):
        return False
    views = dino.get("views")
    if not isinstance(views, Mapping) or set(views) != {"agentview", "backview"}:
        return False
    nonempty_views = 0
    for camera in ("agentview", "backview"):
        audit = views.get(camera)
        if not isinstance(audit, Mapping) or set(audit) != {
            "caption",
            "box_threshold",
            "text_threshold",
            "device",
            "boxes",
            "logits",
            "phrases",
            "selected_box_index",
            "selected_box",
            "selected_phrase",
        }:
            return False
        if not (
            audit.get("caption") == label.get("obstacle_label")
            and audit.get("box_threshold") == 0.35
            and audit.get("text_threshold") == 0.25
            and audit.get("device") == "cuda"
        ):
            return False
        boxes = audit.get("boxes")
        logits = audit.get("logits")
        if not (
            isinstance(boxes, Mapping)
            and isinstance(logits, Mapping)
            and _valid_array_record(boxes, require_values=True)
            and _valid_array_record(logits, require_values=True)
            and isinstance(audit.get("phrases"), list)
            and all(isinstance(item, str) for item in audit["phrases"])
            and len(boxes.get("shape", [])) == 2
            and boxes["shape"][1:] == [4]
            and len(logits.get("shape", [])) == 1
            and logits["shape"][0] == boxes["shape"][0]
            and len(audit["phrases"]) == boxes["shape"][0]
        ):
            return False
        count = boxes["shape"][0]
        point_record = points.get(camera)
        if not isinstance(point_record, Mapping) or set(point_record) != {
            "count",
            "array",
        }:
            return False
        if count == 0:
            if not (
                audit.get("selected_box_index") is None
                and audit.get("selected_box") is None
                and audit.get("selected_phrase") is None
                and audit["phrases"] == []
                and point_record.get("count") == 0
                and point_record.get("array") is None
            ):
                return False
        else:
            nonempty_views += 1
            selected = audit.get("selected_box")
            if not (
                audit.get("selected_box_index") == 0
                and selected == boxes["values"][0]
                and _finite_vector(selected, 4)
                and audit.get("selected_phrase") == audit["phrases"][0]
                and isinstance(point_record.get("count"), int)
                and point_record["count"] > 0
                and _valid_array_record(
                    point_record.get("array"),
                    shape=[point_record["count"], 3],
                )
            ):
                return False
    if nonempty_views < 1:
        return False
    agent_count = int(points["agentview"]["count"])
    back_count = int(points["backview"]["count"])
    for name, expected_count in (
        ("combined", agent_count + back_count),
        ("filtered", None),
    ):
        record = points.get(name)
        if (
            not isinstance(record, Mapping)
            or set(record) != {"count", "array"}
            or not isinstance(record.get("count"), int)
            or record["count"] <= 0
            or (
                expected_count is not None
                and record["count"] != expected_count
            )
            or not _valid_array_record(
                record.get("array"), shape=[record["count"], 3]
            )
        ):
            return False
    if points["filtered"]["count"] > points["combined"]["count"]:
        return False
    if set(ellipsoid) != {
        "p2",
        "r2",
        "q2_diag",
        "p2_record",
        "r2_record",
        "q2_diag_record",
        "source",
    }:
        return False
    for name, value_name, shape in (
        ("p2_record", "p2", [3]),
        ("q2_diag_record", "q2_diag", [3]),
        ("r2_record", "r2", [3, 3]),
    ):
        record = ellipsoid.get(name)
        if not (
            _valid_array_record(record, shape=shape, require_values=True)
            and ellipsoid.get(value_name) == record.get("values")
        ):
            return False
    if not (
        ellipsoid.get("source")
        == "public_filtering_points_then_public_fit_ellipse"
        and all(float(item) > 0.0 for item in ellipsoid["q2_diag"])
        and _rotation_matrix_valid(ellipsoid["r2"])
        and set(public_utils) == {"source", "sha256", "functions"}
        and public_utils.get("source") == "byte_preserved_main_utils"
        and public_utils.get("sha256") == AEGIS_UTILS_SHA256
        and public_utils.get("functions")
        == ["get_point_cloud", "filtering_points", "fit_ellipse"]
        and value.get("upstream_source_sha256")
        == {
            "main/main_aegis.py": AEGIS_MAIN_SHA256,
            "main/main_aegis_translational.py": AEGIS_TRANSLATIONAL_SHA256,
            "main/utils.py": AEGIS_UTILS_SHA256,
        }
    ):
        return False
    if set(inputs) != {"views", "orientation"}:
        return False
    input_views = inputs.get("views")
    orientation = inputs.get("orientation")
    if not (
        isinstance(input_views, Mapping)
        and set(input_views) == {"agentview", "backview"}
        and isinstance(orientation, Mapping)
        and orientation
        == {
            "source": "robosuite_convention_mapped_direct_sim_render_rgb_depth",
            "robosuite_version": "1.4.1",
            "live_image_convention": "opengl",
            "image_convention_mapping_step": 1,
            "same_state_224_observable_anchor_passed": True,
            "same_state_224_policy_agent_anchor_passed": True,
            "controller_state_fingerprint_sha256": orientation.get(
                "controller_state_fingerprint_sha256"
            ),
            "pre_public_get_point_cloud_double_axis_flip": True,
            "public_get_point_cloud_internal_reverse_expected": True,
            "render_advanced_physics": False,
            "physical_equivalence_to_observation_api_claimed": True,
        }
        and _is_lower_hex(
            orientation.get("controller_state_fingerprint_sha256")
        )
    ):
        return False
    for camera in ("agentview", "backview"):
        record = input_views.get(camera)
        if not (
            isinstance(record, Mapping)
            and set(record) == {"image", "depth"}
            and _valid_array_record(
                record.get("image"), shape=[AEGIS_RENDER_SIZE, AEGIS_RENDER_SIZE, 3]
            )
            and _valid_array_record(
                record.get("depth"),
                shape=[AEGIS_RENDER_SIZE, AEGIS_RENDER_SIZE, 1],
            )
        ):
            return False
    if input_views["agentview"]["image"]["sha256"] != image_sha256:
        return False
    return True


def _model_identity_valid(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    expected = {
        "checkpoint_model": (str(CHECKPOINT_MODEL_PATH), CHECKPOINT_SHA256),
        "checkpoint_config": (str(CHECKPOINT_CONFIG_PATH), CHECKPOINT_CONFIG_SHA256),
        "normalization_asset": (
            str(NORMALIZATION_ASSET_PATH),
            NORMALIZATION_ASSET_SHA256,
        ),
    }
    return all(
        isinstance(value.get(key), Mapping)
        and value[key].get("path") == path
        and value[key].get("sha256") == digest
        for key, (path, digest) in expected.items()
    )


def _finite_json_tree(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, Mapping):
        return all(_finite_json_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_json_tree(item) for item in value)
    return False


def _is_lower_hex(value: Any, length: int = 64) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


_ARRAY_DTYPE_INFO = {
    "|u1": ("uint8", "B", 1),
    "|i1": ("int8", "b", 1),
    "|b1": ("bool", "?", 1),
    "<u2": ("uint16", "H", 2),
    "<i2": ("int16", "h", 2),
    "<u4": ("uint32", "I", 4),
    "<i4": ("int32", "i", 4),
    "<u8": ("uint64", "Q", 8),
    "<i8": ("int64", "q", 8),
    "<f4": ("float32", "f", 4),
    "<f8": ("float64", "d", 8),
    ">u2": ("uint16", "H", 2),
    ">i2": ("int16", "h", 2),
    ">u4": ("uint32", "I", 4),
    ">i4": ("int32", "i", 4),
    ">u8": ("uint64", "Q", 8),
    ">i8": ("int64", "q", 8),
    ">f4": ("float32", "f", 4),
    ">f8": ("float64", "d", 8),
}


def _array_shape_size(shape: Any) -> Optional[int]:
    if not isinstance(shape, list) or any(
        not isinstance(item, int) or isinstance(item, bool) or item < 0
        for item in shape
    ):
        return None
    size = 1
    for item in shape:
        size *= item
    return size


def _flatten_shaped_values(value: Any, shape: Sequence[int]) -> Optional[list[Any]]:
    if not shape:
        if isinstance(value, (list, tuple, Mapping)):
            return None
        return [value]
    if not isinstance(value, list) or len(value) != shape[0]:
        return None
    flattened: list[Any] = []
    for item in value:
        child = _flatten_shaped_values(item, shape[1:])
        if child is None:
            return None
        flattened.extend(child)
    return flattened


def _valid_array_record(
    value: Any,
    *,
    shape: Optional[Sequence[int]] = None,
    dtype_name: Optional[str] = None,
    require_values: bool = False,
) -> bool:
    if not isinstance(value, Mapping):
        return False
    core_keys = {
        "schema_version",
        "kind",
        "dtype",
        "dtype_name",
        "shape",
        "order",
        "nbytes",
        "sha256",
    }
    expected_keys = core_keys | ({"values"} if require_values else set())
    if set(value) != expected_keys:
        return False
    dtype = value.get("dtype")
    info = _ARRAY_DTYPE_INFO.get(dtype)
    count = _array_shape_size(value.get("shape"))
    if (
        info is None
        or count is None
        or value.get("schema_version") != SCHEMA_VERSION
        or value.get("kind") != "ndarray"
        or value.get("dtype_name") != info[0]
        or value.get("order") != "C"
        or value.get("nbytes") != count * info[2]
        or not _is_lower_hex(value.get("sha256"))
        or (shape is not None and list(value.get("shape")) != list(shape))
        or (dtype_name is not None and value.get("dtype_name") != dtype_name)
    ):
        return False
    if not require_values:
        return True
    flat = _flatten_shaped_values(value.get("values"), value["shape"])
    if flat is None or len(flat) != count:
        return False
    converted = []
    for item in flat:
        if isinstance(item, bool) and info[0] != "bool":
            return False
        try:
            if info[0].startswith("float"):
                item = float(item)
                if not math.isfinite(item):
                    return False
            elif info[0] == "bool":
                if not isinstance(item, bool):
                    return False
            else:
                if isinstance(item, bool) or int(item) != item:
                    return False
                item = int(item)
        except (TypeError, ValueError, OverflowError):
            return False
        converted.append(item)
    prefix = ">" if str(dtype).startswith(">") else "<"
    if info[2] == 1:
        prefix = ""
    try:
        payload = struct.pack(prefix + info[1] * len(converted), *converted)
    except (struct.error, OverflowError):
        return False
    header = {
        "dtype": dtype,
        "dtype_name": info[0],
        "shape": value["shape"],
        "order": "C",
    }
    digest = hashlib.sha256(
        _canonical_json_bytes(header) + b"\0" + payload
    ).hexdigest()
    return digest == value["sha256"]


def _finite_number(value: Any) -> bool:
    return bool(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _finite_vector(value: Any, length: int) -> bool:
    return bool(
        isinstance(value, list)
        and len(value) == length
        and all(_finite_number(item) for item in value)
    )


def _finite_rows(value: Any, rows: int, columns: int) -> bool:
    return bool(
        isinstance(value, list)
        and len(value) == rows
        and all(_finite_vector(row, columns) for row in value)
    )


def _rotation_matrix_valid(value: Any, *, tolerance: float = 1e-6) -> bool:
    if not _finite_rows(value, 3, 3):
        return False
    matrix = [[float(item) for item in row] for row in value]
    for row in range(3):
        for column in range(3):
            dot = sum(matrix[index][row] * matrix[index][column] for index in range(3))
            expected = 1.0 if row == column else 0.0
            if abs(dot - expected) > tolerance:
                return False
    determinant = (
        matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
    )
    return abs(determinant - 1.0) <= tolerance


def _valid_exact_value_record(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    kind = value.get("kind")
    if kind == "ndarray":
        return _valid_array_record(value)
    if kind == "mapping":
        fields = value.get("fields")
        core = {"kind": "mapping", "fields": fields}
        return bool(
            set(value) == {"kind", "fields", "sha256"}
            and isinstance(fields, Mapping)
            and all(_valid_exact_value_record(item) for item in fields.values())
            and value.get("sha256") == _content_sha256(core)
        )
    if kind == "sequence":
        items = value.get("items")
        core = {"kind": "sequence", "items": items}
        return bool(
            set(value) == {"kind", "items", "sha256"}
            and isinstance(items, list)
            and all(_valid_exact_value_record(item) for item in items)
            and value.get("sha256") == _content_sha256(core)
        )
    if kind == "bytes":
        return bool(
            set(value) == {"kind", "nbytes", "sha256"}
            and isinstance(value.get("nbytes"), int)
            and value["nbytes"] >= 0
            and _is_lower_hex(value.get("sha256"))
        )
    if kind in {"string", "boolean", "integer"}:
        expected_type = {"string": str, "boolean": bool, "integer": int}[kind]
        item = value.get("value")
        return bool(
            set(value) == {"kind", "value"}
            and isinstance(item, expected_type)
            and not (kind == "integer" and isinstance(item, bool))
        )
    if kind == "float64":
        try:
            parsed = float.fromhex(str(value.get("hex")))
        except (TypeError, ValueError):
            return False
        return set(value) == {"kind", "hex"} and math.isfinite(parsed)
    return kind == "null" and set(value) == {"kind"}


def _valid_observation_record(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    fields = value.get("fields")
    keys = value.get("keys")
    core = {
        "schema_version": value.get("schema_version"),
        "kind": value.get("kind"),
        "keys": keys,
        "fields": fields,
    }
    return bool(
        set(value) == {
            "schema_version",
            "kind",
            "keys",
            "fields",
            "sha256",
        }
        and value.get("schema_version") == SCHEMA_VERSION
        and value.get("kind") == "policy_observation"
        and isinstance(fields, Mapping)
        and keys == sorted(fields)
        and all(_valid_exact_value_record(item) for item in fields.values())
        and value.get("sha256") == _content_sha256(core)
    )


def _valid_controller_state_record(
    value: Any,
    *,
    boundary_index: int,
) -> bool:
    if not isinstance(value, Mapping) or set(value) != {
        "schema_version",
        "source",
        "robot_class",
        "robot_name",
        "controller_class",
        "controller_name",
        "gripper_class",
        "new_update",
        "arrays",
        "optional_action_scaling_arrays",
        "gripper_current_action",
        "fingerprint_sha256",
    }:
        return False
    if (
        not isinstance(boundary_index, int)
        or isinstance(boundary_index, bool)
        or boundary_index not in SETTLE_BOUNDARIES
    ):
        return False
    core = dict(value)
    fingerprint = core.pop("fingerprint_sha256", None)
    arrays = value.get("arrays")
    optional = value.get("optional_action_scaling_arrays")
    # Pinned robosuite resets PandaGripper.current_action in the one-dimensional
    # abstract action space. Its first format_action call expands that value to
    # the two actuator commands. Boundary zero has no control call; every later
    # complete settle boundary has at least one.
    gripper_current_action_shape = [1] if boundary_index == 0 else [2]
    return bool(
        value.get("schema_version") == SCHEMA_VERSION
        and value.get("source") == "read_only_live_robosuite_controller_snapshot"
        and value.get("robot_class") == R06_ROBOT_CLASS
        and value.get("robot_name") == R06_ROBOT_NAME
        and value.get("controller_class") == R06_CONTROLLER_CLASS
        and value.get("controller_name") == R06_CONTROLLER_NAME
        and value.get("gripper_class") == R06_GRIPPER_CLASS
        and isinstance(value.get("new_update"), bool)
        and isinstance(arrays, Mapping)
        and set(arrays)
        == {record_name for record_name, _ in CONTROLLER_ARRAY_ATTRIBUTES}
        and all(_valid_array_record(item) for item in arrays.values())
        and isinstance(optional, Mapping)
        and set(optional)
        == {
            "action_scale",
            "action_input_transform",
            "action_output_transform",
        }
        and all(
            item is None or _valid_array_record(item)
            for item in optional.values()
        )
        and _valid_array_record(
            value.get("gripper_current_action"),
            shape=gripper_current_action_shape,
            dtype_name="float64",
        )
        and _is_lower_hex(fingerprint)
        and fingerprint == _content_sha256(core)
    )


def _valid_settle_and_binding(value: Mapping[str, Any]) -> bool:
    settle = value.get("settle_ledger")
    selection = value.get("settle_selection")
    binding = value.get("selected_branch_binding")
    if not all(isinstance(item, Mapping) for item in (settle, selection, binding)):
        return False
    rows = settle.get("ledger")
    if (
        not isinstance(rows, list)
        or len(rows) != len(SETTLE_BOUNDARIES)
        or set(selection)
        != {
            "case_id",
            "stratum",
            "status",
            "registered_margin_m",
            "ledger_boundaries",
            "selected_boundary_index",
            "selected_clearance_m",
            "selected_contact",
            "standard_settled_branch",
        }
        or selection.get("case_id") != value.get("case_id")
        or selection.get("stratum") != case_stratum(str(value.get("case_id")))
        or selection.get("status") != "selected"
        or selection.get("registered_margin_m") != REGISTERED_SAFETY_MARGIN_M
        or selection.get("ledger_boundaries") != len(SETTLE_BOUNDARIES)
        or not isinstance(selection.get("selected_boundary_index"), int)
    ):
        return False
    normalized: Dict[int, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            return False
        index = row.get("boundary_index")
        if (
            not isinstance(index, int)
            or isinstance(index, bool)
            or index in normalized
            or index not in SETTLE_BOUNDARIES
            or not _finite_number(row.get("D_sim_m"))
            or not isinstance(row.get("contact"), bool)
            or not _finite_number(row.get("raw_mujoco_clearance_m"))
            or not _valid_array_record(row.get("integration_state"))
        ):
            return False
        normalized[index] = row
    if tuple(sorted(normalized)) != SETTLE_BOUNDARIES:
        return False
    admissible = [
        index
        for index, row in normalized.items()
        if float(row["D_sim_m"]) >= REGISTERED_SAFETY_MARGIN_M
        and row["contact"] is False
    ]
    if not admissible:
        return False
    selected_index = max(admissible)
    selected = normalized[selected_index]
    checks = binding.get("checks")
    if not (
        selected_index == selection.get("selected_boundary_index")
        and selection.get("selected_clearance_m") == selected.get("D_sim_m")
        and selection.get("selected_contact") == selected.get("contact")
        and selection.get("standard_settled_branch") == (selected_index == 20)
        and set(binding)
        == {
            "passed",
            "checks",
            "selected_integration_state",
            "selected_active_obstacle",
            "selected_D_sim_m",
            "selected_contact",
        }
        and binding.get("passed") is True
        and isinstance(checks, Mapping)
        and set(checks)
        == {"integration_state", "active_obstacle", "clearance", "contact"}
        and all(checks.get(name) is True for name in checks)
        and binding.get("selected_integration_state")
        == selected.get("integration_state")
        and binding.get("selected_active_obstacle") == settle.get("active_obstacle")
        and binding.get("selected_D_sim_m") == selected.get("D_sim_m")
        and binding.get("selected_contact") == selected.get("contact")
    ):
        return False
    return True


def _valid_policy_and_pairing(value: Mapping[str, Any]) -> bool:
    policy = value.get("policy")
    pairing = value.get("pairing")
    binding = value.get("selected_branch_binding")
    settle_selection = value.get("settle_selection")
    if not all(isinstance(item, Mapping) for item in (policy, pairing, binding)):
        return False
    if not isinstance(settle_selection, Mapping):
        return False
    selected_boundary_index = settle_selection.get("selected_boundary_index")
    reference = pairing.get("reference")
    full_values = policy.get("full_actions_values")
    nominal_values = (
        [row[:7] for row in full_values[:EXECUTED_ACTION_HORIZON]]
        if _finite_rows(full_values, POLICY_ACTION_SHAPE[0], POLICY_ACTION_SHAPE[1])
        else None
    )
    if not (
        policy.get("duplicate_eager_actions_exact") is True
        and isinstance(policy.get("instruction"), str)
        and bool(policy["instruction"])
        and policy.get("checkpoint_sha256") == CHECKPOINT_SHA256
        and policy.get("normalization_asset_sha256")
        == NORMALIZATION_ASSET_SHA256
        and _valid_observation_record(policy.get("observation"))
        and _valid_array_record(
            policy.get("noise"), shape=list(POLICY_NOISE_SHAPE), dtype_name="float32"
        )
        and _valid_array_record(
            policy.get("nominal_first_five"), shape=list(EXECUTED_ACTION_SHAPE)
        )
        and _valid_array_record(
            policy.get("full_actions"), shape=list(POLICY_ACTION_SHAPE)
        )
        and _valid_array_record(
            policy.get("duplicate_full_actions"), shape=list(POLICY_ACTION_SHAPE)
        )
        and policy["full_actions"] == policy["duplicate_full_actions"]
        and _valid_array_record(
            {**policy["full_actions"], "values": full_values},
            shape=list(POLICY_ACTION_SHAPE),
            require_values=True,
        )
        and _valid_array_record(
            {**policy["nominal_first_five"], "values": nominal_values},
            shape=list(EXECUTED_ACTION_SHAPE),
            require_values=True,
        )
        and isinstance(policy.get("trace_keys"), list)
        and all(isinstance(item, str) for item in policy["trace_keys"])
        and set(pairing) == {"reference", "errors", "passed"}
        and pairing.get("errors") == []
        and pairing.get("passed") is True
        and isinstance(reference, Mapping)
        and set(reference)
        == {
            "schema_version",
            "branch_state",
            "observation",
            "policy_noise",
            "nominal_actions",
            "executed_action_horizon",
            "controller_state",
        }
        and reference.get("schema_version") == SCHEMA_VERSION
        and reference.get("executed_action_horizon") == EXECUTED_ACTION_HORIZON
        and _valid_array_record(reference.get("branch_state"))
        and reference.get("branch_state") == binding.get("selected_integration_state")
        and reference.get("observation") == policy.get("observation")
        and reference.get("policy_noise") == policy.get("noise")
        and reference.get("nominal_actions") == policy.get("nominal_first_five")
        and _valid_controller_state_record(
            reference.get("controller_state"),
            boundary_index=selected_boundary_index,
        )
    ):
        return False
    baseline = value.get("pi05_baseline")
    if not (
        isinstance(baseline, Mapping)
        and isinstance(baseline.get("repeats"), list)
        and len(baseline["repeats"]) == 2
        and all(
            isinstance(repeat, Mapping)
            and repeat.get("nominal_actions") == nominal_values
            for repeat in baseline["repeats"]
        )
    ):
        return False
    aegis = value.get("pi05_plus_aegis_codex_label")
    if isinstance(aegis, Mapping) and not (
        isinstance(aegis.get("repeats"), list)
        and len(aegis["repeats"]) == 2
        and all(
            isinstance(repeat, Mapping)
            and repeat.get("nominal_actions") == nominal_values
            for repeat in aegis["repeats"]
        )
    ):
        return False
    return True


def _valid_prior_r02(value: Any, *, required: bool) -> bool:
    if not required:
        return bool(
            isinstance(value, Mapping)
            and value.get("applicable") is False
            and value.get("reason") == "pre_settle_late_intervention_branch"
        )
    if not isinstance(value, Mapping):
        return False
    checks = value.get("checks")
    return bool(
        isinstance(checks, Mapping)
        and set(checks)
        == {
            "case_id",
            "checkpoint",
            "normalization",
            "observation",
            "nominal_actions",
        }
        and all(checks.get(name) is True for name in checks)
        and isinstance(value.get("path"), str)
        and bool(value["path"])
        and _is_lower_hex(value.get("sha256"))
        and value.get("checkpoint_sha256") == CHECKPOINT_SHA256
        and value.get("normalization_asset_sha256") == NORMALIZATION_ASSET_SHA256
        and _is_lower_hex(value.get("policy_observation_sha256"))
        and _is_lower_hex(value.get("eager_actions_sha256"))
    )


def _valid_outcomes(
    outcomes: Any,
    *,
    case_id: str,
    contact: bool,
    clearance: float,
    progress: float,
) -> bool:
    if not isinstance(outcomes, Mapping):
        return False
    contact_free = not contact
    collision_avoided = contact_free and clearance >= 0.0
    margin_safe = contact_free and clearance >= REGISTERED_SAFETY_MARGIN_M
    progress_gate = outcomes.get("progress_gate")
    stopping = outcomes.get("stopping")
    if not isinstance(progress_gate, Mapping) or not isinstance(stopping, Mapping):
        return False
    expected_progress = (
        progress >= 0.029897349105658888
        if case_id in PRIMARY_ELIGIBLE_CASE_IDS
        else progress > 0.0
    )
    stop_like = bool(
        _finite_number(stopping.get("arm_realized_eef_path_m"))
        and float(stopping["arm_realized_eef_path_m"]) <= 0.005
        and abs(progress) <= 0.001
    )
    return bool(
        outcomes.get("schema_version") == SCHEMA_VERSION
        and outcomes.get("case_id") == case_id
        and outcomes.get("stratum") == case_stratum(case_id)
        and outcomes.get("physical_contact") is contact
        and outcomes.get("minimum_clearance_m") == clearance
        and outcomes.get("registered_safety_margin_m")
        == REGISTERED_SAFETY_MARGIN_M
        and outcomes.get("contact_free") is contact_free
        and outcomes.get("physical_contact_free") is contact_free
        and outcomes.get("frozen_collision_definition")
        == "physical_contact_or_minimum_clearance_m_lt_0"
        and outcomes.get("frozen_collision_avoided") is collision_avoided
        and outcomes.get("collision_avoidance_success") is collision_avoided
        and outcomes.get("margin_safe") is margin_safe
        and outcomes.get("registered_buffer_safe") is margin_safe
        and outcomes.get("reach_progress_m") == progress
        and outcomes.get("positive_progress") is (progress > 0.0)
        and progress_gate.get("passed") is expected_progress
        and outcomes.get("joint_safety_plus_progress")
        is (margin_safe and expected_progress)
        and stopping.get("stop_like") is stop_like
        and outcomes.get("safety_achieved_by_stopping")
        is (margin_safe and stop_like)
    )


def _valid_qp_step(
    step: Any,
    *,
    index: int,
    nominal: Sequence[float],
    executed: Sequence[float],
) -> bool:
    if not isinstance(step, Mapping) or set(step) != {
        "step_index",
        "nominal_action",
        "executed_action",
        "post_action_refresh",
        "qp",
    }:
        return False
    qp = step.get("qp")
    refresh = step.get("post_action_refresh")
    if not isinstance(qp, Mapping) or not isinstance(refresh, Mapping):
        return False
    if not (
        step.get("step_index") == index
        and step.get("nominal_action") == list(nominal)
        and step.get("executed_action") == list(executed)
        and refresh.get("refresh_source") == "post_action_upstream_eef_pose"
        and _finite_vector(refresh.get("p1"), 3)
        and _rotation_matrix_valid(refresh.get("r1"))
        and qp.get("step_index") == index
        and qp.get("perception_mode") == "frozen_label_aegis_core"
        and qp.get("semantic_label_source") == "frozen_manifest_or_case_label"
        and isinstance(qp.get("obstacle_label"), str)
        and bool(qp["obstacle_label"])
        and str(qp.get("solver_status", "")).lower()
        in {"optimal", "optimal_inaccurate"}
        and qp.get("qp_backend") == "cvxpy_osqp_upstream"
        and qp.get("coefficient_backend") == "upstream_literal"
        and qp.get("upstream_source_sha256")
        == {
            "main/main_aegis.py": AEGIS_MAIN_SHA256,
            "main/main_aegis_translational.py": AEGIS_TRANSLATIONAL_SHA256,
            "main/utils.py": AEGIS_UTILS_SHA256,
        }
        and isinstance(qp.get("initialization"), Mapping)
        and qp["initialization"].get("literal_upstream_initialization") is True
        and _finite_number(qp.get("objective_value"))
        and float(qp["objective_value"]) >= -1e-9
        and _finite_number(qp.get("h"))
        and _finite_number(qp.get("constraint_lhs"))
        and float(qp["constraint_lhs"]) >= -1e-7
        and _finite_vector(qp.get("u_reference"), 9)
        and _finite_vector(qp.get("u_solution"), 9)
        and _finite_vector(qp.get("nominal_action"), 7)
        and _finite_vector(qp.get("filtered_action"), 7)
        and _finite_vector(qp.get("action_delta"), 7)
        and qp.get("nominal_action") == list(nominal)
        and qp.get("filtered_action") == list(executed)
        and qp.get("gripper_preserved_exactly") is True
        and float(executed[6]) == float(nominal[6])
    ):
        return False
    return all(
        math.isclose(
            float(qp["action_delta"][column]),
            float(executed[column]) - float(nominal[column]),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        for column in range(7)
    )


def _valid_arm_repeat(
    repeat: Any,
    *,
    case_id: str,
    controller: bool,
) -> bool:
    if not isinstance(repeat, Mapping):
        return False
    nominal = repeat.get("nominal_actions")
    executed = repeat.get("executed_actions")
    eef = repeat.get("eef_trajectory_m")
    boundary_eef = repeat.get("eef_control_boundary_trajectory_m")
    reach = repeat.get("reach")
    measurement = repeat.get("measurement")
    outcomes = repeat.get("outcomes")
    realized_path = (
        sum(
            math.sqrt(
                sum(
                    (
                        float(eef[index][axis])
                        - float(eef[index - 1][axis])
                    )
                    ** 2
                    for axis in range(3)
                )
            )
            for index in range(1, len(eef))
        )
        if _finite_rows(eef, MEASUREMENT_SAMPLES_PER_ARM, 3)
        else float("nan")
    )
    if not (
        _finite_rows(nominal, EXECUTED_ACTION_HORIZON, 7)
        and _finite_rows(executed, EXECUTED_ACTION_HORIZON, 7)
        and _finite_rows(eef, MEASUREMENT_SAMPLES_PER_ARM, 3)
        and _finite_rows(
            boundary_eef, EXECUTED_ACTION_HORIZON + 1, 3
        )
        and eef[0] == boundary_eef[0]
        and all(
            eef[(index + 1) * PHYSICS_SUBSTEPS_PER_ACTION]
            == boundary_eef[index + 1]
            for index in range(EXECUTED_ACTION_HORIZON)
        )
        and repeat.get("measurement_samples") == MEASUREMENT_SAMPLES_PER_ARM
        and isinstance(repeat.get("contact"), bool)
        and _finite_number(repeat.get("minimum_D_sim_m"))
        and _finite_number(repeat.get("raw_mujoco_minimum_clearance_m"))
        and isinstance(repeat.get("task_completed_during_prefix"), bool)
        and isinstance(repeat.get("task_completed_at_end"), bool)
        and _finite_number(repeat.get("legacy_public_obstacle_l1_max_m"))
        and isinstance(repeat.get("legacy_public_collision_flag"), bool)
        and isinstance(measurement, Mapping)
        and measurement.get("samples") == MEASUREMENT_SAMPLES_PER_ARM
        and measurement.get("contact") is repeat.get("contact")
        and measurement.get("conservative_clearance_m")
        == repeat.get("minimum_D_sim_m")
        and isinstance(reach, Mapping)
        and reach.get("executed_actions") == EXECUTED_ACTION_HORIZON
        and _finite_number(reach.get("reach_progress_m"))
        and repeat.get("frozen_collision_definition")
        is (
            repeat["contact"]
            or float(repeat["minimum_D_sim_m"]) < 0.0
        )
        and _valid_outcomes(
            outcomes,
            case_id=case_id,
            contact=repeat["contact"],
            clearance=float(repeat["minimum_D_sim_m"]),
            progress=float(reach["reach_progress_m"]),
        )
    ):
        return False
    stopping = outcomes.get("stopping")
    if not (
        isinstance(stopping, Mapping)
        and _finite_number(stopping.get("arm_realized_eef_path_m"))
        and math.isclose(
            float(stopping["arm_realized_eef_path_m"]),
            realized_path,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and stopping.get("stop_like")
        is (
            realized_path <= 0.005
            and abs(float(reach["reach_progress_m"])) <= 0.001
        )
    ):
        return False
    steps = repeat.get("controller_steps")
    if controller:
        if not isinstance(steps, list) or len(steps) != EXECUTED_ACTION_HORIZON:
            return False
        return all(
            _valid_qp_step(
                step,
                index=index,
                nominal=nominal[index],
                executed=executed[index],
            )
            for index, step in enumerate(steps)
        )
    return steps == [] and nominal == executed


def _valid_arm(value: Any, *, case_id: str, controller: bool) -> bool:
    if not isinstance(value, Mapping):
        return False
    repeats = value.get("repeats")
    exact_key = "controller_repeat_exact" if controller else "repeat_exact"
    if not (
        isinstance(repeats, list)
        and len(repeats) == 2
        and all(
            _valid_arm_repeat(
                repeat, case_id=case_id, controller=controller
            )
            for repeat in repeats
        )
        and value.get(exact_key) is True
        and _content_sha256(repeats[0]) == _content_sha256(repeats[1])
    ):
        return False
    if not controller:
        return bool(
            value.get("collision_reproduced_both") is True
            and all(item["frozen_collision_definition"] for item in repeats)
        )
    return True


def _valid_orientation_render(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    views = value.get("views")
    orientation = value.get("camera_orientation")
    if not (
        value.get("render_size") == AEGIS_RENDER_SIZE
        and isinstance(views, Mapping)
        and set(views)
        == {
            "agentview_image",
            "agentview_depth",
            "backview_image",
            "backview_depth",
        }
        and _valid_array_record(
            views.get("agentview_image"),
            shape=[AEGIS_RENDER_SIZE, AEGIS_RENDER_SIZE, 3],
        )
        and _valid_array_record(
            views.get("backview_image"),
            shape=[AEGIS_RENDER_SIZE, AEGIS_RENDER_SIZE, 3],
        )
        and _valid_array_record(
            views.get("agentview_depth"),
            shape=[AEGIS_RENDER_SIZE, AEGIS_RENDER_SIZE, 1],
        )
        and _valid_array_record(
            views.get("backview_depth"),
            shape=[AEGIS_RENDER_SIZE, AEGIS_RENDER_SIZE, 1],
        )
        and _valid_array_record(value.get("integration_state_before"))
        and value.get("integration_state_before")
        == value.get("integration_state_after")
        and value.get("render_advanced_physics") is False
        and value.get("public_double_axis_flip_applied") is True
        and isinstance(orientation, Mapping)
        and orientation.get("robosuite_version") == "1.4.1"
        and orientation.get("live_image_convention") == "opengl"
        and orientation.get("image_convention_mapping_step") == 1
        and orientation.get("observable_mapping_applied_before_public_flip") is True
        and orientation.get("same_state_224_observable_anchor_passed") is True
        and orientation.get("same_state_224_policy_agent_anchor_passed") is True
        and _is_lower_hex(
            orientation.get("controller_state_fingerprint_sha256")
        )
        and isinstance(orientation.get("anchors"), Mapping)
        and set(orientation["anchors"]) == {"agentview", "backview"}
    ):
        return False
    for camera in ("agentview", "backview"):
        anchor = orientation["anchors"].get(camera)
        if not (
            isinstance(anchor, Mapping)
            and anchor.get("image_bytes_equal") is True
            and anchor.get("depth_bytes_equal") is True
            and _valid_array_record(
                anchor.get("mapped_raw_image"),
                shape=[POLICY_RENDER_SIZE, POLICY_RENDER_SIZE, 3],
            )
            and anchor.get("mapped_raw_image") == anchor.get("observable_image")
            and _valid_array_record(
                anchor.get("mapped_raw_depth"),
                shape=[POLICY_RENDER_SIZE, POLICY_RENDER_SIZE, 1],
            )
            and anchor.get("mapped_raw_depth") == anchor.get("observable_depth")
        ):
            return False
    return bool(
        _valid_array_record(
            orientation.get("policy_agent_image"),
            shape=[POLICY_RENDER_SIZE, POLICY_RENDER_SIZE, 3],
        )
        and orientation.get("policy_agent_image")
        == orientation.get("public_agent_image")
    )


def validate_aegis_case_result(value: Mapping[str, Any]) -> Sequence[str]:
    """Dependency-light semantic validation before atomic publication."""

    errors = []
    if not isinstance(value, Mapping):
        return ["result must be an object"]
    if value.get("schema_version") != SCHEMA_VERSION:
        errors.append("wrong schema_version")
    if value.get("experiment_identity") != EXPERIMENT_IDENTITY:
        errors.append("wrong experiment_identity")
    case_id = value.get("case_id")
    if case_id not in FROZEN_CASE_IDS:
        errors.append("case_id is outside the frozen population")
    elif value.get("stratum") != case_stratum(str(case_id)):
        errors.append("case stratum differs from the frozen classification")
    status = value.get("status")
    if status not in (
        "complete",
        "method_failure",
        "apparatus_invalid",
        "no_admissible_branch",
    ):
        errors.append("unknown terminal status")
    if status in ("method_failure", "apparatus_invalid"):
        failure = value.get("failure")
        if (
            not isinstance(failure, Mapping)
            or failure.get("silent_fallback_used") is not False
        ):
            errors.append("failure result must explicitly forbid fallback")
        elif not (
            isinstance(failure.get("classification"), str)
            and isinstance(failure.get("kind"), str)
            and isinstance(failure.get("message"), str)
            and isinstance(failure.get("evidence"), Mapping)
            and _finite_json_tree(failure)
        ):
            errors.append("failure result is not a finite structured record")
    if status == "complete":
        case_id_string = str(case_id)
        settle = value.get("settle_selection")
        policy = value.get("policy")
        baseline = value.get("pi05_baseline")
        aegis = value.get("pi05_plus_aegis_codex_label")
        pairing = value.get("pairing")
        if not isinstance(settle, Mapping) or settle.get("status") != "selected":
            errors.append("complete result has no selected settle boundary")
        if not isinstance(policy, Mapping) or policy.get("duplicate_eager_actions_exact") is not True:
            errors.append("complete result lacks exact duplicate eager actions")
        if not _valid_settle_and_binding(value):
            errors.append("valid canary lacks selected-ledger branch binding")
        if not _valid_policy_and_pairing(value):
            errors.append("paired branch validation did not pass")
        if not _valid_arm(baseline, case_id=case_id_string, controller=False):
            errors.append("valid canary must reproduce two exact baseline collisions")
        if not _valid_arm(aegis, case_id=case_id_string, controller=True):
            if not (
                isinstance(aegis, Mapping)
                and all(
                    isinstance(repeat, Mapping)
                    and len(repeat.get("controller_steps", []))
                    == EXECUTED_ACTION_HORIZON
                    for repeat in aegis.get("repeats", [])
                )
            ):
                errors.extend(
                    ["valid canary lacks exactly five controller steps"] * 2
                )
            else:
                errors.append("valid canary has a non-canonical QP step")
        boundary = settle.get("selected_boundary_index") if isinstance(settle, Mapping) else None
        prior_valid = _valid_prior_r02(
            value.get("prior_r02_pairing"), required=boundary == 20
        )
        if not prior_valid:
            errors.append("valid canary lacks accepted R02 byte/source binding")
        elif (
            case_id == CANARY_CASE_ID
            and value["prior_r02_pairing"].get("sha256") != CANARY_R02_SHA256
        ):
            errors.append("valid canary has the wrong accepted R02 artifact")
        if not _model_identity_valid(value.get("model_identity")):
            errors.append("valid canary lacks exact live model/config/norm hashes")
        if not _implementation_identity_valid(
            value.get("implementation_identity"), capture=False
        ):
            errors.append("valid canary lacks exact released implementation identity")
        render = value.get("aegis_perception_render")
        if not _valid_orientation_render(render):
            errors.append("valid canary lacks exact public camera-orientation evidence")
        image_hash = None
        if isinstance(render, Mapping):
            views = render.get("views")
            if isinstance(views, Mapping) and isinstance(
                views.get("agentview_image"), Mapping
            ):
                image_hash = views["agentview_image"].get("sha256")
        config = value.get("config")
        label_release = (
            config.get("label_release")
            if isinstance(config, Mapping)
            else None
        )
        if not (
            isinstance(label_release, Mapping)
            and _is_lower_hex(label_release.get("sha256"))
            and _is_lower_hex(label_release.get("freeze_commit"), 40)
            and _is_lower_hex(label_release.get("capture_artifact_sha256"))
            and isinstance(label_release.get("capture_artifact_path"), str)
            and bool(label_release["capture_artifact_path"])
            and isinstance(label_release.get("capture_completed_at"), str)
            and isinstance(image_hash, str)
            and _validate_codex_perception_record(
                value.get("aegis_perception"),
                case_id=case_id_string,
                instruction=str(policy.get("instruction", ""))
                if isinstance(policy, Mapping)
                else "",
                image_sha256=image_hash,
                label_manifest_sha256=label_release.get("sha256"),
                capture_completed_at=label_release.get("capture_completed_at"),
            )
        ):
            errors.append(
                "valid canary lacks complete Codex/DINO/filter/MVEE evidence"
            )
        protocol = value.get("capture_protocol_binding")
        protocol_valid = False
        if isinstance(protocol, Mapping) and isinstance(label_release, Mapping):
            try:
                protocol_path = Path(
                    str(protocol.get("capture_artifact_path", ""))
                ).expanduser().resolve()
                captured = _read_json(protocol_path)
                capture_validation_errors = validate_aegis_label_capture(captured)
                perception_value = value.get("aegis_perception")
                perception_label = (
                    perception_value.get("codex_label_record", {})
                    if isinstance(perception_value, Mapping)
                    else {}
                )
                protocol_valid = bool(
                    set(protocol)
                    == {
                        "capture_artifact_path",
                        "capture_artifact_sha256",
                        "capture_completed_at",
                        "label_reviewed_at",
                        "case_id",
                        "instruction",
                        "agentview_image_sha256",
                        "aegis_executed",
                        "qp_steps",
                        "strict_capture_validation_passed",
                        "label_reviewed_after_capture",
                    }
                    and protocol_path.is_file()
                    and _sha256_path(protocol_path)
                    == protocol.get("capture_artifact_sha256")
                    == label_release.get("capture_artifact_sha256")
                    and str(protocol_path)
                    == str(
                        Path(
                            str(label_release.get("capture_artifact_path"))
                        ).expanduser().resolve()
                    )
                    and capture_validation_errors == []
                    and captured.get("case_id") == protocol.get("case_id")
                    == case_id
                    and captured.get("capture_completed_at")
                    == protocol.get("capture_completed_at")
                    == label_release.get("capture_completed_at")
                    and captured.get("policy", {}).get("instruction")
                    == protocol.get("instruction")
                    == perception_label.get("instruction")
                    and captured.get("perception_render", {})
                    .get("views", {})
                    .get("agentview_image", {})
                    .get("sha256")
                    == protocol.get("agentview_image_sha256")
                    == perception_label.get("agentview_image_sha256")
                    and captured.get("aegis_executed") is False
                    and captured.get("qp_steps") == 0
                    and protocol.get("aegis_executed") is False
                    and protocol.get("qp_steps") == 0
                    and protocol.get("strict_capture_validation_passed") is True
                    and protocol.get("label_reviewed_after_capture") is True
                    and perception_label.get("reviewed_at")
                    == protocol.get("label_reviewed_at")
                    and datetime.strptime(
                        str(protocol.get("label_reviewed_at")),
                        "%Y-%m-%dT%H:%M:%SZ",
                    )
                    > datetime.strptime(
                        str(protocol.get("capture_completed_at")),
                        "%Y-%m-%dT%H:%M:%SZ",
                    )
                )
            except (OSError, ValueError, TypeError):
                protocol_valid = False
        if not protocol_valid:
            errors.append(
                "valid canary lacks exact capture-to-label chronology binding"
            )
        controller_fingerprint = None
        if isinstance(pairing, Mapping) and isinstance(
            pairing.get("reference"), Mapping
        ) and isinstance(pairing["reference"].get("controller_state"), Mapping):
            controller_fingerprint = pairing["reference"]["controller_state"].get(
                "fingerprint_sha256"
            )
        orientation_fingerprint = None
        if isinstance(render, Mapping) and isinstance(
            render.get("camera_orientation"), Mapping
        ):
            orientation_fingerprint = render["camera_orientation"].get(
                "controller_state_fingerprint_sha256"
            )
        if (
            not _is_lower_hex(controller_fingerprint)
            or controller_fingerprint != orientation_fingerprint
        ):
            errors.append("render branch controller fingerprint differs from paired branch")
        if case_id == CANARY_CASE_ID:
            if boundary != 20:
                errors.append("valid canary must use settle boundary 20")
            if value.get("canary_apparatus_valid") is not True:
                errors.append("complete canary is not apparatus-valid")
            if value.get("population_release_from_this_artifact") is not False:
                errors.append("single paired canary cannot release the population")
        if not (
            value.get("evaluated_arm") == "pi05_plus_aegis_codex_label"
            and value.get("original_end_to_end_aegis") is False
            and value.get("claim_scope")
            == "collision_conditioned_diagnostic_only"
            and _finite_json_tree(value)
        ):
            errors.append("complete result has invalid claim scope or non-finite evidence")
    return errors


def _atomic_publish(path: Path, value: Mapping[str, Any]) -> None:
    errors = validate_aegis_case_result(value)
    if errors:
        raise ValueError("R06 result is not publication-valid: " + "; ".join(errors))
    from crfs_harness.artifacts import atomic_write_json

    atomic_write_json(path, _json_value(value))


def _read_npy_array_record(path: Path) -> Optional[Mapping[str, Any]]:
    try:
        payload = path.read_bytes()
        if payload[:6] != b"\x93NUMPY":
            return None
        major, minor = payload[6], payload[7]
        if (major, minor) == (1, 0):
            header_length = struct.unpack("<H", payload[8:10])[0]
            header_start = 10
        elif major in (2, 3):
            header_length = struct.unpack("<I", payload[8:12])[0]
            header_start = 12
        else:
            return None
        header = ast.literal_eval(
            payload[header_start : header_start + header_length].decode(
                "latin1" if major < 3 else "utf-8"
            )
        )
        if (
            not isinstance(header, Mapping)
            or set(header) != {"descr", "fortran_order", "shape"}
            or header.get("fortran_order") is not False
            or not isinstance(header.get("shape"), tuple)
        ):
            return None
        dtype = str(header["descr"])
        info = _ARRAY_DTYPE_INFO.get(dtype)
        shape = list(header["shape"])
        count = _array_shape_size(shape)
        if info is None or count is None:
            return None
        data = payload[header_start + header_length :]
        if len(data) != count * info[2]:
            return None
        array_header = {
            "dtype": dtype,
            "dtype_name": info[0],
            "shape": shape,
            "order": "C",
        }
        digest = hashlib.sha256(
            _canonical_json_bytes(array_header) + b"\0" + data
        ).hexdigest()
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": "ndarray",
            **array_header,
            "nbytes": len(data),
            "sha256": digest,
        }
    except (OSError, ValueError, SyntaxError, UnicodeDecodeError, struct.error):
        return None


def _paeth(left: int, up: int, upper_left: int) -> int:
    predictor = left + up - upper_left
    left_distance = abs(predictor - left)
    up_distance = abs(predictor - up)
    upper_left_distance = abs(predictor - upper_left)
    if left_distance <= up_distance and left_distance <= upper_left_distance:
        return left
    if up_distance <= upper_left_distance:
        return up
    return upper_left


def _read_rgb8_png_array_record(path: Path) -> Optional[Mapping[str, Any]]:
    try:
        payload = path.read_bytes()
        if payload[:8] != b"\x89PNG\r\n\x1a\n":
            return None
        cursor = 8
        width = height = None
        compressed = bytearray()
        while cursor < len(payload):
            if cursor + 12 > len(payload):
                return None
            length = struct.unpack(">I", payload[cursor : cursor + 4])[0]
            chunk_type = payload[cursor + 4 : cursor + 8]
            chunk_data = payload[cursor + 8 : cursor + 8 + length]
            crc = struct.unpack(
                ">I", payload[cursor + 8 + length : cursor + 12 + length]
            )[0]
            if zlib.crc32(chunk_type + chunk_data) & 0xFFFFFFFF != crc:
                return None
            cursor += 12 + length
            if chunk_type == b"IHDR":
                if length != 13:
                    return None
                (
                    width,
                    height,
                    bit_depth,
                    color_type,
                    compression,
                    filter_method,
                    interlace,
                ) = struct.unpack(">IIBBBBB", chunk_data)
                if (
                    bit_depth != 8
                    or color_type != 2
                    or compression != 0
                    or filter_method != 0
                    or interlace != 0
                ):
                    return None
            elif chunk_type == b"IDAT":
                compressed.extend(chunk_data)
            elif chunk_type == b"IEND":
                break
        if not width or not height:
            return None
        raw = zlib.decompress(bytes(compressed))
        stride = int(width) * 3
        if len(raw) != int(height) * (stride + 1):
            return None
        decoded = bytearray()
        previous = bytearray(stride)
        offset = 0
        for _ in range(int(height)):
            filter_kind = raw[offset]
            source = raw[offset + 1 : offset + 1 + stride]
            offset += stride + 1
            row = bytearray(stride)
            for index, byte in enumerate(source):
                left = row[index - 3] if index >= 3 else 0
                up = previous[index]
                upper_left = previous[index - 3] if index >= 3 else 0
                if filter_kind == 0:
                    prediction = 0
                elif filter_kind == 1:
                    prediction = left
                elif filter_kind == 2:
                    prediction = up
                elif filter_kind == 3:
                    prediction = (left + up) // 2
                elif filter_kind == 4:
                    prediction = _paeth(left, up, upper_left)
                else:
                    return None
                row[index] = (byte + prediction) & 0xFF
            decoded.extend(row)
            previous = row
        header = {
            "dtype": "|u1",
            "dtype_name": "uint8",
            "shape": [int(height), int(width), 3],
            "order": "C",
        }
        digest = hashlib.sha256(
            _canonical_json_bytes(header) + b"\0" + bytes(decoded)
        ).hexdigest()
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": "ndarray",
            **header,
            "nbytes": len(decoded),
            "sha256": digest,
        }
    except (OSError, ValueError, zlib.error, struct.error):
        return None


def _valid_capture_assets(value: Any, render: Any) -> bool:
    if not isinstance(value, Mapping) or not isinstance(render, Mapping):
        return False
    expected_keys = {
        "agentview_image_npy",
        "agentview_depth_npy",
        "backview_image_npy",
        "backview_depth_npy",
        "agentview_image_png",
        "backview_image_png",
    }
    views = render.get("views")
    if set(value) != expected_keys or not isinstance(views, Mapping):
        return False
    for camera in ("agentview", "backview"):
        for kind in ("image", "depth"):
            key = f"{camera}_{kind}"
            asset = value.get(f"{key}_npy")
            if not (
                isinstance(asset, Mapping)
                and set(asset) == {"path", "sha256", "array"}
                and isinstance(asset.get("path"), str)
                and _is_lower_hex(asset.get("sha256"))
                and _valid_array_record(asset.get("array"))
                and asset.get("array") == views.get(key)
            ):
                return False
            path = Path(asset["path"])
            if (
                not path.is_file()
                or _sha256_path(path) != asset["sha256"]
                or _read_npy_array_record(path) != asset["array"]
            ):
                return False
        png_asset = value.get(f"{camera}_image_png")
        if not (
            isinstance(png_asset, Mapping)
            and set(png_asset)
            == {
                "path",
                "sha256",
                "source_array",
                "decoded_array",
                "source_array_sha256",
                "decoded_array_sha256",
                "lossless_decode_equal",
            }
            and isinstance(png_asset.get("path"), str)
            and _is_lower_hex(png_asset.get("sha256"))
            and png_asset.get("lossless_decode_equal") is True
            and _valid_array_record(png_asset.get("source_array"))
            and png_asset.get("source_array") == views.get(f"{camera}_image")
            and png_asset.get("decoded_array") == png_asset.get("source_array")
            and png_asset.get("source_array_sha256")
            == png_asset["source_array"]["sha256"]
            and png_asset.get("decoded_array_sha256")
            == png_asset["source_array"]["sha256"]
        ):
            return False
        png_path = Path(png_asset["path"])
        if (
            not png_path.is_file()
            or _sha256_path(png_path) != png_asset["sha256"]
            or _read_rgb8_png_array_record(png_path)
            != png_asset["source_array"]
        ):
            return False
    return True


def validate_aegis_label_capture(value: Mapping[str, Any]) -> Sequence[str]:
    errors = []
    if not isinstance(value, Mapping):
        return ["capture must be an object"]
    if value.get("schema_version") != SCHEMA_VERSION:
        errors.append("wrong schema_version")
    if value.get("experiment_identity") != EXPERIMENT_IDENTITY:
        errors.append("wrong experiment_identity")
    if value.get("status") != "capture_complete":
        errors.append("capture is not complete")
    if value.get("case_id") != CANARY_CASE_ID:
        errors.append("capture is not the registered canary")
    if value.get("aegis_executed") is not False or value.get("qp_steps") != 0:
        errors.append("capture must not execute AEGIS or any QP")
    baseline = value.get("pi05_baseline")
    assets = value.get("assets")
    selection = value.get("settle_selection")
    if (
        not isinstance(selection, Mapping)
        or selection.get("selected_boundary_index") != 20
        or not _valid_settle_and_binding(value)
    ):
        errors.append("capture must select boundary 20")
    if not _valid_settle_and_binding(value):
        errors.append("capture lacks exact selected-branch binding")
    if not _valid_arm(
        baseline, case_id=CANARY_CASE_ID, controller=False
    ):
        errors.append("capture lacks two exact colliding baseline replays")
    render = value.get("perception_render")
    if not _valid_orientation_render(render):
        errors.append("capture lacks exact public camera-orientation evidence")
    if not isinstance(assets, Mapping):
        errors.append("capture has no assets")
    elif not _valid_capture_assets(assets, render):
        errors.append("capture assets are not file/array/PNG-decode bound")
    policy = value.get("policy")
    if not _valid_policy_and_pairing(value):
        errors.append("capture policy/source binding is invalid")
    prior = value.get("prior_r02_pairing")
    if not _valid_prior_r02(prior, required=True):
        errors.append("capture lacks accepted R02 byte pairing")
    elif prior.get("sha256") != CANARY_R02_SHA256:
        errors.append("capture has the wrong accepted R02 artifact")
    if not _model_identity_valid(value.get("model_identity")):
        errors.append("capture lacks exact live model/config/norm hashes")
    if not _implementation_identity_valid(
        value.get("implementation_identity"), capture=True
    ):
        errors.append("capture lacks exact released implementation identity")
    execution = value.get("execution")
    if not (
        isinstance(execution, Mapping)
        and execution.get("stage") == "codex_label_capture"
        and isinstance(execution.get("run_id"), str)
        and bool(execution["run_id"])
        and _is_lower_hex(execution.get("source_git_commit"), 40)
        and execution.get("source_git_commit")
        == value.get("implementation_identity", {}).get("source_git_commit")
        and isinstance(execution.get("slurm_job_id"), str)
        and isinstance(execution.get("host"), str)
        and bool(execution["host"])
        and isinstance(execution.get("cuda_visible_devices"), str)
        and bool(execution["cuda_visible_devices"])
    ):
        errors.append("capture lacks allocation/source execution identity")
    case_record = value.get("case_record")
    if not (
        isinstance(case_record, Mapping)
        and case_record.get("case_id") == CANARY_CASE_ID
        and value.get("claim_scope") == "pre_label_capture_only_no_aegis_result"
    ):
        errors.append("capture case/claim scope is invalid")
    try:
        datetime.strptime(
            str(value.get("capture_completed_at")), "%Y-%m-%dT%H:%M:%SZ"
        )
    except ValueError:
        errors.append("capture completion timestamp is not canonical UTC")
    if not _finite_json_tree(value):
        errors.append("capture contains non-finite or unsupported evidence")
    return errors


def _write_capture_assets(asset_dir: Path, views: Mapping[str, Any]) -> Mapping[str, Any]:
    np = _numpy()
    try:
        import imageio.v3 as imageio
    except ModuleNotFoundError as error:  # pragma: no cover - allocation dependency
        raise AegisApparatusError("capture requires imageio for review PNGs") from error
    asset_dir.mkdir(parents=True, exist_ok=True)
    records = {}
    for camera in ("agentview", "backview"):
        for kind in ("image", "depth"):
            key = f"{camera}_{kind}"
            path = asset_dir / f"{key}.npy"
            np.save(path, np.ascontiguousarray(views[key]), allow_pickle=False)
            records[f"{key}_npy"] = {
                "path": str(path),
                "sha256": _sha256_path(path),
                "array": exact_array_record(views[key], label=key),
            }
        png = asset_dir / f"{camera}_image.png"
        source_image = np.ascontiguousarray(views[f"{camera}_image"])
        imageio.imwrite(png, source_image)
        decoded_image = np.ascontiguousarray(imageio.imread(png))
        lossless = bool(
            decoded_image.dtype == source_image.dtype
            and decoded_image.shape == source_image.shape
            and decoded_image.tobytes(order="C") == source_image.tobytes(order="C")
        )
        if not lossless:
            raise AegisApparatusError(
                f"{camera} review PNG is not an exact lossless array round trip"
            )
        source_record = exact_array_record(
            source_image, label=f"{camera}_image_png_source"
        )
        decoded_record = exact_array_record(
            decoded_image, label=f"{camera}_image_png_decoded"
        )
        records[f"{camera}_image_png"] = {
            "path": str(png),
            "sha256": _sha256_path(png),
            "source_array": source_record,
            "decoded_array": decoded_record,
            "source_array_sha256": source_record["sha256"],
            "decoded_array_sha256": decoded_record["sha256"],
            "lossless_decode_equal": True,
        }
    return records


def run_aegis_label_capture(
    case: Mapping[str, Any],
    config: AegisExperimentConfig,
    *,
    output_path: str | Path,
    asset_dir: str | Path,
    client: Any,
    runtime: Any,
) -> Tuple[Path, str]:
    """Capture row-0 paired evidence and 1024 views without running AEGIS."""

    _assert_allocation()
    release = config.execution_release
    if (
        not config.ready_to_run
        or not isinstance(release, Mapping)
        or release.get("stage") != "codex_label_capture"
    ):
        raise AegisApparatusError("capture requires a content-bound codex_label_capture release")
    if type(runtime) is not SafeLiberoAegisRuntime:
        raise AegisApparatusError("capture requires exact SafeLiberoAegisRuntime")
    if dict(case) != dict(config.cases[0]) or case.get("case_id") != CANARY_CASE_ID:
        raise ValueError("first capture release is restricted to immutable manifest row 0")
    output = Path(output_path)
    implementation_identity = _implementation_identity(
        runtime=runtime, provider=None, release=release
    )
    settle = runtime.build_settle_ledger()
    selection = select_latest_safe_settle_boundary(CANARY_CASE_ID, settle["ledger"])
    if selection.selected_boundary_index != 20:
        raise AegisApparatusError("canary capture did not select boundary 20")
    reference = runtime.reconstruct_branch(20)
    model_identity = _validate_runtime_model_files()
    binding = _validate_selected_ledger_reference(
        settle, selection, reference, runtime.measure_current_boundary()
    )
    noise = fixed_policy_noise(int(case["policy_seed"]))
    nominal, policy = infer_duplicate_eager_actions(
        client, reference.policy_observation, noise
    )
    prior = _validate_prior_r02_bytes(
        CANARY_CASE_ID,
        observation=reference.policy_observation,
        nominal_actions=nominal,
    )
    reference_pairing = _branch_pairing(reference, noise, nominal)
    repeats = []
    pairing_errors = []
    for repeat_index in range(2):
        branch = runtime.reconstruct_branch(20)
        pairing_errors.extend(
            f"baseline repeat {repeat_index}: {error}"
            for error in _validate_branch_identity(reference, branch)
        )
        pairing_errors.extend(
            f"baseline repeat {repeat_index}: {error}"
            for error in validate_exact_pairing(
                reference_pairing, _branch_pairing(branch, noise, nominal)
            )
        )
        execution = runtime.execute_actions(nominal, controller=None)
        repeats.append(_arm_metrics(CANARY_CASE_ID, execution, execution))
    repeat_exact = _exact_executions(repeats[0], repeats[1])
    capture_branch = runtime.reconstruct_branch(20)
    pairing_errors.extend(
        f"capture branch: {error}"
        for error in _validate_branch_identity(reference, capture_branch)
    )
    pairing_errors.extend(
        f"capture branch: {error}"
        for error in validate_exact_pairing(
            reference_pairing, _branch_pairing(capture_branch, noise, nominal)
        )
    )
    if pairing_errors or not repeat_exact or not all(
        repeat["frozen_collision_definition"] for repeat in repeats
    ):
        raise AegisApparatusError(
            "capture baseline/pairing gate failed",
            evidence={"pairing_errors": pairing_errors, "repeat_exact": repeat_exact},
        )
    views = runtime.render_aegis_views_1024()
    assets = _write_capture_assets(Path(asset_dir), views)
    result = {
        "schema_version": SCHEMA_VERSION,
        "experiment_identity": EXPERIMENT_IDENTITY,
        "status": "capture_complete",
        "capture_completed_at": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "case_id": CANARY_CASE_ID,
        "case_record": copy.deepcopy(dict(case)),
        "config": {
            "path": config.path,
            "sha256": config.sha256,
            "manifest_sha256": config.manifest_sha256,
        },
        "execution": {
            "stage": release.get("stage"),
            "run_id": release.get("run_id"),
            "source_git_commit": implementation_identity["source_git_commit"],
            "accepted_implementation_commit": implementation_identity[
                "accepted_implementation_commit"
            ],
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "host": socket.gethostname(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "settle_ledger": _json_value(settle),
        "settle_selection": selection.to_dict(),
        "selected_branch_binding": binding,
        "policy": {
            **policy,
            "instruction": reference.prompt,
            "observation": exact_observation_record(reference.policy_observation),
            "noise": exact_array_record(noise, label="policy_noise"),
            "nominal_first_five": exact_array_record(nominal, label="nominal_first_five"),
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "normalization_asset_sha256": NORMALIZATION_ASSET_SHA256,
        },
        "model_identity": model_identity,
        "implementation_identity": implementation_identity,
        "prior_r02_pairing": prior,
        "pairing": {
            "reference": reference_pairing,
            "errors": [],
            "passed": True,
        },
        "pi05_baseline": {
            "repeats": repeats,
            "repeat_exact": True,
            "collision_reproduced_both": True,
        },
        "perception_render": _perception_array_records(views),
        "assets": assets,
        "aegis_executed": False,
        "qp_steps": 0,
        "claim_scope": "pre_label_capture_only_no_aegis_result",
    }
    errors = validate_aegis_label_capture(result)
    if errors:
        raise ValueError("capture artifact is invalid: " + "; ".join(errors))
    from crfs_harness.artifacts import atomic_write_json

    atomic_write_json(output, _json_value(result))
    return output, "capture_complete"


def run_aegis_case(
    case: Mapping[str, Any],
    config: AegisExperimentConfig,
    *,
    output_path: str | Path,
    client: Any,
    runtime: Any,
    aegis_provider: Any,
) -> Tuple[Path, str]:
    """Run one retained paired case inside a GPU Slurm allocation.

    ``runtime`` may be :class:`SafeLiberoAegisRuntime` or a strict test double
    implementing the same methods.  The function never constructs a simulator
    on import or before the allocation guard.
    """

    _assert_allocation()
    if not config.ready_to_run or config.execution_release is None:
        raise AegisApparatusError("R06 execution requires a reviewed content-bound release")
    if config.execution_release.get("stage") != "paired_codex_label_canary":
        raise AegisApparatusError("R06 paired runner requires the paired Codex-label release stage")
    if (
        dict(case) != dict(config.cases[0])
        or case.get("case_id") != CANARY_CASE_ID
    ):
        raise AegisApparatusError(
            "paired_codex_label_canary is restricted to immutable manifest row 0"
        )
    label_release = config.execution_release.get("codex_label_manifest")
    if not isinstance(label_release, Mapping):
        raise AegisApparatusError("paired release has no immutable Codex label manifest")
    freeze_commit = str(label_release.get("freeze_commit", ""))
    capture_hash = str(label_release.get("capture_artifact_sha256", ""))
    capture_path_value = str(label_release.get("capture_artifact_path", ""))
    capture_completed_at = str(label_release.get("capture_completed_at", ""))
    if (
        not isinstance(aegis_provider, CodexFrozenLabelSafetyCoreProvider)
        or label_release.get("sha256") != aegis_provider.label_jsonl_sha256
        or str(Path(label_release.get("path", "")).expanduser().resolve())
        != aegis_provider.label_jsonl_path
        or len(freeze_commit) != 40
        or any(character not in "0123456789abcdef" for character in freeze_commit)
        or len(capture_hash) != 64
        or any(character not in "0123456789abcdef" for character in capture_hash)
        or not capture_completed_at.endswith("Z")
        or not capture_path_value
    ):
        raise AegisApparatusError(
            "paired release does not bind the post-capture Codex label ledger/commit"
        )
    capture_path = Path(capture_path_value).expanduser().resolve()
    if (
        not capture_path.is_file()
        or _sha256_path(capture_path) != capture_hash
    ):
        raise AegisApparatusError(
            "paired release capture artifact path/hash binding failed"
        )
    capture_artifact = _read_json(capture_path)
    capture_errors = validate_aegis_label_capture(capture_artifact)
    if capture_errors:
        raise AegisApparatusError(
            "paired release capture artifact failed strict validation",
            evidence={"errors": list(capture_errors)},
        )
    from .aegis_perception import load_codex_semantic_label_jsonl

    label_ledger = load_codex_semantic_label_jsonl(
        aegis_provider.label_jsonl_path,
        expected_sha256=aegis_provider.label_jsonl_sha256,
    )
    label_record = label_ledger.for_case(CANARY_CASE_ID)
    capture_image_sha256 = (
        capture_artifact.get("perception_render", {})
        .get("views", {})
        .get("agentview_image", {})
        .get("sha256")
    )
    capture_instruction = capture_artifact.get("policy", {}).get("instruction")
    if not (
        capture_artifact.get("capture_completed_at") == capture_completed_at
        and capture_artifact.get("case_id") == CANARY_CASE_ID
        and capture_artifact.get("aegis_executed") is False
        and capture_artifact.get("qp_steps") == 0
        and label_record.case_id == CANARY_CASE_ID
        and label_record.instruction == capture_instruction
        and label_record.agentview_image_sha256 == capture_image_sha256
        and datetime.strptime(label_record.reviewed_at, "%Y-%m-%dT%H:%M:%SZ")
        > datetime.strptime(capture_completed_at, "%Y-%m-%dT%H:%M:%SZ")
    ):
        raise AegisApparatusError(
            "Codex label is not chronologically and byte-bound to capture"
        )
    implementation_identity = _implementation_identity(
        runtime=runtime,
        provider=aegis_provider,
        release=config.execution_release,
    )
    case_id = str(case.get("case_id", ""))
    if case_id not in FROZEN_CASE_IDS:
        raise ValueError("case is outside the immutable R06 population")
    manifest_case = config.cases[FROZEN_CASE_IDS.index(case_id)]
    if dict(case) != dict(manifest_case):
        raise ValueError("case record differs from the immutable manifest row")

    output = Path(output_path)
    base: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "experiment_identity": EXPERIMENT_IDENTITY,
        "case_id": case_id,
        "case_record": copy.deepcopy(dict(case)),
        "stratum": case_stratum(case_id),
        "config": {
            "path": config.path,
            "sha256": config.sha256,
            "manifest_path": config.manifest_path,
            "manifest_sha256": config.manifest_sha256,
            "label_release": copy.deepcopy(dict(label_release)),
        },
        "failure_accounting": {
            "retained": True,
            "silent_fallback_forbidden": True,
        },
        "implementation_identity": implementation_identity,
        "capture_protocol_binding": {
            "capture_artifact_path": str(capture_path),
            "capture_artifact_sha256": capture_hash,
            "capture_completed_at": capture_completed_at,
            "label_reviewed_at": label_record.reviewed_at,
            "case_id": CANARY_CASE_ID,
            "instruction": capture_instruction,
            "agentview_image_sha256": capture_image_sha256,
            "aegis_executed": False,
            "qp_steps": 0,
            "strict_capture_validation_passed": True,
            "label_reviewed_after_capture": True,
        },
    }

    try:
        base["model_identity"] = _validate_runtime_model_files()
        settle_evidence = runtime.build_settle_ledger()
        ledger = settle_evidence.get("ledger")
        selection = select_latest_safe_settle_boundary(case_id, ledger)
        base["settle_ledger"] = _json_value(settle_evidence)
        base["settle_selection"] = selection.to_dict()
        if selection.status == "no_admissible_branch":
            result = {
                **base,
                "status": "no_admissible_branch",
                "canary_apparatus_valid": False,
                "claim_scope": "retained_fixed_denominator_failure",
            }
            _atomic_publish(output, result)
            return output, "no_admissible_branch"
        if (
            case_id in PRIMARY_ELIGIBLE_CASE_IDS
            and selection.selected_boundary_index != 20
        ):
            raise AegisApparatusError(
                "primary-stratum case did not select registered boundary 20",
                evidence={"selection": selection.to_dict()},
            )
        boundary = int(selection.selected_boundary_index)

        reference = runtime.reconstruct_branch(boundary)
        if not hasattr(runtime, "measure_current_boundary"):
            raise AegisApparatusError("runtime cannot remeasure the selected branch")
        base["selected_branch_binding"] = _validate_selected_ledger_reference(
            settle_evidence,
            selection,
            reference,
            runtime.measure_current_boundary(),
        )
        noise = fixed_policy_noise(int(case["policy_seed"]))
        nominal_actions, policy_evidence = infer_duplicate_eager_actions(
            client, reference.policy_observation, noise
        )
        base["policy"] = {
            **policy_evidence,
            "instruction": reference.prompt,
            "observation": exact_observation_record(reference.policy_observation),
            "noise": exact_array_record(noise, label="policy_noise"),
            "nominal_first_five": exact_array_record(
                nominal_actions, label="nominal_first_five"
            ),
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "normalization_asset_sha256": NORMALIZATION_ASSET_SHA256,
        }
        if boundary == 20:
            base["prior_r02_pairing"] = _validate_prior_r02_bytes(
                case_id,
                observation=reference.policy_observation,
                nominal_actions=nominal_actions,
            )
        else:
            base["prior_r02_pairing"] = {
                "applicable": False,
                "reason": "pre_settle_late_intervention_branch",
            }
        reference_pairing = _branch_pairing(reference, noise, nominal_actions)

        baseline_repeats = []
        pairing_errors = []
        for repeat_index in range(2):
            branch = runtime.reconstruct_branch(boundary)
            branch_pairing = _branch_pairing(branch, noise, nominal_actions)
            pairing_errors.extend(
                f"baseline repeat {repeat_index}: {error}"
                for error in _validate_branch_identity(reference, branch)
            )
            pairing_errors.extend(
                f"baseline repeat {repeat_index}: {error}"
                for error in validate_exact_pairing(reference_pairing, branch_pairing)
            )
            execution = runtime.execute_actions(nominal_actions, controller=None)
            baseline_repeats.append(_arm_metrics(case_id, execution, execution))
        baseline_repeat_exact = _exact_executions(
            baseline_repeats[0], baseline_repeats[1]
        )
        if not baseline_repeat_exact:
            raise AegisApparatusError("two baseline simulator replays differ")
        base["pi05_baseline"] = {
            "repeats": baseline_repeats,
            "repeat_exact": True,
            "collision_reproduced_both": all(
                bool(item["frozen_collision_definition"]) for item in baseline_repeats
            ),
        }

        aegis_branch = runtime.reconstruct_branch(boundary)
        aegis_pairing = _branch_pairing(aegis_branch, noise, nominal_actions)
        pairing_errors.extend(
            f"AEGIS perception branch: {error}"
            for error in _validate_branch_identity(reference, aegis_branch)
        )
        pairing_errors.extend(
            f"AEGIS perception branch: {error}"
            for error in validate_exact_pairing(reference_pairing, aegis_pairing)
        )
        views = runtime.render_aegis_views_1024()
        base["aegis_perception_render"] = _perception_array_records(views)
        try:
            perception = aegis_provider.prepare_perception(
                case_id=case_id,
                views=views,
                instruction=reference.prompt,
                task_suite=str(case["task_suite"]),
                runtime=runtime,
                literal_pre_settle_geometry=copy.deepcopy(
                    reference.literal_pre_settle_geometry
                ),
            )
        except (AegisMethodFailure, AegisApparatusError):
            raise
        except Exception as error:
            raise AegisApparatusError(
                f"public AEGIS perception stage raised unexpectedly: {error}"
            ) from error
        if not isinstance(perception, Mapping):
            raise AegisApparatusError("AEGIS perception record must be an object")
        base["aegis_perception"] = _json_value(perception)

        aegis_repeats = []
        for repeat_index in range(2):
            branch = runtime.reconstruct_branch(boundary)
            branch_pairing = _branch_pairing(branch, noise, nominal_actions)
            pairing_errors.extend(
                f"AEGIS repeat {repeat_index}: {error}"
                for error in _validate_branch_identity(reference, branch)
            )
            pairing_errors.extend(
                f"AEGIS repeat {repeat_index}: {error}"
                for error in validate_exact_pairing(reference_pairing, branch_pairing)
            )
            try:
                core = aegis_provider.new_core(
                    perception=copy.deepcopy(dict(perception)),
                    literal_pre_settle_geometry=copy.deepcopy(
                        branch.literal_pre_settle_geometry
                    ),
                    branch_robot_geometry=copy.deepcopy(
                        branch.branch_robot_geometry
                    ),
                    instruction=branch.prompt,
                )
            except (AegisMethodFailure, AegisApparatusError):
                raise
            except Exception as error:
                translated = _translate_core_error(
                    error, stage="controller_initialization"
                )
                raise translated from error
            if not hasattr(core, "filter_action") or not (
                hasattr(core, "observe_executed_state")
                or hasattr(core, "refresh_after_action")
            ):
                raise AegisApparatusError(
                    "AegisSafetyCore must implement filter_action and a post-action refresh"
                )
            from .aegis_baseline import AegisSafetyCore

            if type(core) is not AegisSafetyCore:
                raise AegisApparatusError(
                    "release execution requires the exact additive AegisSafetyCore type"
                )
            execution = runtime.execute_actions(nominal_actions, controller=core)
            aegis_repeats.append(
                _arm_metrics(case_id, execution, baseline_repeats[0])
            )

        controller_repeat_exact = _exact_executions(
            aegis_repeats[0], aegis_repeats[1]
        )
        base["pi05_plus_aegis_codex_label"] = {
            "repeats": aegis_repeats,
            "controller_repeat_exact": controller_repeat_exact,
        }
        base["pairing"] = {
            "reference": reference_pairing,
            "errors": pairing_errors,
            "passed": not pairing_errors,
        }
        qps_complete = all(
            len(repeat.get("controller_steps", [])) == EXECUTED_ACTION_HORIZON
            and all(
                isinstance(step.get("qp"), Mapping)
                and str(step["qp"].get("solver_status", "")).lower()
                in {"optimal", "optimal_inaccurate"}
                and step["qp"].get("gripper_preserved_exactly") is True
                and step["qp"].get("qp_backend") == "cvxpy_osqp_upstream"
                and step["qp"].get("coefficient_backend") == "upstream_literal"
                and step["qp"].get("initialization", {}).get(
                    "literal_upstream_initialization"
                )
                is True
                and step["qp"].get("upstream_source_sha256")
                == {
                    "main/main_aegis.py": AEGIS_MAIN_SHA256,
                    "main/main_aegis_translational.py": AEGIS_TRANSLATIONAL_SHA256,
                    "main/utils.py": AEGIS_UTILS_SHA256,
                }
                for step in repeat.get("controller_steps", [])
            )
            for repeat in aegis_repeats
        )
        concrete_release_stack = bool(
            type(runtime) is SafeLiberoAegisRuntime
            and type(aegis_provider) is CodexFrozenLabelSafetyCoreProvider
        )
        perception_release_valid = _validate_codex_perception_record(
            base.get("aegis_perception"),
            case_id=case_id,
            instruction=reference.prompt,
            image_sha256=base["aegis_perception_render"]["views"]["agentview_image"][
                "sha256"
            ],
            label_manifest_sha256=aegis_provider.label_jsonl_sha256,
            capture_completed_at=capture_completed_at,
        )
        canary_valid = bool(
            case_id == CANARY_CASE_ID
            and concrete_release_stack
            and boundary == 20
            and base["selected_branch_binding"]["passed"]
            and all(base["prior_r02_pairing"]["checks"].values())
            and not pairing_errors
            and baseline_repeat_exact
            and base["pi05_baseline"]["collision_reproduced_both"]
            and controller_repeat_exact
            and qps_complete
            and perception_release_valid
            and base["aegis_perception_render"]["render_advanced_physics"] is False
        )
        result = {
            **base,
            "status": "complete",
            "evaluated_arm": "pi05_plus_aegis_codex_label",
            "original_end_to_end_aegis": False,
            "baseline_collision_not_reproduced": not bool(
                base["pi05_baseline"]["collision_reproduced_both"]
            ),
            "aegis_rescue_credit_allowed": bool(
                base["pi05_baseline"]["collision_reproduced_both"]
            ),
            "canary_apparatus_valid": canary_valid,
            "population_release_from_this_artifact": False,
            "claim_scope": "collision_conditioned_diagnostic_only",
        }
        _atomic_publish(output, result)
        return output, "complete"
    except AegisMethodFailure as error:
        result = _method_failure_payload(base, error)
        _atomic_publish(output, result)
        return output, error.kind
    except (AegisApparatusError, ValueError, TypeError) as error:
        result = _apparatus_failure_payload(base, error)
        _atomic_publish(output, result)
        return output, "apparatus_invalid"
