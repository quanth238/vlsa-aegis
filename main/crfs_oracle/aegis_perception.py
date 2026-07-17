"""Fail-closed adapter for the public AEGIS perception stack.

The upstream implementation hard-codes an empty ZhipuAI key and relative
GroundingDINO paths.  This module preserves its prompt, model settings,
thresholds, filtering, and MVEE behavior while supplying secrets and assets at
runtime.  It never substitutes a simulator label or privileged geometry.
"""

from __future__ import annotations

import base64
from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import importlib
import importlib.util
import json
import math
from pathlib import Path
import re
import threading
from typing import Any, Callable, Mapping


GLM_MODEL = "glm-4.5v"
GLM_TEMPERATURE = 0.1
GLM_TOP_P = 0.1
GROUNDING_DINO_BOX_THRESHOLD = 0.35
GROUNDING_DINO_TEXT_THRESHOLD = 0.25
GROUNDING_DINO_CONFIG_BASENAME = "GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT_BASENAME = "groundingdino_swint_ogc.pth"
REGISTERED_GROUNDING_DINO_CONFIG_PATH = (
    "/mnt/data/quanth/cache/uv/archive-v0/hHOpLbugg_lAlUaF/"
    "groundingdino/config/GroundingDINO_SwinT_OGC.py"
)
REGISTERED_GROUNDING_DINO_CONFIG_SHA256 = (
    "172e80017f9395668a9cb5d1b8bd9d061c0e360471c6ed673c83b69bb14399f1"
)
REGISTERED_GROUNDING_DINO_CHECKPOINT_PATH = (
    "/mnt/data/quanth/cache/aegis/groundingdino/groundingdino_swint_ogc.pth"
)
REGISTERED_GROUNDING_DINO_CHECKPOINT_SHA256 = (
    "3b3ca2563c77c69f651d7bd133e97139c186df06231157a64c507099c52bc799"
)
API_KEY_ENVIRONMENT_VARIABLE = "ZHIPUAI_API_KEY"
CONFIG_ENVIRONMENT_VARIABLE = "AEGIS_GROUNDING_DINO_CONFIG"
CHECKPOINT_ENVIRONMENT_VARIABLE = "AEGIS_GROUNDING_DINO_CHECKPOINT"
CODEX_LABEL_SCHEMA_VERSION = "aegis_codex_semantic_label.v1"
CODEX_LABEL_REVIEWER = "codex"
CODEX_LABEL_RECORD_KEYS = frozenset(
    {
        "schema_version",
        "case_id",
        "instruction",
        "agentview_image_sha256",
        "obstacle_label",
        "allowed_label_vocabulary",
        "reviewer",
        "reviewed_at",
    }
)
PUBLIC_OBSTACLE_LABEL_VOCABULARY = (
    "yellow rectangular book",
    "blue moka pot",
    "red mug",
    "white storage box",
    "black wine bottle",
    "red milk carton",
)
PUBLIC_OBSTACLE_LABEL_VOCABULARY_SHA256 = (
    "b21cd3b3c5035e4706cef272d0475b673bd9eb1d31165c9c0833e3ce1da74e0d"
)
CODEX_FROZEN_LABEL_BACKEND_KIND = "codex_frozen_label_groundingdino_diagnostic"
CODEX_FROZEN_LABEL_TEST_BACKEND_KIND = (
    "injected_test_double_codex_frozen_label_groundingdino"
)
_CASE_ID_PATTERN = re.compile(r"^crfs-[0-9a-f]{16}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_REVIEWED_AT_PATTERN = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$"
)
_DINO_PREDICT_CAPTURE_LOCK = threading.Lock()


class AegisPerceptionError(RuntimeError):
    """Structured perception failure with no silent fallback."""

    def __init__(
        self, code: str, message: str, *, details: Mapping[str, Any] | None = None
    ) -> None:
        super().__init__(message)
        self.code = str(code)
        self.details = dict(details or {})

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": str(self), "details": self.details}


class AegisPerceptionApparatusError(AegisPerceptionError):
    """The diagnostic perception apparatus is unavailable or mismatched."""


class AegisPerceptionMethodError(AegisPerceptionError):
    """The dependency-complete diagnostic perception method returned no geometry."""


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: str, *, label: str) -> str:
    digest = str(value)
    if _SHA256_PATTERN.fullmatch(digest) is None:
        raise AegisPerceptionApparatusError(
            "invalid_sha256",
            f"{label} must be a lowercase SHA-256 digest",
            details={"label": label},
        )
    return digest


@dataclass(frozen=True)
class CodexSemanticLabelRecord:
    schema_version: str
    case_id: str
    instruction: str
    agentview_image_sha256: str
    obstacle_label: str
    allowed_label_vocabulary: tuple[str, ...]
    reviewer: str
    reviewed_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "case_id": self.case_id,
            "instruction": self.instruction,
            "agentview_image_sha256": self.agentview_image_sha256,
            "obstacle_label": self.obstacle_label,
            "allowed_label_vocabulary": list(self.allowed_label_vocabulary),
            "reviewer": self.reviewer,
            "reviewed_at": self.reviewed_at,
        }


@dataclass(frozen=True)
class CodexSemanticLabelLedger:
    path: str
    sha256: str
    records: tuple[CodexSemanticLabelRecord, ...]

    def for_case(self, case_id: str) -> CodexSemanticLabelRecord:
        matches = [record for record in self.records if record.case_id == case_id]
        if len(matches) != 1:
            raise AegisPerceptionApparatusError(
                "codex_label_case_mismatch",
                "immutable Codex label ledger must contain exactly one matching case",
                details={"case_id": str(case_id), "matches": len(matches)},
            )
        return matches[0]

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "record_count": len(self.records),
            "allowed_label_vocabulary_sha256": (
                PUBLIC_OBSTACLE_LABEL_VOCABULARY_SHA256
            ),
        }


def validate_codex_semantic_label_record(
    value: Mapping[str, Any],
) -> CodexSemanticLabelRecord:
    """Validate one exact Codex-reviewed semantic-label record.

    The closed key set intentionally has no simulator object-name field.
    ``agentview_image_sha256`` binds the exact C-contiguous array record digest
    used by the paired runner after its declared public double-axis flip.
    """

    if not isinstance(value, Mapping):
        raise AegisPerceptionApparatusError(
            "invalid_codex_label_record",
            "Codex semantic-label record must be a JSON object",
        )
    keys = set(value)
    if keys != CODEX_LABEL_RECORD_KEYS:
        raise AegisPerceptionApparatusError(
            "invalid_codex_label_record_keys",
            "Codex semantic-label record has a non-canonical key set",
            details={
                "missing": sorted(CODEX_LABEL_RECORD_KEYS - keys),
                "extra": sorted(keys - CODEX_LABEL_RECORD_KEYS),
            },
        )
    if value["schema_version"] != CODEX_LABEL_SCHEMA_VERSION:
        raise AegisPerceptionApparatusError(
            "invalid_codex_label_schema",
            "Codex semantic-label schema version is not registered",
        )
    case_id = str(value["case_id"])
    if _CASE_ID_PATTERN.fullmatch(case_id) is None:
        raise AegisPerceptionApparatusError(
            "invalid_codex_label_case_id",
            "Codex semantic-label case_id is not canonical",
            details={"case_id": case_id},
        )
    image_sha256 = _require_sha256(
        str(value["agentview_image_sha256"]),
        label="agentview_image_sha256",
    )
    instruction = str(value["instruction"])
    if not instruction or instruction != instruction.strip():
        raise AegisPerceptionApparatusError(
            "invalid_codex_label_instruction",
            "Codex semantic-label instruction must be non-empty and canonical",
        )
    obstacle_label = str(value["obstacle_label"])
    vocabulary_value = value["allowed_label_vocabulary"]
    if not isinstance(vocabulary_value, list) or any(
        not isinstance(label, str) for label in vocabulary_value
    ):
        raise AegisPerceptionApparatusError(
            "invalid_codex_label_vocabulary",
            "allowed_label_vocabulary must be the registered ordered JSON list",
        )
    vocabulary = tuple(vocabulary_value)
    if vocabulary != PUBLIC_OBSTACLE_LABEL_VOCABULARY:
        raise AegisPerceptionApparatusError(
            "invalid_codex_label_vocabulary",
            "Codex label vocabulary differs from the frozen ordered public list",
            details={
                "expected": list(PUBLIC_OBSTACLE_LABEL_VOCABULARY),
                "actual": list(vocabulary),
                "expected_sha256": PUBLIC_OBSTACLE_LABEL_VOCABULARY_SHA256,
            },
        )
    if obstacle_label not in PUBLIC_OBSTACLE_LABEL_VOCABULARY:
        raise AegisPerceptionApparatusError(
            "invalid_codex_public_label",
            "Codex semantic label is outside the public AEGIS vocabulary",
            details={
                "label": obstacle_label,
                "allowed": list(PUBLIC_OBSTACLE_LABEL_VOCABULARY),
            },
        )
    if value["reviewer"] != CODEX_LABEL_REVIEWER:
        raise AegisPerceptionApparatusError(
            "invalid_codex_label_reviewer",
            "Codex semantic-label reviewer must be exactly 'codex'",
        )
    reviewed_at = str(value["reviewed_at"])
    if _REVIEWED_AT_PATTERN.fullmatch(reviewed_at) is None:
        raise AegisPerceptionApparatusError(
            "invalid_codex_label_reviewed_at",
            "reviewed_at must be a canonical UTC second timestamp",
        )
    try:
        datetime.strptime(reviewed_at, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as error:
        raise AegisPerceptionApparatusError(
            "invalid_codex_label_reviewed_at",
            "reviewed_at is not a valid UTC calendar timestamp",
        ) from error
    return CodexSemanticLabelRecord(
        schema_version=CODEX_LABEL_SCHEMA_VERSION,
        case_id=case_id,
        instruction=instruction,
        agentview_image_sha256=image_sha256,
        obstacle_label=obstacle_label,
        allowed_label_vocabulary=vocabulary,
        reviewer=CODEX_LABEL_REVIEWER,
        reviewed_at=reviewed_at,
    )


def load_codex_semantic_label_jsonl(
    path: str | Path,
    *,
    expected_sha256: str,
) -> CodexSemanticLabelLedger:
    """Load a content-bound, closed-schema Codex semantic-label JSONL file."""

    expected = _require_sha256(expected_sha256, label="expected label JSONL SHA-256")
    label_path = Path(path).expanduser().resolve()
    if not label_path.is_file():
        raise AegisPerceptionApparatusError(
            "missing_codex_label_jsonl",
            "Codex semantic-label JSONL is not a file",
            details={"path": str(label_path)},
        )
    payload = label_path.read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected:
        raise AegisPerceptionApparatusError(
            "codex_label_jsonl_hash_mismatch",
            "Codex semantic-label JSONL differs from its immutable digest",
            details={"expected": expected, "actual": actual},
        )
    records: list[CodexSemanticLabelRecord] = []
    seen_cases: set[str] = set()
    for line_number, raw_line in enumerate(payload.splitlines(), start=1):
        if not raw_line.strip():
            raise AegisPerceptionApparatusError(
                "invalid_codex_label_jsonl",
                "Codex semantic-label JSONL contains a blank line",
                details={"line_number": line_number},
            )
        try:
            value = json.loads(raw_line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise AegisPerceptionApparatusError(
                "invalid_codex_label_jsonl",
                "Codex semantic-label JSONL contains invalid JSON",
                details={"line_number": line_number},
            ) from error
        record = validate_codex_semantic_label_record(value)
        if record.case_id in seen_cases:
            raise AegisPerceptionApparatusError(
                "duplicate_codex_label_case",
                "Codex semantic-label JSONL contains a duplicate case_id",
                details={"case_id": record.case_id},
            )
        seen_cases.add(record.case_id)
        records.append(record)
    if not records:
        raise AegisPerceptionApparatusError(
            "empty_codex_label_jsonl",
            "Codex semantic-label JSONL must contain at least one record",
        )
    return CodexSemanticLabelLedger(
        path=str(label_path),
        sha256=actual,
        records=tuple(records),
    )


def preferred_obstacles(task_suite_name: str) -> list[str]:
    values = [
        "yellow rectangular book",
        "blue moka pot",
        "red mug",
        "white storage box",
        "black wine bottle",
        "red milk carton",
    ]
    if "long" in str(task_suite_name):
        values.append("gray rectangular binder")
    return values


def build_public_obstacle_prompt(
    instruction: str,
    task_suite_name: str,
) -> str:
    """Return the exact public prompt text from ``main/utils.py``."""

    preferences = preferred_obstacles(task_suite_name)
    return (
        f"The robot must follow this instruction: {instruction}. Based on both the "
        "instruction and the image, identify exactly one non-robot object that is most "
        "likely to obstruct the robot's motion during task execution. You must output a "
        "uniquely identifiable obstacle name including both color and object type, "
        f"preferably from this list when applicable: {preferences}. Output only the "
        "object name, with no additional words."
    )


def clean_public_obstacle_response(value: str) -> str:
    return str(value).replace("<|begin_of_box|>", "").replace("<|end_of_box|>", "")


@dataclass(frozen=True)
class OriginalPerceptionDependencies:
    config_path: str
    config_sha256: str
    checkpoint_path: str
    checkpoint_sha256: str
    api_key_environment_variable: str
    api_key_present: bool
    packages: dict[str, bool]
    cuda_visible: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GroundingDinoOnlyDependencies:
    config_path: str
    config_sha256: str
    checkpoint_path: str
    checkpoint_sha256: str
    packages: dict[str, bool]
    cuda_visible: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_original_perception_dependencies(
    environment: Mapping[str, str],
    *,
    expected_config_sha256: str | None = None,
    expected_checkpoint_sha256: str | None = None,
    require_packages: bool = True,
    require_cuda: bool = True,
) -> OriginalPerceptionDependencies:
    """Resolve the exact public stack without exposing the API key value."""

    key = str(environment.get(API_KEY_ENVIRONMENT_VARIABLE, "")).strip()
    if not key:
        raise AegisPerceptionError(
            "missing_glm45v_credential",
            f"{API_KEY_ENVIRONMENT_VARIABLE} is not present in the allocation environment",
        )
    config_value = str(environment.get(CONFIG_ENVIRONMENT_VARIABLE, "")).strip()
    checkpoint_value = str(environment.get(CHECKPOINT_ENVIRONMENT_VARIABLE, "")).strip()
    if not config_value:
        raise AegisPerceptionError(
            "missing_groundingdino_config_path",
            f"{CONFIG_ENVIRONMENT_VARIABLE} is not set",
        )
    if not checkpoint_value:
        raise AegisPerceptionError(
            "missing_groundingdino_checkpoint_path",
            f"{CHECKPOINT_ENVIRONMENT_VARIABLE} is not set",
        )
    config = Path(config_value).expanduser().resolve()
    checkpoint = Path(checkpoint_value).expanduser().resolve()
    for code, label, path, basename in (
        (
            "missing_groundingdino_config",
            "config",
            config,
            GROUNDING_DINO_CONFIG_BASENAME,
        ),
        (
            "missing_groundingdino_checkpoint",
            "checkpoint",
            checkpoint,
            GROUNDING_DINO_CHECKPOINT_BASENAME,
        ),
    ):
        if not path.is_file():
            raise AegisPerceptionError(
                code,
                f"GroundingDINO {label} is not a file",
                details={"path": str(path)},
            )
        if path.name != basename:
            raise AegisPerceptionError(
                f"wrong_groundingdino_{label}_basename",
                f"GroundingDINO {label} must be named {basename}",
                details={"path": str(path)},
            )
    config_hash = file_sha256(config)
    checkpoint_hash = file_sha256(checkpoint)
    if expected_config_sha256 is not None and config_hash != expected_config_sha256:
        raise AegisPerceptionError(
            "groundingdino_config_hash_mismatch",
            "GroundingDINO config hash differs from the execution release",
            details={"expected": expected_config_sha256, "actual": config_hash},
        )
    if (
        expected_checkpoint_sha256 is not None
        and checkpoint_hash != expected_checkpoint_sha256
    ):
        raise AegisPerceptionError(
            "groundingdino_checkpoint_hash_mismatch",
            "GroundingDINO checkpoint hash differs from the execution release",
            details={"expected": expected_checkpoint_sha256, "actual": checkpoint_hash},
        )
    package_names = ("zai", "groundingdino", "cv2", "open3d", "cvxpy", "scipy")
    package_status = {
        name: importlib.util.find_spec(name) is not None for name in package_names
    }
    missing_packages = [name for name, present in package_status.items() if not present]
    if require_packages and missing_packages:
        raise AegisPerceptionError(
            "missing_public_aegis_packages",
            "The allocation runtime is missing public AEGIS dependencies",
            details={"packages": missing_packages},
        )
    visible_device = str(environment.get("CUDA_VISIBLE_DEVICES", "")).strip()
    cuda_visible = bool(visible_device and visible_device != "NoDevFiles")
    if require_cuda and not cuda_visible:
        raise AegisPerceptionError(
            "no_allocation_visible_gpu",
            "Original AEGIS perception requires an allocation-visible GPU",
        )
    return OriginalPerceptionDependencies(
        config_path=str(config),
        config_sha256=config_hash,
        checkpoint_path=str(checkpoint),
        checkpoint_sha256=checkpoint_hash,
        api_key_environment_variable=API_KEY_ENVIRONMENT_VARIABLE,
        api_key_present=True,
        packages=package_status,
        cuda_visible=cuda_visible,
    )


def resolve_groundingdino_only_dependencies(
    environment: Mapping[str, str],
    *,
    expected_config_sha256: str,
    expected_checkpoint_sha256: str,
    require_packages: bool = True,
    require_cuda: bool = True,
) -> GroundingDinoOnlyDependencies:
    """Resolve the frozen-label diagnostic stack without consulting a GLM key."""

    expected_config = _require_sha256(
        expected_config_sha256,
        label="expected GroundingDINO config SHA-256",
    )
    expected_checkpoint = _require_sha256(
        expected_checkpoint_sha256,
        label="expected GroundingDINO checkpoint SHA-256",
    )
    config_value = str(environment.get(CONFIG_ENVIRONMENT_VARIABLE, "")).strip()
    checkpoint_value = str(environment.get(CHECKPOINT_ENVIRONMENT_VARIABLE, "")).strip()
    if not config_value:
        raise AegisPerceptionApparatusError(
            "missing_groundingdino_config_path",
            f"{CONFIG_ENVIRONMENT_VARIABLE} is not set",
        )
    if not checkpoint_value:
        raise AegisPerceptionApparatusError(
            "missing_groundingdino_checkpoint_path",
            f"{CHECKPOINT_ENVIRONMENT_VARIABLE} is not set",
        )
    config = Path(config_value).expanduser().resolve()
    checkpoint = Path(checkpoint_value).expanduser().resolve()
    for code, label, path, basename in (
        (
            "missing_groundingdino_config",
            "config",
            config,
            GROUNDING_DINO_CONFIG_BASENAME,
        ),
        (
            "missing_groundingdino_checkpoint",
            "checkpoint",
            checkpoint,
            GROUNDING_DINO_CHECKPOINT_BASENAME,
        ),
    ):
        if not path.is_file():
            raise AegisPerceptionApparatusError(
                code,
                f"GroundingDINO {label} is not a file",
                details={"path": str(path)},
            )
        if path.name != basename:
            raise AegisPerceptionApparatusError(
                f"wrong_groundingdino_{label}_basename",
                f"GroundingDINO {label} must be named {basename}",
                details={"path": str(path)},
            )
    config_hash = file_sha256(config)
    checkpoint_hash = file_sha256(checkpoint)
    if config_hash != expected_config:
        raise AegisPerceptionApparatusError(
            "groundingdino_config_hash_mismatch",
            "GroundingDINO config hash differs from the diagnostic release",
            details={"expected": expected_config, "actual": config_hash},
        )
    if checkpoint_hash != expected_checkpoint:
        raise AegisPerceptionApparatusError(
            "groundingdino_checkpoint_hash_mismatch",
            "GroundingDINO checkpoint hash differs from the diagnostic release",
            details={"expected": expected_checkpoint, "actual": checkpoint_hash},
        )
    package_names = (
        "numpy",
        "torch",
        "groundingdino",
        "cv2",
        "matplotlib",
        "open3d",
        "cvxpy",
        "scipy",
        "robosuite",
    )
    package_status = {
        name: importlib.util.find_spec(name) is not None for name in package_names
    }
    missing_packages = [name for name, present in package_status.items() if not present]
    if require_packages and missing_packages:
        raise AegisPerceptionApparatusError(
            "missing_groundingdino_only_packages",
            "The allocation runtime is missing frozen-label AEGIS dependencies",
            details={"packages": missing_packages},
        )
    visible_device = str(environment.get("CUDA_VISIBLE_DEVICES", "")).strip()
    cuda_visible = bool(visible_device and visible_device != "NoDevFiles")
    if require_cuda and not cuda_visible:
        raise AegisPerceptionApparatusError(
            "no_allocation_visible_gpu",
            "GroundingDINO-only diagnostic requires an allocation-visible GPU",
        )
    return GroundingDinoOnlyDependencies(
        config_path=str(config),
        config_sha256=config_hash,
        checkpoint_path=str(checkpoint),
        checkpoint_sha256=checkpoint_hash,
        packages=package_status,
        cuda_visible=cuda_visible,
    )


def call_public_glm_obstacle_selector(
    image_path: str | Path,
    *,
    instruction: str,
    task_suite_name: str,
    api_key: str,
    client_factory: Any | None = None,
) -> dict[str, Any]:
    """Execute the public GLM-4.5V selector and return auditable metadata.

    The secret is accepted only as a call argument and is never returned.
    Tests inject ``client_factory``; the real allocation lazily imports the
    same ``ZhipuAiClient`` used by the public code.
    """

    secret = str(api_key).strip()
    if not secret:
        raise AegisPerceptionError(
            "missing_glm45v_credential", "GLM-4.5V credential is empty"
        )
    path = Path(image_path)
    if not path.is_file():
        raise AegisPerceptionError(
            "missing_glm45v_image",
            "GLM-4.5V input image does not exist",
            details={"path": str(path)},
        )
    if client_factory is None:
        from zai import ZhipuAiClient

        client_factory = ZhipuAiClient
    client = client_factory(api_key=secret)
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    prompt = build_public_obstacle_prompt(instruction, task_suite_name)
    response = client.chat.completions.create(
        model=GLM_MODEL,
        messages=[
            {
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
                    },
                    {"type": "text", "text": prompt},
                ],
                "role": "user",
            }
        ],
        temperature=GLM_TEMPERATURE,
        top_p=GLM_TOP_P,
        thinking={"type": "enabled"},
    )
    try:
        raw_content = str(response.choices[0].message.content)
    except (AttributeError, IndexError, TypeError) as error:
        raise AegisPerceptionError(
            "invalid_glm45v_response",
            "GLM-4.5V response has no first message content",
        ) from error
    cleaned = clean_public_obstacle_response(raw_content)
    if not cleaned.strip():
        raise AegisPerceptionError(
            "empty_glm45v_response", "GLM-4.5V returned an empty obstacle name"
        )
    return {
        "model": GLM_MODEL,
        "temperature": GLM_TEMPERATURE,
        "top_p": GLM_TOP_P,
        "thinking": {"type": "enabled"},
        "prompt": prompt,
        "image_path": str(path.resolve()),
        "image_sha256": file_sha256(path),
        "raw_content": raw_content,
        "cleaned_obstacle_name": cleaned,
        "credential_environment_variable": API_KEY_ENVIRONMENT_VARIABLE,
        "credential_value_recorded": False,
    }


def _lazy_numpy() -> Any:
    try:
        return importlib.import_module("numpy")
    except (ImportError, ModuleNotFoundError) as error:
        raise AegisPerceptionApparatusError(
            "missing_numpy",
            "Frozen-label GroundingDINO backend requires NumPy",
        ) from error


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def exact_perception_array_record(
    value: Any,
    *,
    label: str,
    numpy_module: Any | None = None,
) -> dict[str, Any]:
    """Hash an array exactly as the paired runner hashes its 1024 inputs."""

    np = _lazy_numpy() if numpy_module is None else numpy_module
    array = np.asarray(value)
    if bool(getattr(array.dtype, "hasobject", False)):
        raise AegisPerceptionApparatusError(
            "invalid_perception_array",
            f"{label} cannot use object dtype",
        )
    try:
        finite = bool(np.isfinite(array).all())
    except (TypeError, ValueError) as error:
        raise AegisPerceptionApparatusError(
            "invalid_perception_array",
            f"{label} must be a finite numeric array",
        ) from error
    if not finite:
        raise AegisPerceptionApparatusError(
            "invalid_perception_array",
            f"{label} contains non-finite values",
        )
    contiguous = np.ascontiguousarray(array)
    header = {
        "dtype": contiguous.dtype.str,
        "dtype_name": str(contiguous.dtype),
        "shape": [int(item) for item in contiguous.shape],
        "order": "C",
    }
    digest = hashlib.sha256()
    digest.update(_canonical_json_bytes(header))
    digest.update(b"\0")
    digest.update(contiguous.tobytes(order="C"))
    return {
        "schema_version": "1.0",
        "kind": "ndarray",
        **header,
        "nbytes": int(contiguous.nbytes),
        "sha256": digest.hexdigest(),
    }


def agentview_1024_sha256(
    value: Any,
    *,
    numpy_module: Any | None = None,
) -> str:
    record = exact_perception_array_record(
        value,
        label="agentview_1024_image",
        numpy_module=numpy_module,
    )
    if record["shape"] != [1024, 1024, 3]:
        raise AegisPerceptionApparatusError(
            "invalid_agentview_1024_shape",
            "Codex label image must have exact shape 1024x1024x3",
            details={"shape": record["shape"]},
        )
    return str(record["sha256"])


def _array_record_with_values(np: Any, value: Any, *, label: str) -> dict[str, Any]:
    array = np.asarray(value)
    record = exact_perception_array_record(
        array,
        label=label,
        numpy_module=np,
    )
    return {**record, "values": array.tolist()}


def _load_public_utils(repo_root: str | Path) -> tuple[Any, str]:
    path = Path(repo_root).expanduser().resolve() / "main/utils.py"
    if not path.is_file():
        raise AegisPerceptionApparatusError(
            "missing_public_utils",
            "byte-preserved public AEGIS utils.py is missing",
            details={"path": str(path)},
        )
    spec = importlib.util.spec_from_file_location(
        "crfs_public_aegis_utils_runtime",
        path,
    )
    if spec is None or spec.loader is None:
        raise AegisPerceptionApparatusError(
            "invalid_public_utils",
            "could not construct a loader for public AEGIS utils.py",
        )
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as error:
        raise AegisPerceptionApparatusError(
            "public_utils_import_failure",
            "public AEGIS utils.py failed to import in the allocation",
            details={"exception_type": type(error).__name__},
        ) from error
    return module, file_sha256(path)


def _default_model_loader(dependencies: GroundingDinoOnlyDependencies) -> Any:
    try:
        inference = importlib.import_module("groundingdino.util.inference")
        return inference.load_model(
            dependencies.config_path,
            dependencies.checkpoint_path,
        )
    except Exception as error:
        raise AegisPerceptionApparatusError(
            "groundingdino_model_load_failure",
            "GroundingDINO model failed to load",
            details={"exception_type": type(error).__name__},
        ) from error


def _default_public_point_cloud_runner(
    *,
    np: Any,
    public_utils: Any,
    image: Any,
    depth: Any,
    environment: Any,
    view: str,
    obstacle_label: str,
    model: Any,
    output_dir: Path,
) -> tuple[Any, dict[str, Any]]:
    """Call public ``get_point_cloud`` while capturing its actual prediction."""

    try:
        inference = importlib.import_module("groundingdino.util.inference")
    except (ImportError, ModuleNotFoundError) as error:
        raise AegisPerceptionApparatusError(
            "missing_groundingdino_inference",
            "GroundingDINO inference module is unavailable",
        ) from error
    original_predict = inference.predict
    predictions: list[dict[str, Any]] = []

    def capturing_predict(*args: Any, **kwargs: Any) -> Any:
        expected_request = {
            "caption": obstacle_label,
            "box_threshold": GROUNDING_DINO_BOX_THRESHOLD,
            "text_threshold": GROUNDING_DINO_TEXT_THRESHOLD,
            "device": "cuda",
        }
        actual_request = {
            name: kwargs.get(name)
            for name in (
                "caption",
                "box_threshold",
                "text_threshold",
                "device",
            )
        }
        if actual_request != expected_request:
            raise AegisPerceptionApparatusError(
                "groundingdino_request_drift",
                "public get_point_cloud changed a frozen GroundingDINO request",
                details={
                    "view": view,
                    "expected": expected_request,
                    "actual": actual_request,
                },
            )
        result = original_predict(*args, **kwargs)
        if not isinstance(result, tuple) or len(result) != 3:
            raise AegisPerceptionMethodError(
                "invalid_groundingdino_prediction",
                "GroundingDINO predict did not return boxes, logits, and phrases",
                details={"view": view},
            )
        boxes, logits, phrases = result
        box_values = np.asarray(boxes).tolist()
        phrase_values = [str(phrase) for phrase in phrases]
        has_box = int(np.asarray(boxes).shape[0]) > 0
        predictions.append(
            {
                **actual_request,
                "boxes": _array_record_with_values(
                    np, boxes, label=f"{view}_groundingdino_boxes"
                ),
                "logits": _array_record_with_values(
                    np, logits, label=f"{view}_groundingdino_logits"
                ),
                "phrases": phrase_values,
                "selected_box_index": 0 if has_box else None,
                "selected_box": box_values[0] if has_box else None,
                "selected_phrase": phrase_values[0] if phrase_values else None,
            }
        )
        return result

    with _DINO_PREDICT_CAPTURE_LOCK:
        inference.predict = capturing_predict
        try:
            points = public_utils.get_point_cloud(
                image,
                depth,
                environment,
                view,
                obstacle_label,
                model,
                output_dir,
            )
        finally:
            inference.predict = original_predict
    if len(predictions) != 1:
        raise AegisPerceptionMethodError(
            "groundingdino_prediction_count_failure",
            "public get_point_cloud must invoke GroundingDINO predict exactly once",
            details={"view": view, "prediction_calls": len(predictions)},
        )
    return points, predictions[0]


def _validate_groundingdino_prediction_audit(
    prediction: Mapping[str, Any],
    *,
    view: str,
    obstacle_label: str,
) -> dict[str, Any]:
    """Validate evidence for the exact first box consumed by public utils."""

    expected_request = {
        "caption": obstacle_label,
        "box_threshold": GROUNDING_DINO_BOX_THRESHOLD,
        "text_threshold": GROUNDING_DINO_TEXT_THRESHOLD,
        "device": "cuda",
    }
    actual_request = {name: prediction.get(name) for name in expected_request}
    if actual_request != expected_request:
        raise AegisPerceptionMethodError(
            "incomplete_groundingdino_request_audit",
            "GroundingDINO audit does not prove the frozen request parameters",
            details={
                "view": view,
                "expected": expected_request,
                "actual": actual_request,
            },
        )
    boxes = prediction.get("boxes")
    phrases = prediction.get("phrases")
    if not isinstance(boxes, Mapping) or not isinstance(phrases, list):
        raise AegisPerceptionMethodError(
            "incomplete_groundingdino_audit",
            "GroundingDINO audit must contain boxes and phrases",
            details={"view": view},
        )
    shape = boxes.get("shape")
    values = boxes.get("values")
    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or shape[1:] != [4]
        or not isinstance(values, list)
        or shape[0] != len(values)
    ):
        raise AegisPerceptionMethodError(
            "invalid_groundingdino_box_audit",
            "GroundingDINO box audit has an invalid shape or value count",
            details={"view": view},
        )
    if not values:
        if (
            prediction.get("selected_box_index") is not None
            or prediction.get("selected_box") is not None
            or prediction.get("selected_phrase") is not None
            or phrases
        ):
            raise AegisPerceptionMethodError(
                "invalid_empty_groundingdino_audit",
                "an empty GroundingDINO view must not claim a selected box or phrase",
                details={"view": view},
            )
        # The public entrypoint permits one camera to return no points and uses
        # the other camera.  Overall no-detection is rejected later, after both
        # public point-cloud calls have been combined.
        return dict(prediction)
    selected_box = prediction.get("selected_box")
    if (
        prediction.get("selected_box_index") != 0
        or selected_box != values[0]
        or not isinstance(selected_box, list)
        or len(selected_box) != 4
    ):
        raise AegisPerceptionMethodError(
            "invalid_groundingdino_selected_box_audit",
            "audit does not bind the first box selected by public get_point_cloud",
            details={"view": view},
        )
    try:
        selected_box_finite = all(math.isfinite(float(item)) for item in selected_box)
    except (TypeError, ValueError):
        selected_box_finite = False
    if not selected_box_finite:
        raise AegisPerceptionMethodError(
            "invalid_groundingdino_selected_box_audit",
            "selected GroundingDINO box must contain four finite values",
            details={"view": view},
        )
    if not phrases or prediction.get("selected_phrase") != phrases[0]:
        raise AegisPerceptionMethodError(
            "invalid_groundingdino_selected_phrase_audit",
            "audit does not bind the first GroundingDINO phrase",
            details={"view": view},
        )
    return dict(prediction)


def _point_count(np: Any, value: Any, *, label: str) -> tuple[Any, int]:
    points = np.asarray(value)
    if len(points.shape) != 2:
        raise AegisPerceptionMethodError(
            "invalid_public_point_cloud",
            f"{label} point cloud must be two-dimensional",
            details={"shape": [int(item) for item in points.shape]},
        )
    if tuple(int(item) for item in points.shape) == (1, 0):
        return points, 0
    if int(points.shape[1]) != 3:
        raise AegisPerceptionMethodError(
            "invalid_public_point_cloud",
            f"{label} point cloud must have three columns",
            details={"shape": [int(item) for item in points.shape]},
        )
    exact_perception_array_record(
        points,
        label=f"{label}_point_cloud",
        numpy_module=np,
    )
    return points, int(points.shape[0])


def _run_codex_frozen_label_groundingdino_core(
    *,
    label_ledger: CodexSemanticLabelLedger,
    case_id: str,
    views: Mapping[str, Any],
    instruction: str,
    task_suite: str,
    runtime: Any,
    dependencies: GroundingDinoOnlyDependencies,
    output_dir: str | Path,
    repo_root: str | Path,
    _public_utils: Any | None = None,
    _model_loader: Callable[[GroundingDinoOnlyDependencies], Any] | None = None,
    _point_cloud_runner: Callable[..., tuple[Any, Mapping[str, Any]]] | None = None,
    _numpy_module: Any | None = None,
    test_injection_used: bool,
) -> dict[str, Any]:
    """Run the public geometry stack with an immutable Codex semantic label.

    This is an explicitly diagnostic replacement for unavailable GLM-4.5V
    semantics.  It cannot satisfy the canonical ``original_end_to_end`` arm.
    The only geometry source is public two-view GroundingDINO RGB-D points,
    public filtering, and public MVEE; no object identity or geometry is read
    from simulator metadata.
    """

    if not isinstance(label_ledger, CodexSemanticLabelLedger):
        raise AegisPerceptionApparatusError(
            "invalid_codex_label_ledger",
            "runtime backend requires a validated immutable Codex label ledger",
        )
    if not isinstance(dependencies, GroundingDinoOnlyDependencies):
        raise AegisPerceptionApparatusError(
            "invalid_groundingdino_dependencies",
            "runtime backend requires validated GroundingDINO-only dependencies",
        )
    injected_hooks = (
        _public_utils,
        _model_loader,
        _point_cloud_runner,
        _numpy_module,
    )
    any_injected_hook = any(value is not None for value in injected_hooks)
    all_injected_hooks = all(value is not None for value in injected_hooks)
    if any_injected_hook != bool(test_injection_used) or (
        test_injection_used and not all_injected_hooks
    ):
        raise AegisPerceptionApparatusError(
            "invalid_perception_test_injection_state",
            "dependency injection is forbidden in the release runtime backend",
        )
    if (
        dependencies.config_path != REGISTERED_GROUNDING_DINO_CONFIG_PATH
        or dependencies.config_sha256 != REGISTERED_GROUNDING_DINO_CONFIG_SHA256
    ):
        raise AegisPerceptionApparatusError(
            "unregistered_groundingdino_config",
            "diagnostic backend requires the exact setup-job GroundingDINO config",
            details={
                "expected_path": REGISTERED_GROUNDING_DINO_CONFIG_PATH,
                "expected_sha256": REGISTERED_GROUNDING_DINO_CONFIG_SHA256,
                "actual_path": dependencies.config_path,
                "actual_sha256": dependencies.config_sha256,
            },
        )
    if (
        dependencies.checkpoint_path != REGISTERED_GROUNDING_DINO_CHECKPOINT_PATH
        or dependencies.checkpoint_sha256 != REGISTERED_GROUNDING_DINO_CHECKPOINT_SHA256
    ):
        raise AegisPerceptionApparatusError(
            "unregistered_groundingdino_checkpoint",
            "diagnostic backend requires the exact setup-job GroundingDINO checkpoint",
            details={
                "expected_path": REGISTERED_GROUNDING_DINO_CHECKPOINT_PATH,
                "expected_sha256": REGISTERED_GROUNDING_DINO_CHECKPOINT_SHA256,
                "actual_path": dependencies.checkpoint_path,
                "actual_sha256": dependencies.checkpoint_sha256,
            },
        )
    record = label_ledger.for_case(str(case_id))
    if str(instruction) != record.instruction:
        raise AegisPerceptionApparatusError(
            "codex_label_instruction_mismatch",
            "runtime instruction differs from the reviewed Codex label record",
        )
    if record.obstacle_label not in preferred_obstacles(str(task_suite)):
        raise AegisPerceptionApparatusError(
            "codex_label_suite_mismatch",
            "reviewed public label is not allowed for this task suite",
            details={
                "label": record.obstacle_label,
                "task_suite": str(task_suite),
            },
        )
    if views.get("public_double_axis_flip_applied") is not True:
        raise AegisPerceptionApparatusError(
            "aegis_view_orientation_mismatch",
            "1024 views must be direct simulator renders with the public pre-call flip",
        )
    if views.get("render_advanced_physics") is not False:
        raise AegisPerceptionApparatusError(
            "aegis_view_render_mutation",
            "1024 diagnostic views must not advance simulator physics",
        )

    np = _lazy_numpy() if _numpy_module is None else _numpy_module
    required_views = (
        "agentview_image",
        "agentview_depth",
        "backview_image",
        "backview_depth",
    )
    missing = [name for name in required_views if name not in views]
    if missing:
        raise AegisPerceptionApparatusError(
            "missing_aegis_views",
            "GroundingDINO-only backend is missing required RGB-D views",
            details={"missing": missing},
        )
    orientation = views.get("camera_orientation")
    if not isinstance(orientation, Mapping) or not (
        orientation.get("robosuite_version") == "1.4.1"
        and orientation.get("live_image_convention") == "opengl"
        and orientation.get("image_convention_mapping_step") == 1
        and orientation.get("observable_mapping_applied_before_public_flip") is True
        and orientation.get("same_state_224_observable_anchor_passed") is True
        and orientation.get("same_state_224_policy_agent_anchor_passed") is True
    ):
        raise AegisPerceptionApparatusError(
            "aegis_view_observation_anchor_missing",
            "1024 views lack the exact same-state robosuite observation anchor",
        )
    view_records: dict[str, dict[str, Any]] = {}
    for camera in ("agentview", "backview"):
        image_record = exact_perception_array_record(
            views[f"{camera}_image"],
            label=f"{camera}_1024_image",
            numpy_module=np,
        )
        depth_record = exact_perception_array_record(
            views[f"{camera}_depth"],
            label=f"{camera}_1024_depth",
            numpy_module=np,
        )
        if image_record["shape"] != [1024, 1024, 3]:
            raise AegisPerceptionApparatusError(
                "invalid_aegis_rgb_shape",
                f"{camera} RGB must have exact shape 1024x1024x3",
                details={"shape": image_record["shape"]},
            )
        if depth_record["shape"] not in (
            [1024, 1024],
            [1024, 1024, 1],
        ):
            raise AegisPerceptionApparatusError(
                "invalid_aegis_depth_shape",
                f"{camera} depth has an invalid 1024 render shape",
                details={"shape": depth_record["shape"]},
            )
        view_records[camera] = {
            "image": image_record,
            "depth": depth_record,
        }
    if (
        view_records["agentview"]["image"]["sha256"]
        != record.agentview_image_sha256
    ):
        raise AegisPerceptionApparatusError(
            "codex_label_image_mismatch",
            "runtime agentview image differs from the Codex-reviewed 1024 image",
            details={
                "expected": record.agentview_image_sha256,
                "actual": view_records["agentview"]["image"]["sha256"],
            },
        )

    if _public_utils is None:
        public_utils, public_utils_sha256 = _load_public_utils(repo_root)
        public_utils_source = "byte_preserved_main_utils"
    else:
        public_utils = _public_utils
        public_utils_sha256 = None
        public_utils_source = "injected_test_double"
    for function_name in ("get_point_cloud", "filtering_points", "fit_ellipse"):
        if not callable(getattr(public_utils, function_name, None)):
            raise AegisPerceptionApparatusError(
                "invalid_public_utils",
                f"public AEGIS utils lacks callable {function_name}",
            )
    model_loader = _default_model_loader if _model_loader is None else _model_loader
    point_cloud_runner = (
        _default_public_point_cloud_runner
        if _point_cloud_runner is None
        else _point_cloud_runner
    )
    model = model_loader(dependencies)
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    environment = getattr(runtime, "env", runtime)

    raw_points: dict[str, Any] = {}
    prediction_records: dict[str, Mapping[str, Any]] = {}
    raw_counts: dict[str, int] = {}
    for camera in ("agentview", "backview"):
        points_value, prediction = point_cloud_runner(
            np=np,
            public_utils=public_utils,
            image=views[f"{camera}_image"],
            depth=views[f"{camera}_depth"],
            environment=environment,
            view=camera,
            obstacle_label=record.obstacle_label,
            model=model,
            output_dir=output_path,
        )
        if not isinstance(prediction, Mapping):
            raise AegisPerceptionMethodError(
                "missing_groundingdino_audit",
                "GroundingDINO point-cloud call returned no prediction audit",
                details={"view": camera},
            )
        validated_prediction = _validate_groundingdino_prediction_audit(
            prediction,
            view=camera,
            obstacle_label=record.obstacle_label,
        )
        points, count = _point_count(
            np,
            points_value,
            label=camera,
        )
        raw_points[camera] = points
        raw_counts[camera] = count
        prediction_records[camera] = validated_prediction
    available = [
        raw_points[camera]
        for camera in ("agentview", "backview")
        if raw_counts[camera] > 0
    ]
    if not available:
        raise AegisPerceptionMethodError(
            "no_groundingdino_points",
            "public two-view GroundingDINO produced no obstacle points",
        )
    combined = available[0] if len(available) == 1 else np.vstack(available)
    combined, combined_count = _point_count(
        np,
        combined,
        label="combined",
    )
    filtered = public_utils.filtering_points(combined, str(task_suite))
    filtered, filtered_count = _point_count(
        np,
        filtered,
        label="filtered",
    )
    if filtered_count == 0:
        raise AegisPerceptionMethodError(
            "no_filtered_groundingdino_points",
            "public AEGIS filtering removed every GroundingDINO point",
        )
    try:
        fit = public_utils.fit_ellipse(
            filtered,
            plot=True,
            save_path=output_path,
        )
    except Exception as error:
        raise AegisPerceptionMethodError(
            "public_mvee_failure",
            "public AEGIS MVEE failed on filtered GroundingDINO points",
            details={"exception_type": type(error).__name__},
        ) from error
    if not isinstance(fit, tuple) or len(fit) != 3:
        raise AegisPerceptionMethodError(
            "invalid_public_mvee",
            "public fit_ellipse did not return p2, R2, and Q2 diagonal",
        )
    p2, r2, q2_diag = fit
    p2_record = _array_record_with_values(np, p2, label="mvee_p2")
    r2_record = _array_record_with_values(np, r2, label="mvee_r2")
    q2_record = _array_record_with_values(np, q2_diag, label="mvee_q2_diag")
    if p2_record["shape"] != [3] or q2_record["shape"] != [3]:
        raise AegisPerceptionMethodError(
            "invalid_public_mvee_shape",
            "public MVEE p2 and Q2 diagonal must be three-dimensional",
        )
    if r2_record["shape"] != [3, 3]:
        raise AegisPerceptionMethodError(
            "invalid_public_mvee_shape",
            "public MVEE R2 must be a 3x3 matrix",
        )

    point_cloud_records = {
        "agentview": {
            "count": raw_counts["agentview"],
            "array": (
                exact_perception_array_record(
                    raw_points["agentview"],
                    label="agentview_points",
                    numpy_module=np,
                )
                if raw_counts["agentview"] > 0
                else None
            ),
        },
        "backview": {
            "count": raw_counts["backview"],
            "array": (
                exact_perception_array_record(
                    raw_points["backview"],
                    label="backview_points",
                    numpy_module=np,
                )
                if raw_counts["backview"] > 0
                else None
            ),
        },
        "combined": {
            "count": combined_count,
            "array": exact_perception_array_record(
                combined,
                label="combined_points",
                numpy_module=np,
            ),
        },
        "filtered": {
            "count": filtered_count,
            "array": exact_perception_array_record(
                filtered,
                label="filtered_points",
                numpy_module=np,
            ),
        },
    }
    return {
        "status": "ok",
        "backend_kind": (
            CODEX_FROZEN_LABEL_TEST_BACKEND_KIND
            if test_injection_used
            else CODEX_FROZEN_LABEL_BACKEND_KIND
        ),
        "test_injection_used": bool(test_injection_used),
        "release_runtime_backend": not test_injection_used,
        "canonical_original_end_to_end": False,
        "diagnostic_only": True,
        "semantic_label_source": "codex_reviewed_frozen_1024_agentview",
        "original_glm_executed": False,
        "oracle_substitution_used": False,
        "obstacle_label": record.obstacle_label,
        "codex_label_record": record.to_dict(),
        "codex_label_ledger": label_ledger.to_dict(),
        "allowed_label_vocabulary_sha256": (
            PUBLIC_OBSTACLE_LABEL_VOCABULARY_SHA256
        ),
        "groundingdino": {
            "box_threshold": GROUNDING_DINO_BOX_THRESHOLD,
            "text_threshold": GROUNDING_DINO_TEXT_THRESHOLD,
            "config_path": dependencies.config_path,
            "config_sha256": dependencies.config_sha256,
            "checkpoint_path": dependencies.checkpoint_path,
            "checkpoint_sha256": dependencies.checkpoint_sha256,
            "views": prediction_records,
        },
        "inputs": {
            "views": view_records,
            "orientation": {
                "source": "robosuite_convention_mapped_direct_sim_render_rgb_depth",
                "robosuite_version": orientation["robosuite_version"],
                "live_image_convention": orientation["live_image_convention"],
                "image_convention_mapping_step": orientation[
                    "image_convention_mapping_step"
                ],
                "same_state_224_observable_anchor_passed": True,
                "same_state_224_policy_agent_anchor_passed": True,
                "controller_state_fingerprint_sha256": orientation[
                    "controller_state_fingerprint_sha256"
                ],
                "pre_public_get_point_cloud_double_axis_flip": True,
                "public_get_point_cloud_internal_reverse_expected": True,
                "render_advanced_physics": False,
                "physical_equivalence_to_observation_api_claimed": True,
            },
        },
        "point_cloud": point_cloud_records,
        "ellipsoid": {
            "p2": p2_record["values"],
            "r2": r2_record["values"],
            "q2_diag": q2_record["values"],
            "p2_record": p2_record,
            "r2_record": r2_record,
            "q2_diag_record": q2_record,
            "source": "public_filtering_points_then_public_fit_ellipse",
        },
        "public_utils": {
            "source": public_utils_source,
            "sha256": public_utils_sha256,
            "functions": [
                "get_point_cloud",
                "filtering_points",
                "fit_ellipse",
            ],
        },
        "output_dir": str(output_path),
    }


def run_codex_frozen_label_groundingdino_backend(
    *,
    label_jsonl_path: str | Path,
    expected_label_jsonl_sha256: str,
    case_id: str,
    views: Mapping[str, Any],
    instruction: str,
    task_suite: str,
    runtime: Any,
    environment: Mapping[str, str],
    output_dir: str | Path,
    repo_root: str | Path,
) -> dict[str, Any]:
    """Run the release diagnostic from rehashed labels and registered assets.

    This production entrypoint deliberately accepts neither preconstructed
    ledgers/dependency records nor dependency-injection hooks.  It re-reads and
    re-hashes the immutable label JSONL, GroundingDINO config, and checkpoint on
    every invocation before loading the public perception stack.
    """

    label_ledger = load_codex_semantic_label_jsonl(
        label_jsonl_path,
        expected_sha256=expected_label_jsonl_sha256,
    )
    dependencies = resolve_groundingdino_only_dependencies(
        environment,
        expected_config_sha256=REGISTERED_GROUNDING_DINO_CONFIG_SHA256,
        expected_checkpoint_sha256=REGISTERED_GROUNDING_DINO_CHECKPOINT_SHA256,
    )
    return _run_codex_frozen_label_groundingdino_core(
        label_ledger=label_ledger,
        case_id=case_id,
        views=views,
        instruction=instruction,
        task_suite=task_suite,
        runtime=runtime,
        dependencies=dependencies,
        output_dir=output_dir,
        repo_root=repo_root,
        test_injection_used=False,
    )


def _run_codex_frozen_label_groundingdino_backend_for_test(
    *,
    label_ledger: CodexSemanticLabelLedger,
    case_id: str,
    views: Mapping[str, Any],
    instruction: str,
    task_suite: str,
    runtime: Any,
    dependencies: GroundingDinoOnlyDependencies,
    output_dir: str | Path,
    repo_root: str | Path,
    public_utils: Any,
    model_loader: Callable[[GroundingDinoOnlyDependencies], Any],
    point_cloud_runner: Callable[..., tuple[Any, Mapping[str, Any]]],
    numpy_module: Any,
) -> dict[str, Any]:
    """Dependency-light test-only entrypoint with unmistakable telemetry."""

    return _run_codex_frozen_label_groundingdino_core(
        label_ledger=label_ledger,
        case_id=case_id,
        views=views,
        instruction=instruction,
        task_suite=task_suite,
        runtime=runtime,
        dependencies=dependencies,
        output_dir=output_dir,
        repo_root=repo_root,
        _public_utils=public_utils,
        _model_loader=model_loader,
        _point_cloud_runner=point_cloud_runner,
        _numpy_module=numpy_module,
        test_injection_used=True,
    )


__all__ = [
    "API_KEY_ENVIRONMENT_VARIABLE",
    "AegisPerceptionApparatusError",
    "AegisPerceptionError",
    "AegisPerceptionMethodError",
    "CHECKPOINT_ENVIRONMENT_VARIABLE",
    "CODEX_FROZEN_LABEL_BACKEND_KIND",
    "CODEX_FROZEN_LABEL_TEST_BACKEND_KIND",
    "CODEX_LABEL_RECORD_KEYS",
    "CODEX_LABEL_REVIEWER",
    "CODEX_LABEL_SCHEMA_VERSION",
    "CONFIG_ENVIRONMENT_VARIABLE",
    "CodexSemanticLabelLedger",
    "CodexSemanticLabelRecord",
    "GLM_MODEL",
    "GROUNDING_DINO_BOX_THRESHOLD",
    "GROUNDING_DINO_TEXT_THRESHOLD",
    "GroundingDinoOnlyDependencies",
    "OriginalPerceptionDependencies",
    "PUBLIC_OBSTACLE_LABEL_VOCABULARY",
    "PUBLIC_OBSTACLE_LABEL_VOCABULARY_SHA256",
    "REGISTERED_GROUNDING_DINO_CONFIG_PATH",
    "REGISTERED_GROUNDING_DINO_CONFIG_SHA256",
    "REGISTERED_GROUNDING_DINO_CHECKPOINT_PATH",
    "REGISTERED_GROUNDING_DINO_CHECKPOINT_SHA256",
    "agentview_1024_sha256",
    "build_public_obstacle_prompt",
    "call_public_glm_obstacle_selector",
    "clean_public_obstacle_response",
    "exact_perception_array_record",
    "file_sha256",
    "load_codex_semantic_label_jsonl",
    "preferred_obstacles",
    "resolve_groundingdino_only_dependencies",
    "resolve_original_perception_dependencies",
    "run_codex_frozen_label_groundingdino_backend",
    "validate_codex_semantic_label_record",
]
