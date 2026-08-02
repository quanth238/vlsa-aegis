"""Complete-only binding for the post-OSC allocation numerical gate.

The numerical artifact is implementation evidence, not a robot-safety result.
It is nevertheless a mandatory prerequisite for the H100 canary because the
local development interpreter intentionally lacks the MuJoCo / OSQP stack and
therefore skips the allocation-only tests.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Dict, Mapping, Optional, Sequence


NUMERIC_SCHEMA = "vlsa_poisson_numeric_validation.v1"


class OscNumericPrerequisiteError(RuntimeError):
    """Raised when allocation evidence is missing, partial, or unbound."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise OscNumericPrerequisiteError(message)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _string_sequence(value: Any, label: str) -> Sequence[str]:
    _require(
        isinstance(value, Sequence) and not isinstance(value, (str, bytes)),
        "%s must be an array" % label,
    )
    output = list(value)
    _require(
        all(isinstance(item, str) and item for item in output),
        "%s must contain nonempty strings" % label,
    )
    return output


def validate_numeric_prerequisite_artifact(
    path: Path,
    *,
    contract: Mapping[str, Any],
    expected_job_id: str,
    expected_commit: str,
    expected_python_executable: str,
    expected_parent: Optional[Path] = None,
    forbidden_job_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Validate and summarize one immutable, zero-skip H100 test artifact."""

    candidate = Path(path)
    _require(candidate.is_absolute(), "numeric artifact path must be absolute")
    _require(
        bool(re.fullmatch(r"[0-9]+", str(expected_job_id))),
        "numeric Slurm job ID is malformed",
    )
    _require(
        bool(re.fullmatch(r"[0-9a-f]{40}", str(expected_commit))),
        "numeric source commit is malformed",
    )
    _require(
        isinstance(expected_python_executable, str)
        and expected_python_executable.startswith("/")
        and expected_python_executable,
        "numeric Python path is invalid",
    )
    if forbidden_job_id is not None:
        _require(
            str(expected_job_id) != str(forbidden_job_id),
            "numeric and consuming jobs must be distinct",
        )
    if expected_parent is not None:
        parent = Path(expected_parent)
        _require(parent.is_absolute(), "numeric parent must be absolute")
        _require(
            parent.is_dir() and not parent.is_symlink(),
            "numeric parent is absent or symlinked",
        )
        _require(
            candidate.parent == parent,
            "numeric artifact is not a direct child of its registered parent",
        )
    _require(
        candidate.is_file() and not candidate.is_symlink(),
        "numeric artifact is absent or symlinked",
    )
    try:
        raw = candidate.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise OscNumericPrerequisiteError(
            "numeric artifact is unreadable: %s" % error
        ) from error
    _require(isinstance(value, Mapping), "numeric artifact must be an object")
    _require(value.get("schema_version") == NUMERIC_SCHEMA, "numeric schema differs")
    _require(value.get("status") == "passed", "numeric gate did not pass")
    _require(value.get("scientific_result") is False, "numeric artifact overclaims efficacy")
    _require(
        value.get("evidence_tier") == contract.get("evidence_tier"),
        "numeric evidence tier differs",
    )
    payload_sha = value.get("result_payload_sha256")
    _require(
        isinstance(payload_sha, str)
        and bool(re.fullmatch(r"[0-9a-f]{64}", payload_sha)),
        "numeric payload hash is malformed",
    )
    unhashed = dict(value)
    del unhashed["result_payload_sha256"]
    _require(
        hashlib.sha256(_canonical(unhashed)).hexdigest() == payload_sha,
        "numeric payload hash differs",
    )

    source = value.get("source")
    _require(isinstance(source, Mapping), "numeric source record is absent")
    _require(source.get("commit") == expected_commit, "numeric source commit differs")
    _require(source.get("status_short") == [], "numeric source was dirty")

    runtime = value.get("runtime")
    _require(isinstance(runtime, Mapping), "numeric runtime record is absent")
    _require(
        str(runtime.get("slurm_job_id")) == str(expected_job_id),
        "numeric Slurm job ID differs",
    )
    _require(runtime.get("slurm_partition") == "mig", "numeric partition differs")
    _require(
        runtime.get("python_executable") == expected_python_executable,
        "numeric Python executable differs",
    )
    cuda_visible = runtime.get("cuda_visible_devices")
    _require(
        isinstance(cuda_visible, str)
        and cuda_visible
        and "," not in cuda_visible,
        "numeric allocation did not expose exactly one CUDA device",
    )
    inventory = runtime.get("allocated_gpu_inventory")
    devices = inventory.get("devices") if isinstance(inventory, Mapping) else None
    _require(
        isinstance(inventory, Mapping)
        and inventory.get("available") is True
        and inventory.get("error") is None
        and isinstance(devices, Sequence)
        and len(devices) == 1
        and isinstance(devices[0], Mapping)
        and "H100" in str(devices[0].get("name", ""))
        and bool(devices[0].get("uuid"))
        and bool(devices[0].get("driver_version")),
        "numeric single-H100 identity differs",
    )
    packages = runtime.get("packages")
    _require(isinstance(packages, Mapping), "numeric package inventory is absent")
    for name in ("numpy", "scipy", "mujoco", "osqp"):
        _require(
            isinstance(packages.get(name), str) and packages.get(name),
            "numeric package %s is absent" % name,
        )

    tests = value.get("tests")
    _require(isinstance(tests, Mapping), "numeric test record is absent")
    required_modules = list(
        _string_sequence(
            contract.get("required_test_modules"),
            "numeric_prerequisite.required_test_modules",
        )
    )
    observed_modules = list(_string_sequence(tests.get("modules"), "numeric test modules"))
    _require(observed_modules == required_modules, "numeric test module set or order differs")
    minimum = int(contract.get("minimum_test_count", -1))
    tests_run = tests.get("tests_run")
    _require(
        isinstance(tests_run, int)
        and not isinstance(tests_run, bool)
        and tests_run >= minimum
        and int(tests.get("minimum_tests", -1)) == minimum,
        "numeric test count differs",
    )
    _require(
        int(tests.get("failure_count", -1)) == 0
        and int(tests.get("error_count", -1)) == 0
        and int(tests.get("skip_count", -1)) == 0
        and tests.get("failures") == []
        and tests.get("errors") == []
        and tests.get("skipped") == []
        and isinstance(tests.get("output"), str)
        and tests.get("output"),
        "numeric suite was incomplete, skipped, or unsuccessful",
    )
    acceptance = value.get("acceptance")
    _require(isinstance(acceptance, Mapping), "numeric acceptance record is absent")
    for field in (
        "unittest_success",
        "zero_skips",
        "minimum_test_count_met",
        "clean_source",
        "inside_slurm_allocation",
        "allocated_gpu_visible",
        "allocated_devices_are_h100",
        "production_grid_validation_enabled",
        "production_grid_test_module_included",
    ):
        _require(acceptance.get(field) is True, "numeric acceptance.%s differs" % field)
    _require(
        runtime.get("production_grid_validation_enabled") is True,
        "numeric production-grid runtime flag differs",
    )
    duration = value.get("duration_seconds")
    _require(
        isinstance(duration, (int, float))
        and not isinstance(duration, bool)
        and math.isfinite(float(duration))
        and float(duration) >= 0.0,
        "numeric duration is invalid",
    )
    return {
        "schema_version": NUMERIC_SCHEMA,
        "status": "passed",
        "path": str(candidate),
        "file_sha256": _file_sha256(candidate),
        "payload_sha256": payload_sha,
        "slurm_job_id": str(expected_job_id),
        "source_commit": expected_commit,
        "python_executable": expected_python_executable,
        "test_modules": observed_modules,
        "test_count": int(tests_run),
        "skip_count": 0,
        "gpu": dict(devices[0]),
        "scientific_result": False,
    }


__all__ = [
    "NUMERIC_SCHEMA",
    "OscNumericPrerequisiteError",
    "validate_numeric_prerequisite_artifact",
]
