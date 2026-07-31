#!/usr/bin/env python3
"""Run allocation-backed Poisson numerical tests and atomically bind evidence."""

import argparse
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import hashlib
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import tempfile
import time
import unittest


SCHEMA_VERSION = "vlsa_poisson_numeric_validation.v1"
PRODUCTION_GRID_TEST_MODULE = "tests.test_poisson_production_grid"
DEFAULT_TEST_MODULES = (
    "tests.test_poisson_geometry",
    "tests.test_poisson_field",
    "tests.test_poisson_field_bundle",
    PRODUCTION_GRID_TEST_MODULE,
    "tests.test_poisson_surface_sampling",
    "tests.test_poisson_jacobians",
    "tests.test_poisson_cbf_qp",
    "tests.test_poisson_adapter",
    "tests.test_poisson_controller_bridge",
    "tests.test_poisson_measurement",
)


def canonical_json_bytes(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def git_record(root):
    def run(*arguments):
        return subprocess.check_output(
            ["git", "-C", str(root)] + list(arguments),
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        ).strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "status_short": run("status", "--short").splitlines(),
    }


def package_versions(names):
    try:
        from importlib import metadata
    except ImportError:  # Python 3.8 fallback package
        import importlib_metadata as metadata
    output = {}
    for name in names:
        try:
            output[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            output[name] = None
    return output


def allocated_gpu_inventory():
    """Return the devices exposed inside the Slurm allocation.

    This command is intentionally executed by the allocation entry point, not
    on the login node.  A missing or empty inventory prevents this artifact
    from being labelled allocation-backed H100 evidence.
    """

    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,memory.total",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return {"available": False, "devices": [], "error": str(error)}
    devices = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3 or not all(parts):
            return {
                "available": False,
                "devices": [],
                "error": "unexpected nvidia-smi row: %r" % line,
            }
        try:
            memory_mib = int(parts[2])
        except ValueError:
            return {
                "available": False,
                "devices": [],
                "error": "noninteger GPU memory: %r" % parts[2],
            }
        devices.append(
            {"name": parts[0], "uuid": parts[1], "memory_total_mib": memory_mib}
        )
    return {
        "available": bool(devices),
        "devices": devices,
        "error": None if devices else "no allocated GPU was visible",
    }


def atomic_json(path, payload):
    if path.exists() or path.is_symlink():
        raise RuntimeError("refusing to overwrite immutable result: %s" % path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=str(path.parent),
            prefix=".%s." % path.name,
            suffix=".partial",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        # Hard-link publication has create-if-absent semantics.  ``replace``
        # would overwrite a final created after the preflight exists check.
        try:
            os.link(str(temporary), str(path))
        except FileExistsError as error:
            raise RuntimeError(
                "result was concurrently created: %s" % path
            ) from error
        directory_fd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--test-module", action="append", dest="test_modules")
    parser.add_argument("--minimum-tests", type=int, default=30)
    arguments = parser.parse_args()

    root = arguments.repo_root.resolve()
    os.chdir(str(root))
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    started = time.time()
    source = git_record(root)
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    gpu_inventory = allocated_gpu_inventory()
    h100_only = bool(
        gpu_inventory["available"]
        and gpu_inventory["devices"]
        and all("H100" in device["name"] for device in gpu_inventory["devices"])
    )
    modules = tuple(arguments.test_modules or DEFAULT_TEST_MODULES)
    production_grid_enabled = (
        os.environ.get("VLSA_POISSON_RUN_PRODUCTION_GRID_VALIDATION") == "1"
    )
    production_grid_included = PRODUCTION_GRID_TEST_MODULE in modules
    capture = StringIO()
    suite = unittest.defaultTestLoader.loadTestsFromNames(modules)
    # Preserve solver metrics printed by allocation-only validation modules in
    # the immutable evidence artifact, not only in the ephemeral Slurm log.
    with redirect_stdout(capture), redirect_stderr(capture):
        result = unittest.TextTestRunner(stream=capture, verbosity=2).run(suite)
    test_count = int(result.testsRun)
    failures = [str(test) for test, _ in result.failures]
    errors = [str(test) for test, _ in result.errors]
    skipped = [
        {"test": str(test), "reason": str(reason)}
        for test, reason in result.skipped
    ]
    passed = bool(
        result.wasSuccessful()
        and not skipped
        and test_count >= int(arguments.minimum_tests)
        and not source["status_short"]
        and bool(slurm_job_id)
        and gpu_inventory["available"]
        and h100_only
        and production_grid_enabled
        and production_grid_included
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if passed else "failed",
        "scientific_result": False,
        "evidence_tier": "allocation_backed_implementation_validation",
        "claim_limit": "synthetic and numerical tests do not establish SafeLIBERO safety efficacy",
        "created_at_unix": int(time.time()),
        "duration_seconds": float(time.time() - started),
        "source": source,
        "runtime": {
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "host": socket.gethostname(),
            "slurm_job_id": slurm_job_id,
            "slurm_job_name": os.environ.get("SLURM_JOB_NAME"),
            "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "production_grid_validation_enabled": production_grid_enabled,
            "allocated_gpu_inventory": gpu_inventory,
            "packages": package_versions(
                ("numpy", "scipy", "mujoco", "cvxpy", "osqp")
            ),
        },
        "tests": {
            "modules": list(modules),
            "minimum_tests": int(arguments.minimum_tests),
            "tests_run": test_count,
            "failure_count": len(failures),
            "error_count": len(errors),
            "skip_count": len(skipped),
            "failures": failures,
            "errors": errors,
            "skipped": skipped,
            "output": capture.getvalue(),
        },
        "acceptance": {
            "unittest_success": bool(result.wasSuccessful()),
            "zero_skips": not skipped,
            "minimum_test_count_met": test_count >= int(arguments.minimum_tests),
            "clean_source": not source["status_short"],
            "inside_slurm_allocation": bool(slurm_job_id),
            "allocated_gpu_visible": gpu_inventory["available"],
            "allocated_devices_are_h100": h100_only,
            "production_grid_validation_enabled": production_grid_enabled,
            "production_grid_test_module_included": production_grid_included,
        },
    }
    payload["result_payload_sha256"] = sha256_bytes(canonical_json_bytes(payload))
    atomic_json(arguments.output, payload)
    print(json.dumps({"status": payload["status"], "output": str(arguments.output), "tests_run": test_count, "skip_count": len(skipped)}, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
