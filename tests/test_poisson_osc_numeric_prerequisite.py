import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from main.poisson_fullbody.osc_numeric_prerequisite import (
    OscNumericPrerequisiteError,
    validate_numeric_prerequisite_artifact,
)


MODULES = [
    "tests.test_poisson_geometry",
    "tests.test_poisson_field",
    "tests.test_poisson_field_bundle",
    "tests.test_poisson_production_grid",
    "tests.test_poisson_surface_sampling",
    "tests.test_poisson_jacobians",
    "tests.test_poisson_cbf_qp",
    "tests.test_poisson_prephysics_control_wrapper",
    "tests.test_poisson_post_osc_torque_sensitivity",
    "tests.test_poisson_post_osc_torque_shield",
    "tests.test_poisson_osc_arm_link_canary",
    "tests.test_poisson_osc_numeric_prerequisite",
    "tests.test_poisson_osc_arm_link_slurm_contract",
    "tests.test_poisson_numeric_validation_runner",
]
COMMIT = "1" * 40


def canonical(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def artifact():
    value = {
        "schema_version": "vlsa_poisson_numeric_validation.v1",
        "status": "passed",
        "scientific_result": False,
        "evidence_tier": "allocation_backed_implementation_validation",
        "claim_limit": "synthetic and numerical tests do not establish SafeLIBERO safety efficacy",
        "created_at_unix": 1,
        "duration_seconds": 2.5,
        "source": {"commit": COMMIT, "branch": "", "status_short": []},
        "runtime": {
            "python_executable": "/evaluation/python",
            "python_version": "3.11",
            "host": "worker-mig-3g40gb-0",
            "slurm_job_id": "123",
            "slurm_job_name": "numeric",
            "slurm_partition": "mig",
            "cuda_visible_devices": "0",
            "production_grid_validation_enabled": True,
            "allocated_gpu_inventory": {
                "available": True,
                "devices": [
                    {"name": "NVIDIA H100 80GB HBM3", "uuid": "GPU-a", "driver_version": "1"}
                ],
                "error": None,
            },
            "packages": {
                "numpy": "1",
                "scipy": "1",
                "mujoco": "1",
                "cvxpy": None,
                "osqp": "1",
            },
        },
        "tests": {
            "modules": MODULES,
            "minimum_tests": 120,
            "tests_run": 140,
            "failure_count": 0,
            "error_count": 0,
            "skip_count": 0,
            "failures": [],
            "errors": [],
            "skipped": [],
            "output": "OK",
        },
        "acceptance": {
            "unittest_success": True,
            "zero_skips": True,
            "minimum_test_count_met": True,
            "clean_source": True,
            "inside_slurm_allocation": True,
            "allocated_gpu_visible": True,
            "allocated_devices_are_h100": True,
            "production_grid_validation_enabled": True,
            "production_grid_test_module_included": True,
        },
    }
    value["result_payload_sha256"] = hashlib.sha256(canonical(value)).hexdigest()
    return value


class NumericPrerequisiteTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.path = self.root / "numeric.json"
        self.contract = {
            "evidence_tier": "allocation_backed_implementation_validation",
            "required_test_modules": MODULES,
            "minimum_test_count": 120,
        }

    def tearDown(self):
        self.directory.cleanup()

    def write(self, value):
        self.path.write_text(json.dumps(value), encoding="utf-8")

    def validate(self):
        return validate_numeric_prerequisite_artifact(
            self.path,
            contract=self.contract,
            expected_job_id="123",
            expected_commit=COMMIT,
            expected_python_executable="/evaluation/python",
            expected_parent=self.root,
            forbidden_job_id="456",
        )

    def test_complete_zero_skip_h100_artifact_passes(self):
        self.write(artifact())
        record = self.validate()
        self.assertEqual(record["test_count"], 140)
        self.assertEqual(record["skip_count"], 0)

    def test_skip_or_payload_mutation_is_rejected(self):
        for mutate in (
            lambda value: value["tests"].update(skip_count=1),
            lambda value: value["runtime"].update(slurm_job_id="999"),
            lambda value: value["runtime"]["allocated_gpu_inventory"]["devices"][0].update(name="A100"),
        ):
            value = artifact()
            mutate(value)
            self.write(value)
            with self.assertRaises(OscNumericPrerequisiteError):
                self.validate()

    def test_symlink_artifact_is_rejected(self):
        target = self.root / "target.json"
        target.write_text(json.dumps(artifact()), encoding="utf-8")
        self.path.symlink_to(target)
        with self.assertRaises(OscNumericPrerequisiteError):
            self.validate()


if __name__ == "__main__":
    unittest.main()
