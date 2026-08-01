from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "run_poisson_numeric_validation.py"
SPEC = importlib.util.spec_from_file_location(
    "poisson_numeric_validation_under_test", MODULE_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class PoissonNumericGpuInventoryTest(unittest.TestCase):
    def test_default_modules_execute_shadow_runner_and_observer_regressions(self):
        self.assertIn(
            "tests.test_poisson_shadow_identification", MODULE.DEFAULT_TEST_MODULES
        )
        self.assertIn(
            "tests.test_poisson_shadow_identification_runner",
            MODULE.DEFAULT_TEST_MODULES,
        )
        self.assertIn(
            "tests.test_poisson_shadow_identification_artifact_validator",
            MODULE.DEFAULT_TEST_MODULES,
        )

    def inventory_for(self, output: str):
        with mock.patch.object(
            MODULE.subprocess, "check_output", return_value=output
        ):
            return MODULE.allocated_gpu_inventory()

    def test_uses_identity_fields_that_remain_available_in_mig_allocation(self):
        inventory = self.inventory_for(
            "NVIDIA H100 80GB HBM3, GPU-012345, 555.42.06\n"
        )

        self.assertTrue(inventory["available"])
        self.assertIsNone(inventory["error"])
        self.assertEqual(len(inventory["devices"]), 1)
        device = inventory["devices"][0]
        self.assertEqual(device["uuid"], "GPU-012345")
        self.assertEqual(device["driver_version"], "555.42.06")
        self.assertTrue(MODULE.inventory_is_single_h100(inventory))

    def test_unavailable_required_identity_response_fails_closed(self):
        inventory = self.inventory_for(
            "NVIDIA H100 80GB HBM3, GPU-012345, [Insufficient Permissions]\n"
        )

        self.assertFalse(inventory["available"])
        self.assertEqual(inventory["devices"], [])
        self.assertIn("unavailable required GPU identity field", inventory["error"])
        self.assertFalse(MODULE.inventory_is_single_h100(inventory))

    def test_duplicate_uuid_fails_closed(self):
        inventory = self.inventory_for(
            "NVIDIA H100 80GB HBM3, GPU-duplicate, 555.42.06\n"
            "NVIDIA H100 80GB HBM3, GPU-duplicate, 555.42.06\n"
        )

        self.assertFalse(inventory["available"])
        self.assertIn("duplicate GPU UUID", inventory["error"])
        self.assertFalse(MODULE.inventory_is_single_h100(inventory))

    def test_non_h100_or_multiple_devices_do_not_satisfy_allocation_gate(self):
        non_h100 = self.inventory_for("NVIDIA A100, GPU-a100, 555.42.06\n")
        multiple = self.inventory_for(
            "NVIDIA H100 80GB HBM3, GPU-h100-a, 555.42.06\n"
            "NVIDIA H100 80GB HBM3, GPU-h100-b, 555.42.06\n"
        )

        self.assertFalse(MODULE.inventory_is_single_h100(non_h100))
        self.assertFalse(MODULE.inventory_is_single_h100(multiple))


if __name__ == "__main__":
    unittest.main()
