from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from crfs_harness.artifacts import (
    atomic_write_json,
    content_hash,
    scientific_config,
    valid_completion,
    validate_case_result,
)
from crfs_harness.manifest import build_cases, validate_case


class ManifestArtifactTest(unittest.TestCase):
    def test_scientific_config_ignores_only_runtime_transport_fields(self) -> None:
        first = {
            "host": "127.0.0.1",
            "port": 8130,
            "output_root": "/first",
            "run_id": "run-a",
            "safety_margin_m": 0.005,
            "checkpoint_sha256": "a" * 64,
        }
        second = {
            **first,
            "host": "localhost",
            "port": 8131,
            "output_root": "/second",
            "run_id": "run-b",
        }
        self.assertEqual(
            content_hash(scientific_config(first)),
            content_hash(scientific_config(second)),
        )
        second["safety_margin_m"] = 0.006
        self.assertNotEqual(
            content_hash(scientific_config(first)),
            content_hash(scientific_config(second)),
        )

    def test_case_schedule_is_deterministic_and_unique(self) -> None:
        first = build_cases("safelibero_spatial", "II", 0, [0, 1], 3, "test")
        second = build_cases("safelibero_spatial", "II", 0, [0, 1], 3, "test")
        self.assertEqual(first, second)
        self.assertEqual(len({case["case_id"] for case in first}), len(first))
        self.assertFalse([error for case in first for error in validate_case(case)])

    def test_partial_result_is_not_completion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.json"
            atomic_write_json(path, {"status": "completed"})
            self.assertFalse(valid_completion(path))

    def test_completed_result_requires_every_paired_arm(self) -> None:
        result = {
            "schema_version": "1.0",
            "case_id": "case",
            "run_id": "run",
            "status": "completed",
            "config_hash": "a" * 64,
            "provenance": {},
            "repair": {"feasible": True},
            "trials": {},
        }
        errors = validate_case_result(result)
        self.assertTrue(any("paired trials" in error for error in errors))

    def test_failed_result_is_not_a_resumable_completion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.json"
            atomic_write_json(
                path,
                {
                    "schema_version": "1.0",
                    "case_id": "case",
                    "run_id": "run",
                    "status": "failed",
                    "config_hash": "a" * 64,
                    "provenance": {},
                    "repair": {"feasible": False},
                    "trials": {},
                },
            )
            self.assertFalse(valid_completion(path))


if __name__ == "__main__":
    unittest.main()
