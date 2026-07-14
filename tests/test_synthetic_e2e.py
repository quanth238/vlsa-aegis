from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from crfs_harness.artifacts import load_json, valid_completion
from crfs_harness.manifest import build_cases
from crfs_harness.synthetic import run_synthetic_case


class SyntheticEndToEndTest(unittest.TestCase):
    def test_case_is_atomic_restartable_and_labeled_synthetic(self) -> None:
        case = build_cases("safelibero_spatial", "II", 0, [0], 1, "e2e")[0]
        with tempfile.TemporaryDirectory() as directory:
            path, status = run_synthetic_case(case, directory, "test-run")
            self.assertEqual(status, "completed")
            self.assertTrue(valid_completion(path))
            value = load_json(path)
            self.assertEqual(value["provenance"]["evidence_tier"], "synthetic")
            self.assertTrue(value["trials"]["oracle_residual"]["safe"])
            self.assertFalse(value["trials"]["nominal"]["safe"])
            repeated_path, repeated_status = run_synthetic_case(case, directory, "test-run")
            self.assertEqual(repeated_path, path)
            self.assertEqual(repeated_status, "skipped_valid_completion")


if __name__ == "__main__":
    unittest.main()
