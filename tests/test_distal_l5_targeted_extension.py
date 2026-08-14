from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from main.multilink_ellipsoid.targeted_l5_boundary_extension import (
    load_cases, load_config,
)


ROOT = Path(__file__).resolve().parents[1]


class TargetedL5BoundaryExtensionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config_path = (
            ROOT / "configs/vlsa_distal_l5_targeted_extension.v1.json"
        )
        self.manifest_path = (
            ROOT / "manifests/vlsa_distal_l5_targeted_extension.v1.jsonl"
        )
        self.config = load_config(self.config_path, repo_root=ROOT)
        self.cases = load_cases(self.manifest_path, self.config)

    def test_cases_are_preregistered_grouped_and_exclude_sealed_tests(self) -> None:
        self.assertEqual(len(self.cases), 5)
        self.assertEqual(
            [case["split"] for case in self.cases],
            ["train", "train", "train", "validation", "validation"],
        )
        self.assertEqual(len({case["case_id"] for case in self.cases}), 5)
        self.assertFalse(
            {case["case_id"] for case in self.cases}
            & set(self.config["sealed_test_cases"])
        )
        self.assertTrue(all(
            case["protected_contact_body"] == "robot0_link5"
            for case in self.cases
        ))

    def test_parent_method_is_frozen(self) -> None:
        parent = self.config["parent_method"]
        self.assertTrue(parent["candidate_bank_unchanged"])
        self.assertTrue(parent["candidate_plus_fixed_backup_target_unchanged"])
        self.assertTrue(parent["original_AEGIS_EE_filter_unchanged"])
        self.assertTrue(parent["timeout_semantics_unchanged"])

    def test_manifest_mutation_fails_closed(self) -> None:
        rows = self.manifest_path.read_text().splitlines()
        value = json.loads(rows[0])
        value["split"] = "validation"
        rows[0] = json.dumps(value, sort_keys=True, separators=(",", ":"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "changed.jsonl"
            path.write_text("\n".join(rows) + "\n")
            with self.assertRaisesRegex(ValueError, "manifest hash"):
                load_cases(path, self.config)


if __name__ == "__main__":
    unittest.main()
