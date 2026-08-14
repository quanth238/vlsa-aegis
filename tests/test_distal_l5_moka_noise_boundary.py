import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from main.multilink_ellipsoid.moka_noise_boundary import (
    load_cases, load_config, validate_discovery_binding,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "configs/vlsa_distal_l5_moka_noise_boundary.v1.json"
MANIFEST = REPO_ROOT / "manifests/vlsa_distal_l5_moka_noise_boundary.v1.jsonl"


class MokaNoiseBoundaryTest(unittest.TestCase):
    def test_checked_in_contract(self):
        config = load_config(CONFIG, repo_root=REPO_ROOT)
        cases = load_cases(MANIFEST, config)
        self.assertEqual(len(cases), 4)
        self.assertEqual(
            [case["split"] for case in cases],
            ["train", "diagnostic", "diagnostic", "validation"],
        )
        self.assertEqual(len({case["trajectory_group_id"] for case in cases}), 4)

    def test_discovery_binding_rejects_changed_summary(self):
        config = load_config(CONFIG, repo_root=REPO_ROOT)
        cases = load_cases(MANIFEST, config)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "summary.json").write_text("{}")
            (root / "validation.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "summary file differs"):
                validate_discovery_binding(
                    config=config, cases=cases, discovery_root=root
                )

    def test_manifest_source_hashes_are_sha256(self):
        config = load_config(CONFIG, repo_root=REPO_ROOT)
        for case in load_cases(MANIFEST, config):
            for key in (
                "archived_result_file_sha256", "archived_result_payload_sha256"
            ):
                self.assertEqual(len(case[key]), 64)
                int(case[key], 16)


if __name__ == "__main__":
    unittest.main()
