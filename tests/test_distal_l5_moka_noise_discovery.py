from __future__ import annotations

import json
from pathlib import Path
import unittest

from main.multilink_ellipsoid.moka_noise_discovery import (
    canonical, load_config, load_manifest, sha256_bytes, validate_summary,
)


ROOT = Path(__file__).resolve().parents[1]


class MokaNoiseDiscoveryTest(unittest.TestCase):
    def test_preregistered_manifest(self) -> None:
        config = load_config(ROOT / "configs/vlsa_distal_l5_moka_noise_discovery.v1.json")
        rows = load_manifest(ROOT / "manifests/vlsa_distal_l5_moka_noise_discovery.v1.jsonl", config)
        self.assertEqual([row["episode_index"] for row in rows], list(range(0, 50, 5)))
        self.assertTrue(config["execution"]["original_aegis_ee_correction_enabled"])
        self.assertIn("MLP_training", config["forbidden"])

    def test_summary_digest_gate(self) -> None:
        summary = {
            "schema_version": "vlsa_distal_l5_moka_noise_discovery_summary.v1",
            "case_count": 1,
            "eligible_case_count": 0,
            "records": [{"eligible": False}],
        }
        summary["payload_sha256"] = sha256_bytes(canonical(summary))
        validate_summary(summary)
        broken = json.loads(json.dumps(summary))
        broken["case_count"] = 2
        with self.assertRaises(ValueError):
            validate_summary(broken)


if __name__ == "__main__":
    unittest.main()
