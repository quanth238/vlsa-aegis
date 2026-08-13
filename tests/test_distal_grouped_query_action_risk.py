from pathlib import Path
import unittest

from main.multilink_ellipsoid.grouped_query_action_risk import load_config
from main.multilink_ellipsoid.query_action_risk import load_config as load_base


ROOT = Path(__file__).resolve().parents[1]


class GroupedQueryActionRiskTests(unittest.TestCase):
    def test_grouped_protocol_reuses_frozen_method(self):
        config = load_config(
            ROOT / "configs/vlsa_distal_grouped_query_action_risk.v1.json"
        )
        base = load_base(ROOT / config["method"]["base_config"])
        self.assertEqual(
            base["config_payload_sha256"],
            config["method"]["base_config_payload_sha256"],
        )
        self.assertEqual(config["method"]["candidate_count_per_state"], 37)
        self.assertTrue(config["forbidden"]["model_training"])


if __name__ == "__main__":
    unittest.main()
