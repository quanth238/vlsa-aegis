import json
from pathlib import Path
import unittest

from scripts.evaluate_distal_soft_prefix_live_replan_e05 import load_config


class SoftPrefixLiveReplanTest(unittest.TestCase):
    def test_protocol_has_one_distal_intervention_and_no_endpoint_constraint(self):
        path = Path(__file__).resolve().parents[1] / "configs" / "vlsa_distal_soft_prefix_live_replan_e05.v1.json"
        raw = json.loads(path.read_text())
        value = load_config(path)
        self.assertFalse(value["control"]["additional_l5_l7_intervention_after_prefix"])
        self.assertIn("endpoint_preservation", value["forbidden"])
        self.assertEqual(raw["state_protocol"]["first_live_policy_query_index"], 37)
        self.assertEqual(raw["state_protocol"]["repulsive_prefix_steps"], list(range(182, 187)))


if __name__ == "__main__":
    unittest.main()
