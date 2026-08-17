from __future__ import annotations

import unittest
from pathlib import Path

from main.multilink_ellipsoid.tight_prefix_selector_audit import (
    RULES, load_config,
)
from scripts.audit_tight_prefix_selector import _evaluate, _passes


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/vlsa_tight_prefix_selector_audit.v1.json"


def _record(state, name, order, predicted, actual, correction=0.0, l6=-1.0):
    return {
        "state_id": state, "candidate_name": name,
        "candidate_order": order, "predicted_primary": predicted,
        "actual_primary": actual, "actual_L6": l6,
        "correction": correction, "failed_groups": [],
    }


class TightPrefixSelectorAuditTest(unittest.TestCase):
    def test_rules_and_gate_are_frozen(self):
        config = load_config(CONFIG)
        self.assertEqual(tuple(config["selection"]["rules"]), RULES)
        self.assertEqual(config["selection"]["rule_freeze_split"],
                         "validation")
        self.assertIn("test_based_margin_or_rule_selection", config["forbidden"])

    def test_margin_rule_abstains_when_optimism_is_not_covered(self):
        records = [
            _record("s", "nominal", 0, -0.1, 0.2),
            _record("s", "safe", 1, -0.4, -0.1, correction=1.0),
        ]
        result = _evaluate(
            records, rule="validation_margin_least_intervention", margin=0.3,
        )
        self.assertEqual(result["safe_selection_count"], 1)
        self.assertEqual(result["states"][0]["selected_candidate"], "safe")

    def test_validation_gate_requires_all_four_safe_without_abstention(self):
        config = load_config(CONFIG)
        summary = {
            "safe_selection_count": 4, "unsafe_selection_count": 0,
            "abstention_count": 0, "selected_L6_failure_count": 0,
        }
        self.assertTrue(_passes(summary, config=config, split="validation"))
        summary["abstention_count"] = 1
        self.assertFalse(_passes(summary, config=config, split="validation"))


if __name__ == "__main__":
    unittest.main()
