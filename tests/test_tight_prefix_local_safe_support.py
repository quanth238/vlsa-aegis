"""Tests for immutable exact-safe support auditing."""

from __future__ import annotations

import unittest
from pathlib import Path

from main.multilink_ellipsoid.tight_prefix_local_safe_support import (
    load_config, nearest_safe_support, summarize,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/vlsa_tight_prefix_local_safe_support.v1.json"


def actions(x: float, rotation: float = 0.0) -> list[list[float]]:
    return [[x, 0.0, 0.0, rotation, 0.0, 0.0, 0.0] for _ in range(5)]


class TightPrefixLocalSafeSupportTests(unittest.TestCase):
    def test_config_freezes_validation_decision_and_no_retraining(self) -> None:
        value = load_config(CONFIG)
        self.assertEqual(value["audit"]["decision_split"], "validation")
        self.assertEqual(value["audit"]["primary_rows"], list(range(8)))
        self.assertEqual(len(value["prior_probe_anchors"]), 6)
        self.assertIn("training_or_architecture_change", value["forbidden"])

    def test_nearest_safe_support_obeys_ball(self) -> None:
        try:
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("NumPy is not installed in the local system Python")
        result = nearest_safe_support(
            anchor_name="unsafe", anchor_actions=actions(0.0),
            safe_candidates=[
                {"name": "far", "actions": actions(0.2)},
                {"name": "near", "actions": actions(0.1)},
            ],
            radius=0.25, tolerance=1e-12,
            non_translation_tolerance=1e-12,
        )
        self.assertEqual(result["nearest_safe_candidate"]["candidate_name"], "near")
        self.assertTrue(result["safe_candidate_within_radius"])

    def test_non_translation_changes_are_not_reachable(self) -> None:
        try:
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("NumPy is not installed in the local system Python")
        result = nearest_safe_support(
            anchor_name="unsafe", anchor_actions=actions(0.0),
            safe_candidates=[{"name": "rotation", "actions": actions(0.0, 0.1)}],
            radius=0.25, tolerance=1e-12,
            non_translation_tolerance=1e-12,
        )
        self.assertIsNone(result["nearest_safe_candidate"])
        self.assertEqual(result["inaccessible_safe_candidate_count"], 1)
        self.assertFalse(result["safe_candidate_within_radius"])

    def test_summary_counts_supported_roots(self) -> None:
        records = [
            {
                "case_id": "a", "safe_candidate_within_radius": True,
                "nearest_safe_candidate": {"translation_distance_l2_action": 0.1},
            },
            {
                "case_id": "a", "safe_candidate_within_radius": True,
                "nearest_safe_candidate": {"translation_distance_l2_action": 0.2},
            },
            {
                "case_id": "b", "safe_candidate_within_radius": False,
                "nearest_safe_candidate": None,
            },
        ]
        result = summarize(records)
        self.assertEqual(result["local_safe_supported_anchor_count"], 2)
        self.assertEqual(result["local_safe_supported_root_count"], 1)
        self.assertEqual(result["anchor_with_any_known_safe_count"], 2)


if __name__ == "__main__":
    unittest.main()
