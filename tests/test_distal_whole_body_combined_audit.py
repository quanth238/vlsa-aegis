from __future__ import annotations

import unittest
from pathlib import Path

from main.multilink_ellipsoid.whole_body_combined_audit import (
    evaluate_gate, load_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/vlsa_distal_whole_body_combined_audit.v1.json"


def _coverage(count):
    return {"two_sided_state_count": count}


def _summary(*, unsafe=False, palm=(4, 2, 2), l5=(4, 2, 2), l6=(4, 2, 2)):
    groups = ["end_effector", "palm", "L5", "L6", "L7"]
    counts = {"palm": palm, "L5": l5, "L6": l6}
    split_counts = {"train": 16, "validation": 4, "test": 4}
    split_summary = {}
    for index, split in enumerate(("train", "validation", "test")):
        split_summary[split] = {
            "case_count": split_counts[split],
            "per_group": {
                group: _coverage(counts.get(group, (0, 0, 0))[index])
                for group in groups
            },
        }
    per_case = []
    for split, count in split_counts.items():
        for index in range(count):
            per_case.append({
                "case_id": f"{split}-{index}",
                "split": split,
                "initially_safe_across_physical_groups": not (
                    unsafe and split == "train" and index == 0
                ),
                "global_support": {"physical_safe_candidate_count": 1},
            })
    return {"split_summary": split_summary, "per_case": per_case}


class WholeBodyCombinedAuditTest(unittest.TestCase):
    def test_clean_supported_cohort_authorizes_q_only_training(self):
        config = load_config(CONFIG)
        gate = evaluate_gate(
            _summary(), config, source_replay_exact=True,
            source_state_hash_exact=True, physical_false_safe_count=0,
            context_complete=True, maximum_bellman_residual=0.0,
        )
        self.assertTrue(gate["training_authorized"])
        self.assertEqual(gate["authorized_prediction_groups"], [
            "palm", "L5", "L6",
        ])

    def test_initially_unsafe_state_blocks_training_without_dropping_it(self):
        config = load_config(CONFIG)
        gate = evaluate_gate(
            _summary(unsafe=True), config, source_replay_exact=True,
            source_state_hash_exact=True, physical_false_safe_count=0,
            context_complete=True, maximum_bellman_residual=0.0,
        )
        self.assertFalse(gate["training_authorized"])
        self.assertEqual(gate["initially_unsafe_case_ids"], ["train-0"])

    def test_missing_constraint_support_blocks_whole_body_claim(self):
        config = load_config(CONFIG)
        gate = evaluate_gate(
            _summary(palm=(3, 2, 2)), config, source_replay_exact=True,
            source_state_hash_exact=True, physical_false_safe_count=0,
            context_complete=True, maximum_bellman_residual=0.0,
        )
        self.assertFalse(gate["training_authorized"])
        self.assertFalse(gate["group_gates"]["palm"]["passes"])


if __name__ == "__main__":
    unittest.main()
