from __future__ import annotations

import json
import unittest
from pathlib import Path

from main.multilink_ellipsoid.whole_body_support_audit import (
    audit_cases, load_audit_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/vlsa_distal_whole_body_support_audit.v1.json"


def _candidate(name, row_slack, known=True, car=0.0, veto=False):
    groups = {
        "end_effector": min(row_slack[0:1]),
        "palm": min(row_slack[1:2]),
        "L5": min(row_slack[2:5]),
        "L6": min(row_slack[5:7]),
        "L7": min(row_slack[7:9]),
    }
    return {
        "name": name,
        "raw_protected_contact_sample_count": 0,
        "replayed_maximum_CAR_m": car,
        "replayed_physical_veto": veto,
        "exact_group_target": {
            "known_outcome": known,
            "group_future_violation": {
                group: -slack for group, slack in groups.items()
            },
            "group_contact_sample_count": {group: 0 for group in groups},
            "trace": [{"row_normalized_radial_slack": row_slack}],
        },
    }


def _case():
    groups = ["end_effector", "palm", "L5", "L6", "L7"]
    return {
        "case_id": "case-0",
        "selection": {
            "episode_group_id": "episode-0", "split": "train",
            "target_group": "L5",
        },
        "exact_case": {
            "exact_group_target": {
                "initial_group_normalized_radial_slack": {
                    group: 0.2 for group in groups
                },
                "initial_group_contact_sample_count": {group: 0 for group in groups},
            },
            "candidates": [
                _candidate("safe", [0.2] * 9),
                _candidate("unsafe-L5-row1", [0.2, 0.2, 0.2, -0.1, 0.2,
                                                0.2, 0.2, 0.2, 0.2]),
                _candidate("unknown", [0.2] * 9, known=False),
            ],
        },
    }


class WholeBodySupportAuditTest(unittest.TestCase):
    def test_config_freezes_physical_and_diagnostic_groups(self):
        value = load_audit_config(CONFIG)
        self.assertEqual(value["physical_group_order"], ["palm", "L5", "L6", "L7"])
        self.assertEqual(value["robot_rows"]["L5"], [2, 3, 4])

    def test_audit_preserves_unknown_and_reports_per_row_and_global_support(self):
        result = audit_cases([_case()], load_audit_config(CONFIG))
        case = result["per_case"][0]
        self.assertEqual(case["unknown_timeout_count"], 1)
        self.assertTrue(case["per_group"]["L5"]["two_sided_support"])
        self.assertTrue(case["per_row"]["3"]["two_sided_support"])
        self.assertEqual(case["per_row"]["3"]["active_witness_candidate_count"], 1)
        self.assertTrue(case["global_support"]["represented_two_sided_support"])
        self.assertTrue(case["global_support"]["physical_two_sided_support"])
        split = result["split_summary"]["train"]
        self.assertEqual(split["per_group"]["L5"]["two_sided_state_count"], 1)
        self.assertEqual(split["per_row"]["3"]["two_sided_state_count"], 1)
        self.assertEqual(split["physical_global_safe_support_state_count"], 1)

    def test_physical_global_support_is_separate_from_ee_proxy(self):
        case = _case()
        case["exact_case"]["candidates"][0] = _candidate(
            "ee-only-unsafe", [-0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
        )
        result = audit_cases([case], load_audit_config(CONFIG))["per_case"][0]
        self.assertEqual(result["global_support"]["represented_safe_candidate_count"], 0)
        self.assertGreater(result["global_support"]["physical_safe_candidate_count"], 0)


if __name__ == "__main__":
    unittest.main()
