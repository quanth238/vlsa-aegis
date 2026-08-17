import unittest
from pathlib import Path

from main.multilink_ellipsoid.top5_exact_verified_pilot import (
    candidate_summary, load_config, ranked_prefix_summary, scientific_view,
)


class Top5ExactVerifiedPilotTest(unittest.TestCase):
    def test_config_freezes_three_false_safe_cases(self):
        config = load_config(Path(
            "configs/vlsa_distal_top5_exact_verified_pilot.v1.json"
        ))
        self.assertEqual(len(config["cases"]), 3)
        self.assertEqual(config["maximum_ranked_candidates"], 5)

    def test_candidate_requires_every_physical_group_safe(self):
        candidate = {
            "name": "candidate",
            "exact_group_target": {
                "known_outcome": True,
                "safe_terminal": True,
                "group_future_violation": {
                    "end_effector": 1.0,
                    "palm": -0.1, "L5": -0.2, "L6": -0.3, "L7": -0.4,
                },
                "group_contact_sample_count": {
                    "end_effector": 0,
                    "palm": 0, "L5": 0, "L6": 0, "L7": 0,
                },
            },
            "source_raw_protected_contact_count": 0,
            "raw_protected_contact_sample_count": 0,
            "source_physical_veto": False,
            "replayed_physical_veto": False,
            "source_maximum_CAR_m": 0.0,
            "replayed_maximum_CAR_m": 0.0,
            "source_effective_post_AEGIS_correction_l2_action": 1.0,
            "source_executed_actions": [[0.0] * 7 for _ in range(5)],
        }
        summary = candidate_summary(candidate, ["palm", "L5", "L6", "L7"])
        self.assertTrue(summary["physical_safe"])
        candidate["exact_group_target"]["group_future_violation"]["L6"] = 0.01
        summary = candidate_summary(candidate, ["palm", "L5", "L6", "L7"])
        self.assertFalse(summary["physical_safe"])

    def test_scientific_view_ignores_allocation_specific_attempt_hashes(self):
        result = {
            "case_id": "case", "split": "test",
            "attempts": [{
                "candidate_name": "a", "physical_safe": True,
                "fresh_result_file_sha256": "file",
                "fresh_result_payload_sha256": "payload",
            }],
            "selected_rank": 1, "selected_candidate": "a",
            "selected_action_chunk": [[0.0] * 7 for _ in range(5)],
            "top_one_freshly_unsafe": False,
            "verified_safe_selection": True,
            "method_avoids_top_one_collision": False,
        }
        view = scientific_view(result)
        self.assertNotIn("fresh_result_file_sha256", view["attempts"][0])

    def test_one_fresh_bank_stops_at_first_safe_rank(self):
        def row(name, l5):
            return {
                "name": name,
                "exact_group_target": {
                    "known_outcome": True, "safe_terminal": True,
                    "group_future_violation": {
                        "end_effector": -1.0, "palm": -1.0,
                        "L5": l5, "L6": -1.0, "L7": -1.0,
                    },
                    "group_contact_sample_count": {
                        "end_effector": 0, "palm": 0, "L5": 0,
                        "L6": 0, "L7": 0,
                    },
                },
                "source_raw_protected_contact_count": 0,
                "raw_protected_contact_sample_count": 0,
                "source_physical_veto": False,
                "replayed_physical_veto": False,
                "source_maximum_CAR_m": 0.0,
                "replayed_maximum_CAR_m": 0.0,
                "source_effective_post_AEGIS_correction_l2_action": 1.0,
                "source_executed_actions": [[0.0] * 7 for _ in range(5)],
            }

        fresh = {
            "result_payload_sha256": "payload",
            "exact_case": {"candidates": [
                row("unsafe", 0.1), row("safe", -0.1), row("later", -0.2),
            ]},
        }
        value = ranked_prefix_summary(
            fresh, ["unsafe", "safe", "later"], [0.1, 0.2, 0.3],
            ["palm", "L5", "L6", "L7"], "file",
        )
        self.assertEqual(len(value["attempts"]), 2)
        self.assertEqual(value["selected"]["candidate_name"], "safe")


if __name__ == "__main__":
    unittest.main()
