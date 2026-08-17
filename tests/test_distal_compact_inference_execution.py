import inspect
import unittest
from pathlib import Path

from main.multilink_ellipsoid.compact_inference_execution import (
    case_definition, frozen_model_selection, load_config, summarize_pair,
)
from scripts.collect_distal_exact_group_boundary import collect


CONFIG = Path("configs/vlsa_distal_compact_inference_execution.v1.json")


def _candidate(name, *, l5=-0.1, l6=-0.1, palm=-0.1, known=True):
    risks = {
        "end_effector": -0.1, "palm": palm, "L5": l5,
        "L6": l6, "L7": -0.1,
    }
    return {
        "name": name,
        "source_terminal_status": "SAFE_TERMINAL" if max(
            palm, l5, l6
        ) <= 0.0 else "UNSAFE_CONTACT_OR_CAR",
        "exact_group_target": {
            "known_outcome": known,
            "safe_terminal": bool(known and max(palm, l5, l6) <= 0.0),
            "group_future_violation": risks,
            "group_contact_sample_count": {
                group: 0 for group in risks
            },
        },
        "source_raw_protected_contact_count": 0,
        "raw_protected_contact_sample_count": 0,
        "source_physical_veto": False,
        "replayed_physical_veto": False,
        "source_maximum_CAR_m": 0.0,
        "replayed_maximum_CAR_m": 0.0,
        "source_effective_post_AEGIS_correction_l2_action": (
            0.0 if name == "nominal" else 2.0
        ),
        "source_executed_actions": [[0.0] * 7 for _ in range(5)],
    }


class CompactInferenceExecutionTest(unittest.TestCase):
    def test_config_freezes_benefit_and_known_failure_cases(self):
        config = load_config(CONFIG)
        self.assertEqual(len(config["cases"]), 4)
        self.assertEqual(
            [case["stratum"] for case in config["cases"]].count(
                "compact_known_remaining_L5_failure"
            ),
            2,
        )
        self.assertEqual(case_definition(config, 0)["split"], "validation")

    def test_model_selection_returns_only_causal_frozen_fields(self):
        case = load_config(CONFIG)["cases"][0]
        row = {
            "state_id": case["case_id"],
            "selected_candidate": case["selected_candidate"],
            "selected_predicted_primary": case["selected_predicted_primary"],
            "selected_actual_all_physical_safe": True,
            "selected_failed_rows": [],
        }
        compact = {"compact_shared_7D": {"selectors": {
            "validation": {"minimum_predicted_primary_risk": {
                "states": [row],
            }},
        }}}
        selected = frozen_model_selection(
            compact, case, "minimum_predicted_primary_risk",
        )
        self.assertNotIn("selected_actual_all_physical_safe", selected)
        self.assertNotIn("selected_failed_rows", selected)
        self.assertEqual(selected["selected_candidate"], case["selected_candidate"])

    def test_pair_reports_improvement_and_persistent_failure(self):
        exact = {
            "source_snapshot_sha256": "source",
            "replayed_snapshot_sha256": "source",
            "state_hash_matches": True,
            "source_replay_exact": True,
            "exact_group_target": {
                "initial_group_normalized_radial_slack": {
                    "end_effector": 1.0, "palm": 1.0, "L5": 1.0,
                    "L6": 1.0, "L7": 1.0,
                },
            },
            "candidates": [
                _candidate("nominal", l5=0.2),
                _candidate("selected", l5=-0.2),
            ],
        }
        pair = summarize_pair(
            {"exact_case": exact}, "selected", ["palm", "L5", "L6", "L7"],
        )
        self.assertTrue(pair["collision_avoided"])
        exact["candidates"][1] = _candidate("selected", l5=0.1)
        pair = summarize_pair(
            {"exact_case": exact}, "selected", ["palm", "L5", "L6", "L7"],
        )
        self.assertTrue(pair["collision_persisted"])

    def test_collector_subset_is_opt_in_and_sequential(self):
        source = inspect.getsource(collect)
        self.assertIn("candidate_subset_names", source)
        self.assertIn("inference candidate subset must execute sequentially", source)
        self.assertIn(
            'bank.get("selected_candidate_names") is not None', source,
        )
        self.assertIn("else:\n                rows = grid_candidate_definitions", source)


if __name__ == "__main__":
    unittest.main()
