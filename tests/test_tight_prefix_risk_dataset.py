import copy
import unittest
from pathlib import Path

from main.multilink_ellipsoid.compiled_box_risk_target_audit import (
    candidate_action_sequence,
)
from main.multilink_ellipsoid.tight_prefix_risk_dataset import (
    GROUPS, ROLLOUT_SCOPE, load_config, summarize_records,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/vlsa_tight_prefix_risk_dataset.v1.json"


class TightPrefixRiskDatasetTest(unittest.TestCase):
    def test_config_and_split_are_frozen(self):
        config = load_config(CONFIG, repo_root=ROOT)
        self.assertEqual(len(config["cases"]), 24)
        self.assertEqual(config["rollout_scope"], ROLLOUT_SCOPE)
        self.assertFalse(
            config["candidate_bank"]["released_AEGIS_EE_QP_enabled"]
        )
        self.assertFalse(config["candidate_bank"]["continuation_enabled"])

    def test_prefix_scope_drops_continuation_without_changing_default(self):
        candidate = {
            "actions": [[0.0] * 7 for _ in range(5)],
            "backup": {
                "decisions": [{"selected_action": [1.0] * 7}],
                "terminal_hold": {"executed_actions": [[2.0] * 7]},
            },
        }
        prefix, phases = candidate_action_sequence(
            candidate, rollout_scope=ROLLOUT_SCOPE,
        )
        self.assertEqual(len(prefix), 5)
        self.assertEqual(phases, ["prefix"] * 5)
        complete, complete_phases = candidate_action_sequence(candidate)
        self.assertEqual(len(complete), 7)
        self.assertEqual(complete_phases[-2:], ["backup", "terminal_hold"])

    def test_summary_accepts_exact_prefix_population(self):
        config = load_config(CONFIG, repo_root=ROOT)
        records = []
        for index, item in enumerate(config["cases"]):
            candidates = []
            for candidate_index in range(13):
                risks = {group: -1.0 for group in GROUPS}
                if candidate_index == 1:
                    risks["L5"] = 0.1
                candidates.append({
                    "action_count": 5,
                    "sample_count": 125,
                    "phase_sample_counts": {
                        "prefix": 125, "backup": 0, "terminal_hold": 0,
                    },
                    "replayed_maximum_CAR_m": 0.0,
                    "exact_group_target": {
                        "group_future_violation": risks,
                        "group_contact_sample_count": {
                            group: 0 for group in GROUPS
                        },
                    },
                })
            records.append({
                "case_id": item["case_id"],
                "split": item["split"],
                "case": {
                    "source_replay_exact": True,
                    "rollout_scope": ROLLOUT_SCOPE,
                    "candidates": candidates,
                    "exact_group_target": {
                        "robot_primitive_certificate_pass": True,
                        "initial_group_normalized_radial_slack": {
                            group: 1.0 for group in GROUPS
                        },
                        "initial_group_contact_sample_count": {
                            group: 0 for group in GROUPS
                        },
                    },
                },
            })
        summary = summarize_records(records, config)
        self.assertTrue(summary["dataset_gate_pass"])
        self.assertEqual(summary["candidate_count"], 312)
        self.assertEqual(
            summary["split_summary"]["train"]["per_group"]["L5"][
                "two_sided_state_count"
            ],
            14,
        )

    def test_summary_rejects_geometry_false_safe(self):
        config = load_config(CONFIG, repo_root=ROOT)
        records = []
        for item in config["cases"]:
            candidates = []
            for _ in range(13):
                candidates.append({
                    "action_count": 5,
                    "sample_count": 125,
                    "phase_sample_counts": {
                        "prefix": 125, "backup": 0, "terminal_hold": 0,
                    },
                    "replayed_maximum_CAR_m": 0.0,
                    "exact_group_target": {
                        "group_future_violation": {
                            group: -1.0 for group in GROUPS
                        },
                        "group_contact_sample_count": {
                            group: 0 for group in GROUPS
                        },
                    },
                })
            records.append({
                "case_id": item["case_id"], "split": item["split"],
                "case": {
                    "source_replay_exact": True,
                    "rollout_scope": ROLLOUT_SCOPE,
                    "candidates": candidates,
                    "exact_group_target": {
                        "robot_primitive_certificate_pass": True,
                        "initial_group_normalized_radial_slack": {
                            group: 1.0 for group in GROUPS
                        },
                        "initial_group_contact_sample_count": {
                            group: 0 for group in GROUPS
                        },
                    },
                },
            })
        broken = copy.deepcopy(records)
        broken[0]["case"]["candidates"][0]["exact_group_target"][
            "group_contact_sample_count"
        ]["palm"] = 1
        self.assertFalse(summarize_records(broken, config)["dataset_gate_pass"])


if __name__ == "__main__":
    unittest.main()
