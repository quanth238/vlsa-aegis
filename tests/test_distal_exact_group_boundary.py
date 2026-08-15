import json
import tempfile
import unittest
from pathlib import Path

from main.multilink_ellipsoid.exact_group_boundary import (
    GROUPS, load_cases, load_config, summarize_cases, warning_step,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/vlsa_distal_exact_group_boundary_canary.v1.json"
NO_QP_CONFIG = ROOT / "configs/vlsa_distal_no_qp_l5_boundary_canary.v1.json"


def _candidate(group, slack, *, known=True, contact=0):
    return {
        "exact_group_target": {
            "known_outcome": known,
            "group_future_violation": {key: -float(slack if key == group else 1.0) for key in GROUPS},
            "group_minimum_normalized_radial_slack": {key: float(slack if key == group else 1.0) for key in GROUPS},
            "group_contact_sample_count": {key: int(contact if key == group else 0) for key in GROUPS},
        }
    }


def _case(index, group, candidates):
    return {
        "case_id": "case-%d" % index,
        "selection": {"target_group": group},
        "exact_case": {
            "source_replay_exact": True,
            "state_hash_matches": True,
            "exact_group_target": {
                "robot_primitive_certificate_pass": True,
                "initial_group_normalized_radial_slack": {key: 0.2 for key in GROUPS},
                "initial_group_contact_sample_count": {key: 0 for key in GROUPS},
            },
            "candidates": candidates,
        },
    }


class ExactGroupBoundaryTest(unittest.TestCase):
    def test_frozen_contract_and_warning_steps(self):
        config = load_config(CONFIG)
        cases = load_cases(ROOT / config["selection_manifest"], config)
        self.assertEqual([warning_step(case, config) for case in cases], [20, 25, 110])
        self.assertTrue(config["candidate_bank"]["released_AEGIS_EE_applied_to_every_candidate"])
        self.assertFalse(config["learned_correction_QP_enabled"])

    def test_no_qp_l5_contract_and_warning_steps(self):
        config = load_config(NO_QP_CONFIG)
        cases = load_cases(ROOT / config["selection_manifest"], config)
        self.assertEqual(
            [warning_step(case, config) for case in cases],
            [55, 220, 180, 25, 115],
        )
        self.assertEqual([case["target_group"] for case in cases], ["L5"] * 5)
        self.assertFalse(
            config["candidate_bank"][
                "released_AEGIS_EE_applied_to_every_candidate"
            ]
        )
        self.assertFalse(config["learned_correction_QP_enabled"])

    def test_no_qp_gate_uses_exact_state_not_legacy_proxy_comparator(self):
        config = load_config(NO_QP_CONFIG)
        rows = []
        for index in range(5):
            candidates = [_candidate("L5", -0.1), _candidate("L5", 0.1)]
            candidates += [_candidate("L5", 0.2) for _ in range(7)]
            row = _case(index, "L5", candidates)
            row["exact_case"]["source_replay_exact"] = False
            row["exact_case"]["state_hash_matches"] = True
            rows.append(row)
        summary = summarize_cases(rows, config)
        self.assertFalse(summary["source_replay_exact"])
        self.assertTrue(summary["source_state_hash_exact"])
        self.assertTrue(summary["apparatus_pass"])
        self.assertTrue(summary["same_bank_grouped_collection_authorized"])

    def test_two_sided_bank_authorizes_only_grouped_collection(self):
        config = load_config(CONFIG)
        rows = []
        for index, group in enumerate(GROUPS):
            candidates = [_candidate(group, -0.1), _candidate(group, 0.1)]
            candidates += [_candidate(group, 0.2) for _ in range(7)]
            rows.append(_case(index, group, candidates))
        summary = summarize_cases(rows, config)
        self.assertTrue(summary["apparatus_pass"])
        self.assertTrue(summary["same_bank_grouped_collection_authorized"])
        self.assertFalse(summary["training_authorized"])
        self.assertFalse(summary["QP_authorized"])

    def test_timeout_is_not_promoted_and_one_sided_bank_does_not_pass(self):
        config = load_config(CONFIG)
        rows = []
        for index, group in enumerate(GROUPS):
            candidates = [_candidate(group, 0.1) for _ in range(8)]
            candidates.append(_candidate(group, -0.1, known=False))
            rows.append(_case(index, group, candidates))
        summary = summarize_cases(rows, config)
        self.assertTrue(summary["apparatus_pass"])
        self.assertFalse(summary["same_bank_grouped_collection_authorized"])
        self.assertEqual(summary["per_case"][0]["unknown_timeout_count"], 1)

    def test_contact_with_positive_target_slack_fails_apparatus(self):
        config = load_config(CONFIG)
        rows = []
        for index, group in enumerate(GROUPS):
            candidates = [_candidate(group, -0.1), _candidate(group, 0.1)]
            candidates += [_candidate(group, 0.2) for _ in range(7)]
            rows.append(_case(index, group, candidates))
        rows[0]["exact_case"]["candidates"][1] = _candidate("palm", 0.1, contact=1)
        summary = summarize_cases(rows, config)
        self.assertFalse(summary["apparatus_pass"])
        self.assertEqual(summary["physical_false_safe_count"], 1)


if __name__ == "__main__":
    unittest.main()
