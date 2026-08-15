import json
import tempfile
import unittest
from pathlib import Path

from main.multilink_ellipsoid.exact_group_boundary import (
    GROUPS, load_cases, load_config, summarize_cases, warning_step,
)
from main.multilink_ellipsoid.generic_action_boundary import candidate_definitions


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/vlsa_distal_exact_group_boundary_canary.v1.json"
NO_QP_CONFIG = ROOT / "configs/vlsa_distal_no_qp_l5_boundary_canary.v1.json"
GENERIC_CONFIG = ROOT / "configs/vlsa_distal_generic_l5_boundary_canary.v1.json"
OFFSET_COVERAGE_CONFIG = (
    ROOT / "configs/vlsa_distal_generic_l5_offset_coverage_canary.v1.json"
)
TIMING_LOCALIZATION_CONFIG = (
    ROOT / "configs/vlsa_distal_generic_l5_timing_localization_canary.v1.json"
)
TRAJECTORY_VALUE_CONFIG = (
    ROOT / "configs/vlsa_distal_generic_l5_trajectory_value_canary.v1.json"
)
PROSPECTIVE_CONFIG = (
    ROOT / "configs/vlsa_distal_prospective_l5_boundary_population.v1.json"
)


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
    def test_query_risk_canonicalizes_archived_perception_rotation(self):
        source = (
            ROOT / "scripts/evaluate_distal_query_action_risk_e05.py"
        ).read_text()
        self.assertIn(
            "_canonicalize_perception_ellipsoid_rotation(", source,
        )
        self.assertNotIn(
            '"R2": perception["mvee_rotation"]', source,
        )

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

    def test_generic_l5_contract_uses_explicit_query_states(self):
        config = load_config(GENERIC_CONFIG)
        cases = load_cases(ROOT / config["selection_manifest"], config)
        self.assertEqual(
            [warning_step(case, config) for case in cases],
            [60, 215, 185, 25, 115],
        )
        self.assertFalse(
            config["candidate_bank"][
                "released_AEGIS_EE_applied_to_every_candidate"
            ]
        )

    def test_generic_l5_candidates_are_symmetric_and_leave_non_xyz_fixed(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy unavailable in the local structural environment")

        config = load_config(GENERIC_CONFIG)
        nominal = np.zeros((5, 7), dtype=np.float64)
        candidates = candidate_definitions(nominal, config["candidate_bank"])
        self.assertEqual(len(candidates), 13)
        self.assertEqual(candidates[0]["name"], "nominal")
        by_name = {row["name"]: np.asarray(row["actions"]) for row in candidates}
        for radius in (0.5, 1.5):
            for axis in ("x", "y", "z"):
                self.assertTrue(np.allclose(
                    by_name["%s_pos_r%0.2f" % (axis, radius)][:, :3],
                    -by_name["%s_neg_r%0.2f" % (axis, radius)][:, :3],
                ))
        self.assertTrue(all(
            np.array_equal(np.asarray(row["actions"])[:, 3:], nominal[:, 3:])
            for row in candidates
        ))

    def test_offset_coverage_canary_uses_earlier_independent_query_states(self):
        config = load_config(OFFSET_COVERAGE_CONFIG)
        cases = load_cases(ROOT / config["selection_manifest"], config)
        self.assertEqual(
            [warning_step(case, config) for case in cases],
            [210, 140, 15, 90, 60],
        )
        self.assertEqual(len({case["task_level_group_id"] for case in cases}), 5)
        self.assertTrue(all(
            15 <= case["first_target_contact_step"] - case["state_step"] <= 18
            for case in cases
        ))
        self.assertFalse(
            config["candidate_bank"][
                "released_AEGIS_EE_applied_to_every_candidate"
            ]
        )

    def test_timing_localization_changes_only_registered_query_steps(self):
        previous = load_config(OFFSET_COVERAGE_CONFIG)
        current = load_config(TIMING_LOCALIZATION_CONFIG)
        previous_cases = load_cases(ROOT / previous["selection_manifest"], previous)
        current_cases = load_cases(ROOT / current["selection_manifest"], current)
        self.assertEqual(
            [case["case_id"] for case in current_cases],
            [case["case_id"] for case in previous_cases],
        )
        self.assertEqual(
            [warning_step(case, current) for case in current_cases],
            [200, 150, 25, 100, 60],
        )
        self.assertEqual(current["candidate_bank"], previous["candidate_bank"])
        self.assertEqual(current["exact_group_target"], previous["exact_group_target"])
        self.assertEqual(current["risk_target"], previous["risk_target"])

    def test_trajectory_value_canary_changes_only_capture_contract(self):
        previous = load_config(TIMING_LOCALIZATION_CONFIG)
        current = load_config(TRAJECTORY_VALUE_CONFIG)
        self.assertEqual(current["candidate_bank"], previous["candidate_bank"])
        self.assertEqual(current["exact_group_target"], previous["exact_group_target"])
        self.assertEqual(current["risk_target"], previous["risk_target"])
        self.assertEqual(
            [warning_step(case, current) for case in load_cases(
                ROOT / current["selection_manifest"], current,
            )],
            [200, 150, 25, 100, 60],
        )
        self.assertEqual(
            current["trajectory_policy_value"]["value_training_phases"],
            ["backup", "terminal_hold"],
        )
        self.assertEqual(
            current["trajectory_policy_value"]["internal_substeps"],
            "label_authority_only",
        )

    def test_prospective_population_freezes_episode_splits_before_outcomes(self):
        config = load_config(PROSPECTIVE_CONFIG)
        cases = load_cases(ROOT / config["selection_manifest"], config)
        self.assertEqual(len(cases), 10)
        self.assertEqual(
            [case["split"] for case in cases],
            ["train"] * 6 + ["validation"] * 2 + ["test"] * 2,
        )
        self.assertEqual(len({case["episode_group_id"] for case in cases}), 10)
        self.assertTrue(all(
            case["prospective_split_frozen_before_candidate_outcomes"] is True
            and case["state_step"]
            == ((case["first_target_contact_step"] - 5) // 5) * 5
            for case in cases
        ))
        self.assertFalse(config["learned_correction_QP_enabled"])
        self.assertIn("V_loss_or_V_supervision", config["forbidden"])

    def test_prospective_split_gate_requires_two_sided_validation_and_test(self):
        config = load_config(PROSPECTIVE_CONFIG)
        splits = ["train"] * 6 + ["validation"] * 2 + ["test"] * 2
        rows = []
        for index, split in enumerate(splits):
            candidates = [_candidate("L5", 0.2) for _ in range(13)]
            if index < 4 or split != "train":
                candidates[0] = _candidate("L5", -0.1)
            row = _case(index, "L5", candidates)
            row["selection"]["split"] = split
            rows.append(row)
        summary = summarize_cases(rows, config)
        self.assertTrue(summary["apparatus_pass"])
        self.assertEqual(
            summary["prospective_split_summary"]["train"]["two_sided_case_count"],
            4,
        )
        self.assertTrue(summary["q_only_prediction_gate_authorized"])
        self.assertTrue(summary["training_authorized"])
        rows[-1]["exact_case"]["candidates"] = [
            _candidate("L5", 0.2) for _ in range(13)
        ]
        blocked = summarize_cases(rows, config)
        self.assertFalse(blocked["q_only_prediction_gate_authorized"])
        self.assertFalse(blocked["training_authorized"])

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

    def test_optional_targeted_coverage_gate_does_not_authorize_same_bank(self):
        config = load_config(OFFSET_COVERAGE_CONFIG)
        rows = []
        for index in range(5):
            candidates = [_candidate("L5", 0.2) for _ in range(13)]
            if index < 3:
                candidates[0] = _candidate("L5", -0.1)
            rows.append(_case(index, "L5", candidates))
        summary = summarize_cases(rows, config)
        self.assertEqual(summary["two_sided_case_count"], 3)
        self.assertTrue(summary["targeted_coverage_canary_pass"])
        self.assertFalse(summary["same_bank_grouped_collection_authorized"])
        self.assertFalse(summary["training_authorized"])

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
