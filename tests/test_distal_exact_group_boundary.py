import itertools
import json
import tempfile
import unittest
from pathlib import Path

from main.multilink_ellipsoid.exact_group_boundary import (
    GROUPS, WHOLE_BODY_GROUPS, load_cases, load_config, summarize_cases,
    warning_step,
)
from main.multilink_ellipsoid.active_boundary_search import (
    candidate_definitions as grid_candidate_definitions,
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
SPATIAL_PROGRESSIVE_CONFIG = (
    ROOT / "configs/vlsa_distal_spatial_i_t3_progressive_boundary_extension.v1.json"
)
SPATIAL_TIMING_CONFIG = (
    ROOT / "configs/vlsa_distal_spatial_i_t3_timing_localization.v1.json"
)
SPATIAL_E00_EXCITATION_CONFIG = (
    ROOT / "configs/vlsa_distal_spatial_i_t3_e00_candidate_excitation.v1.json"
)
WHOLE_BODY_SUPERSET_CONFIG = (
    ROOT / "configs/vlsa_distal_e00_whole_body_superset_canary.v1.json"
)
WHOLE_BODY_PROSPECTIVE_CONFIG = (
    ROOT / "configs/vlsa_distal_whole_body_prospective_population.v1.json"
)
WHOLE_BODY_EXTENSION_CONFIG = (
    ROOT / "configs/vlsa_distal_whole_body_progressive_extension.v1.json"
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
    def test_collector_can_bind_raw_pi05_state_to_paired_geometry_only(self):
        source = (
            ROOT / "scripts/collect_distal_exact_group_boundary.py"
        ).read_text()
        self.assertIn('selected.get("aegis_geometry_result_relative_path")', source)
        self.assertIn('"state_or_action_source": False', source)
        self.assertIn("nominal_action_source=config[\"state_selection\"].get(", source)

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

    def test_whole_body_superset_freezes_ee_palm_and_distal_records(self):
        config = load_config(WHOLE_BODY_SUPERSET_CONFIG)
        cases = load_cases(ROOT / config["selection_manifest"], config)
        self.assertEqual([warning_step(case, config) for case in cases], [65])
        self.assertEqual(
            config["exact_group_target"]["group_order"],
            list(WHOLE_BODY_GROUPS),
        )
        self.assertEqual(
            config["exact_group_target"]["robot_rows"],
            {
                "end_effector": [0], "palm": [1],
                "L5": [2, 3, 4], "L6": [5, 6], "L7": [7, 8],
            },
        )
        self.assertTrue(
            config["artifact_superset"]["capture_internal_substep_ee_pose"]
        )
        self.assertTrue(
            config["artifact_superset"]["capture_internal_substep_palm_pose"]
        )
        self.assertTrue(config["trajectory_policy_value"]["capture_action_boundaries"])
        self.assertFalse(config["learned_correction_QP_enabled"])

    def test_whole_body_prospective_split_is_frozen_before_outcomes(self):
        config = load_config(WHOLE_BODY_PROSPECTIVE_CONFIG)
        cases = load_cases(ROOT / config["selection_manifest"], config)
        self.assertEqual(
            [case["split"] for case in cases],
            ["train"] * 4 + ["validation"] * 2 + ["test"] * 2,
        )
        self.assertEqual(
            [warning_step(case, config) for case in cases],
            [115, 70, 25, 130, 185, 25, 80, 65],
        )
        self.assertEqual(
            config["exact_group_target"]["group_order"],
            list(WHOLE_BODY_GROUPS),
        )
        self.assertEqual(config["gate"]["required_two_sided_by_split"], {
            "train": 4, "validation": 2, "test": 2,
        })
        self.assertTrue(all(
            case["prospective_split_frozen_before_candidate_outcomes"]
            for case in cases
        ))
        self.assertFalse(config["learned_correction_QP_enabled"])

    def test_whole_body_extension_is_progressive_and_cannot_authorize_alone(self):
        config = load_config(WHOLE_BODY_EXTENSION_CONFIG)
        cases = load_cases(ROOT / config["selection_manifest"], config)
        self.assertEqual(
            [case["split"] for case in cases],
            ["train"] * 10 + ["validation"] * 2 + ["test"] * 4,
        )
        self.assertEqual(
            [warning_step(case, config) for case in cases],
            [20, 25, 130, 105, 105, 100, 150, 55, 220, 20, 110, 180, 65, 115, 25, 105],
        )
        self.assertEqual(config["candidate_bank"]["candidate_count_per_job"], 13)
        self.assertEqual(len(config["candidate_bank"]["selected_candidate_names"]), 13)
        self.assertEqual(
            config["training_authorization_mode"],
            "combined_24_state_external_audit_only",
        )
        self.assertTrue(all(
            case["prospective_split_frozen_before_candidate_outcomes"]
            for case in cases
        ))
        self.assertFalse(config["learned_correction_QP_enabled"])

    def test_whole_body_extension_generates_exact_frozen_13_bank(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy unavailable in the local structural environment")
        from scripts.collect_distal_exact_group_boundary import (
            _frozen_grid_subset_candidates,
        )

        config = load_config(WHOLE_BODY_EXTENSION_CONFIG)
        rows = _frozen_grid_subset_candidates(
            np.zeros((5, 7), dtype=np.float64),
            {
                "normal": [1.0, 0.0, 0.0],
                "tangent_up": [0.0, 1.0, 0.0],
                "tangent_side": [0.0, 0.0, 1.0],
            },
            config["candidate_bank"],
        )
        self.assertEqual(
            [row["name"] for row in rows],
            config["candidate_bank"]["selected_candidate_names"],
        )
        self.assertEqual(len(rows), 13)

    def test_released_ee_proxy_uses_body_orientation_not_grip_site_xmat(self):
        source = (ROOT / "main/multilink_ellipsoid/shadow.py").read_text()
        start = source.index("def _released_aegis_end_effector_ellipsoid")
        end = source.index("\ndef _resolved_rate_nominal", start)
        helper = source[start:end]
        self.assertIn("data.xmat[body_id]", helper)
        self.assertNotIn("data.site_xmat[site_id]", helper)

    def test_whole_body_audit_imports_eef_site_helper_from_shadow(self):
        source = (
            ROOT / "scripts/audit_distal_compiled_box_risk_target.py"
        ).read_text()
        evaluator_import = source.split(
            "from main.multilink_ellipsoid.compiled_box_risk_target_audit",
            1,
        )[0]
        shadow_import = source.split(
            "from main.multilink_ellipsoid.shadow import (", 1,
        )[1].split(")", 1)[0]
        self.assertNotIn("_eef_site_id", evaluator_import)
        self.assertIn("_eef_site_id", shadow_import)

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

    def test_spatial_progressive_population_does_not_require_task_success(self):
        config = load_config(SPATIAL_PROGRESSIVE_CONFIG)
        self.assertFalse(config["state_selection"]["require_archived_task_success"])
        self.assertEqual(
            set(config["progressive_historical_source_commits"]),
            {
                "vlsa-t1-spatial-i-t3-e00", "vlsa-t1-spatial-i-t3-e03",
                "vlsa-t1-spatial-i-t3-e15", "vlsa-t1-spatial-i-t3-e01",
                "vlsa-t1-spatial-i-t3-e04", "vlsa-t1-spatial-i-t3-e13",
            },
        )

    def test_spatial_timing_localization_changes_only_opened_development_steps(self):
        previous = load_config(SPATIAL_PROGRESSIVE_CONFIG)
        current = load_config(SPATIAL_TIMING_CONFIG)
        previous_by_episode = {
            case["episode_group_id"]: case
            for case in load_cases(ROOT / previous["selection_manifest"], previous)
            if case["split"] == "train"
        }
        current_cases = load_cases(ROOT / current["selection_manifest"], current)
        self.assertEqual(
            [case["episode_group_id"] for case in current_cases],
            [
                "vlsa-t1-spatial-i-t3-e00",
                "vlsa-t1-spatial-i-t3-e03",
                "vlsa-t1-spatial-i-t3-e15",
            ],
        )
        self.assertEqual(
            [warning_step(case, current) for case in current_cases],
            [65, 55, 65],
        )
        self.assertTrue(all(
            case["state_step"]
            == previous_by_episode[case["episode_group_id"]]["state_step"] - 5
            for case in current_cases
        ))
        self.assertEqual(current["candidate_bank"], previous["candidate_bank"])
        self.assertEqual(current["exact_group_target"], previous["exact_group_target"])
        self.assertEqual(current["risk_target"], previous["risk_target"])
        self.assertFalse(current["learned_correction_QP_enabled"])

    def test_spatial_e00_excitation_changes_only_candidate_coverage(self):
        previous = load_config(SPATIAL_TIMING_CONFIG)
        current = load_config(SPATIAL_E00_EXCITATION_CONFIG)
        cases = load_cases(ROOT / current["selection_manifest"], current)
        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0]["episode_group_id"], "vlsa-t1-spatial-i-t3-e00")
        self.assertEqual(warning_step(cases[0], current), 65)
        self.assertEqual(current["exact_group_target"], previous["exact_group_target"])
        self.assertEqual(current["risk_target"], previous["risk_target"])
        self.assertFalse(current["learned_correction_QP_enabled"])

        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy unavailable in the local structural environment")
        nominal = np.zeros((5, 7), dtype=np.float64)
        frame = {
            "normal": [1.0, 0.0, 0.0],
            "tangent_up": [0.0, 1.0, 0.0],
            "tangent_side": [0.0, 0.0, 1.0],
        }
        candidates = grid_candidate_definitions(
            nominal,
            frame,
            {"finite_search": current["candidate_bank"]},
            current["candidate_bank"]["temporal_profile"],
        )
        self.assertEqual(len(candidates), 27)
        self.assertEqual(candidates[0]["name"], "nominal")
        self.assertEqual(candidates[0]["requested_alpha"], 0.0)
        self.assertTrue(all(
            row["requested_alpha"] == 2.0 for row in candidates[1:]
        ))
        self.assertTrue(all(
            np.array_equal(np.asarray(row["actions"])[:, 3:], nominal[:, 3:])
            for row in candidates
        ))
        self.assertEqual(
            {tuple(row["spatial_coefficients"]) for row in candidates[1:]},
            {
                coefficients
                for coefficients in itertools.product((-1, 0, 1), repeat=3)
                if coefficients != (0, 0, 0)
            },
        )

    def test_prospective_population_can_capture_policy_value_contexts(self):
        value = json.loads(SPATIAL_PROGRESSIVE_CONFIG.read_text())
        value["trajectory_policy_value"] = {
            "capture_action_boundaries": True,
            "training_state_unit": "controller_action_boundary",
            "value_training_phases": ["backup", "terminal_hold"],
            "internal_substeps": "label_authority_only",
            "target": "exact_reverse_suffix_maximum_positive_is_unsafe",
            "timeouts": "censored_not_training_samples",
            "prefix_state_rule": "exclude_unless_remaining_action_suffix_and_phase_are_explicitly_conditioned",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(value))
            loaded = load_config(path)
        self.assertEqual(
            loaded["trajectory_policy_value"]["value_training_phases"],
            ["backup", "terminal_hold"],
        )

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
