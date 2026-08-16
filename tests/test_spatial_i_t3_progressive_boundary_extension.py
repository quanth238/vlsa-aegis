from __future__ import annotations

from pathlib import Path
import unittest

from main.multilink_ellipsoid.exact_group_boundary import load_cases, load_config


class SpatialIT3ProgressiveBoundaryExtensionTest(unittest.TestCase):
    def test_split_and_candidate_protocol_are_frozen(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = load_config(
            root / "configs/vlsa_distal_spatial_i_t3_progressive_boundary_extension.v1.json"
        )
        cases = load_cases(root / config["selection_manifest"], config)
        self.assertEqual(len(cases), 7)
        self.assertEqual(
            [sum(case["split"] == split for case in cases) for split in ("train", "validation", "test")],
            [3, 2, 2],
        )
        self.assertEqual(config["candidate_bank"]["radii"], [0.5, 1.5])
        self.assertEqual(config["candidate_bank"]["axis_order"], ["x", "y", "z"])
        self.assertFalse(config["candidate_bank"]["released_AEGIS_EE_applied_to_every_candidate"])
        self.assertFalse(config["learned_correction_QP_enabled"])

    def test_opened_cases_are_training_only_and_holdouts_are_new(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = load_config(
            root / "configs/vlsa_distal_spatial_i_t3_progressive_boundary_extension.v1.json"
        )
        cases = load_cases(root / config["selection_manifest"], config)
        by_id = {case["case_id"]: case for case in cases}
        for case_id in config["state_selection"]["prior_opened_cases_training_only"]:
            self.assertEqual(by_id[case_id]["split"], "train")
        for case_id in config["state_selection"]["new_holdout_cases"]:
            self.assertIn(by_id[case_id]["split"], ("validation", "test"))
        self.assertTrue(all(case["prospective_split_frozen_before_candidate_outcomes"] for case in cases))

    def test_active_obstacle_bindings_match_archived_episode_contracts(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = load_config(
            root / "configs/vlsa_distal_spatial_i_t3_progressive_boundary_extension.v1.json"
        )
        cases = load_cases(root / config["selection_manifest"], config)
        self.assertEqual(
            {case["case_id"]: case["active_obstacle_name"] for case in cases},
            {
                "vlsa-t1-spatial-i-t3-e00": "wine_bottle_obstacle_1",
                "vlsa-t1-spatial-i-t3-e03": "wine_bottle_obstacle_1",
                "vlsa-t1-spatial-i-t3-e15": "wine_bottle_obstacle_1",
                "vlsa-t1-spatial-i-t3-e01": "red_coffee_mug_obstacle_1",
                "vlsa-t1-spatial-i-t3-e04": "red_coffee_mug_obstacle_1",
                "vlsa-t1-spatial-i-t3-e02": "yellow_book_obstacle_1",
                "vlsa-t1-spatial-i-t3-e13": "red_coffee_mug_obstacle_1",
            },
        )

    def test_gate_requires_every_new_group_to_be_two_sided(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = load_config(
            root / "configs/vlsa_distal_spatial_i_t3_progressive_boundary_extension.v1.json"
        )
        self.assertEqual(config["gate"]["required_two_sided_by_split"], {"train": 3, "validation": 2, "test": 2})
        self.assertEqual(config["gate"]["required_initially_safe_by_split"], {"train": 3, "validation": 2, "test": 2})
        self.assertIn("MLP_training_before_split_coverage_pass", config["forbidden"])


if __name__ == "__main__":
    unittest.main()
