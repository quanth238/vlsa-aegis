from __future__ import annotations

from pathlib import Path
import unittest

from main.multilink_ellipsoid.prospective_l5_context_ablation import (
    classify_ablation, direct_l5_osc_feature, load_config,
)


def _metrics(*, rmse=1.0, false_safes=4, direction=0.25, support=2, recall=1.0, near=0.05):
    return {
        "global_RMSE": rmse,
        "false_safe_count": false_safes,
        "improvement_direction_accuracy": direction,
        "supported_state_count": support,
        "safe_recall": recall,
        "near_boundary_global_RMSE": near,
    }


class ProspectiveL5ContextAblationTest(unittest.TestCase):
    def test_config_freezes_matched_ablation(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = load_config(root / "configs/vlsa_distal_prospective_l5_context_ablation.v1.json")
        self.assertEqual(config["arms"]["relative_endpoint_9D"]["input_dimension"], 9)
        self.assertEqual(config["arms"]["direct_L5_OSC_33D"]["input_dimension"], 33)
        self.assertTrue(all(config["matched_factors"].values()))
        self.assertIn("QP_or_gradient_correction", config["forbidden"])

    def test_direct_feature_is_exactly_33D_and_causal(self) -> None:
        rows = []
        for index in range(3):
            rows.append({
                "body_name": "robot0_link5",
                "center_m": [1.0 + index, 2.0, 3.0],
                "semiaxes_m": [0.1, 0.2, 0.3],
            })
        exact = {
            "initial_empirical_robot_rows": rows,
            "initial_compiled_obstacle_boxes": [{
                "center_m": [1.0, 1.0, 1.0],
                "rotation": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                "half_extents_m": [0.5, 0.5, 0.5],
            }],
            "exact_group_target": {
                "initial_row_normalized_radial_slack": [9.0, 0.1, 0.2, 0.3],
            },
            "physical_context": {
                "controller_snapshot": {"goal_pos": [0.4, 0.5, 0.6]},
                "eef_position_m": [0.1, 0.2, 0.3],
            },
            "forbidden_future_marker": [999.0],
        }
        feature = direct_l5_osc_feature([0.0] * 9, exact)
        self.assertEqual(len(feature), 33)
        self.assertEqual(feature[27:30], [0.1, 0.2, 0.3])
        for actual, expected in zip(feature[30:33], [0.3, 0.3, 0.3]):
            self.assertAlmostEqual(actual, expected)
        self.assertNotIn(999.0, feature)

    def test_decision_separates_material_and_strong_pass(self) -> None:
        root = Path(__file__).resolve().parents[1]
        decision = load_config(root / "configs/vlsa_distal_prospective_l5_context_ablation.v1.json")["decision"]
        baseline = {"validation": _metrics(), "test": _metrics()}
        candidate = {
            "validation": _metrics(rmse=0.4, false_safes=1, direction=0.5, near=0.2),
            "test": _metrics(rmse=0.7, false_safes=2, direction=0.5, near=0.2),
        }
        verdict, _ = classify_ablation(baseline, candidate, decision)
        self.assertEqual(verdict, "material_representation_improvement_but_prediction_gate_failed")
        candidate = {
            "validation": _metrics(rmse=0.05, false_safes=0, direction=0.9),
            "test": _metrics(rmse=0.05, false_safes=0, direction=0.9),
        }
        verdict, _ = classify_ablation(baseline, candidate, decision)
        self.assertEqual(verdict, "strong_diagnostic_pass_opened_test_only")


if __name__ == "__main__":
    unittest.main()
