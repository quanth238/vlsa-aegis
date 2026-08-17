from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from main.multilink_ellipsoid.compact_selector_ablation import (
    enrich_records, evaluate_arm, exact_float_lists_close, load_config,
    validation_margins,
)


class CompactSelectorAblationTest(unittest.TestCase):
    def test_registered_config(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = load_config(
            root / "configs/vlsa_distal_compact_selector_ablation.v1.json"
        )
        self.assertEqual(config["margin"]["fit_split"], "validation")
        self.assertEqual(config["evaluation"]["new_simulator_rollout_count"], 0)
        self.assertFalse(config["evaluation"]["retraining"])

    @staticmethod
    def _record(state: str, name: str, order: int, correction: float,
                actual: list[float], predicted: list[float]) -> dict:
        return {
            "state_id": state,
            "split": "validation",
            "candidate_name": name,
            "candidate_order": order,
            "correction": correction,
            "known_outcome": True,
            "physical_veto": False,
            "actual_by_row": dict(enumerate(actual)),
            "predicted_by_row": dict(enumerate(predicted)),
            "actual_all_physical_safe": max(actual) <= 0.0,
        }

    def test_validation_margin_and_abstention_are_per_group(self) -> None:
        source = [
            self._record(
                "E0", "nominal", 0, 0.0,
                [-0.1, -0.2, 0.1, 0.0, -0.1, -0.3, -0.2],
                [-0.2, -0.3, -0.1, -0.2, -0.2, -0.4, -0.3],
            ),
            self._record(
                "E0", "repair", 1, 1.0,
                [-0.2, -0.2, -0.2, -0.1, -0.1, -0.3, -0.2],
                [-0.2, -0.2, -0.3, -0.2, -0.2, -0.3, -0.2],
            ),
        ]
        specialist = {
            ("E0", "nominal"): [-0.05, -0.05, -0.05],
            ("E0", "repair"): [-0.25, -0.20, -0.20],
        }
        records = enrich_records(source, specialist)
        margins = validation_margins(records, prediction="compact")
        self.assertAlmostEqual(margins["L5"], 0.2)
        result = evaluate_arm(
            records, name="margin", prediction="compact", margins=margins,
            selection="safe_or_abstain",
        )
        self.assertEqual(result["known_all_physical_safe_selection_count"], 1)
        self.assertEqual(result["states"][0]["selected_candidate"], "repair")
        self.assertEqual(result["per_group_selected_failure_count"]["L5"], 0)

    def test_unknown_selection_is_not_safe(self) -> None:
        record = self._record(
            "E0", "unknown", 0, 0.0,
            [-0.1] * 7, [-0.2] * 7,
        )
        record["known_outcome"] = False
        record["actual_by_row"] = {}
        record["actual_all_physical_safe"] = None
        enriched = enrich_records(
            [record], {("E0", "unknown"): [-0.2, -0.2, -0.2]},
        )
        result = evaluate_arm(
            enriched, name="zero", prediction="compact",
            margins={"end_effector": 0.0, "palm": 0.0, "L5": 0.0},
            selection="safe_or_abstain",
        )
        self.assertEqual(result["unknown_selection_count"], 1)
        self.assertEqual(result["known_all_physical_safe_selection_count"], 0)

    def test_prediction_tolerance(self) -> None:
        self.assertEqual(
            exact_float_lists_close([[1.0]], [[1.000001]], tolerance=1.0e-5)[0],
            True,
        )
        self.assertEqual(
            exact_float_lists_close([[1.0]], [[1.1]], tolerance=1.0e-5)[0],
            False,
        )


if __name__ == "__main__":
    unittest.main()
