from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class LateFlowWaypointSearchTests(unittest.TestCase):
    def test_config_freezes_inference_only_search(self):
        from main.multilink_ellipsoid.late_flow_waypoint_search import load_config

        value = load_config(ROOT / "configs/vlsa_late_flow_waypoint_search.v1.json")
        self.assertEqual(value["waypoint_search"]["unique_candidate_count"], 505)
        self.assertEqual(value["scoring"]["physical_constraint_rows"], [1, 2, 3, 4])
        self.assertTrue(value["forbidden"]["simulator_candidate_rollout"])
        self.assertTrue(value["forbidden"]["action_execution"])

    def test_evaluator_does_not_execute_or_roll_out_waypoint_candidates(self):
        source = (
            ROOT / "scripts/evaluate_late_flow_waypoint_search.py"
        ).read_text()
        self.assertIn('"simulator_candidate_rollout_count": 0', source)
        self.assertIn('"action_executed": False', source)
        self.assertNotIn("env.step(effective", source)
        self.assertNotIn("monitor.execute(effective", source)

    def test_bezier_basis_preserves_endpoint(self):
        from main.multilink_ellipsoid.late_flow_waypoint_search import (
            bezier_action_basis,
        )

        basis = bezier_action_basis()
        self.assertEqual((len(basis), len(basis[0])), (5, 2))
        for column in range(2):
            self.assertAlmostEqual(sum(row[column] for row in basis), 0.0, places=12)

    def test_population_is_deterministic_smooth_and_bounded(self):
        try:
            import numpy as np
        except ModuleNotFoundError:
            self.skipTest("desktop Python lacks optional NumPy")
        from main.multilink_ellipsoid.late_flow_waypoint_search import (
            waypoint_batches,
        )

        kwargs = {
            "outward_world": [1.0, 2.0, 0.5],
            "seed": 2026081701,
            "trust_region_action": 0.25,
        }
        first = waypoint_batches(**kwargs)
        second = waypoint_batches(**kwargs)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 42)
        records = [row for batch in first for row in batch["records"]]
        self.assertEqual(len(records), 504)
        self.assertEqual([row["candidate_index"] for row in records], list(range(1, 505)))
        for batch in first:
            residuals = np.asarray(batch["residuals"], dtype=np.float64)
            self.assertEqual(residuals.shape, (13, 10, 3))
            self.assertTrue(np.array_equal(residuals[0], np.zeros((10, 3))))
            self.assertLessEqual(float(np.max(np.abs(residuals))), 0.25 + 1.0e-12)
            self.assertLessEqual(
                float(np.max(np.abs(np.sum(residuals[:, :5], axis=1)))),
                1.0e-12,
            )

    def test_selector_is_safety_first_then_minimum_change(self):
        try:
            import numpy as np  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("desktop Python lacks optional NumPy")
        from main.multilink_ellipsoid.late_flow_waypoint_search import (
            summarize_predictions,
        )

        records = []
        for index, correction, risk in (
            (1, 0.2, [0.4, -0.1, -0.2, -0.3, -0.4, -0.1, -0.1]),
            (2, 0.1, [0.3, 0.1, -0.2, -0.3, -0.4, -0.1, -0.1]),
        ):
            records.append({
                "candidate_index": index,
                "candidate_name": f"waypoint_{index:04d}",
                "predicted_risk_by_row": risk,
                "effective_correction_l2_action": correction,
                "effective_smoothness": 0.0,
            })
        summary = summarize_predictions(
            nominal_risk=[0.5, 0.2, -0.2, -0.3, -0.4, -0.1, -0.1],
            records=records,
            physical_rows=[1, 2, 3, 4], diagnostic_rows=[0], threshold=0.0,
        )
        self.assertEqual(
            summary["selected_physical_safe_minimum_change"]["candidate_index"], 1,
        )
        self.assertEqual(summary["predicted_proxy_inclusive_safe_count"], 0)


if __name__ == "__main__":
    unittest.main()
