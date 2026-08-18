import json
from pathlib import Path
import unittest


class TightPrefixSingleOracleGradientTests(unittest.TestCase):
    def setUp(self):
        self.repo = Path(__file__).resolve().parents[1]

    def test_registered_config(self):
        from main.multilink_ellipsoid.tight_prefix_single_oracle_gradient import (
            load_config,
        )

        value = load_config(
            self.repo / "configs/vlsa_tight_prefix_single_oracle_gradient.v1.json"
        )
        self.assertEqual(value["case"]["case_id"], "vlsa-t1-spatial-i-t3-e15")
        self.assertEqual(value["support"]["correction_radius_l2"], 0.25)
        self.assertIn("regenerate_terminal_actions_in_replay", value["forbidden"])

    def test_oracle_line_uses_registered_norms(self):
        import numpy as np
        from main.multilink_ellipsoid.tight_prefix_single_oracle_gradient import (
            oracle_line_residuals,
        )

        names, values = oracle_line_residuals(
            [1.0] + [0.0] * 14, [0.0625, 0.125, 0.1875, 0.25]
        )
        self.assertEqual(names[0], "oracle_down_r0")
        self.assertEqual(names[-1], "oracle_down_r3")
        self.assertEqual(
            [float(np.linalg.norm(np.asarray(row)[:5])) for row in values],
            [0.0625, 0.125, 0.1875, 0.25],
        )

    def test_comparison_controls_are_equal_norm(self):
        import numpy as np
        from main.multilink_ellipsoid.tight_prefix_single_oracle_gradient import (
            comparison_residuals,
        )

        names, values, audit = comparison_residuals(
            [1.0] + [0.0] * 14, [0.0, 1.0] + [0.0] * 13,
            radius=0.125, random_seed=20260819,
        )
        self.assertEqual(names[0], "comparison_anchor")
        self.assertLessEqual(audit["maximum_requested_norm_error"], 1e-12)
        self.assertTrue(all(
            abs(float(np.linalg.norm(np.asarray(row)[:5])) - 0.125) <= 1e-12
            for row in values[1:]
        ))

    def test_metrics_keep_oracle_and_learned_separate(self):
        from main.multilink_ellipsoid.tight_prefix_single_oracle_gradient import (
            comparison_metrics,
        )

        risks = {
            "comparison_anchor": 0.1, "learned_down": 0.02,
            "oracle_down": -0.03, "oracle_up": 0.2,
            "random_0": 0.04, "random_1": -0.01,
            "random_2": 0.12, "random_3": 0.08,
        }
        records = [
            {
                "name": name,
                "hard_primary_future_risk": risk,
                "physical_primary_safe": risk <= 0.0,
            }
            for name, risk in risks.items()
        ]
        metrics = comparison_metrics(records)
        self.assertTrue(metrics["oracle_safe"])
        self.assertFalse(metrics["learned_safe"])
        self.assertEqual(metrics["random_safe_count"], 1)
        self.assertEqual(metrics["oracle_random_advantage_rate"], 1.0)

    def test_terminal_probes_pack_into_frozen_thirteen_branch_calls(self):
        import numpy as np
        from main.multilink_ellipsoid.tight_prefix_single_oracle_gradient import (
            terminal_branch_batches,
        )

        names = [f"probe_{index}" for index in range(31)]
        values = np.zeros((31, 10, 3), dtype=np.float64)
        batches = terminal_branch_batches(names, values)
        self.assertEqual([len(batch["request_names"]) for batch in batches], [13] * 3)
        self.assertEqual([len(batch["scientific_names"]) for batch in batches], [12, 12, 7])
        self.assertTrue(all(batch["request_names"][0] == "nominal" for batch in batches))
        self.assertEqual(
            [name for batch in batches for name in batch["scientific_names"]], names,
        )

    def test_config_is_json(self):
        path = self.repo / "configs/vlsa_tight_prefix_single_oracle_gradient.v1.json"
        self.assertIsInstance(json.loads(path.read_text()), dict)


if __name__ == "__main__":
    unittest.main()
