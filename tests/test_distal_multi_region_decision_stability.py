import importlib.util
from pathlib import Path
import unittest

import numpy as np

from main.multilink_ellipsoid.multi_region_affine_oracle import fixed_regions
from main.multilink_ellipsoid.multi_region_decision_stability import (
    fit_resampled_region_target, jaccard, load_config, shared_region_resamples,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "vlsa_distal_multi_region_decision_stability_moka10.v1.json"
PARTITION_CONFIG = ROOT / "configs" / "vlsa_distal_multi_region_affine_oracle_moka10.v1.json"


class MultiRegionDecisionStabilityTests(unittest.TestCase):
    def test_config_and_shared_resamples_are_frozen(self):
        config = load_config(CONFIG)
        indexes = list(range(27))
        first = shared_region_resamples(
            indexes, replicate_count=16, selected_count=22, seed=17
        )
        second = shared_region_resamples(
            indexes, replicate_count=16, selected_count=22, seed=17
        )
        self.assertEqual(first, second)
        self.assertTrue(all(len(item) == 22 for item in first))
        self.assertEqual(config["decision_gate"]["expected_selected_QP_rollout_count"], 800)

    def test_jaccard(self):
        self.assertEqual(jaccard([False, False], [False, False]), 1.0)
        self.assertEqual(jaccard([True, False], [True, True]), 0.5)

    @unittest.skipUnless(importlib.util.find_spec("scipy"), "scipy unavailable")
    def test_refit_recalibrates_on_full_region(self):
        import json
        config = load_config(CONFIG)
        partition = json.loads(PARTITION_CONFIG.read_text())["partition"]
        axis = np.linspace(-1.0, 1.0, 5)
        xyz = np.asarray([[x, y, z] for x in axis for y in axis for z in axis])
        margins = np.column_stack([
            0.003 + 0.01 * xyz[:, 0] + 0.001 * xyz[:, 1] ** 2
            for _ in range(7)
        ])
        region = fixed_regions(xyz, [-1] * 3, [1] * 3, partition)[13]
        selected = region["fit_candidate_indexes"][:22]
        target = fit_resampled_region_target(
            xyz, margins, region, selected, config["ridge_huber"]
        )
        anchor = np.asarray(target["anchor_xyz"])
        gradient = np.asarray(target["gradient_m_per_action"])
        lower = (
            np.asarray(target["anchor_margin_m"])
            - np.asarray(target["one_sided_error_m"])
            + (xyz[region["fit_candidate_indexes"]] - anchor) @ gradient.T
        )
        self.assertLessEqual(float(np.max(
            lower - margins[region["fit_candidate_indexes"]]
        )), 1e-10)


if __name__ == "__main__":
    unittest.main()
