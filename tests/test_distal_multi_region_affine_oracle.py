import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np

from main.multilink_ellipsoid.multi_region_affine_oracle import (
    certificate_at_nominal, containing_region_indexes, fixed_regions,
    fit_region_target, load_config, region_affine_values,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "vlsa_distal_multi_region_affine_oracle_moka10.v1.json"


class MultiRegionAffineOracleTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(CONFIG)
        axis = np.linspace(-1.0, 1.0, 5)
        self.xyz = np.asarray([
            [x, y, z] for x in axis for y in axis for z in axis
        ])
        self.regions = fixed_regions(
            self.xyz, [-1.0] * 3, [1.0] * 3, self.config["partition"]
        )

    def test_partition_is_fixed_overlapping_27_boxes(self):
        self.assertEqual(len(self.regions), 27)
        self.assertTrue(all(len(r["fit_candidate_indexes"]) == 27 for r in self.regions))
        self.assertEqual(len(containing_region_indexes(
            [0.0, 0.0, 0.0], self.regions, 1e-12
        )), 27)
        self.assertEqual(len(containing_region_indexes(
            [-1.0, -1.0, -1.0], self.regions, 1e-12
        )), 1)

    @unittest.skipUnless(importlib.util.find_spec("scipy"), "scipy unavailable")
    def test_local_affine_lower_bound_and_clearance_shift(self):
        margins = np.column_stack([
            0.004 + 0.01 * self.xyz[:, 0] - 0.002 * self.xyz[:, 1]
            for _ in range(7)
        ])
        region = self.regions[13]
        target = fit_region_target(
            self.xyz, margins, region, self.config["ridge_huber"]
        )
        indexes = region["fit_candidate_indexes"]
        values = np.asarray([region_affine_values(target, self.xyz[i]) for i in indexes])
        self.assertLessEqual(float(np.max(values - margins[indexes])), 1e-10)
        certificate = certificate_at_nominal(target, [0.1, 0.0, 0.0], 0.001)
        direct = region_affine_values(target, [0.1, 0.0, 0.0]) - 0.001
        self.assertTrue(np.allclose(certificate["intercept_at_nominal_m"], direct))

    def test_unknown_config_key_is_rejected(self):
        raw = json.loads(CONFIG.read_text())
        raw["unknown"] = True
        path = ROOT / "tests" / ".multi-region-invalid.json"
        try:
            path.write_text(json.dumps(raw))
            with self.assertRaises(ValueError):
                load_config(path)
        finally:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
