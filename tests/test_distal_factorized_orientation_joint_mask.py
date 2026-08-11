from __future__ import annotations

from pathlib import Path
import unittest

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

from main.multilink_ellipsoid.factorized_orientation_joint_mask import (
    _mask_indexes, load_joint_mask_config,
)


class FactorizedOrientationJointMaskTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_joint_mask_config(Path(
            "configs/vlsa_distal_factorized_orientation_joint_mask_moka10.v1.json"
        ))

    def test_only_fixed_post_audit_masks_are_registered(self) -> None:
        masks = self.config["fixed_masks"]
        self.assertEqual(len(masks["joint_redundant_orientation"]), 2)
        self.assertEqual(
            self.config["evaluation"][
                "minimum_joint_mask_terminal_RMSE_reduction_fraction_both_models"
            ],
            0.9,
        )
        self.assertTrue(all(self.config["forbidden_actions"].values()))

    @unittest.skipIf(np is None, "numpy is unavailable")
    def test_variance_floor_mask_uses_raw_training_variance(self) -> None:
        indexes = _mask_indexes(
            ["__all_training_raw_std_below_1e-6__"],
            np.asarray(["a", "b", "c"], dtype=object),
            np.asarray([0.0, 1.0e-7, 2.0e-6]),
            1.0e-6,
        )
        self.assertEqual(indexes.tolist(), [0, 1])


if __name__ == "__main__":
    unittest.main()
