from __future__ import annotations

import itertools
import json
from pathlib import Path
import unittest
from unittest import mock


try:
    import numpy as np
except ImportError:  # pragma: no cover - local optional dependency
    np = None


@unittest.skipIf(np is None, "numpy is unavailable")
class RegionAwareMlpTest(unittest.TestCase):
    def setUp(self) -> None:
        from main.multilink_ellipsoid.two_step_margin import PAIR_FEATURE_NAMES

        self.pair = np.zeros((7, len(PAIR_FEATURE_NAMES)), dtype=np.float64)
        self.axes = [np.linspace(-0.5, 0.5, 5) for _ in range(3)]
        self.actions = np.asarray(
            list(itertools.product(*self.axes)), dtype=np.float64
        )
        self.partition = {
            "coordinates": "per_state_action_box_normalized_minus1_plus1",
            "axis_intervals": [[-1.0, 0.0], [-0.5, 0.5], [0.0, 1.0]],
            "axis_centers": [-0.5, 0.0, 0.5],
            "region_count": 27,
            "fit_actions_per_region": 27,
            "action_limit": 1.0,
            "trust_region_linf_action": 0.5,
            "grid_points_per_dimension": 5,
            "inclusive_membership_tolerance": 1.0e-12,
        }

    def _regions(self):
        from main.multilink_ellipsoid.multi_region_affine_oracle import fixed_regions

        return fixed_regions(
            self.actions, [-0.5] * 3, [0.5] * 3, self.partition
        )

    def test_registered_config(self) -> None:
        from main.multilink_ellipsoid.region_aware_mlp import load_config

        root = Path(__file__).resolve().parents[1]
        config = load_config(
            root / "configs/vlsa_distal_region_aware_mlp_moka10.v1.json"
        )
        self.assertEqual(config["features"]["input_dimension"], 62)
        self.assertEqual(config["split"]["expected_state_counts"]["test"], 15)

    def test_region_features_bind_anchor_and_descriptor(self) -> None:
        from main.multilink_ellipsoid.region_aware_mlp import (
            REGION_FEATURE_NAMES, region_feature_matrix,
        )
        from main.multilink_ellipsoid.two_step_margin import PAIR_CANDIDATE_SLICE

        region = self._regions()[0]
        features = region_feature_matrix(self.pair, region)
        self.assertEqual(features.shape, (7, len(REGION_FEATURE_NAMES)))
        np.testing.assert_allclose(
            features[:, PAIR_CANDIDATE_SLICE],
            np.repeat(np.asarray(region["anchor_xyz"])[None, :], 7, axis=0),
        )
        np.testing.assert_allclose(features[0, -3:], [-0.5, -0.5, -0.5])

    def test_training_arrays_preserve_state_region_constraint_rows(self) -> None:
        from main.multilink_ellipsoid.region_aware_mlp import regional_training_arrays

        regions = self._regions()
        margins = np.full((125, 7), 0.02, dtype=np.float64)
        targets = []
        for region in regions:
            targets.append({
                "region_index": region["region_index"],
                "anchor_xyz": region["anchor_xyz"],
                "anchor_margin_m": [0.02] * 7,
                "one_sided_error_m": [0.001] * 7,
                "gradient_m_per_action": [[0.001, 0.002, 0.003]] * 7,
            })
        state = {
            "state_index": 0, "case_id": "case", "state_step": 10,
            "split": "train", "candidate_first_xyz": self.actions.tolist(),
            "candidate_minimum_distal_margin_m": margins.tolist(),
            "pair_state_feature_vectors": self.pair.tolist(),
        }
        oracle = {
            "state_index": 0, "case_id": "case", "state_step": 10,
            "regions": regions, "regional_targets": targets,
        }
        arrays = regional_training_arrays([state], [oracle])
        self.assertEqual(arrays["features"].shape, (189, 62))
        self.assertEqual(arrays["deltas"].shape, (189, 27, 3))
        np.testing.assert_allclose(arrays["targets_m"][0], [0.019, 0.001, 0.002, 0.003])

    def test_qp_selection_uses_no_simulator_outcome(self) -> None:
        from main.multilink_ellipsoid.region_aware_mlp import solve_regional_qps

        regions = self._regions()
        targets = [{
            "region_index": region["region_index"],
            "anchor_xyz": region["anchor_xyz"],
            "lower_anchor_m": [0.01] * 7,
            "gradient_m_per_action": [[0.0, 0.0, 0.0]] * 7,
            "ensemble_guard_m": [0.0] * 7,
            "calibration_m": [0.0] * 7,
        } for region in regions]
        def fake_solver(nominal, lower, upper, certificate, projection):
            candidate = np.clip(np.asarray(nominal), lower, upper)
            return {
                "valid": True, "reason": "solved",
                "candidate_xyz": candidate.tolist(),
                "correction_l2": float(np.linalg.norm(candidate - nominal)),
                "diagnostics": {"input_constraint_count": 7},
            }

        with mock.patch(
            "main.multilink_ellipsoid.region_aware_mlp.solve_affine_certificate_qp",
            side_effect=fake_solver,
        ):
            result = solve_regional_qps(
                [0.0, 0.0, 0.0], regions, targets,
                {
                    "eps_abs": 1.0e-7, "eps_rel": 1.0e-7,
                    "max_iter": 10000, "residual_tolerance": 1.0e-6,
                    "bound_tolerance_action": 5.0e-8,
                },
            )
        self.assertEqual(len(result["regional_proposals"]), 27)
        self.assertIsNotNone(result["selected"])
        np.testing.assert_allclose(result["selected"]["candidate_xyz"], [0.0] * 3)

    def test_jaccard(self) -> None:
        from main.multilink_ellipsoid.region_aware_mlp import jaccard

        self.assertEqual(jaccard([False, False], [False, False]), 1.0)
        self.assertAlmostEqual(jaccard([True, True, False], [True, False, True]), 1 / 3)

    def test_closed_loop_exact_rollout_is_measurement_only(self) -> None:
        from main.multilink_ellipsoid.region_aware_closed_loop import (
            RegionAwareRecedingFilter,
        )

        class Probe:
            def rollout_chunk(self, env, actions):
                transition = {
                    "minimum_substep_clearance_m": [-0.001] + [0.01] * 7,
                    "raw_protected_contact_count": 1,
                    "maximum_within_step_obstacle_l1_displacement_m": 0.0,
                    "next_state_sha256": "hash",
                }
                return {"transitions": [transition, transition]}

        controller = RegionAwareRecedingFilter(
            models=[], model_state={},
            config={
                "partition": {}, "uncertainty": {}, "projection": {
                    "bound_tolerance_action": 5.0e-8,
                },
                "closed_loop": {
                    "activation_current_distal_clearance_m": 0.05,
                    "maximum_per_step_obstacle_l1_displacement_m": 1.0e-4,
                },
            },
            probe=Probe(),
        )
        chosen = {
            "valid": True, "candidate_xyz": [0.1, 0.2, 0.3],
            "correction_l2": 0.1, "region_index": 4,
        }
        with mock.patch(
            "main.multilink_ellipsoid.region_aware_closed_loop.live_regions",
            return_value=(None, None, [{}] * 27),
        ), mock.patch(
            "main.multilink_ellipsoid.region_aware_closed_loop.guarded_targets",
            return_value=[{}] * 27,
        ), mock.patch(
            "main.multilink_ellipsoid.region_aware_closed_loop.solve_regional_qps",
            return_value={"regional_proposals": [], "selected": chosen},
        ):
            action, record = controller.propose(
                object(), [0.0] * 7, [0.0] * 7, np.zeros((7, 53)),
                [0.0] * 7, step=9
            )
        np.testing.assert_allclose(action[:3], [0.1, 0.2, 0.3])
        self.assertFalse(record["exact_measurement"]["true_distal_safe"])
        self.assertTrue(record["exact_measurement"]["measurement_only"])


if __name__ == "__main__":
    unittest.main()
