import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from main.multilink_ellipsoid.counterfactual_field import (
    fit_paired_field,
    load_counterfactual_field_config,
    matched_random_p_value,
    paired_chunk_directions,
    project_direction,
)
from main.multilink_ellipsoid.iterative_counterfactual_risk import (
    active_witness,
    feasible_correction,
    fit_safety_direction,
    load_iterative_risk_config,
)
from main.multilink_ellipsoid.fixed_step_counterfactual import (
    fixed_step_update,
    load_fixed_step_config,
)
from main.multilink_ellipsoid.multi_witness_counterfactual import (
    bounded_paired_directions,
    fit_clearance_rows,
    load_multi_witness_config,
    select_near_active_witnesses,
    smooth_min_direction,
)


ROOT = Path(__file__).resolve().parents[1]


class CounterfactualFieldTest(unittest.TestCase):
    def test_post_detour_live_gate_config(self):
        from main.multilink_ellipsoid.post_detour_live_gate import (
            load_post_detour_live_config,
        )

        config = load_post_detour_live_config(
            ROOT / "configs" / "vlsa_distal_post_detour_live_gate_e05.v1.json"
        )
        self.assertEqual(
            config["schema_version"],
            "vlsa_distal_post_detour_live_gate_e05_config.v1",
        )
        self.assertEqual(config["state_protocol"]["continuation_steps"], list(range(187, 192)))

    def test_persistent_route_basis_is_orthonormal_and_normed(self):
        from main.multilink_ellipsoid.post_detour_live_gate import (
            persistent_route_correction,
            persistent_route_directions,
        )

        routes = persistent_route_directions([1.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        self.assertTrue(np.allclose(routes["left"], -routes["right"]))
        self.assertAlmostEqual(float(np.dot(routes["retreat"], routes["up"])), 0.0)
        for direction in routes.values():
            self.assertAlmostEqual(np.linalg.norm(direction), 1.0)
        self.assertAlmostEqual(np.linalg.norm(persistent_route_correction(routes["up"], 1.5)), 1.5)

        from main.multilink_ellipsoid.post_detour_live_gate import load_post_detour_route_config

        config = load_post_detour_route_config(
            ROOT / "configs" / "vlsa_distal_post_detour_route_oracle_e05.v1.json"
        )
        self.assertEqual(config["route_search"]["modes"], ["left", "right", "up", "retreat"])

        from main.multilink_ellipsoid.post_detour_live_gate import load_receding_route_config

        receding = load_receding_route_config(
            ROOT / "configs" / "vlsa_distal_receding_route_oracle_e05.v2.json"
        )
        self.assertEqual(receding["state_protocol"]["execute_prefix_actions"], 5)

    def test_raw_multistart_gate0_config_and_transplant_contract(self):
        from main.multilink_ellipsoid.raw_multistart_gate0 import (
            load_raw_multistart_gate0_config,
            transplant_compound_prefix,
        )

        config = load_raw_multistart_gate0_config(
            ROOT / "configs" / "vlsa_distal_raw_multistart_gate0_e05.v1.json"
        )
        self.assertEqual(
            config["schema_version"], "vlsa_distal_raw_multistart_gate0_e05_config.v1"
        )
        raw = np.zeros((20, 7), dtype=np.float64)
        raw[:5, :3] = 0.9
        compound = raw.copy()
        compound[:5, 0] = 1.2
        result = transplant_compound_prefix(raw, compound, 1.0)
        self.assertTrue(np.array_equal(result["actions"][5:], raw[5:]))
        self.assertEqual(result["clipped_coordinate_count"], 5)
        self.assertTrue(np.allclose(result["actions"][:5, 0], 1.0))
        self.assertTrue(np.allclose(result["clipping_delta"][:, 0], -0.2))

        compound[:5, 0] = np.nextafter(0.3, 1.0)
        exact = transplant_compound_prefix(raw, compound, 1.0)
        self.assertTrue(np.array_equal(exact["actions"][:5], compound[:5]))

    def test_smooth_field_attribution_config_and_scale_contract(self):
        from main.multilink_ellipsoid.smooth_field_attribution import (
            load_smooth_field_attribution_config,
            scale_grid,
            select_smallest_verified_scale,
            smooth_min_value,
        )

        config = load_smooth_field_attribution_config(
            ROOT / "configs" / "vlsa_distal_smooth_field_attribution_e05.v1.json"
        )
        self.assertEqual(config["schema_version"], "vlsa_distal_smooth_field_attribution_e05.v1")
        self.assertEqual(len(scale_grid(0.025)), 41)
        self.assertLess(smooth_min_value([0.001, 0.003], 0.002), 0.001)
        records = [
            {"scale": 0.0, "verification_gate": False},
            {"scale": 0.5, "verification_gate": True},
            {"scale": 1.0, "verification_gate": True},
        ]
        self.assertEqual(select_smallest_verified_scale(records)["scale"], 0.5)

    def test_registered_config_loads(self):
        value = load_counterfactual_field_config(
            ROOT / "configs/vlsa_distal_counterfactual_field_e05.v1.json"
        )
        self.assertEqual(value["sampling"]["paired_direction_count"], 64)
        iterative = load_iterative_risk_config(
            ROOT / "configs/vlsa_distal_iterative_counterfactual_risk_e05.v1.json"
        )
        self.assertEqual(iterative["action_space"]["total_trust_radii_action"][-1], 1.0)
        fixed = load_fixed_step_config(
            ROOT / "configs/vlsa_distal_fixed_step_counterfactual_field_e05.v1.json"
        )
        self.assertEqual(fixed["field_estimation"]["fixed_normalized_step_action"], 0.1)
        multi = load_multi_witness_config(
            ROOT / "configs/vlsa_distal_multi_witness_counterfactual_e05.v1.json"
        )
        self.assertEqual(multi["multi_witness"]["maximum_witness_count"], 8)

    def test_config_rejects_protocol_drift(self):
        path = ROOT / "configs/vlsa_distal_counterfactual_field_e05.v1.json"
        value = json.loads(path.read_text())
        value["sampling"]["paired_direction_count"] = 63
        with tempfile.TemporaryDirectory() as temporary:
            changed = Path(temporary) / "changed.json"
            changed.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "pair count"):
                load_counterfactual_field_config(changed)

    def test_directions_are_paired_feasible_and_endpoint_preserving(self):
        nominal = np.asarray(
            [[0.9, 0.1, -0.2], [0.2, -0.1, 0.3], [0.0, 0.2, -0.1], [-0.2, 0.0, 0.1], [-0.1, -0.2, -0.1]],
            dtype=np.float64,
        )
        directions = paired_chunk_directions(64, 5, nominal, 0.1, 1.0)
        self.assertEqual(directions.shape, (64, 15))
        self.assertTrue(np.allclose(np.linalg.norm(directions, axis=1), 1.0))
        self.assertTrue(np.allclose(np.sum(directions.reshape(64, 5, 3), axis=1), 0.0, atol=1e-12))
        self.assertLessEqual(np.max(np.abs(nominal.reshape(1, 15) + 0.1 * directions)), 1.0)
        self.assertLessEqual(np.max(np.abs(nominal.reshape(1, 15) - 0.1 * directions)), 1.0)

    def test_fit_recovers_linear_field_and_holds_out(self):
        rng = np.random.RandomState(8)
        nominal = np.zeros((5, 3), dtype=np.float64)
        directions = paired_chunk_directions(64, 9, nominal, 0.1, 1.0)
        truth = rng.normal(size=15)
        truth -= truth.reshape(5, 3).mean(axis=0).reshape(1, 3).repeat(5, axis=0).reshape(15)
        epsilon = 0.1
        derivative = directions.dot(truth)
        result = fit_paired_field(
            directions,
            epsilon * derivative,
            -epsilon * derivative,
            epsilon,
            48,
            1e-9,
        )
        self.assertGreater(result["heldout_pearson"], 0.999)
        self.assertGreater(result["heldout_sign_accuracy"], 0.99)
        self.assertLess(result["heldout_rmse"], 1e-6)

    def test_soft_endpoint_directions_are_not_forced_to_sum_zero(self):
        nominal = np.zeros((5, 3), dtype=np.float64)
        directions = paired_chunk_directions(
            16, 11, nominal, 0.1, 1.0, preserve_endpoint=False
        )
        sums = np.sum(directions.reshape(16, 5, 3), axis=1)
        self.assertGreater(float(np.max(np.abs(sums))), 0.1)

    def test_comparator_projection_and_random_p_value(self):
        nominal = np.zeros((5, 3), dtype=np.float64)
        basis = paired_chunk_directions(32, 7, nominal, 0.1, 1.0)
        projected = project_direction(np.arange(15, dtype=np.float64), basis)
        self.assertAlmostEqual(float(np.linalg.norm(projected)), 1.0)
        self.assertTrue(np.allclose(np.sum(projected.reshape(5, 3), axis=0), 0.0, atol=1e-10))
        self.assertAlmostEqual(matched_random_p_value(2.0, [0.0, 1.0, 3.0]), 0.5)

    def test_pure_risk_fit_witness_and_feasibility(self):
        nominal = np.zeros((5, 3), dtype=np.float64)
        directions = paired_chunk_directions(32, 17, nominal, 0.05, 1.0)
        truth = np.arange(15, dtype=np.float64)
        truth -= np.tile(truth.reshape(5, 3).mean(axis=0), 5)
        targets = directions.dot(truth)
        fit = fit_safety_direction(
            directions, 0.05 * targets, -0.05 * targets, 0.05, 1e-9
        )
        self.assertLess(fit["fit_rmse"], 1e-6)
        trace = np.ones((20, 7), dtype=np.float64)
        trace[15, 2] = -0.01
        self.assertEqual(active_witness(trace)["action_offset"], 15)
        self.assertEqual(active_witness(trace)["ellipsoid_row"], 2)
        correction = directions[0].reshape(5, 3) * 0.1
        self.assertTrue(
            feasible_correction(
                nominal, correction, radius=0.1, action_limit=1.0, preserve_endpoint=True
            )
        )

    def test_fixed_step_is_full_norm_and_never_silently_clips(self):
        direction = np.zeros(15, dtype=np.float64)
        direction[0] = 1.0
        first = fixed_step_update(np.zeros(15), direction, step=0.1, maximum_norm=1.0)
        self.assertAlmostEqual(float(np.linalg.norm(first)), 0.1)
        with self.assertRaisesRegex(ValueError, "budget exceeded"):
            fixed_step_update(0.95 * direction, direction, step=0.1, maximum_norm=1.0)
        with self.assertRaisesRegex(ValueError, "unit norm"):
            fixed_step_update(np.zeros(15), 2.0 * direction, step=0.1, maximum_norm=1.0)

    def test_multi_witness_rows_remain_separate_and_fit_gradients(self):
        trace = np.ones((20, 7), dtype=np.float64)
        trace[19, 1] = -0.014
        trace[15, 1] = -0.0138
        trace[14, 3] = -0.0135
        witnesses = select_near_active_witnesses(trace, maximum_count=8, threshold_m=0.005)
        self.assertEqual(witnesses[0]["action_offset"], 19)
        self.assertEqual(witnesses[0]["ellipsoid_row"], 1)
        self.assertGreaterEqual(len(witnesses), 3)

        rng = np.random.RandomState(31)
        directions = rng.normal(size=(32, 15))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        truth = rng.normal(size=(20 * 7, 15))
        target = directions.dot(truth.T)
        plus = (0.05 * target).reshape(32, 20, 7)
        minus = (-0.05 * target).reshape(32, 20, 7)
        fit = fit_clearance_rows(directions, plus, minus, 0.05, 1e-9)
        prediction = directions.dot(fit["gradients"].T)
        self.assertLess(float(np.sqrt(np.mean((prediction - target) ** 2))), 1e-5)

        smooth = smooth_min_direction(trace, fit["gradients"], 0.002)
        self.assertAlmostEqual(float(np.sum(smooth["weights"])), 1.0)
        self.assertGreater(float(smooth["weights"][19 * 7 + 1]), 0.1)

    def test_multi_witness_pairs_freeze_coordinates_without_bidirectional_headroom(self):
        nominal = np.zeros((5, 3), dtype=np.float64)
        nominal[0, 0] = 0.98
        nominal[2, 1] = -0.97
        directions = bounded_paired_directions(32, 19, nominal, 0.05, 1.0)
        self.assertTrue(np.allclose(np.linalg.norm(directions, axis=1), 1.0))
        self.assertTrue(np.allclose(directions[:, 0], 0.0))
        self.assertTrue(np.allclose(directions[:, 7], 0.0))
        self.assertLessEqual(
            float(np.max(np.abs(nominal.reshape(1, 15) + 0.05 * directions))), 1.0
        )
        self.assertLessEqual(
            float(np.max(np.abs(nominal.reshape(1, 15) - 0.05 * directions))), 1.0
        )


if __name__ == "__main__":
    unittest.main()
