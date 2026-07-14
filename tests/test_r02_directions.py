from __future__ import annotations

import copy
import importlib.util
import math
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "main/crfs_oracle/r02_directions.py"
SPEC = importlib.util.spec_from_file_location("r02_directions", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
r02_directions = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = r02_directions
SPEC.loader.exec_module(r02_directions)

R02DirectionError = r02_directions.R02DirectionError
analytic_d_opt_ascent_direction = r02_directions.analytic_d_opt_ascent_direction
deterministic_equal_l2_random_direction = (
    r02_directions.deterministic_equal_l2_random_direction
)
delta_star_model_from_witness = r02_directions.delta_star_model_from_witness
random_direction_from_witness = r02_directions.random_direction_from_witness
scale_only_model_conversion = r02_directions.scale_only_model_conversion
select_r01_changed_p_min_witness = (
    r02_directions.select_r01_changed_p_min_witness
)
sphere_obb_d_opt = r02_directions.sphere_obb_d_opt


def _l2(matrix) -> float:
    return math.sqrt(sum(float(value) ** 2 for row in matrix for value in row))


def _nominal() -> list[list[float]]:
    return [
        [0.10, -0.05, 0.02, 0.01 * row, -0.02 * row, 0.03 * row, 1.0]
        for row in range(5)
    ]


def _valid_attempt() -> dict:
    actions = copy.deepcopy(_nominal())
    offsets = (
        (0.05, 0.00, 0.00),
        (-0.04, 0.01, 0.00),
        (0.00, -0.02, 0.03),
        (0.02, 0.00, -0.01),
        (-0.01, 0.02, 0.00),
    )
    for row in range(5):
        for axis in range(3):
            actions[row][axis] += offsets[row][axis]
    return {
        "candidate_index": 7,
        "source": "raw_p_zero_candidate",
        "actions": actions,
        "simulator_safety_pass": True,
        "scene_stationary": True,
        "action_changed_from_nominal": True,
        "nontranslation_preserved": True,
        "direct_replay_exact": True,
        "verified_at_p_min": True,
        "changed_witness_at_p_min": True,
    }


def _artifact() -> dict:
    return {
        "schema_version": "1.0",
        "gate": "R01",
        "status": "verified_safe_progress",
        "case_id": "case-001",
        "provenance": {
            "executed_action_horizon": 5,
            "model_action_horizon": 10,
            "model_action_dimension": 32,
            "random_control_seed": 194,
            "translation_action_bounds": [-1.0, 1.0],
        },
        "calibration": {"p_min_m": 0.01},
        "nominal": {
            "actions": _nominal(),
            "collision_reproduced": True,
        },
        "verification": {
            # This deliberately bogus summary pointer must never be trusted.
            "p_min": {
                "selected": {"actions": [[99.0] * 7 for _ in range(5)]},
                "attempts": [],
            },
            "p_zero": {
                "selected": {"actions": [[-99.0] * 7 for _ in range(5)]},
                "attempts": [_valid_attempt()],
            },
        },
        "outcome": {"p_min_verified": True},
    }


def _identity_box() -> dict:
    return {
        "center_m": (0.10, 0.0, 0.0),
        "half_size_m": (0.01, 0.02, 0.02),
        "rotation_world": (
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
    }


class R01WitnessSelectionTest(unittest.TestCase):
    def test_recomputes_delta_star_from_raw_pooled_attempt(self) -> None:
        witness = select_r01_changed_p_min_witness(_artifact())

        self.assertEqual(witness.search, "p_zero")
        self.assertEqual(witness.raw_attempt_index, 0)
        self.assertEqual(witness.candidate_index, 7)
        self.assertEqual(len(witness.delta_star_physical), 10)
        self.assertTrue(all(len(row) == 32 for row in witness.delta_star_physical))
        expected = _valid_attempt()["actions"]
        nominal = _nominal()
        for row in range(5):
            for axis in range(3):
                self.assertAlmostEqual(
                    witness.delta_star_physical[row][axis],
                    expected[row][axis] - nominal[row][axis],
                )
            self.assertTrue(
                all(value == 0.0 for value in witness.delta_star_physical[row][3:])
            )
        for row in witness.delta_star_physical[5:]:
            self.assertTrue(all(value == 0.0 for value in row))
        self.assertGreater(witness.physical_l2_norm, 0.0)
        self.assertAlmostEqual(
            witness.physical_l2_norm, _l2(witness.delta_star_physical)
        )

    def test_recomputes_model_delta_with_scale_only(self) -> None:
        witness = select_r01_changed_p_min_witness(_artifact())
        scale = (0.5, 0.25, 2.0)
        delta_model = delta_star_model_from_witness(witness, action_scale=scale)
        for row in range(5):
            for axis in range(3):
                self.assertAlmostEqual(
                    delta_model[row][axis],
                    witness.delta_star_physical[row][axis] / scale[axis],
                )
            self.assertTrue(all(value == 0.0 for value in delta_model[row][3:]))
        self.assertTrue(all(value == 0.0 for row in delta_model[5:] for value in row))
        self.assertNotAlmostEqual(_l2(delta_model), witness.physical_l2_norm)

    def test_rejects_raw_claim_with_zero_delta(self) -> None:
        artifact = _artifact()
        artifact["verification"]["p_zero"]["attempts"][0]["actions"] = _nominal()
        with self.assertRaisesRegex(R02DirectionError, "zero norm"):
            select_r01_changed_p_min_witness(artifact)

    def test_rejects_raw_witness_outside_registered_bounds(self) -> None:
        artifact = _artifact()
        artifact["verification"]["p_zero"]["attempts"][0]["actions"][0][0] = 1.01
        with self.assertRaisesRegex(R02DirectionError, "outside registered bounds"):
            select_r01_changed_p_min_witness(artifact)

    def test_top_level_claim_cannot_replace_raw_evidence(self) -> None:
        artifact = _artifact()
        artifact["verification"]["p_zero"]["attempts"] = []
        with self.assertRaisesRegex(R02DirectionError, "raw attempts"):
            select_r01_changed_p_min_witness(artifact)


class RandomDirectionTest(unittest.TestCase):
    def test_registered_seed_is_deterministic_equal_l2_and_endpoint_free(self) -> None:
        witness = select_r01_changed_p_min_witness(_artifact())
        action_scale = (0.5, 0.25, 2.0)
        delta_star_model = delta_star_model_from_witness(
            witness, action_scale=action_scale
        )
        first = random_direction_from_witness(
            witness,
            base_physical_prefix=witness.nominal_prefix,
            action_scale=action_scale,
        )
        second = random_direction_from_witness(
            witness,
            base_physical_prefix=witness.nominal_prefix,
            action_scale=action_scale,
        )
        different_seed = deterministic_equal_l2_random_direction(
            delta_star_model,
            base_physical_prefix=witness.nominal_prefix,
            action_scale=action_scale,
            seed=witness.random_seed + 1,
            action_low=witness.action_low,
            action_high=witness.action_high,
        )

        self.assertEqual(first, second)
        self.assertNotEqual(first, different_seed)
        self.assertAlmostEqual(_l2(first), _l2(delta_star_model), places=12)
        for row_index, row in enumerate(first):
            if row_index >= 5:
                self.assertTrue(all(value == 0.0 for value in row))
            else:
                self.assertTrue(all(value == 0.0 for value in row[3:]))
                for axis in range(3):
                    candidate = (
                        witness.nominal_prefix[row_index][axis]
                        + action_scale[axis] * row[axis]
                    )
                    self.assertGreaterEqual(candidate, witness.action_low)
                    self.assertLessEqual(candidate, witness.action_high)
        endpoint_change = [sum(first[row][axis] for row in range(5)) for axis in range(3)]
        self.assertTrue(any(abs(value) > 1.0e-12 for value in endpoint_change))

    def test_rejects_zero_norm_and_bad_bounds(self) -> None:
        zero = [[0.0] * 32 for _ in range(10)]
        with self.assertRaisesRegex(R02DirectionError, "nonzero"):
            deterministic_equal_l2_random_direction(
                zero,
                base_physical_prefix=_nominal(),
                action_scale=(1.0, 1.0, 1.0),
                seed=1,
                action_low=-1.0,
                action_high=1.0,
            )
        witness = select_r01_changed_p_min_witness(_artifact())
        delta_star_model = delta_star_model_from_witness(
            witness, action_scale=(1.0, 1.0, 1.0)
        )
        with self.assertRaisesRegex(R02DirectionError, "low < high"):
            deterministic_equal_l2_random_direction(
                delta_star_model,
                base_physical_prefix=witness.nominal_prefix,
                action_scale=(1.0, 1.0, 1.0),
                seed=1,
                action_low=1.0,
                action_high=-1.0,
            )

    def test_intermediate_predicted_clean_bounds_are_not_rejected(self) -> None:
        witness = select_r01_changed_p_min_witness(_artifact())
        delta_star_model = delta_star_model_from_witness(
            witness, action_scale=(1.0, 1.0, 1.0)
        )
        outside_base = _nominal()
        outside_base[0][0] = 5.0
        direction = deterministic_equal_l2_random_direction(
            delta_star_model,
            base_physical_prefix=outside_base,
            action_scale=(1.0, 1.0, 1.0),
            seed=1,
            action_low=-1.0,
            action_high=1.0,
        )
        self.assertAlmostEqual(_l2(direction), _l2(delta_star_model), places=12)


class AnalyticDirectionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.nominal = [[0.5, 0.0, 0.0, 0.25, -0.5, 0.75, 1.0] for _ in range(5)]
        primary_box = _identity_box()
        primary_box["name"] = "primary"
        self.geometry = {
            "start_eef_center_m": (0.0, 0.0, 0.0),
            "response_matrix_m_per_action": (
                (0.01, 0.0, 0.0),
                (0.0, 0.01, 0.0),
                (0.0, 0.0, 0.01),
            ),
            "obstacle_boxes": [primary_box],
            "eef_radius_m": 0.005,
        }
        self.action_scale = tuple(
            value
            for row in range(5)
            for value in (float(row + 1), 1.0, 1.0)
        )

    def test_exact_subgradient_improves_d_opt_at_equal_model_l2(self) -> None:
        budget = 0.05
        result = analytic_d_opt_ascent_direction(
            self.nominal,
            action_scale=self.action_scale,
            model_l2_budget=budget,
            **self.geometry,
        )
        direction = result.direction_model
        candidate = copy.deepcopy(self.nominal)
        for row in range(5):
            for axis in range(3):
                candidate[row][axis] += (
                    self.action_scale[3 * row + axis] * direction[row][axis]
                )

        baseline_clearance = sphere_obb_d_opt(self.nominal, **self.geometry)
        candidate_clearance = sphere_obb_d_opt(candidate, **self.geometry)
        self.assertGreater(candidate_clearance, baseline_clearance)
        self.assertAlmostEqual(result.d_opt_m, baseline_clearance)
        self.assertAlmostEqual(_l2(direction), budget, places=12)
        self.assertTrue(all(value == 0.0 for row in direction[:5] for value in row[3:]))
        self.assertTrue(all(value == 0.0 for row in direction[5:] for value in row))
        self.assertTrue(all(direction[row][0] < 0.0 for row in range(5)))
        self.assertEqual(result.selected_segment_index, 4)
        self.assertEqual(result.selected_sample_index, 25)
        self.assertEqual(result.selected_alpha, 1.0)
        self.assertEqual(result.selected_box_name, "primary")
        self.assertEqual(result.local_sdf_gradient, (-1.0, 0.0, 0.0))
        self.assertEqual(result.world_sdf_gradient, (-1.0, 0.0, 0.0))
        for row in range(5):
            self.assertAlmostEqual(result.raw_physical_gradient[row][0], -0.01)
            self.assertAlmostEqual(
                result.raw_model_gradient[row][0],
                -0.01 * self.action_scale[3 * row],
            )
        self.assertEqual(
            result.to_dict()["selected_pair"]["sample_index"], 25
        )

    def test_deterministic_global_tie_uses_box_name(self) -> None:
        box_z = _identity_box()
        box_z["name"] = "z-box"
        box_a = _identity_box()
        box_a["name"] = "a-box"
        tied_geometry = copy.deepcopy(self.geometry)
        tied_geometry["obstacle_boxes"] = [box_z, box_a]
        result = analytic_d_opt_ascent_direction(
            self.nominal,
            action_scale=self.action_scale,
            model_l2_budget=0.05,
            **tied_geometry,
        )
        self.assertEqual(result.selected_box_name, "a-box")
        self.assertEqual(result.selected_box_index, 1)

    def test_inside_sdf_tie_uses_lowest_axis_and_positive_zero_sign(self) -> None:
        distance, local_gradient, world_gradient = (
            r02_directions._point_obb_signed_distance_and_gradient(
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                (
                    (1.0, 0.0, 0.0),
                    (0.0, 1.0, 0.0),
                    (0.0, 0.0, 1.0),
                ),
                (0.1, 0.1, 0.1),
            )
        )
        self.assertAlmostEqual(distance, -0.1)
        self.assertEqual(local_gradient, (1.0, 0.0, 0.0))
        self.assertEqual(world_gradient, (1.0, 0.0, 0.0))

    def test_rejects_zero_gradient_but_not_intermediate_action_bounds(self) -> None:
        zero_response = copy.deepcopy(self.geometry)
        zero_response["response_matrix_m_per_action"] = (
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
        )
        with self.assertRaisesRegex(R02DirectionError, "zero norm"):
            analytic_d_opt_ascent_direction(
                self.nominal,
                action_scale=self.action_scale,
                model_l2_budget=0.05,
                **zero_response,
            )
        result = analytic_d_opt_ascent_direction(
            self.nominal,
            action_scale=self.action_scale,
            model_l2_budget=4.0,
            **self.geometry,
        )
        self.assertAlmostEqual(_l2(result.direction_model), 4.0, places=12)
        physical_candidate = [
            [
                self.nominal[row][axis]
                + self.action_scale[3 * row + axis]
                * result.direction_model[row][axis]
                for axis in range(3)
            ]
            for row in range(5)
        ]
        self.assertTrue(
            any(abs(value) > 1.0 for row in physical_candidate for value in row)
        )

    def test_rejects_malformed_obstacle_evidence(self) -> None:
        malformed = copy.deepcopy(self.geometry)
        malformed["obstacle_boxes"] = [None]
        with self.assertRaisesRegex(R02DirectionError, "must be an object"):
            analytic_d_opt_ascent_direction(
                self.nominal,
                action_scale=self.action_scale,
                model_l2_budget=0.05,
                **malformed,
            )

    def test_frozen_sampling_and_scale_shape_fail_closed(self) -> None:
        with self.assertRaisesRegex(R02DirectionError, "26 samples"):
            analytic_d_opt_ascent_direction(
                self.nominal,
                action_scale=self.action_scale,
                model_l2_budget=0.05,
                samples_per_segment=25,
                **self.geometry,
            )
        with self.assertRaisesRegex(R02DirectionError, "3 or 15"):
            analytic_d_opt_ascent_direction(
                self.nominal,
                action_scale=(1.0, 1.0),
                model_l2_budget=0.05,
                **self.geometry,
            )


class ScaleOnlyConversionTest(unittest.TestCase):
    def test_divides_displacement_by_scale_without_centering(self) -> None:
        converted = scale_only_model_conversion(
            ((2.0, -4.0, 0.0), (1.0, 8.0, -10.0)),
            (2.0, 4.0, 5.0),
        )
        self.assertEqual(converted, ((1.0, -1.0, 0.0), (0.5, 2.0, -2.0)))

    def test_rejects_nonpositive_or_mismatched_scale(self) -> None:
        with self.assertRaisesRegex(R02DirectionError, "positive"):
            scale_only_model_conversion(((1.0, 2.0, 3.0),), (1.0, 0.0, 1.0))
        with self.assertRaisesRegex(R02DirectionError, "rows|shape"):
            scale_only_model_conversion(
                ((1.0, 2.0, 3.0), (4.0, 5.0, 6.0)),
                ((1.0, 1.0, 1.0),),
            )


if __name__ == "__main__":
    unittest.main()
