from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "main/crfs_oracle/endpoint_free_projection.py"
SPEC = importlib.util.spec_from_file_location("endpoint_free_projection", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
endpoint_free_projection = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = endpoint_free_projection
SPEC.loader.exec_module(endpoint_free_projection)
solve_endpoint_free_projection = endpoint_free_projection.solve_endpoint_free_projection


def _box(center=(1.0, 0.0, 0.0), half_size=(0.2, 0.2, 0.2)) -> dict:
    return {
        "center_m": center,
        "rotation_world": (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        "half_size_m": half_size,
    }


def _nominal() -> list[list[float]]:
    return [[0.2, 0.0, 0.0, 0.01 * row, -0.02 * row, 0.03 * row, -1.0] for row in range(5)]


def _search(**overrides):
    parameters = {
        "nominal_prefix": _nominal(),
        "start_eef_center_m": (0.0, 0.0, 0.0),
        "response_matrix": ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        "obstacle_boxes": [_box()],
        "eef_radius_m": 0.05,
        "progress_fn": lambda points: points[-1][1] - points[0][1],
        "minimum_progress": 0.5,
        "optimizer_clearance_margin_m": 0.05,
        "population_size": 8,
        "generations": 2,
        "restarts": 2,
        "template_amplitude": 0.2,
        "max_candidates": 4,
        "samples_per_segment": 8,
        "seed": 91,
    }
    parameters.update(overrides)
    return solve_endpoint_free_projection(**parameters)


class EndpointFreeProjectionTest(unittest.TestCase):
    def test_runtime_type_alias_supports_python38_allocation(self) -> None:
        source = MODULE_PATH.read_text(encoding="utf-8")
        self.assertIn(
            "ProgressFn = Callable[[Tuple[Tuple[float, float, float], ...]], float]",
            source,
        )
        self.assertNotIn("ProgressFn = Callable[[tuple[", source)

    def test_candidate_changes_endpoint_and_preserves_other_channels(self) -> None:
        nominal = _nominal()
        result = _search(nominal_prefix=nominal)
        self.assertEqual(result.outcome, "candidate_found")
        candidate = result.candidates[0]
        self.assertTrue(candidate.clearance_pass)
        self.assertTrue(candidate.progress_pass)
        self.assertGreaterEqual(candidate.d_opt_m, 0.05)
        self.assertGreaterEqual(candidate.progress_opt, 0.5)
        for row, action in enumerate(candidate.actions):
            self.assertEqual(action[3:], nominal[row][3:])
        correction_sum = [
            sum(candidate.actions[row][axis] - nominal[row][axis] for row in range(5))
            for axis in range(3)
        ]
        self.assertGreater(max(abs(value) for value in correction_sum), 0.5)

    def test_search_is_exactly_deterministic_for_fixed_seed(self) -> None:
        first = _search().to_dict()
        second = _search().to_dict()
        self.assertEqual(first, second)
        self.assertEqual(first["generated_samples"], 32)

    def test_not_found_is_bounded_and_not_a_certificate(self) -> None:
        result = _search(
            obstacle_boxes=[_box(center=(0.0, 0.0, 0.0), half_size=(0.2, 0.2, 0.2))],
            minimum_progress=10.0,
            action_low=-0.2,
            action_high=0.2,
        )
        self.assertEqual(result.outcome, "not_found_within_budget")
        self.assertFalse(result.candidate_found)
        self.assertEqual(result.candidates, [])
        serialized = json.dumps(result.to_dict(), sort_keys=True).lower()
        self.assertNotIn("infeasible", serialized)
        self.assertIn("not a geometric certificate", result.message)

    def test_clearance_and_progress_passes_are_reported_separately(self) -> None:
        result = _search(
            obstacle_boxes=[_box(center=(5.0, 0.0, 0.0))],
            minimum_progress=10.0,
            action_low=-0.2,
            action_high=0.2,
        )
        self.assertEqual(result.outcome, "not_found_within_budget")
        self.assertTrue(result.best_attempt.clearance_pass)
        self.assertFalse(result.best_attempt.progress_pass)

    def test_budget_and_structured_controls_are_explicit(self) -> None:
        result = _search(action_low=(-1.0, -1.0, -1.0), action_high=(1.0, 1.0, 1.0))
        sources = {control.source for control in result.controls}
        self.assertIn("nominal", sources)
        self.assertIn("stationary", sources)
        self.assertIn("axis_shift_y_positive", sources)
        self.assertIn("front_detour_z_negative", sources)
        self.assertEqual(result.evaluation_budget, len(result.controls) + 32)
        self.assertLessEqual(result.evaluations, result.evaluation_budget)
        self.assertGreaterEqual(
            result.maximum_clearance_attempt.d_opt_m,
            max(item.d_opt_m for item in result.controls),
        )
        self.assertGreaterEqual(
            result.maximum_progress_attempt.progress_opt,
            max(item.progress_opt for item in result.controls),
        )


if __name__ == "__main__":
    unittest.main()
