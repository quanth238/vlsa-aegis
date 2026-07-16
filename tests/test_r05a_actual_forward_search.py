from __future__ import annotations

import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
IMPORT_ERROR: Exception | None = None
try:
    import numpy as np

    sys.path.insert(0, str(ROOT / "main"))
    from crfs_oracle.r05a_actual_forward_search import (
        ActualForwardEvaluation,
        DT_FLOAT32,
        project_product_ball,
        run_actual_forward_cem,
        transport_increment,
        validate_executed_constraints,
    )
except (ImportError, ModuleNotFoundError) as exc:
    IMPORT_ERROR = exc


@unittest.skipIf(IMPORT_ERROR is not None, f"actual-forward CEM requires NumPy: {IMPORT_ERROR}")
class ActualForwardSearchTest(unittest.TestCase):
    def test_projection_and_transport_are_exact(self) -> None:
        raw = np.zeros((5, 15), dtype=np.float64)
        raw[0, 0] = 2.0
        projected = project_product_ball(raw, 1.0)
        self.assertEqual(projected.dtype, np.float64)
        self.assertEqual(projected[0, 0], 1.0)
        self.assertEqual(projected[1].tobytes(), np.zeros(15, dtype=np.float64).tobytes())
        c32, u32, executed = transport_increment(projected)
        self.assertEqual(c32.dtype, np.float32)
        self.assertEqual(u32.dtype, np.float32)
        np.testing.assert_array_equal(executed, (DT_FLOAT32 * u32).astype(np.float32))

    def test_constraint_rule_accepts_equal_split(self) -> None:
        budget = np.float32(3.6398398876190186)
        radius = np.float32(budget / np.float32(5.0))
        value = np.zeros((5, 15), dtype=np.float32)
        value[:, 0] = radius
        per_step, path = validate_executed_constraints(value, budget)
        self.assertEqual(per_step.dtype, np.float32)
        self.assertEqual(path.dtype, np.float32)

    def test_frozen_cem_uses_exact_520_callbacks_and_persists_state(self) -> None:
        budget = np.float32(1.0)
        arm_velocity = np.zeros((5, 15), dtype=np.float32)
        arm_executed = (DT_FLOAT32 * arm_velocity).astype(np.float32)

        def make_eval(executed: np.ndarray, error_value: float) -> ActualForwardEvaluation:
            error = np.full((5, 7), error_value, dtype=np.float32)
            return ActualForwardEvaluation(
                executed_increment=executed.copy(),
                normalized_final=np.zeros((10, 32), dtype=np.float32),
                returned_physical_action=np.zeros((10, 7), dtype=np.float64),
                physical_error=error,
            )

        arm_eval = make_eval(arm_executed, 1.0)
        calls: list[np.ndarray] = []

        def evaluate(velocity: np.ndarray) -> ActualForwardEvaluation:
            calls.append(velocity.copy())
            executed = (DT_FLOAT32 * velocity).astype(np.float32)
            # Every search point improves over the deliberately bad Arm A;
            # the core still must execute all frozen requests.
            error_value = float(np.linalg.norm(executed.astype(np.float64))) * 1.0e-3
            return make_eval(executed, error_value)

        result = run_actual_forward_cem(
            evaluate,
            arm_a_velocity=arm_velocity,
            arm_a_evaluation=arm_eval,
            budget_float32=budget,
        )
        self.assertEqual(len(calls), 520)
        self.assertEqual(len(result.candidates), 520)
        self.assertEqual(result.raw_normals.shape, (8, 32, 5, 15))
        self.assertEqual(result.mean_states.shape, (9, 5, 15))
        self.assertEqual(result.sigma_states.shape, (9, 5, 15))
        self.assertEqual(result.variance_states.shape, (8, 5, 15))
        self.assertEqual(result.elite_cem_query_indices.shape, (8, 13))
        self.assertEqual(result.candidates[0].global_policy_request_index, 6)
        self.assertEqual(result.candidates[-1].global_policy_request_index, 525)
        self.assertNotEqual(result.selected.source, "A_equal_split")
        # Population order is center, then +z, -z. Generation-zero mean is
        # exact zero here, so the first antithetic pair sums to zero.
        np.testing.assert_array_equal(
            result.candidates[1].raw_proposal + result.candidates[2].raw_proposal,
            np.zeros((5, 15), dtype=np.float64),
        )

    def test_arm_a_remains_finite_miss_incumbent(self) -> None:
        budget = np.float32(1.0)
        velocity = np.zeros((5, 15), dtype=np.float32)
        executed = (DT_FLOAT32 * velocity).astype(np.float32)

        def evaluation(error: float, c: np.ndarray) -> ActualForwardEvaluation:
            return ActualForwardEvaluation(
                executed_increment=c,
                normalized_final=np.zeros((10, 32), dtype=np.float32),
                returned_physical_action=np.zeros((10, 7), dtype=np.float64),
                physical_error=np.full((5, 7), error, dtype=np.float32),
            )

        arm = evaluation(0.1, executed)

        def worse(u: np.ndarray) -> ActualForwardEvaluation:
            return evaluation(0.2, (DT_FLOAT32 * u).astype(np.float32))

        result = run_actual_forward_cem(
            worse,
            arm_a_velocity=velocity,
            arm_a_evaluation=arm,
            budget_float32=budget,
        )
        self.assertEqual(result.selected.source, "A_equal_split")
        self.assertEqual(result.selected.pool_index, 0)


if __name__ == "__main__":
    unittest.main()
