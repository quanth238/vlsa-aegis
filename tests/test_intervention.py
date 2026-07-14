from __future__ import annotations

import unittest

from crfs_harness.intervention import endpoint_project, endpoint_sum, integrate_remaining_flow
from crfs_harness.math3d import matrix_add, matrix_scale, matrix_subtract


class InterventionTest(unittest.TestCase):
    def test_endpoint_projection_has_zero_sum(self) -> None:
        projected = endpoint_project(((1.0, 2.0, 3.0), (-2.0, 4.0, 0.0), (5.0, -1.0, 9.0)))
        for value in endpoint_sum(projected):
            self.assertAlmostEqual(value, 0.0)

    def test_distributed_residual_reaches_corrected_action_for_constant_flow(self) -> None:
        nominal = ((0.2, 0.0), (0.2, 0.0))
        correction = ((0.0, 0.3), (0.0, -0.3))
        noise = ((1.0, -1.0), (-0.5, 0.5))
        t_s = 0.5
        x_t = matrix_add(matrix_scale(noise, t_s), matrix_scale(nominal, 1.0 - t_s))

        def base_velocity(_state, _time):
            return matrix_subtract(noise, nominal)

        final = integrate_remaining_flow(x_t, base_velocity, t_s, 5, correction=correction)
        expected = matrix_add(nominal, correction)
        for actual_row, expected_row in zip(final, expected, strict=True):
            for actual, wanted in zip(actual_row, expected_row, strict=True):
                self.assertAlmostEqual(actual, wanted)

    def test_bridge_edit_is_not_silently_equated_with_residual(self) -> None:
        nominal = ((0.2, 0.0), (0.2, 0.0))
        correction = ((0.0, 0.3), (0.0, -0.3))
        noise = ((1.0, -1.0), (-0.5, 0.5))
        t_s = 0.5
        x_t = matrix_add(matrix_scale(noise, t_s), matrix_scale(nominal, 1.0 - t_s))

        def base_velocity(_state, _time):
            return matrix_subtract(noise, nominal)

        residual = integrate_remaining_flow(x_t, base_velocity, t_s, 5, correction=correction)
        bridge = integrate_remaining_flow(x_t, base_velocity, t_s, 5, correction=correction, one_shot_edit=True)
        self.assertNotEqual(residual, bridge)


if __name__ == "__main__":
    unittest.main()
