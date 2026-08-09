import unittest


class GradientRandomControlTest(unittest.TestCase):
    def test_sampling_is_deterministic_unit_and_feasible(self):
        from main.multilink_ellipsoid.gradient_random_control import (
            sample_feasible_unit_directions,
        )

        first = sample_feasible_unit_directions(
            [0.0, 0.0, -0.5], radius=0.5, count=8, seed=238,
            action_limit=1.0, maximum_attempts=1000,
        )
        second = sample_feasible_unit_directions(
            [0.0, 0.0, -0.5], radius=0.5, count=8, seed=238,
            action_limit=1.0, maximum_attempts=1000,
        )
        self.assertEqual(first, second)
        for direction in first:
            self.assertAlmostEqual(sum(value * value for value in direction), 1.0)
            self.assertTrue(all(-1.0 <= 0.5 * value <= 1.0 for value in direction))

    def test_first_safe_and_empirical_value(self):
        from main.multilink_ellipsoid.gradient_random_control import (
            empirical_equal_or_earlier_p, first_safe_radius,
        )

        radii = [0.1, 0.25, 0.5, 1.0]
        self.assertEqual(first_safe_radius(radii, [False, True, True, True]), 0.25)
        self.assertAlmostEqual(
            empirical_equal_or_earlier_p(0.25, [None, 0.5, 0.25, 0.1]), 0.6
        )


if __name__ == "__main__":
    unittest.main()
