from __future__ import annotations

import importlib.util
import unittest


DEPENDENCIES_PRESENT = all(
    importlib.util.find_spec(name) is not None for name in ("numpy", "scipy")
)


@unittest.skipUnless(DEPENDENCIES_PRESENT, "surface numerical dependencies unavailable")
class SurfaceSamplingTest(unittest.TestCase):
    def setUp(self) -> None:
        import numpy as np

        self.np = np

    def test_triangle_lattice_reports_conservative_cover_bound(self) -> None:
        np = self.np
        from main.poisson_fullbody.surface_sampling import sample_triangle_surface

        vertices = np.array([[0.0, 0.0, 0.0], [0.21, 0.0, 0.0], [0.0, 0.08, 0.0]])
        result = sample_triangle_surface(vertices, [[0, 1, 2]], epsilon_m=0.05)
        self.assertLess(result.certified_cover_radius_m, 0.05)
        self.assertGreater(result.points.shape[0], 3)

        rng = np.random.RandomState(7)
        weights = rng.dirichlet([1.0, 1.0, 1.0], size=10000)
        audit_points = weights @ vertices
        from main.poisson_fullbody.surface_sampling import audit_finite_surface_coverage

        audit = audit_finite_surface_coverage(
            result.points, audit_points, epsilon_m=0.05
        )
        self.assertTrue(audit["passed"], audit)

    def test_exact_edge_multiple_still_has_strict_open_ball_cover(self) -> None:
        np = self.np
        from main.poisson_fullbody.surface_sampling import sample_triangle_surface

        epsilon = 0.05
        vertices = np.asarray(
            [[0.0, 0.0, 0.0], [0.10, 0.0, 0.0], [0.0, 0.10, 0.0]]
        )
        result = sample_triangle_surface(
            vertices, np.asarray([[0, 1, 2]]), epsilon_m=epsilon
        )
        self.assertLess(result.certified_cover_radius_m, epsilon)

    def test_box_mesh_contains_all_six_faces(self) -> None:
        np = self.np
        from main.poisson_fullbody.surface_sampling import (
            box_triangle_mesh,
            sample_triangle_surface,
        )

        half = np.array([0.1, 0.07, 0.03])
        vertices, faces = box_triangle_mesh(half)
        self.assertEqual(faces.shape, (12, 3))
        result = sample_triangle_surface(vertices, faces, epsilon_m=0.04)
        for axis in range(3):
            self.assertTrue(np.any(np.isclose(result.points[:, axis], half[axis])))
            self.assertTrue(np.any(np.isclose(result.points[:, axis], -half[axis])))
        self.assertLess(result.certified_cover_radius_m, 0.04)

    def test_degenerate_triangle_is_rejected(self) -> None:
        from main.poisson_fullbody.surface_sampling import sample_triangle_surface

        with self.assertRaisesRegex(ValueError, "degenerate"):
            sample_triangle_surface(
                [[0.0, 0.0, 0.0]] * 3, [[0, 1, 2]], epsilon_m=0.05
            )


if __name__ == "__main__":
    unittest.main()
