import math
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "main"
if str(MAIN) not in sys.path:
    sys.path.insert(0, str(MAIN))

try:
    import numpy as np
except ImportError:  # pragma: no cover - repository structural environments
    np = None


@unittest.skipIf(np is None, "NumPy is optional outside the Poisson arm")
class PoissonGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from poisson_fullbody.geometry import (
            OrientedBox,
            minimum_point_to_oriented_boxes_distance,
            obb_intersects_aabb,
            oriented_box_to_aabb_distance,
            point_to_oriented_box_distance,
        )
        from poisson_fullbody.voxel_grid import (
            GridSpec,
            OccupancyResult,
            PoissonDomain,
            buffer_cell_union,
            build_connected_domain,
            build_occupancy,
            connected_safe_component,
            incident_cells,
            rasterize_buffered_oriented_boxes,
            rasterize_oriented_boxes,
        )

        cls.OrientedBox = OrientedBox
        cls.minimum_point_to_oriented_boxes_distance = staticmethod(
            minimum_point_to_oriented_boxes_distance
        )
        cls.obb_intersects_aabb = staticmethod(obb_intersects_aabb)
        cls.oriented_box_to_aabb_distance = staticmethod(
            oriented_box_to_aabb_distance
        )
        cls.point_to_oriented_box_distance = staticmethod(
            point_to_oriented_box_distance
        )
        cls.GridSpec = GridSpec
        cls.OccupancyResult = OccupancyResult
        cls.PoissonDomain = PoissonDomain
        cls.buffer_cell_union = staticmethod(buffer_cell_union)
        cls.build_connected_domain = staticmethod(build_connected_domain)
        cls.build_occupancy = staticmethod(build_occupancy)
        cls.connected_safe_component = staticmethod(connected_safe_component)
        cls.incident_cells = staticmethod(incident_cells)
        cls.rasterize_buffered_oriented_boxes = staticmethod(
            rasterize_buffered_oriented_boxes
        )
        cls.rasterize_oriented_boxes = staticmethod(rasterize_oriented_boxes)

    def test_bundle_container_arrays_are_detached_and_nonreversibly_read_only(self):
        center = np.asarray([1.0, 2.0, 3.0])
        rotation = np.asfortranarray(np.eye(3))
        half_extents = np.asarray([0.1, 0.2, 0.3])
        box = self.OrientedBox(center, rotation, half_extents, geom_id=7)

        lower = np.asarray([0.0, 0.0, 0.0])
        spacing = np.asarray([0.1, 0.2, 0.3])
        grid = self.GridSpec(lower, spacing, (2, 2, 2))
        raw = np.zeros(grid.cell_shape, dtype=bool, order="F")
        raw[0, 0, 0] = True
        buffered = raw.copy(order="F")
        buffered[1, 1, 1] = True
        occupancy = self.OccupancyResult(
            grid=grid,
            raw_cells=raw,
            buffered_cells=buffered,
            epsilon=0.05,
            numerical_tolerance=0.0,
        )

        active = np.ones(grid.vertex_shape, dtype=bool, order="F")
        boundary = np.ones(grid.vertex_shape, dtype=bool, order="F")
        boundary[1, 1, 1] = False
        interior = np.zeros(grid.vertex_shape, dtype=bool, order="F")
        interior[1, 1, 1] = True
        component = np.ones(grid.cell_shape, dtype=bool, order="F")
        domain = self.PoissonDomain(
            component_cells=component,
            active_vertices=active,
            boundary_vertices=boundary,
            interior_vertices=interior,
        )

        center[:] = -10.0
        rotation[:] = -10.0
        half_extents[:] = 10.0
        lower[:] = -10.0
        spacing[:] = 10.0
        raw[:] = False
        buffered[:] = False
        active[:] = False
        boundary[:] = False
        interior[:] = False
        component[:] = False
        np.testing.assert_array_equal(box.center, [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(box.R, np.eye(3))
        np.testing.assert_array_equal(box.half_extents, [0.1, 0.2, 0.3])
        np.testing.assert_array_equal(grid.lower, [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(grid.spacing, [0.1, 0.2, 0.3])
        self.assertEqual(int(np.count_nonzero(occupancy.raw_cells)), 1)
        self.assertEqual(int(np.count_nonzero(occupancy.buffered_cells)), 2)
        self.assertTrue(domain.interior_vertices[1, 1, 1])

        for array in (
            box.center,
            box.R,
            box.half_extents,
            grid.lower,
            grid.spacing,
            grid.upper,
            occupancy.raw_cells,
            occupancy.buffered_cells,
            domain.component_cells,
            domain.active_vertices,
            domain.boundary_vertices,
            domain.interior_vertices,
        ):
            self.assertTrue(array.flags.c_contiguous)
            self.assertFalse(array.flags.writeable)
            with self.assertRaises(ValueError):
                array.flat[0] = array.flat[0]
            with self.assertRaises(ValueError):
                array.setflags(write=True)

    def test_grid_cell_and_vertex_semantics_are_explicit(self):
        grid = self.GridSpec(
            lower=[-1.0, 2.0, 5.0],
            spacing=[0.5, 2.0, 4.0],
            cell_shape=(4, 3, 2),
        )
        self.assertEqual(grid.cell_shape, (4, 3, 2))
        self.assertEqual(grid.vertex_shape, (5, 4, 3))
        np.testing.assert_allclose(grid.upper, [1.0, 8.0, 13.0])
        np.testing.assert_allclose(grid.vertex((2, 1, 1)), [0.0, 4.0, 9.0])
        lower, upper = grid.cell_bounds((2, 1, 1))
        np.testing.assert_allclose(lower, [0.0, 4.0, 9.0])
        np.testing.assert_allclose(upper, [0.5, 6.0, 13.0])
        with self.assertRaises(IndexError):
            grid.cell_bounds((4, 0, 0))
        with self.assertRaises(ValueError):
            self.GridSpec([0, 0, 0], [1, 0, 1], (2, 2, 2))
        with self.assertRaises(ValueError):
            self.GridSpec([0, 0, 0], [1, 1, 1], (2.0, 2, 2))

    def test_from_bounds_preserves_vertex_count(self):
        grid = self.GridSpec.from_bounds(
            lower=[0.0, -2.0, 1.0],
            upper=[2.0, 4.0, 9.0],
            vertex_shape=(5, 4, 3),
        )
        self.assertEqual(grid.cell_shape, (4, 3, 2))
        np.testing.assert_allclose(grid.spacing, [0.5, 2.0, 4.0])

    def test_oriented_box_rejects_nonphysical_pose(self):
        with self.assertRaises(ValueError):
            self.OrientedBox([0, 0, 0], np.eye(3), [1, 0, 1])
        bad = np.eye(3)
        bad[0, 0] = 2.0
        with self.assertRaises(ValueError):
            self.OrientedBox([0, 0, 0], bad, [1, 1, 1])
        reflection = np.diag([-1.0, 1.0, 1.0])
        with self.assertRaises(ValueError):
            self.OrientedBox([0, 0, 0], reflection, [1, 1, 1])

    def test_inclusive_sat_counts_touch_but_rejects_separation(self):
        angle = math.radians(35.0)
        rotation = np.asarray(
            [
                [math.cos(angle), -math.sin(angle), 0.0],
                [math.sin(angle), math.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        rotated = self.OrientedBox([0, 0, 0], rotation, [1.0, 0.2, 0.3])
        self.assertTrue(
            self.obb_intersects_aabb(rotated, [-0.1, -0.1, -0.1], [0.1, 0.1, 0.1])
        )
        self.assertFalse(
            self.obb_intersects_aabb(rotated, [2.0, 2.0, 2.0], [2.2, 2.2, 2.2])
        )

        axis_aligned = self.OrientedBox([0, 0, 0], np.eye(3), [0.5, 0.5, 0.5])
        self.assertTrue(
            self.obb_intersects_aabb(axis_aligned, [0.5, -0.1, -0.1], [0.7, 0.1, 0.1])
        )
        self.assertFalse(
            self.obb_intersects_aabb(
                axis_aligned,
                [0.500001, -0.1, -0.1],
                [0.7, 0.1, 0.1],
                tolerance=0.0,
            )
        )

    def test_exact_signed_distance_respects_rotated_box_axes(self):
        angle = math.pi / 2.0
        rotation = np.asarray(
            [
                [math.cos(angle), -math.sin(angle), 0.0],
                [math.sin(angle), math.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        box = self.OrientedBox(
            center=[1.0, 2.0, 3.0],
            R=rotation,
            half_extents=[2.0, 1.0, 0.5],
        )
        self.assertAlmostEqual(
            self.point_to_oriented_box_distance([1.0, 2.0, 3.0], box),
            -0.5,
        )
        surface = box.center + np.matmul(rotation, [2.0, 0.0, 0.0])
        self.assertAlmostEqual(
            self.point_to_oriented_box_distance(surface, box), 0.0, places=14
        )
        outside_corner = box.center + np.matmul(rotation, [3.0, 2.0, 0.0])
        self.assertAlmostEqual(
            self.point_to_oriented_box_distance(outside_corner, box),
            math.sqrt(2.0),
            places=14,
        )

    def test_exact_obb_aabb_distance_covers_face_edge_and_intersection(self):
        axis_aligned = self.OrientedBox(
            [0.0, 0.0, 0.0], np.eye(3), [0.5, 0.5, 0.5]
        )
        self.assertAlmostEqual(
            self.oriented_box_to_aabb_distance(
                axis_aligned, [0.7, -0.2, -0.2], [0.9, 0.2, 0.2], tolerance=0.0
            ),
            0.2,
        )
        self.assertEqual(
            self.oriented_box_to_aabb_distance(
                axis_aligned, [0.4, -0.1, -0.1], [0.7, 0.1, 0.1], tolerance=0.0
            ),
            0.0,
        )
        angle = math.pi / 4.0
        rotated = self.OrientedBox(
            [0.0, 0.0, 0.0],
            np.asarray(
                [
                    [math.cos(angle), -math.sin(angle), 0.0],
                    [math.sin(angle), math.cos(angle), 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
            [0.5, 0.1, 0.1],
        )
        # A point-like AABB query exercises the rotated vertex/edge path and
        # must agree with the independent point-to-OBB formula.
        point = np.asarray([0.8, 0.0, 0.0])
        expected = max(0.0, self.point_to_oriented_box_distance(point, rotated))
        self.assertAlmostEqual(
            self.oriented_box_to_aabb_distance(
                rotated, point, point, tolerance=0.0
            ),
            expected,
            places=13,
        )

    def test_random_obb_aabb_distance_matches_convex_lsq_reference(self):
        try:
            from scipy.optimize import lsq_linear
        except ImportError:
            self.skipTest("SciPy reference optimizer unavailable")
        rng = np.random.RandomState(238)
        for _ in range(250):
            matrix = rng.normal(size=(3, 3))
            rotation, _ = np.linalg.qr(matrix)
            if np.linalg.det(rotation) < 0.0:
                rotation[:, 0] *= -1.0
            box = self.OrientedBox(
                rng.uniform(-1.0, 1.0, size=3),
                rotation,
                rng.uniform(0.03, 0.4, size=3),
            )
            aabb_center = rng.uniform(-1.0, 1.0, size=3)
            aabb_half = rng.uniform(0.03, 0.4, size=3)
            lower = aabb_center - aabb_half
            upper = aabb_center + aabb_half
            # Variables are [u_box_local, v_aabb_world].  The residual is
            # R*u - v + center, with independent box constraints.
            reference = lsq_linear(
                np.concatenate((rotation, -np.eye(3)), axis=1),
                -box.center,
                bounds=(
                    np.concatenate((-box.half_extents, lower)),
                    np.concatenate((box.half_extents, upper)),
                ),
                tol=1.0e-13,
                lsmr_tol=1.0e-13,
                max_iter=500,
            )
            self.assertTrue(reference.success, reference.message)
            expected = float(np.linalg.norm(reference.fun))
            observed = self.oriented_box_to_aabb_distance(
                box, lower, upper, tolerance=0.0
            )
            self.assertAlmostEqual(observed, expected, delta=2.0e-9)

    def test_minimum_box_union_distance_keeps_nearest_geom(self):
        left = self.OrientedBox([-2, 0, 0], np.eye(3), [0.5, 0.5, 0.5], 10)
        right = self.OrientedBox([2, 0, 0], np.eye(3), [0.5, 0.5, 0.5], 11)
        self.assertAlmostEqual(
            self.minimum_point_to_oriented_boxes_distance(
                [1.25, 0.0, 0.0], [left, right]
            ),
            0.25,
        )
        self.assertAlmostEqual(
            self.minimum_point_to_oriented_boxes_distance(
                [2.0, 0.0, 0.0], [left, right]
            ),
            -0.5,
        )
        with self.assertRaises(ValueError):
            self.minimum_point_to_oriented_boxes_distance([0, 0, 0], [])

    def test_rasterization_uses_cell_intersection_not_cell_centres(self):
        grid = self.GridSpec([0, 0, 0], [1, 1, 1], (3, 3, 3))
        # A thin rotated box crosses four cells around the x/y grid corner at
        # (1,1), although no one of those cell centres lies inside it.
        angle = math.pi / 4.0
        rotation = np.asarray(
            [
                [math.cos(angle), -math.sin(angle), 0.0],
                [math.sin(angle), math.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        box = self.OrientedBox(
            center=[1.0, 1.0, 0.5],
            R=rotation,
            half_extents=[0.6, 0.01, 0.05],
            geom_id=17,
        )
        raw = self.rasterize_oriented_boxes(grid, [box])
        self.assertTrue(raw[0, 0, 0])
        self.assertTrue(raw[1, 1, 0])
        self.assertGreaterEqual(int(raw.sum()), 4)

    def test_anisotropic_exact_cell_union_buffer_has_no_hidden_half_diagonal(self):
        grid = self.GridSpec([0, 0, 0], [1.0, 2.0, 4.0], (7, 7, 7))
        raw = np.zeros(grid.cell_shape, dtype=bool)
        raw[3, 3, 3] = True

        buffered = self.buffer_cell_union(grid, raw, epsilon=0.1, tolerance=0.0)
        # Adjacent closed cells have zero gap and are conservatively included.
        self.assertTrue(buffered[4, 4, 4])
        # Two cells away has gap one full axis spacing, not centre distance
        # minus an added half diagonal.  A hidden half-diagonal would include
        # these anisotropic y/z cells even at epsilon=0.1.
        self.assertFalse(buffered[5, 3, 3])
        self.assertFalse(buffered[3, 5, 3])
        self.assertFalse(buffered[3, 3, 5])
        # The operation is one-shot: newly included neighbours do not seed a
        # second morphological expansion.
        self.assertFalse(buffered[6, 3, 3])

        exact_x = self.buffer_cell_union(grid, raw, epsilon=1.0, tolerance=0.0)
        self.assertTrue(exact_x[5, 3, 3])
        self.assertFalse(exact_x[3, 5, 3])
        self.assertFalse(exact_x[3, 3, 5])

    def test_build_occupancy_buffers_exact_geometry_once_and_is_immutable(self):
        grid = self.GridSpec([0, 0, 0], [0.5, 0.5, 0.5], (8, 8, 8))
        box = self.OrientedBox([1.25, 1.25, 1.25], np.eye(3), [0.2, 0.2, 0.2])
        result = self.build_occupancy(grid, [box], epsilon=0.25, tolerance=0.0)
        expected = self.rasterize_buffered_oriented_boxes(
            grid, [box], epsilon=0.25, tolerance=0.0
        )
        np.testing.assert_array_equal(result.buffered_cells, expected)
        self.assertTrue(np.all(result.buffered_cells[result.raw_cells]))
        metadata = result.audit_metadata()
        self.assertEqual(metadata["buffer_epsilon_m"], 0.25)
        self.assertAlmostEqual(
            metadata["grid_cell_diagonal_m"], math.sqrt(3.0) * 0.5
        )
        self.assertEqual(metadata["explicit_half_diagonal_padding_m"], 0.0)
        self.assertEqual(
            metadata["total_buffer_conservatism_upper_bound_m"],
            metadata["grid_cell_diagonal_m"],
        )
        self.assertEqual(
            metadata["buffer_source"],
            "direct_exact_obb_epsilon_neighborhood",
        )
        self.assertTrue(metadata["buffer_applied_once"])
        with self.assertRaises(ValueError):
            result.raw_cells[0, 0, 0] = True
        with self.assertRaisesRegex(ValueError, "selected obstacle geometry"):
            self.build_occupancy(grid, [], epsilon=0.25, tolerance=0.0)
        outside = self.OrientedBox(
            [20.0, 20.0, 20.0], np.eye(3), [0.2, 0.2, 0.2]
        )
        with self.assertRaisesRegex(ValueError, "does not intersect"):
            self.build_occupancy(grid, [outside], epsilon=0.25, tolerance=0.0)

    def test_direct_zero_buffer_does_not_add_raw_cell_neighbors(self):
        grid = self.GridSpec([0, 0, 0], [1.0, 1.0, 1.0], (5, 5, 5))
        box = self.OrientedBox(
            [2.5, 2.5, 2.5], np.eye(3), [0.2, 0.2, 0.2]
        )
        raw = self.rasterize_oriented_boxes(grid, [box], tolerance=0.0)
        direct = self.rasterize_buffered_oriented_boxes(
            grid, [box], epsilon=0.0, tolerance=0.0
        )
        np.testing.assert_array_equal(direct, raw)
        legacy = self.buffer_cell_union(grid, raw, epsilon=0.0, tolerance=0.0)
        self.assertGreater(int(legacy.sum()), int(direct.sum()))

    def test_points_on_faces_return_all_incident_cells(self):
        grid = self.GridSpec([0, 0, 0], [1, 1, 1], (4, 4, 4))
        face_cells = self.incident_cells(grid, [2.0, 1.25, 1.25], tolerance=0.0)
        self.assertEqual(set(face_cells), {(1, 1, 1), (2, 1, 1)})
        vertex_cells = self.incident_cells(grid, [2.0, 2.0, 2.0], tolerance=0.0)
        self.assertEqual(len(vertex_cells), 8)
        outer_cells = self.incident_cells(grid, [0.0, 0.5, 0.5], tolerance=0.0)
        self.assertEqual(outer_cells, ((0, 0, 0),))

    def test_connected_domain_does_not_union_components(self):
        grid = self.GridSpec([0, 0, 0], [1, 1, 1], (5, 5, 5))
        blocked = np.zeros(grid.cell_shape, dtype=bool)
        blocked[2, :, :] = True

        component = self.connected_safe_component(blocked, [(0, 2, 2)])
        self.assertTrue(np.all(component[:2, :, :]))
        self.assertFalse(np.any(component[2:, :, :]))
        with self.assertRaises(ValueError):
            self.connected_safe_component(blocked, [(0, 2, 2), (4, 2, 2)])

        domain = self.build_connected_domain(
            grid, blocked, seed_cells=[(0, 2, 2)]
        )
        self.assertTrue(np.array_equal(domain.component_cells, component))
        self.assertTrue(
            np.array_equal(
                domain.active_vertices,
                domain.boundary_vertices | domain.interior_vertices,
            )
        )
        self.assertFalse(np.any(domain.boundary_vertices & domain.interior_vertices))
        # x=2 is the component face beside the blocked wall and is Dirichlet.
        self.assertTrue(np.all(domain.boundary_vertices[2, :, :]))
        self.assertTrue(domain.interior_vertices[1, 2, 2])
        self.assertFalse(domain.active_vertices[3, 2, 2])

    def test_face_seed_touching_blocked_cell_is_rejected(self):
        grid = self.GridSpec([0, 0, 0], [1, 1, 1], (5, 5, 5))
        blocked = np.zeros(grid.cell_shape, dtype=bool)
        blocked[2, :, :] = True
        # The point is geometrically on the shared face.  Picking only floor
        # cell 2 or only cell 1 would make its status arbitrary.
        with self.assertRaisesRegex(ValueError, "touches a blocked cell"):
            self.build_connected_domain(
                grid,
                blocked,
                seed_points=[[2.0, 2.5, 2.5]],
                tolerance=0.0,
            )

    def test_thin_domain_and_bad_masks_fail_closed(self):
        grid = self.GridSpec([0, 0, 0], [1, 1, 1], (1, 4, 4))
        blocked = np.zeros(grid.cell_shape, dtype=bool)
        with self.assertRaisesRegex(ValueError, "too thin"):
            self.build_connected_domain(grid, blocked, seed_cells=[(0, 1, 1)])
        with self.assertRaises(TypeError):
            self.buffer_cell_union(
                grid, np.zeros(grid.cell_shape, dtype=np.uint8), epsilon=0.1
            )


if __name__ == "__main__":
    unittest.main()
