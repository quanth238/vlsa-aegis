from __future__ import annotations

import importlib.util
import unittest


DEPENDENCIES_PRESENT = all(
    importlib.util.find_spec(name) is not None for name in ("numpy", "scipy")
)
MUJOCO_PRESENT = DEPENDENCIES_PRESENT and importlib.util.find_spec("mujoco") is not None


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

    def test_exact_cylinder_side_and_caps_have_strict_analytic_cover(self) -> None:
        np = self.np
        from main.poisson_fullbody.surface_sampling import (
            audit_finite_surface_coverage,
            sample_cylinder_surface,
        )

        radius = 0.18
        half_length = 0.31
        epsilon = 0.05
        result = sample_cylinder_surface(
            radius, half_length, epsilon_m=epsilon
        )
        self.assertLess(result.certified_cover_radius_m, epsilon)
        self.assertEqual(result.angular_sample_count, 16)
        self.assertEqual(result.axial_interval_count, 9)
        self.assertEqual(result.cap_radial_interval_count, 3)
        self.assertEqual(result.points.shape, (226, 3))
        radial = np.linalg.norm(result.points[:, :2], axis=1)
        on_side = np.isclose(radial, radius, rtol=0.0, atol=1e-12)
        on_cap = np.isclose(
            np.abs(result.points[:, 2]), half_length, rtol=0.0, atol=1e-12
        ) & (radial <= radius + 1e-12)
        self.assertTrue(np.all(on_side | on_cap))
        self.assertTrue(np.any(np.isclose(radial, 0.0, atol=1e-15)))
        self.assertTrue(np.any(on_side & ~on_cap))

        theta = np.linspace(0.0, 2.0 * np.pi, 257, endpoint=False)
        side_z = np.linspace(-half_length, half_length, 41)
        side = np.asarray(
            [
                [radius * np.cos(angle), radius * np.sin(angle), z]
                for z in side_z
                for angle in theta
            ],
            dtype=np.float64,
        )
        cap_radii = np.linspace(0.0, radius, 25)
        caps = np.asarray(
            [
                [rho * np.cos(angle), rho * np.sin(angle), z]
                for z in (-half_length, half_length)
                for rho in cap_radii
                for angle in theta
            ],
            dtype=np.float64,
        )
        audit = audit_finite_surface_coverage(
            result.points,
            np.vstack((side, caps)),
            epsilon_m=epsilon,
        )
        self.assertTrue(audit["passed"], audit)
        self.assertLessEqual(
            audit["maximum_nearest_distance_m"],
            result.certified_cover_radius_m + 1e-12,
        )

    def test_cylinder_invalid_dimensions_are_rejected(self) -> None:
        from main.poisson_fullbody.surface_sampling import sample_cylinder_surface

        for radius, half_length, epsilon in (
            (0.0, 0.1, 0.05),
            (-0.1, 0.1, 0.05),
            (0.1, 0.0, 0.05),
            (0.1, 0.1, 0.0),
            (float("nan"), 0.1, 0.05),
            (0.1, float("inf"), 0.05),
        ):
            with self.subTest(
                radius=radius, half_length=half_length, epsilon=epsilon
            ):
                with self.assertRaises(ValueError):
                    sample_cylinder_surface(
                        radius, half_length, epsilon_m=epsilon
                    )

    def test_cylinder_exact_component_multiple_remains_strict(self) -> None:
        import math

        from main.poisson_fullbody.surface_sampling import sample_cylinder_surface

        epsilon = 0.05
        component = epsilon / math.sqrt(2.0)
        result = sample_cylinder_surface(
            4.0 * component,
            2.0 * component,
            epsilon_m=epsilon,
        )
        self.assertEqual(result.axial_interval_count, 3)
        self.assertEqual(result.cap_radial_interval_count, 3)
        self.assertLess(result.certified_cover_radius_m, epsilon)

    def test_cylinder_pathological_sampling_request_fails_before_allocation(self) -> None:
        from main.poisson_fullbody.surface_sampling import sample_cylinder_surface

        with self.assertRaisesRegex(ValueError, "sampling exceeds"):
            sample_cylinder_surface(0.1, 0.1, epsilon_m=1e-20)

    def test_cylinder_subnormal_epsilon_fails_closed(self) -> None:
        from main.poisson_fullbody.surface_sampling import sample_cylinder_surface

        with self.assertRaisesRegex(ValueError, "implementation range"):
            sample_cylinder_surface(
                0.1,
                0.1,
                epsilon_m=float.fromhex("0x0.0000000000001p-1022"),
            )


@unittest.skipUnless(MUJOCO_PRESENT, "MuJoCo surface dependency unavailable")
class MujocoCylinderSurfaceSamplingTest(unittest.TestCase):
    def test_collision_sampler_accepts_exact_cylinder_with_generic_certificate(self) -> None:
        import mujoco

        from main.poisson_fullbody.surface_sampling import (
            build_robot_collision_samples,
        )

        model = mujoco.MjModel.from_xml_string(
            """
<mujoco>
  <worldbody>
    <body name="robot_mount" pos="0.2 -0.1 0.4">
      <geom name="pedestal_col" type="cylinder" size="0.18 0.31"/>
    </body>
  </worldbody>
</mujoco>
"""
        )
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "robot_mount"
        )
        result = build_robot_collision_samples(
            model, data, body_ids=(body_id,), epsilon_m=0.05
        )
        self.assertLess(result.maximum_surface_cover_radius_m, 0.05)
        self.assertEqual(len(result.sample_ledger_sha256), 64)
        self.assertEqual(len(result.geom_records), 1)
        record = result.geom_records[0]
        self.assertEqual(record["geom_name"], "pedestal_col")
        self.assertEqual(record["geom_type_id"], int(mujoco.mjtGeom.mjGEOM_CYLINDER))
        self.assertEqual(record["geom_type_name"], "cylinder")
        self.assertEqual(record["geom_size"], [0.18, 0.31, 0.0])
        self.assertEqual(record["selection_authority"], "mask_enabled_geoms_under_selected_bodies")
        self.assertEqual(record["geometry_kind"], "exact_cylinder_surface")
        self.assertEqual(
            record["certificate_kind"],
            "analytic_cylinder_parameter_grid_covering_bound",
        )
        self.assertGreater(record["surface_element_count"], 0)
        self.assertLess(record["certified_surface_cover_radius_m"], 0.05)
        self.assertEqual(record["sample_count"], len(result.samples))
        self.assertEqual(
            record["certificate_parameters"]["angular_sample_count"], 16
        )
        self.assertEqual(
            record["certificate_parameters"]["axial_interval_count"], 9
        )
        self.assertEqual(
            record["certificate_parameters"]["cap_radial_interval_count"], 3
        )

        from main.poisson_fullbody.surface_sampling import (
            validate_robot_sample_evidence,
        )

        authoritative = build_robot_collision_samples(
            model,
            data,
            geom_ids=(record["geom_id"],),
            epsilon_m=0.05,
        )
        authoritative_record = authoritative.geom_records[0]
        evidence = {
            "sample_count": len(authoritative.samples),
            "sample_ledger_sha256": authoritative.sample_ledger_sha256,
            "geom_records": list(authoritative.geom_records),
            "epsilon_m": authoritative.epsilon_m,
            "maximum_surface_cover_radius_m": (
                authoritative.maximum_surface_cover_radius_m
            ),
            "coverage_semantics": authoritative.coverage_semantics,
            "roundtrip": {
                "sample_count": len(authoritative.samples),
                "maximum_roundtrip_error_m": 0.0,
                "tolerance_m": 1e-10,
                "passed": True,
            },
        }
        counts = validate_robot_sample_evidence(
            evidence,
            resolved_geom_ids=(authoritative_record["geom_id"],),
            resolved_geom_names=(authoritative_record["geom_name"],),
            resolved_body_ids=(authoritative_record["body_id"],),
            roundtrip_field="roundtrip",
        )
        self.assertEqual(counts, {"mesh": 0, "box": 0, "cylinder": 1})

    def test_authoritative_geom_ids_include_explicit_pair_only_geom(self) -> None:
        import mujoco

        from main.poisson_fullbody.measurement import resolve_collision_geom_sets
        from main.poisson_fullbody.surface_sampling import (
            build_robot_collision_samples,
        )

        model = mujoco.MjModel.from_xml_string(
            """
<mujoco>
  <worldbody>
    <body name="robot_root"><body name="link5">
      <geom name="robot_pair_only" type="box" size="0.02 0.03 0.04"
            contype="0" conaffinity="0"/>
    </body></body>
    <body name="obstacle_root" pos="0.2 0 0">
      <geom name="obstacle_pair_only" type="box" size="0.02 0.03 0.04"
            contype="0" conaffinity="0"/>
    </body>
  </worldbody>
  <contact><pair geom1="robot_pair_only" geom2="obstacle_pair_only"/></contact>
</mujoco>
"""
        )
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        body = lambda name: mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, name
        )
        resolved = resolve_collision_geom_sets(
            model,
            robot_root_body_ids=(body("robot_root"),),
            obstacle_root_body_ids=(body("obstacle_root"),),
            link56_body_ids=(body("link5"),),
        )
        result = build_robot_collision_samples(
            model, data, geom_ids=resolved.robot_geom_ids, epsilon_m=0.05
        )
        self.assertEqual(
            tuple(record["geom_id"] for record in result.geom_records),
            resolved.robot_geom_ids,
        )
        self.assertFalse(result.geom_records[0]["mask_collision_enabled"])
        self.assertEqual(
            result.geom_records[0]["selection_authority"],
            "authoritative_resolved_geom_ids",
        )


if __name__ == "__main__":
    unittest.main()
