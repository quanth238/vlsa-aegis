from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
NUMPY_AVAILABLE = importlib.util.find_spec("numpy") is not None


@unittest.skipUnless(NUMPY_AVAILABLE, "NumPy is optional for the structural gate")
class MultilinkEllipsoidGeometryTests(unittest.TestCase):
    def setUp(self) -> None:
        import numpy as np

        self.np = np
        from main.multilink_ellipsoid.geometry import Ellipsoid

        self.Ellipsoid = Ellipsoid

    def test_box_bound_contains_all_corners(self) -> None:
        from main.multilink_ellipsoid.geometry import primitive_bounding_radii

        np = self.np
        half = np.asarray([0.1, 0.2, 0.3])
        radii, source = primitive_bounding_radii("box", half, 0.5)
        self.assertEqual(source, "loewner_box_enclosing_ellipsoid")
        for signs in (
            (-1, -1, -1),
            (-1, -1, 1),
            (-1, 1, -1),
            (-1, 1, 1),
            (1, -1, -1),
            (1, -1, 1),
            (1, 1, -1),
            (1, 1, 1),
        ):
            point = half * np.asarray(signs)
            self.assertLessEqual(float(np.sum((point / radii) ** 2)), 1.0 + 1e-12)

    def test_capsule_support_is_enclosed_in_every_direction(self) -> None:
        from main.multilink_ellipsoid.geometry import (
            primitive_bounding_radii,
            primitive_enclosure_certificate,
        )

        np = self.np
        radius = 0.04
        half_length = 0.16
        radii, source = primitive_bounding_radii(
            "capsule", [radius, half_length, 0.0], radius + half_length
        )
        self.assertEqual(source, "closed_form_capsule_enclosing_ellipsoid")
        for z_component in np.linspace(0.0, 1.0, 1001):
            radial = math.sqrt(max(0.0, 1.0 - z_component * z_component))
            ellipsoid_support = math.sqrt(
                radii[0] ** 2 * radial ** 2
                + radii[2] ** 2 * z_component ** 2
            )
            capsule_support = radius + half_length * z_component
            self.assertGreaterEqual(ellipsoid_support + 1e-12, capsule_support)
        certificate = primitive_enclosure_certificate(
            "capsule",
            [radius, half_length, 0.0],
            radius + half_length,
            radii,
            source,
        )
        self.assertTrue(certificate["verified"])
        self.assertEqual(certificate["maximum_normalized_quadratic"], 1.0)

    def test_mvee_link_fit_contains_every_mesh_vertex(self) -> None:
        from main.multilink_ellipsoid.geometry import (
            minimum_volume_enclosing_ellipsoid,
        )

        np = self.np
        points = np.asarray(
            [
                [x, y, z]
                for x in (-0.31, 0.29)
                for y in (-0.055, 0.045)
                for z in (-0.04, 0.06)
            ],
            dtype=np.float64,
        )
        ellipsoid = minimum_volume_enclosing_ellipsoid(
            points,
            body_id=5,
            body_name="robot0_link5",
            geom_id=15,
            geom_name="robot0_link5_collision",
            source_body_names=("robot0_link5",),
            source_geom_names=("robot0_link5_collision",),
        )
        local = (points - ellipsoid.center) @ ellipsoid.rotation
        normalized = np.sum((local / ellipsoid.semiaxes_m) ** 2, axis=1)
        self.assertLessEqual(float(np.max(normalized)), 1.0 + 1.0e-12)
        self.assertGreater(float(np.max(normalized)), 0.99999999)
        self.assertEqual(
            ellipsoid.enclosure_certificate["source_vertex_count"], len(points)
        )
        self.assertTrue(ellipsoid.enclosure_certificate["convex_hull_contained"])

    def test_analytic_twist_derivative_matches_central_difference(self) -> None:
        from main.multilink_ellipsoid.barrier import support_gap, twist_coefficients

        np = self.np
        robot = self.Ellipsoid(
            center=[0.1, -0.2, 0.3],
            rotation=np.eye(3),
            semiaxes_m=[0.08, 0.05, 0.12],
        )
        obstacle = self.Ellipsoid(
            center=[0.48, 0.07, 0.42],
            rotation=np.asarray(
                [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
            ),
            semiaxes_m=[0.11, 0.07, 0.09],
        )
        linear_velocity = np.asarray([0.07, -0.03, 0.02])
        angular_velocity = np.asarray([0.2, -0.1, 0.15])
        linear, angular = twist_coefficients(robot, obstacle)
        analytic = float(linear @ linear_velocity + angular @ angular_velocity)

        def skew(value):
            x, y, z = value
            return np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])

        epsilon = 1e-6
        plus_rotation = np.eye(3) + epsilon * skew(angular_velocity)
        minus_rotation = np.eye(3) - epsilon * skew(angular_velocity)
        plus = self.Ellipsoid(
            center=robot.center + epsilon * linear_velocity,
            rotation=np.linalg.svd(plus_rotation)[0] @ np.linalg.svd(plus_rotation)[2],
            semiaxes_m=robot.semiaxes_m,
        )
        minus = self.Ellipsoid(
            center=robot.center - epsilon * linear_velocity,
            rotation=np.linalg.svd(minus_rotation)[0] @ np.linalg.svd(minus_rotation)[2],
            semiaxes_m=robot.semiaxes_m,
        )
        numerical = (support_gap(plus, obstacle) - support_gap(minus, obstacle)) / (
            2.0 * epsilon
        )
        self.assertAlmostEqual(analytic, numerical, places=7)


class MultilinkEllipsoidContractTests(unittest.TestCase):
    def test_preregistered_config_loads_and_separates_clearances(self) -> None:
        from main.multilink_ellipsoid.shadow import load_shadow_config

        config = load_shadow_config(
            ROOT / "configs/vlsa_multilink_ellipsoid_shadow_e05.v1.json"
        )
        self.assertEqual(
            config["protected_body_names"],
            ["robot0_link%d" % index for index in range(1, 8)],
        )
        self.assertTrue(config["simulator_verification"]["distinct_from_D_opt"])
        self.assertEqual(config["optimizer"]["optimizer_clearance_m"], 0.01)
        self.assertEqual(len(config["config_file_sha256"]), 64)

    def test_three_ellipsoid_config_targets_only_links_5_6_7(self) -> None:
        from main.multilink_ellipsoid.shadow import load_shadow_config

        config = load_shadow_config(
            ROOT / "configs/vlsa_distal_three_ellipsoid_shadow_e05.v2.json"
        )
        self.assertEqual(
            config["protected_body_names"],
            ["robot0_link5", "robot0_link6", "robot0_link7"],
        )
        self.assertEqual(
            config["robot_geometry"]["source"],
            "compiled_mujoco_collision_mesh_vertices",
        )

    def test_active_multicbf_config_filters_only_xyz_for_links_5_6_7(self) -> None:
        from main.multilink_ellipsoid.active import load_active_config

        config = load_active_config(
            ROOT / "configs/vlsa_distal_three_ellipsoid_multicbf_e05.v1.json"
        )
        self.assertEqual(
            config["protected_body_names"],
            ["robot0_link5", "robot0_link6", "robot0_link7"],
        )
        self.assertEqual(
            config["control_effect"],
            "modify_archived_aegis_xyz_before_env_step",
        )
        self.assertEqual(
            config["output_action_contract"]["gripper"],
            "preserve_archived_aegis_command",
        )
        self.assertTrue(config["simulator_verification"]["distinct_from_D_opt"])

    @unittest.skipUnless(NUMPY_AVAILABLE, "NumPy is optional locally")
    def test_resolved_rate_action_map_has_expected_linear_contract(self) -> None:
        import numpy as np

        from main.multilink_ellipsoid.active import resolved_rate_action_map

        jacobian = np.zeros((6, 7), dtype=np.float64)
        jacobian[:, :6] = np.eye(6)
        mapping = resolved_rate_action_map(jacobian, damping=0.1)
        self.assertEqual(mapping.shape, (7, 3))
        np.testing.assert_allclose(
            mapping[:3],
            np.eye(3) / 1.01,
            rtol=0.0,
            atol=1.0e-12,
        )
        np.testing.assert_array_equal(mapping[3:], np.zeros((4, 3)))

    def test_evaluator_shadow_is_opt_in_and_read_only(self) -> None:
        source = (ROOT / "main/evaluate_safelibero_aegis.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("--multilink-ellipsoid-shadow-config", source)
        self.assertIn("read_only_no_executed_action_change", source)
        self.assertIn("multilink_ellipsoid_shadow_config: Mapping", source)
        self.assertIn("observation, reward, done, info = env.step(executed)", source)
        self.assertIn("minimum_robot_contact_distance_m", source)
        qp_source = (ROOT / "main/multilink_ellipsoid/qp.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"input_constraint_count"', qp_source)

    def test_config_is_canonical_json_object(self) -> None:
        path = ROOT / "configs/vlsa_multilink_ellipsoid_shadow_e05.v1.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["case_ids"], ["vlsa-t1-goal-ii-t0-e05"])
        self.assertIn("read_only_oracle", payload["claim_scope"])


@unittest.skipUnless(
    NUMPY_AVAILABLE
    and importlib.util.find_spec("scipy") is not None
    and importlib.util.find_spec("osqp") is not None,
    "QP dependencies are allocation/runtime optional",
)
class MultilinkEllipsoidQpTests(unittest.TestCase):
    def test_two_coupled_constraints_are_solved_jointly(self) -> None:
        import numpy as np

        from main.multilink_ellipsoid.qp import MultiConstraintQp

        solver = MultiConstraintQp(eps_abs=1e-9, eps_rel=1e-9)
        result = solver.solve(
            [0.0, 0.0],
            np.eye(2),
            [[1.0, 1.0], [1.0, -1.0]],
            [1.0, 1.0],
            [-10.0, -10.0],
            [10.0, 10.0],
        )
        self.assertTrue(result.valid, result.diagnostics)
        np.testing.assert_allclose(result.qdot_safe, [1.0, 0.0], atol=2e-7)
        self.assertEqual(result.diagnostics["input_constraint_count"], 2)
        self.assertGreater(result.diagnostics["timing"]["total_wall_seconds"], 0.0)


if __name__ == "__main__":
    unittest.main()
