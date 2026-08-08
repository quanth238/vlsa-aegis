from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
NUMPY_AVAILABLE = importlib.util.find_spec("numpy") is not None
SCIPY_AVAILABLE = importlib.util.find_spec("scipy") is not None


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

    def test_partitioned_config_keeps_released_ee_and_refines_l5_l7(self) -> None:
        from main.multilink_ellipsoid.shadow import load_shadow_config

        config = load_shadow_config(
            ROOT / "configs/vlsa_distal_partitioned_ellipsoid_shadow_e05.v3.json"
        )

        self.assertEqual(
            config["robot_geometry"]["part_counts"],
            {"robot0_link5": 3, "robot0_link6": 2, "robot0_link7": 2},
        )
        self.assertEqual(
            config["end_effector_geometry"],
            {
                "center_and_orientation": (
                    "authoritative_robot0_grip_site_pose_plus_released_"
                    "minus_0.08m_local_z_offset"
                ),
                "semiaxes_m": [0.06, 0.12, 0.11],
                "source": "released_aegis_end_effector_proxy",
            },
        )

    @unittest.skipUnless(
        NUMPY_AVAILABLE and SCIPY_AVAILABLE,
        "NumPy and SciPy are optional for the structural gate",
    )
    def test_partitioned_cube_union_certifiably_contains_convex_hull(self) -> None:
        import numpy as np

        from main.multilink_ellipsoid.geometry import (
            partitioned_convex_hull_enclosing_ellipsoids,
        )

        points = np.asarray(
            [
                [x, y, z]
                for x in (-2.0, 2.0)
                for y in (-1.0, 1.0)
                for z in (-0.5, 0.5)
            ],
            dtype=np.float64,
        )
        parts = partitioned_convex_hull_enclosing_ellipsoids(
            points,
            part_count=2,
            body_name="robot0_link5",
            geom_name="robot0_link5_collision",
            source_body_names=("robot0_link5",),
            source_geom_names=("robot0_link5_collision",),
        )

        self.assertEqual(len(parts), 2)
        assigned = []
        for part in parts:
            certificate = part.enclosure_certificate
            self.assertTrue(certificate["partition_cell_contained"])
            self.assertTrue(certificate["convex_hull_contained_by_partition_union"])
            assigned.extend(certificate["partition_face_indices"])
        self.assertEqual(sorted(assigned), list(range(len(set(assigned)))))
        grid = np.asarray(
            [
                [x, y, z]
                for x in np.linspace(-2.0, 2.0, 9)
                for y in np.linspace(-1.0, 1.0, 7)
                for z in np.linspace(-0.5, 0.5, 5)
            ]
        )
        covered = np.zeros(len(grid), dtype=bool)
        for part in parts:
            local = (grid - part.center) @ part.rotation
            covered |= np.sum((local / part.semiaxes_m) ** 2, axis=1) <= 1.0 + 1e-10
        self.assertTrue(np.all(covered))

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

    def test_rollout_multicbf_config_uses_three_discrete_constraints(self) -> None:
        from main.multilink_ellipsoid.rollout import load_rollout_config

        config = load_rollout_config(
            ROOT
            / "configs/vlsa_distal_three_ellipsoid_rollout_multicbf_e05.v1.json"
        )
        self.assertEqual(
            config["protected_body_names"],
            ["robot0_link5", "robot0_link6", "robot0_link7"],
        )
        self.assertEqual(config["finite_difference"]["action_dimensions"], [0, 1, 2])
        self.assertEqual(
            config["verification"]["candidate_scales_toward_stop"],
            [1.0, 0.5, 0.25, 0.125, 0.0],
        )
        self.assertEqual(config["optimizer"]["next_step_minimum_h_opt_m"], 0.0)
        self.assertTrue(config["simulator_verification"]["distinct_from_D_opt"])

    def test_predictive_flow_config_freezes_four_bodies_and_ten_steps(self) -> None:
        from main.multilink_ellipsoid.predictive_flow import (
            load_predictive_flow_config,
        )

        config = load_predictive_flow_config(
            ROOT / "configs/vlsa_predictive_flow_guidance_e05.v1.json"
        )
        self.assertEqual(
            config["protected_body_names"],
            [
                "robot0_link5",
                "robot0_link6",
                "robot0_link7",
                "robot0_end_effector",
            ],
        )
        self.assertEqual(config["trajectory_model"]["horizon_steps"], 10)
        self.assertEqual(config["flow_guidance"]["barrier_decay_gamma"], 0.9)
        self.assertEqual(
            config["flow_guidance"]["dykstra_projection_sweeps_per_euler_step"],
            64,
        )

    def test_embodisteer_config_freezes_paper_mapping_and_surrogate_scope(self) -> None:
        from main.multilink_ellipsoid.embodisteer_flow import (
            load_embodisteer_flow_config,
        )

        config = load_embodisteer_flow_config(
            ROOT / "configs/vlsa_embodisteer_multicbf_e05.v1.json"
        )
        self.assertEqual(config["paper_reference"]["arxiv"], "2606.12965")
        self.assertEqual(
            config["paper_reference"]["fidelity"],
            "osc_action_space_surrogate_not_full_joint_space_embodisteer",
        )
        self.assertEqual(len(config["protected_body_names"]), 4)
        self.assertEqual(
            config["embodisteer_guidance"]["guidance_schedule"],
            {"base_strength": 1.0, "beta": 50.0, "transition": 0.7},
        )

    @unittest.skipUnless(NUMPY_AVAILABLE, "NumPy is optional locally")
    def test_embodisteer_builder_extends_all_rows_with_task_metric(self) -> None:
        import numpy as np

        from main.multilink_ellipsoid.embodisteer_flow import (
            build_embodisteer_guidance_envelope,
        )

        center = np.zeros((10, 7), dtype=np.float64)
        clearance_jacobian = np.zeros((10, 4, 30), dtype=np.float64)
        for step in range(10):
            for body in range(4):
                clearance_jacobian[step, body, step * 3 + body % 3] = (
                    0.01 * (body + 1)
                )
        model = {
            "center_actions": center,
            "base": {"h_opt_m": np.full((10, 4), 0.035)},
            "jacobian": clearance_jacobian,
            "eef_jacobian": np.eye(30).reshape(10, 3, 30),
        }
        envelope, record = build_embodisteer_guidance_envelope(
            model,
            center,
            np.full(4, 0.04),
            gamma=0.9,
            action_limit=1.0,
            projection_tolerance=5.0e-5,
            joint_regularization_lambda=0.01,
            position_weight=1.0,
            guidance_schedule={
                "base_strength": 1.0,
                "beta": 50.0,
                "transition": 0.7,
            },
        )
        self.assertEqual(len(envelope["delta_rows"]), 100)
        self.assertEqual(len(envelope["task_metric_directions"]), 100)
        self.assertEqual(record["trajectory_constraint_definition_count"], 40)
        self.assertTrue(
            record["paper_mapping"]["single_to_multi_constraint_extension"]
        )
        self.assertAlmostEqual(
            envelope["task_metric_condition_number"], 1.0, places=10
        )

    @unittest.skipUnless(NUMPY_AVAILABLE, "NumPy is optional locally")
    def test_predictive_flow_builds_40_cbf_definitions_and_60_bounds(self) -> None:
        import numpy as np

        from main.multilink_ellipsoid.predictive_flow import (
            build_flow_guidance_envelope,
            exact_trajectory_verification,
        )

        current = np.full(4, 0.04, dtype=np.float64)
        future = np.full((10, 4), 0.035, dtype=np.float64)
        jacobian = np.zeros((10, 4, 30), dtype=np.float64)
        for step in range(10):
            for body in range(4):
                jacobian[step, body, step * 3 + body % 3] = 0.01 * (body + 1)
        center = np.zeros((10, 7), dtype=np.float64)
        model = {
            "center_actions": center,
            "base": {"h_opt_m": future},
            "jacobian": jacobian,
        }
        envelope, record = build_flow_guidance_envelope(
            model,
            center,
            current,
            gamma=0.9,
            action_limit=1.0,
            projection_tolerance=5.0e-5,
        )
        self.assertEqual(record["trajectory_constraint_definition_count"], 40)
        self.assertEqual(record["trajectory_projection_row_count"], 40)
        self.assertEqual(record["action_bound_row_count"], 60)
        self.assertEqual(record["projection_row_count"], 100)
        self.assertEqual(len(envelope["delta_rows"]), 100)
        verification = exact_trajectory_verification(
            current,
            {
                "h_opt_m": future,
                "next_state_sha256": ["0" * 64] * 10,
                "synchronization": {"maximum_absolute_error": 0.0},
                "total_env_step_wall_seconds": 0.1,
            },
            gamma=0.9,
            tolerance_m=1.0e-6,
        )
        self.assertTrue(verification["safe"])
        self.assertAlmostEqual(
            verification["minimum_trajectory_cbf_residual_m"], 0.031
        )

    @unittest.skipUnless(NUMPY_AVAILABLE, "NumPy is optional locally")
    def test_discrete_clearance_rows_recover_coupled_linear_transition(self) -> None:
        import numpy as np

        from main.multilink_ellipsoid.rollout import (
            discrete_qp_lower_bounds,
            finite_difference_clearance_jacobian,
        )

        nominal = np.array([0.2, -0.1, 0.4])
        expected = np.array(
            [[0.03, -0.02, 0.01], [-0.01, 0.04, 0.02], [0.02, 0.01, -0.05]]
        )
        base = np.array([0.004, 0.006, 0.008])
        plus_xyz = []
        minus_xyz = []
        plus_h = []
        minus_h = []
        for dimension in range(3):
            plus = nominal.copy()
            minus = nominal.copy()
            plus[dimension] += 0.05
            minus[dimension] -= 0.05
            plus_xyz.append(plus)
            minus_xyz.append(minus)
            plus_h.append(base + expected @ (plus - nominal))
            minus_h.append(base + expected @ (minus - nominal))
        observed = finite_difference_clearance_jacobian(
            plus_h, minus_h, plus_xyz, minus_xyz
        )
        np.testing.assert_allclose(observed, expected, rtol=0.0, atol=1.0e-14)
        lower = discrete_qp_lower_bounds(
            base, observed, nominal, minimum_next_clearance=0.0
        )
        np.testing.assert_allclose(lower, -base + expected @ nominal)

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
