from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEPENDENCIES_PRESENT = all(
    importlib.util.find_spec(name) is not None for name in ("numpy", "mujoco")
)


@unittest.skipUnless(
    DEPENDENCIES_PRESENT, "MuJoCo/NumPy allocation dependencies unavailable"
)
class FullRobotMeasurementTests(unittest.TestCase):
    XML = r"""
<mujoco model="full_robot_measurement_test">
  <option timestep="0.002" gravity="0 0 0"/>
  <worldbody>
    <body name="robot_root" pos="0 0 0">
      {robot_joint}
      <geom name="robot_visual_only" type="sphere" size="0.01"
            contype="0" conaffinity="0" rgba="0 1 0 1"/>
      <body name="link5" pos="0 0 0">
        <geom name="link5_collision" type="box" size="0.05 0.05 0.05"
              mass="0.2" contype="1" conaffinity="1"/>
        <body name="link6" pos="0 0 0.30">
          <geom name="link6_collision" type="box" size="0.05 0.05 0.05"
                mass="0.2" contype="1" conaffinity="1"/>
          <body name="gripper" pos="0 0 0.30">
            <geom name="gripper_collision" type="box" size="0.04 0.04 0.04"
                  mass="0.2" contype="1" conaffinity="1"/>
          </body>
        </body>
      </body>
    </body>
    <body name="selected_obstacle_root" pos="{obstacle_x} 0 {obstacle_z}">
      {obstacle_joint}
      <body name="selected_obstacle_piece">
        <geom name="selected_obstacle_collision" type="box"
              size="0.05 0.05 0.05" contype="1" conaffinity="1"/>
        <geom name="selected_obstacle_collision_upper" type="box" pos="0 0 0.20"
              size="0.02 0.02 0.02" contype="1" conaffinity="1"/>
        <geom name="selected_obstacle_visual_only" type="sphere" size="0.01"
              contype="0" conaffinity="0" rgba="0 0 1 1"/>
      </body>
    </body>
    <body name="other_obstacle" pos="-0.2 0 0">
      <geom name="other_obstacle_collision" type="box" size="0.02 0.02 0.02"/>
    </body>
  </worldbody>
</mujoco>
"""

    def setUp(self) -> None:
        import mujoco
        import numpy as np

        from main.poisson_fullbody.measurement import (
            FullRobotObstacleMonitor,
            SettledObstacleMotionInadmissible,
            StaticObstacleDriftInadmissible,
            clone_forwarded_state,
            copy_integration_state,
            resolve_collision_geom_sets,
        )
        from main.poisson_fullbody.robot_samples import BodySample

        self.mujoco = mujoco
        self.np = np
        self.FullRobotObstacleMonitor = FullRobotObstacleMonitor
        self.SettledObstacleMotionInadmissible = (
            SettledObstacleMotionInadmissible
        )
        self.StaticObstacleDriftInadmissible = StaticObstacleDriftInadmissible
        self.clone_forwarded_state = clone_forwarded_state
        self.copy_integration_state = copy_integration_state
        self.resolve_collision_geom_sets = resolve_collision_geom_sets
        self.BodySample = BodySample

    def _id(self, model, object_type, name):
        identity = int(self.mujoco.mj_name2id(model, object_type, name))
        self.assertGreaterEqual(identity, 0, name)
        return identity

    def _build(
        self,
        *,
        obstacle_x=0.50,
        obstacle_z=0.0,
        free_robot=False,
        free_obstacle=False,
    ):
        xml = self.XML.format(
            robot_joint="<freejoint/>" if free_robot else "",
            obstacle_x=obstacle_x,
            obstacle_z=obstacle_z,
            obstacle_joint="<freejoint/>" if free_obstacle else "",
        )
        model = self.mujoco.MjModel.from_xml_string(xml)
        data = self.mujoco.MjData(model)
        self.mujoco.mj_forward(model, data)
        sim = SimpleNamespace(model=model, data=data)
        body = self.mujoco.mjtObj.mjOBJ_BODY
        geom = self.mujoco.mjtObj.mjOBJ_GEOM
        ids = {
            "robot_root": self._id(model, body, "robot_root"),
            "link5": self._id(model, body, "link5"),
            "link6": self._id(model, body, "link6"),
            "gripper": self._id(model, body, "gripper"),
            "obstacle_root": self._id(model, body, "selected_obstacle_root"),
            "obstacle_piece": self._id(model, body, "selected_obstacle_piece"),
            "other_obstacle": self._id(model, body, "other_obstacle"),
            "link5_geom": self._id(model, geom, "link5_collision"),
            "link6_geom": self._id(model, geom, "link6_collision"),
            "gripper_geom": self._id(model, geom, "gripper_collision"),
            "robot_visual": self._id(model, geom, "robot_visual_only"),
            "obstacle_geom": self._id(model, geom, "selected_obstacle_collision"),
            "obstacle_upper_geom": self._id(
                model, geom, "selected_obstacle_collision_upper"
            ),
            "obstacle_visual": self._id(
                model, geom, "selected_obstacle_visual_only"
            ),
            "other_obstacle_geom": self._id(
                model, geom, "other_obstacle_collision"
            ),
        }
        resolved = self.resolve_collision_geom_sets(
            model,
            robot_root_body_ids=[ids["robot_root"]],
            obstacle_root_body_ids=[ids["obstacle_root"]],
            link56_body_ids=[ids["link5"], ids["link6"]],
        )
        resolved_from_sim = self.resolve_collision_geom_sets(
            sim,
            robot_root_body_ids=[ids["robot_root"]],
            obstacle_root_body_ids=[ids["obstacle_root"]],
            link56_body_ids=[ids["link5"], ids["link6"]],
        )
        self.assertEqual(resolved_from_sim, resolved)
        samples = (
            self.BodySample(
                0,
                ids["link5"],
                "link5",
                ids["link5_geom"],
                "link5_collision",
                (-0.05, 0.0, 0.0),
            ),
            self.BodySample(
                1,
                ids["link6"],
                "link6",
                ids["link6_geom"],
                "link6_collision",
                (-0.05, 0.0, 0.0),
            ),
            self.BodySample(
                2,
                ids["gripper"],
                "gripper",
                ids["gripper_geom"],
                "gripper_collision",
                (-0.04, 0.0, 0.0),
            ),
        )
        return model, data, sim, ids, resolved, samples

    def _moving_obstacle_monitor(
        self,
        *,
        linear_velocity=(0.0, 0.0, 0.0),
        angular_velocity=(0.0, 0.0, 0.0),
        max_linear_speed=0.0,
        max_angular_speed=0.0,
    ):
        model, data, sim, ids, resolved, samples = self._build(
            free_obstacle=True
        )
        data.qvel[:3] = linear_velocity
        data.qvel[3:6] = angular_velocity
        self.mujoco.mj_forward(model, data)
        monitor = self.FullRobotObstacleMonitor(
            sim,
            resolved,
            samples,
            certified_coverage_radius_m=0.02,
            max_selected_geom_surface_drift_m=1e-6,
            max_settled_obstacle_linear_speed_m_per_s=max_linear_speed,
            max_settled_obstacle_angular_speed_rad_per_s=max_angular_speed,
        )
        return monitor, ids

    def test_authoritative_body_resolution_excludes_visual_and_other_objects(self):
        model, _, _, ids, resolved, _ = self._build()
        self.assertEqual(resolved.robot_root_body_ids, (ids["robot_root"],))
        self.assertEqual(
            set(resolved.robot_body_ids),
            {ids["robot_root"], ids["link5"], ids["link6"], ids["gripper"]},
        )
        self.assertEqual(
            set(resolved.obstacle_body_ids),
            {ids["obstacle_root"], ids["obstacle_piece"]},
        )
        self.assertNotIn(ids["robot_visual"], resolved.robot_geom_ids)
        self.assertNotIn(ids["obstacle_visual"], resolved.obstacle_geom_ids)
        self.assertNotIn(ids["other_obstacle_geom"], resolved.obstacle_geom_ids)
        self.assertEqual(
            set(resolved.obstacle_geom_ids),
            {ids["obstacle_geom"], ids["obstacle_upper_geom"]},
        )
        self.assertEqual(
            set(resolved.link56_geom_ids),
            {ids["link5_geom"], ids["link6_geom"]},
        )
        # Descendants of link6 remain robot geoms but are not fabricated as
        # literal link-5/6 contact truth.
        self.assertIn(ids["gripper_geom"], resolved.robot_geom_ids)
        self.assertNotIn(ids["gripper_geom"], resolved.link56_geom_ids)
        expected_pairs = {
            (ids["link5_geom"], ids["obstacle_geom"]),
            (ids["link5_geom"], ids["obstacle_upper_geom"]),
            (ids["link6_geom"], ids["obstacle_geom"]),
            (ids["link6_geom"], ids["obstacle_upper_geom"]),
            (ids["gripper_geom"], ids["obstacle_geom"]),
            (ids["gripper_geom"], ids["obstacle_upper_geom"]),
        }
        self.assertEqual(set(resolved.collision_enabled_pairs), expected_pairs)
        self.assertEqual(
            resolved.robot_geom_names,
            tuple(
                self.mujoco.mj_id2name(
                    model, self.mujoco.mjtObj.mjOBJ_GEOM, geom_id
                )
                for geom_id in resolved.robot_geom_ids
            ),
        )

    def test_positive_clearance_and_settled_pose_drift_are_separate(self):
        model, data, sim, ids, resolved, samples = self._build(obstacle_x=0.50)
        monitor = self.FullRobotObstacleMonitor(
            sim,
            resolved,
            samples,
            certified_coverage_radius_m=0.02,
            max_selected_geom_surface_drift_m=1e-6,
        )
        motion = monitor.settled_state.obstacle_motion_admissibility
        self.assertTrue(motion.admissible)
        self.assertEqual(motion.reasons, ())
        self.assertEqual(motion.maximum_observed_linear_speed_m_per_s, 0.0)
        self.assertEqual(motion.maximum_observed_angular_speed_rad_per_s, 0.0)
        self.assertEqual(
            [record.body_id for record in motion.bodies],
            sorted(set(resolved.obstacle_body_ids)),
        )
        self.assertEqual(
            [record.body_name for record in motion.bodies],
            ["selected_obstacle_root", "selected_obstacle_piece"],
        )
        monitor.make_substep_callback(0, 0)(sim, 0)

        # Move/rotate the fixed synthetic obstacle after the settled reference.
        model.body_pos[ids["obstacle_root"], 0] += 0.02
        angle = 0.10
        model.body_quat[ids["obstacle_root"]] = [
            math.cos(angle / 2.0),
            0.0,
            0.0,
            math.sin(angle / 2.0),
        ]
        self.mujoco.mj_forward(model, data)
        with self.assertRaisesRegex(
            self.StaticObstacleDriftInadmissible,
            "terminate before another physics substep",
        ):
            monitor.make_substep_callback(0, 0)(sim, 1)
        result = monitor.result()

        self.assertEqual(result.observed_physics_substeps, 2)
        self.assertTrue(result.obstacle_pose_drift.surface_drift_threshold_crossed)
        self.assertEqual(
            result.obstacle_pose_drift.first_surface_drift_threshold_crossing_observation_index,
            1,
        )
        self.assertFalse(result.any_robot_obstacle_contact)
        self.assertFalse(result.link56_obstacle_contact)
        self.assertEqual(result.rollout_phase_physical_contact_point_record_count, 0)
        self.assertEqual(result.settled_state.physical_contact_point_record_count, 0)
        self.assertAlmostEqual(
            result.sample_clearance.minimum_exact_sample_to_obb_distance_m,
            0.50,
            places=10,
        )
        self.assertAlmostEqual(
            result.sample_clearance.full_surface_clearance_lower_bound_m,
            0.48,
            places=10,
        )
        self.assertAlmostEqual(result.D_sim_min_m, 0.48, places=10)
        self.assertTrue(result.raw_mj_geom_distance.available)
        self.assertEqual(
            result.raw_mj_geom_distance.authority,
            "advisory_only_never_contact_authority",
        )
        self.assertAlmostEqual(
            result.obstacle_pose_drift.maximum_translation_m, 0.02, places=10
        )
        self.assertAlmostEqual(
            result.obstacle_pose_drift.maximum_rotation_rad, angle, places=10
        )
        self.assertGreater(
            result.obstacle_pose_drift.maximum_surface_point_displacement_m,
            result.obstacle_pose_drift.maximum_translation_m,
        )
        self.assertTrue(result.obstacle_pose_drift.surface_drift_threshold_crossed)
        self.assertEqual(
            result.obstacle_pose_drift.first_surface_drift_threshold_crossing_observation_index,
            1,
        )
        self.assertEqual(
            {record.geom_id for record in result.obstacle_pose_drift.geoms},
            {ids["obstacle_geom"], ids["obstacle_upper_geom"]},
        )
        for record in result.obstacle_pose_drift.geoms:
            self.assertAlmostEqual(record.maximum_translation_m, 0.02, places=10)
            self.assertAlmostEqual(record.maximum_rotation_rad, angle, places=10)
            self.assertGreater(
                record.maximum_surface_point_displacement_m,
                record.maximum_translation_m,
            )
        # The typed payload is finite strict JSON, including all provenance.
        json.dumps(result.to_dict(), allow_nan=False)

    def test_gripper_contact_is_not_relabelled_as_literal_link56_contact(self):
        model, data, sim, ids, resolved, samples = self._build(
            obstacle_x=0.09, obstacle_z=0.60, free_robot=True
        )
        monitor = self.FullRobotObstacleMonitor(
            sim,
            resolved,
            samples,
            certified_coverage_radius_m=0.001,
            max_selected_geom_surface_drift_m=1e-6,
        )
        self.mujoco.mj_step(model, data)
        monitor.make_substep_callback(0, 0)(sim, 0)
        result = monitor.result()
        self.assertTrue(result.any_robot_obstacle_contact)
        self.assertFalse(result.link56_obstacle_contact)
        self.assertTrue(
            all(
                event.robot_geom_id == ids["gripper_geom"]
                for event in result.post_state_physical_contact_point_records
            )
        )

    def test_static_translation_and_rotation_thresholds_are_independent(self):
        for mode in ("translation", "rotation"):
            with self.subTest(mode=mode):
                model, data, sim, ids, resolved, samples = self._build(
                    obstacle_x=0.50
                )
                monitor = self.FullRobotObstacleMonitor(
                    sim,
                    resolved,
                    samples,
                    certified_coverage_radius_m=0.02,
                    max_selected_geom_surface_drift_m=1.0,
                    max_selected_geom_translation_drift_m=(
                        1e-6 if mode == "translation" else 1.0
                    ),
                    max_selected_geom_rotation_drift_rad=(
                        1e-6 if mode == "rotation" else 1.0
                    ),
                )
                if mode == "translation":
                    model.body_pos[ids["obstacle_root"], 0] += 2e-6
                else:
                    angle = 2e-6
                    model.body_quat[ids["obstacle_root"]] = [
                        math.cos(angle / 2.0),
                        0.0,
                        0.0,
                        math.sin(angle / 2.0),
                    ]
                self.mujoco.mj_forward(model, data)
                with self.assertRaisesRegex(
                    self.StaticObstacleDriftInadmissible, mode
                ):
                    monitor.make_substep_callback(0, 0)(sim, 0)

    def test_adapter_diagnostic_mode_records_drift_without_terminating(self):
        model, data, sim, ids, resolved, samples = self._build(obstacle_x=0.50)
        monitor = self.FullRobotObstacleMonitor(
            sim,
            resolved,
            samples,
            certified_coverage_radius_m=0.02,
            max_selected_geom_surface_drift_m=1e-6,
            max_selected_geom_translation_drift_m=1e-6,
            max_selected_geom_rotation_drift_rad=1e-5,
            terminate_on_static_drift=False,
        )
        model.body_pos[ids["obstacle_root"], 0] += 0.02
        self.mujoco.mj_forward(model, data)
        monitor.make_substep_callback(0, 0)(sim, 0)
        result = monitor.result()
        self.assertTrue(result.obstacle_pose_drift.surface_drift_threshold_crossed)
        self.assertAlmostEqual(
            result.obstacle_pose_drift.maximum_translation_m, 0.02, places=10
        )

    def test_contact_is_authority_and_cannot_leave_D_sim_positive(self):
        model, data, sim, ids, resolved, samples = self._build(
            obstacle_x=0.09, free_robot=True
        )
        monitor = self.FullRobotObstacleMonitor(
            sim,
            resolved,
            samples,
            certified_coverage_radius_m=0.001,
            max_selected_geom_surface_drift_m=1e-6,
        )
        self.mujoco.mj_step(model, data)
        monitor.observe_post_integration(
            sim,
            high_level_index=0,
            inner_control_index=0,
            physics_substep_index=0,
        )
        result = monitor.result()

        self.assertTrue(result.any_robot_obstacle_contact)
        self.assertTrue(result.link56_obstacle_contact)
        self.assertGreater(result.rollout_phase_physical_contact_point_record_count, 0)
        self.assertGreaterEqual(
            result.post_state_candidate_contact_point_record_count,
            result.post_state_physical_contact_point_record_count,
        )
        self.assertEqual(
            result.total_physical_contact_point_record_count,
            result.settled_state.physical_contact_point_record_count
            + result.rollout_phase_physical_contact_point_record_count,
        )
        self.assertEqual(
            result.total_candidate_contact_point_record_count,
            result.settled_state.candidate_contact_point_record_count
            + result.live_solver_candidate_contact_point_record_count
            + result.post_state_candidate_contact_point_record_count,
        )
        self.assertLessEqual(result.D_sim_min_m, 0.0)
        self.assertTrue(result.contact_authority_clamped_D_sim)
        first = result.first_physical_contact_point_record
        self.assertIsNotNone(first)
        self.assertEqual(first.source_phase, "settled_post_integration_recomputed")
        self.assertIsNone(first.high_level_index)
        rollout_first = result.first_post_state_physical_contact_point_record
        self.assertEqual(rollout_first.high_level_index, 0)
        self.assertEqual(rollout_first.inner_control_index, 0)
        self.assertEqual(rollout_first.physics_substep_index, 0)
        self.assertEqual(first.robot_geom_id, ids["link5_geom"])
        self.assertEqual(first.obstacle_geom_id, ids["obstacle_geom"])
        self.assertEqual(first.robot_body_name, "link5")
        self.assertEqual(first.obstacle_body_name, "selected_obstacle_piece")
        self.assertEqual(len(first.position_world_m), 3)
        self.assertAlmostEqual(
            float(
                self.np.linalg.norm(
                    first.frame_normal_mujoco_geom1_to_geom2_world
                )
            ),
            1.0,
            places=10,
        )
        self.assertTrue(first.is_physical_nonpositive_distance_contact)
        self.assertLessEqual(first.contact_distance_m, 0.0)
        self.assertFalse(first.force_available)
        self.assertEqual(first.force_semantics, "omitted_recomputed_not_applied")
        self.assertIsNotNone(first.force_unavailable_reason)
        live_force_records = [
            record
            for record in result.live_solver_phase_contact_point_records
            if record.force_available
        ]
        self.assertTrue(live_force_records)
        for record in live_force_records:
            self.assertIn("live_solver_constraint_wrench", record.force_semantics)
            self.assertEqual(len(record.force_contact_frame_n), 6)
            self.assertEqual(len(record.force_world_n), 3)
            self.assertEqual(len(record.impulse_estimate_contact_frame_ns), 6)
            self.assertEqual(len(record.impulse_estimate_world_ns), 3)
        json.dumps(result.to_dict(), allow_nan=False)

    def test_settled_physical_contact_survives_when_absent_at_first_rollout_record(self):
        _, data, sim, _, resolved, samples = self._build(
            obstacle_x=0.09, free_robot=True
        )
        monitor = self.FullRobotObstacleMonitor(
            sim,
            resolved,
            samples,
            certified_coverage_radius_m=0.001,
            max_selected_geom_surface_drift_m=1e-6,
        )
        self.assertGreater(
            monitor.settled_state.physical_contact_point_record_count, 0
        )
        # Change only integration state.  Live contact/kinematic arrays still
        # describe the old solver phase, while the forwarded clone is clear.
        data.qpos[0] = -1.0
        data.qvel[:] = 0.0
        monitor.make_substep_callback(0, 0)(sim, 0)
        result = monitor.result()
        self.assertTrue(result.any_robot_obstacle_contact)
        self.assertTrue(result.rollout_any_robot_obstacle_contact)
        self.assertFalse(result.post_state_any_robot_obstacle_contact)
        self.assertEqual(result.post_state_physical_contact_point_record_count, 0)
        self.assertGreater(result.live_solver_nonpositive_contact_point_record_count, 0)
        self.assertEqual(
            result.first_physical_contact_point_record.source_phase,
            "settled_post_integration_recomputed",
        )
        self.assertLessEqual(result.D_sim_min_m, 0.0)

    def test_live_solver_transient_contact_enters_union_when_post_state_is_clear(self):
        model = self.mujoco.MjModel.from_xml_string(
            r"""
<mujoco>
  <option timestep="0.002" gravity="0 0 0"/>
  <worldbody>
    <body name="robot_root">
      <freejoint/>
      <body name="link5">
        <geom name="robot_box" type="box" size="0.05 0.05 0.05" mass="1"/>
      </body>
    </body>
    <body name="obstacle_root" pos="0.20 0 0">
      <geom name="obstacle_box" type="box" size="0.05 0.05 0.05"/>
    </body>
  </worldbody>
</mujoco>
"""
        )
        data = self.mujoco.MjData(model)
        self.mujoco.mj_forward(model, data)
        sim = SimpleNamespace(model=model, data=data)
        body = self.mujoco.mjtObj.mjOBJ_BODY
        geom = self.mujoco.mjtObj.mjOBJ_GEOM
        robot_root = self._id(model, body, "robot_root")
        link5 = self._id(model, body, "link5")
        obstacle_root = self._id(model, body, "obstacle_root")
        robot_geom = self._id(model, geom, "robot_box")
        obstacle_geom = self._id(model, geom, "obstacle_box")
        resolved = self.resolve_collision_geom_sets(
            model,
            robot_root_body_ids=[robot_root],
            obstacle_root_body_ids=[obstacle_root],
            link56_body_ids=[link5],
        )
        samples = (
            self.BodySample(
                0,
                link5,
                "link5",
                robot_geom,
                "robot_box",
                (0.05, 0.0, 0.0),
            ),
        )
        monitor = self.FullRobotObstacleMonitor(
            sim,
            resolved,
            samples,
            certified_coverage_radius_m=0.0,
            max_selected_geom_surface_drift_m=1e-6,
        )
        self.assertFalse(monitor.settled_state.any_robot_obstacle_contact)

        # Establish a physical pre-integration contact, then move quickly away
        # in one real mj_step.  MuJoCo retains that applied solver phase while
        # post-integration qpos is already clear.
        data.qpos[0] = 0.11
        data.qvel[0] = -100.0
        self.mujoco.mj_forward(model, data)
        self.mujoco.mj_step(model, data)
        self.assertGreater(int(data.ncon), 0)
        self.assertTrue(
            any(float(data.contact[index].dist) <= 0.0 for index in range(data.ncon))
        )
        monitor.make_substep_callback(0, 0)(sim, 0)
        result = monitor.result()

        self.assertTrue(result.live_solver_any_robot_obstacle_contact)
        self.assertFalse(result.post_state_any_robot_obstacle_contact)
        self.assertTrue(result.rollout_any_robot_obstacle_contact)
        self.assertTrue(result.any_robot_obstacle_contact)
        self.assertEqual(result.post_state_physical_contact_point_record_count, 0)
        self.assertGreater(
            result.live_solver_nonpositive_contact_point_record_count, 0
        )
        self.assertEqual(
            result.first_physical_contact_point_record.source_phase,
            "live_solver_phase_preintegration_geometry",
        )
        self.assertLessEqual(result.D_sim_min_m, 0.0)

    def test_zero_substeps_and_callback_gaps_fail_closed(self):
        _, _, sim, _, resolved, samples = self._build()
        monitor = self.FullRobotObstacleMonitor(
            sim,
            resolved,
            samples,
            certified_coverage_radius_m=0.02,
            max_selected_geom_surface_drift_m=1e-6,
        )
        with self.assertRaisesRegex(RuntimeError, "No post-integration"):
            monitor.result()
        with self.assertRaisesRegex(RuntimeError, "gap/duplicate"):
            monitor.observe_post_integration(
                sim,
                high_level_index=0,
                inner_control_index=0,
                physics_substep_index=1,
            )

    def test_integration_state_clone_is_complete_and_does_not_mutate_live_data(self):
        model = self.mujoco.MjModel.from_xml_string(
            r"""
<mujoco>
  <size nuserdata="2"/>
  <worldbody>
    <body name="marker" mocap="true" pos="0 0 1">
      <geom type="sphere" size="0.01"/>
    </body>
    <body name="controlled">
      <joint name="joint" type="hinge"/>
      <geom type="capsule" size="0.02 0.10" mass="1"/>
    </body>
  </worldbody>
  <actuator>
    <general joint="joint" dyntype="filter" dynprm="1"/>
  </actuator>
</mujoco>
"""
        )
        data = self.mujoco.MjData(model)
        data.time = 0.25
        data.qpos[:] = [0.2]
        data.qvel[:] = [-0.3]
        data.act[:] = [0.4]
        data.qacc_warmstart[:] = [0.5]
        data.ctrl[:] = [0.6]
        data.qfrc_applied[:] = [0.7]
        data.xfrc_applied[2, :] = [1, 2, 3, 4, 5, 6]
        data.mocap_pos[0, :] = [0.1, 0.2, 1.3]
        data.mocap_quat[0, :] = [1.0, 0.0, 0.0, 0.0]
        data.userdata[:] = [8.0, 9.0]

        fields = (
            "qpos",
            "qvel",
            "act",
            "qacc_warmstart",
            "ctrl",
            "qfrc_applied",
            "xfrc_applied",
            "mocap_pos",
            "mocap_quat",
            "userdata",
        )
        before = {field: getattr(data, field).copy() for field in fields}
        before_time = float(data.time)
        copied = self.copy_integration_state(model, data)
        specification = int(self.mujoco.mjtState.mjSTATE_INTEGRATION)
        live_state = self.np.empty(
            self.mujoco.mj_stateSize(model, specification), dtype=self.np.float64
        )
        copied_state = self.np.empty_like(live_state)
        self.mujoco.mj_getState(model, data, live_state, specification)
        self.mujoco.mj_getState(model, copied, copied_state, specification)
        self.np.testing.assert_array_equal(copied_state, live_state)

        clone = self.clone_forwarded_state(model, data)

        self.assertIsNot(clone, data)
        self.assertEqual(float(clone.time), before_time)
        for field in fields:
            self.np.testing.assert_array_equal(getattr(data, field), before[field])
        self.assertEqual(float(data.time), before_time)
        for field in (
            "qpos",
            "qvel",
            "act",
            "ctrl",
            "qfrc_applied",
            "xfrc_applied",
            "mocap_pos",
            "mocap_quat",
            "userdata",
        ):
            self.np.testing.assert_array_equal(getattr(clone, field), before[field])

    def test_forwarded_clone_fixes_stale_post_step_kinematics_and_contact(self):
        model = self.mujoco.MjModel.from_xml_string(
            r"""
<mujoco>
  <option timestep="0.002" gravity="0 0 0"/>
  <worldbody>
    <body name="robot_root">
      <freejoint/>
      <body name="link5">
        <geom name="moving_robot_box" type="box" size="0.05 0.05 0.05"
              mass="1" contype="1" conaffinity="1"/>
      </body>
    </body>
    <body name="obstacle_root" pos="0.101 0 0">
      <geom name="obstacle_box" type="box" size="0.05 0.05 0.05"
            contype="1" conaffinity="1"/>
    </body>
  </worldbody>
</mujoco>
"""
        )
        data = self.mujoco.MjData(model)
        data.qvel[0] = 1.0
        self.mujoco.mj_forward(model, data)
        sim = SimpleNamespace(model=model, data=data)
        body = self.mujoco.mjtObj.mjOBJ_BODY
        geom = self.mujoco.mjtObj.mjOBJ_GEOM
        robot_root = self._id(model, body, "robot_root")
        link5 = self._id(model, body, "link5")
        obstacle_root = self._id(model, body, "obstacle_root")
        robot_geom = self._id(model, geom, "moving_robot_box")
        obstacle_geom = self._id(model, geom, "obstacle_box")
        resolved = self.resolve_collision_geom_sets(
            model,
            robot_root_body_ids=[robot_root],
            obstacle_root_body_ids=[obstacle_root],
            link56_body_ids=[link5],
        )
        samples = (
            self.BodySample(
                0,
                link5,
                "link5",
                robot_geom,
                "moving_robot_box",
                (0.05, 0.0, 0.0),
            ),
        )
        monitor = self.FullRobotObstacleMonitor(
            sim,
            resolved,
            samples,
            certified_coverage_radius_m=0.0,
            max_selected_geom_surface_drift_m=1e-6,
        )
        self.assertFalse(monitor.settled_state.any_robot_obstacle_contact)
        self.assertAlmostEqual(monitor.settled_state.D_sim_m, 0.001, places=12)

        self.mujoco.mj_step(model, data)
        # Euler integration updated qpos, but live geom_xpos/contact remain at
        # the force-evaluation configuration until a later forward pass.
        self.assertAlmostEqual(float(data.qpos[0]), 0.002, places=12)
        self.assertAlmostEqual(float(data.geom_xpos[robot_geom, 0]), 0.0, places=12)
        self.assertEqual(int(data.ncon), 0)
        before = {
            "qpos": data.qpos.copy(),
            "qvel": data.qvel.copy(),
            "xpos": data.xpos.copy(),
            "xmat": data.xmat.copy(),
            "geom_xpos": data.geom_xpos.copy(),
            "geom_xmat": data.geom_xmat.copy(),
        }
        before_time = float(data.time)
        monitor.make_substep_callback(0, 0)(sim, 0)
        for field, expected in before.items():
            self.np.testing.assert_array_equal(getattr(data, field), expected)
        self.assertEqual(float(data.time), before_time)
        self.assertEqual(int(data.ncon), 0)

        result = monitor.result()
        self.assertEqual(result.live_solver_candidate_contact_point_record_count, 0)
        self.assertGreater(result.post_state_candidate_contact_point_record_count, 0)
        self.assertGreater(result.post_state_physical_contact_point_record_count, 0)
        self.assertTrue(result.rollout_any_robot_obstacle_contact)
        self.assertFalse(result.settled_state.any_robot_obstacle_contact)
        self.assertLessEqual(result.D_sim_min_m, -0.001 + 1e-12)
        self.assertEqual(
            result.post_state_physical_contact_point_records[0].source_phase,
            "post_integration_recomputed",
        )

    def test_positive_margin_candidate_is_not_physical_contact_or_D_sim_clamp(self):
        model = self.mujoco.MjModel.from_xml_string(
            r"""
<mujoco>
  <option gravity="0 0 0"/>
  <worldbody>
    <body name="robot_root">
      <freejoint/>
      <body name="link5">
        <geom name="robot_box" type="box" size="0.05 0.05 0.05"
              mass="1" contype="0" conaffinity="0"/>
      </body>
    </body>
    <body name="obstacle_root" pos="0.11 0 0">
      <geom name="obstacle_box" type="box" size="0.05 0.05 0.05"
            contype="0" conaffinity="0"/>
    </body>
  </worldbody>
  <contact>
    <pair geom1="robot_box" geom2="obstacle_box" margin="0.03" gap="0.005"/>
  </contact>
</mujoco>
"""
        )
        data = self.mujoco.MjData(model)
        self.mujoco.mj_forward(model, data)
        sim = SimpleNamespace(model=model, data=data)
        body = self.mujoco.mjtObj.mjOBJ_BODY
        geom = self.mujoco.mjtObj.mjOBJ_GEOM
        robot_root = self._id(model, body, "robot_root")
        link5 = self._id(model, body, "link5")
        obstacle_root = self._id(model, body, "obstacle_root")
        robot_geom = self._id(model, geom, "robot_box")
        obstacle_geom = self._id(model, geom, "obstacle_box")
        resolved = self.resolve_collision_geom_sets(
            model,
            robot_root_body_ids=[robot_root],
            obstacle_root_body_ids=[obstacle_root],
            link56_body_ids=[link5],
        )
        self.assertEqual(resolved.robot_geom_ids, (robot_geom,))
        self.assertEqual(resolved.obstacle_geom_ids, (obstacle_geom,))
        self.assertEqual(resolved.collision_enabled_pairs, ((robot_geom, obstacle_geom),))
        samples = (
            self.BodySample(
                0,
                link5,
                "link5",
                robot_geom,
                "robot_box",
                (0.05, 0.0, 0.0),
            ),
        )
        monitor = self.FullRobotObstacleMonitor(
            sim,
            resolved,
            samples,
            certified_coverage_radius_m=0.0,
            max_selected_geom_surface_drift_m=1e-6,
            near_contact_tolerance_m=0.02,
        )
        monitor.make_substep_callback(0, 0)(sim, 0)
        result = monitor.result()

        self.assertGreater(
            result.settled_state.candidate_contact_point_record_count, 0
        )
        self.assertGreater(result.post_state_candidate_contact_point_record_count, 0)
        self.assertEqual(
            result.settled_state.physical_contact_point_record_count, 0
        )
        self.assertEqual(result.post_state_physical_contact_point_record_count, 0)
        self.assertFalse(result.any_robot_obstacle_contact)
        self.assertGreater(result.D_sim_min_m, 0.0)
        self.assertFalse(result.contact_authority_clamped_D_sim)
        event = result.first_candidate_contact_point_record
        self.assertGreater(event.contact_distance_m, 0.0)
        self.assertFalse(event.is_physical_nonpositive_distance_contact)
        self.assertTrue(event.within_registered_near_contact_tolerance)
        self.assertTrue(event.solver_constraint_active)
        self.assertEqual(event.explicit_pair_id, 0)
        self.assertAlmostEqual(event.explicit_pair_margin_m, 0.03)
        self.assertAlmostEqual(event.explicit_pair_gap_m, 0.005)
        # MuJoCo 3.2.3 stores the effective solver-activation threshold in
        # mjContact.includemargin: pair margin minus pair gap.  The raw pair
        # parameters remain separately recorded above, while physical-contact
        # authority continues to use the unshifted signed geom distance.
        self.assertAlmostEqual(event.contact_includemargin_m, 0.025)
        self.assertAlmostEqual(
            event.contact_includemargin_m,
            event.explicit_pair_margin_m - event.explicit_pair_gap_m,
        )

    def test_settled_obstacle_translational_motion_fails_closed(self):
        monitor, ids = self._moving_obstacle_monitor(
            linear_velocity=(0.12, 0.0, 0.0),
            max_linear_speed=0.10,
            max_angular_speed=1.0,
        )
        motion = monitor.settled_state.obstacle_motion_admissibility
        self.assertFalse(motion.admissible)
        self.assertAlmostEqual(
            motion.maximum_observed_linear_speed_m_per_s, 0.12, places=12
        )
        self.assertEqual(
            [record.body_id for record in motion.bodies],
            sorted([ids["obstacle_root"], ids["obstacle_piece"]]),
        )
        self.assertTrue(
            all(record.linear_speed_exceeds_threshold for record in motion.bodies)
        )
        for record in motion.bodies:
            self.np.testing.assert_allclose(
                record.linear_velocity_world_m_per_s,
                (0.12, 0.0, 0.0),
                atol=1e-12,
            )
            self.np.testing.assert_allclose(
                record.angular_velocity_world_rad_per_s,
                (0.0, 0.0, 0.0),
                atol=1e-12,
            )
        self.assertTrue(
            all(
                not record.angular_speed_exceeds_threshold
                for record in motion.bodies
            )
        )
        reason_ids = [
            int(reason.split(" ", 1)[0].split("=", 1)[1])
            for reason in motion.reasons
        ]
        self.assertEqual(reason_ids, sorted(reason_ids))
        json.dumps(monitor.settled_state.to_dict(), allow_nan=False)
        with self.assertRaises(self.SettledObstacleMotionInadmissible):
            monitor.make_substep_callback(0, 0)

    def test_settled_obstacle_angular_motion_fails_closed(self):
        monitor, _ = self._moving_obstacle_monitor(
            angular_velocity=(0.0, 0.0, 0.25),
            max_linear_speed=1.0,
            max_angular_speed=0.20,
        )
        motion = monitor.settled_state.obstacle_motion_admissibility
        self.assertFalse(motion.admissible)
        self.assertAlmostEqual(
            motion.maximum_observed_angular_speed_rad_per_s, 0.25, places=12
        )
        self.assertTrue(
            all(record.angular_speed_exceeds_threshold for record in motion.bodies)
        )
        for record in motion.bodies:
            self.np.testing.assert_allclose(
                record.angular_velocity_world_rad_per_s,
                (0.0, 0.0, 0.25),
                atol=1e-12,
            )
            self.assertLessEqual(record.linear_speed_m_per_s, 1.0)
        self.assertTrue(
            all(
                not record.linear_speed_exceeds_threshold
                for record in motion.bodies
            )
        )
        self.assertTrue(
            all("angular_speed_rad_per_s" in reason for reason in motion.reasons)
        )
        with self.assertRaises(self.SettledObstacleMotionInadmissible):
            monitor.require_settled_obstacle_motion_admissible()

    def test_settled_obstacle_speed_thresholds_are_validated(self):
        _, _, sim, _, resolved, samples = self._build()
        invalid_values = (-1.0, float("nan"), float("inf"), True, "invalid")
        for value in invalid_values:
            with self.subTest(linear=value):
                with self.assertRaises(ValueError):
                    self.FullRobotObstacleMonitor(
                        sim,
                        resolved,
                        samples,
                        certified_coverage_radius_m=0.02,
                        max_selected_geom_surface_drift_m=1e-6,
                        max_settled_obstacle_linear_speed_m_per_s=value,
                    )
            with self.subTest(angular=value):
                with self.assertRaises(ValueError):
                    self.FullRobotObstacleMonitor(
                        sim,
                        resolved,
                        samples,
                        certified_coverage_radius_m=0.02,
                        max_selected_geom_surface_drift_m=1e-6,
                        max_settled_obstacle_angular_speed_rad_per_s=value,
                    )

    def test_full_robot_samples_and_box_obstacles_are_mandatory(self):
        _, _, sim, _, resolved, samples = self._build()
        with self.assertRaisesRegex(ValueError, "every resolved robot geom"):
            self.FullRobotObstacleMonitor(
                sim,
                resolved,
                samples[:-1],
                certified_coverage_radius_m=0.02,
                max_selected_geom_surface_drift_m=1e-6,
            )


if __name__ == "__main__":
    unittest.main()
