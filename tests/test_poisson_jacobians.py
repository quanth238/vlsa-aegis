from __future__ import annotations

import importlib.util
import copy
import json
import struct
import unittest
from unittest import mock


DEPENDENCIES_PRESENT = all(
    importlib.util.find_spec(name) is not None for name in ("numpy", "mujoco")
)


@unittest.skipUnless(DEPENDENCIES_PRESENT, "MuJoCo numerical dependencies unavailable")
class PointJacobianTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import mujoco
        import numpy as np

        from main.poisson_fullbody.robot_samples import BodySample

        xml = """
        <mujoco model="seven_hinge_chain">
          <option gravity="0 0 0"/>
          <worldbody>
            <body name="link1" pos="0 0 0.1">
              <joint name="robot0_joint1" type="hinge" axis="0 0 1"/>
              <geom name="g1" type="capsule" size="0.02 0.05" pos="0.05 0 0" quat="0.70710678 0 0.70710678 0"/>
              <body name="link2" pos="0.1 0 0">
                <joint name="robot0_joint2" type="hinge" axis="0 1 0"/>
                <geom name="g2" type="capsule" size="0.02 0.05" pos="0.05 0 0" quat="0.70710678 0 0.70710678 0"/>
                <body name="link3" pos="0.1 0 0">
                  <joint name="robot0_joint3" type="hinge" axis="1 0 0"/>
                  <geom name="g3" type="capsule" size="0.02 0.05" pos="0.05 0 0" quat="0.70710678 0 0.70710678 0"/>
                  <body name="link4" pos="0.1 0 0">
                    <joint name="robot0_joint4" type="hinge" axis="0 0 1"/>
                    <geom name="g4" type="capsule" size="0.02 0.05" pos="0.05 0 0" quat="0.70710678 0 0.70710678 0"/>
                    <body name="link5" pos="0.1 0 0">
                      <joint name="robot0_joint5" type="hinge" axis="0 1 0"/>
                      <geom name="g5" type="capsule" size="0.02 0.05" pos="0.05 0 0" quat="0.70710678 0 0.70710678 0"/>
                      <body name="link6" pos="0.1 0 0">
                        <joint name="robot0_joint6" type="hinge" axis="1 0 0"/>
                        <geom name="g6" type="capsule" size="0.02 0.05" pos="0.05 0 0" quat="0.70710678 0 0.70710678 0"/>
                        <body name="link7" pos="0.1 0 0">
                          <joint name="robot0_joint7" type="hinge" axis="0 0 1"/>
                          <geom name="g7" type="sphere" size="0.025" pos="0.08 0.02 0.01"/>
                        </body>
                      </body>
                    </body>
                  </body>
                </body>
              </body>
            </body>
          </worldbody>
        </mujoco>
        """
        cls.mujoco = mujoco
        cls.np = np
        cls.model = mujoco.MjModel.from_xml_string(xml)
        cls.data = mujoco.MjData(cls.model)
        cls.data.qpos[:] = np.array([0.1, -0.2, 0.15, -0.1, 0.05, 0.2, -0.05])
        cls.data.qvel[:] = np.linspace(-0.1, 0.1, 7)
        mujoco.mj_forward(cls.model, cls.data)
        body_id = mujoco.mj_name2id(cls.model, mujoco.mjtObj.mjOBJ_BODY, "link7")
        geom_id = mujoco.mj_name2id(cls.model, mujoco.mjtObj.mjOBJ_GEOM, "g7")
        cls.sample = BodySample(
            sample_id=0,
            body_id=int(body_id),
            body_name="link7",
            geom_id=int(geom_id),
            geom_name="g7",
            point_body_local_m=(0.08, 0.02, 0.01),
        )

    @staticmethod
    def differential_config():
        return {
            "state_source": "settled_mujoco_mjstate_integration_clone",
            "perturbation_integrator": "mujoco_mj_integratePos_full_nv_tangent",
            "point_jacobian_delta_rad": 1.0e-6,
            "point_jacobian_absolute_tolerance_m_per_rad": 2.0e-6,
            "point_jacobian_relative_tolerance": 1.0e-4,
            "point_jacobian_near_zero_frobenius_m_per_rad": 1.0e-10,
            "arm_tangent_roundtrip_criterion": (
                "scalar_hinge_two_stage_exact_fraction_binary64_roundoff"
            ),
            "binary64_unit_roundoff": 2.0 ** -53,
            "expected_mujoco_version": "3.2.3",
            "expected_arm_dof_indices": list(range(7)),
            "expected_arm_joint_ids": list(range(7)),
            "expected_arm_qpos_indices": list(range(7)),
            "expected_arm_joint_names": [
                "robot0_joint%d" % index for index in range(1, 8)
            ],
            "required_arm_joint_type": "hinge",
            "legacy_arm_tangent_reconstruction_tolerance_rad_s": 1.0e-10,
            "nonarm_tangent_leakage_tolerance_rad_s": 1.0e-12,
            "joint_velocity_directions_rad_s": [
                [0.5 if row == column else 0.0 for column in range(7)]
                for row in range(7)
            ]
            + [
                [0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5],
                [(index + 1.0) / 14.0 for index in range(7)],
            ],
            "coupled_eta_ladder_s": [
                2.0e-6,
                5.0e-7,
                1.25e-7,
                3.125e-8,
                7.8125e-9,
                1.953125e-9,
                4.8828125e-10,
            ],
            "coupled_absolute_tolerance_m2_per_s": 2.0e-7,
            "coupled_relative_tolerance": 2.0e-4,
            "coupled_near_zero_m2_per_s": 1.0e-10,
            "same_trilinear_cell_required": True,
            "required_direction_count_per_sample": 9,
            "stencil_selection_policy": (
                "largest_eta_with_valid_base_plus_minus_in_same_exact_cell"
            ),
            "finite_difference_resolution_policy": (
                "fail_on_no_certified_stencil_or_detected_cancellation"
            ),
            "failure_policy": "fail_before_active_physics_retain_artifact",
        }

    def linear_field(self, *, lower=None, upper=None, valid_cell_mask=None):
        from main.poisson_fullbody.poisson_field import TrilinearPoissonField
        from main.poisson_fullbody.voxel_grid import GridSpec

        np = self.np
        if lower is None:
            lower = np.asarray([-2.123, -2.234, -2.345])
        if upper is None:
            upper = np.asarray([2.077, 1.966, 1.855])
        grid = GridSpec.from_bounds(lower, upper, (8, 8, 8))
        axes = grid.vertex_axes()
        x, y, z = np.meshgrid(*axes, indexing="ij")
        values = 10.0 + 0.2 * x - 0.3 * y + 0.4 * z
        return TrilinearPoissonField(
            grid, values, valid_cell_mask=valid_cell_mask
        )

    def test_body_local_roundtrip(self) -> None:
        from main.poisson_fullbody.robot_samples import validate_rigid_roundtrip

        result = validate_rigid_roundtrip([self.sample], self.data)
        self.assertTrue(result["passed"], result)
        self.assertLessEqual(result["maximum_roundtrip_error_m"], 1e-10)

    def test_body_pose_accessor_supports_official_and_robosuite_spellings(self) -> None:
        from types import SimpleNamespace

        from main.poisson_fullbody.robot_samples import body_world_pose

        official_position, official_rotation = body_world_pose(self.data, 1)
        wrapper = SimpleNamespace(
            body_xpos=self.data.xpos,
            body_xmat=self.data.xmat,
        )
        wrapped_position, wrapped_rotation = body_world_pose(wrapper, 1)
        self.np.testing.assert_array_equal(wrapped_position, official_position)
        self.np.testing.assert_array_equal(wrapped_rotation, official_rotation)

    def test_arbitrary_point_jacobian_matches_central_difference(self) -> None:
        np = self.np
        from main.poisson_fullbody.jacobians import finite_difference_point_jacobian

        qpos_before = self.data.qpos.copy()
        qvel_before = self.data.qvel.copy()
        result = finite_difference_point_jacobian(
            self.model,
            self.data,
            self.sample,
            list(range(7)),
            list(range(7)),
        )
        self.assertTrue(result["passed"], result)
        np.testing.assert_array_equal(self.data.qpos, qpos_before)
        np.testing.assert_array_equal(self.data.qvel, qvel_before)

    def test_directional_derivative_uses_seven_arm_columns(self) -> None:
        from main.poisson_fullbody.jacobians import (
            directional_field_derivative,
            point_translational_jacobian,
        )

        _, jacobian = point_translational_jacobian(
            self.model, self.data, self.sample, list(range(7))
        )
        value = directional_field_derivative(
            [1.0, -2.0, 0.5], jacobian, self.data.qvel
        )
        expected = float(self.np.array([1.0, -2.0, 0.5]) @ jacobian @ self.data.qvel)
        self.assertAlmostEqual(value, expected, places=12)

    def test_batched_point_jacobians_match_official_mujoco_for_mixed_bodies_and_dofs(
        self,
    ) -> None:
        from main.poisson_fullbody.jacobians import (
            evaluate_point_jacobians,
            mujoco_integration_state_sha256,
        )
        from main.poisson_fullbody.robot_samples import BodySample

        mujoco = self.mujoco
        np = self.np
        link5_id = int(
            mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, "link5"
            )
        )
        link7_id = int(
            mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, "link7"
            )
        )
        g5_id = int(
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "g5")
        )
        g7_id = int(
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "g7")
        )
        # Interleave bodies deliberately: output must retain sample order even
        # though computation is grouped by owning rigid body.
        samples = [
            BodySample(10, link7_id, "link7", g7_id, "g7", (0.08, 0.02, 0.01)),
            BodySample(11, link5_id, "link5", g5_id, "g5", (0.03, -0.01, 0.02)),
            BodySample(12, link7_id, "link7", g7_id, "g7", (-0.02, 0.04, -0.01)),
            BodySample(13, link5_id, "link5", g5_id, "g5", (0.07, 0.015, 0.0)),
        ]
        selected_dofs = [6, 0, 4]
        self.assertFalse(
            np.allclose(self.data.xmat[link5_id].reshape(3, 3), np.eye(3))
        )
        self.assertFalse(
            np.allclose(self.data.xmat[link7_id].reshape(3, 3), np.eye(3))
        )
        expected_points = []
        expected_jacobians = []
        for sample in samples:
            point = sample.world_point(self.data)
            jacp = np.zeros((3, int(self.model.nv)), dtype=np.float64)
            jacr = np.zeros((3, int(self.model.nv)), dtype=np.float64)
            mujoco.mj_jac(
                self.model,
                self.data,
                jacp,
                jacr,
                point,
                int(sample.body_id),
            )
            expected_points.append(point)
            expected_jacobians.append(jacp[:, selected_dofs])

        state_before = mujoco_integration_state_sha256(self.model, self.data)
        xpos_before = self.data.xpos.copy()
        xmat_before = self.data.xmat.copy()
        observed_points, observed_jacobians = evaluate_point_jacobians(
            self.model, self.data, samples, selected_dofs
        )
        self.assertEqual(
            mujoco_integration_state_sha256(self.model, self.data), state_before
        )
        np.testing.assert_array_equal(self.data.xpos, xpos_before)
        np.testing.assert_array_equal(self.data.xmat, xmat_before)
        np.testing.assert_allclose(
            observed_points,
            np.stack(expected_points),
            rtol=0.0,
            atol=2.0e-16,
        )
        np.testing.assert_allclose(
            observed_jacobians,
            np.stack(expected_jacobians),
            rtol=2.0e-14,
            atol=2.0e-15,
        )

    def test_batched_point_jacobians_call_mujoco_once_per_unique_body(self) -> None:
        from main.poisson_fullbody.jacobians import evaluate_point_jacobians
        from main.poisson_fullbody.robot_samples import BodySample

        mujoco = self.mujoco
        link5_id = int(
            mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, "link5"
            )
        )
        link7_id = int(
            mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, "link7"
            )
        )
        g5_id = int(
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "g5")
        )
        g7_id = int(
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "g7")
        )
        samples = [
            BodySample(20, link7_id, "link7", g7_id, "g7", (0.08, 0.02, 0.01)),
            BodySample(21, link5_id, "link5", g5_id, "g5", (0.03, 0.0, 0.0)),
            BodySample(22, link7_id, "link7", g7_id, "g7", (0.0, -0.02, 0.01)),
            BodySample(23, link5_id, "link5", g5_id, "g5", (-0.01, 0.01, 0.0)),
            BodySample(24, link7_id, "link7", g7_id, "g7", (0.02, 0.0, -0.01)),
        ]
        body_calls = []
        original_mj_jac_body = mujoco.mj_jacBody

        def counted_mj_jac_body(*arguments):
            body_calls.append(int(arguments[-1]))
            return original_mj_jac_body(*arguments)

        with mock.patch.object(
            mujoco, "mj_jacBody", side_effect=counted_mj_jac_body
        ), mock.patch.object(
            mujoco,
            "mj_jac",
            side_effect=AssertionError("per-sample mj_jac must not be called"),
        ):
            points, jacobians = evaluate_point_jacobians(
                self.model, self.data, samples, [0, 2, 4, 6]
            )

        self.assertEqual(body_calls, [link7_id, link5_id])
        self.assertEqual(points.shape, (5, 3))
        self.assertEqual(jacobians.shape, (5, 3, 4))

    def test_batched_point_jacobians_reject_invalid_inputs(self) -> None:
        from main.poisson_fullbody.jacobians import evaluate_point_jacobians
        from main.poisson_fullbody.robot_samples import BodySample

        invalid_body = BodySample(
            30,
            int(self.model.nbody),
            "invalid",
            0,
            "invalid_geom",
            (0.0, 0.0, 0.0),
        )
        negative_body = BodySample(
            31,
            0,
            "mutated_invalid",
            0,
            "invalid_geom",
            (0.0, 0.0, 0.0),
        )
        object.__setattr__(negative_body, "body_id", -1)
        duplicate_id = BodySample(
            int(self.sample.sample_id),
            int(self.sample.body_id),
            self.sample.body_name,
            int(self.sample.geom_id),
            self.sample.geom_name,
            (0.0, 0.0, 0.0),
        )
        invalid_dofs = (
            [],
            [True],
            [0.0],
            [0, 0],
            [-1],
            [int(self.model.nv)],
            "0",
        )
        for dofs in invalid_dofs:
            with self.subTest(dofs=dofs), self.assertRaises(ValueError):
                evaluate_point_jacobians(
                    self.model, self.data, [self.sample], dofs
                )
        with self.assertRaisesRegex(ValueError, "BodySample"):
            evaluate_point_jacobians(self.model, self.data, [object()], [0])
        with self.assertRaisesRegex(ValueError, "sample IDs"):
            evaluate_point_jacobians(
                self.model, self.data, [self.sample, duplicate_id], [0]
            )
        with self.assertRaisesRegex(ValueError, "model.nbody"):
            evaluate_point_jacobians(self.model, self.data, [invalid_body], [0])
        with self.assertRaisesRegex(ValueError, "model.nbody"):
            evaluate_point_jacobians(self.model, self.data, [negative_body], [0])

        empty_points, empty_jacobians = evaluate_point_jacobians(
            self.model, self.data, [], [6, 1]
        )
        self.assertEqual(empty_points.shape, (0, 3))
        self.assertEqual(empty_jacobians.shape, (0, 3, 2))

    def test_exhaustive_differential_audit_is_json_native_and_state_preserving(self):
        from main.poisson_fullbody.jacobians import (
            audit_protected_sample_differentials,
            mujoco_integration_state_sha256,
            validate_protected_sample_differential_audit,
        )

        before = mujoco_integration_state_sha256(self.model, self.data)
        result = audit_protected_sample_differentials(
            self.model,
            self.data,
            [self.sample],
            list(range(7)),
            self.linear_field(),
            self.differential_config(),
        )
        self.assertEqual(mujoco_integration_state_sha256(self.model, self.data), before)
        self.assertTrue(result["passed"], result)
        self.assertEqual(result["counts"]["selected_coupled_direction_count"], 9)
        json.dumps(result, allow_nan=False)
        receipt = validate_protected_sample_differential_audit(
            result,
            expected_samples=[self.sample.to_dict()],
            expected_arm_dof_indices=list(range(7)),
            expected_integration_state_sha256=before,
            expected_differential_audit_config=self.differential_config(),
        )
        self.assertTrue(receipt["passed"])

    def test_pure_validator_rejects_rehashed_matrix_and_identity_tampering(self):
        from main.poisson_fullbody.contracts import canonical_json_bytes, sha256_bytes
        from main.poisson_fullbody.jacobians import (
            DIFFERENTIAL_AUDIT_HASH_FIELD,
            DifferentialAuditError,
            audit_protected_sample_differentials,
            mujoco_integration_state_sha256,
            validate_protected_sample_differential_audit,
        )

        state_hash = mujoco_integration_state_sha256(self.model, self.data)
        result = audit_protected_sample_differentials(
            self.model, self.data, [self.sample], list(range(7)),
            self.linear_field(), self.differential_config()
        )

        def rehash(value):
            value.pop(DIFFERENTIAL_AUDIT_HASH_FIELD, None)
            value[DIFFERENTIAL_AUDIT_HASH_FIELD] = sha256_bytes(
                canonical_json_bytes(value)
            )

        forged = copy.deepcopy(result)
        forged["sample_records"][0][
            "numerical_point_jacobian_m_per_rad_3x7"
        ][0][0] += 1.0e-3
        rehash(forged)
        with self.assertRaisesRegex(DifferentialAuditError, "does not reconstruct"):
            validate_protected_sample_differential_audit(
                forged,
                expected_samples=[self.sample],
                expected_arm_dof_indices=list(range(7)),
                expected_integration_state_sha256=state_hash,
                expected_differential_audit_config=self.differential_config(),
            )

        forged = copy.deepcopy(result)
        forged["ordered_samples"][0]["geom_name"] = "wrong_geom"
        forged["sample_records"][0]["identity"]["geom_name"] = "wrong_geom"
        rehash(forged)
        with self.assertRaisesRegex(DifferentialAuditError, "differ from authority"):
            validate_protected_sample_differential_audit(
                forged,
                expected_samples=[self.sample],
                expected_arm_dof_indices=list(range(7)),
                expected_integration_state_sha256=state_hash,
                expected_differential_audit_config=self.differential_config(),
            )

    def test_no_certified_stencil_retains_every_invalid_attempt(self):
        from main.poisson_fullbody.jacobians import (
            DifferentialAuditError,
            audit_protected_sample_differentials,
            mujoco_integration_state_sha256,
            validate_protected_sample_differential_audit,
        )

        point = self.sample.world_point(self.data)
        lower = point - self.np.asarray([0.7, 0.7, 0.7])
        upper = point.copy()  # base is valid on the closed outer boundary
        result = audit_protected_sample_differentials(
            self.model,
            self.data,
            [self.sample],
            list(range(7)),
            self.linear_field(lower=lower, upper=upper),
            self.differential_config(),
        )
        self.assertFalse(result["passed"])
        failed = [
            direction
            for direction in result["sample_records"][0]["coupled_directions"]
            if direction["selected_attempt_index"] is None
        ]
        self.assertTrue(failed)
        for direction in failed:
            self.assertEqual(len(direction["attempts"]), 7)
            self.assertTrue(
                any(
                    "outside_grid" in reason
                    for attempt in direction["attempts"]
                    for reason in attempt["noneligible_reasons"]
                )
            )
        with self.assertRaisesRegex(DifferentialAuditError, "did not pass"):
            validate_protected_sample_differential_audit(
                result,
                expected_samples=[self.sample],
                expected_arm_dof_indices=list(range(7)),
                expected_integration_state_sha256=mujoco_integration_state_sha256(
                    self.model, self.data
                ),
                expected_differential_audit_config=self.differential_config(),
            )

    def test_adaptive_eta_retains_rejected_larger_same_cell_attempt(self):
        from main.poisson_fullbody.jacobians import (
            audit_protected_sample_differentials,
            point_translational_jacobian,
        )

        point, jacobian = point_translational_jacobian(
            self.model, self.data, self.sample, list(range(7))
        )
        direction = self.np.asarray(self.differential_config()[
            "joint_velocity_directions_rad_s"
        ][0])
        velocity = jacobian @ direction
        axis = int(self.np.argmax(self.np.abs(velocity)))
        self.assertGreater(abs(float(velocity[axis])), 1.0e-6)
        largest_eta = self.differential_config()["coupled_eta_ladder_s"][0]
        boundary = float(point[axis] + 0.5 * velocity[axis] * largest_eta)
        spacing = 0.1
        lower = self.np.asarray(point, dtype=float) - 0.337
        lower[axis] = boundary - 3.0 * spacing
        upper = lower + 7.0 * spacing
        result = audit_protected_sample_differentials(
            self.model,
            self.data,
            [self.sample],
            list(range(7)),
            self.linear_field(lower=lower, upper=upper),
            self.differential_config(),
        )
        self.assertTrue(result["passed"], result)
        adaptive = result["sample_records"][0]["coupled_directions"][0]
        self.assertGreater(len(adaptive["attempts"]), 1)
        self.assertFalse(adaptive["attempts"][0]["eligible_same_cell_stencil"])
        self.assertTrue(
            any(
                "cell_differs_from_base" in reason
                for reason in adaptive["attempts"][0]["noneligible_reasons"]
            )
        )
        self.assertTrue(adaptive["attempts"][-1]["selected"])


class ScalarHingeRoundoffCriterionTest(unittest.TestCase):
    @staticmethod
    def _record(*, step_s, velocity_rad_s, dof=3, qpos_rad=-2.35619449):
        from main.poisson_fullbody.jacobians import _tangent_reconstruction
        from tests.test_poisson_shadow_identification import (
            differential_audit_config,
        )

        config = differential_audit_config()
        base_qpos = [0.0] * 7
        base_qpos[dof] = qpos_rad
        requested = [0.0] * 7
        requested[dof] = velocity_rad_s
        plus_qpos = list(base_qpos)
        minus_qpos = list(base_qpos)
        plus_qpos[dof] = float(qpos_rad + step_s * velocity_rad_s)
        minus_qpos[dof] = float(qpos_rad - step_s * velocity_rad_s)
        plus_reconstructed = [0.0] * 7
        minus_reconstructed = [0.0] * 7
        plus_reconstructed[dof] = (
            plus_qpos[dof] - base_qpos[dof]
        ) / step_s
        minus_reconstructed[dof] = (
            minus_qpos[dof] - base_qpos[dof]
        ) / step_s
        arguments = {
            "requested": requested,
            "plus_reconstructed": plus_reconstructed,
            "minus_reconstructed": minus_reconstructed,
            "base_arm_qpos": base_qpos,
            "plus_perturbed_arm_qpos": plus_qpos,
            "minus_perturbed_arm_qpos": minus_qpos,
            "arm_dofs": list(range(7)),
            "perturbation_step_s": step_s,
            "config": config,
        }
        return _tangent_reconstruction(**arguments), arguments

    def test_root_g_signature_and_scale_equivalent_displacement_both_pass(self):
        root_g, _ = self._record(step_s=1.0e-6, velocity_rad_s=1.0)
        scaled, _ = self._record(step_s=2.0e-6, velocity_rad_s=0.5)

        self.assertEqual(
            root_g["maximum_arm_reconstruction_error_rad_s"],
            1.397779669787269e-10,
        )
        self.assertEqual(
            root_g["maximum_arm_reconstruction_displacement_error_rad"],
            1.397779669787269e-16,
        )
        self.assertIs(root_g["legacy_arm_passed"], False)
        self.assertIs(root_g["arm_passed"], True)
        self.assertIs(root_g["passed"], True)

        self.assertEqual(
            scaled["maximum_arm_reconstruction_displacement_error_rad"],
            root_g["maximum_arm_reconstruction_displacement_error_rad"],
        )
        self.assertEqual(
            scaled["plus_perturbed_arm_qpos_rad"],
            root_g["plus_perturbed_arm_qpos_rad"],
        )
        self.assertEqual(
            scaled["minus_perturbed_arm_qpos_rad"],
            root_g["minus_perturbed_arm_qpos_rad"],
        )
        self.assertIs(scaled["arm_passed"], True)
        self.assertIs(scaled["passed"], True)

    def test_velocity_and_perturbed_qpos_corruption_fail(self):
        from main.poisson_fullbody.jacobians import _tangent_reconstruction

        _, arguments = self._record(step_s=1.0e-6, velocity_rad_s=1.0)

        velocity_corruption = copy.deepcopy(arguments)
        velocity_corruption["plus_reconstructed"][3] += 1.0e-8
        result = _tangent_reconstruction(**velocity_corruption)
        self.assertIs(result["plus_differentiation_passed_by_arm_dof"][3], False)
        self.assertIs(result["arm_passed"], False)
        self.assertIs(result["passed"], False)

    def test_zero_tangent_accepts_only_signed_zero_representation_change(self):
        from main.poisson_fullbody.jacobians import _tangent_reconstruction

        result, arguments = self._record(
            step_s=1.0e-6,
            velocity_rad_s=0.0,
            qpos_rad=-0.0,
        )
        self.assertNotEqual(
            struct.pack("<d", arguments["base_arm_qpos"][3]),
            struct.pack("<d", arguments["plus_perturbed_arm_qpos"][3]),
        )
        self.assertEqual(arguments["base_arm_qpos"][3], -0.0)
        self.assertEqual(arguments["plus_perturbed_arm_qpos"][3], 0.0)
        self.assertIs(result["plus_observable_by_arm_dof"][3], True)
        self.assertIs(result["minus_observable_by_arm_dof"][3], True)
        self.assertIs(result["passed"], True)

        corrupted = copy.deepcopy(arguments)
        corrupted["plus_reconstructed"][3] = 1.0e-12
        result = _tangent_reconstruction(**corrupted)
        self.assertIs(result["plus_observable_by_arm_dof"][3], False)
        self.assertIs(result["passed"], False)

        qpos_corruption = copy.deepcopy(arguments)
        qpos_corruption["plus_perturbed_arm_qpos"][3] += 1.0e-12
        result = _tangent_reconstruction(**qpos_corruption)
        self.assertIs(result["plus_integration_passed_by_arm_dof"][3], False)
        self.assertIs(result["arm_passed"], False)
        self.assertIs(result["passed"], False)

    def test_wrong_sign_and_dof_swap_fail(self):
        from main.poisson_fullbody.jacobians import _tangent_reconstruction

        _, arguments = self._record(step_s=1.0e-6, velocity_rad_s=1.0)

        wrong_sign = copy.deepcopy(arguments)
        wrong_sign["minus_reconstructed"][3] = abs(
            wrong_sign["minus_reconstructed"][3]
        )
        result = _tangent_reconstruction(**wrong_sign)
        self.assertIs(result["minus_differentiation_passed_by_arm_dof"][3], False)
        self.assertIs(result["passed"], False)

        dof_swap = copy.deepcopy(arguments)
        dof_swap["plus_reconstructed"] = [0.0] * 7
        dof_swap["minus_reconstructed"] = [0.0] * 7
        dof_swap["plus_perturbed_arm_qpos"] = list(dof_swap["base_arm_qpos"])
        dof_swap["minus_perturbed_arm_qpos"] = list(dof_swap["base_arm_qpos"])
        dof_swap["plus_perturbed_arm_qpos"][5] += 1.0e-6
        dof_swap["minus_perturbed_arm_qpos"][5] -= 1.0e-6
        dof_swap["plus_reconstructed"][5] = 1.0
        dof_swap["minus_reconstructed"][5] = -1.0
        result = _tangent_reconstruction(**dof_swap)
        self.assertIs(result["plus_integration_passed_by_arm_dof"][3], False)
        self.assertIs(result["plus_observable_by_arm_dof"][3], False)
        self.assertIs(result["passed"], False)

    def test_rehashed_bool_topology_and_pass_field_substitutions_fail(self):
        from main.poisson_fullbody.contracts import canonical_json_bytes, sha256_bytes
        from main.poisson_fullbody.jacobians import (
            DIFFERENTIAL_AUDIT_HASH_FIELD,
            DifferentialAuditError,
            validate_protected_sample_differential_audit,
        )
        from tests.test_poisson_shadow_identification import (
            differential_audit_config,
            protected_sample_identities,
            synthetic_integration_state_sha256,
            valid_differential_audit,
        )

        samples = protected_sample_identities()
        config = differential_audit_config()
        state_sha256 = synthetic_integration_state_sha256()
        arguments = {
            "expected_samples": samples,
            "expected_arm_dof_indices": list(range(7)),
            "expected_integration_state_sha256": state_sha256,
            "expected_differential_audit_config": config,
        }

        def rehash(audit):
            audit.pop(DIFFERENTIAL_AUDIT_HASH_FIELD, None)
            audit[DIFFERENTIAL_AUDIT_HASH_FIELD] = sha256_bytes(
                canonical_json_bytes(audit)
            )

        bool_topology, _ = valid_differential_audit(
            samples, state_sha256, config
        )
        bool_topology["arm_dof_indices"][0:2] = [False, True]
        rehash(bool_topology)
        with self.assertRaisesRegex(
            DifferentialAuditError, "arm_dof_indices must contain seven integers"
        ):
            validate_protected_sample_differential_audit(
                bool_topology, **arguments
            )

        pass_substitution, _ = valid_differential_audit(
            samples, state_sha256, config
        )
        pass_substitution["sample_records"][0][
            "point_perturbation_tangent_reconstruction_by_arm_dof"
        ][0]["integration_roundoff_passed"] = 1
        rehash(pass_substitution)
        with self.assertRaisesRegex(
            DifferentialAuditError, "tangent diagnostics do not reconstruct"
        ):
            validate_protected_sample_differential_audit(
                pass_substitution, **arguments
            )

    def test_rehashed_source_bits_and_qpos_mapping_cannot_replace_authority(self):
        from main.poisson_fullbody.contracts import canonical_json_bytes, sha256_bytes
        from main.poisson_fullbody.jacobians import (
            DIFFERENTIAL_AUDIT_HASH_FIELD,
            DifferentialAuditError,
            _stable_audit_hashes,
            validate_protected_sample_differential_audit,
        )
        from tests.test_poisson_shadow_identification import (
            differential_audit_config,
            protected_sample_identities,
            synthetic_integration_state_sha256,
            valid_differential_audit,
        )

        samples = protected_sample_identities()
        config = differential_audit_config()
        state_sha256 = synthetic_integration_state_sha256()
        arguments = {
            "expected_samples": samples,
            "expected_arm_dof_indices": list(range(7)),
            "expected_integration_state_sha256": state_sha256,
            "expected_differential_audit_config": config,
        }

        state_bits, _ = valid_differential_audit(samples, state_sha256, config)
        state_bits["integration_state"]["source_initial_state_f64_le_hex"][1] = (
            "000000000000c03f"
        )
        state_bits.pop(DIFFERENTIAL_AUDIT_HASH_FIELD)
        state_bits[DIFFERENTIAL_AUDIT_HASH_FIELD] = sha256_bytes(
            canonical_json_bytes(state_bits)
        )
        with self.assertRaisesRegex(
            DifferentialAuditError, "serialized integration-state bits"
        ):
            validate_protected_sample_differential_audit(state_bits, **arguments)

        topology, _ = valid_differential_audit(samples, state_sha256, config)
        authority = topology["arm_scalar_hinge_roundoff_authority"]
        authority["arm_qpos_indices"][0:2] = [1, 0]
        authority["arm_jnt_qposadr"][0:2] = [1, 0]
        topology["arm_scalar_hinge_roundoff_authority_sha256"] = sha256_bytes(
            canonical_json_bytes(authority)
        )
        identities = [sample.to_dict() for sample in samples]
        for key, value in _stable_audit_hashes(
            state_sha256,
            list(range(7)),
            identities,
            config,
            authority,
            topology["sample_records"],
        ).items():
            topology[key] = value
        topology.pop(DIFFERENTIAL_AUDIT_HASH_FIELD)
        topology[DIFFERENTIAL_AUDIT_HASH_FIELD] = sha256_bytes(
            canonical_json_bytes(topology)
        )
        with self.assertRaisesRegex(DifferentialAuditError, "qpos topology is wrong"):
            validate_protected_sample_differential_audit(topology, **arguments)

    def test_nonfinite_and_subnormal_roundtrip_inputs_fail_closed(self):
        from main.poisson_fullbody.jacobians import (
            DifferentialAuditError,
            _tangent_reconstruction,
        )

        _, arguments = self._record(step_s=1.0e-6, velocity_rad_s=1.0)
        nonfinite = copy.deepcopy(arguments)
        nonfinite["plus_perturbed_arm_qpos"][3] = float("nan")
        with self.assertRaisesRegex(DifferentialAuditError, "finite number"):
            _tangent_reconstruction(**nonfinite)

        subnormal = copy.deepcopy(arguments)
        subnormal["perturbation_step_s"] = float.fromhex("0x0.0000000000001p-1022")
        with self.assertRaisesRegex(DifferentialAuditError, "subnormal"):
            _tangent_reconstruction(**subnormal)


def _strict_failed_differential_audit_fixture():
    """Construct a self-consistent failed audit without MuJoCo dependencies."""

    from main.poisson_fullbody.contracts import canonical_json_bytes, sha256_bytes
    from main.poisson_fullbody.jacobians import (
        DIFFERENTIAL_AUDIT_HASH_FIELD,
        _stable_audit_hashes,
        _tangent_reconstruction,
    )
    from tests.test_poisson_shadow_identification import (
        differential_audit_config,
        protected_sample_identities,
        synthetic_integration_state_sha256,
        valid_differential_audit,
    )

    samples = protected_sample_identities()
    config = differential_audit_config()
    state_sha256 = synthetic_integration_state_sha256()
    audit, _ = valid_differential_audit(samples, state_sha256, config)
    row = audit["sample_records"][0]
    tangent = row[
        "point_perturbation_tangent_reconstruction_by_arm_dof"
    ][0]
    plus = list(tangent["plus_reconstructed_tangent_nv_rad_s"])
    plus[0] = 2.0
    row["point_perturbation_tangent_reconstruction_by_arm_dof"][0] = (
        _tangent_reconstruction(
            tangent["requested_tangent_nv_rad_s"],
            plus,
            tangent["minus_reconstructed_tangent_nv_rad_s"],
            audit["arm_scalar_hinge_roundoff_authority"]["base_arm_qpos_rad"],
            tangent["plus_perturbed_arm_qpos_rad"],
            tangent["minus_perturbed_arm_qpos_rad"],
            list(range(7)),
            tangent["perturbation_step_s"],
            config,
        )
    )
    row["point_jacobian_passed"] = False
    row["passed"] = False
    audit["counts"]["point_jacobian_passed_sample_count"] -= 1
    audit["counts"]["passed_sample_count"] -= 1
    audit["passed"] = False
    identities = [sample.to_dict() for sample in samples]
    for key, value in _stable_audit_hashes(
        state_sha256,
        list(range(7)),
        identities,
        config,
        audit["arm_scalar_hinge_roundoff_authority"],
        audit["sample_records"],
    ).items():
        audit[key] = value
    audit.pop(DIFFERENTIAL_AUDIT_HASH_FIELD, None)
    audit[DIFFERENTIAL_AUDIT_HASH_FIELD] = sha256_bytes(
        canonical_json_bytes(audit)
    )
    return audit, samples, state_sha256, config


class DifferentialAuditFailureInspectionTest(unittest.TestCase):
    def test_inspection_retains_failure_but_default_validation_still_rejects(self):
        from main.poisson_fullbody.jacobians import (
            DifferentialAuditError,
            inspect_protected_sample_differential_audit,
            validate_protected_sample_differential_audit,
        )

        audit, samples, state_sha256, config = (
            _strict_failed_differential_audit_fixture()
        )
        arguments = {
            "expected_samples": samples,
            "expected_arm_dof_indices": list(range(7)),
            "expected_integration_state_sha256": state_sha256,
            "expected_differential_audit_config": config,
        }
        receipt = inspect_protected_sample_differential_audit(
            audit, **arguments
        )
        self.assertIs(receipt["passed"], False)
        self.assertEqual(receipt["audit_payload_sha256"], audit["audit_payload_sha256"])
        self.assertEqual(receipt["counts"], audit["counts"])
        self.assertEqual(receipt["counts"]["passed_sample_count"], 1)
        with self.assertRaisesRegex(DifferentialAuditError, "did not pass"):
            validate_protected_sample_differential_audit(audit, **arguments)

    def test_inspection_rejects_malformed_and_rehashed_failed_audit(self):
        from main.poisson_fullbody.contracts import canonical_json_bytes, sha256_bytes
        from main.poisson_fullbody.jacobians import (
            DIFFERENTIAL_AUDIT_HASH_FIELD,
            DifferentialAuditError,
            inspect_protected_sample_differential_audit,
        )

        audit, samples, state_sha256, config = (
            _strict_failed_differential_audit_fixture()
        )
        arguments = {
            "expected_samples": samples,
            "expected_arm_dof_indices": list(range(7)),
            "expected_integration_state_sha256": state_sha256,
            "expected_differential_audit_config": config,
        }

        malformed = copy.deepcopy(audit)
        malformed["unexpected"] = "not registered"
        with self.assertRaisesRegex(DifferentialAuditError, "invalid keys"):
            inspect_protected_sample_differential_audit(malformed, **arguments)

        rehashed = copy.deepcopy(audit)
        rehashed["sample_records"][0][
            "numerical_point_jacobian_m_per_rad_3x7"
        ][0][0] += 1.0e-3
        rehashed.pop(DIFFERENTIAL_AUDIT_HASH_FIELD)
        rehashed[DIFFERENTIAL_AUDIT_HASH_FIELD] = sha256_bytes(
            canonical_json_bytes(rehashed)
        )
        with self.assertRaisesRegex(DifferentialAuditError, "does not reconstruct"):
            inspect_protected_sample_differential_audit(rehashed, **arguments)

    def test_passing_inspection_and_default_validation_are_identical_and_strict(self):
        from main.poisson_fullbody.contracts import canonical_json_bytes, sha256_bytes
        from main.poisson_fullbody.jacobians import (
            DIFFERENTIAL_AUDIT_HASH_FIELD,
            DifferentialAuditError,
            inspect_protected_sample_differential_audit,
            validate_protected_sample_differential_audit,
        )
        from tests.test_poisson_shadow_identification import (
            differential_audit_config,
            protected_sample_identities,
            synthetic_integration_state_sha256,
            valid_differential_audit,
        )

        samples = protected_sample_identities()
        config = differential_audit_config()
        state_sha256 = synthetic_integration_state_sha256()
        audit, expected_receipt = valid_differential_audit(
            samples, state_sha256, config
        )
        arguments = {
            "expected_samples": samples,
            "expected_arm_dof_indices": list(range(7)),
            "expected_integration_state_sha256": state_sha256,
            "expected_differential_audit_config": config,
        }
        self.assertEqual(
            inspect_protected_sample_differential_audit(audit, **arguments),
            expected_receipt,
        )
        self.assertEqual(
            validate_protected_sample_differential_audit(audit, **arguments),
            expected_receipt,
        )

        forged_failure = copy.deepcopy(audit)
        forged_failure["passed"] = False
        forged_failure.pop(DIFFERENTIAL_AUDIT_HASH_FIELD)
        forged_failure[DIFFERENTIAL_AUDIT_HASH_FIELD] = sha256_bytes(
            canonical_json_bytes(forged_failure)
        )
        with self.assertRaisesRegex(DifferentialAuditError, "does not reconstruct"):
            inspect_protected_sample_differential_audit(
                forged_failure, **arguments
            )


if __name__ == "__main__":
    unittest.main()
