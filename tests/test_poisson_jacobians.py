from __future__ import annotations

import importlib.util
import copy
import json
import unittest


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
              <joint name="j1" type="hinge" axis="0 0 1"/>
              <geom name="g1" type="capsule" size="0.02 0.05" pos="0.05 0 0" quat="0.70710678 0 0.70710678 0"/>
              <body name="link2" pos="0.1 0 0">
                <joint name="j2" type="hinge" axis="0 1 0"/>
                <geom name="g2" type="capsule" size="0.02 0.05" pos="0.05 0 0" quat="0.70710678 0 0.70710678 0"/>
                <body name="link3" pos="0.1 0 0">
                  <joint name="j3" type="hinge" axis="1 0 0"/>
                  <geom name="g3" type="capsule" size="0.02 0.05" pos="0.05 0 0" quat="0.70710678 0 0.70710678 0"/>
                  <body name="link4" pos="0.1 0 0">
                    <joint name="j4" type="hinge" axis="0 0 1"/>
                    <geom name="g4" type="capsule" size="0.02 0.05" pos="0.05 0 0" quat="0.70710678 0 0.70710678 0"/>
                    <body name="link5" pos="0.1 0 0">
                      <joint name="j5" type="hinge" axis="0 1 0"/>
                      <geom name="g5" type="capsule" size="0.02 0.05" pos="0.05 0 0" quat="0.70710678 0 0.70710678 0"/>
                      <body name="link6" pos="0.1 0 0">
                        <joint name="j6" type="hinge" axis="1 0 0"/>
                        <geom name="g6" type="capsule" size="0.02 0.05" pos="0.05 0 0" quat="0.70710678 0 0.70710678 0"/>
                        <body name="link7" pos="0.1 0 0">
                          <joint name="j7" type="hinge" axis="0 0 1"/>
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
            "arm_tangent_reconstruction_tolerance_rad_s": 1.0e-10,
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
        valid_differential_audit,
    )

    samples = protected_sample_identities()
    config = differential_audit_config()
    state_sha256 = "c" * 64
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
            list(range(7)),
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
            valid_differential_audit,
        )

        samples = protected_sample_identities()
        config = differential_audit_config()
        state_sha256 = "d" * 64
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
