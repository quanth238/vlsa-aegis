from __future__ import annotations

import importlib.util
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


if __name__ == "__main__":
    unittest.main()
