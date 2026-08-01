from __future__ import annotations

import importlib.util
from types import SimpleNamespace
import unittest


NUMPY_PRESENT = importlib.util.find_spec("numpy") is not None
MUJOCO_PRESENT = importlib.util.find_spec("mujoco") is not None


@unittest.skipUnless(NUMPY_PRESENT, "NumPy unavailable")
class ControllerBridgeTest(unittest.TestCase):
    def setUp(self):
        import numpy as np

        from main.poisson_fullbody.controller_bridge import (
            joint_velocity_controller_contract,
            model_physics_contract,
            restore_osc_settled_state_into_joint_velocity_env,
        )

        self.np = np
        self.controller_contract = joint_velocity_controller_contract
        self.model_physics_contract = model_physics_contract
        self.restore = restore_osc_settled_state_into_joint_velocity_env

    @unittest.skipUnless(MUJOCO_PRESENT, "MuJoCo unavailable")
    def test_compiled_model_hash_binds_nested_physics_options(self):
        import mujoco

        xml = (
            '<mujoco><option timestep="0.002" gravity="0 0 -9.81"/>'
            '<worldbody><body><joint name="j" type="hinge"/>'
            '<geom type="sphere" size="0.01" mass="1"/></body></worldbody>'
            '</mujoco>'
        )

        def model():
            return mujoco.MjModel.from_xml_string(xml)

        baseline = model()
        baseline_contract = self.model_physics_contract(baseline)
        self.assertIsNotNone(baseline_contract["compiled_mjb_sha256"])
        mutations = (
            ("timestep", 0.003),
            ("gravity", self.np.array([0.0, 0.0, -1.0])),
            ("integrator", mujoco.mjtIntegrator.mjINT_RK4),
            ("solver", mujoco.mjtSolver.mjSOL_CG),
        )
        for field, value in mutations:
            changed = model()
            setattr(changed.opt, field, value)
            with self.subTest(field=field):
                self.assertNotEqual(
                    baseline_contract["sha256"],
                    self.model_physics_contract(changed)["sha256"],
                )

    def _env(self, controller_name):
        np = self.np

        class Model:
            nq = 9
            nv = 9
            na = 0
            nbody = 2
            ngeom = 1
            njnt = 2
            nu = 7
            body_parentid = np.asarray([0, 0])
            geom_bodyid = np.asarray([1])
            geom_type = np.asarray([6])
            geom_size = np.asarray([[0.1, 0.1, 0.1]])
            actuator_ctrlrange = np.column_stack(
                (np.full(7, -2.0), np.full(7, 2.0))
            )

            def body_id2name(self, index):
                return ("world", "robot")[index]

            def geom_id2name(self, index):
                return "robot_geom"

            def joint_id2name(self, index):
                return ("joint0", "joint1")[index]

            def actuator_id2name(self, index):
                return "actuator%d" % index

        class State:
            def __init__(self, data):
                self.data = data

            def flatten(self):
                return np.concatenate(([self.data.time], self.data.qpos, self.data.qvel))

        class Sim:
            def __init__(self):
                self.model = Model()
                self.data = SimpleNamespace(
                    time=0.0,
                    qpos=np.arange(9, dtype=float) * 0.01,
                    qvel=np.arange(9, dtype=float) * 0.001,
                )

            def get_state(self):
                return State(self.data)

            def set_state_from_flattened(self, state):
                self.data.time = float(state[0])
                self.data.qpos[:] = state[1:10]
                self.data.qvel[:] = state[10:19]

            def forward(self):
                return None

        class Buffer:
            def __init__(self):
                self._size = 3

            def clear(self):
                self._size = 0

        class Controller:
            def __init__(self):
                self.name = controller_name
                self.control_dim = 7
                self.output_min = np.full(7, -0.5)
                self.output_max = np.full(7, 0.5)
                self.input_min = np.full(7, -1.0)
                self.input_max = np.full(7, 1.0)
                self.kp = np.full(7, 3.0)
                self.ki = self.kp * 0.005
                self.kd = self.kp * 0.001
                self.control_freq = 100
                self.velocity_limits = None
                self.joint_index = {
                    "joints": np.arange(7),
                    "qpos": np.arange(1, 8),
                    "qvel": np.arange(1, 8),
                }
                self.actuator_limits = (
                    np.full(7, -2.0),
                    np.full(7, 2.0),
                )
                self.interpolator = None
                self.current_vel = np.ones(7)
                self.last_err = np.ones(7)
                self.summed_err = np.ones(7)
                self.derr_buf = Buffer()
                self.saturated = True
                self.last_joint_vel = np.ones(7)
                self.torques = np.ones(7)
                self.new_update = False
                self.goal_vel = np.ones(7)

            def update_initial_joints(self, joints):
                self.initial_joint = np.asarray(joints).copy()

            def reset_goal(self):
                self.goal_vel = np.zeros(7)

            def update(self, force=False):
                self.updated_force = force

        simulation = Sim()
        robot = SimpleNamespace(
            controller=Controller(),
            _ref_joint_pos_indexes=np.arange(1, 8),
            _ref_joint_vel_indexes=np.arange(1, 8),
            _ref_joint_indexes=np.asarray([0, 1, 0, 1, 0, 1, 0]),
            _ref_joint_actuator_indexes=np.arange(7),
        )
        inner = SimpleNamespace(
            action_dim=8,
            control_freq=100,
            control_timestep=0.01,
            model_timestep=0.002,
            timestep=20,
            cur_time=1.0,
            done=False,
        )
        wrapper = SimpleNamespace(sim=simulation, robots=[robot], env=inner)
        wrapper._post_process = lambda: None
        wrapper._update_observables = lambda force=False: None
        wrapper.check_success = lambda: False
        return wrapper

    def test_controller_contract_and_exact_restore(self):
        source = self._env("OSC_POSE")
        target = self._env("JOINT_VELOCITY")
        target.sim.data.qpos[:] = -3.0
        target.sim.data.qvel[:] = 4.0
        state = source.sim.get_state().flatten().copy()
        contract = self.controller_contract(target)
        self.assertEqual(contract["physics_substeps_per_control"], 5)
        result = self.restore(
            source,
            target,
            state,
            max_arm_qpos_error_rad=1e-10,
            max_arm_qvel_error_rad_s=1e-10,
            require_official_integration_state=False,
        )
        self.assertTrue(result["exact_flattened_state"])
        self.assertEqual(result["maximum_arm_qpos_error_rad"], 0.0)
        self.assertEqual(result["maximum_arm_qvel_error_rad_s"], 0.0)
        self.assertTrue(result["pid_memory_reset"]["goal_velocity_zero"])
        self.assertEqual(result["pid_memory_reset"]["derivative_buffer_size"], 0)
        self.assertEqual(
            len(result["controller_software_state"]["sha256"]), 64
        )
        self.assertIn(
            "derivative_ring_buffer",
            result["controller_software_state"]["fields"],
        )

    def test_wrong_controller_frequency_scaling_or_state_fails_closed(self):
        target = self._env("OSC_POSE")
        with self.assertRaisesRegex(ValueError, "JOINT_VELOCITY"):
            self.controller_contract(target)
        target = self._env("JOINT_VELOCITY")
        target.env.control_freq = 20
        with self.assertRaisesRegex(ValueError, "100 Hz"):
            self.controller_contract(target)
        target = self._env("JOINT_VELOCITY")
        target.robots[0].controller.output_max[0] = 0.4
        with self.assertRaisesRegex(ValueError, "0.5"):
            self.controller_contract(target)
        target = self._env("JOINT_VELOCITY")
        del target.robots[0].controller.kp
        target.robots[0].controller.kv = 0.25
        with self.assertRaisesRegex(ValueError, "kp"):
            self.controller_contract(target)

        source = self._env("OSC_POSE")
        target = self._env("JOINT_VELOCITY")
        wrong = source.sim.get_state().flatten().copy()
        wrong[1] += 1e-4
        with self.assertRaisesRegex(ValueError, "exact state"):
            self.restore(
                source,
                target,
                wrong,
                max_arm_qpos_error_rad=1e-10,
                max_arm_qvel_error_rad_s=1e-10,
                require_official_integration_state=False,
            )

    def test_tolerances_reject_boolean_zero_and_nonfinite(self):
        source = self._env("OSC_POSE")
        target = self._env("JOINT_VELOCITY")
        state = source.sim.get_state().flatten()
        for invalid in (True, 0.0, -1.0, float("inf"), float("nan")):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "finite and positive"):
                    self.restore(
                        source,
                        target,
                        state,
                        max_arm_qpos_error_rad=invalid,
                        max_arm_qvel_error_rad_s=1e-10,
                        require_official_integration_state=False,
                    )

    @unittest.skipUnless(MUJOCO_PRESENT, "MuJoCo unavailable")
    def test_official_integration_state_restores_ctrl_and_warmstart(self):
        import mujoco

        np = self.np
        bodies = []
        close = []
        actuators = []
        for index in range(7):
            bodies.append(
                '<body name="link{0}" pos="0 0 0.1"><joint name="j{0}" '
                'type="hinge" axis="0 0 1" range="-2 2"/>'
                '<geom name="g{0}" type="sphere" size="0.01" mass="0.1"/>'.format(
                    index
                )
            )
            close.append("</body>")
            actuators.append(
                '<motor name="a{0}" joint="j{0}" ctrllimited="true" '
                'ctrlrange="-2 2"/>'.format(index)
            )
        xml = (
            '<mujoco><option timestep="0.002"/><worldbody>'
            + "".join(bodies)
            + "".join(reversed(close))
            + "</worldbody><actuator>"
            + "".join(actuators)
            + "</actuator></mujoco>"
        )

        class State:
            def __init__(self, data):
                self.data = data

            def flatten(self):
                return np.concatenate(
                    ([self.data.time], self.data.qpos, self.data.qvel)
                )

        class Sim:
            def __init__(self):
                self.model = mujoco.MjModel.from_xml_string(xml)
                self.data = mujoco.MjData(self.model)

            def get_state(self):
                return State(self.data)

            def set_state_from_flattened(self, state):
                self.data.time = state[0]
                self.data.qpos[:] = state[1:8]
                self.data.qvel[:] = state[8:15]

            def forward(self):
                mujoco.mj_forward(self.model, self.data)

        class Buffer:
            def __init__(self):
                self._size = 1

            def clear(self):
                self._size = 0

        class Controller:
            name = "JOINT_VELOCITY"
            control_dim = 7
            output_min = np.full(7, -0.5)
            output_max = np.full(7, 0.5)
            input_min = np.full(7, -1.0)
            input_max = np.full(7, 1.0)
            kp = np.full(7, 3.0)
            ki = kp * 0.005
            kd = kp * 0.001
            control_freq = 100
            velocity_limits = None
            interpolator = None

            def __init__(self, model):
                self.current_vel = np.ones(7)
                self.last_err = np.ones(7)
                self.summed_err = np.ones(7)
                self.derr_buf = Buffer()
                self.saturated = True
                self.last_joint_vel = np.ones(7)
                self.torques = np.ones(7)
                self.new_update = False
                self.goal_vel = np.ones(7)
                self.joint_index = {
                    "joints": np.arange(7),
                    "qpos": np.arange(7),
                    "qvel": np.arange(7),
                }
                self.actuator_limits = (
                    np.asarray(model.actuator_ctrlrange[:, 0], dtype=float).copy(),
                    np.asarray(model.actuator_ctrlrange[:, 1], dtype=float).copy(),
                )

            def update_initial_joints(self, joints):
                self.initial_joint = np.asarray(joints).copy()

            def reset_goal(self):
                self.goal_vel = np.zeros(7)

            def update(self, force=False):
                self.updated_force = force

        def environment(controller_name):
            simulation = Sim()
            controller = Controller(simulation.model)
            controller.name = controller_name
            robot = SimpleNamespace(
                controller=controller,
                _ref_joint_pos_indexes=np.arange(7),
                _ref_joint_vel_indexes=np.arange(7),
                _ref_joint_indexes=np.arange(7),
                _ref_joint_actuator_indexes=np.arange(7),
            )
            inner = SimpleNamespace(
                action_dim=8,
                control_freq=100,
                control_timestep=0.01,
                model_timestep=0.002,
                timestep=20,
                cur_time=1.0,
                done=False,
            )
            wrapper = SimpleNamespace(sim=simulation, robots=[robot], env=inner)
            wrapper._post_process = lambda: None
            wrapper._update_observables = lambda force=False: None
            wrapper.check_success = lambda: False
            return wrapper

        source = environment("OSC_POSE")
        target = environment("JOINT_VELOCITY")
        source.sim.data.time = 0.25
        source.sim.data.qpos[:] = np.linspace(-0.2, 0.2, 7)
        source.sim.data.qvel[:] = np.linspace(-0.1, 0.1, 7)
        source.sim.data.ctrl[:] = np.linspace(-0.3, 0.3, 7)
        source.sim.data.qacc_warmstart[:] = np.linspace(0.7, 1.3, 7)
        target.sim.data.ctrl[:] = -9.0
        target.sim.data.qacc_warmstart[:] = 8.0
        state = source.sim.get_state().flatten().copy()
        result = self.restore(
            source,
            target,
            state,
            max_arm_qpos_error_rad=1e-10,
            max_arm_qvel_error_rad_s=1e-10,
        )
        self.assertTrue(result["official_integration_state_available"])
        self.assertIsNotNone(result["compiled_mjb_sha256"])
        self.assertEqual(
            result["official_integration_state_sha256"],
            result["target_official_integration_state_sha256"],
        )
        np.testing.assert_array_equal(target.sim.data.ctrl, source.sim.data.ctrl)
        np.testing.assert_array_equal(
            target.sim.data.qacc_warmstart, source.sim.data.qacc_warmstart
        )


if __name__ == "__main__":
    unittest.main()
