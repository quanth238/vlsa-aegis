from __future__ import annotations

import unittest

from tests.test_poisson_prephysics_control_wrapper import (
    _FakeEnv,
    _FakeIntegrationStateGuard,
    _control_env_class,
    _float_bits,
    np,
)


class _FakeOscController:
    def __init__(self):
        self.goal_calls = []

    def set_goal(self, pose_action):
        self.goal_calls.append(pose_action.copy())


class OscReferenceUpdateWrapperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ControlEnv, cls.source = _control_env_class()

    def wrapper(self):
        events = []
        wrapper = self.ControlEnv.__new__(self.ControlEnv)
        wrapper.env = _FakeEnv(events, substeps=25)
        wrapper.env.action_dim = 7
        controller = _FakeOscController()
        wrapper.env.robots[0].controller = controller
        wrapper.env.received_actions = []
        original = wrapper.env._pre_action

        def observe_action(action, policy_step):
            wrapper.env.received_actions.append((action.copy(), policy_step))
            original(action, policy_step)

        wrapper.env._pre_action = observe_action
        guard = _FakeIntegrationStateGuard(wrapper.env.sim)
        wrapper._official_integration_state_guard = lambda sim: guard
        return wrapper, events, controller, guard

    def test_method_is_opt_in_and_requires_native_25_substeps(self):
        self.assertIn("def step_with_osc_reference_updates(", self.source)
        self.assertIn("update_substeps = (0, 5, 10, 15, 20)", self.source)
        wrapper, _, _, _ = self.wrapper()
        wrapper.env.model_timestep = wrapper.env.control_timestep / 24
        with self.assertRaisesRegex(ValueError, "require 25 physics substeps"):
            wrapper.step_with_osc_reference_updates(np.zeros(7), lambda *_: np.zeros(7))

    def test_updates_goal_at_100hz_without_new_policy_steps(self):
        wrapper, _, controller, _ = self.wrapper()
        source = np.asarray([0.1, -0.2, 0.3, 0.0, 0.0, 0.0, -1.003])
        updated_five = np.asarray([0.4, 0.5, -0.6, 0.1, 0.0, -0.1, 1.004])
        updated_fifteen = np.asarray([-0.7, 0.2, 0.1, 0.0, 0.3, 0.0, -1.007])
        calls = []

        def update(sim, substep_index, current_action):
            calls.append((substep_index, current_action.copy()))
            if substep_index == 0:
                return source
            if substep_index == 5:
                return updated_five
            if substep_index == 15:
                return updated_fifteen
            return None

        wrapper.step_with_osc_reference_updates(
            source,
            update,
            expected_substeps=25,
        )
        self.assertEqual([index for index, _ in calls], [0, 5, 10, 15, 20])
        self.assertEqual(len(controller.goal_calls), 2)
        self.assertEqual(_float_bits(controller.goal_calls[0]), _float_bits(updated_five[:6]))
        self.assertEqual(_float_bits(controller.goal_calls[1]), _float_bits(updated_fifteen[:6]))
        self.assertEqual(len(wrapper.env.received_actions), 25)
        self.assertTrue(wrapper.env.received_actions[0][1])
        self.assertTrue(all(not row[1] for row in wrapper.env.received_actions[1:]))
        expected = []
        for index in range(25):
            if index < 5:
                expected.append(source)
            elif index < 15:
                expected.append(updated_five)
            else:
                expected.append(updated_fifteen)
        for (received, _), wanted in zip(wrapper.env.received_actions, expected):
            self.assertEqual(_float_bits(received), _float_bits(wanted))

    def test_mid_interval_none_retains_full_action_and_zero_requires_action(self):
        wrapper, _, controller, _ = self.wrapper()
        source = np.asarray([0.2, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0033712812594175])
        wrapper.step_with_osc_reference_updates(
            source,
            lambda sim, index, current: source if index == 0 else None,
        )
        self.assertEqual(controller.goal_calls, [])
        for received, _ in wrapper.env.received_actions:
            self.assertEqual(_float_bits(received), _float_bits(source))

        wrapper, _, _, _ = self.wrapper()
        with self.assertRaisesRegex(ValueError, "substep zero"):
            wrapper.step_with_osc_reference_updates(
                source,
                lambda sim, index, current: None,
            )
        self.assertEqual(wrapper.env.sim.physics_steps, 0)
        self.assertEqual(wrapper.env.timestep, 0)

    def test_callback_state_mutation_is_restored_and_rejected(self):
        wrapper, _, _, guard = self.wrapper()
        source = np.zeros(7)
        initial_qpos = wrapper.env.sim.data.qpos.copy()

        def update(sim, substep_index, current):
            if substep_index == 5:
                sim.data.qpos[0] = 4.0
            return source

        with self.assertRaisesRegex(ValueError, "mutated MuJoCo integration state"):
            wrapper.step_with_osc_reference_updates(source, update)
        self.assertEqual(wrapper.env.sim.physics_steps, 5)
        self.assertEqual(wrapper.env.timestep, 1)
        self.assertEqual(guard.restore_count, 1)
        self.assertEqual(_float_bits(wrapper.env.sim.data.qpos), _float_bits(initial_qpos))


if __name__ == "__main__":
    unittest.main()
