from __future__ import annotations

import unittest

from tests.test_poisson_prephysics_control_wrapper import (
    _FakeEnv,
    _FakeIntegrationStateGuard,
    _control_env_class,
    _float_bits,
    np,
)


class ActionReferenceWrapperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ControlEnv, cls.source = _control_env_class()

    def wrapper(self, *, substeps=3):
        events = []
        wrapper = self.ControlEnv.__new__(self.ControlEnv)
        wrapper.env = _FakeEnv(events, substeps=substeps)
        wrapper.env.action_dim = 7
        wrapper.env.received_actions = []
        original = wrapper.env._pre_action

        def observe_action(action, policy_step):
            wrapper.env.received_actions.append(action.copy())
            original(action, policy_step)

        wrapper.env._pre_action = observe_action
        guard = _FakeIntegrationStateGuard(wrapper.env.sim)
        wrapper._official_integration_state_guard = lambda sim: guard
        return wrapper, events, guard

    def test_method_is_additive_and_released_step_stays_unchanged(self):
        self.assertIn(
            "    def step(self, action):\n        return self.env.step(action)",
            self.source,
        )
        self.assertIn("def step_with_action_reference_intervention(", self.source)

    def test_identity_callback_runs_once_and_native_path_gets_exact_action(self):
        wrapper, events, _ = self.wrapper(substeps=3)
        source = np.asarray([0.25, -0.5, 0.0, 0.0, 0.0, 0.0, -1.0])
        calls = []

        def identity(sim, nominal_action):
            events.append(("reference_intervention", sim.physics_steps))
            calls.append(nominal_action.copy())
            return nominal_action

        wrapper.step_with_action_reference_intervention(
            source,
            identity,
            expected_substeps=3,
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(_float_bits(calls[0]), _float_bits(source))
        self.assertEqual(len(wrapper.env.received_actions), 3)
        for received in wrapper.env.received_actions:
            self.assertEqual(_float_bits(received), _float_bits(source))
        self.assertEqual(
            [event[0] for event in events[:4]],
            ["forward", "reference_intervention", "pre", "step"],
        )

    def test_filtered_action_is_held_for_the_native_control_interval(self):
        wrapper, _, _ = self.wrapper(substeps=2)
        source = np.asarray([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.75])
        filtered = np.asarray([0.2, -0.1, 0.0, 0.0, 0.3, 0.0, 0.75])
        wrapper.step_with_action_reference_intervention(
            source,
            lambda sim, nominal: filtered,
            expected_substeps=2,
        )
        for received in wrapper.env.received_actions:
            self.assertEqual(_float_bits(received), _float_bits(filtered))

    def test_finite_out_of_range_gripper_preserves_native_clipping_path(self):
        wrapper, _, _ = self.wrapper(substeps=2)
        source = np.asarray([0.2, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0033712812594175])
        wrapper.step_with_action_reference_intervention(
            source,
            lambda sim, nominal: nominal,
            expected_substeps=2,
        )
        for received in wrapper.env.received_actions:
            self.assertEqual(_float_bits(received), _float_bits(source))

    def test_state_mutation_or_invalid_action_fails_before_physics(self):
        wrapper, _, guard = self.wrapper(substeps=1)
        source = np.zeros(7)
        initial = wrapper.env.sim.data.qpos.copy()

        def mutate(sim, nominal):
            sim.data.qpos[0] = 9.0
            return nominal

        with self.assertRaisesRegex(ValueError, "mutated MuJoCo integration state"):
            wrapper.step_with_action_reference_intervention(
                source,
                mutate,
                expected_substeps=1,
                integration_state_guard=guard,
            )
        self.assertEqual(wrapper.env.sim.physics_steps, 0)
        self.assertEqual(_float_bits(wrapper.env.sim.data.qpos), _float_bits(initial))

        wrapper, _, _ = self.wrapper(substeps=1)
        with self.assertRaisesRegex(ValueError, "finite action"):
            wrapper.step_with_action_reference_intervention(
                source,
                lambda sim, nominal: np.asarray([0.0] * 6 + [float("nan")]),
                expected_substeps=1,
            )
        self.assertEqual(wrapper.env.sim.physics_steps, 0)


if __name__ == "__main__":
    unittest.main()
