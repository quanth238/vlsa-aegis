from __future__ import annotations

import ast
import math
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class _FakeArray(list):
    @property
    def shape(self):
        return (len(self),)


class _FakeNumpy:
    @staticmethod
    def isclose(first, second, rtol=0.0, atol=0.0):
        return math.isclose(first, second, rel_tol=rtol, abs_tol=atol)

    @staticmethod
    def asarray(value, dtype=float):
        return _FakeArray(dtype(item) for item in value)

    @staticmethod
    def isfinite(value):
        if isinstance(value, (list, tuple)):
            return [math.isfinite(item) for item in value]
        return math.isfinite(value)

    @staticmethod
    def all(value):
        return all(value)


class SubstepWrapperContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.source = (
            ROOT / "safelibero/libero/libero/envs/env_wrapper.py"
        ).read_text(encoding="utf-8")

    def test_released_step_delegation_is_byte_present(self) -> None:
        self.assertIn(
            "    def step(self, action):\n        return self.env.step(action)",
            self.source,
        )

    def test_substep_observer_is_additive_and_preserves_bddl_done(self) -> None:
        start = self.source.index("    def step_with_substep_callback(")
        end = self.source.index("\n    def reset(self):", start)
        method = self.source[start:end]
        self.assertIn("substeps = int(self.env.control_timestep / self.env.model_timestep)", method)
        self.assertLess(
            method.index("if substeps != expected_substeps:"),
            method.index("self.env.timestep += 1"),
        )
        self.assertIn("controller cadence gives %d physics substeps, expected %d", method)
        self.assertIn("self.env._pre_action(action, policy_step)", method)
        self.assertLess(method.index("self.env.sim.step()"), method.index("callback(self.env.sim, substep_index)"))
        self.assertIn("policy_step = False", method)
        self.assertIn("self.env.cur_time += self.env.control_timestep", method)
        self.assertIn("reward, done, info = self.env._post_action(action)", method)
        self.assertIn("done = self.env._check_success()", method)
        self.assertIn("forwarded analysis-data mirror", method)

    def test_grouped_active_step_has_one_high_level_bookkeeping_update(self) -> None:
        start = self.source.index(
            "    def step_grouped_actions_with_substep_callback("
        )
        end = self.source.index("\n    def reset(self):", start)
        method = self.source[start:end]
        self.assertEqual(method.count("self.env.timestep += 1"), 1)
        self.assertEqual(method.count("self.env._post_action(final_action)"), 1)
        self.assertEqual(method.count("self.env._check_success()"), 1)
        self.assertIn("for inner_index in range(expected_inner_updates):", method)
        self.assertIn("action_provider(self.env.sim, inner_index)", method)
        self.assertIn("for substep_index in range(substeps):", method)
        self.assertIn(
            "callback(self.env.sim, inner_index, substep_index)", method
        )
        self.assertIn("self.env.cur_time += grouped_dt", method)
        self.assertIn(
            "expected_inner_updates * self.env.control_timestep", method
        )

    def test_grouped_active_step_validates_before_timestep_mutation(self) -> None:
        start = self.source.index(
            "    def step_grouped_actions_with_substep_callback("
        )
        end = self.source.index("\n    def reset(self):", start)
        method = self.source[start:end]
        mutation = method.index("self.env.timestep += 1")
        for validation in (
            "if not callable(action_provider):",
            "substeps != expected_substeps_per_inner",
            "np.isclose(grouped_dt, expected_high_level_dt",
            "if self.env.done:",
        ):
            self.assertLess(method.index(validation), mutation)

    def test_grouped_provider_observes_each_current_inner_state(self) -> None:
        tree = ast.parse(self.source)
        class_node = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "ControlEnv"
        )
        namespace = {"np": _FakeNumpy}
        exec(compile(ast.Module([class_node], []), "env_wrapper.py", "exec"), namespace)
        control_env = namespace["ControlEnv"].__new__(namespace["ControlEnv"])
        events = []

        class FakeSim:
            def __init__(self):
                self.physics_steps = 0

            def forward(self):
                events.append(("forward", self.physics_steps))

            def step(self):
                self.physics_steps += 1
                events.append(("step", self.physics_steps))

        class FakeEnv:
            def __init__(self):
                self.control_timestep = 0.01
                self.model_timestep = 0.002
                self.action_dim = 8
                self.done = False
                self.timestep = 0
                self.cur_time = 0.0
                self.sim = FakeSim()
                self.viewer = None
                self.renderer = "mujoco"
                self.viewer_get_obs = False

            def _pre_action(self, action, policy_step):
                events.append(("pre", self.sim.physics_steps, policy_step))

            def _update_observables(self):
                events.append(("observable", self.sim.physics_steps))

            def _post_action(self, action):
                events.append(("post", self.sim.physics_steps))
                return 1.0, False, {"post": True}

            def _check_success(self):
                events.append(("success", self.sim.physics_steps))
                return True

            def _get_observations(self):
                return {"physics_steps": self.sim.physics_steps}

        control_env.env = FakeEnv()
        provider_states = []
        callback_states = []

        def provider(sim, inner_index):
            provider_states.append((inner_index, sim.physics_steps))
            events.append(("provider", sim.physics_steps))
            return [0.0] * 8

        def callback(sim, inner_index, substep_index):
            callback_states.append(
                (inner_index, substep_index, sim.physics_steps)
            )

        observation, reward, done, info = (
            control_env.step_grouped_actions_with_substep_callback(
                provider, callback
            )
        )
        self.assertEqual(
            provider_states,
            [(0, 0), (1, 5), (2, 10), (3, 15), (4, 20)],
        )
        self.assertEqual(len(callback_states), 25)
        self.assertEqual(callback_states[0], (0, 0, 1))
        self.assertEqual(callback_states[-1], (4, 4, 25))
        for _, state in provider_states:
            provider_event = events.index(("provider", state))
            first_pre = events.index(("pre", state, True))
            self.assertLess(provider_event, first_pre)
        self.assertEqual(control_env.env.timestep, 1)
        self.assertAlmostEqual(control_env.env.cur_time, 0.05)
        self.assertEqual(observation, {"physics_steps": 25})
        self.assertEqual(reward, 1.0)
        self.assertTrue(done)
        self.assertEqual(info, {"post": True})

    def test_invalid_observation_combination_fails_explicitly(self) -> None:
        tree = ast.parse(self.source)
        class_node = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "ControlEnv"
        )
        namespace = {"np": _FakeNumpy}
        exec(compile(ast.Module([class_node], []), "env_wrapper.py", "exec"), namespace)

        class FakeSim:
            def __init__(self):
                self.physics_steps = 0

            def forward(self):
                pass

            def step(self):
                self.physics_steps += 1

        class FakeEnv:
            def __init__(self):
                self.control_timestep = 0.01
                self.model_timestep = 0.002
                self.action_dim = 8
                self.done = False
                self.timestep = 0
                self.cur_time = 0.0
                self.sim = FakeSim()
                self.post_count = 0

            def _post_action(self, action):
                self.post_count += 1
                return 0.0, False, {}

        for grouped in (False, True):
            control_env = namespace["ControlEnv"].__new__(namespace["ControlEnv"])
            control_env.env = FakeEnv()
            with self.assertRaisesRegex(
                ValueError, "collect_observations requires update_observables"
            ):
                if grouped:
                    control_env.step_grouped_actions_with_substep_callback(
                        lambda sim, inner: [0.0] * 8,
                        lambda sim, inner, physics: None,
                        update_observables=False,
                        collect_observations=True,
                    )
                else:
                    control_env.step_with_substep_callback(
                        [0.0] * 8,
                        lambda sim, physics: None,
                        update_observables=False,
                        collect_observations=True,
                    )
            self.assertEqual(control_env.env.timestep, 0)
            self.assertEqual(control_env.env.cur_time, 0.0)
            self.assertEqual(control_env.env.sim.physics_steps, 0)
            self.assertEqual(control_env.env.post_count, 0)


if __name__ == "__main__":
    unittest.main()
