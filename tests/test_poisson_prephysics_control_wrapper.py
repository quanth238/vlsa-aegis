from __future__ import annotations

import ast
import copy
import math
from pathlib import Path
import struct
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]


class _FakeArray:
    def __init__(self, values):
        self.values = copy.deepcopy(values)

    @property
    def shape(self):
        if not isinstance(self.values, list):
            return ()
        if not self.values:
            return (0,)
        if isinstance(self.values[0], list):
            return (len(self.values), len(self.values[0]))
        return (len(self.values),)

    def copy(self):
        return _FakeArray(self.values)

    def tolist(self):
        return copy.deepcopy(self.values)

    def __len__(self):
        return len(self.values)

    def __iter__(self):
        return iter(self.values)

    @staticmethod
    def _indexes(key):
        if isinstance(key, _FakeArray):
            return key.values
        return key

    def __getitem__(self, key):
        if isinstance(key, tuple):
            rows, column = key
            selected = self.__getitem__(rows)
            return _FakeArray([row[column] for row in selected.values])
        key = self._indexes(key)
        if key is Ellipsis:
            return self.copy()
        if isinstance(key, list):
            return _FakeArray([self.values[index] for index in key])
        if isinstance(key, slice):
            return _FakeArray(self.values[key])
        value = self.values[key]
        return _FakeArray(value) if isinstance(value, list) else value

    def __setitem__(self, key, value):
        key = self._indexes(key)
        raw = value.values if isinstance(value, _FakeArray) else value
        if key is Ellipsis:
            self.values = copy.deepcopy(raw)
            return
        if isinstance(key, list):
            for offset, index in enumerate(key):
                self.values[index] = copy.deepcopy(raw[offset])
            return
        if isinstance(key, slice):
            self.values[key] = copy.deepcopy(raw)
            return
        self.values[key] = copy.deepcopy(raw)

    def _compare(self, other, operation):
        other = other.values if isinstance(other, _FakeArray) else other

        def recurse(first, second):
            if isinstance(first, list):
                if isinstance(second, list):
                    return [recurse(a, b) for a, b in zip(first, second)]
                return [recurse(item, second) for item in first]
            return operation(first, second)

        return _FakeArray(recurse(self.values, other))

    def __lt__(self, other):
        return self._compare(other, lambda first, second: first < second)

    def __ge__(self, other):
        return self._compare(other, lambda first, second: first >= second)

    def __gt__(self, other):
        return self._compare(other, lambda first, second: first > second)

    def __add__(self, other):
        return self._compare(other, lambda first, second: first + second)


class _FakeNumpy:
    int64 = int
    float64 = float
    uint8 = int
    nan = float("nan")

    @staticmethod
    def _convert(value, converter):
        if isinstance(value, _FakeArray):
            value = value.values
        if isinstance(value, (list, tuple, range)):
            return [_FakeNumpy._convert(item, converter) for item in value]
        if value is None:
            return None
        return converter(value)

    @staticmethod
    def asarray(value, dtype=None):
        converter = (lambda item: item) if dtype is None else dtype
        return _FakeArray(_FakeNumpy._convert(value, converter))

    @staticmethod
    def arange(start, stop=None, dtype=None):
        if stop is None:
            start, stop = 0, start
        return _FakeNumpy.asarray(range(start, stop), dtype=dtype)

    @staticmethod
    def zeros(length, dtype=float):
        return _FakeNumpy.asarray([0] * length, dtype=dtype)

    @staticmethod
    def ones(length, dtype=float):
        return _FakeNumpy.asarray([1] * length, dtype=dtype)

    @staticmethod
    def full(length, value):
        return _FakeNumpy.asarray([value] * length)

    @staticmethod
    def column_stack(columns):
        raw = [column.values for column in columns]
        return _FakeArray([list(row) for row in zip(*raw)])

    @staticmethod
    def unique(value):
        return _FakeArray(list(dict.fromkeys(value.values)))

    @staticmethod
    def isfinite(value):
        raw = value.values if isinstance(value, _FakeArray) else value

        def recurse(item):
            if isinstance(item, list):
                return [recurse(child) for child in item]
            return math.isfinite(item)

        return _FakeArray(recurse(raw))

    @staticmethod
    def _flatten(value):
        raw = value.values if isinstance(value, _FakeArray) else value
        if isinstance(raw, list):
            for item in raw:
                yield from _FakeNumpy._flatten(item)
        else:
            yield raw

    @staticmethod
    def all(value):
        return all(_FakeNumpy._flatten(value))

    @staticmethod
    def any(value):
        return any(_FakeNumpy._flatten(value))


np = _FakeNumpy()


def _assert_array_equal(test_case, first, second):
    first_values = first.tolist() if isinstance(first, _FakeArray) else first
    second_values = second.tolist() if isinstance(second, _FakeArray) else second
    test_case.assertEqual(first_values, second_values)


def _float_bits(values):
    raw = values.tolist() if isinstance(values, _FakeArray) else values
    return [struct.pack("!d", float(value)) for value in raw]


def _control_env_class():
    source = (
        ROOT / "safelibero/libero/libero/envs/env_wrapper.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ControlEnv"
    )
    namespace = {"np": np}
    exec(compile(ast.Module([class_node], []), "env_wrapper.py", "exec"), namespace)
    return namespace["ControlEnv"], source


class _FakeSim:
    def __init__(self, events):
        self.events = events
        self.model = SimpleNamespace(
            nu=9,
            actuator_ctrllimited=np.ones(9, dtype=np.uint8),
            actuator_ctrlrange=np.column_stack(
                (np.full(9, -2.0), np.full(9, 2.0))
            ),
        )
        self.data = SimpleNamespace(
            ctrl=np.zeros(9, dtype=np.float64),
            qpos=np.asarray([0.1, 0.2, 0.3], dtype=np.float64),
        )
        self.physics_steps = 0
        self.step_controls = []

    def forward(self):
        self.events.append(("forward", self.physics_steps))

    def step(self):
        self.events.append(("step", self.physics_steps))
        self.step_controls.append(self.data.ctrl.copy())
        self.physics_steps += 1


class _FakeEnv:
    ARM_INDEXES = np.arange(1, 8, dtype=np.int64)

    def __init__(self, events, *, substeps=25):
        self.events = events
        self.control_timestep = 0.05
        self.model_timestep = self.control_timestep / substeps
        self.done = False
        self.timestep = 0
        self.cur_time = 0.0
        self.sim = _FakeSim(events)
        self.robots = [
            SimpleNamespace(
                _ref_joint_actuator_indexes=self.ARM_INDEXES.copy()
            )
        ]
        self.viewer = None
        self.renderer = "mujoco"
        self.viewer_get_obs = False
        self.pre_count = 0
        self.post_count = 0
        self.success_count = 0
        self.nominal_controls = []

    def _pre_action(self, action, policy_step):
        self.events.append(("pre", self.sim.physics_steps, policy_step))
        nominal = np.asarray(
            [-0.0, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 1.25],
            dtype=np.float64,
        )
        self.sim.data.ctrl[:] = nominal
        self.nominal_controls.append(nominal.copy())
        self.pre_count += 1

    def _update_observables(self):
        self.events.append(("observable", self.sim.physics_steps))

    def _post_action(self, action):
        self.events.append(("post", self.sim.physics_steps))
        self.post_count += 1
        return 1.0, False, {"post": True}

    def _check_success(self):
        self.events.append(("success", self.sim.physics_steps))
        self.success_count += 1
        return True

    def _get_observations(self):
        return {"physics_steps": self.sim.physics_steps}


class _FakeIntegrationStateGuard:
    def __init__(self, sim):
        self.sim = sim
        self.restore_count = 0

    def capture(self):
        return {
            "qpos": self.sim.data.qpos.copy(),
            "ctrl": self.sim.data.ctrl.copy(),
        }

    def restore(self, state):
        self.sim.data.qpos[...] = state["qpos"]
        self.sim.data.ctrl[...] = state["ctrl"]
        self.restore_count += 1

    def equal(self, first, second):
        return (
            _float_bits(first["qpos"]) == _float_bits(second["qpos"])
            and _float_bits(first["ctrl"]) == _float_bits(second["ctrl"])
        )


class PrephysicsArmControlWrapperTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ControlEnv, cls.source = _control_env_class()

    def _wrapper(self, *, substeps=25):
        events = []
        wrapper = self.ControlEnv.__new__(self.ControlEnv)
        wrapper.env = _FakeEnv(events, substeps=substeps)
        guard = _FakeIntegrationStateGuard(wrapper.env.sim)
        wrapper._official_integration_state_guard = lambda sim: guard
        wrapper.test_integration_state_guard = guard
        return wrapper, events

    def test_method_is_additive_and_released_step_is_unchanged(self):
        self.assertIn(
            "    def step(self, action):\n        return self.env.step(action)",
            self.source,
        )
        self.assertIn(
            "    def step_with_arm_control_intervention(", self.source
        )
        self.assertIn("mjtState.mjSTATE_INTEGRATION", self.source)
        self.assertIn("mujoco.mj_getState", self.source)
        self.assertIn("mujoco.mj_setState", self.source)

    def test_inactive_identity_matches_nominal_and_runs_exact_substeps(self):
        wrapper, events = self._wrapper()
        received = []
        poststep = []

        def identity(sim, substep_index, nominal_arm_ctrl):
            events.append(("intervention", sim.physics_steps))
            received.append((substep_index, nominal_arm_ctrl.copy()))
            return nominal_arm_ctrl

        def after_step(sim, substep_index):
            events.append(("poststep", sim.physics_steps))
            poststep.append((substep_index, sim.physics_steps))

        observation, reward, done, info = (
            wrapper.step_with_arm_control_intervention(
                np.zeros(7),
                identity,
                poststep_callback=after_step,
                expected_substeps=25,
            )
        )

        self.assertEqual(wrapper.env.sim.physics_steps, 25)
        self.assertEqual(len(received), 25)
        self.assertEqual(
            poststep,
            [(index, index + 1) for index in range(25)],
        )
        for recorded, nominal in zip(
            wrapper.env.sim.step_controls, wrapper.env.nominal_controls
        ):
            _assert_array_equal(self, recorded, nominal)
        self.assertEqual(wrapper.env.timestep, 1)
        self.assertAlmostEqual(wrapper.env.cur_time, 0.05)
        self.assertEqual(wrapper.env.post_count, 1)
        self.assertEqual(wrapper.env.success_count, 1)
        self.assertEqual(observation, {"physics_steps": 25})
        self.assertEqual(reward, 1.0)
        self.assertTrue(done)
        self.assertEqual(info, {"post": True})

    def test_only_arm_slots_change_and_nonarm_bits_are_restored(self):
        wrapper, _ = self._wrapper(substeps=1)

        def filter_arm(sim, substep_index, nominal_arm_ctrl):
            # A callback has live simulator access, but direct writes must not
            # escape into the gripper or any other actuator slot.
            sim.data.ctrl[0] = 1.75
            sim.data.ctrl[8] = -1.75
            return nominal_arm_ctrl + 0.25

        wrapper.step_with_arm_control_intervention(
            np.zeros(7),
            filter_arm,
            expected_substeps=1,
            update_observables=False,
            collect_observations=False,
        )
        executed = wrapper.env.sim.step_controls[0]
        nominal = wrapper.env.nominal_controls[0]
        self.assertEqual(
            _float_bits(executed[[0, 8]]),
            _float_bits(nominal[[0, 8]]),
        )
        _assert_array_equal(
            self,
            executed[_FakeEnv.ARM_INDEXES],
            nominal[_FakeEnv.ARM_INDEXES] + 0.25,
        )

    def test_call_order_is_pre_action_intervention_physics_poststep(self):
        wrapper, events = self._wrapper(substeps=2)

        def intervention(sim, substep_index, nominal_arm_ctrl):
            events.append(("intervention", sim.physics_steps))
            return nominal_arm_ctrl

        def poststep(sim, substep_index):
            events.append(("poststep", sim.physics_steps))

        wrapper.step_with_arm_control_intervention(
            np.zeros(7),
            intervention,
            poststep_callback=poststep,
            expected_substeps=2,
        )
        loop_events = [event[0] for event in events[:12]]
        self.assertEqual(
            loop_events,
            [
                "forward",
                "pre",
                "intervention",
                "step",
                "poststep",
                "observable",
                "forward",
                "pre",
                "intervention",
                "step",
                "poststep",
                "observable",
            ],
        )

    def test_invalid_intervention_output_fails_before_physics_without_clipping(self):
        invalid_outputs = {
            "wrong shape": np.zeros(6),
            "nonfinite": np.asarray([0.0] * 6 + [np.nan]),
            "outside compiled limit": np.asarray([0.0] * 6 + [2.01]),
        }
        for label, output in invalid_outputs.items():
            with self.subTest(label=label):
                wrapper, _ = self._wrapper(substeps=1)

                def invalid(sim, substep_index, nominal_arm_ctrl):
                    sim.data.ctrl[8] = -1.75
                    return output

                with self.assertRaises(ValueError):
                    wrapper.step_with_arm_control_intervention(
                        np.zeros(7),
                        invalid,
                        expected_substeps=1,
                        update_observables=False,
                        collect_observations=False,
                    )
                self.assertEqual(wrapper.env.sim.physics_steps, 0)
                self.assertEqual(wrapper.env.sim.step_controls, [])
                self.assertEqual(wrapper.env.timestep, 0)
                nominal = wrapper.env.nominal_controls[0]
                _assert_array_equal(self, wrapper.env.sim.data.ctrl, nominal)

    def test_live_integration_state_mutation_is_restored_and_rejected(self):
        wrapper, _ = self._wrapper(substeps=1)
        guard = _FakeIntegrationStateGuard(wrapper.env.sim)
        original_qpos = wrapper.env.sim.data.qpos.copy()

        def mutating_intervention(sim, substep_index, nominal_arm_ctrl):
            sim.data.qpos[1] = 9.0
            return nominal_arm_ctrl

        with self.assertRaisesRegex(
            ValueError, "mutated MuJoCo integration state"
        ):
            wrapper.step_with_arm_control_intervention(
                np.zeros(7),
                mutating_intervention,
                expected_substeps=1,
                integration_state_guard=guard,
                update_observables=False,
                collect_observations=False,
            )
        self.assertEqual(wrapper.env.sim.physics_steps, 0)
        self.assertEqual(wrapper.env.sim.step_controls, [])
        self.assertEqual(wrapper.env.timestep, 0)
        self.assertEqual(guard.restore_count, 1)
        _assert_array_equal(self, wrapper.env.sim.data.qpos, original_qpos)
        _assert_array_equal(
            self,
            wrapper.env.sim.data.ctrl,
            wrapper.env.nominal_controls[0],
        )

    def test_prephysics_failure_after_partial_execution_keeps_timestep(self):
        wrapper, _ = self._wrapper(substeps=2)

        def fail_on_second_substep(sim, substep_index, nominal_arm_ctrl):
            return nominal_arm_ctrl if substep_index == 0 else np.zeros(6)

        with self.assertRaisesRegex(
            ValueError, "seven finite arm actuator controls"
        ):
            wrapper.step_with_arm_control_intervention(
                np.zeros(7),
                fail_on_second_substep,
                expected_substeps=2,
                update_observables=False,
                collect_observations=False,
            )
        self.assertEqual(wrapper.env.sim.physics_steps, 1)
        self.assertEqual(wrapper.env.timestep, 1)
        self.assertEqual(wrapper.env.cur_time, 0.0)
        self.assertEqual(wrapper.env.post_count, 0)

    def test_expected_substep_mismatch_fails_before_bookkeeping_or_control(self):
        wrapper, events = self._wrapper(substeps=25)
        with self.assertRaisesRegex(ValueError, "25 physics substeps, expected 24"):
            wrapper.step_with_arm_control_intervention(
                np.zeros(7),
                lambda sim, index, nominal: nominal,
                expected_substeps=24,
            )
        self.assertEqual(wrapper.env.timestep, 0)
        self.assertEqual(wrapper.env.sim.physics_steps, 0)
        self.assertEqual(events, [])


if __name__ == "__main__":
    unittest.main()
