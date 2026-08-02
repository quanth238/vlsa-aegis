from __future__ import annotations

import importlib
import importlib.util
import json
from types import SimpleNamespace
import unittest


class DependencyLightTorqueSensitivityImportTest(unittest.TestCase):
    def test_module_import_does_not_eagerly_require_numpy_or_mujoco(self) -> None:
        module = importlib.import_module(
            "main.poisson_fullbody.post_osc_torque_sensitivity"
        )
        self.assertTrue(callable(module.capture_post_osc_integration_state))
        self.assertTrue(callable(module.estimate_post_osc_torque_sensitivity))


NUMPY_PRESENT = importlib.util.find_spec("numpy") is not None
MUJOCO_PRESENT = importlib.util.find_spec("mujoco") is not None


@unittest.skipUnless(NUMPY_PRESENT, "NumPy unavailable for structural dynamics fake")
class PostOscTorqueSensitivityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import numpy as np

        cls.np = np
        cls.module = importlib.import_module(
            "main.poisson_fullbody.post_osc_torque_sensitivity"
        )

    def setUp(self) -> None:
        np = self.np

        class FakeModel:
            def __init__(self) -> None:
                self.nq = 7
                self.nv = 7
                self.nu = 8
                self.opt = SimpleNamespace(timestep=0.002)
                self.actuator_ctrlrange = np.column_stack(
                    (np.full(8, -4.0), np.full(8, 4.0))
                )
                self.actuator_ctrllimited = np.ones(8, dtype=np.int32)
                self.mass = np.diag(np.arange(1.0, 8.0))
                self.nonlinear_torque_coefficient = 0.0
                self.force_contact = False
                self.produce_nan = False

        class FakeData:
            def __init__(self, model) -> None:
                self.time = 0.0
                self.qpos = np.zeros(model.nq, dtype=np.float64)
                self.qvel = np.zeros(model.nv, dtype=np.float64)
                self.qacc_warmstart = np.zeros(model.nv, dtype=np.float64)
                self.ctrl = np.zeros(model.nu, dtype=np.float64)
                self.qM = model.mass.copy()
                self.actuator_moment = np.zeros(
                    (model.nu, model.nv), dtype=np.float64
                )
                self.actuator_moment[:7, :] = np.eye(7)
                self.ncon = 0

        counters = {"forward": 0, "step": 0}
        state_size = 1 + 7 + 7 + 7 + 8

        def get_state(model, data, output, specification):
            del model, specification
            output[:] = np.concatenate(
                (
                    np.array([data.time]),
                    data.qpos,
                    data.qvel,
                    data.qacc_warmstart,
                    data.ctrl,
                )
            )

        def set_state(model, data, state, specification):
            del specification
            cursor = 0
            data.time = float(state[cursor])
            cursor += 1
            data.qpos[:] = state[cursor : cursor + model.nq]
            cursor += model.nq
            data.qvel[:] = state[cursor : cursor + model.nv]
            cursor += model.nv
            data.qacc_warmstart[:] = state[cursor : cursor + model.nv]
            cursor += model.nv
            data.ctrl[:] = state[cursor : cursor + model.nu]

        def forward(model, data):
            counters["forward"] += 1
            data.qM[:] = model.mass
            data.actuator_moment[:] = 0.0
            data.actuator_moment[:7, :] = np.eye(7)
            # This deliberately alters an integration-state field.  The helper
            # must restore the exact snapshot after refreshing derived fields.
            data.qacc_warmstart[:] += 100.0

        def step(model, data):
            counters["step"] += 1
            torque = data.ctrl[:7]
            generalized = torque + model.nonlinear_torque_coefficient * torque**3
            acceleration = np.linalg.solve(model.mass, generalized)
            data.qvel[:] += model.opt.timestep * acceleration
            data.qpos[:] += model.opt.timestep * data.qvel
            data.qacc_warmstart[:] = acceleration
            data.time += model.opt.timestep
            data.ncon = int(model.force_contact)
            if model.produce_nan:
                data.qvel[0] = np.nan

        def full_mass(model, destination, compact):
            del model
            destination[:] = compact

        self.model = FakeModel()
        self.data = FakeData(self.model)
        self.data.time = 0.25
        self.data.qpos[:] = np.linspace(0.1, 0.7, 7)
        self.data.qvel[:] = np.linspace(-0.3, 0.3, 7)
        self.data.qacc_warmstart[:] = np.linspace(1.0, 2.0, 7)
        self.data.ctrl[:] = np.array(
            [0.2, -0.4, 0.6, -0.8, 1.0, -1.2, 1.4, 0.73]
        )
        self.sim = SimpleNamespace(model=self.model, data=self.data)
        self.counters = counters
        self.mujoco = SimpleNamespace(
            mjtState=SimpleNamespace(mjSTATE_INTEGRATION=8191),
            MjModel=FakeModel,
            MjData=FakeData,
            mj_stateSize=lambda model, specification: state_size,
            mj_getState=get_state,
            mj_setState=set_state,
            mj_forward=forward,
            mj_step=step,
            mj_fullM=full_mass,
        )

    def state(self):
        np = self.np
        output = np.empty(
            self.mujoco.mj_stateSize(
                self.model, self.mujoco.mjtState.mjSTATE_INTEGRATION
            )
        )
        self.mujoco.mj_getState(
            self.model,
            self.data,
            output,
            self.mujoco.mjtState.mjSTATE_INTEGRATION,
        )
        return output

    def capture(self):
        return self.module.capture_post_osc_integration_state(
            self.sim,
            arm_actuator_ids=list(range(7)),
            arm_qpos_indices=list(range(7)),
            arm_qvel_indices=list(range(7)),
            mujoco_module=self.mujoco,
            numpy_module=self.np,
        )

    def test_capture_is_exact_read_only_and_does_not_touch_live_data(self) -> None:
        np = self.np
        before = self.state().copy()
        snapshot = self.capture()
        np.testing.assert_array_equal(snapshot.integration_state, before)
        np.testing.assert_array_equal(snapshot.all_ctrl, self.data.ctrl)
        np.testing.assert_array_equal(
            snapshot.nominal_arm_torque_nm, self.data.ctrl[:7]
        )
        self.assertFalse(snapshot.integration_state.flags.writeable)
        self.assertFalse(snapshot.all_ctrl.flags.writeable)
        self.assertRegex(snapshot.integration_state_sha256, r"^[0-9a-f]{64}$")
        np.testing.assert_array_equal(self.state(), before)
        self.assertEqual(self.counters, {"forward": 0, "step": 0})

    def test_one_substep_changes_only_arm_ctrl_on_clone_and_runs_hook(self) -> None:
        np = self.np
        before = self.state().copy()
        snapshot = self.capture()
        candidate = np.linspace(-0.7, 0.7, 7)
        hook_calls = []

        def hook(model, clone):
            self.assertIs(model, self.model)
            self.assertIsNot(clone, self.data)
            hook_calls.append(True)
            return {
                "ncon": int(clone.ncon),
                "gripper_ctrl": float(clone.ctrl[7]),
            }

        transition = self.module.clone_one_substep_transition(
            self.model,
            snapshot=snapshot,
            arm_torque=candidate,
            selected_contact_hook=hook,
            mujoco_module=self.mujoco,
            numpy_module=np,
        )
        expected_qvel = before[1 + 7 : 1 + 7 + 7] + 0.002 * (
            candidate / np.arange(1.0, 8.0)
        )
        np.testing.assert_allclose(transition.next_arm_qvel, expected_qvel)
        self.assertRegex(transition.next_integration_state_sha256, r"^[0-9a-f]{64}$")
        self.assertEqual(transition.selected_contact_data["gripper_ctrl"], 0.73)
        self.assertEqual(hook_calls, [True])
        self.assertEqual(self.counters, {"forward": 1, "step": 1})
        np.testing.assert_array_equal(self.state(), before)

    def test_sensitivity_matches_exact_linear_dynamics_and_preserves_live(self) -> None:
        np = self.np
        before = self.state().copy()
        snapshot = self.capture()
        result = self.module.estimate_post_osc_torque_sensitivity(
            self.sim,
            snapshot=snapshot,
            torque_epsilon_nm=0.01,
            agreement_atol=1e-12,
            agreement_rtol=1e-10,
            mujoco_module=self.mujoco,
            numpy_module=np,
        )
        expected_sensitivity = 0.002 * np.diag(1.0 / np.arange(1.0, 8.0))
        np.testing.assert_allclose(
            result.torque_to_next_arm_qvel_sensitivity,
            expected_sensitivity,
            atol=1e-14,
        )
        expected_nominal = self.data.qvel + expected_sensitivity @ self.data.ctrl[:7]
        np.testing.assert_allclose(
            result.nominal_next_arm_qvel_rad_s, expected_nominal, atol=1e-14
        )
        self.assertEqual(self.counters["step"], 29)
        self.assertTrue(result.analytic_free_dynamics_diagnostic["available"])
        self.assertLess(
            result.analytic_free_dynamics_diagnostic["maximum_absolute_error"],
            1e-14,
        )
        np.testing.assert_array_equal(self.state(), before)
        self.assertEqual(float(self.data.ctrl[7]), 0.73)
        self.assertEqual(len(result.finite_difference_column_stencils), 7)
        self.assertTrue(
            all(
                column["full_resolution"]["stencil"] == "centered"
                for column in result.finite_difference_column_stencils
            )
        )
        json.dumps(result.finite_difference_column_stencils, allow_nan=False)

    def test_upper_bound_uses_backward_two_resolution_stencil(self) -> None:
        np = self.np
        self.data.ctrl[0] = 3.99
        before = self.state().copy()
        snapshot = self.capture()
        result = self.module.estimate_post_osc_torque_sensitivity(
            self.sim,
            snapshot=snapshot,
            torque_epsilon_nm=0.02,
            agreement_atol=1e-12,
            agreement_rtol=1e-10,
            mujoco_module=self.mujoco,
            numpy_module=np,
        )
        first = result.finite_difference_column_stencils[0]
        self.assertEqual(first["full_resolution"]["stencil"], "backward")
        self.assertEqual(first["half_resolution"]["stencil"], "backward")
        np.testing.assert_allclose(
            first["full_resolution"]["sample_deltas_nm"], (-0.02, 0.0)
        )
        np.testing.assert_allclose(
            first["half_resolution"]["sample_deltas_nm"], (-0.01, 0.0)
        )
        self.assertTrue(first["bound_adapted"])
        expected = 0.002 * np.diag(1.0 / np.arange(1.0, 8.0))
        np.testing.assert_allclose(
            result.torque_to_next_arm_qvel_sensitivity, expected, atol=1e-14
        )
        # One nominal step, two one-sided samples, and 24 centered samples.
        self.assertEqual(self.counters["step"], 27)
        np.testing.assert_array_equal(self.state(), before)
        self.assertEqual(float(self.data.ctrl[7]), 0.73)

    def test_lower_bound_uses_forward_two_resolution_stencil(self) -> None:
        np = self.np
        self.data.ctrl[1] = -3.99
        before = self.state().copy()
        snapshot = self.capture()
        result = self.module.estimate_post_osc_torque_sensitivity(
            self.sim,
            snapshot=snapshot,
            torque_epsilon_nm=0.02,
            agreement_atol=1e-12,
            agreement_rtol=1e-10,
            mujoco_module=self.mujoco,
            numpy_module=np,
        )
        second = result.finite_difference_column_stencils[1]
        self.assertEqual(second["full_resolution"]["stencil"], "forward")
        self.assertEqual(second["half_resolution"]["stencil"], "forward")
        np.testing.assert_allclose(
            second["full_resolution"]["sample_deltas_nm"], (0.0, 0.02)
        )
        np.testing.assert_allclose(
            second["half_resolution"]["sample_deltas_nm"], (0.0, 0.01)
        )
        self.assertTrue(second["bound_adapted"])
        np.testing.assert_array_equal(self.state(), before)
        self.assertEqual(float(self.data.ctrl[7]), 0.73)

    def test_epsilon_and_half_epsilon_disagreement_fails_closed(self) -> None:
        self.model.nonlinear_torque_coefficient = 10.0
        self.data.ctrl[0] = 4.0
        snapshot = self.capture()
        with self.assertRaisesRegex(
            self.module.TorqueSensitivityError, "epsilon and half-epsilon disagree"
        ):
            self.module.estimate_post_osc_torque_sensitivity(
                self.model,
                snapshot=snapshot,
                torque_epsilon_nm=0.5,
                agreement_atol=1e-14,
                agreement_rtol=1e-12,
                mujoco_module=self.mujoco,
                numpy_module=self.np,
            )

    def test_unrepresentable_half_resolution_fails_closed_without_steps(self) -> None:
        smallest_positive = self.np.nextafter(0.0, 1.0)
        snapshot = self.capture()
        with self.assertRaisesRegex(
            self.module.TorqueSensitivityError, "no positive perturbation resolution"
        ):
            self.module.estimate_post_osc_torque_sensitivity(
                self.model,
                snapshot=snapshot,
                torque_epsilon_nm=smallest_positive,
                mujoco_module=self.mujoco,
                numpy_module=self.np,
            )
        self.assertEqual(self.counters["step"], 0)

    def test_nonfinite_one_step_state_fails_closed(self) -> None:
        self.model.produce_nan = True
        snapshot = self.capture()
        with self.assertRaisesRegex(
            self.module.TorqueSensitivityError, "non-finite"
        ):
            self.module.clone_one_substep_transition(
                self.model,
                snapshot=snapshot,
                arm_torque=snapshot.nominal_arm_torque_nm,
                mujoco_module=self.mujoco,
                numpy_module=self.np,
            )

    def test_contact_marks_free_dynamics_formula_diagnostic_only_unavailable(self) -> None:
        self.model.force_contact = True
        self.data.ncon = 1
        snapshot = self.capture()
        result = self.module.estimate_post_osc_torque_sensitivity(
            self.model,
            snapshot=snapshot,
            torque_epsilon_nm=0.01,
            mujoco_module=self.mujoco,
            numpy_module=self.np,
        )
        diagnostic = result.analytic_free_dynamics_diagnostic
        self.assertFalse(diagnostic["available"])
        self.assertEqual(diagnostic["reason"], "contact_present")
        self.assertFalse(diagnostic["acceptance_authority"])

    def test_unlimited_selected_actuator_is_rejected(self) -> None:
        self.model.actuator_ctrllimited[3] = 0
        with self.assertRaisesRegex(ValueError, "enabled ctrl limits"):
            self.capture()

    def test_candidate_postcheck_rejects_out_of_bounds_torque(self) -> None:
        snapshot = self.capture()
        invalid = self.np.zeros(7)
        invalid[-1] = 4.1
        with self.assertRaisesRegex(ValueError, "violates compiled"):
            self.module.clone_one_substep_transition(
                self.model,
                snapshot=snapshot,
                arm_torque=invalid,
                mujoco_module=self.mujoco,
                numpy_module=self.np,
            )
        self.assertEqual(self.counters["step"], 0)

    def test_live_simulator_must_still_match_snapshot(self) -> None:
        snapshot = self.capture()
        self.data.qvel[0] += 1e-6
        with self.assertRaisesRegex(ValueError, "no longer matches"):
            self.module.clone_one_substep_transition(
                self.sim,
                snapshot=snapshot,
                arm_torque=snapshot.nominal_arm_torque_nm,
                mujoco_module=self.mujoco,
                numpy_module=self.np,
            )
        with self.assertRaisesRegex(ValueError, "no longer matches"):
            self.module.estimate_post_osc_torque_sensitivity(
                self.sim,
                snapshot=snapshot,
                torque_epsilon_nm=0.01,
                mujoco_module=self.mujoco,
                numpy_module=self.np,
            )
        self.assertEqual(self.counters["step"], 0)


@unittest.skipUnless(
    NUMPY_PRESENT and MUJOCO_PRESENT, "official MuJoCo numerical dependency unavailable"
)
class OfficialMujocoPostOscTorqueSensitivityTest(unittest.TestCase):
    def test_exact_clone_derivative_matches_free_dynamics_diagnostic(self) -> None:
        import mujoco
        import numpy as np

        from main.poisson_fullbody.post_osc_torque_sensitivity import (
            capture_post_osc_integration_state,
            estimate_post_osc_torque_sensitivity,
        )

        axes = ("0 0 1", "0 1 0", "1 0 0")
        bodies = ""
        for index in range(1, 8):
            position = "" if index == 1 else ' pos="0 0 0.08"'
            bodies += (
                '<body name="b%d"%s><joint name="j%d" axis="%s" damping="0"/>'
                '<geom type="capsule" size="0.01 0.03" mass="1" '
                'contype="0" conaffinity="0"/>'
                % (index, position, index, axes[(index - 1) % len(axes)])
            )
        bodies += "</body>" * 7
        motors = "".join(
            '<motor joint="j%d" ctrlrange="-4 4"/>' % index
            for index in range(1, 8)
        )
        # The eighth control stands in for a non-arm/gripper actuator.  Zero
        # gear keeps it dynamically irrelevant while exercising preservation.
        xml = (
            '<mujoco><option timestep="0.002" gravity="0 0 0" '
            'integrator="Euler"/><worldbody>%s</worldbody><actuator>%s'
            '<motor joint="j7" gear="0" ctrlrange="-1 1"/>'
            '</actuator></mujoco>'
        ) % (bodies, motors)
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        data.qpos[:] = np.linspace(-0.1, 0.1, 7)
        data.qvel[:] = np.linspace(-0.2, 0.2, 7)
        mujoco.mj_forward(model, data)
        data.ctrl[:] = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, 0.33]
        specification = int(mujoco.mjtState.mjSTATE_INTEGRATION)
        before = np.empty(mujoco.mj_stateSize(model, specification))
        mujoco.mj_getState(model, data, before, specification)

        sim = SimpleNamespace(model=model, data=data)
        snapshot = capture_post_osc_integration_state(
            sim,
            arm_actuator_ids=list(range(7)),
            arm_qpos_indices=list(range(7)),
            arm_qvel_indices=list(range(7)),
        )
        result = estimate_post_osc_torque_sensitivity(
            sim,
            snapshot=snapshot,
            torque_epsilon_nm=1e-3,
            agreement_atol=1e-8,
            agreement_rtol=1e-3,
        )
        self.assertEqual(result.torque_to_next_arm_qvel_sensitivity.shape, (7, 7))
        self.assertTrue(np.all(np.isfinite(result.nominal_next_arm_qvel_rad_s)))
        diagnostic = result.analytic_free_dynamics_diagnostic
        self.assertTrue(diagnostic["available"], diagnostic)
        self.assertLess(diagnostic["maximum_absolute_error"], 1e-7)
        after = np.empty_like(before)
        mujoco.mj_getState(model, data, after, specification)
        np.testing.assert_array_equal(after, before)
        self.assertEqual(float(data.ctrl[7]), 0.33)
        # The nominal clone must be the literal transition the untouched live
        # simulator would take from the same callback boundary.
        mujoco.mj_step(model, data)
        np.testing.assert_array_equal(
            result.nominal_transition.next_qpos, data.qpos
        )
        np.testing.assert_array_equal(
            result.nominal_transition.next_qvel, data.qvel
        )

    def test_compiled_upper_and_lower_limits_use_one_sided_stencils(self) -> None:
        import mujoco
        import numpy as np

        from main.poisson_fullbody.post_osc_torque_sensitivity import (
            capture_post_osc_integration_state,
            estimate_post_osc_torque_sensitivity,
        )

        axes = ("0 0 1", "0 1 0", "1 0 0")
        bodies = ""
        for index in range(1, 8):
            position = "" if index == 1 else ' pos="0 0 0.08"'
            bodies += (
                '<body name="bounded_b%d"%s><joint name="bounded_j%d" '
                'axis="%s" damping="0"/><geom type="capsule" '
                'size="0.01 0.03" mass="1" contype="0" conaffinity="0"/>'
                % (index, position, index, axes[(index - 1) % len(axes)])
            )
        bodies += "</body>" * 7
        motors = "".join(
            '<motor joint="bounded_j%d" ctrlrange="-4 4"/>' % index
            for index in range(1, 8)
        )
        xml = (
            '<mujoco><option timestep="0.002" gravity="0 0 0" '
            'integrator="Euler"/><worldbody>%s</worldbody><actuator>%s'
            '</actuator></mujoco>'
        ) % (bodies, motors)
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        data.ctrl[:] = [4.0, -4.0, 0.3, -0.4, 0.5, -0.6, 0.7]
        specification = int(mujoco.mjtState.mjSTATE_INTEGRATION)
        before = np.empty(mujoco.mj_stateSize(model, specification))
        mujoco.mj_getState(model, data, before, specification)

        sim = SimpleNamespace(model=model, data=data)
        snapshot = capture_post_osc_integration_state(
            sim,
            arm_actuator_ids=list(range(7)),
            arm_qpos_indices=list(range(7)),
            arm_qvel_indices=list(range(7)),
        )
        result = estimate_post_osc_torque_sensitivity(
            sim,
            snapshot=snapshot,
            torque_epsilon_nm=1e-3,
            agreement_atol=1e-8,
            agreement_rtol=1e-3,
        )
        stencils = result.finite_difference_column_stencils
        self.assertEqual(stencils[0]["full_resolution"]["stencil"], "backward")
        self.assertEqual(stencils[1]["full_resolution"]["stencil"], "forward")
        np.testing.assert_allclose(
            stencils[0]["full_resolution"]["sample_deltas_nm"],
            (-1e-3, 0.0),
            atol=1e-15,
        )
        np.testing.assert_allclose(
            stencils[1]["half_resolution"]["sample_deltas_nm"],
            (0.0, 5e-4),
            atol=1e-15,
        )
        self.assertTrue(
            np.all(np.isfinite(result.torque_to_next_arm_qvel_sensitivity))
        )
        after = np.empty_like(before)
        mujoco.mj_getState(model, data, after, specification)
        np.testing.assert_array_equal(after, before)


if __name__ == "__main__":
    unittest.main()
