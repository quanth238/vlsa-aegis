import os
import numpy as np
import robosuite as suite
import matplotlib.cm as cm

from robosuite.utils.errors import RandomizationError

import libero.libero.envs.bddl_utils as BDDLUtils
from libero.libero.envs import *


class ControlEnv:
    def __init__(
        self,
        bddl_file_name,
        robots=["Panda"],
        controller="OSC_POSE",
        gripper_types="default",
        initialization_noise=None,
        use_camera_obs=True,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names=[
            "agentview",
            "robot0_eye_in_hand",
            "backview"
        ],
        camera_heights=128,
        camera_widths=128,
        camera_depths=False,
        camera_segmentations=None,
        renderer="mujoco",
        renderer_config=None,
        **kwargs,
    ):
        assert os.path.exists(
            bddl_file_name
        ), f"[error] {bddl_file_name} does not exist!"

        controller_configs = suite.load_controller_config(default_controller=controller)

        problem_info = BDDLUtils.get_problem_info(bddl_file_name)
        # Check if we're using a multi-armed environment and use env_configuration argument if so

        # Create environment
        self.problem_name = problem_info["problem_name"]
        self.domain_name = problem_info["domain_name"]
        self.language_instruction = problem_info["language_instruction"]
        self.env = TASK_MAPPING[self.problem_name](
            bddl_file_name,
            robots=robots,
            controller_configs=controller_configs,
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
            **kwargs,
        )

    @property
    def obj_of_interest(self):
        return self.env.obj_of_interest

    def step(self, action):
        return self.env.step(action)

    def step_with_substep_callback(
        self,
        action,
        callback,
        *,
        expected_substeps=None,
        update_observables=True,
        collect_observations=True,
    ):
        """Execute one control action and observe every MuJoCo substep.

        This opt-in method mirrors robosuite 1.4.1's ``MujocoEnv.step`` loop
        and calls ``callback`` immediately after each ``sim.step()``.  The
        callback must use a forwarded analysis-data mirror for integrated-state
        geometry; MuJoCo's live derived arrays may still be solver-phase data.
        The ordinary ``step`` method above remains the exact released
        delegation.
        """
        if not callable(callback):
            raise TypeError("callback must be callable")
        if collect_observations and not update_observables:
            raise ValueError(
                "collect_observations requires update_observables"
            )
        substeps = int(self.env.control_timestep / self.env.model_timestep)
        if expected_substeps is not None:
            if (
                isinstance(expected_substeps, bool)
                or not isinstance(expected_substeps, int)
                or expected_substeps <= 0
            ):
                raise ValueError("expected_substeps must be a positive integer")
            if substeps != expected_substeps:
                raise ValueError(
                    "controller cadence gives %d physics substeps, expected %d"
                    % (substeps, expected_substeps)
                )
        if self.env.done:
            raise ValueError("executing action in terminated episode")

        self.env.timestep += 1
        policy_step = True
        for substep_index in range(substeps):
            self.env.sim.forward()
            self.env._pre_action(action, policy_step)
            self.env.sim.step()
            callback(self.env.sim, substep_index)
            if update_observables:
                self.env._update_observables()
            policy_step = False

        self.env.cur_time += self.env.control_timestep
        reward, done, info = self.env._post_action(action)
        # BDDLBaseDomain.step replaces robosuite's horizon done flag with task
        # success.  This callback path bypasses that override, so reproduce it
        # explicitly after the same _post_action call.
        done = self.env._check_success()
        if self.env.viewer is not None and self.env.renderer != "mujoco":
            self.env.viewer.update()
        observations = None
        if collect_observations:
            observations = (
                self.env.viewer._get_observations()
                if self.env.viewer_get_obs
                else self.env._get_observations()
            )
        return observations, reward, done, info

    def step_with_arm_control_intervention(
        self,
        action,
        intervention,
        *,
        poststep_callback=None,
        expected_substeps=None,
        integration_state_guard=None,
        update_observables=True,
        collect_observations=True,
    ):
        """Execute one action with an opt-in pre-physics arm-control filter.

        The ordinary :meth:`step` path is unchanged.  This method mirrors the
        robosuite 1.4.1 control loop, but exposes the seven arm actuator
        controls after ``_pre_action`` has produced them and immediately before
        each MuJoCo step.  ``intervention(sim, substep_index,
        nominal_arm_ctrl)`` must return exactly seven finite actuator controls
        within the compiled MuJoCo limits.  No clipping is performed.

        Only the seven arm actuator slots may change.  The complete nominal
        control vector is restored after the callback (including when the
        callback raises), and the validated return value is then written only
        to the arm slots.  This keeps gripper and other actuator controls
        bitwise identical to the values produced by robosuite.

        When the live simulator exposes the official MuJoCo model and data,
        the callback is also guarded by the complete ``mjSTATE_INTEGRATION``.
        Any non-control state mutation is restored exactly and rejected before
        physics.  ``integration_state_guard`` is an injection point for
        dependency-light tests; a guard implements ``capture()``,
        ``restore(state)``, and ``equal(first, second)``.
        """
        if not callable(intervention):
            raise TypeError("intervention must be callable")
        if poststep_callback is not None and not callable(poststep_callback):
            raise TypeError("poststep_callback must be callable or None")
        if collect_observations and not update_observables:
            raise ValueError(
                "collect_observations requires update_observables"
            )
        substeps = int(self.env.control_timestep / self.env.model_timestep)
        if expected_substeps is not None:
            if (
                isinstance(expected_substeps, bool)
                or not isinstance(expected_substeps, int)
                or expected_substeps <= 0
            ):
                raise ValueError("expected_substeps must be a positive integer")
            if substeps != expected_substeps:
                raise ValueError(
                    "controller cadence gives %d physics substeps, expected %d"
                    % (substeps, expected_substeps)
                )
        if self.env.done:
            raise ValueError("executing action in terminated episode")

        robots = getattr(self.env, "robots", None)
        if not isinstance(robots, (list, tuple)) or not robots:
            raise ValueError("environment does not expose a robot")
        arm_actuator_indexes = np.asarray(
            getattr(robots[0], "_ref_joint_actuator_indexes", None),
            dtype=np.int64,
        )
        if arm_actuator_indexes.shape != (7,):
            raise ValueError(
                "robot does not expose seven ordered arm actuator indexes"
            )
        model = self.env.sim.model
        actuator_count = int(getattr(model, "nu", -1))
        if (
            actuator_count <= 0
            or np.any(arm_actuator_indexes < 0)
            or np.any(arm_actuator_indexes >= actuator_count)
            or len(np.unique(arm_actuator_indexes)) != 7
        ):
            raise ValueError("robot arm actuator indexes are invalid")
        limited = np.asarray(
            getattr(model, "actuator_ctrllimited", None), dtype=bool
        )
        if limited.shape != (actuator_count,) or not np.all(
            limited[arm_actuator_indexes]
        ):
            raise ValueError("all arm actuators must have compiled control limits")
        compiled_limits = np.asarray(
            model.actuator_ctrlrange[arm_actuator_indexes], dtype=np.float64
        )
        if (
            compiled_limits.shape != (7, 2)
            or not np.all(np.isfinite(compiled_limits))
            or np.any(compiled_limits[:, 0] >= compiled_limits[:, 1])
        ):
            raise ValueError("compiled arm actuator limits are invalid")
        ctrl = np.asarray(self.env.sim.data.ctrl)
        if ctrl.shape != (actuator_count,):
            raise ValueError("MuJoCo control vector has unexpected shape")

        if integration_state_guard is None:
            integration_state_guard = self._official_integration_state_guard(
                self.env.sim
            )
        if integration_state_guard is not None:
            for method_name in ("capture", "restore", "equal"):
                if not callable(
                    getattr(integration_state_guard, method_name, None)
                ):
                    raise TypeError(
                        "integration_state_guard must implement capture, restore, and equal"
                    )

        initial_timestep = self.env.timestep
        completed_physics_substeps = 0
        self.env.timestep += 1
        policy_step = True
        for substep_index in range(substeps):
            try:
                self.env.sim.forward()
                self.env._pre_action(action, policy_step)
                nominal_full_ctrl = np.asarray(self.env.sim.data.ctrl).copy()
                if (
                    nominal_full_ctrl.shape != (actuator_count,)
                    or not np.all(np.isfinite(nominal_full_ctrl))
                ):
                    raise ValueError(
                        "robosuite produced an invalid nominal control"
                    )
                nominal_arm_ctrl = nominal_full_ctrl[
                    arm_actuator_indexes
                ].copy()
                integration_state_before = (
                    integration_state_guard.capture()
                    if integration_state_guard is not None
                    else None
                )
                integration_state_mutated = False
                try:
                    filtered_arm_ctrl = np.asarray(
                        intervention(
                            self.env.sim,
                            substep_index,
                            nominal_arm_ctrl.copy(),
                        ),
                        dtype=np.float64,
                    ).copy()
                finally:
                    # The intervention receives the live simulator for
                    # read-only state access. Restore every nominal ctrl slot
                    # before comparing the rest of the integration state, so
                    # direct ctrl writes are ignored rather than executed.
                    self.env.sim.data.ctrl[...] = nominal_full_ctrl
                    if integration_state_guard is not None:
                        try:
                            integration_state_after = (
                                integration_state_guard.capture()
                            )
                        except BaseException:
                            integration_state_guard.restore(
                                integration_state_before
                            )
                            raise
                        if not integration_state_guard.equal(
                            integration_state_before,
                            integration_state_after,
                        ):
                            integration_state_guard.restore(
                                integration_state_before
                            )
                            integration_state_mutated = True
                if integration_state_mutated:
                    raise ValueError(
                        "intervention mutated MuJoCo integration state"
                    )
                if filtered_arm_ctrl.shape != (7,) or not np.all(
                    np.isfinite(filtered_arm_ctrl)
                ):
                    raise ValueError(
                        "intervention must return seven finite arm actuator controls"
                    )
                if np.any(filtered_arm_ctrl < compiled_limits[:, 0]) or np.any(
                    filtered_arm_ctrl > compiled_limits[:, 1]
                ):
                    raise ValueError(
                        "intervention arm controls exceed compiled actuator limits"
                    )
            except BaseException:
                # Match ordinary bookkeeping during _pre_action, but do not
                # record an executed high-level step when the first physics
                # substep was rejected before integration. A later failure
                # retains the increment because part of the action executed.
                if completed_physics_substeps == 0:
                    self.env.timestep = initial_timestep
                raise
            self.env.sim.data.ctrl[arm_actuator_indexes] = filtered_arm_ctrl
            self.env.sim.step()
            completed_physics_substeps += 1
            if poststep_callback is not None:
                poststep_callback(self.env.sim, substep_index)
            if update_observables:
                self.env._update_observables()
            policy_step = False

        self.env.cur_time += self.env.control_timestep
        reward, done, info = self.env._post_action(action)
        done = self.env._check_success()
        if self.env.viewer is not None and self.env.renderer != "mujoco":
            self.env.viewer.update()
        observations = None
        if collect_observations:
            observations = (
                self.env.viewer._get_observations()
                if self.env.viewer_get_obs
                else self.env._get_observations()
            )
        return observations, reward, done, info

    def step_with_action_reference_intervention(
        self,
        action,
        intervention,
        *,
        poststep_callback=None,
        expected_substeps=None,
        integration_state_guard=None,
        update_observables=True,
        collect_observations=True,
    ):
        """Execute native OSC after one identity-preserving action filter.

        The callback runs once, after the ordinary first ``sim.forward()`` and
        immediately before robosuite receives the high-level action.  It may
        read the forwarded state and must return one complete finite action.
        Native controller and gripper clipping semantics remain downstream.
        When it returns the input byte-for-byte, the subsequent
        ``_pre_action`` calls and MuJoCo transitions are the released native
        OSC path; no controller is replaced and no extra forward is added.

        As with :meth:`step_with_arm_control_intervention`, the official
        integration-state guard rejects callback state mutation before any
        physics.  The ordinary :meth:`step` delegation remains unchanged.
        """
        if not callable(intervention):
            raise TypeError("intervention must be callable")
        if poststep_callback is not None and not callable(poststep_callback):
            raise TypeError("poststep_callback must be callable or None")
        if collect_observations and not update_observables:
            raise ValueError("collect_observations requires update_observables")
        substeps = int(self.env.control_timestep / self.env.model_timestep)
        if expected_substeps is not None:
            if (
                isinstance(expected_substeps, bool)
                or not isinstance(expected_substeps, int)
                or expected_substeps <= 0
            ):
                raise ValueError("expected_substeps must be a positive integer")
            if substeps != expected_substeps:
                raise ValueError(
                    "controller cadence gives %d physics substeps, expected %d"
                    % (substeps, expected_substeps)
                )
        if self.env.done:
            raise ValueError("executing action in terminated episode")
        action_dim = int(self.env.action_dim)
        nominal_action = np.asarray(action, dtype=np.float64).copy()
        if (
            nominal_action.shape != (action_dim,)
            or not np.all(np.isfinite(nominal_action))
        ):
            raise ValueError(
                "nominal action must be finite and match action_dim"
            )

        if integration_state_guard is None:
            integration_state_guard = self._official_integration_state_guard(
                self.env.sim
            )
        if integration_state_guard is not None:
            for method_name in ("capture", "restore", "equal"):
                if not callable(getattr(integration_state_guard, method_name, None)):
                    raise TypeError(
                        "integration_state_guard must implement capture, restore, and equal"
                    )

        initial_timestep = self.env.timestep
        completed_physics_substeps = 0
        filtered_action = None
        self.env.timestep += 1
        policy_step = True
        for substep_index in range(substeps):
            try:
                self.env.sim.forward()
                if policy_step:
                    integration_state_before = (
                        integration_state_guard.capture()
                        if integration_state_guard is not None
                        else None
                    )
                    integration_state_mutated = False
                    try:
                        filtered_action = np.asarray(
                            intervention(
                                self.env.sim,
                                nominal_action.copy(),
                            ),
                            dtype=np.float64,
                        ).copy()
                    finally:
                        if integration_state_guard is not None:
                            try:
                                integration_state_after = (
                                    integration_state_guard.capture()
                                )
                            except BaseException:
                                integration_state_guard.restore(
                                    integration_state_before
                                )
                                raise
                            if not integration_state_guard.equal(
                                integration_state_before,
                                integration_state_after,
                            ):
                                integration_state_guard.restore(
                                    integration_state_before
                                )
                                integration_state_mutated = True
                    if integration_state_mutated:
                        raise ValueError(
                            "intervention mutated MuJoCo integration state"
                        )
                    if (
                        filtered_action.shape != (action_dim,)
                        or not np.all(np.isfinite(filtered_action))
                    ):
                        raise ValueError(
                            "intervention must return one finite action"
                        )
                self.env._pre_action(filtered_action, policy_step)
            except BaseException:
                if completed_physics_substeps == 0:
                    self.env.timestep = initial_timestep
                raise
            self.env.sim.step()
            completed_physics_substeps += 1
            if poststep_callback is not None:
                poststep_callback(self.env.sim, substep_index)
            if update_observables:
                self.env._update_observables()
            policy_step = False

        self.env.cur_time += self.env.control_timestep
        reward, done, info = self.env._post_action(filtered_action)
        done = self.env._check_success()
        if self.env.viewer is not None and self.env.renderer != "mujoco":
            self.env.viewer.update()
        observations = None
        if collect_observations:
            observations = (
                self.env.viewer._get_observations()
                if self.env.viewer_get_obs
                else self.env._get_observations()
            )
        return observations, reward, done, info

    @staticmethod
    def _official_integration_state_guard(sim):
        """Return an exact official-state guard when MuJoCo is available."""

        try:
            import mujoco
        except ImportError:  # pragma: no cover - allocation dependency
            return None
        model_candidate = getattr(sim, "model", None)
        data_candidate = getattr(sim, "data", None)
        model = getattr(model_candidate, "_model", model_candidate)
        data = getattr(data_candidate, "_data", data_candidate)
        if not isinstance(model, mujoco.MjModel) or not isinstance(
            data, mujoco.MjData
        ):
            raise TypeError(
                "simulator does not expose official MuJoCo model/data for state guarding"
            )
        specification = int(mujoco.mjtState.mjSTATE_INTEGRATION)
        state_size = int(mujoco.mj_stateSize(model, specification))
        if state_size <= 0:
            raise ValueError("official MuJoCo integration state is empty")

        class OfficialIntegrationStateGuard:
            def capture(self):
                state = np.empty(state_size, dtype=np.float64)
                mujoco.mj_getState(model, data, state, specification)
                if not np.all(np.isfinite(state)):
                    raise ValueError(
                        "official MuJoCo integration state is non-finite"
                    )
                return state

            def restore(self, state):
                state = np.asarray(state, dtype=np.float64)
                if state.shape != (state_size,) or not np.all(
                    np.isfinite(state)
                ):
                    raise ValueError(
                        "cannot restore invalid MuJoCo integration state"
                    )
                # Refresh derived arrays after state restoration, then write
                # the official state once more so forward cannot change a bit
                # of the integration-state checkpoint.
                mujoco.mj_setState(model, data, state.copy(), specification)
                mujoco.mj_forward(model, data)
                mujoco.mj_setState(model, data, state.copy(), specification)

            def equal(self, first, second):
                first = np.asarray(first, dtype=np.float64)
                second = np.asarray(second, dtype=np.float64)
                return (
                    first.shape == (state_size,)
                    and second.shape == (state_size,)
                    and first.tobytes(order="C") == second.tobytes(order="C")
                )

        return OfficialIntegrationStateGuard()

    def step_grouped_actions_with_substep_callback(
        self,
        action_provider,
        callback,
        *,
        expected_inner_updates=5,
        expected_substeps_per_inner=5,
        expected_high_level_dt=0.05,
        update_observables=True,
        collect_observations=True,
    ):
        """Execute one registered 20 Hz action as five 100 Hz controls.

        The environment must be built with a 100 Hz controller.
        ``action_provider(sim, inner_index)`` is called from the freshly
        forwarded current state immediately before every inner update, so the
        adapter, Jacobians, field queries, and QP can be recomputed at 100 Hz
        and can fail closed before more physics. Physics runs at the model
        cadence, while episode timestep, post-processing, reward, success, and
        observation collection occur exactly once for the grouped 50 ms
        high-level action. This method is opt-in and does not change the
        released ``step`` path.
        """
        if not callable(action_provider):
            raise TypeError("action_provider must be callable")
        if not callable(callback):
            raise TypeError("callback must be callable")
        if collect_observations and not update_observables:
            raise ValueError(
                "collect_observations requires update_observables"
            )
        if (
            isinstance(expected_inner_updates, bool)
            or not isinstance(expected_inner_updates, int)
            or expected_inner_updates <= 0
        ):
            raise ValueError("expected_inner_updates must be a positive integer")
        if (
            isinstance(expected_substeps_per_inner, bool)
            or not isinstance(expected_substeps_per_inner, int)
            or expected_substeps_per_inner <= 0
        ):
            raise ValueError(
                "expected_substeps_per_inner must be a positive integer"
            )
        substeps = int(self.env.control_timestep / self.env.model_timestep)
        if substeps != expected_substeps_per_inner:
            raise ValueError(
                "controller cadence gives %d physics substeps per inner update, expected %d"
                % (substeps, expected_substeps_per_inner)
            )
        grouped_dt = expected_inner_updates * self.env.control_timestep
        if not np.isclose(grouped_dt, expected_high_level_dt, rtol=0.0, atol=1e-12):
            raise ValueError(
                "grouped controller duration is %.17g seconds, expected %.17g"
                % (grouped_dt, expected_high_level_dt)
            )
        action_dim = int(self.env.action_dim)
        if self.env.done:
            raise ValueError("executing action in terminated episode")

        self.env.timestep += 1
        final_action = None
        for inner_index in range(expected_inner_updates):
            # This forward is the same first forward that an ordinary control
            # loop performs; place the provider after it so kinematics describe
            # the current integrated state, without adding a second live
            # forward or running any physics before validation.
            self.env.sim.forward()
            action = np.asarray(
                action_provider(self.env.sim, inner_index), dtype=float
            )
            if action.shape != (action_dim,) or not np.all(np.isfinite(action)):
                raise ValueError(
                    "action_provider must return one finite action of dimension %d"
                    % action_dim
                )
            final_action = action
            policy_step = True
            for substep_index in range(substeps):
                if substep_index:
                    self.env.sim.forward()
                self.env._pre_action(action, policy_step)
                self.env.sim.step()
                callback(self.env.sim, inner_index, substep_index)
                if update_observables:
                    self.env._update_observables()
                policy_step = False

        self.env.cur_time += grouped_dt
        reward, done, info = self.env._post_action(final_action)
        done = self.env._check_success()
        if self.env.viewer is not None and self.env.renderer != "mujoco":
            self.env.viewer.update()
        observations = None
        if collect_observations:
            observations = (
                self.env.viewer._get_observations()
                if self.env.viewer_get_obs
                else self.env._get_observations()
            )
        return observations, reward, done, info

    def reset(self):
        success = False
        while not success:
            try:
                ret = self.env.reset()
                success = True
            except RandomizationError:
                pass
            finally:
                continue

        return ret

    def check_success(self):
        return self.env._check_success()

    @property
    def _visualizations(self):
        return self.env._visualizations

    @property
    def robots(self):
        return self.env.robots

    @property
    def sim(self):
        return self.env.sim

    def get_sim_state(self):
        return self.env.sim.get_state().flatten()

    def _post_process(self):
        return self.env._post_process()

    def _update_observables(self, force=False):
        self.env._update_observables(force=force)

    def set_state(self, mujoco_state):
        self.env.sim.set_state_from_flattened(mujoco_state)

    def reset_from_xml_string(self, xml_string):
        self.env.reset_from_xml_string(xml_string)

    def seed(self, seed):
        self.env.seed(seed)

    def set_init_state(self, init_state):
        return self.regenerate_obs_from_state(init_state)

    def regenerate_obs_from_state(self, mujoco_state):
        self.set_state(mujoco_state)
        self.env.sim.forward()
        self.check_success()
        self._post_process()
        self._update_observables(force=True)
        return self.env._get_observations()

    def close(self):
        self.env.close()
        del self.env


class OffScreenRenderEnv(ControlEnv):
    """
    For visualization and evaluation.
    """

    def __init__(self, **kwargs):
        # This shouldn't be customized
        kwargs["has_renderer"] = False
        kwargs["has_offscreen_renderer"] = True
        super().__init__(**kwargs)


class SegmentationRenderEnv(OffScreenRenderEnv):
    """
    This wrapper will additionally generate the segmentation mask of objects,
    which is useful for comparing attention.
    """

    def __init__(
        self,
        camera_segmentations="instance",
        camera_heights=128,
        camera_widths=128,
        **kwargs,
    ):
        assert camera_segmentations is not None
        kwargs["camera_segmentations"] = camera_segmentations
        kwargs["camera_heights"] = camera_heights
        kwargs["camera_widths"] = camera_widths
        self.segmentation_id_mapping = {}
        self.instance_to_id = {}
        self.segmentation_robot_id = None
        super().__init__(**kwargs)

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        obs = self.env.reset()
        self.segmentation_id_mapping = {}

        for i, instance_name in enumerate(list(self.env.model.instances_to_ids.keys())):
            if instance_name == "Panda0":
                self.segmentation_robot_id = i

        for i, instance_name in enumerate(list(self.env.model.instances_to_ids.keys())):
            if instance_name not in ["Panda0", "RethinkMount0", "PandaGripper0"]:
                self.segmentation_id_mapping[i] = instance_name

        self.instance_to_id = {
            v: k + 1 for k, v in self.segmentation_id_mapping.items()
        }
        return obs

    def get_segmentation_instances(self, segmentation_image):
        # get all instances' segmentation separately
        seg_img_dict = {}
        segmentation_image[segmentation_image > self.segmentation_robot_id] = (
            self.segmentation_robot_id + 1
        )
        seg_img_dict["robot"] = segmentation_image * (
            segmentation_image == self.segmentation_robot_id + 1
        )

        for seg_id, instance_name in self.segmentation_id_mapping.items():
            seg_img_dict[instance_name] = segmentation_image * (
                segmentation_image == seg_id + 1
            )
        return seg_img_dict

    def get_segmentation_of_interest(self, segmentation_image):
        # get the combined segmentation of obj of interest
        # 1 for obj_of_interest
        # -1.0 for robot
        # 0 for other things
        ret_seg = np.zeros_like(segmentation_image)
        for obj in self.obj_of_interest:
            ret_seg[segmentation_image == self.instance_to_id[obj]] = 1.0
        # ret_seg[segmentation_image == self.segmentation_robot_id+1] = -1.0
        ret_seg[segmentation_image == 0] = -1.0
        return ret_seg

    def segmentation_to_rgb(self, seg_im, random_colors=False):
        """
        Helper function to visualize segmentations as RGB frames.
        NOTE: assumes that geom IDs go up to 255 at most - if not,
        multiple geoms might be assigned to the same color.
        """
        # ensure all values lie within [0, 255]
        seg_im = np.mod(seg_im, 256)

        if random_colors:
            colors = randomize_colors(N=256, bright=True)
            return (255.0 * colors[seg_im]).astype(np.uint8)
        else:
            # deterministic shuffling of values to map each geom ID to a random int in [0, 255]
            rstate = np.random.RandomState(seed=2)
            inds = np.arange(256)
            rstate.shuffle(inds)
            seg_img = (
                np.array(255.0 * cm.rainbow(inds[seg_im], 10))
                .astype(np.uint8)[..., :3]
                .astype(np.uint8)
                .squeeze(-2)
            )
            print(seg_img.shape)
            cv2.imshow("Seg Image", seg_img[::-1])
            cv2.waitKey(1)
            # use @inds to map each geom ID to a color
            return seg_img


class DemoRenderEnv(ControlEnv):
    """
    For visualization and evaluation.
    """

    def __init__(self, **kwargs):
        # This shouldn't be customized
        kwargs["has_renderer"] = False
        kwargs["has_offscreen_renderer"] = True
        kwargs["render_camera"] = "frontview"

        super().__init__(**kwargs)

    def _get_observations(self):
        return self.env._get_observations()
