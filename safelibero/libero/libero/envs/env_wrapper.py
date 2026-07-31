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
