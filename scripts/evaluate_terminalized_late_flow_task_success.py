#!/usr/bin/env python3
"""Evaluate task recovery after one frozen terminalized late-flow chunk."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time
from typing import Any, Mapping, Optional, Sequence

from main.multilink_ellipsoid.terminalized_task_success import (
    ARM_NAMES,
    RESULT_SCHEMA,
    canonical,
    contact_group,
    load_config,
    payload_sha256,
    scientific_view,
    selected_arm_actions,
)
from scripts.evaluate_terminal_branch_sampler_canary import _archived_action
from scripts.replay_distal_three_ellipsoid_multicbf import (
    ARCHIVED_FILE_SHA256,
    ARCHIVED_PAYLOAD_SHA256,
    CASE_ID,
    EXPECTED_ACTION_HORIZON,
    PAPER_CAR_THRESHOLD_M,
    _atomic_write,
    _file_sha256,
    _git_identity,
    _load,
    _require,
)


def _allocation_record(*, require_h100: bool) -> dict[str, Any]:
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id or not job_id.isdigit():
        raise ValueError("task-success evaluation requires Slurm")
    device = None
    if require_h100:
        lines = subprocess.check_output(
            [
                "nvidia-smi", "--query-gpu=name,uuid,driver_version",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.STDOUT,
            text=True,
            timeout=30,
        ).strip().splitlines()
        if len(lines) != 1 or "H100" not in lines[0]:
            raise ValueError("task-success producer requires exactly one H100")
        parts = [item.strip() for item in lines[0].split(",")]
        device = {
            "name": parts[0], "uuid": parts[1],
            "driver_version": parts[2],
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        }
    return {
        "slurm_job_id": job_id,
        "host": socket.gethostname(),
        "execution_mode": "allocated_H100" if require_h100 else "CPU_exact_action_replay",
        "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        "slurm_mem_per_node": os.environ.get("SLURM_MEM_PER_NODE"),
        "device": device,
    }


class RawRobotContactMonitor:
    """Read all active-obstacle robot contacts and CAR at every MuJoCo substep."""

    def __init__(
        self, env: Any, obstacle_reference: Any,
        contact_authority: Mapping[str, Any],
        groups: Mapping[str, Sequence[str]],
    ) -> None:
        import numpy as np

        self.env = env
        self.reference = np.asarray(obstacle_reference, dtype=np.float64).reshape(3)
        self.groups = {str(key): [str(item) for item in value] for key, value in groups.items()}
        self.parent = [int(value) for value in contact_authority["body_parent_ids"]]
        self.geom_body = [int(value) for value in contact_authority["geom_body_ids"]]
        self.geom_names = [str(value) for value in contact_authority["geom_names"]]
        self.active_root = int(contact_authority["active_obstacle_root_body_id"])
        robot_bodies = {int(value) for value in contact_authority["robot_body_ids"]}

        def lineage(body: int) -> set[int]:
            values = set()
            while body >= 0 and body not in values:
                values.add(body)
                if body == 0:
                    break
                body = self.parent[body]
            return values

        self.lineages = [lineage(body) for body in self.geom_body]
        self.is_obstacle = [self.active_root in values for values in self.lineages]
        self.is_robot = [bool(values.intersection(robot_bodies)) for values in self.lineages]
        self.total_substeps = 0
        self.maximum_displacement = 0.0
        self.first_car = None
        self.contact_sample_count = 0
        self.group_sample_count = {group: 0 for group in (*self.groups, "other_robot")}
        self.events: list[dict[str, Any]] = []
        self.first_contact = None

    def _measure(self, *, step: int, substep: int) -> dict[str, Any]:
        import numpy as np

        data = self.env.sim.data
        self.total_substeps += 1
        displacement = float(
            np.sum(
                np.abs(
                    np.asarray(data.xpos[self.active_root], dtype=np.float64)
                    - self.reference
                )
            )
        )
        self.maximum_displacement = max(self.maximum_displacement, displacement)
        if displacement > PAPER_CAR_THRESHOLD_M and self.first_car is None:
            self.first_car = {"step": int(step), "substep": int(substep)}
        sample_events = []
        sample_groups = set()
        for contact_index in range(int(data.ncon)):
            contact = data.contact[contact_index]
            geom = [int(contact.geom1), int(contact.geom2)]
            obstacle_sides = [index for index in (0, 1) if self.is_obstacle[geom[index]]]
            if len(obstacle_sides) != 1:
                continue
            obstacle_side = obstacle_sides[0]
            robot_side = 1 - obstacle_side
            if not self.is_robot[geom[robot_side]]:
                continue
            robot_geom = self.geom_names[geom[robot_side]]
            obstacle_geom = self.geom_names[geom[obstacle_side]]
            group = contact_group(robot_geom, self.groups)
            event = {
                "step": int(step), "substep": int(substep),
                "contact_index": int(contact_index),
                "group": group,
                "robot_geom_name": robot_geom,
                "obstacle_geom_name": obstacle_geom,
                "distance_m": float(contact.dist),
                "position_m": np.asarray(contact.pos, dtype=np.float64).tolist(),
            }
            self.events.append(event)
            sample_events.append(event)
            sample_groups.add(group)
            if self.first_contact is None:
                self.first_contact = dict(event)
        if sample_events:
            self.contact_sample_count += 1
        for group in sample_groups:
            self.group_sample_count[group] += 1
        return {
            "substep": int(substep),
            "active_obstacle_l1_displacement_m": displacement,
            "robot_contact_event_count": len(sample_events),
            "robot_contact_groups": sorted(sample_groups),
        }

    def execute(self, action: Any, *, step: int) -> tuple[Any, float, bool, Any, dict[str, Any]]:
        import numpy as np

        value = np.asarray(action, dtype=np.float64).reshape(7)
        record = {
            "substep_count": 0,
            "maximum_active_obstacle_l1_displacement_m": 0.0,
            "robot_contact_event_count": 0,
            "robot_contact_groups": [],
        }
        groups = set()
        original_step = self.env.sim.step

        def instrumented_step(*args: Any, **kwargs: Any) -> Any:
            output = original_step(*args, **kwargs)
            record["substep_count"] += 1
            sample = self._measure(
                step=step, substep=int(record["substep_count"]),
            )
            record["maximum_active_obstacle_l1_displacement_m"] = max(
                float(record["maximum_active_obstacle_l1_displacement_m"]),
                float(sample["active_obstacle_l1_displacement_m"]),
            )
            record["robot_contact_event_count"] += int(
                sample["robot_contact_event_count"]
            )
            groups.update(sample["robot_contact_groups"])
            return output

        self.env.sim.step = instrumented_step
        try:
            observation, reward, done, info = self.env.step(value.tolist())
        finally:
            self.env.sim.step = original_step
        record["robot_contact_groups"] = sorted(groups)
        return observation, float(reward), bool(done), info, record

    def summary(self) -> dict[str, Any]:
        return {
            "substep_count": int(self.total_substeps),
            "maximum_active_obstacle_l1_displacement_m": float(
                self.maximum_displacement
            ),
            "first_paper_CAR": self.first_car,
            "robot_contact_sample_count": int(self.contact_sample_count),
            "robot_contact_sample_count_by_group": dict(self.group_sample_count),
            "first_robot_contact": self.first_contact,
            "robot_contact_events": self.events,
            "robot_contact_events_sha256": hashlib.sha256(
                canonical(self.events)
            ).hexdigest(),
        }


class LiveTerminalizedSelector:
    """Apply the frozen late-flow terminal selector at the current state."""

    def __init__(
        self, *, repo_root: Path, runtime: Mapping[str, Any], env: Any,
        obstacle_name: str, archived: Mapping[str, Any],
        config: Mapping[str, Any], client: Any,
    ) -> None:
        import numpy as np

        from main.multilink_ellipsoid.compact_safety_coordinate_q import (
            payload_sha256 as compact_payload_sha256,
        )
        from main.multilink_ellipsoid.exact_group_boundary import (
            load_config as load_exact_group_config,
        )
        from main.multilink_ellipsoid.obstacle_proxy_audit import (
            compiled_obstacle_boxes,
            minimum_ellipsoid_quadratics_over_boxes,
        )
        from main.multilink_ellipsoid.palm_primitive_audit import (
            fit_compiled_mesh_geom, world_ellipsoid,
        )
        from main.multilink_ellipsoid.shadow import (
            MultilinkEllipsoidShadow,
            _released_aegis_end_effector_ellipsoid,
            load_shadow_config,
        )
        from main.multilink_ellipsoid.terminal_branching import (
            FROZEN_CANDIDATE_NAMES, load_config as load_method_config,
        )
        from scripts.audit_distal_palm_primitive_case import (
            _canonicalize_perception_ellipsoid_rotation,
        )

        binding = config["registered_method"]
        method_path = repo_root / str(binding["config_path"])
        _require(
            _file_sha256(method_path) == binding["config_file_sha256"],
            "task-success registered method config differs",
        )
        method = load_method_config(method_path)
        source_config_path = repo_root / str(binding["source_geometry_config"])
        _require(
            _file_sha256(source_config_path)
            == binding["source_geometry_config_file_sha256"],
            "task-success source geometry config differs",
        )
        source_config = load_exact_group_config(source_config_path)
        exact_cfg = source_config["exact_group_target"]
        exact_geometry_path = repo_root / str(exact_cfg["robot_geometry_config"])
        _require(
            _file_sha256(exact_geometry_path)
            == exact_cfg["robot_geometry_config_file_sha256"],
            "task-success exact robot geometry differs",
        )
        context_path = Path(binding["source_context_path"])
        _require(
            _file_sha256(context_path) == binding["source_context_file_sha256"],
            "task-success source context file differs",
        )
        context = _load(context_path)
        _require(
            context.get("result_payload_sha256")
            == binding["source_context_payload_sha256"],
            "task-success source context payload differs",
        )
        exact_case = context["exact_case"]
        _require(
            [row["name"] for row in exact_case["candidates"]]
            == list(FROZEN_CANDIDATE_NAMES),
            "task-success frozen residual bank differs",
        )
        compact_binding = method["terminal_risk_model"]
        compact_path = Path(compact_binding["path"])
        _require(
            _file_sha256(compact_path) == compact_binding["file_sha256"],
            "task-success compact critic file differs",
        )
        compact = _load(compact_path)
        _require(
            compact.get("result_payload_sha256")
            == compact_binding["payload_sha256"]
            == compact_payload_sha256(compact, "result_payload_sha256"),
            "task-success compact critic payload differs",
        )
        self.state_payload = compact["compact_shared_7D"]["model"][
            "state_payload"
        ]
        _require(
            compact["compact_shared_7D"]["model"]["model_sha256"]
            == compact_binding["model_sha256"],
            "task-success compact critic model differs",
        )
        perception = archived.get("perception")
        _require(isinstance(perception, dict), "task-success perception differs")
        rotation, _ = _canonicalize_perception_ellipsoid_rotation(
            perception["mvee_rotation"]
        )
        shadow_config = load_shadow_config(exact_geometry_path)
        self.shadow = MultilinkEllipsoidShadow.from_aegis_geometry(
            shadow_config,
            {
                "p2": perception["mvee_center"], "R2": rotation,
                "Q2_diag": perception["mvee_semiaxes"],
                "record": {"label": perception["obstacle_label"]},
            },
        )
        palm_fit = exact_cfg["palm_fit"]
        self.palm_template = fit_compiled_mesh_geom(
            env, str(exact_cfg["palm_geom_name"]),
            relative_padding=float(palm_fit["relative_padding"]),
            tolerance=float(palm_fit["khachiyan_tolerance"]),
            max_iterations=int(palm_fit["khachiyan_max_iterations"]),
        )
        certified = self.shadow._slabbed_links(
            env, include_certificates=True,
        )[:int(exact_cfg["distal_row_count"])]
        _require(
            bool((self.palm_template.enclosure_certificate or {}).get("verified"))
            and all(bool((row.enclosure_certificate or {}).get("verified")) for row in certified),
            "task-success robot primitive certificate differs",
        )
        self.runtime = runtime
        self.env = env
        self.obstacle_name = str(obstacle_name)
        self.client = client
        self.method = method
        self.exact_cfg = exact_cfg
        self.compiled_obstacle_boxes = compiled_obstacle_boxes
        self.minimum_quadratics = minimum_ellipsoid_quadratics_over_boxes
        self.world_ellipsoid = world_ellipsoid
        self.released_ee = _released_aegis_end_effector_ellipsoid
        stored_nominal = np.asarray(
            exact_case["source_nominal_five_action_chunk"], dtype=np.float64,
        )
        self.residuals = np.asarray([
            np.asarray(row["source_executed_actions"], dtype=np.float64)[:, :3]
            - stored_nominal[:, :3]
            for row in exact_case["candidates"]
        ], dtype=np.float64)
        _require(self.residuals.shape == (13, 5, 3),
                 "task-success frozen residual shape differs")

    @staticmethod
    def _row_record(row: Any) -> dict[str, Any]:
        import numpy as np

        return {
            "body_name": str(row.body_name), "geom_name": str(row.geom_name),
            "bound_source": str(row.bound_source),
            "center_m": np.asarray(row.center, dtype=np.float64).tolist(),
            "rotation": np.asarray(row.rotation, dtype=np.float64).tolist(),
            "semiaxes_m": np.asarray(row.semiaxes_m, dtype=np.float64).tolist(),
        }

    def _current_exact_case(self) -> dict[str, Any]:
        import numpy as np

        rows = [self.world_ellipsoid(self.env, self.palm_template)]
        rows.extend(self.shadow._slabbed_links(self.env)[
            :int(self.exact_cfg["distal_row_count"])
        ])
        if self.exact_cfg.get("include_released_aegis_end_effector_proxy") is True:
            rows = [self.released_ee(self.env)] + rows
        boxes = self.compiled_obstacle_boxes(self.env, self.obstacle_name)
        quadratics = self.minimum_quadratics(rows, boxes)
        slacks = np.sqrt(np.min(quadratics, axis=1)) - 1.0
        _require(len(rows) >= 7 and len(boxes) >= 1,
                 "task-success live exact context differs")
        return {
            "initial_exact_robot_rows": [self._row_record(row) for row in rows],
            "initial_compiled_obstacle_boxes": [box.to_record() for box in boxes],
            "source_nominal_five_action_chunk": [[0.0] * 7 for _ in range(5)],
            "exact_group_target": {
                "initial_row_normalized_radial_slack": slacks.tolist(),
            },
        }

    def select(
        self, *, observation: Mapping[str, Any], task_description: str,
        rng_seed: int, query_index: int,
    ) -> tuple[Any, dict[str, Any]]:
        import numpy as np

        from main.evaluate_safelibero_aegis import (
            _policy_observation, array_sha256,
        )
        from main.multilink_ellipsoid.terminal_branching import (
            FROZEN_CANDIDATE_NAMES, build_envelope, score_terminal_bank,
        )

        ordinary_input = _policy_observation(
            self.runtime, observation, task_description=task_description,
            resize_size=224, rng_seed=int(rng_seed),
        )
        ordinary_started = time.perf_counter_ns()
        ordinary_response = self.client.infer(ordinary_input)
        ordinary_wall = (time.perf_counter_ns() - ordinary_started) * 1.0e-9
        ordinary = np.asarray(ordinary_response["actions"], dtype=np.float64)
        _require(ordinary.shape == (10, 7), "task-success ordinary chunk differs")
        posthoc_bank = np.repeat(ordinary[None, :, :], 13, axis=0)
        posthoc_bank[:, :5, :3] = np.clip(
            ordinary[None, :5, :3] + self.residuals, -1.0, 1.0,
        )
        envelope = build_envelope(
            ordinary, posthoc_bank, FROZEN_CANDIDATE_NAMES,
            branch_after_euler_step=int(
                self.method["flow"]["branch_after_euler_step"]
            ),
        )
        branch_input = _policy_observation(
            self.runtime, observation, task_description=task_description,
            resize_size=224, rng_seed=int(rng_seed),
        )
        branch_input["__crfs__"]["terminal_branching"] = envelope
        branch_started = time.perf_counter_ns()
        branch_response = self.client.infer(branch_input)
        branch_wall = (time.perf_counter_ns() - branch_started) * 1.0e-9
        diagnostic = branch_response["terminal_branching"]
        late_bank = np.asarray(
            diagnostic["terminal_output_actions"], dtype=np.float64,
        )
        _require(late_bank.shape == (13, 10, 7),
                 "task-success late terminal bank differs")
        _require(float(np.max(np.abs(late_bank[0] - ordinary))) == 0.0,
                 "task-success late terminal nominal branch differs")
        score = score_terminal_bank(
            exact_case=self._current_exact_case(),
            ordinary_terminal_actions=ordinary,
            terminal_action_bank=late_bank,
            candidate_names=FROZEN_CANDIDATE_NAMES,
            state_payload=self.state_payload,
        )
        selected = np.asarray(
            score["selected_effective_first_five_actions"], dtype=np.float64,
        )
        _require(selected.shape == (5, 7),
                 "task-success online selected chunk differs")
        return selected, {
            "query_index": int(query_index), "rng_seed": int(rng_seed),
            "ordinary_actions_sha256": array_sha256(ordinary),
            "late_terminal_bank_sha256": array_sha256(late_bank),
            "selected_candidate": score["selected_candidate"],
            "selected_candidate_order": score["selected_candidate_order"],
            "selected_predicted_primary": score["selected_predicted_primary"],
            "selected_effective_first_five_sha256": array_sha256(selected),
            "ordinary_wall_seconds": float(ordinary_wall),
            "branch_wall_seconds": float(branch_wall),
            "ordinary_server_timing": ordinary_response.get("server_timing"),
            "branch_server_timing": branch_response.get("server_timing"),
        }


def _run_arm(
    *, repo_root: Path, runtime: Mapping[str, Any], case: Mapping[str, Any],
    archived: Mapping[str, Any], config: Mapping[str, Any],
    pilot_arm: Mapping[str, Any], client: Any,
    frozen_arm: Optional[Mapping[str, Any]], output_root: Path,
) -> dict[str, Any]:
    import numpy as np

    from main.evaluate_safelibero_aegis import (
        TABLE_RENDER_RESOLUTION,
        TABLE_SETTLE_ACTIONS,
        TABLE_VIDEO_FPS,
        _active_obstacle,
        _build_environment,
        _contact_model_authority,
        _goal_progress_definition,
        _goal_progress_snapshot,
        _goal_progress_summary,
        _processed_image,
        _settle,
        max_steps_for_case,
        pairing_record,
        query_seed,
    )
    from main.multilink_ellipsoid.rollout import _dynamic_state_vector

    arm = str(pilot_arm["arm"])
    arm_root = output_root / arm
    arm_root.mkdir(parents=True, exist_ok=False)
    video_partial = arm_root / "episode.partial.mp4"
    video_final = arm_root / "episode.mp4"
    final_jpg = arm_root / "final.jpg"
    env = None
    writer = None
    try:
        env, task, observation, selected_initial_state = _build_environment(
            runtime, case, render_resolution=TABLE_RENDER_RESOLUTION,
        )
        observation = _settle(env, observation, TABLE_SETTLE_ACTIONS)
        obstacle_name, _ = _active_obstacle(env, observation)
        initial_obstacle = np.asarray(
            observation[obstacle_name + "_pos"], dtype=np.float64,
        ).copy()
        pairing = pairing_record(
            case=case,
            selected_initial_state=selected_initial_state,
            settled_observation=observation,
            task_description=str(task.language),
            active_obstacle_name=obstacle_name,
            settled_simulator_state=np.asarray(
                env.sim.get_state().flatten(), dtype=np.float64,
            ),
        )
        for key in (
            "manifest_row_sha256", "initial_state_sha256",
            "initial_observation_sha256", "settled_simulator_state_sha256",
            "settled_active_obstacle_position_sha256",
            "policy_noise_schedule_sha256",
        ):
            _require(pairing[key] == archived["pairing"][key], "pairing differs: %s" % key)
        authority = _contact_model_authority(env, obstacle_name)
        monitor = RawRobotContactMonitor(
            env, initial_obstacle, authority, config["physical_contact_groups"],
        )
        goal_definition, goal_atoms = _goal_progress_definition(env)
        initial_goal = _goal_progress_snapshot(
            env, goal_atoms, step=-1, previous_values=None,
        )
        previous_goal = initial_goal["values"]
        action_records = []
        online_selections = []
        plan: collections.deque[Any] = collections.deque()
        writer = None
        if frozen_arm is None:
            writer = runtime["imageio"].get_writer(
                str(video_partial), fps=TABLE_VIDEO_FPS, codec="libx264",
                macro_block_size=None, pixelformat="yuv420p",
                output_params=["-crf", "18", "-movflags", "+faststart"],
            )
        terminal_frame = _processed_image(observation, "agentview_image")
        if writer is not None:
            writer.append_data(terminal_frame)

        frozen_actions = None
        registered_query_count = None
        if frozen_arm is not None:
            frozen_actions = np.asarray(
                [row["executed_action"] for row in frozen_arm["actions"]],
                dtype=np.float64,
            )
            _require(
                frozen_actions.ndim == 2 and frozen_actions.shape[1] == 7,
                "task-success frozen complete action ledger differs",
            )
            registered_query_count = int(frozen_arm["live_policy_query_count"])
            online_selections = list(frozen_arm["online_selections"])

        selected = np.asarray(pilot_arm["actions"], dtype=np.float64)
        _require(selected.shape == (5, 7), "task-success intervention differs")
        step = 0
        query_index = int(config["state_protocol"]["first_live_policy_query_index"])
        done = False
        warning_state_hash = None
        max_steps = max_steps_for_case(case)
        selector = None
        if frozen_arm is None:
            selector = LiveTerminalizedSelector(
                repo_root=repo_root, runtime=runtime, env=env,
                obstacle_name=obstacle_name, archived=archived,
                config=config, client=client,
            )
        physical_failure = False
        while step < max_steps and not done and not physical_failure:
            if step == int(config["state_protocol"]["archived_prefix_end_exclusive"]):
                warning_state_hash = hashlib.sha256(
                    np.asarray(_dynamic_state_vector(env), dtype=np.float64).tobytes()
                ).hexdigest()
                _require(
                    warning_state_hash
                    == config["registered_selected_action_pilot"]["source_snapshot_sha256"],
                    "task-success warning state differs",
                )

            if frozen_actions is not None:
                if step >= len(frozen_actions):
                    break
                action = frozen_actions[step].copy()
                source_name = str(frozen_arm["actions"][step]["source"])
            elif step < int(config["state_protocol"]["archived_prefix_end_exclusive"]):
                action = _archived_action(archived["actions"], step).copy()
                source_name = "immutable_archived_AEGIS_prefix"
            elif step < 185:
                action = selected[step - 180].copy()
                source_name = "frozen_ADR_0194_selected_chunk"
            else:
                if not plan:
                    seed = query_seed(int(case["policy_noise_seed"]), query_index)
                    selected_online, selection = selector.select(
                        observation=observation,
                        task_description=str(task.language),
                        rng_seed=seed, query_index=query_index,
                    )
                    plan.extend(
                        selected_online[index].copy()
                        for index in range(5)
                    )
                    online_selections.append(selection)
                    query_index += 1
                action = np.asarray(plan.popleft(), dtype=np.float64)
                source_name = "online_late_flow_terminalized_compact_selector_no_QP"

            observation, reward, done, _, internal = monitor.execute(
                action, step=step,
            )
            _require(
                int(internal["substep_count"]) == 25,
                "task-success MuJoCo substep count differs",
            )
            goal = _goal_progress_snapshot(
                env, goal_atoms, step=step, previous_values=previous_goal,
            )
            previous_goal = goal["values"]
            _require(goal["all_satisfied"] is bool(done), "task-success goal/done differs")
            terminal_frame = _processed_image(observation, "agentview_image")
            if writer is not None:
                writer.append_data(terminal_frame)
            action_records.append({
                "step": int(step), "source": source_name,
                "executed_action": np.asarray(action, dtype=np.float64).tolist(),
                "executed_action_sha256": hashlib.sha256(
                    canonical(np.asarray(action, dtype=np.float64).tolist())
                ).hexdigest(),
                "reward": float(reward), "done": bool(done),
                "goal_progress": goal, "internal_physical_measurement": internal,
            })
            step += 1
            physical_failure = bool(
                monitor.first_contact is not None or monitor.first_car is not None
            )

        _require(warning_state_hash is not None, "task-success warning state was not reached")
        if frozen_actions is not None:
            _require(
                step == len(frozen_actions),
                "task-success exact replay action horizon differs",
            )
        intervention = np.asarray(
            [row["executed_action"] for row in action_records[180:185]],
            dtype=np.float64,
        )
        _require(
            np.array_equal(intervention, selected),
            "task-success intervention action bytes differ",
        )
        if writer is not None:
            writer.close()
            writer = None
            video_partial.replace(video_final)
            runtime["imageio"].imwrite(str(final_jpg), terminal_frame)
        goal_summary = _goal_progress_summary(initial_goal, action_records)
        task_success = goal_summary["first_all_satisfied_step"] is not None
        timeout = bool(not task_success and not physical_failure and step >= max_steps)
        terminal_reason = (
            "native_task_success" if task_success and not physical_failure
            else "raw_robot_contact" if monitor.first_contact is not None
            else "paper_CAR" if monitor.first_car is not None
            else "timeout" if timeout else "invalid"
        )
        _require(terminal_reason != "invalid",
                 "task-success episode stopped without terminal")
        physical = monitor.summary()
        contact_pass = int(physical["robot_contact_sample_count"]) == 0
        car_pass = (
            float(physical["maximum_active_obstacle_l1_displacement_m"])
            <= PAPER_CAR_THRESHOLD_M
        )
        complete_actions = [row["executed_action"] for row in action_records]
        record = {
            "arm": arm,
            "source_candidate": str(pilot_arm["source_candidate"]),
            "selection": pilot_arm["selection"],
            "pairing": pairing,
            "warning_state_sha256": warning_state_hash,
            "intervention_actions": selected.tolist(),
            "intervention_actions_sha256": str(pilot_arm["actions_sha256"]),
            "complete_executed_actions_sha256": hashlib.sha256(
                canonical(complete_actions)
            ).hexdigest(),
            "action_count": len(action_records),
            "runtime_live_policy_query_count": (
                0 if frozen_arm is not None else len(online_selections)
            ),
            "live_policy_query_count": (
                len(online_selections)
                if registered_query_count is None else registered_query_count
            ),
            "online_selection_count": len(online_selections),
            "online_selections": online_selections,
            "online_selections_sha256": hashlib.sha256(
                canonical(online_selections)
            ).hexdigest(),
            "actions": action_records,
            "goal_progress": {
                **goal_definition, "initial": initial_goal, "summary": goal_summary,
            },
            "native_task_success": bool(task_success),
            "native_task_success_step": goal_summary["first_all_satisfied_step"],
            "timeout": timeout,
            "terminal_reason": terminal_reason,
            "physical_safety": physical,
            "raw_robot_contact_pass": contact_pass,
            "first_raw_robot_contact": physical["first_robot_contact"],
            "robot_contact_sample_count": int(physical["robot_contact_sample_count"]),
            "robot_contact_sample_count_by_group": dict(
                physical["robot_contact_sample_count_by_group"]
            ),
            "robot_contact_events_sha256": physical["robot_contact_events_sha256"],
            "paper_CAR_pass": car_pass,
            "first_paper_CAR_step": (
                None if physical["first_paper_CAR"] is None
                else int(physical["first_paper_CAR"]["step"])
            ),
            "maximum_active_obstacle_l1_displacement_m": float(
                physical["maximum_active_obstacle_l1_displacement_m"]
            ),
            "collision_free_task_success": bool(
                task_success and contact_pass and car_pass
            ),
            "terminal_dynamic_state_sha256": hashlib.sha256(
                np.asarray(_dynamic_state_vector(env), dtype=np.float64).tobytes()
            ).hexdigest(),
            "goal_progress_summary_sha256": hashlib.sha256(
                canonical(goal_summary)
            ).hexdigest(),
            "video": (
                None if frozen_arm is not None else {
                    "path": str(video_final), "file_sha256": _file_sha256(video_final),
                    "frames_written": len(action_records) + 1,
                    "fps": TABLE_VIDEO_FPS,
                }
            ),
            "final_jpg": (
                None if frozen_arm is not None else {
                    "path": str(final_jpg), "file_sha256": _file_sha256(final_jpg),
                }
            ),
        }
        if frozen_arm is not None:
            expected = {key: frozen_arm[key] for key in (
                "complete_executed_actions_sha256", "action_count",
                "native_task_success", "native_task_success_step", "timeout",
                "raw_robot_contact_pass", "first_raw_robot_contact",
                "robot_contact_sample_count", "robot_contact_sample_count_by_group",
                "robot_contact_events_sha256", "paper_CAR_pass",
                "first_paper_CAR_step", "maximum_active_obstacle_l1_displacement_m",
                "collision_free_task_success", "terminal_dynamic_state_sha256",
                "goal_progress_summary_sha256", "online_selection_count",
                "online_selections_sha256", "terminal_reason",
            )}
            actual = {key: record[key] for key in expected}
            _require(actual == expected, "task-success exact action replay differs")
        return record
    finally:
        if writer is not None:
            writer.close()
        if env is not None:
            env.close()


def evaluate(
    *, repo_root: Path, manifest_path: Path, archived_path: Path,
    config_path: Path, expected_commit: str, replica: str,
    host: str, port: int, output_path: Path,
    frozen_producer_result_path: Optional[Path] = None,
) -> dict[str, Any]:
    from main.evaluate_safelibero_aegis import (
        _runtime_imports, read_jsonl, validate_case_row,
    )

    started = time.perf_counter_ns()
    config = load_config(config_path)
    _require(replica in ("producer", "replay"), "task-success replica differs")
    _require(
        (replica == "replay") is (frozen_producer_result_path is not None),
        "task-success frozen producer mode differs",
    )
    source = _git_identity(repo_root, expected_commit)
    allocation = _allocation_record(require_h100=replica == "producer")
    pilot_binding = config["registered_selected_action_pilot"]
    pilot_path = Path(pilot_binding["path"])
    _require(_file_sha256(pilot_path) == pilot_binding["file_sha256"], "task-success pilot file differs")
    pilot = _load(pilot_path)
    _require(
        pilot.get("result_payload_sha256") == pilot_binding["payload_sha256"]
        and pilot.get("source", {}).get("commit") == pilot_binding["source_commit"],
        "task-success pilot payload differs",
    )
    pilot_arms = selected_arm_actions(pilot)
    _require(_file_sha256(archived_path) == ARCHIVED_FILE_SHA256, "task-success Table-1 file differs")
    archived = _load(archived_path)
    _require(archived.get("result_payload_sha256") == ARCHIVED_PAYLOAD_SHA256, "task-success Table-1 payload differs")
    _require(len(archived.get("actions", [])) == EXPECTED_ACTION_HORIZON, "task-success Table-1 horizon differs")
    rows = [row for row in read_jsonl(manifest_path) if row.get("case_id") == CASE_ID]
    _require(len(rows) == 1, "task-success manifest case differs")
    case = rows[0]
    validate_case_row(case, repo_root)
    _require(int(case["replan_steps"]) == 5, "task-success manifest replan differs")
    runtime = _runtime_imports(include_aegis=False)
    client = None
    server_identity = None
    frozen_by_arm = None
    producer_binding = None
    if replica == "producer":
        client = runtime["websocket_client_policy"].WebsocketClientPolicy(host, int(port))
        from main.evaluate_safelibero_aegis import _server_identity
        server_identity = _server_identity(client)
    else:
        producer = _load(frozen_producer_result_path)
        _require(
            producer.get("schema_version") == RESULT_SCHEMA
            and producer.get("status") == "complete"
            and producer.get("replica") == "producer"
            and producer.get("result_payload_sha256")
            == payload_sha256(producer, "result_payload_sha256"),
            "task-success frozen producer differs",
        )
        frozen_by_arm = {str(row["arm"]): row for row in producer["arms"]}
        _require(set(frozen_by_arm) == set(ARM_NAMES), "task-success frozen producer arms differ")
        producer_binding = {
            "path": str(frozen_producer_result_path),
            "file_sha256": _file_sha256(frozen_producer_result_path),
            "payload_sha256": producer["result_payload_sha256"],
            "source_commit": producer["source"]["commit"],
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    arm_root = output_path.parent / "arms"
    arm_root.mkdir(parents=True, exist_ok=False)
    arm_records = []
    for pilot_arm in pilot_arms:
        arm_records.append(_run_arm(
            repo_root=repo_root, runtime=runtime, case=case,
            archived=archived, config=config,
            pilot_arm=pilot_arm, client=client,
            frozen_arm=(
                None if frozen_by_arm is None
                else frozen_by_arm[str(pilot_arm["arm"])]
            ),
            output_root=arm_root,
        ))
    view = scientific_view(
        case_id=CASE_ID,
        source_snapshot_sha256=pilot_binding["source_snapshot_sha256"],
        arm_records=arm_records,
    )
    if frozen_by_arm is not None:
        producer = _load(frozen_producer_result_path)
        _require(view == producer["scientific_view"], "task-success scientific replay differs")
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "complete",
        "scientific_result": True,
        "replica": replica,
        "source": source,
        "allocation": allocation,
        "config": config,
        "registered_selected_action_pilot": pilot_binding,
        "producer_binding": producer_binding,
        "policy_server": server_identity,
        "scientific_view": view,
        "arms": arm_records,
        "task_success_is_distinct_from_safe_terminal": True,
        "population_claim_authorized": False,
        "formal_safety_claim": False,
        "wall_seconds": (time.perf_counter_ns() - started) * 1.0e-9,
    }
    result["result_payload_sha256"] = payload_sha256(result, "result_payload_sha256")
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--archived", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--replica", choices=("producer", "replay"), required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frozen-producer-result", type=Path)
    args = parser.parse_args(argv)
    result = evaluate(
        repo_root=args.repo_root.resolve(),
        manifest_path=args.manifest.resolve(),
        archived_path=args.archived.resolve(),
        config_path=args.config.resolve(),
        expected_commit=args.expected_commit,
        replica=args.replica, host=args.host, port=args.port,
        output_path=args.output.resolve(),
        frozen_producer_result_path=(
            None if args.frozen_producer_result is None
            else args.frozen_producer_result.resolve()
        ),
    )
    _atomic_write(args.output.resolve(), result)
    print(json.dumps({
        "status": result["status"],
        "replica": result["replica"],
        "task_success_count": result["scientific_view"]["task_success_count"],
        "collision_free_task_success_count": result["scientific_view"]["collision_free_task_success_count"],
        "result_payload_sha256": result["result_payload_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
