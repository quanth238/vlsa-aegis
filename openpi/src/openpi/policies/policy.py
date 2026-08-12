from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng or jax.random.key(0)

    @override
    def infer(
        self,
        obs: dict,
        *,
        noise: np.ndarray | None = None,
        rng_seed: int | None = None,
        flow_guidance: dict[str, Any] | None = None,
        repulsive_flow_guidance: dict[str, Any] | None = None,
        embodisteer_guidance: dict[str, Any] | None = None,
        embodisteer_joint_denoising: dict[str, Any] | None = None,
    ) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
            if rng_seed is not None:
                sample_rng_or_pytorch_device = jax.random.key(rng_seed)
        else:
            if rng_seed is not None:
                raise ValueError("Per-request rng_seed is only supported by the JAX policy")
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        if sum(
            value is not None
            for value in (
                flow_guidance,
                repulsive_flow_guidance,
                embodisteer_guidance,
                embodisteer_joint_denoising,
            )
        ) > 1:
            raise ValueError("CRFS guidance and joint denoising are mutually exclusive")
        if embodisteer_joint_denoising is not None:
            if self._is_pytorch_model:
                raise ValueError("EmbodiSteer joint denoising requires JAX pi0.5")
            return self._infer_embodisteer_joint_denoising(
                observation,
                normalized_state=np.asarray(inputs["state"][0]),
                rng=sample_rng_or_pytorch_device,
                request=embodisteer_joint_denoising,
            )
        prepared_guidance = None
        guidance_kind = None
        if repulsive_flow_guidance is not None:
            if self._is_pytorch_model:
                raise ValueError("repulsive flow guidance is supported only by JAX pi0.5")
            prepared_guidance = self._prepare_repulsive_flow_guidance(
                repulsive_flow_guidance,
                normalized_state=np.asarray(inputs["state"][0]),
            )
            sample_kwargs["repulsive_direction_model"] = jnp.asarray(
                prepared_guidance["direction_model"]
            )
            sample_kwargs["repulsive_slot_mask"] = jnp.asarray(
                prepared_guidance["slot_mask"]
            )
            sample_kwargs["repulsive_euler_mask"] = jnp.asarray(
                prepared_guidance["euler_mask"]
            )
            sample_kwargs["repulsive_step_size"] = float(
                prepared_guidance["step_size_action"]
            )
            sample_kwargs["repulsive_model_lower"] = jnp.asarray(
                prepared_guidance["model_lower"]
            )
            sample_kwargs["repulsive_model_upper"] = jnp.asarray(
                prepared_guidance["model_upper"]
            )
            guidance_kind = "repulsive_flow"
        elif flow_guidance is not None:
            if self._is_pytorch_model:
                raise ValueError("flow guidance is supported only by the JAX pi0.5 policy")
            prepared_guidance = self._prepare_flow_guidance(
                flow_guidance,
                normalized_state=np.asarray(inputs["state"][0]),
            )
            sample_kwargs["flow_guidance_rows"] = jnp.asarray(
                prepared_guidance["normalized_rows"]
            )
            sample_kwargs["flow_guidance_lower"] = jnp.asarray(
                prepared_guidance["normalized_lower"]
            )
            guidance_kind = "flow"
        elif embodisteer_guidance is not None:
            if self._is_pytorch_model:
                raise ValueError("EmbodiSteer guidance is supported only by JAX pi0.5")
            prepared_guidance = self._prepare_embodisteer_guidance(
                embodisteer_guidance,
                normalized_state=np.asarray(inputs["state"][0]),
            )
            sample_kwargs["embodisteer_rows"] = jnp.asarray(
                prepared_guidance["normalized_rows"]
            )
            sample_kwargs["embodisteer_lower"] = jnp.asarray(
                prepared_guidance["normalized_lower"]
            )
            sample_kwargs["embodisteer_directions"] = jnp.asarray(
                prepared_guidance["normalized_directions"]
            )
            sample_kwargs["schedule_beta"] = float(
                prepared_guidance["schedule_beta"]
            )
            sample_kwargs["schedule_transition"] = float(
                prepared_guidance["schedule_transition"]
            )
            sample_kwargs["schedule_base_strength"] = float(
                prepared_guidance["schedule_base_strength"]
            )
            guidance_kind = "embodisteer"
        start_time = time.monotonic()
        sample_actions = self._sample_actions
        if guidance_kind == "repulsive_flow":
            sample_actions = getattr(
                self, "_sample_actions_with_repulsive_flow_guidance", None
            )
            if sample_actions is None:
                sample_actions = nnx_utils.module_jit(
                    self._model.sample_actions_with_repulsive_flow_guidance
                )
                self._sample_actions_with_repulsive_flow_guidance = sample_actions
        elif guidance_kind == "flow":
            sample_actions = getattr(
                self, "_sample_actions_with_flow_guidance", None
            )
            if sample_actions is None:
                # Keep ordinary policy construction byte-for-byte equivalent
                # to upstream. The separate guided method is wrapped only
                # after a validated opt-in request actually arrives.
                sample_actions = nnx_utils.module_jit(
                    self._model.sample_actions_with_flow_guidance
                )
                self._sample_actions_with_flow_guidance = sample_actions
        elif guidance_kind == "embodisteer":
            sample_actions = getattr(
                self, "_sample_actions_with_embodisteer_guidance", None
            )
            if sample_actions is None:
                sample_actions = nnx_utils.module_jit(
                    self._model.sample_actions_with_embodisteer_guidance
                )
                self._sample_actions_with_embodisteer_guidance = sample_actions
        outputs = {
            "state": inputs["state"],
            "actions": sample_actions(
                sample_rng_or_pytorch_device, observation, **sample_kwargs
            ),
        }
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._output_transform(outputs)
        if prepared_guidance is not None:
            guided = np.asarray(outputs["actions"], dtype=np.float64)
            if guidance_kind == "repulsive_flow":
                delta = guided[:, :3] - prepared_guidance["nominal_output_xyz"]
                outputs["repulsive_flow_guidance"] = {
                    "schema_version": "crfs_fixed_repulsive_flow_guidance_result.v1",
                    "guided_action_slots": prepared_guidance["guided_action_slots"],
                    "guided_euler_steps": prepared_guidance["guided_euler_steps"],
                    "physical_output_direction": prepared_guidance[
                        "physical_output_direction"
                    ].tolist(),
                    "step_size_action": prepared_guidance["step_size_action"],
                    "guided_slot_output_corrections": delta[
                        prepared_guidance["guided_action_slots"]
                    ].tolist(),
                }
                prepared_guidance = None
            else:
                delta = guided[:, :3].reshape(-1) - prepared_guidance["nominal_xyz"]
                residuals = (
                    prepared_guidance["output_rows"] @ delta
                    - prepared_guidance["delta_lower"]
                )
                diagnostic_key = (
                    "flow_guidance"
                    if guidance_kind == "flow"
                    else "embodisteer_guidance"
                )
        if prepared_guidance is not None:
            outputs[diagnostic_key] = {
                "schema_version": (
                    "crfs_predictive_flow_guidance_result.v1"
                    if guidance_kind == "flow"
                    else "crfs_embodisteer_multicbf_guidance_result.v1"
                ),
                "projection": (
                    "dykstra_euclidean_halfspace_projection_after_each_euler_step"
                    if guidance_kind == "flow"
                    else "scheduled_dykstra_task_metric_multi_halfspace_projection_after_each_euler_step"
                ),
                "projection_sweeps_per_euler_step": 64,
                "constraint_count": int(residuals.size),
                "minimum_output_constraint_residual": float(np.min(residuals)),
                "output_constraints_satisfied": bool(
                    np.all(
                        residuals
                        >= -float(prepared_guidance["projection_tolerance"])
                    )
                ),
                "output_xyz_correction_l2": float(np.linalg.norm(delta)),
                "normalized_xyz_scale": prepared_guidance["scale_xyz"].tolist(),
            }
            if guidance_kind == "embodisteer":
                outputs[diagnostic_key]["schedule"] = {
                    "base_strength": prepared_guidance[
                        "schedule_base_strength"
                    ],
                    "beta": prepared_guidance["schedule_beta"],
                    "transition": prepared_guidance["schedule_transition"],
                }
                outputs[diagnostic_key]["task_metric_condition_number"] = (
                    prepared_guidance["task_metric_condition_number"]
                )
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    def _prepare_repulsive_flow_guidance(
        self,
        guidance: dict[str, Any],
        *,
        normalized_state: np.ndarray,
    ) -> dict[str, Any]:
        """Map a physical XYZ push-away direction to normalized action space."""

        horizon = int(guidance["action_horizon"])
        if horizon != int(self._model.action_horizon):
            raise ValueError("repulsive-flow horizon differs from the policy")
        model_dim = int(self._model.action_dim)
        zeros = np.zeros((horizon, model_dim), dtype=np.float32)

        def decode(model_actions: np.ndarray) -> np.ndarray:
            transformed = self._output_transform(
                {
                    "state": np.array(normalized_state, copy=True),
                    "actions": np.array(model_actions, copy=True),
                }
            )
            output = np.asarray(transformed["actions"], dtype=np.float64)
            if output.shape != (horizon, 7) or not np.all(np.isfinite(output)):
                raise ValueError("repulsive-flow output action transform is invalid")
            return output

        offset = decode(zeros)
        scale_xyz = np.empty((horizon, 3), dtype=np.float64)
        for dimension in range(3):
            probe = zeros.copy()
            probe[:, dimension] = 1.0
            effect = decode(probe) - offset
            cross = effect.copy()
            scale_xyz[:, dimension] = effect[:, dimension]
            cross[:, dimension] = 0.0
            if np.max(np.abs(cross)) > 1.0e-8:
                raise ValueError("repulsive-flow output transform couples XYZ")
        if np.any(np.abs(scale_xyz) <= 1.0e-12):
            raise ValueError("repulsive-flow XYZ normalization scale is invalid")
        physical_direction = np.asarray(
            guidance["physical_output_direction"], dtype=np.float64
        )
        direction_model = physical_direction[None, :] / scale_xyz
        slot_mask = np.zeros((horizon, 1), dtype=np.float32)
        slot_mask[guidance["guided_action_slots"], 0] = 1.0
        euler_mask = np.zeros(10, dtype=np.float32)
        euler_mask[guidance["guided_euler_steps"]] = 1.0
        limit = float(guidance["action_limit"])
        model_lower = (-limit - offset[:, :3]) / scale_xyz
        model_upper = (limit - offset[:, :3]) / scale_xyz
        lower = np.minimum(model_lower, model_upper)
        upper = np.maximum(model_lower, model_upper)
        return {
            "direction_model": direction_model.astype(np.float32),
            "slot_mask": slot_mask,
            "euler_mask": euler_mask,
            "model_lower": lower.astype(np.float32),
            "model_upper": upper.astype(np.float32),
            "nominal_output_xyz": np.asarray(
                guidance["nominal_output_actions"], dtype=np.float64
            )[:, :3],
            "guided_action_slots": list(guidance["guided_action_slots"]),
            "guided_euler_steps": list(guidance["guided_euler_steps"]),
            "physical_output_direction": physical_direction,
            "step_size_action": float(guidance["step_size_action"]),
        }

    def _decode_model_actions(
        self,
        model_actions: np.ndarray,
        *,
        normalized_state: np.ndarray,
    ) -> np.ndarray:
        transformed = self._output_transform(
            {
                "state": np.array(normalized_state, copy=True),
                "actions": np.array(model_actions, copy=True),
            }
        )
        output = np.asarray(transformed["actions"], dtype=np.float64)
        expected = (int(self._model.action_horizon), 7)
        if output.shape != expected or not np.all(np.isfinite(output)):
            raise ValueError("decoded pi0.5 action chunk is invalid")
        return output

    def _output_action_affine(
        self,
        *,
        normalized_state: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Recover the affine model-to-LIBERO action transform exactly."""

        horizon = int(self._model.action_horizon)
        model_dim = int(self._model.action_dim)
        zeros = np.zeros((horizon, model_dim), dtype=np.float32)
        offset = self._decode_model_actions(
            zeros, normalized_state=normalized_state
        )
        scale = np.empty((horizon, 7), dtype=np.float64)
        for dimension in range(7):
            probe = zeros.copy()
            probe[:, dimension] = 1.0
            effect = self._decode_model_actions(
                probe, normalized_state=normalized_state
            ) - offset
            scale[:, dimension] = effect[:, dimension]
            cross = effect.copy()
            cross[:, dimension] = 0.0
            if np.max(np.abs(cross)) > 1.0e-8:
                raise ValueError("pi0.5 output transform couples action dimensions")
        if not np.all(np.isfinite(scale)) or np.any(np.abs(scale) <= 1.0e-12):
            raise ValueError("pi0.5 output normalization scale is invalid")
        return offset, scale

    def _infer_embodisteer_joint_denoising(
        self,
        observation: _model.Observation,
        *,
        normalized_state: np.ndarray,
        rng: at.KeyArrayLike,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        """Serve one barrier-free joint-space denoising primitive.

        FK and damped-Jacobian updates remain in the simulator process so they
        use the exact live Panda model.  This endpoint supplies either the
        paired Gaussian initialization or one unchanged pi0.5 Euler update.
        """

        started = time.monotonic()
        horizon = int(self._model.action_horizon)
        model_dim = int(self._model.action_dim)
        mode = str(request["mode"])
        if mode == "initialize":
            model_actions = np.asarray(
                jax.random.normal(rng, (1, horizon, model_dim))[0],
                dtype=np.float32,
            )
            decoded = self._decode_model_actions(
                model_actions, normalized_state=normalized_state
            )
            record = {
                "schema_version": "crfs_embodisteer_joint_denoising_result.v1",
                "mode": mode,
                "model_actions": model_actions.tolist(),
                "physical_actions": decoded.tolist(),
            }
        elif mode == "step":
            template = np.asarray(request["model_actions"], dtype=np.float32)
            physical_pose = np.asarray(
                request["physical_pose_actions"], dtype=np.float64
            )
            if template.shape != (horizon, model_dim) or not np.all(
                np.isfinite(template)
            ):
                raise ValueError("joint-denoising model sample is invalid")
            if physical_pose.shape != (horizon, 6) or not np.all(
                np.isfinite(physical_pose)
            ):
                raise ValueError("joint-denoising pose action is invalid")
            offset, scale = self._output_action_affine(
                normalized_state=normalized_state
            )
            model_input = np.array(template, copy=True)
            model_input[:, :6] = (
                physical_pose - offset[:, :6]
            ) / scale[:, :6]
            flow_step = getattr(self, "_sample_actions_flow_step", None)
            if flow_step is None:
                flow_step = nnx_utils.module_jit(
                    self._model.sample_actions_flow_step
                )
                self._sample_actions_flow_step = flow_step
            model_next = np.asarray(
                flow_step(
                    observation,
                    noisy_actions=jnp.asarray(model_input[None, ...]),
                    time=jnp.asarray(float(request["time"]), dtype=jnp.float32),
                    num_steps=int(request["num_steps"]),
                )[0],
                dtype=np.float32,
            )
            decoded_input = self._decode_model_actions(
                model_input, normalized_state=normalized_state
            )
            decoded = self._decode_model_actions(
                model_next, normalized_state=normalized_state
            )
            record = {
                "schema_version": "crfs_embodisteer_joint_denoising_result.v1",
                "mode": mode,
                "time": float(request["time"]),
                "input_model_actions": model_input.tolist(),
                "model_actions": model_next.tolist(),
                "input_physical_actions": decoded_input.tolist(),
                "physical_actions": decoded.tolist(),
            }
        else:
            raise ValueError("unknown EmbodiSteer joint-denoising mode")
        return {
            "actions": decoded,
            "embodisteer_joint_denoising": record,
            "policy_timing": {"infer_ms": (time.monotonic() - started) * 1000},
        }

    def _prepare_flow_guidance(
        self,
        guidance: dict[str, Any],
        *,
        normalized_state: np.ndarray,
    ) -> dict[str, Any]:
        """Map output-action displacement constraints into model coordinates.

        Physical action displacements are mapped with normalization scale
        only. The offset is used solely to locate the supplied nominal action
        in model coordinates; it is never subtracted from a displacement.
        """

        horizon = int(guidance["action_horizon"])
        if horizon != int(self._model.action_horizon):
            raise ValueError("flow-guidance horizon differs from the policy")
        model_dim = int(self._model.action_dim)
        zeros = np.zeros((horizon, model_dim), dtype=np.float32)

        def decode(model_actions: np.ndarray) -> np.ndarray:
            transformed = self._output_transform(
                {
                    "state": np.array(normalized_state, copy=True),
                    "actions": np.array(model_actions, copy=True),
                }
            )
            output = np.asarray(transformed["actions"], dtype=np.float64)
            if output.shape != (horizon, 7) or not np.all(np.isfinite(output)):
                raise ValueError("flow-guidance output action transform is invalid")
            return output

        offset = decode(zeros)
        scale_xyz = np.empty((horizon, 3), dtype=np.float64)
        for dimension in range(3):
            probe = zeros.copy()
            probe[:, dimension] = 1.0
            effect = decode(probe) - offset
            cross = effect.copy()
            scale_xyz[:, dimension] = effect[:, dimension]
            cross[:, dimension] = 0.0
            if np.max(np.abs(cross)) > 1.0e-8:
                raise ValueError("flow-guidance output transform couples XYZ dimensions")
        if not np.all(np.isfinite(scale_xyz)) or np.any(np.abs(scale_xyz) <= 1.0e-12):
            raise ValueError("flow-guidance XYZ normalization scale is invalid")

        nominal = np.asarray(guidance["nominal_output_actions"], dtype=np.float64)
        rows = np.asarray(guidance["delta_rows"], dtype=np.float64)
        delta_lower = np.asarray(guidance["delta_lower"], dtype=np.float64)
        variable_count = horizon * 3
        if nominal.shape != (horizon, 7):
            raise ValueError("flow-guidance nominal action shape differs")
        if rows.ndim != 2 or rows.shape[1] != variable_count:
            raise ValueError("flow-guidance output row shape differs")
        if delta_lower.shape != (rows.shape[0],):
            raise ValueError("flow-guidance lower-bound shape differs")
        if not all(
            np.all(np.isfinite(value)) for value in (nominal, rows, delta_lower)
        ):
            raise ValueError("flow-guidance arrays must be finite")

        scale_flat = scale_xyz.reshape(-1)
        offset_xyz = offset[:, :3].reshape(-1)
        nominal_xyz = nominal[:, :3].reshape(-1)
        # A_out (u - u_nom) >= l and u = offset + scale * x. Only
        # A_out * scale converts the physical displacement into normalized
        # coordinates.
        normalized_rows = rows * scale_flat[None, :]
        nominal_model_xyz = (nominal_xyz - offset_xyz) / scale_flat
        normalized_lower = delta_lower + normalized_rows @ nominal_model_xyz
        return {
            "output_rows": rows,
            "delta_lower": delta_lower,
            "normalized_rows": normalized_rows.astype(np.float32),
            "normalized_lower": normalized_lower.astype(np.float32),
            "nominal_xyz": nominal_xyz,
            "scale_xyz": scale_xyz,
            "projection_tolerance": float(guidance["projection_tolerance"]),
        }

    def _prepare_embodisteer_guidance(
        self,
        guidance: dict[str, Any],
        *,
        normalized_state: np.ndarray,
    ) -> dict[str, Any]:
        """Map task-metric multi-CBF rows and directions to model space."""

        prepared = self._prepare_flow_guidance(
            guidance, normalized_state=normalized_state
        )
        directions = np.asarray(
            guidance["task_metric_directions"], dtype=np.float64
        )
        rows = np.asarray(guidance["delta_rows"], dtype=np.float64)
        if directions.shape != rows.shape or not np.all(np.isfinite(directions)):
            raise ValueError("EmbodiSteer metric direction shape or values differ")
        scale_flat = prepared["scale_xyz"].reshape(-1)
        normalized_directions = directions / scale_flat[None, :]
        denominators = np.sum(
            prepared["normalized_rows"] * normalized_directions, axis=1
        )
        if np.any(denominators <= 1.0e-12) or not np.all(
            np.isfinite(denominators)
        ):
            raise ValueError("EmbodiSteer metric projection is not positive")
        schedule = guidance["guidance_schedule"]
        prepared.update(
            {
                "normalized_directions": normalized_directions.astype(np.float32),
                "schedule_base_strength": float(schedule["base_strength"]),
                "schedule_beta": float(schedule["beta"]),
                "schedule_transition": float(schedule["transition"]),
                "task_metric_condition_number": float(
                    guidance["task_metric_condition_number"]
                ),
            }
        )
        return prepared

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(
        self,
        obs: dict,
        *,
        noise: np.ndarray | None = None,
        rng_seed: int | None = None,
        flow_guidance: dict[str, Any] | None = None,
        embodisteer_guidance: dict[str, Any] | None = None,
        repulsive_flow_guidance: dict[str, Any] | None = None,
    ) -> dict:  # type: ignore[misc]
        results = self._policy.infer(
            obs,
            noise=noise,
            rng_seed=rng_seed,
            flow_guidance=flow_guidance,
            embodisteer_guidance=embodisteer_guidance,
            repulsive_flow_guidance=repulsive_flow_guidance,
        )

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
