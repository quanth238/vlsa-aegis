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
        prepared_guidance = None
        if flow_guidance is not None:
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
        start_time = time.monotonic()
        sample_actions = self._sample_actions
        if prepared_guidance is not None:
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
            delta = guided[:, :3].reshape(-1) - prepared_guidance["nominal_xyz"]
            residuals = (
                prepared_guidance["output_rows"] @ delta
                - prepared_guidance["delta_lower"]
            )
            outputs["flow_guidance"] = {
                "schema_version": "crfs_predictive_flow_guidance_result.v1",
                "projection": "dykstra_euclidean_halfspace_projection_after_each_euler_step",
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
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

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
    ) -> dict:  # type: ignore[misc]
        results = self._policy.infer(
            obs,
            noise=noise,
            rng_seed=rng_seed,
            flow_guidance=flow_guidance,
        )

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
