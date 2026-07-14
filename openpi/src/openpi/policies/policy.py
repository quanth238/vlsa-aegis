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
        action_norm_stats: Any | None = None,
        use_quantile_norm: bool = False,
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
            action_norm_stats: Checkpoint action statistics used to convert
                physical CRFS displacement vectors to normalized coordinates.
            use_quantile_norm: Whether action normalization uses q01/q99.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._action_norm_stats = action_norm_stats
        self._use_quantile_norm = use_quantile_norm
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
            self._sample_actions_crfs = getattr(model, "sample_actions_eager", model.sample_actions)
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng or jax.random.key(0)

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        # The websocket client can only send one observation object. Keep CRFS
        # experiment controls in a reserved envelope and remove it before the
        # normal dataset/model transforms see the observation.
        crfs_controls = inputs.pop("__crfs__", None)
        if crfs_controls is not None and not self._is_pytorch_model:
            raise ValueError("CRFS oracle controls are currently supported only by the PyTorch sampler")
        inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        use_crfs_sampler = False
        if crfs_controls is not None:
            if noise is None and crfs_controls.get("noise") is not None:
                noise = np.array(crfs_controls["noise"], copy=True)
            correction = crfs_controls.get("correction")
            if correction is not None:
                correction = np.array(correction, dtype=np.float32, copy=True)
                correction_space = str(crfs_controls.get("correction_space", "model"))
                if correction_space == "physical":
                    correction = self._physical_delta_to_model(correction)
                elif correction_space != "model":
                    raise ValueError(f"Unsupported CRFS correction_space: {correction_space!r}")
                correction = torch.from_numpy(correction).to(self._pytorch_device)
                if correction.ndim == 2:
                    correction = correction[None, ...]
                sample_kwargs["crfs_correction"] = correction
            sample_kwargs["crfs_intervention_step"] = int(crfs_controls.get("intervention_step", 5))
            sample_kwargs["crfs_intervention_mode"] = str(crfs_controls.get("intervention_mode", "none"))
            sample_kwargs["crfs_return_trace"] = bool(crfs_controls.get("return_trace", False))
            # An explicit-noise, no-intervention request is an experiment-only
            # way to exercise the untouched compiled baseline sampler through
            # the WebSocket transport.  Trace dictionaries and interventions
            # still require the eager CRFS path.  With no reserved envelope,
            # this reduces to the original baseline routing exactly.
            use_crfs_sampler = bool(
                sample_kwargs["crfs_return_trace"]
                or sample_kwargs["crfs_intervention_mode"] != "none"
                or correction is not None
            )
            if not use_crfs_sampler:
                # Do not even pass experiment-only keyword arguments to the
                # compiled function.  Apart from the explicit noise supplied
                # below, this is the ordinary baseline sampler call.
                sample_kwargs.pop("crfs_intervention_step")
                sample_kwargs.pop("crfs_intervention_mode")
                sample_kwargs.pop("crfs_return_trace")
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        sample_function = self._sample_actions_crfs if use_crfs_sampler else self._sample_actions
        sampled = sample_function(sample_rng_or_pytorch_device, observation, **sample_kwargs)
        trace = None
        if isinstance(sampled, tuple):
            sampled, trace = sampled
        outputs = {
            "state": inputs["state"],
            "actions": sampled,
        }
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        if trace is not None:
            trace = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), trace)
            # ``predicted_clean`` is an absolute model action, so its physical
            # representation must use the complete existing inverse output
            # transform, including the normalization offset.  This is distinct
            # from displacement corrections, which use scale only in
            # ``_physical_delta_to_model``.
            trace_physical = self._output_transform(
                {
                    "state": np.array(outputs["state"], copy=True),
                    "actions": np.array(trace["predicted_clean"], copy=True),
                }
            )
            trace["predicted_clean_physical"] = np.asarray(trace_physical["actions"])
        outputs = self._output_transform(outputs)
        if trace is not None:
            outputs["crfs_trace"] = trace
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    def _physical_delta_to_model(self, correction: np.ndarray) -> np.ndarray:
        """Scale and pad a displacement without applying a normalization mean."""
        if self._action_norm_stats is None:
            raise ValueError("Checkpoint action normalization statistics are unavailable")
        if correction.ndim != 2:
            raise ValueError(f"Physical CRFS correction must be rank 2, got shape {correction.shape}")
        action_horizon = int(self._model.config.action_horizon)
        action_dim = int(self._model.config.action_dim)
        if correction.shape[0] > action_horizon or correction.shape[1] > action_dim:
            raise ValueError(
                f"Physical CRFS correction shape {correction.shape} exceeds model shape "
                f"({action_horizon}, {action_dim})"
            )
        physical_dim = correction.shape[1]
        if self._use_quantile_norm:
            if self._action_norm_stats.q01 is None or self._action_norm_stats.q99 is None:
                raise ValueError("Quantile normalization requested but q01/q99 action statistics are unavailable")
            q01 = np.asarray(self._action_norm_stats.q01)[..., :physical_dim]
            q99 = np.asarray(self._action_norm_stats.q99)[..., :physical_dim]
            scaled = correction * (2.0 / (q99 - q01 + 1e-6))
        else:
            std = np.asarray(self._action_norm_stats.std)[..., :physical_dim]
            scaled = correction / (std + 1e-6)
        padded = np.zeros((action_horizon, action_dim), dtype=np.float32)
        padded[: correction.shape[0], :physical_dim] = scaled
        return padded

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
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
