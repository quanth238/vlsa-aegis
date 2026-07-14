from collections.abc import Mapping
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
            noise_argument_supplied = noise is not None
            if noise is None and crfs_controls.get("noise") is not None:
                noise = np.array(crfs_controls["noise"], copy=True)
            intervention_mode = str(crfs_controls.get("intervention_mode", "none"))
            return_trace = bool(crfs_controls.get("return_trace", False))
            return_normalized_final = bool(crfs_controls.get("return_normalized_final", False))
            correction = crfs_controls.get("correction")
            resume_control_names = ("resume_latent", "resume_time", "latent_edit")
            if intervention_mode == "analytic_trajectory_field":
                analytic_kwargs, noise = self._analytic_field_sample_kwargs(
                    crfs_controls,
                    noise,
                    noise_argument_supplied=noise_argument_supplied,
                )
                sample_kwargs.update(analytic_kwargs)
            elif intervention_mode == "latent_resume_edit":
                if "correction" in crfs_controls or "correction_space" in crfs_controls:
                    raise ValueError("CRFS latent_resume_edit forbids correction and correction_space")
                if crfs_controls.get("return_trace") is not True:
                    raise ValueError("CRFS latent_resume_edit requires return_trace=true")
                if crfs_controls.get("return_normalized_final") is not True:
                    raise ValueError("CRFS latent_resume_edit requires return_normalized_final=true")
                if crfs_controls.get("latent_edit_space") != "model":
                    raise ValueError("CRFS latent_resume_edit requires latent_edit_space='model'")
                missing = [name for name in resume_control_names if crfs_controls.get(name) is None]
                if missing:
                    raise ValueError(f"CRFS latent_resume_edit is missing required controls: {missing}")
                if noise is None:
                    raise ValueError("CRFS latent_resume_edit requires explicit paired noise")

                action_shape = (int(self._model.config.action_horizon), int(self._model.config.action_dim))
                noise_array = np.asarray(noise)
                if noise_array.dtype != np.dtype(np.float32):
                    raise ValueError(f"CRFS paired noise must preserve float32 dtype, got {noise_array.dtype}")
                if noise_array.shape != action_shape:
                    raise ValueError(
                        f"CRFS paired noise must have unbatched shape {action_shape}, got {noise_array.shape}"
                    )
                if not bool(np.isfinite(noise_array).all()):
                    raise ValueError("CRFS paired noise contains a nonfinite value")
                noise = np.array(noise_array, copy=True)

                for control_name, sample_name, expected_shape in (
                    ("resume_latent", "crfs_resume_latent", action_shape),
                    ("resume_time", "crfs_resume_time", ()),
                    ("latent_edit", "crfs_latent_edit", action_shape),
                ):
                    value = np.asarray(crfs_controls[control_name])
                    if value.dtype != np.dtype(np.float32):
                        raise ValueError(
                            f"CRFS {control_name} must preserve captured float32 dtype, got {value.dtype}"
                        )
                    if value.shape != expected_shape:
                        raise ValueError(
                            f"CRFS {control_name} must have unbatched shape {expected_shape}, got {value.shape}"
                        )
                    if not bool(np.isfinite(value).all()):
                        raise ValueError(f"CRFS {control_name} contains a nonfinite value")
                    value = torch.from_numpy(np.array(value, copy=True)).to(self._pytorch_device)
                    if control_name != "resume_time":
                        value = value[None, ...]
                    sample_kwargs[sample_name] = value
            elif any(name in crfs_controls for name in (*resume_control_names, "latent_edit_space")):
                raise ValueError("CRFS resume controls are valid only for latent_resume_edit")

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
            sample_kwargs["crfs_intervention_mode"] = intervention_mode
            sample_kwargs["crfs_return_trace"] = return_trace
            sample_kwargs["crfs_return_normalized_final"] = return_normalized_final
            # An explicit-noise, no-intervention request is an experiment-only
            # way to exercise the untouched compiled baseline sampler through
            # the WebSocket transport.  Trace dictionaries and interventions
            # still require the eager CRFS path.  With no reserved envelope,
            # this reduces to the original baseline routing exactly.
            use_crfs_sampler = bool(
                sample_kwargs["crfs_return_trace"]
                or sample_kwargs["crfs_intervention_mode"] != "none"
                or correction is not None
                or sample_kwargs["crfs_return_normalized_final"]
            )
            if not use_crfs_sampler:
                # Do not even pass experiment-only keyword arguments to the
                # compiled function.  Apart from the explicit noise supplied
                # below, this is the ordinary baseline sampler call.
                sample_kwargs.pop("crfs_intervention_step")
                sample_kwargs.pop("crfs_intervention_mode")
                sample_kwargs.pop("crfs_return_trace")
                sample_kwargs.pop("crfs_return_normalized_final")
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
            if "predicted_clean" in trace:
                trace_physical = self._output_transform(
                    {
                        "state": np.array(outputs["state"], copy=True),
                        "actions": np.array(trace["predicted_clean"], copy=True),
                    }
                )
                trace["predicted_clean_physical"] = np.asarray(trace_physical["actions"])
            if "predicted_clean_post_edit" in trace:
                post_edit_physical = self._output_transform(
                    {
                        "state": np.array(outputs["state"], copy=True),
                        "actions": np.array(trace["predicted_clean_post_edit"], copy=True),
                    }
                )
                trace["predicted_clean_post_edit_physical"] = np.asarray(post_edit_physical["actions"])
            # Bind the sampler's terminal normalized state to the exact same
            # complete output transform used for the action reply.  R03A uses
            # this opt-in trace leaf to prove that the audited Euler trajectory
            # is the trajectory that the simulator actually executes.
            if (
                sample_kwargs.get("crfs_intervention_mode") == "analytic_trajectory_field"
                and "final_normalized" in trace
            ):
                final_physical = self._output_transform(
                    {
                        "state": np.array(outputs["state"], copy=True),
                        "actions": np.array(trace["final_normalized"], copy=True),
                    }
                )
                trace["final_normalized_physical"] = np.asarray(final_physical["actions"])
        outputs = self._output_transform(outputs)
        if trace is not None:
            outputs["crfs_trace"] = trace
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    def _analytic_field_sample_kwargs(
        self,
        controls: Mapping[str, Any],
        noise: np.ndarray | None,
        *,
        noise_argument_supplied: bool,
    ) -> tuple[dict[str, Any], np.ndarray]:
        """Validate and tensorize the reserved analytic-field envelope."""

        required = {
            "intervention_mode",
            "intervention_step",
            "return_trace",
            "model_l2_path_budget",
            "branch_eef_center_m",
            "response_matrix_m_per_action",
            "obstacle_centers_m",
            "obstacle_rotations_world",
            "obstacle_half_sizes_m",
            "eef_radius_m",
        }
        optional = {"noise", "return_normalized_final"}
        unknown = set(controls) - required - optional
        missing = required - set(controls)
        if unknown:
            raise ValueError(f"CRFS analytic_trajectory_field has unsupported controls: {sorted(unknown)}")
        if missing:
            raise ValueError(f"CRFS analytic_trajectory_field is missing required controls: {sorted(missing)}")
        if controls.get("intervention_mode") != "analytic_trajectory_field":
            raise ValueError("analytic field controls require intervention_mode='analytic_trajectory_field'")
        if controls.get("return_trace") is not True:
            raise ValueError("CRFS analytic_trajectory_field requires return_trace=true")
        if "return_normalized_final" in controls and not isinstance(controls["return_normalized_final"], bool):
            raise ValueError("CRFS analytic_trajectory_field return_normalized_final must be boolean")
        intervention_step = controls.get("intervention_step")
        if not isinstance(intervention_step, int) or isinstance(intervention_step, bool):
            raise ValueError("CRFS analytic_trajectory_field intervention_step must be an integer")
        if noise_argument_supplied and "noise" in controls:
            raise ValueError("CRFS analytic_trajectory_field accepts paired noise from exactly one source")
        if noise is None:
            raise ValueError("CRFS analytic_trajectory_field requires explicit paired noise")

        action_shape = (int(self._model.config.action_horizon), int(self._model.config.action_dim))
        noise_array = np.asarray(noise)
        if noise_array.dtype != np.dtype(np.float32):
            raise ValueError(f"CRFS paired noise must preserve float32 dtype, got {noise_array.dtype}")
        if noise_array.shape != action_shape:
            raise ValueError(f"CRFS paired noise must have unbatched shape {action_shape}, got {noise_array.shape}")
        if not bool(np.isfinite(noise_array).all()):
            raise ValueError("CRFS paired noise contains a nonfinite value")

        def finite_array(name: str, shape: tuple[int, ...] | None = None) -> np.ndarray:
            raw = np.asarray(controls[name])
            if not np.issubdtype(raw.dtype, np.number) or np.issubdtype(raw.dtype, np.bool_):
                raise ValueError(f"CRFS {name} must be numeric")
            value = np.asarray(raw, dtype=np.float64)
            if shape is not None and value.shape != shape:
                raise ValueError(f"CRFS {name} must have shape {shape}, got {value.shape}")
            if not bool(np.isfinite(value).all()):
                raise ValueError(f"CRFS {name} contains a nonfinite value")
            return value

        branch_center = finite_array("branch_eef_center_m", (3,))
        response = finite_array("response_matrix_m_per_action", (3, 3))
        centers = finite_array("obstacle_centers_m")
        half_sizes = finite_array("obstacle_half_sizes_m")
        rotations = finite_array("obstacle_rotations_world")
        if centers.ndim != 2 or centers.shape[0] < 1 or centers.shape[1] != 3:
            raise ValueError(f"CRFS obstacle_centers_m must have shape (num_obbs>=1, 3), got {centers.shape}")
        num_obbs = int(centers.shape[0])
        if half_sizes.shape != (num_obbs, 3):
            raise ValueError(
                f"CRFS obstacle_half_sizes_m must have shape ({num_obbs}, 3), got {half_sizes.shape}"
            )
        if rotations.shape == (num_obbs, 9):
            rotations = rotations.reshape(num_obbs, 3, 3)
        elif rotations.shape != (num_obbs, 3, 3):
            raise ValueError(
                "CRFS obstacle_rotations_world must have shape "
                f"({num_obbs}, 3, 3) or ({num_obbs}, 9), got {rotations.shape}"
            )
        if not bool((half_sizes > 0.0).all()):
            raise ValueError("CRFS obstacle_half_sizes_m must be strictly positive")
        identity = np.broadcast_to(np.eye(3, dtype=np.float64), (num_obbs, 3, 3))
        gram = np.swapaxes(rotations, 1, 2) @ rotations
        determinants = np.linalg.det(rotations)
        if not bool(np.allclose(gram, identity, rtol=1.0e-5, atol=1.0e-5)) or not bool(
            np.allclose(determinants, np.ones(num_obbs), rtol=1.0e-5, atol=1.0e-5)
        ):
            raise ValueError("CRFS obstacle_rotations_world must contain proper orthonormal rotations")

        budget = finite_array("model_l2_path_budget", ())
        eef_radius = finite_array("eef_radius_m", ())
        if float(budget) <= 0.0:
            raise ValueError("CRFS model_l2_path_budget must be strictly positive")
        if float(eef_radius) <= 0.0:
            raise ValueError("CRFS eef_radius_m must be strictly positive")
        action_offset_xyz, action_scale_xyz = self._action_xyz_affine()

        def tensor(value: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(np.asarray(value, dtype=np.float32).copy()).to(self._pytorch_device)

        return (
            {
                "crfs_action_offset_xyz": tensor(action_offset_xyz),
                "crfs_action_scale_xyz": tensor(action_scale_xyz),
                "crfs_branch_eef_center_m": tensor(branch_center),
                "crfs_response_matrix_m_per_action": tensor(response),
                "crfs_obstacle_centers_m": tensor(centers),
                "crfs_obstacle_rotations_world": tensor(rotations),
                "crfs_obstacle_half_sizes_m": tensor(half_sizes),
                "crfs_eef_radius_m": tensor(eef_radius),
                "crfs_model_l2_path_budget": tensor(budget),
            },
            np.array(noise_array, copy=True),
        )

    def _action_xyz_affine(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the checkpoint's exact absolute model-to-physical XYZ affine."""

        if self._action_norm_stats is None:
            raise ValueError("Checkpoint action normalization statistics are unavailable")

        def xyz(value: Any, *, name: str) -> np.ndarray:
            array = np.asarray(value, dtype=np.float64)
            if array.ndim < 1 or array.shape[-1] < 3 or int(np.prod(array.shape[:-1])) != 1:
                raise ValueError(f"Checkpoint {name} must provide one action vector with at least three values")
            result = np.reshape(array, (-1, array.shape[-1]))[0, :3]
            if not bool(np.isfinite(result).all()):
                raise ValueError(f"Checkpoint {name} contains a nonfinite XYZ value")
            return result

        if self._use_quantile_norm:
            if self._action_norm_stats.q01 is None or self._action_norm_stats.q99 is None:
                raise ValueError("Quantile normalization requested but q01/q99 action statistics are unavailable")
            q01 = xyz(self._action_norm_stats.q01, name="action q01")
            q99 = xyz(self._action_norm_stats.q99, name="action q99")
            scale = (q99 - q01 + 1.0e-6) / 2.0
            offset = q01 + scale
        else:
            offset = xyz(self._action_norm_stats.mean, name="action mean")
            scale = xyz(self._action_norm_stats.std, name="action std") + 1.0e-6
        if not bool((scale > 0.0).all()):
            raise ValueError("Checkpoint action normalization XYZ scale must be strictly positive")
        return np.asarray(offset, dtype=np.float32), np.asarray(scale, dtype=np.float32)

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
