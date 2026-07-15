import logging
import math
import time as _time

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F  # noqa: N812
from torch.utils.checkpoint import checkpoint as _torch_checkpoint

import openpi.models.gemma as _gemma
from openpi.models_pytorch.crfs_analytic import SAFETY_MARGIN_M as _CRFS_ANALYTIC_MARGIN_M
from openpi.models_pytorch.crfs_analytic import SAMPLES_PER_SEGMENT as _CRFS_ANALYTIC_SAMPLES
from openpi.models_pytorch.crfs_analytic import SOFTPLUS_TAU_M as _CRFS_ANALYTIC_TAU_M
from openpi.models_pytorch.crfs_analytic import analytic_trajectory_field as _analytic_trajectory_field
from openpi.models_pytorch.crfs_analytic import scale_field_to_velocity as _scale_field_to_velocity
from openpi.models_pytorch.crfs_analytic import validate_analytic_controls as _validate_analytic_controls
import openpi.models_pytorch.crfs_inverse_control as _inverse_control
from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
import openpi.models_pytorch.preprocessing_pytorch as _preprocessing


def get_safe_dtype(target_dtype, device_type):
    """Get a safe dtype for the given device type."""
    if device_type == "cpu":
        # CPU doesn't support bfloat16, use float32 instead
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha, beta, bsize, device):
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


class PI0Pytorch(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pi05 = config.pi05

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, True] if self.pi05 else [False, False],
            precision=config.dtype,
        )

        self.action_in_proj = nn.Linear(32, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, 32)

        if self.pi05:
            self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
            self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)
        else:
            self.state_proj = nn.Linear(32, action_expert_config.width)
            self.action_time_mlp_in = nn.Linear(2 * action_expert_config.width, action_expert_config.width)
            self.action_time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)

        torch.set_float32_matmul_precision("high")
        # Preserve an eager experimental path for CRFS trace dictionaries and
        # intervention controls. The ordinary baseline path stays compiled.
        self.sample_actions_eager = self.sample_actions
        self.sample_actions = torch.compile(self.sample_actions, mode="max-autotune")

        # Initialize gradient checkpointing flag
        self.gradient_checkpointing_enabled = False

        msg = "transformers_replace is not installed correctly. Please install it with `uv pip install transformers==4.53.2` and `cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/`."
        try:
            from transformers.models.siglip import check

            if not check.check_whether_transformers_replace_is_installed_correctly():
                raise ValueError(msg)
        except ImportError:
            raise ValueError(msg) from None

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = True
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True

        logging.info("Enabled gradient checkpointing for PI0Pytorch model")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = False
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False

        logging.info("Disabled gradient checkpointing for PI0Pytorch model")

    def is_gradient_checkpointing_enabled(self):
        """Check if gradient checkpointing is enabled."""
        return self.gradient_checkpointing_enabled

    def _apply_checkpoint(self, func, *args, **kwargs):
        """Helper method to apply gradient checkpointing if enabled."""
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        """Helper method to prepare 4D attention masks for transformer."""
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, -2.3819763e38)

    def _preprocess_observation(self, observation, *, train=True):
        """Helper method to preprocess observation."""
        observation = _preprocessing.preprocess_observation_pytorch(observation, train=train)
        return (
            list(observation.images.values()),
            list(observation.image_masks.values()),
            observation.tokenized_prompt,
            observation.tokenized_prompt_mask,
            observation.state,
        )

    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    def sample_time(self, bsize, device):
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for PaliGemma transformer processing.
        """
        embs = []
        pad_masks = []
        att_masks = []

        # Process images
        for img, img_mask in zip(images, img_masks, strict=True):

            def image_embed_func(img):
                return self.paligemma_with_expert.embed_image(img)

            img_emb = self._apply_checkpoint(image_embed_func, img)

            bsize, num_img_embs = img_emb.shape[:2]

            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))

            # Create attention masks so that image tokens attend to each other
            att_masks += [0] * num_img_embs

        # Process language tokens
        def lang_embed_func(lang_tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * math.sqrt(lang_emb_dim)

        lang_emb = self._apply_checkpoint(lang_embed_func, lang_tokens)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        # full attention between image and language inputs
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)

        # Get batch size from the first dimension of the concatenated tensors
        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def embed_suffix(self, state, noisy_actions, timestep):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        if not self.pi05:
            if self.state_proj.weight.dtype == torch.float32:
                state = state.to(torch.float32)

            # Embed state
            def state_proj_func(state):
                return self.state_proj(state)

            state_emb = self._apply_checkpoint(state_proj_func, state)

            embs.append(state_emb[:, None, :])
            bsize = state_emb.shape[0]
            device = state_emb.device

            state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
            pad_masks.append(state_mask)

            # Set attention masks so that image and language inputs do not attend to state or actions
            att_masks += [1]

        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0, device=timestep.device
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        # Fuse timestep + action information using an MLP
        def action_proj_func(noisy_actions):
            return self.action_in_proj(noisy_actions)

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)

        if not self.pi05:
            time_emb = time_emb[:, None, :].expand_as(action_emb)
            action_time_emb = torch.cat([action_emb, time_emb], dim=2)

            # Apply MLP layers
            def mlp_func(action_time_emb):
                x = self.action_time_mlp_in(action_time_emb)
                x = F.silu(x)  # swish == silu
                return self.action_time_mlp_out(x)

            action_time_emb = self._apply_checkpoint(mlp_func, action_time_emb)
            adarms_cond = None
        else:
            # time MLP (for adaRMS)
            def time_mlp_func(time_emb):
                x = self.time_mlp_in(time_emb)
                x = F.silu(x)  # swish == silu
                x = self.time_mlp_out(x)
                return F.silu(x)

            time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
            action_time_emb = action_emb
            adarms_cond = time_emb

        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.config.action_horizon - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond

    def forward(self, observation, actions, noise=None, time=None) -> Tensor:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=True)

        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, time)
        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        # Prepare attention masks
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        # Apply gradient checkpointing if enabled
        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return suffix_out

        suffix_out = self._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        )

        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)

        # Apply gradient checkpointing to final action projection if enabled
        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)

        return F.mse_loss(u_t, v_t, reduction="none")

    @torch.no_grad()
    def sample_actions(
        self,
        device,
        observation,
        noise=None,
        num_steps=10,
        *,
        crfs_correction=None,
        crfs_intervention_step=None,
        crfs_intervention_mode="none",
        crfs_return_trace=False,
        crfs_resume_latent=None,
        crfs_resume_time=None,
        crfs_latent_edit=None,
        crfs_return_normalized_final=False,
        crfs_action_offset_xyz=None,
        crfs_action_scale_xyz=None,
        crfs_branch_eef_center_m=None,
        crfs_response_matrix_m_per_action=None,
        crfs_obstacle_centers_m=None,
        crfs_obstacle_rotations_world=None,
        crfs_obstacle_half_sizes_m=None,
        crfs_eef_radius_m=None,
        crfs_model_l2_path_budget=None,
        crfs_inverse_target=None,
        crfs_model_to_physical_scale=None,
        crfs_inverse_config=None,
        crfs_inverse_budget=None,
        crfs_residual_schedule=None,
        crfs_schedule_budget=None,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        """Sample an action, optionally applying a CRFS oracle intervention.

        The default call path is unchanged. CRFS controls are deliberately
        explicit and operate in the model's normalized action coordinates.
        ``crfs_correction`` must have the same padded shape as ``x_t``; callers
        are responsible for converting a physical action displacement using
        normalization *scale only* before it reaches this method.

        ``latent_resume_edit`` is a separate, eager-only continuation seam. It
        starts from an absolute saved float32 latent and its captured float32
        time, applies one direct model-coordinate edit, recomputes the velocity
        at the edited latent, and executes the remaining ordinary Euler steps.
        The explicit ``noise`` is retained as a required pairing input but is
        not used to initialize this resume path.

        ``analytic_trajectory_field`` is a separate eager-only comparator.  At
        every active step it differentiates the frozen full-trajectory
        sphere/OBB margin energy with respect to a detached approximate-clean
        leaf.  It therefore never backpropagates through the VLA.  Its positive
        energy gradient is added to the reverse-time velocity; because
        ``dt < 0``, the final action moves along negative energy gradient.
        """
        bsize = observation.state.shape[0]
        actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
        supported_modes = {
            "none",
            "residual",
            "bridge_edit",
            "latent_resume_edit",
            "analytic_trajectory_field",
            "inverse_flow_teacher",
            "residual_schedule",
        }
        if crfs_intervention_mode not in supported_modes:
            raise ValueError(f"Unsupported CRFS intervention mode: {crfs_intervention_mode!r}")
        is_latent_resume = crfs_intervention_mode == "latent_resume_edit"
        is_analytic_field = crfs_intervention_mode == "analytic_trajectory_field"
        is_inverse_teacher = crfs_intervention_mode == "inverse_flow_teacher"
        is_residual_schedule = crfs_intervention_mode == "residual_schedule"
        is_flow_schedule = is_inverse_teacher or is_residual_schedule

        noise_argument_supplied = noise is not None
        if noise is None:
            if is_latent_resume:
                raise ValueError("CRFS latent_resume_edit requires explicit paired noise")
            if is_analytic_field:
                raise ValueError("CRFS analytic_trajectory_field requires explicit paired noise")
            noise = self.sample_noise(actions_shape, device)
        if noise.shape != actions_shape:
            raise ValueError(f"noise shape {tuple(noise.shape)} does not match model shape {actions_shape}")

        if crfs_intervention_step is None:
            crfs_intervention_step = num_steps // 2
        if not 0 <= crfs_intervention_step < num_steps:
            raise ValueError(
                f"crfs_intervention_step must be in [0, {num_steps}), got {crfs_intervention_step}"
            )
        if crfs_return_normalized_final and not crfs_return_trace:
            raise ValueError("crfs_return_normalized_final requires crfs_return_trace")
        if crfs_intervention_mode in {"residual", "bridge_edit"} and crfs_correction is None:
            raise ValueError(f"CRFS mode {crfs_intervention_mode!r} requires crfs_correction")
        if crfs_correction is not None and crfs_correction.shape != actions_shape:
            raise ValueError(
                f"crfs_correction shape {tuple(crfs_correction.shape)} does not match model shape {actions_shape}"
            )

        inverse_values = (
            crfs_inverse_target,
            crfs_model_to_physical_scale,
            crfs_inverse_config,
            crfs_inverse_budget,
        )
        schedule_values = (crfs_residual_schedule, crfs_schedule_budget)
        validated_schedule = None
        validated_schedule_increments = None
        validated_schedule_per_step = None
        validated_schedule_path = None
        if is_flow_schedule:
            if not noise_argument_supplied:
                raise ValueError(f"CRFS {crfs_intervention_mode} requires explicit paired noise")
            if num_steps != 10 or crfs_intervention_step != 5:
                raise ValueError(
                    f"CRFS {crfs_intervention_mode} requires num_steps=10 and intervention_step=5"
                )
            if bsize != 1:
                raise ValueError(f"CRFS {crfs_intervention_mode} requires batch size one")
            if crfs_return_trace is not True or crfs_return_normalized_final is not True:
                raise ValueError(
                    f"CRFS {crfs_intervention_mode} requires crfs_return_trace=True and "
                    "crfs_return_normalized_final=True"
                )
            if crfs_correction is not None:
                raise ValueError(f"CRFS {crfs_intervention_mode} forbids crfs_correction")
            if not isinstance(noise, Tensor) or noise.dtype != torch.float32:
                dtype = None if not isinstance(noise, Tensor) else noise.dtype
                raise ValueError(f"CRFS paired noise must be a float32 Tensor, got {dtype}")
            expected_device = torch.device(device)
            if noise.device.type != expected_device.type or (
                expected_device.index is not None and noise.device.index != expected_device.index
            ):
                raise ValueError(f"CRFS paired noise device {noise.device} does not match sampler device {device}")
            if not bool(torch.isfinite(noise).all().item()):
                raise ValueError("CRFS paired noise contains a nonfinite value")

        if is_inverse_teacher:
            if any(value is None for value in inverse_values):
                raise ValueError(
                    "CRFS inverse_flow_teacher requires crfs_inverse_target, "
                    "crfs_model_to_physical_scale, crfs_inverse_config, and crfs_inverse_budget"
                )
            if any(value is not None for value in schedule_values):
                raise ValueError("CRFS inverse_flow_teacher forbids an explicit residual schedule")
            if not isinstance(crfs_inverse_config, _inverse_control.InverseControlConfig):
                raise ValueError("crfs_inverse_config must be an InverseControlConfig")
            crfs_inverse_config.validate()
            registered_inverse_config = _inverse_control.InverseControlConfig(
                num_steps=10,
                intervention_step=5,
                dt=-0.1,
                max_iterations=128,
                learning_rate=0.02,
                adam_beta1=0.9,
                adam_beta2=0.999,
                adam_epsilon=1.0e-8,
                xyz_max_abs_tolerance=0.010,
                xyz_rms_tolerance=0.005,
                full_max_abs_tolerance=0.050,
                full_rms_tolerance=0.015,
                constraint_slack_ulps=8,
                stop_on_first_feasible=False,
            )
            if crfs_inverse_config != registered_inverse_config:
                raise ValueError("crfs_inverse_config does not match the frozen IFT-00A solver config")
            if crfs_inverse_config.num_steps != num_steps:
                raise ValueError("crfs_inverse_config num_steps must match the sampler")
            for name, value in (
                ("crfs_inverse_target", crfs_inverse_target),
                ("crfs_model_to_physical_scale", crfs_model_to_physical_scale),
            ):
                if not isinstance(value, Tensor):
                    raise ValueError(f"{name} must be a Tensor")
                if value.shape != actions_shape:
                    raise ValueError(f"{name} shape {tuple(value.shape)} does not match {actions_shape}")
                if value.dtype != torch.float32 or value.device != noise.device:
                    raise ValueError(f"{name} must match paired noise float32 dtype and device")
                if not bool(torch.isfinite(value).all().item()):
                    raise ValueError(f"{name} contains a nonfinite value")
            if not bool((crfs_model_to_physical_scale > 0).all().item()):
                raise ValueError("crfs_model_to_physical_scale must be strictly positive")
            if not isinstance(crfs_inverse_budget, Tensor) or tuple(crfs_inverse_budget.shape) != ():
                raise ValueError("crfs_inverse_budget must be a scalar Tensor")
            if crfs_inverse_budget.dtype != torch.float32 or crfs_inverse_budget.device != noise.device:
                raise ValueError("crfs_inverse_budget must match paired noise float32 dtype and device")
            if not bool(torch.isfinite(crfs_inverse_budget).item()) or float(crfs_inverse_budget.item()) < 0.0:
                raise ValueError("crfs_inverse_budget must be finite and nonnegative")
        elif any(value is not None for value in inverse_values):
            raise ValueError("CRFS inverse target, scale, and config are valid only for inverse_flow_teacher")

        if is_residual_schedule:
            if any(value is None for value in schedule_values):
                raise ValueError(
                    "CRFS residual_schedule requires crfs_residual_schedule and crfs_schedule_budget"
                )
            if not isinstance(crfs_residual_schedule, Tensor):
                raise ValueError("crfs_residual_schedule must be a Tensor")
            expected_schedule_shape = (bsize, num_steps, *actions_shape[1:])
            if crfs_residual_schedule.shape != expected_schedule_shape:
                raise ValueError(
                    f"crfs_residual_schedule shape {tuple(crfs_residual_schedule.shape)} "
                    f"does not match {expected_schedule_shape}"
                )
            if crfs_residual_schedule.dtype != torch.float32 or crfs_residual_schedule.device != noise.device:
                raise ValueError("crfs_residual_schedule must match paired noise float32 dtype and device")
            if not isinstance(crfs_schedule_budget, Tensor) or tuple(crfs_schedule_budget.shape) != ():
                raise ValueError("crfs_schedule_budget must be a scalar Tensor")
            if crfs_schedule_budget.dtype != torch.float32 or crfs_schedule_budget.device != noise.device:
                raise ValueError("crfs_schedule_budget must match paired noise float32 dtype and device")
            if not bool(torch.isfinite(crfs_schedule_budget).item()):
                raise ValueError("crfs_schedule_budget must be finite")
            validated_schedule = crfs_residual_schedule.permute(1, 0, 2, 3).contiguous()
            control_mask = _inverse_control.first_five_xyz_mask_like(noise)
            (
                validated_schedule_increments,
                validated_schedule_per_step,
                validated_schedule_path,
            ) = _inverse_control.validate_schedule_constraints(
                validated_schedule,
                control_mask,
                crfs_schedule_budget,
            )
        elif any(value is not None for value in schedule_values):
            raise ValueError("CRFS residual schedule and budget are valid only for residual_schedule")
        resume_values = (crfs_resume_latent, crfs_resume_time, crfs_latent_edit)
        if is_latent_resume:
            if crfs_return_trace is not True or crfs_return_normalized_final is not True:
                raise ValueError(
                    "CRFS latent_resume_edit requires crfs_return_trace=True and "
                    "crfs_return_normalized_final=True"
                )
            if crfs_correction is not None:
                raise ValueError("CRFS latent_resume_edit uses crfs_latent_edit, not crfs_correction")
            if any(value is None for value in resume_values):
                raise ValueError(
                    "CRFS latent_resume_edit requires crfs_resume_latent, crfs_resume_time, and crfs_latent_edit"
                )
            if noise.dtype != torch.float32:
                raise ValueError(f"CRFS paired noise must be float32, got {noise.dtype}")
            if crfs_resume_latent.dtype != torch.float32:
                raise ValueError(f"crfs_resume_latent must be float32, got {crfs_resume_latent.dtype}")
            if crfs_latent_edit.dtype != torch.float32:
                raise ValueError(f"crfs_latent_edit must be float32, got {crfs_latent_edit.dtype}")
            if crfs_resume_time.dtype != torch.float32:
                raise ValueError(f"crfs_resume_time must be float32, got {crfs_resume_time.dtype}")
            expected_device = torch.device(device)
            if noise.device.type != expected_device.type or (
                expected_device.index is not None and noise.device.index != expected_device.index
            ):
                raise ValueError(f"CRFS paired noise device {noise.device} does not match sampler device {device}")
            for name, value in (
                ("crfs_resume_latent", crfs_resume_latent),
                ("crfs_resume_time", crfs_resume_time),
                ("crfs_latent_edit", crfs_latent_edit),
            ):
                if value.device != noise.device:
                    raise ValueError(f"{name} device {value.device} does not match paired noise device {noise.device}")
            if crfs_resume_latent.shape != actions_shape:
                raise ValueError(
                    f"crfs_resume_latent shape {tuple(crfs_resume_latent.shape)} does not match model shape "
                    f"{actions_shape}"
                )
            if crfs_latent_edit.shape != actions_shape:
                raise ValueError(
                    f"crfs_latent_edit shape {tuple(crfs_latent_edit.shape)} does not match model shape "
                    f"{actions_shape}"
                )
            if tuple(crfs_resume_time.shape) != ():
                raise ValueError(f"crfs_resume_time must have scalar shape (), got {tuple(crfs_resume_time.shape)}")
            for name, value in (
                ("paired noise", noise),
                ("crfs_resume_latent", crfs_resume_latent),
                ("crfs_resume_time", crfs_resume_time),
                ("crfs_latent_edit", crfs_latent_edit),
            ):
                if not bool(torch.isfinite(value).all().item()):
                    raise ValueError(f"CRFS {name} contains a nonfinite value")
            resume_time_value = float(crfs_resume_time.item())
            if not 0.0 < resume_time_value <= 1.0:
                raise ValueError(f"crfs_resume_time must be active in (0, 1], got {resume_time_value}")
        elif any(value is not None for value in resume_values):
            raise ValueError("CRFS resume_latent, resume_time, and latent_edit are valid only for latent_resume_edit")

        analytic_values = (
            crfs_action_offset_xyz,
            crfs_action_scale_xyz,
            crfs_branch_eef_center_m,
            crfs_response_matrix_m_per_action,
            crfs_obstacle_centers_m,
            crfs_obstacle_rotations_world,
            crfs_obstacle_half_sizes_m,
            crfs_eef_radius_m,
            crfs_model_l2_path_budget,
        )
        if is_analytic_field:
            if crfs_return_trace is not True:
                raise ValueError("CRFS analytic_trajectory_field requires crfs_return_trace=True")
            if crfs_correction is not None:
                raise ValueError("CRFS analytic_trajectory_field forbids crfs_correction")
            if bsize != 1:
                raise ValueError("CRFS analytic_trajectory_field currently requires batch size one")
            if any(value is None for value in analytic_values):
                raise ValueError("CRFS analytic_trajectory_field requires the complete affine, geometry, and budget")
            expected_device = torch.device(device)
            if noise.dtype != torch.float32:
                raise ValueError(f"CRFS paired noise must be float32, got {noise.dtype}")
            if noise.device.type != expected_device.type or (
                expected_device.index is not None and noise.device.index != expected_device.index
            ):
                raise ValueError(f"CRFS paired noise device {noise.device} does not match sampler device {device}")
            if not bool(torch.isfinite(noise).all().item()):
                raise ValueError("CRFS paired noise contains a nonfinite value")
            _validate_analytic_controls(
                action_horizon=int(self.config.action_horizon),
                action_dim=int(self.config.action_dim),
                device=noise.device,
                action_offset_xyz=crfs_action_offset_xyz,
                action_scale_xyz=crfs_action_scale_xyz,
                branch_eef_center_m=crfs_branch_eef_center_m,
                response_matrix_m_per_action=crfs_response_matrix_m_per_action,
                obstacle_centers_m=crfs_obstacle_centers_m,
                obstacle_rotations_world=crfs_obstacle_rotations_world,
                obstacle_half_sizes_m=crfs_obstacle_half_sizes_m,
                eef_radius_m=crfs_eef_radius_m,
                model_l2_path_budget=crfs_model_l2_path_budget,
            )
        elif any(value is not None for value in analytic_values):
            raise ValueError(
                "CRFS analytic affine, geometry, and budget controls are valid only for analytic_trajectory_field"
            )

        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=False)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Compute image and language key value cache
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        inverse_result = None
        applied_flow_schedule = validated_schedule
        flow_schedule_applied = is_residual_schedule
        parameter_grad_flags_restored = True
        parameter_grads_none_before = True
        parameter_grads_none_after = True
        teacher_cuda_memory = None
        if is_inverse_teacher:

            def inverse_velocity_fn(x_value: Tensor, scalar_time: Tensor, _step: int) -> Tensor:
                expanded_inverse_time = scalar_time.expand(bsize)

                def differentiable_denoise(x_input: Tensor) -> Tensor:
                    return self.denoise_step(
                        state,
                        prefix_pad_masks,
                        past_key_values,
                        x_input,
                        expanded_inverse_time,
                    )

                if torch.is_grad_enabled() and x_value.requires_grad:
                    return _torch_checkpoint(
                        differentiable_denoise,
                        x_value,
                        use_reentrant=False,
                        preserve_rng_state=False,
                    )
                return differentiable_denoise(x_value)

            parameter_grad_flags = tuple(
                (parameter, parameter.requires_grad) for parameter in self.parameters()
            )
            parameter_grads_none_before = all(
                parameter.grad is None for parameter, _requires_grad in parameter_grad_flags
            )
            if noise.is_cuda:
                teacher_cuda_memory = dict(
                    allocated_before=torch.cuda.memory_allocated(noise.device),
                    reserved_before=torch.cuda.memory_reserved(noise.device),
                )
            try:
                for parameter, _requires_grad in parameter_grad_flags:
                    parameter.requires_grad_(requires_grad=False)
                control_mask = _inverse_control.first_five_xyz_mask_like(noise)
                target_mask = _inverse_control.first_five_channels_mask_like(noise, channels=7)
                inverse_result = _inverse_control.solve_inverse_control(
                    noise,
                    crfs_inverse_target,
                    inverse_velocity_fn,
                    control_mask=control_mask,
                    target_mask=target_mask,
                    model_to_physical_scale=crfs_model_to_physical_scale,
                    control_budget=crfs_inverse_budget,
                    config=crfs_inverse_config,
                )
            finally:
                for parameter, requires_grad in parameter_grad_flags:
                    parameter.requires_grad_(requires_grad=requires_grad)
                parameter_grad_flags_restored = all(
                    parameter.requires_grad == requires_grad
                    for parameter, requires_grad in parameter_grad_flags
                )
                parameter_grads_none_after = all(
                    parameter.grad is None for parameter, _requires_grad in parameter_grad_flags
                )
                if noise.is_cuda and teacher_cuda_memory is not None:
                    # These peaks are process-lifetime counters.  Deliberately
                    # do not reset them because the policy server is shared by
                    # all calls within its allocation.
                    teacher_cuda_memory.update(
                        allocated_after_solve=torch.cuda.memory_allocated(noise.device),
                        reserved_after_solve=torch.cuda.memory_reserved(noise.device),
                        process_peak_allocated_after_solve=torch.cuda.max_memory_allocated(noise.device),
                        process_peak_reserved_after_solve=torch.cuda.max_memory_reserved(noise.device),
                    )

            if inverse_result.schedule is not None:
                if inverse_result.budget is None:
                    raise RuntimeError("inverse-flow solver returned a schedule without its budget")
                (
                    validated_schedule_increments,
                    validated_schedule_per_step,
                    validated_schedule_path,
                ) = _inverse_control.validate_schedule_constraints(
                    inverse_result.schedule,
                    control_mask,
                    inverse_result.budget,
                    config=crfs_inverse_config,
                )
                teacher_candidate_valid = bool(
                    inverse_result.converged
                    and inverse_result.schedule_valid
                    and inverse_result.baseline_valid
                    and inverse_result.target_pairing_checked
                    and inverse_result.target_pairing_exact
                    and parameter_grad_flags_restored
                    and parameter_grads_none_before
                    and parameter_grads_none_after
                )
                if teacher_candidate_valid:
                    applied_flow_schedule = inverse_result.schedule.detach().clone()
                    flow_schedule_applied = True
                else:
                    # Keep the failed candidate only in explicitly invalid
                    # diagnostic leaves.  Executed actions remain the exact
                    # paired frozen sampler output.
                    applied_flow_schedule = torch.zeros(
                        (num_steps, *actions_shape),
                        dtype=noise.dtype,
                        device=noise.device,
                    )
                    flow_schedule_applied = False
            else:
                # A failed solve has no schedule to replay.  The ordinary loop
                # still returns the paired frozen recurrence, explicitly marked
                # invalid below; no solver schedule or metrics are fabricated.
                applied_flow_schedule = torch.zeros(
                    (num_steps, *actions_shape),
                    dtype=noise.dtype,
                    device=noise.device,
                )
                flow_schedule_applied = False

        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        if is_latent_resume:
            x_t = crfs_resume_latent.detach().clone()
            time = crfs_resume_time.detach().clone().reshape(())
            step_index = crfs_intervention_step
        else:
            x_t = noise
            time = torch.tensor(1.0, dtype=torch.float32, device=device)
            step_index = 0
        crfs_trace = None
        crfs_residual_horizon = 1.0 - crfs_intervention_step / num_steps
        analytic_active_horizon = (num_steps - crfs_intervention_step) / num_steps
        analytic_integrated_field_l2 = (
            torch.zeros((bsize,), dtype=torch.float32, device=device) if is_analytic_field else None
        )
        analytic_step_records = [] if is_analytic_field else None
        flow_step_records = [] if is_flow_schedule else None
        latent_resume_pending = is_latent_resume
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            if latent_resume_pending:
                x_t_pre_edit = x_t
                # Preserve every unedited coordinate byte-for-byte.  A plain
                # `-0.0 + +0.0` can flip the sign bit, which would make the
                # registered zero-edit resume arm a false no-op even though
                # its numeric values compare equal.
                x_t = torch.where(
                    crfs_latent_edit == 0,
                    x_t,
                    x_t + crfs_latent_edit,
                )
                v_base = self.denoise_step(
                    state,
                    prefix_pad_masks,
                    past_key_values,
                    x_t,
                    expanded_time,
                )
                crfs_trace = {
                    "step_index": torch.full((bsize,), step_index, dtype=torch.int64, device=device),
                    "time": expanded_time.detach().clone(),
                    "x_t_pre_edit": x_t_pre_edit.detach().clone(),
                    "latent_edit": crfs_latent_edit.detach().clone(),
                    "x_t_post_edit": x_t.detach().clone(),
                    "v_post_edit": v_base.detach().clone(),
                    "predicted_clean_post_edit": (x_t - time * v_base).detach().clone(),
                }
                latent_resume_pending = False
            else:
                x_before_intervention = x_t
                v_base = self.denoise_step(
                    state,
                    prefix_pad_masks,
                    past_key_values,
                    x_t,
                    expanded_time,
                )

                if step_index == crfs_intervention_step and not is_analytic_field and not is_flow_schedule:
                    crfs_trace = {
                        "step_index": torch.full((bsize,), step_index, dtype=torch.int64, device=device),
                        "time": expanded_time.detach().clone(),
                        "x_t": x_before_intervention.detach().clone(),
                        "v_base": v_base.detach().clone(),
                        "predicted_clean": (x_before_intervention - time * v_base).detach().clone(),
                    }
                    if crfs_intervention_mode == "bridge_edit":
                        x_t = x_t + (1.0 - time) * crfs_correction
                        v_base = self.denoise_step(
                            state,
                            prefix_pad_masks,
                            past_key_values,
                            x_t,
                            expanded_time,
                        )

            v_t = v_base
            if crfs_intervention_mode == "residual" and step_index >= crfs_intervention_step:
                if crfs_residual_horizon <= 0:
                    raise RuntimeError("CRFS residual intervention has no remaining integration time")
                v_t = v_t - crfs_correction / crfs_residual_horizon
            flow_control = None
            if is_flow_schedule:
                if applied_flow_schedule is None:
                    raise RuntimeError("CRFS flow schedule was not initialized")
                flow_control = applied_flow_schedule[step_index]
                # The zero branch preserves the exact ordinary velocity bytes;
                # nonzero controls use the registered additive velocity field.
                v_t = torch.where(flow_control == 0, v_t, v_t + flow_control)

            if is_analytic_field:
                predicted_clean = (x_t - time * v_base).detach()
                zero_scalar = torch.zeros((bsize,), dtype=torch.float32, device=device)
                zero_bool = torch.zeros((bsize,), dtype=torch.bool, device=device)
                zero_action = torch.zeros_like(x_t)
                field_record = {
                    "physical_predicted_clean_xyz": torch.zeros(
                        (bsize, 5, 3), dtype=torch.float32, device=device
                    ),
                    "predicted_eef_centers_m": torch.zeros((bsize, 5, 3), dtype=torch.float32, device=device),
                    "hard_min_clearance_m": zero_scalar,
                    "energy": zero_scalar,
                    "energy_gradient": zero_action,
                    "normalized_energy_gradient": zero_action,
                    "gradient_l2": zero_scalar,
                    "margin_satisfied": zero_bool,
                    "gradient_finite": zero_bool,
                    "gradient_valid": zero_bool,
                    "applied": zero_bool,
                }
                analytic_gradient_ms = 0.0
                active = step_index >= crfs_intervention_step
                if active:
                    if x_t.is_cuda:
                        torch.cuda.synchronize(x_t.device)
                    analytic_start = _time.perf_counter()
                    field_record = _analytic_trajectory_field(
                        predicted_clean,
                        action_offset_xyz=crfs_action_offset_xyz,
                        action_scale_xyz=crfs_action_scale_xyz,
                        branch_eef_center_m=crfs_branch_eef_center_m,
                        response_matrix_m_per_action=crfs_response_matrix_m_per_action,
                        obstacle_centers_m=crfs_obstacle_centers_m,
                        obstacle_rotations_world=crfs_obstacle_rotations_world,
                        obstacle_half_sizes_m=crfs_obstacle_half_sizes_m,
                        eef_radius_m=crfs_eef_radius_m,
                    )
                    if x_t.is_cuda:
                        torch.cuda.synchronize(x_t.device)
                    analytic_gradient_ms = (_time.perf_counter() - analytic_start) * 1000.0
                    if not bool(field_record["gradient_finite"].all().item()):
                        raise RuntimeError("CRFS analytic trajectory field produced a nonfinite energy or gradient")
                guidance_velocity = _scale_field_to_velocity(
                    field_record["normalized_energy_gradient"],
                    field_record["applied"],
                    model_l2_path_budget=crfs_model_l2_path_budget,
                    active_horizon=analytic_active_horizon,
                )
                guidance_velocity_l2 = torch.linalg.vector_norm(
                    guidance_velocity[:, :5, :3].reshape(bsize, -1), dim=1
                )
                path_increment_l2 = torch.abs(dt) * guidance_velocity_l2
                if analytic_integrated_field_l2 is None:
                    raise RuntimeError("CRFS analytic trajectory field lost its path-budget accumulator")
                analytic_integrated_field_l2 = analytic_integrated_field_l2 + path_increment_l2
                # Positive energy gradient is added to velocity.  The ordinary
                # reverse-time Euler step below has dt < 0, so action motion is
                # along negative energy gradient, away from the obstacle.
                v_t = v_t + guidance_velocity
                analytic_step_records.append(
                    {
                        "step_index": torch.full((bsize,), step_index, dtype=torch.int64, device=device),
                        "time": expanded_time.detach().clone(),
                        "active": torch.full((bsize,), active, dtype=torch.bool, device=device),
                        "field_evaluated": torch.full((bsize,), active, dtype=torch.bool, device=device),
                        "x_t_steps": x_t.detach().clone(),
                        "v_base_steps": v_base.detach().clone(),
                        "predicted_clean_steps": predicted_clean,
                        "physical_predicted_clean_xyz_steps": field_record[
                            "physical_predicted_clean_xyz"
                        ],
                        "predicted_eef_centers_m_steps": field_record["predicted_eef_centers_m"],
                        "hard_min_clearance_m": field_record["hard_min_clearance_m"],
                        "energy": field_record["energy"],
                        "energy_gradient_steps": field_record["energy_gradient"],
                        "normalized_energy_gradient_steps": field_record["normalized_energy_gradient"],
                        "gradient_l2": field_record["gradient_l2"],
                        "margin_satisfied": field_record["margin_satisfied"],
                        "gradient_finite": field_record["gradient_finite"],
                        "gradient_valid": field_record["gradient_valid"],
                        "applied": field_record["applied"],
                        "guidance_velocity_steps": guidance_velocity.detach().clone(),
                        "guidance_velocity_l2": guidance_velocity_l2.detach().clone(),
                        "path_increment_l2": path_increment_l2.detach().clone(),
                        "cumulative_integrated_field_l2": analytic_integrated_field_l2.detach().clone(),
                        "analytic_gradient_ms": torch.full(
                            (bsize,), analytic_gradient_ms, dtype=torch.float32, device=device
                        ),
                    }
                )

            # Keep the ordinary/default Euler statement byte-for-byte.  The
            # opt-in flow modes retain the pre-state only for recurrence audit.
            x_before_step = x_t
            x_t = x_t + dt * v_t
            if flow_step_records is not None:
                if flow_control is None:
                    raise RuntimeError("CRFS flow trace lost its applied control")
                flow_step_records.append(
                    dict(
                        step_index=torch.full((bsize,), step_index, dtype=torch.int64, device=device),
                        time=expanded_time.detach().clone(),
                        active=torch.full(
                            (bsize,), step_index >= crfs_intervention_step, dtype=torch.bool, device=device
                        ),
                        x_t=x_before_step.detach().clone(),
                        v_base=v_base.detach().clone(),
                        control_velocity=flow_control.detach().clone(),
                        total_velocity=v_t.detach().clone(),
                        increment=(dt * flow_control).detach().clone(),
                        x_next=x_t.detach().clone(),
                    )
                )
            time += dt
            step_index += 1
        if flow_step_records is not None:
            if len(flow_step_records) != num_steps:
                raise RuntimeError("CRFS flow trace did not cover every Euler step")
            if is_inverse_teacher and noise.is_cuda and teacher_cuda_memory is not None:
                teacher_cuda_memory.update(
                    allocated_after_replay=torch.cuda.memory_allocated(noise.device),
                    reserved_after_replay=torch.cuda.memory_reserved(noise.device),
                    process_peak_allocated_after_replay=torch.cuda.max_memory_allocated(noise.device),
                    process_peak_reserved_after_replay=torch.cuda.max_memory_reserved(noise.device),
                )
            flow_trace = dict(
                control_source=torch.full(
                    (bsize,), 0 if is_inverse_teacher else 1, dtype=torch.int64, device=device
                ),
                control_valid=torch.full(
                    (bsize,),
                    bool(
                        flow_schedule_applied
                        and (
                            not is_inverse_teacher
                            or (
                                inverse_result is not None
                                and inverse_result.converged
                                and inverse_result.schedule_valid
                            )
                        )
                    ),
                    dtype=torch.bool,
                    device=device,
                ),
                schedule_applied=torch.full(
                    (bsize,), flow_schedule_applied, dtype=torch.bool, device=device
                ),
                step_index_steps=torch.stack(
                    [record["step_index"] for record in flow_step_records], dim=1
                ),
                time_steps=torch.stack([record["time"] for record in flow_step_records], dim=1),
                active_steps=torch.stack([record["active"] for record in flow_step_records], dim=1),
                x_t_steps=torch.stack([record["x_t"] for record in flow_step_records], dim=1),
                v_base_steps=torch.stack([record["v_base"] for record in flow_step_records], dim=1),
                control_velocity_steps=torch.stack(
                    [record["control_velocity"] for record in flow_step_records], dim=1
                ),
                total_velocity_steps=torch.stack(
                    [record["total_velocity"] for record in flow_step_records], dim=1
                ),
                control_increment_steps=torch.stack(
                    [record["increment"] for record in flow_step_records], dim=1
                ),
                x_next_steps=torch.stack([record["x_next"] for record in flow_step_records], dim=1),
                initial_noise=noise.detach().clone(),
                canonical_replay_final=x_t.detach().clone(),
                dt=dt.expand(bsize).detach().clone(),
                intervention_step=torch.full(
                    (bsize,), crfs_intervention_step, dtype=torch.int64, device=device
                ),
                num_steps=torch.full((bsize,), num_steps, dtype=torch.int64, device=device),
                parameter_requires_grad_restored=torch.full(
                    (bsize,), parameter_grad_flags_restored, dtype=torch.bool, device=device
                ),
                parameter_grads_none_before=torch.full(
                    (bsize,), parameter_grads_none_before, dtype=torch.bool, device=device
                ),
                parameter_grads_none_after=torch.full(
                    (bsize,), parameter_grads_none_after, dtype=torch.bool, device=device
                ),
                parameter_grad_check_performed=torch.full(
                    (bsize,), is_inverse_teacher, dtype=torch.bool, device=device
                ),
            )
            if is_residual_schedule:
                if (
                    validated_schedule_increments is None
                    or validated_schedule_per_step is None
                    or validated_schedule_path is None
                ):
                    raise RuntimeError("validated residual schedule metadata is unavailable")
                flow_trace.update(
                    schedule_budget=crfs_schedule_budget.expand(bsize).detach().clone(),
                    schedule_path_length=validated_schedule_path.expand(bsize).detach().clone(),
                    schedule_per_step_increment_l2=validated_schedule_per_step.unsqueeze(0).detach().clone(),
                    schedule_energy=torch.sum(torch.square(validated_schedule_increments))
                    .expand(bsize)
                    .detach()
                    .clone(),
                    solver_status=torch.full((bsize,), -1, dtype=torch.int64, device=device),
                    solver_iterations=torch.zeros((bsize,), dtype=torch.int64, device=device),
                    solver_fields_available=torch.zeros((bsize,), dtype=torch.bool, device=device),
                    solver_nonfinite=torch.zeros((bsize,), dtype=torch.bool, device=device),
                )
            else:
                if inverse_result is None:
                    raise RuntimeError("inverse-flow teacher result is unavailable")
                flow_trace.update(
                    solver_status=torch.full(
                        (bsize,), int(inverse_result.status), dtype=torch.int64, device=device
                    ),
                    solver_converged=torch.full(
                        (bsize,), inverse_result.converged, dtype=torch.bool, device=device
                    ),
                    solver_iterations=torch.full(
                        (bsize,), inverse_result.iterations, dtype=torch.int64, device=device
                    ),
                    solver_fields_available=torch.full(
                        (bsize,), inverse_result.schedule is not None, dtype=torch.bool, device=device
                    ),
                    solver_nonfinite=torch.full(
                        (bsize,), inverse_result.nonfinite_detected, dtype=torch.bool, device=device
                    ),
                    solver_baseline_valid=torch.full(
                        (bsize,), inverse_result.baseline_valid, dtype=torch.bool, device=device
                    ),
                    solver_schedule_valid=torch.full(
                        (bsize,), inverse_result.schedule_valid, dtype=torch.bool, device=device
                    ),
                    target_pairing_checked=torch.full(
                        (bsize,), inverse_result.target_pairing_checked, dtype=torch.bool, device=device
                    ),
                    target_pairing_exact=torch.full(
                        (bsize,), inverse_result.target_pairing_exact, dtype=torch.bool, device=device
                    ),
                    inverse_target=crfs_inverse_target.detach().clone(),
                    model_to_physical_scale=crfs_model_to_physical_scale.detach().clone(),
                    source_control_budget=crfs_inverse_budget.expand(bsize).detach().clone(),
                    solver_config_max_iterations=torch.full(
                        (bsize,), crfs_inverse_config.max_iterations, dtype=torch.int64, device=device
                    ),
                    solver_config_learning_rate=torch.full(
                        (bsize,), crfs_inverse_config.learning_rate, dtype=torch.float32, device=device
                    ),
                    solver_config_adam_beta1=torch.full(
                        (bsize,), crfs_inverse_config.adam_beta1, dtype=torch.float32, device=device
                    ),
                    solver_config_adam_beta2=torch.full(
                        (bsize,), crfs_inverse_config.adam_beta2, dtype=torch.float32, device=device
                    ),
                    solver_config_adam_epsilon=torch.full(
                        (bsize,), crfs_inverse_config.adam_epsilon, dtype=torch.float32, device=device
                    ),
                    solver_config_xyz_max_abs_tolerance=torch.full(
                        (bsize,), crfs_inverse_config.xyz_max_abs_tolerance, dtype=torch.float32, device=device
                    ),
                    solver_config_xyz_rms_tolerance=torch.full(
                        (bsize,), crfs_inverse_config.xyz_rms_tolerance, dtype=torch.float32, device=device
                    ),
                    solver_config_full_max_abs_tolerance=torch.full(
                        (bsize,), crfs_inverse_config.full_max_abs_tolerance, dtype=torch.float32, device=device
                    ),
                    solver_config_full_rms_tolerance=torch.full(
                        (bsize,), crfs_inverse_config.full_rms_tolerance, dtype=torch.float32, device=device
                    ),
                    solver_config_constraint_slack_ulps=torch.full(
                        (bsize,), crfs_inverse_config.constraint_slack_ulps, dtype=torch.int64, device=device
                    ),
                    solver_config_stop_on_first_feasible=torch.full(
                        (bsize,), crfs_inverse_config.stop_on_first_feasible, dtype=torch.bool, device=device
                    ),
                    cuda_memory_available=torch.full(
                        (bsize,), teacher_cuda_memory is not None, dtype=torch.bool, device=device
                    ),
                )
                if teacher_cuda_memory is not None:
                    for trace_name, memory_name in (
                        ("cuda_memory_allocated_before_bytes", "allocated_before"),
                        ("cuda_memory_reserved_before_bytes", "reserved_before"),
                        ("cuda_memory_allocated_after_solve_bytes", "allocated_after_solve"),
                        ("cuda_memory_reserved_after_solve_bytes", "reserved_after_solve"),
                        ("cuda_process_peak_allocated_after_solve_bytes", "process_peak_allocated_after_solve"),
                        ("cuda_process_peak_reserved_after_solve_bytes", "process_peak_reserved_after_solve"),
                        ("cuda_memory_allocated_after_bytes", "allocated_after_replay"),
                        ("cuda_memory_reserved_after_bytes", "reserved_after_replay"),
                        ("cuda_process_peak_allocated_bytes", "process_peak_allocated_after_replay"),
                        ("cuda_process_peak_reserved_bytes", "process_peak_reserved_after_replay"),
                    ):
                        flow_trace[trace_name] = torch.full(
                            (bsize,), teacher_cuda_memory[memory_name], dtype=torch.int64, device=device
                        )
                optional_solver_scalars = (
                    ("schedule_budget", inverse_result.budget),
                    ("schedule_path_length", inverse_result.path_length),
                    ("schedule_energy", inverse_result.energy),
                    ("solver_objective", inverse_result.objective),
                    ("target_pairing_max_abs", inverse_result.target_pairing_max_abs),
                    ("realized_target_delta_norm", inverse_result.realized_target_delta_norm),
                )
                for trace_name, value in optional_solver_scalars:
                    if value is not None:
                        flow_trace[trace_name] = value.expand(bsize).detach().clone()
                optional_solver_actions = (
                    ("solver_schedule", inverse_result.schedule),
                    ("solver_baseline_final", inverse_result.baseline_final),
                    (
                        "solver_internal_replay_final",
                        None if inverse_result.rollout is None else inverse_result.rollout.final,
                    ),
                    ("solver_model_error", inverse_result.model_error),
                    ("solver_fidelity_error", inverse_result.fidelity_error),
                )
                for trace_name, value in optional_solver_actions:
                    if value is None:
                        continue
                    if trace_name == "solver_schedule":
                        value = value.permute(1, 0, 2, 3).contiguous()
                    flow_trace[trace_name] = value.detach().clone()
                if inverse_result.per_step_norms is not None:
                    flow_trace["schedule_per_step_increment_l2"] = (
                        inverse_result.per_step_norms.unsqueeze(0).detach().clone()
                    )
                if inverse_result.metrics is not None:
                    flow_trace.update(
                        fidelity_xyz_max_abs=inverse_result.metrics.xyz_max_abs.expand(bsize).detach().clone(),
                        fidelity_xyz_rms=inverse_result.metrics.xyz_rms.expand(bsize).detach().clone(),
                        fidelity_full_max_abs=inverse_result.metrics.full_max_abs.expand(bsize).detach().clone(),
                        fidelity_full_rms=inverse_result.metrics.full_rms.expand(bsize).detach().clone(),
                    )
            crfs_trace = flow_trace
        if is_analytic_field:
            if analytic_step_records is None or len(analytic_step_records) != num_steps:
                raise RuntimeError("CRFS analytic trajectory trace did not cover every Euler step")
            analytic_trace = {
                key: torch.stack([record[key] for record in analytic_step_records], dim=1)
                for key in analytic_step_records[0]
            }
            analytic_trace.update(
                {
                    "intervention_step": torch.full(
                        (bsize,), crfs_intervention_step, dtype=torch.int64, device=device
                    ),
                    "dt": dt.expand(bsize).detach().clone(),
                    "active_horizon": torch.full(
                        (bsize,), analytic_active_horizon, dtype=torch.float32, device=device
                    ),
                    "model_l2_path_budget": crfs_model_l2_path_budget.expand(bsize).detach().clone(),
                    "velocity_gain": (crfs_model_l2_path_budget / analytic_active_horizon)
                    .expand(bsize)
                    .detach()
                    .clone(),
                    "safety_margin_m": torch.full(
                        (bsize,), _CRFS_ANALYTIC_MARGIN_M, dtype=torch.float32, device=device
                    ),
                    "softplus_tau_m": torch.full(
                        (bsize,), _CRFS_ANALYTIC_TAU_M, dtype=torch.float32, device=device
                    ),
                    "samples_per_segment": torch.full(
                        (bsize,), _CRFS_ANALYTIC_SAMPLES, dtype=torch.int64, device=device
                    ),
                    "integrated_field_l2": analytic_integrated_field_l2.detach().clone(),
                }
            )
            crfs_trace = analytic_trace
        if crfs_return_trace:
            if crfs_trace is None:
                raise RuntimeError("CRFS trace step was not reached")
            if crfs_return_normalized_final:
                crfs_trace["final_normalized"] = x_t.detach().clone()
            return x_t, crfs_trace
        return x_t

    def denoise_step(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        # Prepare attention masks
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)
