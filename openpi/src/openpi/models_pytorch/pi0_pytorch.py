import logging
import math

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F  # noqa: N812

import openpi.models.gemma as _gemma
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
        """
        bsize = observation.state.shape[0]
        actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
        supported_modes = {"none", "residual", "bridge_edit", "latent_resume_edit"}
        if crfs_intervention_mode not in supported_modes:
            raise ValueError(f"Unsupported CRFS intervention mode: {crfs_intervention_mode!r}")
        is_latent_resume = crfs_intervention_mode == "latent_resume_edit"

        if noise is None:
            if is_latent_resume:
                raise ValueError("CRFS latent_resume_edit requires explicit paired noise")
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

                if step_index == crfs_intervention_step:
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

            # Euler step - use new tensor assignment instead of in-place operation
            x_t = x_t + dt * v_t
            time += dt
            step_index += 1
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
