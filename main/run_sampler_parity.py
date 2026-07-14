#!/usr/bin/env python3
"""Allocation-backed public-JAX / converted-PyTorch sampler parity gate.

The ordinary OpenPI samplers remain untouched.  Each backend worker first calls
its ordinary sampler, then runs a harness-local instrumented copy of the same
ten Euler updates.  The instrumented final value must reproduce the ordinary
sampler before any cross-framework trace comparison is accepted.

The parent process obtains one real, frozen SafeLIBERO branch observation in a
short-lived LIBERO worker and passes the exact serialized observation and
explicit noise identity to short-lived JAX and PyTorch workers.  Separate
workers avoid retaining both models in one MIG allocation.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import tempfile
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from crfs_harness.artifacts import (  # noqa: E402
    atomic_write_json,
    content_hash,
    file_sha256,
    load_json,
    validate_jsonl_unique,
)


SCHEMA_VERSION = "1.0"
WORKER_SCHEMA_VERSION = "1.0"
BASELINE_COMMIT = "57b1aef306f212aea3574b0a3b64aa1a3d8f5e4b"
R01_SUMMARY_RELATIVE = Path("evidence/r01/r01-summary.json")
R01_SUMMARY_SHA256 = "715d9326f02df531e0c6d7aa7b94629569b320800b4283c36fff41335d9b8bc5"
REGISTERED_NORM_STATS_SHA256 = "b3a44bb2810436fb62917decaea58bd4d9110255df527dea21e8fd40c960bd84"
FIXTURE_ID = "r01-frozen-manifest-branch"
NUM_STEPS = 10
ACTION_HORIZON = 10
ACTION_DIM = 32
LIBERO_ACTION_DIM = 7
REGISTERED_LIMITS = {
    "step_x_max": 0.10,
    "step_x_rms": 0.025,
    "step_v_max": 0.20,
    "step_v_rms": 0.050,
    "final_model_max": 0.10,
    "final_model_rms": 0.025,
    "physical_xyz5_max": 0.010,
    "physical_xyz5_rms": 0.005,
    "physical_action7_max": 0.050,
    "physical_action7_rms": 0.015,
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require_allocation() -> None:
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("sampler parity must execute inside a Slurm allocation")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible or visible == "NoDevFiles":
        raise RuntimeError("sampler parity requires an allocation-visible GPU")


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _framed_update(digest: Any, value: bytes) -> None:
    digest.update(len(value).to_bytes(8, byteorder="big", signed=False))
    digest.update(value)


def _tree_leaves(value: Any, path: tuple[str, ...] = ()):
    if isinstance(value, Mapping):
        for key in sorted(value):
            yield from _tree_leaves(value[key], (*path, str(key)))
        return
    yield path, value


def fingerprint_tree(value: Any) -> tuple[str, list[dict[str, Any]]]:
    """Hash a nested observation without JSON float or array ambiguity."""
    import numpy as np

    digest = hashlib.sha256()
    manifest: list[dict[str, Any]] = []
    for path, leaf in _tree_leaves(value):
        path_text = "/".join(path)
        _framed_update(digest, path_text.encode("utf-8"))
        if isinstance(leaf, str):
            data = leaf.encode("utf-8")
            _framed_update(digest, b"str")
            _framed_update(digest, data)
            manifest.append(
                {
                    "path": path_text,
                    "kind": "str",
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "value": leaf,
                }
            )
            continue
        array = np.ascontiguousarray(np.asarray(leaf))
        descriptor = json.dumps(
            {"dtype": array.dtype.str, "shape": list(array.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        data = array.tobytes(order="C")
        _framed_update(digest, b"array")
        _framed_update(digest, descriptor)
        _framed_update(digest, data)
        manifest.append(
            {
                "path": path_text,
                "kind": "array",
                "dtype": array.dtype.str,
                "shape": list(array.shape),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    return digest.hexdigest(), manifest


def _atomic_write_npz(path: Path, values: Mapping[str, Any]) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez(handle, **values)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _load_raw_observation(path: Path) -> dict[str, Any]:
    import numpy as np

    with np.load(path, allow_pickle=False) as archive:
        required = {
            "observation_image",
            "observation_wrist_image",
            "observation_state",
            "prompt",
        }
        if set(archive.files) != required:
            raise ValueError(f"raw observation archive keys differ: {sorted(archive.files)}")
        prompt_array = np.asarray(archive["prompt"])
        if prompt_array.shape != ():
            raise ValueError("serialized prompt must be scalar")
        return {
            "observation/image": np.array(archive["observation_image"], copy=True),
            "observation/wrist_image": np.array(
                archive["observation_wrist_image"], copy=True
            ),
            "observation/state": np.array(archive["observation_state"], copy=True),
            "prompt": str(prompt_array.item()),
        }


def _atomic_write_array_tree(path: Path, value: Mapping[str, Any]) -> None:
    """Serialize a dict-only array tree without pickle or framework objects."""
    import numpy as np

    leaves = list(_tree_leaves(value))
    if not leaves or any(isinstance(leaf, str) for _, leaf in leaves):
        raise ValueError("transformed tree must contain only array leaves")
    arrays = {
        f"leaf_{index:04d}": np.asarray(leaf)
        for index, (_, leaf) in enumerate(leaves)
    }
    arrays["tree_paths"] = np.asarray(
        ["/".join(path_parts) for path_parts, _ in leaves], dtype=np.str_
    )
    _atomic_write_npz(path, arrays)


def _load_array_tree(path: Path) -> dict[str, Any]:
    import numpy as np

    with np.load(path, allow_pickle=False) as archive:
        if "tree_paths" not in archive.files:
            raise ValueError("array tree archive has no tree_paths")
        paths = [str(item) for item in np.asarray(archive["tree_paths"]).tolist()]
        expected = {"tree_paths"} | {
            f"leaf_{index:04d}" for index in range(len(paths))
        }
        if set(archive.files) != expected:
            raise ValueError("array tree archive leaf set is inconsistent")
        root: dict[str, Any] = {}
        for index, path_text in enumerate(paths):
            parts = path_text.split("/")
            if not path_text or any(not part for part in parts):
                raise ValueError(f"invalid serialized tree path: {path_text!r}")
            cursor = root
            for part in parts[:-1]:
                child = cursor.setdefault(part, {})
                if not isinstance(child, dict):
                    raise ValueError(f"tree path collision at {path_text!r}")
                cursor = child
            if parts[-1] in cursor:
                raise ValueError(f"duplicate serialized tree path: {path_text!r}")
            cursor[parts[-1]] = np.array(
                archive[f"leaf_{index:04d}"], copy=True
            )
    return root


def _atomic_write_noise(path: Path, noise: Any) -> None:
    _atomic_write_npz(path, {"noise": noise})


def _load_noise(path: Path):
    import numpy as np

    with np.load(path, allow_pickle=False) as archive:
        if archive.files != ["noise"]:
            raise ValueError(f"noise archive keys differ: {archive.files}")
        noise = np.array(archive["noise"], copy=True)
    if noise.shape != (1, ACTION_HORIZON, ACTION_DIM) or noise.dtype != np.float32:
        raise ValueError(
            f"explicit noise must be float32 [1,10,32], got {noise.dtype} {noise.shape}"
        )
    return noise


def _directory_sha256(path: Path) -> dict[str, Any]:
    """Content hash every regular file and symlink in a checkpoint directory."""
    if not path.is_dir():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    files = 0
    total_bytes = 0
    for item in sorted(path.rglob("*"), key=lambda candidate: candidate.as_posix()):
        relative = item.relative_to(path).as_posix().encode("utf-8")
        if item.is_symlink():
            target = os.readlink(item).encode("utf-8")
            _framed_update(digest, b"symlink")
            _framed_update(digest, relative)
            _framed_update(digest, target)
            files += 1
        elif item.is_file():
            _framed_update(digest, b"file")
            _framed_update(digest, relative)
            size = item.stat().st_size
            _framed_update(digest, str(size).encode("ascii"))
            with item.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            files += 1
            total_bytes += size
    if files == 0:
        raise ValueError(f"checkpoint directory is empty: {path}")
    return {"sha256": digest.hexdigest(), "files": files, "bytes": total_bytes}


def _git_state(root: Path) -> dict[str, Any]:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--short"], cwd=root, text=True
    ).splitlines()
    tracked_diff = subprocess.check_output(
        ["git", "diff", "--binary", "HEAD"], cwd=root
    )
    return {
        "commit": commit,
        "dirty": bool(status),
        "status_short": status,
        "tracked_diff_sha256": hashlib.sha256(tracked_diff).hexdigest(),
    }


def _make_noise(seed: int):
    import numpy as np

    return np.random.default_rng(seed).normal(
        size=(1, ACTION_HORIZON, ACTION_DIM)
    ).astype(np.float32)


def _load_transform_context(config_name: str, jax_checkpoint: Path):
    from openpi import transforms
    from openpi.training import checkpoints
    from openpi.training import config as training_config

    train_config = training_config.get_config(config_name)
    if int(train_config.model.action_horizon) != ACTION_HORIZON:
        raise ValueError("training config action horizon is not ten")
    if int(train_config.model.action_dim) != ACTION_DIM:
        raise ValueError("training config action dimension is not 32")
    data_config = train_config.data.create(
        train_config.assets_dirs, train_config.model
    )
    if data_config.asset_id is None:
        raise ValueError("training data config has no normalization asset id")
    norm_stats = checkpoints.load_norm_stats(
        jax_checkpoint / "assets", data_config.asset_id
    )
    if norm_stats is None or "actions" not in norm_stats:
        raise ValueError("checkpoint action normalization statistics are unavailable")
    transform = transforms.compose(
        [
            transforms.InjectDefaultPrompt(None),
            *data_config.data_transforms.inputs,
            transforms.Normalize(
                norm_stats, use_quantiles=data_config.use_quantile_norm
            ),
            *data_config.model_transforms.inputs,
        ]
    )
    output_transform = transforms.compose(
        [
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(
                norm_stats, use_quantiles=data_config.use_quantile_norm
            ),
            *data_config.data_transforms.outputs,
        ]
    )
    return train_config, data_config, norm_stats, transform, output_transform


def _unnormalize_actions(output_transform: Any, transformed: Mapping[str, Any], actions: Any):
    import numpy as np

    value = output_transform(
        {
            "state": np.array(transformed["state"], copy=True),
            "actions": np.asarray(actions, dtype=np.float32),
        }
    )
    physical = np.asarray(value["actions"], dtype=np.float32)
    expected = (ACTION_HORIZON, LIBERO_ACTION_DIM)
    if physical.shape != expected:
        raise ValueError(f"unnormalized LIBERO action shape {physical.shape} != {expected}")
    if not np.all(np.isfinite(physical)):
        raise ValueError("unnormalized actions contain non-finite values")
    return physical


def _jax_instrumented_sampler(model: Any, observation: Any, noise: Any):
    """JIT a traced copy of public Pi0.sample_actions without changing it."""
    import einops
    from flax import nnx
    import jax
    import jax.numpy as jnp
    from openpi.models import model as model_module
    from openpi.models.pi0 import make_attn_mask

    graphdef, state = nnx.split(model)

    def traced(state_value: Any, observation_value: Any, noise_value: Any):
        module = nnx.merge(graphdef, state_value)
        processed = model_module.preprocess_observation(
            None, observation_value, train=False
        )
        dt = -1.0 / NUM_STEPS
        batch_size = processed.state.shape[0]
        prefix_tokens, prefix_mask, prefix_ar_mask = module.embed_prefix(processed)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = module.PaliGemma.llm(
            [prefix_tokens, None],
            mask=prefix_attn_mask,
            positions=prefix_positions,
        )

        def step(carry: Any, unused: Any):
            del unused
            x_t, current_time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = (
                module.embed_suffix(
                    processed,
                    x_t,
                    jnp.broadcast_to(current_time, batch_size),
                )
            )
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_to_suffix_mask = einops.repeat(
                prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1]
            )
            full_attn_mask = jnp.concatenate(
                [prefix_to_suffix_mask, suffix_attn_mask], axis=-1
            )
            positions = (
                jnp.sum(prefix_mask, axis=-1)[:, None]
                + jnp.cumsum(suffix_mask, axis=-1)
                - 1
            )
            (prefix_out, suffix_out), _ = module.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            if prefix_out is not None:
                raise AssertionError("instrumented JAX suffix unexpectedly returned prefix")
            velocity = module.action_out_proj(
                suffix_out[:, -module.action_horizon :]
            )
            x_next = x_t + dt * velocity
            trace = (x_t, velocity, x_next, current_time)
            return (x_next, current_time + dt), trace

        (final, _), (before, velocity, after, times) = jax.lax.scan(
            step,
            (noise_value, 1.0),
            xs=None,
            length=NUM_STEPS,
        )
        return {
            "final": final,
            "state_before": before,
            "velocity": velocity,
            "state_after": after,
            "time": times,
        }

    return jax.jit(traced)(state, observation, noise)


def _torch_instrumented_sampler(model: Any, device: Any, observation: Any, noise: Any):
    """Trace the same eager operations used by PI0Pytorch.sample_actions."""
    import torch
    from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks

    images, img_masks, lang_tokens, lang_masks, state = model._preprocess_observation(  # noqa: SLF001
        observation, train=False
    )
    prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
        images, img_masks, lang_tokens, lang_masks
    )
    prefix_att_2d_masks = make_att_2d_masks(
        prefix_pad_masks, prefix_att_masks
    )
    prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
    prefix_att_2d_masks_4d = model._prepare_attention_masks_4d(  # noqa: SLF001
        prefix_att_2d_masks
    )
    model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = (  # noqa: SLF001
        "eager"
    )
    _, past_key_values = model.paligemma_with_expert.forward(
        attention_mask=prefix_att_2d_masks_4d,
        position_ids=prefix_position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=True,
    )
    dt = torch.tensor(-1.0 / NUM_STEPS, dtype=torch.float32, device=device)
    current_time = torch.tensor(1.0, dtype=torch.float32, device=device)
    x_t = noise
    before = []
    velocities = []
    after = []
    times = []
    for _ in range(NUM_STEPS):
        expanded_time = current_time.expand(state.shape[0])
        velocity = model.denoise_step(
            state,
            prefix_pad_masks,
            past_key_values,
            x_t,
            expanded_time,
        )
        x_next = x_t + dt * velocity
        before.append(x_t)
        velocities.append(velocity)
        after.append(x_next)
        times.append(current_time)
        x_t = x_next
        current_time = current_time + dt
    return {
        "final": x_t,
        "state_before": torch.stack(before, dim=0),
        "velocity": torch.stack(velocities, dim=0),
        "state_after": torch.stack(after, dim=0),
        "time": torch.stack(times, dim=0),
    }


def _run_observation_worker(args: argparse.Namespace) -> int:
    _require_allocation()
    import numpy as np

    for path in (
        ROOT / "src",
        ROOT / "main",
        ROOT / "safelibero",
        ROOT / "openpi/packages/openpi-client/src",
    ):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from crfs_oracle.runner import (
        SafeLiberoCase,
        oracle_config_from_mapping,
        policy_observation,
    )

    records, errors = validate_jsonl_unique(args.manifest, "case_id")
    if errors:
        raise ValueError(f"invalid frozen manifest: {errors}")
    if not 0 <= args.case_index < len(records):
        raise IndexError(args.case_index)
    case = records[args.case_index]
    config_mapping = load_json(args.experiment_config)
    if not isinstance(config_mapping, Mapping):
        raise ValueError("sampler parity experiment config must be an object")
    # The endpoint-free scientific config nests planner-only values, whereas
    # the shared baseline environment wrapper consumes its small legacy
    # runtime surface at top level.  Mirror the already allocation-tested R01
    # CLI adapter; these fields do not alter the reconstructed branch.
    config_mapping = dict(config_mapping)
    config_mapping.setdefault(
        "intervention_step", int(config_mapping["sampler_steps"]) // 2
    )
    planner = config_mapping.get("planner", {})
    config_mapping.setdefault(
        "optimizer_max_iterations",
        int(planner.get("optimizer_max_iterations", 1))
        if isinstance(planner, Mapping)
        else 1,
    )
    config = oracle_config_from_mapping(
        config_mapping,
        host="127.0.0.1",
        port=0,
        checkpoint_id="sampler-parity-observation-only",
        checkpoint_sha256="0" * 64,
        output_root=str(args.worker_output.parent),
        run_id="sampler-parity-observation-only",
    )
    environment = SafeLiberoCase(case, config)
    started = time.monotonic()
    try:
        branch_observation = environment.reset_and_settle()
        raw = policy_observation(
            branch_observation, environment.prompt, config.resize_size
        )
        raw_hash, raw_manifest = fingerprint_tree(raw)
        _atomic_write_npz(
            args.raw_observation_output,
            {
                "observation_image": np.asarray(raw["observation/image"]),
                "observation_wrist_image": np.asarray(
                    raw["observation/wrist_image"]
                ),
                "observation_state": np.asarray(raw["observation/state"]),
                "prompt": np.asarray(raw["prompt"]),
            },
        )
    finally:
        environment.close()
    metadata = {
        "schema_version": WORKER_SCHEMA_VERSION,
        "worker": "observation",
        "fixture_id": FIXTURE_ID,
        "case": case,
        "case_record_sha256": content_hash(case),
        "manifest_sha256": file_sha256(args.manifest),
        "experiment_config_sha256": file_sha256(args.experiment_config),
        "raw_observation_sha256": raw_hash,
        "raw_observation_manifest": raw_manifest,
        "prompt": raw["prompt"],
        "state": np.asarray(raw["observation/state"], dtype=np.float64).tolist(),
        "obstacle_name": environment.obstacle_name,
        "runtime_seconds": time.monotonic() - started,
        "evidence_classification": "real frozen SafeLIBERO branch; sampler parity apparatus evidence only",
    }
    atomic_write_json(args.worker_output, metadata)
    return 0


def _run_model_worker(args: argparse.Namespace) -> int:
    _require_allocation()
    import numpy as np

    started = time.monotonic()
    raw = _load_raw_observation(args.raw_observation_output)
    raw_hash, raw_manifest = fingerprint_tree(raw)
    train_config, data_config, norm_stats, transform, output_transform = (
        _load_transform_context(args.config_name, args.jax_checkpoint)
    )
    if args.worker_mode == "jax":
        # Transform exactly once, in the public-JAX worker.  Persist the
        # already-batched NumPy tensors so the converted backend consumes the
        # identical bytes and never runs an independent input transform.
        transformed_unbatched = transform(raw)
        transformed = _map_tree(
            transformed_unbatched,
            lambda value: np.asarray(value)[np.newaxis, ...],
        )
        _atomic_write_array_tree(args.transformed_observation_output, transformed)
        noise = _make_noise(args.noise_seed)
        _atomic_write_noise(args.noise_output, noise)
    elif args.worker_mode == "pytorch":
        transformed = _load_array_tree(args.transformed_observation_output)
        noise = _load_noise(args.noise_output)
    else:
        raise ValueError(f"unsupported model worker: {args.worker_mode}")
    transformed_hash, transformed_manifest = fingerprint_tree(transformed)
    noise_hash, noise_manifest = fingerprint_tree({"noise": noise})
    transformed_unbatched = _map_tree(
        transformed, lambda value: np.asarray(value)[0]
    )

    from openpi.models import model as model_module

    if args.worker_mode == "jax":
        import jax
        import jax.numpy as jnp
        from openpi.shared import nnx_utils

        if jax.default_backend() != "gpu":
            raise RuntimeError(f"JAX backend is {jax.default_backend()!r}, not GPU")
        model = train_config.model.load(
            model_module.restore_params(
                args.jax_checkpoint / "params", dtype=jnp.bfloat16
            )
        )
        inputs = jax.tree.map(lambda value: jnp.asarray(value), transformed)
        observation = model_module.Observation.from_dict(inputs)
        noise_value = jnp.asarray(noise)
        default_sample = nnx_utils.module_jit(model.sample_actions)(
            jax.random.key(0),
            observation,
            noise=noise_value,
            num_steps=NUM_STEPS,
        )
        default_sample = jax.block_until_ready(default_sample)
        traced = _jax_instrumented_sampler(model, observation, noise_value)
        traced = jax.block_until_ready(traced)
        default_array = np.asarray(default_sample, dtype=np.float32)
        trace_arrays = {
            key: np.asarray(value, dtype=np.float32)
            for key, value in traced.items()
        }
        runtime = {
            "framework": "jax",
            "jax_version": _package_version("jax"),
            "jaxlib_version": _package_version("jaxlib"),
            "backend": jax.default_backend(),
            "devices": [str(device) for device in jax.devices()],
            "ordinary_sampler": "nnx_utils.module_jit(model.sample_actions)",
        }
    elif args.worker_mode == "pytorch":
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch CUDA is unavailable inside the allocation")
        device = torch.device("cuda:0")
        weight_path = args.pytorch_checkpoint / "model.safetensors"
        model = train_config.model.load_pytorch(train_config, str(weight_path))
        model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
        model = model.to(device)
        model.eval()
        inputs = _map_tree(
            transformed,
            lambda value: torch.from_numpy(np.array(value)).to(device),
        )
        observation = model_module.Observation.from_dict(inputs)
        noise_value = torch.from_numpy(noise).to(device)
        with torch.no_grad():
            default_sample = model.sample_actions(
                device,
                observation,
                noise=noise_value,
                num_steps=NUM_STEPS,
            )
            eager_sample, eager_midpoint_trace = model.sample_actions_eager(
                device,
                observation,
                noise=noise_value,
                num_steps=NUM_STEPS,
                crfs_intervention_mode="none",
                crfs_intervention_step=5,
                crfs_return_trace=True,
            )
            traced = _torch_instrumented_sampler(
                model, device, observation, noise_value
            )
        torch.cuda.synchronize(device)
        default_array = default_sample.detach().cpu().float().numpy()
        eager_array = eager_sample.detach().cpu().float().numpy()
        eager_midpoint = {
            key: value.detach().cpu().float().numpy()
            for key, value in eager_midpoint_trace.items()
        }
        trace_arrays = {
            key: value.detach().cpu().float().numpy()
            for key, value in traced.items()
        }
        runtime = {
            "framework": "pytorch",
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "device": torch.cuda.get_device_name(device),
            "ordinary_sampler": "compiled model.sample_actions",
            "eager_trace_only_sampler": "model.sample_actions_eager(mode=none, return_trace=True)",
        }
    else:
        raise ValueError(f"unsupported model worker: {args.worker_mode}")

    expected_default = (1, ACTION_HORIZON, ACTION_DIM)
    expected_trace = (NUM_STEPS, 1, ACTION_HORIZON, ACTION_DIM)
    if default_array.shape != expected_default:
        raise ValueError(
            f"{args.worker_mode} default shape {default_array.shape} != {expected_default}"
        )
    if args.worker_mode == "pytorch" and eager_array.shape != expected_default:
        raise ValueError(
            f"pytorch eager shape {eager_array.shape} != {expected_default}"
        )
    for name in ("state_before", "velocity", "state_after"):
        if trace_arrays[name].shape != expected_trace:
            raise ValueError(
                f"{args.worker_mode} {name} shape {trace_arrays[name].shape} != {expected_trace}"
            )
    if trace_arrays["time"].shape != (NUM_STEPS,):
        raise ValueError(f"{args.worker_mode} trace time shape is not ten scalars")
    if not all(
        np.all(np.isfinite(value))
        for value in [
            default_array,
            *trace_arrays.values(),
            *([eager_array, *eager_midpoint.values()]
              if args.worker_mode == "pytorch" else []),
        ]
    ):
        raise ValueError(f"{args.worker_mode} sampler produced non-finite values")

    default_physical = _unnormalize_actions(
        output_transform, transformed_unbatched, default_array[0]
    )
    traced_physical = _unnormalize_actions(
        output_transform, transformed_unbatched, trace_arrays["final"][0]
    )
    exact_checks = {
        "instrumented_final_array_equal_ordinary_default": bool(
            np.array_equal(default_array, trace_arrays["final"])
        ),
    }
    if args.worker_mode == "pytorch":
        exact_checks["compiled_default_array_equal_eager_trace_only"] = bool(
            np.array_equal(default_array, eager_array)
        )
    output_hash, output_manifest = fingerprint_tree(
        {
            "default_normalized": default_array,
            "traced_normalized": trace_arrays["final"],
            "default_unnormalized": default_physical,
            "traced_unnormalized": traced_physical,
            "trace_state_before": trace_arrays["state_before"],
            "trace_state_after": trace_arrays["state_after"],
            "trace_velocity": trace_arrays["velocity"],
            **(
                {"eager_normalized": eager_array}
                if args.worker_mode == "pytorch"
                else {}
            ),
        }
    )
    worker = {
        "schema_version": WORKER_SCHEMA_VERSION,
        "worker": args.worker_mode,
        "config_name": args.config_name,
        "asset_id": data_config.asset_id,
        "use_quantile_norm": bool(data_config.use_quantile_norm),
        "action_normalization": {
            "q01": np.asarray(norm_stats["actions"].q01, dtype=np.float64).tolist(),
            "q99": np.asarray(norm_stats["actions"].q99, dtype=np.float64).tolist(),
            "scale": (
                (
                    np.asarray(norm_stats["actions"].q99, dtype=np.float64)
                    - np.asarray(norm_stats["actions"].q01, dtype=np.float64)
                    + 1e-6
                )
                / 2.0
            ).tolist(),
        },
        "num_steps": NUM_STEPS,
        "dt": -1.0 / NUM_STEPS,
        "raw_observation_sha256": raw_hash,
        "raw_observation_manifest": raw_manifest,
        "transformed_observation_sha256": transformed_hash,
        "transformed_observation_manifest": transformed_manifest,
        "noise_seed": args.noise_seed,
        "noise_sha256": noise_hash,
        "noise_manifest": noise_manifest,
        "default_normalized": default_array.tolist(),
        "traced_normalized": trace_arrays["final"].tolist(),
        "default_unnormalized": default_physical.tolist(),
        "traced_unnormalized": traced_physical.tolist(),
        "exact_checks": exact_checks,
        "trace": {
            "time": trace_arrays["time"].tolist(),
            "state_before": trace_arrays["state_before"].tolist(),
            "velocity": trace_arrays["velocity"].tolist(),
            "state_after": trace_arrays["state_after"].tolist(),
        },
        **(
            {
                "eager_normalized": eager_array.tolist(),
                "eager_midpoint_trace": {
                    key: value.tolist() for key, value in eager_midpoint.items()
                },
            }
            if args.worker_mode == "pytorch"
            else {}
        ),
        "output_sha256": output_hash,
        "output_manifest": output_manifest,
        "runtime": {
            **runtime,
            "python": platform.python_version(),
            "host": socket.gethostname(),
            "seconds": time.monotonic() - started,
        },
    }
    atomic_write_json(args.worker_output, worker)
    return 0


def _map_tree(value: Any, fn: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _map_tree(item, fn) for key, item in value.items()}
    return fn(value)


def comparison_metrics(
    reference: Any,
    candidate: Any,
    *,
    max_limit: float,
    rms_limit: float,
    include_absolute_error: bool = False,
) -> dict[str, Any]:
    first = _plain_nested(reference)
    second = _plain_nested(candidate)
    first_shape = _nested_shape(first)
    second_shape = _nested_shape(second)
    if first_shape != second_shape:
        return {
            "shape_equal": False,
            "reference_shape": list(first_shape),
            "candidate_shape": list(second_shape),
            "passed": False,
        }
    first_flat = [float(value) for value in _flatten_nested(first)]
    second_flat = [float(value) for value in _flatten_nested(second)]
    if not first_flat:
        raise ValueError("cannot compare empty arrays")
    if not all(math.isfinite(value) for value in [*first_flat, *second_flat]):
        raise ValueError("cannot compare non-finite arrays")
    if not all(
        math.isfinite(value) and value >= 0.0 for value in (max_limit, rms_limit)
    ):
        raise ValueError("comparison limits must be finite and non-negative")
    absolute_flat = [abs(left - right) for left, right in zip(first_flat, second_flat)]
    max_error = max(absolute_flat)
    rms_error = math.sqrt(
        sum(error * error for error in absolute_flat) / len(absolute_flat)
    )
    result = {
        "shape_equal": True,
        "shape": list(first_shape),
        "maximum_error_limit": max_limit,
        "rms_error_limit": rms_limit,
        "bit_exact": first_flat == second_flat,
        "max_absolute_error": max_error,
        "mean_absolute_error": sum(absolute_flat) / len(absolute_flat),
        "rmse": rms_error,
        "values": len(absolute_flat),
        "passed": bool(max_error <= max_limit and rms_error <= rms_limit),
    }
    if include_absolute_error:
        result["absolute_error"] = _reshape_flat(iter(absolute_flat), first_shape)
    return result


def _per_step_metrics(
    reference: Any,
    candidate: Any,
    *,
    max_limit: float,
    rms_limit: float,
) -> list[dict[str, Any]]:
    first = _plain_nested(reference)
    second = _plain_nested(candidate)
    first_shape = _nested_shape(first)
    second_shape = _nested_shape(second)
    if first_shape != second_shape or not first_shape or first_shape[0] != NUM_STEPS:
        raise ValueError(
            f"trace shapes must agree and have ten steps: {first_shape}, {second_shape}"
        )
    return [
        {
            "step_index": index,
            "time": float(1.0 - index / NUM_STEPS),
            **comparison_metrics(
                first[index],
                second[index],
                max_limit=max_limit,
                rms_limit=rms_limit,
                include_absolute_error=True,
            ),
        }
        for index in range(NUM_STEPS)
    ]


def exact_comparison(reference: Any, candidate: Any) -> dict[str, Any]:
    first = _plain_nested(reference)
    second = _plain_nested(candidate)
    first_shape = _nested_shape(first)
    second_shape = _nested_shape(second)
    shape_equal = first_shape == second_shape
    first_flat = list(_flatten_nested(first))
    second_flat = list(_flatten_nested(second))
    if not all(
        math.isfinite(float(value)) for value in [*first_flat, *second_flat]
    ):
        raise ValueError("cannot compare non-finite arrays")
    exact = bool(shape_equal and first_flat == second_flat)
    result: dict[str, Any] = {
        "shape_equal": shape_equal,
        "reference_shape": list(first_shape),
        "candidate_shape": list(second_shape),
        "array_equal": exact,
        "passed": exact,
    }
    if shape_equal:
        absolute = [
            abs(float(left) - float(right))
            for left, right in zip(first_flat, second_flat)
        ]
        result.update(
            {
                "max_absolute_error": max(absolute) if absolute else 0.0,
                "rmse": (
                    math.sqrt(sum(item * item for item in absolute) / len(absolute))
                    if absolute
                    else 0.0
                ),
            }
        )
    return result


def _plain_nested(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, tuple):
        return [_plain_nested(item) for item in value]
    if isinstance(value, list):
        return [_plain_nested(item) for item in value]
    return value


def _nested_shape(value: Any) -> tuple[int, ...]:
    if not isinstance(value, list):
        return ()
    if not value:
        return (0,)
    child_shape = _nested_shape(value[0])
    if any(_nested_shape(item) != child_shape for item in value[1:]):
        raise ValueError("ragged arrays cannot be compared")
    return (len(value), *child_shape)


def _flatten_nested(value: Any):
    if isinstance(value, list):
        for item in value:
            yield from _flatten_nested(item)
    else:
        yield value


def _reshape_flat(values: Any, shape: tuple[int, ...]) -> Any:
    if not shape:
        return float(next(values))
    return [_reshape_flat(values, shape[1:]) for _ in range(shape[0])]


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def validate_parity_artifact(value: Any) -> list[str]:
    """Independently recompute the parity decision from stored raw arrays."""
    errors: list[str] = []
    if not isinstance(value, Mapping):
        return ["artifact must be an object"]
    required = {
        "schema_version",
        "artifact_type",
        "gate",
        "status",
        "identity",
        "observation",
        "acceptance",
        "comparison",
        "checkpoints",
        "samplers",
        "provenance",
    }
    missing = required - set(value)
    if missing:
        errors.append(f"missing required fields: {sorted(missing)}")
    if value.get("schema_version") != SCHEMA_VERSION:
        errors.append("unsupported schema_version")
    if value.get("artifact_type") != "sampler_parity":
        errors.append("artifact_type must be sampler_parity")
    if value.get("gate") != "R02":
        errors.append("gate must be R02")
    if value.get("status") not in {"passed", "failed"}:
        errors.append("status must be passed or failed")

    identity = value.get("identity")
    if not isinstance(identity, Mapping):
        errors.append("identity must be an object")
        identity = {}
    else:
        if identity.get("num_steps") != NUM_STEPS:
            errors.append("identity.num_steps must be ten")
        if identity.get("action_horizon") != ACTION_HORIZON:
            errors.append("identity.action_horizon must be ten")
        if identity.get("action_dim") != ACTION_DIM:
            errors.append("identity.action_dim must be 32")
        if identity.get("config_name") != "pi05_libero":
            errors.append("identity.config_name must be pi05_libero")
        if not isinstance(identity.get("case_id"), str) or not identity.get("case_id"):
            errors.append("identity.case_id must be non-empty")
        if not isinstance(identity.get("case_index"), int) or isinstance(
            identity.get("case_index"), bool
        ):
            errors.append("identity.case_index must be an integer")
        if not isinstance(identity.get("noise_seed"), int) or isinstance(
            identity.get("noise_seed"), bool
        ):
            errors.append("identity.noise_seed must be an integer")
        if identity.get("r01_summary_sha256") != R01_SUMMARY_SHA256:
            errors.append("identity is not content-bound to accepted R01")
        for key in (
            "case_record_sha256",
            "manifest_sha256",
            "jax_params_sha256",
            "pytorch_model_sha256",
            "norm_stats_sha256",
            "runner_sha256",
            "tracked_diff_sha256",
        ):
            if not _is_sha256(identity.get(key)):
                errors.append(f"identity.{key} must be a lowercase SHA-256")

    acceptance = value.get("acceptance")
    if not isinstance(acceptance, Mapping):
        errors.append("acceptance must be an object")
        acceptance = {}
    passed = acceptance.get("passed")
    if not isinstance(passed, bool):
        errors.append("acceptance.passed must be boolean")
    elif (value.get("status") == "passed") != passed:
        errors.append("status conflicts with acceptance.passed")
    criteria = acceptance.get("registered_tolerances")
    if not isinstance(criteria, Mapping):
        errors.append("registered tolerances must be recorded")
    else:
        for key, expected in REGISTERED_LIMITS.items():
            if criteria.get(key) != expected:
                errors.append(
                    f"registered tolerance {key} must equal ADR-0010 value {expected}"
                )

    observation = value.get("observation")
    if not isinstance(observation, Mapping):
        errors.append("observation must be an object")
        observation = {}
    for key in (
        "raw_observation_sha256",
        "transformed_observation_sha256",
        "manifest_sha256",
        "case_record_sha256",
        "experiment_config_sha256",
    ):
        if not _is_sha256(observation.get(key)):
            errors.append(f"observation.{key} must be a lowercase SHA-256")
    if observation.get("fixture_id") != FIXTURE_ID:
        errors.append("observation fixture_id differs from the frozen real branch")
    if observation.get("manifest_sha256") != identity.get("manifest_sha256"):
        errors.append("observation manifest hash conflicts with identity")
    if observation.get("case_record_sha256") != identity.get(
        "case_record_sha256"
    ):
        errors.append("observation case-record hash conflicts with identity")
    if identity.get("fixture_id") != FIXTURE_ID:
        errors.append("identity fixture_id differs from the frozen real branch")

    samplers = value.get("samplers")
    if not isinstance(samplers, Mapping) or set(samplers) != {"jax", "pytorch"}:
        errors.append("samplers must contain exactly jax and pytorch")
        samplers = {}
    jax_worker = samplers.get("jax") if isinstance(samplers, Mapping) else None
    torch_worker = samplers.get("pytorch") if isinstance(samplers, Mapping) else None
    for name, worker in (("jax", jax_worker), ("pytorch", torch_worker)):
        if not isinstance(worker, Mapping):
            errors.append(f"samplers.{name} must be an object")
            continue
        if worker.get("worker") != name:
            errors.append(f"samplers.{name}.worker identity differs")
        if worker.get("num_steps") != NUM_STEPS or worker.get("dt") != -0.1:
            errors.append(f"samplers.{name} must record ten dt=-0.1 updates")
        if worker.get("config_name") != "pi05_libero":
            errors.append(f"samplers.{name}.config_name differs")
        if worker.get("noise_seed") != identity.get("noise_seed"):
            errors.append(f"samplers.{name}.noise_seed conflicts with identity")
        for key in (
            "raw_observation_sha256",
            "transformed_observation_sha256",
            "noise_sha256",
            "output_sha256",
        ):
            if not _is_sha256(worker.get(key)):
                errors.append(f"samplers.{name}.{key} must be a lowercase SHA-256")
        noise_manifest = worker.get("noise_manifest")
        if not isinstance(noise_manifest, list) or len(noise_manifest) != 1:
            errors.append(f"samplers.{name}.noise_manifest must have one leaf")
        else:
            leaf = noise_manifest[0]
            if not isinstance(leaf, Mapping) or leaf.get("path") != "noise":
                errors.append(f"samplers.{name}.noise manifest path differs")
            elif leaf.get("shape") != [1, ACTION_HORIZON, ACTION_DIM] or leaf.get(
                "dtype"
            ) not in {"<f4", ">f4", "=f4"}:
                errors.append(f"samplers.{name} noise is not float32 [1,10,32]")

    checkpoints = value.get("checkpoints")
    if not isinstance(checkpoints, Mapping):
        errors.append("checkpoints must be an object")
        checkpoints = {}
    public_jax = checkpoints.get("public_jax", {})
    converted = checkpoints.get("converted_pytorch", {})
    if not isinstance(public_jax, Mapping) or not isinstance(converted, Mapping):
        errors.append("both checkpoint records must be objects")
        public_jax, converted = {}, {}
    params_directory = public_jax.get("params_directory", {})
    if not isinstance(params_directory, Mapping) or not _is_sha256(
        params_directory.get("sha256")
    ):
        errors.append("public JAX parameter tree hash is invalid")
    elif params_directory.get("sha256") != identity.get("jax_params_sha256"):
        errors.append("public JAX parameter hash conflicts with identity")
    if not _is_sha256(converted.get("model_sha256")):
        errors.append("converted PyTorch model hash is invalid")
    elif converted.get("model_sha256") != identity.get("pytorch_model_sha256"):
        errors.append("converted PyTorch model hash conflicts with identity")
    jax_norm_hash = public_jax.get("norm_stats_sha256")
    torch_norm_hash = converted.get("norm_stats_sha256")
    if not _is_sha256(jax_norm_hash) or not _is_sha256(torch_norm_hash):
        errors.append("checkpoint normalization hashes are invalid")
    elif not (
        jax_norm_hash
        == torch_norm_hash
        == identity.get("norm_stats_sha256")
        == REGISTERED_NORM_STATS_SHA256
    ):
        errors.append("checkpoint normalization hashes differ from registered asset")

    provenance = value.get("provenance")
    if not isinstance(provenance, Mapping):
        errors.append("provenance must be an object")
        provenance = {}
    if provenance.get("runner_sha256") != identity.get("runner_sha256"):
        errors.append("runner hash conflicts with identity")
    if provenance.get("git_commit") != identity.get("git_commit"):
        errors.append("git commit conflicts with identity")
    if provenance.get("git_dirty") != identity.get("git_dirty"):
        errors.append("git dirty state conflicts with identity")
    if provenance.get("tracked_diff_sha256") != identity.get(
        "tracked_diff_sha256"
    ):
        errors.append("tracked diff hash conflicts with identity")
    if provenance.get("experiment_config_sha256") != observation.get(
        "experiment_config_sha256"
    ):
        errors.append("experiment config hash conflicts with observation provenance")
    slurm = provenance.get("slurm")
    if not isinstance(slurm, Mapping) or not slurm.get("job_id"):
        errors.append("allocation-backed Slurm job identity is missing")

    comparison = value.get("comparison")
    if not isinstance(comparison, Mapping):
        errors.append("comparison must be an object")
        comparison = {}
    checks = acceptance.get("checks")
    if not isinstance(checks, Mapping) or not all(
        isinstance(item, bool) for item in checks.values()
    ):
        errors.append("acceptance checks must be booleans")
        checks = {}

    if isinstance(jax_worker, Mapping) and isinstance(torch_worker, Mapping):
        try:
            expected_shapes = {
                "default_normalized": (1, ACTION_HORIZON, ACTION_DIM),
                "traced_normalized": (1, ACTION_HORIZON, ACTION_DIM),
                "default_unnormalized": (ACTION_HORIZON, LIBERO_ACTION_DIM),
                "traced_unnormalized": (ACTION_HORIZON, LIBERO_ACTION_DIM),
            }
            for name, worker in (("jax", jax_worker), ("pytorch", torch_worker)):
                for key, expected_shape in expected_shapes.items():
                    if _nested_shape(_plain_nested(worker[key])) != expected_shape:
                        errors.append(
                            f"samplers.{name}.{key} has the wrong shape"
                        )
                trace = worker["trace"]
                if not isinstance(trace, Mapping):
                    raise ValueError(f"samplers.{name}.trace is not an object")
                for key in ("state_before", "velocity", "state_after"):
                    if _nested_shape(_plain_nested(trace[key])) != (
                        NUM_STEPS,
                        1,
                        ACTION_HORIZON,
                        ACTION_DIM,
                    ):
                        errors.append(f"samplers.{name}.trace.{key} has the wrong shape")
                if _nested_shape(_plain_nested(trace["time"])) != (NUM_STEPS,):
                    errors.append(f"samplers.{name}.trace.time must have ten values")
                if not exact_comparison(
                    trace["state_after"][-1], worker["traced_normalized"]
                )["passed"]:
                    errors.append(f"samplers.{name} trace final is internally inconsistent")
                if not exact_comparison(
                    trace["state_before"][1:], trace["state_after"][:-1]
                )["passed"]:
                    errors.append(f"samplers.{name} trace state continuity failed")

            jax_exact = exact_comparison(
                jax_worker["default_normalized"], jax_worker["traced_normalized"]
            )
            torch_exact = exact_comparison(
                torch_worker["default_normalized"],
                torch_worker["traced_normalized"],
            )
            torch_compiled_eager = exact_comparison(
                torch_worker["default_normalized"],
                torch_worker["eager_normalized"],
            )
            cross_x = _per_step_metrics(
                jax_worker["trace"]["state_before"],
                torch_worker["trace"]["state_before"],
                max_limit=REGISTERED_LIMITS["step_x_max"],
                rms_limit=REGISTERED_LIMITS["step_x_rms"],
            )
            cross_v = _per_step_metrics(
                jax_worker["trace"]["velocity"],
                torch_worker["trace"]["velocity"],
                max_limit=REGISTERED_LIMITS["step_v_max"],
                rms_limit=REGISTERED_LIMITS["step_v_rms"],
            )
            final_model = comparison_metrics(
                jax_worker["default_normalized"],
                torch_worker["default_normalized"],
                max_limit=REGISTERED_LIMITS["final_model_max"],
                rms_limit=REGISTERED_LIMITS["final_model_rms"],
                include_absolute_error=True,
            )
            jax_physical = _plain_nested(jax_worker["default_unnormalized"])
            torch_physical = _plain_nested(torch_worker["default_unnormalized"])
            physical_xyz5 = comparison_metrics(
                [row[:3] for row in jax_physical[:5]],
                [row[:3] for row in torch_physical[:5]],
                max_limit=REGISTERED_LIMITS["physical_xyz5_max"],
                rms_limit=REGISTERED_LIMITS["physical_xyz5_rms"],
                include_absolute_error=True,
            )
            physical_action7 = comparison_metrics(
                jax_physical,
                torch_physical,
                max_limit=REGISTERED_LIMITS["physical_action7_max"],
                rms_limit=REGISTERED_LIMITS["physical_action7_rms"],
                include_absolute_error=True,
            )
            recomputed_comparison = {
                "jax_trace_vs_ordinary_final": jax_exact,
                "pytorch_trace_vs_ordinary_final": torch_exact,
                "pytorch_compiled_vs_eager_trace_only_final": torch_compiled_eager,
                "cross_backend_pre_update_x_t_per_step": cross_x,
                "cross_backend_pre_update_v_t_per_step": cross_v,
                "cross_backend_final_normalized": final_model,
                "cross_backend_final_physical_first_five_xyz": physical_xyz5,
                "cross_backend_final_physical_first_seven": physical_action7,
            }
            for key, recomputed in recomputed_comparison.items():
                if comparison.get(key) != recomputed:
                    errors.append(f"comparison.{key} differs from recomputed errors")

            raw_shared = (
                observation.get("raw_observation_sha256")
                == jax_worker.get("raw_observation_sha256")
                == torch_worker.get("raw_observation_sha256")
            )
            transformed_shared = (
                observation.get("transformed_observation_sha256")
                == jax_worker.get("transformed_observation_sha256")
                == torch_worker.get("transformed_observation_sha256")
            )
            noise_shared = jax_worker.get("noise_sha256") == torch_worker.get(
                "noise_sha256"
            )
            expected_checks = {
                "real_frozen_branch_observation_shared": raw_shared,
                "transformed_observation_byte_identical": transformed_shared,
                "explicit_noise_byte_identical": noise_shared,
                "checkpoint_norm_stats_byte_identical": jax_norm_hash
                == torch_norm_hash,
                "checkpoint_norm_stats_match_registered_asset": jax_norm_hash
                == REGISTERED_NORM_STATS_SHA256,
                "jax_trace_reproduces_ordinary_final": bool(jax_exact["passed"]),
                "pytorch_trace_reproduces_ordinary_final": bool(
                    torch_exact["passed"]
                ),
                "pytorch_compiled_equals_eager_trace_only_final": bool(
                    torch_compiled_eager["passed"]
                ),
                "all_ten_cross_backend_pre_update_states_within_limits": all(
                    item["passed"] for item in cross_x
                ),
                "all_ten_cross_backend_velocities_within_limits": all(
                    item["passed"] for item in cross_v
                ),
                "cross_backend_final_normalized_within_tolerance": bool(
                    final_model["passed"]
                ),
                "cross_backend_final_physical_first_five_xyz_within_limits": bool(
                    physical_xyz5["passed"]
                ),
                "cross_backend_final_physical_first_seven_within_limits": bool(
                    physical_action7["passed"]
                ),
            }
            if checks != expected_checks:
                errors.append("acceptance checks differ from recomputed checks")
            recomputed_pass = all(expected_checks.values())
            if passed != recomputed_pass:
                errors.append("acceptance.passed differs from recomputed gate")
            if (value.get("status") == "passed") != recomputed_pass:
                errors.append("status differs from recomputed gate")
            for name, worker, exact_result in (
                ("jax", jax_worker, jax_exact),
                ("pytorch", torch_worker, torch_exact),
            ):
                exact_checks = worker.get("exact_checks")
                if not isinstance(exact_checks, Mapping) or exact_checks.get(
                    "instrumented_final_array_equal_ordinary_default"
                ) != exact_result["passed"]:
                    errors.append(f"samplers.{name}.exact_checks conflicts with arrays")
            torch_exact_checks = torch_worker.get("exact_checks", {})
            if not isinstance(torch_exact_checks, Mapping) or torch_exact_checks.get(
                "compiled_default_array_equal_eager_trace_only"
            ) != torch_compiled_eager["passed"]:
                errors.append("PyTorch compiled/eager exact check conflicts with arrays")
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            errors.append(f"stored sampler arrays cannot be independently validated: {error}")

    return errors


def _valid_matching_completion(path: Path, identity: Mapping[str, Any]) -> bool:
    try:
        value = load_json(path)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return False
    return (
        isinstance(value, Mapping)
        and not validate_parity_artifact(value)
        and value.get("identity") == identity
    )


def _worker_command(
    args: argparse.Namespace,
    *,
    python: Path,
    mode: str,
    output: Path,
    raw_observation: Path,
    transformed_observation: Path,
    noise_output: Path,
) -> list[str]:
    return [
        str(python),
        str(Path(__file__).resolve()),
        "--worker-mode",
        mode,
        "--worker-output",
        str(output),
        "--raw-observation-output",
        str(raw_observation),
        "--transformed-observation-output",
        str(transformed_observation),
        "--noise-output",
        str(noise_output),
        "--manifest",
        str(args.manifest),
        "--experiment-config",
        str(args.experiment_config),
        "--case-index",
        str(args.case_index),
        "--config-name",
        args.config_name,
        "--jax-checkpoint",
        str(args.jax_checkpoint),
        "--pytorch-checkpoint",
        str(args.pytorch_checkpoint),
        "--noise-seed",
        str(args.noise_seed),
    ]


def _run_parent(args: argparse.Namespace) -> int:
    _require_allocation()
    import numpy as np

    if NUM_STEPS != 10:
        raise AssertionError("the parity gate is registered for ten Euler steps")
    if args.config_name != "pi05_libero":
        raise ValueError("ADR-0010 parity is frozen to config pi05_libero")
    for name, registered in REGISTERED_LIMITS.items():
        value = float(getattr(args, name))
        if not math.isfinite(value) or value != registered:
            raise ValueError(
                f"{name} must equal preregistered ADR-0010 value {registered}, got {value}"
            )
    if not (args.jax_checkpoint / "params").is_dir():
        raise FileNotFoundError(args.jax_checkpoint / "params")
    pytorch_model = args.pytorch_checkpoint / "model.safetensors"
    if not pytorch_model.is_file():
        raise FileNotFoundError(pytorch_model)
    r01_path = ROOT / R01_SUMMARY_RELATIVE
    if file_sha256(r01_path) != R01_SUMMARY_SHA256:
        raise RuntimeError("accepted R01 summary content hash changed")

    source_norm_preflight = (
        args.jax_checkpoint
        / "assets/physical-intelligence/libero/norm_stats.json"
    )
    converted_norm_preflight = (
        args.pytorch_checkpoint
        / "assets/physical-intelligence/libero/norm_stats.json"
    )
    if not source_norm_preflight.is_file() or not converted_norm_preflight.is_file():
        raise FileNotFoundError("pi05_libero normalization assets are missing")
    source_norm_preflight_sha = file_sha256(source_norm_preflight)
    converted_norm_preflight_sha = file_sha256(converted_norm_preflight)
    pytorch_model_sha = file_sha256(pytorch_model)
    jax_params_digest = _directory_sha256(args.jax_checkpoint / "params")
    runner_sha = file_sha256(Path(__file__))
    git = _git_state(ROOT)

    records, manifest_errors = validate_jsonl_unique(args.manifest, "case_id")
    if manifest_errors:
        raise ValueError(f"invalid frozen manifest: {manifest_errors}")
    if not 0 <= args.case_index < len(records):
        raise IndexError(args.case_index)
    case = records[args.case_index]
    if args.noise_seed != int(case["policy_seed"]):
        raise ValueError(
            "parity noise seed must equal the selected frozen case policy_seed"
        )

    identity = {
        "fixture_id": FIXTURE_ID,
        "case_id": case["case_id"],
        "case_index": args.case_index,
        "case_record_sha256": content_hash(case),
        "manifest_sha256": file_sha256(args.manifest),
        "config_name": args.config_name,
        "jax_checkpoint": str(args.jax_checkpoint.resolve()),
        "pytorch_checkpoint": str(args.pytorch_checkpoint.resolve()),
        "jax_params_sha256": jax_params_digest["sha256"],
        "pytorch_model_sha256": pytorch_model_sha,
        "norm_stats_sha256": source_norm_preflight_sha,
        "runner_sha256": runner_sha,
        "git_commit": git["commit"],
        "git_dirty": git["dirty"],
        "tracked_diff_sha256": git["tracked_diff_sha256"],
        "noise_seed": args.noise_seed,
        "num_steps": NUM_STEPS,
        "action_horizon": ACTION_HORIZON,
        "action_dim": ACTION_DIM,
        "r01_summary_sha256": R01_SUMMARY_SHA256,
    }
    if _valid_matching_completion(args.output, identity):
        existing = load_json(args.output)
        print(f"skipped schema-valid final artifact: {args.output}", flush=True)
        return 0 if existing["status"] == "passed" else 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    started_at = _utc_now()
    started = time.monotonic()
    with tempfile.TemporaryDirectory(
        prefix=".sampler-parity-", dir=args.output.parent
    ) as temporary_name:
        temporary = Path(temporary_name)
        raw_path = temporary / "raw-observation.npz"
        transformed_path = temporary / "transformed-observation.npz"
        noise_path = temporary / "explicit-noise.npz"
        observation_output = temporary / "observation.json"
        jax_output = temporary / "jax.json"
        torch_output = temporary / "pytorch.json"

        observation_command = _worker_command(
            args,
            python=args.libero_python,
            mode="observation",
            output=observation_output,
            raw_observation=raw_path,
            transformed_observation=transformed_path,
            noise_output=noise_path,
        )
        subprocess.run(observation_command, cwd=ROOT, check=True)
        for mode, output in (("jax", jax_output), ("pytorch", torch_output)):
            command = _worker_command(
                args,
                python=Path(sys.executable),
                mode=mode,
                output=output,
                raw_observation=raw_path,
                transformed_observation=transformed_path,
                noise_output=noise_path,
            )
            subprocess.run(command, cwd=ROOT, check=True)

        observation = load_json(observation_output)
        jax_worker = load_json(jax_output)
        torch_worker = load_json(torch_output)

    shared_observation = (
        observation["raw_observation_sha256"]
        == jax_worker["raw_observation_sha256"]
        == torch_worker["raw_observation_sha256"]
    )
    transformed_equal = (
        jax_worker["transformed_observation_sha256"]
        == torch_worker["transformed_observation_sha256"]
    )
    noise_equal = jax_worker["noise_sha256"] == torch_worker["noise_sha256"]
    source_norm = (
        args.jax_checkpoint
        / "assets"
        / str(jax_worker["asset_id"])
        / "norm_stats.json"
    )
    converted_norm = (
        args.pytorch_checkpoint
        / "assets"
        / str(jax_worker["asset_id"])
        / "norm_stats.json"
    )
    if not source_norm.is_file() or not converted_norm.is_file():
        raise FileNotFoundError(
            f"normalization asset missing: {source_norm}, {converted_norm}"
        )
    source_norm_sha = file_sha256(source_norm)
    converted_norm_sha = file_sha256(converted_norm)
    if (
        source_norm_sha != source_norm_preflight_sha
        or converted_norm_sha != converted_norm_preflight_sha
    ):
        raise RuntimeError("normalization assets changed during the parity job")
    norms_equal = source_norm_sha == converted_norm_sha
    norm_is_registered = source_norm_sha == REGISTERED_NORM_STATS_SHA256

    jax_backend = exact_comparison(
        jax_worker["default_normalized"],
        jax_worker["traced_normalized"],
    )
    torch_backend = exact_comparison(
        torch_worker["default_normalized"],
        torch_worker["traced_normalized"],
    )
    torch_compiled_vs_eager = exact_comparison(
        torch_worker["default_normalized"],
        torch_worker["eager_normalized"],
    )
    cross_steps = _per_step_metrics(
        jax_worker["trace"]["state_before"],
        torch_worker["trace"]["state_before"],
        max_limit=args.step_x_max,
        rms_limit=args.step_x_rms,
    )
    cross_velocity_steps = _per_step_metrics(
        jax_worker["trace"]["velocity"],
        torch_worker["trace"]["velocity"],
        max_limit=args.step_v_max,
        rms_limit=args.step_v_rms,
    )
    cross_final_normalized = comparison_metrics(
        jax_worker["default_normalized"],
        torch_worker["default_normalized"],
        max_limit=args.final_model_max,
        rms_limit=args.final_model_rms,
        include_absolute_error=True,
    )
    jax_physical = np.asarray(jax_worker["default_unnormalized"])
    torch_physical = np.asarray(torch_worker["default_unnormalized"])
    cross_final_physical_xyz5 = comparison_metrics(
        jax_physical[:5, :3],
        torch_physical[:5, :3],
        max_limit=args.physical_xyz5_max,
        rms_limit=args.physical_xyz5_rms,
        include_absolute_error=True,
    )
    cross_final_physical_action7 = comparison_metrics(
        jax_worker["default_unnormalized"],
        torch_worker["default_unnormalized"],
        max_limit=args.physical_action7_max,
        rms_limit=args.physical_action7_rms,
        include_absolute_error=True,
    )

    checks = {
        "real_frozen_branch_observation_shared": shared_observation,
        "transformed_observation_byte_identical": transformed_equal,
        "explicit_noise_byte_identical": noise_equal,
        "checkpoint_norm_stats_byte_identical": norms_equal,
        "checkpoint_norm_stats_match_registered_asset": norm_is_registered,
        "jax_trace_reproduces_ordinary_final": bool(
            jax_backend["passed"]
            and jax_worker["exact_checks"][
                "instrumented_final_array_equal_ordinary_default"
            ]
        ),
        "pytorch_trace_reproduces_ordinary_final": bool(
            torch_backend["passed"]
            and torch_worker["exact_checks"][
                "instrumented_final_array_equal_ordinary_default"
            ]
        ),
        "pytorch_compiled_equals_eager_trace_only_final": bool(
            torch_compiled_vs_eager["passed"]
            and torch_worker["exact_checks"][
                "compiled_default_array_equal_eager_trace_only"
            ]
        ),
        "all_ten_cross_backend_pre_update_states_within_limits": all(
            item["passed"] for item in cross_steps
        ),
        "all_ten_cross_backend_velocities_within_limits": all(
            item["passed"] for item in cross_velocity_steps
        ),
        "cross_backend_final_normalized_within_tolerance": bool(
            cross_final_normalized["passed"]
        ),
        "cross_backend_final_physical_first_five_xyz_within_limits": bool(
            cross_final_physical_xyz5["passed"]
        ),
        "cross_backend_final_physical_first_seven_within_limits": bool(
            cross_final_physical_action7["passed"]
        ),
    }
    passed = all(checks.values())
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "sampler_parity",
        "gate": "R02",
        "status": "passed" if passed else "failed",
        "identity": identity,
        "observation": {
            **observation,
            "transformed_observation_sha256": jax_worker[
                "transformed_observation_sha256"
            ],
            "transformed_observation_manifest": jax_worker[
                "transformed_observation_manifest"
            ],
            "transform_application": "applied once by public-JAX worker; serialized batched tensors loaded directly by PyTorch worker",
            "classification": "real frozen SafeLIBERO branch; parity is apparatus evidence only",
            "downstream_requirement": "R02 must separately reproduce all 20 frozen nominal collision branches",
        },
        "acceptance": {
            "registered_before_execution": True,
            "registered_tolerances": {
                **REGISTERED_LIMITS,
                "within_backend_rule": "numpy.array_equal",
                "cross_backend_rule": "both maximum absolute error and RMS absolute error must pass",
                "source": "docs/decisions/0010-freeze-r02-parity-and-directions.md",
            },
            "checks": checks,
            "passed": passed,
            "failure_policy": "fail closed; do not interpret R02 policy outcomes",
        },
        "comparison": {
            "jax_trace_vs_ordinary_final": jax_backend,
            "pytorch_trace_vs_ordinary_final": torch_backend,
            "pytorch_compiled_vs_eager_trace_only_final": torch_compiled_vs_eager,
            "cross_backend_pre_update_x_t_per_step": cross_steps,
            "cross_backend_pre_update_v_t_per_step": cross_velocity_steps,
            "cross_backend_final_normalized": cross_final_normalized,
            "cross_backend_final_physical_first_five_xyz": cross_final_physical_xyz5,
            "cross_backend_final_physical_first_seven": cross_final_physical_action7,
        },
        "checkpoints": {
            "public_jax": {
                "path": str(args.jax_checkpoint.resolve()),
                "params_directory": jax_params_digest,
                "norm_stats_path": str(source_norm),
                "norm_stats_sha256": source_norm_sha,
            },
            "converted_pytorch": {
                "path": str(args.pytorch_checkpoint.resolve()),
                "model_path": str(pytorch_model),
                "model_sha256": pytorch_model_sha,
                "model_bytes": pytorch_model.stat().st_size,
                "norm_stats_path": str(converted_norm),
                "norm_stats_sha256": converted_norm_sha,
            },
        },
        "samplers": {
            "jax": jax_worker,
            "pytorch": torch_worker,
        },
        "provenance": {
            "created_at": _utc_now(),
            "started_at": started_at,
            "runtime_seconds": time.monotonic() - started,
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "git_commit": git["commit"],
            "git_dirty": git["dirty"],
            "git_status_short": git["status_short"],
            "tracked_diff_sha256": git["tracked_diff_sha256"],
            "baseline_repository": "THU-RCSCT/VLSA-Aegis",
            "baseline_commit": BASELINE_COMMIT,
            "runner_path": str(Path(__file__).resolve()),
            "runner_sha256": runner_sha,
            "experiment_config_path": str(args.experiment_config.resolve()),
            "experiment_config_sha256": file_sha256(args.experiment_config),
            "slurm": {
                "job_id": os.environ.get("SLURM_JOB_ID"),
                "array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
                "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
                "partition": os.environ.get("SLURM_JOB_PARTITION"),
                "submit_host": os.environ.get("SLURM_SUBMIT_HOST"),
            },
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "device": jax_worker["runtime"].get("devices"),
            "numpy_version": np.__version__,
        },
    }
    validation_errors = validate_parity_artifact(artifact)
    if validation_errors:
        raise RuntimeError(f"refusing invalid parity artifact: {validation_errors}")
    atomic_write_json(args.output, artifact)
    print(
        json.dumps(
            {
                "event": "sampler_parity_complete",
                "status": artifact["status"],
                "output": str(args.output),
                "max_final_normalized_error": cross_final_normalized[
                    "max_absolute_error"
                ],
                "max_final_unnormalized_error": cross_final_physical_action7[
                    "max_absolute_error"
                ],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if passed else 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jax-checkpoint", type=Path)
    parser.add_argument("--pytorch-checkpoint", type=Path)
    parser.add_argument("--config-name", default="pi05_libero")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--experiment-config", type=Path)
    parser.add_argument("--case-index", type=int, default=0)
    parser.add_argument("--noise-seed", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--libero-python", type=Path)
    parser.add_argument("--step-x-max", type=float)
    parser.add_argument("--step-x-rms", type=float)
    parser.add_argument("--step-v-max", type=float)
    parser.add_argument("--step-v-rms", type=float)
    parser.add_argument("--final-model-max", type=float)
    parser.add_argument("--final-model-rms", type=float)
    parser.add_argument("--physical-xyz5-max", type=float)
    parser.add_argument("--physical-xyz5-rms", type=float)
    parser.add_argument("--physical-action7-max", type=float)
    parser.add_argument("--physical-action7-rms", type=float)
    parser.add_argument(
        "--worker-mode", choices=("observation", "jax", "pytorch"), help=argparse.SUPPRESS
    )
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument(
        "--raw-observation-output", type=Path, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--transformed-observation-output", type=Path, help=argparse.SUPPRESS
    )
    parser.add_argument("--noise-output", type=Path, help=argparse.SUPPRESS)
    return parser


def _require_arguments(args: argparse.Namespace, names: tuple[str, ...]) -> None:
    absent = [name for name in names if getattr(args, name) is None]
    if absent:
        raise ValueError(f"missing required arguments: {absent}")


def main() -> int:
    args = _parser().parse_args()
    common = (
        "jax_checkpoint",
        "pytorch_checkpoint",
        "manifest",
        "experiment_config",
        "noise_seed",
    )
    if args.worker_mode == "observation":
        _require_arguments(
            args,
            (*common, "worker_output", "raw_observation_output"),
        )
        return _run_observation_worker(args)
    if args.worker_mode in {"jax", "pytorch"}:
        _require_arguments(
            args,
            (
                *common,
                "worker_output",
                "raw_observation_output",
                "transformed_observation_output",
                "noise_output",
            ),
        )
        return _run_model_worker(args)
    _require_arguments(
        args,
        (
            *common,
            "output",
            "libero_python",
            *tuple(REGISTERED_LIMITS),
        ),
    )
    return _run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
