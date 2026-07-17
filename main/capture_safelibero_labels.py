#!/usr/bin/env python3
"""Capture label-ready SafeLIBERO observations without running an outcome arm.

For every selected manifest case this script performs only:

1. environment construction and reset,
2. exact initial-state restoration,
3. exactly 20 released dummy settle actions, and
4. lossless agent/back RGB and depth capture.

It does not connect to a policy server or execute perception, geometry fitting,
optimization, or semantic selection.  A reviewer can use the PNG plus task
instruction to create a separate ``vlsa_table1_codex_label.v1`` JSONL record
bound to ``settled_agentview_array_sha256``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
import traceback
from typing import Any, Mapping, Sequence

from evaluate_safelibero_aegis import (
    CAPTURE_SCHEMA,
    ProtocolError,
    TABLE_RENDER_RESOLUTION,
    TABLE_SETTLE_ACTIONS,
    _build_environment,
    _processed_image,
    _settle,
    array_sha256,
    atomic_write_json,
    canonical_json_bytes,
    git_identity,
    read_jsonl,
    select_cases,
    sha256_bytes,
    sha256_path,
    validate_case_row,
)


def _capture_runtime() -> dict[str, Any]:
    """Import only the simulator and image dependencies required to capture."""

    try:
        from libero.libero import benchmark
        from libero.libero import get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
        import numpy as np
        from PIL import Image
    except Exception as error:
        raise RuntimeError(
            f"failed to import SafeLIBERO capture runtime: {error}"
        ) from error
    return {
        "benchmark": benchmark,
        "get_libero_path": get_libero_path,
        "OffScreenRenderEnv": OffScreenRenderEnv,
        "np": np,
        "Image": Image,
    }


def _atomic_save_npy(path: Path, array: Any, np: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.save(stream, np.ascontiguousarray(array), allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return sha256_path(path)


def _atomic_save_png(path: Path, array: Any, Image: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.png")
    Image.fromarray(array).save(temporary, format="PNG", compress_level=6)
    os.replace(temporary, path)
    return sha256_path(path)


def _asset_record(
    *,
    case_dir: Path,
    name: str,
    array: Any,
    np: Any,
    Image: Any | None,
) -> dict[str, Any]:
    contiguous = np.ascontiguousarray(array)
    npy_path = case_dir / f"{name}.npy"
    record = {
        "array_sha256": array_sha256(contiguous),
        "shape": list(contiguous.shape),
        "dtype": contiguous.dtype.str,
        "npy_path": npy_path.name,
        "npy_sha256": _atomic_save_npy(npy_path, contiguous, np),
    }
    if Image is not None:
        png_path = case_dir / f"{name}.png"
        record.update(
            {
                "png_path": png_path.name,
                "png_sha256": _atomic_save_png(
                    png_path, contiguous, Image
                ),
            }
        )
    return record


def _capture_resume_is_valid(
    record: Mapping[str, Any],
    *,
    case: Mapping[str, Any],
    case_dir: Path,
) -> bool:
    """Recognize only a complete capture whose exact assets still exist."""

    if (
        record.get("schema_version") != CAPTURE_SCHEMA
        or record.get("protocol_id") != case.get("protocol_id")
        or record.get("case_id") != case.get("case_id")
        or record.get("status") != "complete"
        or record.get("case") != dict(case)
    ):
        return False
    counts = record.get("execution_counts")
    if not isinstance(counts, Mapping):
        return False
    expected_counts = {
        "reset": 1,
        "initial_state_restore": 1,
        "settle_actions": 20,
        "policy_queries": 0,
        "semantic_selector_calls": 0,
        "grounding_calls": 0,
        "point_cloud_filter_calls": 0,
        "mvee_calls": 0,
        "qp_calls": 0,
        "outcome_actions": 0,
    }
    if any(counts.get(key) != value for key, value in expected_counts.items()):
        return False
    assets = record.get("assets")
    if not isinstance(assets, Mapping):
        return False
    expected_assets = {
        "agentview_rgb": True,
        "backview_rgb": True,
        "agentview_depth": False,
        "backview_depth": False,
        "simulator_state": False,
    }
    for name, needs_png in expected_assets.items():
        asset = assets.get(name)
        if not isinstance(asset, Mapping):
            return False
        array_digest = asset.get("array_sha256")
        if (
            not isinstance(array_digest, str)
            or len(array_digest) != 64
            or not isinstance(asset.get("shape"), list)
            or not isinstance(asset.get("dtype"), str)
        ):
            return False
        for path_key, hash_key in (("npy_path", "npy_sha256"),):
            relative = Path(str(asset.get(path_key, "")))
            if (
                not relative.name
                or relative.is_absolute()
                or ".." in relative.parts
            ):
                return False
            path = case_dir / relative
            if (
                not path.is_file()
                or sha256_path(path) != asset.get(hash_key)
            ):
                return False
        if needs_png:
            relative = Path(str(asset.get("png_path", "")))
            path = case_dir / relative
            if (
                not relative.name
                or relative.is_absolute()
                or ".." in relative.parts
                or not path.is_file()
                or sha256_path(path) != asset.get("png_sha256")
            ):
                return False
    return (
        record.get("settled_agentview_array_sha256")
        == assets["agentview_rgb"]["array_sha256"]
    )


def _archive_prior_capture_artifacts(case_dir: Path) -> dict[str, str]:
    """Keep failed, partial, and explicitly overwritten captures."""

    names = (
        "capture.json",
        "capture_failure.json",
        "settled_agentview_rgb.npy",
        "settled_agentview_rgb.png",
        "settled_backview_rgb.npy",
        "settled_backview_rgb.png",
        "settled_agentview_depth.npy",
        "settled_backview_depth.npy",
        "settled_simulator_state.npy",
    )
    archived: dict[str, str] = {}
    archive_dir = case_dir / "attempts"
    for name in names:
        source = case_dir / name
        if not source.is_file():
            continue
        digest = sha256_path(source)
        target = archive_dir / f"{source.stem}-{digest}{source.suffix}"
        archive_dir.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if sha256_path(target) != digest:
                raise ProtocolError(
                    f"capture-attempt archive collision at {target}"
                )
            source.unlink()
        else:
            os.replace(source, target)
        archived[name] = str(target.relative_to(case_dir))
    return archived


def capture_case(
    *,
    case: dict[str, Any],
    repo_root: Path,
    output_root: Path,
    render_resolution: int,
    overwrite: bool,
) -> dict[str, Any]:
    if render_resolution != TABLE_RENDER_RESOLUTION:
        raise ProtocolError("Table 1 label capture requires render=1024")
    validate_case_row(case, repo_root)
    case_id = str(case["case_id"])
    case_dir = output_root / case_id
    capture_path = case_dir / "capture.json"
    case_dir.mkdir(parents=True, exist_ok=True)
    if capture_path.exists() and not overwrite:
        try:
            existing = json.loads(capture_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            existing = {}
        if _capture_resume_is_valid(
            existing, case=case, case_dir=case_dir
        ):
            return existing
    prior_attempt_artifacts = _archive_prior_capture_artifacts(case_dir)
    runtime: dict[str, Any] | None = None
    env: Any = None
    started = time.time()
    result: dict[str, Any] = {
        "schema_version": CAPTURE_SCHEMA,
        "case_id": case_id,
        "case_ordinal": case.get("case_ordinal"),
        "protocol_id": case.get("protocol_id"),
        "status": "apparatus_failure",
        "case": dict(case),
        "source": {
            "git": git_identity(repo_root),
            "required_upstream_commit": case.get("source_commit"),
        },
        "execution_counts": {
            "reset": 0,
            "initial_state_restore": 0,
            "settle_actions": 0,
            "policy_queries": 0,
            "semantic_selector_calls": 0,
            "grounding_calls": 0,
            "point_cloud_filter_calls": 0,
            "mvee_calls": 0,
            "qp_calls": 0,
            "outcome_actions": 0,
        },
        "timing": {"started_unix": started},
    }
    if prior_attempt_artifacts:
        result["prior_attempt_artifacts"] = prior_attempt_artifacts
    try:
        runtime = _capture_runtime()
        np = runtime["np"]
        # Match the released evaluator's ordering: its process-level NumPy
        # seed is set before OffScreenRenderEnv construction.  The constructor
        # performs a temporary randomized placement before the frozen full
        # simulator state is restored, and can fail if this seed is omitted.
        np.random.seed(int(case["environment_seed"]))
        env, task, observation, selected_initial_state = _build_environment(
            runtime,
            case,
            render_resolution=render_resolution,
        )
        result["execution_counts"]["reset"] = 1
        result["execution_counts"]["initial_state_restore"] = 1
        settle_actions = int(
            case.get("settle_actions", TABLE_SETTLE_ACTIONS)
        )
        if settle_actions != TABLE_SETTLE_ACTIONS:
            raise ProtocolError("label capture requires 20 settle actions")
        observation = _settle(env, observation, settle_actions)
        result["execution_counts"]["settle_actions"] = settle_actions

        agent_rgb = _processed_image(observation, "agentview_image")
        back_rgb = _processed_image(observation, "backview_image")
        agent_depth = _processed_image(observation, "agentview_depth")
        back_depth = _processed_image(observation, "backview_depth")
        simulator_state = np.asarray(
            env.sim.get_state().flatten(), dtype=float
        )
        assets = {
            "agentview_rgb": _asset_record(
                case_dir=case_dir,
                name="settled_agentview_rgb",
                array=agent_rgb,
                np=np,
                Image=runtime["Image"],
            ),
            "backview_rgb": _asset_record(
                case_dir=case_dir,
                name="settled_backview_rgb",
                array=back_rgb,
                np=np,
                Image=runtime["Image"],
            ),
            "agentview_depth": _asset_record(
                case_dir=case_dir,
                name="settled_agentview_depth",
                array=agent_depth,
                np=np,
                Image=None,
            ),
            "backview_depth": _asset_record(
                case_dir=case_dir,
                name="settled_backview_depth",
                array=back_depth,
                np=np,
                Image=None,
            ),
            "simulator_state": _asset_record(
                case_dir=case_dir,
                name="settled_simulator_state",
                array=simulator_state,
                np=np,
                Image=None,
            ),
        }
        result.update(
            {
                "status": "complete",
                "task_name": task.name,
                "task_instruction": task.language,
                "camera_convention": (
                    "released 180-degree preprocessing: [::-1, ::-1]"
                ),
                "settled_agentview_array_sha256": assets[
                    "agentview_rgb"
                ]["array_sha256"],
                "selected_initial_state_array_sha256": array_sha256(
                    selected_initial_state
                ),
                "assets": assets,
                "label_template": {
                    "schema_version": "vlsa_table1_codex_label.v1",
                    "case_id": case_id,
                    "settled_agentview_array_sha256": assets[
                        "agentview_rgb"
                    ]["array_sha256"],
                    "obstacle_label": (
                        "<one preregistered allowed label>"
                    ),
                    "reviewer": "codex",
                    "reviewed_at": "<ISO-8601 timestamp>",
                },
            }
        )
    except Exception as error:
        result.update(
            {
                "status": "apparatus_failure",
                "error": {
                    "type": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exc(),
                },
            }
        )
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as error:
                result["environment_close_error"] = (
                    f"{type(error).__name__}: {error}"
                )
        result["timing"]["finished_unix"] = time.time()
        result["timing"]["wall_seconds"] = (
            result["timing"]["finished_unix"] - started
        )
        result["capture_payload_sha256"] = sha256_bytes(
            canonical_json_bytes(
                {
                    key: value
                    for key, value in result.items()
                    if key != "capture_payload_sha256"
                }
            )
        )
        atomic_write_json(
            capture_path
            if result["status"] == "complete"
            else case_dir / "capture_failure.json",
            result,
        )
    return result


def _parse_int_list(values: Sequence[str]) -> list[int]:
    output: list[int] = []
    for value in values:
        for token in value.split(","):
            token = token.strip()
            if token:
                output.append(int(token))
    return output


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Capture settled SafeLIBERO images for Codex labels"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--case-ordinal", action="append", default=[])
    parser.add_argument("--render-resolution", type=int, default=1024)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--repo-root", type=Path, default=root)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    selected = select_cases(
        read_jsonl(args.manifest.resolve()),
        case_ids=args.case_id,
        ordinals=_parse_int_list(args.case_ordinal),
    )
    if not selected:
        raise ProtocolError("no cases selected")
    failures = 0
    for case in selected:
        result = capture_case(
            case=case,
            repo_root=args.repo_root.resolve(),
            output_root=args.output_dir.resolve(),
            render_resolution=args.render_resolution,
            overwrite=args.overwrite,
        )
        failures += int(result["status"] != "complete")
        print(
            json.dumps(
                {
                    "case_id": result["case_id"],
                    "status": result["status"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    print(
        json.dumps(
            {
                "selected_cases": len(selected),
                "complete": len(selected) - failures,
                "apparatus_failures": failures,
            },
            sort_keys=True,
        )
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
