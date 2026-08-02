#!/usr/bin/env python3
"""Run the one-case paired closed-loop AEGIS / Poisson-CBF canary.

The exact historical AEGIS / OSC prefix ends at action 179.  Each matched
joint-velocity suffix arm then queries pi0.5 live from its own observation,
applies a fresh released AEGIS translational correction, and executes through
the same adapter.  Only the treatment arm adds link-5/link-6 Poisson-CBF rows.
"""

from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import json
import math
import os
import platform
import struct
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from scripts import run_poisson_fast_feasibility as fast


class ClosedLoopRunnerError(RuntimeError):
    """Raised when the closed-loop apparatus cannot support interpretation."""


def _array_sha256(array: Any) -> str:
    """Reconstruct the released Table-1 dtype/shape/bytes array hash."""

    try:
        import numpy as np
    except ImportError:  # local structural tests intentionally omit NumPy
        rows = list(array)
        if not rows or any(
            isinstance(row, (str, bytes)) or not isinstance(row, Sequence)
            for row in rows
        ):
            raise ValueError("array hash fallback requires a nonempty matrix")
        width = len(rows[0])
        if width <= 0 or any(len(row) != width for row in rows):
            raise ValueError("array hash fallback requires a rectangular matrix")
        flattened = [float(item) for row in rows for item in row]
        if not all(math.isfinite(item) for item in flattened):
            raise ValueError("array hash fallback rejects nonfinite values")
        byte_order = "<" if sys.byteorder == "little" else ">"
        dtype = "%sf8" % byte_order
        shape = [len(rows), width]
        raw = struct.pack("%s%d" % (byte_order, len(flattened)) + "d", *flattened)
    else:
        value = np.ascontiguousarray(array)
        dtype = value.dtype.str
        shape = list(value.shape)
        raw = memoryview(value).cast("B")
    digest = hashlib.sha256()
    digest.update(b"vlsa-table1-array-v1\0")
    digest.update(
        fast._canonical({"dtype": dtype, "shape": shape})
    )
    digest.update(b"\0")
    digest.update(raw)
    return digest.hexdigest()


def _checkpoint_identity(contract: Mapping[str, Any]) -> Dict[str, Any]:
    """Bind the live server checkpoint to the frozen full-content receipt.

    Rehashing the multi-gigabyte checkpoint for every canary would dominate
    setup time.  The historical receipt already contains a verified full-tree
    hash; the current run cheaply reconstructs and exactly compares its
    per-file filesystem identity before trusting that receipt.
    """

    from scripts.validate_aegis_assets import (
        pi05_checkpoint_filesystem_identity,
        validate_pi05_filesystem_identity_record,
    )

    receipt_path = Path(str(contract.get("checkpoint_receipt")))
    checkpoint_path = Path(str(contract.get("checkpoint")))
    if (
        not receipt_path.is_absolute()
        or receipt_path.is_symlink()
        or not receipt_path.is_file()
        or fast._file_sha256(receipt_path)
        != contract.get("checkpoint_receipt_sha256")
    ):
        raise ClosedLoopRunnerError("pi0.5 checkpoint receipt differs")
    receipt = fast._json(receipt_path, "pi0.5 checkpoint receipt")
    payload = {
        key: value
        for key, value in receipt.items()
        if key != "receipt_payload_sha256"
    }
    checkpoint = receipt.get("checkpoint")
    if (
        receipt.get("schema_version")
        != contract.get("checkpoint_receipt_schema_version")
        or receipt.get("status") != "passed"
        or receipt.get("receipt_payload_sha256")
        != fast._sha256(fast._canonical(payload))
        or not isinstance(checkpoint, Mapping)
        or checkpoint.get("full_content_hash_verified") is not True
        or checkpoint.get("full_content_tree_sha256")
        != contract.get("checkpoint_tree_sha256")
    ):
        raise ClosedLoopRunnerError("pi0.5 checkpoint receipt is not authoritative")
    expected_filesystem = validate_pi05_filesystem_identity_record(
        checkpoint.get("filesystem_identity")
    )
    current_filesystem = pi05_checkpoint_filesystem_identity(
        checkpoint_path,
        tuple(
            str(row["relative_path"])
            for row in expected_filesystem["files"]
        ),
    )
    if current_filesystem != expected_filesystem:
        raise ClosedLoopRunnerError(
            "pi0.5 checkpoint filesystem changed after its full-content receipt"
        )
    return {
        "checkpoint_path": str(checkpoint_path),
        "full_content_tree_sha256": str(
            checkpoint["full_content_tree_sha256"]
        ),
        "receipt_file_sha256": fast._file_sha256(receipt_path),
        "receipt_payload_sha256": str(receipt["receipt_payload_sha256"]),
        "filesystem_identity_sha256": str(
            current_filesystem["identity_sha256"]
        ),
        "current_filesystem_identity_matches_receipt": True,
    }


class RolloutVideo:
    """Stream real simulator frames and atomically publish one arm video."""

    def __init__(
        self,
        *,
        evaluator: Any,
        imageio: Any,
        run_root: Path,
        arm_slug: str,
        fps: int,
    ) -> None:
        self.evaluator = evaluator
        self.imageio = imageio
        self.run_root = run_root
        self.arm_slug = arm_slug
        self.fps = int(fps)
        self.directory = run_root / "videos" / arm_slug
        self.directory.mkdir(parents=True, exist_ok=False)
        self.partial = self.directory / "episode.partial.mp4"
        self.final = self.directory / "episode.mp4"
        self.branch_frame = self.directory / "branch_agentview.npy"
        self.terminal_frame = self.directory / "terminal_agentview.npy"
        if self.partial.exists() or self.final.exists():
            raise ClosedLoopRunnerError("video target already exists")
        self.writer = imageio.get_writer(
            str(self.partial), fps=self.fps, codec="libx264", macro_block_size=None
        )
        self.snapshots: List[Dict[str, Any]] = []
        self.first_source_frame = None
        self.last_source_frame = None
        self.closed = False

    def __call__(
        self,
        observation: Mapping[str, Any],
        *,
        snapshot_kind: str,
        local_action_index: Any,
        source_action_index: int,
    ) -> None:
        if self.closed:
            raise ClosedLoopRunnerError("cannot append to a closed rollout video")
        frame = self.evaluator._processed_image(observation, "agentview_image")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ClosedLoopRunnerError("rollout frame is not RGB")
        self.writer.append_data(frame)
        if self.first_source_frame is None:
            self.first_source_frame = frame.copy()
        self.last_source_frame = frame.copy()
        self.snapshots.append(
            {
                "frame_index": len(self.snapshots),
                "snapshot_kind": str(snapshot_kind),
                "local_action_index": local_action_index,
                "source_action_index": int(source_action_index),
                "source_array_sha256": self.evaluator.array_sha256(frame),
                "height": int(frame.shape[0]),
                "width": int(frame.shape[1]),
            }
        )

    def close(self) -> Dict[str, Any]:
        if not self.closed:
            self.writer.close()
            self.closed = True
            os.replace(str(self.partial), str(self.final))
        if self.first_source_frame is None or self.last_source_frame is None:
            raise ClosedLoopRunnerError("rollout video has no source frames")
        import numpy as np

        for path, value in (
            (self.branch_frame, self.first_source_frame),
            (self.terminal_frame, self.last_source_frame),
        ):
            temporary = Path(str(path) + ".partial")
            with temporary.open("wb") as stream:
                np.save(stream, value, allow_pickle=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(str(temporary), str(path))
        decoded_hashes: List[str] = []
        decoded_shapes: List[List[int]] = []
        decoded_first = None
        decoded_last = None
        reader = self.imageio.get_reader(str(self.final))
        try:
            for frame in reader:
                if decoded_first is None:
                    decoded_first = frame.copy()
                decoded_last = frame.copy()
                decoded_hashes.append(self.evaluator.array_sha256(frame))
                decoded_shapes.append([int(value) for value in frame.shape])
        finally:
            reader.close()
        if len(decoded_hashes) != len(self.snapshots):
            raise ClosedLoopRunnerError("decoded video frame count differs")
        if any(shape != [1024, 1024, 3] for shape in decoded_shapes):
            raise ClosedLoopRunnerError("decoded rollout video dimensions differ")
        endpoint_diagnostics = {}
        for label, source, decoded in (
            ("branch", self.first_source_frame, decoded_first),
            ("terminal", self.last_source_frame, decoded_last),
        ):
            difference = np.asarray(decoded, dtype=np.float64) - np.asarray(
                source, dtype=np.float64
            )
            mae = float(np.mean(np.abs(difference)))
            mse = float(np.mean(np.square(difference)))
            psnr = float("inf") if mse == 0.0 else 20.0 * math.log10(255.0 / math.sqrt(mse))
            if mae > 12.0 or psnr < 20.0:
                raise ClosedLoopRunnerError(
                    "%s decoded video endpoint differs from source" % label
                )
            endpoint_diagnostics[label] = {"mae": mae, "psnr_db": psnr}
        return {
            "arm_slug": self.arm_slug,
            "path": str(self.final.relative_to(self.run_root)),
            "sha256": fast._file_sha256(self.final),
            "byte_count": int(self.final.stat().st_size),
            "fps": self.fps,
            "codec": "libx264",
            "coverage_scope": "branch_boundary_through_live_suffix_terminal",
            "source_action_index_start": int(
                self.snapshots[0]["source_action_index"]
            ),
            "source_action_index_end": int(
                self.snapshots[-1]["source_action_index"]
            ),
            "sampling_semantics": (
                "branch_frame_plus_one_frame_after_each_completed_high_level_"
                "action_plus_partial_contact_terminal_when_applicable"
            ),
            "playback_not_wall_clock": True,
            "frame_count": len(self.snapshots),
            "decoded_frame_count": len(decoded_hashes),
            "decoded_frame_shape": [1024, 1024, 3],
            "decoded_successfully": True,
            "real_simulation_frames": True,
            "two_dimensional_safety_overlay": False,
            "branch_lossless": {
                "path": str(self.branch_frame.relative_to(self.run_root)),
                "sha256": fast._file_sha256(self.branch_frame),
                "array_sha256": self.snapshots[0]["source_array_sha256"],
            },
            "terminal_lossless": {
                "path": str(self.terminal_frame.relative_to(self.run_root)),
                "sha256": fast._file_sha256(self.terminal_frame),
                "array_sha256": self.snapshots[-1]["source_array_sha256"],
            },
            "decoded_endpoint_diagnostics": endpoint_diagnostics,
            "snapshot_trace": list(self.snapshots),
            "decoded_frame_sequence_sha256": fast._sha256(
                fast._canonical(decoded_hashes)
            ),
        }


class LiveAegisPolicy:
    """Produce fresh AEGIS actions from one arm's own live observations."""

    def __init__(
        self,
        *,
        evaluator: Any,
        runtime: Mapping[str, Any],
        client: Any,
        case: Mapping[str, Any],
        task_description: str,
        historical_result: Mapping[str, Any],
        first_query_index: int,
        replan_steps: int,
        model_action_horizon: int,
        expected_first_chunk_sha256: str,
        arm_name: str,
    ) -> None:
        np = runtime["np"]
        perception = historical_result.get("perception")
        actions = historical_result.get("actions")
        if not isinstance(perception, Mapping) or perception.get("status") != "ready":
            raise ClosedLoopRunnerError("historical AEGIS geometry is unavailable")
        if not isinstance(actions, Sequence) or len(actions) != 237:
            raise ClosedLoopRunnerError("historical AEGIS action ledger differs")
        action_179 = actions[179]
        qp_179 = action_179.get("qp") if isinstance(action_179, Mapping) else None
        if not isinstance(qp_179, Mapping) or qp_179.get("status") != "solved":
            raise ClosedLoopRunnerError("historical action-179 AEGIS state is absent")
        z_after = np.asarray(qp_179.get("z_after"), dtype=np.float64)
        if z_after.shape != (3,) or not np.all(np.isfinite(z_after)):
            raise ClosedLoopRunnerError("historical action-179 z_after is invalid")
        if not math.isclose(float(np.linalg.norm(z_after)), 1.0, rel_tol=0.0, abs_tol=1e-10):
            raise ClosedLoopRunnerError("historical action-179 z_after is not unit")
        self.evaluator = evaluator
        self.runtime = runtime
        self.client = client
        self.case = case
        self.task_description = str(task_description)
        self.first_query_index = int(first_query_index)
        self.replan_steps = int(replan_steps)
        self.model_action_horizon = int(model_action_horizon)
        self.expected_first_chunk_sha256 = str(expected_first_chunk_sha256)
        self.arm_name = str(arm_name)
        self.geometry = {
            "enabled": True,
            "p2": np.asarray(perception.get("mvee_center"), dtype=np.float64).copy(),
            "R2": np.asarray(perception.get("mvee_rotation"), dtype=np.float64).copy(),
            "Q2_diag": np.asarray(perception.get("mvee_semiaxes"), dtype=np.float64).copy(),
            "z_fixed": z_after.copy(),
        }
        if (
            self.geometry["p2"].shape != (3,)
            or self.geometry["R2"].shape != (3, 3)
            or self.geometry["Q2_diag"].shape != (3,)
            or not all(np.all(np.isfinite(value)) for value in self.geometry.values() if not isinstance(value, bool))
        ):
            raise ClosedLoopRunnerError("historical AEGIS MVEE geometry is invalid")
        self.initial_z_fixed = z_after.tolist()
        self.expected_first_aegis_action = np.asarray(
            actions[180].get("executed"), dtype=np.float64
        )
        action_180_qp = actions[180].get("qp")
        action_180_context = (
            action_180_qp.get("context")
            if isinstance(action_180_qp, Mapping)
            else None
        )
        if not isinstance(action_180_context, Mapping):
            raise ClosedLoopRunnerError(
                "historical action-180 AEGIS input context is unavailable"
            )
        self.expected_first_aegis_inputs = {
            key: copy.deepcopy(action_180_context.get(key))
            for key in (
                "p1",
                "R1",
                "q1_diag",
                "p2",
                "R2",
                "Q2_diag",
                "z_before",
                "nominal_translational",
            )
        }
        self.expected_first_aegis_outputs = {
            key: copy.deepcopy(action_180_qp.get(key))
            for key in (
                "barrier_h",
                "constraint_lhs",
                "u_solution",
                "z_after",
            )
        }
        if self.expected_first_aegis_action.shape != (7,):
            raise ClosedLoopRunnerError(
                "historical action-180 AEGIS authority is invalid"
            )
        self.action_plan: collections.deque = collections.deque()
        self.action_plan_query_indexes: collections.deque = collections.deque()
        self.action_plan_offsets: collections.deque = collections.deque()
        self.policy_queries: List[Dict[str, Any]] = []
        self.action_trace: List[Dict[str, Any]] = []
        self.first_aegis_input_binding_matches_historical = False
        self.first_aegis_internal_output_matches_historical = False

    def _observation_record(self, env: Any, observation: Mapping[str, Any]) -> Dict[str, Any]:
        from scripts.run_poisson_shadow_parity import _observation_sha256

        np = self.runtime["np"]
        policy_input = self.evaluator._policy_observation(
            self.runtime,
            observation,
            task_description=self.task_description,
            resize_size=self.evaluator.TABLE_POLICY_RESIZE,
            rng_seed=0,
        )
        record = {
            "native_observation_sha256": _observation_sha256(
                observation, self.evaluator, np
            ),
            "official_integration_state_raw_bytes_sha256": fast._sha256(
                fast._official_state(env.sim).tobytes()
            ),
            "agentview_policy_array_sha256": self.evaluator.array_sha256(
                policy_input["observation/image"]
            ),
            "wrist_policy_array_sha256": self.evaluator.array_sha256(
                policy_input["observation/wrist_image"]
            ),
            "policy_state_array_sha256": self.evaluator.array_sha256(
                policy_input["observation/state"]
            ),
            "policy_state": np.asarray(
                policy_input["observation/state"], dtype=np.float64
            ).tolist(),
            "prompt": str(policy_input["prompt"]),
        }
        record["policy_input_fingerprint_sha256"] = fast._sha256(
            fast._canonical(record)
        )
        return record

    def __call__(
        self,
        *,
        env: Any,
        observation: Mapping[str, Any],
        local_action_index: int,
        source_action_index: int,
    ) -> Sequence[float]:
        np = self.runtime["np"]
        expected_source = 180 + int(local_action_index)
        if int(source_action_index) != expected_source:
            raise ClosedLoopRunnerError("live source-action cadence differs")
        observation_record = self._observation_record(env, observation)
        if not self.action_plan:
            query_index = self.first_query_index + len(self.policy_queries)
            seed = self.evaluator.query_seed(
                int(self.case["policy_noise_seed"]), query_index
            )
            policy_input = self.evaluator._policy_observation(
                self.runtime,
                observation,
                task_description=self.task_description,
                resize_size=self.evaluator.TABLE_POLICY_RESIZE,
                rng_seed=seed,
            )
            query_started = time.perf_counter()
            response = self.client.infer(policy_input)
            query_elapsed = time.perf_counter() - query_started
            if "actions" not in response:
                raise ClosedLoopRunnerError("pi0.5 response has no actions")
            chunk = np.asarray(response["actions"], dtype=np.float64)
            if (
                chunk.shape != (self.model_action_horizon, 7)
                or not np.all(np.isfinite(chunk))
            ):
                raise ClosedLoopRunnerError("pi0.5 action chunk is invalid")
            chunk_sha256 = self.evaluator.array_sha256(chunk)
            if not self.policy_queries and chunk_sha256 != self.expected_first_chunk_sha256:
                raise ClosedLoopRunnerError(
                    "first live pi0.5 chunk does not reproduce historical query 36"
                )
            for offset in range(self.replan_steps):
                self.action_plan.append(chunk[offset].copy())
                self.action_plan_query_indexes.append(query_index)
                self.action_plan_offsets.append(offset)
            self.policy_queries.append(
                {
                    "arm": self.arm_name,
                    "query_index": query_index,
                    "rng_seed": seed,
                    "source_action_index": int(source_action_index),
                    "local_action_index": int(local_action_index),
                    **observation_record,
                    "returned_action_shape": list(chunk.shape),
                    "returned_actions": chunk.tolist(),
                    "returned_actions_sha256": chunk_sha256,
                    "elapsed_seconds": float(query_elapsed),
                    "server_timing": response.get("server_timing"),
                }
            )
        raw = np.asarray(self.action_plan.popleft(), dtype=np.float64)
        query_index = int(self.action_plan_query_indexes.popleft())
        chunk_offset = int(self.action_plan_offsets.popleft())
        nominal = self.evaluator.translational_action(raw)
        proxy = self.evaluator._eef_proxy(self.runtime, observation)
        z_before = np.asarray(self.geometry["z_fixed"], dtype=np.float64).copy()
        executed, qp_record = self.evaluator._aegis_action(
            self.runtime,
            nominal_translational=nominal,
            proxy=proxy,
            geometry=self.geometry,
            q1_diag=np.asarray([0.06, 0.12, 0.11], dtype=np.float64),
            diagnostics_enabled=True,
        )
        if int(local_action_index) == 0 and not np.allclose(
            np.asarray(executed, dtype=np.float64),
            self.expected_first_aegis_action,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ClosedLoopRunnerError(
                "fresh action-180 AEGIS output differs from historical authority"
            )
        if int(local_action_index) == 0:
            observed_context = qp_record.get("context")
            if not isinstance(observed_context, Mapping):
                raise ClosedLoopRunnerError(
                    "fresh action-180 AEGIS input context is unavailable"
                )
            for key, expected in self.expected_first_aegis_inputs.items():
                expected_array = np.asarray(expected, dtype=np.float64)
                observed_array = np.asarray(
                    observed_context.get(key), dtype=np.float64
                )
                if (
                    observed_array.shape != expected_array.shape
                    or not np.allclose(
                        observed_array,
                        expected_array,
                        rtol=0.0,
                        atol=1e-12,
                    )
                ):
                    raise ClosedLoopRunnerError(
                        "fresh action-180 AEGIS %s differs from historical authority"
                        % key
                    )
            self.first_aegis_input_binding_matches_historical = True
            for key, expected in self.expected_first_aegis_outputs.items():
                expected_array = np.asarray(expected, dtype=np.float64)
                observed_array = np.asarray(qp_record.get(key), dtype=np.float64)
                if (
                    observed_array.shape != expected_array.shape
                    or not np.allclose(
                        observed_array,
                        expected_array,
                        rtol=0.0,
                        atol=1e-12,
                    )
                ):
                    raise ClosedLoopRunnerError(
                        "fresh action-180 AEGIS %s differs from historical authority"
                        % key
                    )
            self.first_aegis_internal_output_matches_historical = True
        z_after = np.asarray(self.geometry["z_fixed"], dtype=np.float64).copy()
        self.action_trace.append(
            {
                "arm": self.arm_name,
                "local_action_index": int(local_action_index),
                "source_action_index": int(source_action_index),
                "query_index": query_index,
                "query_chunk_offset": chunk_offset,
                **observation_record,
                "nominal_raw": raw.tolist(),
                "nominal_translational": list(nominal),
                "aegis_executed": list(executed),
                "aegis_correction_l2": float(
                    np.linalg.norm(
                        np.asarray(executed, dtype=np.float64)
                        - np.asarray(nominal, dtype=np.float64)
                    )
                ),
                "aegis_z_before": z_before.tolist(),
                "aegis_z_after": z_after.tolist(),
                "aegis_qp": qp_record,
            }
        )
        return list(executed)

    def record(self) -> Dict[str, Any]:
        return {
            "arm": self.arm_name,
            "source": "live_pi05_queries_from_this_arms_own_observations",
            "recorded_suffix_actions_executed": False,
            "first_aegis_input_binding_matches_historical": bool(
                self.first_aegis_input_binding_matches_historical
            ),
            "first_aegis_internal_output_matches_historical": bool(
                self.first_aegis_internal_output_matches_historical
            ),
            "historical_action_180_aegis_inputs": copy.deepcopy(
                self.expected_first_aegis_inputs
            ),
            "historical_action_180_aegis_outputs": copy.deepcopy(
                self.expected_first_aegis_outputs
            ),
            "first_query_index": self.first_query_index,
            "initial_aegis_z_fixed_from_historical_action_179": list(
                self.initial_z_fixed
            ),
            "policy_queries": list(self.policy_queries),
            "high_level_action_trace": list(self.action_trace),
            "live_aegis_action_sequence_sha256": fast._sha256(
                fast._canonical(
                    [row["aegis_executed"] for row in self.action_trace]
                )
            ),
        }


def _provider_contract(
    provider: Mapping[str, Any],
    *,
    expected_actions: int,
    expected_queries: int,
    policy_noise_seed: int = 2026691220,
) -> bool:
    queries = provider.get("policy_queries")
    actions = provider.get("high_level_action_trace")
    if not isinstance(queries, Sequence) or not isinstance(actions, Sequence):
        return False
    if len(actions) != expected_actions or len(queries) != expected_queries:
        return False
    query_map = {int(row["query_index"]): row for row in queries}
    if sorted(query_map) != list(range(36, 36 + expected_queries)):
        return False
    for query_offset, query_index in enumerate(range(36, 36 + expected_queries)):
        query = query_map[query_index]
        raw_returned = query.get("returned_actions")
        valid_returned = bool(
            isinstance(raw_returned, Sequence)
            and not isinstance(raw_returned, (str, bytes))
            and len(raw_returned) == 10
            and all(
                isinstance(row, Sequence)
                and not isinstance(row, (str, bytes))
                and len(row) == 7
                and all(
                    not isinstance(value, bool)
                    and isinstance(value, (int, float))
                    and math.isfinite(float(value))
                    for value in row
                )
                for row in raw_returned
            )
        )
        expected_local = query_offset * 5
        if (
            query.get("query_index") != query_index
            or query.get("local_action_index") != expected_local
            or query.get("source_action_index") != 180 + expected_local
            or query.get("rng_seed")
            != policy_noise_seed + query_index
            or query.get("returned_action_shape") != [10, 7]
            or not valid_returned
            or query.get("returned_actions_sha256")
            != (_array_sha256(raw_returned) if valid_returned else None)
        ):
            return False
    for local_index, row in enumerate(actions):
        query_index = 36 + local_index // 5
        offset = local_index % 5
        query = query_map.get(query_index)
        if query is None:
            return False
        returned = query.get("returned_actions")
        query_observation_mismatch = bool(
            offset == 0
            and row.get("native_observation_sha256")
            != query.get("native_observation_sha256")
        )
        if (
            row.get("local_action_index") != local_index
            or row.get("source_action_index") != 180 + local_index
            or row.get("query_index") != query_index
            or row.get("query_chunk_offset") != offset
            or not isinstance(returned, Sequence)
            or offset >= len(returned)
            or fast._canonical(row.get("nominal_raw"))
            != fast._canonical(returned[offset])
            or query_observation_mismatch
        ):
            return False
    return bool(
        provider.get("recorded_suffix_actions_executed") is False
        and provider.get("first_query_index") == 36
        and provider.get("source")
        == "live_pi05_queries_from_this_arms_own_observations"
    )


def _feedback_metrics(
    baseline: Mapping[str, Any],
    psf: Mapping[str, Any],
    expected_first_chunk_sha256: str,
) -> Dict[str, Any]:
    baseline_actions = baseline["high_level_action_trace"]
    psf_actions = psf["high_level_action_trace"]
    common_actions = min(len(baseline_actions), len(psf_actions))
    divergence_local = None
    for index in range(common_actions):
        if (
            baseline_actions[index]["official_integration_state_raw_bytes_sha256"]
            != psf_actions[index]["official_integration_state_raw_bytes_sha256"]
        ):
            divergence_local = index
            break
    baseline_queries = {int(row["query_index"]): row for row in baseline["policy_queries"]}
    psf_queries = {int(row["query_index"]): row for row in psf["policy_queries"]}
    first_baseline = baseline_queries.get(36, {})
    first_psf = psf_queries.get(36, {})
    first_identical = bool(
        first_baseline.get("policy_input_fingerprint_sha256")
        == first_psf.get("policy_input_fingerprint_sha256")
        and first_baseline.get("rng_seed")
        == first_psf.get("rng_seed")
        == 2026691256
        and first_baseline.get("source_action_index")
        == first_psf.get("source_action_index")
        == 180
        and first_baseline.get("local_action_index")
        == first_psf.get("local_action_index")
        == 0
        and first_baseline.get("returned_actions_sha256")
        == first_psf.get("returned_actions_sha256")
        == expected_first_chunk_sha256
    )
    scheduled_query_indexes_after_divergence: List[int] = []
    divergent_query_indexes: List[int] = []
    if divergence_local is not None:
        divergence_source = 180 + divergence_local
        for query_index in sorted(set(baseline_queries).intersection(psf_queries)):
            b = baseline_queries[query_index]
            p = psf_queries[query_index]
            if int(b["source_action_index"]) <= divergence_source:
                continue
            scheduled_query_indexes_after_divergence.append(query_index)
            if b["policy_input_fingerprint_sha256"] != p["policy_input_fingerprint_sha256"]:
                divergent_query_indexes.append(query_index)
    return {
        "first_divergent_action_input_local_index": divergence_local,
        "first_divergent_action_input_source_index": (
            None if divergence_local is None else 180 + divergence_local
        ),
        "first_live_query_identical": first_identical,
        "post_divergence_query_indexes_with_distinct_policy_inputs": (
            divergent_query_indexes
        ),
        "scheduled_query_indexes_after_divergence": (
            scheduled_query_indexes_after_divergence
        ),
        "post_divergence_policy_inputs_differ": bool(divergent_query_indexes),
        "post_divergence_own_observations_used": bool(
            divergence_local is not None and scheduled_query_indexes_after_divergence
        ),
    }


def _own_observation_chain_valid(
    provider: Mapping[str, Any], arm: Mapping[str, Any]
) -> bool:
    queries = provider.get("policy_queries")
    actions = provider.get("high_level_action_trace")
    task = arm.get("task")
    if (
        not isinstance(queries, Sequence)
        or not isinstance(actions, Sequence)
        or not isinstance(task, Mapping)
    ):
        return False
    returned = task.get("returned_observation_sha256_ledger")
    if not isinstance(returned, Sequence):
        return False
    for action in actions:
        local_index = action.get("local_action_index")
        if isinstance(local_index, bool) or not isinstance(local_index, int):
            return False
        if local_index == 0:
            continue
        preceding_index = local_index - 1
        if preceding_index >= len(returned):
            return False
        if action.get("native_observation_sha256") != returned[preceding_index]:
            return False
    for query in queries:
        local_index = query.get("local_action_index")
        if isinstance(local_index, bool) or not isinstance(local_index, int):
            return False
        if local_index == 0:
            continue
        preceding_index = local_index - 1
        if preceding_index >= len(returned):
            return False
        if query.get("native_observation_sha256") != returned[preceding_index]:
            return False
    return bool(queries)


def _aegis_state_and_command_chain_valid(
    provider: Mapping[str, Any], arm: Mapping[str, Any]
) -> bool:
    import numpy as np

    actions = provider.get("high_level_action_trace")
    commands = arm.get("command_trace")
    initial_z = provider.get("initial_aegis_z_fixed_from_historical_action_179")
    if (
        not isinstance(actions, Sequence)
        or not actions
        or not isinstance(commands, Sequence)
        or not isinstance(initial_z, Sequence)
    ):
        return False
    previous_z = np.asarray(initial_z, dtype=np.float64)
    by_source = {int(row["source_action_index"]): row for row in actions}
    for row in actions:
        before = np.asarray(row.get("aegis_z_before"), dtype=np.float64)
        after = np.asarray(row.get("aegis_z_after"), dtype=np.float64)
        qp = row.get("aegis_qp")
        context = qp.get("context") if isinstance(qp, Mapping) else None
        if (
            before.shape != (3,)
            or after.shape != (3,)
            or not np.allclose(before, previous_z, rtol=0.0, atol=1e-12)
            or not math.isclose(
                float(np.linalg.norm(after)), 1.0, rel_tol=0.0, abs_tol=1e-10
            )
            or not isinstance(context, Mapping)
            or qp.get("status") != "solved"
            or not np.allclose(
                np.asarray(qp.get("z_before"), dtype=np.float64),
                before,
                rtol=0.0,
                atol=1e-12,
            )
            or not np.allclose(
                np.asarray(qp.get("z_after"), dtype=np.float64),
                after,
                rtol=0.0,
                atol=1e-12,
            )
            or not np.allclose(
                np.asarray(context.get("z_before"), dtype=np.float64),
                before,
                rtol=0.0,
                atol=1e-12,
            )
            or not np.allclose(
                np.asarray(context.get("z_after"), dtype=np.float64),
                after,
                rtol=0.0,
                atol=1e-12,
            )
            or not np.allclose(
                np.asarray(context.get("q1_diag"), dtype=np.float64),
                np.asarray([0.06, 0.12, 0.11], dtype=np.float64),
                rtol=0.0,
                atol=1e-12,
            )
            or not np.allclose(
                np.asarray(context.get("nominal_translational"), dtype=np.float64),
                np.asarray(row.get("nominal_translational"), dtype=np.float64),
                rtol=0.0,
                atol=1e-12,
            )
            or not np.allclose(
                np.asarray(context.get("executed_action"), dtype=np.float64),
                np.asarray(row.get("aegis_executed"), dtype=np.float64),
                rtol=0.0,
                atol=1e-12,
            )
        ):
            return False
        previous_z = after
    for command in commands:
        source = int(command["source_action_index"])
        action = by_source.get(source)
        if action is None or not np.allclose(
            np.asarray(command.get("source_action"), dtype=np.float64),
            np.asarray(action.get("aegis_executed"), dtype=np.float64),
            rtol=0.0,
            atol=1e-12,
        ):
            return False
    return True


def _video_complete(video: Mapping[str, Any], completed_actions: int, contact: bool) -> bool:
    expected = completed_actions + (2 if contact else 1)
    return bool(
        video.get("decoded_successfully") is True
        and video.get("real_simulation_frames") is True
        and video.get("two_dimensional_safety_overlay") is False
        and video.get("coverage_scope")
        == "branch_boundary_through_live_suffix_terminal"
        and video.get("playback_not_wall_clock") is True
        and video.get("source_action_index_start") == 179
        and video.get("source_action_index_end")
        == 179 + completed_actions + (1 if contact else 0)
        and video.get("frame_count") == expected
        and video.get("decoded_frame_count") == expected
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--protocol", type=Path, default=Path("configs/vlsa_poisson_closed_loop_canary.v1.json"))
    parser.add_argument("--manifest", type=Path, default=Path("manifests/vlsa_poisson_link56_aegis_car_109.v1.jsonl"))
    parser.add_argument("--historical-result-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--case-id", default="vlsa-t1-goal-ii-t0-e05")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", required=True, type=int)
    arguments = parser.parse_args()

    root = arguments.repo_root.resolve()
    os.chdir(str(root))
    output = None
    result_path = None
    source_env = None
    started = time.time()
    provenance: Dict[str, Any] = {}
    arm_evidence: Dict[str, Any] = {}
    video_evidence: Dict[str, Any] = {}
    providers: Dict[str, Any] = {}
    assumption_evidence: Dict[str, Any] = {}
    try:
        import numpy as np
        import main.evaluate_safelibero_aegis as evaluator
        from main.poisson_fullbody.closed_loop_canary import (
            BASELINE_ARM,
            CASE_ID,
            PSF_ARM,
            RESULT_SCHEMA,
            SOURCE_ARM,
            classify_closed_loop_canary,
            validate_closed_loop_canary_protocol,
        )
        from main.poisson_fullbody.contracts import publish_hashed_json
        from main.poisson_fullbody.feasibility_protocol import load_feasibility_protocol
        from main.poisson_fullbody.field_bundle import build_static_field_bundle
        from main.poisson_fullbody.measurement import clone_forwarded_state, resolve_collision_geom_sets
        from main.poisson_fullbody.robot_samples import validate_rigid_roundtrip
        from main.poisson_fullbody.shadow_replay import load_historical_action_replay
        from main.poisson_fullbody.surface_sampling import build_robot_collision_samples, validate_robot_sample_evidence
        from scripts.run_poisson_shadow_parity import _check_step, _prepare_environment

        if arguments.case_id != CASE_ID:
            raise ClosedLoopRunnerError("closed-loop runner is frozen to %s" % CASE_ID)
        output = fast._new_output(arguments.output_root.resolve(), arguments.run_id)
        result_path = output / "result.json"
        protocol_path = arguments.protocol if arguments.protocol.is_absolute() else root / arguments.protocol
        manifest_path = arguments.manifest if arguments.manifest.is_absolute() else root / arguments.manifest
        protocol = fast._json(protocol_path.resolve(), "closed-loop protocol")
        derived = validate_closed_loop_canary_protocol(protocol)
        checkpoint_identity = _checkpoint_identity(protocol["online_policy"])
        case, case_row_sha256 = fast._manifest_case(manifest_path.resolve(), arguments.case_id)
        if fast._file_sha256(manifest_path.resolve()) != protocol["selection"]["manifest_file_sha256"]:
            raise ClosedLoopRunnerError("manifest raw hash differs")
        selection_path = root / protocol["selection"]["relative_path"]
        if fast._file_sha256(selection_path) != protocol["selection"]["raw_file_sha256"]:
            raise ClosedLoopRunnerError("selection config raw hash differs")
        if case.get("protocol_config_sha256") != protocol["selection"]["raw_file_sha256"]:
            raise ClosedLoopRunnerError("case-to-selection binding differs")
        runtime_path = root / protocol["runtime"]["relative_path"]
        if fast._file_sha256(runtime_path) != protocol["runtime"]["raw_file_sha256"]:
            raise ClosedLoopRunnerError("runtime config raw hash differs")
        runtime_protocol, runtime_hashes = load_feasibility_protocol(
            runtime_path,
            expected_protocol_sha256=protocol["runtime"]["semantic_sha256"],
        )
        if runtime_hashes.parameter_block_sha256 != protocol["runtime"]["parameter_block_sha256"]:
            raise ClosedLoopRunnerError("runtime parameter block differs")

        historical_path = fast._historical_path(arguments.historical_result_root.resolve(), case)
        replay = load_historical_action_replay(
            historical_path,
            expected_case_id=arguments.case_id,
            expected_arm=SOURCE_ARM,
        )
        source_contract = protocol["source"]
        if (
            replay.result_file_sha256 != source_contract["historical_result_file_sha256"]
            or replay.result_payload_sha256 != source_contract["historical_result_payload_sha256"]
            or replay.executed_sequence_sha256 != source_contract["historical_executed_action_sequence_sha256"]
            or len(replay.actions) != source_contract["historical_executed_action_count"]
        ):
            raise ClosedLoopRunnerError("historical AEGIS result differs")
        historical_result = fast._json(historical_path, "historical AEGIS result")
        historical_suffix_reference_sha256 = fast._sha256(
            fast._canonical([list(action) for action in replay.actions[180:237]])
        )
        source = fast._git(root)
        allocation = fast._allocation()
        provenance = {
            "source": source,
            "allocation": allocation,
            "python": platform.python_version(),
            "run_id": arguments.run_id,
            "case_id": arguments.case_id,
            "case_row_sha256": case_row_sha256,
            "protocol_file_sha256": fast._file_sha256(protocol_path.resolve()),
            "historical_result_file_sha256": replay.result_file_sha256,
            "historical_result_payload_sha256": replay.result_payload_sha256,
            "historical_executed_action_sequence_sha256": replay.executed_sequence_sha256,
            "historical_suffix_reference_sha256_never_executed": historical_suffix_reference_sha256,
            "policy_checkpoint_tree_sha256": protocol["online_policy"]["checkpoint_tree_sha256"],
            "policy_checkpoint_receipt_sha256": protocol["online_policy"]["checkpoint_receipt_sha256"],
            "policy_checkpoint_identity": checkpoint_identity,
            "execution": "live_pi05_feedback_after_exact_shared_prefix",
        }

        runtime = evaluator._runtime_imports(include_aegis=True)
        source_env, task, observation, goal_atoms, previous_goal = _prepare_environment(
            evaluator, runtime, case, replay
        )
        obstacle_name, _ = evaluator._active_obstacle(source_env, observation)
        if obstacle_name != protocol["case"]["selected_obstacle_name"] or obstacle_name != case["active_obstacle_name"]:
            raise ClosedLoopRunnerError("selected obstacle differs after settling")
        authority = evaluator._contact_model_authority(source_env, obstacle_name)
        robot_root = fast._body_id(source_env.sim.model, source_env.robots[0].robot_model.root_body)
        link_ids = tuple(fast._body_id(source_env.sim.model, name) for name in protocol["case"]["protected_robot_body_names"])
        raw_model, raw_data = fast._raw_model_data(source_env.sim)
        resolved = resolve_collision_geom_sets(
            raw_model,
            robot_root_body_ids=(robot_root,),
            obstacle_root_body_ids=(int(authority["active_obstacle_root_body_id"]),),
            link56_body_ids=link_ids,
        )
        settled_official = fast._official_state(source_env.sim)
        bundle = build_static_field_bundle(
            raw_model,
            raw_data,
            resolved=resolved,
            protocol=runtime_protocol,
            protocol_hashes=runtime_hashes,
        )
        if not np.array_equal(settled_official, fast._official_state(source_env.sim)):
            raise ClosedLoopRunnerError("field construction changed simulator state")
        forwarded = clone_forwarded_state(raw_model, raw_data)
        full_samples = build_robot_collision_samples(
            source_env.sim.model,
            forwarded,
            geom_ids=resolved.robot_geom_ids,
            epsilon_m=float(runtime_protocol["coverage"]["epsilon_m"]),
        )
        roundtrip = validate_rigid_roundtrip(full_samples.samples, forwarded)
        sample_evidence = {
            "sample_count": len(full_samples.samples),
            "sample_ledger_sha256": full_samples.sample_ledger_sha256,
            "geom_records": list(full_samples.geom_records),
            "epsilon_m": full_samples.epsilon_m,
            "maximum_surface_cover_radius_m": full_samples.maximum_surface_cover_radius_m,
            "coverage_semantics": full_samples.coverage_semantics,
            "roundtrip": roundtrip,
        }
        validate_robot_sample_evidence(
            sample_evidence,
            resolved_geom_ids=resolved.robot_geom_ids,
            resolved_geom_names=resolved.robot_geom_names,
            resolved_body_ids=resolved.robot_body_ids,
            roundtrip_field="roundtrip",
        )
        settled_obstacle_state = fast._obstacle_state(raw_model, forwarded, bundle, resolved.obstacle_body_ids)
        if not fast._static_admissible([settled_obstacle_state], runtime_protocol):
            raise ClosedLoopRunnerError("selected obstacle is not static after settling")
        prefix_obstacle_rows: List[Dict[str, Any]] = []
        for expected_step in replay.steps[:180]:
            observation, reward, done, _ = source_env.step(expected_step.action)
            _, _, previous_goal = _check_step(
                evaluator=evaluator,
                env=source_env,
                observation=observation,
                reward=reward,
                done=done,
                expected_step=expected_step,
                goal_atoms=goal_atoms,
                previous_goal_values=previous_goal,
                np=np,
            )
            evaluator._update_eef_marker(
                source_env, evaluator._eef_proxy(runtime, observation)
            )
            prefix_forwarded = clone_forwarded_state(raw_model, raw_data)
            prefix_obstacle_rows.append(
                {
                    "source_action_index": int(expected_step.step),
                    **fast._obstacle_state(
                        raw_model, prefix_forwarded, bundle, resolved.obstacle_body_ids
                    ),
                }
            )
            if not fast._static_admissible(
                [settled_obstacle_state] + prefix_obstacle_rows, runtime_protocol
            ):
                raise ClosedLoopRunnerError("static field became invalid during shared prefix")
        branch_observation = copy.deepcopy(observation)
        if tuple(previous_goal) != (False,):
            raise ClosedLoopRunnerError("native task is not unsatisfied at branch")
        flattened_B = np.asarray(source_env.sim.get_state().flatten(), dtype=np.float64).copy()
        observed_B_hash = evaluator.array_sha256(flattened_B)
        if not (
            observed_B_hash
            == replay.steps[179].simulator_state_sha256
            == source_contract["post_action_179_flattened_state_sha256"]
        ):
            raise ClosedLoopRunnerError("post-action-179 branch state differs")
        official_B = fast._official_state(source_env.sim)
        drift_rows = [settled_obstacle_state] + prefix_obstacle_rows
        assumption_evidence = {
            "field_construction_boundary": 0,
            "field_construction_state": "after_20_settling_actions",
            "prefix_action_boundary_count": len(prefix_obstacle_rows),
            "maximum_prefix_translation_drift_m": max(row["translation_drift_m"] for row in drift_rows),
            "maximum_prefix_rotation_drift_rad": max(row["rotation_drift_rad"] for row in drift_rows),
            "maximum_prefix_surface_drift_m": max(row["surface_drift_m"] for row in drift_rows),
            "admissible": fast._static_admissible(drift_rows, runtime_protocol),
        }
        if assumption_evidence["admissible"] is not True:
            raise ClosedLoopRunnerError("static field is invalid through shared prefix")

        placeholder_actions = [[0.0] * 7 for _ in range(derived["action_count"])]
        server_metadata = None
        for arm_key, runner_arm, protocol_arm, video_slug in (
            ("baseline", "joint_velocity_adapter_only", BASELINE_ARM, "pi05-aegis-baseline"),
            ("psf", "joint_velocity_adapter_plus_link56_psf", PSF_ARM, "pi05-aegis-poisson-cbf"),
        ):
            client = runtime["websocket_client_policy"].WebsocketClientPolicy(
                arguments.host, arguments.port
            )
            metadata = evaluator._server_identity(client)
            if metadata.get("status") != "available":
                raise ClosedLoopRunnerError("pi0.5 server metadata is unavailable")
            if server_metadata is None:
                server_metadata = metadata
            elif fast._canonical(server_metadata) != fast._canonical(metadata):
                raise ClosedLoopRunnerError("pi0.5 server metadata changed between arms")
            provider = LiveAegisPolicy(
                evaluator=evaluator,
                runtime=runtime,
                client=client,
                case=case,
                task_description=str(task.language),
                historical_result=historical_result,
                first_query_index=derived["first_query_index"],
                replan_steps=derived["replan_steps"],
                model_action_horizon=protocol["online_policy"]["model_action_horizon"],
                expected_first_chunk_sha256=source_contract["first_live_query_expected_action_chunk_sha256"],
                arm_name=protocol_arm,
            )
            video = RolloutVideo(
                evaluator=evaluator,
                imageio=runtime["imageio"],
                run_root=output,
                arm_slug=video_slug,
                fps=protocol["videos"]["fps"],
            )
            try:
                arm = fast._run_arm(
                    arm_name=runner_arm,
                    evaluator=evaluator,
                    runtime=runtime,
                    case=case,
                    source_env=source_env,
                    state_at_B=flattened_B,
                    actions=placeholder_actions,
                    source_start_action=derived["start_action"],
                    bundle=bundle,
                    resolved=resolved,
                    full_samples=full_samples,
                    runtime_protocol=runtime_protocol,
                    nominal_activation_threshold_m2_per_s=derived["thresholds"]["maximum_nominal_cbf_residual_for_activation_m2_per_s"],
                    qp_max_iterations=int(protocol["exploratory_execution"]["qp_max_iterations"]),
                    expected_filter_updates=derived["expected_updates"],
                    expected_physics_substeps=derived["expected_substeps"],
                    expected_boundary_goal_values=(False,),
                    live_action_provider=provider,
                    rollout_frame_observer=video,
                    initial_live_observation=copy.deepcopy(branch_observation),
                    full_clearance_observation_stride=derived[
                        "clearance_diagnostic_stride"
                    ],
                )
            except fast.FastArmExecutionFailure as error:
                arm_evidence[arm_key] = dict(error.evidence)
                raise
            finally:
                active_exception = sys.exc_info()[0] is not None
                finalization_error = None
                try:
                    providers[arm_key] = provider.record()
                except Exception as provider_error:
                    providers[arm_key] = {
                        "status": "finalization_failure",
                        "failure_type": type(provider_error).__name__,
                        "failure_message": str(provider_error),
                    }
                    finalization_error = provider_error
                try:
                    video_evidence[arm_key] = video.close()
                except Exception as video_error:
                    video_evidence[arm_key] = {
                        "status": "finalization_failure",
                        "failure_type": type(video_error).__name__,
                        "failure_message": str(video_error),
                    }
                    if finalization_error is None:
                        finalization_error = video_error
                if not active_exception and finalization_error is not None:
                    raise finalization_error
            arm_evidence[arm_key] = arm

        baseline = arm_evidence["baseline"]
        psf = arm_evidence["psf"]
        baseline_provider = providers["baseline"]
        psf_provider = providers["psf"]
        exact_pair = fast._pair_exact(baseline, psf)
        feedback = _feedback_metrics(
            baseline_provider,
            psf_provider,
            source_contract["first_live_query_expected_action_chunk_sha256"],
        )
        baseline_contract = _provider_contract(
            baseline_provider,
            expected_actions=derived["action_count"],
            expected_queries=derived["queries_per_arm"],
            policy_noise_seed=int(case["policy_noise_seed"]),
        )
        psf_expected_actions = len(psf_provider["high_level_action_trace"])
        psf_expected_queries = (
            0
            if psf_expected_actions == 0
            else (psf_expected_actions + derived["replan_steps"] - 1)
            // derived["replan_steps"]
        )
        psf_contract = _provider_contract(
            psf_provider,
            expected_actions=psf_expected_actions,
            expected_queries=psf_expected_queries,
            policy_noise_seed=int(case["policy_noise_seed"]),
        )
        no_replay = bool(
            baseline_provider.get("recorded_suffix_actions_executed") is False
            and psf_provider.get("recorded_suffix_actions_executed") is False
            and all(
                row.get("source")
                == "live_pi05_queries_from_this_arms_own_observations"
                for row in (baseline_provider, psf_provider)
            )
        )
        historical_action_180 = historical_result["actions"][180]["executed"]
        first_live_aegis_action_matches_historical = bool(
            all(
                provider["high_level_action_trace"]
                and np.allclose(
                    np.asarray(
                        provider["high_level_action_trace"][0]["aegis_executed"],
                        dtype=np.float64,
                    ),
                    np.asarray(historical_action_180, dtype=np.float64),
                    rtol=0.0,
                    atol=1e-12,
                )
                for provider in (baseline_provider, psf_provider)
            )
        )
        aegis_contract = bool(
            all(
                len(provider["high_level_action_trace"]) > 0
                and provider.get(
                    "first_aegis_input_binding_matches_historical"
                )
                is True
                and provider.get(
                    "first_aegis_internal_output_matches_historical"
                )
                is True
                and all(
                    row.get("aegis_qp", {}).get("status") == "solved"
                    and row.get("aegis_qp", {}).get("solver_status")
                    in ("optimal", "optimal_inaccurate")
                    for row in provider["high_level_action_trace"]
                )
                for provider in (baseline_provider, psf_provider)
            )
            and _aegis_state_and_command_chain_valid(
                baseline_provider, baseline
            )
            and _aegis_state_and_command_chain_valid(psf_provider, psf)
        )
        baseline_contact = baseline["literal_contact"]
        psf_contact = psf["literal_contact"]
        baseline_contact_boundary = baseline_contact["first_link56_physical_boundary"]
        material = [
            row
            for row in psf["activation_trace"]
            if row["filter_correction_l2_rad_s"]
            >= derived["thresholds"]["minimum_filter_correction_norm_rad_s"]
            and row["nominal_within_dynamic_joint_bounds"] is True
            and (
                baseline_contact_boundary is None
                or row["physical_boundary"] < baseline_contact_boundary
            )
        ]
        first_correction = material[0]["physical_boundary"] if material else None
        fresh_psf_queries_after_correction = [
            int(row["query_index"])
            for row in psf_provider["policy_queries"]
            if first_correction is not None
            and int(row["source_action_index"]) * 25 > int(first_correction)
        ]
        post_motion = fast._post_correction_motion(
            psf["command_trace"],
            psf["physics_trace"],
            first_correction_physical_boundary=first_correction,
        )
        psf_clearance = psf["conservative_full_robot_clearance"]
        baseline_task = baseline["task"]
        psf_task = psf["task"]
        psf_success_rows_after_correction = [
            row
            for row in psf_task["goal_progress_ledger"]
            if first_correction is not None
            and row.get("snapshot_kind") == "completed_high_level_post_step"
            and row.get("all_satisfied") is True
            and (int(row["source_action_index"]) + 1) * 25
            > int(first_correction)
        ]
        success_after_correction = bool(psf_success_rows_after_correction)
        baseline_video_ok = _video_complete(
            video_evidence["baseline"],
            baseline_task["completed_source_action_count"],
            False,
        )
        psf_video_ok = _video_complete(
            video_evidence["psf"],
            psf_task["completed_source_action_count"],
            bool(psf_contact["any_robot_selected_obstacle_present"]),
        )
        psf_qp_contract = bool(
            psf["precontact_execution_valid"] is True
            and psf["qp_solve_count"] == len(psf["command_trace"])
            and psf["qp_postcheck_count"] == len(psf["command_trace"])
            and psf["joint_limit_postcheck_count"] == len(psf["command_trace"])
        )
        metrics = {
            "exact_paired_start": exact_pair,
            "shared_prefix_complete": bool(
                len(prefix_obstacle_rows) == 180
                and observed_B_hash == source_contract["post_action_179_flattened_state_sha256"]
                and assumption_evidence["admissible"] is True
            ),
            "baseline_exposure_complete": baseline["exposure_complete"],
            "psf_exposure_complete": psf["exposure_complete"],
            "live_policy_contract_valid": bool(baseline_contract and psf_contract),
            "aegis_contract_valid": bool(
                aegis_contract and first_live_aegis_action_matches_historical
            ),
            "videos_complete_and_decodable": bool(baseline_video_ok and psf_video_ok),
            "psf_qp_contract_valid": psf_qp_contract,
            "static_selected_obstacle_admissible": bool(
                assumption_evidence["admissible"]
                and baseline["static_precontact_admissible"]
                and psf["static_precontact_admissible"]
            ),
            "literal_contact_checked_at_every_physics_substep": bool(
                baseline["monitor_observed_physics_substep_count"]
                == len(baseline["physics_trace"])
                and psf["monitor_observed_physics_substep_count"]
                == len(psf["physics_trace"])
                and all(
                    "literal_contact_observed" in row
                    for arm in (baseline, psf)
                    for row in arm["physics_trace"]
                )
            ),
            "released_eef_marker_update_contract_valid": bool(
                all(
                    arm.get("released_eef_marker_update_count")
                    == arm["task"]["completed_source_action_count"]
                    and arm.get("released_eef_marker_update_semantics")
                    == "one_model_update_after_each_completed_high_level_env_step"
                    for arm in (baseline, psf)
                )
            ),
            "first_live_query_identical": feedback["first_live_query_identical"],
            "post_divergence_own_observations_used": feedback["post_divergence_own_observations_used"],
            "post_divergence_policy_inputs_differ": feedback["post_divergence_policy_inputs_differ"],
            "fresh_policy_query_after_material_correction": bool(
                fresh_psf_queries_after_correction
            ),
            "fresh_policy_query_indexes_after_material_correction": (
                fresh_psf_queries_after_correction
            ),
            "own_observation_chain_valid": bool(
                _own_observation_chain_valid(baseline_provider, baseline)
                and _own_observation_chain_valid(psf_provider, psf)
            ),
            "no_recorded_suffix_action_replay": no_replay,
            "first_live_aegis_action_matches_historical": (
                first_live_aegis_action_matches_historical
            ),
            "both_nominal_commands_within_dynamic_joint_bounds": bool(
                baseline["all_nominal_commands_within_dynamic_joint_bounds"]
                and psf["all_nominal_commands_within_dynamic_joint_bounds"]
            ),
            "baseline_link56_contact_present": baseline_contact["link56_present"],
            "baseline_first_selected_obstacle_contact_is_link56": bool(
                baseline_contact["first_link56_physical_boundary"] is not None
                and baseline_contact["first_link56_physical_boundary"]
                == baseline_contact["first_any_robot_physical_boundary"]
            ),
            "psf_link56_contact_present": psf_contact["link56_present"],
            "psf_any_robot_selected_obstacle_contact_present": psf_contact["any_robot_selected_obstacle_present"],
            "psf_clearance_certified": bool(
                psf_clearance["available"]
                and psf_clearance["minimum_full_surface_lower_bound_m"] > 0.0
                and psf_clearance["continuous_every_substep_certificate"]
            ),
            "psf_periodic_clearance_diagnostic_positive": bool(
                psf_clearance["available"]
                and psf_clearance["minimum_full_surface_lower_bound_m"] > 0.0
            ),
            "material_correction_before_baseline_contact": bool(
                first_correction is not None
                and baseline_contact_boundary is not None
                and int(first_correction) < int(baseline_contact_boundary)
            ),
            "first_material_correction_physical_boundary": first_correction,
            "baseline_first_link56_contact_physical_boundary": baseline_contact_boundary,
            "material_correction_update_count": len(material),
            "maximum_correction_norm_rad_s": post_motion["maximum_correction_norm_rad_s"],
            "filter_correction_integral_rad": post_motion["filter_correction_integral_rad"],
            "post_correction_measured_joint_motion_integral_rad": post_motion["measured_joint_motion_integral_rad"],
            "post_correction_cartesian_path_length_m": post_motion["cartesian_path_length_m"],
            "post_correction_executed_command_integral_rad": post_motion["executed_command_integral_rad"],
            "post_correction_zero_command_fraction": post_motion["zero_command_fraction"],
            "baseline_task_success_ever": baseline_task["ever_task_success_at_or_after_branch"],
            "baseline_terminal_task_success": baseline_task["terminal_task_success"],
            "psf_task_success_ever": psf_task["ever_task_success_at_or_after_branch"],
            "psf_terminal_task_success": psf_task["terminal_task_success"],
            "psf_task_success_after_material_correction": success_after_correction,
            "psf_successful_source_action_indexes_after_material_correction": [
                int(row["source_action_index"])
                for row in psf_success_rows_after_correction
            ],
        }
        classification = classify_closed_loop_canary(metrics, protocol)
        candidate = {
            "schema_version": RESULT_SCHEMA,
            "status": "complete",
            "protocol_id": protocol["protocol_id"],
            "run_id": arguments.run_id,
            "case_id": arguments.case_id,
            "provenance": provenance,
            "policy_server": server_metadata,
            "episode": {
                "shared_prefix_action_indexes": list(range(180)),
                "shared_prefix_execution": "exact_historical_aegis_actions_under_unchanged_osc",
                "branch_flattened_state_sha256": observed_B_hash,
                "branch_official_integration_state_raw_bytes_sha256": fast._sha256(official_B.tobytes()),
                "live_suffix_action_indexes": list(range(180, 237)),
                "fixed_live_suffix_after_success": True,
                "historical_suffix_reference_sha256_never_executed": historical_suffix_reference_sha256,
            },
            "field": {
                "bundle_hashes": asdict(bundle.hashes),
                "diagnostics": asdict(bundle.diagnostics),
                "resolved_geometry": resolved.to_dict(),
                "full_robot_sampling": sample_evidence,
                "static_assumption": assumption_evidence,
            },
            "closed_loop_feedback": feedback,
            "providers": providers,
            "arms": arm_evidence,
            "videos": video_evidence,
            "metrics": metrics,
            "post_correction_motion": post_motion,
            "classification": classification,
            "producer_classification_is_preliminary": True,
            "scientific_interpretation_requires_independent_consumer": True,
            "partial_output_interpreted": False,
            "claim_scope": protocol["result_contract"]["claim_scope"],
            "timing": {
                "started_unix": started,
                "finished_unix": time.time(),
                "elapsed_seconds": time.time() - started,
            },
        }
        publish_hashed_json(result_path, candidate)
        print(
            json.dumps(
                {
                    "status": "complete",
                    "classification": classification["classification"],
                    "feasible": classification["feasible"],
                    "result": str(result_path),
                },
                sort_keys=True,
            )
        )
        return 0
    except Exception as error:
        if output is None:
            try:
                output = fast._new_output(arguments.output_root.resolve(), arguments.run_id)
                result_path = output / "result.json"
            except Exception:
                output = None
        if result_path is not None and not result_path.exists():
            try:
                from main.poisson_fullbody.closed_loop_canary import RESULT_SCHEMA
                from main.poisson_fullbody.contracts import publish_hashed_json

                publish_hashed_json(
                    result_path,
                    {
                        "schema_version": RESULT_SCHEMA,
                        "status": "apparatus_failure",
                        "run_id": arguments.run_id,
                        "case_id": arguments.case_id,
                        "provenance": provenance,
                        "arms": arm_evidence,
                        "providers": providers,
                        "videos": video_evidence,
                        "static_assumption": assumption_evidence,
                        "partial_output_interpreted": False,
                        "failure": {
                            "type": type(error).__name__,
                            "message": str(error),
                            "traceback": traceback.format_exc(),
                        },
                        "timing": {
                            "started_unix": started,
                            "finished_unix": time.time(),
                            "elapsed_seconds": time.time() - started,
                        },
                    },
                )
            except Exception as publication_error:
                print(
                    "closed-loop failure publication failed: %s" % publication_error,
                    file=os.sys.stderr,
                )
        print("closed-loop canary failed: %s" % error, file=os.sys.stderr)
        return 1
    finally:
        if source_env is not None:
            source_env.close()


if __name__ == "__main__":
    raise SystemExit(main())
