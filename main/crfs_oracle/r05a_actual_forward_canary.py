"""Allocation-only AF-00A actual-forward CEM canary.

The runner restores the exact R02 source branch, issues only ordinary
``residual_schedule`` requests for all controls, and writes raw immutable
evidence.  It never executes a returned policy or teacher action in the
simulator and never publishes ``results.json``.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import socket
import tempfile
from typing import Any, Mapping, Optional

import numpy as np

from crfs_harness.artifacts import atomic_write_json, file_sha256

from .progress_calibration import _target_contact_at_branch
from .r02_runner import (
    _array_record,
    _json_compatible,
    _observation_fingerprint,
    _trace_record,
    _validate_array_record,
)
from .r03a_runner import _trace_from_record, _trace_pairing_diagnostics
from .r05a_actual_forward_search import (
    ACTION_SHAPE,
    COMPACT_SHAPE,
    DT_FLOAT32,
    ActualForwardEvaluation,
    ActualForwardSearchError,
    run_actual_forward_cem,
    transport_increment,
)
from .r05a_canary import (
    CASE_ID,
    GROUP_ID,
    MANIFEST_SHA256,
    REGISTERED_XYZ_SCALE,
    SOURCE_HOST,
    R05ACanaryPolicyError,
    R05ACanarySourceError,
    _array_exact,
    _deterministic_trace,
    _finite_array,
    _frozen_controls,
    _load_source_r02,
    _request,
    _replay_summary,
    _schedule_controls,
    _source_delta,
    _trace_exact,
)
from .reach_progress import TARGET_OBJECT_NAME, capture_reach_snapshot
from .runner import SafeLiberoCase, policy_observation


PAYLOAD_TYPE = "r05a_actual_forward_cem_raw_payload"
TENSOR_FILENAME = "af00a-tensors.npz"
LEDGER_FILENAME = "query-ledger.json"
PAYLOAD_FILENAME = "af00a-raw-payload.json"
EXACT_POLICY_REQUESTS = 534
SEARCH_REQUESTS = 520
TRACE_ARRAY_KEYS = (
    "step_index_steps",
    "time_steps",
    "active_steps",
    "x_t_steps",
    "v_base_steps",
    "control_velocity_steps",
    "total_velocity_steps",
    "control_increment_steps",
    "x_next_steps",
    "initial_noise",
    "final_normalized",
)


class ActualForwardCanaryError(RuntimeError):
    """The AF-00A pairing, request ledger, or raw artifact is invalid."""


def _bytes_digest(value: Any) -> Mapping[str, Any]:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes())
    return {"dtype": str(array.dtype), "shape": list(array.shape), "sha256": digest.hexdigest()}


def _atomic_write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    if path.exists():
        raise ActualForwardCanaryError(f"immutable tensor artifact already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _validate_actual_config(value: Mapping[str, Any]) -> None:
    if value.get("schema_version") != "1.0" or value.get("experiment_identity") != "AF-00A":
        raise ActualForwardCanaryError("actual-forward config identity changed")
    if value.get("ready_to_run") is not True or value.get("execution_release") is None:
        raise ActualForwardCanaryError("actual-forward config is not execution-released")
    flow = value.get("flow_contract", {})
    cem = value.get("cem", {})
    ledger = value.get("request_ledger", {})
    exact = {
        "flow compact shape": flow.get("compact_shape") == [5, 15],
        "flow dt": np.float32(flow.get("dt_float32")).tobytes() == DT_FLOAT32.tobytes(),
        "ordinary schedule only": flow.get("ordinary_residual_schedule_path_only") is True,
        "CEM dimensions": (
            cem.get("generations"), cem.get("population_size"), cem.get("elite_count")
        ) == (8, 65, 13),
        "CEM seed": cem.get("search_seed") == 20260716,
        "request count": ledger.get("complete_run_exact_policy_request_count")
        == EXACT_POLICY_REQUESTS,
    }
    failed = [name for name, passed in exact.items() if not passed]
    if failed:
        raise ActualForwardCanaryError("actual-forward frozen config changed: " + ", ".join(failed))


def _compact_to_schedule(velocity: np.ndarray) -> np.ndarray:
    compact = np.asarray(velocity)
    if compact.shape != COMPACT_SHAPE or compact.dtype != np.dtype(np.float32):
        raise ActualForwardCanaryError("compact velocity must preserve float32 (5,15)")
    if not bool(np.isfinite(compact).all()):
        raise ActualForwardCanaryError("compact velocity is nonfinite")
    schedule = np.zeros((10, 10, 32), dtype=np.float32)
    schedule[5:, :5, :3] = compact.reshape(5, 5, 3)
    return schedule


def _compact_executed(trace: Mapping[str, Any]) -> np.ndarray:
    increments = _finite_array(
        trace.get("control_increment_steps"),
        name="actual-forward executed increments",
        shape=(10, 10, 32),
    )
    if increments.dtype != np.dtype(np.float32):
        raise R05ACanaryPolicyError("executed increments did not preserve float32")
    return np.ascontiguousarray(increments[5:, :5, :3].reshape(COMPACT_SHAPE))


def _exact_scientific_reply(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return bool(
        _array_exact(left.get("actions"), right.get("actions"))
        and _array_exact(
            left.get("crfs_trace", {}).get("final_normalized"),
            right.get("crfs_trace", {}).get("final_normalized"),
        )
        and _trace_exact(
            _deterministic_trace(left.get("crfs_trace", {})),
            _deterministic_trace(right.get("crfs_trace", {})),
        )
    )


def run_actual_forward_canary(
    case: Mapping[str, Any],
    actual_config_mapping: Mapping[str, Any],
    legacy_config: Any,
    *,
    repo_root: str | Path,
    input_manifest_sha256: str,
    actual_config_path: str | Path,
    legacy_config_path: str | Path,
    client: Any = None,
    environment: Optional[SafeLiberoCase] = None,
) -> tuple[Path, str]:
    """Execute the exact 534-call AF-00A raw-evidence transaction."""

    _validate_actual_config(actual_config_mapping)
    if not os.environ.get("SLURM_JOB_ID"):
        raise ActualForwardCanaryError("AF-00A must execute inside Slurm")
    if not os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
        raise ActualForwardCanaryError("AF-00A requires one visible allocation GPU")
    if socket.gethostname().split(".", 1)[0] != SOURCE_HOST:
        raise ActualForwardCanaryError(f"AF-00A must remain pinned to {SOURCE_HOST}")
    manifest_binding = actual_config_mapping.get("frozen_source_bindings", {}).get("manifest", {})
    if input_manifest_sha256 != MANIFEST_SHA256 or input_manifest_sha256 != manifest_binding.get("sha256"):
        raise R05ACanarySourceError("AF-00A manifest binding changed")
    frozen_case = actual_config_mapping.get("frozen_case", {})
    if dict(case).get("case_id") != CASE_ID or dict(case).get("group_id") != GROUP_ID:
        raise R05ACanarySourceError("AF-00A case/group identity changed")
    if any(dict(case).get(key) != frozen_case.get(key) for key in ("case_id", "group_id", "environment_seed", "policy_seed")):
        raise R05ACanarySourceError("AF-00A case differs from released config")

    actual_path = Path(actual_config_path).resolve()
    legacy_path = Path(legacy_config_path).resolve()
    if json.loads(actual_path.read_text(encoding="utf-8")) != dict(actual_config_mapping):
        raise ActualForwardCanaryError("actual config mapping differs from its file")
    root = Path(repo_root).resolve()
    output_dir = Path(legacy_config.oracle.output_root) / legacy_config.oracle.run_id / CASE_ID
    tensor_path = output_dir / TENSOR_FILENAME
    ledger_path = output_dir / LEDGER_FILENAME
    payload_path = output_dir / PAYLOAD_FILENAME
    if any(path.exists() for path in (tensor_path, ledger_path, payload_path)):
        raise ActualForwardCanaryError("AF-00A immutable raw artifact already exists")

    source_path, raw_r02, source_sha = _load_source_r02(legacy_config)
    if raw_r02.get("provenance", {}).get("case_record") != dict(case):
        raise R05ACanarySourceError("manifest row differs from immutable R02 source")
    delta64, delta_record, source_budget64 = _source_delta(raw_r02)
    source_budget32 = np.float32(raw_r02["directions"]["l2_norms"]["delta_star_model"])
    registered_budget32 = np.float32(
        actual_config_mapping.get("target_contract", {}).get("source_budget_float32")
    )
    if source_budget32.tobytes() != registered_budget32.tobytes():
        raise R05ACanarySourceError("source budget differs from AF-00A config")
    delta32 = np.asarray(delta64, dtype=np.float32)
    noise = np.random.default_rng(int(case["policy_seed"])).normal(size=(10, 32)).astype(np.float32)
    source_noise, noise_errors = _validate_array_record(
        raw_r02.get("provenance", {}).get("noise"), name="source R02 noise", shape=(10, 32)
    )
    if noise_errors or source_noise is None or not _array_exact(noise, np.asarray(source_noise, dtype=np.float32)):
        raise R05ACanarySourceError("reconstructed policy noise differs from R02")

    if client is None:
        from openpi_client import websocket_client_policy

        client = websocket_client_policy.WebsocketClientPolicy(
            legacy_config.oracle.host, legacy_config.oracle.port
        )
    owns_environment = environment is None
    if environment is None:
        environment = SafeLiberoCase(dict(case), legacy_config.oracle)
    else:
        environment.configure_case(dict(case))

    ledger_rows: list[dict[str, Any]] = []
    trace_rows: dict[str, list[np.ndarray]] = {key: [] for key in TRACE_ARRAY_KEYS}
    trace_policy_indices: list[int] = []
    policy_index = 0

    def record_reply(
        phase: str,
        reply: Mapping[str, Any],
        elapsed: float,
        *,
        record_recurrence: bool = False,
        schedule: np.ndarray | None = None,
        cem_query: int | None = None,
        phase_index: int = 0,
        generation: int | None = None,
        population_index: int | None = None,
    ) -> None:
        nonlocal policy_index
        trace = reply.get("crfs_trace")
        trace_slot = None
        if record_recurrence:
            if not isinstance(trace, Mapping):
                raise ActualForwardCanaryError(
                    f"{phase} omitted its residual-schedule recurrence trace"
                )
            trace_slot = len(trace_policy_indices)
            for key in TRACE_ARRAY_KEYS:
                if key not in trace:
                    raise ActualForwardCanaryError(
                        f"{phase} recurrence trace omitted {key}"
                    )
                value = np.asarray(trace[key])
                if not bool(np.isfinite(value).all()):
                    raise ActualForwardCanaryError(f"trace {key} is missing or nonfinite")
                trace_rows[key].append(np.ascontiguousarray(value.copy()))
            trace_policy_indices.append(policy_index)
        row: dict[str, Any] = {
            "ordinal": policy_index,
            "phase": phase,
            "phase_index": phase_index,
            "cem_query_index": cem_query,
            "generation": generation,
            "population_index": population_index,
            "trace_slot": trace_slot,
            "elapsed_seconds": float(elapsed),
            "actions": _bytes_digest(reply["actions"]),
        }
        if schedule is not None:
            row["requested_schedule"] = _bytes_digest(schedule)
        ledger_rows.append(row)
        policy_index += 1

    def request(
        phase: str,
        controls: Mapping[str, Any],
        *,
        require_trace: bool,
        record_recurrence: bool = False,
        schedule: np.ndarray | None = None,
        cem_query: int | None = None,
        phase_index: int = 0,
        generation: int | None = None,
        population_index: int | None = None,
    ) -> tuple[Mapping[str, Any], float]:
        reply, elapsed = _request(
            client, policy_input, controls, require_trace=require_trace
        )
        record_reply(
            phase,
            reply,
            elapsed,
            record_recurrence=record_recurrence,
            schedule=schedule,
            cem_query=cem_query,
            phase_index=phase_index,
            generation=generation,
            population_index=population_index,
        )
        return reply, elapsed

    try:
        initial_observation = environment.reset_and_settle()
        if environment.obstacle_name is None:
            raise ActualForwardCanaryError("AF-00A did not resolve the active obstacle")
        branch = capture_reach_snapshot(environment, TARGET_OBJECT_NAME, environment.obstacle_name)
        if _target_contact_at_branch(environment, TARGET_OBJECT_NAME) or environment.env.check_success():
            raise ActualForwardCanaryError("AF-00A source is no longer a pregrasp branch")
        policy_input = policy_observation(initial_observation, environment.prompt, legacy_config.oracle.resize_size)
        source_pairing = raw_r02.get("pairing")
        if not isinstance(source_pairing, Mapping):
            raise R05ACanarySourceError("R02 source has no pairing record")
        observation_fingerprint = _observation_fingerprint(policy_input)
        branch_snapshot = _json_compatible(branch.to_dict())
        pairing_checks = {
            "observation_exact": observation_fingerprint
            == source_pairing.get("policy_observation"),
            "branch_snapshot_exact": branch_snapshot
            == source_pairing.get("branch_snapshot"),
            "noise_exact": True,
        }
        if not all(pairing_checks.values()):
            raise R05ACanarySourceError(f"AF-00A branch pairing failed: {pairing_checks}")

        compiled_before, _ = request(
            "compiled_frozen_pre", _frozen_controls(noise, return_trace=False), require_trace=False
        )
        source_before, _ = request(
            "eager_source_trace_pre", _frozen_controls(noise, return_trace=True), require_trace=True
        )
        source_actions, action_errors = _validate_array_record(
            source_pairing.get("eager_actions"), name="source eager actions", shape=ACTION_SHAPE
        )
        if action_errors or source_actions is None or not _array_exact(source_before["actions"], source_actions):
            raise R05ACanarySourceError("current source actions differ from immutable R02")
        source_trace = _trace_from_record(source_pairing.get("eager_trace"), name="source eager trace")
        if _trace_pairing_diagnostics(source_trace, source_before["crfs_trace"]).get("exact_native_leaf_pairing") is not True:
            raise R05ACanarySourceError("current source trace differs from immutable R02")
        eager_before, _ = request(
            "eager_normalized_final_pre",
            _frozen_controls(noise, return_trace=True, return_normalized_final=True),
            require_trace=True,
        )
        frozen_final = _finite_array(
            eager_before["crfs_trace"].get("final_normalized"),
            name="fresh frozen final",
            shape=(10, 32),
        )
        if frozen_final.dtype != np.dtype(np.float32) or not _array_exact(eager_before["actions"], source_before["actions"]):
            raise R05ACanaryPolicyError("fresh frozen normalized path changed source actions")
        target = np.where(
            delta32 == np.float32(0.0), frozen_final, np.asarray(frozen_final + delta32, dtype=np.float32)
        ).astype(np.float32, copy=False)
        model_to_physical_scale = np.ones((10, 32), dtype=np.float32)
        model_to_physical_scale[:, :3] = np.asarray(REGISTERED_XYZ_SCALE, dtype=np.float32)
        target_physical = np.asarray(eager_before["actions"], dtype=np.float64).copy()
        target_physical[:5, :3] += np.asarray(delta32[:5, :3], dtype=np.float64) * np.asarray(
            model_to_physical_scale[:5, :3], dtype=np.float64
        )

        zero_schedule = np.zeros((10, 10, 32), dtype=np.float32)
        zero_before, zero_before_elapsed = request(
            "zero_schedule_pre",
            _schedule_controls(noise, zero_schedule, np.float32(0.0)),
            require_trace=True,
            record_recurrence=True,
            schedule=zero_schedule,
        )
        if not _replay_summary(
            zero_before,
            requested_schedule=zero_schedule,
            source_budget_float32=np.float32(0.0),
            paired_noise=noise,
            elapsed_seconds=zero_before_elapsed,
        )["passed"]:
            raise ActualForwardCanaryError("paired zero precheck failed")

        arm_a_requested_c = np.broadcast_to(
            np.asarray(
                delta32[:5, :3].reshape(15) / np.float32(5.0), dtype=np.float32
            ),
            COMPACT_SHAPE,
        ).copy()
        arm_a_velocity = np.ascontiguousarray(
            np.asarray(arm_a_requested_c / DT_FLOAT32, dtype=np.float32)
        )

        seen: dict[bytes, tuple[np.ndarray, np.ndarray, Mapping[str, Any]]] = {}
        special_replies: dict[str, list[Mapping[str, Any]]] = {"A": [], "B": [], "C": []}
        search_traces: dict[int, Mapping[str, Any]] = {}

        def evaluation_from_reply(reply: Mapping[str, Any]) -> ActualForwardEvaluation:
            trace = reply["crfs_trace"]
            executed = _compact_executed(trace)
            final = _finite_array(trace.get("final_normalized"), name="schedule final", shape=(10, 32))
            actions = _finite_array(reply.get("actions"), name="schedule actions", shape=ACTION_SHAPE)
            if actions.dtype != np.dtype(np.float64):
                raise ActualForwardCanaryError(
                    "schedule returned actions must preserve the registered float64 dtype"
                )
            error = np.ascontiguousarray(np.asarray(actions[:5] - target_physical[:5], dtype=np.float64))
            return ActualForwardEvaluation(executed, final, actions, error)

        def schedule_call(
            phase: str,
            velocity: np.ndarray,
            *,
            cem_query: int | None = None,
            phase_index: int = 0,
            generation: int | None = None,
            population_index: int | None = None,
        ) -> tuple[ActualForwardEvaluation, Mapping[str, Any]]:
            schedule = _compact_to_schedule(velocity)
            reply, elapsed = request(
                phase,
                _schedule_controls(noise, schedule, source_budget32),
                require_trace=True,
                record_recurrence=True,
                schedule=schedule,
                cem_query=cem_query,
                phase_index=phase_index,
                generation=generation,
                population_index=population_index,
            )
            summary = _replay_summary(
                reply,
                requested_schedule=schedule,
                source_budget_float32=source_budget32,
                paired_noise=noise,
                elapsed_seconds=elapsed,
            )
            if summary["passed"] is not True:
                raise ActualForwardCanaryError(f"{phase} failed ordinary replay validation")
            evaluation = evaluation_from_reply(reply)
            key = evaluation.executed_increment.tobytes()
            scientific = (
                evaluation.normalized_final,
                np.asarray(evaluation.returned_physical_action),
                _deterministic_trace(reply["crfs_trace"]),
            )
            prior = seen.get(key)
            if prior is not None and not (
                _array_exact(prior[0], scientific[0])
                and _array_exact(prior[1], scientific[1])
                and _trace_exact(prior[2], scientific[2])
            ):
                raise ActualForwardCanaryError("identical executed schedule changed scientific output")
            seen[key] = scientific
            return evaluation, reply

        arm_a_first, arm_a_first_reply = schedule_call(
            "arm_a_equal_split", arm_a_velocity, phase_index=0
        )
        arm_a_duplicate, arm_a_duplicate_reply = schedule_call(
            "arm_a_equal_split", arm_a_velocity, phase_index=1
        )
        special_replies["A"].extend((arm_a_first_reply, arm_a_duplicate_reply))
        if not _exact_scientific_reply(arm_a_first_reply, arm_a_duplicate_reply):
            raise ActualForwardCanaryError("Arm A repeatability failed before CEM")

        def evaluate_cem(velocity: np.ndarray) -> ActualForwardEvaluation:
            cem_query = len(search_traces)
            evaluation, reply = schedule_call(
                "cem_search",
                velocity,
                cem_query=cem_query,
                phase_index=cem_query,
                generation=cem_query // 65,
                population_index=cem_query % 65,
            )
            search_traces[cem_query] = reply["crfs_trace"]
            return evaluation

        result = run_actual_forward_cem(
            evaluate_cem,
            arm_a_velocity=arm_a_velocity,
            arm_a_evaluation=arm_a_first,
            budget_float32=source_budget32,
        )
        if len(search_traces) != SEARCH_REQUESTS:
            raise ActualForwardCanaryError("CEM did not execute exactly 520 requests")

        selected_velocity = np.ascontiguousarray(result.selected.requested_velocity)
        selected_eval = result.selected.evaluation
        b_replies: list[Mapping[str, Any]] = []
        for replicate in range(2):
            evaluation, reply = schedule_call(
                "arm_b_replay", selected_velocity, phase_index=replicate
            )
            if not (
                _array_exact(evaluation.executed_increment, selected_eval.executed_increment)
                and _array_exact(evaluation.normalized_final, selected_eval.normalized_final)
                and _array_exact(evaluation.returned_physical_action, selected_eval.returned_physical_action)
            ):
                raise ActualForwardCanaryError("selected B replay differs from selected evaluation")
            b_replies.append(reply)
        special_replies["B"].extend(b_replies)
        if not _exact_scientific_reply(b_replies[0], b_replies[1]):
            raise ActualForwardCanaryError("selected B replays differ")

        reversed_velocity = np.ascontiguousarray(selected_velocity[::-1])
        c_replies: list[Mapping[str, Any]] = []
        for replicate in range(2):
            _, reply = schedule_call(
                "arm_c_reverse_replay", reversed_velocity, phase_index=replicate
            )
            c_replies.append(reply)
        special_replies["C"].extend(c_replies)
        if not _exact_scientific_reply(c_replies[0], c_replies[1]):
            raise ActualForwardCanaryError("reversed B replays differ")

        zero_after, zero_after_elapsed = request(
            "zero_schedule_post",
            _schedule_controls(noise, zero_schedule, np.float32(0.0)),
            require_trace=True,
            record_recurrence=True,
            schedule=zero_schedule,
        )
        if not _replay_summary(
            zero_after,
            requested_schedule=zero_schedule,
            source_budget_float32=np.float32(0.0),
            paired_noise=noise,
            elapsed_seconds=zero_after_elapsed,
        )["passed"]:
            raise ActualForwardCanaryError("paired zero postcheck failed")
        eager_after, _ = request(
            "eager_normalized_final_post",
            _frozen_controls(noise, return_trace=True, return_normalized_final=True),
            require_trace=True,
        )
        source_after, _ = request(
            "eager_source_trace_post", _frozen_controls(noise, return_trace=True), require_trace=True
        )
        compiled_after, _ = request(
            "compiled_frozen_post", _frozen_controls(noise, return_trace=False), require_trace=False
        )
        post_checks = {
            "zero_exact": _exact_scientific_reply(zero_before, zero_after),
            "eager_exact": _exact_scientific_reply(eager_before, eager_after),
            "source_exact": _exact_scientific_reply(source_before, source_after),
            "compiled_actions_exact": _array_exact(compiled_before["actions"], compiled_after["actions"]),
            "source_trace_pairing_exact": _trace_pairing_diagnostics(
                source_trace, source_after["crfs_trace"]
            ).get("exact_native_leaf_pairing")
            is True,
        }
        if not all(post_checks.values()):
            raise ActualForwardCanaryError(f"post-search paired references changed: {post_checks}")
        if policy_index != EXACT_POLICY_REQUESTS or [row["ordinal"] for row in ledger_rows] != list(range(EXACT_POLICY_REQUESTS)):
            raise ActualForwardCanaryError(f"policy request ledger has {policy_index}, expected 534")

        candidates = result.candidates

        def compact_applied(reply: Mapping[str, Any]) -> np.ndarray:
            value = _finite_array(
                reply["crfs_trace"].get("control_velocity_steps"),
                name="trace-applied schedule",
                shape=(10, 10, 32),
            )
            if value.dtype != np.dtype(np.float32):
                raise ActualForwardCanaryError("trace-applied schedule lost float32")
            return np.ascontiguousarray(value[5:, :5, :3].reshape(5, 15))

        def stack_reply(
            replies: list[Mapping[str, Any]], field: str, *, dtype: Any
        ) -> np.ndarray:
            values: list[np.ndarray] = []
            for reply in replies:
                if field == "applied":
                    value = compact_applied(reply)
                elif field == "executed":
                    value = _compact_executed(reply["crfs_trace"])
                elif field == "final":
                    value = np.asarray(reply["crfs_trace"]["final_normalized"])
                elif field == "final_physical":
                    value = np.asarray(
                        reply["crfs_trace"]["final_normalized_physical"]
                    )
                elif field == "actions":
                    value = np.asarray(reply["actions"])
                else:
                    raise AssertionError(field)
                values.append(np.asarray(value, dtype=dtype))
            return np.ascontiguousarray(np.stack(values))

        def add_recurrence_group(
            destination: dict[str, np.ndarray],
            prefix: str,
            replies: list[Mapping[str, Any]],
        ) -> None:
            fields = {
                "step_index_i64": ("step_index_steps", np.int64),
                "time_f32": ("time_steps", np.float32),
                "active_bool": ("active_steps", np.bool_),
                "x_t_f32": ("x_t_steps", np.float32),
                "v_base_f32": ("v_base_steps", np.float32),
                "total_velocity_f32": ("total_velocity_steps", np.float32),
                "x_next_f32": ("x_next_steps", np.float32),
                "initial_noise_f32": ("initial_noise", np.float32),
            }
            for suffix, (leaf, dtype) in fields.items():
                destination[f"{prefix}_trace_{suffix}"] = np.ascontiguousarray(
                    np.stack(
                        [np.asarray(reply["crfs_trace"][leaf], dtype=dtype) for reply in replies]
                    )
                )

        zero_replies = [zero_before, zero_after]
        arm_a_replies = list(special_replies["A"])
        cem_replies = [
            {"actions": candidates[index].evaluation.returned_physical_action, "crfs_trace": search_traces[index]}
            for index in range(SEARCH_REQUESTS)
        ]
        arm_b_replies = list(special_replies["B"])
        arm_c_replies = list(special_replies["C"])
        selected_requested_c = (
            arm_a_requested_c
            if result.selected.pool_index == 0
            else np.asarray(candidates[result.selected.pool_index - 1].projected_increment, dtype=np.float32)
        )
        reverse_requested_c = np.ascontiguousarray(selected_requested_c[::-1])
        cem_requested_c = np.stack([item.projected_increment for item in candidates]).astype(
            np.float32, copy=False
        )
        cem_requested_u = np.stack([item.requested_velocity for item in candidates]).astype(
            np.float32, copy=False
        )
        generation_index = np.repeat(np.arange(8, dtype=np.int16), 65)
        population_index = np.tile(np.arange(65, dtype=np.int16), 8)
        pair_index = np.full((520,), -1, dtype=np.int16)
        sign = np.zeros((520,), dtype=np.int8)
        for generation in range(8):
            base = generation * 65
            for pair in range(32):
                pair_index[base + 1 + 2 * pair : base + 3 + 2 * pair] = pair
                sign[base + 1 + 2 * pair] = 1
                sign[base + 2 + 2 * pair] = -1

        arrays: dict[str, np.ndarray] = {
            "source_noise_f32": np.asarray(noise, dtype=np.float32),
            "source_frozen_final_f32": np.asarray(frozen_final, dtype=np.float32),
            "source_delta_model_f32": np.asarray(delta32, dtype=np.float32),
            "source_target_normalized_f32": np.asarray(target, dtype=np.float32),
            "source_model_to_physical_scale_f32": model_to_physical_scale,
            "source_target_physical_f64": np.asarray(target_physical, dtype=np.float64),
            "source_budget_f32": np.asarray(source_budget32, dtype=np.float32),
            "source_dt_f32": np.asarray(DT_FLOAT32, dtype=np.float32),
            "source_radius_f32": np.asarray(source_budget32 / np.float32(5.0), dtype=np.float32),
            "reference_compiled_actions_f64": np.stack(
                [compiled_before["actions"], compiled_after["actions"]]
            ).astype(np.float64),
            "reference_source_actions_f64": np.stack(
                [source_before["actions"], source_after["actions"]]
            ).astype(np.float64),
            "reference_eager_actions_f64": np.stack(
                [eager_before["actions"], eager_after["actions"]]
            ).astype(np.float64),
            "reference_eager_final_f32": np.stack(
                [eager_before["crfs_trace"]["final_normalized"], eager_after["crfs_trace"]["final_normalized"]]
            ).astype(np.float32),
            "zero_requested_c_f32": np.zeros((2, 5, 15), dtype=np.float32),
            "zero_requested_velocity_f32": np.zeros((2, 5, 15), dtype=np.float32),
            "zero_applied_velocity_f32": stack_reply(zero_replies, "applied", dtype=np.float32),
            # The registered zero arm is the exact positive-zero reference;
            # signed dt*0 telemetry remains reconstructable in its recurrence
            # trace but is canonicalized here exactly as the CPU contract asks.
            "zero_executed_c_f32": np.zeros((2, 5, 15), dtype=np.float32),
            "zero_final_normalized_f32": stack_reply(zero_replies, "final", dtype=np.float32),
            "zero_final_normalized_physical_f64": stack_reply(
                zero_replies, "final_physical", dtype=np.float64
            ),
            "zero_returned_actions_f64": stack_reply(zero_replies, "actions", dtype=np.float64),
            "arm_a_requested_c_f32": np.broadcast_to(arm_a_requested_c, (2, 5, 15)).copy(),
            "arm_a_requested_velocity_f32": np.broadcast_to(arm_a_velocity, (2, 5, 15)).copy(),
            "arm_a_applied_velocity_f32": stack_reply(arm_a_replies, "applied", dtype=np.float32),
            "arm_a_executed_c_f32": stack_reply(arm_a_replies, "executed", dtype=np.float32),
            "arm_a_final_normalized_f32": stack_reply(arm_a_replies, "final", dtype=np.float32),
            "arm_a_final_normalized_physical_f64": stack_reply(
                arm_a_replies, "final_physical", dtype=np.float64
            ),
            "arm_a_returned_actions_f64": stack_reply(arm_a_replies, "actions", dtype=np.float64),
            "cem_raw_normals_f64": np.asarray(result.raw_normals, dtype=np.float64),
            "cem_mean_state_f64": np.asarray(result.mean_states, dtype=np.float64),
            "cem_sigma_state_f64": np.asarray(result.sigma_states, dtype=np.float64),
            "cem_variance_after_f64": np.asarray(result.variance_states, dtype=np.float64),
            "cem_raw_proposal_f64": np.stack([item.raw_proposal for item in candidates]).astype(np.float64),
            "cem_projected_c_f32": cem_requested_c,
            "cem_requested_velocity_f32": cem_requested_u,
            "cem_applied_velocity_f32": stack_reply(cem_replies, "applied", dtype=np.float32),
            "cem_executed_c_f32": stack_reply(cem_replies, "executed", dtype=np.float32),
            "cem_final_normalized_f32": stack_reply(cem_replies, "final", dtype=np.float32),
            "cem_final_normalized_physical_f64": stack_reply(
                cem_replies, "final_physical", dtype=np.float64
            ),
            "cem_returned_actions_f64": stack_reply(cem_replies, "actions", dtype=np.float64),
            "cem_objective_f64": np.asarray([item.objective for item in candidates], dtype=np.float64),
            "cem_fidelity_metrics_f64": np.asarray([item.metrics for item in candidates], dtype=np.float64),
            "cem_gate_pass_bool": np.asarray([item.gates for item in candidates], dtype=np.bool_),
            "cem_energy_f64": np.asarray([item.energy for item in candidates], dtype=np.float64),
            "cem_generation_i16": generation_index,
            "cem_population_index_i16": population_index,
            "cem_pair_index_i16": pair_index,
            "cem_sign_i8": sign,
            "cem_elite_query_index_i64": np.asarray(result.elite_cem_query_indices, dtype=np.int64),
            "cem_query_elapsed_ns_u64": np.asarray(
                [max(1, int(float(row["elapsed_seconds"]) * 1_000_000_000.0)) for row in ledger_rows[6:526]],
                dtype=np.uint64,
            ),
            "arm_b_requested_c_f32": np.broadcast_to(selected_requested_c, (2, 5, 15)).copy(),
            "arm_b_requested_velocity_f32": np.broadcast_to(selected_velocity, (2, 5, 15)).copy(),
            "arm_b_applied_velocity_f32": stack_reply(arm_b_replies, "applied", dtype=np.float32),
            "arm_b_executed_c_f32": stack_reply(arm_b_replies, "executed", dtype=np.float32),
            "arm_b_final_normalized_f32": stack_reply(arm_b_replies, "final", dtype=np.float32),
            "arm_b_final_normalized_physical_f64": stack_reply(
                arm_b_replies, "final_physical", dtype=np.float64
            ),
            "arm_b_returned_actions_f64": stack_reply(arm_b_replies, "actions", dtype=np.float64),
            "arm_c_requested_c_f32": np.broadcast_to(reverse_requested_c, (2, 5, 15)).copy(),
            "arm_c_requested_velocity_f32": np.broadcast_to(reversed_velocity, (2, 5, 15)).copy(),
            "arm_c_applied_velocity_f32": stack_reply(arm_c_replies, "applied", dtype=np.float32),
            "arm_c_executed_c_f32": stack_reply(arm_c_replies, "executed", dtype=np.float32),
            "arm_c_final_normalized_f32": stack_reply(arm_c_replies, "final", dtype=np.float32),
            "arm_c_final_normalized_physical_f64": stack_reply(
                arm_c_replies, "final_physical", dtype=np.float64
            ),
            "arm_c_returned_actions_f64": stack_reply(arm_c_replies, "actions", dtype=np.float64),
            "selected_pool_index_i64": np.asarray(result.selected.pool_index, dtype=np.int64),
        }
        add_recurrence_group(arrays, "zero", zero_replies)
        add_recurrence_group(arrays, "arm_a", arm_a_replies)
        add_recurrence_group(arrays, "cem", cem_replies)
        add_recurrence_group(arrays, "arm_b", arm_b_replies)
        add_recurrence_group(arrays, "arm_c", arm_c_replies)
        _atomic_write_npz(tensor_path, arrays)

        tensor_hash = file_sha256(tensor_path)
        ledger_payload = {
            "schema_version": "1.0",
            "payload_type": "r05a_actual_forward_query_ledger",
            "case_id": CASE_ID,
            "exact_policy_request_count": policy_index,
            "rows": ledger_rows,
            "tensor_archive_sha256": tensor_hash,
        }
        atomic_write_json(ledger_path, ledger_payload)
        ledger_hash = file_sha256(ledger_path)
        arm_a_pass = result.arm_a.passed
        b_changed_requested = selected_velocity.tobytes() != arm_a_velocity.tobytes()
        b_changed_applied = (
            arrays["arm_b_applied_velocity_f32"][0].tobytes()
            != arrays["arm_a_applied_velocity_f32"][0].tobytes()
        )
        b_changed = bool(b_changed_requested and b_changed_applied)
        if arm_a_pass:
            status = "baseline_sufficient_no_incremental_support"
        elif result.selected.passed and b_changed:
            status = "mechanism_pass"
        else:
            status = "frozen_cem_negative"
        payload = {
            "schema_version": "1.0",
            "payload_type": PAYLOAD_TYPE,
            "run_id": legacy_config.oracle.run_id,
            "status": status,
            "case_id": CASE_ID,
            "group_id": GROUP_ID,
            "source": {
                "r02_path": str(source_path),
                "r02_sha256": source_sha,
                "delta_star_model": delta_record,
                "source_budget_float64": source_budget64,
                "source_budget_float32": float(source_budget32),
                "pairing_checks": pairing_checks,
            },
            "source_pairing_records": {
                "policy_observation_source": source_pairing.get("policy_observation"),
                "policy_observation_current": observation_fingerprint,
                "branch_snapshot_source": source_pairing.get("branch_snapshot"),
                "branch_snapshot_current": branch_snapshot,
                "eager_actions_source": source_pairing.get("eager_actions"),
                "eager_actions_before_current": _array_record(
                    source_before["actions"]
                ),
                "eager_actions_after_current": _array_record(
                    source_after["actions"]
                ),
                "eager_trace_source": source_pairing.get("eager_trace"),
                "eager_trace_before_current": _trace_record(
                    source_before["crfs_trace"]
                ),
                "eager_trace_after_current": _trace_record(
                    source_after["crfs_trace"]
                ),
            },
            "configs": {
                "actual_path": str(actual_path),
                "actual_sha256": file_sha256(actual_path),
                "legacy_path": str(legacy_path),
                "legacy_sha256": file_sha256(legacy_path),
                "repo_root": str(root),
            },
            "request_accounting": {
                "exact": policy_index == EXACT_POLICY_REQUESTS,
                "total": policy_index,
                "cem": len(candidates),
                "trace_rows": len(trace_policy_indices),
            },
            "search_runtime": {
                "numpy_version": result.numpy_version,
                "bit_generator": "PCG64",
                "seed": 20260716,
                "generations": 8,
                "population_size": 65,
                "search_requests": SEARCH_REQUESTS,
            },
            "request_ledger": ledger_rows,
            "tensor_archive": {
                "path": str(tensor_path),
                "sha256": tensor_hash,
            },
            "provenance": {
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
                "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
                "host": socket.gethostname(),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            },
            "selection": {
                "source": result.selected.source,
                "cem_query_index": result.selected.cem_query_index,
                "global_policy_request_index": result.selected.global_policy_request_index,
                "objective": result.selected.objective,
                "energy": result.selected.energy,
                "metrics": list(result.selected.metrics),
                "gates": list(result.selected.gates),
                "passed": result.selected.passed,
                "arm_a_passed": arm_a_pass,
                "b_changed_from_arm_a_requested_bytes": b_changed_requested,
                "b_changed_from_arm_a_applied_bytes": b_changed_applied,
                "b_changed_from_arm_a": b_changed,
            },
            "post_search_checks": post_checks,
            "artifacts": {
                "tensor_archive": tensor_path.name,
                "tensor_archive_sha256": tensor_hash,
                "query_ledger": ledger_path.name,
                "query_ledger_sha256": ledger_hash,
            },
            "execution_boundary": {
                "policy_requests": policy_index,
                "policy_generated_action_steps_executed": 0,
                "teacher_generated_action_steps_executed": 0,
                "efficacy_rollouts_executed": 0,
                "simulator_efficacy_evaluated": False,
                "training": False,
            },
        }
        atomic_write_json(payload_path, payload)
        return payload_path, status
    except (ActualForwardSearchError, R05ACanaryPolicyError) as error:
        raise ActualForwardCanaryError(str(error)) from error
    finally:
        if owns_environment:
            environment.close()
