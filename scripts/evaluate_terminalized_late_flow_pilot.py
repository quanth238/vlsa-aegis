#!/usr/bin/env python3
"""Execute only ordinary, post-hoc-selected, and terminalized-flow-selected chunks."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping, Optional, Sequence

from scripts.collect_distal_exact_group_boundary import collect
from scripts.evaluate_terminal_branch_sampler_canary import _archived_action
from scripts.replay_distal_three_ellipsoid_multicbf import (
    ARCHIVED_FILE_SHA256,
    ARCHIVED_PAYLOAD_SHA256,
    CASE_ID,
    EXPECTED_ACTION_HORIZON,
    _atomic_write,
    _file_sha256,
    _git_identity,
    _load,
    _require,
)


def _selected_definition(
    *, name: str, order: int, actions: Any, ordinary: Any,
    source_candidate: str,
) -> dict[str, Any]:
    import numpy as np

    value = np.asarray(actions, dtype=np.float64)
    baseline = np.asarray(ordinary, dtype=np.float64)
    _require(value.shape == baseline.shape == (5, 7), "pilot action shape differs")
    correction = value[:, :3] - baseline[:, :3]
    return {
        "name": str(name),
        "order": int(order),
        "direction": None,
        "spatial_coefficients": None,
        "sign": 0 if order == 0 else None,
        "temporal_profile": None if order == 0 else "terminal_selected",
        "requested_alpha": float(np.linalg.norm(correction)),
        "requested_correction_l2_action": float(np.linalg.norm(correction)),
        "applied_correction_l2_action": float(np.linalg.norm(correction)),
        "clipped": bool(np.max(np.abs(value[:, :3])) >= 1.0),
        "selection_source_candidate": str(source_candidate),
        "actions": value.tolist(),
    }


def _physical_summary(
    fresh: Mapping[str, Any], score_by_arm: Mapping[str, Any],
) -> dict[str, Any]:
    candidates = fresh["exact_case"]["candidates"]
    expected = [
        "nominal", "terminal_compact_selector",
        "late_flow_terminalized_compact_selector",
    ]
    _require([row["name"] for row in candidates] == expected,
             "pilot executed arm order differs")
    physical_groups = ("palm", "L5", "L6", "L7")
    records = []
    for candidate in candidates:
        target = candidate["exact_group_target"]
        known = bool(target["known_outcome"])
        risks = {
            group: float(value)
            for group, value in target["group_future_violation"].items()
        }
        contacts = {
            group: int(value)
            for group, value in target["group_contact_sample_count"].items()
        }
        raw_collision = bool(
            int(candidate["source_raw_protected_contact_count"]) > 0
            or int(candidate["raw_protected_contact_sample_count"]) > 0
            or any(contacts[group] > 0 for group in physical_groups)
        )
        represented_violation = bool(
            any(risks[group] > 0.0 for group in physical_groups)
        )
        safe = bool(
            known
            and candidate["source_terminal_status"] == "SAFE_TERMINAL"
            and all(risks[group] <= 0.0 for group in physical_groups)
            and all(contacts[group] == 0 for group in physical_groups)
            and int(candidate["source_raw_protected_contact_count"]) == 0
            and int(candidate["raw_protected_contact_sample_count"]) == 0
            and not bool(candidate["source_physical_veto"])
            and not bool(candidate["replayed_physical_veto"])
        )
        failed = [group for group in physical_groups if risks[group] > 0.0]
        records.append({
            "arm": str(candidate["name"]),
            "selection": score_by_arm[str(candidate["name"])],
            "known_outcome": known,
            "terminal_status": str(candidate["source_terminal_status"]),
            "raw_physical_collision": raw_collision,
            "represented_physical_violation": represented_violation,
            "physical_safe": safe,
            "failed_physical_groups": failed,
            "group_future_violation": risks,
            "group_contact_sample_count": contacts,
            "source_raw_protected_contact_count": int(
                candidate["source_raw_protected_contact_count"]
            ),
            "replayed_raw_protected_contact_sample_count": int(
                candidate["raw_protected_contact_sample_count"]
            ),
            "source_maximum_CAR_m": float(candidate["source_maximum_CAR_m"]),
            "replayed_maximum_CAR_m": float(
                candidate["replayed_maximum_CAR_m"]
            ),
            "executed_actions_sha256": hashlib.sha256(
                json.dumps(
                    candidate["source_executed_actions"],
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
        })
    nominal = records[0]
    return {
        "state_hash_matches": bool(fresh["exact_case"]["state_hash_matches"]),
        "source_replay_exact": bool(fresh["exact_case"]["source_replay_exact"]),
        "known_outcome_count": sum(row["known_outcome"] for row in records),
        "unknown_outcome_count": sum(not row["known_outcome"] for row in records),
        "physical_safe_count": sum(row["physical_safe"] for row in records),
        "raw_physical_collision_count": sum(
            row["raw_physical_collision"] for row in records
        ),
        "posthoc_collision_avoided": bool(
            not nominal["physical_safe"] and records[1]["physical_safe"]
        ),
        "late_flow_collision_avoided": bool(
            not nominal["physical_safe"] and records[2]["physical_safe"]
        ),
        "posthoc_raw_collision_avoided": bool(
            nominal["raw_physical_collision"]
            and records[1]["known_outcome"]
            and not records[1]["raw_physical_collision"]
        ),
        "late_flow_raw_collision_avoided": bool(
            nominal["raw_physical_collision"]
            and records[2]["known_outcome"]
            and not records[2]["raw_physical_collision"]
        ),
        "records": records,
    }


def run(
    *, repo_root: Path, manifest_path: Path, table1_root: Path,
    archived_path: Path, config_path: Path, expected_commit: str,
    replica: str, host: str, port: int, run_root: Path,
    frozen_producer_result_path: Optional[Path] = None,
    accepted_producer_commit: Optional[str] = None,
) -> dict[str, Any]:
    import numpy as np

    from main.evaluate_safelibero_aegis import (
        TABLE_RENDER_RESOLUTION,
        TABLE_SETTLE_ACTIONS,
        _active_obstacle,
        _build_environment,
        _policy_observation,
        _runtime_imports,
        _settle,
        array_sha256,
        query_seed,
        read_jsonl,
        validate_case_row,
    )
    from main.multilink_ellipsoid.compact_safety_coordinate_q import (
        payload_sha256 as compact_payload_sha256,
    )
    from main.multilink_ellipsoid.exact_group_boundary import (
        payload_sha256 as exact_payload_sha256,
    )
    from main.multilink_ellipsoid.rollout import _dynamic_state_vector
    from main.multilink_ellipsoid.terminal_branching import (
        FROZEN_CANDIDATE_NAMES,
        PILOT_RESULT_SCHEMA,
        build_envelope,
        load_config,
        payload_sha256,
        score_terminal_bank,
    )

    started = time.perf_counter_ns()
    config = load_config(config_path)
    source = _git_identity(repo_root, expected_commit)
    mechanism = config["mechanism_case"]
    _require(replica in ("producer", "replay"), "pilot replica differs")

    canary_path = Path(config["validated_sampler_canary"]["path"])
    _require(
        _file_sha256(canary_path)
        == config["validated_sampler_canary"]["file_sha256"],
        "pilot sampler validation file differs",
    )
    canary = _load(canary_path)
    _require(
        canary.get("status") == "passing"
        and canary.get("validation_payload_sha256")
        == config["validated_sampler_canary"]["payload_sha256"],
        "pilot sampler validation payload differs",
    )

    context_path = Path(mechanism["exact_context_path"])
    _require(_file_sha256(context_path) == mechanism["exact_context_file_sha256"],
             "pilot exact context file differs")
    context_record = _load(context_path)
    _require(
        context_record.get("result_payload_sha256")
        == mechanism["exact_context_payload_sha256"]
        == exact_payload_sha256(context_record),
        "pilot exact context payload differs",
    )
    exact_case = context_record["exact_case"]
    _require(
        context_record["case_id"] == CASE_ID
        and int(context_record["state_step"]) == int(mechanism["state_step"])
        and exact_case["source_snapshot_sha256"]
        == mechanism["source_snapshot_sha256"],
        "pilot exact context state differs",
    )

    compact_binding = config["terminal_risk_model"]
    compact_path = Path(compact_binding["path"])
    _require(_file_sha256(compact_path) == compact_binding["file_sha256"],
             "pilot compact result file differs")
    compact = _load(compact_path)
    compact_model = compact["compact_shared_7D"]["model"]
    _require(
        compact.get("result_payload_sha256")
        == compact_binding["payload_sha256"]
        == compact_payload_sha256(compact, "result_payload_sha256")
        and compact_model["model_sha256"] == compact_binding["model_sha256"],
        "pilot compact result payload differs",
    )

    _require(_file_sha256(archived_path) == ARCHIVED_FILE_SHA256,
             "pilot Table-1 file differs")
    archived = _load(archived_path)
    _require(archived.get("result_payload_sha256") == ARCHIVED_PAYLOAD_SHA256,
             "pilot Table-1 payload differs")
    archived_actions = archived.get("actions")
    _require(isinstance(archived_actions, list)
             and len(archived_actions) == EXPECTED_ACTION_HORIZON,
             "pilot Table-1 horizon differs")
    rows = read_jsonl(manifest_path)
    matches = [row for row in rows if row.get("case_id") == CASE_ID]
    _require(len(matches) == 1, "pilot manifest case differs")
    case = matches[0]
    validate_case_row(case, repo_root)

    source_config = repo_root / mechanism["source_config"]
    _require(_file_sha256(source_config) == mechanism["source_config_file_sha256"],
             "pilot source config differs")

    if frozen_producer_result_path is not None:
        _require(replica == "replay", "frozen pilot actions require replay replica")
        _require(accepted_producer_commit is not None,
                 "frozen pilot producer commit is missing")
        producer_file_sha256 = _file_sha256(frozen_producer_result_path)
        producer = _load(frozen_producer_result_path)
        _require(
            producer.get("schema_version") == PILOT_RESULT_SCHEMA
            and producer.get("status") == "complete"
            and producer.get("replica") == "producer"
            and producer.get("source", {}).get("commit")
            == accepted_producer_commit
            and producer.get("config_file_sha256")
            == config["config_file_sha256"]
            and producer.get("config_payload_sha256")
            == config["config_payload_sha256"]
            and producer.get("result_payload_sha256")
            == payload_sha256(producer, "result_payload_sha256"),
            "frozen pilot producer result differs",
        )
        producer_view = producer["scientific_view"]
        producer_candidates = producer[
            "fresh_selected_arm_execution"
        ]["exact_case"]["candidates"]
        expected_names = [
            "nominal", "terminal_compact_selector",
            "late_flow_terminalized_compact_selector",
        ]
        _require(
            [row["name"] for row in producer_candidates] == expected_names,
            "frozen pilot producer arms differ",
        )
        ordinary_effective = np.asarray(
            producer_candidates[0]["source_executed_actions"],
            dtype=np.float64,
        )
        producer_outcomes = {
            row["arm"]: row for row in producer_view["outcome"]["records"]
        }
        definitions = [
            _selected_definition(
                name=name, order=index,
                actions=producer_candidates[index]["source_executed_actions"],
                ordinary=ordinary_effective,
                source_candidate=producer_outcomes[name]["selection"][
                    "source_candidate"
                ],
            )
            for index, name in enumerate(expected_names)
        ]
        fresh = collect(
            repo_root=repo_root,
            table1_root=table1_root,
            config_path=source_config,
            case_index=int(mechanism["source_case_index"]),
            expected_commit=expected_commit,
            run_root=run_root / "selected-arm-execution",
            candidate_workers=1,
            source_only=False,
            candidate_definitions_override=definitions,
        )
        _require(
            fresh.get("status") == "complete" and fresh["case_id"] == CASE_ID,
            "frozen pilot selected-arm replay differs",
        )
        scores = {
            name: copy.deepcopy(producer_outcomes[name]["selection"])
            for name in expected_names
        }
        outcome = _physical_summary(fresh, scores)
        scientific_view = copy.deepcopy(producer_view)
        scientific_view["outcome"] = outcome
        _require(
            scientific_view == producer_view,
            "frozen selected actions do not replay exactly",
        )
        result = {
            "schema_version": PILOT_RESULT_SCHEMA,
            "status": "complete",
            "scientific_result": True,
            "replica": replica,
            "source": source,
            "config_file_sha256": config["config_file_sha256"],
            "config_payload_sha256": config["config_payload_sha256"],
            "compact_model": compact_binding,
            "validated_sampler_canary": config["validated_sampler_canary"],
            "frozen_producer_binding": {
                "path": str(frozen_producer_result_path),
                "file_sha256": producer_file_sha256,
                "result_payload_sha256": producer["result_payload_sha256"],
                "source_commit": accepted_producer_commit,
            },
            "scientific_view": scientific_view,
            "fresh_selected_arm_execution": fresh,
            "timing": {
                "total_wall_seconds": (
                    time.perf_counter_ns() - started
                ) * 1.0e-9,
            },
            "correction_safety_authorized": False,
            "formal_safety_claim": False,
        }
        result["result_payload_sha256"] = payload_sha256(
            result, "result_payload_sha256"
        )
        return result

    runtime = _runtime_imports(include_aegis=False)
    env = None
    try:
        env, task, observation, _ = _build_environment(
            runtime, case, render_resolution=TABLE_RENDER_RESOLUTION
        )
        observation = _settle(env, observation, TABLE_SETTLE_ACTIONS)
        active_obstacle, _ = _active_obstacle(env, observation)
        _require(active_obstacle == context_record["selection"]["active_obstacle_name"],
                 "pilot active obstacle differs")
        for step in range(int(mechanism["state_step"])):
            observation, _, done, _ = env.step(
                _archived_action(archived_actions, step).tolist()
            )
            _require(not bool(done), "pilot prefix completed before warning state")
        state_hash = hashlib.sha256(
            np.asarray(_dynamic_state_vector(env), dtype=np.float64).tobytes()
        ).hexdigest()
        _require(state_hash == mechanism["source_snapshot_sha256"],
                 "pilot live warning state differs")
        seed = query_seed(int(case["policy_noise_seed"]), int(mechanism["query_index"]))
        client = runtime["websocket_client_policy"].WebsocketClientPolicy(host, port)
        ordinary_input = _policy_observation(
            runtime, observation, task_description=str(task.language),
            resize_size=224, rng_seed=seed,
        )
        ordinary_response = client.infer(ordinary_input)
        ordinary = np.asarray(ordinary_response["actions"], dtype=np.float64)
        _require(ordinary.shape == (10, 7), "pilot ordinary action differs")

        stored_nominal = np.asarray(
            exact_case["source_nominal_five_action_chunk"], dtype=np.float64
        )
        stored_candidates = exact_case["candidates"]
        _require(
            [row["name"] for row in stored_candidates]
            == list(FROZEN_CANDIDATE_NAMES),
            "pilot stored candidate bank differs",
        )
        posthoc_bank = np.repeat(ordinary[None, :, :], 13, axis=0)
        for index, candidate in enumerate(stored_candidates):
            stored_effective = np.asarray(
                candidate["source_executed_actions"], dtype=np.float64
            )
            residual = stored_effective[:, :3] - stored_nominal[:, :3]
            posthoc_bank[index, :5, :3] = np.clip(
                ordinary[:5, :3] + residual, -1.0, 1.0
            )
        envelope = build_envelope(
            ordinary, posthoc_bank, FROZEN_CANDIDATE_NAMES,
            branch_after_euler_step=int(config["flow"]["branch_after_euler_step"]),
        )
        branch_input = _policy_observation(
            runtime, observation, task_description=str(task.language),
            resize_size=224, rng_seed=seed,
        )
        branch_input["__crfs__"]["terminal_branching"] = envelope
        branch_response = client.infer(branch_input)
        diagnostic = branch_response["terminal_branching"]
        late_bank = np.asarray(diagnostic["terminal_output_actions"], dtype=np.float64)
        _require(late_bank.shape == (13, 10, 7), "pilot late terminal bank differs")
        _require(float(np.max(np.abs(late_bank[0] - ordinary))) == 0.0,
                 "pilot branch zero differs")

        posthoc_score = score_terminal_bank(
            exact_case=exact_case,
            ordinary_terminal_actions=ordinary,
            terminal_action_bank=posthoc_bank,
            candidate_names=FROZEN_CANDIDATE_NAMES,
            state_payload=compact_model["state_payload"],
        )
        late_score = score_terminal_bank(
            exact_case=exact_case,
            ordinary_terminal_actions=ordinary,
            terminal_action_bank=late_bank,
            candidate_names=FROZEN_CANDIDATE_NAMES,
            state_payload=compact_model["state_payload"],
        )
        ordinary_effective = ordinary[:5].copy()
        ordinary_effective[:, :3] = np.clip(ordinary_effective[:, :3], -1.0, 1.0)
        definitions = [
            _selected_definition(
                name="nominal", order=0, actions=ordinary_effective,
                ordinary=ordinary_effective, source_candidate="nominal",
            ),
            _selected_definition(
                name="terminal_compact_selector", order=1,
                actions=posthoc_score["selected_effective_first_five_actions"],
                ordinary=ordinary_effective,
                source_candidate=posthoc_score["selected_candidate"],
            ),
            _selected_definition(
                name="late_flow_terminalized_compact_selector", order=2,
                actions=late_score["selected_effective_first_five_actions"],
                ordinary=ordinary_effective,
                source_candidate=late_score["selected_candidate"],
            ),
        ]
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass

    fresh = collect(
        repo_root=repo_root,
        table1_root=table1_root,
        config_path=source_config,
        case_index=int(mechanism["source_case_index"]),
        expected_commit=expected_commit,
        run_root=run_root / "selected-arm-execution",
        candidate_workers=1,
        source_only=False,
        candidate_definitions_override=definitions,
    )
    _require(fresh.get("status") == "complete" and fresh["case_id"] == CASE_ID,
             "pilot selected-arm execution differs")
    scores = {
        "nominal": {
            "source_candidate": "nominal",
            "selected_predicted_primary": float(
                posthoc_score["records"][0]["predicted_primary"]
            ),
        },
        "terminal_compact_selector": {
            "source_candidate": posthoc_score["selected_candidate"],
            "selected_predicted_primary": posthoc_score["selected_predicted_primary"],
        },
        "late_flow_terminalized_compact_selector": {
            "source_candidate": late_score["selected_candidate"],
            "selected_predicted_primary": late_score["selected_predicted_primary"],
        },
    }
    outcome = _physical_summary(fresh, scores)
    scientific_view = {
        "case_id": CASE_ID,
        "state_step": int(mechanism["state_step"]),
        "query_index": int(mechanism["query_index"]),
        "rng_seed": seed,
        "live_state_sha256": state_hash,
        "ordinary_action_sha256": array_sha256(ordinary),
        "posthoc_terminal_bank_sha256": array_sha256(posthoc_bank),
        "late_flow_terminal_bank_sha256": array_sha256(late_bank),
        "posthoc_score": posthoc_score,
        "late_flow_score": late_score,
        "outcome": outcome,
        "gates": {
            "validated_sampler_bound": True,
            "live_state_exact": state_hash == mechanism["source_snapshot_sha256"],
            "branch_zero_exact": float(np.max(np.abs(late_bank[0] - ordinary))) == 0.0,
            "three_selected_arms_only": len(fresh["exact_case"]["candidates"]) == 3,
            "no_QP": fresh["original_AEGIS_EE_QP_enabled"] is False
            and fresh["learned_correction_QP_enabled"] is False,
            "selection_did_not_use_outcome": True,
        },
    }
    result = {
        "schema_version": PILOT_RESULT_SCHEMA,
        "status": "complete",
        "scientific_result": True,
        "replica": replica,
        "source": source,
        "config_file_sha256": config["config_file_sha256"],
        "config_payload_sha256": config["config_payload_sha256"],
        "compact_model": compact_binding,
        "validated_sampler_canary": config["validated_sampler_canary"],
        "scientific_view": scientific_view,
        "fresh_selected_arm_execution": fresh,
        "timing": {
            "total_wall_seconds": (time.perf_counter_ns() - started) * 1.0e-9,
        },
        "correction_safety_authorized": False,
        "formal_safety_claim": False,
    }
    result["result_payload_sha256"] = payload_sha256(
        result, "result_payload_sha256"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--table1-root", type=Path, required=True)
    parser.add_argument("--archived", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--replica", choices=("producer", "replay"), required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frozen-producer-result", type=Path)
    parser.add_argument("--accepted-producer-commit")
    args = parser.parse_args()
    result = run(
        repo_root=args.repo_root.resolve(), manifest_path=args.manifest.resolve(),
        table1_root=args.table1_root.resolve(), archived_path=args.archived.resolve(),
        config_path=args.config.resolve(), expected_commit=args.expected_commit,
        replica=args.replica, host=args.host, port=args.port,
        run_root=args.run_root.resolve(),
        frozen_producer_result_path=(
            None if args.frozen_producer_result is None
            else args.frozen_producer_result.resolve()
        ),
        accepted_producer_commit=args.accepted_producer_commit,
    )
    _atomic_write(args.output.resolve(), result)
    print(json.dumps({
        "status": result["status"], "replica": result["replica"],
        "outcome": result["scientific_view"]["outcome"],
        "result_payload_sha256": result["result_payload_sha256"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
