#!/usr/bin/env python3
"""Run the frozen nominal-first detour-and-rejoin shield for one episode."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping, Optional, Sequence

from main.multilink_ellipsoid.tight_prefix_full_episode import (
    canonical,
    file_sha256,
    payload_sha256,
    scientific_view,
)
from main.multilink_ellipsoid.tight_prefix_nominal_first_full_episode import (
    DETOUR_CANDIDATE_NAMES,
    RESULT_SCHEMA,
    detour_and_rejoin_candidates,
    load_config,
)
from scripts.evaluate_terminalized_full_episode import _run_episode
from scripts.evaluate_terminalized_late_flow_task_success import (
    _allocation_record,
)
from scripts.evaluate_tight_prefix_full_episode import LiveTightPrefixSelector
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write,
    _git_identity,
    _load,
    _require,
)


class LiveNominalFirstSelector(LiveTightPrefixSelector):
    """Use the critic only near geometry and minimize accepted intervention."""

    def select_nominal_first(
        self, *, observation: Mapping[str, Any], task_description: str,
        rng_seed: int, query_index: int,
    ) -> tuple[Any, dict[str, Any]]:
        import numpy as np

        from main.evaluate_safelibero_aegis import (
            _policy_observation,
            array_sha256,
        )
        from main.multilink_ellipsoid.terminal_branching import (
            score_terminal_bank,
        )

        request = _policy_observation(
            self.runtime, observation, task_description=task_description,
            resize_size=224, rng_seed=int(rng_seed),
        )
        started = time.perf_counter_ns()
        response = self.client.infer(request)
        wall_seconds = (time.perf_counter_ns() - started) * 1.0e-9
        ordinary = np.asarray(response["actions"], dtype=np.float64)
        _require(ordinary.shape == (10, 7),
                 "nominal-first ordinary chunk differs")
        effective_ordinary = ordinary.copy()
        effective_ordinary[:, :3] = np.clip(
            effective_ordinary[:, :3], -1.0, 1.0,
        )
        exact_case, frame = self._context_and_frame()
        slacks = exact_case["exact_group_target"][
            "initial_row_normalized_radial_slack"
        ]
        warning = self.method["warning"]
        warning_slack = min(float(slacks[row]) for row in warning["rows"])
        warning_triggered = bool(
            warning_slack
            < float(warning["normalized_radial_slack_strictly_below"])
        )
        common = {
            "query_index": int(query_index),
            "rng_seed": int(rng_seed),
            "ordinary_actions_sha256": array_sha256(ordinary),
            "active_row": frame["active_row"],
            "active_box": frame["active_box"],
            "active_slack": frame["active_slack"],
            "warning_slack": float(warning_slack),
            "warning_threshold": float(
                warning["normalized_radial_slack_strictly_below"]
            ),
            "warning_triggered": warning_triggered,
            "wall_seconds": float(wall_seconds),
            "server_timing": response.get("server_timing"),
        }
        if not warning_triggered:
            actions = effective_ordinary[:5].copy()
            return actions, {
                **common,
                "critic_invoked": False,
                "candidate_bank_sha256": array_sha256(actions[None, :, :]),
                "candidate_count": 1,
                "accepted_candidate_count": None,
                "abstained": False,
                "nominal_accepted": True,
                "intervened": False,
                "selected_candidate": "nominal",
                "selected_candidate_order": 0,
                "selected_predicted_primary": None,
                "selected_predicted_L6_diagnostic": None,
                "selected_effective_correction_l2_action": 0.0,
                "selected_effective_correction_sum_xyz": [0.0, 0.0, 0.0],
                "selected_endpoint_preserved": True,
                "selected_effective_first_five_sha256": array_sha256(actions),
            }

        candidate_definitions = detour_and_rejoin_candidates(
            effective_ordinary[:5], frame,
            magnitudes=self.method["candidate_generation"]["magnitudes"],
            action_limit=float(
                self.method["candidate_generation"]["action_limit"]
            ),
        )
        terminal_bank = np.repeat(
            effective_ordinary[None, :, :], len(candidate_definitions), axis=0,
        )
        terminal_bank[:, :5, :] = np.asarray(
            [row["actions"] for row in candidate_definitions],
            dtype=np.float64,
        )
        score = score_terminal_bank(
            exact_case=exact_case,
            ordinary_terminal_actions=effective_ordinary,
            terminal_action_bank=terminal_bank,
            candidate_names=DETOUR_CANDIDATE_NAMES,
            required_candidate_names=DETOUR_CANDIDATE_NAMES,
            state_payload=self.state_payload,
            primary_rows=self.method["primary_rows"],
            model_rows=self.method["model_rows"],
            translation_scale=float(
                self.method["translation_scale_m_per_action_unit"]
            ),
        )
        margin = float(self.method["acceptance"]["margin"])
        accepted = [
            row for row in score["records"]
            if all(
                float(row["predicted_by_row"][str(index)]) <= -margin
                for index in self.method["primary_rows"]
            )
        ]
        nominal = next(
            row for row in score["records"]
            if row["candidate_name"] == "nominal"
        )
        if nominal in accepted:
            selected = nominal
        elif accepted:
            selected = min(accepted, key=lambda row: (
                float(row["effective_correction_l2_action"]),
                int(row["candidate_order"]),
            ))
        else:
            selected = None
        if selected is None:
            return None, {
                **common,
                "critic_invoked": True,
                "candidate_bank_sha256": array_sha256(terminal_bank),
                "candidate_count": len(candidate_definitions),
                "accepted_candidate_count": 0,
                "acceptance_margin": margin,
                "nominal_predicted_primary": float(
                    nominal["predicted_primary"]
                ),
                "abstained": True,
                "nominal_accepted": False,
                "intervened": False,
                "selected_candidate": None,
                "selected_candidate_order": None,
                "selected_predicted_primary": None,
                "selected_predicted_L6_diagnostic": None,
                "selected_effective_correction_l2_action": None,
                "selected_effective_correction_sum_xyz": None,
                "selected_endpoint_preserved": None,
                "selected_effective_first_five_sha256": None,
            }

        definition = candidate_definitions[int(selected["candidate_order"])]
        actions = np.asarray(
            selected["effective_first_five_actions"], dtype=np.float64,
        )
        correction_sum = np.sum(
            actions[:, :3] - effective_ordinary[:5, :3], axis=0,
        )
        endpoint_preserved = bool(
            float(np.max(np.abs(correction_sum))) <= 1.0e-12
            and definition["endpoint_preserved"] is True
        )
        _require(endpoint_preserved,
                 "nominal-first selected endpoint differs")
        by_row = selected["predicted_by_row"]
        l6 = max(
            float(by_row[str(row)])
            for row in self.method["diagnostic_rows"]
        )
        correction = float(selected["effective_correction_l2_action"])
        return actions, {
            **common,
            "critic_invoked": True,
            "candidate_bank_sha256": array_sha256(terminal_bank),
            "candidate_count": len(candidate_definitions),
            "accepted_candidate_count": len(accepted),
            "acceptance_margin": margin,
            "nominal_predicted_primary": float(nominal["predicted_primary"]),
            "abstained": False,
            "nominal_accepted": selected is nominal,
            "intervened": correction > 1.0e-12,
            "selected_candidate": selected["candidate_name"],
            "selected_candidate_order": selected["candidate_order"],
            "selected_predicted_primary": selected["predicted_primary"],
            "selected_predicted_L6_diagnostic": float(l6),
            "selected_effective_correction_l2_action": correction,
            "selected_effective_correction_sum_xyz": correction_sum.tolist(),
            "selected_endpoint_preserved": endpoint_preserved,
            "selected_applied_detour_magnitude": float(
                definition["applied_magnitude"]
            ),
            "selected_effective_first_five_sha256": array_sha256(actions),
        }


def evaluate(
    *, repo_root: Path, manifest_path: Path, config_path: Path,
    expected_commit: str, replica: str, host: str, port: int,
    output_path: Path,
    frozen_producer_result_path: Optional[Path] = None,
) -> dict[str, Any]:
    from main.evaluate_safelibero_aegis import (
        _runtime_imports,
        read_jsonl,
        validate_case_row,
    )

    started = time.perf_counter_ns()
    config = load_config(config_path)
    case_id = str(config["case_id"])
    _require(replica in ("producer", "replay"),
             "nominal-first full-episode replica differs")
    _require(
        (replica == "replay") is (frozen_producer_result_path is not None),
        "nominal-first producer binding differs",
    )
    source = _git_identity(repo_root, expected_commit)
    allocation = _allocation_record(require_h100=replica == "producer")
    archive_path = Path(config["source"]["raw_pi05_archive"])
    _require(
        file_sha256(archive_path)
        == config["source"]["raw_pi05_archive_file_sha256"],
        "nominal-first raw archive differs",
    )
    archived = _load(archive_path)
    _require(
        archived.get("result_payload_sha256")
        == config["source"]["raw_pi05_archive_payload_sha256"],
        "nominal-first raw archive payload differs",
    )
    matches = [
        row for row in read_jsonl(manifest_path)
        if row.get("case_id") == case_id
    ]
    _require(len(matches) == 1,
             "nominal-first manifest case differs")
    case = matches[0]
    validate_case_row(case, repo_root)
    runtime = _runtime_imports(include_aegis=False)
    client = None
    server_identity = None
    frozen_episode = None
    producer_binding = None
    if replica == "producer":
        client = runtime["websocket_client_policy"].WebsocketClientPolicy(
            host, int(port),
        )
        from main.evaluate_safelibero_aegis import _server_identity
        server_identity = _server_identity(client)
    else:
        producer = _load(frozen_producer_result_path)
        _require(
            producer.get("schema_version") == RESULT_SCHEMA
            and producer.get("status") == "complete"
            and producer.get("replica") == "producer"
            and producer.get("result_payload_sha256")
            == payload_sha256(producer, "result_payload_sha256"),
            "nominal-first frozen producer differs",
        )
        frozen_episode = producer["episode"]
        producer_binding = {
            "path": str(frozen_producer_result_path),
            "file_sha256": file_sha256(frozen_producer_result_path),
            "payload_sha256": producer["result_payload_sha256"],
            "source_commit": producer["source"]["commit"],
        }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pairing, episode = _run_episode(
        repo_root=repo_root, runtime=runtime, case=case, archived=archived,
        config=config, client=client, frozen_episode=frozen_episode,
        output_root=output_path.parent / "episode",
        selector_factory=LiveNominalFirstSelector,
        selector_config=config,
        selector_method="select_nominal_first",
        selected_action_source="tight_prefix_nominal_first_detour_no_QP",
    )
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "complete",
        "scientific_result": True,
        "replica": replica,
        "source": source,
        "allocation": allocation,
        "config": config,
        "case_id": case_id,
        "pairing": pairing,
        "pairing_sha256": hashlib.sha256(canonical(pairing)).hexdigest(),
        "policy_server": server_identity,
        "producer_binding": producer_binding,
        "episode": episode,
        "candidate_simulator_rollout_count": 0,
        "model_retraining_performed": False,
        "QP_executed": False,
        "population_claim_authorized": False,
        "formal_safety_claim": False,
        "wall_seconds": (time.perf_counter_ns() - started) * 1.0e-9,
    }
    result["scientific_view"] = scientific_view(result)
    if frozen_producer_result_path is not None:
        producer = _load(frozen_producer_result_path)
        _require(result["scientific_view"] == producer["scientific_view"],
                 "nominal-first scientific replay differs")
    result["result_payload_sha256"] = payload_sha256(
        result, "result_payload_sha256",
    )
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--replica", choices=("producer", "replay"), required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frozen-producer-result", type=Path)
    args = parser.parse_args(argv)
    result = evaluate(
        repo_root=args.repo_root.resolve(),
        manifest_path=args.manifest.resolve(),
        config_path=args.config.resolve(),
        expected_commit=args.expected_commit,
        replica=args.replica,
        host=args.host,
        port=args.port,
        output_path=args.output.resolve(),
        frozen_producer_result_path=(
            None if args.frozen_producer_result is None
            else args.frozen_producer_result.resolve()
        ),
    )
    _atomic_write(args.output.resolve(), result)
    print(json.dumps({
        "status": result["status"],
        "case_id": result["case_id"],
        "replica": result["replica"],
        "terminal_reason": result["episode"]["terminal_reason"],
        "action_count": result["episode"]["action_count"],
        "selection_count": result["episode"]["selection_count"],
        "native_task_success": result["episode"]["native_task_success"],
        "collision_free_task_success": result["episode"][
            "collision_free_task_success"
        ],
        "result_payload_sha256": result["result_payload_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
