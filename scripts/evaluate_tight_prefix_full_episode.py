#!/usr/bin/env python3
"""Run one full episode with the frozen tight-prefix minimum-risk selector."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping, Optional, Sequence

from main.multilink_ellipsoid.tight_prefix_full_episode import (
    CASE_ID, RESULT_SCHEMA, canonical, file_sha256, load_config,
    payload_sha256, scientific_view,
)
from scripts.evaluate_terminalized_full_episode import _run_episode
from scripts.evaluate_terminalized_late_flow_task_success import (
    _allocation_record,
)
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _git_identity, _load, _require,
)


class LiveTightPrefixSelector:
    """Generate and score the frozen state-conditioned bank without rollouts."""

    def __init__(
        self, *, repo_root: Path, runtime: Mapping[str, Any], env: Any,
        obstacle_name: str, archived: Mapping[str, Any],
        config: Mapping[str, Any], client: Any,
    ) -> None:
        import numpy as np

        from main.multilink_ellipsoid.exact_group_boundary import (
            load_config as load_exact_config,
        )
        from main.multilink_ellipsoid.palm_primitive_audit import (
            fit_compiled_contact_geom,
        )
        from main.multilink_ellipsoid.shadow import (
            MultilinkEllipsoidShadow, load_shadow_config,
        )
        from main.multilink_ellipsoid.tight_prefix_risk_dataset import (
            load_config as load_dataset_config,
        )
        from main.multilink_ellipsoid.tight_prefix_risk_q_diagnostic import (
            load_config as load_training_config,
            payload_sha256 as training_payload,
        )
        from scripts.audit_distal_palm_primitive_case import (
            _canonicalize_perception_ellipsoid_rotation,
        )

        del archived
        source = config["source"]
        training_config_path = repo_root / source["training_config"]
        _require(
            file_sha256(training_config_path)
            == source["training_config_file_sha256"],
            "tight-prefix full-episode training config differs",
        )
        training_config = load_training_config(training_config_path)
        _require(
            training_config["config_payload_sha256"]
            == source["training_config_payload_sha256"],
            "tight-prefix full-episode training config payload differs",
        )
        result_path = Path(source["training_result"])
        _require(
            file_sha256(result_path) == source["training_result_file_sha256"],
            "tight-prefix full-episode training result differs",
        )
        trained = _load(result_path)
        _require(
            trained.get("result_payload_sha256")
            == source["training_result_payload_sha256"]
            == training_payload(trained, "result_payload_sha256")
            and trained["compact_shared_7D"]["model"]["model_sha256"]
            == source["model_sha256"],
            "tight-prefix full-episode model differs",
        )
        if "acceptance" in config["method"]:
            _require(
                float(trained["compact_shared_7D"]["metrics"]["validation"]
                      ["primary"]["global"]["near_boundary_RMSE"])
                == float(config["method"]["acceptance"]["margin"]),
                "nominal-first empirical margin differs",
            )
        self.state_payload = trained["compact_shared_7D"]["model"][
            "state_payload"
        ]

        if "selector_validation" in source:
            selector_path = Path(source["selector_validation"])
            _require(
                file_sha256(selector_path)
                == source["selector_validation_file_sha256"],
                "tight-prefix full-episode selector validation differs",
            )
            selector = _load(selector_path)
            _require(
                selector.get("validation_payload_sha256")
                == source["selector_validation_payload_sha256"]
                and selector.get("frozen_rule_from_validation_only")
                == "minimum_predicted_primary_risk"
                and selector.get("opened_full_episode_pilot_supported") is True,
                "tight-prefix full-episode selector gate differs",
            )
        else:
            validation_path = Path(source["training_validation"])
            _require(
                file_sha256(validation_path)
                == source["training_validation_file_sha256"],
                "nominal-first training validation differs",
            )
            validation = _load(validation_path)
            _require(
                validation.get("validation_payload_sha256")
                == source["training_validation_payload_sha256"],
                "nominal-first training validation payload differs",
            )
            prior_path = Path(source["prior_full_episode_validation"])
            _require(
                file_sha256(prior_path)
                == source["prior_full_episode_validation_file_sha256"],
                "nominal-first prior full-episode validation differs",
            )
            prior = _load(prior_path)
            _require(
                prior.get("validation_payload_sha256")
                == source["prior_full_episode_validation_payload_sha256"]
                and prior.get("status")
                == "validated_exact_opened_full_episode"
                and prior.get("terminal_reason") == "timeout"
                and prior.get("native_task_success") is False,
                "nominal-first prior full-episode verdict differs",
            )

        dataset_path = repo_root / source["tight_dataset_config"]
        _require(
            file_sha256(dataset_path)
            == source["tight_dataset_config_file_sha256"],
            "tight-prefix full-episode dataset config differs",
        )
        dataset = load_dataset_config(dataset_path, repo_root=repo_root)
        _require(
            dataset["config_payload_sha256"]
            == source["tight_dataset_config_payload_sha256"],
            "tight-prefix full-episode dataset payload differs",
        )
        target = dataset["exact_group_target"]
        geometry_path = repo_root / target["robot_geometry_config"]
        geometry_archive_path = Path(source["geometry_archive"])
        _require(
            file_sha256(geometry_archive_path)
            == source["geometry_archive_file_sha256"],
            "tight-prefix full-episode geometry archive differs",
        )
        geometry_archive = _load(geometry_archive_path)
        _require(
            geometry_archive.get("result_payload_sha256")
            == source["geometry_archive_payload_sha256"],
            "tight-prefix full-episode geometry payload differs",
        )
        perception = geometry_archive.get("perception")
        _require(isinstance(perception, dict),
                 "tight-prefix full-episode perception differs")
        rotation, _ = _canonicalize_perception_ellipsoid_rotation(
            perception["mvee_rotation"]
        )
        self.shadow = MultilinkEllipsoidShadow.from_aegis_geometry(
            load_shadow_config(geometry_path),
            {
                "p2": perception["mvee_center"], "R2": rotation,
                "Q2_diag": perception["mvee_semiaxes"],
                "record": {"label": perception["obstacle_label"]},
            },
        )
        fit = target["compiled_contact_fit"]
        self.templates = [
            fit_compiled_contact_geom(
                env, str(item["geom_name"]),
                relative_padding=float(fit["relative_padding"]),
                tolerance=float(fit["khachiyan_tolerance"]),
                max_iterations=int(fit["khachiyan_max_iterations"]),
            )
            for item in target["compiled_contact_primitives"]
        ]
        distal = self.shadow._slabbed_links(
            env, include_certificates=True,
        )[:int(target["distal_row_count"])]
        _require(
            len(self.templates) == 5 and len(distal) == 5
            and all(
                bool((row.enclosure_certificate or {}).get("verified"))
                for row in self.templates + list(distal)
            ),
            "tight-prefix full-episode primitive certificate differs",
        )
        self.runtime = runtime
        self.env = env
        self.obstacle_name = str(obstacle_name)
        self.client = client
        self.method = config["method"]
        self.candidate_names = list(self.method["candidate_names"])
        self.group_rows = training_config["group_rows"]
        self.grid_config = {
            "finite_search": {
                "spatial_basis": ["normal", "tangent_up", "tangent_side"],
                "coefficient_grid": [-1, 0, 1],
                "correction_l2_action": 2.0,
                "action_limit": 1.0,
                "candidate_count_per_job": 27,
            },
        }
        self.np = np

    @staticmethod
    def _row_record(row: Any) -> dict[str, Any]:
        import numpy as np
        return {
            "body_name": str(row.body_name), "geom_name": str(row.geom_name),
            "bound_source": str(row.bound_source),
            "center_m": np.asarray(row.center, dtype=np.float64).tolist(),
            "rotation": np.asarray(row.rotation, dtype=np.float64).tolist(),
            "semiaxes_m": np.asarray(row.semiaxes_m, dtype=np.float64).tolist(),
        }

    def _context_and_frame(self) -> tuple[dict[str, Any], dict[str, Any]]:
        import numpy as np

        from main.multilink_ellipsoid.obstacle_proxy_audit import (
            compiled_obstacle_boxes, minimum_ellipsoid_quadratics_over_boxes,
        )
        from main.multilink_ellipsoid.palm_primitive_audit import world_ellipsoid

        rows = [world_ellipsoid(self.env, template) for template in self.templates]
        rows.extend(self.shadow._slabbed_links(self.env)[:5])
        boxes = compiled_obstacle_boxes(self.env, self.obstacle_name)
        quadratics = minimum_ellipsoid_quadratics_over_boxes(rows, boxes)
        row_index, box_index = np.unravel_index(
            int(np.argmin(quadratics)), quadratics.shape,
        )
        row = rows[int(row_index)]
        box = boxes[int(box_index)]
        normal = np.asarray(row.center) - np.asarray(box.center)
        normal /= float(np.linalg.norm(normal))
        tangent_up = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
        tangent_up -= float(tangent_up @ normal) * normal
        if float(np.linalg.norm(tangent_up)) <= 1.0e-12:
            fallback = np.eye(3)[int(np.argmin(np.abs(normal)))]
            tangent_up = fallback - float(fallback @ normal) * normal
        tangent_up /= float(np.linalg.norm(tangent_up))
        tangent_side = np.cross(normal, tangent_up)
        tangent_side /= float(np.linalg.norm(tangent_side))
        slacks = np.sqrt(np.min(quadratics, axis=1)) - 1.0
        exact_case = {
            "initial_exact_robot_rows": [self._row_record(item) for item in rows],
            "initial_compiled_obstacle_boxes": [item.to_record() for item in boxes],
            "source_nominal_five_action_chunk": [[0.0] * 7 for _ in range(5)],
            "exact_group_target": {
                "initial_row_normalized_radial_slack": slacks.tolist(),
            },
        }
        frame = {
            "normal": normal.tolist(), "tangent_up": tangent_up.tolist(),
            "tangent_side": tangent_side.tolist(),
            "active_row": int(row_index), "active_box": int(box_index),
            "active_slack": float(slacks[int(row_index)]),
        }
        return exact_case, frame

    def select_minimum_risk(
        self, *, observation: Mapping[str, Any], task_description: str,
        rng_seed: int, query_index: int,
    ) -> tuple[Any, dict[str, Any]]:
        import numpy as np

        from main.evaluate_safelibero_aegis import (
            _policy_observation, array_sha256,
        )
        from main.multilink_ellipsoid.active_boundary_search import (
            candidate_definitions,
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
                 "tight-prefix full-episode ordinary chunk differs")
        exact_case, frame = self._context_and_frame()
        full = candidate_definitions(
            ordinary[:5], frame, self.grid_config, "front_loaded",
        )
        by_name = {row["name"]: row for row in full}
        _require(not (set(self.candidate_names) - set(by_name)),
                 "tight-prefix full-episode candidate bank differs")
        selected_bank = [by_name[name] for name in self.candidate_names]
        terminal_bank = np.repeat(ordinary[None, :, :], 13, axis=0)
        terminal_bank[:, :5, :] = np.asarray(
            [row["actions"] for row in selected_bank], dtype=np.float64,
        )
        score = score_terminal_bank(
            exact_case=exact_case,
            ordinary_terminal_actions=ordinary,
            terminal_action_bank=terminal_bank,
            candidate_names=self.candidate_names,
            state_payload=self.state_payload,
            primary_rows=self.method["primary_rows"],
            model_rows=self.method["model_rows"],
            translation_scale=float(
                self.method["translation_scale_m_per_action_unit"]
            ),
        )
        selected = min(score["records"], key=lambda row: (
            float(row["predicted_primary"]),
            float(row["effective_correction_l2_action"]),
            int(row["candidate_order"]),
        ))
        actions = np.asarray(
            selected["effective_first_five_actions"], dtype=np.float64,
        )
        by_row = selected["predicted_by_row"]
        l6 = max(float(by_row[str(row)]) for row in self.method["diagnostic_rows"])
        return actions, {
            "query_index": int(query_index), "rng_seed": int(rng_seed),
            "ordinary_actions_sha256": array_sha256(ordinary),
            "candidate_bank_sha256": array_sha256(terminal_bank),
            "candidate_count": len(selected_bank),
            "active_row": frame["active_row"],
            "active_box": frame["active_box"],
            "active_slack": frame["active_slack"],
            "abstained": False,
            "selected_candidate": selected["candidate_name"],
            "selected_candidate_order": selected["candidate_order"],
            "selected_predicted_primary": selected["predicted_primary"],
            "selected_predicted_L6_diagnostic": float(l6),
            "selected_effective_correction_l2_action": selected[
                "effective_correction_l2_action"
            ],
            "selected_effective_first_five_sha256": array_sha256(actions),
            "wall_seconds": float(wall_seconds),
            "server_timing": response.get("server_timing"),
        }


def evaluate(
    *, repo_root: Path, manifest_path: Path, config_path: Path,
    expected_commit: str, replica: str, host: str, port: int,
    output_path: Path,
    frozen_producer_result_path: Optional[Path] = None,
) -> dict[str, Any]:
    from main.evaluate_safelibero_aegis import (
        _runtime_imports, read_jsonl, validate_case_row,
    )

    started = time.perf_counter_ns()
    config = load_config(config_path)
    _require(replica in ("producer", "replay"),
             "tight-prefix full-episode replica differs")
    _require(
        (replica == "replay") is (frozen_producer_result_path is not None),
        "tight-prefix full-episode producer binding differs",
    )
    source = _git_identity(repo_root, expected_commit)
    allocation = _allocation_record(require_h100=replica == "producer")
    archive_path = Path(config["source"]["raw_pi05_archive"])
    _require(
        file_sha256(archive_path)
        == config["source"]["raw_pi05_archive_file_sha256"],
        "tight-prefix full-episode raw archive differs",
    )
    archived = _load(archive_path)
    _require(
        archived.get("result_payload_sha256")
        == config["source"]["raw_pi05_archive_payload_sha256"],
        "tight-prefix full-episode raw archive payload differs",
    )
    matches = [
        row for row in read_jsonl(manifest_path)
        if row.get("case_id") == CASE_ID
    ]
    _require(len(matches) == 1,
             "tight-prefix full-episode manifest case differs")
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
            "tight-prefix full-episode frozen producer differs",
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
        selector_factory=LiveTightPrefixSelector,
        selector_config=config,
        selector_method="select_minimum_risk",
        selected_action_source="tight_prefix_minimum_risk_no_QP",
    )
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "complete",
        "scientific_result": True,
        "replica": replica,
        "source": source,
        "allocation": allocation,
        "config": config,
        "case_id": CASE_ID,
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
                 "tight-prefix full-episode scientific replay differs")
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
        replica=args.replica, host=args.host, port=args.port,
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
