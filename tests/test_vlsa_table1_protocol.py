"""Dependency-light tests for the frozen VLSA Table 1 protocol."""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/vlsa_table1_translational.json"
RECEIPT_PATH = ROOT / "manifests/vlsa_table1_population.receipt.json"
BUILDER_PATH = ROOT / "manifests/build_vlsa_table1_population.py"
AGGREGATOR_PATH = ROOT / "analysis/aggregate_safelibero_aegis.py"
TASK_MAP_PATH = (
    ROOT
    / "safelibero/libero/libero/benchmark/libero_suite_task_map.py"
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = load_module("vlsa_table1_manifest_builder", BUILDER_PATH)
aggregator = load_module("vlsa_table1_aggregator", AGGREGATOR_PATH)


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def released_task_map() -> dict[str, list[str]]:
    tree = ast.parse(TASK_MAP_PATH.read_text(encoding="utf-8"))
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id == "libero_task_map"
                for target in node.targets
            )
        ):
            value = ast.literal_eval(node.value)
            if not isinstance(value, dict):
                break
            return value
    raise AssertionError("released libero_task_map was not found")


class Table1ProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        cls.receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
        cls.rows = builder.build_rows(
            repo_root=ROOT,
            config_path=CONFIG_PATH,
        )
        cls.manifest_payload = builder.manifest_bytes(cls.rows)

    def make_result(
        self,
        manifest: dict,
        arm: str,
        *,
        pairing_suffix: str = "",
        status: str = "complete",
        collision: bool | None = None,
        success: bool | None = None,
        executed: int | None = None,
        legacy: int | None = None,
        reason: str | None = None,
        scientific_result: bool = True,
        include_evidence: bool = True,
        precontrol_geometry_failure: bool = False,
    ) -> dict:
        episode = manifest["episode_index"]
        max_steps = manifest["max_steps"]
        if arm == self.config["arms"][0]:
            default_collision = episode < 40
            default_success = episode < 25
            default_executed = 100 if default_success else max_steps
            default_legacy = 99 if default_success else max_steps
            default_reason = (
                "task_success" if default_success else "time_limit"
            )
        else:
            default_collision = episode < 10 or episode == 49
            default_success = episode < 35
            default_executed = 120 if default_success else max_steps
            default_legacy = 119 if default_success else max_steps
            default_reason = (
                "task_success" if default_success else "time_limit"
            )
            if episode == 49:
                status = "method_failure"
                default_collision = False
                default_executed = 0
                default_legacy = 0
                default_reason = "method_failure"

        collision = default_collision if collision is None else collision
        success = default_success if success is None else success
        executed = default_executed if executed is None else executed
        legacy = default_legacy if legacy is None else legacy
        reason = default_reason if reason is None else reason
        if precontrol_geometry_failure:
            if arm != self.config["arms"][1]:
                raise AssertionError(
                    "precontrol geometry failure belongs only to AEGIS"
                )
            status = "method_failure"
            collision = False
            success = False
            executed = 0
            legacy = 0
            reason = "method_failure"
        case_id = manifest["case_id"]
        label_record = {
            "schema_version": "vlsa_table1_codex_label.v1",
            "case_id": case_id,
            "settled_agentview_array_sha256": text_sha256(
                f"{case_id}:settled-agentview"
            ),
            "obstacle_label": "red milk carton",
            "reviewer": "codex",
            "reviewed_at": "2026-07-17T00:00:00+00:00",
        }
        label_hash = aggregator.canonical_record_sha256(label_record)
        settled_contract = {
            "schema_version": aggregator.SETTLED_INPUT_SCHEMA,
            "agentview_array_sha256": text_sha256(
                f"{case_id}:settled-agentview"
            ),
            "agentview_depth_array_sha256": text_sha256(
                f"{case_id}:agent-depth"
            ),
            "backview_array_sha256": text_sha256(
                f"{case_id}:back-rgb"
            ),
            "backview_depth_array_sha256": text_sha256(
                f"{case_id}:back-depth"
            ),
            "wrist_array_sha256": text_sha256(f"{case_id}:wrist"),
            "state_array_sha256": text_sha256(f"{case_id}:state-vector"),
            "active_obstacle_name": "milk_obstacle_1",
            "active_obstacle_position_array_sha256": text_sha256(
                f"{case_id}:obstacle-position"
            ),
            "settled_simulator_state_array_sha256": text_sha256(
                f"{case_id}:sim-state"
            ),
            "prompt": manifest["task_name"],
        }
        schedule = aggregator.expected_policy_noise_schedule(manifest)
        attempted = executed + int(status == "method_failure")
        query_count = (
            attempted + manifest["replan_steps"] - 1
        ) // manifest["replan_steps"]
        # Some negative tests deliberately request an impossible terminal
        # combination (for example, a method failure after the full horizon).
        # Keep fixture construction bounded by the frozen schedule so the
        # validator, rather than this helper, rejects the intended contract.
        query_count = min(query_count, schedule["query_count"])
        if not include_evidence:
            query_count = 1
        if precontrol_geometry_failure:
            query_count = 0
        policy_queries = [
            {
                "query_index": index,
                "rng_seed": schedule["query_seeds"][index],
                "returned_action_shape": [
                    manifest["model_action_horizon"],
                    7,
                ],
                "returned_actions_sha256": text_sha256(
                    f"{case_id}:query:{index}"
                ),
            }
            for index in range(query_count)
        ]
        action_rows = [
            {
                "step": index,
                "nominal_raw": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, -1.0],
                "nominal_translational": [
                    0.1,
                    0.2,
                    0.3,
                    0.0,
                    0.0,
                    0.0,
                    -1.0,
                ],
                "executed": [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, -1.0],
                "control_path": (
                    "pi05_translational_nominal"
                    if arm == self.config["arms"][0]
                    else "aegis_qp"
                ),
                "modified": False,
                "correction_l2": 0.0,
                "qp": (
                    None
                    if arm == self.config["arms"][0]
                    else {"solver": "OSQP", "solver_status": "optimal"}
                ),
                "reward": 0.0,
                "done": success and index == executed - 1,
                "step_elapsed_seconds": 0.01,
                "obstacle_l1_displacement_m": (
                    0.002 if collision and index == 0 else 0.0
                ),
                "robot_obstacle_contact": False,
            }
            for index in range(executed if include_evidence else 0)
        ]
        result = {
            "schema_version": self.config["result_contract"][
                "schema_version"
            ],
            "protocol_id": self.config["protocol_id"],
            "case_id": case_id,
            "arm": arm,
            "status": status,
            "scientific_result": scientific_result,
            "terminal_reason": reason,
            "task_success": success,
            "suite": manifest["suite"],
            "safety_level": manifest["safety_level"],
            "logical_task_index": manifest["logical_task_index"],
            "resolved_task_index": manifest["resolved_task_index"],
            "task_name": manifest["task_name"],
            "episode_index": manifest["episode_index"],
            "pairing": {
                "manifest_row_sha256": (
                    aggregator.canonical_record_sha256(manifest)
                ),
                "initial_state_sha256": text_sha256(
                    f"{case_id}:state"
                ),
                "initial_observation_sha256": text_sha256(
                    aggregator.canonical_json_bytes(settled_contract).decode(
                        "utf-8"
                    )
                ),
                "initial_observation_contract": settled_contract,
                "settled_simulator_state_sha256": settled_contract[
                    "settled_simulator_state_array_sha256"
                ],
                "settled_active_obstacle_position_sha256": (
                    settled_contract[
                        "active_obstacle_position_array_sha256"
                    ]
                ),
                "policy_noise_schedule_id": manifest[
                    "policy_noise_schedule_id"
                ],
                "policy_noise_schedule_sha256": (
                    aggregator.canonical_record_sha256(schedule)
                ),
                "policy_noise_schedule": schedule,
                "initial_policy_action_chunk_sha256": (
                    policy_queries[0]["returned_actions_sha256"]
                    if policy_queries
                    else None
                ),
                "semantic_label_record_sha256": label_hash,
                "semantic_label_settled_agentview_sha256": text_sha256(
                    f"{case_id}:settled-agentview"
                ),
                "semantic_obstacle_label": "red milk carton",
                "max_steps": max_steps,
                "model_action_horizon": manifest[
                    "model_action_horizon"
                ],
                "replan_steps": manifest["replan_steps"],
                "translational_fail_open": (
                    aggregator.TRANSLATIONAL_FAIL_OPEN
                ),
            },
            "settled_observation": {
                "agentview_array_sha256": text_sha256(
                    f"{case_id}:settled-agentview"
                ),
                "label_record": label_record,
                "label_record_sha256": label_hash,
                "obstacle_label": "red milk carton",
            },
            "timing": {"started_unix": 2_000_000_000.0},
            "policy_queries": policy_queries,
            "actions": action_rows,
            "metrics": {
                "public_collision": collision,
                "paper_collision": collision,
                "paper_collision_avoidance": not collision,
                "paper_collision_threshold_m": 0.001,
                "maximum_active_obstacle_l1_displacement_m": (
                    0.002 if collision else 0.0
                ),
                "collision_first_step": 0 if collision else None,
                "task_success": success,
                "legacy_ets_steps": legacy,
                "executed_action_count": executed,
                "termination_reason": reason,
                "safety_by_no_execution": (
                    status == "method_failure" and executed == 0
                ),
            },
            "intervention": {
                "eligible_steps": (
                    0 if arm == self.config["arms"][0] else executed
                ),
                "intervention_count": 0,
                "intervention_rate": 0.0,
                "correction_l2_sum": 0.0,
                "correction_l2_max": 0.0,
            },
            "contact_telemetry": {
                "status": "available",
                "robot_active_obstacle_contact": False,
                "first_contact_step": None,
                "unique_contact_pairs": [],
            },
            "video": {
                "path": (
                    f"{'pi05' if arm == self.config['arms'][0] else 'aegis'}"
                    f"/{case_id}/episode.mp4"
                ),
                "sha256": text_sha256(f"{case_id}:{arm}:video"),
                "frames": executed + int(status == "method_failure"),
                "fps": 30,
                "complete_episode": True,
            },
        }
        if status == "method_failure":
            if precontrol_geometry_failure:
                result["method_failure"] = {
                    "status": "method_failure",
                    "component": "aegis_geometry",
                    "phase": "precontrol",
                    "step": 0,
                    "type": "MethodFailure",
                    "message": "invalid released AEGIS geometry",
                    "safety_by_no_execution": True,
                }
                result["perception"] = {
                    "status": "method_failure",
                    "component": "aegis_geometry",
                    "reason": "invalid released AEGIS geometry",
                }
                result["pairing"].pop(
                    "initial_policy_action_chunk_sha256"
                )
            else:
                result["method_failure"] = {
                    "status": "method_failure",
                    "component": "aegis_qp",
                }
        result["result_payload_sha256"] = aggregator._result_payload_sha256(
            result
        )
        return result

    def complete_results(self) -> dict[tuple[str, str], dict]:
        results = {}
        for manifest in self.rows:
            for arm in self.config["arms"]:
                results[(manifest["case_id"], arm)] = self.make_result(
                    manifest,
                    arm,
                    include_evidence=False,
                )
        return results

    def test_manifest_is_exact_deterministic_population(self) -> None:
        self.assertEqual(len(self.rows), 1600)
        self.assertEqual(
            hashlib.sha256(self.manifest_payload).hexdigest(),
            self.receipt["manifest_sha256"],
        )
        self.assertEqual(
            hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest(),
            self.receipt["protocol_config_sha256"],
        )
        self.assertEqual(self.receipt["paired_results_required"], 3200)
        group_counts: dict[str, int] = {}
        for row in self.rows:
            group = row["task_level_group_id"]
            group_counts[group] = group_counts.get(group, 0) + 1
            self.assertEqual(
                row["required_arms"],
                [
                    "pi05_translational",
                    "pi05_plus_aegis_translational",
                ],
            )
        self.assertEqual(len(group_counts), 32)
        self.assertEqual(set(group_counts.values()), {50})
        stride = self.config["execution"]["policy_noise_case_stride"]
        self.assertGreaterEqual(stride, 110)
        self.assertEqual(
            self.rows[1]["policy_noise_seed"]
            - self.rows[0]["policy_noise_seed"],
            stride,
        )

    def test_frozen_tasks_match_released_map_and_goal_remap(self) -> None:
        self.assertEqual(
            self.config["population"]["tasks"],
            released_task_map(),
        )
        goal_i = next(
            row
            for row in self.rows
            if row["suite"] == "safelibero_goal"
            and row["safety_level"] == "I"
            and row["logical_task_index"] == 3
        )
        goal_ii = next(
            row
            for row in self.rows
            if row["suite"] == "safelibero_goal"
            and row["safety_level"] == "II"
            and row["logical_task_index"] == 3
        )
        self.assertEqual(goal_i["resolved_task_index"], 3)
        self.assertEqual(
            goal_i["task_name"],
            "open_the_top_drawer_and_put_the_bowl_inside",
        )
        self.assertEqual(goal_ii["resolved_task_index"], 4)
        self.assertEqual(
            goal_ii["task_name"],
            "put_the_cream_cheese_in_the_bowl",
        )

    def test_complete_population_retains_method_failures(self) -> None:
        results = self.complete_results()
        aggregator.validate_pairs(
            config=self.config,
            manifests=self.rows,
            results=results,
        )
        summary = aggregator.aggregate(
            config=self.config,
            manifests=self.rows,
            results=results,
        )
        self.assertEqual(
            summary["population"],
            {
                "cases": 1600,
                "arms": 2,
                "results": 3200,
                "task_level_groups": 32,
                "no_results_dropped": True,
            },
        )
        baseline = summary["suites"]["pi05_translational"][
            "safelibero_spatial"
        ]
        aegis = summary["average"]["pi05_plus_aegis_translational"]
        self.assertEqual(baseline["car_percent"], 20.0)
        self.assertEqual(baseline["tsr_percent"], 50.0)
        self.assertEqual(
            summary["average"]["pi05_translational"][
                "legacy_ets_steps_mean"
            ],
            230.75,
        )
        self.assertEqual(aegis["car_percent"], 80.0)
        self.assertEqual(aegis["tsr_percent"], 70.0)
        self.assertEqual(aegis["retained_method_failures"], 32)
        self.assertEqual(
            aegis["status_counts"]["method_failure"],
            32,
        )
        self.assertNotEqual(
            aegis["legacy_ets_steps_mean"],
            aegis["executed_action_count_mean"],
        )

    def test_load_results_rejects_one_missing_arm(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            manifest_path = temp / "manifest.jsonl"
            results_path = temp / "results.jsonl"
            manifest_path.write_bytes(self.manifest_payload)
            config, manifests = aggregator.load_protocol(
                CONFIG_PATH,
                RECEIPT_PATH,
                manifest_path,
            )
            rows = [
                self.make_result(manifest, arm)
                for manifest in manifests
                for arm in config["arms"]
            ]
            rows.pop()
            results_path.write_bytes(
                b"".join(
                    aggregator.canonical_json_bytes(row) + b"\n"
                    for row in rows
                )
            )
            with self.assertRaisesRegex(
                aggregator.AggregationError,
                "incomplete paired population",
            ):
                aggregator.load_results(
                    [results_path],
                    config=config,
                    manifests=manifests,
                    verify_video_files=False,
                )

    def test_pair_validation_rejects_changed_noise(self) -> None:
        results = self.complete_results()
        manifest = self.rows[0]
        aegis_key = (manifest["case_id"], self.config["arms"][1])
        changed = copy.deepcopy(results[aegis_key])
        changed["pairing"]["policy_noise_schedule_sha256"] = text_sha256(
            "different-noise"
        )
        results[aegis_key] = changed
        with self.assertRaisesRegex(
            aggregator.AggregationError,
            "arms are not paired",
        ):
            aggregator.validate_pairs(
                config=self.config,
                manifests=self.rows,
                results=results,
            )

    def test_pair_validation_rejects_changed_semantic_label(self) -> None:
        results = self.complete_results()
        manifest = self.rows[0]
        aegis_key = (manifest["case_id"], self.config["arms"][1])
        changed = copy.deepcopy(results[aegis_key])
        changed["pairing"]["semantic_obstacle_label"] = (
            "yellow rectangular book"
        )
        results[aegis_key] = changed
        with self.assertRaisesRegex(
            aggregator.AggregationError,
            "arms are not paired",
        ):
            aggregator.validate_pairs(
                config=self.config,
                manifests=self.rows,
                results=results,
            )

    def test_apparatus_failure_is_not_a_scientific_result(self) -> None:
        manifest = self.rows[0]
        result = self.make_result(
            manifest,
            self.config["arms"][1],
            status="apparatus_failure",
            scientific_result=False,
        )
        with self.assertRaisesRegex(
            aggregator.AggregationError,
            "non-scientific or nonterminal status",
        ):
            aggregator.validate_result(
                result,
                config=self.config,
                manifest=manifest,
            )

    def test_legacy_ets_success_contract_is_enforced(self) -> None:
        manifest = self.rows[0]
        result = self.make_result(
            manifest,
            self.config["arms"][0],
            success=True,
            executed=100,
            legacy=100,
            reason="task_success",
        )
        with self.assertRaisesRegex(
            aggregator.AggregationError,
            "successful legacy ETS must be actions minus one",
        ):
            aggregator.validate_result(
                result,
                config=self.config,
                manifest=manifest,
            )

    def test_method_failure_status_cannot_hide_a_timeout(self) -> None:
        manifest = self.rows[0]
        result = self.make_result(
            manifest,
            self.config["arms"][1],
            status="method_failure",
            collision=True,
            success=False,
            executed=manifest["max_steps"],
            legacy=manifest["max_steps"],
            reason="time_limit",
        )
        with self.assertRaisesRegex(
            aggregator.AggregationError,
            "hard method failure needs method_failure termination",
        ):
            aggregator.validate_result(
                result,
                config=self.config,
                manifest=manifest,
            )

    def test_precontrol_geometry_failure_is_valid_without_policy_query(
        self,
    ) -> None:
        manifest = self.rows[0]
        baseline = self.make_result(
            manifest,
            self.config["arms"][0],
        )
        aegis = self.make_result(
            manifest,
            self.config["arms"][1],
            precontrol_geometry_failure=True,
        )
        self.assertNotIn(
            "initial_policy_action_chunk_sha256",
            aegis["pairing"],
        )
        self.assertEqual(aegis["policy_queries"], [])
        self.assertEqual(aegis["actions"], [])
        aggregator.validate_result(
            baseline,
            config=self.config,
            manifest=manifest,
        )
        aggregator.validate_result(
            aegis,
            config=self.config,
            manifest=manifest,
        )
        aggregator.validate_pairs(
            config=self.config,
            manifests=[manifest],
            results={
                (manifest["case_id"], self.config["arms"][0]): baseline,
                (manifest["case_id"], self.config["arms"][1]): aegis,
            },
        )

    def test_precontrol_geometry_failure_contract_is_exact(self) -> None:
        manifest = self.rows[0]
        mutations = {
            "wrong phase": lambda result: result["method_failure"].update(
                {"phase": "control"}
            ),
            "wrong step": lambda result: result["method_failure"].update(
                {"step": 1}
            ),
            "Boolean step": lambda result: result["method_failure"].update(
                {"step": False}
            ),
            "unsafe flag": lambda result: result["method_failure"].update(
                {"safety_by_no_execution": False}
            ),
            "unsafe metric": lambda result: result["metrics"].update(
                {"safety_by_no_execution": False}
            ),
            "executed metric": lambda result: result["metrics"].update(
                {"executed_action_count": 1}
            ),
            "executed action": lambda result: result["actions"].append(
                {"step": 0}
            ),
            "policy query": lambda result: result["policy_queries"].append(
                {
                    "query_index": 0,
                    "rng_seed": manifest["policy_noise_seed"],
                    "returned_action_shape": [
                        manifest["model_action_horizon"],
                        7,
                    ],
                    "returned_actions_sha256": text_sha256(
                        f"{manifest['case_id']}:query:0"
                    ),
                }
            ),
            "initial action hash": lambda result: result["pairing"].update(
                {
                    "initial_policy_action_chunk_sha256": text_sha256(
                        f"{manifest['case_id']}:query:0"
                    )
                }
            ),
            "missing evidence frame": lambda result: result["video"].update(
                {"frames": 0}
            ),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                result = self.make_result(
                    manifest,
                    self.config["arms"][1],
                    precontrol_geometry_failure=True,
                )
                mutate(result)
                result["result_payload_sha256"] = (
                    aggregator._result_payload_sha256(result)
                )
                with self.assertRaisesRegex(
                    aggregator.AggregationError,
                    "precontrol AEGIS geometry failure contract is invalid",
                ):
                    aggregator.validate_result(
                        result,
                        config=self.config,
                        manifest=manifest,
                    )

    def test_non_geometry_failure_cannot_omit_policy_evidence(self) -> None:
        manifest = self.rows[0]
        mutations = {
            "missing hash": (
                lambda result: result["pairing"].pop(
                    "initial_policy_action_chunk_sha256"
                ),
                "initial_policy_action_chunk_sha256",
            ),
            "missing query": (
                lambda result: result.update({"policy_queries": []}),
                "policy-query count is invalid",
            ),
        }
        for name, (mutate, message) in mutations.items():
            with self.subTest(name=name):
                result = self.make_result(
                    manifest,
                    self.config["arms"][1],
                    status="method_failure",
                    collision=False,
                    success=False,
                    executed=0,
                    legacy=0,
                    reason="method_failure",
                )
                mutate(result)
                result["result_payload_sha256"] = (
                    aggregator._result_payload_sha256(result)
                )
                with self.assertRaisesRegex(
                    aggregator.AggregationError,
                    message,
                ):
                    aggregator.validate_result(
                        result,
                        config=self.config,
                        manifest=manifest,
                    )

    def test_pair_validation_requires_baseline_query_binding(self) -> None:
        manifest = self.rows[0]
        baseline = self.make_result(
            manifest,
            self.config["arms"][0],
        )
        aegis = self.make_result(
            manifest,
            self.config["arms"][1],
            precontrol_geometry_failure=True,
        )
        baseline["pairing"]["initial_policy_action_chunk_sha256"] = (
            text_sha256("different-baseline-action")
        )
        with self.assertRaisesRegex(
            aggregator.AggregationError,
            "baseline initial policy action is not bound",
        ):
            aggregator.validate_pairs(
                config=self.config,
                manifests=[manifest],
                results={
                    (
                        manifest["case_id"],
                        self.config["arms"][0],
                    ): baseline,
                    (
                        manifest["case_id"],
                        self.config["arms"][1],
                    ): aegis,
                },
            )

    def test_geometry_exception_does_not_unbind_baseline_pair(self) -> None:
        manifest = self.rows[0]
        baseline = self.make_result(
            manifest,
            self.config["arms"][0],
        )
        aegis = self.make_result(
            manifest,
            self.config["arms"][1],
            precontrol_geometry_failure=True,
        )
        baseline["pairing"].pop("initial_policy_action_chunk_sha256")
        with self.assertRaisesRegex(
            aggregator.AggregationError,
            "initial_policy_action_chunk_sha256",
        ):
            aggregator.validate_pairs(
                config=self.config,
                manifests=[manifest],
                results={
                    (
                        manifest["case_id"],
                        self.config["arms"][0],
                    ): baseline,
                    (
                        manifest["case_id"],
                        self.config["arms"][1],
                    ): aegis,
                },
            )

    def test_maximum_displacement_must_come_from_actions(self) -> None:
        manifest = self.rows[35]
        result = self.make_result(
            manifest,
            self.config["arms"][0],
            collision=False,
            success=False,
        )
        result["metrics"][
            "maximum_active_obstacle_l1_displacement_m"
        ] = 0.0005
        result["result_payload_sha256"] = (
            aggregator._result_payload_sha256(result)
        )
        with self.assertRaisesRegex(
            aggregator.AggregationError,
            "not derived from the action ledger",
        ):
            aggregator.validate_result(
                result,
                config=self.config,
                manifest=manifest,
            )

    def test_intervention_summary_must_match_actions(self) -> None:
        manifest = self.rows[35]
        result = self.make_result(
            manifest,
            self.config["arms"][1],
            collision=False,
            success=False,
        )
        result["intervention"]["intervention_count"] = 1
        result["result_payload_sha256"] = (
            aggregator._result_payload_sha256(result)
        )
        with self.assertRaisesRegex(
            aggregator.AggregationError,
            "intervention summary does not match actions",
        ):
            aggregator.validate_result(
                result,
                config=self.config,
                manifest=manifest,
            )

    def test_action_correction_norm_is_recomputed(self) -> None:
        manifest = self.rows[0]
        result = self.make_result(
            manifest,
            self.config["arms"][1],
            collision=True,
            success=True,
        )
        result["actions"][0]["executed"][0] = 0.2
        result["result_payload_sha256"] = (
            aggregator._result_payload_sha256(result)
        )
        with self.assertRaisesRegex(
            aggregator.AggregationError,
            "correction norm changed",
        ):
            aggregator.validate_result(
                result,
                config=self.config,
                manifest=manifest,
            )

    def test_receipt_rejects_manifest_byte_change(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = Path(directory) / "manifest.jsonl"
            manifest_path.write_bytes(self.manifest_payload + b"\n")
            with self.assertRaisesRegex(
                aggregator.AggregationError,
                "manifest hash does not match receipt",
            ):
                aggregator.load_protocol(
                    CONFIG_PATH,
                    RECEIPT_PATH,
                    manifest_path,
                )

    def test_pretty_result_json_and_run_root_are_supported(self) -> None:
        result = self.make_result(
            self.rows[0], self.config["arms"][0]
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result_path = root / "arm" / "case" / "result.json"
            result_path.parent.mkdir(parents=True)
            result_path.write_text(
                json.dumps(result, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )

            self.assertEqual(
                aggregator.load_result_records(result_path),
                [result],
            )
            self.assertEqual(
                aggregator.expand_result_paths([root]),
                [result_path],
            )


if __name__ == "__main__":
    unittest.main()
