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
                default_executed = 0
                default_legacy = 0
                default_reason = "method_failure"

        collision = default_collision if collision is None else collision
        success = default_success if success is None else success
        executed = default_executed if executed is None else executed
        legacy = default_legacy if legacy is None else legacy
        reason = default_reason if reason is None else reason
        case_id = manifest["case_id"]
        return {
            "schema_version": self.config["result_contract"][
                "schema_version"
            ],
            "protocol_id": self.config["protocol_id"],
            "case_id": case_id,
            "arm": arm,
            "status": status,
            "scientific_result": scientific_result,
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
                    f"{case_id}:observation"
                ),
                "policy_noise_schedule_id": manifest[
                    "policy_noise_schedule_id"
                ],
                "policy_noise_schedule_sha256": text_sha256(
                    f"{case_id}:noise{pairing_suffix}"
                ),
                "semantic_label_record_sha256": text_sha256(
                    f"{case_id}:label"
                ),
                "semantic_label_settled_agentview_sha256": text_sha256(
                    f"{case_id}:settled-agentview"
                ),
                "semantic_obstacle_label": "red milk carton",
                "max_steps": max_steps,
                "model_action_horizon": manifest[
                    "model_action_horizon"
                ],
                "replan_steps": manifest["replan_steps"],
            },
            "metrics": {
                "public_collision": collision,
                "task_success": success,
                "legacy_ets_steps": legacy,
                "executed_action_count": executed,
                "termination_reason": reason,
            },
        }

    def complete_results(self) -> dict[tuple[str, str], dict]:
        results = {}
        for manifest in self.rows:
            for arm in self.config["arms"]:
                result = self.make_result(manifest, arm)
                results[(manifest["case_id"], arm)] = (
                    aggregator.validate_result(
                        result,
                        config=self.config,
                        manifest=manifest,
                    )
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
        self.assertEqual(aegis["car_percent"], 78.0)
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
