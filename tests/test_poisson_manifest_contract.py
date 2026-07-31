"""Always-on contract tests for the checked-in 109-case target manifest."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, List
import unittest

from main.poisson_fullbody.feasibility_protocol import (
    load_feasibility_protocol,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/vlsa_poisson_link56_feasibility.v1.json"
BUILDER_PATH = ROOT / "scripts/build_poisson_link56_manifest.py"
MANIFEST_PATH = ROOT / "manifests/vlsa_poisson_link56_aegis_car_109.v1.jsonl"
RECEIPT_PATH = ROOT / "manifests/vlsa_poisson_link56_aegis_car_109.v1.receipt.json"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load {}".format(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = load_module("poisson_link56_manifest_builder", BUILDER_PATH)


class PoissonManifestContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config, cls.config_sha256 = builder.load_config(CONFIG_PATH)
        cls.manifest_payload = MANIFEST_PATH.read_bytes()
        cls.rows = [
            json.loads(line)
            for line in cls.manifest_payload.decode("utf-8").splitlines()
        ]
        cls.receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))

    def test_manifest_is_canonical_and_hash_bound(self) -> None:
        self.assertEqual(self.manifest_payload, builder.manifest_bytes(self.rows))
        self.assertTrue(self.manifest_payload.endswith(b"\n"))
        self.assertEqual(len(self.rows), 109)
        self.assertEqual(
            [row["case_ordinal"] for row in self.rows], list(range(109))
        )
        self.assertEqual(len({row["case_id"] for row in self.rows}), 109)
        self.assertEqual(
            hashlib.sha256(self.manifest_payload).hexdigest(),
            self.receipt["manifest_sha256"],
        )
        self.assertEqual(self.receipt["manifest_rows"], 109)
        self.assertEqual(self.receipt["full_source_join_rows_validated"], 1600)
        self.assertEqual(self.receipt["exact_source_join_rows"], 109)
        self.assertEqual(self.receipt["protocol_config_sha256"], self.config_sha256)
        self.assertEqual(
            self.receipt["source_manifest_sha256"],
            self.config["source"]["table1_manifest_sha256"],
        )
        self.assertEqual(
            self.receipt["source_audit_sha256"],
            self.config["source"]["case_audit_sha256"],
        )
        historical = self.config["source"]["historical_results"]
        self.assertEqual(
            self.receipt["failure_case_ledger_sha256"],
            historical["failure_case_ledger_sha256"],
        )
        self.assertEqual(
            self.receipt["accepted_result_payloads_sha256"],
            historical["accepted_result_payloads_sha256"],
        )
        for field in (
            "source_run_contract_sha256",
            "pi05_tree_sha256",
            "pi05_hash_receipt_sha256",
        ):
            self.assertEqual(self.receipt[field], historical[field])
        self.assertEqual(self.receipt["historical_aegis_results_bound"], 109)
        historical_bindings = [
            {
                "case_id": row["case_id"],
                "result_payload_sha256": row["historical_aegis_result"][
                    "result_payload_sha256"
                ],
                "raw_file_sha256": row["historical_aegis_result"][
                    "raw_file_sha256"
                ],
                "source_identifier": row["historical_aegis_result"][
                    "source_identifier"
                ],
            }
            for row in self.rows
        ]
        self.assertEqual(
            self.receipt["historical_result_bindings_sha256"],
            builder.sha256_bytes(builder.canonical_json_bytes(historical_bindings)),
        )
        self.assertEqual(
            {row["protocol_config_sha256"] for row in self.rows},
            {self.config_sha256},
        )

    def test_checked_runtime_protocol_is_semantically_valid_and_hash_bound(self) -> None:
        binding = self.config["runtime_protocol"]
        path = ROOT / binding["relative_path"]
        _, hashes = load_feasibility_protocol(
            path,
            expected_protocol_sha256=binding["semantic_protocol_sha256"],
        )
        self.assertEqual(
            hashes.parameter_block_sha256,
            binding["parameter_block_sha256"],
        )
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), binding["raw_file_sha256"])
        self.assertEqual(
            binding["selection_status"],
            "development_canary_not_heldout_frozen",
        )

    def test_every_row_has_literal_link_contact_and_disclosure(self) -> None:
        protected = {"robot0_link5", "robot0_link6"}
        for row in self.rows:
            self.assertEqual(row["schema_version"], builder.MANIFEST_SCHEMA)
            evidence = row["selection_evidence"]
            all_bodies = set(evidence["literal_robot_contact_bodies"])
            literal_links = set(evidence["literal_protected_link_contact_bodies"])
            self.assertTrue(literal_links)
            self.assertEqual(literal_links, all_bodies & protected)
            self.assertTrue(evidence["aegis_collision"])
            self.assertTrue(evidence["aegis_postcontrol_robot_contact"])
            self.assertTrue(evidence["aegis_geometry_complete"])
            self.assertTrue(evidence["aegis_all_actions_solved_qp"])
            self.assertTrue(row["outcome_conditioned"])
            self.assertEqual(
                row["population_kind"], "post_hoc_collision_mechanism_population"
            )
            self.assertEqual(row["claim_scope"], "targeted_feasibility_diagnostic_only")
            self.assertIn("unbiased_safelibero_benchmark", row["prohibited_interpretation"])
            self.assertRegex(row["source_manifest_row_sha256"], r"^[0-9a-f]{64}$")
            self.assertRegex(row["source_audit_row_sha256"], r"^[0-9a-f]{64}$")

            historical = row["historical_aegis_result"]
            self.assertRegex(historical["result_payload_sha256"], r"^[0-9a-f]{64}$")
            self.assertRegex(historical["raw_file_sha256"], r"^[0-9a-f]{64}$")
            self.assertGreater(historical["action_count"], 0)
            self.assertEqual(
                historical["action_count"],
                historical["action_invariance_ledger"]["action_count"],
            )
            self.assertRegex(
                historical["action_invariance_ledger"]["executed_sequence_sha256"],
                r"^[0-9a-f]{64}$",
            )
            self.assertFalse(Path(historical["source_relative_path"]).is_absolute())
            self.assertFalse(Path(historical["source_relative_pattern"]).is_absolute())
            self.assertNotIn("/Users/", historical["source_relative_path"])
            self.assertEqual(
                historical["source_relative_pattern"],
                "tasks/task-*/results/aegis/{}/result.json".format(row["case_id"]),
            )
            self.assertEqual(
                historical["source_identifier"],
                "vlsa-table1-contact-authority-population-20260718a::aegis::{}".format(
                    row["case_id"]
                ),
            )
            pairing = historical["pairing"]
            self.assertEqual(pairing["manifest_row_sha256"], row["source_manifest_row_sha256"])
            for value in pairing.values():
                self.assertRegex(value, r"^[0-9a-f]{64}$")
            self.assertEqual(
                pairing["semantic_label_record_sha256"],
                historical["semantic_label_record_sha256"],
            )
            self.assertRegex(
                historical["semantic_label_record_sha256"], r"^[0-9a-f]{64}$"
            )
            direct = historical[
                "direct_protected_link_active_obstacle_contact"
            ]
            self.assertEqual(
                set(direct),
                {
                    "source_field",
                    "active_obstacle_name",
                    "required_protected_link_body_names",
                    "matched_pairs",
                    "matched_pairs_sha256",
                },
            )
            self.assertEqual(
                direct["source_field"],
                "contact_telemetry.unique_contact_pairs",
            )
            self.assertEqual(
                direct["active_obstacle_name"], row["active_obstacle_name"]
            )
            required_links = set(
                evidence["literal_protected_link_contact_bodies"]
            )
            self.assertEqual(
                set(direct["required_protected_link_body_names"]),
                required_links,
            )
            matched_pairs = direct["matched_pairs"]
            self.assertTrue(matched_pairs)
            self.assertEqual(
                matched_pairs,
                sorted(matched_pairs, key=builder.canonical_json_bytes),
            )
            self.assertEqual(
                direct["matched_pairs_sha256"],
                builder.sha256_bytes(builder.canonical_json_bytes(matched_pairs)),
            )
            matched_links = set()
            source_pair_hashes = set()
            for pair in matched_pairs:
                self.assertEqual(
                    set(pair),
                    {
                        "historical_unique_contact_pair_sha256",
                        "protected_link_body_name",
                        "protected_link_geom_name",
                        "active_obstacle_name",
                        "active_obstacle_contact_body_name",
                        "active_obstacle_root_body_name",
                        "active_obstacle_geom_name",
                        "protected_link_body_lineage",
                        "active_obstacle_body_lineage",
                    },
                )
                protected_link = pair["protected_link_body_name"]
                matched_links.add(protected_link)
                self.assertIn(protected_link, required_links)
                self.assertEqual(
                    pair["protected_link_body_lineage"][0], protected_link
                )
                self.assertEqual(
                    pair["active_obstacle_name"], row["active_obstacle_name"]
                )
                self.assertEqual(
                    pair["active_obstacle_body_lineage"][0],
                    pair["active_obstacle_contact_body_name"],
                )
                self.assertIn(
                    pair["active_obstacle_root_body_name"],
                    pair["active_obstacle_body_lineage"],
                )
                root_name = pair["active_obstacle_root_body_name"]
                self.assertTrue(
                    root_name == row["active_obstacle_name"]
                    or root_name.startswith(row["active_obstacle_name"] + "_")
                )
                for geom_name in (
                    pair["protected_link_geom_name"],
                    pair["active_obstacle_geom_name"],
                ):
                    self.assertTrue(
                        geom_name is None
                        or isinstance(geom_name, str) and bool(geom_name)
                    )
                pair_hash = pair["historical_unique_contact_pair_sha256"]
                self.assertRegex(pair_hash, r"^[0-9a-f]{64}$")
                self.assertNotIn(pair_hash, source_pair_hashes)
                source_pair_hashes.add(pair_hash)
            self.assertEqual(matched_links, required_links)

        # A substring must never satisfy the literal contact-body selector.
        self.assertFalse(
            {"not_robot0_link5"}
            & set(self.config["population"]["selection_rule"]["literal_robot_contact_body_any_of"])
        )
        self.assertEqual(builder.literal_contact_bodies("robot0_link5_extra"), ("robot0_link5_extra",))

    def test_registered_strata_and_task_failures_are_exact(self) -> None:
        counts = Counter(row["stratum"] for row in self.rows)
        failures = Counter(
            row["stratum"]
            for row in self.rows
            if row["selection_evidence"]["aegis_task_failure"]
        )
        expected_counts = {
            name: contract["expected_cases"]
            for name, contract in self.config["population"]["strata"].items()
        }
        expected_failures = {
            name: contract["expected_aegis_task_failures"]
            for name, contract in self.config["population"]["strata"].items()
        }
        self.assertEqual(dict(counts), expected_counts)
        self.assertEqual(dict(failures), expected_failures)
        self.assertEqual(sum(failures.values()), 62)
        self.assertEqual(self.receipt["stratum_counts"], expected_counts)
        self.assertEqual(self.receipt["stratum_task_failures"], expected_failures)

        for row in self.rows:
            evidence = row["selection_evidence"]
            h_value = evidence["released_aegis_barrier_h_at_first_relevant_contact"]
            if row["stratum"] == "primary_static_positive_aegis_barrier":
                self.assertFalse(evidence["aegis_settled_relevant_contact"])
                self.assertIsNotNone(h_value)
                self.assertGreater(h_value, 0.0)
            elif row["stratum"] == "support_contact_after_settling_stress":
                self.assertTrue(evidence["aegis_settled_relevant_contact"])
            else:
                self.assertFalse(evidence["aegis_settled_relevant_contact"])
                self.assertIsNotNone(h_value)
                self.assertLessEqual(h_value, 0.0)

    def test_five_twenty_eighty_four_split_is_audited(self) -> None:
        split_names = ("bringup_canary", "parameter_freeze", "heldout_evaluation")
        split_counts = Counter(row["split"] for row in self.rows)
        self.assertEqual(
            dict(split_counts),
            {name: self.config["split"][name]["expected_cases"] for name in split_names},
        )
        self.assertEqual(self.receipt["split_counts"], dict(split_counts))
        self.assertEqual(
            {row["case_id"] for row in self.rows if row["split"] == "bringup_canary"},
            set(self.config["split"]["bringup_canary"]["case_ids"]),
        )

        for split_name in split_names:
            selected = [row for row in self.rows if row["split"] == split_name]
            contract = self.config["split"][split_name]
            actual_failures = sum(
                row["selection_evidence"]["aegis_task_failure"] for row in selected
            )
            actual_strata = {
                name: sum(row["stratum"] == name for row in selected)
                for name in self.config["population"]["strata"]
            }
            actual_suites = {
                name: sum(row["suite"] == name for row in selected)
                for name in self.config["execution"]["max_steps_by_suite"]
            }
            self.assertEqual(actual_failures, contract["expected_aegis_task_failures"])
            self.assertEqual(actual_strata, contract["expected_strata"])
            self.assertEqual(actual_suites, contract["expected_suites"])
            self.assertEqual(
                self.receipt["split_task_failures"][split_name], actual_failures
            )
            self.assertEqual(
                self.receipt["split_stratum_counts"][split_name], actual_strata
            )
            self.assertEqual(self.receipt["split_suite_counts"][split_name], actual_suites)

    def test_task_families_are_blocked_between_development_and_evaluation(self) -> None:
        family_partitions: Dict[str, set] = defaultdict(set)
        for row in self.rows:
            expected_family_id = builder.semantic_task_family_id(
                row["suite"], row["task_name"]
            )
            self.assertEqual(row["task_family_id"], expected_family_id)
            self.assertRegex(row["task_family_id"], r"^[0-9a-f]{64}$")
            family_partitions[expected_family_id].add(row["study_partition"])
        self.assertTrue(all(len(parts) == 1 for parts in family_partitions.values()))

        registered_development = {
            (entry["suite"], entry["task_name"]): entry["expected_cases"]
            for entry in self.config["split"]["development_pool"]["task_families"]
        }
        actual_development = Counter(
            (row["suite"], row["task_name"])
            for row in self.rows
            if row["study_partition"] == "development"
        )
        self.assertEqual(dict(actual_development), registered_development)
        self.assertEqual(sum(actual_development.values()), 25)
        self.assertEqual(self.receipt["development_pool_cases"], 25)

    def test_source_horizons_seeds_and_required_arms_are_preserved(self) -> None:
        required_arms = self.config["required_arms"]
        for row in self.rows:
            expected_horizon = self.config["execution"]["max_steps_by_suite"][row["suite"]]
            self.assertEqual(row["max_steps"], expected_horizon)
            if row["suite"] == "safelibero_long":
                self.assertEqual(row["max_steps"], 550)
            self.assertEqual(row["settle_actions"], 20)
            self.assertEqual(row["required_arms"], required_arms)
            self.assertEqual(row["source_policy_action_space"], "translational_only")
            self.assertEqual(
                row["active_execution_action_space"],
                "joint_velocity_7d_plus_gripper",
            )
        self.assertEqual(
            self.receipt["paired_results_required"],
            109 * len(required_arms),
        )
        self.assertEqual(self.receipt["required_arms"], required_arms)


if __name__ == "__main__":
    unittest.main()
