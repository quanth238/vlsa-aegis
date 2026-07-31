"""Regeneration and one-to-one join tests for the external Table-1 audit."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/vlsa_poisson_link56_feasibility.v1.json"
BUILDER_PATH = ROOT / "scripts/build_poisson_link56_manifest.py"
SOURCE_POPULATION_PATH = ROOT / "manifests/vlsa_table1_population.jsonl"
CHECKED_MANIFEST_PATH = ROOT / "manifests/vlsa_poisson_link56_aegis_car_109.v1.jsonl"
CHECKED_RECEIPT_PATH = ROOT / "manifests/vlsa_poisson_link56_aegis_car_109.v1.receipt.json"
EXTERNAL_AUDIT_PATH = Path(
    "/Users/quanth238/personal/Research/probe_vla/output/"
    "vlsa_aegis_table1/full_failure_reanalysis_20260730/case_audit.csv"
)
HISTORICAL_RESULT_ROOT = Path(
    "/Users/quanth238/personal/Research/probe_vla/output/vlsa_aegis_table1/"
    "population/vlsa-table1-contact-authority-population-20260718a"
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load {}".format(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = load_module("poisson_link56_source_join_builder", BUILDER_PATH)


class PoissonManifestJoinUnitTest(unittest.TestCase):
    def test_join_rejects_missing_extra_duplicate_and_identity_mismatch(self) -> None:
        source = [
            {
                "case_id": "case-a",
                "suite": "suite",
                "task_level_group_id": "group",
                "task_name": "task",
                "safety_level": "I",
                "episode_index": 0,
            }
        ]
        audit = [
            {
                "case_id": "case-a",
                "suite": "suite",
                "task_level_group_id": "group",
                "task_name": "task",
                "safety_level": "I",
                "episode_index": "0",
            }
        ]
        joined = builder.validate_source_join(source, audit)
        self.assertEqual(set(joined), {"case-a"})

        with self.assertRaises(builder.ManifestBuildError):
            builder.validate_source_join(source, [])
        with self.assertRaises(builder.ManifestBuildError):
            builder.validate_source_join(source, audit + [dict(audit[0])])

        mismatched = copy.deepcopy(audit)
        mismatched[0]["task_name"] = "different-task"
        with self.assertRaises(builder.ManifestBuildError):
            builder.validate_source_join(source, mismatched)

    def test_literal_body_parser_does_not_accept_substrings(self) -> None:
        protected = {"robot0_link5", "robot0_link6"}
        self.assertFalse(set(builder.literal_contact_bodies("robot0_link50")) & protected)
        self.assertFalse(set(builder.literal_contact_bodies("prefix_robot0_link6")) & protected)
        self.assertEqual(
            set(builder.literal_contact_bodies("robot0_link5;robot0_link6")) & protected,
            protected,
        )
        with self.assertRaises(builder.ManifestBuildError):
            builder.literal_contact_bodies("robot0_link5;robot0_link5")

    def test_direct_pair_join_rejects_cross_pair_false_positive(self) -> None:
        # Link 5 touches object A, while a different robot link touches the
        # selected active obstacle B.  Unioning names across records would
        # falsely accept this fixture; there is no direct link5--B pair.
        adversarial_pairs = [
            {
                "geom1": "robot0_link5_collision",
                "geom2": "object_a_geom",
                "body_lineage1": ["robot0_link5", "robot0_link4", "world"],
                "body_lineage2": ["object_a_main", "world"],
            },
            {
                "geom1": "active_obstacle_b_geom",
                "geom2": "robot0_link4_collision",
                "body_lineage1": ["active_obstacle_b_main", "world"],
                "body_lineage2": ["robot0_link4", "robot0_link3", "world"],
            },
        ]
        with self.assertRaisesRegex(
            builder.ManifestBuildError,
            "no direct active-obstacle pair for robot0_link5",
        ):
            builder.matched_direct_contact_pair_evidence(
                unique_pairs=adversarial_pairs,
                active_obstacle_name="active_obstacle_b",
                audited_links={"robot0_link5"},
                case_id="adversarial-case",
            )

    def test_direct_pair_evidence_is_same_record_and_deterministic(self) -> None:
        direct_link5 = {
            "geom1": "active_obstacle_b_geom_2",
            "geom2": "robot0_link5_collision",
            "body_lineage1": ["active_obstacle_b_child", "active_obstacle_b_main", "world"],
            "body_lineage2": ["robot0_link5", "robot0_link4", "world"],
        }
        direct_link6 = {
            "geom1": "robot0_link6_collision",
            "geom2": "active_obstacle_b_geom_1",
            "body_lineage1": ["robot0_link6", "robot0_link5", "world"],
            "body_lineage2": ["active_obstacle_b_main", "world"],
        }
        evidence = builder.matched_direct_contact_pair_evidence(
            unique_pairs=[direct_link5, direct_link6],
            active_obstacle_name="active_obstacle_b",
            audited_links={"robot0_link5", "robot0_link6"},
            case_id="positive-case",
        )
        reversed_input = builder.matched_direct_contact_pair_evidence(
            unique_pairs=[direct_link6, direct_link5],
            active_obstacle_name="active_obstacle_b",
            audited_links={"robot0_link5", "robot0_link6"},
            case_id="positive-case",
        )
        self.assertEqual(evidence, reversed_input)
        self.assertEqual(
            {row["protected_link_body_name"] for row in evidence},
            {"robot0_link5", "robot0_link6"},
        )
        link5 = next(
            row for row in evidence if row["protected_link_body_name"] == "robot0_link5"
        )
        self.assertEqual(link5["active_obstacle_contact_body_name"], "active_obstacle_b_child")
        self.assertEqual(link5["active_obstacle_root_body_name"], "active_obstacle_b_main")
        self.assertEqual(link5["protected_link_geom_name"], "robot0_link5_collision")
        self.assertEqual(link5["active_obstacle_geom_name"], "active_obstacle_b_geom_2")
        self.assertEqual(
            link5["historical_unique_contact_pair_sha256"],
            builder.sha256_bytes(builder.canonical_json_bytes(direct_link5)),
        )


@unittest.skipUnless(
    EXTERNAL_AUDIT_PATH.is_file() and HISTORICAL_RESULT_ROOT.is_dir(),
    "external completed Table-1 audit/result root is not available on this machine",
)
class PoissonManifestExternalRegenerationTest(unittest.TestCase):
    def test_external_sources_regenerate_checked_artifacts_exactly(self) -> None:
        rows = builder.build_rows(
            config_path=CONFIG_PATH,
            source_population_path=SOURCE_POPULATION_PATH,
            source_audit_path=EXTERNAL_AUDIT_PATH,
            historical_result_root=HISTORICAL_RESULT_ROOT,
        )
        payload = builder.manifest_bytes(rows)
        self.assertEqual(payload, CHECKED_MANIFEST_PATH.read_bytes())
        receipt = builder.build_receipt(
            config_path=CONFIG_PATH,
            source_population_path=SOURCE_POPULATION_PATH,
            source_audit_path=EXTERNAL_AUDIT_PATH,
            rows=rows,
            payload=payload,
        )
        checked_receipt = json.loads(CHECKED_RECEIPT_PATH.read_text(encoding="utf-8"))
        self.assertEqual(receipt, checked_receipt)

    def test_all_1600_source_cases_join_before_exact_109_selection(self) -> None:
        config, _ = builder.load_config(CONFIG_PATH)
        source_rows = builder.load_source_population(SOURCE_POPULATION_PATH, config)
        audit_rows = builder.load_source_audit(EXTERNAL_AUDIT_PATH, config)
        joined = builder.validate_source_join(source_rows, audit_rows)
        self.assertEqual(len(source_rows), 1600)
        self.assertEqual(len(audit_rows), 1600)
        self.assertEqual(len(joined), 1600)

        historical = builder.load_historical_result_index(
            HISTORICAL_RESULT_ROOT,
            config,
            {row["case_id"] for row in source_rows},
        )
        self.assertEqual(len(historical), 1600)

        selected = [
            row
            for row in source_rows
            if builder._selected_audit_row(joined[row["case_id"]], config)
        ]
        self.assertEqual(len(selected), 109)

    def test_all_109_selected_cases_have_direct_historical_pair_evidence(self) -> None:
        rows = builder.build_rows(
            config_path=CONFIG_PATH,
            source_population_path=SOURCE_POPULATION_PATH,
            source_audit_path=EXTERNAL_AUDIT_PATH,
            historical_result_root=HISTORICAL_RESULT_ROOT,
        )
        self.assertEqual(len(rows), 109)
        for row in rows:
            binding = row["historical_aegis_result"][
                "direct_protected_link_active_obstacle_contact"
            ]
            pairs = binding["matched_pairs"]
            self.assertTrue(pairs)
            self.assertEqual(binding["active_obstacle_name"], row["active_obstacle_name"])
            self.assertEqual(
                binding["matched_pairs_sha256"],
                builder.sha256_bytes(builder.canonical_json_bytes(pairs)),
            )
            matched_links = {pair["protected_link_body_name"] for pair in pairs}
            self.assertEqual(
                set(binding["required_protected_link_body_names"]),
                matched_links,
            )


if __name__ == "__main__":
    unittest.main()
