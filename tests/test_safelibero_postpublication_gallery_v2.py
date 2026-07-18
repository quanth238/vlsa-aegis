"""Tests for the compact evidence-rich post-publication gallery."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "analysis"
    / "build_safelibero_postpublication_gallery_v2.py"
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gallery = load_module("postpublication_gallery_v2", SCRIPT)
aggregate = gallery.aggregate
failures = gallery.failures


def publish(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


class PostpublicationGalleryV2Test(unittest.TestCase):
    def fixture(
        self, root: Path
    ) -> tuple[Path, Path, Path, Path]:
        baseline_video = root / "baseline.mp4"
        aegis_video = root / "aegis.mp4"
        baseline_video.write_bytes(b"baseline-video")
        aegis_video.write_bytes(b"aegis-video")
        result_hashes = {
            failures.BASELINE_ARM: "1" * 64,
            failures.AEGIS_ARM: "2" * 64,
        }
        outcome_base = {
            "paper_collision": True,
            "task_success": False,
            "legacy_ets_steps": 300,
            "executed_action_count": 300,
            "sampled_robot_active_obstacle_contact": False,
        }
        outcome_aegis = {
            "paper_collision": False,
            "task_success": True,
            "legacy_ets_steps": 145,
            "executed_action_count": 145,
            "sampled_robot_active_obstacle_contact": False,
        }
        record = {
            "schema_version": failures.CASE_SCHEMA_V2,
            "case_id": "case-0",
            "case_ordinal": 0,
            "suite": "safelibero_spatial",
            "safety_level": "I",
            "logical_task_index": 0,
            "task_name": "pick and place",
            "frozen_obstacle_label": "blue moka pot",
            "outcomes": {
                failures.BASELINE_ARM: {
                    **outcome_base,
                    "result_payload_sha256": result_hashes[
                        failures.BASELINE_ARM
                    ],
                },
                failures.AEGIS_ARM: {
                    **outcome_aegis,
                    "result_payload_sha256": result_hashes[
                        failures.AEGIS_ARM
                    ],
                },
            },
            "paired_transition": {
                "car": "baseline_collision_to_aegis_safe",
                "task": "baseline_failure_to_aegis_success",
                "joint": (
                    "baseline_collision_task_failure_to_aegis_"
                    "safe_task_success"
                ),
            },
            "aegis_diagnostics": {
                "intervention": {
                    "eligible_steps": 145,
                    "intervention_count": 45,
                    "correction_l2_sum": 3.25,
                }
            },
            "paired_goal_progress": {
                "aegis_minus_pi05": {
                    "final_fraction": 1.0,
                    "maximum_fraction": 1.0,
                    "regression_count": 0,
                }
            },
            "failure_analysis": {
                "primary_car_failure_class": "not_car_failure",
                "no_usable_points_observed_subclass": "not_applicable",
                "observed_evidence_tags": [
                    "paper_car_safe",
                    "task_success",
                ],
                "causal_hypotheses": [],
            },
            "videos": {
                failures.BASELINE_ARM: {
                    "absolute_path": str(baseline_video),
                    "hash_verified": True,
                },
                failures.AEGIS_ARM: {
                    "absolute_path": str(aegis_video),
                    "hash_verified": True,
                },
            },
        }
        record["record_payload_sha256"] = aggregate.sha256_bytes(
            aggregate.canonical_json_bytes(record)
        )
        cases_path = root / "cases.jsonl"
        cases_path.write_bytes(
            aggregate.canonical_json_bytes(record) + b"\n"
        )
        result_ledger = sorted(
            [
                {
                    "case_id": "case-0",
                    "arm": arm,
                    "result_payload_sha256": result_hashes[arm],
                }
                for arm in (
                    failures.BASELINE_ARM,
                    failures.AEGIS_ARM,
                )
            ],
            key=lambda row: (row["case_id"], row["arm"]),
        )
        claim_scope = {
            "method_label": (
                "pi0.5 + AEGIS translational conditioned on frozen "
                "per-case Codex obstacle labels"
            ),
            "baseline_method_label": "pi0.5 translational",
            "aegis_method_label": (
                "pi0.5 + AEGIS translational conditioned on frozen "
                "per-case Codex obstacle labels"
            ),
            "table_scope": "two-row translational Table-1 reproduction",
            "population_scope": "complete frozen one-case test population",
            "openvla_oft_included": False,
            "paper_semantic_selector_reproduced": False,
            "paper_exact_end_to_end_reproduction_claimed": False,
            "clearance_available": False,
            "minimum_clearance_claimed": False,
        }
        summary = {
            "schema_version": aggregate.OUTPUT_SCHEMA_V2,
            "status": "complete_postpublication_analysis_v2",
            "population": {"cases": 1, "arms": 2, "results": 2},
            "source_v1": {
                "v1_publication_receipt_sha256": "a" * 64,
            },
            "accepted_result_payloads_sha256": (
                aggregate.sha256_bytes(
                    aggregate.canonical_json_bytes(result_ledger)
                )
            ),
            "claim_scope": claim_scope,
        }
        summary_path = root / "summary.json"
        publish(summary_path, summary)
        case_ledger = [
            {
                "case_id": "case-0",
                "record_payload_sha256": record[
                    "record_payload_sha256"
                ],
            }
        ]
        report = {
            "schema_version": failures.REPORT_SCHEMA_V2,
            "status": "complete_postpublication_failure_analysis_v2",
            "claim_scope": claim_scope,
            "source": {
                "v1_publication_receipt_sha256": "a" * 64,
                "population_summary_v2_sha256": (
                    aggregate.sha256_path(summary_path)
                ),
                "accepted_result_payloads_sha256": summary[
                    "accepted_result_payloads_sha256"
                ],
            },
            "counts": {
                "case_count": 1,
                "video_count": 2,
                "all_videos_hash_verified": True,
            },
            "case_record_ledger_sha256": aggregate.sha256_bytes(
                aggregate.canonical_json_bytes(case_ledger)
            ),
        }
        report["report_payload_sha256"] = aggregate.sha256_bytes(
            aggregate.canonical_json_bytes(report)
        )
        report_path = root / "report.json"
        publish(report_path, report)
        return summary_path, report_path, cases_path, root / "gallery"

    def test_gallery_is_bound_enriched_and_clearance_scoped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary_path, report_path, cases_path, output_root = (
                self.fixture(root)
            )
            summary, report, records = gallery.load_bound_inputs(
                summary_path=summary_path,
                report_path=report_path,
                cases_path=cases_path,
            )
            document, metadata = gallery.build_gallery_v2(
                summary=summary,
                report=report,
                records=records,
                output_root=output_root,
            )
            self.assertEqual(metadata["cases"], 1)
            self.assertEqual(metadata["videos"], 2)
            self.assertEqual(document.count("<video controls"), 2)
            self.assertIn("Primary observed CAR class", document)
            self.assertIn("Hypotheses—not", document)
            self.assertIn("makes no clearance claim", document)
            self.assertIn('data-contact-disagreement="yes"', document)
            self.assertIn(
                'card.getAttribute("data-"+id)', document
            )

    def test_gallery_rejects_rehashed_result_ledger_substitution(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary_path, report_path, cases_path, _ = self.fixture(root)
            row = json.loads(cases_path.read_text(encoding="utf-8"))
            changed = copy.deepcopy(row)
            changed["outcomes"][failures.AEGIS_ARM][
                "result_payload_sha256"
            ] = "3" * 64
            changed["record_payload_sha256"] = aggregate.sha256_bytes(
                aggregate.canonical_json_bytes(
                    {
                        key: value
                        for key, value in changed.items()
                        if key != "record_payload_sha256"
                    }
                )
            )
            cases_path.write_bytes(
                aggregate.canonical_json_bytes(changed) + b"\n"
            )
            with self.assertRaisesRegex(
                gallery.GalleryV2Error,
                "case ledger differs",
            ):
                gallery.load_bound_inputs(
                    summary_path=summary_path,
                    report_path=report_path,
                    cases_path=cases_path,
                )

    def test_gallery_rejects_different_v1_publication_binding(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary_path, report_path, cases_path, _ = self.fixture(root)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["source"]["v1_publication_receipt_sha256"] = "b" * 64
            report["report_payload_sha256"] = aggregate.sha256_bytes(
                aggregate.canonical_json_bytes(
                    {
                        key: value
                        for key, value in report.items()
                        if key != "report_payload_sha256"
                    }
                )
            )
            publish(report_path, report)
            with self.assertRaisesRegex(
                gallery.GalleryV2Error,
                "not bound to the supplied summary",
            ):
                gallery.load_bound_inputs(
                    summary_path=summary_path,
                    report_path=report_path,
                    cases_path=cases_path,
                )


if __name__ == "__main__":
    unittest.main()
