"""Tests for the dependency-light SafeLIBERO video gallery."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "analysis/build_safelibero_video_gallery.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gallery = load_module("safelibero_video_gallery", SCRIPT)


ARMS = [
    "pi05_translational",
    "pi05_plus_aegis_translational",
]
SUITE = "safelibero_spatial"
PROTOCOL = "gallery-test-protocol"


def summary_fixture() -> dict:
    baseline = {
        "groups": 1,
        "episodes": 2,
        "status_counts": {"complete": 2},
        "retained_method_failures": 0,
        "car_percent": 50.0,
        "tsr_percent": 50.0,
        "legacy_ets_steps_mean": 154.5,
        "executed_action_count_mean": 155.0,
    }
    aegis = {
        "groups": 1,
        "episodes": 2,
        "status_counts": {"complete": 2},
        "retained_method_failures": 0,
        "car_percent": 100.0,
        "tsr_percent": 50.0,
        "legacy_ets_steps_mean": 155.5,
        "executed_action_count_mean": 156.0,
    }
    return {
        "schema_version": "vlsa_table1_population_summary.v1",
        "status": "complete_population_validated",
        "protocol_id": PROTOCOL,
        "population": {
            "cases": 2,
            "arms": 2,
            "results": 4,
            "task_level_groups": 1,
            "no_results_dropped": True,
        },
        "suites": {
            ARMS[0]: {SUITE: copy.deepcopy(baseline)},
            ARMS[1]: {SUITE: copy.deepcopy(aegis)},
        },
        "average": {
            ARMS[0]: copy.deepcopy(baseline),
            ARMS[1]: copy.deepcopy(aegis),
        },
    }


def result_fixture(
    case_index: int,
    arm: str,
    *,
    video_path: str | None = None,
) -> dict:
    case_id = f"case-{case_index}"
    baseline = arm == ARMS[0]
    if case_index == 0:
        collision = baseline
        task_success = False
        legacy = 300
        executed = 300
    else:
        collision = False
        task_success = True
        legacy = 9 if baseline else 11
        executed = 10 if baseline else 12
    result = {
        "schema_version": "vlsa_table1_episode_result.v1",
        "protocol_id": PROTOCOL,
        "case_id": case_id,
        "arm": arm,
        "status": "complete",
        "suite": SUITE,
        "safety_level": "I",
        "logical_task_index": 0,
        "resolved_task_index": 0,
        "task_name": "put_the_bowl_on_the_plate",
        "episode_index": case_index,
        "metrics": {
            "public_collision": collision,
            "task_success": task_success,
            "legacy_ets_steps": legacy,
            "executed_action_count": executed,
        },
        "contact_telemetry": {
            "status": "available",
            "robot_active_obstacle_contact": collision,
        },
        "video": {
            "path": (
                video_path
                if video_path is not None
                else f"{arm}/{case_id}/episode.mp4"
            )
        },
    }
    if not baseline:
        result.update(
            {
                "settled_observation": {
                    "obstacle_label": "red mug",
                    "label_record": {"label": "red mug"},
                },
                "perception": {
                    "status": "ready",
                    "views": {
                        "agentview": {"point_count": 100},
                        "backview": {"point_count": 100},
                    },
                },
                "obstacle": {"selector_mismatch": False},
                "actions": [
                    {"qp": {"solver_status": "optimal"}}
                ],
            }
        )
    return result


def complete_records() -> list[dict]:
    return [
        result_fixture(case_index, arm)
        for case_index in range(2)
        for arm in ARMS
    ]


class VideoGalleryTest(unittest.TestCase):
    def test_complete_gallery_has_paired_relative_links_and_filters(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            for result in complete_records():
                path = output_root / result["video"]["path"]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"video-placeholder")
            document, warnings = gallery.build_gallery(
                summary=summary_fixture(),
                records=complete_records(),
                output_root=output_root,
            )
        self.assertEqual(warnings, [])
        self.assertIn("case-0", document)
        self.assertIn("case-1", document)
        self.assertIn(ARMS[0], document)
        self.assertIn(ARMS[1], document)
        self.assertIn(
            f'src="{ARMS[0]}/case-0/episode.mp4"',
            document,
        )
        self.assertNotIn("file://", document)
        for filter_name in (
            "suite",
            "level",
            "task",
            "arm",
            "car",
            "tsr",
            "status",
            "failure",
        ):
            self.assertIn(f'id="{filter_name}-filter"', document)
        for _, label in gallery.TAXONOMY:
            self.assertIn(label, document)
        self.assertIn("Unknown evidence remains unknown", document)

    def test_missing_pair_requires_explicit_partial_debug_mode(self) -> None:
        records = complete_records()[:-1]
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            with self.assertRaisesRegex(
                gallery.GalleryError,
                "incomplete indexed result set",
            ):
                gallery.build_gallery(
                    summary=summary_fixture(),
                    records=records,
                    output_root=output_root,
                )
            document, warnings = gallery.build_gallery(
                summary=summary_fixture(),
                records=records,
                output_root=output_root,
                allow_partial=True,
            )
        self.assertTrue(
            any("incomplete indexed result set" in item for item in warnings)
        )
        self.assertIn("missing_result", document)
        self.assertIn("Partial debugging mode", document)

    def test_duplicate_result_is_rejected(self) -> None:
        records = complete_records()
        records.append(copy.deepcopy(records[0]))
        with self.assertRaisesRegex(
            gallery.GalleryError,
            "duplicate per-episode result",
        ):
            gallery.index_results(
                summary_fixture(),
                records,
                allow_partial=False,
            )

    def test_unsafe_video_path_is_rejected(self) -> None:
        records = complete_records()
        records[0]["video"]["path"] = "../outside.mp4"
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                gallery.GalleryError,
                "unsafe video path",
            ):
                gallery.build_gallery(
                    summary=summary_fixture(),
                    records=records,
                    output_root=Path(directory),
                )

    def test_summary_metric_mismatch_is_rejected(self) -> None:
        summary = summary_fixture()
        summary["suites"][ARMS[0]][SUITE]["car_percent"] = 75.0
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                gallery.GalleryError,
                "car_percent differs",
            ):
                gallery.build_gallery(
                    summary=summary,
                    records=complete_records(),
                    output_root=Path(directory),
                )

    def test_failure_taxonomy_distinguishes_method_layers(self) -> None:
        result = result_fixture(0, ARMS[1])
        result["status"] = "method_failure_passthrough"
        result["perception"] = {
            "status": "empty",
            "method_failure": "no_grounded_points",
        }
        result["obstacle"]["selector_mismatch"] = True
        result["contact_telemetry"] = {"status": "unavailable"}
        taxonomy = gallery.classify_failure_taxonomy(
            result,
            aegis_arm=True,
        )
        self.assertEqual(taxonomy["semantic_selector"], "absent")
        self.assertEqual(taxonomy["groundingdino"], "present")
        self.assertEqual(taxonomy["point_filtering_mvee"], "unknown")
        self.assertEqual(
            taxonomy["qp_method_failure"], "not_applicable"
        )
        self.assertEqual(
            taxonomy["active_obstacle_displacement"],
            "absent",
        )
        self.assertEqual(
            taxonomy["mujoco_robot_obstacle_contact"],
            "unknown",
        )
        self.assertEqual(
            taxonomy["selector_active_mismatch"],
            "present",
        )
        self.assertEqual(
            taxonomy["safe_but_task_failed_ood_proxy"],
            "present",
        )
        self.assertEqual(taxonomy["apparatus"], "absent")

    def test_missing_failure_evidence_remains_unknown(self) -> None:
        result = {
            "status": "complete",
            "metrics": {
                "public_collision": False,
                "task_success": True,
            },
        }
        taxonomy = gallery.classify_failure_taxonomy(
            result,
            aegis_arm=True,
        )
        self.assertEqual(taxonomy["semantic_selector"], "unknown")
        self.assertEqual(taxonomy["groundingdino"], "unknown")
        self.assertEqual(taxonomy["point_filtering_mvee"], "unknown")
        self.assertEqual(taxonomy["qp_method_failure"], "unknown")
        self.assertEqual(
            taxonomy["mujoco_robot_obstacle_contact"],
            "unknown",
        )
        self.assertEqual(
            taxonomy["selector_active_mismatch"],
            "unknown",
        )
        self.assertEqual(taxonomy["apparatus"], "absent")

    def test_qp_method_failure_uses_retained_method_record(self) -> None:
        result = result_fixture(0, ARMS[1])
        result["status"] = "method_failure"
        result["method_failure"] = {
            "component": "aegis_qp",
            "type": "MethodFailure",
            "message": "AEGIS QP returned no solution",
        }
        taxonomy = gallery.classify_failure_taxonomy(
            result,
            aegis_arm=True,
        )
        self.assertEqual(taxonomy["qp_method_failure"], "present")
        self.assertEqual(taxonomy["apparatus"], "absent")

    def test_apparatus_is_separate_from_method_failure(self) -> None:
        result = result_fixture(0, ARMS[1])
        result["status"] = "apparatus_failure"
        result["apparatus_error"] = {
            "type": "ApparatusError",
            "message": "failed to load GroundingDINO",
        }
        taxonomy = gallery.classify_failure_taxonomy(
            result,
            aegis_arm=True,
        )
        self.assertEqual(taxonomy["groundingdino"], "present")
        self.assertEqual(taxonomy["apparatus"], "present")

    def test_result_loader_reads_only_registered_result_files_in_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = complete_records()[0]
            result_path = root / "arm" / "case" / "result.json"
            result_path.parent.mkdir(parents=True)
            result_path.write_text(
                json.dumps(expected),
                encoding="utf-8",
            )
            (root / "unrelated.json").write_text(
                json.dumps({"not": "an episode"}),
                encoding="utf-8",
            )
            loaded = gallery.load_result_records([root])
        self.assertEqual(loaded, [expected])


if __name__ == "__main__":
    unittest.main()
