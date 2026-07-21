from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

from analysis import validate_aegis_failure_diagnostics as diagnostics
from scripts.aegis_receipt_utils import ReceiptError
from scripts import validate_aegis_run_artifacts as artifacts


SHA = "a" * 64
OTHER_SHA = "b" * 64


class _TrackedResult(dict):
    live = 0
    peak = 0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        type(self).live += 1
        type(self).peak = max(type(self).peak, type(self).live)

    def __del__(self):
        type(self).live -= 1


class PopulationStreamingValidationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.run_root = Path(self.temporary.name)
        self.task_root = self.run_root / "tasks/task-0"
        self.result_root = self.task_root / "results"
        self.config = {
            "arms": [
                "pi05_translational",
                "pi05_plus_aegis_translational",
            ]
        }
        self.manifests = [
            {
                "case_id": f"case-{index:02d}",
                "case_ordinal": index,
            }
            for index in range(50)
        ]
        self.labels = {
            row["case_id"]: {
                "schema_version": "vlsa_table1_codex_label.v1",
                "case_id": row["case_id"],
                "obstacle_label": f"obstacle-{row['case_ordinal']}",
            }
            for row in self.manifests
        }
        for manifest in self.manifests:
            for mode in ("pi05", "aegis"):
                path = (
                    self.result_root
                    / mode
                    / manifest["case_id"]
                    / "result.json"
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _result_for_path(
        self,
        path: Path,
        *,
        mode_override: str | None = None,
        label_override: dict | None = None,
        **kwargs,
    ) -> tuple[_TrackedResult, dict]:
        del kwargs
        mode = path.parents[1].name
        case_id = path.parent.name
        arm = (
            "pi05_translational"
            if mode == "pi05"
            else "pi05_plus_aegis_translational"
        )
        geometry = (
            {
                "status": "not_run",
                "reason": "pi05_baseline_arm",
            }
            if mode == "pi05"
            else {
                "sha256": OTHER_SHA,
                "record": {
                    "status": "failure",
                    "views": {
                        "agentview": {
                            "status": "observer_or_runtime_failure"
                        },
                        "backview": {"status": "not_attempted"},
                    },
                },
            }
        )
        result = _TrackedResult(
            {
                "case_id": case_id,
                "mode": mode_override or mode,
                "arm": arm,
                "status": "complete",
                "obstacle": {
                    "active_name": "moka_pot_obstacle_1",
                },
                "settled_observation": {
                    "label_record": copy.deepcopy(
                        label_override or self.labels[case_id]
                    )
                },
                "failure_diagnostics": {
                    "geometry": geometry,
                    "terminal_frame": {"sha256": SHA},
                    "contacts": {
                        "schema_version": (
                            artifacts.CONTACT_SCHEMA_V3
                        ),
                        "active_obstacle_name": "moka_pot_obstacle_1",
                        "sha256": SHA,
                        "uncompressed_payload_sha256": OTHER_SHA,
                        "model_authority_sha256": SHA,
                        "task_context_sha256": OTHER_SHA,
                        "active_obstacle_root_body_id": 5,
                        "snapshot_count": 1,
                        "event_count": 0,
                        "role_taxonomy": list(
                            artifacts.CONTACT_ROLE_TAXONOMY
                        ),
                        "role_authority_complete": True,
                        "event_counts_by_role": {
                            role: 0
                            for role in artifacts.CONTACT_ROLE_TAXONOMY
                        },
                        "steps_with_contact_by_role": {
                            role: []
                            for role in artifacts.CONTACT_ROLE_TAXONOMY
                        },
                        "first_contact_step_by_role": {
                            role: None
                            for role in artifacts.CONTACT_ROLE_TAXONOMY
                        },
                        "robot_event_count": 0,
                        "nonrobot_event_count": 0,
                    },
                },
                "actions": [],
                "method_failure": (
                    {
                        "component": "aegis_geometry",
                        "phase": "precontrol",
                    }
                    if mode == "aegis"
                    else None
                ),
                "goal_progress": {"summary": {"final_fraction": 0.0}},
                # A large/full-only marker must never escape in inventory.
                "_large_full_result_marker": "x" * 10000,
            }
        )
        return result, {
            "relative_result_path": str(path),
            "result_sha256": SHA,
            "result_payload_sha256": OTHER_SHA,
            "video_sha256": SHA,
            "case_id": case_id,
            "arm": arm,
            "status": "complete",
        }

    @staticmethod
    def _diagnostic_evidence(result, **kwargs):
        del kwargs
        return {
            "case_id": result["case_id"],
            "arm": result["arm"],
            "status": "valid",
            "action_invariance_ledger": {
                "action_count": 0,
                "policy_query_count": 0,
                "combined_ledger_sha256": SHA,
            },
            "video_decode": {
                "status": "fully_decoded",
                "decoded_frame_count": 1,
                "decoded_fps": 30.0,
            },
        }

    def _run_with_plain_patches(
        self,
        *,
        result_loader=None,
        diagnostic_validator=None,
        pairing_validator=None,
    ):
        original_result_loader = artifacts._validate_population_result
        original_diagnostic_validator = (
            artifacts.failure_validation.validate_diagnostic_result
        )
        original_pairing_validator = artifacts.aggregate.validate_pairs
        artifacts._validate_population_result = (
            result_loader or self._result_for_path
        )
        artifacts.failure_validation.validate_diagnostic_result = (
            diagnostic_validator or self._diagnostic_evidence
        )
        artifacts.aggregate.validate_pairs = (
            pairing_validator or (lambda **kwargs: None)
        )
        try:
            return artifacts._validate_population_task_results(
                task_index=0,
                task_root=self.task_root,
                run_root=self.run_root,
                config=self.config,
                task_manifests=self.manifests,
                label_records=self.labels,
                expected_commit="release-commit",
            )
        finally:
            artifacts._validate_population_result = original_result_loader
            (
                artifacts.failure_validation.validate_diagnostic_result
            ) = original_diagnostic_validator
            artifacts.aggregate.validate_pairs = original_pairing_validator

    def test_streams_exactly_one_pair_and_deep_validates_every_result(
        self,
    ) -> None:
        _TrackedResult.live = 0
        _TrackedResult.peak = 0
        diagnostic_calls: list[tuple[str, Path, bool]] = []
        paired_cases: list[str] = []

        def diagnostic_validator(result, **kwargs):
            diagnostic_calls.append(
                (
                    result["case_id"],
                    kwargs["output_root"],
                    kwargs["require_ready_geometry"],
                )
            )
            return self._diagnostic_evidence(result)

        def pairing_validator(*, manifests, results, **kwargs):
            del kwargs
            self.assertEqual(len(manifests), 1)
            self.assertEqual(len(results), 2)
            case_id = manifests[0]["case_id"]
            self.assertEqual(
                {key[0] for key in results},
                {case_id},
            )
            paired_cases.append(case_id)

        validated = self._run_with_plain_patches(
            diagnostic_validator=diagnostic_validator,
            pairing_validator=pairing_validator,
        )

        self.assertEqual(_TrackedResult.peak, 2)
        self.assertEqual(_TrackedResult.live, 0)
        self.assertEqual(len(diagnostic_calls), 100)
        self.assertTrue(
            all(
                root == self.result_root and ready is False
                for _, root, ready in diagnostic_calls
            )
        )
        self.assertEqual(paired_cases, [row["case_id"] for row in self.manifests])
        self.assertEqual(len(validated["inventory"]), 100)
        self.assertEqual(
            validated["mode_counts"],
            {"pi05": 50, "aegis": 50},
        )
        self.assertEqual(
            validated["geometry_status_counts"],
            {"pi05:not_run": 50, "aegis:failure": 50},
        )
        self.assertTrue(
            all(
                "_large_full_result_marker" not in item
                and "_large_full_result_marker"
                not in item["diagnostics"]
                for item in validated["inventory"]
            )
        )
        self.assertTrue(
            all(
                len(item["diagnostics"]["compact_evidence_sha256"]) == 64
                for item in validated["inventory"]
            )
        )
        self.assertEqual(
            validated["contact_event_counts_by_role"],
            {
                role: 0
                for role in artifacts.CONTACT_ROLE_TAXONOMY
            },
        )
        self.assertEqual(
            validated["contact_first_step_histograms_by_role"],
            {
                role: {"none": 100}
                for role in artifacts.CONTACT_ROLE_TAXONOMY
            },
        )
        self.assertTrue(
            all(
                item["diagnostics"]["contact_role_authority_complete"]
                is True
                and item["diagnostics"]["contact_role_taxonomy"]
                == list(artifacts.CONTACT_ROLE_TAXONOMY)
                for item in validated["inventory"]
            )
        )

    def test_rejects_mode_arm_or_frozen_label_corruption(self) -> None:
        first_path = (
            self.result_root
            / "pi05"
            / self.manifests[0]["case_id"]
            / "result.json"
        )

        for name, mutation in (
            (
                "mode",
                lambda path, **kwargs: self._result_for_path(
                    path,
                    mode_override=(
                        "aegis" if path == first_path else None
                    ),
                    **kwargs,
                ),
            ),
            (
                "label",
                lambda path, **kwargs: self._result_for_path(
                    path,
                    label_override=(
                        {
                            **self.labels[path.parent.name],
                            "obstacle_label": "tampered obstacle",
                        }
                        if path == first_path
                        else None
                    ),
                    **kwargs,
                ),
            ),
        ):
            with self.subTest(name=name), self.assertRaises(ReceiptError):
                self._run_with_plain_patches(result_loader=mutation)

    def test_rejects_deep_diagnostic_corruption(self) -> None:
        target = self.manifests[7]["case_id"]

        def diagnostic_validator(result, **kwargs):
            del kwargs
            if result["case_id"] == target and result["mode"] == "aegis":
                raise diagnostics.DiagnosticValidationError(
                    "tampered terminal frame"
                )
            return self._diagnostic_evidence(result)

        with self.assertRaisesRegex(
            ReceiptError,
            "deep diagnostics invalid: tampered terminal frame",
        ):
            self._run_with_plain_patches(
                diagnostic_validator=diagnostic_validator
            )

    def test_rejects_unknown_contact_role_event(self) -> None:
        target = self.manifests[3]["case_id"]

        def result_loader(path, **kwargs):
            result, item = self._result_for_path(path, **kwargs)
            if path.parent.name == target and path.parents[1].name == "aegis":
                contacts = result["failure_diagnostics"]["contacts"]
                contacts["event_count"] = 1
                contacts["nonrobot_event_count"] = 1
                contacts["event_counts_by_role"]["unknown"] = 1
                contacts["steps_with_contact_by_role"]["unknown"] = [-1]
                contacts["first_contact_step_by_role"]["unknown"] = -1
            return result, item

        with self.assertRaisesRegex(
            ReceiptError,
            "unknown contact roles are not publication-valid",
        ):
            self._run_with_plain_patches(result_loader=result_loader)

    def test_accepts_contact_maps_after_sorted_json_round_trip(self) -> None:
        path = (
            self.result_root
            / "pi05"
            / self.manifests[0]["case_id"]
            / "result.json"
        )
        result, _ = self._result_for_path(path)
        round_tripped = json.loads(json.dumps(result, sort_keys=True))

        compact = artifacts._compact_population_diagnostic_evidence(
            round_tripped,
            diagnostic_validation=self._diagnostic_evidence(round_tripped),
        )

        self.assertEqual(
            compact["contact_role_taxonomy"],
            list(artifacts.CONTACT_ROLE_TAXONOMY),
        )
        self.assertEqual(
            compact["contact_event_counts_by_role"],
            {role: 0 for role in artifacts.CONTACT_ROLE_TAXONOMY},
        )

    def test_rejects_missing_or_extra_contact_map_keys(self) -> None:
        path = (
            self.result_root
            / "pi05"
            / self.manifests[0]["case_id"]
            / "result.json"
        )
        for field, mutation in (
            (
                "event_counts_by_role",
                lambda value: value.pop("robot"),
            ),
            (
                "steps_with_contact_by_role",
                lambda value: value.update({"extra": []}),
            ),
        ):
            result, _ = self._result_for_path(path)
            contacts = result["failure_diagnostics"]["contacts"]
            mutation(contacts[field])
            with self.subTest(field=field), self.assertRaisesRegex(
                ReceiptError,
                f"contact {field} keys differs",
            ):
                artifacts._compact_population_diagnostic_evidence(
                    result,
                    diagnostic_validation=self._diagnostic_evidence(result),
                )

    def test_requires_v3_complete_contact_role_authority(self) -> None:
        target = self.manifests[2]["case_id"]
        for field, replacement, message in (
            (
                "schema_version",
                "vlsa_table1_active_obstacle_contacts.v2",
                "detailed contact schema",
            ),
            (
                "role_authority_complete",
                False,
                "contact role authority",
            ),
            (
                "role_taxonomy",
                ["robot", "unknown"],
                "contact role taxonomy",
            ),
            (
                "active_obstacle_name",
                "wrong_obstacle",
                "contact active-obstacle binding",
            ),
        ):
            def result_loader(path, **kwargs):
                result, item = self._result_for_path(path, **kwargs)
                if (
                    path.parent.name == target
                    and path.parents[1].name == "pi05"
                ):
                    result["failure_diagnostics"]["contacts"][field] = (
                        replacement
                    )
                return result, item

            with (
                self.subTest(field=field),
                self.assertRaisesRegex(ReceiptError, message),
            ):
                self._run_with_plain_patches(result_loader=result_loader)

    def test_rejects_any_extra_result_artifact(self) -> None:
        extra = self.result_root / "aegis/extra-case/result.json"
        extra.parent.mkdir(parents=True)
        extra.touch()
        with self.assertRaisesRegex(
            ReceiptError,
            "exact result inventory",
        ):
            self._run_with_plain_patches()


if __name__ == "__main__":
    unittest.main()
