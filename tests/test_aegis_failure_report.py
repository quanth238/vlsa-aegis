"""Dependency-light tests for exhaustive post-population failure analysis."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "analysis/build_aegis_failure_report.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


failure_report = load_module("aegis_failure_report", SCRIPT)


def metrics(*, collision: bool) -> dict:
    return {
        "paper_collision": collision,
        "collision_first_step": 3 if collision else None,
    }


def ready_geometry() -> dict:
    return {"status": "complete", "failure_type": None}


def qp_control(*, barrier: float = 1.0) -> dict:
    return {
        "collision_step_control_path": "aegis_qp",
        "collision_step_barrier_h": barrier,
    }


def contacts(
    *,
    robot: int = 0,
    static_support: int = 0,
    dynamic_task_object: int = 0,
    dynamic_other: int = 0,
    settled_robot: int = 0,
    settled_static_support: int = 0,
    settled_dynamic_task_object: int = 0,
    settled_dynamic_other: int = 0,
) -> dict:
    return {
        "settled_event_count": (
            settled_robot
            + settled_static_support
            + settled_dynamic_task_object
            + settled_dynamic_other
        ),
        "settled_collision_relevant_event_count": (
            settled_robot
            + settled_dynamic_task_object
            + settled_dynamic_other
        ),
        "event_counts_through_collision_by_role": {
            "robot": robot,
            "static_support": static_support,
            "dynamic_task_object": dynamic_task_object,
            "dynamic_other": dynamic_other,
            "unknown": 0,
        },
    }


class AegisFailureReportTest(unittest.TestCase):
    def test_contact_evidence_keeps_support_but_excludes_it_from_failure(
        self,
    ) -> None:
        role_authority = {
            "status": "complete",
            "role_classes": list(failure_report.CONTACT_ROLES),
        }
        payload = {
            "snapshots": [
                {
                    "step": -1,
                    "role_authority": role_authority,
                    "events": [
                        {
                            "step": -1,
                            "other": {
                                "classification": "static_support",
                                "classification_authority": "complete",
                                "body_name": "table",
                            },
                        }
                    ],
                },
                {
                    "step": 0,
                    "role_authority": role_authority,
                    "events": [
                        {
                            "step": 0,
                            "other": {
                                "classification": "static_support",
                                "classification_authority": "complete",
                                "body_name": "table",
                            },
                        },
                        {
                            "step": 0,
                            "other": {
                                "classification": "dynamic_task_object",
                                "classification_authority": "complete",
                                "body_name": "goal_object",
                            },
                        },
                    ],
                },
            ]
        }
        evidence = failure_report._contact_evidence(
            payload, collision_step=0
        )
        self.assertEqual(evidence["settled_event_count"], 1)
        self.assertEqual(
            evidence["settled_collision_relevant_event_count"], 0
        )
        self.assertEqual(
            evidence[
                "static_support_events_excluded_from_collision_role"
            ],
            1,
        )
        self.assertEqual(
            evidence["event_counts_through_collision_by_role"][
                "dynamic_task_object"
            ],
            1,
        )

    def test_contact_evidence_rejects_unknown_role_authority(self) -> None:
        payload = {
            "snapshots": [
                {
                    "step": -1,
                    "role_authority": {
                        "status": "unknown",
                        "role_classes": list(
                            failure_report.CONTACT_ROLES
                        ),
                    },
                    "events": [],
                }
            ]
        }
        with self.assertRaisesRegex(
            failure_report.FailureReportError,
            "lacks complete authoritative roles",
        ):
            failure_report._contact_evidence(
                payload, collision_step=None
            )

    def test_noncollision_has_only_nonfailure_class(self) -> None:
        observed, hypotheses = failure_report.classify_car_failure(
            metrics=metrics(collision=False),
            geometry=ready_geometry(),
            control=qp_control(),
            contacts=contacts(),
        )
        self.assertEqual(observed, "not_car_failure")
        self.assertEqual(hypotheses, [])

    def test_no_grounding_fail_open_is_directly_observed(self) -> None:
        observed, hypotheses = failure_report.classify_car_failure(
            metrics=metrics(collision=True),
            geometry={
                "status": "method_failure_passthrough",
                "failure_type": "no_grounded_points",
            },
            control={
                "collision_step_control_path": (
                    "corrected_translational_nominal"
                )
            },
            contacts=contacts(),
        )
        self.assertEqual(
            observed, "grounding_no_points_fail_open_collision"
        )
        self.assertIn("perception_fail_open", hypotheses)

    def test_point_filter_fail_open_is_separate(self) -> None:
        observed, _ = failure_report.classify_car_failure(
            metrics=metrics(collision=True),
            geometry={
                "status": "method_failure_passthrough",
                "failure_type": "point_filter_removed_all_points",
            },
            control={
                "collision_step_control_path": (
                    "corrected_translational_nominal"
                )
            },
            contacts=contacts(),
        )
        self.assertEqual(
            observed, "filter_removed_all_points_fail_open_collision"
        )

    def test_settled_contact_is_not_misattributed_to_qp(self) -> None:
        observed, _ = failure_report.classify_car_failure(
            metrics={
                "paper_collision": True,
                "collision_first_step": 0,
            },
            geometry=ready_geometry(),
            control=qp_control(),
            contacts=contacts(settled_robot=1, robot=1),
        )
        self.assertEqual(
            observed,
            "preexisting_settled_robot_or_dynamic_contact_then_collision",
        )

    def test_settled_contact_remains_preexisting_if_car_crosses_later(
        self,
    ) -> None:
        observed, _ = failure_report.classify_car_failure(
            metrics={
                "paper_collision": True,
                "collision_first_step": 3,
            },
            geometry=ready_geometry(),
            control=qp_control(),
            contacts=contacts(settled_dynamic_other=1),
        )
        self.assertEqual(
            observed,
            "preexisting_settled_robot_or_dynamic_contact_then_collision",
        )

    def test_settled_static_support_is_not_preexisting_collision(self) -> None:
        observed, _ = failure_report.classify_car_failure(
            metrics={
                "paper_collision": True,
                "collision_first_step": 0,
            },
            geometry=ready_geometry(),
            control=qp_control(),
            contacts=contacts(
                static_support=1,
                settled_static_support=1,
            ),
        )
        self.assertEqual(
            observed,
            "valid_qp_displacement_without_robot_or_dynamic_sampled_contact",
        )

    def test_positive_proxy_barrier_with_robot_contact_is_mismatch(self) -> None:
        observed, hypotheses = failure_report.classify_car_failure(
            metrics=metrics(collision=True),
            geometry=ready_geometry(),
            control=qp_control(barrier=0.25),
            contacts=contacts(robot=1),
        )
        self.assertEqual(
            observed,
            "valid_qp_robot_contact_positive_proxy_barrier",
        )
        self.assertIn(
            "proxy_simulator_geometry_mismatch_observed", hypotheses
        )

    def test_nonpositive_barrier_is_recovery_stratum(self) -> None:
        observed, hypotheses = failure_report.classify_car_failure(
            metrics=metrics(collision=True),
            geometry=ready_geometry(),
            control=qp_control(barrier=0.0),
            contacts=contacts(robot=1),
        )
        self.assertEqual(
            observed,
            "valid_qp_robot_contact_nonpositive_proxy_barrier",
        )
        self.assertIn("discrete_recovery_limit_hypothesis", hypotheses)

    def test_dynamic_task_contact_is_not_called_robot_failure(self) -> None:
        observed, _ = failure_report.classify_car_failure(
            metrics=metrics(collision=True),
            geometry=ready_geometry(),
            control=qp_control(),
            contacts=contacts(dynamic_task_object=2),
        )
        self.assertEqual(
            observed,
            "valid_qp_dynamic_task_object_contact_collision",
        )

    def test_dynamic_other_contact_has_its_own_class(self) -> None:
        observed, _ = failure_report.classify_car_failure(
            metrics=metrics(collision=True),
            geometry=ready_geometry(),
            control=qp_control(),
            contacts=contacts(dynamic_other=2),
        )
        self.assertEqual(
            observed,
            "valid_qp_dynamic_other_contact_collision",
        )

    def test_mixed_collision_relevant_contact_has_explicit_class(
        self,
    ) -> None:
        observed, _ = failure_report.classify_car_failure(
            metrics=metrics(collision=True),
            geometry=ready_geometry(),
            control=qp_control(),
            contacts=contacts(robot=1, dynamic_other=1),
        )
        self.assertEqual(
            observed,
            "valid_qp_mixed_collision_relevant_contact_roles",
        )

    def test_no_relevant_sampled_contact_preserves_evidence_limit(
        self,
    ) -> None:
        observed, hypotheses = failure_report.classify_car_failure(
            metrics=metrics(collision=True),
            geometry=ready_geometry(),
            control=qp_control(),
            contacts=contacts(static_support=20),
        )
        self.assertEqual(
            observed,
            "valid_qp_displacement_without_robot_or_dynamic_sampled_contact",
        )
        self.assertIn(
            "metric_proxy_or_contact_substep_sampling_hypothesis",
            hypotheses,
        )

    def test_unregistered_collision_control_path_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            failure_report.FailureReportError,
            "no registered QP/fail-open path",
        ):
            failure_report.classify_car_failure(
                metrics=metrics(collision=True),
                geometry=ready_geometry(),
                control={"collision_step_control_path": "unknown"},
                contacts=contacts(),
            )

    def test_task_failure_taxonomy_marks_strict_stopping(self) -> None:
        observed = failure_report.classify_task_failure(
            baseline_metrics={"task_success": True},
            aegis_metrics={
                "task_success": False,
                "paper_collision": False,
                "executed_action_count": 50,
            },
            control={"strict_zero_translation": True},
        )
        self.assertEqual(
            observed, "safe_strict_zero_translation_task_failure"
        )

    def test_task_failure_taxonomy_does_not_call_detour_stopping(self) -> None:
        observed = failure_report.classify_task_failure(
            baseline_metrics={"task_success": True},
            aegis_metrics={
                "task_success": False,
                "paper_collision": False,
                "executed_action_count": 50,
            },
            control={"strict_zero_translation": False},
        )
        self.assertEqual(
            observed, "safe_task_regression_vs_successful_baseline"
        )

    def test_summary_requires_every_failure_classified_and_every_video(
        self,
    ) -> None:
        records = []
        for index in range(2):
            collision = index == 0
            records.append(
                {
                    "case_id": f"case-{index}",
                    "suite": "safelibero_spatial",
                    "failure_analysis": {
                        "is_aegis_car_failure": collision,
                        "primary_observed_car_class": (
                            "valid_qp_displacement_without_robot_or_dynamic_"
                            "sampled_contact"
                            if collision
                            else "not_car_failure"
                        ),
                        "aegis_task_failure_class": "not_task_failure",
                        "causal_hypothesis_tags": [],
                    },
                    "outcomes": {
                        failure_report.AEGIS_ARM: {
                            "paper_collision": collision,
                            "task_success": True,
                            "legacy_ets_steps": 9,
                        }
                    },
                    "videos": {
                        failure_report.BASELINE_ARM: {
                            "hash_verified": True
                        },
                        failure_report.AEGIS_ARM: {
                            "hash_verified": True
                        },
                    },
                }
            )
        summary = {
            "schema_version": failure_report.SUMMARY_SCHEMA,
            "status": "complete_population_validated",
            "population": {"no_results_dropped": True},
            "suites": {
                failure_report.AEGIS_ARM: {
                    "safelibero_spatial": {
                        "episodes": 2,
                        "car_percent": 50.0,
                        "tsr_percent": 100.0,
                        "legacy_ets_steps_mean": 9.0,
                    }
                }
            },
        }
        counts = failure_report.validate_report_against_summary(
            records,
            summary=summary,
            expected_cases=2,
        )
        self.assertEqual(counts["aegis_car_failure_count"], 1)
        self.assertEqual(counts["video_count"], 4)
        self.assertEqual(counts["unclassified_aegis_car_failures"], 0)

    def test_summary_rejects_class_outcome_mismatch(self) -> None:
        record = {
            "case_id": "case-0",
            "suite": "safelibero_spatial",
            "failure_analysis": {
                "is_aegis_car_failure": True,
                "primary_observed_car_class": "not_car_failure",
                "aegis_task_failure_class": "not_task_failure",
                "causal_hypothesis_tags": [],
            },
            "outcomes": {
                failure_report.AEGIS_ARM: {
                    "paper_collision": True,
                    "task_success": False,
                    "legacy_ets_steps": 9,
                }
            },
            "videos": {
                failure_report.BASELINE_ARM: {"hash_verified": True},
                failure_report.AEGIS_ARM: {"hash_verified": True},
            },
        }
        summary = {
            "schema_version": failure_report.SUMMARY_SCHEMA,
            "status": "complete_population_validated",
            "population": {"no_results_dropped": True},
            "suites": {
                failure_report.AEGIS_ARM: {
                    "safelibero_spatial": {
                        "episodes": 1,
                        "car_percent": 0.0,
                        "tsr_percent": 0.0,
                        "legacy_ets_steps_mean": 9.0,
                    }
                }
            },
        }
        with self.assertRaisesRegex(
            failure_report.FailureReportError,
            "CAR outcome/class mismatch",
        ):
            failure_report.validate_report_against_summary(
                [record],
                summary=summary,
                expected_cases=1,
            )

    def test_population_receipt_requires_complete_contact_v3_authority(
        self,
    ) -> None:
        receipt = {
            "schema_version": failure_report.PREPUBLISH_SCHEMA,
            "status": "validated",
            "complete_paired_population": True,
            "no_results_dropped": True,
            "result_artifacts": {"count": 4},
            "failure_diagnostics": {
                "count": 4,
                "all_results_deep_validated": True,
                "contact_schema_version": failure_report.CONTACT_SCHEMA,
                "contact_model_authority_schema_version": (
                    failure_report.CONTACT_MODEL_AUTHORITY_SCHEMA
                ),
                "contact_role_taxonomy": list(
                    failure_report.CONTACT_ROLES
                ),
                "contact_role_authority_complete": True,
                "unknown_role_event_count": 0,
            },
        }
        failure_report._require_population_validation_receipt(
            receipt,
            expected_cases=2,
        )
        receipt["failure_diagnostics"]["unknown_role_event_count"] = 1
        with self.assertRaisesRegex(
            failure_report.FailureReportError,
            "contact-v3 evidence",
        ):
            failure_report._require_population_validation_receipt(
                receipt,
                expected_cases=2,
            )


if __name__ == "__main__":
    unittest.main()
