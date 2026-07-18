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

    def test_no_usable_points_subclasses_box_and_point_evidence(
        self,
    ) -> None:
        def geometry(agent: int, back: int) -> dict:
            return {
                "views": {
                    "agentview": {
                        "status": (
                            "detected" if agent else "no_detection"
                        ),
                        "detection_count": agent,
                        "usable_point_count": 0,
                    },
                    "backview": {
                        "status": (
                            "detected" if back else "no_detection"
                        ),
                        "detection_count": back,
                        "usable_point_count": 0,
                    },
                }
            }

        expected = {
            (0, 0): "both_views_no_detection",
            (1, 0): (
                "agentview_detection_without_usable_3d_points"
            ),
            (0, 1): (
                "backview_detection_without_usable_3d_points"
            ),
            (1, 1): (
                "both_views_detections_without_usable_3d_points"
            ),
        }
        for counts, expected_subclass in expected.items():
            with self.subTest(counts=counts):
                subclass, hypotheses = (
                    failure_report._no_usable_points_evidence(
                        geometry(*counts)
                    )
                )
                self.assertEqual(subclass, expected_subclass)
                self.assertTrue(
                    all(
                        value.endswith("_hypothesis")
                        for value in hypotheses
                    )
                )
        invalid = geometry(1, 0)
        invalid["views"]["agentview"]["usable_point_count"] = 2
        with self.assertRaisesRegex(
            failure_report.FailureReportError,
            "evidence is inconsistent",
        ):
            failure_report._no_usable_points_evidence(invalid)

    def test_v2_point_evidence_does_not_change_v1_geometry_row(
        self,
    ) -> None:
        result = {
            "case_id": "case-0",
            "failure_diagnostics": {
                "schema_version": (
                    "vlsa_table1_aegis_failure_diagnostics.v1"
                ),
                "enabled": True,
                "status": "published",
                "control_effect": "read_only_observation",
                "geometry": {
                    "record": {
                        "status": "method_failure_passthrough",
                        "views": {
                            "agentview": {
                                "status": "no_detection",
                                "detections": {
                                    "count": 0,
                                    "selected_index": None,
                                },
                                "returned_point_cloud": {
                                    "shape": [1, 0],
                                    "finite": True,
                                },
                            },
                            "backview": {
                                "status": "detected",
                                "detections": {
                                    "count": 1,
                                    "selected_index": 0,
                                },
                                "returned_point_cloud": {
                                    "shape": [0, 3],
                                    "finite": True,
                                },
                            },
                        },
                        "filtering": None,
                        "mvee": None,
                        "failure": {
                            "component": "grounding",
                            "type": "no_grounded_points",
                        },
                    }
                },
            },
        }
        v1 = failure_report._geometry_evidence(result)
        v2 = failure_report._geometry_evidence_v2(result)
        self.assertEqual(
            set(v1["views"]["agentview"]),
            {"status", "detection_count", "selected_index"},
        )
        self.assertEqual(
            v2["views"]["agentview"]["returned_point_shape"],
            [1, 0],
        )
        self.assertEqual(
            v2["views"]["agentview"]["usable_point_count"],
            0,
        )

    def test_safe_no_points_case_remains_not_a_car_failure(self) -> None:
        geometry = {
            "status": "method_failure_passthrough",
            "failure_type": "no_grounded_points",
            "views": {
                "agentview": {
                    "status": "no_detection",
                    "detection_count": 0,
                    "usable_point_count": 0,
                },
                "backview": {
                    "status": "no_detection",
                    "detection_count": 0,
                    "usable_point_count": 0,
                },
            },
        }
        primary, subclass, hypotheses = (
            failure_report._refine_no_usable_points_v2(
                metrics={"paper_collision": False},
                geometry=geometry,
                primary_class="not_car_failure",
                causal_hypotheses=[],
            )
        )
        self.assertEqual(primary, "not_car_failure")
        self.assertEqual(subclass, "both_views_no_detection")
        self.assertTrue(hypotheses)

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

    def test_zero_translation_ignores_nonzero_gripper_channel(
        self,
    ) -> None:
        self.assertEqual(
            failure_report._vector_norm(
                [0.0, 0.0, 0.0, 0.25, -0.25, 0.5, -1.0],
                label="action",
            ),
            0.0,
        )
        self.assertGreater(
            failure_report._vector_norm(
                [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
                label="action",
            ),
            0.0,
        )

    def test_v2_observations_are_separate_from_causal_hypotheses(
        self,
    ) -> None:
        tags = failure_report._observed_evidence_tags_v2(
            metrics={"paper_collision": True, "task_success": False},
            geometry={"status": "method_failure_passthrough"},
            control={
                "collision_step_control_path": (
                    "corrected_translational_nominal"
                ),
                "collision_step_barrier_h": None,
            },
            contacts={
                "robot_active_obstacle_contact": False,
                "settled_collision_relevant_event_count": 0,
            },
            no_points_subclass="both_views_no_detection",
        )
        self.assertIn("paper_car_collision", tags)
        self.assertIn("both_views_no_detection", tags)
        self.assertIn(
            "no_usable_3d_points_fail_open_path_observed", tags
        )
        self.assertNotIn("perception_fail_open", tags)

    def test_analysis_v2_reports_joint_contact_and_cross_tabs(
        self,
    ) -> None:
        records = []
        outcomes = (
            (True, False, False, True),
            (False, True, True, False),
        )
        for index, (
            baseline_collision,
            baseline_success,
            aegis_collision,
            aegis_success,
        ) in enumerate(outcomes):
            def arm_outcome(
                collision: bool,
                success: bool,
                robot_contact: bool,
            ) -> dict:
                joint = (
                    ("collision" if collision else "safe")
                    + "_task_"
                    + ("success" if success else "failure")
                )
                return {
                    "paper_collision": collision,
                    "task_success": success,
                    "legacy_ets_steps": 9,
                    "executed_action_count": 10,
                    "joint_outcome": joint,
                    "sampled_robot_active_obstacle_contact": (
                        robot_contact
                    ),
                    "sampled_postcontrol_robot_active_obstacle_contact": (
                        robot_contact
                    ),
                    "sampled_active_obstacle_contact_any": robot_contact,
                    "sampled_collision_relevant_active_obstacle_contact": (
                        robot_contact
                    ),
                    "paper_car_contact_stratum": (
                        failure_report._paper_car_contact_stratum(
                            paper_collision=collision,
                            robot_contact=robot_contact,
                        )
                    ),
                }

            primary = (
                "valid_qp_robot_contact_positive_proxy_barrier"
                if aegis_collision
                else "not_car_failure"
            )
            baseline_joint = arm_outcome(
                baseline_collision,
                baseline_success,
                robot_contact=baseline_collision,
            )
            aegis_joint = arm_outcome(
                aegis_collision,
                aegis_success,
                robot_contact=aegis_collision,
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
                "table_scope": (
                    "two-row translational Table-1 reproduction: "
                    "pi0.5 and pi0.5+AEGIS only"
                ),
                "openvla_oft_included": False,
                "paper_semantic_selector_reproduced": False,
                "paper_exact_end_to_end_reproduction_claimed": False,
                "clearance_available": False,
                "minimum_clearance_claimed": False,
            }
            goal_baseline = {
                "goal_definition_sha256": "a" * 64,
                "initial_satisfied_count": 0,
                "final_satisfied_count": 0,
                "maximum_satisfied_count": 0,
                "regression_count": 0,
                "initial_fraction": 0.0,
                "final_fraction": 0.0,
                "maximum_fraction": 0.0,
            }
            goal_aegis = {
                **goal_baseline,
                "final_satisfied_count": 1,
                "maximum_satisfied_count": 1,
                "final_fraction": 0.5,
                "maximum_fraction": 0.5,
            }
            records.append(
                {
                    "case_id": f"case-{index}",
                    "case_ordinal": index,
                    "record_payload_sha256": (
                        f"{index + 1:064x}"
                    ),
                    "suite": "safelibero_spatial",
                    "safety_level": "I",
                    "logical_task_index": 0,
                    "task_name": "task",
                    "frozen_obstacle_label": "blue moka pot",
                    "failure_analysis": {
                        "is_aegis_car_failure": aegis_collision,
                        "primary_observed_car_class": primary,
                        "aegis_task_failure_class": (
                            "not_task_failure"
                            if aegis_success
                            else "safe_shared_task_failure"
                        ),
                        "causal_hypothesis_tags": [],
                        "legacy_v1_strict_safety_by_stopping_flag": (
                            index == 0
                        ),
                        "safe_with_strict_zero_translation": index == 0,
                        "no_usable_points_observed_subclass": (
                            "not_applicable"
                        ),
                    },
                    "outcomes": {
                        failure_report.BASELINE_ARM: baseline_joint,
                        failure_report.AEGIS_ARM: aegis_joint,
                    },
                    "paired_transition": {
                        "car": (
                            (
                                "baseline_collision"
                                if baseline_collision
                                else "baseline_safe"
                            )
                            + "_to_"
                            + (
                                "aegis_collision"
                                if aegis_collision
                                else "aegis_safe"
                            )
                        ),
                        "task": (
                            (
                                "baseline_success"
                                if baseline_success
                                else "baseline_failure"
                            )
                            + "_to_"
                            + (
                                "aegis_success"
                                if aegis_success
                                else "aegis_failure"
                            )
                        ),
                        "joint": (
                            "baseline_"
                            + baseline_joint["joint_outcome"]
                            + "_to_aegis_"
                            + aegis_joint["joint_outcome"]
                        ),
                    },
                    "aegis_diagnostics": {
                        "intervention": {
                            "eligible_steps": 10,
                            "intervention_count": 3,
                            "correction_l2_sum": 1.25,
                        }
                    },
                    "paired_goal_progress": {
                        failure_report.BASELINE_ARM: goal_baseline,
                        failure_report.AEGIS_ARM: goal_aegis,
                        "same_goal_definition": True,
                        "same_initial_goal_progress": True,
                        "aegis_minus_pi05": {
                            "initial_satisfied_count": 0,
                            "final_satisfied_count": 1,
                            "maximum_satisfied_count": 1,
                            "initial_fraction": 0.0,
                            "final_fraction": 0.5,
                            "maximum_fraction": 0.5,
                            "regression_count": 0,
                        }
                    },
                    "claim_scope": claim_scope,
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
        def arm_summary(arm: str) -> dict:
            arm_outcomes = [row["outcomes"][arm] for row in records]
            car_successes = sum(
                not row["paper_collision"] for row in arm_outcomes
            )
            tsr_successes = sum(
                row["task_success"] for row in arm_outcomes
            )
            return {
                "episodes": 2,
                "car_percent": 100.0 * car_successes / 2,
                "tsr_percent": 100.0 * tsr_successes / 2,
                "legacy_ets_steps_mean": 9.0,
                "joint_outcome_counts": (
                    failure_report._joint_outcome_counts(
                        records, arm=arm
                    )
                ),
                "car": {
                    "success_count": car_successes,
                    "failure_count": 2 - car_successes,
                    "denominator": 2,
                },
                "tsr": {
                    "success_count": tsr_successes,
                    "failure_count": 2 - tsr_successes,
                    "denominator": 2,
                },
                "legacy_ets_steps": {"sum": 18, "denominator": 2},
                "executed_action_count": {
                    "sum": 20,
                    "denominator": 2,
                },
            }
        paired = failure_report._paired_transition_counts_v2(records)
        summary = {
            "schema_version": failure_report.SUMMARY_SCHEMA_V2,
            "status": "complete_postpublication_analysis_v2",
            "population": {"no_results_dropped": True},
            "source_v1": {
                "v1_publication_receipt_sha256": "b" * 64,
            },
            "accepted_result_payloads_sha256": "a" * 64,
            "claim_scope": claim_scope,
            "suites": {
                failure_report.AEGIS_ARM: {
                    "safelibero_spatial": arm_summary(
                        failure_report.AEGIS_ARM
                    )
                }
            },
            "average": {
                failure_report.BASELINE_ARM: arm_summary(
                    failure_report.BASELINE_ARM
                ),
                failure_report.AEGIS_ARM: arm_summary(
                    failure_report.AEGIS_ARM
                ),
            },
            "paired_transitions": {
                "overall": {
                    field: paired[field]
                    for field in ("denominator", "car", "task", "joint")
                }
            },
        }
        counts = failure_report.validate_report_against_summary_v2(
            records,
            summary=summary,
            expected_cases=2,
        )
        self.assertEqual(
            counts["physical_contacts"][
                failure_report.AEGIS_ARM
            ]["sampled_robot_contact_count"],
            1,
        )
        self.assertEqual(
            counts["physical_contacts"][
                failure_report.AEGIS_ARM
            ][
                "paper_car_sampled_robot_contact_disagreement_count"
            ],
            0,
        )
        self.assertEqual(
            counts["paired_goal_progress"][
                "aegis_minus_pi05_sum"
            ]["final_satisfied_count"],
            2.0,
        )
        self.assertEqual(
            counts["intervention"]["modified_action_count"], 6
        )
        self.assertEqual(
            counts["paired_transitions"]["denominator"], 2
        )
        self.assertEqual(
            sum(
                counts["paired_transitions"][
                    "paper_car_sampled_robot_contact"
                ].values()
            ),
            2,
        )
        self.assertEqual(
            counts["no_usable_points_observed_subclass_counts"][
                "not_applicable"
            ],
            2,
        )
        self.assertEqual(
            counts["cross_tabs"]["frozen_obstacle_label"][
                "blue moka pot"
            ]["denominator"],
            2,
        )
        report = failure_report.build_report_v2(
            records,
            summary=summary,
            summary_sha256="c" * 64,
            validation_receipt_sha256="d" * 64,
            source_publication_receipt_sha256="b" * 64,
            expected_cases=2,
        )
        self.assertEqual(
            report["source"]["v1_publication_receipt_sha256"],
            summary["source_v1"][
                "v1_publication_receipt_sha256"
            ],
        )
        with self.assertRaisesRegex(
            failure_report.FailureReportError,
            "source differs from its summary",
        ):
            failure_report.build_report_v2(
                records,
                summary=summary,
                summary_sha256="c" * 64,
                validation_receipt_sha256="d" * 64,
                source_publication_receipt_sha256="e" * 64,
                expected_cases=2,
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

    def test_v1_markdown_does_not_gain_v2_claim_fields(self) -> None:
        report = {
            "counts": {
                "case_count": 1,
                "video_count": 2,
                "aegis_car_percent": 100.0,
                "aegis_car_failure_count": 0,
                "primary_car_failure_class_counts": {
                    "not_car_failure": 1
                },
                "evidence_limited_car_failure_count": 0,
                "aegis_task_failure_class_counts": {
                    "not_task_failure": 1
                },
            }
        }
        markdown = failure_report.render_markdown(report)
        for key in failure_report.INTERPRETATION_SCOPE:
            self.assertIn(f"- **{key}:**", markdown)
        for key in set(failure_report.INTERPRETATION_SCOPE_V2) - set(
            failure_report.INTERPRETATION_SCOPE
        ):
            self.assertNotIn(f"- **{key}:**", markdown)


if __name__ == "__main__":
    unittest.main()
