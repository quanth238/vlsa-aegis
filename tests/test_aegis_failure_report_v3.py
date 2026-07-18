"""Focused tests for decision-aligned postpublication analysis v3."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import unittest
from unittest import mock

from analysis import build_aegis_failure_report_v3 as report


def contact_payload() -> dict:
    body_names = [
        "world",
        "eef_marker",
        "robot0_base",
        "robot0_link7",
        "robot0_right_hand",
        "gripper0_right_gripper",
        "gripper0_rightfinger",
        "moka_pot_obstacle_1_main",
    ]
    return {
        "model_authority": {
            "authority_sha256": "a" * 64,
            "body_names": body_names,
            "body_parent_ids": [0, 0, 0, 2, 3, 4, 5, 0],
            "robot_body_ids": [1, 2, 3, 4, 5, 6],
        },
        "snapshots": [
            {
                "step": -1,
                "events": [
                    robot_event(
                        body_id=1,
                        body_name="eef_marker",
                        geom_name="eef_marker_geom",
                    )
                ],
            },
            {
                "step": 0,
                "events": [
                    robot_event(
                        body_id=3,
                        body_name="robot0_link7",
                        geom_name="robot0_link7_collision",
                    )
                ],
            },
            {
                "step": 1,
                "events": [
                    robot_event(
                        body_id=6,
                        body_name="gripper0_rightfinger",
                        geom_name="gripper0_finger2_collision",
                    ),
                    {
                        "other": {
                            "classification": "static_support",
                            "classification_authority": "complete",
                            "body_id": 0,
                            "body_name": "world",
                            "geom_name": "table",
                        }
                    },
                ],
            },
        ],
    }


def robot_event(*, body_id: int, body_name: str, geom_name: str) -> dict:
    return {
        "other": {
            "classification": "robot",
            "classification_authority": "complete",
            "body_id": body_id,
            "body_name": body_name,
            "geom_name": geom_name,
        }
    }


def goal_snapshot(
    step: int,
    values: list[bool],
    *,
    regressed_indices: list[int] | None = None,
) -> dict:
    return {
        "step": step,
        "values": values,
        "regressed_indices": (
            [] if regressed_indices is None else regressed_indices
        ),
    }


def paired_results() -> tuple[dict, dict]:
    zero = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
    baseline_nominal = [
        zero,
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
        [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
        [3.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
    ]
    aegis_nominal = [
        zero,
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
        [2.5, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
        [4.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
    ]
    baseline_values = [
        [False, True],
        [False, True],
        [False, True],
        [True, True],
    ]
    aegis_values = [
        [False, True],
        [False, True],
        [False, True],
        [False, False],
    ]

    def actions(
        nominal: list[list[float]],
        values: list[list[bool]],
        *,
        aegis: bool,
    ) -> list[dict]:
        return [
            {
                "step": index,
                "modified": aegis and index == 1,
                "nominal_translational": vector,
                "goal_progress": goal_snapshot(
                    index,
                    values[index],
                    regressed_indices=(
                        [1] if aegis and index == 3 else []
                    ),
                ),
            }
            for index, vector in enumerate(nominal)
        ]

    initial = goal_snapshot(-1, [False, True])
    baseline = {
        "case_id": "case-0",
        "result_payload_sha256": "b" * 64,
        "actions": actions(
            baseline_nominal, baseline_values, aegis=False
        ),
        "goal_progress": {"initial": copy.deepcopy(initial)},
        "policy_queries": [
            {
                "query_index": 0,
                "returned_actions_sha256": "1" * 64,
            },
            {
                "query_index": 1,
                "returned_actions_sha256": "2" * 64,
            },
        ],
    }
    aegis = {
        "case_id": "case-0",
        "result_payload_sha256": "c" * 64,
        "actions": actions(aegis_nominal, aegis_values, aegis=True),
        "goal_progress": {"initial": copy.deepcopy(initial)},
        "policy_queries": [
            {
                "query_index": 0,
                "returned_actions_sha256": "1" * 64,
            },
            {
                "query_index": 1,
                "returned_actions_sha256": "3" * 64,
            },
        ],
    }
    return baseline, aegis


def decision_record() -> dict:
    return {
        "task_level_group_id": "group-0",
        "outcomes": {
            report.BASELINE_ARM: {
                "task_success": True,
                "paper_collision": True,
            },
            report.AEGIS_ARM: {
                "task_success": False,
                "paper_collision": False,
                "executed_action_count": 4,
            },
        },
        "aegis_diagnostics": {
            "geometry_v3": {
                "status": "complete",
                "mvee_status": "captured",
            },
            "control": {
                "control_path_counts": {"aegis_qp": 4},
                "terminal_qp_failure": None,
                "translation_retention_ratio": 0.8,
                "strict_zero_translation": False,
            },
            "intervention": {
                "intervention_count": 1,
                "correction_l2_sum": 0.5,
            },
        },
        "physical_contacts": {
            report.BASELINE_ARM: {
                "settled_collision_relevant_event_count": 0,
                "collision_relevant_event_count": 1,
                "postcontrol_collision_relevant_event_count": 1,
            },
            report.AEGIS_ARM: {
                "settled_collision_relevant_event_count": 0,
                "collision_relevant_event_count": 0,
                "postcontrol_collision_relevant_event_count": 0,
                "robot_contact_scope_v3": {
                    "sampled_unprotected_robot_link_contact": False,
                },
            }
        },
        "paired_goal_progress": {
            "aegis_minus_pi05": {
                "final_fraction": -0.5,
                "maximum_fraction": -0.5,
                "regression_count": 1,
            }
        },
        "paired_temporal_evidence_v3": {
            "intervention_precedes_or_coincides_nominal_divergence": True,
            "intervention_precedes_or_coincides_goal_divergence": True,
            "intervention_strictly_precedes_nominal_divergence": True,
            "nominal_divergence_precedes_or_coincides_goal_divergence": (
                True
            ),
            "registered_intervention_nominal_goal_chain": True,
        },
    }


class RobotContactScopeV3Tests(unittest.TestCase):
    def test_exact_hand_subtree_partition_and_histograms(self) -> None:
        evidence = report.contact_scope_evidence_v3(
            contact_payload(), collision_step=0
        )
        authority = evidence["authority"]
        self.assertEqual(
            authority["body_ids_by_scope"]["intended_eef_subtree"],
            [4, 5, 6],
        )
        self.assertEqual(
            authority["body_ids_by_scope"]["unprotected_robot_link"],
            [2, 3],
        )
        self.assertEqual(
            authority["body_ids_by_scope"]["diagnostic_eef_marker"],
            [1],
        )
        self.assertEqual(evidence["robot_event_count"], 3)
        self.assertEqual(
            evidence["event_counts_by_scope"],
            {
                "intended_eef_subtree": 1,
                "unprotected_robot_link": 1,
                "diagnostic_eef_marker": 1,
            },
        )
        self.assertEqual(
            evidence["events_through_paper_collision_by_scope"],
            {
                "intended_eef_subtree": 0,
                "unprotected_robot_link": 1,
                "diagnostic_eef_marker": 0,
            },
        )
        self.assertEqual(
            evidence["body_event_histograms_by_scope"][
                "intended_eef_subtree"
            ],
            {"gripper0_rightfinger": 1},
        )
        self.assertTrue(
            evidence["sampled_unprotected_robot_link_contact"]
        )
        self.assertFalse(
            evidence["ellipsoid_geometric_enclosure_certified"]
        )

    def test_topology_authority_fails_closed(self) -> None:
        fixtures: list[tuple[str, callable]] = [
            (
                "duplicate hand",
                lambda payload: payload["model_authority"][
                    "body_names"
                ].__setitem__(6, "robot0_right_hand"),
            ),
            (
                "missing marker",
                lambda payload: payload["model_authority"][
                    "body_names"
                ].__setitem__(1, "marker_missing"),
            ),
            (
                "marker in hand subtree",
                lambda payload: payload["model_authority"][
                    "body_parent_ids"
                ].__setitem__(1, 4),
            ),
            (
                "ancestry cycle",
                lambda payload: (
                    payload["model_authority"][
                        "body_parent_ids"
                    ].__setitem__(2, 3),
                    payload["model_authority"][
                        "body_parent_ids"
                    ].__setitem__(3, 2),
                ),
            ),
            (
                "invalid world root",
                lambda payload: payload["model_authority"][
                    "body_parent_ids"
                ].__setitem__(0, 2),
            ),
        ]
        for name, mutate in fixtures:
            with self.subTest(name=name):
                payload = contact_payload()
                mutate(payload)
                with self.assertRaises(report.FailureReportV3Error):
                    report.derive_robot_scope_authority(payload)

    def test_contact_identity_and_scope_fail_closed(self) -> None:
        unknown = contact_payload()
        unknown["snapshots"][0]["events"] = [
            robot_event(
                body_id=7,
                body_name="moka_pot_obstacle_1_main",
                geom_name="obstacle_geom",
            )
        ]
        with self.assertRaisesRegex(
            report.FailureReportV3Error, "lacks one exact contact scope"
        ):
            report.contact_scope_evidence_v3(
                unknown, collision_step=None
            )

        mismatch = contact_payload()
        mismatch["snapshots"][0]["events"][0]["other"][
            "body_name"
        ] = "robot0_right_hand"
        with self.assertRaisesRegex(
            report.FailureReportV3Error,
            "identity differs",
        ):
            report.contact_scope_evidence_v3(
                mismatch, collision_step=None
            )

    def test_tokenless_descendants_remain_in_robot_contact_scope(
        self,
    ) -> None:
        payload = contact_payload()
        model = payload["model_authority"]
        model["body_names"].extend(
            ["tokenless_finger_pad", "tokenless_link_mount"]
        )
        model["body_parent_ids"].extend([6, 3])
        payload["snapshots"] = [
            {
                "step": 0,
                "events": [
                    robot_event(
                        body_id=8,
                        body_name="tokenless_finger_pad",
                        geom_name="finger_pad_collision",
                    ),
                    robot_event(
                        body_id=9,
                        body_name="tokenless_link_mount",
                        geom_name="link_mount_collision",
                    ),
                ],
            }
        ]
        evidence = report.contact_scope_evidence_v3(
            payload, collision_step=0
        )
        authority = evidence["authority"]
        self.assertIn(
            8,
            authority["body_ids_by_scope"]["intended_eef_subtree"],
        )
        self.assertIn(
            9,
            authority["body_ids_by_scope"]["unprotected_robot_link"],
        )
        self.assertIn(
            8, authority["robot_contact_body_universe_ids"]
        )
        self.assertIn(
            9, authority["robot_contact_body_universe_ids"]
        )
        self.assertEqual(
            evidence["event_counts_by_scope"],
            {
                "intended_eef_subtree": 1,
                "unprotected_robot_link": 1,
                "diagnostic_eef_marker": 0,
            },
        )

    def test_population_scope_aggregation_preserves_denominators(self) -> None:
        evidence = report.contact_scope_evidence_v3(
            contact_payload(), collision_step=0
        )
        rows = [
            {
                "physical_contacts": {
                    report.BASELINE_ARM: {
                        "robot_contact_scope_v3": copy.deepcopy(
                            evidence
                        )
                    },
                    report.AEGIS_ARM: {
                        "robot_contact_scope_v3": copy.deepcopy(
                            evidence
                        )
                    },
                }
            }
        ]
        counts = report.contact_scope_counts_v3(rows)
        self.assertEqual(counts["denominator"], 1)
        self.assertEqual(
            counts["arms"][report.AEGIS_ARM][
                "case_counts_by_scope"
            ]["unprotected_robot_link"],
            1,
        )
        self.assertEqual(
            counts["arms"][report.AEGIS_ARM][
                "body_event_histograms_by_scope"
            ]["intended_eef_subtree"],
            {"gripper0_rightfinger": 1},
        )
        self.assertFalse(
            counts["arms"][report.AEGIS_ARM][
                "ellipsoid_geometric_enclosure_certified"
            ]
        )


class PairedTemporalEvidenceV3Tests(unittest.TestCase):
    def test_intervention_policy_and_goal_divergence_are_ordered(self) -> None:
        baseline, aegis = paired_results()
        evidence = report.paired_temporal_evidence_v3(
            baseline, aegis
        )
        self.assertEqual(
            evidence["schema_version"],
            report.PAIRED_TEMPORAL_EVIDENCE_SCHEMA,
        )
        self.assertEqual(evidence["first_intervention_step"], 1)
        self.assertEqual(
            evidence["first_nominal_action_exact_divergence_step"], 2
        )
        self.assertEqual(
            evidence[
                "first_policy_query_action_hash_divergence_index"
            ],
            1,
        )
        self.assertEqual(
            evidence["first_native_goal_vector_divergence_step"], 3
        )
        self.assertEqual(
            evidence["first_aegis_goal_fraction_deficit_step"], 3
        )
        self.assertAlmostEqual(
            evidence["first_aegis_goal_fraction_deficit"], -1.0
        )
        self.assertEqual(
            evidence["post_intervention_common_step_count"], 3
        )
        self.assertAlmostEqual(
            evidence["post_intervention_nominal_xyz_l2_sum"], 1.5
        )
        self.assertAlmostEqual(
            evidence["post_intervention_nominal_xyz_l2_max"], 1.0
        )
        self.assertEqual(
            evidence["post_intervention_native_goal_hamming_sum"], 2
        )
        self.assertEqual(
            evidence["post_intervention_native_goal_hamming_max"], 2
        )
        self.assertAlmostEqual(
            evidence["post_intervention_native_goal_fraction_sum"],
            1.0,
        )
        self.assertAlmostEqual(
            evidence["post_intervention_native_goal_fraction_max"],
            1.0,
        )
        self.assertEqual(
            evidence["first_native_goal_divergence_hamming"], 2
        )
        self.assertAlmostEqual(
            evidence["first_native_goal_divergence_fraction"], 1.0
        )
        self.assertAlmostEqual(
            evidence[
                "first_native_goal_aegis_minus_baseline_fraction"
            ],
            -1.0,
        )
        self.assertEqual(
            evidence[
                "aegis_goal_regression_count_at_or_after_first_intervention"
            ],
            1,
        )
        self.assertTrue(
            evidence[
                "intervention_precedes_or_coincides_nominal_divergence"
            ]
        )
        self.assertTrue(
            evidence[
                "intervention_precedes_or_coincides_goal_divergence"
            ]
        )
        self.assertTrue(
            evidence[
                "intervention_precedes_or_coincides_both_divergences"
            ]
        )
        self.assertTrue(
            evidence["registered_intervention_nominal_goal_chain"]
        )
        self.assertEqual(
            evidence["source_immutable_records"][
                "baseline_result_payload_sha256"
            ],
            "b" * 64,
        )
        self.assertFalse(evidence["causal_claim_supported"])

    def test_first_divergence_ahead_then_later_deficit_is_distinct(
        self,
    ) -> None:
        baseline, aegis = paired_results()
        for action in aegis["actions"]:
            action["modified"] = False
        aegis["actions"][3]["modified"] = True
        baseline["actions"][1]["goal_progress"]["values"] = [False, False]
        aegis["actions"][1]["goal_progress"]["values"] = [True, False]
        baseline["actions"][2]["goal_progress"]["values"] = [True, True]
        aegis["actions"][2]["goal_progress"]["values"] = [False, False]

        evidence = report.paired_temporal_evidence_v3(
            baseline, aegis
        )

        self.assertEqual(
            evidence["first_native_goal_vector_divergence_step"], 1
        )
        self.assertAlmostEqual(
            evidence[
                "first_native_goal_aegis_minus_baseline_fraction"
            ],
            0.5,
        )
        self.assertEqual(
            evidence["first_aegis_goal_fraction_deficit_step"], 2
        )
        self.assertAlmostEqual(
            evidence["first_aegis_goal_fraction_deficit"], -1.0
        )
        self.assertEqual(evidence["first_intervention_step"], 3)

    def test_no_intervention_is_explicit_not_vacuously_temporal(self) -> None:
        baseline, _ = paired_results()
        aegis = copy.deepcopy(baseline)
        evidence = report.paired_temporal_evidence_v3(
            baseline, aegis
        )
        self.assertIsNone(evidence["first_intervention_step"])
        self.assertIsNone(
            evidence["first_nominal_action_exact_divergence_step"]
        )
        self.assertIsNone(
            evidence["first_native_goal_vector_divergence_step"]
        )
        self.assertEqual(
            evidence["post_intervention_common_step_count"], 0
        )
        self.assertIsNone(
            evidence[
                "intervention_precedes_or_coincides_nominal_divergence"
            ]
        )
        self.assertFalse(
            evidence["registered_intervention_nominal_goal_chain"]
        )

    def test_reversed_and_same_step_temporal_chains_are_downgraded(
        self,
    ) -> None:
        baseline, aegis = paired_results()
        aegis["actions"][1]["nominal_translational"][0] = 1.5
        same_step = report.paired_temporal_evidence_v3(
            baseline, aegis
        )
        self.assertEqual(
            same_step["first_intervention_step"], 1
        )
        self.assertEqual(
            same_step["first_nominal_action_exact_divergence_step"],
            1,
        )
        self.assertFalse(
            same_step[
                "intervention_strictly_precedes_nominal_divergence"
            ]
        )
        self.assertFalse(
            same_step["registered_intervention_nominal_goal_chain"]
        )

        baseline, aegis = paired_results()
        aegis["actions"][1]["goal_progress"]["values"] = [True, True]
        reversed_chain = report.paired_temporal_evidence_v3(
            baseline, aegis
        )
        self.assertEqual(
            reversed_chain["first_native_goal_vector_divergence_step"],
            1,
        )
        self.assertFalse(
            reversed_chain[
                "nominal_divergence_precedes_or_coincides_goal_divergence"
            ]
        )
        self.assertFalse(
            reversed_chain[
                "registered_intervention_nominal_goal_chain"
            ]
        )

    def test_step_and_initial_goal_pairing_fail_closed(self) -> None:
        baseline, aegis = paired_results()
        aegis["actions"][2]["step"] = 9
        with self.assertRaisesRegex(
            report.FailureReportV3Error, "not contiguous"
        ):
            report.paired_temporal_evidence_v3(baseline, aegis)

        baseline, aegis = paired_results()
        aegis["goal_progress"]["initial"]["values"] = [True, True]
        with self.assertRaisesRegex(
            report.FailureReportV3Error, "different native goals"
        ):
            report.paired_temporal_evidence_v3(baseline, aegis)


class FlowDecisionEvidenceV3Tests(unittest.TestCase):
    def classify(self, mutate=None) -> str:
        row = decision_record()
        if mutate is not None:
            mutate(row)
        return report.decision_evidence_v3(row)[
            "decision_partition_class"
        ]

    def test_strong_candidate_requires_every_registered_gate(self) -> None:
        evidence = report.decision_evidence_v3(decision_record())
        self.assertEqual(
            evidence["schema_version"], report.FLOW_EVIDENCE_SCHEMA
        )
        self.assertTrue(
            evidence[
                "baseline_physical_collision_task_success_to_aegis_"
                "physical_safe_task_failure"
            ]
        )
        self.assertTrue(
            evidence["baseline_sampled_physical_collision_confirmed"]
        )
        self.assertEqual(
            evidence["decision_partition_class"],
            "strong_flow_candidate_association",
        )
        self.assertTrue(evidence["strong_flow_candidate_association"])
        self.assertFalse(evidence["causal_flow_claim_supported"])

    def test_group_bootstrap_uses_complete_task_level_groups(self) -> None:
        strong = decision_record()
        strong["flow_direction_evidence_v3"] = (
            report.decision_evidence_v3(strong)
        )
        preserved = decision_record()
        preserved["task_level_group_id"] = "group-1"
        preserved["outcomes"][report.AEGIS_ARM]["task_success"] = True
        preserved["flow_direction_evidence_v3"] = (
            report.decision_evidence_v3(preserved)
        )
        first = report.group_bootstrap_flow_prevalence_v3(
            [strong, preserved],
            seed=17,
            replicates=500,
        )
        second = report.group_bootstrap_flow_prevalence_v3(
            [strong, preserved],
            seed=17,
            replicates=500,
        )
        self.assertEqual(first, second)
        self.assertEqual(first["unit"], "task_level_group_id")
        self.assertEqual(first["estimator"], "ratio_of_sums")
        self.assertEqual(
            first["sampling"], "whole_groups_with_replacement"
        )
        self.assertEqual(
            first["rng_algorithm"],
            report.FLOW_BOOTSTRAP_RNG_ALGORITHM,
        )
        self.assertEqual(first["source_group_count"], 2)
        self.assertEqual(first["draw_group_count_per_replicate"], 2)
        self.assertEqual(
            first["source_group_ids"], ["group-0", "group-1"]
        )
        self.assertEqual(
            first["source_group_case_counts"],
            {"group-0": 1, "group-1": 1},
        )
        self.assertEqual(first["uniform_source_group_size"], 1)
        self.assertEqual(
            first["interval_conditioning"],
            "nonzero_denominator_replicates",
        )
        self.assertEqual(first["task_level_group_count"], 2)
        self.assertEqual(first["eligible_case_count"], 2)
        self.assertEqual(first["candidate_case_count"], 1)
        self.assertEqual(first["candidate_group_ids"], ["group-0"])
        self.assertEqual(
            first["replicates_with_nonzero_denominator"], 500
        )
        self.assertEqual(first["zero_denominator_replicates"], 0)
        self.assertEqual(first["observed_fraction"], 0.5)
        self.assertEqual(first["interval_fraction"]["lower"], 0.0)
        self.assertEqual(first["interval_fraction"]["upper"], 1.0)

    def test_group_bootstrap_reports_zero_eligible_denominator(self) -> None:
        row = decision_record()
        row["physical_contacts"][report.BASELINE_ARM][
            "postcontrol_collision_relevant_event_count"
        ] = 0
        row["physical_contacts"][report.BASELINE_ARM][
            "collision_relevant_event_count"
        ] = 0
        row["flow_direction_evidence_v3"] = (
            report.decision_evidence_v3(row)
        )
        bootstrap = report.group_bootstrap_flow_prevalence_v3(
            [row],
            seed=23,
            replicates=100,
        )
        self.assertEqual(bootstrap["eligible_case_count"], 0)
        self.assertEqual(bootstrap["candidate_case_count"], 0)
        self.assertEqual(
            bootstrap["zero_denominator_replicates"], 100
        )
        self.assertIsNone(bootstrap["observed_fraction"])
        self.assertIsNone(bootstrap["interval_fraction"])

    def test_registered_population_bootstrap_contract_is_exact(self) -> None:
        rows = []
        for group_index in range(
            report.FLOW_BOOTSTRAP_EXPECTED_SOURCE_GROUP_COUNT
        ):
            for _ in range(
                report.FLOW_BOOTSTRAP_EXPECTED_CASES_PER_GROUP
            ):
                row = decision_record()
                row["task_level_group_id"] = f"group-{group_index:02d}"
                row["flow_direction_evidence_v3"] = (
                    report.decision_evidence_v3(row)
                )
                rows.append(row)

        bootstrap = report.group_bootstrap_flow_prevalence_v3(
            rows,
            seed=29,
            replicates=100,
        )

        self.assertEqual(
            bootstrap["source_group_count"],
            report.FLOW_BOOTSTRAP_EXPECTED_SOURCE_GROUP_COUNT,
        )
        self.assertEqual(
            bootstrap["draw_group_count_per_replicate"],
            report.FLOW_BOOTSTRAP_EXPECTED_SOURCE_GROUP_COUNT,
        )
        self.assertEqual(
            bootstrap["uniform_source_group_size"],
            report.FLOW_BOOTSTRAP_EXPECTED_CASES_PER_GROUP,
        )
        self.assertTrue(
            bootstrap["registered_population_contract"]["satisfied"]
        )

    def test_bootstrap_rejects_candidate_outside_eligible_denominator(
        self,
    ) -> None:
        row = decision_record()
        row["flow_direction_evidence_v3"] = (
            report.decision_evidence_v3(row)
        )
        row["flow_direction_evidence_v3"][
            "baseline_sampled_physical_collision_confirmed"
        ] = False
        with self.assertRaisesRegex(
            report.FailureReportV3Error,
            "outside the eligible denominator",
        ):
            report.group_bootstrap_flow_prevalence_v3(
                [row], seed=31, replicates=10
            )

    def test_task_loss_before_intervention_counterexample_is_signed(
        self,
    ) -> None:
        row = decision_record()
        row["paired_temporal_evidence_v3"].update(
            first_aegis_goal_fraction_deficit_step=0,
            first_aegis_goal_fraction_deficit=-0.5,
            first_intervention_step=1,
            intervention_strictly_precedes_nominal_divergence=False,
            nominal_divergence_precedes_or_coincides_goal_divergence=False,
            registered_intervention_nominal_goal_chain=False,
        )
        row["flow_direction_evidence_v3"] = (
            report.decision_evidence_v3(row)
        )
        counts = report.decision_counts_v3([row])
        self.assertEqual(
            counts["task_loss_precedes_intervention_counterexample"][
                "count"
            ],
            1,
        )

        row["paired_temporal_evidence_v3"][
            "first_aegis_goal_fraction_deficit"
        ] = 0.5
        counts = report.decision_counts_v3([row])
        self.assertEqual(
            counts["task_loss_precedes_intervention_counterexample"][
                "count"
            ],
            0,
        )

    def test_temporal_counterexample_uses_physical_eligible_denominator(
        self,
    ) -> None:
        row = decision_record()
        row["physical_contacts"][report.BASELINE_ARM].update(
            collision_relevant_event_count=0,
            postcontrol_collision_relevant_event_count=0,
        )
        row["paired_temporal_evidence_v3"].update(
            first_aegis_goal_fraction_deficit_step=0,
            first_aegis_goal_fraction_deficit=-0.5,
            first_intervention_step=1,
        )
        row["flow_direction_evidence_v3"] = (
            report.decision_evidence_v3(row)
        )

        counts = report.decision_counts_v3([row])

        counterexamples = counts[
            "task_loss_precedes_intervention_counterexample"
        ]
        self.assertEqual(counterexamples["count"], 0)
        self.assertEqual(
            counterexamples[
                "eligible_baseline_collision_task_loss_denominator"
            ],
            0,
        )

    def test_exact_exclusion_precedence(self) -> None:
        cases = {
            "not_baseline_task_success_to_aegis_task_failure": (
                lambda row: row["outcomes"][report.BASELINE_ARM].update(
                    task_success=False
                )
            ),
            "excluded_preexisting_settled_collision_relevant_contact": (
                lambda row: (
                    row["physical_contacts"][report.BASELINE_ARM].update(
                        settled_collision_relevant_event_count=1,
                        collision_relevant_event_count=2,
                    ),
                    row["physical_contacts"][report.AEGIS_ARM].update(
                        settled_collision_relevant_event_count=1,
                        collision_relevant_event_count=1,
                    ),
                )
            ),
            "excluded_baseline_sampled_physical_collision_unconfirmed": (
                lambda row: row["physical_contacts"][
                    report.BASELINE_ARM
                ].update(
                    collision_relevant_event_count=0,
                    postcontrol_collision_relevant_event_count=0,
                )
            ),
            "excluded_paper_car_sampled_contact_mismatch": (
                lambda row: row["outcomes"][
                    report.BASELINE_ARM
                ].update(paper_collision=False)
            ),
            "excluded_no_valid_execution": lambda row: (
                row["outcomes"][report.AEGIS_ARM].update(
                    executed_action_count=0
                ),
                row["aegis_diagnostics"]["control"].update(
                    control_path_counts={}
                ),
            ),
            "excluded_pipeline_not_complete_or_not_all_qp": (
                lambda row: row["aegis_diagnostics"][
                    "geometry_v3"
                ].update(status="failure")
            ),
            "excluded_no_intervention": (
                lambda row: row["aegis_diagnostics"][
                    "intervention"
                ].update(intervention_count=0)
            ),
            "excluded_unprotected_robot_link_contact": (
                lambda row: row["physical_contacts"][
                    report.AEGIS_ARM
                ]["robot_contact_scope_v3"].update(
                    sampled_unprotected_robot_link_contact=True
                )
            ),
            "excluded_sampled_physical_safety_unconfirmed": (
                lambda row: (
                    row["physical_contacts"][report.AEGIS_ARM].update(
                        collision_relevant_event_count=1,
                        postcontrol_collision_relevant_event_count=1,
                    ),
                    row["outcomes"][report.AEGIS_ARM].update(
                        paper_collision=True
                    ),
                )
            ),
            "excluded_no_negative_final_goal_delta": (
                lambda row: row["paired_goal_progress"][
                    "aegis_minus_pi05"
                ].update(final_fraction=0.0)
            ),
            "excluded_temporal_precedence_unavailable": (
                lambda row: row["paired_temporal_evidence_v3"].update(
                    nominal_divergence_precedes_or_coincides_goal_divergence=(
                        False
                    ),
                    registered_intervention_nominal_goal_chain=False,
                )
            ),
        }
        for expected, mutate in cases.items():
            with self.subTest(expected=expected):
                self.assertEqual(self.classify(mutate), expected)

        self.assertEqual(
            self.classify(
                lambda row: row[
                    "paired_temporal_evidence_v3"
                ].update(
                    intervention_strictly_precedes_nominal_divergence=(
                        False
                    ),
                    registered_intervention_nominal_goal_chain=False,
                )
            ),
            "excluded_temporal_precedence_unavailable",
        )

        self.assertEqual(
            self.classify(
                lambda row: (
                    row["physical_contacts"][report.AEGIS_ARM].update(
                        collision_relevant_event_count=1,
                        postcontrol_collision_relevant_event_count=1,
                    ),
                    row["outcomes"][report.AEGIS_ARM].update(
                        paper_collision=True
                    ),
                )
            ),
            "excluded_sampled_physical_safety_unconfirmed",
        )

    def test_aggregate_is_an_exact_partition(self) -> None:
        strong = decision_record()
        strong["flow_direction_evidence_v3"] = (
            report.decision_evidence_v3(strong)
        )
        excluded = decision_record()
        excluded["aegis_diagnostics"]["intervention"][
            "intervention_count"
        ] = 0
        excluded["flow_direction_evidence_v3"] = (
            report.decision_evidence_v3(excluded)
        )
        counts = report.decision_counts_v3([strong, excluded])
        partition = counts["decision_partition_class_counts"]
        self.assertEqual(sum(partition.values()), 2)
        self.assertEqual(
            partition["strong_flow_candidate_association"], 1
        )
        self.assertEqual(partition["excluded_no_intervention"], 1)
        self.assertTrue(counts["exact_partition"])
        self.assertEqual(
            counts["schema_version"],
            report.FLOW_DECISION_COUNTS_SCHEMA,
        )
        self.assertEqual(
            counts["strong_flow_candidate_association"][
                "eligible_denominator"
            ],
            2,
        )
        self.assertEqual(
            counts["strong_flow_candidate_association"][
                "distinct_task_level_group_count"
            ],
            1,
        )
        self.assertFalse(counts["causal_flow_claim_supported"])

    def test_paper_safe_but_sampled_contact_is_not_strong(self) -> None:
        row = decision_record()
        self.assertFalse(
            row["outcomes"][report.AEGIS_ARM]["paper_collision"]
        )
        row["physical_contacts"][report.AEGIS_ARM][
            "collision_relevant_event_count"
        ] = 1
        row["physical_contacts"][report.AEGIS_ARM][
            "postcontrol_collision_relevant_event_count"
        ] = 1
        evidence = report.decision_evidence_v3(row)
        self.assertEqual(
            evidence["decision_partition_class"],
            "excluded_paper_car_sampled_contact_mismatch",
        )
        self.assertFalse(evidence["strong_flow_candidate_association"])

    def test_inconsistent_contact_counts_fail_closed(self) -> None:
        for arm in (report.BASELINE_ARM, report.AEGIS_ARM):
            with self.subTest(arm=arm):
                row = decision_record()
                row["physical_contacts"][arm][
                    "postcontrol_collision_relevant_event_count"
                ] = 2
                with self.assertRaisesRegex(
                    report.FailureReportV3Error,
                    "contact counts differ",
                ):
                    report.decision_evidence_v3(row)

    def test_paper_safe_baseline_contact_is_eligible_but_not_candidate(
        self,
    ) -> None:
        row = decision_record()
        row["outcomes"][report.BASELINE_ARM]["paper_collision"] = False
        evidence = report.decision_evidence_v3(row)
        self.assertEqual(
            evidence["decision_partition_class"],
            "excluded_paper_car_sampled_contact_mismatch",
        )
        self.assertFalse(evidence["strong_flow_candidate_association"])
        self.assertFalse(evidence["baseline_paper_collision"])
        self.assertTrue(
            evidence["baseline_sampled_physical_collision_confirmed"]
        )

    def test_paper_collision_without_sampled_baseline_contact_is_not_strong(
        self,
    ) -> None:
        row = decision_record()
        row["physical_contacts"][report.BASELINE_ARM].update(
            collision_relevant_event_count=0,
            postcontrol_collision_relevant_event_count=0,
        )
        evidence = report.decision_evidence_v3(row)
        self.assertTrue(evidence["baseline_paper_collision"])
        self.assertEqual(
            evidence["decision_partition_class"],
            "excluded_baseline_sampled_physical_collision_unconfirmed",
        )
        self.assertFalse(evidence["strong_flow_candidate_association"])

    def test_settled_plus_postcontrol_contact_is_preexisting_exclusion(
        self,
    ) -> None:
        row = decision_record()
        row["physical_contacts"][report.BASELINE_ARM].update(
            settled_collision_relevant_event_count=1,
            collision_relevant_event_count=2,
        )
        row["physical_contacts"][report.AEGIS_ARM].update(
            settled_collision_relevant_event_count=1,
            collision_relevant_event_count=1,
        )
        evidence = report.decision_evidence_v3(row)
        self.assertEqual(
            evidence["decision_partition_class"],
            "excluded_preexisting_settled_collision_relevant_contact",
        )
        self.assertFalse(
            evidence[
                "paired_initial_sampled_collision_relevant_contact_free"
            ]
        )
        self.assertFalse(
            evidence["baseline_sampled_physical_collision_confirmed"]
        )

    def test_markdown_surfaces_physical_denominator_and_bootstrap(
        self,
    ) -> None:
        partition = {
            key: 0 for key in report.DECISION_PARTITION
        }
        partition["strong_flow_candidate_association"] = 2
        markdown = report.render_markdown_v3(
            {
                "counts": {
                    "flow_direction_decision_v3": {
                        "decision_partition_class_counts": partition,
                        "eligible_baseline_sampled_collision_task_success": {
                            "count": 7,
                        },
                        "baseline_physical_collision_task_success_to_aegis_"
                        "physical_safe_task_failure": {"count": 3},
                        "baseline_paper_collision_task_success_to_aegis_"
                        "paper_safe_task_failure": {"count": 2},
                        "opposite_physical_safety_and_task_preserved": {
                            "count": 4,
                            "distinct_task_level_group_count": 3,
                        },
                        "task_loss_precedes_intervention_counterexample": {
                            "count": 1,
                            "distinct_task_level_group_count": 1,
                        },
                        "strong_flow_candidate_association": {
                            "count": 2,
                            "eligible_denominator": 7,
                            "distinct_task_level_group_count": 2,
                            "task_level_group_ids": [
                                "group-0",
                                "group-1",
                            ],
                            "paper_car_concordant_count": 2,
                            "paper_car_discordant_count": 0,
                            "group_bootstrap_prevalence": {
                                "interval_percent": {
                                    "lower": 10.0,
                                    "upper": 50.0,
                                },
                                "replicates_with_nonzero_denominator": 10000,
                                "zero_denominator_replicates": 0,
                            },
                        },
                    },
                    "robot_contact_scope_v3": {
                        "arms": {
                            arm: {
                                "event_counts_by_scope": {
                                    scope: 0
                                    for scope in (
                                        report.ROBOT_CONTACT_SCOPES
                                    )
                                }
                            }
                            for arm in (
                                report.BASELINE_ARM,
                                report.AEGIS_ARM,
                            )
                        }
                    },
                }
            }
        )
        self.assertIn("**2 / 7** eligible cases", markdown)
        self.assertIn("**[10.00%, 50.00%]**", markdown)
        self.assertIn("`group-0, group-1`", markdown)
        self.assertIn("zero-denominator replicates excluded", markdown)
        self.assertIn(
            "AEGIS first has lower native-goal completion", markdown
        )
        self.assertIn("`strong_flow_candidate_association` | 2", markdown)


class AcceptedV2SummaryBindingV3Tests(unittest.TestCase):
    def summary(self) -> dict:
        return {
            "schema_version": report.v2.SUMMARY_SCHEMA_V2,
            "status": "complete_postpublication_analysis_v2",
            "protocol_id": "accepted-v2-protocol",
            "claim_scope": {},
            "accepted_result_payloads_sha256": "b" * 64,
            "source_v1": {
                "v1_publication_receipt_sha256": "a" * 64,
            },
        }

    def compact_record(
        self,
        *,
        temporal_schema: str | None = None,
        flow_schema: str | None = None,
    ) -> dict:
        row = {
            "case_id": "case-0",
            "schema_version": report.CASE_SCHEMA,
            "paired_temporal_evidence_v3": {
                "schema_version": (
                    report.PAIRED_TEMPORAL_EVIDENCE_SCHEMA
                    if temporal_schema is None
                    else temporal_schema
                )
            },
            "flow_direction_evidence_v3": {
                "schema_version": (
                    report.FLOW_EVIDENCE_SCHEMA
                    if flow_schema is None
                    else flow_schema
                )
            },
        }
        row["record_payload_sha256"] = report.v2.sha256_bytes(
            report.v2.canonical_json_bytes(row)
        )
        return row

    def validation_mocks(self, decision_schema=None, bootstrap_schema=None):
        decision = {
            "schema_version": (
                report.FLOW_DECISION_COUNTS_SCHEMA
                if decision_schema is None
                else decision_schema
            ),
            "strong_flow_candidate_association": {
                "group_bootstrap_prevalence": {
                    "schema_version": (
                        report.FLOW_BOOTSTRAP_SCHEMA
                        if bootstrap_schema is None
                        else bootstrap_schema
                    )
                }
            },
        }
        return (
            mock.patch.object(
                report.v2,
                "validate_report_against_summary_v2",
                return_value={},
            ),
            mock.patch.object(
                report,
                "contact_scope_counts_v3",
                return_value={},
            ),
            mock.patch.object(
                report,
                "pipeline_counts_v3",
                return_value={},
            ),
            mock.patch.object(
                report,
                "decision_counts_v3",
                return_value=decision,
            ),
        )

    def test_validation_consumes_exact_accepted_v2_summary(self) -> None:
        summary = self.summary()
        before = copy.deepcopy(summary)
        with mock.patch.object(
            report.v2,
            "validate_report_against_summary_v2",
            return_value={"accepted_v2": True},
        ) as validator:
            counts = report.validate_report_against_summary_v3(
                [], summary=summary, expected_cases=0
            )
        validator.assert_called_once_with(
            [], summary=summary, expected_cases=0
        )
        self.assertEqual(summary, before)
        self.assertTrue(counts["accepted_v2"])
        self.assertEqual(
            report.SOURCE_SUMMARY_SCHEMA,
            report.v2.SUMMARY_SCHEMA_V2,
        )

    def test_unproduced_v3_summary_contract_is_rejected(self) -> None:
        summary = self.summary()
        summary["schema_version"] = "vlsa_table1_population_summary.v3"
        summary["status"] = "complete_postpublication_analysis_v3"
        with self.assertRaisesRegex(
            report.FailureReportV3Error,
            "accepted analysis-v2 source summary is not complete",
        ):
            report.validate_report_against_summary_v3(
                [], summary=summary, expected_cases=0
            )

    def test_nested_v3_schemas_are_enforced_when_outer_v3_is_reused(
        self,
    ) -> None:
        for field, row in (
            (
                "temporal",
                self.compact_record(temporal_schema="retired.temporal.v1"),
            ),
            (
                "flow",
                self.compact_record(flow_schema="retired.flow.v1"),
            ),
        ):
            with self.subTest(field=field):
                patches = self.validation_mocks()
                with patches[0], patches[1], patches[2], patches[3]:
                    with self.assertRaisesRegex(
                        report.FailureReportV3Error,
                        "nested analysis-v3 schema differs",
                    ):
                        report.validate_report_against_summary_v3(
                            [row],
                            summary=self.summary(),
                            expected_cases=1,
                        )

    def test_aggregate_nested_v3_schemas_are_enforced(self) -> None:
        fixtures = (
            ("decision-count", "retired.counts.v0", None),
            ("bootstrap", None, "retired.bootstrap.v0"),
        )
        for field, decision_schema, bootstrap_schema in fixtures:
            with self.subTest(field=field):
                patches = self.validation_mocks(
                    decision_schema=decision_schema,
                    bootstrap_schema=bootstrap_schema,
                )
                with patches[0], patches[1], patches[2], patches[3]:
                    with self.assertRaisesRegex(
                        report.FailureReportV3Error,
                        "schema differs",
                    ):
                        report.validate_report_against_summary_v3(
                            [self.compact_record()],
                            summary=self.summary(),
                            expected_cases=1,
                        )

    def test_report_requires_exact_source_receipt_equality(self) -> None:
        summary = self.summary()
        records = [
            {
                "case_id": "case-0",
                "case_ordinal": 0,
                "record_payload_sha256": "c" * 64,
            }
        ]
        with mock.patch.object(
            report,
            "validate_report_against_summary_v3",
            return_value={"bound": True},
        ):
            built = report.build_report_v3(
                records,
                summary=summary,
                summary_sha256="d" * 64,
                validation_receipt_sha256="e" * 64,
                source_publication_receipt_sha256="a" * 64,
                expected_cases=1,
            )
            self.assertEqual(
                built["source"]["v1_publication_receipt_sha256"],
                summary["source_v1"][
                    "v1_publication_receipt_sha256"
                ],
            )
            self.assertEqual(
                built["source"]["population_summary_source_schema"],
                report.v2.SUMMARY_SCHEMA_V2,
            )
            self.assertTrue(
                built["source"][
                    "v3_counts_derived_read_only_from_v2_summary"
                ]
            )
            with self.assertRaisesRegex(
                report.FailureReportV3Error,
                "source differs from its accepted v2 summary",
            ):
                report.build_report_v3(
                    records,
                    summary=summary,
                    summary_sha256="d" * 64,
                    validation_receipt_sha256="e" * 64,
                    source_publication_receipt_sha256="f" * 64,
                    expected_cases=1,
                )

    def test_report_rejects_missing_source_binding(self) -> None:
        records = [
            {
                "case_id": "case-0",
                "case_ordinal": 0,
                "record_payload_sha256": "c" * 64,
            }
        ]
        missing_object = self.summary()
        missing_object.pop("source_v1")
        missing_hash = self.summary()
        missing_hash["source_v1"] = {}
        with mock.patch.object(
            report,
            "validate_report_against_summary_v3",
            return_value={"bound": True},
        ):
            with self.assertRaisesRegex(
                report.FailureReportV3Error,
                "source-v1 binding must be an object",
            ):
                report.build_report_v3(
                    records,
                    summary=missing_object,
                    summary_sha256="d" * 64,
                    validation_receipt_sha256="e" * 64,
                    source_publication_receipt_sha256="a" * 64,
                    expected_cases=1,
                )
            with self.assertRaisesRegex(
                report.FailureReportV3Error,
                "summary source v1 publication receipt SHA",
            ):
                report.build_report_v3(
                    records,
                    summary=missing_hash,
                    summary_sha256="d" * 64,
                    validation_receipt_sha256="e" * 64,
                    source_publication_receipt_sha256="a" * 64,
                    expected_cases=1,
                )


@unittest.skipUnless(
    os.environ.get("VLSA_V3_REAL_RESULTS_ROOT"),
    "set VLSA_V3_REAL_RESULTS_ROOT for terminal artifact smoke",
)
class RealArtifactCompatibilityV3Tests(unittest.TestCase):
    """Optional local-only smoke over compact terminal runtime artifacts."""

    def test_terminal_task01_runtime_paths_and_full_overlay(self) -> None:
        tasks_root = Path(
            os.environ["VLSA_V3_REAL_RESULTS_ROOT"]
        ).resolve()
        expected_pairs = int(
            os.environ.get("VLSA_V3_REAL_EXPECTED_PAIRS", "100")
        )
        pairs: dict[str, tuple[Path, Path, Path]] = {}
        for task_index in (0, 1):
            result_root = (
                tasks_root / f"task-{task_index}" / "results"
            )
            for baseline_path in sorted(
                (result_root / "pi05").glob("*/result.json")
            ):
                case_id = baseline_path.parent.name
                aegis_path = (
                    result_root / "aegis" / case_id / "result.json"
                )
                self.assertTrue(aegis_path.is_file(), case_id)
                self.assertNotIn(case_id, pairs)
                pairs[case_id] = (
                    baseline_path,
                    aegis_path,
                    result_root,
                )
        self.assertEqual(len(pairs), expected_pairs)

        for case_id, (baseline_path, aegis_path, _) in pairs.items():
            baseline = json.loads(
                baseline_path.read_text(encoding="utf-8")
            )
            aegis = json.loads(
                aegis_path.read_text(encoding="utf-8")
            )
            self.assertEqual(baseline["case_id"], case_id)
            self.assertEqual(aegis["case_id"], case_id)
            temporal = report.paired_temporal_evidence_v3(
                baseline, aegis
            )
            geometry = report.geometry_evidence_v3(aegis)
            self.assertEqual(
                temporal["schema_version"],
                report.PAIRED_TEMPORAL_EVIDENCE_SCHEMA,
            )
            self.assertIn(
                geometry["status"],
                {"complete", "failure", "method_failure_passthrough"},
            )

        overlay_value = os.environ.get(
            "VLSA_V3_REAL_ARTIFACT_OVERLAY"
        )
        if overlay_value is None:
            self.skipTest(
                "set VLSA_V3_REAL_ARTIFACT_OVERLAY for full records"
            )
        overlay = Path(overlay_value).resolve()
        full_case_ids = sorted(
            path.name
            for path in (overlay / "pi05").iterdir()
            if path.is_dir()
            and (
                overlay
                / "aegis"
                / path.name
                / "active_obstacle_contacts.json.gz"
            ).is_file()
        )
        expected_full = int(
            os.environ.get(
                "VLSA_V3_REAL_EXPECTED_FULL_PAIRS", "2"
            )
        )
        self.assertEqual(len(full_case_ids), expected_full)
        full_records = {}
        for case_id in full_case_ids:
            baseline_path, aegis_path, _ = pairs[case_id]
            baseline = json.loads(
                baseline_path.read_text(encoding="utf-8")
            )
            aegis = json.loads(
                aegis_path.read_text(encoding="utf-8")
            )
            full_records[case_id] = report.build_case_record_v3(
                manifest=baseline["case"],
                baseline=baseline,
                aegis=aegis,
                baseline_artifact_root=overlay,
                aegis_artifact_root=overlay,
            )
            self.assertFalse(
                full_records[case_id]["flow_direction_evidence_v3"][
                    "causal_flow_claim_supported"
                ]
            )

        strong_case = os.environ.get(
            "VLSA_V3_REAL_EXPECT_STRONG_CASE"
        )
        if strong_case:
            self.assertIn(strong_case, full_records)
            evidence = full_records[strong_case][
                "flow_direction_evidence_v3"
            ]
            temporal = full_records[strong_case][
                "paired_temporal_evidence_v3"
            ]
            self.assertEqual(
                evidence["decision_partition_class"],
                "strong_flow_candidate_association",
            )
            self.assertTrue(
                temporal[
                    "registered_intervention_nominal_goal_chain"
                ]
            )


if __name__ == "__main__":
    unittest.main()
