from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import unittest
from unittest import mock

from main.poisson_fullbody.contracts import ArtifactContractError
from main.poisson_fullbody.full_episode_feasibility import (
    classify_full_episode_feasibility,
)
from scripts import validate_poisson_full_episode_feasibility_artifact as validator


ROOT = Path(__file__).resolve().parents[1]
VALIDATOR_PATH = (
    ROOT / "scripts/validate_poisson_full_episode_feasibility_artifact.py"
)


def canonical(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def protocol():
    return json.loads(
        (
            ROOT / "configs/vlsa_poisson_full_episode_feasibility.v1.json"
        ).read_text(encoding="utf-8")
    )


def goal_definition():
    value = {
        "schema_version": "safelibero_goal_progress.v1",
        "source": "native_bddl_goal_predicates",
        "logic": "conjunction",
        "goal_atoms": [
            {
                "index": 0,
                "predicate": "in",
                "arguments": ["cream_cheese_1", "basket_1"],
            }
        ],
    }
    value["goal_definition_sha256"] = hashlib.sha256(canonical(value)).hexdigest()
    return value


def goal_snapshot(step, satisfied, state_hash):
    return {
        "step": step,
        "values": [satisfied],
        "satisfied_count": int(satisfied),
        "fraction": float(satisfied),
        "all_satisfied": satisfied,
        "newly_satisfied_indices": [],
        "regressed_indices": [],
        "argument_poses": [
            {
                "atom_index": 0,
                "arguments": [
                    {
                        "name": "cream_cheese_1",
                        "object_state_type": "object",
                        "position": [0.0, 0.0, 0.1],
                        "quaternion": [1.0, 0.0, 0.0, 0.0],
                    },
                    {
                        "name": "basket_1",
                        "object_state_type": "object",
                        "position": [0.1, 0.0, 0.0],
                        "quaternion": [1.0, 0.0, 0.0, 0.0],
                    },
                ],
            }
        ],
        "simulator_state_sha256_before": state_hash,
        "simulator_state_sha256_after": state_hash,
        "inert": True,
    }


def task_record():
    boundary_hash = "1" * 64
    boundary = goal_snapshot(179, False, boundary_hash)
    boundary.update(
        {
            "transition_metadata_available": False,
            "transition_metadata_semantics": "unavailable",
        }
    )
    ledger_boundary = dict(
        boundary,
        snapshot_kind="branch_boundary_pre_action",
        local_action_index=None,
        source_action_index=179,
        reward=None,
        returned_done=None,
        returned_info=None,
        returned_observation_sha256=None,
    )
    completed = []
    observation_hashes = []
    for local_index, source_index in enumerate(range(180, 237)):
        state_hash = "%064x" % (source_index + 1)
        observation_hash = "%064x" % (source_index + 1000)
        observation_hashes.append(observation_hash)
        row = goal_snapshot(source_index, source_index == 236, state_hash)
        row.update(
            {
                "snapshot_kind": "completed_high_level_post_step",
                "local_action_index": local_index,
                "source_action_index": source_index,
                "reward": 1.0 if source_index == 236 else 0.0,
                "returned_done": source_index == 236,
                "returned_info": {"success": source_index == 236},
                "returned_observation_sha256": observation_hash,
            }
        )
        if source_index == 236:
            row["newly_satisfied_indices"] = [0]
        completed.append(row)
    return {
        "source": "native_bddl_goal_predicates",
        "goal_definition": goal_definition(),
        "expected_boundary_goal_values": [False],
        "boundary_goal_values_exact": True,
        "boundary_goal_snapshot": boundary,
        "goal_progress_ledger": [ledger_boundary] + completed,
        "registered_source_action_count": 57,
        "completed_source_action_count": 57,
        "fixed_exposure_complete": True,
        "continue_fixed_exposure_after_success": True,
        "initial_task_success_at_branch": False,
        "ever_task_success_at_or_after_branch": True,
        "first_task_success_source_action_index": 236,
        "terminal_task_success": True,
        "terminal_goal_fraction": 1.0,
        "reward_sum": 1.0,
        "terminal_reward": 1.0,
        "returned_observation_sha256_ledger": observation_hashes,
        "terminal_simulator_state_sha256": completed[-1][
            "simulator_state_sha256_after"
        ],
        "terminal_official_integration_state_raw_bytes_sha256": "e" * 64,
        "terminal_observation_sha256": "f" * 64,
    }


def positive_metrics():
    return {
        "schema_version": validator.METRICS_SCHEMA,
        "exact_paired_start": True,
        "shared_prefix_complete": True,
        "full_recorded_episode_complete": True,
        "adapter_exposure_complete": True,
        "psf_exposure_complete": True,
        "adapter_physics_monitor_trace_counts_match": True,
        "psf_physics_monitor_trace_counts_match": True,
        "adapter_filter_update_count": 285,
        "psf_filter_update_count": 285,
        "adapter_physics_substep_count": 1425,
        "psf_physics_substep_count": 1425,
        "adapter_completed_suffix_action_count": 57,
        "psf_completed_suffix_action_count": 57,
        "psf_qp_count_complete": True,
        "psf_qp_postchecks_complete": True,
        "psf_joint_limit_postchecks_complete": True,
        "all_issued_commands_within_physical_bounds": True,
        "both_nominal_commands_within_dynamic_joint_bounds": True,
        "psf_invalid_field_query_count": 0,
        "psf_all_post_state_field_queries_valid_and_positive": True,
        "static_selected_obstacle_admissible": True,
        "boundary_goal_unsatisfied": True,
        "adapter_link56_contact_present": True,
        "adapter_first_selected_obstacle_contact_is_link56": True,
        "psf_link56_contact_present": False,
        "psf_any_robot_selected_obstacle_contact_present": False,
        "psf_clearance_certified": True,
        "material_correction_before_adapter_contact": True,
        "first_material_correction_physical_boundary": 4500,
        "adapter_first_link56_contact_physical_boundary": 4677,
        "material_correction_update_count": 32,
        "maximum_correction_norm_rad_s": 0.5,
        "filter_correction_integral_rad": 0.01,
        "post_correction_measured_joint_motion_integral_rad": 0.2,
        "post_correction_cartesian_path_length_m": 0.06,
        "post_correction_executed_command_integral_rad": 0.2,
        "post_correction_zero_command_fraction": 0.0,
        "psf_task_success_ever": True,
        "psf_terminal_task_success": True,
        "psf_first_task_success_source_action_index": 236,
        "psf_task_success_after_material_correction": True,
        "adapter_task_success_ever": True,
        "adapter_terminal_task_success": True,
    }


def synthetic_positive_payload():
    metrics = positive_metrics()
    restore = {
        name: "same"
        for name in (
            "model_topology_sha256",
            "physical_model_sha256",
            "compiled_mjb_sha256",
            "official_integration_state_sha256",
            "target_official_integration_state_sha256",
            "settled_state_sha256",
            "target_state_sha256",
            "controller",
            "controller_software_state",
            "pid_memory_reset",
        )
    }
    common = {
        "restore": restore,
        "start_official_raw_bytes_sha256": validator.EXPECTED_BOUNDARY_RAW_SHA256,
        "fresh_adapter": True,
        "exposure_complete": True,
        "physics_monitor_trace_counts_match": True,
        "all_issued_commands_within_physical_bounds": True,
        "all_nominal_commands_within_dynamic_joint_bounds": True,
        "static_precontact_admissible": True,
        "task": {
            "boundary_goal_values_exact": True,
            "initial_task_success_at_branch": False,
        },
    }
    adapter = dict(
        common,
        arm_name="joint_velocity_adapter_only",
    )
    psf = dict(
        common,
        arm_name="joint_velocity_adapter_plus_link56_psf",
        static_full_window_admissible=True,
        protected_sample_count=1531,
        invalid_field_query_count=0,
        post_state_field_observation_count=1,
        post_state_field_query_count=1531,
        nonpositive_post_state_field_query_count=0,
        all_post_state_field_queries_valid_and_positive=True,
        qp_solve_count=285,
        qp_postcheck_count=285,
        joint_limit_postcheck_count=285,
    )
    post_motion = {
        "first_correction_physical_boundary": 4500,
        "command_update_count": 285,
        "physics_substep_count": 1425,
        "complete_command_interval_count": 285,
        "maximum_correction_norm_rad_s": 0.5,
        "filter_correction_integral_rad": 0.01,
        "executed_command_integral_rad": 0.2,
        "measured_joint_motion_integral_rad": 0.2,
        "cartesian_path_length_m": 0.06,
        "zero_command_fraction": 0.0,
    }
    historical = {
        "authority": "frozen_manifest_selection_evidence_and_historical_replay",
        "diagnostic_only_not_a_new_acceptance_gate": True,
        "aegis_collision": True,
        "aegis_task_success": True,
        "literal_robot_contact_bodies": ["robot0_link5"],
        "first_relevant_contact_step": 187,
        "car_collision_step": 188,
        "released_aegis_barrier_h_at_first_relevant_contact": 0.009312042732980995,
        "released_aegis_barrier_h_at_car_crossing": 0.005099169140377209,
        "released_aegis_barrier_h_values_positive": True,
        "terminal_recorded_action": {
            "source_action_index": 236,
            "done": True,
            "reward": 1.0,
            "native_goal_values": [True],
        },
    }
    result = {
        "schema_version": validator.RESULT_SCHEMA,
        "status": "complete",
        "partial_output_interpreted": False,
        "protocol_id": validator.PROTOCOL_ID,
        "run_id": "synthetic-positive",
        "case_id": validator.EXPECTED_CASE,
        "provenance": {
            "source": {
                "commit": "c" * 40,
                "branch": validator.EXPECTED_BRANCH,
            },
            "allocation": {
                "gpu_name": "NVIDIA H100 80GB HBM3",
                "host": "synthetic-h100",
                "slurm_job_id": "12345",
            },
            "run_id": "synthetic-positive",
            "case_id": validator.EXPECTED_CASE,
            "case_row_sha256": validator.EXPECTED_CASE_ROW_SHA256,
            "protocol_file_sha256": "a" * 64,
            "historical_result_file_sha256": validator.EXPECTED_HISTORICAL_FILE_SHA256,
            "historical_result_payload_sha256": validator.EXPECTED_HISTORICAL_PAYLOAD_SHA256,
            "historical_executed_action_sequence_sha256": validator.EXPECTED_ACTION_SEQUENCE_SHA256,
            "suffix_action_record_sha256": validator.EXPECTED_SUFFIX_RECORD_SHA256,
            "suffix_action_array_sha256": validator.EXPECTED_SUFFIX_ACTION_ARRAY_SHA256,
            "online_policy_query_count": 0,
        },
        "episode": {
            "prefix_start_action_index": 0,
            "prefix_end_action_index_inclusive": 179,
            "prefix_action_count": 180,
            "suffix_start_action_index": 180,
            "suffix_end_action_index_inclusive": 236,
            "suffix_action_count": 57,
            "start_boundary": 4500,
            "end_boundary": 5925,
            "source_action_indexes": list(range(180, 237)),
            "post_action_179_flattened_state_sha256": validator.EXPECTED_BOUNDARY_STATE_SHA256,
            "official_boundary_raw_bytes_sha256": validator.EXPECTED_BOUNDARY_RAW_SHA256,
            "field_construction_boundary": 0,
            "shared_osc_prefix": {
                "execution": "exact_historical_actions_0_through_179_under_unchanged_OSC",
                "completed_action_count": 180,
                "terminal_goal_values": [False],
                "terminal_state_sha256": validator.EXPECTED_BOUNDARY_STATE_SHA256,
                "settled_to_boundary_static_field_assumption": {"admissible": True},
            },
            "paired_joint_velocity_suffix": {
                "action_record_sha256": validator.EXPECTED_SUFFIX_RECORD_SHA256,
                "action_array_sha256": validator.EXPECTED_SUFFIX_ACTION_ARRAY_SHA256,
                "controller_updates_per_arm": 285,
                "physics_substeps_per_arm": 1425,
            },
            "complete_recorded_episode_action_count": 237,
            "historical_aegis_reference": historical,
        },
        "field": {
            "resolved_geometry": {
                "link56_geom_ids": [5],
                "robot_geom_ids": [5, 6],
                "obstacle_geom_ids": [9],
            }
        },
        "arms": {"adapter_only": adapter, "adapter_plus_psf": psf},
        "metrics": metrics,
        "activation_evidence": {
            "threshold_m2_per_s": -5e-7,
            "adapter_contact_physical_boundary": 4677,
            "first_material_correction": {"synthetic": True},
            "material_correction_count": 32,
            "post_correction_motion": post_motion,
        },
        "classification": classify_full_episode_feasibility(metrics, protocol()),
        "claim_scope": validator.EXPECTED_CLAIM_SCOPE,
    }
    return result, post_motion


class FullEpisodeArtifactValidatorTest(unittest.TestCase):
    def test_d_sim_uses_signed_contact_distance_when_more_conservative(self):
        self.assertEqual(validator._expected_d_sim(0.012, None), 0.012)
        self.assertEqual(validator._expected_d_sim(0.012, -0.004), -0.004)
        self.assertEqual(validator._expected_d_sim(-0.002, -0.004), -0.004)

    def test_material_activation_preserves_matched_activation_trace_namespace(self):
        audit = validator._Audit()
        command_raw = {"record_kind": "command_trace"}
        activation_raw = {
            "source_action_index": 180,
            "inner_control_index": 0,
            "physical_boundary": 4500,
            "filter_correction_l2_rad_s": 0.25,
            "nominal_within_dynamic_joint_bounds": True,
            "argmin_protected_sample": {"body_name": "robot0_link5"},
        }
        material = validator.narrow._material_activation(
            audit,
            [
                {
                    "source": 180,
                    "inner": 0,
                    "boundary": 4500,
                    "nominal_residual": -0.01,
                    "correction": 0.25,
                    "nominal_bounds": True,
                    "raw": command_raw,
                }
            ],
            {"activation_trace": [activation_raw]},
            4677,
        )
        self.assertEqual(audit.discrepancies, [])
        self.assertEqual(material[0]["raw"], command_raw)
        self.assertEqual(material[0]["activation_raw"], activation_raw)
        self.assertNotEqual(material[0]["raw"], material[0]["activation_raw"])

    def test_protocol_derives_complete_suffix_population(self):
        observed = validator._protocol_expectations(protocol())
        self.assertEqual(observed["action_count"], 57)
        self.assertEqual(observed["filter_updates"], 285)
        self.assertEqual(observed["physics_substeps"], 1425)
        self.assertEqual(observed["start_boundary"], 4500)
        self.assertEqual(observed["end_boundary"], 5925)

    def test_protocol_rejects_shortened_suffix(self):
        value = protocol()
        value["episode"]["suffix_end_action_index_inclusive"] = 187
        with self.assertRaises(ArtifactContractError):
            validator._protocol_expectations(value)

    def test_native_task_ledger_is_reconstructed_without_importing_classifier(self):
        audit = validator._Audit()
        observed = validator._validate_task(
            audit,
            {"task": task_record()},
            "psf",
            validator._protocol_expectations(protocol()),
            allow_contact_terminal=False,
        )
        self.assertEqual(audit.discrepancies, [])
        self.assertTrue(observed["ever_success"])
        self.assertTrue(observed["terminal_success"])
        self.assertEqual(observed["first_success_source_action_index"], 236)

    def test_native_task_mutations_are_rejected(self):
        for name, mutate, expected in (
            (
                "done",
                lambda value: value["goal_progress_ledger"][-1].__setitem__(
                    "returned_done", False
                ),
                "psf_goal_step_56_done_goal_differs",
            ),
            (
                "inert_state",
                lambda value: value["goal_progress_ledger"][12].__setitem__(
                    "simulator_state_sha256_after", "9" * 64
                ),
                "psf_goal_step_11_state_not_inert",
            ),
            (
                "missing_action",
                lambda value: value["goal_progress_ledger"].pop(20),
                "psf_goal_ledger_count_differs",
            ),
        ):
            with self.subTest(name=name):
                value = task_record()
                mutate(value)
                audit = validator._Audit()
                validator._validate_task(
                    audit,
                    {"task": value},
                    "psf",
                    validator._protocol_expectations(protocol()),
                    allow_contact_terminal=False,
                )
                self.assertIn(expected, audit.discrepancies)

    def test_post_correction_motion_includes_every_later_update(self):
        commands = []
        physics = []
        for update, boundary in enumerate((4495, 4500, 4505)):
            commands.append(
                {
                    "source_action_index": 180,
                    "inner_control_index": update,
                    "physical_boundary": boundary,
                    "executed_qdot_rad_s": [float(update + 1)] + [0.0] * 6,
                    "correction_l2_rad_s": float(update),
                    "eef_position_before_update_world_m": [0.01 * update, 0.0, 0.0],
                }
            )
            for substep in range(5):
                physics.append(
                    {
                        "source_action_index": 180,
                        "inner_control_index": update,
                        "physics_substep_index": substep,
                        "measured_qvel_rad_s": [float(update + 1)] + [0.0] * 6,
                        "eef_position_world_m": [
                            0.01 * update + 0.001 * (substep + 1),
                            0.0,
                            0.0,
                        ],
                    }
                )
        audit = validator._Audit()
        observed = validator._post_correction_motion(
            audit,
            commands,
            physics,
            4500,
            validator._protocol_expectations(protocol()),
        )
        self.assertEqual(audit.discrepancies, [])
        self.assertEqual(observed["command_update_count"], 2)
        self.assertEqual(observed["physics_substep_count"], 10)
        self.assertEqual(observed["complete_command_interval_count"], 2)
        self.assertAlmostEqual(observed["filter_correction_integral_rad"], 0.03)
        self.assertAlmostEqual(observed["executed_command_integral_rad"], 0.05)
        self.assertAlmostEqual(observed["measured_joint_motion_integral_rad"], 0.05)
        self.assertAlmostEqual(observed["cartesian_path_length_m"], 0.01)

    def test_no_activation_motion_uses_registered_typed_sentinel(self):
        audit = validator._Audit()
        expectation = validator._protocol_expectations(protocol())
        observed = validator._post_correction_motion(
            audit,
            [],
            [],
            None,
            expectation,
        )
        self.assertEqual(
            observed["first_correction_physical_boundary"],
            expectation["no_event_boundary"],
        )
        self.assertEqual(observed["command_update_count"], 0)
        self.assertEqual(observed["zero_command_fraction"], 1.0)
        self.assertFalse(audit.discrepancies)

    def test_consumer_does_not_import_producer_classifier(self):
        source = VALIDATOR_PATH.read_text(encoding="utf-8")
        self.assertNotIn("classify_full_episode_feasibility", source)
        self.assertNotIn(
            "from main.poisson_fullbody.full_episode_feasibility import", source
        )

    def test_independent_classifier_matches_all_scientific_outcomes(self):
        expectation = validator._protocol_expectations(protocol())
        variants = []
        positive = positive_metrics()
        variants.append(("positive", positive, "SAFE_TASK_SUCCESS_USEFUL_CORRECTION"))

        task_failure = copy.deepcopy(positive)
        task_failure.update(
            {
                "psf_task_success_ever": False,
                "psf_terminal_task_success": False,
                "psf_first_task_success_source_action_index": None,
                "psf_task_success_after_material_correction": False,
            }
        )
        variants.append(
            ("task_failure", task_failure, "CONTACT_PREVENTED_TASK_FAILED")
        )

        stop_only = copy.deepcopy(positive)
        stop_only["post_correction_measured_joint_motion_integral_rad"] = 0.0
        variants.append(("stop_only", stop_only, "STOP_ONLY"))

        contact = copy.deepcopy(positive)
        contact.update(
            {
                "full_recorded_episode_complete": False,
                "psf_exposure_complete": False,
                "psf_filter_update_count": 40,
                "psf_physics_substep_count": 176,
                "psf_completed_suffix_action_count": 0,
                "psf_qp_count_complete": False,
                "psf_qp_postchecks_complete": False,
                "psf_joint_limit_postchecks_complete": False,
                "psf_all_post_state_field_queries_valid_and_positive": False,
                "psf_any_robot_selected_obstacle_contact_present": True,
                "psf_clearance_certified": False,
                "psf_task_success_ever": False,
                "psf_terminal_task_success": False,
                "psf_first_task_success_source_action_index": None,
                "psf_task_success_after_material_correction": False,
            }
        )
        variants.append(("contact", contact, "CONTACT_REMAINS_OR_SHIFTED"))

        for name, metrics, label in variants:
            with self.subTest(name=name):
                independent = validator._independent_classification(
                    metrics, expectation
                )
                producer = classify_full_episode_feasibility(metrics, protocol())
                self.assertEqual(independent, producer)
                self.assertEqual(independent["classification"], label)

    def test_mutated_producer_label_is_rejected(self):
        expected = validator._independent_classification(
            positive_metrics(), validator._protocol_expectations(protocol())
        )
        mutated = copy.deepcopy(expected)
        mutated["classification"] = "STOP_ONLY"
        audit = validator._Audit()
        validator._validate_producer_classification(audit, mutated, expected)
        self.assertIn("producer_classification_differs", audit.discrepancies)

    def test_controller_updates_bind_to_frozen_suffix_action_array(self):
        expectation = validator._protocol_expectations(protocol())
        actions = [
            [0.001 * index, -0.001 * index, 0.0, 0.0, 0.0, 0.0, 1.0]
            for index in range(57)
        ]
        adapter = []
        psf = []
        for source_index, action in zip(expectation["source_actions"], actions):
            for inner in range(expectation["controls_per_action"]):
                row = {
                    "source_action_index": source_index,
                    "inner_control_index": inner,
                    "source_action": action,
                }
                adapter.append(copy.deepcopy(row))
                psf.append(copy.deepcopy(row))
        expected_hash = hashlib.sha256(canonical(actions)).hexdigest()
        audit = validator._Audit()
        with mock.patch.object(
            validator, "EXPECTED_SUFFIX_ACTION_ARRAY_SHA256", expected_hash
        ):
            validator._validate_frozen_source_actions(
                audit, adapter, psf, expectation
            )
        self.assertFalse(audit.discrepancies)

        psf[0]["source_action"][0] += 0.5
        mutated = validator._Audit()
        with mock.patch.object(
            validator, "EXPECTED_SUFFIX_ACTION_ARRAY_SHA256", expected_hash
        ):
            validator._validate_frozen_source_actions(
                mutated, adapter, psf, expectation
            )
        self.assertIn("paired_source_action_differs", mutated.discrepancies)

    def test_complete_synthetic_positive_payload_validates(self):
        payload, post_motion = synthetic_positive_payload()
        adapter_contact = {
            "any_present": True,
            "link_present": True,
            "first_any_boundary": 4677,
            "first_link_boundary": 4677,
            "record_count": 1,
            "minimum_contact_distance_m": -0.001,
            "robot_geom_ids_contacted": [5],
        }
        psf_contact = {
            "any_present": False,
            "link_present": False,
            "first_any_boundary": None,
            "first_link_boundary": None,
            "record_count": 0,
            "minimum_contact_distance_m": None,
            "robot_geom_ids_contacted": [],
        }
        task = {
            "definition": {"same": True},
            "terminal_success": True,
            "ever_success": True,
            "first_success_source_action_index": 236,
            "completed_action_count": 57,
            "fixed_exposure_complete": True,
        }
        material = [
            {
                "boundary": 4500 + index * 5,
                "raw": {"record_kind": "command_trace"},
                "activation_raw": {"synthetic": True},
            }
            for index in range(32)
        ]
        commands = [{} for _ in range(285)]
        physics = [{} for _ in range(1425)]
        with (
            mock.patch.object(
                validator,
                "_reconstruct_contacts",
                side_effect=[adapter_contact, psf_contact],
            ),
            mock.patch.object(
                validator,
                "_validate_cadence",
                side_effect=[(commands, physics), (commands, physics)],
            ),
            mock.patch.object(validator.narrow, "_validate_pairing"),
            mock.patch.object(validator.narrow, "_validate_exogenous_pairing"),
            mock.patch.object(validator, "_validate_frozen_source_actions"),
            mock.patch.object(validator.narrow, "_validate_adapter_execution"),
            mock.patch.object(
                validator.narrow, "_validate_qp", return_value=commands
            ),
            mock.patch.object(
                validator.narrow, "_material_activation", return_value=material
            ),
            mock.patch.object(
                validator, "_post_correction_motion", return_value=post_motion
            ),
            mock.patch.object(
                validator, "_validate_task", side_effect=[task, task]
            ),
            mock.patch.object(
                validator, "_validate_field_and_clearance", return_value=True
            ),
        ):
            observed = validator.validate_payload(
                payload,
                protocol(),
                protocol_file_sha256="a" * 64,
                expected_commit="c" * 40,
            )
        self.assertEqual(observed["discrepancies"], [])
        self.assertTrue(observed["artifact_valid"])
        self.assertEqual(
            observed["classification"],
            "SAFE_TASK_SUCCESS_USEFUL_CORRECTION",
        )


if __name__ == "__main__":
    unittest.main()
