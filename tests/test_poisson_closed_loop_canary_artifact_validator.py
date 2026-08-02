from __future__ import annotations

import copy
import hashlib
import inspect
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

from main.poisson_fullbody.contracts import ArtifactContractError
from scripts import run_poisson_closed_loop_canary as producer
from scripts import validate_poisson_closed_loop_canary_artifact as validator


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = json.loads(
    (ROOT / "configs/vlsa_poisson_closed_loop_canary.v1.json").read_text(
        encoding="utf-8"
    )
)


def positive_metrics():
    return {
        "exact_paired_start": True,
        "shared_prefix_complete": True,
        "baseline_exposure_complete": True,
        "psf_exposure_complete": True,
        "live_policy_contract_valid": True,
        "aegis_contract_valid": True,
        "both_nominal_commands_within_dynamic_joint_bounds": True,
        "videos_complete_and_decodable": True,
        "psf_qp_contract_valid": True,
        "static_selected_obstacle_admissible": True,
        "literal_contact_checked_at_every_physics_substep": True,
        "released_eef_marker_update_contract_valid": True,
        "first_live_query_identical": True,
        "only_psf_query_36_reused_paired_cache": True,
        "historical_first_live_query_action_chunk_matches_diagnostic": False,
        "first_current_action_180_aegis_inputs_identical": True,
        "first_current_action_180_aegis_outputs_identical": True,
        "own_observation_chain_valid": True,
        "no_recorded_suffix_action_replay": True,
        "first_live_aegis_action_matches_historical": False,
        "historical_action_180_full_output_matches_diagnostic": False,
        "post_divergence_own_observations_used": True,
        "fresh_policy_query_after_material_correction": True,
        "baseline_link56_contact_present": True,
        "baseline_first_selected_obstacle_contact_is_link56": True,
        "psf_any_robot_selected_obstacle_contact_present": False,
        "material_correction_before_baseline_contact": True,
        "maximum_correction_norm_rad_s": 0.4,
        "filter_correction_integral_rad": 0.01,
        "post_correction_measured_joint_motion_integral_rad": 0.2,
        "post_correction_cartesian_path_length_m": 0.04,
        "post_correction_executed_command_integral_rad": 0.2,
        "post_correction_zero_command_fraction": 0.0,
        "baseline_task_success_ever": True,
        "psf_task_success_ever": True,
        "psf_task_success_after_material_correction": True,
    }


def first_live_pair():
    observation = {
        "native_observation_sha256": "a" * 64,
        "official_integration_state_raw_bytes_sha256": "b" * 64,
        "agentview_policy_array_sha256": "c" * 64,
        "wrist_policy_array_sha256": "d" * 64,
        "policy_state_array_sha256": "e" * 64,
        "policy_state": [0.0] * 8,
        "prompt": "put the bowl on the plate",
        "policy_input_fingerprint_sha256": "f" * 64,
    }
    chunk = [[0.01 * row] * 7 for row in range(10)]
    query = {
        **observation,
        "query_index": 36,
        "rng_seed": 2026691256,
        "source_action_index": 180,
        "local_action_index": 0,
        "returned_actions": chunk,
        "returned_actions_sha256": validator._array_sha256(chunk),
        "elapsed_seconds": 1.0,
    }
    context = {
        "p1": [0.0, 0.1, 0.2],
        "R1": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "q1_diag": [0.06, 0.12, 0.11],
        "p2": [0.2, 0.3, 0.4],
        "R2": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "Q2_diag": [0.1, 0.1, 0.1],
        "z_before": [1.0, 0.0, 0.0],
        "nominal_translational": [0.1] * 7,
        "executed_action": [0.2] * 7,
        "solver_stats": {"solve_time": 0.01, "num_iters": 25},
    }
    action = {
        "local_action_index": 0,
        "source_action_index": 180,
        "nominal_raw": chunk[0],
        "nominal_translational": [0.1] * 7,
        "aegis_executed": [0.2] * 7,
        "aegis_correction_l2": 0.1,
        "aegis_z_before": [1.0, 0.0, 0.0],
        "aegis_z_after": [0.0, 1.0, 0.0],
        "aegis_qp": {
            "status": "solved",
            "barrier_h": 0.1,
            "constraint_lhs": 0.2,
            "u_solution": [0.3] * 6,
            "z_after": [0.0, 1.0, 0.0],
            "context": context,
        },
    }
    baseline = {"queries": [query], "actions": [action]}
    return baseline, copy.deepcopy(baseline)


class ProtocolAndHashTests(unittest.TestCase):
    def test_protocol_derives_the_frozen_live_suffix(self):
        expectation = validator._protocol_expectations(PROTOCOL)
        self.assertEqual(expectation["source_actions"], tuple(range(180, 237)))
        self.assertEqual(expectation["filter_updates"], 285)
        self.assertEqual(expectation["physics_substeps"], 1425)
        self.assertEqual(expectation["clearance_stride"], 25)
        self.assertEqual(
            validator.EXPECTED_BOUNDARY_RAW_SHA256,
            "ff1a9a80d03178c9ac85e5dc6750aac98f920a050462df0b17d7ca1cdbb069c7",
        )

    def test_protocol_rejects_changed_first_live_chunk(self):
        candidate = copy.deepcopy(PROTOCOL)
        candidate["source"][
            "historical_first_live_query_action_chunk_sha256_diagnostic"
        ] = "0" * 64
        with self.assertRaises(ArtifactContractError):
            validator._protocol_expectations(candidate)

    def test_independent_array_hash_matches_producer_format(self):
        array = [[float(row * 7 + column) for column in range(7)] for row in range(10)]
        self.assertEqual(validator._array_sha256(array), producer._array_sha256(array))
        array[0][0] = 0.125
        self.assertEqual(validator._array_sha256(array), producer._array_sha256(array))

    def test_generic_video_array_hash_binds_uint8_dtype_shape_and_bytes(self):
        class Dtype:
            str = "|u1"

        class Array(bytearray):
            pass

        array = Array(bytes(range(6)))
        array.dtype = Dtype()
        array.shape = (1, 2, 3)
        numpy = types.ModuleType("numpy")
        numpy.ascontiguousarray = lambda value: value
        expected = hashlib.sha256()
        expected.update(b"vlsa-table1-array-v1\0")
        expected.update(validator._canonical({"dtype": "|u1", "shape": [1, 2, 3]}))
        expected.update(b"\0")
        expected.update(bytes(range(6)))
        with mock.patch.dict(sys.modules, {"numpy": numpy}):
            observed = validator._numpy_array_sha256(array)
        self.assertEqual(observed, expected.hexdigest())

    def test_commit_identity_is_lowercase_sha40(self):
        self.assertEqual(validator._commit("a" * 40, "test"), "a" * 40)
        for value in ("a" * 39, "A" * 40, "g" * 40):
            with self.subTest(value=value):
                with self.assertRaises(ArtifactContractError):
                    validator._commit(value, "test")

    def test_consumer_repo_binding_checks_clean_commit_and_branch(self):
        commit = "a" * 40
        with mock.patch.object(
            validator.subprocess,
            "check_output",
            side_effect=["\n", commit + "\n", validator.EXPECTED_BRANCH + "\n"],
        ) as command:
            validator._validate_consumer_repo(ROOT, commit)
        self.assertEqual(command.call_count, 3)

        with mock.patch.object(
            validator.subprocess,
            "check_output",
            return_value="dirty\n",
        ):
            with self.assertRaises(ArtifactContractError):
                validator._validate_consumer_repo(ROOT, commit)


class IndependentClassificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.expectation = validator._protocol_expectations(PROTOCOL)

    def classify(self, metrics):
        return validator._independent_classification(metrics, self.expectation)

    def test_positive_means_no_contact_useful_motion_and_task_success(self):
        result = self.classify(positive_metrics())
        self.assertEqual(result["classification"], "SAFE_TASK_SUCCESS_USEFUL_CORRECTION")
        self.assertTrue(result["feasible"])
        self.assertTrue(result["useful_correction"])
        self.assertTrue(result["task_successful"])
        self.assertFalse(result["stop_only"])

    def test_baseline_task_failure_does_not_block_psf_feasibility(self):
        metrics = positive_metrics()
        metrics["baseline_task_success_ever"] = False
        result = self.classify(metrics)
        self.assertEqual(
            result["classification"],
            "SAFE_TASK_SUCCESS_USEFUL_CORRECTION",
        )
        self.assertTrue(result["feasible"])
        self.assertTrue(result["task_successful"])

    def test_stop_only_is_not_feasible(self):
        metrics = positive_metrics()
        metrics["post_correction_cartesian_path_length_m"] = 0.0
        result = self.classify(metrics)
        self.assertEqual(result["classification"], "STOP_ONLY")
        self.assertTrue(result["stop_only"])
        self.assertFalse(result["feasible"])

    def test_task_failure_after_contact_prevention_is_not_feasible(self):
        metrics = positive_metrics()
        metrics["psf_task_success_after_material_correction"] = False
        result = self.classify(metrics)
        self.assertEqual(result["classification"], "CONTACT_PREVENTED_TASK_FAILED")
        self.assertFalse(result["feasible"])

    def test_treatment_contact_is_a_valid_scientific_negative(self):
        metrics = positive_metrics()
        metrics.update(
            {
                "psf_exposure_complete": False,
                "psf_any_robot_selected_obstacle_contact_present": True,
                "post_divergence_own_observations_used": False,
                "fresh_policy_query_after_material_correction": False,
            }
        )
        result = self.classify(metrics)
        self.assertEqual(result["classification"], "CONTACT_REMAINS_OR_SHIFTED")
        self.assertTrue(result["pair_complete"])
        self.assertFalse(result["feasible"])

    def test_missing_baseline_contact_is_not_a_positive(self):
        metrics = positive_metrics()
        metrics["baseline_link56_contact_present"] = False
        result = self.classify(metrics)
        self.assertEqual(result["classification"], "BASELINE_CONTACT_NOT_REPRODUCED")
        self.assertFalse(result["feasible"])

    def test_feedback_or_video_failure_is_inconclusive(self):
        for field in (
            "first_live_query_identical",
            "own_observation_chain_valid",
            "post_divergence_own_observations_used",
            "fresh_policy_query_after_material_correction",
            "videos_complete_and_decodable",
        ):
            with self.subTest(field=field):
                metrics = positive_metrics()
                metrics[field] = False
                result = self.classify(metrics)
                self.assertEqual(result["classification"], "INCONCLUSIVE_APPARATUS")

    def test_historical_output_and_query_diagnostics_do_not_gate(self):
        metrics = positive_metrics()
        metrics[
            "historical_first_live_query_action_chunk_matches_diagnostic"
        ] = False
        metrics["first_live_aegis_action_matches_historical"] = False
        metrics[
            "historical_action_180_full_output_matches_diagnostic"
        ] = False
        result = self.classify(metrics)
        self.assertEqual(
            result["classification"],
            "SAFE_TASK_SUCCESS_USEFUL_CORRECTION",
        )
        self.assertTrue(result["feasible"])

    def test_validator_does_not_import_or_call_producer_classifier(self):
        source = inspect.getsource(validator)
        self.assertNotIn("classify_closed_loop_canary", source)
        self.assertIn("def _independent_classification", source)


class PairedLiveAuthorityTests(unittest.TestCase):
    def test_q36_is_inferred_once_then_reused_and_q37_plus_are_live(self):
        cases = (
            (
                "baseline",
                36,
                {
                    "query_execution": "live_inference",
                    "inference_performed": True,
                    "paired_cache_source_arm": None,
                    "paired_cache_source_query_index": None,
                    "paired_cache_source_returned_actions_sha256": None,
                    "returned_actions_sha256": "current",
                    "historical_returned_actions_sha256_diagnostic": validator.EXPECTED_FIRST_CHUNK_SHA256,
                    "matches_historical_returned_actions_sha256_diagnostic": False,
                },
            ),
            (
                "psf",
                36,
                {
                    "query_execution": "paired_cache_reuse",
                    "inference_performed": False,
                    "paired_cache_source_arm": "pi05_plus_aegis_joint_velocity_adapter",
                    "paired_cache_source_query_index": 36,
                    "paired_cache_source_returned_actions_sha256": "current",
                    "returned_actions_sha256": "current",
                    "historical_returned_actions_sha256_diagnostic": validator.EXPECTED_FIRST_CHUNK_SHA256,
                    "matches_historical_returned_actions_sha256_diagnostic": False,
                },
            ),
            (
                "baseline",
                37,
                {
                    "query_execution": "live_inference",
                    "inference_performed": True,
                    "paired_cache_source_arm": None,
                    "paired_cache_source_query_index": None,
                    "paired_cache_source_returned_actions_sha256": None,
                    "historical_returned_actions_sha256_diagnostic": None,
                    "matches_historical_returned_actions_sha256_diagnostic": None,
                },
            ),
            (
                "psf",
                37,
                {
                    "query_execution": "live_inference",
                    "inference_performed": True,
                    "paired_cache_source_arm": None,
                    "paired_cache_source_query_index": None,
                    "paired_cache_source_returned_actions_sha256": None,
                    "historical_returned_actions_sha256_diagnostic": None,
                    "matches_historical_returned_actions_sha256_diagnostic": None,
                },
            ),
        )
        for label, index, query in cases:
            with self.subTest(label=label, index=index):
                audit = validator._Audit()
                validator._validate_query_execution(audit, query, label, index)
                self.assertEqual(audit.discrepancies, [])

        attacks = (
            ("baseline", 36, {"query_execution": "paired_cache_reuse"}),
            ("psf", 36, {"query_execution": "live_inference"}),
            (
                "psf",
                36,
                {
                    "query_execution": "paired_cache_reuse",
                    "paired_cache_source_arm": "wrong-arm",
                    "paired_cache_source_query_index": 36,
                },
            ),
            ("psf", 37, {"query_execution": "paired_cache_reuse"}),
        )
        for label, index, query in attacks:
            with self.subTest(attack=(label, index, query)):
                audit = validator._Audit()
                validator._validate_query_execution(audit, query, label, index)
                self.assertTrue(audit.discrepancies)

    def test_current_q36_and_action180_pair_is_authority_not_historical_hash(self):
        baseline, psf = first_live_pair()
        self.assertNotEqual(
            baseline["queries"][0]["returned_actions_sha256"],
            validator.EXPECTED_FIRST_CHUNK_SHA256,
        )
        audit = validator._Audit()
        self.assertTrue(
            validator._validate_first_live_pair(audit, baseline, psf)
        )
        self.assertEqual(audit.discrepancies, [])

    def test_pair_ignores_timing_only(self):
        baseline, psf = first_live_pair()
        psf["queries"][0]["elapsed_seconds"] = 9.0
        psf["actions"][0]["aegis_qp"]["context"]["solver_stats"][
            "solve_time"
        ] = 8.0
        audit = validator._Audit()
        self.assertTrue(
            validator._validate_first_live_pair(audit, baseline, psf)
        )
        self.assertEqual(audit.discrepancies, [])

    def test_frozen_aegis_projections_match_producer_contract(self):
        baseline, psf = first_live_pair()
        for historical, row in (
            (False, baseline["actions"][0]),
            (False, psf["actions"][0]),
        ):
            self.assertEqual(
                validator._aegis_input_projection(
                    row, historical=historical, include_nominal=True
                ),
                producer._aegis_input_projection(
                    row, historical=historical, include_nominal=True
                ),
            )
            self.assertEqual(
                validator._aegis_output_projection(
                    row, historical=historical
                ),
                producer._aegis_output_projection(
                    row, historical=historical
                ),
            )
        baseline_provider = {
            "high_level_action_trace": baseline["actions"]
        }
        psf_provider = {"high_level_action_trace": psf["actions"]}
        self.assertEqual(
            validator._first_action_pair_metrics(baseline, psf),
            producer._first_action_pair_metrics(
                baseline_provider, psf_provider
            ),
        )

    def test_pair_rejects_q36_nominal_context_output_or_executed_drift(self):
        mutations = (
            lambda value: value["queries"][0]["returned_actions"][0].__setitem__(0, 9.0),
            lambda value: value["actions"][0].__setitem__("nominal_translational", [9.0] * 7),
            lambda value: value["actions"][0]["aegis_qp"]["context"].__setitem__("p1", [9.0] * 3),
            lambda value: value["actions"][0]["aegis_qp"].__setitem__("barrier_h", 9.0),
            lambda value: value["actions"][0].__setitem__("aegis_executed", [9.0] * 7),
        )
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                baseline, psf = first_live_pair()
                mutate(psf)
                audit = validator._Audit()
                self.assertFalse(
                    validator._validate_first_live_pair(audit, baseline, psf)
                )
                self.assertTrue(audit.discrepancies)


class FilesystemAndCadenceTests(unittest.TestCase):
    def test_safe_child_rejects_traversal_and_symlink(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "real.bin"
            target.write_bytes(b"evidence")
            self.assertEqual(validator._safe_child(root, "real.bin", "test"), target)
            with self.assertRaises(ArtifactContractError):
                validator._safe_child(root, "../real.bin", "test")
            link = root / "link.bin"
            link.symlink_to(target)
            with self.assertRaises(ArtifactContractError):
                validator._safe_child(root, "link.bin", "test")

    def test_periodic_clearance_requires_exact_25_substep_schedule(self):
        audit = validator._Audit()
        physics = []
        for index in range(50):
            physics.append(
                {
                    "full_surface_clearance_observed_this_substep": (index + 1) % 25 == 0,
                    "cumulative_full_robot_surface_clearance_lower_bound_m": 0.02,
                }
            )
        arm = {
            "conservative_full_robot_clearance": {
                "available": True,
                "observation_stride_physics_substeps": 25,
                "observation_count_including_branch": 3,
                "continuous_every_substep_certificate": False,
                "minimum_full_surface_lower_bound_m": 0.02,
            }
        }
        expectation = {"clearance_stride": 25, "physics_substeps": 50}
        self.assertTrue(
            validator._validate_periodic_clearance(
                audit, arm, physics, "psf", expectation
            )
        )
        self.assertEqual(audit.discrepancies, [])
        broken = copy.deepcopy(physics)
        broken[0]["full_surface_clearance_observed_this_substep"] = True
        audit = validator._Audit()
        validator._validate_periodic_clearance(audit, arm, broken, "psf", expectation)
        self.assertIn("psf_clearance_stride_differs", audit.discrepancies)

    def test_no_correction_motion_uses_producer_none_sentinel(self):
        audit = validator._Audit()
        motion = validator._post_correction_motion(
            audit,
            [],
            [],
            None,
            {
                "no_event_boundary": 5926,
                "control_dt_s": 0.01,
                "physics_dt_s": 0.002,
                "substeps_per_control": 5,
            },
        )
        self.assertIsNone(motion["first_correction_physical_boundary"])
        self.assertEqual(motion["command_update_count"], 0)
        self.assertEqual(motion["physics_substep_count"], 0)
        self.assertEqual(motion["zero_command_fraction"], 1.0)
        self.assertEqual(audit.discrepancies, [])


if __name__ == "__main__":
    unittest.main()
