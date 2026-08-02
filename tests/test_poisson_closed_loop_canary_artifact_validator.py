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
        "own_observation_chain_valid": True,
        "no_recorded_suffix_action_replay": True,
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
        candidate["source"]["first_live_query_expected_action_chunk_sha256"] = "0" * 64
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

    def test_validator_does_not_import_or_call_producer_classifier(self):
        source = inspect.getsource(validator)
        self.assertNotIn("classify_closed_loop_canary", source)
        self.assertIn("def _independent_classification", source)


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
