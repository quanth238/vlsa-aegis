import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from main.poisson_fullbody.contracts import load_hashed_json, publish_hashed_json
from scripts.validate_poisson_fast_feasibility_artifact import (
    ACTIVATION_RESIDUAL_MAXIMUM,
    CORRECTION_MINIMUM,
    CARTESIAN_PATH_MINIMUM,
    _Audit,
    _active_motion,
    _reconstruct_contacts,
    validate_artifact,
)


ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "scripts/validate_poisson_fast_feasibility_artifact.py"
LOCAL_RESULT = Path(
    "/Users/quanth238/personal/Research/probe_vla/output/"
    "vlsa_poisson_fast_feasibility/vlsa-poisson-fast-window-20260802e/result.json"
)


class FastFeasibilityValidatorUnitTests(unittest.TestCase):
    def test_validator_is_independent_from_producer_classifier(self):
        source = VALIDATOR.read_text(encoding="utf-8")
        self.assertIn("load_hashed_json", source)
        self.assertNotIn("classify_fast_feasibility", source)

    def test_literal_contact_is_reconstructed_from_signed_distance(self):
        record = {
            "contact_distance_m": 0.001,
            "is_physical_nonpositive_distance_contact": True,
            "source_phase": "live_solver_phase_preintegration_geometry",
            "observation_index": 0,
            "high_level_index": 0,
            "inner_control_index": 0,
            "physics_substep_index": 0,
            "robot_geom_id": 44,
            "obstacle_geom_id": 165,
        }
        arm = {
            "measurement": {
                "settled_state": {"any_robot_obstacle_contact": False},
                "live_solver_phase_contact_point_records": [record],
                "post_state_physical_contact_point_records": [],
                "live_solver_nonpositive_contact_point_record_count": 0,
                "post_state_physical_contact_point_record_count": 0,
                "rollout_phase_physical_contact_point_record_count": 0,
                "total_physical_contact_point_record_count": 0,
                "any_robot_obstacle_contact": False,
                "rollout_any_robot_obstacle_contact": False,
                "live_solver_any_robot_obstacle_contact": False,
                "post_state_any_robot_obstacle_contact": False,
                "link56_obstacle_contact": False,
                "physical_contact_distance_semantics": "mujoco_contact_dist_le_0",
            },
            "literal_contact": {
                "any_robot_selected_obstacle_present": False,
                "link56_present": False,
                "first_any_robot_physical_boundary": None,
                "first_link56_physical_boundary": None,
            },
        }
        audit = _Audit()
        reconstructed = _reconstruct_contacts(
            audit, arm, (44,), (44,), (165,), "probe"
        )
        self.assertFalse(reconstructed["any_present"])
        self.assertIn(
            "probe_live_solver_phase_contact_point_records_0_physical_flag_differs",
            audit.discrepancies,
        )

    def test_active_motion_distinguishes_motion_from_stop_only(self):
        command = {
            "source": 180,
            "inner": 0,
            "nominal": (0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            "executed": (0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            "correction": 0.1,
            "raw": {
                "eef_position_before_update_world_m": [0.0, 0.0, 0.0],
                "target_error_before_update_m": 0.2,
            },
        }
        physics = [
            {
                "source_action_index": 180,
                "inner_control_index": 0,
                "physics_substep_index": index,
                "measured_qvel_rad_s": [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "eef_position_world_m": [0.0, 0.0, 0.0],
                "target_error_after_substep_m": 0.19,
            }
            for index in range(5)
        ]
        motion = _active_motion(_Audit(), (command,), physics)
        self.assertGreater(motion["measured_joint_motion_integral_rad"], 0.0)
        self.assertEqual(motion["cartesian_path_length_m"], 0.0)
        self.assertLess(motion["cartesian_path_length_m"], CARTESIAN_PATH_MINIMUM)


@unittest.skipUnless(LOCAL_RESULT.is_file(), "complete local H100 artifact unavailable")
class FastFeasibilityValidatorArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = load_hashed_json(LOCAL_RESULT)

    def _mutated_result(self, mutate):
        payload = copy.deepcopy(self.payload)
        payload.pop("result_payload_sha256")
        mutate(payload)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        run_root = Path(temporary.name) / payload["run_id"]
        run_root.mkdir()
        result = run_root / "result.json"
        publish_hashed_json(result, payload)
        return result

    def test_complete_artifact_validates_as_useful_contact_prevention(self):
        summary = validate_artifact(LOCAL_RESULT, expected_job_id="33907")
        self.assertTrue(summary["artifact_valid"])
        self.assertTrue(summary["empirical_contact_prevention"])
        self.assertTrue(summary["clearance_certified"])
        self.assertTrue(summary["useful_motion"])
        self.assertFalse(summary["stop_only"])
        self.assertFalse(summary["tracking_diagnostic"]["tracking_certified"])
        self.assertFalse(summary["timing_diagnostic"]["realtime_100hz_claim_supported"])

    def test_pairing_mutation_is_rejected(self):
        result = self._mutated_result(
            lambda payload: payload["arms"]["adapter_plus_psf"]["restore"].__setitem__(
                "model_topology_sha256", "0" * 64
            )
        )
        summary = validate_artifact(result)
        self.assertFalse(summary["artifact_valid"])
        self.assertIn("pair_restore_model_topology_sha256_differs", summary["discrepancies"])

    def test_missing_restore_and_raw_boundary_authority_is_rejected(self):
        def mutate(payload):
            payload["window"]["official_boundary_B_raw_bytes_sha256"] = None
            for arm in payload["arms"].values():
                arm["start_official_raw_bytes_sha256"] = None
                arm["restore"].pop("model_topology_sha256")

        result = self._mutated_result(mutate)
        summary = validate_artifact(result)
        self.assertFalse(summary["artifact_valid"])
        self.assertIn("boundary_B_raw_hash_invalid", summary["discrepancies"])
        self.assertIn(
            "adapter_restore_model_topology_sha256_invalid",
            summary["discrepancies"],
        )
        self.assertIn("adapter_start_boundary_hash_invalid", summary["discrepancies"])

    def test_high_level_action_pairing_mutation_is_rejected(self):
        def mutate(payload):
            payload["arms"]["adapter_plus_psf"]["command_trace"][0]["adapter"][
                "diagnostics"
            ]["position_error_m"][0] += 0.01

        result = self._mutated_result(mutate)
        summary = validate_artifact(result)
        self.assertFalse(summary["artifact_valid"])
        self.assertIn(
            "pair_command_0_position_error_m_differs", summary["discrepancies"]
        )

    def test_adapter_command_mutation_is_rejected(self):
        def mutate(payload):
            payload["arms"]["adapter_only"]["command_trace"][0][
                "executed_qdot_rad_s"
            ][0] = 0.75

        result = self._mutated_result(mutate)
        summary = validate_artifact(result)
        self.assertFalse(summary["artifact_valid"])
        self.assertIn(
            "adapter_command_0_nominal_executed_differs",
            summary["discrepancies"],
        )

    def test_contact_geometry_authority_mutation_is_rejected(self):
        def mutate(payload):
            payload["arms"]["adapter_only"]["measurement"][
                "live_solver_phase_contact_point_records"
            ][0]["obstacle_geom_id"] = 999999

        result = self._mutated_result(mutate)
        summary = validate_artifact(result)
        self.assertFalse(summary["artifact_valid"])
        self.assertIn(
            "adapter_live_solver_phase_contact_point_records_0_obstacle_geom_outside_authority",
            summary["discrepancies"],
        )

    def test_raw_contact_mutation_is_rejected_even_when_summary_is_stale(self):
        def mutate(payload):
            adapter_measurement = payload["arms"]["adapter_only"]["measurement"]
            psf_measurement = payload["arms"]["adapter_plus_psf"]["measurement"]
            psf_measurement["live_solver_phase_contact_point_records"].append(
                copy.deepcopy(adapter_measurement["live_solver_phase_contact_point_records"][0])
            )

        result = self._mutated_result(mutate)
        summary = validate_artifact(result)
        self.assertFalse(summary["artifact_valid"])
        self.assertIn("psf_measurement_any_contact_differs", summary["discrepancies"])

    def test_hidden_rollout_candidate_and_physics_contact_flag_are_rejected(self):
        def mutate(payload):
            adapter_measurement = payload["arms"]["adapter_only"]["measurement"]
            psf = payload["arms"]["adapter_plus_psf"]
            psf_measurement = psf["measurement"]
            record = copy.deepcopy(
                adapter_measurement["post_state_candidate_contact_point_records"][0]
            )
            psf_measurement["post_state_candidate_contact_point_records"] = [record]
            psf_measurement["post_state_candidate_contact_point_record_count"] = 1
            psf_measurement["total_candidate_contact_point_record_count"] = 1
            psf["physics_trace"][0]["literal_contact_observed"] = True

        result = self._mutated_result(mutate)
        summary = validate_artifact(result)
        self.assertFalse(summary["artifact_valid"])
        self.assertIn(
            "psf_post_state_physical_ledger_differs", summary["discrepancies"]
        )
        self.assertIn(
            "psf_physics_literal_contact_flags_differ", summary["discrepancies"]
        )

    def test_settled_raw_contact_with_stale_summary_is_rejected(self):
        def mutate(payload):
            record = copy.deepcopy(
                payload["arms"]["adapter_only"]["measurement"]
                ["post_state_physical_contact_point_records"][0]
            )
            record["source_phase"] = "settled_post_integration_recomputed"
            for field in (
                "observation_index",
                "high_level_index",
                "inner_control_index",
                "physics_substep_index",
            ):
                record[field] = None
            settled = payload["arms"]["adapter_plus_psf"]["measurement"][
                "settled_state"
            ]
            settled["candidate_contact_point_records"] = [record]
            settled["candidate_contact_point_record_count"] = 1
            settled["first_candidate_contact_point_record"] = record
            settled["physical_contact_point_records"] = [record]
            settled["physical_contact_point_record_count"] = 1
            settled["first_physical_contact_point_record"] = record

        result = self._mutated_result(mutate)
        summary = validate_artifact(result)
        self.assertFalse(summary["artifact_valid"])
        self.assertIn("psf_settled_contact_present", summary["discrepancies"])
        self.assertIn("psf_settled_any_contact_differs", summary["discrepancies"])

    def test_qp_residual_mutation_is_rejected(self):
        def mutate(payload):
            payload["arms"]["adapter_plus_psf"]["command_trace"][0][
                "safe_cbf_residual_minimum_m2_per_s"
            ] = -0.01

        result = self._mutated_result(mutate)
        summary = validate_artifact(result)
        self.assertFalse(summary["artifact_valid"])
        self.assertIn("psf_command_0_safe_residual_failed", summary["discrepancies"])

    def test_physics_issued_velocity_must_match_filtered_command(self):
        def mutate(payload):
            payload["arms"]["adapter_plus_psf"]["physics_trace"][0][
                "issued_qvel_rad_s"
            ] = [0.0] * 7

        result = self._mutated_result(mutate)
        summary = validate_artifact(result)
        self.assertFalse(summary["artifact_valid"])
        self.assertIn(
            "psf_physics_0_issued_command_differs", summary["discrepancies"]
        )

    def test_explicit_qp_and_field_apparatus_failures_are_rejected(self):
        def mutate(payload):
            psf = payload["arms"]["adapter_plus_psf"]
            psf["precontact_execution_valid"] = False
            psf["precontact_field_queries_valid_and_positive"] = False
            psf["all_post_state_field_queries_valid_and_positive"] = False
            metrics = payload["metrics"]
            metrics["psf_all_qp_postchecks_passed"] = False
            metrics["psf_all_joint_limit_postchecks_passed"] = False
            metrics["psf_precontact_execution_valid"] = False
            metrics["psf_precontact_field_queries_valid_and_positive"] = False
            metrics["psf_all_post_state_field_queries_valid_and_positive"] = False

        result = self._mutated_result(mutate)
        summary = validate_artifact(result)
        self.assertFalse(summary["artifact_valid"])
        self.assertIn(
            "psf_precontact_execution_valid_flag_differs",
            summary["discrepancies"],
        )
        self.assertIn(
            "metric_psf_all_qp_postchecks_passed_differs",
            summary["discrepancies"],
        )

    def test_clearance_radius_must_match_full_robot_sampling_certificate(self):
        def mutate(payload):
            for record in payload["field"]["full_robot_sampling"]["geom_records"]:
                record["certified_surface_cover_radius_m"] = 0.001
            psf = payload["arms"]["adapter_plus_psf"]
            sample = psf["measurement"]["sample_clearance"]
            sample["certified_coverage_radius_m"] = 0.001
            inflated = sample["minimum_exact_sample_to_obb_distance_m"] - 0.001
            sample["full_surface_clearance_lower_bound_m"] = inflated
            psf["measurement"]["D_sim_min_m"] = inflated
            psf["minimum_D_sim_m_diagnostic_only"] = inflated
            psf["conservative_full_robot_clearance"][
                "minimum_full_surface_lower_bound_m"
            ] = inflated
            for row in psf["physics_trace"]:
                row[
                    "cumulative_full_robot_surface_clearance_lower_bound_m"
                ] = inflated
            payload["metrics"][
                "psf_conservative_full_robot_surface_clearance_lower_bound_m"
            ] = inflated
            payload["metrics"]["psf_minimum_D_sim_m"] = inflated

        result = self._mutated_result(mutate)
        summary = validate_artifact(result)
        self.assertFalse(summary["artifact_valid"])
        self.assertIn(
            "full_robot_sampling_hash_differs",
            summary["discrepancies"],
        )

    def test_reconstructed_stop_only_is_kept_separate_from_useful_motion(self):
        def mutate(payload):
            psf = payload["arms"]["adapter_plus_psf"]
            contact_boundary = payload["metrics"][
                "adapter_first_link56_contact_physical_boundary"
            ]
            starts = {}
            for command in psf["command_trace"]:
                correction = command["correction_l2_rad_s"]
                if (
                    command["physical_boundary"] < contact_boundary
                    and command["nominal_cbf_residual_minimum_m2_per_s"]
                    <= ACTIVATION_RESIDUAL_MAXIMUM
                    and correction >= CORRECTION_MINIMUM
                    and command["nominal_within_dynamic_joint_bounds"] is True
                ):
                    starts[
                        (command["source_action_index"], command["inner_control_index"])
                    ] = command["eef_position_before_update_world_m"]
            for row in psf["physics_trace"]:
                key = (row["source_action_index"], row["inner_control_index"])
                if key in starts:
                    row["eef_position_world_m"] = list(starts[key])
            payload["activation_evidence"]["active_interval_motion"][
                "cartesian_path_length_m"
            ] = 0.0
            payload["metrics"]["active_cartesian_path_length_m"] = 0.0
            classification = payload["classification"]
            classification["safety_mechanism"] = "STOP_ONLY"
            classification["motion_preserving_correction"] = False
            classification["stop_only"] = True
            classification["motion_preservation_checks"][
                "material_cartesian_motion"
            ] = False

        result = self._mutated_result(mutate)
        summary = validate_artifact(result)
        self.assertTrue(summary["artifact_valid"], summary["discrepancies"])
        self.assertTrue(summary["empirical_contact_prevention"])
        self.assertFalse(summary["useful_motion"])
        self.assertTrue(summary["stop_only"])

    def test_reconstructed_nonpositive_clearance_is_uncertified(self):
        def mutate(payload):
            psf = payload["arms"]["adapter_plus_psf"]
            minimum = -0.001
            for row in psf["physics_trace"]:
                row[
                    "cumulative_full_robot_surface_clearance_lower_bound_m"
                ] = minimum
            psf["conservative_full_robot_clearance"][
                "minimum_full_surface_lower_bound_m"
            ] = minimum
            psf["minimum_D_sim_m_diagnostic_only"] = minimum
            measurement = psf["measurement"]
            measurement["D_sim_min_m"] = minimum
            sample = measurement["sample_clearance"]
            sample["full_surface_clearance_lower_bound_m"] = minimum
            sample["minimum_exact_sample_to_obb_distance_m"] = (
                sample["certified_coverage_radius_m"] + minimum
            )
            payload["metrics"][
                "psf_conservative_full_robot_surface_clearance_lower_bound_m"
            ] = minimum
            payload["metrics"]["psf_minimum_D_sim_m"] = minimum
            classification = payload["classification"]
            classification["primary_outcome"] = "UNCERTIFIED_CLEARANCE"
            classification["safety_mechanism"] = "NOT_APPLICABLE"
            classification["contact_prevention_label"] = None
            classification["contact_prevention_feasible"] = False
            classification["uncertified_clearance"] = True
            classification["motion_preserving_correction"] = False
            classification["stop_only"] = False

        result = self._mutated_result(mutate)
        summary = validate_artifact(result)
        self.assertTrue(summary["artifact_valid"], summary["discrepancies"])
        self.assertTrue(summary["empirical_contact_prevention"])
        self.assertFalse(summary["clearance_certified"])
        self.assertEqual(
            summary["producer_classification"]["primary_outcome"],
            "UNCERTIFIED_CLEARANCE",
        )

    def test_cli_returns_nonzero_for_rehashed_structural_failure(self):
        result = self._mutated_result(
            lambda payload: payload["window"].__setitem__("end_boundary", 4699)
        )
        completed = subprocess.run(
            [sys.executable, str(VALIDATOR), "--result", str(result)],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 2)
        summary = json.loads(completed.stdout)
        self.assertFalse(summary["artifact_valid"])
        self.assertIn("window_end_differs", summary["discrepancies"])


if __name__ == "__main__":
    unittest.main()
