from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import unittest

from main.poisson_fullbody.full_episode_feasibility import (
    CLASSIFICATIONS,
    METRIC_KEYS,
    METRICS_SCHEMA,
    FullEpisodeFeasibilityError,
    classify_full_episode_feasibility,
    validate_full_episode_feasibility_protocol,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "configs/vlsa_poisson_full_episode_feasibility.v1.json"


def _positive_metrics():
    return {
        "schema_version": METRICS_SCHEMA,
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
        "material_correction_update_count": 35,
        "maximum_correction_norm_rad_s": 0.5,
        "filter_correction_integral_rad": 0.1,
        "post_correction_measured_joint_motion_integral_rad": 0.5,
        "post_correction_cartesian_path_length_m": 0.05,
        "post_correction_executed_command_integral_rad": 0.5,
        "post_correction_zero_command_fraction": 0.0,
        "psf_task_success_ever": True,
        "psf_terminal_task_success": True,
        "psf_first_task_success_source_action_index": 236,
        "psf_task_success_after_material_correction": True,
        "adapter_task_success_ever": True,
        "adapter_terminal_task_success": True,
    }


class FullEpisodeFeasibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))

    def test_protocol_derives_complete_suffix_counts(self):
        derived = validate_full_episode_feasibility_protocol(self.protocol)
        self.assertEqual(derived["start_action"], 180)
        self.assertEqual(derived["end_action"], 236)
        self.assertEqual(derived["action_count"], 57)
        self.assertEqual(derived["start_boundary"], 4500)
        self.assertEqual(derived["end_boundary"], 5925)
        self.assertEqual(derived["expected_updates"], 285)
        self.assertEqual(derived["expected_substeps"], 1425)

    def test_exact_metric_contract_has_all_named_fields(self):
        metrics = _positive_metrics()
        self.assertEqual(len(METRIC_KEYS), 44)
        self.assertEqual(set(metrics), set(METRIC_KEYS))

    def test_positive_requires_safe_useful_native_task_success(self):
        result = classify_full_episode_feasibility(
            _positive_metrics(), self.protocol
        )
        self.assertEqual(
            result["classification"],
            "SAFE_TASK_SUCCESS_USEFUL_CORRECTION",
        )
        self.assertTrue(result["goal_achieved"])
        self.assertTrue(result["contact_prevented"])
        self.assertTrue(result["useful_correction"])
        self.assertTrue(result["task_successful"])

    def test_contact_prevented_but_task_failed_is_not_feasible(self):
        metrics = _positive_metrics()
        metrics["psf_task_success_ever"] = False
        metrics["psf_terminal_task_success"] = False
        metrics["psf_task_success_after_material_correction"] = False
        metrics["psf_first_task_success_source_action_index"] = None
        result = classify_full_episode_feasibility(metrics, self.protocol)
        self.assertEqual(result["classification"], "CONTACT_PREVENTED_TASK_FAILED")
        self.assertFalse(result["goal_achieved"])

    def test_contact_prevented_by_stopping_is_not_feasible(self):
        metrics = _positive_metrics()
        metrics["post_correction_cartesian_path_length_m"] = 0.0
        metrics["psf_task_success_ever"] = False
        metrics["psf_terminal_task_success"] = False
        metrics["psf_task_success_after_material_correction"] = False
        metrics["psf_first_task_success_source_action_index"] = None
        result = classify_full_episode_feasibility(metrics, self.protocol)
        self.assertEqual(result["classification"], "STOP_ONLY")
        self.assertTrue(result["stop_only"])
        self.assertFalse(result["goal_achieved"])

    def test_shifted_robot_contact_is_a_valid_negative(self):
        metrics = _positive_metrics()
        metrics.update(
            {
                "full_recorded_episode_complete": False,
                "psf_exposure_complete": False,
                "psf_filter_update_count": 40,
                "psf_physics_substep_count": 200,
                "psf_completed_suffix_action_count": 8,
                "psf_qp_count_complete": False,
                "psf_qp_postchecks_complete": False,
                "psf_joint_limit_postchecks_complete": False,
                "psf_all_post_state_field_queries_valid_and_positive": False,
                "psf_link56_contact_present": False,
                "psf_any_robot_selected_obstacle_contact_present": True,
                "psf_clearance_certified": False,
                "psf_task_success_ever": False,
                "psf_terminal_task_success": False,
                "psf_task_success_after_material_correction": False,
                "psf_first_task_success_source_action_index": None,
            }
        )
        result = classify_full_episode_feasibility(metrics, self.protocol)
        self.assertEqual(result["classification"], "CONTACT_REMAINS_OR_SHIFTED")
        self.assertTrue(result["apparatus_valid"])
        self.assertFalse(result["goal_achieved"])

    def test_contact_during_first_suffix_action_is_a_valid_negative(self):
        metrics = _positive_metrics()
        metrics.update(
            {
                "full_recorded_episode_complete": False,
                "psf_exposure_complete": False,
                "psf_filter_update_count": 1,
                "psf_physics_substep_count": 1,
                "psf_completed_suffix_action_count": 0,
                "psf_qp_count_complete": False,
                "psf_qp_postchecks_complete": False,
                "psf_joint_limit_postchecks_complete": False,
                "psf_all_post_state_field_queries_valid_and_positive": False,
                "psf_link56_contact_present": True,
                "psf_any_robot_selected_obstacle_contact_present": True,
                "psf_clearance_certified": False,
                "psf_task_success_ever": False,
                "psf_terminal_task_success": False,
                "psf_task_success_after_material_correction": False,
                "psf_first_task_success_source_action_index": None,
            }
        )
        result = classify_full_episode_feasibility(metrics, self.protocol)
        self.assertEqual(result["classification"], "CONTACT_REMAINS_OR_SHIFTED")
        self.assertTrue(result["apparatus_valid"])

    def test_pairing_failure_is_inconclusive(self):
        metrics = _positive_metrics()
        metrics["exact_paired_start"] = False
        result = classify_full_episode_feasibility(metrics, self.protocol)
        self.assertEqual(result["classification"], "INCONCLUSIVE_APPARATUS")
        self.assertIn("paired_start_not_exact", result["apparatus_failure_reasons"])

    def test_unknown_metric_and_protocol_drift_fail_closed(self):
        metrics = _positive_metrics()
        metrics["unregistered"] = True
        with self.assertRaises(FullEpisodeFeasibilityError):
            classify_full_episode_feasibility(metrics, self.protocol)
        protocol = deepcopy(self.protocol)
        protocol["episode"]["suffix_action_count"] = 56
        with self.assertRaises(FullEpisodeFeasibilityError):
            validate_full_episode_feasibility_protocol(protocol)

    def test_label_order_is_frozen(self):
        self.assertEqual(
            CLASSIFICATIONS,
            (
                "SAFE_TASK_SUCCESS_USEFUL_CORRECTION",
                "CONTACT_PREVENTED_TASK_FAILED",
                "STOP_ONLY",
                "CONTACT_REMAINS_OR_SHIFTED",
                "INCONCLUSIVE_APPARATUS",
            ),
        )


if __name__ == "__main__":
    unittest.main()
