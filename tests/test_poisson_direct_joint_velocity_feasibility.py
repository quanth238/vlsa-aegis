from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import unittest

from main.poisson_fullbody.direct_joint_velocity_feasibility import (
    APPARATUS_METRIC_KEYS,
    BASELINE_ARM,
    CLASSIFICATIONS,
    DirectJointVelocityFeasibilityError,
    METRIC_KEYS,
    PROTECTED_BODY_NAMES,
    PSF_ARM,
    classify_direct_joint_velocity,
    validate_direct_joint_velocity_protocol,
)
from scripts.run_poisson_direct_joint_velocity_feasibility import (
    DirectJointVelocityRunnerError,
    _manifest_case,
    _validate_manifest_case_identity,
)


ROOT = Path(__file__).resolve().parents[1]
E05_PATH = ROOT / "configs/vlsa_poisson_direct_joint_velocity_e05.v1.json"
E42_PATH = ROOT / "configs/vlsa_poisson_direct_joint_velocity_e42.v1.json"
MANIFEST_PATH = ROOT / "manifests/vlsa_poisson_link56_aegis_car_109.v1.jsonl"


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _positive_metrics():
    metrics = {key: True for key in APPARATUS_METRIC_KEYS}
    metrics.update(
        {
            "baseline_link56_contact_present": True,
            "treatment_any_robot_selected_obstacle_contact_present": False,
            "treatment_link56_shifted_external_contact_present": False,
            "treatment_method_stop": False,
            "treatment_stalled_after_correction": False,
            "treatment_native_task_success": True,
            "treatment_task_success_after_correction": True,
            "treatment_paper_car_avoided": True,
            "material_correction_present": True,
            "material_correction_before_baseline_contact": True,
            "maximum_correction_norm_rad_s": 0.1,
            "correction_integral_rad": 0.01,
            "post_correction_measured_joint_motion_integral_rad": 0.2,
            "post_correction_eef_path_length_m": 0.02,
            "post_correction_executed_command_integral_rad": 0.2,
            "post_correction_zero_command_fraction": 0.1,
        }
    )
    assert set(metrics) == set(METRIC_KEYS)
    return metrics


class DirectJointVelocityProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.e05 = _load(E05_PATH)
        cls.e42 = _load(E42_PATH)

    def test_two_frozen_cases_use_one_generic_controller(self):
        e05 = validate_direct_joint_velocity_protocol(self.e05)
        e42 = validate_direct_joint_velocity_protocol(self.e42)

        self.assertEqual(e05["case_id"], "vlsa-t1-goal-ii-t0-e05")
        self.assertEqual(e05["horizon_action_count"], 237)
        self.assertEqual(e05["historical_target_link_body_names"], ("robot0_link5",))
        self.assertEqual(e42["case_id"], "vlsa-t1-goal-ii-t3-e42")
        self.assertEqual(e42["horizon_action_count"], 120)
        self.assertEqual(e42["historical_target_link_body_names"], ("robot0_link6",))

        for derived in (e05, e42):
            self.assertEqual(derived["controller_mode"], "JOINT_VELOCITY")
            self.assertEqual(derived["protected_body_names"], PROTECTED_BODY_NAMES)
            self.assertEqual(
                derived["runtime_protocol_sha256"],
                "396de850fb7f03b9ee038b2fec05f90ef1cb0230c5d40a81a43177162193726a",
            )
            self.assertEqual(
                derived["runtime_semantic_sha256"],
                "2125989269a2ffeeb8d3408d56e4aaa74e1dc5816256d8210bf9ee686c5f5a30",
            )

        for protocol in (self.e05, self.e42):
            self.assertEqual(
                protocol["online_policy"]["checkpoint_receipt_schema_version"],
                "vlsa_table1_pi05_hash_receipt.v1",
            )

        self.assertEqual(
            e05["shared_controller_parameter_sha256"],
            e42["shared_controller_parameter_sha256"],
        )

    def test_case_label_cannot_select_a_different_protected_set(self):
        for protocol in (self.e05, self.e42):
            self.assertEqual(
                protocol["execution"]["protected_robot_body_names"],
                ["robot0_link5", "robot0_link6"],
            )
            self.assertFalse(
                protocol["execution"][
                    "historical_target_link_selects_controller_rows"
                ]
            )
            self.assertEqual(
                protocol["historical_evaluation"]["historical_target_role"],
                "evaluation_only_never_controller_input",
            )

        changed = deepcopy(self.e05)
        changed["execution"]["protected_robot_body_names"] = ["robot0_link5"]
        with self.assertRaises(DirectJointVelocityFeasibilityError):
            validate_direct_joint_velocity_protocol(changed)

        changed = deepcopy(self.e42)
        changed["execution"]["historical_target_link_selects_controller_rows"] = True
        with self.assertRaises(DirectJointVelocityFeasibilityError):
            validate_direct_joint_velocity_protocol(changed)

    def test_both_cases_freeze_the_same_method_parameters(self):
        shared_sections = (
            "pairing",
            "online_policy",
            "execution",
            "cadence",
            "runtime_binding",
            "acceptance",
            "result_contract",
        )
        for section in shared_sections:
            self.assertEqual(
                self.e05[section],
                self.e42[section],
                "%s must not be selected per historical target link" % section,
            )
        self.assertEqual(
            self.e05["pairing"]["arms"], [BASELINE_ARM, PSF_ARM]
        )

    def test_protocol_rejects_return_to_post_osc_torque_shield(self):
        mutations = (
            ("controller_mode", "OSC_POSE"),
            ("decision_variable", "arm_torque_nm"),
            ("post_osc_torque_shield_prohibited", False),
            ("finite_difference_torque_sensitivity_prohibited", False),
            ("hard_cbf_constraints", False),
            ("cbf_slack_enabled", True),
            ("cached_or_zero_command_fallback_prohibited", False),
            ("require_safe_settled_protected_samples", False),
            ("require_joint_positions_inside_registered_margin", False),
            ("zero_joint_velocity_is_qp_feasibility_witness", False),
            ("zero_joint_velocity_is_positive_outcome", True),
        )
        for key, value in mutations:
            with self.subTest(key=key):
                changed = deepcopy(self.e05)
                changed["execution"][key] = value
                with self.assertRaises(DirectJointVelocityFeasibilityError):
                    validate_direct_joint_velocity_protocol(changed)

    def test_protocol_rejects_partial_or_unpaired_exposure(self):
        mutations = (
            (("case", "horizon_action_count"), 236),
            (("pairing", "branch_action_index"), 180),
            (("pairing", "first_live_policy_query_shared_bitwise"), False),
            (("pairing", "own_observation_feedback_after_divergence"), False),
            (("pairing", "historical_action_replay_prohibited"), False),
            (("online_policy", "live_inference_required"), False),
            (("result_contract", "partial_output_policy"), "interpret_if_long"),
            (("result_contract", "independent_consumer_required"), False),
        )
        for path, value in mutations:
            with self.subTest(path=path):
                changed = deepcopy(self.e05)
                changed[path[0]][path[1]] = value
                with self.assertRaises(DirectJointVelocityFeasibilityError):
                    validate_direct_joint_velocity_protocol(changed)

    def test_protocol_rejects_runtime_or_selection_rebinding(self):
        changes = (
            (
                "online_policy",
                "checkpoint_receipt_schema_version",
                "wrong.receipt.schema",
            ),
            (
                "runtime_binding",
                "file_sha256",
                "0" * 64,
            ),
            (
                "runtime_binding",
                "semantic_sha256",
                "0" * 64,
            ),
            (
                "selection_binding",
                "manifest_sha256",
                "0" * 64,
            ),
            (
                "selection_binding",
                "source_manifest_row_sha256",
                "0" * 64,
            ),
        )
        for section, key, value in changes:
            with self.subTest(section=section, key=key):
                changed = deepcopy(self.e42)
                changed[section][key] = value
                with self.assertRaises(DirectJointVelocityFeasibilityError):
                    validate_direct_joint_velocity_protocol(changed)

    def test_real_manifest_schema_matches_both_frozen_case_contracts(self):
        for protocol in (self.e05, self.e42):
            case, observed_row_sha256 = _manifest_case(
                MANIFEST_PATH, protocol["case"]["case_id"]
            )
            _validate_manifest_case_identity(case, protocol["case"])
            self.assertEqual(len(observed_row_sha256), 64)
            self.assertEqual(
                case["historical_aegis_result"]["pairing"][
                    "manifest_row_sha256"
                ],
                protocol["selection_binding"]["source_manifest_row_sha256"],
            )

            changed = deepcopy(protocol["case"])
            changed["policy_noise_schedule_sha256"] = "0" * 64
            with self.assertRaisesRegex(
                DirectJointVelocityRunnerError,
                "manifest historical pairing differs",
            ):
                _validate_manifest_case_identity(case, changed)


class DirectJointVelocityClassificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocols = (_load(E05_PATH), _load(E42_PATH))

    def test_useful_correction_and_task_success_is_the_only_positive(self):
        for protocol in self.protocols:
            result = classify_direct_joint_velocity(_positive_metrics(), protocol)
            self.assertEqual(
                result["classification"],
                "SAFE_TASK_SUCCESS_USEFUL_CORRECTION",
            )
            self.assertTrue(result["feasible"])
            self.assertTrue(result["contact_prevented"])
            self.assertTrue(result["material_correction"])
            self.assertTrue(result["useful_motion"])
            self.assertTrue(result["task_success"])
            self.assertFalse(result["safety_by_stopping"])

    def _classification(self, **changes):
        metrics = _positive_metrics()
        metrics.update(changes)
        return classify_direct_joint_velocity(metrics, self.protocols[0])

    def test_method_stop_is_a_negative_not_safety_success(self):
        result = self._classification(treatment_method_stop=True)
        self.assertEqual(result["classification"], "METHOD_STOP")
        self.assertFalse(result["feasible"])
        self.assertTrue(result["safety_by_stopping"])

    def test_typed_method_stop_survives_expected_partial_treatment_exposure(self):
        result = self._classification(
            treatment_method_stop=True,
            treatment_exposure_complete=False,
            hard_psf_rows_enforced=False,
            joint_velocity_tracking_valid=False,
            no_unregistered_fallback_executed=False,
            every_physics_substep_contact_checked=False,
            paper_car_endpoint_ledger_complete=False,
        )
        self.assertEqual(result["classification"], "METHOD_STOP")
        self.assertTrue(result["typed_terminal_negative"])
        self.assertFalse(result["feasible"])

    def test_typed_contact_survives_expected_contact_terminal(self):
        result = self._classification(
            treatment_any_robot_selected_obstacle_contact_present=True,
            treatment_exposure_complete=False,
            joint_velocity_tracking_valid=False,
            no_unregistered_fallback_executed=False,
            every_physics_substep_contact_checked=False,
            paper_car_endpoint_ledger_complete=False,
        )
        self.assertEqual(
            result["classification"], "CONTACT_REMAINS_OR_SHIFTED"
        )
        self.assertTrue(result["typed_terminal_negative"])
        self.assertFalse(result["feasible"])

    def test_stall_or_near_zero_commands_are_stop_only(self):
        witnesses = (
            {"treatment_stalled_after_correction": True},
            {"post_correction_measured_joint_motion_integral_rad": 0.0},
            {"post_correction_eef_path_length_m": 0.0},
            {"post_correction_executed_command_integral_rad": 0.0},
            {"post_correction_zero_command_fraction": 1.0},
        )
        for changes in witnesses:
            with self.subTest(changes=changes):
                result = self._classification(**changes)
                self.assertEqual(result["classification"], "STOP_ONLY")
                self.assertFalse(result["feasible"])
                self.assertTrue(result["safety_by_stopping"])

    def test_remaining_or_shifted_contact_is_negative(self):
        witnesses = (
            {"treatment_any_robot_selected_obstacle_contact_present": True},
            {"treatment_link56_shifted_external_contact_present": True},
        )
        for changes in witnesses:
            with self.subTest(changes=changes):
                result = self._classification(**changes)
                self.assertEqual(
                    result["classification"], "CONTACT_REMAINS_OR_SHIFTED"
                )
                self.assertFalse(result["feasible"])
                self.assertFalse(result["contact_prevented"])

    def test_no_causal_material_correction_is_negative(self):
        witnesses = (
            {"material_correction_present": False},
            {"material_correction_before_baseline_contact": False},
            {"maximum_correction_norm_rad_s": 0.0},
            {"correction_integral_rad": 0.0},
        )
        for changes in witnesses:
            with self.subTest(changes=changes):
                result = self._classification(**changes)
                self.assertEqual(result["classification"], "NO_MATERIAL_CORRECTION")
                self.assertFalse(result["feasible"])

    def test_task_failure_or_car_failure_is_negative(self):
        task_failure = self._classification(treatment_native_task_success=False)
        self.assertEqual(
            task_failure["classification"], "CONTACT_PREVENTED_TASK_FAILED"
        )
        self.assertFalse(task_failure["feasible"])

        car_failure = self._classification(treatment_paper_car_avoided=False)
        self.assertEqual(car_failure["classification"], "CAR_FAILURE")
        self.assertFalse(car_failure["feasible"])

    def test_causal_claim_requires_matched_baseline_contact(self):
        result = self._classification(baseline_link56_contact_present=False)
        self.assertEqual(
            result["classification"], "BASELINE_CONTACT_NOT_REPRODUCED"
        )
        self.assertFalse(result["feasible"])

    def test_incomplete_apparatus_is_never_interpreted(self):
        result = self._classification(baseline_exposure_complete=False)
        self.assertEqual(result["classification"], "INCONCLUSIVE_APPARATUS")
        self.assertFalse(result["feasible"])
        self.assertFalse(result["pair_complete"])

        tracking = self._classification(joint_velocity_tracking_valid=False)
        self.assertEqual(tracking["classification"], "INCONCLUSIVE_APPARATUS")

    def test_metric_contract_is_typed_and_complete(self):
        missing = _positive_metrics()
        del missing["treatment_method_stop"]
        with self.assertRaises(DirectJointVelocityFeasibilityError):
            classify_direct_joint_velocity(missing, self.protocols[0])

        invalid = _positive_metrics()
        invalid["post_correction_zero_command_fraction"] = 1.1
        with self.assertRaises(DirectJointVelocityFeasibilityError):
            classify_direct_joint_velocity(invalid, self.protocols[0])

        invalid = _positive_metrics()
        invalid["maximum_correction_norm_rad_s"] = float("nan")
        with self.assertRaises(DirectJointVelocityFeasibilityError):
            classify_direct_joint_velocity(invalid, self.protocols[0])

    def test_classification_vocabulary_is_compact_and_frozen(self):
        self.assertEqual(
            CLASSIFICATIONS,
            (
                "SAFE_TASK_SUCCESS_USEFUL_CORRECTION",
                "CONTACT_PREVENTED_TASK_FAILED",
                "CAR_FAILURE",
                "STOP_ONLY",
                "NO_MATERIAL_CORRECTION",
                "METHOD_STOP",
                "CONTACT_REMAINS_OR_SHIFTED",
                "BASELINE_CONTACT_NOT_REPRODUCED",
                "INCONCLUSIVE_APPARATUS",
            ),
        )


if __name__ == "__main__":
    unittest.main()
