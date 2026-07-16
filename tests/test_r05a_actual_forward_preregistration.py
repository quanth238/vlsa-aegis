from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "experiments" / "r05a_actual_forward_canary.json"
ADR_PATH = (
    ROOT
    / "docs"
    / "decisions"
    / "0050-preregister-actual-forward-cem-teacher-canary.md"
)


class ActualForwardPreregistrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    def test_source_target_and_budget_are_unchanged(self) -> None:
        source = self.config["frozen_source_bindings"]
        case = self.config["frozen_case"]
        target = self.config["target_contract"]
        self.assertEqual(case["case_id"], "crfs-1069f29a8d76463a")
        self.assertEqual(case["environment_seed"], 924805038)
        self.assertEqual(case["policy_seed"], 1179198633)
        self.assertEqual(
            source["source_r02"]["sha256"],
            "055fcf18781071c6c3575b42a1b44c32a76b474ac1911ad8c5443f78b9c42593",
        )
        self.assertEqual(target["source_budget_float32"], 3.6398398876190186)
        self.assertFalse(target["sum_of_active_increments_must_equal_target_delta"])
        self.assertEqual(
            target["normalization"],
            "physical_displacement_uses_checkpoint_scale_only_never_subtracts_normalization_mean",
        )

    def test_actual_forward_path_has_no_retired_derivative_interface(self) -> None:
        flow = self.config["flow_contract"]
        self.assertEqual(flow["compact_shape"], [5, 15])
        self.assertEqual(flow["compact_dimension"], 75)
        self.assertEqual(flow["active_flow_steps"], [5, 6, 7, 8, 9])
        self.assertTrue(flow["ordinary_residual_schedule_path_only"])
        self.assertFalse(flow["autograd_allowed"])
        self.assertFalse(flow["inverse_flow_teacher_allowed"])
        self.assertFalse(flow["terminal_action_overwrite_allowed"])
        self.assertEqual(flow["dt_float32"], -0.1)
        self.assertEqual(flow["transported_request_variable"], "u32=float32(c32/dt32)")
        self.assertEqual(flow["constraint_validation"]["slack_float32_ulps"], 8)
        self.assertEqual(flow["constraint_validation"]["norm_dtype"], "float32")
        adr = ADR_PATH.read_text(encoding="utf-8")
        self.assertIn("ordinary `residual_schedule`", adr)
        self.assertIn("must not call `inverse_flow_teacher`", adr)

    def test_cem_and_query_budget_are_exact(self) -> None:
        cem = self.config["cem"]
        ledger = self.config["request_ledger"]
        self.assertEqual(cem["search_seed"], 20260716)
        self.assertEqual(cem["generations"], 8)
        self.assertEqual(cem["population_size"], 65)
        self.assertEqual(cem["antithetic_pairs_per_generation"], 32)
        self.assertEqual(cem["elite_count"], 13)
        self.assertEqual(cem["search_rollouts"], 8 * 65)
        self.assertFalse(cem["early_stopping"])
        self.assertTrue(cem["raw_standard_normals_persisted"])
        self.assertIn("mean_g_plus_sigma_g_times_z", cem["candidate_formula"])
        self.assertIn("exact_zero_if_norm_zero", cem["projection_formula"])
        self.assertIn("ddof_0", cem["variance_update"])
        phases = (
            ledger["pre_search_paired_requests"]
            + ledger["equal_split_repeatability_requests"]
            + ledger["cem_search_requests"]
            + ledger["selected_B_replay_requests"]
            + ledger["reversed_B_replay_requests"]
            + ledger["post_search_paired_requests"]
        )
        self.assertEqual(phases, ledger["complete_run_exact_policy_request_count"])
        self.assertEqual(phases, 534)
        ordered = ledger["ordered_phases"]
        self.assertEqual(
            [item["name"] for item in ordered],
            [
                "compiled_frozen_pre",
                "eager_source_trace_pre",
                "eager_normalized_final_pre",
                "zero_schedule_pre",
                "arm_a_equal_split",
                "cem_search",
                "arm_b_replay",
                "arm_c_reverse_replay",
                "zero_schedule_post",
                "eager_normalized_final_post",
                "eager_source_trace_post",
                "compiled_frozen_post",
            ],
        )
        self.assertEqual(ordered[0], {"name": "compiled_frozen_pre", "start": 0, "stop_inclusive": 0})
        self.assertEqual(ordered[-1], {"name": "compiled_frozen_post", "start": 533, "stop_inclusive": 533})
        self.assertTrue(ledger["caught_apparatus_fault_must_issue_no_later_requests"])
        self.assertTrue(
            ledger["hard_failure_may_preserve_wrapper_evidence_without_partial_ledger"]
        )

    def test_selection_preserves_lowest_energy_feasible_rule(self) -> None:
        selection = self.config["objective_and_selection"]
        self.assertEqual(
            selection["feasible_selection"],
            ["minimum_executed_increment_energy", "objective", "global_query_index"],
        )
        self.assertEqual(
            selection["finite_miss_selection"],
            ["objective", "minimum_executed_increment_energy", "global_query_index"],
        )
        self.assertEqual(
            selection["fidelity_limits"],
            {
                "xyz_max_abs": 0.01,
                "xyz_rms": 0.005,
                "full_max_abs": 0.05,
                "full_rms": 0.015,
            },
        )
        self.assertEqual(
            selection["objective"],
            "float64_mean_concat(error_xyz_div_0.005,error_other_div_0.015)_squared_over_35",
        )
        self.assertEqual(selection["rms_denominators"], {"xyz": 15, "full": 35})
        self.assertIn("one_canonical_A", selection["final_pool"])
        self.assertEqual(selection["energy"], "sum_float64_executed_increment_squared_over_75")
        self.assertIn("schedule_bytes_both_differ", selection["B_changed"])

    def test_arm_a_reverse_and_outcome_precedence_are_executable(self) -> None:
        arms = self.config["arms"]
        self.assertEqual(
            arms["A_equal_split"],
            "cA32=float32(Delta32/float32(5))_then_u32=float32(cA32/dt32)_no_adam",
        )
        self.assertIn("requested_float32_velocity_rows_5_through_9", arms["C_reverse_B"])
        self.assertIn("never_reconvert", arms["C_reverse_B"])
        self.assertEqual(
            self.config["outcome_contract"]["precedence"],
            [
                "any_apparatus_fault_is_apparatus_inconclusive",
                "complete_A_pass_is_baseline_sufficient_no_incremental_support",
                "complete_A_fail_changed_B_pass_is_mechanism_pass",
                "other_complete_finite_result_is_frozen_cem_negative",
            ],
        )

    def test_config_is_fail_closed_or_one_exact_release_and_claims_are_narrow(self) -> None:
        release = self.config["execution_release"]
        if self.config["ready_to_run"] is False:
            self.assertEqual(
                self.config["config_status"], "draft_preregistered_not_released"
            )
            self.assertTrue(self.config["blocked_on"])
            self.assertIsNone(release)
            self.assertFalse(
                self.config["preregistration"]["h100_submission_authorized"]
            )
        else:
            self.assertIs(self.config["ready_to_run"], True)
            self.assertEqual(
                self.config["config_status"], "released_exact_single_canary"
            )
            self.assertEqual(self.config["blocked_on"], [])
            self.assertTrue(
                self.config["preregistration"]["h100_submission_authorized"]
            )
            self.assertIsInstance(release, dict)
            self.assertEqual(
                release["artifact_role"],
                "r05a_actual_forward_canary_execution_release",
            )
            self.assertRegex(
                release["accepted_implementation_commit"], r"^[0-9a-f]{40}$"
            )
            self.assertRegex(release["run_id"], r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
            self.assertTrue(release["single_submission"])
            self.assertEqual(release["source_host"], "worker-1")
            self.assertTrue(release["release_only_parent_required"])
            self.assertFalse(release["automatic_resubmission_allowed"])
            self.assertFalse(release["automatic_next_experiment_allowed"])
            self.assertEqual(
                release["resources"],
                {
                    "partition": "main",
                    "account": "normal",
                    "qos": "normal",
                    "gpus": 1,
                    "cpus_per_task": 8,
                    "host_memory_mib": 65536,
                    "time_limit": "02:00:00",
                    "array": "0-0%1",
                    "requeue": False,
                    "validator_partition": "main",
                    "validator_account": "normal",
                    "validator_qos": "normal",
                    "validator_cpus": 2,
                    "validator_host_memory_mib": 8192,
                    "validator_time_limit": "00:15:00",
                    "validator_gpus": 0,
                    "validator_dependency": "afterany",
                },
            )
        boundary = self.config["execution_boundary"]
        self.assertEqual(boundary["policy_generated_action_steps_executed"], 0)
        self.assertEqual(boundary["teacher_generated_action_steps_executed"], 0)
        self.assertFalse(boundary["simulator_efficacy_evaluated"])
        self.assertFalse(boundary["probe_or_mlp_training_authorized"])
        outcomes = self.config["outcome_contract"]
        self.assertFalse(outcomes["infeasibility_claim_allowed"])
        self.assertFalse(outcomes["mechanism_pass_automatically_launches_ift01"])

    def test_bound_local_files_match_registered_hashes(self) -> None:
        bindings = self.config["frozen_source_bindings"]
        for key in ("manifest", "source_r02_config"):
            record = bindings[key]
            data = (ROOT / record["path"]).read_bytes()
            self.assertEqual(hashlib.sha256(data).hexdigest(), record["sha256"])
        artifact = self.config["artifact_contract"]
        schema = (ROOT / artifact["envelope_schema_path"]).read_bytes()
        self.assertEqual(
            hashlib.sha256(schema).hexdigest(), artifact["envelope_schema_sha256"]
        )
        receipt_schema = (
            ROOT / artifact["publication_receipt_schema_path"]
        ).read_bytes()
        self.assertEqual(
            hashlib.sha256(receipt_schema).hexdigest(),
            artifact["publication_receipt_schema_sha256"],
        )
        self.assertEqual(
            artifact["publication_receipt_name"], "cpu-afterany-validation.json"
        )
        self.assertFalse(artifact["gpu_may_publish_results_json"])
        self.assertTrue(artifact["cpu_afterany_is_sole_publisher"])


if __name__ == "__main__":
    unittest.main()
