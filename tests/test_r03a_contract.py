from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiments/r03a_analytic_kill_test.json"
ELIGIBLE_MANIFEST = ROOT / "manifests/r03a_analytic_kill_test_eligible.jsonl"
ORIGINAL_MANIFEST = ROOT / "manifests/oracle_h05_colliding.jsonl"
R03_SUMMARY = ROOT / "evidence/r03/r03-summary.json"
DECISION = ROOT / "docs/decisions/0023-run-strong-analytic-kill-test.md"
ARTIFACT_SCHEMA = ROOT / "schemas/r03a-analytic-kill-test.schema.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class R03AContractTest(unittest.TestCase):
    def test_manifest_is_exact_ordered_r03_eligible_subset(self) -> None:
        original = jsonl(ORIGINAL_MANIFEST)
        eligible = jsonl(ELIGIBLE_MANIFEST)
        summary = json.loads(R03_SUMMARY.read_text(encoding="utf-8"))
        registered_ids = summary["population"]["eligible_case_ids"]

        self.assertEqual(len(eligible), 17)
        self.assertEqual(len({case["case_id"] for case in eligible}), 17)
        self.assertEqual(
            eligible,
            [case for case in original if case["case_id"] in set(registered_ids)],
        )
        self.assertEqual({case["case_id"] for case in eligible}, set(registered_ids))
        self.assertEqual(
            sha256(ELIGIBLE_MANIFEST),
            "241f1b94a5434973dcfb235b43b9386ad15046ff87d35d5b0c34e295c2132916",
        )

    def test_config_binds_every_accepted_source(self) -> None:
        value = json.loads(CONFIG.read_text(encoding="utf-8"))
        settings = value["r03a"]
        self.assertIs(value["ready_to_run"], True)
        self.assertEqual(value["blocked_on"], [])
        self.assertEqual(
            value["manifest"], "manifests/r03a_analytic_kill_test_eligible.jsonl"
        )
        self.assertEqual(settings["eligible_case_count"], 17)
        self.assertEqual(settings["eligible_manifest_sha256"], sha256(ELIGIBLE_MANIFEST))
        self.assertEqual(settings["artifact_schema"], "schemas/r03a-analytic-kill-test.schema.json")
        self.assertEqual(settings["artifact_schema_sha256"], sha256(ARTIFACT_SCHEMA))
        self.assertEqual(settings["source_original_manifest_sha256"], sha256(ORIGINAL_MANIFEST))
        self.assertEqual(settings["source_r02_config_sha256"], sha256(ROOT / settings["source_r02_config"]))
        self.assertEqual(settings["source_r03_summary_sha256"], sha256(R03_SUMMARY))
        self.assertEqual(settings["decision_sha256"], sha256(DECISION))
        summary = json.loads(R03_SUMMARY.read_text(encoding="utf-8"))
        self.assertEqual(
            settings["source_r03_ordered_result_set_digest"],
            summary["ordered_result_set_digest"],
        )
        self.assertEqual(settings["source_checkpoint_sha256"], summary["identities"]["checkpoint_sha256"])

    def test_two_analytic_arms_and_energy_are_frozen_exactly(self) -> None:
        value = json.loads(CONFIG.read_text(encoding="utf-8"))
        settings = value["r03a"]
        self.assertEqual(
            settings["required_arms"],
            ["analytic_trajectory_mid", "analytic_trajectory_early"],
        )
        self.assertEqual(
            settings["intervention_steps"],
            {"analytic_trajectory_mid": 5, "analytic_trajectory_early": 1},
        )
        self.assertEqual(
            settings["expected_active_steps"],
            {"analytic_trajectory_mid": 5, "analytic_trajectory_early": 9},
        )
        self.assertEqual(settings["policy_intervention_mode"], "analytic_trajectory_field")
        self.assertEqual(settings["early_step_semantics"], "begin_after_first_ordinary_euler_update")
        self.assertEqual(settings["approximate_clean_definition"], "x_t - t * stopgrad(v_base)")
        self.assertEqual(settings["energy_definition"], "sum_i softplus((m - d_i) / tau)^2")
        self.assertEqual(settings["energy_margin_m"], 0.005)
        self.assertEqual(settings["energy_temperature_m"], 0.005)
        self.assertEqual(settings["samples_per_segment"], 26)
        self.assertEqual(settings["gradient_jacobian"], "identity_approximate_clean")
        self.assertEqual(settings["gradient_mask"], "first_five_xyz_only")
        self.assertEqual(
            settings["margin_stop"],
            "apply_no_correction_when_hard_min_predicted_clearance_at_least_margin",
        )
        self.assertEqual(
            settings["budget_source"],
            "r02.directions.delta_star_model:first_five_xyz_l2",
        )
        self.assertEqual(settings["budget_schedule"], "B / T_active reverse_time_velocity_magnitude")
        self.assertEqual(settings["clipping_policy"], "fail_without_clipping")
        self.assertEqual(
            settings["saturation_criterion"],
            "any_executed_first_five_xyz_exactly_equals_registered_inclusive_bound",
        )
        self.assertEqual(
            settings["saturation_comparison"],
            "exact_float_equality_no_epsilon_no_clipping",
        )
        self.assertEqual(
            settings["terminal_failure_criterion"],
            "any_simulator_task_success_during_prefix_is_true",
        )
        self.assertEqual(
            settings["evaluated_failure_precedence"],
            [
                "bounds_failure",
                "saturation_failure",
                "budget_failure",
                "terminal_failure",
                "zero_gradient_failure",
                "ordinary_gate",
            ],
        )

    def test_kill_rule_never_authorizes_probe_or_selects_an_arm(self) -> None:
        value = json.loads(CONFIG.read_text(encoding="utf-8"))
        settings = value["r03a"]
        self.assertEqual(settings["kill_threshold_privileged_sps_count"], 9)
        self.assertIs(settings["both_arms_mandatory_report"], True)
        self.assertEqual(settings["selection_between_arms"], "forbidden")
        self.assertIs(settings["probe_training_authorized"], False)
        self.assertIs(settings["confirmatory_r04_unblocked_by_r03a_alone"], False)
        self.assertIs(value["allow_test_tuning"], False)

    def test_latency_protocol_is_descriptive_and_complete(self) -> None:
        latency = json.loads(CONFIG.read_text(encoding="utf-8"))["r03a"]["latency_protocol"]
        self.assertEqual(latency["mode"], "warmed_batch_one")
        self.assertEqual(latency["warmup_policy_calls_before_measurement"], 2)
        self.assertEqual(latency["measured_policy_calls_per_arm_per_case"], 2)
        self.assertEqual(
            latency["timer_scope"],
            "client_round_trip_batch_one_infer_including_analytic_field",
        )
        self.assertEqual(latency["timer_unit"], "seconds")
        self.assertEqual(latency["quantile_method"], "inverted_cdf")
        self.assertEqual(latency["policy_quantiles"], [0.5, 0.95])
        self.assertIs(latency["report_analytic_gradient_time"], True)
        self.assertEqual(latency["analytic_gradient_timing_scope"], "per_active_step")
        self.assertIs(latency["exclude_training_and_direct_search"], True)

    def test_population_contract_loader_accepts_only_the_frozen_sources(self) -> None:
        from main import summarize_r03a

        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        contract = summarize_r03a.load_summary_contract(
            ELIGIBLE_MANIFEST,
            CONFIG,
            R03_SUMMARY,
            r02_results_root=config["r03a"]["source_r02_results_root"],
            run_id="r03a-contract-fixture",
            checkpoint_id="/fixture/checkpoint",
            checkpoint_sha256=config["r03a"]["source_checkpoint_sha256"],
            expected_git_commit="1" * 40,
        )
        self.assertEqual(len(contract.manifest_cases), 17)
        self.assertEqual(contract.manifest_sha256, sha256(ELIGIBLE_MANIFEST))
        self.assertEqual(contract.config_file_sha256, sha256(CONFIG))
        self.assertEqual(contract.expected_git_commit, "1" * 40)


if __name__ == "__main__":
    unittest.main()
