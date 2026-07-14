from __future__ import annotations

import importlib.util
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "summarize_r02", ROOT / "main/summarize_r02.py"
)
assert SPEC is not None and SPEC.loader is not None
SUMMARY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SUMMARY
SPEC.loader.exec_module(SUMMARY)

PARITY_SPEC = importlib.util.spec_from_file_location(
    "sampler_parity_fixture", ROOT / "tests/test_sampler_parity_contract.py"
)
assert PARITY_SPEC is not None and PARITY_SPEC.loader is not None
PARITY_FIXTURE = importlib.util.module_from_spec(PARITY_SPEC)
PARITY_SPEC.loader.exec_module(PARITY_FIXTURE)

MANIFEST = ROOT / "manifests/oracle_h05_colliding.jsonl"
R01_SUMMARY = ROOT / "evidence/r01/r01-summary.json"
CHECKPOINT_ID = "/mnt/data/quanth/cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch"
CHECKPOINT_SHA256 = "b" * 64
RUN_ID = "r02-synthetic-implementation-test"
BASELINE_COMMIT = "57b1aef306f212aea3574b0a3b64aa1a3d8f5e4b"


def manifest_cases() -> list[dict]:
    return [json.loads(line) for line in MANIFEST.read_text(encoding="utf-8").splitlines()]


def parity_artifact() -> dict:
    return PARITY_FIXTURE.valid_parity_artifact(
        pytorch_model_sha256=CHECKPOINT_SHA256
    )


def r02_config(parity_path: Path, parity_sha256: str, r01_raw_root: Path) -> dict:
    return {
        "schema_version": "1.0",
        "name": "r02_oracle_flow_synthetic_implementation_fixture",
        "evidence_tier": SUMMARY.SYNTHETIC_EVIDENCE_TIER,
        "ready_to_run": True,
        "blocked_on": [],
        "manifest": "manifests/oracle_h05_colliding.jsonl",
        "resize_size": 224,
        "settle_steps": 20,
        "executed_prefix": 5,
        "action_horizon": 10,
        "action_dim": 32,
        "sampler_steps": 10,
        "intervention_step": 5,
        "safety_margin_m": 0.005,
        "distance_limit_m": 1.0,
        "eef_radius_m": 0.06,
        "measurement_repeats": 2,
        "stop_after_measurement": False,
        "optimizer_max_iterations": 1,
        "response_matrix_m_per_action": [
            [0.009370281145853376, 0.00014623337157483132, -0.0009171200705674251],
            [0.0000039560765073778535, 0.012040711768393353, 0.0000006816739267718219],
            [-0.002845391485346128, 0.000025633541617775525, 0.011826427878652547],
        ],
        "r02": {
            "phase": "pregrasp_reach",
            "required_arms": list(SUMMARY.RAW_ARMS),
            "clipping_policy": "fail_without_clipping",
            "r01_summary_artifact": "evidence/r01/r01-summary.json",
            "r01_summary_sha256": SUMMARY.file_sha256(R01_SUMMARY),
            "r01_results_root": str(r01_raw_root),
            "sampler_parity_artifact": str(parity_path),
            "sampler_parity_sha256": parity_sha256,
            "target_object": "akita_black_bowl_1",
            "minimum_progress_m": 0.029897349105658888,
            "simulator_safety_margin_m": 0.005,
            "maximum_target_displacement_m": 0.001,
            "maximum_obstacle_displacement_m": 0.001,
            "simulator_repeats": 2,
            "translation_action_bounds": [-1.0, 1.0],
            "normalization_action_scale": [0.8422505, 0.827813, 0.937313],
            "normalization_asset_sha256": "b3a44bb2810436fb62917decaea58bd4d9110255df527dea21e8fd40c960bd84",
            "analytic_samples_per_segment": 26,
            "direction_reference": SUMMARY.DIRECTION_REFERENCE,
            "direction_semantics_decision": SUMMARY.DIRECTION_SEMANTICS_DECISION,
            "direction_semantics_decision_sha256": (
                SUMMARY.DIRECTION_SEMANTICS_DECISION_SHA256
            ),
        },
    }


class R02SummaryFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.r01_raw_root = self.root / "r01-raw"
        self.r01_raw_root.mkdir()
        self.parity_path = self.root / "sampler-parity.json"
        self.parity_path.write_text(json.dumps(parity_artifact()), encoding="utf-8")
        self.config_path = self.root / "r02_oracle_flow.json"
        self.config_path.write_text(
            json.dumps(
                r02_config(
                    self.parity_path,
                    SUMMARY.file_sha256(self.parity_path),
                    self.r01_raw_root,
                )
            ),
            encoding="utf-8",
        )
        self.results_root = self.root / "results"
        base_contract = SUMMARY.load_summary_contract(
            MANIFEST,
            self.config_path,
            R01_SUMMARY,
            run_id=RUN_ID,
            checkpoint_id=CHECKPOINT_ID,
            checkpoint_sha256=CHECKPOINT_SHA256,
            repo_root=ROOT,
        )
        self.cases = manifest_cases()
        self.eligible = list(base_contract.eligible_case_ids)
        self.no_witness = list(base_contract.no_witness_case_ids)
        self.raw_r01: dict[str, dict] = {}
        raw_hashes = {}
        for index, case in enumerate(self.cases):
            case_id = case["case_id"]
            geometry = {
                "start_eef_center_m": [float(index), 0.0, 0.0],
                "branch_obstacle_boxes": [
                    {
                        "name": "synthetic_obstacle",
                        "center_m": [0.1, 0.0, 0.0],
                        "half_size_m": [0.01, 0.01, 0.01],
                    }
                ],
            }
            raw = {
                "schema_version": "1.0",
                "gate": "R01",
                "case_id": case_id,
                "status": (
                    "verified_safe_progress"
                    if case_id in self.eligible
                    else "no_verified_safe_progress"
                ),
                "provenance": {"case_record": dict(case)},
                "nominal": {
                    "branch_snapshot": {"synthetic_case_id": case_id},
                    "actions": [[0.0] * 7 for _ in range(5)],
                    "repeats": [
                        json.loads(json.dumps(geometry)),
                        json.loads(json.dumps(geometry)),
                    ],
                },
            }
            raw_path = self.r01_raw_root / case_id / "endpoint-free-feasibility.json"
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path.write_text(json.dumps(raw), encoding="utf-8")
            raw_hashes[case_id] = SUMMARY.file_sha256(raw_path)
            self.raw_r01[case_id] = raw
        self.contract = replace(
            base_contract,
            r01_results_root=str(self.r01_raw_root),
            r01_result_hashes=raw_hashes,
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def validator(_value: dict) -> list[str]:
        # These artifacts intentionally exercise only the allocation-independent
        # summary. The runner's full validator has its own structural tests.
        return []

    def artifact(self, case: dict, statuses: dict[str, str]) -> dict:
        case_id = case["case_id"]
        eligible = case_id in self.eligible
        arms = {}
        for raw_name in SUMMARY.RAW_ARMS:
            if eligible:
                status = statuses[raw_name]
                passed = True if status == "passed_gate" else False if status == "failed_gate" else None
            elif raw_name == "frozen":
                status = "failed_gate"
                passed = False
            else:
                status = "not_applicable_no_r01_witness"
                passed = None
            arm = {"status": status, "gate": {"passed": passed}}
            if raw_name == "frozen":
                arm["repeats"] = json.loads(
                    json.dumps(self.raw_r01[case_id]["nominal"]["repeats"])
                )
            arms[raw_name] = arm
        raw_nominal = self.raw_r01[case_id]["nominal"]
        return {
            "schema_version": "1.0",
            "artifact_type": "paired_oracle_flow_case",
            "gate": "R02",
            "case_id": case_id,
            "run_id": RUN_ID,
            "status": "completed" if eligible else "not_applicable_no_r01_witness",
            "config_hash": self.contract.config_hash,
            "source_evidence": {
                "r01_summary_sha256": self.contract.r01_summary_sha256,
                "r01_ordered_result_set_digest": self.contract.r01_ordered_result_set_digest,
                "r01_case_sha256": self.contract.r01_result_hashes[case_id],
                "sampler_parity_sha256": self.contract.sampler_parity_sha256,
                "direction_reference": self.contract.direction_reference,
                "direction_semantics_decision": SUMMARY.DIRECTION_SEMANTICS_DECISION,
                "direction_semantics_decision_sha256": (
                    self.contract.direction_semantics_decision_sha256
                ),
            },
            "provenance": {
                "evidence_tier": SUMMARY.SYNTHETIC_EVIDENCE_TIER,
                "git_commit": "d" * 40,
                "git_dirty": False,
                "baseline_commit": BASELINE_COMMIT,
                "host": "synthetic-host",
                "device": "synthetic-cpu",
                "slurm_job_id": "12345",
                "slurm_array_task_id": str(self.cases.index(case)),
                "partition": "synthetic",
                "case_record": dict(case),
                "input_manifest_sha256": self.contract.manifest_sha256,
                **{
                    key: case[key]
                    for key in (
                        "task_suite",
                        "safety_level",
                        "task_index",
                        "episode_index",
                        "group_id",
                        "environment_seed",
                        "policy_seed",
                        "random_control_seed",
                    )
                },
                "checkpoint_id": CHECKPOINT_ID,
                "checkpoint_sha256": CHECKPOINT_SHA256,
                "r01_summary_sha256": self.contract.r01_summary_sha256,
                "r01_case_result_sha256": self.contract.r01_result_hashes[case_id],
                "sampler_parity_sha256": self.contract.sampler_parity_sha256,
                "direction_reference": self.contract.direction_reference,
                "direction_semantics_decision": SUMMARY.DIRECTION_SEMANTICS_DECISION,
                "direction_semantics_decision_sha256": (
                    self.contract.direction_semantics_decision_sha256
                ),
            },
            "pairing": {
                "branch_snapshot": dict(raw_nominal["branch_snapshot"]),
                "r01_branch_snapshot": dict(raw_nominal["branch_snapshot"]),
                "r01_nominal_actions": {
                    "dtype": "float64",
                    "shape": [5, 7],
                    "sha256": "0" * 64,
                    "values": json.loads(json.dumps(raw_nominal["actions"])),
                },
            },
            "directions": {},
            "arms": arms,
            "outcome": {
                "r01_feasible_conditioned": eligible,
                "nominal_collision_reproduced": True,
                "direct_witness_reconfirmed": True if eligible else None,
                "population_mismatch_reason": None,
            },
        }

    def write_population(self, status_for_case) -> None:
        for case in self.cases:
            statuses = status_for_case(case)
            value = self.artifact(case, statuses)
            output = self.results_root / case["case_id"] / SUMMARY.RESULT_FILENAME
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(value), encoding="utf-8")

    def summarize(self) -> dict:
        return SUMMARY.summarize_r02_population(
            MANIFEST,
            self.config_path,
            R01_SUMMARY,
            self.results_root,
            run_id=RUN_ID,
            checkpoint_id=CHECKPOINT_ID,
            checkpoint_sha256=CHECKPOINT_SHA256,
            repo_root=ROOT,
            validator=self.validator,
            r01_validator=self.validator,
            contract_override=self.contract,
            allow_synthetic_implementation_evidence=True,
        )

    def write_authorization_pattern(
        self,
        *,
        random_successes: set[str],
        analytic_successes: set[str],
        oracle_successes: set[str],
    ) -> None:
        def statuses(case: dict) -> dict[str, str]:
            case_id = case["case_id"]
            return {
                "frozen": "failed_gate",
                "direct_witness": "passed_gate",
                "random_residual": (
                    "passed_gate" if case_id in random_successes else "failed_gate"
                ),
                "analytic_geometry_residual": (
                    "passed_gate" if case_id in analytic_successes else "failed_gate"
                ),
                "oracle_residual": (
                    "passed_gate" if case_id in oracle_successes else "failed_gate"
                ),
                "bridge_diagnostic": "failed_gate",
            }

        self.write_population(statuses)


class R02SummaryTest(R02SummaryFixture):
    def test_raw_r01_selected_witness_pointer_is_content_bound(self) -> None:
        raw_pointer = {
            "search": "p_min",
            "candidate_index": 7,
            "source": "pooled_attempt",
            "actions_sha256": "a" * 64,
        }
        raw = {
            "nominal": {
                "branch_snapshot": {"state": 1},
                "actions": [[0.0] * 7 for _ in range(5)],
                "repeats": [
                    {
                        "start_eef_center_m": [0.0, 0.0, 0.0],
                        "branch_obstacle_boxes": [],
                    }
                ],
            },
            "outcome": {"selected_p_min_changed_witness": raw_pointer},
        }
        r02 = {
            "source_evidence": {
                "r01_selected_p_min_changed_witness": dict(raw_pointer),
                "selected_witness_pointer": {
                    **raw_pointer,
                    "actions_array_sha256": "b" * 64,
                },
            },
            "pairing": {
                "branch_snapshot": {"state": 1},
                "r01_branch_snapshot": {"state": 1},
                "r01_nominal_actions": {
                    "values": [[0.0] * 7 for _ in range(5)]
                },
                "historical_r01_diagnostic": {
                    "raw_r01_witness_actions_content_sha256": "a" * 64
                },
            },
            "arms": {"frozen": {"repeats": raw["nominal"]["repeats"]}},
        }
        self.assertEqual(SUMMARY._raw_r01_binding_errors(r02, raw), [])
        r02["pairing"]["historical_r01_diagnostic"][
            "raw_r01_witness_actions_content_sha256"
        ] = "c" * 64
        errors = SUMMARY._raw_r01_binding_errors(r02, raw)
        self.assertIn(
            "historical R01 witness bytes differ from content-bound raw R01",
            errors,
        )

    def test_reports_feasible_and_intent_to_treat_populations_separately(self) -> None:
        def statuses(_case):
            return {
                "frozen": "failed_gate",
                "direct_witness": "passed_gate",
                "random_residual": "failed_gate",
                "analytic_geometry_residual": "failed_gate",
                "oracle_residual": "passed_gate",
                "bridge_diagnostic": "passed_gate",
            }

        self.write_population(statuses)
        summary = self.summarize()

        self.assertTrue(summary["population"]["valid"])
        self.assertTrue(summary["r02_apparatus_passed"])
        self.assertEqual(summary["status"], "implementation_evidence_only")
        self.assertFalse(summary["gate_passed"])
        self.assertTrue(summary["primary"]["criteria_met"])
        feasible = summary["populations"]["feasible_conditioned_17"]["arm_spsr"]
        intent = summary["populations"]["original_20_intent_to_treat"]["arm_spsr"]
        self.assertEqual(feasible["oracle"]["denominator"], 17)
        self.assertEqual(feasible["oracle"]["spsr"], 1.0)
        self.assertEqual(feasible["oracle"]["structural_no_witness_zeroes"], 0)
        self.assertEqual(intent["oracle"]["denominator"], 20)
        self.assertEqual(intent["oracle"]["spsr"], 17 / 20)
        self.assertEqual(intent["oracle"]["structural_no_witness_zeroes"], 3)
        self.assertEqual(intent["frozen"]["structural_no_witness_zeroes"], 0)
        self.assertEqual(
            summary["observed_rollouts"]["arms"]["oracle"]["evaluated_rollouts"],
            17,
        )
        self.assertEqual(
            summary["observed_rollouts"]["arms"]["frozen"]["evaluated_rollouts"],
            20,
        )
        paired = summary["paired_oracle_minus_random"]
        self.assertEqual(paired["point_difference"], 1.0)
        self.assertEqual(paired["bootstrap"]["replicates"], 10_000)
        self.assertEqual(paired["bootstrap"]["group_count"], 17)
        self.assertGreater(paired["bootstrap"]["lower_confidence_bound"], 0.0)

    def test_four_oracle_only_wins_do_not_authorize_probe(self) -> None:
        analytic_successes = set(self.eligible[:5])
        oracle_successes = set(self.eligible[:9])
        self.write_authorization_pattern(
            random_successes=analytic_successes,
            analytic_successes=analytic_successes,
            oracle_successes=oracle_successes,
        )

        summary = self.summarize()

        exact = summary["paired_oracle_minus_analytic"][
            "exact_one_sided_paired_test"
        ]
        self.assertEqual(exact["first_only_wins_b"], 4)
        self.assertEqual(exact["second_only_wins_c"], 0)
        self.assertEqual(exact["p_value"], 0.0625)
        self.assertTrue(summary["primary"]["criteria_met"])
        self.assertFalse(summary["analytic_matched_gate"]["criteria_met"])
        self.assertFalse(
            summary["probe_authorization"][
                "outcome_conditions_met_before_evidence_tier"
            ]
        )
        self.assertFalse(summary["learned_probe_authorized"])

    def test_five_oracle_only_wins_meet_outcome_authorization_conditions(self) -> None:
        analytic_successes = set(self.eligible[:4])
        oracle_successes = set(self.eligible[:9])
        self.write_authorization_pattern(
            random_successes=analytic_successes,
            analytic_successes=analytic_successes,
            oracle_successes=oracle_successes,
        )

        summary = self.summarize()

        exact = summary["paired_oracle_minus_analytic"][
            "exact_one_sided_paired_test"
        ]
        self.assertEqual(exact["first_only_wins_b"], 5)
        self.assertEqual(exact["second_only_wins_c"], 0)
        self.assertEqual(exact["p_value"], 0.03125)
        self.assertGreater(
            summary["paired_oracle_minus_analytic"]["bootstrap"][
                "lower_confidence_bound"
            ],
            0.0,
        )
        self.assertTrue(
            summary["probe_authorization"][
                "outcome_conditions_met_before_evidence_tier"
            ]
        )
        self.assertFalse(summary["learned_probe_authorized"])
        self.assertFalse(summary["scientific_evidence"])

    def test_passing_analytic_matched_gate_blocks_probe_authorization(self) -> None:
        analytic_successes = set(self.eligible[:9])
        oracle_successes = set(self.eligible[:14])
        self.write_authorization_pattern(
            random_successes=set(),
            analytic_successes=analytic_successes,
            oracle_successes=oracle_successes,
        )

        summary = self.summarize()

        exact = summary["paired_oracle_minus_analytic"][
            "exact_one_sided_paired_test"
        ]
        self.assertEqual(exact["p_value"], 0.03125)
        self.assertTrue(summary["primary"]["criteria_met"])
        self.assertTrue(summary["analytic_matched_gate"]["criteria_met"])
        self.assertFalse(
            summary["probe_authorization"][
                "outcome_conditions_met_before_evidence_tier"
            ]
        )
        self.assertFalse(summary["learned_probe_authorized"])

    def test_bounds_and_direction_failures_are_fixed_denominator_failures(self) -> None:
        first_nine = set(self.eligible[:9])
        first_four = set(self.eligible[:4])

        def statuses(case):
            case_id = case["case_id"]
            return {
                "frozen": "failed_gate",
                "direct_witness": "passed_gate",
                "random_residual": "failed_gate",
                "analytic_geometry_residual": (
                    "direction_failure" if case_id in first_four else "failed_gate"
                ),
                "oracle_residual": (
                    "passed_gate" if case_id in first_nine else "bounds_failure"
                ),
                "bridge_diagnostic": "failed_gate",
            }

        self.write_population(statuses)
        summary = self.summarize()

        oracle = summary["populations"]["feasible_conditioned_17"]["arm_spsr"]["oracle"]
        observed = summary["observed_rollouts"]["arms"]["oracle"]
        self.assertEqual(oracle["successes"], 9)
        self.assertEqual(oracle["denominator"], 17)
        self.assertEqual(observed["evaluated_rollouts"], 9)
        self.assertEqual(observed["success_rate"], 1.0)
        self.assertEqual(observed["structural_no_witness_cases_in_denominator"], 0)
        self.assertEqual(summary["failures"]["oracle"]["bounds_failure_count"], 8)
        self.assertEqual(summary["failures"]["analytic"]["direction_failure_count"], 4)
        self.assertTrue(summary["primary"]["criteria_met"])

    def test_missing_extra_hash_and_provenance_mismatches_fail_closed(self) -> None:
        def statuses(_case):
            return {raw_name: "passed_gate" for raw_name in SUMMARY.RAW_ARMS}

        self.write_population(statuses)
        missing_id = self.eligible[0]
        (self.results_root / missing_id / SUMMARY.RESULT_FILENAME).unlink()
        launch_failure = self.results_root / "case-0" / "launch-failure.json"
        launch_failure.parent.mkdir(parents=True, exist_ok=True)
        launch_failure.write_text(
            json.dumps(
                {
                    "status": "failed",
                    "exit_code": 3,
                    "stage": "policy_server_startup",
                    "run_id": RUN_ID,
                    "case_index": 0,
                    "case_id": missing_id,
                    "job_id": "12345",
                    "array_job_id": "12345",
                    "array_task_id": "0",
                    "policy_server_log": "/logs/policy-server.log",
                    "client_log": "/logs/r02-client.log",
                }
            ),
            encoding="utf-8",
        )
        missing_summary = self.summarize()
        self.assertFalse(missing_summary["population"]["valid"])
        self.assertIn(missing_id, missing_summary["population"]["missing_case_ids"])
        failure_audit = missing_summary["population"]["launch_failures"]
        self.assertEqual(failure_audit["found"], 1)
        self.assertEqual(
            failure_audit["missing_final_case_ids_with_launch_failure"], [missing_id]
        )
        self.assertEqual(failure_audit["population_credit"], 0)
        self.assertEqual(failure_audit["valid_records"][0]["exit_code"], 3)

        self.write_population(statuses)
        tampered_id = self.eligible[1]
        path = self.results_root / tampered_id / SUMMARY.RESULT_FILENAME
        value = json.loads(path.read_text(encoding="utf-8"))
        value["config_hash"] = "0" * 64
        value["provenance"]["sampler_parity_sha256"] = "1" * 64
        value["source_evidence"]["direction_reference"] = "historical stale delta"
        value["provenance"]["direction_semantics_decision_sha256"] = "2" * 64
        path.write_text(json.dumps(value), encoding="utf-8")
        tampered_summary = self.summarize()
        self.assertFalse(tampered_summary["population"]["valid"])
        self.assertIn(tampered_id, tampered_summary["population"]["invalid_artifacts"])
        messages = " ".join(tampered_summary["population"]["invalid_artifacts"][tampered_id])
        self.assertIn("config_hash", messages)
        self.assertIn("sampler_parity_sha256", messages)
        self.assertIn("source_evidence.direction_reference", messages)
        self.assertIn("direction_semantics_decision_sha256", messages)

        self.write_population(statuses)
        unexpected = self.artifact(self.cases[0], statuses(self.cases[0]))
        unexpected["case_id"] = "crfs-unexpected"
        output = self.results_root / "crfs-unexpected" / SUMMARY.RESULT_FILENAME
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(unexpected), encoding="utf-8")
        extra_summary = self.summarize()
        self.assertFalse(extra_summary["population"]["valid"])
        self.assertEqual(extra_summary["population"]["unexpected_case_ids"], ["crfs-unexpected"])

    def test_nominal_or_direct_reconfirmation_blocks_interpretation(self) -> None:
        def statuses(_case):
            return {
                "frozen": "failed_gate",
                "direct_witness": "passed_gate",
                "random_residual": "failed_gate",
                "analytic_geometry_residual": "failed_gate",
                "oracle_residual": "passed_gate",
                "bridge_diagnostic": "failed_gate",
            }

        self.write_population(statuses)
        nominal_id = self.eligible[0]
        no_witness_nominal_id = self.no_witness[0]
        direct_id = self.eligible[1]

        for case_id in (nominal_id, no_witness_nominal_id):
            path = self.results_root / case_id / SUMMARY.RESULT_FILENAME
            value = json.loads(path.read_text(encoding="utf-8"))
            value["status"] = "nominal_collision_not_reconfirmed"
            value["outcome"]["nominal_collision_reproduced"] = False
            value["outcome"]["direct_witness_reconfirmed"] = None
            value["outcome"]["population_mismatch_reason"] = value["status"]
            for raw_name in SUMMARY.RAW_ARMS[1:]:
                value["arms"][raw_name]["status"] = SUMMARY.NOT_EVALUATED_STATUS
                value["arms"][raw_name]["gate"]["passed"] = None
            path.write_text(json.dumps(value), encoding="utf-8")

        direct_path = self.results_root / direct_id / SUMMARY.RESULT_FILENAME
        direct_value = json.loads(direct_path.read_text(encoding="utf-8"))
        direct_value["status"] = "direct_witness_not_reconfirmed"
        direct_value["outcome"]["direct_witness_reconfirmed"] = False
        direct_value["outcome"]["population_mismatch_reason"] = direct_value[
            "status"
        ]
        direct_value["arms"]["direct_witness"]["status"] = "failed_gate"
        direct_value["arms"]["direct_witness"]["gate"]["passed"] = False
        for raw_name in SUMMARY.RAW_ARMS[2:]:
            direct_value["arms"][raw_name]["status"] = SUMMARY.NOT_EVALUATED_STATUS
            direct_value["arms"][raw_name]["gate"]["passed"] = None
        direct_path.write_text(json.dumps(direct_value), encoding="utf-8")

        summary = self.summarize()
        self.assertTrue(summary["population"]["valid"])
        self.assertFalse(summary["population"]["paired_baseline_reconfirmed"])
        self.assertEqual(
            summary["population"]["nominal_collision_not_reconfirmed_case_ids"],
            sorted([nominal_id, no_witness_nominal_id]),
        )
        self.assertEqual(
            summary["population"]["direct_witness_not_reconfirmed_case_ids"],
            [direct_id],
        )
        self.assertEqual(summary["status"], "population_mismatch")
        self.assertFalse(summary["gate_passed"])
        self.assertEqual(
            summary["failures"]["oracle"][
                "not_evaluated_after_reconfirmation_failure_count"
            ],
            3,
        )
        self.assertEqual(
            summary["failures"]["direct"][
                "not_evaluated_after_reconfirmation_failure_count"
            ],
            2,
        )
        self.assertEqual(
            summary["observed_rollouts"]["arms"]["frozen"]["evaluated_rollouts"],
            20,
        )
        self.assertEqual(
            summary["observed_rollouts"]["arms"]["direct"]["evaluated_rollouts"],
            16,
        )
        self.assertEqual(
            summary["observed_rollouts"]["arms"]["oracle"]["evaluated_rollouts"],
            15,
        )
        self.assertEqual(
            summary["population"]["terminal_status_counts"],
            {
                "completed": 15,
                "direct_witness_not_reconfirmed": 1,
                "nominal_collision_not_reconfirmed": 2,
                "not_applicable_no_r01_witness": 2,
            },
        )

    def test_raw_r01_action_branch_and_geometry_tampering_is_rejected(self) -> None:
        def statuses(_case):
            return {
                "frozen": "failed_gate",
                "direct_witness": "passed_gate",
                "random_residual": "failed_gate",
                "analytic_geometry_residual": "failed_gate",
                "oracle_residual": "passed_gate",
                "bridge_diagnostic": "failed_gate",
            }

        self.write_population(statuses)
        case_id = self.eligible[0]
        path = self.results_root / case_id / SUMMARY.RESULT_FILENAME
        value = json.loads(path.read_text(encoding="utf-8"))
        value["pairing"]["r01_nominal_actions"]["values"][0][0] = 0.25
        value["pairing"]["r01_branch_snapshot"] = {"forged": True}
        value["arms"]["frozen"]["repeats"][0]["start_eef_center_m"] = [9.0, 9.0, 9.0]
        path.write_text(json.dumps(value), encoding="utf-8")

        summary = self.summarize()
        self.assertFalse(summary["population"]["valid"])
        messages = " ".join(summary["population"]["invalid_artifacts"][case_id])
        self.assertIn("r01_nominal_actions differs", messages)
        self.assertIn("r01_branch_snapshot differs", messages)
        self.assertIn("frozen repeat 0 geometry differs", messages)
        self.assertFalse(summary["gate_passed"])
        self.assertFalse(summary["scientific_evidence"])

    def test_bootstrap_rejects_under_registered_replicate_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "10,000"):
            SUMMARY.grouped_paired_bootstrap_lcb(
                [("group", 1, 0)], replicates=9_999
            )

    def test_cpu_slurm_verifier_is_allocation_backed_and_atomic(self) -> None:
        source = (ROOT / "slurm/r02_summary.sbatch").read_text(encoding="utf-8")
        self.assertIn("SLURM_JOB_ID", source)
        self.assertNotIn("--gres=gpu", source)
        self.assertIn("main/summarize_r02.py", source)
        for flag in (
            "--manifest",
            "--config",
            "--r01-summary",
            "--results-root",
            "--checkpoint-sha256",
            "--output",
        ):
            self.assertIn(flag, source)
        summary_source = (ROOT / "main/summarize_r02.py").read_text(encoding="utf-8")
        self.assertIn("atomic_write_json(args.output, summary)", summary_source)
        self.assertNotIn("feature_list.json", summary_source)


if __name__ == "__main__":
    unittest.main()
