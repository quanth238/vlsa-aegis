from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "main/crfs_oracle/r03a_validation.py"
SPEC = importlib.util.spec_from_file_location("r03a_validation_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
VALIDATION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATION)


def array_record(values: list[list[float]], dtype: str = "float64") -> dict:
    rows = len(values)
    columns = len(values[0]) if rows else 0
    digest = hashlib.sha256()
    digest.update(dtype.encode("utf-8"))
    digest.update(str((rows, columns)).encode("utf-8"))
    form = "<d" if dtype == "float64" else "<f"
    for row in values:
        for item in row:
            digest.update(struct.pack(form, item))
    return {
        "dtype": dtype,
        "shape": [rows, columns],
        "sha256": digest.hexdigest(),
        "values": values,
    }


def native_vector_record(values: list, dtype: str, form: str) -> dict:
    digest = hashlib.sha256()
    digest.update(dtype.encode("utf-8"))
    digest.update(str((len(values),)).encode("utf-8"))
    for item in values:
        digest.update(struct.pack("<" + form, item))
    return {
        "dtype": dtype,
        "shape": [len(values)],
        "sha256": digest.hexdigest(),
        "values": values,
    }


def trace_record() -> dict:
    leaves = {
        "active": native_vector_record([False, True], "bool", "?"),
        "step": array_record([[0.0]]),
        "step_index": native_vector_record([5], "int64", "q"),
        "time": native_vector_record([0.5], "float32", "f"),
    }
    return {"sha256": VALIDATION.content_hash(leaves), "leaves": leaves}


def observation_fingerprint() -> dict:
    leaves = [
        {
            "key": "observation/state",
            "kind": "array",
            "dtype": "float32",
            "shape": [1, 2],
            "sha256": "6" * 64,
        },
        {
            "key": "prompt",
            "kind": "string",
            "sha256": hashlib.sha256(b"fixture prompt").hexdigest(),
            "value": "fixture prompt",
        },
    ]
    digest = hashlib.sha256()
    for leaf in leaves:
        framed = json.dumps(leaf, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        digest.update(len(framed).to_bytes(8, "big"))
        digest.update(framed)
    return {"sha256": digest.hexdigest(), "leaves": leaves}


def hash_binding(record: dict, *, digest: str | None = None) -> dict:
    identity = digest or record["sha256"]
    return {
        "source": copy.deepcopy(record),
        "fresh": copy.deepcopy(record),
        "source_sha256": identity,
        "fresh_sha256": identity,
    }


def timing(values: list[float]) -> dict:
    if not values:
        return {"values": [], "p50": None, "p95": None}
    ordered = sorted(values)
    return {
        "values": values,
        "p50": ordered[math.ceil(0.50 * len(ordered)) - 1],
        "p95": ordered[math.ceil(0.95 * len(ordered)) - 1],
    }


def arm(name: str, *, passed: bool = False, terminal: bool = False) -> dict:
    analytic = name != "frozen"
    if terminal and analytic:
        status = VALIDATION.NOT_EVALUATED_STATUS
        full_actions = None
        executed_actions = None
        trace = None
        policy_duplicate_actions = None
        policy_duplicate_trace = None
        policy_seconds = timing([])
        gradient_seconds = timing([])
        policy_replay_exact = None
        replay_exact = None
        repeats = []
        gate = {"passed": None, "reason": "nominal collision not reconfirmed"}
    else:
        status = "passed_gate" if passed else "failed_gate"
        full_actions = array_record([[0.0] * 7 for _ in range(10)])
        executed_actions = array_record([[0.0] * 7 for _ in range(5)])
        trace = trace_record()
        policy_duplicate_actions = copy.deepcopy(full_actions)
        policy_duplicate_trace = copy.deepcopy(trace)
        policy_seconds = timing([0.02, 0.03])
        gradient_seconds = (
            timing(
                [
                    0.001 + index * 0.000001
                    for index in range(2 * VALIDATION.EXPECTED_ACTIVE_STEPS[name])
                ]
            )
            if analytic
            else timing([])
        )
        policy_replay_exact = True
        replay_exact = True
        repeats = [
            {
                "repeat": 0,
                "task_success": False,
                "task_success_during_prefix": False,
            },
            {
                "repeat": 1,
                "task_success": False,
                "task_success_during_prefix": False,
            },
        ]
        gate = {"passed": passed, "trial_checks": []}
    diagnostics = {}
    if analytic:
        diagnostics = {
            "budget_model_l2": 1.0,
            "integrated_field_model_l2": 1.0 if not terminal else 0.0,
            "active_step_count": VALIDATION.EXPECTED_ACTIVE_STEPS[name] if not terminal else 0,
            "applied_step_count": VALIDATION.EXPECTED_ACTIVE_STEPS[name] - 1 if not terminal else 0,
            "margin_stop_step_count": 1 if not terminal else 0,
            "zero_gradient_step_count": 0,
            "nonfinite_gradient_step_count": 0,
            "d_opt_trace": [] if terminal else [{"step": 5}],
            "realized_final_correction": {
                "model_l2": 0.5 if not terminal else 0.0,
                "model_rms": 0.1 if not terminal else 0.0,
                "physical_l2": 0.4 if not terminal else 0.0,
                "physical_rms": 0.08 if not terminal else 0.0,
            },
        }
    return {
        "mechanism": "frozen baseline" if not analytic else "analytic trajectory field",
        "applicable": True,
        "status": status,
        "controls": {
            "intervention_mode": "none" if not analytic else "analytic_trajectory_field",
            "intervention_step": None if not analytic else VALIDATION.INTERVENTION_STEPS[name],
            "clipping_policy": "fail_without_clipping",
        },
        "full_actions": full_actions,
        "executed_actions": executed_actions,
        "bounds": {"checked": not (terminal and analytic), "passed": None if terminal and analytic else True, "clipped": False},
        "trace": trace,
        "trace_sha256": trace["sha256"] if trace is not None else None,
        "timing": {
            "descriptive_only": True,
            "warmed_batch_one": not (terminal and analytic),
            "timer_scope": "client_round_trip_batch_one_infer_including_analytic_field",
            "quantile_method": "inverted_cdf",
            "policy_seconds": policy_seconds,
            "analytic_gradient_seconds": gradient_seconds,
        },
        "diagnostics": diagnostics,
        "policy_replay_exact": policy_replay_exact,
        "policy_duplicate_actions": policy_duplicate_actions,
        "policy_duplicate_trace": policy_duplicate_trace,
        "replay_exact": replay_exact,
        "repeats": repeats,
        "gate": gate,
    }


def bind_arm(value: dict, name: str, replacement: dict) -> None:
    value["arms"][name] = replacement
    value["outcome"]["arm_gate_pass"][name] = replacement["gate"]["passed"]
    value["outcome"]["arm_status"][name] = replacement["status"]


def unrun_failure_arm(name: str, status: str) -> dict:
    value = arm(name, terminal=True)
    value["status"] = status
    value["failure_reason"] = f"fixture {status} before policy reply"
    value["gate"] = {"passed": False, "trial_checks": [], "reason": value["failure_reason"]}
    value["diagnostics"] = None
    return value


def pre_rollout_failure_arm(name: str, status: str, xyz_value: float) -> dict:
    value = arm(name)
    full = copy.deepcopy(value["full_actions"]["values"])
    executed = copy.deepcopy(value["executed_actions"]["values"])
    full[0][0] = xyz_value
    executed[0][0] = xyz_value
    value["full_actions"] = array_record(full)
    value["executed_actions"] = array_record(executed)
    value["policy_duplicate_actions"] = copy.deepcopy(value["full_actions"])
    value["status"] = status
    value["gate"] = {"passed": False, "trial_checks": [], "reason": status}
    value["replay_exact"] = None
    value["repeats"] = []
    return value


def fixture(*, terminal: bool = False) -> dict:
    case = json.loads(
        (ROOT / "manifests/r03a_analytic_kill_test_eligible.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    delta = [[0.0] * 32 for _ in range(10)]
    delta[0][0] = 1.0
    delta_record = array_record(delta)
    arms = {
        name: arm(name, terminal=terminal)
        for name in VALIDATION.EXPECTED_ARMS
    }
    noise = array_record([[0.0] * 32 for _ in range(10)], dtype="float32")
    observation = observation_fingerprint()
    branch_snapshot = {"fixture_state": [1, 2, 3]}
    branch_geometry = {"fixture_geometry": [4, 5, 6]}
    pairing = {
        "passed": True,
        "policy_observation": hash_binding(observation),
        "branch_snapshot": hash_binding(
            branch_snapshot, digest=VALIDATION.content_hash(branch_snapshot)
        ),
        "branch_geometry": hash_binding(
            branch_geometry, digest=VALIDATION.content_hash(branch_geometry)
        ),
        "noise": hash_binding(noise),
        "source_frozen_actions": hash_binding(arms["frozen"]["full_actions"]),
        "source_frozen_trace": hash_binding(arms["frozen"]["trace"]),
        "checks": {key: True for key in VALIDATION.PAIRING_CHECK_KEYS},
    }
    return {
        "schema_version": "1.0",
        "artifact_type": "r03a_analytic_kill_test_case",
        "gate": "R03A",
        "case_id": case["case_id"],
        "run_id": "r03a-analytic-kill-test-fixture",
        "status": "nominal_collision_not_reconfirmed" if terminal else "completed",
        "config_hash": "1" * 64,
        "source_evidence": {
            "r03_summary_path": "evidence/r03/r03-summary.json",
            "r03_summary_sha256": VALIDATION.R03_SUMMARY_SHA256,
            "r03_ordered_result_set_digest": VALIDATION.R03_ORDERED_RESULT_SET_DIGEST,
            "r02_case_path": "/allocation/source/r02-paired.json",
            "r02_case_sha256": "2" * 64,
            "r02_case_validator": "validate_r02_result:passed",
            "r02_config_sha256": VALIDATION.R02_CONFIG_SHA256,
            "decision_artifact": VALIDATION.DECISION_ARTIFACT,
            "decision_sha256": VALIDATION.DECISION_SHA256,
            "eligible_manifest_sha256": VALIDATION.ELIGIBLE_MANIFEST_SHA256,
        },
        "provenance": {
            "evidence_tier": "synthetic_implementation_evidence_only",
            "git_commit": "3" * 40,
            "git_dirty": False,
            "baseline_commit": VALIDATION.BASELINE_COMMIT,
            "host": "fixture",
            "device": "fixture",
            "slurm_job_id": "123",
            "slurm_array_job_id": None,
            "slurm_array_task_id": None,
            "partition": "fixture",
            "case_record": case,
            "input_manifest_sha256": VALIDATION.ELIGIBLE_MANIFEST_SHA256,
            "config_file_sha256": "4" * 64,
            "checkpoint_id": "/fixture/checkpoint",
            "checkpoint_sha256": VALIDATION.CHECKPOINT_SHA256,
            "noise": copy.deepcopy(noise),
        },
        "pairing": pairing,
        "budget": {
            "definition": VALIDATION.BUDGET_DEFINITION,
            "source_direction_key": "delta_star_model",
            "source_delta_star_model": delta_record,
            "source_array_sha256": delta_record["sha256"],
            "first_five_xyz_model_l2": 1.0,
            "source_reported_full_model_l2": 1.0,
        },
        "arms": arms,
        "outcome": {
            "population": "r03_witness_confirmed_development_diagnostic",
            "source_arm_gate_pass": {
                "frozen": False,
                "direct_witness": True,
                "random_residual": False,
                "analytic_geometry_residual": False,
                "oracle_residual": True,
                "bridge_diagnostic": False,
            },
            "fresh_nominal_collision_reproduced": not terminal,
            "arm_gate_pass": {
                name: arms[name]["gate"]["passed"] for name in VALIDATION.EXPECTED_ARMS
            },
            "arm_status": {
                name: arms[name]["status"] for name in VALIDATION.EXPECTED_ARMS
            },
            "privileged_reference_sps_count": 9,
            "privileged_reference_population_size": 17,
            "kill_threshold_sps_count": 9,
            "population_decision": "deferred_to_population_summary",
        },
    }


class R03AValidationTest(unittest.TestCase):
    def test_completed_and_terminal_fixtures_pass_semantics(self) -> None:
        self.assertEqual(VALIDATION.validate_r03a_result(fixture()), [])
        self.assertEqual(VALIDATION.validate_r03a_result(fixture(terminal=True)), [])

    def test_schema_accepts_both_registered_final_statuses(self) -> None:
        try:
            import jsonschema
        except ImportError:
            self.skipTest("jsonschema is optional in the dependency-free local gate")
        schema = json.loads(
            (ROOT / "schemas/r03a-analytic-kill-test.schema.json").read_text(encoding="utf-8")
        )
        jsonschema.Draft202012Validator(schema).validate(fixture())
        jsonschema.Draft202012Validator(schema).validate(fixture(terminal=True))
        for status in VALIDATION.UNRUN_FAILURE_STATUSES:
            value = fixture()
            bind_arm(value, "analytic_trajectory_mid", unrun_failure_arm("analytic_trajectory_mid", status))
            jsonschema.Draft202012Validator(schema).validate(value)

    def test_source_pairing_budget_and_kill_rule_tampering_fail(self) -> None:
        mutations = []
        source = fixture()
        source["source_evidence"]["r03_summary_sha256"] = "9" * 64
        mutations.append(source)
        pairing = fixture()
        pairing["pairing"]["noise"]["fresh_sha256"] = "8" * 64
        mutations.append(pairing)
        check = fixture()
        check["pairing"]["checks"]["geometry_exact_to_source"] = False
        mutations.append(check)
        budget = fixture()
        budget["budget"]["source_delta_star_model"]["values"][0][0] = 0.5
        mutations.append(budget)
        threshold = fixture()
        threshold["outcome"]["kill_threshold_sps_count"] = 10
        mutations.append(threshold)
        decision = fixture()
        decision["outcome"]["population_decision"] = "necessity_rejected"
        mutations.append(decision)
        for value in mutations:
            self.assertTrue(VALIDATION.validate_r03a_result(value))

    def test_pairing_payloads_and_retained_frozen_records_are_fail_closed(self) -> None:
        missing = fixture()
        del missing["pairing"]["noise"]["source"]

        extra = fixture()
        extra["pairing"]["noise"]["unregistered"] = True

        divergent = fixture()
        divergent["pairing"]["branch_snapshot"]["fresh"]["fixture_state"][0] = 9

        spoofed_trace = fixture()
        for side in ("source", "fresh"):
            spoofed_trace["pairing"]["source_frozen_trace"][side]["leaves"][
                "step"
            ]["values"][0][0] = 1.0

        tampered_analytic_trace = fixture()
        tampered_analytic_trace["arms"]["analytic_trajectory_mid"]["trace"][
            "leaves"
        ]["step"]["values"][0][0] = 1.0

        tampered_duplicate_trace = fixture()
        tampered_duplicate_trace["arms"]["analytic_trajectory_mid"][
            "policy_duplicate_trace"
        ]["leaves"]["step"]["values"][0][0] = 1.0

        tampered_duplicate_actions = fixture()
        tampered_duplicate_actions["arms"]["analytic_trajectory_mid"][
            "policy_duplicate_actions"
        ] = array_record([[0.25] + [0.0] * 6] + [[0.0] * 7 for _ in range(9)])

        relabeled_actions = fixture()
        replacement_actions = array_record(
            [[0.25] + [0.0] * 6] + [[0.0] * 7 for _ in range(9)]
        )
        relabeled_actions["pairing"]["source_frozen_actions"] = hash_binding(
            replacement_actions
        )

        relabeled_noise = fixture()
        replacement_noise = [[0.0] * 32 for _ in range(10)]
        replacement_noise[0][0] = 0.25
        relabeled_noise["provenance"]["noise"] = array_record(
            replacement_noise, dtype="float32"
        )

        for value in (
            missing,
            extra,
            divergent,
            spoofed_trace,
            tampered_analytic_trace,
            tampered_duplicate_trace,
            tampered_duplicate_actions,
            relabeled_actions,
            relabeled_noise,
        ):
            with self.subTest(value=value):
                self.assertTrue(VALIDATION.validate_r03a_result(value))

    def test_arm_set_mode_timing_and_outcome_cross_binding_fail_closed(self) -> None:
        wrong_set = fixture()
        wrong_set["arms"]["unregistered"] = wrong_set["arms"].pop(
            "analytic_trajectory_early"
        )
        wrong_mode = fixture()
        wrong_mode["arms"]["analytic_trajectory_mid"]["controls"]["intervention_mode"] = "analytic_trajectory"
        wrong_timing = fixture()
        wrong_timing["arms"]["analytic_trajectory_early"]["timing"]["policy_seconds"]["p95"] = 0.02
        wrong_outcome = fixture()
        wrong_outcome["outcome"]["arm_status"]["analytic_trajectory_mid"] = "passed_gate"
        for value in (wrong_set, wrong_mode, wrong_timing, wrong_outcome):
            self.assertTrue(VALIDATION.validate_r03a_result(value))

    def test_json_object_sorting_does_not_change_arm_identity(self) -> None:
        value = json.loads(json.dumps(fixture(), sort_keys=True))
        self.assertEqual(VALIDATION.validate_r03a_result(value), [])

    def test_explicit_gradient_failures_partition_all_active_steps(self) -> None:
        value = fixture()
        analytic = value["arms"]["analytic_trajectory_mid"]
        analytic["status"] = "zero_gradient_failure"
        analytic["gate"]["passed"] = False
        analytic["diagnostics"]["applied_step_count"] = 4
        analytic["diagnostics"]["margin_stop_step_count"] = 0
        analytic["diagnostics"]["zero_gradient_step_count"] = 1
        value["outcome"]["arm_status"]["analytic_trajectory_mid"] = "zero_gradient_failure"
        value["outcome"]["arm_gate_pass"]["analytic_trajectory_mid"] = False
        self.assertEqual(VALIDATION.validate_r03a_result(value), [])

        value["arms"]["analytic_trajectory_mid"]["diagnostics"][
            "nonfinite_gradient_step_count"
        ] = 1
        self.assertTrue(VALIDATION.validate_r03a_result(value))

    def test_float32_budget_roundoff_is_accepted_but_material_excess_is_not(self) -> None:
        value = fixture()
        value["arms"]["analytic_trajectory_mid"]["diagnostics"][
            "integrated_field_model_l2"
        ] = 1.000000047
        self.assertEqual(VALIDATION.validate_r03a_result(value), [])

        value["arms"]["analytic_trajectory_mid"]["diagnostics"][
            "integrated_field_model_l2"
        ] = 1.00001
        self.assertTrue(VALIDATION.validate_r03a_result(value))

    def test_terminal_artifact_cannot_hide_an_executed_analytic_arm(self) -> None:
        value = fixture(terminal=True)
        value["arms"]["analytic_trajectory_mid"] = arm("analytic_trajectory_mid")
        value["outcome"]["arm_status"]["analytic_trajectory_mid"] = "failed_gate"
        value["outcome"]["arm_gate_pass"]["analytic_trajectory_mid"] = False
        self.assertTrue(VALIDATION.validate_r03a_result(value))

    def test_pre_reply_policy_and_nonfinite_failures_are_explicit_unrun_arms(self) -> None:
        for status in sorted(VALIDATION.UNRUN_FAILURE_STATUSES):
            with self.subTest(status=status):
                value = fixture()
                replacement = unrun_failure_arm("analytic_trajectory_mid", status)
                bind_arm(value, "analytic_trajectory_mid", replacement)
                self.assertEqual(VALIDATION.validate_r03a_result(value), [])

                missing_reason = copy.deepcopy(value)
                del missing_reason["arms"]["analytic_trajectory_mid"]["failure_reason"]
                self.assertTrue(VALIDATION.validate_r03a_result(missing_reason))

                retained_actions = copy.deepcopy(value)
                retained_actions["arms"]["analytic_trajectory_mid"]["full_actions"] = array_record(
                    [[0.0] * 7 for _ in range(10)]
                )
                self.assertTrue(VALIDATION.validate_r03a_result(retained_actions))

                retained_timing = copy.deepcopy(value)
                retained_timing["arms"]["analytic_trajectory_mid"]["timing"][
                    "policy_seconds"
                ] = timing([0.01])
                self.assertTrue(VALIDATION.validate_r03a_result(retained_timing))

    def test_evaluated_failures_cannot_drop_action_trace_or_diagnostics(self) -> None:
        evaluated = (
            pre_rollout_failure_arm("analytic_trajectory_mid", "bounds_failure", 1.1),
            pre_rollout_failure_arm("analytic_trajectory_mid", "saturation_failure", 1.0),
        )
        for replacement in evaluated:
            with self.subTest(status=replacement["status"]):
                value = fixture()
                bind_arm(value, "analytic_trajectory_mid", replacement)
                self.assertEqual(VALIDATION.validate_r03a_result(value), [])
                for field in ("full_actions", "trace", "diagnostics"):
                    broken = copy.deepcopy(value)
                    broken["arms"]["analytic_trajectory_mid"][field] = None
                    self.assertTrue(
                        VALIDATION.validate_r03a_result(broken),
                        msg=f"missing {field} unexpectedly validated",
                    )

    def test_saturation_uses_exact_inclusive_bound_without_epsilon(self) -> None:
        saturation = fixture()
        bind_arm(
            saturation,
            "analytic_trajectory_mid",
            pre_rollout_failure_arm("analytic_trajectory_mid", "saturation_failure", 1.0),
        )
        self.assertEqual(VALIDATION.validate_r03a_result(saturation), [])

        near_bound = fixture()
        candidate = arm("analytic_trajectory_mid")
        full = copy.deepcopy(candidate["full_actions"]["values"])
        executed = copy.deepcopy(candidate["executed_actions"]["values"])
        full[0][0] = 0.999999999
        executed[0][0] = 0.999999999
        candidate["full_actions"] = array_record(full)
        candidate["executed_actions"] = array_record(executed)
        candidate["policy_duplicate_actions"] = copy.deepcopy(
            candidate["full_actions"]
        )
        bind_arm(near_bound, "analytic_trajectory_mid", candidate)
        self.assertEqual(VALIDATION.validate_r03a_result(near_bound), [])

        mislabeled = fixture()
        candidate = pre_rollout_failure_arm(
            "analytic_trajectory_mid", "bounds_failure", 1.0
        )
        bind_arm(mislabeled, "analytic_trajectory_mid", candidate)
        self.assertTrue(VALIDATION.validate_r03a_result(mislabeled))

    def test_strict_out_of_bound_xyz_is_bounds_failure(self) -> None:
        value = fixture()
        replacement = pre_rollout_failure_arm(
            "analytic_trajectory_early", "bounds_failure", -1.000000001
        )
        bind_arm(value, "analytic_trajectory_early", replacement)
        self.assertEqual(VALIDATION.validate_r03a_result(value), [])

        replacement["status"] = "saturation_failure"
        bind_arm(value, "analytic_trajectory_early", replacement)
        self.assertTrue(VALIDATION.validate_r03a_result(value))

    def test_prefix_task_success_is_terminal_failure(self) -> None:
        value = fixture()
        replacement = arm("analytic_trajectory_mid")
        replacement["repeats"][0]["task_success_during_prefix"] = True
        replacement["status"] = "terminal_failure"
        replacement["gate"]["passed"] = False
        bind_arm(value, "analytic_trajectory_mid", replacement)
        self.assertEqual(VALIDATION.validate_r03a_result(value), [])

        hidden_terminal = fixture()
        hidden_terminal["arms"]["analytic_trajectory_mid"]["repeats"][0][
            "task_success_during_prefix"
        ] = True
        self.assertTrue(VALIDATION.validate_r03a_result(hidden_terminal))

        false_terminal = fixture()
        replacement = arm("analytic_trajectory_mid")
        replacement["status"] = "terminal_failure"
        replacement["gate"]["passed"] = False
        bind_arm(false_terminal, "analytic_trajectory_mid", replacement)
        self.assertTrue(VALIDATION.validate_r03a_result(false_terminal))

        missing_monotone_field = fixture()
        del missing_monotone_field["arms"]["analytic_trajectory_mid"]["repeats"][0][
            "task_success_during_prefix"
        ]
        self.assertTrue(VALIDATION.validate_r03a_result(missing_monotone_field))

        impossible_monotone_pair = fixture()
        repeat = impossible_monotone_pair["arms"]["analytic_trajectory_mid"]["repeats"][0]
        repeat["task_success"] = True
        repeat["task_success_during_prefix"] = False
        self.assertTrue(VALIDATION.validate_r03a_result(impossible_monotone_pair))

    def test_budget_failure_is_the_only_status_allowed_materially_over_budget(self) -> None:
        value = fixture()
        replacement = pre_rollout_failure_arm(
            "analytic_trajectory_mid", "budget_failure", 0.0
        )
        replacement["diagnostics"]["integrated_field_model_l2"] = 1.00001
        bind_arm(value, "analytic_trajectory_mid", replacement)
        self.assertEqual(VALIDATION.validate_r03a_result(value), [])

        no_excess = copy.deepcopy(value)
        no_excess["arms"]["analytic_trajectory_mid"]["diagnostics"][
            "integrated_field_model_l2"
        ] = 1.0
        self.assertTrue(VALIDATION.validate_r03a_result(no_excess))

    def test_evaluated_failure_precedence_is_deterministic_under_cooccurrence(self) -> None:
        # Strict out-of-bounds dominates an exact bound on another coordinate.
        value = fixture()
        replacement = pre_rollout_failure_arm(
            "analytic_trajectory_mid", "bounds_failure", 1.1
        )
        full = copy.deepcopy(replacement["full_actions"]["values"])
        executed = copy.deepcopy(replacement["executed_actions"]["values"])
        full[0][1] = 1.0
        executed[0][1] = 1.0
        replacement["full_actions"] = array_record(full)
        replacement["executed_actions"] = array_record(executed)
        replacement["policy_duplicate_actions"] = copy.deepcopy(
            replacement["full_actions"]
        )
        bind_arm(value, "analytic_trajectory_mid", replacement)
        self.assertEqual(VALIDATION.validate_r03a_result(value), [])

        # Exact saturation dominates a material integrated-budget excess.
        value = fixture()
        replacement = pre_rollout_failure_arm(
            "analytic_trajectory_mid", "saturation_failure", 1.0
        )
        replacement["diagnostics"]["integrated_field_model_l2"] = 1.00001
        bind_arm(value, "analytic_trajectory_mid", replacement)
        self.assertEqual(VALIDATION.validate_r03a_result(value), [])
        wrong = copy.deepcopy(value)
        wrong["arms"]["analytic_trajectory_mid"]["status"] = "budget_failure"
        wrong["outcome"]["arm_status"]["analytic_trajectory_mid"] = "budget_failure"
        self.assertTrue(VALIDATION.validate_r03a_result(wrong))

        # Prefix task success dominates a simultaneously observed zero gradient.
        value = fixture()
        replacement = arm("analytic_trajectory_early")
        replacement["repeats"][0]["task_success_during_prefix"] = True
        replacement["diagnostics"]["applied_step_count"] = (
            VALIDATION.EXPECTED_ACTIVE_STEPS["analytic_trajectory_early"] - 1
        )
        replacement["diagnostics"]["margin_stop_step_count"] = 0
        replacement["diagnostics"]["zero_gradient_step_count"] = 1
        replacement["status"] = "terminal_failure"
        replacement["gate"]["passed"] = False
        bind_arm(value, "analytic_trajectory_early", replacement)
        self.assertEqual(VALIDATION.validate_r03a_result(value), [])
        wrong = copy.deepcopy(value)
        wrong["arms"]["analytic_trajectory_early"]["status"] = "zero_gradient_failure"
        wrong["outcome"]["arm_status"]["analytic_trajectory_early"] = "zero_gradient_failure"
        self.assertTrue(VALIDATION.validate_r03a_result(wrong))

    def test_schema_rejects_unrun_evidence_and_evaluated_evidence_omission(self) -> None:
        try:
            import jsonschema
        except ImportError:
            self.skipTest("jsonschema is optional in the dependency-free local gate")
        schema = json.loads(
            (ROOT / "schemas/r03a-analytic-kill-test.schema.json").read_text(encoding="utf-8")
        )
        validator = jsonschema.Draft202012Validator(schema)

        unrun = fixture()
        bind_arm(
            unrun,
            "analytic_trajectory_mid",
            unrun_failure_arm("analytic_trajectory_mid", "policy_failure"),
        )
        del unrun["arms"]["analytic_trajectory_mid"]["failure_reason"]
        self.assertTrue(list(validator.iter_errors(unrun)))

        evaluated = fixture()
        bind_arm(
            evaluated,
            "analytic_trajectory_mid",
            pre_rollout_failure_arm("analytic_trajectory_mid", "saturation_failure", 1.0),
        )
        evaluated["arms"]["analytic_trajectory_mid"]["trace"] = None
        self.assertTrue(list(validator.iter_errors(evaluated)))

    def test_frozen_schema_rejects_a_semantically_coercible_provenance_type(self) -> None:
        try:
            import jsonschema  # noqa: F401
        except ImportError:
            self.skipTest("jsonschema is optional in the dependency-free local gate")
        self.assertEqual(
            VALIDATION.validate_r03a_schema(
                fixture(), require_jsonschema=True
            ),
            [],
        )
        value = fixture()
        value["provenance"]["host"] = 123
        self.assertTrue(
            VALIDATION.validate_r03a_schema(value, require_jsonschema=True)
        )


if __name__ == "__main__":
    unittest.main()
