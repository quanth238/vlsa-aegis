import copy
from pathlib import Path
import sys
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis import validate_aegis_action_invariant_canary as canary


class AegisActionInvariantCanaryTests(unittest.TestCase):
    def _result(self, *, policy_mode):
        arm = canary.EXPECTED_ARM_BY_POLICY_MODE[policy_mode]
        is_aegis = policy_mode == "aegis"
        return {
            "case_id": "frozen-canary-case",
            "arm": arm,
            "mode": policy_mode,
            "protocol_id": "paired-protocol",
            "status": "complete",
            "terminal_reason": (
                "task_success" if is_aegis else "time_limit"
            ),
            "task_success": is_aegis,
            "metrics": {
                "minimum_clearance_m": 0.012,
                "task_progress": 0.75,
                "paper_collision": not is_aegis,
                "paper_collision_avoidance": is_aegis,
                "collision_first_step": None if is_aegis else 0,
                "executed_action_count": 1,
                "legacy_ets_steps": 1,
            },
            "actions": [
                {
                    "step": 0,
                    "nominal_raw": [
                        0.1,
                        0.2,
                        0.3,
                        0.4,
                        0.5,
                        0.6,
                        -1.0,
                    ],
                    "nominal_translational": [
                        0.1,
                        0.2,
                        0.3,
                        0.0,
                        0.0,
                        0.0,
                        -1.0,
                    ],
                    "executed": [
                        0.1,
                        0.2,
                        0.3,
                        0.0,
                        0.0,
                        0.0,
                        -1.0,
                    ],
                    "env_step_input": [
                        0.1,
                        0.2,
                        0.3,
                        0.0,
                        0.0,
                        0.0,
                        -1.0,
                    ],
                    "post_step_controller_proxy": {"q": [1.0] * 7},
                    "step_elapsed_seconds": 0.01,
                    "qp": {
                        "status": "not_requested",
                        "context": {"observer_only": "off"},
                        "z_before": [0.0, 0.0, 0.0],
                        "z_after": [0.1, 0.2, 0.3],
                    },
                }
            ],
            "policy_queries": [
                {
                    "query_index": 0,
                    "returned_actions_sha256": "a" * 64,
                    "elapsed_seconds": 0.02,
                    "server_timing": {"inference_seconds": 0.01},
                }
            ],
            "contact_telemetry": {
                "physical_contact": False,
                "detailed_artifact": {"path": "contacts.json.gz"},
            },
            "method_failure": {
                "status": "not_applicable",
                "diagnostics": {"observer_only": "off"},
                "diagnostics_payload_sha256": "b" * 64,
            },
            "video": {
                "path": f"{policy_mode}.mp4",
                "fps": 30,
                "sha256": "c" * 64,
            },
            "timing": {"elapsed_seconds": 1.0},
            "result_payload_sha256": "d" * 64,
            "action_invariance_ledger": {"sha256": "e" * 64},
        }

    def _expected_outcome(self, *, policy_mode):
        result = self._result(policy_mode=policy_mode)
        return {
            "status": result["status"],
            "terminal_reason": result["terminal_reason"],
            "task_success": result["task_success"],
            "paper_collision": result["metrics"]["paper_collision"],
            "paper_collision_avoidance": (
                result["metrics"]["paper_collision_avoidance"]
            ),
            "collision_first_step": (
                result["metrics"]["collision_first_step"]
            ),
            "executed_action_count": (
                result["metrics"]["executed_action_count"]
            ),
            "legacy_ets_steps": result["metrics"]["legacy_ets_steps"],
            "policy_query_count": len(result["policy_queries"]),
        }

    def _reference(self):
        return {
            "case_id": "frozen-canary-case",
            "arms": {
                canary.EXPECTED_ARM_BY_POLICY_MODE[policy_mode]: {
                    "expected_outcome": self._expected_outcome(
                        policy_mode=policy_mode
                    )
                }
                for policy_mode in canary.POLICY_MODES
            },
        }

    def _observational_pair(self, *, policy_mode):
        off = self._result(policy_mode=policy_mode)
        on = copy.deepcopy(off)
        on["failure_diagnostics"] = {
            "enabled": True,
            "terminal_frame": {"path": "terminal.npy"},
        }
        on["actions"][0]["qp"].update(
            {
                "status": "solved",
                "context": {"observer_only": "on"},
                "z_before": [9.0, 8.0, 7.0],
            }
        )
        on["actions"][0]["env_step_input"] = [
            9.0,
            8.0,
            7.0,
            0.0,
            0.0,
            0.0,
            -1.0,
        ]
        on["actions"][0]["post_step_controller_proxy"] = {
            "q": [2.0] * 7
        }
        on["actions"][0]["step_elapsed_seconds"] = 9.9
        on["policy_queries"][0]["elapsed_seconds"] = 8.8
        on["policy_queries"][0]["server_timing"] = {
            "inference_seconds": 7.7
        }
        on["contact_telemetry"]["detailed_artifact"] = {
            "path": "contacts-on.json.gz"
        }
        on["method_failure"]["diagnostics"] = {"observer_only": "on"}
        on["method_failure"]["diagnostics_payload_sha256"] = "f" * 64
        on["video"]["sha256"] = "0" * 64
        on["timing"] = {"elapsed_seconds": 99.0}
        on["result_payload_sha256"] = "1" * 64
        on["action_invariance_ledger"] = {"sha256": "2" * 64}
        return off, on

    def _four_results(self):
        results = {}
        for policy_mode in canary.POLICY_MODES:
            off, on = self._observational_pair(policy_mode=policy_mode)
            results[("diagnostics-off", policy_mode)] = off
            results[("diagnostics-on", policy_mode)] = on
        return results

    def test_observational_pair_accepts_diagnostic_only_differences(self):
        off, on = self._observational_pair(policy_mode="aegis")
        action_receipt = {
            "status": "action_invariant",
            "action_ledger": {"sha256": "3" * 64},
        }
        with mock.patch.object(
            canary.failure_validation,
            "validate_action_invariance_pair",
            return_value=action_receipt,
        ) as action_validator:
            receipt = canary.validate_observational_pair(off, on)

        action_validator.assert_called_once_with(off, on)
        self.assertEqual(receipt["status"], "observationally_invariant")
        self.assertEqual(receipt["action_invariance"], action_receipt)
        self.assertEqual(receipt["arm"], on["arm"])

    def test_observational_pair_rejects_scientific_changes(self):
        mutators = {
            "outcome": lambda result: result.__setitem__(
                "task_success", False
            ),
            "status": lambda result: result.__setitem__(
                "status", "method_failure"
            ),
            "metric": lambda result: result["metrics"].__setitem__(
                "minimum_clearance_m", -0.001
            ),
            "action": lambda result: result["actions"][0].__setitem__(
                "executed", [0.0] * 6 + [-1.0]
            ),
        }
        for label, mutate in mutators.items():
            with self.subTest(change=label):
                off, on = self._observational_pair(policy_mode="aegis")
                mutate(on)
                with mock.patch.object(
                    canary.failure_validation,
                    "validate_action_invariance_pair",
                    return_value={"status": "action_invariant"},
                ):
                    with self.assertRaisesRegex(
                        canary.ActionInvariantCanaryError,
                        "diagnostics changed outcome/state semantics",
                    ):
                        canary.validate_observational_pair(off, on)

    def test_exact_four_run_inventory_validates(self):
        results = self._four_results()
        output_roots = {
            "diagnostics-off": Path("/tmp/canary/off"),
            "diagnostics-on": Path("/tmp/canary/on"),
        }
        reference_path = Path("/tmp/frozen-reference.json")
        reference = self._reference()

        with mock.patch.object(
            canary.diagnostics,
            "sha256_path",
            return_value=canary.REFERENCE_SHA256,
        ), mock.patch.object(
            canary.failure_validation,
            "_load_json",
            return_value=reference,
        ), mock.patch.object(
            canary.failure_validation,
            "validate_action_invariance_pair",
            return_value={"status": "action_invariant"},
        ) as action_validator, mock.patch.object(
            canary.failure_validation,
            "validate_canary_reference",
            return_value={"status": "reference_valid"},
        ) as reference_validator, mock.patch.object(
            canary.failure_validation,
            "validate_diagnostic_result",
            return_value={"status": "diagnostics_valid"},
        ) as diagnostic_validator, mock.patch.object(
            canary,
            "_decode_off_video",
            return_value={"status": "video_valid"},
        ) as video_validator:
            receipt = canary.validate_four_run_canary(
                results=results,
                output_roots=output_roots,
                reference_path=reference_path,
            )

        self.assertEqual(
            set(results),
            {
                ("diagnostics-off", "pi05"),
                ("diagnostics-off", "aegis"),
                ("diagnostics-on", "pi05"),
                ("diagnostics-on", "aegis"),
            },
        )
        self.assertEqual(
            receipt["schema_version"],
            canary.CANARY_EVIDENCE_SCHEMA,
        )
        self.assertEqual(receipt["status"], "validated")
        self.assertIs(receipt["scientific_result"], False)
        self.assertEqual(receipt["case_id"], "frozen-canary-case")
        self.assertEqual(
            set(receipt["reference"]["ledgers"]),
            {
                "diagnostics-off/pi05",
                "diagnostics-on/pi05",
                "diagnostics-off/aegis",
                "diagnostics-on/aegis",
            },
        )
        self.assertEqual(
            set(receipt["reference"]["historical_outcomes"]),
            {
                "diagnostics-off/pi05",
                "diagnostics-on/pi05",
                "diagnostics-off/aegis",
                "diagnostics-on/aegis",
            },
        )
        self.assertEqual(action_validator.call_count, 2)
        self.assertEqual(reference_validator.call_count, 4)
        self.assertEqual(diagnostic_validator.call_count, 2)
        self.assertEqual(video_validator.call_count, 2)
        require_ready_values = {
            call.kwargs["require_ready_geometry"]
            for call in diagnostic_validator.call_args_list
        }
        self.assertEqual(require_ready_values, {False, True})

    def test_reference_hash_mismatch_is_rejected(self):
        with mock.patch.object(
            canary.diagnostics,
            "sha256_path",
            return_value="0" * 64,
        ):
            with self.assertRaisesRegex(
                canary.ActionInvariantCanaryError,
                "frozen canary action reference hash changed",
            ):
                canary.validate_four_run_canary(
                    results=self._four_results(),
                    output_roots={
                        "diagnostics-off": Path("/tmp/canary/off"),
                        "diagnostics-on": Path("/tmp/canary/on"),
                    },
                    reference_path=Path("/tmp/frozen-reference.json"),
                )

    def test_missing_result_key_is_rejected(self):
        results = self._four_results()
        del results[("diagnostics-on", "aegis")]
        with self.assertRaisesRegex(
            canary.ActionInvariantCanaryError,
            "four-run canary result inventory changed",
        ):
            canary.validate_four_run_canary(
                results=results,
                output_roots={
                    "diagnostics-off": Path("/tmp/canary/off"),
                    "diagnostics-on": Path("/tmp/canary/on"),
                },
                reference_path=Path("/tmp/frozen-reference.json"),
            )

    def test_result_mode_arm_mapping_mutations_are_rejected(self):
        mutations = {
            "mode": lambda result: result.__setitem__("mode", "aegis"),
            "arm": lambda result: result.__setitem__(
                "arm",
                canary.EXPECTED_ARM_BY_POLICY_MODE["aegis"],
            ),
        }
        for label, mutate in mutations.items():
            with self.subTest(change=label):
                results = self._four_results()
                mutate(results[("diagnostics-off", "pi05")])
                with mock.patch.object(
                    canary.diagnostics,
                    "sha256_path",
                    return_value=canary.REFERENCE_SHA256,
                ), mock.patch.object(
                    canary.failure_validation,
                    "_load_json",
                    return_value=self._reference(),
                ):
                    with self.assertRaisesRegex(
                        canary.ActionInvariantCanaryError,
                        "result mode/arm mapping changed",
                    ):
                        canary.validate_four_run_canary(
                            results=results,
                            output_roots={
                                "diagnostics-off": Path("/tmp/canary/off"),
                                "diagnostics-on": Path("/tmp/canary/on"),
                            },
                            reference_path=Path(
                                "/tmp/frozen-reference.json"
                            ),
                        )

    def test_historical_outcome_rejects_every_discrete_mutation(self):
        result = self._result(policy_mode="pi05")
        expected = self._expected_outcome(policy_mode="pi05")
        mutations = {
            "status": lambda value: value.__setitem__(
                "status", "method_failure"
            ),
            "terminal_reason": lambda value: value.__setitem__(
                "terminal_reason", "task_success"
            ),
            "task_success": lambda value: value.__setitem__(
                "task_success", True
            ),
            "paper_collision": lambda value: value["metrics"].__setitem__(
                "paper_collision", False
            ),
            "paper_collision_avoidance": (
                lambda value: value["metrics"].__setitem__(
                    "paper_collision_avoidance", True
                )
            ),
            "collision_first_step": (
                lambda value: value["metrics"].__setitem__(
                    "collision_first_step", 1
                )
            ),
            "executed_action_count": (
                lambda value: value["metrics"].__setitem__(
                    "executed_action_count", 2
                )
            ),
            "legacy_ets_steps": (
                lambda value: value["metrics"].__setitem__(
                    "legacy_ets_steps", 2
                )
            ),
            "policy_query_count": lambda value: value[
                "policy_queries"
            ].append(copy.deepcopy(value["policy_queries"][0])),
        }
        for label, mutate in mutations.items():
            with self.subTest(field=label):
                changed = copy.deepcopy(result)
                mutate(changed)
                with self.assertRaisesRegex(
                    canary.ActionInvariantCanaryError,
                    "historical canary outcome mismatch",
                ):
                    canary.validate_historical_outcome(
                        changed,
                        expected,
                    )

    def test_historical_outcome_reference_rejects_unrecorded_field(self):
        expected = self._expected_outcome(policy_mode="aegis")
        expected["terminal_state_sha256"] = "0" * 64
        with self.assertRaisesRegex(
            canary.ActionInvariantCanaryError,
            "historical canary outcome reference fields changed",
        ):
            canary.validate_historical_outcome(
                self._result(policy_mode="aegis"),
                expected,
            )

    def test_four_run_canary_enforces_outcome_for_every_result(self):
        with mock.patch.object(
            canary.diagnostics,
            "sha256_path",
            return_value=canary.REFERENCE_SHA256,
        ):
            with mock.patch.object(
                canary.failure_validation,
                "_load_json",
                return_value=self._reference(),
            ):
                with mock.patch.object(
                    canary,
                    "validate_observational_pair",
                    return_value={"status": "observationally_invariant"},
                ):
                    with mock.patch.object(
                        canary.failure_validation,
                        "validate_canary_reference",
                        return_value={"status": "reference_valid"},
                    ):
                        with mock.patch.object(
                            canary.failure_validation,
                            "validate_diagnostic_result",
                            return_value={"status": "diagnostics_valid"},
                        ):
                            with mock.patch.object(
                                canary,
                                "_decode_off_video",
                                return_value={"status": "video_valid"},
                            ):
                                for result_key in (
                                    (
                                        "diagnostics-off",
                                        "pi05",
                                    ),
                                    (
                                        "diagnostics-on",
                                        "pi05",
                                    ),
                                    (
                                        "diagnostics-off",
                                        "aegis",
                                    ),
                                    (
                                        "diagnostics-on",
                                        "aegis",
                                    ),
                                ):
                                    with self.subTest(result=result_key):
                                        results = self._four_results()
                                        results[result_key][
                                            "terminal_reason"
                                        ] = "mutated_terminal"
                                        with self.assertRaisesRegex(
                                            canary.ActionInvariantCanaryError,
                                            (
                                                "historical canary "
                                                "outcome mismatch"
                                            ),
                                        ):
                                            canary.validate_four_run_canary(
                                                results=results,
                                                output_roots={
                                                    "diagnostics-off": Path(
                                                        "/tmp/canary/off"
                                                    ),
                                                    "diagnostics-on": Path(
                                                        "/tmp/canary/on"
                                                    ),
                                                },
                                                reference_path=Path(
                                                    "/tmp/frozen-reference.json"
                                                ),
                                            )


if __name__ == "__main__":
    unittest.main()
