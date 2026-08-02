from __future__ import annotations

from copy import deepcopy
import inspect
import json
import math
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

from main.poisson_fullbody.closed_loop_canary import (
    CLASSIFICATIONS,
    ClosedLoopCanaryError,
    classify_closed_loop_canary,
    validate_closed_loop_canary_protocol,
)
from scripts.run_poisson_closed_loop_canary import (
    _aegis_state_and_command_chain_valid,
    _feedback_metrics,
    _own_observation_chain_valid,
    _provider_contract,
)
from scripts import run_poisson_closed_loop_canary as closed_loop_runner


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "configs/vlsa_poisson_closed_loop_canary.v1.json"


def _positive_metrics():
    """Small, complete witness for the preregistered positive outcome."""

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
        "maximum_correction_norm_rad_s": 0.5,
        "filter_correction_integral_rad": 0.1,
        "post_correction_measured_joint_motion_integral_rad": 0.5,
        "post_correction_cartesian_path_length_m": 0.05,
        "post_correction_executed_command_integral_rad": 0.5,
        "post_correction_zero_command_fraction": 0.0,
        "baseline_task_success_ever": True,
        "psf_task_success_ever": True,
        "psf_task_success_after_material_correction": True,
    }


def _provider(action_count=57, query_count=12, arm="arm"):
    queries = []
    for query_offset in range(query_count):
        query_index = 36 + query_offset
        local_index = query_offset * 5
        returned_actions = [
            [float(query_index), float(row), 2.0, 3.0, 4.0, 5.0, 6.0]
            for row in range(10)
        ]
        queries.append(
            {
                "arm": arm,
                "query_index": query_index,
                "local_action_index": local_index,
                "source_action_index": 180 + local_index,
                "rng_seed": 2026691220 + query_index,
                "native_observation_sha256": "observation-%d" % local_index,
                "policy_input_fingerprint_sha256": "input-%d" % query_index,
                "returned_actions": returned_actions,
                "returned_action_shape": [10, 7],
                "returned_actions_sha256": closed_loop_runner._array_sha256(
                    returned_actions
                ),
            }
        )

    actions = []
    for local_index in range(action_count):
        query_offset = local_index // 5
        query_index = 36 + query_offset
        chunk_offset = local_index % 5
        returned = queries[query_offset]["returned_actions"][chunk_offset]
        actions.append(
            {
                "arm": arm,
                "local_action_index": local_index,
                "source_action_index": 180 + local_index,
                "query_index": query_index,
                "query_chunk_offset": chunk_offset,
                "native_observation_sha256": (
                    "observation-%d" % local_index
                ),
                "official_integration_state_raw_bytes_sha256": (
                    "state-%d" % local_index
                ),
                "nominal_raw": deepcopy(returned),
                "nominal_translational": [0.02] * 7,
                "aegis_executed": [0.01] * 7,
                "aegis_z_before": [1.0, 0.0, 0.0],
                "aegis_z_after": [1.0, 0.0, 0.0],
                "aegis_qp": {
                    "status": "solved",
                    "solver_status": "optimal",
                    "z_before": [1.0, 0.0, 0.0],
                    "z_after": [1.0, 0.0, 0.0],
                    "context": {
                        "q1_diag": [0.06, 0.12, 0.11],
                        "z_before": [1.0, 0.0, 0.0],
                        "z_after": [1.0, 0.0, 0.0],
                        "nominal_translational": [0.02] * 7,
                        "executed_action": [0.01] * 7,
                    },
                },
            }
        )
    return {
        "arm": arm,
        "source": "live_pi05_queries_from_this_arms_own_observations",
        "recorded_suffix_actions_executed": False,
        "first_query_index": 36,
        "initial_aegis_z_fixed_from_historical_action_179": [1.0, 0.0, 0.0],
        "policy_queries": queries,
        "high_level_action_trace": actions,
    }


def _arm_for_provider(provider):
    action_count = len(provider["high_level_action_trace"])
    returned = ["observation-%d" % (index + 1) for index in range(action_count)]
    # A replan at local action k consumes the observation returned by k - 1.
    for query in provider["policy_queries"]:
        local_index = query["local_action_index"]
        if local_index > 0:
            query["native_observation_sha256"] = returned[local_index - 1]
    return {
        "task": {"returned_observation_sha256_ledger": returned},
        "command_trace": [
            {
                "source_action_index": row["source_action_index"],
                "source_action": deepcopy(row["aegis_executed"]),
            }
            for row in provider["high_level_action_trace"]
        ],
    }


class _FakeArray:
    def __init__(self, value):
        if isinstance(value, _FakeArray):
            value = value.values
        self.values = [float(item) for item in value]
        self.shape = (len(self.values),)

    def __iter__(self):
        return iter(self.values)


def _fake_numpy_module():
    module = types.ModuleType("numpy")
    module.float64 = float
    module.asarray = lambda value, dtype=None: _FakeArray(value)

    def allclose(left, right, rtol=0.0, atol=1e-8):
        left_values = list(left)
        right_values = list(right)
        return len(left_values) == len(right_values) and all(
            abs(a - b) <= atol + rtol * abs(b)
            for a, b in zip(left_values, right_values)
        )

    module.allclose = allclose
    module.linalg = types.SimpleNamespace(
        norm=lambda value: math.sqrt(sum(item * item for item in value))
    )
    return module


class ClosedLoopProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))

    def test_protocol_derives_the_full_live_suffix(self):
        derived = validate_closed_loop_canary_protocol(self.protocol)
        self.assertEqual(derived["start_action"], 180)
        self.assertEqual(derived["end_action"], 236)
        self.assertEqual(derived["action_count"], 57)
        self.assertEqual(derived["first_query_index"], 36)
        self.assertEqual(derived["queries_per_arm"], 12)
        self.assertEqual(derived["replan_steps"], 5)
        self.assertEqual(derived["expected_updates"], 285)
        self.assertEqual(derived["expected_substeps"], 1425)
        self.assertEqual(derived["clearance_diagnostic_stride"], 25)

    def test_protocol_rejects_policy_or_execution_drift(self):
        mutations = (
            ("query index", ("online_policy", "first_query_index"), 0),
            (
                "recorded suffix",
                ("online_policy", "recorded_suffix_actions_prohibited"),
                False,
            ),
            (
                "own observations",
                ("online_policy", "own_observation_after_divergence"),
                False,
            ),
            (
                "AEGIS state",
                ("execution", "aegis_state"),
                "shared_mutable_state",
            ),
            (
                "causal arm difference",
                ("execution", "only_low_level_difference"),
                "different_adapters_and_psf",
            ),
            (
                "literal contact cadence",
                (
                    "cadence",
                    "literal_contact_observation_stride_physics_substeps",
                ),
                25,
            ),
            (
                "clearance diagnostic cadence",
                (
                    "cadence",
                    "full_robot_clearance_diagnostic_stride_physics_substeps",
                ),
                1,
            ),
        )
        for label, path, replacement in mutations:
            with self.subTest(label=label):
                protocol = deepcopy(self.protocol)
                protocol[path[0]][path[1]] = replacement
                with self.assertRaises(ClosedLoopCanaryError):
                    validate_closed_loop_canary_protocol(protocol)

    def test_protocol_rejects_disabled_preregistered_acceptance_requirements(self):
        required_true = (
            "require_baseline_link56_contact",
            "require_both_native_task_success_ever",
            "require_literal_contact_check_at_every_physics_substep",
            "require_live_own_observation_feedback_after_divergence",
            "require_material_correction_before_baseline_contact",
            "require_no_treatment_contact_from_any_robot_geometry",
            "require_two_complete_decodable_videos",
            "full_robot_clearance_is_diagnostic_not_acceptance_gate",
        )
        for field in required_true:
            with self.subTest(field=field):
                protocol = deepcopy(self.protocol)
                protocol["acceptance"][field] = False
                with self.assertRaises(ClosedLoopCanaryError):
                    validate_closed_loop_canary_protocol(protocol)

    def test_protocol_rejects_weakened_result_contract(self):
        replacements = {
            "schema_version": "wrong",
            "write_policy": "overwrite",
            "partial_output_policy": "interpret_partial",
            "independent_consumer_required": False,
            "claim_scope": "population_safety",
        }
        for field, replacement in replacements.items():
            with self.subTest(field=field):
                protocol = deepcopy(self.protocol)
                protocol["result_contract"][field] = replacement
                with self.assertRaises(ClosedLoopCanaryError):
                    validate_closed_loop_canary_protocol(protocol)


class ClosedLoopClassificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))

    def classify(self, metrics):
        return classify_closed_loop_canary(metrics, self.protocol)

    def test_positive_requires_safe_task_success_and_useful_correction(self):
        result = self.classify(_positive_metrics())
        self.assertEqual(
            result["classification"], "SAFE_TASK_SUCCESS_USEFUL_CORRECTION"
        )
        self.assertTrue(result["feasible"])
        self.assertTrue(result["closed_loop_feedback_valid"])
        self.assertTrue(result["contact_prevented"])
        self.assertTrue(result["useful_correction"])
        self.assertTrue(result["task_successful"])

    def test_contact_prevented_but_task_failed_is_not_feasible(self):
        metrics = _positive_metrics()
        metrics["psf_task_success_after_material_correction"] = False
        result = self.classify(metrics)
        self.assertEqual(result["classification"], "CONTACT_PREVENTED_TASK_FAILED")
        self.assertFalse(result["feasible"])

    def test_stopping_instead_of_useful_motion_is_not_feasible(self):
        metrics = _positive_metrics()
        metrics["post_correction_cartesian_path_length_m"] = 0.0
        result = self.classify(metrics)
        self.assertEqual(result["classification"], "STOP_ONLY")
        self.assertTrue(result["stop_only"])
        self.assertFalse(result["feasible"])

    def test_missing_baseline_contact_is_not_a_safety_success(self):
        metrics = _positive_metrics()
        metrics["baseline_link56_contact_present"] = False
        result = self.classify(metrics)
        self.assertEqual(
            result["classification"], "BASELINE_CONTACT_NOT_REPRODUCED"
        )
        self.assertFalse(result["feasible"])

    def test_apparatus_or_feedback_failures_are_inconclusive(self):
        for field in (
            "exact_paired_start",
            "shared_prefix_complete",
            "live_policy_contract_valid",
            "aegis_contract_valid",
            "both_nominal_commands_within_dynamic_joint_bounds",
            "videos_complete_and_decodable",
            "psf_qp_contract_valid",
            "static_selected_obstacle_admissible",
            "literal_contact_checked_at_every_physics_substep",
            "released_eef_marker_update_contract_valid",
            "first_live_query_identical",
            "own_observation_chain_valid",
            "no_recorded_suffix_action_replay",
            "post_divergence_own_observations_used",
        ):
            with self.subTest(field=field):
                metrics = _positive_metrics()
                metrics[field] = False
                result = self.classify(metrics)
                self.assertEqual(result["classification"], "INCONCLUSIVE_APPARATUS")

    def test_positive_requires_a_fresh_policy_query_after_material_correction(self):
        metrics = _positive_metrics()
        metrics["fresh_policy_query_after_material_correction"] = False
        result = self.classify(metrics)
        self.assertEqual(result["classification"], "INCONCLUSIVE_APPARATUS")

        source_without_whitespace = "".join(
            inspect.getsource(closed_loop_runner.main).split()
        )
        self.assertIn(
            'int(row["source_action_index"])*25>int(first_correction)',
            source_without_whitespace,
        )

    def test_treatment_contact_is_a_valid_negative_even_when_it_ends_early(self):
        metrics = _positive_metrics()
        metrics.update(
            {
                "psf_exposure_complete": False,
                "psf_any_robot_selected_obstacle_contact_present": True,
                "post_divergence_own_observations_used": False,
                "fresh_policy_query_after_material_correction": False,
                "material_correction_before_baseline_contact": False,
                "maximum_correction_norm_rad_s": 0.0,
                "filter_correction_integral_rad": 0.0,
                "post_correction_measured_joint_motion_integral_rad": 0.0,
                "post_correction_cartesian_path_length_m": 0.0,
                "post_correction_executed_command_integral_rad": 0.0,
                "post_correction_zero_command_fraction": 1.0,
                "psf_task_success_ever": False,
                "psf_task_success_after_material_correction": False,
            }
        )
        result = self.classify(metrics)
        self.assertEqual(result["classification"], "CONTACT_REMAINS_OR_SHIFTED")
        self.assertTrue(result["pair_complete"])
        self.assertFalse(result["feasible"])

    def test_label_order_is_frozen(self):
        self.assertEqual(
            CLASSIFICATIONS,
            (
                "SAFE_TASK_SUCCESS_USEFUL_CORRECTION",
                "CONTACT_PREVENTED_TASK_FAILED",
                "STOP_ONLY",
                "CONTACT_REMAINS_OR_SHIFTED",
                "BASELINE_CONTACT_NOT_REPRODUCED",
                "INCONCLUSIVE_APPARATUS",
            ),
        )


class ClosedLoopEvidenceChainTests(unittest.TestCase):
    def test_provider_maps_57_actions_to_queries_36_through_47(self):
        provider = _provider()
        self.assertTrue(
            _provider_contract(provider, expected_actions=57, expected_queries=12)
        )
        self.assertEqual(
            [row["query_index"] for row in provider["policy_queries"]],
            list(range(36, 48)),
        )
        self.assertEqual(
            [
                (
                    row["source_action_index"],
                    row["query_index"],
                    row["query_chunk_offset"],
                )
                for row in provider["high_level_action_trace"][-2:]
            ],
            [(235, 47, 0), (236, 47, 1)],
        )

    def test_provider_rejects_chunk_mapping_observation_or_replay_drift(self):
        provider = _provider()
        mutations = (
            lambda value: value["high_level_action_trace"][6].__setitem__(
                "query_chunk_offset", 0
            ),
            lambda value: value["high_level_action_trace"][6].__setitem__(
                "nominal_raw", [999.0] * 7
            ),
            lambda value: value["high_level_action_trace"][5].__setitem__(
                "native_observation_sha256", "wrong-query-observation"
            ),
            lambda value: value.__setitem__(
                "recorded_suffix_actions_executed", True
            ),
            lambda value: value["policy_queries"][1].__setitem__(
                "source_action_index", 999
            ),
            lambda value: value["policy_queries"][1].__setitem__(
                "rng_seed", 999
            ),
            lambda value: value["policy_queries"][1].__setitem__(
                "returned_actions_sha256", "wrong"
            ),
        )
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                candidate = deepcopy(provider)
                mutate(candidate)
                self.assertFalse(
                    _provider_contract(
                        candidate, expected_actions=57, expected_queries=12
                    )
                )

    def test_provider_accepts_auditable_early_contact_prefix(self):
        provider = _provider(action_count=1, query_count=1)
        self.assertTrue(
            _provider_contract(provider, expected_actions=1, expected_queries=1)
        )

    def test_checkpoint_identity_reconstructs_registered_relative_paths(self):
        from scripts import validate_aegis_assets as assets

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoint"
            asset = checkpoint / "params" / "metadata"
            asset.parent.mkdir(parents=True)
            asset.write_bytes(b"checkpoint-metadata")
            identity = assets.pi05_checkpoint_filesystem_identity(
                checkpoint, ("params/metadata",)
            )
            receipt = {
                "schema_version": "vlsa_table1_pi05_hash_receipt.v1",
                "status": "passed",
                "checkpoint": {
                    "full_content_hash_verified": True,
                    "full_content_tree_sha256": "a" * 64,
                    "filesystem_identity": identity,
                },
            }
            receipt["receipt_payload_sha256"] = closed_loop_runner.fast._sha256(
                closed_loop_runner.fast._canonical(receipt)
            )
            receipt_path = root / "receipt.json"
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
            contract = {
                "checkpoint": str(checkpoint),
                "checkpoint_tree_sha256": "a" * 64,
                "checkpoint_receipt": str(receipt_path),
                "checkpoint_receipt_sha256": (
                    closed_loop_runner.fast._file_sha256(receipt_path)
                ),
                "checkpoint_receipt_schema_version": (
                    "vlsa_table1_pi05_hash_receipt.v1"
                ),
            }
            with mock.patch.object(
                assets,
                "validate_pi05_filesystem_identity_record",
                return_value=identity,
            ), mock.patch.object(
                assets,
                "pi05_checkpoint_filesystem_identity",
                wraps=assets.pi05_checkpoint_filesystem_identity,
            ) as reconstruct:
                evidence = closed_loop_runner._checkpoint_identity(contract)
            reconstruct.assert_called_once_with(
                checkpoint, ("params/metadata",)
            )
            self.assertTrue(
                evidence["current_filesystem_identity_matches_receipt"]
            )
            self.assertEqual(
                evidence["filesystem_identity_sha256"],
                identity["identity_sha256"],
            )

    def test_every_replan_uses_the_preceding_local_returned_observation(self):
        provider = _provider()
        arm = _arm_for_provider(provider)
        self.assertTrue(_own_observation_chain_valid(provider, arm))

        stale_query = deepcopy(provider)
        stale_query["policy_queries"][1]["native_observation_sha256"] = "stale"
        self.assertFalse(_own_observation_chain_valid(stale_query, arm))

        stale_nonquery_action = deepcopy(provider)
        stale_nonquery_action["high_level_action_trace"][2][
            "native_observation_sha256"
        ] = "stale"
        self.assertFalse(_own_observation_chain_valid(stale_nonquery_action, arm))

    def test_feedback_preserves_first_query_then_uses_each_arm_after_divergence(self):
        baseline = _provider(arm="baseline")
        psf = _provider(arm="psf")
        for provider in (baseline, psf):
            provider["policy_queries"][0][
                "policy_input_fingerprint_sha256"
            ] = "shared-input"
            provider["policy_queries"][0]["returned_actions_sha256"] = "first-chunk"
        psf["high_level_action_trace"][1][
            "official_integration_state_raw_bytes_sha256"
        ] = "psf-diverged-state"
        for query in baseline["policy_queries"][1:]:
            query["policy_input_fingerprint_sha256"] = "baseline-%d" % query[
                "query_index"
            ]
        for query in psf["policy_queries"][1:]:
            query["policy_input_fingerprint_sha256"] = "psf-%d" % query[
                "query_index"
            ]

        metrics = _feedback_metrics(baseline, psf, "first-chunk")
        self.assertTrue(metrics["first_live_query_identical"])
        self.assertEqual(metrics["first_divergent_action_input_local_index"], 1)
        self.assertEqual(metrics["scheduled_query_indexes_after_divergence"][0], 37)
        self.assertEqual(
            metrics["post_divergence_query_indexes_with_distinct_policy_inputs"],
            list(range(37, 48)),
        )
        self.assertTrue(metrics["post_divergence_own_observations_used"])

    def test_aegis_z_state_and_executed_action_chain_are_exact(self):
        provider = _provider(action_count=2, query_count=1)
        first = provider["high_level_action_trace"][0]
        second = provider["high_level_action_trace"][1]
        first["aegis_z_after"] = [0.0, 1.0, 0.0]
        first["aegis_qp"]["z_after"] = [0.0, 1.0, 0.0]
        first["aegis_qp"]["context"]["z_after"] = [0.0, 1.0, 0.0]
        second["aegis_z_before"] = [0.0, 1.0, 0.0]
        second["aegis_qp"]["z_before"] = [0.0, 1.0, 0.0]
        second["aegis_qp"]["context"]["z_before"] = [0.0, 1.0, 0.0]
        arm = _arm_for_provider(provider)
        fake_numpy = _fake_numpy_module()
        with mock.patch.dict(sys.modules, {"numpy": fake_numpy}):
            self.assertTrue(_aegis_state_and_command_chain_valid(provider, arm))

            broken_z = deepcopy(provider)
            broken_z["high_level_action_trace"][1]["aegis_z_before"] = [1.0, 0.0, 0.0]
            self.assertFalse(_aegis_state_and_command_chain_valid(broken_z, arm))

            broken_command = deepcopy(arm)
            broken_command["command_trace"][0]["source_action"] = [0.02] * 7
            self.assertFalse(
                _aegis_state_and_command_chain_valid(provider, broken_command)
            )

            broken_qp_z = deepcopy(provider)
            broken_qp_z["high_level_action_trace"][0]["aegis_qp"][
                "z_after"
            ] = [1.0, 0.0, 0.0]
            self.assertFalse(
                _aegis_state_and_command_chain_valid(broken_qp_z, arm)
            )

            broken_context_z = deepcopy(provider)
            broken_context_z["high_level_action_trace"][0]["aegis_qp"][
                "context"
            ]["z_after"] = [1.0, 0.0, 0.0]
            self.assertFalse(
                _aegis_state_and_command_chain_valid(broken_context_z, arm)
            )

            broken_qp_action = deepcopy(provider)
            broken_qp_action["high_level_action_trace"][0]["aegis_qp"][
                "context"
            ]["executed_action"] = [0.02] * 7
            self.assertFalse(
                _aegis_state_and_command_chain_valid(broken_qp_action, arm)
            )

            broken_q1 = deepcopy(provider)
            broken_q1["high_level_action_trace"][0]["aegis_qp"]["context"][
                "q1_diag"
            ] = [0.6, 1.2, 1.1]
            self.assertFalse(_aegis_state_and_command_chain_valid(broken_q1, arm))


if __name__ == "__main__":
    unittest.main()
