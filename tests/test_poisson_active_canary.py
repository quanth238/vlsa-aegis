from __future__ import annotations

import unittest
import hashlib
import json

from main.poisson_fullbody.active_canary import (
    ActiveCanaryError,
    ControllerTrackingInvalid,
    PrefixCounters,
    TrackingAudit,
    classify_filter_failure,
    fixed_exposure_action_contract,
    paper_car_endpoint,
)


class ActiveCanaryContractTest(unittest.TestCase):
    def test_terminal_fifth_command_is_verified_without_a_next_provider(self):
        audit = TrackingAudit(
            maximum_linf_rad_s=0.05,
            maximum_rmse_rad_s=0.02,
        )
        for inner in range(5):
            command = [0.01 * (inner + 1)] * 7
            audit.register_command(
                high_level_index=0,
                inner_control_index=inner,
                qdot_command_rad_s=command,
            )
            for physics in range(5):
                result = audit.observe_post_integration(
                    high_level_index=0,
                    inner_control_index=inner,
                    physics_substep_index=physics,
                    measured_qvel_rad_s=command,
                )
                self.assertIsNotNone(result)
        audit.require_complete(5)
        self.assertTrue(audit.summary()["final_fifth_command_verified"])
        self.assertEqual(audit.summary()["observed_physics_substep_count"], 25)

    def test_tracking_rejects_a_transient_before_the_fifth_substep(self):
        audit = TrackingAudit(
            maximum_linf_rad_s=0.05,
            maximum_rmse_rad_s=0.05,
        )
        audit.register_command(
            high_level_index=0,
            inner_control_index=0,
            qdot_command_rad_s=[0.0] * 7,
        )
        with self.assertRaises(ControllerTrackingInvalid):
            audit.observe_post_integration(
                high_level_index=0,
                inner_control_index=0,
                physics_substep_index=0,
                measured_qvel_rad_s=[0.051] * 7,
            )
        crossing = audit.summary()["first_threshold_crossing"]
        self.assertEqual(crossing["physics_substep_index"], 0)
        self.assertEqual(crossing["high_level_index"], 0)

    def test_final_command_tracking_failure_is_not_lost(self):
        audit = TrackingAudit(
            maximum_linf_rad_s=0.05,
            maximum_rmse_rad_s=0.02,
        )
        audit.register_command(
            high_level_index=0,
            inner_control_index=0,
            qdot_command_rad_s=[0.0] * 7,
        )
        for physics in range(4):
            audit.observe_post_integration(
                high_level_index=0,
                inner_control_index=0,
                physics_substep_index=physics,
                measured_qvel_rad_s=[0.0] * 7,
            )
        with self.assertRaises(ControllerTrackingInvalid):
            audit.observe_post_integration(
                high_level_index=0,
                inner_control_index=0,
                physics_substep_index=4,
                measured_qvel_rad_s=[0.051] * 7,
            )
        self.assertEqual(audit.verified_count, 1)

    def test_tracking_uses_cumulative_rmse_threshold(self):
        audit = TrackingAudit(
            maximum_linf_rad_s=1.0,
            maximum_rmse_rad_s=0.02,
        )
        audit.register_command(
            high_level_index=0,
            inner_control_index=0,
            qdot_command_rad_s=[0.0] * 7,
        )
        with self.assertRaises(ControllerTrackingInvalid):
            audit.observe_post_integration(
                high_level_index=0,
                inner_control_index=0,
                physics_substep_index=0,
                measured_qvel_rad_s=[0.021] * 7,
            )

    def test_paper_car_is_strict_and_right_censored(self):
        at_threshold = paper_car_endpoint(
            [0.0, 0.0, 0.0],
            [[0.001, 0.0, 0.0]],
            fixed_exposure_complete=True,
        )
        self.assertFalse(at_threshold["collision"])
        over_threshold = paper_car_endpoint(
            [0.0, 0.0, 0.0],
            [[0.0010000001, 0.0, 0.0]],
            fixed_exposure_complete=True,
        )
        self.assertTrue(over_threshold["collision"])
        partial = paper_car_endpoint(
            [0.0, 0.0, 0.0],
            [[1.0, 0.0, 0.0]],
            fixed_exposure_complete=False,
        )
        self.assertFalse(partial["available"])
        self.assertIsNone(partial["collision"])
        self.assertIsNone(partial["avoidance"])

    def test_prefix_counters_cover_exact_five_by_five_cadence(self):
        counters = PrefixCounters()
        counters.enter_high_level(0)
        for inner in range(5):
            counters.enter_inner(0, inner)
            for physics in range(5):
                counters.complete_physics(0, inner, physics)
        self.assertTrue(counters.exposure_complete(1))
        self.assertEqual(counters.record()["physics_exposure_seconds"], 0.05)
        with self.assertRaises(ActiveCanaryError):
            counters.enter_high_level(2)

    def test_qp_failure_mapping_is_fail_closed(self):
        self.assertEqual(
            classify_filter_failure(
                "qp_not_solved", {"status": "primal infeasible"}
            ).completion_class,
            "qp_infeasible",
        )
        self.assertEqual(
            classify_filter_failure("qp_postcheck_failed", {}).completion_class,
            "fail_closed_runtime",
        )
        self.assertEqual(
            classify_filter_failure("qp_solver_exception", {}).completion_class,
            "qp_solver_failure",
        )
        self.assertEqual(
            classify_filter_failure(
                "unsafe_or_invalid_field_start", {}, initial_preflight=True
            ).completion_class,
            "safe_start_inadmissible",
        )
        self.assertEqual(
            classify_filter_failure(
                "unsafe_or_invalid_field_start", {}, initial_preflight=False
            ).completion_class,
            "barrier_invariance_lost",
        )

    def test_frozen_source_contract_rejects_partial_ledger(self):
        actions = [[0.0] * 7 for _ in range(237)]
        expected = hashlib.sha256(
            json.dumps(
                actions,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        self.assertEqual(
            len(
                fixed_exposure_action_contract(
                    actions,
                    expected_count=237,
                    expected_sha256=expected,
                )
            ),
            237,
        )
        with self.assertRaises(ActiveCanaryError):
            fixed_exposure_action_contract(
                actions[:-1],
                expected_count=237,
                expected_sha256=expected,
            )


if __name__ == "__main__":
    unittest.main()
