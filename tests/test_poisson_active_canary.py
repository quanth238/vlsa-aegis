from __future__ import annotations

import unittest
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

from main.poisson_fullbody.active_canary import (
    ActiveCanaryError,
    ControllerTrackingInvalid,
    PrefixCounters,
    TrackingAudit,
    classify_filter_failure,
    fixed_exposure_action_contract,
    paper_car_endpoint,
)
from scripts.run_poisson_active_canary import (
    ActiveRunnerError,
    _require_identification_matches_active_construction,
    _validated_run_output,
)


class ActiveCanaryContractTest(unittest.TestCase):
    def test_active_arms_cannot_start_before_shadow_construction_binding(self):
        source = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "run_poisson_active_canary.py"
        ).read_text(encoding="utf-8")
        binding_call = source.rindex(
            "_require_identification_matches_active_construction("
        )
        first_arm = source.index("for arm in ARMS:", binding_call)
        self.assertLess(binding_call, first_arm)

    def test_shadow_authorization_is_bound_to_active_field_construction(self):
        @dataclass(frozen=True)
        class Diagnostics:
            value: int

        @dataclass(frozen=True)
        class Hashes:
            bundle_sha256: str

        @dataclass(frozen=True)
        class Component:
            geom_id: int

        bundle = SimpleNamespace(
            protocol_id="poisson-protocol",
            protected_body_ids=(50, 60),
            protected_body_names=("robot0_link5", "robot0_link6"),
            diagnostics=Diagnostics(7),
            hashes=Hashes("bundle-hash"),
            protected_samples=SimpleNamespace(
                components=(Component(5), Component(6)),
                samples=(object(), object()),
            ),
        )
        resolved = {"robot_geom_ids": [5, 6], "obstacle_geom_ids": [100]}
        sampling = {
            "sample_count": 2,
            "sample_ledger_sha256": "sample-ledger",
            "roundtrip": {"passed": True},
        }
        construction = {
            "active_obstacle_name": "moka_pot",
            "contact_model_authority_sha256": "contact-authority",
            "robot_root_body_name": "robot0_base",
            "robot_root_body_ids": [1],
            "arm_dof_indices": list(range(7)),
            "resolved_geometry": resolved,
            "field_bundle": {
                "protocol_id": "poisson-protocol",
                "protected_body_ids": [50, 60],
                "protected_body_names": ["robot0_link5", "robot0_link6"],
                "diagnostics": {"value": 7},
                "hashes": {"bundle_sha256": "bundle-hash"},
                "surface_components": [{"geom_id": 5}, {"geom_id": 6}],
                "protected_sample_count": 2,
            },
            "full_robot_measurement_sampling": {
                "sample_count": 2,
                "sample_ledger_sha256": "sample-ledger",
                "rigid_roundtrip": {"passed": True},
            },
            "complete_integration_state_read_only_audit": {
                "mujoco_state_specification": "mjSTATE_INTEGRATION",
                "state_vector_length": 9,
                "before_sha256": "settled-state",
                "after_sha256": "settled-state",
                "exact_array_equal": True,
            },
        }
        prerequisite = {
            "shadow_replay": {"construction": construction}
        }
        arguments = {
            "case": {"active_obstacle_name": "moka_pot"},
            "active_obstacle_name": "moka_pot",
            "contact_model_authority_sha256": "contact-authority",
            "robot_root_body_name": "robot0_base",
            "robot_root_body_ids": (1,),
            "arm_dof_indices": tuple(range(7)),
            "resolved_geometry": resolved,
            "field_bundle": bundle,
            "full_robot_sampling": sampling,
            "settled_integration_state_sha256": "settled-state",
            "settled_integration_state_length": 9,
        }
        _require_identification_matches_active_construction(
            prerequisite, **arguments
        )

        for label, path, replacement in (
            ("obstacle", ("active_obstacle_name",), "other_obstacle"),
            (
                "field hash",
                ("field_bundle", "hashes", "bundle_sha256"),
                "other-field",
            ),
            (
                "resolved geometry",
                ("resolved_geometry", "obstacle_geom_ids"),
                [101],
            ),
            (
                "sampling ledger",
                ("full_robot_measurement_sampling", "sample_ledger_sha256"),
                "other-samples",
            ),
            (
                "settled state",
                (
                    "complete_integration_state_read_only_audit",
                    "before_sha256",
                ),
                "other-state",
            ),
        ):
            with self.subTest(binding=label):
                corrupted = json.loads(json.dumps(prerequisite))
                target = corrupted["shadow_replay"]["construction"]
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = replacement
                with self.assertRaises(ActiveRunnerError):
                    _require_identification_matches_active_construction(
                        corrupted, **arguments
                    )

    def test_run_id_is_portable_and_contained_by_real_output_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected = root.resolve() / "vlsa-poisson-canary-20260731d"
            self.assertEqual(
                _validated_run_output(root, "vlsa-poisson-canary-20260731d"),
                expected,
            )
            for invalid in ("", ".", "../escape", "nested/run", "bad id", "a" * 129):
                with self.subTest(run_id=invalid):
                    with self.assertRaises(ActiveRunnerError):
                        _validated_run_output(root, invalid)

            outside = root / "outside"
            outside.mkdir()
            linked = root / "linked-run"
            linked.symlink_to(outside, target_is_directory=True)
            with self.assertRaises(ActiveRunnerError):
                _validated_run_output(root, linked.name)

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
