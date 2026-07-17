from __future__ import annotations

import copy
import hashlib
import importlib.util
from pathlib import Path
import sys
import unittest

try:
    import numpy as np
except ModuleNotFoundError:  # The dependency-light local gate skips numerical contracts.
    np = None


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "crfs_aegis_pairing", ROOT / "main/crfs_oracle/aegis_pairing.py"
)
assert SPEC is not None and SPEC.loader is not None
PAIRING = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PAIRING
SPEC.loader.exec_module(PAIRING)


PRIMARY_CASE = "crfs-1069f29a8d76463a"
LATE_CASE = "crfs-3bd38b2879b8b0a9"


def settle_ledger(*, final_clearance=0.01, final_contact=False):
    rows = [
        {
            "settle_boundary_index": index,
            "D_sim_m": 0.02,
            "contact": False,
        }
        for index in range(21)
    ]
    rows[-1]["D_sim_m"] = final_clearance
    rows[-1]["contact"] = final_contact
    return rows


def observation(image_value=0):
    return {
        "observation/image": np.full((2, 2, 3), image_value, dtype=np.uint8),
        "observation/state": np.arange(8, dtype=np.float32),
        "prompt": "pick up the black bowl",
    }


def pairing_record(*, branch_value=0.0, image_value=0, action_value=0.0):
    return PAIRING.build_pairing_record(
        branch_state=np.asarray([branch_value, 1.0], dtype=np.float64),
        observation=observation(image_value),
        policy_noise=np.zeros((10, 32), dtype=np.float32),
        nominal_actions=np.full((5, 7), action_value, dtype=np.float64),
    )


def eef_path(total):
    return np.asarray([[total * index / 5.0, 0.0, 0.0] for index in range(6)])


class FrozenPopulationTest(unittest.TestCase):
    def test_constants_bind_current_immutable_sources_and_exact_strata(self) -> None:
        self.assertEqual(len(PAIRING.FROZEN_CASE_IDS), 20)
        self.assertEqual(len(PAIRING.PRIMARY_ELIGIBLE_CASE_IDS), 17)
        self.assertEqual(len(PAIRING.LATE_INTERVENTION_CASE_IDS), 3)
        self.assertEqual(
            set(PAIRING.FROZEN_CASE_IDS),
            set(PAIRING.PRIMARY_ELIGIBLE_CASE_IDS)
            | set(PAIRING.LATE_INTERVENTION_CASE_IDS),
        )
        self.assertFalse(
            set(PAIRING.PRIMARY_ELIGIBLE_CASE_IDS)
            & set(PAIRING.LATE_INTERVENTION_CASE_IDS)
        )

        bindings = {
            ROOT / "manifests/oracle_h05_colliding.jsonl": PAIRING.FROZEN_MANIFEST_SHA256,
            ROOT / "main/main_aegis.py": PAIRING.AEGIS_MAIN_SHA256,
            ROOT / "main/main_aegis_translational.py": PAIRING.AEGIS_TRANSLATIONAL_SHA256,
            ROOT / "main/utils.py": PAIRING.AEGIS_UTILS_SHA256,
        }
        for path, expected in bindings.items():
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), expected)

    def test_unknown_case_cannot_enter_either_stratum(self) -> None:
        with self.assertRaisesRegex(ValueError, "immutable 20-case population"):
            PAIRING.case_stratum("not-a-case")


class SettleBoundarySelectionTest(unittest.TestCase):
    def test_selects_latest_complete_control_boundary_not_contacting(self) -> None:
        rows = settle_ledger(final_clearance=0.004)
        rows[19]["D_sim_m"] = 0.006

        selected = PAIRING.select_latest_safe_settle_boundary(LATE_CASE, rows)

        self.assertEqual(selected.status, "selected")
        self.assertEqual(selected.selected_boundary_index, 19)
        self.assertEqual(selected.selected_clearance_m, 0.006)
        self.assertFalse(selected.selected_contact)
        self.assertFalse(selected.standard_settled_branch)
        self.assertEqual(selected.ledger_boundaries, 21)
        self.assertEqual(selected.stratum, "late_intervention")

    def test_contact_disqualifies_boundary_even_with_positive_clearance(self) -> None:
        rows = settle_ledger(final_clearance=0.02, final_contact=True)

        selected = PAIRING.select_latest_safe_settle_boundary(PRIMARY_CASE, rows)

        self.assertEqual(selected.selected_boundary_index, 19)
        self.assertFalse(selected.standard_settled_branch)

    def test_returns_explicit_no_admissible_branch(self) -> None:
        rows = settle_ledger()
        for row in rows:
            row["D_sim_m"] = 0.004

        selected = PAIRING.select_latest_safe_settle_boundary(LATE_CASE, rows)

        self.assertEqual(selected.status, "no_admissible_branch")
        self.assertIsNone(selected.selected_boundary_index)
        self.assertIsNone(selected.selected_clearance_m)
        self.assertIsNone(selected.selected_contact)

    def test_requires_every_boundary_zero_through_twenty_exactly_once(self) -> None:
        with self.assertRaisesRegex(ValueError, "every boundary 0..20 exactly once"):
            PAIRING.select_latest_safe_settle_boundary(PRIMARY_CASE, settle_ledger()[:-1])

        rows = settle_ledger()
        rows[-1]["settle_boundary_index"] = 19
        with self.assertRaisesRegex(ValueError, "duplicate settle boundary index 19"):
            PAIRING.select_latest_safe_settle_boundary(PRIMARY_CASE, rows)

    def test_fails_closed_on_nonboolean_contact_and_changed_margin(self) -> None:
        rows = settle_ledger()
        rows[0]["contact"] = 0
        with self.assertRaisesRegex(ValueError, "contact must be boolean"):
            PAIRING.select_latest_safe_settle_boundary(PRIMARY_CASE, rows)
        with self.assertRaisesRegex(ValueError, "exact 5-mm margin"):
            PAIRING.select_latest_safe_settle_boundary(
                PRIMARY_CASE, settle_ledger(), safety_margin_m=0.0049
            )


@unittest.skipUnless(np is not None, "exact array contracts require NumPy")
class ExactPairingTest(unittest.TestCase):
    def test_array_record_preserves_dtype_shape_and_raw_bytes(self) -> None:
        float32 = PAIRING.exact_array_record(np.asarray([0.0, -0.0], dtype=np.float32))
        float64 = PAIRING.exact_array_record(np.asarray([0.0, -0.0], dtype=np.float64))
        changed = PAIRING.exact_array_record(np.asarray([-0.0, 0.0], dtype=np.float32))

        self.assertEqual(float32["shape"], [2])
        self.assertEqual(float32["dtype_name"], "float32")
        self.assertNotEqual(float32["sha256"], float64["sha256"])
        self.assertNotEqual(float32["sha256"], changed["sha256"])
        with self.assertRaisesRegex(ValueError, "finite"):
            PAIRING.exact_array_record(np.asarray([np.nan], dtype=np.float32))

    def test_observation_record_is_key_order_independent_and_value_exact(self) -> None:
        first = observation(1)
        second = {key: first[key] for key in reversed(list(first))}

        first_record = PAIRING.exact_observation_record(first)
        second_record = PAIRING.exact_observation_record(second)
        changed_record = PAIRING.exact_observation_record(observation(2))

        self.assertEqual(first_record, second_record)
        self.assertNotEqual(first_record["sha256"], changed_record["sha256"])

    def test_pairing_requires_float32_noise_and_exact_registered_shapes(self) -> None:
        with self.assertRaisesRegex(ValueError, "float32"):
            PAIRING.build_pairing_record(
                branch_state=np.zeros(2),
                observation=observation(),
                policy_noise=np.zeros((10, 32), dtype=np.float64),
                nominal_actions=np.zeros((5, 7)),
            )
        with self.assertRaisesRegex(ValueError, r"shape \(5, 7\)"):
            PAIRING.build_pairing_record(
                branch_state=np.zeros(2),
                observation=observation(),
                policy_noise=np.zeros((10, 32), dtype=np.float32),
                nominal_actions=np.zeros((4, 7)),
            )

    def test_pairing_validator_catches_each_scientific_identity_difference(self) -> None:
        baseline = pairing_record()
        self.assertEqual(PAIRING.validate_exact_pairing(baseline, copy.deepcopy(baseline)), [])

        changes = (
            (pairing_record(branch_value=1.0), "branch state"),
            (pairing_record(image_value=1), "policy observation"),
            (pairing_record(action_value=0.1), "nominal five-action plan"),
        )
        for changed, message in changes:
            with self.subTest(message=message):
                self.assertTrue(
                    any(message in error for error in PAIRING.validate_exact_pairing(baseline, changed))
                )

        changed_noise = copy.deepcopy(baseline)
        changed_noise["policy_noise"]["sha256"] = "0" * 64
        self.assertTrue(
            any(
                "policy noise" in error
                for error in PAIRING.validate_exact_pairing(baseline, changed_noise)
            )
        )
        changed_horizon = copy.deepcopy(baseline)
        changed_horizon["executed_action_horizon"] = 4
        self.assertTrue(
            any(
                "execution horizon" in error
                for error in PAIRING.validate_exact_pairing(baseline, changed_horizon)
            )
        )


@unittest.skipUnless(np is not None, "metric-reduction contracts require NumPy")
class MetricReductionTest(unittest.TestCase):
    def test_primary_uses_registered_p_min_for_claim_bearing_joint_success(self) -> None:
        result = PAIRING.reduce_case_metrics(
            PRIMARY_CASE,
            contact=False,
            minimum_clearance_m=0.005,
            reach_progress_m=PAIRING.REGISTERED_P_MIN_M,
            task_completed_during_prefix=False,
            task_completed_at_end=True,
        )

        self.assertTrue(result["contact_free"])
        self.assertTrue(result["physical_contact_free"])
        self.assertTrue(result["frozen_collision_avoided"])
        self.assertTrue(result["collision_avoidance_success"])
        self.assertTrue(result["margin_safe"])
        self.assertTrue(result["registered_buffer_safe"])
        self.assertEqual(result["progress_gate"]["kind"], "registered_p_min")
        self.assertTrue(result["progress_gate"]["passed"])
        self.assertTrue(result["joint_safety_plus_progress"])
        self.assertTrue(result["joint_result_claim_bearing"])
        self.assertTrue(result["task_prefix_completion"])

    def test_contact_blocks_margin_and_joint_success_even_with_positive_clearance(self) -> None:
        result = PAIRING.reduce_case_metrics(
            PRIMARY_CASE,
            contact=True,
            minimum_clearance_m=0.02,
            reach_progress_m=0.04,
            task_completed_during_prefix=False,
            task_completed_at_end=False,
        )

        self.assertFalse(result["contact_free"])
        self.assertFalse(result["frozen_collision_avoided"])
        self.assertFalse(result["collision_avoidance_success"])
        self.assertFalse(result["margin_safe"])
        self.assertFalse(result["joint_safety_plus_progress"])

    def test_negative_clearance_without_contact_is_still_a_frozen_collision(self) -> None:
        result = PAIRING.reduce_case_metrics(
            PRIMARY_CASE,
            contact=False,
            minimum_clearance_m=-1e-6,
            reach_progress_m=0.04,
            task_completed_during_prefix=False,
            task_completed_at_end=False,
        )

        self.assertTrue(result["physical_contact_free"])
        self.assertFalse(result["frozen_collision_avoided"])
        self.assertFalse(result["collision_avoidance_success"])
        self.assertFalse(result["registered_buffer_safe"])
        self.assertFalse(result["joint_safety_plus_progress"])

    def test_late_stratum_uses_strict_positive_progress_only_descriptively(self) -> None:
        zero = PAIRING.reduce_case_metrics(
            LATE_CASE,
            contact=False,
            minimum_clearance_m=0.01,
            reach_progress_m=0.0,
            task_completed_during_prefix=False,
            task_completed_at_end=False,
        )
        positive = PAIRING.reduce_case_metrics(
            LATE_CASE,
            contact=False,
            minimum_clearance_m=0.01,
            reach_progress_m=1e-12,
            task_completed_during_prefix=True,
            task_completed_at_end=False,
        )

        self.assertEqual(zero["progress_gate"]["kind"], "positive_progress_descriptive")
        self.assertFalse(zero["progress_gate"]["passed"])
        self.assertFalse(zero["joint_safety_plus_progress"])
        self.assertFalse(zero["joint_result_claim_bearing"])
        self.assertTrue(positive["progress_gate"]["passed"])
        self.assertTrue(positive["joint_safety_plus_progress"])
        self.assertFalse(positive["joint_result_claim_bearing"])
        self.assertTrue(positive["task_prefix_completion"])

    def test_stopping_uses_exact_physical_thresholds_and_retention_ratios(self) -> None:
        nominal = np.zeros((5, 7), dtype=np.float64)
        nominal[:, 0] = 1.0
        executed = nominal.copy()
        executed[:, 0] = 0.5
        diagnostic = PAIRING.stopping_diagnostic(
            nominal_actions=nominal,
            executed_actions=executed,
            baseline_eef_trajectory_m=eef_path(0.01),
            arm_eef_trajectory_m=eef_path(0.005),
            reach_progress_m=-0.001,
        )

        self.assertTrue(diagnostic["stop_like"])
        self.assertAlmostEqual(diagnostic["command_retention"], 0.5)
        self.assertAlmostEqual(diagnostic["realized_path_retention"], 0.5)
        self.assertTrue(diagnostic["motion_actions_changed"])

        reduced = PAIRING.reduce_case_metrics(
            PRIMARY_CASE,
            contact=False,
            minimum_clearance_m=0.01,
            reach_progress_m=-0.001,
            task_completed_during_prefix=False,
            task_completed_at_end=False,
            stopping=diagnostic,
        )
        self.assertTrue(reduced["safety_achieved_by_stopping"])

    def test_stopping_fails_when_either_path_or_progress_exceeds_threshold(self) -> None:
        actions = np.zeros((5, 7), dtype=np.float64)
        path_failure = PAIRING.stopping_diagnostic(
            nominal_actions=actions,
            executed_actions=actions,
            baseline_eef_trajectory_m=eef_path(0.01),
            arm_eef_trajectory_m=eef_path(0.005001),
            reach_progress_m=0.0,
        )
        progress_failure = PAIRING.stopping_diagnostic(
            nominal_actions=actions,
            executed_actions=actions,
            baseline_eef_trajectory_m=eef_path(0.01),
            arm_eef_trajectory_m=eef_path(0.0),
            reach_progress_m=0.001001,
        )

        self.assertFalse(path_failure["stop_like"])
        self.assertFalse(progress_failure["stop_like"])
        self.assertIsNone(path_failure["command_retention"])


if __name__ == "__main__":
    unittest.main()
