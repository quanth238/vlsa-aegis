from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "crfs_reach_progress", ROOT / "main/crfs_oracle/reach_progress.py"
)
assert SPEC is not None and SPEC.loader is not None
REACH_PROGRESS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = REACH_PROGRESS
SPEC.loader.exec_module(REACH_PROGRESS)

EXECUTED_REACH_ACTIONS = REACH_PROGRESS.EXECUTED_REACH_ACTIONS
TARGET_OBJECT_NAME = REACH_PROGRESS.TARGET_OBJECT_NAME
TRACKED_REACH_SUBSTEPS = REACH_PROGRESS.TRACKED_REACH_SUBSTEPS
ReachSnapshot = REACH_PROGRESS.ReachSnapshot
annotate_reach_snapshots = REACH_PROGRESS.annotate_reach_snapshots
annotate_reach_rollout = REACH_PROGRESS.annotate_reach_rollout
calibrate_reach_p_min = REACH_PROGRESS.calibrate_reach_p_min
capture_reach_snapshot = REACH_PROGRESS.capture_reach_snapshot


OBSTACLE = "moka_pot_obstacle_1"


def snapshot(*, eef, target=(1.0, 0.0, 0.0), obstacle=(0.0, 2.0, 0.0)) -> ReachSnapshot:
    return ReachSnapshot(
        target_object_name=TARGET_OBJECT_NAME,
        active_obstacle_name=OBSTACLE,
        eef_world_m=eef,
        target_world_m=target,
        active_obstacle_world_m=obstacle,
    )


def annotation(progress: float):
    start = snapshot(eef=(0.0, 0.0, 0.0))
    end = snapshot(eef=(progress, 0.0, 0.0))
    return annotate_reach_snapshots(start, end, executed_actions=EXECUTED_REACH_ACTIONS)


def tracked_motion(
    *,
    branch_eef=(0.0, 0.0, 0.0),
    end_eef=(0.4, 0.0, 0.0),
    branch_target=(1.0, 0.0, 0.0),
    end_target=(1.0, 0.0, 0.0),
    branch_obstacle=(0.0, 2.0, 0.0),
    end_obstacle=(0.0, 2.0, 0.0),
    target_maximum=None,
    obstacle_maximum=None,
):
    def distance(left, right):
        return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(left, right)))

    target_endpoint = distance(branch_target, end_target)
    obstacle_endpoint = distance(branch_obstacle, end_obstacle)
    return {
        "substep_samples": TRACKED_REACH_SUBSTEPS,
        "branch_eef_world_m": branch_eef,
        "end_eef_world_m": end_eef,
        "bodies": {
            TARGET_OBJECT_NAME: {
                "body_id": 0,
                "branch_world_m": branch_target,
                "end_world_m": end_target,
                "maximum_displacement_m": (
                    target_endpoint if target_maximum is None else target_maximum
                ),
                "endpoint_displacement_m": target_endpoint,
            },
            OBSTACLE: {
                "body_id": 1,
                "branch_world_m": branch_obstacle,
                "end_world_m": end_obstacle,
                "maximum_displacement_m": (
                    obstacle_endpoint if obstacle_maximum is None else obstacle_maximum
                ),
                "endpoint_displacement_m": obstacle_endpoint,
            },
        },
    }


class ReachProgressTest(unittest.TestCase):
    def test_allocation_distance_reconstructs_exactly_across_python_versions(self) -> None:
        start = ReachSnapshot(
            target_object_name=TARGET_OBJECT_NAME,
            active_obstacle_name="milk_obstacle_1",
            eef_world_m=(-0.22321332279868714, -0.00268991440858397, 1.1629438753911616),
            target_world_m=(-0.05856095881552641, 0.18592485042105739, 0.8984041501882155),
            active_obstacle_world_m=(-0.07512001092655214, 0.029999821980335475, 1.1785310443198294),
        )
        end = ReachSnapshot(
            target_object_name=TARGET_OBJECT_NAME,
            active_obstacle_name="milk_obstacle_1",
            eef_world_m=(-0.2314635829430369, 0.02744962281845183, 1.1670508969938587),
            target_world_m=(-0.05856095881552639, 0.18592485042105739, 0.898404150188853),
            active_obstacle_world_m=(-0.075120010929088, 0.02999982198027309, 1.1785310443252348),
        )

        result = annotate_reach_snapshots(start, end, executed_actions=5)

        # These are the allocation-era Python 3.8 values stored in the H100
        # population artifact.  Exact reconstruction preserves the complete
        # annotation hash and keeps the progress gate fail-closed.
        self.assertEqual(
            result.end_distance_to_branch_target_m.hex(),
            "0x1.6d2ee216ebf6ep-2",
        )
        self.assertEqual(result.reach_progress_m.hex(), "0x1.f2ca29b3edf80p-8")
        canonical = json.dumps(
            result.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
        self.assertEqual(
            hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
            "34a3c99d1e05e2a995a3df0360fa0c194a9defb521225d17b74c14673d9873a8",
        )

    def test_snapshot_reads_authoritative_simulator_positions(self) -> None:
        data = SimpleNamespace(
            site_xpos=[(0.2, 0.3, 0.4)],
            body_xpos=[(9.0, 9.0, 9.0), (0.7, 0.8, 0.9), (-0.2, 0.1, 0.5)],
        )
        domain = SimpleNamespace(
            # Body zero is deliberately unrelated, guarding against positional assumptions.
            obj_body_id={TARGET_OBJECT_NAME: 1, OBSTACLE: 2},
        )
        environment = SimpleNamespace(
            env=domain,
            robots=[SimpleNamespace(eef_site_id=0)],
            sim=SimpleNamespace(data=data),
            # A stale observation-like value must not affect the snapshot.
            robot0_eef_pos=(100.0, 100.0, 100.0),
        )

        result = capture_reach_snapshot(environment, TARGET_OBJECT_NAME, OBSTACLE)

        self.assertEqual(result.eef_world_m, (0.2, 0.3, 0.4))
        self.assertEqual(result.target_world_m, (0.7, 0.8, 0.9))
        self.assertEqual(result.active_obstacle_world_m, (-0.2, 0.1, 0.5))

    def test_progress_uses_fixed_branch_target_and_reports_motion(self) -> None:
        start = snapshot(eef=(0.0, 0.0, 0.0))
        end = snapshot(
            eef=(0.4, 0.0, 0.0),
            target=(0.7, 0.0, 0.0),
            obstacle=(0.0, 2.0, 0.25),
        )

        result = annotate_reach_snapshots(start, end, executed_actions=5)

        self.assertAlmostEqual(result.start_distance_to_branch_target_m, 1.0)
        self.assertAlmostEqual(result.end_distance_to_branch_target_m, 0.6)
        self.assertAlmostEqual(result.reach_progress_m, 0.4)
        self.assertAlmostEqual(result.target_displacement_m, 0.3)
        self.assertAlmostEqual(result.active_obstacle_displacement_m, 0.25)
        self.assertAlmostEqual(result.maximum_target_displacement_m, 0.3)
        self.assertAlmostEqual(result.maximum_active_obstacle_displacement_m, 0.25)
        self.assertEqual(result.branch_target_world_m, start.target_world_m)

    def test_annotation_requires_exactly_the_committed_five_actions(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly 5 executed actions"):
            annotate_reach_snapshots(
                snapshot(eef=(0.0, 0.0, 0.0)),
                snapshot(eef=(0.1, 0.0, 0.0)),
                executed_actions=10,
            )

    def test_runner_api_returns_original_rollout_and_reach_dict(self) -> None:
        data = SimpleNamespace(
            site_xpos=[(0.0, 0.0, 0.0)],
            body_xpos=[(1.0, 0.0, 0.0), (0.0, 2.0, 0.0)],
        )
        raw_domain = SimpleNamespace(obj_body_id={TARGET_OBJECT_NAME: 0, OBSTACLE: 1})
        render_environment = SimpleNamespace(
            env=raw_domain,
            robots=[SimpleNamespace(eef_site_id=0)],
            sim=SimpleNamespace(data=data),
        )

        class FakeSafeLiberoCase:
            def __init__(self):
                self.env = render_environment
                self.reset_calls = 0
                self.tracked_body_names = None

            def reset_and_settle(self):
                self.reset_calls += 1
                data.site_xpos[0] = (0.0, 0.0, 0.0)

            def rollout(self, actions, *, tracked_body_names=None):
                self.tracked_body_names = tuple(tracked_body_names or ())
                self.reset_and_settle()
                data.site_xpos[0] = (0.4, 0.0, 0.0)
                return {
                    "clearance_m": 0.01,
                    "action_count": len(actions),
                    "tracked_body_motion": tracked_motion(),
                }

        environment = FakeSafeLiberoCase()
        rollout, reach = annotate_reach_rollout(
            environment,
            [[0.0] * 7 for _ in range(5)],
            target_name=TARGET_OBJECT_NAME,
            obstacle_name=OBSTACLE,
        )

        self.assertEqual(rollout["clearance_m"], 0.01)
        self.assertEqual(rollout["action_count"], 5)
        self.assertAlmostEqual(reach["reach_progress_m"], 0.4)
        self.assertEqual(reach["executed_actions"], 5)
        self.assertEqual(environment.reset_calls, 2)
        self.assertEqual(
            environment.tracked_body_names,
            (TARGET_OBJECT_NAME, OBSTACLE),
        )

    def test_transient_body_motion_is_not_hidden_by_endpoint_return(self) -> None:
        class TransientCase:
            def rollout(self, actions, *, tracked_body_names=None):
                self.requested = tuple(tracked_body_names or ())
                return {
                    "clearance_m": 0.01,
                    "tracked_body_motion": tracked_motion(
                        target_maximum=0.02,
                        obstacle_maximum=0.03,
                    ),
                }

        environment = TransientCase()
        initial = snapshot(eef=(0.0, 0.0, 0.0))

        _, reach = annotate_reach_rollout(
            environment,
            [[0.0] * 7 for _ in range(5)],
            target_name=TARGET_OBJECT_NAME,
            obstacle_name=OBSTACLE,
            initial_snapshot=initial,
        )

        self.assertEqual(reach["target_displacement_m"], 0.0)
        self.assertEqual(reach["active_obstacle_displacement_m"], 0.0)
        self.assertEqual(reach["maximum_target_displacement_m"], 0.02)
        self.assertEqual(reach["maximum_active_obstacle_displacement_m"], 0.03)

    def test_annotation_rejects_a_mismatched_actual_reset_branch(self) -> None:
        class MismatchedCase:
            def rollout(self, actions, *, tracked_body_names=None):
                return {
                    "clearance_m": 0.01,
                    "tracked_body_motion": tracked_motion(
                        branch_target=(1.0001, 0.0, 0.0),
                        end_target=(1.0001, 0.0, 0.0),
                    ),
                }

        with self.assertRaisesRegex(RuntimeError, "paired reset branch.*target"):
            annotate_reach_rollout(
                MismatchedCase(),
                [[0.0] * 7 for _ in range(5)],
                target_name=TARGET_OBJECT_NAME,
                obstacle_name=OBSTACLE,
                initial_snapshot=snapshot(eef=(0.0, 0.0, 0.0)),
            )

    def test_calibration_uses_positive_inverted_cdf_q25_and_counts_failures(self) -> None:
        examples = [annotation(value) for value in (-0.1, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05)]

        result = calibrate_reach_p_min(examples, minimum_positive_examples=5)

        self.assertAlmostEqual(result.p_min_m, 0.02)
        self.assertEqual(result.quantile_method, "inverted_cdf")
        self.assertEqual(result.total_examples, 7)
        self.assertEqual(result.positive_examples, 5)
        self.assertEqual(result.nonpositive_examples, 2)

    def test_calibration_refuses_an_underpowered_positive_subset(self) -> None:
        examples = [annotation(value) for value in (-0.1, 0.0, 0.01)]
        with self.assertRaisesRegex(ValueError, "at least 2 positive reach examples"):
            calibrate_reach_p_min(examples, minimum_positive_examples=2)


if __name__ == "__main__":
    unittest.main()
