from __future__ import annotations

import importlib.util
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


class ReachProgressTest(unittest.TestCase):
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

            def reset_and_settle(self):
                self.reset_calls += 1
                data.site_xpos[0] = (0.0, 0.0, 0.0)

            def rollout(self, actions):
                self.reset_and_settle()
                data.site_xpos[0] = (0.4, 0.0, 0.0)
                return {"clearance_m": 0.01, "action_count": len(actions)}

        environment = FakeSafeLiberoCase()
        rollout, reach = annotate_reach_rollout(
            environment,
            [[0.0] * 7 for _ in range(5)],
            target_name=TARGET_OBJECT_NAME,
            obstacle_name=OBSTACLE,
        )

        self.assertEqual(rollout, {"clearance_m": 0.01, "action_count": 5})
        self.assertAlmostEqual(reach["reach_progress_m"], 0.4)
        self.assertEqual(reach["executed_actions"], 5)
        self.assertEqual(environment.reset_calls, 2)

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
