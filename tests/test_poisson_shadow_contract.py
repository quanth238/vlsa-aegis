from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from main.poisson_fullbody.shadow_replay import (
    ReplayContractError,
    load_historical_action_replay,
)


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")


def sha(value):
    return hashlib.sha256(value).hexdigest()


def result_fixture():
    raw_actions = [
        [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, -1.0],
        [0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 1.0],
    ]
    state_hashes = ["4" * 64, "5" * 64]
    actions = []
    for index, action in enumerate(raw_actions):
        actions.append(
            {
                "step": index,
                "executed": action,
                "env_step_input": action,
                "reward": float(index),
                "done": index == 1,
                "goal_progress": {
                    "inert": True,
                    "values": [index == 1],
                    "simulator_state_sha256_before": state_hashes[index],
                    "simulator_state_sha256_after": state_hashes[index],
                },
            }
        )
    payload = {
        "case_id": "case-a",
        "arm": "pi05_plus_aegis_translational",
        "status": "complete",
        "scientific_result": True,
        "actions": actions,
        "action_invariance_ledger": {
            "action_count": len(actions),
            "executed_sequence_sha256": sha(canonical([row["executed"] for row in actions])),
            "policy_query_count": 1,
            "policy_query_schedule_sha256": "6" * 64,
        },
        "pairing": {
            "settled_simulator_state_sha256": "1" * 64,
            "initial_observation_sha256": "2" * 64,
            "settled_active_obstacle_position_sha256": "7" * 64,
            "policy_noise_schedule_sha256": "3" * 64,
        },
        "metrics": {
            "task_success": True,
            "paper_collision": True,
            "collision_first_step": 1,
        },
        "terminal_observation": {
            "after_executed_action_count": len(actions),
            "simulator_state_sha256": state_hashes[-1],
        },
    }
    payload["result_payload_sha256"] = sha(canonical(payload))
    return payload


class HistoricalShadowContractTest(unittest.TestCase):
    def write(self, root, result):
        path = Path(root) / "result.json"
        path.write_text(json.dumps(result, sort_keys=True), encoding="utf-8")
        return path

    def test_loads_exact_executed_actions_and_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            replay = load_historical_action_replay(
                self.write(directory, result_fixture()), expected_case_id="case-a"
            )
        self.assertEqual(len(replay.actions), 2)
        self.assertEqual(len(replay.steps), 2)
        self.assertEqual(replay.actions[0][:3], (0.1, 0.2, 0.3))
        self.assertEqual(replay.steps[-1].simulator_state_sha256, "5" * 64)
        self.assertEqual(replay.source_policy_query_count, 1)
        self.assertEqual(replay.source_policy_query_schedule_sha256, "6" * 64)
        self.assertEqual(
            replay.settled_active_obstacle_position_sha256, "7" * 64
        )
        self.assertEqual(
            replay.provenance()["replay_semantics"],
            "exact_actions[*].executed_not_nominal_raw",
        )

    def test_tampered_action_or_payload_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            result = result_fixture()
            result["actions"][0]["executed"][0] = 0.9
            with self.assertRaisesRegex(ReplayContractError, "payload hash"):
                load_historical_action_replay(
                    self.write(directory, result), expected_case_id="case-a"
                )

    def test_wrong_arm_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ReplayContractError, "arm differs"):
                load_historical_action_replay(
                    self.write(directory, result_fixture()),
                    expected_case_id="case-a",
                    expected_arm="pi05_translational",
                )

    def test_nominal_action_cannot_substitute_for_executed(self):
        with tempfile.TemporaryDirectory() as directory:
            result = result_fixture()
            result["actions"][0]["env_step_input"] = [0.0] * 7
            result.pop("result_payload_sha256")
            result["result_payload_sha256"] = sha(canonical(result))
            with self.assertRaisesRegex(ReplayContractError, "env-step input differs"):
                load_historical_action_replay(
                    self.write(directory, result), expected_case_id="case-a"
                )


if __name__ == "__main__":
    unittest.main()
