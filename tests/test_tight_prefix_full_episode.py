import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from main.multilink_ellipsoid.tight_prefix_full_episode import (
    RESULT_SCHEMA, load_config, payload_sha256,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/vlsa_tight_prefix_full_episode.v1.json"


class TightPrefixFullEpisodeTest(unittest.TestCase):
    def test_config_freezes_minimum_risk_from_query_zero(self):
        config = load_config(CONFIG)
        self.assertEqual(config["method"]["start_step"], 0)
        self.assertEqual(
            config["method"]["selection"],
            "minimum_predicted_primary_risk",
        )
        self.assertEqual(config["method"]["primary_rows"], list(range(8)))
        self.assertTrue(config["method"]["always_execute_selected"])
        self.assertFalse(config["forbidden"]["simulator_candidate_rollout"])

    def test_config_rejects_QP(self):
        value = json.loads(CONFIG.read_text())
        value["forbidden"]["learned_QP"] = True
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "forbidden"):
                load_config(path)

    def test_evaluator_queries_live_policy_and_never_rolls_candidates(self):
        source = (
            ROOT / "scripts/evaluate_tight_prefix_full_episode.py"
        ).read_text()
        self.assertIn("select_minimum_risk", source)
        self.assertIn("self.client.infer(request)", source)
        self.assertIn('"candidate_simulator_rollout_count": 0', source)
        self.assertNotIn("select_safe_or_abstain", source)

    def test_validator_keeps_single_case_scope(self):
        from scripts.validate_tight_prefix_full_episode import validate

        episode = {
            "complete_executed_actions_sha256": "a", "action_count": 5,
            "selection_count": 1, "selection_history_sha256": "b",
            "selection_history": [{
                "candidate_count": 13, "selected_candidate": "nominal",
                "abstained": False,
            }],
            "abstention_count": 0, "terminal_reason": "native_task_success",
            "native_task_success": True, "native_task_success_step": 4,
            "timeout": False, "raw_robot_contact_pass": True,
            "first_raw_robot_contact": None, "robot_contact_sample_count": 0,
            "robot_contact_sample_count_by_group": {},
            "robot_contact_events_sha256": "c", "paper_CAR_pass": True,
            "first_paper_CAR_step": None,
            "maximum_active_obstacle_l1_displacement_m": 0.0,
            "collision_free_task_success": True,
            "terminal_dynamic_state_sha256": "d",
            "goal_progress_summary_sha256": "e",
        }
        view = {"fixed": True}
        producer = {
            "schema_version": RESULT_SCHEMA, "status": "complete",
            "replica": "producer", "case_id": "case", "episode": episode,
            "scientific_view": view,
        }
        replay = dict(producer)
        replay["replica"] = "replay"
        for value in (producer, replay):
            value["result_payload_sha256"] = payload_sha256(
                value, "result_payload_sha256"
            )
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "producer.json"
            r = Path(directory) / "replay.json"
            p.write_text(json.dumps(producer))
            r.write_text(json.dumps(replay))
            result = validate(p, r)
        self.assertTrue(result["collision_free_task_success"])
        self.assertFalse(result["population_claim_authorized"])

    def test_terminal_scorer_accepts_ten_explicit_rows(self):
        try:
            import numpy as np
        except ModuleNotFoundError:
            self.skipTest("desktop Python lacks optional NumPy")
        from main.multilink_ellipsoid.terminal_branching import (
            FROZEN_CANDIDATE_NAMES, score_terminal_bank,
        )

        ordinary = np.zeros((10, 7), dtype=np.float64)
        bank = np.zeros((13, 10, 7), dtype=np.float64)

        def feature(_case, _candidate, row, **kwargs):
            self.assertEqual(tuple(kwargs["model_rows"]), tuple(range(10)))
            return [float(row)] + [0.0] * 6

        with mock.patch(
            "main.multilink_ellipsoid.compact_safety_coordinate_q.safety_coordinate_feature",
            side_effect=feature,
        ), mock.patch(
            "main.multilink_ellipsoid.compact_selector_ablation.predict_serialized_mlp_float32",
            side_effect=lambda values, _state: [[row[0]] for row in values],
        ):
            result = score_terminal_bank(
                exact_case={}, ordinary_terminal_actions=ordinary,
                terminal_action_bank=bank,
                candidate_names=FROZEN_CANDIDATE_NAMES,
                state_payload={}, primary_rows=range(8),
                model_rows=range(10),
            )
        self.assertEqual(len(result["records"][0]["predicted_by_row"]), 10)


if __name__ == "__main__":
    unittest.main()
