import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from main.multilink_ellipsoid.tight_prefix_full_episode import payload_sha256
from main.multilink_ellipsoid.tight_prefix_nominal_first_full_episode import (
    DETOUR_CANDIDATE_NAMES,
    RESULT_SCHEMA,
    detour_and_rejoin_candidates,
    load_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/vlsa_tight_prefix_nominal_first_full_episode.v1.json"


class TightPrefixNominalFirstFullEpisodeTest(unittest.TestCase):
    def test_config_freezes_warning_margin_and_no_retraining(self):
        config = load_config(CONFIG)
        method = config["method"]
        self.assertEqual(
            method["warning"]["normalized_radial_slack_strictly_below"],
            1.0,
        )
        self.assertEqual(method["acceptance"]["margin"], 0.1421400248048467)
        self.assertEqual(method["candidate_names"], list(DETOUR_CANDIDATE_NAMES))
        self.assertFalse(config["forbidden"]["model_training"])
        self.assertFalse(config["forbidden"]["data_collection"])

    def test_detours_are_bounded_and_endpoint_preserving(self):
        try:
            import numpy as np
        except ModuleNotFoundError:
            self.skipTest("desktop Python lacks optional NumPy")
        nominal = np.asarray([
            [0.8, -0.7, 0.2, 0.1, 0.2, 0.3, -1.0],
            [0.2, 0.1, -0.4, 0.1, 0.2, 0.3, -1.0],
            [0.0, 0.0, 0.0, 0.1, 0.2, 0.3, -1.0],
            [-0.2, 0.3, 0.4, 0.1, 0.2, 0.3, -1.0],
            [-0.8, 0.7, -0.2, 0.1, 0.2, 0.3, -1.0],
        ])
        frame = {
            "normal": [1.0, 0.0, 0.0],
            "tangent_up": [0.0, 0.0, 1.0],
            "tangent_side": [0.0, -1.0, 0.0],
        }
        rows = detour_and_rejoin_candidates(nominal, frame)
        self.assertEqual([row["name"] for row in rows], list(DETOUR_CANDIDATE_NAMES))
        for row in rows:
            actions = np.asarray(row["actions"])
            correction = actions[:, :3] - nominal[:, :3]
            self.assertLessEqual(float(np.max(np.abs(actions[:, :3]))), 1.0)
            self.assertLessEqual(float(np.max(np.abs(np.sum(correction, axis=0)))), 1e-12)
            self.assertTrue(row["endpoint_preserved"])
            self.assertTrue(np.array_equal(actions[:, 3:], nominal[:, 3:]))

    def test_terminal_scorer_accepts_registered_detour_names(self):
        try:
            import numpy as np
        except ModuleNotFoundError:
            self.skipTest("desktop Python lacks optional NumPy")
        from main.multilink_ellipsoid.terminal_branching import score_terminal_bank

        ordinary = np.zeros((10, 7), dtype=np.float64)
        bank = np.zeros((13, 10, 7), dtype=np.float64)
        with mock.patch(
            "main.multilink_ellipsoid.compact_safety_coordinate_q.safety_coordinate_feature",
            side_effect=lambda _case, _candidate, row, **_kwargs: [float(row)] + [0.0] * 6,
        ), mock.patch(
            "main.multilink_ellipsoid.compact_selector_ablation.predict_serialized_mlp_float32",
            side_effect=lambda values, _state: [[row[0]] for row in values],
        ):
            result = score_terminal_bank(
                exact_case={}, ordinary_terminal_actions=ordinary,
                terminal_action_bank=bank,
                candidate_names=DETOUR_CANDIDATE_NAMES,
                required_candidate_names=DETOUR_CANDIDATE_NAMES,
                state_payload={}, primary_rows=range(8), model_rows=range(10),
            )
        self.assertEqual(result["candidate_count"], 13)

    def test_evaluator_is_warning_gated_and_has_no_candidate_rollout(self):
        source = (
            ROOT / "scripts/evaluate_tight_prefix_nominal_first_full_episode.py"
        ).read_text()
        self.assertIn("if not warning_triggered", source)
        self.assertIn("detour_and_rejoin_candidates", source)
        self.assertIn('"candidate_simulator_rollout_count": 0', source)
        self.assertNotIn("minimum_predicted_primary_risk", source)

    def test_validator_accepts_exact_synthetic_replay(self):
        from scripts.validate_tight_prefix_nominal_first_full_episode import validate

        selection = {
            "warning_triggered": False, "critic_invoked": False,
            "candidate_count": 1, "selected_candidate": "nominal",
            "abstained": False, "intervened": False,
            "selected_effective_correction_l2_action": 0.0,
        }
        episode = {
            "selection_history": [selection], "selection_count": 1,
            "action_count": 5,
        }
        view = {
            "case_id": "case", "terminal_reason": "native_task_success",
            "action_count": 5, "selection_count": 1,
            "raw_robot_contact_pass": True, "paper_CAR_pass": True,
            "native_task_success": True, "collision_free_task_success": True,
        }
        producer = {
            "schema_version": RESULT_SCHEMA, "status": "complete",
            "scientific_result": True, "replica": "producer",
            "episode": episode, "scientific_view": view,
        }
        replay = dict(producer)
        replay["replica"] = "replay"
        for value in (producer, replay):
            value["result_payload_sha256"] = payload_sha256(
                value, "result_payload_sha256"
            )
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "p.json"
            r = Path(directory) / "r.json"
            p.write_text(json.dumps(producer))
            r.write_text(json.dumps(replay))
            result = validate(p, r)
        self.assertTrue(result["collision_free_task_success"])
        self.assertEqual(result["intervention_count"], 0)


if __name__ == "__main__":
    unittest.main()
