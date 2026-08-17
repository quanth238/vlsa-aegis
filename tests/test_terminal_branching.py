from pathlib import Path
import json
import tempfile
import unittest
from unittest import mock

from tests.test_crfs_websocket_control import _load_server_module


ROOT = Path(__file__).resolve().parents[1]

class TerminalBranchingTests(unittest.TestCase):
    def _envelope(self):
        from main.multilink_ellipsoid.terminal_branching import (
            FROZEN_CANDIDATE_NAMES,
        )

        residuals = [[[0.0, 0.0, 0.0] for _ in range(10)]]
        for index in range(1, 13):
            row = [[0.0, 0.0, 0.0] for _ in range(10)]
            row[0][(index - 1) % 3] = 0.1 * (1 if index % 2 else -1)
            residuals.append(row)
        return {
            "schema_version": "crfs_terminal_branching.v1",
            "action_horizon": 10,
            "action_dimensions": [0, 1, 2],
            "candidate_names": list(FROZEN_CANDIDATE_NAMES),
            "candidate_output_residuals": residuals,
            "branch_after_euler_step": 8,
        }

    def test_config_freezes_terminal_scoring_semantics(self):
        from main.multilink_ellipsoid.terminal_branching import load_config

        config = load_config(
            ROOT / "configs/vlsa_distal_terminalized_late_flow.v1.json"
        )
        self.assertEqual(config["flow"]["branch_after_euler_step"], 8)
        self.assertEqual(config["flow"]["remaining_euler_steps"], 2)
        self.assertEqual(config["flow"]["candidate_count"], 13)
        self.assertEqual(
            config["scoring"]["critic_input"],
            "effective_terminal_five_action_chunk_after_clipping",
        )
        self.assertFalse(config["scoring"]["risk_scored_inside_sampler"])
        self.assertEqual(config["mechanism_case"]["source_case_index"], 11)
        self.assertEqual(
            config["paired_pilot"]["selected_arm_count_per_replica"], 3
        )

    def test_reserved_envelope_is_strict_and_removed(self):
        server = _load_server_module()
        observation, control = server._extract_crfs_control({
            "state": [1, 2, 3],
            "__crfs__": {
                "rng_seed": 19,
                "terminal_branching": self._envelope(),
            },
        })
        self.assertNotIn("__crfs__", observation)
        self.assertEqual(
            control["terminal_branching"]["branch_after_euler_step"], 8
        )
        self.assertEqual(len(control["terminal_branching"]["candidate_names"]), 13)

    def test_reserved_envelope_rejects_nonzero_nominal(self):
        server = _load_server_module()
        envelope = self._envelope()
        envelope["candidate_output_residuals"][0][0][0] = 0.01
        with self.assertRaises(ValueError):
            server._extract_crfs_control({
                "__crfs__": {"rng_seed": 19, "terminal_branching": envelope}
            })

    def test_model_completes_branches_after_one_injection(self):
        source = (ROOT / "openpi/src/openpi/models/pi0.py").read_text()
        start = source.index("def sample_actions_with_terminal_branches")
        end = source.index("def sample_actions_flow_step", start)
        method = source[start:end]
        self.assertIn("x_t + dt * velocity", method)
        self.assertIn("step_index + 1 == branch_step", method)
        self.assertIn("jnp.asarray(branch_after_euler_step", method)
        self.assertNotIn("int(branch_after_euler_step)", method)
        self.assertIn("value + offsets", method)
        self.assertIn("jax.lax.while_loop", method)
        self.assertIn("(1, self.action_horizon, self.action_dim)", method)
        self.assertIn("jnp.repeat(base_noise, branch_count", method)

    def test_collector_override_is_opt_in_and_sequential(self):
        source = (
            ROOT / "scripts/collect_distal_exact_group_boundary.py"
        ).read_text()
        self.assertIn("candidate_definitions_override", source)
        self.assertIn("selected_terminalized_live_policy_arms", source)
        self.assertIn("terminal inference definitions must execute sequentially alone", source)

    def test_policy_maps_residuals_with_scale_only(self):
        source = (ROOT / "openpi/src/openpi/policies/policy.py").read_text()
        start = source.index("def _infer_terminal_branches")
        end = source.index("def _prepare_flow_guidance", start)
        method = source[start:end]
        self.assertIn("offsets_model = residuals / scale", method)
        self.assertNotIn("residuals -", method)
        self.assertIn("self._sample_actions(", method)
        self.assertIn("ordinary_model, terminal_model[1:]", method)
        self.assertIn("batched_zero_branch_max_abs_model_difference", method)
        self.assertIn('"actions": terminal_output[0]', method)
        self.assertIn('"risk_scored_inside_sampler": False', method)

    def test_independent_validator_requires_exact_scientific_view(self):
        from main.multilink_ellipsoid.terminal_branching import (
            RESULT_SCHEMA,
            payload_sha256,
        )
        from scripts.validate_terminal_branch_sampler_canary import validate

        scientific_view = {"gates": {"parity": True}, "terminal_hash": "abc"}
        producer = {
            "schema_version": RESULT_SCHEMA,
            "status": "passing",
            "replica": "producer",
            "scientific_view": scientific_view,
        }
        replay = {
            "schema_version": RESULT_SCHEMA,
            "status": "passing",
            "replica": "replay",
            "scientific_view": scientific_view,
        }
        for value in (producer, replay):
            value["result_payload_sha256"] = payload_sha256(
                value, "result_payload_sha256"
            )
        with tempfile.TemporaryDirectory() as directory:
            producer_path = Path(directory) / "producer.json"
            replay_path = Path(directory) / "replay.json"
            producer_path.write_text(json.dumps(producer), encoding="utf-8")
            replay_path.write_text(json.dumps(replay), encoding="utf-8")
            result = validate(producer_path, replay_path)
        self.assertEqual(result["status"], "passing")
        self.assertTrue(result["checks"]["scientific_view_exact"])

    def test_terminal_scorer_uses_clipped_terminal_actions(self):
        try:
            import numpy as np
        except ModuleNotFoundError:
            self.skipTest("desktop Python lacks optional NumPy")

        from main.multilink_ellipsoid.terminal_branching import (
            FROZEN_CANDIDATE_NAMES,
            score_terminal_bank,
        )

        ordinary = np.zeros((10, 7), dtype=np.float64)
        bank = np.zeros((13, 10, 7), dtype=np.float64)
        bank[1, :5, 0] = 2.0
        bank[2, :5, 0] = -0.5

        def feature(_case, candidate, row, *, translation_scale):
            self.assertEqual(translation_scale, 0.05)
            return [candidate["source_executed_actions"][0][0], float(row)] + [0.0] * 5

        def predict(features, _state):
            # Candidate 2 has the lowest maximum over rows 0--4.
            return [[value[0] + 0.01 * value[1]] for value in features]

        with mock.patch(
            "main.multilink_ellipsoid.compact_safety_coordinate_q.safety_coordinate_feature",
            side_effect=feature,
        ), mock.patch(
            "main.multilink_ellipsoid.compact_selector_ablation.predict_serialized_mlp_float32",
            side_effect=predict,
        ):
            result = score_terminal_bank(
                exact_case={},
                ordinary_terminal_actions=ordinary,
                terminal_action_bank=bank,
                candidate_names=FROZEN_CANDIDATE_NAMES,
                state_payload={},
            )
        self.assertEqual(
            result["selected_candidate"], FROZEN_CANDIDATE_NAMES[2]
        )
        self.assertEqual(
            result["records"][1]["effective_first_five_actions"][0][0],
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
