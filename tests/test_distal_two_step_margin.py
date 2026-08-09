import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


class DistalTwoStepMarginTests(unittest.TestCase):
    def test_grouped_split_and_e05_holdout_are_frozen(self):
        from main.multilink_ellipsoid.two_step_margin import (
            load_selected_manifest,
            load_two_step_config,
        )

        config = load_two_step_config(
            ROOT / "configs/vlsa_distal_two_step_margin_moka10.v1.json"
        )
        rows = load_selected_manifest(
            ROOT / "manifests/vlsa_distal_boundary_generalization_moka10.v1.jsonl",
            config,
        )
        self.assertEqual(sum(item["split"] == "train" for item in rows), 6)
        self.assertEqual(sum(item["split"] == "validation" for item in rows), 1)
        self.assertEqual(sum(item["split"] == "test" for item in rows), 3)
        self.assertEqual(
            next(
                item
                for item in rows
                if item["case_id"] == "vlsa-t1-goal-ii-t0-e05"
            )["split"],
            "test",
        )
        self.assertEqual(config["sampling"]["second_action"], "immutable_nominal")

    @unittest.skipUnless(HAS_NUMPY, "NumPy is an allocation dependency")
    def test_grid_and_feature_shapes_keep_only_first_action_variable(self):
        from main.multilink_ellipsoid.two_step_margin import (
            GLOBAL_FEATURE_NAMES,
            PAIR_FEATURE_NAMES,
            feature_vectors,
            grid_actions,
            load_two_step_config,
        )

        config = load_two_step_config(
            ROOT / "configs/vlsa_distal_two_step_margin_moka10.v1.json"
        )
        lower, upper, actions = grid_actions([0.8, -0.8, 0.0], config)
        self.assertEqual(len(actions), 512)
        self.assertTrue(np.allclose(lower, [0.3, -1.0, -0.5]))
        self.assertTrue(np.allclose(upper, [1.0, -0.3, 0.5]))
        context = {
            "global_prefix": np.arange(len(GLOBAL_FEATURE_NAMES) - 9),
            "pair_prefixes": np.arange(7 * (len(PAIR_FEATURE_NAMES) - 9)).reshape(
                7, -1
            ),
        }
        global_feature, pair_features = feature_vectors(
            context, [0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]
        )
        self.assertEqual(global_feature.shape, (len(GLOBAL_FEATURE_NAMES),))
        self.assertEqual(pair_features.shape, (7, len(PAIR_FEATURE_NAMES)))
        self.assertTrue(np.allclose(global_feature[-6:-3], [0.4, 0.5, 0.6]))
        self.assertTrue(np.allclose(pair_features[:, -3:], [0.7, 0.8, 0.9]))

    @unittest.skipUnless(HAS_NUMPY, "NumPy is an allocation dependency")
    def test_chunk_summary_uses_worst_constraint_over_both_steps(self):
        from main.multilink_ellipsoid.two_step_margin import summarize_chunk

        transitions = []
        for chunk_step, base in enumerate((0.01, -0.002)):
            transitions.append(
                {
                    "minimum_substep_clearance_m": [base + index * 0.001 for index in range(7)],
                    "minimum_substep_witnesses": [
                        {"substep_index": index, "obstacle_primitive_index": index + chunk_step}
                        for index in range(7)
                    ],
                    "raw_protected_contact_count": 0,
                    "maximum_within_step_obstacle_l1_displacement_m": 0.0,
                    "next_state_sha256": "b" if chunk_step else "a",
                    "env_step_wall_seconds": 0.01,
                }
            )
        summary = summarize_chunk({"transitions": transitions})
        self.assertAlmostEqual(summary["minimum_substep_clearance_m"][0], -0.002)
        self.assertEqual(summary["minimum_substep_witnesses"][0]["chunk_step"], 1)
        self.assertFalse(summary["D_opt_proxy_safe"])
        self.assertTrue(summary["D_sim_raw_safe"])

    def test_mutated_horizon_contract_is_rejected(self):
        from main.multilink_ellipsoid.two_step_margin import load_two_step_config

        source = ROOT / "configs/vlsa_distal_two_step_margin_moka10.v1.json"
        config = json.loads(source.read_text())
        config["sampling"]["second_action"] = "optimized"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "sampling differs"):
                load_two_step_config(path)

    def test_h100_protocol_keeps_dataset_gate_before_training(self):
        dataset = (
            ROOT / "slurm/distal_two_step_margin_dataset_moka10.sbatch"
        ).read_text()
        experiment = (
            ROOT / "slurm/distal_two_step_margin_moka10.sbatch"
        ).read_text()
        self.assertIn("H100", dataset)
        self.assertIn("collect_distal_two_step_margin_moka10.py", dataset)
        self.assertNotIn("train_distal_two_step_margin_moka10.py", dataset)
        self.assertIn("train_distal_two_step_margin_moka10.py", experiment)
        self.assertIn("evaluate_distal_two_step_margin_moka10.py", experiment)
        self.assertIn("validate_distal_two_step_margin_moka10.py", experiment)

    def test_closed_loop_has_no_online_clone_oracle(self):
        source = (
            ROOT / "scripts/evaluate_distal_two_step_margin_moka10.py"
        ).read_text()
        closed_loop = source.split("def _closed_loop_e05", 1)[1].split(
            "def main", 1
        )[0]
        self.assertIn("project_first_action", closed_loop)
        self.assertIn("_measured_env_step", closed_loop)
        self.assertNotIn("rollout_chunk", closed_loop)
        self.assertIn('"online_cloned_simulator_oracle_used": False', closed_loop)


if __name__ == "__main__":
    unittest.main()
