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


class DistalFullBoundRecoveryTests(unittest.TestCase):
    def test_registered_diagnostic_is_single_state_and_not_closed_loop(self):
        from main.multilink_ellipsoid.full_bound_recovery import load_full_bound_config

        config = load_full_bound_config(
            ROOT / "configs/vlsa_distal_full_bound_recovery_e05.v1.json"
        )
        self.assertEqual(config["case_id"], "vlsa-t1-goal-ii-t0-e05")
        self.assertEqual(config["source"]["action_step"], 186)
        self.assertEqual(config["geometry"]["constraint_count"], 7)
        self.assertEqual(config["optimizer"]["bounds"], "full_normalized_translation_box_minus1_plus1")
        self.assertEqual(config["decision_gate"]["closed_loop"], "not_authorized_by_this_single_state_diagnostic")

    @unittest.skipUnless(HAS_NUMPY, "NumPy is an allocation dependency")
    def test_full_bound_population_contains_nominal_stop_reverse_and_cube(self):
        from main.multilink_ellipsoid.full_bound_recovery import full_bound_candidates, load_full_bound_config

        config = load_full_bound_config(
            ROOT / "configs/vlsa_distal_full_bound_recovery_e05.v1.json"
        )
        nominal = np.asarray([-0.004, 0.427, -0.694])
        records = full_bound_candidates(nominal, config)
        self.assertEqual(len(records), 731)
        values = {tuple(item["xyz"].tolist()) for item in records}
        self.assertIn((-1.0, -1.0, -1.0), values)
        self.assertIn((1.0, 1.0, 1.0), values)
        self.assertIn((0.0, 0.0, 0.0), values)
        self.assertIn(tuple(nominal.tolist()), values)
        self.assertIn(tuple((-nominal).tolist()), values)

    def test_mutated_bounds_are_rejected(self):
        from main.multilink_ellipsoid.full_bound_recovery import load_full_bound_config

        source = ROOT / "configs/vlsa_distal_full_bound_recovery_e05.v1.json"
        config = json.loads(source.read_text())
        config["action_space"]["action_limit"] = 0.5
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "action space differs"):
                load_full_bound_config(path)

    def test_h100_job_uses_exact_box_and_has_no_training_or_policy_server(self):
        source = (ROOT / "slurm/distal_full_bound_recovery_e05.sbatch").read_text()
        evaluator = (ROOT / "scripts/evaluate_distal_full_bound_recovery_e05.py").read_text()
        self.assertIn("H100", source)
        self.assertIn("evaluate_distal_full_bound_recovery_e05.py", source)
        self.assertNotIn("train_", source)
        self.assertNotIn("WebsocketClientPolicy", evaluator)
        self.assertIn("SubstepEightConstraintProbe", evaluator)
        self.assertIn("obstacle_primitive_union=exact_boxes", evaluator)
        self.assertIn("MultiConstraintQp", evaluator)
        self.assertIn("exact_clone_execution_match", evaluator)


if __name__ == "__main__":
    unittest.main()
