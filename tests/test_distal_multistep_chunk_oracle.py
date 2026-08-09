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


NOMINAL = [
    [-0.00007061773729334232, 0.5512091908695699, -0.6401244209429624, 0.0, 0.0, 0.0, 0.9993441552317739],
    [-0.004188892039597155, 0.4273876839377881, -0.6937277880506518, 0.0, 0.0, 0.0, 1.0006715242265463],
    [0.0015781945169449354, 0.3435099053628126, -0.727207769165427, 0.0, 0.0, 0.0, 1.001],
    [0.0269566935275467, 0.3093378687316752, -0.7128850669997523, 0.0, 0.0, 0.0, 1.001],
    [-0.015180783976108104, 0.3081303287762227, -0.6966789255206464, 0.0, 0.0, 0.0, 1.001],
]


class DistalMultistepChunkOracleTests(unittest.TestCase):
    def test_registered_arms_keep_training_and_closed_loop_blocked(self):
        from main.multilink_ellipsoid.chunk_oracle import load_chunk_oracle_config

        config = load_chunk_oracle_config(
            ROOT / "configs/vlsa_distal_multistep_chunk_oracle_e05.v1.json"
        )
        self.assertEqual(config["source"]["start_action_step"], 185)
        self.assertEqual(
            [(item["horizon"], item["family"]) for item in config["arms"]],
            [(2, "one_shot"), (2, "distributed"), (5, "one_shot"), (5, "distributed")],
        )
        self.assertIn("not_authorized", config["decision_gate"]["neural_training"])
        self.assertIn("not_authorized", config["decision_gate"]["closed_loop"])

    @unittest.skipUnless(HAS_NUMPY, "NumPy is an allocation dependency")
    def test_one_shot_and_distributed_families_are_distinct_and_complete(self):
        from main.multilink_ellipsoid.chunk_oracle import distributed_chunks, load_chunk_oracle_config, one_shot_chunks

        config = load_chunk_oracle_config(
            ROOT / "configs/vlsa_distal_multistep_chunk_oracle_e05.v1.json"
        )
        for horizon in (2, 5):
            nominal = np.asarray(NOMINAL[:horizon], dtype=np.float64)
            one_shot = one_shot_chunks(nominal, config)
            distributed = distributed_chunks(nominal, config)
            self.assertEqual(len(one_shot), 731)
            self.assertEqual(len(distributed), 730)
            self.assertTrue(all(np.allclose(item["correction"][1:], 0.0) for item in one_shot))
            self.assertTrue(all(np.allclose(np.sum(item["correction"], axis=0), 0.0, atol=1e-12) for item in distributed))
            self.assertTrue(any(np.allclose(item["correction"], 0.0) for item in distributed))
            self.assertTrue(all(np.max(np.abs(item["chunk"][:, :3])) <= 1.0 + 1e-12 for item in distributed))

    def test_mutated_endpoint_weights_are_rejected(self):
        from main.multilink_ellipsoid.chunk_oracle import load_chunk_oracle_config

        source = ROOT / "configs/vlsa_distal_multistep_chunk_oracle_e05.v1.json"
        config = json.loads(source.read_text())
        config["candidate_family"]["five_step_residual_weights"][-1] = -0.5
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "candidate family differs"):
                load_chunk_oracle_config(path)

    def test_exact_chunk_job_has_no_training_or_policy_server(self):
        job = (ROOT / "slurm/distal_multistep_chunk_oracle_e05.sbatch").read_text()
        evaluator = (ROOT / "scripts/evaluate_distal_multistep_chunk_oracle_e05.py").read_text()
        oracle = (ROOT / "main/multilink_ellipsoid/oracle_affine.py").read_text()
        self.assertIn("H100", job)
        self.assertNotIn("train_", job)
        self.assertNotIn("WebsocketClientPolicy", evaluator)
        self.assertIn("obstacle_primitive_union=exact_boxes", evaluator)
        self.assertIn("rollout_chunk", evaluator)
        self.assertIn("def rollout_chunk", oracle)
        self.assertIn("sequential_same_cloned_env", oracle)


if __name__ == "__main__":
    unittest.main()
