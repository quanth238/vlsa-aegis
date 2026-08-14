import json
from pathlib import Path
import unittest


class FactorizedBoundaryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1]

    def test_config_and_manifest_are_frozen(self):
        from main.multilink_ellipsoid.l5_factorized_boundary import (
            load_config, load_manifest,
        )

        config = load_config(
            self.root / "configs/vlsa_distal_l5_factorized_boundary.v1.json",
            self.root,
        )
        rows = load_manifest(
            self.root / "manifests/vlsa_distal_l5_factorized_boundary.v1.jsonl",
            config,
        )
        self.assertEqual(len(rows), 12)
        self.assertEqual(
            [row["factorized_split"] for row in rows].count("train"), 8
        )
        self.assertEqual(
            [row["factorized_split"] for row in rows].count("validation"), 4
        )
        self.assertEqual({row["logical_task_index"] for row in rows}, {0, 2})
        self.assertFalse(any(row["episode_index"] in (31, 36) for row in rows))

    def test_phase_coverage_keeps_components_separate(self):
        from main.multilink_ellipsoid.l5_factorized_boundary import phase_coverage

        def candidate(order, prefix, backup, combined, safe):
            return {
                "name": "c%d" % order, "order": order,
                "candidate_prefix_risk": prefix + [-0.1] * 5,
                "backup_risk": None if backup is None else backup + [-0.1] * 5,
                "combined_risk": combined + [-0.1] * 5,
                "exact_safe": safe, "terminal_status": "SAFE_TERMINAL" if safe else "UNSAFE_CONTACT_OR_CAR",
                "residual_binding": {"applied_residual_l2_action": float(order)},
            }

        result = {"candidates": [
            candidate(0, [-0.002, -0.003], [-0.004, -0.004], [-0.002, -0.003], True),
            candidate(1, [0.002, -0.003], [-0.004, -0.004], [0.002, -0.003], False),
            candidate(2, [-0.004, -0.004], [-0.003, -0.002], [-0.003, -0.002], True),
            candidate(3, [-0.004, -0.004], [0.003, -0.002], [0.003, -0.002], False),
            candidate(4, [-0.001, -0.002], [-0.003, -0.003], [-0.001, -0.002], True),
        ]}
        report = phase_coverage(result)
        self.assertTrue(report["known_nominal"])
        self.assertEqual(report["known_response_count"], 4)
        self.assertTrue(report["prefix_dominated_boundary"])
        self.assertTrue(report["backup_dominated_boundary"])

    def test_slurm_collection_stays_on_h100(self):
        script = (
            self.root / "slurm/collect_distal_l5_factorized_boundary.sbatch"
        ).read_text(encoding="utf-8")
        self.assertIn("#SBATCH --gres=gpu:1", script)
        self.assertIn("#SBATCH --array=0-11%2", script)
        self.assertIn("requires exactly one H100", script)
        self.assertIn("--mode aegis", script)
        self.assertIn("collect_distal_l5_factorized_risk.py", script)
        summary = (
            self.root / "slurm/summarize_distal_l5_factorized_boundary.sbatch"
        ).read_text(encoding="utf-8")
        self.assertIn("#SBATCH --gres=gpu:1", summary)
        self.assertIn("H100", summary)


if __name__ == "__main__":
    unittest.main()
