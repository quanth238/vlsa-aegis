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


class DistalBoundaryGeneralizationTests(unittest.TestCase):
    def test_registered_split_is_task_group_disjoint_and_e05_is_test_only(self):
        from main.multilink_ellipsoid.boundary_generalization import load_generalization_config, load_selected_manifest

        config = load_generalization_config(ROOT / "configs/vlsa_distal_boundary_generalization_moka10.v1.json")
        rows = load_selected_manifest(ROOT / "manifests/vlsa_distal_boundary_generalization_moka10.v1.jsonl", config)
        by_group = {}
        for row in rows:
            by_group.setdefault(row["task_level_group_id"], set()).add(row["split"])
        self.assertTrue(all(len(values) == 1 for values in by_group.values()))
        self.assertEqual(next(row for row in rows if row["case_id"] == "vlsa-t1-goal-ii-t0-e05")["split"], "test")
        self.assertEqual(sum(row["split"] == "train" for row in rows), 6)
        self.assertEqual(sum(row["split"] == "validation" for row in rows), 1)
        self.assertEqual(sum(row["split"] == "test" for row in rows), 3)

    def test_manifest_hash_and_split_mutations_are_rejected(self):
        from main.multilink_ellipsoid.boundary_generalization import load_generalization_config, load_selected_manifest

        config = load_generalization_config(ROOT / "configs/vlsa_distal_boundary_generalization_moka10.v1.json")
        source = ROOT / "manifests/vlsa_distal_boundary_generalization_moka10.v1.jsonl"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mutated.jsonl"
            rows = [json.loads(line) for line in source.read_text().splitlines()]
            rows[-1]["split"] = "train"
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            with self.assertRaisesRegex(ValueError, "manifest hash differs"):
                load_selected_manifest(path, config)

    @unittest.skipUnless(HAS_NUMPY, "NumPy is an allocation dependency")
    def test_grid_and_worst_row_anchor_selection_are_balanced(self):
        from main.multilink_ellipsoid.boundary_generalization import grid_actions, load_generalization_config, select_balanced_anchors

        config = load_generalization_config(ROOT / "configs/vlsa_distal_boundary_generalization_moka10.v1.json")
        lower, upper, actions = grid_actions([0.8, -0.8, 0.0], config)
        self.assertEqual(len(actions), 512)
        self.assertTrue(np.allclose(lower, [0.3, -1.0, -0.5]))
        self.assertTrue(np.allclose(upper, [1.0, -0.3, 0.5]))
        records = []
        for index in range(40):
            minimum = np.ones(7)
            minimum[index % 7] = (index + 1) * 1e-5 if index < 20 else -(index - 19) * 1e-5
            records.append({"grid_index": index, "candidate_xyz": [0.0, 0.0, 0.0], "minimum_substep_clearance_m": minimum.tolist()})
        selected = select_balanced_anchors(records, [-0.5] * 3, [0.5] * 3, config)
        self.assertEqual(len(selected), 32)
        self.assertEqual(sum(min(records[index]["minimum_substep_clearance_m"]) >= 0 for index in selected), 16)
        self.assertEqual(sum(min(records[index]["minimum_substep_clearance_m"]) < 0 for index in selected), 16)

    def test_two_stage_h100_protocol_and_no_training_dataset_stage(self):
        dataset = (ROOT / "slurm/distal_boundary_generalization_dataset_moka10.sbatch").read_text()
        experiment = (ROOT / "slurm/distal_boundary_generalization_moka10.sbatch").read_text()
        self.assertIn("H100", dataset)
        self.assertIn("collect_distal_boundary_generalization_moka10.py", dataset)
        self.assertNotIn("OPENPI_PYTHON", dataset)
        self.assertIn("train_distal_boundary_generalization_moka10.py", experiment)
        self.assertIn("evaluate_distal_boundary_generalization_moka10.py", experiment)
        self.assertIn("OPENPI_PYTHON", experiment)

    def test_evaluator_requires_every_test_episode_and_exact_seven_row_projection(self):
        source = (ROOT / "scripts/evaluate_distal_boundary_generalization_moka10.py").read_text()
        collector = (ROOT / "scripts/collect_distal_boundary_generalization_moka10.py").read_text()
        self.assertIn("valid_seven_row_qp", source)
        self.assertIn("projected_exact_proxy_safe", source)
        self.assertIn("projected_exact_raw_safe", source)
        self.assertIn("all(value[\"projection_gate_pass\"]", source)
        self.assertIn("closed_loop_e05_authorized", source)
        self.assertIn("geometry_placeholder_archived", source)
        self.assertIn("geometry_placeholder_archived", collector)
        self.assertIn("clearances_use_live_exact_15_box_union", collector)


if __name__ == "__main__": unittest.main()
