from copy import deepcopy
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


class DistalBoundaryCapacityTests(unittest.TestCase):
    def test_config_freezes_same_state_boundary_and_ordered_gates(self):
        from main.multilink_ellipsoid.boundary_capacity import (
            BOUNDARY_CAPACITY_SCHEMA,
            load_boundary_capacity_config,
        )

        config = load_boundary_capacity_config(
            ROOT / "configs/vlsa_distal_boundary_capacity_e05.v1.json"
        )
        self.assertEqual(config["schema_version"], BOUNDARY_CAPACITY_SCHEMA)
        self.assertEqual(config["state"]["step"], 188)
        self.assertEqual(config["state"]["critical_constraint_name"], "L5_part_1")
        self.assertEqual(config["sampling"]["expected_grid_action_count"], 1000)
        self.assertEqual(config["sampling"]["boundary_band_m"], 0.005)
        self.assertEqual(config["sampling"]["gradient_anchor_safe_count"], 32)
        self.assertEqual(config["sampling"]["gradient_anchor_unsafe_count"], 32)
        self.assertEqual(
            config["training"]["arms"],
            ["boundary_margin", "boundary_margin_gradient"],
        )
        self.assertEqual(
            config["decision_gate"][
                "critical_gradient_cosine_similarity_minimum"
            ],
            0.8,
        )
        self.assertIn("not_state_generalization", config["claim_scope"])

    def test_config_rejects_post_registration_mutation(self):
        from main.multilink_ellipsoid.boundary_capacity import (
            load_boundary_capacity_config,
        )

        source = ROOT / "configs/vlsa_distal_boundary_capacity_e05.v1.json"
        mutated = json.loads(source.read_text())
        mutated["sampling"]["boundary_band_m"] = 0.006
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mutated.json"
            path.write_text(json.dumps(mutated))
            with self.assertRaisesRegex(ValueError, "sampling differs"):
                load_boundary_capacity_config(path)

    @unittest.skipUnless(HAS_NUMPY, "NumPy is allocation dependency")
    def test_grid_is_complete_unique_and_inside_clipped_trust_region(self):
        from main.multilink_ellipsoid.boundary_capacity import (
            load_boundary_capacity_config,
            structured_grid_actions,
        )

        config = load_boundary_capacity_config(
            ROOT / "configs/vlsa_distal_boundary_capacity_e05.v1.json"
        )
        nominal = np.asarray([0.8, -0.8, 0.0])
        lower, upper, actions = structured_grid_actions(nominal, config)
        self.assertTrue(np.allclose(lower, [0.3, -1.0, -0.5], rtol=0.0, atol=1e-15))
        self.assertTrue(np.allclose(upper, [1.0, -0.3, 0.5], rtol=0.0, atol=1e-15))
        self.assertEqual(len(actions), 1000)
        self.assertEqual(len({tuple(item) for item in actions}), 1000)
        self.assertTrue(all(np.all(item >= lower) and np.all(item <= upper) for item in actions))

    @unittest.skipUnless(HAS_NUMPY, "NumPy is allocation dependency")
    def test_anchor_selection_is_balanced_and_nearest_to_boundary(self):
        from main.multilink_ellipsoid.boundary_capacity import (
            load_boundary_capacity_config,
            select_gradient_anchor_indexes,
        )

        config = load_boundary_capacity_config(
            ROOT / "configs/vlsa_distal_boundary_capacity_e05.v1.json"
        )
        records = []
        for index in range(80):
            margin = (index + 1) * 1.0e-5
            if index >= 40:
                margin = -(index - 39) * 1.0e-5
            minimum = np.ones(7)
            minimum[1] = margin
            records.append(
                {
                    "grid_index": index,
                    "candidate_xyz": [0.0, 0.0, 0.0],
                    "minimum_substep_clearance_m": minimum.tolist(),
                }
            )
        selected = select_gradient_anchor_indexes(
            records, [-0.5] * 3, [0.5] * 3, config
        )
        self.assertEqual(len(selected), 64)
        self.assertEqual(sum(records[index]["minimum_substep_clearance_m"][1] >= 0.0 for index in selected), 32)
        self.assertEqual(sum(records[index]["minimum_substep_clearance_m"][1] < 0.0 for index in selected), 32)

    @unittest.skipUnless(HAS_NUMPY, "NumPy is allocation dependency")
    def test_grouped_split_never_separates_finite_difference_probes(self):
        from main.multilink_ellipsoid.boundary_capacity import (
            assign_grouped_splits,
            load_boundary_capacity_config,
        )

        config = load_boundary_capacity_config(
            ROOT / "configs/vlsa_distal_boundary_capacity_e05.v1.json"
        )
        records = []
        for category in (
            "boundary_safe",
            "boundary_unsafe",
            "far_safe",
            "far_unsafe",
        ):
            for index in range(8):
                for probe in range(3):
                    records.append(
                        {
                            "group_id": "%s_%d" % (category, index),
                            "category": category,
                            "probe": probe,
                        }
                    )
        split = assign_grouped_splits(records, config)
        self.assertEqual(len(split), 32)
        self.assertEqual(
            set(split.values()),
            {"train", "validation", "test", "excluded_far"},
        )
        self.assertTrue(
            all(
                value == "excluded_far"
                for group, value in split.items()
                if group.startswith("far_")
            )
        )

    @unittest.skipUnless(HAS_NUMPY, "NumPy is allocation dependency")
    def test_union_support_gap_reports_obstacle_witness(self):
        from main.multilink_ellipsoid.geometry import Ellipsoid
        from main.multilink_ellipsoid.obstacle_primitives import (
            minimum_union_support_gap_witnesses,
        )

        robot = [
            Ellipsoid(
                center=np.zeros(3),
                rotation=np.eye(3),
                semiaxes_m=[0.1] * 3,
            )
        ]
        obstacles = [
            Ellipsoid(
                center=[1.0, 0.0, 0.0],
                rotation=np.eye(3),
                semiaxes_m=[0.1] * 3,
            ),
            Ellipsoid(
                center=[0.3, 0.0, 0.0],
                rotation=np.eye(3),
                semiaxes_m=[0.05] * 3,
            ),
        ]
        gaps, witnesses = minimum_union_support_gap_witnesses(robot, obstacles)
        self.assertAlmostEqual(float(gaps[0]), 0.15)
        self.assertEqual(int(witnesses[0]), 1)

    def test_two_stage_allocations_enforce_no_training_then_paired_H100_test(self):
        dataset = (
            ROOT / "slurm/distal_boundary_capacity_dataset_e05.sbatch"
        ).read_text()
        experiment = (
            ROOT / "slurm/distal_boundary_capacity_e05.sbatch"
        ).read_text()
        self.assertIn("--gres=gpu:1", dataset)
        self.assertIn("H100", dataset)
        self.assertIn("collect_distal_boundary_capacity_e05.py", dataset)
        self.assertIn("validate_distal_boundary_capacity_dataset_e05.py", dataset)
        self.assertNotIn("OPENPI_PYTHON", dataset)
        self.assertNotIn("train_distal_boundary_capacity_e05.py", dataset)
        self.assertIn("OPENPI_PYTHON", experiment)
        self.assertIn("train_distal_boundary_capacity_e05.py", experiment)
        self.assertIn("evaluate_distal_boundary_capacity_e05.py", experiment)
        self.assertIn("validate_distal_boundary_capacity_e05.py", experiment)
        self.assertIn("DATASET_VALIDATION", experiment)

    def test_projection_is_seven_row_and_exactly_verified(self):
        evaluator = (
            ROOT / "scripts/evaluate_distal_boundary_capacity_e05.py"
        ).read_text()
        validator = (
            ROOT / "scripts/validate_distal_boundary_capacity_e05.py"
        ).read_text()
        self.assertIn("_projection_audit", evaluator)
        self.assertIn("boundary_margin_gradient", evaluator)
        self.assertIn("projected_exact_raw_safe", validator)
        self.assertIn("projected_exact_proxy_safe", validator)
        self.assertIn("critical_gradient_cosine_mean", validator)
        self.assertIn("same_state_claim_scope_only", validator)


if __name__ == "__main__":
    unittest.main()
