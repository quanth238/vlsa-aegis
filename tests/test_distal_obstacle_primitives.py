from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


class DistalObstaclePrimitiveTests(unittest.TestCase):
    def test_config_retains_base_oracle_and_changes_only_obstacle_geometry(self):
        from main.multilink_ellipsoid.obstacle_primitives import (
            OBSTACLE_PRIMITIVE_SCHEMA,
            load_obstacle_primitive_config,
        )

        config = load_obstacle_primitive_config(
            ROOT / "configs/vlsa_distal_oracle_mesh_obstacle_e05.v1.json"
        )
        self.assertEqual(config["schema_version"], OBSTACLE_PRIMITIVE_SCHEMA)
        self.assertEqual(
            config["base_audit_config"]["config_file_sha256"],
            "c7019c176e0e8b6379cdb1b83e09d7129c1237a4f28ec2f3daba49a7b57ddc95",
        )
        self.assertEqual(
            config["obstacle_geometry"]["mesh_representation"],
            "one_certified_mvee_per_compiled_mujoco_collision_mesh",
        )
        self.assertIn("not_training", config["claim_scope"])

    def test_exact_box_config_binds_discovery_and_zero_inflation(self):
        from main.multilink_ellipsoid.obstacle_primitives import (
            EXACT_BOX_OBSTACLE_SCHEMA,
            load_obstacle_primitive_config,
        )

        config = load_obstacle_primitive_config(
            ROOT / "configs/vlsa_distal_oracle_exact_box_obstacle_e05.v1.json"
        )
        self.assertEqual(config["schema_version"], EXACT_BOX_OBSTACLE_SCHEMA)
        self.assertEqual(config["discovery_source"]["slurm_job_id"], "37163")
        self.assertEqual(config["discovery_source"]["obstacle_geom_count"], 15)
        self.assertEqual(
            config["obstacle_geometry"]["representation"],
            "exact_compiled_mujoco_oriented_boxes_without_inflation",
        )

    @unittest.skipUnless(HAS_NUMPY, "NumPy is allocation dependency")
    def test_exact_oriented_box_support_and_containment(self):
        from main.multilink_ellipsoid.obstacle_primitives import OrientedBox

        box = OrientedBox(
            center=[1.0, 2.0, 3.0],
            rotation=np.eye(3),
            half_size_m=[1.0, 2.0, 3.0],
            body_id=1,
            body_name="body",
            geom_id=2,
            geom_name="box",
            enclosure_certificate={"verified": True},
        )
        direction = np.asarray([1.0, 1.0, 0.0])
        self.assertAlmostEqual(box.support_radius(direction), 3.0 / np.sqrt(2.0))
        self.assertAlmostEqual(box.containment_value([1.0, 2.0, 3.0]), 0.0)
        self.assertAlmostEqual(box.containment_value([2.0, 4.0, 6.0]), 1.0)
        self.assertGreater(box.containment_value([2.01, 2.0, 3.0]), 1.0)

    @unittest.skipUnless(HAS_NUMPY, "NumPy is allocation dependency")
    def test_exact_box_is_tighter_than_loewner_ellipsoid(self):
        from main.multilink_ellipsoid.geometry import Ellipsoid
        from main.multilink_ellipsoid.obstacle_primitives import OrientedBox

        half = np.asarray([0.01, 0.02, 0.03])
        box = OrientedBox(
            center=np.zeros(3),
            rotation=np.eye(3),
            half_size_m=half,
            body_id=1,
            body_name="body",
            geom_id=2,
            geom_name="box",
            enclosure_certificate={"verified": True},
        )
        loewner = Ellipsoid(
            center=np.zeros(3),
            rotation=np.eye(3),
            semiaxes_m=np.sqrt(3.0) * half,
        )
        directions = [
            np.asarray([1.0, 0.0, 0.0]),
            np.asarray([0.0, 1.0, 0.0]),
            np.asarray([0.0, 0.0, 1.0]),
            np.asarray([1.0, 1.0, 1.0]),
            np.asarray([1.0, -2.0, 3.0]),
        ]
        for direction in directions:
            self.assertLessEqual(
                box.support_radius(direction),
                loewner.support_radius(direction) + 1.0e-15,
            )
        self.assertLess(box.support_radius(directions[0]), loewner.support_radius(directions[0]))

    @unittest.skipUnless(HAS_NUMPY, "NumPy is allocation dependency")
    def test_union_clearance_is_minimum_over_all_pair_gaps(self):
        from main.multilink_ellipsoid.barrier import support_gap
        from main.multilink_ellipsoid.geometry import Ellipsoid
        from main.multilink_ellipsoid.obstacle_primitives import (
            minimum_union_support_gaps,
        )

        robot = [
            Ellipsoid(
                center=[0.0, 0.0, 0.0],
                rotation=np.eye(3),
                semiaxes_m=[0.1, 0.1, 0.1],
            ),
            Ellipsoid(
                center=[0.0, 1.0, 0.0],
                rotation=np.eye(3),
                semiaxes_m=[0.1, 0.1, 0.1],
            ),
        ]
        obstacles = [
            Ellipsoid(
                center=[0.4, 0.0, 0.0],
                rotation=np.eye(3),
                semiaxes_m=[0.1, 0.1, 0.1],
            ),
            Ellipsoid(
                center=[0.0, 0.75, 0.0],
                rotation=np.eye(3),
                semiaxes_m=[0.05, 0.05, 0.05],
            ),
        ]
        result = minimum_union_support_gaps(robot, obstacles)
        expected = np.asarray(
            [
                min(support_gap(robot[0], item) for item in obstacles),
                min(support_gap(robot[1], item) for item in obstacles),
            ]
        )
        self.assertTrue(np.array_equal(result, expected))
        self.assertAlmostEqual(float(result[0]), 0.2)
        self.assertAlmostEqual(float(result[1]), 0.1)

    def test_evaluator_and_allocation_are_explicitly_opt_in(self):
        evaluator = (
            ROOT / "scripts/evaluate_distal_oracle_affine_e05.py"
        ).read_text()
        allocation = (
            ROOT / "slurm/distal_oracle_mesh_obstacle_e05.sbatch"
        ).read_text()
        self.assertIn("--obstacle-primitive-config", evaluator)
        self.assertIn("ConservativeObstaclePrimitiveUnion", evaluator)
        self.assertIn("--gres=gpu:1", allocation)
        self.assertIn("H100", allocation)
        self.assertIn("status --porcelain=v1 --untracked-files=all", allocation)
        self.assertIn("tests.test_distal_obstacle_primitives", allocation)
        self.assertIn("validate_distal_oracle_mesh_obstacle_e05.py", allocation)

    def test_exact_box_evaluator_and_allocation_are_hash_bound(self):
        evaluator = (
            ROOT / "scripts/evaluate_distal_oracle_affine_e05.py"
        ).read_text()
        allocation = (
            ROOT / "slurm/distal_oracle_exact_box_obstacle_e05.sbatch"
        ).read_text()
        self.assertIn("ExactObstacleBoxUnion", evaluator)
        self.assertIn("--obstacle-discovery-result", evaluator)
        self.assertIn("--gres=gpu:1", allocation)
        self.assertIn("H100", allocation)
        self.assertIn("OBSTACLE_DISCOVERY_RESULT", allocation)
        self.assertIn("3db37092b2c5573e", allocation)
        self.assertIn("validate_distal_oracle_exact_box_obstacle_e05.py", allocation)

    def test_contact_gate_requires_source_obstacle_primitive(self):
        source = (
            ROOT / "main/multilink_ellipsoid/oracle_affine.py"
        ).read_text()
        self.assertIn("source_obstacle_contact_point_covered", source)
        self.assertIn(
            "contacted obstacle geom lacks exactly one certified primitive",
            source,
        )


if __name__ == "__main__":
    unittest.main()
