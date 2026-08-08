import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


class DistalOracleAffineTests(unittest.TestCase):
    def test_config_freezes_one_false_safe_state_and_eight_constraints(self):
        from main.multilink_ellipsoid.oracle_affine import (
            ORACLE_AFFINE_SCHEMA,
            load_oracle_affine_config,
        )

        path = ROOT / "configs/vlsa_distal_oracle_affine_e05.v1.json"
        config = load_oracle_affine_config(path)
        self.assertEqual(config["schema_version"], ORACLE_AFFINE_SCHEMA)
        self.assertEqual(config["audit_step"], 192)
        self.assertEqual(config["false_safe_source"]["slurm_job_id"], "37109")
        self.assertEqual(config["protected_geometry"]["distal_constraint_count"], 7)
        self.assertEqual(config["protected_geometry"]["end_effector_constraint_count"], 1)
        self.assertEqual(config["affine_model"]["clearance_target_m"], 0.0)
        self.assertEqual(config["affine_model"]["trust_region_linf_action"], 0.5)

    @unittest.skipUnless(HAS_NUMPY, "NumPy is allocation dependency")
    def test_candidate_set_is_deterministic_unique_and_bounded(self):
        from main.multilink_ellipsoid.oracle_affine import (
            load_oracle_affine_config,
            oracle_candidate_xyz,
        )

        config = load_oracle_affine_config(
            ROOT / "configs/vlsa_distal_oracle_affine_e05.v1.json"
        )
        nominal = np.asarray([0.1, 0.3, -0.7], dtype=np.float64)
        first = oracle_candidate_xyz(nominal, config)
        second = oracle_candidate_xyz(nominal, config)
        self.assertGreaterEqual(len(first), 60)
        self.assertEqual(first[0]["source"], "nominal")
        self.assertTrue(np.array_equal(first[0]["xyz"], nominal))
        first_values = [tuple(item["xyz"].tolist()) for item in first]
        second_values = [tuple(item["xyz"].tolist()) for item in second]
        self.assertEqual(first_values, second_values)
        self.assertEqual(len(first_values), len(set(first_values)))
        self.assertLessEqual(max(abs(value) for row in first_values for value in row), 1.0)

    @unittest.skipUnless(HAS_NUMPY, "NumPy is allocation dependency")
    def test_nominal_anchored_affine_recovers_exact_action_map(self):
        from main.multilink_ellipsoid.oracle_affine import (
            fit_nominal_anchored_affine,
        )

        nominal = np.asarray([0.1, -0.2, 0.3], dtype=np.float64)
        gradients = np.asarray(
            [
                [0.2, -0.1, 0.05],
                [0.0, 0.3, -0.2],
                [0.1, 0.1, 0.1],
                [-0.2, 0.0, 0.2],
                [0.05, 0.0, 0.0],
                [0.0, 0.05, 0.0],
                [0.0, 0.0, 0.05],
                [0.01, 0.02, 0.03],
            ],
            dtype=np.float64,
        )
        base = np.linspace(0.01, 0.08, 8)
        xyz_values = [
            nominal,
            nominal + [0.1, 0.0, 0.0],
            nominal - [0.1, 0.0, 0.0],
            nominal + [0.0, 0.1, 0.0],
            nominal - [0.0, 0.1, 0.0],
            nominal + [0.0, 0.0, 0.1],
            nominal - [0.0, 0.0, 0.1],
            nominal + [0.1, -0.1, 0.1],
        ]
        records = []
        for xyz in xyz_values:
            xyz = np.asarray(xyz, dtype=np.float64)
            records.append(
                {
                    "candidate_xyz": xyz.tolist(),
                    "minimum_substep_clearance_m": (
                        base + gradients @ (xyz - nominal)
                    ).tolist(),
                }
            )
        result = fit_nominal_anchored_affine(
            records,
            nominal,
            trust_region_linf=0.5,
            one_sided_padding_m=1.0e-6,
            clearance_target_m=0.0,
        )
        self.assertTrue(
            np.allclose(result["gradients_m_per_action"], gradients, atol=1.0e-12)
        )
        self.assertEqual(result["candidate_false_safe_count"], 0)
        self.assertTrue(
            np.allclose(
                result["one_sided_overprediction_bound_m"],
                np.full(8, 1.0e-6),
                atol=1.0e-12,
            )
        )

    def test_allocation_contract_requires_h100_and_clean_source(self):
        source = (ROOT / "slurm/distal_oracle_affine_e05.sbatch").read_text()
        self.assertIn("--gres=gpu:1", source)
        self.assertIn("H100", source)
        self.assertIn("status --porcelain=v1 --untracked-files=all", source)
        self.assertIn("tests.test_distal_oracle_affine", source)
        self.assertIn("evaluate_distal_oracle_affine_e05.py", source)
        self.assertIn("validate_distal_oracle_affine_e05.py", source)
        self.assertIn("FALSE_SAFE_RESULT", source)
        self.assertIn("d79a28585e74cece", source)

    def test_substep_hook_and_stop_reason_are_explicit(self):
        source = (
            ROOT / "main/multilink_ellipsoid/oracle_affine.py"
        ).read_text()
        self.assertIn("base._update_observables = traced_update", source)
        self.assertIn(
            "geometry_authority_failed_before_transition_learning", source
        )
        config = json.loads(
            (ROOT / "configs/vlsa_distal_oracle_affine_e05.v1.json").read_text()
        )
        self.assertIn("not_training", config["claim_scope"])


if __name__ == "__main__":
    unittest.main()
