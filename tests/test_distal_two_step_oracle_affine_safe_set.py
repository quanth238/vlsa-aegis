from __future__ import annotations

import itertools
import importlib.util
from pathlib import Path
import unittest

from main.multilink_ellipsoid.oracle_affine_safe_set import (
    ORACLE_AFFINE_SAFE_SET_CONFIG_SCHEMA,
    fit_candidate_conditioned_affine_certificate,
    load_oracle_affine_safe_set_config,
    solve_affine_certificate_qp,
)


SCIPY_OSQP_AVAILABLE = bool(
    importlib.util.find_spec("scipy") is not None
    and importlib.util.find_spec("osqp") is not None
)


class DistalTwoStepOracleAffineSafeSetTests(unittest.TestCase):
    def test_checked_in_config_loads(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = load_oracle_affine_safe_set_config(
            root
            / "configs"
            / "vlsa_distal_two_step_oracle_affine_safe_set_moka_test.v1.json"
        )
        self.assertEqual(config["schema_version"], ORACLE_AFFINE_SAFE_SET_CONFIG_SCHEMA)
        self.assertEqual(len(config["test_case_ids"]), 3)

    @unittest.skipUnless(SCIPY_OSQP_AVAILABLE, "SciPy/OSQP allocation dependency")
    def test_affine_certificate_and_qp_recover_safe_half_space(self) -> None:
        import numpy as np

        xyz = np.asarray(list(itertools.product((-1.0, 0.0, 1.0), repeat=3)))
        scalar = 0.1 * xyz[:, 0] - 0.01
        margins = np.repeat(scalar[:, None], 7, axis=1)
        certificate = fit_candidate_conditioned_affine_certificate(
            xyz, margins, np.zeros(3), np.ones(len(xyz), dtype=bool),
            np.arange(len(xyz)), one_sided_padding_m=1.0e-6,
            target_clearance_m=0.0, postcheck_tolerance_m=1.0e-8,
        )
        self.assertTrue(certificate["valid"])
        self.assertEqual(certificate["sampled_grid_false_safe_candidate_count"], 0)
        result = solve_affine_certificate_qp(
            np.zeros(3), -np.ones(3), np.ones(3), certificate,
            {
                "eps_abs": 1.0e-7, "eps_rel": 1.0e-7, "max_iter": 10000,
                "residual_tolerance": 5.0e-7,
                "bound_tolerance_action": 5.0e-8,
            },
        )
        self.assertTrue(result["valid"])
        self.assertEqual(result["diagnostics"]["input_constraint_count"], 7)
        self.assertGreaterEqual(min(result["predicted_conservative_lower_bound_m"]), -5e-7)

    @unittest.skipUnless(SCIPY_OSQP_AVAILABLE, "SciPy/OSQP allocation dependency")
    def test_no_safe_grid_candidate_is_retained(self) -> None:
        import numpy as np

        xyz = np.asarray(list(itertools.product((-1.0, 1.0), repeat=3)))
        margins = np.full((len(xyz), 7), -0.001)
        certificate = fit_candidate_conditioned_affine_certificate(
            xyz, margins, np.zeros(3), np.ones(len(xyz), dtype=bool),
            np.arange(len(xyz)), one_sided_padding_m=1.0e-6,
            target_clearance_m=0.0, postcheck_tolerance_m=1.0e-8,
        )
        self.assertFalse(certificate["valid"])
        self.assertEqual(certificate["reason"], "no_exact_proxy_raw_safe_grid_candidate")


if __name__ == "__main__":
    unittest.main()
