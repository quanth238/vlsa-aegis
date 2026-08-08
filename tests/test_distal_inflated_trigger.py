from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class DistalInflatedTriggerTests(unittest.TestCase):
    def test_config_freezes_rounded_shell_and_first_crossing(self):
        from main.multilink_ellipsoid.inflated_trigger import (
            INFLATED_TRIGGER_SCHEMA,
            load_inflated_trigger_config,
        )

        config = load_inflated_trigger_config(
            ROOT / "configs/vlsa_distal_exact_box_inflated_trigger_e05.v1.json"
        )
        self.assertEqual(config["schema_version"], INFLATED_TRIGGER_SCHEMA)
        self.assertEqual(config["rounded_box_shell"]["shell_m"], 0.008)
        self.assertFalse(config["rounded_box_shell"]["half_axis_inflation"])
        self.assertEqual(
            config["trigger_scan"]["selection"],
            "first_step_satisfying_current_safe_and_nominal_crossing",
        )
        self.assertEqual(config["prior_results"]["fixed_margin"]["slurm_job_id"], "37175")
        self.assertEqual(config["prior_results"]["exact_box"]["slurm_job_id"], "37180")

    def test_first_crossing_requires_safe_start_and_unsafe_nominal(self):
        from main.multilink_ellipsoid.inflated_trigger import (
            is_first_crossing_candidate,
        )

        safe = [0.02] * 8
        crossing = [0.02] * 7 + [0.007]
        self.assertTrue(is_first_crossing_candidate(safe, crossing, 0.008))
        self.assertFalse(is_first_crossing_candidate(safe, [0.008] * 8, 0.008))
        already_unsafe = [0.02] * 7 + [0.007]
        self.assertFalse(is_first_crossing_candidate(already_unsafe, crossing, 0.008))

    def test_intervention_pass_uses_only_registered_mechanism_gates(self):
        from main.multilink_ellipsoid.inflated_trigger import intervention_pass

        decision = {
            "local_jointly_raw_and_proxy_safe_candidate_exists": True,
            "affine_candidate_gate_pass": True,
            "qp_valid": True,
            "qp_exact_raw_safe": True,
            "qp_exact_proxy_safe": True,
            "geometry_contact_witness_pass": False,
        }
        self.assertTrue(intervention_pass({"decision": decision}))
        decision["qp_exact_raw_safe"] = False
        self.assertFalse(intervention_pass({"decision": decision}))

    def test_h100_allocation_and_validator_are_explicit(self):
        allocation = (
            ROOT / "slurm/distal_exact_box_inflated_trigger_e05.sbatch"
        ).read_text()
        self.assertIn("--gres=gpu:1", allocation)
        self.assertIn("H100", allocation)
        self.assertIn("status --porcelain=v1 --untracked-files=all", allocation)
        self.assertIn("tests.test_distal_inflated_trigger", allocation)
        self.assertIn("validate_distal_exact_box_inflated_trigger_e05.py", allocation)
        self.assertIn("margin8mm-20260809a", allocation)
        self.assertIn("exact-box-20260809a", allocation)


if __name__ == "__main__":
    unittest.main()
