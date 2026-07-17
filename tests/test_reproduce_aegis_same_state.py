"""Dependency-light checks for the clean same-state AEGIS reproducer."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "main" / "reproduce_aegis_same_state.py"
CONFIG = ROOT / "configs" / "aegis_same_state_canary.json"
LABELS = ROOT / "manifests" / "r06_codex_obstacle_labels_canary.jsonl"
UTILS = ROOT / "main" / "utils.py"
SBATCH = ROOT / "slurm" / "reproduce_aegis_same_state.sbatch"
WRAPPER = (
    ROOT / "safelibero" / "libero" / "libero" / "envs" / "env_wrapper.py"
)

SPEC = importlib.util.spec_from_file_location("same_state_repro", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class SameStateAegisReproducerTest(unittest.TestCase):
    def test_sphere_box_clearance_outside_touching_and_inside(self) -> None:
        rotation = np.eye(3)
        half = np.ones(3)
        self.assertAlmostEqual(
            MODULE.sphere_box_clearance(
                [2.5, 0.0, 0.0], 0.5, [0.0, 0.0, 0.0], rotation, half
            ),
            1.0,
        )
        self.assertAlmostEqual(
            MODULE.sphere_box_clearance(
                [1.5, 0.0, 0.0], 0.5, [0.0, 0.0, 0.0], rotation, half
            ),
            0.0,
        )
        self.assertAlmostEqual(
            MODULE.sphere_box_clearance(
                [0.0, 0.0, 0.0], 0.5, [0.0, 0.0, 0.0], rotation, half
            ),
            -1.5,
        )

    def test_config_is_exactly_one_clean_canary(self) -> None:
        value = json.loads(CONFIG.read_text(encoding="utf-8"))
        self.assertEqual(
            value["baseline_commit"],
            "57b1aef306f212aea3574b0a3b64aa1a3d8f5e4b",
        )
        self.assertEqual(value["case_id"], "crfs-1069f29a8d76463a")
        self.assertEqual(value["action_horizon"], 5)
        self.assertEqual(value["settle_steps"], 20)
        self.assertEqual(value["perception"]["device"], "cpu")
        self.assertEqual(
            value["aegis"]["implementation"],
            "released_main_aegis_full_action_space",
        )
        self.assertTrue(value["aegis"]["released_pre_settle_robot_geometry"])

    def test_frozen_codex_label_is_hash_bound(self) -> None:
        value = json.loads(CONFIG.read_text(encoding="utf-8"))
        digest = hashlib.sha256(LABELS.read_bytes()).hexdigest()
        self.assertEqual(
            digest, value["codex_label_manifest"]["sha256"]
        )
        rows = [
            json.loads(line)
            for line in LABELS.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["obstacle_label"], "red milk carton")
        self.assertIn(
            rows[0]["obstacle_label"], rows[0]["allowed_label_vocabulary"]
        )

    def test_original_point_cloud_path_keeps_cuda_default(self) -> None:
        text = UTILS.read_text(encoding="utf-8")
        self.assertIn('device="cuda"', text)
        self.assertIn("DEVICE = device", text)
        self.assertIn("device=DEVICE", text)

    def test_reproducer_uses_frozen_actions_and_distinct_metrics(self) -> None:
        text = SCRIPT.read_text(encoding="utf-8")
        for token in (
            'capture["policy"]["full_actions_values"]',
            '"released_pre_settle_state"',
            "compute_h_coeffs_3d",
            "cp.Variable(9)",
            "problem.solve(solver=cp.OSQP)",
            "PUBLIC_COLLISION_DISPLACEMENT_M",
            '"diagnostic_collision"',
            '"public_obstacle_displacement_collision"',
            '"same_nominal_actions"',
            '"same_agentview_depth"',
            '"same_backview_depth"',
            '"same_controller_state"',
            '"measurement_samples_126"',
            '"reach_progress_abs_error_le_1e-12"',
            '"realized_path_retention"',
            "conditioned_on_frozen_Codex_label",
            '"joint_registered_margin_safety_plus_progress"',
            '"population_release_authorized": False',
            "except AegisMethodFailure as method_error",
            "CPU-device-adapted GroundingDINO",
            '"robot_ellipsoid_center_p1_world_m"',
        ):
            self.assertIn(token, text)
        self.assertIn(
            'capture["pairing"]["reference"][\n        "controller_state"\n    ]',
            text,
        )
        self.assertIn('for obstacle_geom in raw["obstacle_geoms"]', text)
        self.assertNotIn('"joint_diagnostic_safety_plus_progress"', text)
        self.assertNotIn("WebsocketClientPolicy", text)
        self.assertNotIn("obstacle_detection(", text)

    def test_substep_measurement_is_additive(self) -> None:
        text = WRAPPER.read_text(encoding="utf-8")
        self.assertIn(
            "def step(self, action):\n        return self.env.step(action)",
            text,
        )
        self.assertIn("def step_with_substep_callback", text)
        self.assertIn("callback(self.env.sim, substep_index)", text)
        self.assertIn("expected 25 physics substeps", SCRIPT.read_text())

    def test_slurm_job_runs_cpu_perception_on_worker_one(self) -> None:
        text = SBATCH.read_text(encoding="utf-8")
        for token in (
            "#SBATCH --gres=gpu:1",
            "#SBATCH --cpus-per-task=8",
            "#SBATCH --mem=32G",
            "#SBATCH --nodelist=worker-1",
            "HF_HUB_OFFLINE=1",
            "TRANSFORMERS_OFFLINE=1",
            "MUJOCO_GL=egl",
            "/mnt/data/quanth/venvs/safety_vla/main/bin/python",
        ):
            self.assertIn(token, text)
        self.assertNotIn("serve_policy.py", text)


if __name__ == "__main__":
    unittest.main()
