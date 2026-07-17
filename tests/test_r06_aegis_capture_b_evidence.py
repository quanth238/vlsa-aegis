from __future__ import annotations

from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence/r06/aegis-label-capture-canary-20260717b.json"
LABELS = ROOT / "manifests/r06_codex_obstacle_labels_canary.jsonl"
CONFIG = ROOT / "configs/experiments/r06_aegis_collision_conditioned.json"
DECISION = (
    ROOT
    / "docs/decisions/0069-accept-valid-aegis-capture-and-freeze-canary-label.md"
)
EXPECTED_LABEL_SHA256 = (
    "6a22b6d2f3705c008e338be6b217ce947fabfd4f56f70d3a8f442467694d996f"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_perception():
    path = ROOT / "main/crfs_oracle/aegis_perception.py"
    spec = importlib.util.spec_from_file_location(
        "r06_capture_b_aegis_perception", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load AEGIS perception adapter")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class R06AegisCaptureBEvidenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
        cls.config = json.loads(CONFIG.read_text(encoding="utf-8"))

    def test_exact_terminal_job_and_release_are_preserved(self) -> None:
        value = self.evidence
        self.assertEqual(value["status"], "validated_capture_apparatus")
        self.assertEqual(value["run_id"], "r06-aegis-label-capture-canary-20260717b")
        self.assertEqual(value["release"]["release_commit"], "ccb8c5225517f21ca1405f7b1470fe47dab160ee")
        self.assertEqual(value["slurm"]["exact_task_id"], "28428_0")
        self.assertEqual(value["slurm"]["state"], "COMPLETED")
        self.assertEqual(value["slurm"]["exit_code"], "0:0")
        self.assertEqual(value["slurm"]["node"], "worker-1")
        self.assertEqual(
            value["artifacts"]["capture_json_sha256"],
            "f2d32024ac6fac5aa5d133b5138df72501f1590b8cb0927b732edde69dabaaac",
        )
        self.assertEqual(
            value["paths"]["capture_json"],
            (
                "/mnt/data/quanth/experiments/crfs-oracle/"
                "r06-aegis-label-capture-canary-20260717b/"
                "crfs-1069f29a8d76463a/capture.json"
            ),
        )

    def test_pairing_collision_reproduction_and_zero_aegis_are_exact(self) -> None:
        science = self.evidence["scientific_validation"]
        self.assertEqual(science["independent_validator_errors"], [])
        self.assertEqual(science["selected_boundary_index"], 20)
        self.assertEqual(science["settle_ledger_boundaries"], 21)
        self.assertTrue(science["selected_branch_binding_passed"])
        self.assertEqual(
            science["live_controller"]["gripper_current_action_shape"], [2]
        )
        self.assertEqual(science["live_controller"]["robot_name"], "MountedPanda")
        self.assertTrue(all(science["policy_pairing"]["accepted_r02_checks"].values()))
        baseline = science["baseline_replay"]
        self.assertEqual(baseline["repeat_count"], 2)
        self.assertTrue(baseline["repeat_exact"])
        self.assertTrue(baseline["collision_reproduced_both"])
        self.assertEqual(baseline["measurement_samples_per_repeat"], [126, 126])
        self.assertEqual(
            baseline["raw_minimum_clearance_m"],
            [-0.0070070243639765994, -0.0070070243639765994],
        )
        counts = science["execution_counts"]
        self.assertFalse(counts["aegis_executed"])
        self.assertFalse(counts["groundingdino_executed"])
        self.assertFalse(counts["mvee_executed"])
        self.assertEqual(counts["qp_steps"], 0)
        self.assertEqual(counts["semantic_label_steps_inside_allocation"], 0)
        self.assertEqual(counts["training_steps"], 0)

    def test_codex_label_is_closed_schema_image_bound_and_chronological(self) -> None:
        self.assertEqual(_sha256(LABELS), EXPECTED_LABEL_SHA256)
        perception = _load_perception()
        ledger = perception.load_codex_semantic_label_jsonl(
            LABELS, expected_sha256=EXPECTED_LABEL_SHA256
        )
        record = ledger.for_case("crfs-1069f29a8d76463a")
        frozen = self.evidence["codex_label_freeze"]
        camera = self.evidence["scientific_validation"]["camera_and_assets"]
        artifacts = self.evidence["artifacts"]
        self.assertEqual(
            artifacts["agentview_image_npy_sha256"],
            "331ee97b4ee709c38406c6610054a28e839a80f1b29c23ea01e24124bc0f8fb3",
        )
        self.assertEqual(
            artifacts["agentview_image_png_sha256"],
            "f43a8c9816858fedd263358ac432f0fe490839441441125de44d028f427074c5",
        )
        self.assertEqual(
            camera["agentview_array_sha256"],
            "5000894309b66b8eec07155949e25a8f03d0363522280836e3e144213c55cfa8",
        )
        self.assertEqual(record.obstacle_label, "red milk carton")
        self.assertEqual(
            record.instruction,
            (
                "pick up the black bowl between the plate and the ramekin "
                "and place it on the plate"
            ),
        )
        self.assertEqual(
            record.agentview_image_sha256,
            frozen["agentview_image_array_sha256"],
        )
        self.assertEqual(
            record.agentview_image_sha256,
            camera["agentview_array_sha256"],
        )
        self.assertNotEqual(
            record.agentview_image_sha256,
            artifacts["agentview_image_npy_sha256"],
        )
        self.assertNotEqual(
            record.agentview_image_sha256,
            artifacts["agentview_image_png_sha256"],
        )
        self.assertEqual(record.reviewer, "codex")
        self.assertFalse(frozen["simulator_object_name_or_geometry_used"])
        self.assertGreater(
            datetime.strptime(record.reviewed_at, "%Y-%m-%dT%H:%M:%SZ"),
            datetime.strptime(
                self.evidence["scientific_validation"]["capture_completed_at"],
                "%Y-%m-%dT%H:%M:%SZ",
            ),
        )

    def test_capture_is_consumed_and_never_reauthorized(self) -> None:
        value = self.config
        if value["ready_to_run"]:
            release = value["execution_release"]
            self.assertEqual(release["stage"], "paired_codex_label_canary")
            self.assertTrue(release["aegis_execution_allowed"])
            self.assertFalse(release["original_glm_execution_allowed"])
            self.assertFalse(release["automatic_population_launch_authorized"])
        else:
            self.assertIsNone(value["execution_release"])
        self.assertEqual(
            value["codex_label_protocol"]["canary_label_manifest"]["sha256"],
            EXPECTED_LABEL_SHA256,
        )
        self.assertFalse(value["claims"]["probe_or_mlp_training_authorized"])
        self.assertIn(
            "does **not** establish that AEGIS prevents the collision",
            DECISION.read_text(encoding="utf-8"),
        )
        self.assertFalse(self.evidence["claims"]["aegis_safety_result_exists"])
        self.assertFalse(self.evidence["claims"]["paired_canary_result_exists"])
        self.assertFalse(self.evidence["claims"]["population_result_exists"])


if __name__ == "__main__":
    unittest.main()
