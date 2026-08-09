from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

from main.multilink_ellipsoid.object_refined_suffix import (
    CONFIG_SCHEMA, load_config, refinement_chunks, select_candidate,
)


ROOT = Path(__file__).resolve().parents[1]
HAS_NUMPY = importlib.util.find_spec("numpy") is not None


class ObjectRefinedSuffixTests(unittest.TestCase):
    def test_checked_in_protocol_loads(self) -> None:
        config = load_config(
            ROOT / "configs/vlsa_distal_object_refined_suffix_e05.v1.json"
        )
        self.assertEqual(config["schema_version"], CONFIG_SCHEMA)
        self.assertEqual(config["primary_case"]["suffix_start_step"], 230)
        self.assertFalse(config["selector"]["QP_used"])
        self.assertFalse(config["selector"]["learned_model_used"])

    def test_selector_prioritizes_task_then_object_error(self) -> None:
        base = {
            "chunk": [[0.0] * 7], "chunk_correction_l2": 0.1,
            "minimum_all_eight_margin_m": [0.01] * 8, "raw_safe": True,
            "terminal_eef_reference_error_m": 0.01,
        }
        records = [
            dict(base, source="object", task_success_in_chunk=False,
                 terminal_object_reference_error_m=0.001),
            dict(base, source="task", task_success_in_chunk=True,
                 terminal_object_reference_error_m=0.1),
        ]
        selected = select_candidate(records, 1.0e-6)
        self.assertTrue(selected["valid"])
        self.assertEqual(selected["selected_source"], "task")

    @unittest.skipUnless(HAS_NUMPY, "NumPy is an allocation dependency")
    def test_refinement_changes_only_first_translation(self) -> None:
        import numpy as np

        anchor = np.zeros((4, 7), dtype=np.float64)
        anchor[:, 3:] = 0.5
        chunks = refinement_chunks([anchor], 0.25)
        self.assertEqual(len(chunks), 26)
        self.assertTrue(all(np.array_equal(chunk[1:], anchor[1:]) for chunk in chunks))
        self.assertTrue(all(np.array_equal(chunk[:, 3:], anchor[:, 3:]) for chunk in chunks))

    def test_harness_requires_exact_prefix_and_fresh_suffix_verification(self) -> None:
        source = (ROOT / "scripts/evaluate_distal_object_refined_suffix_e05.py").read_text()
        self.assertIn("prefix_state_hash_mismatch", source)
        self.assertIn("fresh_exact_object_refined_verification", source)
        self.assertIn("refinement_chunks", source)
        self.assertNotIn("solve_affine_certificate_qp", source)
        self.assertNotIn("WebsocketClientPolicy", source)


if __name__ == "__main__":
    unittest.main()
