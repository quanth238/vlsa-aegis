from __future__ import annotations

from pathlib import Path
import unittest


class RenderDistalAffineOracleClosedLoopVideoTests(unittest.TestCase):
    def test_replay_uses_one_environment_and_hash_receipts(self) -> None:
        root = Path(__file__).resolve().parents[1]
        source = (
            root / "scripts" / "render_distal_affine_oracle_closed_loop_video.py"
        ).read_text(encoding="utf-8")
        self.assertEqual(source.count("_build_environment("), 1)
        self.assertNotIn("probe_env", source)
        self.assertIn("_dynamic_state_vector", source)
        self.assertIn("next_state_sha256", source)
        self.assertIn("source frame has striped pixel corruption", source)
        self.assertIn("decoded video fidelity differs", source)
        self.assertIn("EXACT_CANDIDATE_RESULT_SCHEMA", source)
        self.assertIn("CONTINUE_RESULT_SCHEMA", source)
        self.assertIn("WAYPOINT_RESULT_SCHEMA", source)
        self.assertIn("video_fps", source)


if __name__ == "__main__":
    unittest.main()
