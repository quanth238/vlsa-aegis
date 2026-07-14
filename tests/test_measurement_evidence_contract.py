from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class MeasurementEvidenceContractTest(unittest.TestCase):
    def test_gate_requires_fifty_unique_saved_states_and_visualization(self) -> None:
        source = (ROOT / "main/summarize_measurement_audits.py").read_text(encoding="utf-8")
        self.assertIn("unique_episodes >= 50", source)
        self.assertIn("len(cases) >= 50", source)
        self.assertIn("visualization_path.exists()", source)
        self.assertIn('>= 1e-4', source)

    def test_visualization_uses_recorded_minimum_pair_geometry(self) -> None:
        source = (ROOT / "main/render_measurement_audit.py").read_text(encoding="utf-8")
        self.assertIn('"conservative_obstacle_rotation_world"', source)
        self.assertIn('"conservative_eef_radius_m"', source)
        self.assertIn("conservative_min_substep_index", source)


if __name__ == "__main__":
    unittest.main()
