from __future__ import annotations

import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class BaselineIntegrationSurfaceTest(unittest.TestCase):
    def test_openpi_sampler_keeps_default_and_exposes_oracle_controls(self) -> None:
        path = ROOT / "openpi/src/openpi/models_pytorch/pi0_pytorch.py"
        source = path.read_text(encoding="utf-8")
        ast.parse(source)
        self.assertIn("crfs_intervention_mode=\"none\"", source)
        self.assertIn("v_t = v_t - crfs_correction / crfs_residual_horizon", source)
        self.assertIn("predicted_clean", source)

    def test_policy_reserves_websocket_control_envelope(self) -> None:
        path = ROOT / "openpi/src/openpi/policies/policy.py"
        source = path.read_text(encoding="utf-8")
        ast.parse(source)
        self.assertIn('inputs.pop("__crfs__", None)', source)
        self.assertIn("_physical_delta_to_model", source)
        self.assertIn("correction_space", source)

    def test_checkpoint_conversion_preserves_nested_norm_assets(self) -> None:
        path = ROOT / "openpi/examples/convert_jax_model_to_pytorch.py"
        source = path.read_text(encoding="utf-8")
        ast.parse(source)
        self.assertIn('pathlib.Path(checkpoint_dir) / "assets"', source)
        self.assertIn("raise FileNotFoundError", source)


if __name__ == "__main__":
    unittest.main()
