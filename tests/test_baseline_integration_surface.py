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
        self.assertIn("self.sample_actions_eager = self.sample_actions", source)

    def test_openpi_sampler_exposes_distinct_exact_latent_resume(self) -> None:
        path = ROOT / "openpi/src/openpi/models_pytorch/pi0_pytorch.py"
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        self.assertIn('"latent_resume_edit"', source)
        self.assertIn("crfs_resume_latent=None", source)
        self.assertIn("crfs_resume_time=None", source)
        self.assertIn("crfs_latent_edit=None", source)
        self.assertIn("crfs_return_normalized_final=False", source)
        self.assertIn("x_t = crfs_resume_latent.detach().clone()", source)
        self.assertIn("time = crfs_resume_time.detach().clone().reshape(())", source)
        self.assertIn("x_t = torch.where(", source)
        self.assertIn("x_t + crfs_latent_edit", source)
        self.assertIn("CRFS latent_resume_edit requires explicit paired noise", source)
        self.assertIn("crfs_return_trace is not True", source)
        self.assertIn("crfs_return_normalized_final is not True", source)
        self.assertIn("torch.isfinite(value)", source)
        self.assertIn("tuple(crfs_resume_time.shape) != ()", source)
        self.assertNotIn("expected_resume_time", source)
        self.assertIn('"x_t_pre_edit"', source)
        self.assertIn('"x_t_post_edit"', source)
        self.assertIn('"v_post_edit"', source)
        self.assertIn('"predicted_clean_post_edit"', source)
        self.assertIn('crfs_trace["final_normalized"]', source)

        sample_actions = next(
            node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "sample_actions"
        )
        trace_schemas = []
        for node in ast.walk(sample_actions):
            if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Dict):
                continue
            if not any(isinstance(target, ast.Name) and target.id == "crfs_trace" for target in node.targets):
                continue
            trace_schemas.append({key.value for key in node.value.keys if isinstance(key, ast.Constant)})
        self.assertCountEqual(
            trace_schemas,
            [
                {
                    "step_index",
                    "time",
                    "x_t_pre_edit",
                    "latent_edit",
                    "x_t_post_edit",
                    "v_post_edit",
                    "predicted_clean_post_edit",
                },
                {"step_index", "time", "x_t", "v_base", "predicted_clean"},
            ],
        )

    def test_policy_reserves_websocket_control_envelope(self) -> None:
        path = ROOT / "openpi/src/openpi/policies/policy.py"
        source = path.read_text(encoding="utf-8")
        ast.parse(source)
        self.assertIn('inputs.pop("__crfs__", None)', source)
        self.assertIn("_physical_delta_to_model", source)
        self.assertIn("correction_space", source)
        self.assertIn("self._sample_actions_crfs if use_crfs_sampler", source)
        self.assertIn('sample_kwargs["crfs_return_trace"]', source)
        self.assertIn('sample_kwargs["crfs_intervention_mode"] != "none"', source)
        self.assertIn('trace["predicted_clean_physical"]', source)
        self.assertIn('np.array(trace["predicted_clean"], copy=True)', source)
        self.assertIn('("resume_latent", "crfs_resume_latent", action_shape)', source)
        self.assertIn('("resume_time", "crfs_resume_time", ())', source)
        self.assertIn('("latent_edit", "crfs_latent_edit", action_shape)', source)
        self.assertIn('crfs_controls.get("return_normalized_final", False)', source)
        self.assertIn('crfs_controls.get("latent_edit_space") != "model"', source)
        self.assertIn("noise_array.shape != action_shape", source)
        self.assertIn("np.isfinite(noise_array)", source)
        self.assertIn("np.isfinite(value)", source)
        self.assertIn('trace["predicted_clean_post_edit_physical"]', source)
        self.assertIn('trace["final_normalized_physical"]', source)
        self.assertIn('np.array(trace["final_normalized"], copy=True)', source)

    def test_default_policy_route_does_not_require_crfs_envelope(self) -> None:
        path = ROOT / "openpi/src/openpi/policies/policy.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        assignments = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "use_crfs_sampler" for target in node.targets)
        ]
        self.assertTrue(assignments)
        self.assertIsInstance(assignments[0].value, ast.Constant)
        self.assertIs(assignments[0].value.value, False)

    def test_checkpoint_conversion_preserves_nested_norm_assets(self) -> None:
        path = ROOT / "openpi/examples/convert_jax_model_to_pytorch.py"
        source = path.read_text(encoding="utf-8")
        ast.parse(source)
        self.assertIn('pathlib.Path(checkpoint_dir) / "assets"', source)
        self.assertIn("raise FileNotFoundError", source)


if __name__ == "__main__":
    unittest.main()
