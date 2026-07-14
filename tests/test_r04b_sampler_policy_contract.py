from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]
IMPORT_ERROR: Exception | None = None
try:
    import numpy as np
    import torch

    sys.path.insert(0, str(ROOT / "openpi/src"))
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
    from openpi.policies.policy import Policy
except ModuleNotFoundError as exc:  # pragma: no cover - local harness intentionally has no ML dependencies.
    IMPORT_ERROR = exc


class R04BSignedZeroStructuralTest(unittest.TestCase):
    def test_latent_resume_preserves_unedited_coordinate_bytes(self) -> None:
        source = (
            ROOT / "openpi/src/openpi/models_pytorch/pi0_pytorch.py"
        ).read_text(encoding="utf-8")
        resume_branch = source[source.index("if latent_resume_pending:") :]
        selection = resume_branch.index("x_t = torch.where(")
        self.assertIn("crfs_latent_edit == 0", resume_branch[selection : selection + 300])
        self.assertIn("x_t + crfs_latent_edit", resume_branch[selection : selection + 300])
        self.assertIn("-0.0 + +0.0", resume_branch[: selection + 300])


@unittest.skipIf(
    IMPORT_ERROR is not None,
    f"R04B runtime checks need the OpenPI dependency environment: {IMPORT_ERROR}",
)
class R04BSamplerPolicyContractTest(unittest.TestCase):
    ACTION_SHAPE = (10, 32)

    def _sampler(self):
        sampler = object.__new__(PI0Pytorch)
        object.__setattr__(
            sampler,
            "config",
            SimpleNamespace(action_horizon=self.ACTION_SHAPE[0], action_dim=self.ACTION_SHAPE[1]),
        )
        return sampler

    def _observation(self):
        return SimpleNamespace(state=torch.zeros((1, 32), dtype=torch.float32))

    def _sampler_kwargs(self) -> dict:
        shape = (1, *self.ACTION_SHAPE)
        return {
            "noise": torch.zeros(shape, dtype=torch.float32),
            "crfs_intervention_step": 5,
            "crfs_intervention_mode": "latent_resume_edit",
            "crfs_return_trace": True,
            "crfs_return_normalized_final": True,
            "crfs_resume_latent": torch.zeros(shape, dtype=torch.float32),
            "crfs_resume_time": torch.tensor(0.4999999, dtype=torch.float32),
            "crfs_latent_edit": torch.zeros(shape, dtype=torch.float32),
        }

    def test_sampler_rejects_incomplete_trace_contract_before_model_work(self) -> None:
        for name in ("crfs_return_trace", "crfs_return_normalized_final"):
            with self.subTest(name=name):
                kwargs = self._sampler_kwargs()
                kwargs[name] = False
                with self.assertRaisesRegex(ValueError, "requires"):
                    PI0Pytorch.sample_actions(self._sampler(), "cpu", self._observation(), **kwargs)

    def test_sampler_rejects_wrong_time_shape_range_and_nonfinite_values(self) -> None:
        invalid_times = (
            torch.tensor([0.5], dtype=torch.float32),
            torch.tensor(0.0, dtype=torch.float32),
            torch.tensor(float("nan"), dtype=torch.float32),
        )
        for resume_time in invalid_times:
            with self.subTest(shape=tuple(resume_time.shape), value=resume_time):
                kwargs = self._sampler_kwargs()
                kwargs["crfs_resume_time"] = resume_time
                with self.assertRaises(ValueError):
                    PI0Pytorch.sample_actions(self._sampler(), "cpu", self._observation(), **kwargs)

        for name in ("noise", "crfs_resume_latent", "crfs_latent_edit"):
            with self.subTest(name=name):
                kwargs = self._sampler_kwargs()
                kwargs[name] = kwargs[name].clone()
                kwargs[name][0, 0, 0] = float("inf")
                with self.assertRaisesRegex(ValueError, "nonfinite"):
                    PI0Pytorch.sample_actions(self._sampler(), "cpu", self._observation(), **kwargs)

    def test_sampler_rejects_resume_tensor_on_a_different_device(self) -> None:
        kwargs = self._sampler_kwargs()
        kwargs["crfs_latent_edit"] = torch.empty((1, *self.ACTION_SHAPE), dtype=torch.float32, device="meta")
        with self.assertRaisesRegex(ValueError, "does not match paired noise device"):
            PI0Pytorch.sample_actions(self._sampler(), "cpu", self._observation(), **kwargs)

    def _policy(self):
        policy = object.__new__(Policy)
        policy._model = SimpleNamespace(
            config=SimpleNamespace(action_horizon=self.ACTION_SHAPE[0], action_dim=self.ACTION_SHAPE[1])
        )
        policy._input_transform = lambda value: value
        policy._sample_kwargs = {}
        policy._is_pytorch_model = True
        policy._pytorch_device = "cpu"
        return policy

    def _policy_controls(self) -> dict:
        return {
            "noise": np.zeros(self.ACTION_SHAPE, dtype=np.float32),
            "intervention_step": 5,
            "intervention_mode": "latent_resume_edit",
            "resume_latent": np.zeros(self.ACTION_SHAPE, dtype=np.float32),
            "resume_time": np.asarray(0.4999999, dtype=np.float32),
            "latent_edit": np.zeros(self.ACTION_SHAPE, dtype=np.float32),
            "latent_edit_space": "model",
            "return_trace": True,
            "return_normalized_final": True,
        }

    def _infer_invalid(self, controls: dict, message: str) -> None:
        observation = {
            "state": np.zeros((32,), dtype=np.float32),
            "__crfs__": controls,
        }
        with self.assertRaisesRegex(ValueError, message):
            self._policy().infer(observation)

    def test_policy_rejects_ambiguous_or_incomplete_resume_controls(self) -> None:
        for name, value, message in (
            ("latent_edit_space", "physical", "latent_edit_space"),
            ("return_trace", False, "return_trace"),
            ("return_normalized_final", False, "return_normalized_final"),
            ("correction", np.zeros(self.ACTION_SHAPE, dtype=np.float32), "forbids correction"),
        ):
            with self.subTest(name=name):
                controls = self._policy_controls()
                controls[name] = value
                self._infer_invalid(controls, message)

        controls = self._policy_controls()
        del controls["resume_latent"]
        self._infer_invalid(controls, "missing required controls")

    def test_policy_rejects_bad_dtype_shape_and_nonfinite_inputs(self) -> None:
        controls = self._policy_controls()
        controls["noise"] = controls["noise"].astype(np.float64)
        self._infer_invalid(controls, "float32")

        controls = self._policy_controls()
        controls["resume_time"] = np.asarray([0.5], dtype=np.float32)
        self._infer_invalid(controls, "unbatched shape")

        controls = self._policy_controls()
        controls["latent_edit"] = controls["latent_edit"].copy()
        controls["latent_edit"][0, 0] = np.inf
        self._infer_invalid(controls, "nonfinite")


if __name__ == "__main__":
    unittest.main()
